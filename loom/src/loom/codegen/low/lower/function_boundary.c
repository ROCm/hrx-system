// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/function_boundary.h"

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/lower/lower_context.h"
#include "loom/codegen/low/lower/source_plan.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/remap.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/registers.h"

static bool loom_low_lower_type_is_none(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_NONE;
}

static iree_status_t loom_low_lower_intern_descriptor_set_key(
    loom_low_lower_context_t* context,
    loom_string_id_t* out_descriptor_set_key) {
  iree_string_view_t descriptor_set_key = loom_low_descriptor_set_string(
      context->descriptor_set, context->descriptor_set->key_string_offset);
  IREE_ASSERT_FALSE(iree_string_view_is_empty(descriptor_set_key));
  return loom_module_intern_string(context->module, descriptor_set_key,
                                   out_descriptor_set_key);
}

static bool loom_low_lower_abi_argument_kind_is_known(
    loom_low_lower_abi_argument_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT:
    case LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE:
      return true;
    default:
      return false;
  }
}

static bool loom_low_lower_resource_import_kind_is_known(
    loom_low_resource_import_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER:
    case LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING:
    case LOOM_LOW_RESOURCE_IMPORT_KIND_COMMAND_INPUT:
      return true;
    default:
      return false;
  }
}

static bool loom_low_lower_function_attr_present(loom_func_like_t function,
                                                 uint8_t attr_index) {
  if (attr_index == LOOM_ATTR_INDEX_NONE) {
    return false;
  }
  return !loom_attr_is_absent(loom_op_attrs(function.op)[attr_index]);
}

static loom_target_abi_kind_t loom_low_lower_function_abi(
    const loom_low_lower_context_t* context) {
  const uint8_t abi_attr_index =
      context->source_function.vtable->abi_attr_index;
  if (loom_low_lower_function_attr_present(context->source_function,
                                           abi_attr_index)) {
    return (loom_target_abi_kind_t)loom_func_like_abi(context->source_function);
  }
  return loom_low_lower_context_bundle(context)->export_plan->abi_kind;
}

static iree_status_t loom_low_lower_map_direct_argument(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_source_type = loom_type_none(),
  };
  return loom_low_lower_map_value(context, source_op, source_argument_id,
                                  &out_argument->abi_type);
}

static iree_status_t loom_low_lower_map_argument(
    loom_low_lower_context_t* context, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  uint32_t previous_error_count = context->result->error_count;
  if (context->policy->map_argument.fn == NULL) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_map_direct_argument(context, context->source_function.op,
                                           source_argument_id, out_argument));
  } else {
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
        .abi_type = loom_type_none(),
        .resource_source_type = loom_type_none(),
    };
    IREE_RETURN_IF_ERROR(context->policy->map_argument.fn(
        context->policy->map_argument.user_data, context,
        context->source_function.op, source_argument_index, source_argument_id,
        out_argument));
  }

  IREE_ASSERT(loom_low_lower_abi_argument_kind_is_known(out_argument->kind));
  if (loom_low_lower_type_is_none(out_argument->abi_type)) {
    if (context->result->error_count == previous_error_count) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(IREE_SV("argument")),
          loom_param_u64(source_argument_id),
      };
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_target_context_error(
          context, context->source_function.op, LOOM_ERR_TARGET_027, params,
          IREE_ARRAYSIZE(params)));
    }
    return iree_ok_status();
  }
  IREE_ASSERT(loom_low_type_is_register(out_argument->abi_type));

  if (out_argument->kind == LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT) {
    return iree_ok_status();
  }
  IREE_ASSERT(loom_low_lower_resource_import_kind_is_known(
      out_argument->resource_import_kind));
  IREE_ASSERT_GE(out_argument->resource_index, 0);
  if (loom_low_lower_type_is_none(out_argument->resource_source_type)) {
    out_argument->resource_source_type =
        loom_module_value_type(context->module, source_argument_id);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_initialize_argument_map(
    loom_low_lower_context_t* context) {
  if (context->lowering.argument_map != NULL) {
    return iree_ok_status();
  }

  uint16_t argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(context->source_function, &argument_count);
  context->lowering.argument_map_count = argument_count;
  if (argument_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&context->function_arena, argument_count,
                                sizeof(*context->lowering.argument_map),
                                (void**)&context->lowering.argument_map));
  for (uint16_t i = 0; i < argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_map_argument(
        context, i, source_arguments[i], &context->lowering.argument_map[i]));
  }
  return iree_ok_status();
}

static uint16_t loom_low_lower_direct_argument_count(
    const loom_low_lower_context_t* context) {
  uint16_t direct_argument_count = 0;
  for (uint16_t i = 0; i < context->lowering.argument_map_count; ++i) {
    if (context->lowering.argument_map[i].kind ==
        LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT) {
      ++direct_argument_count;
    }
  }
  return direct_argument_count;
}

static bool loom_low_lower_first_return_operands(
    loom_region_t* source_body, const loom_op_t** out_return_op,
    loom_value_slice_t* out_operands) {
  *out_return_op = NULL;
  *out_operands = (loom_value_slice_t){0};
  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(source_body, block_index);
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_func_return_isa(op)) {
        continue;
      }
      *out_return_op = op;
      *out_operands = loom_func_return_operands(op);
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_lower_check_function_result(
    loom_low_lower_context_t* context, const loom_op_t* return_op,
    loom_value_slice_t returned_values, uint16_t result_index,
    loom_value_id_t result_id) {
  if (result_index < returned_values.count) {
    loom_type_t low_type = loom_type_none();
    return loom_low_lower_source_plan_check_mapped_value(
        context, return_op, returned_values.values[result_index], &low_type);
  }

  loom_type_t low_type = loom_type_none();
  return loom_low_lower_source_plan_check_mapped_value(
      context, context->source_function.op, result_id, &low_type);
}

iree_status_t loom_low_lower_function_boundary_validate(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  IREE_RETURN_IF_ERROR(loom_low_lower_initialize_argument_map(context));

  const loom_op_t* return_op = NULL;
  loom_value_slice_t returned_values = {0};
  (void)loom_low_lower_first_return_operands(source_body, &return_op,
                                             &returned_values);

  const loom_value_id_t* result_ids =
      loom_op_const_results(context->source_function.op);
  for (uint16_t i = 0; i < context->source_function.op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_check_function_result(
        context, return_op, returned_values, i, result_ids[i]));
  }

  if (context->source_function.op->tied_result_count != 0) {
    const loom_diagnostic_param_t params[] = {
        loom_param_u32(context->source_function.op->tied_result_count),
    };
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_target_context_error(
        context, context->source_function.op, LOOM_ERR_TARGET_029, params,
        IREE_ARRAYSIZE(params)));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_map_signature_types(
    loom_low_lower_context_t* context, loom_region_t* source_body,
    loom_type_t** out_arg_types, iree_host_size_t* out_arg_count,
    loom_type_t** out_result_types, iree_host_size_t* out_result_count) {
  IREE_RETURN_IF_ERROR(loom_low_lower_initialize_argument_map(context));
  *out_arg_types = NULL;
  *out_arg_count = 0;
  *out_result_types = NULL;
  *out_result_count = 0;

  uint16_t argument_count = 0;
  (void)loom_func_like_arg_ids(context->source_function, &argument_count);
  loom_type_t* arg_types = NULL;
  const uint16_t direct_argument_count =
      loom_low_lower_direct_argument_count(context);
  if (direct_argument_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, direct_argument_count, sizeof(*arg_types),
        (void**)&arg_types));
    uint16_t direct_argument_index = 0;
    for (uint16_t i = 0; i < argument_count; ++i) {
      if (context->lowering.argument_map[i].kind !=
          LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT) {
        continue;
      }
      arg_types[direct_argument_index] =
          context->lowering.argument_map[i].abi_type;
      IREE_ASSERT_FALSE(
          loom_low_lower_type_is_none(arg_types[direct_argument_index]));
      ++direct_argument_index;
    }
  }

  const uint16_t result_count = context->source_function.op->result_count;
  loom_type_t* result_types = NULL;
  if (result_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, result_count, sizeof(*result_types), (void**)&result_types));
    const loom_value_id_t* result_ids =
        loom_op_const_results(context->source_function.op);
    const loom_op_t* return_op = NULL;
    loom_value_slice_t returned_values = {0};
    (void)loom_low_lower_first_return_operands(source_body, &return_op,
                                               &returned_values);
    for (uint16_t i = 0; i < result_count; ++i) {
      if (i < returned_values.count) {
        IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
            context, return_op, returned_values.values[i], &result_types[i]));
      } else {
        IREE_RETURN_IF_ERROR(
            loom_low_lower_map_value(context, context->source_function.op,
                                     result_ids[i], &result_types[i]));
      }
      IREE_ASSERT_FALSE(loom_low_lower_type_is_none(result_types[i]));
    }
  }

  *out_arg_types = arg_types;
  *out_arg_count = direct_argument_count;
  *out_result_types = result_types;
  *out_result_count = result_count;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_map_abi_layout(
    loom_low_lower_context_t* context,
    loom_low_lower_abi_layout_kind_t layout_kind, const loom_type_t* arg_types,
    iree_host_size_t arg_count, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_named_attr_slice_t* out_abi_layout) {
  *out_abi_layout = loom_named_attr_slice_empty();
  if (context->policy->map_abi_layout.fn == NULL) {
    return iree_ok_status();
  }
  return context->policy->map_abi_layout.fn(
      context->policy->map_abi_layout.user_data, context, layout_kind,
      arg_types, arg_count, result_types, result_count, out_abi_layout);
}

static loom_region_t* loom_low_lower_low_body(
    const loom_low_lower_context_t* context) {
  if (loom_low_func_def_isa(context->low_func_op)) {
    return loom_low_func_def_body(context->low_func_op);
  }
  if (loom_low_kernel_def_isa(context->low_func_op)) {
    return loom_low_kernel_def_body(context->low_func_op);
  }
  return NULL;
}

static bool loom_low_lower_source_is_kernel_def(
    const loom_low_lower_context_t* context) {
  return loom_kernel_def_isa(context->source_function.op);
}

static iree_status_t loom_low_lower_attach_remapped_predicates(
    loom_low_lower_context_t* context, loom_ir_remap_t* remap) {
  uint16_t predicate_count = 0;
  const loom_predicate_t* source_predicates =
      loom_func_like_predicates(context->source_function, &predicate_count);
  if (predicate_count == 0) return iree_ok_status();

  const loom_func_like_t low_function =
      loom_func_like_cast(context->module, context->low_func_op);
  IREE_ASSERT(loom_func_like_isa(low_function));
  IREE_ASSERT_NE(low_function.vtable->predicates_attr_index,
                 LOOM_ATTR_INDEX_NONE);
  loom_predicate_t* low_predicates = NULL;
  IREE_RETURN_IF_ERROR(loom_ir_remap_predicate_list(
      remap, source_predicates, predicate_count, &low_predicates));
  loom_op_attrs(low_function.op)[low_function.vtable->predicates_attr_index] =
      loom_attr_predicate_list(low_predicates, predicate_count);
  return loom_module_note_op_attribute_value_refs(context->module,
                                                  low_function.op);
}

static iree_status_t loom_low_lower_create_func_op(
    loom_low_lower_context_t* context, loom_region_t* source_body,
    loom_symbol_ref_t low_func_ref, const loom_type_t* arg_types,
    iree_host_size_t arg_count, const loom_type_t* result_types,
    iree_host_size_t result_count) {
  loom_low_func_def_build_flags_t build_flags = 0;
  uint8_t visibility = loom_func_like_visibility(context->source_function);
  uint8_t cc = loom_func_like_cc(context->source_function);
  uint8_t purity = loom_func_like_purity(context->source_function);
  loom_target_abi_kind_t abi = loom_low_lower_function_abi(context);
  loom_named_attr_slice_t abi_attrs =
      loom_func_like_abi_attrs(context->source_function);
  loom_named_attr_slice_t abi_layout = loom_named_attr_slice_empty();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_abi_layout(
      context, LOOM_LOW_LOWER_ABI_LAYOUT_KIND_FUNC, arg_types, arg_count,
      result_types, result_count, &abi_layout));
  loom_string_id_t export_symbol =
      loom_func_like_export_symbol(context->source_function);
  loom_named_attr_slice_t export_attrs =
      loom_func_like_export_attrs(context->source_function);
  if (visibility != 0) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY;
  }
  if (cc != 0) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_CC;
  }
  if (purity != 0) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_PURITY;
  }
  if (abi != 0) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_ABI;
  }
  if (export_symbol != LOOM_STRING_ID_INVALID) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_EXPORT_SYMBOL;
  }
  if (loom_symbol_ref_is_valid(context->options->target_ref)) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_TARGET;
  }
  if (loom_low_lower_function_attr_present(
          context->source_function,
          context->source_function.vtable->abi_attrs_attr_index)) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_ABI_ATTRS;
  }
  if (abi_layout.count > 0) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_ABI_LAYOUT;
  }
  if (loom_low_lower_function_attr_present(
          context->source_function,
          context->source_function.vtable->export_attrs_attr_index)) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_EXPORT_ATTRS;
  }
  uint8_t retain = 0;
  if (loom_low_lower_context_source_is_retained(context)) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_RETAIN;
    retain = LOOM_LOW_RETAIN_RETAIN;
  }
  loom_builder_initialize(context->module, &context->module->arena,
                          loom_module_block(context->module),
                          &context->builder);
  loom_builder_set_before(&context->builder, context->source_function.op);
  loom_string_id_t descriptor_set_key = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_intern_descriptor_set_key(context, &descriptor_set_key));
  IREE_RETURN_IF_ERROR(loom_low_func_def_build(
      &context->builder, build_flags, visibility, retain, cc, purity,
      /*allocation=*/0, /*schedule=*/0, descriptor_set_key,
      context->options->target_ref, abi, abi_attrs, abi_layout, export_symbol,
      export_attrs, low_func_ref, arg_types, arg_count, result_types,
      result_count,
      /*tied_results=*/NULL, /*tied_result_count=*/0,
      /*predicates=*/NULL, /*predicates_count=*/0,
      context->source_function.op->location, &context->low_func_op));

  loom_region_t* low_body = loom_low_lower_low_body(context);
  low_body->flags = source_body->flags;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_create_kernel_op(
    loom_low_lower_context_t* context, loom_region_t* source_body,
    loom_symbol_ref_t low_func_ref, const loom_type_t* arg_types,
    iree_host_size_t arg_count) {
  loom_low_kernel_def_build_flags_t build_flags = 0;
  loom_string_id_t export_symbol =
      loom_func_like_export_symbol(context->source_function);
  if (export_symbol != LOOM_STRING_ID_INVALID) {
    build_flags |= LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_EXPORT_SYMBOL;
  }

  uint8_t export_linkage = 0;
  if (loom_func_like_export_linkage(context->source_function,
                                    &export_linkage)) {
    build_flags |= LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_EXPORT_LINKAGE;
  }
  if (loom_symbol_ref_is_valid(context->options->target_ref)) {
    build_flags |= LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_TARGET;
  }
  loom_target_workgroup_size_t workgroup_size = {0};
  if (iree_any_bit_set(context->result->static_launch_config_flags,
                       LOOM_LOW_LOWER_STATIC_LAUNCH_CONFIG_WORKGROUP_SIZE)) {
    build_flags |= LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_SIZE_X |
                   LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_SIZE_Y |
                   LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_SIZE_Z;
    workgroup_size = context->result->static_workgroup_size;
  }
  loom_target_dispatch_workgroup_count_t workgroup_count = {0};
  if (iree_any_bit_set(context->result->static_launch_config_flags,
                       LOOM_LOW_LOWER_STATIC_LAUNCH_CONFIG_WORKGROUP_COUNT)) {
    build_flags |= LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_COUNT_X |
                   LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_COUNT_Y |
                   LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_COUNT_Z;
    workgroup_count = context->result->static_workgroup_count;
  }
  loom_target_workgroup_cluster_size_t workgroup_cluster_size = {0};
  if (iree_any_bit_set(
          context->result->static_launch_config_flags,
          LOOM_LOW_LOWER_STATIC_LAUNCH_CONFIG_WORKGROUP_CLUSTER_SIZE)) {
    build_flags |= LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_CLUSTER_SIZE_X |
                   LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_CLUSTER_SIZE_Y |
                   LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_CLUSTER_SIZE_Z;
    workgroup_cluster_size = context->result->static_workgroup_cluster_size;
  }
  uint8_t retain = 0;
  if (loom_low_lower_context_source_is_retained(context)) {
    build_flags |= LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_RETAIN;
    retain = LOOM_LOW_RETAIN_RETAIN;
  }

  loom_builder_initialize(context->module, &context->module->arena,
                          loom_module_block(context->module),
                          &context->builder);
  loom_builder_set_before(&context->builder, context->source_function.op);
  loom_named_attr_slice_t abi_layout = loom_named_attr_slice_empty();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_abi_layout(
      context, LOOM_LOW_LOWER_ABI_LAYOUT_KIND_KERNEL, arg_types, arg_count,
      /*result_types=*/NULL, /*result_count=*/0, &abi_layout));
  if (abi_layout.count > 0) {
    build_flags |= LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_ABI_LAYOUT;
  }
  loom_string_id_t descriptor_set_key = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_intern_descriptor_set_key(context, &descriptor_set_key));
  IREE_RETURN_IF_ERROR(loom_low_kernel_def_build(
      &context->builder, build_flags, retain, /*allocation=*/0, /*schedule=*/0,
      descriptor_set_key, context->options->target_ref, abi_layout,
      export_symbol, export_linkage, workgroup_size.x, workgroup_size.y,
      workgroup_size.z, workgroup_count.x, workgroup_count.y, workgroup_count.z,
      workgroup_cluster_size.x, workgroup_cluster_size.y,
      workgroup_cluster_size.z, low_func_ref, arg_types, arg_count,
      /*predicates=*/NULL, /*predicates_count=*/0,
      context->source_function.op->location, &context->low_func_op));

  loom_region_t* low_body = loom_low_lower_low_body(context);
  low_body->flags = source_body->flags;
  return iree_ok_status();
}

iree_status_t loom_low_lower_function_boundary_remap_predicates(
    loom_low_lower_context_t* context) {
  uint16_t predicate_count = 0;
  const loom_predicate_t* source_predicates =
      loom_func_like_predicates(context->source_function, &predicate_count);
  if (predicate_count == 0) return iree_ok_status();

  loom_ir_remap_t remap;
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      context->module, context->module, &context->function_arena,
      /*options=*/NULL, &remap));
  for (uint16_t i = 0; i < predicate_count; ++i) {
    for (uint8_t j = 0; j < source_predicates[i].arg_count; ++j) {
      if (source_predicates[i].arg_tags[j] != LOOM_PRED_ARG_VALUE) continue;
      loom_value_id_t source_value =
          (loom_value_id_t)source_predicates[i].args[j];
      loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_low_lower_lookup_value(context, source_value, &low_value));
      IREE_RETURN_IF_ERROR(
          loom_ir_remap_map_value(&remap, source_value, low_value));
    }
  }

  return loom_low_lower_attach_remapped_predicates(context, &remap);
}

iree_status_t loom_low_lower_function_boundary_create(
    loom_low_lower_context_t* context, loom_region_t* source_body,
    loom_symbol_ref_t low_func_ref) {
  loom_type_t* arg_types = NULL;
  iree_host_size_t arg_count = 0;
  loom_type_t* result_types = NULL;
  iree_host_size_t result_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_lower_map_signature_types(
      context, source_body, &arg_types, &arg_count, &result_types,
      &result_count));

  if (loom_low_lower_source_is_kernel_def(context)) {
    IREE_ASSERT_EQ(result_count, 0);
    IREE_RETURN_IF_ERROR(loom_low_lower_create_kernel_op(
        context, source_body, low_func_ref, arg_types, arg_count));
  } else {
    IREE_RETURN_IF_ERROR(loom_low_lower_create_func_op(
        context, source_body, low_func_ref, arg_types, arg_count, result_types,
        result_count));
  }
  context->result->low_func_op = context->low_func_op;
  context->result->low_func_ref = low_func_ref;

  const loom_value_id_t* source_results =
      loom_op_const_results(context->source_function.op);
  const loom_value_id_t* low_results =
      loom_op_const_results(context->low_func_op);
  for (uint16_t i = 0; i < context->source_function.op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_copy_value_name(
        context, source_results[i], low_results[i]));
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_function_boundary_bind_entry_arguments(
    loom_low_lower_context_t* context, const loom_block_t* source_entry_block,
    loom_block_t* low_entry_block) {
  IREE_ASSERT_EQ(source_entry_block->arg_count,
                 context->lowering.argument_map_count);
  const uint16_t direct_argument_count =
      loom_low_lower_direct_argument_count(context);
  IREE_ASSERT_EQ(low_entry_block->arg_count, direct_argument_count);
  uint16_t direct_argument_index = 0;
  for (uint16_t argument_index = 0;
       argument_index < source_entry_block->arg_count; ++argument_index) {
    if (context->lowering.argument_map[argument_index].kind !=
        LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
        context, source_entry_block->arg_ids[argument_index],
        low_entry_block->arg_ids[direct_argument_index]));
    ++direct_argument_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_map_decl_signature_types(
    loom_low_lower_context_t* context, loom_type_t** out_arg_types,
    iree_host_size_t* out_arg_count, loom_type_t** out_result_types,
    iree_host_size_t* out_result_count) {
  *out_arg_types = NULL;
  *out_arg_count = 0;
  *out_result_types = NULL;
  *out_result_count = 0;

  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(context->source_function, &argument_count);
  loom_type_t* arg_types = NULL;
  if (argument_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, argument_count, sizeof(*arg_types), (void**)&arg_types));
    for (uint16_t i = 0; i < argument_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_low_lower_source_plan_check_mapped_value(
          context, context->source_function.op, argument_ids[i],
          &arg_types[i]));
    }
  }

  const uint16_t result_count = context->source_function.op->result_count;
  loom_type_t* result_types = NULL;
  if (result_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, result_count, sizeof(*result_types), (void**)&result_types));
    const loom_value_id_t* result_ids =
        loom_op_const_results(context->source_function.op);
    for (uint16_t i = 0; i < result_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_low_lower_source_plan_check_mapped_value(
          context, context->source_function.op, result_ids[i],
          &result_types[i]));
    }
  }

  *out_arg_types = arg_types;
  *out_arg_count = argument_count;
  *out_result_types = result_types;
  *out_result_count = result_count;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_copy_decl_signature_names(
    loom_low_lower_context_t* context) {
  uint16_t argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(context->source_function, &argument_count);
  const loom_value_id_t* low_arguments =
      loom_op_const_operands(context->low_func_op);
  for (uint16_t i = 0; i < argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_copy_value_name(
        context, source_arguments[i], low_arguments[i]));
  }

  const loom_value_id_t* source_results =
      loom_op_const_results(context->source_function.op);
  const loom_value_id_t* low_results =
      loom_op_const_results(context->low_func_op);
  for (uint16_t i = 0; i < context->source_function.op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_copy_value_name(
        context, source_results[i], low_results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_remap_decl_predicates(
    loom_low_lower_context_t* context) {
  uint16_t predicate_count = 0;
  (void)loom_func_like_predicates(context->source_function, &predicate_count);
  if (predicate_count == 0) return iree_ok_status();

  const loom_func_like_t low_function =
      loom_func_like_cast(context->module, context->low_func_op);
  uint16_t source_argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(context->source_function, &source_argument_count);
  uint16_t low_argument_count = 0;
  const loom_value_id_t* low_arguments =
      loom_func_like_arg_ids(low_function, &low_argument_count);
  IREE_ASSERT_EQ(source_argument_count, low_argument_count);
  IREE_ASSERT_EQ(context->source_function.op->result_count,
                 low_function.op->result_count);

  loom_ir_remap_t remap;
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      context->module, context->module, &context->function_arena,
      /*options=*/NULL, &remap));
  for (uint16_t i = 0; i < source_argument_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_map_value(&remap, source_arguments[i], low_arguments[i]));
  }
  const loom_value_id_t* source_results =
      loom_op_const_results(context->source_function.op);
  const loom_value_id_t* low_results = loom_op_const_results(low_function.op);
  for (uint16_t i = 0; i < context->source_function.op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_map_value(&remap, source_results[i], low_results[i]));
  }
  return loom_low_lower_attach_remapped_predicates(context, &remap);
}

iree_status_t loom_low_lower_import_declaration(
    loom_module_t* module, loom_func_like_t source_declaration,
    const loom_low_lower_options_t* options,
    loom_low_lower_result_t* out_result) {
  IREE_ASSERT(out_result != NULL);
  IREE_ASSERT(loom_func_like_isa(source_declaration));
  IREE_ASSERT(options != NULL);
  IREE_ASSERT(options->policy != NULL);
  IREE_ASSERT_NE(options->policy->import_decl_kind, 0);
  *out_result = (loom_low_lower_result_t){
      .low_func_ref = loom_symbol_ref_null(),
  };
  if (!iree_allocator_is_null(options->report_allocator)) {
    out_result->report_allocator = options->report_allocator;
    out_result->memory_report_row_allocator = module->allocator;
  }

  const loom_symbol_ref_t low_func_ref =
      loom_func_like_callee(source_declaration);
  IREE_ASSERT(loom_symbol_ref_is_valid(low_func_ref));
  IREE_ASSERT_EQ(low_func_ref.module_id, 0);
  IREE_ASSERT_LT(low_func_ref.symbol_id, module->symbols.count);

  const loom_low_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(
      loom_target_low_descriptor_set_select_for_source_lowering(
          options->descriptor_registry,
          loom_target_facts_bundle(options->target_facts), &descriptor_set));

  loom_low_lower_context_t context = {
      .module = module,
      .source_function = source_declaration,
      .options = options,
      .policy = options->policy,
      .descriptor_set = descriptor_set,
      .result = out_result,
  };
  out_result->descriptor_set = descriptor_set;
  iree_arena_initialize(module->arena.block_pool, &context.function_arena);
  loom_condition_query_initialize(module, &context.function_arena,
                                  &context.lowering.condition_query);
  iree_arena_initialize(module->arena.block_pool, &context.emission_arena);

  loom_type_t* arg_types = NULL;
  iree_host_size_t arg_count = 0;
  loom_type_t* result_types = NULL;
  iree_host_size_t result_count = 0;
  loom_low_lower_emission_scope_begin(&context);
  iree_status_t status = loom_low_lower_map_decl_signature_types(
      &context, &arg_types, &arg_count, &result_types, &result_count);
  if (iree_status_is_ok(status) && out_result->error_count == 0) {
    loom_string_id_t code_symbol =
        loom_func_like_import_symbol(source_declaration);
    if (code_symbol == LOOM_STRING_ID_INVALID) {
      code_symbol = module->symbols.entries[low_func_ref.symbol_id].name_id;
    }
    IREE_ASSERT_NE(code_symbol, LOOM_STRING_ID_INVALID);
    IREE_ASSERT_LT(code_symbol, module->strings.count);
    loom_low_func_decl_build_flags_t build_flags =
        LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_IMPORT_KIND |
        LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_CODE_SYMBOL;
    const uint8_t visibility = loom_func_like_visibility(source_declaration);
    const uint8_t cc = loom_func_like_cc(source_declaration);
    const uint8_t purity = loom_func_like_purity(source_declaration);
    const bool has_abi = loom_low_lower_function_attr_present(
        source_declaration, source_declaration.vtable->abi_attr_index);
    const loom_target_abi_kind_t abi =
        (loom_target_abi_kind_t)loom_func_like_abi(source_declaration);
    loom_named_attr_slice_t abi_attrs =
        loom_func_like_abi_attrs(source_declaration);
    loom_named_attr_slice_t abi_layout = loom_named_attr_slice_empty();
    status = loom_low_lower_map_abi_layout(
        &context, LOOM_LOW_LOWER_ABI_LAYOUT_KIND_FUNC, arg_types, arg_count,
        result_types, result_count, &abi_layout);
    loom_string_id_t export_symbol =
        loom_func_like_export_symbol(source_declaration);
    loom_named_attr_slice_t export_attrs =
        loom_func_like_export_attrs(source_declaration);
    if (visibility != 0) {
      build_flags |= LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_VISIBILITY;
    }
    if (cc != 0) {
      build_flags |= LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_CC;
    }
    if (purity != 0) {
      build_flags |= LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_PURITY;
    }
    if (has_abi) {
      build_flags |= LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_ABI;
    }
    if (export_symbol != LOOM_STRING_ID_INVALID) {
      build_flags |= LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_EXPORT_SYMBOL;
    }
    if (loom_symbol_ref_is_valid(options->target_ref)) {
      build_flags |= LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_TARGET;
    }
    if (loom_low_lower_function_attr_present(
            source_declaration,
            source_declaration.vtable->abi_attrs_attr_index)) {
      build_flags |= LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_ABI_ATTRS;
    }
    if (abi_layout.count > 0) {
      build_flags |= LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_ABI_LAYOUT;
    }
    if (loom_low_lower_function_attr_present(
            source_declaration,
            source_declaration.vtable->export_attrs_attr_index)) {
      build_flags |= LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_EXPORT_ATTRS;
    }
    uint8_t retain = 0;
    if (loom_low_lower_context_source_is_retained(&context)) {
      build_flags |= LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_RETAIN;
      retain = LOOM_LOW_RETAIN_RETAIN;
    }

    if (iree_status_is_ok(status)) {
      loom_builder_initialize(module, &module->arena, loom_module_block(module),
                              &context.builder);
      loom_builder_set_before(&context.builder, source_declaration.op);
      loom_string_id_t descriptor_set_key = LOOM_STRING_ID_INVALID;
      status = loom_low_lower_intern_descriptor_set_key(&context,
                                                        &descriptor_set_key);
      if (iree_status_is_ok(status)) {
        status = loom_low_func_decl_build(
            &context.builder, build_flags, visibility, retain, cc, purity,
            /*allocation=*/0, /*schedule=*/0,
            (uint8_t)options->policy->import_decl_kind, code_symbol,
            descriptor_set_key, options->target_ref, abi, abi_attrs, abi_layout,
            export_symbol, export_attrs, low_func_ref, arg_types, arg_count,
            result_types, result_count, /*tied_results=*/NULL,
            /*tied_result_count=*/0, /*predicates=*/NULL,
            /*predicates_count=*/0, source_declaration.op->location,
            &context.low_func_op);
      }
    }
  }
  loom_low_lower_emission_scope_end(&context);

  if (iree_status_is_ok(status) && out_result->error_count == 0) {
    status = loom_low_lower_remap_decl_predicates(&context);
  }
  if (iree_status_is_ok(status) && out_result->error_count == 0) {
    status = loom_low_lower_copy_decl_signature_names(&context);
  }
  if (iree_status_is_ok(status) && out_result->error_count == 0) {
    out_result->low_func_op = context.low_func_op;
    out_result->low_func_ref = low_func_ref;
    status = loom_op_erase(module, source_declaration.op);
  }
  if (iree_status_is_ok(status) && out_result->error_count == 0) {
    loom_module_link_symbol_defining_op(
        module, context.low_func_op,
        loom_op_vtable(module, context.low_func_op));
  }

  iree_arena_deinitialize(&context.emission_arena);
  iree_arena_deinitialize(&context.function_arena);
  return status;
}

static iree_status_t loom_low_lower_emit_argument_resource_import(
    loom_low_lower_context_t* context, const loom_value_id_t* source_arguments,
    uint16_t argument_index) {
  const loom_low_lower_abi_argument_t* argument =
      &context->lowering.argument_map[argument_index];
  loom_type_t source_type = argument->resource_source_type;
  if (loom_low_lower_type_is_none(source_type)) {
    source_type = loom_module_value_type(context->module,
                                         source_arguments[argument_index]);
  }
  loom_type_id_t source_type_id = LOOM_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_type_id(context->module, source_type,
                                                  &source_type_id));
  loom_op_t* resource_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_resource_build(
      &context->builder, argument->resource_build_flags,
      (uint8_t)argument->resource_import_kind, LOOM_VALUE_ID_INVALID,
      argument->resource_index, source_type_id, argument->resource_extent,
      argument->resource_cache_swizzle_stride, argument->abi_type,
      context->source_function.op->location, &resource_op));
  return loom_low_lower_bind_value(context, source_arguments[argument_index],
                                   loom_low_resource_result(resource_op));
}

iree_status_t loom_low_lower_function_boundary_emit_resource_imports(
    loom_low_lower_context_t* context) {
  uint16_t argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(context->source_function, &argument_count);
  if (argument_count == 0 ||
      argument_count == loom_low_lower_direct_argument_count(context)) {
    return iree_ok_status();
  }

  loom_region_t* low_body = loom_low_lower_low_body(context);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, context->low_func_op, low_body);
  loom_builder_set_block(&context->builder, loom_region_entry_block(low_body));
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < argument_count && iree_status_is_ok(status); ++i) {
    if (context->lowering.argument_map[i].kind !=
        LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE) {
      continue;
    }
    status = loom_low_lower_emit_argument_resource_import(context,
                                                          source_arguments, i);
  }

  loom_builder_restore(&context->builder, saved_ip);
  return status;
}
