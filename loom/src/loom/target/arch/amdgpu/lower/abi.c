// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/abi.h"

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/types.h"

static iree_status_t loom_amdgpu_make_hal_buffer_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  *out_type = loom_type_none();
  loom_string_id_t hal_buffer_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(loom_low_lower_context_module(context),
                                IREE_SV("hal.buffer"), &hal_buffer_id));
  *out_type = loom_type_dialect_opaque(hal_buffer_id);
  return iree_ok_status();
}

static uint32_t loom_amdgpu_hal_binding_index(loom_low_lower_context_t* context,
                                              uint16_t source_argument_index) {
  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids = loom_func_like_arg_ids(
      loom_low_lower_context_source_function(context), &argument_count);
  uint32_t resource_index = 0;
  for (uint16_t i = 0; i < source_argument_index && i < argument_count; ++i) {
    loom_type_t type = loom_module_value_type(
        loom_low_lower_context_module(context), argument_ids[i]);
    if (loom_type_is_buffer(type)) {
      ++resource_index;
    }
  }
  return resource_index;
}

typedef struct loom_amdgpu_buffer_argument_extent_t {
  // Module containing the source function being inspected.
  const loom_module_t* module;
  // Source function facts used to resolve SSA layout and view footprint facts.
  const loom_value_fact_table_t* fact_table;
  // True once a buffer.view derived from the source argument is found.
  bool found_view;
  // True when a derived view has no exact byte range.
  bool found_unbounded_view;
  // Maximum byte extent required by all statically boundable derived views.
  int64_t extent;
} loom_amdgpu_buffer_argument_extent_t;

static void loom_amdgpu_buffer_argument_extent_include_view(
    loom_amdgpu_buffer_argument_extent_t* state,
    const loom_op_t* buffer_view_op) {
  if (state->found_unbounded_view) {
    return;
  }
  state->found_view = true;
  int64_t base_byte_offset = 0;
  int64_t view_byte_extent = 0;
  int64_t extent = 0;
  loom_value_fact_view_reference_t view_reference = {0};
  loom_value_facts_t view_facts = loom_value_fact_table_lookup(
      state->fact_table, loom_buffer_view_result(buffer_view_op));
  if (!loom_value_facts_query_view_reference(&state->fact_table->context,
                                             view_facts, &view_reference) ||
      !loom_value_facts_as_exact_i64(view_reference.base_byte_offset,
                                     &base_byte_offset) ||
      base_byte_offset < 0 ||
      !loom_value_facts_as_exact_i64(view_reference.footprint_byte_length,
                                     &view_byte_extent) ||
      !iree_checked_add_i64(base_byte_offset, view_byte_extent, &extent)) {
    state->found_unbounded_view = true;
    return;
  }
  state->extent = iree_max(state->extent, extent);
}

static void loom_amdgpu_buffer_argument_extent_include_uses(
    loom_amdgpu_buffer_argument_extent_t* state, loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(state->module, value_id);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    const loom_op_t* user_op = loom_use_user_op(*use);
    const uint16_t operand_index = loom_use_operand_index(*use);
    if (operand_index == 0 && loom_buffer_view_isa(user_op)) {
      loom_amdgpu_buffer_argument_extent_include_view(state, user_op);
      continue;
    }
    if (operand_index < user_op->result_count &&
        loom_traits_are_fact_identity(
            loom_op_effective_traits(state->module, user_op))) {
      loom_amdgpu_buffer_argument_extent_include_uses(
          state, loom_op_results(user_op)[operand_index]);
      continue;
    }
    state->found_unbounded_view = true;
  }
}

static bool loom_amdgpu_source_buffer_argument_extent(
    loom_low_lower_context_t* context, loom_value_id_t source_argument_id,
    int64_t* out_extent) {
  *out_extent = 0;
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_amdgpu_buffer_argument_extent_t state = {
      .module = module,
      .fact_table = loom_low_lower_context_fact_table(context),
  };
  if (!state.fact_table) {
    return false;
  }
  loom_amdgpu_buffer_argument_extent_include_uses(&state, source_argument_id);
  if (!state.found_view || state.found_unbounded_view) {
    return false;
  }
  *out_extent = state.extent;
  return true;
}

iree_status_t loom_amdgpu_map_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  (void)user_data;
  loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  if (bundle->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL &&
      loom_type_is_buffer(source_type)) {
    loom_type_t binding_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_range_type(context, 2, &binding_type));
    loom_type_t source_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_hal_buffer_type(context, &source_type));
    loom_low_resource_build_flags_t resource_build_flags = 0;
    int64_t resource_extent = 0;
    if (loom_amdgpu_source_buffer_argument_extent(context, source_argument_id,
                                                  &resource_extent)) {
      resource_build_flags |= LOOM_LOW_RESOURCE_BUILD_FLAG_HAS_EXTENT;
    }
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE,
        .abi_type = binding_type,
        .resource_import_kind = LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING,
        .resource_index =
            loom_amdgpu_hal_binding_index(context, source_argument_index),
        .resource_source_type = source_type,
        .resource_build_flags = resource_build_flags,
        .resource_extent = resource_extent,
    };
    return iree_ok_status();
  }

  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_source_type = loom_type_none(),
  };
  if (bundle->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL &&
      loom_kernel_def_isa(source_function_op) &&
      loom_amdgpu_type_is_f32(source_type)) {
    return loom_amdgpu_make_sgpr_type(context, &out_argument->abi_type);
  }
  return loom_amdgpu_map_value(user_data, context, source_function_op,
                               source_argument_id, source_type,
                               &out_argument->abi_type);
}

static iree_string_view_t loom_amdgpu_argument_value_name(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return iree_string_view_empty();
  }
  const loom_string_id_t name_id = loom_module_value(module, value_id)->name_id;
  if (name_id == LOOM_STRING_ID_INVALID || name_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[name_id];
}

static uint32_t loom_amdgpu_direct_arg_byte_count(loom_type_t abi_type) {
  const uint64_t byte_count =
      (uint64_t)loom_low_register_type_unit_count(abi_type) *
      LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_SIZE;
  IREE_ASSERT_LE(byte_count, UINT32_MAX);
  return (uint32_t)byte_count;
}

static void loom_amdgpu_align_kernarg_offset(uint64_t* inout_offset,
                                             uint32_t alignment) {
  *inout_offset = iree_align_uint64(*inout_offset, alignment);
  IREE_ASSERT_LE(*inout_offset, UINT32_MAX);
}

static void loom_amdgpu_assign_hal_resource_layout(
    const loom_low_lower_abi_argument_t* argument, uint16_t parameter_index,
    uint64_t* inout_kernarg_offset,
    loom_amdgpu_hal_kernarg_resource_t* resources,
    iree_host_size_t resource_count) {
  IREE_ASSERT(argument->resource_index >= 0 &&
                  (uint64_t)argument->resource_index < resource_count,
              "AMDGPU HAL ABI argument mapping produced an invalid resource "
              "index");
  loom_amdgpu_align_kernarg_offset(
      inout_kernarg_offset,
      LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT);
  const iree_host_size_t binding_index =
      (iree_host_size_t)argument->resource_index;
  IREE_ASSERT(resources[binding_index].kernarg_size == 0,
              "AMDGPU HAL ABI argument mapping produced duplicate resource "
              "indexes");
  resources[binding_index] = (loom_amdgpu_hal_kernarg_resource_t){
      .resource_op = NULL,
      .name = iree_string_view_empty(),
      .binding_index = (uint32_t)binding_index,
      .parameter_index = parameter_index,
      .kernarg_offset = (uint32_t)*inout_kernarg_offset,
      .kernarg_size = LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE,
      .kernarg_alignment =
          LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT,
      .source_type = argument->resource_source_type,
      .abi_type = argument->abi_type,
  };
  *inout_kernarg_offset +=
      LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE;
  IREE_ASSERT_LE(*inout_kernarg_offset, UINT32_MAX);
}

static void loom_amdgpu_assign_hal_direct_arg_layout(
    const loom_module_t* module, const loom_value_id_t* source_arguments,
    const loom_low_lower_abi_argument_t* argument, uint16_t parameter_index,
    uint16_t direct_argument_index, uint64_t* inout_kernarg_offset,
    uint64_t* inout_constant_count,
    loom_amdgpu_hal_kernarg_direct_arg_t* direct_args) {
  loom_amdgpu_align_kernarg_offset(
      inout_kernarg_offset,
      LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_ALIGNMENT);
  const uint32_t kernarg_size =
      loom_amdgpu_direct_arg_byte_count(argument->abi_type);
  direct_args[direct_argument_index] = (loom_amdgpu_hal_kernarg_direct_arg_t){
      .arg_id = LOOM_VALUE_ID_INVALID,
      .name = loom_amdgpu_argument_value_name(
          module, source_arguments[parameter_index]),
      .parameter_index = parameter_index,
      .argument_index = direct_argument_index,
      .kernarg_offset = (uint32_t)*inout_kernarg_offset,
      .kernarg_size = kernarg_size,
      .kernarg_alignment =
          LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_ALIGNMENT,
      .abi_type = argument->abi_type,
  };
  *inout_kernarg_offset += kernarg_size;
  *inout_constant_count +=
      kernarg_size / LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_SIZE;
  IREE_ASSERT_LE(*inout_kernarg_offset, UINT32_MAX);
  IREE_ASSERT_LE(*inout_constant_count, UINT32_MAX);
}

iree_status_t loom_amdgpu_map_abi_layout(
    void* user_data, loom_low_lower_context_t* context,
    loom_low_lower_abi_layout_kind_t layout_kind, const loom_type_t* arg_types,
    iree_host_size_t arg_count, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_named_attr_slice_t* out_abi_layout) {
  (void)user_data;
  (void)arg_types;
  (void)result_types;
  *out_abi_layout = loom_named_attr_slice_empty();
  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  if (layout_kind != LOOM_LOW_LOWER_ABI_LAYOUT_KIND_KERNEL ||
      bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return iree_ok_status();
  }
  loom_func_like_t source_function =
      loom_low_lower_context_source_function(context);
  if (loom_func_like_export_symbol(source_function) == LOOM_STRING_ID_INVALID &&
      !loom_low_lower_context_source_is_retained(context)) {
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(result_count, 0,
                 "AMDGPU HAL ABI layout mapping reached a kernel with results");

  uint16_t parameter_count = 0;
  const loom_low_lower_abi_argument_t* argument_map =
      loom_low_lower_context_argument_map(context, &parameter_count);
  if (parameter_count == 0) {
    return iree_ok_status();
  }
  uint16_t source_argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(source_function, &source_argument_count);
  IREE_ASSERT(source_argument_count == parameter_count,
              "AMDGPU HAL ABI layout mapping reached an inconsistent argument "
              "map");

  iree_host_size_t resource_count = 0;
  iree_host_size_t direct_arg_count = 0;
  for (uint16_t i = 0; i < parameter_count; ++i) {
    if (argument_map[i].kind == LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE) {
      ++resource_count;
    } else {
      ++direct_arg_count;
    }
  }
  IREE_ASSERT(
      arg_count == direct_arg_count,
      "AMDGPU HAL ABI layout mapping reached an inconsistent low signature");

  loom_amdgpu_hal_kernarg_resource_t* resources = NULL;
  if (resource_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, resource_count, sizeof(*resources), (void**)&resources));
    memset(resources, 0, resource_count * sizeof(*resources));
  }
  loom_amdgpu_hal_kernarg_direct_arg_t* direct_args = NULL;
  if (direct_arg_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, direct_arg_count, sizeof(*direct_args), (void**)&direct_args));
    memset(direct_args, 0, direct_arg_count * sizeof(*direct_args));
  }

  uint64_t kernarg_offset = 0;
  uint64_t constant_count = 0;
  uint16_t direct_argument_index = 0;
  for (uint16_t parameter_index = 0; parameter_index < parameter_count;
       ++parameter_index) {
    const loom_low_lower_abi_argument_t* argument =
        &argument_map[parameter_index];
    if (argument->kind == LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE) {
      loom_amdgpu_assign_hal_resource_layout(argument, parameter_index,
                                             &kernarg_offset, resources,
                                             resource_count);
      continue;
    }
    loom_amdgpu_assign_hal_direct_arg_layout(
        loom_low_lower_context_module(context), source_arguments, argument,
        parameter_index, direct_argument_index, &kernarg_offset,
        &constant_count, direct_args);
    ++direct_argument_index;
  }
  loom_amdgpu_align_kernarg_offset(
      &kernarg_offset,
      LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT);
  IREE_ASSERT_LE(constant_count, UINT32_MAX);

  const loom_amdgpu_hal_kernel_abi_layout_t layout = {
      .function_op = NULL,
      .parameter_count = parameter_count,
      .kernarg_segment_size = (uint32_t)kernarg_offset,
      .kernarg_segment_alignment =
          LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT,
      .uses_kernarg_segment_ptr = resource_count != 0 || direct_arg_count != 0,
      .constant_count = (uint32_t)constant_count,
      .resources = resources,
      .resource_count = resource_count,
      .direct_args = direct_args,
      .direct_arg_count = direct_arg_count,
  };
  loom_attribute_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_make_layout_attr(
      loom_low_lower_context_module(context), &layout,
      loom_low_lower_context_emission_arena(context), &attr));
  *out_abi_layout = loom_attr_as_dict(attr);
  return iree_ok_status();
}
