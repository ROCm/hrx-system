// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/kernel_config_materializer.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "loom/format/bytecode/function_projection_reader.h"
#include "loom/format/bytecode/selected_reader.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/symbol_map.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/type_registry.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/adaptive_sort.h"

typedef struct loom_link_bytecode_kernel_config_source_t {
  // Indexed bytecode provider retaining the source storage.
  const loom_link_module_index_provider_t* provider;
  // Provider-local source module metadata.
  const loom_bytecode_module_metadata_t* module;
  // Indexed source module containing the selected kernel.
  const loom_link_module_index_module_t* indexed_module;
  // Module-local source kernel metadata.
  const loom_bytecode_symbol_metadata_t* symbol;
  // Module-local source kernel ordinal.
  uint32_t symbol_ordinal;
  // Symbol-local ordinal of the launch-configuration payload.
  uint8_t config_payload_ordinal;
} loom_link_bytecode_kernel_config_source_t;

// Cross-module state for projecting selected materialized IR.
typedef struct loom_link_kernel_config_ir_projection_t {
  // Source module whose selected operations are being projected.
  const loom_module_t* source_module;
  // Exact module-local selection and compact target ordinal mapping.
  const loom_link_plan_module_selection_t* selection;
  // Compact target module under construction.
  loom_module_t* target_module;
  // Scratch storage shared by remap tables and temporary signatures.
  iree_arena_allocator_t* scratch_arena;
  // Target configuration-function refs aligned with selection.symbols.
  loom_symbol_ref_t* configuration_functions;
} loom_link_kernel_config_ir_projection_t;

// One selected source symbol ordered by its defining operation.
typedef struct loom_link_kernel_config_ir_symbol_t {
  // Source operation, or NULL for an unresolved symbol anchor.
  const loom_op_t* source_op;
  // Entry in the module selection.
  iree_host_size_t selection_ordinal;
} loom_link_kernel_config_ir_symbol_t;

// Source-to-target symbol projection for selected bytecode materialization.
typedef struct loom_link_kernel_config_bytecode_projection_t {
  // Exact module-local selection and compact target ordinal mapping.
  const loom_link_plan_module_selection_t* selection;
} loom_link_kernel_config_bytecode_projection_t;

static bool loom_link_kernel_config_ir_symbol_less(
    const loom_link_kernel_config_ir_symbol_t* lhs,
    const loom_link_kernel_config_ir_symbol_t* rhs) {
  return lhs->source_op->block_ordinal < rhs->source_op->block_ordinal;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_link_kernel_config_sort_ir_symbols,
                          loom_link_kernel_config_ir_symbol_t,
                          loom_link_kernel_config_ir_symbol_less)

static loom_diagnostic_sink_t loom_link_kernel_config_diagnostic_sink(
    const loom_link_plan_materialization_environment_t* environment,
    const loom_link_module_index_provider_t* provider) {
  if (environment->diagnostic_sink == NULL) {
    return (loom_diagnostic_sink_t){0};
  }
  return environment->diagnostic_sink(environment->user_data, provider);
}

static iree_status_t loom_link_kernel_config_add_output_symbol(
    loom_module_t* module, iree_string_view_t name,
    loom_symbol_ref_t* out_ref) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, name, &name_id));
  IREE_RETURN_IF_ERROR(
      loom_module_add_symbol(module, name_id, &out_ref->symbol_id));
  out_ref->module_id = 0;
  return iree_ok_status();
}

// Adds a readable private helper name after all authored source names have
// been reserved. Names are presentation only; selected source ordinals remain
// the structural identity throughout projection.
static iree_status_t loom_link_kernel_config_add_helper_symbol(
    loom_module_t* module, iree_string_view_t source_name,
    iree_arena_allocator_t* scratch_arena, loom_symbol_map_t* symbol_map,
    iree_host_size_t* next_conflict_ordinal, loom_symbol_ref_t* out_ref) {
  const iree_string_view_t suffix = IREE_SV("$config");
  iree_host_size_t storage_capacity = 0;
  if (!iree_host_size_checked_add(source_name.size, suffix.size + 32,
                                  &storage_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "configuration symbol name overflow");
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(scratch_arena, storage_capacity, (void**)&storage));
  memcpy(storage, source_name.data, source_name.size);
  memcpy(storage + source_name.size, suffix.data, suffix.size);

  iree_host_size_t name_length = source_name.size + suffix.size;
  while (true) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        module, iree_make_string_view(storage, name_length), &name_id));
    if (loom_symbol_map_find(symbol_map, name_id) == LOOM_SYMBOL_ID_INVALID) {
      IREE_RETURN_IF_ERROR(
          loom_module_add_symbol(module, name_id, &out_ref->symbol_id));
      out_ref->module_id = 0;
      return loom_symbol_map_insert(symbol_map, scratch_arena, name_id,
                                    out_ref->symbol_id);
    }

    const int suffix_length =
        snprintf(storage + source_name.size + suffix.size, 32, "$%" PRIhsz,
                 (*next_conflict_ordinal)++);
    if (suffix_length < 0 || suffix_length >= 32) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "configuration symbol suffix overflow");
    }
    name_length =
        source_name.size + suffix.size + (iree_host_size_t)suffix_length;
  }
}

static iree_status_t loom_link_kernel_config_build_declaration(
    loom_module_t* module, const loom_bytecode_function_header_t* header,
    loom_builder_t* builder, loom_op_t** out_declaration) {
  if (header->result_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel definition has function results");
  }
  const iree_host_size_t operand_count =
      (iree_host_size_t)header->workload_argument_count +
      header->argument_count;
  if (operand_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel signature exceeds operand capacity");
  }
  const loom_op_vtable_t* vtable =
      loom_context_resolve_op(module->context, LOOM_OP_KERNEL_DECL);
  IREE_ASSERT(vtable != NULL);
  const uint16_t operand_segments[] = {
      header->workload_argument_count,
      header->argument_count,
  };
  loom_op_t* declaration = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_segmented_op(
      builder, LOOM_OP_KERNEL_DECL, (uint16_t)operand_count, operand_segments,
      IREE_ARRAYSIZE(operand_segments), /*result_count=*/0, /*region_count=*/0,
      /*tied_result_count=*/0, vtable->attribute_count, LOOM_LOCATION_NONE,
      &declaration));
  if (operand_count != 0) {
    memcpy(loom_op_operands(declaration), header->signature_values,
           operand_count * sizeof(loom_value_id_t));
  }
  loom_attribute_t* attributes = loom_op_attrs(declaration);
  attributes[loom_kernel_decl_callee_ATTR_INDEX] =
      header->attributes[loom_kernel_def_callee_ATTR_INDEX];
  attributes[loom_kernel_decl_target_ATTR_INDEX] =
      header->attributes[loom_kernel_def_target_ATTR_INDEX];
  const loom_attribute_t source_predicates =
      header->attributes[loom_kernel_def_predicates_ATTR_INDEX];
  if (!loom_attr_is_absent(source_predicates)) {
    loom_predicate_t* predicates = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(&module->arena, source_predicates.count,
                                  sizeof(*predicates), (void**)&predicates));
    memcpy(predicates, source_predicates.predicate_list,
           source_predicates.count * sizeof(*predicates));
    attributes[loom_kernel_decl_predicates_ATTR_INDEX] =
        loom_attr_predicate_list(predicates, source_predicates.count);
  }
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, declaration));
  *out_declaration = declaration;
  return iree_ok_status();
}

static bool loom_link_kernel_config_predicate_uses_mapped_values(
    const loom_ir_remap_t* remap, const loom_predicate_t* predicate) {
  for (uint8_t i = 0; i < predicate->arg_count; ++i) {
    if (predicate->arg_tags[i] != LOOM_PRED_ARG_VALUE) continue;
    loom_value_id_t ignored = LOOM_VALUE_ID_INVALID;
    if (!loom_ir_remap_try_lookup_value(
            remap, (loom_value_id_t)predicate->args[i], &ignored)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_link_kernel_config_copy_workload_predicates(
    loom_module_t* module, const loom_bytecode_function_header_t* header,
    loom_ir_remap_t* remap, iree_arena_allocator_t* scratch_arena,
    loom_op_t* helper_op) {
  const loom_attribute_t source_predicates =
      header->attributes[loom_kernel_def_predicates_ATTR_INDEX];
  if (loom_attr_is_absent(source_predicates)) return iree_ok_status();

  loom_predicate_t* workload_predicates = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, source_predicates.count, sizeof(*workload_predicates),
      (void**)&workload_predicates));
  uint16_t workload_predicate_count = 0;
  for (uint16_t i = 0; i < source_predicates.count; ++i) {
    if (loom_link_kernel_config_predicate_uses_mapped_values(
            remap, &source_predicates.predicate_list[i])) {
      workload_predicates[workload_predicate_count++] =
          source_predicates.predicate_list[i];
    }
  }
  if (workload_predicate_count == 0) return iree_ok_status();

  loom_predicate_t* target_predicates = NULL;
  IREE_RETURN_IF_ERROR(loom_ir_remap_predicate_list(remap, workload_predicates,
                                                    workload_predicate_count,
                                                    &target_predicates));
  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, scratch_arena));
  const iree_status_t status = loom_rewriter_set_attr(
      &rewriter, helper_op, loom_func_def_predicates_ATTR_INDEX,
      loom_attr_predicate_list(target_predicates, workload_predicate_count));
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

static iree_status_t loom_link_kernel_config_configure_helper_arguments(
    loom_module_t* module, const loom_bytecode_function_header_t* header,
    loom_func_like_t helper, iree_arena_allocator_t* scratch_arena) {
  uint16_t helper_argument_count = 0;
  const loom_value_id_t* helper_arguments =
      loom_func_like_arg_ids(helper, &helper_argument_count);
  IREE_ASSERT(helper_argument_count == header->workload_argument_count);

  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(module, module, scratch_arena,
                                                /*options=*/NULL, &remap));
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(
      &remap, header->signature_values, helper_arguments,
      header->workload_argument_count));
  for (uint16_t i = 0; i < header->workload_argument_count; ++i) {
    loom_type_t target_type = {0};
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        &remap, loom_module_value_type(module, header->signature_values[i]),
        &target_type));
    IREE_RETURN_IF_ERROR(
        loom_module_set_value_type(module, helper_arguments[i], target_type));
    IREE_RETURN_IF_ERROR(loom_module_copy_value_name(
        module, header->signature_values[i], helper_arguments[i]));
  }
  return loom_link_kernel_config_copy_workload_predicates(
      module, header, &remap, scratch_arena, helper.op);
}

static iree_status_t loom_link_kernel_config_build_helper(
    loom_module_t* module, const loom_bytecode_function_header_t* header,
    loom_symbol_ref_t config_ref, iree_arena_allocator_t* scratch_arena,
    loom_builder_t* builder, loom_func_like_t* out_helper) {
  loom_type_t* argument_types = NULL;
  if (header->workload_argument_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, header->workload_argument_count, sizeof(*argument_types),
        (void**)&argument_types));
    for (uint16_t i = 0; i < header->workload_argument_count; ++i) {
      argument_types[i] = loom_type_none();
    }
  }
  const loom_type_t result_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
  };
  const loom_attribute_t target_attr =
      header->attributes[loom_kernel_def_target_ATTR_INDEX];
  const loom_symbol_ref_t target_ref = loom_attr_is_absent(target_attr)
                                           ? loom_symbol_ref_null()
                                           : loom_attr_as_symbol(target_attr);
  loom_func_def_build_flags_t build_flags =
      LOOM_FUNC_DEF_BUILD_FLAG_HAS_PURITY |
      LOOM_FUNC_DEF_BUILD_FLAG_HAS_INLINE_POLICY;
  if (loom_symbol_ref_is_valid(target_ref)) {
    build_flags |= LOOM_FUNC_DEF_BUILD_FLAG_HAS_TARGET;
  }
  loom_op_t* helper_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_def_build(
      builder, build_flags,
      /*visibility=*/0, /*retain=*/0, /*cc=*/0, LOOM_FUNC_PURITY_PURE,
      /*temperature=*/0, LOOM_INLINE_POLICY_INLINE, target_ref,
      /*abi=*/0, loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), config_ref, argument_types,
      header->workload_argument_count, result_types,
      IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, /*predicates=*/NULL,
      /*predicates_count=*/0, LOOM_LOCATION_NONE, &helper_op));
  const loom_func_like_t helper = loom_func_like_cast(module, helper_op);
  IREE_RETURN_IF_ERROR(loom_link_kernel_config_configure_helper_arguments(
      module, header, helper, scratch_arena));
  *out_helper = helper;
  return iree_ok_status();
}

static iree_status_t loom_link_kernel_config_materialize_body(
    const loom_link_bytecode_kernel_config_source_t* source,
    loom_bytecode_function_projection_reader_t* reader,
    const loom_bytecode_function_header_t* header, loom_module_t* module,
    loom_builder_t* builder, loom_func_like_t helper) {
  uint16_t helper_argument_count = 0;
  const loom_value_id_t* helper_arguments =
      loom_func_like_arg_ids(helper, &helper_argument_count);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_function_projection_reader_materialize_region(
          reader, header, source->config_payload_ordinal, builder, helper.op,
          loom_func_like_body_region_index(helper), helper_arguments,
          helper_argument_count, /*low_descriptor_set=*/NULL));

  loom_block_t* block = loom_region_entry_block(loom_func_like_body(helper));
  loom_op_t* launch_config = block->last_op;
  if (!loom_kernel_launch_config_isa(launch_config)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel configuration does not end in kernel.launch.config");
  }
  loom_value_id_t return_values[3];
  memcpy(return_values, loom_op_const_operands(launch_config),
         sizeof(return_values));
  const loom_location_id_t location = launch_config->location;
  IREE_RETURN_IF_ERROR(loom_op_erase(module, launch_config));
  loom_builder_initialize(module, &module->arena, block, builder);
  loom_op_t* return_op = NULL;
  return loom_func_return_build(builder, return_values,
                                IREE_ARRAYSIZE(return_values), location,
                                &return_op);
}

static iree_status_t loom_link_kernel_config_materialize_projection_body(
    const loom_link_bytecode_kernel_config_source_t* source,
    loom_bytecode_function_projection_reader_t* reader,
    iree_arena_allocator_t* scratch_arena, loom_module_t* module,
    loom_symbol_ref_t config_ref) {
  loom_bytecode_function_header_t header = {0};
  iree_status_t status = loom_bytecode_function_projection_reader_read_header(
      reader, source->symbol_ordinal, &header);

  loom_builder_t builder;
  loom_op_t* declaration = NULL;
  if (iree_status_is_ok(status)) {
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    status = loom_link_kernel_config_build_declaration(module, &header,
                                                       &builder, &declaration);
  }
  loom_func_like_t helper = {0};
  if (iree_status_is_ok(status)) {
    status = loom_link_kernel_config_build_helper(
        module, &header, config_ref, scratch_arena, &builder, &helper);
  }
  if (iree_status_is_ok(status)) {
    status = loom_link_kernel_config_materialize_body(source, reader, &header,
                                                      module, &builder, helper);
  }

  return status;
}

static bool loom_link_kernel_config_selection_is_partial(
    const loom_link_plan_module_symbol_t* selection) {
  return selection->plan_symbol->selected_facet_count !=
         selection->source_symbol->facets.schema.facet_count;
}

static iree_status_t loom_link_kernel_config_validate_partial_selection(
    const loom_link_plan_t* plan,
    const loom_link_plan_module_symbol_t* selection) {
  const iree_host_size_t symbol_ordinal = selection->source_symbol->ordinal;
  if (!iree_all_bits_set(selection->source_symbol->facets.schema.interfaces,
                         LOOM_SYMBOL_INTERFACE_KERNEL) ||
      !loom_link_plan_contains_facet(plan, symbol_ordinal,
                                     LOOM_LINK_SYMBOL_FACET_KERNEL_CONTRACT) ||
      !loom_link_plan_contains_facet(
          plan, symbol_ordinal, LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION) ||
      loom_link_plan_contains_facet(
          plan, symbol_ordinal, LOOM_LINK_SYMBOL_FACET_KERNEL_IMPLEMENTATION)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "partial materialization of symbol '@%.*s' is not a kernel "
        "contract-plus-configuration projection",
        (int)selection->source_symbol->name.size,
        selection->source_symbol->name.data);
  }
  return iree_ok_status();
}

static const loom_link_plan_module_symbol_t*
loom_link_kernel_config_find_selected_source_symbol(
    const loom_link_plan_module_selection_t* selection,
    uint32_t source_symbol_ordinal) {
  iree_host_size_t lower_bound = 0;
  iree_host_size_t upper_bound = selection->symbols.count;
  while (lower_bound < upper_bound) {
    const iree_host_size_t middle =
        lower_bound + (upper_bound - lower_bound) / 2;
    const iree_host_size_t candidate =
        selection->symbols.values[middle].source_symbol->module_symbol_ordinal;
    if (candidate < source_symbol_ordinal) {
      lower_bound = middle + 1;
    } else {
      upper_bound = middle;
    }
  }
  if (lower_bound == selection->symbols.count ||
      selection->symbols.values[lower_bound]
              .source_symbol->module_symbol_ordinal != source_symbol_ordinal) {
    return NULL;
  }
  return &selection->symbols.values[lower_bound];
}

static iree_status_t loom_link_kernel_config_resolve_bytecode_symbol(
    void* user_data, uint32_t source_symbol_ordinal,
    loom_symbol_ref_t* out_target_ref) {
  const loom_link_kernel_config_bytecode_projection_t* projection =
      (const loom_link_kernel_config_bytecode_projection_t*)user_data;
  const loom_link_plan_module_symbol_t* selected =
      loom_link_kernel_config_find_selected_source_symbol(
          projection->selection, source_symbol_ordinal);
  if (!selected) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "kernel configuration plan omitted referenced source symbol %u",
        (unsigned)source_symbol_ordinal);
  }
  *out_target_ref = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = (uint16_t)selected->materialized_symbol_ordinal,
  };
  return iree_ok_status();
}

static iree_status_t loom_link_kernel_config_remap_ir_symbol(
    void* user_data, const loom_module_t* source_module,
    loom_module_t* target_module, loom_symbol_ref_t source_ref,
    loom_symbol_ref_t* out_target_ref) {
  loom_link_kernel_config_ir_projection_t* projection =
      (loom_link_kernel_config_ir_projection_t*)user_data;
  IREE_ASSERT(source_module == projection->source_module);
  IREE_ASSERT(target_module == projection->target_module);
  const loom_link_plan_module_symbol_t* selected =
      loom_link_kernel_config_find_selected_source_symbol(projection->selection,
                                                          source_ref.symbol_id);
  if (!selected) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "kernel configuration plan omitted referenced source symbol %u",
        (unsigned)source_ref.symbol_id);
  }
  *out_target_ref = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = (uint16_t)selected->materialized_symbol_ordinal,
  };
  return iree_ok_status();
}

static loom_ir_remap_options_t loom_link_kernel_config_ir_remap_options(
    loom_link_kernel_config_ir_projection_t* projection) {
  return (loom_ir_remap_options_t){
      .remap_symbol = loom_ir_remap_symbol_callback_make(
          loom_link_kernel_config_remap_ir_symbol, projection),
  };
}

static iree_status_t loom_link_kernel_config_copy_ir_value_name(
    loom_link_kernel_config_ir_projection_t* projection, loom_ir_remap_t* remap,
    loom_value_id_t source_value, loom_value_id_t target_value) {
  const loom_string_id_t source_name =
      loom_module_value(projection->source_module, source_value)->name_id;
  if (source_name == LOOM_STRING_ID_INVALID) return iree_ok_status();
  loom_string_id_t target_name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_ir_remap_string_id(
      remap, source_name, /*allow_invalid=*/false, &target_name));
  return loom_module_set_value_name(projection->target_module, target_value,
                                    target_name);
}

static iree_status_t loom_link_kernel_config_configure_ir_values(
    loom_link_kernel_config_ir_projection_t* projection,
    const loom_value_id_t* source_values, const loom_value_id_t* target_values,
    iree_host_size_t value_count, loom_ir_remap_t* remap) {
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    loom_type_t target_type = {0};
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        remap,
        loom_module_value_type(projection->source_module, source_values[i]),
        &target_type));
    IREE_RETURN_IF_ERROR(loom_module_set_value_type(
        projection->target_module, target_values[i], target_type));
    IREE_RETURN_IF_ERROR(loom_link_kernel_config_copy_ir_value_name(
        projection, remap, source_values[i], target_values[i]));
  }
  return iree_ok_status();
}

typedef enum loom_link_kernel_config_predicate_projection_e {
  LOOM_LINK_KERNEL_CONFIG_PREDICATE_PROJECTION_ALL = 0,
  LOOM_LINK_KERNEL_CONFIG_PREDICATE_PROJECTION_MAPPED_VALUES = 1,
} loom_link_kernel_config_predicate_projection_t;

static iree_status_t loom_link_kernel_config_copy_ir_predicates(
    loom_link_kernel_config_ir_projection_t* projection,
    loom_func_like_t source_function, loom_ir_remap_t* remap,
    loom_link_kernel_config_predicate_projection_t predicate_projection,
    loom_op_t* target_op, uint8_t target_attr_index) {
  uint16_t source_predicate_count = 0;
  const loom_predicate_t* source_predicates =
      loom_func_like_predicates(source_function, &source_predicate_count);
  if (source_predicate_count == 0) return iree_ok_status();

  loom_predicate_t* selected_predicates = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      projection->scratch_arena, source_predicate_count,
      sizeof(*selected_predicates), (void**)&selected_predicates));
  uint16_t selected_predicate_count = 0;
  for (uint16_t i = 0; i < source_predicate_count; ++i) {
    if (predicate_projection ==
            LOOM_LINK_KERNEL_CONFIG_PREDICATE_PROJECTION_MAPPED_VALUES &&
        !loom_link_kernel_config_predicate_uses_mapped_values(
            remap, &source_predicates[i])) {
      continue;
    }
    selected_predicates[selected_predicate_count++] = source_predicates[i];
  }
  if (selected_predicate_count == 0) return iree_ok_status();

  loom_predicate_t* target_predicates = NULL;
  IREE_RETURN_IF_ERROR(loom_ir_remap_predicate_list(remap, selected_predicates,
                                                    selected_predicate_count,
                                                    &target_predicates));
  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(loom_rewriter_initialize(
      &rewriter, projection->target_module, projection->scratch_arena));
  const iree_status_t status = loom_rewriter_set_attr(
      &rewriter, target_op, target_attr_index,
      loom_attr_predicate_list(target_predicates, selected_predicate_count));
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

static iree_status_t loom_link_kernel_config_allocate_none_types(
    iree_host_size_t count, iree_arena_allocator_t* arena,
    loom_type_t** out_types) {
  *out_types = NULL;
  if (count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, count, sizeof(**out_types), (void**)out_types));
  for (iree_host_size_t i = 0; i < count; ++i) {
    (*out_types)[i] = loom_type_none();
  }
  return iree_ok_status();
}

static iree_status_t loom_link_kernel_config_build_ir_declaration(
    loom_link_kernel_config_ir_projection_t* projection,
    const loom_link_plan_module_symbol_t* selected, const loom_op_t* source_op,
    loom_builder_t* builder, loom_op_t** out_declaration) {
  loom_func_like_t source_function =
      loom_func_like_const_cast(projection->source_module, source_op);
  const loom_value_slice_t source_workloads =
      loom_kernel_workload_arg_ids(projection->source_module, source_op);
  uint16_t source_argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(source_function, &source_argument_count);

  loom_ir_remap_t remap = {0};
  const loom_ir_remap_options_t remap_options =
      loom_link_kernel_config_ir_remap_options(projection);
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      projection->source_module, projection->target_module,
      projection->scratch_arena, &remap_options, &remap));

  loom_type_t* workload_types = NULL;
  loom_type_t* argument_types = NULL;
  IREE_RETURN_IF_ERROR(loom_link_kernel_config_allocate_none_types(
      source_workloads.count, projection->scratch_arena, &workload_types));
  IREE_RETURN_IF_ERROR(loom_link_kernel_config_allocate_none_types(
      source_argument_count, projection->scratch_arena, &argument_types));

  loom_symbol_ref_t target = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_ir_remap_symbol_ref(
      &remap, loom_func_like_target(source_function), &target));
  loom_string_id_t export_symbol = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_ir_remap_string_id(
      &remap, loom_func_like_export_symbol(source_function),
      /*allow_invalid=*/true, &export_symbol));
  loom_location_id_t location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_location_id(&remap, source_op->location, &location));

  const loom_attribute_t* source_attrs = loom_op_const_attrs(source_op);
  loom_kernel_decl_build_flags_t build_flags = 0;
  if (!loom_attr_is_absent(source_attrs[loom_kernel_def_retain_ATTR_INDEX])) {
    build_flags |= LOOM_KERNEL_DECL_BUILD_FLAG_HAS_RETAIN;
  }
  if (loom_symbol_ref_is_valid(target)) {
    build_flags |= LOOM_KERNEL_DECL_BUILD_FLAG_HAS_TARGET;
  }
  if (export_symbol != LOOM_STRING_ID_INVALID) {
    build_flags |= LOOM_KERNEL_DECL_BUILD_FLAG_HAS_EXPORT_SYMBOL;
  }
  if (!loom_attr_is_absent(
          source_attrs[loom_kernel_def_export_linkage_ATTR_INDEX])) {
    build_flags |= LOOM_KERNEL_DECL_BUILD_FLAG_HAS_EXPORT_LINKAGE;
  }
  const loom_symbol_ref_t target_callee = {
      .module_id = 0,
      .symbol_id = (uint16_t)selected->materialized_symbol_ordinal,
  };
  IREE_RETURN_IF_ERROR(loom_kernel_decl_build(
      builder, build_flags, loom_kernel_def_retain(source_op), target,
      export_symbol, loom_kernel_def_export_linkage(source_op), target_callee,
      workload_types, source_workloads.count, argument_types,
      source_argument_count, /*predicates=*/NULL, /*predicates_count=*/0,
      location, out_declaration));

  const loom_value_slice_t target_workloads =
      loom_kernel_workload_arg_ids(projection->target_module, *out_declaration);
  loom_func_like_t target_function =
      loom_func_like_cast(projection->target_module, *out_declaration);
  uint16_t target_argument_count = 0;
  const loom_value_id_t* target_arguments =
      loom_func_like_arg_ids(target_function, &target_argument_count);
  IREE_ASSERT_EQ(source_workloads.count, target_workloads.count);
  IREE_ASSERT_EQ(source_argument_count, target_argument_count);
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(&remap, source_workloads.values,
                                                target_workloads.values,
                                                source_workloads.count));
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(
      &remap, source_arguments, target_arguments, source_argument_count));
  IREE_RETURN_IF_ERROR(loom_link_kernel_config_configure_ir_values(
      projection, source_workloads.values, target_workloads.values,
      source_workloads.count, &remap));
  IREE_RETURN_IF_ERROR(loom_link_kernel_config_configure_ir_values(
      projection, source_arguments, target_arguments, source_argument_count,
      &remap));
  return loom_link_kernel_config_copy_ir_predicates(
      projection, source_function, &remap,
      LOOM_LINK_KERNEL_CONFIG_PREDICATE_PROJECTION_ALL, *out_declaration,
      loom_kernel_decl_predicates_ATTR_INDEX);
}

static iree_status_t loom_link_kernel_config_build_ir_helper(
    loom_link_kernel_config_ir_projection_t* projection,
    iree_host_size_t selection_ordinal, const loom_op_t* source_op,
    loom_builder_t* builder) {
  loom_func_like_t source_function =
      loom_func_like_const_cast(projection->source_module, source_op);
  const loom_value_slice_t source_workloads =
      loom_kernel_workload_arg_ids(projection->source_module, source_op);

  loom_ir_remap_t remap = {0};
  const loom_ir_remap_options_t remap_options =
      loom_link_kernel_config_ir_remap_options(projection);
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      projection->source_module, projection->target_module,
      projection->scratch_arena, &remap_options, &remap));

  loom_type_t* argument_types = NULL;
  IREE_RETURN_IF_ERROR(loom_link_kernel_config_allocate_none_types(
      source_workloads.count, projection->scratch_arena, &argument_types));
  const loom_type_t result_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
  };
  loom_symbol_ref_t target = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_ir_remap_symbol_ref(
      &remap, loom_func_like_target(source_function), &target));
  loom_location_id_t location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_location_id(&remap, source_op->location, &location));
  loom_func_def_build_flags_t build_flags =
      LOOM_FUNC_DEF_BUILD_FLAG_HAS_PURITY |
      LOOM_FUNC_DEF_BUILD_FLAG_HAS_INLINE_POLICY;
  if (loom_symbol_ref_is_valid(target)) {
    build_flags |= LOOM_FUNC_DEF_BUILD_FLAG_HAS_TARGET;
  }
  loom_op_t* helper_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_def_build(
      builder, build_flags,
      /*visibility=*/0, /*retain=*/0, /*cc=*/0, LOOM_FUNC_PURITY_PURE,
      /*temperature=*/0, LOOM_INLINE_POLICY_INLINE, target,
      /*abi=*/0, loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(),
      projection->configuration_functions[selection_ordinal], argument_types,
      source_workloads.count, result_types, IREE_ARRAYSIZE(result_types),
      /*tied_results=*/NULL, /*tied_result_count=*/0,
      /*predicates=*/NULL, /*predicates_count=*/0, location, &helper_op));

  const loom_func_like_t helper =
      loom_func_like_cast(projection->target_module, helper_op);
  uint16_t target_argument_count = 0;
  const loom_value_id_t* target_arguments =
      loom_func_like_arg_ids(helper, &target_argument_count);
  IREE_ASSERT_EQ(source_workloads.count, target_argument_count);
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(&remap, source_workloads.values,
                                                target_arguments,
                                                source_workloads.count));
  IREE_RETURN_IF_ERROR(loom_link_kernel_config_configure_ir_values(
      projection, source_workloads.values, target_arguments,
      source_workloads.count, &remap));
  IREE_RETURN_IF_ERROR(loom_link_kernel_config_copy_ir_predicates(
      projection, source_function, &remap,
      LOOM_LINK_KERNEL_CONFIG_PREDICATE_PROJECTION_MAPPED_VALUES, helper_op,
      loom_func_def_predicates_ATTR_INDEX));

  const loom_region_t* source_region = loom_kernel_def_config(source_op);
  const loom_block_t* source_block =
      loom_region_const_entry_block(source_region);
  const loom_op_t* launch_config = loom_kernel_def_launch_config_op(source_op);
  if (!launch_config) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "selected kernel configuration has no launch terminator");
  }
  loom_builder_t body_builder;
  loom_builder_initialize(
      projection->target_module, &projection->target_module->arena,
      loom_region_entry_block(loom_func_like_body(helper)), &body_builder);
  const loom_op_t* source_config_op = NULL;
  loom_block_for_each_op(source_block, source_config_op) {
    if (source_config_op == launch_config) continue;
    loom_op_t* cloned_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_ir_clone_op(&body_builder, source_config_op, &remap, &cloned_op));
  }

  loom_value_id_t return_values[3];
  for (uint8_t i = 0; i < IREE_ARRAYSIZE(return_values); ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
        &remap,
        loom_kernel_launch_config_workgroup_count_operand(
            launch_config, (loom_kernel_dimension_t)i),
        &return_values[i]));
  }
  loom_location_id_t return_location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_ir_remap_location_id(
      &remap, launch_config->location, &return_location));
  loom_op_t* return_op = NULL;
  return loom_func_return_build(&body_builder, return_values,
                                IREE_ARRAYSIZE(return_values), return_location,
                                &return_op);
}

static iree_status_t loom_link_kernel_config_initialize_output_symbols(
    const loom_link_plan_module_selection_t* selection,
    loom_module_t* target_module, iree_arena_allocator_t* scratch_arena,
    loom_symbol_ref_t* configuration_functions) {
  loom_symbol_map_t symbol_map = {0};
  for (iree_host_size_t projection_kind = 0; projection_kind < 2;
       ++projection_kind) {
    const bool select_partial = projection_kind != 0;
    for (iree_host_size_t i = 0; i < selection->symbols.count; ++i) {
      const loom_link_plan_module_symbol_t* selected =
          &selection->symbols.values[i];
      if (loom_link_kernel_config_selection_is_partial(selected) !=
          select_partial) {
        continue;
      }
      loom_symbol_ref_t target_ref = loom_symbol_ref_null();
      IREE_RETURN_IF_ERROR(loom_link_kernel_config_add_output_symbol(
          target_module, selected->source_symbol->name, &target_ref));
      IREE_ASSERT_EQ(target_ref.symbol_id,
                     selected->materialized_symbol_ordinal);
      const loom_string_id_t name_id =
          target_module->symbols.entries[target_ref.symbol_id].name_id;
      IREE_RETURN_IF_ERROR(loom_symbol_map_insert(
          &symbol_map, scratch_arena, name_id, target_ref.symbol_id));
    }
  }

  iree_host_size_t next_conflict_ordinal = 0;
  iree_host_size_t partial_ordinal = 0;
  for (iree_host_size_t i = 0; i < selection->symbols.count; ++i) {
    const loom_link_plan_module_symbol_t* selected =
        &selection->symbols.values[i];
    if (!loom_link_kernel_config_selection_is_partial(selected)) continue;
    IREE_RETURN_IF_ERROR(loom_link_kernel_config_add_helper_symbol(
        target_module, selected->source_symbol->name, scratch_arena,
        &symbol_map, &next_conflict_ordinal, &configuration_functions[i]));
    IREE_ASSERT_EQ(configuration_functions[i].symbol_id,
                   selection->symbols.count + partial_ordinal++);
  }
  return iree_ok_status();
}

static iree_status_t loom_link_kernel_config_project_ir_module(
    const loom_link_plan_module_selection_t* selection,
    const loom_link_plan_materialization_environment_t* environment,
    iree_string_view_t module_name, iree_arena_allocator_t* scratch_arena,
    loom_symbol_ref_t* configuration_functions, loom_module_t** out_module) {
  *out_module = NULL;
  const loom_module_t* source_module =
      selection->source_module->materialized_module;
  IREE_ASSERT(source_module != NULL);
  const loom_module_size_hints_t hints = {
      .symbol_count = selection->projected_symbol_count,
  };
  loom_module_t* target_module = NULL;
  IREE_RETURN_IF_ERROR(loom_module_allocate(
      environment->context, module_name, environment->block_pool, &hints,
      environment->allocator, &target_module));

  loom_link_kernel_config_ir_projection_t projection = {
      .source_module = source_module,
      .selection = selection,
      .target_module = target_module,
      .scratch_arena = scratch_arena,
      .configuration_functions = configuration_functions,
  };
  iree_status_t status = loom_link_kernel_config_initialize_output_symbols(
      selection, target_module, scratch_arena, configuration_functions);

  loom_link_kernel_config_ir_symbol_t* ordered_symbols = NULL;
  if (iree_status_is_ok(status) && selection->symbols.count != 0) {
    status = iree_arena_allocate_array(scratch_arena, selection->symbols.count,
                                       sizeof(*ordered_symbols),
                                       (void**)&ordered_symbols);
  }
  iree_host_size_t ordered_symbol_count = 0;
  for (iree_host_size_t i = 0;
       i < selection->symbols.count && iree_status_is_ok(status); ++i) {
    const iree_host_size_t source_ordinal =
        selection->symbols.values[i].source_symbol->module_symbol_ordinal;
    IREE_ASSERT_LT(source_ordinal, source_module->symbols.count);
    const loom_op_t* source_op =
        source_module->symbols.entries[source_ordinal].defining_op;
    if (!source_op) continue;
    ordered_symbols[ordered_symbol_count++] =
        (loom_link_kernel_config_ir_symbol_t){
            .source_op = source_op,
            .selection_ordinal = i,
        };
  }
  if (iree_status_is_ok(status)) {
    loom_link_kernel_config_sort_ir_symbols(ordered_symbols,
                                            ordered_symbol_count);
  }

  loom_ir_remap_t module_remap = {0};
  if (iree_status_is_ok(status)) {
    const loom_ir_remap_options_t remap_options =
        loom_link_kernel_config_ir_remap_options(&projection);
    status =
        loom_ir_remap_initialize(source_module, target_module, scratch_arena,
                                 &remap_options, &module_remap);
  }
  loom_builder_t builder;
  if (iree_status_is_ok(status)) {
    loom_builder_initialize(target_module, &target_module->arena,
                            loom_module_block(target_module), &builder);
  }
  for (iree_host_size_t i = 0;
       i < ordered_symbol_count && iree_status_is_ok(status); ++i) {
    const loom_link_kernel_config_ir_symbol_t* ordered = &ordered_symbols[i];
    const loom_link_plan_module_symbol_t* selected =
        &selection->symbols.values[ordered->selection_ordinal];
    if (!loom_link_kernel_config_selection_is_partial(selected)) {
      loom_op_t* cloned_op = NULL;
      status = loom_ir_clone_op(&builder, ordered->source_op, &module_remap,
                                &cloned_op);
      continue;
    }
    loom_op_t* declaration = NULL;
    status = loom_link_kernel_config_build_ir_declaration(
        &projection, selected, ordered->source_op, &builder, &declaration);
    if (iree_status_is_ok(status)) {
      status = loom_link_kernel_config_build_ir_helper(
          &projection, ordered->selection_ordinal, ordered->source_op,
          &builder);
    }
  }

  if (!iree_status_is_ok(status)) {
    loom_module_free(target_module);
    return status;
  }
  *out_module = target_module;
  return iree_ok_status();
}

static iree_status_t loom_link_kernel_config_select_complete_symbols(
    const loom_link_plan_module_selection_t* selection,
    iree_arena_allocator_t* arena,
    loom_bytecode_symbol_ordinal_list_t* out_symbols) {
  *out_symbols = (loom_bytecode_symbol_ordinal_list_t){0};
  iree_host_size_t complete_count = 0;
  for (iree_host_size_t i = 0; i < selection->symbols.count; ++i) {
    complete_count += !loom_link_kernel_config_selection_is_partial(
        &selection->symbols.values[i]);
  }
  iree_host_size_t* source_ordinals = NULL;
  if (complete_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, complete_count,
                                                   sizeof(*source_ordinals),
                                                   (void**)&source_ordinals));
  }
  iree_host_size_t complete_ordinal = 0;
  for (iree_host_size_t i = 0; i < selection->symbols.count; ++i) {
    const loom_link_plan_module_symbol_t* symbol =
        &selection->symbols.values[i];
    if (loom_link_kernel_config_selection_is_partial(symbol)) continue;
    IREE_ASSERT_EQ(symbol->materialized_symbol_ordinal, complete_ordinal);
    source_ordinals[complete_ordinal++] =
        symbol->source_symbol->module_symbol_ordinal;
  }
  *out_symbols = (loom_bytecode_symbol_ordinal_list_t){
      .count = complete_count,
      .ordinals = source_ordinals,
  };
  return iree_ok_status();
}

iree_status_t loom_link_plan_project_kernel_config_module(
    const loom_link_plan_t* plan,
    const loom_link_plan_module_selection_t* selection,
    const loom_link_plan_materialization_environment_t* environment,
    iree_string_view_t module_name, iree_arena_allocator_t* arena,
    loom_link_kernel_config_module_projection_t* out_projection) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(selection);
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(environment->context);
  IREE_ASSERT_ARGUMENT(environment->block_pool);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_projection);
  *out_projection = (loom_link_kernel_config_module_projection_t){0};
  if (!loom_link_plan_module_requires_symbol_projection(selection)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "source module has no partial symbol projections");
  }

  for (iree_host_size_t i = 0; i < selection->symbols.count; ++i) {
    if (!loom_link_kernel_config_selection_is_partial(
            &selection->symbols.values[i])) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_link_kernel_config_validate_partial_selection(
        plan, &selection->symbols.values[i]));
  }

  loom_symbol_ref_t* configuration_functions = NULL;
  if (selection->symbols.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, selection->symbols.count, sizeof(*configuration_functions),
        (void**)&configuration_functions));
    for (iree_host_size_t i = 0; i < selection->symbols.count; ++i) {
      configuration_functions[i] = loom_symbol_ref_null();
    }
  }

  if (selection->source_module->materialized_module != NULL) {
    loom_module_t* module = NULL;
    IREE_RETURN_IF_ERROR(loom_link_kernel_config_project_ir_module(
        selection, environment, module_name, arena, configuration_functions,
        &module));
    out_projection->module = module;
    out_projection->configuration_functions.values = configuration_functions;
    out_projection->configuration_functions.count = selection->symbols.count;
    return iree_ok_status();
  }

  const loom_link_module_index_t* index = loom_link_plan_index(plan);
  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_provider_at(
          index, selection->source_module->provider_ordinal);
  IREE_ASSERT(provider != NULL);
  IREE_ASSERT_EQ(provider->kind, LOOM_LINK_PROVIDER_BYTECODE);
  const loom_bytecode_module_metadata_t* source_module =
      &provider->bytecode.metadata
           .modules[selection->source_module->provider_module_ordinal];

  loom_bytecode_symbol_ordinal_list_t complete_symbols = {0};
  IREE_RETURN_IF_ERROR(loom_link_kernel_config_select_complete_symbols(
      selection, arena, &complete_symbols));
  const loom_module_size_hints_t hints = {
      .symbol_count = selection->projected_symbol_count,
  };
  loom_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(loom_module_allocate(environment->context, module_name,
                                            environment->block_pool, &hints,
                                            environment->allocator, &module));

  loom_bytecode_read_options_t read_options = {
      .diagnostic_sink =
          loom_link_kernel_config_diagnostic_sink(environment, provider),
      .low_repr_environment = environment->low_repr_environment,
  };
  loom_bytecode_read_result_t read_result = {0};
  iree_status_t status = loom_link_kernel_config_initialize_output_symbols(
      selection, module, arena, configuration_functions);
  if (iree_status_is_ok(status)) {
    loom_link_kernel_config_bytecode_projection_t projection = {
        .selection = selection,
    };
    const loom_bytecode_source_symbol_resolver_t symbol_resolver = {
        .fn = loom_link_kernel_config_resolve_bytecode_symbol,
        .user_data = &projection,
    };
    status = loom_bytecode_materialize_module_symbols_into(
        provider->bytecode.contents, provider->bytecode.filename,
        environment->block_pool, &provider->bytecode.metadata,
        (uint16_t)selection->source_module->provider_module_ordinal,
        complete_symbols, symbol_resolver, &read_options, &read_result, module,
        environment->allocator);
  }
  if (iree_status_is_ok(status) && read_result.error_count != 0) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel configuration projection emitted %u bytecode error(s)",
        (unsigned)read_result.error_count);
  }

  loom_bytecode_function_projection_reader_t* reader = NULL;
  if (iree_status_is_ok(status)) {
    const loom_bytecode_function_projection_reader_options_t reader_options = {
        .diagnostic_sink =
            loom_link_kernel_config_diagnostic_sink(environment, provider),
        .low_repr_environment = environment->low_repr_environment,
    };
    status = loom_bytecode_function_projection_reader_allocate(
        provider->bytecode.contents, provider->bytecode.filename,
        environment->block_pool, source_module, module, &reader_options,
        environment->allocator, &reader);
  }
  for (iree_host_size_t i = 0;
       i < selection->symbols.count && iree_status_is_ok(status); ++i) {
    const loom_link_plan_module_symbol_t* symbol =
        &selection->symbols.values[i];
    status = loom_bytecode_function_projection_reader_bind_symbol(
        reader, (uint32_t)symbol->source_symbol->module_symbol_ordinal,
        (loom_symbol_ref_t){
            .module_id = 0,
            .symbol_id = (uint16_t)symbol->materialized_symbol_ordinal,
        });
  }
  for (iree_host_size_t i = 0;
       i < selection->symbols.count && iree_status_is_ok(status); ++i) {
    const loom_link_plan_module_symbol_t* symbol =
        &selection->symbols.values[i];
    if (!loom_link_kernel_config_selection_is_partial(symbol)) continue;
    const loom_bytecode_symbol_metadata_t* source_symbol =
        &source_module->symbols[symbol->source_symbol->module_symbol_ordinal];
    if (source_symbol->kernel_workload_region_payload_ordinal_plus_one == 0) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "selected kernel '@%.*s' has no launch-configuration payload",
          (int)source_symbol->name.size, source_symbol->name.data);
      break;
    }
    const loom_link_bytecode_kernel_config_source_t source = {
        .provider = provider,
        .module = source_module,
        .indexed_module = selection->source_module,
        .symbol = source_symbol,
        .symbol_ordinal =
            (uint32_t)symbol->source_symbol->module_symbol_ordinal,
        .config_payload_ordinal =
            source_symbol->kernel_workload_region_payload_ordinal_plus_one - 1,
    };
    status = loom_link_kernel_config_materialize_projection_body(
        &source, reader, arena, module, configuration_functions[i]);
  }
  loom_bytecode_function_projection_reader_free(reader);

  if (!iree_status_is_ok(status)) {
    loom_module_free(module);
    return status;
  }
  out_projection->module = module;
  out_projection->configuration_functions.values = configuration_functions;
  out_projection->configuration_functions.count = selection->symbols.count;
  return iree_ok_status();
}
