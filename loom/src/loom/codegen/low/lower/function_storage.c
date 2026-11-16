// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/function_storage.h"

#include "loom/codegen/low/lower/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/view/ops.h"

iree_status_t loom_low_lower_function_storage_check_lifetime(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_supported) {
  *out_supported = true;
  if (!loom_value_fact_table_block_may_repeat(
          loom_low_lower_context_fact_table(context),
          source_op->parent_block)) {
    return iree_ok_status();
  }
  loom_storage_interference_t* interference = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_storage_interference(context, &interference));
  const loom_value_id_t root_value = loom_op_const_results(source_op)[0];
  *out_supported = loom_storage_interference_root_has_single_live_instance(
      interference, root_value);
  if (*out_supported) {
    return iree_ok_status();
  }
  return loom_low_lower_emit_function_storage_lifetime_unsupported(
      context, source_op, root_value);
}

iree_status_t loom_low_lower_function_storage_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_selected,
    const loom_low_lower_function_storage_plan_t** out_plan) {
  *out_selected = false;
  *out_plan = NULL;
  const bool is_view = loom_view_alloca_isa(source_op);
  const loom_value_id_t result = loom_op_const_results(source_op)[0];
  const loom_type_t result_type =
      loom_module_value_type(context->module, result);
  const loom_low_lower_function_storage_source_t source =
      !is_view ? LOOM_LOW_LOWER_FUNCTION_STORAGE_SOURCE_BUFFER
      : loom_type_has_reference_element(result_type)
          ? LOOM_LOW_LOWER_FUNCTION_STORAGE_SOURCE_REFERENCE_VIEW
          : LOOM_LOW_LOWER_FUNCTION_STORAGE_SOURCE_SCALAR_VIEW;
  const loom_value_id_t pool = is_view ? loom_view_alloca_pool(source_op)
                                       : loom_buffer_alloca_pool(source_op);
  if (pool != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  const loom_low_lower_function_storage_config_t* config =
      &context->policy->function_storage;
  const loom_value_fact_memory_space_t memory_space =
      is_view ? loom_view_alloca_memory_space(source_op)
              : loom_buffer_alloca_memory_space(source_op);
  const loom_low_lower_function_storage_mapping_t* mapping = NULL;
  for (iree_host_size_t i = 0; i < config->count; ++i) {
    if (config->mappings[i].memory_space == memory_space &&
        config->mappings[i].source == source) {
      mapping = &config->mappings[i];
      break;
    }
  }
  if (!mapping) {
    return iree_ok_status();
  }
  *out_selected = true;

  bool lifetime_supported = false;
  IREE_RETURN_IF_ERROR(loom_low_lower_function_storage_check_lifetime(
      context, source_op, &lifetime_supported));
  if (!lifetime_supported) {
    return iree_ok_status();
  }

  const loom_value_fact_table_t* facts =
      loom_low_lower_context_fact_table(context);
  loom_value_facts_t extent;
  int64_t alignment;
  loom_value_id_t extent_value;
  if (is_view) {
    loom_value_fact_view_reference_t reference;
    loom_value_facts_query_view_reference(
        &facts->context, loom_value_fact_table_lookup(facts, result),
        &reference);
    extent = reference.footprint_byte_length;
    alignment = (int64_t)reference.root_minimum_alignment;
    extent_value = result;
    if (source == LOOM_LOW_LOWER_FUNCTION_STORAGE_SOURCE_REFERENCE_VIEW) {
      alignment = iree_max(alignment, reference.static_element_byte_count);
    }
  } else {
    extent_value = loom_buffer_alloca_byte_length(source_op);
    extent = loom_value_fact_table_lookup(facts, extent_value);
    alignment = loom_buffer_alloca_base_alignment(source_op);
  }
  int64_t byte_length = 0;
  if (!loom_value_facts_as_non_negative_i64_maximum(extent, &byte_length)) {
    return loom_low_lower_emit_function_storage_extent_unsupported(
        context, source_op, mapping->storage_space, extent_value);
  }
  loom_low_lower_function_storage_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  *plan = (loom_low_lower_function_storage_plan_t){
      .byte_length = byte_length,
      .byte_alignment = alignment,
      .mapping = mapping,
  };
  loom_type_t address_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, mapping->address_register_class, 1, &address_type));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_plan_value_type(context, result, address_type));
  *out_plan = plan;
  return iree_ok_status();
}

iree_status_t loom_low_lower_function_storage_emit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_function_storage_plan_t* plan) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_value_id_t result = loom_op_const_results(source_op)[0];
  loom_op_t* storage_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_reserve_build(
      builder, plan->byte_length, plan->byte_alignment,
      loom_type_storage(plan->mapping->storage_space), source_op->location,
      &storage_op));
  loom_op_t* address_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_address_build(
      builder, loom_low_storage_reserve_storage(storage_op), /*offset=*/0,
      loom_low_lower_value_binding_type(context, result), source_op->location,
      &address_op));
  const loom_value_id_t address = loom_low_storage_address_result(address_op);
  if (plan->mapping->initialize) {
    IREE_RETURN_IF_ERROR(plan->mapping->initialize(
        context, address, plan->byte_length, source_op->location));
  }
  return loom_low_lower_bind_value(context, result, address);
}
