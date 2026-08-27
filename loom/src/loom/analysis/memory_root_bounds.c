// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/memory_root_bounds.h"

#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"

static iree_string_view_t loom_memory_root_bounds_op_name(
    const loom_module_t* module, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  return vtable ? loom_op_vtable_name(vtable) : IREE_SV("<unknown>");
}

iree_status_t loom_memory_root_bounds_verify_exact_root(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    loom_value_id_t view_value_id, int64_t static_element_byte_count,
    loom_value_facts_t element_end_facts, bool* out_failed) {
  *out_failed = false;
  const loom_fact_context_t* context = &fact_table->context;
  loom_value_fact_view_reference_t view_reference = {0};
  if (!loom_value_facts_query_view_reference(
          context, loom_value_fact_table_lookup(fact_table, view_value_id),
          &view_reference)) {
    return iree_ok_status();
  }
  loom_value_fact_buffer_reference_t root_reference = {0};
  if (!loom_value_facts_query_buffer_reference(
          context,
          loom_value_fact_table_lookup(fact_table,
                                       view_reference.root_value_id),
          &root_reference)) {
    return iree_ok_status();
  }

  const loom_value_facts_t element_byte_count =
      loom_value_facts_exact_i64(static_element_byte_count);
  loom_value_facts_t view_byte_end = loom_value_facts_unknown();
  loom_value_facts_muli(&element_end_facts, &element_byte_count,
                        &view_byte_end);
  loom_value_facts_t root_byte_end = loom_value_facts_unknown();
  loom_value_facts_addi(&view_reference.base_byte_offset, &view_byte_end,
                        &root_byte_end);
  const loom_value_facts_t root_extent = root_reference.maximum_byte_extent;
  if (loom_value_facts_is_float(root_byte_end) ||
      loom_value_facts_is_float(root_extent) ||
      !loom_value_facts_is_exact(root_extent) ||
      root_byte_end.range_hi <= root_extent.range_hi) {
    return iree_ok_status();
  }

  *out_failed = true;
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_memory_root_bounds_op_name(module, op)),
      loom_param_i64(root_byte_end.range_hi),
      loom_param_i64(root_extent.range_hi),
  };
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_SUBRANGE_026,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}
