// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/vector_memory_mask_bounds.h"

#include "loom/analysis/symbolic_expr_proof.h"
#include "loom/ops/vector/ops.h"

static bool loom_vector_memory_mask_facts_exact_i64(loom_value_facts_t facts,
                                                    int64_t* out_value) {
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

static bool loom_vector_memory_mask_facts_all_false(
    const loom_fact_context_t* context, loom_value_facts_t facts) {
  loom_value_fact_uniform_element_t uniform = {0};
  if (loom_value_facts_query_uniform_element(context, facts, &uniform)) {
    int64_t value = 0;
    return loom_vector_memory_mask_facts_exact_i64(uniform.element, &value) &&
           value == 0;
  }

  loom_value_fact_small_static_lanes_t lanes = {0};
  if (!loom_value_facts_query_small_static_lanes(context, facts, &lanes)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < lanes.count; ++i) {
    int64_t value = 0;
    if (!loom_vector_memory_mask_facts_exact_i64(lanes.lanes[i], &value) ||
        value != 0) {
      return false;
    }
  }
  return true;
}

static bool loom_vector_memory_mask_range_op(const loom_module_t* module,
                                             loom_value_id_t mask_value_id,
                                             const loom_op_t** out_op) {
  const loom_value_t* value = loom_module_value(module, mask_value_id);
  if (loom_value_is_block_arg(value)) return false;
  const loom_op_t* op = loom_value_def_op(value);
  if (!op || !loom_vector_mask_range_isa(op)) return false;
  *out_op = op;
  return true;
}

static iree_status_t loom_vector_memory_mask_prove_le(
    loom_symbolic_expr_context_t* expression_context,
    const loom_symbolic_expr_t* left, const loom_symbolic_expr_t* right,
    bool* out_proven) {
  loom_symbolic_proof_result_t result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_prove_le(expression_context, left, right, &result));
  *out_proven = result == LOOM_SYMBOLIC_PROOF_TRUE;
  return iree_ok_status();
}

iree_status_t loom_vector_memory_mask_bounds_analyze(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_symbolic_expr_context_t* expression_context,
    loom_value_id_t mask_value_id,
    loom_vector_memory_mask_bounds_t* out_bounds) {
  *out_bounds = (loom_vector_memory_mask_bounds_t){0};
  const loom_value_facts_t mask_facts =
      loom_value_fact_table_lookup(fact_table, mask_value_id);
  if (loom_vector_memory_mask_facts_all_false(&fact_table->context,
                                              mask_facts)) {
    out_bounds->definitely_empty = true;
    return iree_ok_status();
  }

  const loom_op_t* mask_op = NULL;
  if (!loom_vector_memory_mask_range_op(module, mask_value_id, &mask_op)) {
    return iree_ok_status();
  }
  const loom_value_facts_t step_facts = loom_value_fact_table_lookup(
      fact_table, loom_vector_mask_range_step(mask_op));
  int64_t step = 0;
  if (!loom_vector_memory_mask_facts_exact_i64(step_facts, &step) ||
      step != 1) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
      expression_context, loom_vector_mask_range_lower_bound(mask_op),
      &out_bounds->lower_bound));
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
      expression_context, loom_vector_mask_range_upper_bound(mask_op),
      &out_bounds->upper_bound));
  out_bounds->has_unit_range = true;
  return loom_vector_memory_mask_prove_le(
      expression_context, &out_bounds->upper_bound, &out_bounds->lower_bound,
      &out_bounds->definitely_empty);
}

iree_status_t loom_vector_memory_mask_bounds_tail_end(
    loom_symbolic_expr_context_t* expression_context,
    const loom_vector_memory_mask_bounds_t* bounds,
    const loom_symbolic_expr_t* origin, loom_symbolic_expr_t* out_end,
    bool* out_known) {
  *out_known = false;
  if (!bounds->has_unit_range) return iree_ok_status();

  bool lower_le_origin = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_mask_prove_le(
      expression_context, &bounds->lower_bound, origin, &lower_le_origin));
  if (!lower_le_origin) return iree_ok_status();
  bool origin_le_lower = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_mask_prove_le(
      expression_context, origin, &bounds->lower_bound, &origin_le_lower));
  if (!origin_le_lower) return iree_ok_status();

  *out_end = bounds->upper_bound;
  *out_known = true;
  return iree_ok_status();
}
