// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/vector_memory_mask_bounds.h"

#include "loom/analysis/cfg_value_identity.h"
#include "loom/analysis/condition_fact_scope.h"
#include "loom/analysis/symbolic_expr_proof.h"
#include "loom/ops/vector/ops.h"

struct loom_vector_memory_mask_scope_t {
  // Scalar predicate whose complete derivation is retained below.
  loom_value_id_t condition;

  // True when the lexical scope already proves the predicate false.
  bool definitely_empty;

  // Reusable complete consequence storage, owned by the verifier arena.
  loom_condition_derivation_t derivation;

  // Access-local scope extending the actual lexical parent with derivation.
  loom_condition_fact_scope_t scope;

  // Scope to activate: scope for an unknown predicate, otherwise its parent.
  const loom_condition_fact_scope_t* active_scope;
};

static iree_status_t loom_vector_memory_mask_uniform_scope(
    loom_symbolic_expr_context_t* expression_context,
    const loom_condition_fact_scope_t* lexical_scope, loom_value_id_t condition,
    loom_vector_memory_mask_scope_t** inout_mask_scope, bool* out_changed) {
  loom_vector_memory_mask_scope_t* mask_scope = *inout_mask_scope;
  if (!mask_scope) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        expression_context->arena, sizeof(*mask_scope), (void**)&mask_scope));
    *mask_scope = (loom_vector_memory_mask_scope_t){
        .condition = LOOM_VALUE_ID_INVALID,
    };
    loom_condition_derivation_initialize(expression_context->arena,
                                         &mask_scope->derivation);
    *inout_mask_scope = mask_scope;
  }
  *out_changed = mask_scope->condition != condition ||
                 mask_scope->scope.parent != lexical_scope;
  if (!*out_changed) {
    return iree_ok_status();
  }

  bool truth = false;
  bool proven = false;
  IREE_RETURN_IF_ERROR(loom_condition_fact_scope_proves_condition(
      &expression_context->condition_query, expression_context->fact_table,
      lexical_scope, condition, &truth, &proven));
  mask_scope->condition = condition;
  mask_scope->definitely_empty = proven && !truth;
  mask_scope->active_scope = lexical_scope;
  loom_condition_fact_scope_initialize_local(
      lexical_scope, &mask_scope->derivation, &mask_scope->scope);
  if (!proven) {
    IREE_RETURN_IF_ERROR(loom_condition_facts_query_complete(
        &expression_context->condition_query, expression_context->fact_table,
        condition, /*assumed_truth=*/true, &mask_scope->derivation));
    mask_scope->active_scope = &mask_scope->scope;
  }
  return iree_ok_status();
}

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
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* op = loom_value_def_op(value);
  if (!op || !loom_vector_mask_range_isa(op)) {
    return false;
  }
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
    loom_symbolic_expr_context_t* expression_context,
    const loom_condition_fact_scope_t* lexical_scope,
    loom_value_id_t mask_value_id,
    loom_vector_memory_mask_scope_t** inout_mask_scope,
    loom_vector_memory_mask_bounds_t* out_bounds) {
  *out_bounds = (loom_vector_memory_mask_bounds_t){0};
  const loom_module_t* module = expression_context->module;
  const loom_value_fact_table_t* fact_table = expression_context->fact_table;
  const loom_condition_fact_scope_t* active_scope = lexical_scope;
  bool scope_changed = false;
  if (mask_value_id != LOOM_VALUE_ID_INVALID) {
    mask_value_id = loom_cfg_value_identity_table_lookup(
        expression_context->value_identities, mask_value_id);
    const loom_value_facts_t mask_facts =
        loom_value_fact_table_lookup(fact_table, mask_value_id);
    out_bounds->definitely_empty = loom_vector_memory_mask_facts_all_false(
        &fact_table->context, mask_facts);
    loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
    if (!out_bounds->definitely_empty &&
        loom_value_fact_table_query_uniform_element_origin(
            fact_table, module, mask_value_id, &condition)) {
      condition = loom_cfg_value_identity_table_lookup(
          expression_context->value_identities, condition);
      IREE_RETURN_IF_ERROR(loom_vector_memory_mask_uniform_scope(
          expression_context, lexical_scope, condition, inout_mask_scope,
          &scope_changed));
      active_scope = (*inout_mask_scope)->active_scope;
      out_bounds->definitely_empty = (*inout_mask_scope)->definitely_empty;
    }
  }
  // A reused scope has a stable address even when its derivation changes.
  if (expression_context->condition_scope != active_scope || scope_changed) {
    expression_context->condition_scope = active_scope;
    loom_symbolic_expr_context_reset(expression_context);
  }
  if (mask_value_id == LOOM_VALUE_ID_INVALID || out_bounds->definitely_empty) {
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

  const loom_value_id_t lower = loom_vector_mask_range_lower_bound(mask_op);
  const loom_scalar_type_t coordinate_type =
      loom_type_element_type(loom_module_value_type(module, lower));
  if (coordinate_type != LOOM_SCALAR_TYPE_INDEX) {
    const loom_value_fact_vector_iota_t coordinates = {
        .base = loom_value_fact_table_lookup(fact_table, lower),
        .step = step_facts,
        .bit_count = (uint8_t)loom_scalar_type_bitwidth(coordinate_type),
    };
    int64_t minimum = 0;
    int64_t maximum = 0;
    if (!loom_value_fact_vector_iota_bounds(
            coordinates,
            loom_value_fact_table_maximum_element_count(
                fact_table, loom_module_value_type(module, mask_value_id)),
            &minimum, &maximum)) {
      return iree_ok_status();
    }
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
  if (!bounds->has_unit_range) {
    return iree_ok_status();
  }

  bool lower_le_origin = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_mask_prove_le(
      expression_context, &bounds->lower_bound, origin, &lower_le_origin));
  if (!lower_le_origin) {
    return iree_ok_status();
  }
  bool origin_le_lower = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_mask_prove_le(
      expression_context, origin, &bounds->lower_bound, &origin_le_lower));
  if (!origin_le_lower) {
    return iree_ok_status();
  }

  *out_end = bounds->upper_bound;
  *out_known = true;
  return iree_ok_status();
}
