// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cleanup/integer_division.h"

#include "loom/ir/module.h"
#include "loom/ir/structural_hash.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/cleanup/expression_scope.h"

typedef enum loom_integer_division_family_e {
  LOOM_INTEGER_DIVISION_FAMILY_SCALAR_SIGNED = 0,
  LOOM_INTEGER_DIVISION_FAMILY_SCALAR_UNSIGNED = 1,
  LOOM_INTEGER_DIVISION_FAMILY_INDEX = 2,
  LOOM_INTEGER_DIVISION_FAMILY_VECTOR_SIGNED = 3,
  LOOM_INTEGER_DIVISION_FAMILY_VECTOR_UNSIGNED = 4,
} loom_integer_division_family_t;

typedef enum loom_integer_division_role_e {
  LOOM_INTEGER_DIVISION_ROLE_QUOTIENT = 0,
  LOOM_INTEGER_DIVISION_ROLE_REMAINDER = 1,
} loom_integer_division_role_t;

typedef struct loom_integer_division_plan_t loom_integer_division_plan_t;

typedef struct loom_integer_division_candidate_t {
  // Standalone quotient or remainder operation.
  loom_op_t* op;
  // The operation's only result.
  loom_value_id_t result;
  // Semantic result selected by the operation.
  loom_integer_division_role_t role;
  // Earlier dominating candidate with the same role and key.
  struct loom_integer_division_candidate_t* previous_same_role;
  // Selected combined plan, or NULL while unmatched.
  loom_integer_division_plan_t* plan;
  // Next candidate retired by the selected plan.
  struct loom_integer_division_candidate_t* plan_next;
} loom_integer_division_candidate_t;

struct loom_integer_division_plan_t {
  // Next plan across all function regions.
  loom_integer_division_plan_t* next;
  // Operation before which the combined operation is inserted.
  loom_op_t* anchor;
  // First standalone operation retired by this plan.
  loom_integer_division_candidate_t* first_candidate;
  // Last standalone operation retired by this plan.
  loom_integer_division_candidate_t* last_candidate;
  // Preferred source name for the quotient result.
  loom_value_id_t quotient_name_source;
  // Preferred source name for the remainder result.
  loom_value_id_t remainder_name_source;
  // Combined operation family.
  loom_integer_division_family_t family;
  // Number of standalone operations retired by this plan.
  uint32_t candidate_count;
};

typedef struct loom_integer_division_planner_t {
  // Module containing the immutable function snapshot.
  loom_module_t* module;
  // Retained canonical value facts.
  const loom_value_fact_table_t* facts;
  // Pass arena retaining plans and candidates through application.
  iree_arena_allocator_t* arena;
  // Candidate records for the complete function fact scope.
  loom_integer_division_candidate_t* candidates;
  // Number of populated candidate records.
  uint32_t candidate_count;
  // Maximum candidate records derived from the retained fact index.
  uint32_t candidate_capacity;
  // Selected plan storage for the complete function.
  loom_integer_division_plan_t* plans;
  // Number of populated plans.
  uint32_t plan_count;
  // First selected plan across function regions.
  loom_integer_division_plan_t* first_plan;
  // Last selected plan across function regions.
  loom_integer_division_plan_t* last_plan;
} loom_integer_division_planner_t;

typedef struct loom_integer_division_region_planner_t {
  // Function-wide plan owner.
  loom_integer_division_planner_t* planner;
  // Shared dominance-scoped availability walk.
  loom_expression_walk_t* expression_walk;
} loom_integer_division_region_planner_t;

static bool loom_integer_division_match(
    const loom_op_t* op, loom_integer_division_family_t* out_family,
    loom_integer_division_role_t* out_role) {
  switch (op->kind) {
    case LOOM_OP_SCALAR_DIVSI:
      *out_family = LOOM_INTEGER_DIVISION_FAMILY_SCALAR_SIGNED;
      *out_role = LOOM_INTEGER_DIVISION_ROLE_QUOTIENT;
      return true;
    case LOOM_OP_SCALAR_REMSI:
      *out_family = LOOM_INTEGER_DIVISION_FAMILY_SCALAR_SIGNED;
      *out_role = LOOM_INTEGER_DIVISION_ROLE_REMAINDER;
      return true;
    case LOOM_OP_SCALAR_DIVUI:
      *out_family = LOOM_INTEGER_DIVISION_FAMILY_SCALAR_UNSIGNED;
      *out_role = LOOM_INTEGER_DIVISION_ROLE_QUOTIENT;
      return true;
    case LOOM_OP_SCALAR_REMUI:
      *out_family = LOOM_INTEGER_DIVISION_FAMILY_SCALAR_UNSIGNED;
      *out_role = LOOM_INTEGER_DIVISION_ROLE_REMAINDER;
      return true;
    case LOOM_OP_INDEX_DIV:
      *out_family = LOOM_INTEGER_DIVISION_FAMILY_INDEX;
      *out_role = LOOM_INTEGER_DIVISION_ROLE_QUOTIENT;
      return true;
    case LOOM_OP_INDEX_REM:
      *out_family = LOOM_INTEGER_DIVISION_FAMILY_INDEX;
      *out_role = LOOM_INTEGER_DIVISION_ROLE_REMAINDER;
      return true;
    case LOOM_OP_VECTOR_DIVSI:
      *out_family = LOOM_INTEGER_DIVISION_FAMILY_VECTOR_SIGNED;
      *out_role = LOOM_INTEGER_DIVISION_ROLE_QUOTIENT;
      return true;
    case LOOM_OP_VECTOR_REMSI:
      *out_family = LOOM_INTEGER_DIVISION_FAMILY_VECTOR_SIGNED;
      *out_role = LOOM_INTEGER_DIVISION_ROLE_REMAINDER;
      return true;
    case LOOM_OP_VECTOR_DIVUI:
      *out_family = LOOM_INTEGER_DIVISION_FAMILY_VECTOR_UNSIGNED;
      *out_role = LOOM_INTEGER_DIVISION_ROLE_QUOTIENT;
      return true;
    case LOOM_OP_VECTOR_REMUI:
      *out_family = LOOM_INTEGER_DIVISION_FAMILY_VECTOR_UNSIGNED;
      *out_role = LOOM_INTEGER_DIVISION_ROLE_REMAINDER;
      return true;
    default:
      return false;
  }
}

static bool loom_integer_division_family_is_signed(
    loom_integer_division_family_t family) {
  return family == LOOM_INTEGER_DIVISION_FAMILY_SCALAR_SIGNED ||
         family == LOOM_INTEGER_DIVISION_FAMILY_VECTOR_SIGNED;
}

// Returns true when facts prove that an exact unsigned division produces only
// zero or one. Keeping the quotient and remainder independent preserves their
// cheaper compare and conditional-subtract forms; combining them would instead
// introduce a multiply solely to reconstruct the remainder.
static bool loom_integer_division_has_boolean_unsigned_quotient(
    const loom_module_t* module, const loom_value_fact_table_t* facts,
    const loom_op_t* op, loom_integer_division_family_t family) {
  if (loom_integer_division_family_is_signed(family)) {
    return false;
  }

  const loom_value_id_t* operands = loom_op_const_operands(op);
  int64_t raw_divisor = 0;
  if (!loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(facts, operands[1]), &raw_divisor)) {
    return false;
  }

  uint64_t divisor = 0;
  uint64_t dividend_maximum = 0;
  const loom_value_facts_t dividend_facts =
      loom_value_fact_table_lookup(facts, operands[0]);
  if (family == LOOM_INTEGER_DIVISION_FAMILY_INDEX) {
    if (raw_divisor <= 0) {
      return false;
    }
    divisor = (uint64_t)raw_divisor;
    int64_t exact_dividend = 0;
    int64_t signed_dividend_maximum = 0;
    if (loom_value_facts_as_exact_i64(dividend_facts, &exact_dividend) &&
        exact_dividend >= 0) {
      dividend_maximum = (uint64_t)exact_dividend;
    } else if (loom_value_facts_as_non_negative_i64_maximum(
                   dividend_facts, &signed_dividend_maximum)) {
      dividend_maximum = (uint64_t)signed_dividend_maximum;
    } else {
      return false;
    }
  } else {
    const loom_type_t result_type =
        loom_module_value_type(module, loom_op_const_results(op)[0]);
    const int32_t bit_count =
        loom_scalar_type_bitwidth(loom_type_element_type(result_type));
    if (bit_count <= 0 || bit_count > 64) {
      return false;
    }
    const uint64_t value_mask =
        bit_count == 64 ? UINT64_MAX : (UINT64_C(1) << bit_count) - 1;
    divisor = (uint64_t)raw_divisor & value_mask;
    dividend_maximum = value_mask;

    int64_t exact_dividend = 0;
    int64_t signed_dividend_maximum = 0;
    if (loom_value_facts_as_exact_i64(dividend_facts, &exact_dividend)) {
      dividend_maximum = (uint64_t)exact_dividend & value_mask;
    } else if (loom_value_facts_as_non_negative_i64_maximum(
                   dividend_facts, &signed_dividend_maximum)) {
      dividend_maximum = (uint64_t)signed_dividend_maximum;
    }
  }
  return divisor != 0 && divisor > dividend_maximum / 2;
}

static bool loom_integer_division_should_pair(
    const loom_module_t* module, const loom_value_fact_table_t* facts,
    const loom_op_t* op, loom_integer_division_family_t family) {
  const loom_value_id_t divisor = loom_op_const_operands(op)[1];
  const bool divisor_is_exact =
      loom_value_facts_is_exact(loom_value_fact_table_lookup(facts, divisor));
  // Exact index quotient/remainder pairs commonly encode coordinate digit
  // decomposition. Keeping those semantic operations independent preserves
  // the relationship consumed by topology, address, and bank-service analysis.
  // Dynamic index divisors cannot describe that static coordinate structure
  // and remain eligible for shared execution.
  if (divisor_is_exact && family == LOOM_INTEGER_DIVISION_FAMILY_INDEX) {
    return false;
  }
  return !divisor_is_exact ||
         !loom_integer_division_has_boolean_unsigned_quotient(module, facts, op,
                                                              family);
}

// Counts standalone division results directly from the retained function-fact
// index. Ordinary functions never construct a structural or dominance walk.
static uint32_t loom_integer_division_candidate_count(
    const loom_module_t* module, const loom_value_fact_table_t* facts) {
  uint32_t candidate_count = 0;
  loom_value_fact_scope_iterator_t iterator;
  loom_value_fact_table_scope_begin(facts, &iterator);
  for (loom_value_id_t value_id = loom_value_fact_scope_next(&iterator);
       value_id != LOOM_VALUE_ID_INVALID;
       value_id = loom_value_fact_scope_next(&iterator)) {
    const loom_value_t* value = loom_module_value(module, value_id);
    if (loom_value_is_block_arg(value)) {
      continue;
    }
    const loom_op_t* op = loom_value_def_op(value);
    if (!op || iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
      continue;
    }
    loom_integer_division_family_t family;
    loom_integer_division_role_t role;
    if (!loom_integer_division_match(op, &family, &role)) {
      continue;
    }
    if (!loom_integer_division_should_pair(module, facts, op, family)) {
      continue;
    }
    ++candidate_count;
  }
  return candidate_count;
}

static void loom_integer_division_identity(
    const loom_integer_division_planner_t* planner, const loom_op_t* op,
    loom_integer_division_family_t* out_family, loom_value_id_t* out_lhs,
    loom_value_id_t* out_rhs, loom_type_t* out_type) {
  loom_integer_division_role_t role;
  const bool matched = loom_integer_division_match(op, out_family, &role);
  IREE_ASSERT(matched);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  *out_lhs = loom_value_fact_table_query_identity(planner->facts, operands[0]);
  *out_rhs = loom_value_fact_table_query_identity(planner->facts, operands[1]);
  *out_type =
      loom_module_value_type(planner->module, loom_op_const_results(op)[0]);
}

static uint32_t loom_integer_division_hash(
    const loom_integer_division_planner_t* planner, const loom_op_t* op) {
  loom_integer_division_family_t family;
  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  loom_type_t type = loom_type_none();
  loom_integer_division_identity(planner, op, &family, &lhs, &rhs, &type);
  uint32_t hash = loom_structural_hash_initialize();
  hash = loom_structural_hash_mix_u8(hash, (uint8_t)family);
  hash = loom_structural_hash_mix_u32(hash, lhs);
  hash = loom_structural_hash_mix_u32(hash, rhs);
  hash = loom_structural_hash_mix_u32(hash, loom_type_hash(type));
  return loom_structural_hash_finalize(hash);
}

static bool loom_integer_division_equal(void* user_data,
                                        const loom_module_t* module,
                                        const loom_op_t* lhs_op,
                                        const loom_op_t* rhs_op) {
  (void)module;
  const loom_integer_division_planner_t* planner =
      (const loom_integer_division_planner_t*)user_data;
  loom_integer_division_family_t lhs_family;
  loom_integer_division_family_t rhs_family;
  loom_value_id_t lhs_lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t lhs_rhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs_lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs_rhs = LOOM_VALUE_ID_INVALID;
  loom_type_t lhs_type = loom_type_none();
  loom_type_t rhs_type = loom_type_none();
  loom_integer_division_identity(planner, lhs_op, &lhs_family, &lhs_lhs,
                                 &lhs_rhs, &lhs_type);
  loom_integer_division_identity(planner, rhs_op, &rhs_family, &rhs_lhs,
                                 &rhs_rhs, &rhs_type);
  return lhs_family == rhs_family && lhs_lhs == rhs_lhs && lhs_rhs == rhs_rhs &&
         loom_type_equal(lhs_type, rhs_type);
}

static loom_integer_division_candidate_t*
loom_integer_division_oldest_candidate(
    loom_integer_division_candidate_t* candidate) {
  while (candidate->previous_same_role) {
    candidate = candidate->previous_same_role;
  }
  return candidate;
}

static bool loom_integer_division_candidate_uses_dominated(
    const loom_integer_division_region_planner_t* planner,
    const loom_integer_division_candidate_t* candidate,
    const loom_op_t* anchor) {
  const loom_value_t* value =
      loom_module_value(planner->planner->module, candidate->result);
  if (loom_value_has_attribute_uses(value) ||
      loom_module_value_has_type_uses(planner->planner->module,
                                      candidate->result)) {
    return false;
  }
  const loom_dominance_info_t* dominance =
      loom_expression_walk_dominance_info(planner->expression_walk);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    if (!loom_dominates_op(dominance, anchor, loom_use_user_op(*use))) {
      return false;
    }
  }
  return true;
}

static bool loom_integer_division_chain_uses_dominated(
    const loom_integer_division_region_planner_t* planner,
    const loom_integer_division_candidate_t* candidate,
    const loom_op_t* anchor) {
  for (; candidate; candidate = candidate->previous_same_role) {
    if (!loom_integer_division_candidate_uses_dominated(planner, candidate,
                                                        anchor)) {
      return false;
    }
  }
  return true;
}

static void loom_integer_division_plan_append(
    loom_integer_division_plan_t* plan,
    loom_integer_division_candidate_t* candidate) {
  candidate->plan = plan;
  candidate->plan_next = NULL;
  if (plan->last_candidate) {
    plan->last_candidate->plan_next = candidate;
  } else {
    plan->first_candidate = candidate;
  }
  plan->last_candidate = candidate;
  ++plan->candidate_count;
}

static void loom_integer_division_plan_append_chain(
    loom_integer_division_plan_t* plan,
    loom_integer_division_candidate_t* candidate) {
  for (; candidate; candidate = candidate->previous_same_role) {
    loom_integer_division_plan_append(plan, candidate);
  }
}

static loom_integer_division_plan_t* loom_integer_division_create_plan(
    loom_integer_division_region_planner_t* planner,
    loom_integer_division_family_t family,
    loom_integer_division_candidate_t* quotient,
    loom_integer_division_candidate_t* remainder, loom_op_t* anchor) {
  loom_integer_division_planner_t* function_planner = planner->planner;
  loom_integer_division_plan_t* plan =
      &function_planner->plans[function_planner->plan_count++];
  *plan = (loom_integer_division_plan_t){
      .anchor = anchor,
      .quotient_name_source =
          loom_integer_division_oldest_candidate(quotient)->result,
      .remainder_name_source =
          loom_integer_division_oldest_candidate(remainder)->result,
      .family = family,
  };
  loom_integer_division_plan_append_chain(plan, quotient);
  loom_integer_division_plan_append_chain(plan, remainder);
  if (function_planner->last_plan) {
    function_planner->last_plan->next = plan;
  } else {
    function_planner->first_plan = plan;
  }
  function_planner->last_plan = plan;
  return plan;
}

static iree_status_t loom_integer_division_observe_candidate(
    loom_integer_division_region_planner_t* planner,
    const loom_expression_cursor_t* cursor,
    loom_integer_division_family_t family, loom_integer_division_role_t role) {
  loom_op_t* op = cursor->op;
  loom_integer_division_planner_t* function_planner = planner->planner;
  if (!loom_integer_division_should_pair(function_planner->module,
                                         function_planner->facts, op, family)) {
    return iree_ok_status();
  }
  const uint32_t candidate_index = function_planner->candidate_count++;
  IREE_ASSERT(candidate_index < function_planner->candidate_capacity);
  loom_integer_division_candidate_t* candidate =
      &function_planner->candidates[candidate_index];
  *candidate = (loom_integer_division_candidate_t){
      .op = op,
      .result = loom_op_const_results(op)[0],
      .role = role,
  };
  const uint32_t hash = loom_integer_division_hash(function_planner, op);
  const loom_expression_probe_t probe =
      loom_expression_scope_probe(cursor, hash, 0);
  loom_expression_entry_t* existing_entry = probe.entry;
  loom_integer_division_candidate_t* existing =
      existing_entry ? &function_planner->candidates[existing_entry->metadata]
                     : NULL;
  const loom_expression_entry_t entry = {
      .op = op,
      .hash = hash,
      .metadata = candidate_index,
  };
  if (!existing) {
    loom_expression_scope_insert(cursor, probe, entry);
    return iree_ok_status();
  }
  if (existing->plan) {
    if (loom_dominates_op(
            loom_expression_walk_dominance_info(planner->expression_walk),
            existing->plan->anchor, op)) {
      loom_integer_division_plan_append(existing->plan, candidate);
    }
    loom_expression_scope_insert(cursor, probe, entry);
    return iree_ok_status();
  }
  if (existing->role == role) {
    candidate->previous_same_role = existing;
    loom_expression_scope_insert(cursor, probe, entry);
    return iree_ok_status();
  }

  loom_integer_division_candidate_t* quotient =
      role == LOOM_INTEGER_DIVISION_ROLE_QUOTIENT ? candidate : existing;
  loom_integer_division_candidate_t* remainder =
      role == LOOM_INTEGER_DIVISION_ROLE_REMAINDER ? candidate : existing;
  loom_op_t* anchor = NULL;
  if (loom_integer_division_family_is_signed(family)) {
    anchor = loom_integer_division_oldest_candidate(quotient)->op;
    if (!loom_dominates_op(
            loom_expression_walk_dominance_info(planner->expression_walk),
            anchor, loom_integer_division_oldest_candidate(remainder)->op) &&
        !loom_integer_division_chain_uses_dominated(planner, remainder,
                                                    anchor)) {
      // The quotient's stricter overflow domain cannot move above an earlier
      // signed remainder. Shadow those remainders with the quotient so a later
      // remainder can still combine at the quotient's original position.
      loom_expression_scope_insert(cursor, probe, entry);
      return iree_ok_status();
    }
  } else {
    anchor = loom_integer_division_oldest_candidate(existing)->op;
  }
  loom_integer_division_create_plan(planner, family, quotient, remainder,
                                    anchor);
  loom_expression_scope_insert(cursor, probe, entry);
  return iree_ok_status();
}

static iree_status_t loom_integer_division_plan_region_scratch(
    loom_integer_division_region_planner_t* region_planner,
    loom_region_t* region, iree_arena_allocator_t* arena) {
  const loom_expression_identity_policy_t identity_policy = {
      .user_data = region_planner->planner,
      .equal = loom_integer_division_equal,
  };
  IREE_RETURN_IF_ERROR(loom_expression_walk_allocate(
      region_planner->planner->module, region, &identity_policy,
      region_planner->planner->candidate_capacity, arena,
      &region_planner->expression_walk));
  while (true) {
    loom_expression_cursor_t cursor;
    IREE_RETURN_IF_ERROR(
        loom_expression_walk_next(region_planner->expression_walk, &cursor));
    if (!cursor.op) {
      break;
    }
    loom_integer_division_family_t family;
    loom_integer_division_role_t role;
    if (loom_integer_division_match(cursor.op, &family, &role)) {
      IREE_RETURN_IF_ERROR(loom_integer_division_observe_candidate(
          region_planner, &cursor, family, role));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_integer_division_plan_region(
    loom_integer_division_planner_t* planner, loom_region_t* region) {
  loom_integer_division_region_planner_t region_planner = {
      .planner = planner,
  };
  const iree_arena_checkpoint_t scratch_checkpoint =
      iree_arena_checkpoint_save(planner->arena);
  const iree_status_t status = loom_integer_division_plan_region_scratch(
      &region_planner, region, planner->arena);
  iree_arena_checkpoint_restore(&scratch_checkpoint);
  return status;
}

static iree_status_t loom_integer_division_build_combined(
    loom_rewriter_t* rewriter, loom_integer_division_plan_t* plan,
    loom_op_t** out_op) {
  const loom_value_id_t* operands = loom_op_const_operands(plan->anchor);
  const loom_type_t result_type = loom_module_value_type(
      rewriter->module, loom_op_const_results(plan->anchor)[0]);
  const loom_type_t result_types[2] = {result_type, result_type};
  loom_builder_set_before(&rewriter->builder, plan->anchor);
  switch (plan->family) {
    case LOOM_INTEGER_DIVISION_FAMILY_SCALAR_SIGNED:
      return loom_scalar_divremsi_build(&rewriter->builder, operands[0],
                                        operands[1], result_types,
                                        plan->anchor->location, out_op);
    case LOOM_INTEGER_DIVISION_FAMILY_SCALAR_UNSIGNED:
      return loom_scalar_divremui_build(&rewriter->builder, operands[0],
                                        operands[1], result_types,
                                        plan->anchor->location, out_op);
    case LOOM_INTEGER_DIVISION_FAMILY_INDEX:
      return loom_index_divrem_build(&rewriter->builder, operands[0],
                                     operands[1], plan->anchor->location,
                                     out_op);
    case LOOM_INTEGER_DIVISION_FAMILY_VECTOR_SIGNED:
      return loom_vector_divremsi_build(&rewriter->builder, operands[0],
                                        operands[1], result_types,
                                        plan->anchor->location, out_op);
    case LOOM_INTEGER_DIVISION_FAMILY_VECTOR_UNSIGNED:
      return loom_vector_divremui_build(&rewriter->builder, operands[0],
                                        operands[1], result_types,
                                        plan->anchor->location, out_op);
  }
  IREE_ASSERT_UNREACHABLE("invalid integer division family");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_integer_division_apply_plan(
    loom_rewriter_t* rewriter, loom_integer_division_plan_t* plan) {
  loom_op_t* combined = NULL;
  IREE_RETURN_IF_ERROR(
      loom_integer_division_build_combined(rewriter, plan, &combined));
  const loom_value_id_t* results = loom_op_const_results(combined);
  IREE_RETURN_IF_ERROR(loom_rewriter_copy_value_name(
      rewriter, plan->quotient_name_source, results[0]));
  IREE_RETURN_IF_ERROR(loom_rewriter_copy_value_name(
      rewriter, plan->remainder_name_source, results[1]));
  for (loom_integer_division_candidate_t* candidate = plan->first_candidate;
       candidate; candidate = candidate->plan_next) {
    const loom_value_id_t replacement =
        results[candidate->role == LOOM_INTEGER_DIVISION_ROLE_REMAINDER];
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        rewriter, candidate->result, replacement));
  }
  for (loom_integer_division_candidate_t* candidate = plan->first_candidate;
       candidate; candidate = candidate->plan_next) {
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, candidate->op));
  }
  return iree_ok_status();
}

iree_status_t loom_integer_division_combine(
    loom_module_t* module, loom_func_like_t function,
    const loom_value_fact_table_t* facts, iree_arena_allocator_t* parent_arena,
    loom_integer_division_combine_result_t* out_result) {
  *out_result = (loom_integer_division_combine_result_t){0};
  if (!loom_func_like_body(function)) {
    return iree_ok_status();
  }
  const uint32_t candidate_capacity =
      loom_integer_division_candidate_count(module, facts);
  if (candidate_capacity < 2) {
    return iree_ok_status();
  }

  const iree_arena_checkpoint_t planner_checkpoint =
      iree_arena_checkpoint_save(parent_arena);
  loom_integer_division_planner_t planner = {
      .module = module,
      .facts = facts,
      .arena = parent_arena,
      .candidate_capacity = candidate_capacity,
  };
  iree_status_t status = iree_arena_allocate_array(
      parent_arena, candidate_capacity, sizeof(*planner.candidates),
      (void**)&planner.candidates);
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(parent_arena, candidate_capacity / 2,
                                       sizeof(*planner.plans),
                                       (void**)&planner.plans);
  }
  for (uint8_t i = 0;
       i < loom_func_like_region_count(function) && iree_status_is_ok(status);
       ++i) {
    loom_region_t* region = loom_func_like_region(function, i);
    if (!region) {
      continue;
    }
    status = loom_integer_division_plan_region(&planner, region);
  }

  loom_rewriter_t rewriter;
  bool rewriter_initialized = false;
  if (iree_status_is_ok(status) && planner.first_plan) {
    loom_rewriter_initialize(&rewriter, module, parent_arena);
    rewriter_initialized = true;
    status = loom_rewriter_enable_worklist(&rewriter);
    for (loom_integer_division_plan_t* plan = planner.first_plan;
         plan && iree_status_is_ok(status); plan = plan->next) {
      status = loom_integer_division_apply_plan(&rewriter, plan);
      if (iree_status_is_ok(status)) {
        ++out_result->combined_operation_count;
        out_result->retired_operation_count += plan->candidate_count;
      }
    }
    while (iree_status_is_ok(status)) {
      loom_op_t* op = loom_rewriter_pop(&rewriter);
      if (!op) {
        break;
      }
      bool erased = false;
      status = loom_rewriter_erase_if_dead(&rewriter, op, &erased);
    }
  }
  if (rewriter_initialized) {
    loom_rewriter_deinitialize(&rewriter);
  }
  iree_arena_checkpoint_restore(&planner_checkpoint);
  return status;
}
