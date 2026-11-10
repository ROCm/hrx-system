// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scalar/cleanup_patterns.h"

#include "iree/base/internal/math.h"
#include "loom/analysis/condition_facts.h"
#include "loom/analysis/symbolic_expr.h"
#include "loom/analysis/symbolic_expr_proof.h"
#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/cleanup/patterns.h"

static iree_status_t loom_scalar_replace_single_result_with_exact_i64(
    loom_rewriter_t* rewriter, loom_op_t* op, int64_t value) {
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t result = loom_op_const_results(op)[0];
  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, result);

  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_build_constant(rewriter, loom_value_facts_exact_i64(value),
                                   result_type, op->location, &replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

typedef struct loom_scalar_prepared_operand_replacement_t {
  // Root-local application owned by the shared prepared batch.
  loom_rewrite_action_t action;
  // Live root operand selected as the replacement.
  uint8_t operand;
} loom_scalar_prepared_operand_replacement_t;

static bool loom_scalar_preparation_exact_i64(
    const loom_value_fact_table_t* facts, loom_value_id_t value_id,
    int64_t* out_value) {
  const loom_value_facts_t value_facts =
      loom_value_fact_table_lookup(facts, value_id);
  if (!loom_value_facts_is_exact(value_facts) ||
      loom_value_facts_is_float(value_facts)) {
    return false;
  }
  *out_value = value_facts.range_lo;
  return true;
}

static bool loom_scalar_preparation_all_ones(loom_type_t type, int64_t value) {
  if (value == -1) {
    return true;
  }
  const int32_t bitwidth =
      loom_scalar_type_bitwidth(loom_type_element_type(type));
  return bitwidth > 0 && bitwidth < 63 &&
         value == (((int64_t)1 << bitwidth) - 1);
}

static bool loom_scalar_preparation_fits_low_mask(
    loom_value_facts_t value_facts, int64_t mask) {
  if (mask <= 0) {
    return false;
  }
  const uint64_t unsigned_mask = (uint64_t)mask;
  if (((unsigned_mask + 1) & unsigned_mask) != 0) {
    return false;
  }
  const uint8_t bit_count =
      (uint8_t)iree_math_count_trailing_zeros_u64(unsigned_mask + 1);
  return bit_count != 0 &&
         loom_value_facts_fit_unsigned_bit_count(value_facts, bit_count);
}

static iree_status_t loom_scalar_apply_prepared_operand_replacement(
    const loom_rewrite_action_t* base_action, loom_rewriter_t* rewriter) {
  loom_op_t* op = base_action->root;
  if (loom_rewriter_is_trivially_dead(rewriter, op)) {
    return loom_rewriter_erase(rewriter, op);
  }
  const loom_scalar_prepared_operand_replacement_t* action =
      (const loom_scalar_prepared_operand_replacement_t*)base_action;
  const loom_value_id_t replacement =
      loom_op_const_operands(op)[action->operand];
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static iree_status_t loom_scalar_prepare_andi_pattern(
    const loom_rewrite_pattern_t* pattern, const loom_module_t* module,
    const loom_value_fact_table_t* facts, const loom_op_t* op,
    iree_arena_allocator_t* arena, loom_rewrite_action_t** out_action) {
  (void)pattern;
  *out_action = NULL;
  if (!facts || loom_value_facts_is_exact(loom_value_fact_table_lookup(
                    facts, loom_op_const_results(op)[0]))) {
    return iree_ok_status();
  }

  const loom_value_id_t lhs = loom_scalar_andi_lhs(op);
  const loom_value_id_t rhs = loom_scalar_andi_rhs(op);
  uint8_t replacement_operand = 0;
  if (lhs == rhs) {
    replacement_operand = 0;
  } else {
    const loom_type_t type =
        loom_module_value_type(module, loom_scalar_andi_result(op));
    int64_t lhs_value = 0;
    int64_t rhs_value = 0;
    if (loom_scalar_preparation_exact_i64(facts, lhs, &lhs_value) &&
        (loom_scalar_preparation_all_ones(type, lhs_value) ||
         loom_scalar_preparation_fits_low_mask(
             loom_value_fact_table_lookup(facts, rhs), lhs_value))) {
      replacement_operand = 1;
    } else if (loom_scalar_preparation_exact_i64(facts, rhs, &rhs_value) &&
               (loom_scalar_preparation_all_ones(type, rhs_value) ||
                loom_scalar_preparation_fits_low_mask(
                    loom_value_fact_table_lookup(facts, lhs), rhs_value))) {
      replacement_operand = 0;
    } else {
      return iree_ok_status();
    }
  }

  loom_scalar_prepared_operand_replacement_t* action = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*action), (void**)&action));
  action->action.apply = loom_scalar_apply_prepared_operand_replacement;
  action->operand = replacement_operand;
  *out_action = &action->action;
  return iree_ok_status();
}

static const loom_rewrite_pattern_t kScalarPreparedCanonicalizationPatterns[] =
    {
        {
            .root_kind = LOOM_OP_SCALAR_ANDI,
            .prepare = loom_scalar_prepare_andi_pattern,
        },
};

const loom_rewrite_pattern_provider_t
    loom_scalar_prepared_canonicalization_pattern_provider = {
        .name = IREE_SVL("scalar-prepared-canonicalization"),
        .patterns = kScalarPreparedCanonicalizationPatterns,
        .pattern_count =
            IREE_ARRAYSIZE(kScalarPreparedCanonicalizationPatterns),
};

static iree_status_t loom_scalar_symbolic_cmpi_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  *out_changed = false;
  loom_cleanup_pattern_context_t* cleanup_context =
      (loom_cleanup_pattern_context_t*)context;

  loom_condition_integer_comparison_t comparison;
  if (!loom_condition_integer_comparison_describe(rewriter->module, op,
                                                  &comparison)) {
    return iree_ok_status();
  }
  const loom_value_facts_t lhs_facts =
      loom_rewriter_value_facts(rewriter, comparison.lhs);
  const loom_value_facts_t rhs_facts =
      loom_rewriter_value_facts(rewriter, comparison.rhs);
  loom_condition_integer_relation_t relation = {
      .left = {.kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
               .value_id = comparison.lhs},
      .right = {.kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
                .value_id = comparison.rhs},
  };
  if (!loom_condition_integer_comparison_equivalent_relation(
          &comparison, rewriter->module,
          rewriter->fact_table ? &rewriter->fact_table->context : NULL,
          /*assumed_truth=*/true, &lhs_facts, &rhs_facts, &relation.relation)) {
    return iree_ok_status();
  }

  // Scalar comparisons describe data-path predicates. Use facts already
  // established by surrounding control flow instead of speculating through
  // every select in an unrolled data graph.
  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_prove_value_relation_with_active_facts(
          cleanup_context->symbolic_expression_context, relation.relation,
          relation.left.value_id, relation.right.value_id, &proof));
  if (proof == LOOM_SYMBOLIC_PROOF_UNKNOWN) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_scalar_replace_single_result_with_exact_i64(
      rewriter, op, proof == LOOM_SYMBOLIC_PROOF_TRUE ? 1 : 0));
  *out_changed = true;
  return iree_ok_status();
}

static const loom_rewrite_pattern_t kScalarUniversalPostTypePatterns[] = {
    {
        .root_kind = LOOM_OP_SCALAR_CMPI,
        .match_and_rewrite = loom_scalar_symbolic_cmpi_pattern,
    },
};

const loom_rewrite_pattern_provider_t
    loom_scalar_universal_post_type_pattern_provider = {
        .name = IREE_SVL("scalar"),
        .patterns = kScalarUniversalPostTypePatterns,
        .pattern_count = IREE_ARRAYSIZE(kScalarUniversalPostTypePatterns),
};
