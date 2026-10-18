// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/index/cleanup_patterns.h"

#include "loom/analysis/condition_facts.h"
#include "loom/analysis/symbolic_expr.h"
#include "loom/analysis/symbolic_expr_proof.h"
#include "loom/ir/module.h"
#include "loom/ops/index/carrier.h"
#include "loom/ops/index/compare.h"
#include "loom/ops/index/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/cleanup/patterns.h"

static iree_status_t loom_index_replace_single_result_with_value(
    loom_rewriter_t* rewriter, loom_op_t* op, loom_value_id_t replacement) {
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static iree_status_t loom_index_replace_single_result_with_exact_i64(
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
  return loom_index_replace_single_result_with_value(rewriter, op, replacement);
}

static iree_status_t loom_index_symbolic_sub_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  *out_changed = false;
  loom_cleanup_pattern_context_t* cleanup_context =
      (loom_cleanup_pattern_context_t*)context;

  loom_symbolic_value_difference_t difference = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_simplify_value_difference(
      cleanup_context->symbolic_expression_context, loom_index_sub_lhs(op),
      loom_index_sub_rhs(op), &difference));
  switch (difference.kind) {
    case LOOM_SYMBOLIC_VALUE_DIFFERENCE_CONSTANT: {
      loom_value_facts_t constant_facts =
          loom_value_facts_exact_i64(difference.constant);
      const loom_type_t result_type = loom_module_value_type(
          rewriter->module, loom_op_const_results(op)[0]);
      if (loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_OFFSET) {
        const loom_fact_context_t* fact_context =
            rewriter->fact_table ? &rewriter->fact_table->context : NULL;
        constant_facts =
            loom_index_wrap_offset_facts(fact_context, constant_facts);
        if (!loom_value_facts_is_exact(constant_facts)) {
          return iree_ok_status();
        }
      }
      IREE_RETURN_IF_ERROR(loom_index_replace_single_result_with_exact_i64(
          rewriter, op, constant_facts.range_lo));
      *out_changed = true;
      return iree_ok_status();
    }
    case LOOM_SYMBOLIC_VALUE_DIFFERENCE_VALUE: {
      const loom_type_t result_type = loom_module_value_type(
          rewriter->module, loom_op_const_results(op)[0]);
      const loom_type_t replacement_type =
          loom_module_value_type(rewriter->module, difference.value_id);
      if (!loom_type_equal(result_type, replacement_type)) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(loom_index_replace_single_result_with_value(
          rewriter, op, difference.value_id));
      *out_changed = true;
      return iree_ok_status();
    }
    case LOOM_SYMBOLIC_VALUE_DIFFERENCE_UNKNOWN:
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_index_symbolic_cmp_pattern(
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
  const loom_type_t operand_type =
      loom_module_value_type(rewriter->module, comparison.lhs);
  if (!loom_type_is_scalar(operand_type)) {
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
          &comparison,
          rewriter->fact_table ? &rewriter->fact_table->context : NULL,
          /*assumed_truth=*/true, &lhs_facts, &rhs_facts, &relation.relation)) {
    return iree_ok_status();
  }

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_value_relation(
      cleanup_context->symbolic_expression_context, relation.relation,
      relation.left.value_id, relation.right.value_id, &proof));
  if (proof == LOOM_SYMBOLIC_PROOF_UNKNOWN) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_index_replace_single_result_with_exact_i64(
      rewriter, op, proof == LOOM_SYMBOLIC_PROOF_TRUE ? 1 : 0));
  *out_changed = true;
  return iree_ok_status();
}

static const loom_rewrite_pattern_t kIndexUniversalPostTypePatterns[] = {
    {
        .root_kind = LOOM_OP_INDEX_SUB,
        .match_and_rewrite = loom_index_symbolic_sub_pattern,
    },
    {
        .root_kind = LOOM_OP_INDEX_CMP,
        .match_and_rewrite = loom_index_symbolic_cmp_pattern,
    },
};

const loom_rewrite_pattern_provider_t
    loom_index_universal_post_type_pattern_provider = {
        .name = IREE_SVL("index"),
        .patterns = kIndexUniversalPostTypePatterns,
        .pattern_count = IREE_ARRAYSIZE(kIndexUniversalPostTypePatterns),
};
