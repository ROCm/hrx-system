// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scalar/cleanup_patterns.h"

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
          &comparison,
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
