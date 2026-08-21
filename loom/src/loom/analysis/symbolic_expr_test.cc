// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_expr.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbolic_expr_test_fixture.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"

namespace loom {
namespace {

TEST_F(SymbolicExprTest, UnknownValueIsMemoizedLinearTerm) {
  loom_value_id_t value_id = DefineIndexValue();

  loom_symbolic_expr_summary_t ready_summary = {};
  EXPECT_FALSE(loom_symbolic_expr_context_try_lookup_summary(
      &expression_context_, value_id, &ready_summary));

  loom_symbolic_expr_t first_expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, value_id,
                                               &first_expression));
  ASSERT_TRUE(loom_symbolic_expr_is_linear(&first_expression));
  ASSERT_EQ(first_expression.term_count, 1);
  EXPECT_EQ(first_expression.terms[0].coefficient, 1);
  EXPECT_EQ(first_expression.terms[0].value_id, value_id);

  loom_symbolic_expr_t second_expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, value_id,
                                               &second_expression));
  EXPECT_EQ(second_expression.terms, first_expression.terms);

  EXPECT_TRUE(loom_symbolic_expr_context_try_lookup_summary(
      &expression_context_, value_id, &ready_summary));
  EXPECT_EQ(ready_summary.expression.terms, first_expression.terms);

  loom_symbolic_expr_context_reset(&expression_context_);
  EXPECT_FALSE(loom_symbolic_expr_context_try_lookup_summary(
      &expression_context_, value_id, &ready_summary));
}

TEST_F(SymbolicExprTest, ExactIntegerFactsFoldToConstant) {
  loom_value_id_t value_id = DefineIndexValue();
  DefineFacts(value_id, loom_value_facts_exact_i64(42));

  loom_symbolic_expr_t expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, value_id,
                                               &expression));

  EXPECT_TRUE(loom_symbolic_expr_is_constant(&expression));
  EXPECT_EQ(expression.constant, 42);
  EXPECT_EQ(expression.term_count, 0);
}

TEST_F(SymbolicExprTest, AddAndSubtractNormalizeTerms) {
  loom_value_id_t value_id = DefineIndexValue();
  loom_symbolic_expr_t value = {0};
  IREE_ASSERT_OK(
      loom_symbolic_expr_from_value(&expression_context_, value_id, &value));
  loom_symbolic_expr_t four = {0};
  loom_symbolic_expr_constant(4, &four);
  loom_symbolic_expr_t eight = {0};
  loom_symbolic_expr_constant(8, &eight);

  loom_symbolic_expr_t value_plus_four = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_add(&expression_context_, &value, &four,
                                        &value_plus_four));
  loom_symbolic_expr_t value_plus_eight = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_add(&expression_context_, &value, &eight,
                                        &value_plus_eight));
  loom_symbolic_expr_t difference = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_sub(&expression_context_, &value_plus_four,
                                        &value_plus_eight, &difference));

  EXPECT_TRUE(loom_symbolic_expr_is_constant(&difference));
  EXPECT_EQ(difference.constant, -4);
}

TEST_F(SymbolicExprTest, TermsAreNormalizedByValueId) {
  loom_value_id_t first_value = DefineIndexValue();
  loom_value_id_t second_value = DefineIndexValue();
  loom_symbolic_expr_t first = {0};
  IREE_ASSERT_OK(
      loom_symbolic_expr_from_value(&expression_context_, first_value, &first));
  loom_symbolic_expr_t second = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_,
                                               second_value, &second));

  loom_symbolic_expr_t expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_add(&expression_context_, &second, &first,
                                        &expression));

  ASSERT_EQ(expression.term_count, 2);
  EXPECT_EQ(expression.terms[0].value_id, first_value);
  EXPECT_EQ(expression.terms[1].value_id, second_value);
}

TEST_F(SymbolicExprTest, MemoGrowthPreservesOuterExpansion) {
  loom_value_id_t source = DefineI64Value();
  loom_type_t i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  loom_predicate_t predicate = {
      /*.kind=*/LOOM_PREDICATE_GE,
      /*.arg_count=*/2,
      /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
      /*.reserved=*/{},
      /*.args=*/{source, 0},
  };
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_scalar_assume_build(&builder_, &source, 1, &predicate, 1,
                                          &i64_type, 1, LOOM_LOCATION_UNKNOWN,
                                          &assume_op));
  loom_value_id_t assumed = loom_scalar_assume_results(assume_op).values[0];

  loom_op_t* cast_op = nullptr;
  IREE_ASSERT_OK(loom_index_cast_build(&builder_, assumed, i64_type,
                                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                       LOOM_LOCATION_UNKNOWN, &cast_op));
  loom_value_id_t outer_value = loom_index_cast_result(cast_op);

  loom_value_id_t replacement = DefineI64Value();
  ASSERT_GT(replacement, outer_value);
  IREE_ASSERT_OK(
      loom_value_replace_all_uses_with(module_, source, replacement));

  loom_symbolic_expr_t expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_,
                                               outer_value, &expression));
  ASSERT_EQ(expression.term_count, 1);
  EXPECT_EQ(expression.terms[0].value_id, replacement);

  loom_symbolic_expr_t memoized_expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, outer_value, &memoized_expression));
  EXPECT_EQ(memoized_expression.terms, expression.terms);
}

TEST_F(SymbolicExprTest, DeepProducerChainExpandsIteratively) {
  loom_value_id_t source = DefineI64Value();
  loom_value_id_t value = source;
  for (int i = 0; i < 4096; ++i) {
    loom_op_t* negate_op = nullptr;
    IREE_ASSERT_OK(loom_scalar_negi_build(
        &builder_, value, loom_type_scalar(LOOM_SCALAR_TYPE_I64),
        LOOM_LOCATION_UNKNOWN, &negate_op));
    value = loom_scalar_negi_result(negate_op);
  }

  loom_symbolic_expr_t expression = {0};
  IREE_ASSERT_OK(
      loom_symbolic_expr_from_value(&expression_context_, value, &expression));
  ASSERT_TRUE(loom_symbolic_expr_is_linear(&expression));
  ASSERT_EQ(expression.term_count, 1);
  EXPECT_EQ(expression.terms[0].coefficient, 1);
  EXPECT_EQ(expression.terms[0].value_id, source);

  loom_symbolic_expr_t memoized_expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, value,
                                               &memoized_expression));
  EXPECT_EQ(memoized_expression.terms, expression.terms);
}

TEST_F(SymbolicExprTest, ExpandsIndexMaddWithConstantMultiplier) {
  loom_value_id_t row = DefineIndexValue();
  loom_value_id_t column = DefineIndexValue();
  loom_value_id_t stride = loom_index_constant_result(BuildIndexConstant(16));
  loom_op_t* madd_op = nullptr;
  IREE_ASSERT_OK(loom_index_madd_build(&builder_, row, stride, column,
                                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                       LOOM_LOCATION_UNKNOWN, &madd_op));

  loom_symbolic_expr_t expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, loom_index_madd_result(madd_op), &expression));

  ASSERT_TRUE(loom_symbolic_expr_is_linear(&expression));
  ASSERT_EQ(expression.term_count, 2);
  EXPECT_EQ(expression.terms[0].coefficient, 16);
  EXPECT_EQ(expression.terms[0].value_id, row);
  EXPECT_EQ(expression.terms[1].coefficient, 1);
  EXPECT_EQ(expression.terms[1].value_id, column);
}

TEST_F(SymbolicExprTest, ReadySummaryRetainsConstantFreeMaterialization) {
  loom_value_id_t row = DefineIndexValue();
  loom_value_id_t column = DefineIndexValue();
  loom_value_id_t stride = loom_index_constant_result(BuildIndexConstant(16));
  loom_op_t* linear_op = nullptr;
  IREE_ASSERT_OK(loom_index_madd_build(&builder_, row, stride, column,
                                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                       LOOM_LOCATION_UNKNOWN, &linear_op));
  loom_value_id_t offset = loom_index_constant_result(BuildIndexConstant(4));
  loom_op_t* offset_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_,
                                      loom_index_madd_result(linear_op), offset,
                                      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                      LOOM_LOCATION_UNKNOWN, &offset_op));
  loom_value_id_t offset_value = loom_index_add_result(offset_op);

  loom_symbolic_expr_summary_t summary = {};
  EXPECT_FALSE(loom_symbolic_expr_context_try_lookup_summary(
      &expression_context_, offset_value, &summary));
  loom_symbolic_expr_t expression = {};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_,
                                               offset_value, &expression));
  ASSERT_TRUE(loom_symbolic_expr_context_try_lookup_summary(
      &expression_context_, offset_value, &summary));

  EXPECT_EQ(summary.expression.constant, 4);
  ASSERT_EQ(summary.expression.term_count, 2);
  EXPECT_EQ(summary.materialized_dynamic_value_id,
            loom_index_madd_result(linear_op));
}

TEST_F(SymbolicExprTest, DynamicMultiplyFallsBackToResultSymbol) {
  loom_value_id_t left = DefineIndexValue();
  loom_value_id_t right = DefineIndexValue();
  loom_op_t* mul_op = nullptr;
  IREE_ASSERT_OK(loom_index_mul_build(&builder_, left, right,
                                      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                      LOOM_LOCATION_UNKNOWN, &mul_op));
  loom_value_id_t result = loom_index_mul_result(mul_op);

  loom_symbolic_expr_t expression = {0};
  IREE_ASSERT_OK(
      loom_symbolic_expr_from_value(&expression_context_, result, &expression));

  ASSERT_TRUE(loom_symbolic_expr_is_linear(&expression));
  ASSERT_EQ(expression.term_count, 1);
  EXPECT_EQ(expression.terms[0].coefficient, 1);
  EXPECT_EQ(expression.terms[0].value_id, result);
}

TEST_F(SymbolicExprTest, DynamicMaddFallsBackToResultSymbol) {
  loom_value_id_t left = DefineIndexValue();
  loom_value_id_t right = DefineIndexValue();
  loom_value_id_t addend = DefineIndexValue();
  loom_op_t* madd_op = nullptr;
  IREE_ASSERT_OK(loom_index_madd_build(&builder_, left, right, addend,
                                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                       LOOM_LOCATION_UNKNOWN, &madd_op));
  loom_value_id_t result = loom_index_madd_result(madd_op);

  loom_symbolic_expr_t expression = {0};
  IREE_ASSERT_OK(
      loom_symbolic_expr_from_value(&expression_context_, result, &expression));

  ASSERT_TRUE(loom_symbolic_expr_is_linear(&expression));
  ASSERT_EQ(expression.term_count, 1);
  EXPECT_EQ(expression.terms[0].coefficient, 1);
  EXPECT_EQ(expression.terms[0].value_id, result);
}

TEST_F(SymbolicExprTest, ExpandsShiftWithExactAmount) {
  loom_value_id_t value = DefineIndexValue();
  loom_value_id_t shift = loom_index_constant_result(BuildIndexConstant(3));
  loom_op_t* shift_op = nullptr;
  IREE_ASSERT_OK(loom_index_shli_build(&builder_, value, shift,
                                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                       LOOM_LOCATION_UNKNOWN, &shift_op));

  loom_symbolic_expr_t expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, loom_index_shli_result(shift_op), &expression));

  ASSERT_TRUE(loom_symbolic_expr_is_linear(&expression));
  ASSERT_EQ(expression.term_count, 1);
  EXPECT_EQ(expression.terms[0].coefficient, 8);
  EXPECT_EQ(expression.terms[0].value_id, value);
}

TEST_F(SymbolicExprTest, DynamicShiftFallsBackToResultSymbol) {
  loom_value_id_t value = DefineIndexValue();
  loom_value_id_t shift = DefineIndexValue();
  loom_op_t* shift_op = nullptr;
  IREE_ASSERT_OK(loom_index_shli_build(&builder_, value, shift,
                                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                       LOOM_LOCATION_UNKNOWN, &shift_op));
  loom_value_id_t result = loom_index_shli_result(shift_op);

  loom_symbolic_expr_t expression = {0};
  IREE_ASSERT_OK(
      loom_symbolic_expr_from_value(&expression_context_, result, &expression));

  ASSERT_TRUE(loom_symbolic_expr_is_linear(&expression));
  ASSERT_EQ(expression.term_count, 1);
  EXPECT_EQ(expression.terms[0].coefficient, 1);
  EXPECT_EQ(expression.terms[0].value_id, result);
}

TEST_F(SymbolicExprTest, SelectUsesExactConditionFacts) {
  loom_value_id_t condition = DefineIndexValue();
  loom_value_id_t true_value = DefineIndexValue();
  loom_value_id_t false_value = DefineIndexValue();
  DefineFacts(condition, loom_value_facts_exact_i64(1));
  loom_op_t* select_op = nullptr;
  IREE_ASSERT_OK(loom_scf_select_build(&builder_, condition, true_value,
                                       false_value,
                                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                       LOOM_LOCATION_UNKNOWN, &select_op));

  loom_symbolic_expr_t expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, loom_scf_select_result(select_op), &expression));

  ASSERT_EQ(expression.term_count, 1);
  EXPECT_EQ(expression.terms[0].value_id, true_value);
}

TEST_F(SymbolicExprTest, SelectUsesConstantConditionExpression) {
  loom_value_id_t condition = loom_index_constant_result(BuildIndexConstant(0));
  loom_value_id_t true_value = DefineIndexValue();
  loom_value_id_t false_value = DefineIndexValue();
  loom_op_t* select_op = nullptr;
  IREE_ASSERT_OK(loom_scf_select_build(&builder_, condition, true_value,
                                       false_value,
                                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                       LOOM_LOCATION_UNKNOWN, &select_op));

  loom_symbolic_expr_t expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, loom_scf_select_result(select_op), &expression));

  ASSERT_EQ(expression.term_count, 1);
  EXPECT_EQ(expression.terms[0].value_id, false_value);
}

}  // namespace
}  // namespace loom
