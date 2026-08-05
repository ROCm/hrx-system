// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_expr_bounded.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbolic_expr_test_fixture.h"
#include "loom/ops/index/ops.h"

namespace loom {
namespace {

TEST_F(SymbolicExprTest, BoundedExpressionPreservesOpaqueAssumeResult) {
  loom_value_id_t row = DefineIndexValue();
  loom_predicate_t predicate = {
      /*.kind=*/LOOM_PREDICATE_RANGE,
      /*.arg_count=*/3,
      /*.arg_tags=*/
      {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST, LOOM_PRED_ARG_CONST},
      /*.reserved=*/{},
      /*.args=*/{row, 0, 64},
  };
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(&builder_, &row, 1, &predicate, 1,
                                         &index_type, 1, LOOM_LOCATION_UNKNOWN,
                                         &assume_op));
  loom_value_id_t bounded_row = loom_index_assume_results(assume_op).values[0];

  loom_symbolic_term_t terms[1] = {};
  loom_symbolic_expr_t expression = {};
  loom_symbolic_expr_from_value_bounded(module_, &fact_table_, bounded_row,
                                        terms, IREE_ARRAYSIZE(terms),
                                        &expression);

  ASSERT_TRUE(loom_symbolic_expr_is_linear(&expression));
  EXPECT_EQ(expression.constant, 0);
  ASSERT_EQ(expression.term_count, 1);
  EXPECT_EQ(expression.terms[0].coefficient, 1);
  EXPECT_EQ(expression.terms[0].value_id, bounded_row);
}

TEST_F(SymbolicExprTest, BoundedExpressionExpandsAssumedAffineSource) {
  loom_value_id_t row = DefineIndexValue();
  loom_value_id_t four = loom_index_constant_result(BuildIndexConstant(4));
  loom_op_t* shifted_op = nullptr;
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  IREE_ASSERT_OK(loom_index_add_build(&builder_, row, four, index_type,
                                      LOOM_LOCATION_UNKNOWN, &shifted_op));
  loom_value_id_t shifted = loom_index_add_result(shifted_op);
  loom_predicate_t predicate = {
      /*.kind=*/LOOM_PREDICATE_RANGE,
      /*.arg_count=*/3,
      /*.arg_tags=*/
      {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST, LOOM_PRED_ARG_CONST},
      /*.reserved=*/{},
      /*.args=*/{shifted, 0, 64},
  };
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(&builder_, &shifted, 1, &predicate, 1,
                                         &index_type, 1, LOOM_LOCATION_UNKNOWN,
                                         &assume_op));
  loom_value_id_t bounded_shifted =
      loom_index_assume_results(assume_op).values[0];

  loom_symbolic_term_t terms[1] = {};
  loom_symbolic_expr_t expression = {};
  loom_symbolic_expr_from_value_bounded(module_, &fact_table_, bounded_shifted,
                                        terms, IREE_ARRAYSIZE(terms),
                                        &expression);

  ASSERT_TRUE(loom_symbolic_expr_is_linear(&expression));
  EXPECT_EQ(expression.constant, 4);
  ASSERT_EQ(expression.term_count, 1);
  EXPECT_EQ(expression.terms[0].coefficient, 1);
  EXPECT_EQ(expression.terms[0].value_id, row);
}

TEST_F(SymbolicExprTest, BoundedExpressionNormalizesCancellation) {
  loom_value_id_t row = DefineIndexValue();
  loom_value_id_t column = DefineIndexValue();
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* sum_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, row, column, index_type,
                                      LOOM_LOCATION_UNKNOWN, &sum_op));
  loom_op_t* difference_op = nullptr;
  IREE_ASSERT_OK(loom_index_sub_build(&builder_, loom_index_add_result(sum_op),
                                      column, index_type, LOOM_LOCATION_UNKNOWN,
                                      &difference_op));

  loom_symbolic_term_t terms[1] = {};
  loom_symbolic_expr_t expression = {};
  loom_symbolic_expr_from_value_bounded(
      module_, &fact_table_, loom_index_sub_result(difference_op), terms,
      IREE_ARRAYSIZE(terms), &expression);

  ASSERT_TRUE(loom_symbolic_expr_is_linear(&expression));
  EXPECT_EQ(expression.constant, 0);
  ASSERT_EQ(expression.term_count, 1);
  EXPECT_EQ(expression.terms[0].coefficient, 1);
  EXPECT_EQ(expression.terms[0].value_id, row);
}

TEST_F(SymbolicExprTest, BoundedDynamicMaddFallsBackToResultSymbol) {
  loom_value_id_t left = DefineIndexValue();
  loom_value_id_t right = DefineIndexValue();
  loom_value_id_t addend = DefineIndexValue();
  loom_op_t* madd_op = nullptr;
  IREE_ASSERT_OK(loom_index_madd_build(&builder_, left, right, addend,
                                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                       LOOM_LOCATION_UNKNOWN, &madd_op));
  loom_value_id_t result = loom_index_madd_result(madd_op);

  loom_symbolic_term_t terms[2] = {};
  loom_symbolic_expr_t expression = {};
  loom_symbolic_expr_from_value_bounded(module_, &fact_table_, result, terms,
                                        IREE_ARRAYSIZE(terms), &expression);

  ASSERT_TRUE(loom_symbolic_expr_is_linear(&expression));
  EXPECT_EQ(expression.constant, 0);
  ASSERT_EQ(expression.term_count, 1);
  EXPECT_EQ(expression.terms[0].coefficient, 1);
  EXPECT_EQ(expression.terms[0].value_id, result);
}

}  // namespace
}  // namespace loom
