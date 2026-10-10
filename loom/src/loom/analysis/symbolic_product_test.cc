// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_product.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbolic_expr_proof.h"
#include "loom/analysis/symbolic_expr_test_fixture.h"
#include "loom/target/facts.h"

namespace loom {
namespace {

class SymbolicProductTest : public SymbolicExprTest {
 protected:
  loom_value_id_t DefineBounded(loom_scalar_type_t type, int64_t lower,
                                int64_t upper) {
    loom_value_id_t value;
    IREE_CHECK_OK(
        loom_builder_define_value(&builder_, loom_type_scalar(type), &value));
    DefineFacts(value, loom_value_facts_make(lower, upper, 1));
    return value;
  }

  loom_value_id_t Multiply(loom_value_id_t left, loom_value_id_t right) {
    loom_op_t* op;
    IREE_CHECK_OK(loom_index_mul_build(&builder_, left, right,
                                       LOOM_LOCATION_UNKNOWN, &op));
    ComputeFacts(op);
    return loom_index_mul_result(op);
  }

  loom_symbolic_expr_t Expand(loom_value_id_t value) {
    loom_symbolic_expr_t expression;
    IREE_CHECK_OK(loom_symbolic_expr_from_value(&expression_context_, value,
                                                &expression));
    return expression;
  }

  void ExpectEqual(loom_value_id_t left, loom_value_id_t right) {
    loom_symbolic_value_difference_t difference;
    IREE_ASSERT_OK(loom_symbolic_expr_simplify_value_difference(
        &expression_context_, left, right, &difference));
    EXPECT_EQ(difference.kind, LOOM_SYMBOLIC_VALUE_DIFFERENCE_CONSTANT);
    EXPECT_EQ(difference.constant, 0);
  }
};

TEST_F(SymbolicProductTest, ReassociationPreservesMaterializedValues) {
  const auto rows = DefineBounded(LOOM_SCALAR_TYPE_INDEX, 0, 7);
  const auto columns = DefineBounded(LOOM_SCALAR_TYPE_INDEX, 0, 8);
  const auto width = loom_index_constant_result(BuildIndexConstant(16));
  const auto expanded = Multiply(rows, Multiply(columns, width));
  const auto shared = Multiply(Multiply(columns, rows), width);
  ExpectEqual(expanded, shared);

  const auto expression = Expand(expanded);
  ASSERT_EQ(expression.term_count, 1u);
  EXPECT_EQ(expression.terms[0].value_id, expanded);
  EXPECT_EQ(expression.terms[0].coefficient, 1);
  const auto* product =
      loom_symbolic_expr_lookup_product(&expression_context_, expanded);
  ASSERT_NE(product, nullptr);
  EXPECT_EQ(product->scale, 16);
  ASSERT_EQ(product->factor_count, 2u);
  EXPECT_EQ(product->factors[0], rows);
  EXPECT_EQ(product->factors[1], columns);
  loom_symbolic_expr_summary_t summary;
  ASSERT_TRUE(loom_symbolic_expr_context_try_lookup_summary(
      &expression_context_, expanded, &summary));
  EXPECT_EQ(summary.materialized_dynamic_value_id, expanded);
}

TEST_F(SymbolicProductTest, FactorOrderAndMultiplicityArePreserved) {
  const auto x = DefineBounded(LOOM_SCALAR_TYPE_INDEX, -3, 3);
  const auto y = DefineBounded(LOOM_SCALAR_TYPE_INDEX, -3, 3);
  const auto z = DefineBounded(LOOM_SCALAR_TYPE_INDEX, -3, 3);
  const auto left = Multiply(Multiply(x, y), Multiply(z, x));
  const auto right = Multiply(z, Multiply(x, Multiply(x, y)));
  const auto different = Multiply(z, Multiply(y, Multiply(x, y)));
  ExpectEqual(left, right);
  const auto left_expression = Expand(left);
  const auto different_expression = Expand(different);
  int64_t left_coefficient = 0;
  int64_t right_coefficient = 0;
  EXPECT_FALSE(loom_symbolic_product_match_terms(
      &expression_context_, &left_expression.terms[0],
      &different_expression.terms[0], &left_coefficient, &right_coefficient));
  const auto* product =
      loom_symbolic_expr_lookup_product(&expression_context_, left);
  ASSERT_NE(product, nullptr);
  ASSERT_EQ(product->factor_count, 4u);
  EXPECT_EQ(product->factors[0], x);
  EXPECT_EQ(product->factors[1], x);
  EXPECT_EQ(product->factors[2], y);
  EXPECT_EQ(product->factors[3], z);
}

TEST_F(SymbolicProductTest, FactorBudgetDoesNotInventAnIdentity) {
  expression_context_.maximum_term_count = 3;
  const auto x = DefineBounded(LOOM_SCALAR_TYPE_INDEX, 0, 1);
  const auto square = Multiply(x, x);
  const auto cube = Multiply(square, x);
  const auto fourth = Multiply(cube, x);
  const auto expression = Expand(fourth);
  ASSERT_EQ(expression.term_count, 1u);
  EXPECT_EQ(expression.terms[0].value_id, fourth);
  EXPECT_NE(loom_symbolic_expr_lookup_product(&expression_context_, cube),
            nullptr);
  EXPECT_EQ(loom_symbolic_expr_lookup_product(&expression_context_, fourth),
            nullptr);
}

TEST_F(SymbolicProductTest, FixedWidthProductsRequireMathematicalNoWrap) {
  for (auto scalar_type :
       {LOOM_SCALAR_TYPE_I1, LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_I16,
        LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I64}) {
    SCOPED_TRACE(scalar_type);
    const auto type = loom_type_scalar(scalar_type);
    const auto left = DefineBounded(scalar_type, 0, 1);
    const auto right = DefineBounded(scalar_type, 0, 1);
    loom_op_t* op;
    IREE_ASSERT_OK(loom_scalar_muli_build(&builder_, 0, left, right, type,
                                          LOOM_LOCATION_UNKNOWN, &op));
    ComputeFacts(op);
    const auto result = loom_scalar_muli_result(op);
    Expand(result);
    EXPECT_NE(loom_symbolic_expr_lookup_product(&expression_context_, result),
              nullptr);
    if (scalar_type == LOOM_SCALAR_TYPE_I1) {
      continue;
    }

    const auto width = loom_scalar_type_bitwidth(scalar_type);
    const int64_t maximum =
        width == 64 ? INT64_MAX : (INT64_C(1) << (width - 1)) - 1;
    DefineFacts(left, loom_value_facts_make(1, maximum, 1));
    DefineFacts(right, loom_value_facts_make(1, 2, 1));
    ComputeFacts(op);
    const auto expression = Expand(result);
    ASSERT_EQ(expression.term_count, 1u);
    EXPECT_EQ(expression.terms[0].value_id, result);
    EXPECT_EQ(loom_symbolic_expr_lookup_product(&expression_context_, result),
              nullptr);
  }
}

TEST_F(SymbolicProductTest, ExplicitNoWrapContractPermitsProductProof) {
  const auto type = loom_type_scalar(LOOM_SCALAR_TYPE_I8);
  const auto left = DefineBounded(LOOM_SCALAR_TYPE_I8, -128, 127);
  const auto right = DefineBounded(LOOM_SCALAR_TYPE_I8, -128, 127);
  loom_op_t* op;
  IREE_ASSERT_OK(
      loom_scalar_muli_build(&builder_, LOOM_SCALAR_INTOVERFLOWFLAGS_NSW, left,
                             right, type, LOOM_LOCATION_UNKNOWN, &op));
  ComputeFacts(op);
  const auto result = loom_scalar_muli_result(op);
  Expand(result);
  EXPECT_NE(loom_symbolic_expr_lookup_product(&expression_context_, result),
            nullptr);
}

TEST_F(SymbolicProductTest, ZeroAddendIsProductButNonzeroAddendIsNot) {
  const auto left = DefineBounded(LOOM_SCALAR_TYPE_INDEX, 0, 7);
  const auto right = DefineBounded(LOOM_SCALAR_TYPE_INDEX, 0, 8);
  const auto product = Multiply(left, right);
  for (int64_t addend : {0, 1}) {
    SCOPED_TRACE(addend);
    loom_op_t* op;
    IREE_ASSERT_OK(loom_index_madd_build(
        &builder_, left, right,
        loom_index_constant_result(BuildIndexConstant(addend)),
        LOOM_LOCATION_UNKNOWN, &op));
    ComputeFacts(op);
    const auto result = loom_index_madd_result(op);
    Expand(result);
    EXPECT_EQ(loom_symbolic_expr_lookup_product(&expression_context_, result) !=
                  nullptr,
              addend == 0);
    if (addend == 0) {
      ExpectEqual(product, result);
    }
  }
}

TEST_F(SymbolicProductTest, TargetCarrierAndResetBoundProductLifetime) {
  const auto left = DefineBounded(LOOM_SCALAR_TYPE_INDEX, 0, 65536);
  const auto right = DefineBounded(LOOM_SCALAR_TYPE_INDEX, 0, 65536);
  const auto product = Multiply(left, right);
  loom_target_facts_t target = {};
  fact_table_.context.target_facts = &target;
  for (uint32_t width : {32, 64, 32}) {
    SCOPED_TRACE(width);
    target.storage.snapshot.index_bitwidth = width;
    loom_symbolic_expr_context_reset(&expression_context_);
    EXPECT_EQ(expression_context_.products.count, 0u);
    EXPECT_EQ(loom_symbolic_expr_lookup_product(&expression_context_, product),
              nullptr);
    Expand(product);
    EXPECT_EQ(loom_symbolic_expr_lookup_product(&expression_context_,
                                                product) != nullptr,
              width == 64);
  }
  fact_table_.context.target_facts = nullptr;
}

}  // namespace
}  // namespace loom
