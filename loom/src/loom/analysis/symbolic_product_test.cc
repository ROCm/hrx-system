// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_product.h"

#include <utility>
#include <vector>

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

  loom_value_id_t ScalarMultiply(loom_value_id_t left, loom_value_id_t right,
                                 uint8_t flags = 0) {
    loom_op_t* op;
    IREE_CHECK_OK(loom_scalar_muli_build(&builder_, flags, left, right,
                                         loom_module_value_type(module_, left),
                                         LOOM_LOCATION_UNKNOWN, &op));
    ComputeFacts(op);
    return loom_scalar_muli_result(op);
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
  for (auto type :
       {LOOM_SCALAR_TYPE_I1, LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_I16,
        LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I64}) {
    SCOPED_TRACE(type);
    int64_t lower = 0;
    int64_t upper = 0;
    ASSERT_TRUE(loom_value_facts_scalar_type_domain(type, &lower, &upper));
    const auto left = DefineBounded(type, lower, upper);
    const auto right = DefineBounded(type, lower, upper);
    for (uint8_t flags : {uint8_t{0}, uint8_t{LOOM_SCALAR_INTOVERFLOWFLAGS_NUW},
                          uint8_t{LOOM_SCALAR_INTOVERFLOWFLAGS_NSW},
                          uint8_t{LOOM_SCALAR_INTOVERFLOWFLAGS_NUW |
                                  LOOM_SCALAR_INTOVERFLOWFLAGS_NSW}}) {
      SCOPED_TRACE(flags);
      const auto result = ScalarMultiply(left, right, flags);
      Expand(result);
      EXPECT_EQ(loom_symbolic_expr_lookup_product(&expression_context_,
                                                  result) != nullptr,
                type == LOOM_SCALAR_TYPE_I1 ||
                    (flags & LOOM_SCALAR_INTOVERFLOWFLAGS_NSW) != 0);
    }
  }
}

TEST_F(SymbolicProductTest, ExactSignedEndpointsAreNotSaturatedRanges) {
  const auto select = DefineBounded(LOOM_SCALAR_TYPE_I64, 0, 1);
  for (const auto bounds :
       {std::pair{INT64_MIN, int64_t{0}}, std::pair{int64_t{0}, INT64_MAX}}) {
    SCOPED_TRACE(::testing::PrintToString(bounds));
    const auto input =
        DefineBounded(LOOM_SCALAR_TYPE_I64, bounds.first, bounds.second);
    const auto left = ScalarMultiply(input, select);
    const auto right = ScalarMultiply(select, input);
    ExpectEqual(left, right);
    EXPECT_NE(loom_symbolic_expr_lookup_product(&expression_context_, left),
              nullptr);
  }
}

TEST_F(SymbolicProductTest, ProductRangeMatchesExhaustiveSmallWidthOracle) {
  const std::pair<int64_t, int64_t> intervals[] = {
      {-128, 127}, {-128, -127}, {-16, -1}, {-8, 7},
      {-1, 1},     {0, 1},       {0, 15},   {15, 16},
  };
  for (const auto left_bounds : intervals) {
    for (const auto right_bounds : intervals) {
      SCOPED_TRACE(::testing::PrintToString(left_bounds));
      SCOPED_TRACE(::testing::PrintToString(right_bounds));
      bool exact = true;
      for (int64_t left = left_bounds.first; left <= left_bounds.second;
           ++left) {
        for (int64_t right = right_bounds.first; right <= right_bounds.second;
             ++right) {
          const int64_t product = left * right;
          exact &= product >= -128 && product <= 127;
        }
      }
      const auto left = DefineBounded(LOOM_SCALAR_TYPE_I8, left_bounds.first,
                                      left_bounds.second);
      const auto right = DefineBounded(LOOM_SCALAR_TYPE_I8, right_bounds.first,
                                       right_bounds.second);
      const auto result = ScalarMultiply(left, right);
      Expand(result);
      EXPECT_EQ(loom_symbolic_expr_lookup_product(&expression_context_,
                                                  result) != nullptr,
                exact);
    }
  }
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

TEST_F(SymbolicProductTest, ScalarFusedZeroAddendPreservesProducts) {
  for (auto scalar_type :
       {LOOM_SCALAR_TYPE_I1, LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_I16,
        LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I64}) {
    SCOPED_TRACE(scalar_type);
    const auto type = loom_type_scalar(scalar_type);
    const auto left = DefineBounded(scalar_type, 0, 1);
    const auto right = DefineBounded(scalar_type, 0, 1);
    const auto product = ScalarMultiply(left, right);
    for (int64_t addend : {0, 1}) {
      SCOPED_TRACE(addend);
      loom_op_t* op;
      IREE_ASSERT_OK(loom_scalar_fmai_build(
          &builder_, 0, left, right, DefineBounded(scalar_type, addend, addend),
          type, LOOM_LOCATION_UNKNOWN, &op));
      ComputeFacts(op);
      const auto result = loom_scalar_fmai_result(op);
      Expand(result);
      EXPECT_EQ(loom_symbolic_expr_lookup_product(&expression_context_,
                                                  result) != nullptr,
                addend == 0);
      if (addend == 0) {
        ExpectEqual(product, result);
      }
    }
  }
}

TEST_F(SymbolicProductTest, ExactShiftsAndNegationPreserveFactorIdentity) {
  for (auto scalar_type : {LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_I16,
                           LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I64}) {
    SCOPED_TRACE(scalar_type);
    const auto type = loom_type_scalar(scalar_type);
    const auto left = DefineBounded(scalar_type, -2, 2);
    const auto right = DefineBounded(scalar_type, -2, 2);
    const auto shift = DefineBounded(scalar_type, 2, 2);
    const auto scale = DefineBounded(scalar_type, -4, -4);
    loom_op_t* shifted_op;
    IREE_ASSERT_OK(loom_scalar_shli_build(&builder_, 0, right, shift, type,
                                          LOOM_LOCATION_UNKNOWN, &shifted_op));
    ComputeFacts(shifted_op);
    const auto product =
        ScalarMultiply(left, loom_scalar_shli_result(shifted_op));
    loom_op_t* negated_op;
    IREE_ASSERT_OK(loom_scalar_negi_build(&builder_, product, type,
                                          LOOM_LOCATION_UNKNOWN, &negated_op));
    ComputeFacts(negated_op);
    const auto scaled_product =
        ScalarMultiply(ScalarMultiply(right, left), scale);
    ExpectEqual(loom_scalar_negi_result(negated_op), scaled_product);
  }
}

TEST_F(SymbolicProductTest, CastsPreserveOnlyNumericProductIdentity) {
  const struct {
    // Cast semantics applied to the product.
    decltype(&loom_scalar_extsi_build) build;
    // Integer domain of the factors and their product.
    loom_scalar_type_t input_type;
    // Integer domain after the cast.
    loom_scalar_type_t result_type;
    // Inclusive lower bound of each factor.
    int64_t lower;
    // Inclusive upper bound of each factor.
    int64_t upper;
    // Whether every product is numerically preserved by the cast.
    bool preserves_value;
  } cases[] = {
      {loom_scalar_extsi_build, LOOM_SCALAR_TYPE_I1, LOOM_SCALAR_TYPE_I32, 0, 1,
       false},
      {loom_scalar_extui_build, LOOM_SCALAR_TYPE_I1, LOOM_SCALAR_TYPE_I32, 0, 1,
       true},
      {loom_scalar_extsi_build, LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_I64, -2,
       2, true},
      {loom_scalar_extui_build, LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_I64, -2,
       2, false},
      {loom_scalar_extui_build, LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_I64, 0, 2,
       true},
      {loom_scalar_trunci_build, LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I8, -11,
       11, true},
      {loom_scalar_trunci_build, LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I8, -12,
       12, false},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.input_type);
    SCOPED_TRACE(test_case.result_type);
    SCOPED_TRACE(test_case.lower);
    const auto left =
        DefineBounded(test_case.input_type, test_case.lower, test_case.upper);
    const auto right =
        DefineBounded(test_case.input_type, test_case.lower, test_case.upper);
    const auto product = ScalarMultiply(left, right);
    loom_op_t* cast;
    IREE_ASSERT_OK(test_case.build(
        &builder_, product, loom_type_scalar(test_case.input_type),
        loom_type_scalar(test_case.result_type), LOOM_LOCATION_UNKNOWN, &cast));
    ComputeFacts(cast);
    const auto result = loom_op_const_results(cast)[0];
    const auto expression = Expand(result);
    ASSERT_EQ(expression.term_count, 1u);
    EXPECT_EQ(expression.terms[0].value_id,
              test_case.preserves_value ? product : result);
    if (test_case.preserves_value) {
      ExpectEqual(product, result);
    }
  }
}

TEST_F(SymbolicProductTest, UnrepresentableCoefficientRemainsOpaque) {
  const auto left = DefineBounded(LOOM_SCALAR_TYPE_I64, 0, 1);
  const auto right = DefineBounded(LOOM_SCALAR_TYPE_I64, 0, 1);
  const int64_t coefficient = INT64_C(1) << 32;
  const auto scale =
      DefineBounded(LOOM_SCALAR_TYPE_I64, coefficient, coefficient);
  const auto result =
      ScalarMultiply(ScalarMultiply(left, scale), ScalarMultiply(right, scale),
                     LOOM_SCALAR_INTOVERFLOWFLAGS_NSW);
  const auto expression = Expand(result);
  ASSERT_EQ(expression.term_count, 1u);
  EXPECT_EQ(expression.terms[0].value_id, result);
  EXPECT_EQ(loom_symbolic_expr_lookup_product(&expression_context_, result),
            nullptr);
}

TEST_F(SymbolicProductTest, OffsetScaleUsesItsIndependentUnsignedCarrier) {
  const auto index = DefineBounded(LOOM_SCALAR_TYPE_INDEX, 0, 65535);
  const auto stride = DefineBounded(LOOM_SCALAR_TYPE_OFFSET, 0, 65536);
  loom_op_t* op;
  IREE_ASSERT_OK(loom_index_scale_build(
      &builder_, index, stride, loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
      LOOM_LOCATION_UNKNOWN, &op));
  ComputeFacts(op);
  const auto result = loom_index_scale_result(op);
  loom_target_facts_t target = {};
  fact_table_.context.target_facts = &target;
  for (uint32_t index_width : {32, 64}) {
    for (uint32_t offset_width : {32, 64}) {
      SCOPED_TRACE(index_width);
      SCOPED_TRACE(offset_width);
      target.storage.snapshot.index_bitwidth = index_width;
      target.storage.snapshot.offset_bitwidth = offset_width;
      for (int64_t upper : {INT64_C(65536), INT64_C(65538)}) {
        SCOPED_TRACE(upper);
        DefineFacts(stride, loom_value_facts_make(0, upper, 1));
        ComputeFacts(op);
        Expand(result);
        EXPECT_EQ(loom_symbolic_expr_lookup_product(&expression_context_,
                                                    result) != nullptr,
                  offset_width == 64 || upper == 65536);
      }
    }
  }
  fact_table_.context.target_facts = nullptr;
}

TEST_F(SymbolicProductTest, GrowthPreservesPreviouslyReturnedProductStorage) {
  const auto left = DefineBounded(LOOM_SCALAR_TYPE_INDEX, 0, 2);
  const auto right = DefineBounded(LOOM_SCALAR_TYPE_INDEX, 0, 3);
  std::vector<loom_value_id_t> values;
  for (int i = 0; i < 64; ++i) {
    values.push_back(Multiply(left, right));
  }
  const auto first = Expand(values.front());
  const auto* retained =
      loom_symbolic_expr_lookup_product(&expression_context_, values.front());
  ASSERT_NE(retained, nullptr);
  for (auto value : values) {
    ExpectEqual(values.front(), value);
  }
  EXPECT_EQ(expression_context_.products.count, values.size());
  ASSERT_EQ(retained->factor_count, 2u);
  EXPECT_EQ(retained->factors[0], left);
  EXPECT_EQ(retained->factors[1], right);
  EXPECT_EQ(Expand(values.front()).terms, first.terms);
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
