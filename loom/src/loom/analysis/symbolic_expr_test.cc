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
#include "loom/ops/op_defs.h"
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

TEST_F(SymbolicExprTest, LocalDomainRegistersHighIdsCreatedAfterAcquisition) {
  for (iree_host_size_t i = 0; i < 4096; ++i) {
    DefineIndexValue();
  }

  loom_local_value_domain_t value_domain = {};
  IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
      module_, module_->body, &analysis_arena_, &value_domain));
  loom_symbolic_expr_context_initialize(module_, &value_domain, &fact_table_,
                                        &analysis_arena_, &expression_context_);

  loom_op_t* constant_op = BuildIndexConstant(42);
  const loom_value_id_t value_id = loom_index_constant_result(constant_op);
  ASSERT_GE(value_id, 4096u);

  loom_symbolic_expr_t expression = {};
  IREE_EXPECT_OK(loom_symbolic_expr_from_value(&expression_context_, value_id,
                                               &expression));
  EXPECT_TRUE(loom_symbolic_expr_is_constant(&expression));
  EXPECT_EQ(expression.constant, 42);
  EXPECT_NE(loom_local_value_domain_try_ordinal(&value_domain, value_id),
            LOOM_VALUE_ORDINAL_INVALID);
  EXPECT_LT(expression_context_.memo_capacity, 64u);

  loom_symbolic_expr_context_reset(&expression_context_);
  loom_symbolic_expr_summary_t summary = {};
  EXPECT_FALSE(loom_symbolic_expr_context_try_lookup_summary(
      &expression_context_, value_id, &summary));
  loom_local_value_domain_release(&value_domain);
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

TEST_F(SymbolicExprTest, IndexCastsExpandOnlyWhenTheyPreserveNumericValue) {
  struct {
    // Source representation interpreted by the cast.
    loom_scalar_type_t input_type;
    // Destination representation interpreted by the cast.
    loom_scalar_type_t result_type;
    // Inclusive lower bound of the proven source range.
    int64_t lower_bound;
    // Inclusive upper bound of the proven source range.
    int64_t upper_bound;
    // Whether the cast is a numeric identity over that range.
    bool preserves_value;
  } cases[] = {
      {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_I8, 4294967297, 4294967299,
       false},
      {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_I8, -128, 127, true},
      {LOOM_SCALAR_TYPE_OFFSET, LOOM_SCALAR_TYPE_I32, 2147483648, 4294967295,
       false},
      {LOOM_SCALAR_TYPE_OFFSET, LOOM_SCALAR_TYPE_I32, 0, INT32_MAX, true},
      {LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_OFFSET, -128, -1, false},
      {LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_OFFSET, -1, 1, false},
      {LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_OFFSET, 0, 127, true},
      {LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_INDEX, -128, 127, true},
      {LOOM_SCALAR_TYPE_I64, LOOM_SCALAR_TYPE_INDEX, INT64_MIN, INT64_MAX,
       true},
      {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_OFFSET, 0, INT64_MAX, true},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(static_cast<int>(test_case.input_type));
    SCOPED_TRACE(static_cast<int>(test_case.result_type));
    SCOPED_TRACE(test_case.lower_bound);
    loom_type_t input_type = loom_type_scalar(test_case.input_type);
    loom_value_id_t input = LOOM_VALUE_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_define_value(&builder_, input_type, &input));
    DefineFacts(input, loom_value_facts_make(test_case.lower_bound,
                                             test_case.upper_bound, 1));
    loom_op_t* cast = nullptr;
    IREE_ASSERT_OK(loom_index_cast_build(
        &builder_, input, input_type, loom_type_scalar(test_case.result_type),
        LOOM_LOCATION_UNKNOWN, &cast));
    loom_value_id_t result = loom_index_cast_result(cast);
    loom_symbolic_expr_t expression = {};
    IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, result,
                                                 &expression));
    ASSERT_TRUE(loom_symbolic_expr_is_linear(&expression));
    ASSERT_EQ(expression.term_count, 1);
    EXPECT_EQ(expression.constant, 0);
    EXPECT_EQ(expression.terms[0].coefficient, 1);
    EXPECT_EQ(expression.terms[0].value_id,
              test_case.preserves_value ? input : result);
  }
}

TEST_F(SymbolicExprTest, ScalarCastsExpandOnlyWhenTheyPreserveNumericValue) {
  struct {
    // Cast operation whose numeric interpretation is under test.
    decltype(&loom_scalar_extsi_build) build;
    // Source integer representation.
    loom_scalar_type_t input_type;
    // Destination integer representation.
    loom_scalar_type_t result_type;
    // Inclusive lower bound of the proven source range.
    int64_t lower_bound;
    // Inclusive upper bound of the proven source range.
    int64_t upper_bound;
    // Whether the cast is a numeric identity over that range.
    bool preserves_value;
  } cases[] = {
      {loom_scalar_extsi_build, LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_I32, -128,
       127, true},
      {loom_scalar_extsi_build, LOOM_SCALAR_TYPE_I1, LOOM_SCALAR_TYPE_I32, 0, 1,
       false},
      {loom_scalar_extui_build, LOOM_SCALAR_TYPE_I1, LOOM_SCALAR_TYPE_I32, 0, 1,
       true},
      {loom_scalar_extui_build, LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I64, 0,
       INT32_MAX, true},
      {loom_scalar_extui_build, LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I64, -1,
       1, false},
      {loom_scalar_extui_build, LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_I32, -128,
       -1, false},
      {loom_scalar_trunci_build, LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I8,
       -128, 127, true},
      {loom_scalar_trunci_build, LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I8,
       -128, 128, false},
      {loom_scalar_trunci_build, LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I1, 0,
       1, true},
      {loom_scalar_trunci_build, LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I1, -1,
       0, false},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(static_cast<int>(test_case.input_type));
    SCOPED_TRACE(static_cast<int>(test_case.result_type));
    SCOPED_TRACE(test_case.lower_bound);
    const loom_type_t input_type = loom_type_scalar(test_case.input_type);
    loom_value_id_t input = LOOM_VALUE_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_define_value(&builder_, input_type, &input));
    DefineFacts(input, loom_value_facts_make(test_case.lower_bound,
                                             test_case.upper_bound, 1));
    loom_op_t* cast = nullptr;
    IREE_ASSERT_OK(test_case.build(&builder_, input, input_type,
                                   loom_type_scalar(test_case.result_type),
                                   LOOM_LOCATION_UNKNOWN, &cast));
    ComputeFacts(cast);
    const loom_value_id_t result = loom_op_const_results(cast)[0];
    loom_symbolic_expr_t expression = {};
    IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, result,
                                                 &expression));
    ASSERT_TRUE(loom_symbolic_expr_is_linear(&expression));
    ASSERT_EQ(expression.term_count, 1);
    EXPECT_EQ(expression.constant, 0);
    EXPECT_EQ(expression.terms[0].coefficient, 1);
    EXPECT_EQ(expression.terms[0].value_id,
              test_case.preserves_value ? input : result);
  }
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
  DefineFacts(source, loom_value_facts_make(-1024, 1024, 1));
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

TEST_F(SymbolicExprTest, ErasedProducerChainRetainsMultiResultSemantics) {
  const loom_value_id_t first = DefineIndexValue();
  const loom_value_id_t second = DefineIndexValue();
  const loom_value_id_t inputs[] = {first, second};
  const loom_type_t result_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
  };
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(
      &builder_, inputs, IREE_ARRAYSIZE(inputs), /*predicates=*/nullptr,
      /*predicate_count=*/0, result_types, IREE_ARRAYSIZE(result_types),
      LOOM_LOCATION_UNKNOWN, &assume_op));

  loom_op_t* constant_op = BuildIndexConstant(4);
  loom_op_t* add_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(
      &builder_, loom_index_assume_results(assume_op).values[1],
      loom_index_constant_result(constant_op),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN,
      &add_op));
  const loom_value_id_t result = loom_index_add_result(add_op);

  IREE_ASSERT_OK(loom_op_erase(module_, add_op));
  IREE_ASSERT_OK(loom_op_erase(module_, assume_op));
  IREE_ASSERT_OK(loom_op_erase(module_, constant_op));
  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  loom_symbolic_expr_context_reset(&expression_context_);

  loom_symbolic_expr_t expression = {};
  IREE_ASSERT_OK(
      loom_symbolic_expr_from_value(&expression_context_, result, &expression));
  ASSERT_TRUE(loom_symbolic_expr_is_linear(&expression));
  ASSERT_EQ(expression.term_count, 1);
  EXPECT_EQ(expression.constant, 4);
  EXPECT_EQ(expression.terms[0].coefficient, 1);
  EXPECT_EQ(expression.terms[0].value_id, second);
}

TEST_F(SymbolicExprTest, ExpandsIndexMaddWithConstantMultiplier) {
  loom_value_id_t row = DefineIndexValue();
  loom_value_id_t column = DefineIndexValue();
  loom_value_id_t stride = loom_index_constant_result(BuildIndexConstant(16));
  loom_op_t* madd_op = nullptr;
  IREE_ASSERT_OK(loom_index_madd_build(&builder_, row, stride, column,
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
                                       LOOM_LOCATION_UNKNOWN, &shift_op));

  loom_symbolic_expr_t expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, loom_index_shli_result(shift_op), &expression));

  ASSERT_TRUE(loom_symbolic_expr_is_linear(&expression));
  ASSERT_EQ(expression.term_count, 1);
  EXPECT_EQ(expression.terms[0].coefficient, 8);
  EXPECT_EQ(expression.terms[0].value_id, value);
}

TEST_F(SymbolicExprTest, ScalarShiftRequiresExactAmountAndNonwrappingRange) {
  struct {
    // Fixed-width representation of both operands and the result.
    loom_scalar_type_t type;
    // Inclusive lower bound of the unshifted value.
    int64_t lower_bound;
    // Inclusive upper bound of the unshifted value.
    int64_t upper_bound;
    // Inclusive lower bound of the shift amount.
    int64_t shift_lower_bound;
    // Inclusive upper bound of the shift amount.
    int64_t shift_upper_bound;
    // Whether mathematical multiplication preserves the fixed-width result.
    bool preserves_value;
  } cases[] = {
      {LOOM_SCALAR_TYPE_I8, 0, 31, 2, 2, true},
      {LOOM_SCALAR_TYPE_I8, 0, 32, 2, 2, false},
      {LOOM_SCALAR_TYPE_I8, -32, -1, 2, 2, true},
      {LOOM_SCALAR_TYPE_I8, -33, -1, 2, 2, false},
      {LOOM_SCALAR_TYPE_I32, 0, 255, 2, 2, true},
      {LOOM_SCALAR_TYPE_I32, 0, 255, 1, 2, false},
      {LOOM_SCALAR_TYPE_I64, 0, 1, 62, 62, true},
      {LOOM_SCALAR_TYPE_I64, 0, 2, 62, 62, false},
      {LOOM_SCALAR_TYPE_I64, 0, 1, 63, 63, false},
      {LOOM_SCALAR_TYPE_I8, 0, 1, 8, 8, false},
      {LOOM_SCALAR_TYPE_I8, 0, 1, -1, -1, false},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(static_cast<int>(test_case.type));
    SCOPED_TRACE(test_case.lower_bound);
    SCOPED_TRACE(test_case.upper_bound);
    SCOPED_TRACE(test_case.shift_lower_bound);
    const loom_type_t type = loom_type_scalar(test_case.type);
    loom_value_id_t input = LOOM_VALUE_ID_INVALID;
    loom_value_id_t shift = LOOM_VALUE_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_define_value(&builder_, type, &input));
    IREE_ASSERT_OK(loom_builder_define_value(&builder_, type, &shift));
    DefineFacts(input, loom_value_facts_make(test_case.lower_bound,
                                             test_case.upper_bound, 1));
    DefineFacts(shift, loom_value_facts_make(test_case.shift_lower_bound,
                                             test_case.shift_upper_bound, 1));
    loom_op_t* shift_op = nullptr;
    IREE_ASSERT_OK(loom_scalar_shli_build(&builder_, /*instance_flags=*/0,
                                          input, shift, type,
                                          LOOM_LOCATION_UNKNOWN, &shift_op));
    ComputeFacts(shift_op);
    const loom_value_id_t result = loom_scalar_shli_result(shift_op);
    loom_symbolic_expr_t expression = {};
    IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, result,
                                                 &expression));
    ASSERT_TRUE(loom_symbolic_expr_is_linear(&expression));
    ASSERT_EQ(expression.term_count, 1);
    EXPECT_EQ(expression.constant, 0);
    EXPECT_EQ(expression.terms[0].coefficient,
              test_case.preserves_value
                  ? (int64_t{1} << test_case.shift_lower_bound)
                  : 1);
    EXPECT_EQ(expression.terms[0].value_id,
              test_case.preserves_value ? input : result);
  }
}

TEST_F(SymbolicExprTest, DynamicShiftFallsBackToResultSymbol) {
  loom_value_id_t value = DefineIndexValue();
  loom_value_id_t shift = DefineIndexValue();
  loom_op_t* shift_op = nullptr;
  IREE_ASSERT_OK(loom_index_shli_build(&builder_, value, shift,
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

TEST_F(SymbolicExprTest, SelectUsesScopedComparisonTruth) {
  loom_value_id_t compared = DefineIndexValue();
  loom_value_id_t zero = loom_index_constant_result(BuildIndexConstant(0));
  loom_op_t* condition_op = nullptr;
  IREE_ASSERT_OK(loom_index_cmp_build(&builder_, LOOM_INDEX_CMP_PREDICATE_EQ,
                                      compared, zero, LOOM_LOCATION_UNKNOWN,
                                      &condition_op));
  loom_value_id_t condition = loom_index_cmp_result(condition_op);
  loom_value_id_t true_value = DefineIndexValue();
  loom_value_id_t false_value = DefineIndexValue();
  loom_op_t* select_op = nullptr;
  IREE_ASSERT_OK(loom_scf_select_build(&builder_, condition, true_value,
                                       false_value,
                                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                       LOOM_LOCATION_UNKNOWN, &select_op));

  loom_condition_derivation_t derivation = {};
  loom_condition_derivation_initialize(&analysis_arena_, &derivation);
  loom_condition_fact_scope_t condition_scope = {};
  for (bool assumed_truth : {false, true}) {
    IREE_ASSERT_OK(loom_condition_facts_query_complete(
        &expression_context_.condition_query, &fact_table_, condition,
        assumed_truth, &derivation));
    loom_condition_fact_scope_initialize_local(nullptr, &derivation,
                                               &condition_scope);
    expression_context_.condition_scope = &condition_scope;
    loom_symbolic_expr_context_reset(&expression_context_);

    loom_symbolic_expr_t expression = {};
    IREE_ASSERT_OK(loom_symbolic_expr_from_value(
        &expression_context_, loom_scf_select_result(select_op), &expression));

    ASSERT_EQ(expression.term_count, 1);
    EXPECT_EQ(expression.terms[0].value_id,
              assumed_truth ? true_value : false_value);
  }
  expression_context_.condition_scope = nullptr;
  loom_symbolic_expr_context_reset(&expression_context_);
}

}  // namespace
}  // namespace loom
