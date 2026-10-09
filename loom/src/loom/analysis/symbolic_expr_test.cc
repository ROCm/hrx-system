// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_expr.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbolic_congruence.h"
#include "loom/analysis/symbolic_expr_test_fixture.h"
#include "loom/analysis/symbolic_projection.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/target/facts.h"

namespace loom {
namespace {

TEST_F(SymbolicExprTest, CloneRetainsTermsAndPeriodicProof) {
  const auto value = DefineI64Value();
  const auto periodic_value = DefineI64Value();
  loom_symbolic_expr_t expression;
  loom_symbolic_expr_t periodic;
  IREE_ASSERT_OK(
      loom_symbolic_expr_from_value(&expression_context_, value, &expression));
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_,
                                               periodic_value, &periodic));
  IREE_ASSERT_OK(loom_symbolic_congruence_restrict(
      &expression_context_, &periodic, 256, &expression));
  loom_symbolic_expr_t retained;
  IREE_ASSERT_OK(
      loom_symbolic_expr_clone(&expression, &module_->arena, &retained));
  ASSERT_NE(retained.terms, expression.terms);
  ASSERT_NE(retained.congruence, expression.congruence);
  ASSERT_NE(retained.congruence->expression.terms,
            expression.congruence->expression.terms);
  iree_arena_reset(&analysis_arena_);
  ASSERT_EQ(retained.term_count, 1);
  EXPECT_EQ(retained.terms[0].value_id, value);
  EXPECT_EQ(retained.congruence->modulus, 256);
  ASSERT_EQ(retained.congruence->expression.term_count, 1);
  EXPECT_EQ(retained.congruence->expression.terms[0].value_id, periodic_value);
}

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
  EXPECT_LT(expression_context_.memo.capacity, 64u);

  loom_symbolic_expr_context_reset(&expression_context_);
  loom_symbolic_expr_summary_t summary = {};
  EXPECT_FALSE(loom_symbolic_expr_context_try_lookup_summary(
      &expression_context_, value_id, &summary));
  loom_local_value_domain_release(&value_domain);
}

TEST_F(SymbolicExprTest, BorrowedDomainOutlivesTemporaryQueryStorage) {
  const loom_value_id_t initial =
      loom_index_constant_result(BuildIndexConstant(7));
  loom_local_value_domain_t value_domain = {};
  IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
      module_, module_->body, &analysis_arena_, &value_domain));
  std::vector<loom_value_id_t> values = {initial};
  for (uint32_t round = 0; round < 2; ++round) {
    SCOPED_TRACE(round);
    if (round != 0) {
      loom_local_value_domain_restore(&value_domain);
    }
    const auto initial_capacity = value_domain.value_capacity;
    iree_arena_allocator_t query_arena;
    iree_arena_initialize(&block_pool_, &query_arena);
    loom_symbolic_expr_context_t query = {};
    loom_symbolic_expr_context_initialize(module_, &value_domain, &fact_table_,
                                          &query_arena, &query);
    for (iree_host_size_t i = 0; i <= initial_capacity; ++i) {
      const loom_value_id_t value =
          loom_index_constant_result(BuildIndexConstant(42 + i));
      values.push_back(value);
      loom_symbolic_expr_t expression = {};
      IREE_EXPECT_OK(loom_symbolic_expr_from_value(&query, value, &expression));
      EXPECT_TRUE(loom_symbolic_expr_is_constant(&expression));
      EXPECT_EQ(expression.constant, 42 + i);
    }
    EXPECT_GT(value_domain.value_capacity, initial_capacity);
    iree_arena_deinitialize(&query_arena);
    iree_arena_block_pool_trim(&block_pool_);

    EXPECT_EQ(value_domain.definition_count, 1u);
    EXPECT_EQ(value_domain.value_count, values.size());
    for (loom_value_ordinal_t i = 0; i < values.size(); ++i) {
      EXPECT_EQ(loom_local_value_domain_ordinal(&value_domain, values[i]), i);
      EXPECT_EQ(value_domain.value_ids[i], values[i]);
    }
    loom_local_value_domain_release(&value_domain);
    for (auto value : values) {
      EXPECT_EQ(loom_value_u32_scratch_load(&module_->scratch.values, value),
                LOOM_VALUE_ORDINAL_INVALID);
    }
  }
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

TEST_F(SymbolicExprTest, FactIdentityExpansionPreservesOrdinalRelations) {
  const loom_value_id_t left = DefineIndexValue();
  const loom_value_id_t right = DefineIndexValue();
  const loom_value_id_t values[] = {left, right};
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const loom_type_t result_types[] = {index_type, index_type};
  const loom_predicate_t predicate = {
      .kind = LOOM_PREDICATE_LT,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
      .reserved = {},
      .args = {left, right},
  };
  loom_op_t* assertion_op = nullptr;
  IREE_ASSERT_OK(loom_sanitizer_assert_value_build(
      &builder_, values, IREE_ARRAYSIZE(values), &predicate, 1, result_types,
      IREE_ARRAYSIZE(result_types), LOOM_LOCATION_UNKNOWN, &assertion_op));
  const loom_value_slice_t checked_values =
      loom_sanitizer_assert_value_results(assertion_op);

  for (iree_host_size_t i = 0; i < checked_values.count; ++i) {
    loom_symbolic_expr_t expression = {};
    IREE_ASSERT_OK(loom_symbolic_expr_from_value(
        &expression_context_, checked_values.values[i], &expression));
    ASSERT_TRUE(loom_symbolic_expr_is_linear(&expression));
    ASSERT_EQ(expression.term_count, 1);
    EXPECT_EQ(expression.terms[0].coefficient, 1);
    EXPECT_EQ(expression.terms[0].value_id, values[i]);
    EXPECT_EQ(expression.terms[0].relation_value_id, checked_values.values[i]);
  }
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
      .kind = LOOM_PREDICATE_GE,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
      .reserved = {},
      .args = {source, 0},
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

TEST_F(SymbolicExprTest, IndexDigitsRetainNumericProofWithoutChangingTerms) {
  const loom_value_id_t input = DefineIndexValue();
  DefineFacts(input, loom_value_facts_make(0, 255, 1));
  struct {
    // Producer spelling for one exact numeric digit.
    decltype(&loom_index_div_build) build;
    // Constant right operand: divisor, shift amount, or mask.
    int64_t operand;
    // Divisor and modulus in the expected numeric function.
    loom_symbolic_projection_t expected;
  } cases[] = {
      {loom_index_div_build, 8, {input, 1, 0, 8, 0}},
      {loom_index_shrui_build, 3, {input, 1, 0, 8, 0}},
      {loom_index_shrsi_build, 3, {input, 1, 0, 8, 0}},
      {loom_index_rem_build, 8, {input, 1, 0, 1, 8}},
      {loom_index_andi_build, 7, {input, 1, 0, 1, 8}},
  };
  for (const auto& test_case : cases) {
    const auto operand =
        loom_index_constant_result(BuildIndexConstant(test_case.operand));
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(
        test_case.build(&builder_, input, operand, LOOM_LOCATION_UNKNOWN, &op));
    const auto result = loom_op_const_results(op)[0];
    loom_symbolic_expr_t expression = {};
    IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, result,
                                                 &expression));
    loom_symbolic_expr_summary_t summary = {};
    ASSERT_TRUE(loom_symbolic_expr_context_try_lookup_summary(
        &expression_context_, result, &summary));
    ASSERT_NE(summary.projection, nullptr);
    EXPECT_TRUE(loom_symbolic_projection_equal(summary.projection,
                                               &test_case.expected));
    ASSERT_EQ(expression.term_count, 1u);
    EXPECT_EQ(expression.terms[0].value_id, result);
    EXPECT_EQ(summary.materialized_dynamic_value_id, result);
  }
}

TEST_F(SymbolicExprTest, ProjectionRequiresNonnegativeBoundedInput) {
  const auto input = DefineIndexValue();
  const auto divisor = loom_index_constant_result(BuildIndexConstant(8));
  loom_op_t* quotient = nullptr;
  IREE_ASSERT_OK(loom_index_div_build(&builder_, input, divisor,
                                      LOOM_LOCATION_UNKNOWN, &quotient));
  const auto result = loom_index_div_result(quotient);
  for (const auto facts :
       {loom_value_facts_make(0, 255, 1), loom_value_facts_make(-255, 255, 1),
        loom_value_facts_unknown()}) {
    DefineFacts(input, facts);
    // Updating facts invalidates both the linear summary and its projection.
    loom_symbolic_expr_summary_t summary = {};
    EXPECT_FALSE(loom_symbolic_expr_context_try_lookup_summary(
        &expression_context_, result, &summary));
    loom_symbolic_expr_t expression = {};
    IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, result,
                                                 &expression));
    ASSERT_TRUE(loom_symbolic_expr_context_try_lookup_summary(
        &expression_context_, result, &summary));
    EXPECT_EQ(summary.projection != nullptr, facts.range_lo == 0);
  }
}

TEST_F(SymbolicExprTest, ProjectionGrowthPreservesSummariesAndMaterialization) {
  const auto input = DefineIndexValue();
  DefineFacts(input, loom_value_facts_make(0, 4095, 1));
  loom_symbolic_expr_summary_t first_summary = {};
  loom_value_id_t first_result = LOOM_VALUE_ID_INVALID;
  for (int64_t divisor = 8; divisor < 40; ++divisor) {
    const auto divisor_value =
        loom_index_constant_result(BuildIndexConstant(divisor));
    loom_op_t* quotient = nullptr;
    IREE_ASSERT_OK(loom_index_div_build(&builder_, input, divisor_value,
                                        LOOM_LOCATION_UNKNOWN, &quotient));
    const auto result = loom_index_div_result(quotient);
    loom_symbolic_expr_t expression = {};
    IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, result,
                                                 &expression));
    if (divisor == 8) {
      first_result = result;
      ASSERT_TRUE(loom_symbolic_expr_context_try_lookup_summary(
          &expression_context_, result, &first_summary));
    }
  }
  ASSERT_NE(first_summary.projection, nullptr);
  const loom_symbolic_projection_t expected = {input, 1, 0, 8, 0};
  EXPECT_TRUE(
      loom_symbolic_projection_equal(first_summary.projection, &expected));

  const auto offset = loom_index_constant_result(BuildIndexConstant(4));
  loom_op_t* shifted = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, first_result, offset,
                                      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                      LOOM_LOCATION_UNKNOWN, &shifted));
  loom_symbolic_expr_t expression = {};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, loom_index_add_result(shifted), &expression));
  loom_symbolic_expr_summary_t summary = {};
  ASSERT_TRUE(loom_symbolic_expr_context_try_lookup_summary(
      &expression_context_, loom_index_add_result(shifted), &summary));
  EXPECT_EQ(summary.materialized_dynamic_value_id, first_result);
  EXPECT_EQ(summary.expression.constant, 4);
  EXPECT_EQ(summary.projection, nullptr);

  loom_symbolic_expr_context_reset(&expression_context_);
  EXPECT_FALSE(loom_symbolic_expr_context_try_lookup_summary(
      &expression_context_, first_result, &summary));
}

TEST_F(SymbolicExprTest,
       ComposedDigitRetainsAffineInputAndNumericCastBoundary) {
  auto constant = [&](int64_t value) {
    loom_op_t* op = BuildIndexConstant(value);
    ComputeFacts(op);
    return loom_index_constant_result(op);
  };
  const auto input = DefineIndexValue();
  DefineFacts(input, loom_value_facts_make(0, 4095, 1));
  const auto scale = constant(3);
  const auto offset = constant(7);
  loom_op_t* affine = nullptr;
  IREE_ASSERT_OK(loom_index_madd_build(&builder_, input, scale, offset,
                                       LOOM_LOCATION_UNKNOWN, &affine));
  ComputeFacts(affine);
  const auto eight = constant(8);
  loom_op_t* quotient = nullptr;
  IREE_ASSERT_OK(loom_index_div_build(&builder_, loom_index_madd_result(affine),
                                      eight, LOOM_LOCATION_UNKNOWN, &quotient));
  ComputeFacts(quotient);
  const auto modulus = constant(512);
  loom_op_t* remainder = nullptr;
  IREE_ASSERT_OK(loom_index_rem_build(&builder_,
                                      loom_index_div_result(quotient), modulus,
                                      LOOM_LOCATION_UNKNOWN, &remainder));
  ComputeFacts(remainder);
  for (auto type : {LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I8}) {
    loom_op_t* cast = nullptr;
    IREE_ASSERT_OK(loom_index_cast_build(
        &builder_, loom_index_rem_result(remainder),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), loom_type_scalar(type),
        LOOM_LOCATION_UNKNOWN, &cast));
    ComputeFacts(cast);
    const auto result = loom_index_cast_result(cast);
    loom_symbolic_expr_t expression = {};
    IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, result,
                                                 &expression));
    loom_symbolic_expr_summary_t summary = {};
    ASSERT_TRUE(loom_symbolic_expr_context_try_lookup_summary(
        &expression_context_, result, &summary));
    if (type == LOOM_SCALAR_TYPE_I8) {
      EXPECT_EQ(summary.projection, nullptr);
    } else {
      ASSERT_NE(summary.projection, nullptr);
      const loom_symbolic_projection_t expected = {input, 3, 7, 8, 512};
      EXPECT_TRUE(
          loom_symbolic_projection_equal(summary.projection, &expected));
    }
  }
}

TEST_F(SymbolicExprTest, ProjectionRequiresRepresentableConstantArithmetic) {
  loom_target_facts_t target_facts = {};
  target_facts.storage.snapshot.index_bitwidth = 32;
  fact_table_.context.target_facts = &target_facts;
  const auto input = DefineIndexValue();
  const auto right = DefineIndexValue();
  struct {
    // Operation whose arithmetic cannot be retained as one exact digit.
    decltype(&loom_index_div_build) build;
    // Dividend facts, interpreted in the target's index carrier.
    loom_value_facts_t input_facts;
    // Right operand facts: a runtime divisor or an invalid shift count.
    loom_value_facts_t right_facts;
  } cases[] = {
      {loom_index_div_build, loom_value_facts_make(0, 255, 1),
       loom_value_facts_make(1, 8, 1)},
      {loom_index_div_build, loom_value_facts_make(0, INT64_C(1) << 40, 1),
       loom_value_facts_exact_i64(8)},
      {loom_index_shrui_build, loom_value_facts_make(0, 255, 1),
       loom_value_facts_exact_i64(32)},
  };
  for (const auto& test_case : cases) {
    DefineFacts(input, test_case.input_facts);
    DefineFacts(right, test_case.right_facts);
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(
        test_case.build(&builder_, input, right, LOOM_LOCATION_UNKNOWN, &op));
    const auto result = loom_op_const_results(op)[0];
    loom_symbolic_expr_t expression = {};
    IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, result,
                                                 &expression));
    loom_symbolic_expr_summary_t summary = {};
    ASSERT_TRUE(loom_symbolic_expr_context_try_lookup_summary(
        &expression_context_, result, &summary));
    EXPECT_EQ(summary.projection, nullptr);
  }
}

TEST_F(SymbolicExprTest, ScalarDigitsRespectSignedInterpretation) {
  const auto type = loom_type_scalar(LOOM_SCALAR_TYPE_I8);
  loom_value_id_t input;
  IREE_ASSERT_OK(loom_builder_define_value(&builder_, type, &input));
  struct {
    // Signed or unsigned producer of a quotient, shift, or remainder.
    decltype(&loom_scalar_divsi_build) build;
    // Constant divisor or shift count.
    int64_t operand;
  } cases[] = {{loom_scalar_divsi_build, 8}, {loom_scalar_divui_build, 8},
               {loom_scalar_shrsi_build, 3}, {loom_scalar_shrui_build, 3},
               {loom_scalar_remsi_build, 8}, {loom_scalar_remui_build, 8}};
  for (const auto& test_case : cases) {
    loom_op_t* constant = nullptr;
    IREE_ASSERT_OK(
        loom_scalar_constant_build(&builder_, loom_attr_i64(test_case.operand),
                                   type, LOOM_LOCATION_UNKNOWN, &constant));
    ComputeFacts(constant);
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(test_case.build(&builder_, input,
                                   loom_scalar_constant_result(constant), type,
                                   LOOM_LOCATION_UNKNOWN, &op));
    for (int64_t lower : {0, -128}) {
      DefineFacts(input, loom_value_facts_make(lower, 127, 1));
      ComputeFacts(op);
      const auto result = loom_op_const_results(op)[0];
      loom_symbolic_expr_t expression = {};
      IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, result,
                                                   &expression));
      loom_symbolic_expr_summary_t summary = {};
      ASSERT_TRUE(loom_symbolic_expr_context_try_lookup_summary(
          &expression_context_, result, &summary));
      EXPECT_EQ(summary.projection != nullptr, lower == 0);
    }
  }
}

TEST_F(SymbolicExprTest, SignedAndUnsignedRemaindersHaveDifferentPeriods) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I8);
  loom_value_id_t input;
  IREE_ASSERT_OK(loom_builder_define_value(&builder_, type, &input));
  DefineFacts(input, loom_value_facts_make(-128, 127, 1));
  loom_op_t* divisor;
  IREE_ASSERT_OK(loom_scalar_constant_build(&builder_, loom_attr_i64(3), type,
                                            LOOM_LOCATION_UNKNOWN, &divisor));
  ComputeFacts(divisor);
  const auto divisor_value = loom_scalar_constant_result(divisor);
  loom_op_t* signed_remainder;
  loom_op_t* unsigned_remainder;
  IREE_ASSERT_OK(loom_scalar_remsi_build(&builder_, input, divisor_value, type,
                                         LOOM_LOCATION_UNKNOWN,
                                         &signed_remainder));
  IREE_ASSERT_OK(loom_scalar_remui_build(&builder_, input, divisor_value, type,
                                         LOOM_LOCATION_UNKNOWN,
                                         &unsigned_remainder));
  ComputeFacts(signed_remainder);
  ComputeFacts(unsigned_remainder);
  loom_symbolic_expr_t original, signed_expression, unsigned_expression;
  IREE_ASSERT_OK(
      loom_symbolic_expr_from_value(&expression_context_, input, &original));
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, loom_scalar_remsi_result(signed_remainder),
      &signed_expression));
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, loom_scalar_remui_result(unsigned_remainder),
      &unsigned_expression));
  // Exact addressing retains the remainder itself, never the dividend.
  ASSERT_EQ(signed_expression.term_count, 1u);
  EXPECT_EQ(signed_expression.terms[0].value_id,
            loom_scalar_remsi_result(signed_remainder));
  EXPECT_TRUE(loom_symbolic_congruence_excludes_difference(&signed_expression,
                                                           &original, 1, 1));
  // At input -1, unsigned i8 remainder is 255%3 = 0, so the difference is 1.
  EXPECT_FALSE(loom_symbolic_congruence_excludes_difference(
      &unsigned_expression, &original, 1, 1));
}

TEST_F(SymbolicExprTest, WrappedAdditionRetainsOnlyAModularGuarantee) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I8);
  loom_value_id_t input;
  IREE_ASSERT_OK(loom_builder_define_value(&builder_, type, &input));
  DefineFacts(input, loom_value_facts_make(-128, 127, 1));
  loom_op_t* one;
  IREE_ASSERT_OK(loom_scalar_constant_build(&builder_, loom_attr_i64(1), type,
                                            LOOM_LOCATION_UNKNOWN, &one));
  ComputeFacts(one);
  loom_op_t* added;
  IREE_ASSERT_OK(loom_scalar_addi_build(&builder_, 0, input,
                                        loom_scalar_constant_result(one), type,
                                        LOOM_LOCATION_UNKNOWN, &added));
  ComputeFacts(added);
  loom_symbolic_expr_t original, wrapped;
  IREE_ASSERT_OK(
      loom_symbolic_expr_from_value(&expression_context_, input, &original));
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, loom_scalar_addi_result(added), &wrapped));
  ASSERT_EQ(wrapped.term_count, 1u);
  EXPECT_EQ(wrapped.terms[0].value_id, loom_scalar_addi_result(added));
  EXPECT_TRUE(
      loom_symbolic_congruence_excludes_difference(&wrapped, &original, 0, 0));
  // 127+1 wraps to -128: the difference can be -255, as well as 1.
  EXPECT_FALSE(loom_symbolic_congruence_excludes_difference(&wrapped, &original,
                                                            -255, -255));
  EXPECT_FALSE(
      loom_symbolic_congruence_excludes_difference(&wrapped, &original, 1, 1));
}

class SymbolicExprStorageTest : public SymbolicExprTest {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<SymbolicExprStorageTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        test->allocation_count_++ == test->failure_index_) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected symbolic memo allocation failure");
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  void InitializeStorage(iree_host_size_t block_size = 128 * 1024) {
    allocation_count_ = 0;
    failure_index_ = SIZE_MAX;
    iree_arena_block_pool_initialize(block_size, {this, Allocate}, &memo_pool_);
    iree_arena_initialize(&memo_pool_, &memo_arena_);
    loom_symbolic_expr_context_initialize(module_, nullptr, &fact_table_,
                                          &memo_arena_, &expression_context_);
  }

  void DeinitializeStorage() {
    iree_arena_deinitialize(&memo_arena_);
    iree_arena_block_pool_deinitialize(&memo_pool_);
  }

  void SetUp() override {
    SymbolicExprTest::SetUp();
    InitializeStorage();
  }

  void TearDown() override {
    DeinitializeStorage();
    SymbolicExprTest::TearDown();
  }

  void BuildConstants(uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
      values_.push_back(loom_index_constant_result(BuildIndexConstant(i)));
    }
  }

  void ExpectConstant(uint32_t index) {
    loom_symbolic_expr_t expression = {};
    IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_,
                                                 values_[index], &expression));
    EXPECT_TRUE(loom_symbolic_expr_is_constant(&expression));
    EXPECT_EQ(expression.constant, index);
    loom_symbolic_expr_summary_t summary = {};
    ASSERT_TRUE(loom_symbolic_expr_context_try_lookup_summary(
        &expression_context_, values_[index], &summary));
    EXPECT_TRUE(loom_symbolic_expr_is_constant(&summary.expression));
    EXPECT_EQ(summary.expression.constant, index);
  }

  void ExpectNoOversizedAllocations() {
    iree_arena_block_pool_statistics_t statistics = {};
    iree_arena_block_pool_query_statistics(&memo_pool_, &statistics);
    EXPECT_EQ(statistics.oversized_allocation_count, 0u);
  }

  // Real constant producer results, allocated outside observed memo storage.
  std::vector<loom_value_id_t> values_;
  // Pool observing only expression reconstruction and its temporary arenas.
  iree_arena_block_pool_t memo_pool_ = {};
  // Memo and returned expression payload lifetime.
  iree_arena_allocator_t memo_arena_ = {};
  // Attempted backing allocations, excluding frees.
  iree_host_size_t allocation_count_ = 0;
  // Backing allocation to fail, or SIZE_MAX for normal execution.
  iree_host_size_t failure_index_ = SIZE_MAX;
};

TEST_F(SymbolicExprStorageTest, SparseLocalOrdinalsOnlyAllocateLivePayloads) {
  BuildConstants(2050);
  loom_local_value_domain_t domain = {};
  IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
      module_, module_->body, &analysis_arena_, &domain));
  ASSERT_GE(domain.value_count, 2050u);
  loom_symbolic_expr_context_initialize(module_, &domain, &fact_table_,
                                        &memo_arena_, &expression_context_);
  ASSERT_NO_FATAL_FAILURE(ExpectConstant(2049));
  ExpectNoOversizedAllocations();
  const auto allocations = allocation_count_;
  loom_symbolic_expr_context_reset(&expression_context_);
  ASSERT_NO_FATAL_FAILURE(ExpectConstant(0));
  loom_symbolic_expr_summary_t summary = {};
  EXPECT_FALSE(loom_symbolic_expr_context_try_lookup_summary(
      &expression_context_, values_.back(), &summary));
  ASSERT_NO_FATAL_FAILURE(ExpectConstant(2049));
  EXPECT_EQ(allocation_count_, allocations);
  loom_local_value_domain_release(&domain);
}

class SymbolicExprStorageBoundaryTest
    : public SymbolicExprStorageTest,
      public ::testing::WithParamInterface<uint32_t> {};

TEST_P(SymbolicExprStorageBoundaryTest, ResetReusesPayloadAndInvalidatesKeys) {
  const uint32_t count = GetParam();
  BuildConstants(count + 1);
  for (uint32_t i = 0; i < count; ++i) {
    ASSERT_NO_FATAL_FAILURE(ExpectConstant(i));
  }
  ExpectNoOversizedAllocations();
  const auto allocations = allocation_count_;
  for (int epoch = 0; epoch < 3; ++epoch) {
    loom_symbolic_expr_context_reset(&expression_context_);
    loom_symbolic_expr_summary_t summary = {};
    for (const auto value : values_) {
      EXPECT_FALSE(loom_symbolic_expr_context_try_lookup_summary(
          &expression_context_, value, &summary));
    }
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t index = epoch % 2 == 0 ? count - i - 1 : i;
      ASSERT_NO_FATAL_FAILURE(ExpectConstant(index));
    }
    EXPECT_EQ(allocation_count_, allocations);
  }
}

INSTANTIATE_TEST_SUITE_P(MemoChunks, SymbolicExprStorageBoundaryTest,
                         ::testing::Values(0u, 1u, 31u, 32u, 33u, 511u, 512u,
                                           513u, 2049u));

TEST_F(SymbolicExprStorageTest, LivePayloadFitsSmallPoolBlocks) {
  BuildConstants(257);
  DeinitializeStorage();
  InitializeStorage(4096);
  ASSERT_NO_FATAL_FAILURE(ExpectConstant(256));
  for (uint32_t i = 0; i < 256; ++i) {
    ASSERT_NO_FATAL_FAILURE(ExpectConstant(i));
  }
  ExpectNoOversizedAllocations();
}

TEST_F(SymbolicExprStorageTest, ExpansionFailureCanBeRetriedWithoutReset) {
  const auto source = DefineI64Value();
  DefineFacts(source, loom_value_facts_make(-1024, 1024, 1));
  auto value = source;
  for (int i = 0; i < 256; ++i) {
    loom_op_t* negate = nullptr;
    IREE_ASSERT_OK(loom_scalar_negi_build(
        &builder_, value, loom_type_scalar(LOOM_SCALAR_TYPE_I64),
        LOOM_LOCATION_UNKNOWN, &negate));
    value = loom_scalar_negi_result(negate);
  }
  DeinitializeStorage();
  InitializeStorage(4096);
  loom_symbolic_expr_t expression = {};
  IREE_ASSERT_OK(
      loom_symbolic_expr_from_value(&expression_context_, value, &expression));
  const auto allocations = allocation_count_;
  ASSERT_GT(allocations, 1u);
  for (iree_host_size_t i = 0; i < allocations; ++i) {
    SCOPED_TRACE(i);
    DeinitializeStorage();
    InitializeStorage(4096);
    failure_index_ = i;
    IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          loom_symbolic_expr_from_value(&expression_context_,
                                                        value, &expression));
    EXPECT_EQ(allocation_count_, i + 1);
    failure_index_ = SIZE_MAX;
    IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, value,
                                                 &expression));
    ASSERT_EQ(expression.term_count, 1u);
    EXPECT_EQ(expression.terms[0].value_id, source);
    EXPECT_EQ(expression.terms[0].coefficient, 1);
  }
}

}  // namespace
}  // namespace loom
