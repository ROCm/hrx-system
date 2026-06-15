// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_expr.h"

#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class SymbolicExprTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);

    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t index_vtable_count = 0;
    const loom_op_vtable_t* const* index_vtables =
        loom_index_dialect_vtables(&index_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_INDEX,
                                                 index_vtables,
                                                 (uint16_t)index_vtable_count));
    iree_host_size_t scalar_vtable_count = 0;
    const loom_op_vtable_t* const* scalar_vtables =
        loom_scalar_dialect_vtables(&scalar_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SCALAR, scalar_vtables,
        (uint16_t)scalar_vtable_count));
    iree_host_size_t scf_vtable_count = 0;
    const loom_op_vtable_t* const* scf_vtables =
        loom_scf_dialect_vtables(&scf_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SCF, scf_vtables, (uint16_t)scf_vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    IREE_ASSERT_OK(
        loom_value_fact_table_initialize(&fact_table_, &analysis_arena_, 16));
    loom_symbolic_expr_context_initialize(
        module_, &fact_table_, &analysis_arena_, &expression_context_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_value_id_t DefineIndexValue() {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_value(
        &builder_, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &value_id));
    return value_id;
  }

  loom_value_id_t DefineI64Value() {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_value(
        &builder_, loom_type_scalar(LOOM_SCALAR_TYPE_I64), &value_id));
    return value_id;
  }

  void DefineFacts(loom_value_id_t value_id, loom_value_facts_t facts) {
    IREE_CHECK_OK(loom_value_fact_table_define(&fact_table_, value_id, facts));
    loom_symbolic_expr_context_reset(&expression_context_);
  }

  loom_op_t* BuildIndexConstant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
  loom_value_fact_table_t fact_table_;
  loom_symbolic_expr_context_t expression_context_;
};

TEST_F(SymbolicExprTest, UnknownValueIsMemoizedLinearTerm) {
  loom_value_id_t value_id = DefineIndexValue();

  loom_symbolic_expr_t first_expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, value_id,
                                               &first_expression));
  EXPECT_TRUE(loom_symbolic_expr_is_linear(&first_expression));
  ASSERT_EQ(first_expression.term_count, 1);
  EXPECT_EQ(first_expression.terms[0].coefficient, 1);
  EXPECT_EQ(first_expression.terms[0].value_id, value_id);

  loom_symbolic_expr_t second_expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, value_id,
                                               &second_expression));
  EXPECT_EQ(second_expression.terms, first_expression.terms);
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

TEST_F(SymbolicExprTest, AddSubNormalizeAndCancelTerms) {
  loom_value_id_t value_id = DefineIndexValue();
  loom_symbolic_expr_t value_expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, value_id,
                                               &value_expression));

  loom_symbolic_expr_t four = {0};
  loom_symbolic_expr_constant(4, &four);
  loom_symbolic_expr_t eight = {0};
  loom_symbolic_expr_constant(8, &eight);

  loom_symbolic_expr_t value_plus_four = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_add(&expression_context_, &value_expression,
                                        &four, &value_plus_four));
  loom_symbolic_expr_t value_plus_eight = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_add(&expression_context_, &value_expression,
                                        &eight, &value_plus_eight));

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_le(
      &expression_context_, &value_plus_four, &value_plus_eight, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);

  IREE_ASSERT_OK(loom_symbolic_expr_prove_le(
      &expression_context_, &value_plus_eight, &value_plus_four, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_FALSE);
}

TEST_F(SymbolicExprTest, SimplifiesDifferenceToExistingValue) {
  loom_value_id_t row = DefineIndexValue();
  loom_value_id_t column = DefineIndexValue();
  loom_value_id_t stride = loom_index_constant_result(BuildIndexConstant(16));
  loom_op_t* scaled_op = nullptr;
  IREE_ASSERT_OK(loom_index_mul_build(&builder_, row, stride,
                                      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                      LOOM_LOCATION_UNKNOWN, &scaled_op));
  loom_op_t* address_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_,
                                      loom_index_mul_result(scaled_op), column,
                                      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                      LOOM_LOCATION_UNKNOWN, &address_op));

  loom_symbolic_value_difference_t difference = {};
  IREE_ASSERT_OK(loom_symbolic_expr_simplify_value_difference(
      &expression_context_, loom_index_add_result(address_op),
      loom_index_mul_result(scaled_op), &difference));

  EXPECT_EQ(difference.kind, LOOM_SYMBOLIC_VALUE_DIFFERENCE_VALUE);
  EXPECT_EQ(difference.value_id, column);
}

TEST_F(SymbolicExprTest, SimplifiesDifferenceToConstant) {
  loom_value_id_t value = DefineIndexValue();
  loom_value_id_t eight = loom_index_constant_result(BuildIndexConstant(8));
  loom_value_id_t twenty_four =
      loom_index_constant_result(BuildIndexConstant(24));
  loom_op_t* left_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, value, twenty_four,
                                      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                      LOOM_LOCATION_UNKNOWN, &left_op));
  loom_op_t* right_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, value, eight,
                                      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                      LOOM_LOCATION_UNKNOWN, &right_op));

  loom_symbolic_value_difference_t difference = {};
  IREE_ASSERT_OK(loom_symbolic_expr_simplify_value_difference(
      &expression_context_, loom_index_add_result(left_op),
      loom_index_add_result(right_op), &difference));

  EXPECT_EQ(difference.kind, LOOM_SYMBOLIC_VALUE_DIFFERENCE_CONSTANT);
  EXPECT_EQ(difference.constant, 16);
}

TEST_F(SymbolicExprTest, ProvesRelationsThroughSymbolicCancellation) {
  loom_value_id_t value = DefineIndexValue();
  loom_value_id_t four = loom_index_constant_result(BuildIndexConstant(4));
  loom_value_id_t eight = loom_index_constant_result(BuildIndexConstant(8));
  loom_op_t* value_plus_four_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(
      &builder_, value, four, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      LOOM_LOCATION_UNKNOWN, &value_plus_four_op));
  loom_op_t* value_plus_eight_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(
      &builder_, value, eight, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      LOOM_LOCATION_UNKNOWN, &value_plus_eight_op));

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LT,
      loom_index_add_result(value_plus_four_op),
      loom_index_add_result(value_plus_eight_op), &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);

  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_EQ,
      loom_index_add_result(value_plus_four_op),
      loom_index_add_result(value_plus_eight_op), &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_FALSE);
}

TEST_F(SymbolicExprTest, AssumedValueRelationPredicatesProveRelations) {
  loom_value_id_t induction = DefineIndexValue();
  loom_value_id_t upper_bound = DefineIndexValue();
  loom_predicate_t predicate = {
      /*.kind=*/LOOM_PREDICATE_LT,
      /*.arg_count=*/2,
      /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
      /*.reserved=*/{},
      /*.args=*/{induction, upper_bound},
  };
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(&builder_, &induction, 1, &predicate,
                                         1, &index_type, 1,
                                         LOOM_LOCATION_UNKNOWN, &assume_op));
  loom_value_id_t assumed_induction =
      loom_index_assume_results(assume_op).values[0];

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LT,
      assumed_induction, upper_bound, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);

  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_GE,
      assumed_induction, upper_bound, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_FALSE);

  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_EQ,
      assumed_induction, upper_bound, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_FALSE);

  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_NE,
      assumed_induction, upper_bound, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
}

TEST_F(SymbolicExprTest, ScaledStrictRelationProvesLessEqualWithUnitExtent) {
  loom_value_id_t induction = DefineIndexValue();
  loom_value_id_t upper_bound = DefineIndexValue();
  loom_predicate_t predicate = {
      /*.kind=*/LOOM_PREDICATE_LT,
      /*.arg_count=*/2,
      /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
      /*.reserved=*/{},
      /*.args=*/{induction, upper_bound},
  };
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(&builder_, &induction, 1, &predicate,
                                         1, &index_type, 1,
                                         LOOM_LOCATION_UNKNOWN, &assume_op));
  loom_value_id_t assumed_induction =
      loom_index_assume_results(assume_op).values[0];

  loom_value_id_t four = loom_index_constant_result(BuildIndexConstant(4));
  loom_op_t* scaled_induction_op = nullptr;
  IREE_ASSERT_OK(loom_index_mul_build(&builder_, assumed_induction, four,
                                      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                      LOOM_LOCATION_UNKNOWN,
                                      &scaled_induction_op));
  loom_op_t* scaled_bound_op = nullptr;
  IREE_ASSERT_OK(loom_index_mul_build(&builder_, upper_bound, four,
                                      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                      LOOM_LOCATION_UNKNOWN, &scaled_bound_op));

  loom_symbolic_expr_t scaled_induction = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, loom_index_mul_result(scaled_induction_op),
      &scaled_induction));
  loom_symbolic_expr_t unit_extent = {0};
  loom_symbolic_expr_constant(4, &unit_extent);
  loom_symbolic_expr_t exclusive_end = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_add(&expression_context_, &scaled_induction,
                                        &unit_extent, &exclusive_end));
  loom_symbolic_expr_t scaled_bound = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, loom_index_mul_result(scaled_bound_op),
      &scaled_bound));

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_le(
      &expression_context_, &exclusive_end, &scaled_bound, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
}

TEST_F(SymbolicExprTest, AssumedRightValueRelationPredicatesAreSwapped) {
  loom_value_id_t induction = DefineIndexValue();
  loom_value_id_t upper_bound = DefineIndexValue();
  loom_predicate_t predicate = {
      /*.kind=*/LOOM_PREDICATE_GT,
      /*.arg_count=*/2,
      /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
      /*.reserved=*/{},
      /*.args=*/{upper_bound, induction},
  };
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(&builder_, &upper_bound, 1, &predicate,
                                         1, &index_type, 1,
                                         LOOM_LOCATION_UNKNOWN, &assume_op));
  loom_value_id_t assumed_upper_bound =
      loom_index_assume_results(assume_op).values[0];

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LT, induction,
      assumed_upper_bound, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
}

TEST_F(SymbolicExprTest, ScalarAssumePredicatesProveRelations) {
  loom_value_id_t element = DefineI64Value();
  loom_value_id_t bound = DefineI64Value();
  loom_predicate_t predicate = {
      /*.kind=*/LOOM_PREDICATE_LE,
      /*.arg_count=*/2,
      /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
      /*.reserved=*/{},
      /*.args=*/{element, bound},
  };
  loom_type_t i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_scalar_assume_build(&builder_, &element, 1, &predicate, 1,
                                          &i64_type, 1, LOOM_LOCATION_UNKNOWN,
                                          &assume_op));
  loom_value_id_t assumed_element =
      loom_scalar_assume_results(assume_op).values[0];

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_GT, assumed_element,
      bound, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_FALSE);
}

TEST_F(SymbolicExprTest, MemoTableGrowthPreservesOuterExpansion) {
  loom_value_id_t old_element = DefineI64Value();
  loom_value_id_t bound = DefineI64Value();
  loom_predicate_t predicate = {
      /*.kind=*/LOOM_PREDICATE_LT,
      /*.arg_count=*/2,
      /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
      /*.reserved=*/{},
      /*.args=*/{old_element, bound},
  };
  loom_type_t i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  loom_type_t result_types[] = {i64_type, i64_type};
  loom_value_id_t values[] = {old_element, bound};
  loom_op_t* bounded_op = nullptr;
  IREE_ASSERT_OK(loom_scalar_assume_build(
      &builder_, values, IREE_ARRAYSIZE(values), &predicate, 1, result_types,
      IREE_ARRAYSIZE(result_types), LOOM_LOCATION_UNKNOWN, &bounded_op));
  loom_value_id_t bounded_element =
      loom_scalar_assume_results(bounded_op).values[0];

  loom_predicate_t nonnegative_predicate = {
      /*.kind=*/LOOM_PREDICATE_GE,
      /*.arg_count=*/2,
      /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
      /*.reserved=*/{},
      /*.args=*/{bounded_element, 0},
  };
  loom_op_t* nonnegative_op = nullptr;
  IREE_ASSERT_OK(loom_scalar_assume_build(
      &builder_, &bounded_element, 1, &nonnegative_predicate, 1, &i64_type, 1,
      LOOM_LOCATION_UNKNOWN, &nonnegative_op));
  loom_value_id_t nonnegative_element =
      loom_scalar_assume_results(nonnegative_op).values[0];

  loom_op_t* cast_op = nullptr;
  IREE_ASSERT_OK(loom_index_cast_build(&builder_, nonnegative_element, i64_type,
                                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                       LOOM_LOCATION_UNKNOWN, &cast_op));
  loom_value_id_t origin = loom_index_cast_result(cast_op);

  loom_value_id_t new_element = DefineI64Value();
  ASSERT_GT(new_element, origin);
  IREE_ASSERT_OK(
      loom_value_replace_all_uses_with(module_, old_element, new_element));

  loom_symbolic_expr_t origin_expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, origin,
                                               &origin_expression));
  ASSERT_TRUE(loom_symbolic_expr_is_linear(&origin_expression));
  ASSERT_EQ(origin_expression.term_count, 1);
  EXPECT_EQ(origin_expression.terms[0].value_id, new_element);

  loom_symbolic_expr_t origin_again = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, origin,
                                               &origin_again));
  ASSERT_TRUE(loom_symbolic_expr_is_linear(&origin_again));
  ASSERT_EQ(origin_again.term_count, 1);
  EXPECT_EQ(origin_again.terms[0].value_id, new_element);

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LT, origin, bound,
      &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
}

TEST_F(SymbolicExprTest, TermsAreSortedByValueId) {
  loom_value_id_t first_value = DefineIndexValue();
  loom_value_id_t second_value = DefineIndexValue();
  loom_symbolic_expr_t first_expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_,
                                               first_value, &first_expression));
  loom_symbolic_expr_t second_expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, second_value, &second_expression));

  loom_symbolic_expr_t expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_add(&expression_context_,
                                        &second_expression, &first_expression,
                                        &expression));

  ASSERT_EQ(expression.term_count, 2);
  EXPECT_EQ(expression.terms[0].value_id, first_value);
  EXPECT_EQ(expression.terms[1].value_id, second_value);
}

TEST_F(SymbolicExprTest, ExpandsIndexMaddWithConstantMultiplier) {
  loom_value_id_t row = DefineIndexValue();
  loom_value_id_t column = DefineIndexValue();
  loom_op_t* stride_op = BuildIndexConstant(16);
  loom_value_id_t stride = loom_index_constant_result(stride_op);
  loom_op_t* madd_op = nullptr;
  IREE_ASSERT_OK(loom_index_madd_build(&builder_, row, stride, column,
                                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                       LOOM_LOCATION_UNKNOWN, &madd_op));

  loom_symbolic_expr_t expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, loom_index_madd_result(madd_op), &expression));

  EXPECT_TRUE(loom_symbolic_expr_is_linear(&expression));
  EXPECT_EQ(expression.constant, 0);
  ASSERT_EQ(expression.term_count, 2);
  EXPECT_EQ(expression.terms[0].coefficient, 16);
  EXPECT_EQ(expression.terms[0].value_id, row);
  EXPECT_EQ(expression.terms[1].coefficient, 1);
  EXPECT_EQ(expression.terms[1].value_id, column);
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

  EXPECT_TRUE(loom_symbolic_expr_is_linear(&expression));
  ASSERT_EQ(expression.term_count, 1);
  EXPECT_EQ(expression.terms[0].coefficient, 1);
  EXPECT_EQ(expression.terms[0].value_id, result);
}

TEST_F(SymbolicExprTest, ProvesLessEqualFromTermFacts) {
  loom_value_id_t value_id = DefineIndexValue();
  DefineFacts(value_id, loom_value_facts_make(0, 10, 1));

  loom_symbolic_expr_t value_expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, value_id,
                                               &value_expression));
  loom_symbolic_expr_t scaled_expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_mul_i64(
      &expression_context_, &value_expression, 2, &scaled_expression));
  loom_symbolic_expr_t twenty = {0};
  loom_symbolic_expr_constant(20, &twenty);

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_le(
      &expression_context_, &scaled_expression, &twenty, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
}

TEST_F(SymbolicExprTest, ProvesLessEqualFromExpressionFactsAfterExpansion) {
  loom_value_id_t value_id = DefineIndexValue();
  loom_predicate_t predicate = {
      /*.kind=*/LOOM_PREDICATE_RANGE,
      /*.arg_count=*/3,
      /*.arg_tags=*/
      {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST, LOOM_PRED_ARG_CONST},
      /*.reserved=*/{},
      /*.args=*/{value_id, 0, 10},
  };
  loom_op_t* assume_op = nullptr;
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  IREE_ASSERT_OK(loom_index_assume_build(&builder_, &value_id, 1, &predicate, 1,
                                         &index_type, 1, LOOM_LOCATION_UNKNOWN,
                                         &assume_op));
  loom_value_id_t assumed_value =
      loom_index_assume_results(assume_op).values[0];
  DefineFacts(assumed_value, loom_value_facts_make(0, 10, 1));

  loom_symbolic_expr_t expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_,
                                               assumed_value, &expression));
  ASSERT_TRUE(loom_symbolic_expr_is_linear(&expression));
  ASSERT_EQ(expression.term_count, 1);
  EXPECT_EQ(expression.terms[0].value_id, value_id);

  loom_symbolic_expr_t zero = {0};
  loom_symbolic_expr_constant(0, &zero);
  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_le(&expression_context_, &zero,
                                             &expression, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
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
