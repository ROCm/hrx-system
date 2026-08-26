// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_expr_proof.h"

#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/condition_facts.h"
#include "loom/analysis/symbolic_expr_test_fixture.h"
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

static bool QueryConditionFacts(
    loom_symbolic_expr_context_t* expression_context,
    const loom_value_fact_table_t* fact_table, loom_value_id_t condition,
    bool assumed_truth, loom_condition_fact_set_t* out_facts) {
  bool complete = false;
  IREE_CHECK_OK(loom_condition_facts_query(&expression_context->condition_query,
                                           fact_table, condition, assumed_truth,
                                           out_facts, &complete));
  return complete;
}

static bool AppendConditionFacts(
    loom_symbolic_expr_context_t* expression_context,
    const loom_value_fact_table_t* fact_table, loom_value_id_t condition,
    bool assumed_truth, loom_condition_fact_set_t* inout_facts) {
  bool complete = false;
  IREE_CHECK_OK(loom_condition_facts_query_into(
      &expression_context->condition_query, fact_table, condition,
      assumed_truth, inout_facts, &complete));
  return complete;
}

static iree_status_t ProveSemanticallyEquivalentUpperBound(
    loom_symbolic_expr_context_t* context, loom_value_id_t relation_value,
    loom_value_id_t query_value, loom_symbolic_proof_result_t* out_result) {
  loom_condition_integer_relation_t relation = {
      /*.relation=*/LOOM_SYMBOLIC_INTEGER_RELATION_LE,
      /*.left=*/
      {
          /*.kind=*/LOOM_CONDITION_INTEGER_OPERAND_CONSTANT,
          /*.value_id=*/LOOM_VALUE_ID_INVALID,
          /*.constant=*/0,
      },
      /*.right=*/
      {
          /*.kind=*/LOOM_CONDITION_INTEGER_OPERAND_VALUE,
          /*.value_id=*/relation_value,
          /*.constant=*/0,
      },
  };
  loom_condition_fact_set_t condition_facts = {
      /*.integer_relations=*/&relation,
      /*.integer_relation_count=*/1,
      /*.integer_relation_capacity=*/1,
  };
  context->condition_facts = &condition_facts;
  loom_symbolic_expr_context_reset(context);

  loom_symbolic_expr_t zero = {};
  loom_symbolic_expr_constant(0, &zero);
  loom_symbolic_expr_t query = {};
  iree_status_t status = loom_symbolic_expr_value(context, query_value, &query);
  if (iree_status_is_ok(status)) {
    status = loom_symbolic_expr_prove_le(context, &zero, &query, out_result);
  }

  context->condition_facts = nullptr;
  loom_symbolic_expr_context_reset(context);
  return status;
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

TEST_F(SymbolicExprTest, SemanticMatchVisitsSharedProducerPairsOnce) {
  loom_value_id_t source = DefineI64Value();
  loom_value_id_t left = source;
  loom_value_id_t right = source;
  for (int i = 0; i < 16; ++i) {
    left = BuildScalarAndI(left, left);
    right = BuildScalarAndI(right, right);
  }

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(ProveSemanticallyEquivalentUpperBound(&expression_context_,
                                                       left, right, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
}

TEST_F(SymbolicExprTest, SemanticMatchRejectsDistinctProducerLeaves) {
  loom_value_id_t left = DefineI64Value();
  loom_value_id_t right = DefineI64Value();
  for (int i = 0; i < 8; ++i) {
    left = BuildScalarAndI(left, left);
    right = BuildScalarAndI(right, right);
  }

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(ProveSemanticallyEquivalentUpperBound(&expression_context_,
                                                       left, right, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_UNKNOWN);
}

TEST_F(SymbolicExprTest, SemanticMatchRetainsProducerDepthLimit) {
  loom_value_id_t source = DefineI64Value();
  loom_value_id_t mask = DefineI64Value();
  loom_value_id_t left = source;
  loom_value_id_t right = source;
  for (int i = 0; i < 17; ++i) {
    left = BuildScalarAndI(left, mask);
    right = BuildScalarAndI(right, mask);
  }

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(ProveSemanticallyEquivalentUpperBound(&expression_context_,
                                                       left, right, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_UNKNOWN);
}

TEST_F(SymbolicExprTest, SimplifiesDifferenceToExistingValue) {
  loom_value_id_t row = DefineIndexValue();
  loom_value_id_t column = DefineIndexValue();
  loom_value_id_t stride = loom_index_constant_result(BuildIndexConstant(16));
  loom_op_t* scaled_op = nullptr;
  IREE_ASSERT_OK(loom_index_mul_build(&builder_, row, stride,
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
                                      LOOM_LOCATION_UNKNOWN,
                                      &scaled_induction_op));
  loom_op_t* scaled_bound_op = nullptr;
  IREE_ASSERT_OK(loom_index_mul_build(&builder_, upper_bound, four,
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

TEST_F(SymbolicExprTest, ShiftedStrictRelationProvesWideExtent) {
  loom_value_id_t origin = DefineIndexValue();
  loom_value_id_t element_count = DefineIndexValue();
  loom_value_id_t three = loom_index_constant_result(BuildIndexConstant(3));
  loom_op_t* last_valid_start_op = nullptr;
  IREE_ASSERT_OK(loom_index_sub_build(
      &builder_, element_count, three, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      LOOM_LOCATION_UNKNOWN, &last_valid_start_op));
  loom_value_id_t last_valid_start = loom_index_sub_result(last_valid_start_op);
  loom_predicate_t predicate = {
      /*.kind=*/LOOM_PREDICATE_LT,
      /*.arg_count=*/2,
      /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
      /*.reserved=*/{},
      /*.args=*/{origin, last_valid_start},
  };
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(&builder_, &origin, 1, &predicate, 1,
                                         &index_type, 1, LOOM_LOCATION_UNKNOWN,
                                         &assume_op));
  loom_value_id_t assumed_origin =
      loom_index_assume_results(assume_op).values[0];

  loom_value_id_t four = loom_index_constant_result(BuildIndexConstant(4));
  loom_op_t* end_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, assumed_origin, four,
                                      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                      LOOM_LOCATION_UNKNOWN, &end_op));
  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LE,
      loom_index_add_result(end_op), element_count, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);

  loom_value_id_t five = loom_index_constant_result(BuildIndexConstant(5));
  loom_op_t* too_wide_end_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, assumed_origin, five,
                                      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                      LOOM_LOCATION_UNKNOWN, &too_wide_end_op));
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LE,
      loom_index_add_result(too_wide_end_op), element_count, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_UNKNOWN);
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

TEST_F(SymbolicExprTest, ProvesFlattenedAddressFromDynamicAxisBounds) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t row_source = DefineIndexValue();
  loom_value_id_t row_count_source = DefineIndexValue();
  loom_value_id_t column = DefineIndexValue();
  loom_value_id_t column_count = DefineIndexValue();
  DefineFacts(row_source, loom_value_facts_make(0, 2047, 1));
  DefineFacts(row_count_source, loom_value_facts_make(1, 2048, 1));
  DefineFacts(column, loom_value_facts_make(0, 262143, 1));
  DefineFacts(column_count, loom_value_facts_make(1, 262144, 1));

  loom_value_id_t row_values[] = {row_source, row_count_source};
  loom_predicate_t row_predicate = {
      /*.kind=*/LOOM_PREDICATE_LT,
      /*.arg_count=*/2,
      /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
      /*.reserved=*/{},
      /*.args=*/{row_source, row_count_source, 0},
  };
  loom_type_t row_result_types[] = {index_type, index_type};
  loom_op_t* row_assume_op = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(
      &builder_, row_values, IREE_ARRAYSIZE(row_values), &row_predicate, 1,
      row_result_types, IREE_ARRAYSIZE(row_result_types), LOOM_LOCATION_UNKNOWN,
      &row_assume_op));
  loom_value_slice_t row_results = loom_index_assume_results(row_assume_op);
  loom_value_id_t row = row_results.values[0];
  loom_value_id_t row_count = row_results.values[1];

  loom_op_t* column_compare_op = nullptr;
  IREE_ASSERT_OK(loom_index_cmp_build(
      &builder_, LOOM_INDEX_CMP_PREDICATE_ULT, column, column_count,
      LOOM_LOCATION_UNKNOWN, &column_compare_op));
  loom_condition_integer_relation_t relation_storage[4];
  loom_condition_fact_set_t condition_facts;
  loom_condition_fact_set_initialize(
      relation_storage, IREE_ARRAYSIZE(relation_storage), &condition_facts);
  ASSERT_TRUE(QueryConditionFacts(&expression_context_, &fact_table_,
                                  loom_index_cmp_result(column_compare_op),
                                  true, &condition_facts));
  expression_context_.condition_facts = &condition_facts;
  loom_symbolic_expr_context_reset(&expression_context_);

  loom_op_t* row_base_op = nullptr;
  IREE_ASSERT_OK(loom_index_mul_build(&builder_, row, column_count,
                                      LOOM_LOCATION_UNKNOWN, &row_base_op));
  loom_op_t* origin_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(
      &builder_, loom_index_mul_result(row_base_op), column, index_type,
      LOOM_LOCATION_UNKNOWN, &origin_op));
  loom_op_t* element_count_op = nullptr;
  IREE_ASSERT_OK(loom_index_mul_build(&builder_, row_count, column_count,
                                      LOOM_LOCATION_UNKNOWN,
                                      &element_count_op));

  loom_symbolic_expr_t origin = {};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, loom_index_add_result(origin_op), &origin));
  loom_symbolic_expr_t one = {};
  loom_symbolic_expr_constant(1, &one);
  loom_symbolic_expr_t exclusive_end = {};
  IREE_ASSERT_OK(loom_symbolic_expr_add(&expression_context_, &origin, &one,
                                        &exclusive_end));
  loom_symbolic_expr_t element_count = {};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, loom_index_mul_result(element_count_op),
      &element_count));

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_le(
      &expression_context_, &exclusive_end, &element_count, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);

  loom_op_t* row_nonstrict_compare_op = nullptr;
  IREE_ASSERT_OK(loom_index_cmp_build(
      &builder_, LOOM_INDEX_CMP_PREDICATE_ULE, row_source, row_count_source,
      LOOM_LOCATION_UNKNOWN, &row_nonstrict_compare_op));
  ASSERT_TRUE(QueryConditionFacts(
      &expression_context_, &fact_table_,
      loom_index_cmp_result(row_nonstrict_compare_op), true, &condition_facts));
  ASSERT_TRUE(AppendConditionFacts(&expression_context_, &fact_table_,
                                   loom_index_cmp_result(column_compare_op),
                                   true, &condition_facts));
  loom_symbolic_expr_context_reset(&expression_context_);

  loom_op_t* nonstrict_row_base_op = nullptr;
  IREE_ASSERT_OK(loom_index_mul_build(&builder_, row_source, column_count,
                                      LOOM_LOCATION_UNKNOWN,
                                      &nonstrict_row_base_op));
  loom_op_t* nonstrict_origin_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(
      &builder_, loom_index_mul_result(nonstrict_row_base_op), column,
      index_type, LOOM_LOCATION_UNKNOWN, &nonstrict_origin_op));
  loom_op_t* nonstrict_element_count_op = nullptr;
  IREE_ASSERT_OK(loom_index_mul_build(&builder_, row_count_source, column_count,
                                      LOOM_LOCATION_UNKNOWN,
                                      &nonstrict_element_count_op));
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, loom_index_add_result(nonstrict_origin_op),
      &origin));
  IREE_ASSERT_OK(loom_symbolic_expr_add(&expression_context_, &origin, &one,
                                        &exclusive_end));
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, loom_index_mul_result(nonstrict_element_count_op),
      &element_count));
  IREE_ASSERT_OK(loom_symbolic_expr_prove_le(
      &expression_context_, &exclusive_end, &element_count, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_UNKNOWN);
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

TEST_F(SymbolicExprTest, ProvesIndexRemainderIsBelowDynamicDivisor) {
  loom_value_id_t dividend = DefineIndexValue();
  loom_value_id_t divisor = DefineIndexValue();
  DefineFacts(dividend, loom_value_facts_make(0, 1024, 1));
  DefineFacts(divisor, loom_value_facts_make(1, 512, 1));
  loom_op_t* remainder_op = nullptr;
  IREE_ASSERT_OK(loom_index_rem_build(&builder_, dividend, divisor,
                                      LOOM_LOCATION_UNKNOWN, &remainder_op));
  loom_value_id_t remainder = loom_index_rem_result(remainder_op);

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LT, remainder,
      divisor, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);

  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_GE, remainder,
      divisor, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_FALSE);

  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_GT, divisor,
      remainder, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
}

TEST_F(SymbolicExprTest, ProvesAssumedIndexRemainderIsBelowDynamicDivisor) {
  loom_value_id_t dividend = DefineIndexValue();
  loom_value_id_t divisor = DefineIndexValue();
  DefineFacts(dividend, loom_value_facts_make(0, 1024, 1));
  DefineFacts(divisor, loom_value_facts_make(1, 512, 1));
  loom_op_t* remainder_op = nullptr;
  IREE_ASSERT_OK(loom_index_rem_build(&builder_, dividend, divisor,
                                      LOOM_LOCATION_UNKNOWN, &remainder_op));
  loom_value_id_t remainder = loom_index_rem_result(remainder_op);

  loom_predicate_t predicate = {
      /*.kind=*/LOOM_PREDICATE_GE,
      /*.arg_count=*/2,
      /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
      /*.reserved=*/{},
      /*.args=*/{remainder, 0},
  };
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(&builder_, &remainder, 1, &predicate,
                                         1, &index_type, 1,
                                         LOOM_LOCATION_UNKNOWN, &assume_op));
  loom_value_id_t assumed_remainder =
      loom_index_assume_results(assume_op).values[0];

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LT,
      assumed_remainder, divisor, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
}

TEST_F(SymbolicExprTest, RemainderBoundRequiresUnsignedInputFacts) {
  loom_value_id_t dividend = DefineIndexValue();
  loom_value_id_t divisor = DefineIndexValue();
  DefineFacts(dividend, loom_value_facts_make(-8, 1024, 1));
  DefineFacts(divisor, loom_value_facts_make(1, 512, 1));
  loom_op_t* remainder_op = nullptr;
  IREE_ASSERT_OK(loom_index_rem_build(&builder_, dividend, divisor,
                                      LOOM_LOCATION_UNKNOWN, &remainder_op));
  loom_value_id_t remainder = loom_index_rem_result(remainder_op);

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LT, remainder,
      divisor, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_UNKNOWN);
}

TEST_F(SymbolicExprTest, ProvesScalarUnsignedRemainderIsBelowDivisor) {
  loom_value_id_t dividend = DefineI64Value();
  loom_value_id_t divisor = DefineI64Value();
  DefineFacts(dividend, loom_value_facts_make(0, 1024, 1));
  DefineFacts(divisor, loom_value_facts_make(1, 512, 1));
  loom_op_t* remainder_op = nullptr;
  IREE_ASSERT_OK(loom_scalar_remui_build(&builder_, dividend, divisor,
                                         loom_type_scalar(LOOM_SCALAR_TYPE_I64),
                                         LOOM_LOCATION_UNKNOWN, &remainder_op));
  loom_value_id_t remainder = loom_scalar_remui_result(remainder_op);

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LE, remainder,
      divisor, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);

  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_EQ, remainder,
      divisor, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_FALSE);
}

TEST_F(SymbolicExprTest, ConditionRefinementMemoReusesPersistentStorage) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t source = DefineIndexValue();
  DefineFacts(source, loom_value_facts_make(0, 1023, 1));
  loom_value_id_t upper = loom_index_constant_result(BuildIndexConstant(1024));
  DefineFacts(upper, loom_value_facts_exact_i64(1024));

  loom_op_t* condition_op = nullptr;
  IREE_ASSERT_OK(loom_index_cmp_build(&builder_, LOOM_INDEX_CMP_PREDICATE_ULT,
                                      source, upper, LOOM_LOCATION_UNKNOWN,
                                      &condition_op));
  loom_value_id_t condition = loom_index_cmp_result(condition_op);

  // Each select references the preceding value twice. Condition-refined fact
  // inference memoizes the shared DAG instead of retaining per-visit storage.
  loom_value_id_t selected_value = source;
  for (iree_host_size_t i = 0; i < 8; ++i) {
    loom_op_t* select_op = nullptr;
    IREE_ASSERT_OK(loom_scf_select_build(&builder_, condition, selected_value,
                                         selected_value, index_type,
                                         LOOM_LOCATION_UNKNOWN, &select_op));
    selected_value = loom_scf_select_result(select_op);
  }

  loom_condition_integer_relation_t relation_storage[2];
  loom_condition_fact_set_t condition_facts;
  loom_condition_fact_set_initialize(
      relation_storage, IREE_ARRAYSIZE(relation_storage), &condition_facts);
  ASSERT_TRUE(QueryConditionFacts(&expression_context_, &fact_table_, condition,
                                  /*assumed_truth=*/true, &condition_facts));
  expression_context_.condition_facts = &condition_facts;

  loom_symbolic_expr_t zero = {0};
  loom_symbolic_expr_constant(0, &zero);
  loom_symbolic_expr_t selected_expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_value(&expression_context_, selected_value,
                                          &selected_expression));
  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_le(&expression_context_, &zero,
                                             &selected_expression, &proof));
  ASSERT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);

  const iree_host_size_t warmed_allocation_size =
      analysis_arena_.used_allocation_size;
  for (iree_host_size_t i = 0; i < 4; ++i) {
    loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    IREE_ASSERT_OK(loom_symbolic_expr_prove_le(&expression_context_, &zero,
                                               &selected_expression, &proof));
    EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
  }
  EXPECT_EQ(analysis_arena_.used_allocation_size, warmed_allocation_size);
}

TEST_F(SymbolicExprTest, SelectConditionProvesDynamicLoopLowerBound) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t output = DefineIndexValue();
  DefineFacts(output, loom_value_facts_make(0, 1023, 1));
  loom_value_id_t raw_induction = DefineIndexValue();
  DefineFacts(raw_induction, loom_value_facts_make(0, 2, 1));
  loom_value_id_t zero_value =
      loom_index_constant_result(BuildIndexConstant(0));
  loom_value_id_t one_value = loom_index_constant_result(BuildIndexConstant(1));
  DefineFacts(zero_value, loom_value_facts_exact_i64(0));
  DefineFacts(one_value, loom_value_facts_exact_i64(1));

  loom_op_t* edge_cmp_op = nullptr;
  IREE_ASSERT_OK(loom_index_cmp_build(&builder_, LOOM_INDEX_CMP_PREDICATE_EQ,
                                      output, zero_value, LOOM_LOCATION_UNKNOWN,
                                      &edge_cmp_op));
  loom_op_t* lower_bound_op = nullptr;
  IREE_ASSERT_OK(loom_scf_select_build(
      &builder_, loom_index_cmp_result(edge_cmp_op), one_value, zero_value,
      index_type, LOOM_LOCATION_UNKNOWN, &lower_bound_op));
  loom_value_id_t lower_bound = loom_scf_select_result(lower_bound_op);

  loom_predicate_t lower_bound_predicate = {
      /*.kind=*/LOOM_PREDICATE_GE,
      /*.arg_count=*/2,
      /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
      /*.reserved=*/{},
      /*.args=*/{raw_induction, lower_bound},
  };
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(&builder_, &raw_induction, 1,
                                         &lower_bound_predicate, 1, &index_type,
                                         1, LOOM_LOCATION_UNKNOWN, &assume_op));
  loom_value_id_t induction = loom_index_assume_results(assume_op).values[0];

  loom_op_t* sum_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, output, induction, index_type,
                                      LOOM_LOCATION_UNKNOWN, &sum_op));
  loom_op_t* shifted_op = nullptr;
  IREE_ASSERT_OK(loom_index_sub_build(&builder_, loom_index_add_result(sum_op),
                                      one_value, index_type,
                                      LOOM_LOCATION_UNKNOWN, &shifted_op));
  const loom_value_id_t shifted = loom_index_sub_result(shifted_op);

  loom_symbolic_proof_result_t active_facts_proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation_with_active_facts(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_GE, shifted,
      zero_value, &active_facts_proof));
  EXPECT_EQ(active_facts_proof, LOOM_SYMBOLIC_PROOF_UNKNOWN);

  loom_symbolic_expr_t shifted_expression = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, shifted,
                                               &shifted_expression));
  loom_symbolic_expr_t zero_expression = {0};
  loom_symbolic_expr_constant(0, &zero_expression);
  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_le(
      &expression_context_, &zero_expression, &shifted_expression, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
}

TEST_F(SymbolicExprTest, SelectConditionProvesDynamicLoopDivBounds) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t linear = DefineIndexValue();
  DefineFacts(linear, loom_value_facts_make(0, 268435455, 1));
  loom_value_id_t raw_tap = DefineIndexValue();
  DefineFacts(raw_tap, loom_value_facts_make(0, 2, 1));

  loom_value_id_t zero_value =
      loom_index_constant_result(BuildIndexConstant(0));
  loom_value_id_t one_value = loom_index_constant_result(BuildIndexConstant(1));
  loom_value_id_t two_value = loom_index_constant_result(BuildIndexConstant(2));
  loom_value_id_t three_value =
      loom_index_constant_result(BuildIndexConstant(3));
  loom_value_id_t last_lane_value =
      loom_index_constant_result(BuildIndexConstant(1023));
  DefineFacts(zero_value, loom_value_facts_exact_i64(0));
  DefineFacts(one_value, loom_value_facts_exact_i64(1));
  DefineFacts(two_value, loom_value_facts_exact_i64(2));
  DefineFacts(three_value, loom_value_facts_exact_i64(3));
  DefineFacts(last_lane_value, loom_value_facts_exact_i64(1023));

  loom_op_t* lane_op = nullptr;
  IREE_ASSERT_OK(loom_index_andi_build(&builder_, linear, last_lane_value,
                                       LOOM_LOCATION_UNKNOWN, &lane_op));
  loom_value_id_t lane = loom_index_andi_result(lane_op);
  DefineFacts(lane, loom_value_facts_make(0, 1023, 1));

  loom_op_t* left_edge_cmp_op = nullptr;
  IREE_ASSERT_OK(loom_index_cmp_build(&builder_, LOOM_INDEX_CMP_PREDICATE_EQ,
                                      lane, zero_value, LOOM_LOCATION_UNKNOWN,
                                      &left_edge_cmp_op));
  loom_op_t* lower_bound_op = nullptr;
  IREE_ASSERT_OK(loom_scf_select_build(
      &builder_, loom_index_cmp_result(left_edge_cmp_op), one_value, zero_value,
      index_type, LOOM_LOCATION_UNKNOWN, &lower_bound_op));

  loom_op_t* right_edge_cmp_op = nullptr;
  IREE_ASSERT_OK(loom_index_cmp_build(
      &builder_, LOOM_INDEX_CMP_PREDICATE_EQ, lane, last_lane_value,
      LOOM_LOCATION_UNKNOWN, &right_edge_cmp_op));
  loom_op_t* upper_bound_op = nullptr;
  IREE_ASSERT_OK(loom_scf_select_build(
      &builder_, loom_index_cmp_result(right_edge_cmp_op), two_value,
      three_value, index_type, LOOM_LOCATION_UNKNOWN, &upper_bound_op));

  loom_predicate_t tap_predicates[] = {
      {
          /*.kind=*/LOOM_PREDICATE_GE,
          /*.arg_count=*/2,
          /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
          /*.reserved=*/{},
          /*.args=*/{raw_tap, loom_scf_select_result(lower_bound_op)},
      },
      {
          /*.kind=*/LOOM_PREDICATE_LT,
          /*.arg_count=*/2,
          /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
          /*.reserved=*/{},
          /*.args=*/{raw_tap, loom_scf_select_result(upper_bound_op)},
      },
  };
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(
      &builder_, &raw_tap, 1, tap_predicates, IREE_ARRAYSIZE(tap_predicates),
      &index_type, 1, LOOM_LOCATION_UNKNOWN, &assume_op));
  loom_value_id_t tap = loom_index_assume_results(assume_op).values[0];

  loom_op_t* sum_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, lane, tap, index_type,
                                      LOOM_LOCATION_UNKNOWN, &sum_op));
  loom_op_t* source_lane_op = nullptr;
  IREE_ASSERT_OK(loom_index_sub_build(&builder_, loom_index_add_result(sum_op),
                                      one_value, index_type,
                                      LOOM_LOCATION_UNKNOWN, &source_lane_op));
  loom_op_t* input_lane_op = nullptr;
  IREE_ASSERT_OK(
      loom_index_div_build(&builder_, loom_index_sub_result(source_lane_op),
                           two_value, LOOM_LOCATION_UNKNOWN, &input_lane_op));

  loom_symbolic_expr_t source_lane = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, loom_index_sub_result(source_lane_op),
      &source_lane));
  ASSERT_EQ(source_lane.term_count, 2);
  EXPECT_EQ(source_lane.constant, -1);
  EXPECT_EQ(source_lane.terms[0].value_id, raw_tap);
  EXPECT_EQ(source_lane.terms[0].relation_value_id, tap);
  EXPECT_EQ(source_lane.terms[0].coefficient, 1);
  EXPECT_EQ(source_lane.terms[1].value_id, lane);
  EXPECT_EQ(source_lane.terms[1].coefficient, 1);

  loom_condition_integer_relation_t left_false_storage[16];
  loom_condition_fact_set_t left_false_facts;
  loom_condition_fact_set_initialize(left_false_storage,
                                     IREE_ARRAYSIZE(left_false_storage),
                                     &left_false_facts);
  ASSERT_TRUE(QueryConditionFacts(&expression_context_, &fact_table_,
                                  loom_index_cmp_result(left_edge_cmp_op),
                                  /*assumed_truth=*/false, &left_false_facts));
  expression_context_.condition_facts = &left_false_facts;
  expression_context_.condition_proof_depth = 1;
  loom_symbolic_expr_context_reset(&expression_context_);
  loom_symbolic_expr_t left_false_source_lane = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, loom_index_sub_result(source_lane_op),
      &left_false_source_lane));
  ASSERT_EQ(left_false_source_lane.term_count, 2);
  EXPECT_EQ(left_false_source_lane.constant, -1);
  EXPECT_EQ(left_false_source_lane.terms[0].value_id, raw_tap);
  EXPECT_EQ(left_false_source_lane.terms[0].relation_value_id, tap);
  EXPECT_EQ(left_false_source_lane.terms[0].coefficient, 1);
  EXPECT_EQ(left_false_source_lane.terms[1].value_id, lane);
  EXPECT_EQ(left_false_source_lane.terms[1].coefficient, 1);
  loom_symbolic_expr_t zero = {0};
  loom_symbolic_expr_constant(0, &zero);
  loom_symbolic_proof_result_t left_false_proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_le(
      &expression_context_, &zero, &left_false_source_lane, &left_false_proof));
  EXPECT_EQ(left_false_proof, LOOM_SYMBOLIC_PROOF_TRUE);
  expression_context_.condition_facts = nullptr;
  expression_context_.condition_proof_depth = 0;
  loom_symbolic_expr_context_reset(&expression_context_);

  loom_symbolic_proof_result_t source_lower_proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_le(
      &expression_context_, &zero, &source_lane, &source_lower_proof));
  EXPECT_EQ(source_lower_proof, LOOM_SYMBOLIC_PROOF_TRUE);

  loom_condition_integer_relation_t relation_storage[16];
  loom_condition_fact_set_t false_edge_facts;
  loom_condition_fact_set_initialize(
      relation_storage, IREE_ARRAYSIZE(relation_storage), &false_edge_facts);
  ASSERT_TRUE(QueryConditionFacts(&expression_context_, &fact_table_,
                                  loom_index_cmp_result(left_edge_cmp_op),
                                  /*assumed_truth=*/false, &false_edge_facts));
  ASSERT_TRUE(AppendConditionFacts(&expression_context_, &fact_table_,
                                   loom_index_cmp_result(right_edge_cmp_op),
                                   /*assumed_truth=*/false, &false_edge_facts));
  loom_value_facts_t lane_false_edge_facts =
      loom_value_fact_table_lookup(&fact_table_, lane);
  EXPECT_TRUE(loom_condition_fact_set_apply_to_value_facts(
      &false_edge_facts, &fact_table_, lane, &lane_false_edge_facts));
  EXPECT_EQ(lane_false_edge_facts.range_lo, 1);
  EXPECT_EQ(lane_false_edge_facts.range_hi, 1022);

  loom_symbolic_expr_t input_lane = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, loom_index_div_result(input_lane_op), &input_lane));

  loom_symbolic_proof_result_t lower_proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_le(&expression_context_, &zero,
                                             &input_lane, &lower_proof));
  EXPECT_EQ(lower_proof, LOOM_SYMBOLIC_PROOF_TRUE);

  loom_symbolic_expr_t one = {0};
  loom_symbolic_expr_constant(1, &one);
  loom_symbolic_expr_t exclusive_end = {0};
  IREE_ASSERT_OK(loom_symbolic_expr_add(&expression_context_, &input_lane, &one,
                                        &exclusive_end));
  loom_symbolic_expr_t view_bound = {0};
  loom_symbolic_expr_constant(512, &view_bound);
  loom_symbolic_proof_result_t upper_proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_le(
      &expression_context_, &exclusive_end, &view_bound, &upper_proof));
  EXPECT_EQ(upper_proof, LOOM_SYMBOLIC_PROOF_TRUE);
}

}  // namespace
}  // namespace loom
