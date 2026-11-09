// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbolic_expr_proof.h"
#include "loom/analysis/symbolic_expr_test_fixture.h"
#include "loom/analysis/symbolic_integer_cast.h"
#include "loom/analysis/symbolic_product.h"
#include "loom/analysis/symbolic_projection.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scf/ops.h"

namespace loom {
namespace {

class SymbolicProofLifetimeTest : public SymbolicExprTest {
 protected:
  loom_value_id_t Constant(int64_t value) {
    auto* op = BuildIndexConstant(value);
    ComputeFacts(op);
    return loom_index_constant_result(op);
  }

  loom_value_id_t Add(loom_value_id_t left, loom_value_id_t right) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_add_build(&builder_, left, right,
                                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                       LOOM_LOCATION_UNKNOWN, &op));
    ComputeFacts(op);
    return loom_index_add_result(op);
  }

  loom_value_id_t Select(loom_value_id_t condition, loom_value_id_t left,
                         loom_value_id_t right) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_scf_select_build(
        &builder_, condition, left, right,
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN, &op));
    ComputeFacts(op);
    return loom_scf_select_result(op);
  }

  loom_value_id_t ComplementarySum() {
    const auto zero = Constant(0);
    const auto one = Constant(1);
    const auto condition = DefineI1Value();
    return Add(Select(condition, one, zero), Select(condition, zero, one));
  }

  loom_value_id_t WideComplementarySum() {
    const auto zero = Constant(0);
    const auto condition = DefineI1Value();
    auto sum = zero;
    for (int i = 1; i <= 21; ++i) {
      const auto value = Constant(i);
      const auto twice = Constant(2 * i);
      // Each triple sums to 2*i on either outcome. Distinct source values
      // exercise growth of the proof's compact index and borrowed memo rows.
      sum = Add(sum, Select(condition, value, zero));
      sum = Add(sum, Select(condition, zero, twice));
      sum = Add(sum, Select(condition, value, zero));
    }
    return sum;
  }

  loom_symbolic_expr_t Expression(loom_value_id_t value) {
    loom_symbolic_expr_t expression = {};
    IREE_CHECK_OK(loom_symbolic_expr_from_value(&expression_context_, value,
                                                &expression));
    return expression;
  }

  void ExpectSummary(loom_value_id_t value,
                     const loom_symbolic_expr_summary_t& prepared) {
    loom_symbolic_expr_summary_t retained = {};
    ASSERT_TRUE(loom_symbolic_expr_context_try_lookup_summary(
        &expression_context_, value, &retained));
    EXPECT_EQ(retained.expression.constant, prepared.expression.constant);
    ASSERT_EQ(retained.expression.term_count, prepared.expression.term_count);
    for (iree_host_size_t i = 0; i < retained.expression.term_count; ++i) {
      EXPECT_EQ(retained.expression.terms[i].value_id,
                prepared.expression.terms[i].value_id);
      EXPECT_EQ(retained.expression.terms[i].coefficient,
                prepared.expression.terms[i].coefficient);
    }
    EXPECT_EQ(retained.materialized_dynamic_value_id,
              prepared.materialized_dynamic_value_id);
    EXPECT_EQ(retained.materialized_component_value_id,
              prepared.materialized_component_value_id);
    EXPECT_EQ(retained.projection, prepared.projection);
    EXPECT_EQ(retained.integer_cast, prepared.integer_cast);
  }
};

TEST_F(SymbolicProofLifetimeTest, SelectProofPreservesPreparedAddressSummary) {
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const loom_value_id_t row = DefineIndexValue();
  DefineFacts(row, loom_value_facts_make(0, 1023, 1));
  const loom_value_id_t selector = DefineIndexValue();
  DefineFacts(selector, loom_value_facts_make(0, 1, 1));
  loom_op_t* zero_op = BuildIndexConstant(0);
  loom_op_t* one_op = BuildIndexConstant(1);
  loom_op_t* stride_op = BuildIndexConstant(24);
  loom_op_t* offset_op = BuildIndexConstant(8);
  ComputeFacts(zero_op);
  ComputeFacts(one_op);
  ComputeFacts(stride_op);
  ComputeFacts(offset_op);

  loom_op_t* scaled_row_op = nullptr;
  IREE_ASSERT_OK(loom_index_mul_build(&builder_, row,
                                      loom_index_constant_result(stride_op),
                                      LOOM_LOCATION_UNKNOWN, &scaled_row_op));
  ComputeFacts(scaled_row_op);
  loom_op_t* address_op = nullptr;
  IREE_ASSERT_OK(
      loom_index_add_build(&builder_, loom_index_mul_result(scaled_row_op),
                           loom_index_constant_result(offset_op), index_type,
                           LOOM_LOCATION_UNKNOWN, &address_op));
  ComputeFacts(address_op);

  loom_op_t* compare_op = nullptr;
  IREE_ASSERT_OK(loom_index_cmp_build(
      &builder_, LOOM_INDEX_CMP_PREDICATE_EQ, selector,
      loom_index_constant_result(zero_op), LOOM_LOCATION_UNKNOWN, &compare_op));
  ComputeFacts(compare_op);
  loom_op_t* left_select_op = nullptr;
  IREE_ASSERT_OK(loom_scf_select_build(
      &builder_, loom_index_cmp_result(compare_op),
      loom_index_constant_result(one_op), loom_index_constant_result(zero_op),
      index_type, LOOM_LOCATION_UNKNOWN, &left_select_op));
  ComputeFacts(left_select_op);
  loom_op_t* right_select_op = nullptr;
  IREE_ASSERT_OK(loom_scf_select_build(
      &builder_, loom_index_cmp_result(compare_op),
      loom_index_constant_result(zero_op), loom_index_constant_result(one_op),
      index_type, LOOM_LOCATION_UNKNOWN, &right_select_op));
  ComputeFacts(right_select_op);
  loom_op_t* sum_op = nullptr;
  IREE_ASSERT_OK(
      loom_index_add_build(&builder_, loom_scf_select_result(left_select_op),
                           loom_scf_select_result(right_select_op), index_type,
                           LOOM_LOCATION_UNKNOWN, &sum_op));
  ComputeFacts(sum_op);

  // Memory planning prepares this summary before unrelated rule matching asks
  // whether the two complementary selects always sum to at least one.
  const loom_value_id_t address = loom_index_add_result(address_op);
  loom_symbolic_expr_t address_expression = {};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, address,
                                               &address_expression));
  loom_symbolic_expr_summary_t prepared = {};
  ASSERT_TRUE(loom_symbolic_expr_context_try_lookup_summary(
      &expression_context_, address, &prepared));
  ASSERT_EQ(prepared.expression.constant, 8);
  ASSERT_EQ(prepared.expression.term_count, 1u);
  EXPECT_EQ(prepared.expression.terms[0].coefficient, 24);
  EXPECT_EQ(prepared.materialized_dynamic_value_id,
            loom_index_mul_result(scaled_row_op));

  loom_symbolic_expr_t sum = {};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(
      &expression_context_, loom_index_add_result(sum_op), &sum));
  loom_symbolic_expr_t one = {};
  loom_symbolic_expr_constant(1, &one);
  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(
      loom_symbolic_expr_prove_le(&expression_context_, &one, &sum, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);

  // This is the lookup-only handoff to memory lowering. Recomputing the
  // address here would hide lost producer state and repeat compiler work.
  loom_symbolic_expr_summary_t retained = {};
  ASSERT_TRUE(loom_symbolic_expr_context_try_lookup_summary(
      &expression_context_, address, &retained));
  EXPECT_EQ(retained.expression.constant, prepared.expression.constant);
  ASSERT_EQ(retained.expression.term_count, 1u);
  EXPECT_EQ(retained.expression.terms[0].value_id, row);
  EXPECT_EQ(retained.expression.terms[0].coefficient, 24);
  EXPECT_EQ(retained.materialized_dynamic_value_id,
            prepared.materialized_dynamic_value_id);
}

TEST_F(SymbolicProofLifetimeTest, ProofOutcomesPreserveCanonicalSummaries) {
  const auto sum_value = ComplementarySum();
  const auto sum = Expression(sum_value);
  loom_symbolic_expr_summary_t prepared = {};
  ASSERT_TRUE(loom_symbolic_expr_context_try_lookup_summary(
      &expression_context_, sum_value, &prepared));
  // The complementary pair is exactly one. Its independent interval is [0,2],
  // so both equality directions require hypothetical conditions. The bound at
  // two remains unknown; three is false even in the independent interval.
  for (int repeat = 0; repeat < 3; ++repeat) {
    for (const auto bound : {1, 2, 3}) {
      loom_symbolic_expr_t minimum = {};
      loom_symbolic_expr_constant(bound, &minimum);
      loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
      IREE_ASSERT_OK(loom_symbolic_expr_prove_le(&expression_context_, &minimum,
                                                 &sum, &proof));
      EXPECT_EQ(proof, bound == 1   ? LOOM_SYMBOLIC_PROOF_TRUE
                       : bound == 2 ? LOOM_SYMBOLIC_PROOF_UNKNOWN
                                    : LOOM_SYMBOLIC_PROOF_FALSE);
      ASSERT_NO_FATAL_FAILURE(ExpectSummary(sum_value, prepared));
      EXPECT_EQ(expression_context_.condition_scope, nullptr);
    }
    loom_symbolic_expr_t one = {};
    loom_symbolic_expr_constant(1, &one);
    loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    IREE_ASSERT_OK(
        loom_symbolic_expr_prove_le(&expression_context_, &sum, &one, &proof));
    EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
    ASSERT_NO_FATAL_FAILURE(ExpectSummary(sum_value, prepared));
  }
}

TEST_F(SymbolicProofLifetimeTest, EnclosingConditionAndItsSummarySurvive) {
  const auto input = DefineIndexValue();
  DefineFacts(input, loom_value_facts_make(0, 100, 1));
  const auto sum_value = ComplementarySum();
  auto relation = LiteralRelation(LOOM_SYMBOLIC_INTEGER_RELATION_LE, input, 7);
  const auto facts = SingleRelationFacts(&relation);
  ScopedConditionFacts active(&expression_context_, &facts);
  const auto* enclosing_scope = expression_context_.condition_scope;
  const auto input_expression = Expression(input);
  EXPECT_EQ(input_expression.facts.range_hi, 7);
  loom_symbolic_expr_summary_t prepared = {};
  ASSERT_TRUE(loom_symbolic_expr_context_try_lookup_summary(
      &expression_context_, input, &prepared));

  const auto sum = Expression(sum_value);
  loom_symbolic_expr_t one = {};
  loom_symbolic_expr_constant(1, &one);
  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(
      loom_symbolic_expr_prove_le(&expression_context_, &one, &sum, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
  EXPECT_EQ(expression_context_.condition_scope, enclosing_scope);
  ASSERT_NO_FATAL_FAILURE(ExpectSummary(input, prepared));
  EXPECT_EQ(Expression(input).facts.range_hi, 7);
}

TEST_F(SymbolicProofLifetimeTest, BorrowedRecordsSurviveBranchExpansion) {
  const auto input = DefineIndexValue();
  const auto factor = DefineIndexValue();
  DefineFacts(input, loom_value_facts_make(0, 4095, 1));
  DefineFacts(factor, loom_value_facts_make(1, 3, 1));
  const auto eight = Constant(8);
  loom_op_t* quotient = nullptr;
  IREE_ASSERT_OK(loom_index_div_build(&builder_, input, eight,
                                      LOOM_LOCATION_UNKNOWN, &quotient));
  ComputeFacts(quotient);
  loom_op_t* cast = nullptr;
  IREE_ASSERT_OK(loom_index_cast_build(
      &builder_, input, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_I8), LOOM_LOCATION_UNKNOWN, &cast));
  ComputeFacts(cast);
  loom_op_t* product = nullptr;
  IREE_ASSERT_OK(loom_index_mul_build(&builder_, input, factor,
                                      LOOM_LOCATION_UNKNOWN, &product));
  ComputeFacts(product);

  const auto other = DefineIndexValue();
  DefineFacts(other, loom_value_facts_make(0, 1048575, 1));
  const auto seven = Constant(7);
  loom_op_t* other_quotient = nullptr;
  IREE_ASSERT_OK(loom_index_div_build(&builder_, other, seven,
                                      LOOM_LOCATION_UNKNOWN, &other_quotient));
  ComputeFacts(other_quotient);
  const auto negative = Constant(-1);
  const auto selected =
      Select(DefineI1Value(), loom_index_div_result(other_quotient), negative);

  loom_symbolic_expr_summary_t summaries[3] = {};
  const loom_value_id_t values[] = {loom_index_div_result(quotient),
                                    loom_index_cast_result(cast),
                                    loom_index_mul_result(product)};
  for (int i = 0; i < 3; ++i) {
    Expression(values[i]);
    ASSERT_TRUE(loom_symbolic_expr_context_try_lookup_summary(
        &expression_context_, values[i], &summaries[i]));
  }
  ASSERT_NE(summaries[0].projection, nullptr);
  const auto projection = *summaries[0].projection;
  ASSERT_NE(summaries[1].integer_cast, nullptr);
  const auto integer_cast = *summaries[1].integer_cast;
  const auto* product_record =
      loom_symbolic_expr_lookup_product(&expression_context_, values[2]);
  ASSERT_NE(product_record, nullptr);
  ASSERT_EQ(product_record->factor_count, 2u);
  const auto product_snapshot = *product_record;
  const loom_value_id_t factors[] = {product_record->factors[0],
                                     product_record->factors[1]};
  const auto expression = Expression(selected);
  loom_symbolic_expr_t zero = {};
  loom_symbolic_expr_constant(0, &zero);
  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_le(&expression_context_, &zero,
                                             &expression, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_UNKNOWN);
  for (int i = 0; i < 3; ++i) {
    ASSERT_NO_FATAL_FAILURE(ExpectSummary(values[i], summaries[i]));
  }
  EXPECT_TRUE(
      loom_symbolic_projection_equal(summaries[0].projection, &projection));
  EXPECT_EQ(summaries[1].integer_cast->source_value_id,
            integer_cast.source_value_id);
  EXPECT_TRUE(loom_symbolic_integer_cast_same_interpretation(
      summaries[1].integer_cast, &integer_cast));
  EXPECT_EQ(loom_symbolic_expr_lookup_product(&expression_context_, values[2]),
            product_record);
  EXPECT_EQ(product_record->scale, product_snapshot.scale);
  ASSERT_EQ(product_record->factor_count, 2u);
  EXPECT_EQ(product_record->factors[0], factors[0]);
  EXPECT_EQ(product_record->factors[1], factors[1]);
}

class SymbolicProofFailureTest : public SymbolicProofLifetimeTest {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<SymbolicProofFailureTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        test->allocation_count_++ == test->failure_index_) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected proof allocation failure");
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  void InitializeStorage() {
    allocation_count_ = 0;
    failure_index_ = SIZE_MAX;
    iree_arena_block_pool_initialize(4096, {this, Allocate}, &proof_pool_);
    iree_arena_initialize(&proof_pool_, &proof_arena_);
    loom_symbolic_expr_context_initialize(module_, nullptr, &fact_table_,
                                          &proof_arena_, &expression_context_);
  }

  void DeinitializeStorage() {
    iree_arena_deinitialize(&proof_arena_);
    iree_arena_block_pool_deinitialize(&proof_pool_);
  }

  void SetUp() override {
    SymbolicProofLifetimeTest::SetUp();
    InitializeStorage();
  }

  void TearDown() override {
    DeinitializeStorage();
    SymbolicProofLifetimeTest::TearDown();
  }

  // Storage shared by retained symbolic state and each temporary proof arena.
  iree_arena_block_pool_t proof_pool_ = {};
  // Retained canonical memo, independent of source and fact storage.
  iree_arena_allocator_t proof_arena_ = {};
  // Attempted system allocations since the current proof began.
  iree_host_size_t allocation_count_ = 0;
  // Allocation attempt to fail, or SIZE_MAX for normal allocation.
  iree_host_size_t failure_index_ = SIZE_MAX;
};

TEST_F(SymbolicProofFailureTest, WarmProofReusesPooledStorageAtHighValueIds) {
  // A tiny query near the end of a large function must not repeatedly allocate
  // dense temporary arrays spanning unrelated source identities.
  for (int i = 0; i < 16384; ++i) {
    DefineIndexValue();
  }
  const auto value = ComplementarySum();
  const auto sum = Expression(value);
  loom_symbolic_expr_t one = {};
  loom_symbolic_expr_constant(1, &one);
  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(
      loom_symbolic_expr_prove_le(&expression_context_, &one, &sum, &proof));
  ASSERT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
  const auto retained_bytes = proof_arena_.used_allocation_size;
  const auto allocation_count = allocation_count_;
  for (int i = 0; i < 128; ++i) {
    IREE_ASSERT_OK(
        loom_symbolic_expr_prove_le(&expression_context_, &one, &sum, &proof));
    ASSERT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
  }
  EXPECT_EQ(allocation_count_, allocation_count);
  EXPECT_EQ(proof_arena_.used_allocation_size, retained_bytes);
}

TEST_F(SymbolicProofFailureTest, FailedProofPreservesSummariesAndCanRetry) {
  const auto value = WideComplementarySum();
  loom_symbolic_expr_t minimum = {};
  loom_symbolic_expr_constant(462, &minimum);
  const auto sum = Expression(value);
  allocation_count_ = 0;
  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_le(&expression_context_, &minimum,
                                             &sum, &proof));
  ASSERT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
  const auto allocation_count = allocation_count_;
  ASSERT_GT(allocation_count, 0u);
  for (iree_host_size_t failure = 0; failure < allocation_count; ++failure) {
    SCOPED_TRACE(failure);
    DeinitializeStorage();
    InitializeStorage();
    const auto retained_sum = Expression(value);
    loom_symbolic_expr_summary_t prepared = {};
    ASSERT_TRUE(loom_symbolic_expr_context_try_lookup_summary(
        &expression_context_, value, &prepared));
    allocation_count_ = 0;
    failure_index_ = failure;
    IREE_ASSERT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        loom_symbolic_expr_prove_le(&expression_context_, &minimum,
                                    &retained_sum, &proof));
    ASSERT_NO_FATAL_FAILURE(ExpectSummary(value, prepared));
    EXPECT_EQ(expression_context_.condition_scope, nullptr);
    failure_index_ = SIZE_MAX;
    IREE_ASSERT_OK(loom_symbolic_expr_prove_le(&expression_context_, &minimum,
                                               &retained_sum, &proof));
    EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
    ASSERT_NO_FATAL_FAILURE(ExpectSummary(value, prepared));
  }
}

}  // namespace
}  // namespace loom
