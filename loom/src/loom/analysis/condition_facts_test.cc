// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/condition_facts.h"

#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/facts.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class ConditionFactsTest : public ::testing::Test {
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
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    loom_condition_query_initialize(module_, /*value_domain=*/nullptr,
                                    &analysis_arena_, &condition_query_);
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    IREE_ASSERT_OK(
        loom_value_fact_table_initialize(&fact_table_, &analysis_arena_, 16));
    loom_condition_fact_set_initialize(relation_storage_,
                                       IREE_ARRAYSIZE(relation_storage_),
                                       &condition_facts_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_value_id_t DefineValue(loom_type_t type) {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_value(&builder_, type, &value_id));
    return value_id;
  }

  loom_value_id_t DefineIndexValue() {
    return DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  }

  loom_value_id_t DefineI32Value() {
    return DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I32));
  }

  void DefineFacts(loom_value_id_t value_id, loom_value_facts_t facts) {
    IREE_CHECK_OK(loom_value_fact_table_define(&fact_table_, value_id, facts));
  }

  loom_op_t* BuildIndexCompare(loom_index_cmp_predicate_t predicate,
                               loom_value_id_t left, loom_value_id_t right) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_cmp_build(&builder_, predicate, left, right,
                                       LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_op_t* BuildScalarCompare(loom_scalar_cmpi_predicate_t predicate,
                                loom_value_id_t left, loom_value_id_t right) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_scalar_cmpi_build(&builder_, predicate, left, right,
                                         LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_op_t* BuildBoolAnd(loom_value_id_t left, loom_value_id_t right) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_scalar_andi_build(&builder_, left, right,
                                         loom_type_scalar(LOOM_SCALAR_TYPE_I1),
                                         LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_op_t* BuildBoolOr(loom_value_id_t left, loom_value_id_t right) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_scalar_ori_build(&builder_, left, right,
                                        loom_type_scalar(LOOM_SCALAR_TYPE_I1),
                                        LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_op_t* BuildBoolXor(loom_value_id_t left, loom_value_id_t right) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_scalar_xori_build(&builder_, left, right,
                                         loom_type_scalar(LOOM_SCALAR_TYPE_I1),
                                         LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  bool Query(loom_value_id_t condition_value, bool assumed_truth = true) {
    bool complete = false;
    IREE_CHECK_OK(loom_condition_facts_query(&condition_query_, &fact_table_,
                                             condition_value, assumed_truth,
                                             &condition_facts_, &complete));
    return complete;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_condition_query_t condition_query_;
  loom_builder_t builder_;
  loom_value_fact_table_t fact_table_;
  loom_condition_integer_relation_t relation_storage_[4];
  loom_condition_fact_set_t condition_facts_;
};

TEST_F(ConditionFactsTest, IndexCompareTrueEdgeProducesRelation) {
  loom_value_id_t induction = DefineIndexValue();
  loom_value_id_t upper_bound = DefineIndexValue();
  loom_op_t* compare =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, induction, upper_bound);

  Query(loom_index_cmp_result(compare));

  ASSERT_EQ(condition_facts_.integer_relation_count, 1u);
  const loom_condition_integer_relation_t& relation =
      condition_facts_.integer_relations[0];
  EXPECT_EQ(relation.relation, LOOM_SYMBOLIC_INTEGER_RELATION_LT);
  EXPECT_EQ(relation.left.kind, LOOM_CONDITION_INTEGER_OPERAND_VALUE);
  EXPECT_EQ(relation.left.value_id, induction);
  EXPECT_EQ(relation.right.kind, LOOM_CONDITION_INTEGER_OPERAND_VALUE);
  EXPECT_EQ(relation.right.value_id, upper_bound);
}

TEST_F(ConditionFactsTest, ComparisonCertificatePreservesCarrierAndIdentity) {
  loom_target_facts_t target_facts = {};
  target_facts.storage.snapshot.index_bitwidth = 32;
  fact_table_.context.target_facts = &target_facts;
  const loom_value_id_t left = DefineIndexValue();
  const loom_value_id_t right = DefineIndexValue();
  loom_condition_integer_comparison_t comparison = {};
  ASSERT_TRUE(loom_condition_integer_comparison_describe(
      module_, BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, left, right),
      &comparison));
  const loom_value_facts_t zero = loom_value_facts_exact_i64(0);
  const loom_value_facts_t sign_bit =
      loom_value_facts_exact_i64(INT64_C(1) << 31);
  bool result = false;
  EXPECT_TRUE(loom_condition_integer_comparison_evaluate(
      &comparison, nullptr, &zero, &sign_bit, &result));
  EXPECT_TRUE(result);
  EXPECT_FALSE(loom_condition_integer_comparison_evaluate(
      &comparison, &fact_table_.context, &zero, &sign_bit, &result));

  ASSERT_TRUE(loom_condition_integer_comparison_describe(
      module_, BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_ULT, left, right),
      &comparison));
  EXPECT_TRUE(loom_condition_integer_comparison_evaluate(
      &comparison, &fact_table_.context, &zero, &sign_bit, &result));
  EXPECT_TRUE(result);

  ASSERT_TRUE(loom_condition_integer_comparison_describe(
      module_, BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_EQ, left, left),
      &comparison));
  const loom_value_facts_t unknown = loom_value_facts_unknown();
  EXPECT_TRUE(loom_condition_integer_comparison_evaluate(
      &comparison, &fact_table_.context, &unknown, &unknown, &result));
  EXPECT_TRUE(result);
}

TEST_F(ConditionFactsTest, ComparisonCertificatePreservesSignedBooleanOrder) {
  const loom_value_id_t left =
      DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  const loom_value_id_t right =
      DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  for (bool signed_order : {false, true}) {
    loom_condition_integer_comparison_t comparison = {};
    ASSERT_TRUE(loom_condition_integer_comparison_describe(
        module_,
        BuildScalarCompare(signed_order ? LOOM_SCALAR_CMPI_PREDICATE_SLT
                                        : LOOM_SCALAR_CMPI_PREDICATE_ULT,
                           left, right),
        &comparison));
    for (int64_t a : {0, 1}) {
      for (int64_t b : {0, 1}) {
        const loom_value_facts_t lhs = loom_value_facts_exact_i64(a);
        const loom_value_facts_t rhs = loom_value_facts_exact_i64(b);
        bool result = false;
        ASSERT_TRUE(loom_condition_integer_comparison_evaluate(
            &comparison, nullptr, &lhs, &rhs, &result));
        EXPECT_EQ(result, signed_order ? a > b : a < b);
      }
    }
  }
}

TEST_F(ConditionFactsTest, ComparisonCertificateDoesNotDescribeConsequences) {
  const loom_value_id_t left = DefineIndexValue();
  const loom_value_id_t right = DefineIndexValue();
  const loom_value_id_t opaque =
      DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  loom_op_t* compare =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, left, right);
  loom_condition_integer_comparison_t comparison = {};
  EXPECT_FALSE(loom_condition_integer_comparison_describe(
      module_, BuildBoolAnd(loom_index_cmp_result(compare), opaque),
      &comparison));
}

TEST_F(ConditionFactsTest, DirectConditionQueryDoesNotAllocateScratch) {
  loom_value_id_t induction = DefineIndexValue();
  loom_value_id_t upper_bound = DefineIndexValue();
  loom_op_t* compare =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, induction, upper_bound);
  const iree_host_size_t used_allocation_size =
      analysis_arena_.used_allocation_size;

  ASSERT_TRUE(Query(loom_index_cmp_result(compare)));

  EXPECT_EQ(analysis_arena_.used_allocation_size, used_allocation_size);
}

TEST_F(ConditionFactsTest, LocalDomainRegistersHighIdsCreatedAfterAcquisition) {
  for (iree_host_size_t i = 0; i < 4096; ++i) {
    DefineIndexValue();
  }

  loom_local_value_domain_t value_domain = {};
  IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
      module_, module_->body, &analysis_arena_, &value_domain));
  loom_condition_query_initialize(module_, &value_domain, &analysis_arena_,
                                  &condition_query_);

  loom_value_id_t induction = DefineIndexValue();
  loom_value_id_t upper_bound = DefineIndexValue();
  DefineFacts(induction, loom_value_facts_exact_i64(0));
  DefineFacts(upper_bound, loom_value_facts_exact_i64(1));
  loom_op_t* compare =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, induction, upper_bound);
  ASSERT_GT(loom_index_cmp_result(compare), 4096u);
  const loom_value_id_t condition = loom_scalar_andi_result(BuildBoolAnd(
      loom_index_cmp_result(compare), loom_index_cmp_result(compare)));

  EXPECT_TRUE(Query(condition));
  EXPECT_NE(loom_local_value_domain_try_ordinal(&value_domain,
                                                loom_index_cmp_result(compare)),
            LOOM_VALUE_ORDINAL_INVALID);
  EXPECT_LT(condition_query_.value_state_capacity, 64u);

  bool proven_condition = false;
  bool proven = false;
  IREE_ASSERT_OK(loom_condition_fact_set_proves_condition(
      &condition_query_, &fact_table_, &condition_facts_, condition,
      &proven_condition, &proven));
  EXPECT_TRUE(proven);
  EXPECT_TRUE(proven_condition);
  EXPECT_NE(loom_local_value_domain_try_ordinal(&value_domain, condition),
            LOOM_VALUE_ORDINAL_INVALID);

  loom_local_value_domain_release(&value_domain);
}

TEST_F(ConditionFactsTest, RepeatedConditionDoesNotDuplicateRelation) {
  loom_value_id_t induction = DefineIndexValue();
  loom_value_id_t upper_bound = DefineIndexValue();
  loom_op_t* compare =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, induction, upper_bound);
  const loom_value_id_t condition = loom_index_cmp_result(compare);

  ASSERT_TRUE(Query(condition));
  bool complete = false;
  IREE_ASSERT_OK(loom_condition_facts_query_into(
      &condition_query_, &fact_table_, condition,
      /*assumed_truth=*/true, &condition_facts_, &complete));
  ASSERT_TRUE(complete);

  EXPECT_EQ(condition_facts_.integer_relation_count, 1u);
}

TEST_F(ConditionFactsTest, IndexCompareFalseEdgeInvertsRelation) {
  loom_value_id_t induction = DefineIndexValue();
  loom_value_id_t upper_bound = DefineIndexValue();
  loom_op_t* compare =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, induction, upper_bound);

  Query(loom_index_cmp_result(compare), false);

  ASSERT_EQ(condition_facts_.integer_relation_count, 1u);
  EXPECT_EQ(condition_facts_.integer_relations[0].relation,
            LOOM_SYMBOLIC_INTEGER_RELATION_GE);
}

TEST_F(ConditionFactsTest, ExactOperandFactsPreserveStructuralRelation) {
  loom_value_id_t induction = DefineIndexValue();
  loom_value_id_t upper_bound = DefineIndexValue();
  DefineFacts(upper_bound, loom_value_facts_exact_i64(16));
  loom_op_t* compare =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, induction, upper_bound);

  Query(loom_index_cmp_result(compare));

  ASSERT_EQ(condition_facts_.integer_relation_count, 1u);
  const loom_condition_integer_relation_t& relation =
      condition_facts_.integer_relations[0];
  EXPECT_EQ(relation.left.kind, LOOM_CONDITION_INTEGER_OPERAND_VALUE);
  EXPECT_EQ(relation.left.value_id, induction);
  EXPECT_EQ(relation.right.kind, LOOM_CONDITION_INTEGER_OPERAND_VALUE);
  EXPECT_EQ(relation.right.value_id, upper_bound);
}

TEST_F(ConditionFactsTest, ExactOperandFactsProveEquivalentLiteralRelation) {
  loom_value_id_t value = DefineIndexValue();
  loom_value_id_t expected = DefineIndexValue();
  DefineFacts(expected, loom_value_facts_exact_i64(32));
  loom_op_t* compare =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_EQ, value, expected);
  Query(loom_index_cmp_result(compare), /*assumed_truth=*/false);

  const loom_condition_integer_relation_t queried = {
      /*.relation=*/LOOM_SYMBOLIC_INTEGER_RELATION_EQ,
      /*.left=*/
      {
          /*.kind=*/LOOM_CONDITION_INTEGER_OPERAND_VALUE,
          /*.value_id=*/value,
          /*.constant=*/0,
      },
      /*.right=*/
      {
          /*.kind=*/LOOM_CONDITION_INTEGER_OPERAND_CONSTANT,
          /*.value_id=*/LOOM_VALUE_ID_INVALID,
          /*.constant=*/32,
      },
  };
  bool result = true;
  EXPECT_TRUE(loom_condition_fact_set_proves_integer_relation(
      &condition_facts_, &fact_table_, &queried, &result));
  EXPECT_FALSE(result);
}

TEST_F(ConditionFactsTest, SignedBooleanOrderUsesLogicalRangeRelations) {
  const loom_value_id_t left =
      DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  const loom_value_id_t right =
      DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  DefineFacts(left, loom_value_facts_make(0, 1, 1));
  DefineFacts(right, loom_value_facts_make(0, 1, 1));
  const loom_value_id_t signed_less = loom_scalar_cmpi_result(
      BuildScalarCompare(LOOM_SCALAR_CMPI_PREDICATE_SLT, left, right));
  const loom_value_id_t unsigned_greater = loom_scalar_cmpi_result(
      BuildScalarCompare(LOOM_SCALAR_CMPI_PREDICATE_UGT, left, right));

  ASSERT_TRUE(Query(signed_less));
  ASSERT_EQ(condition_facts_.integer_relation_count, 1u);
  EXPECT_EQ(condition_facts_.integer_relations[0].relation,
            LOOM_SYMBOLIC_INTEGER_RELATION_GT);

  bool value = false;
  bool proven = false;
  IREE_ASSERT_OK(loom_condition_fact_set_proves_condition(
      &condition_query_, &fact_table_, &condition_facts_, unsigned_greater,
      &value, &proven));
  EXPECT_TRUE(proven);
  EXPECT_TRUE(value);

  ASSERT_TRUE(Query(unsigned_greater));
  IREE_ASSERT_OK(loom_condition_fact_set_proves_condition(
      &condition_query_, &fact_table_, &condition_facts_, signed_less, &value,
      &proven));
  EXPECT_TRUE(proven);
  EXPECT_TRUE(value);
}

TEST_F(ConditionFactsTest, AppliesConstantRelationToValueFacts) {
  loom_value_id_t induction = DefineIndexValue();
  loom_value_id_t upper_bound = DefineIndexValue();
  DefineFacts(upper_bound, loom_value_facts_exact_i64(16));
  loom_op_t* compare =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, induction, upper_bound);
  Query(loom_index_cmp_result(compare));

  loom_value_facts_t induction_facts = loom_value_facts_make(0, 100, 1);
  EXPECT_TRUE(loom_condition_fact_set_apply_to_value_facts(
      &condition_facts_, &fact_table_, induction, &induction_facts));
  EXPECT_EQ(induction_facts.range_lo, 0);
  EXPECT_EQ(induction_facts.range_hi, 15);
}

TEST_F(ConditionFactsTest, AppliesSwappedConstantRelationToValueFacts) {
  loom_value_id_t lower_bound = DefineIndexValue();
  loom_value_id_t induction = DefineIndexValue();
  DefineFacts(lower_bound, loom_value_facts_exact_i64(4));
  loom_op_t* compare =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, lower_bound, induction);
  Query(loom_index_cmp_result(compare));

  loom_value_facts_t induction_facts = loom_value_facts_make(0, 100, 1);
  EXPECT_TRUE(loom_condition_fact_set_apply_to_value_facts(
      &condition_facts_, &fact_table_, induction, &induction_facts));
  EXPECT_EQ(induction_facts.range_lo, 5);
  EXPECT_EQ(induction_facts.range_hi, 100);
}

TEST_F(ConditionFactsTest, AppliesBothSidesOfDynamicIntervalRelation) {
  loom_value_id_t lane = DefineIndexValue();
  loom_value_id_t length = DefineIndexValue();
  DefineFacts(lane, loom_value_facts_make(0, 255, 1));
  DefineFacts(length, loom_value_facts_make(0, 1, 1));
  loom_op_t* compare =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_ULT, lane, length);
  ASSERT_TRUE(Query(loom_index_cmp_result(compare)));

  loom_value_facts_t lane_facts =
      loom_value_fact_table_lookup(&fact_table_, lane);
  loom_value_facts_t length_facts =
      loom_value_fact_table_lookup(&fact_table_, length);
  EXPECT_TRUE(loom_condition_fact_set_apply_to_value_facts(
      &condition_facts_, &fact_table_, lane, &lane_facts));
  EXPECT_TRUE(loom_condition_fact_set_apply_to_value_facts(
      &condition_facts_, &fact_table_, length, &length_facts));
  EXPECT_TRUE(loom_value_facts_is_zero(lane_facts));
  EXPECT_TRUE(loom_value_facts_is_exact(length_facts));
  EXPECT_EQ(length_facts.range_lo, 1);

  // The opposite edge does not prove a positive length: lane=length=0
  // satisfies it and must remain possible.
  ASSERT_TRUE(Query(loom_index_cmp_result(compare), /*assumed_truth=*/false));
  length_facts = loom_value_fact_table_lookup(&fact_table_, length);
  loom_condition_fact_set_apply_to_value_facts(&condition_facts_, &fact_table_,
                                               length, &length_facts);
  EXPECT_EQ(length_facts.range_lo, 0);
}

TEST_F(ConditionFactsTest, EdgeFactsProveWiderScalarCompareTrue) {
  loom_value_id_t lane = DefineI32Value();
  loom_value_id_t outer_bound = DefineI32Value();
  loom_value_id_t inner_bound = DefineI32Value();
  DefineFacts(outer_bound, loom_value_facts_exact_i64(8));
  DefineFacts(inner_bound, loom_value_facts_exact_i64(16));
  loom_op_t* outer =
      BuildScalarCompare(LOOM_SCALAR_CMPI_PREDICATE_SLT, lane, outer_bound);
  ASSERT_TRUE(Query(loom_scalar_cmpi_result(outer)));

  loom_op_t* inner =
      BuildScalarCompare(LOOM_SCALAR_CMPI_PREDICATE_SLT, lane, inner_bound);
  bool condition = false;
  bool proven = false;
  IREE_ASSERT_OK(loom_condition_fact_set_proves_condition(
      &condition_query_, &fact_table_, &condition_facts_,
      loom_scalar_cmpi_result(inner), &condition, &proven));
  EXPECT_TRUE(proven);
  EXPECT_TRUE(condition);
}

TEST_F(ConditionFactsTest, EdgeFactsProveDisjointIndexCompareFalse) {
  loom_value_id_t lane = DefineIndexValue();
  loom_value_id_t outer_bound = DefineIndexValue();
  loom_value_id_t inner_bound = DefineIndexValue();
  DefineFacts(outer_bound, loom_value_facts_exact_i64(8));
  DefineFacts(inner_bound, loom_value_facts_exact_i64(16));
  loom_op_t* outer =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, lane, outer_bound);
  ASSERT_TRUE(Query(loom_index_cmp_result(outer)));

  loom_op_t* inner =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SGE, lane, inner_bound);
  bool condition = true;
  bool proven = false;
  IREE_ASSERT_OK(loom_condition_fact_set_proves_condition(
      &condition_query_, &fact_table_, &condition_facts_,
      loom_index_cmp_result(inner), &condition, &proven));
  EXPECT_TRUE(proven);
  EXPECT_FALSE(condition);
}

TEST_F(ConditionFactsTest, SharedBooleanDagProofIsMemoized) {
  loom_value_id_t lane = DefineI32Value();
  loom_value_id_t outer_bound = DefineI32Value();
  loom_value_id_t inner_bound = DefineI32Value();
  DefineFacts(outer_bound, loom_value_facts_exact_i64(8));
  DefineFacts(inner_bound, loom_value_facts_exact_i64(16));
  loom_op_t* outer =
      BuildScalarCompare(LOOM_SCALAR_CMPI_PREDICATE_SLT, lane, outer_bound);
  ASSERT_TRUE(Query(loom_scalar_cmpi_result(outer)));

  loom_op_t* inner =
      BuildScalarCompare(LOOM_SCALAR_CMPI_PREDICATE_SLT, lane, inner_bound);
  loom_value_id_t condition = loom_scalar_cmpi_result(inner);
  for (int i = 0; i < 64; ++i) {
    condition = loom_scalar_andi_result(BuildBoolAnd(condition, condition));
  }

  bool condition_value = false;
  bool proven = false;
  IREE_ASSERT_OK(loom_condition_fact_set_proves_condition(
      &condition_query_, &fact_table_, &condition_facts_, condition,
      &condition_value, &proven));
  EXPECT_TRUE(proven);
  EXPECT_TRUE(condition_value);
}

TEST_F(ConditionFactsTest, UnsignedCompareRequiresNonNegativeOperandFacts) {
  loom_value_id_t induction = DefineIndexValue();
  loom_value_id_t upper_bound = DefineIndexValue();
  loom_op_t* compare =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_ULT, induction, upper_bound);

  Query(loom_index_cmp_result(compare));

  EXPECT_EQ(condition_facts_.integer_relation_count, 0u);
}

TEST_F(ConditionFactsTest, UnsignedCompareUsesSignedRelationWhenNonNegative) {
  loom_value_id_t induction = DefineIndexValue();
  loom_value_id_t upper_bound = DefineIndexValue();
  DefineFacts(induction, loom_value_facts_make(0, 100, 1));
  DefineFacts(upper_bound, loom_value_facts_make(0, 100, 1));
  loom_op_t* compare =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_ULT, induction, upper_bound);

  Query(loom_index_cmp_result(compare), false);

  ASSERT_EQ(condition_facts_.integer_relation_count, 1u);
  EXPECT_EQ(condition_facts_.integer_relations[0].relation,
            LOOM_SYMBOLIC_INTEGER_RELATION_GE);
}

TEST_F(ConditionFactsTest, UnsignedBranchBoundsPreserveSignedBoundaryValues) {
  const loom_index_cmp_predicate_t index_predicates[] = {
      LOOM_INDEX_CMP_PREDICATE_ULT, LOOM_INDEX_CMP_PREDICATE_ULE,
      LOOM_INDEX_CMP_PREDICATE_UGT, LOOM_INDEX_CMP_PREDICATE_UGE};
  const loom_scalar_cmpi_predicate_t scalar_predicates[] = {
      LOOM_SCALAR_CMPI_PREDICATE_ULT, LOOM_SCALAR_CMPI_PREDICATE_ULE,
      LOOM_SCALAR_CMPI_PREDICATE_UGT, LOOM_SCALAR_CMPI_PREDICATE_UGE};
  const int64_t samples[] = {INT64_MIN,     -2,       -1, 0, 1, 7, 8,
                             INT64_MAX - 1, INT64_MAX};
  const int64_t upper_ranges[][2] = {{0, 0},         {1, 1},   {0, 8},
                                     {0, INT64_MAX}, {-2, -1}, {-2, 8}};
  for (auto type : {LOOM_SCALAR_TYPE_I64, LOOM_SCALAR_TYPE_INDEX}) {
    for (const auto& upper_range : upper_ranges) {
      const loom_value_id_t value = DefineValue(loom_type_scalar(type));
      const loom_value_id_t bound = DefineValue(loom_type_scalar(type));
      DefineFacts(bound,
                  loom_value_facts_make(upper_range[0], upper_range[1], 1));
      for (int predicate = 0; predicate < 4; ++predicate) {
        for (bool swapped : {false, true}) {
          const loom_value_id_t left = swapped ? bound : value;
          const loom_value_id_t right = swapped ? value : bound;
          loom_op_t* compare =
              type == LOOM_SCALAR_TYPE_INDEX
                  ? BuildIndexCompare(index_predicates[predicate], left, right)
                  : BuildScalarCompare(scalar_predicates[predicate], left,
                                       right);
          for (bool truth : {false, true}) {
            SCOPED_TRACE(::testing::Message()
                         << "type=" << type << ", predicate=" << predicate
                         << ", swapped=" << swapped << ", truth=" << truth
                         << ", range=" << upper_range[0] << ':'
                         << upper_range[1]);
            ASSERT_TRUE(Query(loom_op_const_results(compare)[0], truth));
            loom_value_facts_t refined = loom_value_facts_unknown();
            loom_condition_fact_set_apply_to_value_facts(
                &condition_facts_, &fact_table_, value, &refined);
            if (upper_range[0] == 0 && upper_range[1] == 8 &&
                truth == ((predicate < 2) != swapped)) {
              EXPECT_EQ(refined.range_lo, 0);
              EXPECT_EQ(refined.range_hi,
                        (predicate % 2 == 0) == truth ? 7 : 8);
            }
            for (int64_t input : samples) {
              for (int64_t limit : samples) {
                if (limit < upper_range[0] || limit > upper_range[1]) {
                  continue;
                }
                const uint64_t a = (uint64_t)(swapped ? limit : input);
                const uint64_t b = (uint64_t)(swapped ? input : limit);
                const bool outcomes[] = {(a < b), (a <= b), (a > b), (a >= b)};
                if (outcomes[predicate] == truth) {
                  EXPECT_LE(refined.range_lo, input);
                  EXPECT_GE(refined.range_hi, input);
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST_F(ConditionFactsTest, UnsignedBoundMustFitSignedTargetCarrier) {
  loom_target_facts_t target_facts = {};
  target_facts.storage.snapshot.index_bitwidth = 32;
  fact_table_.context.target_facts = &target_facts;
  const loom_value_id_t value = DefineIndexValue();
  for (int64_t bound_value :
       {INT64_C(2147483647), INT64_C(2147483648), INT64_C(4294967296)}) {
    const loom_value_id_t bound = DefineIndexValue();
    DefineFacts(bound, loom_value_facts_exact_i64(bound_value));
    loom_op_t* compare =
        BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_ULT, value, bound);
    ASSERT_TRUE(Query(loom_index_cmp_result(compare)));
    loom_value_facts_t refined = loom_value_facts_unknown();
    loom_condition_fact_set_apply_to_value_facts(&condition_facts_,
                                                 &fact_table_, value, &refined);
    if (bound_value == INT32_MAX) {
      EXPECT_EQ(refined.range_lo, 0);
      EXPECT_EQ(refined.range_hi, INT32_MAX - 1);
    } else {
      EXPECT_EQ(refined.range_lo, INT64_MIN);
      EXPECT_EQ(refined.range_hi, INT64_MAX);
    }
  }
}

TEST_F(ConditionFactsTest, ScalarCmpiProducesIntegerRelation) {
  loom_value_id_t left = DefineI32Value();
  loom_value_id_t right = DefineI32Value();
  loom_op_t* compare =
      BuildScalarCompare(LOOM_SCALAR_CMPI_PREDICATE_EQ, left, right);

  Query(loom_scalar_cmpi_result(compare), false);

  ASSERT_EQ(condition_facts_.integer_relation_count, 1u);
  const loom_condition_integer_relation_t& relation =
      condition_facts_.integer_relations[0];
  EXPECT_EQ(relation.relation, LOOM_SYMBOLIC_INTEGER_RELATION_NE);
  EXPECT_EQ(relation.left.value_id, left);
  EXPECT_EQ(relation.right.value_id, right);
}

TEST_F(ConditionFactsTest, RelationProofPreservesFactIdentityOperands) {
  const loom_value_id_t left = DefineIndexValue();
  const loom_value_id_t right = DefineIndexValue();
  const loom_value_id_t other = DefineIndexValue();
  const loom_type_t types[] = {loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                               loom_type_scalar(LOOM_SCALAR_TYPE_INDEX)};
  const loom_value_id_t values[] = {left, right};
  loom_op_t* aliases = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(&builder_, values, 2, nullptr, 0,
                                         types, 2, LOOM_LOCATION_UNKNOWN,
                                         &aliases));
  IREE_ASSERT_OK(
      loom_value_fact_table_compute_op(&fact_table_, module_, aliases));
  loom_op_t* repeated_aliases = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(
      &builder_, loom_op_const_results(aliases), 2, nullptr, 0, types, 2,
      LOOM_LOCATION_UNKNOWN, &repeated_aliases));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&fact_table_, module_,
                                                  repeated_aliases));

  const loom_op_t* compare =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_EQ, left, right);
  ASSERT_TRUE(Query(loom_index_cmp_result(compare)));
  loom_condition_integer_relation_t queried =
      condition_facts_.integer_relations[0];
  queried.left.value_id = loom_op_const_results(repeated_aliases)[0];
  queried.right.value_id = loom_op_const_results(repeated_aliases)[1];
  bool result = false;
  EXPECT_TRUE(loom_condition_fact_set_proves_integer_relation(
      &condition_facts_, &fact_table_, &queried, &result));
  EXPECT_TRUE(result);

  queried.relation = LOOM_SYMBOLIC_INTEGER_RELATION_NE;
  EXPECT_TRUE(loom_condition_fact_set_proves_integer_relation(
      &condition_facts_, &fact_table_, &queried, &result));
  EXPECT_FALSE(result);
  queried.right.value_id = other;
  EXPECT_FALSE(loom_condition_fact_set_proves_integer_relation(
      &condition_facts_, &fact_table_, &queried, &result));
}

TEST_F(ConditionFactsTest, ScalarIdentityPreservesRangeRefinement) {
  const loom_value_id_t input = DefineI32Value();
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* alias = nullptr;
  IREE_ASSERT_OK(loom_scalar_assume_build(&builder_, &input, 1, nullptr, 0,
                                          &type, 1, LOOM_LOCATION_UNKNOWN,
                                          &alias));
  IREE_ASSERT_OK(
      loom_value_fact_table_compute_op(&fact_table_, module_, alias));
  const loom_value_id_t result = loom_op_const_results(alias)[0];
  loom_condition_integer_relation_t relation = {};
  relation.relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT;
  relation.left.kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE;
  relation.left.value_id = input;
  relation.right.kind = LOOM_CONDITION_INTEGER_OPERAND_CONSTANT;
  relation.right.constant = 16;
  loom_value_facts_t facts = loom_value_facts_unknown();
  EXPECT_TRUE(loom_condition_integer_relation_apply_to_value_facts(
      &relation, &fact_table_, result, &facts));
  EXPECT_EQ(facts.range_hi, 15);
  // Edge-local refinement does not change the ambient source facts.
  EXPECT_TRUE(loom_value_facts_is_unknown(
      loom_value_fact_table_lookup(&fact_table_, input)));
}

TEST_F(ConditionFactsTest, BooleanAndTrueEdgeConjoinsRelations) {
  loom_value_id_t induction = DefineIndexValue();
  loom_value_id_t upper_bound = DefineIndexValue();
  loom_value_id_t lane = DefineIndexValue();
  loom_value_id_t lane_limit = DefineIndexValue();
  loom_op_t* first =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, induction, upper_bound);
  loom_op_t* second =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_NE, lane, lane_limit);
  loom_op_t* conjunction =
      BuildBoolAnd(loom_index_cmp_result(first), loom_index_cmp_result(second));

  Query(loom_scalar_andi_result(conjunction));

  ASSERT_EQ(condition_facts_.integer_relation_count, 2u);
  EXPECT_EQ(condition_facts_.integer_relations[0].relation,
            LOOM_SYMBOLIC_INTEGER_RELATION_LT);
  EXPECT_EQ(condition_facts_.integer_relations[0].left.value_id, induction);
  EXPECT_EQ(condition_facts_.integer_relations[0].right.value_id, upper_bound);
  EXPECT_EQ(condition_facts_.integer_relations[1].relation,
            LOOM_SYMBOLIC_INTEGER_RELATION_NE);
  EXPECT_EQ(condition_facts_.integer_relations[1].left.value_id, lane);
  EXPECT_EQ(condition_facts_.integer_relations[1].right.value_id, lane_limit);
}

TEST_F(ConditionFactsTest, BooleanOrFalseEdgeConjoinsInvertedRelations) {
  loom_value_id_t induction = DefineIndexValue();
  loom_value_id_t upper_bound = DefineIndexValue();
  loom_value_id_t lane = DefineIndexValue();
  loom_value_id_t lane_limit = DefineIndexValue();
  loom_op_t* first =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, induction, upper_bound);
  loom_op_t* second =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_NE, lane, lane_limit);
  loom_op_t* disjunction =
      BuildBoolOr(loom_index_cmp_result(first), loom_index_cmp_result(second));

  Query(loom_scalar_ori_result(disjunction), false);

  ASSERT_EQ(condition_facts_.integer_relation_count, 2u);
  EXPECT_EQ(condition_facts_.integer_relations[0].relation,
            LOOM_SYMBOLIC_INTEGER_RELATION_GE);
  EXPECT_EQ(condition_facts_.integer_relations[1].relation,
            LOOM_SYMBOLIC_INTEGER_RELATION_EQ);
}

TEST_F(ConditionFactsTest, BooleanAndFalseEdgeIsDisjunctiveWithoutKnownSide) {
  loom_value_id_t induction = DefineIndexValue();
  loom_value_id_t upper_bound = DefineIndexValue();
  loom_value_id_t lane = DefineIndexValue();
  loom_value_id_t lane_limit = DefineIndexValue();
  loom_op_t* first =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, induction, upper_bound);
  loom_op_t* second =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_NE, lane, lane_limit);
  loom_op_t* conjunction =
      BuildBoolAnd(loom_index_cmp_result(first), loom_index_cmp_result(second));

  Query(loom_scalar_andi_result(conjunction), false);

  EXPECT_EQ(condition_facts_.integer_relation_count, 0u);
}

TEST_F(ConditionFactsTest, DeepBooleanConditionDerivesFacts) {
  loom_value_id_t induction = DefineIndexValue();
  loom_value_id_t upper_bound = DefineIndexValue();
  loom_op_t* compare =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, induction, upper_bound);
  loom_value_id_t condition = loom_index_cmp_result(compare);
  for (int i = 0; i < 64; ++i) {
    condition = loom_scalar_andi_result(
        BuildBoolAnd(condition, loom_index_cmp_result(compare)));
  }

  ASSERT_TRUE(Query(condition));
  ASSERT_EQ(condition_facts_.integer_relation_count, 1u);
  EXPECT_EQ(condition_facts_.integer_relations[0].relation,
            LOOM_SYMBOLIC_INTEGER_RELATION_LT);
}

TEST_F(ConditionFactsTest, CompleteQueryRetainsAllRelationsAndTruth) {
  constexpr iree_host_size_t kRelationCount = 64;
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  loom_value_id_t compare_results[kRelationCount];
  for (iree_host_size_t i = 0; i < kRelationCount; ++i) {
    const loom_value_id_t left = DefineIndexValue();
    const loom_value_id_t right = DefineIndexValue();
    compare_results[i] = loom_index_cmp_result(
        BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, left, right));
    condition = condition == LOOM_VALUE_ID_INVALID
                    ? compare_results[i]
                    : loom_scalar_andi_result(
                          BuildBoolAnd(condition, compare_results[i]));
  }

  loom_condition_derivation_t derivation;
  loom_condition_derivation_initialize(&analysis_arena_, &derivation);
  IREE_ASSERT_OK(loom_condition_facts_query_complete(
      &condition_query_, &fact_table_, condition, /*assumed_truth=*/true,
      &derivation));

  ASSERT_EQ(derivation.integer_facts.integer_relation_count, kRelationCount);
  ASSERT_EQ(derivation.boolean_fact_count, kRelationCount * 2 - 1);
  for (loom_value_id_t compare_result : compare_results) {
    bool found = false;
    for (iree_host_size_t i = 0; i < derivation.boolean_fact_count; ++i) {
      found |= derivation.boolean_facts[i].value_id == compare_result &&
               derivation.boolean_facts[i].value;
    }
    EXPECT_TRUE(found);
  }

  loom_condition_integer_relation_t* relation_storage =
      derivation.integer_facts.integer_relations;
  loom_condition_boolean_fact_t* boolean_storage = derivation.boolean_facts;
  const iree_host_size_t relation_capacity =
      derivation.integer_facts.integer_relation_capacity;
  const iree_host_size_t boolean_capacity = derivation.boolean_fact_capacity;
  IREE_ASSERT_OK(loom_condition_facts_query_complete(
      &condition_query_, &fact_table_, compare_results[0],
      /*assumed_truth=*/true, &derivation));
  EXPECT_EQ(derivation.integer_facts.integer_relation_count, 1u);
  EXPECT_EQ(derivation.boolean_fact_count, 1u);
  EXPECT_EQ(derivation.integer_facts.integer_relations, relation_storage);
  EXPECT_EQ(derivation.boolean_facts, boolean_storage);
  EXPECT_EQ(derivation.integer_facts.integer_relation_capacity,
            relation_capacity);
  EXPECT_EQ(derivation.boolean_fact_capacity, boolean_capacity);
}

TEST_F(ConditionFactsTest, CompleteQueryDeduplicatesEquivalentRelations) {
  constexpr iree_host_size_t kCompareCount = 64;
  const loom_value_id_t left = DefineIndexValue();
  const loom_value_id_t right = DefineIndexValue();
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  for (iree_host_size_t i = 0; i < kCompareCount; ++i) {
    const loom_value_id_t compare = loom_index_cmp_result(
        BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SGT, right, left));
    condition = condition == LOOM_VALUE_ID_INVALID
                    ? compare
                    : loom_scalar_andi_result(BuildBoolAnd(condition, compare));
  }

  loom_condition_derivation_t derivation;
  loom_condition_derivation_initialize(&analysis_arena_, &derivation);
  IREE_ASSERT_OK(loom_condition_facts_query_complete(
      &condition_query_, &fact_table_, condition, /*assumed_truth=*/true,
      &derivation));

  ASSERT_EQ(derivation.integer_facts.integer_relation_count, 1u);
  EXPECT_EQ(derivation.integer_facts.integer_relations[0].relation,
            LOOM_SYMBOLIC_INTEGER_RELATION_LT);
  EXPECT_EQ(derivation.integer_facts.integer_relations[0].left.value_id, left);
  EXPECT_EQ(derivation.integer_facts.integer_relations[0].right.value_id,
            right);
  EXPECT_EQ(derivation.boolean_fact_count, kCompareCount * 2 - 1);
}

TEST_F(ConditionFactsTest, CompleteConjunctionQueryCanonicalizesAllOutcomes) {
  const loom_value_id_t left = DefineIndexValue();
  const loom_value_id_t middle = DefineIndexValue();
  const loom_value_id_t right = DefineIndexValue();
  const loom_value_id_t less = loom_index_cmp_result(
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, left, middle));
  const loom_value_id_t equal = loom_index_cmp_result(
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_EQ, middle, right));
  const loom_condition_assumption_t assumptions[] = {
      {/*.condition=*/less, /*.assumed_truth=*/true},
      {/*.condition=*/equal, /*.assumed_truth=*/false},
      {/*.condition=*/less, /*.assumed_truth=*/true},
  };
  loom_condition_derivation_t derivation;
  loom_condition_derivation_initialize(&analysis_arena_, &derivation);

  IREE_ASSERT_OK(loom_condition_facts_query_conjunction_complete(
      &condition_query_, &fact_table_, assumptions, IREE_ARRAYSIZE(assumptions),
      &derivation));

  ASSERT_EQ(derivation.integer_facts.integer_relation_count, 2u);
  bool found_less = false;
  bool found_not_equal = false;
  for (iree_host_size_t i = 0;
       i < derivation.integer_facts.integer_relation_count; ++i) {
    found_less |= derivation.integer_facts.integer_relations[i].relation ==
                  LOOM_SYMBOLIC_INTEGER_RELATION_LT;
    found_not_equal |= derivation.integer_facts.integer_relations[i].relation ==
                       LOOM_SYMBOLIC_INTEGER_RELATION_NE;
  }
  EXPECT_TRUE(found_less);
  EXPECT_TRUE(found_not_equal);
  ASSERT_EQ(derivation.boolean_fact_count, 2u);
  EXPECT_EQ(derivation.boolean_facts[0].value_id, less);
  EXPECT_TRUE(derivation.boolean_facts[0].value);
  EXPECT_EQ(derivation.boolean_facts[1].value_id, equal);
  EXPECT_FALSE(derivation.boolean_facts[1].value);
}

TEST_F(ConditionFactsTest, OpaqueBooleanConditionProducesEdgeFact) {
  loom_value_id_t condition =
      DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1));

  ASSERT_TRUE(Query(condition));

  ASSERT_EQ(condition_facts_.integer_relation_count, 1u);
  const loom_condition_integer_relation_t& true_relation =
      condition_facts_.integer_relations[0];
  EXPECT_EQ(true_relation.relation, LOOM_SYMBOLIC_INTEGER_RELATION_EQ);
  EXPECT_EQ(true_relation.left.kind, LOOM_CONDITION_INTEGER_OPERAND_VALUE);
  EXPECT_EQ(true_relation.left.value_id, condition);
  EXPECT_EQ(true_relation.right.kind, LOOM_CONDITION_INTEGER_OPERAND_CONSTANT);
  EXPECT_EQ(true_relation.right.constant, 1);

  ASSERT_TRUE(Query(condition, /*assumed_truth=*/false));

  ASSERT_EQ(condition_facts_.integer_relation_count, 1u);
  EXPECT_EQ(condition_facts_.integer_relations[0].right.constant, 0);
}

TEST_F(ConditionFactsTest, CompleteQueryRetainsOpaqueBooleanTruth) {
  const loom_value_id_t condition =
      DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  loom_condition_derivation_t derivation;
  loom_condition_derivation_initialize(&analysis_arena_, &derivation);

  IREE_ASSERT_OK(loom_condition_facts_query_complete(
      &condition_query_, &fact_table_, condition, /*assumed_truth=*/false,
      &derivation));

  ASSERT_EQ(derivation.integer_facts.integer_relation_count, 1u);
  EXPECT_EQ(derivation.integer_facts.integer_relations[0].right.constant, 0);
  ASSERT_EQ(derivation.boolean_fact_count, 1u);
  EXPECT_EQ(derivation.boolean_facts[0].value_id, condition);
  EXPECT_FALSE(derivation.boolean_facts[0].value);
}

TEST_F(ConditionFactsTest, RetainedBooleanTruthProvesOpaqueCondition) {
  loom_value_id_t condition =
      DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  loom_value_id_t other = DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  const loom_value_fact_table_t* ambient_fact_tables[] = {&fact_table_,
                                                          nullptr};
  for (bool assumed_truth : {false, true}) {
    ASSERT_TRUE(Query(condition, assumed_truth));
    for (const loom_value_fact_table_t* ambient_facts : ambient_fact_tables) {
      bool proven_condition = !assumed_truth;
      bool proven = false;
      const iree_host_size_t used_allocation_size =
          analysis_arena_.used_allocation_size;
      IREE_ASSERT_OK(loom_condition_fact_set_proves_condition(
          &condition_query_, ambient_facts, &condition_facts_, condition,
          &proven_condition, &proven));
      EXPECT_TRUE(proven);
      EXPECT_EQ(proven_condition, assumed_truth);
      EXPECT_EQ(analysis_arena_.used_allocation_size, used_allocation_size);

      IREE_ASSERT_OK(loom_condition_fact_set_proves_condition(
          &condition_query_, ambient_facts, &condition_facts_, other,
          &proven_condition, &proven));
      EXPECT_FALSE(proven);
    }
  }

  loom_condition_fact_set_reset(&condition_facts_);
  bool proven_condition = false;
  bool proven = true;
  IREE_ASSERT_OK(loom_condition_fact_set_proves_condition(
      &condition_query_, &fact_table_, &condition_facts_, condition,
      &proven_condition, &proven));
  EXPECT_FALSE(proven);
}

TEST_F(ConditionFactsTest, BooleanCompositionConsumesRetainedOperandTruth) {
  loom_value_id_t left = DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  loom_value_id_t right = DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  const loom_value_id_t conditions[] = {
      loom_scalar_andi_result(BuildBoolAnd(left, right)),
      loom_scalar_ori_result(BuildBoolOr(left, right)),
      loom_scalar_xori_result(BuildBoolXor(left, right)),
  };
  for (bool left_truth : {false, true}) {
    for (bool right_truth : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "left=" << left_truth << ", right=" << right_truth);
      ASSERT_TRUE(Query(left, left_truth));
      bool complete = false;
      IREE_ASSERT_OK(loom_condition_facts_query_into(
          &condition_query_, &fact_table_, right, right_truth,
          &condition_facts_, &complete));
      ASSERT_TRUE(complete);
      const bool expected[] = {
          left_truth && right_truth,
          left_truth || right_truth,
          left_truth != right_truth,
      };
      for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(conditions); ++i) {
        bool proven_condition = !expected[i];
        bool proven = false;
        IREE_ASSERT_OK(loom_condition_fact_set_proves_condition(
            &condition_query_, &fact_table_, &condition_facts_, conditions[i],
            &proven_condition, &proven));
        EXPECT_TRUE(proven);
        EXPECT_EQ(proven_condition, expected[i]);
      }
    }
  }
}

TEST_F(ConditionFactsTest, PartialBooleanTruthPreservesUnknownOutcomes) {
  loom_value_id_t left = DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  loom_value_id_t right = DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  const loom_value_id_t conditions[] = {
      loom_scalar_andi_result(BuildBoolAnd(left, right)),
      loom_scalar_ori_result(BuildBoolOr(left, right)),
      loom_scalar_xori_result(BuildBoolXor(left, right)),
  };
  for (loom_value_id_t known : {left, right}) {
    for (bool assumed_truth : {false, true}) {
      ASSERT_TRUE(Query(known, assumed_truth));
      const bool expected_proven[] = {!assumed_truth, assumed_truth, false};
      for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(conditions); ++i) {
        bool proven_condition = !assumed_truth;
        bool proven = false;
        IREE_ASSERT_OK(loom_condition_fact_set_proves_condition(
            &condition_query_, &fact_table_, &condition_facts_, conditions[i],
            &proven_condition, &proven));
        EXPECT_EQ(proven, expected_proven[i]);
        if (proven) {
          EXPECT_EQ(proven_condition, assumed_truth);
        }
      }
    }
  }
}

static bool EvaluateBooleanOperation(int operation, bool left, bool right) {
  switch (operation) {
    case 0:
      return left && right;
    case 1:
      return left || right;
    case 2:
      return left != right;
    default:
      ADD_FAILURE() << "Invalid Boolean operation";
      return false;
  }
}

TEST_F(ConditionFactsTest, NestedBooleanProofMatchesAllCompletions) {
  const loom_value_id_t inputs[] = {
      DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1)),
      DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1)),
      DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1)),
  };
  const loom_value_id_t inner_conditions[] = {
      loom_scalar_andi_result(BuildBoolAnd(inputs[0], inputs[1])),
      loom_scalar_ori_result(BuildBoolOr(inputs[0], inputs[1])),
      loom_scalar_xori_result(BuildBoolXor(inputs[0], inputs[1])),
  };
  for (int inner = 0; inner < 3; ++inner) {
    for (bool gate_first : {false, true}) {
      const loom_value_id_t left =
          gate_first ? inputs[2] : inner_conditions[inner];
      const loom_value_id_t right =
          gate_first ? inner_conditions[inner] : inputs[2];
      const loom_value_id_t conditions[] = {
          loom_scalar_andi_result(BuildBoolAnd(left, right)),
          loom_scalar_ori_result(BuildBoolOr(left, right)),
          loom_scalar_xori_result(BuildBoolXor(left, right)),
      };
      // Each input is unknown, false or true. Enumerate concrete completions
      // independently of the proof's short-circuit evaluation.
      for (int assignment = 0; assignment < 27; ++assignment) {
        int states[3];
        int remaining = assignment;
        loom_condition_fact_set_reset(&condition_facts_);
        for (int i = 0; i < 3; ++i) {
          states[i] = remaining % 3;
          remaining /= 3;
          if (states[i] == 0) {
            continue;
          }
          condition_facts_
              .integer_relations[condition_facts_.integer_relation_count++] = {
              LOOM_SYMBOLIC_INTEGER_RELATION_EQ,
              {LOOM_CONDITION_INTEGER_OPERAND_VALUE, inputs[i], 0},
              {LOOM_CONDITION_INTEGER_OPERAND_CONSTANT, LOOM_VALUE_ID_INVALID,
               states[i] - 1},
          };
        }
        for (int outer = 0; outer < 3; ++outer) {
          SCOPED_TRACE(::testing::Message()
                       << "inner=" << inner << ", outer=" << outer
                       << ", gate_first=" << gate_first
                       << ", assignment=" << assignment);
          unsigned outcomes = 0;
          for (unsigned concrete = 0; concrete < 8; ++concrete) {
            bool compatible = true;
            for (int i = 0; i < 3; ++i) {
              if (states[i] != 0 &&
                  ((concrete >> i) & 1) != unsigned(states[i] - 1)) {
                compatible = false;
              }
            }
            if (!compatible) {
              continue;
            }
            const bool child = EvaluateBooleanOperation(
                inner, (concrete & 1) != 0, (concrete & 2) != 0);
            const bool gate = (concrete & 4) != 0;
            const bool value = EvaluateBooleanOperation(
                outer, gate_first ? gate : child, gate_first ? child : gate);
            outcomes |= value ? 2u : 1u;
          }
          bool value = false;
          bool proven = false;
          IREE_ASSERT_OK(loom_condition_fact_set_proves_condition(
              &condition_query_, &fact_table_, &condition_facts_,
              conditions[outer], &value, &proven));
          EXPECT_EQ(proven, outcomes != 3u);
          if (proven) {
            EXPECT_EQ(value ? 2u : 1u, outcomes);
          }
        }
      }
    }
  }
}

TEST_F(ConditionFactsTest, DynamicRelationsProveComparisonConditions) {
  const loom_value_id_t left = DefineIndexValue();
  const loom_value_id_t right = DefineIndexValue();
  const loom_value_id_t scalar_left = DefineI32Value();
  const loom_value_id_t scalar_right = DefineI32Value();
  const loom_value_id_t conditions[] = {
      loom_index_cmp_result(
          BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, left, right)),
      loom_index_cmp_result(
          BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SGE, left, right)),
      loom_scalar_cmpi_result(BuildScalarCompare(LOOM_SCALAR_CMPI_PREDICATE_SLT,
                                                 scalar_left, scalar_right)),
      loom_scalar_cmpi_result(BuildScalarCompare(LOOM_SCALAR_CMPI_PREDICATE_SGE,
                                                 scalar_left, scalar_right)),
  };
  for (int i = 0; i < 2; ++i) {
    condition_facts_.integer_relations[i] = {
        LOOM_SYMBOLIC_INTEGER_RELATION_GT,
        {LOOM_CONDITION_INTEGER_OPERAND_VALUE, i == 0 ? right : scalar_right,
         0},
        {LOOM_CONDITION_INTEGER_OPERAND_VALUE, i == 0 ? left : scalar_left, 0},
    };
  }
  condition_facts_.integer_relation_count = 2;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(conditions); ++i) {
    bool value = false;
    bool proven = false;
    IREE_ASSERT_OK(loom_condition_fact_set_proves_condition(
        &condition_query_, &fact_table_, &condition_facts_, conditions[i],
        &value, &proven));
    EXPECT_TRUE(proven);
    EXPECT_EQ(value, (i % 2) == 0);
  }
}

static bool EvaluateRelation(loom_symbolic_integer_relation_t relation,
                             int64_t left, int64_t right) {
  switch (relation) {
    case LOOM_SYMBOLIC_INTEGER_RELATION_EQ:
      return left == right;
    case LOOM_SYMBOLIC_INTEGER_RELATION_NE:
      return left != right;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LT:
      return left < right;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LE:
      return left <= right;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GT:
      return left > right;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GE:
      return left >= right;
  }
  ADD_FAILURE() << "Invalid relation";
  return false;
}

TEST_F(ConditionFactsTest, RelationProofMatchesConjunctionTruthTable) {
  const loom_symbolic_integer_relation_t relations[] = {
      LOOM_SYMBOLIC_INTEGER_RELATION_EQ, LOOM_SYMBOLIC_INTEGER_RELATION_NE,
      LOOM_SYMBOLIC_INTEGER_RELATION_LT, LOOM_SYMBOLIC_INTEGER_RELATION_LE,
      LOOM_SYMBOLIC_INTEGER_RELATION_GT, LOOM_SYMBOLIC_INTEGER_RELATION_GE,
  };
  const loom_condition_integer_operand_t left = {
      LOOM_CONDITION_INTEGER_OPERAND_VALUE, DefineIndexValue(), 0};
  const loom_condition_integer_operand_t right = {
      LOOM_CONDITION_INTEGER_OPERAND_VALUE, DefineIndexValue(), 0};
  condition_facts_.integer_relation_count = 2;
  for (auto queried : relations) {
    for (auto first : relations) {
      for (auto second : relations) {
        for (unsigned swaps = 0; swaps < 4; ++swaps) {
          SCOPED_TRACE(::testing::Message()
                       << "queried=" << queried << ", first=" << first
                       << ", second=" << second << ", swaps=" << swaps);
          const loom_condition_integer_relation_t query = {queried, left,
                                                           right};
          condition_facts_.integer_relations[0] = {
              first, (swaps & 1) ? right : left, (swaps & 1) ? left : right};
          condition_facts_.integer_relations[1] = {
              second, (swaps & 2) ? right : left, (swaps & 2) ? left : right};
          unsigned outcomes = 0;
          for (int64_t a : {-1, 0, 1}) {
            for (int64_t b : {-1, 0, 1}) {
              if (EvaluateRelation(first, (swaps & 1) ? b : a,
                                   (swaps & 1) ? a : b) &&
                  EvaluateRelation(second, (swaps & 2) ? b : a,
                                   (swaps & 2) ? a : b)) {
                outcomes |= EvaluateRelation(queried, a, b) ? 2u : 1u;
              }
            }
          }
          bool value = false;
          const bool proven = loom_condition_fact_set_proves_integer_relation(
              &condition_facts_, &fact_table_, &query, &value);
          EXPECT_EQ(proven, outcomes == 1u || outcomes == 2u);
          if (proven) {
            EXPECT_EQ(value ? 2u : 1u, outcomes);
          }
        }
      }
    }
  }
}

TEST_F(ConditionFactsTest, RelationMeetMatchesConjunctionTruthTable) {
  const loom_symbolic_integer_relation_t relations[] = {
      LOOM_SYMBOLIC_INTEGER_RELATION_EQ, LOOM_SYMBOLIC_INTEGER_RELATION_NE,
      LOOM_SYMBOLIC_INTEGER_RELATION_LT, LOOM_SYMBOLIC_INTEGER_RELATION_LE,
      LOOM_SYMBOLIC_INTEGER_RELATION_GT, LOOM_SYMBOLIC_INTEGER_RELATION_GE,
  };
  const loom_condition_integer_operand_t left_operand = {
      LOOM_CONDITION_INTEGER_OPERAND_VALUE, DefineIndexValue(), 0};
  const loom_condition_integer_operand_t right_operand = {
      LOOM_CONDITION_INTEGER_OPERAND_VALUE, DefineIndexValue(), 0};
  for (auto left : relations) {
    for (auto first : relations) {
      for (auto second : relations) {
        for (uint32_t swaps = 0; swaps < 4; ++swaps) {
          SCOPED_TRACE(::testing::Message()
                       << "left=" << left << ", first=" << first
                       << ", second=" << second << ", swaps=" << swaps);
          const loom_condition_integer_relation_t left_relation = {
              left, left_operand, right_operand};
          const loom_condition_integer_relation_t right_relations[] = {
              {first, (swaps & 1) ? right_operand : left_operand,
               (swaps & 1) ? left_operand : right_operand},
              {second, (swaps & 2) ? right_operand : left_operand,
               (swaps & 2) ? left_operand : right_operand},
          };
          loom_condition_integer_relation_t common = {};
          bool found = loom_condition_integer_relation_meet(
              &left_relation, right_relations, IREE_ARRAYSIZE(right_relations),
              &common);
          bool all_outcomes_allowed = true;
          for (int64_t a : {-1, 0, 1}) {
            for (int64_t b : {-1, 0, 1}) {
              const bool expected =
                  EvaluateRelation(left, a, b) ||
                  (EvaluateRelation(first, (swaps & 1) ? b : a,
                                    (swaps & 1) ? a : b) &&
                   EvaluateRelation(second, (swaps & 2) ? b : a,
                                    (swaps & 2) ? a : b));
              all_outcomes_allowed &= expected;
              if (found) {
                EXPECT_EQ(EvaluateRelation(common.relation, a, b), expected);
              }
            }
          }
          EXPECT_EQ(found, !all_outcomes_allowed);
          if (found) {
            EXPECT_TRUE(loom_condition_integer_operands_equal(common.left,
                                                              left_operand));
            EXPECT_TRUE(loom_condition_integer_operands_equal(common.right,
                                                              right_operand));
          }
        }
      }
    }
  }
}

TEST_F(ConditionFactsTest, RelationMeetIgnoresUnmatchedOperandPairs) {
  const loom_condition_integer_operand_t value = {
      LOOM_CONDITION_INTEGER_OPERAND_VALUE, DefineIndexValue(), 0};
  const loom_condition_integer_operand_t other_value = {
      LOOM_CONDITION_INTEGER_OPERAND_VALUE, DefineIndexValue(), 0};
  const loom_condition_integer_operand_t zero = {
      LOOM_CONDITION_INTEGER_OPERAND_CONSTANT, LOOM_VALUE_ID_INVALID, 0};
  const loom_condition_integer_operand_t one = {
      LOOM_CONDITION_INTEGER_OPERAND_CONSTANT, LOOM_VALUE_ID_INVALID, 1};
  const loom_condition_integer_relation_t left = {
      LOOM_SYMBOLIC_INTEGER_RELATION_EQ, value, zero};
  const loom_condition_integer_relation_t right[] = {
      {LOOM_SYMBOLIC_INTEGER_RELATION_NE, other_value, zero},
      {LOOM_SYMBOLIC_INTEGER_RELATION_NE, value, one},
      {LOOM_SYMBOLIC_INTEGER_RELATION_EQ, zero, value},
  };
  loom_condition_integer_relation_t common = {};
  EXPECT_FALSE(
      loom_condition_integer_relation_meet(&left, nullptr, 0, &common));
  EXPECT_FALSE(loom_condition_integer_relation_meet(&left, right, 2, &common));
  ASSERT_TRUE(loom_condition_integer_relation_meet(
      &left, right, IREE_ARRAYSIZE(right), &common));
  EXPECT_TRUE(loom_condition_integer_relations_equivalent(&left, &common));
}

TEST_F(ConditionFactsTest, RelationCapacityOverflowIsIncomplete) {
  loom_value_id_t induction = DefineIndexValue();
  loom_value_id_t upper_bound = DefineIndexValue();
  loom_op_t* compare =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, induction, upper_bound);
  loom_condition_fact_set_t empty_facts;
  loom_condition_fact_set_initialize(NULL, 0, &empty_facts);

  bool complete = true;
  IREE_ASSERT_OK(loom_condition_facts_query(&condition_query_, &fact_table_,
                                            loom_index_cmp_result(compare),
                                            true, &empty_facts, &complete));
  EXPECT_FALSE(complete);
  EXPECT_EQ(empty_facts.integer_relation_count, 0u);
}

}  // namespace
}  // namespace loom
