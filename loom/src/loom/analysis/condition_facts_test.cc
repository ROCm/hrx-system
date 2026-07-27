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

  loom_op_t* BuildIndexCompare(uint8_t predicate, loom_value_id_t left,
                               loom_value_id_t right) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_cmp_build(&builder_, predicate, left, right,
                                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                       loom_type_scalar(LOOM_SCALAR_TYPE_I1),
                                       LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_op_t* BuildScalarI32Compare(uint8_t predicate, loom_value_id_t left,
                                   loom_value_id_t right) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_scalar_cmpi_build(&builder_, predicate, left, right,
                                         loom_type_scalar(LOOM_SCALAR_TYPE_I32),
                                         loom_type_scalar(LOOM_SCALAR_TYPE_I1),
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

  bool Query(loom_value_id_t condition_value, bool assumed_truth = true) {
    return loom_condition_facts_query(module_, &fact_table_, condition_value,
                                      assumed_truth, &condition_facts_);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
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

TEST_F(ConditionFactsTest, EdgeFactsProveWiderScalarCompareTrue) {
  loom_value_id_t lane = DefineI32Value();
  loom_value_id_t outer_bound = DefineI32Value();
  loom_value_id_t inner_bound = DefineI32Value();
  DefineFacts(outer_bound, loom_value_facts_exact_i64(8));
  DefineFacts(inner_bound, loom_value_facts_exact_i64(16));
  loom_op_t* outer =
      BuildScalarI32Compare(LOOM_SCALAR_CMPI_PREDICATE_SLT, lane, outer_bound);
  ASSERT_TRUE(Query(loom_scalar_cmpi_result(outer)));

  loom_op_t* inner =
      BuildScalarI32Compare(LOOM_SCALAR_CMPI_PREDICATE_SLT, lane, inner_bound);
  bool condition = false;
  EXPECT_TRUE(loom_condition_fact_set_proves_condition(
      module_, &fact_table_, &condition_facts_, loom_scalar_cmpi_result(inner),
      &condition));
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
  EXPECT_TRUE(loom_condition_fact_set_proves_condition(
      module_, &fact_table_, &condition_facts_, loom_index_cmp_result(inner),
      &condition));
  EXPECT_FALSE(condition);
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

TEST_F(ConditionFactsTest, ScalarCmpiProducesIntegerRelation) {
  loom_value_id_t left = DefineI32Value();
  loom_value_id_t right = DefineI32Value();
  loom_op_t* compare =
      BuildScalarI32Compare(LOOM_SCALAR_CMPI_PREDICATE_EQ, left, right);

  Query(loom_scalar_cmpi_result(compare), false);

  ASSERT_EQ(condition_facts_.integer_relation_count, 1u);
  const loom_condition_integer_relation_t& relation =
      condition_facts_.integer_relations[0];
  EXPECT_EQ(relation.relation, LOOM_SYMBOLIC_INTEGER_RELATION_NE);
  EXPECT_EQ(relation.left.value_id, left);
  EXPECT_EQ(relation.right.value_id, right);
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

TEST_F(ConditionFactsTest, UnknownConditionProducesNoFacts) {
  loom_value_id_t condition =
      DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1));

  Query(condition);

  EXPECT_EQ(condition_facts_.integer_relation_count, 0u);
}

TEST_F(ConditionFactsTest, RelationCapacityOverflowIsIncomplete) {
  loom_value_id_t induction = DefineIndexValue();
  loom_value_id_t upper_bound = DefineIndexValue();
  loom_op_t* compare =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, induction, upper_bound);
  loom_condition_fact_set_t empty_facts;
  loom_condition_fact_set_initialize(NULL, 0, &empty_facts);

  EXPECT_FALSE(loom_condition_facts_query(module_, &fact_table_,
                                          loom_index_cmp_result(compare), true,
                                          &empty_facts));
  EXPECT_EQ(empty_facts.integer_relation_count, 0u);
}

}  // namespace
}  // namespace loom
