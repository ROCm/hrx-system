// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/condition_facts.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/facts.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class ConditionAffinePreimageTest : public ::testing::Test {
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
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    IREE_ASSERT_OK(
        loom_value_fact_table_initialize(&fact_table_, &analysis_arena_, 16));
    loom_condition_query_initialize(module_, /*value_domain=*/nullptr,
                                    &analysis_arena_, &condition_query_);
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

  loom_value_id_t DefineValue(loom_scalar_type_t scalar_type) {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_value(
        &builder_, loom_type_scalar(scalar_type), &value_id));
    return value_id;
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

  void Query(loom_value_id_t condition_value, bool assumed_truth = true) {
    bool complete = false;
    IREE_ASSERT_OK(loom_condition_facts_query(&condition_query_, &fact_table_,
                                              condition_value, assumed_truth,
                                              &condition_facts_, &complete));
    ASSERT_TRUE(complete);
  }

  loom_value_facts_t Apply(loom_value_id_t value_id) {
    auto facts = loom_value_fact_table_lookup(&fact_table_, value_id);
    loom_condition_fact_set_apply_to_value_facts(
        &condition_facts_, &fact_table_, value_id, &facts);
    return facts;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
  loom_value_fact_table_t fact_table_;
  loom_condition_query_t condition_query_;
  loom_condition_integer_relation_t relation_storage_[8];
  loom_condition_fact_set_t condition_facts_;
};

TEST_F(ConditionAffinePreimageTest, IndexAddCoversOperandsAndBranchEdges) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  for (bool dynamic_rhs : {false, true}) {
    const loom_value_id_t dynamic = DefineValue(LOOM_SCALAR_TYPE_INDEX);
    const loom_value_id_t constant = DefineValue(LOOM_SCALAR_TYPE_INDEX);
    const loom_value_id_t bound = DefineValue(LOOM_SCALAR_TYPE_INDEX);
    DefineFacts(dynamic, loom_value_facts_make(0, 24, 1));
    DefineFacts(constant, loom_value_facts_exact_i64(25));
    DefineFacts(bound, loom_value_facts_exact_i64(40));
    loom_op_t* add = nullptr;
    IREE_ASSERT_OK(loom_index_add_build(
        &builder_, dynamic_rhs ? constant : dynamic,
        dynamic_rhs ? dynamic : constant, type, LOOM_LOCATION_UNKNOWN, &add));
    const loom_value_id_t sum = loom_index_add_result(add);
    DefineFacts(sum, loom_value_facts_make(25, 49, 1));
    const loom_op_t* compare =
        BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_ULT, sum, bound);

    Query(loom_index_cmp_result(compare));
    auto facts = Apply(dynamic);
    EXPECT_EQ(facts.range_lo, 0);
    EXPECT_EQ(facts.range_hi, 14);

    Query(loom_index_cmp_result(compare), /*assumed_truth=*/false);
    facts = Apply(dynamic);
    EXPECT_EQ(facts.range_lo, 15);
    EXPECT_EQ(facts.range_hi, 24);
  }
}

TEST_F(ConditionAffinePreimageTest, IndexSubtractCoversBothOperands) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  {
    const loom_value_id_t dynamic = DefineValue(LOOM_SCALAR_TYPE_INDEX);
    const loom_value_id_t constant = DefineValue(LOOM_SCALAR_TYPE_INDEX);
    const loom_value_id_t bound = DefineValue(LOOM_SCALAR_TYPE_INDEX);
    DefineFacts(dynamic, loom_value_facts_make(25, 49, 1));
    DefineFacts(constant, loom_value_facts_exact_i64(25));
    DefineFacts(bound, loom_value_facts_exact_i64(15));
    loom_op_t* subtract = nullptr;
    IREE_ASSERT_OK(loom_index_sub_build(&builder_, dynamic, constant, type,
                                        LOOM_LOCATION_UNKNOWN, &subtract));
    const loom_value_id_t difference = loom_index_sub_result(subtract);
    DefineFacts(difference, loom_value_facts_make(0, 24, 1));
    const loom_op_t* compare =
        BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_ULT, difference, bound);

    Query(loom_index_cmp_result(compare));
    const auto facts = Apply(dynamic);
    EXPECT_EQ(facts.range_lo, 25);
    EXPECT_EQ(facts.range_hi, 39);
  }
  {
    const loom_value_id_t constant = DefineValue(LOOM_SCALAR_TYPE_INDEX);
    const loom_value_id_t dynamic = DefineValue(LOOM_SCALAR_TYPE_INDEX);
    const loom_value_id_t bound = DefineValue(LOOM_SCALAR_TYPE_INDEX);
    DefineFacts(constant, loom_value_facts_exact_i64(39));
    DefineFacts(dynamic, loom_value_facts_make(0, 24, 1));
    DefineFacts(bound, loom_value_facts_exact_i64(30));
    loom_op_t* subtract = nullptr;
    IREE_ASSERT_OK(loom_index_sub_build(&builder_, constant, dynamic, type,
                                        LOOM_LOCATION_UNKNOWN, &subtract));
    const loom_value_id_t difference = loom_index_sub_result(subtract);
    DefineFacts(difference, loom_value_facts_make(15, 39, 1));
    const loom_op_t* compare =
        BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_ULT, difference, bound);

    Query(loom_index_cmp_result(compare));
    const auto facts = Apply(dynamic);
    EXPECT_EQ(facts.range_lo, 10);
    EXPECT_EQ(facts.range_hi, 24);
  }
}

TEST_F(ConditionAffinePreimageTest,
       FixedWidthScalarFamilyCoversAddAndSubtract) {
  for (auto scalar_type : {LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_I16,
                           LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I64}) {
    const loom_type_t type = loom_type_scalar(scalar_type);
    for (bool dynamic_rhs : {false, true}) {
      const loom_value_id_t dynamic = DefineValue(scalar_type);
      const loom_value_id_t constant = DefineValue(scalar_type);
      const loom_value_id_t bound = DefineValue(scalar_type);
      DefineFacts(dynamic, loom_value_facts_make(0, 24, 1));
      DefineFacts(constant, loom_value_facts_exact_i64(25));
      DefineFacts(bound, loom_value_facts_exact_i64(40));
      loom_op_t* add = nullptr;
      IREE_ASSERT_OK(loom_scalar_addi_build(
          &builder_, /*instance_flags=*/0, dynamic_rhs ? constant : dynamic,
          dynamic_rhs ? dynamic : constant, type, LOOM_LOCATION_UNKNOWN, &add));
      const loom_value_id_t sum = loom_scalar_addi_result(add);
      DefineFacts(sum, loom_value_facts_make(25, 49, 1));
      const loom_op_t* compare =
          BuildScalarCompare(LOOM_SCALAR_CMPI_PREDICATE_ULT, sum, bound);

      Query(loom_scalar_cmpi_result(compare));
      const auto facts = Apply(dynamic);
      EXPECT_EQ(facts.range_lo, 0);
      EXPECT_EQ(facts.range_hi, 14);
    }

    const loom_value_id_t dynamic_lhs = DefineValue(scalar_type);
    const loom_value_id_t constant_rhs = DefineValue(scalar_type);
    const loom_value_id_t lhs_bound = DefineValue(scalar_type);
    DefineFacts(dynamic_lhs, loom_value_facts_make(25, 49, 1));
    DefineFacts(constant_rhs, loom_value_facts_exact_i64(25));
    DefineFacts(lhs_bound, loom_value_facts_exact_i64(15));
    loom_op_t* subtract = nullptr;
    IREE_ASSERT_OK(loom_scalar_subi_build(&builder_, /*instance_flags=*/0,
                                          dynamic_lhs, constant_rhs, type,
                                          LOOM_LOCATION_UNKNOWN, &subtract));
    loom_value_id_t difference = loom_scalar_subi_result(subtract);
    DefineFacts(difference, loom_value_facts_make(0, 24, 1));
    loom_op_t* compare = BuildScalarCompare(LOOM_SCALAR_CMPI_PREDICATE_ULT,
                                            difference, lhs_bound);
    Query(loom_scalar_cmpi_result(compare));
    auto facts = Apply(dynamic_lhs);
    EXPECT_EQ(facts.range_lo, 25);
    EXPECT_EQ(facts.range_hi, 39);

    const loom_value_id_t constant_lhs = DefineValue(scalar_type);
    const loom_value_id_t dynamic_rhs = DefineValue(scalar_type);
    const loom_value_id_t rhs_bound = DefineValue(scalar_type);
    DefineFacts(constant_lhs, loom_value_facts_exact_i64(39));
    DefineFacts(dynamic_rhs, loom_value_facts_make(0, 24, 1));
    DefineFacts(rhs_bound, loom_value_facts_exact_i64(30));
    subtract = nullptr;
    IREE_ASSERT_OK(loom_scalar_subi_build(&builder_, /*instance_flags=*/0,
                                          constant_lhs, dynamic_rhs, type,
                                          LOOM_LOCATION_UNKNOWN, &subtract));
    difference = loom_scalar_subi_result(subtract);
    DefineFacts(difference, loom_value_facts_make(15, 39, 1));
    compare = BuildScalarCompare(LOOM_SCALAR_CMPI_PREDICATE_ULT, difference,
                                 rhs_bound);
    Query(loom_scalar_cmpi_result(compare));
    facts = Apply(dynamic_rhs);
    EXPECT_EQ(facts.range_lo, 10);
    EXPECT_EQ(facts.range_hi, 24);
  }
}

TEST_F(ConditionAffinePreimageTest, FactIdentityCarriesDerivedRange) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_value_id_t input = DefineValue(LOOM_SCALAR_TYPE_I32);
  DefineFacts(input, loom_value_facts_make(0, 24, 1));
  loom_op_t* alias = nullptr;
  IREE_ASSERT_OK(loom_scalar_assume_build(&builder_, &input, 1, nullptr, 0,
                                          &type, 1, LOOM_LOCATION_UNKNOWN,
                                          &alias));
  IREE_ASSERT_OK(
      loom_value_fact_table_compute_op(&fact_table_, module_, alias));
  const loom_value_id_t aliased = loom_op_const_results(alias)[0];
  const loom_value_id_t constant = DefineValue(LOOM_SCALAR_TYPE_I32);
  const loom_value_id_t bound = DefineValue(LOOM_SCALAR_TYPE_I32);
  DefineFacts(constant, loom_value_facts_exact_i64(25));
  DefineFacts(bound, loom_value_facts_exact_i64(40));
  loom_op_t* add = nullptr;
  IREE_ASSERT_OK(loom_scalar_addi_build(&builder_, /*instance_flags=*/0,
                                        aliased, constant, type,
                                        LOOM_LOCATION_UNKNOWN, &add));
  const loom_value_id_t sum = loom_scalar_addi_result(add);
  DefineFacts(sum, loom_value_facts_make(25, 49, 1));
  const loom_op_t* compare =
      BuildScalarCompare(LOOM_SCALAR_CMPI_PREDICATE_ULT, sum, bound);

  Query(loom_scalar_cmpi_result(compare));
  for (loom_value_id_t value : {input, aliased}) {
    const auto facts = Apply(value);
    EXPECT_EQ(facts.range_lo, 0);
    EXPECT_EQ(facts.range_hi, 14);
  }
  bool found_preimage = false;
  for (iree_host_size_t i = 0; i < condition_facts_.integer_relation_count;
       ++i) {
    const auto& relation = condition_facts_.integer_relations[i];
    if (relation.relation == LOOM_SYMBOLIC_INTEGER_RELATION_LE &&
        relation.right.kind == LOOM_CONDITION_INTEGER_OPERAND_CONSTANT &&
        relation.right.constant == 14) {
      ASSERT_EQ(relation.left.kind, LOOM_CONDITION_INTEGER_OPERAND_VALUE);
      EXPECT_EQ(loom_value_fact_table_query_identity(&fact_table_,
                                                     relation.left.value_id),
                input);
      found_preimage = true;
    }
  }
  EXPECT_TRUE(found_preimage);
}

TEST_F(ConditionAffinePreimageTest, ComparisonRelationsCrossFactIdentity) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_value_id_t input = DefineValue(LOOM_SCALAR_TYPE_I32);
  DefineFacts(input, loom_value_facts_make(0, 24, 1));
  loom_op_t* alias = nullptr;
  IREE_ASSERT_OK(loom_scalar_assume_build(&builder_, &input, 1, nullptr, 0,
                                          &type, 1, LOOM_LOCATION_UNKNOWN,
                                          &alias));
  IREE_ASSERT_OK(
      loom_value_fact_table_compute_op(&fact_table_, module_, alias));
  const loom_value_id_t bound = DefineValue(LOOM_SCALAR_TYPE_I32);
  DefineFacts(bound, loom_value_facts_exact_i64(15));
  const loom_op_t* compare = BuildScalarCompare(
      LOOM_SCALAR_CMPI_PREDICATE_ULT, loom_op_const_results(alias)[0], bound);

  Query(loom_scalar_cmpi_result(compare));
  for (loom_value_id_t value : {input, loom_op_const_results(alias)[0]}) {
    const auto facts = Apply(value);
    EXPECT_EQ(facts.range_lo, 0);
    EXPECT_EQ(facts.range_hi, 14);
  }
}

TEST_F(ConditionAffinePreimageTest, ScalarWrappingRefusesFalseIntervals) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I8);
  for (bool subtract : {false, true}) {
    const loom_value_id_t dynamic = DefineValue(LOOM_SCALAR_TYPE_I8);
    const loom_value_id_t one = DefineValue(LOOM_SCALAR_TYPE_I8);
    const loom_value_id_t zero = DefineValue(LOOM_SCALAR_TYPE_I8);
    DefineFacts(dynamic, loom_value_facts_make(INT8_MIN, INT8_MAX, 1));
    DefineFacts(one, loom_value_facts_exact_i64(1));
    DefineFacts(zero, loom_value_facts_exact_i64(0));
    loom_op_t* arithmetic = nullptr;
    if (subtract) {
      IREE_ASSERT_OK(
          loom_scalar_subi_build(&builder_, /*instance_flags=*/0, dynamic, one,
                                 type, LOOM_LOCATION_UNKNOWN, &arithmetic));
    } else {
      IREE_ASSERT_OK(
          loom_scalar_addi_build(&builder_, /*instance_flags=*/0, dynamic, one,
                                 type, LOOM_LOCATION_UNKNOWN, &arithmetic));
    }
    IREE_ASSERT_OK(
        loom_value_fact_table_compute_op(&fact_table_, module_, arithmetic));
    const loom_value_id_t result = loom_op_const_results(arithmetic)[0];
    const loom_op_t* compare =
        BuildScalarCompare(subtract ? LOOM_SCALAR_CMPI_PREDICATE_SGT
                                    : LOOM_SCALAR_CMPI_PREDICATE_SLT,
                           result, zero);

    Query(loom_scalar_cmpi_result(compare));
    const auto facts = Apply(dynamic);
    EXPECT_EQ(facts.range_lo, INT8_MIN);
    EXPECT_EQ(facts.range_hi, INT8_MAX);
  }
}

TEST_F(ConditionAffinePreimageTest, SignedNoWrapContractEnablesPreimage) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I8);
  {
    const loom_value_id_t dynamic = DefineValue(LOOM_SCALAR_TYPE_I8);
    const loom_value_id_t one = DefineValue(LOOM_SCALAR_TYPE_I8);
    const loom_value_id_t zero = DefineValue(LOOM_SCALAR_TYPE_I8);
    DefineFacts(dynamic, loom_value_facts_make(INT8_MIN, INT8_MAX, 1));
    DefineFacts(one, loom_value_facts_exact_i64(1));
    DefineFacts(zero, loom_value_facts_exact_i64(0));
    loom_op_t* add = nullptr;
    IREE_ASSERT_OK(loom_scalar_addi_build(
        &builder_, LOOM_INTEGER_OVERFLOW_FLAG_NSW, dynamic, one, type,
        LOOM_LOCATION_UNKNOWN, &add));
    IREE_ASSERT_OK(
        loom_value_fact_table_compute_op(&fact_table_, module_, add));
    const loom_op_t* compare = BuildScalarCompare(
        LOOM_SCALAR_CMPI_PREDICATE_SLT, loom_scalar_addi_result(add), zero);

    Query(loom_scalar_cmpi_result(compare));
    const auto facts = Apply(dynamic);
    EXPECT_EQ(facts.range_lo, INT8_MIN);
    EXPECT_EQ(facts.range_hi, -2);
  }
  {
    const loom_value_id_t dynamic = DefineValue(LOOM_SCALAR_TYPE_I8);
    const loom_value_id_t one = DefineValue(LOOM_SCALAR_TYPE_I8);
    const loom_value_id_t zero = DefineValue(LOOM_SCALAR_TYPE_I8);
    DefineFacts(dynamic, loom_value_facts_make(INT8_MIN, INT8_MAX, 1));
    DefineFacts(one, loom_value_facts_exact_i64(1));
    DefineFacts(zero, loom_value_facts_exact_i64(0));
    loom_op_t* subtract = nullptr;
    IREE_ASSERT_OK(loom_scalar_subi_build(
        &builder_, LOOM_INTEGER_OVERFLOW_FLAG_NSW, dynamic, one, type,
        LOOM_LOCATION_UNKNOWN, &subtract));
    IREE_ASSERT_OK(
        loom_value_fact_table_compute_op(&fact_table_, module_, subtract));
    const loom_op_t* compare =
        BuildScalarCompare(LOOM_SCALAR_CMPI_PREDICATE_SGT,
                           loom_scalar_subi_result(subtract), zero);

    Query(loom_scalar_cmpi_result(compare));
    const auto facts = Apply(dynamic);
    EXPECT_EQ(facts.range_lo, 2);
    EXPECT_EQ(facts.range_hi, INT8_MAX);
  }
}

TEST_F(ConditionAffinePreimageTest,
       UnsignedNoWrapDoesNotPromiseSignedAffineResult) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I8);
  const loom_value_id_t dynamic = DefineValue(LOOM_SCALAR_TYPE_I8);
  const loom_value_id_t one = DefineValue(LOOM_SCALAR_TYPE_I8);
  const loom_value_id_t zero = DefineValue(LOOM_SCALAR_TYPE_I8);
  DefineFacts(dynamic, loom_value_facts_make(0, INT8_MAX, 1));
  DefineFacts(one, loom_value_facts_exact_i64(1));
  DefineFacts(zero, loom_value_facts_exact_i64(0));
  loom_op_t* add = nullptr;
  IREE_ASSERT_OK(
      loom_scalar_addi_build(&builder_, LOOM_INTEGER_OVERFLOW_FLAG_NUW, dynamic,
                             one, type, LOOM_LOCATION_UNKNOWN, &add));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&fact_table_, module_, add));
  const loom_op_t* compare = BuildScalarCompare(
      LOOM_SCALAR_CMPI_PREDICATE_SLT, loom_scalar_addi_result(add), zero);

  Query(loom_scalar_cmpi_result(compare));
  const auto facts = Apply(dynamic);
  EXPECT_EQ(facts.range_lo, 0);
  EXPECT_EQ(facts.range_hi, INT8_MAX);
}

TEST_F(ConditionAffinePreimageTest, BooleanCarrierRefusesSignedInversion) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  const loom_value_id_t dynamic = DefineValue(LOOM_SCALAR_TYPE_I1);
  const loom_value_id_t one = DefineValue(LOOM_SCALAR_TYPE_I1);
  const loom_value_id_t zero = DefineValue(LOOM_SCALAR_TYPE_I1);
  DefineFacts(dynamic, loom_value_facts_make(0, 1, 1));
  DefineFacts(one, loom_value_facts_exact_i64(1));
  DefineFacts(zero, loom_value_facts_exact_i64(0));
  loom_op_t* add = nullptr;
  IREE_ASSERT_OK(loom_scalar_addi_build(&builder_, /*instance_flags=*/0,
                                        dynamic, one, type,
                                        LOOM_LOCATION_UNKNOWN, &add));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&fact_table_, module_, add));
  const loom_op_t* compare = BuildScalarCompare(
      LOOM_SCALAR_CMPI_PREDICATE_EQ, loom_scalar_addi_result(add), zero);

  Query(loom_scalar_cmpi_result(compare));
  const auto facts = Apply(dynamic);
  EXPECT_EQ(facts.range_lo, 0);
  EXPECT_EQ(facts.range_hi, 1);
}

TEST_F(ConditionAffinePreimageTest, OffsetPreimagesRespectTargetCarrier) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  for (uint32_t carrier_width : {32u, 64u}) {
    loom_target_facts_t target_facts = {};
    target_facts.storage.snapshot.offset_bitwidth = carrier_width;
    fact_table_.context.target_facts = &target_facts;
    for (bool dynamic_rhs : {false, true}) {
      const loom_value_id_t dynamic = DefineValue(LOOM_SCALAR_TYPE_OFFSET);
      const loom_value_id_t constant = DefineValue(LOOM_SCALAR_TYPE_OFFSET);
      const loom_value_id_t bound = DefineValue(LOOM_SCALAR_TYPE_OFFSET);
      DefineFacts(dynamic, loom_value_facts_make(0, 24, 1));
      DefineFacts(constant, loom_value_facts_exact_i64(25));
      DefineFacts(bound, loom_value_facts_exact_i64(40));
      loom_op_t* add = nullptr;
      IREE_ASSERT_OK(loom_index_add_build(
          &builder_, dynamic_rhs ? constant : dynamic,
          dynamic_rhs ? dynamic : constant, type, LOOM_LOCATION_UNKNOWN, &add));
      IREE_ASSERT_OK(
          loom_value_fact_table_compute_op(&fact_table_, module_, add));
      const loom_op_t* compare = BuildIndexCompare(
          LOOM_INDEX_CMP_PREDICATE_ULT, loom_index_add_result(add), bound);

      Query(loom_index_cmp_result(compare));
      const auto facts = Apply(dynamic);
      EXPECT_EQ(facts.range_lo, 0);
      EXPECT_EQ(facts.range_hi, 14);
    }

    const loom_value_id_t dynamic_lhs = DefineValue(LOOM_SCALAR_TYPE_OFFSET);
    const loom_value_id_t constant_rhs = DefineValue(LOOM_SCALAR_TYPE_OFFSET);
    const loom_value_id_t difference_bound =
        DefineValue(LOOM_SCALAR_TYPE_OFFSET);
    DefineFacts(dynamic_lhs, loom_value_facts_make(25, 49, 1));
    DefineFacts(constant_rhs, loom_value_facts_exact_i64(25));
    DefineFacts(difference_bound, loom_value_facts_exact_i64(15));
    loom_op_t* subtract = nullptr;
    IREE_ASSERT_OK(loom_index_sub_build(&builder_, dynamic_lhs, constant_rhs,
                                        type, LOOM_LOCATION_UNKNOWN,
                                        &subtract));
    IREE_ASSERT_OK(
        loom_value_fact_table_compute_op(&fact_table_, module_, subtract));
    const loom_op_t* compare =
        BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_ULT,
                          loom_index_sub_result(subtract), difference_bound);
    Query(loom_index_cmp_result(compare));
    const auto lhs_facts = Apply(dynamic_lhs);
    EXPECT_EQ(lhs_facts.range_lo, 25);
    EXPECT_EQ(lhs_facts.range_hi, 39);

    const loom_value_id_t constant_lhs = DefineValue(LOOM_SCALAR_TYPE_OFFSET);
    const loom_value_id_t dynamic_rhs = DefineValue(LOOM_SCALAR_TYPE_OFFSET);
    const loom_value_id_t rhs_bound = DefineValue(LOOM_SCALAR_TYPE_OFFSET);
    DefineFacts(constant_lhs, loom_value_facts_exact_i64(39));
    DefineFacts(dynamic_rhs, loom_value_facts_make(0, 24, 1));
    DefineFacts(rhs_bound, loom_value_facts_exact_i64(30));
    subtract = nullptr;
    IREE_ASSERT_OK(loom_index_sub_build(&builder_, constant_lhs, dynamic_rhs,
                                        type, LOOM_LOCATION_UNKNOWN,
                                        &subtract));
    IREE_ASSERT_OK(
        loom_value_fact_table_compute_op(&fact_table_, module_, subtract));
    compare = BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_ULT,
                                loom_index_sub_result(subtract), rhs_bound);
    Query(loom_index_cmp_result(compare));
    const auto rhs_facts = Apply(dynamic_rhs);
    EXPECT_EQ(rhs_facts.range_lo, 10);
    EXPECT_EQ(rhs_facts.range_hi, 24);
  }

  loom_target_facts_t target_facts = {};
  target_facts.storage.snapshot.offset_bitwidth = 32;
  fact_table_.context.target_facts = &target_facts;
  const loom_value_id_t wrapping = DefineValue(LOOM_SCALAR_TYPE_OFFSET);
  const loom_value_id_t one = DefineValue(LOOM_SCALAR_TYPE_OFFSET);
  const loom_value_id_t four = DefineValue(LOOM_SCALAR_TYPE_OFFSET);
  DefineFacts(wrapping, loom_value_facts_make(UINT32_MAX - 4, UINT32_MAX, 1));
  DefineFacts(one, loom_value_facts_exact_i64(1));
  DefineFacts(four, loom_value_facts_exact_i64(4));
  loom_op_t* wrapping_add = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, wrapping, one, type,
                                      LOOM_LOCATION_UNKNOWN, &wrapping_add));
  IREE_ASSERT_OK(
      loom_value_fact_table_compute_op(&fact_table_, module_, wrapping_add));
  const loom_op_t* compare = BuildIndexCompare(
      LOOM_INDEX_CMP_PREDICATE_ULT, loom_index_add_result(wrapping_add), four);

  Query(loom_index_cmp_result(compare));
  const auto facts = Apply(wrapping);
  EXPECT_EQ(facts.range_lo, (int64_t)UINT32_MAX - 4);
  EXPECT_EQ(facts.range_hi, (int64_t)UINT32_MAX);

  const loom_value_id_t wrapping_lhs = DefineValue(LOOM_SCALAR_TYPE_OFFSET);
  const loom_value_id_t large = DefineValue(LOOM_SCALAR_TYPE_OFFSET);
  DefineFacts(wrapping_lhs, loom_value_facts_make(0, 4, 1));
  DefineFacts(large, loom_value_facts_exact_i64(UINT32_MAX - 4));
  loom_op_t* wrapping_subtract = nullptr;
  IREE_ASSERT_OK(loom_index_sub_build(&builder_, wrapping_lhs, one, type,
                                      LOOM_LOCATION_UNKNOWN,
                                      &wrapping_subtract));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&fact_table_, module_,
                                                  wrapping_subtract));
  compare = BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_UGT,
                              loom_index_sub_result(wrapping_subtract), large);
  Query(loom_index_cmp_result(compare));
  const auto lhs_facts = Apply(wrapping_lhs);
  EXPECT_EQ(lhs_facts.range_lo, 0);
  EXPECT_EQ(lhs_facts.range_hi, 4);
  fact_table_.context.target_facts = nullptr;
}

}  // namespace
}  // namespace loom
