// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/value_relation.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"

namespace loom {
namespace {

class ValueRelationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(
        &context_, IREE_SV("value_relation_test"), &block_pool_, nullptr,
        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  loom_value_id_t BuildConstant(loom_type_t type, int64_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_scalar_constant_build(&builder_, loom_attr_i64(value),
                                             type, LOOM_LOCATION_UNKNOWN, &op));
    return loom_scalar_constant_result(op);
  }

  loom_value_id_t BuildIndexConstant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN, &op));
    return loom_index_constant_result(op);
  }

  std::vector<loom_value_relation_t> Collect(
      const loom_op_t* op,
      loom_value_relation_mask_t mask = LOOM_VALUE_RELATION_MASK_ALL) {
    loom_value_relation_iterator_t iterator;
    loom_value_relation_iterator_initialize(module_, op, mask, &iterator);
    std::vector<loom_value_relation_t> relations;
    loom_value_relation_t relation;
    while (loom_value_relation_iterator_next(&iterator, &relation)) {
      relations.push_back(relation);
    }
    return relations;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(ValueRelationTest, CountedLoopSeparatesEntryBypassAndYieldEdges) {
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_value_id_t lower_bound = BuildIndexConstant(0);
  const loom_value_id_t upper_bound = BuildIndexConstant(4);
  const loom_value_id_t step = BuildIndexConstant(1);
  const loom_value_id_t initial = BuildConstant(i32_type, 10);
  const loom_value_id_t yielded = BuildConstant(i32_type, 11);

  loom_op_t* loop = nullptr;
  IREE_ASSERT_OK(loom_scf_for_build(
      &builder_, /*build_flags=*/0, lower_bound, upper_bound, step, &initial, 1,
      /*tied_results=*/nullptr, /*tied_result_count=*/0, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, LOOM_LOCATION_UNKNOWN,
      &loop));
  const loom_value_id_t body_argument =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  const loom_value_id_t result = loom_scf_for_results(loop).values[0];

  const loom_builder_ip_t saved =
      loom_builder_enter_region(&builder_, loop, loom_scf_for_body(loop));
  loom_op_t* yield = nullptr;
  IREE_ASSERT_OK(loom_scf_yield_build(&builder_, &yielded, 1,
                                      LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(&builder_, saved);

  const std::vector<loom_value_relation_t> loop_relations = Collect(loop);
  ASSERT_EQ(loop_relations.size(), 2u);
  EXPECT_EQ(loop_relations[0].kind, LOOM_VALUE_RELATION_LOOP_CARRIED);
  EXPECT_EQ(loop_relations[0].source_value_id, initial);
  EXPECT_EQ(loop_relations[0].destination_value_id, body_argument);
  EXPECT_EQ(loop_relations[1].kind, LOOM_VALUE_RELATION_LOOP_BYPASS);
  EXPECT_EQ(loop_relations[1].source_value_id, initial);
  EXPECT_EQ(loop_relations[1].destination_value_id, result);

  const std::vector<loom_value_relation_t> yield_relations = Collect(yield);
  ASSERT_EQ(yield_relations.size(), 2u);
  EXPECT_EQ(yield_relations[0].kind, LOOM_VALUE_RELATION_LOOP_CARRIED);
  EXPECT_EQ(yield_relations[0].source_value_id, yielded);
  EXPECT_EQ(yield_relations[0].destination_value_id, body_argument);
  EXPECT_EQ(yield_relations[1].kind, LOOM_VALUE_RELATION_LOOP_CARRIED);
  EXPECT_EQ(yield_relations[1].source_value_id, yielded);
  EXPECT_EQ(yield_relations[1].destination_value_id, result);
}

TEST_F(ValueRelationTest, ConditionLoopPreservesDistinctStateEdges) {
  const loom_type_t i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_value_id_t condition = BuildConstant(i1_type, 1);
  const loom_value_id_t initial = BuildConstant(i32_type, 20);
  const loom_value_id_t forwarded = BuildConstant(i32_type, 21);
  const loom_value_id_t yielded = BuildConstant(i32_type, 22);

  loom_op_t* loop = nullptr;
  IREE_ASSERT_OK(loom_scf_while_build(
      &builder_, &initial, 1, /*tied_results=*/nullptr,
      /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN, &loop));
  const loom_value_id_t before_argument =
      loom_region_entry_arg_id(loom_scf_while_before(loop), 0);
  const loom_value_id_t body_argument =
      loom_region_entry_arg_id(loom_scf_while_after(loop), 0);
  const loom_value_id_t result = loom_scf_while_results(loop).values[0];

  loom_builder_ip_t saved =
      loom_builder_enter_region(&builder_, loop, loom_scf_while_before(loop));
  loom_op_t* condition_op = nullptr;
  IREE_ASSERT_OK(loom_scf_condition_build(&builder_, condition, &forwarded, 1,
                                          LOOM_LOCATION_UNKNOWN,
                                          &condition_op));
  loom_builder_restore(&builder_, saved);
  saved =
      loom_builder_enter_region(&builder_, loop, loom_scf_while_after(loop));
  loom_op_t* yield = nullptr;
  IREE_ASSERT_OK(loom_scf_yield_build(&builder_, &yielded, 1,
                                      LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(&builder_, saved);

  const std::vector<loom_value_relation_t> loop_relations = Collect(loop);
  ASSERT_EQ(loop_relations.size(), 1u);
  EXPECT_EQ(loop_relations[0].kind, LOOM_VALUE_RELATION_LOOP_CARRIED);
  EXPECT_EQ(loop_relations[0].source_value_id, initial);
  EXPECT_EQ(loop_relations[0].destination_value_id, before_argument);

  const std::vector<loom_value_relation_t> condition_relations =
      Collect(condition_op);
  ASSERT_EQ(condition_relations.size(), 2u);
  EXPECT_EQ(condition_relations[0].source_value_id, forwarded);
  EXPECT_EQ(condition_relations[0].destination_value_id, body_argument);
  EXPECT_EQ(condition_relations[1].source_value_id, forwarded);
  EXPECT_EQ(condition_relations[1].destination_value_id, result);

  const std::vector<loom_value_relation_t> yield_relations = Collect(yield);
  ASSERT_EQ(yield_relations.size(), 1u);
  EXPECT_EQ(yield_relations[0].source_value_id, yielded);
  EXPECT_EQ(yield_relations[0].destination_value_id, before_argument);
}

TEST_F(ValueRelationTest, LookupPayloadsMapByResultColumn) {
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_value_id_t selector = BuildIndexConstant(0);
  loom_value_id_t payloads[6];
  for (int64_t i = 0; i < 6; ++i) {
    payloads[i] = BuildConstant(i32_type, i);
  }
  const int64_t case_keys[] = {0, 1};
  const loom_type_t result_types[] = {i32_type, i32_type};
  loom_op_t* lookup = nullptr;
  IREE_ASSERT_OK(loom_scf_lookup_build(
      &builder_, selector, case_keys, IREE_ARRAYSIZE(case_keys), payloads,
      IREE_ARRAYSIZE(payloads), result_types, IREE_ARRAYSIZE(result_types),
      /*tied_results=*/nullptr, /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN,
      &lookup));

  const std::vector<loom_value_relation_t> relations = Collect(
      lookup, LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_SELECT_PAYLOAD));
  ASSERT_EQ(relations.size(), IREE_ARRAYSIZE(payloads));
  const loom_value_slice_t results = loom_scf_lookup_results(lookup);
  for (iree_host_size_t i = 0; i < relations.size(); ++i) {
    EXPECT_EQ(relations[i].kind, LOOM_VALUE_RELATION_SELECT_PAYLOAD);
    EXPECT_EQ(relations[i].source_value_id, payloads[i]);
    EXPECT_EQ(relations[i].destination_value_id,
              results.values[i % results.count]);
    EXPECT_EQ(relations[i].source_operand_index, (uint16_t)(i + 1));
  }
}

}  // namespace
}  // namespace loom
