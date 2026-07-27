// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/cfg_condition_facts.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

class CfgConditionFactsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);

    iree_host_size_t cfg_vtable_count = 0;
    const loom_op_vtable_t* const* cfg_vtables =
        loom_cfg_dialect_vtables(&cfg_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_CFG, cfg_vtables, (uint16_t)cfg_vtable_count));
    iree_host_size_t index_vtable_count = 0;
    const loom_op_vtable_t* const* index_vtables =
        loom_index_dialect_vtables(&index_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_INDEX,
                                                 index_vtables,
                                                 (uint16_t)index_vtable_count));
    iree_host_size_t test_vtable_count = 0;
    const loom_op_vtable_t* const* test_vtables =
        loom_test_dialect_vtables(&test_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                 test_vtables,
                                                 (uint16_t)test_vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&module_builder,
                                              IREE_SV("test_fn"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    IREE_ASSERT_OK(loom_test_func_build(
        &module_builder, 0, 0, 0, callee, nullptr, 0, nullptr, 0, nullptr, 0,
        nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op_));
    body_ = loom_func_like_body(loom_func_like_cast(module_, func_op_));
    body_->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(body_), &builder_);
    builder_.ip.parent_op = func_op_;

    IREE_ASSERT_OK(
        loom_value_fact_table_initialize(&fact_table_, &analysis_arena_, 16));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_block_t* AppendBlock() {
    loom_block_t* block = nullptr;
    IREE_CHECK_OK(loom_region_append_block(module_, body_, &block));
    return block;
  }

  void SetBlock(loom_block_t* block) {
    loom_builder_set_block(&builder_, block);
    builder_.ip.parent_op = func_op_;
  }

  loom_value_id_t AddBlockArg(loom_block_t* block) {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_module_define_value(
        module_, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &value_id));
    IREE_CHECK_OK(loom_block_add_arg(module_, block, value_id));
    return value_id;
  }

  loom_value_id_t BuildIndexConstant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN, &op));
    loom_value_id_t result = loom_index_constant_result(op);
    IREE_CHECK_OK(loom_value_fact_table_define(
        &fact_table_, result, loom_value_facts_exact_i64(value)));
    return result;
  }

  loom_value_id_t BuildIndexCompare(uint8_t predicate, loom_value_id_t left,
                                    loom_value_id_t right) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_cmp_build(&builder_, predicate, left, right,
                                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                       loom_type_scalar(LOOM_SCALAR_TYPE_I1),
                                       LOOM_LOCATION_UNKNOWN, &op));
    return loom_index_cmp_result(op);
  }

  void BuildBranch(loom_block_t* dest) {
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_cfg_br_build(&builder_, dest, nullptr, 0,
                                     LOOM_LOCATION_UNKNOWN, &op));
  }

  void BuildConditionalBranch(loom_value_id_t condition,
                              loom_block_t* true_dest,
                              loom_block_t* false_dest) {
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_cfg_cond_br_build(&builder_, condition, true_dest,
                                          false_dest, LOOM_LOCATION_UNKNOWN,
                                          &op));
  }

  bool HasRelation(const loom_cfg_block_entry_condition_facts_t* facts,
                   loom_symbolic_integer_relation_t relation,
                   loom_value_id_t left, loom_value_id_t right) {
    for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
      const loom_condition_integer_relation_t& entry =
          facts->integer_relations[i];
      if (entry.relation == relation &&
          entry.left.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE &&
          entry.left.value_id == left &&
          entry.right.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE &&
          entry.right.value_id == right) {
        return true;
      }
    }
    return false;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_op_t* func_op_ = nullptr;
  loom_region_t* body_ = nullptr;
  loom_builder_t builder_;
  loom_value_fact_table_t fact_table_;
};

TEST_F(CfgConditionFactsTest, PropagatesNestedBranchRelationsToTailBlock) {
  loom_block_t* entry_block = loom_region_entry_block(body_);
  loom_block_t* in_bounds_block = AppendBlock();
  loom_block_t* done_block = AppendBlock();
  loom_block_t* rot_block = AppendBlock();
  loom_block_t* tail_block = AppendBlock();

  SetBlock(entry_block);
  loom_value_id_t lane = AddBlockArg(entry_block);
  loom_value_id_t pair = AddBlockArg(entry_block);
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &fact_table_, lane, loom_value_facts_make(0, 1023, 1)));
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &fact_table_, pair, loom_value_facts_make(0, 1023, 1)));
  loom_value_id_t half_cols = BuildIndexConstant(64);
  loom_value_id_t in_bounds =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_ULT, lane, half_cols);
  BuildConditionalBranch(in_bounds, in_bounds_block, done_block);

  SetBlock(in_bounds_block);
  loom_value_id_t half_dims = BuildIndexConstant(48);
  loom_value_id_t pair_in_rot =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_ULT, pair, half_dims);
  BuildConditionalBranch(pair_in_rot, rot_block, tail_block);

  SetBlock(rot_block);
  BuildBranch(done_block);

  SetBlock(tail_block);
  BuildBranch(done_block);

  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  loom_cfg_graph_t graph = {0};
  IREE_ASSERT_OK(
      loom_cfg_graph_build(module_, body_, &analysis_arena_, &graph));
  loom_dominance_info_t dominance = {0};
  IREE_ASSERT_OK(
      loom_dominance_info_initialize(module_, &analysis_arena_, &dominance));

  loom_cfg_condition_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_cfg_condition_fact_table_compute(
      module_, &graph, &fact_table_, &dominance, &analysis_arena_, &table));

  const loom_cfg_block_entry_condition_facts_t* in_bounds_facts =
      loom_cfg_condition_fact_table_block(&table, 1);
  ASSERT_NE(in_bounds_facts, nullptr);
  EXPECT_TRUE(HasRelation(in_bounds_facts, LOOM_SYMBOLIC_INTEGER_RELATION_LT,
                          lane, half_cols));

  const loom_cfg_block_entry_condition_facts_t* tail_facts =
      loom_cfg_condition_fact_table_block(&table, 4);
  ASSERT_NE(tail_facts, nullptr);
  EXPECT_TRUE(HasRelation(tail_facts, LOOM_SYMBOLIC_INTEGER_RELATION_LT, lane,
                          half_cols));
  EXPECT_TRUE(HasRelation(tail_facts, LOOM_SYMBOLIC_INTEGER_RELATION_GE, pair,
                          half_dims));
}

}  // namespace
}  // namespace loom
