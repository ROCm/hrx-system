// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/cfg_graph.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

class CfgGraphTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);

    iree_host_size_t cfg_vtable_count = 0;
    const loom_op_vtable_t* const* cfg_vtables =
        loom_cfg_dialect_vtables(&cfg_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_CFG, cfg_vtables, (uint16_t)cfg_vtable_count));
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
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&builder_, IREE_SV("test_fn"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    IREE_ASSERT_OK(loom_test_func_build(&builder_, 0, 0, 0, callee, nullptr, 0,
                                        nullptr, 0, nullptr, 0, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &func_op_));
    body_ = loom_func_like_body(loom_func_like_cast(module_, func_op_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(body_), &builder_);
    builder_.ip.parent_op = func_op_;
    body_->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;

    iree_arena_initialize(&block_pool_, &graph_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&graph_arena_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
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

  loom_value_id_t BuildCondition() {
    loom_op_t* condition_op = nullptr;
    IREE_CHECK_OK(loom_test_constant_build(
        &builder_, loom_attr_i64(1), loom_type_scalar(LOOM_SCALAR_TYPE_I1),
        LOOM_LOCATION_UNKNOWN, &condition_op));
    return loom_test_constant_result(condition_op);
  }

  void BuildBranch(loom_block_t* dest) {
    loom_op_t* branch_op = nullptr;
    IREE_ASSERT_OK(loom_cfg_br_build(&builder_, dest, nullptr, 0,
                                     LOOM_LOCATION_UNKNOWN, &branch_op));
  }

  void BuildConditionalBranch(loom_block_t* true_dest,
                              loom_block_t* false_dest) {
    loom_value_id_t condition = BuildCondition();
    loom_op_t* branch_op = nullptr;
    IREE_ASSERT_OK(loom_cfg_cond_br_build(&builder_, condition, true_dest,
                                          false_dest, LOOM_LOCATION_UNKNOWN,
                                          &branch_op));
  }

  void BuildGraph(loom_cfg_graph_t* out_graph) {
    IREE_ASSERT_OK(
        loom_cfg_graph_build(module_, body_, &graph_arena_, out_graph));
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_op_t* func_op_ = nullptr;
  loom_region_t* body_ = nullptr;
  loom_builder_t builder_;
  iree_arena_allocator_t graph_arena_;
};

TEST_F(CfgGraphTest, BuildsSuccessorsAndPredecessorsForDiamond) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* then_block = AppendBlock();
  loom_block_t* else_block = AppendBlock();
  loom_block_t* merge_block = AppendBlock();

  SetBlock(entry);
  BuildConditionalBranch(then_block, else_block);
  SetBlock(then_block);
  BuildBranch(merge_block);
  SetBlock(else_block);
  BuildBranch(merge_block);

  loom_cfg_graph_t graph = {0};
  BuildGraph(&graph);

  EXPECT_FALSE(graph.malformed);
  EXPECT_EQ(graph.block_count, 4u);
  EXPECT_EQ(graph.edge_count, 4u);

  loom_cfg_block_index_span_t entry_successors =
      loom_cfg_graph_successors(&graph, 0);
  ASSERT_EQ(entry_successors.count, 2u);
  EXPECT_EQ(entry_successors.values[0], 1u);
  EXPECT_EQ(entry_successors.values[1], 2u);

  loom_cfg_block_index_span_t merge_predecessors =
      loom_cfg_graph_predecessors(&graph, 3);
  ASSERT_EQ(merge_predecessors.count, 2u);
  EXPECT_EQ(merge_predecessors.values[0], 1u);
  EXPECT_EQ(merge_predecessors.values[1], 2u);

  EXPECT_EQ(loom_cfg_graph_block_index(&graph, merge_block), 3u);
  EXPECT_TRUE(loom_cfg_graph_block_is_reachable(&graph, 0));
  EXPECT_TRUE(loom_cfg_graph_block_is_reachable(&graph, 1));
  EXPECT_TRUE(loom_cfg_graph_block_is_reachable(&graph, 2));
  EXPECT_TRUE(loom_cfg_graph_block_is_reachable(&graph, 3));
}

TEST_F(CfgGraphTest, ReachabilitySkipsUnreachableBlocks) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* reachable = AppendBlock();
  loom_block_t* unreachable = AppendBlock();

  SetBlock(entry);
  BuildBranch(reachable);
  SetBlock(unreachable);
  loom_op_t* dead_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(
      &builder_, loom_attr_i64(0), loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      LOOM_LOCATION_UNKNOWN, &dead_op));

  loom_cfg_graph_t graph = {0};
  BuildGraph(&graph);

  EXPECT_FALSE(graph.malformed);
  EXPECT_EQ(graph.edge_count, 1u);
  EXPECT_TRUE(loom_cfg_graph_block_is_reachable(&graph, 0));
  EXPECT_TRUE(loom_cfg_graph_block_is_reachable(&graph, 1));
  EXPECT_FALSE(loom_cfg_graph_block_is_reachable(&graph, 2));
}

TEST_F(CfgGraphTest, OutsideSuccessorMarksGraphMalformed) {
  loom_block_t* entry = loom_region_entry_block(body_);
  SetBlock(entry);
  BuildBranch(loom_module_block(module_));

  loom_cfg_graph_t graph = {0};
  BuildGraph(&graph);

  EXPECT_TRUE(graph.malformed);
  EXPECT_EQ(graph.edge_count, 0u);
  EXPECT_TRUE(loom_cfg_graph_block_is_reachable(&graph, 0));
}

TEST_F(CfgGraphTest, SuccessorBeforeBlockEndMarksGraphMalformed) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* target = AppendBlock();
  SetBlock(entry);
  BuildBranch(target);
  loom_op_t* after_branch = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(
      &builder_, loom_attr_i64(0), loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      LOOM_LOCATION_UNKNOWN, &after_branch));

  loom_cfg_graph_t graph = {0};
  BuildGraph(&graph);

  EXPECT_TRUE(graph.malformed);
  EXPECT_EQ(graph.edge_count, 1u);
}

}  // namespace
}  // namespace loom
