// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/cfg_loop.h"

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

class CfgLoopTest : public ::testing::Test {
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
    const loom_symbol_ref_t callee = {
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
    IREE_ASSERT_OK(loom_test_func_build(&builder_, 0, 0, 0, callee, nullptr, 0,
                                        nullptr, 0, nullptr, 0, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &func_op_));
    body_ = loom_func_like_body(loom_func_like_cast(module_, func_op_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(body_), &builder_);
    builder_.ip.parent_op = func_op_;
    body_->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;

    iree_arena_initialize(&block_pool_, &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
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

  void BuildBranch(loom_block_t* destination) {
    loom_op_t* branch_op = nullptr;
    IREE_CHECK_OK(loom_cfg_br_build(&builder_, destination, nullptr, 0,
                                    LOOM_LOCATION_UNKNOWN, &branch_op));
  }

  void BuildConditionalBranch(loom_block_t* true_destination,
                              loom_block_t* false_destination) {
    loom_op_t* condition_op = nullptr;
    IREE_CHECK_OK(loom_test_constant_build(
        &builder_, loom_attr_i64(1), loom_type_scalar(LOOM_SCALAR_TYPE_I1),
        LOOM_LOCATION_UNKNOWN, &condition_op));
    loom_op_t* branch_op = nullptr;
    IREE_CHECK_OK(loom_cfg_cond_br_build(
        &builder_, loom_test_constant_result(condition_op), true_destination,
        false_destination, LOOM_LOCATION_UNKNOWN, &branch_op));
  }

  loom_cfg_loop_forest_t BuildForest(loom_cfg_graph_t* out_graph) {
    IREE_CHECK_OK(
        loom_cfg_graph_build(module_, body_, &analysis_arena_, out_graph));
    loom_cfg_loop_forest_t forest = {0};
    IREE_CHECK_OK(
        loom_cfg_loop_forest_build(out_graph, &analysis_arena_, &forest));
    return forest;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_op_t* func_op_ = nullptr;
  loom_region_t* body_ = nullptr;
  loom_builder_t builder_;
  iree_arena_allocator_t analysis_arena_;
};

TEST_F(CfgLoopTest, AcyclicGraphHasEmptyForest) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* exit = AppendBlock();
  SetBlock(entry);
  BuildBranch(exit);

  loom_cfg_graph_t graph = {0};
  const loom_cfg_loop_forest_t forest = BuildForest(&graph);

  EXPECT_EQ(graph.backward_edge_count, 0u);
  EXPECT_EQ(forest.interval_count, 0u);
  EXPECT_EQ(forest.reachable_backward_edge_count, 0u);
  EXPECT_EQ(forest.intervals, nullptr);
  EXPECT_EQ(forest.innermost_loop_indices, nullptr);
}

TEST_F(CfgLoopTest, BuildsNestedCanonicalIntervals) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* outer_header = AppendBlock();
  loom_block_t* inner_preheader = AppendBlock();
  loom_block_t* inner_header = AppendBlock();
  loom_block_t* inner_body = AppendBlock();
  loom_block_t* outer_latch = AppendBlock();
  loom_block_t* exit = AppendBlock();

  SetBlock(entry);
  BuildBranch(outer_header);
  SetBlock(outer_header);
  BuildConditionalBranch(inner_preheader, exit);
  SetBlock(inner_preheader);
  BuildBranch(inner_header);
  SetBlock(inner_header);
  BuildConditionalBranch(inner_body, outer_latch);
  SetBlock(inner_body);
  BuildBranch(inner_header);
  SetBlock(outer_latch);
  BuildBranch(outer_header);

  loom_cfg_graph_t graph = {0};
  const loom_cfg_loop_forest_t forest = BuildForest(&graph);

  ASSERT_EQ(graph.backward_edge_count, 2u);
  ASSERT_EQ(forest.interval_count, 2u);
  EXPECT_EQ(forest.reachable_backward_edge_count, 2u);
  const loom_cfg_loop_interval_t& outer = forest.intervals[0];
  EXPECT_EQ(outer.header_index, 1u);
  EXPECT_EQ(outer.latch_index, 5u);
  EXPECT_EQ(outer.entry_predecessor_index, 0u);
  EXPECT_EQ(outer.parent_loop_index, LOOM_CFG_LOOP_NONE);
  EXPECT_TRUE(outer.is_canonical);
  const loom_cfg_loop_interval_t& inner = forest.intervals[1];
  EXPECT_EQ(inner.header_index, 3u);
  EXPECT_EQ(inner.latch_index, 4u);
  EXPECT_EQ(inner.entry_predecessor_index, 2u);
  EXPECT_EQ(inner.parent_loop_index, 0u);
  EXPECT_TRUE(inner.is_canonical);

  ASSERT_NE(forest.innermost_loop_indices, nullptr);
  EXPECT_EQ(forest.innermost_loop_indices[0], LOOM_CFG_LOOP_NONE);
  EXPECT_EQ(forest.innermost_loop_indices[1], 0u);
  EXPECT_EQ(forest.innermost_loop_indices[2], 0u);
  EXPECT_EQ(forest.innermost_loop_indices[3], 1u);
  EXPECT_EQ(forest.innermost_loop_indices[4], 1u);
  EXPECT_EQ(forest.innermost_loop_indices[5], 0u);
  EXPECT_EQ(forest.innermost_loop_indices[6], LOOM_CFG_LOOP_NONE);

  const uint64_t trip_counts[] = {4, 8};
  uint64_t block_counts[7] = {0};
  EXPECT_TRUE(loom_cfg_loop_forest_calculate_block_execution_counts(
      &forest, &graph, trip_counts, block_counts));
  EXPECT_EQ(block_counts[0], 1u);
  EXPECT_EQ(block_counts[1], 5u);
  EXPECT_EQ(block_counts[2], 4u);
  EXPECT_EQ(block_counts[3], 36u);
  EXPECT_EQ(block_counts[4], 32u);
  EXPECT_EQ(block_counts[5], 4u);
  EXPECT_EQ(block_counts[6], 1u);

  const uint64_t zero_outer_trip_counts[] = {0, 8};
  EXPECT_TRUE(loom_cfg_loop_forest_calculate_block_execution_counts(
      &forest, &graph, zero_outer_trip_counts, block_counts));
  EXPECT_EQ(block_counts[0], 1u);
  EXPECT_EQ(block_counts[1], 1u);
  EXPECT_EQ(block_counts[2], 0u);
  EXPECT_EQ(block_counts[3], 0u);
  EXPECT_EQ(block_counts[4], 0u);
  EXPECT_EQ(block_counts[5], 0u);
  EXPECT_EQ(block_counts[6], 1u);

  const uint64_t header_overflow_trip_counts[] = {UINT64_MAX, 0};
  EXPECT_FALSE(loom_cfg_loop_forest_calculate_block_execution_counts(
      &forest, &graph, header_overflow_trip_counts, block_counts));
  const uint64_t nested_overflow_trip_counts[] = {UINT64_MAX / 2 + 1, 2};
  EXPECT_FALSE(loom_cfg_loop_forest_calculate_block_execution_counts(
      &forest, &graph, nested_overflow_trip_counts, block_counts));
}

TEST_F(CfgLoopTest, RetainsEntryPredecessorWithMultipleSuccessors) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* header = AppendBlock();
  loom_block_t* body = AppendBlock();
  loom_block_t* exit = AppendBlock();

  SetBlock(entry);
  BuildConditionalBranch(header, exit);
  SetBlock(header);
  BuildConditionalBranch(body, exit);
  SetBlock(body);
  BuildBranch(header);

  loom_cfg_graph_t graph = {0};
  const loom_cfg_loop_forest_t forest = BuildForest(&graph);

  ASSERT_EQ(forest.interval_count, 1u);
  EXPECT_EQ(forest.reachable_backward_edge_count, 1u);
  EXPECT_EQ(forest.intervals[0].entry_predecessor_index, 0u);
  EXPECT_TRUE(forest.intervals[0].is_canonical);
}

TEST_F(CfgLoopTest, RejectsUnmodeledBranchingInsideLoop) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* header = AppendBlock();
  loom_block_t* body = AppendBlock();
  loom_block_t* latch = AppendBlock();
  loom_block_t* exit = AppendBlock();

  SetBlock(entry);
  BuildBranch(header);
  SetBlock(header);
  BuildConditionalBranch(body, exit);
  SetBlock(body);
  BuildConditionalBranch(latch, exit);
  SetBlock(latch);
  BuildBranch(header);

  loom_cfg_graph_t graph = {0};
  const loom_cfg_loop_forest_t forest = BuildForest(&graph);

  ASSERT_EQ(forest.interval_count, 1u);
  EXPECT_TRUE(forest.intervals[0].is_canonical);
  const uint64_t trip_counts[] = {4};
  uint64_t block_counts[5] = {0};
  EXPECT_FALSE(loom_cfg_loop_forest_calculate_block_execution_counts(
      &forest, &graph, trip_counts, block_counts));
}

TEST_F(CfgLoopTest, RejectsUnrepresentedBackwardEdges) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* header = AppendBlock();
  loom_block_t* left_latch = AppendBlock();
  loom_block_t* right_latch = AppendBlock();

  SetBlock(entry);
  BuildBranch(header);
  SetBlock(header);
  BuildConditionalBranch(left_latch, right_latch);
  SetBlock(left_latch);
  BuildBranch(header);
  SetBlock(right_latch);
  BuildBranch(header);

  loom_cfg_graph_t graph = {0};
  const loom_cfg_loop_forest_t forest = BuildForest(&graph);

  EXPECT_EQ(forest.interval_count, 0u);
  EXPECT_EQ(forest.reachable_backward_edge_count, 2u);
  uint64_t block_counts[4] = {0};
  EXPECT_FALSE(loom_cfg_loop_forest_calculate_block_execution_counts(
      &forest, &graph, /*trip_counts=*/nullptr, block_counts));
}

TEST_F(CfgLoopTest, SideEntryInvalidatesOnlyInnerInterval) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* outer_header = AppendBlock();
  loom_block_t* inner_preheader = AppendBlock();
  loom_block_t* side_entry = AppendBlock();
  loom_block_t* inner_header = AppendBlock();
  loom_block_t* inner_body = AppendBlock();
  loom_block_t* outer_latch = AppendBlock();
  loom_block_t* exit = AppendBlock();

  SetBlock(entry);
  BuildBranch(outer_header);
  SetBlock(outer_header);
  BuildConditionalBranch(inner_preheader, side_entry);
  SetBlock(inner_preheader);
  BuildBranch(inner_header);
  SetBlock(side_entry);
  BuildBranch(inner_body);
  SetBlock(inner_header);
  BuildConditionalBranch(inner_body, outer_latch);
  SetBlock(inner_body);
  BuildBranch(inner_header);
  SetBlock(outer_latch);
  BuildConditionalBranch(outer_header, exit);

  loom_cfg_graph_t graph = {0};
  const loom_cfg_loop_forest_t forest = BuildForest(&graph);

  ASSERT_EQ(forest.interval_count, 2u);
  EXPECT_EQ(forest.reachable_backward_edge_count, 2u);
  EXPECT_TRUE(forest.intervals[0].is_canonical);
  EXPECT_FALSE(forest.intervals[1].is_canonical);
  EXPECT_EQ(forest.intervals[1].parent_loop_index, 0u);

  const uint64_t trip_counts[] = {4, 8};
  uint64_t block_counts[8] = {0};
  EXPECT_FALSE(loom_cfg_loop_forest_calculate_block_execution_counts(
      &forest, &graph, trip_counts, block_counts));
}

}  // namespace
}  // namespace loom
