// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/dependency_index.h"

#include <array>
#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class ScheduleDependencyIndexTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(128 * 1024, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void Append(loom_low_schedule_dependency_graph_t* graph,
              uint32_t producer_node, uint32_t consumer_node,
              loom_low_schedule_dependency_kind_t kind,
              uint32_t operand_index) {
    const loom_low_schedule_dependency_t dependency = {
        /*producer_node=*/producer_node,
        /*consumer_node=*/consumer_node,
        /*kind=*/kind,
        /*operand_index=*/operand_index,
    };
    IREE_ASSERT_OK(
        loom_low_schedule_dependency_graph_append(graph, dependency, &arena_));
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  iree_arena_allocator_t scratch_arena_;
};

TEST_F(ScheduleDependencyIndexTest, GroupsDuplicateProducerConsumerEdges) {
  loom_low_schedule_dependency_graph_t graph;
  loom_low_schedule_dependency_graph_initialize(&graph);
  Append(&graph, 0, 3, LOOM_LOW_SCHEDULE_DEPENDENCY_SSA, 0);
  Append(&graph, 0, 4, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, UINT32_MAX);
  Append(&graph, 1, 3, LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE, 0);
  Append(&graph, 0, 3, LOOM_LOW_SCHEDULE_DEPENDENCY_SSA, 1);
  Append(&graph, 0, 3, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, UINT32_MAX);
  Append(&graph, 2, 3, LOOM_LOW_SCHEDULE_DEPENDENCY_SSA, 2);

  std::array<uint32_t, 5> indegrees;
  loom_low_schedule_dependency_index_t index;
  loom_low_schedule_dependency_detail_index_t detail_index;
  IREE_ASSERT_OK(loom_low_schedule_dependency_index_initialize(
      &graph, indegrees.size(), &scratch_arena_, &arena_, indegrees.data(),
      &index, &detail_index));

  EXPECT_EQ(index.node_count, static_cast<uint32_t>(indegrees.size()));
  EXPECT_EQ(index.group_count, 4u);
  EXPECT_EQ(detail_index.dependency_count, 6u);
  EXPECT_EQ(indegrees, (std::array<uint32_t, 5>{0, 0, 0, 5, 1}));
  EXPECT_EQ(loom_low_schedule_dependency_index_group_begin(&index, 0), 0u);
  EXPECT_EQ(loom_low_schedule_dependency_index_group_end(&index, 0), 2u);
  const loom_low_schedule_dependency_group_t* group0 =
      loom_low_schedule_dependency_index_group_at(&index, 0);
  const loom_low_schedule_dependency_group_t* group1 =
      loom_low_schedule_dependency_index_group_at(&index, 1);
  const loom_low_schedule_dependency_group_t* group2 =
      loom_low_schedule_dependency_index_group_at(&index, 2);
  EXPECT_EQ(group0->consumer_node, 3u);
  EXPECT_EQ(group0->dependency_count, 3u);
  EXPECT_TRUE(loom_low_schedule_dependency_index_group_has_ssa(&index, 0));
  EXPECT_TRUE(loom_low_schedule_dependency_index_group_has_effect(&index, 0));
  EXPECT_EQ(group1->consumer_node, 4u);
  EXPECT_EQ(group1->dependency_count, 1u);
  EXPECT_FALSE(loom_low_schedule_dependency_index_group_has_ssa(&index, 1));
  EXPECT_TRUE(loom_low_schedule_dependency_index_group_has_effect(&index, 1));
  EXPECT_EQ(loom_low_schedule_dependency_index_group_begin(&index, 1), 2u);
  EXPECT_EQ(loom_low_schedule_dependency_index_group_end(&index, 1), 3u);
  EXPECT_EQ(loom_low_schedule_dependency_index_group_begin(&index, 2), 3u);
  EXPECT_EQ(loom_low_schedule_dependency_index_group_end(&index, 2), 4u);

  loom_low_schedule_dependency_frontier_t frontier;
  IREE_ASSERT_OK(loom_low_schedule_dependency_frontier_initialize(
      &index, &arena_, &frontier));
  EXPECT_EQ(frontier.node_count, static_cast<uint32_t>(indegrees.size()));
  EXPECT_EQ(frontier.remaining_producer_counts[3], 3u);
  EXPECT_EQ(frontier.remaining_producer_counts[4], 1u);
  EXPECT_EQ(
      loom_low_schedule_dependency_frontier_remaining_producer(&frontier, 4),
      0u);
  EXPECT_EQ(
      loom_low_schedule_dependency_frontier_consume_group(&frontier, 0, group0),
      LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_NONE);
  EXPECT_EQ(
      loom_low_schedule_dependency_frontier_consume_group(&frontier, 1, group2),
      2u);
  EXPECT_EQ(frontier.remaining_producer_counts[3], 1u);
  EXPECT_EQ(frontier.consumed_group_count, 2u);
}

TEST_F(ScheduleDependencyIndexTest, FanoutAccountingIsLinear) {
  for (const uint32_t fanout : {
           16u,
           256u,
           4096u,
           LOOM_LOW_SCHEDULE_DEPENDENCY_DETAIL_SEGMENT_CAPACITY + 1u,
       }) {
    loom_low_schedule_dependency_graph_t graph;
    loom_low_schedule_dependency_graph_initialize(&graph);
    for (uint32_t i = 0; i < fanout; ++i) {
      Append(&graph, 0, i + 1, LOOM_LOW_SCHEDULE_DEPENDENCY_SSA, 0);
    }

    std::vector<uint32_t> indegrees(fanout + 1);
    loom_low_schedule_dependency_index_t index;
    loom_low_schedule_dependency_detail_index_t detail_index;
    IREE_ASSERT_OK(loom_low_schedule_dependency_index_initialize(
        &graph, static_cast<uint32_t>(indegrees.size()), &scratch_arena_,
        &arena_, indegrees.data(), &index, &detail_index));
    EXPECT_EQ(index.group_count, fanout);

    loom_low_schedule_dependency_frontier_t frontier;
    IREE_ASSERT_OK(loom_low_schedule_dependency_frontier_initialize(
        &index, &arena_, &frontier));
    for (uint32_t consumer_node = 1; consumer_node <= fanout; ++consumer_node) {
      EXPECT_EQ(loom_low_schedule_dependency_frontier_remaining_producer(
                    &frontier, consumer_node),
                0u);
      EXPECT_EQ(loom_low_schedule_dependency_frontier_remaining_producer(
                    &frontier, consumer_node),
                0u);
    }
    EXPECT_EQ(frontier.consumed_group_count, 0u);
    for (uint32_t group_index = 0; group_index < fanout; ++group_index) {
      const loom_low_schedule_dependency_group_t* group =
          loom_low_schedule_dependency_index_group_at(&index, group_index);
      EXPECT_EQ(loom_low_schedule_dependency_frontier_consume_group(&frontier,
                                                                    0, group),
                LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_NONE);
    }
    EXPECT_EQ(frontier.consumed_group_count, fanout);
  }
}

}  // namespace
}  // namespace loom
