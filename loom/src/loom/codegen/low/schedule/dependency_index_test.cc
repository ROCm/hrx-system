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
              loom_low_schedule_dependency_kind_t kind, uint16_t operand_index,
              int32_t minimum_issue_separation_cycles = 0) {
    loom_low_schedule_dependency_t dependency = {
        .producer_node = producer_node,
        .consumer_node = consumer_node,
        .minimum_issue_separation_cycles = minimum_issue_separation_cycles,
        .producer_attachment_index = UINT16_MAX,
        .consumer_attachment_index = UINT16_MAX,
        .producer_event_id = UINT16_MAX,
        .consumer_event_id = UINT16_MAX,
        .value_operand_index = operand_index,
        .kind = kind,
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
  Append(&graph, 0, 3, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, UINT16_MAX, -4);
  Append(&graph, 0, 4, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, UINT16_MAX);
  Append(&graph, 1, 3, LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE, 0, 3);
  Append(&graph, 1, 3, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, UINT16_MAX);
  Append(&graph, 0, 3, LOOM_LOW_SCHEDULE_DEPENDENCY_SSA, 1, -2);
  Append(&graph, 0, 3, LOOM_LOW_SCHEDULE_DEPENDENCY_SSA, 0, -5);
  Append(&graph, 2, 3, LOOM_LOW_SCHEDULE_DEPENDENCY_SSA, 2);

  std::array<uint32_t, 5> indegrees;
  loom_low_schedule_dependency_index_t index;
  loom_low_schedule_dependency_detail_index_t detail_index;
  const auto allocation_before = arena_.used_allocation_size;
  const auto scratch_before = scratch_arena_.used_allocation_size;
  IREE_ASSERT_OK(loom_low_schedule_dependency_index_initialize(
      &graph, indegrees.size(), &scratch_arena_, &arena_, indegrees.data(),
      &index, &detail_index));
  // Complete indexing for this small graph fits within 1 KiB in each arena.
  EXPECT_LE(arena_.used_allocation_size - allocation_before, 1024u);
  EXPECT_LE(scratch_arena_.used_allocation_size - scratch_before, 1024u);

  EXPECT_EQ(index.node_count, static_cast<uint32_t>(indegrees.size()));
  EXPECT_EQ(index.group_count, 4u);
  EXPECT_EQ(detail_index.dependency_count, 7u);
  EXPECT_EQ(indegrees, (std::array<uint32_t, 5>{0, 0, 0, 6, 1}));
  EXPECT_EQ(loom_low_schedule_dependency_index_group_begin(&index, 0), 0u);
  EXPECT_EQ(loom_low_schedule_dependency_index_group_end(&index, 0), 2u);
  const loom_low_schedule_dependency_group_t* group0 =
      loom_low_schedule_dependency_index_group_at(&index, 0);
  const loom_low_schedule_dependency_group_t* group1 =
      loom_low_schedule_dependency_index_group_at(&index, 1);
  const loom_low_schedule_dependency_group_t* group2 =
      loom_low_schedule_dependency_index_group_at(&index, 2);
  const loom_low_schedule_dependency_group_t* group3 =
      loom_low_schedule_dependency_index_group_at(&index, 3);
  EXPECT_EQ(group0->producer_node, 0u);
  EXPECT_EQ(group0->consumer_node, 3u);
  EXPECT_EQ(group0->dependency_count, 3u);
  EXPECT_EQ(group0->minimum_issue_separation_cycles, -2);
  EXPECT_TRUE(loom_low_schedule_dependency_index_group_has_ssa(&index, 0));
  EXPECT_TRUE(loom_low_schedule_dependency_index_group_has_effect(&index, 0));
  EXPECT_EQ(group1->producer_node, 0u);
  EXPECT_EQ(group1->consumer_node, 4u);
  EXPECT_EQ(group1->dependency_count, 1u);
  EXPECT_FALSE(loom_low_schedule_dependency_index_group_has_ssa(&index, 1));
  EXPECT_TRUE(loom_low_schedule_dependency_index_group_has_effect(&index, 1));
  EXPECT_EQ(loom_low_schedule_dependency_index_group_begin(&index, 1), 2u);
  EXPECT_EQ(loom_low_schedule_dependency_index_group_end(&index, 1), 3u);
  EXPECT_EQ(loom_low_schedule_dependency_index_group_begin(&index, 2), 3u);
  EXPECT_EQ(loom_low_schedule_dependency_index_group_end(&index, 2), 4u);
  EXPECT_FALSE(loom_low_schedule_dependency_index_group_has_ssa(&index, 2));
  EXPECT_TRUE(loom_low_schedule_dependency_index_group_has_effect(&index, 2));
  EXPECT_TRUE(loom_low_schedule_dependency_index_group_has_ssa(&index, 3));
  EXPECT_FALSE(loom_low_schedule_dependency_index_group_has_effect(&index, 3));
  EXPECT_EQ(group2->producer_node, 1u);
  EXPECT_EQ(group2->consumer_node, 3u);
  EXPECT_EQ(group2->dependency_count, 2u);
  EXPECT_EQ(group2->minimum_issue_separation_cycles, 3);
  EXPECT_EQ(group3->producer_node, 2u);
  EXPECT_EQ(group3->consumer_node, 3u);
  EXPECT_EQ(group3->dependency_count, 1u);

  // Leave SSA, non-SSA, and mixed group zero as the final relation in turn.
  for (const auto& producer_order :
       {std::array<uint32_t, 3>{0, 1, 2}, {2, 0, 1}, {1, 2, 0}}) {
    loom_low_schedule_dependency_frontier_t frontier;
    IREE_ASSERT_OK(loom_low_schedule_dependency_frontier_initialize(
        &index, &arena_, &frontier));
    EXPECT_EQ(frontier.node_count, static_cast<uint32_t>(indegrees.size()));
    EXPECT_EQ(frontier.remaining_producer_counts[3], 3u);
    EXPECT_EQ(frontier.remaining_producer_counts[4], 1u);
    EXPECT_EQ(
        loom_low_schedule_dependency_frontier_remaining_group(&frontier, 3),
        LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_NONE);
    EXPECT_EQ(
        loom_low_schedule_dependency_frontier_remaining_group(&frontier, 4),
        1u);
    uint32_t publication_count = 1;
    const uint32_t last_group = loom_low_schedule_dependency_index_group_begin(
        &index, producer_order.back());
    for (uint32_t producer_node : producer_order) {
      const uint32_t group_begin =
          loom_low_schedule_dependency_index_group_begin(&index, producer_node);
      const uint32_t group_end =
          loom_low_schedule_dependency_index_group_end(&index, producer_node);
      for (uint32_t group_index = group_begin; group_index < group_end;
           ++group_index) {
        const loom_low_schedule_dependency_group_t* group =
            loom_low_schedule_dependency_index_group_at(&index, group_index);
        const uint32_t remaining_group =
            loom_low_schedule_dependency_frontier_consume_group(
                &frontier, group_index, group);
        if (group->consumer_node == 3 &&
            frontier.remaining_producer_counts[3] == 1) {
          EXPECT_EQ(remaining_group, last_group);
          EXPECT_EQ(loom_low_schedule_dependency_frontier_remaining_group(
                        &frontier, 3),
                    last_group);
          ++publication_count;
        } else {
          EXPECT_EQ(remaining_group, LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_NONE);
        }
      }
    }
    EXPECT_EQ(publication_count, 2u);
    EXPECT_EQ(frontier.consumed_group_count, index.group_count);
    EXPECT_EQ(
        loom_low_schedule_dependency_frontier_remaining_group(&frontier, 3),
        LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_NONE);
    EXPECT_EQ(
        loom_low_schedule_dependency_frontier_remaining_group(&frontier, 4),
        LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_NONE);
  }
}

TEST_F(ScheduleDependencyIndexTest, FanoutAccountingIsLinear) {
  for (const uint32_t fanout : {
           0u,
           1u,
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
      EXPECT_EQ(loom_low_schedule_dependency_frontier_remaining_group(
                    &frontier, consumer_node),
                consumer_node - 1);
    }
    EXPECT_EQ(frontier.consumed_group_count, 0u);
    for (uint32_t group_index = 0; group_index < fanout; ++group_index) {
      const loom_low_schedule_dependency_group_t* group =
          loom_low_schedule_dependency_index_group_at(&index, group_index);
      EXPECT_EQ(group->producer_node, 0u);
      EXPECT_EQ(group->consumer_node, group_index + 1);
      EXPECT_EQ(loom_low_schedule_dependency_frontier_consume_group(
                    &frontier, group_index, group),
                LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_NONE);
    }
    EXPECT_EQ(frontier.consumed_group_count, fanout);
  }
}

}  // namespace
}  // namespace loom
