// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/source_suffix.h"

#include <numeric>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class ScheduleSourceSuffixTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(128 * 1024, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
    loom_low_schedule_dependency_graph_initialize(&graph_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void SetNodes(uint32_t count) {
    nodes_.resize(count);
    scheduled_nodes_.resize(count);
    std::iota(scheduled_nodes_.begin(), scheduled_nodes_.end(), 0u);
    block_ = {};
    block_.node_count = count;
    block_.scheduled_node_count = count;
    schedule_ = {};
    schedule_.blocks = &block_;
    schedule_.block_count = 1;
    schedule_.nodes = nodes_.data();
    schedule_.node_count = count;
    schedule_.scheduled_node_indices = scheduled_nodes_.data();
    schedule_.scheduled_node_count = count;
  }

  void Append(uint32_t producer_node, uint32_t consumer_node,
              int32_t minimum_issue_separation_cycles,
              loom_low_schedule_dependency_kind_t kind =
                  LOOM_LOW_SCHEDULE_DEPENDENCY_SSA) {
    loom_low_schedule_dependency_t dependency = {
        .producer_node = producer_node,
        .consumer_node = consumer_node,
        .minimum_issue_separation_cycles = minimum_issue_separation_cycles,
        .producer_attachment_index = UINT16_MAX,
        .consumer_attachment_index = UINT16_MAX,
        .producer_event_id = UINT16_MAX,
        .consumer_event_id = UINT16_MAX,
        .value_operand_index = UINT16_MAX,
        .kind = kind};
    IREE_ASSERT_OK(loom_low_schedule_dependency_graph_append(
        &graph_, dependency, &arena_));
  }

  iree_status_t BuildBounds(const uint32_t** out_bounds) {
    std::vector<uint32_t> indegrees(nodes_.size());
    loom_low_schedule_dependency_index_t dependency_index;
    loom_low_schedule_dependency_detail_index_t detail_index;
    IREE_RETURN_IF_ERROR(loom_low_schedule_dependency_index_initialize(
        &graph_, static_cast<uint32_t>(nodes_.size()), &scratch_arena_, &arena_,
        indegrees.data(), &dependency_index, &detail_index));
    return loom_low_schedule_source_suffix_bounds_build(
        &schedule_, &dependency_index, &scratch_arena_, &arena_, out_bounds);
  }

  iree_arena_block_pool_t block_pool_ = {};
  iree_arena_allocator_t arena_ = {};
  iree_arena_allocator_t scratch_arena_ = {};
  loom_low_schedule_dependency_graph_t graph_ = {};
  loom_low_schedule_block_t block_ = {};
  loom_low_schedule_table_t schedule_ = {};
  std::vector<loom_low_schedule_node_t> nodes_;
  std::vector<uint32_t> scheduled_nodes_;
};

TEST_F(ScheduleSourceSuffixTest, CutsDependencyPathsAtEachSourceSuffix) {
  SetNodes(5);
  Append(0, 2, 5);
  Append(1, 3, 7);
  Append(2, 4, 3);
  Append(3, 4, 1);

  const uint32_t* bounds = nullptr;
  IREE_ASSERT_OK(BuildBounds(&bounds));
  ASSERT_NE(bounds, nullptr);
  EXPECT_EQ(bounds[0], 8u);
  EXPECT_EQ(bounds[1], 8u);
  EXPECT_EQ(bounds[2], 3u);
  EXPECT_EQ(bounds[3], 1u);
  EXPECT_EQ(bounds[4], 0u);
}

TEST_F(ScheduleSourceSuffixTest, CarriesMandatorySourceRangeTransitions) {
  SetNodes(4);
  nodes_[2].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY;

  const uint32_t* bounds = nullptr;
  IREE_ASSERT_OK(BuildBounds(&bounds));
  ASSERT_NE(bounds, nullptr);
  EXPECT_EQ(bounds[0], 2u);
  EXPECT_EQ(bounds[1], 2u);
  EXPECT_EQ(bounds[2], 1u);
  EXPECT_EQ(bounds[3], 0u);
}

TEST_F(ScheduleSourceSuffixTest, ZeroWidthNodeRemovesTransitionGuarantee) {
  SetNodes(4);
  nodes_[1].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ZERO_ISSUE_WIDTH;
  nodes_[2].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY;

  const uint32_t* bounds = nullptr;
  IREE_ASSERT_OK(BuildBounds(&bounds));
  ASSERT_NE(bounds, nullptr);
  EXPECT_EQ(bounds[0], 1u);
  EXPECT_EQ(bounds[1], 1u);
  EXPECT_EQ(bounds[2], 1u);
  EXPECT_EQ(bounds[3], 0u);
}

TEST_F(ScheduleSourceSuffixTest, IgnoresPathsThatLeaveTheSourceSuffix) {
  SetNodes(3);
  scheduled_nodes_ = {2, 0, 1};
  schedule_.scheduled_node_indices = scheduled_nodes_.data();
  Append(2, 0, 100);

  const uint32_t* bounds = nullptr;
  IREE_ASSERT_OK(BuildBounds(&bounds));
  ASSERT_NE(bounds, nullptr);
  EXPECT_EQ(bounds[0], 0u);
  EXPECT_EQ(bounds[1], 0u);
  EXPECT_EQ(bounds[2], 0u);
}

TEST_F(ScheduleSourceSuffixTest, NegativeSeparationAddsNoPathDistance) {
  SetNodes(2);
  Append(0, 1, -9);

  const uint32_t* bounds = nullptr;
  IREE_ASSERT_OK(BuildBounds(&bounds));
  ASSERT_NE(bounds, nullptr);
  EXPECT_EQ(bounds[0], 0u);
  EXPECT_EQ(bounds[1], 0u);
}

TEST_F(ScheduleSourceSuffixTest, EmptyScheduleHasNoRetainedTable) {
  SetNodes(0);

  const uint32_t* bounds = nullptr;
  IREE_ASSERT_OK(BuildBounds(&bounds));
  EXPECT_EQ(bounds, nullptr);
}

}  // namespace
}  // namespace loom
