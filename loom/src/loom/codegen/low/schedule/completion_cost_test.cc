// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/completion_cost.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class ScheduleCompletionCostTest : public ::testing::Test {
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

  void SetNodes(uint32_t count) { nodes_.resize(count); }

  void Append(uint32_t producer, uint32_t consumer,
              loom_low_schedule_dependency_kind_t kind) {
    const loom_low_schedule_dependency_t dependency = {
        .producer_node = producer,
        .consumer_node = consumer,
        .producer_attachment_index = UINT16_MAX,
        .consumer_attachment_index = UINT16_MAX,
        .producer_event_id = UINT16_MAX,
        .consumer_event_id = UINT16_MAX,
        .kind = kind,
    };
    IREE_ASSERT_OK(loom_low_schedule_dependency_graph_append(
        &graph_, dependency, &arena_));
  }

  void Initialize() {
    std::vector<uint32_t> indegrees(nodes_.size());
    loom_low_schedule_dependency_detail_index_t details;
    IREE_ASSERT_OK(loom_low_schedule_dependency_index_initialize(
        &graph_, static_cast<uint32_t>(nodes_.size()), &scratch_arena_, &arena_,
        indegrees.data(), &index_, &details));
    IREE_ASSERT_OK(loom_low_schedule_dependency_incoming_index_initialize(
        &index_, nodes_.data(), &arena_, &incoming_index_));
    node_cycles_.resize(nodes_.size());
  }

  uint16_t Query(uint32_t consumer) {
    return loom_low_schedule_completion_cost_query(node_cycles_.data(), &index_,
                                                   &incoming_index_, consumer);
  }

  void Schedule(uint32_t node, uint16_t opened_wait_cycles) {
    loom_low_schedule_completion_cost_note_node_scheduled(
        node_cycles_.data(), &index_, &incoming_index_, node,
        opened_wait_cycles);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  iree_arena_allocator_t scratch_arena_;
  loom_low_schedule_dependency_graph_t graph_;
  std::vector<loom_low_schedule_node_t> nodes_;
  std::vector<uint16_t> node_cycles_;
  loom_low_schedule_dependency_index_t index_;
  loom_low_schedule_dependency_incoming_index_t incoming_index_;
};

TEST_F(ScheduleCompletionCostTest, FirstEffectConsumerConsumesProducerCost) {
  SetNodes(3);
  Append(0, 1, LOOM_LOW_SCHEDULE_DEPENDENCY_SSA);
  Append(0, 1, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT);
  Append(0, 2, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT);
  Initialize();

  Schedule(0, 7);
  EXPECT_EQ(Query(1), 7u);
  EXPECT_EQ(Query(2), 7u);

  Schedule(1, 0);
  EXPECT_EQ(Query(2), 0u);
}

TEST_F(ScheduleCompletionCostTest, ConsumerTransfersCostToNewProducerEpoch) {
  SetNodes(3);
  Append(0, 1, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT);
  Append(1, 2, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT);
  Initialize();

  Schedule(0, 7);
  EXPECT_EQ(Query(1), 7u);

  Schedule(1, 3);
  EXPECT_EQ(Query(2), 3u);
}

TEST_F(ScheduleCompletionCostTest, CrossBlockCostRemainsConservative) {
  SetNodes(2);
  nodes_[1].block_index = 1;
  Append(0, 1, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT);
  Initialize();

  Schedule(0, 7);
  EXPECT_EQ(Query(1), 0u);
  loom_low_schedule_completion_cost_publish_cross_block(node_cycles_.data(), 1,
                                                        7);
  EXPECT_EQ(Query(1), 7u);
}

}  // namespace
}  // namespace loom
