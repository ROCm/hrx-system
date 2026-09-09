// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/completion_demand.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class ScheduleCompletionDemandTest : public ::testing::Test {
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
    for (auto& node : nodes_) {
      node.scheduled_ordinal = LOOM_LOW_SCHEDULE_NODE_NONE;
    }
  }

  void Append(uint32_t producer, uint32_t consumer,
              loom_low_schedule_dependency_kind_t kind =
                  LOOM_LOW_SCHEDULE_DEPENDENCY_SSA) {
    loom_low_schedule_dependency_t dependency = {};
    dependency.producer_node = producer;
    dependency.consumer_node = consumer;
    dependency.kind = kind;
    dependency.producer_attachment_index = UINT16_MAX;
    dependency.consumer_attachment_index = UINT16_MAX;
    dependency.producer_event_id = UINT16_MAX;
    dependency.consumer_event_id = UINT16_MAX;
    IREE_ASSERT_OK(loom_low_schedule_dependency_graph_append(
        &graph_, dependency, &arena_));
  }

  void Initialize(uint16_t domain_count) {
    std::vector<uint32_t> indegrees(nodes_.size());
    loom_low_schedule_dependency_index_t index;
    loom_low_schedule_dependency_detail_index_t details;
    IREE_ASSERT_OK(loom_low_schedule_dependency_index_initialize(
        &graph_, static_cast<uint32_t>(nodes_.size()), &scratch_arena_, &arena_,
        indegrees.data(), &index, &details));
    IREE_ASSERT_OK(loom_low_schedule_completion_demand_initialize(
        &index, nodes_.data(), domain_count, &arena_, &demand_));
  }

  // Independent raw-edge traversal checks the grouped reverse index and the
  // incremental demand sets against the currently selected completion.
  std::vector<bool> ReferenceAncestors(uint32_t root) {
    std::vector<bool> ancestors(nodes_.size());
    std::vector<uint32_t> pending = {root};
    ancestors[root] = true;
    while (!pending.empty()) {
      const uint32_t consumer = pending.back();
      pending.pop_back();
      for (uint32_t i = 0; i < graph_.count; ++i) {
        const auto* edge = loom_low_schedule_dependency_graph_at(&graph_, i);
        const uint32_t producer = edge->producer_node;
        if (edge->consumer_node != consumer ||
            edge->kind != LOOM_LOW_SCHEDULE_DEPENDENCY_SSA ||
            nodes_[producer].block_index != nodes_[consumer].block_index ||
            nodes_[producer].scheduled_ordinal != LOOM_LOW_SCHEDULE_NODE_NONE ||
            ancestors[producer]) {
          continue;
        }
        ancestors[producer] = true;
        pending.push_back(producer);
      }
    }
    return ancestors;
  }

  void ExpectAncestors(uint16_t domain, uint32_t root) {
    const auto ancestors = ReferenceAncestors(root);
    for (uint32_t node = 0; node < nodes_.size(); ++node) {
      if (nodes_[node].scheduled_ordinal != LOOM_LOW_SCHEDULE_NODE_NONE)
        continue;
      EXPECT_EQ(
          loom_low_schedule_completion_demand_contains(&demand_, domain, node),
          ancestors[node])
          << "domain " << domain << ", root " << root << ", node " << node;
    }
  }

  void Complete(uint32_t root) {
    const auto ancestors = ReferenceAncestors(root);
    for (uint32_t node = 0; node < nodes_.size(); ++node) {
      if (ancestors[node]) {
        nodes_[node].scheduled_ordinal = node;
        loom_low_schedule_completion_demand_complete(&demand_, node);
      }
    }
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  iree_arena_allocator_t scratch_arena_;
  loom_low_schedule_dependency_graph_t graph_;
  loom_low_schedule_completion_demand_t demand_;
  std::vector<loom_low_schedule_node_t> nodes_;
};

TEST_F(ScheduleCompletionDemandTest, PinsCompletionUntilItsConsumerRuns) {
  SetNodes(8);
  Append(0, 2);
  Append(1, 2);
  Append(2, 5);
  Append(3, 4);
  Append(4, 5);
  Append(1, 6);
  Append(6, 7);
  Initialize(2);
  const auto allocation_size = arena_.used_allocation_size;

  loom_low_schedule_completion_demand_nominate(&demand_, 0, 5);
  EXPECT_EQ(
      loom_low_schedule_completion_demand_select(&demand_, nodes_.data(), 0),
      5u);
  ExpectAncestors(0, 5);
  loom_low_schedule_completion_demand_nominate(&demand_, 0, 7);
  EXPECT_EQ(
      loom_low_schedule_completion_demand_select(&demand_, nodes_.data(), 0),
      5u);
  ExpectAncestors(0, 5);
  loom_low_schedule_completion_demand_nominate(&demand_, 1, 7);
  EXPECT_EQ(
      loom_low_schedule_completion_demand_select(&demand_, nodes_.data(), 1),
      7u);
  ExpectAncestors(1, 7);

  Complete(5);
  EXPECT_EQ(
      loom_low_schedule_completion_demand_select(&demand_, nodes_.data(), 0),
      7u);
  ExpectAncestors(0, 7);
  ExpectAncestors(1, 7);
  EXPECT_EQ(arena_.used_allocation_size, allocation_size);
}

TEST_F(ScheduleCompletionDemandTest, FiltersDependenciesAndBlockBoundaries) {
  SetNodes(8);
  nodes_[0].block_index = 1;
  Append(0, 1);
  Append(1, 5);
  Append(2, 5, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT);
  Append(3, 5, LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE);
  Append(4, 5, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT);
  Append(4, 5);
  Append(4, 5);
  Append(5, 6);
  nodes_[4].scheduled_ordinal = 0;
  Initialize(1);
  loom_low_schedule_completion_demand_nominate(&demand_, 0, 6);
  EXPECT_EQ(
      loom_low_schedule_completion_demand_select(&demand_, nodes_.data(), 0),
      6u);
  ExpectAncestors(0, 6);
  EXPECT_FALSE(loom_low_schedule_completion_demand_contains(&demand_, 0, 4));
  EXPECT_FALSE(loom_low_schedule_completion_demand_contains(&demand_, 0, 7));
}

TEST_F(ScheduleCompletionDemandTest, ReusesDemandAcrossManyCompletions) {
  SetNodes(130);
  for (uint32_t node = 1; node < nodes_.size(); ++node) {
    Append(0, node);
  }
  Initialize(3);
  const auto allocation_size = arena_.used_allocation_size;
  for (uint32_t root = 1; root < nodes_.size(); ++root) {
    for (uint16_t domain = 0; domain < 3; ++domain) {
      loom_low_schedule_completion_demand_nominate(&demand_, domain, root);
      EXPECT_EQ(loom_low_schedule_completion_demand_select(
                    &demand_, nodes_.data(), domain),
                root);
      ExpectAncestors(domain, root);
    }
    Complete(root);
  }
  EXPECT_EQ(arena_.used_allocation_size, allocation_size);
}

TEST_F(ScheduleCompletionDemandTest, EmptyDomainsHaveNoDemandStorage) {
  SetNodes(4);
  Initialize(0);
  EXPECT_EQ(demand_.incoming_starts, nullptr);
  EXPECT_EQ(demand_.demanded_bits, nullptr);
  EXPECT_EQ(demand_.worklist, nullptr);
  EXPECT_EQ(demand_.roots, nullptr);
  EXPECT_EQ(demand_.nominations.bits, nullptr);
}

TEST_F(ScheduleCompletionDemandTest, NominationsAreOrderedAcrossSummaryLevels) {
  // Cross both a 64-bit leaf boundary and a 64-word summary boundary.
  SetNodes(4098);
  Initialize(2);
  const auto allocation_size = arena_.used_allocation_size;
  for (uint32_t node : {4097u, 4096u, 4095u, 64u, 63u, 1u, 0u}) {
    loom_low_schedule_completion_demand_nominate(&demand_, 0, node);
    loom_low_schedule_completion_demand_nominate(&demand_, 0, node);
  }
  loom_low_schedule_completion_demand_nominate(&demand_, 1, 4097);
  for (uint32_t node : {0u, 1u, 63u, 64u, 4095u, 4096u, 4097u}) {
    EXPECT_EQ(
        loom_low_schedule_completion_demand_select(&demand_, nodes_.data(), 0),
        node);
    Complete(node);
  }
  EXPECT_EQ(
      loom_low_schedule_completion_demand_select(&demand_, nodes_.data(), 0),
      LOOM_LOW_SCHEDULE_NODE_NONE);
  EXPECT_EQ(
      loom_low_schedule_completion_demand_select(&demand_, nodes_.data(), 1),
      LOOM_LOW_SCHEDULE_NODE_NONE);
  EXPECT_EQ(arena_.used_allocation_size, allocation_size);
}

TEST_F(ScheduleCompletionDemandTest,
       EarlierNominationDoesNotInterruptPinnedRoot) {
  SetNodes(65);
  Initialize(1);
  loom_low_schedule_completion_demand_nominate(&demand_, 0, 64);
  EXPECT_EQ(
      loom_low_schedule_completion_demand_select(&demand_, nodes_.data(), 0),
      64u);
  loom_low_schedule_completion_demand_nominate(&demand_, 0, 1);
  EXPECT_EQ(
      loom_low_schedule_completion_demand_select(&demand_, nodes_.data(), 0),
      64u);
  Complete(64);
  EXPECT_EQ(
      loom_low_schedule_completion_demand_select(&demand_, nodes_.data(), 0),
      1u);
  Complete(1);
  EXPECT_EQ(
      loom_low_schedule_completion_demand_select(&demand_, nodes_.data(), 0),
      LOOM_LOW_SCHEDULE_NODE_NONE);
}

}  // namespace
}  // namespace loom
