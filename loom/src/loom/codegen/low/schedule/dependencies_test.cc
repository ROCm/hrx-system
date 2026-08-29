// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/dependencies.h"

#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class ScheduleDependenciesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(128 * 1024, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(ScheduleDependenciesTest, StableAcrossSegmentBoundaries) {
  loom_low_schedule_dependency_graph_t source;
  loom_low_schedule_dependency_graph_initialize(&source);
  constexpr uint32_t kDependencyCount =
      LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_CAPACITY + 1;
  for (uint32_t i = 0; i < kDependencyCount; ++i) {
    loom_low_schedule_dependency_t dependency = {};
    dependency.producer_node = i;
    dependency.consumer_node = i + 1;
    dependency.minimum_issue_separation_cycles = (i & 1u) == 0 ? -3 : 5;
    dependency.producer_attachment_index = static_cast<uint16_t>(i & 7u);
    dependency.consumer_attachment_index = static_cast<uint16_t>(i & 15u);
    dependency.producer_event_id = static_cast<uint16_t>(i & 31u);
    dependency.consumer_event_id = static_cast<uint16_t>(i & 63u);
    dependency.value_operand_index = static_cast<uint16_t>(i & 3u);
    dependency.producer_attachment_kind =
        LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_OPERAND;
    dependency.consumer_attachment_kind =
        LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_EFFECT;
    dependency.kind = LOOM_LOW_SCHEDULE_DEPENDENCY_SSA;
    dependency.separation_source =
        LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR;
    dependency.model_quality = 4;
    IREE_ASSERT_OK(loom_low_schedule_dependency_graph_append(
        &source, dependency, &arena_));
  }

  loom_low_schedule_dependency_graph_t graph;
  loom_low_schedule_dependency_graph_move(&source, &graph);

  EXPECT_EQ(source.count, 0u);
  EXPECT_EQ(graph.count, kDependencyCount);
  for (uint32_t i : {0u, kDependencyCount - 2, kDependencyCount - 1}) {
    const loom_low_schedule_dependency_t* dependency =
        loom_low_schedule_dependency_graph_at(&graph, i);
    EXPECT_EQ(dependency->producer_node, i);
    EXPECT_EQ(dependency->consumer_node, i + 1);
    EXPECT_EQ(dependency->minimum_issue_separation_cycles,
              (i & 1u) == 0 ? -3 : 5);
    EXPECT_EQ(dependency->producer_attachment_index, i & 7u);
    EXPECT_EQ(dependency->consumer_attachment_index, i & 15u);
    EXPECT_EQ(dependency->producer_event_id, i & 31u);
    EXPECT_EQ(dependency->consumer_event_id, i & 63u);
    EXPECT_EQ(dependency->value_operand_index, i & 3u);
    EXPECT_EQ(dependency->producer_attachment_kind,
              LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_OPERAND);
    EXPECT_EQ(dependency->consumer_attachment_kind,
              LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_EFFECT);
    EXPECT_EQ(dependency->separation_source,
              LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR);
    EXPECT_EQ(dependency->model_quality, 4);
  }
}

}  // namespace
}  // namespace loom
