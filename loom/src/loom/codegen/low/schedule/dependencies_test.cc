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
    IREE_ASSERT_OK(loom_low_schedule_dependency_graph_append(
        &source,
        {/*.producer_node=*/i,
         /*.consumer_node=*/i + 1,
         /*.kind=*/LOOM_LOW_SCHEDULE_DEPENDENCY_SSA,
         /*.operand_index=*/i & 3u},
        &arena_));
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
    EXPECT_EQ(dependency->operand_index, i & 3u);
  }
}

}  // namespace
}  // namespace loom
