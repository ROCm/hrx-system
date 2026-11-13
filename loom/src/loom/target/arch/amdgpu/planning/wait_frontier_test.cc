// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_frontier.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/util/cfg_graph_test_util.h"

namespace loom {
namespace {

using Node = loom_amdgpu_wait_frontier_node_t;

Node Access(bool write, bool refined) {
  Node node = {};
  const auto space =
      loom_amdgpu_wait_memory_space_flag(LOOM_LOW_MEMORY_SPACE_GLOBAL);
  if (write) {
    node.write_counter_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE;
    node.write_space_flags = space;
    node.flags =
        refined ? LOOM_AMDGPU_WAIT_FRONTIER_NODE_FLAG_REFINED_WRITE : 0;
  } else {
    node.read_counter_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD;
    node.read_space_flags = space;
    node.flags = refined ? LOOM_AMDGPU_WAIT_FRONTIER_NODE_FLAG_REFINED_READ : 0;
  }
  return node;
}

class AmdgpuWaitFrontierTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Each block has one classified memory access. The query starts before
  // predecessor emission, exercising the static state used for unknown edges.
  void ExpectIncoming(const std::vector<Node>& nodes,
                      const std::vector<std::vector<uint16_t>>& successors,
                      uint16_t block_index, uint32_t expected_counters) {
    testing::CfgGraph graph(successors);
    std::vector<loom_low_schedule_block_t> blocks(nodes.size());
    std::vector<uint32_t> indices(nodes.size());
    std::vector<loom_amdgpu_wait_completion_node_t> completions(nodes.size());
    for (uint32_t index = 0; index < nodes.size(); ++index) {
      blocks[index].scheduled_node_start = index;
      blocks[index].scheduled_node_count = 1;
      indices[index] = index;
    }
    const loom_low_schedule_table_t schedule = {
        .blocks = blocks.data(),
        .block_count = blocks.size(),
        .cfg_graph = *graph.get(),
        .node_count = nodes.size(),
        .scheduled_node_indices = indices.data(),
        .scheduled_node_count = indices.size(),
    };
    loom_amdgpu_wait_frontier_t frontier = {};
    IREE_ASSERT_OK(loom_amdgpu_wait_frontier_initialize(
        &schedule, nullptr, nodes.data(), completions.data(), nullptr, 0,
        nullptr, &arena_, &frontier));
    loom_amdgpu_wait_frontier_begin_block(&frontier, block_index);
    EXPECT_EQ(loom_amdgpu_wait_frontier_memory_dependency_mask(
                  &frontier, &nodes[block_index]),
              expected_counters);
  }

  // Backing storage for the bounded frontier summaries.
  iree_arena_block_pool_t pool_;
  // Analysis storage retained through each incoming-state query.
  iree_arena_allocator_t arena_;
};

TEST_F(AmdgpuWaitFrontierTest, RefinedForwardAccessPreservesDisjointProducer) {
  for (bool write : {false, true}) {
    const Node producer = Access(write, true);
    ExpectIncoming({producer, Access(!write, true), Access(!write, false)},
                   {{1}, {2}, {2}}, 2,
                   producer.read_counter_mask | producer.write_counter_mask);
  }
}

TEST_F(AmdgpuWaitFrontierTest, UnrefinedForwardAccessCompletesRefinedProducer) {
  for (bool write : {false, true}) {
    ExpectIncoming(
        {Access(write, true), Access(!write, false), Access(!write, false)},
        {{1}, {2}, {2}}, 2, 0);
  }
}

TEST_F(AmdgpuWaitFrontierTest, RefinedForwardAccessCompletesCoarseProducer) {
  for (bool write : {false, true}) {
    ExpectIncoming(
        {Access(write, false), Access(!write, true), Access(!write, false)},
        {{1}, {2}, {2}}, 2, 0);
  }
}

TEST_F(AmdgpuWaitFrontierTest, BackedgeLosesSourceRefinement) {
  for (bool write : {false, true}) {
    const Node producer = Access(write, true);
    ExpectIncoming({{}, Access(!write, true), producer}, {{1}, {2}, {1}}, 1,
                   producer.read_counter_mask | producer.write_counter_mask);
  }
}

TEST_F(AmdgpuWaitFrontierTest, OrderingCompletesBothRefinedAccessClasses) {
  const Node ordering = {
      .flags = LOOM_AMDGPU_WAIT_FRONTIER_NODE_FLAG_ORDERS_MEMORY,
  };
  for (bool write : {false, true}) {
    ExpectIncoming({Access(write, true), ordering, Access(!write, false)},
                   {{1}, {2}, {2}}, 2, 0);
  }
}

}  // namespace
}  // namespace loom
