// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/graph_update.h"

#include <algorithm>
#include <array>
#include <span>
#include <vector>

#include "common/context.h"
#include "common/device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Compatibility consumes descriptions and context identity, without executing
// queues. Construct real graphs against the context's ordinary arena pool; only
// the independent scratch allocator is observed or made to fail.
class GraphUpdateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_atomic_ref_count_init(&context_.ref_count);
    context_.device_entry = &device_;
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &device_.block_pool);
    const iree_allocator_t allocator = {
        .self = this,
        .ctl =
            +[](void* self, iree_allocator_command_t command,
                const void* parameters, void** pointer) {
              auto* test = static_cast<GraphUpdateTest*>(self);
              if (command != IREE_ALLOCATOR_COMMAND_FREE) {
                ++test->allocation_count_;
                if (test->fail_allocations_) {
                  return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
                }
              }
              auto system = iree_allocator_system();
              return system.ctl(system.self, command, parameters, pointer);
            },
    };
    iree_arena_block_pool_initialize(64 * 1024, allocator, &scratch_pool_);
    iree_arena_initialize(&scratch_pool_, &scratch_);
  }

  void TearDown() override {
    for (auto* graph : graphs_) {
      iree_hal_streaming_graph_release(graph);
    }
    iree_arena_deinitialize(&scratch_);
    iree_arena_block_pool_deinitialize(&scratch_pool_);
    iree_arena_block_pool_deinitialize(&device_.block_pool);
  }

  iree_hal_streaming_graph_t* CreateGraph() {
    iree_hal_streaming_graph_t* graph = nullptr;
    IREE_CHECK_OK(iree_hal_streaming_graph_create(
        &context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &graph));
    graphs_.push_back(graph);
    return graph;
  }

  iree_hal_streaming_graph_node_t* AddEmpty(
      iree_hal_streaming_graph_t* graph,
      std::span<iree_hal_streaming_graph_node_t*> dependencies = {}) {
    iree_hal_streaming_graph_node_t* node = nullptr;
    IREE_CHECK_OK(iree_hal_streaming_graph_add_empty_node(
        graph, dependencies.data(), dependencies.size(), &node));
    return node;
  }

  void Check(iree_hal_streaming_graph_t* original,
             iree_hal_streaming_graph_t* replacement,
             iree_hal_streaming_graph_exec_update_result_t expected_result =
                 IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_SUCCESS,
             iree_hal_streaming_graph_node_t* expected_node = nullptr) {
    auto checkpoint = iree_arena_checkpoint_save(&scratch_);
    iree_hal_streaming_graph_node_t* error_node = nullptr;
    iree_hal_streaming_graph_exec_update_result_t result;
    IREE_ASSERT_OK(iree_hal_streaming_graph_check_update(
        original, replacement, &scratch_, &error_node, &result));
    EXPECT_EQ(result, expected_result);
    EXPECT_EQ(error_node, expected_node);
    EXPECT_EQ(scratch_.used_allocation_size, checkpoint.used_allocation_size);
    EXPECT_EQ(scratch_.block_head, checkpoint.block_head);
    EXPECT_EQ(scratch_.block_bytes_remaining, checkpoint.block_bytes_remaining);
  }

  // Borrowed context identity and the graph-description storage provider.
  iree_hal_streaming_context_t context_ = {};
  // Owns the ordinary pool used by production graph construction.
  iree_hal_streaming_device_t device_ = {};
  // Graphs constructed by each case, released before their storage provider.
  std::vector<iree_hal_streaming_graph_t*> graphs_;
  // Separate dependency whose allocations the tests can count or reject.
  iree_arena_block_pool_t scratch_pool_ = {};
  // Scratch arena restored by every compatibility check.
  iree_arena_allocator_t scratch_ = {};
  // Number of backing allocations attempted for scratch storage.
  size_t allocation_count_ = 0;
  // Reject backing allocation attempts without affecting graph construction.
  bool fail_allocations_ = false;
};

TEST_F(GraphUpdateTest,
       UnorderedSetsReuseScratchAndRejectDifferentPredecessors) {
  auto* original = CreateGraph();
  auto* replacement = CreateGraph();
  auto* different = CreateGraph();
  std::array<std::array<iree_hal_streaming_graph_node_t*, 257>, 3> parents;
  std::array<iree_hal_streaming_graph_t*, 3> graphs = {original, replacement,
                                                       different};
  std::array<iree_hal_streaming_graph_node_t*, 3> first_sinks;
  for (size_t graph_index = 0; graph_index < graphs.size(); ++graph_index) {
    auto* graph = graphs[graph_index];
    auto& nodes = parents[graph_index];
    for (auto*& node : nodes) {
      node = AddEmpty(graph);
    }
    std::vector<iree_hal_streaming_graph_node_t*> predecessors(nodes.begin(),
                                                               nodes.end() - 1);
    if (graph == different) {
      predecessors.back() = nodes.back();
    }
    if (graph == original) {
      first_sinks[graph_index] = AddEmpty(graph, predecessors);
    } else {
      first_sinks[graph_index] = AddEmpty(graph);
      std::vector<iree_hal_streaming_graph_node_t*> destinations(
          predecessors.size(), first_sinks[graph_index]);
      IREE_ASSERT_OK(iree_hal_streaming_graph_add_dependencies(
          graph, predecessors.data(), destinations.data(),
          predecessors.size()));
    }
    // Reuse the same bits for a second sink, in a different predecessor order.
    if (graph != original) {
      std::reverse(predecessors.begin(), predecessors.end());
    }
    AddEmpty(graph, predecessors);
  }
  fail_allocations_ = true;
  iree_hal_streaming_graph_node_t* error_node = nullptr;
  iree_hal_streaming_graph_exec_update_result_t result;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_streaming_graph_check_update(original, replacement, &scratch_,
                                            &error_node, &result));
  EXPECT_EQ(result, IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_ERROR);
  EXPECT_EQ(scratch_.used_allocation_size, 0u);
  fail_allocations_ = false;
  Check(original, replacement);
  const size_t allocations = allocation_count_;
  fail_allocations_ = true;
  for (size_t i = 0; i < 20; ++i) {
    Check(original, different,
          IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_TOPOLOGY_CHANGED,
          first_sinks[2]);
    Check(original, replacement);
    Check(replacement, original);
  }
  EXPECT_EQ(allocation_count_, allocations);
}

TEST_F(GraphUpdateTest, OrderedDependenciesNeedNoScratch) {
  auto* original = CreateGraph();
  auto* replacement = CreateGraph();
  std::array<iree_hal_streaming_graph_node_t*, 2> previous = {};
  for (size_t i = 0; i < 1024; ++i) {
    previous[0] = AddEmpty(original, {&previous[0], i ? 1u : 0u});
    previous[1] = AddEmpty(replacement, {&previous[1], i ? 1u : 0u});
  }
  fail_allocations_ = true;
  Check(original, replacement);
  EXPECT_EQ(allocation_count_, 0u);
  AddEmpty(replacement);
  Check(original, replacement,
        IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_TOPOLOGY_CHANGED);
  EXPECT_EQ(allocation_count_, 0u);
}

TEST_F(GraphUpdateTest, ChildFailureReportsEnclosingReplacementNode) {
  auto* original = CreateGraph();
  auto* replacement = CreateGraph();
  auto* original_child = CreateGraph();
  auto* replacement_child = CreateGraph();
  AddEmpty(original_child);
  iree_hal_streaming_graph_node_t* host_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      replacement_child, nullptr, 0, +[](void*) {}, nullptr, &host_node));
  Check(original_child, replacement_child,
        IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_NODE_TYPE_CHANGED, host_node);
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_child_graph_node(
      original, nullptr, 0, original_child, nullptr));
  iree_hal_streaming_graph_node_t* child_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_child_graph_node(
      replacement, nullptr, 0, replacement_child, &child_node));
  Check(original, replacement,
        IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_NODE_TYPE_CHANGED, child_node);
}

}  // namespace
