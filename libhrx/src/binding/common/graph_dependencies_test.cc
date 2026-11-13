// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <vector>

#include "common/context.h"
#include "common/device.h"
#include "common/graph.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Exercises production graph storage without creating a device or stream.
class GraphState {
 public:
  explicit GraphState(
      iree_allocator_t host_allocator = iree_allocator_system(),
      iree_allocator_t backing_allocator = iree_allocator_system()) {
    iree_atomic_ref_count_init(&context_.ref_count);
    context_.device_entry = &device_;
    iree_arena_block_pool_initialize(64 * 1024, backing_allocator,
                                     &device_.block_pool);
    IREE_CHECK_OK(iree_hal_streaming_graph_create(
        &context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, host_allocator,
        &graph_));
  }

  ~GraphState() {
    iree_hal_streaming_graph_release(graph_);
    iree_arena_block_pool_deinitialize(&device_.block_pool);
  }

  iree_hal_streaming_graph_t* graph() { return graph_; }

 private:
  // Storage owner supplying the graph's arena block pool.
  iree_hal_streaming_device_t device_ = {};
  // Initial reference belongs to this fixture; graphs borrow the native state.
  iree_hal_streaming_context_t context_ = {};
  // Production graph destroyed before its block pool.
  iree_hal_streaming_graph_t* graph_ = nullptr;
};

struct ProbedAllocator {
  // Allocation attempt ordinal rejected by the current transaction.
  int fail_on_attempt = 0;
  // Number of requests, including failures, reaching this allocator.
  int attempts = 0;
  // Number of successful allocations still owned by the graph or its pool.
  int balance = 0;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<ProbedAllocator*>(self);
    const bool allocates = command == IREE_ALLOCATOR_COMMAND_MALLOC ||
                           command == IREE_ALLOCATOR_COMMAND_CALLOC;
    if (allocates && ++allocator->attempts == allocator->fail_on_attempt) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected allocation failure");
    }
    const auto delegate = iree_allocator_system();
    IREE_RETURN_IF_ERROR(
        delegate.ctl(delegate.self, command, params, inout_ptr));
    allocator->balance += allocates;
    allocator->balance -= command == IREE_ALLOCATOR_COMMAND_FREE;
    return iree_ok_status();
  }

  iree_allocator_t allocator() { return {this, &Control}; }
};

TEST(GraphDependenciesTest, WideDependenciesSurviveRemovalRenumberingAndClone) {
  GraphState state;
  std::array<iree_hal_streaming_graph_node_t*, 128> predecessors;
  for (auto& predecessor : predecessors) {
    IREE_ASSERT_OK(iree_hal_streaming_graph_add_empty_node(
        state.graph(), nullptr, 0, &predecessor));
  }
  std::array<iree_hal_streaming_graph_node_t*, 2> joins;
  for (auto& join : joins) {
    IREE_ASSERT_OK(iree_hal_streaming_graph_add_empty_node(
        state.graph(), predecessors.data(), 8, &join));
  }
  // Interleave two destinations and a permutation of their predecessors. The
  // index must describe each set, independent of the last insertion order.
  for (size_t i = 0; i < 120; ++i) {
    auto* predecessor = predecessors[8 + (i * 37) % 120];
    for (auto* join : joins) {
      IREE_ASSERT_OK(iree_hal_streaming_graph_add_dependencies(
          state.graph(), &predecessor, &join, 1));
    }
  }
  for (size_t removed : {size_t{3}, size_t{11}}) {
    IREE_ASSERT_OK(
        iree_hal_streaming_graph_destroy_node(predecessors[removed]));
  }
  for (size_t i = 0; i < predecessors.size(); ++i) {
    if (i == 3 || i == 11) {
      continue;
    }
    for (auto* join : joins) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                            iree_hal_streaming_graph_add_dependencies(
                                state.graph(), &predecessors[i], &join, 1));
    }
  }
  for (size_t i = 0; i < predecessors.size(); i += 3) {
    if (i == 3) {
      continue;
    }
    for (auto* join : joins) {
      IREE_ASSERT_OK(iree_hal_streaming_graph_remove_dependencies(
          state.graph(), &predecessors[i], &join, 1));
      IREE_ASSERT_OK(iree_hal_streaming_graph_add_dependencies(
          state.graph(), &predecessors[i], &join, 1));
    }
  }

  iree_hal_streaming_graph_t* clone = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_clone(
      state.graph(), IREE_HAL_STREAMING_GRAPH_CLONE_MODE_TEMPLATE, &clone));
  const auto check = [&](iree_hal_streaming_graph_t* graph) {
    std::vector<iree_hal_streaming_graph_node_t*> nodes(graph->node_count);
    iree_hal_streaming_graph_get_nodes(graph, nodes.size(), nodes.data());
    for (size_t i = nodes.size() - joins.size(); i < nodes.size(); ++i) {
      auto* join = nodes[i];
      std::vector<uint32_t> source_ids;
      for (uint32_t j = 0; j < join->dependency_count; ++j) {
        source_ids.push_back(join->dependencies[j]->clone_source_node_index);
      }
      for (auto* edge = iree_hal_streaming_graph_additional_dependencies(join);
           edge; edge = edge->next) {
        source_ids.push_back(edge->from->clone_source_node_index);
      }
      std::sort(source_ids.begin(), source_ids.end());
      EXPECT_EQ(predecessors.size() - 2, source_ids.size());
      EXPECT_EQ(source_ids.end(),
                std::adjacent_find(source_ids.begin(), source_ids.end()));
      for (size_t j = 0; j < predecessors.size(); ++j) {
        EXPECT_EQ(j != 3 && j != 11,
                  std::binary_search(source_ids.begin(), source_ids.end(),
                                     predecessors[j]->clone_source_node_index));
      }
    }
  };
  check(state.graph());
  check(clone);
  iree_hal_streaming_graph_release(clone);
}

TEST(GraphDependenciesTest, BatchRollbackPreservesBothDestinations) {
  // Exercise promotion and preexisting indexes with interleaved growth.
  for (size_t initial_count : {size_t{8}, size_t{9}}) {
    for (int fail_after = 1;; ++fail_after) {
      SCOPED_TRACE(initial_count);
      SCOPED_TRACE(fail_after);
      ProbedAllocator allocator;
      bool reached_duplicate = false;
      {
        GraphState state(allocator.allocator());
        auto* graph = state.graph();
        std::array<iree_hal_streaming_graph_node_t*, 96> predecessors;
        for (auto& predecessor : predecessors) {
          IREE_ASSERT_OK(iree_hal_streaming_graph_add_empty_node(
              graph, nullptr, 0, &predecessor));
        }
        std::array<iree_hal_streaming_graph_node_t*, 2> joins;
        for (auto& join : joins) {
          IREE_ASSERT_OK(iree_hal_streaming_graph_add_empty_node(
              graph, predecessors.data(), 8, &join));
          if (initial_count == 9) {
            IREE_ASSERT_OK(iree_hal_streaming_graph_add_dependencies(
                graph, &predecessors[8], &join, 1));
          }
        }
        std::vector<iree_hal_streaming_graph_node_t*> from;
        std::vector<iree_hal_streaming_graph_node_t*> to;
        for (size_t i = initial_count; i < predecessors.size(); ++i) {
          for (auto* join : joins) {
            from.push_back(predecessors[i]);
            to.push_back(join);
          }
        }
        // Reject the final edge after earlier entries may have grown each
        // destination's table several times. No part of the batch may remain.
        from.back() = predecessors[0];
        const auto used_size = graph->arena.used_allocation_size;
        const auto total_size = graph->arena.total_allocation_size;
        const auto edge_count = graph->additional_edge_count;
        const auto balance = allocator.balance;
        allocator.fail_on_attempt = allocator.attempts + fail_after;
        iree_status_t status = iree_hal_streaming_graph_add_dependencies(
            graph, from.data(), to.data(), from.size());
        EXPECT_TRUE(iree_status_is_resource_exhausted(status) ||
                    iree_status_is_invalid_argument(status));
        reached_duplicate = iree_status_is_invalid_argument(status);
        iree_status_ignore(status);
        EXPECT_EQ(used_size, graph->arena.used_allocation_size);
        EXPECT_EQ(total_size, graph->arena.total_allocation_size);
        EXPECT_EQ(edge_count, graph->additional_edge_count);
        EXPECT_EQ(balance, allocator.balance);
        allocator.fail_on_attempt = 0;
        for (size_t i = 0; i < initial_count; ++i) {
          for (auto* join : joins) {
            IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                                  iree_hal_streaming_graph_add_dependencies(
                                      graph, &predecessors[i], &join, 1));
          }
        }
        from.back() = predecessors.back();
        IREE_ASSERT_OK(iree_hal_streaming_graph_add_dependencies(
            graph, from.data(), to.data(), from.size()));
        EXPECT_EQ(2 * (predecessors.size() - 8), graph->additional_edge_count);
        IREE_ASSERT_OK(iree_hal_streaming_graph_remove_dependencies(
            graph, from.data(), to.data(), from.size()));
        EXPECT_EQ(edge_count, graph->additional_edge_count);
      }
      EXPECT_EQ(0, allocator.balance);
      if (reached_duplicate) {
        break;
      }
    }
  }
}

TEST(GraphDependenciesTest, MembershipSupportsFinalSourceIdentities) {
  GraphState state;
  auto* graph = state.graph();
  graph->next_clone_source_node_index = UINT32_MAX - 17;
  iree_hal_streaming_graph_node_t* join = nullptr;
  IREE_ASSERT_OK(
      iree_hal_streaming_graph_add_empty_node(graph, nullptr, 0, &join));
  std::array<iree_hal_streaming_graph_node_t*, 16> predecessors;
  for (auto& predecessor : predecessors) {
    IREE_ASSERT_OK(iree_hal_streaming_graph_add_empty_node(graph, nullptr, 0,
                                                           &predecessor));
    IREE_ASSERT_OK(iree_hal_streaming_graph_add_dependencies(
        graph, &predecessor, &join, 1));
  }
  for (auto* predecessor : predecessors) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          iree_hal_streaming_graph_add_dependencies(
                              graph, &predecessor, &join, 1));
    IREE_ASSERT_OK(iree_hal_streaming_graph_remove_dependencies(
        graph, &predecessor, &join, 1));
  }
  EXPECT_EQ(0u, graph->additional_edge_count);
}

TEST(GraphDependenciesTest, EdgeStorageFailureReleasesPreparedIndex) {
  ProbedAllocator allocator;
  {
    GraphState state(allocator.allocator(), allocator.allocator());
    auto* graph = state.graph();
    std::array<iree_hal_streaming_graph_node_t*, 9> predecessors;
    for (auto& predecessor : predecessors) {
      IREE_ASSERT_OK(iree_hal_streaming_graph_add_empty_node(graph, nullptr, 0,
                                                             &predecessor));
    }
    iree_hal_streaming_graph_node_t* join = nullptr;
    IREE_ASSERT_OK(iree_hal_streaming_graph_add_empty_node(
        graph, predecessors.data(), 8, &join));
    // Consume the remaining block so this edge needs new arena backing after
    // preparing its membership index. Both allocations are real graph costs.
    void* padding = nullptr;
    IREE_ASSERT_OK(iree_arena_allocate(
        &graph->arena,
        graph->arena.block_bytes_remaining & ~(iree_max_align_t - 1),
        &padding));
    const auto used_size = graph->arena.used_allocation_size;
    const auto total_size = graph->arena.total_allocation_size;
    const auto balance = allocator.balance;
    allocator.fail_on_attempt = allocator.attempts + 2;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          iree_hal_streaming_graph_add_dependencies(
                              graph, &predecessors[8], &join, 1));
    EXPECT_EQ(balance, allocator.balance);
    EXPECT_EQ(used_size, graph->arena.used_allocation_size);
    EXPECT_EQ(total_size, graph->arena.total_allocation_size);
    EXPECT_EQ(0u, graph->additional_edge_count);
    allocator.fail_on_attempt = 0;
    IREE_ASSERT_OK(iree_hal_streaming_graph_add_dependencies(
        graph, &predecessors[8], &join, 1));
    std::array<iree_hal_streaming_graph_node_t*, 9> destinations;
    destinations.fill(join);
    IREE_ASSERT_OK(iree_hal_streaming_graph_remove_dependencies(
        graph, predecessors.data(), destinations.data(), predecessors.size()));
    EXPECT_EQ(0u, join->dependency_count);
    EXPECT_EQ(nullptr, iree_hal_streaming_graph_additional_dependencies(join));
  }
  EXPECT_EQ(0, allocator.balance);
}

}  // namespace
