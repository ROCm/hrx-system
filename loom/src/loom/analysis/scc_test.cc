// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/scc.h"

#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

struct TestGraph {
  // Borrowed outgoing adjacency arrays.
  const iree_host_size_t* const* successors;
  // Number of entries in each outgoing array.
  const iree_host_size_t* successor_counts;
};

class GraphFixture {
 public:
  explicit GraphFixture(std::vector<std::vector<iree_host_size_t>> successors)
      : successors(std::move(successors)),
        visits(this->successors.size()),
        payloads(this->successors.size()) {}

  loom_scc_graph_t graph() {
    return {successors.size(),
            loom_scc_visit_successors_callback_make(Visit, this)};
  }

  // Caller-owned graph adjacency.
  std::vector<std::vector<iree_host_size_t>> successors;
  // Number of callback enumerations per node.
  std::vector<uint32_t> visits;
  // Optional reached node whose callback injects an allocation failure.
  iree_host_size_t failure_node = IREE_HOST_SIZE_MAX;
  // Optional caller arena receiving retained adapter-owned data.
  iree_arena_allocator_t* arena = nullptr;
  // Adapter-owned payload for each enumerated node.
  std::vector<iree_host_size_t*> payloads;
  // Detects unwanted recursive enumeration of the adapter.
  bool enumerating = false;

 private:
  static iree_status_t Visit(void* user_data, iree_host_size_t node,
                             loom_scc_successor_callback_t successor) {
    auto* fixture = static_cast<GraphFixture*>(user_data);
    EXPECT_FALSE(fixture->enumerating);
    fixture->enumerating = true;
    ++fixture->visits[node];
    if (fixture->arena) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          fixture->arena, sizeof(iree_host_size_t),
          reinterpret_cast<void**>(&fixture->payloads[node])));
      *fixture->payloads[node] = node;
    }
    for (auto target : fixture->successors[node]) {
      IREE_RETURN_IF_ERROR(successor.fn(successor.user_data, target));
    }
    fixture->enumerating = false;
    if (node == fixture->failure_node) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "graph adapter allocation failure");
    }
    return iree_ok_status();
  }
};

static iree_status_t VisitTestGraphSuccessors(
    void* user_data, iree_host_size_t node,
    loom_scc_successor_callback_t successor) {
  const TestGraph* graph = (const TestGraph*)user_data;
  for (iree_host_size_t i = 0; i < graph->successor_counts[node]; ++i) {
    IREE_RETURN_IF_ERROR(
        successor.fn(successor.user_data, graph->successors[node][i]));
  }
  return iree_ok_status();
}

class SccTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_scc_graph_t MakeGraph(iree_host_size_t node_count,
                             const TestGraph* graph) {
    return {
        .node_count = node_count,
        .visit_successors = loom_scc_visit_successors_callback_make(
            VisitTestGraphSuccessors, const_cast<TestGraph*>(graph)),
    };
  }

  std::vector<iree_host_size_t> ComponentNodes(const loom_scc_t& component) {
    std::vector<iree_host_size_t> nodes(component.nodes,
                                        component.nodes + component.node_count);
    std::sort(nodes.begin(), nodes.end());
    return nodes;
  }

  void CheckReachability(GraphFixture& fixture,
                         const loom_scc_options_t* options = nullptr) {
    auto graph = fixture.graph();
    const size_t count = graph.node_count;
    std::vector<std::vector<bool>> reaches(count, std::vector<bool>(count));
    for (size_t source = 0; source < count; ++source) {
      std::vector<size_t> pending{source};
      while (!pending.empty()) {
        size_t node = pending.back();
        pending.pop_back();
        if (reaches[source][node]) {
          continue;
        }
        reaches[source][node] = true;
        pending.insert(pending.end(), fixture.successors[node].begin(),
                       fixture.successors[node].end());
      }
    }
    std::vector<bool> included(count, !options || !options->root_nodes);
    if (options && options->root_nodes) {
      for (size_t i = 0; i < options->root_count; ++i) {
        for (size_t node = 0; node < count; ++node) {
          included[node] =
              included[node] || reaches[options->root_nodes[i]][node];
        }
      }
    }
    loom_scc_list_t components;
    IREE_ASSERT_OK(loom_scc_compute(&graph, options, &arena_, &components));
    std::vector<size_t> mapping(count, IREE_HOST_SIZE_MAX);
    for (size_t i = 0; i < components.count; ++i) {
      const auto& component = components.values[i];
      ASSERT_GT(component.node_count, 0u);
      bool cyclic = component.node_count > 1;
      for (size_t j = 0; j < component.node_count; ++j) {
        size_t node = component.nodes[j];
        ASSERT_LT(node, count);
        EXPECT_EQ(mapping[node], IREE_HOST_SIZE_MAX);
        mapping[node] = i;
        cyclic |= std::find(fixture.successors[node].begin(),
                            fixture.successors[node].end(),
                            node) != fixture.successors[node].end();
      }
      EXPECT_EQ(component.is_cycle, cyclic);
    }
    for (size_t source = 0; source < count; ++source) {
      EXPECT_EQ(mapping[source] != IREE_HOST_SIZE_MAX, included[source]);
      EXPECT_EQ(fixture.visits[source], included[source] ? 1u : 0u);
      if (!included[source]) {
        continue;
      }
      for (size_t target = 0; target < count; ++target) {
        if (!included[target]) {
          continue;
        }
        EXPECT_EQ(mapping[source] == mapping[target],
                  reaches[source][target] && reaches[target][source]);
      }
      for (size_t target : fixture.successors[source]) {
        EXPECT_GE(mapping[source], mapping[target]);
      }
    }
    // Only the two result arrays survive; traversal scratch is not retained
    // with every inferred loop or symbol component in a long-lived fact scope.
    EXPECT_LE(arena_.used_allocation_size,
              count * (sizeof(loom_scc_t) + sizeof(iree_host_size_t)) + 64);
    iree_arena_reset(&arena_);
  }

  // Reusable backing storage for analysis allocations.
  iree_arena_block_pool_t block_pool_;
  // Result lifetime, reset between independent graph cases.
  iree_arena_allocator_t arena_;
};

TEST_F(SccTest, AcyclicGraphIsSuccessorBeforePredecessor) {
  const iree_host_size_t successors0[] = {1};
  const iree_host_size_t successors1[] = {2};
  const iree_host_size_t* successors[] = {successors0, successors1, nullptr};
  const iree_host_size_t successor_counts[] = {1, 1, 0};
  TestGraph test_graph = {
      .successors = successors,
      .successor_counts = successor_counts,
  };

  loom_scc_list_t sccs = {};
  loom_scc_graph_t graph = MakeGraph(3, &test_graph);
  IREE_ASSERT_OK(loom_scc_compute(&graph, nullptr, &arena_, &sccs));

  ASSERT_EQ(sccs.count, 3u);
  EXPECT_EQ(sccs.values[0].nodes[0], 2u);
  EXPECT_EQ(sccs.values[1].nodes[0], 1u);
  EXPECT_EQ(sccs.values[2].nodes[0], 0u);
  EXPECT_FALSE(sccs.values[0].is_cycle);
  EXPECT_FALSE(sccs.values[1].is_cycle);
  EXPECT_FALSE(sccs.values[2].is_cycle);
}

TEST_F(SccTest, SelfRecursionIsCycle) {
  const iree_host_size_t successors0[] = {0};
  const iree_host_size_t* successors[] = {successors0};
  const iree_host_size_t successor_counts[] = {1};
  TestGraph test_graph = {
      .successors = successors,
      .successor_counts = successor_counts,
  };

  loom_scc_list_t sccs = {};
  loom_scc_graph_t graph = MakeGraph(1, &test_graph);
  IREE_ASSERT_OK(loom_scc_compute(&graph, nullptr, &arena_, &sccs));

  ASSERT_EQ(sccs.count, 1u);
  EXPECT_EQ(sccs.values[0].node_count, 1u);
  EXPECT_EQ(sccs.values[0].nodes[0], 0u);
  EXPECT_TRUE(sccs.values[0].is_cycle);
}

TEST_F(SccTest, MultiNodeCycleIsOneComponent) {
  const iree_host_size_t successors0[] = {1};
  const iree_host_size_t successors1[] = {2};
  const iree_host_size_t successors2[] = {0};
  const iree_host_size_t* successors[] = {successors0, successors1,
                                          successors2};
  const iree_host_size_t successor_counts[] = {1, 1, 1};
  TestGraph test_graph = {
      .successors = successors,
      .successor_counts = successor_counts,
  };

  loom_scc_list_t sccs = {};
  loom_scc_graph_t graph = MakeGraph(3, &test_graph);
  IREE_ASSERT_OK(loom_scc_compute(&graph, nullptr, &arena_, &sccs));

  ASSERT_EQ(sccs.count, 1u);
  EXPECT_EQ(sccs.values[0].node_count, 3u);
  EXPECT_EQ(ComponentNodes(sccs.values[0]),
            (std::vector<iree_host_size_t>{0, 1, 2}));
  EXPECT_TRUE(sccs.values[0].is_cycle);
}

TEST_F(SccTest, DisconnectedNodesAppearInWholeGraphMode) {
  const iree_host_size_t successors0[] = {1};
  const iree_host_size_t* successors[] = {successors0, nullptr, nullptr};
  const iree_host_size_t successor_counts[] = {1, 0, 0};
  TestGraph test_graph = {
      .successors = successors,
      .successor_counts = successor_counts,
  };

  loom_scc_list_t sccs = {};
  loom_scc_graph_t graph = MakeGraph(3, &test_graph);
  IREE_ASSERT_OK(loom_scc_compute(&graph, nullptr, &arena_, &sccs));

  ASSERT_EQ(sccs.count, 3u);
  EXPECT_EQ(sccs.values[0].nodes[0], 1u);
  EXPECT_EQ(sccs.values[1].nodes[0], 0u);
  EXPECT_EQ(sccs.values[2].nodes[0], 2u);
}

TEST_F(SccTest, RootFilteredModeSkipsUnreachableNodes) {
  const iree_host_size_t successors0[] = {1};
  const iree_host_size_t* successors[] = {successors0, nullptr, nullptr};
  const iree_host_size_t successor_counts[] = {1, 0, 0};
  TestGraph test_graph = {
      .successors = successors,
      .successor_counts = successor_counts,
  };
  const iree_host_size_t roots[] = {0};
  loom_scc_options_t options = {
      .root_nodes = roots,
      .root_count = IREE_ARRAYSIZE(roots),
  };

  loom_scc_list_t sccs = {};
  loom_scc_graph_t graph = MakeGraph(3, &test_graph);
  IREE_ASSERT_OK(loom_scc_compute(&graph, &options, &arena_, &sccs));

  ASSERT_EQ(sccs.count, 2u);
  EXPECT_EQ(sccs.values[0].nodes[0], 1u);
  EXPECT_EQ(sccs.values[1].nodes[0], 0u);
}

TEST_F(SccTest, EmptyRootListProducesEmptyResult) {
  const iree_host_size_t* successors[] = {nullptr};
  const iree_host_size_t successor_counts[] = {0};
  TestGraph test_graph = {
      .successors = successors,
      .successor_counts = successor_counts,
  };
  const iree_host_size_t roots[] = {0};
  loom_scc_options_t options = {
      .root_nodes = roots,
      .root_count = 0,
  };

  loom_scc_list_t sccs = {};
  loom_scc_graph_t graph = MakeGraph(1, &test_graph);
  IREE_ASSERT_OK(loom_scc_compute(&graph, &options, &arena_, &sccs));

  EXPECT_EQ(sccs.count, 0u);
}

TEST_F(SccTest, InvalidSuccessorReportsApiError) {
  const iree_host_size_t successors0[] = {4};
  const iree_host_size_t* successors[] = {successors0};
  const iree_host_size_t successor_counts[] = {1};
  TestGraph test_graph = {
      .successors = successors,
      .successor_counts = successor_counts,
  };

  loom_scc_list_t sccs = {};
  loom_scc_graph_t graph = MakeGraph(1, &test_graph);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_scc_compute(&graph, nullptr, &arena_, &sccs));
}

TEST_F(SccTest, NonZeroRootCountRequiresRootArray) {
  const iree_host_size_t* successors[] = {nullptr};
  const iree_host_size_t successor_counts[] = {0};
  TestGraph test_graph = {
      .successors = successors,
      .successor_counts = successor_counts,
  };
  loom_scc_options_t options = {
      .root_nodes = nullptr,
      .root_count = 1,
  };

  loom_scc_list_t sccs = {};
  loom_scc_graph_t graph = MakeGraph(1, &test_graph);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_scc_compute(&graph, &options, &arena_, &sccs));
}

TEST_F(SccTest, ExhaustiveSmallGraphsMatchMutualReachability) {
  for (uint32_t mask = 0; mask < 512; ++mask) {
    SCOPED_TRACE(mask);
    std::vector<std::vector<size_t>> successors(3);
    for (size_t source = 0; source < 3; ++source) {
      for (size_t target = 0; target < 3; ++target) {
        if (mask & (1u << (source * 3 + target))) {
          successors[source].push_back(target);
        }
      }
    }
    GraphFixture fixture(std::move(successors));
    CheckReachability(fixture);
  }
}

TEST_F(SccTest, RandomGraphsAndRootFiltersMatchMutualReachability) {
  std::mt19937 random(261149);
  for (uint32_t trial = 0; trial < 3000; ++trial) {
    SCOPED_TRACE(trial);
    const size_t count = 1 + random() % 24;
    std::vector<std::vector<size_t>> successors(count);
    for (auto& edges : successors) {
      uint32_t edge_count = random() % 8;
      while (edge_count--) {
        edges.push_back(random() % count);
      }
    }
    GraphFixture fixture(std::move(successors));
    const size_t roots[] = {random() % count, random() % count, 0};
    const loom_scc_options_t options = {roots, random() % 4};
    CheckReachability(fixture, trial % 2 ? &options : nullptr);
  }
}

TEST_F(SccTest, DeepChainAndCycleUseConstantNativeStack) {
  constexpr size_t kCount = 100000;
  for (bool cyclic : {false, true}) {
    std::vector<std::vector<size_t>> successors(kCount);
    for (size_t i = 0; i + 1 < kCount; ++i) {
      successors[i] = {i + 1};
    }
    if (cyclic) {
      successors.back() = {0};
    }
    GraphFixture fixture(std::move(successors));
    auto graph = fixture.graph();
    loom_scc_list_t components;
    IREE_ASSERT_OK(loom_scc_compute(&graph, nullptr, &arena_, &components));
    ASSERT_EQ(components.count, cyclic ? 1u : kCount);
    for (size_t i = 0; i < components.count; ++i) {
      EXPECT_EQ(components.values[i].is_cycle, cyclic);
      EXPECT_EQ(components.values[i].node_count, cyclic ? kCount : 1u);
      if (!cyclic) {
        EXPECT_EQ(components.values[i].nodes[0], kCount - i - 1);
      }
    }
    for (uint32_t visits : fixture.visits) {
      EXPECT_EQ(visits, 1u);
    }
    iree_arena_reset(&arena_);
  }
}

TEST_F(SccTest, DenseActiveFramesPreserveEverySuccessor) {
  constexpr size_t kCount = 128;
  std::vector<std::vector<size_t>> successors(kCount);
  for (size_t i = 0; i < kCount; ++i) {
    for (size_t j = i + 1; j < kCount; ++j) {
      successors[i].push_back(j);
    }
  }
  GraphFixture fixture(std::move(successors));
  CheckReachability(fixture);
}

TEST_F(SccTest, CallbackFailureReclaimsTraversalScratch) {
  GraphFixture fixture({{1, 2}, {2, 3}, {3}, {0}});
  fixture.failure_node = 2;
  auto graph = fixture.graph();
  loom_scc_list_t components;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_scc_compute(&graph, nullptr, &arena_, &components));
  EXPECT_EQ(components.count, 0u);
  EXPECT_EQ(components.values, nullptr);
  EXPECT_EQ(fixture.visits, (std::vector<uint32_t>{1, 1, 1, 0}));
  EXPECT_LE(
      arena_.used_allocation_size,
      graph.node_count * (sizeof(loom_scc_t) + sizeof(iree_host_size_t)) + 64);
}

TEST_F(SccTest, InvalidRootAndEmptyGraph) {
  GraphFixture fixture({{}});
  auto graph = fixture.graph();
  const size_t root = 1;
  const loom_scc_options_t options = {&root, 1};
  loom_scc_list_t components;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_scc_compute(&graph, &options, &arena_, &components));
  EXPECT_EQ(components.count, 0u);
  const loom_scc_graph_t empty = {};
  IREE_ASSERT_OK(loom_scc_compute(&empty, nullptr, &arena_, &components));
  EXPECT_EQ(components.count, 0u);
}

TEST_F(SccTest, AdapterAllocationsRetainTheirCallerOwnedLifetime) {
  GraphFixture fixture({{1, 2}, {3}, {3}, {}});
  fixture.arena = &arena_;
  auto graph = fixture.graph();
  loom_scc_list_t components;
  IREE_ASSERT_OK(loom_scc_compute(&graph, nullptr, &arena_, &components));
  void* overwrite = nullptr;
  IREE_ASSERT_OK(iree_arena_allocate(&arena_, 4096, &overwrite));
  memset(overwrite, 0xA5, 4096);
  for (size_t node = 0; node < graph.node_count; ++node) {
    ASSERT_NE(fixture.payloads[node], nullptr);
    EXPECT_EQ(*fixture.payloads[node], node);
  }
  EXPECT_EQ(components.count, 4u);
}

}  // namespace
}  // namespace loom
