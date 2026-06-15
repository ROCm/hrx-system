// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/scc.h"

#include <algorithm>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

struct TestGraph {
  const iree_host_size_t* const* successors;
  const iree_host_size_t* successor_counts;
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
        /*.node_count=*/node_count,
        /*.visit_successors=*/
        loom_scc_visit_successors_callback_make(VisitTestGraphSuccessors,
                                                const_cast<TestGraph*>(graph)),
    };
  }

  std::vector<iree_host_size_t> ComponentNodes(const loom_scc_t& component) {
    std::vector<iree_host_size_t> nodes(component.nodes,
                                        component.nodes + component.node_count);
    std::sort(nodes.begin(), nodes.end());
    return nodes;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(SccTest, AcyclicGraphIsSuccessorBeforePredecessor) {
  const iree_host_size_t successors0[] = {1};
  const iree_host_size_t successors1[] = {2};
  const iree_host_size_t* successors[] = {successors0, successors1, nullptr};
  const iree_host_size_t successor_counts[] = {1, 1, 0};
  TestGraph test_graph = {
      /*.successors=*/successors,
      /*.successor_counts=*/successor_counts,
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
      /*.successors=*/successors,
      /*.successor_counts=*/successor_counts,
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
      /*.successors=*/successors,
      /*.successor_counts=*/successor_counts,
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
      /*.successors=*/successors,
      /*.successor_counts=*/successor_counts,
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
      /*.successors=*/successors,
      /*.successor_counts=*/successor_counts,
  };
  const iree_host_size_t roots[] = {0};
  loom_scc_options_t options = {
      /*.root_nodes=*/roots,
      /*.root_count=*/IREE_ARRAYSIZE(roots),
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
      /*.successors=*/successors,
      /*.successor_counts=*/successor_counts,
  };
  const iree_host_size_t roots[] = {0};
  loom_scc_options_t options = {
      /*.root_nodes=*/roots,
      /*.root_count=*/0,
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
      /*.successors=*/successors,
      /*.successor_counts=*/successor_counts,
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
      /*.successors=*/successors,
      /*.successor_counts=*/successor_counts,
  };
  loom_scc_options_t options = {
      /*.root_nodes=*/nullptr,
      /*.root_count=*/1,
  };

  loom_scc_list_t sccs = {};
  loom_scc_graph_t graph = MakeGraph(1, &test_graph);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_scc_compute(&graph, &options, &arena_, &sccs));
}

}  // namespace
}  // namespace loom
