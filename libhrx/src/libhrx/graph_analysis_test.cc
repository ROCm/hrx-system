// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "graph_analysis.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using Edge = std::pair<uint32_t, uint32_t>;

class GraphAnalysisTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32768, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  iree_status_t Analyze(uint32_t count, const std::vector<Edge>& edges,
                        const std::vector<uint8_t>& kinds = {}) {
    iree_arena_reset(&arena_);
    hrx_graph_analysis_t analysis;
    IREE_RETURN_IF_ERROR(
        hrx_graph_analysis_initialize(count, &arena_, &analysis));
    for (uint32_t i = 0; i < count; ++i) {
      analysis.sources[i] = nullptr;
      analysis.flags[i] =
          kinds.empty() ? HRX_GRAPH_ANALYSIS_NODE_RECORDABLE : kinds[i];
    }
    for (auto [from, to] : edges) {
      ++analysis.offsets[to + 1];
    }
    IREE_RETURN_IF_ERROR(
        hrx_graph_analysis_allocate_dependencies(&analysis, &arena_));
    for (auto [from, to] : edges) {
      analysis.dependencies[analysis.order[to]++] = from;
    }
    return hrx_graph_analyze(&analysis, &arena_, &schedule_);
  }

  bool Barrier(uint32_t node) const {
    return schedule_.flags[node] & HRX_GRAPH_ANALYSIS_NODE_BARRIER_BEFORE;
  }

  // Check dependency order, independent recordings, and exact barrier cuts
  // against an unbounded set of operations since each preceding barrier.
  void Check(const std::vector<Edge>& edges) {
    const uint32_t count = schedule_.node_count;
    std::vector<uint32_t> ranks(count, UINT32_MAX);
    std::vector<uint32_t> partitions(count);
    std::vector<std::vector<uint32_t>> predecessors(count);
    for (uint32_t i = 0; i < count; ++i) {
      ASSERT_LT(schedule_.order[i], count);
      EXPECT_EQ(ranks[schedule_.order[i]], UINT32_MAX);
      ranks[schedule_.order[i]] = i;
    }
    uint32_t covered = 0;
    for (uint32_t p = 0; p < schedule_.partition_count; ++p) {
      const auto& partition = schedule_.partitions[p];
      ASSERT_EQ(partition.start_index, covered);
      covered += partition.count;
      ASSERT_LE(covered, count);
      for (uint32_t i = partition.start_index; i < covered; ++i) {
        partitions[schedule_.order[i]] = p;
      }
    }
    ASSERT_EQ(covered, count);
    for (auto [from, to] : edges) {
      EXPECT_LT(ranks[from], ranks[to]);
      if (partitions[from] == partitions[to]) {
        EXPECT_EQ(schedule_.flags[from] >> HRX_GRAPH_ANALYSIS_NODE_STREAM_SHIFT,
                  schedule_.flags[to] >> HRX_GRAPH_ANALYSIS_NODE_STREAM_SHIFT);
      }
      predecessors[to].push_back(from);
    }
    for (uint32_t p = 0; p < schedule_.partition_count; ++p) {
      const auto& partition = schedule_.partitions[p];
      const uint32_t first = schedule_.order[partition.start_index];
      if ((schedule_.flags[first] & HRX_GRAPH_ANALYSIS_NODE_KIND_MASK) !=
          HRX_GRAPH_ANALYSIS_NODE_RECORDABLE) {
        EXPECT_EQ(partition.count, 1u);
        EXPECT_FALSE(Barrier(first));
        continue;
      }
      std::array<std::vector<uint32_t>, HRX_GRAPH_MAX_WORKSTREAMS> pending;
      for (uint32_t i = partition.start_index;
           i < partition.start_index + partition.count; ++i) {
        const uint32_t node = schedule_.order[i];
        const uint32_t stream =
            schedule_.flags[node] >> HRX_GRAPH_ANALYSIS_NODE_STREAM_SHIFT;
        ASSERT_LT(stream, partition.stream_count);
        bool hazard = false;
        for (uint32_t predecessor : predecessors[node]) {
          hazard |= std::find(pending[stream].begin(), pending[stream].end(),
                              predecessor) != pending[stream].end();
        }
        EXPECT_EQ(Barrier(node), hazard) << "node " << node;
        if (hazard) {
          pending[stream].clear();
        }
        pending[stream].push_back(node);
      }
    }
  }

  // Reusable compilation storage, with no device or executable allocation.
  iree_arena_block_pool_t pool_;
  // Returned to pool_ between analyses.
  iree_arena_allocator_t arena_;
  // Borrowed result, valid until the next Analyze call.
  hrx_graph_schedule_t schedule_ = {};
};

TEST_F(GraphAnalysisTest, EmptyAndIndependentGraphs) {
  IREE_ASSERT_OK(Analyze(0, {}));
  EXPECT_EQ(schedule_.partition_count, 0u);
  IREE_ASSERT_OK(Analyze(4096, {}));
  Check({});
  for (uint32_t i = 0; i < schedule_.node_count; ++i) {
    EXPECT_EQ(schedule_.order[i], i);
    EXPECT_FALSE(Barrier(i));
  }
}

TEST_F(GraphAnalysisTest, BarriersCoverPriorProducersWithoutSetCapacityLimit) {
  // After 0->1 has been ordered, the many independent nodes do not make a
  // second dependency on 0 require another barrier.
  const std::vector<Edge> edges = {{0, 1}, {0, 14}, {13, 15}};
  IREE_ASSERT_OK(Analyze(16, edges));
  Check(edges);
  EXPECT_TRUE(Barrier(1));
  EXPECT_FALSE(Barrier(14));
  EXPECT_TRUE(Barrier(15));
}

TEST_F(GraphAnalysisTest, ReordersBranchesAndJoins) {
  const std::vector<Edge> edges = {{3, 2}, {3, 1}, {2, 0}, {1, 0}};
  IREE_ASSERT_OK(Analyze(4, edges));
  Check(edges);
  EXPECT_EQ(schedule_.order[0], 3u);
  EXPECT_EQ(schedule_.order[3], 0u);
  EXPECT_TRUE(Barrier(0));
}

TEST_F(GraphAnalysisTest, IsolatedAndEmptyNodesKeepDependencies) {
  const std::vector<uint8_t> kinds = {
      HRX_GRAPH_ANALYSIS_NODE_RECORDABLE, HRX_GRAPH_ANALYSIS_NODE_EMPTY,
      HRX_GRAPH_ANALYSIS_NODE_ISOLATED, HRX_GRAPH_ANALYSIS_NODE_RECORDABLE};
  const std::vector<Edge> edges = {{0, 1}, {1, 2}, {2, 3}};
  IREE_ASSERT_OK(Analyze(4, edges, kinds));
  Check(edges);
  EXPECT_EQ(schedule_.partition_count, 4u);
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(schedule_.flags[i] & HRX_GRAPH_ANALYSIS_NODE_KIND_MASK, kinds[i]);
    EXPECT_FALSE(Barrier(i));
  }
}

TEST_F(GraphAnalysisTest, CyclesAreRejected) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Analyze(3, {{0, 1}, {1, 2}, {2, 0}}));
  EXPECT_EQ(schedule_.partition_count, 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Analyze(1, {{0, 0}}));
}

TEST_F(GraphAnalysisTest, LargeFanInDoesNotTruncatePendingCounts) {
  std::vector<Edge> edges;
  for (uint32_t node = 1; node <= 65536; ++node) {
    edges.emplace_back(node, 0);
  }
  IREE_ASSERT_OK(Analyze(65537, edges));
  Check(edges);
  EXPECT_EQ(schedule_.order[65536], 0u);
}

TEST_F(GraphAnalysisTest, RepeatedEdgesRemainCorrect) {
  const std::vector<Edge> edges = {{2, 1}, {2, 1}, {1, 0}, {1, 0}};
  IREE_ASSERT_OK(Analyze(3, edges));
  Check(edges);
}

TEST_F(GraphAnalysisTest, DependencyCountOverflowRejectsBeforeAllocation) {
  hrx_graph_analysis_t analysis;
  IREE_ASSERT_OK(hrx_graph_analysis_initialize(2, &arena_, &analysis));
  analysis.offsets[1] = UINT32_MAX;
  analysis.offsets[2] = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      hrx_graph_analysis_allocate_dependencies(&analysis, &arena_));
}

TEST_F(GraphAnalysisTest, ExhaustiveSmallDags) {
  for (uint32_t mask = 0; mask < (1u << 15); ++mask) {
    std::vector<Edge> edges;
    uint32_t bit = 0;
    for (uint32_t to = 1; to < 6; ++to) {
      for (uint32_t from = 0; from < to; ++from, ++bit) {
        // Reverse ordinal order exercises the general sort for every edge.
        if (mask & (1u << bit)) {
          edges.emplace_back(5 - from, 5 - to);
        }
      }
    }
    IREE_ASSERT_OK(Analyze(6, edges));
    Check(edges);
  }
}

TEST_F(GraphAnalysisTest, WorkstreamsAndPartitionBoundaries) {
  std::mt19937 random(0x251112);
  for (uint32_t count : {17u, 64u, 257u, 4096u}) {
    for (uint32_t seed = 0; seed < 16; ++seed) {
      std::vector<uint32_t> order(count);
      std::iota(order.begin(), order.end(), 0);
      if (seed & 1) {
        std::shuffle(order.begin(), order.end(), random);
      }
      std::vector<Edge> edges;
      for (uint32_t to = 1; to < count; ++to) {
        // Mix independent roots, chains, and joins without quadratic fixtures.
        if (random() % 3) {
          edges.emplace_back(order[random() % to], order[to]);
        }
        if (random() % 3) {
          edges.emplace_back(order[random() % to], order[to]);
        }
      }
      std::vector<uint8_t> kinds(count, HRX_GRAPH_ANALYSIS_NODE_RECORDABLE);
      if (seed & 2) {
        for (uint32_t i = 31; i < count; i += 32) {
          kinds[i] = HRX_GRAPH_ANALYSIS_NODE_ISOLATED;
        }
      }
      IREE_ASSERT_OK(Analyze(count, edges, kinds));
      Check(edges);
    }
  }
}

}  // namespace
