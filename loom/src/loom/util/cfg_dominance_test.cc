// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/cfg_dominance.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

// Graph-data fixture for the single-graph API. Real IR extraction and verifier
// integration are exercised by the CFG and verification loom-test suites.
class Graph {
 public:
  explicit Graph(std::vector<std::vector<uint16_t>> successors)
      : successors_(std::move(successors)), blocks_(successors_.size()) {
    std::vector<std::vector<uint16_t>> predecessors(blocks_.size());
    for (size_t i = 0; i < blocks_.size(); ++i) {
      blocks_[i].preorder = UINT16_MAX;
      blocks_[i].parent = UINT16_MAX;
      blocks_[i].successor_start = successor_indices_.size();
      blocks_[i].successor_count = successors_[i].size();
      for (uint16_t target : successors_[i]) {
        successor_indices_.push_back(target);
        predecessors[target].push_back(i);
      }
    }
    for (size_t i = 0; i < blocks_.size(); ++i) {
      blocks_[i].predecessor_start = predecessor_indices_.size();
      blocks_[i].predecessor_count = predecessors[i].size();
      predecessor_indices_.insert(predecessor_indices_.end(),
                                  predecessors[i].begin(),
                                  predecessors[i].end());
    }
    if (!blocks_.empty()) {
      std::vector<std::pair<uint16_t, size_t>> stack{{0, 0}};
      uint16_t preorder = 0;
      blocks_[0].reachable = true;
      blocks_[0].preorder = preorder++;
      while (!stack.empty()) {
        auto& frame = stack.back();
        if (frame.second == successors_[frame.first].size()) {
          reverse_postorder_.push_back(frame.first);
          stack.pop_back();
        } else {
          uint16_t child = successors_[frame.first][frame.second++];
          if (!blocks_[child].reachable) {
            blocks_[child].reachable = true;
            blocks_[child].preorder = preorder++;
            blocks_[child].parent = frame.first;
            stack.emplace_back(child, 0);
          }
        }
      }
      std::reverse(reverse_postorder_.begin(), reverse_postorder_.end());
    }
    graph_.blocks = blocks_.data();
    graph_.block_count = blocks_.size();
    graph_.edge_count = successor_indices_.size();
    graph_.successor_indices = successor_indices_.data();
    graph_.predecessor_indices = predecessor_indices_.data();
    graph_.reverse_postorder = {reverse_postorder_.data(),
                                reverse_postorder_.size()};
  }

  const loom_cfg_graph_t* get() const { return &graph_; }

 private:
  // Successor adjacency supplied by the test.
  std::vector<std::vector<uint16_t>> successors_;
  // Adjacency ranges and DFS facts consumed by the analysis.
  std::vector<loom_cfg_block_info_t> blocks_;
  // Dense outgoing adjacency.
  std::vector<uint16_t> successor_indices_;
  // Dense incoming adjacency.
  std::vector<uint16_t> predecessor_indices_;
  // Reachable blocks in reverse DFS postorder.
  std::vector<uint16_t> reverse_postorder_;
  // Non-owning view of the fixture's retained arrays.
  loom_cfg_graph_t graph_ = {};
};

class CfgDominanceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Independent small-graph oracle: removing a dominator disconnects all of
  // its descendants from entry. This deliberately does not use semidominators,
  // graph preorder, or iterative idom intersections.
  void CheckAgainstReachability(const Graph& fixture) {
    const auto* graph = fixture.get();
    const size_t count = graph->block_count;
    loom_cfg_dominance_t dominance;
    IREE_ASSERT_OK(loom_cfg_dominance_build(graph, &arena_, &dominance));
    ASSERT_TRUE(dominance.available);
    ASSERT_EQ(dominance.preorder.count, graph->reverse_postorder.count);
    std::vector<std::vector<bool>> expected(count, std::vector<bool>(count));
    std::vector<size_t> depth(count);
    for (size_t removed = 0; removed < count; ++removed) {
      std::vector<bool> reachable(count);
      std::vector<uint16_t> pending;
      if (removed != 0) {
        pending.push_back(0);
      }
      while (!pending.empty()) {
        uint16_t current = pending.back();
        pending.pop_back();
        if (current == removed || reachable[current]) {
          continue;
        }
        reachable[current] = true;
        auto successors = loom_cfg_graph_successors(graph, current);
        for (size_t i = 0; i < successors.count; ++i) {
          pending.push_back(successors.values[i]);
        }
      }
      for (size_t block = 0; block < count; ++block) {
        expected[removed][block] =
            removed == block ||
            (graph->blocks[block].reachable && !reachable[block]);
        depth[block] += expected[removed][block];
        EXPECT_EQ(
            loom_cfg_dominance_block_dominates(&dominance, removed, block),
            expected[removed][block])
            << removed << " dominates " << block;
      }
    }
    for (size_t block = 0; block < count; ++block) {
      uint16_t parent = LOOM_CFG_DOMINATOR_INVALID;
      if (block == 0) {
        parent = 0;
      } else if (graph->blocks[block].reachable) {
        for (uint16_t ancestor = 0; ancestor < count; ++ancestor) {
          if (ancestor == block || !expected[ancestor][block]) {
            continue;
          }
          if (parent == LOOM_CFG_DOMINATOR_INVALID ||
              depth[ancestor] > depth[parent]) {
            parent = ancestor;
          }
        }
      }
      EXPECT_EQ(dominance.immediate_dominators[block], parent) << block;
    }
    // Removing all parallel edges of one source->target alternative checks
    // mandatory entry independently of the dominator-tree classification.
    for (uint16_t source = 0; source < count; ++source) {
      auto alternatives = loom_cfg_graph_successors(graph, source);
      for (size_t i = 0; i < alternatives.count; ++i) {
        const uint16_t target = alternatives.values[i];
        std::vector<bool> reachable(count);
        std::vector<uint16_t> pending{0};
        while (!pending.empty()) {
          const uint16_t current = pending.back();
          pending.pop_back();
          if (reachable[current]) {
            continue;
          }
          reachable[current] = true;
          auto successors = loom_cfg_graph_successors(graph, current);
          for (size_t j = 0; j < successors.count; ++j) {
            if (current != source || successors.values[j] != target) {
              pending.push_back(successors.values[j]);
            }
          }
        }
        for (uint16_t block = 0; block < count; ++block) {
          EXPECT_EQ(
              dominance.entry_predecessors[target] == source &&
                  loom_cfg_dominance_block_dominates(&dominance, target, block),
              graph->blocks[block].reachable && !reachable[block])
              << source << " -> " << target << " dominates " << block;
        }
      }
    }
    std::vector<bool> seen(count);
    for (size_t i = 0; i < dominance.preorder.count; ++i) {
      uint16_t block = dominance.preorder.values[i];
      ASSERT_LT(block, count);
      EXPECT_FALSE(seen[block]);
      seen[block] = true;
      EXPECT_EQ(static_cast<uint16_t>(dominance.intervals[block]), i);
      if (block != 0) {
        EXPECT_TRUE(seen[dominance.immediate_dominators[block]]);
      }
    }
    iree_arena_reset(&arena_);
  }

  // Reusable allocation pool for analysis invocations.
  iree_arena_block_pool_t pool_;
  // Invocation-owned analysis arrays.
  iree_arena_allocator_t arena_;
};

TEST_F(CfgDominanceTest, EmptyAndSingleBlock) {
  CheckAgainstReachability(Graph({}));
  CheckAgainstReachability(Graph({{}}));
  CheckAgainstReachability(Graph({{0}}));
}

TEST_F(CfgDominanceTest, ReverseChainAndDiamond) {
  CheckAgainstReachability(Graph({{3}, {}, {1}, {2}}));
  CheckAgainstReachability(Graph({{1, 2}, {3}, {3}, {}}));
}

TEST_F(CfgDominanceTest, LoopsDuplicateEdgesAndUnreachablePredecessors) {
  CheckAgainstReachability(Graph({{1, 1}, {2, 3}, {1}, {}, {3, 4}}));
  CheckAgainstReachability(Graph({{1, 2}, {2, 3}, {1, 3}, {}}));
  CheckAgainstReachability(Graph({{}, {2}, {1}}));
}

TEST_F(CfgDominanceTest, EveryThreeBlockGraph) {
  for (uint32_t mask = 0; mask < 512; ++mask) {
    SCOPED_TRACE(mask);
    std::vector<std::vector<uint16_t>> edges(3);
    for (uint16_t source = 0; source < 3; ++source) {
      for (uint16_t target = 0; target < 3; ++target) {
        if (mask & (1u << (source * 3 + target))) {
          edges[source].push_back(target);
        }
      }
    }
    CheckAgainstReachability(Graph(std::move(edges)));
  }
}

TEST_F(CfgDominanceTest, ArbitraryGraphsAgainstReachability) {
  uint32_t seed = 1777;
  auto random = [&]() { return seed = seed * 1664525u + 1013904223u; };
  for (int iteration = 0; iteration < 4000; ++iteration) {
    SCOPED_TRACE(iteration);
    const size_t count = 1 + random() % 24;
    std::vector<std::vector<uint16_t>> edges(count);
    for (auto& successors : edges) {
      const size_t successor_count = random() % 5;
      for (size_t i = 0; i < successor_count; ++i) {
        successors.push_back(random() % count);
      }
    }
    CheckAgainstReachability(Graph(std::move(edges)));
  }
}

TEST_F(CfgDominanceTest, FullBlockIndexRangeWithoutRecursiveStack) {
  constexpr size_t count = UINT16_MAX;
  std::vector<std::vector<uint16_t>> edges(count);
  edges[0].push_back(count - 1);
  for (size_t i = count - 1; i > 1; --i) {
    edges[i].push_back(i - 1);
  }
  Graph fixture(std::move(edges));
  loom_cfg_dominance_t dominance;
  IREE_ASSERT_OK(loom_cfg_dominance_build(fixture.get(), &arena_, &dominance));
  ASSERT_TRUE(dominance.available);
  ASSERT_EQ(dominance.preorder.count, count);
  EXPECT_EQ(dominance.immediate_dominators[count - 1], 0);
  EXPECT_EQ(dominance.entry_predecessors[count - 1], 0);
  EXPECT_EQ(dominance.entry_predecessors[0], LOOM_CFG_DOMINATOR_INVALID);
  for (size_t i = 1; i + 1 < count; ++i) {
    EXPECT_EQ(dominance.immediate_dominators[i], i + 1);
    EXPECT_EQ(dominance.entry_predecessors[i], i + 1);
  }
  EXPECT_EQ(dominance.intervals[0], (count - 1) << 16);
  EXPECT_EQ(dominance.preorder.values[count - 1], 1);
}

TEST_F(CfgDominanceTest, MalformedGraphIsUnavailable) {
  loom_cfg_graph_t graph = {.malformed = true};
  loom_cfg_dominance_t dominance;
  IREE_ASSERT_OK(loom_cfg_dominance_build(&graph, &arena_, &dominance));
  EXPECT_FALSE(dominance.available);
  EXPECT_EQ(dominance.preorder.count, 0u);
  EXPECT_EQ(dominance.immediate_dominators, nullptr);
  EXPECT_EQ(dominance.intervals, nullptr);
  EXPECT_EQ(dominance.entry_predecessors, nullptr);
}

}  // namespace
}  // namespace loom
