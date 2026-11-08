// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/cfg_loop_nest.h"

#include <algorithm>
#include <random>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/util/cfg_graph_test_util.h"
#include "loom/util/cfg_loop_nest_test_util.h"

namespace loom {
namespace {

using testing::CfgGraph;

class CfgLoopNestTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_cfg_loop_nest_t Build(const CfgGraph& fixture) {
    iree_arena_reset(&arena_);
    loom_cfg_dominance_t dominance;
    IREE_CHECK_OK(loom_cfg_dominance_build(fixture.get(), &arena_, &dominance));
    loom_cfg_loop_nest_t nest;
    IREE_CHECK_OK(
        loom_cfg_loop_nest_build(fixture.get(), &dominance, &arena_, &nest));
    return nest;
  }

  void CheckOracle(const std::vector<std::vector<uint16_t>>& successors) {
    CfgGraph fixture(successors);
    const auto nest = Build(fixture);
    IREE_ASSERT_OK(testing::CheckLoopNest(nest));
    if (!nest.reducible) {
      return;
    }
    const auto* graph = fixture.get();
    std::vector<uint16_t> order(graph->reverse_postorder.count, UINT16_MAX);
    IREE_ASSERT_OK(
        loom_cfg_loop_nest_order_blocks(&nest, &arena_, order.data()));
    std::vector<uint16_t> positions(successors.size(), UINT16_MAX);
    for (size_t i = 0; i < order.size(); ++i) {
      ASSERT_LT(order[i], successors.size());
      ASSERT_EQ(positions[order[i]], UINT16_MAX);
      ASSERT_TRUE(graph->blocks[order[i]].reachable);
      positions[order[i]] = i;
    }
    for (uint16_t block : order) {
      for (uint16_t target : successors[block]) {
        const uint16_t loop = loom_cfg_loop_nest_innermost(&nest, target);
        if (loop != LOOM_CFG_LOOP_NEST_NONE &&
            nest.loops[loop].header_index == target &&
            loom_cfg_loop_nest_contains(&nest, loop, block)) {
          continue;
        }
        EXPECT_LT(positions[block], positions[target]);
      }
    }
    for (uint16_t loop = 0; loop < nest.loop_count; ++loop) {
      const size_t begin = positions[nest.loops[loop].header_index];
      size_t count = 0;
      for (uint16_t block : order) {
        count += loom_cfg_loop_nest_contains(&nest, loop, block);
      }
      ASSERT_LE(begin + count, order.size());
      for (size_t i = begin; i < begin + count; ++i) {
        EXPECT_TRUE(loom_cfg_loop_nest_contains(&nest, loop, order[i]));
      }
    }
  }

  // Pool shared by successive independent immutable snapshots.
  iree_arena_block_pool_t pool_;
  // Retained storage for the active test snapshot.
  iree_arena_allocator_t arena_;
};

TEST_F(CfgLoopNestTest, EmptyAcyclicAndUnreachableCycles) {
  CheckOracle({});
  CheckOracle({{}});
  CheckOracle({{3}, {}, {1}, {2}});
  CheckOracle({{1}, {}, {2}});
}

TEST_F(CfgLoopNestTest, DelayedGuardAndReorderedBlocks) {
  // Entry -> header diamond -> guard -> latch -> header; guard also exits.
  CheckOracle({{1}, {2, 3}, {4}, {4}, {5, 6}, {1}, {}});
  CheckOracle({{4}, {3}, {}, {6, 2}, {1, 5}, {3}, {4}});
}

TEST_F(CfgLoopNestTest, ParallelEdgesMultipleEntriesLatchesAndExits) {
  CheckOracle({{1, 1}, {2, 3}, {1}, {1, 4}, {}});
  CheckOracle({{1}, {2, 5}, {3, 4}, {1, 5}, {1, 6}, {}, {}});
  CheckOracle({{1}, {1, 2}, {}});
  CheckOracle({{0}});
}

TEST_F(CfgLoopNestTest, NestedSiblingAndSharedExits) {
  CheckOracle({{1}, {2, 6}, {3, 4}, {2}, {5}, {1}, {7}, {7, 8}, {}});
  CheckOracle({{1}, {2}, {3, 6}, {2, 4}, {1, 5}, {6}, {}});
}

TEST_F(CfgLoopNestTest, CommonAndNestedExitOwnership) {
  CfgGraph shared({{1}, {2, 4}, {3, 4}, {1}, {}});
  const auto shared_nest = Build(shared);
  ASSERT_EQ(shared_nest.loop_count, 1u);
  EXPECT_EQ(shared_nest.loops[0].exits.count, 2u);
  EXPECT_EQ(shared_nest.loops[0].direct_exit_count, 2u);
  EXPECT_EQ(shared_nest.loops[0].continuation_index, 4u);
  IREE_ASSERT_OK(testing::CheckLoopNest(shared_nest));

  // The inner header's exit to block 6 also leaves the outer loop. The outer
  // loop retains that shared continuation without calling the nested edge a
  // directly sourced exit.
  CfgGraph nested({{1}, {2, 6}, {3, 6}, {2, 4}, {1}, {}, {}});
  const auto nested_nest = Build(nested);
  ASSERT_EQ(nested_nest.loop_count, 2u);
  const uint16_t outer_index = loom_cfg_loop_nest_innermost(&nested_nest, 1);
  ASSERT_LT(outer_index, nested_nest.loop_count);
  const auto& outer = nested_nest.loops[outer_index];
  EXPECT_EQ(outer.exits.count, 2u);
  EXPECT_EQ(outer.direct_exit_count, 1u);
  EXPECT_EQ(outer.continuation_index, 6u);
  IREE_ASSERT_OK(testing::CheckLoopNest(nested_nest));
}

TEST_F(CfgLoopNestTest, IrreducibleCyclesRetainNaturalSubloops) {
  CheckOracle({{1, 2}, {2, 3}, {1, 3}, {}});
  CheckOracle({{1, 2}, {3}, {3}, {4, 5}, {3, 1}, {2}});
}

TEST_F(CfgLoopNestTest, EveryThreeBlockGraph) {
  for (uint32_t bits = 0; bits < (1u << 9); ++bits) {
    SCOPED_TRACE(bits);
    std::vector<std::vector<uint16_t>> successors(3);
    for (uint16_t source = 0; source < 3; ++source) {
      for (uint16_t target = 0; target < 3; ++target) {
        if (bits & (1u << (source * 3 + target))) {
          successors[source].push_back(target);
        }
      }
    }
    CheckOracle(successors);
  }
}

TEST_F(CfgLoopNestTest, DeterministicArbitraryGraphs) {
  std::mt19937 random(715321);
  for (size_t sample = 0; sample < 2000; ++sample) {
    SCOPED_TRACE(sample);
    uint16_t count = 2 + random() % 15;
    std::vector<std::vector<uint16_t>> successors(count);
    for (auto& outgoing : successors) {
      size_t edges = random() % 5;
      for (size_t i = 0; i < edges; ++i) {
        outgoing.push_back(random() % count);
      }
    }
    CheckOracle(successors);
  }
}

TEST_F(CfgLoopNestTest, DeepNestingUsesCompactMembership) {
  constexpr uint16_t depth = 8192;
  std::vector<std::vector<uint16_t>> successors(depth * 2 + 2);
  successors[0] = {1};
  for (uint16_t i = 0; i < depth; ++i) {
    uint16_t header = i + 1;
    uint16_t latch = depth + i + 1;
    successors[header] = {uint16_t(i + 1 < depth ? header + 1 : latch),
                          uint16_t(i ? depth + i : successors.size() - 1)};
    successors[latch] = {header};
  }
  CfgGraph fixture(successors);
  const auto nest = Build(fixture);
  ASSERT_TRUE(nest.reducible);
  ASSERT_EQ(nest.loop_count, depth);
  for (uint16_t i = 0; i < depth; ++i) {
    uint16_t loop_index = loom_cfg_loop_nest_innermost(&nest, i + 1);
    ASSERT_LT(loop_index, nest.loop_count);
    const auto& loop = nest.loops[loop_index];
    EXPECT_EQ(loop.entries.count, 1u);
    EXPECT_EQ(loop.backedges.count, 1u);
    EXPECT_EQ(loop.exits.count, 1u);
    EXPECT_EQ(loop.parent_loop_index, i ? loom_cfg_loop_nest_innermost(&nest, i)
                                        : LOOM_CFG_LOOP_NEST_NONE);
    EXPECT_TRUE(loom_cfg_loop_nest_contains(&nest, loop_index, depth));
    EXPECT_FALSE(loom_cfg_loop_nest_contains(&nest, loop_index, 0));
    EXPECT_FALSE(loom_cfg_loop_nest_contains(&nest, loop_index, depth * 2 + 1));
  }
}

TEST_F(CfgLoopNestTest, NestedExecutionCountsDoNotDependOnBlockOrder) {
  // Headers 4 and 2, with latches 1 and 5. Block 7 is unreachable.
  CfgGraph fixture({{4}, {4}, {5, 1}, {}, {2, 6}, {2}, {3}, {7}});
  const auto nest = Build(fixture);
  ASSERT_EQ(nest.loop_count, 2u);
  const uint16_t outer_index = loom_cfg_loop_nest_innermost(&nest, 4);
  const uint16_t inner_index = loom_cfg_loop_nest_innermost(&nest, 2);
  ASSERT_LT(outer_index, nest.loop_count);
  ASSERT_LT(inner_index, nest.loop_count);
  std::vector<uint64_t> trips(nest.loop_count);
  trips[outer_index] = 4;
  trips[inner_index] = 8;
  std::vector<uint64_t> counts(8);
  EXPECT_TRUE(loom_cfg_loop_nest_calculate_block_execution_counts(
      &nest, trips.data(), counts.data()));
  EXPECT_EQ(counts, (std::vector<uint64_t>{1, 4, 36, 1, 5, 32, 1, 0}));
  trips[outer_index] = 0;
  EXPECT_TRUE(loom_cfg_loop_nest_calculate_block_execution_counts(
      &nest, trips.data(), counts.data()));
  EXPECT_EQ(counts, (std::vector<uint64_t>{1, 0, 0, 1, 1, 0, 1, 0}));
  trips[outer_index] = UINT64_MAX;
  EXPECT_FALSE(loom_cfg_loop_nest_calculate_block_execution_counts(
      &nest, trips.data(), counts.data()));
}

TEST_F(CfgLoopNestTest, UnmodeledPathsDoNotProduceExactCounts) {
  const std::vector<std::vector<std::vector<uint16_t>>> cases = {
      {{1, 2}, {3}, {3}, {}},
      {{1, 3}, {2, 3}, {1}, {}},
      {{1}, {2, 3}, {4}, {4}, {5, 6}, {1}, {}},
      {{1}, {2, 5}, {3, 4}, {1}, {1}, {}},
      {{1}, {2, 4}, {1, 4}, {}, {}},
      {{1, 2}, {2, 3}, {1, 3}, {}},
      {{0}},
      // A multiway loop header still chooses among body paths. A unique
      // backedge/exit cannot make both alternatives execute on every turn.
      {{1}, {2, 3, 5}, {4}, {4}, {1}, {}},
  };
  for (const auto& successors : cases) {
    CfgGraph fixture(successors);
    const auto nest = Build(fixture);
    std::vector<uint64_t> trips(nest.loop_count, 4);
    std::vector<uint64_t> counts(successors.size());
    EXPECT_FALSE(loom_cfg_loop_nest_calculate_block_execution_counts(
        &nest, trips.data(), counts.data()));
  }
}

TEST_F(CfgLoopNestTest, StructuralOrderVisitsBodiesBeforeContinuations) {
  struct Case {
    // Successor lists deliberately use non-execution block order.
    std::vector<std::vector<uint16_t>> successors;
    // Each reachable block once, with nested bodies before their exits.
    std::vector<uint16_t> expected;
  };
  const Case cases[] = {
      {{}, {}},
      {{{}}, {0}},
      {{{3}, {}, {1}, {2}}, {0, 3, 2, 1}},
      // Two sibling loops with opposite conditional successor orders.
      {{{4}, {4}, {3, 5}, {2}, {2, 1}, {}, {6}}, {0, 4, 1, 2, 3, 5}},
      // Nested loops with distinct latches and an unreachable self-cycle.
      {{{4}, {4}, {5, 1}, {}, {2, 6}, {2}, {3}, {7}}, {0, 4, 2, 5, 1, 6, 3}},
      // The inner exit is the outer backedge; neither header repeats.
      {{{1}, {2, 4}, {3, 1}, {2}, {}}, {0, 1, 2, 3, 4}},
      // Header-only inner loop, whose exit is again the outer backedge.
      {{{1}, {2, 3}, {1, 2}, {}}, {0, 1, 2, 3}},
  };
  for (const auto& test : cases) {
    CfgGraph fixture(test.successors);
    const auto nest = Build(fixture);
    std::vector<uint64_t> trips(nest.loop_count, 3);
    std::vector<uint64_t> counts(test.successors.size());
    ASSERT_TRUE(loom_cfg_loop_nest_calculate_block_execution_counts(
        &nest, trips.data(), counts.data()));
    std::vector<uint16_t> order(fixture.get()->reverse_postorder.count);
    IREE_ASSERT_OK(
        loom_cfg_loop_nest_order_blocks(&nest, &arena_, order.data()));
    EXPECT_EQ(order, test.expected);
  }
}

TEST_F(CfgLoopNestTest, StructuralOrderClosesDeepSharedLatchesIteratively) {
  constexpr uint16_t depth = 8192;
  std::vector<std::vector<uint16_t>> successors(depth + 2);
  successors[0] = {1};
  for (uint16_t i = 1; i <= depth; ++i) {
    successors[i] = {uint16_t(i == depth ? i : i + 1),
                     uint16_t(i == 1 ? depth + 1 : i - 1)};
  }
  CfgGraph fixture(successors);
  const auto nest = Build(fixture);
  std::vector<uint64_t> trips(nest.loop_count, 1);
  std::vector<uint64_t> counts(successors.size());
  ASSERT_TRUE(loom_cfg_loop_nest_calculate_block_execution_counts(
      &nest, trips.data(), counts.data()));
  std::vector<uint16_t> order(successors.size());
  IREE_ASSERT_OK(loom_cfg_loop_nest_order_blocks(&nest, &arena_, order.data()));
  for (uint16_t i = 0; i < order.size(); ++i) {
    EXPECT_EQ(order[i], i);
  }
}

TEST_F(CfgLoopNestTest, StructuralOrderPreservesConditionalAlternatives) {
  // Diamonds before, within, and after a loop, including a body with nested
  // sibling loops. Reverse successor order to exercise both DFS choices.
  const std::vector<std::vector<std::vector<uint16_t>>> cases = {
      {{1, 2}, {3}, {3}, {4, 5}, {3}, {}},
      {{1}, {2, 6}, {3, 4}, {5}, {5}, {1}, {}},
      {{1}, {2, 3}, {1}, {4, 5}, {6}, {6}, {}},
      {{1}, {2, 9}, {3, 6}, {4, 5}, {3}, {8}, {7, 8}, {6}, {1}, {}},
      // A disconnected branch and self-cycle never enter the order.
      {{4}, {3}, {}, {4}, {1, 2}, {6, 7}, {5}, {7}},
  };
  for (auto successors : cases) {
    CheckOracle(successors);
    for (auto& outgoing : successors) {
      std::reverse(outgoing.begin(), outgoing.end());
    }
    CheckOracle(successors);
  }
}

TEST_F(CfgLoopNestTest, LoopMultipliersSurviveUnmodeledDiamond) {
  // Block 1 is a four-trip loop header. Blocks 3 and 4 are alternatives in
  // the loop body and block 5 is their common latch.
  CfgGraph fixture({{1}, {2, 6}, {3, 4}, {5}, {5}, {1}, {}});
  const auto nest = Build(fixture);
  ASSERT_EQ(nest.loop_count, 1u);
  const uint64_t trip_count = 4;
  std::vector<uint64_t> multipliers(7);
  EXPECT_TRUE(loom_cfg_loop_nest_calculate_block_multipliers(
      &nest, &trip_count, multipliers.data()));
  EXPECT_EQ(multipliers, (std::vector<uint64_t>{1, 5, 4, 4, 4, 4, 1}));
  EXPECT_FALSE(loom_cfg_loop_nest_calculate_block_execution_counts(
      &nest, &trip_count, multipliers.data()));
}

TEST_F(CfgLoopNestTest, UnreachableBranchesDoNotInvalidateCounts) {
  CfgGraph fixture({{1}, {}, {1, 3}, {}});
  const auto nest = Build(fixture);
  std::vector<uint64_t> counts(4);
  EXPECT_TRUE(loom_cfg_loop_nest_calculate_block_execution_counts(
      &nest, nullptr, counts.data()));
  EXPECT_EQ(counts, (std::vector<uint64_t>{1, 1, 0, 0}));
}

TEST_F(CfgLoopNestTest, IndependentLoopBoundsSurviveUnknownTrips) {
  // A finite producer precedes a conditional and an unrelated runtime loop.
  CfgGraph fixture({{1}, {2, 3}, {1}, {4, 5}, {5}, {6, 7}, {5}, {}});
  const auto nest = Build(fixture);
  std::vector<uint64_t> trips(nest.loop_count, UINT64_MAX);
  trips[loom_cfg_loop_nest_innermost(&nest, 1)] = 16;
  std::vector<uint64_t> bounds(8);
  EXPECT_FALSE(loom_cfg_loop_nest_calculate_block_execution_bounds(
      &nest, trips.data(), bounds.data()));
  EXPECT_EQ(bounds, (std::vector<uint64_t>{1, 17, 16, 1, 1, UINT64_MAX,
                                           UINT64_MAX, 1}));
}

TEST_F(CfgLoopNestTest, ConditionalLoopBodiesHaveBoundsButNotExactCounts) {
  CfgGraph fixture({{1}, {2, 6}, {3, 4}, {5}, {5}, {1}, {}});
  const auto nest = Build(fixture);
  const uint64_t trips = 4;
  std::vector<uint64_t> bounds(7);
  EXPECT_FALSE(loom_cfg_loop_nest_calculate_block_execution_bounds(
      &nest, &trips, bounds.data()));
  EXPECT_EQ(bounds, (std::vector<uint64_t>{1, 5, 4, 4, 4, 4, 1}));
}

TEST_F(CfgLoopNestTest, NestedBoundsSaturateAndPreserveZeroTrips) {
  CfgGraph fixture({{4}, {4}, {5, 1}, {}, {2, 6}, {2}, {3}, {7}});
  const auto nest = Build(fixture);
  const auto outer = loom_cfg_loop_nest_innermost(&nest, 4);
  const auto inner = loom_cfg_loop_nest_innermost(&nest, 2);
  std::vector<uint64_t> trips(nest.loop_count);
  std::vector<uint64_t> bounds(8);
  trips[outer] = 4;
  trips[inner] = UINT64_MAX;
  EXPECT_FALSE(loom_cfg_loop_nest_calculate_block_execution_bounds(
      &nest, trips.data(), bounds.data()));
  EXPECT_EQ(bounds,
            (std::vector<uint64_t>{1, 4, UINT64_MAX, 1, 5, UINT64_MAX, 1, 0}));
  trips[inner] = UINT64_MAX / 2;
  EXPECT_FALSE(loom_cfg_loop_nest_calculate_block_execution_bounds(
      &nest, trips.data(), bounds.data()));
  EXPECT_EQ(bounds,
            (std::vector<uint64_t>{1, 4, UINT64_MAX, 1, 5, UINT64_MAX, 1, 0}));
  trips[outer] = 0;
  trips[inner] = UINT64_MAX;
  EXPECT_FALSE(loom_cfg_loop_nest_calculate_block_execution_bounds(
      &nest, trips.data(), bounds.data()));
  EXPECT_EQ(bounds, (std::vector<uint64_t>{1, 0, 0, 1, 1, 0, 1, 0}));
}

TEST_F(CfgLoopNestTest, UnsupportedSiblingDoesNotEraseFinitePrefix) {
  // The second loop has an additional non-header exit.
  CfgGraph fixture({{1}, {2, 3}, {1}, {4, 6}, {3, 6}, {}, {}});
  const auto nest = Build(fixture);
  std::vector<uint64_t> trips(nest.loop_count, 16);
  std::vector<uint64_t> bounds(7);
  EXPECT_FALSE(loom_cfg_loop_nest_calculate_block_execution_bounds(
      &nest, trips.data(), bounds.data()));
  EXPECT_EQ(bounds,
            (std::vector<uint64_t>{1, 17, 16, UINT64_MAX, UINT64_MAX, 0, 1}));
}

TEST_F(CfgLoopNestTest, IrreducibleCyclesKeepOnlyAcyclicBounds) {
  CfgGraph fixture({{1}, {2, 3}, {3, 4}, {2, 4}, {}});
  const auto nest = Build(fixture);
  ASSERT_FALSE(nest.reducible);
  std::vector<uint64_t> trips(nest.loop_count, 1);
  std::vector<uint64_t> bounds(5);
  EXPECT_FALSE(loom_cfg_loop_nest_calculate_block_execution_bounds(
      &nest, trips.data(), bounds.data()));
  EXPECT_EQ(bounds, (std::vector<uint64_t>{1, 1, UINT64_MAX, UINT64_MAX, 1}));
}

}  // namespace
}  // namespace loom
