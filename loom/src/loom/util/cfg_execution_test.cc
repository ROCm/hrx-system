// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/cfg_execution.h"

#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/util/cfg_graph_test_util.h"

namespace loom {
namespace {

class CfgExecutionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  static bool IsModeled(const void* user_data, uint16_t block_index) {
    const auto* modeled = static_cast<const std::vector<uint8_t>*>(user_data);
    return block_index < modeled->size() && (*modeled)[block_index] != 0;
  }

  std::vector<bool> Classify(
      const std::vector<std::vector<uint16_t>>& successors,
      const std::vector<uint16_t>& modeled_selectors = {}) {
    testing::CfgGraph fixture(successors);
    loom_cfg_control_t control = {};
    IREE_EXPECT_OK(loom_cfg_control_build(fixture.get(), &arena_, &control));
    std::vector<uint8_t> modeled(successors.size());
    for (uint16_t block : modeled_selectors) {
      modeled[block] = 1;
    }
    iree_bitmap_t unmodeled = {};
    IREE_EXPECT_OK(loom_cfg_execution_classify_unmodeled_blocks(
        &control, loom_cfg_execution_selector_model_t{IsModeled, &modeled},
        &arena_, &unmodeled));
    std::vector<bool> result(successors.size());
    for (iree_host_size_t i = 0; i < successors.size(); ++i) {
      result[i] = iree_bitmap_test(unmodeled, i);
    }
    iree_arena_reset(&arena_);
    return result;
  }

  iree_arena_block_pool_t pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(CfgExecutionTest, ReconvergenceEndsSelectorUncertainty) {
  EXPECT_EQ(Classify({{1, 2}, {3}, {3}, {4}, {}}),
            (std::vector<bool>{false, true, true, false, false}));
  EXPECT_EQ(Classify({{1, 2, 3}, {4}, {4}, {4}, {}}),
            (std::vector<bool>{false, true, true, true, false}));
}

TEST_F(CfgExecutionTest, NestedAndSequentialControlStayScoped) {
  // The inner merge remains controlled by the outer selector. Only the outer
  // merge recovers an unconditional execution count.
  EXPECT_EQ(
      Classify({{1, 6}, {2, 3}, {4}, {4}, {5}, {7}, {7}, {}}),
      (std::vector<bool>{false, true, true, true, true, true, true, false}));
  EXPECT_EQ(Classify({{1, 2}, {3}, {3}, {4, 5}, {6}, {6}, {}}),
            (std::vector<bool>{false, true, true, false, true, true, false}));
}

TEST_F(CfgExecutionTest, EarlyExitHasNoFalseReconvergence) {
  EXPECT_EQ(Classify({{1, 2}, {}, {3}, {}}),
            (std::vector<bool>{false, true, true, true}));
}

TEST_F(CfgExecutionTest, ModeledLoopSelectorPreservesBodyCounts) {
  // Block 1 is a fixed-trip loop header. The data-dependent diamond inside its
  // body controls blocks 3 and 4, then reconverges at the latch in block 5.
  EXPECT_EQ(Classify({{1}, {2, 6}, {3, 4}, {5}, {5}, {1}, {}}, {1}),
            (std::vector<bool>{false, false, false, true, true, false, false}));
}

TEST_F(CfgExecutionTest, UnavailableControlKeepsReachableBlocksConservative) {
  testing::CfgGraph fixture({{1}, {}, {3}, {}});
  loom_cfg_control_t control = {.graph = fixture.get()};
  iree_bitmap_t unmodeled = {};
  IREE_ASSERT_OK(loom_cfg_execution_classify_unmodeled_blocks(
      &control, loom_cfg_execution_selector_model_t{}, &arena_, &unmodeled));
  EXPECT_TRUE(iree_bitmap_test(unmodeled, 0));
  EXPECT_TRUE(iree_bitmap_test(unmodeled, 1));
  EXPECT_FALSE(iree_bitmap_test(unmodeled, 2));
  EXPECT_FALSE(iree_bitmap_test(unmodeled, 3));
}

static void EnumerateAcyclicPaths(
    const std::vector<std::vector<uint16_t>>& successors, uint16_t block,
    std::vector<bool>* active,
    std::vector<std::vector<bool>>* completed_paths) {
  (*active)[block] = true;
  if (successors[block].empty()) {
    completed_paths->push_back(*active);
  } else {
    for (uint16_t successor : successors[block]) {
      EnumerateAcyclicPaths(successors, successor, active, completed_paths);
    }
  }
  (*active)[block] = false;
}

TEST_F(CfgExecutionTest, ExhaustiveAcyclicGraphsMatchPathMembership) {
  constexpr uint16_t kBlockCount = 5;
  constexpr uint32_t kForwardEdgeCount = kBlockCount * (kBlockCount - 1) / 2;
  for (uint32_t edge_bits = 0; edge_bits < (1u << kForwardEdgeCount);
       ++edge_bits) {
    SCOPED_TRACE(edge_bits);
    std::vector<std::vector<uint16_t>> successors(kBlockCount);
    uint32_t edge_ordinal = 0;
    for (uint16_t source = 0; source < kBlockCount; ++source) {
      for (uint16_t target = source + 1; target < kBlockCount; ++target) {
        if (edge_bits & (1u << edge_ordinal)) {
          successors[source].push_back(target);
        }
        ++edge_ordinal;
      }
    }

    std::vector<bool> active(kBlockCount);
    std::vector<std::vector<bool>> paths;
    EnumerateAcyclicPaths(successors, 0, &active, &paths);
    ASSERT_FALSE(paths.empty());
    std::vector<bool> expected(kBlockCount);
    for (uint16_t block = 0; block < kBlockCount; ++block) {
      bool any = false;
      bool all = true;
      for (const auto& path : paths) {
        any |= path[block];
        all &= path[block];
      }
      expected[block] = any != all;
    }
    EXPECT_EQ(Classify(successors), expected);
  }
}

}  // namespace
}  // namespace loom
