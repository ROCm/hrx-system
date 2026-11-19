// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/lifetime_coalescing.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class LifetimeCoalescingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  // A construction owns a payload alongside its independent children.
  size_t Construction(size_t parent, uint64_t length, uint32_t depth = 0) {
    const size_t owner = lengths_.size();
    lengths_.push_back(length);
    depths_.push_back(depth);
    const size_t node =
        Append(parent, LOOM_PIPELINE_LIFETIME_CONCURRENT, owner);
    Append(node, LOOM_PIPELINE_LIFETIME_OWNER, owner);
    return node;
  }
  size_t Serial(size_t parent) {
    return Append(parent, LOOM_PIPELINE_LIFETIME_SERIAL, nodes_[parent].owner);
  }
  void Expect(uint64_t capacity, std::vector<size_t> expected) {
    loom_pipeline_lifetimes_t lifetimes = {.nodes = nodes_.data(),
                                           .count = nodes_.size(),
                                           .loop_depths = depths_.data()};
    std::vector<size_t> actual(lengths_.size());
    IREE_ASSERT_OK(loom_pipeline_lifetimes_coalesce(
        &lifetimes, lengths_.data(), capacity, &arena_, actual.data()));
    EXPECT_EQ(actual, expected);
  }
  static constexpr size_t none = IREE_HOST_SIZE_MAX;

 private:
  size_t Append(size_t parent, loom_pipeline_lifetime_kind_t kind,
                size_t owner) {
    const size_t index = nodes_.size();
    nodes_.push_back({kind, owner, parent, none, none});
    if (parent != none) {
      if (last_child_[parent] == none) {
        nodes_[parent].first_child = index;
      } else {
        nodes_[last_child_[parent]].next_sibling = index;
      }
      last_child_[parent] = index;
    }
    last_child_.push_back(none);
    return index;
  }
  // Lifetime nodes in construction order.
  std::vector<loom_pipeline_lifetime_t> nodes_;
  // Tail of each construction or schedule's child list.
  std::vector<size_t> last_child_;
  // Independent payload per construction.
  std::vector<uint64_t> lengths_;
  // Source loop nesting per construction.
  std::vector<uint32_t> depths_;
  // Block backing for the placement scratch arena.
  iree_arena_block_pool_t blocks_;
  // Temporary storage for one test.
  iree_arena_allocator_t arena_;
};

TEST_F(LifetimeCoalescingTest, HotSerialChildCompetesAcrossNestedSchedules) {
  size_t root = Construction(none, 2);
  size_t outer = Serial(root);
  Construction(outer, 8);
  size_t nested = Serial(Serial(outer));
  Construction(nested, 4, 2);
  Construction(nested, 7, 1);
  // The four hot units coexist with either eight or seven overlaid units.
  Expect(14, {0, 1, 0, 3});
  Expect(9, {0, 1, 2, 3});
  Expect(21, {0, 0, 0, 0});
}

TEST_F(LifetimeCoalescingTest, IndependentScopesReserveEachOthersPeak) {
  size_t root = Construction(none, 1);
  size_t left = Construction(root, 2);
  size_t right = Construction(root, 3);
  for (size_t parent : {left, right}) {
    size_t serial = Serial(parent);
    Construction(serial, 4, 1);
    Construction(serial, 6);
    Construction(serial, 6);
  }
  // Each side needs six units even before any payload is retained. Four
  // spare units cover one repeated child. Eight cover one in each scope.
  Expect(22, {0, 0, 0, 0, 4, 5, 6, 7, 8});
  Expect(26, {0, 0, 0, 0, 4, 5, 0, 7, 8});
  Expect(32, {0, 0, 0, 0, 0, 0, 0, 7, 8});
}

TEST_F(LifetimeCoalescingTest, EqualPriorityUsesSizeThenConstructionOrdinal) {
  size_t serial = Serial(Construction(none, 0));
  Construction(serial, 1);
  Construction(serial, 5);
  Construction(serial, 5);
  Construction(serial, 0);
  Expect(10, {0, 1, 0, 3, 0});
  Expect(11, {0, 0, 0, 0, 0});
}

}  // namespace
