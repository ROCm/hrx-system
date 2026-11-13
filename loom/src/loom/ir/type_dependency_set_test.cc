// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/type_dependencies.h"

namespace loom {
namespace {

class ValueSetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_value_set_index_t* AllocateIndex() {
    loom_value_set_index_t* index = nullptr;
    IREE_CHECK_OK(loom_value_set_index_allocate(&arena_, &index));
    return index;
  }

  static loom_value_set_id_t Add(loom_value_set_index_t* index,
                                 loom_value_set_id_t set,
                                 loom_value_id_t value) {
    loom_value_set_id_t result = 0;
    IREE_CHECK_OK(loom_value_set_index_add(index, set, value, &result));
    return result;
  }

  static loom_value_set_id_t Union(loom_value_set_index_t* index,
                                   loom_value_set_id_t first,
                                   loom_value_set_id_t second) {
    loom_value_set_id_t result = 0;
    IREE_CHECK_OK(loom_value_set_index_union(index, first, second, &result));
    return result;
  }

  static loom_value_set_id_t Intersection(loom_value_set_index_t* index,
                                          loom_value_set_id_t first,
                                          loom_value_set_id_t second) {
    loom_value_set_id_t result = 0;
    IREE_CHECK_OK(
        loom_value_set_index_intersection(index, first, second, &result));
    return result;
  }

  static std::vector<loom_value_id_t> Members(
      const loom_value_set_index_t* index, loom_value_set_id_t set) {
    loom_value_set_cursor_t cursor;
    loom_value_set_cursor_begin(index, set, &cursor);
    std::vector<loom_value_id_t> members;
    for (loom_value_id_t value = loom_value_set_cursor_next(&cursor);
         value != LOOM_VALUE_ID_INVALID;
         value = loom_value_set_cursor_next(&cursor)) {
      members.push_back(value);
    }
    return members;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(ValueSetTest, CanonicalUnionPreservesExactMembership) {
  loom_value_set_index_t* index = AllocateIndex();
  loom_value_set_id_t evens = 0;
  loom_value_set_id_t odds = 0;
  for (loom_value_id_t value : {0u, 2u, 64u, 4096u}) {
    evens = Add(index, evens, value);
  }
  for (loom_value_id_t value : {1u, 3u, 65u, 4097u}) {
    odds = Add(index, odds, value);
  }

  const loom_value_set_id_t all = Union(index, evens, odds);
  EXPECT_EQ(Members(index, all),
            (std::vector<loom_value_id_t>{0, 1, 2, 3, 64, 65, 4096, 4097}));
  for (loom_value_id_t value : Members(index, all)) {
    EXPECT_TRUE(loom_value_set_index_contains(index, all, value));
  }
  EXPECT_FALSE(loom_value_set_index_contains(index, all, 63));
  EXPECT_FALSE(loom_value_set_index_contains(index, all, 4095));

  EXPECT_EQ(Union(index, odds, evens), all);
  EXPECT_EQ(Union(index, all, evens), all);
  EXPECT_EQ(Add(index, all, 65), all);
}

TEST_F(ValueSetTest, PrefixSharesCompleteSubtrees) {
  loom_value_set_index_t* index = AllocateIndex();
  loom_value_set_id_t set = 0;
  for (loom_value_id_t value : {1u, 63u, 64u, 65u, 127u, 128u, 4096u}) {
    set = Add(index, set, value);
  }

  loom_value_set_id_t prefix = 0;
  IREE_ASSERT_OK(loom_value_set_index_prefix(index, set, 128, &prefix));
  EXPECT_EQ(Members(index, prefix),
            (std::vector<loom_value_id_t>{1, 63, 64, 65, 127}));
  loom_value_set_id_t complete = 0;
  IREE_ASSERT_OK(loom_value_set_index_prefix(index, set, 4097, &complete));
  EXPECT_EQ(complete, set);
  loom_value_set_id_t empty = set;
  IREE_ASSERT_OK(loom_value_set_index_prefix(index, set, 1, &empty));
  EXPECT_EQ(empty, 0u);
}

TEST_F(ValueSetTest, LongNestedSetsRetainEveryMember) {
  loom_value_set_index_t* index = AllocateIndex();
  loom_value_set_id_t set = 0;
  std::array<loom_value_set_id_t, 4096> roots;
  for (loom_value_id_t value = 0; value < roots.size(); ++value) {
    set = Add(index, set, value);
    roots[value] = set;
  }

  EXPECT_EQ(Members(index, roots.back()).size(), roots.size());
  for (loom_value_id_t value = 0; value < roots.size(); ++value) {
    EXPECT_TRUE(loom_value_set_index_contains(index, roots.back(), value));
  }
  EXPECT_EQ(Union(index, roots[1023], roots.back()), roots.back());
}

TEST_F(ValueSetTest, IntersectionPreservesCanonicalMembership) {
  loom_value_set_index_t* index = AllocateIndex();
  const std::array<loom_value_id_t, 8> values = {
      0, 1, 63, 64, 65, 4096, UINT32_C(1) << 31, UINT32_MAX - 1};
  std::array<loom_value_set_id_t, 256> sets = {};
  for (uint32_t mask = 1; mask < sets.size(); ++mask) {
    const uint32_t bit = iree_math_count_trailing_zeros_u32(mask);
    sets[mask] = Add(index, sets[mask & (mask - 1)], values[bit]);
  }
  for (uint32_t left = 0; left < sets.size(); ++left) {
    for (uint32_t right = 0; right < sets.size(); ++right) {
      EXPECT_EQ(Intersection(index, sets[left], sets[right]),
                sets[left & right])
          << left << ", " << right;
    }
    const auto singleton = loom_value_set_index_singleton(index, sets[left]);
    EXPECT_EQ(singleton, left && !(left & (left - 1))
                             ? values[iree_math_count_trailing_zeros_u32(left)]
                             : LOOM_VALUE_ID_INVALID);
  }
}

TEST_F(ValueSetTest, IntersectionSharesNestedSetsWithoutAllocation) {
  loom_value_set_index_t* index = AllocateIndex();
  loom_value_set_id_t set = 0;
  std::array<loom_value_set_id_t, 1024> roots;
  for (loom_value_id_t value = 0; value < roots.size(); ++value) {
    set = Add(index, set, value * 4097);
    roots[value] = set;
  }
  const auto allocated = arena_.used_allocation_size;
  for (auto root : roots) {
    EXPECT_EQ(Intersection(index, root, set), root);
    EXPECT_EQ(Intersection(index, set, root), root);
  }
  EXPECT_EQ(arena_.used_allocation_size, allocated);
}

}  // namespace
}  // namespace loom
