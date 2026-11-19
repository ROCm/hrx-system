// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/index_set.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include "iree/testing/gtest.h"

namespace {

TEST(IndexSetTest, LayoutCoversFullIndexDomain) {
  const auto empty = loom_index_set_calculate_layout(0);
  EXPECT_EQ(empty.word_count, 0u);
  EXPECT_EQ(empty.level_count, 0u);
  EXPECT_EQ(loom_index_set_select(&empty, nullptr, 0), LOOM_INDEX_SET_NONE);

  const auto single = loom_index_set_calculate_layout(64);
  EXPECT_EQ(single.word_count, 1u);
  EXPECT_EQ(single.level_count, 1u);
  const auto two_words = loom_index_set_calculate_layout(65);
  EXPECT_EQ(two_words.word_count, 3u);
  EXPECT_EQ(two_words.level_count, 2u);
  EXPECT_EQ(two_words.level_starts[1], 2u);
  const auto third_level = loom_index_set_calculate_layout(4097);
  EXPECT_EQ(third_level.word_count, 68u);
  EXPECT_EQ(third_level.level_count, 3u);
  EXPECT_EQ(third_level.level_starts[1], 65u);
  EXPECT_EQ(third_level.level_starts[2], 67u);

  // Layout arithmetic covers the complete domain without a giant allocation.
  const auto largest = loom_index_set_calculate_layout(UINT32_MAX);
  EXPECT_EQ(largest.level_count, 6u);
  EXPECT_EQ(largest.word_count, 68174085u);
  EXPECT_EQ(largest.level_starts[5], largest.word_count - 1);
}

TEST(IndexSetTest, MembershipAndCyclicSelectionMatchOrderedSet) {
  for (uint32_t capacity : {1u, 63u, 64u, 65u, 4095u, 4096u, 4097u, 262143u,
                            262144u, 262145u, 16777217u}) {
    SCOPED_TRACE(capacity);
    const auto layout = loom_index_set_calculate_layout(capacity);
    std::vector<uint64_t> words(layout.word_count, 0);
    std::set<uint32_t> members;
    auto insert = [&](uint32_t index) {
      EXPECT_EQ(loom_index_set_insert(&layout, words.data(), index),
                members.insert(index).second);
    };
    auto select = [&](uint32_t start) {
      uint32_t expected = LOOM_INDEX_SET_NONE;
      if (!members.empty()) {
        auto found = members.lower_bound(start);
        expected = found == members.end() ? *members.begin() : *found;
      }
      EXPECT_EQ(loom_index_set_select(&layout, words.data(), start), expected);
    };
    select(0);
    for (uint32_t index : {0u, 1u, 62u, 63u, 64u, 65u, 4095u, 4096u, 4097u,
                           262143u, 262144u, 262145u, capacity - 1}) {
      if (index < capacity) {
        insert(index);
        insert(index);
        select(index);
      }
    }
    std::mt19937 random(12345);
    for (uint32_t iteration = 0; iteration < 2048; ++iteration) {
      const uint32_t index = random() % capacity;
      if (iteration % 3) {
        insert(index);
      } else {
        loom_index_set_erase(&layout, words.data(), index);
        members.erase(index);
      }
      select(0);
      select(random() % capacity);
      select(capacity - 1);
    }
    // Removing the final leaf of a subtree must also retire its summaries.
    while (!members.empty()) {
      const uint32_t index = *members.rbegin();
      loom_index_set_erase(&layout, words.data(), index);
      loom_index_set_erase(&layout, words.data(), index);
      members.erase(index);
      select(index);
    }
    EXPECT_TRUE(std::all_of(words.begin(), words.end(),
                            [](uint64_t word) { return word == 0; }));
  }
}

TEST(IndexSetTest, RangeInsertionPreservesNeighborsAndSummaries) {
  for (uint32_t capacity : {1u, 65u, 4097u, 262145u}) {
    const auto layout = loom_index_set_calculate_layout(capacity);
    for (uint32_t first : {0u, 1u, 63u, 64u, 4095u, capacity - 1}) {
      if (first >= capacity) {
        continue;
      }
      for (uint32_t length : {0u, 1u, 2u, 64u, 4096u, capacity - first}) {
        length = std::min(length, capacity - first);
        SCOPED_TRACE(::testing::Message()
                     << capacity << ":" << first << ":" << length);
        std::vector<uint64_t> words(layout.word_count, 0);
        std::set<uint32_t> members;
        for (uint32_t neighbor : {0u, capacity - 1}) {
          if (neighbor < first || neighbor >= first + length) {
            loom_index_set_insert(&layout, words.data(), neighbor);
            members.insert(neighbor);
          }
        }
        loom_index_set_insert_range(&layout, words.data(), first, length);
        loom_index_set_insert_range(&layout, words.data(), first, length);
        for (uint32_t index = first; index < first + length; ++index) {
          members.insert(index);
        }
        EXPECT_EQ(loom_index_set_select(&layout, words.data(), 0),
                  members.empty() ? LOOM_INDEX_SET_NONE : *members.begin());
        for (uint32_t index : members) {
          EXPECT_EQ(loom_index_set_select(&layout, words.data(), index), index);
          loom_index_set_erase(&layout, words.data(), index);
        }
        EXPECT_TRUE(std::all_of(words.begin(), words.end(),
                                [](uint64_t word) { return word == 0; }));
      }
    }
  }
}

TEST(IndexSetTest, SharedLayoutKeepsAdjacentSetsIndependent) {
  const auto layout = loom_index_set_calculate_layout(129);
  std::vector<uint64_t> words(3 * layout.word_count, 0);
  for (uint32_t set = 0; set < 3; ++set) {
    auto* bits = words.data() + set * layout.word_count;
    EXPECT_TRUE(loom_index_set_insert(&layout, bits, set * 64));
    // Leaves remain directly accessible to set-intersection consumers.
    EXPECT_EQ(bits[set], UINT64_C(1));
  }
  loom_index_set_erase(&layout, words.data() + layout.word_count, 64);
  EXPECT_EQ(loom_index_set_select(&layout, words.data(), 0), 0u);
  EXPECT_EQ(loom_index_set_select(&layout, words.data() + layout.word_count, 0),
            LOOM_INDEX_SET_NONE);
  EXPECT_EQ(
      loom_index_set_select(&layout, words.data() + 2 * layout.word_count, 0),
      128u);
  loom_index_set_clear(&layout, words.data());
  loom_index_set_clear(&layout, words.data());
  EXPECT_EQ(loom_index_set_select(&layout, words.data(), 0),
            LOOM_INDEX_SET_NONE);
  EXPECT_EQ(
      loom_index_set_select(&layout, words.data() + 2 * layout.word_count, 0),
      128u);
}

TEST(IndexSetTest, ClearingSparseAndDenseSetsRetiresEverySummary) {
  for (uint32_t capacity : {0u, 1u, 64u, 65u, 4097u, 262145u}) {
    const auto layout = loom_index_set_calculate_layout(capacity);
    std::vector<uint64_t> words(layout.word_count, 0);
    loom_index_set_clear(&layout, words.data());
    for (uint32_t index = 0; index < capacity; index += 4096) {
      loom_index_set_insert(&layout, words.data(), index);
    }
    loom_index_set_clear(&layout, words.data());
    EXPECT_TRUE(std::all_of(words.begin(), words.end(),
                            [](uint64_t word) { return word == 0; }));
    loom_index_set_insert_range(&layout, words.data(), 0, capacity);
    loom_index_set_clear(&layout, words.data());
    EXPECT_TRUE(std::all_of(words.begin(), words.end(),
                            [](uint64_t word) { return word == 0; }));
  }
}

}  // namespace
