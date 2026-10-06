// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/storage/block_pool.h"

#include <array>
#include <set>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(BlockPoolTest, InterleavedGrowthAndRetiredSuffixReuse) {
  loom_serve_block_pool_t pool;
  IREE_ASSERT_OK(
      loom_serve_block_pool_initialize(8, iree_allocator_system(), &pool));
  uint32_t first[4] = {}, second[4] = {};
  loom_serve_block_pool_acquire(&pool, 1, first);
  loom_serve_block_pool_acquire(&pool, 2, second);
  loom_serve_block_pool_acquire(&pool, 3, first + 1);
  EXPECT_EQ(pool.available, 2u);
  EXPECT_EQ(first[0], 0u);
  EXPECT_EQ(first[1], 3u);
  EXPECT_EQ(first[3], 5u);

  // The first row retires a rejected suffix. Only those pages can be reused;
  // both rows' retained prefixes remain live and unchanged.
  loom_serve_block_pool_release(&pool, 2, first + 2);
  loom_serve_block_pool_acquire(&pool, 2, second + 2);
  EXPECT_EQ(second[2], first[3]);
  EXPECT_EQ(second[3], first[2]);
  EXPECT_EQ(second[0], 1u);
  EXPECT_EQ(second[1], 2u);
  const std::set<uint32_t> owned = {first[0],  first[1],  second[0],
                                    second[1], second[2], second[3]};
  EXPECT_EQ(owned.size(), 6u);
  loom_serve_block_pool_release(&pool, 2, first);
  loom_serve_block_pool_release(&pool, 4, second);
  EXPECT_EQ(pool.available, pool.capacity);

  std::array<uint32_t, 8> all;
  loom_serve_block_pool_acquire(&pool, all.size(), all.data());
  EXPECT_EQ(pool.available, 0u);
  const std::set<uint32_t> unique(all.begin(), all.end());
  EXPECT_EQ(unique, (std::set<uint32_t>{0, 1, 2, 3, 4, 5, 6, 7}));
  loom_serve_block_pool_release(&pool, all.size(), all.data());
  loom_serve_block_pool_deinitialize(&pool);
}

TEST(BlockPoolTest, WarmReuseKeepsBackingAndHandlesEmptyExtents) {
  loom_serve_block_pool_t pool;
  IREE_ASSERT_OK(
      loom_serve_block_pool_initialize(1, iree_allocator_system(), &pool));
  uint32_t* const backing = pool.free_blocks;
  uint32_t block = 0;
  for (int i = 0; i < 1000; ++i) {
    loom_serve_block_pool_acquire(&pool, 0, nullptr);
    loom_serve_block_pool_release(&pool, 0, nullptr);
    loom_serve_block_pool_acquire(&pool, 1, &block);
    EXPECT_EQ(block, 0u);
    EXPECT_EQ(pool.available, 0u);
    loom_serve_block_pool_release(&pool, 1, &block);
    EXPECT_EQ(pool.available, 1u);
    EXPECT_EQ(pool.free_blocks, backing);
  }
  loom_serve_block_pool_deinitialize(&pool);
}

TEST(BlockPoolTest, CompactionMovesOnlyIntoDisjointFreeSlots) {
  // Exhaust every ownership pattern in a small pool, including all-live and
  // all-free. This checks the actual planning and commit contract, not a
  // second implementation of the planner.
  for (uint32_t mask = 0; mask < 256; ++mask) {
    SCOPED_TRACE(mask);
    loom_serve_block_pool_t pool;
    IREE_ASSERT_OK(
        loom_serve_block_pool_initialize(8, iree_allocator_system(), &pool));
    std::array<uint32_t, 8> blocks;
    loom_serve_block_pool_acquire(&pool, blocks.size(), blocks.data());
    std::set<uint32_t> occupied;
    for (uint32_t i = 0; i < blocks.size(); ++i) {
      if (mask & (1u << i)) {
        occupied.insert(i);
      } else {
        loom_serve_block_pool_release(&pool, 1, &blocks[i]);
      }
    }
    std::array<uint32_t, 8> destinations;
    const uint32_t moved =
        loom_serve_block_pool_plan_compaction(&pool, destinations.data());
    uint32_t observed_moves = 0;
    std::set<uint32_t> compacted;
    for (uint32_t i = 0; i < blocks.size(); ++i) {
      if (!occupied.count(i)) {
        EXPECT_EQ(destinations[i], UINT32_MAX);
        continue;
      }
      EXPECT_LT(destinations[i], occupied.size());
      EXPECT_TRUE(compacted.insert(destinations[i]).second);
      if (destinations[i] != i) {
        ++observed_moves;
        EXPECT_EQ(occupied.count(destinations[i]), 0u);
      }
    }
    EXPECT_EQ(moved, observed_moves);
    loom_serve_block_pool_commit_compaction(&pool);
    const uint32_t available = pool.available;
    loom_serve_block_pool_acquire(&pool, available, blocks.data());
    for (uint32_t i = 0; i < available; ++i) {
      EXPECT_EQ(blocks[i], occupied.size() + i);
    }
    loom_serve_block_pool_deinitialize(&pool);
  }
}

}  // namespace
