// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/block_pool.h"

#include <array>
#include <set>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(BlockPoolTest, EmptyPoolHasNoOwnership) {
  loom_runtime_block_pool_t pool;
  IREE_ASSERT_OK(
      loom_runtime_block_pool_initialize(0, iree_allocator_system(), &pool));
  EXPECT_EQ(pool.capacity, 0u);
  EXPECT_EQ(pool.available, 0u);
  EXPECT_EQ(pool.references, nullptr);
  loom_runtime_block_pool_acquire(&pool, 0, nullptr);
  loom_runtime_block_pool_retain(&pool, 0, nullptr);
  loom_runtime_block_pool_release(&pool, 0, nullptr);
  EXPECT_EQ(loom_runtime_block_pool_plan_compaction(&pool, nullptr), 0u);
  loom_runtime_block_pool_commit_compaction(&pool, nullptr);
  loom_runtime_block_pool_deinitialize(&pool);
}

TEST(BlockPoolTest, InterleavedGrowthAndRetiredSuffixReuse) {
  loom_runtime_block_pool_t pool;
  IREE_ASSERT_OK(
      loom_runtime_block_pool_initialize(8, iree_allocator_system(), &pool));
  uint32_t first[4] = {}, second[4] = {};
  loom_runtime_block_pool_acquire(&pool, 1, first);
  loom_runtime_block_pool_acquire(&pool, 2, second);
  loom_runtime_block_pool_acquire(&pool, 3, first + 1);
  EXPECT_EQ(pool.available, 2u);
  EXPECT_EQ(first[0], 0u);
  EXPECT_EQ(first[1], 3u);
  EXPECT_EQ(first[3], 5u);

  // The first row retires a rejected suffix. Only those pages can be reused;
  // both rows' retained prefixes remain live and unchanged.
  loom_runtime_block_pool_release(&pool, 2, first + 2);
  loom_runtime_block_pool_acquire(&pool, 2, second + 2);
  EXPECT_EQ(second[2], first[3]);
  EXPECT_EQ(second[3], first[2]);
  EXPECT_EQ(second[0], 1u);
  EXPECT_EQ(second[1], 2u);
  const std::set<uint32_t> owned = {first[0],  first[1],  second[0],
                                    second[1], second[2], second[3]};
  EXPECT_EQ(owned.size(), 6u);
  loom_runtime_block_pool_release(&pool, 2, first);
  loom_runtime_block_pool_release(&pool, 4, second);
  EXPECT_EQ(pool.available, pool.capacity);

  std::array<uint32_t, 8> all;
  loom_runtime_block_pool_acquire(&pool, all.size(), all.data());
  EXPECT_EQ(pool.available, 0u);
  const std::set<uint32_t> unique(all.begin(), all.end());
  EXPECT_EQ(unique, (std::set<uint32_t>{0, 1, 2, 3, 4, 5, 6, 7}));
  loom_runtime_block_pool_release(&pool, all.size(), all.data());
  loom_runtime_block_pool_deinitialize(&pool);
}

TEST(BlockPoolTest, WarmReuseKeepsBackingAndHandlesEmptyExtents) {
  loom_runtime_block_pool_t pool;
  IREE_ASSERT_OK(
      loom_runtime_block_pool_initialize(1, iree_allocator_system(), &pool));
  uint32_t* const backing = pool.free_blocks;
  uint32_t block = 0;
  for (int i = 0; i < 1000; ++i) {
    loom_runtime_block_pool_acquire(&pool, 0, nullptr);
    loom_runtime_block_pool_release(&pool, 0, nullptr);
    loom_runtime_block_pool_acquire(&pool, 1, &block);
    EXPECT_EQ(block, 0u);
    EXPECT_EQ(pool.available, 0u);
    loom_runtime_block_pool_release(&pool, 1, &block);
    EXPECT_EQ(pool.available, 1u);
    EXPECT_EQ(pool.free_blocks, backing);
  }
  loom_runtime_block_pool_deinitialize(&pool);
}

TEST(BlockPoolTest, CapacityGrowthPreservesSharedIdsAndFreeOrder) {
  loom_runtime_block_pool_t pool;
  IREE_ASSERT_OK(
      loom_runtime_block_pool_initialize(0, iree_allocator_system(), &pool));
  IREE_ASSERT_OK(loom_runtime_block_pool_grow(&pool, 4));
  uint32_t held[3] = {};
  loom_runtime_block_pool_acquire(&pool, 3, held);
  loom_runtime_block_pool_retain(&pool, 1, &held[1]);
  loom_runtime_block_pool_release(&pool, 1, &held[0]);
  IREE_ASSERT_OK(loom_runtime_block_pool_grow(&pool, 129));
  EXPECT_EQ(pool.references[held[1]], 2u);
  EXPECT_EQ(pool.references[held[2]], 1u);
  std::array<uint32_t, 127> free;
  loom_runtime_block_pool_acquire(&pool, free.size(), free.data());
  EXPECT_EQ(free[0], held[0]);
  EXPECT_EQ(free[1], 3u);
  EXPECT_EQ(free[2], 4u);
  EXPECT_EQ(free.back(), 128u);
  const std::set<uint32_t> unique(free.begin(), free.end());
  EXPECT_EQ(unique.size(), free.size());
  EXPECT_EQ(unique.count(held[1]), 0u);
  EXPECT_EQ(unique.count(held[2]), 0u);
  auto* backing = pool.free_blocks;
  IREE_ASSERT_OK(loom_runtime_block_pool_grow(&pool, 129));
  EXPECT_EQ(backing, pool.free_blocks);
  loom_runtime_block_pool_deinitialize(&pool);
}

TEST(BlockPoolTest, SharedOwnersSurviveCompactionAndLastRelease) {
  loom_runtime_block_pool_t pool;
  IREE_ASSERT_OK(
      loom_runtime_block_pool_initialize(8, iree_allocator_system(), &pool));
  std::array<uint32_t, 8> blocks;
  loom_runtime_block_pool_acquire(&pool, blocks.size(), blocks.data());
  const uint32_t shared[] = {blocks[5], blocks[7]};
  loom_runtime_block_pool_retain(&pool, 2, shared);
  loom_runtime_block_pool_retain(&pool, 1, shared);
  loom_runtime_block_pool_release(&pool, 5, blocks.data());
  loom_runtime_block_pool_release(&pool, 1, &blocks[6]);
  EXPECT_EQ(pool.available, 6u);
  EXPECT_EQ(pool.references[5], 3u);
  EXPECT_EQ(pool.references[7], 2u);
  std::array<uint32_t, 8> destinations;
  EXPECT_EQ(loom_runtime_block_pool_plan_compaction(&pool, destinations.data()),
            2u);
  loom_runtime_block_pool_commit_compaction(&pool, destinations.data());
  uint32_t a = destinations[5], b = destinations[7];
  EXPECT_EQ(pool.references[a], 3u);
  EXPECT_EQ(pool.references[b], 2u);
  loom_runtime_block_pool_release(&pool, 1, &a);
  loom_runtime_block_pool_release(&pool, 1, &b);
  EXPECT_EQ(pool.available, 6u);
  EXPECT_TRUE(loom_runtime_block_pool_is_shared(&pool, a));
  EXPECT_FALSE(loom_runtime_block_pool_is_shared(&pool, b));
  loom_runtime_block_pool_release(&pool, 1, &a);
  loom_runtime_block_pool_release(&pool, 1, &b);
  EXPECT_EQ(pool.available, 7u);
  loom_runtime_block_pool_release(&pool, 1, &a);
  EXPECT_EQ(pool.available, pool.capacity);
  loom_runtime_block_pool_acquire(&pool, blocks.size(), blocks.data());
  EXPECT_EQ(std::set<uint32_t>(blocks.begin(), blocks.end()).size(), 8u);
  for (uint32_t block : blocks) {
    EXPECT_EQ(pool.references[block], 1u);
  }
  loom_runtime_block_pool_deinitialize(&pool);
}

TEST(BlockPoolTest, ColdPlacementPrefersFreeBackedIdsWithoutChangingOwners) {
  for (uint32_t mask = 0; mask < 256; ++mask) {
    SCOPED_TRACE(mask);
    loom_runtime_block_pool_t pool;
    IREE_ASSERT_OK(
        loom_runtime_block_pool_initialize(8, iree_allocator_system(), &pool));
    std::array<uint32_t, 8> blocks;
    loom_runtime_block_pool_acquire(&pool, blocks.size(), blocks.data());
    // Hold two IDs, including a shared owner, throughout free-list repair.
    const uint32_t shared = 2;
    loom_runtime_block_pool_retain(&pool, 1, &shared);
    for (uint32_t block : {0, 1, 3, 4, 6, 7}) {
      loom_runtime_block_pool_release(&pool, 1, &block);
    }
    struct Preference {
      // Source-provided backing membership, independent of logical ownership.
      uint32_t mask;
      // Each free ID must be visited once, with no owned IDs examined.
      uint32_t visited = 0;
    } preference = {mask};
    loom_runtime_block_pool_prioritize(
        &pool,
        [](void* self, uint32_t block) {
          auto* preference = static_cast<Preference*>(self);
          EXPECT_EQ(preference->visited & (1u << block), 0u);
          preference->visited |= 1u << block;
          return (preference->mask & (1u << block)) != 0;
        },
        &preference);
    EXPECT_EQ(preference.visited, 0xdbu);
    EXPECT_EQ(pool.references[2], 2u);
    EXPECT_EQ(pool.references[5], 1u);
    loom_runtime_block_pool_acquire(&pool, 6, blocks.data());
    std::set<uint32_t> acquired(blocks.begin(), blocks.begin() + 6);
    EXPECT_EQ(acquired, (std::set<uint32_t>{0, 1, 3, 4, 6, 7}));
    bool unbacked = false;
    for (size_t i = 0; i < 6; ++i) {
      const bool preferred = (mask & (1u << blocks[i])) != 0;
      EXPECT_FALSE(unbacked && preferred);
      unbacked |= !preferred;
    }
    loom_runtime_block_pool_prioritize(&pool, nullptr, nullptr);
    loom_runtime_block_pool_deinitialize(&pool);
  }
}

TEST(BlockPoolTest, CompactionMovesOnlyIntoDisjointFreeSlots) {
  // Exhaust every ownership pattern in a small pool, including all-live and
  // all-free. This checks the actual planning and commit contract, not a
  // second implementation of the planner.
  for (uint32_t mask = 0; mask < 256; ++mask) {
    SCOPED_TRACE(mask);
    loom_runtime_block_pool_t pool;
    IREE_ASSERT_OK(
        loom_runtime_block_pool_initialize(8, iree_allocator_system(), &pool));
    std::array<uint32_t, 8> blocks;
    loom_runtime_block_pool_acquire(&pool, blocks.size(), blocks.data());
    std::set<uint32_t> occupied;
    for (uint32_t i = 0; i < blocks.size(); ++i) {
      if (mask & (1u << i)) {
        occupied.insert(i);
      } else {
        loom_runtime_block_pool_release(&pool, 1, &blocks[i]);
      }
    }
    std::array<uint32_t, 8> destinations;
    const uint32_t moved =
        loom_runtime_block_pool_plan_compaction(&pool, destinations.data());
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
    loom_runtime_block_pool_commit_compaction(&pool, destinations.data());
    const uint32_t available = pool.available;
    loom_runtime_block_pool_acquire(&pool, available, blocks.data());
    for (uint32_t i = 0; i < available; ++i) {
      EXPECT_EQ(blocks[i], occupied.size() + i);
    }
    loom_runtime_block_pool_deinitialize(&pool);
  }
}

}  // namespace
