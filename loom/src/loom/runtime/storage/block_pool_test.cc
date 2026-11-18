// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/block_pool.h"

#include <array>
#include <set>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

std::set<uint32_t> Released(const loom_runtime_block_pool_t& pool) {
  std::set<uint32_t> result;
  for (uint32_t block = loom_runtime_block_pool_find_released(&pool, 0);
       block != UINT32_MAX;
       block = loom_runtime_block_pool_find_released(&pool, block + 1)) {
    result.insert(block);
  }
  return result;
}

TEST(BlockPoolTest, EmptyPoolHasNoOwnership) {
  loom_runtime_block_pool_t pool;
  IREE_ASSERT_OK(
      loom_runtime_block_pool_initialize(0, iree_allocator_system(), &pool));
  EXPECT_EQ(pool.capacity, 0u);
  EXPECT_EQ(pool.available, 0u);
  EXPECT_EQ(pool.references, nullptr);
  EXPECT_FALSE(loom_runtime_block_pool_any_owned(&pool, 0, 0));
  EXPECT_EQ(Released(pool).size(), 0u);
  loom_runtime_block_pool_acknowledge_releases(&pool);
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
  EXPECT_EQ((std::set<uint32_t>{second[2], second[3]}),
            (std::set<uint32_t>{first[2], first[3]}));
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
  uint32_t* const backing = pool.references;
  uint32_t block = 0;
  for (int i = 0; i < 1000; ++i) {
    loom_runtime_block_pool_acquire(&pool, 0, nullptr);
    loom_runtime_block_pool_release(&pool, 0, nullptr);
    loom_runtime_block_pool_acquire(&pool, 1, &block);
    EXPECT_EQ(block, 0u);
    EXPECT_EQ(pool.available, 0u);
    loom_runtime_block_pool_release(&pool, 1, &block);
    EXPECT_EQ(pool.available, 1u);
    EXPECT_EQ(pool.references, backing);
  }
  loom_runtime_block_pool_deinitialize(&pool);
}

TEST(BlockPoolTest, ReleasesFollowSharingRollbackAndReacquisition) {
  loom_runtime_block_pool_t pool;
  IREE_ASSERT_OK(
      loom_runtime_block_pool_initialize(8, iree_allocator_system(), &pool));
  uint32_t blocks[4] = {};
  loom_runtime_block_pool_acquire(&pool, 4, blocks);
  loom_runtime_block_pool_retain(&pool, 1, blocks);
  loom_runtime_block_pool_release(&pool, 4, blocks);
  EXPECT_EQ(Released(pool).size(), 3u);
  EXPECT_EQ(Released(pool), (std::set<uint32_t>{1, 2, 3}));
  EXPECT_TRUE(loom_runtime_block_pool_any_owned(&pool, 0, 1));
  EXPECT_FALSE(loom_runtime_block_pool_any_owned(&pool, 1, 7));

  // Retry before successful maintenance must observe the same candidates.
  EXPECT_EQ(Released(pool).size(), 3u);
  uint32_t replacement = 0;
  loom_runtime_block_pool_acquire(&pool, 1, &replacement);
  EXPECT_EQ(replacement, 1u);
  EXPECT_EQ(Released(pool).size(), 2u);
  IREE_ASSERT_OK(loom_runtime_block_pool_grow(&pool, 129));
  EXPECT_EQ(Released(pool).size(), 2u);
  EXPECT_EQ(Released(pool), (std::set<uint32_t>{2, 3}));
  loom_runtime_block_pool_acknowledge_releases(&pool);
  EXPECT_EQ(Released(pool).size(), 0u);

  // A denied tentative acquisition rolls back IDs into the same release set.
  uint32_t tentative[3] = {};
  loom_runtime_block_pool_acquire(&pool, 3, tentative);
  for (size_t i = 3; i; --i) {
    loom_runtime_block_pool_release(&pool, 1, &tentative[i - 1]);
  }
  EXPECT_EQ(Released(pool).size(), 3u);
  EXPECT_EQ(Released(pool), std::set<uint32_t>(tentative, tentative + 3));
  loom_runtime_block_pool_release(&pool, 1, blocks);
  loom_runtime_block_pool_release(&pool, 1, &replacement);
  EXPECT_EQ(Released(pool).size(), 5u);
  EXPECT_FALSE(loom_runtime_block_pool_any_owned(&pool, 0, pool.capacity));
  loom_runtime_block_pool_deinitialize(&pool);
}

TEST(BlockPoolTest, GrowthPreservesMembershipAcrossSummaryLevels) {
  for (uint32_t initial : {1u, 63u, 64u, 65u, 4095u, 4096u, 4097u}) {
    for (bool occupied : {false, true}) {
      SCOPED_TRACE(initial);
      SCOPED_TRACE(occupied);
      loom_runtime_block_pool_t pool;
      IREE_ASSERT_OK(loom_runtime_block_pool_initialize(
          initial, iree_allocator_system(), &pool));
      std::vector<uint32_t> blocks(initial);
      loom_runtime_block_pool_acquire(&pool, initial, blocks.data());
      const uint32_t retained = occupied ? 1 : 0;
      loom_runtime_block_pool_release(&pool, initial - retained, blocks.data());
      IREE_ASSERT_OK(loom_runtime_block_pool_grow(&pool, 262145));
      EXPECT_EQ(loom_runtime_block_pool_any_owned(&pool, 0, initial), occupied);
      EXPECT_FALSE(
          loom_runtime_block_pool_any_owned(&pool, initial, 262145 - initial));
      if (occupied) {
        EXPECT_FALSE(loom_runtime_block_pool_any_owned(&pool, 0, initial - 1));
        EXPECT_TRUE(loom_runtime_block_pool_any_owned(&pool, initial - 1, 1));
        loom_runtime_block_pool_release(&pool, 1, &blocks.back());
        EXPECT_FALSE(
            loom_runtime_block_pool_any_owned(&pool, 0, pool.capacity));
      }
      loom_runtime_block_pool_deinitialize(&pool);
    }
  }
}

TEST(BlockPoolTest, FailedGrowthPreservesOwnershipAndPendingReleases) {
  bool fail_allocation = false;
  const iree_allocator_t allocator = {
      &fail_allocation,
      [](void* self, iree_allocator_command_t command, const void* parameters,
         void** pointer) -> iree_status_t {
        if (*static_cast<bool*>(self) &&
            command != IREE_ALLOCATOR_COMMAND_FREE) {
          return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
        }
        const auto system = iree_allocator_system();
        return system.ctl(system.self, command, parameters, pointer);
      },
  };
  loom_runtime_block_pool_t pool;
  IREE_ASSERT_OK(loom_runtime_block_pool_initialize(8, allocator, &pool));
  uint32_t blocks[3] = {};
  loom_runtime_block_pool_acquire(&pool, 3, blocks);
  loom_runtime_block_pool_release(&pool, 1, &blocks[1]);
  fail_allocation = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_runtime_block_pool_grow(&pool, 4097));
  EXPECT_EQ(pool.capacity, 8u);
  EXPECT_EQ(pool.available, 6u);
  ASSERT_EQ(Released(pool).size(), 1u);
  EXPECT_EQ(*Released(pool).begin(), blocks[1]);
  EXPECT_TRUE(loom_runtime_block_pool_any_owned(&pool, 0, 1));
  EXPECT_FALSE(loom_runtime_block_pool_any_owned(&pool, 1, 1));
  EXPECT_TRUE(loom_runtime_block_pool_any_owned(&pool, 2, 1));
  // Warm ownership/query operations remain allocation-free while growth fails.
  loom_runtime_block_pool_acquire(&pool, 1, &blocks[1]);
  loom_runtime_block_pool_retain(&pool, 1, &blocks[2]);
  loom_runtime_block_pool_release(&pool, 3, blocks);
  EXPECT_TRUE(loom_runtime_block_pool_any_owned(&pool, 0, 8));
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
  auto* backing = pool.references;
  IREE_ASSERT_OK(loom_runtime_block_pool_grow(&pool, 129));
  EXPECT_EQ(backing, pool.references);
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

TEST(BlockPoolTest, PlacementFactsPreferFreeBackedIdsWithoutChangingOwners) {
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
    // Preference is established while IDs may be owned. Changes preserve the
    // exact pending release set instead of invalidating every free ID.
    for (uint32_t block = 0; block < 8; ++block) {
      loom_runtime_block_pool_set_preferred(&pool, block,
                                            (mask & (1u << block)) != 0);
    }
    EXPECT_EQ(Released(pool), (std::set<uint32_t>{0, 1, 3, 4, 6, 7}));
    loom_runtime_block_pool_acknowledge_releases(&pool);
    for (uint32_t block = 0; block < 8; ++block) {
      loom_runtime_block_pool_set_preferred(&pool, block,
                                            (mask & (1u << block)) != 0);
    }
    EXPECT_TRUE(Released(pool).empty());
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
    for (uint32_t first = 0; first <= pool.capacity; ++first) {
      for (uint32_t count = 0; count <= pool.capacity - first; ++count) {
        const auto member = occupied.lower_bound(first);
        EXPECT_EQ(loom_runtime_block_pool_any_owned(&pool, first, count),
                  member != occupied.end() && *member < first + count);
        EXPECT_EQ(loom_runtime_block_pool_find_owned(&pool, first, count),
                  member != occupied.end() && *member < first + count
                      ? *member
                      : UINT32_MAX);
      }
    }
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
    for (uint32_t first = 0; first <= pool.capacity; ++first) {
      for (uint32_t count = 0; count <= pool.capacity - first; ++count) {
        EXPECT_EQ(loom_runtime_block_pool_any_owned(&pool, first, count),
                  count != 0 && first < occupied.size());
        EXPECT_EQ(loom_runtime_block_pool_find_owned(&pool, first, count),
                  count != 0 && first < occupied.size() ? first : UINT32_MAX);
      }
    }
    EXPECT_EQ(Released(pool).size(), pool.available);
    const uint32_t available = pool.available;
    loom_runtime_block_pool_acquire(&pool, available, blocks.data());
    for (uint32_t i = 0; i < available; ++i) {
      EXPECT_EQ(blocks[i], occupied.size() + i);
    }
    loom_runtime_block_pool_deinitialize(&pool);
  }
}

}  // namespace
