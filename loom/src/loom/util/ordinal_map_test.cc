// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/ordinal_map.h"

#include <algorithm>
#include <random>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class OrdinalMapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32768, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_ordinal_map_initialize(&arena_, &projection_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Shared backing allocation pool for each test's projection.
  iree_arena_block_pool_t pool_;
  // Retained projection storage, including abandoned growth buffers.
  iree_arena_allocator_t arena_;
  // Reached-only map under test.
  loom_ordinal_map_t projection_;
};

TEST_F(OrdinalMapTest, EmptyProjectionMisses) {
  uint32_t target_id = 0;
  EXPECT_FALSE(loom_ordinal_map_lookup(&projection_, 3,
                                       /*source_ordinal=*/0, &target_id));
}

TEST_F(OrdinalMapTest, DomainsAreIndependent) {
  for (uint8_t domain = 0; domain < 8; ++domain) {
    IREE_ASSERT_OK(
        loom_ordinal_map_insert(&projection_, domain, 7, UINT32_MAX - domain));
  }
  for (uint8_t domain = 0; domain < 8; ++domain) {
    uint32_t target_id = 0;
    EXPECT_TRUE(loom_ordinal_map_lookup(&projection_, domain, 7, &target_id));
    EXPECT_EQ(target_id, UINT32_MAX - domain);
  }
}

TEST_F(OrdinalMapTest, GrowthPreserves4096ReachedFacts) {
  constexpr uint32_t kCount = 4096;
  for (uint32_t i = 0; i < kCount; ++i) {
    const auto domain = static_cast<uint8_t>(i % 8u);
    IREE_ASSERT_OK(
        loom_ordinal_map_insert(&projection_, domain, i, kCount - i));
  }

  EXPECT_EQ(projection_.buckets.count, kCount);
  EXPECT_LE(arena_.used_allocation_size, 192u * kCount);
  for (uint32_t i = 0; i < kCount; ++i) {
    const auto domain = static_cast<uint8_t>(i % 8u);
    uint32_t target_id = 0;
    ASSERT_TRUE(loom_ordinal_map_lookup(&projection_, domain, i, &target_id));
    EXPECT_EQ(target_id, kCount - i);
  }
}

TEST_F(OrdinalMapTest, SupportsMaximumPackedIdentities) {
  constexpr uint32_t kMaximum = (UINT32_C(1) << 24) - 1;
  IREE_ASSERT_OK(
      loom_ordinal_map_insert(&projection_, 7, kMaximum, UINT32_MAX));
  uint32_t target_id = 0;
  EXPECT_TRUE(loom_ordinal_map_lookup(&projection_, 7, kMaximum, &target_id));
  EXPECT_EQ(target_id, UINT32_MAX);
}

TEST_F(OrdinalMapTest, IdenticalInsertionIsStable) {
  IREE_ASSERT_OK(loom_ordinal_map_insert(&projection_, 3, 23, 42));
  EXPECT_EQ(arena_.used_allocation_size, 0u);
  const uint32_t capacity = projection_.buckets.capacity;
  IREE_ASSERT_OK(loom_ordinal_map_insert(&projection_, 3, 23, 42));
  EXPECT_EQ(projection_.buckets.count, 1u);
  EXPECT_EQ(projection_.buckets.capacity, capacity);
  EXPECT_EQ(arena_.used_allocation_size, 0u);
}

TEST_F(OrdinalMapTest, SparseOrdinalsRemainCompactInEveryOrder) {
  constexpr uint32_t kCount = 2048;
  std::vector<uint32_t> ordinals(kCount);
  for (uint32_t i = 0; i < kCount; ++i) {
    ordinals[i] = i;
  }
  std::mt19937 random(17);
  for (uint32_t order = 0; order < 3; ++order) {
    SCOPED_TRACE(order);
    iree_arena_reset(&arena_);
    loom_ordinal_map_initialize(&arena_, &projection_);
    if (order == 1) {
      std::reverse(ordinals.begin(), ordinals.end());
    } else if (order == 2) {
      std::shuffle(ordinals.begin(), ordinals.end(), random);
    }
    for (uint32_t i : ordinals) {
      const auto domain = static_cast<uint8_t>(i % 8);
      IREE_ASSERT_OK(
          loom_ordinal_map_insert(&projection_, domain, i * 8192, kCount - i));
    }
    EXPECT_EQ(projection_.buckets.count, kCount);
    EXPECT_LE(arena_.used_allocation_size, 192u * kCount);
    for (uint32_t i = 0; i < kCount; ++i) {
      const auto domain = static_cast<uint8_t>(i % 8);
      uint32_t target = 0;
      ASSERT_TRUE(
          loom_ordinal_map_lookup(&projection_, domain, i * 8192, &target));
      EXPECT_EQ(target, kCount - i);
      EXPECT_FALSE(
          loom_ordinal_map_lookup(&projection_, domain, i * 8192 + 1, &target));
    }
  }
}

TEST_F(OrdinalMapTest, SplitsEveryPrefixAndNibblePosition) {
  const auto type = 3;
  for (uint32_t shift = 0; shift < 24; shift += 4) {
    for (uint32_t nibble = 1; nibble < 16; ++nibble) {
      IREE_ASSERT_OK(loom_ordinal_map_insert(
          &projection_, type, nibble << shift, shift * 16 + nibble));
    }
  }
  for (uint32_t shift = 0; shift < 24; shift += 4) {
    for (uint32_t nibble = 1; nibble < 16; ++nibble) {
      uint32_t target = 0;
      ASSERT_TRUE(loom_ordinal_map_lookup(&projection_, type, nibble << shift,
                                          &target));
      EXPECT_EQ(target, shift * 16 + nibble);
    }
  }
  uint32_t target = 0;
  EXPECT_FALSE(loom_ordinal_map_lookup(&projection_, type, 0, &target));
  EXPECT_FALSE(loom_ordinal_map_lookup(&projection_, type, 0x101, &target));
}

TEST_F(OrdinalMapTest, NestedCollisionsPreserveZeroAndMaximumTargets) {
  // All ordinals share the low bucket bits. Their distinguishing nibbles nest
  // beneath the domain branch, including the packed all-zero identity.
  constexpr uint32_t kOrdinals[] = {0, 0x10, 0x100, 0x1000, 0x10000, 0x100000};
  const auto symbol = 0;
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(kOrdinals); ++i) {
    IREE_ASSERT_OK(
        loom_ordinal_map_insert(&projection_, symbol, kOrdinals[i], i));
  }
  constexpr uint32_t kMaximumTarget = UINT32_MAX;
  IREE_ASSERT_OK(loom_ordinal_map_insert(&projection_, 4, 0, kMaximumTarget));
  const iree_host_size_t used_bytes = arena_.used_allocation_size;
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(kOrdinals); ++i) {
    uint32_t target = UINT32_MAX;
    ASSERT_TRUE(
        loom_ordinal_map_lookup(&projection_, symbol, kOrdinals[i], &target));
    EXPECT_EQ(target, i);
    IREE_ASSERT_OK(
        loom_ordinal_map_insert(&projection_, symbol, kOrdinals[i], i));
  }
  EXPECT_EQ(arena_.used_allocation_size, used_bytes);
  uint32_t target = 0;
  ASSERT_TRUE(loom_ordinal_map_lookup(&projection_, 4, 0, &target));
  EXPECT_EQ(target, kMaximumTarget);
  EXPECT_FALSE(loom_ordinal_map_lookup(&projection_, symbol, 0x20, &target));
}

TEST(OrdinalMapAllocationTest, FailedGrowthPreservesMappingsAndCanRetry) {
  struct AllocationBudget {
    // Number of successful backing allocations allowed before failure.
    uint32_t remaining;
    static iree_status_t Allocate(void* user_data,
                                  iree_allocator_command_t command,
                                  const void* parameters, void** pointer) {
      auto* budget = static_cast<AllocationBudget*>(user_data);
      if (command != IREE_ALLOCATOR_COMMAND_FREE) {
        if (budget->remaining == 0) {
          return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
        }
        --budget->remaining;
      }
      const iree_allocator_t system = iree_allocator_system();
      return system.ctl(system.self, command, parameters, pointer);
    }
  };
  for (uint32_t budget_count = 0; budget_count < 8; ++budget_count) {
    SCOPED_TRACE(budget_count);
    AllocationBudget budget{budget_count};
    iree_arena_block_pool_t pool;
    iree_arena_block_pool_initialize(256, {&budget, AllocationBudget::Allocate},
                                     &pool);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&pool, &arena);
    loom_ordinal_map_t projection;
    loom_ordinal_map_initialize(&arena, &projection);
    const auto type = 3;
    uint32_t count = 0;
    iree_status_t status = iree_ok_status();
    while (count < 4096 && iree_status_is_ok(status)) {
      status = loom_ordinal_map_insert(&projection, type, count * 4096, count);
      if (iree_status_is_ok(status)) {
        ++count;
      }
    }
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    EXPECT_EQ(projection.buckets.count, count);
    for (uint32_t i = 0; i < count; ++i) {
      uint32_t target = UINT32_MAX;
      EXPECT_TRUE(
          loom_ordinal_map_lookup(&projection, type, i * 4096, &target));
      EXPECT_EQ(target, i);
    }
    // Restore backing allocation after the injected failure. A bucket-array
    // rebuild can allocate both its roots and multiple collision branches.
    budget.remaining = UINT32_MAX;
    IREE_EXPECT_OK(
        loom_ordinal_map_insert(&projection, type, count * 4096, count));
    uint32_t target = UINT32_MAX;
    EXPECT_TRUE(
        loom_ordinal_map_lookup(&projection, type, count * 4096, &target));
    EXPECT_EQ(target, count);
    iree_arena_deinitialize(&arena);
    iree_arena_block_pool_deinitialize(&pool);
  }
}

}  // namespace
