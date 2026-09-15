// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/storage_lease_index.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class LowAllocationStorageLeaseIndexTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    classes_[0].alias_set_id = 7;
    classes_[1].alias_set_id = 7;
    descriptors_.reg_classes = classes_;
    descriptors_.reg_class_count = IREE_ARRAYSIZE(classes_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_low_allocation_storage_lease_t Lease(
      uint32_t start, uint32_t end, uint32_t location = 10, uint32_t width = 1,
      uint16_t reg_class = 0,
      loom_low_allocation_location_kind_t kind =
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    loom_low_allocation_storage_lease_t lease = {};
    lease.start_point = start;
    lease.end_point = end;
    lease.location_base = location;
    lease.location_count = width;
    lease.descriptor_reg_class_id = reg_class;
    lease.location_kind = kind;
    lease.release_action_index = LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE;
    return lease;
  }

  void Initialize(std::vector<loom_low_allocation_storage_lease_t>& leases,
                  loom_low_allocation_storage_lease_unit_index_t* index) {
    size_t units = 0;
    for (const auto& lease : leases) units += lease.location_count;
    IREE_ASSERT_OK(loom_low_allocation_storage_lease_unit_index_initialize(
        index, leases.data(), leases.size(), units, units, &arena_));
  }

  std::vector<uint32_t> Query(
      const loom_low_allocation_storage_lease_unit_index_t& index,
      uint32_t minimum_end, uint32_t maximum_start, uint32_t location = 10,
      uint32_t width = 1, uint16_t reg_class = 0,
      const loom_low_allocation_storage_lease_selection_t* selection = nullptr,
      loom_low_allocation_location_kind_t kind =
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    loom_low_allocation_storage_lease_unit_query_t query;
    loom_low_allocation_storage_lease_unit_query_initialize(
        &index, &descriptors_, reg_class, kind, location, width, minimum_end,
        (uint64_t)maximum_start + 1u, selection, &query);
    std::vector<uint32_t> result;
    uint32_t lease_index = 0;
    while (loom_low_allocation_storage_lease_unit_query_next(&query,
                                                             &lease_index)) {
      result.push_back(lease_index);
    }
    std::sort(result.begin(), result.end());
    return result;
  }

  // Arena backing retained for the duration of each API test.
  iree_arena_block_pool_t pool_;
  // Owner of index and selection storage.
  iree_arena_allocator_t arena_;
  // Two aliasing register classes and one independent class.
  loom_low_reg_class_t classes_[3] = {};
  // Descriptor view of the test register classes.
  loom_low_descriptor_set_t descriptors_ = {};
};

TEST_F(LowAllocationStorageLeaseIndexTest, QueriesAliasingPhysicalLeaseUnits) {
  std::vector<loom_low_allocation_storage_lease_t> leases = {
      Lease(1, 20, 10, 3), Lease(2, 20, 11, 2, 1), Lease(3, 20, 11, 1, 2),
      Lease(4, 20, 11, 1, 1, LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID)};
  loom_low_allocation_storage_lease_unit_index_t index;
  Initialize(leases, &index);
  for (uint32_t i = 0; i < leases.size(); ++i) {
    loom_low_allocation_storage_lease_unit_index_insert(&index, &descriptors_,
                                                        i);
  }
  EXPECT_EQ((std::vector<uint32_t>{0, 1}),
            Query(index, 0, UINT32_MAX, 11, 1, 1));
  EXPECT_EQ((std::vector<uint32_t>{0, 0, 0, 1, 1}),
            Query(index, 0, UINT32_MAX, 10, 3, 1));
  EXPECT_EQ((std::vector<uint32_t>{2}), Query(index, 0, UINT32_MAX, 11, 1, 2));
  EXPECT_EQ((std::vector<uint32_t>{3}),
            Query(index, 0, UINT32_MAX, 11, 1, 1, nullptr,
                  LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID));
  EXPECT_TRUE(Query(index, 0, UINT32_MAX, 20, 4).empty());
}

TEST_F(LowAllocationStorageLeaseIndexTest,
       PreservesEndpointAndReleaseSemantics) {
  std::vector<loom_low_allocation_storage_lease_t> leases = {
      Lease(4, 8), Lease(8, 12), Lease(8, 16)};
  loom_low_allocation_storage_lease_unit_index_t index;
  Initialize(leases, &index);
  for (uint32_t i = 0; i < leases.size(); ++i) {
    loom_low_allocation_storage_lease_unit_index_insert(&index, &descriptors_,
                                                        i);
  }
  EXPECT_EQ((std::vector<uint32_t>{0, 1, 2}), Query(index, 8, 8));
  EXPECT_EQ((std::vector<uint32_t>{1, 2}), Query(index, 9, 8));
  leases[2].end_point = 9;
  leases[2].release_action_index = 0;
  loom_low_allocation_storage_lease_unit_index_update(&index, 2);
  EXPECT_EQ((std::vector<uint32_t>{1}), Query(index, 10, 10));
  EXPECT_EQ((std::vector<uint32_t>{1, 2}), Query(index, 9, 9));
  EXPECT_EQ((std::vector<uint32_t>{0, 1, 2}), Query(index, 1, UINT32_MAX));
}

TEST_F(LowAllocationStorageLeaseIndexTest,
       RetainsIndependentIncomingSelections) {
  std::vector<loom_low_allocation_storage_lease_t> leases = {
      Lease(4, 8, 10, 2), Lease(40, 48, 10, 2), Lease(90, 99, 11, 1)};
  loom_low_allocation_storage_lease_unit_index_t index;
  Initialize(leases, &index);
  for (uint32_t i = 0; i < leases.size(); ++i) {
    loom_low_allocation_storage_lease_unit_index_insert(&index, &descriptors_,
                                                        i);
  }
  loom_low_allocation_storage_lease_selection_t incoming, alternate;
  IREE_ASSERT_OK(loom_low_allocation_storage_lease_selection_initialize(
      &index, &arena_, &incoming));
  IREE_ASSERT_OK(loom_low_allocation_storage_lease_selection_initialize(
      &index, &arena_, &alternate));
  loom_low_allocation_storage_lease_selection_set_active(&incoming, 0, true);
  loom_low_allocation_storage_lease_selection_set_active(&incoming, 2, true);
  loom_low_allocation_storage_lease_selection_set_active(&alternate, 1, true);
  EXPECT_EQ((std::vector<uint32_t>{0, 0, 1, 1, 2}),
            Query(index, 44, 44, 10, 2, 0, &incoming));
  EXPECT_EQ((std::vector<uint32_t>{1, 1}),
            Query(index, 20, 20, 10, 2, 0, &alternate));
  loom_low_allocation_storage_lease_selection_set_active(&incoming, 0, false);
  EXPECT_EQ((std::vector<uint32_t>{1, 1, 2}),
            Query(index, 44, 44, 10, 2, 0, &incoming));
  loom_low_allocation_storage_lease_selection_set_active(&incoming, 2, false);
  EXPECT_TRUE(Query(index, 20, 20, 10, 2, 0, &incoming).empty());
}

TEST_F(LowAllocationStorageLeaseIndexTest, PrunesShuffledDisjointHistory) {
  constexpr uint32_t kLeaseCount = 2048;
  std::vector<loom_low_allocation_storage_lease_t> leases;
  for (uint32_t i = 0; i < kLeaseCount; ++i) {
    leases.push_back(Lease(i * 4, i * 4 + 2));
  }
  loom_low_allocation_storage_lease_unit_index_t index;
  Initialize(leases, &index);
  std::vector<uint32_t> order(kLeaseCount);
  std::iota(order.begin(), order.end(), 0u);
  std::mt19937 generator(12345);
  std::shuffle(order.begin(), order.end(), generator);
  for (uint32_t i : order) {
    loom_low_allocation_storage_lease_unit_index_insert(&index, &descriptors_,
                                                        i);
  }
  for (uint32_t i = 0; i < kLeaseCount; ++i) {
    EXPECT_EQ((std::vector<uint32_t>{i}), Query(index, i * 4 + 1, i * 4 + 1));
    EXPECT_TRUE(Query(index, i * 4 + 3, i * 4 + 3).empty());
  }
}

TEST_F(LowAllocationStorageLeaseIndexTest, ReservesForDistinctPhysicalUnits) {
  constexpr uint32_t kLeaseCount = 256;
  std::vector<loom_low_allocation_storage_lease_t> leases;
  for (uint32_t i = 0; i < kLeaseCount; ++i) {
    leases.push_back(Lease(i * 2, i * 2 + 1, 10, 1, 0,
                           i % 2 == 0
                               ? LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER
                               : LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID));
  }
  loom_low_allocation_storage_lease_unit_index_t index;
  IREE_ASSERT_OK(loom_low_allocation_storage_lease_unit_index_initialize(
      &index, leases.data(), leases.size(), kLeaseCount,
      /*distinct_unit_capacity=*/2, &arena_));
  for (uint32_t i = 0; i < kLeaseCount; ++i) {
    loom_low_allocation_storage_lease_unit_index_insert(&index, &descriptors_,
                                                        i);
  }
  EXPECT_EQ(index.node_capacity, 2 * kLeaseCount + 2);
  EXPECT_EQ(index.node_count, 2 * kLeaseCount);
  for (uint32_t i = 0; i < kLeaseCount; ++i) {
    EXPECT_EQ(
        (std::vector<uint32_t>{i}),
        Query(index, i * 2, i * 2, 10, 1, 0, nullptr, leases[i].location_kind));
  }
}

TEST_F(LowAllocationStorageLeaseIndexTest,
       PrunesShortenedDuplicateStartPoints) {
  constexpr uint32_t kLeaseCount = 1024;
  std::vector<loom_low_allocation_storage_lease_t> leases(kLeaseCount,
                                                          Lease(1, 100));
  loom_low_allocation_storage_lease_unit_index_t index;
  Initialize(leases, &index);
  for (uint32_t i = 0; i < kLeaseCount; ++i) {
    loom_low_allocation_storage_lease_unit_index_insert(&index, &descriptors_,
                                                        i);
  }
  for (uint32_t i = 0; i + 1 < kLeaseCount; ++i) {
    leases[i].end_point = 20;
    leases[i].release_action_index = i;
    loom_low_allocation_storage_lease_unit_index_update(&index, i);
  }
  EXPECT_EQ((std::vector<uint32_t>{kLeaseCount - 1}), Query(index, 21, 21));
  std::vector<uint32_t> expected(kLeaseCount);
  std::iota(expected.begin(), expected.end(), 0u);
  EXPECT_EQ(expected, Query(index, 2, 2));
}

TEST_F(LowAllocationStorageLeaseIndexTest, HandlesHighKeysAndEmptyIndexes) {
  std::vector<loom_low_allocation_storage_lease_t> leases;
  loom_low_allocation_storage_lease_unit_index_t empty;
  Initialize(leases, &empty);
  EXPECT_TRUE(Query(empty, 0, UINT32_MAX).empty());
  leases = {Lease(0, 1, 0), Lease(UINT32_MAX - 1, UINT32_MAX, UINT32_MAX),
            Lease(0x80000000u, 0x80000001u, 0x80000000u)};
  loom_low_allocation_storage_lease_unit_index_t index;
  Initialize(leases, &index);
  for (uint32_t i = 0; i < leases.size(); ++i) {
    loom_low_allocation_storage_lease_unit_index_insert(&index, &descriptors_,
                                                        i);
  }
  EXPECT_EQ((std::vector<uint32_t>{0}), Query(index, 0, 0, 0));
  EXPECT_EQ((std::vector<uint32_t>{1}),
            Query(index, UINT32_MAX, UINT32_MAX, UINT32_MAX));
  EXPECT_EQ((std::vector<uint32_t>{2}),
            Query(index, 0x80000001u, 0x80000001u, 0x80000000u));
}

TEST_F(LowAllocationStorageLeaseIndexTest, PreservesEmptyBoundaryQueries) {
  std::vector<loom_low_allocation_storage_lease_t> leases = {
      Lease(0, UINT32_MAX)};
  loom_low_allocation_storage_lease_unit_index_t index;
  Initialize(leases, &index);
  loom_low_allocation_storage_lease_unit_index_insert(&index, &descriptors_, 0);
  for (uint32_t point : {0u, UINT32_MAX}) {
    loom_low_allocation_storage_lease_unit_query_t query;
    loom_low_allocation_storage_lease_unit_query_initialize(
        &index, &descriptors_, 0,
        LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, 10, 1,
        (uint64_t)point + 1u, point,
        /*selection=*/nullptr, &query);
    uint32_t lease_index = 0;
    EXPECT_FALSE(loom_low_allocation_storage_lease_unit_query_next(
        &query, &lease_index));
  }
}

TEST_F(LowAllocationStorageLeaseIndexTest, RetiresMembersDuringIteration) {
  std::vector<loom_low_allocation_storage_lease_t> leases;
  for (uint32_t i = 0; i < 64; ++i) leases.push_back(Lease(i, i + 1));
  loom_low_allocation_storage_lease_unit_index_t index;
  Initialize(leases, &index);
  for (uint32_t i = 0; i < leases.size(); ++i) {
    loom_low_allocation_storage_lease_unit_index_insert(&index, &descriptors_,
                                                        i);
  }
  loom_low_allocation_storage_lease_selection_t incoming;
  IREE_ASSERT_OK(loom_low_allocation_storage_lease_selection_initialize(
      &index, &arena_, &incoming));
  for (uint32_t i = 0; i < leases.size(); ++i) {
    loom_low_allocation_storage_lease_selection_set_active(&incoming, i, true);
  }
  loom_low_allocation_storage_lease_unit_query_t query;
  loom_low_allocation_storage_lease_unit_query_initialize(
      &index, &descriptors_, 0, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      10, 1, 100, 101, &incoming, &query);
  uint32_t lease_index = 0;
  ASSERT_TRUE(
      loom_low_allocation_storage_lease_unit_query_next(&query, &lease_index));
  // A full incoming drain removes candidates already on the query's stack.
  for (uint32_t i = 0; i < leases.size(); ++i) {
    loom_low_allocation_storage_lease_selection_set_active(&incoming, i, false);
  }
  EXPECT_FALSE(
      loom_low_allocation_storage_lease_unit_query_next(&query, &lease_index));
}

TEST_F(LowAllocationStorageLeaseIndexTest, ReleasesLeasesDuringIteration) {
  std::vector<loom_low_allocation_storage_lease_t> leases(64, Lease(0, 100));
  loom_low_allocation_storage_lease_unit_index_t index;
  Initialize(leases, &index);
  for (uint32_t i = 0; i < leases.size(); ++i) {
    loom_low_allocation_storage_lease_unit_index_insert(&index, &descriptors_,
                                                        i);
  }
  loom_low_allocation_storage_lease_unit_query_t query;
  loom_low_allocation_storage_lease_unit_query_initialize(
      &index, &descriptors_, 0, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      10, 1, 2, 10,
      /*selection=*/nullptr, &query);
  uint32_t lease_index = 0;
  uint32_t count = 0;
  while (
      loom_low_allocation_storage_lease_unit_query_next(&query, &lease_index)) {
    leases[lease_index].release_action_index = count++;
    leases[lease_index].end_point = 1;
    loom_low_allocation_storage_lease_unit_index_update(&index, lease_index);
  }
  EXPECT_EQ(count, leases.size());
  EXPECT_TRUE(Query(index, 2, 10).empty());
}

TEST_F(LowAllocationStorageLeaseIndexTest,
       MatchesIndependentMutableOverlapModel) {
  constexpr uint32_t kLeaseCount = 96;
  std::mt19937 generator(67890);
  std::vector<loom_low_allocation_storage_lease_t> leases;
  for (uint32_t i = 0; i < kLeaseCount; ++i) {
    const uint32_t start = generator() % 64;
    leases.push_back(Lease(start, start + 1 + generator() % 32,
                           10 + generator() % 8, 1 + generator() % 3,
                           generator() % 3));
  }
  loom_low_allocation_storage_lease_unit_index_t index;
  Initialize(leases, &index);
  std::vector<uint32_t> order(kLeaseCount);
  std::iota(order.begin(), order.end(), 0u);
  std::shuffle(order.begin(), order.end(), generator);
  for (uint32_t i : order) {
    loom_low_allocation_storage_lease_unit_index_insert(&index, &descriptors_,
                                                        i);
  }
  loom_low_allocation_storage_lease_selection_t incoming;
  IREE_ASSERT_OK(loom_low_allocation_storage_lease_selection_initialize(
      &index, &arena_, &incoming));
  std::vector<bool> active(kLeaseCount);
  for (uint32_t iteration = 0; iteration < 256; ++iteration) {
    const uint32_t changed = generator() % kLeaseCount;
    active[changed] = !active[changed];
    loom_low_allocation_storage_lease_selection_set_active(&incoming, changed,
                                                           active[changed]);
    if (iteration % 2 == 0) {
      leases[changed].end_point = leases[changed].start_point + 1;
      leases[changed].release_action_index = iteration;
      loom_low_allocation_storage_lease_unit_index_update(&index, changed);
    }
    const uint32_t point = 1 + generator() % 100;
    const uint32_t location = 9 + generator() % 12;
    const uint16_t reg_class = generator() % 3;
    std::vector<uint32_t> expected;
    for (uint32_t i = 0; i < kLeaseCount; ++i) {
      const auto& lease = leases[i];
      const bool temporal =
          lease.start_point <= point && lease.end_point >= point;
      if ((!temporal && !active[i]) ||
          (reg_class == 2) != (lease.descriptor_reg_class_id == 2)) {
        continue;
      }
      for (uint32_t unit = location; unit < location + 3; ++unit) {
        if (lease.location_base <= unit &&
            unit < lease.location_base + lease.location_count) {
          expected.push_back(i);
        }
      }
    }
    EXPECT_EQ(expected,
              Query(index, point, point, location, 3, reg_class, &incoming));
  }
}

}  // namespace
}  // namespace loom
