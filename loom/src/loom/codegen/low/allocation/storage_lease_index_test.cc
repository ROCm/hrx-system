// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/storage_lease_index.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <utility>
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
    loom_low_allocation_storage_lease_t lease =
        {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment order
             // differs from declaration order.
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
    for (const auto& lease : leases) {
      units += lease.location_count;
    }
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

  void RefreshAvailability(
      loom_low_allocation_storage_lease_unit_index_t* index,
      const std::vector<loom_low_allocation_storage_lease_t>& leases,
      uint32_t start_point) {
    for (uint32_t i = 0; i < leases.size(); ++i) {
      loom_low_allocation_storage_lease_unit_index_refresh_availability(
          index, &descriptors_, i, start_point);
    }
  }

  bool FindNextAvailable(
      const loom_low_allocation_storage_lease_unit_index_t& index,
      uint32_t candidate_end_point,
      loom_low_allocation_storage_lease_conflict_class_t conflict_class,
      uint32_t minimum_base, uint32_t maximum_base, uint32_t* out_base,
      uint16_t reg_class = 0,
      loom_low_allocation_location_kind_t kind =
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    return loom_low_allocation_storage_lease_unit_index_find_next_available_location(
        &index, &descriptors_, reg_class, kind, candidate_end_point,
        conflict_class, minimum_base, maximum_base, out_base);
  }

  bool FindPreviousAvailable(
      const loom_low_allocation_storage_lease_unit_index_t& index,
      uint32_t candidate_end_point,
      loom_low_allocation_storage_lease_conflict_class_t conflict_class,
      uint32_t minimum_base, uint32_t maximum_base, uint32_t* out_base,
      uint16_t reg_class = 0,
      loom_low_allocation_location_kind_t kind =
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    return loom_low_allocation_storage_lease_unit_index_find_previous_available_location(
        &index, &descriptors_, reg_class, kind, candidate_end_point,
        conflict_class, minimum_base, maximum_base, out_base);
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
                                                        i, 0);
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
                                                        i, 0);
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
       FindsGapsAcrossFutureStartsExpirationsAndPressure) {
  std::vector<loom_low_allocation_storage_lease_t> leases = {
      Lease(/*start=*/1, /*end=*/20, /*location=*/10),
      Lease(/*start=*/8, /*end=*/30, /*location=*/11),
      Lease(/*start=*/15, /*end=*/40, /*location=*/12),
      Lease(/*start=*/2, /*end=*/5, /*location=*/14),
  };
  loom_low_allocation_storage_lease_unit_index_t index;
  Initialize(leases, &index);
  for (uint32_t i = 0; i < leases.size(); ++i) {
    const loom_low_storage_lease_flags_t flags =
        i == 1 ? LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_FOR_PRESSURE : 0;
    loom_low_allocation_storage_lease_unit_index_insert(&index, &descriptors_,
                                                        i, flags);
  }
  RefreshAvailability(&index, leases, /*start_point=*/0);

  uint32_t base = UINT32_MAX;
  EXPECT_TRUE(FindNextAvailable(index, /*candidate_end_point=*/10,
                                LOOM_LOW_ALLOCATION_STORAGE_LEASE_CONFLICT_ALL,
                                10, 14, &base));
  EXPECT_EQ(base, 12u);  // The lease at 12 starts after this candidate ends.
  EXPECT_TRUE(FindPreviousAvailable(
      index, /*candidate_end_point=*/10,
      LOOM_LOW_ALLOCATION_STORAGE_LEASE_CONFLICT_ALL, 10, 14, &base));
  EXPECT_EQ(base, 13u);
  EXPECT_TRUE(FindNextAvailable(
      index, /*candidate_end_point=*/10,
      LOOM_LOW_ALLOCATION_STORAGE_LEASE_CONFLICT_NON_PRESSURE, 10, 14, &base));
  EXPECT_EQ(base, 11u);  // Pressure policy leaves location 11 to full legality.

  RefreshAvailability(&index, leases, /*start_point=*/10);
  EXPECT_TRUE(FindNextAvailable(index, /*candidate_end_point=*/20,
                                LOOM_LOW_ALLOCATION_STORAGE_LEASE_CONFLICT_ALL,
                                10, 14, &base));
  EXPECT_EQ(base, 13u);
  EXPECT_TRUE(FindPreviousAvailable(
      index, /*candidate_end_point=*/20,
      LOOM_LOW_ALLOCATION_STORAGE_LEASE_CONFLICT_ALL, 10, 14, &base));
  EXPECT_EQ(base, 14u);
  EXPECT_FALSE(FindPreviousAvailable(
      index, /*candidate_end_point=*/20,
      LOOM_LOW_ALLOCATION_STORAGE_LEASE_CONFLICT_ALL, 10, 12, &base));

  leases[2].end_point = 12;
  loom_low_allocation_storage_lease_unit_index_update(&index, 2);
  RefreshAvailability(&index, leases, /*start_point=*/12);
  EXPECT_TRUE(FindNextAvailable(index, /*candidate_end_point=*/20,
                                LOOM_LOW_ALLOCATION_STORAGE_LEASE_CONFLICT_ALL,
                                10, 12, &base));
  EXPECT_EQ(base, 12u);

  RefreshAvailability(&index, leases, /*start_point=*/20);
  EXPECT_TRUE(FindNextAvailable(index, /*candidate_end_point=*/25,
                                LOOM_LOW_ALLOCATION_STORAGE_LEASE_CONFLICT_ALL,
                                10, 12, &base));
  EXPECT_EQ(base, 10u);
}

TEST_F(LowAllocationStorageLeaseIndexTest,
       OrderedAvailabilitySharesAliasesAndSeparatesKinds) {
  std::vector<loom_low_allocation_storage_lease_t> leases = {
      Lease(1, 20, 7, 1, /*reg_class=*/0),
      Lease(1, 20, 8, 1, /*reg_class=*/2),
      Lease(1, 20, 9, 1, /*reg_class=*/1,
            LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID),
  };
  loom_low_allocation_storage_lease_unit_index_t index;
  Initialize(leases, &index);
  for (uint32_t i = 0; i < leases.size(); ++i) {
    loom_low_allocation_storage_lease_unit_index_insert(&index, &descriptors_,
                                                        i, 0);
  }
  RefreshAvailability(&index, leases, /*start_point=*/2);

  uint32_t base = UINT32_MAX;
  EXPECT_TRUE(FindNextAvailable(index, /*candidate_end_point=*/3,
                                LOOM_LOW_ALLOCATION_STORAGE_LEASE_CONFLICT_ALL,
                                7, 9, &base,
                                /*reg_class=*/1));
  EXPECT_EQ(base, 8u);
  EXPECT_TRUE(FindNextAvailable(index, /*candidate_end_point=*/3,
                                LOOM_LOW_ALLOCATION_STORAGE_LEASE_CONFLICT_ALL,
                                7, 9, &base,
                                /*reg_class=*/2));
  EXPECT_EQ(base, 7u);
  EXPECT_TRUE(FindNextAvailable(
      index, /*candidate_end_point=*/3,
      LOOM_LOW_ALLOCATION_STORAGE_LEASE_CONFLICT_ALL, 8, 10, &base,
      /*reg_class=*/1, LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID));
  EXPECT_EQ(base, 8u);
}

TEST_F(LowAllocationStorageLeaseIndexTest,
       OrderedAvailabilityMatchesIndependentModel) {
  std::vector<loom_low_allocation_storage_lease_t> leases = {
      Lease(0, 9, 0, 3, 0),
      Lease(7, 28, 0, 1, 1),
      Lease(2, 12, 3, 4, 1),
      Lease(15, 35, 4, 5, 0),
      Lease(1, 30, 6, 1, 0),
      Lease(4, 18, 7, 3, 0),
      Lease(10, 40, 10, 2, 0),
      Lease(3, 22, 2, 5, 2),
      Lease(5, 25, 5, 4, 1, LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID),
      Lease(12, 32, 13, 1, 0),
  };
  const std::vector<loom_low_storage_lease_flags_t> flags = {
      0, LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_FOR_PRESSURE,
      0, LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_FOR_PRESSURE,
      0, LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_FOR_PRESSURE,
      0, 0,
      0, 0,
  };
  loom_low_allocation_storage_lease_unit_index_t index;
  Initialize(leases, &index);
  for (uint32_t i : {6u, 1u, 8u, 3u, 0u, 9u, 4u, 7u, 2u, 5u}) {
    loom_low_allocation_storage_lease_unit_index_insert(&index, &descriptors_,
                                                        i, flags[i]);
  }

  const auto model_conflicts =
      [&](uint32_t start_point, uint32_t end_point,
          loom_low_allocation_storage_lease_conflict_class_t conflict_class,
          uint16_t reg_class, loom_low_allocation_location_kind_t kind,
          uint32_t location) {
        const uint32_t storage_key =
            loom_low_reg_class_storage_key(&descriptors_, reg_class);
        for (uint32_t i = 0; i < leases.size(); ++i) {
          const auto& lease = leases[i];
          if (conflict_class ==
                  LOOM_LOW_ALLOCATION_STORAGE_LEASE_CONFLICT_NON_PRESSURE &&
              iree_any_bit_set(
                  flags[i], LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_FOR_PRESSURE)) {
            continue;
          }
          if (lease.end_point <= start_point ||
              lease.start_point >= end_point || lease.location_kind != kind ||
              loom_low_reg_class_storage_key(&descriptors_,
                                             lease.descriptor_reg_class_id) !=
                  storage_key ||
              location < lease.location_base ||
              location - lease.location_base >= lease.location_count) {
            continue;
          }
          return true;
        }
        return false;
      };
  const std::vector<std::pair<uint32_t, uint32_t>> ranges = {
      {0, 0}, {0, 15}, {2, 8}, {5, 13}, {11, 20}};
  for (uint32_t start_point = 0; start_point <= 30; ++start_point) {
    if (start_point == 10) {
      leases[6].end_point = 10;
      loom_low_allocation_storage_lease_unit_index_update(&index, 6);
    }
    RefreshAvailability(&index, leases, start_point);
    for (uint32_t duration : {1u, 4u, 11u}) {
      const uint32_t end_point = start_point + duration;
      for (const auto conflict_class : {
               LOOM_LOW_ALLOCATION_STORAGE_LEASE_CONFLICT_ALL,
               LOOM_LOW_ALLOCATION_STORAGE_LEASE_CONFLICT_NON_PRESSURE,
           }) {
        for (uint16_t reg_class : {0, 1, 2}) {
          for (const auto kind : {
                   LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
                   LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID,
               }) {
            for (const auto& [minimum_base, maximum_base] : ranges) {
              SCOPED_TRACE(start_point);
              SCOPED_TRACE(end_point);
              SCOPED_TRACE(static_cast<int>(conflict_class));
              SCOPED_TRACE(reg_class);
              SCOPED_TRACE(static_cast<int>(kind));
              SCOPED_TRACE(minimum_base);
              SCOPED_TRACE(maximum_base);

              bool expected_found = false;
              uint32_t expected_base = UINT32_MAX;
              for (uint32_t location = minimum_base; location <= maximum_base;
                   ++location) {
                if (!model_conflicts(start_point, end_point, conflict_class,
                                     reg_class, kind, location)) {
                  expected_found = true;
                  expected_base = location;
                  break;
                }
              }
              uint32_t actual_base = UINT32_MAX;
              EXPECT_EQ(expected_found,
                        FindNextAvailable(index, end_point, conflict_class,
                                          minimum_base, maximum_base,
                                          &actual_base, reg_class, kind));
              if (expected_found) {
                EXPECT_EQ(expected_base, actual_base);
              }

              expected_found = false;
              expected_base = UINT32_MAX;
              for (uint32_t location = maximum_base;; --location) {
                if (!model_conflicts(start_point, end_point, conflict_class,
                                     reg_class, kind, location)) {
                  expected_found = true;
                  expected_base = location;
                  break;
                }
                if (location == minimum_base) {
                  break;
                }
              }
              actual_base = UINT32_MAX;
              EXPECT_EQ(expected_found,
                        FindPreviousAvailable(index, end_point, conflict_class,
                                              minimum_base, maximum_base,
                                              &actual_base, reg_class, kind));
              if (expected_found) {
                EXPECT_EQ(expected_base, actual_base);
              }
            }
          }
        }
      }
    }
  }
}

TEST_F(LowAllocationStorageLeaseIndexTest,
       RetainsIndependentIncomingSelections) {
  std::vector<loom_low_allocation_storage_lease_t> leases = {
      Lease(4, 8, 10, 2), Lease(40, 48, 10, 2), Lease(90, 99, 11, 1)};
  loom_low_allocation_storage_lease_unit_index_t index;
  Initialize(leases, &index);
  for (uint32_t i = 0; i < leases.size(); ++i) {
    loom_low_allocation_storage_lease_unit_index_insert(&index, &descriptors_,
                                                        i, 0);
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
                                                        i, 0);
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
                                                        i, 0);
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
                                                        i, 0);
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
                                                        i, 0);
  }
  EXPECT_EQ((std::vector<uint32_t>{0}), Query(index, 0, 0, 0));
  EXPECT_EQ((std::vector<uint32_t>{1}),
            Query(index, UINT32_MAX, UINT32_MAX, UINT32_MAX));
  EXPECT_EQ((std::vector<uint32_t>{2}),
            Query(index, 0x80000001u, 0x80000001u, 0x80000000u));
}

TEST_F(LowAllocationStorageLeaseIndexTest,
       WalksSparseMaterializedUnitsAcrossTheFullLocationDomain) {
  std::vector<loom_low_allocation_storage_lease_t> leases = {
      Lease(0, 10, 1),
      Lease(0, 10, 0x80000000u),
      Lease(0, 10, UINT32_MAX),
      Lease(0, 10, 1, 1, 2),
  };
  loom_low_allocation_storage_lease_unit_index_t index;
  Initialize(leases, &index);
  for (uint32_t i : {2u, 0u, 3u, 1u}) {
    loom_low_allocation_storage_lease_unit_index_insert(&index, &descriptors_,
                                                        i, 0);
  }

  const auto query_units = [&](uint32_t location_base, uint32_t location_count,
                               uint16_t reg_class) {
    loom_low_allocation_storage_lease_unit_query_t query;
    loom_low_allocation_storage_lease_unit_query_initialize(
        &index, &descriptors_, reg_class,
        LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, location_base,
        location_count,
        /*minimum_end_point=*/1, /*start_point_limit=*/1,
        /*selection=*/nullptr, &query);
    std::vector<std::pair<uint32_t, uint32_t>> result;
    uint32_t lease_index = 0;
    while (loom_low_allocation_storage_lease_unit_query_next(&query,
                                                             &lease_index)) {
      result.emplace_back(query.active_location, lease_index);
    }
    return result;
  };

  EXPECT_EQ((std::vector<std::pair<uint32_t, uint32_t>>{
                {1, 0}, {0x80000000u, 1}, {UINT32_MAX, 2}}),
            query_units(1, UINT32_MAX, 0));
  EXPECT_TRUE(query_units(1, 0, 0).empty());
  EXPECT_TRUE(query_units(2, 0x7FFFFFFEu, 0).empty());
  EXPECT_EQ((std::vector<std::pair<uint32_t, uint32_t>>{{0x80000000u, 1}}),
            query_units(2, 0x7FFFFFFFu, 0));
  EXPECT_EQ((std::vector<std::pair<uint32_t, uint32_t>>{{1, 3}}),
            query_units(1, 1, 2));

  std::swap(leases[0].location_base, leases[2].location_base);
  loom_low_allocation_storage_lease_unit_index_rebuild(&index, &descriptors_,
                                                       leases.size());
  EXPECT_EQ((std::vector<std::pair<uint32_t, uint32_t>>{
                {1, 2}, {0x80000000u, 1}, {UINT32_MAX, 0}}),
            query_units(1, UINT32_MAX, 0));
}

TEST_F(LowAllocationStorageLeaseIndexTest, PreservesEmptyBoundaryQueries) {
  std::vector<loom_low_allocation_storage_lease_t> leases = {
      Lease(0, UINT32_MAX)};
  loom_low_allocation_storage_lease_unit_index_t index;
  Initialize(leases, &index);
  loom_low_allocation_storage_lease_unit_index_insert(&index, &descriptors_, 0,
                                                      0);
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
  for (uint32_t i = 0; i < 64; ++i) {
    leases.push_back(Lease(i, i + 1));
  }
  loom_low_allocation_storage_lease_unit_index_t index;
  Initialize(leases, &index);
  for (uint32_t i = 0; i < leases.size(); ++i) {
    loom_low_allocation_storage_lease_unit_index_insert(&index, &descriptors_,
                                                        i, 0);
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
                                                        i, 0);
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
                                                        i, 0);
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
