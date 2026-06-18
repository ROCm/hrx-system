// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/memory_pool.h"

#include <array>
#include <cstdint>
#include <limits>

#include "iree/testing/gtest.h"

namespace {

using Slot = iree_hip_pool_allocation_slot_t;
using Tracker = iree_hip_pool_allocation_tracker_t;

TEST(HipMemoryPoolAllocationTrackerTest, TracksAndFindsLiveAllocations) {
  std::array<Slot, 4> slots;
  Tracker tracker;
  iree_hip_pool_allocation_tracker_initialize(&tracker, slots.data(),
                                              slots.size());

  EXPECT_TRUE(iree_hip_pool_allocation_tracker_insert(&tracker, 0x1000, 0x100));
  EXPECT_EQ(tracker.live_count, 1u);

  uintptr_t base = 0;
  size_t size = 0;
  EXPECT_TRUE(
      iree_hip_pool_allocation_tracker_find(&tracker, 0x1000, &base, &size));
  EXPECT_EQ(base, 0x1000u);
  EXPECT_EQ(size, 0x100u);

  EXPECT_TRUE(
      iree_hip_pool_allocation_tracker_find(&tracker, 0x10FF, &base, &size));
  EXPECT_EQ(base, 0x1000u);
  EXPECT_EQ(size, 0x100u);

  EXPECT_FALSE(
      iree_hip_pool_allocation_tracker_find(&tracker, 0x1100, &base, &size));
}

TEST(HipMemoryPoolAllocationTrackerTest, RejectsDoubleFreeAndInteriorFree) {
  std::array<Slot, 2> slots;
  Tracker tracker;
  iree_hip_pool_allocation_tracker_initialize(&tracker, slots.data(),
                                              slots.size());

  EXPECT_TRUE(iree_hip_pool_allocation_tracker_insert(&tracker, 0x2000, 0x80));

  size_t released_size = 0;
  EXPECT_FALSE(iree_hip_pool_allocation_tracker_release(&tracker, 0x2040,
                                                        &released_size));
  EXPECT_EQ(released_size, 0u);
  EXPECT_EQ(tracker.live_count, 1u);

  EXPECT_TRUE(iree_hip_pool_allocation_tracker_release(&tracker, 0x2000,
                                                       &released_size));
  EXPECT_EQ(released_size, 0x80u);
  EXPECT_EQ(tracker.live_count, 0u);

  EXPECT_FALSE(iree_hip_pool_allocation_tracker_release(&tracker, 0x2000,
                                                        &released_size));
  EXPECT_EQ(released_size, 0u);
}

TEST(HipMemoryPoolAllocationTrackerTest, ReusesReleasedSlots) {
  std::array<Slot, 2> slots;
  Tracker tracker;
  iree_hip_pool_allocation_tracker_initialize(&tracker, slots.data(),
                                              slots.size());

  EXPECT_TRUE(iree_hip_pool_allocation_tracker_insert(&tracker, 0x1000, 0x100));
  EXPECT_TRUE(iree_hip_pool_allocation_tracker_insert(&tracker, 0x2000, 0x100));
  EXPECT_FALSE(
      iree_hip_pool_allocation_tracker_insert(&tracker, 0x3000, 0x100));

  size_t released_size = 0;
  EXPECT_TRUE(iree_hip_pool_allocation_tracker_release(&tracker, 0x1000,
                                                       &released_size));
  EXPECT_TRUE(iree_hip_pool_allocation_tracker_insert(&tracker, 0x3000, 0x100));

  uintptr_t base = 0;
  size_t size = 0;
  EXPECT_FALSE(
      iree_hip_pool_allocation_tracker_find(&tracker, 0x1000, &base, &size));
  EXPECT_TRUE(
      iree_hip_pool_allocation_tracker_find(&tracker, 0x3000, &base, &size));
  EXPECT_EQ(base, 0x3000u);
  EXPECT_EQ(tracker.live_count, 2u);
  EXPECT_EQ(iree_hip_pool_allocation_tracker_live_size(&tracker), 0x200u);
}

TEST(HipMemoryPoolAllocationTrackerTest, RejectsOverlappingLiveRanges) {
  std::array<Slot, 4> slots;
  Tracker tracker;
  iree_hip_pool_allocation_tracker_initialize(&tracker, slots.data(),
                                              slots.size());

  EXPECT_TRUE(iree_hip_pool_allocation_tracker_insert(&tracker, 0x1000, 0x100));
  EXPECT_FALSE(
      iree_hip_pool_allocation_tracker_insert(&tracker, 0x1000, 0x100));
  EXPECT_FALSE(
      iree_hip_pool_allocation_tracker_insert(&tracker, 0x1080, 0x100));
  EXPECT_FALSE(
      iree_hip_pool_allocation_tracker_insert(&tracker, 0x0F80, 0x100));
  EXPECT_TRUE(iree_hip_pool_allocation_tracker_insert(&tracker, 0x1100, 0x100));
}

TEST(HipMemoryPoolAllocationTrackerTest, ReuseIsBoundedByLiveNotHistorical) {
  std::array<Slot, 1> slots;
  Tracker tracker;
  iree_hip_pool_allocation_tracker_initialize(&tracker, slots.data(),
                                              slots.size());

  for (uintptr_t i = 1; i <= 128; ++i) {
    const uintptr_t address = 0x1000 + i * 0x100;
    EXPECT_TRUE(
        iree_hip_pool_allocation_tracker_insert(&tracker, address, 0x100));
    size_t released_size = 0;
    EXPECT_TRUE(iree_hip_pool_allocation_tracker_release(&tracker, address,
                                                         &released_size));
    EXPECT_EQ(released_size, 0x100u);
    EXPECT_EQ(tracker.live_count, 0u);
  }
}

TEST(HipMemoryPoolAllocationTrackerTest, RejectsWrappedAllocationRanges) {
  std::array<Slot, 2> slots;
  Tracker tracker;
  iree_hip_pool_allocation_tracker_initialize(&tracker, slots.data(),
                                              slots.size());

  constexpr uintptr_t kMax = std::numeric_limits<uintptr_t>::max();
  EXPECT_TRUE(iree_hip_pool_allocation_tracker_insert(&tracker, kMax, 1));
  EXPECT_FALSE(iree_hip_pool_allocation_tracker_insert(&tracker, kMax, 2));

  uintptr_t base = 0;
  size_t size = 0;
  EXPECT_TRUE(
      iree_hip_pool_allocation_tracker_find(&tracker, kMax, &base, &size));
  EXPECT_EQ(base, kMax);
  EXPECT_EQ(size, 1u);
}

TEST(HipMemoryPoolAllocationTrackerTest, ResetClearsLiveState) {
  std::array<Slot, 2> slots;
  Tracker tracker;
  iree_hip_pool_allocation_tracker_initialize(&tracker, slots.data(),
                                              slots.size());

  EXPECT_TRUE(iree_hip_pool_allocation_tracker_insert(&tracker, 0x1000, 0x100));
  iree_hip_pool_allocation_tracker_reset(&tracker);
  EXPECT_EQ(tracker.live_count, 0u);
  EXPECT_EQ(tracker.high_water_mark, 0u);
  EXPECT_FALSE(iree_hip_pool_allocation_tracker_find(&tracker, 0x1000, nullptr,
                                                     nullptr));
  EXPECT_TRUE(iree_hip_pool_allocation_tracker_insert(&tracker, 0x2000, 0x100));
}

TEST(HipMemoryPoolAllocationTrackerTest, LiveSizeTracksOnlyLiveSlots) {
  std::array<Slot, 4> slots;
  Tracker tracker;
  iree_hip_pool_allocation_tracker_initialize(&tracker, slots.data(),
                                              slots.size());

  EXPECT_TRUE(iree_hip_pool_allocation_tracker_insert(&tracker, 0x1000, 0x100));
  EXPECT_TRUE(iree_hip_pool_allocation_tracker_insert(&tracker, 0x2000, 0x300));
  EXPECT_EQ(iree_hip_pool_allocation_tracker_live_size(&tracker), 0x400u);

  size_t released_size = 0;
  EXPECT_TRUE(iree_hip_pool_allocation_tracker_release(&tracker, 0x1000,
                                                       &released_size));
  EXPECT_EQ(iree_hip_pool_allocation_tracker_live_size(&tracker), 0x300u);
}

TEST(HipMemoryPoolSizeTest, AlignsPowerOfTwoSizes) {
  size_t aligned_size = 0;
  EXPECT_TRUE(iree_hip_pool_align_size(1, 256, &aligned_size));
  EXPECT_EQ(aligned_size, 256u);
  EXPECT_TRUE(iree_hip_pool_align_size(256, 256, &aligned_size));
  EXPECT_EQ(aligned_size, 256u);
  EXPECT_TRUE(iree_hip_pool_align_size(257, 256, &aligned_size));
  EXPECT_EQ(aligned_size, 512u);
}

TEST(HipMemoryPoolSizeTest, RejectsInvalidAlignment) {
  size_t aligned_size = 0;
  EXPECT_FALSE(iree_hip_pool_align_size(1, 0, &aligned_size));
  EXPECT_FALSE(iree_hip_pool_align_size(1, 3, &aligned_size));
  EXPECT_FALSE(iree_hip_pool_align_size(1, 255, &aligned_size));
}

TEST(HipMemoryPoolSizeTest, RejectsAlignmentOverflow) {
  size_t aligned_size = 0;
  EXPECT_FALSE(iree_hip_pool_align_size(std::numeric_limits<size_t>::max(), 256,
                                        &aligned_size));
}

}  // namespace
