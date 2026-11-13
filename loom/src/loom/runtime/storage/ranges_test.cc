// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/ranges.h"

#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Fails only the selected host allocation; every successful allocation and
// release uses the real system allocator.
struct ControlledAllocator {
  // Number of successful allocations before rejection, or -1 when unbounded.
  int remaining = -1;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<ControlledAllocator*>(self);
    const bool allocation = command == IREE_ALLOCATOR_COMMAND_MALLOC ||
                            command == IREE_ALLOCATOR_COMMAND_CALLOC ||
                            command == IREE_ALLOCATOR_COMMAND_REALLOC;
    if (allocation && allocator->remaining == 0) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected range allocation failure");
    }
    if (allocation && allocator->remaining > 0) {
      --allocator->remaining;
    }
    const auto delegate = iree_allocator_system();
    return delegate.ctl(delegate.self, command, params, inout_ptr);
  }

  iree_allocator_t value() { return {this, Control}; }
};

class RangesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    for (size_t i = 0; i < roots.size(); ++i) {
      IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
          iree_hal_buffer_placement_undefined(),
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_ALL,
          IREE_HAL_BUFFER_USAGE_MAPPING, storage[i].size(),
          iree_make_byte_span(storage[i].data(), storage[i].size()), {},
          iree_allocator_system(), &roots[i]));
    }
    IREE_ASSERT_OK(loom_runtime_ranges_initialize(8, roots.size(), roots.data(),
                                                  allocator.value(), &ranges));
    loom_runtime_ranges_set_observer(
        &ranges, {+[](void* p) { ++*static_cast<size_t*>(p); }, &wakes});
  }

  void TearDown() override {
    for (auto& view : captured) {
      iree_hal_buffer_release(view);
    }
    for (auto& owner : owners) {
      iree_vm_ref_reset(&owner);
    }
    loom_runtime_ranges_deinitialize(&ranges);
    for (auto* root : roots) {
      iree_hal_buffer_release(root);
    }
  }

  bool Acquire(size_t ordinal, size_t count) {
    bool acquired = false;
    IREE_EXPECT_OK(loom_runtime_ranges_try_acquire(
        &ranges, count, &owners[ordinal], &acquired));
    return acquired;
  }

  // Source-owned storage with two different logical strides.
  std::array<std::array<uint8_t, 128>, 2> storage = {};
  // Stable device roots whose addresses cannot change during admission.
  std::array<iree_hal_buffer_t*, 2> roots = {};
  // Runtime interval ownership under test.
  loom_runtime_ranges_t ranges = {};
  // Allocation boundary for claim, reservation, and captured-view construction.
  ControlledAllocator allocator;
  // Independent request owners, including a capacity-blocked newcomer.
  std::array<iree_vm_ref_t, 3> owners = {};
  // Accepted borrowers that outlive caller interest.
  std::array<iree_hal_buffer_t*, 2> captured = {};
  // Capacity-return notifications after final view release.
  size_t wakes = 0;
};

TEST_F(RangesTest, CancelledRangeWaitsForEveryCapturedRoot) {
  ASSERT_TRUE(Acquire(0, 5));
  ASSERT_TRUE(Acquire(1, 3));
  EXPECT_EQ(loom_runtime_range_begin(owners[0]), 0u);
  EXPECT_EQ(loom_runtime_range_begin(owners[1]), 5u);
  ASSERT_FALSE(Acquire(2, 1));
  for (size_t i = 0; i < roots.size(); ++i) {
    auto* view = loom_runtime_range_buffer(owners[0], i);
    EXPECT_EQ(iree_hal_buffer_allocated_buffer(view), roots[i]);
    EXPECT_EQ(iree_hal_buffer_byte_length(view), storage[i].size());
    IREE_ASSERT_OK(iree_hal_buffer_subspan(
        view, 16, 32, iree_allocator_system(), &captured[i]));
  }
  iree_vm_ref_reset(&owners[0]);
  EXPECT_FALSE(Acquire(2, 5));
  EXPECT_EQ(loom_runtime_ranges_generation(&ranges), 0u);
  iree_hal_buffer_release(captured[0]);
  captured[0] = nullptr;
  EXPECT_FALSE(Acquire(2, 5));
  EXPECT_EQ(wakes, 0u);
  iree_hal_buffer_release(captured[1]);
  captured[1] = nullptr;
  EXPECT_EQ(loom_runtime_ranges_generation(&ranges), 1u);
  EXPECT_EQ(wakes, 1u);
  ASSERT_TRUE(Acquire(2, 5));
  EXPECT_EQ(loom_runtime_range_begin(owners[2]), 0u);
  EXPECT_EQ(loom_runtime_range_begin(owners[1]), 5u);
}

TEST_F(RangesTest, ContinuingDescriptorSharesResetWithoutDanglingOwnership) {
  ASSERT_TRUE(Acquire(0, 3));
  owners[1] = iree_vm_ref_retain(owners[0]);
  captured[0] = loom_runtime_range_buffer(owners[0], 0);
  iree_hal_buffer_retain(captured[0]);
  loom_runtime_range_reset(owners[1]);
  EXPECT_FALSE(loom_runtime_range_is_active(owners[0]));
  EXPECT_EQ(wakes, 0u);
  ASSERT_TRUE(Acquire(0, 5));
  EXPECT_EQ(loom_runtime_range_begin(owners[1]), 3u);
  EXPECT_FALSE(Acquire(2, 1));
  iree_vm_ref_reset(&owners[1]);
  EXPECT_TRUE(loom_runtime_range_is_active(owners[0]));
  iree_hal_buffer_release(captured[0]);
  captured[0] = nullptr;
  ASSERT_TRUE(Acquire(2, 3));
  EXPECT_EQ(loom_runtime_range_begin(owners[2]), 0u);
  EXPECT_EQ(loom_runtime_range_begin(owners[0]), 3u);
}

TEST_F(RangesTest, FragmentedSpaceDoesNotRelocateLiveIntervals) {
  ASSERT_TRUE(Acquire(0, 3));
  ASSERT_TRUE(Acquire(1, 3));
  loom_runtime_range_reset(owners[0]);
  EXPECT_FALSE(Acquire(2, 4));
  EXPECT_EQ(loom_runtime_range_begin(owners[1]), 3u);
  ASSERT_TRUE(Acquire(0, 2));
  EXPECT_EQ(loom_runtime_range_begin(owners[0]), 0u);
  loom_runtime_range_reset(owners[1]);
  ASSERT_TRUE(Acquire(2, 6));
  EXPECT_EQ(loom_runtime_range_begin(owners[2]), 2u);
}

TEST_F(RangesTest, FailedConstructionReturnsEveryClaimedUnit) {
  // Exercise failure before the claim, reservation, and each of two root views.
  for (int budget = 0; budget < 4; ++budget) {
    SCOPED_TRACE(budget);
    allocator.remaining = budget;
    bool acquired = true;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        loom_runtime_ranges_try_acquire(&ranges, 8, &owners[0], &acquired));
    EXPECT_FALSE(acquired);
    EXPECT_EQ(owners[0].object, nullptr);
    allocator.remaining = -1;
    ASSERT_TRUE(Acquire(1, 8));
    EXPECT_EQ(loom_runtime_range_begin(owners[1]), 0u);
    iree_vm_ref_reset(&owners[1]);
  }
}

TEST_F(RangesTest, FailedReacquisitionPreservesSharedOwner) {
  ASSERT_TRUE(Acquire(0, 8));
  owners[1] = iree_vm_ref_retain(owners[0]);
  loom_runtime_range_reset(owners[0]);
  // A reusable reservation needs only a claim and the two root views.
  for (int budget = 0; budget < 3; ++budget) {
    SCOPED_TRACE(budget);
    allocator.remaining = budget;
    bool acquired = true;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        loom_runtime_ranges_try_acquire(&ranges, 8, &owners[0], &acquired));
    EXPECT_FALSE(acquired);
    EXPECT_EQ(owners[0].object, owners[1].object);
    EXPECT_FALSE(loom_runtime_range_is_active(owners[1]));
    allocator.remaining = -1;
    ASSERT_TRUE(Acquire(0, 8));
    EXPECT_EQ(loom_runtime_range_begin(owners[1]), 0u);
    loom_runtime_range_reset(owners[1]);
  }
}

}  // namespace
