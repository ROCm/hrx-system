// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/allocator.h"

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

constexpr iree_hal_memory_type_t kHostOnlyType =
    IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_HOST_CACHED |
    IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
constexpr iree_hal_memory_type_t kHostLocalBit =
    IREE_HAL_MEMORY_TYPE_HOST_LOCAL & ~IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
constexpr iree_hal_memory_type_t kDeviceLocalBit =
    IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL & ~IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;

struct CompatibilityQuery {
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  iree_hal_buffer_params_t resolved = {};
  iree_device_size_t allocation_size = 0;
};

CompatibilityQuery QueryCompatibility(iree_hal_allocator_t* allocator,
                                      iree_hal_memory_type_t type,
                                      iree_hal_buffer_usage_t usage) {
  iree_hal_buffer_params_t params = {};
  params.type = type;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = usage;
  CompatibilityQuery result;
  result.compatibility = iree_hal_allocator_query_buffer_compatibility(
      allocator, params, /*allocation_size=*/16, &result.resolved,
      &result.allocation_size);
  return result;
}

class AllocatorCompatibilityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_amdxdna_allocator_create(
        iree_allocator_system(), /*native_device=*/nullptr, &allocator_));
  }
  void TearDown() override {
    iree_hal_allocator_release(allocator_);
    allocator_ = nullptr;
  }

  iree_hal_allocator_t* allocator_ = nullptr;
};

TEST_F(AllocatorCompatibilityTest, CreateAndRelease) {
  ASSERT_NE(allocator_, nullptr);
}

TEST_F(AllocatorCompatibilityTest,
       TransferUsageDoesNotAdvertiseQueueTransferCompatibility) {
  CompatibilityQuery result =
      QueryCompatibility(allocator_, IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE,
                         IREE_HAL_BUFFER_USAGE_TRANSFER);
  EXPECT_TRUE(iree_all_bits_set(result.compatibility,
                                IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE));
  EXPECT_FALSE(iree_any_bit_set(result.compatibility,
                                IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER));
  EXPECT_TRUE(
      iree_all_bits_set(result.resolved.usage, IREE_HAL_BUFFER_USAGE_TRANSFER));
}

TEST_F(AllocatorCompatibilityTest, AdvertisesCachedHostVisibleDeviceVisible) {
  CompatibilityQuery result =
      QueryCompatibility(allocator_, IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE,
                         IREE_HAL_BUFFER_USAGE_DISPATCH);
  EXPECT_TRUE(iree_all_bits_set(result.compatibility,
                                IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE));
  EXPECT_EQ(result.resolved.type, kHostOnlyType);
  EXPECT_FALSE(iree_any_bit_set(result.resolved.type,
                                IREE_HAL_MEMORY_TYPE_HOST_COHERENT));
  EXPECT_FALSE(iree_any_bit_set(result.resolved.type, kHostLocalBit));
  EXPECT_FALSE(iree_any_bit_set(result.resolved.type, kDeviceLocalBit));
}

TEST_F(AllocatorCompatibilityTest, DispatchUsageAdvertisesQueueDispatch) {
  CompatibilityQuery result =
      QueryCompatibility(allocator_, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
                         IREE_HAL_BUFFER_USAGE_DISPATCH);
  EXPECT_TRUE(iree_all_bits_set(
      result.compatibility, IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
                                IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH));
}

TEST_F(AllocatorCompatibilityTest, HostCoherentIsRejected) {
  CompatibilityQuery result = QueryCompatibility(
      allocator_,
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
      IREE_HAL_BUFFER_USAGE_TRANSFER);
  EXPECT_EQ(result.compatibility, IREE_HAL_BUFFER_COMPATIBILITY_NONE);
}

TEST_F(AllocatorCompatibilityTest, OptimalHostCoherentIsRejected) {
  CompatibilityQuery result = QueryCompatibility(
      allocator_,
      IREE_HAL_MEMORY_TYPE_OPTIMAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
      IREE_HAL_BUFFER_USAGE_TRANSFER);
  EXPECT_EQ(result.compatibility, IREE_HAL_BUFFER_COMPATIBILITY_NONE);
}

TEST_F(AllocatorCompatibilityTest, HostLocalWithoutOptimalIsRejected) {
  CompatibilityQuery result =
      QueryCompatibility(allocator_, IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
                         IREE_HAL_BUFFER_USAGE_TRANSFER);
  EXPECT_EQ(result.compatibility, IREE_HAL_BUFFER_COMPATIBILITY_NONE);
}

TEST_F(AllocatorCompatibilityTest, HostLocalDeviceVisibleIsHostOnly) {
  CompatibilityQuery result = QueryCompatibility(
      allocator_,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_BUFFER_USAGE_TRANSFER);
  EXPECT_TRUE(iree_all_bits_set(result.compatibility,
                                IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE));
  EXPECT_EQ(result.resolved.type, kHostOnlyType);
}

TEST_F(AllocatorCompatibilityTest,
       HistoricalHostLocalCoherentDeviceVisibleIsHostOnly) {
  CompatibilityQuery result = QueryCompatibility(
      allocator_,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_BUFFER_USAGE_TRANSFER);
  EXPECT_TRUE(iree_all_bits_set(result.compatibility,
                                IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE));
  EXPECT_EQ(result.resolved.type, kHostOnlyType);
  EXPECT_FALSE(iree_any_bit_set(result.resolved.type,
                                IREE_HAL_MEMORY_TYPE_HOST_COHERENT));
}

TEST_F(AllocatorCompatibilityTest, DeviceLocalWithoutOptimalIsRejected) {
  CompatibilityQuery result =
      QueryCompatibility(allocator_, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                         IREE_HAL_BUFFER_USAGE_TRANSFER);
  EXPECT_EQ(result.compatibility, IREE_HAL_BUFFER_COMPATIBILITY_NONE);
}

TEST_F(AllocatorCompatibilityTest, DeviceUncachedIsRejected) {
  CompatibilityQuery result =
      QueryCompatibility(allocator_,
                         IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
                             IREE_HAL_MEMORY_TYPE_DEVICE_UNCACHED,
                         IREE_HAL_BUFFER_USAGE_TRANSFER);
  EXPECT_EQ(result.compatibility, IREE_HAL_BUFFER_COMPATIBILITY_NONE);
}

TEST_F(AllocatorCompatibilityTest, HeapTypeMatchesAdvertisedHostOnlyType) {
  iree_hal_allocator_memory_heap_t heap = {};
  iree_host_size_t count = 0;
  IREE_ASSERT_OK(
      iree_hal_allocator_query_memory_heaps(allocator_, 1, &heap, &count));
  EXPECT_EQ(count, 1u);
  EXPECT_EQ(heap.type, kHostOnlyType);
}

}  // namespace
