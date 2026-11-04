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

TEST(AllocatorReuseTest, BestFitAcceptsLargerCapacity) {
  const iree_device_size_t capacities[] = {4096, 1024, 2048, 8192};
  EXPECT_EQ(iree_hal_amdxdna_allocator_find_best_fit(
                capacities, IREE_ARRAYSIZE(capacities), /*required_size=*/1536),
            2u);
  EXPECT_EQ(iree_hal_amdxdna_allocator_find_best_fit(
                capacities, IREE_ARRAYSIZE(capacities), /*required_size=*/1024),
            1u);
}

TEST(AllocatorReuseTest, BestFitRejectsUndersizedCapacities) {
  const iree_device_size_t capacities[] = {512, 1024, 2048};
  EXPECT_EQ(iree_hal_amdxdna_allocator_find_best_fit(
                capacities, IREE_ARRAYSIZE(capacities), /*required_size=*/4096),
            IREE_HOST_SIZE_MAX);
}

TEST(HostRecyclerTest, ReusesSmallestSufficientBlockAndZerosCalloc) {
  iree_hal_amdxdna_host_recycler_t recycler;
  iree_hal_amdxdna_host_recycler_initialize(
      iree_allocator_system(), /*max_retained_bytes=*/4096,
      /*max_retained_count=*/4, &recycler);
  iree_allocator_t allocator =
      iree_hal_amdxdna_host_recycler_allocator(&recycler);

  uint8_t* large = nullptr;
  uint8_t* small = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc_uninitialized(
      allocator, 1024, reinterpret_cast<void**>(&large)));
  IREE_ASSERT_OK(iree_allocator_malloc_uninitialized(
      allocator, 512, reinterpret_cast<void**>(&small)));
  memset(large, 0xAB, 1024);
  iree_allocator_free(allocator, large);
  iree_allocator_free(allocator, small);

  uint8_t* reused = nullptr;
  IREE_ASSERT_OK(
      iree_allocator_malloc(allocator, 256, reinterpret_cast<void**>(&reused)));
  EXPECT_EQ(reused, small);
  for (iree_host_size_t i = 0; i < 256; ++i) EXPECT_EQ(reused[i], 0);
  iree_allocator_free(allocator, reused);
  iree_hal_amdxdna_host_recycler_deinitialize(&recycler);
}

TEST(HostRecyclerTest, EnforcesRetainedByteAndCountBudgets) {
  iree_hal_amdxdna_host_recycler_t recycler;
  iree_hal_amdxdna_host_recycler_initialize(
      iree_allocator_system(), /*max_retained_bytes=*/1024,
      /*max_retained_count=*/1, &recycler);
  iree_allocator_t allocator =
      iree_hal_amdxdna_host_recycler_allocator(&recycler);

  void* first = nullptr;
  void* second = nullptr;
  IREE_ASSERT_OK(
      iree_allocator_malloc_uninitialized(allocator, 768, &first));
  IREE_ASSERT_OK(
      iree_allocator_malloc_uninitialized(allocator, 512, &second));
  iree_allocator_free(allocator, first);
  iree_allocator_free(allocator, second);
  EXPECT_EQ(recycler.retained_count, 1u);
  EXPECT_EQ(recycler.retained_bytes, 768u);
  iree_hal_amdxdna_host_recycler_deinitialize(&recycler);
}

TEST(HostRecyclerTest, ReallocPreservesContents) {
  iree_hal_amdxdna_host_recycler_t recycler;
  iree_hal_amdxdna_host_recycler_initialize(
      iree_allocator_system(), /*max_retained_bytes=*/4096,
      /*max_retained_count=*/4, &recycler);
  iree_allocator_t allocator =
      iree_hal_amdxdna_host_recycler_allocator(&recycler);

  uint8_t* ptr = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc_uninitialized(
      allocator, 32, reinterpret_cast<void**>(&ptr)));
  for (uint8_t i = 0; i < 32; ++i) ptr[i] = i;
  IREE_ASSERT_OK(
      iree_allocator_realloc(allocator, 128, reinterpret_cast<void**>(&ptr)));
  for (uint8_t i = 0; i < 32; ++i) EXPECT_EQ(ptr[i], i);
  iree_allocator_free(allocator, ptr);
  iree_hal_amdxdna_host_recycler_deinitialize(&recycler);
}

TEST(HostRecyclerTest, GrowsLargestUndersizedRetainedBlock) {
  iree_hal_amdxdna_host_recycler_t recycler;
  iree_hal_amdxdna_host_recycler_initialize(
      iree_allocator_system(), /*max_retained_bytes=*/4096,
      /*max_retained_count=*/4, &recycler);
  iree_allocator_t allocator =
      iree_hal_amdxdna_host_recycler_allocator(&recycler);

  void* ptr = nullptr;
  IREE_ASSERT_OK(
      iree_allocator_malloc_uninitialized(allocator, 512, &ptr));
  iree_allocator_free(allocator, ptr);
  IREE_ASSERT_OK(
      iree_allocator_malloc_uninitialized(allocator, 1024, &ptr));
  iree_allocator_free(allocator, ptr);
  EXPECT_EQ(recycler.retained_count, 1u);
  EXPECT_EQ(recycler.retained_bytes, 1024u);
  iree_hal_amdxdna_host_recycler_deinitialize(&recycler);
}

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

TEST_F(AllocatorCompatibilityTest,
       HostLocalDeviceVisibleWithoutOptimalIsRejected) {
  CompatibilityQuery result = QueryCompatibility(
      allocator_,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_BUFFER_USAGE_TRANSFER);
  EXPECT_EQ(result.compatibility, IREE_HAL_BUFFER_COMPATIBILITY_NONE);
}

TEST_F(AllocatorCompatibilityTest,
       HistoricalHostLocalCoherentDeviceVisibleIsRejected) {
  CompatibilityQuery result = QueryCompatibility(
      allocator_,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_BUFFER_USAGE_TRANSFER);
  EXPECT_EQ(result.compatibility, IREE_HAL_BUFFER_COMPATIBILITY_NONE);
}

TEST_F(AllocatorCompatibilityTest, CanonicalHostOnlyBitsAreAccepted) {
  CompatibilityQuery result = QueryCompatibility(
      allocator_, kHostOnlyType, IREE_HAL_BUFFER_USAGE_TRANSFER);
  EXPECT_TRUE(iree_all_bits_set(result.compatibility,
                                IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE));
  EXPECT_EQ(result.resolved.type, kHostOnlyType);
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
