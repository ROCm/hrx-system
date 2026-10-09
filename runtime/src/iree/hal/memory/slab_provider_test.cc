// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/slab_provider.h"

#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_hal_asan_pool_options_t ShadowOptions() {
  iree_hal_asan_pool_options_t options = {
      .mode = IREE_HAL_ASAN_POOL_MODE_SHADOW,
      .shadow_granule_size = 8,
      .redzone_size = 16};
  return options;
}

TEST(SlabProviderTest, CPUPropertiesDescribeAtomicStorage) {
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(
      /*min_alignment=*/0, iree_allocator_system(), &provider));

  iree_hal_slab_provider_properties_t properties;
  iree_hal_slab_provider_query_properties(provider, &properties);
  EXPECT_EQ(properties.memory_type, IREE_HAL_CPU_SLAB_PROVIDER_MEMORY_TYPE);
  EXPECT_TRUE(iree_all_bits_set(properties.memory_type,
                                IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                                    IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
                                    IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));
  EXPECT_TRUE(iree_all_bits_set(properties.supported_usage,
                                IREE_HAL_BUFFER_USAGE_STORAGE));

  const iree_hal_atomic_operation_flags_t expected_32 =
      iree_atomic_int32_is_lock_free() ? IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL
                                       : 0;
  EXPECT_EQ(properties.atomic_operations.device_scope_32, expected_32);
  EXPECT_EQ(properties.atomic_operations.system_scope_32, expected_32);

  const iree_hal_atomic_operation_flags_t expected_64 =
      iree_atomic_int64_is_lock_free() ? IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL
                                       : 0;
  EXPECT_EQ(properties.atomic_operations.device_scope_64, expected_64);
  EXPECT_EQ(properties.atomic_operations.system_scope_64, expected_64);

  iree_hal_slab_provider_release(provider);
}

TEST(SlabProviderTest, CPUAlignmentMatchesActualStorage) {
  const iree_host_size_t alignments[] = {0, 1, 64, 256, 4096};
  for (iree_host_size_t alignment : alignments) {
    SCOPED_TRACE(alignment);
    iree_hal_slab_provider_t* provider = nullptr;
    IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(
        alignment, iree_allocator_system(), &provider));
    iree_hal_slab_provider_properties_t properties;
    iree_hal_slab_provider_query_properties(provider, &properties);
    EXPECT_EQ(properties.allocation_alignment,
              iree_max(alignment, IREE_HAL_HEAP_BUFFER_ALIGNMENT));

    for (iree_host_size_t requested_alignment : {1, 64, 256, 4096}) {
      SCOPED_TRACE(requested_alignment);
      EXPECT_GE(properties.max_allocation_alignment, requested_alignment);
      iree_hal_slab_t slab = {};
      IREE_ASSERT_OK(iree_hal_slab_provider_acquire_slab(
          provider, 127, requested_alignment, &slab));
      EXPECT_EQ(
          reinterpret_cast<uintptr_t>(slab.base_ptr) %
              iree_max(requested_alignment, properties.allocation_alignment),
          0u);
      EXPECT_GE(slab.length, 127u);
      memset(slab.base_ptr, 0x6B, 127);
      for (iree_host_size_t i = 0; i < 127; ++i) {
        EXPECT_EQ(slab.base_ptr[i], 0x6B);
      }
      iree_hal_slab_provider_release_slab(provider, &slab);
    }
    iree_hal_slab_provider_release(provider);
  }
}

TEST(SlabProviderTest, CPUAlignmentMustBeZeroOrPowerOfTwo) {
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_cpu_slab_provider_create(3, iree_allocator_system(), &provider));
  EXPECT_EQ(provider, nullptr);
}

TEST(SlabProviderASANTest, DisabledOptionsAreAlwaysValid) {
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(
      /*min_alignment=*/0, iree_allocator_system(), &provider));

  iree_hal_asan_pool_options_t options = {};
  IREE_EXPECT_OK(
      iree_hal_slab_provider_validate_asan_options(provider, &options));

  iree_hal_slab_provider_release(provider);
}

TEST(SlabProviderASANTest, EnabledOptionsRequireProviderSupport) {
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(
      /*min_alignment=*/0, iree_allocator_system(), &provider));

  iree_hal_asan_pool_options_t options = ShadowOptions();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_slab_provider_validate_asan_options(provider, &options));

  iree_hal_slab_provider_release(provider);
}

}  // namespace
