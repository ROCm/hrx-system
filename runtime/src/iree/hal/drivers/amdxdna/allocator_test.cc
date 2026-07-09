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

TEST(AllocatorTest, CreateAndRelease) {
  iree_hal_allocator_t* allocator = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_allocator_create(
      iree_allocator_system(), /*native_device=*/nullptr, &allocator));
  ASSERT_NE(allocator, nullptr);
  iree_hal_allocator_release(allocator);
}

TEST(AllocatorTest, TransferUsageDoesNotAdvertiseQueueTransferCompatibility) {
  iree_hal_allocator_t* allocator = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_allocator_create(
      iree_allocator_system(), /*native_device=*/nullptr, &allocator));

  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  iree_device_size_t allocation_size = 16;
  iree_hal_buffer_params_t resolved_params = {};
  iree_device_size_t resolved_allocation_size = 0;

  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          allocator, params, allocation_size, &resolved_params,
          &resolved_allocation_size);
  EXPECT_TRUE(iree_all_bits_set(compatibility,
                                IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE));
  EXPECT_FALSE(iree_any_bit_set(compatibility,
                                IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER));
  EXPECT_TRUE(
      iree_all_bits_set(resolved_params.usage, IREE_HAL_BUFFER_USAGE_TRANSFER));

  iree_hal_allocator_release(allocator);
}

}  // namespace
