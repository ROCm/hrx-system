// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/allocator.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(HeapAllocatorTest, DescribesCoherentHostMemory) {
  iree_hal_allocator_t* allocator = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(iree_allocator_system(), &allocator));

  iree_hal_allocator_memory_heap_t heap = {};
  iree_host_size_t heap_count = 0;
  IREE_ASSERT_OK(
      iree_hal_allocator_query_memory_heaps(allocator, 1, &heap, &heap_count));
  ASSERT_EQ(heap_count, 1);
  EXPECT_TRUE(
      iree_all_bits_set(heap.type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                                       IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
                                       IREE_HAL_MEMORY_TYPE_HOST_CACHED |
                                       IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));
  const iree_hal_atomic_operation_capabilities_t expected_atomic_operations =
      iree_hal_atomic_operation_capabilities_for_host(
          IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  EXPECT_EQ(heap.atomic_operations.device_scope_32,
            expected_atomic_operations.device_scope_32);
  EXPECT_EQ(heap.atomic_operations.device_scope_64,
            expected_atomic_operations.device_scope_64);
  EXPECT_EQ(heap.atomic_operations.system_scope_32,
            expected_atomic_operations.system_scope_32);
  EXPECT_EQ(heap.atomic_operations.system_scope_64,
            expected_atomic_operations.system_scope_64);
  EXPECT_EQ(heap.min_alignment, IREE_HAL_HEAP_BUFFER_ALIGNMENT);

  iree_hal_allocator_release(allocator);
}

}  // namespace
