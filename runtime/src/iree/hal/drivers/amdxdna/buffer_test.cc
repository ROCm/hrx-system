// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/buffer.h"

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct CountingAllocatorState {
  int alloc_count = 0;
  int free_count = 0;
};

static iree_status_t CountingAllocatorCtl(void* self,
                                          iree_allocator_command_t command,
                                          const void* params,
                                          void** inout_ptr) {
  auto* state = reinterpret_cast<CountingAllocatorState*>(self);
  switch (command) {
    case IREE_ALLOCATOR_COMMAND_MALLOC:
      ++state->alloc_count;
      return iree_allocator_malloc_uninitialized(
          iree_allocator_system(),
          reinterpret_cast<const iree_allocator_alloc_params_t*>(params)
              ->byte_length,
          inout_ptr);
    case IREE_ALLOCATOR_COMMAND_CALLOC:
      ++state->alloc_count;
      return iree_allocator_malloc(
          iree_allocator_system(),
          reinterpret_cast<const iree_allocator_alloc_params_t*>(params)
              ->byte_length,
          inout_ptr);
    case IREE_ALLOCATOR_COMMAND_REALLOC:
      return iree_allocator_realloc(
          iree_allocator_system(),
          reinterpret_cast<const iree_allocator_alloc_params_t*>(params)
              ->byte_length,
          inout_ptr);
    case IREE_ALLOCATOR_COMMAND_FREE:
      ++state->free_count;
      iree_allocator_free(iree_allocator_system(), *inout_ptr);
      *inout_ptr = nullptr;
      return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported allocator command");
}

TEST(BufferTest, WrapStoresHostAllocatorForDestroy) {
  CountingAllocatorState state;
  iree_allocator_t allocator = {&state, CountingAllocatorCtl};

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_buffer_wrap(
      /*native_buffer=*/nullptr, iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_DEFAULT, /*allocation_size=*/64,
      /*byte_offset=*/0, /*byte_length=*/64,
      iree_hal_buffer_release_callback_null(), allocator, &buffer));

  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(state.alloc_count, 1);
  EXPECT_EQ(state.free_count, 0);

  iree_hal_buffer_release(buffer);
  EXPECT_EQ(state.free_count, 1);
}

TEST(BufferTest, ResolveRootRangeAddsMappingOffset) {
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_buffer_wrap(
      /*native_buffer=*/nullptr, iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_HOST_CACHED |
          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_DEFAULT,
      /*allocation_size=*/256, /*byte_offset=*/128, /*byte_length=*/64,
      iree_hal_buffer_release_callback_null(), iree_allocator_system(),
      &buffer));

  iree_device_size_t root_byte_offset = 0;
  iree_device_size_t byte_length = 0;
  IREE_ASSERT_OK(iree_hal_amdxdna_buffer_resolve_root_range(
      buffer, /*local_byte_offset=*/16, /*local_byte_length=*/8,
      &root_byte_offset, &byte_length));
  EXPECT_EQ(root_byte_offset, 144u);
  EXPECT_EQ(byte_length, 8u);

  IREE_ASSERT_OK(iree_hal_amdxdna_buffer_resolve_root_range(
      buffer, /*local_byte_offset=*/0, IREE_HAL_WHOLE_BUFFER, &root_byte_offset,
      &byte_length));
  EXPECT_EQ(root_byte_offset, 128u);
  EXPECT_EQ(byte_length, 64u);

  iree_status_t overflow = iree_hal_amdxdna_buffer_resolve_root_range(
      buffer, /*local_byte_offset=*/56, /*local_byte_length=*/16,
      &root_byte_offset, &byte_length);
  EXPECT_EQ(iree_status_code(overflow), IREE_STATUS_OUT_OF_RANGE);
  iree_status_ignore(overflow);

  iree_hal_buffer_release(buffer);
}

TEST(BufferTest, FlushWithoutNativeBufferFailsClosed) {
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_buffer_wrap(
      /*native_buffer=*/nullptr, iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_HOST_CACHED |
          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_DEFAULT,
      /*allocation_size=*/256, /*byte_offset=*/128, /*byte_length=*/64,
      iree_hal_buffer_release_callback_null(), iree_allocator_system(),
      &buffer));

  iree_status_t status = iree_hal_amdxdna_buffer_flush_range(
      buffer, /*local_byte_offset=*/16, /*local_byte_length=*/8);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_FAILED_PRECONDITION);
  iree_status_ignore(status);

  status = iree_hal_amdxdna_buffer_invalidate_range(
      buffer, /*local_byte_offset=*/16, /*local_byte_length=*/8);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_FAILED_PRECONDITION);
  iree_status_ignore(status);

  iree_hal_buffer_release(buffer);
}

}  // namespace

