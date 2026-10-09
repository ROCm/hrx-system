// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstring>
#include <set>

#include "iree/hal/allocator.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(HeapAllocatorTest, ImportsByteAlignedStorageWithExplicitAlignment) {
  iree_hal_allocator_t* allocator = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(IREE_SV("test"), iree_allocator_system(),
                                     iree_allocator_system(), &allocator));
  alignas(64) uint8_t storage[18] = {};
  storage[0] = 0xA5;
  storage[17] = 0x5A;
  iree_hal_external_buffer_t external_buffer = {
      .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION, .size = 16};
  external_buffer.handle.host_allocation.ptr = storage + 1;
  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_MAPPING;
  int release_count = 0;
  const iree_hal_buffer_release_callback_t release_callback = {
      [](void* user_data, iree_hal_buffer_t*) {
        ++*static_cast<int*>(user_data);
      },
      &release_count,
  };
  for (iree_device_size_t alignment : {0, 1, 2, 3, 64}) {
    SCOPED_TRACE(alignment);
    params.min_alignment = alignment;
    const int releases_before_import = release_count;
    iree_hal_buffer_t* buffer = nullptr;
    iree_status_t status = iree_hal_allocator_import_buffer(
        allocator, params, &external_buffer, release_callback, &buffer);
    if (alignment > 1) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
      EXPECT_EQ(nullptr, buffer);
      EXPECT_EQ(releases_before_import, release_count);
      continue;
    }
    IREE_ASSERT_OK(status);
    iree_hal_buffer_mapping_t mapping = {};
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_ALL,
        IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 16, &mapping));
    EXPECT_EQ(storage + 1, mapping.contents.data);
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
    const uint32_t pattern = 0x12345678;
    IREE_ASSERT_OK(
        iree_hal_buffer_map_fill(buffer, 0, 16, &pattern, sizeof(pattern)));
    uint32_t values[4] = {};
    IREE_ASSERT_OK(iree_hal_buffer_map_read(buffer, 0, values, sizeof(values)));
    EXPECT_THAT(values, ::testing::Each(pattern));
    iree_hal_buffer_release(buffer);
    EXPECT_EQ(0xA5, storage[0]);
    EXPECT_EQ(0x5A, storage[17]);
  }
  EXPECT_EQ(2, release_count);
  iree_hal_allocator_release(allocator);
}

TEST(HeapAllocatorTest, ProvidesCoherentUnifiedMemory) {
  iree_hal_allocator_t* allocator = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(IREE_SV("test"), iree_allocator_system(),
                                     iree_allocator_system(), &allocator));

  iree_hal_allocator_memory_heap_t heap;
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

  const iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, 16, &buffer));
  EXPECT_TRUE(iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                                IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                                    IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
                                    IREE_HAL_MEMORY_TYPE_HOST_CACHED |
                                    IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));

  iree_hal_buffer_release(buffer);
  iree_hal_allocator_release(allocator);
}

// Records the pointers handed out by TrackingAllocatorCtl so that every free
// can be matched against the allocation that produced it.
struct TrackingAllocatorState {
  // Block bases this allocator returned that have not been freed yet. An
  // aligned allocation is served as one underlying block, so the entry is that
  // block's base and not the interior pointer the caller of
  // iree_allocator_malloc_aligned receives.
  std::set<void*> live_allocations;
  // Number of frees of pointers this allocator never returned.
  iree_host_size_t unowned_free_count = 0;
};

static iree_status_t TrackingAllocatorCtl(void* self,
                                          iree_allocator_command_t command,
                                          const void* params,
                                          void** inout_ptr) {
  auto* state = static_cast<TrackingAllocatorState*>(self);
  const iree_allocator_t system_allocator = iree_allocator_system();
  switch (command) {
    case IREE_ALLOCATOR_COMMAND_MALLOC:
    case IREE_ALLOCATOR_COMMAND_CALLOC: {
      iree_status_t status = system_allocator.ctl(system_allocator.self,
                                                  command, params, inout_ptr);
      if (iree_status_is_ok(status)) {
        state->live_allocations.insert(*inout_ptr);
      }
      return status;
    }
    case IREE_ALLOCATOR_COMMAND_FREE: {
      auto it = state->live_allocations.find(*inout_ptr);
      if (it == state->live_allocations.end()) {
        // Forwarding this would hand the system allocator a pointer it never
        // returned; the free is recorded and dropped so that the mismatch is
        // reported as a failed expectation.
        ++state->unowned_free_count;
        return iree_ok_status();
      }
      state->live_allocations.erase(it);
      return system_allocator.ctl(system_allocator.self, command, params,
                                  inout_ptr);
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported tracking allocator command %u",
                              (unsigned)command);
  }
}

static iree_allocator_t TrackingAllocator(TrackingAllocatorState* state) {
  return iree_allocator_t{
      .self = state,
      .ctl = TrackingAllocatorCtl,
  };
}

TEST(HeapAllocatorTest, HonorsAlignmentWithCombinedAndSeparateStorage) {
  TrackingAllocatorState data_state;
  for (iree_allocator_t data_allocator :
       {iree_allocator_system(), TrackingAllocator(&data_state)}) {
    SCOPED_TRACE(data_allocator.self == &data_state ? "separate" : "combined");
    iree_hal_allocator_t* allocator = nullptr;
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(
        IREE_SV("test"), data_allocator, iree_allocator_system(), &allocator));
    iree_hal_buffer_params_t params = {};
    params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
    params.usage = IREE_HAL_BUFFER_USAGE_MAPPING;
    for (iree_device_size_t alignment : {0, 1, 64, 256, 4096}) {
      SCOPED_TRACE(alignment);
      params.min_alignment = alignment;
      iree_hal_buffer_t* buffer = nullptr;
      IREE_ASSERT_OK(
          iree_hal_allocator_allocate_buffer(allocator, params, 17, &buffer));
      iree_hal_buffer_mapping_t mapping = {};
      IREE_ASSERT_OK(iree_hal_buffer_map_range(
          buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_ALL,
          IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 17, &mapping));
      EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(mapping.contents.data) %
                        std::max<iree_device_size_t>(
                            alignment, IREE_HAL_HEAP_BUFFER_ALIGNMENT));
      memset(mapping.contents.data, 0xA5, mapping.contents.data_length);
      IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
      uint8_t values[17] = {};
      IREE_ASSERT_OK(
          iree_hal_buffer_map_read(buffer, 0, values, sizeof(values)));
      EXPECT_THAT(values, ::testing::Each(0xA5));
      iree_hal_buffer_release(buffer);
      EXPECT_TRUE(data_state.live_allocations.empty());
    }
    params.min_alignment = 3;
    iree_hal_buffer_t* buffer = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_allocator_allocate_buffer(allocator, params, 17, &buffer));
    EXPECT_EQ(nullptr, buffer);
    EXPECT_EQ(0u, data_state.unowned_free_count);
    iree_hal_allocator_release(allocator);
  }
}

// Buffers from an allocator with distinct data and host allocators keep their
// storage in an allocation separate from their metadata; this pins that the
// storage block is handed back to its data allocator as the pointer that
// allocator returned, which is the block base and not the interior pointer
// carved out of it.
TEST(HeapAllocatorTest, FreesSplitStorageAsItsDataAllocatorReturnedIt) {
  TrackingAllocatorState data_state;
  TrackingAllocatorState host_state;
  iree_hal_allocator_t* allocator = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_create_heap(
      IREE_SV("test"), TrackingAllocator(&data_state),
      TrackingAllocator(&host_state), &allocator));

  // The allocator object is itself a host allocation; the buffer metadata
  // shows up as growth of the host live set across the allocation below.
  const size_t host_allocation_count_before_buffer =
      host_state.live_allocations.size();

  const iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, 16, &buffer));

  // The data allocator differs from the host allocator, so the buffer storage
  // comes from the data allocator and the buffer metadata from the host
  // allocator; the checks below are vacuous unless that split path ran.
  ASSERT_EQ(data_state.live_allocations.size(), 1u);
  ASSERT_EQ(host_state.live_allocations.size(),
            host_allocation_count_before_buffer + 1);

  iree_hal_buffer_release(buffer);
  iree_hal_allocator_release(allocator);

  EXPECT_EQ(data_state.unowned_free_count, 0u);
  EXPECT_TRUE(data_state.live_allocations.empty());

  // The host allocator served the allocator object as well as the buffer
  // metadata, so it is only drained once both have been released.
  EXPECT_EQ(host_state.unowned_free_count, 0u);
  EXPECT_TRUE(host_state.live_allocations.empty());
}

}  // namespace
