// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/binding_storage.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct TrackingAllocator {
  bool fail_allocations = false;
  iree_host_size_t allocation_count = 0;
  iree_host_size_t free_count = 0;

  iree_allocator_t allocator() { return {this, Control}; }

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<TrackingAllocator*>(self);
    switch (command) {
      case IREE_ALLOCATOR_COMMAND_MALLOC:
      case IREE_ALLOCATOR_COMMAND_CALLOC:
        if (allocator->fail_allocations) {
          return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
        }
        ++allocator->allocation_count;
        break;
      case IREE_ALLOCATOR_COMMAND_FREE:
        ++allocator->free_count;
        break;
      default:
        break;
    }
    iree_allocator_t system_allocator = iree_allocator_system();
    return system_allocator.ctl(system_allocator.self, command, params,
                                inout_ptr);
  }
};

TEST(BufferRefListStorageTest, EmptyUsesNoStorage) {
  TrackingAllocator allocator;
  iree_hal_remote_server_buffer_ref_list_storage_t storage;
  IREE_ASSERT_OK(iree_hal_remote_server_buffer_ref_list_storage_initialize(
      /*count=*/0, &storage, allocator.allocator()));
  EXPECT_EQ(storage.list.count, 0u);
  EXPECT_EQ(storage.list.values, nullptr);
  EXPECT_EQ(storage.values, nullptr);
  EXPECT_EQ(storage.allocated_values, nullptr);
  EXPECT_EQ(allocator.allocation_count, 0u);

  iree_hal_remote_server_buffer_ref_list_storage_deinitialize(&storage);
  EXPECT_EQ(allocator.free_count, 0u);
}

TEST(BufferRefListStorageTest, InlineBoundaryUsesNoAllocation) {
  TrackingAllocator allocator;
  iree_hal_remote_server_buffer_ref_list_storage_t storage;
  IREE_ASSERT_OK(iree_hal_remote_server_buffer_ref_list_storage_initialize(
      IREE_HAL_REMOTE_SERVER_INLINE_BINDING_CAPACITY, &storage,
      allocator.allocator()));
  EXPECT_EQ(storage.list.count, IREE_HAL_REMOTE_SERVER_INLINE_BINDING_CAPACITY);
  EXPECT_EQ(storage.list.values, storage.inline_values);
  EXPECT_EQ(storage.values, storage.inline_values);
  EXPECT_EQ(storage.allocated_values, nullptr);
  EXPECT_EQ(allocator.allocation_count, 0u);

  storage.values[31] = iree_hal_make_indirect_buffer_ref(
      /*buffer_slot=*/31, /*offset=*/32, /*length=*/64);
  EXPECT_EQ(storage.list.values[31].buffer_slot, 31u);
  EXPECT_EQ(storage.list.values[31].offset, 32u);
  EXPECT_EQ(storage.list.values[31].length, 64u);

  iree_hal_remote_server_buffer_ref_list_storage_deinitialize(&storage);
  EXPECT_EQ(allocator.free_count, 0u);
  EXPECT_EQ(storage.list.count, 0u);
  EXPECT_EQ(storage.values, nullptr);
}

TEST(BufferRefListStorageTest, WiderListSpillsOnce) {
  TrackingAllocator allocator;
  iree_hal_remote_server_buffer_ref_list_storage_t storage;
  constexpr iree_host_size_t kCount =
      IREE_HAL_REMOTE_SERVER_INLINE_BINDING_CAPACITY + 1;
  IREE_ASSERT_OK(iree_hal_remote_server_buffer_ref_list_storage_initialize(
      kCount, &storage, allocator.allocator()));
  EXPECT_EQ(storage.list.count, kCount);
  EXPECT_EQ(storage.list.values, storage.allocated_values);
  EXPECT_EQ(storage.values, storage.allocated_values);
  EXPECT_NE(storage.values, storage.inline_values);
  EXPECT_EQ(allocator.allocation_count, 1u);

  iree_hal_remote_server_buffer_ref_list_storage_deinitialize(&storage);
  EXPECT_EQ(allocator.free_count, 1u);
}

TEST(BufferRefListStorageTest, AllocationFailureIsRecoverable) {
  TrackingAllocator allocator;
  allocator.fail_allocations = true;
  iree_hal_remote_server_buffer_ref_list_storage_t storage;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_remote_server_buffer_ref_list_storage_initialize(
          IREE_HAL_REMOTE_SERVER_INLINE_BINDING_CAPACITY + 1, &storage,
          allocator.allocator()));
  EXPECT_EQ(storage.list.count, 0u);
  EXPECT_EQ(storage.values, nullptr);
  EXPECT_EQ(storage.allocated_values, nullptr);

  iree_hal_remote_server_buffer_ref_list_storage_deinitialize(&storage);
  EXPECT_EQ(allocator.free_count, 0u);
}

TEST(BufferRefListStorageTest, CountOverflowIsRejected) {
  TrackingAllocator allocator;
  iree_hal_remote_server_buffer_ref_list_storage_t storage;
  constexpr iree_host_size_t kOverflowCount =
      IREE_HOST_SIZE_MAX / sizeof(iree_hal_buffer_ref_t) + 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_remote_server_buffer_ref_list_storage_initialize(
          kOverflowCount, &storage, allocator.allocator()));
  EXPECT_EQ(allocator.allocation_count, 0u);

  iree_hal_remote_server_buffer_ref_list_storage_deinitialize(&storage);
  EXPECT_EQ(allocator.free_count, 0u);
}

TEST(BufferBindingTableStorageTest, EmptyUsesNoStorage) {
  TrackingAllocator allocator;
  iree_hal_remote_server_buffer_binding_table_storage_t storage;
  IREE_ASSERT_OK(iree_hal_remote_server_buffer_binding_table_storage_initialize(
      /*count=*/0, &storage, allocator.allocator()));
  EXPECT_EQ(storage.table.count, 0u);
  EXPECT_EQ(storage.table.bindings, nullptr);
  EXPECT_EQ(storage.bindings, nullptr);
  EXPECT_EQ(storage.allocated_bindings, nullptr);
  EXPECT_EQ(allocator.allocation_count, 0u);

  iree_hal_remote_server_buffer_binding_table_storage_deinitialize(&storage);
  EXPECT_EQ(allocator.free_count, 0u);
}

TEST(BufferBindingTableStorageTest, InlineBoundaryUsesNoAllocation) {
  TrackingAllocator allocator;
  iree_hal_remote_server_buffer_binding_table_storage_t storage;
  IREE_ASSERT_OK(iree_hal_remote_server_buffer_binding_table_storage_initialize(
      IREE_HAL_REMOTE_SERVER_INLINE_BINDING_CAPACITY, &storage,
      allocator.allocator()));
  EXPECT_EQ(storage.table.count,
            IREE_HAL_REMOTE_SERVER_INLINE_BINDING_CAPACITY);
  EXPECT_EQ(storage.table.bindings, storage.inline_bindings);
  EXPECT_EQ(storage.bindings, storage.inline_bindings);
  EXPECT_EQ(storage.allocated_bindings, nullptr);
  EXPECT_EQ(allocator.allocation_count, 0u);

  storage.bindings[31] = {
      /*.buffer=*/nullptr,
      /*.offset=*/32,
      /*.length=*/64,
  };
  EXPECT_EQ(storage.table.bindings[31].offset, 32u);
  EXPECT_EQ(storage.table.bindings[31].length, 64u);

  iree_hal_remote_server_buffer_binding_table_storage_deinitialize(&storage);
  EXPECT_EQ(allocator.free_count, 0u);
}

TEST(BufferBindingTableStorageTest, WiderTableSpillsOnce) {
  TrackingAllocator allocator;
  iree_hal_remote_server_buffer_binding_table_storage_t storage;
  constexpr iree_host_size_t kCount =
      IREE_HAL_REMOTE_SERVER_INLINE_BINDING_CAPACITY + 1;
  IREE_ASSERT_OK(iree_hal_remote_server_buffer_binding_table_storage_initialize(
      kCount, &storage, allocator.allocator()));
  EXPECT_EQ(storage.table.count, kCount);
  EXPECT_EQ(storage.table.bindings, storage.allocated_bindings);
  EXPECT_EQ(storage.bindings, storage.allocated_bindings);
  EXPECT_NE(storage.bindings, storage.inline_bindings);
  EXPECT_EQ(allocator.allocation_count, 1u);

  iree_hal_remote_server_buffer_binding_table_storage_deinitialize(&storage);
  EXPECT_EQ(allocator.free_count, 1u);
}

TEST(BufferBindingTableStorageTest, AllocationFailureIsRecoverable) {
  TrackingAllocator allocator;
  allocator.fail_allocations = true;
  iree_hal_remote_server_buffer_binding_table_storage_t storage;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_remote_server_buffer_binding_table_storage_initialize(
          IREE_HAL_REMOTE_SERVER_INLINE_BINDING_CAPACITY + 1, &storage,
          allocator.allocator()));
  EXPECT_EQ(storage.table.count, 0u);
  EXPECT_EQ(storage.bindings, nullptr);
  EXPECT_EQ(storage.allocated_bindings, nullptr);

  iree_hal_remote_server_buffer_binding_table_storage_deinitialize(&storage);
  EXPECT_EQ(allocator.free_count, 0u);
}

TEST(BufferBindingTableStorageTest, CountOverflowIsRejected) {
  TrackingAllocator allocator;
  iree_hal_remote_server_buffer_binding_table_storage_t storage;
  constexpr iree_host_size_t kOverflowCount =
      IREE_HOST_SIZE_MAX / sizeof(iree_hal_buffer_binding_t) + 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_remote_server_buffer_binding_table_storage_initialize(
          kOverflowCount, &storage, allocator.allocator()));
  EXPECT_EQ(allocator.allocation_count, 0u);

  iree_hal_remote_server_buffer_binding_table_storage_deinitialize(&storage);
  EXPECT_EQ(allocator.free_count, 0u);
}

}  // namespace
