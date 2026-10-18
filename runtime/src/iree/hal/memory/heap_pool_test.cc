// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/heap_pool.h"

#include <cstdint>
#include <cstring>

#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(HeapPoolTest, AllocatesMappedHostStorageWithoutProgressOwner) {
  const iree_hal_pool_host_access_t host_access = {
      .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      .modes = IREE_HAL_MAPPING_MODE_SCOPED | IREE_HAL_MAPPING_MODE_PERSISTENT,
  };
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_pool_create(host_access, iree_allocator_system(),
                                           iree_allocator_system(), &pool));

  EXPECT_EQ(iree_hal_pool_notification(pool), nullptr);
  const iree_hal_pool_host_access_t actual_host_access =
      iree_hal_pool_query_host_access(pool);
  EXPECT_EQ(actual_host_access.access, host_access.access);
  EXPECT_EQ(actual_host_access.modes, host_access.modes);
  EXPECT_EQ(actual_host_access.cacheability,
            IREE_HAL_HOST_CACHEABILITY_WRITE_BACK);

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(4096, 256),
      iree_infinite_timeout(), &buffer));
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.bytes_reserved, 4096u);
  EXPECT_EQ(stats.bytes_committed, 4096u);
  EXPECT_EQ(stats.reservation_count, 1u);
  EXPECT_EQ(stats.slab_count, 1u);

  iree_hal_buffer_mapping_t mapping = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, IREE_HAL_WHOLE_BUFFER, &mapping));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(mapping.contents.data) % 256, 0u);
  memset(mapping.contents.data, 0x5A, mapping.contents.data_length);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_external_storage_t storage = {};
  iree_hal_buffer_range_t logical_range = {};
  IREE_ASSERT_OK(iree_hal_buffer_export(
      buffer, iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
      iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
      &storage, &logical_range));
  EXPECT_NE(storage.payload.host_pointer, nullptr);
  EXPECT_EQ(storage.byte_length, 4096u);
  EXPECT_EQ(logical_range.byte_offset, 0u);
  EXPECT_EQ(logical_range.byte_length, 4096u);
  EXPECT_EQ(static_cast<uint8_t*>(storage.payload.host_pointer)[4095], 0x5A);
  iree_hal_external_storage_release(&storage);

  iree_hal_buffer_release(buffer);
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.bytes_reserved, 0u);
  EXPECT_EQ(stats.bytes_committed, 0u);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.slab_count, 0u);
  EXPECT_EQ(stats.reserve_count, 1u);
  EXPECT_EQ(stats.release_count, 1u);
  iree_hal_pool_release(pool);
}

TEST(HeapPoolTest, BorrowedMaterializationRetainsBackingAfterReservation) {
  const iree_hal_pool_host_access_t host_access = {
      .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      .modes = IREE_HAL_MAPPING_MODE_SCOPED,
  };
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_pool_create(host_access, iree_allocator_system(),
                                           iree_allocator_system(), &pool));

  const iree_hal_pool_reservation_request_t request =
      iree_hal_make_pool_reservation_request(64, 16);
  iree_hal_pool_reservation_t reservation = {};
  iree_hal_pool_acquire_info_t info = {};
  iree_hal_pool_acquire_result_t result = IREE_HAL_POOL_ACQUIRE_NONE;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
      pool, 1, &request,
      /*requester_frontier=*/nullptr, &reservation, &info, &result,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      pool, 1, &request, &reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
      &buffer));
  iree_hal_pool_release_reservations(pool, 1, &reservation,
                                     /*death_frontier=*/nullptr);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.bytes_reserved, 0u);
  EXPECT_EQ(stats.bytes_committed, 64u);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.slab_count, 1u);

  iree_hal_buffer_mapping_t mapping = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, IREE_HAL_WHOLE_BUFFER, &mapping));
  memset(mapping.contents.data, 0xA5, mapping.contents.data_length);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_buffer_release(buffer);
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);
  EXPECT_EQ(stats.slab_count, 0u);
  EXPECT_EQ(stats.release_count, 1u);
  iree_hal_pool_release(pool);
}

}  // namespace
