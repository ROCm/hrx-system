// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/transient_buffer.h"

#include "iree/hal/memory/heap_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TransientBufferTest : public ::testing::TestWithParam<size_t> {};

TEST_P(TransientBufferTest, SubspanTracksCommitAndExplicitDecommit) {
  iree_arena_block_pool_t blocks;
  iree_arena_block_pool_initialize(GetParam(), iree_allocator_system(),
                                   &blocks);
  const iree_hal_pool_host_access_t access = {
      .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      .modes = IREE_HAL_MAPPING_MODE_SCOPED,
  };
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_pool_create(access, iree_allocator_system(),
                                           iree_allocator_system(), &pool));
  iree_hal_buffer_t* root = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_fabric_transient_buffer_create(
      {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      access.access, IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED, 128, 128, pool,
      &blocks, &root));
  iree_hal_buffer_t* span = nullptr;
  IREE_ASSERT_OK(
      iree_hal_buffer_subspan(root, 17, 31, iree_allocator_system(), &span));
  iree_hal_buffer_mapping_t mapping;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_buffer_map_range(span, IREE_HAL_MAPPING_MODE_SCOPED,
                                access.access, IREE_HAL_BUFFER_MAP_FLAG_NONE, 0,
                                IREE_HAL_WHOLE_BUFFER, &mapping));

  const auto request = iree_hal_make_pool_reservation_request(128, 16);
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool, 1, &request, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation,
      &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      pool, 1, &request, &reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
      &backing));
  const iree_hal_pool_reservation_view_t view = {
      .buffer = backing,
      .byte_offset = iree_hal_buffer_byte_offset(backing),
      .byte_length = iree_hal_buffer_byte_length(backing),
      .memory = iree_hal_buffer_memory_view(backing),
  };
  iree_hal_amd_fabric_transient_buffer_attach_reservation(root, pool,
                                                          &reservation);
  iree_hal_amd_fabric_transient_buffer_stage_backing(root, &view);
  iree_hal_buffer_release(backing);
  iree_hal_amd_fabric_transient_buffer_commit(root);
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      span, IREE_HAL_MAPPING_MODE_SCOPED, access.access,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, IREE_HAL_WHOLE_BUFFER, &mapping));
  EXPECT_EQ(mapping.contents.data_length, 31u);
  memset(mapping.contents.data, 0xA7, mapping.contents.data_length);
  const iree_hal_buffer_native_binding_slot_t host_slot = {
      .index = span->host_binding_index,
      .type = IREE_HAL_BUFFER_INTERFACE_HOST,
  };
  EXPECT_EQ(iree_hal_buffer_native_binding(span, host_slot).host_pointer,
            mapping.contents.data);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      root, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 17, 31, &mapping));
  for (size_t i = 0; i < mapping.contents.data_length; ++i) {
    EXPECT_EQ(mapping.contents.data[i], 0xA7);
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_pool_t* returned_pool = nullptr;
  IREE_ASSERT_OK(
      iree_hal_buffer_allocation_begin_dealloca(root, &returned_pool));
  EXPECT_EQ(returned_pool, pool);
  iree_hal_buffer_allocation_abort_dealloca(root);
  IREE_ASSERT_OK(
      iree_hal_buffer_allocation_begin_dealloca(root, &returned_pool));
  iree_hal_buffer_allocation_take_dealloca_reservation(root, &returned_pool,
                                                       &reservation);
  iree_hal_buffer_allocation_decommit(root);
  iree_hal_pool_release_reservations(returned_pool, 1, &reservation, nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_buffer_map_range(span, IREE_HAL_MAPPING_MODE_SCOPED,
                                access.access, IREE_HAL_BUFFER_MAP_FLAG_NONE, 0,
                                IREE_HAL_WHOLE_BUFFER, &mapping));
  iree_hal_buffer_release(root);
  iree_hal_buffer_release(span);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.bytes_committed, 0u);
  iree_hal_pool_release(pool);
  iree_arena_block_pool_deinitialize(&blocks);
}

// Both dedicated oversized metadata and ordinary reusable blocks own exactly
// the same buffer lifecycle; block geometry is not a native-table size limit.
INSTANTIATE_TEST_SUITE_P(MetadataStorage, TransientBufferTest,
                         ::testing::Values(128, 4096));

}  // namespace
