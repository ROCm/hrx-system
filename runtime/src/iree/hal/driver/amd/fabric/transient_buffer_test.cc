// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/fabric/transient_buffer.h"

#include "iree/hal/memory/heap_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

IREE_ASYNC_FIXED_FRONTIER_TYPE(ReuseHistory, UINT8_MAX);

class TransientBufferTest : public ::testing::TestWithParam<uint8_t> {};

TEST_P(TransientBufferTest, SubspanTracksCommitAndExplicitDecommit) {
  iree_hal_amd_fabric_transient_buffer_pool_t buffers;
  iree_hal_amd_fabric_transient_buffer_pool_initialize(iree_allocator_system(),
                                                       &buffers);
  const iree_hal_pool_host_access_t access = {
      .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      .modes = IREE_HAL_MAPPING_MODE_SCOPED,
  };
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_pool_create(access, iree_allocator_system(),
                                           iree_allocator_system(), &pool));
  const iree_hal_pool_reservation_request_t requests[] = {
      iree_hal_make_pool_reservation_request(128, 16),
      iree_hal_make_pool_reservation_request(128, 16),
      iree_hal_make_pool_reservation_request(128, 16)};
  iree_hal_buffer_t* roots[3] = {};
  IREE_ASSERT_OK(iree_hal_amd_fabric_transient_buffer_create_batch(
      {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      access.access, IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED, pool, &buffers,
      IREE_ARRAYSIZE(requests), requests, roots));
  iree_hal_buffer_t* root = roots[1];
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
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
      pool, 1, &request, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation,
      &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      pool, 1, &request, &reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
      &backing));
  ReuseHistory history = {};
  const uint8_t history_count = GetParam();
  history.entry_count = history_count;
  for (uint32_t i = 0; i < history_count; ++i) {
    history.entries[i] = {i + 1, 41};
  }
  iree_hal_pool_reservation_view_t view = {
      .buffer = backing,
      .byte_offset = iree_hal_buffer_byte_offset(backing),
      .byte_length = iree_hal_buffer_byte_length(backing),
      .memory = iree_hal_buffer_memory_view(backing),
  };
  view.memory.reuse_frontier =
      history_count ? iree_async_fixed_frontier_as_const_frontier(&history)
                    : nullptr;
  iree_hal_amd_fabric_transient_buffer_attach_reservation(root, pool,
                                                          &reservation);
  IREE_ASSERT_OK(
      iree_hal_amd_fabric_transient_buffer_stage_backing(root, &view));
  iree_hal_buffer_release(backing);
  iree_hal_amd_fabric_transient_buffer_commit(root);
  // The reservation owner may recycle its history before the old native view
  // retires. Every already-published view must keep the old epoch's facts.
  const auto published = iree_hal_buffer_memory_view(root);
  for (uint32_t i = 0; i < history_count; ++i) {
    history.entries[i].epoch = 99;
    EXPECT_EQ(published.reuse_frontier->entries[i].epoch, 41u);
  }
  EXPECT_EQ(published.reuse_frontier != nullptr, history_count != 0);
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
  iree_hal_pool_release_reservations(returned_pool, 1, &reservation, nullptr);
  iree_hal_buffer_release(root);
  root = iree_hal_buffer_allocated_buffer(span);
  // Neighbor slots return independently while this committed root's child
  // still borrows its native table and copied history.
  iree_hal_buffer_release(roots[0]);
  iree_hal_buffer_release(roots[2]);
  iree_hal_amd_fabric_transient_buffer_pool_trim(&buffers);
  iree_hal_buffer_t* replacement = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_fabric_transient_buffer_create_batch(
      {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      access.access, IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED, pool, &buffers, 1,
      requests, &replacement));
  EXPECT_NE(replacement, root);
  iree_hal_buffer_release(replacement);
  EXPECT_NE(iree_hal_buffer_native_binding(root, host_slot).host_pointer,
            nullptr);
  for (uint32_t i = 0; i < history_count; ++i) {
    EXPECT_EQ(published.reuse_frontier->entries[i].epoch, 41u);
  }
  iree_hal_buffer_allocation_decommit(root);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_buffer_map_range(span, IREE_HAL_MAPPING_MODE_SCOPED,
                                access.access, IREE_HAL_BUFFER_MAP_FLAG_NONE, 0,
                                IREE_HAL_WHOLE_BUFFER, &mapping));
  iree_hal_buffer_release(span);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.bytes_committed, 0u);
  iree_hal_pool_release(pool);
  iree_hal_amd_fabric_transient_buffer_pool_deinitialize(&buffers);
}

// Inline and separately slotted history preserve the same allocation epoch.
INSTANTIATE_TEST_SUITE_P(MetadataStorage, TransientBufferTest,
                         ::testing::Values(0, 1, 12, 13, 255));

}  // namespace
