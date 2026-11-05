// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <utility>
#include <vector>

#include "loom/serve/runtime/storage/memory_test_fixture.h"

namespace {

using loom_serve::testing::memory::MemoryTest;

TEST_F(MemoryTest, FootprintRetainsAllocationClassesForAdmittedCommit) {
  const auto bulk = loom_serve_pool_weight_geometry(storage_owner);
  const auto state = loom_serve_pool_state_geometry(storage_owner);
  const uint64_t mapping = state.mapping_size;
  ASSERT_LT(2 * mapping, bulk.mapping_size);
  ASSERT_GE(state.backing_size, 2 * mapping);
  const uint64_t physical_bytes =
      bulk.mapping_size + state.backing_size + 2 * mapping;
  ASSERT_NO_FATAL_FAILURE(
      UseDeferredBacking(physical_bytes, nullptr, /*distinct_classes=*/true));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, bulk, bulk.mapping_size + mapping, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, bulk, 2 * mapping, 256, &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, state, mapping, 256,
                                                    &statistics[2], &third));
  // A short weight tail shares a state parent; a larger tail is dedicated.
  // Overlapping ranges must preserve both byte totals and allocation classes.
  const loom_serve_memory_range_t ranges[] = {
      {first, bulk.mapping_size - 4, 8},
      {first, bulk.mapping_size, mapping},
      {second, 0, 4},
      {third, 0, 4},
      {second, 4, 4},
  };
  loom_serve_memory_footprint_t footprint;
  IREE_ASSERT_OK(loom_serve_memory_ranges_footprint(IREE_ARRAYSIZE(ranges),
                                                    ranges, &footprint));
  const uint64_t logical_bytes = bulk.mapping_size + 4 * mapping;
  EXPECT_EQ(footprint.total_bytes, logical_bytes);
  EXPECT_EQ(footprint.missing_bytes, logical_bytes);
  EXPECT_EQ(footprint.missing_demand.weight_bytes, bulk.mapping_size);
  EXPECT_EQ(footprint.missing_demand.packed_bytes, 2 * mapping);
  EXPECT_EQ(footprint.missing_demand.dedicated_bytes, 2 * mapping);
  loom_serve_memory_admission_t admission;
  IREE_ASSERT_OK(loom_serve_memory_pool_prepare(pool, footprint.missing_demand,
                                                &admission));
  ASSERT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
  EXPECT_EQ(loom_serve_memory_pool_snapshot(pool).pending_bytes, 0u);
  EXPECT_EQ(loom_serve_memory_pool_snapshot(pool).committed_bytes, 0u);

  deferred_backing->defer = true;
  loom_serve_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_serve_memory_pool_commit_admitted_async(
      pool, IREE_ARRAYSIZE(ranges), ranges, nullptr, 0, &operation));
  const auto release_operation = [](loom_serve_memory_operation_t* operation) {
    loom_serve_memory_operation_await(operation);
    loom_serve_memory_operation_free(operation);
  };
  std::unique_ptr<loom_serve_memory_operation_t, decltype(release_operation)>
      operation_owner(operation, release_operation);
  deferred_backing->WaitForAcquisition();
  EXPECT_EQ(loom_serve_memory_pool_snapshot(pool).pending_bytes, logical_bytes);
  IREE_EXPECT_OK(loom_serve_memory_ranges_footprint(IREE_ARRAYSIZE(ranges),
                                                    ranges, &footprint));
  EXPECT_EQ(footprint.total_bytes, logical_bytes);
  EXPECT_EQ(footprint.missing_bytes, 0u);
  EXPECT_EQ(footprint.missing_demand.weight_bytes, 0u);
  EXPECT_EQ(footprint.missing_demand.packed_bytes, 0u);
  EXPECT_EQ(footprint.missing_demand.dedicated_bytes, 0u);
  deferred_backing->ResumeAcquisitions();
  loom_serve_memory_operation_await(operation);
  IREE_ASSERT_OK(loom_serve_memory_operation_query_status(operation));
  operation_owner.reset();
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            physical_bytes);
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  RoundTrip(loom_serve_virtual_buffer_handle(first), bulk.mapping_size,
            0x23456789);
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x3456789A);
  RoundTrip(loom_serve_virtual_buffer_handle(third), 0, 0x456789AB);
}

TEST_F(MemoryTest, BackingSelectionCountsActualUnionAndClearsFailedVisitors) {
  const auto geometry = loom_serve_pool_state_geometry(storage_owner);
  const uint64_t sparse_length = iree_device_align(
      iree_min(kHighOffset, reservation_capabilities.max_byte_length / 2),
      geometry.mapping_size);
  ASSERT_GT(sparse_length, 4 * geometry.mapping_size);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, geometry, sparse_length, 256, &statistics[0], &first));
  const uint64_t high = sparse_length - geometry.mapping_size;
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 1));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, high, 1));
  EXPECT_TRUE(
      loom_serve_virtual_buffer_is_backed(first, 0, geometry.mapping_size));
  EXPECT_FALSE(
      loom_serve_virtual_buffer_is_backed(first, 0, geometry.mapping_size + 1));
  EXPECT_TRUE(
      loom_serve_virtual_buffer_is_backed(first, high, geometry.mapping_size));
  struct Selection {
    // Exact mapped extents observed without requiring directory iteration
    // order.
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    // Fail the first callback; every remaining mark must still be cleared.
    bool fail = false;
  } selected;
  const auto visit = [](void* self, uint64_t offset, uint64_t length) {
    auto* selected = static_cast<Selection*>(self);
    selected->ranges.emplace_back(offset, length);
    return selected->fail ? iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                             "query scratch")
                          : iree_ok_status();
  };
  for (bool fail : {true, false}) {
    selected = {{}, fail};
    native.ResetAllocationMetrics();
    // A huge declaration neither allocates metadata nor charges absent pages.
    loom_serve_virtual_buffer_select_backing(first, 0, sparse_length);
    loom_serve_virtual_buffer_select_backing(first, high + 3, 16);
    auto status =
        loom_serve_virtual_buffer_visit_backing(first, visit, &selected);
    if (fail) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
      EXPECT_EQ(selected.ranges.size(), 1u);
    } else {
      IREE_ASSERT_OK(status);
      std::sort(selected.ranges.begin(), selected.ranges.end());
      EXPECT_EQ(selected.ranges, (std::vector<std::pair<uint64_t, uint64_t>>{
                                     {0, geometry.mapping_size},
                                     {high, geometry.mapping_size}}));
    }
    EXPECT_EQ(native.allocation_metrics().request_count, 0u);
    const auto count = selected.ranges.size();
    IREE_ASSERT_OK(
        loom_serve_virtual_buffer_visit_backing(first, visit, &selected));
    EXPECT_EQ(selected.ranges.size(), count);
    const loom_serve_memory_range_t range = {first, high, 1};
    loom_serve_memory_footprint_t footprint;
    IREE_ASSERT_OK(loom_serve_memory_ranges_footprint(1, &range, &footprint));
    EXPECT_EQ(footprint.total_bytes, geometry.mapping_size);
    EXPECT_EQ(footprint.missing_bytes, 0u);
  }
  loom_serve_virtual_buffer_begin_trim(first);
  Keep(first, high, 1);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  EXPECT_FALSE(loom_serve_virtual_buffer_is_backed(first, 0, 1));
  EXPECT_TRUE(loom_serve_virtual_buffer_is_backed(first, high, 1));
  RoundTrip(loom_serve_virtual_buffer_handle(first), high, 0x12345678);
}

TEST_F(MemoryTest, PendingAndFailedMappingRetainActualBacking) {
  mapping_queue.Initialize(loom_serve_device_transfer_queue(device));
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking(kSlabSize, &mapping_queue));
  const auto geometry = loom_serve_pool_weight_geometry(storage_owner);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, geometry, 2 * kSlabSize, 256, &statistics[0], &first));
  deferred_backing->defer = true;
  mapping_queue.fail_next_map = true;
  const loom_serve_memory_range_t range = {first, 0, kSlabSize};
  loom_serve_memory_admission_t admission;
  loom_serve_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
      pool, 1, &range, nullptr, 0, &admission, &operation));
  const auto free_operation = [](loom_serve_memory_operation_t* operation) {
    loom_serve_memory_operation_await(operation);
    loom_serve_memory_operation_free(operation);
  };
  const std::unique_ptr<loom_serve_memory_operation_t, decltype(free_operation)>
      owner(operation, free_operation);
  deferred_backing->WaitForAcquisition();
  EXPECT_EQ(statistics[0].pending_bytes, kSlabSize);
  const auto visit = [](void* self, uint64_t offset, uint64_t length) {
    EXPECT_EQ(offset, 0u);
    *static_cast<uint64_t*>(self) += length;
    return iree_ok_status();
  };
  uint64_t pending_bytes = 0;
  loom_serve_virtual_buffer_select_backing(first, 0, 2 * kSlabSize);
  IREE_ASSERT_OK(
      loom_serve_virtual_buffer_visit_backing(first, visit, &pending_bytes));
  EXPECT_EQ(pending_bytes, kSlabSize);
  deferred_backing->ResumeAcquisitions();
  loom_serve_memory_operation_await(operation);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_memory_operation_query_status(operation));
  EXPECT_EQ(statistics[0].pending_bytes, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, kSlabSize);
  uint64_t failed_bytes = 0;
  loom_serve_virtual_buffer_select_backing(first, 0, 2 * kSlabSize);
  IREE_ASSERT_OK(
      loom_serve_virtual_buffer_visit_backing(first, visit, &failed_bytes));
  EXPECT_EQ(failed_bytes, kSlabSize);
  // Failure is not release. The reservation's ordinary checked detach in
  // teardown remains responsible for returning this physical allocation.
}

}  // namespace
