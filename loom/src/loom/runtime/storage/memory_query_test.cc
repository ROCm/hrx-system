// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <utility>
#include <vector>

#include "loom/runtime/storage/memory_test_fixture.h"

namespace {

using loom_runtime::testing::memory::MemoryTest;

TEST_F(MemoryTest, BackingSelectionCountsActualUnionAndClearsFailedVisitors) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t sparse_length = iree_device_align(
      iree_min(kHighOffset, reservation_capabilities.max_byte_length / 2),
      geometry.mapping_size);
  ASSERT_GT(sparse_length, 4 * geometry.mapping_size);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, sparse_length, 256, &statistics[0], &first));
  const uint64_t high = sparse_length - geometry.mapping_size;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, 1));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, high, 1));
  EXPECT_TRUE(
      loom_runtime_virtual_buffer_is_backed(first, 0, geometry.mapping_size));
  EXPECT_FALSE(loom_runtime_virtual_buffer_is_backed(
      first, 0, geometry.mapping_size + 1));
  EXPECT_TRUE(loom_runtime_virtual_buffer_is_backed(first, high,
                                                    geometry.mapping_size));
  struct Selection {
    // Exact mapped extents observed without requiring directory iteration
    // order.
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    // Fail the first callback; every remaining mark must still be cleared.
    bool fail = false;
  } selected;
  const auto visit = [](void* self, iree_device_size_t offset,
                        iree_device_size_t length) {
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
    loom_runtime_virtual_buffer_select_backing(first, 0, sparse_length);
    loom_runtime_virtual_buffer_select_backing(first, high + 3, 16);
    auto status =
        loom_runtime_virtual_buffer_visit_backing(first, visit, &selected);
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
        loom_runtime_virtual_buffer_visit_backing(first, visit, &selected));
    EXPECT_EQ(selected.ranges.size(), count);
    const loom_runtime_memory_range_t range = {first, high, 1};
    loom_runtime_memory_footprint_t footprint;
    IREE_ASSERT_OK(loom_runtime_memory_ranges_footprint(1, &range, &footprint));
    EXPECT_EQ(footprint.total_bytes, geometry.mapping_size);
    EXPECT_EQ(footprint.missing_bytes, 0u);
  }
  loom_runtime_virtual_buffer_begin_trim(first);
  Keep(first, high, 1);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_trim(first));
  EXPECT_FALSE(loom_runtime_virtual_buffer_is_backed(first, 0, 1));
  EXPECT_TRUE(loom_runtime_virtual_buffer_is_backed(first, high, 1));
  RoundTrip(loom_runtime_virtual_buffer_handle(first), high, 0x12345678);
}

TEST_F(MemoryTest, PendingAndFailedMappingRetainActualBacking) {
  mapping_queue.Initialize(loom_runtime_device_transfer_queue(device));
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking(kSlabSize, &mapping_queue));
  const auto geometry = loom_runtime_pool_weight_geometry(storage_owner);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, 2 * kSlabSize, 256, &statistics[0], &first));
  deferred_backing->defer = true;
  mapping_queue.fail_next_map = true;
  const loom_runtime_memory_range_t range = {first, 0, kSlabSize};
  loom_runtime_memory_admission_t admission;
  loom_runtime_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_runtime_memory_pool_commit_async(
      pool, 1, &range, nullptr, 0, &admission, &operation));
  const auto free_operation = [](loom_runtime_memory_operation_t* operation) {
    loom_runtime_memory_operation_await(operation);
    loom_runtime_memory_operation_free(operation);
  };
  const std::unique_ptr<loom_runtime_memory_operation_t,
                        decltype(free_operation)>
      owner(operation, free_operation);
  deferred_backing->WaitForAcquisition();
  EXPECT_EQ(statistics[0].pending_bytes, kSlabSize);
  const auto visit = [](void* self, iree_device_size_t offset,
                        iree_device_size_t length) {
    EXPECT_EQ(offset, 0u);
    *static_cast<uint64_t*>(self) += length;
    return iree_ok_status();
  };
  uint64_t pending_bytes = 0;
  loom_runtime_virtual_buffer_select_backing(first, 0, 2 * kSlabSize);
  IREE_ASSERT_OK(
      loom_runtime_virtual_buffer_visit_backing(first, visit, &pending_bytes));
  EXPECT_EQ(pending_bytes, kSlabSize);
  deferred_backing->ResumeAcquisitions();
  loom_runtime_memory_operation_await(operation);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_runtime_memory_operation_query_status(operation));
  EXPECT_EQ(statistics[0].pending_bytes, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, kSlabSize);
  uint64_t failed_bytes = 0;
  loom_runtime_virtual_buffer_select_backing(first, 0, 2 * kSlabSize);
  IREE_ASSERT_OK(
      loom_runtime_virtual_buffer_visit_backing(first, visit, &failed_bytes));
  EXPECT_EQ(failed_bytes, kSlabSize);
  // Failure is not release. The reservation's ordinary checked detach in
  // teardown remains responsible for returning this physical allocation.
}

}  // namespace
