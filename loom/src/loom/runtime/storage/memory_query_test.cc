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

TEST_F(MemoryTest, SlabLayoutFollowsPhysicalLeasesAcrossReuseAndDisposal) {
  const auto weight = loom_runtime_pool_weight_geometry(storage_owner);
  const auto state = loom_runtime_pool_state_geometry(storage_owner);
  loom_runtime_slab_layout_t slabs[2];
  ASSERT_EQ(loom_runtime_memory_pool_query_slabs(pool, 2, slabs), 0u);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, weight, 2 * kSlabSize, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, 2 * kSlabSize));
  ASSERT_EQ(loom_runtime_memory_pool_query_slabs(pool, 2, slabs), 2u);
  EXPECT_EQ(slabs[0].state, LOOM_RUNTIME_SLAB_READY);
  EXPECT_EQ(slabs[0].leased_count, slabs[0].page_count);
  EXPECT_EQ(slabs[0].free_mask, 0u);
  IREE_ASSERT_OK(Trim(first));
  loom_runtime_memory_pool_trim(pool, kSlabSize);
  IREE_ASSERT_OK(loom_runtime_memory_pool_await(pool));
  ASSERT_EQ(loom_runtime_memory_pool_query_slabs(pool, 2, slabs), 2u);
  ASSERT_NE(slabs[0].page_count == 0, slabs[1].page_count == 0);
  const size_t live = slabs[0].page_count ? 0 : 1;
  const auto& slab = slabs[live];
  EXPECT_EQ(slab.page_bytes, state.mapping_size);
  ASSERT_EQ(slab.page_count, kSlabSize / state.mapping_size);
  EXPECT_EQ(slab.leased_count, 0u);
  loom_runtime_slab_layout_t latest;
  ASSERT_EQ(loom_runtime_memory_pool_query_slabs(pool, 1, &latest), 2u);
  EXPECT_EQ(latest.page_count, slabs[1].page_count);

  // Reuse this same physical slab for a different virtual root. Crossing a
  // leaf boundary moves the window; punching a hole moves it back.
  const size_t count = std::min<size_t>(65, slab.page_count);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, state, count * state.mapping_size, 256, &statistics[1], &second));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(
      second, 0, count * state.mapping_size));
  ASSERT_EQ(loom_runtime_memory_pool_query_slabs(pool, 2, slabs), 2u);
  EXPECT_EQ(slab.leased_count, count);
  EXPECT_EQ(slab.page_offset, count < slab.page_count ? count / 64 * 64 : 0u);
  const size_t hole = count / 2;
  IREE_ASSERT_OK(Trim(second, hole * state.mapping_size, state.mapping_size));
  ASSERT_EQ(loom_runtime_memory_pool_query_slabs(pool, 2, slabs), 2u);
  EXPECT_EQ(slab.leased_count, count - 1);
  EXPECT_EQ(slab.page_offset, 0u);
  // All other captured slots below count remain leased.
  const uint64_t prefix = UINT64_MAX >> (64 - std::min<size_t>(64, count));
  EXPECT_EQ(slab.free_mask & prefix, UINT64_C(1) << hole);
}

TEST_F(MemoryTest, PreparedCommitCanBeCancelledAndDeniedWithoutBacking) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, 3 * page, 256, &statistics[0], &first));
  const loom_runtime_memory_range_t ranges[] = {
      {first, page, 2 * page}, {first, page - 4, 8}, {first, 2 * page, 256}};
  loom_runtime_memory_admission_t admission;
  loom_runtime_memory_commit_t prepared = {};
  IREE_ASSERT_OK(loom_runtime_memory_pool_prepare_commit(
      pool, IREE_ARRAYSIZE(ranges), ranges, &admission, &prepared));
  ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
  ASSERT_GT(prepared.mapping_count, 0u);
  EXPECT_EQ(loom_runtime_memory_demand_total(prepared.demand), 3 * page);
  EXPECT_EQ(statistics[0].pending_bytes, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, 0u);
  EXPECT_EQ(loom_runtime_memory_pool_retained_bytes(pool), 0u);
  // Admission queries can read the footprint while selection is retained.
  // Their temporary marks cannot discard the prepared transaction's entries.
  loom_runtime_memory_footprint_t footprint;
  IREE_ASSERT_OK(loom_runtime_memory_ranges_footprint(IREE_ARRAYSIZE(ranges),
                                                      ranges, &footprint));
  EXPECT_EQ(footprint.total_bytes, 3 * page);
  EXPECT_EQ(footprint.missing_bytes, 3 * page);
  IREE_ASSERT_OK(loom_runtime_memory_pool_cancel_commit(&prepared));

  // Cancellation neither keeps capacity nor leaves marks affecting a retry.
  IREE_ASSERT_OK(loom_runtime_memory_pool_prepare_commit(
      pool, IREE_ARRAYSIZE(ranges), ranges, &admission, &prepared));
  ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
  EXPECT_EQ(loom_runtime_memory_demand_total(prepared.demand), 3 * page);
  loom_runtime_memory_operation_t* operation = nullptr;
  // Required parameters already fill the budget, leaving no room for state.
  const loom_runtime_memory_demand_t parameters = {
      .weight_bytes = 3 * kSlabSize,
  };
  IREE_ASSERT_OK(loom_runtime_memory_pool_commit_prepared_async(
      pool, &prepared, parameters, {}, nullptr, 0, &admission, &operation));
  EXPECT_EQ(prepared.ranges, nullptr);
  EXPECT_EQ(operation, nullptr);
  EXPECT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_CAPACITY);
  EXPECT_EQ(statistics[0].pending_bytes, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, 0u);
  EXPECT_EQ(loom_runtime_memory_pool_retained_bytes(pool), 0u);

  IREE_ASSERT_OK(loom_runtime_memory_pool_commit_async(
      pool, IREE_ARRAYSIZE(ranges), ranges, nullptr, 0, &admission,
      &operation));
  ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
  loom_runtime_memory_operation_await(operation);
  IREE_ASSERT_OK(loom_runtime_memory_operation_query_status(operation));
  loom_runtime_memory_operation_free(operation);
  EXPECT_EQ(statistics[0].committed_bytes, 3 * page);
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 2 * page, 0x87654321);

  // Fully backed requests need neither edit storage nor native synchronization.
  native.ResetAllocationMetrics();
  IREE_ASSERT_OK(loom_runtime_memory_pool_prepare_commit(
      pool, IREE_ARRAYSIZE(ranges), ranges, &admission, &prepared));
  ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
  EXPECT_EQ(prepared.mapping_count, 0u);
  EXPECT_EQ(loom_runtime_memory_demand_total(prepared.demand), 0u);
  IREE_ASSERT_OK(loom_runtime_memory_pool_commit_prepared_async(
      pool, &prepared, {}, {}, nullptr, 0, &admission, &operation));
  EXPECT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
  EXPECT_EQ(operation, nullptr);
  EXPECT_EQ(native.allocation_metrics().request_count, 0u);
}

TEST_F(MemoryTest, FailedPreparedCommitAllocationLeavesRetryIntact) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, geometry.mapping_size, 256, &statistics[0], &first));
  const loom_runtime_memory_range_t range = {first, 0, 256};
  loom_runtime_memory_admission_t admission;
  loom_runtime_memory_commit_t prepared = {};
  native.FailNextAllocation(IREE_ALLOCATOR_COMMAND_CALLOC);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_runtime_memory_pool_prepare_commit(
                            pool, 1, &range, &admission, &prepared));
  EXPECT_EQ(prepared.ranges, nullptr);

  IREE_ASSERT_OK(loom_runtime_memory_pool_prepare_commit(
      pool, 1, &range, &admission, &prepared));
  ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
  native.FailNextAllocation(IREE_ALLOCATOR_COMMAND_CALLOC);
  loom_runtime_memory_operation_t* operation = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_runtime_memory_pool_commit_prepared_async(
          pool, &prepared, {}, {}, nullptr, 0, &admission, &operation));
  EXPECT_EQ(prepared.ranges, nullptr);
  EXPECT_EQ(operation, nullptr);
  EXPECT_EQ(statistics[0].pending_bytes, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, 0u);
  EXPECT_EQ(loom_runtime_memory_pool_retained_bytes(pool), 0u);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, 256));
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);
}

TEST_F(MemoryTest, FailedCancellationShrinkClearsSelectionAndKeepsLiveData) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  // Spread the rejected selection over enough directory words to require
  // compaction when only the original page survives.
  constexpr uint64_t kPages = 1024;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, kPages * page, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, page));
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);

  const loom_runtime_memory_range_t range = {first, 0, kPages * page};
  loom_runtime_memory_admission_t admission;
  loom_runtime_memory_commit_t prepared = {};
  IREE_ASSERT_OK(loom_runtime_memory_pool_prepare_commit(
      pool, 1, &range, &admission, &prepared));
  ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
  EXPECT_EQ(prepared.mapping_count, kPages - 1);
  native.FailNextAllocation(IREE_ALLOCATOR_COMMAND_CALLOC);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_runtime_memory_pool_cancel_commit(&prepared));
  EXPECT_EQ(prepared.ranges, nullptr);
  EXPECT_EQ(statistics[0].pending_bytes, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, page);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);

  // Failed host compaction cannot leave a selection that suppresses new work.
  IREE_ASSERT_OK(loom_runtime_memory_pool_prepare_commit(
      pool, 1, &range, &admission, &prepared));
  ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
  EXPECT_EQ(prepared.mapping_count, kPages - 1);
  EXPECT_EQ(loom_runtime_memory_demand_total(prepared.demand),
            (kPages - 1) * page);
  IREE_ASSERT_OK(loom_runtime_memory_pool_cancel_commit(&prepared));
}

TEST_F(MemoryTest, FailedPreparationCancelsOnlyItsSelectedSparsePrefix) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  const uint64_t length =
      iree_min(kHighOffset, reservation_capabilities.max_byte_length / 2);
  ASSERT_GT(length, 32 * page);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, length, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, page));

  // The first few selections fit the existing directory; its next growth
  // fails. Cancellation consumes that produced prefix, not the huge untouched
  // virtual suffix whose backing was never inspected.
  const loom_runtime_memory_range_t range = {first, 0, length};
  loom_runtime_memory_admission_t admission;
  loom_runtime_memory_commit_t prepared = {};
  native.FailNextAllocation(IREE_ALLOCATOR_COMMAND_CALLOC);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_runtime_memory_pool_prepare_commit(
                            pool, 1, &range, &admission, &prepared));
  EXPECT_EQ(prepared.ranges, nullptr);
  EXPECT_EQ(statistics[0].pending_bytes, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, page);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, page, page));
  RoundTrip(loom_runtime_virtual_buffer_handle(first), page, 0x12345678);
}

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

  IREE_ASSERT_OK(Trim(first, 0, high));
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
