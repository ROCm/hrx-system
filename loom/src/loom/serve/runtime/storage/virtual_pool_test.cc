// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/storage/memory_test_fixture.h"

namespace {

using VirtualPoolTest = loom_serve::testing::memory::MemoryTest;

TEST_F(VirtualPoolTest,
       VirtualCommandPoolReusesRetainedSlabsAcrossTrimAndRegrowth) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  const auto geometry = loom_serve_device_weight_geometry(device);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, geometry, 2 * kSlabSize, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, geometry, 2 * kSlabSize, 256, &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, geometry, kSlabSize, 256, &statistics[2], &third));
  auto* const scratch_address = loom_serve_virtual_buffer_handle(second);

  // Logical command allocation retains the virtual range, not its physical
  // slabs. The same TLSF instance survives every physical detach below.
  const iree_hal_tlsf_pool_options_t options = {
      .tlsf_options = {.alignment = 256},
  };
  iree_hal_pool_t* logical_handle = nullptr;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create_from_buffer(
      scratch_address, 0, IREE_HAL_WHOLE_BUFFER, &options,
      iree_allocator_system(), &logical_handle));
  const std::unique_ptr<iree_hal_pool_t, decltype(&iree_hal_pool_release)>
      logical(logical_handle, iree_hal_pool_release);

  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 2 * kSlabSize));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(third, 0, kSlabSize));
  ASSERT_NO_FATAL_FAILURE(
      RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678));
  ASSERT_NO_FATAL_FAILURE(
      RoundTrip(loom_serve_virtual_buffer_handle(third), 0, 0x87654321));
  iree_hal_pool_stats_t baseline;
  iree_hal_pool_query_stats(deferred_backing, &baseline);

  // Live retained data occupies the entire physical budget. Preparing scratch
  // cannot take its backing or impair the independently usable warm range.
  const loom_serve_memory_range_t scratch = {second, 0, 2 * kSlabSize};
  bool admitted = true;
  IREE_ASSERT_OK(
      loom_serve_memory_pool_try_commit(pool, 1, &scratch, &admitted));
  ASSERT_FALSE(admitted);
  ASSERT_NO_FATAL_FAILURE(
      CheckRetained(loom_serve_virtual_buffer_handle(third), 0, 0x87654321));
  loom_serve_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));

  // Repeat actual queue allocation, device writes, downloads and deallocation
  // with differently sized command workspaces. Source slabs are prepared once;
  // neither warm execution nor cross-role reuse acquires another native slab.
  auto execute = [&] {
    const iree_hal_pool_reservation_request_t requests[] = {
        iree_hal_make_pool_reservation_request(3 * kSlabSize / 2, 256),
        iree_hal_make_pool_reservation_request(kSlabSize / 4, 256),
    };
    iree_hal_buffer_t* buffers[IREE_ARRAYSIZE(requests)] = {};
    IREE_ASSERT_OK(loom_serve_execution_alloca_from_pool(
        execution, logical.get(), IREE_ARRAYSIZE(requests), requests, buffers));
    std::unique_ptr<iree_hal_buffer_t, decltype(&iree_hal_buffer_release)>
        first_buffer(buffers[0], iree_hal_buffer_release);
    std::unique_ptr<iree_hal_buffer_t, decltype(&iree_hal_buffer_release)>
        second_buffer(buffers[1], iree_hal_buffer_release);
    ASSERT_NO_FATAL_FAILURE(RoundTrip(buffers[0], kSlabSize - 128, 0xabcdef01));
    ASSERT_NO_FATAL_FAILURE(RoundTrip(buffers[1], 0, 0x10203040));
    uint64_t completion = 0;
    for (auto* buffer : buffers) {
      IREE_ASSERT_OK(
          loom_serve_execution_dealloca(execution, buffer, &completion));
    }
    first_buffer.reset();
    second_buffer.reset();
    IREE_ASSERT_OK(loom_serve_execution_drain(execution));
    // Collect retired queue roots after joining their device uses.
    loom_serve_workspace_statistics(loom_serve_device_workspace(device));
  };
  for (int cycle = 0; cycle < 2; ++cycle) {
    IREE_ASSERT_OK(
        loom_serve_memory_pool_try_commit(pool, 1, &scratch, &admitted));
    ASSERT_TRUE(admitted);
    EXPECT_EQ(loom_serve_virtual_buffer_handle(second), scratch_address);
    ASSERT_NO_FATAL_FAILURE(execute());
    ASSERT_NO_FATAL_FAILURE(execute());
    auto memory = loom_serve_memory_pool_statistics(pool);
    EXPECT_EQ(memory.committed_bytes, 3 * kSlabSize);
    EXPECT_EQ(memory.peak_bytes, 3 * kSlabSize);

    loom_serve_virtual_buffer_begin_trim(second);
    IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(second));
    IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 2 * kSlabSize));
    ASSERT_NO_FATAL_FAILURE(RoundTrip(loom_serve_virtual_buffer_handle(first),
                                      kSlabSize, 0x11223344));
    ASSERT_NO_FATAL_FAILURE(
        CheckRetained(loom_serve_virtual_buffer_handle(third), 0, 0x87654321));
    loom_serve_virtual_buffer_begin_trim(first);
    IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
    IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
    iree_hal_pool_stats_t reused;
    iree_hal_pool_query_stats(deferred_backing, &reused);
    EXPECT_EQ(reused.reserve_count, baseline.reserve_count);
  }

  // A true physical trim does not invalidate virtual addresses or TLSF reuse
  // history. Regrowth acquires only the two slabs the next invocation needs.
  loom_serve_virtual_buffer_begin_trim(third);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(third));
  loom_serve_memory_pool_trim(pool, 0);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, 0u);
  IREE_ASSERT_OK(
      loom_serve_memory_pool_try_commit(pool, 1, &scratch, &admitted));
  ASSERT_TRUE(admitted);
  EXPECT_EQ(loom_serve_virtual_buffer_handle(second), scratch_address);
  ASSERT_NO_FATAL_FAILURE(execute());
  iree_hal_pool_stats_t regrown;
  iree_hal_pool_query_stats(deferred_backing, &regrown);
  EXPECT_EQ(regrown.reserve_count, baseline.reserve_count + 2);
}

TEST_F(VirtualPoolTest, QueueDrivenVirtualBackingReusesRetainedSlabs) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  const auto geometry = loom_serve_device_weight_geometry(device);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, geometry, 2 * kSlabSize, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 2 * kSlabSize));
  ASSERT_NO_FATAL_FAILURE(
      RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x13579bdf));
  loom_serve_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  iree_hal_pool_stats_t baseline;
  iree_hal_pool_query_stats(deferred_backing, &baseline);
  ASSERT_NO_FATAL_FAILURE(CreateVirtualPool(true));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_virtual_pool_free(virtual_pool));
  const iree_hal_buffer_backing_facts_t* address_owner = nullptr;
  auto execute = [&] {
    // A dedicated logical extent still obtains two ordinary physical slabs.
    const auto request =
        iree_hal_make_pool_reservation_request(3 * kSlabSize / 2, 256);
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(
        loom_serve_execution_alloca(execution, 1, &request, &buffer));
    std::unique_ptr<iree_hal_buffer_t, decltype(&iree_hal_buffer_release)>
        allocation(buffer, iree_hal_buffer_release);
    ASSERT_NO_FATAL_FAILURE(RoundTrip(buffer, kSlabSize - 128, 0xabcdef01));
    const auto* backing = iree_hal_buffer_memory_view(buffer).backing;
    if (!address_owner) {
      address_owner = backing;
    }
    EXPECT_EQ(backing, address_owner);
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_dealloca(execution, buffer, &completion));
    allocation.reset();
    IREE_ASSERT_OK(loom_serve_execution_wait(execution, completion));
    IREE_ASSERT_OK(loom_serve_execution_drain(execution));
    while (true) {
      loom_serve_workspace_statistics(virtual_workspace);
      iree_hal_pool_stats_t logical;
      iree_hal_pool_query_stats(logical_pool, &logical);
      if (!logical.reservation_count) {
        break;
      }
      std::this_thread::yield();
    }
  };
  ASSERT_NO_FATAL_FAILURE(execute());
  IREE_ASSERT_OK(
      loom_serve_workspace_reserve(virtual_workspace, 8 * kSlabSize, 4096));
  iree_hal_pool_t* same_pool = nullptr;
  IREE_ASSERT_OK(
      loom_serve_workspace_acquire_pool(virtual_workspace, &same_pool));
  EXPECT_EQ(same_pool, logical_pool);
  iree_hal_pool_release(same_pool);
  ASSERT_NO_FATAL_FAILURE(execute());
  iree_hal_pool_stats_t reused;
  iree_hal_pool_query_stats(deferred_backing, &reused);
  EXPECT_EQ(reused.reserve_count, baseline.reserve_count);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            2 * kSlabSize);

  iree_hal_pool_trim(loom_serve_virtual_pool_handle(virtual_pool),
                     IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  ASSERT_NO_FATAL_FAILURE(AwaitVirtualPool());
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 2 * kSlabSize));
  ASSERT_NO_FATAL_FAILURE(
      RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x2468ace0));
  iree_hal_pool_query_stats(deferred_backing, &reused);
  EXPECT_EQ(reused.reserve_count, baseline.reserve_count);
  loom_serve_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  loom_serve_memory_pool_trim(pool, 0);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, 0u);
  ASSERT_NO_FATAL_FAILURE(execute());
  iree_hal_pool_query_stats(deferred_backing, &reused);
  EXPECT_EQ(reused.reserve_count, baseline.reserve_count + 2);
}

TEST_F(VirtualPoolTest, VirtualBackingBatchDenialPreservesWarmPrefix) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  ASSERT_NO_FATAL_FAILURE(CreateVirtualPool());
  auto* source = loom_serve_virtual_pool_handle(virtual_pool);
  struct Completion {
    // Number of accepted callbacks, including inline warm completion.
    size_t count = 0;
    // Capacity outcome from the most recent completion.
    iree_hal_pool_acquire_result_t result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
    // Owned failure, consumed by the test after completion.
    iree_status_t status = iree_ok_status();
  } completion;
  const iree_hal_pool_acquire_callback_t callback = {
      .fn =
          [](void* user_data, iree_status_t status,
             iree_hal_pool_acquire_result_t result) {
            auto* completion = static_cast<Completion*>(user_data);
            ++completion->count;
            completion->result = result;
            completion->status = status;
          },
      .user_data = &completion,
  };
  const iree_hal_pool_reservation_request_t requests[] = {
      iree_hal_make_pool_reservation_request(2 * kSlabSize, 256),
      iree_hal_make_pool_reservation_request(2 * kSlabSize, 256),
  };
  iree_hal_pool_reservation_t reservations[2] = {};
  iree_hal_pool_acquire_info_t infos[2] = {};
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      source, 1, requests, nullptr, reservations, infos, callback,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  ASSERT_NO_FATAL_FAILURE(AwaitVirtualPool());
  IREE_ASSERT_OK(completion.status);
  const auto original = reservations[0];
  iree_hal_pool_reservation_view_t view;
  ASSERT_TRUE(
      iree_hal_pool_query_reservation_views(source, 1, reservations, &view));
  ASSERT_NO_FATAL_FAILURE(RoundTrip(view.buffer, 0, 0x12345678));
  iree_hal_pool_release_reservations(source, 1, reservations,
                                     infos[0].reuse_frontier);

  // The warm first range must roll back when the complete four-slab request
  // cannot fit the three-slab budget. Caller output slots stay untouched.
  reservations[0].block_handle = 17;
  reservations[1].block_handle = 29;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      source, 2, requests, nullptr, reservations, infos, callback,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  ASSERT_NO_FATAL_FAILURE(AwaitVirtualPool());
  IREE_ASSERT_OK(completion.status);
  EXPECT_EQ(completion.result, IREE_HAL_POOL_ACQUIRE_OVER_BUDGET);
  EXPECT_EQ(reservations[0].block_handle, 17u);
  EXPECT_EQ(reservations[1].block_handle, 29u);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            2 * kSlabSize);

  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      source, 1, requests, nullptr, reservations, infos, callback,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  // No owner pass is needed for a warm hit after rollback.
  EXPECT_EQ(completion.count, 3u);
  IREE_ASSERT_OK(completion.status);
  EXPECT_EQ(reservations[0].block_handle, original.block_handle);
  ASSERT_TRUE(
      iree_hal_pool_query_reservation_views(source, 1, reservations, &view));
  ASSERT_NO_FATAL_FAILURE(CheckRetained(view.buffer, 0, 0x12345678));
  iree_hal_buffer_t* owned = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      source, 1, requests, reservations,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, &owned));
  iree_hal_buffer_release(owned);
  iree_hal_pool_stats_t statistics;
  iree_hal_pool_query_stats(source, &statistics);
  EXPECT_EQ(statistics.reservation_count, 0u);
}

TEST_F(VirtualPoolTest, VirtualBackingWaitsForIncompatibleNativeReturn) {
  ASSERT_NO_FATAL_FAILURE(
      UseDeferredBacking(3 * kSlabSize, nullptr, /*distinct_classes=*/true));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_state_geometry(device), 3 * kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 3 * kSlabSize));
  ASSERT_NO_FATAL_FAILURE(
      RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678));
  loom_serve_virtual_buffer_begin_trim(first);
  loom_serve_virtual_buffer_keep(first, 0, kSlabSize);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  ASSERT_NO_FATAL_FAILURE(CreateVirtualPool());
  auto* source = loom_serve_virtual_pool_handle(virtual_pool);
  struct Completion {
    // Owner-side acquisition callback count.
    size_t count = 0;
    // Owned terminal status, consumed after acquisition.
    iree_status_t status = iree_ok_status();
    // Capacity outcome published alongside status.
    iree_hal_pool_acquire_result_t result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  } completion;
  const iree_hal_pool_acquire_callback_t callback = {
      .fn =
          [](void* user_data, iree_status_t status,
             iree_hal_pool_acquire_result_t result) {
            auto* completion = static_cast<Completion*>(user_data);
            ++completion->count;
            completion->status = status;
            completion->result = result;
          },
      .user_data = &completion,
  };
  const auto request =
      iree_hal_make_pool_reservation_request(2 * kSlabSize, 256);
  iree_hal_pool_reservation_t reservation = {};
  iree_hal_pool_acquire_info_t info = {};
  deferred_backing->DeferReturns();
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      source, 1, &request, nullptr, &reservation, &info, callback,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  IREE_ASSERT_OK(loom_serve_virtual_pool_advance(virtual_pool));
  ASSERT_EQ(completion.count, 0u);
  deferred_backing->WaitForReturn();
  EXPECT_TRUE(loom_serve_virtual_pool_is_pending(virtual_pool));
  EXPECT_GT(deferred_backing->PendingReturnCount(), 0u);
  ASSERT_NO_FATAL_FAILURE(
      CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0x12345678));
  deferred_backing->CompleteReturns();
  ASSERT_NO_FATAL_FAILURE(AwaitVirtualPool());
  ASSERT_EQ(completion.count, 1u);
  IREE_ASSERT_OK(completion.status);
  ASSERT_EQ(completion.result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      source, 1, &request, &reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, &buffer));
  ASSERT_NO_FATAL_FAILURE(RoundTrip(buffer, kSlabSize, 0x87654321));
  iree_hal_buffer_release(buffer);
}

TEST_F(VirtualPoolTest,
       VirtualBackingMappingFailureCompletesWithoutPublishingViews) {
  mapping_queue.Initialize(loom_serve_device_transfer_queue(device));
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking(3 * kSlabSize, &mapping_queue));
  ASSERT_NO_FATAL_FAILURE(CreateVirtualPool());
  mapping_queue.fail_next_map = true;
  struct Completion {
    // Accepted transaction has delivered its one terminal callback.
    bool complete = false;
    // Owned failure delivered independently of physical cleanup.
    iree_status_t status = iree_ok_status();
  } completion;
  const iree_hal_pool_acquire_callback_t callback = {
      .fn =
          [](void* user_data, iree_status_t status,
             iree_hal_pool_acquire_result_t result) {
            auto* completion = static_cast<Completion*>(user_data);
            completion->complete = true;
            completion->status = status;
          },
      .user_data = &completion,
  };
  const auto request = iree_hal_make_pool_reservation_request(kSlabSize, 256);
  iree_hal_pool_reservation_t reservation = {.block_handle = 17};
  iree_hal_pool_acquire_info_t info = {};
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      loom_serve_virtual_pool_handle(virtual_pool), 1, &request, nullptr,
      &reservation, &info, callback, IREE_HAL_POOL_RESERVE_FLAG_NONE));
  ASSERT_NO_FATAL_FAILURE(AwaitVirtualPool());
  ASSERT_TRUE(completion.complete);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, completion.status);
  EXPECT_EQ(reservation.block_handle, 17u);
  // The failed range still owns its physical storage. Checked outer cleanup,
  // not callback failure, is what permits reuse.
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, kSlabSize);
}

TEST_F(VirtualPoolTest, VirtualBackingTrimWaitsForActualHistoryCompletion) {
  ASSERT_NO_FATAL_FAILURE(CreateVirtualPool());
  auto* source = loom_serve_virtual_pool_handle(virtual_pool);
  struct Completion {
    // Accepted acquisitions deliver their terminal result on the owner.
    size_t count = 0;
    // Owned status consumed immediately after each completed acquisition.
    iree_status_t status = iree_ok_status();
  } completion;
  const iree_hal_pool_acquire_callback_t callback = {
      .fn =
          [](void* user_data, iree_status_t status,
             iree_hal_pool_acquire_result_t result) {
            auto* completion = static_cast<Completion*>(user_data);
            ++completion->count;
            completion->status = status;
          },
      .user_data = &completion,
  };
  const auto request = iree_hal_make_pool_reservation_request(kSlabSize, 256);
  iree_hal_pool_reservation_t reservation = {};
  iree_hal_pool_acquire_info_t info = {};
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      source, 1, &request, nullptr, &reservation, &info, callback,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  ASSERT_NO_FATAL_FAILURE(AwaitVirtualPool());
  IREE_ASSERT_OK(completion.status);
  iree_hal_pool_reservation_view_t view;
  ASSERT_TRUE(
      iree_hal_pool_query_reservation_views(source, 1, &reservation, &view));
  ASSERT_NO_FATAL_FAILURE(RoundTrip(view.buffer, 0, 0x23571113));

  // An unpublished immutable join supplies a real pending completion point
  // without registering a colliding native queue axis or creating a thread.
  iree_async_frontier_timeline_t* timeline = nullptr;
  IREE_ASSERT_OK(iree_async_frontier_tracker_acquire_timeline(
      source->frontier_tracker, &timeline));
  std::unique_ptr<iree_async_frontier_timeline_t,
                  decltype(&iree_async_frontier_timeline_release)>
      timeline_owner(timeline, iree_async_frontier_timeline_release);
  iree_async_frontier_join_t* join = nullptr;
  IREE_ASSERT_OK(iree_async_frontier_join_allocate(timeline, &join));
  const auto point = iree_async_frontier_join_begin(join);
  std::unique_ptr<iree_async_frontier_join_t,
                  decltype(&iree_async_frontier_join_publish)>
      completion_gate(join, iree_async_frontier_join_publish);
  IREE_ASYNC_FIXED_FRONTIER_TYPE(Frontier, 1);
  const Frontier history = {.entry_count = 1, .entries = {point}};
  const auto* frontier = iree_async_fixed_frontier_as_const_frontier(&history);
  iree_hal_pool_release_reservations(source, 1, &reservation, frontier);
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      source, 1, &request, frontier, &reservation, &info, callback,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  EXPECT_EQ(completion.count, 2u);
  IREE_ASSERT_OK(completion.status);
  EXPECT_EQ(info.result, IREE_HAL_POOL_ACQUIRE_OK);
  // Dominance licenses ordered logical reuse, not physical detach.
  iree_hal_pool_release_reservations(source, 1, &reservation,
                                     info.reuse_frontier);
  iree_hal_pool_trim(source, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  IREE_ASSERT_OK(loom_serve_virtual_pool_advance(virtual_pool));
  EXPECT_TRUE(loom_serve_virtual_pool_is_pending(virtual_pool));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, kSlabSize);
  ASSERT_NO_FATAL_FAILURE(CheckRetained(view.buffer, 0, 0x23571113));

  completion_gate.reset();
  timeline_owner.reset();
  ASSERT_NO_FATAL_FAILURE(AwaitVirtualPool());
  iree_hal_pool_stats_t detached;
  iree_hal_pool_query_stats(source, &detached);
  EXPECT_EQ(detached.bytes_committed, 0u);
  // The common cache still owns the physical slab, available to any role.
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).cached_bytes, kSlabSize);
}

}  // namespace
