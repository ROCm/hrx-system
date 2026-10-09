// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <thread>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/maintenance_thread.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TLSFPoolReservationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), allocator_, &proactor_));
    IREE_ASSERT_OK(iree_async_notification_create(
        proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification_));
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), allocator_, &tracker_));
    IREE_ASSERT_OK(
        iree_async_frontier_tracker_register_axis(tracker_, Axis(), nullptr));
    IREE_ASSERT_OK(iree_hal_memory_maintenance_thread_create({}, allocator_,
                                                             &maintenance_));
    IREE_ASSERT_OK(
        iree_hal_cpu_slab_provider_create(256, allocator_, &provider_));
    IREE_ASSERT_OK(
        iree_hal_passthrough_pool_create({}, provider_, notification_, tracker_,
                                         maintenance_, allocator_, &native_));
    iree_hal_slab_cache_options_t options = {.slab = Request(4096, 16),
                                             .max_count = 4};
    IREE_ASSERT_OK(
        iree_hal_slab_cache_create(native_, &options, allocator_, &cache_));
  }

  void TearDown() override {
    for (auto i = pools_.rbegin(); i != pools_.rend(); ++i) {
      iree_hal_pool_release(*i);
    }
    iree_hal_pool_release(cache_);
    iree_hal_pool_release(native_);
    iree_hal_slab_provider_release(provider_);
    iree_hal_memory_maintenance_release(maintenance_);
    iree_async_frontier_tracker_release(tracker_);
    iree_async_notification_release(notification_);
    iree_async_proactor_release(proactor_);
  }

  static iree_async_axis_t Axis() {
    return iree_async_axis_make_queue(1, 0, 0, 0, 0);
  }

  static iree_hal_pool_reservation_request_t Request(
      iree_device_size_t size, iree_device_size_t alignment = 16) {
    iree_hal_pool_reservation_request_t request = {};
    request.params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
    request.params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    request.params.usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
    request.params.min_alignment = alignment;
    request.allocation_size = size;
    return request;
  }

  iree_status_t Create(iree_hal_pool_t* parent, iree_device_size_t budget,
                       iree_allocator_t allocator, iree_hal_pool_t** out_pool) {
    iree_hal_tlsf_pool_options_t options = {};
    options.tlsf_options.range_length = 4096;
    options.tlsf_options.alignment = 16;
    options.tlsf_options.frontier_capacity = 2;
    options.budget_limit = budget;
    iree_status_t status =
        iree_hal_tlsf_pool_create(parent, &options, allocator, out_pool);
    if (iree_status_is_ok(status)) {
      pools_.push_back(*out_pool);
    }
    return status;
  }

  void Drain() {
    iree_hal_memory_maintenance_call(
        maintenance_,
        [](void* user_data) {
          auto* owner = static_cast<iree_hal_memory_maintenance_t*>(user_data);
          while (iree_hal_memory_maintenance_run_one(owner)) {
          }
        },
        maintenance_);
  }

  void CheckBytes(iree_hal_buffer_t* buffer, uint8_t pattern,
                  iree_device_size_t alignment) {
    iree_hal_buffer_mapping_t mapping;
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_ALL,
        IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, IREE_HAL_WHOLE_BUFFER, &mapping));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(mapping.contents.data) % alignment,
              0u);
    memset(mapping.contents.data, pattern, mapping.contents.data_length);
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
    std::vector<uint8_t> actual(iree_hal_buffer_byte_length(buffer));
    IREE_ASSERT_OK(
        iree_hal_buffer_map_read(buffer, 0, actual.data(), actual.size()));
    for (uint8_t value : actual) {
      EXPECT_EQ(value, pattern);
    }
  }

  // Source allocation and progress owners outlive every child.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Native notification execution context.
  iree_async_proactor_t* proactor_ = nullptr;
  // Shared capacity event.
  iree_async_notification_t* notification_ = nullptr;
  // Exact completion state for the fixture's production queue axis.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
  // Captured cold memory executor.
  iree_hal_memory_maintenance_t* maintenance_ = nullptr;
  // Aligned real host storage provider.
  iree_hal_slab_provider_t* provider_ = nullptr;
  // Ordinary native leaf retained until all prepared views return.
  iree_hal_pool_t* native_ = nullptr;
  // Explicit idle retention for regular backing, with dedicated bypass.
  iree_hal_pool_t* cache_ = nullptr;
  // Children in construction order, destroyed in reverse order.
  std::vector<iree_hal_pool_t*> pools_;
};

TEST_F(TLSFPoolReservationTest, MixedSmallLargeAndAlignedBatchUsesOneSource) {
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(Create(cache_, 0, allocator_, &pool));
  const std::array requests = {Request(128), Request(8193), Request(257, 256),
                               Request(512), Request(16384, 128)};
  std::array<iree_hal_pool_reservation_t, requests.size()> reservations;
  std::array<iree_hal_pool_acquire_info_t, requests.size()> infos;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool, requests.size(), requests.data(), nullptr,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, reservations.data(), infos.data(),
      &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  std::array<iree_hal_buffer_t*, requests.size()> buffers;
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      pool, requests.size(), requests.data(), reservations.data(),
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP,
      buffers.data()));
  for (size_t i = 0; i < requests.size(); ++i) {
    CheckBytes(buffers[i], static_cast<uint8_t>(0x31 + i),
               requests[i].params.min_alignment);
  }
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reservation_count, requests.size());
  EXPECT_EQ(stats.slab_count, 4u);
  iree_hal_slab_cache_stats_t cache_stats;
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache_, &cache_stats));
  EXPECT_EQ(cache_stats.miss_count, 1u);
  EXPECT_EQ(cache_stats.bypass_count, 3u);
  for (auto* buffer : buffers) {
    iree_hal_buffer_release(buffer);
  }
  Drain();
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.bytes_reserved, 0u);
  EXPECT_EQ(stats.bytes_committed, 0u);
}

TEST_F(TLSFPoolReservationTest, DedicatedMetadataDoesNotAllocateOffsetTables) {
  struct AllocationCounts {
    // Allocations admitted after pool construction.
    size_t count = 0;
    // Largest metadata allocation in bytes.
    size_t maximum_size = 0;
  } counts;
  const iree_allocator_t metadata_allocator = {
      &counts,
      +[](void* self, iree_allocator_command_t command, const void* params,
          void** pointer) -> iree_status_t {
        if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
            command == IREE_ALLOCATOR_COMMAND_CALLOC) {
          auto* counts = static_cast<AllocationCounts*>(self);
          ++counts->count;
          const auto size =
              static_cast<const iree_allocator_alloc_params_t*>(params)
                  ->byte_length;
          counts->maximum_size = iree_max(counts->maximum_size, size);
        }
        const auto allocator = iree_allocator_system();
        return allocator.ctl(allocator.self, command, params, pointer);
      }};
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(Create(cache_, 0, metadata_allocator, &pool));
  counts = {};
  const auto request = Request(8192);
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool, 1, &request, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation,
      &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(counts.count, 1u);
  EXPECT_LT(counts.maximum_size, sizeof(iree_hal_memory_tlsf_t));
  iree_hal_pool_release_reservations(pool, 1, &reservation, nullptr);
  // The allocator context is local, so finish its pool before returning.
  pools_.pop_back();
  iree_hal_pool_release(pool);
}

TEST_F(TLSFPoolReservationTest,
       NoGrowthAndBudgetRefuseBeforeNativeAcquisition) {
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(Create(cache_, 8192, allocator_, &pool));
  const auto invalid_requests = {Request(0), Request(128, 3),
                                 Request(IREE_DEVICE_SIZE_MAX)};
  for (const auto request : invalid_requests) {
    iree_hal_pool_reservation_t reservation = {};
    iree_hal_pool_acquire_info_t info;
    iree_hal_pool_acquire_result_t result;
    IREE_EXPECT_STATUS_IS(
        request.allocation_size == IREE_DEVICE_SIZE_MAX
            ? IREE_STATUS_OUT_OF_RANGE
            : IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_pool_acquire_reservations(pool, 1, &request, nullptr,
                                           IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                           &reservation, &info, &result));
    EXPECT_EQ(reservation.block_handle, 0u);
  }
  for (const auto request :
       {Request(8192), Request(128, 256), Request(128, 512), Request(8193)}) {
    iree_hal_pool_reservation_t reservation = {};
    iree_hal_pool_acquire_info_t info;
    iree_hal_pool_acquire_result_t result;
    IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
        pool, 1, &request, nullptr, IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH,
        &reservation, &info, &result));
    EXPECT_EQ(result, request.allocation_size > 8192
                          ? IREE_HAL_POOL_ACQUIRE_OVER_BUDGET
                          : IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
    EXPECT_EQ(reservation.block_handle, 0u);
  }
  const auto request = Request(8193);
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool, 1, &request, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation,
      &info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OVER_BUDGET);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(native_, &stats);
  EXPECT_EQ(stats.reserve_count, 0u);
}

TEST_F(TLSFPoolReservationTest, MixedRollbackPreservesDedicatedParentHistory) {
  const auto parent_request = Request(8192);
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      native_, parent_request.params, 8192, iree_infinite_timeout(), &backing));
  iree_hal_fixed_block_pool_options_t parent_options = {.block_size = 8192,
                                                        .frontier_capacity = 2};
  iree_hal_pool_t* parent = nullptr;
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
      backing, 0, IREE_HAL_WHOLE_BUFFER, &parent_options, allocator_, &parent));
  pools_.push_back(parent);
  iree_hal_buffer_release(backing);
  iree_hal_pool_reservation_t original;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      parent, 1, &parent_request, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      &original, &info, &result));
  iree_async_single_frontier_t prior_use;
  iree_async_single_frontier_initialize(&prior_use, Axis(), 9);
  const auto* frontier = iree_async_single_frontier_as_frontier(&prior_use);
  iree_hal_pool_release_reservations(parent, 1, &original, frontier);

  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(Create(parent, 6144, allocator_, &pool));
  const std::array requests = {Request(6144), Request(128)};
  std::array<iree_hal_pool_reservation_t, 2> reservations = {};
  std::array<iree_hal_pool_acquire_info_t, 2> infos;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool, requests.size(), requests.data(), nullptr,
      IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER, reservations.data(),
      infos.data(), &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OVER_BUDGET);
  EXPECT_EQ(reservations[0].block_handle, 0u);
  EXPECT_EQ(reservations[1].block_handle, 0u);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.bytes_reserved, 0u);
  EXPECT_EQ(stats.bytes_committed, 0u);
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      parent, 1, &parent_request, nullptr,
      IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER, &original, &info,
      &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  ASSERT_NE(info.reuse_frontier, nullptr);
  ASSERT_EQ(info.reuse_frontier->entry_count, 1u);
  EXPECT_EQ(info.reuse_frontier->entries[0].axis, Axis());
  EXPECT_EQ(info.reuse_frontier->entries[0].epoch, 9u);
  iree_hal_pool_release_reservations(parent, 1, &original, info.reuse_frontier);
}

TEST_F(TLSFPoolReservationTest,
       DedicatedReturnPreservesUntouchedParentHistory) {
  // A cache returns its complete class extent. Alignment leaves the final byte
  // outside the child's managed range but inside its parent reservation.
  const auto parent_request = Request(8193);
  iree_hal_slab_cache_options_t parent_options = {.slab = parent_request,
                                                  .max_count = 1};
  iree_hal_pool_t* parent = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_cache_create(native_, &parent_options,
                                            allocator_, &parent));
  pools_.push_back(parent);
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      parent, 1, &parent_request, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      &reservation, &info, &result));
  iree_async_single_frontier_t prior_use;
  iree_async_single_frontier_initialize(&prior_use, Axis(), 9);
  iree_hal_pool_release_reservations(
      parent, 1, &reservation,
      iree_async_single_frontier_as_frontier(&prior_use));

  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(Create(parent, 0, allocator_, &pool));
  const auto request = Request(6144);
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool, 1, &request, iree_async_single_frontier_as_frontier(&prior_use),
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
  const auto next_axis = iree_async_axis_make_queue(1, 0, 0, 1, 0);
  IREE_ASSERT_OK(
      iree_async_frontier_tracker_register_axis(tracker_, next_axis, nullptr));
  iree_async_single_frontier_t next_use;
  iree_async_single_frontier_initialize(&next_use, next_axis, 7);
  iree_hal_pool_release_reservations(
      pool, 1, &reservation, iree_async_single_frontier_as_frontier(&next_use));
  Drain();
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      parent, 1, &parent_request, nullptr,
      IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER, &reservation, &info,
      &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  ASSERT_NE(info.reuse_frontier, nullptr);
  ASSERT_EQ(info.reuse_frontier->entry_count, 2u);
  EXPECT_EQ(info.reuse_frontier->entries[0].axis, Axis());
  EXPECT_EQ(info.reuse_frontier->entries[0].epoch, 9u);
  EXPECT_EQ(info.reuse_frontier->entries[1].axis, next_axis);
  EXPECT_EQ(info.reuse_frontier->entries[1].epoch, 7u);
  iree_async_frontier_tracker_advance(tracker_, Axis(), 9);
  iree_async_frontier_tracker_advance(tracker_, next_axis, 7);
  iree_hal_pool_release_reservations(parent, 1, &reservation, nullptr);
}

TEST_F(TLSFPoolReservationTest, ConcurrentDedicatedReturnAndRegularGrowth) {
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(Create(cache_, 0, allocator_, &pool));
  std::array<std::thread, 4> threads;
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&, i] {
      for (size_t j = 0; j < 20; ++j) {
        const auto request = Request(i % 2 ? 8192 : 512, i % 2 ? 128 : 16);
        iree_hal_buffer_t* buffer = nullptr;
        IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
            pool, request.params, request.allocation_size,
            iree_infinite_timeout(), &buffer));
        CheckBytes(buffer, static_cast<uint8_t>(i + j),
                   request.params.min_alignment);
        iree_hal_buffer_release(buffer);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  Drain();
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.slab_count, 0u);
}

}  // namespace
