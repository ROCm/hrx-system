// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <cstring>
#include <thread>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/util/proactor_thread.h"
#include "iree/base/status_cc.h"
#include "iree/hal/api.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_async_frontier_tracker_t* test_frontier_tracker() {
  static iree_async_frontier_tracker_t* tracker = nullptr;
  if (!tracker) {
    IREE_CHECK_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker));
    atexit([] {
      iree_async_frontier_tracker_release(tracker);
      tracker = nullptr;
    });
  }
  return tracker;
}

static iree_async_proactor_t* test_proactor() {
  static iree_async_proactor_t* proactor = nullptr;
  static iree_async_proactor_thread_t* thread = nullptr;
  if (!proactor) {
    auto options = iree_async_proactor_options_default();
    options.threading_mode = IREE_ASYNC_PROACTOR_THREADING_CROSS_THREAD;
    IREE_CHECK_OK(iree_async_proactor_create_platform(
        options, iree_allocator_system(), &proactor));
    IREE_CHECK_OK(iree_async_proactor_thread_create(
        proactor, iree_async_proactor_thread_options_default(),
        iree_allocator_system(), &thread));
    atexit([] {
      iree_async_proactor_thread_request_stop(thread);
      IREE_CHECK_OK(
          iree_async_proactor_thread_join(thread, IREE_DURATION_INFINITE));
      IREE_CHECK_OK(iree_async_proactor_thread_consume_status(thread));
      iree_async_proactor_thread_release(thread);
      thread = nullptr;
      iree_async_proactor_release(proactor);
      proactor = nullptr;
    });
  }
  return proactor;
}

// Uses a finite real pool with no optional completion probe. Returned capacity
// can remain pending after its one capacity notification has been consumed.
class PoolFrontierWaitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker_));
    IREE_ASSERT_OK(
        iree_async_frontier_tracker_register_axis(tracker_, axis_, nullptr));
    IREE_ASSERT_OK(iree_async_notification_create(
        test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification_));
    IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(
        /*min_alignment=*/0, iree_allocator_system(), &provider_));
    iree_hal_fixed_block_pool_options_t options = {
        .block_size = kByteLength,
        .blocks_per_slab = 1,
        .frontier_capacity = 2,
    };
    IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
        {}, provider_, notification_, tracker_, iree_allocator_system(),
        &backing_pool_));
    iree_hal_pool_reservation_request_t backing_request;
    IREE_ASSERT_OK(iree_hal_fixed_block_pool_query_backing_request(
        backing_pool_, &options, &backing_request));
    iree_hal_buffer_t* backing_buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(backing_pool_, backing_request,
                                                 iree_infinite_timeout(),
                                                 &backing_buffer));
    options.blocks_per_slab = 0;
    iree_status_t status = iree_hal_fixed_block_pool_create_from_buffer(
        backing_buffer, 0, IREE_HAL_WHOLE_BUFFER, &options,
        iree_allocator_system(), &pool_);
    iree_hal_buffer_release(backing_buffer);
    IREE_ASSERT_OK(status);
    // Exercise the generic waiter's completion handling without the pool's
    // optional early completion probe resolving the prerequisite first.
    pool_->epoch_query = iree_hal_pool_epoch_query_null();
    request_ = iree_hal_make_pool_reservation_request(kByteLength, 0);
  }

  void TearDown() override {
    iree_hal_buffer_release(buffer_);
    iree_hal_pool_release(pool_);
    iree_hal_pool_release(backing_pool_);
    iree_hal_slab_provider_release(provider_);
    iree_async_notification_release(notification_);
    iree_async_frontier_tracker_release(tracker_);
  }

  void AcquireFresh(iree_hal_pool_reservation_t* reservation) {
    iree_hal_pool_acquire_info_t info = {};
    iree_hal_pool_acquire_result_t result;
    IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
        pool_, 1, &request_, nullptr, reservation, &info, &result,
        IREE_HAL_POOL_RESERVE_FLAG_NONE));
    ASSERT_TRUE(result == IREE_HAL_POOL_ACQUIRE_OK ||
                result == IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  }

  void ReleasePending(const iree_hal_pool_reservation_t& reservation,
                      iree_async_axis_t axis, uint64_t epoch) {
    iree_async_single_frontier_t frontier;
    iree_async_single_frontier_initialize(&frontier, axis, epoch);
    iree_hal_pool_release_reservations(
        pool_, 1, &reservation,
        iree_async_single_frontier_as_const_frontier(&frontier));
  }

  void SeedPending(iree_async_axis_t axis, uint64_t epoch) {
    iree_hal_pool_reservation_t reservation;
    ASSERT_NO_FATAL_FAILURE(AcquireFresh(&reservation));
    ReleasePending(reservation, axis, epoch);
  }

  void ExpectPending(iree_async_axis_t axis, uint64_t epoch) {
    iree_hal_pool_reservation_t reservation;
    iree_hal_pool_acquire_info_t info = {};
    iree_hal_pool_acquire_result_t result;
    IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
        pool_, 1, &request_, nullptr, &reservation, &info, &result,
        IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER));
    ASSERT_TRUE(result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT ||
                result == IREE_HAL_POOL_ACQUIRE_OK ||
                result == IREE_HAL_POOL_ACQUIRE_OK_FRESH);
    EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
    if (result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT) {
      EXPECT_EQ(info.reuse_frontier->entry_count, 1);
      EXPECT_EQ(info.reuse_frontier->entries[0].axis, axis);
      EXPECT_EQ(info.reuse_frontier->entries[0].epoch, epoch);
    }
    iree_hal_pool_release_reservations(pool_, 1, &reservation,
                                       info.reuse_frontier);
  }

  iree_status_t Allocate(iree_timeout_t timeout,
                         iree_hal_buffer_t** out_buffer) {
    return iree_hal_pool_allocate_buffer(
        pool_,
        iree_hal_make_pool_reservation_request(kByteLength,
                                               request_.min_alignment),
        timeout, out_buffer);
  }

  iree_hal_pool_stats_t Stats() {
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(pool_, &stats);
    return stats;
  }

  void ExpectUsableBuffer() {
    ASSERT_NE(buffer_, nullptr);
    iree_hal_buffer_mapping_t mapping = {};
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        buffer_, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE,
        IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, kByteLength, &mapping));
    memset(mapping.contents.data, 0xA5, kByteLength);
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        buffer_, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
        IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, kByteLength, &mapping));
    for (iree_host_size_t i = 0; i < kByteLength; ++i) {
      EXPECT_EQ(mapping.contents.data[i], 0xA5);
    }
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  }

  // Full usable extent of the pool's single block, in bytes.
  static constexpr iree_device_size_t kByteLength = 64;
  // Registered completion domain for this pool; no state crosses test cases.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
  // Queue axis whose completion controls reuse of the sole block.
  iree_async_axis_t axis_ = iree_async_axis_make_queue(1, 0, 0, 0, 0);
  // Capacity notification retained by the fixture and pool.
  iree_async_notification_t* notification_ = nullptr;
  // Actual host-memory backing provider.
  iree_hal_slab_provider_t* provider_ = nullptr;
  // Native owner of the finite arena.
  iree_hal_pool_t* backing_pool_ = nullptr;
  // One-block pool with a borrowed reference to tracker_.
  iree_hal_pool_t* pool_ = nullptr;
  // Common allocation request for the full block.
  iree_hal_pool_reservation_request_t request_ = {};
  // Owned materialized buffer released before its pool.
  iree_hal_buffer_t* buffer_ = nullptr;
};

TEST_F(PoolFrontierWaitTest, ReadyCompletionCanReturnReservationInline) {
  struct Completion {
    // Pool borrowed while the submitter and callback access the reservation.
    iree_hal_pool_t* pool;
    // Output published before the callback runs.
    iree_hal_pool_reservation_t reservation;
    // Submission thread, which must execute this ready completion.
    std::thread::id submitter;
    // Terminal callbacks observed before submission returns.
    int count;
  } completion = {.pool = pool_, .submitter = std::this_thread::get_id()};
  iree_hal_pool_acquire_info_t info;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool_, 1, &request_, nullptr, &completion.reservation, &info,
      {
          .fn =
              [](void* user_data, iree_status_t status,
                 iree_hal_pool_acquire_result_t result) {
                auto* completion = static_cast<Completion*>(user_data);
                ++completion->count;
                IREE_EXPECT_OK(status);
                EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
                EXPECT_EQ(std::this_thread::get_id(), completion->submitter);
                // Reentry proves acquisition drops every metadata lock before
                // handing ownership to the caller's completion.
                iree_hal_pool_release_reservations(
                    completion->pool, 1, &completion->reservation, nullptr);
              },
          .user_data = &completion,
      },
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  EXPECT_EQ(completion.count, 1);
  EXPECT_EQ(Stats().reservation_count, 0u);
  IREE_ASSERT_OK(Allocate(iree_immediate_timeout(), &buffer_));
  ExpectUsableBuffer();
}

TEST_F(PoolFrontierWaitTest, CapacityCompletionAndRejectedCaptureAreDistinct) {
  iree_hal_pool_reservation_t occupied;
  ASSERT_NO_FATAL_FAILURE(AcquireFresh(&occupied));
  iree_hal_pool_reservation_t reservation;
  memset(&reservation, 0xA5, sizeof(reservation));
  const auto original = reservation;
  iree_hal_pool_acquire_info_t info;
  int completion_count = 0;
  const iree_hal_pool_acquire_callback_t completion = {
      .fn =
          [](void* user_data, iree_status_t status,
             iree_hal_pool_acquire_result_t result) {
            ++*static_cast<int*>(user_data);
            IREE_EXPECT_OK(status);
            EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
          },
      .user_data = &completion_count,
  };
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool_, 1, &request_, nullptr, &reservation, &info, completion,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  EXPECT_EQ(completion_count, 1);
  EXPECT_EQ(info.result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
  EXPECT_EQ(memcmp(&reservation, &original, sizeof(reservation)), 0);

  const auto invalid_request = iree_hal_make_pool_reservation_request(0, 0);
  const auto original_info = info;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_pool_acquire_reservations(pool_, 1, &invalid_request, nullptr,
                                         &reservation, &info, completion,
                                         IREE_HAL_POOL_RESERVE_FLAG_NONE));
  EXPECT_EQ(completion_count, 1);
  EXPECT_EQ(memcmp(&reservation, &original, sizeof(reservation)), 0);
  EXPECT_EQ(memcmp(&info, &original_info, sizeof(info)), 0);
  iree_hal_pool_release_reservations(pool_, 1, &occupied, nullptr);
}

TEST_F(PoolFrontierWaitTest, AlreadyCompletedFrontierMaterializesInline) {
  ASSERT_NO_FATAL_FAILURE(SeedPending(axis_, 1));
  iree_async_frontier_tracker_advance(tracker_, axis_, 1);
  IREE_ASSERT_OK(Allocate(iree_infinite_timeout(), &buffer_));
  EXPECT_EQ(Stats().wait_count, 1u);
  ExpectUsableBuffer();
}

TEST_F(PoolFrontierWaitTest, ImmediateAllocationAcceptsCompletedFrontier) {
  ASSERT_NO_FATAL_FAILURE(SeedPending(axis_, 1));
  iree_async_frontier_tracker_advance(tracker_, axis_, 1);
  IREE_ASSERT_OK(Allocate(iree_immediate_timeout(), &buffer_));
  EXPECT_EQ(Stats().wait_count, 1u);
  ExpectUsableBuffer();
}

TEST_F(PoolFrontierWaitTest, ImmediateAllocationPreservesPendingRange) {
  ASSERT_NO_FATAL_FAILURE(SeedPending(axis_, 1));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED,
                        Allocate(iree_immediate_timeout(), &buffer_));
  EXPECT_EQ(buffer_, nullptr);
  EXPECT_EQ(Stats().wait_count, 1u);
  ExpectPending(axis_, 1);
}

TEST_F(PoolFrontierWaitTest, TimeoutPreservesFrontierAndOutput) {
  ASSERT_NO_FATAL_FAILURE(SeedPending(axis_, 1));
  auto* sentinel = reinterpret_cast<iree_hal_buffer_t*>(uintptr_t{1});
  auto* output = sentinel;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED,
                        Allocate(iree_make_timeout_ms(1), &output));
  EXPECT_EQ(output, sentinel);
  ASSERT_NO_FATAL_FAILURE(ExpectPending(axis_, 1));
  // Completion, without another capacity signal, makes the same range usable.
  iree_async_frontier_tracker_advance(tracker_, axis_, 1);
  IREE_ASSERT_OK(Allocate(iree_infinite_timeout(), &buffer_));
  ExpectUsableBuffer();
}

TEST_F(PoolFrontierWaitTest, FailedAxisPreservesFrontier) {
  ASSERT_NO_FATAL_FAILURE(SeedPending(axis_, 1));
  iree_async_frontier_tracker_fail_axis(
      tracker_, axis_, iree_status_from_code(IREE_STATUS_DATA_LOSS));
  for (auto timeout : {iree_immediate_timeout(), iree_infinite_timeout()}) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Allocate(timeout, &buffer_));
    EXPECT_EQ(buffer_, nullptr);
    ASSERT_NO_FATAL_FAILURE(ExpectPending(axis_, 1));
  }
}

TEST_F(PoolFrontierWaitTest, CapacityWakeThenExactCompletion) {
  iree_hal_pool_reservation_t occupied;
  ASSERT_NO_FATAL_FAILURE(AcquireFresh(&occupied));
  std::atomic<bool> finished{false};
  iree::Status allocation_status;
  std::thread allocating([&] {
    allocation_status = Allocate(iree_infinite_timeout(), &buffer_);
    finished.store(true, std::memory_order_release);
  });
  while (Stats().exhausted_count == 0 &&
         !finished.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  ReleasePending(occupied, axis_, 1);
  while (Stats().wait_count == 0 && !finished.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  EXPECT_FALSE(finished.load(std::memory_order_acquire));
  iree_async_frontier_tracker_advance(tracker_, axis_, 1);
  allocating.join();
  IREE_ASSERT_OK(allocation_status);
  ExpectUsableBuffer();
}

TEST_F(PoolFrontierWaitTest, CapacityTimeoutPreservesAllocationAndOutput) {
  iree_hal_pool_reservation_t occupied;
  ASSERT_NO_FATAL_FAILURE(AcquireFresh(&occupied));
  auto* sentinel = reinterpret_cast<iree_hal_buffer_t*>(uintptr_t{1});
  auto* output = sentinel;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED,
                        Allocate(iree_make_timeout_ms(1), &output));
  EXPECT_EQ(output, sentinel);
  iree_hal_pool_release_reservations(pool_, 1, &occupied, nullptr);
  IREE_ASSERT_OK(Allocate(iree_infinite_timeout(), &buffer_));
  ExpectUsableBuffer();
}

TEST_F(PoolFrontierWaitTest, FailureDuringAllocationPropagatesAndRollsBack) {
  ASSERT_NO_FATAL_FAILURE(SeedPending(axis_, 1));
  std::atomic<bool> finished{false};
  iree::Status allocation_status;
  std::thread allocating([&] {
    allocation_status = Allocate(iree_infinite_timeout(), &buffer_);
    finished.store(true, std::memory_order_release);
  });
  while (Stats().wait_count == 0 && !finished.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  iree_async_frontier_tracker_fail_axis(
      tracker_, axis_, iree_status_from_code(IREE_STATUS_DATA_LOSS));
  allocating.join();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, allocation_status);
  EXPECT_EQ(buffer_, nullptr);
  ExpectPending(axis_, 1);
}

TEST_F(PoolFrontierWaitTest, TimeoutRacingCompletionReleasesWaitStorage) {
  // Either cancellation or dispatch may win. Advancing after every call also
  // detects a waiter left pointing at the allocating thread's expired stack.
  for (uint64_t epoch = 1; epoch <= 128; ++epoch) {
    ASSERT_NO_FATAL_FAILURE(SeedPending(axis_, epoch));
    const uint64_t previous_wait_count = Stats().wait_count;
    std::atomic<bool> finished{false};
    iree::Status allocation_status;
    std::thread allocating([&] {
      allocation_status = Allocate(iree_make_timeout_ns(1), &buffer_);
      finished.store(true, std::memory_order_release);
    });
    while (Stats().wait_count == previous_wait_count &&
           !finished.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    iree_async_frontier_tracker_advance(tracker_, axis_, epoch);
    allocating.join();
    if (allocation_status.code() == iree::StatusCode::kDeadlineExceeded) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED, allocation_status);
      EXPECT_EQ(buffer_, nullptr);
      IREE_ASSERT_OK(Allocate(iree_infinite_timeout(), &buffer_));
    } else {
      IREE_ASSERT_OK(allocation_status);
    }
    ASSERT_NO_FATAL_FAILURE(ExpectUsableBuffer());
    iree_hal_buffer_release(buffer_);
    buffer_ = nullptr;
  }
}

}  // namespace
