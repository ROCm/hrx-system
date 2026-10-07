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
#include "iree/hal/memory/maintenance_thread.h"
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

typedef struct iree_hal_routing_test_pool_t {
  // Base resource header for vtable dispatch and ref counting.
  iree_hal_pool_t base;

  // Capabilities returned when registering this pool in a pool set.
  iree_hal_pool_capabilities_t capabilities = {};
} iree_hal_routing_test_pool_t;

static void iree_hal_routing_test_pool_destroy(iree_hal_pool_t* base_pool) {
  iree_hal_pool_deinitialize(base_pool);
  delete (iree_hal_routing_test_pool_t*)base_pool;
}

static iree_status_t iree_hal_routing_test_pool_acquire_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reserve_flags_t flags,
    iree_hal_pool_reservation_t* out_reservations,
    iree_hal_pool_acquire_info_t* out_infos,
    iree_hal_pool_acquire_result_t* out_result) {
  (void)base_pool;
  (void)request_count;
  (void)requests;
  (void)requester_frontier;
  (void)flags;
  (void)out_reservations;
  (void)out_infos;
  (void)out_result;
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "acquire_reservations must not be called");
}

static void iree_hal_routing_test_pool_release_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    const iree_async_frontier_t* death_frontier) {
  (void)base_pool;
  (void)reservation_count;
  (void)reservations;
  (void)death_frontier;
}

static iree_status_t iree_hal_routing_test_pool_materialize_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffers) {
  (void)base_pool;
  (void)reservation_count;
  (void)requests;
  (void)reservations;
  (void)flags;
  (void)out_buffers;
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "materialize_reservations must not be called");
}

static void iree_hal_routing_test_pool_query_capabilities(
    const iree_hal_pool_t* base_pool,
    iree_hal_pool_capabilities_t* out_capabilities) {
  auto* pool = (iree_hal_routing_test_pool_t*)base_pool;
  *out_capabilities = pool->capabilities;
}

static void iree_hal_routing_test_pool_query_stats(
    const iree_hal_pool_t* base_pool, iree_hal_pool_stats_t* out_stats) {
  (void)base_pool;
  memset(out_stats, 0, sizeof(*out_stats));
}

static void iree_hal_routing_test_pool_trim(
    iree_hal_pool_t* base_pool, iree_hal_pool_trim_flags_t flags,
    iree_device_size_t min_bytes_to_keep) {
  (void)base_pool;
  (void)flags;
  (void)min_bytes_to_keep;
}

static const iree_hal_pool_vtable_t iree_hal_routing_test_pool_vtable = {
    /*.destroy=*/iree_hal_routing_test_pool_destroy,
    /*.acquire_reservations=*/iree_hal_routing_test_pool_acquire_reservations,
    /*.release_reservations=*/iree_hal_routing_test_pool_release_reservations,
    /*.query_reservation_views=*/nullptr,
    /*.materialize_reservations=*/
    iree_hal_routing_test_pool_materialize_reservations,
    /*.query_capabilities=*/iree_hal_routing_test_pool_query_capabilities,
    /*.validate_asan=*/nullptr,
    /*.query_stats=*/iree_hal_routing_test_pool_query_stats,
    /*.trim=*/iree_hal_routing_test_pool_trim,
    /*.advise_asan_reservations=*/nullptr,
};

static iree_hal_routing_test_pool_t* CreateRoutingTestPool(
    iree_device_size_t max_allocation_size) {
  auto* pool = new iree_hal_routing_test_pool_t;
  iree_async_notification_t* notification = nullptr;
  IREE_CHECK_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  IREE_CHECK_OK(iree_hal_pool_initialize(
      &iree_hal_routing_test_pool_vtable, nullptr, notification, {},
      test_frontier_tracker(), iree_allocator_system(), &pool->base));
  iree_async_notification_release(notification);
  pool->capabilities.memory_type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  pool->capabilities.allowed_access = IREE_HAL_MEMORY_ACCESS_ALL;
  pool->capabilities.supported_usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  pool->capabilities.queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY;
  pool->capabilities.min_allocation_size = 0;
  pool->capabilities.max_allocation_size = max_allocation_size;
  pool->capabilities.max_allocation_alignment = 64;
  pool->capabilities.maintenance_alignment = 1;
  return pool;
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
    iree_hal_fixed_block_pool_options_t options = {};
    options.block_size = kByteLength;
    options.blocks_per_slab = 1;
    options.frontier_capacity = 2;
    IREE_ASSERT_OK(iree_hal_memory_maintenance_thread_create(
        {}, iree_allocator_system(), &maintenance_));
    IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
        {}, provider_, notification_, tracker_, maintenance_,
        iree_allocator_system(), &backing_pool_));
    iree_hal_pool_reservation_request_t backing_request;
    IREE_ASSERT_OK(iree_hal_fixed_block_pool_query_backing_request(
        backing_pool_, &options, &backing_request));
    iree_hal_buffer_t* backing_buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        backing_pool_, backing_request.params, backing_request.allocation_size,
        iree_infinite_timeout(), &backing_buffer));
    options.blocks_per_slab = 0;
    iree_status_t status = iree_hal_fixed_block_pool_create_from_buffer(
        backing_buffer, 0, IREE_HAL_WHOLE_BUFFER, &options,
        iree_allocator_system(), &pool_);
    iree_hal_buffer_release(backing_buffer);
    IREE_ASSERT_OK(status);
    // Exercise the generic waiter's completion handling without the pool's
    // optional early completion probe resolving the prerequisite first.
    pool_->epoch_query = iree_hal_pool_epoch_query_null();
    request_.allocation_size = kByteLength;
    request_.params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
    request_.params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    request_.params.usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  }

  void TearDown() override {
    iree_hal_buffer_release(buffer_);
    iree_hal_pool_release(pool_);
    iree_hal_pool_release(backing_pool_);
    iree_hal_memory_maintenance_release(maintenance_);
    iree_hal_slab_provider_release(provider_);
    iree_async_notification_release(notification_);
    iree_async_frontier_tracker_release(tracker_);
  }

  void AcquireFresh(iree_hal_pool_reservation_t* reservation) {
    iree_hal_pool_acquire_info_t info = {};
    iree_hal_pool_acquire_result_t result;
    IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
        pool_, 1, &request_, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
        reservation, &info, &result));
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
    IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
        pool_, 1, &request_, nullptr,
        IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER, &reservation, &info,
        &result));
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
    return iree_hal_pool_allocate_buffer(pool_, request_.params, kByteLength,
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
  // Captured owner of native memory operations.
  iree_hal_memory_maintenance_t* maintenance_ = nullptr;
  // Native owner of the finite arena.
  iree_hal_pool_t* backing_pool_ = nullptr;
  // One-block pool with a borrowed reference to tracker_.
  iree_hal_pool_t* pool_ = nullptr;
  // Common allocation request for the full block.
  iree_hal_pool_reservation_request_t request_ = {};
  // Owned materialized buffer released before its pool.
  iree_hal_buffer_t* buffer_ = nullptr;
};

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

TEST(PoolSetTest, SelectsHighestPriorityCompatiblePoolBySize) {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_hal_pool_set_t pool_set;
  IREE_ASSERT_OK(iree_hal_pool_set_initialize(/*initial_capacity=*/2,
                                              host_allocator, &pool_set));

  iree_hal_routing_test_pool_t* direct_pool = CreateRoutingTestPool(0);
  iree_hal_routing_test_pool_t* tlsf_pool = CreateRoutingTestPool(1024);
  IREE_ASSERT_OK(
      iree_hal_pool_set_register(&pool_set, 0, (iree_hal_pool_t*)direct_pool));
  IREE_ASSERT_OK(
      iree_hal_pool_set_register(&pool_set, 10, (iree_hal_pool_t*)tlsf_pool));

  iree_hal_buffer_params_t params = {0};
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  EXPECT_EQ((iree_hal_pool_t*)tlsf_pool,
            iree_hal_pool_set_select(&pool_set, params, 512));
  EXPECT_EQ((iree_hal_pool_t*)direct_pool,
            iree_hal_pool_set_select(&pool_set, params, 2048));

  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_OPTIMAL;
  EXPECT_EQ((iree_hal_pool_t*)tlsf_pool,
            iree_hal_pool_set_select(&pool_set, params, 512));

  params.usage = IREE_HAL_BUFFER_USAGE_STORAGE;
  EXPECT_EQ(nullptr, iree_hal_pool_set_select(&pool_set, params, 512));

  iree_hal_pool_set_deinitialize(&pool_set);
  iree_hal_pool_release((iree_hal_pool_t*)tlsf_pool);
  iree_hal_pool_release((iree_hal_pool_t*)direct_pool);
}

TEST(PoolSetTest, SelectsPoolCoveringRequestedQueueFamilies) {
  iree_hal_pool_set_t pool_set;
  IREE_ASSERT_OK(iree_hal_pool_set_initialize(
      /*initial_capacity=*/2, iree_allocator_system(), &pool_set));

  iree_hal_routing_test_pool_t* any_pool = CreateRoutingTestPool(0);
  iree_hal_routing_test_pool_t* restricted_pool = CreateRoutingTestPool(0);
  const iree_hal_queue_family_affinity_t family_0 =
      iree_hal_make_queue_family_affinity(0);
  const iree_hal_queue_family_affinity_t family_1 =
      iree_hal_make_queue_family_affinity(1);
  const iree_hal_queue_family_affinity_t family_2 =
      iree_hal_make_queue_family_affinity(2);
  restricted_pool->capabilities.queue_family_affinity = family_0 | family_1;
  IREE_ASSERT_OK(iree_hal_pool_set_register(&pool_set, /*priority=*/0,
                                            (iree_hal_pool_t*)any_pool));
  IREE_ASSERT_OK(iree_hal_pool_set_register(&pool_set, /*priority=*/10,
                                            (iree_hal_pool_t*)restricted_pool));

  iree_hal_buffer_params_t params = {0};
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  params.queue_family_affinity = family_0;
  EXPECT_EQ((iree_hal_pool_t*)restricted_pool,
            iree_hal_pool_set_select(&pool_set, params, 4096));

  params.queue_family_affinity = family_2;
  EXPECT_EQ((iree_hal_pool_t*)any_pool,
            iree_hal_pool_set_select(&pool_set, params, 4096));

  params.queue_family_affinity = family_0 | family_2;
  EXPECT_EQ((iree_hal_pool_t*)any_pool,
            iree_hal_pool_set_select(&pool_set, params, 4096));

  params.queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY;
  EXPECT_EQ((iree_hal_pool_t*)any_pool,
            iree_hal_pool_set_select(&pool_set, params, 4096));

  iree_hal_pool_set_deinitialize(&pool_set);
  iree_hal_pool_release((iree_hal_pool_t*)restricted_pool);
  iree_hal_pool_release((iree_hal_pool_t*)any_pool);
}

TEST(PoolSetTest, SelectsNoncoherentUnifiedMemory) {
  iree_hal_pool_set_t pool_set;
  IREE_ASSERT_OK(iree_hal_pool_set_initialize(
      /*initial_capacity=*/1, iree_allocator_system(), &pool_set));

  iree_hal_routing_test_pool_t* unified_pool = CreateRoutingTestPool(0);
  unified_pool->capabilities.memory_type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  EXPECT_FALSE(iree_all_bits_set(unified_pool->capabilities.memory_type,
                                 IREE_HAL_MEMORY_TYPE_HOST_COHERENT));
  IREE_ASSERT_OK(iree_hal_pool_set_register(&pool_set, /*priority=*/0,
                                            (iree_hal_pool_t*)unified_pool));

  iree_hal_buffer_params_t params = {0};
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  EXPECT_EQ((iree_hal_pool_t*)unified_pool,
            iree_hal_pool_set_select(&pool_set, params, 4096));

  params.type |= IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  EXPECT_EQ(nullptr, iree_hal_pool_set_select(&pool_set, params, 4096));

  iree_hal_pool_set_deinitialize(&pool_set);
  iree_hal_pool_release((iree_hal_pool_t*)unified_pool);
}

TEST(PoolSetTest, RoutesUsingPreparedAccessAndAlignment) {
  auto* general = CreateRoutingTestPool(0);
  auto* read_only = CreateRoutingTestPool(0);
  read_only->capabilities.allowed_access = IREE_HAL_MEMORY_ACCESS_READ;
  read_only->capabilities.max_allocation_alignment = 16;
  iree_hal_pool_set_t pool_set;
  IREE_ASSERT_OK(
      iree_hal_pool_set_initialize(2, iree_allocator_system(), &pool_set));
  IREE_ASSERT_OK(iree_hal_pool_set_register(&pool_set, 0, &general->base));
  IREE_ASSERT_OK(iree_hal_pool_set_register(&pool_set, 1, &read_only->base));
  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  params.access = IREE_HAL_MEMORY_ACCESS_READ;
  params.min_alignment = 16;
  EXPECT_EQ(iree_hal_pool_set_select(&pool_set, params, 128), &read_only->base);
  params.access = IREE_HAL_MEMORY_ACCESS_WRITE;
  EXPECT_EQ(iree_hal_pool_set_select(&pool_set, params, 128), &general->base);
  params.access = IREE_HAL_MEMORY_ACCESS_READ;
  params.min_alignment = 32;
  EXPECT_EQ(iree_hal_pool_set_select(&pool_set, params, 128), &general->base);
  params.min_alignment = 128;
  EXPECT_EQ(iree_hal_pool_set_select(&pool_set, params, 128), nullptr);
  iree_hal_pool_set_deinitialize(&pool_set);
  iree_hal_pool_release(&read_only->base);
  iree_hal_pool_release(&general->base);
}

}  // namespace
