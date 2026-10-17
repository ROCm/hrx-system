// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/allocation.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <semaphore>
#include <thread>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/testing/deferred_pool.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/hal/utils/host_semaphore.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct CaptureAllocator {
  // Successful host allocations, including oversized transaction arrays.
  std::atomic<size_t> allocations{0};
  // Allocations not yet returned, including idle reusable blocks.
  std::atomic<size_t> live{0};
  // Cold test configuration; changed only when no capture is using it.
  size_t allocation_limit = SIZE_MAX;

  iree_allocator_t allocator() { return {this, Control}; }

  static iree_status_t Control(void* user_data,
                               iree_allocator_command_t command,
                               const void* params, void** inout_pointer) {
    auto* self = static_cast<CaptureAllocator*>(user_data);
    const bool had_pointer = *inout_pointer != nullptr;
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        self->allocations.load() >= self->allocation_limit) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "capture allocator exhausted");
    }
    const auto system = iree_allocator_system();
    iree_status_t status =
        system.ctl(system.self, command, params, inout_pointer);
    if (iree_status_is_ok(status)) {
      if (command == IREE_ALLOCATOR_COMMAND_FREE && had_pointer) {
        self->live.fetch_sub(1);
      } else if (command != IREE_ALLOCATOR_COMMAND_FREE && !had_pointer) {
        self->live.fetch_add(1);
        self->allocations.fetch_add(1);
      }
    }
    return status;
  }
};

struct Completion {
  // Actual public HAL point supplied to the allocation operation.
  iree_hal_semaphore_t* semaphore = nullptr;
  // Requested value, stable through submission capture.
  uint64_t value = 1;
  // Observes signaling without a polling worker.
  iree_async_semaphore_timepoint_t timepoint;
  // Callback join, independent of the operation's device lifetime.
  std::binary_semaphore done{0};
  // Owned terminal status transferred by the public point callback.
  iree::Status status;
  // Actual completion context, published through done.
  std::thread::id thread;
  // Observes resources at the public signal boundary, before waking the test.
  std::function<void()> observe;

  explicit Completion(iree_async_proactor_t* proactor) {
    IREE_CHECK_OK(iree_hal_host_semaphore_create(
        proactor, 0, iree_allocator_system(), &semaphore));
    timepoint = {
        .callback =
            [](void* user_data, iree_async_semaphore_timepoint_t*,
               iree_status_t status) {
              auto* self = static_cast<Completion*>(user_data);
              self->status = iree::Status(std::move(status));
              self->thread = std::this_thread::get_id();
              if (self->observe) {
                self->observe();
              }
              self->done.release();
            },
        .user_data = this,
    };
    IREE_CHECK_OK(iree_async_semaphore_acquire_timepoint(
        reinterpret_cast<iree_async_semaphore_t*>(semaphore), value,
        &timepoint));
  }
  ~Completion() { iree_hal_semaphore_release(semaphore); }
  iree_hal_semaphore_list_t signals() { return {1, &semaphore, &value}; }
  void Await() { done.acquire(); }
};

class AllocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    iree_hal_mock_device_options_t options;
    iree_hal_mock_device_options_initialize(&options);
    IREE_ASSERT_OK(iree_hal_mock_device_create(
        &options, iree_allocator_system(), &device_));
    native_.event_sink = iree_hal_device_event_sink_discard();
    iree_hal_amd_gpu_allocation_context_initialize(
        device_, &native_, &publication_, 0, allocator_.allocator(), &context_);
    sources_[0] =
        iree_hal_amd_gpu_allocation_context_publication_source(&context_);
    sources_[1] = {
        .publish =
            [](void* user_data) {
              auto* self = static_cast<AllocationTest*>(user_data);
              self->entered_.release();
              self->resume_.acquire();
              return false;
            },
        .user_data = this,
    };
    IREE_ASSERT_OK(iree_hal_amd_gpu_publication_initialize(
        2, sources_, {}, iree_allocator_system(), &publication_));
    iree_hal_slab_provider_t* provider = nullptr;
    IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(0, iree_allocator_system(),
                                                     &provider));
    IREE_ASSERT_OK(iree_async_notification_create(
        proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification_));
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker_));
    IREE_ASSERT_OK(
        iree_hal_passthrough_pool_create({}, provider, notification_, tracker_,
                                         iree_allocator_system(), &pool_));
    iree_hal_slab_provider_release(provider);
    IREE_ASSERT_OK(
        iree::hal::testing::DeferredBackingPool::Create(pool_, &deferred_));
  }

  void TearDown() override {
    iree_hal_amd_gpu_publication_deinitialize(&publication_);
    iree_hal_pool_release(deferred_);
    iree_hal_pool_release(pool_);
    iree_async_frontier_tracker_release(tracker_);
    iree_async_notification_release(notification_);
    iree_hal_amd_gpu_allocation_context_deinitialize(&context_);
    EXPECT_EQ(allocator_.live.load(), 0u);
    iree_hal_device_release(device_);
    iree_async_proactor_release(proactor_);
  }

  iree_hal_buffer_t* Allocate(iree_hal_pool_t* pool, Completion& completion) {
    const auto request = iree_hal_make_pool_reservation_request(128, 16);
    iree_hal_buffer_t* buffer = nullptr;
    IREE_CHECK_OK(iree_hal_amd_gpu_queue_alloca(
        &context_, 0, iree_hal_semaphore_list_empty(), completion.signals(),
        pool, 1, &request, &buffer));
    return buffer;
  }

  void Release(iree_hal_buffer_t* buffer) {
    Completion released(proactor_);
    IREE_ASSERT_OK(iree_hal_amd_gpu_queue_dealloca(
        &context_, 0, iree_hal_semaphore_list_empty(), released.signals(), 1,
        &buffer));
    released.Await();
    IREE_EXPECT_OK(released.status);
    EXPECT_EQ(released.thread, std::this_thread::get_id());
    iree_hal_buffer_release(buffer);
  }

  void HoldPublisher() {
    iree_hal_amd_gpu_publication_notify(&publication_, 1);
    entered_.acquire();
  }

  // Real host semaphore infrastructure, without a poll thread.
  iree_async_proactor_t* proactor_ = nullptr;
  // Pool capacity and exact retirement bookkeeping use real async primitives.
  iree_async_notification_t* notification_ = nullptr;
  // Retained by the native pool through any retirement descriptors.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
  // Parent identity retained by actual transient buffers and captures.
  iree_hal_device_t* device_ = nullptr;
  // Diagnostic sink; these CPU ownership tests issue no native GPU calls.
  iree_hal_amd_gpu_context_t native_ = {};
  // Actual capture storage dependency, used to verify rollback and reclamation.
  CaptureAllocator allocator_;
  // Actual queue-allocation state under test.
  iree_hal_amd_gpu_allocation_context_t context_;
  // Actual deferred publisher, independent of backing production.
  iree_hal_amd_gpu_publication_t publication_;
  // Fixed allocation result source and explicit test gate.
  iree_hal_amd_gpu_publication_source_t sources_[2];
  // Gate proves which work needs publication without sleeps or deadlines.
  std::binary_semaphore entered_{0};
  // Releases the publisher after the test observes the required ordering.
  std::binary_semaphore resume_{0};
  // Real CPU storage and reservation/materialization behavior.
  iree_hal_pool_t* pool_ = nullptr;
  // Only the acquisition dependency is delayed; buffers and tokens are real.
  iree::hal::testing::DeferredBackingPool* deferred_ = nullptr;
};

TEST_F(AllocationTest, WarmAllocationAndReturnNeedNoPublisher) {
  HoldPublisher();
  Completion allocated(proactor_);
  auto* buffer = Allocate(pool_, allocated);
  allocated.Await();
  IREE_EXPECT_OK(allocated.status);
  EXPECT_EQ(allocated.thread, std::this_thread::get_id());
  IREE_ASSERT_OK(iree_hal_buffer_map_fill(buffer, 0, 128, "abcd", 4));
  Release(buffer);
  resume_.release();
}

TEST_F(AllocationTest, DeferredBackingDoesNotBlockIndependentWarmAllocation) {
  deferred_->defer = true;
  Completion cold(proactor_), warm(proactor_);
  auto* cold_buffer = Allocate(deferred_, cold);
  ASSERT_EQ(deferred_->pending.count, 1u);
  auto* warm_buffer = Allocate(pool_, warm);
  warm.Await();
  IREE_EXPECT_OK(warm.status);
  EXPECT_EQ(warm.thread, std::this_thread::get_id());
  Release(warm_buffer);
  uint64_t value = UINT64_MAX;
  IREE_ASSERT_OK(iree_hal_semaphore_query(cold.semaphore, &value));
  EXPECT_EQ(value, 0u);
  deferred_->CompleteOne();
  cold.Await();
  IREE_EXPECT_OK(cold.status);
  EXPECT_NE(cold.thread, std::this_thread::get_id());
  Release(cold_buffer);
}

TEST_F(AllocationTest, NativeCompletionOnlyPublishesItsResult) {
  HoldPublisher();
  deferred_->defer = true;
  Completion allocated(proactor_);
  auto* buffer = Allocate(deferred_, allocated);
  deferred_->CompleteOne();
  uint64_t value = UINT64_MAX;
  IREE_ASSERT_OK(iree_hal_semaphore_query(allocated.semaphore, &value));
  EXPECT_EQ(value, 0u);
  resume_.release();
  allocated.Await();
  IREE_EXPECT_OK(allocated.status);
  EXPECT_NE(allocated.thread, std::this_thread::get_id());
  IREE_ASSERT_OK(iree_hal_buffer_map_fill(buffer, 0, 128, "abcd", 4));
  Release(buffer);
}

TEST_F(AllocationTest, FailedNativeAcquisitionLeavesNoReservationToReturn) {
  deferred_->defer = true;
  Completion allocated(proactor_);
  auto* buffer = Allocate(deferred_, allocated);
  deferred_->FailOne();
  allocated.Await();
  EXPECT_THAT(allocated.status, iree::testing::status::StatusIs(
                                    iree::StatusCode::kResourceExhausted));
  iree_hal_buffer_release(buffer);
}

class AllocationBatchTest : public AllocationTest,
                            public ::testing::WithParamInterface<size_t> {};

TEST_P(AllocationBatchTest, CommitsAndReturnsWholeBatchBeforePublicSignals) {
  const size_t count = GetParam();
  std::vector<iree_hal_pool_reservation_request_t> requests(
      count, iree_hal_make_pool_reservation_request(16, 16));
  std::vector<iree_hal_buffer_t*> buffers(count, nullptr);
  Completion allocated(proactor_);
  allocated.observe = [&] {
    IREE_EXPECT_OK(allocated.status);
    for (auto* buffer : buffers) {
      EXPECT_EQ(iree_atomic_ref_count_load(&buffer->resource.ref_count), 1);
      IREE_EXPECT_OK(iree_hal_buffer_map_fill(buffer, 0, 16, "abcd", 4));
    }
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(pool_, &stats);
    EXPECT_EQ(stats.reservation_count, count);
  };
  deferred_->defer = true;
  IREE_ASSERT_OK(iree_hal_amd_gpu_queue_alloca(
      &context_, 0, {}, allocated.signals(), deferred_, count, requests.data(),
      buffers.data()));
  EXPECT_EQ(deferred_->pending.count, count);
  for (auto* buffer : buffers) {
    EXPECT_EQ(iree_atomic_ref_count_load(&buffer->resource.ref_count), 2);
  }
  // The caller can discard every input array and its signal reference after
  // acceptance. The operation owns the complete capture until publication.
  std::fill(requests.begin(), requests.end(),
            iree_hal_pool_reservation_request_t{});
  iree_hal_semaphore_release(allocated.semaphore);
  allocated.semaphore = nullptr;
  deferred_->CompleteOne();
  allocated.Await();
  IREE_ASSERT_OK(allocated.status);

  Completion released(proactor_);
  released.observe = [&] {
    IREE_EXPECT_OK(released.status);
    for (auto* buffer : buffers) {
      EXPECT_EQ(iree_atomic_ref_count_load(&buffer->resource.ref_count), 1);
      const auto binding = iree_hal_buffer_native_binding(
          buffer, {buffer->host_binding_index, IREE_HAL_BUFFER_INTERFACE_HOST});
      EXPECT_EQ(binding.host_pointer, nullptr);
    }
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(pool_, &stats);
    EXPECT_EQ(stats.reservation_count, 0u);
    EXPECT_EQ(stats.reserve_count, count);
    EXPECT_EQ(stats.release_count, count);
  };
  IREE_ASSERT_OK(iree_hal_amd_gpu_queue_dealloca(
      &context_, 0, {}, released.signals(), count, buffers.data()));
  released.Await();
  IREE_ASSERT_OK(released.status);
  for (auto* buffer : buffers) {
    iree_hal_buffer_release(buffer);
  }
}

TEST_P(AllocationBatchTest,
       FailedAcquisitionReleasesCaptureBeforeFailingSignal) {
  const size_t count = GetParam();
  std::vector<iree_hal_pool_reservation_request_t> requests(
      count, iree_hal_make_pool_reservation_request(16, 16));
  std::vector<iree_hal_buffer_t*> buffers(count, nullptr);
  Completion allocated(proactor_);
  allocated.observe = [&] {
    EXPECT_THAT(allocated.status, iree::testing::status::StatusIs(
                                      iree::StatusCode::kResourceExhausted));
    for (auto* buffer : buffers) {
      EXPECT_EQ(iree_atomic_ref_count_load(&buffer->resource.ref_count), 1);
      const auto binding = iree_hal_buffer_native_binding(
          buffer, {buffer->host_binding_index, IREE_HAL_BUFFER_INTERFACE_HOST});
      EXPECT_EQ(binding.host_pointer, nullptr);
    }
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(pool_, &stats);
    EXPECT_EQ(stats.reservation_count, 0u);
    EXPECT_EQ(stats.reserve_count, 0u);
  };
  deferred_->defer = true;
  IREE_ASSERT_OK(iree_hal_amd_gpu_queue_alloca(
      &context_, 0, {}, allocated.signals(), deferred_, count, requests.data(),
      buffers.data()));
  deferred_->FailOne();
  allocated.Await();
  for (auto* buffer : buffers) {
    iree_hal_buffer_release(buffer);
  }
}

INSTANTIATE_TEST_SUITE_P(CaptureBoundaries, AllocationBatchTest,
                         ::testing::Values(8, 9, 127, 128, 1023, 1024, 1660,
                                           65536));

TEST_F(AllocationTest, PartialCaptureFailurePreservesOutputsAndReferences) {
  constexpr size_t kCount = 1660;
  const auto request = iree_hal_make_pool_reservation_request(16, 16);
  std::vector<iree_hal_pool_reservation_request_t> requests(kCount, request);
  Completion sentinel_ready(proactor_);
  auto* sentinel = Allocate(pool_, sentinel_ready);
  sentinel_ready.Await();
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_host_semaphore_create(
      proactor_, 0, iree_allocator_system(), &semaphore));
  uint64_t value = 1;
  const auto* device_resource =
      reinterpret_cast<const iree_hal_resource_t*>(device_);
  const auto* semaphore_resource =
      reinterpret_cast<const iree_hal_resource_t*>(semaphore);
  const auto device_references =
      iree_atomic_ref_count_load(&device_resource->ref_count);
  // Exercise failure before the capture, in variable arrays, and after long
  // initialized root prefixes. Successful prior captures cannot mask failure
  // by leaving reusable metadata blocks available.
  for (size_t limit : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 64u, 1024u, 1660u}) {
    SCOPED_TRACE(limit);
    iree_arena_block_pool_trim(&context_.blocks);
    const auto live = allocator_.live.load();
    allocator_.allocation_limit = allocator_.allocations.load() + limit;
    std::vector<iree_hal_buffer_t*> buffers(kCount, sentinel);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          iree_hal_amd_gpu_queue_alloca(
                              &context_, 0, {}, {1, &semaphore, &value}, pool_,
                              kCount, requests.data(), buffers.data()));
    for (auto* buffer : buffers) {
      EXPECT_EQ(buffer, sentinel);
    }
    EXPECT_EQ(iree_atomic_ref_count_load(&device_resource->ref_count),
              device_references);
    EXPECT_EQ(iree_atomic_ref_count_load(&semaphore_resource->ref_count), 1);
    uint64_t signaled = UINT64_MAX;
    IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &signaled));
    EXPECT_EQ(signaled, 0u);
    iree_arena_block_pool_trim(&context_.blocks);
    EXPECT_EQ(allocator_.live.load(), live);
  }
  allocator_.allocation_limit = SIZE_MAX;
  iree_hal_semaphore_release(semaphore);
  Release(sentinel);
}

TEST_F(AllocationTest, FailedLargeDeallocationRestoresEveryRootBeforeSignal) {
  constexpr size_t kCount = 1660;
  std::vector<iree_hal_pool_reservation_request_t> requests(
      kCount, iree_hal_make_pool_reservation_request(16, 16));
  std::vector<iree_hal_buffer_t*> buffers(kCount, nullptr);
  Completion allocated(proactor_);
  IREE_ASSERT_OK(iree_hal_amd_gpu_queue_alloca(
      &context_, 0, {}, allocated.signals(), pool_, kCount, requests.data(),
      buffers.data()));
  allocated.Await();
  IREE_ASSERT_OK(allocated.status);

  iree_hal_semaphore_t* dependency = nullptr;
  IREE_ASSERT_OK(iree_hal_host_semaphore_create(
      proactor_, 0, iree_allocator_system(), &dependency));
  uint64_t dependency_value = 1;
  Completion cancelled(proactor_), retried(proactor_);
  cancelled.observe = [&] {
    EXPECT_THAT(cancelled.status,
                iree::testing::status::StatusIs(iree::StatusCode::kCancelled));
    for (auto* buffer : buffers) {
      EXPECT_EQ(iree_atomic_ref_count_load(&buffer->resource.ref_count), 1);
      IREE_EXPECT_OK(iree_hal_buffer_map_fill(buffer, 0, 16, "abcd", 4));
    }
    // The terminal signal may immediately admit another use. Every exclusive
    // deallocation mark must already have been rolled back at this boundary.
    IREE_CHECK_OK(iree_hal_amd_gpu_queue_dealloca(
        &context_, 0, {}, retried.signals(), kCount, buffers.data()));
  };
  IREE_ASSERT_OK(iree_hal_amd_gpu_queue_dealloca(
      &context_, 0, {1, &dependency, &dependency_value}, cancelled.signals(),
      kCount, buffers.data()));
  iree_hal_semaphore_fail(
      dependency, iree_make_status(IREE_STATUS_CANCELLED, "request cancelled"));
  cancelled.Await();
  retried.Await();
  IREE_EXPECT_OK(retried.status);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.reserve_count, kCount);
  EXPECT_EQ(stats.release_count, kCount);
  iree_hal_semaphore_release(dependency);
  for (auto* buffer : buffers) {
    iree_hal_buffer_release(buffer);
  }
}

}  // namespace
