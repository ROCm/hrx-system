// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/allocation.h"

#include <algorithm>
#include <atomic>
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

// Observes only preparation. Tests own acceptance and retirement explicitly;
// semaphore admission and native ordering are exercised by the real queues.
struct Preparation {
  // Joins the last callback access with the observing test thread.
  std::binary_semaphore done{0};
  // Owned preparation result, published through done.
  iree::Status status;
  // Actual completion context, published through done.
  std::thread::id thread;

  static void Ready(void* user_data, iree_status_t status) {
    auto* self = static_cast<Preparation*>(user_data);
    self->status = iree::Status(std::move(status));
    self->thread = std::this_thread::get_id();
    self->done.release();
  }

  bool Begin(iree_hal_amd_gpu_allocation_t* operation) {
    bool pending = false;
    iree_status_t result = iree_hal_amd_gpu_allocation_prepare(
        operation, 0, 0, nullptr, Ready, this, &pending);
    if (!pending) {
      Ready(this, result);
    }
    return pending;
  }
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
    iree_hal_amd_gpu_allocation_context_initialize(
        device_, &publication_, 0, allocator_.allocator(), &context_);
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

  iree_hal_buffer_t* Capture(iree_hal_pool_t* pool,
                             iree_hal_amd_gpu_allocation_t** out_operation) {
    const auto request = iree_hal_make_pool_reservation_request(128, 16);
    iree_hal_buffer_t* buffer = nullptr;
    IREE_CHECK_OK(iree_hal_amd_gpu_allocation_capture_alloca(
        &context_, pool, 1, &request, &buffer, out_operation));
    return buffer;
  }

  void Release(iree_hal_buffer_t* buffer) {
    iree_hal_amd_gpu_allocation_t* operation = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_gpu_allocation_capture_dealloca(
        &context_, 1, &buffer, &operation));
    Preparation released;
    EXPECT_FALSE(released.Begin(operation));
    released.Await();
    IREE_ASSERT_OK(released.status);
    iree_hal_amd_gpu_allocation_accept(operation, nullptr);
    iree_hal_amd_gpu_allocation_destroy(operation);
    iree_hal_buffer_release(buffer);
  }

  void HoldPublisher() {
    iree_hal_amd_gpu_publication_notify(&publication_, 1);
    entered_.acquire();
  }

  // Real pool notification infrastructure, without a poll thread.
  iree_async_proactor_t* proactor_ = nullptr;
  // Pool capacity and exact retirement bookkeeping use real async primitives.
  iree_async_notification_t* notification_ = nullptr;
  // Retained by the native pool through any retirement descriptors.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
  // Parent identity borrowed by actual transient buffers and captures.
  iree_hal_device_t* device_ = nullptr;
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
  iree_hal_amd_gpu_allocation_t* operation = nullptr;
  auto* buffer = Capture(pool_, &operation);
  Preparation prepared;
  EXPECT_FALSE(prepared.Begin(operation));
  prepared.Await();
  IREE_EXPECT_OK(prepared.status);
  EXPECT_EQ(prepared.thread, std::this_thread::get_id());
  iree_hal_amd_gpu_allocation_accept(operation, nullptr);
  iree_hal_amd_gpu_allocation_destroy(operation);
  IREE_EXPECT_OK(iree_hal_buffer_map_fill(buffer, 0, 128, "abcd", 4));
  Release(buffer);
  resume_.release();
}

TEST_F(AllocationTest, DeferredBackingDoesNotBlockIndependentWarmAllocation) {
  deferred_->defer = true;
  iree_hal_amd_gpu_allocation_t* cold = nullptr;
  iree_hal_amd_gpu_allocation_t* warm = nullptr;
  auto* cold_buffer = Capture(deferred_, &cold);
  Preparation cold_prepared, warm_prepared;
  EXPECT_TRUE(cold_prepared.Begin(cold));
  ASSERT_EQ(deferred_->pending.count, 1u);
  auto* warm_buffer = Capture(pool_, &warm);
  EXPECT_FALSE(warm_prepared.Begin(warm));
  warm_prepared.Await();
  IREE_EXPECT_OK(warm_prepared.status);
  EXPECT_EQ(warm_prepared.thread, std::this_thread::get_id());
  iree_hal_amd_gpu_allocation_accept(warm, nullptr);
  iree_hal_amd_gpu_allocation_destroy(warm);
  Release(warm_buffer);
  EXPECT_FALSE(cold_prepared.done.try_acquire());
  deferred_->CompleteOne();
  cold_prepared.Await();
  IREE_EXPECT_OK(cold_prepared.status);
  EXPECT_NE(cold_prepared.thread, std::this_thread::get_id());
  iree_hal_amd_gpu_allocation_accept(cold, nullptr);
  iree_hal_amd_gpu_allocation_destroy(cold);
  Release(cold_buffer);
}

TEST_F(AllocationTest, NativeCompletionOnlyPublishesItsResult) {
  HoldPublisher();
  deferred_->defer = true;
  iree_hal_amd_gpu_allocation_t* operation = nullptr;
  auto* buffer = Capture(deferred_, &operation);
  Preparation prepared;
  EXPECT_TRUE(prepared.Begin(operation));
  deferred_->CompleteOne();
  EXPECT_FALSE(prepared.done.try_acquire());
  resume_.release();
  prepared.Await();
  IREE_EXPECT_OK(prepared.status);
  EXPECT_NE(prepared.thread, std::this_thread::get_id());
  iree_hal_amd_gpu_allocation_accept(operation, nullptr);
  iree_hal_amd_gpu_allocation_destroy(operation);
  IREE_EXPECT_OK(iree_hal_buffer_map_fill(buffer, 0, 128, "abcd", 4));
  Release(buffer);
}

class AllocationBatchTest : public AllocationTest,
                            public ::testing::WithParamInterface<size_t> {};

TEST_P(AllocationBatchTest, AcceptanceCommitsAndRetirementDecommitsWholeBatch) {
  const size_t count = GetParam();
  std::vector<iree_hal_pool_reservation_request_t> requests(
      count, iree_hal_make_pool_reservation_request(16, 16));
  std::vector<iree_hal_buffer_t*> buffers(count, nullptr);
  iree_hal_amd_gpu_allocation_t* operation = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_allocation_capture_alloca(
      &context_, deferred_, count, requests.data(), buffers.data(),
      &operation));
  // The caller can discard the input array immediately after capture.
  std::fill(requests.begin(), requests.end(),
            iree_hal_pool_reservation_request_t{});
  deferred_->defer = true;
  Preparation prepared;
  EXPECT_TRUE(prepared.Begin(operation));
  EXPECT_EQ(deferred_->pending.count, count);
  deferred_->CompleteOne();
  prepared.Await();
  IREE_ASSERT_OK(prepared.status);
  for (auto* buffer : buffers) {
    const auto binding = iree_hal_buffer_native_binding(
        buffer, {buffer->host_binding_index, IREE_HAL_BUFFER_INTERFACE_HOST});
    EXPECT_EQ(binding.host_pointer, nullptr);
  }
  // Native capacity deferral retries preparation without a second acquisition.
  Preparation retried;
  EXPECT_FALSE(retried.Begin(operation));
  retried.Await();
  IREE_EXPECT_OK(retried.status);
  iree_hal_amd_gpu_allocation_accept(operation, nullptr);
  iree_hal_amd_gpu_allocation_destroy(operation);
  for (auto* buffer : buffers) {
    EXPECT_EQ(iree_atomic_ref_count_load(&buffer->resource.ref_count), 1);
    IREE_EXPECT_OK(iree_hal_buffer_map_fill(buffer, 0, 16, "abcd", 4));
  }

  IREE_ASSERT_OK(iree_hal_amd_gpu_allocation_capture_dealloca(
      &context_, count, buffers.data(), &operation));
  Preparation returned;
  EXPECT_FALSE(returned.Begin(operation));
  returned.Await();
  IREE_ASSERT_OK(returned.status);
  iree_hal_amd_gpu_allocation_accept(operation, nullptr);
  // Logical return must preserve old binding metadata until native retirement.
  for (auto* buffer : buffers) {
    const auto binding = iree_hal_buffer_native_binding(
        buffer, {buffer->host_binding_index, IREE_HAL_BUFFER_INTERFACE_HOST});
    EXPECT_NE(binding.host_pointer, nullptr);
  }
  iree_hal_amd_gpu_allocation_destroy(operation);
  for (auto* buffer : buffers) {
    EXPECT_EQ(iree_atomic_ref_count_load(&buffer->resource.ref_count), 1);
    const auto binding = iree_hal_buffer_native_binding(
        buffer, {buffer->host_binding_index, IREE_HAL_BUFFER_INTERFACE_HOST});
    EXPECT_EQ(binding.host_pointer, nullptr);
    iree_hal_buffer_release(buffer);
  }
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.reserve_count, count);
  EXPECT_EQ(stats.release_count, count);
}

TEST_P(AllocationBatchTest, FailedAcquisitionLeavesNoReservationToReturn) {
  const size_t count = GetParam();
  std::vector<iree_hal_pool_reservation_request_t> requests(
      count, iree_hal_make_pool_reservation_request(16, 16));
  std::vector<iree_hal_buffer_t*> buffers(count, nullptr);
  iree_hal_amd_gpu_allocation_t* operation = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_allocation_capture_alloca(
      &context_, deferred_, count, requests.data(), buffers.data(),
      &operation));
  deferred_->defer = true;
  Preparation prepared;
  EXPECT_TRUE(prepared.Begin(operation));
  deferred_->FailOne();
  prepared.Await();
  EXPECT_THAT(prepared.status, iree::testing::status::StatusIs(
                                   iree::StatusCode::kResourceExhausted));
  iree_hal_amd_gpu_allocation_destroy(operation);
  for (auto* buffer : buffers) {
    EXPECT_EQ(iree_atomic_ref_count_load(&buffer->resource.ref_count), 1);
    const auto binding = iree_hal_buffer_native_binding(
        buffer, {buffer->host_binding_index, IREE_HAL_BUFFER_INTERFACE_HOST});
    EXPECT_EQ(binding.host_pointer, nullptr);
    iree_hal_buffer_release(buffer);
  }
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.reserve_count, 0u);
}

INSTANTIATE_TEST_SUITE_P(CaptureBoundaries, AllocationBatchTest,
                         ::testing::Values(1, 8, 9, 127, 128, 1023, 1024, 1660,
                                           65536));

TEST_F(AllocationTest, PartialCaptureFailurePreservesOutputsAndReferences) {
  constexpr size_t kCount = 1660;
  const auto request = iree_hal_make_pool_reservation_request(16, 16);
  std::vector<iree_hal_pool_reservation_request_t> requests(kCount, request);
  iree_hal_amd_gpu_allocation_t* operation = nullptr;
  auto* sentinel = Capture(pool_, &operation);
  Preparation prepared;
  EXPECT_FALSE(prepared.Begin(operation));
  prepared.Await();
  IREE_ASSERT_OK(prepared.status);
  iree_hal_amd_gpu_allocation_accept(operation, nullptr);
  iree_hal_amd_gpu_allocation_destroy(operation);
  const auto* device_resource =
      reinterpret_cast<const iree_hal_resource_t*>(device_);
  const auto device_references =
      iree_atomic_ref_count_load(&device_resource->ref_count);
  // Exercise failure before capture, in variable arrays, and after long
  // initialized root prefixes, with no cached metadata masking allocation.
  for (size_t limit : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 64u, 1024u, 1660u}) {
    SCOPED_TRACE(limit);
    iree_arena_block_pool_trim(&context_.blocks);
    const auto live = allocator_.live.load();
    allocator_.allocation_limit = allocator_.allocations.load() + limit;
    std::vector<iree_hal_buffer_t*> buffers(kCount, sentinel);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          iree_hal_amd_gpu_allocation_capture_alloca(
                              &context_, pool_, kCount, requests.data(),
                              buffers.data(), &operation));
    EXPECT_EQ(operation, nullptr);
    for (auto* buffer : buffers) {
      EXPECT_EQ(buffer, sentinel);
    }
    EXPECT_EQ(iree_atomic_ref_count_load(&device_resource->ref_count),
              device_references);
    iree_arena_block_pool_trim(&context_.blocks);
    EXPECT_EQ(allocator_.live.load(), live);
  }
  allocator_.allocation_limit = SIZE_MAX;
  Release(sentinel);
}

TEST_F(AllocationTest, RejectedPreparationReturnsUnusedReservation) {
  iree_hal_amd_gpu_allocation_t* operation = nullptr;
  auto* buffer = Capture(pool_, &operation);
  Preparation prepared;
  EXPECT_FALSE(prepared.Begin(operation));
  prepared.Await();
  IREE_ASSERT_OK(prepared.status);
  iree_hal_amd_gpu_allocation_destroy(operation);
  EXPECT_EQ(iree_atomic_ref_count_load(&buffer->resource.ref_count), 1);
  const auto binding = iree_hal_buffer_native_binding(
      buffer, {buffer->host_binding_index, IREE_HAL_BUFFER_INTERFACE_HOST});
  EXPECT_EQ(binding.host_pointer, nullptr);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.reserve_count, 1u);
  EXPECT_EQ(stats.release_count, 1u);
  iree_hal_buffer_release(buffer);
}

TEST_F(AllocationTest, RejectedDeallocationRestoresEveryRoot) {
  constexpr size_t kCount = 1660;
  std::vector<iree_hal_pool_reservation_request_t> requests(
      kCount, iree_hal_make_pool_reservation_request(16, 16));
  std::vector<iree_hal_buffer_t*> buffers(kCount, nullptr);
  iree_hal_amd_gpu_allocation_t* operation = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_allocation_capture_alloca(
      &context_, pool_, kCount, requests.data(), buffers.data(), &operation));
  Preparation prepared;
  EXPECT_FALSE(prepared.Begin(operation));
  prepared.Await();
  IREE_ASSERT_OK(prepared.status);
  iree_hal_amd_gpu_allocation_accept(operation, nullptr);
  iree_hal_amd_gpu_allocation_destroy(operation);

  IREE_ASSERT_OK(iree_hal_amd_gpu_allocation_capture_dealloca(
      &context_, kCount, buffers.data(), &operation));
  iree_hal_amd_gpu_allocation_destroy(operation);
  for (auto* buffer : buffers) {
    EXPECT_EQ(iree_atomic_ref_count_load(&buffer->resource.ref_count), 1);
    IREE_EXPECT_OK(iree_hal_buffer_map_fill(buffer, 0, 16, "abcd", 4));
  }
  // Rejection must release every exclusive mark before another capture.
  IREE_ASSERT_OK(iree_hal_amd_gpu_allocation_capture_dealloca(
      &context_, kCount, buffers.data(), &operation));
  Preparation returned;
  EXPECT_FALSE(returned.Begin(operation));
  returned.Await();
  IREE_ASSERT_OK(returned.status);
  iree_hal_amd_gpu_allocation_accept(operation, nullptr);
  iree_hal_amd_gpu_allocation_destroy(operation);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.reserve_count, kCount);
  EXPECT_EQ(stats.release_count, kCount);
  for (auto* buffer : buffers) {
    iree_hal_buffer_release(buffer);
  }
}

}  // namespace
