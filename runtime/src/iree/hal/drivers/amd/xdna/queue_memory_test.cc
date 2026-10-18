// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <thread>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/hal/drivers/amd/xdna/queue.h"
#include "iree/hal/drivers/amd/xdna/testing/queue_test_util.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/hal/testing/deferred_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::hal::amd::xdna::testing::ExpectHostAllocationsUnchanged;
using iree::hal::amd::xdna::testing::HostAllocationCounters;
using iree::hal::amd::xdna::testing::HostAllocationSnapshot;
using iree::hal::amd::xdna::testing::NativeProvider;
using iree::hal::amd::xdna::testing::QueueHarness;
using iree::hal::amd::xdna::testing::SnapshotHostAllocations;
using iree::hal::amd::xdna::testing::TrackingAllocator;
using iree::hal::testing::DeferredBackingPool;

struct ScopedMemoryGate {
  // Native dependency gate released even after a test assertion returns.
  NativeProvider::MemoryGate* gate;
  ~ScopedMemoryGate() { gate->Release(); }
};

TEST(XdnaQueueTest, CachedQueueAllocationPublishesContractBeforeDispatch) {
  QueueHarness harness;
  harness.native.hold_retirement = true;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.SealDeviceGroup());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());

  iree_hal_pool_t* source_pool = nullptr;
  IREE_ASSERT_OK(harness.CreateCachedTLSFPool(
      IREE_HAL_BUFFER_USAGE_STORAGE_READ, &source_pool));
  iree_hal_executable_t* executable = nullptr;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(harness.LoadExecutable(
      iree::hal::amd::xdna::testing::ImageFixture(), &executable, &function));
  const uint64_t expected_allocation_address = harness.native.next_address;

  iree_hal_semaphore_t* allocation_ready = nullptr;
  iree_hal_semaphore_t* dispatch_done = nullptr;
  iree_hal_semaphore_t* deallocation_done = nullptr;
  for (auto** semaphore :
       {&allocation_ready, &dispatch_done, &deallocation_done}) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, semaphore));
  }

  std::array<iree_hal_pool_reservation_request_t, 2> requests = {};
  for (auto& request : requests) {
    request.allocation_size = 64;
  }
  std::array<iree_hal_buffer_t*, 2> buffers = {};
  uint64_t value = 1;
  const size_t memory_create_count = harness.native.memory_create_count;
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      harness.queue, {}, {1, &allocation_ready, &value}, source_pool,
      requests.size(), requests.data(), buffers.data()));
  ASSERT_NE(buffers[0], nullptr);
  ASSERT_NE(buffers[1], nullptr);
  EXPECT_EQ(iree_hal_buffer_allowed_access(buffers[0]),
            IREE_HAL_MEMORY_ACCESS_READ);
  EXPECT_EQ(iree_hal_buffer_allowed_access(buffers[1]),
            IREE_HAL_MEMORY_ACCESS_READ);
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(allocation_ready, value));
  EXPECT_EQ(harness.native.memory_create_count, memory_create_count + 1);
  iree_hal_memory_scope_t program_scope;
  const iree_hal_memory_site_t program_site = {
      .kind = IREE_HAL_MEMORY_SITE_PROGRAM,
      .family = iree_hal_queue_family(harness.queue),
  };
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      harness.group, program_site, &program_scope));
  iree_hal_buffer_native_binding_slot_t slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      source_pool, program_scope, IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
      &slot));
  const uint64_t first_address =
      iree_hal_buffer_native_binding(buffers[0], slot).device_address;
  const uint64_t second_address =
      iree_hal_buffer_native_binding(buffers[1], slot).device_address;
  EXPECT_EQ(first_address, expected_allocation_address);
  EXPECT_NE(second_address, first_address);

  const iree_hal_buffer_ref_t binding =
      iree_hal_make_buffer_ref(buffers[0], 0, requests[0].allocation_size);
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      harness.queue, {1, &allocation_ready, &value},
      {1, &dispatch_done, &value}, executable, function,
      iree_hal_make_static_dispatch_config(1, 1, 1), {}, {1, &binding},
      /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(1));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilNotificationCount(1));
  {
    std::lock_guard<std::mutex> lock(harness.native.mutex);
    ASSERT_EQ(harness.native.pending_commands.size(), 1u);
    const auto& command_bytes = harness.native.pending_commands[0].bytes;
    ASSERT_GE(command_bytes.size(), 16u);
    const uint64_t patched_address = expected_allocation_address + 4;
    EXPECT_EQ(iree_unaligned_load_le_u32(command_bytes.data() + 8),
              (uint32_t)patched_address | 1u);
    EXPECT_EQ(iree_unaligned_load_le_u32(command_bytes.data() + 12),
              UINT32_C(0xA5A50000) | (uint32_t)(patched_address >> 32));
  }

  harness.native.hold_retirement = false;
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(dispatch_done, value));
  IREE_ASSERT_OK(iree_hal_queue_dealloca(
      harness.queue, {1, &dispatch_done, &value},
      {1, &deallocation_done, &value}, buffers.size(), buffers.data()));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(deallocation_done, value));

  for (iree_hal_buffer_t* buffer : buffers) {
    iree_hal_buffer_release(buffer);
  }
  iree_hal_executable_release(executable);
  iree_hal_pool_release(source_pool);
  iree_hal_semaphore_release(deallocation_done);
  iree_hal_semaphore_release(dispatch_done);
  iree_hal_semaphore_release(allocation_ready);
}

TEST(XdnaQueueTest, QueueTransfersUsePrivateCachedPoolMapping) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.SealDeviceGroup());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());

  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      harness.CreateCachedTLSFPool(IREE_HAL_BUFFER_USAGE_TRANSFER, &pool));
  ASSERT_NE(pool->memory_contract, nullptr);
  EXPECT_EQ(pool->memory_contract->host.access, IREE_HAL_MEMORY_ACCESS_NONE);

  iree_hal_semaphore_t* progress = nullptr;
  iree_hal_semaphore_t* retired = nullptr;
  for (auto** semaphore : {&progress, &retired}) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, semaphore));
  }

  const iree_hal_pool_reservation_request_t request = {
      .allocation_size = 64,
  };
  iree_hal_buffer_t* buffer = nullptr;
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      harness.queue, {}, {1, &progress, &value}, pool, 1, &request, &buffer));

  const std::array<uint32_t, 4> expected = {
      UINT32_C(0x13579BDF), UINT32_C(0x2468ACE0), UINT32_C(0xA5A55A5A),
      UINT32_C(0xC001D00D)};
  iree_hal_transfer_operation_t upload = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
  };
  upload.upload.source = expected.data();
  upload.upload.target_buffer = buffer;
  upload.upload.length = sizeof(expected);
  uint64_t upload_value = 2;
  IREE_ASSERT_OK(iree_hal_queue_transfer(harness.queue, {1, &progress, &value},
                                         {1, &progress, &upload_value}, 1,
                                         &upload, /*barriers=*/NULL));

  std::array<uint32_t, 4> actual = {};
  iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
  };
  download.download.source_buffer = buffer;
  download.download.target = actual.data();
  download.download.length = sizeof(actual);
  uint64_t download_value = 3;
  IREE_ASSERT_OK(iree_hal_queue_transfer(
      harness.queue, {1, &progress, &upload_value},
      {1, &progress, &download_value}, 1, &download, /*barriers=*/NULL));

  IREE_ASSERT_OK(iree_hal_queue_dealloca(harness.queue,
                                         {1, &progress, &download_value},
                                         {1, &retired, &value}, 1, &buffer));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(retired, value));
  EXPECT_EQ(actual, expected);

  iree_hal_buffer_release(buffer);
  iree_hal_semaphore_release(retired);
  iree_hal_semaphore_release(progress);
  iree_hal_pool_release(pool);
}

TEST(XdnaQueueTest, LargeAllocationBatchRetainsTransfersUntilRetirement) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.SealDeviceGroup());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      harness.CreateCachedTLSFPool(IREE_HAL_BUFFER_USAGE_TRANSFER, &pool));
  iree_hal_semaphore_t* progress = nullptr;
  iree_hal_semaphore_t* retired = nullptr;
  for (auto** semaphore : {&progress, &retired}) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, semaphore));
  }

  // Both the captured request arrays and the temporary caller-output array
  // exceed an ordinary metadata block. Transfer descriptors do as well.
  constexpr size_t kBufferCount = 600;
  std::vector<iree_hal_pool_reservation_request_t> requests(
      kBufferCount, iree_hal_make_pool_reservation_request(64, 0));
  std::vector<iree_hal_buffer_t*> buffers(kBufferCount);
  uint64_t allocated_value = 1;
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      harness.queue, {}, {1, &progress, &allocated_value}, pool,
      requests.size(), requests.data(), buffers.data()));

  std::vector<uint32_t> expected(kBufferCount);
  std::vector<uint32_t> actual(kBufferCount);
  std::vector<iree_hal_transfer_operation_t> uploads(kBufferCount);
  std::vector<iree_hal_transfer_operation_t> downloads(kBufferCount);
  for (size_t i = 0; i < kBufferCount; ++i) {
    ASSERT_NE(buffers[i], nullptr);
    expected[i] = 0x12340000u + static_cast<uint32_t>(i);
    uploads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
    uploads[i].upload.source = &expected[i];
    uploads[i].upload.target_buffer = buffers[i];
    uploads[i].upload.length = sizeof(expected[i]);
    downloads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    downloads[i].download.source_buffer = buffers[i];
    downloads[i].download.target = &actual[i];
    downloads[i].download.length = sizeof(actual[i]);
  }
  uint64_t uploaded_value = 2;
  uint64_t downloaded_value = 3;
  IREE_ASSERT_OK(iree_hal_queue_transfer(
      harness.queue, {1, &progress, &allocated_value},
      {1, &progress, &uploaded_value}, uploads.size(), uploads.data(),
      /*barriers=*/nullptr));
  IREE_ASSERT_OK(iree_hal_queue_transfer(
      harness.queue, {1, &progress, &uploaded_value},
      {1, &progress, &downloaded_value}, downloads.size(), downloads.data(),
      /*barriers=*/nullptr));
  IREE_ASSERT_OK(iree_hal_queue_dealloca(
      harness.queue, {1, &progress, &downloaded_value},
      {1, &retired, &allocated_value}, buffers.size(), buffers.data()));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(retired, allocated_value));
  EXPECT_EQ(actual, expected);
  for (iree_hal_buffer_t* buffer : buffers) {
    iree_hal_buffer_release(buffer);
  }
  iree_hal_semaphore_release(retired);
  iree_hal_semaphore_release(progress);
  iree_hal_pool_release(pool);
}

TEST(XdnaQueueTest, AllocationFailureLeavesEveryOutputUntouched) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.SealDeviceGroup());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());

  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      harness.CreateSlabPool(IREE_HAL_BUFFER_USAGE_STORAGE_READ, &pool));
  std::array<iree_hal_pool_reservation_request_t, 2> requests = {};
  for (auto& request : requests) {
    request.allocation_size = 64;
  }
  // The first row reaches wrapper creation before validation rejects the
  // second row, exercising transaction rollback inside the driver.
  requests[1].min_alignment = 3;

  auto* const sentinel0 = reinterpret_cast<iree_hal_buffer_t*>(uintptr_t{1});
  auto* const sentinel1 = reinterpret_cast<iree_hal_buffer_t*>(uintptr_t{2});
  std::array<iree_hal_buffer_t*, 2> outputs = {sentinel0, sentinel1};
  uint64_t value = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_alloca(harness.queue, {}, {1, &harness.done, &value}, pool,
                            requests.size(), requests.data(), outputs.data()));
  EXPECT_EQ(outputs[0], sentinel0);
  EXPECT_EQ(outputs[1], sentinel1);

  uint64_t reached_value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(harness.done, &reached_value));
  EXPECT_EQ(reached_value, 0u);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  iree_hal_pool_release(pool);
}

TEST(XdnaQueueTest, QueueAllocationWaitsBeforeGrowingSourcePool) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.SealDeviceGroup());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());

  iree_hal_pool_t* source_pool = nullptr;
  IREE_ASSERT_OK(
      harness.CreateSlabPool(IREE_HAL_BUFFER_USAGE_STORAGE_READ, &source_pool));
  iree_hal_semaphore_t* gate = nullptr;
  iree_hal_semaphore_t* allocation_ready = nullptr;
  iree_hal_semaphore_t* deallocation_done = nullptr;
  for (auto** semaphore : {&gate, &allocation_ready, &deallocation_done}) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, semaphore));
  }

  iree_hal_pool_reservation_request_t request = {
      .allocation_size = 64,
  };
  iree_hal_buffer_t* buffer = nullptr;
  uint64_t value = 1;
  const size_t memory_create_count = harness.native.memory_create_count;
  IREE_ASSERT_OK(iree_hal_queue_alloca(harness.queue, {1, &gate, &value},
                                       {1, &allocation_ready, &value},
                                       source_pool, 1, &request, &buffer));

  // Unresolved waits are registered during submission. Neither a native
  // allocation nor a proactor admission hop is needed to leave this pending.
  EXPECT_EQ(harness.native.memory_create_count, memory_create_count);
  uint64_t reached_value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(allocation_ready, &reached_value));
  EXPECT_EQ(reached_value, 0u);

  // A real queue producer publishes the dependency. Its callback can run
  // inside admission; memory progress must not reacquire the queue state lock.
  IREE_ASSERT_OK(iree_hal_queue_barrier(harness.queue, {}, {1, &gate, &value},
                                        nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(allocation_ready, value));
  EXPECT_EQ(harness.native.memory_create_count, memory_create_count + 1);
  IREE_ASSERT_OK(
      iree_hal_queue_dealloca(harness.queue, {1, &allocation_ready, &value},
                              {1, &deallocation_done, &value}, 1, &buffer));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(deallocation_done, value));

  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(source_pool);
  iree_hal_semaphore_release(deallocation_done);
  iree_hal_semaphore_release(allocation_ready);
  iree_hal_semaphore_release(gate);
}

TEST(XdnaQueueTest, WarmQueueMemoryOperationsAllocateNoHostStorage) {
  constexpr uint64_t kWarmupIterations = 4;
  constexpr uint64_t kMeasuredIterations = 64;
  constexpr uint64_t kTotalIterations = kWarmupIterations + kMeasuredIterations;

  HostAllocationCounters allocation_counters;
  {
    QueueHarness harness;
    ASSERT_NO_FATAL_FAILURE(
        harness.Initialize(TrackingAllocator(&allocation_counters)));
    ASSERT_NO_FATAL_FAILURE(harness.SealDeviceGroup());
    ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());

    iree_hal_pool_t* pool = nullptr;
    IREE_ASSERT_OK(harness.CreateCachedTLSFPool(
        IREE_HAL_BUFFER_USAGE_STORAGE_READ, &pool));
    iree_hal_semaphore_t* anchor_ready = nullptr;
    iree_hal_semaphore_t* allocation_ready = nullptr;
    iree_hal_semaphore_t* deallocation_done = nullptr;
    iree_hal_semaphore_t* anchor_done = nullptr;
    for (auto** semaphore :
         {&anchor_ready, &allocation_ready, &deallocation_done, &anchor_done}) {
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_NONE, semaphore));
    }

    iree_hal_pool_reservation_request_t request = {
        .allocation_size = 64,
    };
    iree_hal_buffer_t* anchor_buffer = nullptr;
    uint64_t anchor_value = 1;
    IREE_ASSERT_OK(iree_hal_queue_alloca(harness.queue, {},
                                         {1, &anchor_ready, &anchor_value},
                                         pool, 1, &request, &anchor_buffer));
    ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(anchor_ready, anchor_value));

    HostAllocationSnapshot before = {};
    size_t native_allocations_before = 0;
    for (uint64_t iteration = 1; iteration <= kTotalIterations; ++iteration) {
      if (iteration == kWarmupIterations + 1) {
        before = SnapshotHostAllocations(allocation_counters);
        native_allocations_before = harness.native.memory_create_count;
      }
      iree_hal_buffer_t* buffer = nullptr;
      IREE_ASSERT_OK(iree_hal_queue_alloca(harness.queue, {},
                                           {1, &allocation_ready, &iteration},
                                           pool, 1, &request, &buffer));
      uint64_t reached_value = 0;
      IREE_ASSERT_OK(
          iree_hal_semaphore_query(allocation_ready, &reached_value));
      EXPECT_EQ(reached_value, iteration);
      IREE_ASSERT_OK(iree_hal_queue_dealloca(
          harness.queue, {1, &allocation_ready, &iteration},
          {1, &deallocation_done, &iteration}, 1, &buffer));
      IREE_ASSERT_OK(
          iree_hal_semaphore_query(deallocation_done, &reached_value));
      EXPECT_EQ(reached_value, iteration);
      iree_hal_buffer_release(buffer);
    }

    ExpectHostAllocationsUnchanged(before, allocation_counters);
    EXPECT_EQ(harness.native.memory_create_count, native_allocations_before);
    IREE_ASSERT_OK(iree_hal_queue_dealloca(
        harness.queue, {1, &anchor_ready, &anchor_value},
        {1, &anchor_done, &anchor_value}, 1, &anchor_buffer));
    ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(anchor_done, anchor_value));
    iree_hal_buffer_release(anchor_buffer);
    iree_hal_semaphore_release(anchor_done);
    iree_hal_semaphore_release(deallocation_done);
    iree_hal_semaphore_release(allocation_ready);
    iree_hal_semaphore_release(anchor_ready);
    iree_hal_pool_release(pool);
  }
  EXPECT_EQ(allocation_counters.live_allocations.load(), 0);
}

TEST(XdnaQueueMemoryTest, CachedWorkPassesPendingBackingOnSameQueue) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.SealDeviceGroup());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(
      harness.CreateSlabPool(IREE_HAL_BUFFER_USAGE_TRANSFER, &source));
  DeferredBackingPool* deferred = nullptr;
  IREE_ASSERT_OK(DeferredBackingPool::Create(source, &deferred));
  iree_hal_tlsf_pool_options_t options = {};
  options.tlsf_options.range_length = 64 * 1024;
  options.tlsf_options.alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
  options.tlsf_options.frontier_capacity = 64;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create(deferred, &options,
                                           iree_allocator_system(), &pool));

  iree_hal_semaphore_t* progress = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, 0, &progress));
  const auto small = iree_hal_make_pool_reservation_request(64, 0);
  const auto large = iree_hal_make_pool_reservation_request(128 * 1024, 0);
  iree_hal_buffer_t* anchor = nullptr;
  iree_hal_buffer_t* pending = nullptr;
  iree_hal_buffer_t* ready = nullptr;
  uint64_t anchor_value = 1;
  IREE_ASSERT_OK(iree_hal_queue_alloca(harness.queue, {},
                                       {1, &progress, &anchor_value}, pool, 1,
                                       &small, &anchor));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(progress, anchor_value));

  deferred->defer = true;
  uint64_t held_value = 1;
  IREE_ASSERT_OK(iree_hal_queue_alloca(harness.queue, {},
                                       {1, &harness.done, &held_value}, pool, 1,
                                       &large, &pending));
  ASSERT_EQ(deferred->pending.count, 1u);
  uint64_t allocated_value = 2;
  IREE_ASSERT_OK(iree_hal_queue_alloca(harness.queue, {},
                                       {1, &progress, &allocated_value}, pool,
                                       1, &small, &ready));
  uint64_t reached_value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(progress, &reached_value));
  EXPECT_EQ(reached_value, allocated_value);

  const std::array<uint32_t, 4> expected = {17, 31, 0xA55A1234, 0x87654321};
  std::array<uint32_t, 4> actual = {};
  iree_hal_transfer_operation_t upload = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
  };
  upload.upload.source = expected.data();
  upload.upload.target_buffer = ready;
  upload.upload.length = sizeof(expected);
  uint64_t uploaded_value = 3;
  IREE_ASSERT_OK(iree_hal_queue_transfer(
      harness.queue, {1, &progress, &allocated_value},
      {1, &progress, &uploaded_value}, 1, &upload, nullptr));
  iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
  };
  download.download.source_buffer = ready;
  download.download.target = actual.data();
  download.download.length = sizeof(actual);
  uint64_t downloaded_value = 4;
  IREE_ASSERT_OK(iree_hal_queue_transfer(
      harness.queue, {1, &progress, &uploaded_value},
      {1, &progress, &downloaded_value}, 1, &download, nullptr));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(progress, downloaded_value));
  EXPECT_EQ(actual, expected);
  IREE_ASSERT_OK(iree_hal_semaphore_query(harness.done, &reached_value));
  EXPECT_EQ(reached_value, 0u);

  std::thread completion([&] { deferred->CompleteOne(); });
  completion.join();
  // Backing now performs an asynchronous native allocation. Completion still
  // needs no proactor polling or host-service handoff to publish readiness.
  IREE_ASSERT_OK(iree_hal_semaphore_wait(harness.done, held_value,
                                         iree_infinite_timeout(), 0));
  IREE_ASSERT_OK(iree_hal_semaphore_query(harness.done, &reached_value));
  EXPECT_EQ(reached_value, held_value);
  const std::array<iree_hal_buffer_t*, 3> buffers = {anchor, pending, ready};
  uint64_t released_value = 5;
  IREE_ASSERT_OK(iree_hal_queue_dealloca(harness.queue, {},
                                         {1, &progress, &released_value},
                                         buffers.size(), buffers.data()));
  IREE_ASSERT_OK(iree_hal_semaphore_query(progress, &reached_value));
  EXPECT_EQ(reached_value, released_value);
  for (auto* buffer : buffers) {
    iree_hal_buffer_release(buffer);
  }
  iree_hal_semaphore_release(progress);
  iree_hal_pool_release(pool);
  iree_hal_pool_release(deferred);
  iree_hal_pool_release(source);
}

TEST(XdnaQueueMemoryTest, DeferredBackingFailureReachesSignal) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.SealDeviceGroup());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(
      harness.CreateSlabPool(IREE_HAL_BUFFER_USAGE_TRANSFER, &source));
  DeferredBackingPool* deferred = nullptr;
  IREE_ASSERT_OK(DeferredBackingPool::Create(source, &deferred));
  deferred->defer = true;
  const auto request = iree_hal_make_pool_reservation_request(64, 0);
  uint64_t value = 1;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_queue_alloca(harness.queue, {},
                                       {1, &harness.done, &value}, deferred, 1,
                                       &request, &buffer));
  ASSERT_EQ(deferred->pending.count, 1u);
  std::thread completion([&] { deferred->FailOne(); });
  completion.join();
  uint64_t reached_value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_semaphore_query(harness.done, &reached_value));
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(deferred);
  iree_hal_pool_release(source);
}

TEST(XdnaQueueMemoryTest, CapacityReturnedBeforeDeferredResultIsObserved) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.SealDeviceGroup());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(
      harness.CreateSlabPool(IREE_HAL_BUFFER_USAGE_TRANSFER, &source));
  iree_hal_fixed_block_pool_options_t options;
  iree_hal_fixed_block_pool_options_initialize(&options);
  options.block_size = 64;
  options.blocks_per_slab = 1;
  options.budget_limit = 64;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create(
      source, &options, iree_allocator_system(), &pool));
  DeferredBackingPool* deferred = nullptr;
  IREE_ASSERT_OK(DeferredBackingPool::Create(pool, &deferred));
  deferred->defer = true;

  const auto request = iree_hal_make_pool_reservation_request(64, 0);
  iree_hal_buffer_t* initial = nullptr;
  iree_hal_buffer_t* pending = nullptr;
  uint64_t value = 1;
  iree_hal_semaphore_t* initial_ready = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(harness.device,
                                           IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                                           0, 0, &initial_ready));
  IREE_ASSERT_OK(iree_hal_queue_alloca(harness.queue, {},
                                       {1, &initial_ready, &value}, pool, 1,
                                       &request, &initial));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(initial_ready, value));
  IREE_ASSERT_OK(iree_hal_queue_alloca(harness.queue, {},
                                       {1, &harness.done, &value}, deferred, 1,
                                       &request, &pending));
  ASSERT_EQ(deferred->pending.count, 1u);
  auto call = deferred->pending;
  deferred->pending = {};
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
      pool, call.count, call.requests, call.frontier, call.reservations,
      call.infos, &result, call.flags));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OVER_BUDGET);

  // Native acquisition already produced an unavailable result, but its
  // callback has not published that result yet. Capacity returns in between.
  IREE_ASSERT_OK(iree_hal_queue_dealloca(harness.queue, {}, {}, 1, &initial));
  iree_hal_buffer_release(initial);
  deferred->defer = false;
  call.completion.fn(call.completion.user_data, iree_ok_status(), result);
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(harness.done, value));
  IREE_ASSERT_OK(iree_hal_queue_dealloca(harness.queue, {}, {}, 1, &pending));
  iree_hal_buffer_release(pending);
  iree_hal_semaphore_release(initial_ready);
  iree_hal_pool_release(deferred);
  iree_hal_pool_release(pool);
  iree_hal_pool_release(source);
}

TEST(XdnaQueueMemoryTest, SharedPrerequisiteDoesNotOverflowLargeBatch) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.SealDeviceGroup());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(
      harness.CreateSlabPool(IREE_HAL_BUFFER_USAGE_TRANSFER, &source));
  constexpr size_t kCount = 300;
  constexpr size_t kBlockSize = 64;
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, iree_hal_make_pool_reservation_request(kCount * kBlockSize, 0),
      iree_infinite_timeout(), &backing));
  iree_hal_fixed_block_pool_options_t options;
  iree_hal_fixed_block_pool_options_initialize(&options);
  options.block_size = kBlockSize;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
      backing, 0, kCount * kBlockSize, &options, iree_allocator_system(),
      &pool));
  std::array<iree_hal_pool_reservation_request_t, kCount> requests;
  requests.fill(iree_hal_make_pool_reservation_request(kBlockSize, 0));
  std::array<iree_hal_pool_reservation_t, kCount> reservations = {};
  std::array<iree_hal_pool_acquire_info_t, kCount> infos = {};
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
      pool, kCount, requests.data(), nullptr, reservations.data(), infos.data(),
      &result, IREE_HAL_POOL_RESERVE_FLAG_NONE));
  ASSERT_NE(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
  const auto axis = iree_async_axis_make_queue(1, 2, 3, 4, 5);
  IREE_ASSERT_OK(iree_async_frontier_tracker_register_axis(
      pool->frontier_tracker, axis, nullptr));
  iree_async_single_frontier_t frontier;
  iree_async_single_frontier_initialize(&frontier, axis, 1);
  iree_hal_pool_release_reservations(
      pool, kCount, reservations.data(),
      iree_async_single_frontier_as_const_frontier(&frontier));

  std::array<iree_hal_buffer_t*, kCount> buffers = {};
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_alloca(harness.queue, {},
                                       {1, &harness.done, &value}, pool, kCount,
                                       requests.data(), buffers.data()));
  uint64_t reached_value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(harness.done, &reached_value));
  EXPECT_EQ(reached_value, 0u);
  // Three hundred rows depend on one axis, not three hundred distinct axes.
  std::thread completion([&] {
    iree_async_frontier_tracker_advance(pool->frontier_tracker, axis, 1);
  });
  completion.join();
  IREE_ASSERT_OK(iree_hal_semaphore_query(harness.done, &reached_value));
  EXPECT_EQ(reached_value, value);
  IREE_ASSERT_OK(iree_hal_queue_dealloca(harness.queue, {}, {}, buffers.size(),
                                         buffers.data()));
  for (auto* buffer : buffers) {
    iree_hal_buffer_release(buffer);
  }
  iree_hal_pool_release(pool);
  iree_hal_buffer_release(backing);
  iree_hal_pool_release(source);
}

TEST(XdnaQueueMemoryTest, NativeGrowthLeavesCachedMemoryAndExecutionReady) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.SealDeviceGroup());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(harness.CreateCachedTLSFPool(
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  iree_hal_executable_t* executable = nullptr;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(harness.LoadExecutable(
      iree::hal::amd::xdna::testing::ImageFixture(), &executable, &function));
  iree_hal_semaphore_t* progress = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, 0, &progress));
  const auto small = iree_hal_make_pool_reservation_request(64, 0);
  const auto large = iree_hal_make_pool_reservation_request(128 * 1024, 0);
  iree_hal_buffer_t* anchor = nullptr;
  iree_hal_buffer_t* growing = nullptr;
  iree_hal_buffer_t* cached = nullptr;
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      harness.queue, {}, {1, &progress, &value}, pool, 1, &small, &anchor));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(progress, value));

  harness.native.allocation_gate.Arm(large.allocation_size);
  ScopedMemoryGate gate{&harness.native.allocation_gate};
  IREE_ASSERT_OK(iree_hal_queue_alloca(harness.queue, {},
                                       {1, &harness.done, &value}, pool, 1,
                                       &large, &growing));
  harness.native.allocation_gate.AwaitEntry();
  uint64_t allocated = 2;
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      harness.queue, {}, {1, &progress, &allocated}, pool, 1, &small, &cached));
  uint64_t observed = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(progress, &observed));
  EXPECT_EQ(observed, allocated);
  const std::array<uint32_t, 4> expected = {19, 23, 0xABCD1234, 0x98765432};
  std::array<uint32_t, 4> actual = {};
  iree_hal_transfer_operation_t upload = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
  };
  upload.upload.source = expected.data();
  upload.upload.target_buffer = cached;
  upload.upload.length = sizeof(expected);
  uint64_t uploaded = 3;
  IREE_ASSERT_OK(
      iree_hal_queue_transfer(harness.queue, {1, &progress, &allocated},
                              {1, &progress, &uploaded}, 1, &upload, nullptr));
  const auto binding = iree_hal_make_buffer_ref(cached, 0, 64);
  uint64_t dispatched = 4;
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      harness.queue, {1, &progress, &uploaded}, {1, &progress, &dispatched},
      executable, function, iree_hal_make_static_dispatch_config(1, 1, 1), {},
      {1, &binding}, nullptr, 0));
  iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
  };
  download.download.source_buffer = cached;
  download.download.target = actual.data();
  download.download.length = sizeof(actual);
  uint64_t downloaded = 5;
  IREE_ASSERT_OK(iree_hal_queue_transfer(
      harness.queue, {1, &progress, &dispatched}, {1, &progress, &downloaded},
      1, &download, nullptr));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(progress, downloaded));
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(harness.native.submission_count, 1u);
  IREE_ASSERT_OK(iree_hal_semaphore_query(harness.done, &observed));
  EXPECT_EQ(observed, 0u);

  gate.gate->Release();
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(harness.done, value, iree_infinite_timeout(), 0));
  const std::array<iree_hal_buffer_t*, 3> buffers = {anchor, growing, cached};
  IREE_ASSERT_OK(iree_hal_queue_dealloca(harness.queue, {}, {}, buffers.size(),
                                         buffers.data()));
  for (auto* buffer : buffers) {
    iree_hal_buffer_release(buffer);
  }
  iree_hal_semaphore_release(progress);
  iree_hal_executable_release(executable);
  iree_hal_pool_release(pool);
}

TEST(XdnaQueueMemoryTest, PendingNativeFreeRetainsDeviceAndFabric) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.SealDeviceGroup());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(harness.CreateSlabPool(IREE_HAL_BUFFER_USAGE_TRANSFER, &pool));
  // Match the native profile's allocation granularity so the gate observes
  // exactly this allocation's destruction.
  constexpr iree_device_size_t kAllocationSize = 4096;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(kAllocationSize, 0),
      iree_infinite_timeout(), &buffer));
  harness.native.release_gate.Arm(kAllocationSize);
  ScopedMemoryGate gate{&harness.native.release_gate};
  iree_hal_buffer_release(buffer);
  harness.native.release_gate.AwaitEntry();
  iree_hal_pool_release(pool);
  harness.ReleaseDevice();
  EXPECT_EQ(harness.native.context_destroy_count, 0u);
  EXPECT_EQ(harness.native.instance_destroy_count, 0u);
  gate.gate->Release();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDeviceDestroyed());
  EXPECT_EQ(harness.native.instance_destroy_count, 1u);
  EXPECT_EQ(harness.native.live_memories, 0u);
}

}  // namespace
