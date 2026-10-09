// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/cts/util/pool_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/device_group.h"
#include "iree/hal/drivers/task/device.h"
#include "iree/hal/drivers/task/queue/queue.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/maintenance.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

#if defined(IREE_PLATFORM_WINDOWS)
#include "iree/async/platform/iocp/api.h"
#else
#include "iree/async/platform/posix/api.h"
#endif

namespace {

enum class PoolKind { kFixedBlock, kTlsf };

class TaskQueueAllocaTest : public ::testing::TestWithParam<iree_host_size_t> {
 protected:
  virtual iree_host_size_t WorkerCount() const { return 2; }

  void SetUp() override {
    iree_task_topology_t topology;
    iree_task_topology_initialize_from_group_count(WorkerCount(), &topology);
    iree_task_executor_options_t executor_options;
    iree_task_executor_options_initialize(&executor_options);
    iree_status_t status = iree_task_executor_create(
        executor_options, &topology, iree_allocator_system(), &executor_);
    iree_task_topology_deinitialize(&topology);
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(
        IREE_SV("shared_pool"), iree_allocator_system(),
        iree_allocator_system(), &allocator_));

    auto progress_options = iree_async_proactor_pool_options_default();
    // The test owns polling so it can observe both waits registered on the
    // notification owner without racing the backend's intrusive wait list.
    progress_options.runner = {};
#if defined(IREE_PLATFORM_WINDOWS)
    progress_options.proactor_create = iree_async_proactor_create_iocp;
#else
    progress_options.proactor_create = iree_async_proactor_create_posix;
#endif
    const iree_string_view_t identifiers[] = {IREE_SV("consumer_a"),
                                              IREE_SV("consumer_b")};
    for (iree_host_size_t i = 0; i < devices_.size(); ++i) {
      IREE_ASSERT_OK(iree_async_proactor_pool_create(
          1, /*node_ids=*/nullptr, progress_options, iree_allocator_system(),
          &progress_[i]));
      iree_hal_task_device_params_t device_params;
      iree_hal_task_device_params_initialize(&device_params);
      auto create_params = iree_hal_device_create_params_default();
      create_params.proactor_pool = progress_[i];
      IREE_ASSERT_OK(iree_hal_task_device_create(
          identifiers[i], &device_params, 1, &executor_, 0, nullptr, allocator_,
          &create_params, iree_allocator_system(), &devices_[i]));
      queues_[i] = iree_hal_device_queue(devices_[i], 0, 0);
    }
    ASSERT_NE(reinterpret_cast<iree_hal_task_queue_t*>(queues_[0])->proactor,
              reinterpret_cast<iree_hal_task_queue_t*>(queues_[1])->proactor);

    iree_async_frontier_tracker_t* tracker = nullptr;
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker));
    iree_hal_device_group_builder_t builder;
    iree_hal_device_group_builder_initialize(&builder, tracker);
    iree_async_frontier_tracker_release(tracker);
    for (iree_host_size_t i = 0;
         i < devices_.size() && iree_status_is_ok(status); ++i) {
      status = iree_hal_device_group_builder_add_device(&builder, devices_[i]);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_builder_finalize(
          &builder, iree_allocator_system(), &group_);
    }
    iree_hal_device_group_builder_deinitialize(&builder);
    IREE_ASSERT_OK(status);

    iree_hal_queue_pool_backend_t backend = {};
    IREE_ASSERT_OK(iree_hal_device_query_queue_pool_backend(
        devices_[GetParam()], iree_hal_queue_family(queues_[GetParam()]),
        &backend));
    IREE_ASSERT_OK(CreatePool(backend, &pool_));
    notification_ = iree_hal_pool_notification(pool_);
    for (iree_host_size_t i = 0; i < semaphores_.size(); ++i) {
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          devices_[i == 0 ? 0 : i - 1], IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphores_[i]));
    }
  }

  iree_status_t CreatePool(const iree_hal_queue_pool_backend_t& backend,
                           iree_hal_pool_t** out_pool) {
    iree_hal_fixed_block_pool_options_t options = {.block_size = kBlockSize,
                                                   .blocks_per_slab = 2,
                                                   .frontier_capacity = 2,
                                                   .asan = backend.asan};
    return iree::hal::cts::CreateFiniteBlockPool(
        backend, options, iree_allocator_system(), &backing_pool_, out_pool);
  }

  void TearDown() override {
    for (auto* buffer : initial_buffers_) {
      iree_hal_buffer_release(buffer);
    }
    for (auto* buffer : pending_buffers_) {
      iree_hal_buffer_release(buffer);
    }
    for (auto* semaphore : semaphores_) {
      iree_hal_semaphore_release(semaphore);
    }
    iree_hal_pool_release(pool_);
    iree_hal_pool_release(backing_pool_);
    iree_hal_device_group_release(group_);
    for (auto* device : devices_) {
      iree_hal_device_release(device);
    }
    for (auto* progress : progress_) {
      iree_async_proactor_pool_release(progress);
    }
    iree_hal_allocator_release(allocator_);
    iree_task_executor_release(executor_);
  }

  void PollOwner() {
    iree_status_t status = iree_async_proactor_poll(
        notification_->proactor, iree_immediate_timeout(), nullptr);
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_free(status);
    } else {
      IREE_ASSERT_OK(status);
    }
    std::this_thread::yield();
  }

  void Wait(iree_hal_semaphore_t* semaphore, uint64_t value) {
    uint64_t current_value = 0;
    IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &current_value));
    while (current_value < value) {
      ASSERT_NO_FATAL_FAILURE(PollOwner());
      IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &current_value));
    }
  }

  iree_host_size_t RegisteredWaitCount(
      iree_async_notification_t* notification) {
    // Only this thread polls the owner, so its wait list cannot change while
    // inspected. The other queue's proactor is never polled.
#if defined(IREE_PLATFORM_WINDOWS)
    auto* wait = notification->platform.iocp.pending_waits;
#else
    auto* wait = notification->platform.posix.pending_waits;
#endif
    iree_host_size_t count = 0;
    for (; wait;
         wait = reinterpret_cast<iree_async_notification_wait_operation_t*>(
             wait->base.next)) {
      ++count;
    }
    return count;
  }

  void CheckFailedBatchParks(PoolKind kind) {
    if (kind == PoolKind::kTlsf) {
      iree_hal_queue_pool_backend_t backend = {};
      IREE_ASSERT_OK(iree_hal_device_query_queue_pool_backend(
          devices_[GetParam()], iree_hal_queue_family(queues_[GetParam()]),
          &backend));
      iree_hal_tlsf_pool_options_t options = {};
      options.tlsf_options.range_length = 2 * kBlockSize;
      options.tlsf_options.alignment = 16;
      options.tlsf_options.frontier_capacity = 2;
      options.budget_limit = 2 * kBlockSize;
      iree_hal_pool_release(pool_);
      pool_ = nullptr;
      iree_hal_passthrough_pool_options_t backing_options = {
          .epoch_query = backend.epoch_query};
      iree_hal_pool_t* backing_pool = nullptr;
      IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
          backing_options, backend.slab_provider, backend.notification,
          backend.frontier_tracker, backend.maintenance,
          iree_allocator_system(), &backing_pool));
      iree_status_t status = iree_hal_tlsf_pool_create(
          backing_pool, &options, iree_allocator_system(), &pool_);
      iree_hal_pool_release(backing_pool);
      IREE_ASSERT_OK(status);
      notification_ = iree_hal_pool_notification(pool_);
    }
    std::array<iree_hal_pool_reservation_request_t, 2> requests = {};
    for (auto& request : requests) {
      request.allocation_size = kBlockSize;
      request.params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
      request.params.access = IREE_HAL_MEMORY_ACCESS_ALL;
      request.params.usage =
          IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
      request.params.queue_family_affinity =
          iree_hal_make_queue_family_affinity(0);
    }
    uint64_t allocated_value = 1;
    uint64_t filled_value = 2;
    uint64_t released_value = 3;
    const auto no_waits = iree_hal_semaphore_list_empty();
    iree_hal_semaphore_list_t initial_allocated = {1, &semaphores_[0],
                                                   &allocated_value};
    iree_hal_semaphore_list_t initial_released = {1, &semaphores_[0],
                                                  &released_value};
    IREE_ASSERT_OK(
        iree_hal_queue_alloca(queues_[0], no_waits, initial_allocated, pool_, 1,
                              requests.data(), initial_buffers_.data()));
    ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[0], allocated_value));

    const uint32_t epoch = iree_async_notification_query_epoch(notification_);
    iree_hal_semaphore_list_t allocated = {1, &semaphores_[1],
                                           &allocated_value};
    IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[1], no_waits, allocated, pool_,
                                         requests.size(), requests.data(),
                                         pending_buffers_.data()));
    while (RegisteredWaitCount(notification_) != 1) {
      ASSERT_NO_FATAL_FAILURE(PollOwner());
    }
    EXPECT_EQ(iree_async_notification_query_epoch(notification_), epoch);
    iree_async_proactor_wake(notification_->proactor);
    ASSERT_NO_FATAL_FAILURE(PollOwner());
    EXPECT_EQ(RegisteredWaitCount(notification_), 1u);
    EXPECT_EQ(iree_async_notification_query_epoch(notification_), epoch);

    IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[0], initial_allocated,
                                           initial_released, 1,
                                           initial_buffers_.data()));
    ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[0], released_value));
    ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[1], allocated_value));
    for (iree_host_size_t i = 0; i < pending_buffers_.size(); ++i) {
      iree_hal_semaphore_list_t filled = {1, &semaphores_[i + 1],
                                          &filled_value};
      const uint32_t pattern = 0xCAFE1000u + i;
      IREE_ASSERT_OK(iree_hal_queue_fill(
          queues_[1], allocated, filled, pending_buffers_[i], 0, kBlockSize,
          &pattern, sizeof(pattern), /*barriers=*/NULL,
          IREE_HAL_FILL_FLAG_NONE));
      ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[i + 1], filled_value));
      std::array<uint32_t, kBlockSize / sizeof(uint32_t)> actual;
      IREE_ASSERT_OK(iree_hal_buffer_map_read(pending_buffers_[i], 0,
                                              actual.data(), sizeof(actual)));
      for (uint32_t value : actual) {
        EXPECT_EQ(value, pattern);
      }
    }
    uint64_t filled_values[2] = {filled_value, filled_value};
    iree_hal_semaphore_list_t all_filled = {2, &semaphores_[1], filled_values};
    iree_hal_semaphore_list_t released = {1, &semaphores_[1], &released_value};
    IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[1], all_filled, released,
                                           pending_buffers_.size(),
                                           pending_buffers_.data()));
    ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[1], released_value));
  }

  static constexpr iree_device_size_t kBlockSize = 512;
  // Workers shared by the two independent Task devices.
  iree_task_executor_t* executor_ = nullptr;
  // Heap allocator retained by both devices.
  iree_hal_allocator_t* allocator_ = nullptr;
  // Distinct caller-driven progress services retained by each device.
  std::array<iree_async_proactor_pool_t*, 2> progress_ = {};
  // Devices retained independently and by the sealed group.
  std::array<iree_hal_device_t*, 2> devices_ = {};
  // Group establishing the devices' shared frontier coordinates.
  iree_hal_device_group_t* group_ = nullptr;
  // Provisioned queues borrowed from the devices.
  std::array<iree_hal_queue_t*, 2> queues_ = {};
  // Native owner retained until the finite arena and its buffers are gone.
  iree_hal_pool_t* backing_pool_ = nullptr;
  // Finite memory pool shared by both consumers.
  iree_hal_pool_t* pool_ = nullptr;
  // Local capacity notification borrowed from the active pool.
  iree_async_notification_t* notification_ = nullptr;
  // Initial allocation and the two consumers' completion timelines.
  std::array<iree_hal_semaphore_t*, 3> semaphores_ = {};
  // Buffers occupying every block until explicit deallocation.
  std::array<iree_hal_buffer_t*, 2> initial_buffers_ = {};
  // Buffers materialized after the owner wakes both consumers.
  std::array<iree_hal_buffer_t*, 2> pending_buffers_ = {};
};

TEST_P(TaskQueueAllocaTest, FixedBlockFailedBatchParksUntilExternalRelease) {
  ASSERT_NO_FATAL_FAILURE(CheckFailedBatchParks(PoolKind::kFixedBlock));
}

TEST_P(TaskQueueAllocaTest, TlsfFailedBatchParksUntilExternalRelease) {
  ASSERT_NO_FATAL_FAILURE(CheckFailedBatchParks(PoolKind::kTlsf));
}

TEST_P(TaskQueueAllocaTest, SharedPoolResumesThroughNotificationOwner) {
  std::array<iree_hal_pool_reservation_request_t, 2> requests = {};
  for (auto& request : requests) {
    request.allocation_size = kBlockSize;
    request.params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
    request.params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    request.params.usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
    request.params.queue_family_affinity =
        iree_hal_make_queue_family_affinity(0);
  }
  uint64_t allocated_value = 1;
  uint64_t filled_value = 2;
  uint64_t released_value = 3;
  const auto no_waits = iree_hal_semaphore_list_empty();
  iree_hal_semaphore_list_t initial_allocated = {1, &semaphores_[0],
                                                 &allocated_value};
  IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], no_waits, initial_allocated,
                                       pool_, requests.size(), requests.data(),
                                       initial_buffers_.data()));
  ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[0], allocated_value));

  for (iree_host_size_t i = 0; i < queues_.size(); ++i) {
    iree_hal_semaphore_list_t allocated = {1, &semaphores_[i + 1],
                                           &allocated_value};
    IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[i], no_waits, allocated, pool_,
                                         1, &requests[i],
                                         &pending_buffers_[i]));
  }
  // Explicit registration proves both queues reached exhaustion before any
  // block is returned; a submission-order or sleep-based check would not.
  while (RegisteredWaitCount(notification_) != queues_.size()) {
    ASSERT_NO_FATAL_FAILURE(PollOwner());
  }

  iree_hal_semaphore_list_t initial_released = {1, &semaphores_[0],
                                                &released_value};
  IREE_ASSERT_OK(iree_hal_queue_dealloca(
      queues_[0], initial_allocated, initial_released, initial_buffers_.size(),
      initial_buffers_.data()));
  ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[0], released_value));
  for (iree_host_size_t i = 0; i < queues_.size(); ++i) {
    ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[i + 1], allocated_value));
  }

  for (iree_host_size_t i = 0; i < queues_.size(); ++i) {
    iree_hal_semaphore_list_t allocated = {1, &semaphores_[i + 1],
                                           &allocated_value};
    iree_hal_semaphore_list_t filled = {1, &semaphores_[i + 1], &filled_value};
    iree_hal_semaphore_list_t released = {1, &semaphores_[i + 1],
                                          &released_value};
    const uint32_t pattern = 0xBADC0000u + i;
    IREE_ASSERT_OK(iree_hal_queue_fill(
        queues_[i], allocated, filled, pending_buffers_[i], 0, kBlockSize,
        &pattern, sizeof(pattern), /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
    ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[i + 1], filled_value));
    iree_hal_buffer_mapping_t mapping = {};
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        pending_buffers_[i], IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_BUFFER_MAP_FLAG_NONE, 0,
        kBlockSize, &mapping));
    const auto* output =
        reinterpret_cast<const uint32_t*>(mapping.contents.data);
    for (iree_host_size_t word = 0; word < kBlockSize / sizeof(pattern);
         ++word) {
      EXPECT_EQ(output[word], pattern);
    }
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
    IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[i], filled, released, 1,
                                           &pending_buffers_[i]));
    ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[i + 1], released_value));
  }
}

TEST_P(TaskQueueAllocaTest, SiblingPoolResumesThroughBackingNotificationOwner) {
  using iree::hal::cts::Ref;
  iree_hal_queue_pool_backend_t backend = {};
  IREE_ASSERT_OK(iree_hal_device_query_queue_pool_backend(
      devices_[GetParam()], iree_hal_queue_family(queues_[GetParam()]),
      &backend));
  Ref<iree_hal_pool_t> native;
  Ref<iree_hal_pool_t> source;
  iree_hal_fixed_block_pool_options_t source_options = {.block_size = 65536,
                                                        .blocks_per_slab = 1,
                                                        .frontier_capacity = 2,
                                                        .asan = backend.asan};
  IREE_ASSERT_OK(iree::hal::cts::CreateFiniteBlockPool(
      backend, source_options, iree_allocator_system(), native.out(),
      source.out()));
  iree_hal_slab_cache_options_t cache_options;
  iree_hal_slab_cache_options_initialize(&cache_options);
  cache_options.slab.allocation_size = 65536;
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(source, &capabilities);
  cache_options.slab.params.min_alignment =
      iree_min(4096, capabilities.max_allocation_alignment);
  cache_options.slab.params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  cache_options.slab.params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  cache_options.slab.params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  cache_options.slab.params.queue_family_affinity =
      iree_hal_make_queue_family_affinity(0);
  Ref<iree_hal_pool_t> cache;
  IREE_ASSERT_OK(iree_hal_slab_cache_create(
      source, &cache_options, iree_allocator_system(), cache.out()));
  Ref<iree_hal_pool_t> producer;
  iree_hal_tlsf_pool_options_t producer_options = {};
  producer_options.tlsf_options.range_length = 4096;
  producer_options.tlsf_options.frontier_capacity = 2;
  producer_options.asan = backend.asan;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create(
      cache, &producer_options, iree_allocator_system(), producer.out()));
  Ref<iree_hal_pool_t> consumer;
  iree_hal_fixed_block_pool_options_t consumer_options = {
      .block_size = kBlockSize,
      .blocks_per_slab = 2,
      .frontier_capacity = 2,
      .asan = backend.asan};
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create(
      cache, &consumer_options, iree_allocator_system(), consumer.out()));
  auto* consumer_notification = iree_hal_pool_notification(consumer);
  auto* cache_notification = iree_hal_pool_notification(cache);
  ASSERT_NE(consumer_notification, cache_notification);
  iree_hal_pool_reservation_request_t request = {cache_options.slab.params,
                                                 kBlockSize};
  request.params.min_alignment = 16;
  uint64_t allocated_value = 1;
  uint64_t filled_value = 2;
  uint64_t released_value = 3;
  const auto no_waits = iree_hal_semaphore_list_empty();
  iree_hal_semaphore_list_t first_allocated = {1, &semaphores_[0],
                                               &allocated_value};
  iree_hal_semaphore_list_t first_released = {1, &semaphores_[0],
                                              &released_value};
  Ref<iree_hal_buffer_t> first;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], no_waits, first_allocated,
                                       producer, 1, &request, first.out()));
  ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[0], allocated_value));
  const auto* backing = iree_hal_buffer_memory_view(first).backing;
  iree_hal_semaphore_list_t allocated = {1, &semaphores_[1], &allocated_value};
  iree_hal_semaphore_list_t filled = {1, &semaphores_[1], &filled_value};
  iree_hal_semaphore_list_t released = {1, &semaphores_[1], &released_value};
  Ref<iree_hal_buffer_t> second;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[1], no_waits, allocated,
                                       consumer, 1, &request, second.out()));
  // Poll only the captured memory owner, never the other queue's proactor.
  // Both independent source waits must be registered before capacity returns.
  while (RegisteredWaitCount(consumer_notification) != 1 ||
         RegisteredWaitCount(cache_notification) != 1) {
    ASSERT_NO_FATAL_FAILURE(PollOwner());
    uint64_t value = 0;
    IREE_ASSERT_OK(iree_hal_semaphore_query(semaphores_[1], &value));
    ASSERT_EQ(value, 0u);
  }
  iree_hal_buffer_t* first_buffer = first;
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[0], first_allocated,
                                         first_released, 1, &first_buffer));
  ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[0], released_value));
  first.reset();
  ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[1], allocated_value));
  EXPECT_EQ(iree_hal_buffer_memory_view(second).backing, backing);
  const uint32_t pattern = 0x1234CAFEu;
  IREE_ASSERT_OK(iree_hal_queue_fill(
      queues_[1], allocated, filled, second, 0, kBlockSize, &pattern,
      sizeof(pattern), /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
  ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[1], filled_value));
  std::array<uint32_t, kBlockSize / sizeof(uint32_t)> actual;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(second, 0, actual.data(), sizeof(actual)));
  for (uint32_t value : actual) {
    EXPECT_EQ(value, pattern);
  }
  iree_hal_buffer_t* second_buffer = second;
  IREE_ASSERT_OK(
      iree_hal_queue_dealloca(queues_[1], filled, released, 1, &second_buffer));
  ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[1], released_value));
  second.reset();
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(native, &stats);
  EXPECT_EQ(stats.reserve_count, 1u);
  EXPECT_EQ(RegisteredWaitCount(consumer_notification), 0u);
  EXPECT_EQ(RegisteredWaitCount(cache_notification), 0u);
}

TEST_P(TaskQueueAllocaTest, CompletedDeallocationIsImmediatelyReusable) {
  std::array<iree_hal_pool_reservation_request_t, 2> requests = {};
  for (auto& request : requests) {
    request.allocation_size = kBlockSize;
    request.params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
    request.params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    request.params.usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
    request.params.queue_family_affinity =
        iree_hal_make_queue_family_affinity(0);
  }
  uint64_t allocated_value = 1;
  uint64_t released_value = 2;
  const auto no_waits = iree_hal_semaphore_list_empty();
  iree_hal_semaphore_list_t allocated = {1, &semaphores_[0], &allocated_value};
  iree_hal_semaphore_list_t released = {1, &semaphores_[0], &released_value};
  IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], no_waits, allocated, pool_,
                                       requests.size(), requests.data(),
                                       initial_buffers_.data()));
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[0], allocated, released,
                                         initial_buffers_.size(),
                                         initial_buffers_.data()));
  ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[0], released_value));

  // Completion publishes empty prerequisites, independent of the optional
  // completion query inherited by the prepared range.
  std::array<iree_hal_pool_reservation_t, 2> reservations;
  std::array<iree_hal_pool_acquire_info_t, 2> infos;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool_, requests.size(), requests.data(), nullptr,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, reservations.data(), infos.data(),
      &result));
  ASSERT_EQ(IREE_HAL_POOL_ACQUIRE_OK_FRESH, result);
  for (const auto& info : infos) {
    EXPECT_EQ(info.reuse_frontier, nullptr);
  }
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      pool_, requests.size(), requests.data(), reservations.data(),
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP,
      pending_buffers_.data()));
  uint64_t filled_value = 0;
  for (auto*& buffer : pending_buffers_) {
    const uint32_t pattern = 0x1234ABCDu;
    ++filled_value;
    iree_hal_semaphore_list_t filled = {1, &semaphores_[1], &filled_value};
    IREE_ASSERT_OK(iree_hal_queue_fill(
        queues_[1], no_waits, filled, buffer, 0, kBlockSize, &pattern,
        sizeof(pattern), /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
    ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[1], filled_value));
    iree_hal_buffer_mapping_t mapping = {};
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
        IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, kBlockSize, &mapping));
    const auto* output =
        reinterpret_cast<const uint32_t*>(mapping.contents.data);
    for (iree_host_size_t word = 0; word < kBlockSize / sizeof(pattern);
         ++word) {
      EXPECT_EQ(pattern, output[word]);
    }
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  }
}

INSTANTIATE_TEST_SUITE_P(NotificationOwners, TaskQueueAllocaTest,
                         ::testing::Values(0, 1));

// Observes the real CPU provider's next native allocation and final free.
// Provider metadata and buffer wrappers continue using the same allocator;
// only the explicitly captured allocation participates in the observation.
class NativeBackingObserver {
 public:
  iree_allocator_t allocator() { return {this, Control}; }

  void ObserveNextAllocation() {
    std::lock_guard<std::mutex> lock(mutex_);
    observe_next_allocation_ = true;
  }

  void AwaitRelease() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return released_; });
  }

  std::thread::id release_thread() {
    std::lock_guard<std::mutex> lock(mutex_);
    return release_thread_;
  }

  bool released() {
    std::lock_guard<std::mutex> lock(mutex_);
    return released_;
  }

 private:
  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_pointer) {
    auto* observer = static_cast<NativeBackingObserver*>(self);
    bool observed_release = false;
    if (command == IREE_ALLOCATOR_COMMAND_FREE) {
      std::lock_guard<std::mutex> lock(observer->mutex_);
      observed_release = *inout_pointer == observer->allocation_;
      if (observed_release) {
        observer->allocation_ = nullptr;
      }
    }
    iree_allocator_t system_allocator = iree_allocator_system();
    iree_status_t status = system_allocator.ctl(system_allocator.self, command,
                                                params, inout_pointer);
    if (iree_status_is_ok(status)) {
      std::lock_guard<std::mutex> lock(observer->mutex_);
      if (observed_release) {
        observer->release_thread_ = std::this_thread::get_id();
        observer->released_ = true;
        observer->condition_.notify_all();
      } else if (command != IREE_ALLOCATOR_COMMAND_FREE &&
                 observer->observe_next_allocation_) {
        observer->allocation_ = *inout_pointer;
        observer->observe_next_allocation_ = false;
      }
    }
    return status;
  }

  // Protects the selected allocation and its release observation.
  std::mutex mutex_;
  // Wakes the test after the underlying native free returns.
  std::condition_variable condition_;
  // Selects the next native allocation after provider construction.
  bool observe_next_allocation_ = false;
  // Underlying system allocation, before aligned-pointer adjustment.
  void* allocation_ = nullptr;
  // True after the selected native allocation has actually been freed.
  bool released_ = false;
  // Thread executing that native free.
  std::thread::id release_thread_;
};

class TaskQueueNativeRetirementTest : public TaskQueueAllocaTest {
 protected:
  // Completion-driven pool destruction must not wait on this sole worker.
  iree_host_size_t WorkerCount() const override { return 1; }

  void UseNativePool() {
    iree_hal_queue_pool_backend_t backend = {};
    IREE_ASSERT_OK(iree_hal_device_query_queue_pool_backend(
        devices_[GetParam()], iree_hal_queue_family(queues_[GetParam()]),
        &backend));
    struct OwnerProbe : iree_hal_memory_maintenance_entry_t {
      // Protects the identity and the callback's final notification access.
      std::mutex mutex;
      // Wakes the test after the captured owner runs the probe.
      std::condition_variable condition;
      // Identity of the independent memory-domain worker.
      std::thread::id thread;
      // Set by the owner under the mutex.
      bool complete = false;
    } probe;
    probe.fn = [](iree_hal_memory_maintenance_entry_t* entry) {
      auto* probe = static_cast<OwnerProbe*>(entry);
      std::lock_guard<std::mutex> lock(probe->mutex);
      probe->thread = std::this_thread::get_id();
      probe->complete = true;
      probe->condition.notify_all();
    };
    iree_hal_memory_maintenance_enqueue(backend.maintenance, &probe);
    {
      std::unique_lock<std::mutex> lock(probe.mutex);
      probe.condition.wait(lock, [&] { return probe.complete; });
      memory_thread_ = probe.thread;
    }
    iree_hal_slab_provider_t* provider = nullptr;
    IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(
        /*min_alignment=*/0, backing_.allocator(), &provider));
    iree_hal_pool_release(pool_);
    pool_ = nullptr;
    iree_status_t status = iree_hal_passthrough_pool_create(
        {}, provider, backend.notification, backend.frontier_tracker,
        backend.maintenance, iree_allocator_system(), &pool_);
    iree_hal_slab_provider_release(provider);
    IREE_ASSERT_OK(status);
    notification_ = iree_hal_pool_notification(pool_);
  }

  // Remains alive through the base fixture's pool and device teardown.
  NativeBackingObserver backing_;
  // Identity captured directly from the selected maintenance owner.
  std::thread::id memory_thread_;
};

TEST_P(TaskQueueNativeRetirementTest, QueueBytesRetireOnCapturedOwner) {
  ASSERT_NO_FATAL_FAILURE(UseNativePool());
  struct OwnerProbe {
    // Live pool whose trim must be callable from the sole executor worker.
    iree_hal_pool_t* pool;
    // Identity observed by the explicit queue host call.
    std::thread::id executor_thread;
  } owner_probe = {pool_, {}};
  const uint64_t args[4] = {};
  const auto call = iree_hal_make_host_call(
      [](void* user_data, const uint64_t args[4],
         iree_hal_host_call_context_t* context) -> iree_status_t {
        auto* probe = static_cast<OwnerProbe*>(user_data);
        iree_hal_pool_trim(probe->pool, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
        probe->executor_thread = std::this_thread::get_id();
        return iree_ok_status();
      },
      &owner_probe);
  uint64_t first_value = 1;
  const iree_hal_semaphore_list_t observed = {1, &semaphores_[0], &first_value};
  IREE_ASSERT_OK(iree_hal_queue_host_call(
      queues_[GetParam()], iree_hal_semaphore_list_empty(), observed, call,
      args, IREE_HAL_HOST_CALL_FLAG_NONE));
  ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[0], first_value));

  iree_hal_pool_reservation_request_t request = {.allocation_size = kBlockSize};
  request.params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  request.params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  request.params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  request.params.queue_family_affinity = iree_hal_make_queue_family_affinity(0);
  backing_.ObserveNextAllocation();
  const iree_hal_semaphore_list_t allocated = {1, &semaphores_[1],
                                               &first_value};
  IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], observed, allocated, pool_,
                                       1, &request, &initial_buffers_[0]));
  ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[1], first_value));
  uint64_t filled_value = 2;
  const iree_hal_semaphore_list_t filled = {1, &semaphores_[1], &filled_value};
  const uint32_t pattern = 0xC0FFEE12u;
  IREE_ASSERT_OK(iree_hal_queue_fill(
      queues_[0], allocated, filled, initial_buffers_[0], 0, kBlockSize,
      &pattern, sizeof(pattern), /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
  ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[1], filled_value));
  std::array<uint32_t, kBlockSize / sizeof(uint32_t)> actual;
  IREE_ASSERT_OK(iree_hal_buffer_map_read(initial_buffers_[0], 0, actual.data(),
                                          sizeof(actual)));
  for (uint32_t value : actual) {
    EXPECT_EQ(value, pattern);
  }

  // Deallocation is explicitly gated; submission order supplies no dependency.
  uint64_t release_wait_values[] = {filled_value, first_value};
  const iree_hal_semaphore_list_t release_waits = {2, &semaphores_[1],
                                                   release_wait_values};
  uint64_t released_value = 3;
  const iree_hal_semaphore_list_t released = {1, &semaphores_[1],
                                              &released_value};
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[0], release_waits, released, 1,
                                         &initial_buffers_[0]));
  EXPECT_FALSE(backing_.released());
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphores_[2], first_value,
                                           /*frontier=*/nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[1], released_value));
  iree_hal_buffer_release(initial_buffers_[0]);
  initial_buffers_[0] = nullptr;
  // No subsequent allocation, trim, or destruction drives this native free.
  backing_.AwaitRelease();
  EXPECT_EQ(backing_.release_thread(), memory_thread_);
  EXPECT_NE(backing_.release_thread(), owner_probe.executor_thread);
  EXPECT_NE(backing_.release_thread(), std::this_thread::get_id());
}

TEST_P(TaskQueueNativeRetirementTest,
       BorrowedAddressesRetainCrossQueueHistory) {
  ASSERT_NO_FATAL_FAILURE(UseNativePool());
  iree_hal_pool_reservation_request_t request = {.allocation_size = kBlockSize};
  request.params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  request.params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  request.params.usage = IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
  backing_.ObserveNextAllocation();
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool_, 1, &request, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      &reservation, &info, &result));
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      pool_, 1, &request, &reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
      &initial_buffers_[0]));
  iree_hal_external_buffer_t external = {};
  IREE_ASSERT_OK(iree_hal_buffer_export(
      initial_buffers_[0], IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external));

  // These host kernels receive borrowed addresses, as native/custom dispatch
  // arguments can. Their caller explicitly supplies the complete lifetime
  // frontier; no buffer references or inferred dependencies keep bytes alive.
  struct Output {
    // Final user-visible readback copied while the address is still valid.
    std::array<uint32_t, kBlockSize / sizeof(uint32_t)> words;
    // Identity of the sole execution worker, independent of native cleanup.
    std::thread::id executor_thread;
  } output;
  const auto call = iree_hal_make_host_call(
      [](void* user_data, const uint64_t args[4],
         iree_hal_host_call_context_t* context) -> iree_status_t {
        auto* output = static_cast<Output*>(user_data);
        auto* words = reinterpret_cast<uint32_t*>(args[0]);
        const size_t half_length = output->words.size() / 2;
        const size_t offset = args[1] * half_length;
        for (size_t i = 0; i < half_length; ++i) {
          words[offset + i] = static_cast<uint32_t>(0xCAFE1000u + args[1]);
        }
        if (args[1] == 1) {
          memcpy(output->words.data(), words, sizeof(output->words));
          output->executor_thread = std::this_thread::get_id();
        }
        return iree_ok_status();
      },
      &output);
  IREE_ASYNC_FIXED_FRONTIER_TYPE(PairFrontier, 2);
  PairFrontier frontier = {.entry_count = 2};
  for (size_t i = 0; i < queues_.size(); ++i) {
    const auto* queue = reinterpret_cast<iree_hal_task_queue_t*>(queues_[i]);
    // The fixture owns both fresh queues and submits exactly one operation to
    // each, so completion epoch one identifies these two specific host calls.
    ASSERT_EQ(iree_atomic_load(&queue->epoch, iree_memory_order_acquire), 0);
    frontier.entries[i] = {queue->axis, 1};
    const uint64_t args[4] = {
        reinterpret_cast<uintptr_t>(external.handle.host_allocation.ptr), i, 0,
        0};
    iree_hal_semaphore_t* wait_semaphores[] = {semaphores_[i], semaphores_[2]};
    uint64_t wait_values[] = {1, 1};
    const iree_hal_semaphore_list_t waits = {i + 1, wait_semaphores,
                                             wait_values};
    uint64_t completed_value = i + 1;
    const iree_hal_semaphore_list_t completed = {1, &semaphores_[2],
                                                 &completed_value};
    IREE_ASSERT_OK(iree_hal_queue_host_call(queues_[i], waits, completed, call,
                                            args,
                                            IREE_HAL_HOST_CALL_FLAG_NONE));
  }
  iree_hal_pool_release_reservations(
      pool_, 1, &reservation,
      iree_async_fixed_frontier_as_const_frontier(&frontier));
  memset(&frontier, 0, sizeof(frontier));
  iree_hal_buffer_release(initial_buffers_[0]);
  initial_buffers_[0] = nullptr;
  EXPECT_FALSE(backing_.released());

  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphores_[0], 1,
                                           /*frontier=*/nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[2], 1));
  EXPECT_FALSE(backing_.released());
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.bytes_committed, kBlockSize);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphores_[1], 1,
                                           /*frontier=*/nullptr));
  ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[2], 2));
  backing_.AwaitRelease();
  EXPECT_EQ(backing_.release_thread(), memory_thread_);
  EXPECT_NE(backing_.release_thread(), output.executor_thread);
  for (size_t i = 0; i < output.words.size(); ++i) {
    EXPECT_EQ(output.words[i], 0xCAFE1000u + i / (output.words.size() / 2));
  }
}

TEST_P(TaskQueueNativeRetirementTest, CompletionCanDestroyItsPool) {
  ASSERT_NO_FATAL_FAILURE(UseNativePool());
  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.queue_family_affinity = iree_hal_make_queue_family_affinity(0);
  backing_.ObserveNextAllocation();
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(pool_, params, kBlockSize,
                                               iree_infinite_timeout(),
                                               &initial_buffers_[0]));

  const uint32_t pattern = 0xDCBA9876u;
  uint64_t completed_value = 1;
  const iree_hal_semaphore_list_t filled = {1, &semaphores_[0],
                                            &completed_value};
  IREE_ASSERT_OK(iree_hal_queue_fill(
      queues_[0], iree_hal_semaphore_list_empty(), filled, initial_buffers_[0],
      0, kBlockSize, &pattern, sizeof(pattern), /*barriers=*/NULL,
      IREE_HAL_FILL_FLAG_NONE));

  struct Completion {
    // Final pool reference, transferred to the accepted host call.
    iree_hal_pool_t* pool;
    // Final owned buffer reference after the preceding fill completes.
    iree_hal_buffer_t* buffer;
    // Output copied before releasing native storage.
    std::array<uint32_t, kBlockSize / sizeof(uint32_t)> words;
    // Task worker that destroys the buffer and pool.
    std::thread::id thread;
  } completion = {pool_, initial_buffers_[0], {}, {}};
  pool_ = nullptr;
  initial_buffers_[0] = nullptr;
  const auto call = iree_hal_make_host_call(
      [](void* user_data, const uint64_t args[4],
         iree_hal_host_call_context_t* context) -> iree_status_t {
        auto* completion = static_cast<Completion*>(user_data);
        completion->thread = std::this_thread::get_id();
        iree_status_t status = iree_hal_buffer_map_read(
            completion->buffer, 0, completion->words.data(),
            sizeof(completion->words));
        // Final pool release on a Task worker must join native cleanup without
        // depending on work scheduled to that same worker.
        iree_hal_buffer_release(completion->buffer);
        iree_hal_pool_trim(completion->pool, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
        iree_hal_pool_release(completion->pool);
        return status;
      },
      &completion);
  const uint64_t args[4] = {};
  const iree_hal_semaphore_list_t completed = {1, &semaphores_[1],
                                               &completed_value};
  iree_status_t status = iree_hal_queue_host_call(
      queues_[0], filled, completed, call, args, IREE_HAL_HOST_CALL_FLAG_NONE);
  if (!iree_status_is_ok(status)) {
    pool_ = completion.pool;
    initial_buffers_[0] = completion.buffer;
  }
  IREE_ASSERT_OK(status);
  ASSERT_NO_FATAL_FAILURE(Wait(semaphores_[1], completed_value));
  EXPECT_TRUE(backing_.released());
  EXPECT_EQ(backing_.release_thread(), memory_thread_);
  EXPECT_NE(backing_.release_thread(), completion.thread);
  for (uint32_t value : completion.words) {
    EXPECT_EQ(value, pattern);
  }
}

INSTANTIATE_TEST_SUITE_P(MemoryOwners, TaskQueueNativeRetirementTest,
                         ::testing::Values(0, 1));

}  // namespace
