// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_pending.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>

#include "iree/async/frontier.h"
#include "iree/async/notification.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/status_cc.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/pool_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/allocator.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::hal::cts::Ref;

class HostQueuePendingTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    host_allocator_ = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (iree_status_is_unavailable(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_with_defaults(
        &libhsa_, &topology_));
    if (topology_.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology_);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
  }

  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t HostQueuePendingTest::host_allocator_;
iree_hal_amdgpu_libhsa_t HostQueuePendingTest::libhsa_;
iree_hal_amdgpu_topology_t HostQueuePendingTest::topology_;

class TestLogicalDevice {
 public:
  ~TestLogicalDevice() {
    iree_hal_device_release(base_device_);
    iree_hal_device_group_release(device_group_);
  }

  iree_status_t Initialize(
      const iree_hal_amdgpu_logical_device_options_t* options,
      const iree_hal_amdgpu_libhsa_t* libhsa,
      const iree_hal_amdgpu_topology_t* topology,
      iree_allocator_t host_allocator) {
    IREE_RETURN_IF_ERROR(create_context_.Initialize(host_allocator));
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_create(
        IREE_SV("amdgpu"), options, libhsa, topology, create_context_.params(),
        host_allocator, &base_device_));
    return iree_hal_device_group_create_from_device(
        base_device_, create_context_.frontier_tracker(), host_allocator,
        &device_group_);
  }

  iree_hal_device_t* base_device() const { return base_device_; }

  iree_hal_device_group_t* device_group() const { return device_group_; }

  iree_hal_queue_t* queue() const {
    return iree_hal_device_queue(base_device_, /*family_ordinal=*/0,
                                 /*queue_ordinal=*/0);
  }

  iree_hal_allocator_t* allocator() const {
    return iree_hal_device_allocator(base_device_);
  }

  iree_hal_amdgpu_logical_device_t* logical_device() const {
    return (iree_hal_amdgpu_logical_device_t*)base_device_;
  }

  iree_hal_amdgpu_host_queue_t* first_host_queue() const {
    iree_hal_amdgpu_logical_device_t* logical_device = this->logical_device();
    if (logical_device->physical_device_count == 0) {
      return NULL;
    }
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[0];
    if (physical_device->host_queue_count == 0) {
      return NULL;
    }
    return &physical_device->host_queues[0];
  }

 private:
  // Creation context supplying the proactor pool and frontier tracker.
  iree::hal::cts::DeviceCreateContext create_context_;

  // Test-owned device reference released before the topology-owning group.
  iree_hal_device_t* base_device_ = NULL;

  // Device group that owns the topology assigned to |base_device_|.
  iree_hal_device_group_t* device_group_ = NULL;
};

static iree_status_t CreateHostVisibleTransferBuffer(
    iree_hal_allocator_t* allocator, iree_device_size_t buffer_size,
    iree_hal_buffer_t** out_buffer) {
  const iree_hal_amdgpu_buffer_spec_t spec = {
      .type = IREE_HAL_MEMORY_TYPE_OPTIMAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
              IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .usage =
          IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED,
      .queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
  };
  return iree_hal_amdgpu_allocator_allocate_direct_buffer(
      allocator, spec, buffer_size, out_buffer);
}

static iree_status_t CreateSemaphore(iree_hal_device_t* device,
                                     iree_hal_semaphore_t** out_semaphore) {
  return iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, out_semaphore);
}

static iree_hal_semaphore_list_t MakeSemaphoreList(
    iree_hal_semaphore_t** semaphore, uint64_t* payload_value) {
  return iree_hal_semaphore_list_t{
      /*count=*/1,
      /*semaphores=*/semaphore,
      /*payload_values=*/payload_value,
  };
}

static iree_status_t QueueAlloca(iree_hal_amdgpu_host_queue_t* queue,
                                 iree_hal_pool_t* pool,
                                 iree_hal_semaphore_list_t signal_list,
                                 iree_device_size_t allocation_size,
                                 iree_device_size_t min_alignment,
                                 iree_hal_buffer_t** out_buffer) {
  const iree_hal_pool_reservation_request_t request =
      iree_hal_make_pool_reservation_request(allocation_size, min_alignment);
  return iree_hal_queue_alloca(&queue->base, iree_hal_semaphore_list_empty(),
                               signal_list, pool,
                               /*request_count=*/1, &request, out_buffer);
}

static iree_status_t QueueDealloca(iree_hal_amdgpu_host_queue_t* queue,
                                   iree_hal_semaphore_list_t wait_list,
                                   iree_hal_semaphore_list_t signal_list,
                                   iree_hal_buffer_t* buffer) {
  return iree_hal_queue_dealloca(&queue->base, wait_list, signal_list,
                                 /*buffer_count=*/1, &buffer);
}

// Cancels the queue's deferred operations the way the queue itself does:
// admission closed first, so no submission can be admitted alongside the
// cancellation pass.
static void CancelPendingWithTestStatus(iree_hal_amdgpu_host_queue_t* queue) {
  iree_hal_amdgpu_host_queue_begin_deinitialize(queue);
  iree_status_t cancellation_status =
      iree_make_status(IREE_STATUS_CANCELLED, "test cancellation");
  iree_hal_amdgpu_host_queue_cancel_pending(queue, cancellation_status);
  iree_status_free(cancellation_status);
}

static bool HostQueueHasPendingOps(iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  const bool has_pending_ops = queue->pending_head != NULL;
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  return has_pending_ops;
}

static bool HostQueueHasPostDrainAction(iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.post_drain_mutex);
  const bool has_action = queue->post_drain.head != NULL;
  iree_slim_mutex_unlock(&queue->locks.post_drain_mutex);
  return has_action;
}

static void EnqueueRawBlockingBarrier(
    iree_hal_amdgpu_host_queue_t* queue, hsa_signal_t blocker_signal,
    hsa_signal_t completion_signal = iree_hsa_signal_null()) {
  const uint64_t packet_id =
      iree_hal_amdgpu_aql_ring_reserve(&queue->aql_ring, /*count=*/1);
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  const hsa_signal_t dep_signals[1] = {blocker_signal};
  const uint16_t header = iree_hal_amdgpu_aql_emit_barrier_and(
      &packet->barrier_and, dep_signals, IREE_ARRAYSIZE(dep_signals),
      iree_hal_amdgpu_aql_packet_control_barrier_system(), completion_signal);
  iree_hal_amdgpu_aql_ring_commit(packet, header, /*setup=*/0);
  iree_hal_amdgpu_aql_ring_doorbell(&queue->aql_ring, packet_id);
}

static iree_status_t CreateExplicitFixedBlockPool(
    iree_hal_device_group_t* device_group,
    const iree_hal_queue_family_t* queue_family, iree_device_size_t block_size,
    iree_hal_pool_t** out_native_pool, iree_hal_pool_t** out_pool) {
  *out_native_pool = nullptr;
  *out_pool = nullptr;
  Ref<iree_hal_pool_t> backing_pool;
  IREE_RETURN_IF_ERROR(iree::hal::cts::CreateQueueSlabPool(
      device_group, queue_family,
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
      /*host_access=*/{}, iree_allocator_system(), backing_pool.out()));
  iree_hal_fixed_block_pool_options_t options = {
      .block_size = block_size,
      .blocks_per_slab = 1,
      .frontier_capacity = 2,
  };
  IREE_RETURN_IF_ERROR(iree::hal::cts::CreateFiniteBlockPool(
      backing_pool, options, iree_allocator_system(), out_pool));
  *out_native_pool = backing_pool.release();
  return iree_ok_status();
}

// Holds a native provider's metadata allocation while real ROCr queue work
// continues. Caller-side capture allocations never enter the hold.
class NativeAllocationHold {
 public:
  ~NativeAllocationHold() { Release(); }

  iree_allocator_t allocator() { return {this, Control}; }

  void Arm() {
    std::lock_guard<std::mutex> lock(mutex_);
    armed_ = true;
    entered_ = released_ = false;
  }

  void WaitEntered() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return entered_; });
  }

  void Release() {
    std::lock_guard<std::mutex> lock(mutex_);
    armed_ = false;
    released_ = true;
    condition_.notify_all();
  }

 private:
  static iree_status_t Control(void* user_data,
                               iree_allocator_command_t command,
                               const void* params, void** inout_pointer) {
    auto* hold = static_cast<NativeAllocationHold*>(user_data);
    {
      std::unique_lock<std::mutex> lock(hold->mutex_);
      if (hold->armed_ && command == IREE_ALLOCATOR_COMMAND_CALLOC &&
          std::this_thread::get_id() != hold->caller_) {
        hold->armed_ = false;
        hold->entered_ = true;
        hold->condition_.notify_all();
        hold->condition_.wait(lock, [&] { return hold->released_; });
      }
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, params, inout_pointer);
  }

  // Serializes dependency ownership between the test and native worker.
  std::mutex mutex_;
  // Publishes native entry and releases the held dependency.
  std::condition_variable condition_;
  // Thread performing public queue capture, excluded from the hold.
  std::thread::id caller_ = std::this_thread::get_id();
  // Exactly one non-caller allocation is selected after setup completes.
  bool armed_ = false;
  // The allocation worker owns the selected dependency.
  bool entered_ = false;
  // The owner has authorized the native call to continue.
  bool released_ = false;
};

TEST_F(HostQueuePendingTest, NativeAllocationDoesNotBlockCachedQueueWork) {
  NativeAllocationHold native;
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;
  TestLogicalDevice device;
  IREE_ASSERT_OK(
      device.Initialize(&options, &libhsa_, &topology_, native.allocator()));
  auto* queue = device.first_host_queue();
  ASSERT_NE(queue, nullptr);
  Ref<iree_hal_pool_t> source;
  Ref<iree_hal_pool_t> finite;
  IREE_ASSERT_OK(CreateExplicitFixedBlockPool(
      device.device_group(), iree_hal_queue_family(&queue->base), 4096,
      source.out(), finite.out()));
  Ref<iree_hal_buffer_t> readback;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      device.allocator(), sizeof(uint32_t), readback.out()));
  iree::hal::cts::SemaphoreList growing_ready(device.base_device(), {0}, {1});
  iree::hal::cts::SemaphoreList cached_ready(device.base_device(), {0}, {1});
  iree::hal::cts::SemaphoreList filled(device.base_device(), {0}, {1});
  iree::hal::cts::SemaphoreList copied(device.base_device(), {0}, {1});
  iree::hal::cts::SemaphoreList returned(device.base_device(), {0}, {1});
  Ref<iree_hal_buffer_t> growing;
  Ref<iree_hal_buffer_t> cached;
  const auto request = iree_hal_make_pool_reservation_request(4096, 0);
  const auto no_waits = iree_hal_semaphore_list_empty();
  native.Arm();
  // Release before resource teardown even when an assertion fails.
  struct ScopedRelease {
    ~ScopedRelease() { native.Release(); }
    // Borrowed allocator dependency outliving all device resources.
    NativeAllocationHold& native;
  } release{native};
  IREE_ASSERT_OK(iree_hal_queue_alloca(&queue->base, no_waits, growing_ready,
                                       source, 1, &request, growing.out()));
  native.WaitEntered();
  EXPECT_FALSE(iree_hal_semaphore_list_poll(growing_ready));
  IREE_ASSERT_OK(iree_hal_queue_alloca(&queue->base, no_waits, cached_ready,
                                       finite, 1, &request, cached.out()));
  const uint32_t pattern = 0xC01DCAFE;
  IREE_ASSERT_OK(iree_hal_queue_fill(
      &queue->base, cached_ready, filled, cached, 0, sizeof(pattern), &pattern,
      sizeof(pattern), nullptr, IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_queue_copy(&queue->base, filled, copied, cached, 0,
                                     readback, 0, sizeof(pattern), nullptr,
                                     IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(copied, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  uint32_t result = 0;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(readback, 0, &result, sizeof(result)));
  EXPECT_EQ(result, pattern);
  auto* buffer = cached.get();
  IREE_ASSERT_OK(
      iree_hal_queue_dealloca(&queue->base, copied, returned, 1, &buffer));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(returned, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_FALSE(iree_hal_semaphore_list_poll(growing_ready));
  native.Release();
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      growing_ready, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
}

TEST_F(HostQueuePendingTest,
       SynchronousPoolWaitsForNativeReclaimAfterTimeoutRollback) {
  constexpr iree_device_size_t kByteLength = 512;
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;
  TestLogicalDevice device;
  IREE_ASSERT_OK(
      device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  auto* queue = device.first_host_queue();
  ASSERT_NE(queue, nullptr);
  const auto* family = iree_hal_queue_family(&queue->base);
  Ref<iree_hal_pool_t> native_pool;
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateExplicitFixedBlockPool(device.device_group(), family,
                                              kByteLength, native_pool.out(),
                                              pool.out()));
  const iree_hal_pool_reservation_request_t request =
      iree_hal_make_pool_reservation_request(kByteLength,
                                             /*min_alignment=*/0);
  Ref<iree_hal_semaphore_t> completion;
  IREE_ASSERT_OK(CreateSemaphore(device.base_device(), completion.out()));
  iree_hal_semaphore_t* semaphore = completion.get();
  uint64_t allocated_value = 1, used_value = 2, released_value = 3;
  uint64_t filled_value = 4, copied_value = 5;
  const auto allocated = MakeSemaphoreList(&semaphore, &allocated_value);
  const auto used = MakeSemaphoreList(&semaphore, &used_value);
  const auto released = MakeSemaphoreList(&semaphore, &released_value);
  const auto filled = MakeSemaphoreList(&semaphore, &filled_value);
  const auto copied = MakeSemaphoreList(&semaphore, &copied_value);
  const auto no_waits = iree_hal_semaphore_list_empty();
  Ref<iree_hal_buffer_t> original;
  IREE_ASSERT_OK(iree_hal_queue_alloca(&queue->base, no_waits, allocated, pool,
                                       1, &request, original.out()));
  const uint32_t old_pattern = 0x11111111u;
  IREE_ASSERT_OK(iree_hal_queue_fill(
      &queue->base, allocated, used, original, 0, kByteLength, &old_pattern,
      sizeof(old_pattern), /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
  // Complete the last-use edge first: this backend parks deallocation behind
  // unfinished users. The gate below instead withholds native reclaim.
  IREE_ASSERT_OK(iree_hal_semaphore_wait(completion, used_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  Ref<iree_hal_buffer_t> readback;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(device.allocator(),
                                                 kByteLength, readback.out()));

  hsa_signal_t blocker = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(IREE_LIBHSA(&libhsa_), 1, 0,
                                            nullptr, 0, &blocker));
  hsa_signal_t gate_completion = iree_hsa_signal_null();
  iree_status_t status = iree_hsa_amd_signal_create(
      IREE_LIBHSA(&libhsa_), 1, 0, nullptr, 0, &gate_completion);
  if (!iree_status_is_ok(status)) {
    IREE_EXPECT_OK(iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker));
  }
  IREE_ASSERT_OK(status);
  EnqueueRawBlockingBarrier(queue, blocker, gate_completion);
  iree_hal_buffer_t* original_ptr = original.get();
  status =
      iree_hal_queue_dealloca(&queue->base, used, released, 1, &original_ptr);
  const bool dealloca_submitted = iree_status_is_ok(status);
  Ref<iree_hal_buffer_t> recycled;
  iree::Status allocation_status;
  std::atomic<bool> finished{false};
  std::thread allocating;
  if (dealloca_submitted) {
    // Wait for the reservation to return before starting the timeout. This
    // separates pending reuse from delays in publishing returned capacity.
    iree_hal_pool_stats_t returned_stats = {};
    do {
      iree_hal_pool_query_stats(pool, &returned_stats);
      std::this_thread::yield();
    } while (returned_stats.release_count == 0);
    // Timeout is the behavior under test; native progress stays explicitly
    // held.
    iree_status_t timeout_status =
        iree_hal_pool_allocate_buffer(pool,
                                      iree_hal_make_pool_reservation_request(
                                          kByteLength, request.min_alignment),
                                      iree_make_timeout_ms(1), recycled.out());
    const bool timed_out = iree_status_is_deadline_exceeded(timeout_status);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED, timeout_status);
    if (timed_out) {
      allocating = std::thread([&] {
        allocation_status = iree_hal_pool_allocate_buffer(
            pool,
            iree_hal_make_pool_reservation_request(kByteLength,
                                                   request.min_alignment),
            iree_infinite_timeout(), recycled.out());
        finished.store(true, std::memory_order_release);
      });
      iree_hal_pool_stats_t stats = {};
      do {
        iree_hal_pool_query_stats(pool, &stats);
        std::this_thread::yield();
      } while (stats.wait_count < 2 &&
               !finished.load(std::memory_order_acquire));
      // The second allocation must still inherit the original death frontier.
      EXPECT_EQ(stats.wait_count, 2u);
      EXPECT_FALSE(finished.load(std::memory_order_acquire));
    }
  }

  // Every error path opens and joins the native gate before destroying signals.
  // Completion also wakes the exact waiter without another capacity broadcast.
  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker, 0);
  if (allocating.joinable()) {
    allocating.join();
  }
  EXPECT_EQ(iree_hsa_signal_wait_scacquire(
                IREE_LIBHSA(&libhsa_), gate_completion, HSA_SIGNAL_CONDITION_EQ,
                0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
            0);
  IREE_EXPECT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), gate_completion));
  IREE_EXPECT_OK(iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker));
  status = iree_status_join(status, allocation_status.release());
  if (dealloca_submitted) {
    status = iree_status_join(
        status, iree_hal_semaphore_wait(completion, released_value,
                                        iree_infinite_timeout(),
                                        IREE_ASYNC_WAIT_FLAG_NONE));
  }
  IREE_ASSERT_OK(status);
  ASSERT_NE(recycled.get(), nullptr);
  const uint32_t new_pattern = 0xAABBCCDDu;
  IREE_ASSERT_OK(iree_hal_queue_fill(
      &queue->base, no_waits, filled, recycled, 0, kByteLength, &new_pattern,
      sizeof(new_pattern), /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_queue_copy(
      &queue->base, filled, copied, recycled, 0, readback, 0, kByteLength,
      /*barriers=*/NULL, IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(completion, copied_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_buffer_mapping_t mapping = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      readback, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, kByteLength, &mapping));
  const auto* words = reinterpret_cast<const uint32_t*>(mapping.contents.data);
  for (iree_host_size_t i = 0; i < kByteLength / sizeof(uint32_t); ++i) {
    EXPECT_EQ(words[i], new_pattern);
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
}

static iree_status_t CreateExplicitTlsfPool(
    iree_hal_device_group_t* device_group,
    const iree_hal_queue_family_t* queue_family, iree_device_size_t slab_size,
    iree_hal_pool_t** out_pool) {
  iree_hal_tlsf_pool_options_t options = {};
  options.tlsf_options.range_length = slab_size;
  options.tlsf_options.alignment = 16;
  options.tlsf_options.initial_block_capacity = 16;
  options.tlsf_options.frontier_capacity = 2;
  iree_hal_pool_t* backing_pool = nullptr;
  IREE_RETURN_IF_ERROR(iree::hal::cts::CreateQueueSlabPool(
      device_group, queue_family,
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
      /*host_access=*/{}, iree_allocator_system(), &backing_pool));
  iree_status_t status = iree_hal_tlsf_pool_create(
      backing_pool, &options, iree_allocator_system(), out_pool);
  iree_hal_pool_release(backing_pool);
  return status;
}

static iree_status_t SeedWaitableFixedBlockReservation(
    iree_hal_pool_t* pool, iree_device_size_t allocation_size,
    iree_async_axis_t death_axis) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t acquire_info;
  iree_hal_pool_acquire_result_t acquire_result;
  const iree_hal_pool_reservation_request_t request =
      iree_hal_make_pool_reservation_request(allocation_size,
                                             /*min_alignment=*/1);
  IREE_RETURN_IF_ERROR(iree_hal_pool_acquire_reservations_sync(
      pool, 1, &request,
      /*requester_frontier=*/NULL, &reservation, &acquire_info, &acquire_result,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  if (acquire_result != IREE_HAL_POOL_ACQUIRE_OK &&
      acquire_result != IREE_HAL_POOL_ACQUIRE_OK_FRESH) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "expected fresh fixed-block reservation");
  }

  iree_async_single_frontier_t death_frontier;
  iree_async_single_frontier_initialize(&death_frontier, death_axis, 1);
  iree_hal_pool_release_reservations(
      pool, 1, &reservation,
      iree_async_single_frontier_as_const_frontier(&death_frontier));
  return iree_ok_status();
}

typedef struct HostActionState {
  // Notification posted after the action callback records its result.
  iree_notification_t notification;

  // Number of times the action callback has run.
  iree_atomic_int32_t call_count;

  // Last callback status code.
  iree_atomic_int32_t status_code;

  // Whether the callback received a reclaim entry.
  iree_atomic_int32_t had_entry;
} HostActionState;

static void HostActionStateInitialize(HostActionState* state) {
  iree_notification_initialize(&state->notification);
  iree_atomic_store(&state->call_count, 0, iree_memory_order_relaxed);
  iree_atomic_store(&state->status_code, IREE_STATUS_UNKNOWN,
                    iree_memory_order_relaxed);
  iree_atomic_store(&state->had_entry, 0, iree_memory_order_relaxed);
}

static void HostActionStateDeinitialize(HostActionState* state) {
  iree_notification_deinitialize(&state->notification);
}

static bool HostActionStateWasCalled(void* user_data) {
  HostActionState* state = (HostActionState*)user_data;
  return iree_atomic_load(&state->call_count, iree_memory_order_acquire) != 0;
}

static void RecordHostAction(iree_hal_amdgpu_reclaim_entry_t* entry,
                             void* user_data, const iree_status_t status) {
  HostActionState* state = (HostActionState*)user_data;
  iree_atomic_store(&state->had_entry, entry ? 1 : 0,
                    iree_memory_order_release);
  iree_atomic_store(&state->status_code, iree_status_code(status),
                    iree_memory_order_release);
  iree_atomic_fetch_add(&state->call_count, 1, iree_memory_order_acq_rel);
  iree_notification_post(&state->notification, IREE_ALL_WAITERS);
}

static int32_t HostActionCallCount(HostActionState* state) {
  return iree_atomic_load(&state->call_count, iree_memory_order_acquire);
}

static iree_status_code_t HostActionStatusCode(HostActionState* state) {
  return (iree_status_code_t)iree_atomic_load(&state->status_code,
                                              iree_memory_order_acquire);
}

static bool HostActionHadEntry(HostActionState* state) {
  return iree_atomic_load(&state->had_entry, iree_memory_order_acquire) != 0;
}

TEST_F(HostQueuePendingTest,
       DeferredHostActionFailureRunsSynchronousCallbackOnce) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_semaphore_t> wait_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), wait_semaphore.out()));
  iree_hal_semaphore_fail(
      wait_semaphore,
      iree_make_status(IREE_STATUS_CANCELLED, "test wait failed"));
  uint64_t wait_value = 1;
  iree_hal_semaphore_t* wait_semaphore_ptr = wait_semaphore.get();
  const iree_hal_semaphore_list_t wait_list =
      MakeSemaphoreList(&wait_semaphore_ptr, &wait_value);

  HostActionState action_state;
  HostActionStateInitialize(&action_state);
  IREE_ASSERT_OK(iree_hal_amdgpu_host_queue_enqueue_host_action(
      queue, wait_list,
      iree_hal_amdgpu_reclaim_action_t{
          .fn = RecordHostAction,
          .user_data = &action_state,
      },
      /*operation_resources=*/NULL, /*operation_resource_count=*/0,
      (iree_hal_amdgpu_queue_barrier_t){0}));

  EXPECT_EQ(HostActionCallCount(&action_state), 1);
  EXPECT_EQ(HostActionStatusCode(&action_state), IREE_STATUS_CANCELLED);
  EXPECT_FALSE(HostActionHadEntry(&action_state));
  EXPECT_FALSE(HostQueueHasPendingOps(queue));

  HostActionStateDeinitialize(&action_state);
}

TEST_F(HostQueuePendingTest, CancelPendingFillFailsSignalSemaphore) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), target_buffer.out()));

  Ref<iree_hal_semaphore_t> wait_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), wait_semaphore.out()));
  uint64_t wait_value = 1;
  iree_hal_semaphore_t* wait_semaphore_ptr = wait_semaphore.get();
  const iree_hal_semaphore_list_t wait_list =
      MakeSemaphoreList(&wait_semaphore_ptr, &wait_value);

  Ref<iree_hal_semaphore_t> signal_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), signal_semaphore.out()));
  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_semaphore_ptr = signal_semaphore.get();
  const iree_hal_semaphore_list_t signal_list =
      MakeSemaphoreList(&signal_semaphore_ptr, &signal_value);

  const uint32_t pattern = 0xCACE1100u;
  IREE_ASSERT_OK(iree_hal_queue_fill(
      test_device.queue(), wait_list, signal_list, target_buffer,
      /*target_offset=*/0, sizeof(pattern), &pattern, sizeof(pattern),
      /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
  ASSERT_TRUE(HostQueueHasPendingOps(queue));

  CancelPendingWithTestStatus(queue);
  EXPECT_FALSE(HostQueueHasPendingOps(queue));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        iree_hal_semaphore_wait(signal_semaphore, signal_value,
                                                iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_EXPECT_OK(
      iree_hal_semaphore_signal(wait_semaphore, wait_value, /*frontier=*/NULL));
}

TEST_F(HostQueuePendingTest, DeferredIssueFailureReachesDependentOperation) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), target_buffer.out()));

  // Neither operation can issue before the host releases the first semaphore.
  // Failing the first operation must synchronously fail the dependent one,
  // including its pending-list removal on this same queue.
  Ref<iree_hal_semaphore_t> semaphores[3];
  for (auto& semaphore : semaphores) {
    IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), semaphore.out()));
  }
  uint64_t value = 1;
  for (int i = 0; i < 2; ++i) {
    iree_hal_semaphore_t* wait_semaphore = semaphores[i].get();
    iree_hal_semaphore_t* signal_semaphore = semaphores[i + 1].get();
    const uint32_t pattern = 0xCACE1100u;
    IREE_ASSERT_OK(iree_hal_queue_fill(
        test_device.queue(), MakeSemaphoreList(&wait_semaphore, &value),
        MakeSemaphoreList(&signal_semaphore, &value), target_buffer,
        /*target_offset=*/0, sizeof(pattern), &pattern, sizeof(pattern),
        /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
  }
  ASSERT_TRUE(HostQueueHasPendingOps(queue));

  // Closing admission before the wait resolves makes deferred issue fail.
  // This uses the normal shutdown contract without cancelling the pending
  // list first, so the issue path owns failure propagation and retirement.
  iree_hal_amdgpu_host_queue_begin_deinitialize(queue);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphores[0], value,
                                           /*frontier=*/NULL));
  EXPECT_FALSE(HostQueueHasPendingOps(queue));
  for (int i = 1; i < 3; ++i) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_CANCELLED,
        iree_hal_semaphore_wait(semaphores[i], value, iree_infinite_timeout(),
                                IREE_ASYNC_WAIT_FLAG_NONE));
    uint64_t observed_value = 0;
    iree_status_t status =
        iree_hal_semaphore_query(semaphores[i], &observed_value);
    EXPECT_TRUE(iree_string_view_equal(iree_status_message(status),
                                       IREE_SV("queue shutting down")));
    IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, status);
  }
}

TEST_F(HostQueuePendingTest, CapacityParkedHostActionRetriesAfterPostDrain) {
  static constexpr uint32_t kAqlCapacity = 64;
  static constexpr uint32_t kNotificationCapacity = 1;
  static constexpr uint32_t kKernargCapacity = 2 * kAqlCapacity;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.host_queues.aql_capacity = kAqlCapacity;
  options.host_queues.notification_capacity = kNotificationCapacity;
  options.host_queues.kernarg_capacity = kKernargCapacity;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_buffer_t> pressure_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), pressure_buffer.out()));

  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(&libhsa_), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/NULL, /*attributes=*/0, &blocker_signal));
  EnqueueRawBlockingBarrier(queue, blocker_signal);

  Ref<iree_hal_semaphore_t> pressure_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), pressure_signal.out()));
  uint64_t pressure_signal_value = 1;
  iree_hal_semaphore_t* pressure_signal_ptr = pressure_signal.get();
  const iree_hal_semaphore_list_t pressure_signal_list =
      MakeSemaphoreList(&pressure_signal_ptr, &pressure_signal_value);
  const uint32_t pressure_pattern = 0xABCD1234u;
  iree_status_t status = iree_hal_queue_fill(
      test_device.queue(), iree_hal_semaphore_list_empty(),
      pressure_signal_list, pressure_buffer,
      /*target_offset=*/0, sizeof(pressure_pattern), &pressure_pattern,
      sizeof(pressure_pattern), /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE);

  HostActionState action_state;
  HostActionStateInitialize(&action_state);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_enqueue_host_action(
        queue, iree_hal_semaphore_list_empty(),
        iree_hal_amdgpu_reclaim_action_t{
            .fn = RecordHostAction,
            .user_data = &action_state,
        },
        /*operation_resources=*/NULL, /*operation_resource_count=*/0,
        (iree_hal_amdgpu_queue_barrier_t){0});
  }
  const bool retry_parked =
      iree_status_is_ok(status) && HostQueueHasPostDrainAction(queue);

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);

  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(pressure_signal, pressure_signal_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    ASSERT_TRUE(iree_notification_await(&action_state.notification,
                                        HostActionStateWasCalled, &action_state,
                                        iree_infinite_timeout()));
  }
  IREE_EXPECT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));

  IREE_ASSERT_OK(status);
  EXPECT_TRUE(retry_parked);
  EXPECT_EQ(HostActionCallCount(&action_state), 1);
  EXPECT_EQ(HostActionStatusCode(&action_state), IREE_STATUS_OK);
  EXPECT_TRUE(HostActionHadEntry(&action_state));

  HostActionStateDeinitialize(&action_state);
}

TEST_F(HostQueuePendingTest,
       QueueAllocaTlsfGrowthCompletesWithoutArtificialExhaustion) {
  const iree_device_size_t allocation_size = 4096;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateExplicitTlsfPool(test_device.device_group(),
                                        iree_hal_queue_family(&queue->base),
                                        allocation_size, pool.out()));

  Ref<iree_hal_semaphore_t> alloca0_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), alloca0_signal.out()));
  uint64_t alloca0_signal_value = 1;
  iree_hal_semaphore_t* alloca0_signal_ptr = alloca0_signal.get();
  const iree_hal_semaphore_list_t alloca0_signal_list =
      MakeSemaphoreList(&alloca0_signal_ptr, &alloca0_signal_value);

  iree_hal_buffer_t* buffer0 = NULL;
  IREE_ASSERT_OK(QueueAlloca(queue, pool, alloca0_signal_list, allocation_size,
                             /*min_alignment=*/0, &buffer0));
  ASSERT_NE(buffer0, nullptr);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(alloca0_signal, alloca0_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  Ref<iree_hal_semaphore_t> alloca1_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), alloca1_signal.out()));
  uint64_t alloca1_signal_value = 1;
  iree_hal_semaphore_t* alloca1_signal_ptr = alloca1_signal.get();
  const iree_hal_semaphore_list_t alloca1_signal_list =
      MakeSemaphoreList(&alloca1_signal_ptr, &alloca1_signal_value);

  iree_hal_buffer_t* buffer1 = NULL;
  IREE_ASSERT_OK(QueueAlloca(queue, pool, alloca1_signal_list, allocation_size,
                             /*min_alignment=*/0, &buffer1));
  ASSERT_NE(buffer1, nullptr);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(alloca1_signal, alloca1_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_FALSE(HostQueueHasPendingOps(queue));

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_GE(stats.slab_count, 2u);
  EXPECT_EQ(stats.exhausted_count, 0u);

  iree_hal_buffer_release(buffer1);
  iree_hal_buffer_release(buffer0);
}

TEST_F(HostQueuePendingTest,
       QueueOpsWaitForPoolBlockedAllocaBackingMaterialization) {
  const iree_device_size_t allocation_size = 4096;
  const uint32_t expected_value = 0xCACE1100u;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_pool_t> native_pool;
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateExplicitFixedBlockPool(
      test_device.device_group(), iree_hal_queue_family(&queue->base),
      allocation_size, native_pool.out(), pool.out()));

  Ref<iree_hal_semaphore_t> alloca0_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), alloca0_signal.out()));
  uint64_t alloca0_signal_value = 1;
  iree_hal_semaphore_t* alloca0_signal_ptr = alloca0_signal.get();
  const iree_hal_semaphore_list_t alloca0_signal_list =
      MakeSemaphoreList(&alloca0_signal_ptr, &alloca0_signal_value);

  iree_hal_buffer_t* buffer0 = NULL;
  IREE_ASSERT_OK(QueueAlloca(queue, pool, alloca0_signal_list, allocation_size,
                             /*min_alignment=*/0, &buffer0));
  ASSERT_NE(buffer0, nullptr);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(alloca0_signal, alloca0_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  Ref<iree_hal_semaphore_t> alloca1_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), alloca1_signal.out()));
  uint64_t alloca1_signal_value = 1;
  iree_hal_semaphore_t* alloca1_signal_ptr = alloca1_signal.get();
  const iree_hal_semaphore_list_t alloca1_signal_list =
      MakeSemaphoreList(&alloca1_signal_ptr, &alloca1_signal_value);

  iree_hal_buffer_t* buffer1 = NULL;
  IREE_ASSERT_OK(QueueAlloca(queue, pool, alloca1_signal_list, allocation_size,
                             /*min_alignment=*/0, &buffer1));
  ASSERT_NE(buffer1, nullptr);
  EXPECT_FALSE(iree_hal_semaphore_list_poll(alloca1_signal_list));
  ASSERT_TRUE(HostQueueHasPendingOps(queue));

  Ref<iree_hal_semaphore_t> update_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), update_signal.out()));
  uint64_t update_signal_value = 1;
  iree_hal_semaphore_t* update_signal_ptr = update_signal.get();
  const iree_hal_semaphore_list_t update_signal_list =
      MakeSemaphoreList(&update_signal_ptr, &update_signal_value);
  IREE_ASSERT_OK(iree_hal_queue_update(
      test_device.queue(), alloca1_signal_list, update_signal_list,
      &expected_value, /*source_offset=*/0, buffer1,
      /*target_offset=*/0, sizeof(expected_value), /*barriers=*/NULL,
      IREE_HAL_UPDATE_FLAG_NONE));

  Ref<iree_hal_buffer_t> readback_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(expected_value), readback_buffer.out()));
  Ref<iree_hal_semaphore_t> copy_signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), copy_signal.out()));
  uint64_t copy_signal_value = 1;
  iree_hal_semaphore_t* copy_signal_ptr = copy_signal.get();
  const iree_hal_semaphore_list_t copy_signal_list =
      MakeSemaphoreList(&copy_signal_ptr, &copy_signal_value);
  IREE_ASSERT_OK(iree_hal_queue_copy(
      test_device.queue(), update_signal_list, copy_signal_list, buffer1,
      /*source_offset=*/0, readback_buffer,
      /*target_offset=*/0, sizeof(expected_value), /*barriers=*/NULL,
      IREE_HAL_COPY_FLAG_NONE));

  Ref<iree_hal_semaphore_t> dealloca0_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), dealloca0_signal.out()));
  uint64_t dealloca0_signal_value = 1;
  iree_hal_semaphore_t* dealloca0_signal_ptr = dealloca0_signal.get();
  const iree_hal_semaphore_list_t dealloca0_signal_list =
      MakeSemaphoreList(&dealloca0_signal_ptr, &dealloca0_signal_value);
  IREE_ASSERT_OK(QueueDealloca(queue, iree_hal_semaphore_list_empty(),
                               dealloca0_signal_list, buffer0));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      dealloca0_signal, dealloca0_signal_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_buffer_release(buffer0);

  IREE_ASSERT_OK(iree_hal_semaphore_wait(copy_signal, copy_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      readback_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_BUFFER_MAP_FLAG_NONE,
      /*byte_offset=*/0, sizeof(expected_value), &mapping));
  uint32_t actual_value = 0;
  memcpy(&actual_value, mapping.contents.data, sizeof(actual_value));
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  EXPECT_EQ(actual_value, expected_value);

  Ref<iree_hal_semaphore_t> dealloca1_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), dealloca1_signal.out()));
  uint64_t dealloca1_signal_value = 1;
  iree_hal_semaphore_t* dealloca1_signal_ptr = dealloca1_signal.get();
  const iree_hal_semaphore_list_t dealloca1_signal_list =
      MakeSemaphoreList(&dealloca1_signal_ptr, &dealloca1_signal_value);
  IREE_ASSERT_OK(
      QueueDealloca(queue, copy_signal_list, dealloca1_signal_list, buffer1));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      dealloca1_signal, dealloca1_signal_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_buffer_release(buffer1);
  EXPECT_FALSE(HostQueueHasPendingOps(queue));
}

TEST_F(HostQueuePendingTest, CancelPendingAllocaFrontierWait) {
  const iree_device_size_t allocation_size = 4096;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_pool_t> native_pool;
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateExplicitFixedBlockPool(
      test_device.device_group(), iree_hal_queue_family(&queue->base),
      allocation_size, native_pool.out(), pool.out()));
  const iree_async_axis_t death_axis = queue->axis;
  IREE_ASSERT_OK(
      SeedWaitableFixedBlockReservation(pool, allocation_size, death_axis));
  queue->wait_barrier_strategy = IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_DEFER;

  Ref<iree_hal_semaphore_t> alloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), alloca_signal.out()));
  uint64_t alloca_signal_value = 1;
  iree_hal_semaphore_t* alloca_signal_ptr = alloca_signal.get();
  const iree_hal_semaphore_list_t alloca_signal_list =
      MakeSemaphoreList(&alloca_signal_ptr, &alloca_signal_value);

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(QueueAlloca(queue, pool, alloca_signal_list, allocation_size,
                             /*min_alignment=*/0, &buffer));
  ASSERT_NE(buffer, nullptr);
  EXPECT_FALSE(iree_hal_semaphore_list_poll(alloca_signal_list));
  ASSERT_TRUE(HostQueueHasPendingOps(queue));

  CancelPendingWithTestStatus(queue);
  EXPECT_FALSE(HostQueueHasPendingOps(queue));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_hal_semaphore_wait(alloca_signal, alloca_signal_value,
                              iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  iree_hal_buffer_release(buffer);
}

TEST_F(HostQueuePendingTest, CancelColdAllocaPreservesInheritedFrontier) {
  constexpr iree_device_size_t kSlabSize = 65536;
  constexpr iree_device_size_t kAllocationSize = 4096;
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  auto* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);
  queue->wait_barrier_strategy = IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_DEFER;

  Ref<iree_hal_pool_t> native_pool;
  Ref<iree_hal_pool_t> backing_pool;
  IREE_ASSERT_OK(CreateExplicitFixedBlockPool(
      test_device.device_group(), iree_hal_queue_family(&queue->base),
      kSlabSize, native_pool.out(), backing_pool.out()));
  IREE_ASSERT_OK(
      SeedWaitableFixedBlockReservation(backing_pool, kSlabSize, queue->axis));
  iree_hal_tlsf_pool_options_t pool_options = {};
  pool_options.tlsf_options.range_length = kSlabSize;
  pool_options.tlsf_options.alignment = 16;
  pool_options.tlsf_options.frontier_capacity = 2;
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create(backing_pool, &pool_options,
                                           host_allocator_, pool.out()));

  iree::hal::cts::SemaphoreList allocated(test_device.base_device(), {0}, {1});
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(QueueAlloca(queue, pool, allocated, kAllocationSize,
                             /*min_alignment=*/0, buffer.out()));
  ASSERT_TRUE(HostQueueHasPendingOps(queue));
  EXPECT_FALSE(iree_hal_semaphore_list_poll(allocated));
  iree_hal_pool_stats_t stats = {};
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reserve_count, 1u);
  EXPECT_EQ(stats.wait_count, 1u);
  EXPECT_EQ(stats.release_count, 0u);
  EXPECT_EQ(stats.reservation_count, 1u);

  CancelPendingWithTestStatus(queue);
  EXPECT_FALSE(HostQueueHasPendingOps(queue));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_hal_semaphore_list_wait(allocated, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.release_count, 1u);
  buffer.reset();

  // Cancellation rolls back bookkeeping without retiring the prior user's
  // dependency. The next allocation must inherit the exact same edge.
  const iree_hal_pool_reservation_request_t request =
      iree_hal_make_pool_reservation_request(kAllocationSize,
                                             /*min_alignment=*/0);
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
      pool, 1, &request, nullptr, &reservation, &info, &result,
      IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  ASSERT_EQ(info.reuse_frontier->entry_count, 1u);
  EXPECT_EQ(info.reuse_frontier->entries[0].axis, queue->axis);
  EXPECT_EQ(info.reuse_frontier->entries[0].epoch, 1u);
  iree_hal_pool_release_reservations(pool, 1, &reservation,
                                     info.reuse_frontier);
}

TEST_F(HostQueuePendingTest, CancelPendingAllocaPoolNotificationWait) {
  const iree_device_size_t allocation_size = 4096;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_pool_t> native_pool;
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateExplicitFixedBlockPool(
      test_device.device_group(), iree_hal_queue_family(&queue->base),
      allocation_size, native_pool.out(), pool.out()));

  Ref<iree_hal_semaphore_t> alloca0_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), alloca0_signal.out()));
  uint64_t alloca0_signal_value = 1;
  iree_hal_semaphore_t* alloca0_signal_ptr = alloca0_signal.get();
  const iree_hal_semaphore_list_t alloca0_signal_list =
      MakeSemaphoreList(&alloca0_signal_ptr, &alloca0_signal_value);

  iree_hal_buffer_t* buffer0 = NULL;
  IREE_ASSERT_OK(QueueAlloca(queue, pool, alloca0_signal_list, allocation_size,
                             /*min_alignment=*/0, &buffer0));
  ASSERT_NE(buffer0, nullptr);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(alloca0_signal, alloca0_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  Ref<iree_hal_semaphore_t> alloca1_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), alloca1_signal.out()));
  uint64_t alloca1_signal_value = 1;
  iree_hal_semaphore_t* alloca1_signal_ptr = alloca1_signal.get();
  const iree_hal_semaphore_list_t alloca1_signal_list =
      MakeSemaphoreList(&alloca1_signal_ptr, &alloca1_signal_value);

  iree_hal_buffer_t* buffer1 = NULL;
  IREE_ASSERT_OK(QueueAlloca(queue, pool, alloca1_signal_list, allocation_size,
                             /*min_alignment=*/0, &buffer1));
  ASSERT_NE(buffer1, nullptr);
  EXPECT_FALSE(iree_hal_semaphore_list_poll(alloca1_signal_list));
  ASSERT_TRUE(HostQueueHasPendingOps(queue));

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_GE(stats.exhausted_count, 1u);

  auto* notification = iree_hal_pool_notification(pool);
  const uint32_t epoch = iree_async_notification_begin_observe(notification);
  CancelPendingWithTestStatus(queue);
  EXPECT_EQ(iree_async_notification_query_epoch(notification), epoch);
  iree_async_notification_end_observe(notification);
  EXPECT_FALSE(HostQueueHasPendingOps(queue));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_hal_semaphore_wait(alloca1_signal, alloca1_signal_value,
                              iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_release(buffer1);
  iree_hal_buffer_release(buffer0);
}

}  // namespace
}  // namespace iree::hal::amdgpu
