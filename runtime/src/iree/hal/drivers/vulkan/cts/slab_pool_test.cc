// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "iree/async/notification.h"
#include "iree/hal/cts/util/pool_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/vulkan/queue_test_util.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"

namespace iree::hal::cts {
namespace {

struct ResourceRefCountObservation {
  // Borrowed resource kept live by the test until observation completes.
  iree_hal_resource_t* resource = nullptr;
  // Result published by the terminal semaphore callback.
  std::promise<uint32_t> result;
};

static void ObserveResourceRefCount(void* user_data,
                                    iree_async_semaphore_timepoint_t* timepoint,
                                    iree_status_t status) {
  (void)timepoint;
  auto* observation = static_cast<ResourceRefCountObservation*>(user_data);
  const uint32_t ref_count =
      iree_atomic_ref_count_load(&observation->resource->ref_count);
  iree_status_free(status);
  observation->result.set_value(ref_count);
}

// Holds one real device allocator call from its native allocation worker.
// Native Vulkan objects, the slab provider, and queue operations remain real.
struct NativeAllocationGate {
  // Synchronizes the hold and final allocator-lifetime observation.
  std::mutex mutex;
  // Publishes entry, release, and complete native-owner destruction.
  std::condition_variable condition;
  // Caller thread, whose queue capture allocations are never held.
  std::thread::id caller = std::this_thread::get_id();
  // Allocator command selected for the next non-caller operation.
  iree_allocator_command_t command = IREE_ALLOCATOR_COMMAND_MALLOC;
  // One matching dependency will stop before entering the system allocator.
  bool armed = false;
  // The selected native dependency reached the hold.
  bool entered = false;
  // The test released that dependency.
  bool released = false;
  // Live allocations belonging to the fresh device and its native worker.
  size_t allocation_count = 0;

  static iree_status_t Control(void* user_data,
                               iree_allocator_command_t command,
                               const void* params, void** inout_pointer) {
    auto* gate = static_cast<NativeAllocationGate*>(user_data);
    const bool was_allocated = *inout_pointer != nullptr;
    {
      std::unique_lock<std::mutex> lock(gate->mutex);
      if (gate->armed && gate->command == command &&
          std::this_thread::get_id() != gate->caller) {
        gate->armed = false;
        gate->entered = true;
        gate->condition.notify_all();
        gate->condition.wait(lock, [&] { return gate->released; });
      }
    }
    const auto allocator = iree_allocator_system();
    iree_status_t status =
        allocator.ctl(allocator.self, command, params, inout_pointer);
    std::lock_guard<std::mutex> lock(gate->mutex);
    if (iree_status_is_ok(status)) {
      if (command == IREE_ALLOCATOR_COMMAND_FREE) {
        gate->allocation_count -= was_allocated ? 1 : 0;
      } else if (!was_allocated) {
        ++gate->allocation_count;
      }
    }
    gate->condition.notify_all();
    return status;
  }

  iree_allocator_t allocator() { return {this, Control}; }

  void Arm(iree_allocator_command_t held_command) {
    std::lock_guard<std::mutex> lock(mutex);
    command = held_command;
    entered = released = false;
    armed = true;
  }

  void WaitEntered() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return entered; });
  }

  void Release() {
    std::lock_guard<std::mutex> lock(mutex);
    armed = false;
    released = true;
    condition.notify_all();
  }

  void WaitEmpty() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return allocation_count == 0; });
  }
};

struct ScopedNativeAllocationHold {
  ScopedNativeAllocationHold(NativeAllocationGate& gate,
                             iree_allocator_command_t command)
      : gate(gate) {
    gate.Arm(command);
  }
  ~ScopedNativeAllocationHold() { gate.Release(); }
  // Dependency whose hold always ends before captured device resources unwind.
  NativeAllocationGate& gate;
};

class VulkanSlabPoolTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    const auto* queues =
        iree_hal_device_spec_queues(iree_hal_device_spec(device_));
    for (iree_host_size_t i = 0; i < queues->family_count; ++i) {
      if (!queues->families[i].provisioned_queue_count) {
        continue;
      }
      auto* queue = iree_hal_device_queue(
          device_, static_cast<iree_hal_queue_family_ordinal_t>(i), 0);
      if (!queue ||
          !iree_any_bit_set(queues->families[i].role_flags,
                            IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER)) {
        continue;
      }
      queues_.push_back(queue);
      iree_hal_pool_family_access_t access = {
          .family = iree_hal_queue_family(queue),
          .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
      };
      families_.push_back(access);
    }
    ASSERT_FALSE(families_.empty());
  }

  iree_hal_pool_scope_t Scope() const {
    iree_hal_pool_scope_t scope = {
        .family_count = families_.size(),
        .families = families_.data(),
    };
    return scope;
  }

  iree_status_t CreateSource(iree_hal_pool_scope_t scope,
                             iree_hal_pool_t** out_pool) {
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    return iree_hal_slab_pool_create(device_group_, scope, &options,
                                     iree_allocator_system(), out_pool);
  }

  void Wait(iree_hal_semaphore_list_t semaphores) {
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        semaphores, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  void WaitCommittedBytes(iree_hal_pool_t* pool, iree_device_size_t expected) {
    auto* notification = iree_hal_pool_notification(pool);
    for (;;) {
      const uint32_t token =
          iree_async_notification_begin_observe(notification);
      iree_hal_pool_stats_t stats;
      iree_hal_pool_query_stats(pool, &stats);
      if (stats.bytes_committed == expected) {
        iree_async_notification_end_observe(notification);
        return;
      }
      const bool signaled = iree_async_notification_wait_for_token(
          notification, token, iree_infinite_timeout());
      iree_async_notification_end_observe(notification);
      ASSERT_TRUE(signaled);
    }
  }

  // Borrowed provisioned transfer queues with distinct canonical families.
  std::vector<iree_hal_queue_t*> queues_;
  // Construction-time grants for each queue's exact family.
  std::vector<iree_hal_pool_family_access_t> families_;
};

TEST_P(VulkanSlabPoolTest, CapturesExactFamiliesWithoutAllocatingBacking) {
  const auto all_families = families_;
  families_.resize(1);
  families_[0].usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET;
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.reserve_count, 0u);
  EXPECT_EQ(stats.bytes_committed, 0u);
  EXPECT_EQ(iree_hal_pool_query_host_access(source).access,
            IREE_HAL_MEMORY_ACCESS_NONE);
  for (size_t i = 0; i < all_families.size(); ++i) {
    for (auto kind :
         {IREE_HAL_MEMORY_SITE_QUEUE, IREE_HAL_MEMORY_SITE_PROGRAM}) {
      const iree_hal_memory_site_t site = {kind, all_families[i].family};
      iree_hal_memory_scope_t scope;
      IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(device_group_,
                                                                site, &scope));
      for (auto interface : {IREE_HAL_BUFFER_INTERFACE_VULKAN_BUFFER,
                             IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS}) {
        iree_hal_buffer_native_binding_slot_t slot;
        if (i == 0) {
          IREE_ASSERT_OK(
              iree_hal_pool_resolve_binding(source, scope, interface, &slot));
          EXPECT_NE(slot.index, IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE);
        } else {
          IREE_EXPECT_STATUS_IS(
              IREE_STATUS_PERMISSION_DENIED,
              iree_hal_pool_resolve_binding(source, scope, interface, &slot));
        }
      }
    }
  }

  Ref<iree_hal_buffer_t> buffer;
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(source, &capabilities);
  const iree_device_size_t min_alignment =
      iree_min((iree_device_size_t)256, capabilities.max_allocation_alignment);
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source,
      iree_hal_make_pool_reservation_request(sizeof(uint32_t), min_alignment),
      iree_infinite_timeout(), buffer.out()));
  EXPECT_EQ(iree_hal_buffer_allowed_usage(buffer),
            IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET);
  const uint32_t value = 0xAABBCCDD;
  SemaphoreList filled(device_, {0}, {1});
  for (size_t i = 1; i < queues_.size(); ++i) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_PERMISSION_DENIED,
        iree_hal_queue_fill(queues_[i], iree_hal_semaphore_list_empty(), filled,
                            buffer, 0, sizeof(value), &value, sizeof(value),
                            /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
  }
  IREE_ASSERT_OK(
      iree_hal_queue_fill(queues_[0], iree_hal_semaphore_list_empty(), filled,
                          buffer, 0, sizeof(value), &value, sizeof(value),
                          /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
  Wait(filled);
  uint32_t result = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_download(queues_[0], filled,
                              iree_hal_semaphore_list_empty(), buffer, 0,
                              &result, sizeof(result), /*barriers=*/NULL));
  buffer.reset();
}

TEST_P(VulkanSlabPoolTest, NativeAllocationDoesNotBlockCachedWorkOrRetirement) {
  NativeAllocationGate native;
  DeviceCreateContext context;
  IREE_ASSERT_OK(context.Initialize(iree_allocator_system()));
  Ref<iree_hal_device_t> device;
  Ref<iree_hal_device_group_t> group;
  IREE_ASSERT_OK(iree_hal_driver_create_default_device(
      driver_, context.params(), native.allocator(), device.out()));
  IREE_ASSERT_OK(iree_hal_device_group_create_from_device(
      device, context.frontier_tracker(), iree_allocator_system(),
      group.out()));
  auto* queue = iree_hal_device_queue(
      device, iree_hal_queue_family_ordinal(iree_hal_queue_family(queues_[0])),
      0);
  ASSERT_NE(queue, nullptr);
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateQueueSlabPool(
      group, iree_hal_queue_family(queue), IREE_HAL_BUFFER_USAGE_TRANSFER,
      /*host_access=*/{}, iree_allocator_system(), source.out()));
  Ref<iree_hal_pool_t> finite;
  const iree_hal_fixed_block_pool_options_t options = {
      .block_size = 4096,
      .blocks_per_slab = 1,
      .frontier_capacity = 2,
  };
  IREE_ASSERT_OK(CreateFiniteBlockPool(source, options, iree_allocator_system(),
                                       finite.out()));
  const auto request = iree_hal_make_pool_reservation_request(4096, 0);
  Ref<iree_hal_buffer_t> growing;
  {
    SemaphoreList allocated(device, {0}, {1});
    {
      ScopedNativeAllocationHold hold(native, IREE_ALLOCATOR_COMMAND_CALLOC);
      IREE_ASSERT_OK(
          iree_hal_queue_alloca(queue, iree_hal_semaphore_list_empty(),
                                allocated, source, 1, &request, growing.out()));
      native.WaitEntered();
      EXPECT_FALSE(iree_hal_semaphore_list_poll(allocated));
      Ref<iree_hal_buffer_t> cached;
      SemaphoreList cached_ready(device, {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_alloca(
          queue, iree_hal_semaphore_list_empty(), cached_ready, finite, 1,
          &request, cached.out()));
      EXPECT_TRUE(iree_hal_semaphore_list_poll(cached_ready));
      const uint32_t pattern = 0xC01DCAFE;
      SemaphoreList filled(device, {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_fill(
          queue, cached_ready, filled, cached, 0, sizeof(pattern), &pattern,
          sizeof(pattern), nullptr, IREE_HAL_FILL_FLAG_NONE));
      uint32_t result = 0;
      SemaphoreList downloaded(device, {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_download(queue, filled, downloaded, cached,
                                             0, &result, sizeof(result),
                                             nullptr));
      Wait(downloaded);
      EXPECT_EQ(result, pattern);
      SemaphoreList returned(device, {0}, {1});
      auto* buffer = cached.get();
      IREE_ASSERT_OK(
          iree_hal_queue_dealloca(queue, downloaded, returned, 1, &buffer));
      EXPECT_TRUE(iree_hal_semaphore_list_poll(returned));
      EXPECT_FALSE(iree_hal_semaphore_list_poll(allocated));
    }
    Wait(allocated);
  }
  finite.reset();
  {
    ScopedNativeAllocationHold hold(native, IREE_ALLOCATOR_COMMAND_FREE);
    growing.reset();
    native.WaitEntered();
    // Pool return stays asynchronous. Final device destruction is the native
    // ownership join, so it cannot finish while cleanup remains held.
    source.reset();
    group.reset();
    iree_hal_device_t* retiring = device.get();
    iree_hal_device_retain(retiring);
    device.reset();
    std::promise<void> started;
    std::promise<void> destroyed;
    auto destroyed_future = destroyed.get_future();
    std::thread teardown([&] {
      started.set_value();
      iree_hal_device_release(retiring);
      destroyed.set_value();
    });
    started.get_future().get();
    EXPECT_EQ(destroyed_future.wait_for(std::chrono::seconds(0)),
              std::future_status::timeout);
    native.Release();
    teardown.join();
    destroyed_future.get();
  }
  native.WaitEmpty();
}

TEST_P(VulkanSlabPoolTest, PendingAllocationDoesNotBlockIndependentTransfer) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  Ref<iree_hal_buffer_t> resident;
  const auto request = iree_hal_make_pool_reservation_request(4096, 0);
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, request, iree_infinite_timeout(), resident.out()));
  iree_hal_pool_stats_t before;
  iree_hal_pool_query_stats(source, &before);

  SemaphoreList gate(device_, {0}, {1});
  SemaphoreList allocated(device_, {0}, {1});
  Ref<iree_hal_buffer_t> pending;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], gate, allocated, source, 1,
                                       &request, pending.out()));
  iree_hal_pool_stats_t blocked;
  iree_hal_pool_query_stats(source, &blocked);
  EXPECT_EQ(blocked.reserve_count, before.reserve_count);
  EXPECT_EQ(blocked.bytes_committed, before.bytes_committed);

  constexpr uint32_t pattern = 0xACCE551B;
  SemaphoreList filled(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_fill(
      queues_[0], iree_hal_semaphore_list_empty(), filled, resident, 0, 4096,
      &pattern, sizeof(pattern), nullptr, IREE_HAL_FILL_FLAG_NONE));
  Wait(filled);
  EXPECT_EQ(ReadBufferData<uint32_t>(resident)[0], pattern);
  EXPECT_FALSE(iree_hal_semaphore_list_poll(allocated));

  IREE_ASSERT_OK(iree_hal_semaphore_list_signal(gate, nullptr));
  Wait(allocated);
  SemaphoreList returned(device_, {0}, {1});
  auto* buffer = pending.get();
  IREE_ASSERT_OK(
      iree_hal_queue_dealloca(queues_[0], allocated, returned, 1, &buffer));
  Wait(returned);
}

TEST_P(VulkanSlabPoolTest, TransfersResolveNestedViewsAfterAllocation) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  const auto request = iree_hal_make_pool_reservation_request(512, 0);
  for (iree_device_size_t update_offset : {0, 1}) {
    SCOPED_TRACE(update_offset);
    SemaphoreList gate(device_, {0}, {1});
    SemaphoreList allocated(device_, {0}, {1});
    Ref<iree_hal_buffer_t> allocation;
    IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], gate, allocated, source, 1,
                                         &request, allocation.out()));
    Ref<iree_hal_buffer_t> outer;
    Ref<iree_hal_buffer_t> view;
    IREE_ASSERT_OK(iree_hal_buffer_subspan(
        allocation, 8, 264, iree_allocator_system(), outer.out()));
    IREE_ASSERT_OK(iree_hal_buffer_subspan(
        outer, 4, 256, iree_allocator_system(), view.out()));
    outer.reset();
    Ref<iree_hal_buffer_t> destination;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        source, request, iree_infinite_timeout(), destination.out()));
    const uint8_t pattern = 0xCD;
    const auto update = MakeDeterministicBytes(16);
    SemaphoreList filled(device_, {0}, {1});
    SemaphoreList updated(device_, {0}, {1});
    SemaphoreList copied(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], allocated, filled, view, 0,
                                       256, &pattern, sizeof(pattern), nullptr,
                                       IREE_HAL_FILL_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_queue_update(
        queues_[0], filled, updated, update.data(), 0, view, update_offset,
        update.size(), nullptr, IREE_HAL_UPDATE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_queue_copy(queues_[0], updated, copied, view, 0,
                                       destination, 0, 256, nullptr,
                                       IREE_HAL_COPY_FLAG_NONE));
    EXPECT_FALSE(iree_hal_semaphore_list_poll(copied));
    IREE_ASSERT_OK(iree_hal_semaphore_list_signal(gate, nullptr));
    Wait(copied);
    std::vector<uint8_t> expected(256, pattern);
    std::copy(update.begin(), update.end(), expected.begin() + update_offset);
    EXPECT_EQ(ReadBufferBytes(destination, 0, 256), expected);
    view.reset();
    SemaphoreList returned(device_, {0}, {1});
    auto* buffer = allocation.get();
    IREE_ASSERT_OK(
        iree_hal_queue_dealloca(queues_[0], copied, returned, 1, &buffer));
    Wait(returned);
  }
}

TEST_P(VulkanSlabPoolTest, AllocationFailureCancelsUnsignaledPrerequisites) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  const auto request = iree_hal_make_pool_reservation_request(4096, 0);
  SemaphoreList gates(device_, {0, 0}, {1, 1});
  SemaphoreList allocated(device_, {0}, {1});
  Ref<iree_hal_buffer_t> pending;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], gates, allocated, source, 1,
                                       &request, pending.out()));
  iree_hal_semaphore_fail(gates.semaphores[0],
                          iree_status_from_code(IREE_STATUS_ABORTED));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_semaphore_list_wait(allocated, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE));
  pending.reset();
  source.reset();
  // The unused prerequisite no longer holds an operation callback.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gates.semaphores[1], 1, nullptr));
}

TEST_P(VulkanSlabPoolTest, QueueRetirementCancelsPendingAllocation) {
  DeviceCreateContext context;
  IREE_ASSERT_OK(context.Initialize(iree_allocator_system()));
  Ref<iree_hal_device_group_t> group;
  Ref<iree_hal_device_t> device;
  IREE_ASSERT_OK(iree_hal_driver_create_default_device(
      driver_, context.params(), iree_allocator_system(), device.out()));
  IREE_ASSERT_OK(iree_hal_device_group_create_from_device(
      device, context.frontier_tracker(), iree_allocator_system(),
      group.out()));
  auto* queue = iree_hal_device_queue(
      device, iree_hal_queue_family_ordinal(iree_hal_queue_family(queues_[0])),
      0);
  ASSERT_NE(queue, nullptr);
  const iree_hal_pool_family_access_t access = {
      .family = iree_hal_queue_family(queue),
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
  };
  const iree_hal_pool_scope_t scope = {1, &access, {}};
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group, scope, &options, iree_allocator_system(), source.out()));
  const auto request = iree_hal_make_pool_reservation_request(4096, 0);
  SemaphoreList gate(device, {0}, {1});
  SemaphoreList allocated(device, {0}, {1});
  Ref<iree_hal_buffer_t> pending;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queue, gate, allocated, source, 1,
                                       &request, pending.out()));

  iree_hal_fixed_block_pool_options_t block_options = {
      .block_size = 4096,
      .blocks_per_slab = 1,
      .frontier_capacity = 2,
      .budget_limit = 4096,
  };
  Ref<iree_hal_pool_t> blocks;
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create(
      source, &block_options, iree_allocator_system(), blocks.out()));
  Ref<iree_hal_buffer_t> occupied;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      blocks, request, iree_infinite_timeout(), occupied.out()));
  SemaphoreList capacity_ready(device, {0}, {1});
  Ref<iree_hal_buffer_t> capacity_pending;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queue, iree_hal_semaphore_list_empty(),
                                       capacity_ready, blocks, 1, &request,
                                       capacity_pending.out()));
  EXPECT_FALSE(iree_hal_semaphore_list_poll(capacity_ready));

  RetireVulkanQueue(queue);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_hal_semaphore_list_wait(allocated, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_hal_semaphore_list_wait(capacity_ready, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE));
  capacity_pending.reset();
  occupied.reset();
  blocks.reset();
  pending.reset();
  source.reset();
  IREE_ASSERT_OK(iree_hal_semaphore_list_signal(gate, nullptr));
}

TEST_P(VulkanSlabPoolTest, ConcurrentDependencyFailureRetiresCapture) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  const auto request = iree_hal_make_pool_reservation_request(4096, 0);
  for (int i = 0; i < 64; ++i) {
    SemaphoreList gates(device_, {0, 0}, {1, 1});
    SemaphoreList allocated(device_, {0}, {1});
    Ref<iree_hal_buffer_t> pending;
    IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], gates, allocated, source,
                                         1, &request, pending.out()));
    std::thread failure([&] {
      iree_hal_semaphore_fail(gates.semaphores[0],
                              iree_status_from_code(IREE_STATUS_ABORTED));
    });
    std::thread success([&] {
      IREE_EXPECT_OK(
          iree_hal_semaphore_signal(gates.semaphores[1], 1, nullptr));
    });
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_ABORTED,
        iree_hal_semaphore_list_wait(allocated, iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE));
    pending.reset();
    success.join();
    failure.join();
  }
}

TEST_P(VulkanSlabPoolTest, FailedDeallocationLeavesAllocationLive) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  const auto request = iree_hal_make_pool_reservation_request(4096, 0);
  SemaphoreList allocated(device_, {0}, {1});
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(
      iree_hal_queue_alloca(queues_[0], iree_hal_semaphore_list_empty(),
                            allocated, source, 1, &request, buffer.out()));
  Wait(allocated);
  SemaphoreList gates(device_, {0, 0}, {1, 1});
  SemaphoreList failed_return(device_, {0}, {1});
  auto* raw_buffer = buffer.get();
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[0], gates, failed_return, 1,
                                         &raw_buffer));
  iree_hal_semaphore_fail(gates.semaphores[0],
                          iree_status_from_code(IREE_STATUS_ABORTED));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_semaphore_list_wait(failed_return, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE));
  constexpr uint32_t pattern = 0xA110CA;
  SemaphoreList filled(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], allocated, filled, buffer, 0,
                                     4096, &pattern, sizeof(pattern), nullptr,
                                     IREE_HAL_FILL_FLAG_NONE));
  Wait(filled);
  EXPECT_EQ(ReadBufferData<uint32_t>(buffer)[0], pattern);
  SemaphoreList returned(device_, {0}, {1});
  IREE_ASSERT_OK(
      iree_hal_queue_dealloca(queues_[0], filled, returned, 1, &raw_buffer));
  Wait(returned);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gates.semaphores[1], 1, nullptr));
}

TEST_P(VulkanSlabPoolTest, QueueReleasesBuffersBeforeTerminalSignal) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  for (bool fail_dependency : {false, true}) {
    SCOPED_TRACE(fail_dependency);
    Ref<iree_hal_buffer_t> buffer;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        source, iree_hal_make_pool_reservation_request(sizeof(uint32_t), 0),
        iree_infinite_timeout(), buffer.out()));

    SemaphoreList ready(device_, {0}, {1});
    SemaphoreList completed(device_, {0}, {1});
    const uint32_t pattern = 0xA5A5A5A5u;
    IREE_ASSERT_OK(iree_hal_queue_fill(
        queues_[0], ready, completed, buffer, 0, sizeof(pattern), &pattern,
        sizeof(pattern), /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));

    ResourceRefCountObservation observation = {&buffer.get()->resource};
    std::future<uint32_t> observed_ref_count = observation.result.get_future();
    iree_async_semaphore_timepoint_t timepoint = {
        .callback = ObserveResourceRefCount,
        .user_data = &observation,
    };
    IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
        reinterpret_cast<iree_async_semaphore_t*>(completed.semaphores[0]), 1,
        &timepoint));
    if (fail_dependency) {
      iree_hal_semaphore_fail(ready.semaphores[0],
                              iree_status_from_code(IREE_STATUS_CANCELLED));
    } else {
      IREE_ASSERT_OK(iree_hal_semaphore_signal(ready.semaphores[0], 1, NULL));
    }

    EXPECT_EQ(observed_ref_count.get(), 1u);
    if (fail_dependency) {
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_CANCELLED,
          iree_hal_semaphore_list_wait(completed, iree_infinite_timeout(),
                                       IREE_ASYNC_WAIT_FLAG_NONE));
    } else {
      Wait(completed);
    }
    EXPECT_EQ(iree_atomic_ref_count_load(&buffer.get()->resource.ref_count),
              1u);
    buffer.reset();
  }
}

TEST_P(VulkanSlabPoolTest,
       NativeReplayDoesNotRetainStaticBufferCommandBuffers) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, iree_hal_make_pool_reservation_request(sizeof(uint32_t), 0),
      iree_infinite_timeout(), buffer.out()));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(queues_[0]), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, /*binding_capacity=*/0,
      command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  const uint32_t pattern = 0x5A5A5A5Au;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_buffer_ref(buffer, /*offset=*/0, sizeof(pattern)), &pattern,
      sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  SemaphoreList ready(device_, {0}, {1});
  SemaphoreList completed(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_execute(
      queues_[0], ready, completed, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));

  ResourceRefCountObservation observation = {&command_buffer.get()->resource};
  std::future<uint32_t> observed_ref_count = observation.result.get_future();
  iree_async_semaphore_timepoint_t timepoint = {
      .callback = ObserveResourceRefCount,
      .user_data = &observation,
  };
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      reinterpret_cast<iree_async_semaphore_t*>(completed.semaphores[0]), 1,
      &timepoint));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(ready.semaphores[0], 1, NULL));

  EXPECT_EQ(observed_ref_count.get(), 1u);
  Wait(completed);
  command_buffer.reset();
  EXPECT_EQ(iree_atomic_ref_count_load(&buffer.get()->resource.ref_count), 1u);
  buffer.reset();
}

TEST_P(VulkanSlabPoolTest, RequiredAndPreferredHostGrants) {
  for (auto access : {IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_MEMORY_ACCESS_WRITE,
                      IREE_HAL_MEMORY_ACCESS_ALL}) {
    for (auto modes :
         {IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MAPPING_MODE_PERSISTENT}) {
      SCOPED_TRACE(::testing::Message()
                   << "access=" << access << " modes=" << modes);
      auto scope = Scope();
      scope.host.access = access;
      scope.host.modes = modes;
      Ref<iree_hal_pool_t> source;
      IREE_ASSERT_OK(CreateSource(scope, source.out()));
      const auto host = iree_hal_pool_query_host_access(source);
      EXPECT_EQ(host.access, access);
      EXPECT_EQ(host.modes, modes);
      iree_hal_pool_capabilities_t capabilities;
      iree_hal_pool_query_capabilities(source, &capabilities);
      ASSERT_GE(capabilities.max_allocation_alignment, 256u);
      Ref<iree_hal_buffer_t> buffer;
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          source, iree_hal_make_pool_reservation_request(sizeof(uint32_t), 256),
          iree_infinite_timeout(), buffer.out()));
      const auto memory = iree_hal_buffer_memory_view(buffer);
      ASSERT_NE(memory.backing, nullptr);
      EXPECT_GE(memory.backing->allocation_alignment, 256u);
      const uint32_t value = 0x1234ABCD;
      SemaphoreList filled(device_, {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_fill(
          queues_[0], iree_hal_semaphore_list_empty(), filled, buffer, 0,
          sizeof(value), &value, sizeof(value), /*barriers=*/NULL,
          IREE_HAL_FILL_FLAG_NONE));
      Wait(filled);
      iree_hal_buffer_mapping_t mapping = {};
      for (auto denied :
           {IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_MEMORY_ACCESS_WRITE}) {
        if (iree_any_bit_set(access, denied)) {
          continue;
        }
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_PERMISSION_DENIED,
            iree_hal_buffer_map_range(buffer, modes, denied,
                                      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0,
                                      sizeof(value), &mapping));
      }
      IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, modes, access,
                                               IREE_HAL_BUFFER_MAP_FLAG_NONE, 0,
                                               sizeof(value), &mapping));
      if (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_READ)) {
        IREE_ASSERT_OK(iree_hal_buffer_mapping_invalidate_range(&mapping, 0,
                                                                sizeof(value)));
        uint32_t result = 0;
        std::memcpy(&result, mapping.contents.data, sizeof(result));
        EXPECT_EQ(result, value);
      }
      if (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
        std::memcpy(mapping.contents.data, &value, sizeof(value));
        IREE_ASSERT_OK(
            iree_hal_buffer_mapping_flush_range(&mapping, 0, sizeof(value)));
      }
      IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
      buffer.reset();
    }
  }

  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  options.preferences.host.access = IREE_HAL_MEMORY_ACCESS_READ;
  options.preferences.host.modes = IREE_HAL_MAPPING_MODE_SCOPED;
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      device_group_, Scope(), &options, iree_allocator_system(), source.out()));
  EXPECT_EQ(iree_hal_pool_query_host_access(source).access,
            IREE_HAL_MEMORY_ACCESS_READ);
}

TEST_P(VulkanSlabPoolTest, CachedHostAccessIsAnExactNativeClass) {
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  options.preferences.host.access = IREE_HAL_MEMORY_ACCESS_READ;
  options.preferences.host.modes = IREE_HAL_MAPPING_MODE_SCOPED;
  options.preferences.host.cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK;
  Ref<iree_hal_pool_t> preferred;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(device_group_, Scope(), &options,
                                           iree_allocator_system(),
                                           preferred.out()));
  const auto host = iree_hal_pool_query_host_access(preferred);
  auto scope = Scope();
  scope.host = options.preferences.host;
  Ref<iree_hal_pool_t> required;
  if (!host.access) {
    // A native owner without cached host memory cannot promise that class.
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                          CreateSource(scope, required.out()));
    EXPECT_FALSE(required);
    return;
  }
  EXPECT_EQ(host.cacheability, IREE_HAL_HOST_CACHEABILITY_WRITE_BACK);
  IREE_ASSERT_OK(CreateSource(scope, required.out()));
  EXPECT_EQ(iree_hal_pool_query_host_access(required).cacheability,
            IREE_HAL_HOST_CACHEABILITY_WRITE_BACK);
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      required, iree_hal_make_pool_reservation_request(sizeof(uint32_t), 0),
      iree_infinite_timeout(), buffer.out()));
  const uint32_t value = 0xABCD1234;
  SemaphoreList filled(device_, {0}, {1});
  IREE_ASSERT_OK(
      iree_hal_queue_fill(queues_[0], iree_hal_semaphore_list_empty(), filled,
                          buffer, 0, sizeof(value), &value, sizeof(value),
                          /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
  Wait(filled);
  iree_hal_buffer_mapping_t mapping = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, sizeof(value), &mapping));
  IREE_ASSERT_OK(
      iree_hal_buffer_mapping_invalidate_range(&mapping, 0, sizeof(value)));
  uint32_t result = 0;
  std::memcpy(&result, mapping.contents.data, sizeof(result));
  EXPECT_EQ(result, value);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  buffer.reset();
}

TEST_P(VulkanSlabPoolTest, ImportsHostStorageThroughSelectedNativeRoute) {
  const iree_hal_pool_import_class_t import_class = {
      .storage_class = iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
  };
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  options.owned.mode = IREE_HAL_SLAB_POOL_IMPORT_ONLY;
  options.imports.class_count = 1;
  options.imports.classes = &import_class;
  auto scope = Scope();
  scope.host.access = IREE_HAL_MEMORY_ACCESS_ALL;
  scope.host.modes = IREE_HAL_MAPPING_MODE_SCOPED;
  Ref<iree_hal_pool_t> source;
  iree_status_t create_status = iree_hal_slab_pool_create(
      device_group_, scope, &options, iree_allocator_system(), source.out());
  if (iree_status_is_unavailable(create_status)) {
    iree::Status unavailable_status(std::move(create_status));
    GTEST_SKIP() << unavailable_status.ToString();
  }
  IREE_ASSERT_OK(create_status);

  iree_hal_pool_import_requirements_t requirements;
  ASSERT_TRUE(iree_hal_pool_query_import_requirements(
      source, import_class.storage_class, &requirements));
  const iree_device_size_t allocation_size = iree_device_align(
      std::max<iree_device_size_t>(requirements.min_byte_length, 4096),
      requirements.byte_length_granularity);
  const iree_device_size_t allocation_alignment =
      std::max<iree_device_size_t>(requirements.base_alignment, 4096);
  const iree_device_size_t storage_length =
      allocation_alignment + allocation_size;
  ASSERT_LE(storage_length, IREE_HOST_SIZE_MAX);
  auto allocation = std::make_unique<ImportedHostAllocation>();
  allocation->host_allocator = iree_allocator_system();
  allocation->storage_class = import_class.storage_class;
  std::future<ImportedHostAllocationReleaseResult> release_result =
      allocation->release_result.get_future();
  IREE_ASSERT_OK(iree_allocator_malloc_aligned(
      allocation->host_allocator, static_cast<iree_host_size_t>(storage_length),
      static_cast<iree_host_size_t>(allocation_alignment), /*offset=*/0,
      &allocation->pointer));
  void* host_pointer = allocation->pointer;
  std::memset(host_pointer, 0, static_cast<size_t>(storage_length));

  iree_hal_external_storage_t storage = {
      .storage_class = import_class.storage_class,
      .byte_length = storage_length,
  };
  storage.payload.host_pointer = host_pointer;
  storage.host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK;
  storage.maintenance_range =
      iree_hal_make_buffer_range(allocation_alignment, allocation_size);
  storage.release = ReleaseImportedHostAllocation;
  storage.release_user_data = allocation.get();
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_import_buffer(
      source, &storage,
      iree_hal_make_buffer_range(allocation_alignment, allocation_size),
      buffer.out()));
  allocation.release();
  EXPECT_EQ(storage.storage_class.type, IREE_HAL_EXTERNAL_STORAGE_TYPE_NONE);

  const uint32_t value = 0xABCD1234;
  SemaphoreList filled(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_fill(
      queues_.front(), iree_hal_semaphore_list_empty(), filled, buffer, 0,
      sizeof(value), &value, sizeof(value), /*barriers=*/NULL,
      IREE_HAL_FILL_FLAG_NONE));
  Wait(filled);
  iree_hal_buffer_mapping_t mapping = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, sizeof(value), &mapping));
  IREE_ASSERT_OK(
      iree_hal_buffer_mapping_invalidate_range(&mapping, 0, sizeof(value)));
  uint32_t result = 0;
  std::memcpy(&result, mapping.contents.data, sizeof(result));
  EXPECT_EQ(result, value);
  EXPECT_EQ(mapping.contents.data,
            static_cast<uint8_t*>(host_pointer) + allocation_alignment);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  buffer.reset();
  const ImportedHostAllocationReleaseResult released = release_result.get();
  EXPECT_TRUE(released.storage_class_matches);
  EXPECT_TRUE(released.payload_matches);
}

TEST_P(VulkanSlabPoolTest, NativeOnlyAndHostOnlyDoNotGrantQueueOperations) {
  for (auto& access : families_) {
    access.usage = IREE_HAL_BUFFER_USAGE_NONE;
  }
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, iree_hal_make_pool_reservation_request(64, 0),
      iree_infinite_timeout(), buffer.out()));
  EXPECT_EQ(iree_hal_buffer_allowed_usage(buffer), IREE_HAL_BUFFER_USAGE_NONE);
  iree_hal_memory_scope_t scope;
  const iree_hal_memory_site_t site = {IREE_HAL_MEMORY_SITE_PROGRAM,
                                       families_[0].family};
  IREE_ASSERT_OK(
      iree_hal_device_group_resolve_memory_scope(device_group_, site, &scope));
  iree_hal_buffer_native_binding_slot_t slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      source, scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS, &slot));
  const auto address =
      iree_hal_buffer_native_binding(buffer, slot).device_address;
  EXPECT_NE(address, 0u);
  const auto memory = iree_hal_buffer_memory_view(buffer);
  ASSERT_NE(memory.backing, nullptr);
  EXPECT_EQ(address % memory.backing->allocation_alignment, 0u);
  const uint32_t value = 42;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_update(queues_[0], iree_hal_semaphore_list_empty(),
                            iree_hal_semaphore_list_empty(), &value, 0, buffer,
                            0, sizeof(value), /*barriers=*/NULL,
                            IREE_HAL_UPDATE_FLAG_NONE));
  buffer.reset();

  iree_hal_pool_scope_t host_scope = {};
  host_scope.host.access = IREE_HAL_MEMORY_ACCESS_ALL;
  host_scope.host.modes = IREE_HAL_MAPPING_MODE_SCOPED;
  Ref<iree_hal_pool_t> host_source;
  IREE_ASSERT_OK(CreateSource(host_scope, host_source.out()));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      host_source, iree_hal_make_pool_reservation_request(64, 0),
      iree_infinite_timeout(), buffer.out()));
  iree_hal_buffer_mapping_t mapping = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, sizeof(value), &mapping));
  std::memcpy(mapping.contents.data, &value, sizeof(value));
  uint32_t result = 0;
  std::memcpy(&result, mapping.contents.data, sizeof(result));
  EXPECT_EQ(result, value);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  SemaphoreList allocated(device_, {0}, {1});
  iree_hal_pool_reservation_request_t request = {
      .allocation_size = 64,
  };
  Ref<iree_hal_buffer_t> queued;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_alloca(queues_[0], iree_hal_semaphore_list_empty(),
                            allocated, host_source, 1, &request, queued.out()));
  buffer.reset();
}

TEST_P(VulkanSlabPoolTest, RejectsUnpreparedInterfacesAndNativeProperties) {
  for (auto interface :
       {IREE_HAL_BUFFER_INTERFACE_HOST, IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
        IREE_HAL_BUFFER_INTERFACE_XDNA_FIRMWARE, IREE_HAL_BUFFER_INTERFACE_RDMA,
        IREE_HAL_BUFFER_INTERFACE_REGISTERED_IO,
        IREE_HAL_BUFFER_INTERFACE_REMOTE}) {
    families_[0].interfaces = UINT64_C(1) << interface;
    Ref<iree_hal_pool_t> source;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                          CreateSource(Scope(), source.out()));
    EXPECT_FALSE(source);
  }
  families_[0].interfaces = 0;
  for (auto requirement : {IREE_HAL_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST,
                           IREE_HAL_POOL_ACCESS_REQUIRE_UNCACHED}) {
    families_[0].requirements = requirement;
    auto scope = Scope();
    scope.host.access = IREE_HAL_MEMORY_ACCESS_ALL;
    scope.host.modes = IREE_HAL_MAPPING_MODE_SCOPED;
    Ref<iree_hal_pool_t> source;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                          CreateSource(scope, source.out()));
    EXPECT_FALSE(source);
  }
  families_[0].requirements = 0;
  for (auto cacheability : {IREE_HAL_HOST_CACHEABILITY_WRITE_COMBINED,
                            IREE_HAL_HOST_CACHEABILITY_UNCACHED}) {
    auto scope = Scope();
    scope.host.access = IREE_HAL_MEMORY_ACCESS_WRITE;
    scope.host.modes = IREE_HAL_MAPPING_MODE_SCOPED;
    scope.host.cacheability = cacheability;
    Ref<iree_hal_pool_t> source;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                          CreateSource(scope, source.out()));
    EXPECT_FALSE(source);
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    options.preferences.host = scope.host;
    IREE_ASSERT_OK(iree_hal_slab_pool_create(device_group_, Scope(), &options,
                                             iree_allocator_system(),
                                             source.out()));
    EXPECT_EQ(iree_hal_pool_query_host_access(source).access,
              IREE_HAL_MEMORY_ACCESS_NONE);
  }
}

TEST_P(VulkanSlabPoolTest, PlacementAndFamilyOrderAreIndependent) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  iree_hal_pool_capabilities_t original;
  iree_hal_pool_query_capabilities(source, &original);
  std::reverse(families_.begin(), families_.end());
  Ref<iree_hal_pool_t> reversed;
  IREE_ASSERT_OK(CreateSource(Scope(), reversed.out()));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(reversed, &capabilities);
  EXPECT_EQ(original.placement.mode, capabilities.placement.mode);
  EXPECT_EQ(original.placement.node, capabilities.placement.node);
  EXPECT_EQ(original.memory_type, capabilities.memory_type);
  EXPECT_EQ(iree_hal_pool_notification(source),
            iree_hal_pool_notification(reversed));
  if (original.placement.mode == IREE_HAL_POOL_PLACEMENT_REQUIRED) {
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    options.placement = original.placement;
    Ref<iree_hal_pool_t> required;
    IREE_ASSERT_OK(iree_hal_slab_pool_create(device_group_, Scope(), &options,
                                             iree_allocator_system(),
                                             required.out()));
    iree_hal_pool_query_capabilities(required, &capabilities);
    EXPECT_EQ(capabilities.placement.mode, IREE_HAL_POOL_PLACEMENT_REQUIRED);
    EXPECT_EQ(capabilities.placement.node, options.placement.node);
  }
}

TEST_P(VulkanSlabPoolTest, SharedBackingAcrossFamiliesAndAllocationPolicies) {
  if (families_.size() > 1) {
    families_.front().usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET;
    families_.back().usage = IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE;
  }
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(source, &capabilities);
  iree_hal_slab_cache_options_t cache_options;
  iree_hal_slab_cache_options_initialize(&cache_options);
  cache_options.slab.allocation_size = 65536;
  cache_options.slab.min_alignment =
      iree_min((iree_device_size_t)4096, capabilities.max_allocation_alignment);
  Ref<iree_hal_pool_t> cache;
  IREE_ASSERT_OK(iree_hal_slab_cache_create(
      source, &cache_options, iree_allocator_system(), cache.out()));
  std::array<Ref<iree_hal_pool_t>, 2> children;
  iree_hal_tlsf_pool_options_t tlsf_options = {};
  tlsf_options.tlsf_options.range_length = 4096;
  tlsf_options.tlsf_options.alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
  tlsf_options.tlsf_options.frontier_capacity = 4;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create(
      cache, &tlsf_options, iree_allocator_system(), children[0].out()));
  iree_hal_fixed_block_pool_options_t block_options = {
      .block_size = 256,
      .blocks_per_slab = 16,
      .frontier_capacity = 4,
  };
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create(
      cache, &block_options, iree_allocator_system(), children[1].out()));

  const iree_hal_buffer_backing_facts_t* first_backing = nullptr;
  for (auto& child : children) {
    for (uint32_t cycle = 0; cycle < 3; ++cycle) {
      iree_hal_pool_reservation_request_t request = {
          .allocation_size = 32,
      };
      Ref<iree_hal_buffer_t> buffer;
      SemaphoreList allocated(device_, {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_alloca(
          queues_.front(), iree_hal_semaphore_list_empty(), allocated, child, 1,
          &request, buffer.out()));
      const std::array<uint32_t, 8> input = {11, 22, 1, 2, 3, 4, 77, 88};
      SemaphoreList uploaded(device_, {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_upload(queues_.front(), allocated, uploaded,
                                           input.data(), buffer, 0,
                                           sizeof(input), /*barriers=*/NULL));
      const uint32_t pattern = cycle + 42;
      SemaphoreList filled(device_, {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_fill(
          queues_.front(), uploaded, filled, buffer, 2 * sizeof(uint32_t),
          4 * sizeof(uint32_t), &pattern, sizeof(pattern), /*barriers=*/NULL,
          IREE_HAL_FILL_FLAG_NONE));
      std::array<uint32_t, 8> output = {};
      SemaphoreList downloaded(device_, {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_download(
          queues_.back(), filled, downloaded, buffer, 0, output.data(),
          sizeof(output), /*barriers=*/NULL));
      Wait(downloaded);
      EXPECT_THAT(output, ::testing::ElementsAre(11, 22, pattern, pattern,
                                                 pattern, pattern, 77, 88));
      const auto memory = iree_hal_buffer_memory_view(buffer);
      ASSERT_NE(memory.backing, nullptr);
      if (!first_backing) {
        first_backing = memory.backing;
      }
      EXPECT_EQ(memory.backing, first_backing);
      iree_hal_buffer_mapping_t mapping = {};
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_PERMISSION_DENIED,
          iree_hal_buffer_map_range(
              buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
              IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, sizeof(output), &mapping));
      SemaphoreList deallocated(device_, {0}, {1});
      auto* raw_buffer = buffer.get();
      IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_.back(), downloaded,
                                             deallocated, 1, &raw_buffer));
      Wait(deallocated);
      buffer.reset();
      iree_hal_pool_stats_t stats;
      iree_hal_pool_query_stats(child, &stats);
      EXPECT_EQ(stats.bytes_committed, 0u);
    }
  }
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.reserve_count, 1u);
  iree_hal_slab_cache_stats_t cache_stats;
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache, &cache_stats));
  EXPECT_EQ(cache_stats.hit_count, 5u);
  EXPECT_EQ(cache_stats.ready_count, 1u);
  iree_hal_pool_trim(cache, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  ASSERT_NO_FATAL_FAILURE(WaitCommittedBytes(source, 0));
  iree_hal_pool_query_stats(cache, &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);
  iree_hal_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.bytes_reserved, 0u);
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache, &cache_stats));
  EXPECT_EQ(cache_stats.ready_count, 0u);
}

TEST_P(VulkanSlabPoolTest, RejectsScopesCrossingNativeOwners) {
  std::array<DeviceCreateContext, 2> contexts;
  std::array<Ref<iree_hal_device_t>, 2> devices;
  for (size_t i = 0; i < devices.size(); ++i) {
    IREE_ASSERT_OK(contexts[i].Initialize(iree_allocator_system()));
    IREE_ASSERT_OK(iree_hal_driver_create_default_device(
        driver_, contexts[i].params(), iree_allocator_system(),
        devices[i].out()));
  }
  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder,
                                           contexts[0].frontier_tracker());
  iree_status_t status = iree_ok_status();
  for (auto& device : devices) {
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_builder_add_device(&builder, device);
    }
  }
  Ref<iree_hal_device_group_t> group;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_builder_finalize(
        &builder, iree_allocator_system(), group.out());
  }
  iree_hal_device_group_builder_deinitialize(&builder);
  IREE_ASSERT_OK(status);
  std::array<iree_hal_pool_family_access_t, 2> families = {};
  for (size_t i = 0; i < families.size(); ++i) {
    families[i].family = iree_hal_device_queue_family(devices[i], 0);
    families[i].usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  }
  const iree_hal_pool_scope_t scope = {families.size(), families.data(), {}};
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  Ref<iree_hal_pool_t> source;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_slab_pool_create(group, scope, &options, iree_allocator_system(),
                                source.out()));
  EXPECT_FALSE(source);
}

CTS_REGISTER_TEST_SUITE(VulkanSlabPoolTest);

}  // namespace
}  // namespace iree::hal::cts
