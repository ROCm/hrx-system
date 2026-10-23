// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_STORAGE_MEMORY_TEST_FIXTURE_H_
#define LOOM_SERVE_RUNTIME_STORAGE_MEMORY_TEST_FIXTURE_H_

#include <array>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/hal/testing/deferred_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/runtime/execution/queues.h"
#include "loom/serve/runtime/execution/workspace.h"
#include "loom/serve/runtime/storage/memory.h"
#include "loom/serve/runtime/storage/virtual_pool.h"
#include "loom/serve/testing/device.h"
#include "loom/serve/testing/native_allocator.h"

namespace loom_serve::testing::memory {

constexpr uint64_t kMemoryTestSlabSize = 2 * 1024 * 1024;

class MemoryTestEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    loom_serve_host_t* host = nullptr;
    iree_hal_driver_t* driver = nullptr;
    auto status =
        loom_serve_host_create(1, nullptr, iree_allocator_system(), &host);
    if (iree_status_is_ok(status)) {
      status = loom_serve_test_driver_acquire(host, &driver);
    }
    if (iree_status_is_ok(status)) {
      const loom_serve_device_options_t options = {
          .driver = driver,
          .uri = loom_serve_test_device()->device_uri,
          .backing = LOOM_SERVE_DEVICE_BACKING_ELASTIC,
          .weight_slab_size = kMemoryTestSlabSize,
          .state_slab_size = kMemoryTestSlabSize / 4,
          .memory_limit = 3 * kMemoryTestSlabSize,
      };
      status = loom_serve_device_group_allocate(host, 1, &options,
                                                native.allocator(), &group);
    }
    iree_hal_driver_release(driver);
    loom_serve_host_release(host);
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(native.Observe(loom_serve_device_group_at(group, 0)));
  }

  void TearDown() override {
    IREE_EXPECT_OK(loom_serve_device_group_free(group));
    group = nullptr;
    native.AwaitReleased();
  }

  // Allocator dependency retained through all asynchronous device cleanup.
  loom_serve::testing::NativeAllocator native;
  // One native device shared by all cases; pools and reservations are per case.
  loom_serve_device_group_t* group = nullptr;
};

extern MemoryTestEnvironment* const memory_test_environment;

// Observes and gates real mapping submissions, or rejects one before
// acceptance. The storage owner uses only allocation, deallocation, and mapping
// operations.
class MappingQueue : public iree_hal_queue_t {
 public:
  ~MappingQueue() { Reset(); }

  void Initialize(iree_hal_queue_t* queue) {
    queue_ = queue;
    iree_hal_queue_retain(queue_);
    const iree_hal_queue_params_t params = {
        .priority = queue->priority,
        .features = queue->features,
        .execution_resources = queue->execution_resources,
    };
    static const iree_hal_queue_vtable_t vtable = {
        .destroy =
            [](iree_hal_queue_t* base) {
              auto* self = static_cast<MappingQueue*>(base);
              iree_hal_queue_release(self->queue_);
              self->queue_ = nullptr;
            },
        .alloca =
            [](iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
               iree_hal_semaphore_list_t signals, iree_hal_pool_t* pool,
               iree_host_size_t count,
               const iree_hal_pool_reservation_request_t* requests,
               iree_hal_buffer_t** buffers) {
              return iree_hal_queue_alloca(
                  static_cast<MappingQueue*>(base)->queue_, waits, signals,
                  pool, count, requests, buffers);
            },
        .dealloca =
            [](iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
               iree_hal_semaphore_list_t signals, iree_host_size_t count,
               iree_hal_buffer_t* const* buffers) {
              return iree_hal_queue_dealloca(
                  static_cast<MappingQueue*>(base)->queue_, waits, signals,
                  count, buffers);
            },
        .map =
            [](iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
               iree_hal_semaphore_list_t signals, iree_host_size_t count,
               const iree_hal_memory_mapping_t* mappings,
               iree_hal_queue_map_flags_t flags) {
              auto* self = static_cast<MappingQueue*>(base);
              if (self->observe_mappings) {
                self->mapping_waits.emplace_back();
                for (iree_host_size_t i = 0; i < waits.count; ++i) {
                  self->mapping_waits.back().push_back(waits.semaphores[i]);
                  EXPECT_EQ(waits.payload_values[i], 1u);
                }
                EXPECT_EQ(signals.count, 1u);
                self->mapping_signals.push_back(signals.semaphores[0]);
              }
              if (self->fail_next_map) {
                self->fail_next_map = false;
                return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
              }
              if (self->gate_next_mapping) {
                std::vector<iree_hal_semaphore_t*> semaphores;
                std::vector<uint64_t> values;
                for (iree_host_size_t i = 0; i < waits.count; ++i) {
                  semaphores.push_back(waits.semaphores[i]);
                  values.push_back(waits.payload_values[i]);
                }
                semaphores.push_back(self->gate_next_mapping);
                values.push_back(1);
                self->gate_next_mapping = nullptr;
                return iree_hal_queue_map(
                    self->queue_,
                    {semaphores.size(), semaphores.data(), values.data()},
                    signals, count, mappings, flags);
              }
              return iree_hal_queue_map(self->queue_, waits, signals, count,
                                        mappings, flags);
            },
    };
    iree_hal_queue_initialize(iree_hal_queue_family(queue), &params, &vtable,
                              this);
  }

  void Reset() {
    if (queue_) {
      iree_hal_queue_release(this);
    }
  }

  // Rejects one mapping submission without changing its signal semaphores.
  bool fail_next_map = false;
  // Borrowed prerequisite added to exactly one real mapping submission.
  iree_hal_semaphore_t* gate_next_mapping = nullptr;
  // Records dependencies before the real queue captures a mapping batch.
  bool observe_mappings = false;
  // Borrowed wait identities, inspected while operations retain their stages.
  std::vector<std::vector<iree_hal_semaphore_t*>> mapping_waits;
  // Borrowed mapping-stage identities, inspected before operation retirement.
  std::vector<iree_hal_semaphore_t*> mapping_signals;

 private:
  // Retained native queue used for all accepted operations.
  iree_hal_queue_t* queue_ = nullptr;
};

class MemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    group = memory_test_environment->group;
    device = loom_serve_device_group_at(group, 0);
    ASSERT_NO_FATAL_FAILURE(SetUpPool());
    IREE_ASSERT_OK(loom_serve_device_create_execution(device, &execution));
  }

  virtual void SetUpPool() { UseBacking(false, 3 * kSlabSize, nullptr); }

  void TearDown() override {
    if (deferred_backing) {
      deferred_backing->ResumeAcquisitions();
      deferred_backing->CompleteReturns();
    }
    iree_hal_buffer_release(workspace);
    if (virtual_pool) {
      const auto frontier = loom_serve_execution_frontier(execution);
      for (size_t i = 0; i < IREE_ARRAYSIZE(frontier.semaphores); ++i) {
        iree_status_ignore(loom_serve_virtual_pool_wait(
            virtual_pool, frontier.semaphores[i], frontier.values[i]));
      }
      AwaitVirtualPool();
    }
    if (execution) {
      IREE_ASSERT_OK(loom_serve_execution_drain(execution));
    }
    if (virtual_workspace) {
      loom_serve_execution_release(execution);
      execution = nullptr;
      loom_serve_workspace_release(virtual_workspace);
      virtual_workspace = nullptr;
    }
    if (logical_pool) {
      // Raw queue captures can retire after timeline readiness.
      while (iree_atomic_ref_count_load(&logical_pool->resource.ref_count) >
             1) {
        loom_serve_workspace_statistics(loom_serve_device_workspace(device));
        std::this_thread::yield();
      }
      iree_hal_pool_release(logical_pool);
    }
    IREE_ASSERT_OK(loom_serve_virtual_pool_free(virtual_pool));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_free(first));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_free(second));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_free(third));
    if (pool) {
      loom_serve_memory_pool_trim(pool, 0);
      // Intentional failure cases retain their original status while native
      // disposal still reaches its physical capacity boundary.
      iree_status_ignore(loom_serve_memory_pool_await(pool));
      const auto statistics = loom_serve_memory_pool_statistics(pool);
      EXPECT_EQ(statistics.committed_bytes, 0u);
      const auto workspace_usage = loom_serve_workspace_memory_usage(
          loom_serve_device_workspace(device));
      EXPECT_EQ(statistics.reserved_bytes,
                owned_pool ? 0u : workspace_usage.reserved_bytes);
    }
    loom_serve_memory_pool_free(owned_pool);
    iree_hal_pool_release(deferred_backing);
    mapping_queue.Reset();
    loom_serve_execution_release(execution);
    IREE_EXPECT_OK(loom_serve_test_device_trim_and_drain(device, 0));
  }

  void UseDeferredBacking(uint64_t limit_bytes = 3 * kSlabSize,
                          iree_hal_queue_t* queue = nullptr,
                          bool distinct_classes = false) {
    UseBacking(true, limit_bytes, queue, distinct_classes);
  }

  void UseBacking(bool deferred, uint64_t limit_bytes, iree_hal_queue_t* queue,
                  bool distinct_classes = true) {
    // Selection happens before this case allocates any reservations.
    loom_serve_memory_pool_free(owned_pool);
    owned_pool = nullptr;
    pool = nullptr;
    const auto host_allocator = native.allocator();
    const auto* dispatch =
        iree_hal_queue_family(loom_serve_device_dispatch_queue(device));
    if (!queue) {
      queue = loom_serve_device_transfer_queue(device);
    }
    const auto* transfer = iree_hal_queue_family(queue);
    iree_hal_pool_family_access_t families[] = {
        {
            .family = dispatch,
            .usage =
                IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
            .interfaces = UINT64_C(1)
                          << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
        },
        {
            .family = transfer,
            .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
            .interfaces = UINT64_C(1)
                          << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
        },
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = dispatch == transfer ? 1u : 2u,
        .families = families,
    };
    auto* hal_group = loom_serve_device_group(device);
    iree_hal_pool_t* backing = nullptr;
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_ASSERT_OK(iree_hal_slab_pool_create(hal_group, scope, &options,
                                             host_allocator, &backing));
    if (deferred) {
      auto status = iree::hal::testing::DeferredBackingPool::Create(
          backing, &deferred_backing);
      iree_hal_pool_release(backing);
      IREE_ASSERT_OK(status);
      backing = deferred_backing;
      iree_hal_pool_retain(backing);
    }
    for (auto& family : families) {
      family.interfaces = 0;
    }
    iree_hal_virtual_buffer_capabilities_t capabilities;
    auto status = iree_hal_virtual_buffer_query_capabilities(
        hal_group, scope, host_allocator, &capabilities);
    loom_serve_memory_geometry_t geometries[2] = {};
    if (iree_status_is_ok(status)) {
      status = loom_serve_memory_pool_query_geometry(&capabilities, backing,
                                                     kSlabSize, &geometries[0]);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_memory_pool_query_geometry(
          &capabilities, backing, distinct_classes ? kSlabSize / 4 : kSlabSize,
          &geometries[1]);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_memory_pool_allocate(hal_group, scope, backing, queue,
                                               geometries, limit_bytes,
                                               host_allocator, &owned_pool);
    }
    iree_hal_pool_release(backing);
    IREE_ASSERT_OK(status);
    pool = owned_pool;
  }

  void RoundTrip(iree_hal_buffer_t* buffer, uint64_t offset, uint32_t value) {
    std::array<uint32_t, 64> observed = {};
    const iree_hal_transfer_operation_t operations[] = {
        {
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
            .fill =
                {
                    .target_buffer = buffer,
                    .target_offset = offset,
                    .length = sizeof(observed),
                    .pattern = &value,
                    .pattern_length = sizeof(value),
                },
        },
        {
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
            .download =
                {
                    .source_buffer = buffer,
                    .source_offset = offset,
                    .target = observed.data(),
                    .length = sizeof(observed),
                },
        },
    };
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_transfer(execution, 1, operations, &completion));
    IREE_ASSERT_OK(loom_serve_execution_transfer(execution, 1, operations + 1,
                                                 &completion));
    if (virtual_pool && !virtual_workspace) {
      const auto frontier = loom_serve_execution_frontier(execution);
      IREE_ASSERT_OK(loom_serve_virtual_pool_wait(
          virtual_pool, frontier.semaphores[0], completion));
    } else {
      IREE_ASSERT_OK(loom_serve_execution_wait(execution, completion));
    }
    for (const auto word : observed) {
      EXPECT_EQ(word, value);
    }
  }

  void CreateVirtualPool(bool with_workspace = false) {
    IREE_ASSERT_OK(loom_serve_virtual_pool_allocate(
        pool, loom_serve_device_weight_geometry(device),
        iree_allocator_system(), &virtual_pool));
    if (with_workspace) {
      const auto range = iree_hal_make_pool_reservation_request(kSlabSize, 256);
      IREE_ASSERT_OK(loom_serve_workspace_create_from_virtual(
          loom_serve_device_group(device), virtual_pool, range,
          iree_allocator_system(), &virtual_workspace));
      loom_serve_execution_queues_t* queues = nullptr;
      IREE_ASSERT_OK(loom_serve_execution_queues_create(
          iree_hal_queue_family(loom_serve_device_dispatch_queue(device)),
          iree_hal_queue_family(loom_serve_device_transfer_queue(device)),
          iree_allocator_system(), &queues));
      loom_serve_execution_release(execution);
      execution = nullptr;
      auto status = loom_serve_execution_create(
          queues, virtual_workspace, iree_allocator_system(), &execution);
      loom_serve_execution_queues_release(queues);
      IREE_ASSERT_OK(status);
      IREE_ASSERT_OK(
          loom_serve_workspace_acquire_pool(virtual_workspace, &logical_pool));
      return;
    }
    const iree_hal_tlsf_pool_options_t options = {
        .tlsf_options = {.range_length = kSlabSize, .alignment = 256},
    };
    IREE_ASSERT_OK(iree_hal_tlsf_pool_create(
        loom_serve_virtual_pool_handle(virtual_pool), &options,
        iree_allocator_system(), &logical_pool));
  }

  void AwaitVirtualPool() {
    iree_notification_t notification;
    iree_notification_initialize(&notification);
    const loom_serve_wake_callback_t wake = {
        [](void* user_data) {
          iree_notification_post(static_cast<iree_notification_t*>(user_data),
                                 IREE_ALL_WAITERS);
        },
        &notification};
    const auto previous =
        loom_serve_memory_pool_set_observer(pool, wake.wake, wake.user_data);
    loom_serve_virtual_pool_set_observer(virtual_pool, wake);
    iree_status_t status = iree_ok_status();
    bool pending = true;
    while (pending && iree_status_is_ok(status)) {
      const auto token = iree_notification_prepare_wait(&notification);
      status = loom_serve_virtual_pool_advance(virtual_pool);
      pending = loom_serve_virtual_pool_is_pending(virtual_pool);
      if (pending && iree_status_is_ok(status)) {
        iree_notification_commit_wait(&notification, token, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      } else {
        iree_notification_cancel_wait(&notification);
      }
    }
    loom_serve_virtual_pool_set_observer(virtual_pool, {});
    loom_serve_memory_pool_set_observer(pool, previous.wake,
                                        previous.user_data);
    iree_notification_deinitialize(&notification);
    IREE_EXPECT_OK(status);
  }

  void CheckRetained(iree_hal_buffer_t* buffer, uint64_t offset,
                     uint32_t value) {
    std::array<uint32_t, 64> observed = {};
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = buffer,
                .source_offset = offset,
                .target = observed.data(),
                .length = sizeof(observed),
            },
    };
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_transfer(execution, 1, &download, &completion));
    IREE_ASSERT_OK(loom_serve_execution_wait(execution, completion));
    for (const auto word : observed) {
      EXPECT_EQ(word, value);
    }
  }

  // Process-owned allocator dependency shared by the native device and pools.
  loom_serve::testing::NativeAllocator& native =
      memory_test_environment->native;
  // Optional mapping dependency outliving the private pool's native cleanup.
  MappingQueue mapping_queue;
  // Per-case physical owner, isolating statistics and terminal pool failures.
  loom_serve_memory_pool_t* owned_pool = nullptr;
  // Optional adapter retaining real parent tokens beyond logical completion.
  iree::hal::testing::DeferredBackingPool* deferred_backing = nullptr;
  // Shared production workspace, outliving its execution's raw root captures.
  loom_serve_workspace_t* virtual_workspace = nullptr;
  // Coarse physical slabs, independent of logical KV block size.
  static constexpr uint64_t kSlabSize = kMemoryTestSlabSize;
  // Offset deliberately crossing the 32-bit byte-address boundary.
  static constexpr uint64_t kHighOffset = (uint64_t{1} << 32) + 512;
  // Borrowed process-owned group enclosing all native device resources.
  loom_serve_device_group_t* group = nullptr;
  // Borrowed device record within the application group.
  loom_serve_device_t* device = nullptr;
  // Owned model timeline for transfers and workspace allocations.
  loom_serve_execution_t* execution = nullptr;
  // One physical budget shared by the independent reservations.
  loom_serve_memory_pool_t* pool = nullptr;
  // Independent accounting groups competing for the same physical budget.
  std::array<loom_serve_memory_statistics_t, 3> statistics = {};
  // First model-like reservation, sparsely backed beyond four GiB.
  loom_serve_virtual_buffer_t* first = nullptr;
  // Independent reservation competing for the same physical budget.
  loom_serve_virtual_buffer_t* second = nullptr;
  // Concurrent third model sharing physical capacity with the second.
  loom_serve_virtual_buffer_t* third = nullptr;
  // Private command allocation retained until its queue release is submitted.
  iree_hal_buffer_t* workspace = nullptr;
  // Queue-facing source whose cold work advances on this test's owner.
  loom_serve_virtual_pool_t* virtual_pool = nullptr;
  // Persistent logical allocator above virtual_pool.
  iree_hal_pool_t* logical_pool = nullptr;
};

// These cases hold a native allocation worker while other device work proceeds.
// Task devices share that worker with execution, so their composition excludes
// this fixture instead of blocking the sole CPU worker needed for progress.
using DedicatedNativeWorkerMemoryTest = MemoryTest;

// Device-wide trim and group ownership use the real device-owned pool. Other
// cases isolate their statistics and intentional failures in private pools.
class DeviceMemoryTest : public MemoryTest {
 protected:
  void SetUpPool() override { pool = loom_serve_device_memory_pool(device); }
};

}  // namespace loom_serve::testing::memory

#endif  // LOOM_SERVE_RUNTIME_STORAGE_MEMORY_TEST_FIXTURE_H_
