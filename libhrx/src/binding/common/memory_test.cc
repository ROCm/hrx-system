// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/memory.h"

#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <semaphore>
#include <thread>
#include <vector>

#include "common/context.h"
#include "common/device.h"
#include "common/direct_transfer.h"
#include "common/graph.h"
#include "common/host_callback.h"
#include "common/hrx_bridge.h"
#include "common/stream.h"
#include "common/stream_queue.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libhrx/src/libhrx/buffer.h"
#include "libhrx/src/libhrx/device.h"

namespace {

struct InjectedFlushQueue {
  // Public queue wrapper forwarding accepted work to the real task queue.
  iree_hal_queue_t base;
  // Retained queue whose execution and ownership semantics are under test.
  iree_hal_queue_t* target = nullptr;
  // Optional additional prerequisite borrowed through each submitted operation.
  iree_hal_semaphore_t* execute_gate = nullptr;
  // Value required on the additional prerequisite.
  uint64_t execute_gate_value = 0;
  // Injects a flush error after the real queue has accepted work.
  std::atomic<bool> fail_flush = false;
  // Rejects the next transfer before any work is accepted.
  std::atomic<bool> fail_transfer = false;
  // Number of gated command buffers accepted by the target queue.
  std::atomic<int> injected_execute_count = 0;
  // Number of transfer batches accepted by the target queue.
  std::atomic<int> accepted_transfer_count = 0;
  // Optional observer notified after the real queue accepts a transfer.
  std::binary_semaphore* transfer_accepted = nullptr;
  // Borrowed source of the observed upload, published through the observer.
  const void* upload_source = nullptr;
  // First dependency captured at native acceptance, borrowed until test join.
  iree_hal_semaphore_t* accepted_wait_semaphore = nullptr;
  // Required value corresponding to |accepted_wait_semaphore|.
  uint64_t accepted_wait_value = 0;
  // First signal captured at native acceptance, borrowed until test join.
  iree_hal_semaphore_t* accepted_signal_semaphore = nullptr;
  // Published value corresponding to |accepted_signal_semaphore|.
  uint64_t accepted_signal_value = 0;
  // Number of completion signals owned by the accepted batch.
  iree_host_size_t accepted_signal_count = 0;
  // Number of injected flush errors observed by the caller.
  std::atomic<int> injected_flush_count = 0;
  // Announces flush failure without timing-dependent polling.
  std::counting_semaphore<> flush_observed{0};
};

// Stack storage for the stream prerequisite and the test's additional gate.
struct InjectedQueueWaits {
  // Borrowed semaphores, read only during queue acceptance.
  std::array<iree_hal_semaphore_t*, 2> semaphores = {};
  // Corresponding required values.
  std::array<uint64_t, 2> values = {};
  // List referencing this object's arrays or the unmodified input list.
  iree_hal_semaphore_list_t list;

  InjectedQueueWaits(InjectedFlushQueue* queue, iree_hal_semaphore_list_t waits)
      : list(waits) {
    if (!queue->execute_gate) {
      return;
    }
    EXPECT_LE(waits.count, 1u);
    for (iree_host_size_t i = 0; i < waits.count; ++i) {
      semaphores[i] = waits.semaphores[i];
      values[i] = waits.payload_values[i];
    }
    semaphores[waits.count] = queue->execute_gate;
    values[waits.count] = queue->execute_gate_value;
    list = {waits.count + 1, semaphores.data(), values.data()};
  }
};

InjectedFlushQueue* CastInjectedFlushQueue(iree_hal_queue_t* base_queue) {
  return reinterpret_cast<InjectedFlushQueue*>(base_queue);
}

void DestroyInjectedFlushQueue(iree_hal_queue_t* base_queue) {
  auto* queue = CastInjectedFlushQueue(base_queue);
  iree_hal_queue_release(queue->target);
  queue->target = nullptr;
}

iree_status_t InjectedFlushQueueBarrier(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_queue_barrier_flags_t flags) {
  return iree_hal_queue_barrier(CastInjectedFlushQueue(base_queue)->target,
                                wait_semaphore_list, signal_semaphore_list,
                                barriers, flags);
}

iree_status_t InjectedFlushQueueExecute(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_queue_execute_flags_t flags) {
  auto* queue = CastInjectedFlushQueue(base_queue);
  InjectedQueueWaits waits(queue, wait_semaphore_list);
  iree_status_t status =
      iree_hal_queue_execute(queue->target, waits.list, signal_semaphore_list,
                             command_buffer, binding_table, flags);
  if (iree_status_is_ok(status)) {
    queue->injected_execute_count.fetch_add(1, std::memory_order_acq_rel);
  }
  return status;
}

iree_status_t InjectedFlushQueueTransfer(
    iree_hal_queue_t* base_queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations,
    const iree_hal_queue_barriers_t* barriers) {
  auto* queue = CastInjectedFlushQueue(base_queue);
  if (queue->fail_transfer.exchange(false, std::memory_order_acq_rel)) {
    return iree_make_status(IREE_STATUS_ABORTED, "injected transfer rejection");
  }
  InjectedQueueWaits gated_waits(queue, waits);
  iree_status_t status =
      iree_hal_queue_transfer(queue->target, gated_waits.list, signals,
                              operation_count, operations, barriers);
  if (iree_status_is_ok(status)) {
    queue->accepted_transfer_count.fetch_add(1, std::memory_order_acq_rel);
    queue->accepted_wait_semaphore =
        waits.count ? waits.semaphores[0] : nullptr;
    queue->accepted_wait_value = waits.count ? waits.payload_values[0] : 0;
    queue->accepted_signal_semaphore =
        signals.count ? signals.semaphores[0] : nullptr;
    queue->accepted_signal_value =
        signals.count ? signals.payload_values[0] : 0;
    queue->accepted_signal_count = signals.count;
    if (operation_count == 1 &&
        operations[0].type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD) {
      queue->upload_source = operations[0].upload.source;
    }
    if (queue->transfer_accepted) {
      queue->transfer_accepted->release();
    }
  }
  return status;
}

iree_status_t InjectedFlushQueueFlush(iree_hal_queue_t* base_queue) {
  auto* queue = CastInjectedFlushQueue(base_queue);
  iree_status_t status = iree_hal_queue_flush(queue->target);
  if (iree_status_is_ok(status) &&
      queue->fail_flush.load(std::memory_order_acquire)) {
    queue->injected_flush_count.fetch_add(1, std::memory_order_acq_rel);
    queue->flush_observed.release();
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "injected post-accept queue flush failure");
  }
  return status;
}

const iree_hal_queue_vtable_t kInjectedFlushQueueVtable = {
    .destroy = DestroyInjectedFlushQueue,
    .barrier = InjectedFlushQueueBarrier,
    .execute = InjectedFlushQueueExecute,
    .host_call = nullptr,
    .query_dispatch_concurrency = nullptr,
    .dispatch = nullptr,
    .atomic_wait = nullptr,
    .atomic_store = nullptr,
    .atomic_rmw = nullptr,
    .timestamp = nullptr,
    .flush = InjectedFlushQueueFlush,
    .alloca = nullptr,
    .dealloca = nullptr,
    .transfer = InjectedFlushQueueTransfer,
    .read = nullptr,
    .write = nullptr,
};

void InitializeInjectedFlushQueue(iree_hal_queue_t* target,
                                  InjectedFlushQueue* out_queue) {
  out_queue->target = target;
  iree_hal_queue_retain(target);
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  params.priority = iree_hal_queue_priority(target);
  params.features = iree_hal_queue_features(target);
  params.execution_resources = iree_hal_queue_execution_resources(target);
  iree_hal_queue_initialize(iree_hal_queue_family(target), &params,
                            &kInjectedFlushQueueVtable, &out_queue->base);
}

iree_hal_queue_t* ReplaceStreamQueue(iree_hal_streaming_stream_t* stream,
                                     iree_hal_queue_t* replacement) {
  iree_slim_mutex_lock(&stream->mutex);
  iree_hal_queue_t* previous = stream->queue;
  stream->queue = replacement;
  iree_slim_mutex_unlock(&stream->mutex);
  return previous;
}
struct FailingSnapshotAllocator {
  // Allocator that owns all context and completion-record storage.
  iree_allocator_t delegate = iree_allocator_system();
  // Armed only for the snapshot allocation being exercised.
  std::atomic<bool> fail_next_allocation = false;
  // Announces the failed allocation without timing-dependent polling.
  std::binary_semaphore failure_observed{0};

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<FailingSnapshotAllocator*>(self);
    if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
        command == IREE_ALLOCATOR_COMMAND_CALLOC ||
        command == IREE_ALLOCATOR_COMMAND_REALLOC) {
      if (allocator->fail_next_allocation.exchange(false,
                                                   std::memory_order_acq_rel)) {
        allocator->failure_observed.release();
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "injected stream snapshot allocation failure");
      }
    }
    return allocator->delegate.ctl(allocator->delegate.self, command, params,
                                   inout_ptr);
  }

  iree_allocator_t AsAllocator() {
    return iree_allocator_t{this, &FailingSnapshotAllocator::Control};
  }
};

class ScopedHeldCallback {
 public:
  explicit ScopedHeldCallback(iree_hal_streaming_stream_t* stream)
      : stream_(stream) {}
  ~ScopedHeldCallback() {
    released_.release();
    IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
    iree_hal_streaming_stream_release(stream_);
  }

  iree_status_t Start() {
    return iree_hal_streaming_launch_host_function(stream_, Run, this);
  }
  void WaitUntilEntered() { entered_.acquire(); }
  bool returned() const { return returned_.load(std::memory_order_acquire); }

 private:
  static void Run(void* user_data) {
    auto* callback = static_cast<ScopedHeldCallback*>(user_data);
    callback->entered_.release();
    callback->released_.acquire();
    callback->returned_.store(true, std::memory_order_release);
  }

  // Owned stream, synchronized and released after releasing the callback.
  iree_hal_streaming_stream_t* stream_;
  // Publishes that the frontend worker has entered the application callback.
  std::binary_semaphore entered_{0};
  // Releases the application callback on every exit from the enclosing test.
  std::binary_semaphore released_{0};
  // Set before the application callback returns to its execution owner.
  std::atomic<bool> returned_{false};
};

class CpuStreamingMemoryTest : public ::testing::Test {
 protected:
  // Owns the task device and its runtime through fixture cleanup.
  hrx_device_group_t group_ = nullptr;

  void SetUp() override {
    iree_hal_streaming_host_callback_queue_initialize(iree_allocator_system(),
                                                      &host_callbacks_);
    hrx_runtime_t runtime = nullptr;
    IREE_ASSERT_OK(HRX_CALL(
        hrx_runtime_create(nullptr, hrx_host_allocator_system(), &runtime)));
    const hrx_device_create_params_t params = {.uri = "task://"};
    hrx_status_t status = hrx_device_group_create(runtime, 1, &params, &group_);
    hrx_runtime_release(runtime);
    IREE_ASSERT_OK(HRX_CALL(status));
    hrx_device_t hrx_device = hrx_device_group_get_device(group_, 0);

    device_entry_.hrx_device = hrx_device;
    device_entry_.hal_device = hrx_device_hal(hrx_device);
    device_entry_.host_callbacks = &host_callbacks_;
    iree_slim_mutex_initialize(&device_entry_.primary_context_mutex);
    iree_slim_mutex_initialize(&device_entry_.graph_memory_mutex);
    iree_arena_block_pool_initialize(/*block_size=*/64 * 1024,
                                     iree_allocator_system(),
                                     &device_entry_.block_pool);

    iree_hal_streaming_context_flags_t context_flags = {
        .scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO,
    };
    IREE_ASSERT_OK(iree_hal_streaming_context_create(
        &device_entry_, context_flags, snapshot_allocator_.AsAllocator(),
        &context_));
    IREE_ASSERT_OK(iree_hal_streaming_stream_create(
        context_, context_->queue, IREE_HAL_STREAMING_STREAM_FLAG_NONE,
        /*priority=*/0, iree_allocator_system(), &stream_));
  }

  void TearDown() override {
    if (context_) {
      IREE_EXPECT_OK(iree_hal_streaming_context_synchronize(context_));
    }
    iree_hal_streaming_host_callback_queue_stop(&host_callbacks_);
    iree_hal_streaming_memory_release_wrapped_buffer(host_buffer_);
    iree_hal_streaming_memory_release_wrapped_buffer(buffer_);
    buffer_ = nullptr;
    device_pointer_ = 0;
    iree_hal_streaming_stream_release(stream_);
    iree_hal_streaming_context_release(context_);
    iree_arena_block_pool_deinitialize(&device_entry_.block_pool);
    iree_slim_mutex_deinitialize(&device_entry_.graph_memory_mutex);
    iree_slim_mutex_deinitialize(&device_entry_.primary_context_mutex);
    hrx_device_group_release(group_);
    group_ = nullptr;
    iree_hal_streaming_host_callback_queue_deinitialize(&host_callbacks_);
  }

  // Explicit frontend callout owner shared by this fixture's device contexts.
  iree_hal_streaming_host_callback_queue_t host_callbacks_ = {};
  // Lives through context teardown and every asynchronous completion callback.
  FailingSnapshotAllocator snapshot_allocator_;
  iree_hal_streaming_device_t device_entry_ = {};
  iree_hal_streaming_context_t* context_ = nullptr;
  iree_hal_streaming_stream_t* stream_ = nullptr;
  iree_hal_streaming_buffer_t* buffer_ = nullptr;
  // Registered host destination held through asynchronous copy completion.
  iree_hal_streaming_buffer_t* host_buffer_ = nullptr;
  // Caller-owned registered pages, live until TearDown releases their import.
  std::array<uint8_t, 16> registered_storage_ = {};
  iree_hal_streaming_deviceptr_t device_pointer_ = 0;
};

TEST_F(CpuStreamingMemoryTest, HostRegistrationAddressOutlivesRegistryEntry) {
  uint8_t storage[64] = {};
  IREE_ASSERT_OK(HRX_CALL(hrx_host_memory_register(
      device_entry_.hrx_device, storage, sizeof(storage), 0)));
  hrx_buffer_t buffer = nullptr;
  size_t offset = 0;
  IREE_ASSERT_OK(HRX_CALL(hrx_buffer_lookup(device_entry_.hrx_device,
                                            storage + 13, &buffer, &offset)));
  EXPECT_EQ(13u, offset);
  hrx_buffer_retain(buffer);
  IREE_ASSERT_OK(
      HRX_CALL(hrx_host_memory_unregister(device_entry_.hrx_device, storage)));
  void* device_pointer = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_buffer_query_device_pointer(
      buffer, device_entry_.hrx_device, &device_pointer));
  EXPECT_EQ(storage, device_pointer);
  void* mapped_pointer = storage;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      HRX_CALL(hrx_buffer_map(buffer, HRX_MAP_READ, 0, sizeof(storage),
                              &mapped_pointer)));
  EXPECT_EQ(nullptr, mapped_pointer);
  IREE_ASSERT_OK(HRX_CALL(hrx_buffer_unmap(buffer)));
  IREE_ASSERT_OK(iree_hal_streaming_buffer_query_device_pointer(
      buffer, device_entry_.hrx_device, &device_pointer));
  EXPECT_EQ(storage, device_pointer);
  hrx_buffer_release(buffer);
}

TEST_F(CpuStreamingMemoryTest,
       NativeAddressesPreserveScopedMappingsAndImportedContents) {
  const iree_hal_buffer_usage_t usages[] = {
      IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
          IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
      IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED,
  };
  for (iree_hal_buffer_usage_t usage : usages) {
    alignas(64) uint8_t storage[64];
    for (size_t i = 0; i < sizeof(storage); ++i) {
      storage[i] = i;
    }
    iree_hal_buffer_t* root = nullptr;
    IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
        iree_hal_buffer_placement_undefined(), IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
        IREE_HAL_MEMORY_ACCESS_ALL, usage, sizeof(storage),
        iree_make_byte_span(storage, sizeof(storage)),
        iree_hal_buffer_release_callback_null(), iree_allocator_system(),
        &root));
    iree_hal_buffer_t* view = nullptr;
    IREE_ASSERT_OK(
        iree_hal_buffer_subspan(root, 13, 23, iree_allocator_system(), &view));
    hrx_buffer_t buffer = nullptr;
    IREE_ASSERT_OK(hrx_buffer_create_from_hal(
        view, device_entry_.hrx_device->group, HRX_MEMORY_TYPE_HOST_LOCAL, 23,
        nullptr, &buffer));
    iree_hal_buffer_release(view);
    iree_hal_buffer_release(root);

    // Cover no mapping, an interior read-only mapping, and a whole-buffer map.
    // None of these states can change native export permissions or its base.
    for (int mapping_case = 0; mapping_case < 3; ++mapping_case) {
      const size_t map_offset = mapping_case == 1 ? 5 : 0;
      const size_t map_length = mapping_case == 1 ? 7 : 23;
      if (mapping_case != 0) {
        void* mapped_pointer = nullptr;
        IREE_ASSERT_OK(
            HRX_CALL(hrx_buffer_map(buffer, HRX_MAP_READ | HRX_MAP_MAY_ALIAS,
                                    map_offset, map_length, &mapped_pointer)));
        EXPECT_EQ(storage + 13 + map_offset, mapped_pointer);
      }
      const iree_hal_buffer_mapping_t mapping = buffer->mapping;
      for (int query = 0; query < 2; ++query) {
        void* pointer = storage;
        iree_status_t status = iree_hal_streaming_buffer_query_device_pointer(
            buffer, device_entry_.hrx_device, &pointer);
        if (usage & IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT) {
          IREE_ASSERT_OK(status);
          EXPECT_EQ(storage + 13, pointer);
        } else {
          IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED, status);
          EXPECT_EQ(nullptr, pointer);
        }
        EXPECT_EQ(mapping_case != 0, buffer->is_mapped);
        EXPECT_EQ(0, memcmp(&mapping, &buffer->mapping, sizeof(mapping)));
      }
      IREE_ASSERT_OK(HRX_CALL(hrx_buffer_unmap(buffer)));
    }
    hrx_buffer_release(buffer);
    for (size_t i = 0; i < sizeof(storage); ++i) {
      EXPECT_EQ(i, storage[i]);
    }
  }
}

TEST_F(CpuStreamingMemoryTest,
       AllocationIdentityDoesNotFollowUnifiedMemoryPlacement) {
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, 16, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_host(
      context_, 16, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT,
      /*fine_grained=*/false, &host_buffer_));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(host_buffer_->host_ptr),
            host_buffer_->device_ptr);
  const iree_hal_memory_type_t shared_memory =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  EXPECT_TRUE(iree_all_bits_set(buffer_->memory_type, shared_memory));
  EXPECT_TRUE(iree_all_bits_set(host_buffer_->memory_type, shared_memory));
  EXPECT_EQ(buffer_->memory_kind, IREE_HAL_STREAMING_MEMORY_KIND_DEVICE);
  EXPECT_EQ(host_buffer_->memory_kind, IREE_HAL_STREAMING_MEMORY_KIND_HOST);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_streaming_memory_free_host(context_, buffer_->host_ptr));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_streaming_memory_unregister_host(context_, buffer_->host_ptr));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_streaming_memory_free_device(
                            context_, host_buffer_->device_ptr));
  iree_hal_streaming_host_register_flags_t flags;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_streaming_memory_host_flags(
                            context_, buffer_->host_ptr, &flags));
  IREE_EXPECT_OK(iree_hal_streaming_memory_host_flags(
      context_, host_buffer_->host_ptr, &flags));
  EXPECT_EQ(flags, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT);
}

TEST_F(CpuStreamingMemoryTest,
       ManagedAndRegisteredHostAllocationsRejectWrongReleaseOwnership) {
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_managed(
      context_, 16, /*allocation_flags=*/0, &buffer_));
  EXPECT_EQ(buffer_->memory_kind, IREE_HAL_STREAMING_MEMORY_KIND_MANAGED);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(buffer_->host_ptr),
            buffer_->device_ptr);
  IREE_ASSERT_OK(iree_hal_streaming_memory_register_host(
      context_, registered_storage_.data(), registered_storage_.size(),
      IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT, &host_buffer_));
  EXPECT_EQ(host_buffer_->memory_kind, IREE_HAL_STREAMING_MEMORY_KIND_HOST);
  EXPECT_TRUE(host_buffer_->imported_host_allocation);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_streaming_memory_free_host(
                            context_, registered_storage_.data()));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_streaming_memory_free_host(context_, buffer_->host_ptr));
}

TEST_F(CpuStreamingMemoryTest,
       PitchedTransfersUseAllocationIdentityOnSharedMemory) {
  constexpr size_t kSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_host(
      context_, kSize, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT,
      /*fine_grained=*/false, &host_buffer_));
  const std::array<uint8_t, kSize> source = {1, 2,  3,  4,  5,  6,  7,  8,
                                             9, 10, 11, 12, 13, 14, 15, 16};
  std::memset(host_buffer_->host_ptr, 0xcd, kSize);
  IREE_ASSERT_OK(iree_hal_streaming_memcpy_host_to_device_2d(
      context_, device_pointer_, /*dst_pitch=*/8, source.data(),
      /*src_pitch=*/8, /*width=*/4, /*height=*/2, stream_));
  IREE_ASSERT_OK(iree_hal_streaming_memcpy_device_to_host_2d(
      context_, host_buffer_->host_ptr, /*dst_pitch=*/8, device_pointer_,
      /*src_pitch=*/8, /*width=*/4, /*height=*/2, stream_));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
  const auto* destination = static_cast<const uint8_t*>(host_buffer_->host_ptr);
  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_EQ(destination[i], i % 8 < 4 ? source[i] : 0xcd);
  }
}

TEST_F(CpuStreamingMemoryTest, PageableUpdatesCaptureBytesBeforeReturning) {
  constexpr size_t kAllocationSize = 256 * 1024 + 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kAllocationSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
  InjectedFlushQueue queue = {};
  iree_hal_queue_t* original_queue = stream_->queue;
  InitializeInjectedFlushQueue(original_queue, &queue);
  queue.execute_gate = gate;
  EXPECT_EQ(original_queue, ReplaceStreamQueue(stream_, &queue.base));
  iree_hal_queue_release(original_queue);

  for (size_t length : {4096u, 65539u, 256u * 1024u}) {
    SCOPED_TRACE(length);
    std::vector<uint8_t> source(length);
    std::vector<uint8_t> expected(kAllocationSize, 0xcd);
    for (size_t i = 0; i < length; ++i) {
      source[i] = static_cast<uint8_t>(i);
      expected[3 + i] = source[i];
    }
    std::memset(buffer_->host_ptr, 0xcd, kAllocationSize);
    ++queue.execute_gate_value;
    const int accepted_before = queue.accepted_transfer_count.load();
    IREE_EXPECT_OK(iree_hal_streaming_memcpy_host_to_device(
        context_, device_pointer_ + 3, source.data(), length, stream_));
    EXPECT_EQ(queue.accepted_transfer_count.load(), accepted_before + 1);
    EXPECT_EQ(queue.injected_execute_count.load(), 0);
    EXPECT_EQ(stream_->command_buffer, nullptr);
    EXPECT_EQ(static_cast<const uint8_t*>(buffer_->host_ptr)[3], 0xcd);
    std::memset(source.data(), 0xee, source.size());
    IREE_EXPECT_OK(
        iree_hal_semaphore_signal(gate, queue.execute_gate_value, nullptr));
    IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
    EXPECT_EQ(0,
              std::memcmp(buffer_->host_ptr, expected.data(), expected.size()));
  }

  iree_hal_queue_retain(original_queue);
  iree_hal_queue_release(ReplaceStreamQueue(stream_, original_queue));
  iree_hal_semaphore_release(gate);
}

TEST_F(CpuStreamingMemoryTest,
       PostAcceptFlushFailureJoinsCapturedUploadSnapshots) {
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, 32, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
  InjectedFlushQueue queue = {};
  iree_hal_queue_t* original_queue = stream_->queue;
  InitializeInjectedFlushQueue(original_queue, &queue);
  queue.execute_gate = gate;
  queue.fail_flush = true;
  EXPECT_EQ(original_queue, ReplaceStreamQueue(stream_, &queue.base));
  iree_hal_queue_release(original_queue);

  std::array<uint8_t, 16> source;
  source.fill(0x42);
  std::memset(buffer_->host_ptr, 0xcd, 32);
  ++queue.execute_gate_value;
  std::atomic<bool> returned = false;
  std::thread caller([&] {
    iree_status_t status = iree_hal_streaming_memcpy_host_to_device(
        context_, device_pointer_, source.data(), 16, stream_);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL, status);
    returned.store(true, std::memory_order_release);
  });
  // The real task queue has captured the source but cannot execute it yet.
  queue.flush_observed.acquire();
  EXPECT_FALSE(returned.load(std::memory_order_acquire));
  EXPECT_EQ(static_cast<const uint8_t*>(buffer_->host_ptr)[0], 0xcd);
  source.fill(0xee);
  queue.fail_flush = false;
  IREE_EXPECT_OK(
      iree_hal_semaphore_signal(gate, queue.execute_gate_value, nullptr));
  caller.join();
  IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
  const auto* bytes = static_cast<const uint8_t*>(buffer_->host_ptr);
  for (size_t i = 0; i < 32; ++i) {
    EXPECT_EQ(bytes[i], i < 16 ? 0x42 : 0xcd);
  }

  iree_hal_queue_retain(original_queue);
  iree_hal_queue_release(ReplaceStreamQueue(stream_, original_queue));
  iree_hal_semaphore_release(gate);
}

TEST_F(CpuStreamingMemoryTest,
       DownloadsValidateAllocationExtentsBeforeSubmitting) {
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, 16, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_host(
      context_, 8, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT,
      /*fine_grained=*/false, &host_buffer_));
  const uint64_t pending_value = stream_->pending_value;
  std::array<uint8_t, 16> pageable_destination = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_streaming_memcpy_device_to_host(
                            context_, pageable_destination.data(),
                            device_pointer_ + 8, 9, stream_));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_streaming_memcpy_device_to_host(
          context_, static_cast<uint8_t*>(host_buffer_->host_ptr) + 4,
          device_pointer_, 8, stream_));
  EXPECT_EQ(stream_->pending_value, pending_value);
  EXPECT_EQ(host_callbacks_.thread, nullptr);
}

TEST_F(CpuStreamingMemoryTest,
       SynchronousMappedDeviceUploadUsesNativeTransfer) {
  constexpr size_t kLength = 64;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kLength, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  ASSERT_NE(buffer_->host_ptr, nullptr);
  std::memset(buffer_->host_ptr, 0xcd, kLength);
  const std::array<uint8_t, 7> source = {1, 19, 3, 67, 5, 241, 7};
  InjectedFlushQueue queue = {};
  auto* original = context_->queue;
  InitializeInjectedFlushQueue(original, &queue);
  context_->queue = &queue.base;
  auto status = iree_hal_streaming_memcpy_host_to_device(
      context_, iree_hal_streaming_buffer_device_pointer(buffer_) + 3,
      source.data(), source.size(), nullptr);
  context_->queue = original;
  iree_hal_queue_release(&queue.base);
  IREE_ASSERT_OK(status);
  EXPECT_EQ(queue.upload_source, source.data());
  const auto* bytes = static_cast<const uint8_t*>(buffer_->host_ptr);
  for (size_t i = 0; i < kLength; ++i) {
    EXPECT_EQ(bytes[i], i >= 3 && i < 3 + source.size() ? source[i - 3] : 0xcd);
  }
}

TEST_F(CpuStreamingMemoryTest, PageableUploadBorrowsSourceWithoutContextLock) {
  constexpr size_t kLength = 1024 * 1024 + 123;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kLength + 16, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  std::memset(buffer_->host_ptr, 0xcd, kLength + 16);
  std::vector<uint8_t> source(kLength, 0x42);
  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
  std::binary_semaphore accepted(0);
  InjectedFlushQueue queue = {};
  iree_hal_queue_t* original_queue = stream_->queue;
  InitializeInjectedFlushQueue(original_queue, &queue);
  queue.execute_gate = gate;
  queue.execute_gate_value = 1;
  queue.transfer_accepted = &accepted;
  EXPECT_EQ(original_queue, ReplaceStreamQueue(stream_, &queue.base));
  iree_hal_queue_release(original_queue);

  std::atomic<bool> returned(false);
  std::thread caller([&] {
    IREE_EXPECT_OK(iree_hal_streaming_memcpy_host_to_device(
        context_, iree_hal_streaming_buffer_device_pointer(buffer_),
        source.data(), source.size(), stream_));
    returned.store(true, std::memory_order_release);
  });
  accepted.acquire();
  EXPECT_EQ(queue.upload_source, source.data());
  EXPECT_FALSE(returned.load(std::memory_order_acquire));
  const bool acquired = iree_slim_mutex_try_lock(&context_->mutex);
  EXPECT_TRUE(acquired);
  if (acquired) {
    iree_slim_mutex_unlock(&context_->mutex);
  }

  // An independent task queue operation can finish while this upload waits.
  // It needs neither the borrowed source nor the upload's reserved stream tail.
  iree_hal_semaphore_t* independent = nullptr;
  IREE_EXPECT_OK(iree_hal_semaphore_create(
      context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &independent));
  if (independent) {
    uint64_t value = 1;
    const iree_hal_semaphore_list_t signals = {1, &independent, &value};
    iree_status_t status = iree_hal_queue_barrier(
        original_queue, iree_hal_semaphore_list_empty(), signals, nullptr,
        IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
    if (iree_status_is_ok(status)) {
      status =
          iree_hal_semaphore_wait(independent, value, iree_infinite_timeout(),
                                  IREE_ASYNC_WAIT_FLAG_NONE);
    }
    IREE_EXPECT_OK(status);
    iree_hal_semaphore_release(independent);
  }
  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  caller.join();
  EXPECT_EQ(0, std::memcmp(buffer_->host_ptr, source.data(), kLength));
  const auto* bytes = static_cast<const uint8_t*>(buffer_->host_ptr);
  for (size_t i = kLength; i < kLength + 16; ++i) {
    EXPECT_EQ(bytes[i], 0xcd);
  }
  iree_hal_queue_retain(original_queue);
  iree_hal_queue_release(ReplaceStreamQueue(stream_, original_queue));
  iree_hal_semaphore_release(gate);
  EXPECT_EQ(host_callbacks_.thread, nullptr);
}

TEST_F(CpuStreamingMemoryTest, PageableUploadValidationAndRejection) {
  constexpr size_t kLength = 1024 * 1024 + 123;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kLength, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  std::memset(buffer_->host_ptr, 0xcd, kLength);
  std::vector<uint8_t> source(kLength + 1, 0x42);
  const uint64_t pending_before = stream_->pending_value;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_streaming_memcpy_host_to_device(
          context_, device_pointer_, source.data(), source.size(), stream_));
  EXPECT_EQ(stream_->pending_value, pending_before);

  InjectedFlushQueue queue = {};
  iree_hal_queue_t* original_queue = stream_->queue;
  InitializeInjectedFlushQueue(original_queue, &queue);
  queue.fail_transfer = true;
  EXPECT_EQ(original_queue, ReplaceStreamQueue(stream_, &queue.base));
  iree_hal_queue_release(original_queue);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_streaming_memcpy_host_to_device(
          context_, device_pointer_, source.data(), kLength, stream_));
  EXPECT_EQ(queue.accepted_transfer_count.load(), 0);
  const auto* bytes = static_cast<const uint8_t*>(buffer_->host_ptr);
  for (size_t i = 0; i < kLength; ++i) {
    EXPECT_EQ(bytes[i], 0xcd);
  }
  uint64_t value = 0;
  IREE_EXPECT_OK(iree_hal_semaphore_query(stream_->timeline_semaphore, &value));
  EXPECT_EQ(stream_->pending_value, pending_before);
  iree_hal_queue_retain(original_queue);
  iree_hal_queue_release(ReplaceStreamQueue(stream_, original_queue));
  IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
}

TEST_F(CpuStreamingMemoryTest,
       PageableTransfersAcceptBeforeWaitingForPredecessor) {
  constexpr size_t kLength = 1024 * 1024 + 123;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kLength + 16, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  std::vector<uint8_t> host_bytes(kLength + 16);
  for (const bool upload : {true, false}) {
    SCOPED_TRACE(upload ? "upload" : "download");
    std::memset(buffer_->host_ptr, upload ? 0xcd : 0x42, kLength);
    std::memset(static_cast<uint8_t*>(buffer_->host_ptr) + kLength, 0xcd, 16);
    std::memset(host_bytes.data(), upload ? 0x42 : 0xcd, kLength);
    std::memset(host_bytes.data() + kLength, 0xcd, 16);
    iree_hal_semaphore_t* gate = nullptr;
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
    std::binary_semaphore accepted(0);
    InjectedFlushQueue queue = {};
    auto* original_queue = stream_->queue;
    InitializeInjectedFlushQueue(original_queue, &queue);
    EXPECT_EQ(original_queue, ReplaceStreamQueue(stream_, &queue.base));
    iree_hal_queue_release(original_queue);
    queue.execute_gate = gate;
    queue.execute_gate_value = 1;
    IREE_ASSERT_OK(
        iree_hal_streaming_direct_transfer_enqueue(stream_, 0, nullptr));
    const uint64_t predecessor = stream_->pending_value;
    queue.execute_gate = nullptr;
    queue.transfer_accepted = &accepted;
    std::atomic<bool> returned(false);
    std::thread caller([&] {
      IREE_EXPECT_OK(upload ? iree_hal_streaming_memcpy_host_to_device(
                                  context_, device_pointer_, host_bytes.data(),
                                  kLength, stream_)
                            : iree_hal_streaming_memcpy_device_to_host(
                                  context_, host_bytes.data(), device_pointer_,
                                  kLength, stream_));
      returned.store(true, std::memory_order_release);
    });

    // Native acceptance carries the unresolved predecessor; the caller waits
    // only after its copy and stream signal have been accepted together.
    accepted.acquire();
    EXPECT_FALSE(returned.load(std::memory_order_acquire));
    EXPECT_EQ(queue.accepted_wait_semaphore, stream_->timeline_semaphore);
    EXPECT_EQ(queue.accepted_wait_value, predecessor);
    EXPECT_EQ(queue.accepted_signal_semaphore, stream_->timeline_semaphore);
    EXPECT_EQ(queue.accepted_signal_value, predecessor + 1);
    EXPECT_EQ(queue.accepted_signal_count, 2u);

    // A later stream dependency may remain pending after this copy returns.
    queue.transfer_accepted = nullptr;
    queue.execute_gate = gate;
    queue.execute_gate_value = 2;
    IREE_EXPECT_OK(
        iree_hal_streaming_direct_transfer_enqueue(stream_, 0, nullptr));
    IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
    caller.join();
    EXPECT_TRUE(returned.load(std::memory_order_acquire));
    EXPECT_EQ(0, std::memcmp(buffer_->host_ptr, host_bytes.data(), kLength));
    for (size_t i = kLength; i < kLength + 16; ++i) {
      EXPECT_EQ(static_cast<const uint8_t*>(buffer_->host_ptr)[i], 0xcd);
      EXPECT_EQ(host_bytes[i], 0xcd);
    }
    IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, 2, nullptr));
    IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
    iree_hal_queue_retain(original_queue);
    iree_hal_queue_release(ReplaceStreamQueue(stream_, original_queue));
    iree_hal_semaphore_release(gate);
  }
  EXPECT_EQ(host_callbacks_.thread, nullptr);
}

TEST_F(CpuStreamingMemoryTest, PageableDownloadExtendsActiveStreamQueueLease) {
  constexpr size_t kLength = IREE_HAL_STREAMING_SMALL_TRANSFER_MAX_LENGTH + 1;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kLength, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  std::memset(buffer_->host_ptr, 0x42, kLength);
  std::vector<uint8_t> destination(kLength);
  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
  uint64_t gate_value = 1;
  IREE_ASSERT_OK(iree_hal_streaming_stream_wait_semaphores(
      stream_, {1, &gate, &gate_value}));
  auto* lane = stream_->value_wait_lane;
  ASSERT_NE(nullptr, lane);
  const uint64_t predecessor = stream_->pending_value;
  std::atomic<bool> returned(false);
  std::thread caller([&] {
    IREE_EXPECT_OK(iree_hal_streaming_memcpy_device_to_host(
        context_, destination.data(),
        iree_hal_streaming_buffer_device_pointer(buffer_), kLength, stream_));
    returned.store(true, std::memory_order_release);
  });
  // Both accepted operations remain pending on the real task queue. The copy
  // must extend the existing lease before its unresolved predecessor completes.
  while (!returned.load(std::memory_order_acquire) &&
         iree_atomic_load(&context_->active_value_wait_observer_count,
                          iree_memory_order_acquire) < 2) {
    std::this_thread::yield();
  }
  iree_slim_mutex_lock(&stream_->mutex);
  EXPECT_EQ(stream_->value_wait_lane, lane);
  EXPECT_EQ(stream_->pending_value, predecessor + 1);
  iree_slim_mutex_unlock(&stream_->mutex);
  EXPECT_FALSE(returned.load(std::memory_order_acquire));
  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  caller.join();
  EXPECT_EQ(context_->transfer_queue, nullptr);
  for (uint8_t value : destination) {
    EXPECT_EQ(value, 0x42);
  }
  IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
  iree_hal_semaphore_release(gate);
}

TEST_F(CpuStreamingMemoryTest, BulkCopiesReuseOneContextQueue) {
  constexpr size_t kSmall = IREE_HAL_STREAMING_SMALL_TRANSFER_MAX_LENGTH;
  constexpr size_t kLength = kSmall + 17;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kLength + 16, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  const auto pointer = iree_hal_streaming_buffer_device_pointer(buffer_);
  std::memset(buffer_->host_ptr, 0x42, kLength + 16);
  std::vector<uint8_t> host(kLength + 16, 0xcd);
  IREE_ASSERT_OK(iree_hal_streaming_memcpy_device_to_host(
      context_, host.data(), pointer, kSmall, stream_));
  EXPECT_EQ(context_->transfer_queue, nullptr);
  std::memset(host.data(), 0xa5, kLength);
  IREE_ASSERT_OK(iree_hal_streaming_direct_transfer_h2d(
      context_, host.data(), buffer_->buffer, 0, kLength));
  auto* transfer_queue = context_->transfer_queue;
  ASSERT_NE(transfer_queue, nullptr);
  EXPECT_NE(transfer_queue, stream_->queue);
  EXPECT_EQ(0, std::memcmp(host.data(), buffer_->host_ptr, kLength));
  std::memset(host.data(), 0xcd, kLength);
  IREE_ASSERT_OK(iree_hal_streaming_memcpy_device_to_host(
      context_, host.data(), pointer, kLength, stream_));
  EXPECT_EQ(context_->transfer_queue, transfer_queue);
  EXPECT_EQ(0, std::memcmp(host.data(), buffer_->host_ptr, kLength));
  for (size_t i = kLength; i < host.size(); ++i) {
    EXPECT_EQ(host[i], 0xcd);
  }

  std::memset(host.data(), 0x3c, kLength);
  IREE_ASSERT_OK(iree_hal_streaming_memcpy_host_to_device(
      context_, pointer, host.data(), kLength, stream_));
  EXPECT_EQ(context_->transfer_queue, transfer_queue);
  EXPECT_EQ(0, std::memcmp(host.data(), buffer_->host_ptr, kLength));
  for (size_t i = kLength; i < host.size(); ++i) {
    EXPECT_EQ(static_cast<const uint8_t*>(buffer_->host_ptr)[i], 0x42);
  }
  std::memset(host.data(), 0xcd, kLength);
  IREE_ASSERT_OK(iree_hal_streaming_direct_transfer_d2h(
      context_, buffer_->buffer, 0, host.data(), kLength));
  EXPECT_EQ(context_->transfer_queue, transfer_queue);
  EXPECT_EQ(0, std::memcmp(host.data(), buffer_->host_ptr, kLength));
  for (size_t i = kLength; i < host.size(); ++i) {
    EXPECT_EQ(host[i], 0xcd);
  }
  EXPECT_EQ(host_callbacks_.thread, nullptr);
}

TEST_F(CpuStreamingMemoryTest, BulkBatchCountsItsSmallRows) {
  constexpr size_t kRowLength =
      IREE_HAL_STREAMING_SMALL_TRANSFER_MAX_LENGTH / 2 + 1;
  constexpr size_t kLength = 2 * kRowLength;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kLength, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  std::memset(buffer_->host_ptr, 0x42, kLength);
  std::vector<uint8_t> host(kLength + 16, 0xcd);
  iree_hal_transfer_operation_t operations[2] = {};
  for (size_t i = 0; i < 2; ++i) {
    operations[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    operations[i].download.source_buffer = buffer_->buffer;
    operations[i].download.source_offset = i * kRowLength;
    operations[i].download.target = host.data() + i * kRowLength;
    operations[i].download.length = kRowLength;
  }
  IREE_ASSERT_OK(
      iree_hal_streaming_direct_transfer_and_wait(stream_, 2, operations));
  ASSERT_NE(context_->transfer_queue, nullptr);
  EXPECT_NE(context_->transfer_queue, stream_->queue);
  EXPECT_EQ(0, std::memcmp(host.data(), buffer_->host_ptr, kLength));
  for (size_t i = kLength; i < host.size(); ++i) {
    EXPECT_EQ(host[i], 0xcd);
  }
}

TEST_F(CpuStreamingMemoryTest, BulkCopiesPreserveExplicitStreamQueue) {
  constexpr size_t kLength = IREE_HAL_STREAMING_SMALL_TRANSFER_MAX_LENGTH + 1;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kLength, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  std::memset(buffer_->host_ptr, 0x42, kLength);
  std::vector<uint8_t> host(kLength);
  iree_hal_queue_t* explicit_queue = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_device_acquire_transfer_queue(
      &device_entry_, context_->queue, &explicit_queue));
  auto* original_queue = ReplaceStreamQueue(stream_, explicit_queue);
  IREE_EXPECT_OK(iree_hal_streaming_memcpy_device_to_host(
      context_, host.data(), iree_hal_streaming_buffer_device_pointer(buffer_),
      kLength, stream_));
  EXPECT_EQ(context_->transfer_queue, nullptr);
  EXPECT_EQ(0, std::memcmp(host.data(), buffer_->host_ptr, kLength));
  iree_hal_queue_release(ReplaceStreamQueue(stream_, original_queue));
}

TEST_F(CpuStreamingMemoryTest, SynchronizeRetainedDetachedStream) {
  // Model context synchronization retaining a snapshot before another thread
  // destroys the public stream. The snapshot remains valid after detachment.
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
  iree_hal_streaming_context_unregister_stream(context_, stream_);
  ASSERT_EQ(nullptr, stream_->context);
  IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize_flushed(stream_));
  iree_hal_streaming_stream_release(stream_);
  stream_ = nullptr;
}

TEST_F(CpuStreamingMemoryTest,
       PublicStreamFailureDoesNotReturnBorrowedDownloadEarly) {
  constexpr size_t kLength = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kLength, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  std::memset(buffer_->host_ptr, 0x42, kLength);
  std::array<uint8_t, kLength> destination = {};
  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
  std::binary_semaphore accepted(0);
  InjectedFlushQueue queue = {};
  auto* original_queue = stream_->queue;
  InitializeInjectedFlushQueue(original_queue, &queue);
  queue.execute_gate = gate;
  queue.execute_gate_value = 1;
  queue.transfer_accepted = &accepted;
  EXPECT_EQ(original_queue, ReplaceStreamQueue(stream_, &queue.base));
  iree_hal_queue_release(original_queue);
  std::atomic<bool> returned(false);
  std::thread caller([&] {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                          iree_hal_streaming_memcpy_device_to_host(
                              context_, destination.data(),
                              iree_hal_streaming_buffer_device_pointer(buffer_),
                              kLength, stream_));
    returned.store(true, std::memory_order_release);
  });
  accepted.acquire();
  iree_hal_semaphore_fail(
      stream_->timeline_semaphore,
      iree_make_status(IREE_STATUS_ABORTED, "later stream operation failed"));
  EXPECT_FALSE(returned.load(std::memory_order_acquire));
  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  caller.join();
  for (uint8_t value : destination) {
    EXPECT_EQ(value, 0x42);
  }
  iree_hal_queue_retain(original_queue);
  iree_hal_queue_release(ReplaceStreamQueue(stream_, original_queue));
  iree_hal_semaphore_release(gate);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_hal_streaming_stream_synchronize(stream_));
  iree_hal_streaming_context_unregister_stream(context_, stream_);
  iree_hal_streaming_stream_release(stream_);
  stream_ = nullptr;
}

TEST_F(CpuStreamingMemoryTest,
       PageableTransferFlushFailureJoinsBorrowedHostBytes) {
  constexpr size_t kLength = 1024 * 1024 + 123;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kLength, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  std::vector<uint8_t> host_bytes(kLength);
  for (const bool upload : {true, false}) {
    SCOPED_TRACE(upload ? "upload" : "download");
    std::memset(buffer_->host_ptr, upload ? 0xcd : 0x42, kLength);
    std::memset(host_bytes.data(), upload ? 0x42 : 0xcd, kLength);
    iree_hal_semaphore_t* gate = nullptr;
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
    InjectedFlushQueue queue = {};
    auto* original_queue = stream_->queue;
    InitializeInjectedFlushQueue(original_queue, &queue);
    queue.execute_gate = gate;
    queue.execute_gate_value = 1;
    queue.fail_flush = true;
    EXPECT_EQ(original_queue, ReplaceStreamQueue(stream_, &queue.base));
    iree_hal_queue_release(original_queue);
    std::atomic<bool> returned(false);
    std::thread caller([&] {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                            upload ? iree_hal_streaming_memcpy_host_to_device(
                                         context_, device_pointer_,
                                         host_bytes.data(), kLength, stream_)
                                   : iree_hal_streaming_memcpy_device_to_host(
                                         context_, host_bytes.data(),
                                         device_pointer_, kLength, stream_));
      returned.store(true, std::memory_order_release);
    });
    queue.flush_observed.acquire();
    EXPECT_FALSE(returned.load(std::memory_order_acquire));
    IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
    caller.join();
    EXPECT_EQ(0, std::memcmp(buffer_->host_ptr, host_bytes.data(), kLength));
    iree_hal_queue_retain(original_queue);
    iree_hal_queue_release(ReplaceStreamQueue(stream_, original_queue));
    iree_hal_semaphore_release(gate);
    IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
  }
}

TEST_F(CpuStreamingMemoryTest,
       RegisteredDownloadsRemainAsyncAndProgressPastBlockedCallbacks) {
  constexpr size_t kSourceSize = 16;
  constexpr size_t kDestinationSize = 32;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kSourceSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  const std::array<uint8_t, kSourceSize> source = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::memcpy(buffer_->host_ptr, source.data(), source.size());
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_host(
      context_, kDestinationSize, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT,
      /*fine_grained=*/false, &host_buffer_));
  std::array<uint8_t, kDestinationSize> registered_storage;
  iree_hal_streaming_buffer_t* registered_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_memory_register_host(
      context_, registered_storage.data(), registered_storage.size(),
      IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT, &registered_buffer));
  const auto release_buffer = [&](iree_hal_streaming_buffer_t* buffer) {
    IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
    iree_hal_streaming_memory_release_wrapped_buffer(buffer);
  };
  std::unique_ptr<iree_hal_streaming_buffer_t, decltype(release_buffer)>
      registered_owner(registered_buffer, release_buffer);
  const std::array<void*, 2> destinations = {host_buffer_->host_ptr,
                                             registered_storage.data()};
  for (void* destination : destinations) {
    std::memset(destination, 0xcd, kDestinationSize);
  }

  iree_hal_streaming_stream_t* callback_stream = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_stream_create(
      context_, context_->queue, IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING, 0,
      iree_allocator_system(), &callback_stream));
  ScopedHeldCallback callback(callback_stream);

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
  const auto release_gate = [](iree_hal_semaphore_t* semaphore) {
    IREE_EXPECT_OK(iree_hal_semaphore_signal(semaphore, 1, nullptr));
    iree_hal_semaphore_release(semaphore);
  };
  std::unique_ptr<iree_hal_semaphore_t, decltype(release_gate)> copy_gate(
      semaphore, release_gate);
  uint64_t gate_value = 1;
  const iree_hal_semaphore_list_t gate_wait = {1, &semaphore, &gate_value};
  IREE_ASSERT_OK(iree_hal_streaming_stream_wait_semaphores(stream_, gate_wait));

  IREE_ASSERT_OK(callback.Start());
  callback.WaitUntilEntered();

  for (void* destination : destinations) {
    IREE_ASSERT_OK(iree_hal_streaming_memcpy_device_to_host(
        context_, destination, device_pointer_, kSourceSize, stream_));
    IREE_ASSERT_OK(iree_hal_streaming_memcpy_device_to_host_2d(
        context_, static_cast<uint8_t*>(destination) + kSourceSize,
        /*dst_pitch=*/8, device_pointer_, /*src_pitch=*/8, /*width=*/4,
        /*height=*/2, stream_));
    // The stream's unsatisfied wait proves these APIs did not wait for their
    // copies or modify the destination on their calling thread.
    for (size_t i = 0; i < kDestinationSize; ++i) {
      EXPECT_EQ(static_cast<const uint8_t*>(destination)[i], 0xcd);
    }
  }
  copy_gate.reset();
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
  EXPECT_FALSE(callback.returned());
  for (const void* destination : destinations) {
    const auto* bytes = static_cast<const uint8_t*>(destination);
    EXPECT_EQ(0, std::memcmp(bytes, source.data(), source.size()));
    for (size_t i = 0; i < kSourceSize; ++i) {
      EXPECT_EQ(bytes[kSourceSize + i], i % 8 < 4 ? source[i] : 0xcd);
    }
  }
}

TEST_F(CpuStreamingMemoryTest,
       CapturedRegisteredCopiesProgressPastBlockedCallbacks) {
  constexpr size_t kSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  IREE_ASSERT_OK(iree_hal_streaming_memory_register_host(
      context_, registered_storage_.data(), registered_storage_.size(),
      IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT, &host_buffer_));
  iree_hal_streaming_buffer_t* input_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_host(
      context_, kSize, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT,
      /*fine_grained=*/false, &input_buffer));
  const auto release_input = [&](iree_hal_streaming_buffer_t* buffer) {
    IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
    iree_hal_streaming_memory_release_wrapped_buffer(buffer);
  };
  std::unique_ptr<iree_hal_streaming_buffer_t, decltype(release_input)>
      input_owner(input_buffer, release_input);

  IREE_ASSERT_OK(iree_hal_streaming_begin_capture(
      stream_, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  iree_status_t status = iree_hal_streaming_memcpy_host_to_device(
      context_, device_pointer_, input_buffer->host_ptr, kSize, stream_);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_memcpy_device_to_host(
        context_, registered_storage_.data(), device_pointer_, kSize, stream_);
  }
  iree_hal_streaming_graph_t* graph = nullptr;
  status =
      iree_status_join(status, iree_hal_streaming_end_capture(stream_, &graph));
  std::unique_ptr<iree_hal_streaming_graph_t,
                  decltype(&iree_hal_streaming_graph_release)>
      graph_owner(graph, iree_hal_streaming_graph_release);
  IREE_ASSERT_OK(status);
  ASSERT_NE(nullptr, graph);
  // Registered copies are native graph nodes, with no hidden CPU copy jobs or
  // transfer-sized staging allocations.
  EXPECT_EQ(graph->node_count, 2u);
  iree_hal_streaming_graph_exec_t* executable = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &executable));
  std::unique_ptr<iree_hal_streaming_graph_exec_t,
                  decltype(&iree_hal_streaming_graph_exec_release)>
      executable_owner(executable, iree_hal_streaming_graph_exec_release);

  iree_hal_streaming_stream_t* callback_stream = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_stream_create(
      context_, context_->queue, IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING, 0,
      iree_allocator_system(), &callback_stream));
  ScopedHeldCallback callback(callback_stream);
  IREE_ASSERT_OK(callback.Start());
  callback.WaitUntilEntered();

  for (uint8_t value : {0x12, 0x34}) {
    std::memset(input_buffer->host_ptr, value, kSize);
    registered_storage_.fill(0xcd);
    iree_hal_streaming_graph_exec_launch_result_t launch_result;
    IREE_ASSERT_OK(iree_hal_streaming_graph_exec_launch(executable, stream_,
                                                        &launch_result));
    EXPECT_EQ(IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS, launch_result);
    IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
    EXPECT_FALSE(callback.returned());
    for (uint8_t byte : registered_storage_) {
      EXPECT_EQ(byte, value);
    }
  }
}

TEST_F(CpuStreamingMemoryTest, ExecutableCopiesOwnDistinctStaging) {
  constexpr size_t kSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  std::array<uint8_t, kSize> source;
  source.fill(0x37);
  std::array<uint8_t, kSize> output = {};
  IREE_ASSERT_OK(iree_hal_streaming_begin_capture(
      stream_, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  iree_status_t status = iree_hal_streaming_memcpy_host_to_device(
      context_, device_pointer_, source.data(), source.size(), stream_);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_memcpy_device_to_host(
        context_, output.data(), device_pointer_, output.size(), stream_);
  }
  iree_hal_streaming_graph_t* graph = nullptr;
  status =
      iree_status_join(status, iree_hal_streaming_end_capture(stream_, &graph));
  std::unique_ptr<iree_hal_streaming_graph_t,
                  decltype(&iree_hal_streaming_graph_release)>
      graph_owner(graph, iree_hal_streaming_graph_release);
  IREE_ASSERT_OK(status);
  iree_hal_streaming_graph_t* clone = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_clone(
      graph, IREE_HAL_STREAMING_GRAPH_CLONE_MODE_TEMPLATE, &clone));
  std::unique_ptr<iree_hal_streaming_graph_t,
                  decltype(&iree_hal_streaming_graph_release)>
      clone_owner(clone, iree_hal_streaming_graph_release);
  graph_owner.reset();

  iree_hal_streaming_graph_exec_t* first = nullptr;
  iree_hal_streaming_graph_exec_t* second = nullptr;
  const auto release_executable = [&](iree_hal_streaming_graph_exec_t* value) {
    IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
    iree_hal_streaming_graph_exec_release(value);
  };
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      clone, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &first));
  std::unique_ptr<iree_hal_streaming_graph_exec_t, decltype(release_executable)>
      first_owner(first, release_executable);
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      clone, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &second));
  std::unique_ptr<iree_hal_streaming_graph_exec_t, decltype(release_executable)>
      second_owner(second, release_executable);
  size_t staged_copies = 0;
  for (auto* block = clone->node_blocks; block; block = block->next) {
    for (size_t i = 0; i < block->count; ++i) {
      auto* node = block->nodes[i];
      if (node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY) {
        continue;
      }
      ++staged_copies;
      const bool upload = node->attrs.memcpy.staging.direction ==
                          IREE_HAL_STREAMING_GRAPH_STAGING_UPLOAD;
      EXPECT_EQ(nullptr, upload ? node->attrs.memcpy.src_ref.buffer
                                : node->attrs.memcpy.dst_ref.buffer);
      auto* first_node = iree_hal_streaming_graph_exec_lock_node(first, node);
      auto* second_node = iree_hal_streaming_graph_exec_lock_node(second, node);
      auto* first_buffer = upload ? first_node->attrs.memcpy.src_ref.buffer
                                  : first_node->attrs.memcpy.dst_ref.buffer;
      auto* second_buffer = upload ? second_node->attrs.memcpy.src_ref.buffer
                                   : second_node->attrs.memcpy.dst_ref.buffer;
      EXPECT_NE(nullptr, first_buffer);
      EXPECT_NE(nullptr, second_buffer);
      EXPECT_NE(first_buffer, second_buffer);
      EXPECT_NE(first_buffer->host_ptr, second_buffer->host_ptr);
      iree_hal_streaming_graph_exec_unlock(second);
      iree_hal_streaming_graph_exec_unlock(first);
    }
  }
  EXPECT_EQ(staged_copies, 2u);
  for (auto* executable : {first, second, first}) {
    iree_hal_streaming_graph_exec_launch_result_t launch_result;
    IREE_ASSERT_OK(iree_hal_streaming_graph_exec_launch(executable, stream_,
                                                        &launch_result));
    EXPECT_EQ(launch_result, IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS);
    IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
    EXPECT_EQ(output, source);
    source.fill(static_cast<uint8_t>(source[0] + 1));
  }
}

TEST_F(CpuStreamingMemoryTest,
       CapturedHostCopiesRejectRegisteredRangeOverruns) {
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, 16, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  IREE_ASSERT_OK(iree_hal_streaming_memory_register_host(
      context_, registered_storage_.data(), registered_storage_.size(),
      IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT, &host_buffer_));
  registered_storage_.fill(0xcd);

  for (int operation : {0, 1}) {
    IREE_ASSERT_OK(iree_hal_streaming_begin_capture(
        stream_, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
    iree_status_t status;
    switch (operation) {
      case 0:
        status = iree_hal_streaming_memcpy_host_to_device(
            context_, device_pointer_, registered_storage_.data() + 8, 9,
            stream_);
        break;
      default:
        status = iree_hal_streaming_memcpy_device_to_host(
            context_, registered_storage_.data() + 8, device_pointer_, 9,
            stream_);
        break;
    }
    IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, status);
    iree_hal_streaming_graph_t* graph = nullptr;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                          iree_hal_streaming_end_capture(stream_, &graph));
    EXPECT_EQ(nullptr, graph);
    iree_hal_streaming_graph_release(graph);
    for (uint8_t byte : registered_storage_) {
      EXPECT_EQ(0xcd, byte);
    }
  }
}

TEST_F(CpuStreamingMemoryTest, RegisteredPitchedCopiesSubmitOneBatchPerCall) {
  constexpr size_t kWidth = 5, kHeight = 3;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, 128, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_host(
      context_, 64, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT,
      /*fine_grained=*/false, &host_buffer_));
  auto* host = static_cast<uint8_t*>(host_buffer_->host_ptr);
  std::memset(host, 0xCD, 64);
  for (size_t row = 0; row < kHeight; ++row) {
    for (size_t column = 0; column < kWidth; ++column) {
      host[row * 8 + column] = static_cast<uint8_t>(row * 17 + column);
    }
  }

  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
  InjectedFlushQueue queue = {};
  iree_hal_queue_t* original_queue = stream_->queue;
  InitializeInjectedFlushQueue(original_queue, &queue);
  queue.execute_gate = gate;
  queue.execute_gate_value = 1;
  EXPECT_EQ(original_queue, ReplaceStreamQueue(stream_, &queue.base));
  iree_hal_queue_release(original_queue);

  // Every operation remains pending at this point. A later operation must
  // depend on its producer through the stream timeline, not submission order.
  IREE_EXPECT_OK(iree_hal_streaming_memcpy_host_to_device_2d(
      context_, device_pointer_, 12, host, 8, kWidth, kHeight, stream_));
  IREE_EXPECT_OK(iree_hal_streaming_memcpy_device_to_device_2d(
      context_, device_pointer_ + 64, 9, device_pointer_, 12, kWidth, kHeight,
      stream_));
  IREE_EXPECT_OK(iree_hal_streaming_memcpy_device_to_host_2d(
      context_, host + 32, 8, device_pointer_ + 64, 9, kWidth, kHeight,
      stream_));
  EXPECT_EQ(3, queue.accepted_transfer_count.load());
  EXPECT_EQ(0, queue.injected_execute_count.load());
  EXPECT_EQ(nullptr, stream_->command_buffer);
  for (size_t i = 32; i < 64; ++i) {
    EXPECT_EQ(host[i], 0xCD);
  }

  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
  for (size_t i = 0; i < 32; ++i) {
    const uint8_t expected = i / 8 < kHeight && i % 8 < kWidth
                                 ? static_cast<uint8_t>((i / 8) * 17 + i % 8)
                                 : 0xCD;
    EXPECT_EQ(host[32 + i], expected);
  }
  iree_hal_queue_retain(original_queue);
  iree_hal_queue_release(ReplaceStreamQueue(stream_, original_queue));
  iree_hal_semaphore_release(gate);
}

TEST_F(CpuStreamingMemoryTest, PitchedCopyRejectsAllRowsBeforeQueueAcceptance) {
  constexpr iree_device_size_t kAllocationSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kAllocationSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  ASSERT_NE(nullptr, buffer_->host_ptr);

  const std::array<uint8_t, kAllocationSize> initial = {
      1, 2, 3, 4, 21, 22, 23, 24, 41, 42, 43, 44, 61, 62, 63, 64};
  std::memcpy(buffer_->host_ptr, initial.data(), initial.size());

  // Row zero is disjoint, but row one overlaps itself. Whole-batch validation
  // rejects both rows without publishing a new stream point.
  const uint64_t pending_before = stream_->pending_value;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_streaming_memcpy_device_to_device_2d(
          context_, device_pointer_ + 4, /*dst_pitch=*/4, device_pointer_,
          /*src_pitch=*/8, /*width=*/4, /*height=*/2, stream_));

  const auto* contents = static_cast<const uint8_t*>(buffer_->host_ptr);
  EXPECT_EQ(pending_before, stream_->pending_value);
  EXPECT_EQ(0, std::memcmp(contents, initial.data(), initial.size()));
}

TEST_F(CpuStreamingMemoryTest,
       PostAcceptFlushFailureDrainsPitchedBatchBeforeReturning) {
  constexpr iree_device_size_t kAllocationSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kAllocationSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  ASSERT_NE(nullptr, buffer_->host_ptr);

  const std::array<uint8_t, kAllocationSize> initial = {
      1, 2, 3, 4, 21, 22, 23, 24, 41, 42, 43, 44, 61, 62, 63, 64};
  std::memcpy(buffer_->host_ptr, initial.data(), initial.size());

  InjectedFlushQueue fault_queue = {};
  iree_hal_queue_t* original_queue = stream_->queue;
  InitializeInjectedFlushQueue(original_queue, &fault_queue);
  EXPECT_EQ(original_queue, ReplaceStreamQueue(stream_, &fault_queue.base));
  // Transfer the stream's old queue reference to the wrapper target.
  iree_hal_queue_release(original_queue);

  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
  uint64_t gate_value = 1;
  fault_queue.execute_gate = gate;
  fault_queue.execute_gate_value = gate_value;
  fault_queue.fail_flush.store(true, std::memory_order_release);

  std::atomic<bool> copy_returned = false;
  std::atomic<iree_status_code_t> copy_status_code = IREE_STATUS_UNKNOWN;
  std::thread copy_thread([&] {
    iree_status_t status = iree_hal_streaming_memcpy_device_to_device_2d(
        context_, device_pointer_ + 4, /*dst_pitch=*/8, device_pointer_,
        /*src_pitch=*/8, /*width=*/4, /*height=*/2, stream_);
    copy_status_code.store(iree_status_code(status), std::memory_order_release);
    iree_status_ignore(status);
    copy_returned.store(true, std::memory_order_release);
  });

  fault_queue.flush_observed.acquire();
  EXPECT_FALSE(copy_returned.load(std::memory_order_acquire));

  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, gate_value,
                                           /*frontier=*/nullptr));
  copy_thread.join();
  EXPECT_TRUE(copy_returned.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_INTERNAL,
            copy_status_code.load(std::memory_order_acquire));
  EXPECT_EQ(1, fault_queue.accepted_transfer_count.load());
  const auto* contents = static_cast<const uint8_t*>(buffer_->host_ptr);
  EXPECT_EQ(0, std::memcmp(contents + 4, initial.data(), 4));
  EXPECT_EQ(0, std::memcmp(contents + 12, initial.data() + 8, 4));

  iree_hal_queue_retain(original_queue);
  iree_hal_queue_t* installed_queue =
      ReplaceStreamQueue(stream_, original_queue);
  iree_hal_queue_release(installed_queue);
  iree_hal_semaphore_release(gate);
}

TEST_F(CpuStreamingMemoryTest,
       SnapshotAllocationFailureDrainsEverySelectedStreamTail) {
  constexpr iree_device_size_t kAllocationSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kAllocationSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  ASSERT_NE(nullptr, buffer_->host_ptr);

  iree_hal_streaming_stream_t* second_stream = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_stream_create(
      context_, context_->queue, IREE_HAL_STREAMING_STREAM_FLAG_NONE,
      /*priority=*/0, iree_allocator_system(), &second_stream));
  iree_hal_streaming_buffer_t* staging[2] = {};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(staging); ++i) {
    IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_host_staging(
        context_, kAllocationSize, &staging[i]));
    ASSERT_NE(nullptr, staging[i]->host_ptr);
  }
  std::array<uint8_t, kAllocationSize> expected;
  expected.fill(0x2A);
  std::memcpy(buffer_->host_ptr, expected.data(), expected.size());
  std::memset(staging[0]->host_ptr, 0, expected.size());
  std::memset(staging[1]->host_ptr, 0, expected.size());

  // Leave one recorded fill behind each independently gated stream. The
  // allocation-free fallback visits streams in stable ID order, so releasing
  // only the first gate proves it cannot return without submitting and draining
  // the second selected tail.
  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
  uint64_t gate_values[2] = {1, 2};
  iree_hal_streaming_stream_t* streams[2] = {stream_, second_stream};
  uint64_t second_initial_pending = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(streams); ++i) {
    const iree_hal_semaphore_list_t gate_wait = {
        .count = 1,
        .semaphores = &gate,
        .payload_values = &gate_values[i],
    };
    IREE_ASSERT_OK(
        iree_hal_streaming_stream_wait_semaphores(streams[i], gate_wait));
    IREE_ASSERT_OK(iree_hal_streaming_memory_memset(
        context_, iree_hal_streaming_buffer_device_pointer(staging[i]),
        kAllocationSize, expected.data(), 1, streams[i]));
  }
  iree_slim_mutex_lock(&second_stream->mutex);
  second_initial_pending = second_stream->pending_value;
  iree_slim_mutex_unlock(&second_stream->mutex);

  // Fail snapshot construction, while allowing later queue submission to
  // allocate its completion records. The fallback promises to drain accepted
  // work; it does not promise that new submissions cannot allocate.
  snapshot_allocator_.fail_next_allocation.store(true,
                                                 std::memory_order_release);

  std::atomic<bool> synchronize_returned = false;
  std::atomic<iree_status_code_t> synchronize_status_code = IREE_STATUS_UNKNOWN;
  std::thread synchronize_thread([&] {
    iree_status_t status = iree_hal_streaming_context_synchronize(context_);
    synchronize_status_code.store(iree_status_code(status),
                                  std::memory_order_release);
    iree_status_ignore(status);
    synchronize_returned.store(true, std::memory_order_release);
  });

  snapshot_allocator_.failure_observed.acquire();
  EXPECT_FALSE(synchronize_returned.load(std::memory_order_acquire));

  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, gate_values[0],
                                           /*frontier=*/nullptr));
  bool observed_second_submission = false;
  while (!synchronize_returned.load(std::memory_order_acquire)) {
    iree_slim_mutex_lock(&second_stream->mutex);
    observed_second_submission =
        second_stream->pending_value > second_initial_pending;
    iree_slim_mutex_unlock(&second_stream->mutex);
    if (observed_second_submission) {
      break;
    }
    std::this_thread::yield();
  }
  EXPECT_TRUE(observed_second_submission);
  EXPECT_FALSE(synchronize_returned.load(std::memory_order_acquire));

  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, gate_values[1],
                                           /*frontier=*/nullptr));
  synchronize_thread.join();

  EXPECT_TRUE(synchronize_returned.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED,
            synchronize_status_code.load(std::memory_order_acquire));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(staging); ++i) {
    EXPECT_EQ(
        0, std::memcmp(staging[i]->host_ptr, expected.data(), expected.size()));
  }

  iree_hal_semaphore_release(gate);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(staging); ++i) {
    iree_hal_streaming_memory_release_wrapped_buffer(staging[i]);
  }
  iree_hal_streaming_context_unregister_stream(context_, second_stream);
  iree_hal_streaming_stream_release(second_stream);
}

TEST_F(CpuStreamingMemoryTest, PostAcceptFlushFailureDrainsD2HBeforeReturning) {
  constexpr iree_device_size_t kAllocationSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kAllocationSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  ASSERT_NE(nullptr, buffer_->host_ptr);
  const std::array<uint8_t, kAllocationSize> expected = {
      7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112};
  std::memcpy(buffer_->host_ptr, expected.data(), expected.size());
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_host(
      context_, kAllocationSize, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT,
      /*fine_grained=*/false, &host_buffer_));
  void* destination = host_buffer_->host_ptr;

  InjectedFlushQueue fault_queue = {};
  iree_hal_queue_t* original_queue = stream_->queue;
  InitializeInjectedFlushQueue(original_queue, &fault_queue);
  EXPECT_EQ(original_queue, ReplaceStreamQueue(stream_, &fault_queue.base));
  // Transfer the stream's old queue reference to the wrapper target.
  iree_hal_queue_release(original_queue);

  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
  uint64_t gate_value = 1;
  fault_queue.execute_gate = gate;
  fault_queue.execute_gate_value = gate_value;
  fault_queue.fail_flush.store(true, std::memory_order_release);

  std::atomic<bool> copy_returned = false;
  std::atomic<iree_status_code_t> copy_status_code = IREE_STATUS_UNKNOWN;
  std::thread copy_thread([&] {
    iree_status_t status = iree_hal_streaming_memcpy_device_to_host(
        context_, destination, device_pointer_, kAllocationSize, stream_);
    copy_status_code.store(iree_status_code(status), std::memory_order_release);
    iree_status_ignore(status);
    copy_returned.store(true, std::memory_order_release);
  });

  fault_queue.flush_observed.acquire();
  EXPECT_FALSE(copy_returned.load(std::memory_order_acquire));

  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, gate_value,
                                           /*frontier=*/nullptr));
  copy_thread.join();
  EXPECT_TRUE(copy_returned.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_INTERNAL,
            copy_status_code.load(std::memory_order_acquire));
  EXPECT_EQ(0, std::memcmp(destination, expected.data(), expected.size()));

  iree_hal_queue_retain(original_queue);
  iree_hal_queue_t* installed_queue =
      ReplaceStreamQueue(stream_, original_queue);
  iree_hal_queue_release(installed_queue);
  iree_hal_semaphore_release(gate);
}

TEST_F(CpuStreamingMemoryTest, RejectedDownloadPreservesStreamAndDestination) {
  constexpr iree_device_size_t kSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_host(
      context_, kSize, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT,
      /*fine_grained=*/false, &host_buffer_));
  std::memset(buffer_->host_ptr, 0x42, kSize);
  std::memset(host_buffer_->host_ptr, 0xCD, kSize);

  InjectedFlushQueue fault_queue = {};
  iree_hal_queue_t* original_queue = stream_->queue;
  InitializeInjectedFlushQueue(original_queue, &fault_queue);
  EXPECT_EQ(original_queue, ReplaceStreamQueue(stream_, &fault_queue.base));
  iree_hal_queue_release(original_queue);
  fault_queue.fail_transfer.store(true, std::memory_order_release);

  const uint64_t pending_before = stream_->pending_value;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_streaming_memcpy_device_to_host(
          context_, host_buffer_->host_ptr, device_pointer_, kSize, stream_));
  EXPECT_EQ(pending_before, stream_->pending_value);
  EXPECT_EQ(0, fault_queue.accepted_transfer_count.load());
  const auto* bytes = static_cast<const uint8_t*>(host_buffer_->host_ptr);
  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_EQ(bytes[i], 0xCD);
  }

  iree_hal_queue_retain(original_queue);
  iree_hal_queue_release(ReplaceStreamQueue(stream_, original_queue));
}

TEST_F(CpuStreamingMemoryTest,
       AsyncFreeFlushFailureRestoresAllocationWhileWorkIsPending) {
  constexpr iree_device_size_t kAllocationSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kAllocationSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  ASSERT_NE(nullptr, buffer_->host_ptr);
  std::memset(buffer_->host_ptr, 0, kAllocationSize);
  uint32_t pattern = 0xA5A5A5A5u;
  IREE_ASSERT_OK(iree_hal_streaming_memory_memset(context_, device_pointer_,
                                                  kAllocationSize, &pattern,
                                                  sizeof(pattern), stream_));

  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
  uint64_t gate_value = 1;
  InjectedFlushQueue fault_queue = {};
  iree_hal_queue_t* original_queue = stream_->queue;
  InitializeInjectedFlushQueue(original_queue, &fault_queue);
  EXPECT_EQ(original_queue, ReplaceStreamQueue(stream_, &fault_queue.base));
  iree_hal_queue_release(original_queue);
  fault_queue.execute_gate = gate;
  fault_queue.execute_gate_value = gate_value;
  fault_queue.fail_flush.store(true, std::memory_order_release);

  // The fill is accepted, but the flush error prevents enqueueing the free.
  // Rollback restores the caller's allocation using its reserved table slot;
  // it neither allocates nor waits for accepted work to complete.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        iree_hal_streaming_memory_free_device_async(
                            context_, device_pointer_, stream_));
  EXPECT_EQ(1, fault_queue.injected_execute_count.load());
  EXPECT_EQ(1, fault_queue.injected_flush_count.load());

  iree_hal_queue_retain(original_queue);
  iree_hal_queue_t* installed_queue =
      ReplaceStreamQueue(stream_, original_queue);
  iree_hal_queue_release(installed_queue);

  uint64_t completed_value = 0;
  IREE_EXPECT_OK(
      iree_hal_semaphore_query(stream_->timeline_semaphore, &completed_value));
  EXPECT_LT(completed_value, stream_->pending_value);

  // A normal preparation lease proves both lookup and admission were restored.
  iree_hal_streaming_retained_buffer_ref_t restored_ref = {};
  IREE_EXPECT_OK(iree_hal_streaming_memory_lookup_range_retain(
      context_, device_pointer_, kAllocationSize, &restored_ref));
  EXPECT_EQ(buffer_, restored_ref.owner_wrapper);
  iree_hal_streaming_retained_buffer_ref_deinitialize(&restored_ref);

  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, gate_value,
                                           /*frontier=*/nullptr));
  IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
  iree_hal_semaphore_release(gate);

  std::array<uint8_t, kAllocationSize> expected;
  expected.fill(0xA5);
  EXPECT_EQ(0,
            std::memcmp(buffer_->host_ptr, expected.data(), expected.size()));

  // The failed call did not consume ownership: the caller can free it normally.
  IREE_ASSERT_OK(iree_hal_streaming_memory_free_device_async(
      context_, device_pointer_, stream_));
  buffer_ = nullptr;
  device_pointer_ = 0;
  IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
}
}  // namespace
