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

#include "common/context.h"
#include "common/device.h"
#include "common/host_callback.h"
#include "common/hrx_bridge.h"
#include "common/stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libhrx/src/libhrx/device.h"

#if defined(IREE_PLATFORM_LINUX)
static std::atomic<bool> g_fail_next_memory_barrier = false;

extern "C" iree_status_t __real_iree_hal_command_buffer_barrier(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_barrier_t* barrier);

extern "C" iree_status_t __wrap_iree_hal_command_buffer_barrier(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_barrier_t* barrier) {
  if (g_fail_next_memory_barrier.exchange(false, std::memory_order_acq_rel)) {
    return iree_make_status(IREE_STATUS_ABORTED,
                            "injected post-copy barrier failure");
  }
  return __real_iree_hal_command_buffer_barrier(command_buffer, barrier);
}

#endif  // IREE_PLATFORM_LINUX

namespace {

struct InjectedFlushQueue {
  iree_hal_queue_t base;
  iree_hal_queue_t* target = nullptr;
  iree_hal_semaphore_t* execute_gate = nullptr;
  uint64_t execute_gate_value = 0;
  std::atomic<bool> fail_flush = false;
  std::atomic<int> injected_execute_count = 0;
  std::atomic<int> injected_flush_count = 0;
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
  if (!queue->execute_gate) {
    return iree_hal_queue_execute(queue->target, wait_semaphore_list,
                                  signal_semaphore_list, command_buffer,
                                  binding_table, flags);
  }
  if (wait_semaphore_list.count > 1) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "injected queue supports at most one input wait");
  }
  std::array<iree_hal_semaphore_t*, 2> wait_semaphores = {};
  std::array<uint64_t, 2> wait_values = {};
  for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
    wait_semaphores[i] = wait_semaphore_list.semaphores[i];
    wait_values[i] = wait_semaphore_list.payload_values[i];
  }
  wait_semaphores[wait_semaphore_list.count] = queue->execute_gate;
  wait_values[wait_semaphore_list.count] = queue->execute_gate_value;
  const iree_hal_semaphore_list_t gated_waits = {
      .count = wait_semaphore_list.count + 1,
      .semaphores = wait_semaphores.data(),
      .payload_values = wait_values.data(),
  };
  iree_status_t status =
      iree_hal_queue_execute(queue->target, gated_waits, signal_semaphore_list,
                             command_buffer, binding_table, flags);
  if (iree_status_is_ok(status)) {
    queue->injected_execute_count.fetch_add(1, std::memory_order_acq_rel);
  }
  return status;
}

iree_status_t InjectedFlushQueueFlush(iree_hal_queue_t* base_queue) {
  auto* queue = CastInjectedFlushQueue(base_queue);
  iree_status_t status = iree_hal_queue_flush(queue->target);
  if (iree_status_is_ok(status) &&
      queue->fail_flush.load(std::memory_order_acquire)) {
    queue->injected_flush_count.fetch_add(1, std::memory_order_acq_rel);
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
    .transfer = nullptr,
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
  iree_allocator_t delegate = iree_allocator_system();
  std::atomic<bool> fail_allocations = false;
  std::atomic<int> allocation_attempt_count = 0;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<FailingSnapshotAllocator*>(self);
    if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
        command == IREE_ALLOCATOR_COMMAND_CALLOC ||
        command == IREE_ALLOCATOR_COMMAND_REALLOC) {
      allocator->allocation_attempt_count.fetch_add(1,
                                                    std::memory_order_acq_rel);
      if (allocator->fail_allocations.load(std::memory_order_acquire)) {
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
  void SetUp() override {
    iree_hal_streaming_host_callback_queue_initialize(iree_allocator_system(),
                                                      &host_callbacks_);
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_initialize(/*flags=*/0)));
    hrx_device_t hrx_device = nullptr;
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_device_get(/*index=*/0, &hrx_device)));

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
        &device_entry_, context_flags, iree_allocator_system(), &context_));
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
    IREE_EXPECT_OK(HRX_CALL(hrx_cpu_shutdown()));
    iree_hal_streaming_host_callback_queue_deinitialize(&host_callbacks_);
  }

  // Explicit frontend callout owner shared by this fixture's device contexts.
  iree_hal_streaming_host_callback_queue_t host_callbacks_ = {};
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

TEST_F(CpuStreamingMemoryTest,
       AllocationIdentityDoesNotFollowUnifiedMemoryPlacement) {
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, 16, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_host(
      context_, 16, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT,
      &host_buffer_));
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
      &host_buffer_));
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

TEST_F(CpuStreamingMemoryTest,
       DownloadsValidateAllocationExtentsBeforeSubmitting) {
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, 16, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_host(
      context_, 8, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT,
      &host_buffer_));
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
      &host_buffer_));
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
       PitchedCopyDrainsAcceptedRowsBeforeReturningRecordingError) {
  constexpr iree_device_size_t kAllocationSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kAllocationSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  ASSERT_NE(nullptr, buffer_->host_ptr);

  const std::array<uint8_t, kAllocationSize> initial = {
      1, 2, 3, 4, 21, 22, 23, 24, 41, 42, 43, 44, 61, 62, 63, 64};
  std::memcpy(buffer_->host_ptr, initial.data(), initial.size());

  // Row zero copies [0, 4) to the disjoint range [4, 8). Row one then tries
  // to copy [8, 12) onto itself, forcing command-buffer validation to fail
  // only after the first row has been accepted.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_streaming_memcpy_device_to_device_2d(
          context_, device_pointer_ + 4, /*dst_pitch=*/4, device_pointer_,
          /*src_pitch=*/8, /*width=*/4, /*height=*/2, stream_));

  const auto* contents = static_cast<const uint8_t*>(buffer_->host_ptr);
  EXPECT_EQ(0, std::memcmp(contents + 4, initial.data(), 4));
  EXPECT_EQ(0, std::memcmp(contents + 8, initial.data() + 8, 4));
}

TEST_F(CpuStreamingMemoryTest,
       PostAcceptFlushFailureDrainsPrefixBeforeReturningOriginalError) {
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
        context_, device_pointer_ + 4, /*dst_pitch=*/4, device_pointer_,
        /*src_pitch=*/8, /*width=*/4, /*height=*/2, stream_);
    copy_status_code.store(iree_status_code(status), std::memory_order_release);
    iree_status_ignore(status);
    copy_returned.store(true, std::memory_order_release);
  });

  bool observed_flush_failure = false;
  for (int i = 0; i < 1000000; ++i) {
    if (fault_queue.injected_flush_count.load(std::memory_order_acquire) > 0) {
      observed_flush_failure = true;
      break;
    }
    std::this_thread::yield();
  }
  if (!observed_flush_failure) {
    IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, gate_value,
                                             /*frontier=*/nullptr));
    copy_thread.join();
    iree_hal_queue_retain(original_queue);
    iree_hal_queue_t* installed_queue =
        ReplaceStreamQueue(stream_, original_queue);
    iree_hal_queue_release(installed_queue);
    iree_hal_semaphore_release(gate);
    FAIL() << "accepted command buffer did not reach injected queue_flush";
    return;
  }
  EXPECT_FALSE(copy_returned.load(std::memory_order_acquire));

  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, gate_value,
                                           /*frontier=*/nullptr));
  copy_thread.join();
  EXPECT_TRUE(copy_returned.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_INVALID_ARGUMENT,
            copy_status_code.load(std::memory_order_acquire));
  const auto* contents = static_cast<const uint8_t*>(buffer_->host_ptr);
  EXPECT_EQ(0, std::memcmp(contents + 4, initial.data(), 4));
  EXPECT_EQ(0, std::memcmp(contents + 8, initial.data() + 8, 4));

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
  const std::array<uint8_t, kAllocationSize> expected = {
      3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48};
  std::memcpy(buffer_->host_ptr, expected.data(), expected.size());
  std::memset(staging[0]->host_ptr, 0, expected.size());
  std::memset(staging[1]->host_ptr, 0, expected.size());

  // Leave one recorded D2H-shaped copy behind each independently gated
  // registered stream. The allocation-free fallback visits streams in stable
  // ID order, so releasing only the first gate proves it cannot return without
  // submitting and draining the second selected tail.
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
    IREE_ASSERT_OK(iree_hal_streaming_memory_memcpy(
        context_, iree_hal_streaming_buffer_device_pointer(staging[i]),
        device_pointer_, kAllocationSize, streams[i]));
  }
  iree_slim_mutex_lock(&second_stream->mutex);
  second_initial_pending = second_stream->pending_value;
  iree_slim_mutex_unlock(&second_stream->mutex);

  FailingSnapshotAllocator allocator;
  const iree_allocator_t original_allocator = context_->host_allocator;
  context_->host_allocator = allocator.AsAllocator();
  allocator.fail_allocations.store(true, std::memory_order_release);

  std::atomic<bool> synchronize_returned = false;
  std::atomic<iree_status_code_t> synchronize_status_code = IREE_STATUS_UNKNOWN;
  std::thread synchronize_thread([&] {
    iree_status_t status = iree_hal_streaming_context_synchronize(context_);
    synchronize_status_code.store(iree_status_code(status),
                                  std::memory_order_release);
    iree_status_ignore(status);
    synchronize_returned.store(true, std::memory_order_release);
  });

  bool observed_snapshot_failure = false;
  for (int i = 0; i < 1000000; ++i) {
    if (allocator.allocation_attempt_count.load(std::memory_order_acquire) >
        0) {
      observed_snapshot_failure = true;
      break;
    }
    std::this_thread::yield();
  }
  EXPECT_TRUE(observed_snapshot_failure);
  EXPECT_FALSE(synchronize_returned.load(std::memory_order_acquire));

  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, gate_values[0],
                                           /*frontier=*/nullptr));
  bool observed_second_submission = false;
  for (int i = 0; i < 1000000; ++i) {
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
  allocator.fail_allocations.store(false, std::memory_order_release);
  context_->host_allocator = original_allocator;

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
      &host_buffer_));
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

  bool observed_flush_failure = false;
  for (int i = 0; i < 1000000; ++i) {
    if (fault_queue.injected_flush_count.load(std::memory_order_acquire) > 0) {
      observed_flush_failure = true;
      break;
    }
    std::this_thread::yield();
  }
  EXPECT_TRUE(observed_flush_failure);
  for (int i = 0; i < 100000 && !copy_returned.load(std::memory_order_acquire);
       ++i) {
    std::this_thread::yield();
  }
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

#if defined(IREE_PLATFORM_LINUX)
TEST_F(CpuStreamingMemoryTest,
       ScalarD2HBarrierFailureDrainsRecordedCopyBeforeReturning) {
  constexpr iree_device_size_t kAllocationSize = 16;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, kAllocationSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &buffer_));
  device_pointer_ = iree_hal_streaming_buffer_device_pointer(buffer_);
  ASSERT_NE(nullptr, buffer_->host_ptr);
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_host(
      context_, kAllocationSize, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT,
      &host_buffer_));
  void* destination = host_buffer_->host_ptr;

  InjectedFlushQueue fault_queue = {};
  iree_hal_queue_t* original_queue = stream_->queue;
  InitializeInjectedFlushQueue(original_queue, &fault_queue);
  EXPECT_EQ(original_queue, ReplaceStreamQueue(stream_, &fault_queue.base));
  iree_hal_queue_release(original_queue);

  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
  uint64_t gate_value = 1;
  fault_queue.execute_gate = gate;
  fault_queue.execute_gate_value = gate_value;
  g_fail_next_memory_barrier.store(true, std::memory_order_release);

  std::atomic<bool> copy_returned = false;
  std::atomic<iree_status_code_t> copy_status_code = IREE_STATUS_UNKNOWN;
  std::thread copy_thread([&] {
    iree_status_t status = iree_hal_streaming_memcpy_device_to_host(
        context_, destination, device_pointer_, kAllocationSize, stream_);
    copy_status_code.store(iree_status_code(status), std::memory_order_release);
    iree_status_ignore(status);
    copy_returned.store(true, std::memory_order_release);
  });

  bool observed_execute = false;
  for (int i = 0; i < 1000000; ++i) {
    if (fault_queue.injected_execute_count.load(std::memory_order_acquire) >
        0) {
      observed_execute = true;
      break;
    }
    std::this_thread::yield();
  }
  EXPECT_TRUE(observed_execute);
  EXPECT_FALSE(copy_returned.load(std::memory_order_acquire));

  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, gate_value,
                                           /*frontier=*/nullptr));
  copy_thread.join();
  g_fail_next_memory_barrier.store(false, std::memory_order_release);
  EXPECT_TRUE(copy_returned.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_ABORTED,
            copy_status_code.load(std::memory_order_acquire));

  iree_hal_queue_retain(original_queue);
  iree_hal_queue_t* installed_queue =
      ReplaceStreamQueue(stream_, original_queue);
  iree_hal_queue_release(installed_queue);
  iree_hal_semaphore_release(gate);
}
#endif  // IREE_PLATFORM_LINUX

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
