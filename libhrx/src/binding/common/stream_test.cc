// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/stream.h"

#include <array>
#include <atomic>
#include <barrier>
#include <thread>

#include "common/context.h"
#include "common/device.h"
#include "common/event.h"
#include "common/hrx_bridge.h"
#include "common/memory.h"
#include "common/module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libhrx/src/libhrx/device.h"

namespace {

// Observes the actual launch boundary and delegates timeline completion to a
// real task queue. Executable bytes are never executed by this dependency.
struct LaunchBoundaryQueue {
  // Queue interface retained by the stream under test.
  iree_hal_queue_t base;
  // Retained context queue that executes completion dependencies.
  iree_hal_queue_t* target = nullptr;
  // Argument contract expected for the next launch.
  bool native = true;
  // Required release scope for the next accepted native dispatch.
  bool expect_system_release = true;
  // Rejects one dispatch before accepting arguments or timeline ownership.
  bool reject_dispatch = false;
  // Number of launches that reached the queue boundary.
  size_t launches = 0;
  // Accepted system-release barriers, excluding simulated dispatch completion.
  size_t releases = 0;
  // Incoming private timeline value of the last accepted release.
  uint64_t release_wait_value = 0;
  // Outgoing private timeline value of the last accepted release.
  uint64_t release_signal_value = 0;
  // Incoming private timeline value of the last host callback.
  uint64_t host_wait_value = 0;
  // Optional owned gate holding system release after dispatch completion.
  iree_hal_semaphore_t* release_gate = nullptr;
  // Rejects one release before accepting any ownership.
  bool reject_release = false;
  // Reports one flush error after the preceding operation was accepted.
  bool fail_flush = false;

  static void Destroy(iree_hal_queue_t* base) {
    auto* self = reinterpret_cast<LaunchBoundaryQueue*>(base);
    iree_hal_semaphore_release(self->release_gate);
    iree_hal_queue_release(self->target);
    delete self;
  }
  static iree_status_t Flush(iree_hal_queue_t* base) {
    auto* self = reinterpret_cast<LaunchBoundaryQueue*>(base);
    if (self->fail_flush) {
      self->fail_flush = false;
      return iree_make_status(IREE_STATUS_UNAVAILABLE);
    }
    return iree_hal_queue_flush(self->target);
  }
  static iree_status_t Barrier(iree_hal_queue_t* base,
                               iree_hal_semaphore_list_t waits,
                               iree_hal_semaphore_list_t signals,
                               const iree_hal_queue_barriers_t* barriers,
                               iree_hal_queue_barrier_flags_t flags) {
    auto* self = reinterpret_cast<LaunchBoundaryQueue*>(base);
    EXPECT_NE(barriers, nullptr);
    EXPECT_NE(barriers->before, nullptr);
    EXPECT_NE(barriers->after, nullptr);
    EXPECT_EQ(barriers->before->count, 1u);
    EXPECT_EQ(barriers->after->count, 0u);
    const auto& release = barriers->before->values[0];
    EXPECT_EQ(release.source_stage_mask, IREE_HAL_EXECUTION_STAGE_DISPATCH);
    EXPECT_EQ(release.target_stage_mask, 0u);
    EXPECT_EQ(release.flags, IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE);
    EXPECT_EQ(waits.count, 1u);
    EXPECT_EQ(signals.count, 1u);
    if (self->reject_release) {
      self->reject_release = false;
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    iree_hal_semaphore_t* wait_semaphores[] = {waits.semaphores[0],
                                               self->release_gate};
    uint64_t wait_values[] = {waits.payload_values[0], 1};
    const iree_hal_semaphore_list_t release_waits = {
        self->release_gate ? 2u : 1u, wait_semaphores, wait_values};
    iree_status_t status = iree_hal_queue_barrier(self->target, release_waits,
                                                  signals, barriers, flags);
    if (iree_status_is_ok(status)) {
      ++self->releases;
      self->release_wait_value = waits.payload_values[0];
      self->release_signal_value = signals.payload_values[0];
    }
    return status;
  }
  static iree_status_t Execute(iree_hal_queue_t* base,
                               iree_hal_semaphore_list_t waits,
                               iree_hal_semaphore_list_t signals,
                               iree_hal_command_buffer_t* command_buffer,
                               iree_hal_buffer_binding_table_t bindings,
                               iree_hal_queue_execute_flags_t flags) {
    return iree_hal_queue_execute(
        reinterpret_cast<LaunchBoundaryQueue*>(base)->target, waits, signals,
        command_buffer, bindings, flags);
  }
  static iree_status_t HostCall(iree_hal_queue_t* base,
                                iree_hal_semaphore_list_t waits,
                                iree_hal_semaphore_list_t signals,
                                iree_hal_host_call_t call,
                                const uint64_t args[4],
                                iree_hal_host_call_flags_t flags) {
    auto* self = reinterpret_cast<LaunchBoundaryQueue*>(base);
    EXPECT_EQ(waits.count, 1u);
    self->host_wait_value = waits.payload_values[0];
    return iree_hal_queue_host_call(self->target, waits, signals, call, args,
                                    flags);
  }
  static iree_status_t Dispatch(
      iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
      iree_hal_semaphore_list_t signals, iree_hal_executable_t*,
      iree_hal_executable_function_t, iree_hal_dispatch_config_t,
      iree_const_byte_span_t, iree_hal_buffer_ref_list_t bindings,
      const iree_hal_queue_barriers_t* barriers,
      iree_hal_dispatch_flags_t flags) {
    auto* self = reinterpret_cast<LaunchBoundaryQueue*>(base);
    if (self->reject_dispatch) {
      self->reject_dispatch = false;
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    ++self->launches;
    EXPECT_EQ(bool(flags & IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS),
              self->native);
    if (self->native) {
      EXPECT_EQ(bindings.count, 0u);
      EXPECT_NE(barriers, nullptr);
      if (barriers) {
        EXPECT_NE(barriers->before, nullptr);
        EXPECT_NE(barriers->after, nullptr);
        if (barriers->before && barriers->after) {
          EXPECT_EQ(barriers->before->count, 1u);
          EXPECT_EQ(barriers->after->count, 1u);
          const auto& before = barriers->before->values[0];
          const auto& after = barriers->after->values[0];
          EXPECT_EQ(before.flags, 0u);
          EXPECT_EQ(before.source_stage_mask,
                    IREE_HAL_EXECUTION_STAGE_DISPATCH |
                        IREE_HAL_EXECUTION_STAGE_TRANSFER);
          EXPECT_EQ(before.target_stage_mask,
                    IREE_HAL_EXECUTION_STAGE_DISPATCH);
          EXPECT_EQ(before.memory_barrier_count, 1u);
          if (before.memory_barrier_count) {
            EXPECT_EQ(before.memory_barriers[0].source_scope, 0u);
            EXPECT_EQ(before.memory_barriers[0].target_scope,
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                          IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE);
          }
          EXPECT_EQ(after.source_stage_mask, IREE_HAL_EXECUTION_STAGE_DISPATCH);
          if (self->expect_system_release) {
            EXPECT_EQ(after.flags, IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE);
          } else {
            EXPECT_EQ(after.flags, 0u);
            EXPECT_EQ(after.memory_barrier_count, 1u);
            if (after.memory_barrier_count) {
              EXPECT_EQ(after.memory_barriers[0].source_scope,
                        IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE);
              EXPECT_EQ(after.memory_barriers[0].target_scope, 0u);
            }
          }
          self->expect_system_release = false;
        }
      }
    } else {
      EXPECT_EQ(bindings.count, 1u);
      EXPECT_EQ(barriers, nullptr);
    }
    return iree_hal_queue_barrier(self->target, waits, signals, nullptr,
                                  IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
  }
};

class StreamTest : public ::testing::Test {
 protected:
  // Owns the task device and its runtime through fixture cleanup.
  hrx_device_group_t group_ = nullptr;

  void SetUp() override {
    hrx_runtime_t runtime = nullptr;
    IREE_ASSERT_OK(HRX_CALL(
        hrx_runtime_create(nullptr, hrx_host_allocator_system(), &runtime)));
    const hrx_device_create_params_t params = {.uri = "task://"};
    hrx_status_t status = hrx_device_group_create(runtime, 1, &params, &group_);
    hrx_runtime_release(runtime);
    IREE_ASSERT_OK(HRX_CALL(status));
    hrx_device_t device = hrx_device_group_get_device(group_, 0);
    device_entry_.hrx_device = device;
    device_entry_.hal_device = hrx_device_hal(device);
    iree_slim_mutex_initialize(&device_entry_.primary_context_mutex);
    iree_slim_mutex_initialize(&device_entry_.graph_memory_mutex);
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &device_entry_.block_pool);
    IREE_ASSERT_OK(iree_hal_streaming_context_create(
        &device_entry_,
        {.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO},
        iree_allocator_system(), &context_));
  }

  void TearDown() override {
    iree_hal_streaming_context_release(context_);
    iree_arena_block_pool_deinitialize(&device_entry_.block_pool);
    iree_slim_mutex_deinitialize(&device_entry_.graph_memory_mutex);
    iree_slim_mutex_deinitialize(&device_entry_.primary_context_mutex);
    hrx_device_group_release(group_);
    group_ = nullptr;
  }

  // CPU registry entry borrowed by the context until synchronous teardown.
  iree_hal_streaming_device_t device_entry_ = {};
  // Real context shared by independent application threads and streams.
  iree_hal_streaming_context_t* context_ = nullptr;
};

TEST_F(StreamTest, ConcurrentFirstUseAndCompletion) {
  constexpr size_t kThreadCount = 8;
  std::barrier<> start(static_cast<std::ptrdiff_t>(kThreadCount));
  std::array<iree_hal_streaming_stream_t*, kThreadCount> streams = {};
  std::array<std::thread, kThreadCount> threads;
  for (size_t i = 0; i < kThreadCount; ++i) {
    threads[i] = std::thread([&, i] {
      start.arrive_and_wait();
      IREE_ASSERT_OK(iree_hal_streaming_stream_create(
          context_, context_->queue,
          IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING, 0,
          iree_allocator_system(), &streams[i]));

      // A rejected launch still validates admission on the stream.
      // No executable is needed to exercise concurrent launch admission.
      iree_hal_streaming_symbol_t symbol = {
          .type = IREE_HAL_STREAMING_SYMBOL_TYPE_GLOBAL,
      };
      const iree_hal_streaming_dispatch_params_t params = {};
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_INVALID_ARGUMENT,
          iree_hal_streaming_launch_kernel(&symbol, &params, streams[i]));

      // Submit a real command buffer and join its queue completion. Independent
      // streams must independently publish and complete their timeline values.
      IREE_ASSERT_OK(iree_hal_streaming_stream_begin(streams[i]));
      IREE_ASSERT_OK(iree_hal_streaming_stream_flush(streams[i]));
      IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize_flushed(streams[i]));
      EXPECT_EQ(1u, streams[i]->pending_value);
      EXPECT_EQ(streams[i]->pending_value, streams[i]->completed_value);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (auto* stream : streams) {
    iree_hal_streaming_stream_release(stream);
  }
}

class NativeStreamTest : public StreamTest {
 protected:
  void SetUp() override {
    StreamTest::SetUp();
    static const iree_hal_queue_vtable_t queue_vtable = {
        .destroy = LaunchBoundaryQueue::Destroy,
        .barrier = LaunchBoundaryQueue::Barrier,
        .execute = LaunchBoundaryQueue::Execute,
        .host_call = LaunchBoundaryQueue::HostCall,
        .dispatch = LaunchBoundaryQueue::Dispatch,
        .flush = LaunchBoundaryQueue::Flush,
    };
    queue_ = new LaunchBoundaryQueue;
    queue_->target = context_->queue;
    iree_hal_queue_retain(queue_->target);
    iree_hal_queue_params_t queue_params;
    iree_hal_queue_params_initialize(&queue_params);
    iree_hal_queue_initialize(iree_hal_queue_family(queue_->target),
                              &queue_params, &queue_vtable, &queue_->base);
    static const iree_hal_executable_vtable_t executable_vtable = {
        .destroy = [](iree_hal_executable_t*) {},
    };
    iree_hal_executable_initialize(iree_hal_queue_family(queue_->target),
                                   &executable_vtable, &executable_);
    IREE_ASSERT_OK(iree_hal_streaming_stream_create(
        context_, &queue_->base, IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING, 0,
        iree_allocator_system(), &stream_));
  }

  void TearDown() override {
    OpenRelease();
    queue_->reject_release = false;
    queue_->fail_flush = false;
    IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize_flushed(stream_));
    iree_hal_streaming_stream_release(stream_);
    iree_hal_executable_release(&executable_);
    iree_hal_queue_release(&queue_->base);
    StreamTest::TearDown();
  }

  iree_status_t Launch(bool release_system = false) {
    queue_->expect_system_release = release_system;
    iree_hal_streaming_symbol_t symbol = {
        .type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION,
        .executable = &executable_,
    };
    const iree_hal_streaming_dispatch_params_t params = {
        .grid_dim = {1, 1, 1},
        .block_dim = {1, 1, 1},
    };
    return iree_hal_streaming_launch_kernel(&symbol, &params, stream_);
  }

  iree_status_t LaunchBurst() {
    IREE_RETURN_IF_ERROR(Launch(true));
    return Launch();
  }

  void HoldRelease() {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &queue_->release_gate));
  }

  void OpenRelease() {
    if (!queue_->release_gate) {
      return;
    }
    IREE_EXPECT_OK(iree_hal_semaphore_signal(queue_->release_gate, 1, nullptr));
    iree_hal_semaphore_release(queue_->release_gate);
    queue_->release_gate = nullptr;
  }

  void WaitFor(uint64_t value) {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(stream_->timeline_semaphore, value,
                                           iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
  }

  // Observing queue owned by the fixture and retained by the real stream.
  LaunchBoundaryQueue* queue_ = nullptr;
  // Real frontend stream whose submission and observation paths are tested.
  iree_hal_streaming_stream_t* stream_ = nullptr;
  // Native function handle consumed only by the observing queue dependency.
  iree_hal_executable_t executable_;
};

TEST_F(NativeStreamTest, NativeLaunchesAcquireAndReleaseDeviceWrites) {
  auto* queue = queue_;
  auto* stream = stream_;
  auto& executable = executable_;
  iree_hal_streaming_symbol_t symbol = {
      .type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION,
      .executable = &executable,
  };
  iree_hal_streaming_dispatch_params_t params = {
      .grid_dim = {1, 1, 1},
      .block_dim = {1, 1, 1},
  };
  IREE_EXPECT_OK(iree_hal_streaming_launch_kernel(&symbol, &params, stream));

  uint64_t argument = 123;
  params.buffer = &argument;
  params.buffer_size = sizeof(argument);
  params.flags = IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED;
  IREE_EXPECT_OK(iree_hal_streaming_launch_kernel(&symbol, &params, stream));

  iree_hal_streaming_parameter_op_t operation = {
      .copy = {.size = sizeof(argument)},
  };
  symbol.parameters = {
      .buffer_size = sizeof(argument),
      .constant_bytes = sizeof(argument),
      .direct_arg_bytes = sizeof(argument),
      .copy_count = 1,
      .ops = &operation,
  };
  void* arguments[] = {&argument};
  params.buffer = arguments;
  params.flags = IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY;
  IREE_EXPECT_OK(iree_hal_streaming_launch_kernel(&symbol, &params, stream));

  // A reflected HAL buffer still needs the default range-maintenance contract.
  iree_hal_streaming_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      context_, 16, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer));
  iree_hal_streaming_deviceptr_t pointer = buffer->device_ptr;
  operation = {.resolve = {}};
  symbol.parameters = {
      .buffer_size = sizeof(pointer),
      .direct_arg_bytes = sizeof(pointer),
      .binding_count = 1,
      .ops = &operation,
  };
  queue->native = false;
  params.buffer = &pointer;
  params.buffer_size = sizeof(pointer);
  params.flags = IREE_HAL_STREAMING_DISPATCH_FLAG_NONE;
  IREE_EXPECT_OK(iree_hal_streaming_launch_kernel(&symbol, &params, stream));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize_flushed(stream));
  EXPECT_EQ(queue->launches, 4u);
  iree_hal_streaming_memory_release_wrapped_buffer(buffer);
}

TEST_F(NativeStreamTest, OneSystemReleasePublishesEachNativeBurst) {
  for (int i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(Launch(i == 0));
  }
  EXPECT_EQ(queue_->releases, 0u);
  EXPECT_EQ(stream_->native_release_state,
            IREE_HAL_STREAMING_NATIVE_RELEASE_DEVICE);
  IREE_ASSERT_OK(iree_hal_streaming_stream_flush(stream_));
  EXPECT_EQ(queue_->releases, 1u);
  EXPECT_EQ(queue_->release_wait_value, 3u);
  EXPECT_EQ(queue_->release_signal_value, 4u);
  EXPECT_EQ(stream_->native_release_state,
            IREE_HAL_STREAMING_NATIVE_RELEASE_EMPTY);
  IREE_ASSERT_OK(iree_hal_streaming_stream_flush(stream_));
  EXPECT_EQ(queue_->releases, 1u);

  IREE_ASSERT_OK(LaunchBurst());
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize_flushed(stream_));
  EXPECT_EQ(queue_->releases, 2u);
  EXPECT_EQ(queue_->release_wait_value, 6u);
  EXPECT_EQ(stream_->completed_value, 7u);
}

TEST_F(NativeStreamTest, SingletonObservationUsesItsDispatchRelease) {
  for (uint64_t value = 1; value <= 3; ++value) {
    IREE_ASSERT_OK(Launch(true));
    EXPECT_EQ(stream_->native_release_state,
              IREE_HAL_STREAMING_NATIVE_RELEASE_SYSTEM);
    IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize_flushed(stream_));
    EXPECT_EQ(stream_->completed_value, value);
    EXPECT_EQ(queue_->releases, 0u);
  }
}

TEST_F(NativeStreamTest, RejectedDispatchPreservesReleaseState) {
  queue_->reject_dispatch = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Launch(true));
  EXPECT_EQ(stream_->native_release_state,
            IREE_HAL_STREAMING_NATIVE_RELEASE_EMPTY);
  IREE_ASSERT_OK(Launch(true));
  queue_->reject_dispatch = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Launch());
  EXPECT_EQ(stream_->native_release_state,
            IREE_HAL_STREAMING_NATIVE_RELEASE_SYSTEM);
  EXPECT_EQ(stream_->pending_value, 1u);
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize_flushed(stream_));
  EXPECT_EQ(queue_->releases, 0u);
}

TEST_F(NativeStreamTest, QueryWaitsForSystemReleaseAfterDispatchCompletion) {
  HoldRelease();
  IREE_ASSERT_OK(LaunchBurst());
  WaitFor(2);
  auto completion = IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE;
  IREE_ASSERT_OK(iree_hal_streaming_stream_query(stream_, &completion));
  EXPECT_EQ(completion, IREE_HAL_STREAMING_COMPLETION_STATE_PENDING);
  EXPECT_EQ(stream_->pending_value, 3u);
  EXPECT_EQ(queue_->releases, 1u);
  OpenRelease();
  WaitFor(3);
  IREE_ASSERT_OK(iree_hal_streaming_stream_query(stream_, &completion));
  EXPECT_EQ(completion, IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE);
  EXPECT_EQ(queue_->releases, 1u);
}

TEST_F(NativeStreamTest, RejectedReleasePreservesUnpublishedTail) {
  IREE_ASSERT_OK(LaunchBurst());
  queue_->reject_release = true;
  auto completion = IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_streaming_stream_query(stream_, &completion));
  EXPECT_EQ(completion, IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE);
  EXPECT_EQ(stream_->pending_value, 2u);
  EXPECT_EQ(stream_->native_release_state,
            IREE_HAL_STREAMING_NATIVE_RELEASE_DEVICE);
  EXPECT_EQ(queue_->releases, 0u);
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize_flushed(stream_));
  EXPECT_EQ(stream_->completed_value, 3u);
  EXPECT_EQ(queue_->releases, 1u);
}

TEST_F(NativeStreamTest, FlushFailurePreservesAcceptedReleaseOwnership) {
  IREE_ASSERT_OK(LaunchBurst());
  queue_->fail_flush = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_streaming_stream_flush(stream_));
  EXPECT_EQ(stream_->pending_value, 3u);
  EXPECT_EQ(stream_->native_release_state,
            IREE_HAL_STREAMING_NATIVE_RELEASE_EMPTY);
  EXPECT_EQ(queue_->releases, 1u);
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize_flushed(stream_));
  EXPECT_EQ(stream_->completed_value, 3u);
  EXPECT_EQ(queue_->releases, 1u);
}

TEST_F(NativeStreamTest, SingletonFlushFailurePreservesNativeTail) {
  IREE_ASSERT_OK(Launch(true));
  queue_->fail_flush = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_streaming_stream_flush(stream_));
  EXPECT_EQ(stream_->pending_value, 1u);
  EXPECT_EQ(stream_->native_release_state,
            IREE_HAL_STREAMING_NATIVE_RELEASE_SYSTEM);
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize_flushed(stream_));
  EXPECT_EQ(queue_->releases, 0u);
  EXPECT_EQ(stream_->completed_value, 1u);
}

TEST_F(NativeStreamTest, RecordingPublishesPrecedingNativeWrites) {
  IREE_ASSERT_OK(LaunchBurst());
  IREE_ASSERT_OK(iree_hal_streaming_stream_begin(stream_));
  IREE_ASSERT_OK(iree_hal_streaming_stream_flush(stream_));
  EXPECT_EQ(stream_->native_release_state,
            IREE_HAL_STREAMING_NATIVE_RELEASE_EMPTY);
  EXPECT_EQ(stream_->pending_value, 3u);
  // The existing recording carries its terminal release, with no additional
  // queue barrier or timeline point.
  EXPECT_EQ(queue_->releases, 0u);
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize_flushed(stream_));
  EXPECT_EQ(stream_->completed_value, 3u);
}

TEST_F(NativeStreamTest, UntimedEventNamesTheSystemRelease) {
  HoldRelease();
  IREE_ASSERT_OK(LaunchBurst());
  WaitFor(2);
  iree_hal_streaming_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  IREE_EXPECT_OK(iree_hal_streaming_event_record(event, stream_, false));
  auto completion = IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE;
  IREE_EXPECT_OK(iree_hal_streaming_event_query(event, &completion));
  EXPECT_EQ(completion, IREE_HAL_STREAMING_COMPLETION_STATE_PENDING);
  iree_hal_streaming_recorded_point_t point;
  iree_hal_streaming_event_acquire_recorded_point(event, &point);
  EXPECT_EQ(point.semaphore, stream_->timeline_semaphore);
  EXPECT_EQ(point.value, 3u);
  iree_hal_streaming_event_release_recorded_point(&point);
  OpenRelease();
  IREE_EXPECT_OK(iree_hal_streaming_event_synchronize(event));
  iree_hal_streaming_event_release(event);
}

TEST_F(NativeStreamTest, HostCallbackWaitsForPublishedWrites) {
  HoldRelease();
  IREE_ASSERT_OK(LaunchBurst());
  WaitFor(2);
  std::atomic<bool> called = false;
  iree_hal_host_call_t call = iree_hal_make_host_call(
      [](void* user_data, const uint64_t[4], iree_hal_host_call_context_t*) {
        static_cast<std::atomic<bool>*>(user_data)->store(true);
        return iree_ok_status();
      },
      &called);
  const uint64_t args[4] = {};
  IREE_ASSERT_OK(iree_hal_streaming_queue_host_call(
      stream_, call, args, IREE_HAL_HOST_CALL_FLAG_NONE));
  EXPECT_EQ(queue_->host_wait_value, 3u);
  EXPECT_EQ(stream_->pending_value, 4u);
  EXPECT_FALSE(called.load());
  OpenRelease();
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize_flushed(stream_));
  EXPECT_TRUE(called.load());
}

}  // namespace
