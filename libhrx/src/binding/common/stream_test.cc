// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/stream.h"

#include <array>
#include <barrier>
#include <thread>

#include "common/context.h"
#include "common/device.h"
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
  // Number of launches that reached the queue boundary.
  size_t launches = 0;

  static void Destroy(iree_hal_queue_t* base) {
    auto* self = reinterpret_cast<LaunchBoundaryQueue*>(base);
    iree_hal_queue_release(self->target);
    delete self;
  }
  static iree_status_t Flush(iree_hal_queue_t* base) {
    return iree_hal_queue_flush(
        reinterpret_cast<LaunchBoundaryQueue*>(base)->target);
  }
  static iree_status_t Dispatch(
      iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
      iree_hal_semaphore_list_t signals, iree_hal_executable_t*,
      iree_hal_executable_function_t, iree_hal_dispatch_config_t,
      iree_const_byte_span_t, iree_hal_buffer_ref_list_t bindings,
      const iree_hal_queue_barriers_t* barriers,
      iree_hal_dispatch_flags_t flags) {
    auto* self = reinterpret_cast<LaunchBoundaryQueue*>(base);
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
          EXPECT_EQ(before.flags, IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE);
          EXPECT_EQ(before.target_stage_mask,
                    IREE_HAL_EXECUTION_STAGE_DISPATCH);
          EXPECT_EQ(after.flags, IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE);
          EXPECT_EQ(after.source_stage_mask, IREE_HAL_EXECUTION_STAGE_DISPATCH);
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

TEST_F(StreamTest, NativeLaunchesRequestDirectionalSystemBoundaries) {
  static const iree_hal_queue_vtable_t queue_vtable = {
      .destroy = LaunchBoundaryQueue::Destroy,
      .dispatch = LaunchBoundaryQueue::Dispatch,
      .flush = LaunchBoundaryQueue::Flush,
  };
  auto* queue = new LaunchBoundaryQueue;
  queue->target = context_->queue;
  iree_hal_queue_retain(queue->target);
  iree_hal_queue_params_t queue_params;
  iree_hal_queue_params_initialize(&queue_params);
  iree_hal_queue_initialize(iree_hal_queue_family(queue->target), &queue_params,
                            &queue_vtable, &queue->base);
  static const iree_hal_executable_vtable_t executable_vtable = {
      .destroy = [](iree_hal_executable_t*) {},
  };
  iree_hal_executable_t executable;
  iree_hal_executable_initialize(iree_hal_queue_family(queue->target),
                                 &executable_vtable, &executable);
  iree_hal_streaming_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_stream_create(
      context_, &queue->base, IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING, 0,
      iree_allocator_system(), &stream));
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
  iree_hal_streaming_stream_release(stream);
  iree_hal_executable_release(&executable);
  iree_hal_queue_release(&queue->base);
}

}  // namespace
