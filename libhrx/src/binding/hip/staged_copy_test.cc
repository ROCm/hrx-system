// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/staged_copy.h"

#include <array>
#include <atomic>
#include <cstring>
#include <semaphore>

#include "common/context.h"
#include "common/device.h"
#include "common/graph.h"
#include "common/host_callback.h"
#include "common/hrx_bridge.h"
#include "common/memory.h"
#include "common/stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libhrx/src/libhrx/device.h"

namespace {

// A transfer dependency that accepts the first row and refuses the second.
// All accepted work uses the real task queue and its ordinary completion.
struct RejectSecondTransferQueue {
  // Queue interface borrowed by the frontend context while the test runs.
  iree_hal_queue_t base;
  // Retained native queue that performs the accepted transfer.
  iree_hal_queue_t* target = nullptr;
  // Number of transfer calls, written only by the compatibility worker.
  int call_count = 0;

  void Initialize(iree_hal_queue_t* native_queue) {
    target = native_queue;
    iree_hal_queue_retain(target);
    static const iree_hal_queue_vtable_t vtable = {
        .destroy =
            +[](iree_hal_queue_t* base) {
              auto* queue = reinterpret_cast<RejectSecondTransferQueue*>(base);
              iree_hal_queue_release(queue->target);
            },
        .transfer =
            +[](iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
                iree_hal_semaphore_list_t signals, iree_host_size_t count,
                const iree_hal_transfer_operation_t* operations,
                const iree_hal_queue_barriers_t* barriers) {
              auto* queue = reinterpret_cast<RejectSecondTransferQueue*>(base);
              if (++queue->call_count == 2) {
                return iree_make_status(
                    IREE_STATUS_ABORTED,
                    "transfer dependency rejected the second row");
              }
              return iree_hal_queue_transfer(queue->target, waits, signals,
                                             count, operations, barriers);
            },
    };
    iree_hal_queue_params_t params;
    iree_hal_queue_params_initialize(&params);
    params.features = iree_hal_queue_features(target);
    iree_hal_queue_initialize(iree_hal_queue_family(target), &params, &vtable,
                              &base);
  }
};

class StagedCopyTest : public ::testing::Test {
 protected:
  // Owns the task device and its runtime through fixture cleanup.
  inline static hrx_device_group_t group_ = nullptr;

  static void SetUpTestSuite() {
    hrx_runtime_t runtime = nullptr;
    IREE_ASSERT_OK(HRX_CALL(
        hrx_runtime_create(nullptr, hrx_host_allocator_system(), &runtime)));
    const hrx_device_create_params_t params = {.uri = "task://"};
    hrx_status_t status = hrx_device_group_create(runtime, 1, &params, &group_);
    hrx_runtime_release(runtime);
    IREE_ASSERT_OK(HRX_CALL(status));
  }
  static void TearDownTestSuite() {
    hrx_device_group_release(group_);
    group_ = nullptr;
  }

  void SetUp() override {
    device_entry_.hrx_device = hrx_device_group_get_device(group_, 0);
    device_entry_.hal_device = hrx_device_hal(device_entry_.hrx_device);
    device_entry_.host_callbacks = &callbacks_;
    iree_hal_streaming_host_callback_queue_initialize(iree_allocator_system(),
                                                      &callbacks_);
    iree_slim_mutex_initialize(&device_entry_.primary_context_mutex);
    iree_slim_mutex_initialize(&device_entry_.graph_memory_mutex);
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &device_entry_.block_pool);
    const iree_hal_streaming_context_flags_t flags = {
        .scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO,
    };
    for (auto& context : contexts_) {
      IREE_ASSERT_OK(iree_hal_streaming_context_create(
          &device_entry_, flags, iree_allocator_system(), &context));
    }
    IREE_ASSERT_OK(iree_hal_streaming_stream_create(
        contexts_[0], contexts_[0]->queue,
        IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING, 0, iree_allocator_system(),
        &stream_));
    for (size_t i = 0; i < buffers_.size(); ++i) {
      IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
          contexts_[i], 64, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffers_[i]));
      ASSERT_NE(buffers_[i]->host_ptr, nullptr);
      memset(buffers_[i]->host_ptr, 0, 64);
    }
  }

  void TearDown() override {
    if (!released_) {
      release_.release();
    }
    iree_hal_streaming_host_callback_queue_stop(&callbacks_);
    if (original_context_queue_) {
      contexts_[0]->queue = original_context_queue_;
      iree_hal_queue_release(&fault_queue_.base);
    }
    if (stream_) {
      IREE_EXPECT_STATUS_IS(expected_status_,
                            iree_hal_streaming_stream_synchronize(stream_));
    }
    iree_hal_streaming_graph_exec_release(executable_);
    iree_hal_streaming_graph_release(graph_);
    for (auto* buffer : buffers_) {
      iree_hal_streaming_memory_release_wrapped_buffer(buffer);
    }
    iree_hal_streaming_stream_release(stream_);
    for (auto* context : contexts_) {
      iree_hal_streaming_context_release(context);
    }
    iree_arena_block_pool_deinitialize(&device_entry_.block_pool);
    iree_slim_mutex_deinitialize(&device_entry_.graph_memory_mutex);
    iree_slim_mutex_deinitialize(&device_entry_.primary_context_mutex);
    iree_hal_streaming_host_callback_queue_deinitialize(&callbacks_);
  }

  iree_hip_staged_copy_endpoint_t DeviceEndpoint(size_t index) {
    return {
        .context = contexts_[index],
        .buffer = buffers_[index]->buffer,
        .offset = 1,
        .host_pointer = nullptr,
        .row_pitch = 8,
        .slice_pitch = 24,
    };
  }

  void LaunchCopy(iree_hip_staged_copy_endpoint_t destination,
                  iree_hip_staged_copy_endpoint_t source) {
    destination_ = destination;
    source_ = source;
    IREE_ASSERT_OK(iree_hal_streaming_graph_create(
        contexts_[0], IREE_HAL_STREAMING_GRAPH_FLAG_NONE,
        iree_allocator_system(), &graph_));
    iree_hal_streaming_graph_node_t* copy_node = nullptr;
    IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_operation_node(
        graph_, nullptr, 0,
        +[](void* user_data) {
          auto* test = static_cast<StagedCopyTest*>(user_data);
          test->release_.acquire();
          return iree_hip_staged_copy_3d(nullptr, &test->destination_,
                                         &test->source_, 4, 2, 2);
        },
        this, &copy_node));
    IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
        graph_, &copy_node, 1,
        +[](void* user_data) {
          static_cast<StagedCopyTest*>(user_data)->dependent_ran_ = true;
        },
        this, nullptr));
    IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
        graph_, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &executable_));
    iree_hal_streaming_graph_exec_launch_result_t launch_result;
    IREE_ASSERT_OK(iree_hal_streaming_graph_exec_launch(executable_, stream_,
                                                        &launch_result));
    EXPECT_EQ(launch_result, IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS);
    released_ = true;
    release_.release();
  }

  // Frontend owner shared by both contexts and every graph callback.
  iree_hal_streaming_host_callback_queue_t callbacks_ = {};
  // Registry entry borrowing the suite's single native CPU device.
  iree_hal_streaming_device_t device_entry_ = {};
  // Separate compatibility contexts exercise peer staging on one CPU device.
  std::array<iree_hal_streaming_context_t*, 2> contexts_ = {};
  // One host-visible allocation per context, live through graph completion.
  std::array<iree_hal_streaming_buffer_t*, 2> buffers_ = {};
  // Graph submission stream owned by the first context.
  iree_hal_streaming_stream_t* stream_ = nullptr;
  // Template owning the operation and its dependent marker.
  iree_hal_streaming_graph_t* graph_ = nullptr;
  // Executable retained until all operation callbacks have ended.
  iree_hal_streaming_graph_exec_t* executable_ = nullptr;
  // Source descriptor borrowed by the operation.
  iree_hip_staged_copy_endpoint_t source_ = {};
  // Destination descriptor borrowed by the operation.
  iree_hip_staged_copy_endpoint_t destination_ = {};
  // Host source remains live through cleanup even when a test assertion fails.
  std::array<uint8_t, 64> host_source_ = {};
  // Holds the operation until every dependent block is accepted.
  std::binary_semaphore release_{0};
  // Tracks whether teardown still needs to release the operation.
  bool released_ = false;
  // True only if the graph's dependent host function executes.
  std::atomic<bool> dependent_ran_{false};
  // Expected terminal stream status, including an injected dependency failure.
  iree_status_code_t expected_status_ = IREE_STATUS_OK;
  // Context's owned native queue, restored before context teardown.
  iree_hal_queue_t* original_context_queue_ = nullptr;
  // CPU-only dependency wrapper, live until the frontend worker is joined.
  RejectSecondTransferQueue fault_queue_ = {};
};

TEST_F(StagedCopyTest, GraphCopyBetweenContextsPreservesPitchesAndCompletes) {
  auto* source = static_cast<uint8_t*>(buffers_[0]->host_ptr);
  for (size_t i = 0; i < 64; ++i) {
    source[i] = static_cast<uint8_t>(i + 1);
  }
  LaunchCopy(DeviceEndpoint(1), DeviceEndpoint(0));
  ASSERT_FALSE(HasFatalFailure());
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
  EXPECT_TRUE(dependent_ran_);
  auto* destination = static_cast<uint8_t*>(buffers_[1]->host_ptr);
  for (size_t i = 0; i < 64; ++i) {
    const bool copied = (i >= 1 && i < 5) || (i >= 9 && i < 13) ||
                        (i >= 25 && i < 29) || (i >= 33 && i < 37);
    EXPECT_EQ(destination[i], copied ? source[i] : 0) << i;
  }
}

TEST_F(StagedCopyTest, ContextQueueRejectionStopsRowsAndFailsDependents) {
  host_source_.fill(0x5A);
  original_context_queue_ = contexts_[0]->queue;
  fault_queue_.Initialize(original_context_queue_);
  contexts_[0]->queue = &fault_queue_.base;
  expected_status_ = IREE_STATUS_ABORTED;
  const iree_hip_staged_copy_endpoint_t source_endpoint = {
      .context = nullptr,
      .buffer = nullptr,
      .offset = 0,
      .host_pointer = host_source_.data() + 1,
      .row_pitch = 8,
      .slice_pitch = 24,
  };
  LaunchCopy(DeviceEndpoint(0), source_endpoint);
  ASSERT_FALSE(HasFatalFailure());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_hal_streaming_stream_synchronize(stream_));
  EXPECT_FALSE(dependent_ran_);
  EXPECT_EQ(fault_queue_.call_count, 2);
  auto* destination = static_cast<uint8_t*>(buffers_[0]->host_ptr);
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(destination[i], i >= 1 && i < 5 ? 0x5A : 0) << i;
  }
}

}  // namespace
