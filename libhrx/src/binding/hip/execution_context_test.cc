// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/execution_context.h"

#include <array>
#include <cstring>

#include "binding/hip/execution_context_test_util.h"
#include "common/init_test_util.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct ProbedHostAllocator {
  iree_allocator_t delegate = iree_allocator_system();
  iree_host_size_t allocation_attempt_count = 0;
  iree_host_size_t fail_on_attempt = 0;
  iree_host_size_t failed_allocation_count = 0;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<ProbedHostAllocator*>(self);
    if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
        command == IREE_ALLOCATOR_COMMAND_CALLOC ||
        command == IREE_ALLOCATOR_COMMAND_REALLOC) {
      ++allocator->allocation_attempt_count;
      if (allocator->allocation_attempt_count == allocator->fail_on_attempt) {
        ++allocator->failed_allocation_count;
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "injected allocation failure");
      }
    }
    return allocator->delegate.ctl(allocator->delegate.self, command, params,
                                   inout_ptr);
  }

  iree_allocator_t AsAllocator() {
    return iree_allocator_t{this, &ProbedHostAllocator::Control};
  }
};

class ExecutionContextTest : public ::testing::Test {
 protected:
  static iree_status_t CreatePrimaryResource(
      iree_hal_streaming_device_t* device,
      const iree_hal_queue_family_t* queue_family,
      hipDevResource* out_resource) {
    (void)device;
    (void)queue_family;
    memset(out_resource, 0, sizeof(*out_resource));
    out_resource->type = hipDevResourceTypeSm;
    return iree_ok_status();
  }

  void SetUp() override {
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_initialize(/*flags=*/0)));
    cpu_initialized_ = true;
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_device_get(/*index=*/0, &hrx_device_)));

    device_entry_ = &device_registry_.devices[0];
    device_entry_->ordinal = 0;
    device_entry_->hrx_device = hrx_device_;
    device_entry_->hal_device = hrx_device_hal(hrx_device_);
    iree_slim_mutex_initialize(&device_entry_->primary_context_mutex);
    primary_context_mutex_initialized_ = true;
    iree_slim_mutex_initialize(&device_entry_->graph_memory_mutex);
    graph_memory_mutex_initialized_ = true;
    iree_arena_block_pool_initialize(/*block_size=*/64 * 1024,
                                     iree_allocator_system(),
                                     &device_entry_->block_pool);
    block_pool_initialized_ = true;
    IREE_ASSERT_OK(iree_hal_streaming_execution_resource_table_initialize(
        device_entry_->hal_device, iree_allocator_system(),
        &device_entry_->execution_resource_table));
    execution_resource_table_initialized_ = true;

    device_registry_.host_allocator = host_allocator_.AsAllocator();
    device_registry_.device_count = 1;
    iree_slim_mutex_initialize(&device_registry_.context_list.mutex);
    iree_notification_initialize(&device_registry_.context_list.changed);
    context_list_initialized_ = true;
    iree_hal_streaming_set_device_registry_for_testing(&device_registry_);
    device_registry_installed_ = true;

    previous_primary_resource_factory_ =
        iree_hip_execution_context_exchange_primary_resource_factory_for_testing(
            &ExecutionContextTest::CreatePrimaryResource);
    primary_resource_factory_installed_ = true;
    ASSERT_EQ(nullptr, previous_primary_resource_factory_);
    ASSERT_EQ(hipSuccess, iree_hip_execution_context_primary(
                              device_entry_, &execution_context_));
    EXPECT_EQ(
        &ExecutionContextTest::CreatePrimaryResource,
        iree_hip_execution_context_exchange_primary_resource_factory_for_testing(
            previous_primary_resource_factory_));
    primary_resource_factory_installed_ = false;
    context_ = device_entry_->primary_context;
    ASSERT_NE(nullptr, context_);
  }

  void TearDown() override {
    if (primary_resource_factory_installed_) {
      EXPECT_EQ(
          &ExecutionContextTest::CreatePrimaryResource,
          iree_hip_execution_context_exchange_primary_resource_factory_for_testing(
              previous_primary_resource_factory_));
      primary_resource_factory_installed_ = false;
    }
    host_allocator_.fail_on_attempt = 0;
    for (iree_hal_streaming_stream_t* stream : capture_streams_) {
      if (!stream ||
          stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
        continue;
      }
      iree_hal_streaming_graph_t* graph = nullptr;
      iree_status_ignore(iree_hal_streaming_end_capture(stream, &graph));
      iree_hal_streaming_graph_release(graph);
    }
    iree_hal_streaming_event_release(event_);
    iree_hal_streaming_stream_release(extra_stream_);

    EXPECT_EQ(hipSuccess, iree_hip_execution_context_reset_all());
    execution_context_ = nullptr;
    context_ = nullptr;
    if (device_entry_ && device_entry_->primary_context) {
      iree_hal_streaming_context_release(device_entry_->primary_context);
      device_entry_->primary_context = nullptr;
    }
    if (device_entry_) {
      hrx_mem_pool_release(device_entry_->current_mem_pool);
      device_entry_->current_mem_pool = nullptr;
      hrx_mem_pool_release(device_entry_->default_mem_pool);
      device_entry_->default_mem_pool = nullptr;
    }

    if (device_registry_installed_) {
      EXPECT_EQ(nullptr, device_registry_.context_list.head);
      EXPECT_EQ(nullptr, device_registry_.context_list.tail);
      iree_hal_streaming_set_device_registry_for_testing(nullptr);
    }
    if (context_list_initialized_) {
      iree_notification_deinitialize(&device_registry_.context_list.changed);
      iree_slim_mutex_deinitialize(&device_registry_.context_list.mutex);
    }
    if (execution_resource_table_initialized_) {
      iree_hal_streaming_execution_resource_table_deinitialize(
          &device_entry_->execution_resource_table);
    }
    if (block_pool_initialized_) {
      iree_arena_block_pool_deinitialize(&device_entry_->block_pool);
    }
    if (graph_memory_mutex_initialized_) {
      iree_slim_mutex_deinitialize(&device_entry_->graph_memory_mutex);
    }
    if (primary_context_mutex_initialized_) {
      iree_slim_mutex_deinitialize(&device_entry_->primary_context_mutex);
    }
    if (cpu_initialized_) {
      IREE_EXPECT_OK(HRX_CALL(hrx_cpu_shutdown()));
    }
  }

  iree_status_t CreateNonBlockingStream(
      iree_hal_streaming_stream_t** out_stream) {
    const iree_hal_queue_family_t* queue_family =
        iree_hal_device_queue_family(context_->device, /*family_ordinal=*/0);
    if (!queue_family) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "task device has no queue family");
    }
    iree_hal_queue_params_t queue_params;
    iree_hal_queue_params_initialize(&queue_params);
    iree_hal_queue_t* queue = nullptr;
    IREE_RETURN_IF_ERROR(
        iree_hal_queue_acquire(queue_family, &queue_params, &queue));
    iree_status_t status = iree_hal_streaming_stream_create(
        context_, queue, IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING,
        /*priority=*/0, host_allocator_.AsAllocator(), out_stream);
    iree_hal_queue_release(queue);
    return status;
  }

  ProbedHostAllocator host_allocator_;
  bool cpu_initialized_ = false;
  bool primary_context_mutex_initialized_ = false;
  bool graph_memory_mutex_initialized_ = false;
  bool block_pool_initialized_ = false;
  bool execution_resource_table_initialized_ = false;
  bool context_list_initialized_ = false;
  bool device_registry_installed_ = false;
  bool primary_resource_factory_installed_ = false;
  iree_hip_execution_context_primary_resource_factory_t
      previous_primary_resource_factory_ = nullptr;
  hrx_device_t hrx_device_ = nullptr;
  iree_hal_streaming_device_registry_t device_registry_ = {};
  iree_hal_streaming_device_t* device_entry_ = nullptr;
  hipExecutionCtx_t execution_context_ = nullptr;
  iree_hal_streaming_context_t* context_ = nullptr;
  iree_hal_streaming_stream_t* extra_stream_ = nullptr;
  iree_hal_streaming_event_t* event_ = nullptr;
  std::array<iree_hal_streaming_stream_t*, 2> capture_streams_ = {};
};

TEST_F(ExecutionContextTest,
       WaitEventTargetSnapshotFailureChangesNeitherCaptureSession) {
  capture_streams_[0] = context_->default_stream;
  IREE_ASSERT_OK(CreateNonBlockingStream(&extra_stream_));
  capture_streams_[1] = extra_stream_;
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_NONE,
      host_allocator_.AsAllocator(), &event_));

  for (iree_hal_streaming_stream_t* stream : capture_streams_) {
    IREE_ASSERT_OK(iree_hal_streaming_begin_capture(
        stream, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  }
  IREE_ASSERT_OK(iree_hal_streaming_event_record(event_, capture_streams_[0]));

  // Source invalidation prepares the registry context array and its retained
  // stream array first. Fail the next allocation: the target execution-context
  // stream snapshot. Neither already-prepared source nor target may change.
  const iree_host_size_t initial_attempt_count =
      host_allocator_.allocation_attempt_count;
  host_allocator_.fail_on_attempt = initial_attempt_count + 3;
  EXPECT_EQ(hipErrorOutOfMemory,
            iree_hip_execution_context_wait_event(execution_context_, event_));
  host_allocator_.fail_on_attempt = 0;
  EXPECT_EQ(initial_attempt_count + 3,
            host_allocator_.allocation_attempt_count);
  EXPECT_EQ(1u, host_allocator_.failed_allocation_count);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE,
            capture_streams_[0]->capture_status);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE,
            capture_streams_[1]->capture_status);

  EXPECT_EQ(hipErrorStreamCaptureUnsupported,
            iree_hip_execution_context_wait_event(execution_context_, event_));
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED,
            capture_streams_[0]->capture_status);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED,
            capture_streams_[1]->capture_status);
}

}  // namespace
