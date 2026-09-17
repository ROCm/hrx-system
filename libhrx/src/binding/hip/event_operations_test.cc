// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/event_operations.h"

#include <array>

#include "common/init_test_util.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct ProbedHostAllocator {
  iree_allocator_t delegate = iree_allocator_system();
  bool fail_allocations = false;
  iree_host_size_t fail_larger_than = 0;
  iree_host_size_t last_allocation_size = 0;
  int failed_allocation_count = 0;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<ProbedHostAllocator*>(self);
    if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
        command == IREE_ALLOCATOR_COMMAND_CALLOC ||
        command == IREE_ALLOCATOR_COMMAND_REALLOC) {
      const auto* alloc_params =
          static_cast<const iree_allocator_alloc_params_t*>(params);
      allocator->last_allocation_size = alloc_params->byte_length;
      if (allocator->fail_allocations ||
          (allocator->fail_larger_than != 0 &&
           alloc_params->byte_length > allocator->fail_larger_than)) {
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

class EventOperationsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_initialize(/*flags=*/0)));
    cpu_initialized_ = true;
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_device_get(/*index=*/0, &hrx_device_)));

    device_entry_.hrx_device = hrx_device_;
    device_entry_.hal_device = hrx_device_hal(hrx_device_);
    iree_slim_mutex_initialize(&device_entry_.primary_context_mutex);
    iree_slim_mutex_initialize(&device_entry_.graph_memory_mutex);
    iree_arena_block_pool_initialize(/*block_size=*/64 * 1024,
                                     iree_allocator_system(),
                                     &device_entry_.block_pool);
    device_entry_initialized_ = true;

    device_registry_.host_allocator = registry_allocator_.AsAllocator();
    iree_slim_mutex_initialize(&device_registry_.context_list.mutex);
    iree_notification_initialize(&device_registry_.context_list.changed);
    iree_hal_streaming_set_device_registry_for_testing(&device_registry_);
    device_registry_installed_ = true;

    iree_hal_streaming_context_flags_t context_flags = {};
    context_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
    IREE_ASSERT_OK(iree_hal_streaming_context_create(
        &device_entry_, context_flags, context_allocator_.AsAllocator(),
        &context_));
  }

  void TearDown() override {
    registry_allocator_.fail_allocations = false;
    registry_allocator_.fail_larger_than = 0;
    for (iree_host_size_t i = 0; i < streams_.size(); ++i) {
      if (streams_[i] && streams_[i]->capture_status !=
                             IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
        iree_hal_streaming_graph_t* graph = nullptr;
        iree_status_ignore(iree_hal_streaming_end_capture(streams_[i], &graph));
        iree_hal_streaming_graph_release(graph);
      }
      iree_hal_streaming_event_release(events_[i]);
      iree_hal_streaming_stream_release(streams_[i]);
    }
    iree_hal_streaming_context_release(context_);
    if (device_registry_installed_) {
      EXPECT_EQ(nullptr, device_registry_.context_list.head);
      EXPECT_EQ(nullptr, device_registry_.context_list.tail);
      iree_hal_streaming_set_device_registry_for_testing(nullptr);
      iree_notification_deinitialize(&device_registry_.context_list.changed);
      iree_slim_mutex_deinitialize(&device_registry_.context_list.mutex);
    }
    if (device_entry_initialized_) {
      iree_arena_block_pool_deinitialize(&device_entry_.block_pool);
      iree_slim_mutex_deinitialize(&device_entry_.graph_memory_mutex);
      iree_slim_mutex_deinitialize(&device_entry_.primary_context_mutex);
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
        /*priority=*/0, context_allocator_.AsAllocator(), out_stream);
    iree_hal_queue_release(queue);
    return status;
  }

  iree_status_t BeginCapturedEvent(iree_host_size_t index) {
    if (IREE_UNLIKELY(index >= streams_.size())) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "captured-event fixture index out of range");
    }
    IREE_RETURN_IF_ERROR(CreateNonBlockingStream(&streams_[index]));
    IREE_RETURN_IF_ERROR(iree_hal_streaming_event_create(
        context_, IREE_HAL_STREAMING_EVENT_FLAG_NONE,
        context_allocator_.AsAllocator(), &events_[index]));
    IREE_RETURN_IF_ERROR(iree_hal_streaming_begin_capture(
        streams_[index], IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
    return iree_hal_streaming_event_record(events_[index], streams_[index]);
  }

  void ExpectCaptureStatus(iree_host_size_t index,
                           iree_hal_streaming_capture_status_t status) {
    ASSERT_LT(index, streams_.size());
    EXPECT_EQ(status, streams_[index]->capture_status);
  }

  void EndInvalidatedCapture(iree_host_size_t index) {
    ASSERT_LT(index, streams_.size());
    iree_hal_streaming_graph_t* graph = nullptr;
    iree_status_t status =
        iree_hal_streaming_end_capture(streams_[index], &graph);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, status);
    EXPECT_EQ(nullptr, graph);
    iree_hal_streaming_graph_release(graph);
  }

  ProbedHostAllocator registry_allocator_;
  ProbedHostAllocator context_allocator_;
  bool cpu_initialized_ = false;
  bool device_entry_initialized_ = false;
  bool device_registry_installed_ = false;
  hrx_device_t hrx_device_ = nullptr;
  iree_hal_streaming_device_t device_entry_ = {};
  iree_hal_streaming_device_registry_t device_registry_ = {};
  iree_hal_streaming_context_t* context_ = nullptr;
  std::array<iree_hal_streaming_stream_t*, 2> streams_ = {};
  std::array<iree_hal_streaming_event_t*, 2> events_ = {};
};

TEST_F(EventOperationsTest,
       QuerySnapshotFailureReturnsOutOfMemoryAndPreservesSession) {
  IREE_ASSERT_OK(BeginCapturedEvent(/*index=*/0));
  registry_allocator_.fail_allocations = true;
  bool captured_path = false;
  EXPECT_EQ(hipErrorOutOfMemory,
            iree_hip_event_query_retained(events_[0], &captured_path));
  registry_allocator_.fail_allocations = false;
  EXPECT_TRUE(captured_path);
  EXPECT_EQ(1, registry_allocator_.failed_allocation_count);
  ExpectCaptureStatus(/*index=*/0, IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE);

  EXPECT_EQ(hipErrorCapturedEvent,
            iree_hip_event_query_retained(events_[0], &captured_path));
  EXPECT_TRUE(captured_path);
  ExpectCaptureStatus(/*index=*/0,
                      IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED);
  EndInvalidatedCapture(/*index=*/0);
}

TEST_F(EventOperationsTest,
       SynchronizeSnapshotFailureReturnsOutOfMemoryAndPreservesSession) {
  IREE_ASSERT_OK(BeginCapturedEvent(/*index=*/0));
  registry_allocator_.fail_allocations = true;
  bool captured_path = false;
  EXPECT_EQ(hipErrorOutOfMemory,
            iree_hip_event_synchronize_retained(events_[0], &captured_path));
  registry_allocator_.fail_allocations = false;
  EXPECT_TRUE(captured_path);
  EXPECT_EQ(1, registry_allocator_.failed_allocation_count);
  ExpectCaptureStatus(/*index=*/0, IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE);

  EXPECT_EQ(hipErrorCapturedEvent,
            iree_hip_event_synchronize_retained(events_[0], &captured_path));
  EXPECT_TRUE(captured_path);
  ExpectCaptureStatus(/*index=*/0,
                      IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED);
  EndInvalidatedCapture(/*index=*/0);
}

TEST_F(EventOperationsTest,
       ElapsedSnapshotFailureChangesNeitherCapturedSession) {
  // Measure the registry snapshot allocation for one session. The two-session
  // operation must allocate a larger context table before mutating either
  // session; a sequential implementation would make two smaller allocations
  // and would therefore miss the injected failure after changing session 0.
  IREE_ASSERT_OK(BeginCapturedEvent(/*index=*/0));
  bool captured_path = false;
  EXPECT_EQ(hipErrorCapturedEvent,
            iree_hip_event_query_retained(events_[0], &captured_path));
  ASSERT_TRUE(captured_path);
  const iree_host_size_t single_session_snapshot_size =
      registry_allocator_.last_allocation_size;
  ASSERT_GT(single_session_snapshot_size, 0u);
  EndInvalidatedCapture(/*index=*/0);
  iree_hal_streaming_event_release(events_[0]);
  events_[0] = nullptr;
  iree_hal_streaming_stream_release(streams_[0]);
  streams_[0] = nullptr;

  IREE_ASSERT_OK(BeginCapturedEvent(/*index=*/0));
  IREE_ASSERT_OK(BeginCapturedEvent(/*index=*/1));
  ASSERT_NE(streams_[0]->capture_graph, streams_[1]->capture_graph);
  registry_allocator_.fail_larger_than = single_session_snapshot_size;
  float milliseconds = -1.0f;
  EXPECT_EQ(hipErrorOutOfMemory, iree_hip_event_elapsed_time_retained(
                                     &milliseconds, events_[0], events_[1]));
  registry_allocator_.fail_larger_than = 0;
  EXPECT_EQ(-1.0f, milliseconds);
  EXPECT_EQ(1, registry_allocator_.failed_allocation_count);
  ExpectCaptureStatus(/*index=*/0, IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE);
  ExpectCaptureStatus(/*index=*/1, IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE);

  EXPECT_EQ(hipErrorCapturedEvent, iree_hip_event_elapsed_time_retained(
                                       &milliseconds, events_[0], events_[1]));
  ExpectCaptureStatus(/*index=*/0,
                      IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED);
  ExpectCaptureStatus(/*index=*/1,
                      IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED);
  EndInvalidatedCapture(/*index=*/0);
  EndInvalidatedCapture(/*index=*/1);
}

}  // namespace
