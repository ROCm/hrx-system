// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kernel_queue.h"

#include <ntstatus.h>

#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/wddm/device.h"
#include "libamdf/src/wait.h"

namespace {

struct FakeQueueState {
  // Mapped completion storage owned by the native queue dependency.
  volatile uint64_t progress = 0;
  // Native submission rejection selected by the test.
  NTSTATUS submit_status = STATUS_SUCCESS;
  // Native CPU wait registration result selected by the test.
  NTSTATUS wait_status = STATUS_SUCCESS;
  // Native teardown result returned by the bridge dependency.
  NTSTATUS destroy_status = STATUS_SUCCESS;
  // Number of final bridge release calls.
  uint32_t destroy_count = 0;
  // Host queue metadata still owned by the production provider.
  uint32_t live_allocation_count = 0;
  // Progress published concurrently with a failed wait registration.
  uint64_t progress_on_wait_error = 0;
  // Explicit fence value published by a successful native wait, or zero to
  // publish its requested point.
  uint64_t progress_on_wait_success = 0;
  // Values assigned to actual native submission attempts.
  std::vector<uint64_t> submitted_values;
  // Number of native wait registration attempts.
  uint32_t wait_count = 0;
  // Event retained by an accepted wait whose completion is supplied later.
  HANDLE deferred_wait_event = nullptr;
  // Number of cold-path device-state diagnostics.
  uint32_t diagnostic_count = 0;
};

FakeQueueState* current_state = nullptr;

void* AMDF_CALL Allocate(void* user_data, uint64_t byte_length,
                         uint64_t minimum_alignment) {
  const auto allocator = amdf_allocator_system();
  void* pointer =
      allocator.allocate(allocator.user_data, byte_length, minimum_alignment);
  if (pointer != nullptr) {
    ++static_cast<FakeQueueState*>(user_data)->live_allocation_count;
  }
  return pointer;
}

void AMDF_CALL Free(void* user_data, void* allocation) {
  if (allocation == nullptr) {
    return;
  }
  --static_cast<FakeQueueState*>(user_data)->live_allocation_count;
  const auto allocator = amdf_allocator_system();
  allocator.free(allocator.user_data, allocation);
}

amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL FakeCreateQueue(
    amdf_wkmi_bridge_gpu_adapter_t* adapter,
    const amdf_wkmi_bridge_gpu_kernel_queue_create_info_t* create_info,
    amdf_wkmi_bridge_gpu_kernel_queue_t** out_queue,
    amdf_wkmi_bridge_gpu_kernel_queue_info_t* out_info,
    uint32_t* out_native_status) {
  auto* state = reinterpret_cast<FakeQueueState*>(adapter);
  EXPECT_EQ(create_info->device_handle, 0x10u);
  EXPECT_EQ(create_info->command_type,
            AMDF_WKMI_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4);
  *out_info = {};
  out_info->structure_size = sizeof(*out_info);
  out_info->command_buffer_alignment = 4;
  out_info->maximum_command_buffer_byte_length = 4096;
  out_info->progress_fence_handle = 0x20;
  out_info->progress_fence_pointer = &state->progress;
  out_info->progress_fence_device_address = 0x10000;
  *out_native_status = 0;
  *out_queue = reinterpret_cast<amdf_wkmi_bridge_gpu_kernel_queue_t*>(state);
  return AMDF_WKMI_BRIDGE_RESULT_SUCCESS;
}

amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL
FakeSubmitQueue(amdf_wkmi_bridge_gpu_kernel_queue_t* queue,
                uint64_t command_address, uint64_t command_byte_length,
                uint64_t progress_value, uint32_t* out_native_status) {
  auto* state = reinterpret_cast<FakeQueueState*>(queue);
  EXPECT_EQ(command_address, 0x20000u);
  EXPECT_EQ(command_byte_length, 64u);
  state->submitted_values.push_back(progress_value);
  *out_native_status = static_cast<uint32_t>(state->submit_status);
  return state->submit_status == STATUS_SUCCESS
             ? AMDF_WKMI_BRIDGE_RESULT_SUCCESS
             : AMDF_WKMI_BRIDGE_RESULT_NATIVE_FAILURE;
}

amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL FakeDestroyQueue(
    amdf_wkmi_bridge_gpu_kernel_queue_t* queue, uint32_t* out_native_status) {
  auto* state = reinterpret_cast<FakeQueueState*>(queue);
  ++state->destroy_count;
  *out_native_status = static_cast<uint32_t>(state->destroy_status);
  return state->destroy_status == STATUS_SUCCESS
             ? AMDF_WKMI_BRIDGE_RESULT_SUCCESS
             : AMDF_WKMI_BRIDGE_RESULT_NATIVE_FAILURE;
}

NTSTATUS APIENTRY FakeGetDeviceState(D3DKMT_GETDEVICESTATE* query) {
  ++current_state->diagnostic_count;
  query->ExecutionState = D3DKMT_DEVICEEXECUTION_ACTIVE;
  return STATUS_SUCCESS;
}

NTSTATUS APIENTRY
FakeWaitFromCpu(const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU* wait) {
  ++current_state->wait_count;
  EXPECT_EQ(wait->hDevice, 0x10u);
  EXPECT_EQ(wait->ObjectCount, 1u);
  if (current_state->wait_status == STATUS_SUCCESS) {
    current_state->progress = current_state->progress_on_wait_success != 0
                                  ? current_state->progress_on_wait_success
                                  : wait->FenceValueArray[0];
    EXPECT_TRUE(SetEvent(wait->hAsyncEvent));
  } else {
    current_state->progress = current_state->progress_on_wait_error;
  }
  return current_state->wait_status;
}

class WindowsGpuKernelQueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    current_state = &state_;
    bridge_.gpu_kernel_queue_create = FakeCreateQueue;
    bridge_.gpu_kernel_queue_submit = FakeSubmitQueue;
    bridge_.gpu_kernel_queue_destroy = FakeDestroyQueue;
    kmt_.get_device_state = FakeGetDeviceState;
    kmt_.wait_from_cpu = FakeWaitFromCpu;
    device_.host_allocator = {&state_, Allocate, nullptr, Free};
    device_.device = 0x10;
    device_.kmt = &kmt_;
    device_.wkmi_adapter.api = &bridge_;
    device_.wkmi_adapter.native =
        reinterpret_cast<amdf_wkmi_bridge_gpu_adapter_t*>(&state_);
    amdf_kmt_device_status_initialize(&device_.status);
    ASSERT_EQ(amdf_gpu_umd_kernel_queue_create(
                  &device_, AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, &queue_),
              AMDF_STATUS_OK);
  }

  void TearDown() override {
    // This fixture submits no hardware work. Publish the dependency's final
    // retirement before invoking production teardown of an idle native queue.
    if (!state_.submitted_values.empty()) {
      state_.progress = state_.submitted_values.back();
    }
    if (queue_) {
      EXPECT_EQ(amdf_gpu_umd_kernel_queue_destroy(queue_), AMDF_STATUS_OK);
    }
    EXPECT_EQ(state_.live_allocation_count, 0u);
    current_state = nullptr;
  }

  amdf_status_t Submit(uint64_t submission) {
    return amdf_gpu_umd_kernel_queue_submit(queue_, 0x20000, 64, submission);
  }

  // Native dependency responses, with no hardware access.
  FakeQueueState state_;
  // Existing WKMI function table seam, not a replacement queue implementation.
  amdf_wkmi_bridge_api_t bridge_ = {};
  // Native KMT procedures needed by the production UMD wait path.
  amdf_kmt_api_t kmt_ = {};
  // Device state consumed directly by the production queue.
  amdf_gpu_umd_device_t device_ = {};
  // Production UMD queue under test.
  amdf_gpu_umd_kernel_queue_t* queue_ = nullptr;
};

TEST_F(WindowsGpuKernelQueueTest, BridgeFailureConsumesProviderOwner) {
  state_.destroy_status = STATUS_DEVICE_BUSY;
  ASSERT_EQ(state_.live_allocation_count, 1u);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_destroy(queue_),
            amdf_kmt_make_status(STATUS_DEVICE_BUSY));
  queue_ = nullptr;
  EXPECT_EQ(state_.destroy_count, 1u);
  EXPECT_EQ(state_.live_allocation_count, 0u);
}

TEST_F(WindowsGpuKernelQueueTest,
       PublishesSeveralPointsBeforeWaitingForPrefix) {
  ASSERT_EQ(Submit(1), AMDF_STATUS_OK);
  ASSERT_EQ(Submit(2), AMDF_STATUS_OK);
  ASSERT_EQ(Submit(3), AMDF_STATUS_OK);
  EXPECT_EQ(state_.submitted_values, (std::vector<uint64_t>{1, 2, 3}));
  EXPECT_EQ(state_.progress, 0u);
  EXPECT_EQ(state_.wait_count, 0u);
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_wait(queue_, 3, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_progress(queue_), 3u);
  EXPECT_EQ(state_.wait_count, 1u);
}

TEST_F(WindowsGpuKernelQueueTest, RejectedCompletionPointCanBeSubmittedAgain) {
  constexpr uint64_t submission = 1;
  state_.submit_status = STATUS_NO_MEMORY;
  EXPECT_EQ(Submit(submission), amdf_kmt_make_status(STATUS_NO_MEMORY));
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_terminal_status(queue_),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_progress(queue_), 0u);
  EXPECT_EQ(state_.diagnostic_count, 1u);

  state_.submit_status = STATUS_SUCCESS;
  EXPECT_EQ(Submit(submission), AMDF_STATUS_OK);
  EXPECT_EQ(state_.submitted_values, (std::vector<uint64_t>{1, 1}));
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_wait(queue_, submission, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.diagnostic_count, 1u);
}

TEST_F(WindowsGpuKernelQueueTest, FatalRejectionStopsAllQueuesOnDevice) {
  amdf_gpu_umd_kernel_queue_t* sibling = nullptr;
  ASSERT_EQ(amdf_gpu_umd_kernel_queue_create(
                &device_, AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, &sibling),
            AMDF_STATUS_OK);
  state_.submit_status = STATUS_DEVICE_REMOVED;
  constexpr uint64_t submission = 1;
  const amdf_status_t lost = amdf_kmt_make_status(STATUS_DEVICE_REMOVED);
  EXPECT_EQ(Submit(submission), lost);
  state_.submit_status = STATUS_SUCCESS;
  EXPECT_EQ(Submit(submission), lost);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_submit(sibling, 0x20000, 64, submission),
            lost);
  EXPECT_EQ(state_.submitted_values.size(), 1u);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_terminal_status(sibling), lost);
  EXPECT_EQ(state_.diagnostic_count, 0u);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_destroy(sibling), AMDF_STATUS_OK);
}

TEST_F(WindowsGpuKernelQueueTest, RecoverableWaitErrorDoesNotReplayCommand) {
  constexpr uint64_t submission = 1;
  ASSERT_EQ(Submit(submission), AMDF_STATUS_OK);
  state_.wait_status = STATUS_NO_MEMORY;
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_wait(queue_, submission, &deadline),
            amdf_kmt_make_status(STATUS_NO_MEMORY));
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_terminal_status(queue_),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_progress(queue_), 0u);
  state_.wait_status = STATUS_SUCCESS;
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_wait(queue_, submission, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.submitted_values.size(), 1u);
  EXPECT_EQ(state_.wait_count, 2u);
}

TEST_F(WindowsGpuKernelQueueTest,
       NativeWaitFailureIsNotHiddenByConcurrentProgress) {
  constexpr uint64_t submission = 1;
  ASSERT_EQ(Submit(submission), AMDF_STATUS_OK);
  state_.wait_status = STATUS_DEVICE_REMOVED;
  state_.progress_on_wait_error = submission;
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_wait(queue_, submission, &deadline),
            amdf_kmt_make_status(STATUS_DEVICE_REMOVED));
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_progress(queue_), submission);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_terminal_status(queue_),
            amdf_kmt_make_status(STATUS_DEVICE_REMOVED));
}

TEST_F(WindowsGpuKernelQueueTest, TimeoutDoesNotRetireOrFailAcceptedWork) {
  constexpr uint64_t submission = 1;
  ASSERT_EQ(Submit(submission), AMDF_STATUS_OK);
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(0, 0, &deadline), AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_wait(queue_, submission, &deadline),
            amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED));
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_progress(queue_), 0u);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_terminal_status(queue_),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.wait_count, 0u);
  EXPECT_EQ(state_.diagnostic_count, 0u);
  ASSERT_EQ(amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_wait(queue_, submission, &deadline),
            AMDF_STATUS_OK);
}

TEST_F(WindowsGpuKernelQueueTest,
       EarlierWaitCanProgressAfterLaterWaitTimesOut) {
  ASSERT_EQ(Submit(1), AMDF_STATUS_OK);
  ASSERT_EQ(Submit(2), AMDF_STATUS_OK);
  ASSERT_EQ(Submit(3), AMDF_STATUS_OK);
  kmt_.wait_from_cpu =
      [](const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU* wait) -> NTSTATUS {
    ++current_state->wait_count;
    EXPECT_EQ(wait->FenceValueArray[0], 2u);
    current_state->deferred_wait_event = wait->hAsyncEvent;
    return STATUS_SUCCESS;
  };
  const auto timed_out =
      amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
  // These finite waits exercise timeout retention. The native dependency
  // deliberately leaves point 2 incomplete until after point 1 is observed.
  constexpr uint64_t timeout = UINT64_C(100000000);
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(timeout, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_wait(queue_, 2, &deadline), timed_out);
  ASSERT_NE(state_.deferred_wait_event, nullptr);
  EXPECT_EQ(state_.wait_count, 1u);
  EXPECT_EQ(state_.progress, 0u);

  // Repeating the same timed-out wait reuses its accepted registration.
  ASSERT_EQ(amdf_wait_deadline_initialize(timeout, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_wait(queue_, 2, &deadline), timed_out);
  EXPECT_EQ(state_.wait_count, 1u);

  // An earlier request needs its own notification. Native failure must be
  // returned without discarding the accepted later request or either command.
  kmt_.wait_from_cpu = FakeWaitFromCpu;
  state_.wait_status = STATUS_NO_MEMORY;
  ASSERT_EQ(amdf_wait_deadline_initialize(timeout, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_wait(queue_, 1, &deadline),
            amdf_kmt_make_status(STATUS_NO_MEMORY));
  EXPECT_EQ(state_.wait_count, 2u);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_terminal_status(queue_),
            AMDF_STATUS_OK);
  state_.wait_status = STATUS_SUCCESS;
  ASSERT_EQ(amdf_wait_deadline_initialize(timeout, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_wait(queue_, 1, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.progress, 1u);
  EXPECT_EQ(state_.wait_count, 3u);

  // The original notification can still arrive. Reusing the event for point
  // 3 must observe its actual progress, not confuse the old wake with it.
  state_.progress = 2;
  EXPECT_TRUE(SetEvent(state_.deferred_wait_event));
  ASSERT_EQ(amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_wait(queue_, 3, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.progress, 3u);
  EXPECT_EQ(state_.wait_count, 4u);
  EXPECT_EQ(state_.submitted_values, (std::vector<uint64_t>{1, 2, 3}));
}

TEST_F(WindowsGpuKernelQueueTest, ResetFenceDoesNotCompleteAcceptedWork) {
  constexpr uint64_t submission = 1;
  ASSERT_EQ(Submit(submission), AMDF_STATUS_OK);
  state_.progress = UINT64_MAX;
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_progress(queue_), 0u);
  const auto lost = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_terminal_status(queue_), lost);
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_wait(queue_, submission, &deadline),
            lost);
  EXPECT_EQ(Submit(submission + 1), lost);
  EXPECT_EQ(state_.submitted_values.size(), 1u);
  EXPECT_EQ(state_.wait_count, 0u);
  EXPECT_EQ(state_.diagnostic_count, 0u);
}

TEST_F(WindowsGpuKernelQueueTest, ResetDuringWaitDoesNotCompleteAcceptedWork) {
  constexpr uint64_t submission = 1;
  ASSERT_EQ(Submit(submission), AMDF_STATUS_OK);
  state_.progress_on_wait_success = UINT64_MAX;
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline),
            AMDF_STATUS_OK);
  const auto lost = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_wait(queue_, submission, &deadline),
            lost);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_progress(queue_), 0u);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_terminal_status(queue_), lost);
  EXPECT_EQ(state_.submitted_values.size(), 1u);
  EXPECT_EQ(state_.wait_count, 1u);
  EXPECT_EQ(state_.diagnostic_count, 0u);
}

}  // namespace
