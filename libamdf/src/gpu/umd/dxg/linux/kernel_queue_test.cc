// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/linux/kernel_queue.h"

#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/dxg/linux/api.h"
#include "libamdf/src/gpu/umd/dxg/linux/device.h"
#include "libamdf/src/wait.h"

namespace {

constexpr NTSTATUS kStatusSuccess = 0;
constexpr NTSTATUS kStatusNoMemory = static_cast<NTSTATUS>(0xC0000017u);
constexpr NTSTATUS kStatusDeviceRemoved = static_cast<NTSTATUS>(0xC00002B6u);
constexpr NTSTATUS kStatusUnsuccessful = static_cast<NTSTATUS>(0xC0000001u);

struct FakeQueueState {
  // Mapped completion storage owned by the native queue dependency.
  volatile uint64_t progress = 0;
  // Native submission rejection selected by the test.
  NTSTATUS submit_status = kStatusSuccess;
  // Native completion-signal result after command acceptance.
  NTSTATUS signal_status = kStatusSuccess;
  // Native CPU wait registration result selected by the test.
  NTSTATUS wait_status = kStatusSuccess;
  // Native teardown result returned by the provider dependency.
  NTSTATUS destroy_status = kStatusSuccess;
  // Device execution state returned by cold-path diagnosis.
  D3DKMT_DEVICEEXECUTION_STATE execution_state = D3DKMT_DEVICEEXECUTION_ACTIVE;
  // Number of final provider release calls.
  uint32_t destroy_count = 0;
  // Host queue metadata still owned by the production provider.
  uint32_t live_allocation_count = 0;
  // Total provider allocations, including allocations freed before return.
  uint32_t allocation_count = 0;
  // Command family selected by the parameterized fixture.
  amdf_dxg_provider_gpu_queue_command_type_t command_type =
      AMDF_DXG_PROVIDER_GPU_QUEUE_COMMAND_TYPE_PM4;
  // Progress published concurrently with a failed wait registration.
  uint64_t progress_on_wait_error = 0;
  // Explicit progress published by a successful native wait.
  uint64_t progress_on_wait_success = 0;
  // Values assigned to actual native submission attempts.
  std::vector<uint64_t> submitted_values;
  // Number of native wait registration attempts.
  uint32_t wait_count = 0;
  // Number of cold-path device-state diagnostics.
  uint32_t diagnostic_count = 0;
  // Whether a successful registration immediately completes and wakes.
  bool signal_wait = true;
  // Eventfd retained by a successful asynchronous registration.
  int deferred_wait_event = -1;
  // Whether queue construction publishes a native hardware queue.
  bool has_hardware_queue = true;
  // Native mapping publication outcome.
  NTSTATUS mapping_status = kStatusSuccess;
  // Native dependency publication outcome after mapping acceptance.
  NTSTATUS ordering_status = kStatusSuccess;
  // Predecessor observed by native mapping publication.
  uint64_t mapping_predecessor = 0;
  // Queue wait point, absent when mapping publication was rejected.
  uint64_t ordering_point = 0;
  // Exact descriptor forwarded by the last mapping publication.
  D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION mapping_operation = {};
};

FakeQueueState* current_state = nullptr;
amdf_linux_dxg_connection_t* current_connection = nullptr;

void* AMDF_CALL Allocate(void* user_data, uint64_t byte_length,
                         uint64_t minimum_alignment) {
  const auto allocator = amdf_allocator_system();
  void* pointer =
      allocator.allocate(allocator.user_data, byte_length, minimum_alignment);
  if (pointer != nullptr) {
    auto* state = static_cast<FakeQueueState*>(user_data);
    ++state->live_allocation_count;
    ++state->allocation_count;
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

amdf_dxg_provider_result_t FakeCreateQueue(
    amdf_dxg_provider_gpu_adapter_t* adapter,
    const amdf_dxg_provider_gpu_kernel_queue_create_info_t* create_info,
    amdf_dxg_provider_gpu_kernel_queue_t** out_queue,
    amdf_dxg_provider_gpu_kernel_queue_info_t* out_info,
    uint32_t* out_native_status) {
  auto* state = reinterpret_cast<FakeQueueState*>(adapter);
  EXPECT_EQ(create_info->device_handle, 0x10u);
  EXPECT_EQ(create_info->command_type, state->command_type);
  *out_info = {};
  out_info->command_buffer_alignment = 4;
  out_info->maximum_command_buffer_byte_length = 4096;
  out_info->progress_fence_handle = 0x20;
  out_info->progress_fence_pointer = &state->progress;
  out_info->progress_fence_device_address = 0x10000;
  out_info->context_handle = 0x30;
  out_info->queue_handle = state->has_hardware_queue ? 0x40 : 0;
  *out_native_status = 0;
  *out_queue = reinterpret_cast<amdf_dxg_provider_gpu_kernel_queue_t*>(state);
  return AMDF_DXG_PROVIDER_RESULT_SUCCESS;
}

amdf_dxg_provider_result_t FakeSubmitQueue(
    amdf_dxg_provider_gpu_kernel_queue_t* queue, uint64_t command_address,
    uint64_t command_byte_length, uint64_t progress_value,
    uint32_t* out_accepted, uint32_t* out_native_status) {
  auto* state = reinterpret_cast<FakeQueueState*>(queue);
  EXPECT_EQ(command_address, 0x20000u);
  EXPECT_EQ(command_byte_length, 64u);
  state->submitted_values.push_back(progress_value);
  *out_accepted = state->submit_status == kStatusSuccess;
  const NTSTATUS status =
      *out_accepted ? state->signal_status : state->submit_status;
  *out_native_status = static_cast<uint32_t>(status);
  return status == kStatusSuccess ? AMDF_DXG_PROVIDER_RESULT_SUCCESS
                                  : AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE;
}

amdf_dxg_provider_result_t FakeFailCreateQueue(
    amdf_dxg_provider_gpu_adapter_t*,
    const amdf_dxg_provider_gpu_kernel_queue_create_info_t*,
    amdf_dxg_provider_gpu_kernel_queue_t**,
    amdf_dxg_provider_gpu_kernel_queue_info_t*, uint32_t* out_native_status) {
  *out_native_status = static_cast<uint32_t>(kStatusNoMemory);
  return AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE;
}

amdf_dxg_provider_result_t FakeDestroyQueue(
    amdf_dxg_provider_gpu_kernel_queue_t* queue, uint32_t* out_native_status) {
  auto* state = reinterpret_cast<FakeQueueState*>(queue);
  ++state->destroy_count;
  *out_native_status = static_cast<uint32_t>(state->destroy_status);
  return state->destroy_status == kStatusSuccess
             ? AMDF_DXG_PROVIDER_RESULT_SUCCESS
             : AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE;
}

extern "C" uint32_t __wrap_amdf_linux_dxg_connection_invoke(void* context,
                                                            uint64_t request,
                                                            void* arguments) {
  EXPECT_EQ(context, current_connection);
  switch (_IOC_NR(request)) {
    case 0x0E: {
      ++current_state->diagnostic_count;
      auto* query = static_cast<D3DKMT_GETDEVICESTATE*>(arguments);
      EXPECT_EQ(query->hDevice, 0x10u);
      EXPECT_EQ(query->StateType, D3DKMT_DEVICESTATE_EXECUTION);
      query->ExecutionState = current_state->execution_state;
      return 0;
    }
    case 0x3A: {
      ++current_state->wait_count;
      const auto* wait =
          static_cast<D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU*>(arguments);
      EXPECT_EQ(wait->hDevice, 0x10u);
      EXPECT_EQ(wait->ObjectCount, 1u);
      EXPECT_EQ(wait->ObjectHandleArray[0], 0x20u);
      EXPECT_EQ(wait->Flags.Value, 0u);
      const int event_descriptor =
          static_cast<int>(reinterpret_cast<uintptr_t>(wait->hAsyncEvent));
      EXPECT_GE(event_descriptor, 0);
      if (current_state->wait_status != kStatusSuccess) {
        current_state->progress = current_state->progress_on_wait_error;
        return static_cast<uint32_t>(current_state->wait_status);
      }
      current_state->deferred_wait_event = event_descriptor;
      if (current_state->signal_wait) {
        current_state->progress = current_state->progress_on_wait_success != 0
                                      ? current_state->progress_on_wait_success
                                      : wait->FenceValueArray[0];
        EXPECT_EQ(eventfd_write(event_descriptor, 1), 0);
      }
      return 0;
    }
    case 0x39: {
      const auto* update =
          static_cast<D3DKMT_UPDATEGPUVIRTUALADDRESS*>(arguments);
      EXPECT_EQ(update->hDevice, 0x10u);
      EXPECT_EQ(update->hContext, 0x30u);
      EXPECT_EQ(update->hFenceObject, 0x20u);
      EXPECT_EQ(update->Flags.Value, 0u);
      EXPECT_EQ(update->NumOperations, 1u);
      current_state->mapping_predecessor = update->FenceValue;
      current_state->mapping_operation = update->Operations[0];
      return static_cast<uint32_t>(current_state->mapping_status);
    }
    case 0x36: {
      const auto* wait =
          static_cast<D3DKMT_SUBMITWAITFORSYNCOBJECTSTOHWQUEUE*>(arguments);
      EXPECT_EQ(wait->hHwQueue, 0x40u);
      EXPECT_EQ(wait->ObjectCount, 1u);
      EXPECT_EQ(wait->ObjectHandleArray[0], 0x20u);
      current_state->ordering_point = wait->FenceValueArray[0];
      return static_cast<uint32_t>(current_state->ordering_status);
    }
    default:
      ADD_FAILURE() << "unexpected DXG operation " << _IOC_NR(request);
      return static_cast<uint32_t>(kStatusUnsuccessful);
  }
}

class LinuxDxgKernelQueueTest
    : public ::testing::TestWithParam<amdf_queue_command_type_t> {
 protected:
  void SetUp() override {
    current_state = &state_;
    current_connection = &connection_;
    state_.command_type = GetParam() == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
                              ? AMDF_DXG_PROVIDER_GPU_QUEUE_COMMAND_TYPE_PM4
                              : AMDF_DXG_PROVIDER_GPU_QUEUE_COMMAND_TYPE_SDMA;
    provider_.gpu_kernel_queue_create = FakeCreateQueue;
    provider_.gpu_kernel_queue_submit = FakeSubmitQueue;
    provider_.gpu_kernel_queue_destroy = FakeDestroyQueue;
    connection_.descriptor = 1;
    kmt_.connection = &connection_;
    device_.host_allocator = {&state_, Allocate, nullptr, Free};
    device_.logical_device.kmt = &kmt_;
    device_.logical_device.native = 0x10;
    device_.logical_device.provider_adapter.api = &provider_;
    device_.logical_device.provider_adapter.native =
        reinterpret_cast<amdf_dxg_provider_gpu_adapter_t*>(&state_);
    amdf_gpu_linux_dxg_device_status_initialize(&device_.status);
    ASSERT_EQ(
        amdf_gpu_linux_dxg_kernel_queue_create(&device_, GetParam(), &queue_),
        AMDF_STATUS_OK);
  }

  void TearDown() override {
    if (!state_.submitted_values.empty()) {
      state_.progress = state_.submitted_values.back();
    }
    if (queue_ != nullptr) {
      EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_destroy(queue_),
                AMDF_STATUS_OK);
    }
    EXPECT_EQ(state_.live_allocation_count, 0u);
    if (notification_event_ >= 0) {
      EXPECT_EQ(close(notification_event_), 0);
    }
    current_connection = nullptr;
    current_state = nullptr;
  }

  amdf_status_t Submit(uint64_t submission) {
    return amdf_gpu_linux_dxg_kernel_queue_submit(queue_, 0x20000, 64,
                                                  submission);
  }

  void CreateNotificationEvent() {
    notification_event_ = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    ASSERT_GE(notification_event_, 0);
    event_.type = AMDF_NATIVE_EVENT_TYPE_EVENTFD;
    event_.payload.file_descriptor = notification_event_;
  }

  void ExpectNotification() {
    eventfd_t value = 0;
    ASSERT_EQ(eventfd_read(notification_event_, &value), 0);
    EXPECT_EQ(value, 1u);
  }

  amdf_status_t Notify(uint64_t submission) {
    return amdf_gpu_linux_dxg_kernel_queue_request_notification(
        queue_, submission, &event_);
  }

  // Caller-owned event kept alive through native queue teardown.
  amdf_native_event_t event_ = {};
  // Descriptor backing `event_`, or -1 before construction.
  int notification_event_ = -1;
  // Native dependency responses, with no hardware access.
  FakeQueueState state_;
  // Existing provider function table seam, not a replacement queue.
  amdf_dxg_provider_api_t provider_ = {};
  // Instance-owned transport borrowed by the production queue.
  amdf_linux_dxg_connection_t connection_ = {};
  // Typed Linux KMT boundary used by notification and diagnosis.
  amdf_linux_dxg_kmt_t kmt_ = {};
  // Device state consumed directly by the production queue.
  amdf_gpu_umd_device_t device_ = {};
  // Production UMD queue under test.
  amdf_gpu_umd_kernel_queue_t* queue_ = nullptr;
};

TEST_P(LinuxDxgKernelQueueTest, CreatesColdQueueWithCompleteCapabilities) {
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_notification_types(queue_),
            AMDF_NATIVE_EVENT_TYPE_BIT_EVENTFD);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_mapping_capacity(queue_), 128u);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_progress(queue_), 0u);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_terminal_status(queue_),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.allocation_count, 1u);
}

TEST_P(LinuxDxgKernelQueueTest, SoftwareQueueOmitsMappingPublication) {
  state_.has_hardware_queue = false;
  amdf_gpu_umd_kernel_queue_t* software_queue = nullptr;
  ASSERT_EQ(amdf_gpu_linux_dxg_kernel_queue_create(&device_, GetParam(),
                                                   &software_queue),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_mapping_capacity(software_queue),
            0u);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_destroy(software_queue),
            AMDF_STATUS_OK);
}

TEST_P(LinuxDxgKernelQueueTest,
       QueueMappingPublishesPredecessorThenHardwareDependency) {
  D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION operation = {
      .OperationType = D3DDDI_UPDATEGPUVIRTUALADDRESS_UNMAP,
      .Unmap =
          {
              .BaseAddress = 0x40000,
              .SizeInBytes = 4096,
              .Protection = {.NoAccess = 1},
          },
  };
  bool mutation_started = false;
  bool accepted = false;
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_update_memory(
                queue_, 1, &operation, 19, &mutation_started, &accepted),
            AMDF_STATUS_OK);
  EXPECT_TRUE(mutation_started);
  EXPECT_TRUE(accepted);
  EXPECT_EQ(state_.mapping_predecessor, 18u);
  EXPECT_EQ(state_.ordering_point, 19u);
  EXPECT_EQ(state_.mapping_operation.OperationType,
            D3DDDI_UPDATEGPUVIRTUALADDRESS_UNMAP);
  EXPECT_EQ(state_.mapping_operation.Unmap.BaseAddress, 0x40000u);
  EXPECT_EQ(state_.mapping_operation.Unmap.SizeInBytes, 4096u);
  EXPECT_EQ(state_.mapping_operation.Unmap.Protection.NoAccess, 1u);
}

TEST_P(LinuxDxgKernelQueueTest, RejectedMappingLeavesPointRetryable) {
  state_.mapping_status = kStatusNoMemory;
  D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION operation = {};
  bool mutation_started = false;
  bool accepted = false;
  const amdf_status_t failure = amdf_make_status(
      AMDF_STATUS_DOMAIN_NTSTATUS, static_cast<uint32_t>(kStatusNoMemory));
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_update_memory(
                queue_, 1, &operation, 19, &mutation_started, &accepted),
            failure);
  EXPECT_TRUE(mutation_started);
  EXPECT_FALSE(accepted);
  EXPECT_EQ(state_.ordering_point, 0u);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_terminal_status(queue_),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.diagnostic_count, 1u);
}

TEST_P(LinuxDxgKernelQueueTest,
       FailedOrderingPreservesAcceptanceAndStopsQueue) {
  state_.ordering_status = kStatusNoMemory;
  D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION operation = {};
  bool mutation_started = false;
  bool accepted = false;
  const amdf_status_t failure = amdf_make_status(
      AMDF_STATUS_DOMAIN_NTSTATUS, static_cast<uint32_t>(kStatusNoMemory));
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_update_memory(
                queue_, 1, &operation, 19, &mutation_started, &accepted),
            failure);
  EXPECT_TRUE(mutation_started);
  EXPECT_TRUE(accepted);
  EXPECT_EQ(state_.mapping_predecessor, 18u);
  EXPECT_EQ(state_.ordering_point, 19u);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_terminal_status(queue_),
            failure);
  EXPECT_EQ(Submit(20), failure);
}

TEST_P(LinuxDxgKernelQueueTest, ValidatesCommandsBeforeNativePublication) {
  const auto invalid = amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  const auto out_of_range = amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_submit(queue_, 0, 64, 1), invalid);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_submit(queue_, 0x20001, 64, 1),
            invalid);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_submit(queue_, 0x20000, 0, 1),
            invalid);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_submit(queue_, 0x20000, 62, 1),
            invalid);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_submit(queue_, 0x20000, 4100, 1),
            out_of_range);
  EXPECT_TRUE(state_.submitted_values.empty());
}

TEST_P(LinuxDxgKernelQueueTest, NotificationUsesCallerEventfdAndStableFence) {
  ASSERT_NO_FATAL_FAILURE(CreateNotificationEvent());
  ASSERT_EQ(Submit(1), AMDF_STATUS_OK);
  const uint32_t allocations = state_.allocation_count;
  EXPECT_EQ(Notify(1), AMDF_STATUS_OK);
  EXPECT_EQ(state_.wait_count, 1u);
  ASSERT_NO_FATAL_FAILURE(ExpectNotification());
  EXPECT_EQ(state_.allocation_count, allocations);
  EXPECT_EQ(state_.submitted_values, (std::vector<uint64_t>{1}));
  EXPECT_EQ(state_.diagnostic_count, 0u);
}

TEST_P(LinuxDxgKernelQueueTest, CheckedPointsSignalWithoutNativeRegistration) {
  ASSERT_NO_FATAL_FAILURE(CreateNotificationEvent());
  ASSERT_EQ(Submit(1), AMDF_STATUS_OK);
  state_.progress = 1;
  EXPECT_EQ(Notify(1), AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(ExpectNotification());
  state_.progress = UINT64_MAX;
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_progress(queue_), 0u);
  EXPECT_EQ(Notify(0), AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(ExpectNotification());
  EXPECT_EQ(state_.wait_count, 0u);
}

TEST_P(LinuxDxgKernelQueueTest, RecoverableRejectionLeavesPointRetryable) {
  state_.submit_status = kStatusNoMemory;
  const amdf_status_t rejected = amdf_make_status(
      AMDF_STATUS_DOMAIN_NTSTATUS, static_cast<uint32_t>(kStatusNoMemory));
  EXPECT_EQ(Submit(1), rejected);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_terminal_status(queue_),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.diagnostic_count, 1u);
  state_.submit_status = kStatusSuccess;
  EXPECT_EQ(Submit(1), AMDF_STATUS_OK);
  EXPECT_EQ(state_.submitted_values, (std::vector<uint64_t>{1, 1}));
}

TEST_P(LinuxDxgKernelQueueTest, FatalRejectionStopsSiblingQueues) {
  amdf_gpu_umd_kernel_queue_t* sibling = nullptr;
  ASSERT_EQ(
      amdf_gpu_linux_dxg_kernel_queue_create(&device_, GetParam(), &sibling),
      AMDF_STATUS_OK);
  state_.submit_status = kStatusDeviceRemoved;
  const amdf_status_t lost = amdf_make_status(
      AMDF_STATUS_DOMAIN_NTSTATUS, static_cast<uint32_t>(kStatusDeviceRemoved));
  EXPECT_EQ(Submit(1), lost);
  state_.submit_status = kStatusSuccess;
  EXPECT_EQ(Submit(1), lost);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_submit(sibling, 0x20000, 64, 1),
            lost);
  EXPECT_EQ(state_.submitted_values.size(), 1u);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_terminal_status(sibling),
            lost);
  EXPECT_EQ(state_.diagnostic_count, 0u);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_destroy(sibling), AMDF_STATUS_OK);
}

TEST_P(LinuxDxgKernelQueueTest,
       FailedCompletionSignalPreservesAcceptedSubmission) {
  state_.signal_status = kStatusNoMemory;
  const amdf_status_t failure = amdf_make_status(
      AMDF_STATUS_DOMAIN_NTSTATUS, static_cast<uint32_t>(kStatusNoMemory));
  EXPECT_EQ(Submit(1), AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_terminal_status(queue_),
            failure);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_progress(queue_), 0u);
  EXPECT_EQ(Submit(2), failure);
  EXPECT_EQ(state_.submitted_values, (std::vector<uint64_t>{1}));
  state_.progress = 1;
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_progress(queue_), 1u);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_terminal_status(queue_),
            failure);
}

TEST_P(LinuxDxgKernelQueueTest, WaitUsesPrecreatedEventAndRetiresPoint) {
  ASSERT_EQ(Submit(1), AMDF_STATUS_OK);
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_wait(queue_, 1, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_progress(queue_), 1u);
  EXPECT_EQ(state_.wait_count, 1u);
  EXPECT_EQ(state_.allocation_count, 1u);
}

TEST_P(LinuxDxgKernelQueueTest, RecoverableWaitErrorNeverReplaysCommand) {
  ASSERT_EQ(Submit(1), AMDF_STATUS_OK);
  state_.wait_status = kStatusNoMemory;
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline),
            AMDF_STATUS_OK);
  const amdf_status_t failure = amdf_make_status(
      AMDF_STATUS_DOMAIN_NTSTATUS, static_cast<uint32_t>(kStatusNoMemory));
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_wait(queue_, 1, &deadline),
            failure);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_terminal_status(queue_),
            AMDF_STATUS_OK);
  state_.wait_status = kStatusSuccess;
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_wait(queue_, 1, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.submitted_values.size(), 1u);
  EXPECT_EQ(state_.wait_count, 2u);
}

TEST_P(LinuxDxgKernelQueueTest,
       WaitFailureRemainsVisibleDespiteConcurrentProgress) {
  ASSERT_EQ(Submit(1), AMDF_STATUS_OK);
  state_.wait_status = kStatusDeviceRemoved;
  state_.progress_on_wait_error = 1;
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline),
            AMDF_STATUS_OK);
  const amdf_status_t lost = amdf_make_status(
      AMDF_STATUS_DOMAIN_NTSTATUS, static_cast<uint32_t>(kStatusDeviceRemoved));
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_wait(queue_, 1, &deadline), lost);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_progress(queue_), 1u);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_terminal_status(queue_),
            lost);
}

TEST_P(LinuxDxgKernelQueueTest, TimeoutPreservesAcceptedWorkAndQueueHealth) {
  ASSERT_EQ(Submit(1), AMDF_STATUS_OK);
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(0, 0, &deadline), AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_wait(queue_, 1, &deadline),
            amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED));
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_progress(queue_), 0u);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_terminal_status(queue_),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.wait_count, 0u);
}

TEST_P(LinuxDxgKernelQueueTest, TimedWaitReusesRegistrationAndDrainsStaleWake) {
  ASSERT_EQ(Submit(1), AMDF_STATUS_OK);
  ASSERT_EQ(Submit(2), AMDF_STATUS_OK);
  ASSERT_EQ(Submit(3), AMDF_STATUS_OK);
  state_.signal_wait = false;
  const amdf_status_t timed_out =
      amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
  constexpr uint64_t timeout = UINT64_C(2000000);
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(timeout, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_wait(queue_, 2, &deadline),
            timed_out);
  ASSERT_GE(state_.deferred_wait_event, 0);
  EXPECT_EQ(state_.wait_count, 1u);

  ASSERT_EQ(amdf_wait_deadline_initialize(timeout, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_wait(queue_, 2, &deadline),
            timed_out);
  EXPECT_EQ(state_.wait_count, 1u);

  state_.wait_status = kStatusNoMemory;
  ASSERT_EQ(amdf_wait_deadline_initialize(timeout, 0, &deadline),
            AMDF_STATUS_OK);
  const amdf_status_t rejected = amdf_make_status(
      AMDF_STATUS_DOMAIN_NTSTATUS, static_cast<uint32_t>(kStatusNoMemory));
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_wait(queue_, 1, &deadline),
            rejected);
  EXPECT_EQ(state_.wait_count, 2u);

  state_.wait_status = kStatusSuccess;
  state_.signal_wait = true;
  ASSERT_EQ(amdf_wait_deadline_initialize(timeout, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_wait(queue_, 1, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.wait_count, 3u);

  state_.progress = 2;
  ASSERT_EQ(eventfd_write(state_.deferred_wait_event, 1), 0);
  ASSERT_EQ(amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_wait(queue_, 3, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.progress, 3u);
  EXPECT_EQ(state_.wait_count, 4u);
}

TEST_P(LinuxDxgKernelQueueTest, ResetFenceNeverCompletesAcceptedWork) {
  ASSERT_EQ(Submit(1), AMDF_STATUS_OK);
  state_.progress = UINT64_MAX;
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_progress(queue_), 0u);
  const amdf_status_t lost = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_terminal_status(queue_),
            lost);
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_wait(queue_, 1, &deadline), lost);
  EXPECT_EQ(Submit(2), lost);
  EXPECT_EQ(state_.submitted_values.size(), 1u);
  EXPECT_EQ(state_.wait_count, 0u);
}

TEST_P(LinuxDxgKernelQueueTest, ResetDuringWaitNeverCompletesAcceptedWork) {
  ASSERT_EQ(Submit(1), AMDF_STATUS_OK);
  state_.progress_on_wait_success = UINT64_MAX;
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline),
            AMDF_STATUS_OK);
  const amdf_status_t lost = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_wait(queue_, 1, &deadline), lost);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_progress(queue_), 0u);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_terminal_status(queue_),
            lost);
  EXPECT_EQ(state_.wait_count, 1u);
}

TEST_P(LinuxDxgKernelQueueTest,
       InactivationStopsNativeWorkAndPreservesCheckedProgress) {
  ASSERT_EQ(Submit(1), AMDF_STATUS_OK);
  state_.progress = 1;
  ASSERT_EQ(amdf_gpu_linux_dxg_kernel_queue_inactivate(queue_), AMDF_STATUS_OK);
  EXPECT_EQ(state_.destroy_count, 1u);
  state_.progress = UINT64_MAX;
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_progress(queue_), 1u);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_terminal_status(queue_),
            amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION));
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_inactivate(queue_), AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_destroy(queue_), AMDF_STATUS_OK);
  queue_ = nullptr;
  EXPECT_EQ(state_.destroy_count, 1u);
}

TEST_P(LinuxDxgKernelQueueTest, InactivationPreservesDeviceFailure) {
  amdf_gpu_linux_dxg_device_status_record_reset(&device_.status);
  const amdf_status_t lost = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  ASSERT_EQ(amdf_gpu_linux_dxg_kernel_queue_inactivate(queue_), AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_query_terminal_status(queue_),
            lost);
}

TEST_P(LinuxDxgKernelQueueTest, FailedInactivationNeverRetriesRemoval) {
  state_.destroy_status = kStatusUnsuccessful;
  const amdf_status_t failure = amdf_make_status(
      AMDF_STATUS_DOMAIN_NTSTATUS, static_cast<uint32_t>(kStatusUnsuccessful));
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_inactivate(queue_), failure);
  EXPECT_EQ(state_.destroy_count, 1u);
  state_.destroy_status = kStatusSuccess;
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_inactivate(queue_), failure);
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_destroy(queue_), failure);
  queue_ = nullptr;
  EXPECT_EQ(state_.destroy_count, 1u);
}

TEST_P(LinuxDxgKernelQueueTest, UnsupportedCommandTypePreservesOutput) {
  auto* sentinel = reinterpret_cast<amdf_gpu_umd_kernel_queue_t*>(uintptr_t{1});
  EXPECT_EQ(amdf_gpu_linux_dxg_kernel_queue_create(
                &device_, AMDF_QUEUE_COMMAND_TYPE_XDNA, &sentinel),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  EXPECT_EQ(sentinel,
            reinterpret_cast<amdf_gpu_umd_kernel_queue_t*>(uintptr_t{1}));
}

TEST_P(LinuxDxgKernelQueueTest,
       NativeCreationFailureReleasesColdStateAndPreservesOutput) {
  provider_.gpu_kernel_queue_create = FakeFailCreateQueue;
  auto* sentinel = reinterpret_cast<amdf_gpu_umd_kernel_queue_t*>(uintptr_t{1});
  const amdf_status_t failure = amdf_make_status(
      AMDF_STATUS_DOMAIN_NTSTATUS, static_cast<uint32_t>(kStatusNoMemory));
  EXPECT_EQ(
      amdf_gpu_linux_dxg_kernel_queue_create(&device_, GetParam(), &sentinel),
      failure);
  EXPECT_EQ(sentinel,
            reinterpret_cast<amdf_gpu_umd_kernel_queue_t*>(uintptr_t{1}));
  EXPECT_EQ(state_.live_allocation_count, 1u);
}

INSTANTIATE_TEST_SUITE_P(CommandFormats, LinuxDxgKernelQueueTest,
                         ::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                           AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA));

}  // namespace
