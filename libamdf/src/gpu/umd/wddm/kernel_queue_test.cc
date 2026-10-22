// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/kernel_queue.h"

#include <ntstatus.h>

#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/wddm/device.h"
#include "libamdf/src/kernel_queue_dependency.h"
#include "libamdf/src/wait.h"

namespace {

struct FakeQueueState {
  // Mapped completion storage owned by the native queue dependency.
  volatile uint64_t progress = 0;
  // Exportable progress remains independent of command completion.
  volatile uint64_t shared_progress = 0;
  // Zero selects a software-scheduled native context.
  D3DKMT_HANDLE hardware_queue = 0x40;
  // Explicit shared-fence export failure selected before publication.
  NTSTATUS share_status = STATUS_SUCCESS;
  // Consumer import failure selected before publication.
  NTSTATUS import_status = STATUS_SUCCESS;
  // Result of publishing a dependency wait.
  NTSTATUS dependency_wait_status = STATUS_SUCCESS;
  // Result of publishing the interop retirement signal.
  NTSTATUS shared_signal_status = STATUS_SUCCESS;
  // Native synchronization references released by production cleanup.
  std::vector<D3DKMT_HANDLE> destroyed_fences;
  // Native dependency points accepted before command submission.
  std::vector<uint64_t> dependency_waits;
  // Retirement signals attempted after an accepted native prefix.
  std::vector<uint64_t> shared_signals;
  // Native submission rejection selected by the test.
  NTSTATUS submit_status = STATUS_SUCCESS;
  // Native completion-signal result after command acceptance.
  NTSTATUS signal_status = STATUS_SUCCESS;
  // Native CPU wait registration result selected by the test.
  NTSTATUS wait_status = STATUS_SUCCESS;
  // Native teardown result returned by the bridge dependency.
  NTSTATUS destroy_status = STATUS_SUCCESS;
  // Number of final bridge release calls.
  uint32_t destroy_count = 0;
  // Host queue metadata still owned by the production provider.
  uint32_t live_allocation_count = 0;
  // Total provider allocations, including allocations freed before return.
  uint32_t allocation_count = 0;
  // Command family selected by the parameterized native fixture.
  amdf_wkmi_bridge_gpu_queue_command_type_t command_type =
      AMDF_WKMI_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4;
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
  // Native mapping publication outcome.
  NTSTATUS mapping_status = STATUS_SUCCESS;
  // Native dependency publication outcome after mapping acceptance.
  NTSTATUS ordering_status = STATUS_SUCCESS;
  // Predecessor observed by native mapping publication.
  uint64_t mapping_predecessor = 0;
  // Queue wait point, absent when mapping publication was rejected.
  uint64_t ordering_point = 0;
};

FakeQueueState* current_state = nullptr;

void* AMDF_CALL Allocate(void* user_data, uint64_t byte_length,
                         uint64_t minimum_alignment) {
  const auto allocator = amdf_allocator_system();
  void* pointer =
      allocator.allocate(allocator.user_data, byte_length, minimum_alignment);
  if (pointer != nullptr) {
    ++static_cast<FakeQueueState*>(user_data)->live_allocation_count;
    ++static_cast<FakeQueueState*>(user_data)->allocation_count;
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
  EXPECT_EQ(create_info->command_type, state->command_type);
  *out_info = {};
  out_info->structure_size = sizeof(*out_info);
  out_info->command_buffer_alignment = 4;
  out_info->maximum_command_buffer_byte_length = 4096;
  out_info->progress_fence_handle = 0x20;
  out_info->progress_fence_pointer = &state->progress;
  out_info->progress_fence_device_address = 0x10000;
  out_info->context_handle = 0x30;
  out_info->queue_handle = state->hardware_queue;
  *out_native_status = 0;
  *out_queue = reinterpret_cast<amdf_wkmi_bridge_gpu_kernel_queue_t*>(state);
  return AMDF_WKMI_BRIDGE_RESULT_SUCCESS;
}

amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL FakeSubmitQueue(
    amdf_wkmi_bridge_gpu_kernel_queue_t* queue, uint64_t command_address,
    uint64_t command_byte_length, uint64_t progress_value,
    uint32_t* out_accepted, uint32_t* out_native_status) {
  auto* state = reinterpret_cast<FakeQueueState*>(queue);
  EXPECT_EQ(command_address, 0x20000u);
  EXPECT_EQ(command_byte_length, 64u);
  state->submitted_values.push_back(progress_value);
  *out_accepted = state->submit_status == STATUS_SUCCESS;
  const NTSTATUS status =
      *out_accepted ? state->signal_status : state->submit_status;
  *out_native_status = static_cast<uint32_t>(status);
  return status == STATUS_SUCCESS ? AMDF_WKMI_BRIDGE_RESULT_SUCCESS
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
  EXPECT_EQ(wait->ObjectHandleArray[0], 0x20u);
  EXPECT_EQ(wait->Flags.Value, 0u);
  EXPECT_NE(wait->hAsyncEvent, nullptr);
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

class WindowsGpuKernelQueueTest
    : public ::testing::TestWithParam<amdf_queue_command_type_t> {
 protected:
  void SetUp() override {
    current_state = &state_;
    state_.command_type = GetParam() == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
                              ? AMDF_WKMI_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4
                              : AMDF_WKMI_BRIDGE_GPU_QUEUE_COMMAND_TYPE_SDMA;
    bridge_.gpu_kernel_queue_create = FakeCreateQueue;
    bridge_.gpu_kernel_queue_submit = FakeSubmitQueue;
    bridge_.gpu_kernel_queue_destroy = FakeDestroyQueue;
    kmt_.get_device_state = FakeGetDeviceState;
    kmt_.wait_from_cpu = FakeWaitFromCpu;
    kmt_.update_gpu_virtual_address =
        [](const D3DKMT_UPDATEGPUVIRTUALADDRESS* update) {
          EXPECT_EQ(update->hDevice, 0x10u);
          EXPECT_EQ(update->hContext, 0x30u);
          EXPECT_EQ(update->hFenceObject, 0x20u);
          EXPECT_EQ(update->Flags.DoNotWait, 0u);
          current_state->mapping_predecessor = update->FenceValue;
          return current_state->mapping_status;
        };
    kmt_.wait_on_hardware_queue =
        [](const D3DKMT_SUBMITWAITFORSYNCOBJECTSTOHWQUEUE* wait) {
          EXPECT_EQ(wait->hHwQueue, 0x40u);
          EXPECT_EQ(wait->ObjectCount, 1u);
          EXPECT_EQ(wait->ObjectHandleArray[0], 0x20u);
          current_state->ordering_point = wait->FenceValueArray[0];
          return current_state->ordering_status;
        };
    device_.host_allocator = {&state_, Allocate, nullptr, Free};
    device_.device = 0x10;
    device_.kmt = &kmt_;
    device_.wkmi_adapter.api = &bridge_;
    device_.wkmi_adapter.native =
        reinterpret_cast<amdf_wkmi_bridge_gpu_adapter_t*>(&state_);
    amdf_kmt_device_status_initialize(&device_.status);
    ASSERT_EQ(
        amdf_gpu_umd_kernel_queue_create(&device_, GetParam(), 0, &queue_),
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
    if (event_.payload.native_handle != nullptr) {
      EXPECT_TRUE(CloseHandle(event_.payload.native_handle));
    }
    current_state = nullptr;
  }

  amdf_status_t Submit(uint64_t submission) {
    uint32_t accepted_wait_count = 0;
    return amdf_gpu_umd_kernel_queue_submit(queue_, 0x20000, 64, submission, 0,
                                            nullptr, &accepted_wait_count);
  }

  void EnableInterop(bool hardware_scheduling = true) {
    ASSERT_EQ(amdf_gpu_umd_kernel_queue_destroy(queue_), AMDF_STATUS_OK);
    queue_ = nullptr;
    state_.progress = 0;
    state_.shared_progress = 0;
    state_.submitted_values.clear();
    state_.dependency_waits.clear();
    state_.shared_signals.clear();
    state_.hardware_queue = hardware_scheduling ? 0x40 : 0;
    kmt_.create_synchronization_object =
        [](D3DKMT_CREATESYNCHRONIZATIONOBJECT2* create) -> NTSTATUS {
      EXPECT_EQ(create->Info.Type, D3DDDI_MONITORED_FENCE);
      EXPECT_TRUE(create->Info.Flags.Shared);
      EXPECT_TRUE(create->Info.Flags.NtSecuritySharing);
      create->hSyncObject = 0x50;
      create->Info.MonitoredFence.FenceValueCPUVirtualAddress =
          const_cast<uint64_t*>(&current_state->shared_progress);
      return STATUS_SUCCESS;
    };
    kmt_.destroy_synchronization_object =
        [](const D3DKMT_DESTROYSYNCHRONIZATIONOBJECT* destroy) -> NTSTATUS {
      current_state->destroyed_fences.push_back(destroy->hSyncObject);
      return STATUS_SUCCESS;
    };
    kmt_.share_objects = [](UINT, const D3DKMT_HANDLE*, POBJECT_ATTRIBUTES,
                            DWORD, HANDLE* handle) -> NTSTATUS {
      if (current_state->share_status == STATUS_SUCCESS) {
        *handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        EXPECT_NE(*handle, nullptr);
      }
      return current_state->share_status;
    };
    kmt_.open_synchronization_object =
        [](D3DKMT_OPENSYNCOBJECTFROMNTHANDLE2* open) -> NTSTATUS {
      if (current_state->import_status == STATUS_SUCCESS) {
        open->hSyncObject = 0x60;
      }
      return current_state->import_status;
    };
    kmt_.wait_on_hardware_queue =
        [](const D3DKMT_SUBMITWAITFORSYNCOBJECTSTOHWQUEUE* wait) -> NTSTATUS {
      EXPECT_EQ(wait->hHwQueue, 0x40u);
      if (wait->ObjectHandleArray[0] == 0x20) {
        current_state->ordering_point = wait->FenceValueArray[0];
        return current_state->ordering_status;
      }
      EXPECT_EQ(wait->ObjectHandleArray[0], 0x60u);
      if (current_state->dependency_wait_status == STATUS_SUCCESS) {
        current_state->dependency_waits.push_back(wait->FenceValueArray[0]);
      }
      return current_state->dependency_wait_status;
    };
    kmt_.wait_on_context =
        [](const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU* wait) -> NTSTATUS {
      EXPECT_EQ(wait->hContext, 0x30u);
      EXPECT_EQ(wait->ObjectHandleArray[0], 0x60u);
      if (current_state->dependency_wait_status == STATUS_SUCCESS) {
        current_state->dependency_waits.push_back(
            wait->MonitoredFenceValueArray[0]);
      }
      return current_state->dependency_wait_status;
    };
    kmt_.signal_on_hardware_queue =
        [](const D3DKMT_SUBMITSIGNALSYNCOBJECTSTOHWQUEUE* signal) -> NTSTATUS {
      EXPECT_EQ(signal->BroadcastHwQueueArray[0], 0x40u);
      EXPECT_EQ(signal->ObjectHandleArray[0], 0x50u);
      current_state->shared_signals.push_back(signal->FenceValueArray[0]);
      return current_state->shared_signal_status;
    };
    kmt_.signal_on_context =
        [](const D3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMGPU* signal)
        -> NTSTATUS {
      EXPECT_EQ(signal->hContext, 0x30u);
      EXPECT_EQ(signal->ObjectHandleArray[0], 0x50u);
      current_state->shared_signals.push_back(
          signal->MonitoredFenceValueArray[0]);
      return current_state->shared_signal_status;
    };
    ASSERT_EQ(
        amdf_gpu_umd_kernel_queue_create(
            &device_, GetParam(),
            AMDF_KERNEL_QUEUE_CAPABILITY_EXTERNAL_SYNCHRONIZATION, &queue_),
        AMDF_STATUS_OK);
  }

  void ImportDependency(amdf_kernel_queue_dependency_t* dependency) {
    const auto* progress =
        amdf_gpu_umd_kernel_queue_query_native_progress(queue_);
    ASSERT_NE(progress, nullptr);
    ASSERT_EQ(amdf_gpu_umd_kernel_queue_create_native_dependency(
                  queue_, progress, &dependency->native),
              AMDF_STATUS_OK);
  }

  void CreateNotificationEvent() {
    event_.type = AMDF_NATIVE_EVENT_TYPE_WIN32_EVENT;
    event_.payload.native_handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ASSERT_NE(event_.payload.native_handle, nullptr);
  }

  amdf_status_t Notify(uint64_t submission) {
    return amdf_gpu_umd_kernel_queue_request_notification(queue_, submission,
                                                          &event_);
  }

  // Caller-owned event kept alive through native queue teardown.
  amdf_native_event_t event_ = {};
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

TEST_P(WindowsGpuKernelQueueTest, SharedRetirementCoversNativeDependencyUse) {
  for (const bool hardware_scheduling : {true, false}) {
    ASSERT_NO_FATAL_FAILURE(EnableInterop(hardware_scheduling));
    amdf_kernel_queue_dependency_t dependency = {};
    ASSERT_NO_FATAL_FAILURE(ImportDependency(&dependency));
    const amdf_kernel_queue_dependency_wait_t wait = {&dependency, 9};
    const uint32_t allocations = state_.allocation_count;
    uint32_t accepted_wait_count = 0;
    EXPECT_EQ(amdf_gpu_umd_kernel_queue_submit(queue_, 0x20000, 64, 1, 1, &wait,
                                               &accepted_wait_count),
              AMDF_STATUS_OK);
    EXPECT_EQ(accepted_wait_count, 1u);
    EXPECT_EQ(state_.dependency_waits, (std::vector<uint64_t>{9}));
    EXPECT_EQ(state_.shared_signals, (std::vector<uint64_t>{1}));
    EXPECT_EQ(state_.allocation_count, allocations);
    state_.progress = 1;
    EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_progress(queue_), 0u);
    state_.shared_progress = 1;
    EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_progress(queue_), 1u);
    EXPECT_EQ(dependency.native->destroy(dependency.native), AMDF_STATUS_OK);
  }
}

TEST_P(WindowsGpuKernelQueueTest, NativeFailurePreservesTheAcceptedPrefix) {
  const auto rejected = amdf_kmt_make_status(STATUS_NO_MEMORY);
  for (const bool hardware_scheduling : {true, false}) {
    for (uint32_t failed_operation = 0; failed_operation < 3;
         ++failed_operation) {
      SCOPED_TRACE(failed_operation);
      ASSERT_NO_FATAL_FAILURE(EnableInterop(hardware_scheduling));
      amdf_kernel_queue_dependency_t dependency = {};
      ASSERT_NO_FATAL_FAILURE(ImportDependency(&dependency));
      const amdf_kernel_queue_dependency_wait_t wait = {&dependency, 9};
      state_.dependency_wait_status =
          failed_operation == 0 ? STATUS_NO_MEMORY : STATUS_SUCCESS;
      state_.submit_status =
          failed_operation == 1 ? STATUS_NO_MEMORY : STATUS_SUCCESS;
      state_.shared_signal_status =
          failed_operation == 2 ? STATUS_NO_MEMORY : STATUS_SUCCESS;
      uint32_t accepted_wait_count = UINT32_MAX;
      EXPECT_EQ(amdf_gpu_umd_kernel_queue_submit(queue_, 0x20000, 64, 1, 1,
                                                 &wait, &accepted_wait_count),
                failed_operation == 0 ? rejected : AMDF_STATUS_OK);
      EXPECT_EQ(accepted_wait_count, failed_operation == 0 ? 0u : 1u);
      EXPECT_EQ(state_.submitted_values.size(),
                failed_operation == 0 ? 0u : 1u);
      EXPECT_EQ(state_.shared_signals.size(), failed_operation == 0 ? 0u : 1u);
      EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_terminal_status(queue_),
                failed_operation == 0 ? AMDF_STATUS_OK : rejected);
      // Command completion cannot retire an accepted wait or a failed later
      // signal. Only the separately observed shared fence establishes it.
      state_.progress = 1;
      EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_progress(queue_), 0u);
      if (failed_operation != 0) {
        EXPECT_EQ(Submit(2), rejected);
        EXPECT_EQ(state_.submitted_values.size(), 1u);
      }
      // The mock supplies quiescence before native-view destruction. No reset
      // or publication failure is injected into physical hardware.
      state_.shared_progress = 1;
      EXPECT_EQ(dependency.native->destroy(dependency.native), AMDF_STATUS_OK);
    }
  }
}

TEST_P(WindowsGpuKernelQueueTest, MappingRetirementIncludesSharedSignal) {
  ASSERT_NO_FATAL_FAILURE(EnableInterop());
  D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION operation = {};
  bool mutation_started = false;
  bool accepted = false;
  ASSERT_EQ(amdf_gpu_wddm_kernel_queue_update_memory(
                queue_, 1, &operation, 1, &mutation_started, &accepted),
            AMDF_STATUS_OK);
  EXPECT_TRUE(mutation_started);
  EXPECT_TRUE(accepted);
  EXPECT_EQ(state_.ordering_point, 1u);
  EXPECT_EQ(state_.shared_signals, (std::vector<uint64_t>{1}));
  state_.progress = 1;
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_progress(queue_), 0u);
  state_.shared_progress = 1;
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_progress(queue_), 1u);
}

TEST_P(WindowsGpuKernelQueueTest,
       FailedInteropConstructionRollsBackNativeOwners) {
  ASSERT_NO_FATAL_FAILURE(EnableInterop());
  state_.import_status = STATUS_NO_MEMORY;
  auto* const progress =
      amdf_gpu_umd_kernel_queue_query_native_progress(queue_);
  auto* native = reinterpret_cast<amdf_kernel_queue_native_dependency_t*>(1);
  const uint32_t live_allocations = state_.live_allocation_count;
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_create_native_dependency(queue_, progress,
                                                               &native),
            amdf_kmt_make_status(STATUS_NO_MEMORY));
  EXPECT_EQ(native,
            reinterpret_cast<amdf_kernel_queue_native_dependency_t*>(1));
  EXPECT_EQ(state_.live_allocation_count, live_allocations);

  state_.share_status = STATUS_NO_MEMORY;
  auto* unpublished = reinterpret_cast<amdf_gpu_umd_kernel_queue_t*>(1);
  const uint32_t native_releases = state_.destroy_count;
  EXPECT_EQ(
      amdf_gpu_umd_kernel_queue_create(
          &device_, GetParam(),
          AMDF_KERNEL_QUEUE_CAPABILITY_EXTERNAL_SYNCHRONIZATION, &unpublished),
      amdf_kmt_make_status(STATUS_NO_MEMORY));
  EXPECT_EQ(unpublished, reinterpret_cast<amdf_gpu_umd_kernel_queue_t*>(1));
  EXPECT_EQ(state_.destroyed_fences, (std::vector<D3DKMT_HANDLE>{0x50}));
  EXPECT_EQ(state_.destroy_count, native_releases + 1);
  EXPECT_EQ(state_.live_allocation_count, live_allocations);
}

TEST_P(WindowsGpuKernelQueueTest, NotificationUsesCallerEventAndStableFence) {
  ASSERT_NO_FATAL_FAILURE(CreateNotificationEvent());
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_notification_types(queue_),
            AMDF_NATIVE_EVENT_TYPE_BIT_WIN32_EVENT);
  ASSERT_EQ(Submit(1), AMDF_STATUS_OK);
  EXPECT_EQ(state_.wait_count, 0u);
  const uint32_t allocations = state_.allocation_count;
  EXPECT_EQ(Notify(1), AMDF_STATUS_OK);
  EXPECT_EQ(state_.wait_count, 1u);
  EXPECT_EQ(WaitForSingleObject(event_.payload.native_handle, 0),
            WAIT_OBJECT_0);
  EXPECT_EQ(WaitForSingleObject(event_.payload.native_handle, 0), WAIT_TIMEOUT);
  EXPECT_EQ(state_.allocation_count, allocations);
  EXPECT_EQ(state_.submitted_values, (std::vector<uint64_t>{1}));
  EXPECT_EQ(state_.diagnostic_count, 0u);
}

TEST_P(WindowsGpuKernelQueueTest, CheckedAndCompletedPointsSignalImmediately) {
  ASSERT_NO_FATAL_FAILURE(CreateNotificationEvent());
  ASSERT_EQ(Submit(1), AMDF_STATUS_OK);
  state_.progress = 1;
  EXPECT_EQ(Notify(1), AMDF_STATUS_OK);
  EXPECT_EQ(WaitForSingleObject(event_.payload.native_handle, 0),
            WAIT_OBJECT_0);
  // An earlier checked point remains signalable after a later device reset.
  state_.progress = UINT64_MAX;
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_progress(queue_), 0u);
  EXPECT_EQ(Notify(0), AMDF_STATUS_OK);
  EXPECT_EQ(WaitForSingleObject(event_.payload.native_handle, 0),
            WAIT_OBJECT_0);
  EXPECT_EQ(state_.wait_count, 0u);
}

TEST_P(WindowsGpuKernelQueueTest, RepeatedNotificationsAreIndependentOfWait) {
  ASSERT_NO_FATAL_FAILURE(CreateNotificationEvent());
  ASSERT_EQ(Submit(1), AMDF_STATUS_OK);
  kmt_.wait_from_cpu =
      [](const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU* wait) -> NTSTATUS {
    ++current_state->wait_count;
    EXPECT_EQ(wait->ObjectHandleArray[0], 0x20u);
    EXPECT_EQ(wait->FenceValueArray[0], 1u);
    current_state->deferred_wait_event = wait->hAsyncEvent;
    return STATUS_SUCCESS;
  };
  EXPECT_EQ(Notify(1), AMDF_STATUS_OK);
  EXPECT_EQ(Notify(1), AMDF_STATUS_OK);
  EXPECT_EQ(state_.wait_count, 2u);
  EXPECT_EQ(state_.deferred_wait_event, event_.payload.native_handle);
  EXPECT_EQ(state_.progress, 0u);
  EXPECT_EQ(WaitForSingleObject(event_.payload.native_handle, 0), WAIT_TIMEOUT);

  kmt_.wait_from_cpu =
      [](const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU* wait) -> NTSTATUS {
    EXPECT_NE(wait->hAsyncEvent, current_state->deferred_wait_event);
    return FakeWaitFromCpu(wait);
  };
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_wait(queue_, 1, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.wait_count, 3u);
  // The synchronous wait neither consumes nor cancels the caller's two hints.
  EXPECT_TRUE(SetEvent(state_.deferred_wait_event));
  EXPECT_TRUE(SetEvent(state_.deferred_wait_event));
  EXPECT_EQ(WaitForSingleObject(event_.payload.native_handle, 0),
            WAIT_OBJECT_0);
  EXPECT_EQ(WaitForSingleObject(event_.payload.native_handle, 0), WAIT_TIMEOUT);
}

TEST_P(WindowsGpuKernelQueueTest, NotificationFailurePreservesNativeStatus) {
  ASSERT_NO_FATAL_FAILURE(CreateNotificationEvent());
  ASSERT_EQ(Submit(1), AMDF_STATUS_OK);
  state_.wait_status = STATUS_NO_MEMORY;
  EXPECT_EQ(Notify(1), amdf_kmt_make_status(STATUS_NO_MEMORY));
  EXPECT_EQ(state_.progress, 0u);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_terminal_status(queue_),
            AMDF_STATUS_OK);
  EXPECT_EQ(WaitForSingleObject(event_.payload.native_handle, 0), WAIT_TIMEOUT);
  state_.wait_status = STATUS_SUCCESS;
  EXPECT_EQ(Notify(1), AMDF_STATUS_OK);
  EXPECT_EQ(WaitForSingleObject(event_.payload.native_handle, 0),
            WAIT_OBJECT_0);
  EXPECT_EQ(state_.submitted_values, (std::vector<uint64_t>{1}));
  EXPECT_EQ(state_.diagnostic_count, 1u);
}

TEST_P(WindowsGpuKernelQueueTest, FatalNotificationFailureIsNotCompletion) {
  ASSERT_NO_FATAL_FAILURE(CreateNotificationEvent());
  ASSERT_EQ(Submit(1), AMDF_STATUS_OK);
  state_.wait_status = STATUS_DEVICE_REMOVED;
  const auto lost = amdf_kmt_make_status(STATUS_DEVICE_REMOVED);
  EXPECT_EQ(Notify(1), lost);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_terminal_status(queue_), lost);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_progress(queue_), 0u);
  EXPECT_EQ(Notify(1), lost);
  EXPECT_EQ(state_.wait_count, 1u);
  EXPECT_EQ(WaitForSingleObject(event_.payload.native_handle, 0), WAIT_TIMEOUT);
}

TEST_P(WindowsGpuKernelQueueTest, ResetWakeRequiresIndependentStatusCheck) {
  ASSERT_NO_FATAL_FAILURE(CreateNotificationEvent());
  ASSERT_EQ(Submit(1), AMDF_STATUS_OK);
  state_.progress_on_wait_success = UINT64_MAX;
  EXPECT_EQ(Notify(1), AMDF_STATUS_OK);
  EXPECT_EQ(WaitForSingleObject(event_.payload.native_handle, 0),
            WAIT_OBJECT_0);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_progress(queue_), 0u);
  const auto lost = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_terminal_status(queue_), lost);
  EXPECT_EQ(Notify(1), lost);
  EXPECT_EQ(state_.wait_count, 1u);
  EXPECT_EQ(WaitForSingleObject(event_.payload.native_handle, 0), WAIT_TIMEOUT);
}

TEST_P(WindowsGpuKernelQueueTest,
       MappingWaitUsesTheSameProgressWithoutCpuWait) {
  D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION operation = {};
  operation.OperationType = D3DDDI_UPDATEGPUVIRTUALADDRESS_UNMAP;
  bool mutated = false;
  bool accepted = false;
  EXPECT_EQ(amdf_gpu_wddm_kernel_queue_update_memory(queue_, 1, &operation, 19,
                                                     &mutated, &accepted),
            AMDF_STATUS_OK);
  EXPECT_TRUE(mutated);
  EXPECT_TRUE(accepted);
  EXPECT_EQ(state_.mapping_predecessor, 18u);
  EXPECT_EQ(state_.ordering_point, 19u);
  EXPECT_EQ(state_.wait_count, 0u);
  EXPECT_EQ(state_.progress, 0u);
}

TEST_P(WindowsGpuKernelQueueTest,
       RejectedMappingDoesNotEnqueueAnUnresolvableWait) {
  state_.mapping_status = STATUS_NO_MEMORY;
  D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION operation = {};
  bool mutated = false;
  bool accepted = false;
  EXPECT_EQ(amdf_gpu_wddm_kernel_queue_update_memory(queue_, 1, &operation, 19,
                                                     &mutated, &accepted),
            amdf_kmt_make_status(STATUS_NO_MEMORY));
  EXPECT_TRUE(mutated);
  EXPECT_FALSE(accepted);
  EXPECT_EQ(state_.ordering_point, 0u);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_terminal_status(queue_),
            AMDF_STATUS_OK);
}

TEST_P(WindowsGpuKernelQueueTest,
       FailedDependencyPreservesAcceptanceAndFailsQueue) {
  state_.ordering_status = STATUS_NO_MEMORY;
  D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION operation = {};
  bool mutated = false;
  bool accepted = false;
  const auto failure = amdf_kmt_make_status(STATUS_NO_MEMORY);
  EXPECT_EQ(amdf_gpu_wddm_kernel_queue_update_memory(queue_, 1, &operation, 19,
                                                     &mutated, &accepted),
            failure);
  EXPECT_TRUE(mutated);
  EXPECT_TRUE(accepted);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_terminal_status(queue_), failure);
  EXPECT_EQ(Submit(20), failure);
  EXPECT_TRUE(state_.submitted_values.empty());
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_progress(queue_), 0u);
  state_.progress = 19;
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_progress(queue_), 19u);
}

TEST_P(WindowsGpuKernelQueueTest, BridgeFailureConsumesProviderOwner) {
  state_.destroy_status = STATUS_DEVICE_BUSY;
  ASSERT_EQ(state_.live_allocation_count, 1u);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_destroy(queue_),
            amdf_kmt_make_status(STATUS_DEVICE_BUSY));
  queue_ = nullptr;
  EXPECT_EQ(state_.destroy_count, 1u);
  EXPECT_EQ(state_.live_allocation_count, 0u);
}

TEST_P(WindowsGpuKernelQueueTest,
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

TEST_P(WindowsGpuKernelQueueTest, RejectedCompletionPointCanBeSubmittedAgain) {
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

TEST_P(WindowsGpuKernelQueueTest, FatalRejectionStopsAllQueuesOnDevice) {
  amdf_gpu_umd_kernel_queue_t* sibling = nullptr;
  ASSERT_EQ(amdf_gpu_umd_kernel_queue_create(&device_, GetParam(), 0, &sibling),
            AMDF_STATUS_OK);
  state_.submit_status = STATUS_DEVICE_REMOVED;
  constexpr uint64_t submission = 1;
  const amdf_status_t lost = amdf_kmt_make_status(STATUS_DEVICE_REMOVED);
  EXPECT_EQ(Submit(submission), lost);
  state_.submit_status = STATUS_SUCCESS;
  EXPECT_EQ(Submit(submission), lost);
  uint32_t accepted_wait_count = 0;
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_submit(sibling, 0x20000, 64, submission,
                                             0, nullptr, &accepted_wait_count),
            lost);
  EXPECT_EQ(state_.submitted_values.size(), 1u);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_terminal_status(sibling), lost);
  EXPECT_EQ(state_.diagnostic_count, 0u);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_destroy(sibling), AMDF_STATUS_OK);
}

TEST_P(WindowsGpuKernelQueueTest,
       FailedSignalPreservesAcceptanceAndStopsPublication) {
  state_.signal_status = STATUS_NO_MEMORY;
  const auto failure = amdf_kmt_make_status(STATUS_NO_MEMORY);
  EXPECT_EQ(Submit(1), AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_terminal_status(queue_), failure);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_progress(queue_), 0u);
  EXPECT_EQ(Submit(1), failure);
  EXPECT_EQ(Submit(2), failure);
  EXPECT_EQ(state_.submitted_values, (std::vector<uint64_t>{1}));
  CreateNotificationEvent();
  EXPECT_EQ(Notify(1), failure);
  EXPECT_EQ(state_.wait_count, 0u);
  EXPECT_EQ(state_.allocation_count, 1u);
  // Only independently supplied native progress proves final access. Cached
  // failure remains visible after that point; it is never treated as success.
  state_.progress = 1;
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_progress(queue_), 1u);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_query_terminal_status(queue_), failure);
}

TEST_P(WindowsGpuKernelQueueTest, SoftwareContextOmitsHardwareMappingOrdering) {
  EXPECT_GT(amdf_gpu_umd_kernel_queue_mapping_capacity(queue_), 0u);
  auto create = bridge_.gpu_kernel_queue_create;
  bridge_.gpu_kernel_queue_create =
      [](amdf_wkmi_bridge_gpu_adapter_t* adapter,
         const amdf_wkmi_bridge_gpu_kernel_queue_create_info_t* create_info,
         amdf_wkmi_bridge_gpu_kernel_queue_t** out_queue,
         amdf_wkmi_bridge_gpu_kernel_queue_info_t* out_info,
         uint32_t* out_native_status) {
        const auto result = FakeCreateQueue(adapter, create_info, out_queue,
                                            out_info, out_native_status);
        out_info->queue_handle = 0;
        return result;
      };
  amdf_gpu_umd_kernel_queue_t* software_queue = nullptr;
  ASSERT_EQ(amdf_gpu_umd_kernel_queue_create(&device_, GetParam(), 0,
                                             &software_queue),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_mapping_capacity(software_queue), 0u);
  EXPECT_EQ(amdf_gpu_umd_kernel_queue_destroy(software_queue), AMDF_STATUS_OK);
  bridge_.gpu_kernel_queue_create = create;
}

TEST_P(WindowsGpuKernelQueueTest, RecoverableWaitErrorDoesNotReplayCommand) {
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

TEST_P(WindowsGpuKernelQueueTest,
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

TEST_P(WindowsGpuKernelQueueTest, TimeoutDoesNotRetireOrFailAcceptedWork) {
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

TEST_P(WindowsGpuKernelQueueTest,
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

TEST_P(WindowsGpuKernelQueueTest, ResetFenceDoesNotCompleteAcceptedWork) {
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

TEST_P(WindowsGpuKernelQueueTest, ResetDuringWaitDoesNotCompleteAcceptedWork) {
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

INSTANTIATE_TEST_SUITE_P(CommandFormats, WindowsGpuKernelQueueTest,
                         ::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                           AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA));

}  // namespace
