// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/provider/kernel_queue.h"

#include <malloc.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif  // WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX
#define WIN32_NO_STATUS
#include <windows.h>
#include <winternl.h>
#undef WIN32_NO_STATUS

#include <d3dkmthk.h>
#include <ntstatus.h>

#include "libamdf/src/gpu/umd/dxg/private_records.h"
#include "libamdf/src/gpu/umd/wddm/provider/adapter_state.h"

namespace {

constexpr D3DKMT_HANDLE kDevice = 0x10;
constexpr D3DKMT_HANDLE kContext = 0x20;
constexpr D3DKMT_HANDLE kQueue = 0x30;
constexpr D3DKMT_HANDLE kFence = 0x40;

struct FakeNativeState {
  // Scheduler profile installed in the adapter for the selected command family.
  amdf_dxg_private_queue_profile_t profile = {
      .scheduler_id = 5,
      .node_ordinal = 0,
      .is_supported = 1,
      .uses_hardware_scheduling = 1,
  };
  // Whether private records enable firmware state shadowing.
  uint32_t supports_firmware_state_shadowing = 0;
  // Native result selected for context construction.
  NTSTATUS create_context_status = STATUS_SUCCESS;
  // Native result selected for queue construction.
  NTSTATUS create_queue_status = STATUS_SUCCESS;
  // Native result selected for software-queue fence construction.
  NTSTATUS create_fence_status = STATUS_SUCCESS;
  // Native command submission result, before completion-signal publication.
  NTSTATUS submit_status = STATUS_SUCCESS;
  // Native completion-signal result after SWS command acceptance.
  NTSTATUS signal_status = STATUS_SUCCESS;
  // Native result selected for queue destruction.
  NTSTATUS destroy_queue_status = STATUS_SUCCESS;
  // Native result selected for context destruction.
  NTSTATUS destroy_context_status = STATUS_SUCCESS;
  // Native result selected for separately owned fence destruction.
  NTSTATUS destroy_fence_status = STATUS_SUCCESS;
  // Fence handle returned on successful queue construction.
  D3DKMT_HANDLE fence = kFence;
  // Native-owned progress mapping, independent of provider host allocations.
  uint64_t progress = 0;
  // Number of native context creation calls.
  uint32_t context_create_count = 0;
  // Number of native hardware queue creation calls.
  uint32_t queue_create_count = 0;
  // Number of native monitored fence creation calls.
  uint32_t fence_create_count = 0;
  // Number of native queue destruction calls.
  uint32_t queue_destroy_count = 0;
  // Number of native context destruction calls.
  uint32_t context_destroy_count = 0;
  // Number of separately owned native fence destruction calls.
  uint32_t fence_destroy_count = 0;
  // Number of submitted commands.
  uint32_t submit_count = 0;
  // Number of separate SWS completion-signal publication calls.
  uint32_t signal_count = 0;
  // Host allocation ordinal rejected, or SIZE_MAX for no injected failure.
  size_t failure_ordinal = SIZE_MAX;
  // Number of attempted host allocations.
  size_t allocation_count = 0;
  // Number of host allocations still owned by the provider.
  size_t live_allocation_count = 0;
  // Number of requests violating the public allocator alignment contract.
  size_t invalid_alignment_count = 0;
  // Number of contract violations observed by native dependencies.
  uint32_t dependency_failure_count = 0;
};

FakeNativeState* current_state = nullptr;

void ExpectDependency(bool condition) {
  if (!condition) {
    ++current_state->dependency_failure_count;
  }
}

uint32_t Load32(const void* pointer) {
  uint32_t value = 0;
  std::memcpy(&value, pointer, sizeof(value));
  return value;
}

uint64_t Load64(const void* pointer) {
  uint64_t value = 0;
  std::memcpy(&value, pointer, sizeof(value));
  return value;
}

void ExpectSubmissionRecord(const void* record, uint32_t byte_length) {
  ExpectDependency(byte_length ==
                   AMDF_DXG_PRIVATE_SUBMISSION_RECORD_BYTE_LENGTH);
  const auto* bytes = static_cast<const uint8_t*>(record);
  ExpectDependency(Load64(bytes + 0x00) == UINT64_C(0x100000006));
  ExpectDependency(Load64(bytes + 0x08) == (UINT64_C(1) << 38));
  ExpectDependency(Load32(bytes + 0x10) == 64);
  ExpectDependency(bytes[0x14] == 4 && bytes[0x1C] == 1);
  ExpectDependency(Load64(bytes + 0x20) == 0x2000);
}

void* AMDF_CALL Allocate(void* user_data, uint64_t byte_length,
                         uint64_t minimum_alignment) {
  auto* state = static_cast<FakeNativeState*>(user_data);
  if (state->allocation_count++ == state->failure_ordinal) {
    return nullptr;
  }
  if (minimum_alignment < amdf_max_align_t ||
      (minimum_alignment & (minimum_alignment - 1)) != 0) {
    ++state->invalid_alignment_count;
  }
  void* pointer = _aligned_malloc(static_cast<size_t>(byte_length),
                                  static_cast<size_t>(minimum_alignment));
  if (pointer != nullptr) {
    ++state->live_allocation_count;
  }
  return pointer;
}

void AMDF_CALL Free(void* user_data, void* allocation) {
  if (allocation == nullptr) {
    return;
  }
  auto* state = static_cast<FakeNativeState*>(user_data);
  --state->live_allocation_count;
  _aligned_free(allocation);
}

void InitializeAdapter(amdf_dxg_provider_gpu_adapter_t* adapter,
                       FakeNativeState* state) {
  adapter->host_allocator = {state, Allocate, nullptr, Free};
  adapter->private_info.supports_firmware_state_shadowing =
      state->supports_firmware_state_shadowing;
  adapter->pm4_queue_profile = state->profile;
  adapter->sdma_queue_profile = state->profile;
}

amdf_dxg_provider_gpu_kernel_queue_create_info_t MakeCreateInfo(
    amdf_dxg_provider_gpu_queue_command_type_t command_type =
        AMDF_DXG_PROVIDER_GPU_QUEUE_COMMAND_TYPE_PM4) {
  amdf_dxg_provider_gpu_kernel_queue_create_info_t info = {};
  info.device_handle = kDevice;
  info.command_type = command_type;
  return info;
}

bool Check(bool condition, const char* expression, int line) {
  if (!condition) {
    std::fprintf(stderr, "kernel_queue_test.cc:%d: check failed: %s\n", line,
                 expression);
  }
  return condition;
}

#define AMDF_EXPECT(expression) \
  passed &= Check(static_cast<bool>(expression), #expression, __LINE__)

}  // namespace

extern "C" NTSTATUS APIENTRY
D3DKMTCreateContextVirtual(D3DKMT_CREATECONTEXTVIRTUAL* create) {
  ++current_state->context_create_count;
  ExpectDependency(create->hDevice == kDevice);
  ExpectDependency(create->NodeOrdinal == current_state->profile.node_ordinal);
  ExpectDependency(create->Flags.HwQueueSupported ==
                   current_state->profile.uses_hardware_scheduling);
  ExpectDependency(create->pPrivateDriverData != nullptr);
  ExpectDependency(create->PrivateDriverDataSize ==
                   AMDF_DXG_PRIVATE_CONTEXT_RECORD_BYTE_LENGTH);
  const auto* record = static_cast<const uint8_t*>(create->pPrivateDriverData);
  ExpectDependency(Load64(record + 0x00) == UINT64_C(0x0010010000000040));
  ExpectDependency(Load32(record + 0x38) ==
                   (current_state->profile.scheduler_id |
                    (current_state->supports_firmware_state_shadowing << 21)));
  ExpectDependency(Load32(record + 0x40) == 0x168);
  ExpectDependency(Load64(record + 0x150) == 0);
  if (current_state->create_context_status == STATUS_SUCCESS) {
    create->hContext = kContext;
  }
  return current_state->create_context_status;
}

extern "C" NTSTATUS APIENTRY D3DKMTCreateHwQueue(D3DKMT_CREATEHWQUEUE* create) {
  ++current_state->queue_create_count;
  ExpectDependency(create->hHwContext == kContext);
  ExpectDependency(create->Flags.DisableGpuTimeout == 0);
  ExpectDependency(create->pPrivateDriverData != nullptr);
  ExpectDependency(create->PrivateDriverDataSize ==
                   AMDF_DXG_PRIVATE_HARDWARE_QUEUE_RECORD_BYTE_LENGTH);
  const auto* record = static_cast<const uint8_t*>(create->pPrivateDriverData);
  ExpectDependency(Load32(record + 0x00) ==
                   AMDF_DXG_PRIVATE_HARDWARE_QUEUE_RECORD_BYTE_LENGTH);
  ExpectDependency(
      Load32(record + 0x10) ==
      ((1u << 12) | (current_state->supports_firmware_state_shadowing << 17)));
  ExpectDependency(Load32(record + 0x30) == 0);
  if (current_state->create_queue_status == STATUS_SUCCESS) {
    create->hHwQueue = kQueue;
    create->hHwQueueProgressFence = current_state->fence;
    create->HwQueueProgressFenceCPUVirtualAddress = &current_state->progress;
    create->HwQueueProgressFenceGPUVirtualAddress = 0x1000;
  }
  return current_state->create_queue_status;
}

extern "C" NTSTATUS APIENTRY
D3DKMTDestroyHwQueue(const D3DKMT_DESTROYHWQUEUE* destroy) {
  ++current_state->queue_destroy_count;
  ExpectDependency(destroy->hHwQueue == kQueue);
  return current_state->destroy_queue_status;
}

extern "C" NTSTATUS APIENTRY
D3DKMTDestroyContext(const D3DKMT_DESTROYCONTEXT* destroy) {
  ++current_state->context_destroy_count;
  ExpectDependency(destroy->hContext == kContext);
  return current_state->destroy_context_status;
}

extern "C" NTSTATUS APIENTRY D3DKMTCreateSynchronizationObject2(
    D3DKMT_CREATESYNCHRONIZATIONOBJECT2* create) {
  ++current_state->fence_create_count;
  ExpectDependency(create->hDevice == kDevice);
  ExpectDependency(create->Info.Type == D3DDDI_MONITORED_FENCE);
  ExpectDependency(create->Info.MonitoredFence.InitialFenceValue == 0);
  ExpectDependency(create->Info.MonitoredFence.EngineAffinity == 1);
  if (current_state->create_fence_status == STATUS_SUCCESS) {
    create->hSyncObject = current_state->fence;
    create->Info.MonitoredFence.FenceValueCPUVirtualAddress =
        &current_state->progress;
    create->Info.MonitoredFence.FenceValueGPUVirtualAddress = 0x1000;
  }
  return current_state->create_fence_status;
}

extern "C" NTSTATUS APIENTRY D3DKMTDestroySynchronizationObject(
    const D3DKMT_DESTROYSYNCHRONIZATIONOBJECT* destroy) {
  ++current_state->fence_destroy_count;
  ExpectDependency(destroy->hSyncObject == kFence);
  return current_state->destroy_fence_status;
}

extern "C" NTSTATUS APIENTRY
D3DKMTSubmitCommand(const D3DKMT_SUBMITCOMMAND* submit) {
  ++current_state->submit_count;
  ExpectDependency(submit->Commands == 0x2000 && submit->CommandLength == 64);
  ExpectDependency(submit->BroadcastContextCount == 1);
  ExpectDependency(submit->BroadcastContext[0] == kContext);
  ExpectDependency(submit->pPrivateDriverData != nullptr);
  ExpectSubmissionRecord(submit->pPrivateDriverData,
                         submit->PrivateDriverDataSize);
  return current_state->submit_status;
}

extern "C" NTSTATUS APIENTRY D3DKMTSignalSynchronizationObjectFromGpu(
    const D3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMGPU* signal) {
  ++current_state->signal_count;
  ExpectDependency(signal->hContext == kContext);
  ExpectDependency(signal->ObjectCount == 1);
  ExpectDependency(signal->ObjectHandleArray[0] == kFence);
  ExpectDependency(signal->MonitoredFenceValueArray[0] == 7);
  if (current_state->signal_status == STATUS_SUCCESS) {
    current_state->progress = signal->MonitoredFenceValueArray[0];
  }
  return current_state->signal_status;
}

extern "C" NTSTATUS APIENTRY
D3DKMTSubmitCommandToHwQueue(const D3DKMT_SUBMITCOMMANDTOHWQUEUE* submit) {
  ++current_state->submit_count;
  ExpectDependency(submit->hHwQueue == kQueue);
  ExpectDependency(submit->HwQueueProgressFenceId == 7);
  ExpectDependency(submit->CommandBuffer == 0x2000);
  ExpectDependency(submit->CommandLength == 64);
  ExpectDependency(submit->pPrivateDriverData != nullptr);
  ExpectSubmissionRecord(submit->pPrivateDriverData,
                         submit->PrivateDriverDataSize);
  if (current_state->submit_status == STATUS_SUCCESS) {
    current_state->progress = submit->HwQueueProgressFenceId;
  }
  return current_state->submit_status;
}

namespace {

bool UnsupportedProfileHasNoOwnershipEffects() {
  bool passed = true;
  FakeNativeState state;
  state.profile.is_supported = 0;
  current_state = &state;
  amdf_dxg_provider_gpu_adapter_t adapter;
  InitializeAdapter(&adapter, &state);
  const auto create = MakeCreateInfo();
  auto* queue = reinterpret_cast<amdf_dxg_provider_gpu_kernel_queue_t*>(1);
  amdf_dxg_provider_gpu_kernel_queue_info_t info;
  std::memset(&info, 0xA5, sizeof(info));
  const auto original_info = info;
  uint32_t native_status = 0xA5;
  AMDF_EXPECT(amdf::wddm_provider::GpuKernelQueueCreate(
                  &adapter, &create, &queue, &info, &native_status) ==
              AMDF_DXG_PROVIDER_RESULT_UNSUPPORTED);
  AMDF_EXPECT(reinterpret_cast<uintptr_t>(queue) == 1);
  AMDF_EXPECT(std::memcmp(&info, &original_info, sizeof(info)) == 0);
  AMDF_EXPECT(native_status == 0);
  AMDF_EXPECT(state.allocation_count == 0);
  AMDF_EXPECT(state.context_create_count == 0);
  AMDF_EXPECT(adapter.live_queue_count.load() == 0);
  current_state = nullptr;
  return passed;
}

bool HostAllocationFailureLeavesNoNativeOwnership() {
  bool passed = true;
  FakeNativeState state;
  state.failure_ordinal = 0;
  current_state = &state;
  amdf_dxg_provider_gpu_adapter_t adapter;
  InitializeAdapter(&adapter, &state);
  const auto create = MakeCreateInfo();
  auto* queue = reinterpret_cast<amdf_dxg_provider_gpu_kernel_queue_t*>(1);
  amdf_dxg_provider_gpu_kernel_queue_info_t info;
  std::memset(&info, 0xA5, sizeof(info));
  const auto original_info = info;
  uint32_t native_status = 1;
  AMDF_EXPECT(amdf::wddm_provider::GpuKernelQueueCreate(
                  &adapter, &create, &queue, &info, &native_status) ==
              AMDF_DXG_PROVIDER_RESULT_RESOURCE_EXHAUSTED);
  AMDF_EXPECT(reinterpret_cast<uintptr_t>(queue) == 1);
  AMDF_EXPECT(std::memcmp(&info, &original_info, sizeof(info)) == 0);
  AMDF_EXPECT(native_status == 0);
  AMDF_EXPECT(state.allocation_count == 1);
  AMDF_EXPECT(state.live_allocation_count == 0);
  AMDF_EXPECT(state.context_create_count == 0);
  AMDF_EXPECT(adapter.live_queue_count.load() == 0);
  current_state = nullptr;
  return passed;
}

bool NativeFailureRollsBackEveryPublishedHandle() {
  bool passed = true;
  struct FailureCase {
    NTSTATUS create_context;
    NTSTATUS create_queue;
    D3DKMT_HANDLE progress_fence;
    NTSTATUS destroy_queue;
    NTSTATUS destroy_context;
    NTSTATUS expected_status;
    uint32_t queue_destroy_count;
    uint32_t context_destroy_count;
  };
  const FailureCase cases[] = {
      {STATUS_NO_MEMORY, STATUS_SUCCESS, kFence, STATUS_SUCCESS, STATUS_SUCCESS,
       STATUS_NO_MEMORY, 0, 0},
      {STATUS_SUCCESS, STATUS_NO_MEMORY, kFence, STATUS_SUCCESS, STATUS_SUCCESS,
       STATUS_NO_MEMORY, 0, 1},
      {STATUS_SUCCESS, STATUS_SUCCESS, 0, STATUS_SUCCESS, STATUS_SUCCESS,
       STATUS_INVALID_HANDLE, 1, 1},
      {STATUS_SUCCESS, STATUS_SUCCESS, 0, STATUS_DEVICE_BUSY, STATUS_SUCCESS,
       STATUS_DEVICE_BUSY, 1, 1},
      {STATUS_SUCCESS, STATUS_SUCCESS, 0, STATUS_SUCCESS, STATUS_DEVICE_BUSY,
       STATUS_DEVICE_BUSY, 1, 1},
  };
  for (const FailureCase& failure : cases) {
    FakeNativeState state;
    state.create_context_status = failure.create_context;
    state.create_queue_status = failure.create_queue;
    state.fence = failure.progress_fence;
    state.destroy_queue_status = failure.destroy_queue;
    state.destroy_context_status = failure.destroy_context;
    current_state = &state;
    amdf_dxg_provider_gpu_adapter_t adapter;
    InitializeAdapter(&adapter, &state);
    const auto create = MakeCreateInfo();
    auto* queue = reinterpret_cast<amdf_dxg_provider_gpu_kernel_queue_t*>(1);
    amdf_dxg_provider_gpu_kernel_queue_info_t info;
    std::memset(&info, 0xA5, sizeof(info));
    const auto original_info = info;
    uint32_t native_status = 0;
    AMDF_EXPECT(amdf::wddm_provider::GpuKernelQueueCreate(
                    &adapter, &create, &queue, &info, &native_status) ==
                AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE);
    AMDF_EXPECT(native_status ==
                static_cast<uint32_t>(failure.expected_status));
    AMDF_EXPECT(reinterpret_cast<uintptr_t>(queue) == 1);
    AMDF_EXPECT(std::memcmp(&info, &original_info, sizeof(info)) == 0);
    AMDF_EXPECT(state.live_allocation_count == 0);
    AMDF_EXPECT(adapter.live_queue_count.load() == 0);
    AMDF_EXPECT(state.queue_destroy_count == failure.queue_destroy_count);
    AMDF_EXPECT(state.context_destroy_count == failure.context_destroy_count);
    AMDF_EXPECT(state.submit_count == 0);
    AMDF_EXPECT(state.dependency_failure_count == 0);
  }
  current_state = nullptr;
  return passed;
}

bool HardwareQueueUsesSharedRecordsAndConsumesEveryResult() {
  bool passed = true;
  for (uint32_t failure_stage = 0; failure_stage < 3; ++failure_stage) {
    FakeNativeState state;
    state.supports_firmware_state_shadowing = 1;
    current_state = &state;
    amdf_dxg_provider_gpu_adapter_t adapter;
    InitializeAdapter(&adapter, &state);
    const auto create = MakeCreateInfo();
    amdf_dxg_provider_gpu_kernel_queue_t* queue = nullptr;
    amdf_dxg_provider_gpu_kernel_queue_info_t info = {};
    uint32_t native_status = 0;
    AMDF_EXPECT(amdf::wddm_provider::GpuKernelQueueCreate(
                    &adapter, &create, &queue, &info, &native_status) ==
                AMDF_DXG_PROVIDER_RESULT_SUCCESS);
    if (queue == nullptr) {
      return false;
    }
    AMDF_EXPECT(info.context_handle == kContext);
    AMDF_EXPECT(info.queue_handle == kQueue);
    AMDF_EXPECT(info.progress_fence_handle == kFence);
    AMDF_EXPECT(info.progress_fence_pointer == &state.progress);
    AMDF_EXPECT(info.progress_fence_device_address == 0x1000);
    AMDF_EXPECT(state.allocation_count == 1);
    AMDF_EXPECT(state.live_allocation_count == 1);
    AMDF_EXPECT(adapter.live_queue_count.load() == 1);
    AMDF_EXPECT(amdf::wddm_provider::PrepareGpuAdapterClose(
                    &adapter, &native_status) == AMDF_DXG_PROVIDER_RESULT_BUSY);
    uint32_t accepted = 0;
    AMDF_EXPECT(amdf::wddm_provider::GpuKernelQueueSubmit(
                    queue, 0x2000, 64, 7, &accepted, &native_status) ==
                AMDF_DXG_PROVIDER_RESULT_SUCCESS);
    AMDF_EXPECT(accepted == 1);
    AMDF_EXPECT(*info.progress_fence_pointer == 7);
    if (failure_stage == 1) {
      state.destroy_queue_status = STATUS_DEVICE_BUSY;
    } else if (failure_stage == 2) {
      state.destroy_context_status = STATUS_DEVICE_BUSY;
    }
    AMDF_EXPECT(
        amdf::wddm_provider::GpuKernelQueueDestroy(queue, &native_status) ==
        (failure_stage == 0 ? AMDF_DXG_PROVIDER_RESULT_SUCCESS
                            : AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE));
    AMDF_EXPECT(native_status ==
                static_cast<uint32_t>(failure_stage == 0 ? STATUS_SUCCESS
                                                         : STATUS_DEVICE_BUSY));
    AMDF_EXPECT(state.live_allocation_count == 0);
    AMDF_EXPECT(adapter.live_queue_count.load() == 0);
    AMDF_EXPECT(state.queue_create_count == 1);
    AMDF_EXPECT(state.queue_destroy_count == 1);
    AMDF_EXPECT(state.context_destroy_count == 1);
    AMDF_EXPECT(state.submit_count == 1);
    AMDF_EXPECT(state.dependency_failure_count == 0);
  }
  current_state = nullptr;
  return passed;
}

bool SoftwareSubmissionSeparatesAcceptanceFromSignal() {
  bool passed = true;
  for (uint32_t failure_stage = 0; failure_stage < 3; ++failure_stage) {
    FakeNativeState state;
    state.profile = {
        .scheduler_id = 7,
        .node_ordinal = 1,
        .is_supported = 1,
        .uses_hardware_scheduling = 0,
    };
    state.submit_status =
        failure_stage == 1 ? STATUS_NO_MEMORY : STATUS_SUCCESS;
    state.signal_status =
        failure_stage == 2 ? STATUS_NO_MEMORY : STATUS_SUCCESS;
    current_state = &state;
    amdf_dxg_provider_gpu_adapter_t adapter;
    InitializeAdapter(&adapter, &state);
    const auto create =
        MakeCreateInfo(AMDF_DXG_PROVIDER_GPU_QUEUE_COMMAND_TYPE_SDMA);
    amdf_dxg_provider_gpu_kernel_queue_t* queue = nullptr;
    amdf_dxg_provider_gpu_kernel_queue_info_t info = {};
    uint32_t native_status = 0;
    AMDF_EXPECT(amdf::wddm_provider::GpuKernelQueueCreate(
                    &adapter, &create, &queue, &info, &native_status) ==
                AMDF_DXG_PROVIDER_RESULT_SUCCESS);
    if (queue == nullptr) {
      return false;
    }
    AMDF_EXPECT(info.queue_handle == 0 && info.context_handle == kContext);
    AMDF_EXPECT(info.progress_fence_handle == kFence);
    AMDF_EXPECT(info.progress_fence_pointer == &state.progress);
    AMDF_EXPECT(state.queue_create_count == 0);
    AMDF_EXPECT(state.fence_create_count == 1);
    AMDF_EXPECT(state.allocation_count == 1);
    const size_t allocation_count = state.allocation_count;
    uint32_t accepted = UINT32_MAX;
    AMDF_EXPECT(amdf::wddm_provider::GpuKernelQueueSubmit(
                    queue, 0x2000, 64, 7, &accepted, &native_status) ==
                (failure_stage ? AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE
                               : AMDF_DXG_PROVIDER_RESULT_SUCCESS));
    AMDF_EXPECT(accepted == (failure_stage == 1 ? 0u : 1u));
    AMDF_EXPECT(native_status == static_cast<uint32_t>(failure_stage
                                                           ? STATUS_NO_MEMORY
                                                           : STATUS_SUCCESS));
    AMDF_EXPECT(state.submit_count == 1);
    AMDF_EXPECT(state.signal_count == (failure_stage == 1 ? 0u : 1u));
    AMDF_EXPECT(state.progress == (failure_stage == 0 ? 7u : 0u));
    AMDF_EXPECT(state.allocation_count == allocation_count);
    // The dependency supplies final access independently of signal failure.
    state.progress = 7;
    AMDF_EXPECT(amdf::wddm_provider::GpuKernelQueueDestroy(
                    queue, &native_status) == AMDF_DXG_PROVIDER_RESULT_SUCCESS);
    AMDF_EXPECT(state.queue_destroy_count == 0);
    AMDF_EXPECT(state.context_destroy_count == 1);
    AMDF_EXPECT(state.fence_destroy_count == 1);
    AMDF_EXPECT(state.live_allocation_count == 0);
    AMDF_EXPECT(adapter.live_queue_count.load() == 0);
    AMDF_EXPECT(state.dependency_failure_count == 0);
  }
  current_state = nullptr;
  return passed;
}

bool SoftwareConstructionAndReleasePreserveOwnership() {
  bool passed = true;
  for (uint32_t failure_stage = 0; failure_stage < 5; ++failure_stage) {
    FakeNativeState state;
    state.profile = {
        .scheduler_id = 7,
        .node_ordinal = 1,
        .is_supported = 1,
        .uses_hardware_scheduling = 0,
    };
    if (failure_stage == 0) {
      state.create_fence_status = STATUS_NO_MEMORY;
    } else if (failure_stage == 1) {
      state.fence = 0;
    }
    current_state = &state;
    amdf_dxg_provider_gpu_adapter_t adapter;
    InitializeAdapter(&adapter, &state);
    const auto create =
        MakeCreateInfo(AMDF_DXG_PROVIDER_GPU_QUEUE_COMMAND_TYPE_SDMA);
    auto* queue = reinterpret_cast<amdf_dxg_provider_gpu_kernel_queue_t*>(1);
    amdf_dxg_provider_gpu_kernel_queue_info_t info;
    std::memset(&info, 0xA5, sizeof(info));
    const auto original_info = info;
    uint32_t native_status = 0;
    const auto result = amdf::wddm_provider::GpuKernelQueueCreate(
        &adapter, &create, &queue, &info, &native_status);
    if (failure_stage < 2) {
      AMDF_EXPECT(result == AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE);
      AMDF_EXPECT(reinterpret_cast<uintptr_t>(queue) == 1);
      AMDF_EXPECT(std::memcmp(&info, &original_info, sizeof(info)) == 0);
      AMDF_EXPECT(state.context_create_count == 1);
      AMDF_EXPECT(state.context_destroy_count == 1);
      AMDF_EXPECT(state.fence_destroy_count == 0);
    } else {
      AMDF_EXPECT(result == AMDF_DXG_PROVIDER_RESULT_SUCCESS);
      if (result != AMDF_DXG_PROVIDER_RESULT_SUCCESS) {
        return false;
      }
      if (failure_stage == 3) {
        state.destroy_context_status = STATUS_DEVICE_BUSY;
      } else if (failure_stage == 4) {
        state.destroy_fence_status = STATUS_DEVICE_BUSY;
      }
      AMDF_EXPECT(
          amdf::wddm_provider::GpuKernelQueueDestroy(queue, &native_status) ==
          (failure_stage == 2 ? AMDF_DXG_PROVIDER_RESULT_SUCCESS
                              : AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE));
      AMDF_EXPECT(native_status ==
                  static_cast<uint32_t>(failure_stage == 2
                                            ? STATUS_SUCCESS
                                            : STATUS_DEVICE_BUSY));
      AMDF_EXPECT(state.context_destroy_count == 1);
      AMDF_EXPECT(state.fence_destroy_count == 1);
    }
    AMDF_EXPECT(state.queue_create_count == 0);
    AMDF_EXPECT(state.queue_destroy_count == 0);
    AMDF_EXPECT(state.live_allocation_count == 0);
    AMDF_EXPECT(adapter.live_queue_count.load() == 0);
    AMDF_EXPECT(state.dependency_failure_count == 0);
    AMDF_EXPECT(state.invalid_alignment_count == 0);
  }
  current_state = nullptr;
  return passed;
}

#undef AMDF_EXPECT

}  // namespace

int main() {
  struct TestCase {
    const char* name;
    bool (*run)();
  };
  const TestCase cases[] = {
      {"UnsupportedProfileHasNoOwnershipEffects",
       UnsupportedProfileHasNoOwnershipEffects},
      {"HostAllocationFailureLeavesNoNativeOwnership",
       HostAllocationFailureLeavesNoNativeOwnership},
      {"NativeFailureRollsBackEveryPublishedHandle",
       NativeFailureRollsBackEveryPublishedHandle},
      {"HardwareQueueUsesSharedRecordsAndConsumesEveryResult",
       HardwareQueueUsesSharedRecordsAndConsumesEveryResult},
      {"SoftwareSubmissionSeparatesAcceptanceFromSignal",
       SoftwareSubmissionSeparatesAcceptanceFromSignal},
      {"SoftwareConstructionAndReleasePreserveOwnership",
       SoftwareConstructionAndReleasePreserveOwnership},
  };
  bool passed = true;
  for (const TestCase& test : cases) {
    std::fprintf(stderr, "[ RUN      ] %s\n", test.name);
    if (test.run()) {
      std::fprintf(stderr, "[       OK ] %s\n", test.name);
    } else {
      std::fprintf(stderr, "[  FAILED  ] %s\n", test.name);
      passed = false;
    }
  }
  return passed ? 0 : 1;
}
