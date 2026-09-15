// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/wkmi/kernel_queue.h"

#include <malloc.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "libamdf/src/gpu/umd/wddm/wkmi/adapter_state.h"

namespace {

constexpr D3DKMT_HANDLE kDevice = 0x10;
constexpr D3DKMT_HANDLE kContext = 0x20;
constexpr D3DKMT_HANDLE kQueue = 0x30;
constexpr D3DKMT_HANDLE kFence = 0x40;

struct FakeNativeState {
  // Native result selected for context construction.
  NTSTATUS create_context_status = STATUS_SUCCESS;
  // Native result selected for queue construction.
  NTSTATUS create_queue_status = STATUS_SUCCESS;
  // Native result selected for queue destruction.
  NTSTATUS destroy_queue_status = STATUS_SUCCESS;
  // Native result selected for context destruction.
  NTSTATUS destroy_context_status = STATUS_SUCCESS;
  // Fence handle returned on successful queue construction.
  D3DKMT_HANDLE fence = kFence;
  // Native-owned progress mapping, independent of bridge host allocations.
  uint64_t progress = 0;
  // Number of native context creation calls.
  uint32_t context_create_count = 0;
  // Number of native queue destruction calls.
  uint32_t queue_destroy_count = 0;
  // Number of native context destruction calls.
  uint32_t context_destroy_count = 0;
  // Number of submitted commands.
  uint32_t submit_count = 0;
  // Host allocation ordinal rejected, or SIZE_MAX for no injected failure.
  size_t failure_ordinal = SIZE_MAX;
  // Number of attempted host allocations.
  size_t allocation_count = 0;
  // Number of host allocations still owned by the bridge.
  size_t live_allocation_count = 0;
  // Number of contract violations observed by native dependencies.
  uint32_t dependency_failure_count = 0;
};

FakeNativeState* current_state = nullptr;

void ExpectDependency(bool condition) {
  if (!condition) ++current_state->dependency_failure_count;
}

void* AMDF_CALL Allocate(void* user_data, uint64_t byte_length,
                         uint64_t minimum_alignment) {
  auto* state = static_cast<FakeNativeState*>(user_data);
  if (state->allocation_count++ == state->failure_ordinal) return nullptr;
  void* pointer = _aligned_malloc(static_cast<size_t>(byte_length),
                                  static_cast<size_t>(minimum_alignment));
  if (pointer != nullptr) ++state->live_allocation_count;
  return pointer;
}

void AMDF_CALL Free(void* user_data, void* allocation) {
  if (allocation == nullptr) return;
  auto* state = static_cast<FakeNativeState*>(user_data);
  --state->live_allocation_count;
  _aligned_free(allocation);
}

amdf_wkmi_bridge_gpu_kernel_queue_create_info_t MakeCreateInfo() {
  amdf_wkmi_bridge_gpu_kernel_queue_create_info_t info = {};
  info.structure_size = sizeof(info);
  info.device_handle = kDevice;
  info.command_type = AMDF_WKMI_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4;
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
  ExpectDependency(create->Flags.HwQueueSupported == 1);
  ExpectDependency(create->pPrivateDriverData != nullptr);
  ExpectDependency(create->PrivateDriverDataSize == 16);
  if (current_state->create_context_status == STATUS_SUCCESS) {
    create->hContext = kContext;
  }
  return current_state->create_context_status;
}

extern "C" NTSTATUS APIENTRY D3DKMTCreateHwQueue(D3DKMT_CREATEHWQUEUE* create) {
  ExpectDependency(create->hHwContext == kContext);
  ExpectDependency(create->Flags.DisableGpuTimeout == 0);
  ExpectDependency(create->pPrivateDriverData != nullptr);
  ExpectDependency(create->PrivateDriverDataSize == 32);
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

extern "C" NTSTATUS APIENTRY
D3DKMTSubmitCommandToHwQueue(const D3DKMT_SUBMITCOMMANDTOHWQUEUE* submit) {
  ++current_state->submit_count;
  ExpectDependency(submit->hHwQueue == kQueue);
  ExpectDependency(submit->HwQueueProgressFenceId == 7);
  ExpectDependency(submit->CommandBuffer == 0x2000);
  ExpectDependency(submit->CommandLength == 64);
  ExpectDependency(submit->pPrivateDriverData != nullptr);
  ExpectDependency(submit->PrivateDriverDataSize == 48);
  current_state->progress = submit->HwQueueProgressFenceId;
  return STATUS_SUCCESS;
}

namespace Wkmi {

int EngineOrdinal(int, DeviceInfo*) { return 0; }
bool GetHwsEnabled(int, DeviceInfo*) { return true; }
int GetContextPrivDataSize() { return 16; }
int GetHwQueuePrivDataSize() { return 32; }
int GetSubmitPrivDataSize() { return 48; }

void FillinContextPrivData(void* private_data, bool, uint32_t, uint64_t) {
  ExpectDependency(private_data != nullptr);
}

void FillinHwQueuePrivData(void* private_data, bool, SchedLevel, bool aql,
                           uint64_t queue_address, uint64_t queue_size,
                           uint64_t write_pointer, uint64_t read_pointer,
                           D3DKMT_HANDLE descriptor, uint32_t** doorbell) {
  ExpectDependency(private_data != nullptr);
  ExpectDependency(!aql && queue_address == 0 && queue_size == 0 &&
                   write_pointer == 0 && read_pointer == 0 && descriptor == 0 &&
                   doorbell == nullptr);
}

void FillinSubmitPrivData(void* private_data, D3DKMT_HANDLE queue,
                          uint64_t address, uint64_t byte_length,
                          bool hardware_queue) {
  ExpectDependency(private_data != nullptr && queue == kQueue &&
                   address == 0x2000 && byte_length == 64 && hardware_queue);
}

}  // namespace Wkmi

namespace {

bool AllocationFailureLeavesNoNativeOwnership() {
  bool passed = true;
  for (size_t ordinal = 0; ordinal < 4; ++ordinal) {
    FakeNativeState state;
    state.failure_ordinal = ordinal;
    current_state = &state;
    amdf_wkmi_bridge_gpu_adapter_t adapter;
    adapter.host_allocator = {&state, Allocate, nullptr, Free};
    const auto create = MakeCreateInfo();
    auto* queue = reinterpret_cast<amdf_wkmi_bridge_gpu_kernel_queue_t*>(1);
    amdf_wkmi_bridge_gpu_kernel_queue_info_t info;
    std::memset(&info, 0xA5, sizeof(info));
    const auto original = info;
    uint32_t native_status = 1;
    AMDF_EXPECT(amdf::wkmi_bridge::GpuKernelQueueCreate(
                    &adapter, &create, &queue, &info, &native_status) ==
                AMDF_WKMI_BRIDGE_RESULT_RESOURCE_EXHAUSTED);
    AMDF_EXPECT(reinterpret_cast<uintptr_t>(queue) == 1);
    AMDF_EXPECT(std::memcmp(&info, &original, sizeof(info)) == 0);
    AMDF_EXPECT(native_status == 0);
    AMDF_EXPECT(state.live_allocation_count == 0);
    AMDF_EXPECT(state.context_create_count == 0);
    AMDF_EXPECT(adapter.live_queue_count == 0);
  }
  current_state = nullptr;
  return passed;
}

bool NativeFailureLeavesNoAdapterCleanupObligation() {
  bool passed = true;
  struct FailureCase {
    // Context creation result.
    NTSTATUS create_context;
    // Queue creation result.
    NTSTATUS create_queue;
    // Queue destruction result.
    NTSTATUS destroy_queue;
    // Context destruction result.
    NTSTATUS destroy_context;
    // Error reported by construction after rollback.
    NTSTATUS expected_status;
    // Expected queue destruction calls.
    uint32_t queue_destroy_count;
    // Expected context destruction calls.
    uint32_t context_destroy_count;
  };
  const FailureCase cases[] = {
      {STATUS_NO_MEMORY, STATUS_SUCCESS, STATUS_SUCCESS, STATUS_SUCCESS,
       STATUS_NO_MEMORY, 0, 0},
      {STATUS_SUCCESS, STATUS_NO_MEMORY, STATUS_SUCCESS, STATUS_SUCCESS,
       STATUS_NO_MEMORY, 0, 1},
      {STATUS_SUCCESS, STATUS_SUCCESS, STATUS_SUCCESS, STATUS_SUCCESS,
       STATUS_INVALID_HANDLE, 1, 1},
      {STATUS_SUCCESS, STATUS_SUCCESS, STATUS_DEVICE_BUSY, STATUS_SUCCESS,
       STATUS_DEVICE_BUSY, 1, 0},
      {STATUS_SUCCESS, STATUS_SUCCESS, STATUS_SUCCESS, STATUS_DEVICE_BUSY,
       STATUS_DEVICE_BUSY, 1, 1},
  };
  for (const auto& failure : cases) {
    FakeNativeState state;
    state.create_context_status = failure.create_context;
    state.create_queue_status = failure.create_queue;
    state.destroy_queue_status = failure.destroy_queue;
    state.destroy_context_status = failure.destroy_context;
    state.fence = 0;
    current_state = &state;
    amdf_wkmi_bridge_gpu_adapter_t adapter;
    adapter.host_allocator = {&state, Allocate, nullptr, Free};
    const auto create = MakeCreateInfo();
    auto* queue = reinterpret_cast<amdf_wkmi_bridge_gpu_kernel_queue_t*>(1);
    amdf_wkmi_bridge_gpu_kernel_queue_info_t info;
    std::memset(&info, 0xA5, sizeof(info));
    const auto original = info;
    uint32_t native_status = 0;
    AMDF_EXPECT(amdf::wkmi_bridge::GpuKernelQueueCreate(
                    &adapter, &create, &queue, &info, &native_status) ==
                AMDF_WKMI_BRIDGE_RESULT_NATIVE_FAILURE);
    AMDF_EXPECT(native_status ==
                static_cast<uint32_t>(failure.expected_status));
    AMDF_EXPECT(reinterpret_cast<uintptr_t>(queue) == 1);
    AMDF_EXPECT(std::memcmp(&info, &original, sizeof(info)) == 0);
    AMDF_EXPECT(state.live_allocation_count == 0);
    AMDF_EXPECT(adapter.live_queue_count == 0);
    AMDF_EXPECT(
        amdf::wkmi_bridge::PrepareGpuAdapterClose(&adapter, &native_status) ==
        AMDF_WKMI_BRIDGE_RESULT_SUCCESS);
    AMDF_EXPECT(state.queue_destroy_count == failure.queue_destroy_count);
    AMDF_EXPECT(state.context_destroy_count == failure.context_destroy_count);
    AMDF_EXPECT(state.submit_count == 0);
    AMDF_EXPECT(state.dependency_failure_count == 0);
  }
  current_state = nullptr;
  return passed;
}

bool PublishedQueueOwnsSubmissionUntilExplicitDestruction() {
  bool passed = true;
  FakeNativeState state;
  current_state = &state;
  amdf_wkmi_bridge_gpu_adapter_t adapter;
  adapter.host_allocator = {&state, Allocate, nullptr, Free};
  const auto create = MakeCreateInfo();
  amdf_wkmi_bridge_gpu_kernel_queue_t* queue = nullptr;
  amdf_wkmi_bridge_gpu_kernel_queue_info_t info = {};
  uint32_t native_status = 0;
  AMDF_EXPECT(amdf::wkmi_bridge::GpuKernelQueueCreate(&adapter, &create, &queue,
                                                      &info, &native_status) ==
              AMDF_WKMI_BRIDGE_RESULT_SUCCESS);
  if (queue == nullptr) return false;
  AMDF_EXPECT(info.progress_fence_handle == kFence);
  AMDF_EXPECT(info.progress_fence_pointer == &state.progress);
  AMDF_EXPECT(state.live_allocation_count == 2);
  AMDF_EXPECT(adapter.live_queue_count == 1);
  AMDF_EXPECT(amdf::wkmi_bridge::GpuKernelQueueSubmit(queue, 0x2000, 64, 7,
                                                      &native_status) ==
              AMDF_WKMI_BRIDGE_RESULT_SUCCESS);
  AMDF_EXPECT(*info.progress_fence_pointer == 7);
  state.destroy_queue_status = STATUS_DEVICE_BUSY;
  AMDF_EXPECT(amdf::wkmi_bridge::GpuKernelQueueDestroy(queue, &native_status) ==
              AMDF_WKMI_BRIDGE_RESULT_NATIVE_FAILURE);
  AMDF_EXPECT(native_status == static_cast<uint32_t>(STATUS_DEVICE_BUSY));
  AMDF_EXPECT(amdf::wkmi_bridge::PrepareGpuAdapterClose(
                  &adapter, &native_status) == AMDF_WKMI_BRIDGE_RESULT_BUSY);
  AMDF_EXPECT(state.live_allocation_count == 2);
  AMDF_EXPECT(state.context_destroy_count == 0);
  state.destroy_queue_status = STATUS_SUCCESS;
  AMDF_EXPECT(amdf::wkmi_bridge::GpuKernelQueueDestroy(queue, &native_status) ==
              AMDF_WKMI_BRIDGE_RESULT_SUCCESS);
  AMDF_EXPECT(state.live_allocation_count == 0);
  AMDF_EXPECT(adapter.live_queue_count == 0);
  AMDF_EXPECT(state.queue_destroy_count == 2);
  AMDF_EXPECT(state.context_destroy_count == 1);
  AMDF_EXPECT(state.submit_count == 1);
  AMDF_EXPECT(state.dependency_failure_count == 0);
  current_state = nullptr;
  return passed;
}

#undef AMDF_EXPECT

}  // namespace

int main() {
  struct TestCase {
    // Name printed with the result.
    const char* name;
    // Case returning whether every expectation passed.
    bool (*run)();
  };
  const TestCase cases[] = {
      {"AllocationFailureLeavesNoNativeOwnership",
       AllocationFailureLeavesNoNativeOwnership},
      {"NativeFailureLeavesNoAdapterCleanupObligation",
       NativeFailureLeavesNoAdapterCleanupObligation},
      {"PublishedQueueOwnsSubmissionUntilExplicitDestruction",
       PublishedQueueOwnsSubmissionUntilExplicitDestruction},
  };
  bool passed = true;
  for (const auto& test : cases) {
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
