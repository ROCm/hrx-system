// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/wkmi/allocation.h"

#include <malloc.h>

#include <array>
#include <cstdint>
#include <cstdio>

#include "libamdf/src/gpu/umd/wddm/wkmi/adapter_state.h"

namespace {

constexpr NTSTATUS kStatusNoMemory = static_cast<NTSTATUS>(0xC0000017u);
constexpr D3DKMT_HANDLE kDeviceHandle = 0x10;
constexpr D3DKMT_HANDLE kResourceHandle = 0x21;
constexpr D3DKMT_HANDLE kAllocationHandle = 0x20;

struct TestAllocatorState {
  // Allocation ordinal rejected by the callback, or `SIZE_MAX` for none.
  size_t failure_ordinal = SIZE_MAX;
  // Number of allocation callback invocations.
  size_t allocation_count = 0;
  // Number of callback allocations not yet freed.
  size_t live_allocation_count = 0;
  // Number of requests violating the public allocator alignment contract.
  size_t invalid_alignment_count = 0;
};

void* AMDF_CALL TestAllocate(void* user_data, uint64_t byte_length,
                             uint64_t minimum_alignment) {
  auto* state = static_cast<TestAllocatorState*>(user_data);
  const size_t ordinal = state->allocation_count++;
  if (ordinal == state->failure_ordinal) return nullptr;
  if (minimum_alignment < amdf_max_align_t ||
      (minimum_alignment & (minimum_alignment - 1)) != 0) {
    ++state->invalid_alignment_count;
  }
  void* pointer = _aligned_malloc(static_cast<size_t>(byte_length),
                                  static_cast<size_t>(minimum_alignment));
  if (pointer != nullptr) ++state->live_allocation_count;
  return pointer;
}

void AMDF_CALL TestFree(void* user_data, void* allocation) {
  auto* state = static_cast<TestAllocatorState*>(user_data);
  if (allocation == nullptr) return;
  --state->live_allocation_count;
  _aligned_free(allocation);
}

amdf_allocator_t MakeTestAllocator(TestAllocatorState* state) {
  return amdf_allocator_t{state, TestAllocate, nullptr, TestFree};
}

struct FakeNativeState {
  // Status returned by native allocation creation.
  NTSTATUS create_status = STATUS_SUCCESS;
  // Number of successful creates that return a zero allocation handle.
  uint32_t malformed_create_count = 0;
  // Number of native create attempts.
  uint32_t create_attempt_count = 0;
  // Number of native destroy attempts.
  uint32_t destroy_attempt_count = 0;
  // Number of contract violations observed inside a fake dependency.
  uint32_t dependency_failure_count = 0;
};

FakeNativeState* g_fake_native_state = nullptr;

void RecordDependencyExpectation(bool condition) {
  if (!condition) {
    ++g_fake_native_state->dependency_failure_count;
  }
}

bool Check(bool condition, const char* expression, int line) {
  if (!condition) {
    std::fprintf(stderr, "allocation_test.cc:%d: check failed: %s\n", line,
                 expression);
  }
  return condition;
}

#define AMDF_EXPECT(expression) \
  passed &= Check(static_cast<bool>(expression), #expression, __LINE__)

amdf_wkmi_bridge_gpu_allocation_create_info_t MakeSystemCreateInfo(
    void* host_pointer, uint64_t byte_length) {
  amdf_wkmi_bridge_gpu_allocation_create_info_t create_info = {};
  create_info.structure_size = sizeof(create_info);
  create_info.device_handle = kDeviceHandle;
  create_info.domain = AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_SYSTEM;
  create_info.byte_length = byte_length;
  create_info.host_pointer = host_pointer;
  return create_info;
}

}  // namespace

extern "C" NTSTATUS APIENTRY
D3DKMTCreateAllocation2(D3DKMT_CREATEALLOCATION* create) {
  ++g_fake_native_state->create_attempt_count;
  RecordDependencyExpectation(create->hDevice == kDeviceHandle);
  RecordDependencyExpectation(create->NumAllocations == 1);
  RecordDependencyExpectation(create->pPrivateDriverData != nullptr);
  RecordDependencyExpectation(create->pAllocationInfo2 != nullptr);
  if (g_fake_native_state->create_status != STATUS_SUCCESS) {
    return g_fake_native_state->create_status;
  }
  create->hResource = kResourceHandle;
  create->pAllocationInfo2[0].hAllocation =
      g_fake_native_state->malformed_create_count == 0 ? kAllocationHandle : 0;
  if (g_fake_native_state->malformed_create_count != 0) {
    --g_fake_native_state->malformed_create_count;
  }
  return STATUS_SUCCESS;
}

extern "C" NTSTATUS APIENTRY
D3DKMTDestroyAllocation2(const D3DKMT_DESTROYALLOCATION2*) {
  ++g_fake_native_state->destroy_attempt_count;
  return kStatusNoMemory;
}

namespace Wkmi {

void GetAllocPrivDataSize(int* driver_private_data_size,
                          int* allocation_private_data_size) {
  *driver_private_data_size = 16;
  *allocation_private_data_size = 32;
}

void FillinAllocPrivDrvData(void* driver_private_data,
                            int allocation_private_data_size) {
  RecordDependencyExpectation(driver_private_data != nullptr);
  RecordDependencyExpectation(allocation_private_data_size == 32);
}

void SetAllocationInfo(void* allocation_private_data, uint64_t byte_length,
                       AllocDomain domain, uint64_t address,
                       uint32_t memory_flags, uint32_t engine_flag,
                       const DeviceInfo&) {
  RecordDependencyExpectation(allocation_private_data != nullptr);
  RecordDependencyExpectation(byte_length == 4096);
  RecordDependencyExpectation(domain == kSystem);
  RecordDependencyExpectation(address == 0);
  RecordDependencyExpectation(memory_flags == 0);
  RecordDependencyExpectation(engine_flag == KCOMPUTE0);
}

}  // namespace Wkmi

namespace {

bool MalformedNativeSuccessPublishesResourceToCaller() {
  bool passed = true;
  FakeNativeState native_state;
  TestAllocatorState allocator_state;
  native_state.malformed_create_count = 1;
  g_fake_native_state = &native_state;
  {
    amdf_wkmi_bridge_gpu_adapter_t adapter;
    adapter.host_allocator = MakeTestAllocator(&allocator_state);
    alignas(4096) std::array<uint8_t, 4096> host_storage = {};
    const amdf_wkmi_bridge_gpu_allocation_create_info_t create_info =
        MakeSystemCreateInfo(host_storage.data(), host_storage.size());
    uint32_t allocation_handle = 0xA0;
    uint32_t resource_handle = 0xA1;
    uint32_t allocation_count = 0xA2;
    uint32_t native_status = 0xA3;

    AMDF_EXPECT(amdf::wkmi_bridge::GpuAllocationCreate(
                    &adapter, &create_info, 1, &allocation_handle,
                    &resource_handle, &allocation_count,
                    &native_status) == AMDF_WKMI_BRIDGE_RESULT_SUCCESS);
    AMDF_EXPECT(allocation_handle == 0);
    AMDF_EXPECT(resource_handle == kResourceHandle);
    AMDF_EXPECT(allocation_count == 1);
    AMDF_EXPECT(native_status == 0);
    AMDF_EXPECT(native_state.create_attempt_count == 1);
    AMDF_EXPECT(native_state.destroy_attempt_count == 0);

    native_status = 0xB0;
    AMDF_EXPECT(
        amdf::wkmi_bridge::PrepareGpuAdapterClose(&adapter, &native_status) ==
        AMDF_WKMI_BRIDGE_RESULT_SUCCESS);
    AMDF_EXPECT(native_status == 0);
    AMDF_EXPECT(native_state.destroy_attempt_count == 0);
  }
  AMDF_EXPECT(native_state.dependency_failure_count == 0);
  AMDF_EXPECT(allocator_state.live_allocation_count == 0);
  AMDF_EXPECT(allocator_state.invalid_alignment_count == 0);
  g_fake_native_state = nullptr;
  return passed;
}

bool SuccessfulCreatePublishesNativeHandles() {
  bool passed = true;
  FakeNativeState native_state;
  TestAllocatorState allocator_state;
  g_fake_native_state = &native_state;
  {
    amdf_wkmi_bridge_gpu_adapter_t adapter;
    adapter.host_allocator = MakeTestAllocator(&allocator_state);
    alignas(4096) std::array<uint8_t, 4096> host_storage = {};
    const amdf_wkmi_bridge_gpu_allocation_create_info_t create_info =
        MakeSystemCreateInfo(host_storage.data(), host_storage.size());
    uint32_t allocation_handle = 0xA0;
    uint32_t resource_handle = 0xA1;
    uint32_t allocation_count = 0xA2;
    uint32_t native_status = 0xA3;

    AMDF_EXPECT(amdf::wkmi_bridge::GpuAllocationCreate(
                    &adapter, &create_info, 1, &allocation_handle,
                    &resource_handle, &allocation_count,
                    &native_status) == AMDF_WKMI_BRIDGE_RESULT_SUCCESS);
    AMDF_EXPECT(allocation_handle == kAllocationHandle);
    AMDF_EXPECT(resource_handle == kResourceHandle);
    AMDF_EXPECT(allocation_count == 1);
    AMDF_EXPECT(native_status == 0);
    AMDF_EXPECT(native_state.destroy_attempt_count == 0);

    AMDF_EXPECT(
        amdf::wkmi_bridge::PrepareGpuAdapterClose(&adapter, &native_status) ==
        AMDF_WKMI_BRIDGE_RESULT_SUCCESS);
    AMDF_EXPECT(native_state.destroy_attempt_count == 0);
  }
  AMDF_EXPECT(native_state.dependency_failure_count == 0);
  AMDF_EXPECT(allocator_state.live_allocation_count == 0);
  AMDF_EXPECT(allocator_state.invalid_alignment_count == 0);
  g_fake_native_state = nullptr;
  return passed;
}

bool NativeCreateFailureLeavesOutputsUnchanged() {
  bool passed = true;
  FakeNativeState native_state;
  TestAllocatorState allocator_state;
  native_state.create_status = kStatusNoMemory;
  g_fake_native_state = &native_state;
  {
    amdf_wkmi_bridge_gpu_adapter_t adapter;
    adapter.host_allocator = MakeTestAllocator(&allocator_state);
    alignas(4096) std::array<uint8_t, 4096> host_storage = {};
    const amdf_wkmi_bridge_gpu_allocation_create_info_t create_info =
        MakeSystemCreateInfo(host_storage.data(), host_storage.size());
    uint32_t allocation_handle = 0xA0;
    uint32_t resource_handle = 0xA1;
    uint32_t allocation_count = 0xA2;
    uint32_t native_status = 0xA3;

    AMDF_EXPECT(amdf::wkmi_bridge::GpuAllocationCreate(
                    &adapter, &create_info, 1, &allocation_handle,
                    &resource_handle, &allocation_count,
                    &native_status) == AMDF_WKMI_BRIDGE_RESULT_NATIVE_FAILURE);
    AMDF_EXPECT(allocation_handle == 0xA0);
    AMDF_EXPECT(resource_handle == 0xA1);
    AMDF_EXPECT(allocation_count == 0xA2);
    AMDF_EXPECT(native_status == static_cast<uint32_t>(kStatusNoMemory));
    AMDF_EXPECT(native_state.destroy_attempt_count == 0);
  }
  AMDF_EXPECT(native_state.dependency_failure_count == 0);
  AMDF_EXPECT(allocator_state.live_allocation_count == 0);
  AMDF_EXPECT(allocator_state.invalid_alignment_count == 0);
  g_fake_native_state = nullptr;
  return passed;
}

bool InsufficientCapacityReportsCountWithoutAllocating() {
  bool passed = true;
  FakeNativeState native_state;
  TestAllocatorState allocator_state;
  g_fake_native_state = &native_state;
  {
    amdf_wkmi_bridge_gpu_adapter_t adapter;
    adapter.host_allocator = MakeTestAllocator(&allocator_state);
    alignas(4096) std::array<uint8_t, 4096> host_storage = {};
    const amdf_wkmi_bridge_gpu_allocation_create_info_t create_info =
        MakeSystemCreateInfo(host_storage.data(), host_storage.size());
    uint32_t resource_handle = 0xA1;
    uint32_t allocation_count = 0;
    uint32_t native_status = 0xA3;
    AMDF_EXPECT(amdf::wkmi_bridge::GpuAllocationCreate(
                    &adapter, &create_info, 0, nullptr, &resource_handle,
                    &allocation_count, &native_status) ==
                AMDF_WKMI_BRIDGE_RESULT_BUFFER_TOO_SMALL);
    AMDF_EXPECT(resource_handle == 0xA1);
    AMDF_EXPECT(allocation_count == 1);
    AMDF_EXPECT(native_status == 0);
    AMDF_EXPECT(native_state.create_attempt_count == 0);
    AMDF_EXPECT(allocator_state.allocation_count == 0);
  }
  g_fake_native_state = nullptr;
  return passed;
}

bool QueryLayoutPublishesOnlyOnSuccess() {
  bool passed = true;
  amdf_wkmi_bridge_gpu_adapter_t adapter;
  uint32_t allocation_count = 0xA0;
  uint64_t maximum_allocation_byte_length = 0xA1;

  AMDF_EXPECT(
      amdf::wkmi_bridge::GpuAllocationQueryLayout(
          &adapter, 1, &allocation_count, &maximum_allocation_byte_length) ==
      AMDF_WKMI_BRIDGE_RESULT_INVALID_ARGUMENT);
  AMDF_EXPECT(allocation_count == 0xA0);
  AMDF_EXPECT(maximum_allocation_byte_length == 0xA1);

  AMDF_EXPECT(
      amdf::wkmi_bridge::GpuAllocationQueryLayout(
          &adapter, 4096, &allocation_count, &maximum_allocation_byte_length) ==
      AMDF_WKMI_BRIDGE_RESULT_SUCCESS);
  AMDF_EXPECT(allocation_count == 1);
  AMDF_EXPECT(maximum_allocation_byte_length ==
              UINT64_C(2) * 1024 * 1024 * 1024);
  return passed;
}

bool MetadataAllocationFailureLeavesOwnershipOutputsUnchanged() {
  bool passed = true;
  FakeNativeState native_state;
  TestAllocatorState allocator_state;
  allocator_state.failure_ordinal = 0;
  g_fake_native_state = &native_state;
  {
    amdf_wkmi_bridge_gpu_adapter_t adapter;
    adapter.host_allocator = MakeTestAllocator(&allocator_state);
    alignas(4096) std::array<uint8_t, 4096> host_storage = {};
    const amdf_wkmi_bridge_gpu_allocation_create_info_t create_info =
        MakeSystemCreateInfo(host_storage.data(), host_storage.size());
    uint32_t allocation_handle = 0xA0;
    uint32_t resource_handle = 0xA1;
    uint32_t allocation_count = 0xA2;
    uint32_t native_status = 0xA3;

    AMDF_EXPECT(amdf::wkmi_bridge::GpuAllocationCreate(
                    &adapter, &create_info, 1, &allocation_handle,
                    &resource_handle, &allocation_count, &native_status) ==
                AMDF_WKMI_BRIDGE_RESULT_RESOURCE_EXHAUSTED);
    AMDF_EXPECT(allocation_handle == 0xA0);
    AMDF_EXPECT(resource_handle == 0xA1);
    AMDF_EXPECT(allocation_count == 0xA2);
    AMDF_EXPECT(native_status == 0);
    AMDF_EXPECT(native_state.create_attempt_count == 0);
  }
  AMDF_EXPECT(allocator_state.live_allocation_count == 0);
  AMDF_EXPECT(allocator_state.invalid_alignment_count == 0);
  g_fake_native_state = nullptr;
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
  const TestCase test_cases[] = {
      {"MalformedNativeSuccessPublishesResourceToCaller",
       MalformedNativeSuccessPublishesResourceToCaller},
      {"SuccessfulCreatePublishesNativeHandles",
       SuccessfulCreatePublishesNativeHandles},
      {"NativeCreateFailureLeavesOutputsUnchanged",
       NativeCreateFailureLeavesOutputsUnchanged},
      {"QueryLayoutPublishesOnlyOnSuccess", QueryLayoutPublishesOnlyOnSuccess},
      {"InsufficientCapacityReportsCountWithoutAllocating",
       InsufficientCapacityReportsCountWithoutAllocating},
      {"MetadataAllocationFailureLeavesOwnershipOutputsUnchanged",
       MetadataAllocationFailureLeavesOwnershipOutputsUnchanged},
  };

  bool passed = true;
  for (const TestCase& test_case : test_cases) {
    std::fprintf(stderr, "[ RUN      ] %s\n", test_case.name);
    if (test_case.run()) {
      std::fprintf(stderr, "[       OK ] %s\n", test_case.name);
    } else {
      std::fprintf(stderr, "[  FAILED  ] %s\n", test_case.name);
      passed = false;
    }
  }
  return passed ? 0 : 1;
}
