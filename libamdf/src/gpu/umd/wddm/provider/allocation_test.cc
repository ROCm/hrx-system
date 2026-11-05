// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/provider/allocation.h"

#include <malloc.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

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
  if (ordinal == state->failure_ordinal) {
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

void AMDF_CALL TestFree(void* user_data, void* allocation) {
  auto* state = static_cast<TestAllocatorState*>(user_data);
  if (allocation == nullptr) {
    return;
  }
  --state->live_allocation_count;
  _aligned_free(allocation);
}

amdf_allocator_t MakeTestAllocator(TestAllocatorState* state) {
  return amdf_allocator_t{state, TestAllocate, nullptr, TestFree};
}

struct NativeAllocationRequest {
  // Exact payload bytes passed to the native allocation encoder.
  uint64_t byte_length;
  // Native placement or caller-page registration class.
  amdf_dxg_private_allocation_domain_t domain;
  // Native memory-policy bits encoded into the private allocation payload.
  amdf_dxg_private_allocation_flags_t flags = 0;
};

struct FakeNativeState {
  // Procedure table injected into the provider adapter.
  amdf_kmt_api_t kmt = {};
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
  // Complete native payload requests expected before allocation creation.
  std::vector<NativeAllocationRequest> expected_allocations = {
      {4096, AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_SYSTEM}};
  // Adapter facts controlling allocation segment and cache policy encoding.
  amdf_dxg_private_adapter_info_t adapter_info = {.gfx_ip_major = 11};
  // Number of allocation records encoded before the native create call.
  size_t allocation_info_count = 0;
};

FakeNativeState* g_fake_native_state = nullptr;

NTSTATUS APIENTRY FakeCreateAllocation(D3DKMT_CREATEALLOCATION* create);
NTSTATUS APIENTRY
FakeDestroyAllocation(const D3DKMT_DESTROYALLOCATION2* destroy);

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

void InitializeAdapter(amdf_dxg_provider_gpu_adapter_t* adapter,
                       TestAllocatorState* allocator_state,
                       FakeNativeState& native_state) {
  native_state.kmt.create_allocation = FakeCreateAllocation;
  native_state.kmt.destroy_allocation = FakeDestroyAllocation;
  adapter->host_allocator = MakeTestAllocator(allocator_state);
  adapter->kmt = &native_state.kmt;
  adapter->private_info = native_state.adapter_info;
}

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

amdf_dxg_provider_gpu_allocation_create_info_t MakeSystemCreateInfo(
    void* host_pointer, uint64_t byte_length) {
  amdf_dxg_provider_gpu_allocation_create_info_t create_info = {};
  create_info.device_handle = kDeviceHandle;
  create_info.domain = AMDF_DXG_PROVIDER_GPU_ALLOCATION_DOMAIN_SYSTEM;
  create_info.byte_length = byte_length;
  create_info.host_pointer = host_pointer;
  return create_info;
}

NTSTATUS APIENTRY FakeCreateAllocation(D3DKMT_CREATEALLOCATION* create) {
  ++g_fake_native_state->create_attempt_count;
  RecordDependencyExpectation(create->hDevice == kDeviceHandle);
  RecordDependencyExpectation(create->NumAllocations ==
                              g_fake_native_state->expected_allocations.size());
  RecordDependencyExpectation(create->pPrivateDriverData != nullptr);
  RecordDependencyExpectation(
      create->PrivateDriverDataSize ==
      AMDF_DXG_PRIVATE_ALLOCATION_DRIVER_RECORD_BYTE_LENGTH);
  RecordDependencyExpectation(create->pAllocationInfo2 != nullptr);
  const auto* driver_record =
      static_cast<const uint8_t*>(create->pPrivateDriverData);
  RecordDependencyExpectation(driver_record[0x08] == 0x80);
  RecordDependencyExpectation(Load32(driver_record + 0x3C) ==
                              AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH);
  for (uint32_t i = 0; i < create->NumAllocations; ++i) {
    const D3DDDI_ALLOCATIONINFO2& allocation = create->pAllocationInfo2[i];
    RecordDependencyExpectation(allocation.PrivateDriverDataSize ==
                                AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH);
    RecordDependencyExpectation(allocation.pPrivateDriverData != nullptr);
    RecordDependencyExpectation(allocation.VidPnSourceId ==
                                D3DDDI_ID_UNINITIALIZED);
    const size_t ordinal = g_fake_native_state->allocation_info_count++;
    RecordDependencyExpectation(
        ordinal < g_fake_native_state->expected_allocations.size());
    if (ordinal >= g_fake_native_state->expected_allocations.size()) {
      continue;
    }
    const NativeAllocationRequest& expected =
        g_fake_native_state->expected_allocations[ordinal];
    const auto* record =
        static_cast<const uint8_t*>(allocation.pPrivateDriverData);
    RecordDependencyExpectation(Load32(record) == 0x1D8);
    RecordDependencyExpectation(Load64(record + 0xA8) == expected.byte_length);
    if (expected.domain == AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_LOCAL) {
      RecordDependencyExpectation(allocation.pSystemMem == nullptr);
      const bool has_invisible_heap =
          g_fake_native_state->adapter_info.invisible_local_heap_byte_length !=
          0;
      RecordDependencyExpectation(record[0x50] == (has_invisible_heap ? 7 : 5));
      RecordDependencyExpectation(record[0x5C] == (has_invisible_heap ? 3 : 2));
    } else {
      RecordDependencyExpectation(allocation.pSystemMem != nullptr);
      const uint32_t expected_segment_flags =
          expected.domain == AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_SYSTEM
              ? 0x24002008
              : 0x04000008;
      RecordDependencyExpectation(Load32(record + 0x50) ==
                                  expected_segment_flags);
      RecordDependencyExpectation(record[0x5C] == 1 && record[0x60] == 4);
      const bool expects_native_queue =
          (expected.flags & AMDF_DXG_PRIVATE_ALLOCATION_FLAG_QUEUE_STORAGE) !=
              0 &&
          (g_fake_native_state->adapter_info.native_queue_flags &
           AMDF_DXG_PRIVATE_NATIVE_QUEUE_AQL) != 0;
      RecordDependencyExpectation(((record[0xA6] & 1) != 0) ==
                                  expects_native_queue);
      const bool expects_uncached =
          (g_fake_native_state->adapter_info.gfx_ip_major < 12 &&
           (expected.flags & AMDF_DXG_PRIVATE_ALLOCATION_FLAG_FINE_GRAIN) !=
               0) ||
          (expected.flags & AMDF_DXG_PRIVATE_ALLOCATION_FLAG_DEVICE_UNCACHED) !=
              0;
      RecordDependencyExpectation(Load32(record + 0x1D0) ==
                                  (expects_uncached ? 4u : 0u));
    }
  }
  if (g_fake_native_state->create_status != STATUS_SUCCESS) {
    return g_fake_native_state->create_status;
  }
  create->hResource = kResourceHandle;
  for (uint32_t i = 0; i < create->NumAllocations; ++i) {
    create->pAllocationInfo2[i].hAllocation =
        g_fake_native_state->malformed_create_count == 0 ? kAllocationHandle + i
                                                         : 0;
  }
  if (g_fake_native_state->malformed_create_count != 0) {
    --g_fake_native_state->malformed_create_count;
  }
  return STATUS_SUCCESS;
}

NTSTATUS APIENTRY FakeDestroyAllocation(const D3DKMT_DESTROYALLOCATION2*) {
  ++g_fake_native_state->destroy_attempt_count;
  return kStatusNoMemory;
}

bool MalformedNativeSuccessPublishesResourceToCaller() {
  bool passed = true;
  FakeNativeState native_state;
  TestAllocatorState allocator_state;
  native_state.malformed_create_count = 1;
  g_fake_native_state = &native_state;
  {
    amdf_dxg_provider_gpu_adapter_t adapter;
    InitializeAdapter(&adapter, &allocator_state, native_state);
    alignas(4096) std::array<uint8_t, 4096> host_storage = {};
    const amdf_dxg_provider_gpu_allocation_create_info_t create_info =
        MakeSystemCreateInfo(host_storage.data(), host_storage.size());
    uint32_t allocation_handle = 0xA0;
    uint32_t resource_handle = 0xA1;
    uint32_t allocation_count = 0xA2;
    uint32_t native_status = 0xA3;

    AMDF_EXPECT(amdf::wddm_provider::GpuAllocationCreate(
                    &adapter, &create_info, 1, &allocation_handle,
                    &resource_handle, &allocation_count,
                    &native_status) == AMDF_DXG_PROVIDER_RESULT_SUCCESS);
    AMDF_EXPECT(allocation_handle == 0);
    AMDF_EXPECT(resource_handle == kResourceHandle);
    AMDF_EXPECT(allocation_count == 1);
    AMDF_EXPECT(native_status == 0);
    AMDF_EXPECT(native_state.create_attempt_count == 1);
    AMDF_EXPECT(native_state.destroy_attempt_count == 0);

    native_status = 0xB0;
    AMDF_EXPECT(
        amdf::wddm_provider::PrepareGpuAdapterClose(&adapter, &native_status) ==
        AMDF_DXG_PROVIDER_RESULT_SUCCESS);
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
    amdf_dxg_provider_gpu_adapter_t adapter;
    InitializeAdapter(&adapter, &allocator_state, native_state);
    alignas(4096) std::array<uint8_t, 4096> host_storage = {};
    const amdf_dxg_provider_gpu_allocation_create_info_t create_info =
        MakeSystemCreateInfo(host_storage.data(), host_storage.size());
    uint32_t allocation_handle = 0xA0;
    uint32_t resource_handle = 0xA1;
    uint32_t allocation_count = 0xA2;
    uint32_t native_status = 0xA3;

    AMDF_EXPECT(amdf::wddm_provider::GpuAllocationCreate(
                    &adapter, &create_info, 1, &allocation_handle,
                    &resource_handle, &allocation_count,
                    &native_status) == AMDF_DXG_PROVIDER_RESULT_SUCCESS);
    AMDF_EXPECT(allocation_handle == kAllocationHandle);
    AMDF_EXPECT(resource_handle == kResourceHandle);
    AMDF_EXPECT(allocation_count == 1);
    AMDF_EXPECT(native_status == 0);
    AMDF_EXPECT(native_state.destroy_attempt_count == 0);

    AMDF_EXPECT(
        amdf::wddm_provider::PrepareGpuAdapterClose(&adapter, &native_status) ==
        AMDF_DXG_PROVIDER_RESULT_SUCCESS);
    AMDF_EXPECT(native_state.destroy_attempt_count == 0);
  }
  AMDF_EXPECT(native_state.dependency_failure_count == 0);
  AMDF_EXPECT(allocator_state.live_allocation_count == 0);
  AMDF_EXPECT(allocator_state.invalid_alignment_count == 0);
  g_fake_native_state = nullptr;
  return passed;
}

bool AllocationFlagsPreserveIndependentPolicies() {
  bool passed = true;
  FakeNativeState native_state;
  TestAllocatorState allocator_state;
  native_state.expected_allocations = {{
      4096,
      AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_SYSTEM,
      AMDF_DXG_PRIVATE_ALLOCATION_FLAG_FINE_GRAIN |
          AMDF_DXG_PRIVATE_ALLOCATION_FLAG_DEVICE_UNCACHED |
          AMDF_DXG_PRIVATE_ALLOCATION_FLAG_QUEUE_STORAGE,
  }};
  native_state.adapter_info.native_queue_flags =
      AMDF_DXG_PRIVATE_NATIVE_QUEUE_AQL;
  g_fake_native_state = &native_state;
  {
    amdf_dxg_provider_gpu_adapter_t adapter;
    InitializeAdapter(&adapter, &allocator_state, native_state);
    alignas(4096) std::array<uint8_t, 4096> host_storage = {};
    amdf_dxg_provider_gpu_allocation_create_info_t create_info =
        MakeSystemCreateInfo(host_storage.data(), host_storage.size());
    create_info.flags = AMDF_DXG_PROVIDER_GPU_ALLOCATION_FLAG_FINE_GRAIN |
                        AMDF_DXG_PROVIDER_GPU_ALLOCATION_FLAG_QUEUE_STORAGE |
                        AMDF_DXG_PROVIDER_GPU_ALLOCATION_FLAG_DEVICE_UNCACHED;
    uint32_t allocation_handle = 0;
    uint32_t resource_handle = 0;
    uint32_t allocation_count = 0;
    uint32_t native_status = 0;

    AMDF_EXPECT(amdf::wddm_provider::GpuAllocationCreate(
                    &adapter, &create_info, 1, &allocation_handle,
                    &resource_handle, &allocation_count,
                    &native_status) == AMDF_DXG_PROVIDER_RESULT_SUCCESS);
    AMDF_EXPECT(native_state.dependency_failure_count == 0);
    AMDF_EXPECT(native_state.allocation_info_count == 1);
  }
  AMDF_EXPECT(allocator_state.live_allocation_count == 0);
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
    amdf_dxg_provider_gpu_adapter_t adapter;
    InitializeAdapter(&adapter, &allocator_state, native_state);
    alignas(4096) std::array<uint8_t, 4096> host_storage = {};
    const amdf_dxg_provider_gpu_allocation_create_info_t create_info =
        MakeSystemCreateInfo(host_storage.data(), host_storage.size());
    uint32_t allocation_handle = 0xA0;
    uint32_t resource_handle = 0xA1;
    uint32_t allocation_count = 0xA2;
    uint32_t native_status = 0xA3;

    AMDF_EXPECT(amdf::wddm_provider::GpuAllocationCreate(
                    &adapter, &create_info, 1, &allocation_handle,
                    &resource_handle, &allocation_count,
                    &native_status) == AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE);
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
    amdf_dxg_provider_gpu_adapter_t adapter;
    InitializeAdapter(&adapter, &allocator_state, native_state);
    alignas(4096) std::array<uint8_t, 4096> host_storage = {};
    const amdf_dxg_provider_gpu_allocation_create_info_t create_info =
        MakeSystemCreateInfo(host_storage.data(), host_storage.size());
    uint32_t resource_handle = 0xA1;
    uint32_t allocation_count = 0;
    uint32_t native_status = 0xA3;
    AMDF_EXPECT(amdf::wddm_provider::GpuAllocationCreate(
                    &adapter, &create_info, 0, nullptr, &resource_handle,
                    &allocation_count, &native_status) ==
                AMDF_DXG_PROVIDER_RESULT_BUFFER_TOO_SMALL);
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
  amdf_dxg_provider_gpu_adapter_t adapter;
  uint32_t allocation_count = 0xA0;
  uint64_t maximum_allocation_byte_length = 0xA1;

  AMDF_EXPECT(
      amdf::wddm_provider::GpuAllocationQueryLayout(
          &adapter, 1, &allocation_count, &maximum_allocation_byte_length) ==
      AMDF_DXG_PROVIDER_RESULT_INVALID_ARGUMENT);
  AMDF_EXPECT(allocation_count == 0xA0);
  AMDF_EXPECT(maximum_allocation_byte_length == 0xA1);

  AMDF_EXPECT(
      amdf::wddm_provider::GpuAllocationQueryLayout(
          &adapter, 4096, &allocation_count, &maximum_allocation_byte_length) ==
      AMDF_DXG_PROVIDER_RESULT_SUCCESS);
  AMDF_EXPECT(allocation_count == 1);
  AMDF_EXPECT(maximum_allocation_byte_length ==
              UINT64_C(2) * 1024 * 1024 * 1024);
  return passed;
}

bool NativeChunksPreserveExactPayloadExtent() {
  bool passed = true;
  constexpr uint64_t kChunkSize = UINT64_C(2) * 1024 * 1024 * 1024;
  constexpr uint64_t kAddress = UINT64_C(0x100000);
  struct TestCase {
    // Complete page-aligned native payload requested by the caller.
    uint64_t byte_length;
    // Expected native records, including the final partial chunk.
    std::vector<NativeAllocationRequest> allocations;
  };
  const TestCase cases[] = {
      {kChunkSize - 4096,
       {{kChunkSize - 4096, AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_LOCAL}}},
      {kChunkSize, {{kChunkSize, AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_LOCAL}}},
      {kChunkSize + 4096,
       {{kChunkSize, AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_LOCAL},
        {4096, AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_LOCAL}}},
      {2 * kChunkSize + 65536,
       {{kChunkSize, AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_LOCAL},
        {kChunkSize, AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_LOCAL},
        {65536, AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_LOCAL}}},
  };
  for (const TestCase& test_case : cases) {
    FakeNativeState native_state;
    TestAllocatorState allocator_state;
    native_state.expected_allocations = test_case.allocations;
    g_fake_native_state = &native_state;
    amdf_dxg_provider_gpu_adapter_t adapter;
    InitializeAdapter(&adapter, &allocator_state, native_state);
    uint32_t allocation_count = 0;
    uint64_t maximum_allocation_byte_length = 0;
    AMDF_EXPECT(amdf::wddm_provider::GpuAllocationQueryLayout(
                    &adapter, test_case.byte_length, &allocation_count,
                    &maximum_allocation_byte_length) ==
                AMDF_DXG_PROVIDER_RESULT_SUCCESS);
    AMDF_EXPECT(allocation_count == test_case.allocations.size());
    AMDF_EXPECT(maximum_allocation_byte_length == kChunkSize);
    AMDF_EXPECT(allocator_state.allocation_count == 0);
    AMDF_EXPECT(native_state.create_attempt_count == 0);

    amdf_dxg_provider_gpu_allocation_create_info_t create_info = {};
    create_info.device_handle = kDeviceHandle;
    create_info.domain = AMDF_DXG_PROVIDER_GPU_ALLOCATION_DOMAIN_LOCAL;
    create_info.byte_length = test_case.byte_length;
    create_info.placement_device_address = kAddress;
    std::array<uint32_t, 3> handles = {};
    uint32_t resource_handle = 0;
    uint32_t native_status = 0;
    AMDF_EXPECT(amdf::wddm_provider::GpuAllocationCreate(
                    &adapter, &create_info,
                    static_cast<uint32_t>(handles.size()), handles.data(),
                    &resource_handle, &allocation_count,
                    &native_status) == AMDF_DXG_PROVIDER_RESULT_SUCCESS);
    AMDF_EXPECT(allocation_count == test_case.allocations.size());
    AMDF_EXPECT(native_state.allocation_info_count ==
                test_case.allocations.size());
    AMDF_EXPECT(resource_handle == kResourceHandle);
    AMDF_EXPECT(native_status == 0);
    AMDF_EXPECT(native_state.create_attempt_count == 1);
    AMDF_EXPECT(native_state.dependency_failure_count == 0);
    AMDF_EXPECT(allocator_state.live_allocation_count == 0);
    AMDF_EXPECT(allocator_state.invalid_alignment_count == 0);
    for (uint32_t i = 0; i < allocation_count && i < handles.size(); ++i) {
      AMDF_EXPECT(handles[i] == kAllocationHandle + i);
    }
  }
  g_fake_native_state = nullptr;
  return passed;
}

bool MetadataAllocationFailureLeavesOwnershipOutputsUnchanged() {
  bool passed = true;
  FakeNativeState native_state;
  TestAllocatorState allocator_state;
  allocator_state.failure_ordinal = 0;
  g_fake_native_state = &native_state;
  {
    amdf_dxg_provider_gpu_adapter_t adapter;
    InitializeAdapter(&adapter, &allocator_state, native_state);
    alignas(4096) std::array<uint8_t, 4096> host_storage = {};
    const amdf_dxg_provider_gpu_allocation_create_info_t create_info =
        MakeSystemCreateInfo(host_storage.data(), host_storage.size());
    uint32_t allocation_handle = 0xA0;
    uint32_t resource_handle = 0xA1;
    uint32_t allocation_count = 0xA2;
    uint32_t native_status = 0xA3;

    AMDF_EXPECT(amdf::wddm_provider::GpuAllocationCreate(
                    &adapter, &create_info, 1, &allocation_handle,
                    &resource_handle, &allocation_count, &native_status) ==
                AMDF_DXG_PROVIDER_RESULT_RESOURCE_EXHAUSTED);
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
      {"AllocationFlagsPreserveIndependentPolicies",
       AllocationFlagsPreserveIndependentPolicies},
      {"NativeCreateFailureLeavesOutputsUnchanged",
       NativeCreateFailureLeavesOutputsUnchanged},
      {"QueryLayoutPublishesOnlyOnSuccess", QueryLayoutPublishesOnlyOnSuccess},
      {"NativeChunksPreserveExactPayloadExtent",
       NativeChunksPreserveExactPayloadExtent},
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
