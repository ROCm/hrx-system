// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/device.h"

#include <malloc.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/windows/endpoint.h"

namespace {

constexpr NTSTATUS kSuccess = 0;
constexpr NTSTATUS kFailure = static_cast<NTSTATUS>(0xC0000001u);

enum class Operation {
  kQueryMemoryCapabilities,
  kCreateDevice,
  kCreatePagingQueue,
  kDestroyPagingQueue,
  kDestroyDevice,
  kCloseAdapter,
};

struct FakeKmtState {
  // Native result returned by the GPU memory-capability query.
  NTSTATUS memory_capabilities_status = kSuccess;
  // Number of paging-queue releases rejected before native consumption.
  uint32_t paging_queue_destroy_failures_remaining = 1;
  // Sync-object handle returned with the paging queue, or zero for malformed
  // output.
  D3DKMT_HANDLE paging_sync_object = 0;
  // Ordered native KMT operations used to verify local and published ownership.
  std::vector<Operation> operations;
  // Host allocations still owned by endpoint/device bookkeeping.
  uint32_t live_allocation_count = 0;
  // Number of successfully released paging queues.
  uint32_t paging_queue_destroy_success_count = 0;
  // Number of successfully released logical devices.
  uint32_t device_destroy_success_count = 0;
  // Number of successfully closed endpoint adapters.
  uint32_t adapter_close_success_count = 0;
  // Mapped paging progress returned with the malformed queue result.
  volatile uint64_t paging_progress = 0;
};

FakeKmtState* current_state = nullptr;

void* AMDF_CALL Allocate(void* user_data, uint64_t byte_length,
                         uint64_t minimum_alignment) {
  auto* state = static_cast<FakeKmtState*>(user_data);
  void* pointer = _aligned_malloc(static_cast<size_t>(byte_length),
                                  static_cast<size_t>(minimum_alignment));
  if (pointer != nullptr) ++state->live_allocation_count;
  return pointer;
}

void AMDF_CALL Free(void* user_data, void* allocation) {
  if (allocation == nullptr) return;
  auto* state = static_cast<FakeKmtState*>(user_data);
  EXPECT_NE(state->live_allocation_count, 0u);
  --state->live_allocation_count;
  _aligned_free(allocation);
}

NTSTATUS APIENTRY FakeQueryAdapterInfo(const D3DKMT_QUERYADAPTERINFO* query) {
  current_state->operations.push_back(Operation::kQueryMemoryCapabilities);
  EXPECT_EQ(query->hAdapter, 0x08u);
  EXPECT_EQ(query->Type, KMTQAITYPE_QUERY_GPUMMU_CAPS);
  EXPECT_EQ(query->PrivateDriverDataSize, sizeof(D3DKMT_QUERY_GPUMMU_CAPS));
  if (current_state->memory_capabilities_status != kSuccess) {
    return current_state->memory_capabilities_status;
  }
  auto* gpu_mmu =
      static_cast<D3DKMT_QUERY_GPUMMU_CAPS*>(query->pPrivateDriverData);
  EXPECT_EQ(gpu_mmu->PhysicalAdapterIndex, 0u);
  gpu_mmu->Caps.Flags.ReadOnlyMemorySupported = 1;
  gpu_mmu->Caps.Flags.NoExecuteMemorySupported = 1;
  gpu_mmu->Caps.Flags.CacheCoherentMemorySupported = 1;
  gpu_mmu->Caps.VirtualAddressBitCount = 48;
  return kSuccess;
}

NTSTATUS APIENTRY FakeCreateDevice(D3DKMT_CREATEDEVICE* create) {
  current_state->operations.push_back(Operation::kCreateDevice);
  EXPECT_EQ(create->hAdapter, 0x08u);
  create->hDevice = 0x10;
  return kSuccess;
}

NTSTATUS APIENTRY FakeDestroyDevice(const D3DKMT_DESTROYDEVICE* destroy) {
  current_state->operations.push_back(Operation::kDestroyDevice);
  EXPECT_EQ(destroy->hDevice, 0x10u);
  ++current_state->device_destroy_success_count;
  return kSuccess;
}

NTSTATUS APIENTRY FakeGetDeviceState(D3DKMT_GETDEVICESTATE*) {
  ADD_FAILURE() << "device-state query is not part of this construction path";
  return kFailure;
}

NTSTATUS APIENTRY FakeCreatePagingQueue(D3DKMT_CREATEPAGINGQUEUE* create) {
  current_state->operations.push_back(Operation::kCreatePagingQueue);
  EXPECT_EQ(create->hDevice, 0x10u);
  create->hPagingQueue = 0x20;
  create->hSyncObject = current_state->paging_sync_object;
  create->FenceValueCPUVirtualAddress =
      const_cast<uint64_t*>(&current_state->paging_progress);
  return kSuccess;
}

NTSTATUS APIENTRY FakeDestroyPagingQueue(D3DDDI_DESTROYPAGINGQUEUE* destroy) {
  current_state->operations.push_back(Operation::kDestroyPagingQueue);
  EXPECT_EQ(destroy->hPagingQueue, 0x20u);
  if (current_state->paging_queue_destroy_failures_remaining != 0) {
    --current_state->paging_queue_destroy_failures_remaining;
    return kFailure;
  }
  ++current_state->paging_queue_destroy_success_count;
  return kSuccess;
}

NTSTATUS APIENTRY FakeCloseAdapter(const D3DKMT_CLOSEADAPTER* close) {
  current_state->operations.push_back(Operation::kCloseAdapter);
  EXPECT_EQ(close->hAdapter, 0x08u);
  ++current_state->adapter_close_success_count;
  return kSuccess;
}

using ResetBridgeFn = void(__cdecl*)(void);
using SetBridgeCountFn = void(__cdecl*)(uint32_t);
using QueryBridgeCountFn = uint32_t(__cdecl*)(void);

class WindowsGpuDeviceRollbackTest
    : public ::testing::TestWithParam<amdf_native_lifetime_t> {
 protected:
  void SetUp() override {
    current_state = &state_;
    instance_.host_allocator = {
        .user_data = &state_, .allocate = Allocate, .free = Free};
    instance_.kmt.query_adapter_info = FakeQueryAdapterInfo;
    instance_.kmt.create_device = FakeCreateDevice;
    instance_.kmt.destroy_device = FakeDestroyDevice;
    instance_.kmt.get_device_state = FakeGetDeviceState;
    instance_.kmt.create_paging_queue = FakeCreatePagingQueue;
    instance_.kmt.destroy_paging_queue = FakeDestroyPagingQueue;
    instance_.kmt.close_adapter = FakeCloseAdapter;

    ASSERT_EQ(amdf_calloc(instance_.host_allocator, sizeof(*endpoint_),
                          amdf_alignof(amdf_platform_endpoint_t),
                          reinterpret_cast<void**>(&endpoint_)),
              AMDF_STATUS_OK);
    endpoint_->instance = &instance_;
    endpoint_->adapter = 0x08;
    endpoint_->physical_adapter_index = 0;

    const DWORD path_capacity =
        GetEnvironmentVariableW(L"AMDF_WKMI_BRIDGE_PATH", nullptr, 0);
    ASSERT_GT(path_capacity, 0u);
    std::vector<wchar_t> path(path_capacity);
    ASSERT_LT(GetEnvironmentVariableW(L"AMDF_WKMI_BRIDGE_PATH", path.data(),
                                      path_capacity),
              path_capacity);
    const DWORD absolute_capacity =
        GetFullPathNameW(path.data(), 0, nullptr, nullptr);
    ASSERT_GT(absolute_capacity, 0u);
    std::vector<wchar_t> absolute_path(absolute_capacity);
    ASSERT_LT(GetFullPathNameW(path.data(), absolute_capacity,
                               absolute_path.data(), nullptr),
              absolute_capacity);
    bridge_module_ = LoadLibraryExW(
        absolute_path.data(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    ASSERT_NE(bridge_module_, nullptr);

    reset_bridge_ = reinterpret_cast<ResetBridgeFn>(
        GetProcAddress(bridge_module_, "amdf_test_wkmi_bridge_reset"));
    set_bridge_close_failures_ = reinterpret_cast<SetBridgeCountFn>(
        GetProcAddress(bridge_module_,
                       "amdf_test_wkmi_bridge_set_adapter_close_failures"));
    query_bridge_open_success_count_ =
        reinterpret_cast<QueryBridgeCountFn>(GetProcAddress(
            bridge_module_,
            "amdf_test_wkmi_bridge_query_adapter_open_success_count"));
    query_bridge_close_attempt_count_ =
        reinterpret_cast<QueryBridgeCountFn>(GetProcAddress(
            bridge_module_,
            "amdf_test_wkmi_bridge_query_adapter_close_attempt_count"));
    query_bridge_close_success_count_ =
        reinterpret_cast<QueryBridgeCountFn>(GetProcAddress(
            bridge_module_,
            "amdf_test_wkmi_bridge_query_adapter_close_success_count"));
    ASSERT_NE(reset_bridge_, nullptr);
    ASSERT_NE(set_bridge_close_failures_, nullptr);
    ASSERT_NE(query_bridge_open_success_count_, nullptr);
    ASSERT_NE(query_bridge_close_attempt_count_, nullptr);
    ASSERT_NE(query_bridge_close_success_count_, nullptr);
    reset_bridge_();
  }

  void TearDown() override {
    if (endpoint_ != nullptr) {
      EXPECT_EQ(amdf_platform_endpoint_close(endpoint_), AMDF_STATUS_OK);
      endpoint_ = nullptr;
    }
    if (bridge_module_ != nullptr) {
      EXPECT_TRUE(FreeLibrary(bridge_module_));
      bridge_module_ = nullptr;
    }
    EXPECT_EQ(state_.live_allocation_count, 0u);
    current_state = nullptr;
  }

  // Native dependencies and construction/teardown observations.
  FakeKmtState state_;
  // Platform instance owning the injected KMT table.
  amdf_platform_instance_t instance_ = {};
  // Live endpoint owned through native construction rollback.
  amdf_platform_endpoint_t* endpoint_ = nullptr;
  // Test-owned module reference for inspecting the bridge dependency.
  HMODULE bridge_module_ = nullptr;
  // Resets bridge observations and failures before each case.
  ResetBridgeFn reset_bridge_ = nullptr;
  // Selects the number of rejected bridge adapter closes.
  SetBridgeCountFn set_bridge_close_failures_ = nullptr;
  // Queries successful native adapter opens in the bridge.
  QueryBridgeCountFn query_bridge_open_success_count_ = nullptr;
  // Queries attempted native adapter releases in the bridge.
  QueryBridgeCountFn query_bridge_close_attempt_count_ = nullptr;
  // Queries successful native adapter releases in the bridge.
  QueryBridgeCountFn query_bridge_close_success_count_ = nullptr;
};

TEST_P(WindowsGpuDeviceRollbackTest,
       CapabilityQueryFailureDoesNotPublishOrCreateNativeDevice) {
  state_.memory_capabilities_status = kFailure;
  amdf_gpu_umd_device_t* device =
      reinterpret_cast<amdf_gpu_umd_device_t*>(uintptr_t{1});
  amdf_gpu_umd_device_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const amdf_gpu_umd_device_result_t original = result;

  EXPECT_EQ(
      amdf_gpu_umd_device_create(nullptr, endpoint_, instance_.host_allocator,
                                 GetParam(), &device, &result),
      amdf_kmt_make_status(kFailure));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(device), uintptr_t{1});
  EXPECT_EQ(std::memcmp(&result, &original, sizeof(result)), 0);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kQueryMemoryCapabilities}));
  EXPECT_EQ(query_bridge_open_success_count_(), 0u);
  EXPECT_EQ(query_bridge_close_attempt_count_(), 0u);
}

TEST_P(WindowsGpuDeviceRollbackTest,
       FailedBridgeRollbackLeavesNoEndpointCleanupObligation) {
  amdf_gpu_umd_device_t* device =
      reinterpret_cast<amdf_gpu_umd_device_t*>(uintptr_t{1});
  amdf_gpu_umd_device_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const amdf_gpu_umd_device_result_t original = result;

  const amdf_status_t status =
      amdf_gpu_umd_device_create(nullptr, endpoint_, instance_.host_allocator,
                                 GetParam(), &device, &result);

  EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_BUSY);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(device), uintptr_t{1});
  EXPECT_EQ(std::memcmp(&result, &original, sizeof(result)), 0);
  EXPECT_EQ(state_.live_allocation_count, 1u);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kQueryMemoryCapabilities,
                                    Operation::kCreateDevice,
                                    Operation::kCreatePagingQueue}));
  EXPECT_EQ(query_bridge_open_success_count_(), 1u);
  EXPECT_EQ(query_bridge_close_attempt_count_(), 1u);
  EXPECT_EQ(query_bridge_close_success_count_(), 0u);
  EXPECT_EQ(state_.paging_queue_destroy_success_count, 0u);
  EXPECT_EQ(state_.device_destroy_success_count, 0u);
  EXPECT_EQ(state_.adapter_close_success_count, 0u);

  EXPECT_EQ(amdf_platform_endpoint_close(endpoint_), AMDF_STATUS_OK);
  endpoint_ = nullptr;
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kQueryMemoryCapabilities, Operation::kCreateDevice,
                Operation::kCreatePagingQueue, Operation::kCloseAdapter}));
  EXPECT_EQ(query_bridge_close_attempt_count_(), 1u);
  EXPECT_EQ(query_bridge_close_success_count_(), 0u);
  EXPECT_EQ(state_.paging_queue_destroy_success_count, 0u);
  EXPECT_EQ(state_.device_destroy_success_count, 0u);
  EXPECT_EQ(state_.adapter_close_success_count, 1u);
  EXPECT_EQ(state_.live_allocation_count, 0u);
}

TEST_P(WindowsGpuDeviceRollbackTest,
       FailedPagingRollbackLeavesNoEndpointCleanupObligation) {
  set_bridge_close_failures_(0);
  amdf_gpu_umd_device_t* device =
      reinterpret_cast<amdf_gpu_umd_device_t*>(uintptr_t{1});
  amdf_gpu_umd_device_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const amdf_gpu_umd_device_result_t original = result;
  EXPECT_EQ(
      amdf_gpu_umd_device_create(nullptr, endpoint_, instance_.host_allocator,
                                 GetParam(), &device, &result),
      amdf_kmt_make_status(kFailure));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(device), uintptr_t{1});
  EXPECT_EQ(std::memcmp(&result, &original, sizeof(result)), 0);
  EXPECT_EQ(state_.live_allocation_count, 1u);
  EXPECT_EQ(amdf_platform_endpoint_close(endpoint_), AMDF_STATUS_OK);
  endpoint_ = nullptr;
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kQueryMemoryCapabilities, Operation::kCreateDevice,
                Operation::kCreatePagingQueue, Operation::kDestroyPagingQueue,
                Operation::kCloseAdapter}));
  EXPECT_EQ(query_bridge_close_attempt_count_(), 1u);
  EXPECT_EQ(query_bridge_close_success_count_(), 1u);
  EXPECT_EQ(state_.paging_queue_destroy_success_count, 0u);
  EXPECT_EQ(state_.device_destroy_success_count, 0u);
  EXPECT_EQ(state_.live_allocation_count, 0u);
}

TEST_P(WindowsGpuDeviceRollbackTest,
       ExplicitDestroyFailurePreservesPublishedDevice) {
  set_bridge_close_failures_(0);
  state_.paging_sync_object = 0x30;
  amdf_gpu_umd_device_t* device = nullptr;
  amdf_gpu_umd_device_result_t result = {};
  ASSERT_EQ(
      amdf_gpu_umd_device_create(nullptr, endpoint_, instance_.host_allocator,
                                 GetParam(), &device, &result),
      AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_umd_device_destroy(device),
            amdf_kmt_make_status(kFailure));
  EXPECT_EQ(state_.live_allocation_count, 2u);
  EXPECT_EQ(state_.device_destroy_success_count, 0u);
  EXPECT_EQ(amdf_gpu_umd_device_destroy(device), AMDF_STATUS_OK);
  EXPECT_EQ(query_bridge_close_attempt_count_(), 1u);
  EXPECT_EQ(state_.paging_queue_destroy_success_count, 1u);
  EXPECT_EQ(state_.device_destroy_success_count, 1u);
  EXPECT_EQ(state_.live_allocation_count, 1u);
}

INSTANTIATE_TEST_SUITE_P(NativeLifetime, WindowsGpuDeviceRollbackTest,
                         ::testing::Values(AMDF_NATIVE_LIFETIME_PROCESS,
                                           AMDF_NATIVE_LIFETIME_INSTANCE));

}  // namespace
