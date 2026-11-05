// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/linux/device.h"

#include <sys/ioctl.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/dxg/linux/api.h"
#include "libamdf/src/gpu/umd/dxg/wsl/api.h"

namespace {

constexpr NTSTATUS kNativeFailure = static_cast<NTSTATUS>(0xC0000001u);

enum class Operation {
  kQueryMemoryCapabilities,
  kQueryStoreCapabilities,
  kGetBridgeApi,
  kCreateDevice,
  kOpenBridgeAdapter,
  kCreatePagingQueue,
  kDestroyPagingQueue,
  kCloseBridgeAdapter,
  kDestroyDevice,
};

enum class FailureStage {
  kNone,
  kMemoryCapabilities,
  kStoreCapabilities,
  kCreateDevice,
  kOpenBridgeAdapter,
  kCreatePagingQueue,
  kMalformedPagingQueue,
};

struct FakeState {
  // Construction stage that returns an injected failure.
  FailureStage failure_stage = FailureStage::kNone;
  // Number of paging-queue destruction calls that report a terminal failure.
  uint32_t paging_destroy_failures_remaining = 0;
  // Host allocations still owned by device bookkeeping.
  uint32_t live_allocation_count = 0;
  // Monotonic paging progress returned with the native queue.
  volatile uint64_t paging_progress = 0;
  // Ordered native and provider operations.
  std::vector<Operation> operations;
};

FakeState* current_state = nullptr;
amdf_platform_endpoint_t* current_endpoint = nullptr;
amdf_linux_dxg_kmt_t* current_kmt = nullptr;

void* AMDF_CALL Allocate(void* user_data, uint64_t byte_length,
                         uint64_t minimum_alignment) {
  auto* state = static_cast<FakeState*>(user_data);
  const amdf_allocator_t system = amdf_allocator_system();
  void* allocation =
      system.allocate(system.user_data, byte_length, minimum_alignment);
  if (allocation != nullptr) {
    ++state->live_allocation_count;
  }
  return allocation;
}

void AMDF_CALL Free(void* user_data, void* allocation) {
  if (allocation == nullptr) {
    return;
  }
  auto* state = static_cast<FakeState*>(user_data);
  EXPECT_NE(state->live_allocation_count, 0u);
  --state->live_allocation_count;
  const amdf_allocator_t system = amdf_allocator_system();
  system.free(system.user_data, allocation);
}

extern "C" uint32_t __wrap_amdf_linux_dxg_connection_invoke(void* context,
                                                            uint64_t request,
                                                            void* arguments) {
  EXPECT_EQ(context, current_kmt->connection);
  switch (_IOC_NR(request)) {
    case 0x02: {
      current_state->operations.push_back(Operation::kCreateDevice);
      if (current_state->failure_stage == FailureStage::kCreateDevice) {
        return static_cast<uint32_t>(kNativeFailure);
      }
      auto* create = static_cast<D3DKMT_CREATEDEVICE*>(arguments);
      EXPECT_EQ(create->hAdapter, current_endpoint->adapter);
      create->hDevice = 0x10;
      return 0;
    }
    case 0x07: {
      current_state->operations.push_back(Operation::kCreatePagingQueue);
      if (current_state->failure_stage == FailureStage::kCreatePagingQueue) {
        return static_cast<uint32_t>(kNativeFailure);
      }
      auto* create = static_cast<D3DKMT_CREATEPAGINGQUEUE*>(arguments);
      EXPECT_EQ(create->hDevice, 0x10u);
      EXPECT_EQ(create->PhysicalAdapterIndex,
                current_endpoint->physical_adapter_index);
      EXPECT_EQ(create->Priority, D3DDDI_PAGINGQUEUE_PRIORITY_NORMAL);
      create->hPagingQueue = 0x20;
      create->hSyncObject =
          current_state->failure_stage == FailureStage::kMalformedPagingQueue
              ? 0
              : 0x30;
      create->FenceValueCPUVirtualAddress =
          const_cast<uint64_t*>(&current_state->paging_progress);
      return 0;
    }
    case 0x09: {
      auto* query = static_cast<D3DKMT_QUERYADAPTERINFO*>(arguments);
      EXPECT_EQ(query->hAdapter, current_endpoint->adapter);
      if (query->Type == KMTQAITYPE_QUERY_GPUMMU_CAPS) {
        current_state->operations.push_back(
            Operation::kQueryMemoryCapabilities);
        EXPECT_EQ(query->PrivateDriverDataSize,
                  sizeof(D3DKMT_QUERY_GPUMMU_CAPS));
        if (current_state->failure_stage == FailureStage::kMemoryCapabilities) {
          return static_cast<uint32_t>(kNativeFailure);
        }
        auto* memory =
            static_cast<D3DKMT_QUERY_GPUMMU_CAPS*>(query->pPrivateDriverData);
        EXPECT_EQ(memory->PhysicalAdapterIndex,
                  current_endpoint->physical_adapter_index);
        memory->Caps.VirtualAddressBitCount = 48;
        memory->Caps.Flags.ReadOnlyMemorySupported = 1;
        memory->Caps.Flags.NoExecuteMemorySupported = 1;
        memory->Caps.Flags.CacheCoherentMemorySupported = 1;
        return 0;
      }
      current_state->operations.push_back(Operation::kQueryStoreCapabilities);
      EXPECT_EQ(query->Type, KMTQAITYPE_WDDM_2_0_CAPS);
      EXPECT_EQ(query->PrivateDriverDataSize, sizeof(D3DKMT_WDDM_2_0_CAPS));
      if (current_state->failure_stage == FailureStage::kStoreCapabilities) {
        return static_cast<uint32_t>(kNativeFailure);
      }
      auto* scheduling =
          static_cast<D3DKMT_WDDM_2_0_CAPS*>(query->pPrivateDriverData);
      scheduling->Support64BitAtomics = 1;
      return 0;
    }
    case 0x19: {
      current_state->operations.push_back(Operation::kDestroyDevice);
      const auto* destroy = static_cast<D3DKMT_DESTROYDEVICE*>(arguments);
      EXPECT_EQ(destroy->hDevice, 0x10u);
      return 0;
    }
    case 0x1C: {
      current_state->operations.push_back(Operation::kDestroyPagingQueue);
      const auto* destroy = static_cast<D3DDDI_DESTROYPAGINGQUEUE*>(arguments);
      EXPECT_EQ(destroy->hPagingQueue, 0x20u);
      if (current_state->paging_destroy_failures_remaining != 0) {
        --current_state->paging_destroy_failures_remaining;
        return static_cast<uint32_t>(kNativeFailure);
      }
      return 0;
    }
    default:
      ADD_FAILURE() << "unexpected DXG operation " << _IOC_NR(request);
      return static_cast<uint32_t>(kNativeFailure);
  }
}

amdf_dxg_provider_result_t FakeAdapterOpen(
    const amdf_dxg_provider_gpu_adapter_open_info_t* open_info,
    const amdf_allocator_t*, amdf_dxg_provider_gpu_adapter_t** out_adapter,
    amdf_dxg_provider_gpu_properties_t* out_properties,
    uint32_t* out_native_status) {
  current_state->operations.push_back(Operation::kOpenBridgeAdapter);
  EXPECT_EQ(open_info->adapter_handle, current_endpoint->adapter);
  EXPECT_EQ(open_info->device_handle, 0x10u);
  EXPECT_EQ(open_info->physical_adapter_index,
            current_endpoint->physical_adapter_index);
  EXPECT_EQ(open_info->adapter_luid,
            current_endpoint->info.native_identity.value.windows_adapter.luid);
  EXPECT_EQ(open_info->native_transport.context, current_kmt);
  EXPECT_EQ(open_info->native_transport.invoke, amdf_linux_dxg_kmt_invoke);
  *out_native_status = 0;
  if (current_state->failure_stage == FailureStage::kOpenBridgeAdapter) {
    *out_native_status = static_cast<uint32_t>(kNativeFailure);
    return AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE;
  }
  *out_adapter =
      reinterpret_cast<amdf_dxg_provider_gpu_adapter_t*>(uintptr_t{1});
  *out_properties = {
      .gfx_ip_major = 11,
      .gfx_ip_minor = 5,
      .is_discrete = 0,
      .platform_atomic_support = 1,
  };
  return AMDF_DXG_PROVIDER_RESULT_SUCCESS;
}

amdf_dxg_provider_result_t FakeAdapterClose(
    amdf_dxg_provider_gpu_adapter_t* adapter, uint32_t* out_native_status) {
  current_state->operations.push_back(Operation::kCloseBridgeAdapter);
  EXPECT_EQ(adapter,
            reinterpret_cast<amdf_dxg_provider_gpu_adapter_t*>(uintptr_t{1}));
  *out_native_status = 0;
  return AMDF_DXG_PROVIDER_RESULT_SUCCESS;
}

const amdf_dxg_provider_api_t kProviderApi = {
    .gpu_adapter_open = FakeAdapterOpen,
    .gpu_adapter_close = FakeAdapterClose,
};

extern "C" const amdf_dxg_provider_api_t* __wrap_amdf_gpu_dxg_wsl_default_api(
    void) {
  current_state->operations.push_back(Operation::kGetBridgeApi);
  return &kProviderApi;
}

class LinuxDxgDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    current_state = &state_;
    instance_.host_allocator = {
        .user_data = &state_,
        .allocate = Allocate,
        .free = Free,
    };
    instance_.connection.descriptor = 1;
    instance_.kmt.connection = &instance_.connection;
    instance_.kmt.host_allocator = &instance_.host_allocator;
    current_kmt = &instance_.kmt;
    endpoint_.instance = &instance_;
    endpoint_.adapter = 0x08;
    endpoint_.physical_adapter_index = 2;
    endpoint_.info.id.words[0] = UINT64_C(0x0123456789ABCDEF);
    endpoint_.info.native_identity.type =
        AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_WINDOWS_ADAPTER;
    endpoint_.info.native_identity.value.windows_adapter.luid =
        UINT64_C(0x1122334455667788);
    endpoint_.info.native_identity.value.windows_adapter
        .physical_adapter_index = endpoint_.physical_adapter_index;
    current_endpoint = &endpoint_;
  }

  void TearDown() override {
    EXPECT_EQ(state_.live_allocation_count, 0u);
    current_endpoint = nullptr;
    current_kmt = nullptr;
    current_state = nullptr;
  }

  // Native behavior and ordered observations selected by the active test.
  FakeState state_;
  // Platform instance owning the injected direct DXG transport.
  amdf_platform_instance_t instance_ = {};
  // Endpoint borrowing one query adapter from the platform instance.
  amdf_platform_endpoint_t endpoint_ = {};
};

TEST_F(LinuxDxgDeviceTest, OwnsCompleteLogicalAndPagingDevice) {
  amdf_gpu_umd_device_t* device = nullptr;
  amdf_gpu_umd_device_result_t result = {};

  ASSERT_EQ(amdf_gpu_linux_dxg_device_create(
                nullptr, &endpoint_, instance_.host_allocator,
                AMDF_NATIVE_LIFETIME_INSTANCE, &device, &result),
            AMDF_STATUS_OK);

  ASSERT_NE(device, nullptr);
  EXPECT_EQ(device->logical_device.native, 0x10u);
  EXPECT_EQ(device->paging_queue.native, 0x20u);
  EXPECT_EQ(device->paging_queue.synchronization_object, 0x30u);
  EXPECT_EQ(device->memory_capabilities.virtual_address_bit_count, 48u);
  EXPECT_EQ(device->memory_capabilities.atomic_update_64_supported, 1u);
  EXPECT_EQ(device->memory_capabilities.d3d12_resource_supported, 0u);
  EXPECT_EQ(device->memory_capabilities.program_gfx_major, 11u);
  EXPECT_EQ(device->memory_capabilities.program_host_is_fabric, 1u);
  EXPECT_EQ(device->memory_capabilities.program_host_rmw_supported, 1u);
  EXPECT_EQ(result.id.words[0], endpoint_.info.id.words[0]);
  EXPECT_EQ(result.id.words[1], UINT64_C(0x0000002000000010));
  EXPECT_EQ(result.reset_epoch, 1u);
  EXPECT_EQ(result.features, AMDF_GPU_DEVICE_FEATURE_HOST_REGISTRATION |
                                 AMDF_GPU_DEVICE_FEATURE_DEVICE_RECREATION |
                                 AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kQueryMemoryCapabilities,
                Operation::kQueryStoreCapabilities, Operation::kGetBridgeApi,
                Operation::kCreateDevice, Operation::kOpenBridgeAdapter,
                Operation::kCreatePagingQueue}));

  ASSERT_EQ(amdf_gpu_linux_dxg_device_destroy(device), AMDF_STATUS_OK);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kQueryMemoryCapabilities,
                Operation::kQueryStoreCapabilities, Operation::kGetBridgeApi,
                Operation::kCreateDevice, Operation::kOpenBridgeAdapter,
                Operation::kCreatePagingQueue, Operation::kDestroyPagingQueue,
                Operation::kCloseBridgeAdapter, Operation::kDestroyDevice}));
}

class LinuxDxgDeviceConstructionFailureTest
    : public LinuxDxgDeviceTest,
      public ::testing::WithParamInterface<FailureStage> {};

TEST_P(LinuxDxgDeviceConstructionFailureTest,
       RollsBackInReverseOrderAndPreservesOutputs) {
  state_.failure_stage = GetParam();
  auto* device = reinterpret_cast<amdf_gpu_umd_device_t*>(uintptr_t{1});
  amdf_gpu_umd_device_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const amdf_gpu_umd_device_result_t original_result = result;

  EXPECT_FALSE(amdf_status_is_ok(amdf_gpu_linux_dxg_device_create(
      nullptr, &endpoint_, instance_.host_allocator,
      AMDF_NATIVE_LIFETIME_PROCESS, &device, &result)));
  EXPECT_EQ(device, reinterpret_cast<amdf_gpu_umd_device_t*>(uintptr_t{1}));
  EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);

  std::vector<Operation> expected = {Operation::kQueryMemoryCapabilities};
  if (GetParam() != FailureStage::kMemoryCapabilities) {
    expected.push_back(Operation::kQueryStoreCapabilities);
  }
  if (GetParam() > FailureStage::kStoreCapabilities) {
    expected.push_back(Operation::kGetBridgeApi);
    expected.push_back(Operation::kCreateDevice);
  }
  if (GetParam() > FailureStage::kCreateDevice) {
    expected.push_back(Operation::kOpenBridgeAdapter);
  }
  if (GetParam() > FailureStage::kOpenBridgeAdapter) {
    expected.push_back(Operation::kCreatePagingQueue);
  }
  if (GetParam() == FailureStage::kMalformedPagingQueue) {
    expected.push_back(Operation::kDestroyPagingQueue);
  }
  if (GetParam() > FailureStage::kOpenBridgeAdapter) {
    expected.push_back(Operation::kCloseBridgeAdapter);
  }
  if (GetParam() > FailureStage::kCreateDevice) {
    expected.push_back(Operation::kDestroyDevice);
  }
  EXPECT_EQ(state_.operations, expected);
}

INSTANTIATE_TEST_SUITE_P(
    NativeStage, LinuxDxgDeviceConstructionFailureTest,
    ::testing::Values(
        FailureStage::kMemoryCapabilities, FailureStage::kStoreCapabilities,
        FailureStage::kCreateDevice, FailureStage::kOpenBridgeAdapter,
        FailureStage::kCreatePagingQueue, FailureStage::kMalformedPagingQueue));

TEST_F(LinuxDxgDeviceTest, ConsumedPagingFailureRequiresOnlyHostOwnerRetry) {
  amdf_gpu_umd_device_t* device = nullptr;
  amdf_gpu_umd_device_result_t result = {};
  ASSERT_EQ(amdf_gpu_linux_dxg_device_create(
                nullptr, &endpoint_, instance_.host_allocator,
                AMDF_NATIVE_LIFETIME_PROCESS, &device, &result),
            AMDF_STATUS_OK);
  state_.paging_destroy_failures_remaining = 1;

  EXPECT_EQ(amdf_gpu_linux_dxg_device_destroy(device),
            amdf_linux_dxg_kmt_make_status(kNativeFailure));
  const size_t operation_count = state_.operations.size();
  EXPECT_EQ(state_.live_allocation_count, 1u);
  EXPECT_EQ(state_.operations[operation_count - 3],
            Operation::kDestroyPagingQueue);
  EXPECT_EQ(state_.operations[operation_count - 2],
            Operation::kCloseBridgeAdapter);
  EXPECT_EQ(state_.operations[operation_count - 1], Operation::kDestroyDevice);

  EXPECT_EQ(amdf_gpu_linux_dxg_device_destroy(device), AMDF_STATUS_OK);
  EXPECT_EQ(state_.operations.size(), operation_count);
}

}  // namespace
