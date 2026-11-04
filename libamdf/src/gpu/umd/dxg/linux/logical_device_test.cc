// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/linux/logical_device.h"

#include <sys/ioctl.h>

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"

namespace {

enum class Operation {
  kCreateDevice,
  kOpenBridgeAdapter,
  kCloseBridgeAdapter,
  kDestroyDevice,
};

struct FakeState {
  // Native device handle returned by the create operation.
  D3DKMT_HANDLE device = 0x10;
  // Bridge result returned while opening private adapter state.
  amdf_dxg_bridge_result_t bridge_open_result = AMDF_DXG_BRIDGE_RESULT_SUCCESS;
  // Ordered native and bridge lifecycle operations.
  std::vector<Operation> operations;
};

FakeState* current_state = nullptr;
amdf_linux_dxg_connection_t* current_connection = nullptr;
amdf_linux_dxg_kmt_t* current_kmt = nullptr;
amdf_platform_endpoint_t* current_endpoint = nullptr;

extern "C" uint32_t __wrap_amdf_linux_dxg_connection_invoke(void* context,
                                                            uint64_t request,
                                                            void* arguments) {
  EXPECT_EQ(context, current_connection);
  switch (_IOC_NR(request)) {
    case 0x02: {
      current_state->operations.push_back(Operation::kCreateDevice);
      auto* create = static_cast<D3DKMT_CREATEDEVICE*>(arguments);
      EXPECT_EQ(create->hAdapter, current_endpoint->adapter);
      create->hDevice = current_state->device;
      return 0;
    }
    case 0x19: {
      current_state->operations.push_back(Operation::kDestroyDevice);
      const auto* destroy = static_cast<D3DKMT_DESTROYDEVICE*>(arguments);
      EXPECT_EQ(destroy->hDevice, current_state->device);
      return 0;
    }
    default:
      ADD_FAILURE() << "unexpected DXG operation " << _IOC_NR(request);
      return 0xC0000001u;
  }
}

amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL FakeAdapterOpen(
    const amdf_dxg_bridge_gpu_adapter_open_info_t* open_info,
    const amdf_allocator_t*, amdf_dxg_bridge_gpu_adapter_t** out_adapter,
    amdf_dxg_bridge_gpu_properties_t* out_properties, uint32_t*) {
  current_state->operations.push_back(Operation::kOpenBridgeAdapter);
  EXPECT_EQ(open_info->adapter_handle, current_endpoint->adapter);
  EXPECT_EQ(open_info->device_handle, current_state->device);
  EXPECT_EQ(open_info->physical_adapter_index,
            current_endpoint->physical_adapter_index);
  EXPECT_EQ(open_info->adapter_luid,
            current_endpoint->info.native_identity.value.windows_adapter.luid);
  EXPECT_EQ(open_info->native_transport.context, current_kmt);
  EXPECT_EQ(open_info->native_transport.invoke, amdf_linux_dxg_kmt_invoke);
  if (current_state->bridge_open_result != AMDF_DXG_BRIDGE_RESULT_SUCCESS) {
    return current_state->bridge_open_result;
  }
  *out_adapter = reinterpret_cast<amdf_dxg_bridge_gpu_adapter_t*>(uintptr_t{1});
  *out_properties = {
      .gfx_ip_major = 11,
      .gfx_ip_minor = 5,
      .gfx_ip_stepping = 0,
  };
  return AMDF_DXG_BRIDGE_RESULT_SUCCESS;
}

amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL
FakeAdapterClose(amdf_dxg_bridge_gpu_adapter_t* adapter, uint32_t*) {
  current_state->operations.push_back(Operation::kCloseBridgeAdapter);
  EXPECT_EQ(adapter,
            reinterpret_cast<amdf_dxg_bridge_gpu_adapter_t*>(uintptr_t{1}));
  return AMDF_DXG_BRIDGE_RESULT_SUCCESS;
}

const amdf_dxg_bridge_api_t kBridgeApi = {
    .structure_size = sizeof(amdf_dxg_bridge_api_t),
    .abi_version = AMDF_DXG_BRIDGE_ABI_VERSION_13,
    .gpu_adapter_open = FakeAdapterOpen,
    .gpu_adapter_close = FakeAdapterClose,
};

class LinuxDxgLogicalDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    current_state = &state_;
    current_kmt = &instance_.kmt;
    current_endpoint = &endpoint_;
    instance_.host_allocator = amdf_allocator_system();
    instance_.connection.descriptor = 1;
    instance_.kmt.connection = &instance_.connection;
    instance_.kmt.host_allocator = &instance_.host_allocator;
    current_connection = &instance_.connection;
    endpoint_.instance = &instance_;
    endpoint_.adapter = 0x08;
    endpoint_.physical_adapter_index = 2;
    endpoint_.info.native_identity.type =
        AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_WINDOWS_ADAPTER;
    endpoint_.info.native_identity.value.windows_adapter.luid =
        UINT64_C(0x1122334455667788);
    endpoint_.info.native_identity.value.windows_adapter
        .physical_adapter_index = endpoint_.physical_adapter_index;
  }

  void TearDown() override {
    current_endpoint = nullptr;
    current_kmt = nullptr;
    current_connection = nullptr;
    current_state = nullptr;
  }

  // Native lifecycle behavior selected by the active test.
  FakeState state_;
  // Platform instance supplying the borrowed typed transport.
  amdf_platform_instance_t instance_ = {};
  // Endpoint carrying the exact adapter identity into the bridge.
  amdf_platform_endpoint_t endpoint_ = {};
};

TEST_F(LinuxDxgLogicalDeviceTest, OwnsLogicalAndBridgeAdaptersTogether) {
  amdf_gpu_linux_dxg_logical_device_t device = {};
  amdf_dxg_bridge_gpu_properties_t properties = {};
  ASSERT_EQ(amdf_gpu_linux_dxg_logical_device_initialize(
                &kBridgeApi, &endpoint_, instance_.host_allocator, &device,
                &properties),
            AMDF_STATUS_OK);

  EXPECT_EQ(device.native, state_.device);
  EXPECT_NE(device.bridge_adapter.native, nullptr);
  EXPECT_EQ(properties.gfx_ip_major, 11);
  EXPECT_EQ(properties.gfx_ip_minor, 5);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kCreateDevice,
                                    Operation::kOpenBridgeAdapter}));

  amdf_gpu_linux_dxg_logical_device_deinitialize(&device);
  EXPECT_EQ(device.native, 0u);
  EXPECT_EQ(device.bridge_adapter.native, nullptr);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kCreateDevice, Operation::kOpenBridgeAdapter,
                Operation::kCloseBridgeAdapter, Operation::kDestroyDevice}));
}

TEST_F(LinuxDxgLogicalDeviceTest, BridgeRejectionConsumesLogicalDevice) {
  state_.bridge_open_result = AMDF_DXG_BRIDGE_RESULT_UNSUPPORTED;
  auto* sentinel =
      reinterpret_cast<amdf_dxg_bridge_gpu_adapter_t*>(uintptr_t{1});
  amdf_gpu_linux_dxg_logical_device_t device = {
      .bridge_adapter = {.native = sentinel},
  };
  amdf_dxg_bridge_gpu_properties_t properties = {.gfx_ip_major = 99};

  EXPECT_EQ(amdf_gpu_linux_dxg_logical_device_initialize(
                &kBridgeApi, &endpoint_, instance_.host_allocator, &device,
                &properties),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  EXPECT_EQ(device.kmt, nullptr);
  EXPECT_EQ(device.native, 0u);
  EXPECT_EQ(device.bridge_adapter.native, sentinel);
  EXPECT_EQ(properties.gfx_ip_major, 99);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kCreateDevice,
                                    Operation::kOpenBridgeAdapter,
                                    Operation::kDestroyDevice}));
}

TEST_F(LinuxDxgLogicalDeviceTest, RejectsMissingNativeDevice) {
  state_.device = 0;
  amdf_gpu_linux_dxg_logical_device_t device = {};
  amdf_dxg_bridge_gpu_properties_t properties = {};

  EXPECT_EQ(amdf_gpu_linux_dxg_logical_device_initialize(
                &kBridgeApi, &endpoint_, instance_.host_allocator, &device,
                &properties),
            amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL));
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kCreateDevice}));
}

}  // namespace
