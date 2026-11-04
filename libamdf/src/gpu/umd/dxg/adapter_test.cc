// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/adapter.h"

#include <cstdint>
#include <cstring>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"

struct amdf_dxg_provider_gpu_adapter_t {};
struct amdf_dxg_provider_gpu_kernel_queue_t {};

namespace {

struct FakeProviderState {
  amdf_dxg_provider_result_t open_result = AMDF_DXG_PROVIDER_RESULT_SUCCESS;
  amdf_dxg_provider_result_t close_result = AMDF_DXG_PROVIDER_RESULT_SUCCESS;
  uint32_t native_status = 0;
  amdf_dxg_provider_gpu_adapter_t native_adapter;
};

FakeProviderState state;

amdf_dxg_provider_result_t GpuAdapterOpen(
    const amdf_dxg_provider_gpu_adapter_open_info_t*, const amdf_allocator_t*,
    amdf_dxg_provider_gpu_adapter_t** out_adapter,
    amdf_dxg_provider_gpu_properties_t* out_properties,
    uint32_t* out_native_status) {
  if (state.open_result == AMDF_DXG_PROVIDER_RESULT_SUCCESS) {
    *out_adapter = &state.native_adapter;
    *out_properties = {
        .gfx_ip_major = 11,
        .gfx_ip_minor = 5,
    };
  }
  *out_native_status = state.native_status;
  return state.open_result;
}

amdf_dxg_provider_result_t GpuAdapterClose(amdf_dxg_provider_gpu_adapter_t*,
                                           uint32_t* out_native_status) {
  *out_native_status = state.native_status;
  return state.close_result;
}

amdf_dxg_provider_result_t GpuClockSample(
    amdf_dxg_provider_gpu_adapter_t*, uint32_t, uint32_t,
    amdf_dxg_provider_gpu_queue_command_type_t,
    amdf_dxg_provider_gpu_clock_sample_t*, uint32_t* out_native_status) {
  *out_native_status = 0;
  return AMDF_DXG_PROVIDER_RESULT_UNSUPPORTED;
}

const amdf_dxg_provider_api_t api = {
    .gpu_adapter_open = GpuAdapterOpen,
    .gpu_adapter_close = GpuAdapterClose,
    .gpu_clock_sample = GpuClockSample,
};

class DxgAdapterTest : public ::testing::Test {
 protected:
  void SetUp() override { state = {}; }
};

TEST_F(DxgAdapterTest, FailedInitializationLeavesOutputsUnchanged) {
  state.open_result = AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE;
  state.native_status = 0xC0000001u;
  const amdf_gpu_dxg_adapter_t sentinel_adapter = {
      reinterpret_cast<const amdf_dxg_provider_api_t*>(uintptr_t{1}),
      reinterpret_cast<amdf_dxg_provider_gpu_adapter_t*>(uintptr_t{2}),
  };
  amdf_gpu_dxg_adapter_t adapter = sentinel_adapter;
  amdf_dxg_provider_gpu_properties_t properties = {.gfx_ip_major = 99};
  const amdf_dxg_provider_gpu_properties_t sentinel_properties = properties;
  const amdf_dxg_provider_gpu_adapter_open_info_t open_info = {
      .adapter_handle = 8,
  };

  EXPECT_EQ(
      amdf_gpu_dxg_adapter_initialize(&api, &open_info, amdf_allocator_system(),
                                      &adapter, &properties),
      amdf_make_status(AMDF_STATUS_DOMAIN_NTSTATUS, state.native_status));
  EXPECT_EQ(adapter.api, sentinel_adapter.api);
  EXPECT_EQ(adapter.native, sentinel_adapter.native);
  EXPECT_EQ(std::memcmp(&properties, &sentinel_properties, sizeof(properties)),
            0);
}

TEST_F(DxgAdapterTest, RetainsLiveStateUntilCloseSucceeds) {
  const amdf_dxg_provider_gpu_adapter_open_info_t open_info = {
      .adapter_handle = 8,
  };
  amdf_gpu_dxg_adapter_t adapter = {};
  amdf_dxg_provider_gpu_properties_t properties = {};
  ASSERT_EQ(
      amdf_gpu_dxg_adapter_initialize(&api, &open_info, amdf_allocator_system(),
                                      &adapter, &properties),
      AMDF_STATUS_OK);
  EXPECT_EQ(adapter.api, &api);
  EXPECT_EQ(adapter.native, &state.native_adapter);
  EXPECT_EQ(properties.gfx_ip_major, 11);
  EXPECT_EQ(properties.gfx_ip_minor, 5);

  const amdf_dxg_provider_gpu_clock_sample_t sentinel_sample = {17, 19, 23};
  auto sample = sentinel_sample;
  EXPECT_EQ(amdf_gpu_dxg_adapter_query_clock_sample(
                &adapter, 8, 0, AMDF_DXG_PROVIDER_GPU_QUEUE_COMMAND_TYPE_PM4,
                &sample),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  EXPECT_EQ(std::memcmp(&sample, &sentinel_sample, sizeof(sample)), 0);

  state.close_result = AMDF_DXG_PROVIDER_RESULT_BUSY;
  EXPECT_EQ(amdf_gpu_dxg_adapter_deinitialize(&adapter),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  EXPECT_EQ(adapter.api, &api);
  EXPECT_EQ(adapter.native, &state.native_adapter);

  state.close_result = AMDF_DXG_PROVIDER_RESULT_SUCCESS;
  EXPECT_EQ(amdf_gpu_dxg_adapter_deinitialize(&adapter), AMDF_STATUS_OK);
  EXPECT_EQ(adapter.api, nullptr);
  EXPECT_EQ(adapter.native, nullptr);
}

}  // namespace
