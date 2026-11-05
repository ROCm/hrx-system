// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/clock.h"

#include <cstdint>
#include <cstring>

#include "gtest/gtest.h"

struct amdf_dxg_provider_gpu_adapter_t {};

namespace {

struct FakeState {
  amdf_dxg_provider_result_t result = AMDF_DXG_PROVIDER_RESULT_SUCCESS;
  uint32_t native_status = 0;
  uint32_t adapter_handle = 0;
  uint32_t physical_adapter_index = 0;
  amdf_dxg_provider_gpu_queue_command_type_t command_type =
      AMDF_DXG_PROVIDER_GPU_QUEUE_COMMAND_TYPE_PM4;
  amdf_dxg_provider_gpu_clock_sample_t sample = {
      .frequency_hz = 27'000'000,
      .timestamp = 31,
      .host_timestamp = 37,
  };
  uint64_t host_timestamps[2] = {41, 43};
  uint32_t host_query_count = 0;
};

FakeState state;
amdf_dxg_provider_gpu_adapter_t native_adapter;

amdf_dxg_provider_result_t GpuClockSample(
    amdf_dxg_provider_gpu_adapter_t*, uint32_t adapter_handle,
    uint32_t physical_adapter_index,
    amdf_dxg_provider_gpu_queue_command_type_t command_type,
    amdf_dxg_provider_gpu_clock_sample_t* out_sample,
    uint32_t* out_native_status) {
  state.adapter_handle = adapter_handle;
  state.physical_adapter_index = physical_adapter_index;
  state.command_type = command_type;
  if (state.result == AMDF_DXG_PROVIDER_RESULT_SUCCESS) {
    *out_sample = state.sample;
  }
  *out_native_status = state.native_status;
  return state.result;
}

const amdf_dxg_provider_api_t api = {
    .gpu_clock_sample = GpuClockSample,
};

class DxgClockTest : public ::testing::Test {
 protected:
  void SetUp() override {
    state = {};
    adapter_ = {
        .api = &api,
        .native = &native_adapter,
    };
  }

  amdf_gpu_dxg_adapter_t adapter_ = {};
};

TEST_F(DxgClockTest, ReportsQueueClockRate) {
  amdf_observation_clock_info_t info = {};
  ASSERT_EQ(amdf_gpu_dxg_query_queue_clock_info(
                &adapter_, 17, 3, AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, &info),
            AMDF_STATUS_OK);
  EXPECT_EQ(state.adapter_handle, 17u);
  EXPECT_EQ(state.physical_adapter_index, 3u);
  EXPECT_EQ(state.command_type, AMDF_DXG_PROVIDER_GPU_QUEUE_COMMAND_TYPE_PM4);
  EXPECT_EQ(info.frequency_hz, state.sample.frequency_hz);

  ASSERT_EQ(amdf_gpu_dxg_query_queue_clock_info(
                &adapter_, 19, 5, AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA, &info),
            AMDF_STATUS_OK);
  EXPECT_EQ(state.adapter_handle, 19u);
  EXPECT_EQ(state.physical_adapter_index, 5u);
  EXPECT_EQ(state.command_type, AMDF_DXG_PROVIDER_GPU_QUEUE_COMMAND_TYPE_SDMA);
}

TEST_F(DxgClockTest, CorrelatesDeviceAndHostClocks) {
  amdf_observation_clock_sample_t sample = {};
  ASSERT_EQ(amdf_gpu_dxg_query_clock_sample(
                &adapter_, 23, 7, AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA, &sample),
            AMDF_STATUS_OK);
  EXPECT_EQ(sample.clock.frequency_hz, state.sample.frequency_hz);
  EXPECT_EQ(sample.timestamp, state.sample.timestamp);
  EXPECT_EQ(sample.host.before, state.host_timestamps[0]);
  EXPECT_EQ(sample.host.timestamp, state.sample.host_timestamp);
  EXPECT_EQ(sample.host.after, state.host_timestamps[1]);
  EXPECT_EQ(state.host_query_count, 2u);
}

TEST_F(DxgClockTest, FailurePreservesOutput) {
  state.result = AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE;
  state.native_status = 0xC0000001u;
  amdf_observation_clock_sample_t sample = {
      .clock = {.frequency_hz = 47},
      .timestamp = 53,
  };
  const amdf_observation_clock_sample_t original = sample;
  EXPECT_EQ(amdf_gpu_dxg_query_clock_sample(
                &adapter_, 29, 11, AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, &sample),
            amdf_make_status(AMDF_STATUS_DOMAIN_NTSTATUS, state.native_status));
  EXPECT_EQ(std::memcmp(&sample, &original, sizeof(sample)), 0);
  EXPECT_EQ(state.host_query_count, 1u);
}

}  // namespace

extern "C" amdf_status_t __wrap_amdf_platform_observation_query_host_clock(
    uint64_t* out_timestamp) {
  *out_timestamp = state.host_timestamps[state.host_query_count++];
  return AMDF_STATUS_OK;
}
