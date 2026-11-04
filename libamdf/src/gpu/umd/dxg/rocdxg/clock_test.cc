// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/rocdxg/clock.h"

#include <sys/ioctl.h>
#include <time.h>

#include <cstdint>

#include "gtest/gtest.h"
#include "libamdf/src/gpu/umd/dxg/rocdxg/adapter_state.h"
#include "shared/include/d3dkmt_types.h"

namespace {

constexpr NTSTATUS kNativeFailure = static_cast<NTSTATUS>(0xC0000001u);
uint32_t host_query_count = 0;
int32_t host_failure_ordinal = -1;

class RocdxgClockTest : public ::testing::Test {
 protected:
  void SetUp() override {
    host_query_count = 0;
    host_failure_ordinal = -1;
    adapter_.native_transport = {
        .context = this,
        .invoke = Invoke,
    };
    adapter_.pm4_queue_profile = {
        .node_ordinal = 3,
        .is_supported = true,
    };
    adapter_.sdma_queue_profile = {
        .node_ordinal = 7,
        .is_supported = true,
    };
  }

  static uint32_t Invoke(void* context, uint64_t request, void* arguments) {
    return static_cast<RocdxgClockTest*>(context)->Invoke(request, arguments);
  }

  uint32_t Invoke(uint64_t request, void* arguments) {
    EXPECT_EQ(_IOC_TYPE(request), static_cast<unsigned long>('G'));
    EXPECT_EQ(_IOC_NR(request), 0x3Du);
    EXPECT_EQ(_IOC_SIZE(request), sizeof(D3DKMT_QUERYCLOCKCALIBRATION));
    auto* query = static_cast<D3DKMT_QUERYCLOCKCALIBRATION*>(arguments);
    EXPECT_EQ(query->hAdapter, 0x42u);
    EXPECT_EQ(query->NodeOrdinal, expected_node_ordinal_);
    EXPECT_EQ(query->PhysicalAdapterIndex, 2u);
    if (native_status_ == STATUS_SUCCESS) {
      query->ClockData.GpuFrequency = frequency_hz_;
      query->ClockData.GpuClockCounter = 123;
      query->ClockData.CpuClockCounter = 456;
      query->ClockData.Flags.ContextManagementProcessor = context_clock_;
    }
    return static_cast<uint32_t>(native_status_);
  }

  amdf_dxg_bridge_result_t Sample(
      amdf_dxg_bridge_gpu_queue_command_type_t command_type,
      amdf_dxg_bridge_gpu_clock_sample_t* out_sample,
      uint32_t* out_native_status) {
    expected_node_ordinal_ =
        command_type == AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4 ? 3 : 7;
    return amdf::rocdxg_bridge::GpuClockSample(&adapter_, 0x42, 2, command_type,
                                               out_sample, out_native_status);
  }

  amdf_dxg_bridge_gpu_adapter_t adapter_ = {};
  uint32_t expected_node_ordinal_ = 0;
  NTSTATUS native_status_ = STATUS_SUCCESS;
  uint64_t frequency_hz_ = 27'000'000;
  bool context_clock_ = false;
};

TEST_F(RocdxgClockTest, SamplesExactQueueEngineClocks) {
  for (const auto command_type : {
           AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4,
           AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_SDMA,
       }) {
    host_query_count = 0;
    amdf_dxg_bridge_gpu_clock_sample_t sample = {};
    uint32_t native_status = 99;
    EXPECT_EQ(Sample(command_type, &sample, &native_status),
              AMDF_DXG_BRIDGE_RESULT_SUCCESS);
    EXPECT_EQ(native_status, 0u);
    EXPECT_EQ(sample.frequency_hz, frequency_hz_);
    EXPECT_EQ(sample.timestamp, 123u);
    EXPECT_EQ(sample.host_timestamp, 110u);
  }
}

TEST_F(RocdxgClockTest, PreservesOutputWhenHostClockFails) {
  for (const int32_t failure_ordinal : {0, 1}) {
    SCOPED_TRACE(failure_ordinal);
    host_query_count = 0;
    host_failure_ordinal = failure_ordinal;
    amdf_dxg_bridge_gpu_clock_sample_t sample = {11, 13, 17};
    uint32_t native_status = 99;
    EXPECT_EQ(Sample(AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4, &sample,
                     &native_status),
              AMDF_DXG_BRIDGE_RESULT_INTERNAL);
    EXPECT_EQ(native_status, 0u);
    EXPECT_EQ(sample.frequency_hz, 11u);
    EXPECT_EQ(sample.timestamp, 13u);
    EXPECT_EQ(sample.host_timestamp, 17u);
  }
}

TEST_F(RocdxgClockTest, PreservesUnsupportedAndNativeFailureResults) {
  amdf_dxg_bridge_gpu_clock_sample_t sample = {11, 13, 17};
  uint32_t native_status = 99;

  context_clock_ = true;
  EXPECT_EQ(Sample(AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4, &sample,
                   &native_status),
            AMDF_DXG_BRIDGE_RESULT_UNSUPPORTED);
  EXPECT_EQ(native_status, 0u);
  EXPECT_EQ(sample.frequency_hz, 11u);

  context_clock_ = false;
  frequency_hz_ = 0;
  EXPECT_EQ(Sample(AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_SDMA, &sample,
                   &native_status),
            AMDF_DXG_BRIDGE_RESULT_UNSUPPORTED);
  EXPECT_EQ(sample.frequency_hz, 11u);

  native_status_ = kNativeFailure;
  EXPECT_EQ(Sample(AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4, &sample,
                   &native_status),
            AMDF_DXG_BRIDGE_RESULT_NATIVE_FAILURE);
  EXPECT_EQ(native_status, static_cast<uint32_t>(kNativeFailure));
  EXPECT_EQ(sample.frequency_hz, 11u);
}

TEST_F(RocdxgClockTest, RejectsUnavailableCommandFamilies) {
  adapter_.sdma_queue_profile.is_supported = false;
  amdf_dxg_bridge_gpu_clock_sample_t sample = {11, 13, 17};
  uint32_t native_status = 99;
  EXPECT_EQ(Sample(AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_SDMA, &sample,
                   &native_status),
            AMDF_DXG_BRIDGE_RESULT_UNSUPPORTED);
  EXPECT_EQ(native_status, 0u);
  EXPECT_EQ(sample.frequency_hz, 11u);
}

}  // namespace

extern "C" int __wrap_clock_gettime(clockid_t clock_id, timespec* out_time) {
  EXPECT_EQ(clock_id, CLOCK_MONOTONIC_RAW);
  if (static_cast<int32_t>(host_query_count) == host_failure_ordinal) {
    ++host_query_count;
    return -1;
  }
  const uint64_t timestamp =
      100 + static_cast<uint64_t>(host_query_count++) * 20;
  out_time->tv_sec = static_cast<time_t>(timestamp / 1'000'000'000u);
  out_time->tv_nsec = static_cast<long>(timestamp % 1'000'000'000u);
  return 0;
}
