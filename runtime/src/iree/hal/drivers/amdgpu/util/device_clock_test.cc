// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/device_clock.h"

#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

TEST(DeviceClockTest, ValidateCounters) {
  iree_hal_amdgpu_device_clock_counters_t counters = {
      /*.device_clock_counter=*/1,
      /*.host_cpu_timestamp_ns=*/2,
      /*.host_system_timestamp=*/3,
      /*.host_system_frequency_hz=*/4,
  };
  IREE_EXPECT_OK(
      iree_hal_amdgpu_device_clock_counters_validate(1234, &counters));

  counters.device_clock_counter = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_amdgpu_device_clock_counters_validate(1234, &counters));
  counters.device_clock_counter = 1;

  counters.host_cpu_timestamp_ns = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_amdgpu_device_clock_counters_validate(1234, &counters));
  counters.host_cpu_timestamp_ns = 2;

  counters.host_system_timestamp = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_amdgpu_device_clock_counters_validate(1234, &counters));
  counters.host_system_timestamp = 3;

  counters.host_system_frequency_hz = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_amdgpu_device_clock_counters_validate(1234, &counters));
}

TEST(DeviceClockTest, UnavailableSourceSampleFailsExplicitly) {
  iree_hal_amdgpu_device_clock_source_t source = {
      /*.type=*/IREE_HAL_AMDGPU_DEVICE_CLOCK_SOURCE_TYPE_UNAVAILABLE,
      /*.state=*/{},
  };
  iree_hal_amdgpu_device_clock_counters_t counters = {
      /*.device_clock_counter=*/1,
      /*.host_cpu_timestamp_ns=*/2,
      /*.host_system_timestamp=*/3,
      /*.host_system_frequency_hz=*/4,
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_amdgpu_device_clock_source_sample(
          &source, /*libhsa=*/nullptr, hsa_agent_t{0}, 1234, &counters));
  EXPECT_EQ(counters.device_clock_counter, 0);
  EXPECT_EQ(counters.host_cpu_timestamp_ns, 0);
  EXPECT_EQ(counters.host_system_timestamp, 0);
  EXPECT_EQ(counters.host_system_frequency_hz, 0);
}

struct DeviceClockLiveTest : public ::testing::Test {
  static iree_allocator_t host_allocator;
  static iree_hal_amdgpu_libhsa_t libhsa;
  static iree_hal_amdgpu_topology_t topology;

  static void SetUpTestSuite() {
    host_allocator = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator, &libhsa);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(
        iree_hal_amdgpu_topology_initialize_with_defaults(&libhsa, &topology));
    if (topology.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
  }
};
iree_allocator_t DeviceClockLiveTest::host_allocator;
iree_hal_amdgpu_libhsa_t DeviceClockLiveTest::libhsa;
iree_hal_amdgpu_topology_t DeviceClockLiveTest::topology;

TEST_F(DeviceClockLiveTest, PlatformSourceSamplesClockCounters) {
  uint32_t driver_uid = 0;
  IREE_ASSERT_OK(iree_hsa_agent_get_info(
      IREE_LIBHSA(&libhsa), topology.gpu_agents[0],
      (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DRIVER_UID, &driver_uid));

  iree_hal_amdgpu_device_clock_source_t source = {
      /*.type=*/IREE_HAL_AMDGPU_DEVICE_CLOCK_SOURCE_TYPE_UNAVAILABLE,
      /*.state=*/{},
  };
  IREE_ASSERT_OK(iree_hal_amdgpu_device_clock_source_initialize(&source));

#if defined(IREE_PLATFORM_WINDOWS)
  const iree_time_t host_time_begin_ns = iree_time_now();
#endif  // IREE_PLATFORM_WINDOWS
  iree_hal_amdgpu_device_clock_counters_t counters = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_device_clock_source_sample(
      &source, &libhsa, topology.gpu_agents[0], driver_uid, &counters));
#if defined(IREE_PLATFORM_WINDOWS)
  const iree_time_t host_time_end_ns = iree_time_now();
#endif  // IREE_PLATFORM_WINDOWS

  EXPECT_NE(counters.device_clock_counter, 0);
  EXPECT_NE(counters.host_cpu_timestamp_ns, 0);
  EXPECT_NE(counters.host_system_timestamp, 0);
  EXPECT_NE(counters.host_system_frequency_hz, 0);
#if defined(IREE_PLATFORM_WINDOWS)
  EXPECT_LE(host_time_begin_ns, counters.host_cpu_timestamp_ns);
  EXPECT_LE(counters.host_cpu_timestamp_ns, host_time_end_ns);
  EXPECT_LE(host_time_begin_ns, counters.host_system_timestamp);
  EXPECT_LE(counters.host_system_timestamp, host_time_end_ns);
#endif  // IREE_PLATFORM_WINDOWS

  iree_hal_amdgpu_device_clock_source_deinitialize(&source);
}

}  // namespace
}  // namespace iree::hal::amdgpu
