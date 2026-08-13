// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device_spec_builder.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

// Builds physical-device params for the given |processor| arch ticking at
// |timestamp_frequency_hz|. Everything is synthetic, so the spec builder is
// exercised without creating a device or querying an agent.
static void MakePhysicalDeviceParams(
    iree_string_view_t processor, uint64_t timestamp_frequency_hz,
    uint32_t physical_ordinal,
    iree_hal_amdgpu_device_spec_physical_device_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  iree_hal_amdgpu_target_identity_t identity = {};
  IREE_ASSERT_OK(
      iree_hal_amdgpu_target_identity_parse_processor(processor, &identity));

  iree_hal_amdgpu_device_spec_physical_device_params_t physical_device = {
      /*.identity=*/identity,
      /*.uuid=*/{{0x11}},
      /*.pci=*/{/*.domain=*/0, /*.bus=*/3, /*.device=*/0, /*.function=*/0},
      /*.timestamp_frequency_hz=*/timestamp_frequency_hz,
      /*.numa=*/{/*.node_id=*/1},
      /*.physical_ordinal=*/physical_ordinal,
      /*.queue_count=*/2,
      /*.compute_unit_count=*/40,
      /*.wavefront_size=*/64,
      /*.maximum_waves_per_compute_unit=*/32,
      /*.maximum_workgroup_local_memory_size=*/64 * 1024,
      /*.flags=*/IREE_HAL_AMDGPU_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_UUID |
          IREE_HAL_AMDGPU_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_PCI_ADDRESS,
  };
  *out_params = physical_device;
}

// Builds device spec params covering |physical_devices|.
static iree_hal_amdgpu_device_spec_params_t MakeDeviceSpecParams(
    const iree_hal_amdgpu_device_spec_physical_device_params_t*
        physical_devices,
    iree_host_size_t physical_device_count, iree_hal_allocator_t* allocator) {
  iree_hal_amdgpu_device_spec_params_t params = {
      /*.logical_device_id=*/IREE_SV("amdgpu://0"),
      /*.display_name=*/IREE_SV("AMDGPU test device"),
      /*.physical_device_count=*/physical_device_count,
      /*.physical_devices=*/physical_devices,
      /*.device_memory_capacity_bytes=*/64ull * 1024ull * 1024ull * 1024ull,
      /*.device_allocator=*/allocator,
      /*.sanitizer=*/{},
      /*.flags=*/IREE_HAL_AMDGPU_DEVICE_SPEC_PARAM_FLAG_DMABUF,
  };
  return params;
}

// Builds a device spec for a single physical device of the given |processor|
// arch ticking at |timestamp_frequency_hz|.
static void CreateDeviceSpecForProcessor(
    iree_string_view_t processor, uint64_t timestamp_frequency_hz,
    iree_hal_allocator_t* allocator, iree_hal_device_spec_t** out_device_spec) {
  iree_hal_amdgpu_device_spec_physical_device_params_t physical_device;
  MakePhysicalDeviceParams(processor, timestamp_frequency_hz,
                           /*physical_ordinal=*/7, &physical_device);
  const iree_hal_amdgpu_device_spec_params_t params =
      MakeDeviceSpecParams(&physical_device, 1, allocator);
  IREE_ASSERT_OK(iree_hal_amdgpu_device_spec_create(
      &params, iree_allocator_system(), out_device_spec));
}

// Builds a device spec for two gfx942 physical devices ticking at the given
// rates.
static iree_status_t CreateTwoDeviceSpec(
    uint64_t first_timestamp_frequency_hz,
    uint64_t second_timestamp_frequency_hz, iree_hal_allocator_t* allocator,
    iree_hal_device_spec_t** out_device_spec) {
  iree_hal_amdgpu_device_spec_physical_device_params_t physical_devices[2];
  MakePhysicalDeviceParams(IREE_SV("gfx942"), first_timestamp_frequency_hz,
                           /*physical_ordinal=*/0, &physical_devices[0]);
  MakePhysicalDeviceParams(IREE_SV("gfx942"), second_timestamp_frequency_hz,
                           /*physical_ordinal=*/1, &physical_devices[1]);
  const iree_hal_amdgpu_device_spec_params_t params = MakeDeviceSpecParams(
      physical_devices, IREE_ARRAYSIZE(physical_devices), allocator);
  return iree_hal_amdgpu_device_spec_create(&params, iree_allocator_system(),
                                            out_device_spec);
}

// The GPU agent wallclock rate on CDNA3 parts.
static constexpr uint64_t kAgentTimestampFrequencyHz = 100000000ull;

// gfx8 falls outside the gfx9-gfx12 range the PM4 strategy table covers, and
// the advertised timestamp domain does not depend on that strategy.
TEST(DeviceSpecBuilderTest, AdvertisesTheTimestampDomainWithoutPm4Support) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(IREE_SV("test"), iree_allocator_system(),
                                     iree_allocator_system(), &allocator));

  iree_hal_device_spec_t* device_spec = NULL;
  CreateDeviceSpecForProcessor(IREE_SV("gfx803"), kAgentTimestampFrequencyHz,
                               allocator, &device_spec);

  const iree_hal_device_timing_spec_t* timing =
      iree_hal_device_spec_timing(device_spec);
  ASSERT_NE(timing, nullptr);
  EXPECT_TRUE(iree_all_bits_set(
      timing->flags, IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS));
  EXPECT_EQ(timing->timestamp_frequency_hz, kAgentTimestampFrequencyHz);
  EXPECT_EQ(timing->timestamp_valid_bits, 64u);

  iree_hal_device_spec_release(device_spec);
  iree_hal_allocator_release(allocator);
}

// The advertised rate is the physical device's own, at both device and
// queue-family scope.
TEST(DeviceSpecBuilderTest, AdvertisesThePhysicalDeviceTickRate) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(IREE_SV("test"), iree_allocator_system(),
                                     iree_allocator_system(), &allocator));

  iree_hal_device_spec_t* device_spec = NULL;
  CreateDeviceSpecForProcessor(IREE_SV("gfx942"), kAgentTimestampFrequencyHz,
                               allocator, &device_spec);

  const iree_hal_device_timing_spec_t* timing =
      iree_hal_device_spec_timing(device_spec);
  ASSERT_NE(timing, nullptr);
  EXPECT_TRUE(iree_all_bits_set(
      timing->flags, IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS));
  EXPECT_EQ(timing->timestamp_frequency_hz, kAgentTimestampFrequencyHz);
  EXPECT_EQ(timing->timestamp_valid_bits, 64u);

  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(device_spec);
  ASSERT_NE(queues, nullptr);
  ASSERT_EQ(queues->family_count, 1);
  EXPECT_EQ(queues->families[0].queue_affinity, 3u);
  EXPECT_EQ(queues->families[0].timestamp_frequency_hz,
            kAgentTimestampFrequencyHz);
  EXPECT_EQ(queues->families[0].timestamp_valid_bits, 64u);

  iree_hal_device_spec_release(device_spec);
  iree_hal_allocator_release(allocator);
}

// Each queue family reports the tick rate of the physical device it covers
// rather than a single device-wide value.
TEST(DeviceSpecBuilderTest, QueueFamiliesReportPerPhysicalDeviceFrequency) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(IREE_SV("test"), iree_allocator_system(),
                                     iree_allocator_system(), &allocator));

  iree_hal_device_spec_t* device_spec = NULL;
  IREE_ASSERT_OK(CreateTwoDeviceSpec(kAgentTimestampFrequencyHz,
                                     kAgentTimestampFrequencyHz, allocator,
                                     &device_spec));

  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(device_spec);
  ASSERT_NE(queues, nullptr);
  ASSERT_EQ(queues->family_count, 2);
  EXPECT_EQ(queues->families[0].queue_affinity, 3u);
  EXPECT_EQ(queues->families[1].queue_affinity, 12u);
  EXPECT_EQ(queues->families[0].timestamp_frequency_hz,
            kAgentTimestampFrequencyHz);
  EXPECT_EQ(queues->families[0].timestamp_valid_bits, 64u);
  EXPECT_EQ(queues->families[1].timestamp_frequency_hz,
            kAgentTimestampFrequencyHz);
  EXPECT_EQ(queues->families[1].timestamp_valid_bits, 64u);

  iree_hal_device_spec_release(device_spec);
  iree_hal_allocator_release(allocator);
}

// Several physical devices ticking at one rate publish that rate as the
// device-scope rate. Equal rates are a common scale, not one counter: each
// agent keeps its own epoch.
TEST(DeviceSpecBuilderTest, UniformPhysicalDevicesPublishOneDeviceScopeRate) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(IREE_SV("test"), iree_allocator_system(),
                                     iree_allocator_system(), &allocator));

  iree_hal_device_spec_t* device_spec = NULL;
  IREE_ASSERT_OK(CreateTwoDeviceSpec(kAgentTimestampFrequencyHz,
                                     kAgentTimestampFrequencyHz, allocator,
                                     &device_spec));

  const iree_hal_device_timing_spec_t* timing =
      iree_hal_device_spec_timing(device_spec);
  ASSERT_NE(timing, nullptr);
  EXPECT_TRUE(iree_all_bits_set(
      timing->flags, IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS));
  EXPECT_EQ(timing->timestamp_frequency_hz, kAgentTimestampFrequencyHz);
  EXPECT_EQ(timing->timestamp_valid_bits, 64u);

  iree_hal_device_spec_release(device_spec);
  iree_hal_allocator_release(allocator);
}

// The device-scope timing spec holds one tick rate for the whole logical
// device, so physical devices that disagree have no correct value to report.
TEST(DeviceSpecBuilderTest, MismatchedTimestampFrequencyIsRejected) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(IREE_SV("test"), iree_allocator_system(),
                                     iree_allocator_system(), &allocator));

  iree_hal_device_spec_t* device_spec = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateTwoDeviceSpec(kAgentTimestampFrequencyHz,
                                            kAgentTimestampFrequencyHz * 2,
                                            allocator, &device_spec));
  EXPECT_EQ(device_spec, nullptr);

  iree_hal_allocator_release(allocator);
}

// A physical device with no tick rate cannot describe a timestamp domain at
// all; the builder rejects it instead of publishing a zero divisor.
TEST(DeviceSpecBuilderTest, ZeroTimestampFrequencyIsRejected) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(IREE_SV("test"), iree_allocator_system(),
                                     iree_allocator_system(), &allocator));

  iree_hal_amdgpu_device_spec_physical_device_params_t physical_device;
  MakePhysicalDeviceParams(IREE_SV("gfx942"), /*timestamp_frequency_hz=*/0,
                           /*physical_ordinal=*/0, &physical_device);
  const iree_hal_amdgpu_device_spec_params_t params =
      MakeDeviceSpecParams(&physical_device, 1, allocator);
  iree_hal_device_spec_t* device_spec = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_device_spec_create(
                            &params, iree_allocator_system(), &device_spec));
  EXPECT_EQ(device_spec, nullptr);

  iree_hal_allocator_release(allocator);
}

// A zero rate anywhere in the params is rejected by the zero check, which runs
// first; only a zero beside a nonzero rate is also a frequency mismatch.
TEST(DeviceSpecBuilderTest, ZeroTimestampFrequencyOnSecondDeviceIsRejected) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(IREE_SV("test"), iree_allocator_system(),
                                     iree_allocator_system(), &allocator));

  iree_hal_device_spec_t* device_spec = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateTwoDeviceSpec(kAgentTimestampFrequencyHz,
                                            /*second_timestamp_frequency_hz=*/0,
                                            allocator, &device_spec));
  EXPECT_EQ(device_spec, nullptr);

  iree_hal_allocator_release(allocator);
}

}  // namespace
}  // namespace iree::hal::amdgpu
