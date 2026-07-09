// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device_spec_builder.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

// Builds a device spec for a single physical device of the given |processor|
// arch with the given HSA |timestamp_frequency_hz|. The spec is GPU-free: it is
// derived purely from the synthetic params (target_id.version), so this can run
// on a normal CPU box.
static void CreateDeviceSpecForProcessor(
    iree_string_view_t processor, uint64_t timestamp_frequency_hz,
    iree_hal_allocator_t* allocator, iree_hal_device_spec_t** out_device_spec) {
  iree_hal_amdgpu_target_id_t target_id = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_target_id_parse(
      processor, IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_ARCH_ONLY,
      &target_id));

  iree_hal_amdgpu_device_spec_physical_device_params_t physical_device = {
      /*.target_id=*/target_id,
      /*.uuid=*/{{0x11}},
      /*.pci=*/{/*.domain=*/0, /*.bus=*/3, /*.device=*/0, /*.function=*/0},
      /*.numa=*/{/*.node_id=*/1},
      /*.physical_ordinal=*/7,
      /*.queue_count=*/2,
      /*.compute_unit_count=*/40,
      /*.wavefront_size=*/64,
      /*.maximum_workgroup_local_memory_size=*/64 * 1024,
      /*.flags=*/IREE_HAL_AMDGPU_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_UUID |
          IREE_HAL_AMDGPU_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_PCI_ADDRESS,
  };
  iree_hal_amdgpu_device_spec_params_t params = {
      /*.logical_device_id=*/IREE_SV("amdgpu://0"),
      /*.display_name=*/IREE_SV("AMDGPU test device"),
      /*.timestamp_frequency_hz=*/timestamp_frequency_hz,
      /*.physical_device_count=*/1,
      /*.physical_devices=*/&physical_device,
      /*.device_memory_capacity_bytes=*/64ull * 1024ull * 1024ull * 1024ull,
      /*.device_allocator=*/allocator,
      /*.sanitizer=*/{},
      /*.flags=*/IREE_HAL_AMDGPU_DEVICE_SPEC_PARAM_FLAG_DMABUF,
  };
  IREE_ASSERT_OK(iree_hal_amdgpu_device_spec_create(
      &params, iree_allocator_system(), out_device_spec));
}

// On an arch whose PM4 timestamp strategy is NONE (gfx8) the builder must clear
// DEVICE_TIMESTAMPS and report a device-scope timestamp frequency of zero so
// the capability query never over-promises.
TEST(DeviceSpecBuilderTest, UnsupportedArchClearsDeviceTimestamps) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(IREE_SV("test"), iree_allocator_system(),
                                     iree_allocator_system(), &allocator));

  iree_hal_device_spec_t* device_spec = NULL;
  CreateDeviceSpecForProcessor(IREE_SV("gfx803"), 1000000000ull, allocator,
                               &device_spec);

  const iree_hal_device_timing_spec_t* timing =
      iree_hal_device_spec_timing(device_spec);
  ASSERT_NE(timing, nullptr);
  EXPECT_FALSE(iree_all_bits_set(
      timing->flags, IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS));
  EXPECT_EQ(timing->timestamp_frequency_hz, 0ull);

  iree_hal_device_spec_release(device_spec);
  iree_hal_allocator_release(allocator);
}

// On a supported arch (gfx942) the gating must set DEVICE_TIMESTAMPS and
// advertise the requested device-scope timestamp frequency.
TEST(DeviceSpecBuilderTest, SupportedArchSetsDeviceTimestamps) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(IREE_SV("test"), iree_allocator_system(),
                                     iree_allocator_system(), &allocator));

  iree_hal_device_spec_t* device_spec = NULL;
  CreateDeviceSpecForProcessor(IREE_SV("gfx942"), 1000000000ull, allocator,
                               &device_spec);

  const iree_hal_device_timing_spec_t* timing =
      iree_hal_device_spec_timing(device_spec);
  ASSERT_NE(timing, nullptr);
  EXPECT_TRUE(iree_all_bits_set(
      timing->flags, IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS));
  EXPECT_EQ(timing->timestamp_frequency_hz, 1000000000ull);

  iree_hal_device_spec_release(device_spec);
  iree_hal_allocator_release(allocator);
}

}  // namespace
}  // namespace iree::hal::amdgpu
