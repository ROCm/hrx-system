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

// The GPU agent wallclock rate on CDNA3 parts.
static constexpr uint64_t kAgentTimestampFrequencyHz = 100000000ull;

static void CreateDeviceSpecForProcessor(
    iree_string_view_t processor, uint32_t wavefront_size,
    iree_hal_allocator_t* allocator, iree_hal_device_spec_t** out_device_spec) {
  iree_hal_amdgpu_target_identity_t identity = {};
  IREE_ASSERT_OK(
      iree_hal_amdgpu_target_identity_parse_artifact_key(processor, &identity));

  iree_hal_amdgpu_device_spec_physical_device_params_t physical_device = {
      /*.identity=*/identity,
      /*.uuid=*/{{0x11}},
      /*.pci=*/{/*.domain=*/0, /*.bus=*/3, /*.device=*/0, /*.function=*/0},
      /*.timestamp_frequency_hz=*/kAgentTimestampFrequencyHz,
      /*.numa=*/{/*.node_id=*/1},
      /*.physical_ordinal=*/7,
      /*.queue_count=*/2,
      /*.compute_unit_count=*/40,
      /*.wavefront_size=*/wavefront_size,
      /*.maximum_waves_per_compute_unit=*/64,
      /*.maximum_workgroup_local_memory_size=*/64 * 1024,
      /*.vendor_packet_capabilities=*/0,
      /*.flags=*/IREE_HAL_AMDGPU_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_UUID |
          IREE_HAL_AMDGPU_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_PCI_ADDRESS,
  };
  iree_hal_amdgpu_device_spec_params_t params = {
      /*.logical_device_id=*/IREE_SV("amdgpu://0"),
      /*.display_name=*/IREE_SV("AMDGPU test device"),
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

TEST(DeviceSpecTest, CreatesSpecFromParams) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(IREE_SV("test"), iree_allocator_system(),
                                     iree_allocator_system(), &allocator));

  iree_hal_device_spec_t* device_spec = NULL;
  CreateDeviceSpecForProcessor(IREE_SV("gfx1100"), 32, allocator, &device_spec);

  const iree_hal_device_identity_spec_t* identity =
      iree_hal_device_spec_identity(device_spec);
  ASSERT_NE(identity, nullptr);
  EXPECT_TRUE(iree_string_view_equal(identity->driver_id, IREE_SV("amdgpu")));
  EXPECT_TRUE(iree_string_view_equal(identity->backend_id, IREE_SV("hsa")));
  ASSERT_EQ(identity->physical_device_count, 1);
  EXPECT_EQ(identity->physical_devices[0].physical_ordinal, 7);
  EXPECT_TRUE(
      iree_all_bits_set(identity->physical_devices[0].identity.flags,
                        IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_UUID |
                            IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_PCI_ADDRESS |
                            IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_NUMA_NODE));

  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(device_spec);
  ASSERT_NE(queues, nullptr);
  ASSERT_EQ(queues->family_count, 1);
  EXPECT_EQ(queues->families[0].provisioned_queue_count, 2);
  EXPECT_EQ(queues->families[0].timestamp_frequency_hz,
            kAgentTimestampFrequencyHz);

  const iree_hal_device_dispatch_spec_t* dispatch =
      iree_hal_device_spec_dispatch(device_spec);
  ASSERT_NE(dispatch, nullptr);
  EXPECT_EQ(dispatch->subgroup.default_size, 32);
  EXPECT_EQ(dispatch->subgroup.minimum_size, 32);
  EXPECT_EQ(dispatch->subgroup.maximum_size, 64);
  EXPECT_EQ(dispatch->subgroup.supported_size_mask, 1ull << 32);
  EXPECT_EQ(dispatch->execution.unit_count, 40);
  EXPECT_EQ(dispatch->execution.maximum_resident_invocation_count, 2048);
  EXPECT_EQ(dispatch->execution.maximum_resident_subgroup_count, 64);
  EXPECT_EQ(dispatch->execution.maximum_workgroup_local_memory_size, 64 * 1024);
  EXPECT_EQ(dispatch->execution.maximum_workgroup_local_memory_size_optin,
            64 * 1024);

  const iree_hal_device_memory_spec_t* memory =
      iree_hal_device_spec_memory(device_spec);
  ASSERT_NE(memory, nullptr);
  ASSERT_EQ(memory->heap_count, 1);
  EXPECT_EQ(memory->heaps[0].capacity_bytes,
            64ull * 1024ull * 1024ull * 1024ull);
  EXPECT_FALSE(iree_all_bits_set(
      memory->heaps[0].flags, IREE_HAL_MEMORY_HEAP_SPEC_FLAG_CAPACITY_UNKNOWN));

  const iree_hal_device_executable_spec_t* executables =
      iree_hal_device_spec_executables(device_spec);
  ASSERT_NE(executables, nullptr);
  ASSERT_GE(executables->target_count, 1);
  EXPECT_TRUE(iree_string_view_equal(executables->targets[0].family,
                                     IREE_SV("amdgpu")));
  EXPECT_TRUE(iree_string_view_equal(executables->targets[0].target_key,
                                     IREE_SV("gfx1100")));

  iree_hal_device_spec_release(device_spec);
  iree_hal_allocator_release(allocator);
}

TEST(DeviceSpecTest, ReportsProcessorWavefrontSupport) {
  struct Case {
    iree_string_view_t processor;
    uint32_t hsa_wavefront_size;
    uint32_t expected_default_size;
    uint32_t expected_minimum_size;
    uint32_t expected_maximum_size;
    uint64_t expected_supported_size_mask;
  };
  static const Case cases[] = {
      {IREE_SV("gfx1100"), 32, 32, 32, 64, 1ull << 32},
      {IREE_SV("gfx942"), 64, 64, 64, 64, 0},
      {IREE_SV("gfx1250"), 32, 32, 32, 32, 1ull << 32},
  };

  iree_hal_allocator_t* allocator = NULL;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(IREE_SV("test"), iree_allocator_system(),
                                     iree_allocator_system(), &allocator));

  for (const Case& test_case : cases) {
    iree_hal_device_spec_t* device_spec = NULL;
    CreateDeviceSpecForProcessor(test_case.processor,
                                 test_case.hsa_wavefront_size, allocator,
                                 &device_spec);

    const iree_hal_device_dispatch_spec_t* dispatch =
        iree_hal_device_spec_dispatch(device_spec);
    ASSERT_NE(dispatch, nullptr);
    EXPECT_EQ(dispatch->subgroup.default_size, test_case.expected_default_size);
    EXPECT_EQ(dispatch->subgroup.minimum_size, test_case.expected_minimum_size);
    EXPECT_EQ(dispatch->subgroup.maximum_size, test_case.expected_maximum_size);
    EXPECT_EQ(dispatch->subgroup.supported_size_mask,
              test_case.expected_supported_size_mask);

    iree_hal_device_spec_release(device_spec);
  }

  iree_hal_allocator_release(allocator);
}

TEST(DeviceSpecTest, AdvertisesTargetsPerPhysicalDevice) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(IREE_SV("test"), iree_allocator_system(),
                                     iree_allocator_system(), &allocator));

  iree_hal_amdgpu_target_identity_t identities[2];
  IREE_ASSERT_OK(iree_hal_amdgpu_target_identity_parse_artifact_key(
      IREE_SV("gfx1100"), &identities[0]));
  IREE_ASSERT_OK(iree_hal_amdgpu_target_identity_parse_artifact_key(
      IREE_SV("gfx1101"), &identities[1]));
  iree_hal_amdgpu_device_spec_physical_device_params_t physical_devices[2] = {
      {
          /*.identity=*/identities[0],
          /*.uuid=*/{},
          /*.pci=*/{},
          /*.timestamp_frequency_hz=*/kAgentTimestampFrequencyHz,
          /*.numa=*/{},
          /*.physical_ordinal=*/4,
          /*.queue_count=*/1,
          /*.compute_unit_count=*/40,
          /*.wavefront_size=*/32,
          /*.maximum_waves_per_compute_unit=*/64,
          /*.maximum_workgroup_local_memory_size=*/64 * 1024,
          /*.vendor_packet_capabilities=*/0,
          /*.flags=*/IREE_HAL_AMDGPU_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_NONE,
      },
      {
          /*.identity=*/identities[1],
          /*.uuid=*/{},
          /*.pci=*/{},
          /*.timestamp_frequency_hz=*/kAgentTimestampFrequencyHz,
          /*.numa=*/{},
          /*.physical_ordinal=*/9,
          /*.queue_count=*/1,
          /*.compute_unit_count=*/40,
          /*.wavefront_size=*/32,
          /*.maximum_waves_per_compute_unit=*/64,
          /*.maximum_workgroup_local_memory_size=*/64 * 1024,
          /*.vendor_packet_capabilities=*/0,
          /*.flags=*/IREE_HAL_AMDGPU_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_NONE,
      },
  };
  iree_hal_amdgpu_device_spec_params_t params = {
      /*.logical_device_id=*/IREE_SV("amdgpu://group"),
      /*.display_name=*/IREE_SV("AMDGPU test group"),
      /*.physical_device_count=*/IREE_ARRAYSIZE(physical_devices),
      /*.physical_devices=*/physical_devices,
      /*.device_memory_capacity_bytes=*/0,
      /*.device_allocator=*/allocator,
      /*.sanitizer=*/{},
      /*.flags=*/IREE_HAL_AMDGPU_DEVICE_SPEC_PARAM_FLAG_NONE,
  };
  iree_hal_device_spec_t* device_spec = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_device_spec_create(
      &params, iree_allocator_system(), &device_spec));

  const iree_hal_device_executable_spec_t* executables =
      iree_hal_device_spec_executables(device_spec);
  ASSERT_NE(executables, nullptr);
  ASSERT_EQ(executables->target_count, 3);

  iree_hal_executable_target_selection_t selection = {
      /*.family=*/IREE_SV("amdgpu"),
      /*.target_key=*/IREE_SV("gfx1100"),
      /*.kind_flags=*/IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
      /*.physical_device_affinity=*/0,
  };
  iree_hal_executable_target_selection_result_t result =
      iree_hal_device_spec_select_executable_target(device_spec, &selection);
  ASSERT_EQ(result.outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
  ASSERT_NE(result.target, nullptr);
  EXPECT_EQ(result.target->physical_device_affinity, 1ull);

  selection.target_key = IREE_SV("gfx11-generic");
  selection.kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_GENERIC;
  result =
      iree_hal_device_spec_select_executable_target(device_spec, &selection);
  ASSERT_EQ(result.outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
  ASSERT_NE(result.target, nullptr);
  EXPECT_EQ(result.target->physical_device_affinity, 3ull);

  selection.target_key = IREE_SV("gfx1101");
  selection.kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT;
  result =
      iree_hal_device_spec_select_executable_target(device_spec, &selection);
  ASSERT_EQ(result.outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
  ASSERT_NE(result.target, nullptr);
  EXPECT_EQ(result.target->physical_device_affinity, 2ull);

  iree_hal_device_spec_release(device_spec);
  iree_hal_allocator_release(allocator);
}

}  // namespace
}  // namespace iree::hal::amdgpu
