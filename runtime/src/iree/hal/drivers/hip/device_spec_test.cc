// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "iree/hal/drivers/hip/device_spec_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::hip {
namespace {

static iree_hal_hip_device_facts_t MakeTestSpec() {
  iree_hal_hip_device_facts_t spec = {
      /*.architecture=*/
      {/*.gcn_arch_name=*/""},
      /*.launch=*/
      {
          /*.maximum_workgroup_invocations=*/1024,
          /*.maximum_workgroup_size=*/{1024, 1024, 64},
          /*.maximum_workgroup_count=*/{2147483647u, 65535u, 65535u},
          /*.maximum_workgroups_per_execution_unit=*/40,
          /*.maximum_invocations_per_execution_unit=*/2048,
          /*.maximum_workgroup_register_count=*/65536,
          /*.maximum_local_memory_size=*/64 * 1024,
          /*.maximum_workgroup_local_memory_size=*/64 * 1024,
      },
      /*.clocks=*/
      {/*.clock_instruction_frequency_hz=*/1000000000ull},
      /*.execution_unit_count=*/60,
      /*.subgroup_size=*/32,
  };
  std::strncpy(spec.architecture.gcn_arch_name, "gfx1100",
               sizeof(spec.architecture.gcn_arch_name) - 1);
  return spec;
}

TEST(DeviceSpecTest, CreatesSpecFromParams) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(IREE_SV("test"), iree_allocator_system(),
                                     iree_allocator_system(), &allocator));

  iree_hal_hip_device_spec_physical_device_params_t physical_device = {
      /*.display_name=*/IREE_SV("HIP test device"),
      /*.backend_path=*/IREE_SV("GPU-00000000-0000-0000-0000-000000000000"),
      /*.uuid=*/{{0x11}},
      /*.pci=*/{/*.domain=*/0, /*.bus=*/3, /*.device=*/0, /*.function=*/0},
      /*.physical_ordinal=*/7,
      /*.facts=*/MakeTestSpec(),
      /*.flags=*/IREE_HAL_HIP_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_UUID |
          IREE_HAL_HIP_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_PCI_ADDRESS,
  };
  iree_hal_hip_device_spec_params_t params = {
      /*.logical_device_id=*/IREE_SV("hip://0"),
      /*.display_name=*/IREE_SV("HIP test logical device"),
      /*.queue_count=*/1,
      /*.physical_device_count=*/1,
      /*.physical_devices=*/&physical_device,
      /*.device_allocator=*/allocator,
  };

  iree_hal_device_spec_t* device_spec = NULL;
  IREE_ASSERT_OK(iree_hal_hip_device_spec_create(
      &params, iree_allocator_system(), &device_spec));

  const iree_hal_device_identity_spec_t* identity =
      iree_hal_device_spec_identity(device_spec);
  ASSERT_NE(identity, nullptr);
  EXPECT_TRUE(iree_string_view_equal(identity->driver_id, IREE_SV("hip")));
  EXPECT_TRUE(iree_string_view_equal(identity->backend_id, IREE_SV("hip")));
  ASSERT_EQ(identity->physical_device_count, 1);
  EXPECT_EQ(identity->physical_devices[0].physical_ordinal, 7);
  EXPECT_TRUE(iree_all_bits_set(
      identity->physical_devices[0].identity.flags,
      IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_UUID |
          IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_PCI_ADDRESS));

  const iree_hal_device_memory_spec_t* memory =
      iree_hal_device_spec_memory(device_spec);
  ASSERT_NE(memory, nullptr);
  ASSERT_EQ(memory->memory_type_count, 1);
  EXPECT_TRUE(iree_all_bits_set(
      memory->memory_types[0].memory_type,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));

  iree_hal_external_buffer_handle_selection_t host_buffer_selection = {
      /*.handle_type_mask=*/IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE,
      /*.direction_flags=*/IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT,
      /*.buffer_usage=*/IREE_HAL_BUFFER_USAGE_MAPPING,
      /*.memory_access=*/IREE_HAL_MEMORY_ACCESS_READ,
      /*.compatible_memory_type_mask=*/1u << 0,
      /*.capability_flags=*/IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_BORROWED,
  };
  const iree_hal_external_buffer_handle_spec_t* host_buffer_handle =
      iree_hal_device_spec_find_external_buffer_handle(device_spec,
                                                       &host_buffer_selection);
  ASSERT_NE(host_buffer_handle, nullptr);
  EXPECT_TRUE(
      iree_all_bits_set(host_buffer_handle->direction_flags,
                        IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT));
  EXPECT_TRUE(iree_all_bits_set(host_buffer_handle->allowed_buffer_usage,
                                IREE_HAL_BUFFER_USAGE_MAPPING));
  EXPECT_TRUE(iree_all_bits_set(host_buffer_handle->allowed_memory_access,
                                IREE_HAL_MEMORY_ACCESS_ALL));

  iree_hal_external_buffer_handle_selection_t device_buffer_selection = {
      /*.handle_type_mask=*/IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE,
      /*.direction_flags=*/IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_EXPORT,
      /*.buffer_usage=*/IREE_HAL_BUFFER_USAGE_DISPATCH,
      /*.memory_access=*/IREE_HAL_MEMORY_ACCESS_NONE,
      /*.compatible_memory_type_mask=*/1u << 0,
      /*.capability_flags=*/IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_BORROWED,
  };
  const iree_hal_external_buffer_handle_spec_t* device_buffer_handle =
      iree_hal_device_spec_find_external_buffer_handle(
          device_spec, &device_buffer_selection);
  ASSERT_NE(device_buffer_handle, nullptr);
  EXPECT_NE(device_buffer_handle, host_buffer_handle);
  EXPECT_TRUE(
      iree_all_bits_set(device_buffer_handle->direction_flags,
                        IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT |
                            IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_EXPORT));
  EXPECT_TRUE(iree_all_bits_set(
      device_buffer_handle->allowed_buffer_usage,
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH));
  EXPECT_EQ(device_buffer_handle->allowed_memory_access,
            IREE_HAL_MEMORY_ACCESS_NONE);

  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(device_spec);
  ASSERT_NE(queues, nullptr);
  ASSERT_EQ(queues->family_count, 1);
  EXPECT_EQ(queues->families[0].queue_count, 1);
  EXPECT_EQ(queues->families[0].queue_affinity, 1u);
  EXPECT_EQ(queues->families[0].timestamp_frequency_hz, 1000000000ull);
  ASSERT_EQ(queues->external_timepoint_handle_count, 1);
  const iree_hal_external_timepoint_handle_spec_t* hip_event_timepoint =
      &queues->external_timepoint_handles[0];
  EXPECT_EQ(hip_event_timepoint->handle_type,
            IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_HIP_EVENT);
  EXPECT_TRUE(
      iree_all_bits_set(hip_event_timepoint->direction_flags,
                        IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT |
                            IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_EXPORT));
  EXPECT_TRUE(
      iree_all_bits_set(hip_event_timepoint->compatibility,
                        IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT |
                            IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_WAIT));
  EXPECT_EQ(hip_event_timepoint->flags,
            IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_NONE);
  iree_hal_external_timepoint_handle_selection_t timepoint_selection = {
      /*.handle_type=*/IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_HIP_EVENT,
      /*.direction_flags=*/IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT,
      /*.compatibility=*/IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_WAIT,
      /*.capability_flags=*/IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_NONE,
  };
  EXPECT_EQ(iree_hal_device_spec_find_external_timepoint_handle(
                device_spec, &timepoint_selection),
            hip_event_timepoint);

  const iree_hal_device_dispatch_spec_t* dispatch =
      iree_hal_device_spec_dispatch(device_spec);
  ASSERT_NE(dispatch, nullptr);
  EXPECT_EQ(dispatch->launch.maximum_workgroup_invocations, 1024);
  EXPECT_EQ(dispatch->launch.maximum_workgroup_size[2], 64);
  EXPECT_EQ(dispatch->subgroup.default_size, 32);
  EXPECT_EQ(dispatch->execution.unit_count, 60);

  const iree_hal_device_executable_spec_t* executables =
      iree_hal_device_spec_executables(device_spec);
  ASSERT_NE(executables, nullptr);
  ASSERT_EQ(executables->target_count, 2);

  iree_hal_executable_target_selection_t selection = {
      /*.family=*/IREE_SV("amdgpu"),
      /*.target_key=*/IREE_SV("gfx1100"),
      /*.kind_flags=*/IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
      /*.physical_device_affinity=*/0,
  };
  iree_hal_executable_target_selection_result_t exact_result =
      iree_hal_device_spec_select_executable_target(device_spec, &selection);
  ASSERT_EQ(exact_result.outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
  ASSERT_NE(exact_result.target, nullptr);
  EXPECT_EQ(exact_result.target->physical_device_affinity, 1ull);

  selection.target_key = IREE_SV("gfx11-generic");
  selection.kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_GENERIC;
  const iree_hal_executable_target_selection_result_t generic_result =
      iree_hal_device_spec_select_executable_target(device_spec, &selection);
  ASSERT_EQ(generic_result.outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
  ASSERT_NE(generic_result.target, nullptr);
  EXPECT_LT(generic_result.target->priority, exact_result.target->priority);
  EXPECT_EQ(generic_result.target->physical_device_affinity, 1ull);

  iree_hal_device_spec_release(device_spec);
  iree_hal_allocator_release(allocator);
}

}  // namespace
}  // namespace iree::hal::hip
