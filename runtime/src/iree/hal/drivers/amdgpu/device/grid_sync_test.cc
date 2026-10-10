// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/grid_sync.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

TEST(GridSyncTest, InitializesSingleGridInfo) {
  const uint32_t workgroup_count[3] = {3, 5, 7};
  const uint16_t workgroup_size[3] = {8, 4, 2};
  iree_amdgpu_grid_sync_info_t info;
  std::memset(&info, 0xFD, sizeof(info));

  IREE_ASSERT_OK(iree_hal_amdgpu_grid_sync_info_initialize(
      IREE_HAL_AMDGPU_GRID_SYNC_STRATEGY_MEMORY, workgroup_count,
      workgroup_size, &info));

  EXPECT_EQ(info.multi_grid_sync, 0u);
  EXPECT_EQ(info.grid_ordinal, 0u);
  EXPECT_EQ(info.grid_count, 1u);
  EXPECT_EQ(info.preceding_workitem_count, 0u);
  EXPECT_EQ(info.total_workitem_count, 6720u);
  EXPECT_EQ(info.single_grid_sync.word0, 0u);
  EXPECT_EQ(info.single_grid_sync.word1, 0u);
  EXPECT_EQ(info.workgroup_count, 105u);
  EXPECT_EQ(info.reserved, 0u);
}

TEST(GridSyncTest, EnforcesStrategyWorkgroupLimits) {
  const uint32_t maximum_memory_workgroup_count[3] = {UINT16_MAX, 1, 1};
  const uint32_t larger_workgroup_count[3] = {UINT16_MAX + 1u, 1, 1};
  const uint16_t workgroup_size[3] = {1, 1, 1};
  iree_amdgpu_grid_sync_info_t info = {};

  IREE_ASSERT_OK(iree_hal_amdgpu_grid_sync_info_initialize(
      IREE_HAL_AMDGPU_GRID_SYNC_STRATEGY_MEMORY, maximum_memory_workgroup_count,
      workgroup_size, &info));
  EXPECT_EQ(info.workgroup_count, UINT16_MAX);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_amdgpu_grid_sync_info_initialize(
                            IREE_HAL_AMDGPU_GRID_SYNC_STRATEGY_MEMORY,
                            larger_workgroup_count, workgroup_size, &info));
  IREE_ASSERT_OK(iree_hal_amdgpu_grid_sync_info_initialize(
      IREE_HAL_AMDGPU_GRID_SYNC_STRATEGY_GWS, larger_workgroup_count,
      workgroup_size, &info));
  EXPECT_EQ(info.workgroup_count, UINT16_MAX + 1u);
}

TEST(GridSyncTest, LeavesOutputUnchangedOnFailure) {
  const uint32_t workgroup_count[3] = {UINT32_MAX, 1, 1};
  const uint16_t workgroup_size[3] = {UINT16_MAX, UINT16_MAX, UINT16_MAX};
  iree_amdgpu_grid_sync_info_t info;
  std::memset(&info, 0xFD, sizeof(info));
  const iree_amdgpu_grid_sync_info_t sentinel = info;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_amdgpu_grid_sync_info_initialize(
                            IREE_HAL_AMDGPU_GRID_SYNC_STRATEGY_GWS,
                            workgroup_count, workgroup_size, &info));

  EXPECT_EQ(std::memcmp(&info, &sentinel, sizeof(info)), 0);
}

TEST(GridSyncTest, GwsInitializeEncodesParticipatingWorkgroups) {
  iree_hal_amdgpu_device_kernel_args_t kernel_args = {
      .kernel_object = 0x12345678u,
      .setup = 1,
  };
  kernel_args.workgroup_size[0] = 1;
  kernel_args.workgroup_size[1] = 1;
  kernel_args.workgroup_size[2] = 1;
  kernel_args.kernarg_size =
      IREE_HAL_AMDGPU_DEVICE_GRID_SYNC_GWS_INITIALIZE_KERNARG_SIZE;
  kernel_args.kernarg_alignment =
      IREE_HAL_AMDGPU_DEVICE_GRID_SYNC_GWS_INITIALIZE_KERNARG_ALIGNMENT;
  iree_hsa_kernel_dispatch_packet_t packet = {};
  uint16_t setup = 0xBEEFu;
  packet.header = 0xFFFFu;
  packet.setup = 0xA5A5u;
  iree_hal_amdgpu_device_grid_sync_gws_initialize_kernargs_t kernargs = {};

  iree_hal_amdgpu_device_grid_sync_gws_initialize_emplace(
      &kernel_args, /*workgroup_count=*/257, &packet, &kernargs, &setup);

  EXPECT_EQ(packet.header, 0xFFFFu);
  EXPECT_EQ(packet.setup, 0xA5A5u);
  EXPECT_EQ(setup, 1u);
  EXPECT_EQ(kernargs.workgroup_count_minus_one, 256u);
  EXPECT_EQ(packet.workgroup_size[0], 1u);
  EXPECT_EQ(packet.workgroup_size[1], 1u);
  EXPECT_EQ(packet.workgroup_size[2], 1u);
  EXPECT_EQ(packet.grid_size[0], 1u);
  EXPECT_EQ(packet.grid_size[1], 1u);
  EXPECT_EQ(packet.grid_size[2], 1u);
  EXPECT_EQ(packet.kernel_object, kernel_args.kernel_object);
  EXPECT_EQ(packet.kernarg_address, &kernargs);
}

}  // namespace
}  // namespace iree::hal::amdgpu
