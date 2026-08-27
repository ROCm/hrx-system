// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/pm4_command_program_set.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

TEST(PM4CommandProgramSetTest, IsolatesReusableProfilePlansByQueue) {
  iree_hal_amdgpu_pm4_command_buffer_profile_plan_t profile_plans[2];
  iree_hal_amdgpu_pm4_command_program_set_t program_set;
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_command_program_set_initialize(
      /*queue_affinity=*/0xCull, /*physical_device_ordinal=*/1,
      /*physical_queue_count=*/2,
      IREE_HAL_AMDGPU_PM4_COMMAND_PROGRAM_SET_FLAG_PROFILE, profile_plans,
      &program_set));

  EXPECT_EQ(program_set.eligible_queue_mask, 0x3ull);
  EXPECT_EQ(program_set.profile_plan_count, 2u);
  EXPECT_EQ(
      iree_hal_amdgpu_pm4_command_program_set_select_profile(&program_set, 0),
      &profile_plans[0]);
  EXPECT_EQ(
      iree_hal_amdgpu_pm4_command_program_set_select_profile(&program_set, 1),
      &profile_plans[1]);
}

TEST(PM4CommandProgramSetTest, SharesSerialProfilePlan) {
  iree_hal_amdgpu_pm4_command_buffer_profile_plan_t profile_plans[2];
  iree_hal_amdgpu_pm4_command_program_set_t program_set;
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_command_program_set_initialize(
      /*queue_affinity=*/0x3ull, /*physical_device_ordinal=*/0,
      /*physical_queue_count=*/2,
      IREE_HAL_AMDGPU_PM4_COMMAND_PROGRAM_SET_FLAG_SERIAL_PROFILE |
          IREE_HAL_AMDGPU_PM4_COMMAND_PROGRAM_SET_FLAG_PROFILE,
      profile_plans, &program_set));

  EXPECT_EQ(program_set.profile_plan_count, 1u);
  EXPECT_EQ(
      iree_hal_amdgpu_pm4_command_program_set_select_profile(&program_set, 0),
      &profile_plans[0]);
  EXPECT_EQ(
      iree_hal_amdgpu_pm4_command_program_set_select_profile(&program_set, 1),
      &profile_plans[0]);
}

TEST(PM4CommandProgramSetTest, SelectsSparseEligibleQueues) {
  iree_hal_amdgpu_pm4_command_buffer_profile_plan_t profile_plans[4];
  iree_hal_amdgpu_pm4_command_program_set_t program_set;
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_command_program_set_initialize(
      /*queue_affinity=*/0xAull, /*physical_device_ordinal=*/0,
      /*physical_queue_count=*/4,
      IREE_HAL_AMDGPU_PM4_COMMAND_PROGRAM_SET_FLAG_PROFILE, profile_plans,
      &program_set));

  EXPECT_EQ(program_set.eligible_queue_mask, 0xAull);
  EXPECT_EQ(program_set.profile_plan_count, 2u);
  EXPECT_EQ(
      iree_hal_amdgpu_pm4_command_program_set_select_profile(&program_set, 0),
      nullptr);
  EXPECT_EQ(
      iree_hal_amdgpu_pm4_command_program_set_select_profile(&program_set, 1),
      &profile_plans[0]);
  EXPECT_EQ(
      iree_hal_amdgpu_pm4_command_program_set_select_profile(&program_set, 3),
      &profile_plans[1]);
}

TEST(PM4CommandProgramSetTest, RejectsCrossDeviceAffinity) {
  iree_hal_amdgpu_pm4_command_buffer_profile_plan_t profile_plans[2];
  iree_hal_amdgpu_pm4_command_program_set_t program_set;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_pm4_command_program_set_initialize(
          /*queue_affinity=*/0x5ull, /*physical_device_ordinal=*/1,
          /*physical_queue_count=*/2,
          IREE_HAL_AMDGPU_PM4_COMMAND_PROGRAM_SET_FLAG_NONE, profile_plans,
          &program_set));
}

TEST(PM4CommandProgramSetTest, LaysOutOneNormalAndQueuePrivateProfiles) {
  iree_hal_amdgpu_pm4_command_buffer_profile_plan_t profile_plans[2];
  iree_hal_amdgpu_pm4_command_program_set_t program_set;
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_command_program_set_initialize(
      /*queue_affinity=*/0x3ull, /*physical_device_ordinal=*/0,
      /*physical_queue_count=*/2,
      IREE_HAL_AMDGPU_PM4_COMMAND_PROGRAM_SET_FLAG_PROFILE, profile_plans,
      &program_set));

  iree_hal_amdgpu_pm4_command_program_layout_t layout;
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_command_program_layout_calculate(
      &program_set, /*program_dword_count=*/11,
      /*profile_program_dword_count=*/17, /*template_byte_length=*/13,
      /*fixup_entry_count=*/2, /*profile_fixup_entry_count=*/3, &layout));

  EXPECT_EQ(layout.program_byte_length, 11u * sizeof(uint32_t));
  EXPECT_GE(layout.profile_program_offset,
            layout.program_offset + layout.program_byte_length);
  EXPECT_EQ(iree_hal_amdgpu_pm4_command_program_layout_profile_program_offset(
                &layout, 1),
            layout.profile_program_offset + layout.profile_program_stride);
  EXPECT_GE(layout.template_offset,
            layout.profile_program_offset +
                layout.profile_program_stride * program_set.profile_plan_count);
  EXPECT_GE(layout.fixup_offset,
            layout.template_offset + layout.template_byte_length);
  EXPECT_GE(layout.profile_fixup_offset,
            layout.fixup_offset + layout.fixup_byte_length);
  EXPECT_GE(layout.dummy_ticks_offset,
            layout.profile_fixup_offset +
                layout.profile_fixup_stride * program_set.profile_plan_count);
  EXPECT_GE(
      layout.total_byte_length,
      layout.dummy_ticks_offset + sizeof(iree_hal_amdgpu_timestamp_range_t) *
                                      program_set.profile_plan_count);
}

}  // namespace
}  // namespace iree::hal::amdgpu
