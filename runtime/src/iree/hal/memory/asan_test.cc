// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/asan.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_hal_asan_pool_options_t ShadowOptions() {
  iree_hal_asan_pool_options_t options = {};
  options.mode = IREE_HAL_ASAN_POOL_MODE_SHADOW;
  options.shadow_granule_size = 8;
  options.redzone_size = 16;
  return options;
}

TEST(ASANPoolOptionsTest, ZeroOptionsAreDisabled) {
  iree_hal_asan_pool_options_t options = {};
  EXPECT_FALSE(iree_hal_asan_pool_options_is_enabled(&options));
  IREE_EXPECT_OK(iree_hal_asan_pool_options_validate(&options));
}

TEST(ASANPoolOptionsTest, ValidShadowOptionsAreEnabled) {
  iree_hal_asan_pool_options_t options = ShadowOptions();
  EXPECT_TRUE(iree_hal_asan_pool_options_is_enabled(&options));
  IREE_EXPECT_OK(iree_hal_asan_pool_options_validate(&options));
}

TEST(ASANPoolOptionsTest, RejectsInvalidMode) {
  iree_hal_asan_pool_options_t options = ShadowOptions();
  options.mode = static_cast<iree_hal_asan_pool_mode_t>(99);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_asan_pool_options_validate(&options));
}

TEST(ASANPoolOptionsTest, RejectsInvalidShadowGranule) {
  iree_hal_asan_pool_options_t options = ShadowOptions();
  options.shadow_granule_size = 7;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_asan_pool_options_validate(&options));
}

TEST(ASANPoolOptionsTest, RejectsMissingRedzone) {
  iree_hal_asan_pool_options_t options = ShadowOptions();
  options.redzone_size = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_asan_pool_options_validate(&options));
}

TEST(ASANPoolOptionsTest, RejectsInvalidBackingAlignment) {
  iree_hal_asan_pool_options_t options = ShadowOptions();
  options.backing_alignment = 24;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_asan_pool_options_validate(&options));
}

TEST(ASANAllocationLayoutTest, CalculatesAlignedBackingRange) {
  iree_hal_asan_pool_options_t options = ShadowOptions();
  options.backing_alignment = 32;

  iree_hal_asan_allocation_layout_t layout;
  IREE_ASSERT_OK(
      iree_hal_asan_calculate_allocation_layout(&options, 13, 16, &layout));

  EXPECT_EQ(layout.backing_offset_alignment, 32u);
  EXPECT_EQ(layout.backing_length_alignment, 32u);
  EXPECT_EQ(layout.backing_length, 64u);
  EXPECT_EQ(layout.user_offset, 32u);
  EXPECT_EQ(layout.user_length, 13u);
  EXPECT_EQ(layout.left_redzone_length, 32u);
  EXPECT_EQ(layout.right_redzone_length, 19u);
}

TEST(ASANAllocationLayoutTest, UserAlignmentCanDriveBackingOffsetAlignment) {
  iree_hal_asan_pool_options_t options = ShadowOptions();

  iree_hal_asan_allocation_layout_t layout;
  IREE_ASSERT_OK(
      iree_hal_asan_calculate_allocation_layout(&options, 8, 64, &layout));

  EXPECT_EQ(layout.backing_offset_alignment, 64u);
  EXPECT_EQ(layout.backing_length_alignment, 8u);
  EXPECT_EQ(layout.backing_length, 88u);
  EXPECT_EQ(layout.user_offset, 64u);
  EXPECT_EQ(layout.user_length, 8u);
  EXPECT_EQ(layout.left_redzone_length, 64u);
  EXPECT_EQ(layout.right_redzone_length, 16u);
}

TEST(ASANAllocationLayoutTest, ExtendsBackingIntoRightRedzone) {
  iree_hal_asan_pool_options_t options = ShadowOptions();
  iree_hal_asan_allocation_layout_t layout;
  IREE_ASSERT_OK(
      iree_hal_asan_calculate_allocation_layout(&options, 13, 16, &layout));

  IREE_ASSERT_OK(iree_hal_asan_extend_allocation_layout(80, &layout));
  EXPECT_EQ(layout.backing_length, 80u);
  EXPECT_EQ(layout.user_offset, 16u);
  EXPECT_EQ(layout.user_length, 13u);
  EXPECT_EQ(layout.right_redzone_length, 51u);
}

TEST(ASANAllocationLayoutTest, ExtendRejectsShorterBackingLength) {
  iree_hal_asan_pool_options_t options = ShadowOptions();
  iree_hal_asan_allocation_layout_t layout;
  IREE_ASSERT_OK(
      iree_hal_asan_calculate_allocation_layout(&options, 13, 16, &layout));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_asan_extend_allocation_layout(16, &layout));
}

TEST(ASANAllocationLayoutTest, ExtendRejectsMisalignedBackingLength) {
  iree_hal_asan_pool_options_t options = ShadowOptions();
  iree_hal_asan_allocation_layout_t layout;
  IREE_ASSERT_OK(
      iree_hal_asan_calculate_allocation_layout(&options, 13, 16, &layout));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_asan_extend_allocation_layout(81, &layout));
}

TEST(ASANAllocationLayoutTest, RejectsDisabledOptions) {
  iree_hal_asan_pool_options_t options = {};
  iree_hal_asan_allocation_layout_t layout;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_asan_calculate_allocation_layout(&options, 13, 16, &layout));
}

TEST(ASANAllocationLayoutTest, RejectsEmptyUserRange) {
  iree_hal_asan_pool_options_t options = ShadowOptions();
  iree_hal_asan_allocation_layout_t layout;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_asan_calculate_allocation_layout(&options, 0, 16, &layout));
}

TEST(ASANAllocationLayoutTest, RejectsInvalidUserAlignment) {
  iree_hal_asan_pool_options_t options = ShadowOptions();
  iree_hal_asan_allocation_layout_t layout;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_asan_calculate_allocation_layout(&options, 13, 24, &layout));
}

TEST(ASANAllocationLayoutTest, RejectsOverflow) {
  iree_hal_asan_pool_options_t options = ShadowOptions();
  iree_hal_asan_allocation_layout_t layout;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_asan_calculate_allocation_layout(
                            &options, ~(iree_device_size_t)0, 8, &layout));
}

}  // namespace
