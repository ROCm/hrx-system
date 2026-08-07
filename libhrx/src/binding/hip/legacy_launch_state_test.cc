// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/legacy_launch_state.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

#include "iree/testing/gtest.h"

namespace {

class LegacyLaunchStateTest : public ::testing::Test {
 protected:
  void TearDown() override {
    iree_hip_legacy_launch_frame_t frame = {};
    while (iree_hip_legacy_launch_state_pop(&frame) == hipSuccess) {
      iree_hip_legacy_launch_frame_deinitialize(&frame);
    }
  }
};

TEST_F(LegacyLaunchStateTest, PreservesNestedFrames) {
  const dim3 outer_grid = {1, 2, 3};
  const dim3 outer_block = {4, 5, 6};
  const dim3 inner_grid = {7, 8, 9};
  const dim3 inner_block = {10, 11, 12};
  const uint32_t outer_argument = 0x11223344u;
  const uint32_t inner_argument = 0x55667788u;

  EXPECT_EQ(hipSuccess, iree_hip_legacy_launch_state_push(
                            outer_grid, outer_block, 13, nullptr));
  EXPECT_EQ(hipSuccess, iree_hip_legacy_launch_state_setup_argument(
                            &outer_argument, sizeof(outer_argument), 0));
  EXPECT_EQ(hipSuccess, iree_hip_legacy_launch_state_push(
                            inner_grid, inner_block, 14,
                            reinterpret_cast<hipStream_t>(uintptr_t{1})));
  EXPECT_EQ(hipSuccess, iree_hip_legacy_launch_state_setup_argument(
                            &inner_argument, sizeof(inner_argument), 0));

  iree_hip_legacy_launch_frame_t frame = {};
  ASSERT_EQ(hipSuccess, iree_hip_legacy_launch_state_pop(&frame));
  EXPECT_EQ(7u, frame.grid_dimension.x);
  EXPECT_EQ(11u, frame.block_dimension.y);
  EXPECT_EQ(14u, frame.shared_memory_bytes);
  EXPECT_EQ(reinterpret_cast<hipStream_t>(uintptr_t{1}), frame.stream);
  ASSERT_EQ(sizeof(inner_argument), frame.argument_length);
  EXPECT_EQ(0, std::memcmp(frame.argument_data, &inner_argument,
                           sizeof(inner_argument)));
  iree_hip_legacy_launch_frame_deinitialize(&frame);

  ASSERT_EQ(hipSuccess, iree_hip_legacy_launch_state_pop(&frame));
  EXPECT_EQ(1u, frame.grid_dimension.x);
  EXPECT_EQ(6u, frame.block_dimension.z);
  EXPECT_EQ(13u, frame.shared_memory_bytes);
  EXPECT_EQ(nullptr, frame.stream);
  ASSERT_EQ(sizeof(outer_argument), frame.argument_length);
  EXPECT_EQ(0, std::memcmp(frame.argument_data, &outer_argument,
                           sizeof(outer_argument)));
  iree_hip_legacy_launch_frame_deinitialize(&frame);
}

TEST_F(LegacyLaunchStateTest, InitializesOnlyNativeLayoutGaps) {
  EXPECT_EQ(hipSuccess, iree_hip_legacy_launch_state_push(
                            dim3{1, 1, 1}, dim3{1, 1, 1}, 0, nullptr));

  const std::array<uint8_t, 4> prefix = {1, 2, 3, 4};
  const std::array<uint8_t, 4> suffix = {5, 6, 7, 8};
  const std::array<uint8_t, 2> replacement = {9, 10};
  EXPECT_EQ(hipSuccess, iree_hip_legacy_launch_state_setup_argument(
                            prefix.data(), prefix.size(), 0));
  EXPECT_EQ(hipSuccess, iree_hip_legacy_launch_state_setup_argument(
                            suffix.data(), suffix.size(), 8));
  EXPECT_EQ(hipSuccess, iree_hip_legacy_launch_state_setup_argument(
                            replacement.data(), replacement.size(), 9));

  iree_hip_legacy_launch_frame_t frame = {};
  ASSERT_EQ(hipSuccess, iree_hip_legacy_launch_state_pop(&frame));
  const std::array<uint8_t, 12> expected = {1, 2, 3, 4, 0,  0,
                                            0, 0, 5, 9, 10, 8};
  ASSERT_EQ(expected.size(), frame.argument_length);
  EXPECT_EQ(0,
            std::memcmp(frame.argument_data, expected.data(), expected.size()));
  iree_hip_legacy_launch_frame_deinitialize(&frame);
}

TEST_F(LegacyLaunchStateTest, RejectsInvalidArgumentRanges) {
  EXPECT_EQ(hipErrorMissingConfiguration,
            iree_hip_legacy_launch_state_setup_argument(nullptr, 0, 0));
  EXPECT_EQ(hipSuccess, iree_hip_legacy_launch_state_push(
                            dim3{1, 1, 1}, dim3{1, 1, 1}, 0, nullptr));
  EXPECT_EQ(hipErrorInvalidValue,
            iree_hip_legacy_launch_state_setup_argument(nullptr, 1, 0));
  const uint8_t value = 1;
  EXPECT_EQ(hipErrorInvalidValue,
            iree_hip_legacy_launch_state_setup_argument(
                &value, 2, std::numeric_limits<size_t>::max()));
}

}  // namespace
