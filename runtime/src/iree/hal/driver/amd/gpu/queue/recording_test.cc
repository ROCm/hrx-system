// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/recording.h"

#include <vector>

#include "iree/hal/driver/amd/gpu/queue/fixup.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class RecordingTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    iree_hal_amd_gpu_recording_initialize(iree_allocator_system(), &recording_);
  }
  void TearDown() override {
    iree_hal_amd_gpu_recording_deinitialize(&recording_);
  }
  // Production recording under test, including its temporary chunk ownership.
  iree_hal_amd_gpu_recording_t recording_ = {};
};

TEST_P(RecordingTest, NativeCommandsAndArgumentsHaveIndependentBases) {
  uint32_t offset = 0;
  void* data = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_recording_commands(&recording_, 16, 4, &offset, &data));
  memset(data, 0, 16);
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_recording_data(&recording_, 17, 64, &offset, &data));
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_recording_command_data(&recording_, 20, &offset, &data));
  const auto wide = IREE_HAL_AMD_GPU_RECORDING_FIXUP_WIDE;
  IREE_ASSERT_OK(iree_hal_amd_gpu_recording_fixup(
      &recording_, 0, 0, 3, 0,
      wide | IREE_HAL_AMD_GPU_RECORDING_FIXUP_ADDEND_DATA));
  IREE_ASSERT_OK(iree_hal_amd_gpu_recording_fixup(
      &recording_, 8, 0, 4, 0,
      wide | IREE_HAL_AMD_GPU_RECORDING_FIXUP_ADDEND_COMMAND_DATA));
  IREE_ASSERT_OK(iree_hal_amd_gpu_recording_fixup(
      &recording_, 1, 1, 8, 0,
      wide | IREE_HAL_AMD_GPU_RECORDING_FIXUP_TARGET_DATA));
  IREE_ASSERT_OK(iree_hal_amd_gpu_recording_fixup(
      &recording_, 0, 0, 16, 0,
      wide | IREE_HAL_AMD_GPU_RECORDING_FIXUP_TARGET_COMMAND_DATA |
          IREE_HAL_AMD_GPU_RECORDING_FIXUP_ADDEND_COMMAND_DATA));
  IREE_ASSERT_OK(iree_hal_amd_gpu_recording_fixup(
      &recording_, 8, 1, 16, 0,
      wide | IREE_HAL_AMD_GPU_RECORDING_FIXUP_TARGET_COMMAND_DATA));

  ASSERT_EQ(iree_hal_amd_gpu_recording_data_offset(&recording_), 64u);
  ASSERT_EQ(iree_hal_amd_gpu_recording_command_data_offset(&recording_), 84u);
  std::vector<uint8_t> contents(105, 0xCC);
  iree_hal_amd_gpu_fixup_t fixups[10];
  const bool separate = GetParam();
  iree_hal_amd_gpu_recording_write(&recording_, contents.data(),
                                   separate ? 2 : 0, fixups);
  for (size_t i = 16; i < 104; ++i) {
    EXPECT_EQ(contents[i], 0u) << i;
  }
  EXPECT_EQ(contents.back(), 0xCC);

  const uint64_t arguments = UINT64_C(0x123400000000);
  const uint64_t commands =
      separate ? UINT64_C(0x567800000000) : arguments + 20;
  const uint64_t external = UINT64_C(0xABCD00000000);
  const uint64_t values[] = {arguments - 64, external, commands - 84};
  iree_hal_amd_gpu_fixup_apply(fixups, 10, values, contents.data());
  const struct {
    // Logical byte coordinate in the combined CPU template.
    size_t offset;
    // Fully resolved native pointer expected at that coordinate.
    uint64_t value;
  } expected[] = {
      {0, arguments + 3},  {8, commands + 4},   {65, external + 8},
      {84, commands + 16}, {92, external + 16},
  };
  for (const auto& entry : expected) {
    uint64_t value = 0;
    memcpy(&value, contents.data() + entry.offset, sizeof(value));
    EXPECT_EQ(value, entry.value) << entry.offset;
  }
  EXPECT_EQ(contents.back(), 0xCC);
}

INSTANTIATE_TEST_SUITE_P(Placement, RecordingTest, ::testing::Bool());

}  // namespace
