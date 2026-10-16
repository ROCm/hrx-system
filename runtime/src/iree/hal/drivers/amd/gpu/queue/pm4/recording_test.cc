// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/recording.h"

#include <vector>

#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {
class RecordingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hal_amd_gpu_pm4_recording_initialize(iree_allocator_system(),
                                              &recording_);
  }
  void TearDown() override {
    iree_hal_amd_gpu_pm4_recording_deinitialize(&recording_);
  }
  // Actual temporary native recorder, freed after each test's sealed output.
  iree_hal_amd_gpu_pm4_recording_t recording_;
};

TEST_F(RecordingTest, NativeCommandsAndPackedArgumentsSpanIndependentChunks) {
  constexpr uint32_t kCount = 5000;
  for (uint32_t i = 0; i < kCount; ++i) {
    uint32_t data_offset = 0;
    void* data = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_recording_data(&recording_, 9, 64,
                                                       &data_offset, &data));
    EXPECT_EQ(data_offset, i * 64);
    static_cast<uint8_t*>(data)[0] = 0x7B;
    uint32_t command_offset = 0;
    uint32_t* words = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_recording_commands(
        &recording_, 5, &command_offset, &words));
    iree_hal_amd_gpu_pm4_emit_write(words, 0, 1, &i);
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_recording_fixup(
        &recording_, false, command_offset + 8, 0, data_offset + 1, true, 0,
        true));
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_recording_fixup(
        &recording_, true, data_offset + 1, 3, i, false, 0, true));
  }
  const uint32_t data_offset =
      iree_hal_amd_gpu_pm4_recording_data_offset(&recording_);
  EXPECT_EQ(data_offset % 64, 0u);
  EXPECT_EQ(iree_hal_amd_gpu_pm4_recording_fixup_count(&recording_),
            kCount * 4);
  std::vector<uint8_t> contents(data_offset + recording_.data.length, 0xAA);
  std::vector<iree_hal_amd_gpu_pm4_fixup_t> fixups(kCount * 4);
  iree_hal_amd_gpu_pm4_recording_write(&recording_, contents.data(),
                                       fixups.data());
  iree_hal_amd_gpu_pm4_recording_deinitialize(&recording_);
  for (uint32_t i = 0; i < kCount; ++i) {
    uint32_t expected[5];
    iree_hal_amd_gpu_pm4_emit_write(expected, 0, 1, &i);
    EXPECT_EQ(memcmp(contents.data() + i * 20, expected, sizeof(expected)), 0);
    EXPECT_EQ(contents[data_offset + i * 64], 0x7B);
    for (uint32_t j = 1; j < (i + 1 == kCount ? 9u : 64u); ++j) {
      EXPECT_EQ(contents[data_offset + i * 64 + j], 0u);
    }
    const auto* entries = fixups.data() + i * 4;
    EXPECT_EQ(entries[0].target_offset, i * 20 + 8);
    EXPECT_EQ(entries[0].addend, data_offset + i * 64 + 1);
    EXPECT_EQ(entries[1].target_offset, i * 20 + 12);
    EXPECT_EQ(entries[1].source, 32u << 24);
    EXPECT_EQ(entries[1].addend, entries[0].addend);
    EXPECT_EQ(entries[2].target_offset, data_offset + i * 64 + 1);
    EXPECT_EQ(entries[2].source, 3u);
    EXPECT_EQ(entries[2].addend, i);
    EXPECT_EQ(entries[3].target_offset, data_offset + i * 64 + 5);
    EXPECT_EQ(entries[3].source, 3u | (32u << 24));
    EXPECT_EQ(entries[3].addend, i);
  }
}

TEST_F(RecordingTest, NativeBlockFieldRejectsAnUnrepresentableExtent) {
  uint32_t offset = 0;
  uint32_t* words = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_amd_gpu_pm4_recording_commands(
                            &recording_, UINT32_MAX, &offset, &words));
  EXPECT_EQ(recording_.commands.length, 0u);
}
}  // namespace
