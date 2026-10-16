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
    iree_hal_amd_gpu_pm4_recording_initialize(iree_allocator_system(), 7,
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
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_recording_end(&recording_, 9, false));
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

class RecordingBlockTest : public RecordingTest,
                           public ::testing::WithParamInterface<uint32_t> {};

TEST_P(RecordingBlockTest, SuccessorEdgesPreservePacketsAndInstanceFixups) {
  const bool chain = GetParam() & 8;
  const uint32_t residue = GetParam() & 7;
  // Three indivisible packet sequences force three native blocks. The first
  // covers every padding residue, including the one-DWORD hole requiring an
  // extra alignment unit. A small middle block verifies its exact edge count.
  const uint32_t lengths[] = {0xffff0u - residue, 32, 0xffff0u};
  uint32_t offsets[3];
  for (uint32_t i = 0; i < 3; ++i) {
    uint32_t* words = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_recording_commands(
        &recording_, lengths[i], &offsets[i], &words));
    for (uint32_t remaining = lengths[i]; remaining;) {
      const uint32_t count = iree_min(remaining, 16384u);
      words = iree_hal_amd_gpu_pm4_emit_nop(words, count);
      remaining -= count;
    }
  }
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_recording_end(&recording_, 9, chain));
  const uint32_t fixup_count =
      iree_hal_amd_gpu_pm4_recording_fixup_count(&recording_);
  ASSERT_EQ(fixup_count, chain ? 6u : 4u);
  std::vector<uint32_t> words(
      iree_hal_amd_gpu_pm4_recording_data_offset(&recording_) / 4);
  std::vector<iree_hal_amd_gpu_pm4_fixup_t> fixups(fixup_count);
  iree_hal_amd_gpu_pm4_recording_write(
      &recording_, reinterpret_cast<uint8_t*>(words.data()), fixups.data());
  uint32_t word_count = recording_.entry_word_count;
  for (uint32_t block = 0; block < 3; ++block) {
    EXPECT_EQ(offsets[block] % 32, 0u);
    EXPECT_EQ(word_count % 8, 0u);
    ASSERT_LE(word_count, 0xfffffu);
    ASSERT_GE(word_count, lengths[block] + 4);
    const uint32_t end = offsets[block] / 4 + word_count;
    const uint32_t edge = end - 4;
    uint32_t position = offsets[block] / 4;
    while (position < edge) {
      ASSERT_EQ(words[position] & 0xc000ff00u, 0xc0001000u);
      position += ((words[position] >> 16) & 0x3fff) + 2;
    }
    ASSERT_EQ(position, edge);
    if (block == 2 && !chain) {
      EXPECT_EQ(words[edge], 0xc0021000u);
      EXPECT_EQ(words[edge + 1], 0u);
      EXPECT_EQ(words[edge + 2], 0u);
      EXPECT_EQ(words[edge + 3], 0u);
    } else {
      EXPECT_EQ(words[edge], 0xc0023f00u);
      EXPECT_EQ(words[edge + 1], 0u);
      EXPECT_EQ(words[edge + 2], 0u);
      EXPECT_EQ(words[edge + 3] & 0xfff00000u, 0x00900000u);
      const auto* entries = fixups.data() + block * 2;
      const uint32_t source = block == 2 ? 9u : 7u;
      const uint64_t addend = block == 2 ? 0u : offsets[block + 1];
      EXPECT_EQ(entries[0].target_offset, (edge + 1) * 4);
      EXPECT_EQ(entries[1].target_offset, (edge + 2) * 4);
      EXPECT_EQ(entries[0].source, source);
      EXPECT_EQ(entries[1].source, source | (32u << 24));
      EXPECT_EQ(entries[0].addend, addend);
      EXPECT_EQ(entries[1].addend, addend);
      word_count = words[edge + 3] & 0xfffff;
      if (block == 2) {
        EXPECT_EQ(word_count, 32u);
      }
    }
    if (block != 2) {
      EXPECT_EQ(offsets[block + 1], end * 4);
    } else {
      EXPECT_EQ(recording_.commands.length, end * 4);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(PaddingAndExit, RecordingBlockTest,
                         ::testing::Range(0u, 16u));
}  // namespace
