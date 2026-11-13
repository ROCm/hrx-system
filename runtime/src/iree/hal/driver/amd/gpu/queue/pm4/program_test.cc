// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/pm4/program.h"

#include <algorithm>
#include <array>
#include <tuple>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ProgramTest
    : public ::testing::TestWithParam<
          std::tuple<uint32_t, bool, uint32_t, bool, bool, bool>> {};

TEST_P(ProgramTest, NativeEntryPublishesItsActualProducers) {
  const auto [profile_index, initialized, fixup_mode, large, join, chain] =
      GetParam();
  const bool fixups = fixup_mode != 0;
  const bool packets = fixup_mode == 2;
  const iree_hal_amd_gpu_gfxip_version_t versions[] = {
      {11, 0, 0}, {12, 0, 0}, {12, 5, 0}};
  const auto version = versions[profile_index];
  iree_hal_amd_gpu_pm4_barrier_profile_t profile;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_pm4_barrier_profile_initialize(version, &profile));
  iree_hal_amd_gpu_pm4_program_pool_t pool = {
      .chain = chain,
  };
  iree_const_byte_span_t code;
  iree_hal_amd_gpu_image_kernel_t kernel;
  IREE_ASSERT_OK(iree_hal_amd_gpu_fixup_program_query(version, &code, &kernel));
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_launch_initialize(version, 0, &kernel,
                                                        0x100000, &pool.fixup));
  IREE_ASSERT_OK(iree_hal_amd_gpu_blit_program_query(version, &code, &kernel));
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_launch_initialize(version, 0, &kernel,
                                                        0x200000, &pool.blit));

  // Metadata admitted by the recorder and held under an exclusive invocation
  // reservation. Native allocation is irrelevant to this pure packet encoder.
  const uint32_t byte_length = large ? 0x04000000 : 4096;
  iree_hal_amd_gpu_program_t program = {
      .pool = &pool.storage,
      .upload_address = UINT64_C(0x1234560000),
      .byte_length = byte_length,
      .fixup_count = packets  ? 4u
                     : fixups ? 513u
                              : 0u,
      .fixup_offset = byte_length,
  };
  iree_hal_amd_gpu_program_instance_t instance = {
      .address = UINT64_C(0x2345670000),
      .initialized = initialized,
      .join_previous = join,
  };
  const iree_hal_amd_gpu_pm4_program_block_t entry = {
      .offset = 256,
      .word_count = 1024,
  };
  constexpr uint64_t kArgumentAddress = UINT64_C(0x3456780000);
  const uint32_t fixup_words[] = {
      0xc0043700, 0x00100200, 0x45670100, 0x23, 0x55667788, 0x11223344,
      0xc0043700, 0x00100200, 0x45670110, 0x23, 0x45670100, 0x23,
  };
  const uint32_t fixup_word_count = packets ? IREE_ARRAYSIZE(fixup_words) : 0;
  const uint32_t count = iree_hal_amd_gpu_pm4_program_word_count(
      &program, &instance, &profile, program.fixup_count, fixup_word_count);
  const uint32_t padding = chain ? 6 : 0;
  iree_hal_amd_gpu_pm4_program_arguments_t arguments = {};
  std::array<uint32_t, 128> words;
  words.fill(0xDEADBEEF);
  auto* end = iree_hal_amd_gpu_pm4_program_emit(
      &program, &instance, entry, &profile, kArgumentAddress, 0x900008,
      &arguments, program.fixup_count, fixup_word_count, fixup_words, padding,
      words.data());
  ASSERT_EQ(end - words.data(), count + padding);
  ASSERT_LT(count + padding, words.size());
  EXPECT_EQ(*end, 0xDEADBEEFu);
  if (fixups && !packets) {
    EXPECT_EQ(arguments.fixup.entries,
              program.upload_address + program.fixup_offset);
    EXPECT_EQ(arguments.fixup.values, 0x900008u);
    EXPECT_EQ(arguments.fixup.target, instance.address);
    EXPECT_EQ(arguments.fixup.entry_count, program.fixup_count);
  } else {
    EXPECT_EQ(arguments.fixup.entry_count, 0u);
  }
  if (!initialized && large) {
    EXPECT_EQ(arguments.copy.source, program.upload_address);
    EXPECT_EQ(arguments.copy.target, instance.address);
    EXPECT_EQ(arguments.copy.length,
              program.fixup_offset +
                  program.fixup_count * sizeof(iree_hal_amd_gpu_fixup_t));
    EXPECT_EQ(arguments.copy.fill, 0u);
  }

  std::vector<uint32_t> cache_controls;
  std::vector<uint32_t> opcodes;
  for (uint32_t offset = 0; offset < count + padding;) {
    const uint32_t header = words[offset];
    ASSERT_EQ(header >> 30, 3u);
    const uint32_t opcode = (header >> 8) & 0xff;
    const uint32_t length = ((header >> 16) & 0x3fff) + 2;
    ASSERT_LE(offset + length, count + padding);
    opcodes.push_back(opcode);
    if (opcode == 0x58) {
      ASSERT_EQ(length, 8u);
      cache_controls.push_back(words[offset + 7]);
    }
    offset += length;
  }
  std::vector<uint32_t> expected_cache = {profile.acquire};
  const uint32_t publication = profile_index ? profile.release : 0;
  if (!initialized && large) {
    // A shader clone must publish before a bypassing CP overwrites its bytes.
    expected_cache.push_back(profile.shader |
                             (!fixups || packets ? publication : 0));
  } else if (!initialized && !fixups && publication) {
    expected_cache.push_back(publication);
  }
  if (fixups) {
    expected_cache.push_back(packets && profile.command_processor_bypasses_l2
                                 ? profile.acquire
                                 : profile.shader |
                                       (packets ? 0 : publication));
  }
  EXPECT_EQ(cache_controls, expected_cache);
  EXPECT_EQ(std::count(opcodes.begin(), opcodes.end(), 0x15u),
            (!initialized && large ? 1 : 0) + (fixups && !packets ? 1 : 0));
  EXPECT_EQ(std::count(opcodes.begin(), opcodes.end(), 0x37u), packets ? 2 : 0);
  EXPECT_EQ(opcodes.front(), join && fixups ? 0x46u : 0x58u);
  ASSERT_EQ(opcodes.back(), 0x3fu);
  EXPECT_EQ(end[-3], static_cast<uint32_t>(instance.address + entry.offset));
  EXPECT_EQ(end[-2], static_cast<uint32_t>(instance.address >> 32));
  EXPECT_EQ(end[-1], entry.word_count | (1u << 23) | (chain ? 1u << 20 : 0));
  if (chain) {
    EXPECT_EQ(opcodes[opcodes.size() - 2], 0x10u);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ProducerAndFetchPaths, ProgramTest,
    ::testing::Combine(::testing::Values(0u, 1u, 2u), ::testing::Bool(),
                       ::testing::Values(0u, 1u, 2u), ::testing::Bool(),
                       ::testing::Bool(), ::testing::Bool()));

TEST(ProgramReplayTest, PreservedAddressesNeedNoRepairOrReaderJoin) {
  const iree_hal_amd_gpu_gfxip_version_t versions[] = {
      {11, 0, 0}, {12, 0, 0}, {12, 5, 0}};
  for (const auto version : versions) {
    iree_hal_amd_gpu_pm4_barrier_profile_t profile;
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_pm4_barrier_profile_initialize(version, &profile));
    iree_hal_amd_gpu_pm4_program_pool_t pool = {};
    const iree_hal_amd_gpu_program_t program = {
        .pool = &pool.storage,
        .fixup_count = 4,
    };
    const iree_hal_amd_gpu_program_instance_t instance = {
        .address = UINT64_C(0x2345670000),
        .last_epoch = 1,
        .initialized = true,
        .join_previous = true,
    };
    const iree_hal_amd_gpu_pm4_program_block_t entry = {256, 1024};
    std::array<uint32_t, 13> words;
    words.fill(0xDEADBEEF);
    auto* end = iree_hal_amd_gpu_pm4_program_emit(&program, &instance, entry,
                                                  &profile, 0, 0, nullptr, 0, 0,
                                                  nullptr, 0, words.data());
    EXPECT_EQ(end - words.data(), iree_hal_amd_gpu_pm4_program_word_count(
                                      &program, &instance, &profile, 0, 0));
    const std::vector<uint32_t> expected = {
        0xc0065800, 0,          0xffffffff, 0xff,
        0,          0,          0x0a,       profile.acquire,
        0xc0023f00, 0x45670100, 0x23,       1024u | (1u << 23),
    };
    EXPECT_EQ(std::vector<uint32_t>(words.data(), end), expected);
    EXPECT_EQ(*end, 0xDEADBEEFu);
  }
}

TEST(FixupPacketTest, RelocatesPointersAndShiftedCodeAddressesWithoutReads) {
  const uint64_t values[] = {UINT64_C(0x1234fffffff0),
                             UINT64_C(0x56789abcde00)};
  const iree_hal_amd_gpu_fixup_t entries[] = {
      {.target_offset = 0x100, .source = 0, .addend = 0x30},
      {.target_offset = 0x104, .source = 32u << 24, .addend = 0x30},
      {.target_offset = 0x200, .source = 1 | (8u << 24), .addend = 0x200},
      {.target_offset = 0x204, .source = 1 | (40u << 24), .addend = 0x200},
  };
  std::array<uint32_t, 16> storage;
  storage.fill(0xDEADBEEF);
  const uint32_t word_count = iree_hal_amd_gpu_pm4_fixup_packet_word_count(
      entries, IREE_ARRAYSIZE(entries));
  auto* end = iree_hal_amd_gpu_pm4_fixup_packets_emit(
      entries, IREE_ARRAYSIZE(entries), values, UINT64_C(0x23ffffff00),
      storage.data() + 1);
  const std::vector<uint32_t> expected = {
      0xc0043700, 0x00100200, 0,     0x24, 0x20,       0x1235,
      0xc0043700, 0x00100200, 0x100, 0x24, 0x789abce0, 0x56,
  };
  EXPECT_EQ(word_count, expected.size());
  EXPECT_EQ(std::vector<uint32_t>(storage.data() + 1, end), expected);
  EXPECT_EQ(storage.front(), 0xDEADBEEFu);
  EXPECT_EQ(*end, 0xDEADBEEFu);
}

TEST(FixupPacketTest, UnalignedWordsKeepTheByteStoreShader) {
  const iree_hal_amd_gpu_fixup_t entries[] = {
      {.target_offset = 0x102, .source = 0, .addend = 0},
      {.target_offset = 0x106, .source = 32u << 24, .addend = 0},
  };
  EXPECT_EQ(iree_hal_amd_gpu_pm4_fixup_packet_word_count(
                entries, IREE_ARRAYSIZE(entries)),
            0u);
}

}  // namespace
