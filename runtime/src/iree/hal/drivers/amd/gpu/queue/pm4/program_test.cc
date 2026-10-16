// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/program.h"

#include <array>
#include <tuple>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ProgramTest : public ::testing::TestWithParam<
                        std::tuple<uint32_t, bool, bool, bool, bool, bool>> {};

TEST_P(ProgramTest, NativeEntryPublishesItsActualProducers) {
  const auto [profile_index, initialized, fixups, large, join, chain] =
      GetParam();
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
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_pm4_fixup_program_query(version, &code, &kernel));
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_launch_initialize(version, 0, &kernel,
                                                        0x100000, &pool.fixup));
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_pm4_blit_program_query(version, &code, &kernel));
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_launch_initialize(version, 0, &kernel,
                                                        0x200000, &pool.blit));

  // Metadata admitted by the recorder and held under an exclusive invocation
  // reservation. Native allocation is irrelevant to this pure packet encoder.
  const uint32_t byte_length = large ? 0x04000000 : 4096;
  iree_hal_amd_gpu_pm4_program_t program = {
      .pool = &pool,
      .upload_address = UINT64_C(0x1234560000),
      .byte_length = byte_length,
      .fixup_count = fixups ? 513u : 0u,
      .fixup_offset = byte_length,
  };
  iree_hal_amd_gpu_pm4_program_instance_t instance = {
      .address = UINT64_C(0x2345670000),
      .initialized = initialized,
      .join_previous = join,
  };
  const iree_hal_amd_gpu_pm4_program_block_t entry = {
      .offset = 256,
      .word_count = 1024,
  };
  constexpr uint64_t kArgumentAddress = UINT64_C(0x3456780000);
  const uint32_t count =
      iree_hal_amd_gpu_pm4_program_word_count(&program, &instance, &profile);
  const uint32_t padding = chain ? 6 : 0;
  iree_hal_amd_gpu_pm4_program_arguments_t arguments = {};
  std::array<uint32_t, 128> words;
  words.fill(0xDEADBEEF);
  auto* end = iree_hal_amd_gpu_pm4_program_emit(
      &program, &instance, entry, &profile, kArgumentAddress, &arguments,
      padding, words.data());
  ASSERT_EQ(end - words.data(), count + padding);
  ASSERT_LT(count + padding, words.size());
  EXPECT_EQ(*end, 0xDEADBEEFu);
  EXPECT_EQ(arguments.fixup.entries,
            program.upload_address + program.fixup_offset);
  EXPECT_EQ(arguments.fixup.values, kArgumentAddress + sizeof(arguments));
  EXPECT_EQ(arguments.fixup.target, instance.address);
  EXPECT_EQ(arguments.fixup.entry_count, program.fixup_count);
  if (!initialized && large) {
    EXPECT_EQ(arguments.copy.source, program.upload_address);
    EXPECT_EQ(arguments.copy.target, instance.address);
    EXPECT_EQ(arguments.copy.length,
              program.fixup_offset +
                  program.fixup_count * sizeof(iree_hal_amd_gpu_pm4_fixup_t));
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
  std::vector<uint32_t> expected_cache = {profile.acquire | 1u};
  const uint32_t publication = profile_index ? profile.release : 0;
  if (!initialized && large) {
    expected_cache.push_back(profile.shader | (fixups ? 0 : publication));
  } else if (!initialized && !fixups && publication) {
    expected_cache.push_back(publication);
  }
  if (fixups) {
    expected_cache.push_back(profile.shader | publication);
  }
  EXPECT_EQ(cache_controls, expected_cache);
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
                       ::testing::Bool(), ::testing::Bool(), ::testing::Bool(),
                       ::testing::Bool()));

}  // namespace
