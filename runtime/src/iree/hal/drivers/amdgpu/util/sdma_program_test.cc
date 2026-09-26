// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/sdma_program.h"

#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

iree_hal_amdgpu_sdma_program_params_t Params() {
  iree_hal_amdgpu_sdma_program_params_t p{};
  IREE_CHECK_OK(
      iree_hal_amdgpu_sdma_query_capabilities({12, 0, 1}, &p.capabilities));
  p.completion_address = 0x10000;
  p.completion_value = 7;
  return p;
}

TEST(SdmaProgramTest, SplitCopyAndCompletionOrdering) {
  auto p = Params();
  iree_hal_amdgpu_sdma_copy_t copies[] = {
      {0x10000000, 0x20000000, (1u << 22) + 17}, {0, 0, 0}};
  p.copies = copies;
  p.copy_count = 2;
  p.wait_address = 0x20000;
  p.wait_value = 3;
  p.start_timestamp_address = 0x30000;
  p.end_timestamp_address = 0x30020;
  uint32_t count = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_sdma_program_measure(&p, &count));
  EXPECT_EQ(count, 40u);
  std::array<uint32_t, 40> words;
  words.fill(UINT32_MAX);
  IREE_ASSERT_OK(iree_hal_amdgpu_sdma_program_emit(&p, words.size(),
                                                   words.data(), &count));
  EXPECT_EQ(words[0], 0xb0000008u);  // Wait.
  EXPECT_EQ(words[6], 0x20du);       // Start timestamp.
  EXPECT_EQ(words[9], 0x111u);       // Acquire.
  EXPECT_EQ(words[14], 1u);
  EXPECT_EQ(words[15], 0x3fffffu);
  EXPECT_EQ(words[16], 0u);  // Reserved copy fields initialized.
  EXPECT_EQ(words[21], 1u);
  EXPECT_EQ(words[22], 16u);
  EXPECT_EQ(words[24], 0x10400000u);
  EXPECT_EQ(words[26], 0x20400000u);
  EXPECT_EQ(words[28], 0x111u);
  EXPECT_EQ(words[30], 0x80400000u);  // Release.
  EXPECT_EQ(words[33], 0x20du);
  EXPECT_EQ(words[36], 0x130005u);
  EXPECT_EQ(words[39], 7u);
}

TEST(SdmaProgramTest, FailureDoesNotWrite) {
  auto p = Params();
  std::array<uint32_t, 64> words;
  words.fill(0xdeadbeef);
  auto before = words;
  uint32_t count = 123;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amdgpu_sdma_program_emit(&p, 13, words.data(), &count));
  EXPECT_EQ(words, before);
  EXPECT_EQ(count, 123u);
  p.end_timestamp_address = 0x20000;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_sdma_program_emit(&p, 64, words.data(), &count));
  EXPECT_EQ(words, before);
  p.end_timestamp_address = 0;
  iree_hal_amdgpu_sdma_copy_t copy{UINT64_MAX, 4, 2};
  p.copies = &copy;
  p.copy_count = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amdgpu_sdma_program_emit(&p, 64, words.data(), &count));
  EXPECT_EQ(words, before);
}

TEST(SdmaProgramTest, UnknownTargetFailsWithoutPublishingCapabilities) {
  auto p = Params();
  auto before = p.capabilities;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_amdgpu_sdma_query_capabilities({12, 5, 0}, &p.capabilities));
  EXPECT_EQ(p.capabilities.profile, before.profile);
  IREE_ASSERT_OK(
      iree_hal_amdgpu_sdma_query_capabilities({12, 0, 0}, &p.capabilities));
  EXPECT_FALSE(p.capabilities.supports_indirect_buffers);
}

TEST(SdmaProgramTest, EmptyProgramStillReleasesAndCompletes) {
  auto p = Params();
  uint32_t count = 0;
  std::array<uint32_t, 14> words;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_sdma_program_emit(&p, 14, words.data(), &count));
  EXPECT_EQ(count, 14u);
  EXPECT_EQ(words[10], 0x130005u);
  EXPECT_EQ(words[13], 7u);
}
}  // namespace
