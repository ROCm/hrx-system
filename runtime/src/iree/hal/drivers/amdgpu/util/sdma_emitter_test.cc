// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/sdma_emitter.h"

#include <array>

#include "iree/testing/gtest.h"

namespace {

TEST(SdmaEmitterTest, NativePacketLayouts) {
  std::array<uint32_t, 8> p = {};
  EXPECT_EQ(iree_hal_amdgpu_sdma_emit_copy(8, p.data(), 0x1234567890ull,
                                           0xabcdef1230ull, 4096),
            7u);
  EXPECT_EQ(p[0], 1u);
  EXPECT_EQ(p[1], 4095u);
  EXPECT_EQ(p[3], 0x34567890u);
  EXPECT_EQ(p[4], 0x12u);
  EXPECT_EQ(p[5], 0xcdef1230u);
  EXPECT_EQ(p[6], 0xabu);
  EXPECT_EQ(iree_hal_amdgpu_sdma_emit_poll32(8, p.data(), 0x1234567800ull, 7),
            6u);
  EXPECT_EQ(p[0], 0xb0000008u);
  EXPECT_EQ(p[3], 7u);
  EXPECT_EQ(p[4], 0xffffffffu);
  EXPECT_EQ(p[5], 0x0fff0004u);
  EXPECT_EQ(iree_hal_amdgpu_sdma_emit_fence32(8, p.data(), 0x1234567800ull, 9),
            4u);
  EXPECT_EQ(p[0], 0x00130005u);
  EXPECT_EQ(p[3], 9u);
}

TEST(SdmaEmitterTest, BoundsAndFailureDoNotWrite) {
  std::array<uint32_t, 8> p;
  p.fill(0xdeadbeef);
  const auto original = p;
  EXPECT_EQ(iree_hal_amdgpu_sdma_emit_copy(6, p.data(), 0, 0, 1), 0u);
  EXPECT_EQ(iree_hal_amdgpu_sdma_emit_copy(8, p.data(), 0, 0, 0), 0u);
  EXPECT_EQ(iree_hal_amdgpu_sdma_emit_copy(8, p.data(), 0, 0, (1u << 22) + 1),
            0u);
  EXPECT_EQ(iree_hal_amdgpu_sdma_emit_copy(8, p.data(), UINT64_MAX, 0, 2), 0u);
  EXPECT_EQ(iree_hal_amdgpu_sdma_emit_poll32(8, p.data(), 3, 1), 0u);
  EXPECT_EQ(iree_hal_amdgpu_sdma_emit_fence32(8, p.data(), 1, 1), 0u);
  EXPECT_EQ(p, original);
  EXPECT_EQ(iree_hal_amdgpu_sdma_emit_copy(8, p.data(), 0, 0, 1u << 22), 7u);
  EXPECT_EQ(p[1], 0x3fffffu);
}

TEST(SdmaEmitterTest, CacheControl) {
  uint32_t p[5];
  ASSERT_EQ(iree_hal_amdgpu_sdma_emit_gcr(5, p, false), 5u);
  EXPECT_EQ(p[0], 0x111u);
  EXPECT_EQ(p[2], 0x80400000u);
  ASSERT_EQ(iree_hal_amdgpu_sdma_emit_gcr(5, p, true), 5u);
  EXPECT_EQ(p[2], 0xc3c00000u);
}

TEST(SdmaEmitterTest, IndirectPlacementAndBounds) {
  std::array<uint32_t, 16> words;
  for (uint64_t offset = 0; offset < 16; ++offset) {
    words.fill(0xdeadbeef);
    uint32_t count = iree_hal_amdgpu_sdma_emit_indirect(
        words.size(), words.data(), offset, 0x1234567800ull, 128, 8,
        0xabcdef0000ull);
    ASSERT_GE(count, 6u);
    ASSERT_LE(count, 13u);
    EXPECT_EQ((offset + count) % 8, 0u);
    uint32_t padding = count - 6;
    for (uint32_t i = 0; i < padding; ++i) EXPECT_EQ(words[i], 0u);
    EXPECT_EQ(words[padding], 0x80004u);
    EXPECT_EQ(words[padding + 1], 0x34567800u);
    EXPECT_EQ(words[padding + 2], 0x12u);
    EXPECT_EQ(words[padding + 3], 128u);
    EXPECT_EQ(words[padding + 4], 0xcdef0000u);
    EXPECT_EQ(words[padding + 5], 0xabu);
    EXPECT_EQ(words[count], 0xdeadbeefu);
  }
  words.fill(0xdeadbeef);
  const auto original = words;
  EXPECT_EQ(iree_hal_amdgpu_sdma_emit_indirect(7, words.data(), 0, 32, 8, 0, 0),
            0u);
  EXPECT_EQ(
      iree_hal_amdgpu_sdma_emit_indirect(16, words.data(), 0, 33, 8, 0, 0), 0u);
  EXPECT_EQ(
      iree_hal_amdgpu_sdma_emit_indirect(16, words.data(), 0, 32, 0, 0, 0), 0u);
  EXPECT_EQ(iree_hal_amdgpu_sdma_emit_indirect(16, words.data(), 0, 32,
                                               1u << 20, 0, 0),
            0u);
  EXPECT_EQ(
      iree_hal_amdgpu_sdma_emit_indirect(16, words.data(), 0, 32, 8, 16, 0),
      0u);
  EXPECT_EQ(
      iree_hal_amdgpu_sdma_emit_indirect(16, words.data(), 0, 32, 8, 0, 3), 0u);
  EXPECT_EQ(iree_hal_amdgpu_sdma_emit_indirect(16, words.data(), 0,
                                               UINT64_MAX - 31, 9, 0, 0),
            0u);
  EXPECT_EQ(words, original);
}
}  // namespace
