// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/sdma/encoding/commands.h"

#include <array>

#include "gtest/gtest.h"

namespace {

// Native family shapes: legacy, classic fence, explicit system, and scoped.
constexpr std::array<amdf_queue_format_features_t, 4> kFeatures = {
    0,
    AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_MEMORY_TYPE,
    AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM,
    AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM |
        AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE,
};

TEST(SdmaEncodingTest, UserGcrUsesWholeCacheDataAcquireAndRelease) {
  std::array<uint32_t, 11> words;
  words.fill(0x9ac7135b);
  SdmaCommandWriter commands(words.data(),
                             AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR);
  commands.AcquireFromSystem();
  commands.ReleaseToSystem();
  // ROCr BuildGCRCommand: USER suboperation, zero range/base/limit, GL2/GLK
  // writeback on both paths, and GL2/GL1/GLV/GLK invalidation on acquire.
  const std::array<uint32_t, 11> expected = {0x00000111, 0, 0xc3c00000, 0, 0,
                                             0x00000111, 0, 0x80400000, 0, 0,
                                             0x9ac7135b};
  EXPECT_EQ(commands.word_count(), 10u);
  EXPECT_EQ(words, expected);
}

TEST(SdmaEncodingTest, LinearByteCountAndUncachedCompletion) {
  std::array<uint32_t, 12> words = {};
  words.back() = 0x24681357;
  SdmaCommandWriter commands(words.data(),
                             AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_MEMORY_TYPE);
  commands.CopyLinear(UINT64_C(0x1234567887654320),
                      UINT64_C(0x2345678998765430), 1028);
  commands.Fence32(UINT64_C(0x34567890abcdef00), 19);
  // COPY_LINEAR carries bytes-minus-one; FENCE is an independent four-DWORD
  // packet, with uncached MTYPE and no implicit GCR or scope fields.
  const std::array<uint32_t, 11> expected = {
      1,          1027,       0,          0x87654320, 0x12345678, 0x98765430,
      0x23456789, 0x00030005, 0xabcdef00, 0x34567890, 19};
  ASSERT_EQ(commands.word_count(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(words[i], expected[i]) << i;
  }
  EXPECT_EQ(words.back(), 0x24681357u);
}

TEST(SdmaEncodingTest, GlobalTimestampUsesFullAddressAndNoImplicitFence) {
  // ROCr's scoped timestamp adds SYS at header bits25:24. Neither form
  // includes a fence or changes the three-DWORD packet extent.
  constexpr std::array<uint32_t, 4> kHeaders = {0x0000020d, 0x0000020d,
                                                0x0000020d, 0x0300020d};
  for (size_t i = 0; i < kFeatures.size(); ++i) {
    SCOPED_TRACE(kFeatures[i]);
    std::array<uint32_t, 4> words = {0, 0, 0, 0x9876abcd};
    SdmaCommandWriter commands(words.data(), kFeatures[i]);
    commands.WriteGlobalTimestamp(UINT64_C(0x12345678abcdef20));
    const std::array<uint32_t, 4> expected = {kHeaders[i], 0xabcdef20,
                                              0x12345678, 0x9876abcd};
    ASSERT_EQ(commands.word_count(), 3u);
    EXPECT_EQ(words, expected);
  }
}

TEST(SdmaEncodingTest, DependentCopiesHaveOneDwordNopAndFinalFence) {
  std::array<uint32_t, 20> words = {};
  words.back() = 0x24681357;
  SdmaCommandWriter commands(words.data(),
                             AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_MEMORY_TYPE);
  commands.CopyLinear(UINT64_C(0x1234567887654380),
                      UINT64_C(0x2345678998765500), 4096);
  commands.Noop();
  commands.CopyLinear(UINT64_C(0x2345678998765500),
                      UINT64_C(0x3456789aabcdef80), 4096);
  commands.Fence32(UINT64_C(0x456789ab12345640), 2);
  // PAL and Mesa use the zero header with no NOP body. The next copy reads
  // the first destination; only the final classic UC3 FENCE publishes a word.
  const std::array<uint32_t, 20> expected = {
      1,          0x00000fff, 0,          0x87654380, 0x12345678,
      0x98765500, 0x23456789, 0,          1,          0x00000fff,
      0,          0x98765500, 0x23456789, 0xabcdef80, 0x3456789a,
      0x00030005, 0x12345640, 0x456789ab, 2,          0x24681357};
  EXPECT_EQ(commands.word_count(), 19u);
  EXPECT_EQ(words, expected);
}

TEST(SdmaEncodingTest, DwordFillHasByteCountAndTargetScope) {
  constexpr std::array<uint32_t, 4> kHeaders = {0x8000000b, 0x8000000b,
                                                0x8000000b, 0x8300000b};
  constexpr std::array<uint32_t, 3> kByteLengths = {4, 8, 1028};
  constexpr std::array<uint32_t, 3> kCounts = {0x00000003, 0x00000007,
                                               0x00000403};
  for (size_t feature_index = 0; feature_index < kFeatures.size();
       ++feature_index) {
    SCOPED_TRACE(kFeatures[feature_index]);
    for (size_t i = 0; i < kByteLengths.size(); ++i) {
      SCOPED_TRACE(kByteLengths[i]);
      std::array<uint32_t, 6> words = {};
      words.back() = 0x24681357;
      SdmaCommandWriter commands(words.data(), kFeatures[feature_index]);
      commands.Fill32(UINT64_C(0x1234567887654320), 0x6d2ac491,
                      kByteLengths[i]);
      // PAL c5e800072a32 WriteFillMemoryCmd and Mesa 0ba4b08edc65
      // ac_emit_sdma_constant_fill use opcode11, fillsize2 and bytes-minus-one.
      // The low two count bits are ignored in DWORD mode. ROCr places SYS at
      // header bits25:24; NPD at bit29 remains zero for dependent streams.
      const std::array<uint32_t, 6> expected = {kHeaders[feature_index],
                                                0x87654320,
                                                0x12345678,
                                                0x6d2ac491,
                                                kCounts[i],
                                                0x24681357};
      EXPECT_EQ(commands.word_count(), 5u);
      EXPECT_EQ(words, expected);
    }
  }
}

TEST(SdmaEncodingTest, FenceFieldsFollowTheAdvertisedEncoding) {
  // ROCr BuildFenceCommand uses opcode-only for gfx9, UC3 for gfx10/11,
  // UC3 plus SYS for gfx12, and system scope for the scoped packet layout.
  constexpr std::array<uint32_t, 4> kHeaders = {0x00000005, 0x00030005,
                                                0x00130005, 0x03130005};
  for (size_t i = 0; i < kFeatures.size(); ++i) {
    SCOPED_TRACE(kFeatures[i]);
    std::array<uint32_t, 5> words = {0, 0, 0, 0, 0x72349681};
    SdmaCommandWriter commands(words.data(), kFeatures[i]);
    commands.Fence32(UINT64_C(0x1234567887654320), 0x98765432);
    const std::array<uint32_t, 5> expected = {
        kHeaders[i], 0x87654320, 0x12345678, 0x98765432, 0x72349681};
    EXPECT_EQ(commands.word_count(), 4u);
    EXPECT_EQ(words, expected);
  }
}

TEST(SdmaEncodingTest, MemoryEqualityPollWaitsWithoutFiniteRetryLimit) {
  constexpr std::array<uint32_t, 4> kRetryScopes = {0x0fff0004, 0x0fff0004,
                                                    0x0fff0004, 0x3fff0004};
  for (size_t i = 0; i < kFeatures.size(); ++i) {
    SCOPED_TRACE(kFeatures[i]);
    for (uint32_t value : {0u, 0x98765432u}) {
      SCOPED_TRACE(value);
      std::array<uint32_t, 7> words = {};
      words.back() = 0x72349681;
      SdmaCommandWriter commands(words.data(), kFeatures[i]);
      commands.WaitMemory32(UINT64_C(0x1234567887654320), value);
      // ROCr BuildPollCommand: memory=bit31, equality=3 at bits30:28,
      // full mask, interval4 and the 12-bit retry-forever value. The scoped
      // layout adds SYS at DW5 bits29:28, not the header's function field.
      const std::array<uint32_t, 7> expected = {
          0xb0000008, 0x87654320,      0x12345678, value,
          0xffffffff, kRetryScopes[i], 0x72349681};
      EXPECT_EQ(commands.word_count(), 6u);
      EXPECT_EQ(words, expected);
    }
  }
}

TEST(SdmaEncodingTest, AtLeastMemoryPollPreservesScopeAndRetryFields) {
  constexpr std::array<uint32_t, 4> kRetryScopes = {0x0fff0004, 0x0fff0004,
                                                    0x0fff0004, 0x3fff0004};
  for (size_t i = 0; i < kFeatures.size(); ++i) {
    SCOPED_TRACE(kFeatures[i]);
    std::array<uint32_t, 7> words = {};
    words.back() = 0x72349681;
    SdmaCommandWriter commands(words.data(), kFeatures[i]);
    commands.WaitMemory32(UINT64_C(0x1234567887654320), 0x12345,
                          SdmaMemoryComparison::kGreaterOrEqual);
    // Mesa's SDMA gang join passes comparison 5 to ac_emit_sdma_wait_mem.
    // The comparison occupies the header; it changes neither the full mask
    // nor the independent retry/scope word used by the ordinary poll form.
    const std::array<uint32_t, 7> expected = {
        0xd0000008, 0x87654320,      0x12345678, 0x00012345,
        0xffffffff, kRetryScopes[i], 0x72349681};
    EXPECT_EQ(commands.word_count(), 6u);
    EXPECT_EQ(words, expected);
  }
}

TEST(SdmaEncodingTest, LinearShortTransfersKeepByteCountUnits) {
  constexpr std::array<uint32_t, 4> kParameters = {0, 0, 0, 0x0c0c0000};
  for (size_t i = 0; i < kFeatures.size(); ++i) {
    SCOPED_TRACE(kFeatures[i]);
    for (uint32_t byte_length : {1u, 2u, 3u, 4u, 31u, 4101u}) {
      SCOPED_TRACE(byte_length);
      std::array<uint32_t, 8> words = {};
      words.back() = 0x31415926;
      SdmaCommandWriter commands(words.data(), kFeatures[i]);
      commands.CopyLinear(UINT64_C(0x1234567800000fff),
                          UINT64_C(0x2345678900001003), byte_length);
      ASSERT_EQ(commands.word_count(), 7u);
      EXPECT_EQ(words[0], 1u);  // NPD stays clear in every layout.
      EXPECT_EQ(words[1], byte_length - 1);
      EXPECT_EQ(words[2], kParameters[i]);
      EXPECT_EQ(words[3], 0x00000fffu);
      EXPECT_EQ(words[4], 0x12345678u);
      EXPECT_EQ(words[5], 0x00001003u);
      EXPECT_EQ(words[6], 0x23456789u);
      EXPECT_EQ(words.back(), 0x31415926u);
    }
  }
}

}  // namespace
