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

TEST(SdmaEncodingTest, InlineWritesCarryDwordCountsAndCopiedValues) {
  constexpr std::array<uint32_t, 4> kScopes = {0, 0, 0, 0x0c000000};
  for (size_t i = 0; i < kFeatures.size(); ++i) {
    SCOPED_TRACE(kFeatures[i]);
    std::array<uint32_t, 14> words;
    words.fill(0x9ac7135b);
    std::array<uint32_t, 3> values = {0x6d2ac491, 0xb730e85a, 0x1fe43962};
    SdmaCommandWriter commands(words.data(), kFeatures[i]);
    commands.WriteLinear(UINT64_C(0x1234567800000ffc),
                         std::span(values).first(1));
    commands.WriteLinear(UINT64_C(0x2345678900001ffc), values);
    // Inline data has been copied into the stream before publication. A
    // subsequent packet starts after its complete data, not after the header.
    values.fill(0);
    commands.Noop();
    // KFD SDMAWriteDataPacket and PAL BuildUpdateMemoryPacket encode DWORD
    // counts minus one. Only the scoped layout adds SYS at DW3 bits27:26.
    const std::array<uint32_t, 14> expected = {
        2,          0x00000ffc, 0x12345678, kScopes[i],     0x6d2ac491,
        2,          0x00001ffc, 0x23456789, kScopes[i] | 2, 0x6d2ac491,
        0xb730e85a, 0x1fe43962, 0,          0x9ac7135b};
    EXPECT_EQ(commands.word_count(), 13u);
    EXPECT_EQ(words, expected);
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

TEST(SdmaEncodingTest, NativeEventMailboxAndInterruptKeepDistinctOperands) {
  constexpr std::array<uint32_t, 4> kHeaders = {0x00000005, 0x00030005,
                                                0x00130005, 0x03130005};
  const amdf_gpu_event_info_t event = {
      .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_INFO,
      .structure_size = sizeof(event),
      .mailbox_address = UINT64_C(0x1234567887654320),
      .trigger_data = 23,
      .interrupt_context = 11,
  };
  for (size_t i = 0; i < kFeatures.size(); ++i) {
    SCOPED_TRACE(kFeatures[i]);
    std::array<uint32_t, 7> words;
    words.fill(0x9ac7135b);
    SdmaCommandWriter commands(words.data(), kFeatures[i]);
    commands.NotifyKfdEvent(event);
    // ROCr's notification tail writes one mailbox DWORD before the separate
    // TRAP context. FENCE memory policy follows the queue's native layout.
    const std::array<uint32_t, 7> expected = {
        kHeaders[i], 0x87654320, 0x12345678, 23, 6, 11, 0x9ac7135b};
    EXPECT_EQ(commands.word_count(), 6u);
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

TEST(SdmaEncodingTest, MemoryComparisonFamilyPreservesMaskScopeAndRetryFields) {
  constexpr std::array<SdmaMemoryComparison, 7> kComparisons = {
      SdmaMemoryComparison::kAlways,      SdmaMemoryComparison::kLess,
      SdmaMemoryComparison::kLessOrEqual, SdmaMemoryComparison::kEqual,
      SdmaMemoryComparison::kNotEqual,    SdmaMemoryComparison::kGreaterOrEqual,
      SdmaMemoryComparison::kGreater};
  // The classic POLL_REGMEM selector is in header bits 30:28. Each literal
  // includes memory-space selection and opcode 8; selector 7 is not admitted.
  constexpr std::array<uint32_t, 7> kHeaders = {
      0x80000008, 0x90000008, 0xa0000008, 0xb0000008,
      0xc0000008, 0xd0000008, 0xe0000008};
  constexpr std::array<uint32_t, 4> kRetryScopes = {0x0fff0004, 0x0fff0004,
                                                    0x0fff0004, 0x3fff0004};
  for (size_t i = 0; i < kFeatures.size(); ++i) {
    SCOPED_TRACE(kFeatures[i]);
    for (size_t j = 0; j < kComparisons.size(); ++j) {
      SCOPED_TRACE(j);
      for (uint32_t mask : {UINT32_MAX, 0x00ffff00u}) {
        SCOPED_TRACE(mask);
        std::array<uint32_t, 7> words = {};
        words.back() = 0x72349681;
        SdmaCommandWriter commands(words.data(), kFeatures[i]);
        commands.WaitMemory32(UINT64_C(0x1234567887654320), 0x00123400,
                              kComparisons[j], mask);
        const std::array<uint32_t, 7> expected = {
            kHeaders[j], 0x87654320,      0x12345678, 0x00123400,
            mask,        kRetryScopes[i], 0x72349681};
        EXPECT_EQ(commands.word_count(), 6u);
        EXPECT_EQ(words, expected);
      }
    }
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

constexpr std::array<amdf_queue_format_features_t, 4> kRectangleFeatures = {
    AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT,
    AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT |
        AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_EXTENDED_Z |
        AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_MEMORY_TYPE |
        AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR,
    AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT |
        AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_WIDE |
        AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM |
        AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR,
    AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT |
        AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_WIDE |
        AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM |
        AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE,
};

TEST(SdmaEncodingTest, RectangularElementsAndLayoutsHaveExactOperands) {
  constexpr std::array<uint32_t, 5> kHeaders = {
      0x00000401, 0x20000401, 0x40000401, 0x60000401, 0x80000401};
  constexpr std::array<uint32_t, 4> kSourcePitchZ = {0x0007e001, 0x0007e001,
                                                     0x003f0001, 0x003f0001};
  constexpr std::array<uint32_t, 4> kTargetPitchZ = {0x000be002, 0x000be002,
                                                     0x005f0002, 0x005f0002};
  constexpr std::array<uint32_t, 4> kDepthScopes = {1, 1, 1, 0x0c0c0001};
  for (size_t layout = 0; layout < kRectangleFeatures.size(); ++layout) {
    SCOPED_TRACE(layout);
    for (uint32_t element_log2 = 0; element_log2 < kHeaders.size();
         ++element_log2) {
      SCOPED_TRACE(element_log2);
      std::array<uint32_t, 14> words;
      words.fill(0xdeadbeef);
      SdmaCommandWriter commands(words.data(), kRectangleFeatures[layout]);
      commands.CopyLinearRect(0x1122334455667780, {3, 2, 1, 64, 512},
                              0x8877665544332200, {5, 1, 2, 96, 960},
                              {17, 3, 2}, element_log2);
      // Only geometry placement and separately admitted scope differ. The
      // header's NPD remains zero even for the scoped form.
      const std::array<uint32_t, 14> expected = {kHeaders[element_log2],
                                                 0x55667780,
                                                 0x11223344,
                                                 0x00020003,
                                                 kSourcePitchZ[layout],
                                                 0x000001ff,
                                                 0x44332200,
                                                 0x88776655,
                                                 0x00010005,
                                                 kTargetPitchZ[layout],
                                                 0x000003bf,
                                                 0x00020010,
                                                 kDepthScopes[layout],
                                                 0xdeadbeef};
      EXPECT_EQ(commands.word_count(), 13u);
      EXPECT_EQ(words, expected);
    }
  }
}

TEST(SdmaEncodingTest, RectangularZCoordinatesAndDepthUseTheirFullFields) {
  constexpr std::array<uint32_t, 4> kDepthCounts = {2048, 8192, 16384, 16384};
  constexpr std::array<uint32_t, 4> kSourcePitchZ = {0x000067fe, 0x00007ffe,
                                                     0x00033ffe, 0x00033ffe};
  constexpr std::array<uint32_t, 4> kTargetPitchZ = {0x000067ff, 0x00007fff,
                                                     0x00033fff, 0x00033fff};
  constexpr std::array<uint32_t, 4> kPitch = {0x6000, 0x6000, 0x30000, 0x30000};
  constexpr std::array<uint32_t, 4> kScopes = {0, 0, 0, 0x0c0c0000};
  constexpr std::array<uint32_t, 4> kDepthScopes = {0x000007ff, 0x00001fff,
                                                    0x00003fff, 0x0c0c3fff};
  for (size_t layout = 0; layout < kRectangleFeatures.size(); ++layout) {
    SCOPED_TRACE(layout);
    std::array<uint32_t, 27> words;
    words.fill(0xdeadbeef);
    SdmaCommandWriter commands(words.data(), kRectangleFeatures[layout]);
    commands.CopyLinearRect(0x100000000, {0, 0, kDepthCounts[layout] - 2, 4, 4},
                            0x200000000, {0, 0, kDepthCounts[layout] - 1, 4, 4},
                            {1, 1, 1}, 0);
    commands.CopyLinearRect(0x100000000, {0, 0, 0, 4, 4}, 0x200000000,
                            {0, 0, 0, 4, 4}, {1, 1, kDepthCounts[layout]}, 0);
    const std::array<uint32_t, 27> expected = {0x401,
                                               0,
                                               1,
                                               0,
                                               kSourcePitchZ[layout],
                                               3,
                                               0,
                                               2,
                                               0,
                                               kTargetPitchZ[layout],
                                               3,
                                               0,
                                               kScopes[layout],
                                               0x401,
                                               0,
                                               1,
                                               0,
                                               kPitch[layout],
                                               3,
                                               0,
                                               2,
                                               0,
                                               kPitch[layout],
                                               3,
                                               0,
                                               kDepthScopes[layout],
                                               0xdeadbeef};
    EXPECT_EQ(commands.word_count(), 26u);
    EXPECT_EQ(words, expected);
  }
}

TEST(SdmaEncodingTest, RectangularPitchCountsPreserveTheWideSliceLimit) {
  constexpr std::array<uint32_t, 4> kRowCounts = {524288, 524288, 65536, 65536};
  constexpr std::array<uint64_t, 4> kSliceCounts = {
      UINT64_C(1) << 28, UINT64_C(1) << 28, UINT64_C(1) << 32,
      UINT64_C(1) << 32};
  constexpr std::array<uint32_t, 4> kPitches = {0xffffe000, 0xffffe000,
                                                0xffff0000, 0xffff0000};
  constexpr std::array<uint32_t, 4> kSlices = {0x0fffffff, 0x0fffffff,
                                               0xffffffff, 0xffffffff};
  constexpr std::array<uint32_t, 4> kScopes = {0, 0, 0, 0x0c0c0000};
  for (size_t layout = 0; layout < kRectangleFeatures.size(); ++layout) {
    SCOPED_TRACE(layout);
    std::array<uint32_t, 14> words;
    words.fill(0xdeadbeef);
    SdmaCommandWriter commands(words.data(), kRectangleFeatures[layout]);
    const SdmaLinearLayout geometry = {0, 0, 0, kRowCounts[layout],
                                       kSliceCounts[layout]};
    commands.CopyLinearRect(0x100000000, geometry, 0x200000000, geometry,
                            {1, 1, 1}, 0);
    const std::array<uint32_t, 14> expected = {0x401,
                                               0,
                                               1,
                                               0,
                                               kPitches[layout],
                                               kSlices[layout],
                                               0,
                                               2,
                                               0,
                                               kPitches[layout],
                                               kSlices[layout],
                                               0,
                                               kScopes[layout],
                                               0xdeadbeef};
    EXPECT_EQ(commands.word_count(), 13u);
    EXPECT_EQ(words, expected);
  }
}

}  // namespace
