// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/sdma/transfer.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "iree/hal/driver/amd/gpu/queue/buffer.h"
#include "iree/hal/driver/amd/gpu/queue/dma_transfer.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/commands.h"
#include "iree/hal/driver/amd/gpu/queue/transfer.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

uint64_t Address(const uint32_t* words) {
  return uint64_t{words[0]} | (uint64_t{words[1]} << 32);
}

// Byte oracle for the prepared data path, not a native ordering simulator.
// Physical CTS qualifies visibility and execution of these packet families.
void Apply(const iree_hal_amd_gpu_dma_transfer_chunk_t& chunk,
           amdf_queue_format_features_t features = 0) {
  for (size_t i = 0; i < chunk.word_count;) {
    const auto* words = chunk.words + i;
    if ((words[0] & 0xff) == 1) {
      if (((words[0] >> 8) & 0xff) == 4) {
        ASSERT_LE(i + 13, chunk.word_count);
        const uint32_t element = words[0] >> 29;
        const bool wide =
            (features & AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_WIDE) !=
            0;
        const uint32_t xy_mask = wide ? 0xffff : 0x3fff;
        const uint32_t pitch_shift = wide ? 16 : 13;
        const uint32_t z_mask =
            wide ? 0x3fff
            : features &
                    AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_EXTENDED_Z
                ? 0x1fff
                : 0x7ff;
        const uint64_t source_pitch = uint64_t{(words[4] >> pitch_shift) + 1}
                                      << element;
        const uint64_t target_pitch = uint64_t{(words[9] >> pitch_shift) + 1}
                                      << element;
        const uint64_t source_slice = (uint64_t{words[5]} + 1) << element;
        const uint64_t target_slice = (uint64_t{words[10]} + 1) << element;
        const uint32_t width = ((words[11] & xy_mask) + 1) << element;
        const uint32_t rows = (words[11] >> 16) + 1;
        const uint32_t slices = (words[12] & z_mask) + 1;
        const auto* source = reinterpret_cast<const uint8_t*>(
            Address(words + 1) + ((words[3] & xy_mask) << element));
        auto* target = reinterpret_cast<uint8_t*>(
            Address(words + 6) + ((words[8] & xy_mask) << element));
        for (uint32_t slice = 0; slice < slices; ++slice) {
          for (uint32_t row = 0; row < rows; ++row) {
            memcpy(target + slice * target_slice + row * target_pitch,
                   source + slice * source_slice + row * source_pitch, width);
          }
        }
        i += 13;
        continue;
      }
      ASSERT_LE(i + 7, chunk.word_count);
      memcpy(reinterpret_cast<void*>(Address(words + 5)),
             reinterpret_cast<const void*>(Address(words + 3)), words[1] + 1);
      i += 7;
    } else {
      ASSERT_EQ(words[0] & 0xff, 11u);
      ASSERT_LE(i + 5, chunk.word_count);
      auto* target = reinterpret_cast<uint8_t*>(Address(words + 1));
      for (uint32_t j = 0; j <= words[4]; ++j) {
        target[j] = static_cast<uint8_t>(words[3] >> ((j & 3) * 8));
      }
      i += 5;
    }
  }
}

const amdf_queue_format_features_t kRectangleFormats[] = {
    AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT,
    AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT |
        AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_EXTENDED_Z,
    AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT |
        AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_WIDE,
    AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT |
        AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_WIDE |
        AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE,
};

TEST(TransferTest, NativeRectangleFieldsMatchEachAdmittedLayout) {
  const iree_hal_transfer_region_t region = {
      .row_length = 37,
      .row_count = 7,
      .slice_count = 3,
      .source_strides = {48, 512},
      .target_strides = {64, 768},
  };
  for (auto features : kRectangleFormats) {
    const bool wide =
        (features & AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_WIDE) != 0;
    std::array<uint32_t, 15> words;
    words.fill(0x95d3c7a1);
    const auto packet = iree_hal_amd_gpu_sdma_transfer_emit_region(
        features, UINT64_C(0x123400002003), UINT64_C(0x567800004001), &region,
        0, 13, words.data() + 1);
    ASSERT_EQ(packet.word_count, 13u);
    EXPECT_EQ(packet.length, 37u * 7 * 3);
    EXPECT_EQ(packet.staging_length, 0u);
    EXPECT_EQ(
        (std::vector<uint32_t>{words.begin() + 1, words.end() - 1}),
        (std::vector<uint32_t>{
            0x401, 0x2000, 0x1234, 3, wide ? 0x002f0000u : 0x0005e000u, 511,
            0x4000, 0x5678, 1, wide ? 0x003f0000u : 0x0007e000u, 767,
            0x00060024,
            features & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE ? 0x0c0c0002u
                                                                 : 2u}));
    EXPECT_EQ(words.front(), 0x95d3c7a1u);
    EXPECT_EQ(words.back(), 0x95d3c7a1u);
    // Insufficient packet capacity and a partial row leave all storage alone.
    const auto previous = words;
    EXPECT_EQ(iree_hal_amd_gpu_sdma_transfer_emit_region(
                  features, 0x2000, 0x4000, &region, 0, 12, words.data() + 1)
                  .length,
              0u);
    EXPECT_EQ(iree_hal_amd_gpu_sdma_transfer_emit_region(
                  features, 0x2000, 0x4000, &region, 1, 13, words.data() + 1)
                  .length,
              0u);
    EXPECT_EQ(iree_hal_amd_gpu_sdma_transfer_emit_region(
                  0, 0x2000, 0x4000, &region, 0, 13, words.data() + 1)
                  .length,
              0u);
    EXPECT_EQ(words, previous);
  }
}

TEST(TransferTest, RectanglesPreserveBytesPaddingAndNativeScope) {
  for (auto features : kRectangleFormats) {
    for (uint64_t width : {1u, 3u, 16u, 37u, 256u}) {
      for (uint32_t displacement : {0u, 1u, 3u}) {
        for (uint64_t rows : {1u, 7u}) {
          SCOPED_TRACE(::testing::Message() << features << '/' << width << '/'
                                            << displacement << '/' << rows);
          const uint64_t source_pitch = (width + 11) & ~uint64_t{3};
          const uint64_t target_pitch = (width + 23) & ~uint64_t{3};
          const iree_hal_transfer_region_t region = {
              .row_length = width,
              .row_count = rows,
              .slice_count = 3,
              .source_strides = {source_pitch, rows * source_pitch + 64},
              .target_strides = {target_pitch, rows * target_pitch + 96},
          };
          std::vector<uint8_t> source(4 * region.source_strides.slice_stride);
          std::vector<uint8_t> target(4 * region.target_strides.slice_stride,
                                      0x71);
          for (size_t i = 0; i < source.size(); ++i) {
            source[i] = i * 17 + 31;
          }
          auto expected = target;
          for (uint32_t slice = 0; slice < 3; ++slice) {
            for (uint32_t row = 0; row < rows; ++row) {
              memcpy(expected.data() + 7 + displacement +
                         slice * region.target_strides.slice_stride +
                         row * target_pitch,
                     source.data() + displacement +
                         slice * region.source_strides.slice_stride +
                         row * source_pitch,
                     width);
            }
          }
          iree_hal_amd_gpu_transfer_operation_t operation = {
              .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
              .rectangular = true,
              .source =
                  {
                      .device = iree_hal_amd_gpu_buffer_reference_borrow(
                          reinterpret_cast<uintptr_t>(source.data() +
                                                      displacement)),
                  },
              .target =
                  {
                      .device = iree_hal_amd_gpu_buffer_reference_borrow(
                          reinterpret_cast<uintptr_t>(target.data() + 7 +
                                                      displacement)),
                  },
              .region = &region,
          };
          const auto encoding =
              iree_hal_amd_gpu_sdma_transfer_encoding(features);
          const iree_hal_amd_gpu_transfer_t transfer = {
              .count = 1,
              .operations = &operation,
          };
          iree_hal_amd_gpu_dma_transfer_chunk_t chunk;
          IREE_ASSERT_OK(iree_hal_amd_gpu_dma_transfer_prepare_chunk(
              &encoding, &transfer, 13, {}, 0, &chunk));
          ASSERT_EQ(chunk.word_count, 13u);
          EXPECT_EQ(chunk.next.index, 1u);
          EXPECT_EQ(chunk.staging_length, 0u);
          EXPECT_EQ(chunk.download.length, 0u);
          EXPECT_EQ(chunk.words[0] & 0x1fffffff, 0x401u);
          EXPECT_EQ(chunk.words[12] & 0x0c0c0000,
                    features & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE
                        ? 0x0c0c0000u
                        : 0u);
          ASSERT_NO_FATAL_FAILURE(Apply(chunk, features));
          EXPECT_EQ(target, expected);
        }
      }
    }
  }
}

TEST(TransferTest, RectangleLimitsPreserveTheExistingUsefulByteCursor) {
  for (auto features : kRectangleFormats) {
    const bool wide =
        (features & AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_WIDE) != 0;
    const uint32_t maximum_rows = 1u << (wide ? 16 : 14);
    const iree_hal_transfer_region_t region = {
        .row_length = 4,
        .row_count = maximum_rows + 3,
        .slice_count = 2,
        .source_strides = {8, uint64_t{maximum_rows + 3} * 8 + 16},
        .target_strides = {12, uint64_t{maximum_rows + 3} * 12 + 32},
    };
    std::vector<uint8_t> source(3 * region.source_strides.slice_stride);
    std::vector<uint8_t> target(3 * region.target_strides.slice_stride, 0x71);
    for (size_t i = 0; i < source.size(); ++i) {
      source[i] = i * 31 + 7;
    }
    auto expected = target;
    for (uint32_t slice = 0; slice < 2; ++slice) {
      for (uint32_t row = 0; row < region.row_count; ++row) {
        memcpy(expected.data() + slice * region.target_strides.slice_stride +
                   row * 12,
               source.data() + slice * region.source_strides.slice_stride +
                   row * 8,
               4);
      }
    }
    iree_hal_amd_gpu_transfer_operation_t operation = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
        .rectangular = true,
        .source =
            {
                .device = iree_hal_amd_gpu_buffer_reference_borrow(
                    reinterpret_cast<uintptr_t>(source.data())),
            },
        .target =
            {
                .device = iree_hal_amd_gpu_buffer_reference_borrow(
                    reinterpret_cast<uintptr_t>(target.data())),
            },
        .region = &region,
    };
    iree_hal_amd_gpu_transfer_t transfer = {
        .count = 1,
        .operations = &operation,
    };
    const auto encoding = iree_hal_amd_gpu_sdma_transfer_encoding(features);
    uint32_t packets = 0;
    while (transfer.cursor.index != 1) {
      iree_hal_amd_gpu_dma_transfer_chunk_t chunk, retry;
      IREE_ASSERT_OK(iree_hal_amd_gpu_dma_transfer_prepare_chunk(
          &encoding, &transfer, 13, {}, 0, &chunk));
      IREE_ASSERT_OK(iree_hal_amd_gpu_dma_transfer_prepare_chunk(
          &encoding, &transfer, 13, {}, 0, &retry));
      ASSERT_EQ(chunk.word_count, 13u);
      EXPECT_EQ(std::memcmp(chunk.words, retry.words, 13 * sizeof(uint32_t)),
                0);
      ASSERT_NO_FATAL_FAILURE(Apply(chunk, features));
      transfer.cursor = chunk.next;
      ++packets;
    }
    EXPECT_EQ(packets, 4u);
    EXPECT_EQ(target, expected);
  }
}

TEST(TransferTest, WidePitchesUseSlicesAndRespectNativeDepthLimits) {
  for (auto features : kRectangleFormats) {
    const bool wide =
        (features & AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_WIDE) != 0;
    const uint32_t depth =
        1u << (wide ? 14
               : features &
                       AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_EXTENDED_Z
                   ? 13
                   : 11);
    const uint64_t pitch = (UINT64_C(1) << (wide ? 16 : 19)) * 16 + 16;
    const iree_hal_transfer_region_t region = {
        .row_length = 16,
        .row_count = depth + 3,
        .slice_count = 1,
        .source_strides = {pitch, 0},
        .target_strides = {pitch + 16, 0},
    };
    std::array<uint32_t, 15> words;
    words.fill(0x95d3c7a1);
    const auto first = iree_hal_amd_gpu_sdma_transfer_emit_region(
        features, UINT64_C(0x123400000000), UINT64_C(0x567800000000), &region,
        0, 13, words.data() + 1);
    ASSERT_EQ(first.word_count, 13u);
    EXPECT_EQ(first.length, depth * 16);
    EXPECT_EQ(words[12] >> 16, 0u);
    EXPECT_EQ(words[13] & 0x3fff, depth - 1);
    EXPECT_EQ(words[6], pitch / 16 - 1);
    EXPECT_EQ(words[11], (pitch + 16) / 16 - 1);
    const auto second = iree_hal_amd_gpu_sdma_transfer_emit_region(
        features, UINT64_C(0x123400000000), UINT64_C(0x567800000000), &region,
        first.length, 13, words.data() + 1);
    EXPECT_EQ(second.length, 3u * 16);
    EXPECT_EQ(Address(words.data() + 2),
              UINT64_C(0x123400000000) + depth * pitch);
    EXPECT_EQ(words.front(), 0x95d3c7a1u);
    EXPECT_EQ(words.back(), 0x95d3c7a1u);
  }
}

TEST(TransferTest, RectanglesKeepBoundedPacketLengthsAndWideSlicePitches) {
  constexpr auto kFeatures = AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT |
                             AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_WIDE;
  constexpr uint64_t kSliceLength = UINT64_C(1024) * 65536;
  const iree_hal_transfer_region_t region = {
      .row_length = 1024,
      .row_count = 65536,
      .slice_count = 66,
      .source_strides = {1040, UINT64_C(1) << 36},
      .target_strides = {1056, (UINT64_C(1) << 36) - 16},
  };
  std::array<uint32_t, 13> words;
  const auto first = iree_hal_amd_gpu_sdma_transfer_emit_region(
      kFeatures, UINT64_C(0x123400000000), UINT64_C(0x567800000000), &region, 0,
      words.size(), words.data());
  ASSERT_EQ(first.word_count, 13u);
  EXPECT_EQ(first.length, 63 * kSliceLength);
  EXPECT_EQ(words[0], 0x80000401u);
  EXPECT_EQ(words[5], UINT32_MAX);
  EXPECT_EQ(words[10], UINT32_MAX - 1);
  EXPECT_EQ(words[12], 62u);
  const auto second = iree_hal_amd_gpu_sdma_transfer_emit_region(
      kFeatures, UINT64_C(0x123400000000), UINT64_C(0x567800000000), &region,
      first.length, words.size(), words.data());
  EXPECT_EQ(second.length, 3 * kSliceLength);
  EXPECT_EQ(Address(words.data() + 1),
            UINT64_C(0x123400000000) + 63 * region.source_strides.slice_stride);
  EXPECT_EQ(Address(words.data() + 6),
            UINT64_C(0x567800000000) + 63 * region.target_strides.slice_stride);
  EXPECT_EQ(words[12], 2u);
}

TEST(TransferTest, LinearPacketsPreservePartialRowsBroadcastAndOddPitches) {
  const iree_hal_transfer_region_t regions[] = {
      {7, 5, 3, {12, 68}, {20, 112}}, {7, 5, 3, {0, 16}, {12, 64}},
      {7, 5, 3, {12, 0}, {12, 64}},   {7, 5, 3, {13, 71}, {17, 91}},
      {7, 5, 3, {12, 65}, {20, 103}},
  };
  for (const auto& region : regions) {
    for (uint32_t capacity : {7u, 12u, 13u, 224u}) {
      for (auto features : kRectangleFormats) {
        std::array<uint8_t, 512> source, target;
        for (size_t i = 0; i < source.size(); ++i) {
          source[i] = i * 31 + 7;
        }
        target.fill(0x71);
        auto expected = target;
        uint64_t logical_offset = 0;
        for (uint32_t slice = 0; slice < region.slice_count; ++slice) {
          for (uint32_t row = 0; row < region.row_count; ++row) {
            for (uint32_t column = 0; column < region.row_length; ++column) {
              if (logical_offset++ < 3) {
                continue;
              }
              expected[3 + slice * region.target_strides.slice_stride +
                       row * region.target_strides.row_stride + column] =
                  source[1 + slice * region.source_strides.slice_stride +
                         row * region.source_strides.row_stride + column];
            }
          }
        }
        iree_hal_amd_gpu_transfer_operation_t operation = {
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
            .rectangular = true,
            .source =
                {
                    .device = iree_hal_amd_gpu_buffer_reference_borrow(
                        reinterpret_cast<uintptr_t>(source.data() + 1)),
                },
            .target =
                {
                    .device = iree_hal_amd_gpu_buffer_reference_borrow(
                        reinterpret_cast<uintptr_t>(target.data() + 3)),
                },
            .region = &region,
        };
        iree_hal_amd_gpu_transfer_t transfer = {
            .count = 1,
            .operations = &operation,
            .cursor = {0, 3},
        };
        const auto encoding = iree_hal_amd_gpu_sdma_transfer_encoding(features);
        while (transfer.cursor.index != 1) {
          iree_hal_amd_gpu_dma_transfer_chunk_t chunk;
          IREE_ASSERT_OK(iree_hal_amd_gpu_dma_transfer_prepare_chunk(
              &encoding, &transfer, capacity, {}, 0, &chunk));
          ASSERT_GT(chunk.word_count, 0u);
          ASSERT_LE(chunk.word_count, capacity);
          ASSERT_NO_FATAL_FAILURE(Apply(chunk, features));
          transfer.cursor = chunk.next;
        }
        EXPECT_EQ(target, expected);
      }
    }
  }
}

TEST(TransferTest, FileCopyStreamsFullAddressesWithoutPayloadStaging) {
  constexpr uint64_t kLength = (UINT64_C(5) << 30) + 17;
  constexpr uint64_t kSource = UINT64_C(0x1234fffffff3);
  constexpr uint64_t kTarget = UINT64_C(0x5678fffffffd);
  const iree_hal_amd_gpu_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
      .source = {.device = iree_hal_amd_gpu_buffer_reference_borrow(kSource)},
      .target = {.device = iree_hal_amd_gpu_buffer_reference_borrow(kTarget)},
      .length = kLength,
  };
  for (auto features : {amdf_queue_format_features_t{0},
                        amdf_queue_format_features_t{
                            AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE}}) {
    const auto encoding = iree_hal_amd_gpu_sdma_transfer_encoding(features);
    auto owned_operation = operation;
    iree_hal_amd_gpu_transfer_t transfer = {
        .count = 1,
        .operations = &owned_operation,
    };
    uint64_t total = 0;
    while (transfer.cursor.index < transfer.count) {
      iree_hal_amd_gpu_dma_transfer_chunk_t chunk, retry;
      IREE_ASSERT_OK(iree_hal_amd_gpu_dma_transfer_prepare_chunk(
          &encoding, &transfer, IREE_HAL_AMD_GPU_DMA_TRANSFER_MAX_WORDS,
          iree_byte_span_empty(), 0, &chunk));
      // Native backpressure may retry this candidate before accepting it.
      IREE_ASSERT_OK(iree_hal_amd_gpu_dma_transfer_prepare_chunk(
          &encoding, &transfer, IREE_HAL_AMD_GPU_DMA_TRANSFER_MAX_WORDS,
          iree_byte_span_empty(), 0, &retry));
      ASSERT_GT(chunk.word_count, 0u);
      ASSERT_EQ(chunk.word_count % 7, 0u);
      EXPECT_EQ(chunk.staging_length, 0u);
      EXPECT_EQ(chunk.download.length, 0u);
      EXPECT_EQ(
          std::vector<uint32_t>(chunk.words, chunk.words + chunk.word_count),
          std::vector<uint32_t>(retry.words, retry.words + retry.word_count));
      for (uint32_t i = 0; i < chunk.word_count; i += 7) {
        const uint32_t length = static_cast<uint32_t>(std::min<uint64_t>(
            kLength - total, IREE_HAL_AMD_GPU_SDMA_MAX_PACKET_LENGTH));
        const uint64_t source = kSource + total, target = kTarget + total;
        EXPECT_EQ(
            std::vector<uint32_t>(chunk.words + i, chunk.words + i + 7),
            (std::vector<uint32_t>{1, length - 1, features ? 0x0c0c0000u : 0u,
                                   static_cast<uint32_t>(source),
                                   static_cast<uint32_t>(source >> 32),
                                   static_cast<uint32_t>(target),
                                   static_cast<uint32_t>(target >> 32)}));
        total += length;
      }
      transfer.cursor = chunk.next;
    }
    EXPECT_EQ(total, kLength);
  }
}

TEST(TransferTest, MixedPacketsKeepStagingAndDownloadOwnershipExact) {
  for (uint32_t word_capacity : {7u, 12u, 224u}) {
    for (size_t staging_capacity : {1u, 7u, 32u, 127u}) {
      SCOPED_TRACE(::testing::Message()
                   << word_capacity << '/' << staging_capacity);
      alignas(64) std::array<uint8_t, 256> payload, source, expected;
      std::array<uint8_t, 258> output;
      payload.fill(0x39);
      output.fill(0x71);
      for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<uint8_t>(0x51 + i * 13);
      }
      expected = payload;
      std::array<iree_hal_amd_gpu_transfer_operation_t, 7> operations = {};
      operations[0] = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
          .pattern = 0x91c37a25,
          .target =
              {
                  .device = iree_hal_amd_gpu_buffer_reference_borrow(
                      reinterpret_cast<uintptr_t>(payload.data() + 1)),
              },
          .length = 13,
      };
      for (uint32_t i = 0; i < 13; ++i) {
        expected[1 + i] = static_cast<uint8_t>(0x91c37a25u >> ((i & 3) * 8));
      }
      for (uint32_t i = 1; i <= 3; ++i) {
        operations[i] = {
            .type = static_cast<uint32_t>(
                i == 1   ? IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE
                : i == 2 ? IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD
                         : IREE_HAL_TRANSFER_OPERATION_TYPE_COPY),
            .target =
                {
                    .device = iree_hal_amd_gpu_buffer_reference_borrow(
                        reinterpret_cast<uintptr_t>(payload.data() + i * 40)),
                },
            .length = 31,
        };
        if (i == 3) {
          operations[i].source.device =
              iree_hal_amd_gpu_buffer_reference_borrow(
                  reinterpret_cast<uintptr_t>(source.data() + i * 40));
        } else {
          operations[i].source.host = source.data() + i * 40;
        }
        memcpy(expected.data() + i * 40, source.data() + i * 40, 31);
      }
      // Adjacent downloads share a retirement copy. A following operation ends
      // that run so it cannot reuse bytes still belonging to the host copy.
      for (uint32_t i = 0; i < 2; ++i) {
        operations[4 + i] = {
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
            .source =
                {
                    .device = iree_hal_amd_gpu_buffer_reference_borrow(
                        reinterpret_cast<uintptr_t>(payload.data() + i * 128)),
                },
            .target = {.host = output.data() + 1 + i * 128},
            .length = 128,
        };
      }
      operations[6] = operations[0];
      operations[6].pattern = 0xeca57231;
      iree_hal_amd_gpu_transfer_t transfer = {
          .count = operations.size(),
          .operations = operations.data(),
      };
      const auto encoding = iree_hal_amd_gpu_sdma_transfer_encoding(
          AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE);
      while (transfer.cursor.index < transfer.count) {
        std::array<uint8_t, 130> staging;
        staging.fill(0xda);
        iree_hal_amd_gpu_dma_transfer_chunk_t chunk;
        std::memset(&chunk, 0xa5, sizeof(chunk));
        IREE_ASSERT_OK(iree_hal_amd_gpu_dma_transfer_prepare_chunk(
            &encoding, &transfer, word_capacity,
            iree_make_byte_span(staging.data() + 1, staging_capacity),
            reinterpret_cast<uintptr_t>(staging.data() + 1), &chunk));
        ASSERT_GT(chunk.word_count, 0u);
        ASSERT_LE(chunk.word_count, word_capacity);
        EXPECT_LE(chunk.staging_length, staging_capacity);
        for (size_t i = chunk.word_count; i < std::size(chunk.words); ++i) {
          EXPECT_EQ(chunk.words[i], 0xa5a5a5a5u);
        }
        ASSERT_NO_FATAL_FAILURE(Apply(chunk));
        iree_hal_amd_gpu_download_complete(&chunk.download);
        EXPECT_EQ(staging.front(), 0xda);
        for (size_t i = 1 + chunk.staging_length; i < staging.size(); ++i) {
          EXPECT_EQ(staging[i], 0xda);
        }
        transfer.cursor = chunk.next;
      }
      EXPECT_EQ(output.front(), 0x71);
      EXPECT_EQ(output.back(), 0x71);
      EXPECT_TRUE(
          std::equal(expected.begin(), expected.end(), output.begin() + 1));
      for (uint32_t i = 0; i < 13; ++i) {
        expected[1 + i] = static_cast<uint8_t>(0xeca57231u >> ((i & 3) * 8));
      }
      EXPECT_EQ(payload, expected);
    }
  }
}

}  // namespace
