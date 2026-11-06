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
void Apply(const iree_hal_amd_gpu_dma_transfer_chunk_t& chunk) {
  for (size_t i = 0; i < chunk.word_count;) {
    const auto* words = chunk.words + i;
    if ((words[0] & 0xff) == 1) {
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
            .type = i == 1   ? IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE
                    : i == 2 ? IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD
                             : IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
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
