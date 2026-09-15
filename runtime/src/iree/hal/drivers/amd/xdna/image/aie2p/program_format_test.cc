// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/aie2p/program_format.h"

#include <array>
#include <cstdint>
#include <initializer_list>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

template <size_t N>
static void ExpectBytes(const std::array<uint8_t, N>& actual,
                        std::initializer_list<uint8_t> expected) {
  ASSERT_EQ(expected.size(), actual.size());
  size_t i = 0;
  for (uint8_t value : expected) {
    EXPECT_EQ(value, actual[i]) << "byte offset " << i;
    ++i;
  }
}

TEST(Aie2pProgramFormatTest, RegisterWrite32RoundTrips) {
  const iree_hal_amd_xdna_aie2p_register_write32_t value = {
      /*.address=*/UINT32_C(0x12345678),
      /*.value=*/UINT32_C(0x90ABCDEF),
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_AIE2P_REGISTER_WRITE32_RECORD_SIZE>
      storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_register_write32(
      &value, iree_make_byte_span(storage.data(), storage.size())));
  ExpectBytes(storage, {
                           0x01,
                           0x00,
                           0x00,
                           0x00,
                           0x10,
                           0x00,
                           0x00,
                           0x00,
                           0x78,
                           0x56,
                           0x34,
                           0x12,
                           0xEF,
                           0xCD,
                           0xAB,
                           0x90,
                       });

  iree_hal_amd_xdna_aie2p_program_record_t decoded;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_decode_program_record(
      iree_make_const_byte_span(storage.data(), storage.size()), &decoded));
  EXPECT_EQ(IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32,
            decoded.type);
  EXPECT_EQ(value.address, decoded.value.register_write32.address);
  EXPECT_EQ(value.value, decoded.value.register_write32.value);
}

TEST(Aie2pProgramFormatTest, RegisterMaskWrite32RoundTrips) {
  const iree_hal_amd_xdna_aie2p_register_mask_write32_t value = {
      /*.address=*/UINT32_C(0x00112244),
      /*.mask=*/UINT32_C(0x00FF00FF),
      /*.value=*/UINT32_C(0x005500AA),
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_AIE2P_REGISTER_MASK_WRITE32_RECORD_SIZE>
      storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_register_mask_write32(
      &value, iree_make_byte_span(storage.data(), storage.size())));
  ExpectBytes(storage,
              {
                  0x02, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x44, 0x22,
                  0x11, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xAA, 0x00, 0x55, 0x00,
              });

  iree_hal_amd_xdna_aie2p_program_record_t decoded;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_decode_program_record(
      iree_make_const_byte_span(storage.data(), storage.size()), &decoded));
  EXPECT_EQ(IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32,
            decoded.type);
  EXPECT_EQ(value.address, decoded.value.register_mask_write32.address);
  EXPECT_EQ(value.mask, decoded.value.register_mask_write32.mask);
  EXPECT_EQ(value.value, decoded.value.register_mask_write32.value);
}

TEST(Aie2pProgramFormatTest, RegisterBlockWrite32RoundTrips) {
  constexpr std::array<uint32_t, 3> kWords = {
      UINT32_C(0x11223344),
      UINT32_C(0xAABBCCDD),
      UINT32_C(0x01020304),
  };
  iree_host_size_t byte_length = 0;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_measure_register_block_write32(
      kWords.size(), &byte_length));
  ASSERT_EQ(28u, byte_length);
  std::array<uint8_t, 28> storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_register_block_write32(
      UINT32_C(0x00102030), kWords.size(), kWords.data(),
      iree_make_byte_span(storage.data(), storage.size())));
  ExpectBytes(storage,
              {
                  0x03, 0x00, 0x00, 0x00, 0x1C, 0x00, 0x00, 0x00, 0x30, 0x20,
                  0x10, 0x00, 0x03, 0x00, 0x00, 0x00, 0x44, 0x33, 0x22, 0x11,
                  0xDD, 0xCC, 0xBB, 0xAA, 0x04, 0x03, 0x02, 0x01,
              });

  iree_hal_amd_xdna_aie2p_program_record_t decoded;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_decode_program_record(
      iree_make_const_byte_span(storage.data(), storage.size()), &decoded));
  EXPECT_EQ(IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32,
            decoded.type);
  EXPECT_EQ(UINT32_C(0x00102030), decoded.value.register_block_write32.address);
  EXPECT_EQ(kWords.size(), decoded.value.register_block_write32.word_count);
  EXPECT_EQ(kWords.size() * sizeof(uint32_t),
            decoded.value.register_block_write32.word_data.data_length);
  for (iree_host_size_t i = 0; i < kWords.size(); ++i) {
    uint32_t word = 0;
    IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_program_record_read_block_word(
        &decoded, i, &word));
    EXPECT_EQ(kWords[i], word);
  }
}

TEST(Aie2pProgramFormatTest, TileProgramLoadRoundTrips) {
  const iree_hal_amd_xdna_aie2p_tile_program_load_t value = {
      /*.program_header_ordinal=*/UINT32_C(0x10203040),
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_AIE2P_TILE_PROGRAM_LOAD_RECORD_SIZE>
      storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_tile_program_load(
      &value, iree_make_byte_span(storage.data(), storage.size())));
  ExpectBytes(storage, {
                           0x04,
                           0x00,
                           0x00,
                           0x00,
                           0x0C,
                           0x00,
                           0x00,
                           0x00,
                           0x40,
                           0x30,
                           0x20,
                           0x10,
                       });

  iree_hal_amd_xdna_aie2p_program_record_t decoded;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_decode_program_record(
      iree_make_const_byte_span(storage.data(), storage.size()), &decoded));
  EXPECT_EQ(IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD,
            decoded.type);
  EXPECT_EQ(value.program_header_ordinal,
            decoded.value.tile_program_load.program_header_ordinal);
}

TEST(Aie2pProgramFormatTest, DmaTaskWaitRoundTrips) {
  const iree_hal_amd_xdna_aie2p_dma_task_wait_t value = {
      /*.column=*/2,
      /*.row=*/3,
      /*.direction=*/
      IREE_HAL_AMD_XDNA_AIE2P_DMA_DIRECTION_MEMORY_TO_STREAM,
      /*.dma_channel=*/1,
      /*.column_count=*/4,
      /*.row_count=*/5,
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_AIE2P_DMA_TASK_WAIT_RECORD_SIZE>
      storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_dma_task_wait(
      &value, iree_make_byte_span(storage.data(), storage.size())));
  ExpectBytes(storage, {
                           0x05,
                           0x00,
                           0x00,
                           0x00,
                           0x10,
                           0x00,
                           0x00,
                           0x00,
                           0x02,
                           0x03,
                           0x01,
                           0x01,
                           0x04,
                           0x05,
                           0x00,
                           0x00,
                       });

  iree_hal_amd_xdna_aie2p_program_record_t decoded;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_decode_program_record(
      iree_make_const_byte_span(storage.data(), storage.size()), &decoded));
  EXPECT_EQ(IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_DMA_TASK_WAIT, decoded.type);
  EXPECT_EQ(value.column, decoded.value.dma_task_wait.column);
  EXPECT_EQ(value.row, decoded.value.dma_task_wait.row);
  EXPECT_EQ(value.direction, decoded.value.dma_task_wait.direction);
  EXPECT_EQ(value.dma_channel, decoded.value.dma_task_wait.dma_channel);
  EXPECT_EQ(value.column_count, decoded.value.dma_task_wait.column_count);
  EXPECT_EQ(value.row_count, decoded.value.dma_task_wait.row_count);
}

TEST(Aie2pProgramFormatTest, RejectsMalformedRecordFraming) {
  std::array<uint8_t, IREE_HAL_AMD_XDNA_AIE2P_REGISTER_WRITE32_RECORD_SIZE>
      storage = {
          0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
          0x00, 0x10, 0x00, 0x00, 0xEF, 0xBE, 0xAD, 0xDE,
      };
  iree_hal_amd_xdna_aie2p_program_record_t decoded;

  storage[2] = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_xdna_aie2p_decode_program_record(
          iree_make_const_byte_span(storage.data(), storage.size()), &decoded));
  storage[2] = 0;
  storage[0] = 0xFF;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_amd_xdna_aie2p_decode_program_record(
          iree_make_const_byte_span(storage.data(), storage.size()), &decoded));
  storage[0] = 1;
  storage[4] = 12;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_xdna_aie2p_decode_program_record(
          iree_make_const_byte_span(storage.data(), storage.size()), &decoded));
}

TEST(Aie2pProgramFormatTest, RejectsMalformedBlockAndWaitRecords) {
  std::array<uint8_t, 20> block_storage = {
      0x03, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x10,
      0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x78, 0x56, 0x34, 0x12,
  };
  iree_hal_amd_xdna_aie2p_program_record_t decoded;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_xdna_aie2p_decode_program_record(
          iree_make_const_byte_span(block_storage.data(), block_storage.size()),
          &decoded));

  std::array<uint8_t, IREE_HAL_AMD_XDNA_AIE2P_DMA_TASK_WAIT_RECORD_SIZE>
      wait_storage = {
          0x05, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
          0x02, 0x03, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00,
      };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_xdna_aie2p_decode_program_record(
          iree_make_const_byte_span(wait_storage.data(), wait_storage.size()),
          &decoded));
}

TEST(Aie2pProgramFormatTest, RejectsInvalidConstruction) {
  std::array<uint8_t, IREE_HAL_AMD_XDNA_AIE2P_REGISTER_WRITE32_RECORD_SIZE>
      write_storage;
  const iree_hal_amd_xdna_aie2p_register_write32_t unaligned_write = {
      /*.address=*/3,
      /*.value=*/0,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amd_xdna_aie2p_encode_register_write32(
          &unaligned_write,
          iree_make_byte_span(write_storage.data(), write_storage.size())));

  iree_host_size_t byte_length = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_xdna_aie2p_measure_register_block_write32(0, &byte_length));
  EXPECT_EQ(0u, byte_length);

  std::array<uint8_t, IREE_HAL_AMD_XDNA_AIE2P_DMA_TASK_WAIT_RECORD_SIZE>
      wait_storage;
  const iree_hal_amd_xdna_aie2p_dma_task_wait_t invalid_wait = {
      /*.column=*/255,
      /*.row=*/0,
      /*.direction=*/
      IREE_HAL_AMD_XDNA_AIE2P_DMA_DIRECTION_STREAM_TO_MEMORY,
      /*.dma_channel=*/0,
      /*.column_count=*/2,
      /*.row_count=*/1,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amd_xdna_aie2p_encode_dma_task_wait(
          &invalid_wait,
          iree_make_byte_span(wait_storage.data(), wait_storage.size())));
}

}  // namespace
