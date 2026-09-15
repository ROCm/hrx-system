// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/aie2p/transaction.h"

#include <array>
#include <cstdint>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::StatusCode;

TEST(Aie2pTransactionTest, MatchesIndependentTransaction01Encoding) {
  constexpr iree_hal_amd_xdna_aie2p_transaction_target_t kTarget = {
      /*.device_generation=*/4,
      /*.row_count=*/6,
      /*.column_count=*/5,
      /*.memory_tile_row_count=*/1,
  };
  std::array<uint8_t, 112> storage = {};
  iree_hal_amd_xdna_aie2p_transaction_writer_t writer;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_transaction_writer_initialize(
      &kTarget, iree_make_byte_span(storage.data(), storage.size()), &writer));
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_transaction_writer_append_write32(
      &writer, 0x06400DEF, 0x42));
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_transaction_writer_append_mask_write32(
      &writer, 0x00012340, 0xAB, 0xFF));
  const iree_hal_amd_xdna_aie2p_dma_task_wait_t wait = {
      /*.column=*/1,
      /*.row=*/2,
      /*.direction=*/
      IREE_HAL_AMD_XDNA_AIE2P_DMA_DIRECTION_STREAM_TO_MEMORY,
      /*.dma_channel=*/3,
      /*.column_count=*/4,
      /*.row_count=*/5,
  };
  IREE_ASSERT_OK(
      iree_hal_amd_xdna_aie2p_transaction_writer_append_dma_task_wait(&writer,
                                                                      &wait));
  constexpr std::array<uint32_t, 3> kBlockWords = {
      UINT32_C(0xDEADBEEF),
      UINT32_C(0x12345678),
      UINT32_C(0x9ABCDEF0),
  };
  IREE_ASSERT_OK(
      iree_hal_amd_xdna_aie2p_transaction_writer_append_block_write32(
          &writer, 0x00021000, 6, 7,
          iree_make_const_byte_span(kBlockWords.data(), sizeof(kBlockWords))));
  iree_const_byte_span_t transaction = iree_const_byte_span_empty();
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_transaction_writer_finalize(
      &writer, &transaction));

  constexpr std::array<uint32_t, 28> kExpectedWords = {
      0x06040100, 0x00000105, 0x00000004, 0x00000070, 0x00000000, 0x00000000,
      0x06400DEF, 0x00000000, 0x00000042, 0x00000018, 0x00000003, 0x00000000,
      0x00012340, 0x00000000, 0x000000AB, 0x000000FF, 0x0000001C, 0x00000080,
      0x00000010, 0x00010200, 0x03040500, 0x00000001, 0x00000706, 0x00021000,
      0x0000001C, 0xDEADBEEF, 0x12345678, 0x9ABCDEF0,
  };
  ASSERT_EQ(transaction.data_length, sizeof(kExpectedWords));
  EXPECT_EQ(
      memcmp(transaction.data, kExpectedWords.data(), sizeof(kExpectedWords)),
      0);
}

TEST(Aie2pTransactionTest, PreservesWriterStateAfterRejectedAppend) {
  constexpr iree_hal_amd_xdna_aie2p_transaction_target_t kTarget = {
      /*.device_generation=*/4,
      /*.row_count=*/6,
      /*.column_count=*/1,
      /*.memory_tile_row_count=*/1,
  };
  std::array<uint8_t, 40> storage = {};
  iree_hal_amd_xdna_aie2p_transaction_writer_t writer;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_transaction_writer_initialize(
      &kTarget, iree_make_byte_span(storage.data(), storage.size()), &writer));
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_transaction_writer_append_write32(
      &writer, 0x1000, 1));

  IREE_EXPECT_STATUS_IS(
      StatusCode::kResourceExhausted,
      iree_hal_amd_xdna_aie2p_transaction_writer_append_write32(&writer, 0x1004,
                                                                2));
  EXPECT_EQ(writer.byte_offset, storage.size());
  EXPECT_EQ(writer.operation_count, 1u);

  iree_const_byte_span_t transaction = iree_const_byte_span_empty();
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_transaction_writer_finalize(
      &writer, &transaction));
  EXPECT_EQ(transaction.data_length, storage.size());
  EXPECT_EQ(iree_unaligned_load_le_u32(transaction.data + 8), 1u);
}

}  // namespace
