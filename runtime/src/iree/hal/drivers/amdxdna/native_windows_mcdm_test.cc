// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <vector>

#include "iree/hal/drivers/amdxdna/native_windows_mcdm_internal.h"
#include "iree/testing/gtest.h"

namespace {

constexpr size_t kHeaderSize = 4 * sizeof(uint32_t);
constexpr size_t kBlockWriteSize = 12 * sizeof(uint32_t);
constexpr size_t kDdrPatchSize = 12 * sizeof(uint32_t);
constexpr size_t kWrite32Size = 6 * sizeof(uint32_t);

void WriteU32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
  bytes[offset + 0] = static_cast<uint8_t>(value);
  bytes[offset + 1] = static_cast<uint8_t>(value >> 8);
  bytes[offset + 2] = static_cast<uint8_t>(value >> 16);
  bytes[offset + 3] = static_cast<uint8_t>(value >> 24);
}

size_t AppendBlockWrite(std::vector<uint8_t>& bytes, uint32_t location) {
  const size_t offset = bytes.size();
  bytes.resize(offset + kBlockWriteSize);
  bytes[offset] = 1;
  WriteU32(bytes, offset + 8, location);
  WriteU32(bytes, offset + 12, static_cast<uint32_t>(kBlockWriteSize));
  return offset;
}

size_t AppendDdrPatch(std::vector<uint8_t>& bytes, uint32_t location) {
  const size_t offset = bytes.size();
  bytes.resize(offset + kDdrPatchSize);
  bytes[offset] = 129;
  WriteU32(bytes, offset + 4, static_cast<uint32_t>(kDdrPatchSize));
  WriteU32(bytes, offset + 24, location);
  return offset;
}

size_t AppendQueuePush(std::vector<uint8_t>& bytes, uint32_t location,
                       uint32_t bd_id) {
  const size_t offset = bytes.size();
  bytes.resize(offset + kWrite32Size);
  bytes[offset] = 0;
  WriteU32(bytes, offset + 8, location);
  WriteU32(bytes, offset + 16, bd_id);
  WriteU32(bytes, offset + 20, static_cast<uint32_t>(kWrite32Size));
  return offset;
}

TEST(NativeWindowsMcdmTxnTest, ResolvesReusedBdBeforeEachQueuePush) {
  constexpr uint32_t kColumn = 1;
  constexpr uint32_t kRow = 2;
  constexpr uint32_t kBdId = 3;
  constexpr uint32_t kTile = (kColumn << 25) | (kRow << 20);
  constexpr uint32_t kDmaLocation = kTile | (kBdId << 5);
  constexpr uint32_t kDdrLocation = kDmaLocation | 4;
  constexpr uint32_t kQueueLocation = kTile | 0x1D200;
  constexpr uint32_t kKey = (kColumn << 16) | (kRow << 8) | kBdId;

  std::vector<uint8_t> bytes(kHeaderSize);
  const size_t first_dma = AppendBlockWrite(bytes, kDmaLocation);
  const size_t first_ddr = AppendDdrPatch(bytes, kDdrLocation);
  const size_t first_queue = AppendQueuePush(bytes, kQueueLocation, kBdId);
  const size_t second_dma = AppendBlockWrite(bytes, kDmaLocation);
  const size_t second_ddr = AppendDdrPatch(bytes, kDdrLocation);
  const size_t second_queue = AppendQueuePush(bytes, kQueueLocation, kBdId);

  const uint8_t* dma = nullptr;
  const uint8_t* ddr = nullptr;
  ASSERT_TRUE(
      iree_hal_amdxdna_native_windows_find_partial_elf_bd_ops(
          bytes.data(), bytes.size(), /*op_count=*/6, first_queue, kKey, &dma,
          &ddr));
  EXPECT_EQ(dma, bytes.data() + first_dma);
  EXPECT_EQ(ddr, bytes.data() + first_ddr);

  ASSERT_TRUE(
      iree_hal_amdxdna_native_windows_find_partial_elf_bd_ops(
          bytes.data(), bytes.size(), /*op_count=*/6, second_queue, kKey, &dma,
          &ddr));
  EXPECT_EQ(dma, bytes.data() + second_dma);
  EXPECT_EQ(ddr, bytes.data() + second_ddr);
}

}  // namespace
