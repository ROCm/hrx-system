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

TEST(NativeWindowsMcdmTxnTest, RejectsMalformedOperationSizes) {
  std::vector<uint8_t> bytes(kHeaderSize);
  const size_t malformed = bytes.size();
  bytes.resize(malformed + 4 * sizeof(uint32_t));
  bytes[malformed] = 1;
  WriteU32(bytes, malformed + 12, 0);

  const uint8_t* dma = reinterpret_cast<const uint8_t*>(1);
  const uint8_t* ddr = reinterpret_cast<const uint8_t*>(1);
  EXPECT_FALSE(iree_hal_amdxdna_native_windows_find_partial_elf_bd_ops(
      bytes.data(), bytes.size(), /*op_count=*/1, bytes.size(), /*key=*/0,
      &dma, &ddr));
  EXPECT_EQ(dma, nullptr);
  EXPECT_EQ(ddr, nullptr);

  WriteU32(bytes, malformed + 12, 64);
  EXPECT_FALSE(iree_hal_amdxdna_native_windows_find_partial_elf_bd_ops(
      bytes.data(), bytes.size(), /*op_count=*/1, bytes.size(), /*key=*/0,
      &dma, &ddr));
}

TEST(NativeWindowsMcdmTxnTest, IgnoresDescriptorsAfterQueuePush) {
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
  const size_t queue = AppendQueuePush(bytes, kQueueLocation, kBdId);
  AppendBlockWrite(bytes, kDmaLocation);
  AppendDdrPatch(bytes, kDdrLocation);

  const uint8_t* dma = nullptr;
  const uint8_t* ddr = nullptr;
  ASSERT_TRUE(iree_hal_amdxdna_native_windows_find_partial_elf_bd_ops(
      bytes.data(), bytes.size(), /*op_count=*/5, queue, kKey, &dma, &ddr));
  EXPECT_EQ(dma, bytes.data() + first_dma);
  EXPECT_EQ(ddr, bytes.data() + first_ddr);
}

TEST(NativeWindowsMcdmTxnTest, ComputesMultidimensionalDmaSpan) {
  std::vector<uint8_t> dma(kBlockWriteSize);
  WriteU32(dma, 16, 24);
  WriteU32(dma, 28, (2u << 20) | 3u);
  WriteU32(dma, 32, (3u << 20) | 7u);
  WriteU32(dma, 36, 15u);
  WriteU32(dma, 40, (2u << 20) | 31u);

  // dim2_size=24/(2*3)=4. Strided span is
  // 1 + 1*4 + 2*8 + 3*16 = 69, then two iterator strides of 32.
  EXPECT_EQ(iree_hal_amdxdna_native_windows_partial_elf_dma_span_words(
                dma.data()),
            133u);
}

TEST(NativeWindowsMcdmTxnTest, HandlesMaximumEncodedDmaSpan) {
  std::vector<uint8_t> dma(kBlockWriteSize);
  WriteU32(dma, 16, UINT32_MAX);
  WriteU32(dma, 28, (1023u << 20) | 0xFFFFFu);
  WriteU32(dma, 32, (1023u << 20) | 0xFFFFFu);
  WriteU32(dma, 36, 0xFFFFFu);
  WriteU32(dma, 40, (1023u << 20) | 0xFFFFFu);

  EXPECT_EQ(iree_hal_amdxdna_native_windows_partial_elf_dma_span_words(
                dma.data()),
            7518289921ull);
}

TEST(NativeWindowsMcdmBufferRangeTest,
     CoalescesPerBufferAndPreservesDisjointRanges) {
  void* buffer_a = reinterpret_cast<void*>(0x1000);
  void* buffer_b = reinterpret_cast<void*>(0x2000);
  iree_hal_amdxdna_native_windows_buffer_range_t ranges[] = {
      {buffer_a, 64, 64}, {buffer_b, 0, 16}, {buffer_a, 0, 32},
      {buffer_a, 32, 32}, {nullptr, 0, 64},  {buffer_a, 256, 0},
      {buffer_a, 256, 16}};

  const size_t count =
      iree_hal_amdxdna_native_windows_coalesce_buffer_ranges(
          ranges, std::size(ranges));

  ASSERT_EQ(count, 3u);
  EXPECT_EQ(ranges[0].buffer, buffer_a);
  EXPECT_EQ(ranges[0].offset, 0u);
  EXPECT_EQ(ranges[0].length, 128u);
  EXPECT_EQ(ranges[1].buffer, buffer_a);
  EXPECT_EQ(ranges[1].offset, 256u);
  EXPECT_EQ(ranges[1].length, 16u);
  EXPECT_EQ(ranges[2].buffer, buffer_b);
  EXPECT_EQ(ranges[2].offset, 0u);
  EXPECT_EQ(ranges[2].length, 16u);
}

TEST(NativeWindowsMcdmBufferRangeTest, SaturatesOverflowingRangeEnd) {
  void* buffer = reinterpret_cast<void*>(0x1000);
  iree_hal_amdxdna_native_windows_buffer_range_t ranges[] = {
      {buffer, UINT64_MAX - 7, 16}, {buffer, UINT64_MAX - 3, 4}};

  const size_t count =
      iree_hal_amdxdna_native_windows_coalesce_buffer_ranges(
          ranges, std::size(ranges));

  ASSERT_EQ(count, 1u);
  EXPECT_EQ(ranges[0].offset, UINT64_MAX - 7);
  EXPECT_EQ(ranges[0].length, 7u);
}

TEST(NativeWindowsMcdmPacketTest, CalculatesInitializedPacketBytes) {
  size_t packet_bytes = 0;
  EXPECT_TRUE(iree_hal_amdxdna_native_windows_calculate_ert_packet_bytes(
      /*payload_dword_count=*/5, /*allocation_size=*/4096, &packet_bytes));
  EXPECT_EQ(packet_bytes, 24u);

  EXPECT_TRUE(iree_hal_amdxdna_native_windows_calculate_ert_packet_bytes(
      /*payload_dword_count=*/1023, /*allocation_size=*/4096, &packet_bytes));
  EXPECT_EQ(packet_bytes, 4096u);
}

TEST(NativeWindowsMcdmPacketTest, RejectsPacketsOutsideAllocation) {
  size_t packet_bytes = 0;
  EXPECT_FALSE(iree_hal_amdxdna_native_windows_calculate_ert_packet_bytes(
      /*payload_dword_count=*/0, /*allocation_size=*/0, &packet_bytes));
  EXPECT_FALSE(iree_hal_amdxdna_native_windows_calculate_ert_packet_bytes(
      /*payload_dword_count=*/1024, /*allocation_size=*/4096, &packet_bytes));
  EXPECT_FALSE(iree_hal_amdxdna_native_windows_calculate_ert_packet_bytes(
      /*payload_dword_count=*/0, /*allocation_size=*/4096, nullptr));
}

TEST(NativeWindowsMcdmBufferTest, DefersOnlyContextOwnedCommandStorage) {
  EXPECT_FALSE(iree_hal_amdxdna_native_windows_buffer_requires_context(
      IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_HOST_ONLY));
  EXPECT_TRUE(iree_hal_amdxdna_native_windows_buffer_requires_context(
      IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_CACHEABLE));
  EXPECT_TRUE(iree_hal_amdxdna_native_windows_buffer_requires_context(
      IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION));
}

TEST(NativeWindowsMcdmCompletionSlotTest, ReservesAndReleasesAtomically) {
  uint8_t slots[] = {0, 1, 0, 0};
  uint32_t offsets[3] = {};
  size_t next_slot = 0;
  ASSERT_TRUE(iree_hal_amdxdna_native_windows_reserve_completion_slots(
      slots, std::size(slots), std::size(offsets), /*start_slot=*/0, offsets,
      &next_slot));
  EXPECT_EQ(offsets[0], 8u);
  EXPECT_EQ(offsets[1], 24u);
  EXPECT_EQ(offsets[2], 32u);
  EXPECT_EQ(next_slot, 0u);
  EXPECT_EQ(std::vector<uint8_t>(std::begin(slots), std::end(slots)),
            (std::vector<uint8_t>{1, 1, 1, 1}));

  uint32_t unavailable_offset = 0;
  EXPECT_FALSE(iree_hal_amdxdna_native_windows_reserve_completion_slots(
      slots, std::size(slots), 1, /*start_slot=*/next_slot,
      &unavailable_offset, &next_slot));
  EXPECT_TRUE(iree_hal_amdxdna_native_windows_release_completion_slots(
      slots, std::size(slots), std::size(offsets), offsets));
  EXPECT_EQ(std::vector<uint8_t>(std::begin(slots), std::end(slots)),
            (std::vector<uint8_t>{0, 1, 0, 0}));
}

TEST(NativeWindowsMcdmCompletionSlotTest, RotatesAfterReleasedSlots) {
  uint8_t slots[] = {0, 0, 0};
  uint32_t offset = 0;
  size_t next_slot = 0;
  ASSERT_TRUE(iree_hal_amdxdna_native_windows_reserve_completion_slots(
      slots, std::size(slots), 1, next_slot, &offset, &next_slot));
  EXPECT_EQ(offset, 8u);
  EXPECT_EQ(next_slot, 1u);
  ASSERT_TRUE(iree_hal_amdxdna_native_windows_release_completion_slots(
      slots, std::size(slots), 1, &offset));

  ASSERT_TRUE(iree_hal_amdxdna_native_windows_reserve_completion_slots(
      slots, std::size(slots), 1, next_slot, &offset, &next_slot));
  EXPECT_EQ(offset, 16u);
  EXPECT_EQ(next_slot, 2u);
}

TEST(NativeWindowsMcdmCompletionSlotTest,
     RejectsInvalidReleaseWithoutPartialMutation) {
  uint8_t slots[] = {1, 1, 0};
  const uint32_t duplicate_offsets[] = {8, 8};
  EXPECT_FALSE(iree_hal_amdxdna_native_windows_release_completion_slots(
      slots, std::size(slots), std::size(duplicate_offsets),
      duplicate_offsets));
  EXPECT_EQ(std::vector<uint8_t>(std::begin(slots), std::end(slots)),
            (std::vector<uint8_t>{1, 1, 0}));

  const uint32_t invalid_offsets[] = {8, 32};
  EXPECT_FALSE(iree_hal_amdxdna_native_windows_release_completion_slots(
      slots, std::size(slots), std::size(invalid_offsets), invalid_offsets));
  EXPECT_EQ(std::vector<uint8_t>(std::begin(slots), std::end(slots)),
            (std::vector<uint8_t>{1, 1, 0}));
}

TEST(NativeWindowsMcdmCompletionSlotTest, DetectsUnownedReservedSlots) {
  const uint8_t free_slots[] = {0, 0, 0};
  EXPECT_TRUE(iree_hal_amdxdna_native_windows_completion_slots_are_free(
      free_slots, std::size(free_slots)));

  const uint8_t leaked_slot[] = {0, 1, 0};
  EXPECT_FALSE(iree_hal_amdxdna_native_windows_completion_slots_are_free(
      leaked_slot, std::size(leaked_slot)));
  EXPECT_TRUE(iree_hal_amdxdna_native_windows_completion_slots_are_free(
      nullptr, 0));
  EXPECT_FALSE(iree_hal_amdxdna_native_windows_completion_slots_are_free(
      nullptr, 1));
}

}  // namespace
