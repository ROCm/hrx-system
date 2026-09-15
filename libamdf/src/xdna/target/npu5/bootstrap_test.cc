// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/target/npu5/bootstrap.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace {

uint32_t ReadU32(const uint8_t* bytes, size_t offset) {
  return static_cast<uint32_t>(bytes[offset]) |
         (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
         (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
         (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

bool IsStreamSwitchAddress(uint32_t address) {
  const uint32_t register_block = address & UINT32_C(0x00FFF000);
  return register_block == UINT32_C(0x0003F000) ||
         register_block == UINT32_C(0x001B0000) ||
         register_block == UINT32_C(0x0023F000);
}

TEST(Npu5BootstrapTest, PreservesInterpreterRoutesWithoutDataPlaneFanout) {
  constexpr size_t kPdiByteLength = 2000;
  constexpr size_t kCdoOffset = 336;
  constexpr size_t kCdoHeaderByteLength = 20;
  constexpr uint32_t kCdoWriteCommand = UINT32_C(0x0103);
  constexpr std::array<std::pair<uint32_t, uint32_t>, 15> kExpectedEffects = {{
      // Leave the application input DMA destination disabled.
      {UINT32_C(0x0003F034), UINT32_C(0x00000000)},
      {UINT32_C(0x0003F114), UINT32_C(0x80000000)},
      {UINT32_C(0x0003F010), UINT32_C(0x8000000F)},
      {UINT32_C(0x0003F13C), UINT32_C(0x80000000)},
      {UINT32_C(0x0003F008), UINT32_C(0xC0000045)},
      {UINT32_C(0x0003F100), UINT32_C(0xC0000000)},
      {UINT32_C(0x0003F200), UINT32_C(0x0F1F0135)},
      {UINT32_C(0x001B0030), UINT32_C(0x80000008)},
      {UINT32_C(0x001B0120), UINT32_C(0x80000000)},
      {UINT32_C(0x001B0020), UINT32_C(0x8000000E)},
      {UINT32_C(0x001B0138), UINT32_C(0x80000000)},
      {UINT32_C(0x0023F004), UINT32_C(0x80000006)},
      {UINT32_C(0x0023F118), UINT32_C(0x80000000)},
      // Leave the application output DMA destination disabled.
      {UINT32_C(0x0023F018), UINT32_C(0x00000000)},
      {UINT32_C(0x0023F104), UINT32_C(0x80000000)},
  }};

  ASSERT_EQ(amdf_xdna_npu5_bootstrap.pdi_byte_length, kPdiByteLength);
  const uint8_t* bytes =
      static_cast<const uint8_t*>(amdf_xdna_npu5_bootstrap.pdi_bytes);
  ASSERT_NE(bytes, nullptr);
  ASSERT_EQ(ReadU32(bytes, kCdoOffset), 4u);
  ASSERT_EQ(ReadU32(bytes, kCdoOffset + 4), UINT32_C(0x004F4443));
  ASSERT_EQ(ReadU32(bytes, kCdoOffset + 8), UINT32_C(0x00000200));

  const uint32_t command_word_count = ReadU32(bytes, kCdoOffset + 12);
  const size_t cdo_end =
      kCdoOffset + kCdoHeaderByteLength + command_word_count * sizeof(uint32_t);
  ASSERT_LE(cdo_end, kPdiByteLength);
  EXPECT_EQ(ReadU32(bytes, kCdoOffset) + ReadU32(bytes, kCdoOffset + 4) +
                ReadU32(bytes, kCdoOffset + 8) +
                ReadU32(bytes, kCdoOffset + 12) +
                ReadU32(bytes, kCdoOffset + 16),
            UINT32_MAX);

  std::vector<std::pair<uint32_t, uint32_t>> effects;
  size_t offset = kCdoOffset + kCdoHeaderByteLength;
  while (offset < cdo_end) {
    const uint32_t header = ReadU32(bytes, offset);
    offset += sizeof(uint32_t);
    uint32_t argument_count = (header >> 16) & UINT32_C(0xFF);
    if (argument_count == UINT32_C(0xFF)) {
      ASSERT_LE(offset + sizeof(uint32_t), cdo_end);
      argument_count = ReadU32(bytes, offset);
      offset += sizeof(uint32_t);
    }
    const size_t argument_byte_length =
        static_cast<size_t>(argument_count) * sizeof(uint32_t);
    ASSERT_LE(offset + argument_byte_length, cdo_end);
    if ((header & UINT32_C(0xFFFF)) == kCdoWriteCommand &&
        argument_count >= 2) {
      const uint32_t address = ReadU32(bytes, offset);
      if (IsStreamSwitchAddress(address)) {
        effects.emplace_back(address, ReadU32(bytes, offset + 4));
      }
    }
    offset += argument_byte_length;
  }

  EXPECT_EQ(offset, cdo_end);
  const std::vector<std::pair<uint32_t, uint32_t>> expected_effects(
      kExpectedEffects.begin(), kExpectedEffects.end());
  EXPECT_EQ(effects, expected_effects);
}

}  // namespace
