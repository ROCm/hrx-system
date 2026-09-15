// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/target/npu4/bootstrap.h"

#include <cstddef>
#include <cstdint>

#include "gtest/gtest.h"

namespace {

uint32_t ReadU32(const uint8_t* bytes, size_t offset) {
  return uint32_t{bytes[offset]} | (uint32_t{bytes[offset + 1]} << 8) |
         (uint32_t{bytes[offset + 2]} << 16) |
         (uint32_t{bytes[offset + 3]} << 24);
}

TEST(Npu4BootstrapTest, ContainsOnlyCoreResetAdmissionEffects) {
  ASSERT_EQ(amdf_xdna_npu4_bootstrap.pdi_byte_length, 432u);
  const auto* bytes =
      static_cast<const uint8_t*>(amdf_xdna_npu4_bootstrap.pdi_bytes);
  constexpr size_t kCdoOffset = 336;
  EXPECT_EQ(ReadU32(bytes, kCdoOffset), 4u);
  EXPECT_EQ(ReadU32(bytes, kCdoOffset + 4), UINT32_C(0x004F4443));
  EXPECT_EQ(ReadU32(bytes, kCdoOffset + 8), UINT32_C(0x200));
  EXPECT_EQ(ReadU32(bytes, kCdoOffset + 12), 16u);
  uint32_t checksum = 0;
  for (size_t offset = kCdoOffset; offset < kCdoOffset + 20; offset += 4) {
    checksum += ReadU32(bytes, offset);
  }
  EXPECT_EQ(checksum, UINT32_MAX);
  // Four masked core-reset writes: no application DMA or stream-switch setup.
  for (uint32_t row = 2; row < 6; ++row) {
    const size_t offset = kCdoOffset + 20 + (row - 2) * 16;
    EXPECT_EQ(ReadU32(bytes, offset), UINT32_C(0x00030102));
    EXPECT_EQ(ReadU32(bytes, offset + 4), (row << 20) | UINT32_C(0x32000));
    EXPECT_EQ(ReadU32(bytes, offset + 8), 1u);
    EXPECT_EQ(ReadU32(bytes, offset + 12), 1u);
  }
  for (size_t offset = kCdoOffset + 20 + 64; offset < 432; ++offset) {
    EXPECT_EQ(bytes[offset], 0u);
  }
}

}  // namespace
