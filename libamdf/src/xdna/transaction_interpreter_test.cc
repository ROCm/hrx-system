// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/transaction_interpreter.h"

#include <array>
#include <cstring>

#include "gtest/gtest.h"

namespace {

TEST(XdnaTransactionInterpreterTest, EncodesCallingConvention) {
  amdf_xdna_transaction_interpreter_packet_t packet;
  amdf_xdna_transaction_interpreter_packet_build(0x04008000, 300, &packet);
  const std::array<uint8_t, 68> expected = {
      0x01, 0x00, 0x01, 0x30, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x04,
      0x00, 0x00, 0x00, 0x00, 0x4b, 0x00, 0x00, 0x00};
  EXPECT_EQ(std::memcmp(packet.bytes, expected.data(), expected.size()), 0);
}

TEST(XdnaTransactionInterpreterTest, ClearsUnusedBindingsAndReservedBytes) {
  amdf_xdna_transaction_interpreter_packet_t packet;
  std::memset(&packet, 0xa5, sizeof(packet));
  amdf_xdna_transaction_interpreter_packet_build(0x04010000, 20, &packet);
  for (size_t i = 12; i < 16; ++i) EXPECT_EQ(packet.bytes[i], 0);
  for (size_t i = 28; i < sizeof(packet); ++i) EXPECT_EQ(packet.bytes[i], 0);
}

}  // namespace
