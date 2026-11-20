// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/control_packet.h"

#include <vector>

#include "iree/testing/gtest.h"

namespace {

TEST(ControlPacketTest, WritesEachWordWithinItsAddressBlock) {
  uint8_t source[40];
  for (size_t i = 0; i < sizeof(source); ++i) {
    source[i] = static_cast<uint8_t>(i + 1);
  }
  const uint32_t header = loom_aie2p_control_packet_parity(5);
  for (uint32_t offset = 0; offset < 16; offset += 4) {
    const uint32_t address = 0x20000 + offset;
    for (size_t length = 0; length < sizeof(source); ++length) {
      SCOPED_TRACE(::testing::Message()
                   << "offset=" << offset << ", length=" << length);
      const size_t word_count =
          loom_aie2p_control_packet_write_word_count(address, length);
      std::vector<uint32_t> words(word_count + 1, UINT32_MAX);
      EXPECT_EQ(
          loom_aie2p_control_packet_store_writes(
              header, address, iree_make_const_byte_span(source + 1, length),
              words.data()),
          words.data() + word_count);
      EXPECT_EQ(words.back(), UINT32_MAX);
      size_t written = 0;
      for (size_t i = 0; i < word_count;) {
        ASSERT_GE(word_count - i, 3u);
        EXPECT_EQ(words[i++], header);
        const uint32_t command = words[i++];
        EXPECT_EQ(command & 0xfffffu, address + written);
        const uint32_t count = ((command >> 20) & 3u) + 1;
        EXPECT_LE(((address + written) & 15u) + count * 4, 16u);
        ASSERT_LE(count, word_count - i);
        for (uint32_t j = 0; j < count; ++j) {
          const uint32_t word = words[i++];
          for (uint32_t byte = 0; byte < 4; ++byte, ++written) {
            EXPECT_EQ((word >> (8 * byte)) & 255u,
                      written < length ? source[written + 1] : 0u);
          }
        }
      }
      EXPECT_EQ(written, (length + 3u) / 4u * 4u);
    }
  }
}

}  // namespace
