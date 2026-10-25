// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/jpeg.h"

#include <array>
#include <cstdint>

#include "gtest/gtest.h"

namespace {

constexpr uint64_t kCompressedAddress = UINT64_C(0x123456789abc0000);
constexpr uint64_t kOutputAddress = UINT64_C(0x23456789abc00000);
constexpr uint32_t kCompressedByteLength = 2048;

amdf_media_drm_frame_t MakeOutput() {
  amdf_media_drm_frame_t output = {};
  output.addresses[0] = kOutputAddress;
  output.addresses[1] = kOutputAddress + 0x18000;
  output.row_pitches[0] = 512;
  output.row_pitches[1] = 512;
  return output;
}

template <size_t N>
void ExpectWords(const uint32_t* actual,
                 const std::array<uint32_t, N>& expected) {
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(actual[i], expected[i]);
  }
}

TEST(JpegPacketTest, SelectsOnlyImplementedProtocolFamilies) {
  amdf_media_drm_command_protocol_t protocol =
      AMDF_MEDIA_DRM_COMMAND_PROTOCOL_VCN4;
  EXPECT_FALSE(amdf_media_drm_jpeg_protocol(0x010000, &protocol));
  EXPECT_EQ(protocol, AMDF_MEDIA_DRM_COMMAND_PROTOCOL_VCN4);
  EXPECT_TRUE(amdf_media_drm_jpeg_protocol(0x020000, &protocol));
  EXPECT_EQ(protocol, AMDF_MEDIA_DRM_COMMAND_PROTOCOL_JPEG_INDEXED);
  EXPECT_TRUE(amdf_media_drm_jpeg_protocol(0x040002, &protocol));
  EXPECT_EQ(protocol, AMDF_MEDIA_DRM_COMMAND_PROTOCOL_JPEG_INDEXED);
  EXPECT_TRUE(amdf_media_drm_jpeg_protocol(0x040003, &protocol));
  EXPECT_EQ(protocol, AMDF_MEDIA_DRM_COMMAND_PROTOCOL_JPEG_DIRECT);
  EXPECT_TRUE(amdf_media_drm_jpeg_protocol(0x050000, &protocol));
  EXPECT_EQ(protocol, AMDF_MEDIA_DRM_COMMAND_PROTOCOL_JPEG_DIRECT);
  EXPECT_TRUE(amdf_media_drm_jpeg_protocol(0x050300, &protocol));
  EXPECT_EQ(protocol, AMDF_MEDIA_DRM_COMMAND_PROTOCOL_JPEG_DIRECT);
  EXPECT_FALSE(amdf_media_drm_jpeg_protocol(0x060000, &protocol));
  EXPECT_EQ(protocol, AMDF_MEDIA_DRM_COMMAND_PROTOCOL_JPEG_DIRECT);
}

TEST(JpegPacketTest, IndexedProtocolMatchesCompleteNativePacket) {
  constexpr std::array<uint32_t, 80> kExpected = {
      0x0000402f, 0x00000001, 0x0000408e, 0x01400200, 0x0000408f, 0x00010000,
      0x3300402f, 0x00010000, 0x0000402f, 0x00000000, 0x0000408f, 0x00000000,
      0x3300402f, 0x00010000, 0x000040e1, 0x12345678, 0x000040e0, 0x9abc0000,
      0x00004001, 0x00000000, 0x00004004, 0xfffffff0, 0x00004002, 0x00000200,
      0x0000401f, 0x00000020, 0x00004020, 0x00000020, 0x00004027, 0x00000000,
      0x00004024, 0x00000000, 0x00004025, 0x00000000, 0x000040e3, 0x23456789,
      0x000040e2, 0xabc00000, 0x0000402c, 0x00000000, 0x0000402d, 0x00000000,
      0x0000402c, 0x00000001, 0x0000402d, 0x00018000, 0x0000402c, 0x00000002,
      0x0000402d, 0x00000000, 0x0000400f, 0x00000000, 0x0000401e, 0x00000000,
      0x0000401c, 0x000014c7, 0x0000400a, 0xfffffffe, 0x00004000, 0x00000006,
      0x0000408f, 0x00000200, 0x0000408e, 0x01400200, 0x33004003, 0xffffffff,
      0x0000408f, 0xffffffff, 0x3300401d, 0x00000001, 0x00004000, 0x00000004,
      0x60000000, 0x00000000, 0x60000000, 0x00000000, 0x60000000, 0x00000000,
      0x60000000, 0x00000000,
  };
  uint32_t words[128] = {};
  const amdf_media_drm_frame_t output = MakeOutput();
  const uint32_t byte_length = amdf_media_drm_jpeg_emit(
      AMDF_MEDIA_DRM_COMMAND_PROTOCOL_JPEG_INDEXED, words, kCompressedAddress,
      kCompressedByteLength, &output);
  ASSERT_EQ(byte_length, sizeof(kExpected));
  ExpectWords(words, kExpected);
}

TEST(JpegPacketTest, DirectProtocolMatchesCompleteNativePacket) {
  constexpr std::array<uint32_t, 80> kExpected = {
      0x00004051, 0x00000001, 0x0000408e, 0x01400200, 0x0000408f, 0x00010000,
      0x33004051, 0x00010000, 0x00004051, 0x00000000, 0x0000408f, 0x00000000,
      0x33004051, 0x00010000, 0x000040b3, 0x12345678, 0x000040b2, 0x9abc0000,
      0x00004001, 0x00000000, 0x00004004, 0xfffffff0, 0x00004002, 0x00000200,
      0x00004043, 0x00000020, 0x00004044, 0x00000020, 0x0000404b, 0x00000000,
      0x00004048, 0x00000000, 0x00004049, 0x00000000, 0x000040b5, 0x23456789,
      0x000040b4, 0xabc00000, 0x000041c0, 0x00000000, 0x000041c1, 0x00018000,
      0x000041c2, 0x00000000, 0x0000401b, 0x00000000, 0x0000401c, 0x00010001,
      0x00004052, 0x0000ff21, 0x0000400e, 0x00000000, 0x00004042, 0x00000000,
      0x00004040, 0x000014c7, 0x0000400a, 0xfffffffe, 0x00004000, 0x00000006,
      0x0000408f, 0x00000200, 0x0000408e, 0x01400200, 0x33004003, 0xffffffff,
      0x0000408f, 0xffffffff, 0x33004041, 0x00000001, 0x00004000, 0x00000004,
      0x60000000, 0x00000000, 0x60000000, 0x00000000, 0x60000000, 0x00000000,
      0x60000000, 0x00000000,
  };
  uint32_t words[128] = {};
  const amdf_media_drm_frame_t output = MakeOutput();
  const uint32_t byte_length = amdf_media_drm_jpeg_emit(
      AMDF_MEDIA_DRM_COMMAND_PROTOCOL_JPEG_DIRECT, words, kCompressedAddress,
      kCompressedByteLength, &output);
  ASSERT_EQ(byte_length, sizeof(kExpected));
  ExpectWords(words, kExpected);
}

}  // namespace
