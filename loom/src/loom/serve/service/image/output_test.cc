// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/image/output.h"

#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(ImageOutputTest, ExactRgbaPixelsAndPngChecksums) {
  const uint8_t rgba[] = {0,  128, 255, 64, 255, 0, 32, 255,
                          17, 67,  201, 0,  1,   2, 3,  127};
  iree_byte_span_t png;
  IREE_ASSERT_OK(loom_serve_image_encode_rgba8_png(
      2, 2, iree_make_const_byte_span(rgba, sizeof(rgba)),
      iree_allocator_system(), &png));
  // Two rows with distinct color and alpha values, independently encoded with
  // zlib stored blocks and PNG CRCs. This also fixes the byte/row ordering.
  const uint8_t expected[] = {
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00,
      0x0d, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00,
      0x00, 0x02, 0x08, 0x06, 0x00, 0x00, 0x00, 0x72, 0xb6, 0x0d, 0x24,
      0x00, 0x00, 0x00, 0x1d, 0x49, 0x44, 0x41, 0x54, 0x78, 0x01, 0x01,
      0x12, 0x00, 0xed, 0xff, 0x00, 0x00, 0x80, 0xff, 0x40, 0xff, 0x00,
      0x20, 0xff, 0x00, 0x11, 0x43, 0xc9, 0x00, 0x01, 0x02, 0x03, 0x7f,
      0x3a, 0x6e, 0x05, 0x80, 0x89, 0x6e, 0x97, 0xfa, 0x00, 0x00, 0x00,
      0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
  };
  EXPECT_EQ(std::vector<uint8_t>(png.data, png.data + png.data_length),
            std::vector<uint8_t>(expected, expected + sizeof(expected)));
  iree_allocator_free(iree_allocator_system(), png.data);
}

TEST(ImageOutputTest, StoredBlockBoundariesAreIndependentOfRows) {
  for (auto shape : {std::pair<uint32_t, uint32_t>{64, 255},
                     {16384, 1},
                     {16383, 1},
                     {1, 13108},
                     {3, 2}}) {
    SCOPED_TRACE(::testing::PrintToString(shape));
    const auto [width, height] = shape;
    const size_t raw_length = (width * 4 + 1) * height;
    std::vector<uint8_t> rgba(size_t{width} * height * 4);
    iree_byte_span_t png;
    IREE_ASSERT_OK(loom_serve_image_encode_rgba8_png(
        width, height, iree_make_const_byte_span(rgba.data(), rgba.size()),
        iree_allocator_system(), &png));
    iree_host_size_t predicted_length = 0;
    IREE_ASSERT_OK(
        loom_serve_image_rgba8_png_size(width, height, &predicted_length));
    EXPECT_EQ(predicted_length, png.data_length);
    const uint8_t* cursor = png.data + 43;
    size_t decoded = 0;
    do {
      const bool final = *cursor++ == 1;
      const uint16_t length = iree_unaligned_load_le_u16(cursor);
      EXPECT_EQ(iree_unaligned_load_le_u16(cursor + 2), uint16_t(~length));
      cursor += 4;
      EXPECT_EQ(final, decoded + length == raw_length);
      EXPECT_NE(length, 0);
      for (size_t i = 0; i < length; ++i) {
        EXPECT_EQ(*cursor++, 0);
      }
      decoded += length;
    } while (decoded < raw_length);
    EXPECT_EQ(decoded, raw_length);
    const uint32_t adler = (uint32_t(raw_length % 65521) << 16) | 1;
    EXPECT_EQ(cursor[0], adler >> 24);
    EXPECT_EQ(cursor[1], (adler >> 16) & 255);
    EXPECT_EQ(cursor[2], 0);
    EXPECT_EQ(cursor[3], 1);
    EXPECT_EQ(size_t(cursor - png.data) + 20, png.data_length);
    iree_allocator_free(iree_allocator_system(), png.data);
  }
}

TEST(ImageOutputTest, RejectsUnencodableShapeWithoutAllocating) {
  iree_host_size_t length = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_image_rgba8_png_size(UINT32_MAX, 1, &length));
  EXPECT_EQ(length, 0u);
  if (IREE_HOST_SIZE_MAX > UINT32_MAX) {
    // RGBA storage fits a 64-bit host, but the PNG IDAT cannot represent it.
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        loom_serve_image_rgba8_png_size(32768, 32768, &length));
    EXPECT_EQ(length, 0u);
    // Framing arithmetic must reject enormous valid host extents too.
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        loom_serve_image_rgba8_png_size(INT32_MAX, INT32_MAX, &length));
    EXPECT_EQ(length, 0u);
  }
}

TEST(ImageOutputTest, RejectsInvalidShapeAndDeviceOutput) {
  uint8_t rgba[4] = {};
  iree_byte_span_t png;
  for (auto shape : {std::pair<uint32_t, uint32_t>{0, 1},
                     {1, 0},
                     {2, 1},
                     {UINT32_MAX, UINT32_MAX}}) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_image_encode_rgba8_png(
                              shape.first, shape.second,
                              iree_make_const_byte_span(rgba, sizeof(rgba)),
                              iree_allocator_system(), &png));
    EXPECT_EQ(png.data, nullptr);
  }
  uint8_t oversized[5] = {};
  for (size_t length : {size_t{3}, size_t{5}}) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_image_encode_rgba8_png(
            1, 1, iree_make_const_byte_span(oversized, length),
            iree_allocator_system(), &png));
    EXPECT_EQ(png.data, nullptr);
  }
}

}  // namespace
