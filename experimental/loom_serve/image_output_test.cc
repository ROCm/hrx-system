// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/image_output.h"

#include <limits>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(ImageOutputTest, ExactRgbConversionAndPngChecksums) {
  uint8_t rgb[12];
  iree_unaligned_store_le_f32(rgb, -1.0f);
  iree_unaligned_store_le_f32(rgb + 4, 0.0f);
  iree_unaligned_store_le_f32(rgb + 8, 1.0f);
  iree_byte_span_t png;
  IREE_ASSERT_OK(loom_serve_image_encode_rgb_f32_png(
      1, 1, iree_make_const_byte_span(rgb, sizeof(rgb)), &png,
      iree_allocator_system()));
  // One RGB pixel (0,128,255), with a complete zlib stream and all PNG CRCs.
  const uint8_t expected[] = {
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
      0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
      0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xde, 0x00, 0x00, 0x00,
      0x0f, 0x49, 0x44, 0x41, 0x54, 0x78, 0x01, 0x01, 0x04, 0x00, 0xfb, 0xff,
      0x00, 0x00, 0x80, 0xff, 0x02, 0x03, 0x01, 0x80, 0x9d, 0x7f, 0x4c, 0xcd,
      0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};
  EXPECT_EQ(std::vector<uint8_t>(png.data, png.data + png.data_length),
            std::vector<uint8_t>(expected, expected + sizeof(expected)));
  iree_allocator_free(iree_allocator_system(), png.data);
}

TEST(ImageOutputTest, StoredBlockBoundariesAreIndependentOfRows) {
  for (auto shape : {std::pair<uint32_t, uint32_t>{28, 771},
                     {21845, 1},
                     {21844, 1},
                     {1, 16384},
                     {3, 2}}) {
    SCOPED_TRACE(::testing::PrintToString(shape));
    const auto [width, height] = shape;
    const size_t raw_length = (width * 3 + 1) * height;
    std::vector<uint8_t> rgb(size_t{width} * height * 12);
    for (size_t i = 0; i < rgb.size(); i += 4) {
      iree_unaligned_store_le_f32(rgb.data() + i, -1.0f);
    }
    iree_byte_span_t png;
    IREE_ASSERT_OK(loom_serve_image_encode_rgb_f32_png(
        width, height, iree_make_const_byte_span(rgb.data(), rgb.size()), &png,
        iree_allocator_system()));
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

TEST(ImageOutputTest, RejectsInvalidShapeAndDeviceOutput) {
  uint8_t rgb[12] = {};
  iree_byte_span_t png;
  for (auto shape : {std::pair<uint32_t, uint32_t>{0, 1},
                     {1, 0},
                     {2, 1},
                     {UINT32_MAX, UINT32_MAX}}) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_image_encode_rgb_f32_png(
                              shape.first, shape.second,
                              iree_make_const_byte_span(rgb, sizeof(rgb)), &png,
                              iree_allocator_system()));
    EXPECT_EQ(png.data, nullptr);
  }
  for (float value : {2.0f, -2.0f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
    iree_unaligned_store_le_f32(rgb, value);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                          loom_serve_image_encode_rgb_f32_png(
                              1, 1, iree_make_const_byte_span(rgb, sizeof(rgb)),
                              &png, iree_allocator_system()));
    EXPECT_EQ(png.data, nullptr);
  }
}

}  // namespace
