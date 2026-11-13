// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <vector>

#include "iree/base/internal/base64.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/media/image.h"

namespace {

class PamTest : public ::testing::Test {
 protected:
  void TearDown() override {
    loom_serve_media_image_input_deinitialize(&input_);
  }

  iree_status_t Open(const std::string& bytes) {
    loom_serve_media_image_input_deinitialize(&input_);
    std::string encoded(iree_base64_encoded_size(bytes.size()), '\0');
    size_t written = 0;
    IREE_RETURN_IF_ERROR(iree_base64_encode(
        iree_make_const_byte_span(bytes.data(), bytes.size()),
        iree_make_mutable_string_view(encoded.data(), encoded.size()),
        &written));
    const std::string url =
        "data:image/x-portable-arbitrarymap;base64," + encoded;
    // The exact payload bound also exercises the longer PAM data-URL prefix.
    const loom_serve_media_image_limits_t limits = {
        .compressed_bytes = bytes.size(),
        .width = 64,
        .height = 64,
        .pixels = 4096,
    };
    return loom_serve_media_image_input_initialize(
        loom_serve_media_image_source_from_data_url(
            iree_make_string_view(url.data(), url.size())),
        &limits, iree_allocator_system(), &input_);
  }

  iree_status_t Decode(std::vector<uint8_t>& pixels) {
    const loom_serve_media_image_input_t* input = &input_;
    const auto destination = iree_make_byte_span(pixels.data(), pixels.size());
    iree_status_t result;
    IREE_RETURN_IF_ERROR(loom_serve_media_image_decode_batch(
        1, &input, &destination, iree_allocator_system(), &result));
    return result;
  }

  static std::string Header(const char* tuple, uint32_t depth,
                            uint32_t maximum) {
    return "P7\nWIDTH 2\nHEIGHT 1\nDEPTH " + std::to_string(depth) +
           "\nMAXVAL " + std::to_string(maximum) + "\nTUPLTYPE " + tuple +
           "\nENDHDR\n";
  }

  // Owned immutable input, released after each case and during teardown.
  loom_serve_media_image_input_t input_ = {};
};

TEST_F(PamTest, AllVisualTuplesPreserveColorAndStraightAlpha) {
  struct loom_serve_test_pam_case_t {
    // Standard visual tuple name.
    const char* tuple;
    // Encoded samples per pixel.
    uint32_t depth;
    // Largest sample value in the input representation.
    uint32_t maximum;
    // Two pixels, including alpha when declared by the tuple.
    std::vector<uint16_t> samples;
    // Exact normalized RGB8 or RGBA8 output.
    std::vector<uint8_t> expected;
  };
  const loom_serve_test_pam_case_t cases[] = {
      {"BLACKANDWHITE", 1, 1, {0, 1}, {0, 0, 0, 255, 255, 255}},
      {"BLACKANDWHITE_ALPHA",
       2,
       1,
       {1, 0, 0, 1},
       {255, 255, 255, 0, 0, 0, 0, 255}},
      {"GRAYSCALE", 1, 15, {7, 15}, {119, 119, 119, 255, 255, 255}},
      {"GRAYSCALE_ALPHA",
       2,
       256,
       {128, 0, 256, 128},
       {128, 128, 128, 0, 255, 255, 255, 128}},
      {"RGB", 3, 255, {10, 35, 255, 0, 127, 1}, {10, 35, 255, 0, 127, 1}},
      {"RGB_ALPHA",
       4,
       255,
       {10, 20, 30, 0, 40, 50, 60, 128},
       {10, 20, 30, 0, 40, 50, 60, 128}},
      {"RGB",
       3,
       65535,
       {0, 32768, 65535, 257, 514, 771},
       {0, 128, 255, 1, 2, 3}},
      {"RGB_ALPHA",
       4,
       1000,
       {0, 500, 1000, 0, 1000, 0, 500, 500},
       {0, 128, 255, 0, 255, 0, 128, 128}},
  };
  for (const auto& item : cases) {
    SCOPED_TRACE(item.tuple);
    SCOPED_TRACE(item.maximum);
    std::string bytes = Header(item.tuple, item.depth, item.maximum);
    for (uint16_t sample : item.samples) {
      if (item.maximum > 255) {
        bytes.push_back(char(sample >> 8));
      }
      bytes.push_back(char(sample & 255));
    }
    IREE_ASSERT_OK(Open(bytes));
    EXPECT_EQ(input_.info.format, LOOM_SERVE_MEDIA_IMAGE_FORMAT_PAM);
    EXPECT_EQ(input_.info.width, 2u);
    EXPECT_EQ(input_.info.height, 1u);
    EXPECT_EQ(input_.info.display_width, 2u);
    EXPECT_EQ(input_.info.display_height, 1u);
    EXPECT_EQ(input_.info.orientation, 1u);
    EXPECT_EQ(input_.info.pixel_length, item.expected.size());
    std::vector<uint8_t> pixels(input_.info.pixel_length, 0xa5);
    IREE_ASSERT_OK(Decode(pixels));
    EXPECT_EQ(pixels, item.expected);
  }
}

TEST_F(PamTest, HeaderOrderCommentsAndWhitespaceDoNotConsumeRaster) {
  const std::string bytes =
      "P7\n# RGB bytes begin with whitespace and a comment marker.\n\n"
      "TUPLTYPE RGB\n MAXVAL\t255 \r\nDEPTH 3\nHEIGHT 1\nWIDTH 2\nENDHDR\n"
      "\n# \t\r\n";
  IREE_ASSERT_OK(Open(bytes));
  std::vector<uint8_t> pixels(6);
  IREE_ASSERT_OK(Decode(pixels));
  EXPECT_EQ(pixels, (std::vector<uint8_t>{10, 35, 32, 9, 13, 10}));
}

TEST_F(PamTest, RejectsMalformedAndNonvisualHeadersBeforeRealization) {
  const std::string valid = Header("RGB", 3, 255);
  std::vector<std::string> malformed = {
      "P6\n2 1\n255\n",
      "P7\n",
      "P7\nENDHDR\n",
      Header("RGB", 4, 255),
      Header("RGB_ALPHA", 3, 255),
      Header("BLACKANDWHITE", 1, 2),
      Header("GRAYSCALE", 2, 255),
      Header("XYZ", 3, 255),
      Header("RGB_ALPHA_ALPHA", 4, 255),
      Header("RGB", 3, 0),
      Header("RGB", 3, 65536),
      valid.substr(0, valid.size() - 1),
  };
  for (const char* field : {"WIDTH 2", "HEIGHT 1", "DEPTH 3", "MAXVAL 255"}) {
    std::string repeated = valid;
    repeated.insert(3, std::string(field) + "\n");
    malformed.push_back(repeated);
  }
  for (const char* number : {"", "0", "-1", "+1", "2junk", "0x2", "2 1",
                             "4294967296", "9999999999999999999999999"}) {
    std::string bad_number = valid;
    bad_number.replace(bad_number.find("WIDTH 2"), 7,
                       std::string("WIDTH ") + number);
    malformed.push_back(bad_number);
  }
  for (const char* field : {"UNKNOWN 1\n", "TUPLTYPE RGB\n", "TUPLTYPE\n"}) {
    std::string bad_field = valid;
    bad_field.insert(3, field);
    malformed.push_back(bad_field);
  }
  for (const auto& bytes : malformed) {
    SCOPED_TRACE(bytes);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Open(bytes));
    EXPECT_EQ(input_.allocation, nullptr);
    EXPECT_EQ(input_.info.width, 0u);
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Open(""));
}

TEST_F(PamTest, AppliesFacadeGeometryBoundsBeforeAllocatingPixels) {
  for (const char* width : {"65", "4294967295"}) {
    std::string bytes = Header("RGB", 3, 255);
    bytes.replace(bytes.find("WIDTH 2"), 7, std::string("WIDTH ") + width);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Open(bytes));
    EXPECT_EQ(input_.allocation, nullptr);
  }
}

TEST_F(PamTest, RejectsEveryTruncationAtItsOwningBoundary) {
  const std::string header = Header("RGB_ALPHA", 4, 65535);
  const std::string bytes = header + std::string(16, '\x80');
  for (size_t length = 1; length < bytes.size(); ++length) {
    SCOPED_TRACE(length);
    if (length < header.size()) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                            Open(bytes.substr(0, length)));
      EXPECT_EQ(input_.allocation, nullptr);
    } else {
      IREE_ASSERT_OK(Open(bytes.substr(0, length)));
      std::vector<uint8_t> output(input_.info.pixel_length);
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Decode(output));
    }
  }
}

TEST_F(PamTest, RejectsInvalidRasterDuringRealization) {
  const std::string header = Header("RGB", 3, 255);
  const std::string pixels(6, '\x20');
  const std::string malformed[] = {
      header,
      header + pixels.substr(0, 5),
      header + pixels + "x",
      header + pixels + header + pixels,
      Header("GRAYSCALE", 1, 15) + std::string("\x00\x10", 2),
      Header("GRAYSCALE_ALPHA", 2, 256) +
          std::string("\x00\x00\x00\x00\x01\x01\x00\x00", 8),
  };
  for (const auto& bytes : malformed) {
    IREE_ASSERT_OK(Open(bytes));
    std::vector<uint8_t> output(input_.info.pixel_length);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Decode(output));
  }
}

}  // namespace
