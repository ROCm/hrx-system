// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/media/image.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

constexpr char kPng[] =
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP4z8DA8J+"
    "BkYHh////DAAe9gT9Ce00PgAAAABJRU5ErkJggg==";
constexpr char kGif[] =
    "R0lGODdhAgACAIEAAP///wD/AP8AAAAA/ywAAAAAAgACAAAIBwAFBBgAICAAOw==";
constexpr char kWebP[] =
    "UklGRiwAAABXRUJQVlA4TB8AAAAvAUAAAB8gEEjeHzqN+RcQFPwf3fxHZA/gBgwR/"
    "Q8BAA==";
constexpr char kOrientedJpeg[] =
    "/9j/4AAQSkZJRgABAQAAAQABAAD/4QAiRXhpZgAATU0AKgAAAAgAAQESAAMAAAABAAYA"
    "AAAAAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJ"
    "BwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoK"
    "CgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAADAA"
    "IDAREAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAg"
    "EDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2Jy"
    "ggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd"
    "4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1"
    "NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAEC"
    "AwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEI"
    "FEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVld"
    "YWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tb"
    "a3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIR"
    "AxEAPwD9Cf8AgnB4S8CeJf8Agnj8BfEXiL4aeFNQ1DUPgv4Wub+/vfCljLNczPpNqzy"
    "SO0JZ3ZiWLEkkkk1/kf4o+MXHGE8TM8oUHhoQhjMTGMVgMAlFKtNJJLDWSS0SWiR34j"
    "ww8Os/rzzPM8qw9fE126lWrUpRnUqVJvmnUqTknKc5yblOcm5Sk2222f/Z";
constexpr char kAnimatedGif[] =
    "R0lGODlhAgACAIEAAP///wD/AP8AAAAA/yH/C05FVFNDQVBFMi4wAwEAAAAh+QQAAQAA"
    "ACwAAAAAAgACAAAIBwAFBBgAICAAIfkEAQEAAQAsAAAAAAIAAgCBAQIDAAAAAAAAAAAA"
    "CAYAAQgEEBAAOw==";
constexpr char kAnimatedWebP[] =
    "UklGRpgAAABXRUJQVlA4WAoAAAACAAAAAQAAAQAAQU5JTQYAAAAAAAAAAABBTk1GOAAA"
    "AAAAAAAAAAEAAAEAAAoAAAJWUDhMHwAAAC8BQAAAHyAQSN4fOo35FxAU/B/d/EdkD+AG"
    "DBH9DwEAQU5NRiwAAAAAAAAAAAABAAABAAAKAAAAVlA4TBQAAAAvAUAAAAdQgVQIIAAK"
    "mv7HiIj+Bw==";
constexpr char kAnimatedPng[] =
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAACGFjVEwAAAACAAAAAPON"
    "k3AAAAAaZmNUTAAAAAAAAAACAAAAAgAAAAAAAAAAAAEAZAAAqh06KgAAABZJREFUeJxj"
    "+M/AwPCfgZGB4f///wwAHvYE/QntND4AAAAaZmNUTAAAAAEAAAACAAAAAgAAAAAAAAAA"
    "AAEAZAAAMW7Q/gAAABpmZEFUAAAAAnicY2RkYmZgYGBiYGBgYGAAAABwAAqS4G35AAAA"
    "AElFTkSuQmCC";

constexpr loom_serve_media_image_limits_t kLimits = {
    .compressed_bytes = 2048,
    .width = 64,
    .height = 64,
    .pixels = 4096,
};

std::string DataUrl(const char* media_type, const char* encoded) {
  return std::string("data:") + media_type + ";base64," + encoded;
}

iree_string_view_t View(const std::string& value) {
  return iree_make_string_view(value.data(), value.size());
}

std::vector<uint8_t> Pixels(const loom_serve_media_image_t& image) {
  return std::vector<uint8_t>(image.pixels.data,
                              image.pixels.data + image.pixels.data_length);
}

TEST(ImageTest, InspectsAndDecodesLosslessFormats) {
  struct Case {
    const char* media_type;
    const char* encoded;
    loom_serve_media_image_format_t format;
  };
  const Case cases[] = {
      {"image/png", kPng, LOOM_SERVE_MEDIA_IMAGE_FORMAT_PNG},
      {"image/gif", kGif, LOOM_SERVE_MEDIA_IMAGE_FORMAT_GIF},
      {"image/webp", kWebP, LOOM_SERVE_MEDIA_IMAGE_FORMAT_WEBP}};
  const std::vector<uint8_t> expected = {255, 0, 0,   0,   255, 0,
                                         0,   0, 255, 255, 255, 255};
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.media_type);
    const std::string url = DataUrl(test_case.media_type, test_case.encoded);
    loom_serve_media_image_info_t info;
    IREE_ASSERT_OK(loom_serve_media_image_inspect(
        View(url), &kLimits, iree_allocator_system(), &info));
    EXPECT_EQ(info.format, test_case.format);
    EXPECT_EQ(info.width, 2u);
    EXPECT_EQ(info.height, 2u);
    EXPECT_EQ(info.display_width, 2u);
    EXPECT_EQ(info.display_height, 2u);
    EXPECT_EQ(info.orientation, 1u);

    loom_serve_media_image_t image;
    IREE_ASSERT_OK(loom_serve_media_image_decode(
        View(url), &kLimits, &info, iree_allocator_system(), &image));
    EXPECT_EQ(Pixels(image), expected);
    loom_serve_media_image_deinitialize(&image);
  }
}

TEST(ImageTest, PreservesOrientationForDeviceSampling) {
  const std::string url = DataUrl("image/jpeg", kOrientedJpeg);
  loom_serve_media_image_info_t info;
  IREE_ASSERT_OK(loom_serve_media_image_inspect(
      View(url), &kLimits, iree_allocator_system(), &info));
  EXPECT_EQ(info.format, LOOM_SERVE_MEDIA_IMAGE_FORMAT_JPEG);
  EXPECT_EQ(info.width, 2u);
  EXPECT_EQ(info.height, 3u);
  EXPECT_EQ(info.display_width, 3u);
  EXPECT_EQ(info.display_height, 2u);
  EXPECT_EQ(info.orientation, 6u);

  loom_serve_media_image_t image;
  IREE_ASSERT_OK(loom_serve_media_image_decode(
      View(url), &kLimits, &info, iree_allocator_system(), &image));
  EXPECT_EQ(image.pixels.data_length, 18u);
  loom_serve_media_image_deinitialize(&image);
}

TEST(ImageTest, DecodesIntoCallerOwnedExactStorage) {
  const std::string url = DataUrl("image/png", kPng);
  loom_serve_media_image_info_t info;
  IREE_ASSERT_OK(loom_serve_media_image_inspect(
      View(url), &kLimits, iree_allocator_system(), &info));
  std::vector<uint8_t> pixels(12);
  IREE_ASSERT_OK(loom_serve_media_image_decode_into(
      View(url), &kLimits, &info,
      iree_make_byte_span(pixels.data(), pixels.size()),
      iree_allocator_system()));
  EXPECT_EQ(pixels, (std::vector<uint8_t>{255, 0, 0, 0, 255, 0, 0, 0, 255, 255,
                                          255, 255}));

  pixels.pop_back();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_media_image_decode_into(
                            View(url), &kLimits, &info,
                            iree_make_byte_span(pixels.data(), pixels.size()),
                            iree_allocator_system()));
}

TEST(ImageTest, AcceptsEscapedJsonStringContent) {
  std::string url = DataUrl("image/png", kPng);
  for (iree_host_size_t position = 0;
       (position = url.find('/', position)) != std::string::npos;
       position += 2) {
    url.replace(position, 1, "\\/");
  }
  loom_serve_media_image_info_t info;
  IREE_ASSERT_OK(loom_serve_media_image_inspect(
      View(url), &kLimits, iree_allocator_system(), &info));
  EXPECT_EQ(info.width, 2u);
  EXPECT_EQ(info.height, 2u);
}

TEST(ImageTest, RejectsAnimationsBeforePixelAllocation) {
  const std::string urls[] = {DataUrl("image/png", kAnimatedPng),
                              DataUrl("image/gif", kAnimatedGif),
                              DataUrl("image/webp", kAnimatedWebP)};
  for (const std::string& url : urls) {
    loom_serve_media_image_info_t info;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_media_image_inspect(View(url), &kLimits,
                                       iree_allocator_system(), &info));
  }
}

TEST(ImageTest, RejectsMimeSignatureMismatchAndResourceExcess) {
  loom_serve_media_image_info_t info;
  const std::string mismatch = DataUrl("image/jpeg", kPng);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_media_image_inspect(View(mismatch), &kLimits,
                                     iree_allocator_system(), &info));

  auto narrow_limits = kLimits;
  narrow_limits.width = 1;
  const std::string png = DataUrl("image/png", kPng);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_media_image_inspect(View(png), &narrow_limits,
                                     iree_allocator_system(), &info));
}

TEST(ImageTest, RefusesBytesDifferentFromAdmissionIdentity) {
  const std::string png = DataUrl("image/png", kPng);
  loom_serve_media_image_info_t info;
  IREE_ASSERT_OK(loom_serve_media_image_inspect(
      View(png), &kLimits, iree_allocator_system(), &info));
  info.digest.bytes[0] ^= 1;
  loom_serve_media_image_t image;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_serve_media_image_decode(View(png), &kLimits, &info,
                                    iree_allocator_system(), &image));
}

}  // namespace
