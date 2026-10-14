// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/media/image_request.h"

#include <cstring>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

constexpr char kPng[] =
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP4z8DA8J+"
    "BkYHh////DAAe9gT9Ce00PgAAAABJRU5ErkJggg==";
constexpr loom_serve_media_image_limits_t kLimits = {
    .compressed_bytes = 2048,
    .width = 64,
    .height = 64,
    .pixels = 4096,
};

uint32_t Field(iree_const_byte_span_t bytes,
               loom_serve_media_image_request_field_t field) {
  return iree_unaligned_load_le_u32(bytes.data + field * sizeof(uint32_t));
}

TEST(ImageRequestTest, PacksMetadataAndDecodedPixelsOnce) {
  const std::string url = std::string("data:image/png;base64,") + kPng;
  const iree_string_view_t url_view =
      iree_make_string_view(url.data(), url.size());
  loom_serve_media_image_info_t info;
  IREE_ASSERT_OK(loom_serve_media_image_inspect(
      url_view, &kLimits, iree_allocator_system(), &info));
  const uint8_t metadata[] = {9, 8, 7};
  iree_vm_buffer_t* request = nullptr;
  IREE_ASSERT_OK(loom_serve_media_image_request_create(
      url_view, &kLimits, &info,
      iree_make_const_byte_span(metadata, sizeof(metadata)), 5,
      iree_allocator_system(), &request));
  iree_const_byte_span_t bytes;
  IREE_ASSERT_OK(iree_vm_buffer_map_read(
      request, 0, iree_vm_buffer_length(request), &bytes));
  EXPECT_EQ(Field(bytes, LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_VERSION), 1u);
  EXPECT_EQ(Field(bytes, LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_FORMAT),
            LOOM_SERVE_MEDIA_IMAGE_FORMAT_PNG);
  EXPECT_EQ(Field(bytes, LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_WIDTH), 2u);
  EXPECT_EQ(Field(bytes, LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_HEIGHT), 2u);
  EXPECT_EQ(Field(bytes, LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_OUTPUT_ROWS), 5u);
  EXPECT_EQ(Field(bytes, LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_METADATA_OFFSET),
            64u);
  EXPECT_EQ(Field(bytes, LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_METADATA_LENGTH),
            3u);
  const uint32_t pixel_offset =
      Field(bytes, LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_PIXEL_OFFSET);
  EXPECT_EQ(pixel_offset, 80u);
  EXPECT_EQ(Field(bytes, LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_PIXEL_LENGTH),
            12u);
  EXPECT_EQ(memcmp(bytes.data + 64, metadata, sizeof(metadata)), 0);
  const uint8_t expected[] = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255};
  EXPECT_EQ(memcmp(bytes.data + pixel_offset, expected, sizeof(expected)), 0);
  iree_vm_buffer_release(request);
}

}  // namespace
