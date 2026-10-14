// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_MEDIA_IMAGE_H_
#define LOOM_SERVE_MEDIA_IMAGE_H_

#include "iree/base/api.h"
#include "loom/serve/media/digest.h"

#ifdef __cplusplus
extern "C" {
#endif

// Supported OpenAI image_url data-URL payload formats.
typedef enum loom_serve_media_image_format_e {
  LOOM_SERVE_MEDIA_IMAGE_FORMAT_PNG = 1,
  LOOM_SERVE_MEDIA_IMAGE_FORMAT_JPEG = 2,
  LOOM_SERVE_MEDIA_IMAGE_FORMAT_GIF = 3,
  LOOM_SERVE_MEDIA_IMAGE_FORMAT_WEBP = 4,
} loom_serve_media_image_format_t;

// Resource bounds applied before codec allocation or pixel realization.
typedef struct loom_serve_media_image_limits_t {
  // Maximum decoded compressed payload length.
  iree_host_size_t compressed_bytes;
  // Maximum encoded image width.
  uint32_t width;
  // Maximum encoded image height.
  uint32_t height;
  // Maximum encoded pixel count, independent of width and height bounds.
  uint64_t pixels;
} loom_serve_media_image_limits_t;

// Structurally inspected immutable image identity. Encoded geometry describes
// the stored pixel plane; display geometry includes EXIF orientation.
typedef struct loom_serve_media_image_info_t {
  // Codec selected jointly by the declared MIME type and byte signature.
  loom_serve_media_image_format_t format;
  // Encoded pixel-plane width.
  uint32_t width;
  // Encoded pixel-plane height.
  uint32_t height;
  // Display width after applying orientation.
  uint32_t display_width;
  // Display height after applying orientation.
  uint32_t display_height;
  // TIFF/EXIF orientation in [1, 8]. Device preprocessing applies it while
  // sampling so the host never allocates a second rotated pixel plane.
  uint8_t orientation;
  // Exact compressed payload length after base64 decoding.
  iree_host_size_t compressed_length;
  // SHA-256 of the decoded compressed bytes.
  loom_serve_media_digest_t digest;
} loom_serve_media_image_info_t;

// Realized packed RGB8 plane in encoded row-major orientation.
typedef struct loom_serve_media_image_t {
  // Inspected identity and source geometry.
  loom_serve_media_image_info_t info;
  // Allocator owning pixels.
  iree_allocator_t host_allocator;
  // Owned width * height * 3 packed RGB bytes.
  iree_byte_span_t pixels;
} loom_serve_media_image_t;

// Inspects one raw JSON string value containing a base64 image data URL.
// |escaped_data_url| excludes its surrounding quotes and may contain JSON
// escapes. The temporary compressed representation is released before return.
// Animated inputs and MIME/signature mismatches are rejected. No pixel plane is
// allocated. Failure zeros |out_info|.
iree_status_t loom_serve_media_image_inspect(
    iree_string_view_t escaped_data_url,
    const loom_serve_media_image_limits_t* limits,
    iree_allocator_t host_allocator, loom_serve_media_image_info_t* out_info);

// Decodes a previously inspected immutable data URL to packed RGB8. The
// descriptor is re-established from the bytes and must exactly match
// |expected_info| before codec output is accepted. Orientation remains metadata
// for device preprocessing. Failure releases partial storage and zeros
// |out_image|.
iree_status_t loom_serve_media_image_decode(
    iree_string_view_t escaped_data_url,
    const loom_serve_media_image_limits_t* limits,
    const loom_serve_media_image_info_t* expected_info,
    iree_allocator_t host_allocator, loom_serve_media_image_t* out_image);

// Releases an image; zero-initialized values are accepted.
void loom_serve_media_image_deinitialize(loom_serve_media_image_t* image);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_MEDIA_IMAGE_H_
