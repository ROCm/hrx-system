// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_MEDIA_IMAGE_TYPES_H_
#define LOOM_SERVICE_MEDIA_IMAGE_TYPES_H_

#include "iree/codec/image/image.h"
#include "loom/service/media/digest.h"

// Resource bounds applied before pixel realization. Codecs may allocate bounded
// header state while inspecting a compressed payload.
typedef struct loom_service_media_image_limits_t {
  // Maximum decoded compressed payload length.
  iree_host_size_t compressed_bytes;
  // Maximum encoded image width.
  uint32_t width;
  // Maximum encoded image height.
  uint32_t height;
  // Maximum encoded pixel count, independent of width and height bounds.
  uint64_t pixels;
} loom_service_media_image_limits_t;

// Structurally inspected immutable image identity. Encoded geometry describes
// the stored pixel plane; display geometry includes EXIF orientation.
typedef struct loom_service_media_image_info_t {
  // Codec-owned facts about the immutable compressed input and pixel layout.
  iree_codec_image_info_t image;
  // Display width after applying orientation.
  uint32_t display_width;
  // Display height after applying orientation.
  uint32_t display_height;
  // Exact compressed payload length after base64 decoding.
  iree_host_size_t compressed_length;
  // Checked width * height * pixel_format destination extent in bytes.
  iree_host_size_t pixel_length;
  // SHA-256 of the decoded compressed bytes.
  loom_service_media_digest_t digest;
} loom_service_media_image_info_t;

#endif  // LOOM_SERVICE_MEDIA_IMAGE_TYPES_H_
