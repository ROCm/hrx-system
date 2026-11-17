// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEC_IMAGE_IMAGE_H_
#define IREE_CODEC_IMAGE_IMAGE_H_

#include "iree/base/api.h"

// Supported compressed image formats.
typedef enum iree_codec_image_format_e {
  IREE_CODEC_IMAGE_FORMAT_PNG = 1,
  IREE_CODEC_IMAGE_FORMAT_JPEG = 2,
  IREE_CODEC_IMAGE_FORMAT_GIF = 3,
  IREE_CODEC_IMAGE_FORMAT_WEBP = 4,
  // Built-in PAM visual tuples, normalized from 1..65535 maxval to RGB8/RGBA8.
  IREE_CODEC_IMAGE_FORMAT_PAM = 5,
} iree_codec_image_format_t;

// Packed byte layouts. Values are the number of bytes per pixel. Alpha is
// straight (unassociated); the consumer owns any compositing policy.
typedef enum iree_codec_image_pixel_format_e {
  IREE_CODEC_IMAGE_PIXEL_FORMAT_RGB8 = 3,
  IREE_CODEC_IMAGE_PIXEL_FORMAT_RGBA8 = 4,
} iree_codec_image_pixel_format_t;

// Inspected geometry and packed output layout of one immutable image.
// Pixel storage remains in encoded orientation; consumers apply orientation
// while sampling without requiring another decoded plane.
typedef struct iree_codec_image_info_t {
  // Compressed format established by the codec's signature and header checks.
  iree_codec_image_format_t format;
  // Packed channel layout emitted by the codec, including alpha when present.
  iree_codec_image_pixel_format_t pixel_format;
  // Encoded pixel-plane width.
  uint32_t width;
  // Encoded pixel-plane height.
  uint32_t height;
  // TIFF/EXIF orientation in [1, 8].
  uint8_t orientation;
} iree_codec_image_info_t;

#endif  // IREE_CODEC_IMAGE_IMAGE_H_
