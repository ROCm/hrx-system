// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEC_IMAGE_CODEC_H_
#define IREE_CODEC_IMAGE_CODEC_H_

#include "iree/codec/image/image.h"

#ifdef __cplusplus
extern "C" {
#endif

// One immutable compressed image and its independently owned inspection result.
// A provider writes encoded geometry, pixel format, orientation, and status.
// The caller establishes resource bounds before allocating pixel storage.
typedef struct iree_codec_image_inspection_t {
  // Compressed bytes borrowed through inspection.
  iree_const_byte_span_t data;
  // Inspected geometry; zeroed when inspection fails.
  iree_codec_image_info_t info;
  // Result owned by the caller and consumed exactly once.
  iree_status_t status;
} iree_codec_image_inspection_t;

// One inspected immutable image and caller-owned decode destination.
// The caller retains the inspected compressed bytes and supplies the exact
// width * height * pixel_format destination extent. A failing item does not
// prevent other items from decoding.
typedef struct iree_codec_image_decoding_t {
  // Compressed bytes borrowed through decode.
  iree_const_byte_span_t data;
  // Matching inspected geometry borrowed through decode.
  const iree_codec_image_info_t* info;
  // Exact packed RGB8 or RGBA8 destination; undefined on failure.
  iree_byte_span_t pixels;
  // Result owned by the caller and consumed exactly once.
  iree_status_t status;
} iree_codec_image_decoding_t;

// Immutable codec implementation. CPU implementations may unroll a batch;
// callers retain the cohort boundary. No provider owns a persistent image
// cache or pixel allocation. Pixels remain in encoded row-major orientation.
// Providers emit RGB or straight-alpha RGBA samples without color-profile or
// gamma conversion. The consumer owns compositing, resizing, and
// normalization, so no information is discarded at the codec boundary.
typedef struct iree_codec_image_codec_t {
  // Stable format identity shared by inspection and realization.
  iree_codec_image_format_t format;
  // Canonical advertised MIME type.
  iree_string_view_t mime_type;
  // Inspects every item without realizing a pixel plane.
  void(IREE_API_PTR* inspect)(iree_host_size_t count,
                              iree_codec_image_inspection_t* images);
  // Decodes every item into its exact borrowed destination.
  void(IREE_API_PTR* decode)(iree_host_size_t count,
                             iree_codec_image_decoding_t* images);
} iree_codec_image_codec_t;

// Immutable implementations with process lifetime. Each implementation has
// its own link target; applications choose which implementations to include.
IREE_API_EXPORT_VARIABLE const iree_codec_image_codec_t
    iree_codec_image_codec_pam;
IREE_API_EXPORT_VARIABLE const iree_codec_image_codec_t
    iree_codec_image_codec_png;
IREE_API_EXPORT_VARIABLE const iree_codec_image_codec_t
    iree_codec_image_codec_jpeg;
IREE_API_EXPORT_VARIABLE const iree_codec_image_codec_t
    iree_codec_image_codec_gif;
IREE_API_EXPORT_VARIABLE const iree_codec_image_codec_t
    iree_codec_image_codec_webp;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_CODEC_IMAGE_CODEC_H_
