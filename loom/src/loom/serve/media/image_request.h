// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_MEDIA_IMAGE_REQUEST_H_
#define LOOM_SERVE_MEDIA_IMAGE_REQUEST_H_

#include "iree/vm/buffer.h"
#include "loom/serve/media/image.h"

#ifdef __cplusplus
extern "C" {
#endif

// Source-visible little-endian u32 fields in a finite image request. The fixed
// header occupies 64 bytes; unenumerated trailing words are zero. Metadata is
// source-private checked policy output. Pixels are packed RGB8 in encoded
// row-major orientation and begin at a 16-byte-aligned offset.
typedef enum loom_serve_media_image_request_field_e {
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_VERSION = 0,
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_FORMAT = 1,
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_WIDTH = 2,
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_HEIGHT = 3,
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_DISPLAY_WIDTH = 4,
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_DISPLAY_HEIGHT = 5,
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_ORIENTATION = 6,
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_OUTPUT_ROWS = 7,
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_METADATA_OFFSET = 8,
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_METADATA_LENGTH = 9,
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_PIXEL_OFFSET = 10,
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_PIXEL_LENGTH = 11,
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_COUNT = 12,
} loom_serve_media_image_request_field_t;

enum {
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_VERSION = 1,
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_HEADER_BYTES = 64,
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_PIXEL_ALIGNMENT = 16,
};

// Builds one exact source request from an inspected immutable data URL. The
// returned VM buffer owns its metadata copy and decoded RGB8 plane. Decoding
// re-establishes |expected_info| before accepting pixels. |output_rows| is the
// source-declared result extent checked again at finite batch preparation.
// Failure leaves |out_request| null.
iree_status_t loom_serve_media_image_request_create(
    iree_string_view_t escaped_data_url,
    const loom_serve_media_image_limits_t* limits,
    const loom_serve_media_image_info_t* expected_info,
    iree_const_byte_span_t metadata, uint32_t output_rows,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_MEDIA_IMAGE_REQUEST_H_
