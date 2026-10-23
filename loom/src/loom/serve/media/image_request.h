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

// Source-visible little-endian u32 fields in a finite image request. The
// 64-byte header has zero reserved words. Source-private metadata follows it;
// packed RGB8/RGBA8 pixels begin at a 16-byte-aligned offset in encoded
// row-major orientation. Device preprocessing owns orientation, compositing,
// resizing, and normalization. FORMAT identifies the codec, not the channels.
// Complete packets include zero tail padding to 16 bytes so concatenation
// preserves header and pixel alignment across differently shaped images.
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
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_PIXEL_FORMAT = 12,
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_COUNT = 13,
} loom_serve_media_image_request_field_t;

enum {
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_VERSION = 1,
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_HEADER_BYTES = 64,
  LOOM_SERVE_MEDIA_IMAGE_REQUEST_PIXEL_ALIGNMENT = 16,
};

// One inspected image and its source-owned projection policy. Instances from
// several text requests can share a cohort without changing their ordinals.
typedef struct loom_serve_media_image_request_t {
  // Borrowed immutable input already inspected by the configured codec facade.
  const loom_serve_media_image_input_t* image;
  // Borrowed checked preprocessing metadata, copied into the source packet.
  iree_const_byte_span_t metadata;
  // Positive source-declared row count, checked at finite batch preparation.
  uint32_t output_rows;
} loom_serve_media_image_request_t;

// Exact checked packet extent, available before transport or pixel allocation.
// Admission uses the same layout calculation as packet realization.
iree_status_t loom_serve_media_image_request_size(
    const loom_serve_media_image_info_t* info, iree_host_size_t metadata_length,
    iree_host_size_t* out_length);

// Creates only the header and source metadata for memory forecasting, without
// retaining the compressed image or decoding pixels. The descriptor is the
// exact prefix of the eventual packet, ending at PIXEL_OFFSET. PIXEL_LENGTH
// still describes the future plane; the descriptor does not contain it and is
// not an execution input. The inspected info and metadata are borrowed only
// through this call. Success transfers one VM buffer.
iree_status_t loom_serve_media_image_request_describe(
    const loom_serve_media_image_info_t* info, iree_const_byte_span_t metadata,
    uint32_t output_rows, iree_allocator_t host_allocator,
    iree_vm_buffer_t** out_descriptor);

// Realizes a cohort directly into exact source packets, with no intermediate
// pixel planes. Inputs/metadata remain borrowed through return. Each successful
// output owns one VM buffer; a per-image error leaves that output null without
// rejecting valid peers. All out_statuses must be consumed, including when the
// returned shared allocation status fails. Shared failure releases every output
// buffer while preserving any per-image errors already produced.
//
// The owner selects the cohort within its byte budget before calling. Retained
// packet bytes are align(align(64 + metadata_length, 16) + pixel_length, 16)
// per image, plus VM ownership overhead. Temporary scratch is O(count), never
// O(pixels). No persistent cache is created. Empty cohorts allocate nothing.
iree_status_t loom_serve_media_image_request_create_batch(
    iree_host_size_t count, const loom_serve_media_image_request_t* requests,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_packets,
    iree_status_t* out_statuses);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_MEDIA_IMAGE_REQUEST_H_
