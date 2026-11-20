// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_MEDIA_IMAGE_REQUEST_H_
#define LOOM_SERVICE_MEDIA_IMAGE_REQUEST_H_

#include "iree/vm/buffer.h"
#include "loom/service/media/image.h"

#ifdef __cplusplus
extern "C" {
#endif

// Source-visible little-endian u32 fields in one finite projection request.
// The header precedes a directory of distinct input planes, source-private
// metadata, and aligned RGB8/RGBA8 planes in encoded row-major orientation.
// Device preprocessing owns orientation, compositing, resizing, and temporal
// grouping. Every plane and the complete packet are padded to 16 bytes.
typedef enum loom_service_media_image_request_field_e {
  LOOM_SERVICE_MEDIA_IMAGE_REQUEST_FIELD_VERSION = 0,
  LOOM_SERVICE_MEDIA_IMAGE_REQUEST_FIELD_PLANE_COUNT = 1,
  LOOM_SERVICE_MEDIA_IMAGE_REQUEST_FIELD_OUTPUT_ROWS = 2,
  LOOM_SERVICE_MEDIA_IMAGE_REQUEST_FIELD_METADATA_OFFSET = 3,
  LOOM_SERVICE_MEDIA_IMAGE_REQUEST_FIELD_METADATA_LENGTH = 4,
  LOOM_SERVICE_MEDIA_IMAGE_REQUEST_FIELD_PLANES_OFFSET = 5,
  LOOM_SERVICE_MEDIA_IMAGE_REQUEST_FIELD_PIXEL_OFFSET = 6,
  LOOM_SERVICE_MEDIA_IMAGE_REQUEST_FIELD_PACKET_LENGTH = 7,
  LOOM_SERVICE_MEDIA_IMAGE_REQUEST_FIELD_COUNT = 8,
} loom_service_media_image_request_field_t;

// One plane directory record. FORMAT identifies the codec; PIXEL_FORMAT
// identifies the channel layout. Offsets are relative to the complete packet.
// The remaining words in each 48-byte record are zero.
typedef enum loom_service_media_image_plane_field_e {
  LOOM_SERVICE_MEDIA_IMAGE_PLANE_FIELD_FORMAT = 0,
  LOOM_SERVICE_MEDIA_IMAGE_PLANE_FIELD_WIDTH = 1,
  LOOM_SERVICE_MEDIA_IMAGE_PLANE_FIELD_HEIGHT = 2,
  LOOM_SERVICE_MEDIA_IMAGE_PLANE_FIELD_DISPLAY_WIDTH = 3,
  LOOM_SERVICE_MEDIA_IMAGE_PLANE_FIELD_DISPLAY_HEIGHT = 4,
  LOOM_SERVICE_MEDIA_IMAGE_PLANE_FIELD_ORIENTATION = 5,
  LOOM_SERVICE_MEDIA_IMAGE_PLANE_FIELD_PIXEL_OFFSET = 6,
  LOOM_SERVICE_MEDIA_IMAGE_PLANE_FIELD_PIXEL_LENGTH = 7,
  LOOM_SERVICE_MEDIA_IMAGE_PLANE_FIELD_PIXEL_FORMAT = 8,
  LOOM_SERVICE_MEDIA_IMAGE_PLANE_FIELD_COUNT = 9,
} loom_service_media_image_plane_field_t;

enum {
  LOOM_SERVICE_MEDIA_IMAGE_REQUEST_VERSION = 2,
  LOOM_SERVICE_MEDIA_IMAGE_REQUEST_HEADER_BYTES = 32,
  LOOM_SERVICE_MEDIA_IMAGE_REQUEST_PLANE_BYTES = 48,
  LOOM_SERVICE_MEDIA_IMAGE_REQUEST_PIXEL_ALIGNMENT = 16,
};

// One nonempty inspected image group and its source-owned projection policy.
// Groups from several text requests can share a cohort without changing their
// ordinals. Repeated temporal padding aliases a plane in source, not here.
typedef struct loom_service_media_image_request_t {
  // Number of distinct inputs in this group.
  iree_host_size_t image_count;
  // Borrowed contiguous immutable inputs inspected by the codec facade.
  const loom_service_media_image_input_t* images;
  // Borrowed checked preprocessing metadata, copied into the source packet.
  iree_const_byte_span_t metadata;
  // Positive source-declared row count, checked at finite batch preparation.
  uint32_t output_rows;
} loom_service_media_image_request_t;

// Exact checked packet extent, available before transport or pixel allocation.
// Admission uses the same layout calculation as packet realization.
// infos names image_count inspected records separated by info_stride bytes;
// this borrows facts embedded in either transport inputs or chat descriptors
// without allocating a second array. Groups are nonempty.
iree_status_t loom_service_media_image_request_size(
    iree_host_size_t image_count, const loom_service_media_image_info_t* infos,
    iree_host_size_t info_stride, iree_host_size_t metadata_length,
    iree_host_size_t* out_length);

// Creates only the header and source metadata for memory forecasting, without
// retaining the compressed image or decoding pixels. The descriptor is the
// exact prefix of the eventual packet, ending at PIXEL_OFFSET. The directory
// still describes every future plane; the descriptor contains no pixels and is
// not an execution input. The inspected info and metadata are borrowed only
// through this call. Success transfers one VM buffer.
iree_status_t loom_service_media_image_request_describe(
    iree_host_size_t image_count, const loom_service_media_image_info_t* infos,
    iree_host_size_t info_stride, iree_const_byte_span_t metadata,
    uint32_t output_rows, iree_allocator_t host_allocator,
    iree_vm_buffer_t** out_descriptor);

// Realizes a cohort directly into exact source packets, with no intermediate
// pixel planes. Inputs/metadata remain borrowed through return. Each successful
// output owns one VM buffer; a per-plane error leaves its group's output null
// without rejecting valid peer groups. All out_statuses must be consumed,
// including when the returned shared allocation status fails. Shared failure
// releases every output buffer while preserving any per-group errors already
// produced.
//
// The owner selects the cohort within its byte budget before calling. Retained
// packet bytes are align(32 + 48*image_count + metadata_length, 16), plus the
// sum of aligned plane lengths and VM ownership overhead. Temporary scratch
// is proportional to input count, never pixels. Empty cohorts allocate nothing.
iree_status_t loom_service_media_image_request_create_batch(
    iree_host_size_t count, const loom_service_media_image_request_t* requests,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_packets,
    iree_status_t* out_statuses);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVICE_MEDIA_IMAGE_REQUEST_H_
