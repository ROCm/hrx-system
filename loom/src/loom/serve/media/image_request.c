// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/media/image_request.h"

#include <string.h>

iree_status_t loom_serve_media_image_request_create(
    iree_string_view_t escaped_data_url,
    const loom_serve_media_image_limits_t* limits,
    const loom_serve_media_image_info_t* expected_info,
    iree_const_byte_span_t metadata, uint32_t output_rows,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_request) {
  *out_request = NULL;
  iree_host_size_t pixel_count = 0;
  iree_host_size_t pixel_length = 0;
  iree_host_size_t metadata_end = 0;
  iree_host_size_t pixel_offset = 0;
  iree_host_size_t request_length = 0;
  if (!iree_host_size_checked_mul((iree_host_size_t)expected_info->width,
                                  (iree_host_size_t)expected_info->height,
                                  &pixel_count) ||
      !iree_host_size_checked_mul(pixel_count, 3, &pixel_length) ||
      !iree_host_size_checked_add(LOOM_SERVE_MEDIA_IMAGE_REQUEST_HEADER_BYTES,
                                  metadata.data_length, &metadata_end) ||
      !iree_host_size_checked_align(
          metadata_end, LOOM_SERVE_MEDIA_IMAGE_REQUEST_PIXEL_ALIGNMENT,
          &pixel_offset) ||
      !iree_host_size_checked_add(pixel_offset, pixel_length,
                                  &request_length) ||
      metadata.data_length > UINT32_MAX || pixel_offset > UINT32_MAX ||
      pixel_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "finite image request exceeds 32-bit extents");
  }
  iree_vm_buffer_t* request = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_create(
      request_length, LOOM_SERVE_MEDIA_IMAGE_REQUEST_PIXEL_ALIGNMENT,
      host_allocator, &request));
  iree_byte_span_t bytes;
  iree_status_t status =
      iree_vm_buffer_map_write(request, 0, request_length, &bytes);
  if (iree_status_is_ok(status)) {
    memset(bytes.data, 0, bytes.data_length);
    const uint32_t fields[LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_COUNT] = {
        [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_VERSION] =
            LOOM_SERVE_MEDIA_IMAGE_REQUEST_VERSION,
        [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_FORMAT] = expected_info->format,
        [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_WIDTH] = expected_info->width,
        [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_HEIGHT] = expected_info->height,
        [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_DISPLAY_WIDTH] =
            expected_info->display_width,
        [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_DISPLAY_HEIGHT] =
            expected_info->display_height,
        [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_ORIENTATION] =
            expected_info->orientation,
        [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_OUTPUT_ROWS] = output_rows,
        [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_METADATA_OFFSET] =
            LOOM_SERVE_MEDIA_IMAGE_REQUEST_HEADER_BYTES,
        [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_METADATA_LENGTH] =
            (uint32_t)metadata.data_length,
        [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_PIXEL_OFFSET] =
            (uint32_t)pixel_offset,
        [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_PIXEL_LENGTH] =
            (uint32_t)pixel_length,
    };
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(fields); ++i) {
      iree_unaligned_store_le_u32(bytes.data + i * sizeof(uint32_t), fields[i]);
    }
    if (metadata.data_length) {
      memcpy(bytes.data + LOOM_SERVE_MEDIA_IMAGE_REQUEST_HEADER_BYTES,
             metadata.data, metadata.data_length);
    }
    status = loom_serve_media_image_decode_into(
        escaped_data_url, limits, expected_info,
        iree_make_byte_span(bytes.data + pixel_offset, pixel_length),
        host_allocator);
  }
  if (iree_status_is_ok(status)) {
    *out_request = request;
  } else {
    iree_vm_buffer_release(request);
  }
  return status;
}
