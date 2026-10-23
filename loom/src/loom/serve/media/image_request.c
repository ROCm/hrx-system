// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/media/image_request.h"

#include <string.h>

iree_status_t loom_serve_media_image_request_size(
    const loom_serve_media_image_info_t* info, iree_host_size_t metadata_length,
    iree_host_size_t* out_length) {
  *out_length = 0;
  iree_host_size_t metadata_end = 0;
  iree_host_size_t pixel_offset = 0;
  iree_host_size_t pixel_end = 0;
  iree_host_size_t packet_length = 0;
  if (!iree_host_size_checked_add(LOOM_SERVE_MEDIA_IMAGE_REQUEST_HEADER_BYTES,
                                  metadata_length, &metadata_end) ||
      !iree_host_size_checked_align(
          metadata_end, LOOM_SERVE_MEDIA_IMAGE_REQUEST_PIXEL_ALIGNMENT,
          &pixel_offset) ||
      !iree_host_size_checked_add(pixel_offset, info->pixel_length,
                                  &pixel_end) ||
      !iree_host_size_checked_align(
          pixel_end, LOOM_SERVE_MEDIA_IMAGE_REQUEST_PIXEL_ALIGNMENT,
          &packet_length) ||
      packet_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "finite image request exceeds 32-bit extent");
  }
  *out_length = packet_length;
  return iree_ok_status();
}

static void loom_serve_media_image_request_write_descriptor(
    const loom_serve_media_image_info_t* info, iree_const_byte_span_t metadata,
    uint32_t output_rows, iree_byte_span_t bytes) {
  const iree_host_size_t pixel_offset = iree_host_align(
      LOOM_SERVE_MEDIA_IMAGE_REQUEST_HEADER_BYTES + metadata.data_length,
      LOOM_SERVE_MEDIA_IMAGE_REQUEST_PIXEL_ALIGNMENT);
  memset(bytes.data, 0, LOOM_SERVE_MEDIA_IMAGE_REQUEST_HEADER_BYTES);
  const uint32_t fields[LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_COUNT] = {
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_VERSION] =
          LOOM_SERVE_MEDIA_IMAGE_REQUEST_VERSION,
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_FORMAT] = info->format,
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_WIDTH] = info->width,
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_HEIGHT] = info->height,
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_DISPLAY_WIDTH] =
          info->display_width,
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_DISPLAY_HEIGHT] =
          info->display_height,
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_ORIENTATION] = info->orientation,
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_OUTPUT_ROWS] = output_rows,
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_METADATA_OFFSET] =
          LOOM_SERVE_MEDIA_IMAGE_REQUEST_HEADER_BYTES,
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_METADATA_LENGTH] =
          (uint32_t)metadata.data_length,
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_PIXEL_OFFSET] =
          (uint32_t)pixel_offset,
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_PIXEL_LENGTH] =
          (uint32_t)info->pixel_length,
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_PIXEL_FORMAT] = info->pixel_format,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(fields); ++i) {
    iree_unaligned_store_le_u32(bytes.data + i * sizeof(uint32_t), fields[i]);
  }
  if (metadata.data_length) {
    memcpy(bytes.data + LOOM_SERVE_MEDIA_IMAGE_REQUEST_HEADER_BYTES,
           metadata.data, metadata.data_length);
  }
  const iree_host_size_t metadata_end =
      LOOM_SERVE_MEDIA_IMAGE_REQUEST_HEADER_BYTES + metadata.data_length;
  memset(bytes.data + metadata_end, 0, pixel_offset - metadata_end);
}

iree_status_t loom_serve_media_image_request_describe(
    const loom_serve_media_image_info_t* info, iree_const_byte_span_t metadata,
    uint32_t output_rows, iree_allocator_t host_allocator,
    iree_vm_buffer_t** out_descriptor) {
  *out_descriptor = NULL;
  iree_host_size_t packet_length = 0;
  IREE_RETURN_IF_ERROR(loom_serve_media_image_request_size(
      info, metadata.data_length, &packet_length));
  const iree_host_size_t length = iree_host_align(
      LOOM_SERVE_MEDIA_IMAGE_REQUEST_HEADER_BYTES + metadata.data_length,
      LOOM_SERVE_MEDIA_IMAGE_REQUEST_PIXEL_ALIGNMENT);
  iree_vm_buffer_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_create(
      length, LOOM_SERVE_MEDIA_IMAGE_REQUEST_PIXEL_ALIGNMENT, host_allocator,
      &descriptor));
  iree_byte_span_t bytes;
  iree_status_t status =
      iree_vm_buffer_map_write(descriptor, 0, length, &bytes);
  if (iree_status_is_ok(status)) {
    loom_serve_media_image_request_write_descriptor(info, metadata, output_rows,
                                                    bytes);
    *out_descriptor = descriptor;
  } else {
    iree_vm_buffer_release(descriptor);
  }
  return status;
}

static iree_status_t loom_serve_media_image_request_allocate(
    const loom_serve_media_image_request_t* request,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_packet,
    iree_byte_span_t* out_pixels) {
  const loom_serve_media_image_info_t* info = &request->image->info;
  iree_host_size_t packet_length = 0;
  IREE_RETURN_IF_ERROR(loom_serve_media_image_request_size(
      info, request->metadata.data_length, &packet_length));
  const iree_host_size_t metadata_end =
      LOOM_SERVE_MEDIA_IMAGE_REQUEST_HEADER_BYTES +
      request->metadata.data_length;
  const iree_host_size_t pixel_offset = iree_host_align(
      metadata_end, LOOM_SERVE_MEDIA_IMAGE_REQUEST_PIXEL_ALIGNMENT);
  const iree_host_size_t pixel_end = pixel_offset + info->pixel_length;
  iree_vm_buffer_t* packet = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_create(
      packet_length, LOOM_SERVE_MEDIA_IMAGE_REQUEST_PIXEL_ALIGNMENT,
      host_allocator, &packet));
  iree_byte_span_t bytes;
  iree_status_t status =
      iree_vm_buffer_map_write(packet, 0, packet_length, &bytes);
  if (iree_status_is_ok(status)) {
    // Initialize only control and alignment bytes; the codec writes the exact
    // complete plane. A failed decoder never publishes a partial packet.
    loom_serve_media_image_request_write_descriptor(
        info, request->metadata, request->output_rows, bytes);
    memset(bytes.data + pixel_end, 0, packet_length - pixel_end);
    *out_pixels =
        iree_make_byte_span(bytes.data + pixel_offset, info->pixel_length);
    *out_packet = packet;
  } else {
    iree_vm_buffer_release(packet);
  }
  return status;
}

iree_status_t loom_serve_media_image_request_create_batch(
    iree_host_size_t count, const loom_serve_media_image_request_t* requests,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_packets,
    iree_status_t* out_statuses) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    out_packets[i] = NULL;
    out_statuses[i] = iree_ok_status();
  }
  if (!count) {
    return iree_ok_status();
  }
  iree_host_size_t scratch_length = 0, inputs_offset = 0, pixels_offset = 0,
                   indices_offset = 0, statuses_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &scratch_length,
      IREE_STRUCT_FIELD(count, const loom_serve_media_image_input_t*,
                        &inputs_offset),
      IREE_STRUCT_FIELD(count, iree_byte_span_t, &pixels_offset),
      IREE_STRUCT_FIELD(count, iree_host_size_t, &indices_offset),
      IREE_STRUCT_FIELD(count, iree_status_t, &statuses_offset)));
  uint8_t* scratch = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      host_allocator, scratch_length, (void**)&scratch));
  const loom_serve_media_image_input_t** inputs =
      (const loom_serve_media_image_input_t**)(scratch + inputs_offset);
  iree_byte_span_t* pixels = (iree_byte_span_t*)(scratch + pixels_offset);
  iree_host_size_t* indices = (iree_host_size_t*)(scratch + indices_offset);
  iree_status_t* statuses = (iree_status_t*)(scratch + statuses_offset);
  iree_host_size_t selected_count = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    out_statuses[i] = loom_serve_media_image_request_allocate(
        &requests[i], host_allocator, &out_packets[i], &pixels[selected_count]);
    if (iree_status_is_ok(out_statuses[i])) {
      inputs[selected_count] = requests[i].image;
      indices[selected_count++] = i;
    }
  }
  iree_status_t status = loom_serve_media_image_decode_batch(
      selected_count, inputs, pixels, statuses, host_allocator);
  for (iree_host_size_t i = 0; i < selected_count; ++i) {
    out_statuses[indices[i]] = statuses[i];
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (!iree_status_is_ok(status) || !iree_status_is_ok(out_statuses[i])) {
      iree_vm_buffer_release(out_packets[i]);
      out_packets[i] = NULL;
    }
  }
  iree_allocator_free(host_allocator, scratch);
  return status;
}
