// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/media/image_request.h"

#include <string.h>

typedef struct loom_serve_media_image_request_layout_t {
  // First metadata byte after the complete plane directory.
  iree_host_size_t metadata_offset;
  // First pixel byte; also the complete descriptor extent.
  iree_host_size_t pixel_offset;
  // Complete aligned packet extent, including every plane.
  iree_host_size_t packet_length;
} loom_serve_media_image_request_layout_t;

static const loom_serve_media_image_info_t* loom_serve_media_image_info_at(
    const loom_serve_media_image_info_t* infos, iree_host_size_t stride,
    iree_host_size_t index) {
  return (const loom_serve_media_image_info_t*)((const uint8_t*)infos +
                                                index * stride);
}

static iree_status_t loom_serve_media_image_request_layout(
    iree_host_size_t image_count, const loom_serve_media_image_info_t* infos,
    iree_host_size_t info_stride, iree_host_size_t metadata_length,
    loom_serve_media_image_request_layout_t* out_layout) {
  iree_host_size_t directory_length = 0;
  iree_host_size_t metadata_offset = 0;
  iree_host_size_t metadata_end = 0;
  iree_host_size_t pixel_offset = 0;
  if (!iree_host_size_checked_mul(image_count,
                                  LOOM_SERVE_MEDIA_IMAGE_REQUEST_PLANE_BYTES,
                                  &directory_length) ||
      !iree_host_size_checked_add(LOOM_SERVE_MEDIA_IMAGE_REQUEST_HEADER_BYTES,
                                  directory_length, &metadata_offset) ||
      !iree_host_size_checked_add(metadata_offset, metadata_length,
                                  &metadata_end) ||
      !iree_host_size_checked_align(
          metadata_end, LOOM_SERVE_MEDIA_IMAGE_REQUEST_PIXEL_ALIGNMENT,
          &pixel_offset) ||
      pixel_offset > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "finite image request exceeds 32-bit extent");
  }
  iree_host_size_t packet_length = pixel_offset;
  for (iree_host_size_t i = 0; i < image_count; ++i) {
    const loom_serve_media_image_info_t* info =
        loom_serve_media_image_info_at(infos, info_stride, i);
    if (!iree_host_size_checked_add(packet_length, info->pixel_length,
                                    &packet_length) ||
        !iree_host_size_checked_align(
            packet_length, LOOM_SERVE_MEDIA_IMAGE_REQUEST_PIXEL_ALIGNMENT,
            &packet_length) ||
        packet_length > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "finite image request exceeds 32-bit extent");
    }
  }
  *out_layout = (loom_serve_media_image_request_layout_t){
      .metadata_offset = metadata_offset,
      .pixel_offset = pixel_offset,
      .packet_length = packet_length,
  };
  return iree_ok_status();
}

iree_status_t loom_serve_media_image_request_size(
    iree_host_size_t image_count, const loom_serve_media_image_info_t* infos,
    iree_host_size_t info_stride, iree_host_size_t metadata_length,
    iree_host_size_t* out_length) {
  *out_length = 0;
  loom_serve_media_image_request_layout_t layout;
  IREE_RETURN_IF_ERROR(loom_serve_media_image_request_layout(
      image_count, infos, info_stride, metadata_length, &layout));
  *out_length = layout.packet_length;
  return iree_ok_status();
}

static void loom_serve_media_image_request_write_descriptor(
    iree_host_size_t image_count, const loom_serve_media_image_info_t* infos,
    iree_host_size_t info_stride, iree_const_byte_span_t metadata,
    uint32_t output_rows, const loom_serve_media_image_request_layout_t* layout,
    uint8_t* bytes) {
  const uint32_t fields[LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_COUNT] = {
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_VERSION] =
          LOOM_SERVE_MEDIA_IMAGE_REQUEST_VERSION,
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_PLANE_COUNT] =
          (uint32_t)image_count,
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_OUTPUT_ROWS] = output_rows,
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_METADATA_OFFSET] =
          (uint32_t)layout->metadata_offset,
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_METADATA_LENGTH] =
          (uint32_t)metadata.data_length,
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_PLANES_OFFSET] =
          LOOM_SERVE_MEDIA_IMAGE_REQUEST_HEADER_BYTES,
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_PIXEL_OFFSET] =
          (uint32_t)layout->pixel_offset,
      [LOOM_SERVE_MEDIA_IMAGE_REQUEST_FIELD_PACKET_LENGTH] =
          (uint32_t)layout->packet_length,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(fields); ++i) {
    iree_unaligned_store_le_u32(bytes + i * sizeof(uint32_t), fields[i]);
  }
  iree_host_size_t pixel_offset = layout->pixel_offset;
  for (iree_host_size_t i = 0; i < image_count; ++i) {
    const loom_serve_media_image_info_t* info =
        loom_serve_media_image_info_at(infos, info_stride, i);
    uint8_t* record = bytes + LOOM_SERVE_MEDIA_IMAGE_REQUEST_HEADER_BYTES +
                      i * LOOM_SERVE_MEDIA_IMAGE_REQUEST_PLANE_BYTES;
    const uint32_t plane[] = {
        (uint32_t)info->format,
        info->width,
        info->height,
        info->display_width,
        info->display_height,
        info->orientation,
        (uint32_t)pixel_offset,
        (uint32_t)info->pixel_length,
        (uint32_t)info->pixel_format,
        0,
        0,
        0,
    };
    for (iree_host_size_t field = 0; field < IREE_ARRAYSIZE(plane); ++field) {
      iree_unaligned_store_le_u32(record + field * sizeof(uint32_t),
                                  plane[field]);
    }
    pixel_offset =
        iree_host_align(pixel_offset + info->pixel_length,
                        LOOM_SERVE_MEDIA_IMAGE_REQUEST_PIXEL_ALIGNMENT);
  }
  if (metadata.data_length) {
    memcpy(bytes + layout->metadata_offset, metadata.data,
           metadata.data_length);
  }
  const iree_host_size_t metadata_end =
      layout->metadata_offset + metadata.data_length;
  memset(bytes + metadata_end, 0, layout->pixel_offset - metadata_end);
}

iree_status_t loom_serve_media_image_request_describe(
    iree_host_size_t image_count, const loom_serve_media_image_info_t* infos,
    iree_host_size_t info_stride, iree_const_byte_span_t metadata,
    uint32_t output_rows, iree_allocator_t host_allocator,
    iree_vm_buffer_t** out_descriptor) {
  *out_descriptor = NULL;
  loom_serve_media_image_request_layout_t layout;
  IREE_RETURN_IF_ERROR(loom_serve_media_image_request_layout(
      image_count, infos, info_stride, metadata.data_length, &layout));
  iree_vm_buffer_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_create(
      layout.pixel_offset, LOOM_SERVE_MEDIA_IMAGE_REQUEST_PIXEL_ALIGNMENT,
      host_allocator, &descriptor));
  loom_serve_media_image_request_write_descriptor(
      image_count, infos, info_stride, metadata, output_rows, &layout,
      iree_vm_buffer_data(descriptor));
  *out_descriptor = descriptor;
  return iree_ok_status();
}

static iree_status_t loom_serve_media_image_request_allocate(
    const loom_serve_media_image_request_t* request,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_packet,
    iree_byte_span_t* out_pixels) {
  loom_serve_media_image_request_layout_t layout;
  IREE_RETURN_IF_ERROR(loom_serve_media_image_request_layout(
      request->image_count, &request->images[0].info, sizeof(*request->images),
      request->metadata.data_length, &layout));
  iree_vm_buffer_t* packet = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_create(
      layout.packet_length, LOOM_SERVE_MEDIA_IMAGE_REQUEST_PIXEL_ALIGNMENT,
      host_allocator, &packet));
  uint8_t* bytes = iree_vm_buffer_data(packet);
  loom_serve_media_image_request_write_descriptor(
      request->image_count, &request->images[0].info, sizeof(*request->images),
      request->metadata, request->output_rows, &layout, bytes);
  iree_host_size_t pixel_offset = layout.pixel_offset;
  for (iree_host_size_t i = 0; i < request->image_count; ++i) {
    const iree_host_size_t length = request->images[i].info.pixel_length;
    const iree_host_size_t end = pixel_offset + length;
    const iree_host_size_t next =
        iree_host_align(end, LOOM_SERVE_MEDIA_IMAGE_REQUEST_PIXEL_ALIGNMENT);
    // Only alignment bytes are cleared; codecs fill the complete planes.
    memset(bytes + end, 0, next - end);
    out_pixels[i] = iree_make_byte_span(bytes + pixel_offset, length);
    pixel_offset = next;
  }
  *out_packet = packet;
  return iree_ok_status();
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
  iree_host_size_t image_count = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (!iree_host_size_checked_add(image_count, requests[i].image_count,
                                    &image_count)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "finite image cohort exceeds host storage");
    }
  }
  iree_host_size_t scratch_length = 0, inputs_offset = 0, pixels_offset = 0,
                   indices_offset = 0, statuses_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &scratch_length,
      IREE_STRUCT_FIELD(image_count, const loom_serve_media_image_input_t*,
                        &inputs_offset),
      IREE_STRUCT_FIELD(image_count, iree_byte_span_t, &pixels_offset),
      IREE_STRUCT_FIELD(image_count, iree_host_size_t, &indices_offset),
      IREE_STRUCT_FIELD(image_count, iree_status_t, &statuses_offset)));
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
      for (iree_host_size_t j = 0; j < requests[i].image_count; ++j) {
        inputs[selected_count] = &requests[i].images[j];
        indices[selected_count++] = i;
      }
    }
  }
  iree_status_t status = loom_serve_media_image_decode_batch(
      selected_count, inputs, pixels, host_allocator, statuses);
  for (iree_host_size_t i = 0; i < selected_count; ++i) {
    out_statuses[indices[i]] =
        iree_status_join(out_statuses[indices[i]], statuses[i]);
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
