// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/image.h"

#include "libamdf/src/allocator.h"
#include "libamdf/src/structure.h"

static uint64_t amdf_media_drm_image_pitch(
    const amdf_media_image_description_t* description) {
  const uint64_t width =
      (uint64_t)description->width *
      (description->format == AMDF_MEDIA_PIXEL_FORMAT_P010 ? 2u : 1u);
  return (width + 255) & ~UINT64_C(255);
}

static uint32_t amdf_media_drm_image_rows(
    const amdf_media_image_description_t* description) {
  // VCN's separate reference surfaces use a larger reconstruction extent than
  // the linear decode output. Main10 references require complete 64-row tiles.
  const uint32_t alignment =
      (description->roles & AMDF_MEDIA_IMAGE_ROLE_DECODE_REFERENCE) == 0 ? 16
      : description->format == AMDF_MEDIA_PIXEL_FORMAT_P010              ? 64
                                                                         : 32;
  return (description->height + alignment - 1) & ~(alignment - 1);
}

amdf_status_t AMDF_CALL amdf_media_drm_image_requirements(
    amdf_device_t* base, const amdf_media_image_description_t* description,
    amdf_media_image_requirements_t* out_requirements) {
  if (description->width == 0 || description->height == 0 ||
      description->array_layer_count == 0 || description->roles == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if ((description->format != AMDF_MEDIA_PIXEL_FORMAT_NV12 &&
       description->format != AMDF_MEDIA_PIXEL_FORMAT_P010) ||
      (description->roles & ~(AMDF_MEDIA_IMAGE_ROLE_DECODE_OUTPUT |
                              AMDF_MEDIA_IMAGE_ROLE_DECODE_REFERENCE)) != 0 ||
      ((description->width | description->height) & 1) != 0 ||
      description->width > 65534 || description->height > 65520) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const amdf_media_drm_device_t* device = (const amdf_media_drm_device_t*)base;
  const uint64_t pitch = amdf_media_drm_image_pitch(description);
  const uint64_t rows = amdf_media_drm_image_rows(description);
  const uint64_t layer_length = pitch * (rows + rows / 2);
  if (description->array_layer_count >
      (device->address_end - device->address_begin) / layer_length) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  *out_requirements = (amdf_media_image_requirements_t){
      .description = *description,
      .byte_length = layer_length * description->array_layer_count,
      .byte_offset_alignment = 256,
  };
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_media_drm_image_create(
    amdf_device_t* device, const amdf_media_image_create_info_t* create_info,
    amdf_media_image_t** out_image) {
  amdf_media_image_requirements_t requirements;
  amdf_status_t status = amdf_media_drm_image_requirements(
      device, &create_info->requirements.description, &requirements);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (create_info->requirements.byte_length != requirements.byte_length ||
      create_info->requirements.byte_offset_alignment !=
          requirements.byte_offset_alignment) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (create_info->byte_offset % requirements.byte_offset_alignment != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_media_drm_buffer_t* buffer;
  uint64_t address;
  status = amdf_media_drm_memory_range(
      (amdf_media_drm_device_t*)device, create_info->memory,
      create_info->media_access_ordinal, create_info->byte_offset,
      requirements.byte_length, &buffer, &address);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (address % requirements.byte_offset_alignment != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  for (uint32_t i = 0;
       amdf_status_is_ok(status) && i < create_info->layout_access_count; ++i) {
    if (create_info->layout_access_ordinals[i] >=
        create_info->memory->info.access_count) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    }
  }
  amdf_media_drm_image_t* image = NULL;
  if (amdf_status_is_ok(status)) {
    status = amdf_calloc(device->host_allocator, sizeof(*image),
                         amdf_alignof(amdf_media_drm_image_t), (void**)&image);
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  image->base.device = (amdf_media_device_t*)device;
  image->requirements = requirements;
  image->memory = create_info->memory;
  image->access_ordinal = create_info->media_access_ordinal;
  image->byte_offset = create_info->byte_offset;
  image->row_pitch = amdf_media_drm_image_pitch(&requirements.description);
  image->allocated_height =
      amdf_media_drm_image_rows(&requirements.description);
  image->layer_byte_length =
      requirements.byte_length / requirements.description.array_layer_count;
  *out_image = &image->base;
  return AMDF_STATUS_OK;
}

static void amdf_media_drm_image_planes(const amdf_media_drm_image_t* image,
                                        uint32_t access_ordinal, uint32_t layer,
                                        amdf_media_frame_plane_t planes[2]) {
  uint64_t offset = image->byte_offset + image->layer_byte_length * layer;
  for (uint32_t i = 0; i < 2; ++i) {
    const uint64_t length = image->row_pitch * (image->allocated_height >> i);
    planes[i] = (amdf_media_frame_plane_t){
        .memory = image->memory,
        .access_ordinal = access_ordinal,
        .byte_offset = offset,
        .byte_length = length,
        .row_pitch = image->row_pitch,
    };
    offset += length;
  }
}

amdf_status_t AMDF_CALL amdf_media_drm_image_query_info(
    amdf_media_image_t* base, uint32_t access_ordinal,
    const amdf_media_image_query_t* query, amdf_media_image_info_t* out_info) {
  const amdf_media_drm_image_t* image = (const amdf_media_drm_image_t*)base;
  if (access_ordinal >= image->memory->info.access_count ||
      query->array_layer >= image->requirements.description.array_layer_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  amdf_media_linear_image_layout_t* layout = out_info->next;
  if (layout != NULL) {
    const amdf_status_t status = amdf_structure_validate_output(
        layout, AMDF_STRUCTURE_TYPE_MEDIA_LINEAR_IMAGE_LAYOUT, sizeof(*layout));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    layout->plane_count = 2;
    layout->array_layer = query->array_layer;
    amdf_media_drm_image_planes(image, access_ordinal, query->array_layer,
                                layout->planes);
  }
  out_info->memory = image->memory;
  out_info->requirements = image->requirements;
  out_info->byte_offset = image->byte_offset;
  return AMDF_STATUS_OK;
}

void AMDF_CALL amdf_media_drm_image_destroy(amdf_media_image_t* image) {
  amdf_free(image->device->base.host_allocator, image);
}

amdf_status_t amdf_media_drm_frame_prepare(
    amdf_media_drm_device_t* device, const amdf_media_frame_t* frame,
    const amdf_media_image_description_t* description,
    amdf_media_drm_frame_t* out_frame) {
  amdf_status_t status = amdf_structure_validate_input(
      frame, AMDF_STRUCTURE_TYPE_MEDIA_FRAME, sizeof(*frame));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (frame->format != description->format ||
      frame->width != description->width ||
      frame->height != description->height) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_media_frame_plane_t native_planes[2];
  const amdf_media_frame_plane_t* planes;
  if (frame->storage_kind == AMDF_MEDIA_FRAME_STORAGE_KIND_NATIVE_IMAGE) {
    const amdf_media_drm_image_t* image =
        (const amdf_media_drm_image_t*)frame->storage.image.image;
    if (image == NULL || image->base.device != &device->base ||
        image->memory != frame->storage.image.memory ||
        image->byte_offset != frame->storage.image.byte_offset ||
        frame->storage.image.array_layer >=
            image->requirements.description.array_layer_count ||
        image->requirements.description.format != description->format ||
        image->requirements.description.width != description->width ||
        image->requirements.description.height != description->height ||
        (image->requirements.description.roles & description->roles) !=
            description->roles) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
    amdf_media_drm_image_planes(image, image->access_ordinal,
                                frame->storage.image.array_layer,
                                native_planes);
    planes = native_planes;
  } else if (frame->storage_kind ==
             AMDF_MEDIA_FRAME_STORAGE_KIND_LINEAR_PLANES) {
    if (frame->storage.linear.plane_count != 2 ||
        frame->storage.linear.planes == NULL) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
    planes = frame->storage.linear.planes;
  } else {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_media_drm_frame_t result = {0};
  const uint64_t width =
      (uint64_t)description->width *
      (description->format == AMDF_MEDIA_PIXEL_FORMAT_P010 ? 2u : 1u);
  for (uint32_t i = 0; amdf_status_is_ok(status) && i < 2; ++i) {
    const uint64_t rows = amdf_media_drm_image_rows(description) >> i;
    if (planes[i].row_pitch < width || planes[i].row_pitch > UINT32_MAX ||
        planes[i].row_pitch % 256 != 0 ||
        planes[i].byte_length < planes[i].row_pitch * rows) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    } else {
      status = amdf_media_drm_memory_range(
          device, planes[i].memory, planes[i].access_ordinal,
          planes[i].byte_offset, planes[i].byte_length, &result.buffers[i],
          &result.addresses[i]);
      result.row_pitches[i] = (uint32_t)planes[i].row_pitch;
      result.row_counts[i] = (uint32_t)rows;
    }
  }
  if (amdf_status_is_ok(status) &&
      (result.addresses[0] % 256 != 0 || result.addresses[1] % 256 != 0 ||
       result.addresses[1] < result.addresses[0] ||
       result.addresses[1] - result.addresses[0] > UINT32_MAX ||
       (result.buffers[0] == result.buffers[1] &&
        result.addresses[1] - result.addresses[0] <
            (uint64_t)result.row_pitches[0] *
                amdf_media_drm_image_rows(description)))) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (amdf_status_is_ok(status)) {
    *out_frame = result;
  }
  return status;
}
