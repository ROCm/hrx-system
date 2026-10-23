// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_DRM_IMAGE_H_
#define AMDF_SRC_MEDIA_DRM_IMAGE_H_

#include "libamdf/src/media/drm/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct amdf_media_drm_image_t {
  // Borrowed media activation; the image owns no pixel allocation.
  amdf_media_image_t base;
  // Immutable description and complete layer-array footprint.
  amdf_media_image_requirements_t requirements;
  // Direct core backing borrowed through image destruction.
  amdf_memory_t* memory;
  // Exact media consumer of that backing.
  uint32_t access_ordinal;
  // Logical placement of the first layer.
  uint64_t byte_offset;
  // Byte distance between adjacent layers.
  uint64_t layer_byte_length;
  // Byte distance between adjacent rows in either component plane.
  uint64_t row_pitch;
  // Allocated luma row count including native block padding.
  uint32_t allocated_height;
} amdf_media_drm_image_t;

typedef struct amdf_media_drm_frame_t {
  // Direct native attachments retained by the application's memory owners.
  amdf_media_drm_buffer_t* buffers[2];
  // Complete native addresses of the luma and interleaved chroma planes.
  uint64_t addresses[2];
  // Independently specified byte pitches for both component planes.
  uint32_t row_pitches[2];
  // Native write footprint in rows, including codec-block padding.
  uint32_t row_counts[2];
} amdf_media_drm_frame_t;

amdf_status_t AMDF_CALL amdf_media_drm_image_requirements(
    amdf_device_t* device, const amdf_media_image_description_t* description,
    amdf_media_image_requirements_t* out_requirements);
amdf_status_t AMDF_CALL amdf_media_drm_image_create(
    amdf_device_t* device, const amdf_media_image_create_info_t* create_info,
    amdf_media_image_t** out_image);
amdf_status_t AMDF_CALL amdf_media_drm_image_query_info(
    amdf_media_image_t* image, uint32_t access_ordinal,
    const amdf_media_image_query_t* query, amdf_media_image_info_t* out_info);
void AMDF_CALL amdf_media_drm_image_destroy(amdf_media_image_t* image);

// Resolves public linear or placed operands into checked native ranges.
amdf_status_t amdf_media_drm_frame_prepare(
    amdf_media_drm_device_t* device, const amdf_media_frame_t* frame,
    const amdf_media_image_description_t* description,
    amdf_media_drm_frame_t* out_frame);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEDIA_DRM_IMAGE_H_
