// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_DRM_MEMORY_H_
#define AMDF_SRC_MEDIA_DRM_MEMORY_H_

#include "libamdf/src/media/drm/buffer.h"
#include "libamdf/src/memory_resource.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct amdf_media_drm_memory_t {
  // Owned reference to the render file's canonical native buffer.
  amdf_media_drm_buffer_t* buffer;
  // Logical byte zero relative to the native buffer's full extent.
  uint64_t byte_offset;
} amdf_media_drm_memory_t;

// Resolves a validated exact consumer without a query, search, or allocation.
static inline amdf_media_drm_memory_t* amdf_media_drm_memory(
    amdf_memory_t* memory, uint32_t access_ordinal) {
  return (amdf_media_drm_memory_t*)memory
      ->accesses[memory->accesses[access_ordinal].native_owner_ordinal]
      .native;
}

amdf_status_t amdf_media_drm_memory_profile(
    amdf_device_t* device, uint32_t ordinal,
    amdf_memory_native_profile_t* out_profile);
amdf_status_t amdf_media_drm_memory_prepare(
    amdf_memory_t* memory, const amdf_memory_native_group_t* group,
    const amdf_memory_native_create_info_t* create_info,
    amdf_memory_info_t* out_info);
amdf_status_t amdf_media_drm_memory_prepare_import(
    amdf_memory_t* memory, uint32_t access_ordinal,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_import_info_t* import_info,
    const amdf_external_memory_t* external_memory,
    amdf_memory_info_t* out_info);

// Public operand boundary: validates exact native reach and logical bounds.
amdf_status_t amdf_media_drm_memory_range(
    amdf_media_drm_device_t* device, amdf_memory_t* memory,
    uint32_t access_ordinal, uint64_t byte_offset, uint64_t byte_length,
    amdf_media_drm_buffer_t** out_buffer, uint64_t* out_address);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEDIA_DRM_MEMORY_H_
