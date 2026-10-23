// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_DRM_DEVICE_H_
#define AMDF_SRC_MEDIA_DRM_DEVICE_H_

#include <drm/amdgpu_drm.h>
#include <pthread.h>

#include "libamdf/src/media/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct amdf_media_drm_buffer_t amdf_media_drm_buffer_t;

typedef struct amdf_media_drm_engine_t {
  // Kernel hardware-IP selector for publication on this family.
  uint32_t ip_type;
  // Immutable native generation, rings, and command alignment.
  struct drm_amdgpu_info_hw_ip information;
} amdf_media_drm_engine_t;

typedef struct amdf_media_drm_device_t {
  // Independently activated core media device and provider dispatch.
  amdf_media_device_t base;
  // Owned render file, including its private GEM table and GPU VM.
  int descriptor;
  // Native page granularity used for allocations and VM mappings.
  uint64_t page_size;
  // First usable byte address in the native VM.
  uint64_t address_begin;
  // Exclusive end of the native VM's admitted low address range.
  uint64_t address_end;
  // Serializes cold handle acquisition, native ownership, and VA placement.
  pthread_mutex_t memory_mutex;
  // Weak VA-ordered index; resource attachments own every entry's references.
  amdf_media_drm_buffer_t* buffers;
  // First native ownership failure; a compromised handle table cannot import
  // another alias whose ownership would no longer be represented.
  amdf_status_t memory_status;
  // Native codec envelopes; advertising a schema also requires its lowering.
  struct drm_amdgpu_info_video_caps decode_limits;
  // Native engines indexed by the corresponding public family ordinal.
  amdf_media_drm_engine_t engines[2];
  // Immutable execution families exposed by this activation.
  amdf_queue_family_info_t families[2];
} amdf_media_drm_device_t;

// Copies native query data into a caller-owned value before publication.
amdf_status_t amdf_media_drm_query(int descriptor, uint32_t query,
                                   uint32_t parameter, void* data,
                                   uint32_t byte_length);

// Validates a public configuration and answers exact native support.
amdf_status_t amdf_media_drm_configuration(
    amdf_media_drm_device_t* device, uint32_t family_ordinal,
    const amdf_media_configuration_t* configuration, const void* next,
    amdf_media_configuration_info_t* out_info);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEDIA_DRM_DEVICE_H_
