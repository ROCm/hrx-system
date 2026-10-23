// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_DRM_BUFFER_H_
#define AMDF_SRC_MEDIA_DRM_BUFFER_H_

#include "libamdf/src/media/drm/device.h"

#ifdef __cplusplus
extern "C" {
#endif

struct amdf_media_drm_buffer_t {
  // Borrowed device whose GEM table and VM contain this native owner.
  amdf_media_drm_device_t* device;
  // Next weakly indexed native owner in ascending VA order.
  amdf_media_drm_buffer_t* next;
  // Native handle shared by all PRIME aliases in this render file.
  uint32_t handle;
  // Number of resource attachments owning this handle, under memory_mutex.
  uint32_t reference_count;
  // Result of native preparation; concurrent aliases cannot publish a partial
  // attachment while its original owner is rolling back.
  amdf_status_t preparation_status;
  // Complete physical extent, independent of an import's logical subrange.
  uint64_t byte_length;
  // VM reservation start; zero until a range has been reserved.
  uint64_t address;
  // True once the native VM mapping has been established.
  bool mapped;
  // Owned full-buffer host mapping, or NULL when no host view was requested.
  void* host_pointer;
};

// Acquires state into a live resource's initially NULL binding. Native partial
// acquisition remains in that binding for release/abandon on every return.
// descriptor < 0 creates a fresh GTT buffer; otherwise imports borrowed
// DMA-BUF. Only fresh allocations expose a host mapping through this operation.
amdf_status_t amdf_media_drm_buffer_prepare(amdf_media_drm_device_t* device,
                                            int descriptor,
                                            uint64_t byte_length,
                                            bool host_visible,
                                            amdf_media_drm_buffer_t** binding);

// Releases one attachment. Success consumes and clears binding; failure leaves
// the exact remaining native state for terminal abandon by its live owner.
amdf_status_t amdf_media_drm_buffer_release(amdf_media_drm_buffer_t** binding);

// Consumes metadata while preserving native payload after terminal failure.
// The private handle table is quarantined from further memory acquisitions.
void amdf_media_drm_buffer_abandon(amdf_media_drm_buffer_t** binding);

// Exports one independent DMA-BUF descriptor; failure leaves output unchanged.
amdf_status_t amdf_media_drm_buffer_export(
    const amdf_media_drm_buffer_t* buffer, int* out_descriptor);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEDIA_DRM_BUFFER_H_
