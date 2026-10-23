// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_DRM_QUEUE_H_
#define AMDF_SRC_MEDIA_DRM_QUEUE_H_

#include "libamdf/src/atomics.h"
#include "libamdf/src/media/drm/device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct amdf_media_drm_progress_t {
  // Native dependency protocol and publication cost.
  amdf_kernel_queue_native_progress_t base;
  // Borrowed file that owns the producer's timeline handle.
  int descriptor;
  // Timeline sync object shared through independent cold imports.
  uint32_t handle;
} amdf_media_drm_progress_t;

typedef struct amdf_media_drm_queue_t {
  // Core queue lifetime, cached information, and checked-retirement operations.
  amdf_kernel_queue_t base;
  // Native scheduling context, private to this ordered media queue.
  uint32_t context;
  // Native ring selected from the admitted engine's available-ring mask.
  uint32_t ring;
  // Native JPEG or unified VCN selector fixed at cold queue construction.
  uint32_t ip_type;
  // Cold producer identity used for timeline observation and dependency import.
  amdf_media_drm_progress_t progress;
  // Original-job sync objects, indexed by submission modulo explicit capacity.
  uint32_t* original_fences;
  // Initialized prefix of original_fences, including during construction.
  uint32_t original_fence_count;
  // Native dependency operand storage provisioned at queue creation.
  struct drm_amdgpu_cs_chunk_syncobj* waits;
  // Exclusive publication claim; concurrent publishers fail without mutation.
  amdf_atomic_uint32_t publishing;
  // Exclusive nonblocking retirement claim; other observers read cached state.
  amdf_atomic_uint32_t observing;
  // Greatest accepted native submission, published after its complete CS ioctl.
  amdf_atomic_uint64_t submitted;
  // Greatest prefix whose original jobs were checked successfully.
  amdf_atomic_uint64_t retired;
  // First original native execution failure, separate from observation errors.
  amdf_atomic_uint64_t terminal_status;
} amdf_media_drm_queue_t;

amdf_status_t AMDF_CALL amdf_media_drm_queue_create(
    amdf_device_t* device,
    const amdf_media_kernel_queue_create_info_t* create_info,
    amdf_kernel_queue_t** out_queue);
amdf_status_t AMDF_CALL amdf_media_drm_queue_submit(
    amdf_kernel_queue_t* queue,
    const amdf_media_kernel_queue_submission_info_t* submission_info,
    uint64_t* accepted_submission);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEDIA_DRM_QUEUE_H_
