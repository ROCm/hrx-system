// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_DRM_SESSION_H_
#define AMDF_SRC_MEDIA_DRM_SESSION_H_

#include "amdf/media/h264.h"
#include "amdf/media/hevc.h"
#include "libamdf/src/media/drm/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct amdf_media_drm_session_t {
  // Borrowed queue and provider identity.
  amdf_media_session_t base;
  // Exact admitted configuration and resource geometry.
  amdf_media_configuration_info_t requirements;
  // Recordings borrowing this codec state, independent of submission status.
  amdf_child_tracker_t commands;
  // Parsed envelope selected by requirements.configuration.codec; no borrowed
  // extension-chain pointers survive construction.
  union {
    // H264 profile and level.
    amdf_media_h264_configuration_t h264;
    // HEVC profile and actual sample precision.
    amdf_media_hevc_configuration_t hevc;
  } codec;
  // Owned firmware session and collocated context; no pixel storage.
  amdf_media_drm_buffer_t* storage;
  // Bytes of codec-specific context following the 128 KiB session prefix.
  uint32_t context_byte_length;
  // Opaque firmware stream identity selected during cold construction.
  uint32_t stream_handle;
  // Last accepted lifecycle transition, serialized by the owning queue.
  amdf_media_command_operation_t operation;
} amdf_media_drm_session_t;

amdf_status_t AMDF_CALL amdf_media_drm_session_create(
    amdf_kernel_queue_t* queue,
    const amdf_media_session_create_info_t* create_info,
    amdf_media_session_t** out_session);
amdf_status_t AMDF_CALL amdf_media_drm_session_query_info(
    amdf_media_session_t* session, amdf_media_session_info_t* out_info);
amdf_status_t AMDF_CALL
amdf_media_drm_session_destroy(amdf_media_session_t* session);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEDIA_DRM_SESSION_H_
