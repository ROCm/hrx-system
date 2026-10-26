// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_DRM_SESSION_H_
#define AMDF_SRC_MEDIA_DRM_SESSION_H_

#include "amdf/media/av1.h"
#include "amdf/media/h264.h"
#include "amdf/media/hevc.h"
#include "amdf/media/vp9.h"
#include "libamdf/src/media/drm/buffer.h"
#include "libamdf/src/media/encode.h"

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
    // VP9 profile and actual sample precision.
    amdf_media_vp9_configuration_t vp9;
    // AV1 profile and actual sample precision.
    amdf_media_av1_configuration_t av1;
  } codec;
  // Common immutable encode controls, valid only for encode sessions.
  amdf_media_encode_configuration_t encode;
  // Owned temporal-layer records referenced by encode after construction.
  amdf_media_encode_rate_control_layer_t
      rate_control_layers[AMDF_MEDIA_ENCODE_TEMPORAL_LAYER_STORAGE_CAPACITY];
  // Codec-specific immutable encode geometry selected by the codec field.
  union {
    // H264 frame-number and picture-order geometry.
    amdf_media_h264_encode_configuration_t h264;
    // HEVC block, level, tier, and picture-order geometry.
    amdf_media_hevc_encode_configuration_t hevc;
    // AV1 level, tier, and sequence-tool geometry.
    amdf_media_av1_encode_configuration_t av1;
  } encode_codec;
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
