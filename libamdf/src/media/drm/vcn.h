// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_DRM_VCN_H_
#define AMDF_SRC_MEDIA_DRM_VCN_H_

#include "libamdf/src/media/drm/image.h"
#include "libamdf/src/media/drm/session.h"

#ifdef __cplusplus
extern "C" {
#endif

// Provisioned command-buffer offsets. AVC/HEVC use
// scaling storage; VP9's smaller message leaves room for its larger probability
// payload at an earlier offset. AV1 reserves its larger message and grain
// tables explicitly. The layouts are mutually exclusive.
enum {
  AMDF_MEDIA_DRM_VCN_MESSAGE_OFFSET = 256,
  AMDF_MEDIA_DRM_VCN_SCALING_OFFSET = 2048,
  AMDF_MEDIA_DRM_VCN_PROBABILITY_OFFSET = 1024,
  AMDF_MEDIA_DRM_VCN_AV1_AUXILIARY_OFFSET = 4352,
  AMDF_MEDIA_DRM_VCN_AV1_AUXILIARY_BYTES = 20736,
};

// Infallible emission after public operand admission. The returned extent is
// the unified-engine IB, separate from the embedded firmware message.
uint32_t amdf_media_drm_vcn_control(amdf_media_drm_buffer_t* storage,
                                    const amdf_media_drm_session_t* session,
                                    amdf_media_command_operation_t operation);
uint32_t amdf_media_drm_vcn_h264(
    amdf_media_drm_buffer_t* storage, const amdf_media_drm_session_t* session,
    uint64_t compressed_address, uint32_t compressed_byte_length,
    const amdf_media_drm_frame_t* output, const amdf_media_drm_frame_t* current,
    const amdf_media_drm_frame_t* references, uint32_t current_slot,
    const amdf_media_h264_decode_picture_info_t* picture);
uint32_t amdf_media_drm_vcn_hevc(
    amdf_media_drm_buffer_t* storage, const amdf_media_drm_session_t* session,
    uint64_t compressed_address, uint32_t compressed_byte_length,
    const amdf_media_drm_frame_t* output, const amdf_media_drm_frame_t* current,
    const amdf_media_drm_frame_t* references, uint32_t current_slot,
    const amdf_media_hevc_decode_picture_info_t* picture);
uint32_t amdf_media_drm_vcn_vp9(
    amdf_media_drm_buffer_t* storage, const amdf_media_drm_session_t* session,
    uint64_t compressed_address, uint32_t compressed_byte_length,
    const amdf_media_drm_frame_t* output, const amdf_media_drm_frame_t* current,
    const amdf_media_drm_frame_t* references, uint32_t current_slot,
    const amdf_media_vp9_decode_picture_info_t* picture);

uint32_t amdf_media_drm_vcn_av1(
    amdf_media_drm_buffer_t* storage, const amdf_media_drm_session_t* session,
    uint64_t compressed_address, uint32_t compressed_byte_length,
    const amdf_media_drm_frame_t* output, const amdf_media_drm_frame_t* current,
    const amdf_media_drm_frame_t* references, uint32_t current_slot,
    const amdf_media_av1_decode_picture_info_t* picture);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEDIA_DRM_VCN_H_
