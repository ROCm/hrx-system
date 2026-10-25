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
  // VCN 5's 256-byte displayable reference-surface swizzle.
  AMDF_MEDIA_DRM_VCN_REFERENCE_SWIZZLE_256B_D = 1,
};

typedef struct amdf_media_drm_vcn_protocol_info_t {
  // Unified firmware command family selected for this IP generation.
  amdf_media_drm_command_protocol_t command_protocol;
  // AV1 private-context and film-grain representation.
  amdf_media_drm_av1_protocol_t av1_protocol;
  // Address-library selector embedded in decode-target and reference fields.
  uint32_t decode_array_mode;
  // Firmware swizzle selector for decoder-private reference storage.
  uint32_t reference_swizzle_mode;
  // Required luma-height alignment for application-owned reference images.
  uint32_t reference_height_alignment;
} amdf_media_drm_vcn_protocol_info_t;

// Selects immutable unified-decode facts for one native discovery version.
// Returns false without producing output when the version family is unknown.
bool amdf_media_drm_vcn_protocol(
    uint32_t ip_discovery_version,
    amdf_media_drm_vcn_protocol_info_t* out_protocol);

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
