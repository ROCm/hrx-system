// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_DRM_VCN_ENCODE_H_
#define AMDF_SRC_MEDIA_DRM_VCN_ENCODE_H_

#include "libamdf/src/media/drm/image.h"
#include "libamdf/src/media/drm/session.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // Complete firmware feedback record written for one encoded picture.
  AMDF_MEDIA_DRM_VCN_ENCODE_FEEDBACK_BYTES = 40,
};

// Resolved application-owned storage consumed by one encode packet. Pixel
// layers precede codec state in reconstruction_address; every byte remains
// owned and retained by the application command.
typedef struct amdf_media_drm_vcn_encode_resources_t {
  // Source luma and chroma planes.
  amdf_media_drm_frame_t input;
  // Base address of the complete reconstruction image array and codec tail.
  uint64_t reconstruction_address;
  // Byte distance between adjacent reconstruction pixel layers.
  uint32_t reconstruction_layer_byte_length;
  // Luma byte extent within each reconstruction layer.
  uint32_t reconstruction_luma_byte_length;
  // Common byte pitch of reconstruction luma and interleaved chroma planes.
  uint32_t reconstruction_row_pitch;
  // Number of application-owned reconstruction slots in the array.
  uint32_t reconstruction_slot_count;
  // Compressed output start.
  uint64_t compressed_address;
  // Complete writable compressed output capacity.
  uint32_t compressed_byte_length;
  // Firmware feedback-record start.
  uint64_t feedback_address;
} amdf_media_drm_vcn_encode_resources_t;

// Infallible packet emission after public configuration and operand admission.
// The returned extent is the complete unified encode-engine IB.
uint32_t amdf_media_drm_vcn_encode_control(
    amdf_media_drm_buffer_t* storage, const amdf_media_drm_session_t* session,
    amdf_media_command_operation_t operation);
uint32_t amdf_media_drm_vcn_encode_h264(
    amdf_media_drm_buffer_t* storage, const amdf_media_drm_session_t* session,
    const amdf_media_drm_vcn_encode_resources_t* resources,
    uint32_t reconstruction_slot,
    const amdf_media_h264_encode_picture_info_t* picture);
uint32_t amdf_media_drm_vcn_encode_hevc(
    amdf_media_drm_buffer_t* storage, const amdf_media_drm_session_t* session,
    const amdf_media_drm_vcn_encode_resources_t* resources,
    uint32_t reconstruction_slot,
    const amdf_media_hevc_encode_picture_info_t* picture);
uint32_t amdf_media_drm_vcn_encode_av1(
    amdf_media_drm_buffer_t* storage, const amdf_media_drm_session_t* session,
    const amdf_media_drm_vcn_encode_resources_t* resources,
    uint32_t reconstruction_slot,
    const amdf_media_av1_encode_picture_info_t* picture);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEDIA_DRM_VCN_ENCODE_H_
