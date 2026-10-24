// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_D3D12_ENCODE_H_
#define AMDF_SRC_MEDIA_D3D12_ENCODE_H_

#include "libamdf/src/media/d3d12/session.h"

namespace amdf::media::d3d12 {

// Complete transient native picture arguments. Every pointer in arguments
// refers to storage in this value and is consumed synchronously by EncodeFrame.
struct EncodePicture {
  // Shared sequence, picture, reference, and input arguments.
  D3D12_VIDEO_ENCODER_ENCODEFRAME_INPUT_ARGUMENTS arguments;
  // H264 picture controls selected when the session codec is H264.
  D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_H264 h264;
  // HEVC picture controls selected when the session codec is HEVC.
  D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC hevc;
  // AV1 picture controls selected when the session codec is AV1.
  D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_CODEC_DATA av1;
  // H264 native reference descriptors indexed by list0.
  D3D12_VIDEO_ENCODER_REFERENCE_PICTURE_DESCRIPTOR_H264 h264_references[8];
  // HEVC native reference descriptors indexed by list0.
  D3D12_VIDEO_ENCODER_REFERENCE_PICTURE_DESCRIPTOR_HEVC hevc_references[8];
  // Native list0 descriptor indices.
  UINT list0[8];
  // Provider-neutral frame and stable-slot bindings used for resource checks.
  amdf_media_reference_t references[8];
  // Number of initialized entries in references.
  uint32_t reference_count;
  // Whether the current reconstruction remains a later reference.
  bool used_as_reference;
};

// Validates and lowers one typed encode picture without touching image or
// compressed bytes. Temporary public arrays are copied into out_picture.
amdf_status_t PrepareEncode(const Session& session,
                            const amdf_media_encode_picture_info_t& encode,
                            uint32_t maximum_reference_count,
                            EncodePicture* out_picture);

// Interprets already-retired resolved metadata and publishes the explicit
// result-bearing protocol. Caller region capacity can return BUFFER_TOO_SMALL.
amdf_status_t ParseEncodeResult(const Session& session, const uint8_t* feedback,
                                uint64_t feedback_byte_length,
                                uint64_t compressed_byte_length,
                                amdf_media_encode_result_info_t* out_info);

// Validates and clears typed encode output when an accepted command retired
// without executing its media operation.
amdf_status_t ClearEncodeResult(const Session& session,
                                amdf_media_encode_result_info_t* out_info);

}  // namespace amdf::media::d3d12

#endif  // AMDF_SRC_MEDIA_D3D12_ENCODE_H_
