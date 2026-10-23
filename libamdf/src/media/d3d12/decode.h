// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_D3D12_DECODE_H_
#define AMDF_SRC_MEDIA_D3D12_DECODE_H_

#include "libamdf/src/media/d3d12/h264.h"
#include "libamdf/src/media/d3d12/hevc.h"

namespace amdf::media::d3d12 {

// Fixed per-slice provisioning shared by the implemented native schemas.
constexpr size_t kDecodeSliceByteLength = sizeof(DXVA_Slice_H264_Short) >
                                                  sizeof(DXVA_Slice_HEVC_Short)
                                              ? sizeof(DXVA_Slice_H264_Short)
                                              : sizeof(DXVA_Slice_HEVC_Short);

// Preparation-local native arguments. No state survives recording and no
// native pointers escape the preparation call.
struct DecodePicture {
  // Codec-selected picture argument, consumed by DecodeFrame.
  union {
    // Off-host H264 parsing arguments.
    DXVA_PicParams_H264 h264;
    // Off-host HEVC parsing arguments.
    DXVA_PicParams_HEVC hevc;
  } parameters;
  // Codec-selected scaling data in native scan order.
  union {
    // H264 inverse quantization matrices.
    DXVA_Qmatrix_H264 h264;
    // HEVC inverse quantization matrices, present only when enabled.
    DXVA_Qmatrix_HEVC hevc;
  } matrices;
  // Native argument views into this owner and the command's slice storage.
  D3D12_VIDEO_DECODE_INPUT_STREAM_ARGUMENTS arguments;
  // Validated number of borrowed reconstructed images.
  uint32_t reference_count;
  // Dense caller list, preserving each picture's stable native slot.
  amdf_media_reference_t references[16];
};

// Public syntax boundary followed by native lowering. Slice storage was
// provisioned cold for maximum_slice_count; this call neither allocates nor
// reads compressed/image memory. Native identity validation follows in command
// preparation before any recording owner is reset.
amdf_status_t PrepareDecode(const Session& session, const void* picture,
                            uint64_t compressed_byte_length,
                            uint32_t current_reference_slot,
                            uint32_t maximum_reference_count,
                            uint32_t maximum_slice_count, void* slices,
                            DecodePicture* out_picture);

}  // namespace amdf::media::d3d12

#endif  // AMDF_SRC_MEDIA_D3D12_DECODE_H_
