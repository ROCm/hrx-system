// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_D3D12_SESSION_H_
#define AMDF_SRC_MEDIA_D3D12_SESSION_H_

#include "amdf/media/av1.h"
#include "amdf/media/h264.h"
#include "amdf/media/hevc.h"
#include "amdf/media/vp9.h"
#include "libamdf/src/media/d3d12/configuration.h"
#include "libamdf/src/media/d3d12/device.h"
#include "libamdf/src/media/encode.h"

namespace amdf::media::d3d12 {

struct Session {
  // Provider-neutral queue association and dispatch.
  amdf_media_session_t base;
  // Native decoder implementing an admitted decode envelope.
  ComPtr<ID3D12VideoDecoder> decoder;
  // Native decoder state, with no hidden reconstructed-image pool.
  ComPtr<ID3D12VideoDecoderHeap> decoder_heap;
  // Native encoder implementing an admitted encode envelope.
  ComPtr<ID3D12VideoEncoder> encoder;
  // Native encoder state, with no hidden reconstruction pool.
  ComPtr<ID3D12VideoEncoderHeap> encoder_heap;
  // Immutable realized configuration and native state charges.
  amdf_media_session_info_t info;
  // Parsed syntax envelope selected by info.configuration.codec.
  union {
    // H264 syntax facts used during picture lowering.
    amdf_media_h264_configuration_t h264;
    // HEVC profile and component precision used during picture lowering.
    amdf_media_hevc_configuration_t hevc;
    // VP9 profile and component precision used during picture lowering.
    amdf_media_vp9_configuration_t vp9;
    // AV1 profile and actual sample precision.
    amdf_media_av1_configuration_t av1;
  };
  // Common immutable encode controls, valid only for encode sessions.
  amdf_media_encode_configuration_t encode;
  // Owned temporal-layer records referenced by encode after construction.
  amdf_media_encode_rate_control_layer_t
      rate_control_layers[AMDF_MEDIA_ENCODE_TEMPORAL_LAYER_STORAGE_CAPACITY];
  // Codec-specific immutable encode geometry selected by the codec field.
  union {
    // H264 frame-number and picture-order geometry.
    amdf_media_h264_encode_configuration_t h264_encode;
    // HEVC block, level, tier, and picture-order geometry.
    amdf_media_hevc_encode_configuration_t hevc_encode;
    // AV1 level, tier, and sequence-tool geometry.
    amdf_media_av1_encode_configuration_t av1_encode;
  };
  // Self-contained native encoder descriptors retained for every recording.
  EncodeConfiguration native_encode;
  // Retained placement and buffer requirements from native admission.
  amdf_media_configuration_info_t requirements;
  // Recording owners that still borrow this native session.
  amdf_child_tracker_t commands;
};

amdf_status_t AMDF_CALL
SessionCreate(amdf_kernel_queue_t* queue,
              const amdf_media_session_create_info_t* create_info,
              amdf_media_session_t** out_session);
amdf_status_t AMDF_CALL SessionQueryInfo(amdf_media_session_t* session,
                                         amdf_media_session_info_t* out_info);
amdf_status_t AMDF_CALL SessionDestroy(amdf_media_session_t* session);

}  // namespace amdf::media::d3d12

#endif  // AMDF_SRC_MEDIA_D3D12_SESSION_H_
