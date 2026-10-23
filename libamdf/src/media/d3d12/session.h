// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_D3D12_SESSION_H_
#define AMDF_SRC_MEDIA_D3D12_SESSION_H_

#include "amdf/media/hevc.h"
#include "libamdf/src/media/d3d12/device.h"

namespace amdf::media::d3d12 {

struct Session {
  // Provider-neutral queue association and dispatch.
  amdf_media_session_t base;
  // Native decoder implementing the admitted codec envelope.
  ComPtr<ID3D12VideoDecoder> decoder;
  // Native codec state, with no hidden reconstructed-image pool.
  ComPtr<ID3D12VideoDecoderHeap> heap;
  // Immutable realized configuration and native state charges.
  amdf_media_session_info_t info;
  // Parsed syntax envelope selected by info.configuration.codec.
  union {
    // H264 syntax facts used during picture lowering.
    amdf_media_h264_configuration_t h264;
    // HEVC profile and component precision used during picture lowering.
    amdf_media_hevc_configuration_t hevc;
  };
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
