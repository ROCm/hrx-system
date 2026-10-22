// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_D3D12_H264_H_
#define AMDF_SRC_MEDIA_D3D12_H264_H_

// DXVA consumes Windows scalar types and DEFINE_GUID without including them.
// clang-format off
#include <windows.h>
#include <dxva.h>
// clang-format on

#include "libamdf/src/media/d3d12/session.h"

namespace amdf::media::d3d12 {

// Checks the public parsed syntax without reading compressed or image bytes.
// Image identities and recording capacities are checked by command preparation.
amdf_status_t ValidateH264(const Session& session,
                           const amdf_media_h264_decode_picture_info_t& picture,
                           uint64_t compressed_byte_length);

// Lowers validated syntax into native off-host parsing arguments. Reference
// indices are dense within this recording; the current target follows them.
void LowerH264(const Session& session,
               const amdf_media_h264_decode_picture_info_t& picture,
               DXVA_PicParams_H264* native_picture,
               DXVA_Qmatrix_H264* native_matrices,
               DXVA_Slice_H264_Short* native_slices);

}  // namespace amdf::media::d3d12

#endif  // AMDF_SRC_MEDIA_D3D12_H264_H_
