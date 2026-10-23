// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_D3D12_HEVC_H_
#define AMDF_SRC_MEDIA_D3D12_HEVC_H_

// DXVA consumes Windows scalar types and DEFINE_GUID without including them.
// clang-format off
#include <windows.h>
#include <dxva.h>
// clang-format on

#include "libamdf/src/media/d3d12/session.h"

namespace amdf::media::d3d12 {

// Validates caller-authored syntax and reference sets without payload access.
amdf_status_t ValidateHevc(const Session& session,
                           const amdf_media_hevc_decode_picture_info_t& picture,
                           uint64_t compressed_byte_length);

// Consumes validated syntax and the application-owned stable surface slots.
void LowerHevc(const Session& session, uint32_t current_reference_slot,
               const amdf_media_hevc_decode_picture_info_t& picture,
               DXVA_PicParams_HEVC* native_picture,
               DXVA_Qmatrix_HEVC* native_matrices,
               DXVA_Slice_HEVC_Short* native_slices);

}  // namespace amdf::media::d3d12

#endif  // AMDF_SRC_MEDIA_D3D12_HEVC_H_
