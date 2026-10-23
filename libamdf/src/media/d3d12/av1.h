// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_D3D12_AV1_H_
#define AMDF_SRC_MEDIA_D3D12_AV1_H_

// DXVA consumes Windows scalar types without including them.
// clang-format off
#include <windows.h>
#include <dxva.h>
// clang-format on

#include "amdf/media/av1.h"
#include "libamdf/src/media/d3d12/session.h"

namespace amdf::media::d3d12 {

void LowerAv1(const Session& session, uint32_t current_reference_slot,
              const amdf_media_av1_decode_picture_info_t& picture,
              DXVA_PicParams_AV1* native, DXVA_Tile_AV1* tiles);

}  // namespace amdf::media::d3d12

#endif  // AMDF_SRC_MEDIA_D3D12_AV1_H_
