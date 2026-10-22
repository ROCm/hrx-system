// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_D3D12_IMAGE_H_
#define AMDF_SRC_MEDIA_D3D12_IMAGE_H_

#include "amdf/media/gfx11.h"
#include "libamdf/src/media/d3d12/device.h"

namespace amdf::media::d3d12 {

struct Image {
  // Provider-neutral identity and device dispatch.
  amdf_media_image_t base;
  // Caller-owned core backing borrowed until image release.
  amdf_memory_t* memory;
  // Exact constructing media consumer in the backing's access list.
  uint32_t media_access_ordinal;
  // Placement of the native identity relative to logical backing byte zero.
  uint64_t byte_offset;
  // Immutable shape and exact native placement requirements.
  amdf_media_image_requirements_t requirements;
  // Native placed identity, which allocates no image payload.
  ComPtr<ID3D12Resource> resource;
  // Directly indexed consumer layouts in this owner's trailing allocation.
  // NULL entries were not requested; initialized entries contain every layer.
  amdf_media_gfx11_image_layout_t** layouts;
};

amdf_status_t AMDF_CALL ImageCreate(
    amdf_device_t* device, const amdf_media_image_create_info_t* create_info,
    amdf_media_image_t** out_image);
amdf_status_t AMDF_CALL ImageQueryInfo(amdf_media_image_t* image,
                                       uint32_t access_ordinal,
                                       const amdf_media_image_query_t* query,
                                       amdf_media_image_info_t* out_info);
void AMDF_CALL ImageDestroy(amdf_media_image_t* image);

// Validates one external frame against its explicit native identity and role.
// The returned image is borrowed from the frame; no lookup or import occurs.
amdf_status_t ValidateFrame(Device* device, const amdf_media_frame_t* frame,
                            amdf_media_image_roles_t role, Image** out_image);

}  // namespace amdf::media::d3d12

#endif  // AMDF_SRC_MEDIA_D3D12_IMAGE_H_
