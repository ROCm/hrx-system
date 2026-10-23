// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_DRM_JPEG_H_
#define AMDF_SRC_MEDIA_DRM_JPEG_H_

#include "libamdf/src/media/drm/image.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits the qualified VCN 2/3/4 JPEG register protocol into at least 128
// DWORDs. Operands are validated at preparation; emission neither parses input
// nor reads output pixels. Returns the complete 64-byte-aligned IB byte extent.
uint32_t amdf_media_drm_jpeg_emit(uint32_t* words, uint64_t compressed_address,
                                  uint32_t compressed_byte_length,
                                  const amdf_media_drm_frame_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEDIA_DRM_JPEG_H_
