// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_DRM_AV1_GRAIN_H_
#define AMDF_SRC_MEDIA_DRM_AV1_GRAIN_H_

#include "amdf/media/av1.h"
#include "libamdf/src/media/drm/device.h"

// Native grain parameter and auxiliary-table extents.
enum {
  AMDF_MEDIA_DRM_AV1_GRAIN_PARAMETER_BYTES = 166,
  AMDF_MEDIA_DRM_AV1_GRAIN_AUXILIARY_BYTES = 20480,
};

#ifdef __cplusplus
extern "C" {
#endif

// Writes generation-selected native grain parameters and tables into zeroed,
// provisioned recording storage. The public boundary has validated point
// ordering, counts, coefficients, and bit precision.
void amdf_media_drm_av1_grain(amdf_media_drm_av1_protocol_t protocol,
                              const amdf_media_av1_film_grain_t* grain,
                              uint32_t sample_bit_depth, void* parameters,
                              void* auxiliary);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEDIA_DRM_AV1_GRAIN_H_
