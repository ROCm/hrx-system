// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_DRM_VP9_CONTEXT_H_
#define AMDF_SRC_MEDIA_DRM_VP9_CONTEXT_H_

#include "amdf/base.h"

// Firmware context charge for VCN 2 and newer, with verified 8/10-bit
// precision.
uint32_t amdf_media_drm_vp9_context_byte_length(uint32_t sample_bit_depth);

// Installs standard defaults into zeroed caller-provisioned context storage.
void amdf_media_drm_vp9_context_initialize(void* context);

#endif  // AMDF_SRC_MEDIA_DRM_VP9_CONTEXT_H_
