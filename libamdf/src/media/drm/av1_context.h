// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_DRM_AV1_CONTEXT_H_
#define AMDF_SRC_MEDIA_DRM_AV1_CONTEXT_H_

#include "amdf/base.h"

#ifdef __cplusplus
extern "C" {
#endif

// VCN4's fixed codec-state charge, independent of picture extent/precision.
enum { AMDF_MEDIA_DRM_AV1_CONTEXT_BYTES = 36389888 };

// VCN4's immutable encoder probability table. This is session-private codec
// state and is independent of picture extent, precision, and reference count.
enum { AMDF_MEDIA_DRM_AV1_ENCODE_CDF_BYTES = 81920 };

// Installs the four standard CDF sets into a zeroed context. All remaining
// native entropy, motion, restoration, and grain state begins zeroed.
void amdf_media_drm_av1_context_initialize(void* context);

// Installs VCN4's four encoder CDF sets. The encoder and decoder consume the
// same normative distributions, but firmware requires reserved encoder words
// to carry its 0xCDCD sentinel instead of zero.
void amdf_media_drm_av1_encode_cdf_initialize(void* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEDIA_DRM_AV1_CONTEXT_H_
