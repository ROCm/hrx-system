// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_VP9_H_
#define AMDF_SRC_MEDIA_VP9_H_

#include "amdf/media/vp9.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validates caller syntax, partition bounds, and logical reference indices.
// Frame identity, native storage, and stable slot admission follow separately.
amdf_status_t amdf_media_vp9_validate(
    const amdf_media_configuration_t* configuration,
    const amdf_media_vp9_decode_picture_info_t* picture,
    uint64_t compressed_byte_length);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEDIA_VP9_H_
