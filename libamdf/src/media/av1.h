// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_AV1_H_
#define AMDF_SRC_MEDIA_AV1_H_

#include "amdf/media/av1.h"

#ifdef __cplusplus
extern "C" {
#endif

// Public syntax/array boundary; consumes no compressed or image bytes.
amdf_status_t amdf_media_av1_validate(
    const amdf_media_configuration_t* configuration,
    const amdf_media_av1_decode_picture_info_t* picture,
    uint64_t compressed_byte_length);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEDIA_AV1_H_
