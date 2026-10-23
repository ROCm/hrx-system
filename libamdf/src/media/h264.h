// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_H264_H_
#define AMDF_SRC_MEDIA_H264_H_

#include "amdf/media/h264.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validates public progressive eight-bit parsed syntax without reading the
// compressed payload. Native resource and slot admission belongs to
// preparation.
amdf_status_t amdf_media_h264_validate(
    const amdf_media_configuration_t* configuration,
    const amdf_media_h264_configuration_t* h264,
    const amdf_media_h264_decode_picture_info_t* picture,
    uint64_t compressed_byte_length);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEDIA_H264_H_
