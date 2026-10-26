// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_HEVC_H_
#define AMDF_SRC_MEDIA_HEVC_H_

#include "amdf/media/hevc.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validates public progressive Main/Main10 parsed syntax without reading the
// compressed payload. Native resource and slot admission belongs to
// preparation.
amdf_status_t amdf_media_hevc_validate(
    const amdf_media_configuration_t* configuration,
    const amdf_media_hevc_configuration_t* hevc,
    const amdf_media_hevc_decode_picture_info_t* picture,
    uint64_t compressed_byte_length);

// Validates the Main/Main10 IP-only encode envelope and extension chain.
amdf_status_t amdf_media_hevc_validate_encode_configuration(
    const amdf_media_configuration_t* configuration,
    const amdf_media_hevc_configuration_t* hevc,
    const amdf_media_encode_configuration_t* encode,
    const amdf_media_hevc_encode_configuration_t* hevc_encode);

// Validates HEVC picture ordering and references before native recording.
amdf_status_t amdf_media_hevc_validate_encode_picture(
    const amdf_media_configuration_info_t* requirements,
    const amdf_media_encode_configuration_t* configuration,
    const amdf_media_encode_picture_info_t* picture,
    const amdf_media_hevc_encode_picture_info_t* hevc_picture);

// Converts a verified raster coefficient grid of width four or eight into the
// diagonal scan shared by the VCN and DXVA HEVC interfaces. No allocation.
void amdf_media_hevc_scan_scaling_list(const uint8_t* source, uint32_t width,
                                       uint8_t* target);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEDIA_HEVC_H_
