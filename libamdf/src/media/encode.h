// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_ENCODE_H_
#define AMDF_SRC_MEDIA_ENCODE_H_

#include "amdf/media/encode.h"

#ifdef __cplusplus
extern "C" {
#endif

// Provider-owned fixed storage capacity. Public callers discover the exact
// per-device maximum through amdf_media_encode_configuration_info_t.
#define AMDF_MEDIA_ENCODE_TEMPORAL_LAYER_STORAGE_CAPACITY 4u

// Applies native safety exclusions to a provider-reported temporal-layer
// limit. The returned count is never greater than native_limit.
uint32_t amdf_media_encode_restrict_temporal_layer_limit(
    const amdf_endpoint_info_t* endpoint_info, amdf_media_codec_t codec,
    uint32_t native_limit);

// Validates the shared public encode envelope before a native capability query.
// Codec-specific configuration chained from encode remains with its owner.
amdf_status_t amdf_media_encode_validate_configuration(
    const amdf_media_configuration_t* configuration,
    const amdf_media_encode_configuration_t* encode);

// Validates shared picture operands and the exact admitted compressed range.
// Provider-native image identity and codec syntax remain separate checks.
amdf_status_t amdf_media_encode_validate_picture(
    const amdf_media_configuration_info_t* requirements,
    const amdf_media_encode_configuration_t* configuration,
    const amdf_media_encode_picture_info_t* picture);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEDIA_ENCODE_H_
