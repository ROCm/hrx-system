// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_MEDIA_IMAGE_CODECS_H_
#define LOOM_SERVICE_MEDIA_IMAGE_CODECS_H_

#include "iree/codec/image/codec.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the immutable, build-selected provider table. An empty table means
// image input is unavailable. Table storage has process lifetime; there is no
// mutable global registration or per-request registry allocation.
iree_host_size_t loom_service_media_image_codecs(
    const iree_codec_image_codec_t* const** out_codecs);

// Returns NULL when this build has no provider for the format.
const iree_codec_image_codec_t* loom_service_media_image_codec(
    iree_codec_image_format_t format);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVICE_MEDIA_IMAGE_CODECS_H_
