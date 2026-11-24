// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/media/image_codecs.h"

static const iree_codec_image_codec_t* const codecs[] = {
#if LOOM_MEDIA_IMAGE_INPUT
    &iree_codec_image_codec_pam,
#endif
#if LOOM_MEDIA_IMAGE_CODEC_PNG
    &iree_codec_image_codec_png,
#endif
#if LOOM_MEDIA_IMAGE_CODEC_JPEG
    &iree_codec_image_codec_jpeg,
#endif
#if LOOM_MEDIA_IMAGE_CODEC_GIF
    &iree_codec_image_codec_gif,
#endif
#if LOOM_MEDIA_IMAGE_CODEC_WEBP
    &iree_codec_image_codec_webp,
#endif
    NULL,
};

iree_host_size_t loom_service_media_image_codecs(
    const iree_codec_image_codec_t* const** out_codecs) {
  *out_codecs = codecs;
  return IREE_ARRAYSIZE(codecs) - 1;
}

const iree_codec_image_codec_t* loom_service_media_image_codec(
    iree_codec_image_format_t format) {
  for (iree_host_size_t i = 0; i + 1 < IREE_ARRAYSIZE(codecs); ++i) {
    if (codecs[i]->format == format) {
      return codecs[i];
    }
  }
  return NULL;
}
