// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/media/audio_decoder.h"

iree_status_t loom_serve_audio_decoder_allocate(
    iree_const_byte_span_t encoded, uint32_t sample_rate, uint32_t channels,
    iree_allocator_t host_allocator, loom_serve_audio_decoder_t** out_decoder) {
  *out_decoder = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "encoded audio input is disabled in this build");
}

uint32_t loom_serve_audio_decoder_sample_rate(
    const loom_serve_audio_decoder_t* decoder) {
  return 0;
}

uint32_t loom_serve_audio_decoder_channels(
    const loom_serve_audio_decoder_t* decoder) {
  return 0;
}

iree_status_t loom_serve_audio_decoder_read(loom_serve_audio_decoder_t* decoder,
                                            iree_host_size_t frame_capacity,
                                            float* frames,
                                            iree_host_size_t* out_frame_count,
                                            bool* out_complete) {
  *out_frame_count = 0;
  *out_complete = false;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "encoded audio input is disabled in this build");
}

void loom_serve_audio_decoder_free(loom_serve_audio_decoder_t* decoder) {}
