// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_MEDIA_AUDIO_RECORDING_H_
#define LOOM_SERVE_MEDIA_AUDIO_RECORDING_H_

#include "loom/serve/media/audio_source.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_audio_recording_t loom_serve_audio_recording_t;

// Decodes a complete, bounded recording on a CPU worker. Raw PCM and optional
// WAV/MP3/FLAC decoding share the dictation input contract. The resulting
// finite interleaved F32LE samples preserve the original channel count and
// sample rate; channel mixing, resampling, and padding belong to the consuming
// model. max_bytes bounds decoded sample storage, independently of compressed
// size. Failure retains nothing. The caller's source is borrowed only during
// this call.
iree_status_t loom_serve_audio_recording_allocate(
    const loom_serve_audio_source_t* source, iree_host_size_t max_bytes,
    iree_allocator_t host_allocator,
    loom_serve_audio_recording_t** out_recording);

// Returns immutable PCM borrowed until recording_free.
loom_serve_audio_source_t loom_serve_audio_recording_source(
    const loom_serve_audio_recording_t* recording);

// Releases decoded samples. NULL is accepted.
void loom_serve_audio_recording_free(loom_serve_audio_recording_t* recording);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_MEDIA_AUDIO_RECORDING_H_
