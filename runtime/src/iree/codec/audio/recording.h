// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEC_AUDIO_RECORDING_H_
#define IREE_CODEC_AUDIO_RECORDING_H_

#include "iree/codec/audio/source.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_codec_audio_recording_t iree_codec_audio_recording_t;

// Decodes a complete, bounded recording on a CPU worker. Raw PCM and optional
// WAV/MP3/FLAC decoding share the dictation input contract. The resulting
// finite interleaved F32LE samples preserve the original channel count and
// sample rate; channel mixing, resampling, and padding belong to the consuming
// model. max_bytes bounds decoded sample storage, independently of compressed
// size. Failure retains nothing. The caller's source is borrowed only during
// this call.
IREE_API_EXPORT iree_status_t iree_codec_audio_recording_allocate(
    const iree_codec_audio_source_t* source, iree_host_size_t max_bytes,
    iree_allocator_t host_allocator,
    iree_codec_audio_recording_t** out_recording);

// Returns immutable PCM borrowed until recording_free.
IREE_API_EXPORT iree_codec_audio_source_t iree_codec_audio_recording_source(
    const iree_codec_audio_recording_t* recording);

// Releases decoded samples. NULL is accepted.
IREE_API_EXPORT void iree_codec_audio_recording_free(
    iree_codec_audio_recording_t* recording);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_CODEC_AUDIO_RECORDING_H_
