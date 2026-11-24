// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEC_AUDIO_DECODER_H_
#define IREE_CODEC_AUDIO_DECODER_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_codec_audio_decoder_t iree_codec_audio_decoder_t;

// Opens an in-memory WAV, MP3, or FLAC stream, converting to interleaved F32
// at the requested rate/channel count. A zero rate or channel count preserves
// that component of the original geometry, allowing model-owned conversion.
// The encoded bytes remain borrowed until
// free. Container metadata selects the input format; filenames are never
// interpreted. Initialization and reads belong on a CPU worker, not the serving
// owner. No file access, device IO, or private workers are used. Each decoder
// owns independent codec and filtered resampling history.
IREE_API_EXPORT iree_status_t iree_codec_audio_decoder_allocate(
    iree_const_byte_span_t encoded, uint32_t sample_rate, uint32_t channels,
    iree_allocator_t host_allocator, iree_codec_audio_decoder_t** out_decoder);

// Returns the resolved geometry, including original values requested with zero.
IREE_API_EXPORT uint32_t
iree_codec_audio_decoder_sample_rate(const iree_codec_audio_decoder_t* decoder);
IREE_API_EXPORT uint32_t
iree_codec_audio_decoder_channels(const iree_codec_audio_decoder_t* decoder);

// Produces up to frame_capacity whole frames into caller-owned native F32
// storage of frame_capacity * channels floats. Zero capacity does no work.
// EOF is explicit and may accompany a final nonempty block. Output is not
// padded to a block boundary. A failure invalidates this read's output and
// terminates decoding; the caller frees the decoder after its task retires.
IREE_API_EXPORT iree_status_t iree_codec_audio_decoder_read(
    iree_codec_audio_decoder_t* decoder, iree_host_size_t frame_capacity,
    float* frames, iree_host_size_t* out_frame_count, bool* out_complete);

// Releases codec/conversion storage, never the borrowed encoded bytes. All
// reads have retired before this call. NULL is accepted.
IREE_API_EXPORT void iree_codec_audio_decoder_free(
    iree_codec_audio_decoder_t* decoder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_CODEC_AUDIO_DECODER_H_
