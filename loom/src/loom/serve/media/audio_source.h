// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_MEDIA_AUDIO_SOURCE_H_
#define LOOM_SERVE_MEDIA_AUDIO_SOURCE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_serve_audio_encoding_e {
  LOOM_SERVE_AUDIO_ENCODING_CONTAINER = 0,
  LOOM_SERVE_AUDIO_ENCODING_S16LE,
  LOOM_SERVE_AUDIO_ENCODING_F32LE,
} loom_serve_audio_encoding_t;

// One immutable recording. PCM geometry is explicit; encoded containers supply
// their own geometry when decoded. No filename participates in format
// selection.
typedef struct loom_serve_audio_source_t {
  // Borrowed input bytes, retained through the last accepted reader.
  iree_const_byte_span_t data;
  // Container bytes or one of the explicitly declared raw PCM encodings.
  loom_serve_audio_encoding_t encoding;
  // Raw frames per second; zero for a container.
  uint32_t sample_rate;
  // Interleaved samples per raw frame; zero for a container.
  uint32_t channels;
  // Complete raw frames; zero until decoding for a container.
  uint64_t frame_count;
} loom_serve_audio_source_t;

// Parses a nonempty uploaded recording without allocating or decoding it.
// audio/pcm requires positive rate and channels, with optional format=s16le
// (the default) or format=f32le. Unknown/duplicate PCM parameters and partial
// frames reject. Other media types select container sniffing by the codec.
iree_status_t loom_serve_audio_source_parse(
    iree_string_view_t content_type, iree_const_byte_span_t data,
    loom_serve_audio_source_t* out_source);

// Computes ceil(input_frames * output_rate / input_rate) without overflowing
// intermediate multiplication. Both rates are positive producer-established
// geometry; an unrepresentable duration is an external input failure.
iree_status_t loom_serve_audio_frame_count(uint64_t input_frames,
                                           uint32_t input_rate,
                                           uint32_t output_rate,
                                           uint64_t* out_frame_count);

// Converts a complete raw frame range into disjoint native F32 storage on a
// CPU worker. The caller supplies a range within source.frame_count. F32 input
// must be finite and normalized to [-1, 1]; failure invalidates this output.
// No channel mixing or resampling occurs here; those are consumer policies.
iree_status_t loom_serve_audio_source_read(
    const loom_serve_audio_source_t* source, uint64_t frame_offset,
    iree_host_size_t frame_count, float* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_MEDIA_AUDIO_SOURCE_H_
