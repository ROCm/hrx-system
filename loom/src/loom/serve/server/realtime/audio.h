// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_AUDIO_H_
#define LOOM_SERVE_SERVER_REALTIME_AUDIO_H_

#include "loom/serve/media/audio_resampler.h"
#include "loom/serve/service/speech/generator.h"

#ifdef __cplusplus
extern "C" {
#endif

// Worker-owned conversion into the standard mono 24 kHz PCM16 wire clock.
// Coefficients survive response boundaries; each begin creates a fresh cursor.
typedef struct loom_serve_realtime_audio_t {
  // Allocator for filter coefficients and the bounded convolution window.
  iree_allocator_t host_allocator;
  // Immutable source geometry of this registered speech deployment.
  loom_serve_speech_format_t format;
  // Immutable source-rate to wire-rate coefficients, shared across responses.
  loom_serve_audio_resampler_filter_t filter;
  // Successful coefficient construction, independent of partial failure state.
  bool filter_ready;
  // One response's independent mono history, replaced only after CPU release.
  loom_serve_audio_resampler_t resampler;
  // Frames already read from the currently borrowed native waveform block.
  iree_host_size_t source_offset;
  // Wire frames generated for this response, independent of network admission.
  uint64_t frames;
} loom_serve_realtime_audio_t;

// Worker-only start/reset. An initialized value retains coefficients; format
// belongs to the bound registration and does not change between responses.
iree_status_t loom_serve_realtime_audio_begin(
    loom_serve_speech_format_t format, iree_allocator_t host_allocator,
    loom_serve_realtime_audio_t* audio);

// Converts at most 4096 source and output frames, appending base64 PCM to
// output. The owner keeps block alive until out_consumed. A complete source
// block flushes the response's semantic EOF. Source channels are averaged
// before conversion; finite samples saturate to normalized PCM range before
// filtering. No packet boundary resets history, pads audio, or changes the
// output clock.
iree_status_t loom_serve_realtime_audio_convert(
    loom_serve_realtime_audio_t* audio, const loom_serve_speech_block_t* block,
    iree_string_builder_t* output, bool* out_consumed);
void loom_serve_realtime_audio_deinitialize(loom_serve_realtime_audio_t* audio);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_REALTIME_AUDIO_H_
