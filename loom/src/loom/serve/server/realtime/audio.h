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
// Each response owns its cursor and borrows independently retained
// coefficients.
typedef struct loom_serve_realtime_audio_t {
  // Immutable source geometry of this registered speech deployment.
  loom_serve_speech_format_t format;
  // Independent mono history borrowing its immutable filter through retirement.
  loom_serve_audio_resampler_t resampler;
  // Frames already read from the currently borrowed native waveform block.
  iree_host_size_t source_offset;
  // Wire frames generated for this response, independent of network admission.
  uint64_t frames;
} loom_serve_realtime_audio_t;

// Constructs immutable coefficients on a worker. The owner publishes success
// after that worker releases, and retains the filter through every cursor.
iree_status_t loom_serve_realtime_audio_filter_initialize(
    loom_serve_speech_format_t format, iree_allocator_t host_allocator,
    loom_serve_audio_resampler_filter_t* filter);

// Worker-only start/reset. The caller supplies coefficients for format's rate
// and retains them through deinitialization, including accepted CPU borrowers.
iree_status_t loom_serve_realtime_audio_begin(
    loom_serve_speech_format_t format,
    const loom_serve_audio_resampler_filter_t* filter,
    iree_allocator_t host_allocator, loom_serve_realtime_audio_t* audio);

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
