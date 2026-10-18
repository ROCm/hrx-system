// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/audio.h"

#include <math.h>

#include "iree/base/internal/base64.h"
#include "loom/serve/service/speech/output.h"

enum {
  LOOM_SERVE_REALTIME_AUDIO_FRAMES = 4096,
  LOOM_SERVE_REALTIME_AUDIO_BYTES = 4 * 1024 * 1024,
};

iree_status_t loom_serve_realtime_audio_begin(
    loom_serve_speech_format_t format, iree_allocator_t host_allocator,
    loom_serve_realtime_audio_t* audio) {
  audio->host_allocator = host_allocator;
  audio->format = format;
  audio->source_offset = 0;
  audio->frames = 0;
  if (!audio->filter_ready) {
    loom_serve_audio_resampler_filter_deinitialize(&audio->filter);
    IREE_RETURN_IF_ERROR(loom_serve_audio_resampler_filter_initialize(
        format.sample_rate, 24000, LOOM_SERVE_REALTIME_AUDIO_BYTES,
        host_allocator, &audio->filter));
    audio->filter_ready = true;
  }
  loom_serve_audio_resampler_deinitialize(&audio->resampler);
  return loom_serve_audio_resampler_initialize(
      &audio->filter, 1, LOOM_SERVE_REALTIME_AUDIO_FRAMES,
      LOOM_SERVE_REALTIME_AUDIO_BYTES, host_allocator, &audio->resampler);
}

iree_status_t loom_serve_realtime_audio_convert(
    loom_serve_realtime_audio_t* audio, const loom_serve_speech_block_t* block,
    iree_string_builder_t* output, bool* out_consumed) {
  *out_consumed = false;
  float samples[LOOM_SERVE_REALTIME_AUDIO_FRAMES];
  iree_host_size_t count = 0, received = 0;
  while (count < LOOM_SERVE_REALTIME_AUDIO_FRAMES) {
    if (block->complete && audio->source_offset == block->frame_count) {
      loom_serve_audio_resampler_finish(&audio->resampler);
    }
    count += loom_serve_audio_resampler_read(
        &audio->resampler, LOOM_SERVE_REALTIME_AUDIO_FRAMES - count,
        samples + count);
    if (count == LOOM_SERVE_REALTIME_AUDIO_FRAMES ||
        received == LOOM_SERVE_REALTIME_AUDIO_FRAMES ||
        audio->source_offset == block->frame_count) {
      break;
    }
    iree_host_size_t capacity = 0;
    float* input =
        loom_serve_audio_resampler_reserve(&audio->resampler, &capacity);
    const iree_host_size_t take = iree_min(
        block->frame_count - audio->source_offset,
        iree_min(capacity, LOOM_SERVE_REALTIME_AUDIO_FRAMES - received));
    for (iree_host_size_t i = 0; i < take; ++i) {
      double sum = 0;
      for (uint32_t channel = 0; channel < audio->format.channels; ++channel) {
        const float sample = iree_unaligned_load_le_f32(
            block->samples.data +
            ((audio->source_offset + i) * audio->format.channels + channel) *
                sizeof(float));
        if (!isfinite(sample)) {
          return iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "speech produced a nonfinite waveform");
        }
        sum += sample;
      }
      const double mono = sum / audio->format.channels;
      input[i] = (float)iree_min(1.0, iree_max(-1.0, mono));
    }
    loom_serve_audio_resampler_commit(&audio->resampler, take);
    audio->source_offset += take;
    received += take;
  }
  // A full destination may leave readable output. Keep borrowing the source
  // block until the next read establishes that its contribution has drained.
  *out_consumed =
      audio->source_offset == block->frame_count &&
      (block->complete
           ? loom_serve_audio_resampler_is_complete(&audio->resampler)
           : count < LOOM_SERVE_REALTIME_AUDIO_FRAMES);
  if (*out_consumed) {
    audio->source_offset = 0;
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_unaligned_store_le_f32(samples + i, samples[i]);
  }
  char pcm_storage[LOOM_SERVE_REALTIME_AUDIO_FRAMES * sizeof(int16_t) + 1];
  iree_string_builder_t pcm;
  iree_string_builder_initialize_with_storage(pcm_storage, sizeof(pcm_storage),
                                              &pcm);
  IREE_RETURN_IF_ERROR(loom_serve_speech_encode_pcm16(
      iree_make_const_byte_span(samples, count * sizeof(float)), &pcm));
  char* encoded = NULL;
  iree_host_size_t capacity = 0, encoded_length = 0;
  IREE_RETURN_IF_ERROR(iree_string_builder_reserve_for_append(
      output, iree_base64_encoded_size(pcm.size), &encoded, &capacity));
  IREE_RETURN_IF_ERROR(iree_base64_encode(
      iree_make_const_byte_span(pcm.buffer, pcm.size),
      iree_make_mutable_string_view(encoded, capacity), &encoded_length));
  iree_string_builder_commit_append(output, encoded_length);
  audio->frames += count;
  return iree_ok_status();
}

void loom_serve_realtime_audio_deinitialize(
    loom_serve_realtime_audio_t* audio) {
  loom_serve_audio_resampler_deinitialize(&audio->resampler);
  loom_serve_audio_resampler_filter_deinitialize(&audio->filter);
}
