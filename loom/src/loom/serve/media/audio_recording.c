// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/media/audio_recording.h"

#include <math.h>

#include "loom/serve/media/audio_decoder.h"

struct loom_serve_audio_recording_t {
  // Allocator owning this object and its decoded sample storage.
  iree_allocator_t host_allocator;
  // Finite interleaved F32LE storage and original geometry.
  loom_serve_audio_source_t source;
};

static iree_status_t loom_serve_audio_recording_decode(
    loom_serve_audio_decoder_t* decoder, iree_host_size_t max_frames,
    loom_serve_audio_recording_t* recording) {
  const iree_host_size_t frame_bytes =
      recording->source.channels * sizeof(float);
  iree_host_size_t capacity = 0;
  iree_host_size_t length = 0;
  float* samples = NULL;
  bool complete = false;
  iree_status_t status = iree_ok_status();
  while (!complete && iree_status_is_ok(status)) {
    if (length == capacity && capacity < max_frames) {
      const iree_host_size_t growth = capacity ? capacity : 4096;
      capacity += iree_min(growth, max_frames - capacity);
      status = iree_allocator_realloc(recording->host_allocator,
                                      capacity * frame_bytes, (void**)&samples);
    }
    iree_host_size_t count = 0;
    if (iree_status_is_ok(status)) {
      // At the limit, one extra frame distinguishes exact EOF from overflow.
      // Overwriting the first frame is safe: overflow rejects the entire
      // result.
      status = loom_serve_audio_decoder_read(
          decoder, length == capacity ? 1 : capacity - length,
          samples +
              (length == capacity ? 0 : length * recording->source.channels),
          &count, &complete);
    }
    if (iree_status_is_ok(status) && count > max_frames - length) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "decoded audio exceeds the model's upload limit");
    }
    if (iree_status_is_ok(status)) {
      length += count;
    }
  }
  recording->source.data =
      iree_make_const_byte_span(samples, length * frame_bytes);
  recording->source.frame_count = length;
  return status;
}

iree_status_t loom_serve_audio_recording_allocate(
    const loom_serve_audio_source_t* source, iree_host_size_t max_bytes,
    iree_allocator_t host_allocator,
    loom_serve_audio_recording_t** out_recording) {
  *out_recording = NULL;
  loom_serve_audio_recording_t* recording = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*recording),
                                             (void**)&recording));
  recording->host_allocator = host_allocator;
  recording->source = *source;
  recording->source.encoding = LOOM_SERVE_AUDIO_ENCODING_F32LE;
  recording->source.data = iree_const_byte_span_empty();
  loom_serve_audio_decoder_t* decoder = NULL;
  iree_status_t status = iree_ok_status();
  if (source->encoding == LOOM_SERVE_AUDIO_ENCODING_CONTAINER) {
    status = loom_serve_audio_decoder_allocate(source->data, 0, 0,
                                               host_allocator, &decoder);
    if (iree_status_is_ok(status)) {
      recording->source.sample_rate =
          loom_serve_audio_decoder_sample_rate(decoder);
      recording->source.channels = loom_serve_audio_decoder_channels(decoder);
    }
  }
  const uint64_t frame_bytes =
      (uint64_t)recording->source.channels * sizeof(float);
  if (iree_status_is_ok(status) && frame_bytes > max_bytes) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "one audio frame exceeds the model's upload limit");
  }
  if (iree_status_is_ok(status) && decoder) {
    status = loom_serve_audio_recording_decode(decoder, max_bytes / frame_bytes,
                                               recording);
  } else if (iree_status_is_ok(status)) {
    if (source->frame_count > max_bytes / frame_bytes) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "decoded audio exceeds the model's upload limit");
    } else {
      float* samples = NULL;
      const iree_host_size_t length = source->frame_count * frame_bytes;
      status = iree_allocator_malloc_uninitialized(host_allocator, length,
                                                   (void**)&samples);
      recording->source.data = iree_make_const_byte_span(samples, length);
      if (iree_status_is_ok(status)) {
        status = loom_serve_audio_source_read(source, 0, source->frame_count,
                                              samples);
      }
    }
  }
  loom_serve_audio_decoder_free(decoder);
  if (iree_status_is_ok(status) && !recording->source.frame_count) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "audio recording must contain at least one frame");
  }
  float* samples = (float*)recording->source.data.data;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) &&
       i < recording->source.data.data_length / sizeof(float);
       ++i) {
    const float sample = samples[i];
    if (source->encoding == LOOM_SERVE_AUDIO_ENCODING_CONTAINER &&
        !isfinite(sample)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "audio recording contains a nonfinite sample");
    } else {
      iree_unaligned_store_le_f32(samples + i, sample);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_recording = recording;
  } else {
    loom_serve_audio_recording_free(recording);
  }
  return status;
}

loom_serve_audio_source_t loom_serve_audio_recording_source(
    const loom_serve_audio_recording_t* recording) {
  return recording->source;
}

void loom_serve_audio_recording_free(loom_serve_audio_recording_t* recording) {
  if (recording) {
    const iree_allocator_t allocator = recording->host_allocator;
    iree_allocator_free(allocator, (void*)recording->source.data.data);
    iree_allocator_free(allocator, recording);
  }
}
