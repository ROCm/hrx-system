// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/media/audio_decoder.h"

#include "miniaudio.h"

struct loom_serve_audio_decoder_t {
  // Allocator owning this object and all codec/converter allocations.
  iree_allocator_t host_allocator;
  // First callback allocation failure, transferred by the enclosing operation.
  iree_status_t allocation_status;
  // Codec and resampler state borrowing the encoded upload.
  ma_decoder decoder;
  // Successful initialization authorizes codec destruction.
  bool initialized;
  // Explicit EOF, independent of caller output-block boundaries.
  bool complete;
};

static void* loom_serve_audio_allocate(size_t size, void* user_data) {
  loom_serve_audio_decoder_t* decoder = user_data;
  if (!iree_status_is_ok(decoder->allocation_status)) {
    return NULL;
  }
  void* pointer = NULL;
  decoder->allocation_status = iree_allocator_malloc_uninitialized(
      decoder->host_allocator, size, &pointer);
  return pointer;
}

static void* loom_serve_audio_reallocate(void* pointer, size_t size,
                                         void* user_data) {
  loom_serve_audio_decoder_t* decoder = user_data;
  if (!iree_status_is_ok(decoder->allocation_status)) {
    return NULL;
  }
  decoder->allocation_status =
      iree_allocator_realloc(decoder->host_allocator, size, &pointer);
  return iree_status_is_ok(decoder->allocation_status) ? pointer : NULL;
}

static void loom_serve_audio_free(void* pointer, void* user_data) {
  loom_serve_audio_decoder_t* decoder = user_data;
  iree_allocator_free(decoder->host_allocator, pointer);
}

// Allocation callbacks cannot propagate IREE diagnostics through ma_result.
// Retain the original failure until the enclosing operation can transfer it.
static iree_status_t loom_serve_audio_result(
    loom_serve_audio_decoder_t* decoder, ma_result result) {
  if (!iree_status_is_ok(decoder->allocation_status)) {
    iree_status_t status = decoder->allocation_status;
    decoder->allocation_status = iree_ok_status();
    return status;
  }
  if (result == MA_SUCCESS) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "audio decoding failed: %s (%d)",
                          ma_result_description(result), (int)result);
}

iree_status_t loom_serve_audio_decoder_allocate(
    iree_const_byte_span_t encoded, uint32_t sample_rate, uint32_t channels,
    iree_allocator_t host_allocator, loom_serve_audio_decoder_t** out_decoder) {
  *out_decoder = NULL;
  if (!encoded.data_length || channels > MA_MAX_CHANNELS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "audio requires nonempty bytes and valid output "
                            "sample rate/channel count");
  }
  loom_serve_audio_decoder_t* decoder = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*decoder),
                                             (void**)&decoder));
  decoder->host_allocator = host_allocator;
  ma_decoder_config config =
      ma_decoder_config_init(ma_format_f32, channels, sample_rate);
  config.resampling.linear.lpfOrder = MA_MAX_FILTER_ORDER;
  config.allocationCallbacks =
      (ma_allocation_callbacks){.pUserData = decoder,
                                .onMalloc = loom_serve_audio_allocate,
                                .onRealloc = loom_serve_audio_reallocate,
                                .onFree = loom_serve_audio_free};
  const ma_result result = ma_decoder_init_memory(
      encoded.data, encoded.data_length, &config, &decoder->decoder);
  decoder->initialized = result == MA_SUCCESS;
  iree_status_t status = loom_serve_audio_result(decoder, result);
  if (iree_status_is_ok(status)) {
    *out_decoder = decoder;
  } else {
    loom_serve_audio_decoder_free(decoder);
  }
  return status;
}

uint32_t loom_serve_audio_decoder_sample_rate(
    const loom_serve_audio_decoder_t* decoder) {
  return decoder->decoder.outputSampleRate;
}

uint32_t loom_serve_audio_decoder_channels(
    const loom_serve_audio_decoder_t* decoder) {
  return decoder->decoder.outputChannels;
}

iree_status_t loom_serve_audio_decoder_read(loom_serve_audio_decoder_t* decoder,
                                            iree_host_size_t frame_capacity,
                                            float* frames,
                                            iree_host_size_t* out_frame_count,
                                            bool* out_complete) {
  *out_frame_count = 0;
  *out_complete = decoder->complete;
  if (!frame_capacity || decoder->complete) {
    return iree_ok_status();
  }
  ma_uint64 count = 0;
  const ma_result result = ma_decoder_read_pcm_frames(&decoder->decoder, frames,
                                                      frame_capacity, &count);
  iree_status_t status = loom_serve_audio_result(
      decoder, result == MA_AT_END ? MA_SUCCESS : result);
  if (iree_status_is_ok(status)) {
    decoder->complete = result == MA_AT_END;
    *out_frame_count = (iree_host_size_t)count;
    *out_complete = decoder->complete;
  }
  return status;
}

void loom_serve_audio_decoder_free(loom_serve_audio_decoder_t* decoder) {
  if (!decoder) {
    return;
  }
  if (decoder->initialized) {
    ma_decoder_uninit(&decoder->decoder);
  }
  iree_status_free(decoder->allocation_status);
  iree_allocator_free(decoder->host_allocator, decoder);
}
