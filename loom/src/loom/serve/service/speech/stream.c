// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/speech/stream.h"

#include <math.h>
#include <string.h>

struct loom_serve_speech_stream_t {
  // Allocator owning the handle and its bounded waveform storage.
  iree_allocator_t host_allocator;
  // Immutable geometry copied from the prepared model.
  loom_serve_speech_format_t format;
  // Caller ownership, consumed by close on the serialized owner.
  bool caller_open;
  // Service ownership, consumed by end only after all CPU borrowers retire.
  bool service_open;
  // Irreversible loss of output interest, independent of caller ownership.
  bool cancelled;
  // Owner-published block; workers cannot write while this is true.
  bool published;
  // Terminal result retained after service/model destruction.
  iree_status_t status;
  // CPU-written metadata, exposed only after publication on the owner.
  loom_serve_speech_block_t block;
  // Bounded copied F32 waveform, never a borrowed source allocation.
  uint8_t* samples;
};

static void loom_serve_speech_stream_release(
    loom_serve_speech_stream_t* stream) {
  if (!stream->caller_open && !stream->service_open) {
    iree_status_free(stream->status);
    iree_allocator_free(stream->host_allocator, stream);
  }
}

static bool loom_serve_speech_stream_open(void* self) {
  const loom_serve_speech_stream_t* stream = self;
  return stream->caller_open && !stream->cancelled;
}

static bool loom_serve_speech_stream_credit(void* self) {
  const loom_serve_speech_stream_t* stream = self;
  return !stream->published;
}

static iree_status_t loom_serve_speech_stream_encode(
    void* self, const loom_serve_speech_block_t* block) {
  loom_serve_speech_stream_t* stream = self;
  for (iree_host_size_t i = 0; i < block->samples.data_length;
       i += sizeof(float)) {
    if (!isfinite(iree_unaligned_load_le_f32(block->samples.data + i))) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "source produced a nonfinite waveform sample");
    }
  }
  stream->block = *block;
  if (block->samples.data_length) {
    memcpy(stream->samples, block->samples.data, block->samples.data_length);
  }
  stream->block.samples =
      iree_make_const_byte_span(stream->samples, block->samples.data_length);
  return iree_ok_status();
}

static iree_status_t loom_serve_speech_stream_publish(void* self,
                                                      bool* out_published) {
  loom_serve_speech_stream_t* stream = self;
  stream->published = stream->block.frame_count || stream->block.complete;
  *out_published = true;
  return iree_ok_status();
}

static void loom_serve_speech_stream_cancel(void* self) {
  loom_serve_speech_stream_t* stream = self;
  stream->cancelled = true;
  stream->published = false;
}

static iree_status_t loom_serve_speech_stream_end(
    void* self, loom_serve_telemetry_outcome_t outcome,
    const iree_status_t status) {
  loom_serve_speech_stream_t* stream = self;
  stream->service_open = false;
  if (outcome != LOOM_SERVE_TELEMETRY_COMPLETED) {
    stream->published = false;
    stream->status = iree_status_is_ok(status)
                         ? iree_status_from_code(IREE_STATUS_CANCELLED)
                         : iree_status_clone(status);
  }
  loom_serve_speech_stream_release(stream);
  return iree_ok_status();
}

iree_status_t loom_serve_speech_stream_allocate(
    loom_serve_speech_format_t format, iree_host_size_t block_capacity,
    iree_allocator_t host_allocator, loom_serve_speech_stream_t** out_stream,
    loom_serve_speech_consumer_t* out_consumer) {
  *out_stream = NULL;
  *out_consumer = (loom_serve_speech_consumer_t){0};
  const uint64_t frame_bytes = (uint64_t)format.channels * sizeof(float);
  if (block_capacity >
      (SIZE_MAX - sizeof(loom_serve_speech_stream_t)) / frame_bytes) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "native speech output allocation is too large");
  }
  loom_serve_speech_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator,
      sizeof(*stream) + block_capacity * format.channels * sizeof(float),
      (void**)&stream));
  stream->host_allocator = host_allocator;
  stream->format = format;
  stream->caller_open = true;
  stream->service_open = true;
  stream->samples = (uint8_t*)(stream + 1);
  *out_stream = stream;
  *out_consumer = (loom_serve_speech_consumer_t){
      .self = stream,
      .is_open = loom_serve_speech_stream_open,
      .can_write = loom_serve_speech_stream_credit,
      .encode = loom_serve_speech_stream_encode,
      .publish = loom_serve_speech_stream_publish,
      .cancel = loom_serve_speech_stream_cancel,
      .end = loom_serve_speech_stream_end,
  };
  return iree_ok_status();
}

loom_serve_speech_format_t loom_serve_speech_stream_format(
    const loom_serve_speech_stream_t* stream) {
  return stream->format;
}

const loom_serve_speech_block_t* loom_serve_speech_stream_peek(
    const loom_serve_speech_stream_t* stream) {
  return stream->published ? &stream->block : NULL;
}

void loom_serve_speech_stream_consume(loom_serve_speech_stream_t* stream) {
  stream->published = false;
}

bool loom_serve_speech_stream_is_complete(
    const loom_serve_speech_stream_t* stream) {
  return !stream->service_open && !stream->published;
}

iree_status_t loom_serve_speech_stream_status(
    const loom_serve_speech_stream_t* stream) {
  return iree_status_clone(stream->status);
}

void loom_serve_speech_stream_close(loom_serve_speech_stream_t* stream) {
  stream->caller_open = false;
  stream->published = false;
  loom_serve_speech_stream_release(stream);
}
