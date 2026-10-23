// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_SPEECH_GENERATOR_H_
#define LOOM_SERVE_SERVICE_SPEECH_GENERATOR_H_

#include "iree/tokenizer/tokenizer.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/runtime/parameter/residency.h"
#include "loom/serve/runtime/storage/memory.h"
#include "loom/serve/service/speech/request.h"

#ifdef __cplusplus
extern "C" {
#endif

// Fixed waveform geometry for one prepared model. Samples are interleaved
// little-endian F32; a frame contains one sample from every channel.
typedef struct loom_serve_speech_format_t {
  // Waveform frames per second, independent of model steps or HTTP packets.
  uint32_t sample_rate;
  // Samples in each interleaved frame.
  uint32_t channels;
} loom_serve_speech_format_t;

// One source-produced, retired waveform block. No VM references escape.
typedef struct loom_serve_speech_block_t {
  // Absolute playback-frame position, independent of model/cache cursors.
  uint64_t position;
  // Number of live waveform frames; zero does not imply end-of-stream.
  iree_host_size_t frame_count;
  // Borrowed interleaved F32 bytes, exactly frame_count * channels * 4 bytes.
  // Batch release invalidates this view.
  iree_const_byte_span_t samples;
  // The source has no more waveform output after this block.
  bool complete;
} loom_serve_speech_block_t;

// Source-established compatibility of one request's next numerical operation.
// Different progress is permitted within a tag; phase and geometry are source
// facts. These facts belong to one retired model cut, not to request lifetime.
typedef struct loom_serve_speech_work_t {
  // Positive additive cost within the model's batch-unit capacity.
  uint64_t units;
  // Equal tags permit one source batch when the sum of units fits.
  uint64_t tag;
} loom_serve_speech_work_t;

// Independently resident requests sharing one source CPU owner and one accepted
// numerical batch. CPU tasks prepare work/batches and collect retired output;
// the shared owner admits, submits, and releases. None waits for device work.
// Failed readiness leaves accepted ownership with model teardown.
typedef struct loom_serve_speech_generator_t {
  // Borrowed model owner, outliving every request, task and completion wait.
  void* self;
  // Immutable model output geometry, independent of its internal codec.
  loom_serve_speech_format_t format;
  // Maximum live waveform frames in one block, bounding encoder storage.
  iree_host_size_t block_capacity;
  // Number of stable resident rows, independent of executable batch shapes.
  iree_host_size_t request_capacity;
  // Maximum sum of source work units in one compatible numerical batch.
  uint64_t batch_units;
  // CPU preparation returns an owned request handle, with no device submission
  // or shared admission mutation. INVALID_ARGUMENT rejects a client request;
  // other failures are terminal. Failure retains no request and returns NULL.
  // row_index is a free service-owned slot below request_capacity and remains
  // reserved until discard/release consumes the returned handle.
  iree_status_t (*prepare)(void* self, iree_host_size_t row_index,
                           const loom_serve_speech_request_t* request,
                           void** out_invocation);
  // Releases an unadmitted request and its admission interest on the owner.
  void (*discard)(void* self, void* invocation);
  // Shared-owner admission of parameters, row backing, and cohort scratch.
  // ACQUIRED owns the row, activation pin, and readiness frontier; denial owns
  // neither row nor pin. Cancellation retains accepted work until successful
  // completion; failure leaves it under model retirement.
  iree_status_t (*try_admit)(void* self, void* invocation,
                             loom_serve_residency_admission_t* out_admission,
                             loom_serve_execution_frontier_t* out_frontier);
  // Collects successful admission readiness, retaining the admitted row and
  // pin. Capacity is decided before submission, never again at completion.
  void (*complete_admit)(void* self, void* invocation);
  // CPU-only next-work preparation at a retired model cut. Does not advance
  // the request or submit work. Retained source bytes belong to the handle.
  // Requery after another batch completes; conditioning facts may have changed.
  iree_status_t (*prepare_work)(void* self, void* invocation,
                                loom_serve_speech_work_t* out_work);
  // CPU-only packing of distinct admitted handles with freshly queried,
  // compatible work. The source owns numerical packing and phase/shape choice.
  // The nonempty cohort and plan remain retained until release_batch. Failure
  // retains no batch; request state is unchanged. No earlier batch is live.
  iree_status_t (*prepare_batch)(void* self, iree_host_size_t count,
                                 void* const* invocations);
  // Queues the prepared numerical batch once. Returned frontiers and all
  // selected requests remain retained through actual retirement.
  iree_status_t (*submit)(void* self,
                          loom_serve_execution_frontier_t* out_frontier);
  // CPU-only interpretation of successfully retired feedback. Checks external
  // source metadata once, then returns a trusted block. The view remains live
  // through encoding; no new submission occurs until the consumer releases it.
  // Original batch ordinal is stable even when peers cancel. Every successful
  // unit is collected, including source phases producing no audio/readback.
  // Failure terminates only this ordinal's request and invalidates its block;
  // the remaining ordinals stay collectible. Release the failed request after
  // every batch borrower retires, using the ordinary release contract.
  iree_status_t (*collect)(void* self, iree_host_size_t ordinal,
                           loom_serve_speech_block_t* out_block);
  // Releases an unsubmitted plan, or a successfully retired batch after every
  // CPU/encoding borrower releases its feedback views. Request rows survive.
  void (*release_batch)(void* self);
  // Releases an admitted request at a successfully retired cut, after its
  // batch/CPU borrowers stop. True consumes the pin/handle. False retains it
  // through the returned borrowed frontier; call again after successful ALL
  // readiness, excluding other model mutation. Backing release never blocks
  // the owner. Platform failure retains the handle for model teardown.
  iree_status_t (*release)(void* self, void* invocation, bool* out_complete,
                           loom_serve_execution_frontier_t* out_frontier);
  // Optional immutable tokenizer, live through model retirement.
  const iree_tokenizer_t* tokenizer;
  // Shared admission identity; equal non-NULL domains can release each other's
  // capacity. Borrowed through service retirement and never dereferenced here.
  const void* capacity_domain;
  // Optional copied parameter activity, queried only on serialized owner turns.
  loom_serve_weight_activity_t (*weight_activity)(void* self);
} loom_serve_speech_generator_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_SPEECH_GENERATOR_H_
