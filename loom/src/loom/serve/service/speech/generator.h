// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_SPEECH_GENERATOR_H_
#define LOOM_SERVE_SERVICE_SPEECH_GENERATOR_H_

#include "iree/tokenizer/tokenizer.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/runtime/execution/scheduler.h"
#include "loom/serve/runtime/parameter/residency.h"
#include "loom/serve/service/speech/input.h"
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

// Source-selected next operation after one bounded CPU input step.
typedef enum loom_serve_speech_work_kind_e {
  LOOM_SERVE_SPEECH_WORK_NUMERICAL = 0,
  LOOM_SERVE_SPEECH_WORK_CPU = 1,
  LOOM_SERVE_SPEECH_WORK_INPUT = 2,
  LOOM_SERVE_SPEECH_WORK_COMPLETE = 3,
} loom_serve_speech_work_kind_t;

// Source-established input acceptance and next operation at a retired cut.
// Different progress is permitted within a numerical tag; phase and geometry
// are source facts. Input acceptance is independent of numerical selection.
typedef struct loom_serve_speech_work_t {
  // Numerical work, another bounded CPU step, input waiting, or final output.
  loom_serve_speech_work_kind_t kind;
  // Complete-scalar input prefix accepted by this CPU call, even if a later
  // operation fails. The owner returns this credit only after task retirement.
  iree_host_size_t consumed_bytes;
  // Metadata for the selected operation; CPU and INPUT carry no extra data.
  union {
    // Compatible next numerical work, independently of request progress.
    struct {
      // Positive additive cost within the model's batch-unit capacity.
      uint64_t units;
      // Equal tags permit one source batch when the sum of units fits.
      uint64_t tag;
    } numerical;
    // Validated playback position of an empty final waveform block.
    uint64_t frame_position;
  } data;
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
  // Maximum complete-scalar text window and native input ring size, >= 4 bytes.
  iree_host_size_t text_capacity;
  // Maximum raw UTF-8 bytes accepted over the complete request lifetime.
  uint64_t text_limit;
  // CPU preparation returns an owned request handle, with no device submission
  // or shared admission mutation. INVALID_ARGUMENT rejects a client request;
  // other failures are terminal. Failure retains no request and returns NULL.
  // row_index is a free service-owned slot below request_capacity and remains
  // reserved until discard/release consumes the returned handle.
  iree_status_t (*prepare)(void* self, iree_host_size_t row_index,
                           const loom_serve_speech_request_t* request,
                           void** out_invocation);
  // Releases an unadmitted request and its admission interest on the owner.
  // Any accepted activation has completed and its pin has been released.
  void (*discard)(void* self, void* invocation);
  // Shared-owner admission. ACQUIRED owns an activation pin and a frontier even
  // when row backing cannot fit. Cancellation retains this operation until
  // successful completion; failure leaves it under model retirement.
  iree_status_t (*try_admit)(void* self, void* invocation,
                             loom_serve_residency_admission_t* out_admission,
                             loom_serve_execution_frontier_t* out_frontier);
  // Consumes successful admission readiness. True retains the row and pin;
  // false releases the activation pin and leaves an unadmitted request.
  bool (*complete_admit)(void* self, void* invocation);
  // CPU-only input acceptance and next-work preparation at a retired cut.
  // Snapshot bytes are borrowed only through this call; retained source text
  // belongs to the handle. INPUT requires the offered source window drained
  // without EOF; COMPLETE requires final input accepted and all output drained.
  // INVALID_ARGUMENT rejects this request; other failures are terminal. The
  // accepted count remains meaningful on failure. Requery numerical offers
  // after another batch completes; shared conditioning facts may have changed.
  iree_status_t (*prepare_work)(void* self, void* invocation,
                                const loom_serve_speech_input_block_t* input,
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
  // Borrowed numerical controller for the actual execution device, or NULL
  // when the embedding caller manages independent submission itself.
  loom_serve_execution_scheduler_t* scheduler;
  // Optional copied parameter activity, queried only on serialized owner turns.
  loom_serve_weight_activity_t (*weight_activity)(void* self);
} loom_serve_speech_generator_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_SPEECH_GENERATOR_H_
