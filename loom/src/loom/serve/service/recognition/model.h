// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_RECOGNITION_MODEL_H_
#define LOOM_SERVE_SERVICE_RECOGNITION_MODEL_H_

#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/program/compilation.h"
#include "loom/serve/runtime/program/sources.h"
#include "loom/serve/service/recognition/generator.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_recognition_model_t loom_serve_recognition_model_t;
typedef struct loom_serve_recognition_load_t loom_serve_recognition_load_t;

// Alignment of dynamic model buffers; interior layout belongs to source.
#define LOOM_SERVE_RECOGNITION_BUFFER_ALIGNMENT 16

typedef struct loom_serve_recognition_model_options_t {
  // Explicit source providers and local base, borrowed through load completion.
  loom_serve_sources_t sources;
  // Model package assets, interpreted by its cold source program.
  iree_string_view_t asset_directory;
  // Source-owned JSON object selecting capacities and model policy.
  iree_string_view_t options;
} loom_serve_recognition_model_options_t;

// Completion callbacks only schedule the serialized owner; they never reenter
// the load or block. Options/device/payload live through terminal advance.
typedef void (*loom_serve_recognition_load_wake_fn_t)(void* user_data);

// Allocates an unstarted load without reading source or accepting asynchronous
// work. Private preparation/JIT uses the shared host; owner turns alone mutate
// shared residency and allocation state.
iree_status_t loom_serve_recognition_load_allocate(
    loom_serve_device_t* device,
    const loom_serve_recognition_model_options_t* options,
    loom_serve_recognition_load_wake_fn_t wake, void* user_data,
    iree_allocator_t host_allocator, loom_serve_recognition_load_t** out_load);

enum loom_serve_recognition_load_flag_bits_e {
  // Retires accepted preparation before terminal cancellation.
  LOOM_SERVE_RECOGNITION_LOAD_FLAG_CANCEL = 1u << 0,
};
typedef uint32_t loom_serve_recognition_load_flags_t;

// Bounded nonblocking owner turn. A nonterminal result owes a wake. Terminal
// completion, including failure, means no callback can still borrow the load.
iree_status_t loom_serve_recognition_load_advance(
    loom_serve_recognition_load_t* load,
    loom_serve_recognition_load_flags_t flags, bool* out_complete);

// Terminal-only queries. Evidence is borrowed; take transfers a successful
// unactivated model and leaves the load empty.
const loom_serve_compilation_report_t* loom_serve_recognition_load_report(
    const loom_serve_recognition_load_t* load);
loom_serve_recognition_model_t* loom_serve_recognition_load_take_model(
    loom_serve_recognition_load_t* load);

// Frees a terminal operation and any untaken model. A platform release failure
// preserves the remaining ownership for the caller.
iree_status_t loom_serve_recognition_load_free(
    loom_serve_recognition_load_t* load);

// Source ABI: cold prepare(asset_root, options: buffer) ->
//   (request_capacity, batch_units, input_stride, conditioning_capacity,
//    stream_stride, batch_input_capacity, output_capacity, feedback_capacity,
//    input_frames, frame_limit, event_bytes, sample_rate, channels,
//    event_capacity: i64, outputs: i32,
//    tokenizer_asset, name, model_state, initial_conditioning: buffer).
// Positive geometry is checked at this external source boundary. Input/stream
// strides are alignment multiples; whole arenas fit signed device offsets.
// Tokenizer is optional: a code-facing tokenizer identity for a card-backed
// model, otherwise a path relative to asset_root. Initial conditioning and
// model_state survive the cold program. Card checkpoints use
// prepare.source_scope; direct programs declare prepare.source. Both bind
// sources to resident policy inputs with prepare.parameter.
// outputs declares a nonempty subset of recognition/events.h output flags.
// Request preparation receives the explicitly selected subset, and completion
// may emit only those event types. event_capacity bounds decoded event count;
// event_bytes independently bounds the complete encoded packet.
//
// Warm prepare_request(model_state, language, prompt: buffer, temperature: f32,
//                      outputs: i32)
//   -> (request_state, payload: buffer).
// prepare_work(model_state, request_state, pcm: buffer, position: i64, eof:
// i32)
//   -> (work: buffer, units, compatibility_tag: i64).
// prepare_batch(model_state, stage_tags, ranges, packed_work: buffer)
//   -> (payload, plan: buffer).
// generate(plan: buffer, work, requests, conditioning, streams,
//          output: hal.buffer).
// complete(model_state, request_state, plan: buffer, ordinal: i64,
//          feedback: buffer)
//   -> (consumed_frames: i64, events: buffer, event_bytes: i64, complete: i32).
//
// Request payload may be empty when source carries all request inputs in its
// host state or subsequent work. Admission uploads only nonempty payloads and
// conditioning, while still retiring accepted residency activation.
//
// prepare_work sees a bounded interleaved F32 little-endian input snapshot at
// the absolute input position. It does not advance source progress. Zero units
// waits for more input; EOF requires a positive offer that can drain the
// source. Equal tags and additive units fitting batch_units permit a cohort
// regardless of each row's position. ranges has one little-endian i64 triple
// (resident_row, byte_offset, byte_length) per selected request, indexing
// packed_work. Source owns phase/shape compatibility and numerical packing,
// including EOF padding.
//
// generate imports runner.execute_5 and runner.feedback. Dynamic roots hold
// compact work, stable immutable requests, conditioning, stable stream state,
// and packed output. Persistent history belongs in stream rows, not temporary
// executable workspace. Compiler workspace and checkpoint domains use the same
// shared execution and residency owners as synthesis and other modalities.
//
// Each original ordinal is collected after actual retirement, even when
// cancelled. consumed_frames acknowledges only a prefix of its input snapshot;
// EOF completion requires the whole snapshot to be consumed. event_bytes is
// the produced extent within events, not its allocation capacity; zero means no
// event. The buffer remains non-NULL and may be reused for later ordinals. The
// native caller copies and checks the bounded packet before another source call
// can mutate it. Nonempty packets follow recognition/events.h. Revisions
// replace the whole hypothesis identified by their segment, including speaker
// and alignment evidence. The transcript owner retains segment history and
// checks revision order/finality; multiple provisional segments may coexist.
// Finalized segments are immutable, and completion leaves none uncommitted.
// Silence may complete without emitting text. Coverage is not word alignment.
//
// One model CPU owner and one accepted numerical batch serialize source calls.
// Request state, uploads, and original ordinals survive membership changes and
// cancellation. Caller-owned event packets survive batch release and model
// destruction. Source state changes and conditioning publication occur only
// after their accepted producer retires. No model math is linked in the server.
//
// Cold outer-owner constructor: joins the same asynchronous load, never on a
// shared worker. Live registration uses load_advance. Creation reads metadata
// and compiles stages without loading checkpoint payloads. The shared device
// outlives the model. Failure leaves *out_model NULL.
iree_status_t loom_serve_recognition_model_allocate(
    loom_serve_device_t* device,
    const loom_serve_recognition_model_options_t* options,
    iree_allocator_t host_allocator,
    loom_serve_recognition_model_t** out_model);

// Begins irreversible removal after service CPU tasks and waits retire. No more
// model calls are permitted. Queue consumers are relinquished, while host
// uploads, feedback and residency protection remain through actual retirement.
void loom_serve_recognition_model_retire(loom_serve_recognition_model_t* model,
                                         void (*wake)(void* user_data),
                                         void* user_data);
// True authorizes free_retired; failed semaphore readiness alone is
// insufficient.
bool loom_serve_recognition_model_is_retired(
    loom_serve_recognition_model_t* model);
// Final owner-side destruction; platform failure preserves remaining ownership.
iree_status_t loom_serve_recognition_model_free_retired(
    loom_serve_recognition_model_t* model);
// Cold outer-owner join and destruction, never used on shared worker callbacks.
iree_status_t loom_serve_recognition_model_free(
    loom_serve_recognition_model_t* model);

// Borrowed admission and serialized invocation interfaces, live through model
// retirement. The generator preserves request ownership on terminal failure.
loom_serve_residency_t* loom_serve_recognition_model_residency(
    const loom_serve_recognition_model_t* model);
loom_serve_recognition_generator_t loom_serve_recognition_model_generator(
    loom_serve_recognition_model_t* model);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_RECOGNITION_MODEL_H_
