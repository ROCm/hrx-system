// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_SPEECH_MODEL_H_
#define LOOM_SERVE_SERVICE_SPEECH_MODEL_H_

#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/program/compilation.h"
#include "loom/serve/runtime/program/sources.h"
#include "loom/serve/service/speech/generator.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_speech_model_t loom_serve_speech_model_t;
typedef struct loom_serve_speech_load_t loom_serve_speech_load_t;

// Alignment of dynamic model buffers; interior layout belongs to source.
#define LOOM_SERVE_SPEECH_BUFFER_ALIGNMENT 16

typedef struct loom_serve_speech_model_options_t {
  // Explicit source providers and local base, borrowed through load completion.
  loom_serve_sources_t sources;
  // Model package assets, interpreted by its cold source program.
  iree_string_view_t asset_directory;
  // Source-owned JSON object selecting capacities, voices and model policy.
  iree_string_view_t options;
} loom_serve_speech_model_options_t;

// Completion callbacks only schedule the serialized owner; they never reenter
// the load or block. Options/device/payload live through terminal advance.
typedef void (*loom_serve_speech_load_wake_fn_t)(void* user_data);

// Allocates an unstarted load without reading source or accepting asynchronous
// work. Private preparation/JIT uses the shared host; owner turns alone mutate
// shared residency and allocation state.
iree_status_t loom_serve_speech_load_allocate(
    loom_serve_device_t* device,
    const loom_serve_speech_model_options_t* options,
    loom_serve_speech_load_wake_fn_t wake, void* user_data,
    iree_allocator_t host_allocator, loom_serve_speech_load_t** out_load);

enum loom_serve_speech_load_flag_bits_e {
  // Retires accepted preparation before terminal cancellation.
  LOOM_SERVE_SPEECH_LOAD_FLAG_CANCEL = 1u << 0,
};
typedef uint32_t loom_serve_speech_load_flags_t;

// Bounded nonblocking owner turn. A nonterminal result owes a wake. Terminal
// completion, including failure, means no callback can still borrow the load.
iree_status_t loom_serve_speech_load_advance(
    loom_serve_speech_load_t* load, loom_serve_speech_load_flags_t flags,
    bool* out_complete);

// Terminal-only queries. Evidence is borrowed; take transfers a successful
// unactivated model and leaves the load empty.
const loom_serve_compilation_report_t* loom_serve_speech_load_report(
    const loom_serve_speech_load_t* load);
loom_serve_speech_model_t* loom_serve_speech_load_take_model(
    loom_serve_speech_load_t* load);

// Frees a terminal operation and any untaken model. A platform release failure
// preserves the remaining ownership for the caller.
iree_status_t loom_serve_speech_load_free(loom_serve_speech_load_t* load);

// Speech ABI1 (and the unversioned development contract):
// cold prepare(asset_root, options: buffer) ->
//   (request_capacity, batch_units, input_stride, conditioning_capacity,
//    stream_stride, batch_input_capacity, output_capacity, feedback_capacity,
//    block_frames, sample_rate, channels: i64,
//    tokenizer_asset, name, model_state, initial_conditioning: buffer).
// Capacities and geometry are positive host sizes. Input/stream strides are
// alignment multiples; whole arenas fit signed device offsets. Tokenizer is
// optional and relative to asset_root. Opaque model_state and
// initial_conditioning survive the cold program. Empty initial conditioning
// requires source to initialize its device bank before use. Checkpoints use
// prepare.parameter.
//
// Warm prepare_request(model_state, input, voice, instructions: buffer,
//                      seed: i64, speed: f32)
//   -> (request_state, payload: buffer).
// prepare_work(model_state, request_state: buffer)
//   -> (work: buffer, units, compatibility_tag: i64).
// prepare_batch(model_state, stage_tags, ranges, packed_work: buffer)
//   -> (payload, plan: buffer).
// generate(plan: buffer, work, requests, conditioning, streams,
//          output: hal.buffer).
// complete(model_state, request_state, plan: buffer, ordinal: i64,
//          feedback: buffer)
//   -> (frame_position, pcm_byte_offset, frame_count: i64, complete: i32).
//
// Speech ABI2 retains the same numerical scheduler and waveform contract.
// Cold speech_prepare_v2 inserts voice_capacity and voice_audio_capacity (i64)
// after channels and before the four buffer results. Both are positive; the
// latter bounds decoded original-recording bytes, not compressed upload size.
// All service exports use the speech_*_v2 suffix in the selected model card.
//
// speech_prepare_voice_v2(model_state: buffer, voice_slot: i64,
//                         original_pcm: buffer, sample_rate, channels: i64)
//   -> (voice_state, request_state, payload: buffer).
// PCM is finite interleaved F32LE at its original rate and channel count. This
// CPU call owns model-specific mixing, resampling, trimming, and padding. Its
// request proceeds through prepare_work/batch, generate, and complete as usual;
// it must produce no waveform frames and report EOF after conditioning is
// ready. voice_slot is reserved exclusively for this upload until it and every
// later synthesis capture retire. The slot's numerical storage belongs to
// conditioning. voice_state is immutable opaque CPU metadata, retained until
// the last capture retires; source may retain PCM or derived buffers in its
// request state.
//
// speech_prepare_request_v2 receives captured voice_state instead of a voice-ID
// string. IDs and slot reuse belong to the native service. It publishes an ID
// only after successful upload and row-release retirement. Cancellation cannot
// publish unfinished conditioning; removal hides the ID while preserving
// already accepted synthesis. Each slot reuse must overwrite all conditioning
// that its new incarnation will read. Preparing a voice never runs again during
// synthesis.
//
// Request payload may be empty when source carries all request inputs in its
// host state or subsequent work. Admission uploads only nonempty payloads and
// conditioning, while still retiring accepted residency activation.
//
// prepare_work is CPU-only and does not advance request progress. Offers are
// refreshed at a retired model cut. Equal tags and additive units fitting
// batch_units permit a cohort; different progress alone need not separate it.
// ranges has one little-endian i64 triple (resident_row, byte_offset,
// byte_length) per selected request, indexing packed_work. The service owns
// eligibility and selects distinct admitted rows; source owns shape/phase
// selection and packing. Payload and plan survive accepted work and every
// completion borrower.
//
// generate imports runner.execute_5 and runner.feedback. Five dynamic roots
// hold compact batch work, stable request rows, shared conditioning, stable
// stream rows and packed output. Rows survive batch membership changes.
// Compiler-private workspace lasts for one command; persistent histories belong
// in the stream bank. Each command declares its fixed parameter bindings. Roots
// sharing an ordered checkpoint file list and preparation policy share
// compatible reflected storage; disjoint roots retain their placement. Distinct
// file lists or policies own independent storage, even for identically named
// tensors. Direct stages use the same five dynamic roots without checkpoint
// domains.
//
// Every retired unit calls complete for each original batch ordinal, including
// phases producing no audio or readback. Source owns generation, codec, RNG and
// cache positions. Completion publishes continuous interleaved F32 samples and
// explicit EOF independently of whether a unit emits samples. The CPU consumer
// copies output before releasing the batch; request rows remain independently
// resident. No model math is linked into the server.
//
// One model CPU owner and one accepted numerical batch serialize shared state.
// Model-state changes and conditioning publication occur only after their
// accepted producer retires. Partial preparation cannot mark an unfinished
// conditioning entry reusable. Conditioned references and initial assets remain
// model-defined, independently of per-request stream progress.
//
// Cold outer-owner constructor: joins the same asynchronous load, never on a
// shared worker. Live registration uses load_advance. Creation reads metadata
// and compiles stages without loading checkpoint payloads. The shared device
// outlives the model. Failure leaves *out_model NULL.
iree_status_t loom_serve_speech_model_allocate(
    loom_serve_device_t* device,
    const loom_serve_speech_model_options_t* options,
    iree_allocator_t host_allocator, loom_serve_speech_model_t** out_model);

// Begins irreversible removal after service CPU tasks and waits retire. No more
// model calls are permitted. Queue consumers are relinquished, while host
// uploads, feedback and residency protection remain through actual retirement.
void loom_serve_speech_model_retire(loom_serve_speech_model_t* model,
                                    void (*wake)(void* user_data),
                                    void* user_data);
// True authorizes free_retired; failed semaphore readiness alone is
// insufficient.
bool loom_serve_speech_model_is_retired(loom_serve_speech_model_t* model);
// Final owner-side destruction; platform failure preserves remaining ownership.
iree_status_t loom_serve_speech_model_free_retired(
    loom_serve_speech_model_t* model);
// Cold outer-owner join and destruction, never used on shared worker callbacks.
iree_status_t loom_serve_speech_model_free(loom_serve_speech_model_t* model);

// Borrowed admission and serialized invocation interfaces, live through model
// retirement. The generator preserves request ownership on terminal failure.
loom_serve_residency_t* loom_serve_speech_model_residency(
    const loom_serve_speech_model_t* model);
loom_serve_speech_generator_t loom_serve_speech_model_generator(
    loom_serve_speech_model_t* model);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_SPEECH_MODEL_H_
