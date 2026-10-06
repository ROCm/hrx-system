// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_TEXT_MODEL_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_TEXT_MODEL_H_

#include "experimental/loom_serve/runtime/device.h"
#include "experimental/loom_serve/scheduling/packing.h"
#include "experimental/loom_serve/storage/memory.h"
#include "experimental/loom_serve/text/chat.h"
#include "iree/base/api.h"
#include "iree/tokenizer/tokenizer.h"
#include "loomc/sanitizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// One packed autoregressive model residency. A single host owner multiplexes
// retained rows through the same VM process, commands, weights and workspace.
// Calls wait for their result; no row-local VM or thread is required. This
// runner-private interface exposes model work and shared source policy, not
// network sessions. Input-capacity rejection submits no work. An
// execution/submission failure ends the run: destroy drains accepted work
// before any host payload is reused.
typedef struct loom_serve_text_model_t loom_serve_text_model_t;
typedef struct loom_serve_text_row_t loom_serve_text_row_t;

// Four-input verification uses at most 64 selected-token entries per epoch.
// Retained arenas are sized to row_count, not this control-payload bound.
enum { LOOM_SERVE_TEXT_ROW_CAPACITY = 16 };

typedef struct loom_serve_text_options_t {
  // Portable source directory containing prepare.loom, control.loom and the
  // sources.txt command/kernel catalog.
  iree_string_view_t source_directory;
  // Maximum input count specialized into isolated prefill.
  iree_host_size_t prefill_capacity;
  // Logical attention-position ceiling per row, not a pooled reservation.
  iree_host_size_t context_capacity;
  // Shared cache slot capacity, divisible by the source-declared page
  // size. Nonzero selects paged packed execution; zero uses dense addressing.
  // This is independent of the logical context ceiling and requires epochs.
  iree_host_size_t pool_capacity;
  // Number of cached packed-epoch stages; zero selects isolated execution.
  iree_host_size_t epoch_count;
  // Borrowed shapes specialized from the shared catalog during creation.
  const loom_serve_packing_shape_t* epoch_shapes;
  // Prepare MTP proposal, catch-up, and verification using shared target
  // weights.
  bool enable_mtp;
  // Device assertion classes/reporting applied to every JIT kernel pipeline.
  loomc_sanitizer_options_t kernel_sanitizer;
  // Parameter file indexed here and streamed on activation.
  iree_string_view_t weights_path;
  // Hugging Face tokenizer.json loaded once during creation.
  iree_string_view_t tokenizer_path;
  // Number of addressable retained rows in one stable state arena (1-16).
  iree_host_size_t row_count;
} loom_serve_text_options_t;

enum loom_serve_text_span_flag_bits_e {
  // Select the next token after consuming this span. An intermediate input
  // chunk can omit this flag; its previous prediction then becomes invalid.
  LOOM_SERVE_TEXT_SPAN_FLAG_SELECT = 1u << 0,
  // Generate three candidates on device before verification. The span reserves
  // four inputs but token_ids supplies only the pending anchor. Requires SELECT
  // and a nonzero output limit in model_verify.
  LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE = 1u << 1,
};
typedef uint32_t loom_serve_text_span_flags_t;

// One ready input span. Decode is an ordinary length-one span containing the
// row's previously selected token; model math does not distinguish its source.
typedef struct loom_serve_text_span_t {
  // Resident row index, independent of this span's packed activation position.
  iree_host_size_t row_index;
  // Nonempty prefix appended at the row's current consumed position.
  iree_host_size_t token_count;
  // Borrowed known IDs, copied before submission. PROPOSE supplies one anchor;
  // the other three reserved inputs are generated and consumed on device.
  const int32_t* token_ids;
  // Output selection and optional device proposal generation for this span.
  loom_serve_text_span_flags_t flags;
} loom_serve_text_span_t;

// A pre-issued second plan borrowing rows from the first plan. Known spans
// append new input after an unselected first chunk. Speculative spans require
// SELECT|PROPOSE and consume the first result's final prediction on device;
// their token_ids is unused. The producer compacts survivors without a host
// wait. The shape bounds all entries before EOS/credit filtering.
typedef struct loom_serve_text_continuation_t {
  // Cached shape admitting the second plan's worst-case inputs and spans.
  iree_host_size_t shape_index;
  // Nonempty subset of distinct rows present in the first plan.
  iree_host_size_t span_count;
  // Borrowed next input chunks or device-produced speculative anchors.
  const loom_serve_text_span_t* spans;
  // Zero for known input; otherwise total output credit including the first
  // plan's outputs. Device acceptance subtracts those before continuing.
  const uint32_t* output_limits;
} loom_serve_text_continuation_t;

// Committed progress for one known or speculative span. A speculative span
// consumes its pending anchor plus matched drafts; its final output stays
// pending. Consumed and output counts therefore agree for speculative spans.
typedef struct loom_serve_text_result_t {
  // Inputs consumed into retained target and MTP state.
  iree_host_size_t consumed_count;
  // Consumed caller-supplied inputs, excluding speculative anchor/draft work.
  // This separates prompt progress when a cohort also starts generation.
  iree_host_size_t known_count;
  // Selected output IDs, including an EOS or the first rejected replacement.
  iree_host_size_t output_count;
  // Number of speculative verifications that advanced this span.
  iree_host_size_t verification_count;
  // Valid prefix of output_count IDs across at most two device-fed epochs.
  int32_t tokens[8];
} loom_serve_text_result_t;

typedef struct loom_serve_text_metrics_t {
  // Active input tokens consumed since the last reset.
  uint64_t prefill_tokens;
  // Completed prefill invocations, including non-final chunks.
  uint64_t prefill_steps;
  // Host-observed upload, VM submission, execution and readback nanoseconds.
  iree_duration_t prefill_duration;
  // Completed decode invocations; excludes predictions made by prefill.
  uint64_t decode_steps;
  // Host-observed decode submission, execution and readback nanoseconds.
  iree_duration_t decode_duration;
} loom_serve_text_metrics_t;

// Logical pool capacity measured in token positions, including page
// rounding. Dense comparison storage reports zero capacity and availability.
typedef struct loom_serve_text_pool_usage_t {
  // Number of positions in one indivisible private cache page.
  iree_host_size_t block_size;
  // Total addressable positions in target and optional draft storage.
  iree_host_size_t capacity;
  // Positions in pages not currently owned by any row.
  iree_host_size_t available;
} loom_serve_text_pool_usage_t;

typedef struct loom_serve_text_trim_result_t {
  // Logical blocks moved across every target/draft plane.
  uint32_t moved_blocks;
  // Device bytes copied to compact surviving state.
  uint64_t copied_bytes;
  // Physical bytes returned to the allocator during this maintenance cut.
  uint64_t released_bytes;
} loom_serve_text_trim_result_t;

// Retires accepted work, compacts live private blocks, publishes target/draft
// maps, then releases physical slabs containing no live state. Live rows and
// cached commands survive; reset rows can regrow in the same virtual buffers.
// This is an explicit maintenance operation, not part of ordinary decoding.
// Fixed/dense comparison backing is unchanged and reports zero reclamation.
// Execution/platform failure ends the run, as with model_epoch.
iree_status_t loom_serve_text_model_trim(
    loom_serve_text_model_t* model, loom_serve_text_trim_result_t* out_result);

// Cold setup compiles stages, records commands and indexes parameters without
// loading weight payloads. The shared device owner outlives the model and
// serializes model calls; its pool supplies elastic backing or explicit fixed
// allocations. Pooled state grows on row/page activation. Options strings are
// borrowed only for this call. Failure releases partial ownership.
iree_status_t loom_serve_text_model_create(
    loom_serve_device_t* device, const loom_serve_text_options_t* options,
    loom_serve_text_model_t** out_model, iree_allocator_t host_allocator);

// Drains accepted work, releases all residency, and reports terminal failure.
// Null is accepted. All borrowed row/tokenizer pointers become invalid.
iree_status_t loom_serve_text_model_destroy(loom_serve_text_model_t* model);

// At a serialized model boundary, release/reload parameter backing while
// preserving compiled commands and all retained row state. Deactivation
// requires unpinned elastic backing. Inference pins through retirement and
// activates on demand; explicit activation warms without retaining a pin.
// Admission can reclaim idle weights of other models, never mutable state.
// Execution/I/O failure is terminal.
iree_status_t loom_serve_text_model_activate(loom_serve_text_model_t* model);
iree_status_t loom_serve_text_model_deactivate(loom_serve_text_model_t* model);

// Borrowed admission/retention handle. A caller can try_acquire before choosing
// work, retaining the returned pin across invocations until release. Ordinary
// capacity backpressure leaves all model state unchanged.
loom_serve_residency_t* loom_serve_text_model_residency(
    const loom_serve_text_model_t* model);

// Parameter-only virtual/physical statistics, excluding mutable state.
loom_serve_memory_statistics_t loom_serve_text_model_weight_statistics(
    const loom_serve_text_model_t* model);

// Queries return borrowed state owned by model. The row index is below the
// configured row count. The tokenizer can be shared by independent encoders.
loom_serve_text_row_t* loom_serve_text_model_row(loom_serve_text_model_t* model,
                                                 iree_host_size_t index);
iree_tokenizer_t* loom_serve_text_model_tokenizer(
    loom_serve_text_model_t* model);
// Borrowed source policy; uses the same serialized invocation as model work.
const loom_serve_text_chat_policy_t* loom_serve_text_model_chat_policy(
    const loom_serve_text_model_t* model);
iree_host_size_t loom_serve_text_model_context_capacity(
    const loom_serve_text_model_t* model);
// Copies physical pool accounting at the single owner's completed-stage
// boundary. Request completion reservations are service policy, not physical
// ownership.
loom_serve_text_pool_usage_t loom_serve_text_model_pool_usage(
    const loom_serve_text_model_t* model);
// Mutable virtual state only; immutable weights and shared workspace are
// separate allocations. Fixed backing reports zero virtual-pool statistics.
loom_serve_memory_statistics_t loom_serve_text_model_memory_statistics(
    const loom_serve_text_model_t* model);
// Positions in the row's owned pages; zero for dense comparison storage.
iree_host_size_t loom_serve_text_row_pool_usage(
    const loom_serve_text_row_t* row);
// Maximum chunk accepted by row_prefill. Pooled/MTP execution is also bounded
// by the largest packed shape, which owns mapped target/draft state updates.
iree_host_size_t loom_serve_text_model_prefill_capacity(
    const loom_serve_text_model_t* model);
// Shapes are in option order and borrow model-owned immutable storage.
// Zero means no packed-epoch stages were prepared.
iree_host_size_t loom_serve_text_model_shape_count(
    const loom_serve_text_model_t* model);
const loom_serve_packing_shape_t* loom_serve_text_model_shapes(
    const loom_serve_text_model_t* model);

// Advances distinct resident rows in one packed model traversal. The nonempty
// span list and its total token count must fit the prepared capacities, and
// each span must fit its row's remaining context. Validation rejects the whole
// epoch before submission. Completion commits positions and selected tokens;
// execution failure ends the run and destroy drains all accepted work.
// Weights, commands and VM state are reused. Elastic backing grows only when
// activating new row/page ranges; resident work allocates no physical memory.
iree_status_t loom_serve_text_model_epoch(loom_serve_text_model_t* model,
                                          iree_host_size_t shape_index,
                                          iree_host_size_t span_count,
                                          const loom_serve_text_span_t* spans);

// Mixes ordinary known spans with four-input speculative spans on one cached
// target shape. An optional continuation issues a second mixed plan without
// an intermediate host wait. A zero output limit denotes known input.
// Limits 1 through 4 (8 with continuation) denote speculative input, require
// SELECT and a pending non-EOS prediction, and cap the number of new selected
// outputs. PROPOSE spans supply only the pending token: three draft rounds
// publish directly into the verifier input buffer. Other speculative spans
// supply {pending token, three proposals} themselves. Greedy acceptance stops
// at the first mismatch, EOS, or limit. Only accepted state is published;
// catch-up pairs accepted inputs with target hidden state. Proposal,
// verification, commit and catch-up have no intermediate host wait or readback.
// Only completed output/progress records cross back to the caller. Each known
// input chunk executes once. Continuation advances known chunks or promotes a
// selected first result to speculation, stopping speculative work at EOS,
// output credit or context bound. All inputs are copied before submission.
// Results are in caller order and valid on success. Validation and submission
// failure have the same contracts as model_epoch.
iree_status_t loom_serve_text_model_verify(
    loom_serve_text_model_t* model, iree_host_size_t shape_index,
    iree_host_size_t span_count, const loom_serve_text_span_t* spans,
    const uint32_t* output_limits,
    const loom_serve_text_continuation_t* continuation,
    loom_serve_text_result_t* out_results);

// Clears recurrent state and position, then releases retired private KV pages.
// Physical backing stays warm until model_trim. Attention beyond the new prefix
// is inaccessible and need not clear before its pages are assigned again.
iree_status_t loom_serve_text_row_reset(loom_serve_text_row_t* row);

// Appends one active chunk at the retained absolute position and selects the
// next token. Input count is in [1, prefill_capacity] and must fit the context.
// The selected token has NOT been consumed into KV/recurrent state. Another
// prefill may discard that prediction; decode consumes it. Input is copied
// before any submission, so its memory is not borrowed after return, even on
// failure.
iree_status_t loom_serve_text_row_prefill(loom_serve_text_row_t* row,
                                          iree_host_size_t count,
                                          const int32_t* token_ids);

// Consumes the previously selected token and selects its successor. The caller
// has completed prefill and chosen to continue; EOS policy belongs to it.
iree_status_t loom_serve_text_row_decode(loom_serve_text_row_t* row);

// Token/EOS queries require a successfully selected prediction, not a reset or
// an epoch span that omitted SELECT. Position counts consumed tokens, excluding
// the currently selected token. Metrics cover isolated prefill/decode calls;
// a shared epoch's duration belongs to the batch, not to each constituent row.
int32_t loom_serve_text_row_token(const loom_serve_text_row_t* row);
bool loom_serve_text_row_is_eos(const loom_serve_text_row_t* row);
iree_host_size_t loom_serve_text_row_position(const loom_serve_text_row_t* row);
loom_serve_text_metrics_t loom_serve_text_row_metrics(
    const loom_serve_text_row_t* row);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_TEXT_MODEL_H_
