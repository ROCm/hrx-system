// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_QWEN_MODEL_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_QWEN_MODEL_H_

#include "experimental/loom_serve/qwen_schedule.h"
#include "iree/base/api.h"
#include "iree/tokenizer/tokenizer.h"
#include "loomc/sanitizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// One concrete Qwen3.8-27B residency. A single host owner multiplexes retained
// rows through the same VM process, commands, weights and workspace. Calls wait
// for their result; no row-local VM or thread is required. This runner-private
// interface deliberately exposes model work, not chat or network sessions.
// Input-capacity rejection submits no work. An execution/submission failure
// ends the run: destroy drains accepted work before any host payload is reused.
typedef struct loom_serve_qwen_model_t loom_serve_qwen_model_t;
typedef struct loom_serve_qwen_row_t loom_serve_qwen_row_t;

// Four-input verification uses at most 64 selected-token entries per epoch.
// Retained arenas are sized to row_count, not this control-payload bound.
enum { LOOM_SERVE_QWEN_ROW_CAPACITY = 16 };

typedef struct loom_serve_qwen_options_t {
  // Portable model source directory, including sources.txt and control.loom.
  iree_string_view_t source_directory;
  // Maximum input count specialized into isolated prefill.
  iree_host_size_t prefill_capacity;
  // Logical attention-position ceiling per row, not a pooled reservation.
  iree_host_size_t context_capacity;
  // Shared physical token capacity, a multiple of 64. Nonzero selects paged
  // packed execution; zero retains dense addressing for comparison. This is
  // independent of the per-row logical context ceiling and requires epochs.
  iree_host_size_t pool_capacity;
  // Number of cached packed-epoch stages; zero selects isolated execution.
  iree_host_size_t epoch_count;
  // Borrowed shapes specialized from the shared catalog during creation.
  const loom_serve_qwen_shape_t* epoch_shapes;
  // Prepare MTP proposal, catch-up, and verification using shared target
  // weights.
  bool enable_mtp;
  // Device assertion classes/reporting applied to every JIT kernel pipeline.
  loomc_sanitizer_options_t kernel_sanitizer;
  // Canonical UD-Q5_K_XL GGUF file loaded once during creation.
  iree_string_view_t weights_path;
  // Hugging Face tokenizer.json loaded once during creation.
  iree_string_view_t tokenizer_path;
  // Number of retained rows preallocated in one fixed state arena (1-16).
  iree_host_size_t row_count;
} loom_serve_qwen_options_t;

enum loom_serve_qwen_span_flag_bits_e {
  // Select the next token after consuming this span. An intermediate input
  // chunk can omit this flag; its previous prediction then becomes invalid.
  LOOM_SERVE_QWEN_SPAN_FLAG_SELECT = 1u << 0,
  // Generate three candidates on device before verification. The span reserves
  // four inputs but token_ids supplies only the pending anchor. Requires SELECT
  // and a nonzero output limit in model_verify.
  LOOM_SERVE_QWEN_SPAN_FLAG_PROPOSE = 1u << 1,
};
typedef uint32_t loom_serve_qwen_span_flags_t;

// One ready input span. Decode is an ordinary length-one span containing the
// row's previously selected token; model math does not distinguish its source.
typedef struct loom_serve_qwen_span_t {
  // Resident row index, independent of this span's packed activation position.
  iree_host_size_t row_index;
  // Nonempty prefix appended at the row's current consumed position.
  iree_host_size_t token_count;
  // Borrowed known IDs, copied before submission. PROPOSE supplies one anchor;
  // the other three reserved inputs are generated and consumed on device.
  const int32_t* token_ids;
  // Output selection and optional device proposal generation for this span.
  loom_serve_qwen_span_flags_t flags;
} loom_serve_qwen_span_t;

// Committed progress for one known or speculative span. A speculative span
// consumes its pending anchor plus matched drafts; its final output stays
// pending. Consumed and output counts therefore agree for speculative spans.
typedef struct loom_serve_qwen_result_t {
  // Inputs consumed into retained target and MTP state.
  iree_host_size_t consumed_count;
  // Selected output IDs, including an EOS or the first rejected replacement.
  iree_host_size_t output_count;
  // Valid prefix of output_count IDs in generation order.
  int32_t tokens[4];
} loom_serve_qwen_result_t;

typedef struct loom_serve_qwen_metrics_t {
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
} loom_serve_qwen_metrics_t;

// Physical pooled capacity measured in token positions, including page
// rounding. Dense comparison storage reports zero capacity and availability.
typedef struct loom_serve_qwen_pool_usage_t {
  // Number of positions in one indivisible private cache page.
  iree_host_size_t block_size;
  // Total positions backed by target and optional draft storage.
  iree_host_size_t capacity;
  // Positions in pages not currently owned by any row.
  iree_host_size_t available;
} loom_serve_qwen_pool_usage_t;

// Cold setup allocates all device state and prepares immutable stages. Options
// strings are borrowed only for this call. Failure releases partial ownership.
iree_status_t loom_serve_qwen_model_create(
    const loom_serve_qwen_options_t* options, iree_allocator_t host_allocator,
    loom_serve_qwen_model_t** out_model);

// Drains accepted work, releases all residency, and reports terminal failure.
// Null is accepted. All borrowed row/tokenizer pointers become invalid.
iree_status_t loom_serve_qwen_model_destroy(loom_serve_qwen_model_t* model);

// Queries return borrowed state owned by model. The row index is below the
// configured row count. The tokenizer can be shared by independent encoders.
loom_serve_qwen_row_t* loom_serve_qwen_model_row(loom_serve_qwen_model_t* model,
                                                 iree_host_size_t index);
iree_tokenizer_t* loom_serve_qwen_model_tokenizer(
    loom_serve_qwen_model_t* model);
iree_host_size_t loom_serve_qwen_model_context_capacity(
    const loom_serve_qwen_model_t* model);
// Copies physical pool accounting at the single owner's completed-stage
// boundary. Request completion reservations are service policy, not physical
// ownership.
loom_serve_qwen_pool_usage_t loom_serve_qwen_model_pool_usage(
    const loom_serve_qwen_model_t* model);
// Positions in the row's owned pages; zero for dense comparison storage.
iree_host_size_t loom_serve_qwen_row_pool_usage(
    const loom_serve_qwen_row_t* row);
// Maximum chunk accepted by row_prefill. Pooled/MTP execution is also bounded
// by the largest packed shape, which owns mapped target/draft state updates.
iree_host_size_t loom_serve_qwen_model_prefill_capacity(
    const loom_serve_qwen_model_t* model);
// Shapes are in option order and borrow model-owned immutable storage.
// Zero means no packed-epoch stages were prepared.
iree_host_size_t loom_serve_qwen_model_shape_count(
    const loom_serve_qwen_model_t* model);
const loom_serve_qwen_shape_t* loom_serve_qwen_model_shapes(
    const loom_serve_qwen_model_t* model);

// Advances distinct resident rows in one packed model traversal. The nonempty
// span list and its total token count must fit the prepared capacities, and
// each span must fit its row's remaining context. Validation rejects the whole
// epoch before submission. Completion commits positions and selected tokens;
// execution failure ends the run and destroy drains all accepted work.
// Model storage, weights, commands and VM state are preallocated and reused.
iree_status_t loom_serve_qwen_model_epoch(loom_serve_qwen_model_t* model,
                                          iree_host_size_t shape_index,
                                          iree_host_size_t span_count,
                                          const loom_serve_qwen_span_t* spans);

// Mixes ordinary known spans with four-input speculative spans on one cached
// target shape. A zero output limit denotes known input. Limits 1-4 denote
// speculative input, require SELECT and a pending non-EOS prediction, and cap
// the number of new selected outputs. PROPOSE spans supply only the pending
// token: three draft rounds publish directly into the verifier input buffer.
// Other speculative spans supply {pending token, three proposals} themselves.
// Greedy acceptance stops at the first mismatch, EOS, or limit. Only accepted
// state is published; catch-up pairs accepted inputs with target hidden state.
// Proposal, verification, commit and catch-up have no intermediate host wait or
// readback. Only completed output/progress records cross back to the caller.
// Results are in caller order and valid on success. Validation and submission
// failure have the same contracts as model_epoch; all storage is reused.
iree_status_t loom_serve_qwen_model_verify(
    loom_serve_qwen_model_t* model, iree_host_size_t shape_index,
    iree_host_size_t span_count, const loom_serve_qwen_span_t* spans,
    const uint32_t* output_limits, loom_serve_qwen_result_t* out_results);

// Clears recurrent state and position, then releases retired private KV pages.
// Backing allocations remain fixed. Attention beyond the new logical prefix
// is inaccessible and need not clear before its pages are assigned again.
iree_status_t loom_serve_qwen_row_reset(loom_serve_qwen_row_t* row);

// Appends one active chunk at the retained absolute position and selects the
// next token. Input count is in [1, prefill_capacity] and must fit the context.
// The selected token has NOT been consumed into KV/GDN state. Another prefill
// may discard that prediction; decode consumes it. Input is copied before any
// submission, so its memory is not borrowed after return, even on failure.
iree_status_t loom_serve_qwen_row_prefill(loom_serve_qwen_row_t* row,
                                          iree_host_size_t count,
                                          const int32_t* token_ids);

// Consumes the previously selected token and selects its successor. The caller
// has completed prefill and chosen to continue; EOS policy belongs to it.
iree_status_t loom_serve_qwen_row_decode(loom_serve_qwen_row_t* row);

// Token/EOS queries require a successfully selected prediction, not a reset or
// an epoch span that omitted SELECT. Position counts consumed tokens, excluding
// the currently selected token. Metrics cover isolated prefill/decode calls;
// a shared epoch's duration belongs to the batch, not to each constituent row.
int32_t loom_serve_qwen_row_token(const loom_serve_qwen_row_t* row);
bool loom_serve_qwen_row_is_eos(const loom_serve_qwen_row_t* row);
iree_host_size_t loom_serve_qwen_row_position(const loom_serve_qwen_row_t* row);
loom_serve_qwen_metrics_t loom_serve_qwen_row_metrics(
    const loom_serve_qwen_row_t* row);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_QWEN_MODEL_H_
