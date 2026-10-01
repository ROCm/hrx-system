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

typedef struct loom_serve_qwen_options_t {
  // Compiled prefill directory, including its compile-time config.json.
  iree_string_view_t prefill_directory;
  // Compiled decode directory with identical parameter placement and context.
  iree_string_view_t decode_directory;
  // Number of cached packed-epoch stages; zero selects isolated execution.
  iree_host_size_t epoch_count;
  // Borrowed directories with identical weights and retained-state placement.
  const iree_string_view_t* epoch_directories;
  // Optional MTP command bundle: draft, begin, and warm<capacity> directories.
  // Uses the same retained rows and canonical target embedding/output weights.
  iree_string_view_t mtp_directory;
  // Canonical UD-Q5_K_XL GGUF file loaded once during creation.
  iree_string_view_t weights_path;
  // Hugging Face tokenizer.json loaded once during creation.
  iree_string_view_t tokenizer_path;
  // Number of retained rows preallocated in one fixed state arena (1-8).
  iree_host_size_t row_count;
} loom_serve_qwen_options_t;

enum loom_serve_qwen_span_flag_bits_e {
  // Select the next token after consuming this span. An intermediate input
  // chunk can omit this flag; its previous prediction then becomes invalid.
  LOOM_SERVE_QWEN_SPAN_FLAG_SELECT = 1u << 0,
};
typedef uint32_t loom_serve_qwen_span_flags_t;

// One ready input span. Decode is an ordinary length-one span containing the
// row's previously selected token; model math does not distinguish its source.
typedef struct loom_serve_qwen_span_t {
  // Resident row index, independent of this span's packed activation position.
  iree_host_size_t row_index;
  // Nonempty prefix appended at the row's current consumed position.
  iree_host_size_t token_count;
  // Borrowed tokenizer/model-produced IDs, copied before any submission.
  const int32_t* token_ids;
  // Output selection policy for this span.
  loom_serve_qwen_span_flags_t flags;
} loom_serve_qwen_span_t;

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
iree_host_size_t loom_serve_qwen_model_prefill_capacity(
    const loom_serve_qwen_model_t* model);
// Shapes are in artifact option order and borrow model-owned immutable storage.
// Zero means no packed-epoch stages were loaded.
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

// Proposes three candidate successors for distinct rows with pending tokens.
// Does not consume target inputs or change committed predictions/positions.
// All rows must have four remaining context slots. Candidate KV/hidden state
// is private to MTP; a following target epoch overwrites its committed prefix.
// |out_tokens| contains three IDs per row in caller order, valid on success.
iree_status_t loom_serve_qwen_model_propose(loom_serve_qwen_model_t* model,
                                            iree_host_size_t row_count,
                                            const iree_host_size_t* row_indices,
                                            int32_t (*out_tokens)[3]);

// Clears recurrent state and position without allocating or changing ownership.
// Attention beyond the new logical prefix is inaccessible and need not clear.
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
