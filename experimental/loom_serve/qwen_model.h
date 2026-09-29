// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_QWEN_MODEL_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_QWEN_MODEL_H_

#include "iree/base/api.h"
#include "iree/tokenizer/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// One concrete Qwen3.8-27B residency. A single host owner multiplexes retained
// rows through the same VM process, commands, weights and workspace. Calls wait
// for their result; no row-local VM or thread is required. This runner-private
// interface deliberately exposes stages, not chat or network sessions.
// Input-capacity rejection submits no work. An execution/submission failure
// ends the run: destroy drains accepted work before any host payload is reused.
typedef struct loom_serve_qwen_model_t loom_serve_qwen_model_t;
typedef struct loom_serve_qwen_row_t loom_serve_qwen_row_t;

typedef struct loom_serve_qwen_options_t {
  // Compiled prefill directory, including its compile-time config.json.
  iree_string_view_t prefill_directory;
  // Compiled decode directory with identical parameter placement and context.
  iree_string_view_t decode_directory;
  // Canonical UD-Q5_K_XL GGUF file loaded once during creation.
  iree_string_view_t weights_path;
  // Hugging Face tokenizer.json loaded once during creation.
  iree_string_view_t tokenizer_path;
  // Number of retained rows preallocated in one fixed state arena (1-8).
  iree_host_size_t row_count;
} loom_serve_qwen_options_t;

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

// Result queries are valid after a successful stage. Position counts consumed
// tokens, excluding the currently selected token. Reset clears all metrics.
int32_t loom_serve_qwen_row_token(const loom_serve_qwen_row_t* row);
bool loom_serve_qwen_row_is_eos(const loom_serve_qwen_row_t* row);
iree_host_size_t loom_serve_qwen_row_position(const loom_serve_qwen_row_t* row);
loom_serve_qwen_metrics_t loom_serve_qwen_row_metrics(
    const loom_serve_qwen_row_t* row);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_QWEN_MODEL_H_
