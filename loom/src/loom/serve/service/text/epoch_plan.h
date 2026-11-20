// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_EPOCH_PLAN_H_
#define LOOM_SERVE_SERVICE_TEXT_EPOCH_PLAN_H_

#include "loom/serve/service/text/model.h"

#ifdef __cplusplus
extern "C" {
#endif

struct loom_serve_text_state_row_t;

// Native bounds retained until the model publishes both submissions.
typedef struct loom_serve_text_epoch_grant_t {
  // Resident owner of this first-plan span.
  uint32_t row_index;
  // Maximum caller-supplied inputs the owner may report consuming.
  uint32_t known_limit;
  // Shared output credit for the owner, bounded by the native result capacity.
  uint32_t output_limit;
} loom_serve_text_epoch_grant_t;

// Source-visible little-endian u32 request and publication record layouts.
// Causal and source positions are distinct; neither may be reconstructed from
// token IDs or the other's increment.
enum {
  TEXT_PLAN_FIELD_LENGTH = 0,
  TEXT_PLAN_FIELD_POSITION = 1,
  TEXT_PLAN_FIELD_SOURCE_POSITION = 2,
  TEXT_PLAN_FIELD_ROW = 3,
  TEXT_PLAN_FIELD_INPUT_BEGIN = 4,
  TEXT_PLAN_FIELD_INPUT_COUNT = 5,
  TEXT_PLAN_FIELD_FLAGS = 6,
  TEXT_PLAN_FIELD_OUTPUT_LIMIT = 7,
  TEXT_PLAN_FIELD_FIRST_SPAN = 8,
  TEXT_PLAN_FIELD_COUNT = 9,
};

enum {
  TEXT_PROGRESS_FIELD_CONSUMED_COUNT = 0,
  TEXT_PROGRESS_FIELD_KNOWN_COUNT = 1,
  TEXT_PROGRESS_FIELD_OUTPUT_COUNT = 2,
  TEXT_PROGRESS_FIELD_VERIFICATION_COUNT = 3,
  TEXT_PROGRESS_FIELD_SOURCE_POSITION = 4,
  TEXT_PROGRESS_FIELD_TOKENS = 5,
  TEXT_PROGRESS_FIELD_COUNT =
      TEXT_PROGRESS_FIELD_TOKENS + LOOM_SERVICE_TEXT_MAX_EPOCH_OUTPUTS,
};

// Serializes a validated partition and copies its known IDs exactly once.
// rows is the model's resident row table. plan and input are its capacity-sized
// host buffers. A continuation supplies first_indices and writes after the
// first plan and its known IDs. Returns the next free input ordinal.
iree_host_size_t loom_serve_text_epoch_plan_write(
    struct loom_serve_text_state_row_t* const* rows, iree_byte_span_t plan,
    iree_byte_span_t input, iree_host_size_t span_count,
    const loom_serve_text_span_t* spans, const uint32_t* output_limits,
    const uint32_t* first_indices, iree_host_size_t record_begin,
    iree_host_size_t input_begin);

// Checks the complete VM publication against native reservations before any
// caller-visible result or resident position is committed.
iree_status_t loom_serve_text_epoch_plan_validate_progress(
    iree_host_size_t span_count, const loom_serve_text_epoch_grant_t* grants,
    const uint32_t* extents, const uint8_t* records);

// Decodes a validated record. Only the live prefix of token IDs is published.
void loom_serve_text_epoch_plan_read_progress(
    const uint8_t* record, loom_serve_text_result_t* out_result,
    uint32_t* out_source_position);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVICE_TEXT_EPOCH_PLAN_H_
