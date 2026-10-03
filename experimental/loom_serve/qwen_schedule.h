// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_QWEN_SCHEDULE_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_QWEN_SCHEDULE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Trusted readiness from the model owner after request and output-credit
// checks.
typedef struct loom_serve_qwen_ready_span_t {
  // Available input tokens; zero means this row is not eligible.
  iree_host_size_t token_count;
  // Smallest legal admission: one for known input, the full verifier otherwise.
  // For eligible rows this is positive and no larger than token_count.
  iree_host_size_t minimum_count;
} loom_serve_qwen_ready_span_t;

typedef struct loom_serve_qwen_scheduled_span_t {
  // Resident row, independent of its position in the packed activation matrix.
  iree_host_size_t row_index;
  // Nonempty prefix of that row's currently ready input.
  iree_host_size_t token_count;
} loom_serve_qwen_scheduled_span_t;

// Capacities of one prepared model traversal; token and span counts are
// independent dimensions. A shape does not own row state or model storage.
typedef struct loom_serve_qwen_shape_t {
  // Maximum number of packed input tokens.
  iree_host_size_t token_capacity;
  // Maximum number of independently advancing resident rows.
  iree_host_size_t span_capacity;
} loom_serve_qwen_shape_t;

// Five token classes through 512 and five span classes through 16.
enum { LOOM_SERVE_QWEN_DEFAULT_SHAPE_CAPACITY = 25 };

// Cold catalog construction for validated row_count in [1, 16] and maximum
// token capacity in [1, 512]. Token classes start at 32; span classes at one.
// Each axis doubles and includes its exact terminal capacity, including odd
// counts. Shapes with more spans than tokens are omitted. Writes at most
// DEFAULT_SHAPE_CAPACITY entries and returns the nonzero count.
iree_host_size_t loom_serve_qwen_default_shapes(
    iree_host_size_t row_count, iree_host_size_t token_capacity,
    loom_serve_qwen_shape_t* shapes);

// Reserves page-rounded capacity for a validated request through completion.
// input_count includes retained and appended input. The final selected output
// stays pending and does not enter KV. Legal speculative steps may transiently
// write past that committed endpoint, even when output credit is only two.
// Positive input/output counts already fit context_capacity; proposal_depth is
// zero or three and block_size is positive. No allocation or state mutation.
iree_host_size_t loom_serve_qwen_request_reservation(
    iree_host_size_t context_capacity, iree_host_size_t input_count,
    iree_host_size_t output_count, iree_host_size_t proposal_depth,
    iree_host_size_t block_size);

// Packs trusted ready spans without allocating or modifying row state. Every
// admitted row gets its minimum_count before remaining capacity is filled from
// longer known spans. chunk_size limits expansion, never an indivisible
// minimum. Rows whose minimum does not fit are skipped; smaller spans can fill
// the gap. This preserves complete verifiers while allowing prompt tails to
// share work.
//
// row_count, token_capacity, span_capacity and chunk_size are positive. cursor
// is below row_count. spans has room for min(row_count, span_capacity) entries.
// Rotation bounds starvation when the shape cannot admit every ready row and
// distributes large prompt chunks when it can. Only the caller commits model
// progress, after execution succeeds; output credit determines readiness.
iree_host_size_t loom_serve_qwen_schedule(
    iree_host_size_t row_count, const loom_serve_qwen_ready_span_t* ready,
    iree_host_size_t token_capacity, iree_host_size_t span_capacity,
    iree_host_size_t chunk_size, iree_host_size_t* cursor,
    loom_serve_qwen_scheduled_span_t* spans);

// Evaluates each cached shape against the same readiness and rotating cursor.
// Chooses the most useful tokens, breaking ties by more ready rows advanced,
// then smaller token and span capacity. Equally occupied prompt-only shapes
// cannot displace plans that also advance ready peers. This is an occupancy
// policy, not a measured execution-cost model.
// shapes is nonempty; each shape satisfies schedule's capacity contract.
// spans and scratch each have row_count entries. Only the winning plan advances
// cursor. Returns its span count and writes its index, including for empty
// work.
iree_host_size_t loom_serve_qwen_schedule_shapes(
    iree_host_size_t row_count, const loom_serve_qwen_ready_span_t* ready,
    iree_host_size_t shape_count, const loom_serve_qwen_shape_t* shapes,
    iree_host_size_t chunk_size, iree_host_size_t* cursor,
    loom_serve_qwen_scheduled_span_t* spans,
    loom_serve_qwen_scheduled_span_t* scratch, iree_host_size_t* out_shape);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_QWEN_SCHEDULE_H_
