// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_PACKING_H_
#define LOOM_SERVE_SERVICE_TEXT_PACKING_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Trusted readiness from the model owner after request and output-credit
// checks.
typedef struct loom_serve_ready_span_t {
  // Available input tokens; zero means this row is not eligible.
  iree_host_size_t token_count;
  // Smallest indivisible prefix, such as one known token or a full verifier.
  // For eligible rows this is positive and no larger than token_count.
  iree_host_size_t minimum_count;
} loom_serve_ready_span_t;

typedef struct loom_serve_packed_span_t {
  // Resident row, independent of its position in the packed activation matrix.
  iree_host_size_t row_index;
  // Nonempty prefix of that row's currently ready input.
  iree_host_size_t token_count;
} loom_serve_packed_span_t;

// Capacities of one prepared model traversal; token and span counts are
// independent dimensions. A shape does not own row state or model storage.
typedef struct loom_serve_packing_shape_t {
  // Maximum number of packed input tokens.
  iree_host_size_t token_capacity;
  // Maximum number of independently advancing resident rows.
  iree_host_size_t span_capacity;
} loom_serve_packing_shape_t;

// Selected partition and the producer-owned priority facts needed to continue
// it. Resident priority and rotation within its compact cohort are distinct:
// the first deferred resident may not belong to the cohort at all.
typedef struct loom_serve_packing_plan_t {
  // Number of populated output spans.
  iree_host_size_t span_count;
  // Sum of their live input tokens, excluding shape padding.
  iree_host_size_t token_count;
  // Selected catalog entry; zero when packing one explicit shape.
  iree_host_size_t shape_index;
  // Next priority in the original readiness row domain.
  iree_host_size_t next_row;
  // Next priority in the packed span domain, or zero for an empty cohort.
  iree_host_size_t next_span;
  // next_row names an eligible row omitted for lack of token or span capacity.
  bool deferred;
} loom_serve_packing_plan_t;

// Packs trusted ready spans without allocating or modifying row state. Every
// admitted row gets its minimum_count before remaining capacity is filled from
// longer known spans. chunk_size limits expansion, never an indivisible
// minimum. Rows whose minimum does not fit are skipped; smaller spans can fill
// the gap. This preserves complete verifiers while allowing prompt tails to
// share work.
//
// row_count, token_capacity, span_capacity, and chunk_size are positive. cursor
// is below row_count. spans has room for min(row_count, span_capacity) entries.
// The first deferred row whose minimum fits this shape gets next priority.
// When every eligible row fits, priority advances past the first expanded row,
// or the first admitted row if none expanded. Indivisible spans and idle gaps
// therefore do not delay rotation of prompt expansion.
// Empty readiness preserves cursor in next_row. Only the caller commits model
// progress, after execution succeeds; output credit determines readiness.
loom_serve_packing_plan_t loom_serve_pack_spans(
    iree_host_size_t row_count, const loom_serve_ready_span_t* ready,
    iree_host_size_t token_capacity, iree_host_size_t span_capacity,
    iree_host_size_t chunk_size, iree_host_size_t cursor,
    loom_serve_packed_span_t* spans);

// Evaluates each cached shape against the same readiness and rotating cursor.
// Chooses the most useful tokens, breaking ties by more ready rows advanced,
// then smaller token and span capacity. Equally occupied prompt-only shapes
// cannot displace plans that also advance ready peers. This is an occupancy
// policy, not a measured execution-cost model.
// shapes is nonempty; each shape satisfies the capacity contract above.
// spans and scratch each have row_count entries. Only the winning plan supplies
// the next priorities; candidate evaluation never modifies the caller's cursor.
loom_serve_packing_plan_t loom_serve_pack_shapes(
    iree_host_size_t row_count, const loom_serve_ready_span_t* ready,
    iree_host_size_t shape_count, const loom_serve_packing_shape_t* shapes,
    iree_host_size_t chunk_size, iree_host_size_t cursor,
    loom_serve_packed_span_t* spans, loom_serve_packed_span_t* scratch);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_PACKING_H_
