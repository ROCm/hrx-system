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

// Selected partition and the producer-owned priority facts needed to continue
// it. Resident priority and rotation within its compact cohort are distinct:
// the first deferred resident may not belong to the cohort at all.
typedef struct loom_serve_packing_plan_t {
  // Number of populated output spans.
  iree_host_size_t span_count;
  // Sum of their live input tokens, across the submitted spans.
  iree_host_size_t token_count;
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
// The first deferred row whose minimum fits the token budget gets next
// priority. When every eligible row fits, priority advances past the first
// expanded row, or the first admitted row if none expanded. Indivisible spans
// and idle gaps therefore do not delay rotation of prompt expansion. Empty
// readiness preserves cursor in next_row. Only the caller commits model
// progress, after execution succeeds; output credit determines readiness.
loom_serve_packing_plan_t loom_serve_pack_spans(
    iree_host_size_t row_count, const loom_serve_ready_span_t* ready,
    iree_host_size_t token_capacity, iree_host_size_t span_capacity,
    iree_host_size_t chunk_size, iree_host_size_t cursor,
    loom_serve_packed_span_t* spans);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_PACKING_H_
