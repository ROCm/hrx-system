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

// Packs trusted ready counts without allocating or modifying row state. Zero
// means unavailable, one includes ordinary decode, and longer spans are known
// prompt input. Every admitted row gets one token before remaining capacity is
// filled from prompt spans, up to chunk_size per row. This preserves decode
// progress while allowing short prompt tails to share the same traversal.
//
// row_count, token_capacity, span_capacity and chunk_size are positive. cursor
// is below row_count. spans has room for min(row_count, span_capacity) entries.
// Rotation bounds starvation when the shape cannot admit every ready row and
// distributes large prompt chunks when it can. Only the caller commits model
// progress, after execution succeeds; output credit determines readiness.
iree_host_size_t loom_serve_qwen_schedule(
    iree_host_size_t row_count, const iree_host_size_t* ready_counts,
    iree_host_size_t token_capacity, iree_host_size_t span_capacity,
    iree_host_size_t chunk_size, iree_host_size_t* cursor,
    loom_serve_qwen_scheduled_span_t* spans);

// Evaluates each cached shape against the same readiness and rotating cursor.
// Chooses the most useful tokens, breaking ties by smaller token then span
// capacity. This is an occupancy policy, not a measured execution-cost model.
// shapes is nonempty; each shape satisfies schedule's capacity contract.
// spans and scratch each have row_count entries. Only the winning plan advances
// cursor. Returns its span count and writes its index, including for empty
// work.
iree_host_size_t loom_serve_qwen_schedule_shapes(
    iree_host_size_t row_count, const iree_host_size_t* ready_counts,
    iree_host_size_t shape_count, const loom_serve_qwen_shape_t* shapes,
    iree_host_size_t chunk_size, iree_host_size_t* cursor,
    loom_serve_qwen_scheduled_span_t* spans,
    loom_serve_qwen_scheduled_span_t* scratch, iree_host_size_t* out_shape);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_QWEN_SCHEDULE_H_
