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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_QWEN_SCHEDULE_H_
