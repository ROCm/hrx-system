// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/qwen_schedule.h"

#include <string.h>

iree_host_size_t loom_serve_qwen_schedule(
    iree_host_size_t row_count, const loom_serve_qwen_ready_span_t* ready,
    iree_host_size_t token_capacity, iree_host_size_t span_capacity,
    iree_host_size_t chunk_size, iree_host_size_t* cursor,
    loom_serve_qwen_scheduled_span_t* spans) {
  iree_host_size_t count = 0;
  iree_host_size_t remaining = token_capacity;
  iree_host_size_t visited = 0;
  for (; visited < row_count && count < span_capacity && remaining; ++visited) {
    const iree_host_size_t row = (*cursor + visited) % row_count;
    if (ready[row].token_count && ready[row].minimum_count <= remaining) {
      spans[count++] =
          (loom_serve_qwen_scheduled_span_t){row, ready[row].minimum_count};
      remaining -= ready[row].minimum_count;
    }
  }
  for (iree_host_size_t i = 0; i < count && remaining; ++i) {
    loom_serve_qwen_scheduled_span_t* span = &spans[i];
    const iree_host_size_t limit =
        iree_max(span->token_count,
                 iree_min(ready[span->row_index].token_count, chunk_size));
    const iree_host_size_t extra =
        iree_min(remaining, limit - span->token_count);
    span->token_count += extra;
    remaining -= extra;
  }
  if (count) {
    *cursor = (*cursor + (visited < row_count ? visited : 1)) % row_count;
  }
  return count;
}

iree_host_size_t loom_serve_qwen_schedule_shapes(
    iree_host_size_t row_count, const loom_serve_qwen_ready_span_t* ready,
    iree_host_size_t shape_count, const loom_serve_qwen_shape_t* shapes,
    iree_host_size_t chunk_size, iree_host_size_t* cursor,
    loom_serve_qwen_scheduled_span_t* spans,
    loom_serve_qwen_scheduled_span_t* scratch, iree_host_size_t* out_shape) {
  iree_host_size_t best_count = 0;
  iree_host_size_t best_tokens = 0;
  iree_host_size_t best_shape = 0;
  iree_host_size_t best_cursor = *cursor;
  for (iree_host_size_t i = 0; i < shape_count; ++i) {
    iree_host_size_t candidate_cursor = *cursor;
    const iree_host_size_t count = loom_serve_qwen_schedule(
        row_count, ready, shapes[i].token_capacity, shapes[i].span_capacity,
        chunk_size, &candidate_cursor, scratch);
    iree_host_size_t tokens = 0;
    for (iree_host_size_t j = 0; j < count; ++j) {
      tokens += scratch[j].token_count;
    }
    const loom_serve_qwen_shape_t best = shapes[best_shape];
    const bool smaller = shapes[i].token_capacity < best.token_capacity ||
                         (shapes[i].token_capacity == best.token_capacity &&
                          shapes[i].span_capacity < best.span_capacity);
    if (i == 0 || tokens > best_tokens ||
        (tokens == best_tokens &&
         (count > best_count || (count == best_count && smaller)))) {
      best_count = count;
      best_tokens = tokens;
      best_shape = i;
      best_cursor = candidate_cursor;
      memcpy(spans, scratch, count * sizeof(*spans));
    }
  }
  *cursor = best_cursor;
  *out_shape = best_shape;
  return best_count;
}
