// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/packing.h"

loom_serve_packing_plan_t loom_serve_pack_spans(
    iree_host_size_t row_count, const loom_serve_ready_span_t* ready,
    iree_host_size_t token_capacity, iree_host_size_t span_capacity,
    iree_host_size_t chunk_size, iree_host_size_t cursor,
    loom_serve_packed_span_t* spans) {
  iree_host_size_t count = 0;
  iree_host_size_t remaining = token_capacity;
  iree_host_size_t deferred_row = row_count;
  for (iree_host_size_t visited = 0; visited < row_count; ++visited) {
    const iree_host_size_t row = (cursor + visited) % row_count;
    if (!ready[row].token_count || ready[row].minimum_count > token_capacity) {
      continue;
    }
    if (count < span_capacity && ready[row].minimum_count <= remaining) {
      spans[count++] =
          (loom_serve_packed_span_t){row, ready[row].minimum_count};
      remaining -= ready[row].minimum_count;
    } else if (deferred_row == row_count) {
      // Filling a later gap must not lose the first deferred row's priority.
      deferred_row = row;
    }
    // Once admission is full and next priority is known, later rows cannot
    // change either decision.
    if (deferred_row < row_count && (!remaining || count == span_capacity)) {
      break;
    }
  }
  iree_host_size_t expanded_span = count;
  for (iree_host_size_t i = 0; i < count && remaining; ++i) {
    loom_serve_packed_span_t* span = &spans[i];
    const iree_host_size_t limit =
        iree_max(span->token_count,
                 iree_min(ready[span->row_index].token_count, chunk_size));
    const iree_host_size_t extra =
        iree_min(remaining, limit - span->token_count);
    if (extra && expanded_span == count) {
      expanded_span = i;
    }
    span->token_count += extra;
    remaining -= extra;
  }
  iree_host_size_t next_row = cursor;
  iree_host_size_t next_span = 0;
  if (count) {
    const iree_host_size_t first = expanded_span < count ? expanded_span : 0;
    const iree_host_size_t following = spans[first].row_index + 1;
    next_row = following == row_count ? 0 : following;
    next_span = first + 1 == count ? 0 : first + 1;
  }
  const bool deferred = deferred_row < row_count;
  return (loom_serve_packing_plan_t){
      .span_count = count,
      .token_count = token_capacity - remaining,
      .next_row = deferred ? deferred_row : next_row,
      .next_span = next_span,
      .deferred = deferred,
  };
}
