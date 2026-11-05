// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/storage_bounds.h"

#include <stdlib.h>

static int loom_serve_text_storage_bound_compare(const void* lhs,
                                                 const void* rhs) {
  const loom_serve_text_storage_bound_t* a = lhs;
  const loom_serve_text_storage_bound_t* b = rhs;
  if (a->allocation != b->allocation) {
    return a->allocation < b->allocation ? -1 : 1;
  }
  if (a->begin != b->begin) {
    return a->begin < b->begin ? -1 : 1;
  }
  if (a->end != b->end) {
    return a->end < b->end ? -1 : 1;
  }
  return 0;
}

uint64_t loom_serve_text_storage_bound(
    iree_host_size_t count, loom_serve_text_storage_bound_t* bounds) {
  if (!count) {
    return 0;
  }
  qsort(bounds, count, sizeof(*bounds), loom_serve_text_storage_bound_compare);
  uint64_t total = 0;
  for (iree_host_size_t begin = 0; begin < count;) {
    uint64_t end = bounds[begin].end;
    uint64_t minimum = 0;
    uint64_t fixed_end = bounds[begin].begin;
    uint64_t fixed_units = 0;
    iree_host_size_t i = begin;
    do {
      const loom_serve_text_storage_bound_t* bound = &bounds[i];
      minimum = iree_max(minimum, bound->minimum);
      if (bound->minimum == bound->end - bound->begin) {
        const uint64_t first = iree_max(fixed_end, bound->begin);
        if (bound->end > first) {
          fixed_units += bound->end - first;
        }
        fixed_end = iree_max(fixed_end, bound->end);
      }
      end = iree_max(end, bound->end);
      ++i;
    } while (i < count && bounds[i].allocation == bounds[begin].allocation &&
             bounds[i].begin < end);
    total += iree_min(iree_max(minimum, fixed_units), UINT64_MAX - total);
    begin = i;
  }
  return total;
}
