// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/liveness_events.h"

#include <string.h>

#include "loom/util/adaptive_sort.h"

static bool loom_liveness_event_less(const loom_liveness_event_t* lhs,
                                     const loom_liveness_event_t* rhs) {
  if (lhs->point != rhs->point) {
    return lhs->point < rhs->point;
  }
  if (lhs->value_delta != rhs->value_delta) {
    return lhs->value_delta < rhs->value_delta;
  }
  return lhs->value_id < rhs->value_id;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_liveness_events_compare_sort,
                          loom_liveness_event_t, loom_liveness_event_less)

typedef struct loom_liveness_event_bucket_t {
  // First event reserved for this program point in the partitioned array.
  iree_host_size_t start;
  // Histogram count during construction, then the next unfilled event slot.
  iree_host_size_t cursor;
} loom_liveness_event_bucket_t;

iree_status_t loom_liveness_events_sort(loom_liveness_event_t* events,
                                        iree_host_size_t event_count,
                                        uint32_t maximum_point,
                                        iree_arena_allocator_t* scratch_arena) {
  if (event_count <= LOOM_ADAPTIVE_SORT_INSERTION_COUNT_THRESHOLD ||
      maximum_point >= event_count) {
    loom_liveness_events_compare_sort(events, event_count);
    return iree_ok_status();
  }

  // Program points are usually denser than live-range endpoints. Partition
  // by point first so only simultaneous events require comparison sorting.
  const iree_host_size_t bucket_count = (iree_host_size_t)maximum_point + 1;
  loom_liveness_event_bucket_t* buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, bucket_count, sizeof(*buckets), (void**)&buckets));
  memset(buckets, 0, bucket_count * sizeof(*buckets));
  for (iree_host_size_t i = 0; i < event_count; ++i) {
    ++buckets[events[i].point].cursor;
  }
  iree_host_size_t next_start = 0;
  for (iree_host_size_t i = 0; i < bucket_count; ++i) {
    buckets[i].start = next_start;
    next_start += buckets[i].cursor;
    buckets[i].cursor = buckets[i].start;
  }

  for (iree_host_size_t i = 0; i < bucket_count; ++i) {
    const iree_host_size_t end =
        i + 1 < bucket_count ? buckets[i + 1].start : event_count;
    // Each swap fills one slot in a later bucket. Earlier buckets are already
    // complete, and the current cursor advances only once its point matches.
    while (buckets[i].cursor < end) {
      loom_liveness_event_t* event = &events[buckets[i].cursor];
      if (event->point == i) {
        ++buckets[i].cursor;
      } else {
        loom_liveness_event_t* destination =
            &events[buckets[event->point].cursor++];
        const loom_liveness_event_t temporary = *event;
        *event = *destination;
        *destination = temporary;
      }
    }
    loom_liveness_events_compare_sort(events + buckets[i].start,
                                      end - buckets[i].start);
  }
  return iree_ok_status();
}
