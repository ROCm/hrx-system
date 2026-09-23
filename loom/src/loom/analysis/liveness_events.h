// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Ordered live-range endpoints for deterministic register-pressure sweeps.

#ifndef LOOM_ANALYSIS_LIVENESS_EVENTS_H_
#define LOOM_ANALYSIS_LIVENESS_EVENTS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// One endpoint of a nonempty, half-open live segment. The analysis emits a
// start and an end for each segment, retaining its canonical value ordinal.
typedef struct loom_liveness_event_t {
  // Program point where the live set changes.
  uint32_t point;
  // SSA identity used to order simultaneous events deterministically.
  loom_value_id_t value_id;
  // Retained index of the interval entering or leaving the live set.
  loom_value_ordinal_t value_ordinal;
  // Value-count delta: -1 at a segment end, +1 at a segment start.
  int8_t value_delta;
} loom_liveness_event_t;

static_assert(sizeof(loom_liveness_event_t) == 16,
              "liveness events must remain compact for sorting");

// Orders events in place by point, end before start, then SSA identity.
// This order also makes pressure-class discovery and peak selection stable.
//
// The producer supplies |maximum_point| while constructing the endpoints;
// every event point must be at most that bound. Dense point domains use arena
// scratch bounded by the event array's byte size. Small and sparse domains
// use an allocation-free comparison sort. Scratch may be discarded on return.
// The only possible failure is scratch allocation.
iree_status_t loom_liveness_events_sort(loom_liveness_event_t* events,
                                        iree_host_size_t event_count,
                                        uint32_t maximum_point,
                                        iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_LIVENESS_EVENTS_H_
