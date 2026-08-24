// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_PROFILE_CLOCK_ALIGNMENT_H_
#define IREE_HAL_UTILS_PROFILE_CLOCK_ALIGNMENT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Tracks whether calibrated device clock samples bound device event ticks.
//
// Clock samples and event timestamps intended for the same device timeline
// must share an epoch. A producer records its first clock sample before device
// work, records event ranges as they are harvested, and records another clock
// sample after each harvest. Once an event falls outside the sampled clock
// range the alignment is permanently invalid for the session.
//
// The tracker owns no synchronization. Callers must serialize all mutations or
// provide an external lock shared by clock-sampling and event-harvest paths.
typedef struct iree_hal_profile_clock_alignment_t {
  // Earliest calibrated device clock tick observed during the session.
  uint64_t minimum_clock_tick;

  // Latest calibrated device clock tick observed during the session.
  uint64_t maximum_clock_tick;

  // Earliest device event tick observed during the session.
  uint64_t minimum_event_tick;

  // Latest device event tick observed during the session.
  uint64_t maximum_event_tick;

  // True when at least one calibrated device clock tick has been observed.
  bool has_clock_ticks;

  // True when at least one device event tick has been observed.
  bool has_event_ticks;

  // True when an event range is malformed or outside sampled clock ticks.
  bool has_invalid_alignment;
} iree_hal_profile_clock_alignment_t;

// Resets |alignment| for a new profiling session.
void iree_hal_profile_clock_alignment_reset(
    iree_hal_profile_clock_alignment_t* alignment);

// Records one inclusive device event tick range.
//
// A reversed range permanently invalidates alignment for the session. A valid
// range is not compared until a subsequent clock sample is recorded, allowing
// the ending sample to extend the clock bounds beyond newly harvested events.
// A range may summarize any number of events; batch-oriented producers should
// coalesce each harvest and record its minimum and maximum ticks once.
void iree_hal_profile_clock_alignment_record_event_range(
    iree_hal_profile_clock_alignment_t* alignment, uint64_t start_tick,
    uint64_t end_tick);

// Records one calibrated device clock tick.
//
// Returns true when alignment has been invalidated at any point in the active
// session.
bool iree_hal_profile_clock_alignment_record_clock_tick(
    iree_hal_profile_clock_alignment_t* alignment, uint64_t clock_tick);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_PROFILE_CLOCK_ALIGNMENT_H_
