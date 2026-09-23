// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pressure accounting over canonical per-value live segments.

#ifndef LOOM_ANALYSIS_LIVENESS_PRESSURE_H_
#define LOOM_ANALYSIS_LIVENESS_PRESSURE_H_

#include "loom/analysis/liveness.h"

#ifdef __cplusplus
extern "C" {
#endif

// Computes region-tree pressure from finalized intervals, segment ranges, and
// block points. These are trusted liveness results: each value's segments are
// nonempty, increasing, nonoverlapping, and covered by its interval.
//
// Summaries retain the first maximum of (live units, live values). Their order
// follows each class's first live point, with SSA identity breaking ties.
// Segment pressure identifies the containing block but not a particular op.
// Returned summaries belong to result_arena.
//
// Dense point/class domains accumulate endpoint differences without sorting.
// Their scratch table is bounded by the equivalent endpoint array's size;
// sparse domains sort endpoints instead. Classification is shared by both
// paths and indexed once per live value. Scratch may be discarded on return.
// Failures are allocation failure or a pressure/event count exceeding its
// representation.
iree_status_t loom_liveness_compute_segment_pressure(
    const loom_liveness_analysis_t* analysis,
    iree_arena_allocator_t* scratch_arena, iree_arena_allocator_t* result_arena,
    const loom_liveness_pressure_summary_t** out_summaries,
    iree_host_size_t* out_summary_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_LIVENESS_PRESSURE_H_
