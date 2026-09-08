// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_PIPELINE_FIRING_H_
#define LOOM_ANALYSIS_PIPELINE_FIRING_H_

#include "loom/analysis/pipeline_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// One finite frame of a resident scheduling group. Record stages execute in
// graph order for each input record. Each folded stage contributes to its own
// accumulator. Completion stages execute once, after every contribution in the
// frame, and before the next frame resets the accumulators. Combining kinds and
// floating-point permissions remain on the individual logical stages.
typedef struct loom_pipeline_firing_group_t {
  // Ordered temporal dimensions consumed by the record phase.
  loom_pipeline_plan_record_shape_t record_shape;

  // Number of complete frames per lane and pipeline activation.
  uint32_t frame_count;

  // Number of record-phase iterations in one frame. One when no stage folds.
  uint32_t records_per_frame;

  // Number of independent stage outputs accumulated across the record phase.
  uint32_t fold_count;

  // First record-phase stage in the firing plan's stage-index table.
  uint32_t stage_start;

  // Number of record-phase stages, followed immediately by completion stages.
  uint32_t record_stage_count;

  // Number of stages executed once after the frame's final accumulation.
  uint32_t completion_stage_count;
} loom_pipeline_firing_group_t;

typedef struct loom_pipeline_firing_plan_t {
  // One firing contract per group in the source pipeline plan's group order.
  const loom_pipeline_firing_group_t* groups;

  // Source stage indices, grouped by worker and phase, preserving dependency
  // order within each phase. Contains every logical stage exactly once.
  const uint32_t* stage_indices;
} loom_pipeline_firing_plan_t;

// Plans record and completion phases from the immutable concrete flow graph.
// No source IR is rescanned or mutated. Dimensions borrow the source plan's
// arena; all new storage belongs to |arena|. The source plan must outlive the
// firing plan. Independent cadences and nested frame reductions require a
// richer firing schedule and are rejected here, not guessed by a materializer.
iree_status_t loom_pipeline_firing_plan_build(
    const loom_pipeline_plan_t* plan, iree_arena_allocator_t* arena,
    loom_pipeline_firing_plan_t* out_firing);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_FIRING_H_
