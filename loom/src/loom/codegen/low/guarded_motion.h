// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Guarded memory-operand preparation over a retained native schedule.

#ifndef LOOM_CODEGEN_LOW_GUARDED_MOTION_H_
#define LOOM_CODEGEN_LOW_GUARDED_MOTION_H_

#include "loom/codegen/low/schedule/types.h"
#include "loom/rewrite/rewriter.h"

#ifdef __cplusplus
extern "C" {
#endif

// A contiguous, speculatable prefix feeding the first memory operation in a
// single-predecessor guarded block.
typedef struct loom_low_guarded_motion_region_t {
  // First moved source node in the baseline schedule.
  uint32_t node_start;
  // Number of moved nodes in source order.
  uint32_t node_count;
  // Unmoved predecessor instruction before which the prefix is placed.
  loom_op_t* destination;
  // First unmoved source instruction, retained for exact rollback.
  loom_op_t* source;
} loom_low_guarded_motion_region_t;

// Arena-owned proposal borrowing the immutable baseline schedule. Operations
// may move while the proposal is live; the baseline table remains unchanged.
// Rejection restores source order before the baseline frame is consumed again.
typedef struct loom_low_guarded_motion_plan_t {
  // Borrowed baseline node table, including stable operation identities.
  const loom_low_schedule_node_t* nodes;
  // Proposed independent prefixes in block order.
  loom_low_guarded_motion_region_t* regions;
  // Number of entries in |regions|.
  uint32_t region_count;
  // Blocks whose instructions and local dependencies change under the plan.
  iree_bitmap_t changed_blocks;
} loom_low_guarded_motion_plan_t;

// Proposes legal guarded-prefix movement, without mutating the function.
// A coarse timing bound rejects prefixes that cannot fit in the predecessor.
// Acceptance still requires a complete scheduled and allocated trial;
// unchanged blocks may retain their accepted schedules.
iree_status_t loom_low_guarded_motion_plan(
    const loom_low_schedule_table_t* schedule, iree_arena_allocator_t* arena,
    loom_low_guarded_motion_plan_t* out_plan);

// Moves the proposed prefixes to their retained predecessor anchors.
iree_status_t loom_low_guarded_motion_apply(
    loom_rewriter_t* rewriter, const loom_low_guarded_motion_plan_t* plan);

// Restores every proposed prefix in original source order.
iree_status_t loom_low_guarded_motion_rollback(
    loom_rewriter_t* rewriter, const loom_low_guarded_motion_plan_t* plan);

// Requires a strict improvement in at least one block and no increase in any
// block's modeled issue extent. This bounds both outcomes of every guard;
// acceptance does not assume a branch probability. Allocation economics are
// checked separately by the emission-frame owner.
bool loom_low_guarded_motion_improves_schedule(
    const loom_low_schedule_table_t* baseline,
    const loom_low_schedule_table_t* trial);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_GUARDED_MOTION_H_
