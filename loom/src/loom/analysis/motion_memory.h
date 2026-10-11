// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Retained write and ordering evidence for source loop motion.

#ifndef LOOM_ANALYSIS_MOTION_MEMORY_H_
#define LOOM_ANALYSIS_MOTION_MEMORY_H_

#include "loom/analysis/motion.h"

#ifdef __cplusplus
extern "C" {
#endif

// Captures writes and ordering effects once in movement's function region tree.
// Nested loop bodies name slices of the same effect sequence. Roots and byte
// footprints come from movement's view-region and symbolic-expression owners,
// on the first query that can cross the retained ordering effects. All storage
// belongs to movement's arena.
//
// The snapshot survives relocation of ordinary reads and effect-free subtrees
// that preserve operand identities and region block topology. Writes, fences,
// unknown effects, and their enclosing loop scopes must remain fixed.
// Rewriting any of those invalidates the snapshot; availability decisions are
// never cached.
iree_status_t loom_motion_memory_create(loom_movement_analysis_t* movement,
                                        loom_motion_memory_t** out_memory);

// Checks the retained effects of loop's body against a candidate read already
// proven available before the loop. This query neither traverses source IR nor
// reconstructs write footprints. Byte-range proofs use shared symbolic scratch.
iree_status_t loom_motion_memory_evaluate_loop(
    loom_motion_memory_t* memory, const loom_op_t* loop,
    const loom_movement_endpoint_t* read,
    loom_motion_loop_hoist_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_MOTION_MEMORY_H_
