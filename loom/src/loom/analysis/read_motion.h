// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_READ_MOTION_H_
#define LOOM_ANALYSIS_READ_MOTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_read_motion_barrier_t {
  // Operation that an ordinary read cannot cross.
  const loom_op_t* op;
} loom_read_motion_barrier_t;

// Indexed ordering boundaries for ordinary reads in one region tree.
typedef struct loom_read_motion_t {
  // Module owning the analyzed operations.
  const loom_module_t* module;
  // Boundaries sorted by block identity and then source order.
  loom_read_motion_barrier_t* barriers;
  // Number of populated ordering boundaries.
  iree_host_size_t barrier_count;
} loom_read_motion_t;

// Analyzes |region| and its nested regions once, retaining the boundaries from
// loom_motion_read_can_cross_op. All storage is owned by |arena|. The index is
// invalidated by changes to boundary semantics, membership, or relative order.
// Moving ordinary reads within their original block preserves the index.
iree_status_t loom_read_motion_analyze_region(const loom_module_t* module,
                                              loom_region_t* region,
                                              iree_arena_allocator_t* arena,
                                              loom_read_motion_t* out_analysis);

// Returns whether the ordinary |read_op| can be sunk immediately before
// |before_op| in the same block. Both operations must be in the analyzed tree.
// The query uses the retained boundary index and never traverses source IR.
// Splitting a read around the consumer's effects requires a separate proof.
bool loom_read_motion_can_sink_before(const loom_read_motion_t* analysis,
                                      const loom_op_t* read_op,
                                      const loom_op_t* before_op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_READ_MOTION_H_
