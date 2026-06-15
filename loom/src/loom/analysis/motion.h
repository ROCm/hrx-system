// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared legality checks for IR motion.
//
// Motion legality has two independent parts:
//
//   1. Execution-safety policy. Some transforms only move work while preserving
//      the original dynamic execution predicate. Others speculate work onto
//      control paths where it may execute more often. These are distinct
//      contracts: PURE is enough for conservative effect-free relocation, while
//      true speculation additionally requires SAFE_TO_SPECULATE. Convergent ops
//      reject both policies because their dynamic participant set is part of
//      their semantics.
//
//   2. SSA availability. Moving an op also moves its result types, attributes,
//      and nested regions. Every ordinary operand and every SSA value embedded
//      in moved types, attributes, predicates, static encodings, and block
//      argument types must either be defined outside the moved subtree and
//      dominate the insertion point, or be defined inside the moved subtree
//      itself.
//
// Keep pass-specific profitability outside this file. This layer answers
// whether a proposed motion preserves IR contracts, not whether it is a good
// optimization.

#ifndef LOOM_ANALYSIS_MOTION_H_
#define LOOM_ANALYSIS_MOTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/availability.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Motion analysis state
//===----------------------------------------------------------------------===//

typedef struct loom_motion_region_stack_t {
  // Region pointers waiting to be checked.
  loom_region_t** regions;
  // Number of queued region pointers.
  iree_host_size_t count;
  // Allocated region pointer capacity.
  iree_host_size_t capacity;
} loom_motion_region_stack_t;

typedef struct loom_motion_analysis_t {
  // Module containing the IR being queried.
  const loom_module_t* module;
  // Scratch arena used for temporary traversal stacks.
  iree_arena_allocator_t* arena;
  // Value, type, and attribute capture availability queries.
  loom_availability_analysis_t availability;
  // Reusable stack for subtree region walks.
  loom_motion_region_stack_t region_stack;
} loom_motion_analysis_t;

// Initializes shared motion analysis state. The caller owns |arena| and must
// keep it live for the analysis object's lifetime.
iree_status_t loom_motion_analysis_initialize(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_motion_analysis_t* out_analysis);

//===----------------------------------------------------------------------===//
// Local classification
//===----------------------------------------------------------------------===//

// Local classification checks direct op traits and retained region summaries,
// not recursive SSA availability. Use the subtree queries below when moving an
// op with nested regions or when an insertion point is known.

// Returns true if |op| has no live uses, no retained compiler hints, and no
// semantic effects that must keep it in the IR.
bool loom_motion_op_can_erase(const loom_module_t* module, const loom_op_t* op);

// Returns true if |op| may be moved by a transform that preserves the original
// dynamic execution predicate. This rejects tied results, terminator roots,
// hints, convergence, unknown effects, reads, writes, and nondeterminism. It
// intentionally does not reject UNIQUE_IDENTITY beyond the ordinary
// purity/effect checks: relocating one execution site without duplicating it is
// not CSE.
bool loom_motion_op_can_relocate_effect_free(const loom_module_t* module,
                                             const loom_op_t* op);

// Returns true if |op| may be executed on additional control paths. This
// requires SAFE_TO_SPECULATE and rejects any region side effects, convergence,
// or hints.
bool loom_motion_op_can_speculate(const loom_module_t* module,
                                  const loom_op_t* op);

//===----------------------------------------------------------------------===//
// Subtree motion
//===----------------------------------------------------------------------===//

// Returns true if |candidate_op| and all nested ops may be relocated as a unit
// immediately before |before_op| using the conservative effect-free relocation
// policy. SSA values defined inside |candidate_op|'s subtree may be referenced
// by other moved ops; all other operand and type references must already be
// available before |before_op|.
iree_status_t loom_motion_subtree_can_relocate_before(
    loom_motion_analysis_t* analysis, const loom_op_t* candidate_op,
    const loom_op_t* before_op, bool* out_can_relocate);

// Same as loom_motion_subtree_can_relocate_before, but every non-terminator op
// in the moved subtree must be SAFE_TO_SPECULATE because the caller may execute
// the subtree more often than the source IR did.
iree_status_t loom_motion_subtree_can_speculate_before(
    loom_motion_analysis_t* analysis, const loom_op_t* candidate_op,
    const loom_op_t* before_op, bool* out_can_speculate);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_MOTION_H_
