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
//      contracts: effect-free relocation requires PURE, while true speculation
//      additionally requires SAFE_TO_SPECULATE. Convergent ops reject both
//      policies because their dynamic participant set is part of their
//      semantics. Pending source expansion also retains its control-flow
//      context independently of purity: branch facts may determine which
//      implementation is applicable. Erasing unused pure work remains legal.
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
#include "loom/analysis/movement.h"
#include "loom/analysis/read_motion.h"
#include "loom/ir/ir.h"
#include "loom/ir/local_value_domain.h"
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

typedef struct loom_motion_memory_t loom_motion_memory_t;

typedef struct loom_motion_analysis_t {
  // Module containing the IR being queried.
  const loom_module_t* module;
  // Scratch arena used for temporary traversal stacks.
  iree_arena_allocator_t* arena;
  // Function facts used for counted-loop and symbolic memory proofs.
  loom_value_fact_table_t* fact_table;
  // Active function-local value domain extended by movement analysis.
  loom_local_value_domain_t* value_domain;
  // Value, type, and attribute capture availability queries.
  loom_availability_analysis_t availability;
  // Lazily populated source memory movement analysis.
  loom_movement_analysis_t movement;
  // True when movement owns initialized analysis storage.
  bool movement_initialized;
  // Demand-built write and ordering snapshot, retained across legal hoists.
  loom_motion_memory_t* memory;
  // Reusable stack for subtree region walks.
  loom_motion_region_stack_t region_stack;
} loom_motion_analysis_t;

// Lazily constructed index of the block-local effect segments that ordinary
// reads cannot cross.
typedef struct loom_motion_read_barrier_table_t
    loom_motion_read_barrier_table_t;

// Initializes shared motion analysis state for |region| and its nested region
// tree. All queried insertion points and CFG relationships must remain in that
// tree. |fact_table| and |value_domain| may both be NULL for effect-free
// relocation clients. Memory-aware loop placement requires both. The caller
// owns all borrowed state and must keep it live for the analysis lifetime.
// Memory queries retain write/ordering evidence. Moving ordinary reads and
// effect-free subtrees preserves that evidence. Changing effects, operands,
// their enclosing loop scopes, or region block topology requires a new
// analysis instance. Availability is queried at each proposed insertion point
// rather than cached across moves.
iree_status_t loom_motion_analysis_initialize_region(
    const loom_module_t* module, const loom_region_t* region,
    loom_value_fact_table_t* fact_table,
    loom_local_value_domain_t* value_domain, iree_arena_allocator_t* arena,
    loom_motion_analysis_t* out_analysis);

//===----------------------------------------------------------------------===//
// Loop hoist evaluation
//===----------------------------------------------------------------------===//

typedef uint32_t loom_motion_loop_hoist_rejection_flags_t;

#define LOOM_MOTION_LOOP_HOIST_REJECTION_INVALID_REQUEST ((uint32_t)1u << 0)
#define LOOM_MOTION_LOOP_HOIST_REJECTION_CANDIDATE_SEMANTICS ((uint32_t)1u << 1)
#define LOOM_MOTION_LOOP_HOIST_REJECTION_CAPTURE_UNAVAILABLE ((uint32_t)1u << 2)
#define LOOM_MOTION_LOOP_HOIST_REJECTION_PREDICATE_CROSSING ((uint32_t)1u << 3)
#define LOOM_MOTION_LOOP_HOIST_REJECTION_LOOP_MAY_NOT_EXECUTE \
  ((uint32_t)1u << 4)
#define LOOM_MOTION_LOOP_HOIST_REJECTION_MOVEMENT_UNAVAILABLE \
  ((uint32_t)1u << 5)
#define LOOM_MOTION_LOOP_HOIST_REJECTION_UNKNOWN_INTERFERENCE \
  ((uint32_t)1u << 6)
#define LOOM_MOTION_LOOP_HOIST_REJECTION_OVERLAPPING_WRITE ((uint32_t)1u << 7)
#define LOOM_MOTION_LOOP_HOIST_REJECTION_ORDERING_INTERFERENCE \
  ((uint32_t)1u << 8)

// Stable legality result consumed by LICM and the placement planner.
typedef struct loom_motion_loop_hoist_result_t {
  // Bitset of loom_motion_loop_hoist_rejection_flags_t values. Zero means the
  // motion is legal.
  loom_motion_loop_hoist_rejection_flags_t rejection_bits;
  // Movement classifier details when MOVEMENT_UNAVAILABLE is set.
  loom_movement_rejection_flags_t movement_rejection_bits;
  // Operation proving the rejection, or NULL when no op applies.
  const loom_op_t* blocking_op;
} loom_motion_loop_hoist_result_t;

static inline bool loom_motion_loop_hoist_result_is_legal(
    const loom_motion_loop_hoist_result_t* result) {
  return result && result->rejection_bits == 0;
}

// Evaluates moving |candidate_op| immediately before |loop.op|. Effect-free
// subtrees must be safe to speculate because the loop may execute zero times.
// Ordinary unmasked view/vector loads may instead use a proven-nonempty loop
// contract when every write in the loop is proven disjoint from the loaded byte
// region. Loads nested under another predicate remain in place.
iree_status_t loom_motion_subtree_evaluate_hoist_before_loop(
    loom_motion_analysis_t* analysis, loom_loop_like_t loop,
    const loom_op_t* candidate_op, loom_motion_loop_hoist_result_t* out_result);

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
// hints, convergence, pending source expansion, unknown effects, reads, writes,
// and nondeterminism. It intentionally does not reject UNIQUE_IDENTITY beyond
// the ordinary purity/effect checks: relocating one execution site without
// duplicating it is not CSE.
bool loom_motion_op_can_relocate_effect_free(const loom_module_t* module,
                                             const loom_op_t* op);

// Returns true if |op| may be rebuilt independently while preserving or
// narrowing its original dynamic execution predicate. This requires pure,
// deterministic, identity-free semantics and rejects successors, retained
// regions, terminators, hints, poison boundaries, convergence, callable
// boundaries, and pending source expansion. Call purity does not describe
// returned identity or execution cost. This does not require SAFE_TO_SPECULATE
// because rematerialization must not introduce execution on an additional
// control path.
bool loom_motion_op_can_rematerialize_effect_free(const loom_module_t* module,
                                                  const loom_op_t* op);

// Returns true if |op| may be executed on additional control paths. This
// requires SAFE_TO_SPECULATE and rejects any region side effects, convergence,
// hints, or pending source expansion.
bool loom_motion_op_can_speculate(const loom_module_t* module,
                                  const loom_op_t* op);

// Returns true when |op| is an ordinary unmasked view or vector load with no
// write, ordering, convergence, identity, or nondeterministic semantics.
bool loom_motion_op_is_ordinary_load(const loom_module_t* module,
                                     const loom_op_t* op);

// Creates a lazy read-barrier table using |arena| storage. The first query in a
// block captures source-order segments for every operation then live in that
// block. Later-created operations are indexed once as contiguous runs when
// their surrounding captured segments agree; conflicting runs conservatively
// fail queries. Rewrites may retain the table while replacing operations in
// place, but must not move captured operations across observable effect
// boundaries.
iree_status_t loom_motion_read_barrier_table_create(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_motion_read_barrier_table_t** out_table);

// Returns whether an ordinary read at |read_op| can move immediately before
// |before_op| without crossing an ordering barrier. Both operations must be in
// the same block with |read_op| ordered first. The query checks operations
// strictly between the endpoints; callers separately classify the read and
// account for effects of |before_op| itself.
iree_status_t loom_motion_read_barrier_table_can_cross(
    loom_motion_read_barrier_table_t* table, const loom_op_t* read_op,
    const loom_op_t* before_op, bool* out_can_cross);

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
