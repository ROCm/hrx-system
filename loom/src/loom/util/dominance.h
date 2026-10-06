// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Dominance queries for loom IR.
//
// For single-block structured regions, dominance
// is read directly from the IR structure via parent_op chains — no
// precomputation required. Each query walks the ancestry of the two
// ops to determine their relationship:
//
//   Same block:      a dominates b if a comes before b (op ordinal).
//   Ancestor scope:  an op in an enclosing region dominates all ops
//                    in nested regions, unless an isolated-from-above
//                    boundary intervenes.
//
// For CFG or multi-block regions, the dominance info struct caches a block
// graph and the shared CFG dominator tree. Entry dominates reachable blocks;
// unreachable blocks dominate only themselves. Region caches cover the entire
// requested region tree. Bulk single-region consumers use cfg_dominance.h
// directly to retain indexed queries without cache or ancestry lookup.
//
// Usage:
//
//   loom_dominance_info_t dom_info;
//   IREE_RETURN_IF_ERROR(
//       loom_dominance_info_initialize(module, &arena, &dom_info));
//
//   if (loom_dominates_op(&dom_info, producer, consumer)) {
//     // producer's results are visible at consumer.
//   }
//
//   if (loom_dominates_value(&dom_info, value_id, use_op)) {
//     // value_id is in scope at use_op.
//   }

#ifndef LOOM_UTIL_DOMINANCE_H_
#define LOOM_UTIL_DOMINANCE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dominance info
//===----------------------------------------------------------------------===//

typedef struct loom_cfg_dominance_region_t loom_cfg_dominance_region_t;
typedef struct loom_cfg_graph_t loom_cfg_graph_t;
typedef struct loom_cfg_dominance_t loom_cfg_dominance_t;

// Dominance analysis state. For structured IR this is lightweight: queries walk
// parent_op chains and block order directly. For CFG regions, initialization
// caches block graphs and dominator trees allocated from the caller arena.
typedef struct loom_dominance_info_t {
  // Module containing the IR being queried.
  const loom_module_t* module;
  // Scratch arena that owns CFG dominance caches.
  iree_arena_allocator_t* arena;
  // Linked list of cached CFG-region dominance records.
  loom_cfg_dominance_region_t* cfg_regions;
} loom_dominance_info_t;

// Initializes dominance info for the given module. The caller must keep |arena|
// and |module| live and must rebuild the analysis after mutating region block
// structure or successor edges.
iree_status_t loom_dominance_info_initialize(const loom_module_t* module,
                                             iree_arena_allocator_t* arena,
                                             loom_dominance_info_t* out_info);

// Initializes dominance info for |region| and its nested region tree.
//
// All queries requiring CFG dominance must remain within that region tree.
// Structured ancestry and same-block order queries remain allocation-free, but
// querying a CFG region outside the initialized scope conservatively reports no
// dominance. The caller must rebuild the analysis after mutating scoped region
// block structure or successor edges; moving operations without changing CFG
// topology leaves the analysis valid.
iree_status_t loom_dominance_info_initialize_region(
    const loom_module_t* module, const loom_region_t* region,
    iree_arena_allocator_t* arena, loom_dominance_info_t* out_info);

// Adds dominators for an existing graph without extracting its structure again.
// A caller with precomputed graphs starts with {.module = module, .arena =
// arena} and adds each relevant region once before querying dominance. An
// optional |dominance| borrows an already-computed dominator tree; NULL builds
// it in the info arena. Borrowed graph and dominator arrays must outlive all
// queries. Region structure and successor edges must remain unchanged.
iree_status_t loom_dominance_info_add_cfg_graph(
    loom_dominance_info_t* info, const loom_cfg_graph_t* graph,
    const loom_cfg_dominance_t* dominance);

//===----------------------------------------------------------------------===//
// Dominance queries
//===----------------------------------------------------------------------===//

// Returns the nesting depth of |op| by counting parent_op hops.
// Top-level ops (parent_op == NULL) have depth 0.
uint16_t loom_op_nesting_depth(const loom_op_t* op);

// Returns the ancestor of |op| at the given |target_depth|.
// Walks parent_op pointers |depth - target_depth| times.
// Returns |op| itself if already at the target depth.
const loom_op_t* loom_op_ancestor_at_depth(const loom_op_t* op,
                                           uint16_t target_depth);

// Does op |a| dominate op |b|?
//
// Dominance rules for structured IR:
//   - Self-dominance: a dominates a.
//   - Same block: a dominates b if a appears before b in the block's
//     ordered op list.
//   - Ancestor scope: a dominates b if a is in an enclosing scope
//     (reachable via b's parent_op chain) and no isolated-from-above
//     boundary is crossed.
//   - Different blocks: the defining block must dominate the use block in the
//     region's CFG, independently of their source order.
//
// Both ops must have valid parent_op/parent_block pointers (set by
// the builder or loom_module_compute_uses).
bool loom_dominates_op(const loom_dominance_info_t* info, const loom_op_t* a,
                       const loom_op_t* b);

// Does block |a| dominate block |b| within their shared region?
//
// Multi-block and CFG regions use the cached dominator tree. A block dominates
// itself, including when unreachable. Blocks from different regions never
// dominate each other.
bool loom_dominates_block(const loom_dominance_info_t* info,
                          const loom_block_t* a, const loom_block_t* b);

// Returns true when |block| participates in a directed cycle in its parent
// CFG. Unknown or malformed CFG structure is conservatively cyclic.
bool loom_dominance_block_is_cyclic(const loom_dominance_info_t* info,
                                    const loom_block_t* block);

// Returns the immediate dominator of |block| within its parent region.
//
// Returns NULL for region entry blocks, unreachable CFG blocks, malformed CFG
// dominance caches, blocks outside of any region, and blocks whose dominance is
// not known. Multi-block regions are treated as CFG-shaped even before cleanup
// passes stamp the region flag, because successor-bearing blocks are a
// structural property of the IR rather than a pass-order promise.
const loom_block_t* loom_dominance_immediate_dominator_block(
    const loom_dominance_info_t* info, const loom_block_t* block);

// Does the value defined by |value_id| dominate op |use_op|?
//
// For op results: equivalent to loom_dominates_op(defining_op, use_op).
// For block arguments: the value dominates all ops in its block and
// all ops in nested regions (subject to isolation boundaries).
bool loom_dominates_value(const loom_dominance_info_t* info,
                          loom_value_id_t value_id, const loom_op_t* use_op);

// Returns true if |value_id| can be referenced by an op inserted immediately
// before |before_op|.
//
// This differs from plain dominance for values defined by |before_op| itself:
// an op result self-dominates for ordinary use-site queries, but it is not
// available before the defining op. Invalid value IDs are treated as
// unavailable so analyses can be conservative on malformed IR while verifiers
// own structured user diagnostics.
//
// For ordinary SSA values in verified IR, definition-site verification makes
// every value referenced by the type visible when the carrier is defined.
// Dominance is transitive, so CFG consumers must not separately rescan the
// carrier type after this query succeeds.
bool loom_value_is_available_before_op(const loom_dominance_info_t* info,
                                       loom_value_id_t value_id,
                                       const loom_op_t* before_op);

//===----------------------------------------------------------------------===//
// Dominator-order traversal
//===----------------------------------------------------------------------===//

typedef struct loom_dominance_walk_t loom_dominance_walk_t;

enum loom_dominance_walk_scope_flag_bits_e {
  // Incoming control flow can observe a different mutable state. This is set
  // at joins, loop headers, and blocks whose sole predecessor is not their
  // immediate dominator.
  LOOM_DOMINANCE_WALK_SCOPE_FLAG_STATE_BARRIER = 1u << 0,
  // The enclosing operation hides values defined outside this region.
  LOOM_DOMINANCE_WALK_SCOPE_FLAG_ISOLATED = 1u << 1,
};
typedef uint8_t loom_dominance_walk_scope_flags_t;

typedef iree_status_t (*loom_dominance_walk_enter_scope_fn_t)(
    void* user_data, loom_dominance_walk_scope_flags_t flags);

typedef void (*loom_dominance_walk_leave_scope_fn_t)(void* user_data);

// Scope lifetime notifications for a dominator-order walk. Notifications are
// properly nested: every successful enter receives one leave, including when
// traversal completes. An enclosing block remains entered while the walk
// visits nested regions and dominating CFG descendants.
typedef struct loom_dominance_walk_callbacks_t {
  // Opaque value forwarded to both callbacks.
  void* user_data;
  // Optional callback invoked before the first operation in a dominance scope.
  loom_dominance_walk_enter_scope_fn_t enter_scope;
  // Optional callback invoked after the last dominated operation in a scope.
  loom_dominance_walk_leave_scope_fn_t leave_scope;
} loom_dominance_walk_callbacks_t;

typedef struct loom_dominance_walk_cursor_t {
  // Module containing the traversal.
  loom_module_t* module;
  // Current live operation, or NULL when traversal is complete.
  loom_op_t* op;
  // Effective traits evaluated before visiting nested regions.
  loom_trait_flags_t traits;
} loom_dominance_walk_cursor_t;

// Creates a traversal over |region| and all nested regions. Blocks are visited
// in dominator preorder and operations retain block order. Nested regions are
// visited after their owner and before its next operation. Scope callbacks let
// clients maintain rollback maps without rescanning dominated subtrees.
//
// Erasing a returned regionless operation is supported because the next
// operation is saved before returning. Changing block structure, successor
// edges, or nested regions invalidates the walk.
iree_status_t loom_dominance_walk_create(
    loom_module_t* module, loom_region_t* region,
    loom_dominance_walk_callbacks_t callbacks, iree_arena_allocator_t* arena,
    loom_dominance_walk_t** out_walk);

// Returns the next live operation. A null cursor operation denotes completion.
// Completion also leaves every remaining active scope. If an enter callback
// fails, every previously entered scope is left before the error is returned.
iree_status_t loom_dominance_walk_next(
    loom_dominance_walk_t* walk, loom_dominance_walk_cursor_t* out_cursor);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_DOMINANCE_H_
