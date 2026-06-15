// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Dominance queries for loom IR.
//
// For structured regions (no LOOM_REGION_INSTANCE_FLAG_CFG), dominance
// is read directly from the IR structure via parent_op chains — no
// precomputation required. Each query walks the ancestry of the two
// ops to determine their relationship:
//
//   Same block:      a dominates b if a comes before b (op ordinal).
//   Ancestor scope:  an op in an enclosing region dominates all ops
//                    in nested regions, unless an isolated-from-above
//                    boundary intervenes.
//   Entry block:     block 0 of a multi-block region dominates all
//                    sibling blocks.
//
// For CFG regions, the dominance info struct caches a computed block graph and
// dominator tree from predecessor edges. The query API is identical — callers
// don't need to know which strategy is in use.
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
//   - Entry block dominance: an op in block 0 of a multi-block region
//     dominates ops in sibling blocks of the same region.
//
// Both ops must have valid parent_op/parent_block pointers (set by
// the builder or loom_module_compute_uses).
bool loom_dominates_op(const loom_dominance_info_t* info, const loom_op_t* a,
                       const loom_op_t* b);

// Does block |a| dominate block |b| within their shared region?
//
// For CFG regions this uses the cached dominator tree. For structured
// multi-block regions this follows the region entry-block rule. A block
// dominates itself. Blocks from different regions never dominate each other.
bool loom_dominates_block(const loom_dominance_info_t* info,
                          const loom_block_t* a, const loom_block_t* b);

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
bool loom_value_is_available_before_op(const loom_dominance_info_t* info,
                                       loom_value_id_t value_id,
                                       const loom_op_t* before_op);

// Returns true if every SSA value referenced by |type| can be referenced by an
// op inserted immediately before |before_op|.
//
// This walks dynamic dimensions, dynamic pool sizes, SSA encodings/layouts, and
// nested function/dialect type parameters. Static types with no SSA references
// are always available.
bool loom_type_is_available_before_op(const loom_dominance_info_t* info,
                                      loom_type_t type,
                                      const loom_op_t* before_op);

// Returns true if the type of |value_id| can be materialized immediately before
// |before_op|. Invalid value IDs are treated as unavailable.
bool loom_value_type_is_available_before_op(const loom_dominance_info_t* info,
                                            loom_value_id_t value_id,
                                            const loom_op_t* before_op);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_DOMINANCE_H_
