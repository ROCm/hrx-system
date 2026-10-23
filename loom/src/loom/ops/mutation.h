// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Operation erasure, structural removal, and bulk value replacement.
// These edits preserve the module's indexed references and semantic summaries.

#ifndef LOOM_OPS_MUTATION_H_
#define LOOM_OPS_MUTATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true if every result of |op| has zero operand uses and no attribute
// or value type references from outside |op|. References carried by |op|'s
// own attributes or result types do not keep the whole op alive.
bool loom_op_results_unused(const loom_module_t* module, const loom_op_t* op);

// Returns true if |op| is trivially dead: it has results, does not
// write to any resource, has no unknown effects, and every result is
// unused. Read-only and non-deterministic ops without writes are dead
// when unused — a read with no observer is a no-op. Symbols and hints are
// retained independently of their SSA result uses.
bool loom_op_is_trivially_dead(const loom_module_t* module,
                               const loom_op_t* op);

//===----------------------------------------------------------------------===//
// Operation and region mutation
//===----------------------------------------------------------------------===//

// Removes selected results from a variadic-result op and compacts trailing
// storage in-place.
//
// |remove_results| has one entry per current result. Removed result values must
// have no operand, attribute, or incoming type uses. Dropped values remain in
// the module value table but no longer carry defining-op identity or outgoing
// type-use records. Kept result values keep their IDs and receive updated
// definition indices. Tied results targeting removed result slots are rejected;
// kept tied result indices are remapped.
iree_status_t loom_op_remove_results(loom_module_t* module, loom_op_t* op,
                                     const bool* remove_results,
                                     iree_arena_allocator_t* scratch_arena,
                                     uint16_t* out_removed_count);

// Erases an op after verifying that every result has no operand uses or
// attribute/type uses from outside the op (caller must RAUW results first).
// Removes operand and attribute use records, drops type-use records carried by
// results, owned declaration arguments and nested block arguments, then marks
// the op dead. Result values retain their defining op and result index as
// immutable producer provenance. The remaining operation payload is left
// unchanged and may be inspected as immutable provenance until module
// destruction. Dead ops are skipped by enumeration macros and will not be
// serialized. The memory is not freed (arena-owned). Returns
// IREE_STATUS_FAILED_PRECONDITION if any result still has uses.
iree_status_t loom_op_erase(loom_module_t* module, loom_op_t* op);

// Erases a producer-proven closed set of disjoint operation subtrees. Results
// may reference other members, but no live operation or type outside the set
// may use a removed value. The consuming rewrite establishes that closure
// before calling; this path does not repeat its use/def analysis. Unlike
// repeated single-op erasure, source order need not be reverse topological.
void loom_op_erase_closed_set(loom_module_t* module, loom_op_t* const* ops,
                              iree_host_size_t count);

// Removes a closed set of non-entry blocks from |region| and compacts the
// region block table in place.
//
// |remove_blocks| must contain exactly |remove_block_count| entries, one per
// current block index in |region|. Entry block removal is rejected. Any kept op
// successor targeting a removed block is rejected. Values defined by removed
// block arguments or removed op subtrees may only have operand, type and
// attribute uses inside the removed set; callers must retarget or replace
// external uses before removing blocks. Closure and scratch-allocation failures
// leave IR unchanged and |out_removed_count| zero.
//
// Uses O(B) space in |scratch_arena| to index B removed nested blocks; flat
// block sets allocate no scratch. Validation is O(S + (B + E) log(B + 2)) for
// S inspected IR nodes/value slots and E incoming references/successors.
//
// Removed block/op/value objects remain arena-owned for diagnostics, but the
// blocks are detached from the region, their operations are marked dead, and
// block-argument identity/type-use records are dropped.
iree_status_t loom_region_remove_blocks(loom_module_t* module,
                                        loom_region_t* region,
                                        const bool* remove_blocks,
                                        uint16_t remove_block_count,
                                        iree_arena_allocator_t* scratch_arena,
                                        uint16_t* out_removed_count);

//===----------------------------------------------------------------------===//
// Value replacement
//===----------------------------------------------------------------------===//

// Replaces all uses of |old_id| with |new_id|. Walks old's operand use list,
// patches each user op's operand slot, bulk-transfers those use entries to
// new's list, and rewrites SSA references embedded in value types and operation
// attributes with one shared immutable substitution context. Each embedded
// reference owner publishes consistently with its index, but an allocation
// failure can leave earlier owners changed. Ordinary operands are transferred
// only after all embedded references succeed. No-op if old_id == new_id.
iree_status_t loom_value_replace_all_uses_with(loom_module_t* module,
                                               loom_value_id_t old_id,
                                               loom_value_id_t new_id);

// Same as replace_all_uses_with, but skips uses where the user op is
// |except_op|. This filtered form only rewrites operand slots; embedded type
// references have no user op to predicate against. Used during pattern rewrites
// where the replacement op also references the old value.
iree_status_t loom_value_replace_all_uses_except(loom_module_t* module,
                                                 loom_value_id_t old_id,
                                                 loom_value_id_t new_id,
                                                 const loom_op_t* except_op);

// Predicate-based RAUW. Replaces operand uses of |old_id| with |new_id| only
// where |predicate| returns true for the user op. Embedded type references are
// intentionally not rewritten by this filtered form.
typedef bool (*loom_use_predicate_fn)(const loom_op_t* user_op,
                                      void* user_data);
iree_status_t loom_value_replace_uses_if(loom_module_t* module,
                                         loom_value_id_t old_id,
                                         loom_value_id_t new_id,
                                         loom_use_predicate_fn predicate,
                                         void* user_data);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_MUTATION_H_
