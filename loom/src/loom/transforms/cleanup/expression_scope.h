// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Expression identity and dominance-scoped availability.
//
// Exact structural identity is the default. Semantic combiners may supply a
// compatible equality relation and hash while reusing the same block traversal
// and rollback index. Callers own eligibility policy and metadata. No target
// descriptor or physical register policy lives here.

#ifndef LOOM_TRANSFORMS_CLEANUP_EXPRESSION_SCOPE_H_
#define LOOM_TRANSFORMS_CLEANUP_EXPRESSION_SCOPE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/util/dominance.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_expression_walk_t loom_expression_walk_t;
typedef struct loom_expression_scope_t loom_expression_scope_t;

// Returns the caller-owned semantic identity for one SSA value. The returned
// key need not name a live value, but equal keys must mean equal value content
// for the lifetime of the expression walk.
typedef loom_value_id_t (*loom_expression_value_identity_fn_t)(
    void* user_data, loom_value_id_t value_id);

// Compares the semantic identity of two operations for one availability walk.
// The caller's hash must describe the same identity relation.
typedef bool (*loom_expression_identity_equal_fn_t)(void* user_data,
                                                    const loom_module_t* module,
                                                    const loom_op_t* lhs,
                                                    const loom_op_t* rhs);

typedef struct loom_expression_identity_policy_t {
  // Opaque state borrowed for the lifetime of the walk.
  void* user_data;
  // Optional semantic equality. NULL selects exact structural equality.
  loom_expression_identity_equal_fn_t equal;
} loom_expression_identity_policy_t;

// One available expression. The walk owns the record; leaving its dominance
// scope invalidates borrowed pointers to it.
typedef struct loom_expression_entry_t {
  // Live producer.
  loom_op_t* op;
  // Structural identity hash, including result types and attribute contents.
  uint32_t hash;
  // Caller-owned metadata retained with the available definition.
  uint32_t metadata;
} loom_expression_entry_t;

typedef struct loom_expression_cursor_t {
  // Walk owning the shared availability index.
  loom_expression_walk_t* walk;
  // Module containing the traversal.
  loom_module_t* module;
  // Dominance scope containing the operation.
  loom_expression_scope_t* scope;
  // Current live operation, which may be replaced and erased by the caller.
  loom_op_t* op;
  // Effective traits evaluated before visiting nested regions.
  loom_trait_flags_t traits;
  // Number of value-producing operations visited before this operation.
  uint32_t epoch;
} loom_expression_cursor_t;

enum loom_expression_lookup_flag_bits_e {
  // Stops at joins and loop headers that can observe another state epoch.
  LOOM_EXPRESSION_LOOKUP_FLAG_STATEFUL = 1u << 0,
};
typedef uint8_t loom_expression_lookup_flags_t;

// Execution and memory state shared by source and target-low expressions.
typedef struct loom_expression_barriers_t {
  // Definition frontier at the latest dynamic participant change.
  uint32_t convergence;
  // Definition frontier at the latest observable memory write.
  uint32_t memory;
} loom_expression_barriers_t;

// Allocates a root-region walk and its dominance scopes in |arena|. A zero
// |maximum_entry_count| bounds the shared index by value-producing operations;
// a nonzero bound lets a sparse semantic consumer avoid that storage.
// |identity_policy| may be NULL to select exact structural identity. A
// stackless inventory counts the dominance scopes.
// Index storage is allocated once; traversal links are allocated once per
// block. Erasing regionless operations preserves the walk. Changing block
// structure, successor edges or nested regions invalidates it.
iree_status_t loom_expression_walk_allocate(
    loom_module_t* module, loom_region_t* region,
    const loom_expression_identity_policy_t* identity_policy,
    uint32_t maximum_entry_count, iree_arena_allocator_t* arena,
    loom_expression_walk_t** out_walk);

// Visits in dominator-before-dominated order, returning a null op at
// completion. Nested regions are visited after their owner. Isolated and
// independently executing regions hide parent expressions; mutable-state
// lookups cross only straight-line CFG edges.
iree_status_t loom_expression_walk_next(loom_expression_walk_t* walk,
                                        loom_expression_cursor_t* out_cursor);

// Returns the dominance snapshot shared by the ordered walk.
const loom_dominance_info_t* loom_expression_walk_dominance_info(
    const loom_expression_walk_t* walk);

// Hashes regionless operations by their exact SSA structural identity.
uint32_t loom_expression_hash(const loom_module_t* module, const loom_op_t* op);

// Hashes regionless operations after replacing each operand with the
// caller-indexed semantic value identity. All non-operand structure remains
// exact. |value_identity| must be paired with the matching equality helper.
uint32_t loom_expression_hash_with_value_identity(
    const loom_module_t* module, const loom_op_t* op,
    loom_expression_value_identity_fn_t value_identity, void* user_data);

// Compares regionless operations by kind, exact SSA operands, result types,
// attribute contents, operand segments, and instance flags.
bool loom_expression_equal(const loom_module_t* module, const loom_op_t* a,
                           const loom_op_t* b);

// Compares operations after replacing each operand with the caller-indexed
// semantic value identity. All non-operand structure remains exact.
bool loom_expression_equal_with_value_identity(
    const loom_module_t* module, const loom_op_t* a, const loom_op_t* b,
    loom_expression_value_identity_fn_t value_identity, void* user_data);

// Finds the nearest equivalent producer visible from |cursor| in the shared
// index, without walking enclosing scopes. The caller-supplied hash must match
// the walk's identity policy.
loom_expression_entry_t* loom_expression_scope_find(
    const loom_expression_cursor_t* cursor, uint32_t hash,
    loom_expression_lookup_flags_t flags);

// Inserts at most one entry for the current value-producing operation,
// replacing an older equivalent entry in this block. Scope exit restores the
// prior bindings; no table clearing or allocation occurs during lookup/update.
void loom_expression_scope_insert(const loom_expression_cursor_t* cursor,
                                  loom_expression_entry_t entry);

// Records the current FactIdentity operation as the preferred representative
// of each corresponding operand within the current dominance scope. Storage is
// allocated lazily on the first recorded identity; walks without fact
// identities retain only a null state pointer. Nested refinements chain to the
// original value and are restored when their dominance scope exits.
iree_status_t loom_expression_scope_record_fact_identity(
    const loom_expression_cursor_t* cursor);

// Returns the strongest recorded fact identity for |value_id| visible from the
// current cursor, or |value_id| when no type-compatible refinement is active.
// Results with ownership-sensitive uses are not shared.
loom_value_id_t loom_expression_scope_preferred_value(
    const loom_expression_cursor_t* cursor, loom_value_id_t value_id);

// Advances shared effect epochs and returns a conservative minimum reuse epoch.
// PURE excludes memory-write barriers but retains convergence barriers for
// machine materializations with implicit execution state. Source-only callers
// can ignore the returned epoch for eligible pure SSA expressions. Nested
// effects apply before entering regions.
uint32_t loom_expression_observe_barriers(
    const loom_expression_cursor_t* cursor,
    loom_expression_barriers_t* barriers);

// Tests semantic eligibility: deterministic, regionless, no observable writes,
// convergence, unique identity, or linear ownership transfer/consumption.
bool loom_expression_op_is_reusable(const loom_module_t* module,
                                    const loom_op_t* op,
                                    loom_trait_flags_t traits);

// Tests whether sharing results preserves ownership: the op neither transfers
// operand ownership into results nor produces results consumed by a later move
// or tied update, or shared through a required-identity use. Effect and
// structural eligibility remain the caller's policy.
bool loom_expression_can_share_result_ownership(const loom_module_t* module,
                                                const loom_op_t* op);

// Replaces corresponding result uses and erases the redundant operation.
// Ordinary operand consumers use the strongest dominating same-typed fact
// identity. FactIdentity consumers and embedded SSA references use the
// canonical producer so repeated identity refinements remain CSE-equivalent.
iree_status_t loom_expression_replace(const loom_expression_cursor_t* cursor,
                                      loom_op_t* op, loom_op_t* existing);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_CLEANUP_EXPRESSION_SCOPE_H_
