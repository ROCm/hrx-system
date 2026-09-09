// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structural expression identity and dominance-scoped availability.
//
// This utility owns expression keys, block traversal and replacement mechanics.
// Callers own eligibility policy and the epochs that make a previous definition
// reusable. No target descriptor or physical register policy lives here.

#ifndef LOOM_TRANSFORMS_CLEANUP_EXPRESSION_SCOPE_H_
#define LOOM_TRANSFORMS_CLEANUP_EXPRESSION_SCOPE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_expression_walk_t loom_expression_walk_t;
typedef struct loom_expression_scope_t loom_expression_scope_t;

// One previously visited expression. The scope owns the record.
typedef struct loom_expression_entry_t {
  // Live producer; NULL marks an unused table slot.
  loom_op_t* op;
  // Structural identity hash, including result types and attribute contents.
  uint32_t hash;
  // Definition frontier at insertion. Earlier definitions have smaller epochs.
  uint32_t epoch;
} loom_expression_entry_t;

typedef struct loom_expression_cursor_t {
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

// Initializes a root-region walk and its dominance scopes in |arena|. Scope
// tables and traversal links are allocated once per block; there is no growing
// traversal array. Erasing regionless operations preserves the walk. Changing
// block structure, successor edges or nested regions invalidates it.
iree_status_t loom_expression_walk_initialize(
    loom_module_t* module, loom_region_t* region, iree_arena_allocator_t* arena,
    loom_expression_walk_t** out_walk);

// Visits in dominator-before-dominated order, returning a null op at
// completion. Nested regions are visited after their owner. Isolated regions
// hide parent expressions; mutable-state lookups cross only straight-line CFG
// edges.
iree_status_t loom_expression_walk_next(loom_expression_walk_t* walk,
                                        loom_expression_cursor_t* out_cursor);

// Hashes regionless operations by their exact SSA structural identity.
uint32_t loom_expression_hash(const loom_module_t* module, const loom_op_t* op);

// Finds the nearest structurally equivalent producer visible from |cursor|.
// Callers compare its epoch with their own invalidation epochs.
loom_expression_entry_t* loom_expression_scope_find(
    const loom_expression_cursor_t* cursor, uint32_t hash,
    loom_expression_lookup_flags_t flags);

// Inserts in the current block, replacing an older equivalent entry there.
void loom_expression_scope_insert(const loom_expression_cursor_t* cursor,
                                  loom_expression_entry_t entry);

// Advances shared effect epochs and returns the current expression's minimum
// reusable epoch. PURE expressions are independent of memory writes but not
// convergent participant changes. Nested effects apply before entering regions.
uint32_t loom_expression_observe_barriers(
    const loom_expression_cursor_t* cursor,
    loom_expression_barriers_t* barriers);

// Tests semantic eligibility: deterministic, regionless, no observable writes,
// convergence, unique identity, or linear ownership transfer/consumption.
bool loom_expression_is_reusable(const loom_expression_cursor_t* cursor);

// Replaces all corresponding result uses and erases the redundant operation.
iree_status_t loom_expression_replace(loom_module_t* module, loom_op_t* op,
                                      loom_op_t* existing);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_CLEANUP_EXPRESSION_SCOPE_H_
