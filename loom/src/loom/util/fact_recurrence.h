// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Loop-owned additive carried-state equations. Construction visits demanded
// update chains once; numeric queries consume the retained equations.

#ifndef LOOM_UTIL_FACT_RECURRENCE_H_
#define LOOM_UTIL_FACT_RECURRENCE_H_

#include "iree/base/internal/arena.h"
#include "loom/analysis/loop_domain.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_value_fact_table_t loom_value_fact_table_t;
typedef struct loom_value_fact_recurrence_t loom_value_fact_recurrence_t;

// An equation input retains literal semantics, never a snapshot of a capture's
// inferred facts. Literal values survive resetting a cyclic solve's entries.
typedef struct loom_value_fact_recurrence_operand_t {
  // SSA input, or INVALID for an absent equation input.
  loom_value_id_t value;
  // True when the defining operation is an immutable integer literal.
  bool is_literal;
  // Numeric literal value, meaningful only when is_literal is true.
  int64_t literal;
} loom_value_fact_recurrence_operand_t;

// Retains the original input carrying applicable facts and captures literal
// semantics through same-type identities at equation construction.
loom_value_fact_recurrence_operand_t loom_value_fact_recurrence_operand_make(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value);

// Returns immutable literal facts or the input's current table facts.
loom_value_facts_t loom_value_fact_recurrence_operand_facts(
    const loom_value_fact_table_t* table,
    loom_value_fact_recurrence_operand_t operand);

// Structural scope supplied by the owning loop analysis during construction.
// The returned equations do not retain this scope or any borrowed pointers.
typedef struct loom_value_fact_recurrence_scope_t {
  // State borrowed by is_invariant for the duration of construction.
  void* user_data;
  // Returns whether a nonliteral value is invariant in the owning loop.
  bool (*is_invariant)(void* user_data, loom_value_id_t value);
  // Optional condition-loop body whose arguments forward header values.
  const loom_block_t* forwarding_block;
  // Header values corresponding to forwarding_block's arguments.
  loom_value_slice_t forwarding_values;
} loom_value_fact_recurrence_scope_t;

// Solve-owned equations indexed by carried-state slot, with storage only for
// recognized scalar translations. Published value facts do not borrow these
// equations; the owner releases the set's storage when the solve ends.
typedef struct loom_value_fact_recurrence_set_t {
  // Sparse records indexed by slot, or NULL when none were recognized.
  const loom_value_fact_recurrence_t** records;
} loom_value_fact_recurrence_set_t;

// Constructs equations for header arguments beginning at argument_offset.
// Initial and next values contain one value per carried-state slot. These are
// verified loop-boundary tuples supplied by the owner. Only same-type scalar
// add/sub translations of the corresponding header argument are retained.
// Scratch is released before return; retained records have arena lifetime.
iree_status_t loom_value_fact_recurrence_set_build(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_block_t* header_block, uint16_t argument_offset,
    loom_value_slice_t initial_values, loom_value_slice_t next_values,
    const loom_value_fact_recurrence_scope_t* scope,
    iree_arena_allocator_t* arena, loom_value_fact_recurrence_set_t* out_set);

// Returns the equation for an owner-validated carried-state slot, or NULL when
// its update was not a recognized scalar translation.
static inline const loom_value_fact_recurrence_t*
loom_value_fact_recurrence_set_lookup(
    const loom_value_fact_recurrence_set_t* set, uint16_t slot) {
  return set->records ? set->records[slot] : NULL;
}

// Evaluates a retained equation using the current initial/increment facts and
// an independently proven exact or upper-bounded trip count. Literal operands
// retain their immutable numeric value through a solver reset. Every
// intermediate must fit its source and selected target carrier. Returns false
// with unknown ranges when that proof is unavailable; the supplied count kind
// is preserved. Bounded counts retain all possible header states as the exit
// envelope. Output facts contain numeric range/divisibility only, not lane
// distribution.
bool loom_value_fact_recurrence_evaluate(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_recurrence_t* recurrence, uint64_t trip_count,
    loom_loop_trip_count_kind_t trip_count_kind,
    loom_loop_recurrence_facts_t* out_facts);

// Refines numeric range/divisibility with a proven recurrence observation while
// retaining distribution and other applicable facts from the current solve.
loom_value_facts_t loom_value_fact_recurrence_refine(loom_value_facts_t current,
                                                     loom_value_facts_t proven);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_FACT_RECURRENCE_H_
