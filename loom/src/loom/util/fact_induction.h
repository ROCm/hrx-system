// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Recurrence equations shared by CFG and structured condition-loop facts.

#ifndef LOOM_UTIL_FACT_INDUCTION_H_
#define LOOM_UTIL_FACT_INDUCTION_H_

#include "loom/analysis/loop_domain.h"
#include "loom/ir/ir.h"
#include "loom/util/fact_recurrence.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_value_fact_table_t loom_value_fact_table_t;

// Retained control and update semantics of an induction equation.
enum loom_value_fact_induction_flag_bits_e {
  LOOM_VALUE_FACT_INDUCTION_NONE = 0,
  // The literal header condition selects the exit; no backedge executes.
  LOOM_VALUE_FACT_INDUCTION_EXITS_AT_HEADER = 1u << 0,
  // The backedge subtracts the retained step operand instead of adding it.
  LOOM_VALUE_FACT_INDUCTION_SUBTRACT_STEP = 1u << 1,
  // Other exits may retire the loop before its controlling header exit.
  LOOM_VALUE_FACT_INDUCTION_MAY_EXIT_EARLY = 1u << 2,
};
typedef uint8_t loom_value_fact_induction_flags_t;

// A recognized header-tested recurrence. Its inputs are invariant within the
// loop. Additional exits preserve its iteration bound, not an exact count.
// The owning fact solver refreshes this equation after semantic edits; numeric
// queries consume current table entries, never cached counts.
typedef struct loom_value_fact_induction_t {
  // Header argument, or INVALID for an unrecognized or literal exit guard.
  loom_value_id_t value;
  // Value entering the header from outside the loop.
  loom_value_fact_recurrence_operand_t initial_value;
  // Invariant bound tested by the header guard.
  loom_value_fact_recurrence_operand_t bound;
  // Invariant operand added or subtracted by the backedge. An absent operand
  // can still establish zero trips from a false entry guard.
  loom_value_fact_recurrence_operand_t step;
  // Signedness, direction and inclusivity of the guard.
  loom_loop_bound_flags_t bound_flags;
  // Header termination and backedge operation semantics.
  loom_value_fact_induction_flags_t flags;
} loom_value_fact_induction_t;

// Returns the current additive delta, interpreting the retained backedge sign.
// Exact deltas retain modular i64 bits; ranged deltas use mathematical bounds.
loom_value_facts_t loom_value_fact_induction_step_facts(
    const loom_value_fact_table_t* table,
    const loom_value_fact_induction_t* induction);

// Evaluates a retained recurrence against current facts and target carriers.
// Bounded inputs can establish conservative recurrence envelopes; exact-count
// consumers distinguish those results from constant execution counts.
loom_loop_recurrence_facts_t loom_value_fact_induction_facts(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_induction_t* induction);

// Recognizes a necessary guard's recurrence at a verified LoopLike boundary.
// The summary owner supplies a structurally implied guard and its truth, after
// computing declared SSA aliases. A guard other than the condition selector
// establishes a maximum count, not exact execution. Queries consume the
// retained equation instead of rediscovering its structure.
loom_value_fact_induction_t loom_value_fact_condition_loop_induction(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_loop_like_t loop, loom_value_id_t guard, bool guard_truth);

// Builds additive state equations after the loop body has established its SSA
// identities. Counted loops use the body boundary; condition loops use the
// header boundary and its declared forwarding into the body.
iree_status_t loom_value_fact_loop_build_recurrences(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_loop_like_t loop, iree_arena_allocator_t* arena,
    loom_value_fact_recurrence_set_t* out_set);

// Publishes the current equation with the condition region's fact-scope
// lifetime. Recomputing a structured summary replaces the previous equation.
iree_status_t loom_value_fact_table_set_condition_induction(
    loom_value_fact_table_t* table, const loom_region_t* condition_region,
    loom_value_fact_induction_t induction);

// Returns the published equation, or NULL before this region is summarized.
const loom_value_fact_induction_t*
loom_value_fact_table_lookup_condition_induction(
    const loom_value_fact_table_t* table,
    const loom_region_t* condition_region);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_FACT_INDUCTION_H_
