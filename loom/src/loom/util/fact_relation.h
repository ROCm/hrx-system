// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Retained signed integer relations carried by scoped value aliases.

#ifndef LOOM_UTIL_FACT_RELATION_H_
#define LOOM_UTIL_FACT_RELATION_H_

#include "loom/analysis/integer_relation.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Canonical immutable relation sets within one populated table scope. Roots
// remain comparable across inference and withdrawal until the scope is cleared.
typedef struct loom_value_fact_relation_roots_t {
  // Counterparts excluded from each signed ordering outcome.
  uint32_t excluded[LOOM_INTEGER_ORDER_OUTCOME_COUNT];
  // Signed literals excluded from equality with the value.
  uint32_t excluded_literal_equal;
} loom_value_fact_relation_roots_t;

// Captures the retained relation sets without enumerating their counterparts.
loom_value_fact_relation_roots_t loom_value_fact_integer_relations_snapshot(
    const loom_value_fact_table_t* table, loom_value_id_t value_id);

// Returns the possible signed comparison outcomes for |lhs| relative to
// |rhs|. Identity, scalar integer facts, retained alias relations, and literal
// exclusions are resolved without walking producers or predicates. Unknown or
// contradictory retained state admits every outcome.
loom_integer_order_outcome_bits_t
loom_value_fact_table_query_signed_integer_order(
    const loom_value_fact_table_t* table, loom_value_id_t lhs,
    loom_value_id_t rhs);

// Returns the possible signed comparison outcomes for |value| relative to
// |literal|. Scalar integer facts and retained alias-local literal exclusions
// are resolved without materializing an SSA constant. Unknown or contradictory
// retained state admits every outcome.
loom_integer_order_outcome_bits_t
loom_value_fact_table_query_signed_integer_literal_order(
    const loom_value_fact_table_t* table, loom_value_id_t value,
    int64_t literal);

// Tightens integer interval endpoints using retained literal exclusions and
// scalar nonzero facts. Work is bounded by the excluded endpoint values, not
// the interval width; interior holes do not change the interval.
void loom_value_fact_table_refine_integer_exclusions(
    const loom_value_fact_table_t* table, loom_value_id_t value,
    loom_value_facts_t* inout_facts);

// Visitor for one constrained retained relation from an alias to a canonical
// counterpart. |outcomes| is the nonempty, non-universal set of possible
// signed comparison outcomes for the alias relative to |counterpart|.
typedef bool (*loom_value_fact_signed_integer_relation_visit_fn_t)(
    void* user_data, loom_value_id_t counterpart,
    loom_integer_order_outcome_bits_t outcomes);

// Visits each constrained signed relation published for |value| once. This
// enumerates the alias's sparse retained sets without allocation or producer
// traversal. Relations published only for a differently scoped alias remain in
// that alias's scope. Returns false when |visit| stops iteration.
bool loom_value_fact_table_for_each_signed_integer_relation_while(
    const loom_value_fact_table_t* table, loom_value_id_t value,
    loom_value_fact_signed_integer_relation_visit_fn_t visit, void* user_data);

// Clears all scope-local retained relation state. The caller releases the
// transient arena after clearing the complete fact-table scope.
void loom_value_fact_integer_relations_clear_scope(
    loom_value_fact_table_t* table);

// Withdraws relations published for |value_id| before cyclic or incremental
// recomputation. The sparse entry remains reusable within the current scope.
void loom_value_fact_integer_relations_undefine(loom_value_fact_table_t* table,
                                                loom_value_id_t value_id);

// Rebuilds one selected alias's canonical relation sets in |target|.
iree_status_t loom_value_fact_integer_relations_clone_value(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    loom_value_id_t value_id);

// Replaces the counted induction value's relations with lower <= value < upper.
// The loop owner publishes these body-local facts when it seeds the argument;
// aliases inherit them through the ordinary relation lifecycle.
iree_status_t loom_value_fact_integer_relations_define_counted_range(
    loom_value_fact_table_t* table, loom_value_id_t value,
    loom_value_id_t lower_bound, loom_value_id_t upper_bound);

// Propagates inherited alias relations and retains signed predicates declared
// by a value-identity operation. All result plans are completed before any
// result relation is published.
iree_status_t loom_value_fact_integer_relations_propagate_origins(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, loom_trait_flags_t traits, bool* inout_changed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_UTIL_FACT_RELATION_H_
