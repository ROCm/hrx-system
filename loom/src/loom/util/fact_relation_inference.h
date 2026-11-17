// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Publication and transfer of retained signed integer relations.

#ifndef LOOM_UTIL_FACT_RELATION_INFERENCE_H_
#define LOOM_UTIL_FACT_RELATION_INFERENCE_H_

#include "loom/util/fact_relation.h"

#ifdef __cplusplus
extern "C" {
#endif

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

// Replaces the counted induction value's relations with lower <= value < upper
// and their consequences through the endpoints' retained relations. The loop
// owner publishes these body-local facts when it seeds the argument; aliases
// inherit them through the ordinary relation lifecycle.
iree_status_t loom_value_fact_integer_relations_define_counted_range(
    loom_value_fact_table_t* table, loom_value_id_t value,
    loom_value_id_t lower_bound, loom_value_id_t upper_bound);

// Replaces a result's relations with upper bounds and their transitive
// consequences. |relation| is LT or LE. The producer supplies the current
// definition-scope result and endpoint facts; result facts need not have been
// published yet. Disjoint intervals need no retained relation unless an
// endpoint carries further dynamic constraints. INVALID endpoints are omitted,
// and an empty publication withdraws prior relations. Optional |equivalents|
// names additional endpoints equal to the corresponding bounds in this
// definition's scope. Both endpoints retain their own transitive relations;
// their equality is not published for uses outside the scope. The caller owns
// publication and includes auxiliary changes in its ordinary inference change
// accumulator.
iree_status_t loom_value_fact_integer_relations_define_upper_bounds(
    loom_value_fact_table_t* table, loom_value_id_t value,
    const loom_value_id_t* upper_bounds, const loom_value_id_t* equivalents,
    const loom_value_facts_t* upper_facts, iree_host_size_t bound_count,
    loom_symbolic_integer_relation_t relation,
    const loom_value_facts_t* value_facts, bool* inout_changed);

// Propagates inherited alias relations and retains signed consequences of a
// value-identity operation's predicates. Unsigned order contributes only when
// scoped numeric facts and the predicate place both values in one sign
// partition. Consequences through each incoming endpoint share its retained
// sets, without following producer chains. All result plans are completed
// before any relation is published.
iree_status_t loom_value_fact_integer_relations_propagate_origins(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, loom_trait_flags_t traits, bool* inout_changed);

// Shares immutable relation roots after the producer proves numeric equality.
// INVALID source withdraws prior roots without changing SSA identity.
iree_status_t loom_value_fact_integer_relations_forward_alias(
    loom_value_fact_table_t* table, loom_value_id_t source,
    loom_value_id_t result, bool* inout_changed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_UTIL_FACT_RELATION_INFERENCE_H_
