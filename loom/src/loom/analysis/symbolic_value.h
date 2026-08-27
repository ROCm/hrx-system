// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Direct semantic reasoning over Loom SSA values.

#ifndef LOOM_ANALYSIS_SYMBOLIC_VALUE_H_
#define LOOM_ANALYSIS_SYMBOLIC_VALUE_H_

#include "loom/analysis/symbolic_expr_proof.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the facts for |value_id| after applying edge-local facts active on
// |context|. The query does not infer facts through arbitrary producers.
iree_status_t loom_symbolic_expr_context_lookup_facts(
    loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    loom_value_facts_t* out_facts);

// Returns value facts after applying the active condition facts, producer fact
// inference, and predicates carried by identity operations.
iree_status_t loom_symbolic_value_lookup_condition_refined_facts(
    loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    loom_value_facts_t* out_facts);

// Applies predicates carried by the identity chain beginning at |start_value|
// to |inout_facts|.
iree_status_t loom_symbolic_value_apply_identity_chain_predicates_to_facts(
    loom_symbolic_expr_context_t* context, loom_value_id_t start_value,
    loom_value_facts_t* inout_facts);

// Returns whether two SSA values are algebraically identical.
iree_status_t loom_symbolic_values_match(loom_symbolic_expr_context_t* context,
                                         loom_value_id_t left_value,
                                         loom_value_id_t right_value,
                                         bool* out_match);

// Returns whether two SSA values have equivalent deterministic producer trees.
iree_status_t loom_symbolic_values_semantically_match(
    loom_symbolic_expr_context_t* context, loom_value_id_t left_value,
    loom_value_id_t right_value, bool* out_match);

// Returns true when the current facts prove |value_id| is non-negative.
iree_status_t loom_symbolic_value_is_non_negative(
    loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    bool* out_is_non_negative);

// Returns the two factors when |value_id| is defined by integer multiplication.
bool loom_symbolic_value_product_factors(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    loom_value_id_t* out_left, loom_value_id_t* out_right);

// Attempts direct value-semantic proofs without expanding algebraic
// expressions.
iree_status_t loom_symbolic_value_prove_relation(
    loom_symbolic_expr_context_t* context,
    loom_symbolic_integer_relation_t relation, loom_value_id_t left_value,
    loom_value_id_t right_value, loom_symbolic_proof_result_t* out_result);

// Attempts to discharge a scaled relation from predicates on an identity chain.
iree_status_t loom_symbolic_value_prove_scaled_assumption(
    loom_symbolic_expr_context_t* context, loom_value_id_t start_value,
    loom_value_id_t positive_value, loom_value_id_t negative_value,
    int64_t scale, int64_t constant, bool* out_matched,
    loom_symbolic_proof_result_t* out_result);

// Attempts to discharge a scaled static upper bound from predicates on an
// identity chain.
iree_status_t loom_symbolic_value_prove_scaled_static_assumption(
    loom_symbolic_expr_context_t* context, loom_value_id_t start_value,
    loom_value_id_t positive_value, int64_t scale, int64_t constant,
    bool* out_matched, loom_symbolic_proof_result_t* out_result);

// Returns the select condition directly controlling |value_id| when present.
bool loom_symbolic_value_select_condition(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    loom_value_id_t* out_condition);

// Appends unique select conditions referenced by predicates on the identity
// chain beginning at |start_value|.
void loom_symbolic_value_collect_identity_chain_select_conditions(
    const loom_symbolic_expr_context_t* context, loom_value_id_t start_value,
    loom_value_id_t* conditions, iree_host_size_t condition_capacity,
    iree_host_size_t* inout_condition_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SYMBOLIC_VALUE_H_
