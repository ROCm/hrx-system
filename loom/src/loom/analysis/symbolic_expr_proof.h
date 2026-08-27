// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Proofs over symbolic expressions and SSA values.

#ifndef LOOM_ANALYSIS_SYMBOLIC_EXPR_PROOF_H_
#define LOOM_ANALYSIS_SYMBOLIC_EXPR_PROOF_H_

#include "loom/analysis/integer_relation.h"
#include "loom/analysis/symbolic_expr.h"

#ifdef __cplusplus
extern "C" {
#endif

// Tri-state proof result for symbolic comparisons.
typedef enum loom_symbolic_proof_result_e {
  // The relation could not be proven either way.
  LOOM_SYMBOLIC_PROOF_UNKNOWN = 0,

  // The relation is proven true.
  LOOM_SYMBOLIC_PROOF_TRUE = 1,

  // The relation is proven false.
  LOOM_SYMBOLIC_PROOF_FALSE = 2,
} loom_symbolic_proof_result_t;

// Compact replacement form for a symbolic value difference.
typedef enum loom_symbolic_value_difference_kind_e {
  // The difference is not representable as a single existing value or constant.
  LOOM_SYMBOLIC_VALUE_DIFFERENCE_UNKNOWN = 0,

  // The difference is the exact integer in |constant|.
  LOOM_SYMBOLIC_VALUE_DIFFERENCE_CONSTANT = 1,

  // The difference is the existing SSA value |value_id|.
  LOOM_SYMBOLIC_VALUE_DIFFERENCE_VALUE = 2,
} loom_symbolic_value_difference_kind_t;

// Difference summary for left_value - right_value.
typedef struct loom_symbolic_value_difference_t {
  // Kind of replacement represented by this difference.
  loom_symbolic_value_difference_kind_t kind;

  // Exact integer when kind is LOOM_SYMBOLIC_VALUE_DIFFERENCE_CONSTANT.
  int64_t constant;

  // Existing SSA value when kind is LOOM_SYMBOLIC_VALUE_DIFFERENCE_VALUE.
  loom_value_id_t value_id;
} loom_symbolic_value_difference_t;

// Attempts to prove left <= right. The implementation uses exact term
// cancellation first and falls back to interval facts without allocating new
// retained expression storage.
iree_status_t loom_symbolic_expr_prove_le(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_proof_result_t* out_result);

// Simplifies left_value - right_value when the normalized difference is a
// single existing value or an exact integer constant.
iree_status_t loom_symbolic_expr_simplify_value_difference(
    loom_symbolic_expr_context_t* context, loom_value_id_t left_value,
    loom_value_id_t right_value,
    loom_symbolic_value_difference_t* out_difference);

// Attempts to prove an integer relation between two SSA values.
iree_status_t loom_symbolic_expr_prove_value_relation(
    loom_symbolic_expr_context_t* context,
    loom_symbolic_integer_relation_t relation, loom_value_id_t left_value,
    loom_value_id_t right_value, loom_symbolic_proof_result_t* out_result);

// Attempts to prove an integer relation using only facts already active on
// |context|. This does not enumerate hypothetical select outcomes, keeping the
// query suitable for speculative canonicalization on large value graphs.
iree_status_t loom_symbolic_expr_prove_value_relation_with_active_facts(
    loom_symbolic_expr_context_t* context,
    loom_symbolic_integer_relation_t relation, loom_value_id_t left_value,
    loom_value_id_t right_value, loom_symbolic_proof_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SYMBOLIC_EXPR_PROOF_H_
