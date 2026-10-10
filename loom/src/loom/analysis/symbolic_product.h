// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exact products retained alongside materialized symbolic expressions.

#ifndef LOOM_ANALYSIS_SYMBOLIC_PRODUCT_H_
#define LOOM_ANALYSIS_SYMBOLIC_PRODUCT_H_

#include "loom/analysis/symbolic_expr.h"

#ifdef __cplusplus
extern "C" {
#endif

// Represents scale * product(factors). Factors are sorted SSA identities with
// multiplicity preserved, including factors that may be zero. The expansion
// owner establishes nonwrapping arithmetic before composing this record.
// Storage shares the symbolic context's arena and invalidation boundary. This
// is a numeric proof, not a recipe for replacing materialized SSA arithmetic.
typedef struct loom_symbolic_product_t {
  // Nonzero signed coefficient multiplying all factors.
  int64_t scale;
  // Arena-owned, sorted SSA identities; duplicates represent repeated factors.
  const loom_value_id_t* factors;
  // Number of factors, or zero when composition exceeds the proof budget.
  iree_host_size_t factor_count;
} loom_symbolic_product_t;

// Computes the mathematical product range without saturating its endpoints.
// Returns false when an endpoint is outside the signed proof representation.
bool loom_symbolic_product_range(loom_value_facts_t left,
                                 loom_value_facts_t right,
                                 loom_value_facts_t* out_facts);

// Composes already-expanded operands of an exact multiplication. Non-product
// operands remain whole SSA factors; affine sums are not distributed. Returns
// an empty product when the coefficient or factor count is not representable.
// Only factor storage allocation can fail.
iree_status_t loom_symbolic_product_multiply(
    loom_symbolic_expr_context_t* context, loom_value_id_t left_value,
    const loom_symbolic_expr_t* left_expression, loom_value_id_t right_value,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_product_t* out_product);

// Matches retained factor identities and returns the terms' full coefficients,
// including scales absorbed inside their products. Reads computed summaries
// only; never expands producers, divides factors, or allocates storage.
bool loom_symbolic_product_match_terms(
    const loom_symbolic_expr_context_t* context,
    const loom_symbolic_term_t* left, const loom_symbolic_term_t* right,
    int64_t* out_left_coefficient, int64_t* out_right_coefficient);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SYMBOLIC_PRODUCT_H_
