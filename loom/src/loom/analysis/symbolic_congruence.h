// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Producer-owned periodic guarantees over exact symbolic expressions.

#ifndef LOOM_ANALYSIS_SYMBOLIC_CONGRUENCE_H_
#define LOOM_ANALYSIS_SYMBOLIC_CONGRUENCE_H_

#include "loom/analysis/symbolic_expr.h"

#ifdef __cplusplus
extern "C" {
#endif

// The represented value differs from |expression| by a multiple of |modulus|.
// The affine form is normalized and contains no nested congruence. It shares
// the symbolic context's arena and invalidation boundary. Its terms identify
// values within one evaluation; callers must establish that common domain
// before canceling identities across two memory accesses.
struct loom_symbolic_congruence_t {
  // Positive modulus greater than one.
  uint64_t modulus;
  // Normalized affine form, independent of the represented value's range.
  loom_symbolic_expr_t expression;
};

// Attaches input modulo |modulus| to an already constructed exact result.
// A modulus of one, unknown affine form, or unrepresentable combination
// contributes no additional guarantee. |modulus| must be positive.
iree_status_t loom_symbolic_congruence_restrict(
    loom_symbolic_expr_context_t* context, const loom_symbolic_expr_t* input,
    uint64_t modulus, loom_symbolic_expr_t* output);

// Propagates left + sign*right to an already constructed exact result.
// |sign| is either 1 or -1. Exact-only inputs need no additional storage.
iree_status_t loom_symbolic_congruence_combine(
    loom_symbolic_expr_context_t* context, const loom_symbolic_expr_t* left,
    const loom_symbolic_expr_t* right, int64_t sign,
    loom_symbolic_expr_t* output);

// Propagates multiplication by an exact constant to the exact result.
iree_status_t loom_symbolic_congruence_scale(
    loom_symbolic_expr_context_t* context, const loom_symbolic_expr_t* input,
    int64_t multiplier, loom_symbolic_expr_t* output);

// Proves that left-right cannot lie in the inclusive interval [lower, upper].
// Consumes normalized forms without allocation or source/CFG traversal. Exact
// terms cancel only within the evaluation domain established by the caller.
bool loom_symbolic_congruence_excludes_difference(
    const loom_symbolic_expr_t* left, const loom_symbolic_expr_t* right,
    int64_t lower, int64_t upper);

// Attempts the same proof using retained single-source congruences from exact
// affine terms and indexed value ranges for the bounded residual. This does
// not expand producers or allocate retained storage.
bool loom_symbolic_congruence_prove_difference_outside_interval(
    const loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression, int64_t lower, int64_t upper);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SYMBOLIC_CONGRUENCE_H_
