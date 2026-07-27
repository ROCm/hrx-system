// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared integer relation vocabulary for symbolic proofs and edge facts.

#ifndef LOOM_ANALYSIS_INTEGER_RELATION_H_
#define LOOM_ANALYSIS_INTEGER_RELATION_H_

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Integer relations over two scalar address/integer values.
typedef enum loom_symbolic_integer_relation_e {
  // left == right.
  LOOM_SYMBOLIC_INTEGER_RELATION_EQ = 0,
  // left != right.
  LOOM_SYMBOLIC_INTEGER_RELATION_NE = 1,
  // left < right.
  LOOM_SYMBOLIC_INTEGER_RELATION_LT = 2,
  // left <= right.
  LOOM_SYMBOLIC_INTEGER_RELATION_LE = 3,
  // left > right.
  LOOM_SYMBOLIC_INTEGER_RELATION_GT = 4,
  // left >= right.
  LOOM_SYMBOLIC_INTEGER_RELATION_GE = 5,
} loom_symbolic_integer_relation_t;

// Returns the logical negation of |relation|.
loom_symbolic_integer_relation_t loom_symbolic_integer_relation_invert(
    loom_symbolic_integer_relation_t relation);

// Returns the equivalent relation after swapping left and right operands.
loom_symbolic_integer_relation_t loom_symbolic_integer_relation_swap(
    loom_symbolic_integer_relation_t relation);

// Attempts to prove |queried_relation| from one known |implied_relation| over
// the same ordered operands. Returns false when implication is unknown.
bool loom_symbolic_integer_relation_implies(
    loom_symbolic_integer_relation_t implied_relation,
    loom_symbolic_integer_relation_t queried_relation, bool* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_INTEGER_RELATION_H_
