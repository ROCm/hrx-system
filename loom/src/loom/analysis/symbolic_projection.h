// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exact integer digits retained alongside materialized symbolic expressions.

#ifndef LOOM_ANALYSIS_SYMBOLIC_PROJECTION_H_
#define LOOM_ANALYSIS_SYMBOLIC_PROJECTION_H_

#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Represents ((scale * value + offset) / divisor) % modulus, with a zero
// modulus retaining the full quotient. The producer proves a nonnegative,
// non-wrapping numerator and positive divisors before retaining this record.
// The record shares its symbolic context's arena and invalidation boundary.
// It is a numeric proof, not a recipe for replacing a materialized SSA value.
typedef struct loom_symbolic_projection_t {
  // SSA identity of the source coordinate before affine arithmetic.
  loom_value_id_t value_id;
  // Positive coefficient applied before division and remainder.
  int64_t scale;
  // Nonnegative constant added before division and remainder.
  int64_t offset;
  // Positive divisor applied to the affine numerator.
  int64_t divisor;
  // Positive remainder modulus, or zero to retain the quotient.
  int64_t modulus;
} loom_symbolic_projection_t;

// Composes division by a positive constant when the result remains one digit.
// Returns false when composition needs a more general expression or overflows
// the proof representation. Inputs and output may alias.
bool loom_symbolic_projection_divide(const loom_symbolic_projection_t* input,
                                     int64_t divisor,
                                     loom_symbolic_projection_t* output);

// Composes remainder by a positive constant when the result remains one digit.
// Inputs and output may alias.
bool loom_symbolic_projection_remainder(const loom_symbolic_projection_t* input,
                                        int64_t modulus,
                                        loom_symbolic_projection_t* output);

// Removes numerator slack that cannot affect the quotient when value_id is a
// multiple of known_divisor. Preserves both quotient and remainder values.
void loom_symbolic_projection_refine(int64_t known_divisor,
                                     loom_symbolic_projection_t* projection);

// Compares retained numeric functions without source traversal or allocation.
bool loom_symbolic_projection_equal(const loom_symbolic_projection_t* left,
                                    const loom_symbolic_projection_t* right);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SYMBOLIC_PROJECTION_H_
