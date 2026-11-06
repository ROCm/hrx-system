// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared algebraic combining vocabulary for reductions and scans.
//
// Generated dialect APIs alias this enum type directly so vector, kernel, and
// target lowering code use one semantic domain for ordinary associative
// combining operations.

#ifndef LOOM_OPS_COMBINING_H_
#define LOOM_OPS_COMBINING_H_

#include "iree/base/api.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/combining_kind.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |kind| is a known loom_combining_kind_t value.
static inline bool loom_combining_kind_is_valid(loom_combining_kind_t kind) {
  return (uint32_t)kind < (uint32_t)LOOM_COMBINING_KIND_COUNT_;
}

// Returns true for either floating-point minimum semantic. Other combining
// kinds return false.
static inline bool loom_combining_kind_is_minimum_float_extremum(
    loom_combining_kind_t kind) {
  return kind == LOOM_COMBINING_KIND_MINIMUMF ||
         kind == LOOM_COMBINING_KIND_MINNUMF;
}

// Returns true when |kind| operates on integer element types.
bool loom_combining_kind_accepts_integer(loom_combining_kind_t kind);

// Returns true when |kind| operates on floating-point element types.
bool loom_combining_kind_accepts_float(loom_combining_kind_t kind);

// Returns true when repeating operands does not change the result.
// This property permits reduction plans to fill otherwise unused packet lanes
// with copies of authored values instead of materializing an identity.
bool loom_combining_kind_is_idempotent(loom_combining_kind_t kind);

// Returns the bitwise operation exactly equivalent to |kind| for normalized
// i1 operands. Integer arithmetic and extrema collapse to Boolean algebra at
// one bit; floating-point kinds have no equivalent and return false.
bool loom_combining_kind_i1_equivalent(loom_combining_kind_t kind,
                                       loom_combining_kind_t* out_kind);

// Returns the exact fixed-width integer identity for |kind| and
// |element_type|. The returned signed value carries the identity bit pattern;
// i1 identities are normalized to zero or one. Returns false for non-integer
// element types or unknown combining kinds.
bool loom_combining_kind_integer_identity(loom_combining_kind_t kind,
                                          loom_scalar_type_t element_type,
                                          int64_t* out_identity);

// Returns the exact floating-point identity for |kind|. Addition uses negative
// zero so signed-zero inputs remain unchanged; multiplication uses positive
// one. Extrema kinds rely on idempotence instead of an invented infinity.
bool loom_combining_kind_float_identity(loom_combining_kind_t kind,
                                        double* out_identity);

// Returns the empty-prefix seed used by an exclusive floating-point scan.
// Arithmetic scans use their exact identities. Minimum scans use positive
// infinity and maximum scans use negative infinity; the seed is returned only
// for an empty prefix and is never combined with an authored input value.
bool loom_combining_kind_float_scan_seed(loom_combining_kind_t kind,
                                         double* out_seed);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_COMBINING_H_
