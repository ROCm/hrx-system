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

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_combining_kind_e {
  // Integer addition.
  LOOM_COMBINING_KIND_ADDI = 0,
  // Floating-point addition.
  LOOM_COMBINING_KIND_ADDF = 1,
  // Integer multiplication.
  LOOM_COMBINING_KIND_MULI = 2,
  // Floating-point multiplication.
  LOOM_COMBINING_KIND_MULF = 3,
  // Signed integer minimum.
  LOOM_COMBINING_KIND_MINSI = 4,
  // Signed integer maximum.
  LOOM_COMBINING_KIND_MAXSI = 5,
  // Unsigned integer minimum.
  LOOM_COMBINING_KIND_MINUI = 6,
  // Unsigned integer maximum.
  LOOM_COMBINING_KIND_MAXUI = 7,
  // Bitwise AND.
  LOOM_COMBINING_KIND_ANDI = 8,
  // Bitwise OR.
  LOOM_COMBINING_KIND_ORI = 9,
  // Bitwise XOR.
  LOOM_COMBINING_KIND_XORI = 10,
  // IEEE 754 floating-point minimum.
  LOOM_COMBINING_KIND_MINIMUMF = 11,
  // IEEE 754 floating-point maximum.
  LOOM_COMBINING_KIND_MAXIMUMF = 12,
  // C99 fmin-style floating-point minimum.
  LOOM_COMBINING_KIND_MINNUMF = 13,
  // C99 fmax-style floating-point maximum.
  LOOM_COMBINING_KIND_MAXNUMF = 14,
  LOOM_COMBINING_KIND_COUNT_,
} loom_combining_kind_t;

// Returns true when |kind| is a known loom_combining_kind_t value.
static inline bool loom_combining_kind_is_valid(loom_combining_kind_t kind) {
  return (uint32_t)kind < (uint32_t)LOOM_COMBINING_KIND_COUNT_;
}

// Returns true when |kind| operates on integer element types.
bool loom_combining_kind_accepts_integer(loom_combining_kind_t kind);

// Returns true when |kind| operates on floating-point element types.
bool loom_combining_kind_accepts_float(loom_combining_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_COMBINING_H_
