// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Integer math helpers for value analysis transfer functions.
//
// Portable checked signed int64 arithmetic with overflow detection,
// GCD/LCM for divisibility analysis, and bit operations. Every
// function is static inline — no .c file.
//
// Checked arithmetic returns true on success, false on overflow.
// Output is written via pointer parameter; undefined on overflow.
// Uses compiler builtins when available (__builtin_*_overflow on
// GCC/Clang), with portable manual fallbacks for MSVC.
//
// GCD always returns a value >= 1 (sentinel for the facts system
// where known_divisor >= 1). LCM is overflow-checked and returns
// false if the result would exceed INT64_MAX.
//
// Typical usage in transfer functions:
//
//   int64_t lo;
//   if (!loom_checked_add_i64(lhs.range_lo, rhs.range_lo, &lo)) {
//     return loom_value_facts_unknown();  // Overflow: lose precision.
//   }

#ifndef LOOM_UTIL_MATH_H_
#define LOOM_UTIL_MATH_H_

#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Checked arithmetic
//===----------------------------------------------------------------------===//
//
// Signed int64 arithmetic with overflow detection. Returns true on
// success (no overflow), false on overflow (*out is undefined).

// Checked signed addition.
static inline bool loom_checked_add_i64(int64_t a, int64_t b, int64_t* out) {
#if IREE_HAVE_BUILTIN(__builtin_add_overflow)
  return !__builtin_add_overflow(a, b, out);
#else
  // Unsigned addition is well-defined (wraps). Signed overflow occurs
  // when operands have the same sign but the result has a different sign.
  *out = (int64_t)((uint64_t)a + (uint64_t)b);
  return !((a ^ b) >= 0 && (a ^ *out) < 0);
#endif
}

// Checked signed subtraction.
static inline bool loom_checked_sub_i64(int64_t a, int64_t b, int64_t* out) {
#if IREE_HAVE_BUILTIN(__builtin_sub_overflow)
  return !__builtin_sub_overflow(a, b, out);
#else
  // Subtraction overflow occurs when operands have different signs and
  // the result sign disagrees with the first operand.
  *out = (int64_t)((uint64_t)a - (uint64_t)b);
  return !((a ^ b) < 0 && (a ^ *out) < 0);
#endif
}

// Checked signed multiplication.
static inline bool loom_checked_mul_i64(int64_t a, int64_t b, int64_t* out) {
#if IREE_HAVE_BUILTIN(__builtin_mul_overflow)
  return !__builtin_mul_overflow(a, b, out);
#else
  // Zero is the trivial non-overflow case.
  if (a == 0 || b == 0) {
    *out = 0;
    return true;
  }
  // Four quadrants of sign combinations, checking against INT64_MAX/MIN
  // divided by the other operand.
  if (a > 0) {
    if (b > 0) {
      if (a > INT64_MAX / b) return false;
    } else {
      if (b < INT64_MIN / a) return false;
    }
  } else {
    if (b > 0) {
      if (a < INT64_MIN / b) return false;
    } else {
      if (a < INT64_MAX / b) return false;
    }
  }
  *out = a * b;
  return true;
#endif
}

//===----------------------------------------------------------------------===//
// GCD / LCM
//===----------------------------------------------------------------------===//

// Greatest common divisor of two integers. Returns >= 1 always: if
// both inputs are 0, returns 1 (sentinel for the facts system where
// known_divisor is always >= 1). Handles negative inputs via absolute
// value.
static inline int64_t loom_gcd_i64(int64_t a, int64_t b) {
  // Use unsigned absolute values to avoid llabs(INT64_MIN) UB.
  uint64_t ua = (a >= 0) ? (uint64_t)a : (uint64_t)(-(a + 1)) + 1;
  uint64_t ub = (b >= 0) ? (uint64_t)b : (uint64_t)(-(b + 1)) + 1;
  if (ua == 0) return ub > 0 ? (int64_t)ub : 1;
  if (ub == 0) return (int64_t)ua;
  while (ub != 0) {
    uint64_t temp = ub;
    ub = ua % ub;
    ua = temp;
  }
  // The GCD of two int64 values always fits in int64 (it divides both).
  return (int64_t)ua;
}

// Least common multiple. Returns false on overflow. If either input
// is 0, the LCM is 0 (stored in *out).
static inline bool loom_lcm_i64(int64_t a, int64_t b, int64_t* out) {
  if (a == 0 || b == 0) {
    *out = 0;
    return true;
  }
  int64_t gcd = loom_gcd_i64(a, b);
  // Compute |a / gcd| * |b| to avoid intermediate overflow.
  // Use absolute values safely.
  int64_t a_abs = (a > 0) ? a : ((a == INT64_MIN) ? INT64_MAX : -a);
  int64_t b_abs = (b > 0) ? b : ((b == INT64_MIN) ? INT64_MAX : -b);
  return loom_checked_mul_i64(a_abs / gcd, b_abs, out);
}

//===----------------------------------------------------------------------===//
// Bit operations
//===----------------------------------------------------------------------===//

// Masks a raw unsigned value to the low |bitwidth| bits. A bitwidth of 64 keeps
// the value unchanged.
static inline uint64_t loom_mask_to_bitwidth_u64(uint64_t value,
                                                 int32_t bitwidth) {
  if (bitwidth <= 0) return 0;
  if (bitwidth >= 64) return value;
  return value & ((((uint64_t)1) << bitwidth) - 1);
}

// Counts leading zeros in the low |bitwidth| bits of |value|.
static inline int64_t loom_count_leading_zeros_u64_width(uint64_t value,
                                                         int32_t bitwidth) {
  if (bitwidth > 64) bitwidth = 64;
  value = loom_mask_to_bitwidth_u64(value, bitwidth);
  if (value == 0) return bitwidth > 0 ? bitwidth : 0;
  if (bitwidth >= 64) return iree_math_count_leading_zeros_u64(value);
  return iree_math_count_leading_zeros_u64(value) - (64 - bitwidth);
}

// Counts trailing zeros in the low |bitwidth| bits of |value|.
static inline int64_t loom_count_trailing_zeros_u64_width(uint64_t value,
                                                          int32_t bitwidth) {
  if (bitwidth > 64) bitwidth = 64;
  value = loom_mask_to_bitwidth_u64(value, bitwidth);
  if (value == 0) return bitwidth > 0 ? bitwidth : 0;
  return iree_math_count_trailing_zeros_u64(value);
}

// Counts set bits in the low |bitwidth| bits of |value|.
static inline int64_t loom_count_ones_u64_width(uint64_t value,
                                                int32_t bitwidth) {
  if (bitwidth > 64) bitwidth = 64;
  return iree_math_count_ones_u64(loom_mask_to_bitwidth_u64(value, bitwidth));
}

// Floor of log base 2. Undefined for value <= 0.
static inline int32_t loom_ilog2_i64(int64_t value) {
  return 63 - iree_math_count_leading_zeros_u64((uint64_t)value);
}

// Returns true if value is a positive power of 2 (1, 2, 4, 8, ...).
static inline bool loom_is_power_of_two_i64(int64_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}

static inline int64_t loom_min_i64(int64_t a, int64_t b) {
  return a < b ? a : b;
}
static inline int64_t loom_max_i64(int64_t a, int64_t b) {
  return a > b ? a : b;
}

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_MATH_H_
