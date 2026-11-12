// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Adapted from LLVM libc's generic/FMod.h, revision
// e9846648fd6183ee6d8cbdb4502213fcf902a211, under the same license.

#include "iree/math/remainder.h"

#include "iree/base/internal/math.h"
#include "iree/math/float_bits.h"

// Encodes a nonzero reduced significand with the divisor's binary exponent.
static IREE_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_math_fmod_encode(uint64_t number, int exponent, unsigned fraction_bits) {
  int shift = iree_math_count_leading_zeros_u64(number) - (63 - fraction_bits);
  number <<= shift;
  exponent -= shift;
  if (exponent >= 0) {
    return (number & ((UINT64_C(1) << fraction_bits) - 1)) |
           ((uint64_t)(exponent + 1) << fraction_bits);
  }
  return number >> -exponent;
}

// The typed entry points specialize these representation constants, matching
// the upstream template. F32 uses a U32 short-quotient path and the same U64
// general reducer as F64. The largest finite exponent gap takes at most 186
// chunks; each consumes the unused leading and trailing significand bits.
static IREE_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_math_fmod_bits(uint64_t numerator, uint64_t denominator, unsigned width,
                    unsigned fraction_bits, unsigned bias) {
  const uint64_t sign_mask = UINT64_C(1) << (width - 1);
  const uint64_t sign = numerator & sign_mask;
  const uint64_t infinity = (uint64_t)(2 * bias + 1) << fraction_bits;
  const uint64_t fraction_mask = (UINT64_C(1) << fraction_bits) - 1;
  numerator &= sign_mask - 1;
  denominator &= sign_mask - 1;
  // All classification predicates are pure integer operations. Combining them
  // directly keeps short-circuit Boolean branches off the ordinary path.
  if (IREE_UNLIKELY((denominator == 0) | (numerator >= infinity) |
                    (denominator > infinity))) {
    return infinity | (UINT64_C(1) << (fraction_bits - 1));
  }
  if (numerator <= denominator) {
    return sign | (numerator < denominator ? numerator : 0);
  }
  int numerator_exponent = (int)(numerator >> fraction_bits);
  int denominator_exponent = (int)(denominator >> fraction_bits);
  if (IREE_LIKELY((denominator_exponent > (int)fraction_bits) &
                  (numerator_exponent - denominator_exponent <=
                   (int)(width - fraction_bits - 1)))) {
    uint64_t numerator_mantissa =
        (numerator & fraction_mask) | (UINT64_C(1) << fraction_bits);
    uint64_t denominator_mantissa =
        (denominator & fraction_mask) | (UINT64_C(1) << fraction_bits);
    uint64_t remainder;
    if (numerator_exponent == denominator_exponent) {
      remainder = numerator_mantissa - denominator_mantissa;
    } else {
      numerator_mantissa <<= numerator_exponent - denominator_exponent;
      remainder = width == 32 ? (uint32_t)numerator_mantissa %
                                    (uint32_t)denominator_mantissa
                              : numerator_mantissa % denominator_mantissa;
    }
    if (remainder == 0) {
      return sign;
    }
    // This exponent bound guarantees a normal result. Preserve that invariant
    // instead of routing the common path through subnormal reconstruction.
    const int normalization =
        iree_math_count_leading_zeros_u64(remainder) - (63 - fraction_bits);
    return sign | ((remainder << normalization) & fraction_mask) |
           ((uint64_t)(denominator_exponent - normalization) << fraction_bits);
  }
  if (numerator_exponent == 0 && denominator_exponent == 0) {
    return sign | (width == 32 ? (uint32_t)numerator % (uint32_t)denominator
                               : numerator % denominator);
  }
  uint64_t numerator_mantissa =
      (numerator & fraction_mask) | (UINT64_C(1) << fraction_bits);
  --numerator_exponent;
  uint64_t denominator_mantissa =
      (denominator & fraction_mask) | (UINT64_C(1) << fraction_bits);
  const int default_leading = 63 - fraction_bits;
  int leading = default_leading;
  if (denominator_exponent > 0) {
    --denominator_exponent;
  } else {
    denominator_mantissa = denominator;
    leading = iree_math_count_leading_zeros_u64(denominator_mantissa);
  }
  const int trailing = iree_math_count_trailing_zeros_u64(denominator_mantissa);
  const int chunk = leading + trailing;
  int difference = numerator_exponent - denominator_exponent;
  int shift = difference < trailing ? difference : trailing;
  denominator_mantissa >>= shift;
  difference -= shift;
  denominator_exponent += shift;
  shift = difference < default_leading ? difference : default_leading;
  numerator_mantissa <<= shift;
  difference -= shift;
  numerator_mantissa %= denominator_mantissa;
  if (numerator_mantissa == 0) {
    return sign;
  }
  // Once all divisor trailing zeros have been removed, a nonzero remainder
  // cannot become zero by multiplication with a power of two. Its reciprocal
  // estimates each quotient; exact integer subtraction corrects the estimate.
  if (difference > chunk) {
    const uint64_t reciprocal = UINT64_MAX / denominator_mantissa;
    while (difference > chunk) {
      difference -= chunk;
      const uint64_t quotient =
          (numerator_mantissa * reciprocal) >> (64 - chunk);
      numerator_mantissa <<= chunk;
      numerator_mantissa -= quotient * denominator_mantissa;
      if (IREE_UNLIKELY(numerator_mantissa > denominator_mantissa)) {
        do {
          numerator_mantissa -= denominator_mantissa;
        } while (numerator_mantissa > denominator_mantissa);
      }
    }
    const uint64_t quotient =
        (numerator_mantissa * reciprocal) >> (64 - difference);
    numerator_mantissa <<= difference;
    numerator_mantissa -= quotient * denominator_mantissa;
    if (IREE_UNLIKELY(numerator_mantissa > denominator_mantissa)) {
      do {
        numerator_mantissa -= denominator_mantissa;
      } while (numerator_mantissa > denominator_mantissa);
    }
  } else if (difference != 0) {
    numerator_mantissa <<= difference;
    numerator_mantissa %= denominator_mantissa;
  }
  return sign | iree_math_fmod_encode(numerator_mantissa, denominator_exponent,
                                      fraction_bits);
}

IREE_API_EXPORT float iree_math_fmod_f32(float numerator, float denominator) {
  return iree_math_f32_from_bits((uint32_t)iree_math_fmod_bits(
      iree_math_f32_to_bits(numerator), iree_math_f32_to_bits(denominator), 32,
      23, 127));
}

IREE_API_EXPORT double iree_math_fmod_f64(double numerator,
                                          double denominator) {
  return iree_math_f64_from_bits(
      iree_math_fmod_bits(iree_math_f64_to_bits(numerator),
                          iree_math_f64_to_bits(denominator), 64, 52, 1023));
}
