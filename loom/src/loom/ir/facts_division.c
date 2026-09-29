// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/math.h"
#include "loom/ir/facts.h"

typedef struct loom_value_facts_unsigned_range_t {
  // Inclusive minimum unsigned value.
  uint64_t minimum;
  // Inclusive maximum unsigned value.
  uint64_t maximum;
} loom_value_facts_unsigned_range_t;

static loom_value_facts_unsigned_range_t loom_value_facts_unsigned_range(
    loom_value_facts_t facts, uint64_t bit_mask) {
  if (facts.range_lo >= 0) {
    return (loom_value_facts_unsigned_range_t){(uint64_t)facts.range_lo,
                                               (uint64_t)facts.range_hi};
  }
  if (facts.range_hi < 0) {
    return (loom_value_facts_unsigned_range_t){
        (uint64_t)facts.range_lo & bit_mask,
        (uint64_t)facts.range_hi & bit_mask};
  }
  return (loom_value_facts_unsigned_range_t){0, bit_mask};
}

static loom_value_facts_t loom_value_facts_make_unsigned_result_range(
    uint64_t minimum, uint64_t maximum, int32_t bit_count,
    int64_t known_divisor) {
  if (bit_count == 1 || maximum < (UINT64_C(1) << (bit_count - 1))) {
    return loom_value_facts_make((int64_t)minimum, (int64_t)maximum,
                                 known_divisor);
  }
  known_divisor &= -known_divisor;
  if (minimum >= (UINT64_C(1) << (bit_count - 1))) {
    const int64_t signed_minimum =
        loom_value_facts_make_signed_raw_bits(minimum, bit_count).range_lo;
    const int64_t signed_maximum =
        loom_value_facts_make_signed_raw_bits(maximum, bit_count).range_lo;
    return loom_value_facts_make(signed_minimum, signed_maximum, known_divisor);
  }
  const loom_value_facts_t domain =
      loom_value_facts_make_signed_bit_count_range(bit_count);
  return loom_value_facts_make(domain.range_lo, domain.range_hi, known_divisor);
}

static int64_t loom_value_facts_unsigned_known_divisor(
    loom_value_facts_t facts) {
  return facts.range_lo < 0 ? facts.known_divisor & -facts.known_divisor
                            : facts.known_divisor;
}

typedef enum loom_value_facts_unsigned_division_kind_e {
  LOOM_VALUE_FACTS_UNSIGNED_QUOTIENT,
  LOOM_VALUE_FACTS_UNSIGNED_REMAINDER,
} loom_value_facts_unsigned_division_kind_t;

static void loom_value_facts_unsigned_division(
    const loom_value_facts_t* lhs, const loom_value_facts_t* rhs,
    int32_t bit_count, loom_value_facts_unsigned_division_kind_t kind,
    loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts =
      loom_value_facts_wrap_integer(*lhs, bit_count);
  loom_value_facts_t rhs_facts = loom_value_facts_wrap_integer(*rhs, bit_count);
  rhs_facts.flags |= rhs->flags & LOOM_VALUE_FACT_NON_ZERO;
  uint64_t lhs_bits = 0;
  uint64_t rhs_bits = 0;
  const bool lhs_is_exact =
      loom_value_facts_as_exact_raw_bits(lhs_facts, bit_count, &lhs_bits);
  const bool rhs_is_exact =
      loom_value_facts_as_exact_raw_bits(rhs_facts, bit_count, &rhs_bits);
  if (lhs_is_exact && rhs_is_exact && rhs_bits != 0) {
    const uint64_t result = kind == LOOM_VALUE_FACTS_UNSIGNED_QUOTIENT
                                ? lhs_bits / rhs_bits
                                : lhs_bits % rhs_bits;
    *out = bit_count == 1
               ? loom_value_facts_exact_i64((int64_t)result)
               : loom_value_facts_make_signed_raw_bits(result, bit_count);
  } else {
    const uint64_t bit_mask =
        iree_math_mask_low_bits_u64(UINT64_MAX, bit_count);
    const loom_value_facts_unsigned_range_t lhs_range =
        loom_value_facts_unsigned_range(lhs_facts, bit_mask);
    const loom_value_facts_unsigned_range_t rhs_range =
        loom_value_facts_unsigned_range(rhs_facts, bit_mask);
    const bool rhs_may_be_zero = rhs_facts.range_lo <= 0 &&
                                 rhs_facts.range_hi >= 0 &&
                                 !loom_value_facts_is_non_zero(rhs_facts);
    if (rhs_may_be_zero || rhs_range.maximum == 0) {
      *out = loom_value_facts_make_unsigned_result_range(0, bit_mask, bit_count,
                                                         1);
    } else if (kind == LOOM_VALUE_FACTS_UNSIGNED_REMAINDER) {
      const uint64_t lhs_divisor =
          (uint64_t)loom_value_facts_unsigned_known_divisor(lhs_facts);
      if (rhs_is_exact && lhs_divisor % rhs_bits == 0) {
        *out = loom_value_facts_exact_i64(0);
      } else {
        *out = loom_value_facts_make_unsigned_result_range(
            0, iree_min(lhs_range.maximum, rhs_range.maximum - 1), bit_count,
            1);
      }
    } else {
      uint64_t rhs_minimum = rhs_range.minimum;
      if (rhs_minimum == 0) {
        rhs_minimum = rhs_facts.range_hi > 0
                          ? 1
                          : (uint64_t)rhs_facts.range_lo & bit_mask;
      }
      const uint64_t lhs_divisor =
          (uint64_t)loom_value_facts_unsigned_known_divisor(lhs_facts);
      const int64_t divisor = rhs_is_exact && lhs_divisor % rhs_bits == 0
                                  ? (int64_t)(lhs_divisor / rhs_bits)
                                  : 1;
      *out = loom_value_facts_make_unsigned_result_range(
          lhs_range.minimum / rhs_range.maximum,
          lhs_range.maximum / rhs_minimum, bit_count, divisor);
    }
  }
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
}

void loom_value_facts_divui(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs, int32_t bit_count,
                            loom_value_facts_t* out) {
  loom_value_facts_unsigned_division(lhs, rhs, bit_count,
                                     LOOM_VALUE_FACTS_UNSIGNED_QUOTIENT, out);
}

void loom_value_facts_remui(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs, int32_t bit_count,
                            loom_value_facts_t* out) {
  loom_value_facts_unsigned_division(lhs, rhs, bit_count,
                                     LOOM_VALUE_FACTS_UNSIGNED_REMAINDER, out);
}

void loom_value_facts_divsi(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  int64_t lhs_lo = lhs_facts.range_lo, lhs_hi = lhs_facts.range_hi;
  int64_t rhs_lo = rhs_facts.range_lo, rhs_hi = rhs_facts.range_hi;
  int64_t lhs_divisor = lhs_facts.known_divisor;

  // Compute divisor first (independent of range sign).
  int64_t divisor = 1;
  if (rhs_lo == rhs_hi && rhs_lo != 0) {
    int64_t abs_rhs =
        (rhs_lo > 0) ? rhs_lo : ((rhs_lo == INT64_MIN) ? INT64_MAX : -rhs_lo);
    if (lhs_divisor % abs_rhs == 0) {
      divisor = lhs_divisor / abs_rhs;
    }
  }

  // If the divisor range includes zero, we cannot compute range.
  if (rhs_lo <= 0 && rhs_hi >= 0) {
    *out = loom_value_facts_make(INT64_MIN, INT64_MAX, divisor);
    loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
    return;
  }
  // Both operands non-negative: straightforward range.
  if (lhs_lo >= 0 && rhs_lo > 0) {
    int64_t lo = lhs_lo / rhs_hi;
    int64_t hi = lhs_hi / rhs_lo;
    *out = loom_value_facts_make(lo, hi, divisor);
    loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
    return;
  }
  // Mixed signs: conservatively unknown range, but divisor is valid.
  *out = loom_value_facts_make(INT64_MIN, INT64_MAX, divisor);
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
}

// Computes a signed remainder with truncation-toward-zero semantics. The
// divisor must be nonzero. Unsigned division makes the INT64_MIN / -1 overflow
// pair representable and its remainder is zero as required.
static int64_t loom_value_facts_remainder_i64(int64_t dividend,
                                              int64_t divisor) {
  const uint64_t remainder =
      iree_math_magnitude_i64(dividend) % iree_math_magnitude_i64(divisor);
  if (dividend >= 0 || remainder == 0) {
    return (int64_t)remainder;
  }
  return -(int64_t)remainder;
}

void loom_value_facts_remsi(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  int64_t rhs_lo = rhs_facts.range_lo, rhs_hi = rhs_facts.range_hi;

  // Divisor range must not include zero.
  if (rhs_lo <= 0 && rhs_hi >= 0) {
    *out = loom_value_facts_unknown();
    loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
    return;
  }
  // Both exact: fold directly.
  if (lhs_facts.range_lo == lhs_facts.range_hi && rhs_lo == rhs_hi) {
    *out = loom_value_facts_exact_i64(
        loom_value_facts_remainder_i64(lhs_facts.range_lo, rhs_lo));
    loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
    return;
  }
  // Result magnitude is bounded by |divisor| - 1.
  const uint64_t maximum_divisor_magnitude = iree_max(
      iree_math_magnitude_i64(rhs_lo), iree_math_magnitude_i64(rhs_hi));
  // The nonzero divisor magnitude is in [1, 2^63], so subtracting one always
  // produces a representable signed upper bound.
  const int64_t remainder_bound =
      (int64_t)(maximum_divisor_magnitude - UINT64_C(1));
  // Signed remainder preserves the sign of the dividend. Conservative
  // bound: [-remainder_bound, remainder_bound].
  *out = loom_value_facts_make(-remainder_bound, remainder_bound, 1);
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
}
