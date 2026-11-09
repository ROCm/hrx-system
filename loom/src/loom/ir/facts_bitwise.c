// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/math.h"
#include "loom/ir/facts.h"

//===----------------------------------------------------------------------===//
// Transfer functions: bitwise
//===----------------------------------------------------------------------===//

// Only power-of-two factors survive bitwise operations. Exact payloads also
// retain the low zero bits of zero and INT64_MIN, whose general divisor is one.
// Their strongest representable power-of-two divisor is 2^62: known_divisor
// must be positive and cannot represent 2^63.
static int64_t loom_value_facts_bitwise_divisor(
    const loom_value_facts_t* facts) {
  uint64_t bits = loom_value_facts_is_exact(*facts)
                      ? (uint64_t)facts->range_lo
                      : (uint64_t)facts->known_divisor;
  uint64_t divisor = bits & (UINT64_C(0) - bits);
  return divisor == 0 || divisor > INT64_MAX ? INT64_C(1) << 62
                                             : (int64_t)divisor;
}

// OR and XOR cannot introduce bits above either nonnegative operand's highest
// possible set bit. Filling the lower bits is necessary: combining just the
// endpoints would miss results such as 7 | 8 from two [0, 8] ranges. The signed
// nonnegative domain limits bit_count to 63, including an INT64_MAX endpoint.
static int64_t loom_value_facts_non_negative_bitwise_upper_bound(
    int64_t lhs_hi, int64_t rhs_hi) {
  uint64_t maximum_operand = (uint64_t)iree_max(lhs_hi, rhs_hi);
  if (maximum_operand == 0) {
    return 0;
  }
  uint32_t bit_count =
      64u - (uint32_t)iree_math_count_leading_zeros_u64(maximum_operand);
  return (int64_t)((UINT64_C(1) << bit_count) - 1);
}

static void loom_value_facts_propagate_bitwise_flags(
    const loom_value_facts_t* lhs, const loom_value_facts_t* rhs,
    loom_value_facts_t* out) {
  const bool preserves_subgroup_lane_mask =
      (loom_value_facts_is_subgroup_lane_mask(*lhs) &&
       loom_value_facts_is_subgroup_lane_mask(*rhs)) ||
      (loom_value_facts_is_subgroup_lane_mask(*lhs) &&
       loom_value_facts_is_zero(*rhs)) ||
      (loom_value_facts_is_zero(*lhs) &&
       loom_value_facts_is_subgroup_lane_mask(*rhs));
  if (preserves_subgroup_lane_mask) {
    loom_value_facts_mark_subgroup_lane_mask(out);
  }

  loom_value_facts_propagate_binary_distribution(*lhs, *rhs, out);
}

void loom_value_facts_andi(const loom_value_facts_t* lhs,
                           const loom_value_facts_t* rhs,
                           loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  int64_t lhs_lo = lhs_facts.range_lo, lhs_hi = lhs_facts.range_hi;
  int64_t rhs_lo = rhs_facts.range_lo, rhs_hi = rhs_facts.range_hi;

  // Exact folding.
  if (lhs_lo == lhs_hi && rhs_lo == rhs_hi) {
    *out = loom_value_facts_exact_i64(lhs_lo & rhs_lo);
    loom_value_facts_propagate_bitwise_flags(&lhs_facts, &rhs_facts, out);
    return;
  }
  // AND preserves every trailing zero bit proven on either input. Arbitrary
  // odd divisors do not survive masking, so retain only power-of-two factors.
  int64_t lhs_divisor = loom_value_facts_bitwise_divisor(&lhs_facts);
  int64_t rhs_divisor = loom_value_facts_bitwise_divisor(&rhs_facts);
  int64_t divisor = iree_max(lhs_divisor, rhs_divisor);

  // Any nonnegative operand clears the sign bit and bounds the result, even
  // when the other operand is a negative mask.
  int64_t lo = INT64_MIN;
  int64_t hi = INT64_MAX;
  if (lhs_lo >= 0) {
    lo = 0;
    hi = lhs_hi;
  }
  if (rhs_lo >= 0) {
    lo = 0;
    hi = iree_min(hi, rhs_hi);
  }
  // The largest possible result cannot set any of the retained low zero bits.
  *out = loom_value_facts_make(lo, hi & -divisor, divisor);
  loom_value_facts_propagate_bitwise_flags(&lhs_facts, &rhs_facts, out);
}

void loom_value_facts_ori(const loom_value_facts_t* lhs,
                          const loom_value_facts_t* rhs,
                          loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  int64_t lhs_lo = lhs_facts.range_lo, lhs_hi = lhs_facts.range_hi;
  int64_t rhs_lo = rhs_facts.range_lo;
  int64_t rhs_hi = rhs_facts.range_hi;

  // Exact folding.
  if (lhs_lo == lhs_hi && rhs_lo == rhs_hi) {
    *out = loom_value_facts_exact_i64(lhs_lo | rhs_lo);
    loom_value_facts_propagate_bitwise_flags(&lhs_facts, &rhs_facts, out);
    return;
  }

  int64_t divisor = iree_min(loom_value_facts_bitwise_divisor(&lhs_facts),
                             loom_value_facts_bitwise_divisor(&rhs_facts));
  // OR retains each nonnegative operand's set bits without introducing bits
  // above the wider operand's bound.
  int64_t lo = INT64_MIN;
  int64_t hi = INT64_MAX;
  if (lhs_lo >= 0 && rhs_lo >= 0) {
    lo = iree_max(lhs_lo, rhs_lo);
    hi = loom_value_facts_non_negative_bitwise_upper_bound(lhs_hi, rhs_hi);
  }
  *out = loom_value_facts_make(lo, hi & -divisor, divisor);
  // A set bit survives OR regardless of sign or how nonzero was established.
  if (loom_value_facts_is_non_zero(lhs_facts) ||
      loom_value_facts_is_non_zero(rhs_facts)) {
    out->flags |= LOOM_VALUE_FACT_NON_ZERO;
  }
  loom_value_facts_propagate_bitwise_flags(&lhs_facts, &rhs_facts, out);
}

void loom_value_facts_xori(const loom_value_facts_t* lhs,
                           const loom_value_facts_t* rhs,
                           loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  int64_t lhs_lo = lhs_facts.range_lo, lhs_hi = lhs_facts.range_hi;
  int64_t rhs_lo = rhs_facts.range_lo, rhs_hi = rhs_facts.range_hi;

  // Exact folding.
  if (lhs_lo == lhs_hi && rhs_lo == rhs_hi) {
    *out = loom_value_facts_exact_i64(lhs_lo ^ rhs_lo);
    loom_value_facts_propagate_bitwise_flags(&lhs_facts, &rhs_facts, out);
    return;
  }
  int64_t divisor = iree_min(loom_value_facts_bitwise_divisor(&lhs_facts),
                             loom_value_facts_bitwise_divisor(&rhs_facts));
  int64_t lo = INT64_MIN;
  int64_t hi = INT64_MAX;
  if (lhs_lo >= 0 && rhs_lo >= 0) {
    lo = 0;
    hi = loom_value_facts_non_negative_bitwise_upper_bound(lhs_hi, rhs_hi);
  }
  *out = loom_value_facts_make(lo, hi & -divisor, divisor);
  loom_value_facts_propagate_bitwise_flags(&lhs_facts, &rhs_facts, out);
}

//===----------------------------------------------------------------------===//
// Transfer functions: shifts
//===----------------------------------------------------------------------===//

// Performs a 64-bit arithmetic right shift without relying on the host C
// implementation's handling of signed negative operands. The shift must be in
// [0, 63].
static int64_t loom_value_facts_arithmetic_shift_right_i64(int64_t value,
                                                           uint32_t shift) {
  if (value >= 0) {
    return (int64_t)((uint64_t)value >> shift);
  }
  const uint64_t complement_magnitude = (uint64_t)(-(value + INT64_C(1)));
  return -INT64_C(1) - (int64_t)(complement_magnitude >> shift);
}

// The largest amount is at most 62, so both powers of two fit the signed
// fact carrier. Multiplication is monotonic within each sign half.
static loom_value_facts_t loom_value_facts_shift_left_range(
    loom_value_facts_t source, uint32_t minimum_shift, uint32_t maximum_shift) {
  const int64_t minimum_factor = INT64_C(1) << minimum_shift;
  const int64_t maximum_factor = INT64_C(1) << maximum_shift;
  int64_t divisor = 0;
  if (!iree_checked_mul_i64(source.known_divisor, minimum_factor, &divisor)) {
    divisor = source.known_divisor;
  }
  int64_t lo = 0, hi = 0;
  if (!iree_checked_mul_i64(
          source.range_lo,
          source.range_lo < 0 ? maximum_factor : minimum_factor, &lo) ||
      !iree_checked_mul_i64(
          source.range_hi,
          source.range_hi < 0 ? minimum_factor : maximum_factor, &hi)) {
    return loom_value_facts_make(INT64_MIN, INT64_MAX, divisor);
  }
  return loom_value_facts_make(lo, hi, divisor);
}

void loom_value_facts_shli(const loom_value_facts_t* lhs,
                           const loom_value_facts_t* rhs,
                           loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  if (rhs_facts.range_lo < 0 || rhs_facts.range_hi > 63) {
    *out = loom_value_facts_unknown();
  } else if (rhs_facts.range_hi < 63) {
    *out = loom_value_facts_shift_left_range(
        lhs_facts, (uint32_t)rhs_facts.range_lo, (uint32_t)rhs_facts.range_hi);
  } else {
    // Only the low source bit survives a shift by 63. Join that raw-bit
    // result with the ordinary range when other amounts are also possible.
    uint64_t lhs_bits = 0;
    if (loom_value_facts_as_exact_raw_bits(lhs_facts, 64, &lhs_bits)) {
      *out = loom_value_facts_make_signed_raw_bits(lhs_bits << 63, 64);
    } else if (lhs_facts.known_divisor % 2 == 0) {
      *out = loom_value_facts_exact_i64(0);
    } else {
      *out = loom_value_facts_make(INT64_MIN, 0, 1);
    }
    if (rhs_facts.range_lo < 63) {
      const loom_value_facts_t ordinary = loom_value_facts_shift_left_range(
          lhs_facts, (uint32_t)rhs_facts.range_lo, 62);
      const loom_value_facts_t sign_bit = *out;
      loom_value_facts_meet(&ordinary, &sign_bit, out);
    }
  }
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
}

void loom_value_facts_shrui(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs, int32_t bit_count,
                            loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts =
      bit_count == 64 ? *lhs : loom_value_facts_wrap_integer(*lhs, bit_count);
  const loom_value_facts_t rhs_facts = *rhs;
  const int64_t minimum_shift = rhs_facts.range_lo;
  const int64_t maximum_shift = rhs_facts.range_hi;
  if (minimum_shift < 0 || maximum_shift >= bit_count) {
    *out = loom_value_facts_unknown();
  } else if (maximum_shift == 0) {
    // Identity retains the signed fact domain, including the source sign bit.
    *out = lhs_facts;
  } else {
    const uint64_t mask = UINT64_MAX >> (64 - bit_count);
    const int64_t positive_shift = iree_max(minimum_shift, INT64_C(1));
    int64_t lo = 0;
    int64_t hi = (int64_t)(mask >> positive_shift);
    if (lhs_facts.range_lo >= 0 || lhs_facts.range_hi < 0) {
      // Within either half of the signed domain, raw bits are monotonic.
      // A range crossing zero spans both ends of the unsigned domain.
      lo = (int64_t)(((uint64_t)lhs_facts.range_lo & mask) >> maximum_shift);
      hi = (int64_t)(((uint64_t)lhs_facts.range_hi & mask) >> positive_shift);
    }
    if (minimum_shift == 0) {
      // A zero shift preserves signed values; every positive shift clears
      // the sign bit. Both interpretations must remain represented.
      lo = iree_min(lo, lhs_facts.range_lo);
      hi = iree_max(hi, lhs_facts.range_hi);
    }
    int64_t divisor = lhs_facts.known_divisor;
    if (lhs_facts.range_lo < 0) {
      // Reinterpreting a signed value adds 2^bit_count. Only the power-of-two
      // factor of its divisor necessarily survives that addition.
      divisor &= -divisor;
    }
    const uint64_t factor = UINT64_C(1) << maximum_shift;
    divisor = (uint64_t)divisor % factor == 0
                  ? (int64_t)((uint64_t)divisor / factor)
                  : 1;
    *out = loom_value_facts_make(lo, hi, divisor);
  }
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
}

void loom_value_facts_shrsi(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  if (rhs_facts.range_lo < 0 || rhs_facts.range_hi > 63) {
    *out = loom_value_facts_unknown();
  } else {
    const uint32_t minimum_shift = (uint32_t)rhs_facts.range_lo;
    const uint32_t maximum_shift = (uint32_t)rhs_facts.range_hi;
    const int64_t lo = loom_value_facts_arithmetic_shift_right_i64(
        lhs_facts.range_lo,
        lhs_facts.range_lo < 0 ? minimum_shift : maximum_shift);
    const int64_t hi = loom_value_facts_arithmetic_shift_right_i64(
        lhs_facts.range_hi,
        lhs_facts.range_hi < 0 ? maximum_shift : minimum_shift);
    const uint64_t factor = UINT64_C(1) << maximum_shift;
    const int64_t divisor =
        (uint64_t)lhs_facts.known_divisor % factor == 0
            ? (int64_t)((uint64_t)lhs_facts.known_divisor / factor)
            : 1;
    *out = loom_value_facts_make(lo, hi, divisor);
  }
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
}

static void loom_value_facts_rotate(const loom_value_facts_t* lhs,
                                    const loom_value_facts_t* rhs,
                                    int32_t bit_count, bool rotate_left,
                                    loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  int64_t value = 0;
  int64_t amount = 0;
  if (bit_count <= 0 || bit_count > 64 ||
      !loom_value_facts_as_exact_i64(lhs_facts, &value) ||
      !loom_value_facts_as_exact_i64(rhs_facts, &amount)) {
    *out = loom_value_facts_unknown();
    loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
    return;
  }

  int64_t normalized_amount = amount % bit_count;
  if (normalized_amount < 0) {
    normalized_amount += bit_count;
  }
  uint32_t shift = (uint32_t)normalized_amount;
  uint64_t mask =
      bit_count == 64 ? UINT64_MAX : (UINT64_C(1) << (uint32_t)bit_count) - 1;
  uint64_t raw_value = (uint64_t)value & mask;
  uint64_t rotated = raw_value;
  if (shift != 0) {
    uint32_t complement = (uint32_t)bit_count - shift;
    rotated = rotate_left ? (raw_value << shift) | (raw_value >> complement)
                          : (raw_value >> shift) | (raw_value << complement);
  }
  *out = bit_count == 1
             ? loom_value_facts_exact_i64((int64_t)(rotated & 1))
             : loom_value_facts_make_signed_raw_bits(rotated & mask, bit_count);
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
}

void loom_value_facts_rotli(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs, int32_t bit_count,
                            loom_value_facts_t* out) {
  loom_value_facts_rotate(lhs, rhs, bit_count, /*rotate_left=*/true, out);
}

void loom_value_facts_rotri(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs, int32_t bit_count,
                            loom_value_facts_t* out) {
  loom_value_facts_rotate(lhs, rhs, bit_count, /*rotate_left=*/false, out);
}
