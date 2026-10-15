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
