// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/facts.h"

#include <math.h>

// Computes flags from integer range facts. Does not set floating-point, float
// predicate, or execution distribution flags; those are preserved by the caller
// when appropriate.
static uint32_t loom_value_facts_compute_flags(int64_t lo, int64_t hi) {
  uint32_t flags = 0;
  if (lo == hi) flags |= LOOM_VALUE_FACT_EXACT;
  if (lo >= 0) flags |= LOOM_VALUE_FACT_NON_NEGATIVE;
  if (lo > 0 || hi < 0) flags |= LOOM_VALUE_FACT_NON_ZERO;
  if (lo > 0) flags |= LOOM_VALUE_FACT_POSITIVE;
  if (lo >= 0 && hi <= 1) flags |= LOOM_VALUE_FACT_BOOLEAN;
  // Power-of-two only provable for exact values from range alone.
  if (lo == hi && lo > 0 && (lo & (lo - 1)) == 0) {
    flags |= LOOM_VALUE_FACT_POWER_OF_TWO;
  }
  return flags;
}

static int64_t loom_value_facts_exact_i64_divisor(int64_t value) {
  // llabs(INT64_MIN) is undefined behavior. Use 1 as the conservative
  // divisor for the INT64_MIN and zero cases.
  return (value != 0 && value != INT64_MIN) ? llabs(value) : 1;
}

void loom_value_facts_propagate_unary_distribution(loom_value_facts_t input,
                                                   loom_value_facts_t* out) {
  if (loom_value_facts_is_exact(*out)) {
    loom_value_facts_mark_uniform(out);
  } else if (loom_value_facts_is_lane_predicate(input) ||
             loom_value_facts_is_lane_varying(input)) {
    loom_value_facts_mark_lane_varying(out);
  } else if (loom_value_facts_is_uniform(input)) {
    loom_value_facts_mark_uniform(out);
  }
}

void loom_value_facts_propagate_binary_distribution(loom_value_facts_t lhs,
                                                    loom_value_facts_t rhs,
                                                    loom_value_facts_t* out) {
  if (loom_value_facts_is_exact(*out)) {
    loom_value_facts_mark_uniform(out);
  } else if (loom_value_facts_is_lane_predicate(lhs) ||
             loom_value_facts_is_lane_predicate(rhs) ||
             loom_value_facts_is_lane_varying(lhs) ||
             loom_value_facts_is_lane_varying(rhs)) {
    loom_value_facts_mark_lane_varying(out);
  } else if (loom_value_facts_is_uniform(lhs) &&
             loom_value_facts_is_uniform(rhs)) {
    loom_value_facts_mark_uniform(out);
  }
}

void loom_value_facts_propagate_ternary_distribution(loom_value_facts_t a,
                                                     loom_value_facts_t b,
                                                     loom_value_facts_t c,
                                                     loom_value_facts_t* out) {
  if (loom_value_facts_is_exact(*out)) {
    loom_value_facts_mark_uniform(out);
  } else if (loom_value_facts_is_lane_predicate(a) ||
             loom_value_facts_is_lane_predicate(b) ||
             loom_value_facts_is_lane_predicate(c) ||
             loom_value_facts_is_lane_varying(a) ||
             loom_value_facts_is_lane_varying(b) ||
             loom_value_facts_is_lane_varying(c)) {
    loom_value_facts_mark_lane_varying(out);
  } else if (loom_value_facts_is_uniform(a) && loom_value_facts_is_uniform(b) &&
             loom_value_facts_is_uniform(c)) {
    loom_value_facts_mark_uniform(out);
  }
}

//===----------------------------------------------------------------------===//
// Constructors
//===----------------------------------------------------------------------===//

loom_value_facts_t loom_value_facts_exact_i64(int64_t value) {
  loom_value_facts_t facts = {0};
  facts.range_lo = value;
  facts.range_hi = value;
  facts.known_divisor = loom_value_facts_exact_i64_divisor(value);
  facts.flags = loom_value_facts_compute_flags(value, value);
  loom_value_facts_mark_uniform(&facts);
  return facts;
}

loom_value_facts_t loom_value_facts_exact_f64(double value) {
  loom_value_facts_t facts = {0};
  memcpy(&facts.range_lo, &value, sizeof(double));
  facts.range_hi = facts.range_lo;
  facts.known_divisor = 1;
  facts.flags = LOOM_VALUE_FACT_EXACT | LOOM_VALUE_FACT_FLOAT;
  if (!isnan(value)) facts.flags |= LOOM_VALUE_FACT_NOT_NAN;
  if (!isinf(value)) facts.flags |= LOOM_VALUE_FACT_NOT_INF;
  if (isfinite(value)) {
    facts.flags |= LOOM_VALUE_FACT_NOT_NAN | LOOM_VALUE_FACT_NOT_INF |
                   LOOM_VALUE_FACT_FINITE;
  }
  loom_value_facts_mark_uniform(&facts);
  return facts;
}

loom_value_facts_t loom_value_facts_make(int64_t lo, int64_t hi,
                                         int64_t known_divisor) {
  if (known_divisor < 1) known_divisor = 1;
  // Invalid range: fall back to unknown.
  if (lo > hi) return loom_value_facts_unknown();
  if (lo == hi) known_divisor = loom_value_facts_exact_i64_divisor(lo);
  loom_value_facts_t facts = {0};
  facts.range_lo = lo;
  facts.range_hi = hi;
  facts.known_divisor = known_divisor;
  facts.flags = loom_value_facts_compute_flags(lo, hi);
  return facts;
}

bool loom_value_facts_scalar_type_domain(loom_scalar_type_t scalar_type,
                                         int64_t* out_lo, int64_t* out_hi) {
  return loom_scalar_type_integer_domain(scalar_type, out_lo, out_hi);
}

loom_value_facts_t loom_value_facts_clamp_domain(loom_value_facts_t facts,
                                                 int64_t lo, int64_t hi) {
  if (loom_value_facts_is_float(facts)) {
    return loom_value_facts_make(lo, hi, 1);
  }
  int64_t range_lo = loom_max_i64(facts.range_lo, lo);
  int64_t range_hi = loom_min_i64(facts.range_hi, hi);
  if (range_lo > range_hi) {
    return loom_value_facts_make(lo, hi, 1);
  }
  loom_value_facts_t result =
      loom_value_facts_make(range_lo, range_hi, facts.known_divisor);
  if (iree_any_bit_set(facts.flags, LOOM_VALUE_FACT_POWER_OF_TWO) &&
      range_hi > 0) {
    result.flags |= LOOM_VALUE_FACT_POWER_OF_TWO;
  }
  const uint32_t preserved_topology_flags =
      range_lo == facts.range_lo && range_hi == facts.range_hi
          ? facts.flags & LOOM_VALUE_FACT_TOPOLOGY_DOMAIN_MASK
          : 0;
  result.flags |=
      (facts.flags &
       (LOOM_VALUE_FACT_UNIFORM | LOOM_VALUE_FACT_LANE_VARYING |
        LOOM_VALUE_FACT_LANE_PREDICATE | LOOM_VALUE_FACT_SUBGROUP_LANE_MASK)) |
      preserved_topology_flags;
  result.extension_id = facts.extension_id;
  return result;
}

//===----------------------------------------------------------------------===//
// Raw integer bit helpers
//===----------------------------------------------------------------------===//

bool loom_value_facts_as_exact_raw_bits(loom_value_facts_t facts,
                                        int32_t bit_count, uint64_t* out_bits) {
  int64_t value = 0;
  if (bit_count <= 0 || bit_count > 64 ||
      !loom_value_facts_as_exact_i64(facts, &value)) {
    return false;
  }
  *out_bits = loom_mask_to_bitwidth_u64((uint64_t)value, bit_count);
  return true;
}

bool loom_value_facts_make_unsigned_raw_bits(uint64_t raw_bits,
                                             int32_t bit_count,
                                             loom_value_facts_t* out_facts) {
  if (bit_count <= 0 || bit_count > 64) {
    return false;
  }
  uint64_t masked = loom_mask_to_bitwidth_u64(raw_bits, bit_count);
  if (masked > (uint64_t)INT64_MAX) {
    return false;
  }
  *out_facts = loom_value_facts_exact_i64((int64_t)masked);
  return true;
}

static int64_t loom_value_facts_sign_extend_raw_bits(uint64_t raw_bits,
                                                     int32_t bit_count) {
  if (bit_count <= 0) return 0;
  if (bit_count >= 64) return (int64_t)raw_bits;
  uint64_t sign_bit = UINT64_C(1) << (bit_count - 1);
  uint64_t mask = (UINT64_C(1) << bit_count) - 1;
  uint64_t masked = raw_bits & mask;
  return (int64_t)((masked ^ sign_bit) - sign_bit);
}

loom_value_facts_t loom_value_facts_make_signed_raw_bits(uint64_t raw_bits,
                                                         int32_t bit_count) {
  if (bit_count <= 0 || bit_count > 64) {
    return loom_value_facts_unknown();
  }
  return loom_value_facts_exact_i64(
      loom_value_facts_sign_extend_raw_bits(raw_bits, bit_count));
}

loom_value_facts_t loom_value_facts_make_unsigned_bit_count_range(
    int64_t bit_count) {
  if (bit_count <= 0 || bit_count > 63) {
    return loom_value_facts_unknown();
  }
  const int64_t maximum_value =
      bit_count == 63 ? INT64_MAX : (int64_t)((UINT64_C(1) << bit_count) - 1);
  return loom_value_facts_make(0, maximum_value, 1);
}

loom_value_facts_t loom_value_facts_make_signed_bit_count_range(
    int64_t bit_count) {
  if (bit_count <= 0 || bit_count > 64) {
    return loom_value_facts_unknown();
  }
  if (bit_count == 64) {
    return loom_value_facts_make(INT64_MIN, INT64_MAX, 1);
  }
  const int64_t minimum_value = -(INT64_C(1) << (bit_count - 1));
  const int64_t maximum_value = (INT64_C(1) << (bit_count - 1)) - 1;
  return loom_value_facts_make(minimum_value, maximum_value, 1);
}

static bool loom_value_facts_extract_bitfield(loom_value_facts_t source,
                                              int32_t source_bit_count,
                                              int64_t offset, int64_t width,
                                              bool signed_extract,
                                              loom_value_facts_t* out_facts) {
  if (offset < 0 || width <= 0 || source_bit_count <= 0 ||
      source_bit_count > 64 || offset > source_bit_count ||
      width > source_bit_count - offset || width > 64) {
    return false;
  }
  uint64_t source_bits = 0;
  if (!loom_value_facts_as_exact_raw_bits(source, source_bit_count,
                                          &source_bits)) {
    *out_facts = signed_extract
                     ? loom_value_facts_make_signed_bit_count_range(width)
                     : loom_value_facts_make_unsigned_bit_count_range(width);
    return true;
  }
  uint64_t field_bits =
      loom_mask_to_bitwidth_u64(source_bits >> offset, (int32_t)width);
  if (signed_extract) {
    *out_facts =
        loom_value_facts_make_signed_raw_bits(field_bits, (int32_t)width);
    return true;
  }
  return loom_value_facts_make_unsigned_raw_bits(field_bits, (int32_t)width,
                                                 out_facts);
}

bool loom_value_facts_extract_unsigned_bitfield(loom_value_facts_t source,
                                                int32_t source_bit_count,
                                                int64_t offset, int64_t width,
                                                loom_value_facts_t* out_facts) {
  return loom_value_facts_extract_bitfield(source, source_bit_count, offset,
                                           width, false, out_facts);
}

bool loom_value_facts_extract_signed_bitfield(loom_value_facts_t source,
                                              int32_t source_bit_count,
                                              int64_t offset, int64_t width,
                                              loom_value_facts_t* out_facts) {
  return loom_value_facts_extract_bitfield(source, source_bit_count, offset,
                                           width, true, out_facts);
}

//===----------------------------------------------------------------------===//
// Flag recomputation
//===----------------------------------------------------------------------===//

void loom_value_facts_recompute_flags(loom_value_facts_t* facts) {
  // Preserve flags that come from external sources (predicates, not
  // range analysis).
  uint32_t preserved =
      facts->flags &
      (LOOM_VALUE_FACT_POWER_OF_TWO | LOOM_VALUE_FACT_FLOAT |
       LOOM_VALUE_FACT_NOT_NAN | LOOM_VALUE_FACT_NOT_INF |
       LOOM_VALUE_FACT_FINITE | LOOM_VALUE_FACT_UNIFORM |
       LOOM_VALUE_FACT_LANE_VARYING | LOOM_VALUE_FACT_LANE_PREDICATE |
       LOOM_VALUE_FACT_SUBGROUP_LANE_MASK |
       LOOM_VALUE_FACT_TOPOLOGY_DOMAIN_MASK);
  facts->flags =
      loom_value_facts_compute_flags(facts->range_lo, facts->range_hi) |
      preserved;
}

//===----------------------------------------------------------------------===//
// Predicate application
//===----------------------------------------------------------------------===//

void loom_value_facts_apply_predicate(loom_value_facts_t* facts,
                                      const loom_predicate_t* predicate) {
  // This scalar fact lattice can consume predicates with literal bounds. Value
  // operands are still useful to symbolic relation analysis, but treating a
  // value ID as an integer literal here would corrupt range facts.
  if (predicate->kind == LOOM_PREDICATE_POW2 ||
      predicate->kind == LOOM_PREDICATE_NOT_NAN ||
      predicate->kind == LOOM_PREDICATE_NOT_INF ||
      predicate->kind == LOOM_PREDICATE_FINITE) {
    if (predicate->arg_count < 1 ||
        predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE) {
      return;
    }
  } else if (predicate->kind == LOOM_PREDICATE_RANGE) {
    if (predicate->arg_count < 3 ||
        predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE ||
        predicate->arg_tags[1] != LOOM_PRED_ARG_CONST ||
        predicate->arg_tags[2] != LOOM_PRED_ARG_CONST) {
      return;
    }
  } else if (predicate->arg_count < 2 ||
             predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE ||
             predicate->arg_tags[1] != LOOM_PRED_ARG_CONST) {
    return;
  }

  // The constant argument is in args[1] for most predicates. For RANGE, lo is
  // in args[1] and hi is in args[2].
  int64_t constant = predicate->args[1];
  uint32_t preserved_predicate_flags = facts->flags & LOOM_VALUE_FACT_NON_ZERO;

  switch ((loom_predicate_kind_t)predicate->kind) {
    case LOOM_PREDICATE_EQ:
      *facts = loom_value_facts_exact_i64(constant);
      return;  // exact_i64 already computes all flags.

    case LOOM_PREDICATE_NE:
      // A not-equal predicate excludes one value. This lattice has intervals
      // rather than disjoint ranges, so the only generally-sound tightening is
      // when the excluded value sits on an interval edge.
      if (constant == facts->range_lo && facts->range_lo < facts->range_hi &&
          constant < INT64_MAX) {
        facts->range_lo = constant + 1;
      } else if (constant == facts->range_hi &&
                 facts->range_lo < facts->range_hi && constant > INT64_MIN) {
        facts->range_hi = constant - 1;
      }
      loom_value_facts_recompute_flags(facts);
      if (constant == 0) {
        facts->flags |= LOOM_VALUE_FACT_NON_ZERO;
      }
      return;

    case LOOM_PREDICATE_LT:
      // a < N → range_hi = min(range_hi, N - 1).
      if (constant > INT64_MIN) {  // N - 1 would underflow for INT64_MIN.
        facts->range_hi = loom_min_i64(facts->range_hi, constant - 1);
      }
      break;

    case LOOM_PREDICATE_LE:
      facts->range_hi = loom_min_i64(facts->range_hi, constant);
      break;

    case LOOM_PREDICATE_GT:
      // a > N → range_lo = max(range_lo, N + 1).
      if (constant < INT64_MAX) {  // N + 1 would overflow for INT64_MAX.
        facts->range_lo = loom_max_i64(facts->range_lo, constant + 1);
      }
      break;

    case LOOM_PREDICATE_GE:
      facts->range_lo = loom_max_i64(facts->range_lo, constant);
      break;

    case LOOM_PREDICATE_MUL: {
      // a is a multiple of N → known_divisor = lcm(known_divisor, N).
      int64_t new_divisor;
      if (loom_lcm_i64(facts->known_divisor, constant, &new_divisor)) {
        facts->known_divisor = new_divisor;
      }
      // On overflow: keep existing divisor (conservative).
      break;
    }

    case LOOM_PREDICATE_MIN:
      // a >= N (same semantics as GE with a constant).
      facts->range_lo = loom_max_i64(facts->range_lo, constant);
      break;

    case LOOM_PREDICATE_MAX:
      // a <= N (same semantics as LE with a constant).
      facts->range_hi = loom_min_i64(facts->range_hi, constant);
      break;

    case LOOM_PREDICATE_POW2:
      facts->flags |= LOOM_VALUE_FACT_POWER_OF_TWO;
      break;

    case LOOM_PREDICATE_NOT_NAN:
      facts->flags |= LOOM_VALUE_FACT_NOT_NAN;
      if (loom_value_facts_is_not_inf(*facts)) {
        facts->flags |= LOOM_VALUE_FACT_FINITE;
      }
      return;

    case LOOM_PREDICATE_NOT_INF:
      facts->flags |= LOOM_VALUE_FACT_NOT_INF;
      if (loom_value_facts_is_not_nan(*facts)) {
        facts->flags |= LOOM_VALUE_FACT_FINITE;
      }
      return;

    case LOOM_PREDICATE_FINITE:
      facts->flags |= LOOM_VALUE_FACT_NOT_NAN | LOOM_VALUE_FACT_NOT_INF |
                      LOOM_VALUE_FACT_FINITE;
      return;

    case LOOM_PREDICATE_RANGE: {
      int64_t lo = predicate->args[1];
      int64_t hi = predicate->args[2];
      facts->range_lo = loom_max_i64(facts->range_lo, lo);
      facts->range_hi = loom_min_i64(facts->range_hi, hi);
      break;
    }

    default:
      break;
  }

  loom_value_facts_recompute_flags(facts);
  facts->flags |= preserved_predicate_flags;
}

//===----------------------------------------------------------------------===//
// Transfer functions: binary arithmetic
//===----------------------------------------------------------------------===//

void loom_value_facts_addi(const loom_value_facts_t* lhs,
                           const loom_value_facts_t* rhs,
                           loom_value_facts_t* out) {
  // Read inputs before writing output (output may alias one input).
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  int64_t lhs_lo = lhs_facts.range_lo, lhs_hi = lhs_facts.range_hi;
  int64_t rhs_lo = rhs_facts.range_lo, rhs_hi = rhs_facts.range_hi;
  int64_t lhs_divisor = lhs_facts.known_divisor;
  int64_t rhs_divisor = rhs_facts.known_divisor;

  // Divisor is independent of range — always computable.
  int64_t divisor = loom_gcd_i64(lhs_divisor, rhs_divisor);

  int64_t lo, hi;
  if (!loom_checked_add_i64(lhs_lo, rhs_lo, &lo) ||
      !loom_checked_add_i64(lhs_hi, rhs_hi, &hi)) {
    // Range overflowed but divisor is still valid.
    *out = loom_value_facts_make(INT64_MIN, INT64_MAX, divisor);
    loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
    return;
  }
  *out = loom_value_facts_make(lo, hi, divisor);
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
}

void loom_value_facts_subi(const loom_value_facts_t* lhs,
                           const loom_value_facts_t* rhs,
                           loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  int64_t lhs_lo = lhs_facts.range_lo, lhs_hi = lhs_facts.range_hi;
  int64_t rhs_lo = rhs_facts.range_lo, rhs_hi = rhs_facts.range_hi;
  int64_t lhs_divisor = lhs_facts.known_divisor;
  int64_t rhs_divisor = rhs_facts.known_divisor;

  int64_t divisor = loom_gcd_i64(lhs_divisor, rhs_divisor);

  // Subtraction: lo = lhs_lo - rhs_hi, hi = lhs_hi - rhs_lo (bound swap).
  int64_t lo, hi;
  if (!loom_checked_sub_i64(lhs_lo, rhs_hi, &lo) ||
      !loom_checked_sub_i64(lhs_hi, rhs_lo, &hi)) {
    *out = loom_value_facts_make(INT64_MIN, INT64_MAX, divisor);
    loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
    return;
  }
  *out = loom_value_facts_make(lo, hi, divisor);
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
}

void loom_value_facts_muli(const loom_value_facts_t* lhs,
                           const loom_value_facts_t* rhs,
                           loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  int64_t lhs_lo = lhs_facts.range_lo, lhs_hi = lhs_facts.range_hi;
  int64_t rhs_lo = rhs_facts.range_lo, rhs_hi = rhs_facts.range_hi;
  int64_t lhs_divisor = lhs_facts.known_divisor;
  int64_t rhs_divisor = rhs_facts.known_divisor;

  // Divisibility: product of divisors, fallback to GCD on overflow.
  // Computed before range so it's available on range overflow.
  int64_t divisor = 0;
  if (!loom_checked_mul_i64(lhs_divisor, rhs_divisor, &divisor)) {
    divisor = loom_gcd_i64(lhs_divisor, rhs_divisor);
  }

  // Four corner products; take min/max.
  int64_t corners[4];
  if (!loom_checked_mul_i64(lhs_lo, rhs_lo, &corners[0]) ||
      !loom_checked_mul_i64(lhs_lo, rhs_hi, &corners[1]) ||
      !loom_checked_mul_i64(lhs_hi, rhs_lo, &corners[2]) ||
      !loom_checked_mul_i64(lhs_hi, rhs_hi, &corners[3])) {
    *out = loom_value_facts_make(INT64_MIN, INT64_MAX, divisor);
    loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
    return;
  }
  int64_t lo = corners[0], hi = corners[0];
  for (int i = 1; i < 4; ++i) {
    if (corners[i] < lo) lo = corners[i];
    if (corners[i] > hi) hi = corners[i];
  }

  *out = loom_value_facts_make(lo, hi, divisor);
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
}

void loom_value_facts_divui(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  int64_t lhs_lo = lhs_facts.range_lo, lhs_hi = lhs_facts.range_hi;
  int64_t rhs_lo = rhs_facts.range_lo, rhs_hi = rhs_facts.range_hi;
  int64_t lhs_divisor = lhs_facts.known_divisor;

  // Compute divisor first (independent of range sign).
  int64_t divisor = 1;
  if (rhs_lo == rhs_hi && rhs_lo > 0 && lhs_divisor % rhs_lo == 0) {
    divisor = lhs_divisor / rhs_lo;
  }

  // Unsigned division requires both operands non-negative and
  // divisor provably positive (no division by zero).
  if (lhs_lo < 0 || rhs_lo <= 0) {
    *out = loom_value_facts_make(INT64_MIN, INT64_MAX, divisor);
    loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
    return;
  }
  // Smallest numerator / largest denominator gives lowest result.
  int64_t lo = lhs_lo / rhs_hi;
  int64_t hi = lhs_hi / rhs_lo;
  *out = loom_value_facts_make(lo, hi, divisor);
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
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

void loom_value_facts_remui(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  int64_t lhs_lo = lhs_facts.range_lo, lhs_hi = lhs_facts.range_hi;
  int64_t rhs_lo = rhs_facts.range_lo, rhs_hi = rhs_facts.range_hi;
  int64_t lhs_divisor = lhs_facts.known_divisor;

  // Requires non-negative operands and positive divisor.
  if (lhs_lo < 0 || rhs_lo <= 0) {
    *out = loom_value_facts_unknown();
    loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
    return;
  }
  // Both exact: fold directly. Non-negative int64 facts are a subset of the
  // unsigned domain and the result remains representable as int64.
  if (lhs_facts.range_lo == lhs_facts.range_hi && rhs_lo == rhs_hi) {
    *out = loom_value_facts_exact_i64(lhs_lo % rhs_lo);
    loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
    return;
  }
  // If exact divisor and numerator divisible: remainder is zero.
  if (rhs_lo == rhs_hi && lhs_divisor % rhs_lo == 0) {
    *out = loom_value_facts_exact_i64(0);
    loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
    return;
  }
  // Result is in [0, min(dividend, divisor - 1)]. This is important for
  // dynamic divisor shapes where the divisor upper bound may be much wider than
  // the lane or flat-index dividend being decomposed.
  *out = loom_value_facts_make(0, loom_min_i64(lhs_hi, rhs_hi - 1), 1);
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
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
    *out = loom_value_facts_exact_i64(lhs_facts.range_lo % rhs_lo);
    loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
    return;
  }
  // Result magnitude is bounded by |divisor| - 1.
  // Use unsigned abs to avoid llabs(INT64_MIN) UB.
  uint64_t abs_lo =
      (rhs_lo >= 0) ? (uint64_t)rhs_lo : (uint64_t)(-(rhs_lo + 1)) + 1;
  uint64_t abs_hi =
      (rhs_hi >= 0) ? (uint64_t)rhs_hi : (uint64_t)(-(rhs_hi + 1)) + 1;
  int64_t abs_bound = (int64_t)(abs_lo > abs_hi ? abs_lo : abs_hi);
  // Signed remainder preserves the sign of the dividend. Conservative
  // bound: [-(abs_bound - 1), abs_bound - 1].
  *out = loom_value_facts_make(-(abs_bound - 1), abs_bound - 1, 1);
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
}

//===----------------------------------------------------------------------===//
// Transfer functions: shifts
//===----------------------------------------------------------------------===//

void loom_value_facts_shli(const loom_value_facts_t* lhs,
                           const loom_value_facts_t* rhs,
                           loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  int64_t lhs_lo = lhs_facts.range_lo, lhs_hi = lhs_facts.range_hi;
  int64_t rhs_lo = rhs_facts.range_lo, rhs_hi = rhs_facts.range_hi;
  int64_t lhs_divisor = lhs_facts.known_divisor;

  // Shift amount must be exact and in [0, 63].
  if (rhs_lo != rhs_hi || rhs_lo < 0 || rhs_lo > 63) {
    *out = loom_value_facts_unknown();
    loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
    return;
  }
  int64_t shift = rhs_lo;
  int64_t factor = (int64_t)1 << shift;

  // Divisor: lhs_divisor * 2^shift. Computed before range.
  int64_t divisor = 0;
  if (!loom_checked_mul_i64(lhs_divisor, factor, &divisor)) {
    divisor = lhs_divisor;  // Overflow: keep existing.
  }

  int64_t lo, hi;
  if (!loom_checked_mul_i64(lhs_lo, factor, &lo) ||
      !loom_checked_mul_i64(lhs_hi, factor, &hi)) {
    *out = loom_value_facts_make(INT64_MIN, INT64_MAX, divisor);
    loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
    return;
  }
  *out = loom_value_facts_make(lo, hi, divisor);
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
}

void loom_value_facts_shrui(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  int64_t lhs_lo = lhs_facts.range_lo, lhs_hi = lhs_facts.range_hi;
  int64_t rhs_lo = rhs_facts.range_lo, rhs_hi = rhs_facts.range_hi;
  int64_t lhs_divisor = lhs_facts.known_divisor;

  // Shift amount must be exact and in [0, 63].
  if (rhs_lo != rhs_hi || rhs_lo < 0 || rhs_lo > 63) {
    *out = loom_value_facts_unknown();
    loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
    return;
  }
  int64_t shift = rhs_lo;
  int64_t factor = (int64_t)1 << shift;

  // Divisor: independent of sign. Computed before range check.
  int64_t divisor = (lhs_divisor % factor == 0) ? lhs_divisor / factor : 1;

  // Range requires non-negative input for unsigned shift semantics.
  if (lhs_lo < 0) {
    *out = loom_value_facts_make(INT64_MIN, INT64_MAX, divisor);
    loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
    return;
  }
  int64_t lo = lhs_lo >> shift;
  int64_t hi = lhs_hi >> shift;
  *out = loom_value_facts_make(lo, hi, divisor);
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
}

void loom_value_facts_shrsi(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  int64_t lhs_lo = lhs_facts.range_lo, lhs_hi = lhs_facts.range_hi;
  int64_t rhs_lo = rhs_facts.range_lo, rhs_hi = rhs_facts.range_hi;
  int64_t lhs_divisor = lhs_facts.known_divisor;

  // Shift amount must be exact and in [0, 63].
  if (rhs_lo != rhs_hi || rhs_lo < 0 || rhs_lo > 63) {
    *out = loom_value_facts_unknown();
    loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
    return;
  }
  int64_t shift = rhs_lo;
  // Arithmetic right shift preserves sign.
  int64_t lo = lhs_lo >> shift;
  int64_t hi = lhs_hi >> shift;
  int64_t factor = (int64_t)1 << shift;
  int64_t divisor = (lhs_divisor % factor == 0) ? lhs_divisor / factor : 1;
  *out = loom_value_facts_make(lo, hi, divisor);
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
}

//===----------------------------------------------------------------------===//
// Transfer functions: bitwise
//===----------------------------------------------------------------------===//

static int64_t loom_value_facts_non_negative_bitwise_upper_bound(
    int64_t lhs_hi, int64_t rhs_hi) {
  uint64_t maximum_operand = (uint64_t)loom_max_i64(lhs_hi, rhs_hi);
  if (maximum_operand == 0) return 0;
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

  if (loom_value_facts_is_exact(*out)) {
    loom_value_facts_mark_uniform(out);
  } else if (loom_value_facts_is_lane_predicate(*lhs) ||
             loom_value_facts_is_lane_predicate(*rhs) ||
             loom_value_facts_is_lane_varying(*lhs) ||
             loom_value_facts_is_lane_varying(*rhs)) {
    loom_value_facts_mark_lane_varying(out);
  } else if (loom_value_facts_is_uniform(*lhs) &&
             loom_value_facts_is_uniform(*rhs)) {
    loom_value_facts_mark_uniform(out);
  }
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
  // If either operand is an exact mask, derive divisibility from its
  // trailing zeros: andi(%x, mask) clears the bottom ctz(mask) bits,
  // so the result is divisible by 2^ctz(mask). AND forces divisibility
  // (LCM, not GCD — the mask guarantees the trailing zeros).
  int64_t exact_mask = 0;
  int64_t other_divisor = 1;
  if (rhs_lo == rhs_hi && rhs_lo != 0) {
    exact_mask = rhs_lo;
    other_divisor = lhs_facts.known_divisor;
  } else if (lhs_lo == lhs_hi && lhs_lo != 0) {
    exact_mask = lhs_lo;
    other_divisor = rhs_facts.known_divisor;
  }
  if (exact_mask != 0) {
    int64_t mask_divisor =
        (int64_t)1 << iree_math_count_trailing_zeros_u64((uint64_t)exact_mask);
    int64_t divisor = 0;
    if (!loom_lcm_i64(other_divisor, mask_divisor, &divisor)) {
      divisor = mask_divisor;
    }
    if (exact_mask >= 0) {
      *out = loom_value_facts_make(0, exact_mask, divisor);
    } else {
      *out = loom_value_facts_make(INT64_MIN, INT64_MAX, divisor);
    }
    loom_value_facts_propagate_bitwise_flags(&lhs_facts, &rhs_facts, out);
    return;
  }
  // Both non-negative: result is non-negative with bounded range.
  if (lhs_lo >= 0 && rhs_lo >= 0) {
    *out = loom_value_facts_make(0, loom_min_i64(lhs_hi, rhs_hi), 1);
    loom_value_facts_propagate_bitwise_flags(&lhs_facts, &rhs_facts, out);
    return;
  }
  *out = loom_value_facts_unknown();
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

  // OR with a known non-zero exact value always produces non-zero.
  // Use range [1, MAX] when either operand is exact and non-zero
  // and the other is non-negative.
  bool either_exact_nonzero =
      (lhs_lo == lhs_hi && lhs_lo != 0) || (rhs_lo == rhs_hi && rhs_lo != 0);

  // Both non-negative: result is non-negative.
  if (lhs_lo >= 0 && rhs_lo >= 0) {
    int64_t lo = loom_max_i64(lhs_lo, rhs_lo);
    if (either_exact_nonzero && lo == 0) lo = 1;
    *out = loom_value_facts_make(lo, INT64_MAX, 1);
    loom_value_facts_propagate_bitwise_flags(&lhs_facts, &rhs_facts, out);
    return;
  }
  // General case: if either operand is exact non-zero, the result
  // is guaranteed non-zero (OR preserves set bits).
  if (either_exact_nonzero) {
    *out = loom_value_facts_make(INT64_MIN, INT64_MAX, 1);
    out->flags |= LOOM_VALUE_FACT_NON_ZERO;
  } else {
    *out = loom_value_facts_unknown();
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
  if (lhs_lo >= 0 && rhs_lo >= 0) {
    *out = loom_value_facts_make(
        0, loom_value_facts_non_negative_bitwise_upper_bound(lhs_hi, rhs_hi),
        1);
    loom_value_facts_propagate_bitwise_flags(&lhs_facts, &rhs_facts, out);
    return;
  }
  *out = loom_value_facts_unknown();
  loom_value_facts_propagate_bitwise_flags(&lhs_facts, &rhs_facts, out);
}

//===----------------------------------------------------------------------===//
// Transfer functions: min / max
//===----------------------------------------------------------------------===//

void loom_value_facts_minsi(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  *out = loom_value_facts_make(
      loom_min_i64(lhs_facts.range_lo, rhs_facts.range_lo),
      loom_min_i64(lhs_facts.range_hi, rhs_facts.range_hi),
      loom_gcd_i64(lhs_facts.known_divisor, rhs_facts.known_divisor));
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
}

void loom_value_facts_maxsi(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  *out = loom_value_facts_make(
      loom_max_i64(lhs_facts.range_lo, rhs_facts.range_lo),
      loom_max_i64(lhs_facts.range_hi, rhs_facts.range_hi),
      loom_gcd_i64(lhs_facts.known_divisor, rhs_facts.known_divisor));
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
}

void loom_value_facts_minui(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  // For non-negative ranges, unsigned min == signed min.
  if (lhs_facts.range_lo >= 0 && rhs_facts.range_lo >= 0) {
    loom_value_facts_minsi(&lhs_facts, &rhs_facts, out);
    return;
  }
  *out = loom_value_facts_unknown();
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
}

void loom_value_facts_maxui(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  if (lhs_facts.range_lo >= 0 && rhs_facts.range_lo >= 0) {
    loom_value_facts_maxsi(&lhs_facts, &rhs_facts, out);
    return;
  }
  *out = loom_value_facts_unknown();
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
}

//===----------------------------------------------------------------------===//
// Transfer functions: ternary
//===----------------------------------------------------------------------===//

void loom_value_facts_fmai(const loom_value_facts_t* a,
                           const loom_value_facts_t* b,
                           const loom_value_facts_t* c,
                           loom_value_facts_t* out) {
  // fmai(a, b, c) computes a*b + c.
  // Read all inputs first (output may alias inputs).
  const loom_value_facts_t a_facts = *a;
  const loom_value_facts_t b_facts = *b;
  const loom_value_facts_t c_facts = *c;
  int64_t a_lo = a_facts.range_lo, a_hi = a_facts.range_hi;
  int64_t b_lo = b_facts.range_lo, b_hi = b_facts.range_hi;
  int64_t c_lo = c_facts.range_lo, c_hi = c_facts.range_hi;
  int64_t a_div = a_facts.known_divisor;
  int64_t b_div = b_facts.known_divisor;
  int64_t c_div = c_facts.known_divisor;

  // Compute product divisibility: a_div * b_div with overflow check.
  int64_t product_divisor = 0;
  if (!loom_checked_mul_i64(a_div, b_div, &product_divisor)) {
    product_divisor = loom_gcd_i64(a_div, b_div);
  }

  // Final divisibility: GCD of product_divisor and c_div.
  int64_t divisor = loom_gcd_i64(product_divisor, c_div);

  // Compute four corners of a*b.
  int64_t products[4];
  if (!loom_checked_mul_i64(a_lo, b_lo, &products[0]) ||
      !loom_checked_mul_i64(a_lo, b_hi, &products[1]) ||
      !loom_checked_mul_i64(a_hi, b_lo, &products[2]) ||
      !loom_checked_mul_i64(a_hi, b_hi, &products[3])) {
    // Product range overflowed but divisor is still valid.
    *out = loom_value_facts_make(INT64_MIN, INT64_MAX, divisor);
    loom_value_facts_propagate_ternary_distribution(a_facts, b_facts, c_facts,
                                                    out);
    return;
  }

  // Find min/max of products.
  int64_t prod_lo = products[0], prod_hi = products[0];
  for (int i = 1; i < 4; ++i) {
    if (products[i] < prod_lo) prod_lo = products[i];
    if (products[i] > prod_hi) prod_hi = products[i];
  }

  // Add c: result range is [prod_lo + c_lo, prod_hi + c_hi].
  int64_t lo, hi;
  if (!loom_checked_add_i64(prod_lo, c_lo, &lo) ||
      !loom_checked_add_i64(prod_hi, c_hi, &hi)) {
    *out = loom_value_facts_make(INT64_MIN, INT64_MAX, divisor);
    loom_value_facts_propagate_ternary_distribution(a_facts, b_facts, c_facts,
                                                    out);
    return;
  }

  *out = loom_value_facts_make(lo, hi, divisor);
  loom_value_facts_propagate_ternary_distribution(a_facts, b_facts, c_facts,
                                                  out);
}

//===----------------------------------------------------------------------===//
// Transfer functions: unary
//===----------------------------------------------------------------------===//

void loom_value_facts_negi(const loom_value_facts_t* input,
                           loom_value_facts_t* out) {
  const loom_value_facts_t input_facts = *input;
  int64_t in_lo = input_facts.range_lo, in_hi = input_facts.range_hi;
  int64_t in_divisor = input_facts.known_divisor;

  // Negation swaps and negates bounds: new_lo = -old_hi, new_hi = -old_lo.
  int64_t lo, hi;
  if (!loom_checked_sub_i64(0, in_hi, &lo) ||
      !loom_checked_sub_i64(0, in_lo, &hi)) {
    // Range overflowed but negation preserves divisibility.
    *out = loom_value_facts_make(INT64_MIN, INT64_MAX, in_divisor);
    loom_value_facts_propagate_unary_distribution(input_facts, out);
    return;
  }
  *out = loom_value_facts_make(lo, hi, in_divisor);
  loom_value_facts_propagate_unary_distribution(input_facts, out);
}

void loom_value_facts_absi(const loom_value_facts_t* input,
                           loom_value_facts_t* out) {
  const loom_value_facts_t input_facts = *input;
  int64_t in_lo = input_facts.range_lo, in_hi = input_facts.range_hi;
  int64_t in_divisor = input_facts.known_divisor;

  // Already non-negative: abs is identity.
  if (in_lo >= 0) {
    *out = loom_value_facts_make(in_lo, in_hi, in_divisor);
    loom_value_facts_propagate_unary_distribution(input_facts, out);
    return;
  }
  // Entirely non-positive: negate the range.
  if (in_hi <= 0) {
    int64_t lo, hi;
    if (!loom_checked_sub_i64(0, in_hi, &lo) ||
        !loom_checked_sub_i64(0, in_lo, &hi)) {
      // Overflow: abs preserves divisibility, result is non-negative.
      *out = loom_value_facts_make(0, INT64_MAX, in_divisor);
      loom_value_facts_propagate_unary_distribution(input_facts, out);
      return;
    }
    *out = loom_value_facts_make(lo, hi, in_divisor);
    loom_value_facts_propagate_unary_distribution(input_facts, out);
    return;
  }
  // Range spans zero: result is [0, max(|lo|, hi)].
  int64_t neg_lo;
  if (!loom_checked_sub_i64(0, in_lo, &neg_lo)) {
    // Overflow on negating lo: result is non-negative with known divisor.
    *out = loom_value_facts_make(0, INT64_MAX, in_divisor);
    loom_value_facts_propagate_unary_distribution(input_facts, out);
    return;
  }
  *out = loom_value_facts_make(0, loom_max_i64(neg_lo, in_hi), in_divisor);
  loom_value_facts_propagate_unary_distribution(input_facts, out);
}

//===----------------------------------------------------------------------===//
// Shaped type helpers
//===----------------------------------------------------------------------===//

static bool loom_value_facts_type_dim(loom_type_t type, uint8_t index,
                                      uint64_t* out_dim) {
  if (index >= loom_type_rank(type)) return false;
  if (loom_type_has_inline_dims(type)) {
    if (index >= IREE_ARRAYSIZE(type.dims)) return false;
    *out_dim = type.dims[index];
    return true;
  }
  const loom_overflow_dim_t* overflow_dims =
      (const loom_overflow_dim_t*)(uintptr_t)type.dims[0];
  if (!overflow_dims) return false;
  *out_dim = overflow_dims[index];
  return true;
}

loom_value_facts_t loom_value_facts_non_negative_extent(
    loom_value_facts_t facts) {
  if (loom_value_facts_is_float(facts) || facts.range_hi < 0) {
    loom_value_facts_t result = loom_value_facts_make(0, INT64_MAX, 1);
    result.flags |=
        facts.flags & (LOOM_VALUE_FACT_UNIFORM | LOOM_VALUE_FACT_LANE_VARYING |
                       LOOM_VALUE_FACT_LANE_PREDICATE);
    return result;
  }
  int64_t lo = loom_max_i64(facts.range_lo, 0);
  loom_value_facts_t result =
      loom_value_facts_make(lo, facts.range_hi, facts.known_divisor);
  result.flags |=
      facts.flags & (LOOM_VALUE_FACT_UNIFORM | LOOM_VALUE_FACT_LANE_VARYING |
                     LOOM_VALUE_FACT_LANE_PREDICATE);
  return result;
}

static void loom_value_facts_clamp_non_negative(loom_value_facts_t* facts) {
  *facts = loom_value_facts_non_negative_extent(*facts);
}

static loom_value_facts_t loom_value_facts_dynamic_extent(
    loom_value_id_t value_id, const loom_value_facts_t* value_facts,
    iree_host_size_t value_fact_count) {
  if (!value_facts || value_id >= value_fact_count ||
      value_facts[value_id].known_divisor == 0) {
    return loom_value_facts_make(0, INT64_MAX, 1);
  }
  return loom_value_facts_non_negative_extent(value_facts[value_id]);
}

void loom_value_facts_element_count(loom_type_t type,
                                    const loom_value_facts_t* value_facts,
                                    iree_host_size_t value_fact_count,
                                    loom_value_facts_t* out_count) {
  *out_count = loom_value_facts_unknown();
  if (!loom_type_is_shaped(type)) return;

  loom_value_facts_t accumulator = loom_value_facts_exact_i64(1);
  uint8_t rank = loom_type_rank(type);
  for (uint8_t i = 0; i < rank; ++i) {
    uint64_t packed_dim = 0;
    if (!loom_value_facts_type_dim(type, i, &packed_dim)) {
      *out_count = loom_value_facts_make(0, INT64_MAX, 1);
      return;
    }
    loom_value_facts_t dim_facts =
        loom_dim_is_dynamic(packed_dim)
            ? loom_value_facts_dynamic_extent(loom_dim_value_id(packed_dim),
                                              value_facts, value_fact_count)
            : loom_value_facts_exact_i64(loom_dim_static_size(packed_dim));
    loom_value_facts_muli(&accumulator, &dim_facts, &accumulator);
    loom_value_facts_clamp_non_negative(&accumulator);
  }
  *out_count = accumulator;
}

int64_t loom_value_facts_element_count_divisor(
    loom_type_t type, const loom_value_facts_t* value_facts,
    iree_host_size_t value_fact_count) {
  loom_value_facts_t element_count = {0};
  loom_value_facts_element_count(type, value_facts, value_fact_count,
                                 &element_count);
  return element_count.known_divisor;
}

static bool loom_value_facts_shaped_dims_equal(loom_type_t lhs_type,
                                               loom_type_t rhs_type) {
  if (!loom_type_is_shaped(lhs_type) || !loom_type_is_shaped(rhs_type)) {
    return false;
  }
  uint8_t rank = loom_type_rank(lhs_type);
  if (rank != loom_type_rank(rhs_type)) return false;
  for (uint8_t i = 0; i < rank; ++i) {
    uint64_t lhs_dim = 0;
    uint64_t rhs_dim = 0;
    if (!loom_value_facts_type_dim(lhs_type, i, &lhs_dim) ||
        !loom_value_facts_type_dim(rhs_type, i, &rhs_dim)) {
      return false;
    }
    if (lhs_dim != rhs_dim) return false;
  }
  return true;
}

bool loom_value_facts_element_counts_equal(
    loom_type_t lhs_type, const loom_value_facts_t* lhs_value_facts,
    iree_host_size_t lhs_value_fact_count, loom_type_t rhs_type,
    const loom_value_facts_t* rhs_value_facts,
    iree_host_size_t rhs_value_fact_count) {
  if (loom_value_facts_shaped_dims_equal(lhs_type, rhs_type)) return true;

  loom_value_facts_t lhs_count = {0};
  loom_value_facts_element_count(lhs_type, lhs_value_facts,
                                 lhs_value_fact_count, &lhs_count);
  if (!loom_value_facts_is_exact(lhs_count) ||
      loom_value_facts_is_float(lhs_count)) {
    return false;
  }
  loom_value_facts_t rhs_count = {0};
  loom_value_facts_element_count(rhs_type, rhs_value_facts,
                                 rhs_value_fact_count, &rhs_count);
  return loom_value_facts_is_exact(rhs_count) &&
         !loom_value_facts_is_float(rhs_count) &&
         lhs_count.range_lo == rhs_count.range_lo;
}
