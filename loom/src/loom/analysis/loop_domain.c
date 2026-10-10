// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/loop_domain.h"

#include "loom/ir/facts.h"

static bool loom_loop_domain_has_positive_step(loom_value_facts_t lower_bound,
                                               loom_value_facts_t upper_bound,
                                               loom_value_facts_t step) {
  return !loom_value_facts_is_float(lower_bound) &&
         !loom_value_facts_is_float(upper_bound) &&
         !loom_value_facts_is_float(step) && loom_value_facts_is_positive(step);
}

bool loom_loop_domain_proven_empty(loom_value_facts_t lower_bound,
                                   loom_value_facts_t upper_bound,
                                   loom_value_facts_t step) {
  return loom_loop_domain_has_positive_step(lower_bound, upper_bound, step) &&
         lower_bound.range_lo >= upper_bound.range_hi;
}

bool loom_loop_domain_proven_nonempty(loom_value_facts_t lower_bound,
                                      loom_value_facts_t upper_bound,
                                      loom_value_facts_t step) {
  return loom_loop_domain_has_positive_step(lower_bound, upper_bound, step) &&
         lower_bound.range_hi < upper_bound.range_lo;
}

bool loom_loop_domain_trip_count(loom_loop_bound_flags_t bound_flags,
                                 uint8_t bitwidth, uint64_t initial_value,
                                 uint64_t bound, uint64_t step,
                                 uint64_t* out_trip_count) {
  *out_trip_count = 0;
  const bool is_signed = iree_any_bit_set(bound_flags, LOOM_LOOP_BOUND_SIGNED);
  const bool is_inclusive =
      iree_any_bit_set(bound_flags, LOOM_LOOP_BOUND_INCLUSIVE);
  const bool is_descending =
      iree_any_bit_set(bound_flags, LOOM_LOOP_BOUND_DESCENDING);

  // Flipping the sign bit maps signed order to unsigned order. Complementing
  // that rank reverses descending order and negates its modular increment, so
  // every comparison direction shares the increasing-rank proof.
  const uint64_t mask = UINT64_MAX >> (64 - bitwidth);
  const uint64_t order_mask = (is_signed ? UINT64_C(1) << (bitwidth - 1) : 0) ^
                              (is_descending ? mask : 0);
  const uint64_t initial = (initial_value & mask) ^ order_mask;
  const uint64_t limit = (bound & mask) ^ order_mask;
  const uint64_t increment = (is_descending ? UINT64_C(0) - step : step) & mask;
  if (initial > limit || (initial == limit && !is_inclusive)) {
    return true;
  }
  if (increment == 0 || (is_inclusive && limit == mask)) {
    return false;
  }

  const uint64_t distance = limit - initial + (is_inclusive ? 1 : 0);
  const uint64_t trip_count = (distance - 1) / increment + 1;
  if (trip_count > (mask - initial) / increment) {
    return false;
  }
  *out_trip_count = trip_count;
  return true;
}

static int64_t loom_loop_domain_value_from_rank(uint64_t rank) {
  const uint64_t sign_bit = UINT64_C(1) << 63;
  return rank >= sign_bit ? (int64_t)(rank - sign_bit)
                          : INT64_MIN + (int64_t)rank;
}

static int64_t loom_loop_domain_recurrence_divisor(
    loom_value_facts_t initial_values, uint64_t magnitude) {
  const uint64_t initial_divisor =
      loom_value_facts_is_exact(initial_values)
          ? iree_math_magnitude_i64(initial_values.range_lo)
          : (uint64_t)initial_values.known_divisor;
  const uint64_t divisor = iree_math_gcd_u64(initial_divisor, magnitude);
  // The fact representation cannot store the magnitude of INT64_MIN as its
  // positive signed divisor, matching the exact-integer constructor.
  return divisor > (uint64_t)INT64_MAX ? 1 : (int64_t)divisor;
}

bool loom_loop_domain_additive_recurrence_facts(
    loom_value_facts_t initial_values, int64_t step, uint64_t trip_count,
    loom_loop_recurrence_facts_t* out_facts) {
  *out_facts = (loom_loop_recurrence_facts_t){
      .values = loom_value_facts_unknown(),
      .body_values = loom_value_facts_unknown(),
      .exit_value = loom_value_facts_unknown(),
      .trip_count = trip_count,
      .trip_count_kind = LOOM_LOOP_TRIP_COUNT_EXACT,
  };
  initial_values =
      loom_value_facts_make(initial_values.range_lo, initial_values.range_hi,
                            initial_values.known_divisor);
  if (trip_count == 0 || step == 0) {
    out_facts->values = initial_values;
    out_facts->exit_value = initial_values;
    if (trip_count != 0) {
      out_facts->body_values = initial_values;
    }
    return true;
  }

  // Signed-order ranks cover the full i64 span without signed subtraction or
  // a signed trip-count product. Checking the advancing endpoint proves that
  // every initial value and intermediate state remain representable.
  const uint64_t sign_bit = UINT64_C(1) << 63;
  const uint64_t initial_lo_rank = (uint64_t)initial_values.range_lo ^ sign_bit;
  const uint64_t initial_hi_rank = (uint64_t)initial_values.range_hi ^ sign_bit;
  const uint64_t magnitude = iree_math_magnitude_i64(step);
  const uint64_t available =
      step > 0 ? UINT64_MAX - initial_hi_rank : initial_lo_rank;
  if (trip_count > available / magnitude) {
    return false;
  }
  const uint64_t distance = trip_count * magnitude;
  const uint64_t exit_lo_rank =
      step > 0 ? initial_lo_rank + distance : initial_lo_rank - distance;
  const uint64_t exit_hi_rank =
      step > 0 ? initial_hi_rank + distance : initial_hi_rank - distance;
  const int64_t exit_lo = loom_loop_domain_value_from_rank(exit_lo_rank);
  const int64_t exit_hi = loom_loop_domain_value_from_rank(exit_hi_rank);
  const int64_t divisor =
      loom_loop_domain_recurrence_divisor(initial_values, magnitude);
  out_facts->values = loom_value_facts_make(
      step > 0 ? initial_values.range_lo : exit_lo,
      step > 0 ? exit_hi : initial_values.range_hi, divisor);
  out_facts->body_values = loom_value_facts_make(
      step > 0 ? initial_values.range_lo
               : loom_loop_domain_value_from_rank(exit_lo_rank + magnitude),
      step > 0 ? loom_loop_domain_value_from_rank(exit_hi_rank - magnitude)
               : initial_values.range_hi,
      trip_count == 1 ? initial_values.known_divisor : divisor);
  out_facts->exit_value = loom_value_facts_make(
      exit_lo, exit_hi,
      (int64_t)iree_math_gcd_u64((uint64_t)initial_values.known_divisor,
                                 distance));
  return true;
}

static loom_loop_recurrence_facts_t loom_loop_domain_exact_recurrence_facts(
    loom_loop_bound_flags_t bound_flags, uint8_t bitwidth,
    int64_t initial_value, int64_t bound, int64_t step) {
  loom_loop_recurrence_facts_t result = {
      .values = loom_value_facts_unknown(),
      .body_values = loom_value_facts_unknown(),
      .exit_value = loom_value_facts_unknown(),
  };
  if (loom_loop_domain_trip_count(bound_flags, bitwidth,
                                  (uint64_t)initial_value, (uint64_t)bound,
                                  (uint64_t)step, &result.trip_count)) {
    result.trip_count_kind = LOOM_LOOP_TRIP_COUNT_EXACT;
  }
  const int64_t maximum = INT64_MAX >> (64 - bitwidth);
  const int64_t minimum = -maximum - 1;
  if (result.trip_count_kind == LOOM_LOOP_TRIP_COUNT_UNKNOWN ||
      initial_value < minimum || initial_value > maximum) {
    return result;
  }
  if (result.trip_count == 0) {
    result.values = loom_value_facts_exact_i64(initial_value);
    result.exit_value = result.values;
    return result;
  }
  const bool is_descending =
      iree_any_bit_set(bound_flags, LOOM_LOOP_BOUND_DESCENDING);
  if (is_descending ? step >= 0 || step < minimum
                    : step <= 0 || step > maximum) {
    return result;
  }
  if (loom_loop_domain_additive_recurrence_facts(
          loom_value_facts_exact_i64(initial_value), step, result.trip_count,
          &result) &&
      loom_value_facts_fit_signed_bit_count(result.values, bitwidth)) {
    result.exit_value = loom_value_facts_exact_i64(result.exit_value.range_lo);
  } else {
    result.values = loom_value_facts_unknown();
    result.body_values = loom_value_facts_unknown();
    result.exit_value = loom_value_facts_unknown();
  }
  return result;
}

// Each accepted source interval maps monotonically into comparison order.
// Unsigned negative and nonnegative intervals are individually contiguous, but
// their union crosses the raw carrier's wrap and cannot use endpoint ranks.
static bool loom_loop_domain_ordered_interval(
    loom_loop_bound_flags_t bound_flags, uint8_t bitwidth,
    loom_value_facts_t facts, uint64_t* out_lower, uint64_t* out_upper) {
  const bool is_signed = iree_any_bit_set(bound_flags, LOOM_LOOP_BOUND_SIGNED);
  if (is_signed || facts.range_lo < 0) {
    if (!loom_value_facts_fit_signed_bit_count(facts, bitwidth) ||
        (!is_signed && facts.range_hi >= 0)) {
      return false;
    }
  } else if (!loom_value_facts_fit_unsigned_bit_count(facts, bitwidth)) {
    return false;
  }
  const uint64_t mask = UINT64_MAX >> (64 - bitwidth);
  const bool is_descending =
      iree_any_bit_set(bound_flags, LOOM_LOOP_BOUND_DESCENDING);
  const uint64_t order_mask = (is_signed ? UINT64_C(1) << (bitwidth - 1) : 0) ^
                              (is_descending ? mask : 0);
  *out_lower =
      ((uint64_t)(is_descending ? facts.range_hi : facts.range_lo) & mask) ^
      order_mask;
  *out_upper =
      ((uint64_t)(is_descending ? facts.range_lo : facts.range_hi) & mask) ^
      order_mask;
  return true;
}

static int64_t loom_loop_domain_source_from_ordered(
    loom_loop_bound_flags_t bound_flags, uint8_t bitwidth, uint64_t ordered) {
  const uint64_t sign_bit = UINT64_C(1) << (bitwidth - 1);
  const uint64_t mask = UINT64_MAX >> (64 - bitwidth);
  const uint64_t bits =
      ordered ^
      (iree_any_bit_set(bound_flags, LOOM_LOOP_BOUND_SIGNED) ? sign_bit : 0) ^
      (iree_any_bit_set(bound_flags, LOOM_LOOP_BOUND_DESCENDING) ? mask : 0);
  return bits < sign_bit ? (int64_t)bits : -1 - (int64_t)(mask - bits);
}

loom_loop_recurrence_facts_t loom_loop_domain_recurrence_facts(
    loom_loop_bound_flags_t bound_flags, uint8_t bitwidth,
    loom_value_facts_t initial_values, loom_value_facts_t bounds,
    loom_value_facts_t step) {
  loom_loop_recurrence_facts_t result = {
      .values = loom_value_facts_unknown(),
      .body_values = loom_value_facts_unknown(),
      .exit_value = loom_value_facts_unknown(),
  };
  if (loom_value_facts_is_float(initial_values) ||
      loom_value_facts_is_float(bounds) || loom_value_facts_is_float(step)) {
    return result;
  }
  int64_t increment = 0;
  const bool exact_step = loom_value_facts_as_exact_i64(step, &increment);
  if (loom_value_facts_is_exact(initial_values) &&
      loom_value_facts_is_exact(bounds)) {
    // An unknown step cannot prove a nonempty loop, but a false initial guard
    // still proves zero trips through the same concrete carrier semantics.
    return loom_loop_domain_exact_recurrence_facts(
        bound_flags, bitwidth, initial_values.range_lo, bounds.range_lo,
        exact_step ? increment : 0);
  }

  uint64_t initial_lower = 0, initial_upper = 0;
  uint64_t bound_lower = 0, bound_upper = 0;
  if (!loom_loop_domain_ordered_interval(bound_flags, bitwidth, initial_values,
                                         &initial_lower, &initial_upper) ||
      !loom_loop_domain_ordered_interval(bound_flags, bitwidth, bounds,
                                         &bound_lower, &bound_upper)) {
    return result;
  }
  const bool is_inclusive =
      iree_any_bit_set(bound_flags, LOOM_LOOP_BOUND_INCLUSIVE);
  const bool is_descending =
      iree_any_bit_set(bound_flags, LOOM_LOOP_BOUND_DESCENDING);
  if (initial_lower > bound_upper ||
      (initial_lower == bound_upper && !is_inclusive)) {
    result.trip_count_kind = LOOM_LOOP_TRIP_COUNT_EXACT;
    if (loom_value_facts_fit_signed_bit_count(initial_values, bitwidth)) {
      result.values = loom_value_facts_make(initial_values.range_lo,
                                            initial_values.range_hi,
                                            initial_values.known_divisor);
      result.exit_value = result.values;
    }
    return result;
  }
  const uint64_t mask = UINT64_MAX >> (64 - bitwidth);
  const uint64_t stride = iree_math_magnitude_i64(increment);
  if (!exact_step || (is_descending ? increment >= 0 : increment <= 0) ||
      stride > mask || (is_inclusive && bound_upper == mask)) {
    return result;
  }

  // In normalized comparison order, completing the greatest bound completes
  // every smaller bound. Across active initial values, terminal overshoot
  // advances by one modulo the step. Include every residue reached by that
  // interval: its least initial rank alone can miss an overflowing final step.
  const uint64_t threshold = bound_upper + (is_inclusive ? 1 : 0);
  const uint64_t distance = threshold - initial_lower;
  const uint64_t remainder = distance % stride;
  const uint64_t first_overshoot = remainder == 0 ? 0 : stride - remainder;
  const uint64_t initial_span =
      iree_min(initial_upper, threshold - 1) - initial_lower;
  const uint64_t maximum_overshoot =
      first_overshoot + iree_min(initial_span, stride - 1 - first_overshoot);
  if (maximum_overshoot > mask - threshold) {
    return result;
  }
  result.trip_count = (distance - 1) / stride + 1;
  result.trip_count_kind = LOOM_LOOP_TRIP_COUNT_BOUNDED;

  const uint64_t terminal_upper = threshold + maximum_overshoot;
  const uint64_t header_upper = iree_max(initial_upper, terminal_upper);
  const uint64_t sign_bit = UINT64_C(1) << (bitwidth - 1);
  if (!loom_value_facts_fit_signed_bit_count(initial_values, bitwidth) ||
      !loom_value_facts_fit_signed_bit_count(step, bitwidth) ||
      (!iree_any_bit_set(bound_flags, LOOM_LOOP_BOUND_SIGNED) &&
       (initial_lower < sign_bit) != (header_upper < sign_bit))) {
    return result;
  }
  const int64_t divisor =
      loom_loop_domain_recurrence_divisor(initial_values, stride);
  const int64_t header_terminal =
      loom_loop_domain_source_from_ordered(bound_flags, bitwidth, header_upper);
  const int64_t body_terminal = loom_loop_domain_source_from_ordered(
      bound_flags, bitwidth, terminal_upper - stride);
  result.values = loom_value_facts_make(
      is_descending ? header_terminal : initial_values.range_lo,
      is_descending ? initial_values.range_hi : header_terminal, divisor);
  result.body_values = loom_value_facts_make(
      is_descending ? body_terminal : initial_values.range_lo,
      is_descending ? initial_values.range_hi : body_terminal,
      result.trip_count == 1 ? initial_values.known_divisor : divisor);
  result.exit_value = result.values;
  return result;
}
