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
                                 uint64_t upper_bound, uint64_t step,
                                 uint64_t* out_trip_count) {
  *out_trip_count = 0;
  const bool is_signed = iree_any_bit_set(bound_flags, LOOM_LOOP_BOUND_SIGNED);
  const bool is_inclusive =
      iree_any_bit_set(bound_flags, LOOM_LOOP_BOUND_INCLUSIVE);

  // Flipping the sign bit maps signed order to unsigned order. Modular addition
  // is unchanged by this rotation, so both comparison domains share one proof.
  const uint64_t mask = UINT64_MAX >> (64 - bitwidth);
  const uint64_t sign_bit = is_signed ? UINT64_C(1) << (bitwidth - 1) : 0;
  const uint64_t initial = (initial_value & mask) ^ sign_bit;
  const uint64_t upper = (upper_bound & mask) ^ sign_bit;
  const uint64_t increment = step & mask;
  if (initial > upper || (initial == upper && !is_inclusive)) {
    return true;
  }
  if (increment == 0 || (is_inclusive && upper == mask)) {
    return false;
  }

  const uint64_t distance = upper - initial + (is_inclusive ? 1 : 0);
  const uint64_t trip_count = (distance - 1) / increment + 1;
  if (trip_count > (mask - initial) / increment) {
    return false;
  }
  *out_trip_count = trip_count;
  return true;
}

loom_loop_recurrence_facts_t loom_loop_domain_recurrence_facts(
    loom_loop_bound_flags_t bound_flags, uint8_t bitwidth,
    int64_t initial_value, int64_t upper_bound, int64_t step) {
  loom_loop_recurrence_facts_t result = {
      .values = loom_value_facts_unknown(),
  };
  result.trip_count_known = loom_loop_domain_trip_count(
      bound_flags, bitwidth, (uint64_t)initial_value, (uint64_t)upper_bound,
      (uint64_t)step, &result.trip_count);
  const int64_t maximum = INT64_MAX >> (64 - bitwidth);
  const int64_t minimum = -maximum - 1;
  if (!result.trip_count_known || initial_value < minimum ||
      initial_value > maximum) {
    return result;
  }
  if (result.trip_count == 0) {
    result.values = loom_value_facts_exact_i64(initial_value);
    return result;
  }
  if (step <= 0 || step > maximum) {
    return result;
  }
  // Unsigned distance also covers a signed recurrence spanning zero without
  // overflowing host signed arithmetic. The division proves the product fits.
  const uint64_t available = (uint64_t)maximum - (uint64_t)initial_value;
  if (result.trip_count > available / (uint64_t)step) {
    return result;
  }
  const uint64_t distance = result.trip_count * (uint64_t)step;
  const uint64_t rank =
      ((uint64_t)initial_value ^ (UINT64_C(1) << 63)) + distance;
  const int64_t terminal = rank >= (UINT64_C(1) << 63)
                               ? (int64_t)(rank - (UINT64_C(1) << 63))
                               : INT64_MIN + (int64_t)rank;
  result.values = loom_value_facts_make(initial_value, terminal, 1);
  return result;
}
