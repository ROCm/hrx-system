// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Dialect-independent loop-domain proofs. Callers supply bound facts or
// concrete integer recurrences; this component owns their numeric semantics.

#ifndef LOOM_ANALYSIS_LOOP_DOMAIN_H_
#define LOOM_ANALYSIS_LOOP_DOMAIN_H_

#include "iree/base/api.h"
#include "loom/ir/facts.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Half-open counted range [lower_bound, upper_bound) with a positive step.
// SSA value IDs keep the domain independent of the operation defining it.
typedef struct loom_loop_domain_t {
  // Inclusive lower bound of the counted loop domain.
  loom_value_id_t lower_bound;
  // Exclusive upper bound of the counted loop domain.
  loom_value_id_t upper_bound;
  // Positive step between consecutive induction values.
  loom_value_id_t step;
} loom_loop_domain_t;

// Returns true when every value admitted by the domain facts produces zero
// iterations. The proof requires a positive integer step and lower_bound >=
// upper_bound for the complete fact ranges.
bool loom_loop_domain_proven_empty(loom_value_facts_t lower_bound,
                                   loom_value_facts_t upper_bound,
                                   loom_value_facts_t step);

// Returns true when every value admitted by the domain facts produces at least
// one iteration. The proof requires a positive integer step and lower_bound <
// upper_bound for the complete fact ranges.
bool loom_loop_domain_proven_nonempty(loom_value_facts_t lower_bound,
                                      loom_value_facts_t upper_bound,
                                      loom_value_facts_t step);

// Upper-bound comparison semantics for a header-tested integer recurrence.
enum loom_loop_bound_flag_bits_e {
  // Unsigned exclusive comparison: induction value < upper bound.
  LOOM_LOOP_BOUND_NONE = 0,
  // Use signed two's-complement order for the induction value and bound.
  LOOM_LOOP_BOUND_SIGNED = 1u << 0,
  // Continue while the induction value <= bound instead of < bound.
  LOOM_LOOP_BOUND_INCLUSIVE = 1u << 1,
};
typedef uint8_t loom_loop_bound_flags_t;

// Counts body executions of a header-tested loop whose backedge adds |step| to
// the induction value. |bound_flags| selects signed/unsigned and exclusive/
// inclusive comparison with |upper_bound|. The initial value, bound, and step
// are raw carrier bits; only their low |bitwidth| bits participate. |bitwidth|
// is the verified carrier width in [1, 64].
//
// A false initial guard proves zero trips independently of the step. Nonempty
// loops require an increasing recurrence that reaches the exit without wrapping
// in the comparison's ordered carrier domain, including the terminal increment.
// This differs from the mathematical cardinality of a counted range: a finite
// range may still require an overflowing terminal increment in a lowered loop.
// Returns false when this proof cannot establish an exact count; it does not
// imply that the loop is infinite. |out_trip_count| is zero on failure.
bool loom_loop_domain_trip_count(loom_loop_bound_flags_t bound_flags,
                                 uint8_t bitwidth, uint64_t initial_value,
                                 uint64_t upper_bound, uint64_t step,
                                 uint64_t* out_trip_count);

// Facts for the controlling value of a finite-width, header-tested recurrence.
typedef struct loom_loop_recurrence_facts_t {
  // Inclusive range of all header observations, including the terminal value.
  // Unknown when the recurrence crosses the signed source representation.
  loom_value_facts_t values;
  // Exact body execution count when trip_count_known is true; zero otherwise.
  uint64_t trip_count;
  // True when the recurrence reaches its exit without wrapping in guard order.
  bool trip_count_known;
} loom_loop_recurrence_facts_t;

// Proves the same recurrence as trip_count, retaining a source-integer range
// when its initial value and positive increments remain representable in the
// signed carrier. An exact count alone does not imply such a range: unsigned
// order can cross the sign bit, and a modular increment can be negative in the
// source representation. A zero-trip range contains only the initial value.
loom_loop_recurrence_facts_t loom_loop_domain_recurrence_facts(
    loom_loop_bound_flags_t bound_flags, uint8_t bitwidth,
    int64_t initial_value, int64_t upper_bound, int64_t step);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_LOOP_DOMAIN_H_
