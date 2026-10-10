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

// Bound comparison semantics for a header-tested integer recurrence.
enum loom_loop_bound_flag_bits_e {
  // Unsigned ascending exclusive comparison: induction value < bound.
  LOOM_LOOP_BOUND_NONE = 0,
  // Use signed two's-complement order for the induction value and bound.
  LOOM_LOOP_BOUND_SIGNED = 1u << 0,
  // Include equality in the continuing comparison (<= or >=).
  LOOM_LOOP_BOUND_INCLUSIVE = 1u << 1,
  // Continue while the induction value > bound instead of < bound.
  LOOM_LOOP_BOUND_DESCENDING = 1u << 2,
};
typedef uint8_t loom_loop_bound_flags_t;

// Counts body executions of a header-tested loop whose backedge adds |step| to
// the induction value. |bound_flags| selects signed/unsigned, ascending/
// descending, and exclusive/inclusive comparison with |bound|. The initial
// value, bound, and additive step are raw carrier bits; only their low
// |bitwidth| bits participate. |bitwidth| is the verified carrier width in
// [1, 64].
//
// A false initial guard proves zero trips independently of the step. Nonempty
// loops require a recurrence advancing toward the bound without wrapping in
// the comparison's ordered carrier domain, including the terminal increment.
// This differs from the mathematical cardinality of a counted range: a finite
// range may still require an overflowing terminal increment in a lowered loop.
// Returns false when this proof cannot establish an exact count; it does not
// imply that the loop is infinite. |out_trip_count| is zero on failure.
bool loom_loop_domain_trip_count(loom_loop_bound_flags_t bound_flags,
                                 uint8_t bitwidth, uint64_t initial_value,
                                 uint64_t bound, uint64_t step,
                                 uint64_t* out_trip_count);

// Precision of a proven body execution count.
enum loom_loop_trip_count_kind_e {
  // No finite execution count is proven.
  LOOM_LOOP_TRIP_COUNT_UNKNOWN = 0,
  // The count is an inclusive upper bound, without a positive lower bound.
  LOOM_LOOP_TRIP_COUNT_BOUNDED,
  // Every represented execution has exactly the supplied count.
  LOOM_LOOP_TRIP_COUNT_EXACT,
};
typedef uint8_t loom_loop_trip_count_kind_t;

// Source-integer facts for an additive recurrence over a finite loop domain.
typedef struct loom_loop_recurrence_facts_t {
  // Inclusive range of all header observations, including the exit state.
  // Unknown when the source-integer recurrence is unproven.
  loom_value_facts_t values;
  // Inclusive range observed on entry to the body, excluding the terminal
  // value. Unknown for zero trips or an unproven source-integer recurrence.
  loom_value_facts_t body_values;
  // Range of exit states, including zero-trip exits. A bounded count retains
  // the complete header envelope, not the state after its maximum count.
  // Unknown when the source-integer recurrence is unproven.
  loom_value_facts_t exit_value;
  // Maximum body execution count for BOUNDED, exact count for EXACT, and zero
  // for UNKNOWN. A proven zero maximum is represented as EXACT.
  uint64_t trip_count;
  // Count precision, independent of whether source-integer ranges are proven.
  loom_loop_trip_count_kind_t trip_count_kind;
} loom_loop_recurrence_facts_t;

// Evaluates initial_values + iteration * step for the supplied exact trip
// count. |initial_values| contains integer range and divisibility facts. The
// header includes iterations [0, trip_count], the body [0, trip_count), and
// the exit only trip_count. A zero-trip body has unknown facts.
//
// Returns true when every represented initial value and every resulting state
// fit the mathematical signed i64 source domain. On failure all three ranges
// are unknown; the supplied trip count remains known. The result retains only
// numeric range and divisibility facts, not context-local extensions or
// distribution. This proof neither establishes a loop's trip count nor selects
// overflow semantics. Target-carrier representability remains with the caller.
bool loom_loop_domain_additive_recurrence_facts(
    loom_value_facts_t initial_values, int64_t step, uint64_t trip_count,
    loom_loop_recurrence_facts_t* out_facts);

// Proves execution counts and source-integer ranges for a header-tested loop.
// Exact initial and bound values use trip_count's raw-carrier semantics. Ranged
// operands must fit the carrier without truncation and describe contiguous
// comparison-order intervals. Nonempty ranged loops require an exact step
// matching the selected direction and prove every possible terminal increment
// remains in the carrier. A descending step is a negative additive delta,
// including INT64_MIN when the carrier can represent it.
// An unsigned operand range crossing the sign discontinuity is unproven.
//
// Count knowledge does not imply source-integer ranges: unsigned order can
// cross the sign bit, and a modular increment can have the opposite source
// sign. Source ranges require representable signed initial values and deltas
// matching the selected direction. A bounded count's exit range is its whole
// header envelope. A zero-trip header and exit contain only the initial values;
// no body range is established, and the proof does not require a known step.
loom_loop_recurrence_facts_t loom_loop_domain_recurrence_facts(
    loom_loop_bound_flags_t bound_flags, uint8_t bitwidth,
    loom_value_facts_t initial_values, loom_value_facts_t bounds,
    loom_value_facts_t step);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_LOOP_DOMAIN_H_
