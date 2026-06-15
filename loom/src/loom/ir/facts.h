// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Value facts: per-value dataflow analysis properties.
//
// A loom_value_facts_t is a 32-byte summary of what is known about an
// SSA value: signed integer range [lo, hi], largest known divisor, and
// cached predicate flags (non-negative, non-zero, power-of-two, etc.).
//
// Facts are the building blocks of loom's always-on value analysis.
// They are maintained incrementally by the rewriter during
// canonicalization and propagated through arithmetic via transfer
// functions. Constants produce exact facts (lo == hi). Assume
// predicates tighten facts. Arithmetic transfer functions compute
// output facts from input facts.
//
// Transfer functions take facts by pointer and write the result into
// an output parameter. The output MAY alias one input for in-place
// accumulation (e.g., computing element counts by chaining muli):
//
//   loom_value_facts_t accumulator = loom_value_facts_exact_i64(1);
//   for (uint8_t i = 0; i < rank; ++i) {
//     loom_value_facts_t dim_facts = facts[dim_value_id];
//     loom_value_facts_muli(&accumulator, &dim_facts, &accumulator);
//   }

#ifndef LOOM_IR_FACTS_H_
#define LOOM_IR_FACTS_H_

#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/util/math.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Flags
//===----------------------------------------------------------------------===//

// Cached predicate flags derived from range, divisor, or target-independent
// execution distribution. Range flags are always consistent with the scalar
// range/divisor fields. Distribution flags describe whether all lanes in an
// invocation group observe the same SSA value.
enum loom_value_fact_flag_bits_e {
  // The signed range lower bound is >= 0.
  LOOM_VALUE_FACT_NON_NEGATIVE = 1u << 0,
  // The range does not include zero (lo > 0 or hi < 0).
  LOOM_VALUE_FACT_NON_ZERO = 1u << 1,
  // The signed range lower bound is > 0. Implies NON_NEGATIVE and
  // NON_ZERO.
  LOOM_VALUE_FACT_POSITIVE = 1u << 2,
  // The value is known to be a power of two. Set by pow2() predicates
  // or when an exact value is a power of two. Preserved across
  // recompute_flags since it may come from predicates rather than
  // range analysis.
  LOOM_VALUE_FACT_POWER_OF_TWO = 1u << 3,
  // The range is a single point (range_lo == range_hi). The value is
  // a known compile-time constant. For integer types, range_lo is the
  // constant value. For float types, range_lo contains the IEEE 754
  // bit pattern of the double (see FLOAT flag).
  LOOM_VALUE_FACT_EXACT = 1u << 4,
  // The range is [0, 1]. Typical for i1 comparison results.
  LOOM_VALUE_FACT_BOOLEAN = 1u << 5,
  // The value has a floating-point type. When EXACT is also set,
  // range_lo/range_hi contain the IEEE 754 double bit pattern (via
  // memcpy), not an integer range. Without EXACT, float facts are
  // unknown (no float range analysis).
  LOOM_VALUE_FACT_FLOAT = 1u << 6,
  // The value is known to be identical for every active lane observing it.
  LOOM_VALUE_FACT_UNIFORM = 1u << 7,
  // The value may differ between active lanes observing it.
  LOOM_VALUE_FACT_LANE_VARYING = 1u << 8,
  // The i1 value is a lane mask/predicate rather than a scalar condition code.
  LOOM_VALUE_FACT_LANE_PREDICATE = 1u << 9,
  // The integer value encodes subgroup lane membership as one bit per lane.
  // This is independent of whether each lane observes the same mask value.
  LOOM_VALUE_FACT_SUBGROUP_LANE_MASK = 1u << 10,
};
typedef uint32_t loom_value_fact_flags_t;

#define LOOM_VALUE_FACT_DISTRIBUTION_MASK \
  (LOOM_VALUE_FACT_UNIFORM | LOOM_VALUE_FACT_LANE_VARYING)

// Context-local extension payload ID. Zero means the fact has no extension.
typedef uint32_t loom_value_fact_extension_id_t;
#define LOOM_VALUE_FACT_EXTENSION_ID_NONE ((loom_value_fact_extension_id_t)0)

// Target-independent memory space for storage roots, views, and fences.
typedef enum loom_value_fact_memory_space_e {
  // No usable memory-space fact is known.
  LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN = 0,
  // Device-visible global storage.
  LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL = 1,
  // Workgroup/shared storage.
  LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP = 2,
  // Invocation-private storage.
  LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE = 3,
  // Read-only constant storage.
  LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT = 4,
  // Host-visible storage.
  LOOM_VALUE_FACT_MEMORY_SPACE_HOST = 5,
  // Descriptor-backed storage identity.
  LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR = 6,
  // Target-generic device storage.
  LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC = 7,
} loom_value_fact_memory_space_t;

//===----------------------------------------------------------------------===//
// Struct
//===----------------------------------------------------------------------===//

// Per-value analysis facts. 32 bytes, cache-friendly for dense arrays.
typedef struct loom_value_facts_t {
  // Signed integer range [lo, hi], inclusive. Default (unknown):
  // [INT64_MIN, INT64_MAX]. For float types with EXACT flag set,
  // range_lo contains the IEEE 754 bit pattern of the double value
  // (via memcpy); range_hi == range_lo.
  int64_t range_lo;
  int64_t range_hi;

  // Largest known divisor (>= 1). GCD semantics: if the value is
  // known to be a multiple of both 16 and 24, known_divisor =
  // gcd(16, 24) = 8. For exact integer values, known_divisor = |value|
  // (or 1 if value is 0 or INT64_MIN). Default (unknown): 1.
  int64_t known_divisor;

  // Cached predicate bitflags. Derived from range/divisor but cheaper
  // to test (single bit check vs. integer comparison).
  uint32_t flags;

  // One-based ID into the current fact context extension table. Zero means no
  // extension. Extension IDs are context-local and must only be interpreted by
  // APIs that receive the context that produced the facts.
  loom_value_fact_extension_id_t extension_id;
} loom_value_facts_t;

static_assert(sizeof(loom_value_facts_t) == 32,
              "loom_value_facts_t must be 32 bytes");

//===----------------------------------------------------------------------===//
// Constructors
//===----------------------------------------------------------------------===//

// Unknown facts: the conservative default. No information about the
// value beyond its type.
static inline loom_value_facts_t loom_value_facts_unknown(void) {
  loom_value_facts_t facts = {0};
  facts.range_lo = INT64_MIN;
  facts.range_hi = INT64_MAX;
  facts.known_divisor = 1;
  return facts;
}

// Exact integer value. Computes all flags from the value.
loom_value_facts_t loom_value_facts_exact_i64(int64_t value);

// Exact float value. Stores the IEEE 754 bit pattern in range_lo.
// Float facts are exact-only (no float range analysis).
loom_value_facts_t loom_value_facts_exact_f64(double value);

// Extracts the double from an exact float fact.
static inline double loom_value_facts_as_f64(loom_value_facts_t facts) {
  double value;
  memcpy(&value, &facts.range_lo, sizeof(double));
  return value;
}

// General constructor: range with divisor. Computes flags from the
// range and divisor values.
loom_value_facts_t loom_value_facts_make(int64_t lo, int64_t hi,
                                         int64_t known_divisor);

// Returns the target-independent signed integer range represented by a scalar
// integer or address type. Returns false for floating-point scalar types.
bool loom_value_facts_scalar_type_domain(loom_scalar_type_t scalar_type,
                                         int64_t* out_lo, int64_t* out_hi);

// Clamps facts to an integer domain range. If the current facts are
// incompatible with the domain or describe a float, returns the domain as the
// conservative fact set for the value.
loom_value_facts_t loom_value_facts_clamp_domain(loom_value_facts_t facts,
                                                 int64_t lo, int64_t hi);

// Tightens facts to the dynamic extent domain. Extents are non-negative
// integer sizes; incompatible float or negative-only facts degrade to the
// conservative non-negative extent range.
loom_value_facts_t loom_value_facts_non_negative_extent(
    loom_value_facts_t facts);

// Returns the exact integer value masked to |bit_count| raw bits. This is
// useful for transfer functions whose semantics operate on two's-complement
// bit patterns instead of signed fact ranges.
bool loom_value_facts_as_exact_raw_bits(loom_value_facts_t facts,
                                        int32_t bit_count, uint64_t* out_bits);

// Returns exact facts for a zero-extended raw bit pattern. Returns false when
// the unsigned value cannot be represented in the signed i64 fact domain.
bool loom_value_facts_make_unsigned_raw_bits(uint64_t raw_bits,
                                             int32_t bit_count,
                                             loom_value_facts_t* out_facts);

// Returns exact facts for a sign-extended raw bit pattern.
loom_value_facts_t loom_value_facts_make_signed_raw_bits(uint64_t raw_bits,
                                                         int32_t bit_count);

// Returns conservative unsigned range facts for a value with |bit_count|
// meaningful bits.
loom_value_facts_t loom_value_facts_make_unsigned_bit_count_range(
    int64_t bit_count);

// Returns conservative signed range facts for a value with |bit_count|
// meaningful bits.
loom_value_facts_t loom_value_facts_make_signed_bit_count_range(
    int64_t bit_count);

// Extracts a fixed unsigned bitfield from an integer fact summary. Returns
// false only for invalid transfer parameters; unknown source facts produce the
// conservative range implied by |width|.
bool loom_value_facts_extract_unsigned_bitfield(loom_value_facts_t source,
                                                int32_t source_bit_count,
                                                int64_t offset, int64_t width,
                                                loom_value_facts_t* out_facts);

// Extracts a fixed signed bitfield from an integer fact summary. Returns false
// only for invalid transfer parameters; unknown source facts produce the
// conservative range implied by |width|.
bool loom_value_facts_extract_signed_bitfield(loom_value_facts_t source,
                                              int32_t source_bit_count,
                                              int64_t offset, int64_t width,
                                              loom_value_facts_t* out_facts);

//===----------------------------------------------------------------------===//
// Predicates
//===----------------------------------------------------------------------===//

static inline bool loom_value_facts_is_exact(loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_EXACT) != 0;
}

static inline bool loom_value_facts_is_non_negative(loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_NON_NEGATIVE) != 0;
}

static inline bool loom_value_facts_is_non_zero(loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_NON_ZERO) != 0;
}

static inline bool loom_value_facts_is_positive(loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_POSITIVE) != 0;
}

static inline bool loom_value_facts_is_power_of_two(loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_POWER_OF_TWO) != 0;
}

static inline bool loom_value_facts_is_boolean(loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_BOOLEAN) != 0;
}

static inline bool loom_value_facts_is_float(loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_FLOAT) != 0;
}

static inline bool loom_value_facts_is_uniform(loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_UNIFORM) != 0;
}

static inline bool loom_value_facts_is_lane_varying(loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_LANE_VARYING) != 0;
}

static inline bool loom_value_facts_is_lane_predicate(
    loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_LANE_PREDICATE) != 0;
}

static inline bool loom_value_facts_is_subgroup_lane_mask(
    loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_SUBGROUP_LANE_MASK) != 0;
}

static inline void loom_value_facts_mark_uniform(loom_value_facts_t* facts) {
  facts->flags &=
      ~(LOOM_VALUE_FACT_DISTRIBUTION_MASK | LOOM_VALUE_FACT_LANE_PREDICATE);
  facts->flags |= LOOM_VALUE_FACT_UNIFORM;
}

static inline void loom_value_facts_mark_lane_varying(
    loom_value_facts_t* facts) {
  facts->flags &= ~(LOOM_VALUE_FACT_UNIFORM | LOOM_VALUE_FACT_LANE_PREDICATE);
  facts->flags |= LOOM_VALUE_FACT_LANE_VARYING;
}

static inline void loom_value_facts_mark_lane_predicate(
    loom_value_facts_t* facts) {
  facts->flags &=
      ~(LOOM_VALUE_FACT_UNIFORM | LOOM_VALUE_FACT_SUBGROUP_LANE_MASK);
  facts->flags |= LOOM_VALUE_FACT_LANE_VARYING |
                  LOOM_VALUE_FACT_LANE_PREDICATE | LOOM_VALUE_FACT_BOOLEAN;
}

static inline void loom_value_facts_mark_subgroup_lane_mask(
    loom_value_facts_t* facts) {
  facts->flags &= ~LOOM_VALUE_FACT_LANE_PREDICATE;
  facts->flags |= LOOM_VALUE_FACT_SUBGROUP_LANE_MASK;
}

// Applies the conservative execution-distribution transfer for a unary op. An
// exact result is uniform; otherwise lane-varying inputs produce lane-varying
// results and uniform inputs produce uniform results.
void loom_value_facts_propagate_unary_distribution(loom_value_facts_t input,
                                                   loom_value_facts_t* out);

// Applies the conservative execution-distribution transfer for a binary op.
void loom_value_facts_propagate_binary_distribution(loom_value_facts_t lhs,
                                                    loom_value_facts_t rhs,
                                                    loom_value_facts_t* out);

// Applies the conservative execution-distribution transfer for a ternary op.
void loom_value_facts_propagate_ternary_distribution(loom_value_facts_t a,
                                                     loom_value_facts_t b,
                                                     loom_value_facts_t c,
                                                     loom_value_facts_t* out);

// Returns true when the integer fact domain proves the value is exactly zero.
static inline bool loom_value_facts_is_zero(loom_value_facts_t facts) {
  return !loom_value_facts_is_float(facts) && facts.range_lo == 0 &&
         facts.range_hi == 0;
}

// Returns true when the integer fact domain proves the value is false.
static inline bool loom_value_facts_is_false(loom_value_facts_t facts) {
  return loom_value_facts_is_zero(facts);
}

// Returns true when the integer fact domain proves the value is true.
static inline bool loom_value_facts_is_true(loom_value_facts_t facts) {
  return loom_value_facts_is_non_zero(facts);
}

static inline bool loom_value_facts_as_exact_i64(loom_value_facts_t facts,
                                                 int64_t* out_value) {
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

static inline bool loom_value_facts_as_exact_bool(loom_value_facts_t facts,
                                                  bool* out_value) {
  int64_t value = 0;
  if (!loom_value_facts_as_exact_i64(facts, &value)) return false;
  *out_value = value != 0;
  return true;
}

static inline bool loom_value_facts_divisible_by(loom_value_facts_t facts,
                                                 int64_t divisor) {
  return facts.known_divisor % divisor == 0;
}

// Returns true if all values described by |facts| fit in a signed integer with
// |bit_count| bits. Unknown facts never fit sub-64-bit domains because that
// would permit silent narrowing.
static inline bool loom_value_facts_fit_signed_bit_count(
    loom_value_facts_t facts, uint8_t bit_count) {
  if (bit_count == 0 || loom_value_facts_is_float(facts)) {
    return false;
  }
  if (bit_count >= 64) {
    return true;
  }
  const int64_t minimum_value = -(INT64_C(1) << (bit_count - 1));
  const int64_t maximum_value = (INT64_C(1) << (bit_count - 1)) - 1;
  return facts.range_lo >= minimum_value && facts.range_hi <= maximum_value;
}

// Returns true if all values described by |facts| fit in an unsigned integer
// with |bit_count| bits. Facts use signed ranges, so any negative lower bound
// rejects the unsigned domain.
static inline bool loom_value_facts_fit_unsigned_bit_count(
    loom_value_facts_t facts, uint8_t bit_count) {
  if (bit_count == 0 || loom_value_facts_is_float(facts) ||
      facts.range_lo < 0) {
    return false;
  }
  if (bit_count >= 63) {
    return true;
  }
  const uint64_t maximum_value = (UINT64_C(1) << bit_count) - 1;
  return (uint64_t)facts.range_hi <= maximum_value;
}

// Returns true if the facts carry no information (full range, unit
// divisor, no flags).
static inline bool loom_value_facts_is_unknown(loom_value_facts_t facts) {
  return facts.range_lo == INT64_MIN && facts.range_hi == INT64_MAX &&
         facts.known_divisor == 1 && facts.flags == 0 &&
         facts.extension_id == LOOM_VALUE_FACT_EXTENSION_ID_NONE;
}

//===----------------------------------------------------------------------===//
// Comparison
//===----------------------------------------------------------------------===//

// Equality check for propagation damping. 32-byte memcmp.
static inline bool loom_value_facts_equal(loom_value_facts_t a,
                                          loom_value_facts_t b) {
  return memcmp(&a, &b, sizeof(loom_value_facts_t)) == 0;
}

//===----------------------------------------------------------------------===//
// Meet
//===----------------------------------------------------------------------===//

// Conservative join: widens range to the outer bounds, weakens divisor
// to the GCD. Used at join points (scf.if yields, block arguments).
static inline void loom_value_facts_meet(
    const loom_value_facts_t* IREE_RESTRICT a,
    const loom_value_facts_t* IREE_RESTRICT b,
    loom_value_facts_t* IREE_RESTRICT out) {
  const bool preserves_subgroup_lane_mask =
      (loom_value_facts_is_subgroup_lane_mask(*a) &&
       loom_value_facts_is_subgroup_lane_mask(*b)) ||
      (loom_value_facts_is_subgroup_lane_mask(*a) &&
       loom_value_facts_is_zero(*b)) ||
      (loom_value_facts_is_zero(*a) &&
       loom_value_facts_is_subgroup_lane_mask(*b));
  *out = loom_value_facts_make(
      a->range_lo < b->range_lo ? a->range_lo : b->range_lo,
      a->range_hi > b->range_hi ? a->range_hi : b->range_hi,
      loom_gcd_i64(a->known_divisor, b->known_divisor));
  if (preserves_subgroup_lane_mask) {
    loom_value_facts_mark_subgroup_lane_mask(out);
  }
  if (loom_value_facts_is_lane_predicate(*a) ||
      loom_value_facts_is_lane_predicate(*b)) {
    loom_value_facts_mark_lane_predicate(out);
  } else if (loom_value_facts_is_lane_varying(*a) ||
             loom_value_facts_is_lane_varying(*b)) {
    loom_value_facts_mark_lane_varying(out);
  } else if (loom_value_facts_is_uniform(*a) &&
             loom_value_facts_is_uniform(*b)) {
    loom_value_facts_mark_uniform(out);
  }
}

//===----------------------------------------------------------------------===//
// Predicate application
//===----------------------------------------------------------------------===//

// Recomputes the cached flags from range_lo, range_hi, and
// known_divisor. Preserves POWER_OF_TWO and FLOAT flags if already
// set (they may come from predicates, not just range analysis).
void loom_value_facts_recompute_flags(loom_value_facts_t* facts);

// Tightens facts using a single predicate constraint. Modifies the
// facts in place and recomputes flags afterward.
void loom_value_facts_apply_predicate(loom_value_facts_t* facts,
                                      const loom_predicate_t* predicate);

//===----------------------------------------------------------------------===//
// Transfer functions
//===----------------------------------------------------------------------===//
//
// Compute output facts from input facts for each arithmetic operation.
// The output pointer MAY alias one input for in-place accumulation
// (e.g., loom_value_facts_muli(&acc, &dim, &acc)). Do not alias both
// inputs with the output simultaneously.
//
// For exact inputs, these produce exact outputs (constant folding).
// For range inputs, they compute range bounds with checked arithmetic
// and fall back to unknown on overflow.

// Binary arithmetic.
void loom_value_facts_addi(const loom_value_facts_t* lhs,
                           const loom_value_facts_t* rhs,
                           loom_value_facts_t* out);
void loom_value_facts_subi(const loom_value_facts_t* lhs,
                           const loom_value_facts_t* rhs,
                           loom_value_facts_t* out);
void loom_value_facts_muli(const loom_value_facts_t* lhs,
                           const loom_value_facts_t* rhs,
                           loom_value_facts_t* out);
void loom_value_facts_divui(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out);
void loom_value_facts_divsi(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out);
void loom_value_facts_remui(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out);
void loom_value_facts_remsi(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out);

// Shifts.
void loom_value_facts_shli(const loom_value_facts_t* lhs,
                           const loom_value_facts_t* rhs,
                           loom_value_facts_t* out);
void loom_value_facts_shrui(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out);
void loom_value_facts_shrsi(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out);

// Bitwise.
void loom_value_facts_andi(const loom_value_facts_t* lhs,
                           const loom_value_facts_t* rhs,
                           loom_value_facts_t* out);
void loom_value_facts_ori(const loom_value_facts_t* lhs,
                          const loom_value_facts_t* rhs,
                          loom_value_facts_t* out);
void loom_value_facts_xori(const loom_value_facts_t* lhs,
                           const loom_value_facts_t* rhs,
                           loom_value_facts_t* out);

// Min / max.
void loom_value_facts_minsi(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out);
void loom_value_facts_maxsi(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out);
void loom_value_facts_minui(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out);
void loom_value_facts_maxui(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out);

// Unary.
void loom_value_facts_negi(const loom_value_facts_t* input,
                           loom_value_facts_t* out);
void loom_value_facts_absi(const loom_value_facts_t* input,
                           loom_value_facts_t* out);

// Ternary: fused multiply-add (a*b + c).
void loom_value_facts_fmai(const loom_value_facts_t* a,
                           const loom_value_facts_t* b,
                           const loom_value_facts_t* c,
                           loom_value_facts_t* out);

//===----------------------------------------------------------------------===//
// Shaped type helpers
//===----------------------------------------------------------------------===//

// Computes facts for the element count of a shaped type.
//
// Static dimensions contribute exact factors. Dynamic dimensions are looked up
// in |value_facts| by their SSA value ID and intersected with the structural
// extent domain [0, INT64_MAX] before multiplication. Missing or malformed
// dynamic-dimension facts degrade to an unknown non-negative extent.
void loom_value_facts_element_count(loom_type_t type,
                                    const loom_value_facts_t* value_facts,
                                    iree_host_size_t value_fact_count,
                                    loom_value_facts_t* out_count);

// Returns the largest known divisor of the shaped type element count.
int64_t loom_value_facts_element_count_divisor(
    loom_type_t type, const loom_value_facts_t* value_facts,
    iree_host_size_t value_fact_count);

// Returns true when two shaped types are proven to have equal element counts.
//
// This is intentionally proof-oriented rather than similarity-oriented:
// structurally identical dimension lists are equal, and independently computed
// exact element counts with the same value are equal. Matching non-exact fact
// ranges are not enough to prove equality. Structural equality assumes both
// types use the same SSA value-id namespace; cross-module comparisons should
// rely on exact-count proof or a remap-aware caller.
bool loom_value_facts_element_counts_equal(
    loom_type_t lhs_type, const loom_value_facts_t* lhs_value_facts,
    iree_host_size_t lhs_value_fact_count, loom_type_t rhs_type,
    const loom_value_facts_t* rhs_value_facts,
    iree_host_size_t rhs_value_fact_count);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_IR_FACTS_H_
