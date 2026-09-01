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
#include "iree/base/internal/math.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Flags
//===----------------------------------------------------------------------===//

// Storage for cached predicate flags.
typedef uint32_t loom_value_fact_flags_t;

// Cached predicate flag bits derived from range, divisor, floating-point
// predicates, or target-independent execution distribution. Range flags are
// always consistent with the scalar range/divisor fields. Distribution flags
// describe the widest execution scope over which invocations observe the same
// SSA value.
enum loom_value_fact_flag_bits_e {
  // The signed range lower bound is >= 0.
  LOOM_VALUE_FACT_NON_NEGATIVE = 1u << 0,
  // The value cannot be zero. This may be range-derived or asserted by a
  // nonzero predicate.
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
  // constant value. For float types, range_lo contains a host double carrying
  // the value rounded to its declared scalar type (see FLOAT flag).
  LOOM_VALUE_FACT_EXACT = 1u << 4,
  // The range is [0, 1]. Typical for i1 comparison results.
  LOOM_VALUE_FACT_BOOLEAN = 1u << 5,
  // The value has a floating-point type. When EXACT is also set,
  // range_lo/range_hi contain a host double carrying the value rounded to its
  // declared scalar type (via memcpy), not an integer range. The declared type
  // is external to this compact summary and must accompany payload access.
  // Non-exact float facts do not carry range bounds, but may carry semantic
  // predicate facts such as NOT_NAN or FINITE.
  LOOM_VALUE_FACT_FLOAT = 1u << 6,
  // The value cannot be NaN. This may come from an exact float value or a
  // checked predicate; it is meaningful only for floating-point typed values.
  LOOM_VALUE_FACT_NOT_NAN = 1u << 11,
  // The value cannot be positive or negative infinity. This may come from an
  // exact float value or a checked predicate; it is meaningful only for
  // floating-point typed values.
  LOOM_VALUE_FACT_NOT_INF = 1u << 12,
  // The value cannot be NaN or infinity. This may come from an exact float
  // value or a checked predicate and implies NOT_NAN and NOT_INF.
  LOOM_VALUE_FACT_FINITE = 1u << 13,
  // The value cannot be a positive or negative subnormal number. This may come
  // from an exact float value or a storage contract that flushes subnormal
  // payloads to zero; it is meaningful only for floating-point typed values.
  LOOM_VALUE_FACT_NOT_SUBNORMAL = 1u << 23,
  // The value is known to be identical for every active lane in a subgroup.
  LOOM_VALUE_FACT_SUBGROUP_UNIFORM = 1u << 7,
  // The value may differ between active lanes observing it.
  LOOM_VALUE_FACT_LANE_VARYING = 1u << 8,
  // The i1 value is a lane mask/predicate rather than a scalar condition code.
  LOOM_VALUE_FACT_LANE_PREDICATE = 1u << 9,
  // The integer value encodes subgroup lane membership as one bit per lane.
  // This is independent of whether each lane observes the same mask value.
  LOOM_VALUE_FACT_SUBGROUP_LANE_MASK = 1u << 10,
  // The value is the full x-dimension workitem id domain for a kernel launch.
  LOOM_VALUE_FACT_TOPOLOGY_WORKITEM_X = 1u << 14,
  // The value is the full y-dimension workitem id domain for a kernel launch.
  LOOM_VALUE_FACT_TOPOLOGY_WORKITEM_Y = 1u << 15,
  // The value is the full z-dimension workitem id domain for a kernel launch.
  LOOM_VALUE_FACT_TOPOLOGY_WORKITEM_Z = 1u << 16,
  // The value is the full subgroup lane id domain for a kernel launch.
  LOOM_VALUE_FACT_TOPOLOGY_SUBGROUP_LANE = 1u << 17,
  // The value is the full x-dimension workgroup id domain for a kernel launch.
  LOOM_VALUE_FACT_TOPOLOGY_WORKGROUP_X = 1u << 18,
  // The value is the full y-dimension workgroup id domain for a kernel launch.
  LOOM_VALUE_FACT_TOPOLOGY_WORKGROUP_Y = 1u << 19,
  // The value is the full z-dimension workgroup id domain for a kernel launch.
  LOOM_VALUE_FACT_TOPOLOGY_WORKGROUP_Z = 1u << 20,
  // The floating-point value is known to be NaN. This does not imply EXACT:
  // facts derived from a raw signaling-NaN bit pattern intentionally preserve
  // the producing operation because the compact fact payload does not retain
  // NaN payload bits.
  LOOM_VALUE_FACT_NAN = 1u << 21,
  // The floating-point value is known to be positive or negative infinity.
  LOOM_VALUE_FACT_INF = 1u << 22,
  // The value is known to be identical for every invocation in a workgroup.
  // This implies SUBGROUP_UNIFORM and both bits are set together.
  LOOM_VALUE_FACT_WORKGROUP_UNIFORM = 1u << 24,
  // The value is the full x-dimension workgroup-cluster id domain for a
  // kernel launch.
  LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_X = 1u << 25,
  // The value is the full y-dimension workgroup-cluster id domain for a
  // kernel launch.
  LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_Y = 1u << 26,
  // The value is the full z-dimension workgroup-cluster id domain for a
  // kernel launch.
  LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_Z = 1u << 27,
  // The value is the full x-dimension within-cluster workgroup id domain.
  LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_WORKGROUP_X = 1u << 28,
  // The value is the full y-dimension within-cluster workgroup id domain.
  LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_WORKGROUP_Y = 1u << 29,
  // The value is the full z-dimension within-cluster workgroup id domain.
  LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_WORKGROUP_Z = 1u << 30,
};

// The value is known to be identical for every invocation in every workgroup
// of a workgroup cluster. This implies WORKGROUP_UNIFORM and SUBGROUP_UNIFORM,
// and all three bits are set together. Bit 31 is an unsigned integer constant
// instead of an enumerator because C requires enumerator values to fit in int.
#define LOOM_VALUE_FACT_CLUSTER_UNIFORM UINT32_C(0x80000000)

#define LOOM_VALUE_FACT_UNIFORM_SCOPE_MASK                                \
  (LOOM_VALUE_FACT_SUBGROUP_UNIFORM | LOOM_VALUE_FACT_WORKGROUP_UNIFORM | \
   LOOM_VALUE_FACT_CLUSTER_UNIFORM)

#define LOOM_VALUE_FACT_DISTRIBUTION_MASK \
  (LOOM_VALUE_FACT_UNIFORM_SCOPE_MASK | LOOM_VALUE_FACT_LANE_VARYING)

// Execution scope over which a value is proven identical. Larger values are
// stronger facts and satisfy queries for every smaller scope.
typedef enum loom_value_fact_uniform_scope_e {
  LOOM_VALUE_FACT_UNIFORM_SCOPE_NONE = 0,
  LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP = 1,
  LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP = 2,
  LOOM_VALUE_FACT_UNIFORM_SCOPE_CLUSTER = 3,
} loom_value_fact_uniform_scope_t;

#define LOOM_VALUE_FACT_TOPOLOGY_DOMAIN_MASK                                   \
  (LOOM_VALUE_FACT_TOPOLOGY_WORKITEM_X | LOOM_VALUE_FACT_TOPOLOGY_WORKITEM_Y | \
   LOOM_VALUE_FACT_TOPOLOGY_WORKITEM_Z |                                       \
   LOOM_VALUE_FACT_TOPOLOGY_WORKGROUP_X |                                      \
   LOOM_VALUE_FACT_TOPOLOGY_WORKGROUP_Y |                                      \
   LOOM_VALUE_FACT_TOPOLOGY_WORKGROUP_Z | LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_X | \
   LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_Y | LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_Z |   \
   LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_WORKGROUP_X |                              \
   LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_WORKGROUP_Y |                              \
   LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_WORKGROUP_Z |                              \
   LOOM_VALUE_FACT_TOPOLOGY_SUBGROUP_LANE)

// Floating-point semantic predicate facts. These may describe scalar values or
// shaped floating values when the predicate is known for every lane.
#define LOOM_VALUE_FACT_FLOAT_PREDICATE_MASK                           \
  (LOOM_VALUE_FACT_FLOAT | LOOM_VALUE_FACT_NAN | LOOM_VALUE_FACT_INF | \
   LOOM_VALUE_FACT_NOT_NAN | LOOM_VALUE_FACT_NOT_INF |                 \
   LOOM_VALUE_FACT_FINITE | LOOM_VALUE_FACT_NOT_SUBNORMAL)

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

// Target-independent topology value family.
typedef enum loom_value_fact_topology_value_kind_e {
  // No topology value family is known.
  LOOM_VALUE_FACT_TOPOLOGY_VALUE_NONE = 0,
  // Per-workitem coordinate within the current workgroup.
  LOOM_VALUE_FACT_TOPOLOGY_VALUE_WORKITEM_ID = 1,
  // Per-workgroup coordinate within the current dispatch grid.
  // For clustered launches and a selected axis, this is reconstructed as
  // cluster.id * cluster.size + cluster.workgroup.id.
  LOOM_VALUE_FACT_TOPOLOGY_VALUE_WORKGROUP_ID = 2,
  // Per-subgroup-lane coordinate within the current subgroup.
  LOOM_VALUE_FACT_TOPOLOGY_VALUE_SUBGROUP_LANE_ID = 3,
  // Per-cluster coordinate within the dispatch cluster grid. Together with a
  // same-axis CLUSTER_WORKGROUP_ID, this decomposes WORKGROUP_ID according to
  // the selected static cluster size.
  LOOM_VALUE_FACT_TOPOLOGY_VALUE_CLUSTER_ID = 4,
  // Per-workgroup coordinate within the current workgroup cluster. Together
  // with a same-axis CLUSTER_ID, this decomposes WORKGROUP_ID according to the
  // selected static cluster size.
  LOOM_VALUE_FACT_TOPOLOGY_VALUE_CLUSTER_WORKGROUP_ID = 5,
  LOOM_VALUE_FACT_TOPOLOGY_VALUE_COUNT_ = 6,
} loom_value_fact_topology_value_kind_t;

// Target-independent topology coordinate axis.
typedef enum loom_value_fact_topology_axis_e {
  LOOM_VALUE_FACT_TOPOLOGY_AXIS_X = 0,
  LOOM_VALUE_FACT_TOPOLOGY_AXIS_Y = 1,
  LOOM_VALUE_FACT_TOPOLOGY_AXIS_Z = 2,
  LOOM_VALUE_FACT_TOPOLOGY_AXIS_LANE = 3,
  LOOM_VALUE_FACT_TOPOLOGY_AXIS_COUNT_ = 4,
} loom_value_fact_topology_axis_t;

// Decoded topology fact domain.
typedef struct loom_value_fact_topology_domain_t {
  // Fact flag that uniquely identifies the topology domain.
  loom_value_fact_flags_t fact_flag;
  // Value family represented by the domain.
  loom_value_fact_topology_value_kind_t value_kind;
  // Coordinate axis represented by the domain.
  loom_value_fact_topology_axis_t axis;
  // Stable value-family spelling for structured diagnostics.
  iree_string_view_t value_kind_name;
  // Stable axis spelling for structured diagnostics.
  iree_string_view_t axis_name;
} loom_value_fact_topology_domain_t;

//===----------------------------------------------------------------------===//
// Struct
//===----------------------------------------------------------------------===//

// Per-value analysis facts. 32 bytes, cache-friendly for dense arrays.
typedef struct loom_value_facts_t {
  // Signed integer range [lo, hi], inclusive. Default (unknown):
  // [INT64_MIN, INT64_MAX]. For float types with EXACT set, range_lo contains
  // a host double carrying the value rounded to its declared scalar type (via
  // memcpy); range_hi == range_lo. The declared type is stored in the IR.
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

// Returns the topology domain encoded by |flags|. Exactly one topology domain
// bit must be present.
const loom_value_fact_topology_domain_t*
loom_value_fact_topology_domain_from_flags(loom_value_fact_flags_t flags);

// Returns the topology domain encoded by |facts|, if any.
const loom_value_fact_topology_domain_t* loom_value_facts_topology_domain(
    loom_value_facts_t facts);

// Returns the topology domain matching |value_kind| and |axis|, if any.
const loom_value_fact_topology_domain_t* loom_value_fact_topology_domain_lookup(
    loom_value_fact_topology_value_kind_t value_kind,
    loom_value_fact_topology_axis_t axis);

// Marks |facts| with the topology domain matching |value_kind| and |axis|.
// Returns false if the requested domain does not exist.
bool loom_value_facts_mark_topology_domain(
    loom_value_facts_t* facts, loom_value_fact_topology_value_kind_t value_kind,
    loom_value_fact_topology_axis_t axis);

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

// Returns facts for a signed extension from |source_bit_count|. Fixed-width
// integer facts already use their signed numeric domain except for logical i1
// facts, whose 0/1 domain is mapped to 0/-1.
loom_value_facts_t loom_value_facts_sign_extend(loom_value_facts_t source_facts,
                                                int32_t source_bit_count);

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

static inline bool loom_value_facts_is_nan(loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_NAN) != 0;
}

static inline bool loom_value_facts_is_inf(loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_INF) != 0;
}

static inline bool loom_value_facts_is_not_nan(loom_value_facts_t facts) {
  return (facts.flags & (LOOM_VALUE_FACT_NOT_NAN | LOOM_VALUE_FACT_FINITE)) !=
         0;
}

static inline bool loom_value_facts_is_not_inf(loom_value_facts_t facts) {
  return (facts.flags & (LOOM_VALUE_FACT_NOT_INF | LOOM_VALUE_FACT_FINITE)) !=
         0;
}

static inline bool loom_value_facts_is_finite(loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_FINITE) != 0;
}

static inline bool loom_value_facts_is_not_subnormal(loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_NOT_SUBNORMAL) != 0;
}

static inline loom_value_fact_uniform_scope_t loom_value_facts_uniform_scope(
    loom_value_facts_t facts) {
  if (iree_all_bits_set(facts.flags, LOOM_VALUE_FACT_CLUSTER_UNIFORM)) {
    return LOOM_VALUE_FACT_UNIFORM_SCOPE_CLUSTER;
  }
  if (iree_all_bits_set(facts.flags, LOOM_VALUE_FACT_WORKGROUP_UNIFORM)) {
    return LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP;
  }
  if (iree_all_bits_set(facts.flags, LOOM_VALUE_FACT_SUBGROUP_UNIFORM)) {
    return LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP;
  }
  return LOOM_VALUE_FACT_UNIFORM_SCOPE_NONE;
}

static inline bool loom_value_facts_is_uniform_at_scope(
    loom_value_facts_t facts, loom_value_fact_uniform_scope_t scope) {
  return loom_value_facts_uniform_scope(facts) >= scope;
}

static inline bool loom_value_facts_is_subgroup_uniform(
    loom_value_facts_t facts) {
  return loom_value_facts_is_uniform_at_scope(
      facts, LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP);
}

static inline bool loom_value_facts_is_workgroup_uniform(
    loom_value_facts_t facts) {
  return loom_value_facts_is_uniform_at_scope(
      facts, LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP);
}

static inline bool loom_value_facts_is_cluster_uniform(
    loom_value_facts_t facts) {
  return loom_value_facts_is_uniform_at_scope(
      facts, LOOM_VALUE_FACT_UNIFORM_SCOPE_CLUSTER);
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

static inline void loom_value_facts_mark_uniform_at_scope(
    loom_value_facts_t* facts, loom_value_fact_uniform_scope_t scope) {
  facts->flags &=
      ~(LOOM_VALUE_FACT_DISTRIBUTION_MASK | LOOM_VALUE_FACT_LANE_PREDICATE);
  if (scope >= LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP) {
    facts->flags |= LOOM_VALUE_FACT_SUBGROUP_UNIFORM;
  }
  if (scope >= LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP) {
    facts->flags |= LOOM_VALUE_FACT_WORKGROUP_UNIFORM;
  }
  if (scope >= LOOM_VALUE_FACT_UNIFORM_SCOPE_CLUSTER) {
    facts->flags |= LOOM_VALUE_FACT_CLUSTER_UNIFORM;
  }
}

static inline void loom_value_facts_mark_subgroup_uniform(
    loom_value_facts_t* facts) {
  loom_value_facts_mark_uniform_at_scope(
      facts, LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP);
}

static inline void loom_value_facts_mark_workgroup_uniform(
    loom_value_facts_t* facts) {
  loom_value_facts_mark_uniform_at_scope(
      facts, LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP);
}

static inline void loom_value_facts_mark_cluster_uniform(
    loom_value_facts_t* facts) {
  loom_value_facts_mark_uniform_at_scope(facts,
                                         LOOM_VALUE_FACT_UNIFORM_SCOPE_CLUSTER);
}

static inline void loom_value_facts_mark_lane_varying(
    loom_value_facts_t* facts) {
  facts->flags &=
      ~(LOOM_VALUE_FACT_UNIFORM_SCOPE_MASK | LOOM_VALUE_FACT_LANE_PREDICATE);
  facts->flags |= LOOM_VALUE_FACT_LANE_VARYING;
}

static inline void loom_value_facts_mark_lane_predicate(
    loom_value_facts_t* facts) {
  facts->flags &= ~(LOOM_VALUE_FACT_UNIFORM_SCOPE_MASK |
                    LOOM_VALUE_FACT_SUBGROUP_LANE_MASK);
  facts->flags |= LOOM_VALUE_FACT_LANE_VARYING |
                  LOOM_VALUE_FACT_LANE_PREDICATE | LOOM_VALUE_FACT_BOOLEAN;
}

static inline void loom_value_facts_mark_subgroup_lane_mask(
    loom_value_facts_t* facts) {
  facts->flags &= ~LOOM_VALUE_FACT_LANE_PREDICATE;
  facts->flags |= LOOM_VALUE_FACT_SUBGROUP_LANE_MASK;
}

// Applies the conservative execution-distribution transfer for a unary op. An
// exact result is cluster-uniform; otherwise lane-varying inputs produce
// lane-varying results and uniform inputs preserve their uniform scope.
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

// Returns the reservable maximum of a proven non-negative integer range.
// INT64_MAX denotes an unbounded range and is not a reservable maximum.
static inline bool loom_value_facts_as_non_negative_i64_maximum(
    loom_value_facts_t facts, int64_t* out_maximum) {
  if (loom_value_facts_is_float(facts) || facts.range_lo < 0 ||
      facts.range_hi == INT64_MAX) {
    return false;
  }
  *out_maximum = facts.range_hi;
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
  if (loom_value_facts_is_float(*a) || loom_value_facts_is_float(*b)) {
    *out = loom_value_facts_unknown();
    if (loom_value_facts_is_float(*a) && loom_value_facts_is_float(*b)) {
      out->flags = LOOM_VALUE_FACT_FLOAT |
                   (a->flags & b->flags &
                    (LOOM_VALUE_FACT_NAN | LOOM_VALUE_FACT_INF |
                     LOOM_VALUE_FACT_NOT_NAN | LOOM_VALUE_FACT_NOT_INF |
                     LOOM_VALUE_FACT_FINITE | LOOM_VALUE_FACT_NOT_SUBNORMAL));
      if (loom_value_facts_is_exact(*a) && loom_value_facts_is_exact(*b) &&
          a->range_lo == b->range_lo) {
        out->range_lo = a->range_lo;
        out->range_hi = a->range_hi;
        out->flags |= LOOM_VALUE_FACT_EXACT;
      }
    }
    if (loom_value_facts_is_exact(*out)) {
      loom_value_facts_mark_cluster_uniform(out);
    } else if (loom_value_facts_is_lane_varying(*a) ||
               loom_value_facts_is_lane_varying(*b)) {
      loom_value_facts_mark_lane_varying(out);
    } else {
      const loom_value_fact_uniform_scope_t uniform_scope =
          iree_min(loom_value_facts_uniform_scope(*a),
                   loom_value_facts_uniform_scope(*b));
      loom_value_facts_mark_uniform_at_scope(out, uniform_scope);
    }
    return;
  }

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
      iree_math_gcd_i64(a->known_divisor, b->known_divisor));
  if (preserves_subgroup_lane_mask) {
    loom_value_facts_mark_subgroup_lane_mask(out);
  }
  if (loom_value_facts_is_lane_predicate(*a) ||
      loom_value_facts_is_lane_predicate(*b)) {
    loom_value_facts_mark_lane_predicate(out);
  } else if (loom_value_facts_is_lane_varying(*a) ||
             loom_value_facts_is_lane_varying(*b)) {
    loom_value_facts_mark_lane_varying(out);
  } else {
    const loom_value_fact_uniform_scope_t uniform_scope = iree_min(
        loom_value_facts_uniform_scope(*a), loom_value_facts_uniform_scope(*b));
    loom_value_facts_mark_uniform_at_scope(out, uniform_scope);
  }
}

//===----------------------------------------------------------------------===//
// Predicate application
//===----------------------------------------------------------------------===//

typedef enum loom_value_fact_predicate_conflict_kind_e {
  LOOM_VALUE_FACT_PREDICATE_CONFLICT_NONE = 0,
  LOOM_VALUE_FACT_PREDICATE_CONFLICT_RANGE = 1,
  LOOM_VALUE_FACT_PREDICATE_CONFLICT_EXACT_I64 = 2,
  LOOM_VALUE_FACT_PREDICATE_CONFLICT_FLOAT_CLASS = 3,
} loom_value_fact_predicate_conflict_kind_t;

typedef enum loom_value_fact_predicate_conflict_float_class_e {
  LOOM_VALUE_FACT_PREDICATE_CONFLICT_FLOAT_CLASS_NONE = 0,
  LOOM_VALUE_FACT_PREDICATE_CONFLICT_FLOAT_CLASS_NAN = 1,
  LOOM_VALUE_FACT_PREDICATE_CONFLICT_FLOAT_CLASS_INFINITY = 2,
} loom_value_fact_predicate_conflict_float_class_t;

typedef struct loom_value_fact_predicate_conflict_t {
  // Kind of contradiction proven by the fact lattice.
  loom_value_fact_predicate_conflict_kind_t kind;

  // Minimum known value from the incoming fact range.
  int64_t known_minimum;

  // Maximum known value from the incoming fact range.
  int64_t known_maximum;

  // Minimum value required by the predicate range.
  int64_t required_minimum;

  // Maximum value required by the predicate range.
  int64_t required_maximum;

  // Exact integer value proven by incoming facts.
  int64_t known_value;

  // Integer predicate parameter that conflicts with |known_value|.
  int64_t predicate_value;

  // Exact floating-point class that conflicts with a float predicate.
  loom_value_fact_predicate_conflict_float_class_t known_float_class;
} loom_value_fact_predicate_conflict_t;

// Recomputes the cached flags from range_lo, range_hi, and known_divisor.
// Preserves facts that are not derived from the integer range, such as
// predicate, floating-point type, and execution distribution flags.
void loom_value_facts_recompute_flags(loom_value_facts_t* facts);

// Returns true when |predicate| is statically impossible for |facts|. This is a
// validation query, not a transfer function: callers that accept user assumes
// should diagnose true results before applying the predicate to avoid turning
// provably empty facts into ordinary unknown facts.
bool loom_value_facts_predicate_conflict(
    loom_value_facts_t facts, const loom_predicate_t* predicate,
    loom_value_fact_predicate_conflict_t* out_conflict);

// Tightens facts using a single predicate constraint. Modifies the facts in
// place and recomputes flags afterward. The caller is responsible for enforcing
// that the predicate is legal for the value type.
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
