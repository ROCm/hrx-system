// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Storage-reference facts shared by inference and memory analyses.

#ifndef LOOM_UTIL_REFERENCE_FACTS_H_
#define LOOM_UTIL_REFERENCE_FACTS_H_

#include "loom/ir/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

// Intrinsic storage provenance within a projected function invocation.
// These are allocation guarantees, independent of explicit noalias promises.
// Entry storage exists before allocations made during the same invocation.
// Derived aliases retain their origin. Unknown or mixed origins prove nothing.
enum loom_value_fact_reference_origin_kind_e {
  LOOM_VALUE_FACT_REFERENCE_ORIGIN_UNKNOWN = 0,
  LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY = 1,
  LOOM_VALUE_FACT_REFERENCE_ORIGIN_ALLOCATION = 2,
};
typedef uint8_t loom_value_fact_reference_origin_kind_t;

typedef struct loom_value_fact_reference_origin_t {
  // Module-local symbol naming the function whose invocation owns the origin.
  uint16_t function_symbol_id;
  // Projected region of that function, separating configuration and execution.
  uint8_t region_index;
  // Storage's relationship to the entry of the projected function region.
  loom_value_fact_reference_origin_kind_t kind;
  // Stable SSA identity of the entry argument supplying storage, or INVALID
  // for a join of inputs or a local allocation. Direct calls substitute the
  // actual argument at this value's current ordinal after signature changes.
  loom_value_id_t entry_value_id;
} loom_value_fact_reference_origin_t;

static_assert(sizeof(loom_value_fact_reference_origin_t) == 8,
              "reference origins must remain eight bytes");

// Returns whether both records describe the same origin guarantee.
bool loom_value_fact_reference_origin_equal(
    loom_value_fact_reference_origin_t lhs,
    loom_value_fact_reference_origin_t rhs);

// Retains the origin guarantee common to both alternatives at a control join.
loom_value_fact_reference_origin_t loom_value_fact_reference_origin_meet(
    loom_value_fact_reference_origin_t lhs,
    loom_value_fact_reference_origin_t rhs);

// Proves entry storage and storage freshly allocated during the same invocation
// disjoint. Call-boundary producers translate origins through actual arguments;
// callee entry origins cannot be copied into a caller without substitution.
bool loom_value_fact_reference_origins_are_disjoint(
    loom_value_fact_reference_origin_t lhs,
    loom_value_fact_reference_origin_t rhs);

// Known reference nullability for storage-like values.
typedef uint8_t loom_value_fact_reference_nullability_t;
#define LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN \
  ((loom_value_fact_reference_nullability_t)0)
#define LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NULL \
  ((loom_value_fact_reference_nullability_t)1)
#define LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NON_NULL \
  ((loom_value_fact_reference_nullability_t)2)

// Comparable alias scope for storage-like values. NONE means root_value_id is
// only provenance for addressing and same-root propagation; consumers must not
// use it to prove disjointness against another root.
typedef loom_value_id_t loom_value_fact_alias_scope_id_t;
#define LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE \
  ((loom_value_fact_alias_scope_id_t)LOOM_VALUE_ID_INVALID)

// Buffer value is an opaque storage root.
typedef struct loom_value_fact_buffer_reference_t {
  // Conservative byte extent facts for the root storage allocation.
  loom_value_facts_t maximum_byte_extent;

  // Minimum provable byte alignment of the root storage base. One means
  // unknown beyond byte alignment.
  uint64_t minimum_alignment;

  // Target-independent memory space for the storage root.
  loom_value_fact_memory_space_t memory_space;

  // SSA value that represents the root storage identity. INVALID means the
  // value carrying these facts is itself the dynamic storage root.
  loom_value_id_t root_value_id;

  // Comparable alias scope for disjointness proofs, or NONE.
  loom_value_fact_alias_scope_id_t alias_scope_id;

  // Known nullability for the storage root.
  loom_value_fact_reference_nullability_t nullability;

  // Intrinsic storage origin relative to a projected function entry.
  loom_value_fact_reference_origin_t origin;
} loom_value_fact_buffer_reference_t;

static_assert(sizeof(loom_value_fact_buffer_reference_t) == 64,
              "buffer-reference facts must remain 64 bytes");

// Resolves the concrete storage root for |reference_value_id|. Buffer fact
// joins use a self-root when control flow chooses between distinct roots.
static inline loom_value_id_t
loom_value_fact_buffer_reference_resolve_root_value(
    loom_value_fact_buffer_reference_t reference,
    loom_value_id_t reference_value_id) {
  return reference.root_value_id == LOOM_VALUE_ID_INVALID
             ? reference_value_id
             : reference.root_value_id;
}

// View value is a typed projection over a storage root.
typedef struct loom_value_fact_view_reference_t {
  // Byte offset facts for the view base relative to root_value_id.
  loom_value_facts_t base_byte_offset;

  // Conservative byte length facts for the whole-view footprint envelope.
  loom_value_facts_t footprint_byte_length;

  // Minimum provable alignment of base_byte_offset relative to root_value_id.
  // The root's own absolute pointer alignment is tracked separately.
  uint64_t minimum_alignment;

  // Minimum provable byte alignment of the root storage base. One means
  // unknown beyond byte alignment.
  uint64_t root_minimum_alignment;

  // Static addressed element byte count, or -1 for sub-byte/unknown elements.
  int32_t static_element_byte_count;

  // Target-independent memory space for the underlying storage root.
  loom_value_fact_memory_space_t memory_space;

  // SSA value that represents the root storage identity. INVALID means the
  // value carrying these facts is its own dynamic root with base offset zero.
  loom_value_id_t root_value_id;

  // Buffer SSA value that materializes the root, including its authored
  // alignment and alias assumptions. INVALID when no single buffer is shared
  // by every incoming view. Availability at a use is a dominance question.
  loom_value_id_t buffer_value_id;

  // Comparable alias scope for disjointness proofs, or NONE.
  loom_value_fact_alias_scope_id_t alias_scope_id;

  // Known nullability for the underlying storage root.
  loom_value_fact_reference_nullability_t nullability;

  // Unsigned source address-carrier width guaranteed for valid active accesses
  // through this view, or zero when no source carrier contract is known.
  uint8_t address_bitwidth;

  // Intrinsic storage origin relative to a projected function entry.
  loom_value_fact_reference_origin_t origin;
} loom_value_fact_view_reference_t;

static_assert(sizeof(loom_value_fact_view_reference_t) == 112,
              "view-reference facts must remain 112 bytes");

// Resolves the storage root of a view, including joins of distinct roots.
static inline loom_value_id_t loom_value_fact_view_reference_resolve_root_value(
    loom_value_fact_view_reference_t reference,
    loom_value_id_t reference_value_id) {
  return reference.root_value_id == LOOM_VALUE_ID_INVALID
             ? reference_value_id
             : reference.root_value_id;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_UTIL_REFERENCE_FACTS_H_
