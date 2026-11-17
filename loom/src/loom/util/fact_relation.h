// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Retained signed integer relations carried by scoped value aliases.

#ifndef LOOM_UTIL_FACT_RELATION_H_
#define LOOM_UTIL_FACT_RELATION_H_

#include "loom/analysis/integer_relation.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Canonical immutable relation sets within one populated table scope. Roots
// remain comparable across inference and withdrawal until the scope is cleared.
typedef struct loom_value_fact_relation_roots_t {
  // Counterparts excluded from each signed ordering outcome.
  uint32_t excluded[LOOM_INTEGER_ORDER_OUTCOME_COUNT];
  // Signed literals excluded from equality with the value.
  uint32_t excluded_literal_equal;
} loom_value_fact_relation_roots_t;

// Persistent set root containing interned relation-atom IDs. Empty is zero,
// the high bit tags an inline singleton, and other values name canonical
// set-index roots. Interning preserves the full 32-bit SSA value-ID and 64-bit
// literal domains while keeping one-counterpart relations out of the index.
typedef uint32_t loom_value_fact_relation_root_t;

typedef struct loom_value_fact_relation_entry_t {
  // Scoped alias whose canonical relation sets are retained.
  loom_value_id_t value_id;
  // Current publication; zero roots retain reusable empty entries.
  loom_value_fact_relation_roots_t roots;
} loom_value_fact_relation_entry_t;

typedef enum loom_value_fact_relation_atom_kind_e {
  LOOM_VALUE_FACT_RELATION_ATOM_VALUE = 0,
  LOOM_VALUE_FACT_RELATION_ATOM_LITERAL = 1,
} loom_value_fact_relation_atom_kind_t;

// One interned counterpart identity or signed literal. IDs are one-based and
// remain stable as storage grows. Hash chains are populated only after the
// bounded linear-scan path fills.
typedef struct loom_value_fact_relation_atom_t {
  // Counterpart value ID or the bit pattern of a signed literal.
  uint64_t bits;
  // Next one-based atom ID in its hash collision chain, or zero.
  uint32_t next;
  // Whether bits represents an SSA identity or a signed literal.
  uint8_t kind;
  // Padding kept zero when an atom is constructed.
  uint8_t reserved[3];
} loom_value_fact_relation_atom_t;

static_assert(sizeof(loom_value_fact_relation_roots_t) == 16,
              "relation roots must remain 16 bytes");
static_assert(sizeof(loom_value_fact_relation_entry_t) == 20,
              "sparse relation entries must remain 20 bytes");
static_assert(sizeof(loom_value_fact_relation_atom_t) == 16,
              "relation atoms must remain 16 bytes");

struct loom_value_fact_integer_relations_t {
  // Canonical atom sets shared by every sparse entry. Allocated only when one
  // root grows beyond its inline singleton.
  loom_value_set_index_t* index;

  // Sparse entries sorted by scoped alias value ID.
  loom_value_fact_relation_entry_t* entries;

  // Number of sparse entries, including reusable zero-root tombstones.
  iree_host_size_t count;

  // Allocated sparse entry capacity.
  iree_host_size_t capacity;

  // Interned counterpart identities and literal disequalities.
  struct {
    // Interned atom records in ID order.
    loom_value_fact_relation_atom_t* values;

    // Number of interned atom records.
    iree_host_size_t value_count;

    // Allocated atom record capacity.
    iree_host_size_t value_capacity;

    // One-based atom-ID heads for hash collision chains.
    uint32_t* buckets;

    // Power-of-two bucket count, or zero on the bounded linear scan path.
    iree_host_size_t bucket_count;

  } atoms;

  // Reusable storage for identity-operation plans larger than the stack path.
  struct {
    // Canonical source identity per result ordinal.
    loom_value_id_t* identities;

    // Inherited roots and completed roots per result ordinal.
    loom_value_fact_relation_roots_t* plans;

    // Predicate-local roots keyed by first canonical identity ordinal.
    loom_value_fact_relation_roots_t* locals;

    // Allocated entries in each parallel scratch array.
    uint16_t capacity;
  } scratch;
};

// Inline singleton encoding shared by publication and query operations.
#define LOOM_VALUE_FACT_RELATION_ROOT_SINGLETON_BIT UINT32_C(0x80000000)

static inline bool loom_value_fact_relation_root_is_singleton(
    loom_value_fact_relation_root_t root) {
  return (root & LOOM_VALUE_FACT_RELATION_ROOT_SINGLETON_BIT) != 0;
}

static inline uint32_t loom_value_fact_relation_singleton_id(
    loom_value_fact_relation_root_t root) {
  IREE_ASSERT(loom_value_fact_relation_root_is_singleton(root));
  return root & ~LOOM_VALUE_FACT_RELATION_ROOT_SINGLETON_BIT;
}

// Finds an alias's sparse entry or its insertion position.
iree_host_size_t loom_value_fact_relation_entry_lower_bound(
    const loom_value_fact_integer_relations_t* relations,
    loom_value_id_t value_id, bool* out_found);

// Hashes a retained counterpart identity or signed literal.
uint32_t loom_value_fact_relation_atom_hash(
    loom_value_fact_relation_atom_kind_t kind, uint64_t bits);

// Finds an interned counterpart without allocating; zero denotes absence.
uint32_t loom_value_fact_relation_atom_find(
    const loom_value_fact_integer_relations_t* relations,
    loom_value_fact_relation_atom_kind_t kind, uint64_t bits);

// Tests membership in an inline singleton or canonical retained set.
bool loom_value_fact_relation_root_contains(
    const loom_value_fact_integer_relations_t* relations,
    loom_value_fact_relation_root_t root, uint32_t atom_id);

// Captures the retained relation sets without enumerating their counterparts.
loom_value_fact_relation_roots_t loom_value_fact_integer_relations_snapshot(
    const loom_value_fact_table_t* table, loom_value_id_t value_id);

// Returns the possible signed comparison outcomes for |lhs| relative to
// |rhs|. Identity, scalar integer facts, retained alias relations, and literal
// exclusions are resolved without walking producers or predicates. Unknown or
// contradictory retained state admits every outcome.
loom_integer_order_outcome_bits_t
loom_value_fact_table_query_signed_integer_order(
    const loom_value_fact_table_t* table, loom_value_id_t lhs,
    loom_value_id_t rhs);

// Returns the possible signed comparison outcomes for |value| relative to
// |literal|. Scalar integer facts and retained alias-local literal exclusions
// are resolved without materializing an SSA constant. Unknown or contradictory
// retained state admits every outcome.
loom_integer_order_outcome_bits_t
loom_value_fact_table_query_signed_integer_literal_order(
    const loom_value_fact_table_t* table, loom_value_id_t value,
    int64_t literal);

// Tightens integer interval endpoints using retained literal exclusions and
// scalar nonzero facts. Work is bounded by the excluded endpoint values, not
// the interval width; interior holes do not change the interval.
void loom_value_fact_table_refine_integer_exclusions(
    const loom_value_fact_table_t* table, loom_value_id_t value,
    loom_value_facts_t* inout_facts);

// Visitor for one constrained retained relation from an alias to a canonical
// counterpart. |outcomes| is the nonempty, non-universal set of possible
// signed comparison outcomes for the alias relative to |counterpart|.
typedef bool (*loom_value_fact_signed_integer_relation_visit_fn_t)(
    void* user_data, loom_value_id_t counterpart,
    loom_integer_order_outcome_bits_t outcomes);

// Visits each constrained signed relation published for |value| once. This
// enumerates the alias's sparse retained sets without allocation or producer
// traversal. Relations published only for a differently scoped alias remain in
// that alias's scope. Returns false when |visit| stops iteration.
bool loom_value_fact_table_for_each_signed_integer_relation_while(
    const loom_value_fact_table_t* table, loom_value_id_t value,
    loom_value_fact_signed_integer_relation_visit_fn_t visit, void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_UTIL_FACT_RELATION_H_
