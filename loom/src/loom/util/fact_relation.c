// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_relation.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/structural_hash.h"
#include "loom/util/fact_ordinal_map.h"

#define LOOM_VALUE_FACT_RELATION_INITIAL_CAPACITY 8
#define LOOM_VALUE_FACT_RELATION_STACK_CAPACITY 16
#define LOOM_VALUE_FACT_RELATION_INDEX_PREDICATE_COUNT 4
#define LOOM_VALUE_FACT_RELATION_ATOM_INITIAL_CAPACITY 8
#define LOOM_VALUE_FACT_RELATION_ATOM_LINEAR_SCAN_LIMIT 8
#define LOOM_VALUE_FACT_RELATION_ROOT_SINGLETON_BIT UINT32_C(0x80000000)

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
  uint64_t bits;
  uint32_t next;
  uint8_t kind;
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

static bool loom_value_fact_relation_roots_equal(
    loom_value_fact_relation_roots_t lhs,
    loom_value_fact_relation_roots_t rhs) {
  if (lhs.excluded_literal_equal != rhs.excluded_literal_equal) {
    return false;
  }
  for (uint8_t i = 0; i < LOOM_INTEGER_ORDER_OUTCOME_COUNT; ++i) {
    if (lhs.excluded[i] != rhs.excluded[i]) {
      return false;
    }
  }
  return true;
}

static bool loom_value_fact_relation_roots_empty(
    loom_value_fact_relation_roots_t roots) {
  return roots.excluded[0] == 0 && roots.excluded[1] == 0 &&
         roots.excluded[2] == 0 && roots.excluded_literal_equal == 0;
}

static iree_host_size_t loom_value_fact_relation_entry_lower_bound(
    const loom_value_fact_integer_relations_t* relations,
    loom_value_id_t value_id, bool* out_found) {
  iree_host_size_t lower = 0;
  iree_host_size_t upper = relations ? relations->count : 0;
  while (lower < upper) {
    const iree_host_size_t middle = lower + (upper - lower) / 2;
    if (relations->entries[middle].value_id < value_id) {
      lower = middle + 1;
    } else {
      upper = middle;
    }
  }
  *out_found = relations && lower < relations->count &&
               relations->entries[lower].value_id == value_id;
  return lower;
}

loom_value_fact_relation_roots_t loom_value_fact_integer_relations_snapshot(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  loom_value_fact_table_record_read(table, value_id);
  if (!table || !table->integer_relations) {
    return (loom_value_fact_relation_roots_t){0};
  }
  bool found = false;
  const iree_host_size_t ordinal = loom_value_fact_relation_entry_lower_bound(
      table->integer_relations, value_id, &found);
  return found ? table->integer_relations->entries[ordinal].roots
               : (loom_value_fact_relation_roots_t){0};
}

static iree_status_t loom_value_fact_integer_relations_ensure_owner(
    loom_value_fact_table_t* table,
    loom_value_fact_integer_relations_t** out_relations) {
  if (!table->integer_relations) {
    loom_value_fact_integer_relations_t* relations = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        table->transient_arena, sizeof(*relations), (void**)&relations));
    memset(relations, 0, sizeof(*relations));
    table->integer_relations = relations;
  }
  *out_relations = table->integer_relations;
  return iree_ok_status();
}

static uint32_t loom_value_fact_relation_atom_hash(
    loom_value_fact_relation_atom_kind_t kind, uint64_t bits) {
  uint32_t hash = loom_structural_hash_mix_u64(
      loom_structural_hash_initialize(), (uint64_t)kind);
  return loom_structural_hash_finalize(
      loom_structural_hash_mix_u64(hash, bits));
}

static bool loom_value_fact_relation_atom_equal(
    const loom_value_fact_relation_atom_t* atom,
    loom_value_fact_relation_atom_kind_t kind, uint64_t bits) {
  return atom->kind == (uint8_t)kind && atom->bits == bits;
}

static uint32_t loom_value_fact_relation_atom_find(
    const loom_value_fact_integer_relations_t* relations,
    loom_value_fact_relation_atom_kind_t kind, uint64_t bits) {
  if (!relations) {
    return 0;
  }
  if (relations->atoms.bucket_count == 0) {
    for (iree_host_size_t i = 0; i < relations->atoms.value_count; ++i) {
      if (loom_value_fact_relation_atom_equal(&relations->atoms.values[i], kind,
                                              bits)) {
        return (uint32_t)i + 1;
      }
    }
    return 0;
  }
  const iree_host_size_t bucket =
      loom_value_fact_relation_atom_hash(kind, bits) &
      (relations->atoms.bucket_count - 1);
  for (uint32_t id = relations->atoms.buckets[bucket]; id != 0;
       id = relations->atoms.values[id - 1].next) {
    if (loom_value_fact_relation_atom_equal(&relations->atoms.values[id - 1],
                                            kind, bits)) {
      return id;
    }
  }
  return 0;
}

static iree_status_t loom_value_fact_relation_atoms_rehash(
    loom_value_fact_table_t* table,
    loom_value_fact_integer_relations_t* relations,
    iree_host_size_t required_count) {
  iree_host_size_t bucket_count = relations->atoms.bucket_count;
  if (bucket_count == 0) {
    bucket_count = 16;
  }
  while (required_count > bucket_count - bucket_count / 4) {
    if (!iree_host_size_checked_mul(bucket_count, 2, &bucket_count)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "relation atom hash capacity overflow");
    }
  }
  if (bucket_count == relations->atoms.bucket_count) {
    return iree_ok_status();
  }
  uint32_t* buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(table->transient_arena,
                                                 bucket_count, sizeof(*buckets),
                                                 (void**)&buckets));
  memset(buckets, 0, bucket_count * sizeof(*buckets));
  for (iree_host_size_t i = 0; i < relations->atoms.value_count; ++i) {
    loom_value_fact_relation_atom_t* atom = &relations->atoms.values[i];
    const iree_host_size_t bucket =
        loom_value_fact_relation_atom_hash(
            (loom_value_fact_relation_atom_kind_t)atom->kind, atom->bits) &
        (bucket_count - 1);
    atom->next = buckets[bucket];
    buckets[bucket] = (uint32_t)i + 1;
  }
  relations->atoms.buckets = buckets;
  relations->atoms.bucket_count = bucket_count;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_relation_atom_intern(
    loom_value_fact_table_t* table, loom_value_fact_relation_atom_kind_t kind,
    uint64_t bits, uint32_t* out_id) {
  loom_value_fact_integer_relations_t* relations = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_integer_relations_ensure_owner(table, &relations));
  const uint32_t existing =
      loom_value_fact_relation_atom_find(relations, kind, bits);
  if (existing != 0) {
    *out_id = existing;
    return iree_ok_status();
  }
  if (relations->atoms.value_count >= INT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "relation atom ID capacity exceeded");
  }
  if (relations->atoms.value_count == relations->atoms.value_capacity) {
    iree_host_size_t new_capacity =
        LOOM_VALUE_FACT_RELATION_ATOM_INITIAL_CAPACITY;
    if (relations->atoms.value_capacity &&
        !iree_host_size_checked_mul(relations->atoms.value_capacity, 2,
                                    &new_capacity)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "relation atom capacity overflow");
    }
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        table->transient_arena, relations->atoms.value_count, new_capacity,
        sizeof(*relations->atoms.values), &relations->atoms.value_capacity,
        (void**)&relations->atoms.values));
  }
  const iree_host_size_t required_count = relations->atoms.value_count + 1;
  if (required_count > LOOM_VALUE_FACT_RELATION_ATOM_LINEAR_SCAN_LIMIT) {
    IREE_RETURN_IF_ERROR(loom_value_fact_relation_atoms_rehash(table, relations,
                                                               required_count));
  }
  const uint32_t id = (uint32_t)required_count;
  uint32_t next = 0;
  if (relations->atoms.bucket_count != 0) {
    const iree_host_size_t bucket =
        loom_value_fact_relation_atom_hash(kind, bits) &
        (relations->atoms.bucket_count - 1);
    next = relations->atoms.buckets[bucket];
    relations->atoms.buckets[bucket] = id;
  }
  relations->atoms.values[id - 1] = (loom_value_fact_relation_atom_t){
      .bits = bits,
      .next = next,
      .kind = (uint8_t)kind,
      .reserved = {},
  };
  relations->atoms.value_count = required_count;
  *out_id = id;
  return iree_ok_status();
}

static bool loom_value_fact_relation_root_is_singleton(
    loom_value_fact_relation_root_t root) {
  return (root & LOOM_VALUE_FACT_RELATION_ROOT_SINGLETON_BIT) != 0;
}

static loom_value_fact_relation_root_t loom_value_fact_relation_singleton(
    uint32_t atom_id) {
  IREE_ASSERT_GT(atom_id, 0u);
  IREE_ASSERT_LT(atom_id, LOOM_VALUE_FACT_RELATION_ROOT_SINGLETON_BIT);
  return LOOM_VALUE_FACT_RELATION_ROOT_SINGLETON_BIT | atom_id;
}

static uint32_t loom_value_fact_relation_singleton_id(
    loom_value_fact_relation_root_t root) {
  IREE_ASSERT(loom_value_fact_relation_root_is_singleton(root));
  return root & ~LOOM_VALUE_FACT_RELATION_ROOT_SINGLETON_BIT;
}

static iree_status_t loom_value_fact_integer_relations_ensure_index(
    loom_value_fact_table_t* table,
    loom_value_fact_integer_relations_t* relations) {
  if (!relations->index) {
    IREE_RETURN_IF_ERROR(loom_value_set_index_allocate(table->transient_arena,
                                                       &relations->index));
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_relation_root_materialize(
    loom_value_fact_table_t* table,
    loom_value_fact_integer_relations_t* relations,
    loom_value_fact_relation_root_t root, loom_value_set_id_t* out_set) {
  if (!root) {
    *out_set = 0;
    return iree_ok_status();
  }
  if (!loom_value_fact_relation_root_is_singleton(root)) {
    *out_set = root;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_value_fact_integer_relations_ensure_index(table, relations));
  return loom_value_set_index_add(
      relations->index, 0,
      (loom_value_id_t)loom_value_fact_relation_singleton_id(root), out_set);
}

static iree_status_t loom_value_fact_relation_root_add(
    loom_value_fact_table_t* table, loom_value_fact_relation_root_t source_root,
    uint32_t atom_id, loom_value_fact_relation_root_t* out_root) {
  if (!source_root) {
    *out_root = loom_value_fact_relation_singleton(atom_id);
    return iree_ok_status();
  }
  if (loom_value_fact_relation_root_is_singleton(source_root) &&
      loom_value_fact_relation_singleton_id(source_root) == atom_id) {
    *out_root = source_root;
    return iree_ok_status();
  }
  loom_value_fact_integer_relations_t* relations = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_integer_relations_ensure_owner(table, &relations));
  loom_value_set_id_t source_set = 0;
  IREE_RETURN_IF_ERROR(loom_value_fact_relation_root_materialize(
      table, relations, source_root, &source_set));
  loom_value_set_id_t result_set = 0;
  IREE_RETURN_IF_ERROR(loom_value_set_index_add(
      relations->index, source_set, (loom_value_id_t)atom_id, &result_set));
  IREE_ASSERT_FALSE(loom_value_fact_relation_root_is_singleton(result_set));
  *out_root = result_set;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_relation_root_union(
    loom_value_fact_table_t* table, loom_value_fact_relation_root_t lhs,
    loom_value_fact_relation_root_t rhs,
    loom_value_fact_relation_root_t* out_root) {
  if (!lhs || lhs == rhs) {
    *out_root = rhs;
    return iree_ok_status();
  }
  if (!rhs) {
    *out_root = lhs;
    return iree_ok_status();
  }
  loom_value_fact_integer_relations_t* relations = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_integer_relations_ensure_owner(table, &relations));
  loom_value_set_id_t lhs_set = 0;
  loom_value_set_id_t rhs_set = 0;
  IREE_RETURN_IF_ERROR(loom_value_fact_relation_root_materialize(
      table, relations, lhs, &lhs_set));
  IREE_RETURN_IF_ERROR(loom_value_fact_relation_root_materialize(
      table, relations, rhs, &rhs_set));
  loom_value_set_id_t result_set = 0;
  IREE_RETURN_IF_ERROR(loom_value_set_index_union(relations->index, lhs_set,
                                                  rhs_set, &result_set));
  IREE_ASSERT_FALSE(loom_value_fact_relation_root_is_singleton(result_set));
  *out_root = result_set;
  return iree_ok_status();
}

static bool loom_value_fact_relation_root_contains(
    const loom_value_fact_integer_relations_t* relations,
    loom_value_fact_relation_root_t root, uint32_t atom_id) {
  if (!root || !atom_id) {
    return false;
  }
  if (loom_value_fact_relation_root_is_singleton(root)) {
    return loom_value_fact_relation_singleton_id(root) == atom_id;
  }
  IREE_ASSERT(relations->index);
  return loom_value_set_index_contains(relations->index, root,
                                       (loom_value_id_t)atom_id);
}

static iree_status_t loom_value_fact_integer_relations_reserve_entries(
    loom_value_fact_table_t* table, iree_host_size_t required_capacity) {
  loom_value_fact_integer_relations_t* relations = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_integer_relations_ensure_owner(table, &relations));
  if (required_capacity <= relations->capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity = LOOM_VALUE_FACT_RELATION_INITIAL_CAPACITY;
  if (relations->capacity &&
      !iree_host_size_checked_mul(relations->capacity, 2, &new_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "relation entry capacity overflow");
  }
  new_capacity = iree_max(new_capacity, required_capacity);
  return iree_arena_grow_array(table->transient_arena, relations->count,
                               new_capacity, sizeof(*relations->entries),
                               &relations->capacity,
                               (void**)&relations->entries);
}

// Publishes into pre-reserved storage and returns whether roots changed.
static bool loom_value_fact_integer_relations_publish_reserved(
    loom_value_fact_integer_relations_t* relations, loom_value_id_t value_id,
    loom_value_fact_relation_roots_t roots) {
  bool found = false;
  const iree_host_size_t ordinal =
      loom_value_fact_relation_entry_lower_bound(relations, value_id, &found);
  if (found) {
    loom_value_fact_relation_entry_t* entry = &relations->entries[ordinal];
    if (loom_value_fact_relation_roots_equal(entry->roots, roots)) {
      return false;
    }
    entry->roots = roots;
    return true;
  }
  if (loom_value_fact_relation_roots_empty(roots)) {
    return false;
  }
  IREE_ASSERT_LT(relations->count, relations->capacity);
  memmove(&relations->entries[ordinal + 1], &relations->entries[ordinal],
          (relations->count - ordinal) * sizeof(*relations->entries));
  relations->entries[ordinal] = (loom_value_fact_relation_entry_t){
      .value_id = value_id,
      .roots = roots,
  };
  ++relations->count;
  return true;
}

static iree_status_t loom_value_fact_integer_relations_publish_one(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_relation_roots_t roots, bool* out_changed) {
  *out_changed = false;
  bool found = false;
  if (table->integer_relations) {
    (void)loom_value_fact_relation_entry_lower_bound(table->integer_relations,
                                                     value_id, &found);
  }
  if (!found && loom_value_fact_relation_roots_empty(roots)) {
    return iree_ok_status();
  }
  const iree_host_size_t required_capacity =
      table->integer_relations ? table->integer_relations->count + !found : 1;
  IREE_RETURN_IF_ERROR(loom_value_fact_integer_relations_reserve_entries(
      table, required_capacity));
  *out_changed = loom_value_fact_integer_relations_publish_reserved(
      table->integer_relations, value_id, roots);
  return iree_ok_status();
}

static iree_status_t loom_value_fact_relation_add_exclusions(
    loom_value_fact_table_t* table,
    loom_value_fact_relation_roots_t* inout_roots,
    loom_integer_order_outcome_bits_t allowed_outcomes,
    loom_value_id_t counterpart_identity) {
  const loom_integer_order_outcome_bits_t excluded_outcomes =
      LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL & ~allowed_outcomes;
  if (!excluded_outcomes) {
    return iree_ok_status();
  }
  uint32_t atom_id = 0;
  IREE_RETURN_IF_ERROR(loom_value_fact_relation_atom_intern(
      table, LOOM_VALUE_FACT_RELATION_ATOM_VALUE,
      (uint64_t)counterpart_identity, &atom_id));
  for (uint8_t outcome = 0; outcome < LOOM_INTEGER_ORDER_OUTCOME_COUNT;
       ++outcome) {
    if (!(excluded_outcomes & (1u << outcome))) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_relation_root_add(
        table, inout_roots->excluded[outcome], atom_id,
        &inout_roots->excluded[outcome]));
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_relation_union_roots(
    loom_value_fact_table_t* table, loom_value_fact_relation_roots_t source,
    loom_value_fact_relation_roots_t* inout_target) {
  if (loom_value_fact_relation_roots_empty(source)) {
    return iree_ok_status();
  }
  for (uint8_t i = 0; i < LOOM_INTEGER_ORDER_OUTCOME_COUNT; ++i) {
    IREE_RETURN_IF_ERROR(loom_value_fact_relation_root_union(
        table, inout_target->excluded[i], source.excluded[i],
        &inout_target->excluded[i]));
  }
  return loom_value_fact_relation_root_union(
      table, inout_target->excluded_literal_equal,
      source.excluded_literal_equal, &inout_target->excluded_literal_equal);
}

static bool loom_value_fact_integer_facts_exclude(loom_value_facts_t facts,
                                                  int64_t value) {
  if (value < facts.range_lo || value > facts.range_hi) {
    return true;
  }
  if (value == 0 && loom_value_facts_is_non_zero(facts)) {
    return true;
  }
  return facts.known_divisor > 1 && value % facts.known_divisor != 0;
}

static loom_integer_order_outcome_bits_t
loom_value_fact_integer_facts_order_outcomes(loom_value_facts_t lhs,
                                             loom_value_facts_t rhs) {
  if (loom_value_facts_is_float(lhs) || loom_value_facts_is_float(rhs)) {
    return LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL;
  }
  loom_integer_order_outcome_bits_t outcomes =
      LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL;
  if (lhs.range_lo >= rhs.range_hi) {
    outcomes &= ~LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS;
  }
  if (lhs.range_hi <= rhs.range_lo) {
    outcomes &= ~LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER;
  }
  if (lhs.range_hi < rhs.range_lo || rhs.range_hi < lhs.range_lo) {
    outcomes &= ~LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL;
  }
  int64_t exact_value = 0;
  if ((loom_value_facts_as_exact_i64(rhs, &exact_value) &&
       loom_value_fact_integer_facts_exclude(lhs, exact_value)) ||
      (loom_value_facts_as_exact_i64(lhs, &exact_value) &&
       loom_value_fact_integer_facts_exclude(rhs, exact_value))) {
    outcomes &= ~LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL;
  }
  return outcomes;
}

loom_integer_order_outcome_bits_t
loom_value_fact_table_query_signed_integer_order(
    const loom_value_fact_table_t* table, loom_value_id_t lhs,
    loom_value_id_t rhs) {
  if (lhs == LOOM_VALUE_ID_INVALID || rhs == LOOM_VALUE_ID_INVALID) {
    return LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL;
  }
  if (lhs == rhs) {
    return LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL;
  }
  if (!table) {
    return LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL;
  }

  const loom_value_id_t lhs_identity =
      loom_value_fact_table_query_identity(table, lhs);
  const loom_value_id_t rhs_identity =
      loom_value_fact_table_query_identity(table, rhs);
  if (lhs_identity == rhs_identity) {
    return LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL;
  }

  const loom_value_facts_t lhs_facts = loom_value_fact_table_lookup(table, lhs);
  const loom_value_facts_t rhs_facts = loom_value_fact_table_lookup(table, rhs);
  if (loom_value_facts_is_exact(lhs_facts) &&
      loom_value_facts_is_exact(rhs_facts) &&
      !loom_value_facts_is_float(lhs_facts) &&
      !loom_value_facts_is_float(rhs_facts)) {
    if (lhs_facts.range_lo < rhs_facts.range_lo) {
      return LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS;
    }
    if (lhs_facts.range_lo > rhs_facts.range_lo) {
      return LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER;
    }
    return LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL;
  }

  loom_integer_order_outcome_bits_t outcomes =
      loom_value_fact_integer_facts_order_outcomes(lhs_facts, rhs_facts);
  if (!table->integer_relations) {
    return outcomes ? outcomes : LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL;
  }
  const loom_value_fact_relation_roots_t lhs_roots =
      loom_value_fact_integer_relations_snapshot(table, lhs);
  const loom_value_fact_relation_roots_t rhs_roots =
      loom_value_fact_integer_relations_snapshot(table, rhs);
  const uint32_t lhs_identity_atom = loom_value_fact_relation_atom_find(
      table->integer_relations, LOOM_VALUE_FACT_RELATION_ATOM_VALUE,
      (uint64_t)lhs_identity);
  const uint32_t rhs_identity_atom = loom_value_fact_relation_atom_find(
      table->integer_relations, LOOM_VALUE_FACT_RELATION_ATOM_VALUE,
      (uint64_t)rhs_identity);
  for (uint8_t outcome = 0; outcome < LOOM_INTEGER_ORDER_OUTCOME_COUNT;
       ++outcome) {
    if (loom_value_fact_relation_root_contains(table->integer_relations,
                                               lhs_roots.excluded[outcome],
                                               rhs_identity_atom)) {
      outcomes &= ~(1u << outcome);
    }
    const loom_integer_order_outcome_t swapped =
        loom_integer_order_outcome_swap(outcome);
    if (loom_value_fact_relation_root_contains(table->integer_relations,
                                               rhs_roots.excluded[swapped],
                                               lhs_identity_atom)) {
      outcomes &= ~(1u << outcome);
    }
  }

  int64_t exact_value = 0;
  if (loom_value_facts_as_exact_i64(rhs_facts, &exact_value)) {
    const uint32_t literal_atom = loom_value_fact_relation_atom_find(
        table->integer_relations, LOOM_VALUE_FACT_RELATION_ATOM_LITERAL,
        (uint64_t)exact_value);
    if (loom_value_fact_relation_root_contains(table->integer_relations,
                                               lhs_roots.excluded_literal_equal,
                                               literal_atom)) {
      outcomes &= ~LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL;
    }
  }
  if (loom_value_facts_as_exact_i64(lhs_facts, &exact_value)) {
    const uint32_t literal_atom = loom_value_fact_relation_atom_find(
        table->integer_relations, LOOM_VALUE_FACT_RELATION_ATOM_LITERAL,
        (uint64_t)exact_value);
    if (loom_value_fact_relation_root_contains(table->integer_relations,
                                               rhs_roots.excluded_literal_equal,
                                               literal_atom)) {
      outcomes &= ~LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL;
    }
  }
  return outcomes ? outcomes : LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL;
}

loom_integer_order_outcome_bits_t
loom_value_fact_table_query_signed_integer_literal_order(
    const loom_value_fact_table_t* table, loom_value_id_t value,
    int64_t literal) {
  if (!table || value == LOOM_VALUE_ID_INVALID) {
    return LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL;
  }

  const loom_value_facts_t value_facts =
      loom_value_fact_table_lookup(table, value);
  const loom_value_facts_t literal_facts = loom_value_facts_exact_i64(literal);
  loom_integer_order_outcome_bits_t outcomes =
      loom_value_fact_integer_facts_order_outcomes(value_facts, literal_facts);
  if (!table->integer_relations) {
    return outcomes ? outcomes : LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL;
  }

  const loom_value_fact_relation_roots_t roots =
      loom_value_fact_integer_relations_snapshot(table, value);
  const uint32_t literal_atom = loom_value_fact_relation_atom_find(
      table->integer_relations, LOOM_VALUE_FACT_RELATION_ATOM_LITERAL,
      (uint64_t)literal);
  if (loom_value_fact_relation_root_contains(table->integer_relations,
                                             roots.excluded_literal_equal,
                                             literal_atom)) {
    outcomes &= ~LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL;
  }
  return outcomes ? outcomes : LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL;
}

typedef struct loom_value_fact_relation_root_cursor_t {
  loom_value_fact_relation_root_t root;
  loom_value_set_cursor_t set_cursor;
  uint32_t atom_id;
} loom_value_fact_relation_root_cursor_t;

static void loom_value_fact_relation_root_cursor_initialize(
    const loom_value_fact_integer_relations_t* relations,
    loom_value_fact_relation_root_t root,
    loom_value_fact_relation_root_cursor_t* out_cursor) {
  *out_cursor = (loom_value_fact_relation_root_cursor_t){
      .root = root,
      .atom_id = 0,
  };
  if (!root) {
    return;
  }
  if (loom_value_fact_relation_root_is_singleton(root)) {
    out_cursor->atom_id = loom_value_fact_relation_singleton_id(root);
    return;
  }
  loom_value_set_cursor_begin(relations->index, root, &out_cursor->set_cursor);
  const loom_value_id_t atom_id =
      loom_value_set_cursor_next(&out_cursor->set_cursor);
  out_cursor->atom_id = atom_id == LOOM_VALUE_ID_INVALID ? 0 : atom_id;
}

static void loom_value_fact_relation_root_cursor_advance(
    loom_value_fact_relation_root_cursor_t* cursor) {
  if (loom_value_fact_relation_root_is_singleton(cursor->root)) {
    cursor->atom_id = 0;
    return;
  }
  const loom_value_id_t atom_id =
      loom_value_set_cursor_next(&cursor->set_cursor);
  cursor->atom_id = atom_id == LOOM_VALUE_ID_INVALID ? 0 : atom_id;
}

bool loom_value_fact_table_for_each_signed_integer_relation_while(
    const loom_value_fact_table_t* table, loom_value_id_t value,
    loom_value_fact_signed_integer_relation_visit_fn_t visit, void* user_data) {
  loom_value_fact_table_record_read(table, value);
  if (!table || !table->integer_relations) {
    return true;
  }
  const loom_value_fact_relation_roots_t roots =
      loom_value_fact_integer_relations_snapshot(table, value);
  loom_value_fact_relation_root_cursor_t
      cursors[LOOM_INTEGER_ORDER_OUTCOME_COUNT];
  for (loom_integer_order_outcome_t outcome = 0;
       outcome < LOOM_INTEGER_ORDER_OUTCOME_COUNT; ++outcome) {
    loom_value_fact_relation_root_cursor_initialize(
        table->integer_relations, roots.excluded[outcome], &cursors[outcome]);
  }

  while (true) {
    uint32_t next_atom_id = 0;
    for (loom_integer_order_outcome_t outcome = 0;
         outcome < LOOM_INTEGER_ORDER_OUTCOME_COUNT; ++outcome) {
      const uint32_t atom_id = cursors[outcome].atom_id;
      if (atom_id && (!next_atom_id || atom_id < next_atom_id)) {
        next_atom_id = atom_id;
      }
    }
    if (!next_atom_id) {
      return true;
    }
    for (loom_integer_order_outcome_t outcome = 0;
         outcome < LOOM_INTEGER_ORDER_OUTCOME_COUNT; ++outcome) {
      if (cursors[outcome].atom_id == next_atom_id) {
        loom_value_fact_relation_root_cursor_advance(&cursors[outcome]);
      }
    }

    IREE_ASSERT_LE(next_atom_id, table->integer_relations->atoms.value_count);
    const loom_value_fact_relation_atom_t* atom =
        &table->integer_relations->atoms.values[next_atom_id - 1];
    IREE_ASSERT_EQ(atom->kind, LOOM_VALUE_FACT_RELATION_ATOM_VALUE);
    const loom_value_id_t counterpart = (loom_value_id_t)atom->bits;
    const loom_integer_order_outcome_bits_t outcomes =
        loom_value_fact_table_query_signed_integer_order(table, value,
                                                         counterpart);
    if (outcomes != LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL &&
        !visit(user_data, counterpart, outcomes)) {
      return false;
    }
  }
}

void loom_value_fact_integer_relations_clear_scope(
    loom_value_fact_table_t* table) {
  table->integer_relations = NULL;
}

void loom_value_fact_integer_relations_undefine(loom_value_fact_table_t* table,
                                                loom_value_id_t value_id) {
  if (!table->integer_relations) {
    return;
  }
  bool found = false;
  const iree_host_size_t ordinal = loom_value_fact_relation_entry_lower_bound(
      table->integer_relations, value_id, &found);
  if (found) {
    table->integer_relations->entries[ordinal].roots =
        (loom_value_fact_relation_roots_t){0};
  }
}

static iree_status_t loom_value_fact_relation_clone_root(
    loom_value_fact_table_t* target,
    const loom_value_fact_integer_relations_t* source_relations,
    loom_value_fact_relation_root_t source_root,
    loom_value_fact_relation_root_t* out_root) {
  *out_root = 0;
  if (!source_root) {
    return iree_ok_status();
  }
  loom_value_set_cursor_t cursor;
  loom_value_id_t source_atom_id = LOOM_VALUE_ID_INVALID;
  if (loom_value_fact_relation_root_is_singleton(source_root)) {
    source_atom_id = loom_value_fact_relation_singleton_id(source_root);
  } else {
    loom_value_set_cursor_begin(source_relations->index, source_root, &cursor);
    source_atom_id = loom_value_set_cursor_next(&cursor);
  }
  while (source_atom_id != LOOM_VALUE_ID_INVALID) {
    IREE_ASSERT_LE(source_atom_id, source_relations->atoms.value_count);
    const loom_value_fact_relation_atom_t* source_atom =
        &source_relations->atoms.values[source_atom_id - 1];
    uint32_t target_atom_id = 0;
    IREE_RETURN_IF_ERROR(loom_value_fact_relation_atom_intern(
        target, (loom_value_fact_relation_atom_kind_t)source_atom->kind,
        source_atom->bits, &target_atom_id));
    IREE_RETURN_IF_ERROR(loom_value_fact_relation_root_add(
        target, *out_root, target_atom_id, out_root));
    source_atom_id = loom_value_fact_relation_root_is_singleton(source_root)
                         ? LOOM_VALUE_ID_INVALID
                         : loom_value_set_cursor_next(&cursor);
  }
  return iree_ok_status();
}

iree_status_t loom_value_fact_integer_relations_clone_value(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    loom_value_id_t value_id) {
  if (target == source) {
    return iree_ok_status();
  }
  const loom_value_fact_relation_roots_t source_roots =
      loom_value_fact_integer_relations_snapshot(source, value_id);
  loom_value_fact_relation_roots_t target_roots = {0};
  for (uint8_t outcome = 0; outcome < LOOM_INTEGER_ORDER_OUTCOME_COUNT;
       ++outcome) {
    IREE_RETURN_IF_ERROR(loom_value_fact_relation_clone_root(
        target, source->integer_relations, source_roots.excluded[outcome],
        &target_roots.excluded[outcome]));
  }
  IREE_RETURN_IF_ERROR(loom_value_fact_relation_clone_root(
      target, source->integer_relations, source_roots.excluded_literal_equal,
      &target_roots.excluded_literal_equal));

  bool changed = false;
  return loom_value_fact_integer_relations_publish_one(target, value_id,
                                                       target_roots, &changed);
}

static iree_status_t loom_value_fact_integer_relations_ensure_scratch(
    loom_value_fact_table_t* table, uint16_t count,
    loom_value_id_t** out_identities,
    loom_value_fact_relation_roots_t** out_plans,
    loom_value_fact_relation_roots_t** out_locals) {
  loom_value_fact_integer_relations_t* relations = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_integer_relations_ensure_owner(table, &relations));
  if (count > relations->scratch.capacity) {
    loom_value_id_t* identities = NULL;
    loom_value_fact_relation_roots_t* plans = NULL;
    loom_value_fact_relation_roots_t* locals = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(table->transient_arena,
                                                   count, sizeof(*identities),
                                                   (void**)&identities));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        table->transient_arena, count, sizeof(*plans), (void**)&plans));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        table->transient_arena, count, sizeof(*locals), (void**)&locals));
    relations->scratch.identities = identities;
    relations->scratch.plans = plans;
    relations->scratch.locals = locals;
    relations->scratch.capacity = count;
  }
  *out_identities = relations->scratch.identities;
  *out_plans = relations->scratch.plans;
  *out_locals = relations->scratch.locals;
  return iree_ok_status();
}

typedef struct loom_value_fact_relation_plan_t {
  // Incoming canonical identities, borrowed from caller or table scratch.
  loom_value_id_t* identities;
  // Planned relation roots, using scratch parallel to identities.
  loom_value_fact_relation_roots_t* plans;
  // Newly inferred local roots, using scratch parallel to identities.
  loom_value_fact_relation_roots_t* locals;
  // Ordinal lookup borrowing identities and leasing the table's map scratch.
  loom_value_fact_ordinal_map_t identity_map;
  // Whether lazy initialization succeeded before relation publication.
  bool initialized;
} loom_value_fact_relation_plan_t;

static iree_status_t loom_value_fact_relation_plan_initialize(
    loom_value_fact_table_t* table, const loom_value_id_t* operands,
    uint16_t pair_count, iree_host_size_t predicate_count, bool use_stack,
    loom_value_id_t* stack_identities,
    loom_value_fact_relation_roots_t* stack_plans,
    loom_value_fact_relation_roots_t* stack_locals,
    loom_value_fact_relation_plan_t* plan) {
  if (use_stack) {
    plan->identities = stack_identities;
    plan->plans = stack_plans;
    plan->locals = stack_locals;
  } else {
    IREE_RETURN_IF_ERROR(loom_value_fact_integer_relations_ensure_scratch(
        table, pair_count, &plan->identities, &plan->plans, &plan->locals));
  }
  for (uint16_t i = 0; i < pair_count; ++i) {
    plan->identities[i] =
        loom_value_fact_table_query_identity(table, operands[i]);
    plan->plans[i] =
        loom_value_fact_integer_relations_snapshot(table, operands[i]);
    plan->locals[i] = (loom_value_fact_relation_roots_t){0};
  }
  const bool use_index =
      pair_count > LOOM_VALUE_FACT_RELATION_STACK_CAPACITY &&
      predicate_count > LOOM_VALUE_FACT_RELATION_INDEX_PREDICATE_COUNT;
  IREE_RETURN_IF_ERROR(loom_value_fact_ordinal_map_initialize(
      table, plan->identities, pair_count, use_index, &plan->identity_map));
  plan->initialized = true;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_relation_plan_add(
    loom_value_fact_table_t* table, loom_value_fact_relation_plan_t* plan,
    loom_value_id_t lhs, loom_symbolic_integer_relation_t relation,
    loom_value_id_t rhs) {
  const loom_value_id_t lhs_identity =
      loom_value_fact_table_query_identity(table, lhs);
  const loom_value_id_t rhs_identity =
      loom_value_fact_table_query_identity(table, rhs);
  uint16_t ordinal = 0;
  if (loom_value_fact_ordinal_map_find(&plan->identity_map, lhs_identity,
                                       &ordinal)) {
    IREE_RETURN_IF_ERROR(loom_value_fact_relation_add_exclusions(
        table, &plan->locals[ordinal],
        loom_symbolic_integer_relation_outcomes(relation), rhs_identity));
  }
  if (loom_value_fact_ordinal_map_find(&plan->identity_map, rhs_identity,
                                       &ordinal)) {
    IREE_RETURN_IF_ERROR(loom_value_fact_relation_add_exclusions(
        table, &plan->locals[ordinal],
        loom_symbolic_integer_relation_outcomes(
            loom_symbolic_integer_relation_swap(relation)),
        lhs_identity));
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_relation_plan_add_literal_exclusion(
    loom_value_fact_table_t* table, loom_value_fact_relation_plan_t* plan,
    loom_value_id_t value, int64_t excluded_literal) {
  const loom_value_id_t identity =
      loom_value_fact_table_query_identity(table, value);
  uint16_t ordinal = 0;
  if (!loom_value_fact_ordinal_map_find(&plan->identity_map, identity,
                                        &ordinal)) {
    return iree_ok_status();
  }
  uint32_t literal_atom = 0;
  IREE_RETURN_IF_ERROR(loom_value_fact_relation_atom_intern(
      table, LOOM_VALUE_FACT_RELATION_ATOM_LITERAL, (uint64_t)excluded_literal,
      &literal_atom));
  return loom_value_fact_relation_root_add(
      table, plan->locals[ordinal].excluded_literal_equal, literal_atom,
      &plan->locals[ordinal].excluded_literal_equal);
}

static bool loom_value_fact_relation_value_is_integer(
    const loom_module_t* module, int64_t value) {
  if (value < 0) {
    return false;
  }
  const loom_type_t type =
      loom_module_value_type(module, (loom_value_id_t)value);
  return !loom_scalar_type_is_float(loom_type_element_type(type));
}

static bool loom_value_fact_relation_predicate_is_supported(
    const loom_module_t* module, const loom_predicate_t* predicate) {
  if (predicate->kind == LOOM_PREDICATE_RANGE) {
    return predicate->arg_count == 3 &&
           predicate->arg_tags[0] == LOOM_PRED_ARG_VALUE &&
           loom_value_fact_relation_value_is_integer(module,
                                                     predicate->args[0]) &&
           ((predicate->arg_tags[1] == LOOM_PRED_ARG_VALUE &&
             loom_value_fact_relation_value_is_integer(module,
                                                       predicate->args[1])) ||
            (predicate->arg_tags[2] == LOOM_PRED_ARG_VALUE &&
             loom_value_fact_relation_value_is_integer(module,
                                                       predicate->args[2])));
  }
  if (predicate->kind == LOOM_PREDICATE_NE && predicate->arg_count == 2 &&
      predicate->arg_tags[0] == LOOM_PRED_ARG_VALUE &&
      predicate->arg_tags[1] == LOOM_PRED_ARG_CONST &&
      predicate->args[1] != 0) {
    return loom_value_fact_relation_value_is_integer(module,
                                                     predicate->args[0]);
  }
  loom_symbolic_integer_relation_t relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
  return predicate->arg_count == 2 &&
         predicate->arg_tags[0] == LOOM_PRED_ARG_VALUE &&
         predicate->arg_tags[1] == LOOM_PRED_ARG_VALUE &&
         loom_value_fact_relation_value_is_integer(module,
                                                   predicate->args[0]) &&
         loom_value_fact_relation_value_is_integer(module,
                                                   predicate->args[1]) &&
         loom_symbolic_integer_relation_from_predicate_kind(predicate->kind,
                                                            &relation);
}

static iree_status_t loom_value_fact_relation_plan_add_predicate(
    loom_value_fact_table_t* table, loom_value_fact_relation_plan_t* plan,
    const loom_predicate_t* predicate) {
  if (predicate->kind == LOOM_PREDICATE_RANGE) {
    const loom_value_id_t value = (loom_value_id_t)predicate->args[0];
    if (predicate->arg_tags[1] == LOOM_PRED_ARG_VALUE &&
        predicate->args[1] >= 0) {
      IREE_RETURN_IF_ERROR(loom_value_fact_relation_plan_add(
          table, plan, (loom_value_id_t)predicate->args[1],
          LOOM_SYMBOLIC_INTEGER_RELATION_LE, value));
    }
    if (predicate->arg_tags[2] == LOOM_PRED_ARG_VALUE &&
        predicate->args[2] >= 0) {
      IREE_RETURN_IF_ERROR(loom_value_fact_relation_plan_add(
          table, plan, value, LOOM_SYMBOLIC_INTEGER_RELATION_LE,
          (loom_value_id_t)predicate->args[2]));
    }
    return iree_ok_status();
  }
  if (predicate->kind == LOOM_PREDICATE_NE &&
      predicate->arg_tags[1] == LOOM_PRED_ARG_CONST) {
    return loom_value_fact_relation_plan_add_literal_exclusion(
        table, plan, (loom_value_id_t)predicate->args[0], predicate->args[1]);
  }
  loom_symbolic_integer_relation_t relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
  const bool converted = loom_symbolic_integer_relation_from_predicate_kind(
      predicate->kind, &relation);
  IREE_ASSERT(converted);
  return loom_value_fact_relation_plan_add(
      table, plan, (loom_value_id_t)predicate->args[0], relation,
      (loom_value_id_t)predicate->args[1]);
}

static bool loom_value_fact_integer_relations_identity_needs_plan(
    const loom_value_fact_table_t* table, const loom_value_id_t* operands,
    const loom_value_id_t* results, uint16_t pair_count) {
  if (!table->integer_relations) {
    return false;
  }
  for (uint16_t i = 0; i < pair_count; ++i) {
    if (!loom_value_fact_relation_roots_empty(
            loom_value_fact_integer_relations_snapshot(table, operands[i])) ||
        !loom_value_fact_relation_roots_empty(
            loom_value_fact_integer_relations_snapshot(table, results[i]))) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_value_fact_integer_relations_publish_plan(
    loom_value_fact_table_t* table, const loom_value_id_t* results,
    uint16_t pair_count, loom_value_fact_relation_plan_t* plan,
    bool* inout_changed) {
  if (!table->integer_relations) {
    return iree_ok_status();
  }
  iree_host_size_t missing_count = 0;
  for (uint16_t i = 0; i < pair_count; ++i) {
    uint16_t group_ordinal = 0;
    const bool found_group = loom_value_fact_ordinal_map_find(
        &plan->identity_map, plan->identities[i], &group_ordinal);
    IREE_ASSERT(found_group);
    IREE_RETURN_IF_ERROR(loom_value_fact_relation_union_roots(
        table, plan->locals[group_ordinal], &plan->plans[i]));

    bool found_entry = false;
    (void)loom_value_fact_relation_entry_lower_bound(table->integer_relations,
                                                     results[i], &found_entry);
    if (!found_entry && !loom_value_fact_relation_roots_empty(plan->plans[i])) {
      ++missing_count;
    }
  }
  IREE_RETURN_IF_ERROR(loom_value_fact_integer_relations_reserve_entries(
      table, table->integer_relations->count + missing_count));
  for (uint16_t i = 0; i < pair_count; ++i) {
    if (loom_value_fact_integer_relations_publish_reserved(
            table->integer_relations, results[i], plan->plans[i]) &&
        inout_changed) {
      *inout_changed = true;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_integer_relations_propagate_identity(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, bool* inout_changed) {
  const uint16_t pair_count = iree_min(op->operand_count, op->result_count);
  if (pair_count == 0) {
    return iree_ok_status();
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_value_id_t* results = loom_op_const_results(op);

  iree_host_size_t predicate_count = 0;
  const loom_attribute_t* attributes = loom_op_const_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    if (attributes[i].kind == LOOM_ATTR_PREDICATE_LIST) {
      predicate_count += attributes[i].count;
    }
  }

  loom_value_id_t stack_identities[LOOM_VALUE_FACT_RELATION_STACK_CAPACITY];
  loom_value_fact_relation_roots_t
      stack_plans[LOOM_VALUE_FACT_RELATION_STACK_CAPACITY];
  loom_value_fact_relation_roots_t
      stack_locals[LOOM_VALUE_FACT_RELATION_STACK_CAPACITY];
  loom_value_fact_relation_plan_t plan = {0};
  const bool use_stack = pair_count <= LOOM_VALUE_FACT_RELATION_STACK_CAPACITY;
  iree_status_t status = iree_ok_status();
  if (loom_value_fact_integer_relations_identity_needs_plan(
          table, operands, results, pair_count)) {
    status = loom_value_fact_relation_plan_initialize(
        table, operands, pair_count, predicate_count, use_stack,
        stack_identities, stack_plans, stack_locals, &plan);
  }

  for (uint8_t i = 0; i < op->attribute_count && iree_status_is_ok(status);
       ++i) {
    if (attributes[i].kind != LOOM_ATTR_PREDICATE_LIST) {
      continue;
    }
    for (uint16_t j = 0; j < attributes[i].count && iree_status_is_ok(status);
         ++j) {
      const loom_predicate_t* predicate = &attributes[i].predicate_list[j];
      if (!loom_value_fact_relation_predicate_is_supported(module, predicate)) {
        continue;
      }
      if (!plan.initialized) {
        status = loom_value_fact_relation_plan_initialize(
            table, operands, pair_count, predicate_count, use_stack,
            stack_identities, stack_plans, stack_locals, &plan);
      }
      if (iree_status_is_ok(status)) {
        status = loom_value_fact_relation_plan_add_predicate(table, &plan,
                                                             predicate);
      }
    }
  }
  if (iree_status_is_ok(status) && plan.initialized) {
    status = loom_value_fact_integer_relations_publish_plan(
        table, results, pair_count, &plan, inout_changed);
  }
  loom_value_fact_ordinal_map_reset(&plan.identity_map);
  return status;
}

static iree_status_t loom_value_fact_integer_relations_forward_alias(
    loom_value_fact_table_t* table, loom_value_id_t source,
    loom_value_id_t result, bool* inout_changed) {
  const loom_value_fact_relation_roots_t source_roots =
      loom_value_fact_integer_relations_snapshot(table, source);
  const loom_value_fact_relation_roots_t result_roots =
      loom_value_fact_integer_relations_snapshot(table, result);
  if (loom_value_fact_relation_roots_equal(source_roots, result_roots)) {
    return iree_ok_status();
  }
  bool changed = false;
  IREE_RETURN_IF_ERROR(loom_value_fact_integer_relations_publish_one(
      table, result, source_roots, &changed));
  if (changed && inout_changed) {
    *inout_changed = true;
  }
  return iree_ok_status();
}

iree_status_t loom_value_fact_integer_relations_propagate_origins(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, loom_trait_flags_t traits, bool* inout_changed) {
  if (loom_traits_are_fact_identity(traits)) {
    return loom_value_fact_integer_relations_propagate_identity(
        table, module, op, inout_changed);
  }
  if (loom_traits_are_value_alias(traits) && op->operand_count >= 1 &&
      op->result_count >= 1) {
    return loom_value_fact_integer_relations_forward_alias(
        table, loom_op_const_operands(op)[0], loom_op_const_results(op)[0],
        inout_changed);
  }
  return iree_ok_status();
}
