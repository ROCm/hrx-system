// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_relation.h"

#include "loom/ir/structural_hash.h"

iree_host_size_t loom_value_fact_relation_entry_lower_bound(
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

uint32_t loom_value_fact_relation_atom_hash(
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

uint32_t loom_value_fact_relation_atom_find(
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

bool loom_value_fact_relation_root_contains(
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

// Enumeration supplies exclusions already established by its set cursors.
// Direct pair queries pass zero and resolve that side from the retained roots.
static loom_integer_order_outcome_bits_t
loom_value_fact_relation_query_signed_integer_order(
    const loom_value_fact_table_t* table, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_integer_order_outcome_bits_t lhs_excluded) {
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
      loom_value_fact_table_query_numeric_identity(table, lhs);
  const loom_value_id_t rhs_identity =
      loom_value_fact_table_query_numeric_identity(table, rhs);
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
  if (!lhs_excluded) {
    const uint32_t rhs_identity_atom = loom_value_fact_relation_atom_find(
        table->integer_relations, LOOM_VALUE_FACT_RELATION_ATOM_VALUE,
        (uint64_t)rhs_identity);
    for (uint8_t outcome = 0; outcome < LOOM_INTEGER_ORDER_OUTCOME_COUNT;
         ++outcome) {
      if (loom_value_fact_relation_root_contains(table->integer_relations,
                                                 lhs_roots.excluded[outcome],
                                                 rhs_identity_atom)) {
        lhs_excluded |= 1u << outcome;
      }
    }
  }
  outcomes &= ~lhs_excluded;
  for (uint8_t outcome = 0; outcome < LOOM_INTEGER_ORDER_OUTCOME_COUNT;
       ++outcome) {
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
loom_value_fact_table_query_signed_integer_order(
    const loom_value_fact_table_t* table, loom_value_id_t lhs,
    loom_value_id_t rhs) {
  return loom_value_fact_relation_query_signed_integer_order(
      table, lhs, rhs,
      /*lhs_excluded=*/0);
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

static bool loom_value_fact_relation_excludes_literal(
    const loom_value_fact_integer_relations_t* relations,
    loom_value_fact_relation_root_t root, bool nonzero, int64_t literal) {
  if (nonzero && literal == 0) {
    return true;
  }
  if (!root) {
    return false;
  }
  const uint32_t atom = loom_value_fact_relation_atom_find(
      relations, LOOM_VALUE_FACT_RELATION_ATOM_LITERAL, (uint64_t)literal);
  return loom_value_fact_relation_root_contains(relations, root, atom);
}

void loom_value_fact_table_refine_integer_exclusions(
    const loom_value_fact_table_t* table, loom_value_id_t value,
    loom_value_facts_t* inout_facts) {
  if (loom_value_facts_is_float(*inout_facts)) {
    return;
  }
  const bool nonzero =
      loom_value_facts_is_non_zero(*inout_facts) ||
      (table && loom_value_facts_is_non_zero(
                    loom_value_fact_table_lookup(table, value)));
  if (nonzero) {
    inout_facts->flags |= LOOM_VALUE_FACT_NON_ZERO;
  }
  const loom_value_fact_relation_roots_t roots =
      loom_value_fact_integer_relations_snapshot(table, value);
  const loom_value_fact_integer_relations_t* relations =
      table ? table->integer_relations : NULL;
  int64_t lower = inout_facts->range_lo;
  int64_t upper = inout_facts->range_hi;
  while (lower < upper &&
         loom_value_fact_relation_excludes_literal(
             relations, roots.excluded_literal_equal, nonzero, lower)) {
    ++lower;
  }
  while (lower < upper &&
         loom_value_fact_relation_excludes_literal(
             relations, roots.excluded_literal_equal, nonzero, upper)) {
    --upper;
  }
  if (lower != inout_facts->range_lo || upper != inout_facts->range_hi) {
    *inout_facts = loom_value_facts_clamp_domain(*inout_facts, lower, upper);
  }
}

typedef struct loom_value_fact_relation_root_cursor_t {
  // Retained root determining singleton versus indexed iteration.
  loom_value_fact_relation_root_t root;
  // Indexed traversal state, unused for an inline singleton.
  loom_value_set_cursor_t set_cursor;
  // Current one-based atom ID, or zero when exhausted.
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
    loom_integer_order_outcome_bits_t excluded = 0;
    for (loom_integer_order_outcome_t outcome = 0;
         outcome < LOOM_INTEGER_ORDER_OUTCOME_COUNT; ++outcome) {
      if (cursors[outcome].atom_id == next_atom_id) {
        excluded |= 1u << outcome;
        loom_value_fact_relation_root_cursor_advance(&cursors[outcome]);
      }
    }

    IREE_ASSERT_LE(next_atom_id, table->integer_relations->atoms.value_count);
    const loom_value_fact_relation_atom_t* atom =
        &table->integer_relations->atoms.values[next_atom_id - 1];
    IREE_ASSERT_EQ(atom->kind, LOOM_VALUE_FACT_RELATION_ATOM_VALUE);
    const loom_value_id_t counterpart = (loom_value_id_t)atom->bits;
    const loom_integer_order_outcome_bits_t outcomes =
        loom_value_fact_relation_query_signed_integer_order(
            table, value, counterpart, excluded);
    if (outcomes != LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL &&
        !visit(user_data, counterpart, outcomes)) {
      return false;
    }
  }
}
