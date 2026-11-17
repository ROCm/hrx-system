// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/condition_derivation.h"

#include "loom/ir/intern_table.h"
#include "loom/ir/structural_hash.h"
#include "loom/util/adaptive_sort.h"
#include "loom/util/fact_relation.h"
#include "loom/util/segmented_storage.h"

#define LOOM_CONDITION_DERIVATION_INITIAL_CAPACITY 16

void loom_condition_fact_set_initialize(
    loom_condition_integer_relation_t* integer_relation_storage,
    iree_host_size_t integer_relation_capacity,
    loom_condition_fact_set_t* out_facts) {
  *out_facts = (loom_condition_fact_set_t){
      .integer_relations = integer_relation_storage,
      .integer_relation_count = 0,
      .integer_relation_capacity = integer_relation_capacity,
  };
}

void loom_condition_fact_set_reset(loom_condition_fact_set_t* facts) {
  facts->integer_relation_count = 0;
}

void loom_condition_derivation_initialize(
    iree_arena_allocator_t* arena,
    loom_condition_derivation_t* out_derivation) {
  *out_derivation = (loom_condition_derivation_t){
      .arena = arena,
  };
}

void loom_condition_derivation_reset(loom_condition_derivation_t* derivation) {
  loom_condition_fact_set_reset(&derivation->integer_facts);
  derivation->boolean_fact_count = 0;
}

bool loom_condition_integer_operands_equal(
    loom_condition_integer_operand_t left,
    loom_condition_integer_operand_t right) {
  if (left.kind != right.kind) {
    return false;
  }
  switch (left.kind) {
    case LOOM_CONDITION_INTEGER_OPERAND_VALUE:
      return left.value_id == right.value_id;
    case LOOM_CONDITION_INTEGER_OPERAND_CONSTANT:
      return left.constant == right.constant;
    default:
      return false;
  }
}

static int loom_condition_integer_operand_compare(
    loom_condition_integer_operand_t left,
    loom_condition_integer_operand_t right) {
  if (left.kind != right.kind) {
    return left.kind < right.kind ? -1 : 1;
  }
  if (left.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
    return (left.value_id > right.value_id) - (left.value_id < right.value_id);
  }
  return (left.constant > right.constant) - (left.constant < right.constant);
}

static loom_condition_integer_relation_t
loom_condition_integer_relation_canonicalize(
    loom_condition_integer_relation_t relation) {
  if (loom_condition_integer_operand_compare(relation.right, relation.left) <
      0) {
    const loom_condition_integer_operand_t left = relation.left;
    relation.left = relation.right;
    relation.right = left;
    relation.relation = loom_symbolic_integer_relation_swap(relation.relation);
  }
  return relation;
}

static bool loom_condition_integer_relation_less(
    const loom_condition_integer_relation_t* left,
    const loom_condition_integer_relation_t* right) {
  int comparison =
      loom_condition_integer_operand_compare(left->left, right->left);
  if (comparison != 0) {
    return comparison < 0;
  }
  comparison =
      loom_condition_integer_operand_compare(left->right, right->right);
  return comparison != 0 ? comparison < 0 : left->relation < right->relation;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_condition_sort_integer_relations,
                          loom_condition_integer_relation_t,
                          loom_condition_integer_relation_less)

static bool loom_condition_boolean_fact_less(
    const loom_condition_boolean_fact_t* left,
    const loom_condition_boolean_fact_t* right) {
  return left->value_id < right->value_id ||
         (left->value_id == right->value_id && left->value < right->value);
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_condition_sort_boolean_facts,
                          loom_condition_boolean_fact_t,
                          loom_condition_boolean_fact_less)

static iree_status_t loom_condition_fact_set_reserve_integer_relations(
    loom_condition_derivation_t* derivation) {
  loom_condition_fact_set_t* facts = &derivation->integer_facts;
  if (facts->integer_relation_count < facts->integer_relation_capacity) {
    return iree_ok_status();
  }
  const iree_host_size_t minimum_capacity =
      facts->integer_relation_capacity == 0
          ? LOOM_CONDITION_DERIVATION_INITIAL_CAPACITY
          : facts->integer_relation_count + 1;
  return iree_arena_grow_array(
      derivation->arena, facts->integer_relation_count, minimum_capacity,
      sizeof(*facts->integer_relations), &facts->integer_relation_capacity,
      (void**)&facts->integer_relations);
}

iree_status_t loom_condition_derivation_append_integer_relation(
    loom_condition_derivation_t* derivation,
    loom_condition_integer_relation_t relation) {
  IREE_RETURN_IF_ERROR(
      loom_condition_fact_set_reserve_integer_relations(derivation));
  loom_condition_fact_set_t* facts = &derivation->integer_facts;
  facts->integer_relations[facts->integer_relation_count++] =
      loom_condition_integer_relation_canonicalize(relation);
  return iree_ok_status();
}

iree_status_t loom_condition_derivation_append_boolean_fact(
    loom_condition_derivation_t* derivation, loom_value_id_t value_id,
    bool value) {
  if (derivation == NULL) {
    return iree_ok_status();
  }
  if (derivation->boolean_fact_count >= derivation->boolean_fact_capacity) {
    const iree_host_size_t minimum_capacity =
        derivation->boolean_fact_capacity == 0
            ? LOOM_CONDITION_DERIVATION_INITIAL_CAPACITY
            : derivation->boolean_fact_count + 1;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        derivation->arena, derivation->boolean_fact_count, minimum_capacity,
        sizeof(*derivation->boolean_facts), &derivation->boolean_fact_capacity,
        (void**)&derivation->boolean_facts));
  }
  derivation->boolean_facts[derivation->boolean_fact_count++] =
      (loom_condition_boolean_fact_t){
          .value_id = value_id,
          .value = value,
      };
  return iree_ok_status();
}

static void loom_condition_derivation_finalize_integer_relations(
    loom_condition_derivation_t* derivation) {
  loom_condition_fact_set_t* facts = &derivation->integer_facts;
  loom_condition_sort_integer_relations(facts->integer_relations,
                                        facts->integer_relation_count);
  iree_host_size_t relation_count = 0;
  for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
    if (relation_count == 0 ||
        loom_condition_integer_relation_less(
            &facts->integer_relations[relation_count - 1],
            &facts->integer_relations[i])) {
      facts->integer_relations[relation_count++] = facts->integer_relations[i];
    }
  }
  facts->integer_relation_count = relation_count;
}

static void loom_condition_derivation_finalize(
    loom_condition_derivation_t* derivation) {
  loom_condition_derivation_finalize_integer_relations(derivation);
  loom_condition_sort_boolean_facts(derivation->boolean_facts,
                                    derivation->boolean_fact_count);
  iree_host_size_t boolean_fact_count = 0;
  for (iree_host_size_t i = 0; i < derivation->boolean_fact_count; ++i) {
    const loom_condition_boolean_fact_t fact = derivation->boolean_facts[i];
    if (boolean_fact_count == 0 ||
        loom_condition_boolean_fact_less(
            &derivation->boolean_facts[boolean_fact_count - 1], &fact)) {
      derivation->boolean_facts[boolean_fact_count++] = fact;
    }
  }
  derivation->boolean_fact_count = boolean_fact_count;
}

enum { LOOM_CONDITION_COMPOSITION_SEGMENT_CAPACITY = 64 };

typedef struct loom_condition_composed_relation_t {
  // SSA value ID or ordinal in the query's literal table.
  uint32_t left;
  // SSA value ID or ordinal in the query's literal table.
  uint32_t right;
  // Original symbolic predicate, independent of operand interpretation.
  uint8_t relation;
  // Left operand interpretation.
  uint8_t left_kind;
  // Right operand interpretation.
  uint8_t right_kind;
} loom_condition_composed_relation_t;

static_assert(sizeof(loom_condition_composed_relation_t) == 12,
              "composed rows retain full-width value and literal IDs");

struct loom_condition_composition_scratch_t {
  // Reusable right-incidence row ordinals, owned by the query arena.
  iree_host_size_t* right_indices;
  // High-water right-incidence capacity.
  iree_host_size_t right_capacity;
  // Reusable index emptied through its current keys before each return.
  loom_intern_table_t derived_relations;
  // Stable compact rows; initialized prefix is the index's count.
  loom_segmented_storage_t rows;
  // Sorted distinct literals from the current original relation prefix.
  int64_t* literals;
  // Number of current literals.
  iree_host_size_t literal_count;
  // High-water literal capacity.
  iree_host_size_t literal_capacity;
};

typedef struct loom_condition_composition_t {
  // Arena retaining reusable construction capacity independently of results.
  iree_arena_allocator_t* arena;
  // Lazily populated scratch owner, reused by consecutive derivations.
  loom_condition_composition_scratch_t** scratch_owner;
  // Original canonical source prefix and destination for the completed result.
  loom_condition_derivation_t* derivation;
  // Sorted original relation prefix; new relations never alter these rows.
  iree_host_size_t source_count;
  // Original row ordinals sorted by their right operand.
  iree_host_size_t* right_indices;
  // Active compact construction state, absent until the first composed row.
  loom_condition_composition_scratch_t* scratch;
} loom_condition_composition_t;

static iree_status_t loom_condition_composition_ensure_scratch(
    iree_arena_allocator_t* arena,
    loom_condition_composition_scratch_t** inout_scratch) {
  if (*inout_scratch != NULL) {
    return iree_ok_status();
  }
  loom_condition_composition_scratch_t* scratch = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*scratch), (void**)&scratch));
  *scratch = (loom_condition_composition_scratch_t){0};
  IREE_RETURN_IF_ERROR(
      loom_intern_table_initialize(arena, 0, &scratch->derived_relations));
  loom_segmented_storage_initialize(
      LOOM_CONDITION_COMPOSITION_SEGMENT_CAPACITY *
          sizeof(loom_condition_composed_relation_t),
      iree_alignof(loom_condition_composed_relation_t), &scratch->rows);
  *inout_scratch = scratch;
  return iree_ok_status();
}

static bool loom_condition_composition_right_less(
    const loom_condition_integer_relation_t* sources,
    const iree_host_size_t* left, const iree_host_size_t* right) {
  const int comparison = loom_condition_integer_operand_compare(
      sources[*left].right, sources[*right].right);
  return comparison != 0 ? comparison < 0 : *left < *right;
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT(loom_condition_composition_sort_right,
                                       iree_host_size_t,
                                       const loom_condition_integer_relation_t*,
                                       loom_condition_composition_right_less)

static iree_host_size_t loom_condition_composition_lower_bound(
    const loom_condition_composition_t* composition,
    loom_condition_integer_operand_t operand, bool right) {
  const loom_condition_integer_relation_t* sources =
      composition->derivation->integer_facts.integer_relations;
  iree_host_size_t begin = 0;
  iree_host_size_t end = composition->source_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    const loom_condition_integer_operand_t key =
        right ? sources[composition->right_indices[middle]].right
              : sources[middle].left;
    if (loom_condition_integer_operand_compare(key, operand) < 0) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin;
}

static bool loom_condition_composition_is_anchor(
    const loom_condition_composition_t* composition,
    loom_condition_integer_operand_t operand) {
  const loom_condition_integer_relation_t* sources =
      composition->derivation->integer_facts.integer_relations;
  const iree_host_size_t left =
      loom_condition_composition_lower_bound(composition, operand, false);
  if (left < composition->source_count &&
      loom_condition_integer_operands_equal(sources[left].left, operand)) {
    return true;
  }
  const iree_host_size_t right =
      loom_condition_composition_lower_bound(composition, operand, true);
  return right < composition->source_count &&
         loom_condition_integer_operands_equal(
             sources[composition->right_indices[right]].right, operand);
}

static bool loom_condition_composition_literal_less(const int64_t* left,
                                                    const int64_t* right) {
  return *left < *right;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_condition_composition_sort_literals, int64_t,
                          loom_condition_composition_literal_less)

static loom_condition_composed_relation_t* loom_condition_composition_row(
    loom_condition_composition_scratch_t* scratch, uint32_t index) {
  loom_condition_composed_relation_t* segment = loom_segmented_storage_segment(
      &scratch->rows, index / LOOM_CONDITION_COMPOSITION_SEGMENT_CAPACITY);
  return &segment[index % LOOM_CONDITION_COMPOSITION_SEGMENT_CAPACITY];
}

static uint32_t loom_condition_composition_encode_operand(
    const loom_condition_composition_scratch_t* scratch,
    loom_condition_integer_operand_t operand) {
  if (operand.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
    return operand.value_id;
  }
  iree_host_size_t begin = 0;
  iree_host_size_t end = scratch->literal_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (scratch->literals[middle] < operand.constant) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return (uint32_t)begin;
}

static loom_condition_integer_operand_t
loom_condition_composition_decode_operand(
    const loom_condition_composition_scratch_t* scratch, uint32_t value,
    uint8_t kind) {
  return kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE
             ? (loom_condition_integer_operand_t){.kind = kind,
                                                  .value_id = value}
             : (loom_condition_integer_operand_t){
                   .kind = kind, .constant = scratch->literals[value]};
}

static loom_condition_integer_relation_t loom_condition_composition_expand(
    const loom_condition_composition_scratch_t* scratch,
    loom_condition_composed_relation_t row) {
  return (loom_condition_integer_relation_t){
      .relation = row.relation,
      .left = loom_condition_composition_decode_operand(scratch, row.left,
                                                        row.left_kind),
      .right = loom_condition_composition_decode_operand(scratch, row.right,
                                                         row.right_kind),
  };
}

typedef struct loom_condition_composition_probe_t {
  // Current compact rows, whose indices survive growth.
  loom_condition_composition_scratch_t* scratch;
  // Canonical candidate searched in the composition index.
  loom_condition_composed_relation_t row;
} loom_condition_composition_probe_t;

static bool loom_condition_composition_equal(const void* user_data,
                                             uint32_t index) {
  const loom_condition_composition_probe_t* probe = user_data;
  const loom_condition_composed_relation_t* row =
      loom_condition_composition_row(probe->scratch, index);
  return row->left == probe->row.left && row->right == probe->row.right &&
         row->relation == probe->row.relation &&
         row->left_kind == probe->row.left_kind &&
         row->right_kind == probe->row.right_kind;
}

static uint32_t loom_condition_composition_hash(
    loom_condition_composed_relation_t row) {
  uint32_t hash =
      loom_structural_hash_mix_u32(loom_structural_hash_initialize(), row.left);
  hash = loom_structural_hash_mix_u32(hash, row.right);
  hash = loom_structural_hash_mix_u32(
      hash, row.relation | ((uint32_t)row.left_kind << 8) |
                ((uint32_t)row.right_kind << 16));
  return loom_structural_hash_finalize(hash);
}

static iree_status_t loom_condition_composition_begin_rows(
    loom_condition_composition_t* composition) {
  IREE_RETURN_IF_ERROR(loom_condition_composition_ensure_scratch(
      composition->arena, composition->scratch_owner));
  loom_condition_composition_scratch_t* scratch = *composition->scratch_owner;
  scratch->literal_count = 0;
  const loom_condition_integer_relation_t* sources =
      composition->derivation->integer_facts.integer_relations;
  for (iree_host_size_t i = 0; i < composition->source_count; ++i) {
    const loom_condition_integer_operand_t operands[] = {sources[i].left,
                                                         sources[i].right};
    for (iree_host_size_t j = 0; j < 2; ++j) {
      if (operands[j].kind != LOOM_CONDITION_INTEGER_OPERAND_CONSTANT) {
        continue;
      }
      if (scratch->literal_count == scratch->literal_capacity) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            composition->arena, scratch->literal_count,
            scratch->literal_count + 1, sizeof(*scratch->literals),
            &scratch->literal_capacity, (void**)&scratch->literals));
      }
      scratch->literals[scratch->literal_count++] = operands[j].constant;
    }
  }
  loom_condition_composition_sort_literals(scratch->literals,
                                           scratch->literal_count);
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < scratch->literal_count; ++i) {
    if (!count || scratch->literals[count - 1] != scratch->literals[i]) {
      scratch->literals[count++] = scratch->literals[i];
    }
  }
  scratch->literal_count = count;
  if (count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "condition literal index capacity exceeded");
  }
  composition->scratch = scratch;
  return iree_ok_status();
}

// Results also enumerate every occupied key. Clear only those keys, ignoring
// holes left by earlier removals; a later query never scans old high-water
// capacity. No insertion or normal lookup occurs during this terminal clear.
static void loom_condition_composition_end_rows(
    loom_condition_composition_scratch_t* scratch) {
  if (scratch == NULL) {
    return;
  }
  loom_intern_table_t* table = &scratch->derived_relations;
  const iree_host_size_t mask = table->capacity - 1;
  for (uint32_t i = 0; i < table->count; ++i) {
    iree_host_size_t slot = loom_condition_composition_hash(
                                *loom_condition_composition_row(scratch, i)) &
                            mask;
    while (loom_intern_table_const_bucket(table, slot)->index != i) {
      slot = (slot + 1) & mask;
    }
    loom_intern_table_bucket(table, slot)->index = UINT32_MAX;
  }
  table->count = 0;
}

static iree_status_t loom_condition_derivation_append_composed_relation(
    loom_condition_composition_t* composition,
    loom_condition_integer_operand_t left,
    loom_integer_order_outcome_bits_t outcomes,
    loom_condition_integer_operand_t right) {
  if (loom_condition_integer_operands_equal(left, right)) {
    return iree_ok_status();
  }
  loom_symbolic_integer_relation_t relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
  if (!loom_symbolic_integer_relation_from_outcomes(outcomes, &relation)) {
    return iree_ok_status();
  }
  const loom_condition_integer_relation_t composed =
      loom_condition_integer_relation_canonicalize(
          (loom_condition_integer_relation_t){
              .relation = relation, .left = left, .right = right});
  loom_condition_derivation_t* derivation = composition->derivation;
  loom_condition_fact_set_t* facts = &derivation->integer_facts;
  iree_host_size_t begin = 0;
  iree_host_size_t end = composition->source_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (loom_condition_integer_relation_less(&facts->integer_relations[middle],
                                             &composed)) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  if (begin < composition->source_count &&
      !loom_condition_integer_relation_less(&composed,
                                            &facts->integer_relations[begin])) {
    return iree_ok_status();
  }
  if (composition->scratch == NULL) {
    IREE_RETURN_IF_ERROR(loom_condition_composition_begin_rows(composition));
  }
  loom_condition_composition_scratch_t* scratch = composition->scratch;
  const loom_condition_composed_relation_t row = {
      .left = loom_condition_composition_encode_operand(scratch, composed.left),
      .right =
          loom_condition_composition_encode_operand(scratch, composed.right),
      .relation = (uint8_t)composed.relation,
      .left_kind = (uint8_t)composed.left.kind,
      .right_kind = (uint8_t)composed.right.kind,
  };
  const uint32_t hash = loom_condition_composition_hash(row);
  const loom_condition_composition_probe_t probe = {.scratch = scratch,
                                                    .row = row};
  loom_intern_probe_t slot =
      loom_intern_table_probe(&scratch->derived_relations, hash,
                              loom_condition_composition_equal, &probe);
  if (slot.index != UINT32_MAX) {
    return iree_ok_status();
  }
  const iree_host_size_t index = scratch->derived_relations.count;
  if (index >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "condition relation index capacity exceeded");
  }
  if (index / LOOM_CONDITION_COMPOSITION_SEGMENT_CAPACITY >=
      scratch->rows.segment_count) {
    void* segment = NULL;
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
        &scratch->rows, composition->arena, &segment));
  }
  IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(
      composition->arena, &scratch->derived_relations, hash, 1, &slot.slot));
  *loom_condition_composition_row(scratch, (uint32_t)index) = row;
  loom_intern_table_insert(&scratch->derived_relations, slot.slot, hash,
                           (uint32_t)index);
  return iree_ok_status();
}

typedef struct loom_condition_retained_composition_t {
  // Producer and append-only indexed result.
  loom_condition_composition_t* composition;
  // Condition-owned endpoint whose consequences are being expanded.
  loom_condition_integer_operand_t anchor;
  // Known relation from anchor to the ambient alias being enumerated.
  loom_integer_order_outcome_bits_t anchor_middle_outcomes;
  // Allocation failure stops the ambient iterator.
  iree_status_t status;
} loom_condition_retained_composition_t;

static bool loom_condition_derivation_compose_retained_relation(
    void* user_data, loom_value_id_t counterpart,
    loom_integer_order_outcome_bits_t middle_counterpart_outcomes) {
  loom_condition_retained_composition_t* composition =
      (loom_condition_retained_composition_t*)user_data;
  const loom_integer_order_outcome_bits_t outcomes =
      loom_integer_order_outcomes_compose(composition->anchor_middle_outcomes,
                                          middle_counterpart_outcomes);
  const loom_condition_integer_operand_t counterpart_operand = {
      .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
      .value_id = counterpart,
  };
  composition->status = loom_condition_derivation_append_composed_relation(
      composition->composition, composition->anchor, outcomes,
      counterpart_operand);
  return iree_status_is_ok(composition->status);
}

static iree_status_t loom_condition_derivation_compose_from_middle(
    const loom_value_fact_table_t* fact_table,
    loom_condition_composition_t* composition,
    loom_condition_integer_operand_t anchor,
    loom_integer_order_outcome_bits_t anchor_middle_outcomes,
    loom_condition_integer_operand_t middle) {
  const iree_host_size_t left_begin =
      loom_condition_composition_lower_bound(composition, middle, false);
  for (iree_host_size_t i = left_begin; i < composition->source_count; ++i) {
    const loom_condition_integer_relation_t source =
        composition->derivation->integer_facts.integer_relations[i];
    if (!loom_condition_integer_operands_equal(middle, source.left)) {
      break;
    }
    IREE_RETURN_IF_ERROR(loom_condition_derivation_append_composed_relation(
        composition, anchor,
        loom_integer_order_outcomes_compose(
            anchor_middle_outcomes,
            loom_symbolic_integer_relation_outcomes(source.relation)),
        source.right));
  }
  const iree_host_size_t right_begin =
      loom_condition_composition_lower_bound(composition, middle, true);
  for (iree_host_size_t i = right_begin; i < composition->source_count; ++i) {
    const loom_condition_integer_relation_t source =
        composition->derivation->integer_facts
            .integer_relations[composition->right_indices[i]];
    if (!loom_condition_integer_operands_equal(middle, source.right)) {
      break;
    }
    if (loom_condition_integer_operands_equal(middle, source.left)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_condition_derivation_append_composed_relation(
        composition, anchor,
        loom_integer_order_outcomes_compose(
            anchor_middle_outcomes,
            loom_integer_order_outcomes_swap(
                loom_symbolic_integer_relation_outcomes(source.relation))),
        source.left));
  }
  if (middle.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE || !fact_table) {
    return iree_ok_status();
  }
  loom_condition_retained_composition_t retained = {
      .composition = composition,
      .anchor = anchor,
      .anchor_middle_outcomes = anchor_middle_outcomes,
      .status = iree_ok_status(),
  };
  (void)loom_value_fact_table_for_each_signed_integer_relation_while(
      fact_table, middle.value_id,
      loom_condition_derivation_compose_retained_relation, &retained);
  return retained.status;
}

// Expands condition-owned anchors through indexed local incidence and retained
// alias relations. Only newly discovered relations enter the result worklist;
// no iteration scans unrelated source edges or the growing output set.
static iree_status_t loom_condition_derivation_compose_integer_relations(
    const loom_value_fact_table_t* fact_table, iree_arena_allocator_t* arena,
    loom_condition_composition_scratch_t** inout_scratch,
    loom_condition_derivation_t* derivation) {
  loom_condition_derivation_finalize_integer_relations(derivation);
  const iree_host_size_t source_count =
      derivation->integer_facts.integer_relation_count;
  if (!source_count) {
    return iree_ok_status();
  }
  iree_host_size_t inline_right_index = 0;
  loom_condition_composition_t composition = {
      .arena = arena,
      .scratch_owner = inout_scratch,
      .derivation = derivation,
      .source_count = source_count,
      .right_indices = &inline_right_index,
  };
  if (source_count > 1) {
    IREE_RETURN_IF_ERROR(
        loom_condition_composition_ensure_scratch(arena, inout_scratch));
    loom_condition_composition_scratch_t* scratch = *inout_scratch;
    if (source_count > scratch->right_capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          arena, 0, source_count, sizeof(*scratch->right_indices),
          &scratch->right_capacity, (void**)&scratch->right_indices));
    }
    composition.right_indices = scratch->right_indices;
    for (iree_host_size_t i = 0; i < source_count; ++i) {
      composition.right_indices[i] = i;
    }
    loom_condition_composition_sort_right(
        derivation->integer_facts.integer_relations, composition.right_indices,
        source_count);
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) &&
       i < source_count + (composition.scratch
                               ? composition.scratch->derived_relations.count
                               : 0);
       ++i) {
    const loom_condition_integer_relation_t relation =
        i < source_count
            ? derivation->integer_facts.integer_relations[i]
            : loom_condition_composition_expand(
                  composition.scratch,
                  *loom_condition_composition_row(
                      composition.scratch, (uint32_t)(i - source_count)));
    const loom_integer_order_outcome_bits_t outcomes =
        loom_symbolic_integer_relation_outcomes(relation.relation);
    if (loom_condition_composition_is_anchor(&composition, relation.left)) {
      status = loom_condition_derivation_compose_from_middle(
          fact_table, &composition, relation.left, outcomes, relation.right);
    }
    if (iree_status_is_ok(status) &&
        loom_condition_composition_is_anchor(&composition, relation.right)) {
      status = loom_condition_derivation_compose_from_middle(
          fact_table, &composition, relation.right,
          loom_integer_order_outcomes_swap(outcomes), relation.left);
    }
  }
  if (iree_status_is_ok(status) && composition.scratch != NULL) {
    const iree_host_size_t count = composition.scratch->derived_relations.count;
    const iree_host_size_t total_count = source_count + count;
    loom_condition_fact_set_t* facts = &derivation->integer_facts;
    if (total_count > facts->integer_relation_capacity) {
      // Grow once for the complete result. Geometric capacity also bounds
      // retained array generations when successive queries get larger.
      status = iree_arena_grow_array(
          derivation->arena, source_count, total_count,
          sizeof(*facts->integer_relations), &facts->integer_relation_capacity,
          (void**)&facts->integer_relations);
    }
    if (iree_status_is_ok(status)) {
      for (uint32_t i = 0; i < count; ++i) {
        facts->integer_relations[source_count + i] =
            loom_condition_composition_expand(
                composition.scratch,
                *loom_condition_composition_row(composition.scratch, i));
      }
      facts->integer_relation_count = total_count;
    }
  }
  loom_condition_composition_end_rows(composition.scratch);
  return status;
}

iree_status_t loom_condition_derivation_complete(
    const loom_value_fact_table_t* fact_table,
    iree_arena_allocator_t* scratch_arena,
    loom_condition_composition_scratch_t** inout_scratch,
    loom_condition_derivation_t* derivation) {
  iree_status_t status = loom_condition_derivation_compose_integer_relations(
      fact_table, scratch_arena, inout_scratch, derivation);
  if (iree_status_is_ok(status)) {
    loom_condition_derivation_finalize(derivation);
  }
  return status;
}

static bool loom_condition_value_exact_integer(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    int64_t* out_value) {
  return fact_table != NULL &&
         loom_value_facts_as_exact_i64(
             loom_value_fact_table_lookup(fact_table, value_id), out_value);
}

static bool loom_condition_operand_matches_value(
    const loom_value_fact_table_t* fact_table,
    loom_condition_integer_operand_t operand, loom_value_id_t value_id) {
  return operand.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE &&
         loom_value_fact_table_query_numeric_identity(fact_table,
                                                      operand.value_id) ==
             loom_value_fact_table_query_numeric_identity(fact_table, value_id);
}

bool loom_condition_integer_relation_make_predicate_for_value(
    const loom_condition_integer_relation_t* relation,
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_predicate_t* out_predicate) {
  loom_symbolic_integer_relation_t normalized_relation = relation->relation;
  loom_condition_integer_operand_t other = {0};
  if (loom_condition_operand_matches_value(fact_table, relation->left,
                                           value_id)) {
    other = relation->right;
  } else if (loom_condition_operand_matches_value(fact_table, relation->right,
                                                  value_id)) {
    other = relation->left;
    normalized_relation =
        loom_symbolic_integer_relation_swap(normalized_relation);
  } else {
    return false;
  }

  const loom_predicate_kind_t predicate_kind =
      loom_symbolic_integer_relation_to_predicate_kind(normalized_relation);

  *out_predicate = (loom_predicate_t){
      .kind = predicate_kind,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_NONE, LOOM_PRED_ARG_NONE},
      .args = {value_id, 0, 0},
  };
  switch (other.kind) {
    case LOOM_CONDITION_INTEGER_OPERAND_CONSTANT:
      out_predicate->arg_tags[1] = LOOM_PRED_ARG_CONST;
      out_predicate->args[1] = other.constant;
      return true;
    case LOOM_CONDITION_INTEGER_OPERAND_VALUE: {
      int64_t constant = 0;
      if (loom_condition_value_exact_integer(fact_table, other.value_id,
                                             &constant)) {
        out_predicate->arg_tags[1] = LOOM_PRED_ARG_CONST;
        out_predicate->args[1] = constant;
      } else {
        out_predicate->arg_tags[1] = LOOM_PRED_ARG_VALUE;
        out_predicate->args[1] = other.value_id;
      }
      return true;
    }
    default:
      return false;
  }
}

bool loom_condition_integer_relation_apply_to_value_facts(
    const loom_condition_integer_relation_t* relation,
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_value_facts_t* inout_facts) {
  loom_predicate_t predicate = {0};
  if (!loom_condition_integer_relation_make_predicate_for_value(
          relation, fact_table, value_id, &predicate)) {
    return false;
  }
  if (predicate.arg_tags[1] == LOOM_PRED_ARG_CONST) {
    loom_value_facts_apply_predicate(inout_facts, &predicate);
    return true;
  }
  return loom_value_facts_refine_relation(
      predicate.kind, *inout_facts,
      loom_value_fact_table_lookup(fact_table,
                                   (loom_value_id_t)predicate.args[1]),
      inout_facts, NULL);
}

bool loom_condition_fact_set_apply_to_value_facts(
    const loom_condition_fact_set_t* facts,
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_value_facts_t* inout_facts) {
  bool applied = false;
  for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
    applied |= loom_condition_integer_relation_apply_to_value_facts(
        &facts->integer_relations[i], fact_table, value_id, inout_facts);
  }
  return applied;
}

static bool loom_condition_integer_operand_exact_i64(
    const loom_value_fact_table_t* fact_table,
    loom_condition_integer_operand_t operand, int64_t* out_value) {
  switch (operand.kind) {
    case LOOM_CONDITION_INTEGER_OPERAND_VALUE:
      return loom_condition_value_exact_integer(fact_table, operand.value_id,
                                                out_value);
    case LOOM_CONDITION_INTEGER_OPERAND_CONSTANT:
      *out_value = operand.constant;
      return true;
    default:
      return false;
  }
}

static bool loom_condition_integer_operands_equivalent_with_facts(
    loom_condition_integer_operand_t left,
    loom_condition_integer_operand_t right,
    const loom_value_fact_table_t* fact_table) {
  if (loom_condition_integer_operands_equal(left, right)) {
    return true;
  }
  if (left.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE &&
      right.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE &&
      loom_value_fact_table_query_numeric_identity(fact_table, left.value_id) ==
          loom_value_fact_table_query_numeric_identity(fact_table,
                                                       right.value_id)) {
    return true;
  }
  int64_t left_value = 0;
  int64_t right_value = 0;
  return fact_table != NULL &&
         loom_condition_integer_operand_exact_i64(fact_table, left,
                                                  &left_value) &&
         loom_condition_integer_operand_exact_i64(fact_table, right,
                                                  &right_value) &&
         left_value == right_value;
}

bool loom_condition_integer_relation_implies(
    const loom_condition_integer_relation_t* known,
    const loom_condition_integer_relation_t* queried, bool* out_result) {
  if (loom_condition_integer_operands_equal(known->left, queried->left) &&
      loom_condition_integer_operands_equal(known->right, queried->right)) {
    return loom_symbolic_integer_relation_implies(
        known->relation, queried->relation, out_result);
  }

  if (loom_condition_integer_operands_equal(known->left, queried->right) &&
      loom_condition_integer_operands_equal(known->right, queried->left)) {
    return loom_symbolic_integer_relation_implies(
        loom_symbolic_integer_relation_swap(known->relation), queried->relation,
        out_result);
  }

  return false;
}

loom_integer_order_outcome_bits_t loom_condition_fact_set_query_integer_order(
    const loom_condition_fact_set_t* facts,
    const loom_value_fact_table_t* fact_table,
    loom_condition_integer_operand_t left,
    loom_condition_integer_operand_t right) {
  if (!facts) {
    return LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL;
  }
  loom_integer_order_outcome_bits_t outcomes =
      LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL;
  for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
    const loom_condition_integer_relation_t* known =
        &facts->integer_relations[i];
    if (loom_condition_integer_operands_equivalent_with_facts(known->left, left,
                                                              fact_table) &&
        loom_condition_integer_operands_equivalent_with_facts(
            known->right, right, fact_table)) {
      outcomes &= loom_symbolic_integer_relation_outcomes(known->relation);
    } else if (loom_condition_integer_operands_equivalent_with_facts(
                   known->left, right, fact_table) &&
               loom_condition_integer_operands_equivalent_with_facts(
                   known->right, left, fact_table)) {
      outcomes &= loom_symbolic_integer_relation_outcomes(
          loom_symbolic_integer_relation_swap(known->relation));
    }
  }
  return outcomes;
}

bool loom_condition_fact_set_proves_integer_relation(
    const loom_condition_fact_set_t* facts,
    const loom_value_fact_table_t* fact_table,
    const loom_condition_integer_relation_t* queried, bool* out_result) {
  const loom_integer_order_outcome_bits_t outcomes =
      loom_condition_fact_set_query_integer_order(
          facts, fact_table, queried->left, queried->right);
  return loom_symbolic_integer_relation_resolve(outcomes, queried->relation,
                                                out_result);
}

bool loom_condition_integer_relations_equivalent(
    const loom_condition_integer_relation_t* left,
    const loom_condition_integer_relation_t* right) {
  bool left_implies_right = false;
  if (!loom_condition_integer_relation_implies(left, right,
                                               &left_implies_right) ||
      !left_implies_right) {
    return false;
  }

  bool right_implies_left = false;
  return loom_condition_integer_relation_implies(right, left,
                                                 &right_implies_left) &&
         right_implies_left;
}

bool loom_condition_integer_relation_meet(
    const loom_condition_integer_relation_t* left,
    const loom_condition_integer_relation_t* right,
    iree_host_size_t right_count,
    loom_condition_integer_relation_t* out_relation) {
  // Conjunction within one edge intersects outcomes; a CFG join unions them.
  loom_integer_order_outcome_bits_t right_outcomes =
      LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL;
  for (iree_host_size_t i = 0; i < right_count; ++i) {
    loom_symbolic_integer_relation_t relation = right[i].relation;
    if (loom_condition_integer_operands_equal(left->left, right[i].left) &&
        loom_condition_integer_operands_equal(left->right, right[i].right)) {
      right_outcomes &= loom_symbolic_integer_relation_outcomes(relation);
    } else if (loom_condition_integer_operands_equal(left->left,
                                                     right[i].right) &&
               loom_condition_integer_operands_equal(left->right,
                                                     right[i].left)) {
      relation = loom_symbolic_integer_relation_swap(relation);
      right_outcomes &= loom_symbolic_integer_relation_outcomes(relation);
    }
  }
  const loom_integer_order_outcome_bits_t common_outcomes =
      loom_symbolic_integer_relation_outcomes(left->relation) | right_outcomes;
  if (common_outcomes == LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL) {
    return false;
  }
  *out_relation = *left;
  return loom_symbolic_integer_relation_from_outcomes(common_outcomes,
                                                      &out_relation->relation);
}
