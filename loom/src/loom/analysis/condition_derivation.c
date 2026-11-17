// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/condition_derivation.h"

#include "loom/util/adaptive_sort.h"
#include "loom/util/fact_relation.h"

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

static void loom_condition_derivation_finalize(
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

static bool loom_condition_fact_set_contains_integer_relation(
    const loom_condition_fact_set_t* facts,
    loom_condition_integer_relation_t relation) {
  relation = loom_condition_integer_relation_canonicalize(relation);
  for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
    const loom_condition_integer_relation_t* existing =
        &facts->integer_relations[i];
    if (existing->relation == relation.relation &&
        loom_condition_integer_operands_equal(existing->left, relation.left) &&
        loom_condition_integer_operands_equal(existing->right,
                                              relation.right)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_condition_derivation_append_composed_relation(
    loom_condition_derivation_t* derivation,
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
  const loom_condition_integer_relation_t composed = {
      .relation = relation,
      .left = left,
      .right = right,
  };
  if (loom_condition_fact_set_contains_integer_relation(
          &derivation->integer_facts, composed)) {
    return iree_ok_status();
  }
  return loom_condition_derivation_append_integer_relation(derivation,
                                                           composed);
}

typedef struct loom_condition_retained_composition_t {
  // Complete result receiving composed relations.
  loom_condition_derivation_t* derivation;
  // Original condition operand whose consequences are expanded.
  loom_condition_integer_operand_t anchor;
  // Known relation from the anchor to the current ambient alias.
  loom_integer_order_outcome_bits_t anchor_middle_outcomes;
  // Allocation failure stops the ambient relation visitor.
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
      composition->derivation, composition->anchor, outcomes,
      counterpart_operand);
  return iree_status_is_ok(composition->status);
}

static iree_status_t loom_condition_derivation_compose_from_middle(
    const loom_value_fact_table_t* fact_table,
    loom_condition_derivation_t* derivation,
    iree_host_size_t source_relation_count,
    loom_condition_integer_operand_t anchor,
    loom_integer_order_outcome_bits_t anchor_middle_outcomes,
    loom_condition_integer_operand_t middle) {
  for (iree_host_size_t i = 0; i < source_relation_count; ++i) {
    const loom_condition_integer_relation_t source =
        derivation->integer_facts.integer_relations[i];
    loom_condition_integer_operand_t counterpart = {0};
    loom_integer_order_outcome_bits_t middle_counterpart_outcomes = 0;
    if (loom_condition_integer_operands_equal(middle, source.left)) {
      counterpart = source.right;
      middle_counterpart_outcomes =
          loom_symbolic_integer_relation_outcomes(source.relation);
    } else if (loom_condition_integer_operands_equal(middle, source.right)) {
      counterpart = source.left;
      middle_counterpart_outcomes = loom_integer_order_outcomes_swap(
          loom_symbolic_integer_relation_outcomes(source.relation));
    } else {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_condition_derivation_append_composed_relation(
        derivation, anchor,
        loom_integer_order_outcomes_compose(anchor_middle_outcomes,
                                            middle_counterpart_outcomes),
        counterpart));
  }

  if (middle.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE || !fact_table) {
    return iree_ok_status();
  }
  loom_condition_retained_composition_t composition = {
      .derivation = derivation,
      .anchor = anchor,
      .anchor_middle_outcomes = anchor_middle_outcomes,
      .status = iree_ok_status(),
  };
  (void)loom_value_fact_table_for_each_signed_integer_relation_while(
      fact_table, middle.value_id,
      loom_condition_derivation_compose_retained_relation, &composition);
  return composition.status;
}

static bool loom_condition_derivation_operand_is_source_anchor(
    const loom_condition_derivation_t* derivation,
    iree_host_size_t source_relation_count,
    loom_condition_integer_operand_t operand) {
  for (iree_host_size_t i = 0; i < source_relation_count; ++i) {
    const loom_condition_integer_relation_t* source =
        &derivation->integer_facts.integer_relations[i];
    if (loom_condition_integer_operands_equal(operand, source->left) ||
        loom_condition_integer_operands_equal(operand, source->right)) {
      return true;
    }
  }
  return false;
}

// Expands each condition-owned anchor through other local condition relations
// and sparse relations retained by the intermediate alias. Derived relations
// are persisted with the edge derivation, keeping downstream proof queries
// indexed and preventing per-use relation walks. Expansion never connects two
// ambient-only values, which bounds closure to the condition's own operands.
static iree_status_t loom_condition_derivation_compose_integer_relations(
    const loom_value_fact_table_t* fact_table,
    loom_condition_derivation_t* derivation) {
  const iree_host_size_t source_relation_count =
      derivation->integer_facts.integer_relation_count;
  for (iree_host_size_t i = 0;
       i < derivation->integer_facts.integer_relation_count; ++i) {
    const loom_condition_integer_relation_t relation =
        derivation->integer_facts.integer_relations[i];
    const loom_integer_order_outcome_bits_t outcomes =
        loom_symbolic_integer_relation_outcomes(relation.relation);
    if (loom_condition_derivation_operand_is_source_anchor(
            derivation, source_relation_count, relation.left)) {
      IREE_RETURN_IF_ERROR(loom_condition_derivation_compose_from_middle(
          fact_table, derivation, source_relation_count, relation.left,
          outcomes, relation.right));
    }
    if (loom_condition_derivation_operand_is_source_anchor(
            derivation, source_relation_count, relation.right)) {
      IREE_RETURN_IF_ERROR(loom_condition_derivation_compose_from_middle(
          fact_table, derivation, source_relation_count, relation.right,
          loom_integer_order_outcomes_swap(outcomes), relation.left));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_condition_derivation_complete(
    const loom_value_fact_table_t* fact_table,
    loom_condition_derivation_t* derivation) {
  iree_status_t status = loom_condition_derivation_compose_integer_relations(
      fact_table, derivation);
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
