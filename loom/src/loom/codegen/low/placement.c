// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/placement.h"

#include "iree/base/internal/math.h"

// Composes source -> intermediate and intermediate -> result placement over
// the units covered by both relations. The returned relation maps the first
// source directly into the second result while retaining the second
// relation's placement semantics.
bool loom_low_placement_relation_compose(
    const loom_low_placement_relation_t* source_to_intermediate,
    const loom_low_placement_relation_t* intermediate_to_result,
    loom_low_placement_relation_t* out_relation) {
  if (source_to_intermediate->result_ordinal !=
      intermediate_to_result->source_ordinal) {
    return false;
  }
  const uint32_t overlap_offset =
      iree_max(source_to_intermediate->result_unit_offset,
               intermediate_to_result->source_unit_offset);
  const uint64_t source_to_intermediate_end =
      (uint64_t)source_to_intermediate->result_unit_offset +
      source_to_intermediate->unit_count;
  const uint64_t intermediate_to_result_end =
      (uint64_t)intermediate_to_result->source_unit_offset +
      intermediate_to_result->unit_count;
  const uint64_t overlap_end =
      iree_min(source_to_intermediate_end, intermediate_to_result_end);
  if (overlap_offset >= overlap_end) {
    return false;
  }
  const uint32_t overlap_count = (uint32_t)(overlap_end - overlap_offset);
  *out_relation = *intermediate_to_result;
  // Bit identity belongs to the selected endpoints, not arbitrary storage
  // composition (which may cross a destructive tie).
  out_relation->flags &= ~LOOM_LOW_PLACEMENT_RELATION_FLAG_IDENTITY_EDGE;
  out_relation->source_ordinal = source_to_intermediate->source_ordinal;
  out_relation->source_operand_index = LOOM_LOW_PLACEMENT_SOURCE_OPERAND_NONE;
  out_relation->source_unit_offset =
      source_to_intermediate->source_unit_offset +
      (overlap_offset - source_to_intermediate->result_unit_offset);
  out_relation->result_unit_offset =
      intermediate_to_result->result_unit_offset +
      (overlap_offset - intermediate_to_result->source_unit_offset);
  out_relation->unit_count = overlap_count;
  return true;
}

bool loom_low_placement_relation_compose_tied_concat_source(
    const loom_low_placement_relation_t* tied_relation,
    const loom_low_placement_relation_t* concat_relation,
    loom_low_placement_relation_t* out_relation) {
  // Carries a concat source slice backward through an exact tied-result alias
  // so the operand can reserve the eventual aligned aggregate.
  if (tied_relation->cause != LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT ||
      concat_relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT ||
      !loom_low_placement_relation_can_alias(concat_relation) ||
      !iree_all_bits_set(
          tied_relation->flags,
          LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD |
              LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE)) {
    return false;
  }
  return loom_low_placement_relation_compose(tied_relation, concat_relation,
                                             out_relation);
}

bool loom_low_placement_relation_can_alias(
    const loom_low_placement_relation_t* relation) {
  IREE_ASSERT_ARGUMENT(relation);
  return iree_any_bit_set(relation->flags,
                          LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE);
}

bool loom_low_placement_cause_is_edge(loom_low_placement_cause_t cause) {
  switch (cause) {
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_LOOP_ENTRY:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_YIELD:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_CONDITION:
      return true;
    default:
      return false;
  }
}

loom_low_placement_relation_range_t
loom_low_placement_relation_range_for_value_ordinal(
    const loom_low_placement_table_t* table,
    loom_value_ordinal_t result_ordinal) {
  IREE_ASSERT_LT(result_ordinal, table->value_count);
  IREE_ASSERT(table->ranges_by_result_ordinal != NULL);
  return table->ranges_by_result_ordinal[result_ordinal];
}

const loom_low_placement_relation_t*
loom_low_placement_defining_transfer_for_value_ordinal(
    const loom_low_placement_table_t* table,
    loom_value_ordinal_t value_ordinal) {
  const loom_low_placement_relation_range_t range =
      table->ranges_by_result_ordinal[value_ordinal];
  if (range.count == 0) {
    return NULL;
  }
  const loom_low_placement_relation_t* relation =
      &table->relations[range.start];
  return relation->cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY ||
                 relation->cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_MOVE
             ? relation
             : NULL;
}

loom_value_ordinal_t loom_low_placement_tied_source_for_value_ordinal(
    const loom_low_placement_table_t* table,
    loom_value_ordinal_t value_ordinal) {
  const loom_low_placement_relation_range_t range =
      loom_low_placement_relation_range_for_value_ordinal(table, value_ordinal);
  IREE_ASSERT_GT(range.count, 0);
  const loom_low_placement_relation_t* relation =
      &table->relations[range.start];
  IREE_ASSERT_EQ(relation->cause, LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT);
  return relation->source_ordinal;
}

loom_low_placement_relation_range_t
loom_low_placement_relation_range_for_source_value_ordinal(
    const loom_low_placement_table_t* table,
    loom_value_ordinal_t source_ordinal) {
  IREE_ASSERT_LT(source_ordinal, table->value_count);
  IREE_ASSERT(table->ranges_by_source_ordinal != NULL);
  return table->ranges_by_source_ordinal[source_ordinal];
}

loom_value_id_t loom_low_placement_value_id(
    const loom_low_placement_table_t* table,
    loom_value_ordinal_t value_ordinal) {
  IREE_ASSERT_LT(value_ordinal, table->value_count);
  return table->value_ids[value_ordinal];
}

loom_value_id_t loom_low_placement_pair_value_id(
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_value_ref_t* ref) {
  const loom_op_t* operations[] = {use->first_op, use->second_op};
  return loom_low_placement_operation_value_id(operations, ref);
}

static bool loom_low_placement_pair_value_ref_equal(
    const loom_low_placement_value_ref_t* lhs,
    const loom_low_placement_value_ref_t* rhs) {
  return lhs->operation_index == rhs->operation_index &&
         lhs->kind == rhs->kind && lhs->index == rhs->index;
}

static bool loom_low_placement_pair_alternative_is_possible_after_separation(
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_preference_t* const* preferences,
    uint16_t preference_count,
    const loom_low_placement_value_ref_t* separated_ref) {
  for (uint16_t i = 0; i < preference_count; ++i) {
    const loom_low_placement_preference_t* preference = preferences[i];
    const loom_low_placement_predicate_t* relation = preference->predicates;
    const loom_low_placement_value_ref_t* result =
        &preference->values[relation->result];
    const loom_low_placement_value_ref_t* source =
        &preference->values[relation->source];
    if (relation->kind !=
            LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION &&
        relation->kind != LOOM_LOW_PLACEMENT_RELATION_DISJOINT_STORAGE) {
      continue;
    }
    if (separated_ref != NULL) {
      const bool separates_result =
          loom_low_placement_pair_value_ref_equal(separated_ref, result);
      const bool separates_source =
          loom_low_placement_pair_value_ref_equal(separated_ref, source);
      if (separates_result != separates_source) {
        continue;
      }
    }
    const loom_value_id_t result_value_id =
        loom_low_placement_pair_value_id(use, result);
    const loom_value_id_t source_value_id =
        loom_low_placement_pair_value_id(use, source);
    if (result_value_id != source_value_id) {
      continue;
    }
    if (relation->kind ==
        LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION) {
      if (relation->result_unit_offset == relation->source_unit_offset) {
        return false;
      }
    } else {
      const uint32_t result_end =
          (uint32_t)relation->result_unit_offset + relation->unit_count;
      const uint32_t source_end =
          (uint32_t)relation->source_unit_offset + relation->unit_count;
      if (relation->result_unit_offset < source_end &&
          relation->source_unit_offset < result_end) {
        return false;
      }
    }
  }
  return true;
}

bool loom_low_placement_pair_alternative_can_separate_ref(
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_preference_t* const* preferences,
    uint16_t preference_count,
    const loom_low_placement_value_ref_t* separated_ref) {
  IREE_ASSERT_ARGUMENT(use);
  IREE_ASSERT_ARGUMENT(preferences);
  IREE_ASSERT_ARGUMENT(separated_ref);
  return loom_low_placement_pair_alternative_is_possible_after_separation(
      use, preferences, preference_count, separated_ref);
}

uint16_t loom_low_placement_pair_possible_alternative_count(
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_pair_recipe_t* recipe) {
  IREE_ASSERT_ARGUMENT(use);
  IREE_ASSERT_ARGUMENT(recipe);
  uint16_t possible_count = 0;
  for (uint16_t i = 0; i < recipe->alternative_count; ++i) {
    const loom_low_placement_preference_t* const* preferences =
        &recipe->preferences[i * recipe->preference_count];
    if (loom_low_placement_pair_alternative_is_possible_after_separation(
            use, preferences, recipe->preference_count,
            /*separated_ref=*/NULL)) {
      ++possible_count;
    }
  }
  return possible_count;
}

const loom_low_placement_preference_t* const*
loom_low_placement_select_pair_alternative(
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_pair_recipe_t* recipe) {
  IREE_ASSERT_NE(recipe->preference_count, 0);
  IREE_ASSERT_NE(recipe->alternative_count, 0);
  for (uint16_t i = 0; i < recipe->alternative_count; ++i) {
    const loom_low_placement_preference_t* const* preferences =
        &recipe->preferences[i * recipe->preference_count];
    if (loom_low_placement_pair_alternative_is_possible_after_separation(
            use, preferences, recipe->preference_count,
            /*separated_ref=*/NULL)) {
      return preferences;
    }
  }
  return recipe->preferences;
}
