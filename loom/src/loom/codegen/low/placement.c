// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/placement.h"

#include <string.h>

#include "loom/codegen/low/storage_relation.h"

typedef struct loom_low_placement_build_state_t {
  // Module containing the analyzed low region.
  loom_module_t* module;
  // Low function body region being analyzed.
  const loom_region_t* region;
  // Acquired local value domain for |region|.
  const loom_local_value_domain_t* value_domain;
  // Liveness analysis over |value_domain|.
  const loom_liveness_analysis_t* liveness;
  // Concrete scheduled pair opportunities to convert into relations.
  loom_low_placement_pair_use_list_t pair_uses;
  // Arena owning placement table storage.
  iree_arena_allocator_t* arena;
  // Resettable arena owning transient collected relations.
  iree_arena_allocator_t* scratch_arena;
  // Transient relations collected during the single IR walk.
  loom_low_placement_relation_t* collected_relations;
  // Allocated collected_relations capacity.
  iree_host_size_t collected_relation_capacity;
  // Mutable relation records being populated.
  loom_low_placement_relation_t* relations;
  // Relation ranges indexed by result value ordinal.
  loom_low_placement_relation_range_t* ranges_by_result_ordinal;
  // Relation indices grouped by source value ordinal.
  uint32_t* relation_indices_by_source_ordinal;
  // Relation ranges indexed by source value ordinal.
  loom_low_placement_relation_range_t* ranges_by_source_ordinal;
  // Number of relation records counted or populated.
  uint32_t relation_count;
  // Number of collected concrete-location relations.
  iree_host_size_t location_relation_count;
  // Number of low.copy/move/slice/concat operations that may require packet
  // moves.
  uint32_t packet_move_group_count;
  // Total units covered by low.copy/move/slice/concat relations.
  iree_host_size_t packet_move_unit_count;
  // Number of low.br operations that may require edge copies.
  uint32_t edge_copy_group_count;
  // Total units covered by low.br relations.
  iree_host_size_t branch_unit_count;
  // Number of relation records appended after range prefixing.
  iree_host_size_t appended_relation_count;
  // Number of source relation indices appended after range prefixing.
  iree_host_size_t appended_source_relation_count;
} loom_low_placement_build_state_t;

enum loom_low_placement_move_group_flag_bits_e {
  LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_PACKET = 1u << 0,
  LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_EDGE = 1u << 1,
};
typedef uint8_t loom_low_placement_move_group_flags_t;

static bool loom_low_placement_cause_can_alias(
    loom_low_placement_cause_t cause) {
  switch (cause) {
    case LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_MOVE:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_LOOP_ENTRY:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_YIELD:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_CONDITION:
      return true;
    default:
      return false;
  }
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

static loom_value_ordinal_t loom_low_placement_value_ordinal(
    const loom_low_placement_build_state_t* state, loom_value_id_t value_id) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(state->value_domain, value_id);
  IREE_ASSERT(value_ordinal != LOOM_VALUE_ORDINAL_INVALID,
              "verified low placement value must be inside the local value "
              "domain");
  return value_ordinal;
}

static const loom_liveness_interval_t* loom_low_placement_interval_for_ordinal(
    const loom_low_placement_build_state_t* state,
    loom_value_ordinal_t value_ordinal) {
  const loom_liveness_interval_t* interval =
      loom_liveness_interval_for_value_ordinal(state->liveness, value_ordinal);
  IREE_ASSERT(interval != NULL,
              "verified low placement value must have a liveness interval");
  return interval;
}

static iree_status_t loom_low_placement_collect_relation(
    loom_low_placement_build_state_t* state,
    const loom_low_placement_relation_t* relation) {
  IREE_ASSERT_LT(state->relation_count, UINT32_MAX);
  loom_low_placement_relation_range_t* result_range =
      &state->ranges_by_result_ordinal[relation->result_ordinal];
  IREE_ASSERT_LT(result_range->count, UINT32_MAX);
  loom_low_placement_relation_range_t* source_range =
      &state->ranges_by_source_ordinal[relation->source_ordinal];
  IREE_ASSERT_LT(source_range->count, UINT32_MAX);
  if (state->relation_count == state->collected_relation_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->scratch_arena, state->relation_count, state->relation_count + 1,
        sizeof(*state->collected_relations),
        &state->collected_relation_capacity,
        (void**)&state->collected_relations));
  }
  loom_low_placement_relation_t* collected_relation =
      &state->collected_relations[state->relation_count];
  *collected_relation = *relation;
  if (relation->kind >= LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE &&
      relation->kind <= LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART &&
      loom_low_placement_cause_can_alias(relation->cause)) {
    collected_relation->flags |=
        LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
  }
  if (relation->kind == LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION ||
      relation->kind == LOOM_LOW_PLACEMENT_RELATION_DISJOINT_STORAGE) {
    ++state->location_relation_count;
  }
  ++result_range->count;
  ++source_range->count;
  ++state->relation_count;
  return iree_ok_status();
}

static void loom_low_placement_prefix_range_array(
    loom_low_placement_relation_range_t* ranges,
    loom_value_ordinal_t range_count) {
  uint32_t relation_start = 0;
  for (loom_value_ordinal_t i = 0; i < range_count; ++i) {
    loom_low_placement_relation_range_t* range = &ranges[i];
    relation_start += range->count;
    range->start = relation_start - range->count;
    range->count = 0;
  }
}

static void loom_low_placement_prefix_ranges(
    loom_low_placement_build_state_t* state) {
  loom_low_placement_prefix_range_array(state->ranges_by_result_ordinal,
                                        state->value_domain->value_count);
  loom_low_placement_prefix_range_array(state->ranges_by_source_ordinal,
                                        state->value_domain->value_count);
}

static void loom_low_placement_append_relation(
    loom_low_placement_build_state_t* state,
    const loom_low_placement_relation_t* relation) {
  loom_low_placement_relation_range_t* result_range =
      &state->ranges_by_result_ordinal[relation->result_ordinal];
  const iree_host_size_t relation_index =
      (iree_host_size_t)result_range->start + result_range->count;
  IREE_ASSERT_LT(relation_index, state->relation_count);
  state->relations[relation_index] = *relation;
  ++result_range->count;

  loom_low_placement_relation_range_t* source_range =
      &state->ranges_by_source_ordinal[relation->source_ordinal];
  const iree_host_size_t source_index =
      (iree_host_size_t)source_range->start + source_range->count;
  IREE_ASSERT_LT(source_index, state->relation_count);
  state->relation_indices_by_source_ordinal[source_index] =
      (uint32_t)relation_index;
  ++source_range->count;
  ++state->appended_relation_count;
  ++state->appended_source_relation_count;
}

static loom_low_placement_relation_kind_t
loom_low_placement_kind_from_storage_relation(
    loom_low_storage_relation_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_STORAGE_RELATION_SAME_STORAGE:
      return LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE;
    case LOOM_LOW_STORAGE_RELATION_SUBRANGE:
      return LOOM_LOW_PLACEMENT_RELATION_SUBRANGE;
    case LOOM_LOW_STORAGE_RELATION_CONTIGUOUS_PART:
      return LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART;
    case LOOM_LOW_STORAGE_RELATION_DISJOINT_STORAGE:
      return LOOM_LOW_PLACEMENT_RELATION_DISJOINT_STORAGE;
    case LOOM_LOW_STORAGE_RELATION_UNKNOWN:
      return LOOM_LOW_PLACEMENT_RELATION_UNKNOWN;
  }
  return LOOM_LOW_PLACEMENT_RELATION_UNKNOWN;
}

static loom_low_placement_cause_t
loom_low_placement_collect_storage_relation_cause(
    loom_low_placement_build_state_t* state,
    loom_low_placement_move_group_flags_t* move_group_flags,
    loom_low_storage_relation_cause_t cause, uint32_t unit_count) {
  switch (cause) {
    case LOOM_LOW_STORAGE_RELATION_CAUSE_TIED_RESULT:
      return LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT;
    case LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_COPY:
      if ((*move_group_flags & LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_PACKET) ==
          0) {
        *move_group_flags |= LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_PACKET;
        ++state->packet_move_group_count;
      }
      state->packet_move_unit_count += unit_count;
      return LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY;
    case LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_MOVE:
      if ((*move_group_flags & LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_PACKET) ==
          0) {
        *move_group_flags |= LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_PACKET;
        ++state->packet_move_group_count;
      }
      state->packet_move_unit_count += unit_count;
      return LOOM_LOW_PLACEMENT_CAUSE_LOW_MOVE;
    case LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SLICE:
      if ((*move_group_flags & LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_PACKET) ==
          0) {
        *move_group_flags |= LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_PACKET;
        ++state->packet_move_group_count;
      }
      state->packet_move_unit_count += unit_count;
      return LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE;
    case LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_CONCAT:
      if ((*move_group_flags & LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_PACKET) ==
          0) {
        *move_group_flags |= LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_PACKET;
        ++state->packet_move_group_count;
      }
      state->packet_move_unit_count += unit_count;
      return LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT;
    case LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_BRANCH:
      if ((*move_group_flags & LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_EDGE) == 0) {
        *move_group_flags |= LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_EDGE;
        ++state->edge_copy_group_count;
      }
      state->branch_unit_count += unit_count;
      return LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH;
    case LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_LOOP_ENTRY:
      return LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_LOOP_ENTRY;
    case LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_YIELD:
      return LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_YIELD;
    case LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_CONDITION:
      return LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_CONDITION;
    case LOOM_LOW_STORAGE_RELATION_CAUSE_UNKNOWN:
      return LOOM_LOW_PLACEMENT_CAUSE_UNKNOWN;
  }
  return LOOM_LOW_PLACEMENT_CAUSE_UNKNOWN;
}

static loom_low_placement_relation_flags_t
loom_low_placement_flags_from_storage_relation(
    loom_low_storage_relation_flags_t flags) {
  loom_low_placement_relation_flags_t placement_flags = 0;
  if (iree_any_bit_set(flags, LOOM_LOW_STORAGE_RELATION_FLAG_HARD)) {
    placement_flags |= LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD;
  }
  if (iree_any_bit_set(flags, LOOM_LOW_STORAGE_RELATION_FLAG_PREFERRED)) {
    placement_flags |= LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED;
  }
  return placement_flags;
}

static void loom_low_placement_assert_storage_relation_units(
    const loom_low_placement_build_state_t* state,
    const loom_low_storage_relation_t* relation,
    loom_value_ordinal_t result_ordinal, loom_value_ordinal_t source_ordinal) {
  const loom_liveness_interval_t* result_interval =
      loom_low_placement_interval_for_ordinal(state, result_ordinal);
  const loom_liveness_interval_t* source_interval =
      loom_low_placement_interval_for_ordinal(state, source_ordinal);
  IREE_ASSERT(
      relation->destination_unit_offset <= result_interval->unit_count &&
          relation->unit_count <=
              result_interval->unit_count - relation->destination_unit_offset,
      "verified low storage destination range must fit liveness "
      "units");
  IREE_ASSERT(relation->source_unit_offset <= source_interval->unit_count &&
                  relation->unit_count <= source_interval->unit_count -
                                              relation->source_unit_offset,
              "verified low storage source range must fit liveness units");
}

static iree_status_t loom_low_placement_collect_op_relations(
    loom_low_placement_build_state_t* state, const loom_op_t* op) {
  loom_low_placement_move_group_flags_t move_group_flags = 0;
  loom_low_storage_relation_iterator_t iterator;
  loom_low_storage_relation_iterator_initialize(state->module, op, &iterator);
  loom_low_storage_relation_t storage_relation;
  while (
      loom_low_storage_relation_iterator_next(&iterator, &storage_relation)) {
    const loom_value_ordinal_t result_ordinal =
        loom_low_placement_value_ordinal(state,
                                         storage_relation.destination_value_id);
    const loom_value_ordinal_t source_ordinal =
        loom_low_placement_value_ordinal(state,
                                         storage_relation.source_value_id);
    loom_low_placement_assert_storage_relation_units(
        state, &storage_relation, result_ordinal, source_ordinal);
    const loom_low_placement_relation_t placement_relation = {
        .op = storage_relation.op,
        .result_ordinal = result_ordinal,
        .source_ordinal = source_ordinal,
        .result_unit_offset = storage_relation.destination_unit_offset,
        .source_unit_offset = storage_relation.source_unit_offset,
        .unit_count = storage_relation.unit_count,
        .kind = loom_low_placement_kind_from_storage_relation(
            storage_relation.kind),
        .cause = loom_low_placement_collect_storage_relation_cause(
            state, &move_group_flags, storage_relation.cause,
            storage_relation.unit_count),
        .flags = loom_low_placement_flags_from_storage_relation(
            storage_relation.flags),
        .priority = 1,
    };
    IREE_RETURN_IF_ERROR(
        loom_low_placement_collect_relation(state, &placement_relation));
  }
  return iree_ok_status();
}

static const loom_op_t* loom_low_placement_pair_component_op(
    const loom_low_placement_pair_use_t* use,
    loom_low_placement_pair_component_t component) {
  switch (component) {
    case LOOM_LOW_PLACEMENT_PAIR_COMPONENT_FIRST:
      return use->first_op;
    case LOOM_LOW_PLACEMENT_PAIR_COMPONENT_SECOND:
      return use->second_op;
    default:
      IREE_ASSERT_UNREACHABLE("unknown low placement pair component");
      return NULL;
  }
}

loom_value_id_t loom_low_placement_pair_value_id(
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_pair_value_ref_t* ref) {
  IREE_ASSERT_ARGUMENT(use);
  IREE_ASSERT_ARGUMENT(ref);
  const loom_op_t* op =
      loom_low_placement_pair_component_op(use, ref->component);
  switch (ref->kind) {
    case LOOM_LOW_PLACEMENT_PAIR_VALUE_OPERAND:
      IREE_ASSERT_LT(ref->index, op->operand_count);
      return loom_op_const_operands(op)[ref->index];
    case LOOM_LOW_PLACEMENT_PAIR_VALUE_RESULT:
      IREE_ASSERT_LT(ref->index, op->result_count);
      return loom_op_const_results(op)[ref->index];
    default:
      IREE_ASSERT_UNREACHABLE("unknown low placement pair value kind");
      return LOOM_VALUE_ID_INVALID;
  }
}

static bool loom_low_placement_pair_value_ref_equal(
    const loom_low_placement_pair_value_ref_t* lhs,
    const loom_low_placement_pair_value_ref_t* rhs) {
  return lhs->component == rhs->component && lhs->kind == rhs->kind &&
         lhs->index == rhs->index;
}

static bool loom_low_placement_pair_alternative_is_possible_after_separation(
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_pair_relation_t* relations,
    uint16_t relation_count,
    const loom_low_placement_pair_value_ref_t* separated_ref) {
  for (uint16_t i = 0; i < relation_count; ++i) {
    const loom_low_placement_pair_relation_t* relation = &relations[i];
    if (relation->kind !=
            LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION &&
        relation->kind != LOOM_LOW_PLACEMENT_RELATION_DISJOINT_STORAGE) {
      continue;
    }
    if (separated_ref != NULL) {
      const bool separates_result = loom_low_placement_pair_value_ref_equal(
          separated_ref, &relation->result);
      const bool separates_source = loom_low_placement_pair_value_ref_equal(
          separated_ref, &relation->source);
      if (separates_result != separates_source) {
        continue;
      }
    }
    const loom_value_id_t result_value_id =
        loom_low_placement_pair_value_id(use, &relation->result);
    const loom_value_id_t source_value_id =
        loom_low_placement_pair_value_id(use, &relation->source);
    if (result_value_id != source_value_id) {
      continue;
    }
    if (relation->kind ==
        LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION) {
      if (relation->result.unit_offset == relation->source.unit_offset) {
        return false;
      }
    } else {
      const uint32_t result_end =
          (uint32_t)relation->result.unit_offset + relation->unit_count;
      const uint32_t source_end =
          (uint32_t)relation->source.unit_offset + relation->unit_count;
      if (relation->result.unit_offset < source_end &&
          relation->source.unit_offset < result_end) {
        return false;
      }
    }
  }
  return true;
}

bool loom_low_placement_pair_alternative_can_separate_ref(
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_pair_relation_t* relations,
    uint16_t relation_count,
    const loom_low_placement_pair_value_ref_t* separated_ref) {
  IREE_ASSERT_ARGUMENT(use);
  IREE_ASSERT_ARGUMENT(relations);
  IREE_ASSERT_ARGUMENT(separated_ref);
  return loom_low_placement_pair_alternative_is_possible_after_separation(
      use, relations, relation_count, separated_ref);
}

uint16_t loom_low_placement_pair_possible_alternative_count(
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_pair_recipe_t* recipe) {
  IREE_ASSERT_ARGUMENT(use);
  IREE_ASSERT_ARGUMENT(recipe);
  uint16_t possible_count = 0;
  for (uint16_t i = 0; i < recipe->alternative_count; ++i) {
    const loom_low_placement_pair_relation_t* relations =
        &recipe->relations[i * recipe->relation_count];
    if (loom_low_placement_pair_alternative_is_possible_after_separation(
            use, relations, recipe->relation_count,
            /*separated_ref=*/NULL)) {
      ++possible_count;
    }
  }
  return possible_count;
}

static const loom_low_placement_pair_relation_t*
loom_low_placement_select_pair_alternative(
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_pair_recipe_t* recipe) {
  IREE_ASSERT_NE(recipe->relation_count, 0);
  IREE_ASSERT_NE(recipe->alternative_count, 0);
  for (uint16_t i = 0; i < recipe->alternative_count; ++i) {
    const loom_low_placement_pair_relation_t* relations =
        &recipe->relations[i * recipe->relation_count];
    if (loom_low_placement_pair_alternative_is_possible_after_separation(
            use, relations, recipe->relation_count,
            /*separated_ref=*/NULL)) {
      return relations;
    }
  }
  return recipe->relations;
}

static iree_status_t loom_low_placement_collect_pair_relations(
    loom_low_placement_build_state_t* state,
    const loom_low_placement_pair_use_t* use) {
  IREE_ASSERT(use->first_op != NULL);
  IREE_ASSERT(use->second_op != NULL);
  IREE_ASSERT(use->placement_recipe_index !=
              LOOM_LOW_PLACEMENT_PAIR_RECIPE_NONE);
  const uint16_t recipe_index = (uint16_t)(use->placement_recipe_index - 1u);
  IREE_ASSERT_LT(recipe_index, state->pair_uses.placement_recipe_count);
  const loom_low_placement_pair_recipe_t* recipe =
      &state->pair_uses.placement_recipes[recipe_index];
  const loom_low_placement_pair_relation_t* selected_relations =
      loom_low_placement_select_pair_alternative(use, recipe);
  for (uint16_t i = 0; i < recipe->relation_count; ++i) {
    const loom_low_placement_pair_relation_t* recipe_relation =
        &selected_relations[i];
    const loom_value_id_t result_value_id =
        loom_low_placement_pair_value_id(use, &recipe_relation->result);
    const loom_value_id_t source_value_id =
        loom_low_placement_pair_value_id(use, &recipe_relation->source);
    if (result_value_id == source_value_id) {
      continue;
    }
    const loom_value_ordinal_t result_ordinal =
        loom_low_placement_value_ordinal(state, result_value_id);
    const loom_value_ordinal_t source_ordinal =
        loom_low_placement_value_ordinal(state, source_value_id);
    const loom_liveness_interval_t* result_interval =
        loom_low_placement_interval_for_ordinal(state, result_ordinal);
    const loom_liveness_interval_t* source_interval =
        loom_low_placement_interval_for_ordinal(state, source_ordinal);
    IREE_ASSERT_LE(recipe_relation->result.unit_offset,
                   result_interval->unit_count);
    IREE_ASSERT_LE(
        recipe_relation->unit_count,
        result_interval->unit_count - recipe_relation->result.unit_offset);
    IREE_ASSERT_LE(recipe_relation->source.unit_offset,
                   source_interval->unit_count);
    IREE_ASSERT_LE(
        recipe_relation->unit_count,
        source_interval->unit_count - recipe_relation->source.unit_offset);
    const loom_low_placement_relation_t relation = {
        .op = use->second_op,
        .result_ordinal = result_ordinal,
        .source_ordinal = source_ordinal,
        .result_unit_offset = recipe_relation->result.unit_offset,
        .source_unit_offset = recipe_relation->source.unit_offset,
        .unit_count = recipe_relation->unit_count,
        .location_mask = recipe_relation->location_mask,
        .kind = recipe_relation->kind,
        .cause = LOOM_LOW_PLACEMENT_CAUSE_SCHEDULE_PAIR_AFFINITY,
        .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED,
        .priority = use->priority,
    };
    IREE_RETURN_IF_ERROR(loom_low_placement_collect_relation(state, &relation));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_placement_visit_region_ops(
    loom_low_placement_build_state_t* state, const loom_region_t* region) {
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(loom_low_placement_collect_op_relations(state, op));
      if (!iree_any_bit_set(state->value_domain->flags,
                            LOOM_LOCAL_VALUE_DOMAIN_FLAG_REGION_TREE)) {
        continue;
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (regions[i] == NULL) {
          continue;
        }
        IREE_RETURN_IF_ERROR(
            loom_low_placement_visit_region_ops(state, regions[i]));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_placement_visit_ops(
    loom_low_placement_build_state_t* state) {
  return loom_low_placement_visit_region_ops(state, state->region);
}

static iree_status_t loom_low_placement_build(
    loom_low_placement_build_state_t* state,
    loom_low_placement_table_t* out_table) {
  const loom_value_ordinal_t value_count = state->value_domain->value_count;
  if (value_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, value_count, sizeof(*state->ranges_by_result_ordinal),
        (void**)&state->ranges_by_result_ordinal));
    memset(state->ranges_by_result_ordinal, 0,
           value_count * sizeof(*state->ranges_by_result_ordinal));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, value_count, sizeof(*state->ranges_by_source_ordinal),
        (void**)&state->ranges_by_source_ordinal));
    memset(state->ranges_by_source_ordinal, 0,
           value_count * sizeof(*state->ranges_by_source_ordinal));
  }

  IREE_RETURN_IF_ERROR(loom_low_placement_visit_ops(state));
  for (iree_host_size_t i = 0; i < state->pair_uses.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_placement_collect_pair_relations(
        state, &state->pair_uses.values[i]));
  }

  const iree_host_size_t relation_count = state->relation_count;
  if (relation_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->arena, relation_count,
                                                   sizeof(*state->relations),
                                                   (void**)&state->relations));
    memset(state->relations, 0, relation_count * sizeof(*state->relations));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, relation_count,
        sizeof(*state->relation_indices_by_source_ordinal),
        (void**)&state->relation_indices_by_source_ordinal));
    memset(state->relation_indices_by_source_ordinal, 0,
           relation_count * sizeof(*state->relation_indices_by_source_ordinal));
  }
  loom_low_placement_prefix_ranges(state);
  for (iree_host_size_t i = 0; i < relation_count; ++i) {
    loom_low_placement_append_relation(state, &state->collected_relations[i]);
  }
  IREE_ASSERT_EQ(state->appended_relation_count, relation_count);
  IREE_ASSERT_EQ(state->appended_source_relation_count, relation_count);

  *out_table = (loom_low_placement_table_t){
      .module = state->module,
      .region = state->region,
      .value_ids = state->liveness->value_ids,
      .value_count = (loom_value_ordinal_t)state->liveness->value_count,
      .relations = state->relations,
      .relation_count = relation_count,
      .location_relation_count = state->location_relation_count,
      .packet_move_group_count = state->packet_move_group_count,
      .packet_move_unit_count = state->packet_move_unit_count,
      .edge_copy_group_count = state->edge_copy_group_count,
      .branch_unit_count = state->branch_unit_count,
      .ranges_by_result_ordinal = state->ranges_by_result_ordinal,
      .relation_indices_by_source_ordinal =
          state->relation_indices_by_source_ordinal,
      .ranges_by_source_ordinal = state->ranges_by_source_ordinal,
  };
  return iree_ok_status();
}

iree_status_t loom_low_placement_analyze_region(
    loom_module_t* module, const loom_region_t* region,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness,
    loom_low_placement_pair_use_list_t pair_uses, iree_arena_allocator_t* arena,
    loom_low_placement_table_t* out_table) {
  IREE_ASSERT(loom_local_value_domain_is_acquired(value_domain));
  *out_table = (loom_low_placement_table_t){0};
  IREE_ASSERT(value_domain->value_count == liveness->value_count &&
                  value_domain->value_ids == liveness->value_ids,
              "low placement requires liveness over the same local value "
              "domain");

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  loom_low_placement_build_state_t state = {
      .module = module,
      .region = region,
      .value_domain = value_domain,
      .liveness = liveness,
      .pair_uses = pair_uses,
      .arena = arena,
      .scratch_arena = &scratch_arena,
  };
  const iree_status_t status = loom_low_placement_build(&state, out_table);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

loom_low_placement_relation_range_t
loom_low_placement_relation_range_for_value_ordinal(
    const loom_low_placement_table_t* table,
    loom_value_ordinal_t result_ordinal) {
  IREE_ASSERT_LT(result_ordinal, table->value_count);
  IREE_ASSERT(table->ranges_by_result_ordinal != NULL);
  return table->ranges_by_result_ordinal[result_ordinal];
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
