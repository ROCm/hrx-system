// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/target_pressure.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/target/residency.h"

static uint64_t loom_low_schedule_project_live_units(
    uint64_t current_live_units, int64_t delta_units) {
  if (delta_units < 0) {
    const uint64_t removed_units = (uint64_t)(-delta_units);
    IREE_ASSERT_LE(removed_units, current_live_units);
    return current_live_units - removed_units;
  }
  const uint64_t added_units = (uint64_t)delta_units;
  IREE_ASSERT_LE(added_units, UINT64_MAX - current_live_units);
  return current_live_units + added_units;
}

static void loom_low_schedule_record_crossed_pressure_cliff(
    loom_low_schedule_candidate_score_t* score,
    loom_low_schedule_pressure_source_kind_t source_kind, uint16_t source_id,
    uint32_t cliff_units) {
  if (score->pressure_cliff_units != LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE) {
    return;
  }
  score->pressure_cliff_source_kind = source_kind;
  score->pressure_cliff_source_id = source_id;
  score->pressure_cliff_units = cliff_units;
  score->units_until_pressure_cliff = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE;
}

static void loom_low_schedule_record_upcoming_pressure_cliff(
    loom_low_schedule_candidate_score_t* score,
    loom_low_schedule_pressure_source_kind_t source_kind, uint16_t source_id,
    uint32_t units_until_cliff) {
  if (score->pressure_cliff_units != LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE ||
      units_until_cliff >= score->units_until_pressure_cliff) {
    return;
  }
  score->pressure_cliff_source_kind = source_kind;
  score->pressure_cliff_source_id = source_id;
  score->units_until_pressure_cliff = units_until_cliff;
}

static void loom_low_schedule_project_candidate_resource_pressure(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state) {
  for (iree_host_size_t i = 0;
       i < pressure_state->candidate_delta_touched_count; ++i) {
    const uint16_t reg_class_id =
        pressure_state->candidate_delta_touched_reg_class_ids[i];
    const uint64_t projected_live_units = loom_low_schedule_project_live_units(
        pressure_state->current_live_units_by_reg_class[reg_class_id],
        pressure_state->candidate_delta_units_by_reg_class[reg_class_id]);
    const uint64_t peak_live_units =
        pressure_state->resources.peak_live_units_by_reg_class[reg_class_id];
    if (projected_live_units <= peak_live_units) continue;
    const loom_target_residency_derived_member_range_t range =
        loom_target_residency_derived_resource_member_range(
            state->pressure_resources, reg_class_id);
    for (uint16_t j = 0; j < range.count; ++j) {
      const uint16_t member_index =
          state->pressure_resources
              ->member_indices_by_direct_resource[range.start + j];
      const loom_target_residency_derived_member_t* member =
          &state->pressure_resources->members[member_index];
      IREE_ASSERT_EQ(member->direct_resource_id, reg_class_id);
      const uint64_t peak_contribution =
          loom_target_residency_round_resource_units(
              peak_live_units, member->contribution_granularity);
      const uint64_t projected_contribution =
          loom_target_residency_round_resource_units(
              projected_live_units, member->contribution_granularity);
      loom_low_schedule_resource_pressure_record_t* record =
          &pressure_state->resources.records[member->resource_id];
      if (!iree_any_bit_set(
              record->flags,
              LOOM_LOW_SCHEDULE_RESOURCE_PRESSURE_FLAG_CANDIDATE_TOUCHED)) {
        record->flags |=
            LOOM_LOW_SCHEDULE_RESOURCE_PRESSURE_FLAG_CANDIDATE_TOUCHED;
        pressure_state->resources.candidate_touched_ids
            [pressure_state->resources.candidate_touched_count++] =
            member->resource_id;
      }
      record->candidate_added_units = iree_math_saturating_add_u64(
          record->candidate_added_units,
          projected_contribution - peak_contribution);
    }
  }
}

static void loom_low_schedule_score_candidate_resource_pressure(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_candidate_score_t* score) {
  uint32_t resource_penalty = pressure_state->resources.pressure_cliff_penalty;
  for (uint16_t i = 0; i < pressure_state->resources.candidate_touched_count;
       ++i) {
    const uint16_t resource_id =
        pressure_state->resources.candidate_touched_ids[i];
    const loom_low_schedule_resource_pressure_record_t* record =
        &pressure_state->resources.records[resource_id];
    const uint64_t projected_peak_units = iree_math_saturating_add_u64(
        record->current_peak_units, record->candidate_added_units);
    const loom_target_residency_derived_resource_t* resource =
        &state->pressure_resources->resources[resource_id];
    const uint16_t cliff_end = resource->cliff_start + resource->cliff_count;
    if (record->next_cliff_index == cliff_end) continue;
    const loom_target_residency_cliff_t* cliffs =
        &state->pressure_resources->cliffs[record->next_cliff_index];
    const iree_host_size_t cliff_count = cliff_end - record->next_cliff_index;
    loom_target_residency_cliff_evaluation_t evaluation;
    loom_target_residency_evaluate_cliffs(cliffs, cliff_count,
                                          cliffs[0].tier_before,
                                          projected_peak_units, &evaluation);
    const uint32_t penalty = cliffs[0].tier_before - evaluation.tier;
    resource_penalty = iree_math_saturating_add_u32(resource_penalty, penalty);
    if (penalty != 0) {
      loom_low_schedule_record_crossed_pressure_cliff(
          score, LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_RESOURCE, resource_id,
          cliffs[0].cliff_units);
    }
    if (iree_any_bit_set(
            evaluation.flags,
            LOOM_TARGET_RESIDENCY_CLIFF_EVALUATION_FLAG_HAS_WORSE_TIER)) {
      loom_low_schedule_record_upcoming_pressure_cliff(
          score, LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_RESOURCE, resource_id,
          (uint32_t)evaluation.additional_units_to_worse_tier);
    }
  }
  score->pressure_cliff_penalty = iree_math_saturating_add_u32(
      score->pressure_cliff_penalty, resource_penalty);
}

static void loom_low_schedule_score_candidate_pressure_cliffs_for_class(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_candidate_score_t* score, uint16_t reg_class_id) {
  const uint64_t current_live_units =
      pressure_state->current_live_units_by_reg_class[reg_class_id];
  const int64_t delta_units =
      pressure_state->candidate_delta_touched_flags[reg_class_id]
          ? pressure_state->candidate_delta_units_by_reg_class[reg_class_id]
          : 0;
  if (current_live_units == 0 && delta_units == 0) {
    return;
  }
  const uint64_t projected_live_units =
      loom_low_schedule_project_live_units(current_live_units, delta_units);
  const loom_target_residency_cliff_range_t range =
      loom_target_residency_direct_resource_cliff_range(state->pressure_cliffs,
                                                        reg_class_id);
  IREE_ASSERT(pressure_state->first_actionable_pressure_cliff_indices != NULL);
  const uint32_t first_actionable_cliff =
      pressure_state->first_actionable_pressure_cliff_indices[reg_class_id];
  const uint32_t cliff_end = range.start + range.count;
  if (first_actionable_cliff == cliff_end) return;
  const loom_target_residency_cliff_t* cliffs =
      &state->pressure_cliffs->cliffs[first_actionable_cliff];
  loom_target_residency_cliff_evaluation_t evaluation;
  loom_target_residency_evaluate_cliffs(
      cliffs, cliff_end - first_actionable_cliff, cliffs[0].tier_before,
      projected_live_units, &evaluation);
  // Protect target tiers that the source order preserves. Cliffs already
  // crossed by the authored function are excluded so greedy local decisions
  // do not attempt a global residency recovery.
  const uint32_t penalty = cliffs[0].tier_before - evaluation.tier;
  score->pressure_cliff_penalty =
      iree_math_saturating_add_u32(score->pressure_cliff_penalty, penalty);
  if (penalty != 0) {
    loom_low_schedule_record_crossed_pressure_cliff(
        score, LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_CLASS, reg_class_id,
        cliffs[0].cliff_units);
  }
  if (iree_any_bit_set(
          evaluation.flags,
          LOOM_TARGET_RESIDENCY_CLIFF_EVALUATION_FLAG_HAS_WORSE_TIER)) {
    loom_low_schedule_record_upcoming_pressure_cliff(
        score, LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_CLASS, reg_class_id,
        (uint32_t)evaluation.additional_units_to_worse_tier);
  }
}

static void loom_low_schedule_score_candidate_pressure_cliffs(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_candidate_score_t* score) {
  if (state->pressure_cliffs == NULL ||
      state->pressure_cliffs->cliff_count == 0 ||
      pressure_state->current_live_units_by_reg_class == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < pressure_state->block_reg_class_count; ++i) {
    loom_low_schedule_score_candidate_pressure_cliffs_for_class(
        state, pressure_state, score, pressure_state->block_reg_class_ids[i]);
  }
  for (iree_host_size_t i = 0;
       i < pressure_state->candidate_delta_touched_count; ++i) {
    const uint16_t reg_class_id =
        pressure_state->candidate_delta_touched_reg_class_ids[i];
    if (pressure_state->block_reg_class_touched_flags[reg_class_id]) {
      continue;
    }
    loom_low_schedule_score_candidate_pressure_cliffs_for_class(
        state, pressure_state, score, reg_class_id);
  }
}

static void loom_low_schedule_score_candidate_pressure_limit(
    loom_low_schedule_candidate_score_t* score, uint16_t reg_class_id,
    uint32_t limit_units, uint64_t current_live_units, int64_t delta_units,
    uint64_t early_added_units, uint32_t packing_reserve_units,
    bool is_unspillable) {
  if (limit_units == UINT32_MAX) {
    return;
  }
  if (current_live_units == 0 && delta_units == 0 && early_added_units == 0) {
    return;
  }
  const uint64_t projected_live_units =
      loom_low_schedule_project_live_units(current_live_units, delta_units);
  uint64_t candidate_live_units = projected_live_units;
  if (early_added_units != 0) {
    candidate_live_units = iree_max(
        candidate_live_units,
        iree_math_saturating_add_u64(current_live_units, early_added_units));
  }
  const uint64_t activation_units =
      delta_units > 0 ? score->activation_reserve_units : 0;
  if (is_unspillable && candidate_live_units > limit_units &&
      candidate_live_units > current_live_units) {
    score->flags |=
        LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_EXCEEDS_UNSPILLABLE_CAPACITY;
  }
  uint64_t required_live_units =
      iree_math_saturating_add_u64(projected_live_units, activation_units);
  required_live_units = iree_max(required_live_units, candidate_live_units);
  required_live_units =
      iree_math_saturating_add_u64(required_live_units, packing_reserve_units);
  if (projected_live_units >= limit_units) {
    const uint64_t persistent_limit_debt =
        projected_live_units - limit_units + 1;
    const uint32_t persistent_penalty = persistent_limit_debt > UINT32_MAX
                                            ? UINT32_MAX
                                            : (uint32_t)persistent_limit_debt;
    score->persistent_pressure_cliff_penalty = iree_math_saturating_add_u32(
        score->persistent_pressure_cliff_penalty, persistent_penalty);
  }
  if (required_live_units >= limit_units) {
    const uint64_t required_limit_debt = required_live_units - limit_units + 1;
    const uint32_t penalty = required_limit_debt > UINT32_MAX
                                 ? UINT32_MAX
                                 : (uint32_t)required_limit_debt;
    score->pressure_cliff_penalty =
        iree_math_saturating_add_u32(score->pressure_cliff_penalty, penalty);
    loom_low_schedule_record_crossed_pressure_cliff(
        score, LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_CLASS, reg_class_id,
        limit_units);
    return;
  }
  const uint64_t units_until_limit = limit_units - required_live_units;
  loom_low_schedule_record_upcoming_pressure_cliff(
      score, LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_CLASS, reg_class_id,
      (uint32_t)units_until_limit);
}

static void loom_low_schedule_score_candidate_pressure_limit_for_class(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_candidate_score_t* score, uint16_t reg_class_id) {
  if (state->pressure_limits.alias_sets != NULL &&
      state->target.descriptor_set->reg_classes[reg_class_id].alias_set_id !=
          0) {
    return;
  }
  const int64_t delta_units =
      pressure_state->candidate_delta_touched_flags[reg_class_id]
          ? pressure_state->candidate_delta_units_by_reg_class[reg_class_id]
          : 0;
  const uint64_t early_added_units =
      pressure_state->candidate_delta_touched_flags[reg_class_id]
          ? pressure_state
                ->candidate_early_added_units_by_reg_class[reg_class_id]
          : 0;
  loom_low_schedule_score_candidate_pressure_limit(
      score, reg_class_id, state->pressure_limits.by_reg_class[reg_class_id],
      pressure_state->current_live_units_by_reg_class[reg_class_id],
      delta_units, early_added_units,
      pressure_state->packing_reserve_units_by_reg_class[reg_class_id],
      iree_all_bits_set(
          state->target.descriptor_set->reg_classes[reg_class_id].flags,
          LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE));
}

static void loom_low_schedule_score_candidate_pressure_limit_for_alias_set(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_candidate_score_t* score, uint16_t alias_set_id) {
  const loom_low_schedule_alias_pressure_record_t* record =
      &pressure_state->alias_sets.records[alias_set_id];
  loom_low_schedule_score_candidate_pressure_limit(
      score,
      state->pressure_limits.alias_sets[alias_set_id]
          .representative_reg_class_id,
      state->pressure_limits.alias_sets[alias_set_id].live_unit_limit,
      record->current_live_units, record->candidate_delta_units,
      record->candidate_early_added_units, record->packing_reserve_units,
      state->pressure_limits.alias_sets[alias_set_id].all_classes_unspillable);
}

static void loom_low_schedule_score_candidate_pressure_limits(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_candidate_score_t* score) {
  if (state->pressure_limits.by_reg_class == NULL ||
      pressure_state->current_live_units_by_reg_class == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < pressure_state->block_reg_class_count; ++i) {
    loom_low_schedule_score_candidate_pressure_limit_for_class(
        state, pressure_state, score, pressure_state->block_reg_class_ids[i]);
  }
  for (iree_host_size_t i = 0;
       i < pressure_state->candidate_delta_touched_count; ++i) {
    const uint16_t reg_class_id =
        pressure_state->candidate_delta_touched_reg_class_ids[i];
    if (pressure_state->block_reg_class_touched_flags[reg_class_id]) {
      continue;
    }
    loom_low_schedule_score_candidate_pressure_limit_for_class(
        state, pressure_state, score, reg_class_id);
  }
  for (iree_host_size_t i = 0; i < pressure_state->alias_sets.block_count;
       ++i) {
    loom_low_schedule_score_candidate_pressure_limit_for_alias_set(
        state, pressure_state, score, pressure_state->alias_sets.block_ids[i]);
  }
  for (iree_host_size_t i = 0;
       i < pressure_state->alias_sets.candidate_delta_touched_count; ++i) {
    const uint16_t alias_set_id =
        pressure_state->alias_sets.candidate_delta_touched_ids[i];
    if (iree_any_bit_set(pressure_state->alias_sets.records[alias_set_id].flags,
                         LOOM_LOW_SCHEDULE_ALIAS_PRESSURE_FLAG_BLOCK_TOUCHED)) {
      continue;
    }
    loom_low_schedule_score_candidate_pressure_limit_for_alias_set(
        state, pressure_state, score, alias_set_id);
  }
}

uint64_t loom_low_schedule_register_packing_contribution(
    uint64_t register_units,
    const loom_low_register_packing_resource_member_t* member) {
  const uint64_t group_count =
      register_units / member->register_unit_count +
      (register_units % member->register_unit_count != 0);
  return iree_math_saturating_mul_u64(group_count, member->resource_unit_count);
}

static bool loom_low_schedule_register_packing_resource_contains_class(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_register_packing_resource_t* resource,
    uint16_t reg_class_id) {
  const uint16_t member_end = resource->member_start + resource->member_count;
  for (uint16_t member_index = resource->member_start;
       member_index < member_end; ++member_index) {
    if (descriptor_set->register_packing_resource_members[member_index]
            .reg_class_id == reg_class_id) {
      return true;
    }
  }
  return false;
}

// Returns true when a resource contains values that occupy an indivisible
// group of register units. Aggregate capacity alone is handled by ordinary
// pressure recovery; completion identity matters when opening several live
// groups could leave enough units free but no legal group placement.
static bool loom_low_schedule_register_packing_resource_has_aggregate_member(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_register_packing_resource_t* resource) {
  const uint16_t member_end = resource->member_start + resource->member_count;
  for (uint16_t member_index = resource->member_start;
       member_index < member_end; ++member_index) {
    if (descriptor_set->register_packing_resource_members[member_index]
            .register_unit_count > 1) {
      return true;
    }
  }
  return false;
}

static uint32_t loom_low_schedule_value_register_packing_completion_sink(
    const loom_low_schedule_build_state_t* state,
    loom_value_ordinal_t value_ordinal, uint16_t resource_id) {
  const uint32_t producer_node = state->values[value_ordinal].producer_node;
  if (producer_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return LOOM_LOW_SCHEDULE_NODE_NONE;
  }
  return loom_low_schedule_const_register_packing_row(
      state, state->node_register_packing_completion_sinks,
      producer_node)[resource_id];
}

void loom_low_schedule_target_pressure_reset_packing_completions(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state) {
  const uint16_t resource_count =
      state->target.descriptor_set->register_packing_resource_count;
  if (resource_count == 0 ||
      pressure_state->active_register_packing_completion_sinks == NULL) {
    return;
  }
  memset(pressure_state->active_register_packing_completion_sinks, 0xFF,
         resource_count *
             sizeof(*pressure_state->active_register_packing_completion_sinks));
  memset(pressure_state->active_register_packing_completion_value_counts, 0,
         resource_count *
             sizeof(*pressure_state
                         ->active_register_packing_completion_value_counts));
}

void loom_low_schedule_target_pressure_add_packing_completion_value(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t value_ordinal) {
  if (pressure_state->active_register_packing_completion_sinks == NULL) {
    return;
  }
  const loom_low_schedule_value_record_t* value = &state->values[value_ordinal];
  const uint16_t reg_class_id = value->register_class_id;
  if (reg_class_id == LOOM_LOW_REG_CLASS_NONE || value->unit_count == 0) {
    return;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  for (uint16_t resource_id = 0;
       resource_id < descriptor_set->register_packing_resource_count;
       ++resource_id) {
    const loom_low_register_packing_resource_t* resource =
        &descriptor_set->register_packing_resources[resource_id];
    if (!loom_low_schedule_register_packing_resource_contains_class(
            descriptor_set, resource, reg_class_id)) {
      continue;
    }
    const uint32_t completion_sink =
        loom_low_schedule_value_register_packing_completion_sink(
            state, value_ordinal, resource_id);
    if (completion_sink == LOOM_LOW_SCHEDULE_NODE_NONE ||
        state->nodes[completion_sink].scheduled_ordinal !=
            LOOM_LOW_SCHEDULE_NODE_NONE) {
      continue;
    }
    uint32_t* active_sink =
        &pressure_state->active_register_packing_completion_sinks[resource_id];
    if (*active_sink == LOOM_LOW_SCHEDULE_NODE_NONE ||
        completion_sink < *active_sink) {
      *active_sink = completion_sink;
      pressure_state
          ->active_register_packing_completion_value_counts[resource_id] = 1;
    } else if (completion_sink == *active_sink) {
      uint32_t* active_value_count =
          &pressure_state
               ->active_register_packing_completion_value_counts[resource_id];
      IREE_ASSERT_NE(*active_value_count, UINT32_MAX);
      ++*active_value_count;
    }
  }
}

void loom_low_schedule_target_pressure_remove_packing_completion_value(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t value_ordinal) {
  if (pressure_state->active_register_packing_completion_sinks == NULL) {
    return;
  }
  const loom_low_schedule_value_record_t* value = &state->values[value_ordinal];
  const uint16_t reg_class_id = value->register_class_id;
  if (reg_class_id == LOOM_LOW_REG_CLASS_NONE || value->unit_count == 0) {
    return;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  for (uint16_t resource_id = 0;
       resource_id < descriptor_set->register_packing_resource_count;
       ++resource_id) {
    const loom_low_register_packing_resource_t* resource =
        &descriptor_set->register_packing_resources[resource_id];
    if (!loom_low_schedule_register_packing_resource_contains_class(
            descriptor_set, resource, reg_class_id)) {
      continue;
    }
    const uint32_t completion_sink =
        loom_low_schedule_value_register_packing_completion_sink(
            state, value_ordinal, resource_id);
    if (completion_sink == LOOM_LOW_SCHEDULE_NODE_NONE) continue;
    if (pressure_state->active_register_packing_completion_sinks[resource_id] !=
        completion_sink) {
      continue;
    }
    uint32_t* active_value_count =
        &pressure_state
             ->active_register_packing_completion_value_counts[resource_id];
    IREE_ASSERT_NE(*active_value_count, 0u);
    --*active_value_count;
  }
}

void loom_low_schedule_target_pressure_repair_packing_completions(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t node_index) {
  if (pressure_state->active_register_packing_completion_sinks == NULL) {
    return;
  }
  const uint16_t resource_count =
      state->target.descriptor_set->register_packing_resource_count;
  for (uint16_t resource_id = 0; resource_id < resource_count; ++resource_id) {
    const loom_low_register_packing_resource_t* resource =
        &state->target.descriptor_set->register_packing_resources[resource_id];
    uint32_t* active_sink =
        &pressure_state->active_register_packing_completion_sinks[resource_id];
    uint32_t* active_value_count =
        &pressure_state
             ->active_register_packing_completion_value_counts[resource_id];
    if (*active_sink != LOOM_LOW_SCHEDULE_NODE_NONE &&
        state->nodes[*active_sink].scheduled_ordinal ==
            LOOM_LOW_SCHEDULE_NODE_NONE &&
        *active_value_count != 0) {
      continue;
    }
    *active_sink = LOOM_LOW_SCHEDULE_NODE_NONE;
    *active_value_count = 0;
    for (iree_host_size_t i = 0; i < pressure_state->block_value_count; ++i) {
      const loom_value_ordinal_t value_ordinal =
          pressure_state->block_value_ordinals[i];
      const loom_low_schedule_value_record_t* value =
          &state->values[value_ordinal];
      if (!iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE) ||
          !loom_low_schedule_register_packing_resource_contains_class(
              state->target.descriptor_set, resource,
              value->register_class_id)) {
        continue;
      }
      const uint32_t completion_sink =
          loom_low_schedule_value_register_packing_completion_sink(
              state, value_ordinal, resource_id);
      if (completion_sink == LOOM_LOW_SCHEDULE_NODE_NONE ||
          state->nodes[completion_sink].scheduled_ordinal !=
              LOOM_LOW_SCHEDULE_NODE_NONE) {
        continue;
      }
      if (*active_sink == LOOM_LOW_SCHEDULE_NODE_NONE ||
          completion_sink < *active_sink) {
        *active_sink = completion_sink;
        *active_value_count = 1;
      } else if (completion_sink == *active_sink) {
        IREE_ASSERT_NE(*active_value_count, UINT32_MAX);
        ++*active_value_count;
      }
    }
  }
}

uint64_t loom_low_schedule_node_register_packing_operand_units(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node,
    const loom_low_register_packing_resource_t* resource) {
  uint64_t resource_units = 0;
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  const uint16_t member_end = resource->member_start + resource->member_count;
  for (uint16_t member_index = resource->member_start;
       member_index < member_end; ++member_index) {
    const loom_low_register_packing_resource_member_t* member =
        &state->target.descriptor_set
             ->register_packing_resource_members[member_index];
    uint64_t register_units = 0;
    for (uint16_t operand_index = 0; operand_index < node->operand_count;
         ++operand_index) {
      const loom_value_ordinal_t operand_ordinal =
          operand_ordinals[operand_index];
      bool is_duplicate = false;
      for (uint16_t previous_index = 0; previous_index < operand_index;
           ++previous_index) {
        if (operand_ordinals[previous_index] == operand_ordinal) {
          is_duplicate = true;
          break;
        }
      }
      const loom_low_schedule_value_record_t* value =
          &state->values[operand_ordinal];
      if (!is_duplicate && value->register_class_id == member->reg_class_id) {
        register_units =
            iree_math_saturating_add_u64(register_units, value->unit_count);
      }
    }
    resource_units = iree_math_saturating_add_u64(
        resource_units, loom_low_schedule_register_packing_contribution(
                            register_units, member));
  }
  return resource_units;
}

uint64_t loom_low_schedule_node_register_packing_result_units(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node,
    const loom_low_register_packing_resource_t* resource) {
  uint64_t resource_units = 0;
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  const uint16_t member_end = resource->member_start + resource->member_count;
  for (uint16_t member_index = resource->member_start;
       member_index < member_end; ++member_index) {
    const loom_low_register_packing_resource_member_t* member =
        &state->target.descriptor_set
             ->register_packing_resource_members[member_index];
    uint64_t register_units = 0;
    for (uint16_t result_index = 0; result_index < node->result_count;
         ++result_index) {
      const loom_low_schedule_value_record_t* value =
          &state->values[result_ordinals[result_index]];
      if (value->register_class_id == member->reg_class_id) {
        register_units =
            iree_math_saturating_add_u64(register_units, value->unit_count);
      }
    }
    resource_units = iree_math_saturating_add_u64(
        resource_units, loom_low_schedule_register_packing_contribution(
                            register_units, member));
  }
  return resource_units;
}

static uint64_t loom_low_schedule_node_register_packing_working_set(
    const loom_low_schedule_build_state_t* state, uint32_t node_index,
    const loom_low_register_packing_resource_t* resource,
    uint64_t* out_activation_units) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  const uint16_t resource_id =
      (uint16_t)(resource -
                 state->target.descriptor_set->register_packing_resources);
  const uint64_t result_units =
      loom_low_schedule_node_register_packing_result_units(state, node,
                                                           resource);
  const uint64_t activation_units =
      state->node_register_packing_activation_units
          [(iree_host_size_t)node_index *
               state->target.descriptor_set->register_packing_resource_count +
           resource_id];
  *out_activation_units = activation_units;
  return iree_math_saturating_add_u64(result_units, activation_units);
}

static bool loom_low_schedule_candidate_advances_register_packing_completion(
    const loom_low_schedule_build_state_t* state, uint32_t candidate_node_index,
    const loom_low_register_packing_resource_t* resource) {
  if (candidate_node_index == LOOM_LOW_SCHEDULE_NODE_NONE) return false;
  const uint16_t resource_id =
      (uint16_t)(resource -
                 state->target.descriptor_set->register_packing_resources);
  const uint32_t completion_sink =
      state->node_register_packing_completion_sinks
          [(iree_host_size_t)candidate_node_index *
               state->target.descriptor_set->register_packing_resource_count +
           resource_id];
  if (completion_sink != LOOM_LOW_SCHEDULE_NODE_NONE) {
    const loom_low_schedule_node_t* sink = &state->nodes[completion_sink];
    const loom_value_ordinal_t* sink_operand_ordinals =
        loom_low_schedule_node_const_operand_ordinals(sink);
    const uint16_t member_end = resource->member_start + resource->member_count;
    for (uint16_t operand_index = 0; operand_index < sink->operand_count;
         ++operand_index) {
      const loom_low_schedule_value_record_t* value =
          &state->values[sink_operand_ordinals[operand_index]];
      if (!iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
        continue;
      }
      for (uint16_t member_index = resource->member_start;
           member_index < member_end; ++member_index) {
        if (state->target.descriptor_set
                ->register_packing_resource_members[member_index]
                .reg_class_id == value->register_class_id) {
          return true;
        }
      }
    }
  }
  uint64_t candidate_activation_units = 0;
  const uint64_t candidate_working_set =
      loom_low_schedule_node_register_packing_working_set(
          state, candidate_node_index, resource, &candidate_activation_units);
  const loom_low_schedule_node_t* candidate =
      &state->nodes[candidate_node_index];
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(candidate);
  for (uint16_t operand_index = 0; operand_index < candidate->operand_count;
       ++operand_index) {
    const loom_low_schedule_value_record_t* value =
        &state->values[operand_ordinals[operand_index]];
    if (!iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE) ||
        value->producer_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
        state->nodes[value->producer_node].block_index !=
            candidate->block_index) {
      continue;
    }
    bool is_resource_member = false;
    const uint16_t member_end = resource->member_start + resource->member_count;
    for (uint16_t member_index = resource->member_start;
         member_index < member_end; ++member_index) {
      if (state->target.descriptor_set
              ->register_packing_resource_members[member_index]
              .reg_class_id == value->register_class_id) {
        is_resource_member = true;
        break;
      }
    }
    if (!is_resource_member) continue;
    uint64_t producer_activation_units = 0;
    const uint64_t producer_working_set =
        loom_low_schedule_node_register_packing_working_set(
            state, value->producer_node, resource, &producer_activation_units);
    if (producer_activation_units != 0 &&
        candidate_working_set <= producer_working_set) {
      return true;
    }
  }
  return false;
}

static bool loom_low_schedule_candidate_reaches_active_packing_completion(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    uint32_t candidate_node, uint16_t resource_id) {
  const uint32_t active_completion_sink =
      pressure_state->active_register_packing_completion_sinks[resource_id];
  if (active_completion_sink == LOOM_LOW_SCHEDULE_NODE_NONE) return false;
  const uint32_t candidate_completion_sink =
      loom_low_schedule_const_register_packing_row(
          state, state->node_register_packing_completion_sinks,
          candidate_node)[resource_id];
  return candidate_node == active_completion_sink ||
         candidate_completion_sink == active_completion_sink;
}

uint32_t loom_low_schedule_target_pressure_active_packing_completion_capacity(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    uint32_t candidate_node) {
  if (pressure_state->active_register_packing_completion_sinks == NULL) {
    return UINT32_MAX;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  uint32_t active_capacity = UINT32_MAX;
  for (uint16_t resource_id = 0;
       resource_id < descriptor_set->register_packing_resource_count;
       ++resource_id) {
    const loom_low_register_packing_resource_t* resource =
        &descriptor_set->register_packing_resources[resource_id];
    if (!loom_low_schedule_register_packing_resource_has_aggregate_member(
            descriptor_set, resource)) {
      continue;
    }
    if (loom_low_schedule_candidate_reaches_active_packing_completion(
            state, pressure_state, candidate_node, resource_id)) {
      active_capacity = iree_min(active_capacity, resource->capacity);
    }
  }
  return active_capacity;
}

// Materializes the exact same-block SSA ancestors of |completion_sink| into
// one cached completion-domain column. Node order is topological within a low
// block, so a single reverse scan resolves the transitive relation.
static void loom_low_schedule_prepare_unspillable_completion_ancestors(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    uint16_t completion_domain_id, uint32_t completion_sink) {
  uint32_t* cached_sink =
      &pressure_state
           ->cached_unspillable_completion_sinks[completion_domain_id];
  if (*cached_sink == completion_sink) return;
  *cached_sink = completion_sink;

  const uint32_t node_count = state->dependency_index.node_count;
  const uint32_t sink_block_index = state->nodes[completion_sink].block_index;
  for (uint32_t i = node_count; i > 0; --i) {
    const uint32_t node_index = i - 1;
    uint32_t* completion_ancestors =
        loom_low_schedule_unspillable_completion_signature_row(
            state, state->node_unspillable_completion_signatures, node_index);
    bool reaches_completion = node_index == completion_sink;
    if (!reaches_completion &&
        state->nodes[node_index].block_index == sink_block_index) {
      const uint32_t group_begin =
          loom_low_schedule_dependency_index_group_begin(
              &state->dependency_index, node_index);
      const uint32_t group_end = loom_low_schedule_dependency_index_group_end(
          &state->dependency_index, node_index);
      for (uint32_t group_index = group_begin; group_index < group_end;
           ++group_index) {
        if (!loom_low_schedule_dependency_index_group_has_ssa(
                &state->dependency_index, group_index)) {
          continue;
        }
        const loom_low_schedule_dependency_group_t* group =
            loom_low_schedule_dependency_index_group_at(
                &state->dependency_index, group_index);
        const uint32_t consumer_node = group->consumer_node;
        if (consumer_node >= node_count ||
            state->nodes[consumer_node].block_index != sink_block_index) {
          continue;
        }
        const uint32_t* consumer_ancestors =
            loom_low_schedule_const_unspillable_completion_signature_row(
                state, state->node_unspillable_completion_signatures,
                consumer_node);
        if (consumer_ancestors[completion_domain_id] != 0) {
          reaches_completion = true;
          break;
        }
      }
    }
    completion_ancestors[completion_domain_id] = reaches_completion ? 1u : 0u;
  }
}

uint32_t
loom_low_schedule_target_pressure_active_unspillable_completion_capacity(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    uint32_t candidate_node) {
  if (state->node_unspillable_completion_signatures == NULL ||
      pressure_state->active_unspillable_completion_values == NULL) {
    return UINT32_MAX;
  }
  const uint32_t* candidate_completion_signatures =
      loom_low_schedule_const_unspillable_completion_signature_row(
          state, state->node_unspillable_completion_signatures, candidate_node);
  uint32_t active_capacity = UINT32_MAX;
  const uint16_t completion_domain_count =
      state->pressure_limits.unspillable_completion_domain_count;
  for (uint16_t completion_domain_id = 0;
       completion_domain_id < completion_domain_count; ++completion_domain_id) {
    const uint32_t capacity =
        state->pressure_limits
            .unspillable_completion_capacities[completion_domain_id];
    if (capacity >= active_capacity) continue;
    const loom_value_ordinal_t active_value =
        pressure_state
            ->active_unspillable_completion_values[completion_domain_id];
    if (active_value == LOOM_VALUE_ORDINAL_INVALID) continue;
    const uint16_t reg_class_id = state->values[active_value].register_class_id;
    const loom_low_reg_class_t* reg_class =
        &state->target.descriptor_set->reg_classes[reg_class_id];
    const uint64_t current_live_units =
        reg_class->alias_set_id != 0
            ? pressure_state->alias_sets.records[reg_class->alias_set_id]
                  .current_live_units
            : pressure_state->current_live_units_by_reg_class[reg_class_id];
    if (current_live_units < capacity) continue;
    const uint32_t active_completion_sink =
        pressure_state->remaining_consumer_node_xors[active_value];
    loom_low_schedule_prepare_unspillable_completion_ancestors(
        state, pressure_state, completion_domain_id, active_completion_sink);
    if (candidate_completion_signatures[completion_domain_id] != 0) {
      active_capacity = capacity;
    }
  }
  return active_capacity;
}

static void loom_low_schedule_score_candidate_register_packing_resources(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    uint32_t candidate_node_index, loom_low_schedule_candidate_score_t* score) {
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  if (descriptor_set->register_packing_resource_count == 0 ||
      pressure_state->current_live_units_by_reg_class == NULL ||
      pressure_state->candidate_register_packing_activation_units == NULL) {
    return;
  }
  for (uint16_t resource_id = 0;
       resource_id < descriptor_set->register_packing_resource_count;
       ++resource_id) {
    const loom_low_register_packing_resource_t* resource =
        &descriptor_set->register_packing_resources[resource_id];
    const bool advances_packing_completion =
        loom_low_schedule_candidate_advances_register_packing_completion(
            state, candidate_node_index, resource);
    if (advances_packing_completion) {
      score->flags |=
          LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_ADVANCES_CONSTRAINED_COMPLETION;
    }
    uint64_t current_units = 0;
    uint64_t persistent_units = 0;
    uint64_t early_required_units = 0;
    const uint16_t member_end = resource->member_start + resource->member_count;
    for (uint16_t member_index = resource->member_start;
         member_index < member_end; ++member_index) {
      const loom_low_register_packing_resource_member_t* member =
          &descriptor_set->register_packing_resource_members[member_index];
      const uint16_t reg_class_id = member->reg_class_id;
      const uint64_t current_live_units =
          pressure_state->current_live_units_by_reg_class[reg_class_id];
      const int64_t candidate_delta_units =
          pressure_state->candidate_delta_touched_flags[reg_class_id]
              ? pressure_state->candidate_delta_units_by_reg_class[reg_class_id]
              : 0;
      const uint64_t projected_live_units =
          loom_low_schedule_project_live_units(current_live_units,
                                               candidate_delta_units);
      uint64_t early_live_units = projected_live_units;
      if (pressure_state->candidate_delta_touched_flags[reg_class_id]) {
        early_live_units = iree_max(
            early_live_units,
            iree_math_saturating_add_u64(
                current_live_units,
                pressure_state
                    ->candidate_early_added_units_by_reg_class[reg_class_id]));
      }

      const uint64_t persistent_contribution =
          loom_low_schedule_register_packing_contribution(projected_live_units,
                                                          member);
      const uint64_t current_contribution =
          loom_low_schedule_register_packing_contribution(current_live_units,
                                                          member);
      const uint64_t early_required_contribution =
          loom_low_schedule_register_packing_contribution(early_live_units,
                                                          member);
      current_units =
          iree_math_saturating_add_u64(current_units, current_contribution);
      persistent_units = iree_math_saturating_add_u64(persistent_units,
                                                      persistent_contribution);
      early_required_units = iree_math_saturating_add_u64(
          early_required_units, early_required_contribution);
    }
    const uint64_t activation_units =
        pressure_state
            ->candidate_register_packing_activation_units[resource_id];
    const uint64_t activated_units =
        iree_math_saturating_add_u64(persistent_units, activation_units);
    const uint64_t required_units =
        iree_max(early_required_units, activated_units);
    const bool has_aggregate_member =
        loom_low_schedule_register_packing_resource_has_aggregate_member(
            descriptor_set, resource);
    if (has_aggregate_member && advances_packing_completion &&
        loom_low_schedule_candidate_reaches_active_packing_completion(
            state, pressure_state, candidate_node_index, resource_id)) {
      score->active_register_packing_completion_capacity =
          iree_min(score->active_register_packing_completion_capacity,
                   resource->capacity);
      score->flags |= LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_EXACT_PACKING_COMPLETION;
    }
    if (persistent_units > current_units) {
      score->flags |= LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_GROWS_PACKING_RESOURCE;
    }

    if (persistent_units == 0 && required_units == 0 && activation_units == 0) {
      continue;
    }

    if (persistent_units > resource->capacity) {
      const uint64_t debt = persistent_units - resource->capacity;
      score->persistent_pressure_cliff_penalty = iree_math_saturating_add_u32(
          score->persistent_pressure_cliff_penalty,
          debt > UINT32_MAX ? UINT32_MAX : (uint32_t)debt);
    }
    if (required_units > resource->capacity) {
      const uint64_t debt = required_units - resource->capacity;
      score->pressure_cliff_penalty = iree_math_saturating_add_u32(
          score->pressure_cliff_penalty,
          debt > UINT32_MAX ? UINT32_MAX : (uint32_t)debt);
      loom_low_schedule_record_crossed_pressure_cliff(
          score, LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_PACKING_RESOURCE,
          resource_id,
          resource->capacity == UINT32_MAX ? UINT32_MAX
                                           : resource->capacity + 1u);
      continue;
    }

    const uint64_t units_until_capacity = resource->capacity - required_units;
    if (has_aggregate_member && activation_units != 0 &&
        units_until_capacity < activation_units) {
      score->flags |=
          LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_NEEDS_COMPLETION_RECOVERY;
    }
    loom_low_schedule_record_upcoming_pressure_cliff(
        score, LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_PACKING_RESOURCE,
        resource_id,
        units_until_capacity > UINT32_MAX ? UINT32_MAX
                                          : (uint32_t)units_until_capacity);
  }
}

void loom_low_schedule_target_pressure_score_candidate(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    uint32_t candidate_node_index, loom_low_schedule_candidate_score_t* score) {
  loom_low_schedule_score_candidate_pressure_cliffs(state, pressure_state,
                                                    score);
  if (state->pressure_resources != NULL) {
    loom_low_schedule_project_candidate_resource_pressure(state,
                                                          pressure_state);
    loom_low_schedule_score_candidate_resource_pressure(state, pressure_state,
                                                        score);
  }
  score->persistent_pressure_cliff_penalty = score->pressure_cliff_penalty;
  loom_low_schedule_score_candidate_register_packing_resources(
      state, pressure_state, candidate_node_index, score);
  loom_low_schedule_score_candidate_pressure_limits(state, pressure_state,
                                                    score);
}
