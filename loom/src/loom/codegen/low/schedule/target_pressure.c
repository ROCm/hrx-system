// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/target_pressure.h"

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
    uint64_t early_added_units, uint32_t packing_reserve_units) {
  if (limit_units == UINT32_MAX) {
    return;
  }
  if (current_live_units == 0 && delta_units == 0 && early_added_units == 0) {
    return;
  }
  const uint64_t projected_live_units =
      loom_low_schedule_project_live_units(current_live_units, delta_units);
  const uint64_t activation_units =
      delta_units > 0 ? score->activation_reserve_units : 0;
  uint64_t required_live_units =
      iree_math_saturating_add_u64(projected_live_units, activation_units);
  if (early_added_units != 0) {
    const uint64_t early_live_units =
        iree_math_saturating_add_u64(current_live_units, early_added_units);
    required_live_units = iree_max(required_live_units, early_live_units);
  }
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
      pressure_state->packing_reserve_units_by_reg_class[reg_class_id]);
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
      record->candidate_early_added_units, record->packing_reserve_units);
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

void loom_low_schedule_target_pressure_score_candidate(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_candidate_score_t* score) {
  loom_low_schedule_score_candidate_pressure_cliffs(state, pressure_state,
                                                    score);
  if (state->pressure_resources != NULL) {
    loom_low_schedule_project_candidate_resource_pressure(state,
                                                          pressure_state);
    loom_low_schedule_score_candidate_resource_pressure(state, pressure_state,
                                                        score);
  }
  score->persistent_pressure_cliff_penalty = score->pressure_cliff_penalty;
  loom_low_schedule_score_candidate_pressure_limits(state, pressure_state,
                                                    score);
}
