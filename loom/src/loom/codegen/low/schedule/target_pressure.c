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

uint64_t loom_low_schedule_register_packing_contribution(
    uint64_t register_units,
    const loom_low_register_packing_resource_member_t* member) {
  const uint64_t group_count =
      register_units / member->register_unit_count +
      (register_units % member->register_unit_count != 0);
  return iree_math_saturating_mul_u64(group_count, member->resource_unit_count);
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
    if (loom_low_schedule_candidate_advances_register_packing_completion(
            state, candidate_node_index, resource)) {
      score->flags |=
          LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_ADVANCES_PACKING_COMPLETION;
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
    if (activation_units != 0 && units_until_capacity < activation_units) {
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
