// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/pressure.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/schedule/completion_wait.h"
#include "loom/codegen/low/schedule/ready_frontier.h"
#include "loom/codegen/low/schedule/target_pressure.h"
#include "loom/target/registers.h"

#define LOOM_LOW_SCHEDULE_DESCRIPTOR_FRONTIER_CAPACITY 16u

// Incremental pressure summary for consumers unlocked by one producer.
struct loom_low_schedule_unlock_record_t {
  // Sum of downstream pressure demand across unlocked consumers.
  uint32_t demand_units;
  // Maximum downstream activation width across unlocked consumers.
  uint32_t activation_units;
  // Maximum latency among descriptor consumers unlocked by the producer.
  uint16_t descriptor_latency_cycles;
  // Number of retained descriptor consumers, capped at capacity + 1.
  uint8_t descriptor_count;
  // Candidate facts published by final unscheduled consumers.
  uint8_t candidate_flags;
};

typedef enum loom_low_schedule_resource_high_water_mode_e {
  LOOM_LOW_SCHEDULE_RESOURCE_HIGH_WATER_SOURCE_BASELINE = 0,
  LOOM_LOW_SCHEDULE_RESOURCE_HIGH_WATER_SCHEDULED = 1,
} loom_low_schedule_resource_high_water_mode_t;

typedef struct loom_low_schedule_pressure_demand_t {
  // Downstream visible register demand reached through structural nodes.
  uint32_t demand_units;
  // Maximum register activation of a descriptor made ready by the candidate.
  uint32_t activation_units;
  // Pair-affinity reward made available by the candidate.
  uint16_t pair_affinity_score;
  // Maximum latency among non-growing descriptor consumers unlocked by the
  // candidate.
  uint16_t non_growing_descriptor_latency_cycles;
  // Candidate properties discovered while traversing the ready frontier.
  uint8_t candidate_flags;
} loom_low_schedule_pressure_demand_t;

static uint32_t loom_low_schedule_positive_delta_u32(uint32_t lhs,
                                                     uint32_t rhs) {
  return lhs > rhs ? lhs - rhs : 0;
}

static iree_string_view_t loom_low_schedule_reg_class_name(
    const loom_low_schedule_build_state_t* state, uint16_t reg_class_id) {
  IREE_ASSERT_LT(reg_class_id, state->target.descriptor_set->reg_class_count);
  return loom_low_descriptor_set_string(
      state->target.descriptor_set,
      state->target.descriptor_set->reg_classes[reg_class_id]
          .name_string_offset);
}

iree_string_view_t loom_low_schedule_pressure_source_name(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_source_kind_t source_kind, uint16_t source_id) {
  switch (source_kind) {
    case LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_CLASS:
      return loom_low_schedule_reg_class_name(state, source_id);
    case LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_RESOURCE:
      IREE_ASSERT(state->pressure_resources != NULL);
      IREE_ASSERT_LT(source_id, state->pressure_resources->resource_count);
      return state->pressure_resources->resources[source_id].name;
    default:
      return iree_string_view_empty();
  }
}

iree_status_t loom_low_schedule_pressure_initialize(
    loom_low_schedule_build_state_t* state, iree_host_size_t node_count,
    loom_low_schedule_pressure_state_t* out_pressure_state) {
  *out_pressure_state = (loom_low_schedule_pressure_state_t){0};
  if (!loom_low_schedule_strategy_uses_pressure(state->options->strategy)) {
    return iree_ok_status();
  }
  const loom_value_ordinal_t value_count = state->value_domain->value_count;
  if (value_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, value_count,
        sizeof(*out_pressure_state->block_value_ordinals),
        (void**)&out_pressure_state->block_value_ordinals));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, value_count,
        sizeof(*out_pressure_state->candidate_operand_use_counts),
        (void**)&out_pressure_state->candidate_operand_use_counts));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, value_count,
        sizeof(*out_pressure_state->candidate_operand_ordinals),
        (void**)&out_pressure_state->candidate_operand_ordinals));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, value_count,
        sizeof(*out_pressure_state->candidate_scratch_counts),
        (void**)&out_pressure_state->candidate_scratch_counts));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, value_count,
        sizeof(*out_pressure_state->remaining_consumer_counts),
        (void**)&out_pressure_state->remaining_consumer_counts));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, value_count,
        sizeof(*out_pressure_state->remaining_consumer_node_xors),
        (void**)&out_pressure_state->remaining_consumer_node_xors));
    memset(out_pressure_state->candidate_operand_use_counts, 0,
           value_count *
               sizeof(*out_pressure_state->candidate_operand_use_counts));
    memset(out_pressure_state->candidate_scratch_counts, 0,
           value_count * sizeof(*out_pressure_state->candidate_scratch_counts));
    memset(
        out_pressure_state->remaining_consumer_counts, 0,
        value_count * sizeof(*out_pressure_state->remaining_consumer_counts));
    memset(out_pressure_state->remaining_consumer_node_xors, 0,
           value_count *
               sizeof(*out_pressure_state->remaining_consumer_node_xors));
  }
  IREE_RETURN_IF_ERROR(loom_low_schedule_pressure_alias_initialize(
      state, &out_pressure_state->storage_aliases));
  const uint32_t reg_class_count =
      state->target.descriptor_set->reg_class_count;
  if (reg_class_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count,
        sizeof(*out_pressure_state->current_live_units_by_reg_class),
        (void**)&out_pressure_state->current_live_units_by_reg_class));
    memset(out_pressure_state->current_live_units_by_reg_class, 0,
           reg_class_count *
               sizeof(*out_pressure_state->current_live_units_by_reg_class));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count,
        sizeof(*out_pressure_state->packing_reserve_units_by_reg_class),
        (void**)&out_pressure_state->packing_reserve_units_by_reg_class));
    memset(out_pressure_state->packing_reserve_units_by_reg_class, 0,
           reg_class_count *
               sizeof(*out_pressure_state->packing_reserve_units_by_reg_class));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count,
        sizeof(*out_pressure_state->block_reg_class_ids),
        (void**)&out_pressure_state->block_reg_class_ids));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count,
        sizeof(*out_pressure_state->block_reg_class_touched_flags),
        (void**)&out_pressure_state->block_reg_class_touched_flags));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count,
        sizeof(*out_pressure_state->candidate_delta_units_by_reg_class),
        (void**)&out_pressure_state->candidate_delta_units_by_reg_class));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count,
        sizeof(*out_pressure_state->candidate_early_added_units_by_reg_class),
        (void**)&out_pressure_state->candidate_early_added_units_by_reg_class));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count,
        sizeof(*out_pressure_state->candidate_delta_touched_flags),
        (void**)&out_pressure_state->candidate_delta_touched_flags));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count,
        sizeof(*out_pressure_state->candidate_delta_touched_reg_class_ids),
        (void**)&out_pressure_state->candidate_delta_touched_reg_class_ids));
    if (state->pressure_cliffs != NULL &&
        state->pressure_cliffs->cliff_count != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->arena, reg_class_count,
          sizeof(*out_pressure_state->first_actionable_pressure_cliff_indices),
          (void**)&out_pressure_state
              ->first_actionable_pressure_cliff_indices));
      for (uint16_t reg_class_id = 0; reg_class_id < reg_class_count;
           ++reg_class_id) {
        const loom_target_residency_cliff_range_t range =
            loom_target_residency_direct_resource_cliff_range(
                state->pressure_cliffs, reg_class_id);
        out_pressure_state
            ->first_actionable_pressure_cliff_indices[reg_class_id] =
            range.start;
      }
    }
    memset(out_pressure_state->candidate_delta_units_by_reg_class, 0,
           reg_class_count *
               sizeof(*out_pressure_state->candidate_delta_units_by_reg_class));
    memset(
        out_pressure_state->candidate_early_added_units_by_reg_class, 0,
        reg_class_count *
            sizeof(
                *out_pressure_state->candidate_early_added_units_by_reg_class));
    memset(out_pressure_state->block_reg_class_touched_flags, 0,
           reg_class_count *
               sizeof(*out_pressure_state->block_reg_class_touched_flags));
    memset(out_pressure_state->candidate_delta_touched_flags, 0,
           reg_class_count *
               sizeof(*out_pressure_state->candidate_delta_touched_flags));
  }
  const uint16_t alias_set_count = state->pressure_limits.alias_set_count;
  if (alias_set_count != 0) {
    const iree_host_size_t alias_set_slot_count =
        (iree_host_size_t)alias_set_count + 1;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, alias_set_slot_count,
        sizeof(*out_pressure_state->alias_sets.records),
        (void**)&out_pressure_state->alias_sets.records));
    memset(
        out_pressure_state->alias_sets.records, 0,
        alias_set_slot_count * sizeof(*out_pressure_state->alias_sets.records));
    uint16_t* touched_ids = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, (iree_host_size_t)alias_set_count * 2,
        sizeof(*touched_ids), (void**)&touched_ids));
    out_pressure_state->alias_sets.block_ids = touched_ids;
    out_pressure_state->alias_sets.candidate_delta_touched_ids =
        touched_ids + alias_set_count;
  }
  if (state->pressure_resources != NULL) {
    const uint16_t resource_count = state->pressure_resources->resource_count;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count,
        sizeof(*out_pressure_state->resources.peak_live_units_by_reg_class),
        (void**)&out_pressure_state->resources.peak_live_units_by_reg_class));
    memset(
        out_pressure_state->resources.peak_live_units_by_reg_class, 0,
        reg_class_count *
            sizeof(
                *out_pressure_state->resources.peak_live_units_by_reg_class));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, resource_count,
        sizeof(*out_pressure_state->resources.records),
        (void**)&out_pressure_state->resources.records));
    memset(out_pressure_state->resources.records, 0,
           resource_count * sizeof(*out_pressure_state->resources.records));
    for (uint16_t resource_id = 0; resource_id < resource_count;
         ++resource_id) {
      out_pressure_state->resources.records[resource_id].next_cliff_index =
          state->pressure_resources->resources[resource_id].cliff_start;
    }
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, resource_count,
        sizeof(*out_pressure_state->resources.candidate_touched_ids),
        (void**)&out_pressure_state->resources.candidate_touched_ids));
  }
  return iree_ok_status();
}

static void loom_low_schedule_note_block_pressure_reg_class(
    loom_low_schedule_pressure_state_t* pressure_state, uint16_t reg_class_id) {
  if (reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
      pressure_state->block_reg_class_touched_flags == NULL ||
      pressure_state->block_reg_class_touched_flags[reg_class_id]) {
    return;
  }
  pressure_state->block_reg_class_touched_flags[reg_class_id] = 1;
  pressure_state->block_reg_class_ids[pressure_state->block_reg_class_count++] =
      reg_class_id;
}

static inline uint16_t loom_low_schedule_alias_set_id(
    const loom_low_schedule_build_state_t* state, uint16_t reg_class_id) {
  return reg_class_id == LOOM_LOW_REG_CLASS_NONE
             ? 0
             : state->target.descriptor_set->reg_classes[reg_class_id]
                   .alias_set_id;
}

static void loom_low_schedule_note_block_pressure_alias_set(
    loom_low_schedule_pressure_state_t* pressure_state, uint16_t alias_set_id) {
  if (alias_set_id == 0 || pressure_state->alias_sets.records == NULL) {
    return;
  }
  loom_low_schedule_alias_pressure_record_t* record =
      &pressure_state->alias_sets.records[alias_set_id];
  if (iree_any_bit_set(record->flags,
                       LOOM_LOW_SCHEDULE_ALIAS_PRESSURE_FLAG_BLOCK_TOUCHED)) {
    return;
  }
  record->flags |= LOOM_LOW_SCHEDULE_ALIAS_PRESSURE_FLAG_BLOCK_TOUCHED;
  pressure_state->alias_sets
      .block_ids[pressure_state->alias_sets.block_count++] = alias_set_id;
}

static inline void loom_low_schedule_adjust_alias_pressure(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint16_t reg_class_id,
    int64_t delta_units) {
  if (pressure_state->alias_sets.records == NULL || delta_units == 0) {
    return;
  }
  const uint16_t alias_set_id =
      loom_low_schedule_alias_set_id(state, reg_class_id);
  if (alias_set_id == 0) {
    return;
  }
  loom_low_schedule_alias_pressure_record_t* record =
      &pressure_state->alias_sets.records[alias_set_id];
  if (delta_units < 0) {
    const uint64_t removed_units = (uint64_t)-delta_units;
    IREE_ASSERT_LE(removed_units, record->current_live_units);
    record->current_live_units -= removed_units;
  } else {
    loom_low_schedule_note_block_pressure_alias_set(pressure_state,
                                                    alias_set_id);
    record->current_live_units += (uint64_t)delta_units;
  }
}

static void loom_low_schedule_reset_block_alias_pressure(
    loom_low_schedule_pressure_state_t* pressure_state) {
  for (iree_host_size_t i = 0; i < pressure_state->alias_sets.block_count;
       ++i) {
    loom_low_schedule_alias_pressure_record_t* record =
        &pressure_state->alias_sets
             .records[pressure_state->alias_sets.block_ids[i]];
    record->current_live_units = 0;
    record->packing_reserve_units = 0;
    record->flags &= ~LOOM_LOW_SCHEDULE_ALIAS_PRESSURE_FLAG_BLOCK_TOUCHED;
  }
  pressure_state->alias_sets.block_count = 0;
}

static void loom_low_schedule_advance_resource_cliffs(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint16_t resource_id,
    loom_low_schedule_resource_high_water_mode_t mode) {
  const loom_target_residency_derived_resource_t* resource =
      &state->pressure_resources->resources[resource_id];
  loom_low_schedule_resource_pressure_record_t* record =
      &pressure_state->resources.records[resource_id];
  const uint16_t cliff_end = resource->cliff_start + resource->cliff_count;
  while (record->next_cliff_index < cliff_end) {
    const loom_target_residency_cliff_t* cliff =
        &state->pressure_resources->cliffs[record->next_cliff_index];
    if (cliff->cliff_units > record->current_peak_units) break;
    if (mode == LOOM_LOW_SCHEDULE_RESOURCE_HIGH_WATER_SCHEDULED) {
      const uint32_t penalty = cliff->tier_before - cliff->tier_after;
      record->pressure_cliff_penalty =
          iree_math_saturating_add_u32(record->pressure_cliff_penalty, penalty);
      pressure_state->resources.pressure_cliff_penalty =
          iree_math_saturating_add_u32(
              pressure_state->resources.pressure_cliff_penalty, penalty);
    }
    ++record->next_cliff_index;
  }
}

static void loom_low_schedule_note_resource_high_water(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint16_t reg_class_id,
    loom_low_schedule_resource_high_water_mode_t mode) {
  if (state->pressure_resources == NULL) return;
  const uint64_t current_live_units =
      pressure_state->current_live_units_by_reg_class[reg_class_id];
  uint64_t* peak_live_units =
      &pressure_state->resources.peak_live_units_by_reg_class[reg_class_id];
  if (current_live_units <= *peak_live_units) return;
  const uint64_t previous_peak_live_units = *peak_live_units;
  *peak_live_units = current_live_units;
  const loom_target_residency_derived_member_range_t range =
      loom_target_residency_derived_resource_member_range(
          state->pressure_resources, reg_class_id);
  for (uint16_t i = 0; i < range.count; ++i) {
    const uint16_t member_index =
        state->pressure_resources
            ->member_indices_by_direct_resource[range.start + i];
    const loom_target_residency_derived_member_t* member =
        &state->pressure_resources->members[member_index];
    IREE_ASSERT_EQ(member->direct_resource_id, reg_class_id);
    const uint64_t previous_contribution =
        loom_target_residency_round_resource_units(
            previous_peak_live_units, member->contribution_granularity);
    const uint64_t current_contribution =
        loom_target_residency_round_resource_units(
            current_live_units, member->contribution_granularity);
    loom_low_schedule_resource_pressure_record_t* record =
        &pressure_state->resources.records[member->resource_id];
    record->current_peak_units = iree_math_saturating_add_u64(
        record->current_peak_units,
        current_contribution - previous_contribution);
    loom_low_schedule_advance_resource_cliffs(state, pressure_state,
                                              member->resource_id, mode);
  }
}

static void loom_low_schedule_reset_source_resource_pressure(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state) {
  if (state->pressure_resources == NULL) return;
  memset(pressure_state->resources.peak_live_units_by_reg_class, 0,
         state->target.descriptor_set->reg_class_count *
             sizeof(*pressure_state->resources.peak_live_units_by_reg_class));
  for (uint16_t resource_id = 0;
       resource_id < state->pressure_resources->resource_count; ++resource_id) {
    loom_low_schedule_resource_pressure_record_t* record =
        &pressure_state->resources.records[resource_id];
    record->current_peak_units = 0;
    record->candidate_added_units = 0;
    record->pressure_cliff_penalty = 0;
    record->flags = 0;
  }
  pressure_state->resources.candidate_touched_count = 0;
  pressure_state->resources.pressure_cliff_penalty = 0;
}

static void loom_low_schedule_advance_source_pressure_cliff_floor(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint16_t reg_class_id) {
  if (pressure_state->first_actionable_pressure_cliff_indices == NULL) {
    return;
  }
  const loom_target_residency_cliff_range_t range =
      loom_target_residency_direct_resource_cliff_range(state->pressure_cliffs,
                                                        reg_class_id);
  const uint32_t range_end = range.start + range.count;
  uint32_t* first_actionable_cliff =
      &pressure_state->first_actionable_pressure_cliff_indices[reg_class_id];
  const uint64_t source_live_units =
      pressure_state->current_live_units_by_reg_class[reg_class_id];
  while (*first_actionable_cliff < range_end &&
         state->pressure_cliffs->cliffs[*first_actionable_cliff].cliff_units <=
             source_live_units) {
    ++*first_actionable_cliff;
  }
}

static uint32_t loom_low_schedule_remove_live_pressure_value(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t value_ordinal) {
  loom_low_schedule_value_record_t* value = &state->values[value_ordinal];
  if (!iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
    return 0;
  }
  const uint32_t transfer_units =
      loom_low_schedule_pressure_alias_transfer_from_source(
          state, pressure_state, value_ordinal);
  IREE_ASSERT_LE(transfer_units, value->live_unit_count);
  const uint32_t unit_count = value->live_unit_count - transfer_units;
  value->flags &= ~LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
  value->live_unit_count = 0;
  loom_low_schedule_pressure_alias_deactivate_result(state, pressure_state,
                                                     value_ordinal);
  IREE_ASSERT_LE(unit_count, pressure_state->current_live_units);
  pressure_state->current_live_units -= unit_count;
  const uint16_t reg_class_id = value->register_class_id;
  if (reg_class_id != LOOM_LOW_REG_CLASS_NONE) {
    IREE_ASSERT_LE(
        unit_count,
        pressure_state->current_live_units_by_reg_class[reg_class_id]);
    pressure_state->current_live_units_by_reg_class[reg_class_id] -= unit_count;
    loom_low_schedule_adjust_alias_pressure(state, pressure_state, reg_class_id,
                                            -(int64_t)unit_count);
  }
  return unit_count;
}

static void loom_low_schedule_add_source_pressure_value(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_block_t* block, loom_value_ordinal_t value_ordinal) {
  loom_low_schedule_value_record_t* value = &state->values[value_ordinal];
  if (iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE) ||
      value->unit_count == 0) {
    return;
  }
  const uint32_t source_owned_units =
      loom_low_schedule_pressure_alias_append_source_baseline_result(
          state, pressure_state, block, value_ordinal);
  IREE_ASSERT_LE(source_owned_units, value->unit_count);
  value->flags |= LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
  value->live_unit_count = value->unit_count - source_owned_units;
  pressure_state->block_value_ordinals[pressure_state->block_value_count++] =
      value_ordinal;
  uint32_t transferred_units = 0;
  if (value->live_unit_count == value->unit_count &&
      !iree_any_bit_set(value->flags,
                        LOOM_LOW_SCHEDULE_VALUE_FLAG_ACTIVE_PRESSURE_ALIAS)) {
    transferred_units = loom_low_schedule_pressure_alias_transfer_to_source(
        state, pressure_state, value_ordinal);
  }
  IREE_ASSERT_LE(transferred_units, value->live_unit_count);
  const uint32_t added_units = value->live_unit_count - transferred_units;
  pressure_state->current_live_units += added_units;
  const uint16_t reg_class_id = value->register_class_id;
  if (reg_class_id == LOOM_LOW_REG_CLASS_NONE) {
    return;
  }
  loom_low_schedule_note_block_pressure_reg_class(pressure_state, reg_class_id);
  pressure_state->current_live_units_by_reg_class[reg_class_id] += added_units;
  loom_low_schedule_adjust_alias_pressure(state, pressure_state, reg_class_id,
                                          (int64_t)added_units);
  loom_low_schedule_advance_source_pressure_cliff_floor(state, pressure_state,
                                                        reg_class_id);
  loom_low_schedule_note_resource_high_water(
      state, pressure_state, reg_class_id,
      LOOM_LOW_SCHEDULE_RESOURCE_HIGH_WATER_SOURCE_BASELINE);
}

static void loom_low_schedule_reverse_source_pressure_node(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_node_t* node) {
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t result_index = 0; result_index < node->result_count;
       ++result_index) {
    loom_low_schedule_remove_live_pressure_value(state, pressure_state,
                                                 result_ordinals[result_index]);
  }
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  for (uint16_t operand_index = 0; operand_index < node->operand_count;
       ++operand_index) {
    loom_low_schedule_add_source_pressure_value(
        state, pressure_state, node->block, operand_ordinals[operand_index]);
  }
}

static void loom_low_schedule_remove_source_pressure_block_arguments(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_block_t* block) {
  for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
    const loom_value_ordinal_t value_ordinal = loom_local_value_domain_ordinal(
        state->value_domain, loom_block_arg_id(block, arg_index));
    loom_low_schedule_remove_live_pressure_value(state, pressure_state,
                                                 value_ordinal);
  }
}

static void loom_low_schedule_reset_source_pressure_sweep(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state) {
  for (iree_host_size_t i = 0; i < pressure_state->block_value_count; ++i) {
    loom_low_schedule_value_record_t* value =
        &state->values[pressure_state->block_value_ordinals[i]];
    value->flags &= ~(LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE |
                      LOOM_LOW_SCHEDULE_VALUE_FLAG_ACTIVE_PRESSURE_ALIAS);
    value->live_unit_count = 0;
  }
  pressure_state->block_value_count = 0;
  for (iree_host_size_t i = 0; i < pressure_state->block_reg_class_count; ++i) {
    const uint16_t reg_class_id = pressure_state->block_reg_class_ids[i];
    pressure_state->block_reg_class_touched_flags[reg_class_id] = 0;
    pressure_state->current_live_units_by_reg_class[reg_class_id] = 0;
    pressure_state->packing_reserve_units_by_reg_class[reg_class_id] = 0;
  }
  pressure_state->block_reg_class_count = 0;
  loom_low_schedule_reset_block_alias_pressure(pressure_state);
  loom_low_schedule_reset_source_resource_pressure(state, pressure_state);
  loom_low_schedule_pressure_alias_reset(&pressure_state->storage_aliases);
  pressure_state->current_live_units = 0;
}

static void loom_low_schedule_reset_candidate_operand_uses(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state) {
  for (iree_host_size_t i = 0; i < pressure_state->candidate_operand_count;
       ++i) {
    const loom_value_ordinal_t value_ordinal =
        pressure_state->candidate_operand_ordinals[i];
    pressure_state->candidate_operand_use_counts[value_ordinal] = 0;
    pressure_state->candidate_scratch_counts[value_ordinal] = 0;
    state->values[value_ordinal].flags &=
        ~LOOM_LOW_SCHEDULE_VALUE_FLAG_CANDIDATE_ALIAS_CLAIM;
  }
  pressure_state->candidate_operand_count = 0;
}

static void loom_low_schedule_note_candidate_operand_use(
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t value_ordinal) {
  uint16_t* use_count =
      &pressure_state->candidate_operand_use_counts[value_ordinal];
  if (*use_count == 0) {
    pressure_state->candidate_operand_ordinals
        [pressure_state->candidate_operand_count++] = value_ordinal;
  }
  ++*use_count;
}

static uint32_t loom_low_schedule_saturate_u64_to_u32(uint64_t value) {
  return value > UINT32_MAX ? UINT32_MAX : (uint32_t)value;
}

static uint64_t loom_low_schedule_ready_pressure_key(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  uint64_t killed_units = 0;
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  for (uint16_t i = 0; i < node->operand_count; ++i) {
    loom_low_schedule_note_candidate_operand_use(pressure_state,
                                                 operand_ordinals[i]);
  }
  for (iree_host_size_t i = 0; i < pressure_state->candidate_operand_count;
       ++i) {
    const loom_value_ordinal_t value_ordinal =
        pressure_state->candidate_operand_ordinals[i];
    const loom_low_schedule_value_record_t* value =
        &state->values[value_ordinal];
    if (iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE) &&
        value->remaining_use_count ==
            pressure_state->candidate_operand_use_counts[value_ordinal]) {
      killed_units += value->live_unit_count;
    }
  }
  loom_low_schedule_reset_candidate_operand_uses(state, pressure_state);

  uint64_t produced_units = 0;
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t i = 0; i < node->result_count; ++i) {
    const loom_low_schedule_value_record_t* value =
        &state->values[result_ordinals[i]];
    if (value->remaining_use_count != 0 &&
        !iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
      produced_units += value->unit_count;
    }
  }
  const uint32_t growth = loom_low_schedule_saturate_u64_to_u32(
      produced_units > killed_units ? produced_units - killed_units : 0);
  const uint32_t relief = loom_low_schedule_saturate_u64_to_u32(
      killed_units > produced_units ? killed_units - produced_units : 0);
  return ((uint64_t)growth << 32) | (uint64_t)(UINT32_MAX - relief);
}

static uint64_t loom_low_schedule_ready_schedule_key(
    const loom_low_schedule_build_state_t* state, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  switch (state->options->strategy) {
    case LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL: {
      const uint32_t critical_path =
          state->node_critical_path_cycles != NULL
              ? state->node_critical_path_cycles[node_index]
              : 0;
      return UINT32_MAX - critical_path;
    }
    case LOOM_LOW_SCHEDULE_STRATEGY_LATENCY_HIDING: {
      const uint16_t dependency_latency =
          state->node_dependency_latency_cycles != NULL
              ? state->node_dependency_latency_cycles[node_index]
              : 0;
      const uint16_t latency = node->schedule_class != NULL
                                   ? node->schedule_class->latency_cycles
                                   : 0;
      return ((uint64_t)dependency_latency << 32) |
             (uint64_t)(UINT16_MAX - latency);
    }
    default:
      return node->source_ordinal;
  }
}

static uint64_t loom_low_schedule_ready_storage_key(
    const loom_low_schedule_build_state_t* state, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  const uint16_t relation_count = node->storage_relation_count;
  return relation_count == 0 ? UINT64_MAX
                             : (uint64_t)(UINT16_MAX - relation_count);
}

loom_low_schedule_ready_keys_t loom_low_schedule_pressure_ready_keys(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  return (loom_low_schedule_ready_keys_t){
      .values =
          {
              [LOOM_LOW_SCHEDULE_READY_VIEW_SOURCE] = node->source_ordinal,
              [LOOM_LOW_SCHEDULE_READY_VIEW_PRESSURE] =
                  loom_low_schedule_ready_pressure_key(state, pressure_state,
                                                       node_index),
              [LOOM_LOW_SCHEDULE_READY_VIEW_SCHEDULE] =
                  loom_low_schedule_ready_schedule_key(state, node_index),
              [LOOM_LOW_SCHEDULE_READY_VIEW_STORAGE] =
                  loom_low_schedule_ready_storage_key(state, node_index),
          },
  };
}

static void loom_low_schedule_note_block_pressure_use(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t value_ordinal, uint16_t use_count) {
  loom_low_schedule_value_record_t* value = &state->values[value_ordinal];
  IREE_ASSERT_LE(use_count, UINT32_MAX - value->remaining_use_count);
  if (value->remaining_use_count == 0) {
    pressure_state->block_value_ordinals[pressure_state->block_value_count++] =
        value_ordinal;
    const uint16_t reg_class_id = value->register_class_id;
    if (reg_class_id != LOOM_LOW_REG_CLASS_NONE) {
      const uint32_t packing_reserve_units =
          loom_low_register_unit_alignment(value->unit_count) - 1u;
      loom_low_schedule_note_block_pressure_reg_class(pressure_state,
                                                      reg_class_id);
      uint32_t* reg_class_reserve =
          &pressure_state->packing_reserve_units_by_reg_class[reg_class_id];
      *reg_class_reserve = iree_max(*reg_class_reserve, packing_reserve_units);
      const uint16_t alias_set_id =
          loom_low_schedule_alias_set_id(state, reg_class_id);
      if (alias_set_id != 0) {
        loom_low_schedule_note_block_pressure_alias_set(pressure_state,
                                                        alias_set_id);
        loom_low_schedule_alias_pressure_record_t* alias_record =
            &pressure_state->alias_sets.records[alias_set_id];
        alias_record->packing_reserve_units = iree_max(
            alias_record->packing_reserve_units, packing_reserve_units);
      }
    }
  }
  value->remaining_use_count += use_count;
}

void loom_low_schedule_pressure_initialize_block(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_block_t* block_record,
    loom_low_schedule_pressure_state_t* pressure_state) {
  pressure_state->current_live_units = 0;
  for (iree_host_size_t i = 0; i < pressure_state->block_reg_class_count; ++i) {
    const uint16_t reg_class_id = pressure_state->block_reg_class_ids[i];
    pressure_state->block_reg_class_touched_flags[reg_class_id] = 0;
    pressure_state->packing_reserve_units_by_reg_class[reg_class_id] = 0;
  }
  pressure_state->block_reg_class_count = 0;
  loom_low_schedule_reset_block_alias_pressure(pressure_state);
  loom_low_schedule_pressure_alias_reset(&pressure_state->storage_aliases);
  for (iree_host_size_t i = 0; i < pressure_state->block_value_count; ++i) {
    loom_value_ordinal_t ordinal = pressure_state->block_value_ordinals[i];
    pressure_state->remaining_consumer_counts[ordinal] = 0;
    pressure_state->remaining_consumer_node_xors[ordinal] = 0;
    state->values[ordinal].remaining_use_count = 0;
    state->values[ordinal].live_unit_count = 0;
    state->values[ordinal].flags &=
        ~(LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE |
          LOOM_LOW_SCHEDULE_VALUE_FLAG_ACTIVE_PRESSURE_ALIAS);
  }
  pressure_state->block_value_count = 0;
  if (pressure_state->current_live_units_by_reg_class) {
    memset(pressure_state->current_live_units_by_reg_class, 0,
           state->target.descriptor_set->reg_class_count *
               sizeof(*pressure_state->current_live_units_by_reg_class));
  }

  const uint32_t block_node_end =
      block_record->node_start + block_record->node_count;
  for (uint32_t node_index = block_record->node_start;
       node_index < block_node_end; ++node_index) {
    const loom_low_schedule_node_t* node = &state->nodes[node_index];
    const loom_value_ordinal_t* operand_ordinals =
        loom_low_schedule_node_const_operand_ordinals(node);
    for (uint16_t operand_index = 0; operand_index < node->operand_count;
         ++operand_index) {
      loom_low_schedule_note_candidate_operand_use(
          pressure_state, operand_ordinals[operand_index]);
    }
    for (iree_host_size_t i = 0; i < pressure_state->candidate_operand_count;
         ++i) {
      const loom_value_ordinal_t value_ordinal =
          pressure_state->candidate_operand_ordinals[i];
      IREE_ASSERT_NE(pressure_state->remaining_consumer_counts[value_ordinal],
                     UINT32_MAX);
      ++pressure_state->remaining_consumer_counts[value_ordinal];
      pressure_state->remaining_consumer_node_xors[value_ordinal] ^= node_index;
      loom_low_schedule_note_block_pressure_use(
          state, pressure_state, value_ordinal,
          pressure_state->candidate_operand_use_counts[value_ordinal]);
    }
    loom_low_schedule_reset_candidate_operand_uses(state, pressure_state);
  }

  for (iree_host_size_t i = 0; i < pressure_state->block_value_count; ++i) {
    loom_low_schedule_value_record_t* value =
        &state->values[pressure_state->block_value_ordinals[i]];
    if (value->remaining_use_count == 0) {
      continue;
    }
    const uint32_t producer_node = value->producer_node;
    if (producer_node != LOOM_LOW_SCHEDULE_NODE_NONE &&
        state->nodes[producer_node].block == block_record->block) {
      continue;
    }
    const uint32_t unit_count = value->unit_count;
    if (unit_count == 0) {
      continue;
    }
    value->flags |= LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
    pressure_state->current_live_units += unit_count;
    value->live_unit_count = unit_count;
    const uint16_t reg_class_id = value->register_class_id;
    if (reg_class_id != LOOM_LOW_REG_CLASS_NONE &&
        pressure_state->current_live_units_by_reg_class) {
      loom_low_schedule_note_block_pressure_reg_class(pressure_state,
                                                      reg_class_id);
      pressure_state->current_live_units_by_reg_class[reg_class_id] +=
          unit_count;
      loom_low_schedule_adjust_alias_pressure(
          state, pressure_state, reg_class_id, (int64_t)unit_count);
      loom_low_schedule_note_resource_high_water(
          state, pressure_state, reg_class_id,
          LOOM_LOW_SCHEDULE_RESOURCE_HIGH_WATER_SCHEDULED);
    }
  }
}

static void loom_low_schedule_score_candidate_resources(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node, uint32_t prerequisite_stall_cycles,
    loom_low_schedule_candidate_score_t* score) {
  score->resource_stall_cycles = 0;
  score->bottleneck_resource_id = LOOM_LOW_RESOURCE_NONE;
  const loom_low_schedule_class_t* schedule_class = node->schedule_class;
  if (state->options->strategy != LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL ||
      schedule_class == NULL) {
    return;
  }
  const uint32_t proposed_issue_cycle = iree_math_saturating_add_u32(
      state->current_issue_cycle, prerequisite_stall_cycles);
  const uint32_t earliest_issue_cycle =
      loom_low_schedule_resource_calendar_find_earliest_issue_cycle(
          &state->resource_calendar, schedule_class, proposed_issue_cycle,
          &score->bottleneck_resource_id);
  score->resource_stall_cycles = loom_low_schedule_positive_delta_u32(
      earliest_issue_cycle, proposed_issue_cycle);
}

static uint32_t loom_low_schedule_min_distance_hazard_stall(
    const loom_low_schedule_build_state_t* state,
    const loom_low_hazard_t* hazard) {
  uint32_t stall_cycles = 0;
  for (iree_host_size_t i = 0; i < state->hazard_state_count; ++i) {
    const loom_low_schedule_hazard_state_t* hazard_state =
        &state->hazard_states[i];
    if (hazard_state->kind != hazard->kind ||
        hazard_state->reference_kind != hazard->reference_kind ||
        hazard_state->reference_id != hazard->reference_id ||
        hazard_state->block_index != state->current_block_index ||
        hazard_state->producer_stage != hazard->consumer_stage) {
      continue;
    }
    const uint16_t required_distance = hazard_state->distance > hazard->distance
                                           ? hazard_state->distance
                                           : hazard->distance;
    const uint32_t actual_distance =
        state->current_issue_cycle >= hazard_state->issue_cycle
            ? state->current_issue_cycle - hazard_state->issue_cycle
            : 0;
    if (actual_distance < required_distance) {
      const uint32_t required_stall = required_distance - actual_distance;
      if (required_stall > stall_cycles) {
        stall_cycles = required_stall;
      }
    }
  }
  return stall_cycles;
}

static void loom_low_schedule_score_candidate_hazards(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node, uint32_t node_index,
    loom_low_schedule_candidate_score_t* score) {
  score->completion_wait_cycles = 0;
  score->hazard_stall_cycles = 0;
  if (state->options->strategy != LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
    return;
  }
  if (state->node_completion_wait_cycles != NULL) {
    score->completion_wait_cycles =
        state->node_completion_wait_cycles[node_index];
  }
  const loom_low_schedule_class_t* schedule_class = node->schedule_class;
  if (schedule_class == NULL) return;
  for (uint16_t i = 0; i < schedule_class->hazard_count; ++i) {
    const loom_low_hazard_t* hazard =
        &state->target.descriptor_set
             ->hazards[schedule_class->hazard_start + i];
    if (hazard->kind != LOOM_LOW_HAZARD_KIND_MIN_DISTANCE) {
      continue;
    }
    score->hazard_stall_cycles =
        iree_max(score->hazard_stall_cycles,
                 loom_low_schedule_min_distance_hazard_stall(state, hazard));
  }
}

static void loom_low_schedule_reset_candidate_pressure_deltas(
    loom_low_schedule_pressure_state_t* pressure_state) {
  for (iree_host_size_t i = 0;
       i < pressure_state->candidate_delta_touched_count; ++i) {
    const uint16_t reg_class_id =
        pressure_state->candidate_delta_touched_reg_class_ids[i];
    pressure_state->candidate_delta_units_by_reg_class[reg_class_id] = 0;
    pressure_state->candidate_early_added_units_by_reg_class[reg_class_id] = 0;
    pressure_state->candidate_delta_touched_flags[reg_class_id] = 0;
  }
  pressure_state->candidate_delta_touched_count = 0;
  for (iree_host_size_t i = 0;
       i < pressure_state->alias_sets.candidate_delta_touched_count; ++i) {
    loom_low_schedule_alias_pressure_record_t* record =
        &pressure_state->alias_sets.records
             [pressure_state->alias_sets.candidate_delta_touched_ids[i]];
    record->candidate_delta_units = 0;
    record->candidate_early_added_units = 0;
    record->flags &= ~LOOM_LOW_SCHEDULE_ALIAS_PRESSURE_FLAG_CANDIDATE_TOUCHED;
  }
  pressure_state->alias_sets.candidate_delta_touched_count = 0;
  for (uint16_t i = 0; i < pressure_state->resources.candidate_touched_count;
       ++i) {
    loom_low_schedule_resource_pressure_record_t* record =
        &pressure_state->resources
             .records[pressure_state->resources.candidate_touched_ids[i]];
    record->candidate_added_units = 0;
    record->flags &=
        ~LOOM_LOW_SCHEDULE_RESOURCE_PRESSURE_FLAG_CANDIDATE_TOUCHED;
  }
  pressure_state->resources.candidate_touched_count = 0;
}

static void loom_low_schedule_note_candidate_pressure_delta(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint16_t reg_class_id,
    int64_t delta_units) {
  if (reg_class_id == LOOM_LOW_REG_CLASS_NONE || delta_units == 0 ||
      pressure_state->candidate_delta_units_by_reg_class == NULL) {
    return;
  }
  if (!pressure_state->candidate_delta_touched_flags[reg_class_id]) {
    pressure_state->candidate_delta_touched_flags[reg_class_id] = 1;
    pressure_state->candidate_delta_touched_reg_class_ids
        [pressure_state->candidate_delta_touched_count++] = reg_class_id;
  }
  pressure_state->candidate_delta_units_by_reg_class[reg_class_id] +=
      delta_units;
  if (pressure_state->alias_sets.records == NULL) {
    return;
  }
  const uint16_t alias_set_id =
      loom_low_schedule_alias_set_id(state, reg_class_id);
  if (alias_set_id == 0) {
    return;
  }
  loom_low_schedule_alias_pressure_record_t* record =
      &pressure_state->alias_sets.records[alias_set_id];
  if (!iree_any_bit_set(
          record->flags,
          LOOM_LOW_SCHEDULE_ALIAS_PRESSURE_FLAG_CANDIDATE_TOUCHED)) {
    record->flags |= LOOM_LOW_SCHEDULE_ALIAS_PRESSURE_FLAG_CANDIDATE_TOUCHED;
    pressure_state->alias_sets.candidate_delta_touched_ids
        [pressure_state->alias_sets.candidate_delta_touched_count++] =
        alias_set_id;
  }
  record->candidate_delta_units += delta_units;
}

static void loom_low_schedule_note_candidate_early_added_pressure(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint16_t reg_class_id,
    uint32_t unit_count) {
  if (reg_class_id == LOOM_LOW_REG_CLASS_NONE || unit_count == 0 ||
      pressure_state->candidate_early_added_units_by_reg_class == NULL) {
    return;
  }
  IREE_ASSERT(pressure_state->candidate_delta_touched_flags[reg_class_id]);
  pressure_state->candidate_early_added_units_by_reg_class[reg_class_id] =
      iree_math_saturating_add_u64(
          pressure_state
              ->candidate_early_added_units_by_reg_class[reg_class_id],
          unit_count);
  if (pressure_state->alias_sets.records == NULL) {
    return;
  }
  const uint16_t alias_set_id =
      loom_low_schedule_alias_set_id(state, reg_class_id);
  if (alias_set_id == 0) {
    return;
  }
  loom_low_schedule_alias_pressure_record_t* record =
      &pressure_state->alias_sets.records[alias_set_id];
  IREE_ASSERT(iree_any_bit_set(
      record->flags, LOOM_LOW_SCHEDULE_ALIAS_PRESSURE_FLAG_CANDIDATE_TOUCHED));
  record->candidate_early_added_units = iree_math_saturating_add_u64(
      record->candidate_early_added_units, unit_count);
}

static uint32_t loom_low_schedule_candidate_early_added_units(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node, uint16_t result_index,
    uint32_t unit_count) {
  const loom_low_operand_t* result_operand =
      &state->target.descriptor_set
           ->operands[node->descriptor->operand_start + result_index];
  if (!iree_any_bit_set(result_operand->flags,
                        LOOM_LOW_OPERAND_FLAG_EARLY_CLOBBER)) {
    return 0;
  }
  return iree_any_bit_set(result_operand->flags, LOOM_LOW_OPERAND_FLAG_TIED)
             ? 0
             : unit_count;
}

void loom_low_schedule_pressure_initialize_current_cliff_penalty(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state) {
  loom_low_schedule_reset_candidate_pressure_deltas(pressure_state);
  loom_low_schedule_candidate_score_t score = {
      .pressure_cliff_units = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
      .units_until_pressure_cliff = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
  };
  loom_low_schedule_target_pressure_score_candidate(state, pressure_state,
                                                    &score);
  pressure_state->current_persistent_pressure_penalty =
      score.persistent_pressure_cliff_penalty;
}

static bool loom_low_schedule_descriptor_frontier_is_non_growing(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    uint32_t candidate_node_index, uint16_t candidate_storage_relation_count,
    uint64_t candidate_killed_units, uint64_t candidate_produced_units,
    const uint32_t* consumer_nodes, iree_host_size_t consumer_count) {
  const iree_host_size_t candidate_operand_count =
      pressure_state->candidate_operand_count;
  for (iree_host_size_t i = 0; i < candidate_operand_count; ++i) {
    const loom_value_ordinal_t value_ordinal =
        pressure_state->candidate_operand_ordinals[i];
    pressure_state->candidate_scratch_counts[value_ordinal] = 0;
  }

  for (iree_host_size_t i = 0; i < consumer_count; ++i) {
    const loom_low_schedule_node_t* consumer = &state->nodes[consumer_nodes[i]];
    const loom_value_ordinal_t* operand_ordinals =
        loom_low_schedule_node_const_operand_ordinals(consumer);
    for (uint16_t operand_index = 0; operand_index < consumer->operand_count;
         ++operand_index) {
      const loom_value_ordinal_t value_ordinal =
          operand_ordinals[operand_index];
      uint32_t* consumer_use_count =
          &pressure_state->candidate_scratch_counts[value_ordinal];
      if ((*consumer_use_count)++ == 0 &&
          pressure_state->candidate_operand_use_counts[value_ordinal] == 0) {
        pressure_state->candidate_operand_ordinals
            [pressure_state->candidate_operand_count++] = value_ordinal;
      }
    }
  }

  uint64_t killed_units = 0;
  for (iree_host_size_t i = 0; i < pressure_state->candidate_operand_count;
       ++i) {
    const loom_value_ordinal_t value_ordinal =
        pressure_state->candidate_operand_ordinals[i];
    const uint32_t consumer_use_count =
        pressure_state->candidate_scratch_counts[value_ordinal];
    if (consumer_use_count == 0) {
      continue;
    }
    const loom_low_schedule_value_record_t* value =
        &state->values[value_ordinal];
    const uint32_t candidate_use_count =
        pressure_state->candidate_operand_use_counts[value_ordinal];
    IREE_ASSERT_GE(value->remaining_use_count, candidate_use_count);
    const uint32_t remaining_use_count =
        value->remaining_use_count - candidate_use_count;
    IREE_ASSERT_GE(remaining_use_count, consumer_use_count);
    if (remaining_use_count != consumer_use_count) {
      continue;
    }
    if (candidate_storage_relation_count != 0) {
      continue;
    }
    if (iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
      killed_units += value->live_unit_count;
    } else if (value->producer_node == candidate_node_index) {
      killed_units += value->unit_count;
    }
  }

  uint64_t produced_units = 0;
  for (iree_host_size_t i = 0; i < consumer_count; ++i) {
    const loom_low_schedule_node_t* consumer = &state->nodes[consumer_nodes[i]];
    const loom_value_ordinal_t* result_ordinals =
        loom_low_schedule_node_const_result_ordinals(consumer);
    for (uint16_t result_index = 0; result_index < consumer->result_count;
         ++result_index) {
      const loom_low_schedule_value_record_t* value =
          &state->values[result_ordinals[result_index]];
      if (value->remaining_use_count != 0) {
        produced_units += value->unit_count;
      }
    }
  }

  for (iree_host_size_t i = 0; i < pressure_state->candidate_operand_count;
       ++i) {
    const loom_value_ordinal_t value_ordinal =
        pressure_state->candidate_operand_ordinals[i];
    pressure_state->candidate_scratch_counts[value_ordinal] = 0;
  }
  pressure_state->candidate_operand_count = candidate_operand_count;
  const uint64_t total_killed_units =
      iree_math_saturating_add_u64(candidate_killed_units, killed_units);
  const uint64_t total_produced_units =
      iree_math_saturating_add_u64(candidate_produced_units, produced_units);
  return total_produced_units <= total_killed_units;
}

void loom_low_schedule_pressure_publish_unlock_consumer(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t producer_node,
    uint32_t consumer_node) {
  if (state->nodes[producer_node].block_index !=
      state->nodes[consumer_node].block_index) {
    return;
  }
  loom_low_schedule_unlock_record_t* record =
      &pressure_state->unlocks.records[producer_node];
  record->demand_units = iree_math_saturating_add_u32(
      record->demand_units, state->node_pressure_demand_units[consumer_node]);
  record->activation_units =
      iree_max(record->activation_units,
               state->node_pressure_activation_units[consumer_node]);
  if (state->nodes[consumer_node].descriptor != NULL) {
    record->descriptor_latency_cycles =
        iree_max(record->descriptor_latency_cycles,
                 state->nodes[consumer_node].schedule_class->latency_cycles);
    record->candidate_flags |=
        LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_DESCRIPTOR;
    if (pressure_state->unlocks.descriptor_heads != NULL &&
        record->descriptor_count <
            LOOM_LOW_SCHEDULE_DESCRIPTOR_FRONTIER_CAPACITY) {
      pressure_state->unlocks.descriptor_next_nodes[consumer_node] =
          pressure_state->unlocks.descriptor_heads[producer_node];
      pressure_state->unlocks.descriptor_heads[producer_node] = consumer_node;
    }
    if (record->descriptor_count <=
        LOOM_LOW_SCHEDULE_DESCRIPTOR_FRONTIER_CAPACITY) {
      ++record->descriptor_count;
    }
  }
  record->candidate_flags |=
      (uint8_t)(state->nodes[consumer_node].flags &
                LOOM_LOW_SCHEDULE_NODE_FLAG_DESCRIPTOR_SETUP);
  ++state->unlock_summary_publication_count;
}

iree_status_t loom_low_schedule_pressure_initialize_unlock_summaries(
    loom_low_schedule_build_state_t* state, uint32_t node_count,
    loom_low_schedule_pressure_state_t* pressure_state) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_dependency_frontier_initialize(
      &state->dependency_index, state->arena,
      &pressure_state->unlocks.frontier));
  if (node_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, node_count, sizeof(*pressure_state->unlocks.records),
      (void**)&pressure_state->unlocks.records));
  memset(pressure_state->unlocks.records, 0,
         node_count * sizeof(*pressure_state->unlocks.records));
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, node_count,
        sizeof(*pressure_state->unlocks.descriptor_heads),
        (void**)&pressure_state->unlocks.descriptor_heads));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, node_count,
        sizeof(*pressure_state->unlocks.descriptor_next_nodes),
        (void**)&pressure_state->unlocks.descriptor_next_nodes));
    memset(pressure_state->unlocks.descriptor_heads, 0xFF,
           node_count * sizeof(*pressure_state->unlocks.descriptor_heads));
  }
  for (uint32_t consumer_node = 0; consumer_node < node_count;
       ++consumer_node) {
    const uint32_t producer_node =
        loom_low_schedule_dependency_frontier_remaining_producer(
            &pressure_state->unlocks.frontier, consumer_node);
    if (producer_node != LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_NONE) {
      loom_low_schedule_pressure_publish_unlock_consumer(
          state, pressure_state, producer_node, consumer_node);
    }
  }
  return iree_ok_status();
}

static loom_low_schedule_pressure_demand_t
loom_low_schedule_score_candidate_pressure_demand(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_ready_policy_t* ready_policy,
    const uint32_t* indegrees, uint32_t node_index,
    uint16_t storage_relation_count, uint64_t killed_live_units,
    uint64_t produced_live_units) {
  loom_low_schedule_pressure_demand_t demand = {
      .demand_units = state->node_pressure_demand_units != NULL
                          ? state->node_pressure_demand_units[node_index]
                          : 1,
      .activation_units =
          state->node_pressure_activation_units != NULL
              ? state->node_pressure_activation_units[node_index]
              : 1,
  };
  const loom_low_schedule_unlock_record_t* unlock_record =
      &pressure_state->unlocks.records[node_index];
  demand.demand_units =
      iree_max(demand.demand_units, unlock_record->demand_units);
  demand.activation_units =
      iree_max(demand.activation_units, unlock_record->activation_units);
  demand.candidate_flags = unlock_record->candidate_flags;
  if (pressure_state->unlocks.descriptor_heads != NULL &&
      unlock_record->descriptor_count != 0 &&
      unlock_record->descriptor_count <=
          LOOM_LOW_SCHEDULE_DESCRIPTOR_FRONTIER_CAPACITY) {
    uint32_t
        descriptor_consumers[LOOM_LOW_SCHEDULE_DESCRIPTOR_FRONTIER_CAPACITY];
    uint32_t consumer_node =
        pressure_state->unlocks.descriptor_heads[node_index];
    for (uint8_t i = 0; i < unlock_record->descriptor_count; ++i) {
      IREE_ASSERT_NE(consumer_node, LOOM_LOW_SCHEDULE_NODE_NONE);
      descriptor_consumers[i] = consumer_node;
      consumer_node =
          pressure_state->unlocks.descriptor_next_nodes[consumer_node];
    }
    if (loom_low_schedule_descriptor_frontier_is_non_growing(
            state, pressure_state, node_index, storage_relation_count,
            killed_live_units, produced_live_units, descriptor_consumers,
            unlock_record->descriptor_count)) {
      demand.candidate_flags |=
          LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_NON_GROWING_DESCRIPTOR;
      demand.non_growing_descriptor_latency_cycles =
          unlock_record->descriptor_latency_cycles;
    }
  }
  demand.pair_affinity_score =
      loom_low_schedule_ready_policy_score_setup_unlocks(state, ready_policy,
                                                         indegrees, node_index);
  return demand;
}

static loom_low_schedule_pressure_progress_kind_t
loom_low_schedule_classify_candidate_pressure_progress(
    const loom_low_schedule_candidate_score_t* score) {
  const bool unlocks_descriptor = iree_any_bit_set(
      score->flags, LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_DESCRIPTOR);
  const bool unlocks_non_growing_descriptor = iree_any_bit_set(
      score->flags,
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_NON_GROWING_DESCRIPTOR);
  const bool has_register_activity = score->killed_live_value_count != 0 ||
                                     score->produced_live_value_count != 0;
  if (score->killed_live_units > score->produced_live_units) {
    return LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_REDUCTION;
  }
  const bool is_non_growing =
      score->killed_live_units >= score->produced_live_units;
  if (score->storage_relation_count != 0 && is_non_growing) {
    return LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_STORAGE;
  }
  if (unlocks_descriptor && unlocks_non_growing_descriptor) {
    return LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_FRONTIER;
  }
  if (has_register_activity && is_non_growing) {
    return LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_NON_GROWING;
  }
  return LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_NONE;
}

static loom_low_schedule_pressure_risk_t
loom_low_schedule_classify_candidate_pressure_risk(
    const loom_low_schedule_candidate_score_t* score,
    uint32_t current_persistent_pressure_penalty) {
  if (score->pressure_cliff_penalty > current_persistent_pressure_penalty) {
    return LOOM_LOW_SCHEDULE_PRESSURE_RISK_DEBT;
  }
  if (score->units_until_pressure_cliff !=
          LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE &&
      score->units_until_pressure_cliff <=
          iree_max(score->activation_reserve_units,
                   score->pressure_demand_units)) {
    return LOOM_LOW_SCHEDULE_PRESSURE_RISK_NEAR_CLIFF;
  }
  return LOOM_LOW_SCHEDULE_PRESSURE_RISK_NONE;
}

void loom_low_schedule_pressure_score_candidate(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_ready_policy_t* ready_policy,
    const uint32_t* indegrees, uint32_t node_index,
    loom_low_schedule_candidate_score_t* out_score) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  loom_low_schedule_reset_candidate_pressure_deltas(pressure_state);
  uint64_t killed_live_units = 0;
  uint32_t killed_live_value_count = 0;
  uint64_t produced_live_units = 0;
  uint32_t produced_live_value_count = 0;
  const uint16_t storage_relation_count = node->storage_relation_count;
  const bool has_early_clobber =
      iree_any_bit_set(node->flags, LOOM_LOW_SCHEDULE_NODE_FLAG_EARLY_CLOBBER);

  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  for (uint16_t operand_index = 0; operand_index < node->operand_count;
       ++operand_index) {
    const loom_value_ordinal_t value_ordinal = operand_ordinals[operand_index];
    loom_low_schedule_note_candidate_operand_use(pressure_state, value_ordinal);
  }
  for (iree_host_size_t i = 0; i < pressure_state->candidate_operand_count;
       ++i) {
    const loom_value_ordinal_t value_ordinal =
        pressure_state->candidate_operand_ordinals[i];
    const loom_low_schedule_value_record_t* value =
        &state->values[value_ordinal];
    if (!iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
      continue;
    }
    const uint32_t candidate_use_count =
        pressure_state->candidate_operand_use_counts[value_ordinal];
    if (value->remaining_use_count != candidate_use_count) {
      continue;
    }
    loom_low_schedule_pressure_alias_note_candidate_result_releases(
        state, pressure_state, value_ordinal);
    const uint32_t transfer_units =
        loom_low_schedule_pressure_alias_candidate_transfer_from_source(
            state, pressure_state, value_ordinal);
    IREE_ASSERT(transfer_units <= value->live_unit_count,
                "transferred alias units must be covered by source pressure");
    const uint32_t unit_count = value->live_unit_count - transfer_units;
    killed_live_units += unit_count;
    ++killed_live_value_count;
    loom_low_schedule_note_candidate_pressure_delta(
        state, pressure_state, value->register_class_id, -(int64_t)unit_count);
  }
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t result_index = 0; result_index < node->result_count;
       ++result_index) {
    const loom_low_schedule_value_record_t* value =
        &state->values[result_ordinals[result_index]];
    if (value->remaining_use_count == 0) {
      continue;
    }
    if (iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
      continue;
    }
    const uint32_t alias_units =
        loom_low_schedule_pressure_alias_candidate_result_units(
            state, pressure_state, node_index, result_ordinals[result_index]);
    const uint32_t unit_count = value->unit_count - alias_units;
    produced_live_units += unit_count;
    if (unit_count != 0) {
      ++produced_live_value_count;
    }
    loom_low_schedule_note_candidate_pressure_delta(
        state, pressure_state, value->register_class_id, (int64_t)unit_count);
    if (has_early_clobber) {
      loom_low_schedule_note_candidate_early_added_pressure(
          state, pressure_state, value->register_class_id,
          loom_low_schedule_candidate_early_added_units(
              state, node, result_index, unit_count));
    }
  }
  IREE_ASSERT_LE(killed_live_units, pressure_state->current_live_units);
  uint64_t projected_live_units =
      pressure_state->current_live_units - killed_live_units;
  projected_live_units += produced_live_units;
  const loom_low_schedule_pressure_demand_t pressure_demand =
      loom_low_schedule_score_candidate_pressure_demand(
          state, pressure_state, ready_policy, indegrees, node_index,
          storage_relation_count, killed_live_units, produced_live_units);
  loom_low_schedule_reset_candidate_operand_uses(state, pressure_state);
  const uint16_t dependency_latency_cycles =
      state->node_dependency_latency_cycles != NULL
          ? state->node_dependency_latency_cycles[node_index]
          : 0;
  const bool is_storage_setup =
      iree_any_bit_set(node->flags, LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP);
  uint32_t data_ready_stall_cycles = 0;
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL &&
      state->node_ready_issue_cycles != NULL && !is_storage_setup) {
    data_ready_stall_cycles = loom_low_schedule_positive_delta_u32(
        state->node_ready_issue_cycles[node_index], state->current_issue_cycle);
  }
  uint16_t pair_placement_option_count = 0;
  const uint16_t direct_pair_affinity_score =
      loom_low_schedule_ready_policy_score_candidate_pair(
          state, node_index, &pair_placement_option_count);
  const uint16_t preferred_anchor_score =
      loom_low_schedule_ready_policy_preferred_anchor_priority(state, indegrees,
                                                               node_index);
  const uint16_t latency_cycles =
      node->schedule_class != NULL ? node->schedule_class->latency_cycles : 0;
  *out_score = (loom_low_schedule_candidate_score_t){
      .projected_live_units = projected_live_units,
      .killed_live_units = killed_live_units,
      .killed_live_value_count = killed_live_value_count,
      .produced_live_units = produced_live_units,
      .produced_live_value_count = produced_live_value_count,
      .dependency_latency_cycles = dependency_latency_cycles,
      .latency_cycles = latency_cycles,
      .unlocked_non_growing_descriptor_latency_cycles =
          pressure_demand.non_growing_descriptor_latency_cycles,
      .critical_path_cycles = state->node_critical_path_cycles != NULL
                                  ? state->node_critical_path_cycles[node_index]
                                  : latency_cycles,
      .pressure_demand_units = pressure_demand.demand_units,
      .activation_reserve_units =
          produced_live_units > killed_live_units &&
                  !iree_any_bit_set(
                      pressure_demand.candidate_flags,
                      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_DESCRIPTOR)
              ? pressure_demand.activation_units
              : 0,
      .data_ready_stall_cycles = data_ready_stall_cycles,
      .opened_completion_latency_cycles =
          state->node_opened_completion_latency_cycles != NULL
              ? state->node_opened_completion_latency_cycles[node_index]
              : 0,
      .pair_affinity_score =
          iree_max(preferred_anchor_score,
                   iree_max(direct_pair_affinity_score,
                            pressure_demand.pair_affinity_score)),
      .pair_placement_option_count = pair_placement_option_count,
      .storage_relation_count = storage_relation_count,
      .bottleneck_resource_id = LOOM_LOW_RESOURCE_NONE,
      .pressure_cliff_units = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
      .units_until_pressure_cliff = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
      .source_ordinal = node->source_ordinal,
      .flags =
          pressure_demand.candidate_flags |
          (uint8_t)((node->flags & LOOM_LOW_SCHEDULE_NODE_FLAG_PAIR_TRANSPARENT)
                    << 1u),
  };
  loom_low_schedule_target_pressure_score_candidate(state, pressure_state,
                                                    out_score);
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
    out_score->pressure_progress_kind =
        loom_low_schedule_classify_candidate_pressure_progress(out_score);
    out_score->pressure_risk =
        loom_low_schedule_classify_candidate_pressure_risk(
            out_score, pressure_state->current_persistent_pressure_penalty);
  }
  loom_low_schedule_score_candidate_hazards(state, node, node_index, out_score);
  const uint32_t prerequisite_stall_cycles =
      iree_max(out_score->data_ready_stall_cycles,
               iree_max(out_score->hazard_stall_cycles,
                        out_score->completion_wait_cycles));
  loom_low_schedule_score_candidate_resources(
      state, node, prerequisite_stall_cycles, out_score);
  out_score->effective_stall_cycles = iree_math_saturating_add_u32(
      prerequisite_stall_cycles, out_score->resource_stall_cycles);
}

void loom_low_schedule_pressure_note_node_scheduled(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t node_index,
    const loom_low_schedule_candidate_score_t* score) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  uint64_t live_units_before = pressure_state->current_live_units;
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  for (uint16_t operand_index = 0; operand_index < node->operand_count;
       ++operand_index) {
    loom_low_schedule_value_record_t* value =
        &state->values[operand_ordinals[operand_index]];
    if (value->remaining_use_count == 0) {
      continue;
    }
    --value->remaining_use_count;
    if (value->remaining_use_count == 0 &&
        iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
      loom_low_schedule_remove_live_pressure_value(
          state, pressure_state, operand_ordinals[operand_index]);
    }
  }

  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t result_index = 0; result_index < node->result_count;
       ++result_index) {
    loom_low_schedule_value_record_t* value =
        &state->values[result_ordinals[result_index]];
    if (value->remaining_use_count == 0) {
      continue;
    }
    if (iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
      continue;
    }
    value->flags |= LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
    const uint32_t alias_units =
        loom_low_schedule_pressure_alias_append_scheduled_result(
            state, pressure_state, node_index, result_ordinals[result_index]);
    const uint32_t unit_count = value->unit_count - alias_units;
    value->live_unit_count = unit_count;
    pressure_state->current_live_units += unit_count;
    const uint16_t reg_class_id = value->register_class_id;
    if (reg_class_id != LOOM_LOW_REG_CLASS_NONE &&
        pressure_state->current_live_units_by_reg_class) {
      loom_low_schedule_note_block_pressure_reg_class(pressure_state,
                                                      reg_class_id);
      pressure_state->current_live_units_by_reg_class[reg_class_id] +=
          unit_count;
      loom_low_schedule_adjust_alias_pressure(
          state, pressure_state, reg_class_id, (int64_t)unit_count);
      loom_low_schedule_note_resource_high_water(
          state, pressure_state, reg_class_id,
          LOOM_LOW_SCHEDULE_RESOURCE_HIGH_WATER_SCHEDULED);
    }
  }
  IREE_ASSERT_EQ(pressure_state->current_live_units,
                 score->projected_live_units);
  // The realized state excludes temporary overlap and downstream activation
  // headroom used only to prove that the candidate is physically schedulable.
  pressure_state->current_persistent_pressure_penalty =
      score->persistent_pressure_cliff_penalty;
  if (state->pressure_steps) {
    state->pressure_steps[state->pressure_step_count++] =
        (loom_low_schedule_pressure_step_t){
            .node_index = node_index,
            .block_index = node->block_index,
            .scheduled_ordinal = node->scheduled_ordinal,
            .issue_cycle = node->issue_cycle,
            .live_units_before = live_units_before,
            .killed_live_units = score->killed_live_units,
            .produced_live_units = score->produced_live_units,
            .live_units_after = pressure_state->current_live_units,
        };
  }
}

void loom_low_schedule_pressure_update_ready_consumers(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_ready_policy_t* ready_policy, uint32_t scheduled_node) {
  if (pressure_state->remaining_consumer_counts == NULL) return;
  const loom_low_schedule_node_t* node = &state->nodes[scheduled_node];
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  for (uint16_t operand_index = 0; operand_index < node->operand_count;
       ++operand_index) {
    loom_low_schedule_note_candidate_operand_use(
        pressure_state, operand_ordinals[operand_index]);
  }
  for (iree_host_size_t i = 0; i < pressure_state->candidate_operand_count;
       ++i) {
    const loom_value_ordinal_t value_ordinal =
        pressure_state->candidate_operand_ordinals[i];
    uint32_t* remaining_consumer_count =
        &pressure_state->remaining_consumer_counts[value_ordinal];
    uint32_t* remaining_consumer_node_xor =
        &pressure_state->remaining_consumer_node_xors[value_ordinal];
    IREE_ASSERT_NE(*remaining_consumer_count, 0u);
    --*remaining_consumer_count;
    *remaining_consumer_node_xor ^= scheduled_node;
    // A candidate can kill the value only when it is the sole distinct
    // consumer. The XOR identifies that consumer without walking the fan-out.
    if (*remaining_consumer_count == 1) {
      const uint32_t consumer_node = *remaining_consumer_node_xor;
      if (!loom_low_schedule_ready_frontier_contains(&ready_policy->frontier,
                                                     consumer_node)) {
        continue;
      }
      loom_low_schedule_ready_frontier_update_key(
          &ready_policy->frontier, LOOM_LOW_SCHEDULE_READY_VIEW_PRESSURE,
          consumer_node,
          loom_low_schedule_ready_pressure_key(state, pressure_state,
                                               consumer_node));
    }
  }
  loom_low_schedule_reset_candidate_operand_uses(state, pressure_state);
}

void loom_low_schedule_pressure_compute_node_priorities(
    loom_low_schedule_build_state_t* state, iree_host_size_t node_count,
    const loom_low_schedule_dependency_detail_index_t* dependency_details,
    loom_low_schedule_pressure_state_t* pressure_state) {
  if (state->node_critical_path_cycles == NULL &&
      state->node_dependency_latency_cycles == NULL &&
      state->node_opened_completion_latency_cycles == NULL &&
      state->node_pressure_demand_units == NULL &&
      state->node_pressure_activation_units == NULL &&
      pressure_state->first_actionable_pressure_cliff_indices == NULL) {
    return;
  }
  for (iree_host_size_t i = node_count; i > 0; --i) {
    const uint32_t node_index = (uint32_t)(i - 1);
    loom_low_schedule_node_t* node = &state->nodes[node_index];
    const bool is_storage_setup = iree_any_bit_set(
        node->flags, LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP);
    uint16_t dependency_latency_cycles = 0;
    if (state->node_dependency_latency_cycles != NULL) {
      const loom_value_ordinal_t* operand_ordinals =
          loom_low_schedule_node_const_operand_ordinals(node);
      for (uint16_t operand_index = 0; operand_index < node->operand_count;
           ++operand_index) {
        const uint32_t producer_node =
            state->values[operand_ordinals[operand_index]].producer_node;
        if (producer_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
            state->nodes[producer_node].block != node->block) {
          continue;
        }
        const loom_low_schedule_class_t* producer_schedule_class =
            state->nodes[producer_node].schedule_class;
        const uint16_t producer_latency =
            producer_schedule_class != NULL
                ? producer_schedule_class->latency_cycles
                : 0;
        dependency_latency_cycles =
            iree_max(dependency_latency_cycles, producer_latency);
      }
      state->node_dependency_latency_cycles[node_index] =
          dependency_latency_cycles;
    }
    uint32_t successor_path_cycles = 0;
    uint32_t pressure_demand_units = 0;
    uint32_t pressure_activation_units = 0;
    bool has_effect_consumer = false;
    if (dependency_details->dependency_count != 0) {
      const uint32_t dependency_begin =
          dependency_details->producer_dependency_starts[node_index];
      const uint32_t dependency_end =
          dependency_details->producer_dependency_starts[node_index + 1];
      for (uint32_t i = dependency_begin; i < dependency_end; ++i) {
        const uint32_t dependency_index =
            loom_low_schedule_dependency_detail_index_at(dependency_details, i);
        const loom_low_schedule_dependency_t* dependency =
            loom_low_schedule_dependency_graph_at(&state->dependencies,
                                                  dependency_index);
        if (dependency->producer_node != node_index ||
            dependency->consumer_node >= node_count) {
          continue;
        }
        const loom_low_schedule_node_t* consumer =
            &state->nodes[dependency->consumer_node];
        if (consumer->block_index != node->block_index) {
          continue;
        }
        has_effect_consumer |=
            dependency->kind == LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT;
        if (state->node_critical_path_cycles != NULL) {
          successor_path_cycles = iree_max(
              successor_path_cycles,
              state->node_critical_path_cycles[dependency->consumer_node]);
        }
        if (state->node_pressure_demand_units != NULL &&
            state->node_pressure_activation_units != NULL &&
            dependency->kind == LOOM_LOW_SCHEDULE_DEPENDENCY_SSA) {
          if (is_storage_setup &&
              (consumer->kind == LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR ||
               iree_any_bit_set(
                   consumer->flags,
                   LOOM_LOW_SCHEDULE_NODE_FLAG_DESCRIPTOR_SETUP))) {
            node->flags |= LOOM_LOW_SCHEDULE_NODE_FLAG_DESCRIPTOR_SETUP;
          }
          uint32_t consumer_demand =
              consumer->kind == LOOM_LOW_SCHEDULE_NODE_STRUCTURAL
                  ? state->node_pressure_demand_units[dependency->consumer_node]
                  : 0;
          if (consumer_demand == 0 &&
              dependency->value_operand_index < consumer->operand_count) {
            const loom_value_ordinal_t operand_ordinal =
                loom_low_schedule_node_const_operand_ordinals(
                    consumer)[dependency->value_operand_index];
            consumer_demand = state->values[operand_ordinal].unit_count;
          }
          if (consumer_demand == 0) {
            consumer_demand = 1;
          }
          pressure_demand_units = iree_math_saturating_add_u32(
              pressure_demand_units, consumer_demand);
          const uint32_t consumer_activation =
              consumer->kind == LOOM_LOW_SCHEDULE_NODE_STRUCTURAL
                  ? state->node_pressure_activation_units[dependency
                                                              ->consumer_node]
                  : consumer_demand;
          pressure_activation_units =
              iree_max(pressure_activation_units, consumer_activation);
        }
      }
    }
    if (state->node_critical_path_cycles != NULL) {
      const uint16_t latency_cycles = node->schedule_class != NULL
                                          ? node->schedule_class->latency_cycles
                                          : 0;
      state->node_critical_path_cycles[node_index] =
          iree_math_saturating_add_u32(latency_cycles, successor_path_cycles);
    }
    if (state->node_opened_completion_latency_cycles != NULL &&
        has_effect_consumer) {
      uint16_t completion_wait_cycles = 0;
      if (loom_low_schedule_class_query_completion_wait(
              state->target.descriptor_set, node->schedule_class,
              &completion_wait_cycles)) {
        state->node_opened_completion_latency_cycles[node_index] =
            node->schedule_class->latency_cycles;
      }
    }
    if (state->node_pressure_demand_units != NULL) {
      state->node_pressure_demand_units[node_index] =
          pressure_demand_units != 0 ? pressure_demand_units : 1;
    }
    if (state->node_pressure_activation_units != NULL) {
      state->node_pressure_activation_units[node_index] =
          pressure_activation_units != 0 ? pressure_activation_units : 1;
    }
    if (pressure_state->first_actionable_pressure_cliff_indices != NULL ||
        state->pressure_resources != NULL) {
      loom_low_schedule_reverse_source_pressure_node(state, pressure_state,
                                                     node);
      const loom_low_schedule_block_t* block_record =
          &state->blocks[node->block_index];
      if (node_index == block_record->node_start) {
        loom_low_schedule_remove_source_pressure_block_arguments(
            state, pressure_state, node->block);
      }
    }
  }
  if (pressure_state->first_actionable_pressure_cliff_indices != NULL ||
      state->pressure_resources != NULL) {
    // The reverse source sweep shares the existing priority traversal and
    // leaves only its immutable per-class cliff floors behind.
    loom_low_schedule_reset_source_pressure_sweep(state, pressure_state);
  }
}
