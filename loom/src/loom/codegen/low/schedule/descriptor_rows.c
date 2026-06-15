// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/descriptor_rows.h"

#include <inttypes.h>

static iree_status_t loom_low_schedule_append_resource_use(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_resource_use_t resource_use) {
  if (state->resource_use_count >= state->resource_use_capacity) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "low schedule exceeded precomputed resource-use capacity");
  }
  if (!state->resource_summaries ||
      resource_use.resource_id >=
          state->target.descriptor_set->resource_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low schedule resource use cannot be summarized");
  }
  loom_low_schedule_resource_summary_t* summary =
      &state->resource_summaries[resource_use.resource_id];
  if (summary->use_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "low schedule resource-use count overflows");
  }
  const uint64_t occupied_cycles = resource_use.cycles;
  const uint64_t unit_cycles =
      (uint64_t)resource_use.cycles * (uint64_t)resource_use.units;
  if (summary->total_occupied_cycles > UINT64_MAX - occupied_cycles ||
      summary->total_unit_cycles > UINT64_MAX - unit_cycles) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "low schedule resource summary counters overflow");
  }
  ++summary->use_count;
  summary->total_occupied_cycles += occupied_cycles;
  summary->total_unit_cycles += unit_cycles;
  summary->estimated_min_cycles =
      1 + (summary->total_unit_cycles - 1) / summary->capacity_per_cycle;
  if (resource_use.units > summary->peak_units_per_cycle) {
    summary->peak_units_per_cycle = resource_use.units;
  }
  if (state->resource_ready_issue_cycles != NULL) {
    const uint32_t use_start = loom_low_schedule_saturating_add_u32(
        state->current_issue_cycle, resource_use.stage);
    const uint32_t use_end =
        loom_low_schedule_saturating_add_u32(use_start, resource_use.cycles);
    if (use_end >
        state->resource_ready_issue_cycles[resource_use.resource_id]) {
      state->resource_ready_issue_cycles[resource_use.resource_id] = use_end;
    }
  }
  state->resource_uses[state->resource_use_count++] = resource_use;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_append_effect_use(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_use_t effect_use) {
  if (state->effect_use_count >= state->effect_use_capacity) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low schedule exceeded precomputed effect-use capacity");
  }
  state->effect_uses[state->effect_use_count++] = effect_use;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_append_hazard_use(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_hazard_use_t hazard_use) {
  if (state->hazard_use_count >= state->hazard_use_capacity) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "low schedule exceeded precomputed hazard-use capacity");
  }
  state->hazard_uses[state->hazard_use_count++] = hazard_use;
  return iree_ok_status();
}

static bool loom_low_schedule_hazard_state_matches(
    const loom_low_schedule_hazard_state_t* hazard_state,
    const loom_low_schedule_hazard_use_t* hazard_use, uint16_t producer_stage) {
  return hazard_state->kind == hazard_use->kind &&
         hazard_state->reference_kind == hazard_use->reference_kind &&
         hazard_state->reference_id == hazard_use->reference_id &&
         hazard_state->producer_stage == producer_stage &&
         hazard_state->block_index == hazard_use->block_index;
}

static iree_status_t loom_low_schedule_append_hazard_gap(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_hazard_gap_t hazard_gap) {
  if (state->hazard_gap_count >= state->hazard_gap_capacity) {
    iree_host_size_t new_capacity =
        state->hazard_gap_capacity == 0 ? 4 : state->hazard_gap_capacity * 2;
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(state->arena, state->hazard_gap_count,
                              new_capacity, sizeof(*state->hazard_gaps),
                              &new_capacity, (void**)&state->hazard_gaps));
    state->hazard_gap_capacity = new_capacity;
  }
  state->hazard_gaps[state->hazard_gap_count++] = hazard_gap;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_update_hazard_state(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_hazard_use_t* hazard_use) {
  for (iree_host_size_t i = 0; i < state->hazard_state_count; ++i) {
    loom_low_schedule_hazard_state_t* hazard_state = &state->hazard_states[i];
    if (!loom_low_schedule_hazard_state_matches(hazard_state, hazard_use,
                                                hazard_use->producer_stage)) {
      continue;
    }
    hazard_state->node_index = hazard_use->node_index;
    hazard_state->scheduled_ordinal = hazard_use->scheduled_ordinal;
    hazard_state->issue_cycle = state->current_issue_cycle;
    hazard_state->hazard_ordinal = hazard_use->hazard_ordinal;
    hazard_state->distance = hazard_use->distance;
    hazard_state->hazard_flags = hazard_use->hazard_flags;
    return iree_ok_status();
  }

  if (state->hazard_state_count >= state->hazard_state_capacity) {
    iree_host_size_t new_capacity = state->hazard_state_capacity == 0
                                        ? 4
                                        : state->hazard_state_capacity * 2;
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(state->arena, state->hazard_state_count,
                              new_capacity, sizeof(*state->hazard_states),
                              &new_capacity, (void**)&state->hazard_states));
    state->hazard_state_capacity = new_capacity;
  }
  state->hazard_states[state->hazard_state_count++] =
      (loom_low_schedule_hazard_state_t){
          .kind = hazard_use->kind,
          .reference_kind = hazard_use->reference_kind,
          .reference_id = hazard_use->reference_id,
          .producer_stage = hazard_use->producer_stage,
          .block_index = hazard_use->block_index,
          .node_index = hazard_use->node_index,
          .scheduled_ordinal = hazard_use->scheduled_ordinal,
          .issue_cycle = state->current_issue_cycle,
          .hazard_ordinal = hazard_use->hazard_ordinal,
          .distance = hazard_use->distance,
          .hazard_flags = hazard_use->hazard_flags,
      };
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_note_min_distance_hazard(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_hazard_use_t* hazard_use) {
  if (hazard_use->kind != LOOM_LOW_HAZARD_KIND_MIN_DISTANCE) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < state->hazard_state_count; ++i) {
    const loom_low_schedule_hazard_state_t* hazard_state =
        &state->hazard_states[i];
    if (!loom_low_schedule_hazard_state_matches(hazard_state, hazard_use,
                                                hazard_use->consumer_stage)) {
      continue;
    }
    if (hazard_use->scheduled_ordinal < hazard_state->scheduled_ordinal) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low schedule hazard producer appears after consumer");
    }
    const uint32_t actual_distance =
        hazard_use->scheduled_ordinal - hazard_state->scheduled_ordinal;
    const uint16_t required_distance =
        hazard_state->distance > hazard_use->distance ? hazard_state->distance
                                                      : hazard_use->distance;
    if (actual_distance < required_distance) {
      IREE_RETURN_IF_ERROR(loom_low_schedule_append_hazard_gap(
          state,
          (loom_low_schedule_hazard_gap_t){
              .producer_node = hazard_state->node_index,
              .consumer_node = hazard_use->node_index,
              .block_index = hazard_use->block_index,
              .producer_scheduled_ordinal = hazard_state->scheduled_ordinal,
              .consumer_scheduled_ordinal = hazard_use->scheduled_ordinal,
              .producer_hazard_ordinal = hazard_state->hazard_ordinal,
              .consumer_hazard_ordinal = hazard_use->hazard_ordinal,
              .kind = hazard_use->kind,
              .reference_kind = hazard_use->reference_kind,
              .reference_id = hazard_use->reference_id,
              .resource_name = hazard_use->resource_name,
              .producer_stage = hazard_state->producer_stage,
              .consumer_stage = hazard_use->consumer_stage,
              .required_distance = required_distance,
              .actual_distance = actual_distance,
              .required_delay = (uint16_t)(required_distance - actual_distance),
              .hazard_flags =
                  hazard_state->hazard_flags | hazard_use->hazard_flags,
          }));
    }
    break;
  }
  return loom_low_schedule_update_hazard_state(state, hazard_use);
}

void loom_low_schedule_compact_resource_summaries(
    loom_low_schedule_build_state_t* state) {
  if (!state->resource_summaries) {
    return;
  }
  iree_host_size_t write_index = 0;
  for (iree_host_size_t read_index = 0;
       read_index < state->target.descriptor_set->resource_count;
       ++read_index) {
    if (state->resource_summaries[read_index].use_count == 0) {
      continue;
    }
    state->resource_summaries[write_index++] =
        state->resource_summaries[read_index];
  }
  state->resource_summary_count = write_index;
}

void loom_low_schedule_compact_model_summaries(
    loom_low_schedule_build_state_t* state) {
  if (!state->model_summaries) {
    return;
  }
  iree_host_size_t write_index = 0;
  for (iree_host_size_t read_index = 0;
       read_index < state->target.descriptor_set->schedule_class_count;
       ++read_index) {
    if (state->model_summaries[read_index].use_count == 0) {
      continue;
    }
    state->model_summaries[write_index++] = state->model_summaries[read_index];
  }
  state->model_summary_count = write_index;
}

iree_status_t loom_low_schedule_note_descriptor_rows_for_node(
    loom_low_schedule_build_state_t* state, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  if (node->schedule_class_id == LOOM_LOW_SCHEDULE_CLASS_NONE) {
    return iree_ok_status();
  }
  if (node->schedule_class_id >=
      state->target.descriptor_set->schedule_class_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low schedule node references invalid schedule class %" PRIu16,
        node->schedule_class_id);
  }
  const loom_low_schedule_class_t* schedule_class =
      &state->target.descriptor_set->schedule_classes[node->schedule_class_id];
  const loom_low_descriptor_t* descriptor = node->descriptor;
  if (state->model_summaries) {
    loom_low_schedule_model_summary_t* model_summary =
        &state->model_summaries[node->schedule_class_id];
    if (model_summary->use_count == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "low schedule model summary counters overflow");
    }
    if (model_summary->use_count == 0) {
      iree_string_view_t schedule_class_name = loom_low_descriptor_set_string(
          state->target.descriptor_set, schedule_class->name_string_offset);
      *model_summary = (loom_low_schedule_model_summary_t){
          .first_node = node_index,
          .schedule_class_id = node->schedule_class_id,
          .schedule_class_name = schedule_class_name,
          .latency_cycles = schedule_class->latency_cycles,
          .latency_kind = schedule_class->latency_kind,
          .model_quality = schedule_class->model_quality,
          .issue_use_count = schedule_class->issue_use_count,
          .hazard_count = schedule_class->hazard_count,
      };
    }
    ++model_summary->use_count;
  }
  if (descriptor) {
    for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
      const loom_low_effect_t* effect =
          &state->target.descriptor_set->effects[descriptor->effect_start + i];
      IREE_RETURN_IF_ERROR(loom_low_schedule_append_effect_use(
          state, (loom_low_schedule_effect_use_t){
                     .node_index = node_index,
                     .block_index = node->block_index,
                     .scheduled_ordinal = node->scheduled_ordinal,
                     .effect_ordinal = i,
                     .kind = effect->kind,
                     .memory_space = effect->memory_space,
                     .scope_id = effect->scope_id,
                     .effect_flags = effect->flags,
                     .counter_id = effect->counter_id,
                     .width_bits = effect->width_bits,
                 }));
    }
  }
  for (uint16_t i = 0; i < schedule_class->issue_use_count; ++i) {
    const loom_low_issue_use_t* issue_use =
        &state->target.descriptor_set
             ->issue_uses[schedule_class->issue_use_start + i];
    if (issue_use->resource_id >=
        state->target.descriptor_set->resource_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low schedule issue-use references invalid resource %" PRIu16,
          issue_use->resource_id);
    }
    const loom_low_resource_t* resource =
        &state->target.descriptor_set->resources[issue_use->resource_id];
    iree_string_view_t resource_name = loom_low_descriptor_set_string(
        state->target.descriptor_set, resource->name_string_offset);
    IREE_RETURN_IF_ERROR(loom_low_schedule_append_resource_use(
        state, (loom_low_schedule_resource_use_t){
                   .node_index = node_index,
                   .block_index = node->block_index,
                   .scheduled_ordinal = node->scheduled_ordinal,
                   .issue_use_ordinal = i,
                   .resource_id = issue_use->resource_id,
                   .resource_name = resource_name,
                   .resource_kind = resource->kind,
                   .resource_flags = resource->flags,
                   .capacity_per_cycle = resource->capacity_per_cycle,
                   .contention_group_id = resource->contention_group_id,
                   .stage = issue_use->stage,
                   .cycles = issue_use->cycles,
                   .units = issue_use->units,
               }));
  }
  for (uint16_t i = 0; i < schedule_class->hazard_count; ++i) {
    const loom_low_hazard_t* hazard =
        &state->target.descriptor_set
             ->hazards[schedule_class->hazard_start + i];
    iree_string_view_t resource_name = iree_string_view_empty();
    if (hazard->reference_kind == LOOM_LOW_HAZARD_REFERENCE_KIND_RESOURCE) {
      if (hazard->reference_id >=
          state->target.descriptor_set->resource_count) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "low schedule hazard references invalid resource %" PRIu16,
            hazard->reference_id);
      }
      const loom_low_resource_t* resource =
          &state->target.descriptor_set->resources[hazard->reference_id];
      resource_name = loom_low_descriptor_set_string(
          state->target.descriptor_set, resource->name_string_offset);
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_append_hazard_use(
        state, (loom_low_schedule_hazard_use_t){
                   .node_index = node_index,
                   .block_index = node->block_index,
                   .scheduled_ordinal = node->scheduled_ordinal,
                   .hazard_ordinal = i,
                   .kind = hazard->kind,
                   .reference_kind = hazard->reference_kind,
                   .reference_id = hazard->reference_id,
                   .resource_name = resource_name,
                   .producer_stage = hazard->producer_stage,
                   .consumer_stage = hazard->consumer_stage,
                   .distance = hazard->distance,
                   .hazard_flags = hazard->flags,
               }));
    IREE_RETURN_IF_ERROR(loom_low_schedule_note_min_distance_hazard(
        state, &state->hazard_uses[state->hazard_use_count - 1]));
  }
  return iree_ok_status();
}
