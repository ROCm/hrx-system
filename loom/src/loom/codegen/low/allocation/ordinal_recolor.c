// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/ordinal_recolor.h"

#include "loom/codegen/low/allocation/active_unit.h"
#include "loom/codegen/low/allocation/assignment_map.h"
#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/allocation/storage_lease.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/allocation/write_interference.h"

typedef struct loom_low_allocation_ordinal_recolor_state_t {
  // Current location search and every allocation fact it borrows.
  loom_low_allocation_search_context_t* search;
  // Mutable assignment table also borrowed read-only by |search|.
  loom_low_allocation_assignment_map_t* assignment_map;
  // Mutable records indexed by |assignment_map|.
  loom_low_allocation_assignment_t* assignments;
  // Local value lookup shared with liveness and placement.
  const loom_local_value_domain_t* value_domain;
  // Optional values already bound to non-register storage.
  const loom_low_storage_transport_t* storage_transport;
  // Assignment indices in the component being recolored.
  uint32_t* assignment_indices;
  // Original location bases paired with |assignment_indices|.
  uint32_t* original_location_bases;
  // Every component value ignored while validating the atomic update.
  loom_value_id_t* value_ids;
} loom_low_allocation_ordinal_recolor_state_t;

static loom_value_ordinal_t loom_low_allocation_ordinal_recolor_storage_origin(
    const loom_low_placement_table_t* placement,
    loom_value_ordinal_t value_ordinal) {
  return placement->tied_storage_origins_by_value_ordinal != NULL
             ? placement->tied_storage_origins_by_value_ordinal[value_ordinal]
             : value_ordinal;
}

static loom_low_allocation_assignment_t
loom_low_allocation_ordinal_recolor_prepare_assignment(
    const loom_low_allocation_ordinal_recolor_state_t* state,
    const loom_low_allocation_assignment_t* assignment,
    const loom_liveness_interval_t* interval,
    loom_value_ordinal_t value_ordinal) {
  loom_low_allocation_assignment_t stored_assignment = *assignment;
  const loom_low_allocation_unit_liveness_t* unit_liveness =
      state->search->unit_liveness;
  const uint32_t acquisition_start_point =
      unit_liveness->values[value_ordinal].acquisition_start_point;
  stored_assignment.flags |= loom_low_allocation_unit_liveness_assignment_flags(
      unit_liveness, interval, value_ordinal,
      state->search->expanded_execution_values);
  if (stored_assignment.start_point == acquisition_start_point) {
    stored_assignment.liveness_segments =
        loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
            unit_liveness, state->search->liveness, value_ordinal);
  }
  stored_assignment.unit_point_start =
      loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
          unit_liveness, state->search->liveness, value_ordinal);
  stored_assignment.end_point =
      loom_low_allocation_live_range_assignment_max_unit_end_point(
          unit_liveness->end_points, unit_liveness->point_count,
          &stored_assignment);
  return stored_assignment;
}

static bool loom_low_allocation_ordinal_recolor_collect_component(
    loom_low_allocation_ordinal_recolor_state_t* state,
    loom_low_placement_value_component_t component,
    uint16_t* out_current_candidate_ordinal, uint16_t* out_candidate_count,
    uint32_t* out_assignment_count, uint16_t* out_value_count) {
  const loom_liveness_analysis_t* liveness = state->search->liveness;
  if (component.count > UINT16_MAX ||
      state->search->retained_assignments_by_value_ordinal != NULL) {
    return false;
  }
  const loom_low_placement_table_t* placement = state->search->placement;
  const loom_low_descriptor_set_t* descriptors = state->search->descriptor_set;
  uint32_t assignment_count = 0;
  uint16_t value_count = 0;
  uint16_t current_candidate_ordinal = UINT16_MAX;
  uint16_t candidate_count = 0;
  for (uint32_t i = 0; i < component.count; ++i) {
    const loom_value_ordinal_t value_ordinal = component.members[i];
    const loom_liveness_interval_t* interval =
        loom_liveness_interval_for_value_ordinal(liveness, value_ordinal);
    if (interval == NULL || interval->unit_count != 1 ||
        interval->value_class.type_kind != LOOM_TYPE_REGISTER) {
      return false;
    }
    const uint16_t reg_class_id = interval->value_class.register_class_id;
    const loom_low_reg_class_t* reg_class =
        &descriptors->reg_classes[reg_class_id];
    if (!loom_low_reg_class_uses_explicit_physical_registers(reg_class) ||
        reg_class->allocatable_count == 0 ||
        (candidate_count != 0 &&
         candidate_count != reg_class->allocatable_count)) {
      return false;
    }
    candidate_count = reg_class->allocatable_count;
    const loom_value_ordinal_t storage_origin =
        loom_low_allocation_ordinal_recolor_storage_origin(placement,
                                                           value_ordinal);
    const loom_liveness_interval_t* origin_interval =
        loom_liveness_interval_for_value_ordinal(liveness, storage_origin);
    if (origin_interval == NULL ||
        origin_interval->value_class.type_kind != LOOM_TYPE_REGISTER ||
        origin_interval->value_class.register_class_id != reg_class_id) {
      // Exact tied-storage aliases can expose another physical class whose
      // candidate ordinals do not have the same meaning. Recoloring that
      // component requires an explicit cross-class storage mapping.
      return false;
    }
    const loom_value_id_t value_id =
        loom_low_placement_value_id(placement, value_ordinal);
    state->value_ids[value_count++] = value_id;
    if (loom_low_allocation_target_constraints_fixed_value_for_value_ordinal(
            state->search->target_constraints, value_ordinal) != NULL ||
        (state->storage_transport != NULL &&
         state->storage_transport->bindings_by_value_ordinal[value_ordinal] !=
             UINT32_MAX) ||
        loom_low_allocation_storage_lease_state_value_has_records(
            state->search->storage_leases, liveness, value_id)) {
      return false;
    }
    if (placement->operand_constraints_by_interval != NULL &&
        placement
            ->operand_constraints_by_interval[interval - liveness->intervals]
            .has_target_address_state) {
      return false;
    }

    uint32_t assignment_index = UINT32_MAX;
    const loom_low_allocation_assignment_t* assignment =
        loom_low_allocation_assignment_map_assignment_for_value_ordinal(
            state->assignment_map, value_ordinal, &assignment_index);
    if (assignment == NULL) {
      continue;
    }
    if (assignment->location_kind !=
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
        assignment->descriptor_reg_class_id != reg_class_id ||
        assignment->location_count != 1) {
      return false;
    }
    uint16_t assignment_candidate_ordinal = 0;
    if (!loom_low_descriptor_set_find_physical_register_candidate(
            descriptors, assignment->descriptor_reg_class_id,
            assignment->location_base, &assignment_candidate_ordinal)) {
      return false;
    }
    if (current_candidate_ordinal == UINT16_MAX) {
      current_candidate_ordinal = assignment_candidate_ordinal;
    } else if (assignment_candidate_ordinal != current_candidate_ordinal) {
      return false;
    }
    state->assignment_indices[assignment_count++] = assignment_index;
  }
  if (assignment_count == 0) {
    return false;
  }
  *out_current_candidate_ordinal = current_candidate_ordinal;
  *out_candidate_count = candidate_count;
  *out_assignment_count = assignment_count;
  *out_value_count = value_count;
  return true;
}

static void loom_low_allocation_ordinal_recolor_set_component_locations(
    loom_low_allocation_ordinal_recolor_state_t* state,
    uint32_t assignment_count, uint16_t candidate_ordinal,
    const uint32_t* restored_bases) {
  loom_low_allocation_assignment_map_t* assignment_map = state->assignment_map;
  loom_low_allocation_active_set_t* active = state->search->active_set;
  for (uint32_t i = 0; i < assignment_count; ++i) {
    const uint32_t assignment_index = state->assignment_indices[i];
    if (active->entries[assignment_index].position != UINT32_MAX) {
      loom_low_allocation_active_unit_index_remove_assignment(
          &active->units, state->assignments, assignment_map->assignment_count,
          assignment_index);
    }
  }
  for (uint32_t i = 0; i < assignment_count; ++i) {
    loom_low_allocation_assignment_t* assignment =
        &state->assignments[state->assignment_indices[i]];
    assignment->location_base =
        restored_bases != NULL
            ? restored_bases[i]
            : loom_low_descriptor_set_physical_register_candidate(
                  state->search->descriptor_set,
                  assignment->descriptor_reg_class_id, candidate_ordinal);
  }
  for (uint32_t i = 0; i < assignment_count; ++i) {
    const uint32_t assignment_index = state->assignment_indices[i];
    if (active->entries[assignment_index].position != UINT32_MAX) {
      loom_low_allocation_active_unit_index_insert_assignment(
          &active->units, state->search->descriptor_set, state->assignments,
          assignment_map->assignment_count, assignment_index);
    }
  }
  loom_low_allocation_target_constraints_rebuild_assignment_location_ends(
      state->search->target_constraints, state->assignments,
      assignment_map->assignment_count);
  loom_low_allocation_write_interference_reset_inference(
      state->search->unit_liveness->write_interference);
}

static bool loom_low_allocation_ordinal_recolor_candidate_conflicts(
    loom_low_allocation_ordinal_recolor_state_t* state,
    uint32_t component_index, const loom_low_allocation_assignment_t* candidate,
    uint16_t ignored_value_count) {
  if (loom_low_allocation_search_assignment_conflicts(
          state->search, candidate, state->value_ids, ignored_value_count,
          state->value_ids, ignored_value_count,
          LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN)) {
    return true;
  }
  const loom_low_allocation_assignment_map_t* assignment_map =
      state->assignment_map;
  for (iree_host_size_t i = 0; i < assignment_map->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* existing = &state->assignments[i];
    const loom_value_ordinal_t existing_ordinal =
        loom_local_value_domain_ordinal(state->value_domain,
                                        existing->value_id);
    if (loom_low_placement_hard_register_ordinal_component(
            state->search->placement, existing_ordinal)
            .index == component_index) {
      continue;
    }
    if (loom_low_allocation_live_range_assignments_conflict(
            state->search->descriptor_set,
            state->search->unit_liveness->storage_segments.entries,
            state->search->unit_liveness->start_points,
            state->search->unit_liveness->end_points,
            state->search->unit_liveness->point_count, candidate, existing)) {
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_ordinal_recolor_component_conflicts(
    loom_low_allocation_ordinal_recolor_state_t* state,
    loom_low_placement_value_component_t component, uint16_t candidate_ordinal,
    uint16_t ignored_value_count) {
  const loom_low_placement_table_t* placement = state->search->placement;
  for (uint32_t i = 0; i < component.count; ++i) {
    const loom_value_ordinal_t value_ordinal = component.members[i];
    const loom_low_allocation_assignment_t* assignment =
        loom_low_allocation_assignment_map_assignment_for_value_ordinal(
            state->assignment_map, value_ordinal, NULL);
    if (assignment != NULL) {
      if (loom_low_allocation_ordinal_recolor_candidate_conflicts(
              state, component.index, assignment, ignored_value_count)) {
        return true;
      }
      continue;
    }
    if (loom_low_allocation_ordinal_recolor_storage_origin(
            placement, value_ordinal) != value_ordinal) {
      continue;
    }
    const loom_liveness_interval_t* interval =
        loom_liveness_interval_for_value_ordinal(state->search->liveness,
                                                 value_ordinal);
    const uint16_t reg_class_id = interval->value_class.register_class_id;
    const loom_low_allocation_assignment_t projected =
        loom_low_allocation_ordinal_recolor_prepare_assignment(
            state,
            &(loom_low_allocation_assignment_t){
                .value_id = interval->value_id,
                .descriptor_reg_class_id = reg_class_id,
                .start_point =
                    state->search->unit_liveness->values[value_ordinal]
                        .acquisition_start_point,
                .end_point =
                    loom_low_allocation_live_range_interval_storage_end_point(
                        interval),
                .unit_count = 1,
                .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
                .location_base =
                    loom_low_descriptor_set_physical_register_candidate(
                        state->search->descriptor_set, reg_class_id,
                        candidate_ordinal),
                .location_count = 1,
            },
            interval, value_ordinal);
    if (loom_low_allocation_ordinal_recolor_candidate_conflicts(
            state, component.index, &projected, ignored_value_count)) {
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_ordinal_recolor_component(
    loom_low_allocation_ordinal_recolor_state_t* state,
    loom_low_placement_value_component_t component,
    const loom_liveness_interval_t* requested_interval,
    const loom_low_allocation_class_capacity_t* requested_capacity,
    uint32_t* out_location_base) {
  uint16_t current_candidate_ordinal = 0;
  uint16_t candidate_count = 0;
  uint32_t assignment_count = 0;
  uint16_t value_count = 0;
  if (!loom_low_allocation_ordinal_recolor_collect_component(
          state, component, &current_candidate_ordinal, &candidate_count,
          &assignment_count, &value_count)) {
    return false;
  }
  const uint32_t original_residency_tier =
      loom_low_allocation_search_assignment_residency_tier(
          state->search, &state->assignments[state->assignment_indices[0]]);
  const uint32_t original_retained_fixed_value_index_plus_one =
      state->search->retained_fixed_value_index_plus_one;
  for (uint32_t i = 0; i < assignment_count; ++i) {
    state->original_location_bases[i] =
        state->assignments[state->assignment_indices[i]].location_base;
  }

  for (uint16_t candidate_ordinal = 0; candidate_ordinal < candidate_count;
       ++candidate_ordinal) {
    if (candidate_ordinal == current_candidate_ordinal) {
      continue;
    }
    loom_low_allocation_ordinal_recolor_set_component_locations(
        state, assignment_count, candidate_ordinal,
        /*restored_bases=*/NULL);
    const uint32_t candidate_residency_tier =
        loom_low_allocation_search_assignment_residency_tier(
            state->search, &state->assignments[state->assignment_indices[0]]);
    bool accepted = candidate_residency_tier >= original_residency_tier &&
                    !loom_low_allocation_ordinal_recolor_component_conflicts(
                        state, component, candidate_ordinal, value_count);
    uint32_t requested_base = 0;
    if (accepted) {
      accepted = loom_low_allocation_search_find_free_location(
          state->search, requested_interval, *requested_capacity,
          &requested_base);
    }
    if (accepted) {
      *out_location_base = requested_base;
      return true;
    }
    loom_low_allocation_ordinal_recolor_set_component_locations(
        state, assignment_count, /*candidate_ordinal=*/0,
        state->original_location_bases);
    state->search->retained_fixed_value_index_plus_one =
        original_retained_fixed_value_index_plus_one;
  }
  return false;
}

static bool loom_low_allocation_ordinal_recolor_try_impl(
    loom_low_allocation_ordinal_recolor_state_t* state,
    const loom_liveness_interval_t* requested_interval,
    loom_value_ordinal_t requested_value_ordinal,
    const loom_low_allocation_class_capacity_t* requested_capacity,
    uint32_t* out_location_base) {
  const loom_low_placement_value_component_t requested_component =
      loom_low_placement_hard_register_ordinal_component(
          state->search->placement, requested_value_ordinal);
  if (requested_component.index == UINT32_MAX) {
    return false;
  }
  const loom_low_descriptor_set_t* descriptors = state->search->descriptor_set;
  const loom_low_reg_class_t* reg_class =
      &descriptors->reg_classes[requested_capacity->descriptor_reg_class_id];
  for (uint16_t candidate_ordinal = 0;
       candidate_ordinal < reg_class->allocatable_count; ++candidate_ordinal) {
    const uint32_t location_base =
        loom_low_descriptor_set_physical_register_candidate(
            descriptors, requested_capacity->descriptor_reg_class_id,
            candidate_ordinal);
    const loom_low_allocation_assignment_t requested =
        loom_low_allocation_ordinal_recolor_prepare_assignment(
            state,
            &(loom_low_allocation_assignment_t){
                .value_id = requested_interval->value_id,
                .descriptor_reg_class_id =
                    requested_capacity->descriptor_reg_class_id,
                .start_point = state->search->unit_liveness
                                   ->values[requested_value_ordinal]
                                   .acquisition_start_point,
                .end_point =
                    loom_low_allocation_live_range_interval_storage_end_point(
                        requested_interval),
                .unit_count = 1,
                .location_kind = requested_capacity->location_kind,
                .location_base = location_base,
                .location_count = 1,
            },
            requested_interval, requested_value_ordinal);
    if (requested_capacity->is_bounded &&
        loom_low_allocation_storage_assignment_pressure_extent(
            descriptors, &requested) > requested_capacity->max_units) {
      continue;
    }
    loom_low_allocation_active_set_t* active = state->search->active_set;
    for (iree_host_size_t i = 0; i < active->count; ++i) {
      const uint32_t assignment_index = active->assignment_indices[i];
      const loom_low_allocation_assignment_t* blocking_assignment =
          &state->assignments[assignment_index];
      if (!loom_low_allocation_live_range_assignments_conflict(
              descriptors,
              state->search->unit_liveness->storage_segments.entries,
              state->search->unit_liveness->start_points,
              state->search->unit_liveness->end_points,
              state->search->unit_liveness->point_count, &requested,
              blocking_assignment)) {
        continue;
      }
      const loom_value_ordinal_t blocking_ordinal =
          loom_local_value_domain_ordinal(state->value_domain,
                                          blocking_assignment->value_id);
      const loom_low_placement_value_component_t blocking_component =
          loom_low_placement_hard_register_ordinal_component(
              state->search->placement, blocking_ordinal);
      if (blocking_component.index == UINT32_MAX ||
          blocking_component.index == requested_component.index) {
        continue;
      }
      if (loom_low_allocation_ordinal_recolor_component(
              state, blocking_component, requested_interval, requested_capacity,
              out_location_base)) {
        return true;
      }
    }
  }
  return false;
}

iree_status_t loom_low_allocation_ordinal_recolor_try(
    const loom_local_value_domain_t* value_domain,
    const loom_low_storage_transport_t* storage_transport,
    loom_low_allocation_search_context_t* search_context,
    loom_low_allocation_assignment_map_t* assignment_map,
    loom_low_allocation_assignment_t* assignments,
    iree_arena_allocator_t* scratch_arena,
    const loom_liveness_interval_t* requested_interval,
    loom_value_ordinal_t requested_value_ordinal,
    const loom_low_allocation_class_capacity_t* requested_capacity,
    uint32_t* out_location_base, bool* out_recolored) {
  *out_recolored = false;
  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(scratch_arena);
  loom_low_allocation_ordinal_recolor_state_t state = {
      .search = search_context,
      .assignment_map = assignment_map,
      .assignments = assignments,
      .value_domain = value_domain,
      .storage_transport = storage_transport,
  };
  const iree_host_size_t value_count = search_context->liveness->value_count;
  iree_status_t status = iree_arena_allocate_array(
      scratch_arena, value_count, sizeof(*state.assignment_indices),
      (void**)&state.assignment_indices);
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(scratch_arena, value_count,
                                       sizeof(*state.original_location_bases),
                                       (void**)&state.original_location_bases);
  }
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(scratch_arena, value_count,
                                       sizeof(*state.value_ids),
                                       (void**)&state.value_ids);
  }
  if (iree_status_is_ok(status)) {
    *out_recolored = loom_low_allocation_ordinal_recolor_try_impl(
        &state, requested_interval, requested_value_ordinal, requested_capacity,
        out_location_base);
  }
  iree_arena_checkpoint_restore(&checkpoint);
  return status;
}
