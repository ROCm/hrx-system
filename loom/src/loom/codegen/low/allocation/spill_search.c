// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/spill_search.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/rematerialization_facts.h"
#include "loom/ir/module.h"

iree_status_t loom_low_allocation_spill_search_assignment_capacity(
    const loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* assignment, bool* out_can_spill,
    loom_low_allocation_class_capacity_t* out_capacity) {
  *out_can_spill = false;
  if (!loom_low_allocation_assignment_is_register_like(assignment)) {
    return iree_ok_status();
  }
  if (loom_low_allocation_target_constraints_fixed_value_for_value(
          context->target_constraints, assignment->value_id)) {
    return iree_ok_status();
  }
  if (iree_any_bit_set(assignment->flags,
                       LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REQUIRES_REGISTER)) {
    return iree_ok_status();
  }
  loom_low_allocation_class_capacity_t capacity = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_target_constraints_reg_class_capacity(
          context->target_constraints, assignment->descriptor_reg_class_id,
          &capacity));
  if (!capacity.is_spillable) {
    return iree_ok_status();
  }
  if (out_capacity) {
    *out_capacity = capacity;
  }
  *out_can_spill = true;
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_spill_search_assignment_spill_traffic_cost(
    loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* assignment,
    const loom_low_allocation_class_capacity_t* capacity, uint64_t* out_cost) {
  *out_cost = 0;
  const loom_value_ordinal_t value_ordinal =
      loom_module_value_ordinal_scratch_lookup(context->module,
                                               assignment->value_id);
  loom_low_allocation_spill_plan_traffic_t traffic =
      context->spill_traffic_by_value_ordinal[value_ordinal];
  if (traffic.store_count == UINT32_MAX) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_spill_plan_traffic(
        context->module, context->cfg_graph, assignment,
        capacity->spill_unit_bits, context->spill_traffic_workspace, &traffic));
    context->spill_traffic_by_value_ordinal[value_ordinal] = traffic;
  }
  if (!traffic.has_narrow_uses &&
      iree_any_bit_set(context->rematerialization_flags[value_ordinal],
                       LOOM_LOW_REMATERIALIZATION_VALUE_CLOSED)) {
    // Whole-value users can initialize a closed recipe without memory traffic.
    // Narrow users retain their wide producer instead of rebuilding every unit.
    return iree_ok_status();
  }
  *out_cost =
      iree_math_saturating_add_u64(traffic.store_bytes, traffic.reload_bytes);
  return iree_ok_status();
}

static bool loom_low_allocation_spill_search_relation_forwards_storage(
    const loom_low_placement_relation_t* relation) {
  return relation->cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT &&
         relation->kind == LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART &&
         relation->unit_count != 0 &&
         loom_low_placement_relation_can_alias(relation) &&
         !iree_any_bit_set(relation->flags,
                           LOOM_LOW_PLACEMENT_RELATION_FLAG_WRITES_STORAGE |
                               LOOM_LOW_PLACEMENT_RELATION_FLAG_CAPTURED_PART);
}

static bool loom_low_allocation_spill_search_relation_forwards_complete_storage(
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* source_assignment) {
  return relation->source_unit_offset == 0 &&
         relation->unit_count == source_assignment->unit_count &&
         loom_low_allocation_spill_search_relation_forwards_storage(relation);
}

// Finds the sole still-live result that forwards all storage of
// |source_assignment|. Spilling that result creates a materialized observation
// of the source bits and replaces later result uses with reloads, allowing the
// required source reservation to end at the new store. Placement owns the
// indexed relation graph, so this query visits each direct structural user at
// most once. Keep this exceptional recovery out of the hot spill-candidate
// loop; ordinary assignments never need to inspect the relation graph.
IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_low_allocation_spill_search_find_covering_spill_assignment(
    loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* source_assignment,
    uint32_t program_point, uint32_t* out_assignment_index,
    loom_low_allocation_class_capacity_t* out_capacity,
    uint64_t* out_traffic_cost) {
  *out_assignment_index = UINT32_MAX;
  *out_capacity = (loom_low_allocation_class_capacity_t){0};
  *out_traffic_cost = UINT64_MAX;
  if (context->placement->relation_count == 0) {
    return iree_ok_status();
  }
  loom_value_ordinal_t source_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_assignment_map_value_ordinal_for_value(
          context->assignment_map, source_assignment->value_id,
          &source_ordinal)) {
    return iree_ok_status();
  }
  const loom_low_placement_relation_range_t range =
      loom_low_placement_relation_range_for_source_value_ordinal(
          context->placement, source_ordinal);
  const loom_liveness_interval_t* source_interval =
      loom_liveness_interval_for_value_ordinal(context->liveness,
                                               source_ordinal);
  if (source_interval == NULL) {
    return iree_ok_status();
  }
  const iree_bitmap_t incomplete_storage_segments =
      context->unit_liveness->values_with_incomplete_storage_segments;
  if (source_ordinal >= incomplete_storage_segments.bit_count ||
      !iree_bitmap_test(incomplete_storage_segments, source_ordinal)) {
    return iree_ok_status();
  }
  const uint32_t observation_node =
      context->unit_liveness->observations.value_links == NULL
          ? 0
          : context->unit_liveness->observations.value_links[source_ordinal];
  if (observation_node <= context->placement->value_count) {
    return iree_ok_status();
  }
  if (iree_bitmap_test(
          context->unit_liveness->values_with_independent_storage_extensions,
          source_ordinal)) {
    return iree_ok_status();
  }
  const loom_value_ordinal_t* tied_origins =
      context->placement->tied_storage_origins_by_value_ordinal;
  if (tied_origins != NULL && tied_origins[source_ordinal] != source_ordinal) {
    return iree_ok_status();
  }

  loom_value_ordinal_t covering_result_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  bool covers_complete_source = false;
  for (uint32_t i = 0; i < range.count; ++i) {
    const uint32_t relation_index =
        context->placement->relation_indices_by_source_ordinal[range.start + i];
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[relation_index];
    // A tied result keeps the source's complete storage component observable.
    // Materializing an independent forwarding result cannot retire it.
    if (relation->cause == LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT) {
      return iree_ok_status();
    }
    if (!loom_low_allocation_spill_search_relation_forwards_storage(relation)) {
      continue;
    }
    const loom_low_allocation_assignment_t* result_assignment =
        loom_low_allocation_assignment_map_assignment_for_value_ordinal(
            context->assignment_map, relation->result_ordinal, NULL);
    if (result_assignment == NULL ||
        result_assignment->end_point <= program_point) {
      continue;
    }
    if (covering_result_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
      covering_result_ordinal = relation->result_ordinal;
    } else if (covering_result_ordinal != relation->result_ordinal) {
      return iree_ok_status();
    }
    covers_complete_source |=
        loom_low_allocation_spill_search_relation_forwards_complete_storage(
            relation, source_assignment);
  }
  if (covering_result_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      !covers_complete_source) {
    return iree_ok_status();
  }

  uint32_t result_assignment_index = UINT32_MAX;
  const loom_low_allocation_assignment_t* result_assignment =
      loom_low_allocation_assignment_map_assignment_for_value_ordinal(
          context->assignment_map, covering_result_ordinal,
          &result_assignment_index);
  const loom_liveness_interval_t* result_interval =
      loom_liveness_interval_for_value_ordinal(context->liveness,
                                               covering_result_ordinal);
  if (result_assignment == NULL || result_assignment_index == UINT32_MAX ||
      context->active_set->entries[result_assignment_index].position ==
          UINT32_MAX ||
      result_assignment->end_point <= program_point ||
      result_interval == NULL ||
      result_interval->definition_point >= program_point ||
      source_interval->end_point > result_interval->definition_point) {
    return iree_ok_status();
  }

  bool can_spill = false;
  loom_low_allocation_class_capacity_t capacity = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_spill_search_assignment_capacity(
      context, result_assignment, &can_spill, &capacity));
  if (!can_spill) {
    return iree_ok_status();
  }
  uint64_t traffic_cost = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_spill_search_assignment_spill_traffic_cost(
          context, result_assignment, &capacity, &traffic_cost));
  *out_assignment_index = result_assignment_index;
  *out_capacity = capacity;
  *out_traffic_cost = traffic_cost;
  return iree_ok_status();
}

// Lazily starts duplicate tracking after the first covering spill maps a
// source conflict onto a different assignment. Direct conflict assignments are
// unique, so ordinary spill searches do not need this bitmap or its arena
// storage.
IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_low_allocation_spill_search_begin_covering_spill_selection(
    const loom_low_allocation_active_set_t* active_set,
    const uint32_t* assignment_indices, uint16_t assignment_count,
    loom_low_allocation_spill_search_workspace_t* workspace,
    iree_arena_allocator_t* arena) {
  if (workspace->candidate_selected_assignments.bit_count <
      workspace->capacity) {
    const iree_host_size_t selected_word_count =
        iree_bitmap_calculate_words(workspace->capacity);
    uint64_t* selected_words = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, selected_word_count,
                                                   sizeof(*selected_words),
                                                   (void**)&selected_words));
    memset(selected_words, 0, selected_word_count * sizeof(*selected_words));
    workspace->candidate_selected_assignments = (iree_bitmap_t){
        .bit_count = workspace->capacity,
        .words = selected_words,
    };
  }
  for (uint16_t i = 0; i < assignment_count; ++i) {
    const uint32_t active_position =
        active_set->entries[assignment_indices[i]].position;
    IREE_ASSERT_NE(active_position, UINT32_MAX);
    iree_bitmap_set(workspace->candidate_selected_assignments, active_position);
  }
  return iree_ok_status();
}

static bool loom_low_allocation_spill_search_spill_victim_set_is_better(
    bool use_pressure_efficiency, uint16_t candidate_count,
    uint32_t candidate_unit_count, uint32_t candidate_latest_end_point,
    uint64_t candidate_traffic_cost, uint64_t candidate_remaining_pressure_area,
    uint32_t candidate_location_ordinal, uint16_t best_count,
    uint32_t best_unit_count, uint32_t best_latest_end_point,
    uint64_t best_traffic_cost, uint64_t best_remaining_pressure_area,
    uint32_t best_location_ordinal) {
  if (best_count == 0) {
    return true;
  }
  if (use_pressure_efficiency) {
    uint64_t candidate_high = 0, candidate_low = 0;
    uint64_t best_high = 0, best_low = 0;
    iree_math_mul_u64_to_u128(candidate_traffic_cost,
                              best_remaining_pressure_area, &candidate_high,
                              &candidate_low);
    iree_math_mul_u64_to_u128(best_traffic_cost,
                              candidate_remaining_pressure_area, &best_high,
                              &best_low);
    if (candidate_high != best_high) {
      return candidate_high < best_high;
    }
    if (candidate_low != best_low) {
      return candidate_low < best_low;
    }
  }
  if (candidate_traffic_cost != best_traffic_cost) {
    return candidate_traffic_cost < best_traffic_cost;
  }
  if (candidate_count != best_count) {
    return candidate_count < best_count;
  }
  if (candidate_unit_count != best_unit_count) {
    return candidate_unit_count < best_unit_count;
  }
  if (candidate_latest_end_point != best_latest_end_point) {
    return candidate_latest_end_point > best_latest_end_point;
  }
  return candidate_location_ordinal < best_location_ordinal;
}

// Appends only new active conflicts. The union cannot exceed the active set;
// the existing ignored-value scratch deduplicates peers without another array.
static iree_status_t loom_low_allocation_spill_search_append_active_conflicts(
    const loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* candidate,
    uint32_t* assignment_indices, loom_value_id_t* ignored_value_ids,
    uint16_t* inout_count) {
  const uint16_t ignored_count = *inout_count;
  for (uint16_t i = 0; i < ignored_count; ++i) {
    ignored_value_ids[i] =
        context->assignment_map->assignments[assignment_indices[i]].value_id;
  }
  if (loom_low_allocation_active_unit_index_is_enabled(
          &context->active_set->units)) {
    return loom_low_allocation_active_unit_index_collect_conflicts(
        &context->active_set->units, &context->active_set->live_range_sweep,
        context->descriptor_set, context->unit_liveness,
        context->assignment_map->assignments,
        context->assignment_map->assignment_count, candidate, ignored_value_ids,
        ignored_count, assignment_indices, (uint16_t)context->active_set->count,
        inout_count);
  }
  for (iree_host_size_t i = 0; i < context->active_set->count; ++i) {
    const uint32_t assignment_index =
        context->active_set->assignment_indices[i];
    if (loom_low_allocation_active_assignment_conflicts(
            &context->active_set->live_range_sweep, context->descriptor_set,
            context->unit_liveness, context->assignment_map->assignments,
            context->assignment_map->assignment_count, assignment_index,
            candidate, ignored_value_ids, ignored_count,
            /*shared_content=*/NULL)) {
      assignment_indices[(*inout_count)++] = assignment_index;
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_spill_search_collect_active_spill_victim_set(
    loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_allocation_class_capacity_t* capacity,
    loom_low_placement_value_component_t component,
    loom_value_ordinal_t candidate_origin, uint32_t location_base,
    bool interval_requires_register, uint64_t traffic_limit,
    uint32_t* assignment_indices, loom_value_id_t* ignored_value_ids,
    loom_low_allocation_spill_search_workspace_t* workspace,
    iree_arena_allocator_t* arena, uint16_t* out_assignment_count,
    uint32_t* out_unit_count, uint32_t* out_latest_end_point,
    uint64_t* out_traffic_cost, uint64_t* out_remaining_pressure_area,
    uint16_t* out_released_value_count, bool* out_blocked) {
  *out_assignment_count = 0;
  *out_unit_count = 0;
  *out_latest_end_point = 0;
  *out_traffic_cost = 0;
  *out_remaining_pressure_area = 0;
  *out_released_value_count = 0;
  *out_blocked = false;
  const loom_low_allocation_assignment_t candidate =
      loom_low_allocation_search_candidate_assignment(
          context, interval, capacity->descriptor_reg_class_id,
          capacity->location_kind, location_base, interval->unit_count);
  const uint32_t interval_end = candidate.end_point;

  uint16_t conflict_assignment_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_spill_search_append_active_conflicts(
      context, &candidate, assignment_indices, ignored_value_ids,
      &conflict_assignment_count));

  if (component.count != 0) {
    uint16_t candidate_ordinal = 0;
    loom_low_descriptor_set_find_physical_register_candidate(
        context->descriptor_set, candidate.descriptor_reg_class_id,
        candidate.location_base, &candidate_ordinal);
    for (uint32_t i = 0; i < component.origin_count; ++i) {
      const loom_value_ordinal_t peer_ordinal = component.members[i];
      if (peer_ordinal == candidate_origin) {
        continue;
      }
      const loom_low_allocation_assignment_t* peer =
          loom_low_allocation_assignment_map_assignment_for_value_ordinal(
              context->assignment_map, peer_ordinal, NULL);
      if (peer != NULL) {
        continue;
      }
      const loom_liveness_interval_t* peer_interval =
          loom_liveness_interval_for_value_ordinal(context->liveness,
                                                   peer_ordinal);
      const uint16_t peer_class = peer_interval->value_class.register_class_id;
      const loom_low_allocation_assignment_t projected =
          loom_low_allocation_search_candidate_assignment(
              context, peer_interval, peer_class,
              LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
              loom_low_descriptor_set_physical_register_candidate(
                  context->descriptor_set, peer_class, candidate_ordinal),
              1);
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_spill_search_append_active_conflicts(
              context, &projected, assignment_indices, ignored_value_ids,
              &conflict_assignment_count));
    }
  }

  uint16_t assignment_count = 0;
  uint16_t ignored_value_count = 0;
  uint32_t unit_count = 0;
  uint32_t latest_end_point = 0;
  uint64_t traffic_cost = 0;
  uint64_t remaining_pressure_area = 0;
  bool has_covering_spill = false;
  for (uint16_t i = 0; i < conflict_assignment_count; ++i) {
    const uint32_t assignment_index = assignment_indices[i];
    const loom_low_allocation_assignment_t* conflict_assignment =
        &context->assignment_map->assignments[assignment_index];
    bool can_spill = false;
    loom_low_allocation_class_capacity_t spill_capacity = {0};
    IREE_RETURN_IF_ERROR(loom_low_allocation_spill_search_assignment_capacity(
        context, conflict_assignment, &can_spill, &spill_capacity));
    const loom_low_allocation_assignment_t* assignment = conflict_assignment;
    uint32_t spill_assignment_index = assignment_index;
    uint64_t assignment_traffic_cost = 0;
    if (!can_spill) {
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_spill_search_find_covering_spill_assignment(
              context, conflict_assignment, candidate.start_point,
              &spill_assignment_index, &spill_capacity,
              &assignment_traffic_cost));
      if (spill_assignment_index == UINT32_MAX) {
        *out_blocked = true;
        break;
      }
      if (!has_covering_spill) {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_spill_search_begin_covering_spill_selection(
                context->active_set, assignment_indices, assignment_count,
                workspace, arena));
      }
      has_covering_spill = true;
      assignment =
          &context->assignment_map->assignments[spill_assignment_index];
    }
    ignored_value_ids[ignored_value_count++] = conflict_assignment->value_id;
    uint32_t active_position = UINT32_MAX;
    if (has_covering_spill) {
      active_position =
          context->active_set->entries[spill_assignment_index].position;
      IREE_ASSERT_NE(active_position, UINT32_MAX);
      if (iree_bitmap_test(workspace->candidate_selected_assignments,
                           active_position)) {
        continue;
      }
    }
    if (can_spill) {
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_spill_search_assignment_spill_traffic_cost(
              context, assignment, &spill_capacity, &assignment_traffic_cost));
    }
    if (!interval_requires_register && assignment->end_point <= interval_end) {
      *out_blocked = true;
      break;
    }
    if (assignment_count == UINT16_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "active spill victim set exceeds uint16_t");
    }
    if (unit_count > UINT32_MAX - assignment->unit_count) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "active spill victim unit count overflow");
    }
    if (has_covering_spill) {
      iree_bitmap_set(workspace->candidate_selected_assignments,
                      active_position);
    }
    assignment_indices[assignment_count] = spill_assignment_index;
    ++assignment_count;
    unit_count += assignment->unit_count;
    traffic_cost =
        iree_math_saturating_add_u64(traffic_cost, assignment_traffic_cost);
    // Further victims cannot reduce traffic. Equal costs still compete on
    // victim count, footprint, remaining lifetime, and location order.
    if (traffic_cost > traffic_limit) {
      *out_blocked = true;
      break;
    }
    if (context->use_pressure_efficiency && !interval_requires_register) {
      remaining_pressure_area = iree_math_saturating_add_u64(
          remaining_pressure_area,
          iree_math_saturating_mul_u64(
              assignment->end_point - candidate.start_point,
              assignment->unit_count));
    }
    if (latest_end_point < assignment->end_point) {
      latest_end_point = assignment->end_point;
    }
  }

  if (!*out_blocked) {
    const loom_value_id_t* ignored_storage_lease_value_ids = ignored_value_ids;
    uint16_t ignored_storage_lease_value_count = ignored_value_count;
    if (has_covering_spill) {
      if (workspace->ignored_storage_lease_capacity < assignment_count) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            arena, /*existing_count=*/0, assignment_count,
            sizeof(*workspace->ignored_storage_lease_value_ids),
            &workspace->ignored_storage_lease_capacity,
            (void**)&workspace->ignored_storage_lease_value_ids));
      }
      for (uint16_t i = 0; i < assignment_count; ++i) {
        workspace->ignored_storage_lease_value_ids[i] =
            context->assignment_map->assignments[assignment_indices[i]]
                .value_id;
      }
      ignored_storage_lease_value_ids =
          workspace->ignored_storage_lease_value_ids;
      ignored_storage_lease_value_count = assignment_count;
    }
    // Every active conflict was collected and covered by the selected victims.
    // Fixed storage, retained reads, leases, and hard relations still apply.
    if (assignment_count == 0 ||
        loom_low_allocation_search_assignment_conflicts(
            context, &candidate, /*check_active_assignments=*/false,
            ignored_value_ids, ignored_value_count,
            ignored_storage_lease_value_ids, ignored_storage_lease_value_count,
            LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FOR_PRESSURE,
            /*shared_content=*/NULL)) {
      *out_blocked = true;
    }
  }

  if (!*out_blocked) {
    uint16_t released_value_count = 0;
    if (has_covering_spill) {
      for (uint16_t i = 0; i < ignored_value_count; ++i) {
        loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
        const bool has_ordinal =
            loom_low_allocation_assignment_map_value_ordinal_for_value(
                context->assignment_map, ignored_value_ids[i], &value_ordinal);
        IREE_ASSERT(has_ordinal);
        const uint32_t assignment_index =
            context->assignment_map
                ->assignment_indices_by_value_ordinal[value_ordinal];
        IREE_ASSERT_NE(assignment_index, UINT32_MAX);
        const uint32_t active_position =
            context->active_set->entries[assignment_index].position;
        IREE_ASSERT_NE(active_position, UINT32_MAX);
        if (!iree_bitmap_test(workspace->candidate_selected_assignments,
                              active_position)) {
          ignored_value_ids[released_value_count++] = ignored_value_ids[i];
        }
      }
    }
    *out_assignment_count = assignment_count;
    *out_unit_count = unit_count;
    *out_latest_end_point = latest_end_point;
    *out_traffic_cost = traffic_cost;
    *out_remaining_pressure_area = remaining_pressure_area;
    *out_released_value_count = released_value_count;
  }
  if (has_covering_spill) {
    for (uint16_t i = 0; i < assignment_count; ++i) {
      const uint32_t active_position =
          context->active_set->entries[assignment_indices[i]].position;
      IREE_ASSERT_NE(active_position, UINT32_MAX);
      iree_bitmap_reset(workspace->candidate_selected_assignments,
                        active_position);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_spill_search_find_victims(
    loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_allocation_class_capacity_t* capacity,
    bool interval_requires_register,
    loom_low_allocation_spill_search_workspace_t* workspace,
    iree_arena_allocator_t* arena,
    loom_low_allocation_spill_search_victim_set_t* out_victim_set) {
  *out_victim_set = (loom_low_allocation_spill_search_victim_set_t){0};
  const loom_low_reg_class_t* reg_class =
      &context->descriptor_set->reg_classes[capacity->descriptor_reg_class_id];
  const bool uses_explicit_physical_registers =
      loom_low_reg_class_uses_explicit_physical_registers(reg_class);
  if (!uses_explicit_physical_registers && capacity->is_bounded &&
      interval->unit_count > capacity->max_units) {
    return iree_ok_status();
  }
  if (context->active_set->count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "active allocation set exceeds uint16_t");
  }

  uint32_t last_base = 0;
  const uint32_t alignment = loom_low_allocation_live_range_interval_alignment(
      context->descriptor_set, context->liveness,
      context->placement->operand_constraints_by_interval, interval);
  if (!uses_explicit_physical_registers && capacity->is_bounded) {
    last_base = capacity->max_units - interval->unit_count;
  } else if (!uses_explicit_physical_registers) {
    const uint32_t search_limit =
        loom_low_allocation_target_constraints_assigned_location_search_limit(
            context->target_constraints, capacity->descriptor_reg_class_id,
            capacity->location_kind);
    const uint64_t aligned_limit =
        ((uint64_t)search_limit + alignment - 1u) & ~((uint64_t)alignment - 1u);
    if (aligned_limit > UINT32_MAX) {
      return iree_ok_status();
    }
    last_base = (uint32_t)aligned_limit;
  }

  if (context->active_set->count > workspace->capacity) {
    // Only the previous search used these contents. Geometric growth bounds
    // abandoned arena storage without copying obsolete candidates.
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, /*existing_count=*/0, context->active_set->count,
        sizeof(*workspace->candidate_assignment_indices) +
            sizeof(*workspace->best_assignment_indices) +
            sizeof(*workspace->ignored_value_ids),
        &workspace->capacity,
        (void**)&workspace->candidate_assignment_indices));
    workspace->best_assignment_indices =
        workspace->candidate_assignment_indices + workspace->capacity;
    workspace->ignored_value_ids =
        workspace->best_assignment_indices + workspace->capacity;
  }
  uint32_t* candidate_assignment_indices =
      workspace->candidate_assignment_indices;
  uint32_t* best_assignment_indices = workspace->best_assignment_indices;
  loom_value_id_t* ignored_value_ids = workspace->ignored_value_ids;

  uint16_t best_assignment_count = 0;
  uint32_t best_unit_count = 0;
  uint32_t best_latest_end_point = 0;
  uint64_t best_traffic_cost = 0;
  uint64_t best_remaining_pressure_area = 0;
  uint16_t best_released_value_count = 0;
  uint32_t best_location_base = 0;
  uint32_t best_location_ordinal = 0;
  const uint32_t explicit_pressure_limit =
      uses_explicit_physical_registers
          ? iree_min((uint32_t)reg_class->allocatable_count,
                     capacity->is_bounded ? capacity->max_units : UINT32_MAX)
          : 0;
  const bool uses_candidate_ordinals =
      uses_explicit_physical_registers &&
      loom_low_reg_class_uses_contiguous_physical_register_candidates(
          reg_class);
  const uint64_t candidate_count =
      uses_explicit_physical_registers
          ? loom_low_allocation_storage_explicit_location_candidate_count(
                context->descriptor_set, capacity->descriptor_reg_class_id,
                interval->unit_count)
          : (uint64_t)last_base / alignment + 1u;
  const bool use_pressure_efficiency =
      context->use_pressure_efficiency && !interval_requires_register;
  loom_low_placement_value_component_t component = {0};
  loom_value_ordinal_t candidate_origin = LOOM_VALUE_ORDINAL_INVALID;
  if (context->placement->hard_register_ordinal_component_count != 0) {
    const loom_value_ordinal_t value_ordinal =
        loom_module_value_ordinal_scratch_lookup(context->module,
                                                 interval->value_id);
    component = loom_low_placement_hard_register_ordinal_component(
        context->placement, value_ordinal);
    const loom_value_ordinal_t* origins =
        context->placement->tied_storage_origins_by_value_ordinal;
    candidate_origin = origins != NULL ? origins[value_ordinal] : value_ordinal;
  }
  for (uint64_t candidate_index = 0; candidate_index < candidate_count;
       ++candidate_index) {
    uint32_t candidate_ordinal = (uint32_t)candidate_index;
    uint32_t base = candidate_ordinal * alignment;
    if (uses_explicit_physical_registers) {
      uint32_t pressure_extent = 0;
      if (!loom_low_allocation_storage_explicit_location_candidate(
              context->descriptor_set, capacity->descriptor_reg_class_id,
              interval->unit_count, (uint32_t)candidate_index, &base,
              &candidate_ordinal, &pressure_extent,
              /*out_packing_rank=*/NULL) ||
          pressure_extent > explicit_pressure_limit ||
          (uses_candidate_ordinals && base % alignment != 0)) {
        continue;
      }
    }
    uint16_t candidate_assignment_count = 0;
    uint32_t candidate_unit_count = 0;
    uint32_t candidate_latest_end_point = 0;
    uint64_t candidate_traffic_cost = 0;
    uint64_t candidate_remaining_pressure_area = 0;
    uint16_t candidate_released_value_count = 0;
    bool blocked = false;
    // Pressure-area denominators may grow with later victims, so raw traffic
    // alone bounds only absolute-cost ranking.
    const uint64_t traffic_limit =
        !use_pressure_efficiency && best_assignment_count != 0
            ? best_traffic_cost
            : UINT64_MAX;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_spill_search_collect_active_spill_victim_set(
            context, interval, capacity, component, candidate_origin, base,
            interval_requires_register, traffic_limit,
            candidate_assignment_indices, ignored_value_ids, workspace, arena,
            &candidate_assignment_count, &candidate_unit_count,
            &candidate_latest_end_point, &candidate_traffic_cost,
            &candidate_remaining_pressure_area, &candidate_released_value_count,
            &blocked));
    if (!blocked &&
        loom_low_allocation_spill_search_spill_victim_set_is_better(
            use_pressure_efficiency, candidate_assignment_count,
            candidate_unit_count, candidate_latest_end_point,
            candidate_traffic_cost, candidate_remaining_pressure_area,
            candidate_ordinal, best_assignment_count, best_unit_count,
            best_latest_end_point, best_traffic_cost,
            best_remaining_pressure_area, best_location_ordinal)) {
      best_assignment_count = candidate_assignment_count;
      best_unit_count = candidate_unit_count;
      best_latest_end_point = candidate_latest_end_point;
      best_traffic_cost = candidate_traffic_cost;
      best_remaining_pressure_area = candidate_remaining_pressure_area;
      best_released_value_count = candidate_released_value_count;
      best_location_base = base;
      best_location_ordinal = candidate_ordinal;
      memcpy(best_assignment_indices, candidate_assignment_indices,
             (iree_host_size_t)candidate_assignment_count *
                 sizeof(*best_assignment_indices));
      if (candidate_released_value_count != 0) {
        if (workspace->released_value_capacity < workspace->capacity) {
          IREE_RETURN_IF_ERROR(iree_arena_grow_array(
              arena, /*existing_count=*/0, workspace->capacity,
              sizeof(*workspace->best_released_value_ids),
              &workspace->released_value_capacity,
              (void**)&workspace->best_released_value_ids));
        }
        memcpy(workspace->best_released_value_ids, ignored_value_ids,
               (iree_host_size_t)candidate_released_value_count *
                   sizeof(*workspace->best_released_value_ids));
      }
    }
  }

  if (best_assignment_count == 0) {
    return iree_ok_status();
  }
  if (!interval_requires_register) {
    const loom_low_allocation_assignment_t incoming =
        loom_low_allocation_search_candidate_assignment(
            context, interval, capacity->descriptor_reg_class_id,
            capacity->location_kind, best_location_base, interval->unit_count);
    uint64_t incoming_traffic_cost = 0;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_spill_search_assignment_spill_traffic_cost(
            context, &incoming, capacity, &incoming_traffic_cost));
    // A spilled result still needs physical storage at its definition. Charge
    // one spill/reload round trip for that temporary footprint instead of
    // treating an incoming spill as an immediately free register assignment.
    const uint64_t definition_bytes =
        ((uint64_t)incoming.unit_count * capacity->spill_unit_bits + 7u) / 8u;
    incoming_traffic_cost = iree_math_saturating_add_u64(incoming_traffic_cost,
                                                         definition_bytes * 2u);
    // Compare traffic per remaining register lifetime. Spilling the incoming
    // value can require another victim when its definition is materialized;
    // a longer-lived active victim also removes pressure at later definitions.
    const uint64_t active_pressure =
        (uint64_t)(best_latest_end_point - incoming.start_point) *
        best_unit_count;
    const uint64_t incoming_pressure =
        (uint64_t)iree_max(incoming.end_point - incoming.start_point, 1u) *
        incoming.unit_count;
    uint64_t active_high = 0, active_low = 0;
    uint64_t incoming_high = 0, incoming_low = 0;
    iree_math_mul_u64_to_u128(best_traffic_cost, incoming_pressure,
                              &active_high, &active_low);
    iree_math_mul_u64_to_u128(incoming_traffic_cost, active_pressure,
                              &incoming_high, &incoming_low);
    if (active_high > incoming_high ||
        (active_high == incoming_high && active_low > incoming_low)) {
      return iree_ok_status();
    }
  }
  *out_victim_set = (loom_low_allocation_spill_search_victim_set_t){
      .location_base = best_location_base,
      .assignment_indices = best_assignment_indices,
      .assignment_count = best_assignment_count,
      .released_value_ids = workspace->best_released_value_ids,
      .released_value_count = best_released_value_count,
      .found = true,
  };
  return iree_ok_status();
}
