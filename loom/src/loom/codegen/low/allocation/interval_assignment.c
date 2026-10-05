// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/interval_assignment.h"

#include "loom/analysis/consumption.h"
#include "loom/codegen/low/allocation/active_capacity.h"
#include "loom/codegen/low/allocation/active_set.h"
#include "loom/codegen/low/allocation/coalescing.h"
#include "loom/codegen/low/allocation/interval_order.h"
#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/physical_domains.h"
#include "loom/codegen/low/allocation/scalar_packing.h"
#include "loom/codegen/low/allocation/search.h"
#include "loom/codegen/low/allocation/spill_plan.h"
#include "loom/codegen/low/allocation/spill_traffic.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/ir/local_value_domain.h"

typedef struct loom_low_allocation_spill_decision_t {
  // Assignment selected for spilling, in decision order.
  uint32_t assignment_index;
  // Register budget active when this spill was selected.
  uint32_t budget_units;
} loom_low_allocation_spill_decision_t;

typedef struct loom_low_allocation_interval_assignment_state_t {
  // Caller-provided facts and mutable owner state.
  const loom_low_allocation_interval_assignment_context_t* context;
  // Owns working indexes and decisions that do not escape assignment.
  iree_arena_allocator_t* scratch_arena;
  // Scalar/aggregate lifetime preferences shared by every location query.
  loom_low_allocation_scalar_packing_t scalar_packing;
  // Full-scalar physical candidate preferences retained before assignment.
  loom_low_allocation_physical_domains_t physical_domains;
  // Reusable peer snapshots, private to this assignment attempt.
  loom_low_allocation_preference_workspace_t preference_workspace;
  // Reusable consumed-value query for the allocated function body.
  loom_consumption_region_query_t function_consumption_query;
  // Reusable consumed-value query for the current nested relation region.
  loom_consumption_region_query_t nested_consumption_query;
  // Assignment-index window still live at the current interval start.
  loom_low_allocation_active_set_t active;
  // Victim-search arrays reused after each selected set has been consumed.
  loom_low_allocation_search_workspace_t search_workspace;
  // Cached predicted spill traffic, dense by liveness value ordinal.
  loom_low_allocation_spill_plan_traffic_t* spill_traffic_by_value_ordinal;
  // Lazily allocated decision array. Each allocatable interval spills at most
  // once, so the interval count is a fixed upper bound without growth.
  loom_low_allocation_spill_decision_t* spill_decisions;
  // Upper bound for assignments and spill decisions.
  iree_host_size_t interval_count;
  // Next free spill-slot allocation unit.
  uint32_t next_spill_slot;
  // Mutable assignment, spill, remark, and lookup state being built.
  loom_low_allocation_interval_assignment_result_t result;
} loom_low_allocation_interval_assignment_state_t;

static uint32_t
loom_low_allocation_interval_assignment_unit_point_start_for_value_ordinal(
    const loom_low_allocation_interval_assignment_state_t* state,
    loom_value_ordinal_t value_ordinal) {
  return loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
      state->context->unit_liveness, state->context->liveness, value_ordinal);
}

static uint32_t
loom_low_allocation_interval_assignment_max_unit_end_point_for_interval(
    const loom_low_allocation_interval_assignment_state_t* state,
    const loom_liveness_interval_t* interval,
    loom_value_ordinal_t value_ordinal) {
  const loom_low_allocation_assignment_t candidate = {
      .value_id = interval->value_id,
      .start_point = state->context->unit_liveness->values[value_ordinal]
                         .acquisition_start_point,
      .end_point =
          loom_low_allocation_live_range_interval_storage_end_point(interval),
      .unit_count = interval->unit_count,
      .unit_point_start =
          loom_low_allocation_interval_assignment_unit_point_start_for_value_ordinal(
              state, value_ordinal),
  };
  return loom_low_allocation_live_range_assignment_max_unit_end_point(
      state->context->unit_liveness->end_points,
      state->context->unit_liveness->point_count, &candidate);
}

static loom_low_allocation_search_context_t
loom_low_allocation_interval_assignment_search_context(
    loom_low_allocation_interval_assignment_state_t* state) {
  return (loom_low_allocation_search_context_t){
      .module = state->context->module,
      .cfg_graph = state->context->function_cfg_graph,
      .descriptor_set = state->context->target->descriptor_set,
      .liveness = state->context->liveness,
      .unit_liveness = state->context->unit_liveness,
      .target_constraints = state->context->target_constraints,
      .residency = state->context->residency,
      .assignment_map = &state->result.assignment_map,
      .placement = state->context->placement,
      .preferences = state->context->preferences,
      .preference_workspace = &state->preference_workspace,
      .active_set = &state->active,
      .storage_leases = state->context->storage_leases,
      .required_register_values = state->context->required_register_values,
      .spill_traffic_by_value_ordinal = state->spill_traffic_by_value_ordinal,
      .scalar_packing = state->scalar_packing,
      .physical_domains = &state->physical_domains,
  };
}

static uint32_t loom_low_allocation_interval_assignment_peak_live_units(
    const loom_liveness_analysis_t* liveness,
    loom_liveness_value_class_t value_class, uint32_t fallback_units) {
  for (iree_host_size_t i = 0; i < liveness->pressure_summary_count; ++i) {
    const loom_liveness_pressure_summary_t* summary =
        &liveness->pressure_summaries[i];
    if (loom_liveness_value_class_equal(summary->value_class, value_class)) {
      return summary->peak_live_units;
    }
  }
  return fallback_units;
}

// Incoming locations are preferences for the complete interval, not fixed
// storage. In particular, implicit instruction writes can force entry transport
// even when no other SSA value occupies the incoming register.
static bool loom_low_allocation_interval_assignment_find_entry_location(
    loom_low_allocation_interval_assignment_state_t* state,
    loom_low_allocation_search_context_t* search_context,
    const loom_liveness_interval_t* interval,
    loom_value_ordinal_t value_ordinal,
    const loom_low_allocation_class_capacity_t* capacity, uint32_t* out_base) {
  const loom_low_allocation_interval_assignment_context_t* context =
      state->context;
  if (value_ordinal >= context->entry_location_count) {
    return false;
  }
  const loom_low_allocation_entry_location_t* entry =
      &context->entry_locations[value_ordinal];
  if (entry->location_kind != capacity->location_kind) {
    return false;
  }
  const loom_low_descriptor_set_t* descriptors =
      context->target->descriptor_set;
  const loom_low_reg_class_t* reg_class =
      &descriptors->reg_classes[capacity->descriptor_reg_class_id];
  if (loom_low_reg_class_uses_explicit_physical_registers(reg_class)) {
    uint32_t first_candidate_ordinal = 0;
    uint32_t pressure_extent = 0;
    if (!loom_low_allocation_storage_explicit_physical_register_view(
            descriptors, capacity->descriptor_reg_class_id,
            entry->location_base, interval->unit_count,
            &first_candidate_ordinal, &pressure_extent) ||
        (capacity->is_bounded && pressure_extent > capacity->max_units)) {
      return false;
    }
  } else {
    const uint32_t alignment =
        loom_low_allocation_live_range_interval_alignment(
            descriptors, context->liveness, context->placement, interval);
    if (entry->location_base % alignment != 0 ||
        (capacity->is_bounded &&
         (uint64_t)entry->location_base + interval->unit_count >
             capacity->max_units)) {
      return false;
    }
  }
  if (loom_low_allocation_search_location_conflicts(
          search_context, interval, capacity->descriptor_reg_class_id,
          entry->location_kind, entry->location_base, interval->unit_count,
          /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0,
          /*ignored_storage_lease_value_ids=*/NULL,
          /*ignored_storage_lease_value_count=*/0,
          LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN)) {
    return false;
  }
  *out_base = entry->location_base;
  return true;
}

static bool loom_low_allocation_interval_assignment_align_up_u32(
    uint32_t value, uint32_t alignment, uint32_t* out_value) {
  if (alignment <= 1) {
    *out_value = value;
    return true;
  }
  const uint32_t remainder = value % alignment;
  if (remainder == 0) {
    *out_value = value;
    return true;
  }
  const uint32_t increment = alignment - remainder;
  if (value > UINT32_MAX - increment) {
    return false;
  }
  *out_value = value + increment;
  return true;
}

static loom_low_allocation_assignment_t
loom_low_allocation_interval_assignment_failure_candidate(
    const loom_low_allocation_interval_assignment_state_t* state,
    const loom_liveness_interval_t* interval,
    loom_value_ordinal_t value_ordinal,
    const loom_low_allocation_class_capacity_t* capacity,
    uint32_t location_base) {
  const loom_liveness_segment_range_t segment_range =
      loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
          state->context->unit_liveness, state->context->liveness,
          value_ordinal);
  loom_low_allocation_assignment_t candidate = {
      .value_id = interval->value_id,
      .descriptor_reg_class_id = capacity->descriptor_reg_class_id,
      .start_point = state->context->unit_liveness->values[value_ordinal]
                         .acquisition_start_point,
      .end_point =
          loom_low_allocation_live_range_interval_storage_end_point(interval),
      .liveness_segments = segment_range,
      .unit_count = interval->unit_count,
      .location_kind = capacity->location_kind,
      .location_base = location_base,
      .location_count = interval->unit_count,
      .unit_point_start =
          loom_low_allocation_interval_assignment_unit_point_start_for_value_ordinal(
              state, value_ordinal),
  };
  candidate.end_point =
      loom_low_allocation_interval_assignment_max_unit_end_point_for_interval(
          state, interval, value_ordinal);
  return candidate;
}

static void loom_low_allocation_interval_assignment_failure_set_conflict(
    loom_low_allocation_failure_t* failure, uint32_t assignment_index,
    const loom_low_allocation_assignment_t* assignment) {
  failure->conflict_assignment_index = assignment_index;
  failure->conflict_value_id = assignment->value_id;
  failure->conflict_start_point = assignment->start_point;
  failure->conflict_end_point = assignment->end_point;
  failure->conflict_location_kind = assignment->location_kind;
  failure->conflict_location_base = assignment->location_base;
  failure->conflict_location_count = assignment->location_count;
}

static iree_status_t loom_low_allocation_interval_assignment_record_failure(
    loom_low_allocation_interval_assignment_state_t* state,
    const loom_liveness_interval_t* interval,
    loom_value_ordinal_t value_ordinal,
    const loom_low_allocation_class_capacity_t* capacity, uint32_t budget_units,
    bool interval_requires_register, iree_string_view_t failure_code) {
  loom_low_allocation_failure_t failure = {
      .failure_code = failure_code,
      .op = loom_low_diagnostic_value_origin_op(state->context->module,
                                                interval->value_id,
                                                state->context->function_op),
      .value_id = interval->value_id,
      .value_class = interval->value_class,
      .descriptor_reg_class_id = capacity->descriptor_reg_class_id,
      .start_point = state->context->unit_liveness->values[value_ordinal]
                         .acquisition_start_point,
      .end_point =
          loom_low_allocation_interval_assignment_max_unit_end_point_for_interval(
              state, interval, value_ordinal),
      .required_unit_count = interval->unit_count,
      .budget_units = budget_units,
      .peak_live_units =
          loom_low_allocation_interval_assignment_peak_live_units(
              state->context->liveness, interval->value_class,
              interval->unit_count),
      .location_kind = capacity->location_kind,
      .location_base = UINT32_MAX,
      .location_count = 0,
      .blocking_kind = LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_UNKNOWN,
      .conflict_assignment_index = UINT32_MAX,
      .conflict_value_id = LOOM_VALUE_ID_INVALID,
      .conflict_start_point = UINT32_MAX,
      .conflict_end_point = UINT32_MAX,
      .conflict_location_kind = LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED,
      .conflict_location_base = UINT32_MAX,
      .conflict_location_count = 0,
  };

  if (capacity->is_bounded && interval->unit_count > capacity->max_units) {
    failure.blocking_kind =
        LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_INTERVAL_EXCEEDS_BUDGET;
    state->context->target_constraints->failure = failure;
    return iree_ok_status();
  }

  const uint32_t alignment = loom_low_allocation_live_range_interval_alignment(
      state->context->target->descriptor_set, state->context->liveness,
      state->context->placement, interval);
  const loom_low_reg_class_t* reg_class =
      &state->context->target->descriptor_set
           ->reg_classes[capacity->descriptor_reg_class_id];
  const bool uses_explicit_physical_registers =
      loom_low_reg_class_uses_explicit_physical_registers(reg_class);
  uint32_t last_base = 0;
  if (!uses_explicit_physical_registers && capacity->is_bounded) {
    last_base = capacity->max_units - interval->unit_count;
  } else if (!uses_explicit_physical_registers) {
    const uint32_t search_limit =
        loom_low_allocation_target_constraints_assigned_location_search_limit(
            state->context->target_constraints,
            capacity->descriptor_reg_class_id, capacity->location_kind);
    if (!loom_low_allocation_interval_assignment_align_up_u32(
            search_limit, alignment, &last_base)) {
      failure.blocking_kind =
          LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_NO_ASSIGNABLE_LOCATION;
      state->context->target_constraints->failure = failure;
      return iree_ok_status();
    }
  }

  loom_low_allocation_search_context_t search_context =
      loom_low_allocation_interval_assignment_search_context(state);
  const uint32_t interval_end = failure.end_point;
  const uint32_t explicit_pressure_limit =
      uses_explicit_physical_registers
          ? iree_min((uint32_t)reg_class->allocatable_count,
                     capacity->is_bounded ? capacity->max_units : UINT32_MAX)
          : 0;
  const uint64_t candidate_count =
      uses_explicit_physical_registers
          ? state->context->target->descriptor_set->physical_register_count
          : (uint64_t)last_base / alignment + 1u;
  for (uint64_t candidate_index = 0; candidate_index < candidate_count;
       ++candidate_index) {
    uint32_t candidate_ordinal = (uint32_t)candidate_index;
    uint32_t base = candidate_ordinal * alignment;
    if (uses_explicit_physical_registers) {
      uint32_t pressure_extent = 0;
      base = (uint32_t)candidate_index;
      if (!loom_low_allocation_storage_explicit_physical_register_view(
              state->context->target->descriptor_set,
              capacity->descriptor_reg_class_id, base, interval->unit_count,
              &candidate_ordinal, &pressure_extent) ||
          pressure_extent > explicit_pressure_limit) {
        continue;
      }
    }
    loom_low_allocation_assignment_t candidate =
        loom_low_allocation_interval_assignment_failure_candidate(
            state, interval, value_ordinal, capacity, base);
    failure.location_base = base;
    failure.location_count = interval->unit_count;

    bool saw_active_conflict = false;
    for (iree_host_size_t i = 0; i < state->active.count; ++i) {
      const uint32_t assignment_index = state->active.assignment_indices[i];
      IREE_ASSERT_LT(assignment_index, state->result.assignment_count);
      const loom_low_allocation_assignment_t* assignment =
          &state->result.assignments[assignment_index];
      if (!loom_low_allocation_active_assignment_conflicts(
              state->context->target->descriptor_set,
              state->context->unit_liveness, assignment, &candidate,
              /*ignored_value_ids=*/NULL,
              /*ignored_value_count=*/0)) {
        continue;
      }
      saw_active_conflict = true;
      if (failure.conflict_value_id == LOOM_VALUE_ID_INVALID) {
        loom_low_allocation_interval_assignment_failure_set_conflict(
            &failure, assignment_index, assignment);
      }
      bool can_spill = false;
      IREE_RETURN_IF_ERROR(loom_low_allocation_search_assignment_spill_capacity(
          &search_context, assignment, &can_spill, NULL));
      if (!can_spill || (!interval_requires_register &&
                         assignment->end_point <= interval_end)) {
        failure.blocking_kind =
            LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_ACTIVE_ASSIGNMENT;
        loom_low_allocation_interval_assignment_failure_set_conflict(
            &failure, assignment_index, assignment);
        state->context->target_constraints->failure = failure;
        return iree_ok_status();
      }
    }

    if (loom_low_allocation_target_constraints_fixed_storage_conflicts(
            state->context->target_constraints, state->context->unit_liveness,
            &candidate,
            /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0) ||
        loom_low_allocation_target_constraints_reserved_range_conflicts(
            state->context->target_constraints,
            capacity->descriptor_reg_class_id, capacity->location_kind, base,
            interval->unit_count) ||
        loom_low_allocation_storage_lease_state_conflicts(
            state->context->storage_leases,
            state->context->target->descriptor_set, state->context->liveness,
            &candidate, /*ignored_value_ids=*/NULL,
            /*ignored_value_count=*/0,
            LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN)) {
      failure.blocking_kind =
          LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_LOCATION_CONSTRAINT;
      state->context->target_constraints->failure = failure;
      return iree_ok_status();
    }

    if (!saw_active_conflict) {
      failure.blocking_kind =
          LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_NO_ASSIGNABLE_LOCATION;
      state->context->target_constraints->failure = failure;
      return iree_ok_status();
    }
  }

  failure.blocking_kind =
      LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_NO_ASSIGNABLE_LOCATION;
  state->context->target_constraints->failure = failure;
  return iree_ok_status();
}

static const loom_low_allocation_assignment_t*
loom_low_allocation_interval_assignment_tied_source_assignment(
    const loom_low_allocation_interval_assignment_state_t* state,
    loom_value_ordinal_t value_ordinal) {
  const loom_low_placement_table_t* placement = state->context->placement;
  const loom_low_placement_relation_range_t range =
      loom_low_placement_relation_range_for_value_ordinal(placement,
                                                          value_ordinal);
  for (uint32_t i = 0; i < range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &placement->relations[range.start + i];
    if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT) {
      continue;
    }
    const uint32_t assignment_index =
        state->result
            .assignment_indices_by_value_ordinal[relation->source_ordinal];
    IREE_ASSERT_NE(assignment_index, UINT32_MAX,
                   "a tied source must be assigned before its result");
    const loom_low_allocation_assignment_t* assignment =
        &state->result.assignments[assignment_index];
    return loom_low_allocation_assignment_is_register_like(assignment)
               ? assignment
               : NULL;
  }
  return NULL;
}

static iree_status_t loom_low_allocation_interval_assignment_consumption_query(
    loom_low_allocation_interval_assignment_state_t* state,
    const loom_region_t* region, loom_consumption_region_query_t** out_query) {
  if (region == state->context->body) {
    loom_consumption_region_query_t* query = &state->function_consumption_query;
    if (query->region != region) {
      loom_consumption_region_query_initialize_with_cfg_graph(
          state->context->module, region, state->context->function_cfg_graph,
          state->context->liveness, state->context->value_domain,
          state->scratch_arena, query);
    }
    *out_query = query;
    return iree_ok_status();
  }
  loom_consumption_region_query_t* query = &state->nested_consumption_query;
  if (query->region != region) {
    loom_consumption_region_query_initialize(state->context->module, region,
                                             state->scratch_arena, query);
  }
  *out_query = query;
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_interval_assignment_consumption_query_callback(
    void* user_data, const loom_region_t* region,
    loom_consumption_region_query_t** out_query) {
  return loom_low_allocation_interval_assignment_consumption_query(
      (loom_low_allocation_interval_assignment_state_t*)user_data, region,
      out_query);
}

static iree_status_t loom_low_allocation_interval_assignment_record_spill(
    loom_low_allocation_interval_assignment_state_t* state,
    uint32_t assignment_index,
    const loom_low_allocation_class_capacity_t* capacity,
    uint32_t retained_fixed_value_index_plus_one) {
  if (!state->spill_decisions) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, state->interval_count,
        sizeof(*state->spill_decisions), (void**)&state->spill_decisions));
  }
  state->spill_decisions[state->result.spill_count++] =
      (loom_low_allocation_spill_decision_t){
          .assignment_index = assignment_index,
          .budget_units =
              capacity->is_bounded ? capacity->max_units : UINT32_MAX,
      };
  if (retained_fixed_value_index_plus_one != 0) {
    loom_low_allocation_retained_fixed_values_t* retained =
        &state->result.retained_fixed_values;
    if (retained->next_indices_plus_one == NULL) {
      const iree_host_size_t fixed_value_count =
          state->context->target_constraints->fixed_value_count;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(state->context->arena, fixed_value_count,
                                    sizeof(*retained->next_indices_plus_one),
                                    (void**)&retained->next_indices_plus_one));
      memset(retained->next_indices_plus_one, 0,
             fixed_value_count * sizeof(*retained->next_indices_plus_one));
    }
    if (retained_fixed_value_index_plus_one != retained->last_index_plus_one &&
        retained->next_indices_plus_one[retained_fixed_value_index_plus_one -
                                        1] == 0) {
      if (retained->last_index_plus_one != 0) {
        retained->next_indices_plus_one[retained->last_index_plus_one - 1] =
            retained_fixed_value_index_plus_one;
      } else {
        retained->first_index_plus_one = retained_fixed_value_index_plus_one;
      }
      retained->last_index_plus_one = retained_fixed_value_index_plus_one;
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_interval_assignment_spill_active_assignment(
    loom_low_allocation_interval_assignment_state_t* state,
    uint32_t assignment_index,
    const loom_low_allocation_class_capacity_t* capacity) {
  loom_low_allocation_assignment_t* assignment =
      &state->result.assignments[assignment_index];
  if (assignment->unit_count > UINT32_MAX - state->next_spill_slot) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation spill slots exceed uint32_t range");
  }
  loom_low_allocation_active_set_remove(
      &state->active, state->result.assignments, state->result.assignment_count,
      assignment_index);
  assignment->location_kind = LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT;
  assignment->location_base = state->next_spill_slot;
  assignment->location_count = assignment->unit_count;
  state->next_spill_slot += assignment->unit_count;
  return loom_low_allocation_interval_assignment_record_spill(
      state, assignment_index, capacity,
      /*retained_fixed_value_index_plus_one=*/0);
}

static iree_status_t
loom_low_allocation_interval_assignment_spill_active_assignment_set(
    loom_low_allocation_interval_assignment_state_t* state,
    const uint32_t* assignment_indices, uint16_t assignment_count) {
  loom_low_allocation_search_context_t search_context =
      loom_low_allocation_interval_assignment_search_context(state);
  for (uint16_t i = 0; i < assignment_count; ++i) {
    const uint32_t assignment_index = assignment_indices[i];
    const loom_low_allocation_assignment_t* assignment =
        &state->result.assignments[assignment_index];
    loom_low_allocation_class_capacity_t assignment_capacity = {0};
    bool can_spill = false;
    IREE_RETURN_IF_ERROR(loom_low_allocation_search_assignment_spill_capacity(
        &search_context, assignment, &can_spill, &assignment_capacity));
    if (!can_spill) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "active spill victim set became stale while assigning value %u",
          (unsigned)assignment->value_id);
    }
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_interval_assignment_spill_active_assignment(
            state, assignment_index, &assignment_capacity));
  }
  return iree_ok_status();
}

static loom_low_allocation_assignment_t
loom_low_allocation_interval_assignment_prepare_assignment(
    const loom_low_allocation_interval_assignment_state_t* state,
    const loom_low_allocation_assignment_t* assignment,
    loom_value_ordinal_t value_ordinal) {
  loom_low_allocation_assignment_t stored_assignment = *assignment;
  if (stored_assignment.start_point ==
      state->context->unit_liveness->values[value_ordinal]
          .acquisition_start_point) {
    stored_assignment.liveness_segments =
        loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
            state->context->unit_liveness, state->context->liveness,
            value_ordinal);
  } else {
    // Optional storage reserved beyond the canonical component lifetime is not
    // represented by its sparse segments. Keep the range empty so conflicts
    // use the conservative linear storage lifetime.
    stored_assignment.liveness_segments = (loom_liveness_segment_range_t){0};
  }
  stored_assignment.unit_point_start =
      loom_low_allocation_interval_assignment_unit_point_start_for_value_ordinal(
          state, value_ordinal);
  stored_assignment.end_point =
      loom_low_allocation_live_range_assignment_max_unit_end_point(
          state->context->unit_liveness->end_points,
          state->context->unit_liveness->point_count, &stored_assignment);
  return stored_assignment;
}

// Publishes one value's assignment and attached lease records. Required ties
// inherit an existing physical reservation; only fresh acquisitions release
// conflicting leases before reaching this common owner.
static uint32_t loom_low_allocation_interval_assignment_publish_assignment(
    loom_low_allocation_interval_assignment_state_t* state,
    const loom_low_allocation_assignment_t* assignment,
    loom_value_ordinal_t value_ordinal) {
  const uint32_t assignment_index = (uint32_t)state->result.assignment_count;
  state->result.assignments[state->result.assignment_count++] = *assignment;
  state->result.assignment_map.assignment_count =
      state->result.assignment_count;
  state->result.assignment_indices_by_value_ordinal[value_ordinal] =
      assignment_index;
  loom_low_allocation_target_constraints_record_location_extent(
      state->context->target_constraints, assignment->descriptor_reg_class_id,
      assignment->location_kind, assignment->location_base,
      assignment->location_count);
  loom_low_allocation_storage_lease_state_record_assignment(
      state->context->storage_leases, state->context->target->descriptor_set,
      state->context->liveness, assignment, assignment_index, value_ordinal);
  // Spill slots are unique, monotonically assigned storage. They cannot
  // conflict with register candidates and need no active-set membership.
  if (loom_low_allocation_assignment_is_register_like(assignment)) {
    loom_low_allocation_active_set_insert(
        &state->active, state->context->target->descriptor_set,
        state->result.assignments, state->result.assignment_count,
        assignment_index);
  }
  return assignment_index;
}

static iree_status_t loom_low_allocation_interval_assignment_append_assignment(
    loom_low_allocation_interval_assignment_state_t* state,
    const loom_low_allocation_assignment_t* assignment,
    const loom_value_id_t* ignored_storage_lease_value_ids,
    uint16_t ignored_storage_lease_value_count,
    uint32_t* out_assignment_index) {
  const loom_value_ordinal_t value_ordinal = loom_local_value_domain_ordinal(
      state->context->value_domain, assignment->value_id);
  const loom_low_allocation_assignment_t stored_assignment =
      loom_low_allocation_interval_assignment_prepare_assignment(
          state, assignment, value_ordinal);
  // Fixed locations are validated when resolved; search and coalescing only
  // select locations within the resolved capacity. Acquiring that selection
  // records completion actions without repeating input-boundary checks.
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_storage_lease_state_record_release_actions(
          state->context->storage_leases,
          state->context->target->descriptor_set, state->context->liveness,
          &stored_assignment, ignored_storage_lease_value_ids,
          ignored_storage_lease_value_count));
  const uint32_t assignment_index =
      loom_low_allocation_interval_assignment_publish_assignment(
          state, &stored_assignment, value_ordinal);
  if (out_assignment_index) {
    *out_assignment_index = assignment_index;
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_interval_assignment_append_assignment_callback(
    void* user_data, const loom_low_allocation_assignment_t* assignment,
    const loom_value_id_t* ignored_storage_lease_value_ids,
    uint16_t ignored_storage_lease_value_count) {
  return loom_low_allocation_interval_assignment_append_assignment(
      (loom_low_allocation_interval_assignment_state_t*)user_data, assignment,
      ignored_storage_lease_value_ids, ignored_storage_lease_value_count,
      /*out_assignment_index=*/NULL);
}

static iree_status_t
loom_low_allocation_interval_assignment_assign_fixed_interval(
    loom_low_allocation_interval_assignment_state_t* state,
    const loom_liveness_interval_t* interval, bool* out_assigned) {
  *out_assigned = false;
  const loom_low_allocation_resolved_fixed_value_t* fixed_value =
      loom_low_allocation_target_constraints_preassigned_fixed_value_for_value(
          state->context->target_constraints, interval->value_id);
  if (!fixed_value) {
    return iree_ok_status();
  }

  loom_low_allocation_search_context_t search_context =
      loom_low_allocation_interval_assignment_search_context(state);
  // A fixed destination may reuse dead storage after releasing an asynchronous
  // lease. The normal append path records the required completion action.
  if (loom_low_allocation_search_assignment_conflicts(
          &search_context, &fixed_value->assignment,
          /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0,
          /*ignored_storage_lease_value_ids=*/NULL,
          /*ignored_storage_lease_value_count=*/0,
          LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED)) {
    return loom_low_allocation_target_constraints_emit_fixed_value_conflict(
        state->context->target_constraints, &fixed_value->assignment);
  }

  IREE_RETURN_IF_ERROR(
      loom_low_allocation_interval_assignment_append_assignment(
          state, &fixed_value->assignment,
          /*ignored_storage_lease_value_ids=*/NULL,
          /*ignored_storage_lease_value_count=*/0, NULL));
  *out_assigned = true;
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_interval_assignment_initialize_result_storage(
    loom_low_allocation_interval_assignment_state_t* state,
    const loom_low_allocation_interval_order_t* order) {
  if (state->context->liveness->value_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->context->arena, state->context->liveness->value_count,
        sizeof(*state->result.assignment_indices_by_value_ordinal),
        (void**)&state->result.assignment_indices_by_value_ordinal));
    for (iree_host_size_t i = 0; i < state->context->liveness->value_count;
         ++i) {
      state->result.assignment_indices_by_value_ordinal[i] = UINT32_MAX;
    }
  }
  if (order->interval_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->context->arena, order->interval_count,
                                  sizeof(*state->result.assignments),
                                  (void**)&state->result.assignments));
    loom_low_allocation_active_capacity_t capacity = {0};
    IREE_RETURN_IF_ERROR(loom_low_allocation_active_capacity_calculate(
        state->context->target->descriptor_set, state->context->liveness,
        state->context->unit_liveness, state->context->placement,
        state->scratch_arena->block_pool, &capacity));
    IREE_RETURN_IF_ERROR(loom_low_allocation_active_set_initialize(
        order->interval_count, capacity.program_point_count,
        capacity.unit_count, state->scratch_arena, &state->active));
  }

  state->result.assignment_map = (loom_low_allocation_assignment_map_t){
      .module = state->context->module,
      .liveness = state->context->liveness,
      .assignments = state->result.assignments,
      .assignment_count = 0,
      .assignment_indices_by_value_ordinal =
          state->result.assignment_indices_by_value_ordinal,
  };
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_interval_assignment_initialize_spill_traffic_cache(
    loom_low_allocation_interval_assignment_state_t* state) {
  if (state->spill_traffic_by_value_ordinal) {
    return iree_ok_status();
  }
  const iree_host_size_t value_count = state->context->liveness->value_count;
  if (value_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, value_count,
      sizeof(*state->spill_traffic_by_value_ordinal),
      (void**)&state->spill_traffic_by_value_ordinal));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    state->spill_traffic_by_value_ordinal[i] =
        (loom_low_allocation_spill_plan_traffic_t){
            .store_count = UINT32_MAX,
        };
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_interval_assignment_assign(
    loom_low_allocation_interval_assignment_state_t* state) {
  const loom_low_allocation_interval_assignment_context_t* context =
      state->context;
  loom_low_allocation_interval_order_t order = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_interval_order_build(
      context->target->descriptor_set, context->liveness,
      context->unit_liveness, context->placement, state->scratch_arena,
      &order));
  state->interval_count = order.interval_count;
  state->result.has_packable_aggregates = order.has_packable_aggregates;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_interval_assignment_initialize_result_storage(
          state, &order));
  if (order.interval_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_allocation_preference_workspace_initialize(
      context->preferences, state->scratch_arena,
      &state->preference_workspace));
  IREE_RETURN_IF_ERROR(loom_low_allocation_physical_domains_build(
      context->target->descriptor_set, context->liveness,
      context->unit_liveness, context->placement, state->scratch_arena,
      &state->physical_domains));
  if (context->search_strategy ==
      LOOM_LOW_ALLOCATION_SEARCH_STRATEGY_FRAGMENTATION_REPAIR) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_scalar_packing_build(
        context->target->descriptor_set, context->liveness, &order,
        state->scratch_arena, &state->scalar_packing));
  }

  for (iree_host_size_t i = 0; i < order.interval_count; ++i) {
    const loom_low_allocation_interval_order_entry_t* entry =
        &order.intervals[i];
    const loom_liveness_interval_t* interval = entry->interval;
    loom_low_allocation_active_set_expire(
        &state->active, state->result.assignments,
        state->result.assignment_count, entry->acquisition_start_point);

    const loom_value_ordinal_t value_ordinal = loom_local_value_domain_ordinal(
        context->value_domain, interval->value_id);
    if (state->result.assignment_indices_by_value_ordinal[value_ordinal] !=
        UINT32_MAX) {
      continue;
    }

    const loom_low_allocation_assignment_t* tied_source =
        loom_low_allocation_interval_assignment_tied_source_assignment(
            state, value_ordinal);
    if (tied_source != NULL) {
      // Placement and unit liveness established the whole component's location
      // constraints and reservation before its first member was assigned.
      // Verified ties preserve the source's complete register class and width.
      const loom_low_allocation_assignment_t inherited_assignment = {
          .value_id = interval->value_id,
          .descriptor_reg_class_id = tied_source->descriptor_reg_class_id,
          .start_point = entry->acquisition_start_point,
          .end_point =
              loom_low_allocation_live_range_interval_storage_end_point(
                  interval),
          .unit_count = interval->unit_count,
          .location_kind = tied_source->location_kind,
          .location_base = tied_source->location_base,
          .location_count = tied_source->location_count,
      };
      const loom_low_allocation_assignment_t assignment =
          loom_low_allocation_interval_assignment_prepare_assignment(
              state, &inherited_assignment, value_ordinal);
      loom_low_allocation_interval_assignment_publish_assignment(
          state, &assignment, value_ordinal);
      continue;
    }

    loom_low_allocation_search_context_t search_context =
        loom_low_allocation_interval_assignment_search_context(state);
    loom_low_allocation_coalescing_context_t coalescing_context = {
        .arena = state->scratch_arena,
        .liveness = context->liveness,
        .schedule = context->schedule,
        .placement = context->placement,
        .target_constraints = context->target_constraints,
        .assignment_map = &state->result.assignment_map,
        .search_context = &search_context,
        .append_assignment =
            loom_low_allocation_interval_assignment_append_assignment_callback,
        .consumption_query =
            loom_low_allocation_interval_assignment_consumption_query_callback,
        .user_data = state,
    };

    bool assigned_concat_source_interval = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_assign_concat_source_interval(
            &coalescing_context, interval, &assigned_concat_source_interval));
    if (assigned_concat_source_interval) {
      continue;
    }

    // Honor an already assigned consumer before choosing an incoming source.
    // In particular, a join feeding a loop retains the loop's storage instead
    // of adopting a different predecessor's location and copying on backedges.
    bool assigned_edge_source_interval = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_assign_edge_source_interval(
            &coalescing_context, interval, &assigned_edge_source_interval));
    if (assigned_edge_source_interval) {
      continue;
    }

    bool assigned_structural_interval = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_assign_structural_interval(
            &coalescing_context, interval, &assigned_structural_interval));
    if (assigned_structural_interval) {
      continue;
    }

    // Fixed bindings constrain every candidate but retain the same required
    // ties and structural ownership handoffs as unconstrained values.
    bool assigned_fixed_interval = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_interval_assignment_assign_fixed_interval(
            state, interval, &assigned_fixed_interval));
    if (context->target_constraints->error_count != 0) {
      return iree_ok_status();
    }
    if (assigned_fixed_interval) {
      continue;
    }

    loom_low_allocation_class_capacity_t capacity = {0};
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_target_constraints_interval_capacity(
            context->target_constraints, context->liveness, context->placement,
            interval, &capacity));
    if (interval->unit_count > UINT32_MAX - state->next_spill_slot) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "allocation spill slots exceed uint32_t range");
    }

    uint32_t location_base = 0;
    bool assigned = loom_low_allocation_interval_assignment_find_entry_location(
                        state, &search_context, interval, value_ordinal,
                        &capacity, &location_base) ||
                    loom_low_allocation_search_find_free_location(
                        &search_context, interval, capacity, &location_base);
    const uint32_t retained_fixed_value_index_plus_one =
        search_context.retained_fixed_value_index_plus_one;
    const bool requires_register =
        loom_low_allocation_storage_lease_state_value_has_records(
            context->storage_leases, context->liveness, interval->value_id) ||
        (interval->value_id < context->required_register_values.bit_count &&
         iree_bitmap_test(context->required_register_values,
                          interval->value_id)) ||
        loom_low_allocation_spill_traffic_interval_requires_register_location(
            context->module, interval);
    const bool interval_requires_register =
        !capacity.is_spillable || requires_register;
    if (!assigned) {
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_interval_assignment_initialize_spill_traffic_cache(
              state));
      search_context =
          loom_low_allocation_interval_assignment_search_context(state);
      loom_low_allocation_search_spill_victim_set_t victim_set = {0};
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_search_find_active_spill_victim_set(
              &search_context, interval, &capacity, interval_requires_register,
              &state->search_workspace, state->scratch_arena, &victim_set));
      if (victim_set.found) {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_interval_assignment_spill_active_assignment_set(
                state, victim_set.assignment_indices,
                victim_set.assignment_count));
        location_base = victim_set.location_base;
        assigned = true;
      }
    }
    if (!assigned && (!capacity.is_spillable || requires_register)) {
      const uint32_t budget_units =
          capacity.is_bounded ? capacity.max_units : UINT32_MAX;
      const iree_string_view_t failure_code =
          requires_register ? IREE_SV("spill-traffic-register-exhausted")
                            : IREE_SV("unspillable-register-exhausted");
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_interval_assignment_record_failure(
              state, interval, value_ordinal, &capacity, budget_units,
              requires_register, failure_code));
      ++context->target_constraints->error_count;
      return iree_ok_status();
    }

    const loom_low_allocation_assignment_t assignment = {
        .value_id = interval->value_id,
        .descriptor_reg_class_id = capacity.descriptor_reg_class_id,
        .start_point = entry->acquisition_start_point,
        .end_point =
            loom_low_allocation_live_range_interval_storage_end_point(interval),
        .unit_count = interval->unit_count,
        .location_kind = assigned ? capacity.location_kind
                                  : LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT,
        .location_base = assigned ? location_base : state->next_spill_slot,
        .location_count = interval->unit_count,
    };

    uint32_t assignment_index = 0;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_interval_assignment_append_assignment(
            state, &assignment, /*ignored_storage_lease_value_ids=*/NULL,
            /*ignored_storage_lease_value_count=*/0, &assignment_index));
    if (!assigned) {
      state->next_spill_slot += interval->unit_count;
      IREE_RETURN_IF_ERROR(loom_low_allocation_interval_assignment_record_spill(
          state, assignment_index, &capacity,
          retained_fixed_value_index_plus_one));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_interval_assignment_finalize_spills(
    loom_low_allocation_interval_assignment_state_t* state) {
  loom_low_allocation_interval_assignment_result_t* result = &state->result;
  if (!result->spill_count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->context->arena, result->spill_count, sizeof(*result->spill_plans),
      (void**)&result->spill_plans));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->context->arena, result->spill_count, sizeof(*result->remarks),
      (void**)&result->remarks));
  for (iree_host_size_t i = 0; i < result->spill_count; ++i) {
    const loom_low_allocation_spill_decision_t* decision =
        &state->spill_decisions[i];
    const loom_low_allocation_assignment_t* assignment =
        &result->assignments[decision->assignment_index];
    const loom_low_reg_class_t* reg_class =
        &state->context->target->descriptor_set
             ->reg_classes[assignment->descriptor_reg_class_id];
    IREE_RETURN_IF_ERROR(loom_low_allocation_spill_plan_record(
        state->context->module, state->context->function_cfg_graph, assignment,
        decision->assignment_index, reg_class->alloc_unit_bits,
        (loom_low_spill_slot_space_t)reg_class->spill_slot_space,
        result->spill_plans, &result->spill_plan_count,
        &result->spill_traffic_bytes));
    loom_low_allocation_spill_remark_record(
        result->remarks, &result->remark_count, decision->assignment_index,
        decision->budget_units, assignment->unit_count);
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_interval_assignment_build(
    const loom_low_allocation_interval_assignment_context_t* context,
    iree_arena_allocator_t* scratch_arena,
    loom_low_allocation_interval_assignment_result_t* out_result) {
  *out_result = (loom_low_allocation_interval_assignment_result_t){0};
  const iree_arena_checkpoint_t scratch_checkpoint =
      iree_arena_checkpoint_save(scratch_arena);
  loom_low_allocation_interval_assignment_state_t state = {
      .context = context,
      .scratch_arena = scratch_arena,
  };
  iree_status_t status = loom_low_allocation_interval_assignment_assign(&state);
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_interval_assignment_finalize_spills(&state);
  }
  if (iree_status_is_ok(status)) {
    *out_result = state.result;
  }
  iree_arena_checkpoint_restore(&scratch_checkpoint);
  return status;
}
