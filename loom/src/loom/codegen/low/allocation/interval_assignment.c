// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/interval_assignment.h"

#include <string.h>

#include "loom/analysis/consumption.h"
#include "loom/codegen/low/allocation/active_set.h"
#include "loom/codegen/low/allocation/coalescing.h"
#include "loom/codegen/low/allocation/interval_order.h"
#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/search.h"
#include "loom/codegen/low/allocation/spill_plan.h"
#include "loom/codegen/low/allocation/spill_traffic.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/diagnostics.h"

typedef struct loom_low_allocation_interval_assignment_state_t {
  // Caller-provided facts and mutable owner state.
  const loom_low_allocation_interval_assignment_context_t* context;
  // Reusable consumed-value query for the allocated function body.
  loom_consumption_region_query_t function_consumption_query;
  // Reusable consumed-value query for the current nested relation region.
  loom_consumption_region_query_t nested_consumption_query;
  // Assignment-index window still live at the current interval start.
  loom_low_allocation_active_set_t active;
  // Cached predicted spill traffic, dense by liveness value ordinal.
  loom_low_allocation_spill_plan_traffic_t* spill_traffic_by_value_ordinal;
  // Next free spill-slot allocation unit.
  uint32_t next_spill_slot;
  // Mutable assignment, spill, remark, and lookup state being built.
  loom_low_allocation_interval_assignment_result_t result;
} loom_low_allocation_interval_assignment_state_t;

static bool loom_low_allocation_interval_assignment_value_ordinal_for_value(
    const loom_low_allocation_interval_assignment_state_t* state,
    loom_value_id_t value_id, loom_value_ordinal_t* out_value_ordinal) {
  return loom_low_allocation_assignment_map_value_ordinal_for_value(
      &state->result.assignment_map, value_id, out_value_ordinal);
}

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
      .value_class = interval->value_class,
      .start_point = interval->start_point,
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
      .residency_model = state->context->residency_model,
      .assignment_map = &state->result.assignment_map,
      .placement = state->context->placement,
      .active_set = &state->active,
      .storage_leases = state->context->storage_leases,
      .required_register_values = state->context->required_register_values,
      .spill_traffic_by_value_ordinal = state->spill_traffic_by_value_ordinal,
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
      .value_class = interval->value_class,
      .descriptor_reg_class_id = capacity->descriptor_reg_class_id,
      .start_point = interval->start_point,
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
      .value_id = interval->value_id,
      .value_class = interval->value_class,
      .descriptor_reg_class_id = capacity->descriptor_reg_class_id,
      .start_point = interval->start_point,
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
    state->result.failure = failure;
    return iree_ok_status();
  }

  const uint32_t alignment =
      iree_max((uint32_t)1, loom_low_allocation_live_range_interval_alignment(
                                interval, capacity->reg_class_flags));
  uint32_t last_base = 0;
  if (capacity->is_bounded) {
    last_base = capacity->max_units - interval->unit_count;
  } else {
    const uint32_t search_limit =
        loom_low_allocation_target_constraints_assigned_location_search_limit(
            state->context->target_constraints,
            capacity->descriptor_reg_class_id, capacity->location_kind);
    if (!loom_low_allocation_interval_assignment_align_up_u32(
            search_limit, alignment, &last_base)) {
      failure.blocking_kind =
          LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_NO_ASSIGNABLE_LOCATION;
      state->result.failure = failure;
      return iree_ok_status();
    }
  }

  loom_low_allocation_search_context_t search_context =
      loom_low_allocation_interval_assignment_search_context(state);
  const uint32_t interval_end =
      loom_low_allocation_live_range_interval_storage_end_point(interval);
  for (uint32_t base = 0; base <= last_base;) {
    loom_low_allocation_assignment_t candidate =
        loom_low_allocation_interval_assignment_failure_candidate(
            state, interval, value_ordinal, capacity, base);
    failure.location_base = base;
    failure.location_count = interval->unit_count;

    bool saw_active_conflict = false;
    for (iree_host_size_t i = 0; i < state->active.count; ++i) {
      const uint32_t assignment_index =
          state->active.assignment_indices[state->active.start + i];
      IREE_ASSERT_LT(assignment_index, state->result.assignment_count);
      const loom_low_allocation_assignment_t* assignment =
          &state->result.assignments[assignment_index];
      if (!loom_low_allocation_active_assignment_conflicts(
              state->context->target->descriptor_set, state->context->liveness,
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
        state->result.failure = failure;
        return iree_ok_status();
      }
    }

    if (loom_low_allocation_target_constraints_fixed_value_conflicts(
            state->context->target_constraints, state->context->liveness,
            state->context->unit_liveness, &candidate,
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
      state->result.failure = failure;
      return iree_ok_status();
    }

    if (!saw_active_conflict) {
      failure.blocking_kind =
          LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_NO_ASSIGNABLE_LOCATION;
      state->result.failure = failure;
      return iree_ok_status();
    }
    if (base > UINT32_MAX - alignment) {
      break;
    }
    base += alignment;
  }

  failure.blocking_kind =
      LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_NO_ASSIGNABLE_LOCATION;
  state->result.failure = failure;
  return iree_ok_status();
}

static const loom_low_allocation_assignment_t*
loom_low_allocation_interval_assignment_current_assignment_for_value_ordinal(
    const loom_low_allocation_interval_assignment_state_t* state,
    loom_value_ordinal_t value_ordinal) {
  return loom_low_allocation_assignment_map_assignment_for_value_ordinal(
      &state->result.assignment_map, value_ordinal, NULL);
}

static iree_status_t
loom_low_allocation_interval_assignment_value_ordinal_for_interval(
    const loom_low_allocation_interval_assignment_state_t* state,
    const loom_liveness_interval_t* interval,
    loom_value_ordinal_t* out_value_ordinal) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_interval_assignment_value_ordinal_for_value(
          state, interval->value_id, &value_ordinal)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation interval value %u is outside the local value domain",
        (unsigned)interval->value_id);
  }
  *out_value_ordinal = value_ordinal;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_interval_assignment_consumption_query(
    loom_low_allocation_interval_assignment_state_t* state,
    const loom_region_t* region, loom_consumption_region_query_t** out_query) {
  if (region == state->context->body) {
    loom_consumption_region_query_t* query = &state->function_consumption_query;
    if (query->region != region) {
      loom_consumption_region_query_initialize_with_cfg_graph(
          state->context->module, region, state->context->function_cfg_graph,
          state->context->arena, query);
    }
    *out_query = query;
    return iree_ok_status();
  }
  loom_consumption_region_query_t* query = &state->nested_consumption_query;
  if (query->region != region) {
    loom_consumption_region_query_initialize(state->context->module, region,
                                             state->context->arena, query);
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
  loom_low_allocation_active_set_remove_assignment_units(
      &state->active, state->result.assignments, state->result.assignment_count,
      assignment_index);
  assignment->location_kind = LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT;
  assignment->location_base = state->next_spill_slot;
  assignment->location_count = assignment->unit_count;
  state->next_spill_slot += assignment->unit_count;
  ++state->result.spill_count;
  IREE_RETURN_IF_ERROR(loom_low_allocation_spill_plan_record(
      state->context->module, state->context->function_cfg_graph, assignment,
      assignment_index, capacity->alloc_unit_bits, capacity->spill_slot_space,
      state->result.spill_plans, &state->result.spill_plan_count));
  loom_low_allocation_spill_remark_record(
      state->result.remarks, &state->result.remark_count, assignment_index,
      capacity->is_bounded ? capacity->max_units : UINT32_MAX,
      assignment->unit_count);
  return iree_ok_status();
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

static iree_status_t loom_low_allocation_interval_assignment_append_assignment(
    loom_low_allocation_interval_assignment_state_t* state,
    const loom_low_allocation_assignment_t* assignment,
    const loom_value_id_t* ignored_storage_lease_value_ids,
    uint16_t ignored_storage_lease_value_count,
    uint32_t* out_assignment_index) {
  if (state->result.assignment_count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation table exceeds uint32_t range");
  }
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_interval_assignment_value_ordinal_for_value(
          state, assignment->value_id, &value_ordinal)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation saw assignment for value %u outside the analyzed "
        "liveness value range",
        (unsigned)assignment->value_id);
  }
  if (state->result.assignment_indices_by_value_ordinal[value_ordinal] !=
      UINT32_MAX) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low allocation saw duplicate assignment for value "
                            "%u",
                            (unsigned)assignment->value_id);
  }
  if (loom_low_allocation_location_kind_is_register_like(
          assignment->location_kind)) {
    bool valid_range = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_target_constraints_validate_register_location_capacity(
            state->context->target_constraints,
            assignment->descriptor_reg_class_id, assignment->location_kind,
            assignment->location_base, assignment->location_count,
            IREE_SV("assignment"), state->context->function_op, &valid_range));
    if (!valid_range) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low allocation produced an assignment outside register-class "
          "capacity");
    }
  }
  const uint32_t assignment_index = (uint32_t)state->result.assignment_count;
  loom_low_allocation_assignment_t stored_assignment = *assignment;
  const loom_liveness_interval_t* semantic_interval =
      loom_liveness_interval_for_value_ordinal(state->context->liveness,
                                               value_ordinal);
  if (stored_assignment.start_point == semantic_interval->start_point) {
    stored_assignment.liveness_segments =
        loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
            state->context->unit_liveness, state->context->liveness,
            value_ordinal);
  } else {
    // Storage reserved outside semantic SSA liveness is not represented by the
    // value's sparse segments. Keep the range empty so conflicts use the
    // conservative linear storage lifetime.
    stored_assignment.liveness_segments = (loom_liveness_segment_range_t){0};
  }
  stored_assignment.unit_point_start =
      loom_low_allocation_interval_assignment_unit_point_start_for_value_ordinal(
          state, value_ordinal);
  stored_assignment.end_point =
      loom_low_allocation_live_range_assignment_max_unit_end_point(
          state->context->unit_liveness->end_points,
          state->context->unit_liveness->point_count, &stored_assignment);
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_storage_lease_state_record_release_actions(
          state->context->storage_leases,
          state->context->target->descriptor_set, state->context->liveness,
          &stored_assignment, ignored_storage_lease_value_ids,
          ignored_storage_lease_value_count));
  state->result.assignments[state->result.assignment_count++] =
      stored_assignment;
  state->result.assignment_map.assignment_count =
      state->result.assignment_count;
  state->result.assignment_indices_by_value_ordinal[value_ordinal] =
      assignment_index;
  loom_low_allocation_target_constraints_record_location_extent(
      state->context->target_constraints,
      stored_assignment.descriptor_reg_class_id,
      stored_assignment.location_kind, stored_assignment.location_base,
      stored_assignment.location_count);
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_storage_lease_state_record_assignment(
          state->context->storage_leases,
          state->context->target->descriptor_set, state->context->liveness,
          &stored_assignment, assignment_index, value_ordinal));
  loom_low_allocation_active_set_insert(
      &state->active, state->context->target->descriptor_set,
      state->result.assignments, state->result.assignment_count,
      assignment_index);
  if (out_assignment_index) {
    *out_assignment_index = assignment_index;
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_interval_assignment_append_assignment_callback(
    void* user_data, const loom_low_allocation_assignment_t* assignment,
    const loom_value_id_t* ignored_storage_lease_value_ids,
    uint16_t ignored_storage_lease_value_count,
    uint32_t* out_assignment_index) {
  return loom_low_allocation_interval_assignment_append_assignment(
      (loom_low_allocation_interval_assignment_state_t*)user_data, assignment,
      ignored_storage_lease_value_ids, ignored_storage_lease_value_count,
      out_assignment_index);
}

static bool loom_low_allocation_interval_assignment_value_id_is_listed(
    const loom_value_id_t* value_ids, uint16_t value_count,
    loom_value_id_t value_id) {
  for (uint16_t i = 0; i < value_count; ++i) {
    if (value_ids[i] == value_id) {
      return true;
    }
  }
  return false;
}

static iree_status_t
loom_low_allocation_interval_assignment_append_unique_value_id(
    loom_value_id_t* value_ids, uint16_t value_capacity, uint16_t* value_count,
    loom_value_id_t value_id) {
  if (loom_low_allocation_interval_assignment_value_id_is_listed(
          value_ids, *value_count, value_id)) {
    return iree_ok_status();
  }
  if (*value_count >= value_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low allocation fixed ignored-value list "
                            "exhausted");
  }
  value_ids[(*value_count)++] = value_id;
  return iree_ok_status();
}

static bool loom_low_allocation_interval_assignment_tied_location_matches(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_placement_relation_t* relation, bool fixed_is_result,
    uint16_t fixed_reg_class_id,
    loom_low_allocation_location_kind_t fixed_location_kind,
    uint32_t fixed_location_base, uint32_t fixed_location_count,
    uint16_t counterpart_reg_class_id,
    loom_low_allocation_location_kind_t counterpart_location_kind,
    uint32_t counterpart_location_base, uint32_t counterpart_location_count) {
  if (fixed_location_kind != counterpart_location_kind ||
      !loom_low_allocation_storage_reg_classes_share(
          descriptor_set, fixed_reg_class_id, counterpart_reg_class_id)) {
    return false;
  }
  if (relation->unit_count != fixed_location_count ||
      relation->unit_count != counterpart_location_count) {
    return false;
  }

  const uint32_t fixed_unit_offset = fixed_is_result
                                         ? relation->result_unit_offset
                                         : relation->source_unit_offset;
  const uint32_t counterpart_unit_offset = fixed_is_result
                                               ? relation->source_unit_offset
                                               : relation->result_unit_offset;
  if (fixed_unit_offset > fixed_location_count ||
      relation->unit_count > fixed_location_count - fixed_unit_offset ||
      counterpart_unit_offset > counterpart_location_count ||
      relation->unit_count >
          counterpart_location_count - counterpart_unit_offset ||
      fixed_location_base > UINT32_MAX - fixed_unit_offset ||
      counterpart_location_base > UINT32_MAX - counterpart_unit_offset) {
    return false;
  }
  return fixed_location_base + fixed_unit_offset ==
         counterpart_location_base + counterpart_unit_offset;
}

static bool loom_low_allocation_interval_assignment_tied_fixed_location_matches(
    const loom_low_allocation_interval_assignment_state_t* state,
    const loom_low_allocation_resolved_fixed_value_t* fixed_value,
    const loom_low_placement_relation_t* relation, bool fixed_is_result,
    loom_value_id_t counterpart_value_id) {
  const loom_low_allocation_resolved_fixed_value_t* counterpart_fixed_value =
      loom_low_allocation_target_constraints_fixed_value_for_value(
          state->context->target_constraints, counterpart_value_id);
  if (!counterpart_fixed_value) {
    return false;
  }
  return loom_low_allocation_interval_assignment_tied_location_matches(
      state->context->target->descriptor_set, relation, fixed_is_result,
      fixed_value->assignment.descriptor_reg_class_id,
      fixed_value->assignment.location_kind,
      fixed_value->assignment.location_base,
      fixed_value->assignment.location_count,
      counterpart_fixed_value->assignment.descriptor_reg_class_id,
      counterpart_fixed_value->assignment.location_kind,
      counterpart_fixed_value->assignment.location_base,
      counterpart_fixed_value->assignment.location_count);
}

static bool
loom_low_allocation_interval_assignment_tied_assignment_location_matches(
    const loom_low_allocation_interval_assignment_state_t* state,
    const loom_low_allocation_resolved_fixed_value_t* fixed_value,
    const loom_low_placement_relation_t* relation, bool fixed_is_result,
    loom_value_ordinal_t counterpart_ordinal) {
  const loom_low_allocation_assignment_t* counterpart_assignment =
      loom_low_allocation_interval_assignment_current_assignment_for_value_ordinal(
          state, counterpart_ordinal);
  if (!counterpart_assignment) {
    return false;
  }
  return loom_low_allocation_interval_assignment_tied_location_matches(
      state->context->target->descriptor_set, relation, fixed_is_result,
      fixed_value->assignment.descriptor_reg_class_id,
      fixed_value->assignment.location_kind,
      fixed_value->assignment.location_base,
      fixed_value->assignment.location_count,
      counterpart_assignment->descriptor_reg_class_id,
      counterpart_assignment->location_kind,
      counterpart_assignment->location_base,
      counterpart_assignment->location_count);
}

static iree_status_t
loom_low_allocation_interval_assignment_collect_fixed_tied_counterpart(
    const loom_low_allocation_interval_assignment_state_t* state,
    const loom_low_allocation_resolved_fixed_value_t* fixed_value,
    const loom_low_placement_relation_t* relation, bool fixed_is_result,
    loom_value_ordinal_t counterpart_ordinal,
    loom_value_id_t* ignored_value_ids, uint16_t ignored_value_capacity,
    uint16_t* ignored_value_count,
    loom_value_id_t* ignored_storage_lease_value_ids,
    uint16_t ignored_storage_lease_value_capacity,
    uint16_t* ignored_storage_lease_value_count) {
  if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT ||
      !iree_all_bits_set(relation->flags,
                         LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD)) {
    return iree_ok_status();
  }

  const loom_value_id_t counterpart_value_id = loom_low_placement_value_id(
      state->context->placement, counterpart_ordinal);
  if (!loom_low_allocation_interval_assignment_tied_fixed_location_matches(
          state, fixed_value, relation, fixed_is_result,
          counterpart_value_id) &&
      !loom_low_allocation_interval_assignment_tied_assignment_location_matches(
          state, fixed_value, relation, fixed_is_result, counterpart_ordinal)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_low_allocation_interval_assignment_append_unique_value_id(
          ignored_value_ids, ignored_value_capacity, ignored_value_count,
          counterpart_value_id));
  return loom_low_allocation_interval_assignment_append_unique_value_id(
      ignored_storage_lease_value_ids, ignored_storage_lease_value_capacity,
      ignored_storage_lease_value_count, counterpart_value_id);
}

static iree_status_t
loom_low_allocation_interval_assignment_collect_fixed_tied_counterparts(
    const loom_low_allocation_interval_assignment_state_t* state,
    const loom_low_allocation_resolved_fixed_value_t* fixed_value,
    loom_value_id_t* ignored_value_ids, uint16_t ignored_value_capacity,
    uint16_t* ignored_value_count,
    loom_value_id_t* ignored_storage_lease_value_ids,
    uint16_t ignored_storage_lease_value_capacity,
    uint16_t* ignored_storage_lease_value_count) {
  const loom_low_placement_relation_range_t result_range =
      loom_low_placement_relation_range_for_value_ordinal(
          state->context->placement, fixed_value->value_ordinal);
  for (uint32_t i = 0; i < result_range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &state->context->placement->relations[result_range.start + i];
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_interval_assignment_collect_fixed_tied_counterpart(
            state, fixed_value, relation, /*fixed_is_result=*/true,
            relation->source_ordinal, ignored_value_ids, ignored_value_capacity,
            ignored_value_count, ignored_storage_lease_value_ids,
            ignored_storage_lease_value_capacity,
            ignored_storage_lease_value_count));
  }

  const loom_low_placement_relation_range_t source_range =
      loom_low_placement_relation_range_for_source_value_ordinal(
          state->context->placement, fixed_value->value_ordinal);
  for (uint32_t i = 0; i < source_range.count; ++i) {
    const uint32_t relation_index =
        state->context->placement
            ->relation_indices_by_source_ordinal[source_range.start + i];
    const loom_low_placement_relation_t* relation =
        &state->context->placement->relations[relation_index];
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_interval_assignment_collect_fixed_tied_counterpart(
            state, fixed_value, relation, /*fixed_is_result=*/false,
            relation->result_ordinal, ignored_value_ids, ignored_value_capacity,
            ignored_value_count, ignored_storage_lease_value_ids,
            ignored_storage_lease_value_capacity,
            ignored_storage_lease_value_count));
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_interval_assignment_assign_fixed_interval(
    loom_low_allocation_interval_assignment_state_t* state,
    const loom_liveness_interval_t* interval, bool* out_assigned) {
  *out_assigned = false;
  const loom_low_allocation_resolved_fixed_value_t* fixed_value =
      loom_low_allocation_target_constraints_fixed_value_for_value(
          state->context->target_constraints, interval->value_id);
  if (!fixed_value) {
    return iree_ok_status();
  }
  const loom_low_placement_relation_range_t result_range =
      loom_low_placement_relation_range_for_value_ordinal(
          state->context->placement, fixed_value->value_ordinal);
  const loom_low_placement_relation_range_t source_range =
      loom_low_placement_relation_range_for_source_value_ordinal(
          state->context->placement, fixed_value->value_ordinal);
  const uint32_t ignored_value_capacity_u32 =
      1u + result_range.count + source_range.count;
  if (ignored_value_capacity_u32 > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low allocation fixed tied counterpart count "
                            "exceeds uint16_t");
  }
  const uint16_t ignored_value_capacity = (uint16_t)ignored_value_capacity_u32;
  loom_value_id_t inline_ignored_value_ids[8];
  loom_value_id_t* ignored_value_ids = inline_ignored_value_ids;
  loom_value_id_t inline_ignored_storage_lease_value_ids[8];
  loom_value_id_t* ignored_storage_lease_value_ids =
      inline_ignored_storage_lease_value_ids;
  if (ignored_value_capacity > IREE_ARRAYSIZE(inline_ignored_value_ids)) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->context->arena, ignored_value_capacity,
        sizeof(*ignored_value_ids), (void**)&ignored_value_ids));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->context->arena, ignored_value_capacity,
                                  sizeof(*ignored_storage_lease_value_ids),
                                  (void**)&ignored_storage_lease_value_ids));
  }
  ignored_value_ids[0] = interval->value_id;
  uint16_t ignored_value_count = 1;
  uint16_t ignored_storage_lease_value_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_interval_assignment_collect_fixed_tied_counterparts(
          state, fixed_value, ignored_value_ids, ignored_value_capacity,
          &ignored_value_count, ignored_storage_lease_value_ids,
          ignored_value_capacity, &ignored_storage_lease_value_count));

  loom_low_allocation_search_context_t search_context =
      loom_low_allocation_interval_assignment_search_context(state);
  if (loom_low_allocation_search_location_conflicts(
          &search_context, interval,
          fixed_value->assignment.descriptor_reg_class_id,
          fixed_value->assignment.location_kind,
          fixed_value->assignment.location_base,
          fixed_value->assignment.location_count, ignored_value_ids,
          ignored_value_count, ignored_storage_lease_value_ids,
          ignored_storage_lease_value_count,
          LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low fixed value %u cannot occupy register class %u location %u:%u "
        "without overlapping another live interval or reserved range",
        (unsigned)interval->value_id,
        (unsigned)fixed_value->assignment.descriptor_reg_class_id,
        (unsigned)fixed_value->assignment.location_base,
        (unsigned)fixed_value->assignment.location_count);
  }

  IREE_RETURN_IF_ERROR(
      loom_low_allocation_interval_assignment_append_assignment(
          state, &fixed_value->assignment, ignored_storage_lease_value_ids,
          ignored_storage_lease_value_count, NULL));
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
    memset(state->result.assignments, 0,
           order->interval_count * sizeof(*state->result.assignments));
    IREE_RETURN_IF_ERROR(loom_low_allocation_active_set_initialize(
        state->context->liveness, order->interval_count, order->unit_count,
        state->context->arena, &state->active));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->context->arena, order->interval_count,
                                  sizeof(*state->result.spill_plans),
                                  (void**)&state->result.spill_plans));
    memset(state->result.spill_plans, 0,
           order->interval_count * sizeof(*state->result.spill_plans));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->context->arena, order->interval_count,
        sizeof(*state->result.remarks), (void**)&state->result.remarks));
    memset(state->result.remarks, 0,
           order->interval_count * sizeof(*state->result.remarks));
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
      state->context->arena, value_count,
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

iree_status_t loom_low_allocation_interval_assignment_build(
    const loom_low_allocation_interval_assignment_context_t* context,
    loom_low_allocation_interval_assignment_result_t* out_result) {
  *out_result = (loom_low_allocation_interval_assignment_result_t){0};
  loom_low_allocation_interval_assignment_state_t state = {
      .context = context,
  };

  loom_low_allocation_interval_order_t order = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_interval_order_build(
      context->liveness, context->arena, &order));
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_interval_assignment_initialize_result_storage(
          &state, &order));
  if (order.interval_count == 0) {
    *out_result = state.result;
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < order.interval_count; ++i) {
    const loom_liveness_interval_t* interval = order.intervals[i];
    loom_low_allocation_active_set_expire(
        &state.active, state.result.assignments, state.result.assignment_count,
        interval->start_point);

    loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_interval_assignment_value_ordinal_for_interval(
            &state, interval, &value_ordinal));
    if (loom_low_allocation_interval_assignment_current_assignment_for_value_ordinal(
            &state, value_ordinal)) {
      continue;
    }

    bool assigned_fixed_interval = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_interval_assignment_assign_fixed_interval(
            &state, interval, &assigned_fixed_interval));
    if (assigned_fixed_interval) {
      continue;
    }

    loom_low_allocation_search_context_t search_context =
        loom_low_allocation_interval_assignment_search_context(&state);
    loom_low_allocation_coalescing_context_t coalescing_context = {
        .arena = context->arena,
        .liveness = context->liveness,
        .schedule = context->schedule,
        .placement = context->placement,
        .target_constraints = context->target_constraints,
        .assignment_map = &state.result.assignment_map,
        .search_context = &search_context,
        .append_assignment =
            loom_low_allocation_interval_assignment_append_assignment_callback,
        .consumption_query =
            loom_low_allocation_interval_assignment_consumption_query_callback,
        .user_data = &state,
    };

    bool assigned_tied_interval = false;
    IREE_RETURN_IF_ERROR(loom_low_allocation_coalescing_assign_tied_interval(
        &coalescing_context, interval, &assigned_tied_interval));
    if (assigned_tied_interval) {
      continue;
    }

    bool assigned_concat_source_interval = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_assign_concat_source_interval(
            &coalescing_context, interval, &assigned_concat_source_interval));
    if (assigned_concat_source_interval) {
      continue;
    }

    bool assigned_structural_interval = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_assign_structural_interval(
            &coalescing_context, interval, &assigned_structural_interval));
    if (assigned_structural_interval) {
      continue;
    }

    bool assigned_edge_source_interval = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_assign_edge_source_interval(
            &coalescing_context, interval, &assigned_edge_source_interval));
    if (assigned_edge_source_interval) {
      continue;
    }

    loom_low_allocation_class_capacity_t capacity = {0};
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_target_constraints_interval_capacity(
            context->target_constraints, interval, &capacity));
    if (interval->unit_count > UINT32_MAX - state.next_spill_slot) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "allocation spill slots exceed uint32_t range");
    }

    uint32_t location_base = 0;
    bool assigned = loom_low_allocation_search_find_free_location(
        &search_context, interval, capacity, &location_base);
    const bool requires_register =
        loom_low_allocation_storage_lease_state_value_has_records(
            context->storage_leases, context->liveness, interval->value_id) ||
        (interval->value_id < context->required_register_values.bit_count &&
         iree_bitmap_test(context->required_register_values,
                          interval->value_id)) ||
        loom_low_allocation_spill_traffic_interval_requires_register_location(
            context->module, interval);
    if (!assigned && (capacity.is_spillable || requires_register)) {
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_interval_assignment_initialize_spill_traffic_cache(
              &state));
      search_context =
          loom_low_allocation_interval_assignment_search_context(&state);
      loom_low_allocation_search_spill_victim_set_t victim_set = {0};
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_search_find_active_spill_victim_set(
              &search_context, interval, &capacity, requires_register,
              context->arena, &victim_set));
      if (victim_set.found) {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_interval_assignment_spill_active_assignment_set(
                &state, victim_set.assignment_indices,
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
              &state, interval, value_ordinal, &capacity, budget_units,
              requires_register, failure_code));
      IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_emit_failure(
          context->target_constraints,
          loom_low_diagnostic_value_origin_op(
              context->module, interval->value_id, context->function_op),
          interval->value_class, budget_units,
          state.result.failure.peak_live_units, failure_code));
      *out_result = state.result;
      return iree_ok_status();
    }

    const loom_low_allocation_assignment_t assignment = {
        .value_id = interval->value_id,
        .value_class = interval->value_class,
        .descriptor_reg_class_id = capacity.descriptor_reg_class_id,
        .start_point = interval->start_point,
        .end_point =
            loom_low_allocation_live_range_interval_storage_end_point(interval),
        .unit_count = interval->unit_count,
        .location_kind = assigned ? capacity.location_kind
                                  : LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT,
        .location_base = assigned ? location_base : state.next_spill_slot,
        .location_count = interval->unit_count,
    };

    uint32_t assignment_index = 0;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_interval_assignment_append_assignment(
            &state, &assignment, /*ignored_storage_lease_value_ids=*/NULL,
            /*ignored_storage_lease_value_count=*/0, &assignment_index));
    if (!assigned) {
      state.next_spill_slot += interval->unit_count;
      ++state.result.spill_count;
      IREE_RETURN_IF_ERROR(loom_low_allocation_spill_plan_record(
          context->module, context->function_cfg_graph,
          &state.result.assignments[assignment_index], assignment_index,
          capacity.alloc_unit_bits, capacity.spill_slot_space,
          state.result.spill_plans, &state.result.spill_plan_count));
      loom_low_allocation_spill_remark_record(
          state.result.remarks, &state.result.remark_count, assignment_index,
          capacity.is_bounded ? capacity.max_units : UINT32_MAX,
          interval->unit_count);
    }
  }

  *out_result = state.result;
  return iree_ok_status();
}
