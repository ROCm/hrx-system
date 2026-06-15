// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/search.h"

#include <string.h>

#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/spill_traffic.h"

static uint32_t loom_low_allocation_search_unit_end_point_start_for_value(
    const loom_low_allocation_search_context_t* context,
    loom_value_id_t value_id) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_assignment_map_value_ordinal_for_value(
          context->assignment_map, value_id, &value_ordinal)) {
    return UINT32_MAX;
  }
  return loom_low_allocation_unit_liveness_end_point_start_for_value_ordinal(
      context->unit_liveness, context->liveness, value_ordinal);
}

static bool loom_low_allocation_search_align_up_u32(uint32_t value,
                                                    uint32_t alignment,
                                                    uint32_t* out_value) {
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

bool loom_low_allocation_search_location_conflicts(
    loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* interval, uint16_t reg_class_id,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count, const loom_value_id_t* ignored_value_ids,
    uint16_t ignored_value_count,
    const loom_value_id_t* ignored_storage_lease_value_ids,
    uint16_t ignored_storage_lease_value_count,
    loom_low_allocation_storage_release_policy_t release_policy) {
  loom_low_allocation_assignment_t candidate = {
      .value_id = interval->value_id,
      .value_class = interval->value_class,
      .descriptor_reg_class_id = reg_class_id,
      .start_point = interval->start_point,
      .end_point =
          loom_low_allocation_live_range_interval_storage_end_point(interval),
      .unit_count = interval->unit_count,
      .location_kind = location_kind,
      .location_base = location_base,
      .location_count = location_count,
      .unit_end_point_start =
          loom_low_allocation_search_unit_end_point_start_for_value(
              context, interval->value_id),
  };
  if (loom_low_allocation_active_set_conflicts(
          context->active_set, context->descriptor_set, context->liveness,
          context->unit_liveness->end_points,
          context->unit_liveness->end_point_count,
          context->assignment_map->assignments,
          context->assignment_map->assignment_count, &candidate,
          ignored_value_ids, ignored_value_count)) {
    return true;
  }
  if (loom_low_allocation_target_constraints_fixed_value_conflicts(
          context->target_constraints, context->liveness,
          context->unit_liveness, &candidate, ignored_value_ids,
          ignored_value_count)) {
    return true;
  }
  if (loom_low_allocation_target_constraints_reserved_range_conflicts(
          context->target_constraints, reg_class_id, location_kind,
          location_base, location_count)) {
    return true;
  }
  if (loom_low_allocation_storage_lease_state_conflicts(
          context->storage_leases, context->descriptor_set, context->liveness,
          &candidate, ignored_storage_lease_value_ids,
          ignored_storage_lease_value_count, release_policy)) {
    return true;
  }
  return false;
}

bool loom_low_allocation_search_find_free_location(
    loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* interval,
    loom_low_allocation_class_capacity_t capacity, uint32_t* out_base) {
  if (capacity.is_bounded && interval->unit_count > capacity.max_units) {
    return false;
  }

  const uint32_t alignment =
      loom_low_allocation_live_range_interval_alignment(interval);
  uint32_t last_base = 0;
  if (capacity.is_bounded) {
    last_base = capacity.max_units - interval->unit_count;
  } else {
    const uint32_t search_limit =
        loom_low_allocation_target_constraints_assigned_location_search_limit(
            context->target_constraints, capacity.descriptor_reg_class_id,
            capacity.location_kind);
    if (!loom_low_allocation_search_align_up_u32(search_limit, alignment,
                                                 &last_base)) {
      return false;
    }
  }

  for (uint32_t base = 0; base <= last_base;) {
    if (!loom_low_allocation_search_location_conflicts(
            context, interval, capacity.descriptor_reg_class_id,
            capacity.location_kind, base, interval->unit_count,
            /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0,
            /*ignored_storage_lease_value_ids=*/NULL,
            /*ignored_storage_lease_value_count=*/0,
            LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN)) {
      *out_base = base;
      return true;
    }
    if (base > UINT32_MAX - alignment) {
      break;
    }
    base += alignment;
  }
  for (uint32_t base = 0; base <= last_base;) {
    if (!loom_low_allocation_search_location_conflicts(
            context, interval, capacity.descriptor_reg_class_id,
            capacity.location_kind, base, interval->unit_count,
            /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0,
            /*ignored_storage_lease_value_ids=*/NULL,
            /*ignored_storage_lease_value_count=*/0,
            LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED)) {
      *out_base = base;
      return true;
    }
    if (base > UINT32_MAX - alignment) {
      break;
    }
    base += alignment;
  }
  return false;
}

iree_status_t loom_low_allocation_search_assignment_spill_capacity(
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
  if (loom_low_allocation_spill_traffic_value_requires_register_location(
          context->module, assignment->value_id)) {
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

static bool loom_low_allocation_search_spill_victim_set_is_better(
    uint16_t candidate_count, uint32_t candidate_unit_count,
    uint32_t candidate_latest_end_point, uint32_t candidate_location_base,
    uint16_t best_count, uint32_t best_unit_count,
    uint32_t best_latest_end_point, uint32_t best_location_base) {
  if (best_count == 0) {
    return true;
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
  return candidate_location_base < best_location_base;
}

static iree_status_t loom_low_allocation_search_collect_active_spill_victim_set(
    loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_allocation_class_capacity_t* capacity,
    uint32_t location_base, bool interval_requires_register,
    uint32_t* assignment_indices, loom_value_id_t* ignored_value_ids,
    uint16_t* out_assignment_count, uint32_t* out_unit_count,
    uint32_t* out_latest_end_point, bool* out_blocked) {
  *out_assignment_count = 0;
  *out_unit_count = 0;
  *out_latest_end_point = 0;
  *out_blocked = false;
  const uint32_t interval_end =
      loom_low_allocation_live_range_interval_storage_end_point(interval);

  const loom_low_allocation_assignment_t candidate = {
      .value_id = interval->value_id,
      .value_class = interval->value_class,
      .descriptor_reg_class_id = capacity->descriptor_reg_class_id,
      .start_point = interval->start_point,
      .end_point = interval_end,
      .unit_count = interval->unit_count,
      .location_kind = capacity->location_kind,
      .location_base = location_base,
      .location_count = interval->unit_count,
      .unit_end_point_start =
          loom_low_allocation_search_unit_end_point_start_for_value(
              context, interval->value_id),
  };

  uint16_t assignment_count = 0;
  uint32_t unit_count = 0;
  uint32_t latest_end_point = 0;
  for (iree_host_size_t i = 0; i < context->active_set->count; ++i) {
    const uint32_t assignment_index =
        context->active_set->assignment_indices[context->active_set->start + i];
    IREE_ASSERT_LT(assignment_index, context->assignment_map->assignment_count);
    const loom_low_allocation_assignment_t* assignment =
        &context->assignment_map->assignments[assignment_index];
    if (!loom_low_allocation_active_assignment_conflicts(
            context->descriptor_set, context->liveness,
            context->unit_liveness->end_points,
            context->unit_liveness->end_point_count, assignment, &candidate,
            /*ignored_value_ids=*/NULL,
            /*ignored_value_count=*/0)) {
      continue;
    }
    bool can_spill = false;
    IREE_RETURN_IF_ERROR(loom_low_allocation_search_assignment_spill_capacity(
        context, assignment, &can_spill, NULL));
    if (!can_spill) {
      *out_blocked = true;
      return iree_ok_status();
    }
    if (!interval_requires_register && assignment->end_point <= interval_end) {
      *out_blocked = true;
      return iree_ok_status();
    }
    if (assignment_count == UINT16_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "active spill victim set exceeds uint16_t");
    }
    if (unit_count > UINT32_MAX - assignment->unit_count) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "active spill victim unit count overflow");
    }
    assignment_indices[assignment_count] = assignment_index;
    ignored_value_ids[assignment_count] = assignment->value_id;
    ++assignment_count;
    unit_count += assignment->unit_count;
    if (latest_end_point < assignment->end_point) {
      latest_end_point = assignment->end_point;
    }
  }

  if (assignment_count == 0 ||
      loom_low_allocation_search_location_conflicts(
          context, interval, capacity->descriptor_reg_class_id,
          capacity->location_kind, location_base, interval->unit_count,
          ignored_value_ids, assignment_count,
          /*ignored_storage_lease_value_ids=*/NULL,
          /*ignored_storage_lease_value_count=*/0,
          LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN)) {
    *out_blocked = true;
    return iree_ok_status();
  }

  *out_assignment_count = assignment_count;
  *out_unit_count = unit_count;
  *out_latest_end_point = latest_end_point;
  return iree_ok_status();
}

iree_status_t loom_low_allocation_search_find_active_spill_victim_set(
    loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_allocation_class_capacity_t* capacity,
    bool interval_requires_register, iree_arena_allocator_t* arena,
    loom_low_allocation_search_spill_victim_set_t* out_victim_set) {
  *out_victim_set = (loom_low_allocation_search_spill_victim_set_t){0};
  if (capacity->is_bounded && interval->unit_count > capacity->max_units) {
    return iree_ok_status();
  }
  if (context->active_set->count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "active allocation set exceeds uint16_t");
  }

  uint32_t last_base = 0;
  const uint32_t alignment =
      loom_low_allocation_live_range_interval_alignment(interval);
  if (capacity->is_bounded) {
    last_base = capacity->max_units - interval->unit_count;
  } else {
    const uint32_t search_limit =
        loom_low_allocation_target_constraints_assigned_location_search_limit(
            context->target_constraints, capacity->descriptor_reg_class_id,
            capacity->location_kind);
    if (!loom_low_allocation_search_align_up_u32(search_limit, alignment,
                                                 &last_base)) {
      return iree_ok_status();
    }
  }

  uint32_t* candidate_assignment_indices = NULL;
  uint32_t* best_assignment_indices = NULL;
  loom_value_id_t* ignored_value_ids = NULL;
  if (context->active_set->count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, context->active_set->count,
                                  sizeof(*candidate_assignment_indices),
                                  (void**)&candidate_assignment_indices));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, context->active_set->count, sizeof(*best_assignment_indices),
        (void**)&best_assignment_indices));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, context->active_set->count, sizeof(*ignored_value_ids),
        (void**)&ignored_value_ids));
  }

  uint16_t best_assignment_count = 0;
  uint32_t best_unit_count = 0;
  uint32_t best_latest_end_point = 0;
  uint32_t best_location_base = 0;
  for (uint32_t base = 0; base <= last_base;) {
    uint16_t candidate_assignment_count = 0;
    uint32_t candidate_unit_count = 0;
    uint32_t candidate_latest_end_point = 0;
    bool blocked = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_search_collect_active_spill_victim_set(
            context, interval, capacity, base, interval_requires_register,
            candidate_assignment_indices, ignored_value_ids,
            &candidate_assignment_count, &candidate_unit_count,
            &candidate_latest_end_point, &blocked));
    if (!blocked &&
        loom_low_allocation_search_spill_victim_set_is_better(
            candidate_assignment_count, candidate_unit_count,
            candidate_latest_end_point, base, best_assignment_count,
            best_unit_count, best_latest_end_point, best_location_base)) {
      best_assignment_count = candidate_assignment_count;
      best_unit_count = candidate_unit_count;
      best_latest_end_point = candidate_latest_end_point;
      best_location_base = base;
      memcpy(best_assignment_indices, candidate_assignment_indices,
             (iree_host_size_t)candidate_assignment_count *
                 sizeof(*best_assignment_indices));
    }
    if (base > UINT32_MAX - alignment) {
      break;
    }
    base += alignment;
  }

  if (best_assignment_count == 0) {
    return iree_ok_status();
  }
  *out_victim_set = (loom_low_allocation_search_spill_victim_set_t){
      .location_base = best_location_base,
      .assignment_indices = best_assignment_indices,
      .assignment_count = best_assignment_count,
      .found = true,
  };
  return iree_ok_status();
}
