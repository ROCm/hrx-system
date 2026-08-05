// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/search.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/spill_traffic.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/target/residency.h"

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

static bool loom_low_allocation_search_has_storage_release_records(
    const loom_low_allocation_search_context_t* context) {
  return context->storage_leases && context->storage_leases->lease_table &&
         context->storage_leases->lease_table->record_count != 0;
}

static bool loom_low_allocation_search_has_pressure_release_records(
    const loom_low_allocation_search_context_t* context) {
  return context->storage_leases &&
         context->storage_leases->pressure_release_record_count != 0;
}

static loom_low_allocation_assignment_t
loom_low_allocation_search_candidate_assignment(
    const loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* interval, uint16_t reg_class_id,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  const bool has_value_ordinal =
      loom_low_allocation_assignment_map_value_ordinal_for_value(
          context->assignment_map, interval->value_id, &value_ordinal);
  const loom_liveness_segment_range_t segment_range =
      has_value_ordinal
          ? loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
                context->unit_liveness, context->liveness, value_ordinal)
          : (loom_liveness_segment_range_t){0};
  loom_low_allocation_assignment_t candidate = {
      .value_id = interval->value_id,
      .value_class = interval->value_class,
      .descriptor_reg_class_id = reg_class_id,
      .start_point = interval->start_point,
      .end_point =
          loom_low_allocation_live_range_interval_storage_end_point(interval),
      .liveness_segments = segment_range,
      .unit_count = interval->unit_count,
      .location_kind = location_kind,
      .location_base = location_base,
      .location_count = location_count,
      .unit_point_start =
          has_value_ordinal
              ? loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
                    context->unit_liveness, context->liveness, value_ordinal)
              : UINT32_MAX,
  };
  candidate.end_point =
      loom_low_allocation_live_range_assignment_max_unit_end_point(
          context->unit_liveness->end_points,
          context->unit_liveness->point_count, &candidate);
  return candidate;
}

static bool loom_low_allocation_search_relation_is_location_preference(
    const loom_low_placement_relation_t* relation) {
  return relation->kind ==
             LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION ||
         relation->kind == LOOM_LOW_PLACEMENT_RELATION_DISJOINT_STORAGE;
}

static uint32_t loom_low_allocation_search_relation_penalty(
    const loom_low_allocation_search_context_t* context,
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* candidate,
    bool candidate_is_result) {
  if (!loom_low_allocation_search_relation_is_location_preference(relation)) {
    return 0;
  }
  const loom_value_ordinal_t counterpart_ordinal =
      candidate_is_result ? relation->source_ordinal : relation->result_ordinal;
  const loom_low_allocation_assignment_t* counterpart =
      loom_low_allocation_assignment_map_assignment_for_value_ordinal(
          context->assignment_map, counterpart_ordinal, NULL);
  if (counterpart == NULL) {
    return 0;
  }
  const loom_low_allocation_assignment_t* result_assignment =
      candidate_is_result ? candidate : counterpart;
  const loom_low_allocation_assignment_t* source_assignment =
      candidate_is_result ? counterpart : candidate;
  if (loom_low_allocation_storage_placement_relation_satisfied(
          context->descriptor_set, relation, result_assignment,
          source_assignment)) {
    return 0;
  }
  return iree_max((uint32_t)1, (uint32_t)relation->priority);
}

typedef struct loom_low_allocation_search_location_preference_t {
  // Placement table owning the relation ranges, or NULL when inert.
  const loom_low_placement_table_t* placement;
  // Relations where the candidate is the result assignment.
  loom_low_placement_relation_range_t result_range;
  // Relations where the candidate is the source assignment.
  loom_low_placement_relation_range_t source_range;
} loom_low_allocation_search_location_preference_t;

static bool loom_low_allocation_search_relation_is_actionable(
    const loom_low_allocation_search_context_t* context,
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* candidate,
    bool candidate_is_result) {
  if (!loom_low_allocation_search_relation_is_location_preference(relation)) {
    return false;
  }
  const loom_value_ordinal_t counterpart_ordinal =
      candidate_is_result ? relation->source_ordinal : relation->result_ordinal;
  const loom_low_allocation_assignment_t* counterpart =
      loom_low_allocation_assignment_map_assignment_for_value_ordinal(
          context->assignment_map, counterpart_ordinal, NULL);
  if (counterpart == NULL) {
    return false;
  }
  return loom_low_allocation_storage_assignment_classes_share(
      context->descriptor_set, candidate, counterpart);
}

static loom_low_allocation_search_location_preference_t
loom_low_allocation_search_location_preference(
    const loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* candidate) {
  loom_low_allocation_search_location_preference_t preference = {0};
  const loom_low_placement_table_t* placement = context->placement;
  if (placement == NULL || placement->location_relation_count == 0) {
    return preference;
  }
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_assignment_map_value_ordinal_for_value(
          context->assignment_map, candidate->value_id, &value_ordinal)) {
    return preference;
  }

  preference.result_range = loom_low_placement_relation_range_for_value_ordinal(
      placement, value_ordinal);
  for (uint32_t i = 0; i < preference.result_range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &placement->relations[preference.result_range.start + i];
    if (loom_low_allocation_search_relation_is_actionable(context, relation,
                                                          candidate, true)) {
      preference.placement = placement;
      break;
    }
  }
  preference.source_range =
      loom_low_placement_relation_range_for_source_value_ordinal(placement,
                                                                 value_ordinal);
  if (preference.placement == NULL) {
    for (uint32_t i = 0; i < preference.source_range.count; ++i) {
      const uint32_t relation_index =
          placement->relation_indices_by_source_ordinal
              [preference.source_range.start + i];
      const loom_low_placement_relation_t* relation =
          &placement->relations[relation_index];
      if (loom_low_allocation_search_relation_is_actionable(context, relation,
                                                            candidate, false)) {
        preference.placement = placement;
        break;
      }
    }
  }
  return preference;
}

static uint32_t loom_low_allocation_search_location_preference_penalty(
    const loom_low_allocation_search_context_t* context,
    const loom_low_allocation_search_location_preference_t* preference,
    const loom_low_allocation_assignment_t* candidate) {
  const loom_low_placement_table_t* placement = preference->placement;
  if (placement == NULL) {
    return 0;
  }

  uint32_t penalty = 0;
  for (uint32_t i = 0; i < preference->result_range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &placement->relations[preference->result_range.start + i];
    penalty = iree_math_saturating_add_u32(
        penalty, loom_low_allocation_search_relation_penalty(context, relation,
                                                             candidate, true));
  }
  for (uint32_t i = 0; i < preference->source_range.count; ++i) {
    const uint32_t relation_index =
        placement->relation_indices_by_source_ordinal
            [preference->source_range.start + i];
    const loom_low_placement_relation_t* relation =
        &placement->relations[relation_index];
    penalty = iree_math_saturating_add_u32(
        penalty, loom_low_allocation_search_relation_penalty(context, relation,
                                                             candidate, false));
  }
  return penalty;
}

bool loom_low_allocation_search_assignment_conflicts(
    loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count,
    const loom_value_id_t* ignored_storage_lease_value_ids,
    uint16_t ignored_storage_lease_value_count,
    loom_low_allocation_storage_release_policy_t release_policy) {
  if (loom_low_allocation_active_set_conflicts(
          context->active_set, context->descriptor_set, context->unit_liveness,
          context->assignment_map->assignments,
          context->assignment_map->assignment_count, candidate,
          ignored_value_ids, ignored_value_count)) {
    return true;
  }
  if (loom_low_allocation_target_constraints_fixed_value_conflicts(
          context->target_constraints, context->liveness,
          context->unit_liveness, candidate, ignored_value_ids,
          ignored_value_count)) {
    return true;
  }
  if (loom_low_allocation_target_constraints_reserved_range_conflicts(
          context->target_constraints, candidate->descriptor_reg_class_id,
          candidate->location_kind, candidate->location_base,
          candidate->location_count)) {
    return true;
  }
  if (loom_low_allocation_storage_lease_state_conflicts(
          context->storage_leases, context->descriptor_set, context->liveness,
          candidate, ignored_storage_lease_value_ids,
          ignored_storage_lease_value_count, release_policy)) {
    return true;
  }
  return false;
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
  const loom_low_allocation_assignment_t candidate =
      loom_low_allocation_search_candidate_assignment(
          context, interval, reg_class_id, location_kind, location_base,
          location_count);
  return loom_low_allocation_search_assignment_conflicts(
      context, &candidate, ignored_value_ids, ignored_value_count,
      ignored_storage_lease_value_ids, ignored_storage_lease_value_count,
      release_policy);
}

typedef struct loom_low_allocation_search_location_choice_t {
  // Base location of the best legal candidate.
  uint32_t base;
  // Base location of the first legal candidate under the release policy.
  uint32_t first_base;
  // Soft placement penalty for base.
  uint32_t preference_penalty;
  // True when base and preference_penalty are populated.
  bool found;
} loom_low_allocation_search_location_choice_t;

static void loom_low_allocation_search_find_location_for_release_policy(
    loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* candidate_template,
    const loom_low_allocation_search_location_preference_t* preference,
    uint32_t last_base, uint32_t alignment,
    loom_low_allocation_storage_release_policy_t release_policy,
    loom_low_allocation_search_location_choice_t* out_choice) {
  *out_choice = (loom_low_allocation_search_location_choice_t){0};
  for (uint32_t base = 0; base <= last_base;) {
    loom_low_allocation_assignment_t candidate = *candidate_template;
    candidate.location_base = base;
    if (!loom_low_allocation_search_assignment_conflicts(
            context, &candidate,
            /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0,
            /*ignored_storage_lease_value_ids=*/NULL,
            /*ignored_storage_lease_value_count=*/0, release_policy)) {
      const uint32_t preference_penalty =
          loom_low_allocation_search_location_preference_penalty(
              context, preference, &candidate);
      const uint32_t first_base =
          out_choice->found ? out_choice->first_base : base;
      if (!out_choice->found ||
          preference_penalty < out_choice->preference_penalty) {
        *out_choice = (loom_low_allocation_search_location_choice_t){
            .base = base,
            .first_base = first_base,
            .preference_penalty = preference_penalty,
            .found = true,
        };
        if (preference_penalty == 0) {
          return;
        }
      }
    }
    if (base > UINT32_MAX - alignment) {
      break;
    }
    base += alignment;
  }
}

static uint32_t loom_low_allocation_search_location_residency_tier(
    const loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* candidate_template,
    const loom_low_allocation_search_location_choice_t* choice) {
  const loom_target_residency_model_t* model = context->residency_model;
  const uint16_t reg_class_id = candidate_template->descriptor_reg_class_id;
  IREE_ASSERT_EQ(model->direct_resources.resource_count,
                 context->descriptor_set->reg_class_count);
  IREE_ASSERT_LT(reg_class_id, model->direct_resources.resource_count);
  const uint32_t current_units =
      context->target_constraints
          ->max_assigned_location_end_by_reg_class[reg_class_id];
  const uint32_t choice_units = iree_max(
      current_units, iree_math_saturating_add_u32(
                         choice->base, candidate_template->location_count));
  return loom_target_residency_evaluate_tier_with_direct_resource_override(
      model,
      context->target_constraints->max_assigned_location_end_by_reg_class,
      reg_class_id, choice_units);
}

static bool loom_low_allocation_search_location_crosses_residency_cliff(
    const loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* candidate_template,
    const loom_low_allocation_search_location_choice_t* choice,
    uint32_t* out_choice_tier) {
  const uint16_t reg_class_id = candidate_template->descriptor_reg_class_id;
  const uint32_t* current_units_by_reg_class =
      context->target_constraints->max_assigned_location_end_by_reg_class;
  const uint32_t current_tier =
      loom_target_residency_evaluate_tier_with_direct_resource_override(
          context->residency_model, current_units_by_reg_class, reg_class_id,
          current_units_by_reg_class[reg_class_id]);
  *out_choice_tier = loom_low_allocation_search_location_residency_tier(
      context, candidate_template, choice);
  return *out_choice_tier < current_tier;
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

  const loom_low_allocation_assignment_t candidate_template =
      loom_low_allocation_search_candidate_assignment(
          context, interval, capacity.descriptor_reg_class_id,
          capacity.location_kind, /*location_base=*/0, interval->unit_count);
  const loom_low_allocation_search_location_preference_t preference =
      loom_low_allocation_search_location_preference(context,
                                                     &candidate_template);
  loom_low_allocation_search_location_choice_t release_free = {0};
  loom_low_allocation_search_find_location_for_release_policy(
      context, &candidate_template, &preference, last_base, alignment,
      LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN, &release_free);
  loom_low_allocation_search_location_choice_t pressure_release = {0};
  if (loom_low_allocation_search_has_pressure_release_records(context)) {
    loom_low_allocation_search_find_location_for_release_policy(
        context, &candidate_template, &preference, last_base, alignment,
        LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FOR_PRESSURE, &pressure_release);
  }
  if (pressure_release.found &&
      (!release_free.found ||
       pressure_release.first_base < release_free.first_base)) {
    *out_base = pressure_release.base;
    return true;
  }
  const bool has_storage_release_records =
      loom_low_allocation_search_has_storage_release_records(context);
  loom_low_allocation_search_location_choice_t release_allowed = {0};
  bool searched_release_allowed = false;
  if (release_free.found && has_storage_release_records &&
      !loom_target_residency_model_is_empty(context->residency_model)) {
    uint32_t release_free_tier = 0;
    if (loom_low_allocation_search_location_crosses_residency_cliff(
            context, &candidate_template, &release_free, &release_free_tier)) {
      loom_low_allocation_search_find_location_for_release_policy(
          context, &candidate_template, &preference, last_base, alignment,
          LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED, &release_allowed);
      searched_release_allowed = true;
      if (release_allowed.found && release_allowed.base < release_free.base &&
          loom_low_allocation_search_location_residency_tier(
              context, &candidate_template, &release_allowed) >
              release_free_tier) {
        *out_base = release_allowed.base;
        return true;
      }
    }
  }
  if (release_free.found) {
    *out_base = release_free.base;
    return true;
  }
  if (!has_storage_release_records) {
    return false;
  }
  if (!searched_release_allowed) {
    loom_low_allocation_search_find_location_for_release_policy(
        context, &candidate_template, &preference, last_base, alignment,
        LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED, &release_allowed);
  }
  if (release_allowed.found) {
    *out_base = release_allowed.base;
    return true;
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
  if (loom_low_allocation_storage_lease_state_value_has_records(
          context->storage_leases, context->liveness, assignment->value_id)) {
    return iree_ok_status();
  }
  if ((assignment->value_id < context->required_register_values.bit_count &&
       iree_bitmap_test(context->required_register_values,
                        assignment->value_id)) ||
      loom_low_allocation_spill_traffic_value_requires_register_location(
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

static iree_status_t loom_low_allocation_search_assignment_spill_traffic_cost(
    loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* assignment,
    const loom_low_allocation_class_capacity_t* capacity, uint64_t* out_cost) {
  *out_cost = 0;
  loom_low_allocation_spill_plan_traffic_t traffic = {0};
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  const bool has_cached_traffic =
      context->spill_traffic_by_value_ordinal &&
      loom_low_allocation_assignment_map_value_ordinal_for_value(
          context->assignment_map, assignment->value_id, &value_ordinal);
  if (has_cached_traffic) {
    traffic = context->spill_traffic_by_value_ordinal[value_ordinal];
  }
  if (!has_cached_traffic || traffic.store_count == UINT32_MAX) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_spill_plan_traffic(
        context->module, context->cfg_graph, assignment,
        capacity->alloc_unit_bits, &traffic));
    if (has_cached_traffic) {
      context->spill_traffic_by_value_ordinal[value_ordinal] = traffic;
    }
  }
  *out_cost =
      iree_math_saturating_add_u64(traffic.store_bytes, traffic.reload_bytes);
  return iree_ok_status();
}

static bool loom_low_allocation_search_spill_victim_set_is_better(
    uint16_t candidate_count, uint32_t candidate_unit_count,
    uint32_t candidate_latest_end_point, uint64_t candidate_traffic_cost,
    uint32_t candidate_location_base, uint16_t best_count,
    uint32_t best_unit_count, uint32_t best_latest_end_point,
    uint64_t best_traffic_cost, uint32_t best_location_base) {
  if (best_count == 0) {
    return true;
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
  return candidate_location_base < best_location_base;
}

static iree_status_t loom_low_allocation_search_collect_active_spill_victim_set(
    loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_allocation_class_capacity_t* capacity,
    uint32_t location_base, bool interval_requires_register,
    uint32_t* assignment_indices, loom_value_id_t* ignored_value_ids,
    uint16_t* out_assignment_count, uint32_t* out_unit_count,
    uint32_t* out_latest_end_point, uint64_t* out_traffic_cost,
    bool* out_blocked) {
  *out_assignment_count = 0;
  *out_unit_count = 0;
  *out_latest_end_point = 0;
  *out_traffic_cost = 0;
  *out_blocked = false;
  const uint16_t conflict_assignment_capacity =
      (uint16_t)context->active_set->count;

  const loom_low_allocation_assignment_t candidate =
      loom_low_allocation_search_candidate_assignment(
          context, interval, capacity->descriptor_reg_class_id,
          capacity->location_kind, location_base, interval->unit_count);
  const uint32_t interval_end = candidate.end_point;

  uint16_t conflict_assignment_count = 0;
  const bool active_unit_index_enabled =
      loom_low_allocation_active_unit_index_is_enabled(
          &context->active_set->units);
  if (active_unit_index_enabled) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_active_unit_index_collect_conflicts(
            &context->active_set->units, context->descriptor_set,
            context->liveness, context->unit_liveness,
            context->assignment_map->assignments,
            context->assignment_map->assignment_count, &candidate,
            /*ignored_value_ids=*/NULL,
            /*ignored_value_count=*/0, assignment_indices,
            conflict_assignment_capacity, &conflict_assignment_count));
  }
  const bool scan_all = !active_unit_index_enabled;
  const bool scan_unindexed =
      !scan_all && loom_low_allocation_active_unit_index_unindexed_count(
                       &context->active_set->units) != 0;
  if (scan_all || scan_unindexed) {
    for (iree_host_size_t i = 0; i < context->active_set->count; ++i) {
      const uint32_t assignment_index =
          context->active_set
              ->assignment_indices[context->active_set->start + i];
      IREE_ASSERT_LT(assignment_index,
                     context->assignment_map->assignment_count);
      if (scan_unindexed &&
          loom_low_allocation_active_unit_index_contains_assignment(
              &context->active_set->units, assignment_index)) {
        continue;
      }
      const loom_low_allocation_assignment_t* assignment =
          &context->assignment_map->assignments[assignment_index];
      if (!loom_low_allocation_active_assignment_conflicts(
              context->descriptor_set, context->liveness,
              context->unit_liveness, assignment, &candidate,
              /*ignored_value_ids=*/NULL,
              /*ignored_value_count=*/0)) {
        continue;
      }
      if (conflict_assignment_count == conflict_assignment_capacity) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "active allocation conflict set exceeds capacity");
      }
      assignment_indices[conflict_assignment_count++] = assignment_index;
    }
  }

  uint16_t assignment_count = 0;
  uint32_t unit_count = 0;
  uint32_t latest_end_point = 0;
  uint64_t traffic_cost = 0;
  for (uint16_t i = 0; i < conflict_assignment_count; ++i) {
    const uint32_t assignment_index = assignment_indices[i];
    const loom_low_allocation_assignment_t* assignment =
        &context->assignment_map->assignments[assignment_index];
    bool can_spill = false;
    loom_low_allocation_class_capacity_t spill_capacity = {0};
    IREE_RETURN_IF_ERROR(loom_low_allocation_search_assignment_spill_capacity(
        context, assignment, &can_spill, &spill_capacity));
    if (!can_spill) {
      *out_blocked = true;
      return iree_ok_status();
    }
    uint64_t assignment_traffic_cost = 0;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_search_assignment_spill_traffic_cost(
            context, assignment, &spill_capacity, &assignment_traffic_cost));
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
    traffic_cost =
        iree_math_saturating_add_u64(traffic_cost, assignment_traffic_cost);
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
          LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FOR_PRESSURE)) {
    *out_blocked = true;
    return iree_ok_status();
  }

  *out_assignment_count = assignment_count;
  *out_unit_count = unit_count;
  *out_latest_end_point = latest_end_point;
  *out_traffic_cost = traffic_cost;
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
  uint64_t best_traffic_cost = 0;
  uint32_t best_location_base = 0;
  for (uint32_t base = 0; base <= last_base;) {
    uint16_t candidate_assignment_count = 0;
    uint32_t candidate_unit_count = 0;
    uint32_t candidate_latest_end_point = 0;
    uint64_t candidate_traffic_cost = 0;
    bool blocked = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_search_collect_active_spill_victim_set(
            context, interval, capacity, base, interval_requires_register,
            candidate_assignment_indices, ignored_value_ids,
            &candidate_assignment_count, &candidate_unit_count,
            &candidate_latest_end_point, &candidate_traffic_cost, &blocked));
    if (!blocked &&
        loom_low_allocation_search_spill_victim_set_is_better(
            candidate_assignment_count, candidate_unit_count,
            candidate_latest_end_point, candidate_traffic_cost, base,
            best_assignment_count, best_unit_count, best_latest_end_point,
            best_traffic_cost, best_location_base)) {
      best_assignment_count = candidate_assignment_count;
      best_unit_count = candidate_unit_count;
      best_latest_end_point = candidate_latest_end_point;
      best_traffic_cost = candidate_traffic_cost;
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
