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
#include "loom/codegen/low/allocation/write_interference.h"
#include "loom/ir/module.h"
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
  const loom_value_ordinal_t value_ordinal =
      loom_module_value_ordinal_scratch_lookup(context->module,
                                               interval->value_id);
  const loom_liveness_segment_range_t segment_range =
      loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
          context->unit_liveness, context->liveness, value_ordinal);
  loom_low_allocation_assignment_t candidate = {
      .value_id = interval->value_id,
      .descriptor_reg_class_id = reg_class_id,
      .start_point =
          context->unit_liveness->values[value_ordinal].acquisition_start_point,
      .end_point =
          loom_low_allocation_live_range_interval_storage_end_point(interval),
      .liveness_segments = segment_range,
      .unit_count = interval->unit_count,
      .location_kind = location_kind,
      .location_base = location_base,
      .location_count = location_count,
      .unit_point_start =
          loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
              context->unit_liveness, context->liveness, value_ordinal),
  };
  candidate.end_point =
      loom_low_allocation_live_range_assignment_max_unit_end_point(
          context->unit_liveness->end_points,
          context->unit_liveness->point_count, &candidate);
  return candidate;
}

typedef struct loom_low_allocation_search_location_query_t {
  // Known active conflicts for the first word of scalar linear locations.
  uint64_t active_conflicts;
  // Physical-domain candidate ranks retained for this scalar interval.
  loom_low_allocation_physical_domain_row_t physical_domain;
  // Common structural and instruction preference objective.
  loom_low_allocation_preference_query_t preferences;
  // Best attainable tier before extending the current physical frontier.
  uint32_t tier_limit;
} loom_low_allocation_search_location_query_t;

static loom_low_allocation_search_location_query_t
loom_low_allocation_search_location_query(
    const loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* candidate) {
  loom_low_allocation_search_location_query_t query = {0};
  if (context->placement == NULL) {
    return query;
  }
  loom_value_ordinal_t ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_assignment_map_value_ordinal_for_value(
          context->assignment_map, candidate->value_id, &ordinal)) {
    return query;
  }
  query.preferences = loom_low_allocation_preference_prepare(
      context->preferences, context->placement, context->assignment_map,
      context->target_constraints, ordinal, LOOM_VALUE_ORDINAL_INVALID,
      context->preference_workspace);
  if ((query.preferences.use_count != 0 ||
       query.preferences.structural.placement != NULL) &&
      !loom_target_residency_model_is_empty(context->residency.model)) {
    const uint16_t reg_class = candidate->descriptor_reg_class_id;
    const uint32_t* extents =
        context->target_constraints->max_assigned_location_end_by_reg_class;
    query.tier_limit =
        loom_target_residency_evaluate_tier_with_direct_resource_override(
            context->residency, extents, reg_class, extents[reg_class]);
  }
  return query;
}

static uint32_t loom_low_allocation_search_location_preference_penalty(
    const loom_low_allocation_search_context_t* context,
    const loom_low_allocation_search_location_query_t* query,
    const loom_low_allocation_assignment_t* candidate) {
  return loom_low_allocation_preference_penalty(
      context->descriptor_set, &query->preferences, candidate, NULL);
}

static bool loom_low_allocation_search_hard_relation_conflicts(
    const loom_low_allocation_search_context_t* context,
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* candidate,
    bool candidate_is_result) {
  if (relation->kind != LOOM_LOW_PLACEMENT_RELATION_SAME_REGISTER_ORDINAL ||
      !iree_any_bit_set(relation->flags,
                        LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD)) {
    return false;
  }
  loom_value_ordinal_t counterpart_ordinal =
      candidate_is_result ? relation->source_ordinal : relation->result_ordinal;
  if (context->placement->tied_storage_origins_by_value_ordinal != NULL) {
    counterpart_ordinal =
        context->placement
            ->tied_storage_origins_by_value_ordinal[counterpart_ordinal];
  }
  const loom_low_allocation_assignment_t* counterpart =
      loom_low_allocation_assignment_map_assignment_for_value_ordinal(
          context->assignment_map, counterpart_ordinal, NULL);
  if (counterpart == NULL) {
    const loom_low_allocation_resolved_fixed_value_t* fixed =
        loom_low_allocation_target_constraints_fixed_value_for_value(
            context->target_constraints,
            loom_low_placement_value_id(context->placement,
                                        counterpart_ordinal));
    counterpart = fixed ? &fixed->assignment : NULL;
  }
  if (counterpart == NULL) {
    return false;
  }
  const loom_low_allocation_assignment_t* result_assignment =
      candidate_is_result ? candidate : counterpart;
  const loom_low_allocation_assignment_t* source_assignment =
      candidate_is_result ? counterpart : candidate;
  return !loom_low_allocation_storage_placement_relation_satisfied(
      context->descriptor_set, relation, result_assignment, source_assignment);
}

static bool loom_low_allocation_search_hard_relations_conflict(
    const loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* candidate) {
  const loom_low_placement_table_t* placement = context->placement;
  if (placement == NULL || placement->hard_location_relation_count == 0) {
    return false;
  }
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_assignment_map_value_ordinal_for_value(
          context->assignment_map, candidate->value_id, &value_ordinal)) {
    return false;
  }
  if (placement->tied_storage_origins_by_value_ordinal != NULL) {
    value_ordinal =
        placement->tied_storage_origins_by_value_ordinal[value_ordinal];
  }

  const loom_low_placement_relation_range_t result_range =
      loom_low_placement_relation_range_for_value_ordinal(placement,
                                                          value_ordinal);
  for (uint32_t i = 0; i < result_range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &placement->relations[result_range.start + i];
    if (loom_low_allocation_search_hard_relation_conflicts(
            context, relation, candidate, /*candidate_is_result=*/true)) {
      return true;
    }
  }

  const loom_low_placement_relation_range_t source_range =
      loom_low_placement_relation_range_for_source_value_ordinal(placement,
                                                                 value_ordinal);
  for (uint32_t i = 0; i < source_range.count; ++i) {
    const uint32_t relation_index =
        placement->relation_indices_by_source_ordinal[source_range.start + i];
    const loom_low_placement_relation_t* relation =
        &placement->relations[relation_index];
    if (loom_low_allocation_search_hard_relation_conflicts(
            context, relation, candidate, /*candidate_is_result=*/false)) {
      return true;
    }
  }
  return false;
}

bool loom_low_allocation_search_assignment_conflicts(
    loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count,
    const loom_value_id_t* ignored_storage_lease_value_ids,
    uint16_t ignored_storage_lease_value_count,
    loom_low_allocation_storage_release_policy_t release_policy) {
  const loom_value_ordinal_t retained_origin =
      loom_low_allocation_write_interference_conflicting_read(
          context->unit_liveness->write_interference, context->assignment_map,
          candidate);
  if (retained_origin != LOOM_VALUE_ORDINAL_INVALID) {
    if (context->retained_fixed_value_index_plus_one == 0 &&
        context->target_constraints->fixed_value_count != 0) {
      context->retained_fixed_value_index_plus_one =
          context->target_constraints
              ->fixed_value_indices_by_ordinal[retained_origin];
    }
    return true;
  }
  if (loom_low_allocation_search_hard_relations_conflict(context, candidate)) {
    return true;
  }
  if (loom_low_allocation_active_set_conflicts(
          context->active_set, context->descriptor_set, context->unit_liveness,
          context->assignment_map->assignments,
          context->assignment_map->assignment_count, candidate,
          ignored_value_ids, ignored_value_count)) {
    return true;
  }
  if (loom_low_allocation_target_constraints_fixed_storage_conflicts(
          context->target_constraints, context->unit_liveness, candidate,
          ignored_value_ids, ignored_value_count)) {
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
  // Semantic candidate ordinal of the selected location.
  uint32_t candidate_ordinal;
  // Smallest legal semantic ordinal for explicit physical-view pressure-release
  // comparisons. Linear searches select the policy before packing placement.
  uint32_t first_candidate_ordinal;
  // Aggregate-preserving rank used between equal-penalty physical candidates.
  uint32_t packing_rank;
  // Soft placement penalty for base.
  uint32_t preference_penalty;
  // Attainable residency after placing this candidate.
  uint32_t residency_tier;
  // Retained physical-domain rank after placement preferences.
  uint32_t physical_domain_rank;
  // Whether the candidate consumes reserved narrower-domain capacity.
  bool physical_domain_reserved;
  // True when the choice fields are populated.
  bool found;
} loom_low_allocation_search_location_choice_t;

static bool loom_low_allocation_search_explicit_choice_is_better(
    bool candidate_reserved, uint32_t candidate_tier,
    uint32_t candidate_preference_penalty, uint32_t candidate_domain_rank,
    uint32_t candidate_packing_rank, uint32_t candidate_ordinal,
    const loom_low_allocation_search_location_choice_t* best) {
  if (!best->found) {
    return true;
  }
  if (candidate_reserved != best->physical_domain_reserved) {
    return !candidate_reserved;
  }
  if (candidate_tier != best->residency_tier) {
    return candidate_tier > best->residency_tier;
  }
  if (candidate_preference_penalty != best->preference_penalty) {
    return candidate_preference_penalty < best->preference_penalty;
  }
  if (candidate_domain_rank != best->physical_domain_rank) {
    return candidate_domain_rank < best->physical_domain_rank;
  }
  return candidate_packing_rank < best->packing_rank ||
         (candidate_packing_rank == best->packing_rank &&
          candidate_ordinal < best->candidate_ordinal);
}

uint32_t loom_low_allocation_search_linear_candidate_base(
    uint64_t candidate_index, uint32_t last_base, uint32_t required_alignment,
    uint32_t preferred_alignment) {
  if (required_alignment == preferred_alignment) {
    return (uint32_t)(candidate_index * required_alignment);
  }
  const uint64_t preferred_count =
      (uint64_t)last_base / preferred_alignment + 1;
  if (candidate_index < preferred_count) {
    return (uint32_t)(candidate_index * preferred_alignment);
  }
  const uint64_t remaining_index = candidate_index - preferred_count;
  const uint32_t remaining_per_group =
      preferred_alignment / required_alignment - 1;
  return (
      uint32_t)((remaining_index + 1 + remaining_index / remaining_per_group) *
                required_alignment);
}

uint32_t loom_low_allocation_search_assignment_residency_tier(
    const loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* candidate) {
  if (loom_target_residency_model_is_empty(context->residency.model)) {
    return 0;
  }
  const uint16_t reg_class = candidate->descriptor_reg_class_id;
  const uint32_t* extents =
      context->target_constraints->max_assigned_location_end_by_reg_class;
  const uint32_t units =
      iree_max(extents[reg_class],
               loom_low_allocation_storage_assignment_pressure_extent(
                   context->descriptor_set, candidate));
  return loom_target_residency_evaluate_tier_with_direct_resource_override(
      context->residency, extents, reg_class, units);
}

static void loom_low_allocation_search_find_location_for_release_policy(
    loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* candidate_template,
    const loom_low_allocation_search_location_query_t* query,
    uint32_t minimum_base, uint32_t last_base, uint32_t alignment,
    uint32_t scalar_packing_frontier,
    loom_low_allocation_storage_release_policy_t release_policy,
    loom_low_allocation_search_location_choice_t* out_choice) {
  *out_choice = (loom_low_allocation_search_location_choice_t){0};
  const uint64_t candidate_count = (uint64_t)last_base / alignment + 1;
  const uint64_t packing_count =
      ((uint64_t)scalar_packing_frontier + alignment - 1) / alignment;
  const uint32_t preferred_alignment = iree_max(
      alignment,
      loom_low_reg_class_preferred_unit_alignment(
          &context->descriptor_set
               ->reg_classes[candidate_template->descriptor_reg_class_id],
          candidate_template->unit_count));

  // Scalars pack from high to low below their frontier, then low to high above
  // it. Tuples visit preferred-aligned bases before the remaining legal bases.
  // This breaks ties between equal penalties, so the first legal zero-penalty
  // candidate is final and later equal penalties cannot improve it.
  for (uint64_t i = 0; i < candidate_count; ++i) {
    const uint64_t ordinal = i < packing_count ? packing_count - i - 1 : i;
    const uint32_t base = loom_low_allocation_search_linear_candidate_base(
        ordinal, last_base, alignment, preferred_alignment);
    if (base < minimum_base) {
      continue;
    }
    if (base < 64 && (query->active_conflicts & (UINT64_C(1) << base))) {
      continue;
    }
    loom_low_allocation_assignment_t candidate = *candidate_template;
    candidate.location_base = base;
    if (loom_low_allocation_search_assignment_conflicts(
            context, &candidate,
            /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0,
            /*ignored_storage_lease_value_ids=*/NULL,
            /*ignored_storage_lease_value_count=*/0, release_policy)) {
      continue;
    }
    const uint32_t preference_penalty =
        loom_low_allocation_search_location_preference_penalty(context, query,
                                                               &candidate);
    const uint32_t tier =
        query->preferences.structural.placement != NULL ||
                query->preferences.use_count != 0
            ? loom_low_allocation_search_assignment_residency_tier(context,
                                                                   &candidate)
            : query->tier_limit;
    if (out_choice->found &&
        (tier < out_choice->residency_tier ||
         (tier == out_choice->residency_tier &&
          preference_penalty >= out_choice->preference_penalty))) {
      continue;
    }
    *out_choice = (loom_low_allocation_search_location_choice_t){
        .base = base,
        .candidate_ordinal = base,
        .preference_penalty = preference_penalty,
        .residency_tier = tier,
        .found = true,
    };
    if (preference_penalty == 0 && tier == query->tier_limit) {
      return;
    }
  }
}

// Pressure release can only add feasible bases. Its minimum is strictly lower
// than the forbidden minimum exactly when the first pressure-legal candidate
// still has a forbidden lease conflict. Select that policy before searching in
// packing order, reusing the minimum to exclude the proven infeasible prefix.
static loom_low_allocation_storage_release_policy_t
loom_low_allocation_search_find_linear_pressure_choice(
    loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* candidate_template,
    const loom_low_allocation_search_location_query_t* query,
    uint32_t last_base, uint32_t alignment, uint32_t scalar_packing_frontier,
    loom_low_allocation_search_location_choice_t* out_choice) {
  *out_choice = (loom_low_allocation_search_location_choice_t){0};
  for (uint64_t base = 0; base <= last_base; base += alignment) {
    if (base < 64 && (query->active_conflicts & (UINT64_C(1) << base))) {
      continue;
    }
    loom_low_allocation_assignment_t candidate = *candidate_template;
    candidate.location_base = (uint32_t)base;
    if (loom_low_allocation_search_assignment_conflicts(
            context, &candidate,
            /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0,
            /*ignored_storage_lease_value_ids=*/NULL,
            /*ignored_storage_lease_value_count=*/0,
            LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FOR_PRESSURE)) {
      continue;
    }
    const loom_low_allocation_storage_release_policy_t policy =
        loom_low_allocation_storage_lease_state_conflicts(
            context->storage_leases, context->descriptor_set, context->liveness,
            &candidate, /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0,
            LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN)
            ? LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FOR_PRESSURE
            : LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN;
    loom_low_allocation_search_find_location_for_release_policy(
        context, candidate_template, query, (uint32_t)base, last_base,
        alignment, scalar_packing_frontier, policy, out_choice);
    return policy;
  }
  return LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN;
}

// Returns the exclusive upper frontier where scalar values should pack from
// high to low. Semantic pressure can count overlapping aliases separately, so
// exceeding the capacity does not prove that physical storage cannot fit.
static uint32_t loom_low_allocation_search_scalar_packing_frontier(
    const loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_allocation_class_capacity_t* capacity) {
  if (interval->unit_count != 1 || !capacity->is_bounded ||
      context->scalar_packing.overlapping_intervals.bit_count == 0) {
    return 0;
  }
  const iree_host_size_t interval_index =
      interval - context->liveness->intervals;
  if (!iree_bitmap_test(context->scalar_packing.overlapping_intervals,
                        interval_index)) {
    return 0;
  }
  const uint32_t frontier =
      context->scalar_packing
          .frontiers_by_reg_class[capacity->descriptor_reg_class_id];
  return iree_min(frontier, capacity->max_units);
}

static void
loom_low_allocation_search_find_explicit_physical_register_for_release_policy(
    loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* candidate_template,
    const loom_low_allocation_search_location_query_t* query,
    uint16_t maximum_pressure_extent,
    loom_low_allocation_storage_release_policy_t release_policy,
    loom_low_allocation_search_location_choice_t* out_choice) {
  *out_choice = (loom_low_allocation_search_location_choice_t){0};
  const loom_low_descriptor_set_t* descriptor_set = context->descriptor_set;
  const uint16_t reg_class_id = candidate_template->descriptor_reg_class_id;
  const loom_low_reg_class_t* reg_class =
      &descriptor_set->reg_classes[reg_class_id];
  const uint32_t unit_count = candidate_template->location_count;
  const uint32_t candidate_count =
      unit_count == 1 ? reg_class->allocatable_count
                      : descriptor_set->physical_register_view_count;
  const bool needs_first_candidate =
      loom_low_allocation_search_has_pressure_release_records(context);
  for (uint32_t i = 0; i < candidate_count; ++i) {
    uint32_t physical_register_id = 0;
    uint32_t candidate_ordinal = 0;
    uint32_t packing_rank = i;
    uint32_t pressure_extent = 0;
    if (unit_count == 1) {
      candidate_ordinal =
          descriptor_set->physical_register_allocation_ordinals
              [reg_class->physical_register_candidate_start + i];
      if (candidate_ordinal >= maximum_pressure_extent) {
        continue;
      }
      physical_register_id =
          loom_low_descriptor_set_physical_register_candidate(
              descriptor_set, reg_class_id, (uint16_t)candidate_ordinal);
    } else {
      const loom_low_physical_register_view_t* view =
          &descriptor_set->physical_register_views[i];
      if (view->reg_class_id != reg_class_id ||
          view->unit_count != unit_count) {
        continue;
      }
      const uint16_t* ordinals =
          loom_low_descriptor_set_physical_register_view_unit_candidate_ordinals(
              descriptor_set, view);
      candidate_ordinal = ordinals[0];
      packing_rank = view->packing_rank;
      for (uint32_t unit = 0; unit < unit_count; ++unit) {
        pressure_extent =
            iree_max(pressure_extent, (uint32_t)ordinals[unit] + 1);
      }
      if (pressure_extent > maximum_pressure_extent) {
        continue;
      }
      physical_register_id = view->physical_register_id;
    }
    const uint32_t domain_penalty =
        loom_low_allocation_physical_domain_row_candidate_rank(
            query->physical_domain, (uint16_t)candidate_ordinal);
    const bool domain_reserved =
        loom_low_allocation_physical_domain_row_candidate_is_reserved(
            query->physical_domain, (uint16_t)candidate_ordinal);
    // Reserved narrower-domain capacity cannot improve on an already legal
    // non-reserved choice. Pressure release still needs every legal ordinal.
    if (!needs_first_candidate && out_choice->found &&
        !out_choice->physical_domain_reserved && domain_reserved) {
      continue;
    }
    loom_low_allocation_assignment_t candidate = *candidate_template;
    candidate.location_base = physical_register_id;
    if (loom_low_allocation_search_assignment_conflicts(
            context, &candidate,
            /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0,
            /*ignored_storage_lease_value_ids=*/NULL,
            /*ignored_storage_lease_value_count=*/0, release_policy)) {
      continue;
    }
    const uint32_t preference_penalty =
        loom_low_allocation_search_location_preference_penalty(context, query,
                                                               &candidate);
    const uint32_t tier =
        query->preferences.structural.placement != NULL ||
                query->preferences.use_count != 0
            ? loom_low_allocation_search_assignment_residency_tier(context,
                                                                   &candidate)
            : query->tier_limit;
    const uint32_t first_candidate_ordinal =
        out_choice->found
            ? iree_min(out_choice->first_candidate_ordinal, candidate_ordinal)
            : candidate_ordinal;
    if (loom_low_allocation_search_explicit_choice_is_better(
            domain_reserved, tier, preference_penalty, domain_penalty,
            packing_rank, candidate_ordinal, out_choice)) {
      *out_choice = (loom_low_allocation_search_location_choice_t){
          .base = physical_register_id,
          .candidate_ordinal = candidate_ordinal,
          .first_candidate_ordinal = first_candidate_ordinal,
          .packing_rank = packing_rank,
          .preference_penalty = preference_penalty,
          .residency_tier = tier,
          .physical_domain_rank = domain_penalty,
          .physical_domain_reserved = domain_reserved,
          .found = true,
      };
    } else {
      out_choice->first_candidate_ordinal = first_candidate_ordinal;
    }
    // Scalars are visited in packing order, so the first unconstrained,
    // zero-penalty choice is final unless pressure-release comparison also
    // needs the minimum semantic ordinal. Views retain physical-ID order for
    // indexed lookup.
    if (unit_count == 1 && !domain_reserved && preference_penalty == 0 &&
        tier == query->tier_limit && domain_penalty == 0 &&
        !needs_first_candidate) {
      return;
    }
  }
}

static void loom_low_allocation_search_find_for_release_policy(
    loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* candidate_template,
    const loom_low_allocation_search_location_query_t* query,
    bool uses_explicit_physical_registers, uint16_t explicit_candidate_count,
    uint32_t last_base, uint32_t alignment, uint32_t scalar_packing_frontier,
    loom_low_allocation_storage_release_policy_t release_policy,
    loom_low_allocation_search_location_choice_t* out_choice) {
  if (uses_explicit_physical_registers) {
    loom_low_allocation_search_find_explicit_physical_register_for_release_policy(
        context, candidate_template, query, explicit_candidate_count,
        release_policy, out_choice);
  } else {
    loom_low_allocation_search_find_location_for_release_policy(
        context, candidate_template, query, /*minimum_base=*/0, last_base,
        alignment, scalar_packing_frontier, release_policy, out_choice);
  }
}

static uint32_t loom_low_allocation_search_location_residency_tier(
    const loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* candidate_template,
    const loom_low_allocation_search_location_choice_t* choice) {
  loom_low_allocation_assignment_t choice_assignment = *candidate_template;
  choice_assignment.location_base = choice->base;
  return loom_low_allocation_search_assignment_residency_tier(
      context, &choice_assignment);
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
          context->residency, current_units_by_reg_class, reg_class_id,
          current_units_by_reg_class[reg_class_id]);
  *out_choice_tier = loom_low_allocation_search_location_residency_tier(
      context, candidate_template, choice);
  return *out_choice_tier < current_tier;
}

bool loom_low_allocation_search_find_free_location(
    loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* interval,
    loom_low_allocation_class_capacity_t capacity, uint32_t* out_base) {
  context->retained_fixed_value_index_plus_one = 0;
  const loom_low_reg_class_t* reg_class =
      &context->descriptor_set->reg_classes[capacity.descriptor_reg_class_id];
  const bool uses_explicit_physical_registers =
      loom_low_reg_class_uses_explicit_physical_registers(reg_class);
  if (!uses_explicit_physical_registers && capacity.is_bounded &&
      interval->unit_count > capacity.max_units) {
    return false;
  }

  const uint32_t alignment = loom_low_allocation_live_range_interval_alignment(
      context->descriptor_set, context->liveness, context->placement, interval);
  uint32_t last_base = 0;
  if (!uses_explicit_physical_registers && capacity.is_bounded) {
    last_base = capacity.max_units - interval->unit_count;
  } else if (!uses_explicit_physical_registers) {
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
  loom_low_allocation_search_location_query_t query =
      loom_low_allocation_search_location_query(context, &candidate_template);
  // A word of known conflicts avoids repeated hash probes while preserving
  // every constraint check for the remaining candidates. Bound projection
  // work independently of function size; large active sets keep indexed probes.
  if (!uses_explicit_physical_registers && interval->unit_count == 1 &&
      context->active_set->count <= 64) {
    query.active_conflicts =
        loom_low_allocation_active_set_conflicting_locations(
            context->active_set, context->descriptor_set,
            context->unit_liveness, context->assignment_map->assignments,
            &candidate_template);
  }
  if (uses_explicit_physical_registers && interval->unit_count == 1) {
    query.physical_domain = loom_low_allocation_physical_domains_for_interval(
        context->physical_domains, context->liveness, interval);
  }
  const uint32_t scalar_packing_frontier =
      loom_low_allocation_search_scalar_packing_frontier(context, interval,
                                                         &capacity);
  loom_low_allocation_search_location_choice_t release_free = {0};
  const uint16_t explicit_candidate_count =
      uses_explicit_physical_registers
          ? (uint16_t)iree_min(
                (uint32_t)reg_class->allocatable_count,
                capacity.is_bounded ? capacity.max_units : UINT32_MAX)
          : 0;
  if (!uses_explicit_physical_registers &&
      loom_low_allocation_search_has_pressure_release_records(context)) {
    const loom_low_allocation_storage_release_policy_t policy =
        loom_low_allocation_search_find_linear_pressure_choice(
            context, &candidate_template, &query, last_base, alignment,
            scalar_packing_frontier, &release_free);
    if (policy == LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FOR_PRESSURE) {
      *out_base = release_free.base;
      return true;
    }
  } else {
    loom_low_allocation_search_find_for_release_policy(
        context, &candidate_template, &query, uses_explicit_physical_registers,
        explicit_candidate_count, last_base, alignment, scalar_packing_frontier,
        LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN, &release_free);
  }
  loom_low_allocation_search_location_choice_t pressure_release = {0};
  if (uses_explicit_physical_registers &&
      loom_low_allocation_search_has_pressure_release_records(context) &&
      (!release_free.found || release_free.first_candidate_ordinal != 0)) {
    loom_low_allocation_search_find_for_release_policy(
        context, &candidate_template, &query, uses_explicit_physical_registers,
        explicit_candidate_count, last_base, alignment, scalar_packing_frontier,
        LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FOR_PRESSURE, &pressure_release);
  }
  if (pressure_release.found &&
      (!release_free.found || pressure_release.first_candidate_ordinal <
                                  release_free.first_candidate_ordinal)) {
    *out_base = pressure_release.base;
    return true;
  }
  const bool has_storage_release_records =
      loom_low_allocation_search_has_storage_release_records(context);
  loom_low_allocation_search_location_choice_t release_allowed = {0};
  bool searched_release_allowed = false;
  if (release_free.found && has_storage_release_records &&
      !loom_target_residency_model_is_empty(context->residency.model)) {
    uint32_t release_free_tier = 0;
    if (loom_low_allocation_search_location_crosses_residency_cliff(
            context, &candidate_template, &release_free, &release_free_tier)) {
      loom_low_allocation_search_find_for_release_policy(
          context, &candidate_template, &query,
          uses_explicit_physical_registers, explicit_candidate_count, last_base,
          alignment, scalar_packing_frontier,
          LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED, &release_allowed);
      searched_release_allowed = true;
      if (release_allowed.found &&
          release_allowed.candidate_ordinal < release_free.candidate_ordinal &&
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
    loom_low_allocation_search_find_for_release_policy(
        context, &candidate_template, &query, uses_explicit_physical_registers,
        explicit_candidate_count, last_base, alignment, scalar_packing_frontier,
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
    uint32_t candidate_location_ordinal, uint16_t best_count,
    uint32_t best_unit_count, uint32_t best_latest_end_point,
    uint64_t best_traffic_cost, uint32_t best_location_ordinal) {
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
  return candidate_location_ordinal < best_location_ordinal;
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
            context->unit_liveness, context->assignment_map->assignments,
            context->assignment_map->assignment_count, &candidate,
            /*ignored_value_ids=*/NULL,
            /*ignored_value_count=*/0, assignment_indices,
            conflict_assignment_capacity, &conflict_assignment_count));
  } else {
    for (iree_host_size_t i = 0; i < context->active_set->count; ++i) {
      const uint32_t assignment_index =
          context->active_set->assignment_indices[i];
      IREE_ASSERT_LT(assignment_index,
                     context->assignment_map->assignment_count);
      const loom_low_allocation_assignment_t* assignment =
          &context->assignment_map->assignments[assignment_index];
      if (!loom_low_allocation_active_assignment_conflicts(
              context->descriptor_set, context->unit_liveness, assignment,
              &candidate,
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
    bool interval_requires_register,
    loom_low_allocation_search_workspace_t* workspace,
    iree_arena_allocator_t* arena,
    loom_low_allocation_search_spill_victim_set_t* out_victim_set) {
  *out_victim_set = (loom_low_allocation_search_spill_victim_set_t){0};
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
      context->descriptor_set, context->liveness, context->placement, interval);
  if (!uses_explicit_physical_registers && capacity->is_bounded) {
    last_base = capacity->max_units - interval->unit_count;
  } else if (!uses_explicit_physical_registers) {
    const uint32_t search_limit =
        loom_low_allocation_target_constraints_assigned_location_search_limit(
            context->target_constraints, capacity->descriptor_reg_class_id,
            capacity->location_kind);
    if (!loom_low_allocation_search_align_up_u32(search_limit, alignment,
                                                 &last_base)) {
      return iree_ok_status();
    }
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
  uint32_t best_location_base = 0;
  uint32_t best_location_ordinal = 0;
  const uint32_t explicit_pressure_limit =
      uses_explicit_physical_registers
          ? iree_min((uint32_t)reg_class->allocatable_count,
                     capacity->is_bounded ? capacity->max_units : UINT32_MAX)
          : 0;
  const uint64_t candidate_count =
      uses_explicit_physical_registers
          ? context->descriptor_set->physical_register_count
          : (uint64_t)last_base / alignment + 1u;
  for (uint64_t candidate_index = 0; candidate_index < candidate_count;
       ++candidate_index) {
    uint32_t candidate_ordinal = (uint32_t)candidate_index;
    uint32_t base = candidate_ordinal * alignment;
    if (uses_explicit_physical_registers) {
      uint32_t pressure_extent = 0;
      base = (uint32_t)candidate_index;
      if (!loom_low_allocation_storage_explicit_physical_register_view(
              context->descriptor_set, capacity->descriptor_reg_class_id, base,
              interval->unit_count, &candidate_ordinal, &pressure_extent) ||
          pressure_extent > explicit_pressure_limit) {
        continue;
      }
    }
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
            candidate_latest_end_point, candidate_traffic_cost,
            candidate_ordinal, best_assignment_count, best_unit_count,
            best_latest_end_point, best_traffic_cost, best_location_ordinal)) {
      best_assignment_count = candidate_assignment_count;
      best_unit_count = candidate_unit_count;
      best_latest_end_point = candidate_latest_end_point;
      best_traffic_cost = candidate_traffic_cost;
      best_location_base = base;
      best_location_ordinal = candidate_ordinal;
      memcpy(best_assignment_indices, candidate_assignment_indices,
             (iree_host_size_t)candidate_assignment_count *
                 sizeof(*best_assignment_indices));
    }
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
