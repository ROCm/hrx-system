// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/concat_reservation.h"

#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/target/residency.h"

static bool loom_low_allocation_concat_reservation_align_up_u32(
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

// Returns whether all concat sources are produced in one compact assembly
// window immediately before the result. Reserving a long-lived result for an
// early source carries its transient pressure through unrelated packets; a
// compact source cluster instead has no useful placement lifetime of its own.
static bool
loom_low_allocation_concat_reservation_sources_form_compact_assembly(
    const loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* result_interval,
    const loom_low_placement_relation_range_t* result_range) {
  uint32_t source_count = 0;
  uint32_t earliest_source_start = result_interval->start_point;
  for (uint32_t i = 0; i < result_range->count; ++i) {
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[result_range->start + i];
    if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT) {
      continue;
    }
    const loom_liveness_interval_t* source_interval =
        loom_liveness_interval_for_value_ordinal(context->liveness,
                                                 relation->source_ordinal);
    if (source_interval == NULL ||
        source_interval->end_point != result_interval->start_point) {
      return false;
    }
    earliest_source_start =
        iree_min(earliest_source_start, source_interval->start_point);
    ++source_count;
  }
  return source_count != 0 &&
         result_interval->start_point - earliest_source_start <= source_count;
}

// Returns whether immediate hard tied-result sources construct a concat across
// nonconsecutive program points. Reserving those units protects partially
// filled destinations from interleaved temporaries while leaving tightly
// packed source sequences on the ordinary concat path.
static bool
loom_low_allocation_concat_reservation_has_fragmented_tied_source_assembly(
    const loom_low_allocation_search_context_t* context,
    const loom_low_placement_relation_range_t* result_range,
    const uint32_t* result_unit_start_points) {
  uint32_t earliest_start_point = UINT32_MAX;
  uint32_t latest_start_point = 0;
  uint32_t tied_piece_count = 0;
  for (uint32_t i = 0; i < result_range->count; ++i) {
    const loom_low_placement_relation_t* concat_relation =
        &context->placement->relations[result_range->start + i];
    if (concat_relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT) {
      continue;
    }
    const loom_liveness_interval_t* source_interval =
        loom_liveness_interval_for_value_ordinal(
            context->liveness, concat_relation->source_ordinal);
    if (source_interval == NULL) {
      continue;
    }
    const loom_low_placement_relation_range_t source_result_range =
        loom_low_placement_relation_range_for_value_ordinal(
            context->placement, concat_relation->source_ordinal);
    uint32_t piece_start_point = UINT32_MAX;
    for (uint32_t j = 0; j < source_result_range.count; ++j) {
      const loom_low_placement_relation_t* tied_relation =
          &context->placement->relations[source_result_range.start + j];
      loom_low_placement_relation_t composed_relation;
      if (!loom_low_placement_relation_compose_tied_concat_source(
              tied_relation, concat_relation, &composed_relation)) {
        continue;
      }
      if (tied_relation->op == NULL || concat_relation->op == NULL ||
          tied_relation->op->parent_block !=
              concat_relation->op->parent_block) {
        continue;
      }
      for (uint32_t unit_index = 0; unit_index < composed_relation.unit_count;
           ++unit_index) {
        const uint32_t start_point =
            result_unit_start_points[composed_relation.result_unit_offset +
                                     unit_index];
        if (start_point < source_interval->start_point) {
          piece_start_point = iree_min(piece_start_point, start_point);
        }
      }
    }
    if (piece_start_point != UINT32_MAX) {
      earliest_start_point = iree_min(earliest_start_point, piece_start_point);
      latest_start_point = iree_max(latest_start_point, piece_start_point);
      ++tied_piece_count;
    }
  }
  return tied_piece_count != 0 &&
         latest_start_point - earliest_start_point >= tied_piece_count;
}

// Returns whether any source has already committed to a location. Result
// reservations are anticipatory: once allocation has begun assembling the
// sources, sibling coalescing or the materialized-concat path owns the
// remaining placement decision.
static bool loom_low_allocation_concat_reservation_has_assigned_source(
    const loom_low_allocation_search_context_t* context,
    const loom_low_placement_relation_range_t* result_range) {
  for (uint32_t i = 0; i < result_range->count; ++i) {
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[result_range->start + i];
    if (relation->cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT &&
        loom_low_allocation_assignment_map_assignment_for_value_ordinal(
            context->assignment_map, relation->source_ordinal, NULL) != NULL) {
      return true;
    }
  }
  return false;
}

// Returns whether normal placement of the current source can be extended to
// every sibling source without a materialized concat. This predicts the same
// locations that sibling coalescing will request later. This establishes a
// legal assembly candidate; instruction preferences may still favor another.
static iree_status_t
loom_low_allocation_concat_reservation_default_source_assembles_result(
    loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* source_interval,
    const loom_low_placement_relation_t* relation,
    const loom_liveness_interval_t* result_interval,
    loom_low_allocation_class_capacity_t source_capacity,
    loom_low_allocation_class_capacity_t result_capacity,
    uint32_t source_location_base,
    const loom_low_placement_relation_range_t* result_range,
    bool* out_assembles_result) {
  *out_assembles_result = false;
  if (source_location_base > UINT32_MAX - relation->source_unit_offset) {
    return iree_ok_status();
  }
  const uint32_t source_unit_location =
      source_location_base + relation->source_unit_offset;
  if (source_unit_location < relation->result_unit_offset) {
    return iree_ok_status();
  }
  const uint32_t result_location_base =
      source_unit_location - relation->result_unit_offset;
  const uint32_t result_alignment =
      loom_low_allocation_live_range_interval_alignment(
          context->descriptor_set, context->liveness, context->placement,
          result_interval);
  if (result_location_base % result_alignment != 0 ||
      !loom_low_allocation_storage_reg_classes_share(
          context->descriptor_set, source_capacity.descriptor_reg_class_id,
          result_capacity.descriptor_reg_class_id) ||
      !loom_low_allocation_target_constraints_location_range_fits_capacity(
          context->descriptor_set, &result_capacity,
          source_capacity.location_kind, result_location_base,
          result_interval->unit_count)) {
    return iree_ok_status();
  }

  for (uint32_t i = 0; i < result_range->count; ++i) {
    const loom_low_placement_relation_t* sibling_relation =
        &context->placement->relations[result_range->start + i];
    if (sibling_relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT ||
        !loom_low_placement_relation_can_alias(sibling_relation)) {
      continue;
    }
    const loom_liveness_interval_t* sibling_interval =
        loom_liveness_interval_for_value_ordinal(
            context->liveness, sibling_relation->source_ordinal);
    if (sibling_interval == NULL ||
        !loom_liveness_value_class_equal(source_interval->value_class,
                                         sibling_interval->value_class) ||
        result_location_base >
            UINT32_MAX - sibling_relation->result_unit_offset) {
      return iree_ok_status();
    }
    const uint32_t sibling_unit_location =
        result_location_base + sibling_relation->result_unit_offset;
    if (sibling_unit_location < sibling_relation->source_unit_offset) {
      return iree_ok_status();
    }
    const uint32_t sibling_location_base =
        sibling_unit_location - sibling_relation->source_unit_offset;
    loom_low_allocation_class_capacity_t sibling_capacity = {0};
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_target_constraints_interval_capacity(
            context->target_constraints, context->liveness, context->placement,
            sibling_interval, &sibling_capacity));
    const uint32_t sibling_alignment =
        loom_low_allocation_live_range_interval_alignment(
            context->descriptor_set, context->liveness, context->placement,
            sibling_interval);
    if (!loom_low_allocation_storage_reg_classes_share(
            context->descriptor_set, source_capacity.descriptor_reg_class_id,
            sibling_capacity.descriptor_reg_class_id) ||
        !loom_low_allocation_target_constraints_location_range_fits_capacity(
            context->descriptor_set, &sibling_capacity,
            source_capacity.location_kind, sibling_location_base,
            sibling_interval->unit_count) ||
        sibling_location_base % sibling_alignment != 0 ||
        loom_low_allocation_search_location_conflicts(
            context, sibling_interval, sibling_capacity.descriptor_reg_class_id,
            source_capacity.location_kind, sibling_location_base,
            sibling_interval->unit_count,
            /*ignored_value_ids=*/NULL,
            /*ignored_value_count=*/0,
            /*ignored_storage_lease_value_ids=*/NULL,
            /*ignored_storage_lease_value_count=*/0,
            LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FOR_PRESSURE)) {
      return iree_ok_status();
    }
  }

  // A future reservation can begin after every source has been consumed but
  // still overlap the assembled result. Piece placement alone does not prove
  // that the result can keep the same storage for its longer lifetime.
  *out_assembles_result = !loom_low_allocation_search_location_conflicts(
      context, result_interval, result_capacity.descriptor_reg_class_id,
      source_capacity.location_kind, result_location_base,
      result_interval->unit_count,
      /*ignored_value_ids=*/NULL,
      /*ignored_value_count=*/0,
      /*ignored_storage_lease_value_ids=*/NULL,
      /*ignored_storage_lease_value_count=*/0,
      LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FOR_PRESSURE);
  return iree_ok_status();
}

// Chooses a concat result span that can also accept the current source slice.
// Scheduled allocation may see scalar concat sources long before the concat op,
// so selecting only for the future result interval can reserve a span that the
// current source cannot occupy without a packet-local move.
static bool loom_low_allocation_concat_reservation_find_location_for_source(
    loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* source_interval,
    const loom_low_placement_relation_t* relation,
    const loom_liveness_interval_t* result_interval,
    loom_low_allocation_class_capacity_t source_capacity,
    loom_low_allocation_class_capacity_t capacity,
    uint32_t reservation_start_point,
    loom_low_allocation_assignment_flags_t reservation_flags,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count,
    const loom_low_allocation_preference_query_t* preferences,
    uint32_t default_result_location_base, uint32_t* out_result_location_base) {
  const loom_low_reg_class_t* reg_class =
      &context->descriptor_set->reg_classes[capacity.descriptor_reg_class_id];
  const bool is_explicit =
      loom_low_reg_class_uses_explicit_physical_registers(reg_class);
  if (is_explicit) {
    // Reserve only a local assembly with one placement for this source and no
    // downstream storage affinity. A fresh aggregate cannot choose placement
    // for a broadcast, shared source, or loop/join component independently.
    if (loom_low_placement_relation_range_for_source_value_ordinal(
            context->placement, relation->source_ordinal)
                .count != 1 ||
        loom_low_placement_relation_range_for_source_value_ordinal(
            context->placement, relation->result_ordinal)
                .count != 0) {
      return false;
    }
  }
  if (result_interval->unit_count == 0 ||
      (capacity.is_bounded &&
       result_interval->unit_count > capacity.max_units)) {
    return false;
  }

  const uint32_t result_alignment =
      loom_low_allocation_live_range_interval_alignment(
          context->descriptor_set, context->liveness, context->placement,
          result_interval);
  const uint32_t source_alignment =
      loom_low_allocation_live_range_interval_alignment(
          context->descriptor_set, context->liveness, context->placement,
          source_interval);
  const uint32_t assigned_limit =
      loom_low_allocation_target_constraints_assigned_location_search_limit(
          context->target_constraints, capacity.descriptor_reg_class_id,
          capacity.location_kind);

  uint32_t last_base = 0;
  if (!is_explicit) {
    if (capacity.is_bounded) {
      last_base = capacity.max_units - result_interval->unit_count;
    } else if (!loom_low_allocation_concat_reservation_align_up_u32(
                   assigned_limit, result_alignment, &last_base)) {
      return false;
    }
  }

  loom_low_allocation_assignment_t reservation = {
      .value_id = result_interval->value_id,
      .descriptor_reg_class_id = capacity.descriptor_reg_class_id,
      .start_point = reservation_start_point,
      .end_point = loom_low_allocation_live_range_interval_storage_end_point(
          result_interval),
      .unit_count = result_interval->unit_count,
      .location_kind = capacity.location_kind,
      .location_count = result_interval->unit_count,
      .unit_point_start =
          loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
              context->unit_liveness, context->liveness,
              relation->result_ordinal),
      .flags = reservation_flags,
  };
  if (reservation_start_point ==
      context->unit_liveness->values[relation->result_ordinal]
          .acquisition_start_point) {
    reservation.liveness_segments =
        loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
            context->unit_liveness, context->liveness,
            relation->result_ordinal);
  }
  reservation.end_point =
      loom_low_allocation_live_range_assignment_max_unit_end_point(
          context->unit_liveness->end_points,
          context->unit_liveness->point_count, &reservation);
  const loom_low_descriptor_set_t* descriptor_set = context->descriptor_set;
  const uint32_t preferred_alignment = iree_max(
      result_alignment,
      loom_low_reg_class_preferred_unit_alignment(
          &descriptor_set->reg_classes[capacity.descriptor_reg_class_id],
          result_interval->unit_count));
  loom_low_allocation_assignment_t source = {
      .value_id = source_interval->value_id,
      .descriptor_reg_class_id = source_capacity.descriptor_reg_class_id,
      .unit_count = source_interval->unit_count,
      .location_kind = source_capacity.location_kind,
      .location_count = source_interval->unit_count,
  };
  bool found = default_result_location_base != UINT32_MAX;
  uint32_t best_penalty = UINT32_MAX;
  uint32_t best_tier = 0;
  uint32_t tier_limit = 0;
  bool uniform_tier = true;
  if (preferences->use_count != 0 &&
      !loom_target_residency_model_is_empty(context->residency.model)) {
    const uint16_t reg_class_id = capacity.descriptor_reg_class_id;
    const uint32_t* extents =
        context->target_constraints->max_assigned_location_end_by_reg_class;
    tier_limit =
        loom_target_residency_evaluate_tier_with_direct_resource_override(
            context->residency, extents, reg_class_id, extents[reg_class_id]);
    // Residency is nonincreasing in the direct footprint. Equal endpoint
    // tiers therefore cover every legal linear candidate in the domain.
    uniform_tier =
        !is_explicit && capacity.is_bounded &&
        tier_limit ==
            loom_target_residency_evaluate_tier_with_direct_resource_override(
                context->residency, extents, reg_class_id,
                iree_max(extents[reg_class_id], capacity.max_units));
  }
  if (found) {
    reservation.location_base = default_result_location_base;
    source.location_base = default_result_location_base +
                           relation->result_unit_offset -
                           relation->source_unit_offset;
    best_penalty = loom_low_allocation_preference_penalty(
        descriptor_set, preferences, &source, &reservation);
    best_tier = uniform_tier
                    ? tier_limit
                    : loom_low_allocation_search_assignment_residency_tier(
                          context, &reservation);
    *out_result_location_base = default_result_location_base;
    if (best_penalty == 0 && best_tier == tier_limit) {
      return true;
    }
  }
  // Intersect source and result bounds once for their fixed relative offset.
  // A composed tied source need not be contained in the result. Both spans
  // must have representable endpoints even for unbounded register classes.
  const int64_t source_delta = (int64_t)relation->result_unit_offset -
                               (int64_t)relation->source_unit_offset;
  uint32_t first_base = 0;
  if (!is_explicit) {
    const uint32_t result_limit =
        capacity.is_bounded ? capacity.max_units : UINT32_MAX;
    const uint32_t source_limit =
        source_capacity.is_bounded ? source_capacity.max_units : UINT32_MAX;
    if (source_interval->unit_count > source_limit) {
      return found;
    }
    first_base = source_delta < 0 ? (uint32_t)-source_delta : 0;
    const int64_t last_source_base =
        (int64_t)(source_limit - source_interval->unit_count) - source_delta;
    if (last_source_base < first_base) {
      return found;
    }
    last_base = (uint32_t)iree_min(
        (uint64_t)iree_min(last_base,
                           result_limit - result_interval->unit_count),
        (uint64_t)last_source_base);
    if (last_base < first_base) {
      return found;
    }
  }
  const uint64_t candidate_count =
      is_explicit ? descriptor_set->physical_register_view_count
                  : (uint64_t)last_base / result_alignment + 1;
  for (uint64_t candidate_index = 0; candidate_index < candidate_count;
       ++candidate_index) {
    uint32_t base = 0;
    if (is_explicit) {
      const loom_low_physical_register_view_t* view =
          &descriptor_set->physical_register_views[candidate_index];
      if (view->reg_class_id != capacity.descriptor_reg_class_id ||
          view->unit_count != result_interval->unit_count) {
        continue;
      }
      base = view->physical_register_id;
      if (!loom_low_allocation_target_constraints_location_range_fits_capacity(
              descriptor_set, &capacity, capacity.location_kind, base,
              result_interval->unit_count)) {
        continue;
      }
    } else {
      base = loom_low_allocation_search_linear_candidate_base(
          candidate_index, last_base, result_alignment, preferred_alignment);
      if (base < first_base) {
        continue;
      }
    }
    reservation.location_base = base;
    bool source_location_ok = false;
    uint32_t source_location_base = 0;
    if (is_explicit) {
      if (source_interval->unit_count == 1) {
        // The declared result view already identifies each direct source
        // unit; physical IDs need not be consecutive or ordered by encoding.
        const loom_low_physical_register_view_t* view =
            &descriptor_set->physical_register_views[candidate_index];
        const uint16_t* ordinals =
            loom_low_descriptor_set_physical_register_view_unit_candidate_ordinals(
                descriptor_set, view);
        source_location_base =
            loom_low_descriptor_set_physical_register_candidate(
                descriptor_set, capacity.descriptor_reg_class_id,
                ordinals[relation->result_unit_offset]);
        source_location_ok = true;
      } else {
        source_location_ok =
            loom_low_allocation_storage_find_subrange_alias_location(
                descriptor_set, capacity.descriptor_reg_class_id,
                capacity.location_kind, source_interval->unit_count,
                relation->source_unit_offset, &reservation,
                relation->result_unit_offset, relation->unit_count,
                &source_location_base);
      }
      source_location_ok =
          source_location_ok &&
          loom_low_allocation_target_constraints_location_range_fits_capacity(
              descriptor_set, &source_capacity, source_capacity.location_kind,
              source_location_base, source_interval->unit_count);
    } else {
      source_location_base = (uint32_t)((int64_t)base + source_delta);
      source_location_ok = source_location_base % source_alignment == 0;
    }
    if (!source_location_ok) {
      continue;
    }
    uint32_t penalty = 0;
    uint32_t tier = 0;
    if (preferences->use_count != 0) {
      source.location_base = source_location_base;
      penalty = loom_low_allocation_preference_penalty(
          descriptor_set, preferences, &source, &reservation);
      if (found && best_tier == tier_limit && penalty >= best_penalty) {
        continue;
      }
      tier = uniform_tier
                 ? tier_limit
                 : loom_low_allocation_search_assignment_residency_tier(
                       context, &reservation);
      if (found && (tier < best_tier ||
                    (tier == best_tier && penalty >= best_penalty))) {
        continue;
      }
    }
    if (loom_low_allocation_search_assignment_conflicts(
            context, &reservation, ignored_value_ids, ignored_value_count,
            ignored_value_ids, ignored_value_count,
            LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FOR_PRESSURE) ||
        loom_low_allocation_search_location_conflicts(
            context, source_interval, source_capacity.descriptor_reg_class_id,
            source_capacity.location_kind, source_location_base,
            source_interval->unit_count,
            /*ignored_value_ids=*/NULL,
            /*ignored_value_count=*/0,
            /*ignored_storage_lease_value_ids=*/NULL,
            /*ignored_storage_lease_value_count=*/0,
            LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FOR_PRESSURE)) {
      continue;
    }
    found = true;
    best_tier = tier;
    best_penalty = penalty;
    *out_result_location_base = base;
    if (preferences->use_count == 0 ||
        (best_penalty == 0 && best_tier == tier_limit)) {
      return true;
    }
  }
  return found;
}

iree_status_t loom_low_allocation_concat_reservation_find(
    loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* source_interval,
    const loom_low_placement_relation_t* relation,
    const loom_liveness_interval_t* result_interval,
    const loom_low_placement_relation_range_t* result_range,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count,
    loom_low_allocation_assignment_t* out_assignment) {
  *out_assignment =
      (loom_low_allocation_assignment_t){.value_id = LOOM_VALUE_ID_INVALID};
  const uint32_t result_lifetime =
      result_interval->end_point - result_interval->start_point;
  loom_low_allocation_class_capacity_t capacity = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_interval_capacity(
      context->target_constraints, context->liveness, context->placement,
      result_interval, &capacity));

  if (ignored_value_count == 0) {
    return iree_ok_status();
  }

  // Reuse the ordinary source choice if both assembly prediction and residency
  // comparison need it. No assignment is published between these decisions.
  enum {
    SOURCE_LOCATION_UNQUERIED,
    SOURCE_LOCATION_UNAVAILABLE,
    SOURCE_LOCATION_AVAILABLE,
  } source_location_state = SOURCE_LOCATION_UNQUERIED;
  uint32_t source_location_base = 0;
  loom_low_allocation_class_capacity_t source_capacity = {0};
  const bool has_preferences =
      loom_low_allocation_preference_has_uses(
          context->preferences, context->placement, relation->source_ordinal) ||
      loom_low_allocation_preference_has_uses(
          context->preferences, context->placement, relation->result_ordinal);
  bool default_source_assembles_result = false;

  const bool sources_form_compact_assembly =
      loom_low_allocation_concat_reservation_sources_form_compact_assembly(
          context, result_interval, result_range);
  const uint32_t* result_unit_start_points =
      loom_low_allocation_unit_liveness_start_points_for_value_ordinal(
          context->unit_liveness, context->liveness, relation->result_ordinal);
  IREE_ASSERT(result_unit_start_points != NULL);
  const bool has_fragmented_tied_source_assembly =
      loom_low_allocation_concat_reservation_has_fragmented_tied_source_assembly(
          context, result_range, result_unit_start_points);
  // Scalar sources can cheaply reserve a future aggregate, and packet-local
  // or tied in-place sources must reserve before independently allocated
  // temporaries fragment their required span. Explicit physical views also
  // require choosing the aggregate before its first source: an independently
  // placed vector can occupy the wrong subrange even when all sources fit.
  if (source_interval->unit_count != 1 && result_lifetime != 1) {
    const bool uses_explicit_physical_registers =
        loom_low_reg_class_uses_explicit_physical_registers(
            &context->descriptor_set
                 ->reg_classes[capacity.descriptor_reg_class_id]);
    const bool favor_result_reservation = uses_explicit_physical_registers ||
                                          sources_form_compact_assembly ||
                                          has_fragmented_tied_source_assembly;
    if (loom_low_allocation_concat_reservation_has_assigned_source(
            context, result_range)) {
      return iree_ok_status();
    }
    if (favor_result_reservation && !uses_explicit_physical_registers) {
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_target_constraints_interval_capacity(
              context->target_constraints, context->liveness,
              context->placement, source_interval, &source_capacity));
      source_location_state =
          loom_low_allocation_search_find_free_location(
              context, source_interval, source_capacity, &source_location_base)
              ? SOURCE_LOCATION_AVAILABLE
              : SOURCE_LOCATION_UNAVAILABLE;
      if (source_location_state == SOURCE_LOCATION_AVAILABLE) {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_concat_reservation_default_source_assembles_result(
                context, source_interval, relation, result_interval,
                source_capacity, capacity, source_location_base, result_range,
                &default_source_assembles_result));
      }
      if (default_source_assembles_result && !has_preferences) {
        return iree_ok_status();
      }
    } else if (!favor_result_reservation &&
               loom_target_residency_model_is_empty(context->residency.model)) {
      return iree_ok_status();
    }
  }

  // Finish the ordinary query before borrowing aggregate query storage. Both
  // alternatives then score the same merged use set, with no double counting
  // when a preference names both the source and its future aggregate.
  if (has_preferences && source_location_state == SOURCE_LOCATION_UNQUERIED) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_target_constraints_interval_capacity(
            context->target_constraints, context->liveness, context->placement,
            source_interval, &source_capacity));
    source_location_state =
        loom_low_allocation_search_find_free_location(
            context, source_interval, source_capacity, &source_location_base)
            ? SOURCE_LOCATION_AVAILABLE
            : SOURCE_LOCATION_UNAVAILABLE;
    if (source_location_state == SOURCE_LOCATION_AVAILABLE &&
        !loom_low_reg_class_uses_explicit_physical_registers(
            &context->descriptor_set
                 ->reg_classes[capacity.descriptor_reg_class_id])) {
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_concat_reservation_default_source_assembles_result(
              context, source_interval, relation, result_interval,
              source_capacity, capacity, source_location_base, result_range,
              &default_source_assembles_result));
    }
  }
  if (source_location_state == SOURCE_LOCATION_UNQUERIED) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_target_constraints_interval_capacity(
            context->target_constraints, context->liveness, context->placement,
            source_interval, &source_capacity));
  }
  const loom_low_allocation_preference_query_t preferences =
      loom_low_allocation_preference_prepare(
          context->preferences, context->placement, context->assignment_map,
          context->target_constraints, relation->source_ordinal,
          relation->result_ordinal, context->preference_workspace);
  const uint32_t default_result_location_base =
      default_source_assembles_result
          ? source_location_base + relation->source_unit_offset -
                relation->result_unit_offset
          : UINT32_MAX;

  uint32_t reservation_start_point =
      context->unit_liveness->values[relation->result_ordinal]
          .acquisition_start_point;
  loom_low_allocation_assignment_flags_t reservation_flags = 0;
  if (has_fragmented_tied_source_assembly) {
    reservation_flags = LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS;
    for (uint32_t unit_index = 0; unit_index < result_interval->unit_count;
         ++unit_index) {
      reservation_start_point = iree_min(reservation_start_point,
                                         result_unit_start_points[unit_index]);
    }
  }

  uint32_t result_location_base = 0;
  if (!loom_low_allocation_concat_reservation_find_location_for_source(
          context, source_interval, relation, result_interval, source_capacity,
          capacity, reservation_start_point, reservation_flags,
          ignored_value_ids, ignored_value_count, &preferences,
          default_result_location_base, &result_location_base)) {
    return iree_ok_status();
  }
  if (result_location_base == default_result_location_base) {
    return iree_ok_status();
  }

  // Even compact assemblies can cross a residency cliff when preserving a
  // source lease. Compare ordinary placement only when the reservation lowers
  // residency; within the current tier, a second location search has no value.
  // Resources with no direct cliffs or derived consumers cannot lower a tier.
  const loom_target_residency_model_t* residency_model =
      context->residency.model;
  const uint16_t reg_class_id = capacity.descriptor_reg_class_id;
  const uint32_t* current_units_by_reg_class =
      context->target_constraints->max_assigned_location_end_by_reg_class;
  const uint32_t current_location_end =
      current_units_by_reg_class[reg_class_id];
  const loom_low_allocation_assignment_t result_assignment = {
      .descriptor_reg_class_id = reg_class_id,
      .location_kind = capacity.location_kind,
      .location_base = result_location_base,
      .location_count = result_interval->unit_count,
  };
  const uint32_t result_location_end =
      loom_low_allocation_storage_assignment_pressure_extent(
          context->descriptor_set, &result_assignment);
  if (result_location_end > current_location_end &&
      !loom_target_residency_model_is_empty(residency_model) &&
      (loom_target_residency_direct_resource_cliff_range(
           &residency_model->direct_resources, reg_class_id)
               .count != 0 ||
       !loom_target_residency_derived_resource_table_is_empty(
           &residency_model->derived_resources))) {
    const uint32_t current_tier =
        loom_target_residency_evaluate_tier_with_direct_resource_override(
            context->residency, current_units_by_reg_class, reg_class_id,
            current_location_end);
    const uint32_t result_tier =
        loom_target_residency_evaluate_tier_with_direct_resource_override(
            context->residency, current_units_by_reg_class, reg_class_id,
            result_location_end);
    if (result_tier < current_tier &&
        source_location_state == SOURCE_LOCATION_UNQUERIED) {
      source_location_state =
          loom_low_allocation_search_find_free_location(
              context, source_interval, source_capacity, &source_location_base)
              ? SOURCE_LOCATION_AVAILABLE
              : SOURCE_LOCATION_UNAVAILABLE;
    }
    if (result_tier < current_tier &&
        source_location_state == SOURCE_LOCATION_AVAILABLE) {
      const loom_low_allocation_assignment_t source_assignment = {
          .descriptor_reg_class_id = reg_class_id,
          .location_kind = capacity.location_kind,
          .location_base = source_location_base,
          .location_count = source_interval->unit_count,
      };
      const uint32_t source_location_end =
          iree_max(current_location_end,
                   loom_low_allocation_storage_assignment_pressure_extent(
                       context->descriptor_set, &source_assignment));
      if (result_location_end > source_location_end) {
        const uint32_t source_tier =
            loom_target_residency_evaluate_tier_with_direct_resource_override(
                context->residency, current_units_by_reg_class, reg_class_id,
                source_location_end);
        if (result_tier < source_tier) {
          *out_assignment = source_assignment;
          out_assignment->value_id = source_interval->value_id;
          out_assignment->start_point =
              context->unit_liveness->values[relation->source_ordinal]
                  .acquisition_start_point;
          out_assignment->end_point =
              loom_low_allocation_live_range_interval_storage_end_point(
                  source_interval);
          out_assignment->unit_count = source_interval->unit_count;
          return iree_ok_status();
        }
      }
    }
  }

  *out_assignment = (loom_low_allocation_assignment_t){
      .value_id = result_interval->value_id,
      .descriptor_reg_class_id = capacity.descriptor_reg_class_id,
      .start_point = reservation_start_point,
      .end_point = loom_low_allocation_live_range_interval_storage_end_point(
          result_interval),
      .unit_count = result_interval->unit_count,
      .location_kind = capacity.location_kind,
      .location_base = result_location_base,
      .location_count = result_interval->unit_count,
      .flags = reservation_flags,
  };
  return iree_ok_status();
}
