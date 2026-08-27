// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/loop_edge_relocation.h"

#include <string.h>

#include "loom/codegen/low/allocation/assignment_map.h"
#include "loom/codegen/low/allocation/edge_alias.h"
#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/ops/low/ops.h"

typedef struct loom_low_allocation_loop_edge_candidate_t {
  // Assignment table index of the loop-header destination.
  uint32_t destination_assignment_index;
  // Assignment table index of the selected backedge source.
  uint32_t source_assignment_index;
  // Destination assignment with the proposed source location.
  loom_low_allocation_assignment_t assignment;
} loom_low_allocation_loop_edge_candidate_t;

typedef struct loom_low_allocation_loop_edge_eviction_t {
  // Assignment table index of the value being recolored.
  uint32_t assignment_index;
  // Original assignment whose location will change.
  loom_low_allocation_assignment_t assignment;
  // Allocation interval governing alignment and target capacity.
  const loom_liveness_interval_t* interval;
  // Target capacity for the value's defining result.
  loom_low_allocation_class_capacity_t capacity;
} loom_low_allocation_loop_edge_eviction_t;

typedef struct loom_low_allocation_loop_edge_relocation_state_t {
  // Caller-provided immutable and mutable allocation state.
  const loom_low_allocation_loop_edge_relocation_context_t* context;
  // Assignment lookup over context assignments.
  loom_low_allocation_assignment_map_t assignment_map;
  // Reusable path-sensitive consumption query for the function body.
  loom_consumption_region_query_t consumption_query;
} loom_low_allocation_loop_edge_relocation_state_t;

static iree_status_t loom_low_allocation_loop_edge_relocation_consumption_query(
    void* user_data, const loom_region_t* region,
    loom_consumption_region_query_t** out_query) {
  loom_low_allocation_loop_edge_relocation_state_t* state =
      (loom_low_allocation_loop_edge_relocation_state_t*)user_data;
  if (region != state->context->body) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "loop edge relocation relation is outside the allocated body");
  }
  *out_query = &state->consumption_query;
  return iree_ok_status();
}

static bool loom_low_allocation_loop_edge_relocation_find_backedge(
    const loom_cfg_graph_t* graph, uint16_t header_index,
    const loom_cfg_edge_info_t** out_backedge) {
  *out_backedge = NULL;
  bool has_entry_edge = false;
  const loom_cfg_edge_index_span_t predecessor_edges =
      loom_cfg_graph_predecessor_edges(graph, header_index);
  for (iree_host_size_t i = 0; i < predecessor_edges.count; ++i) {
    const loom_cfg_edge_info_t* edge =
        loom_cfg_graph_edge(graph, predecessor_edges.values[i]);
    if (edge == NULL ||
        !loom_cfg_graph_block_is_reachable(graph, edge->source_block_index)) {
      continue;
    }
    if (edge->source_block_index < header_index) {
      has_entry_edge = true;
      continue;
    }
    if (*out_backedge != NULL) {
      *out_backedge = NULL;
      return false;
    }
    *out_backedge = edge;
  }
  if (!has_entry_edge || *out_backedge == NULL) {
    *out_backedge = NULL;
    return false;
  }
  const loom_op_t* terminator = (*out_backedge)->terminator;
  const loom_block_t* header = graph->blocks[header_index].block;
  if (terminator == NULL || !loom_low_br_isa(terminator) ||
      loom_low_br_dest(terminator) != header ||
      (*out_backedge)->successor_index != 0) {
    *out_backedge = NULL;
    return false;
  }
  return true;
}

static bool
loom_low_allocation_loop_edge_relocation_destination_relations_supported(
    const loom_low_placement_table_t* placement,
    loom_value_ordinal_t destination_ordinal) {
  const loom_low_placement_relation_range_t result_range =
      loom_low_placement_relation_range_for_value_ordinal(placement,
                                                          destination_ordinal);
  if (result_range.count < 2) {
    return false;
  }
  for (uint32_t i = 0; i < result_range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &placement->relations[result_range.start + i];
    if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH ||
        relation->kind != LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE ||
        relation->result_unit_offset != 0 ||
        relation->source_unit_offset != 0 || relation->unit_count != 1) {
      return false;
    }
  }
  const loom_low_placement_relation_range_t source_range =
      loom_low_placement_relation_range_for_source_value_ordinal(
          placement, destination_ordinal);
  return source_range.count == 0;
}

static const loom_low_placement_relation_t*
loom_low_allocation_loop_edge_relocation_find_relation(
    const loom_low_placement_table_t* placement,
    loom_value_ordinal_t destination_ordinal,
    const loom_op_t* backedge_terminator) {
  const loom_low_placement_relation_range_t range =
      loom_low_placement_relation_range_for_value_ordinal(placement,
                                                          destination_ordinal);
  const loom_low_placement_relation_t* result = NULL;
  for (uint32_t i = 0; i < range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &placement->relations[range.start + i];
    if (relation->op != backedge_terminator ||
        relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH) {
      continue;
    }
    if (result != NULL) {
      return NULL;
    }
    result = relation;
  }
  return result;
}

static iree_status_t loom_low_allocation_loop_edge_relocation_collect_candidate(
    loom_low_allocation_loop_edge_relocation_state_t* state,
    loom_value_id_t destination_value_id, const loom_op_t* backedge_terminator,
    loom_low_allocation_loop_edge_candidate_t* out_candidate,
    bool* out_candidate_found) {
  *out_candidate_found = false;
  loom_value_ordinal_t destination_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_assignment_map_value_ordinal_for_value(
          &state->assignment_map, destination_value_id, &destination_ordinal)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "loop header value %u is outside the allocation value domain",
        (unsigned)destination_value_id);
  }
  if (!loom_low_allocation_loop_edge_relocation_destination_relations_supported(
          state->context->placement, destination_ordinal)) {
    return iree_ok_status();
  }
  const loom_low_placement_relation_t* relation =
      loom_low_allocation_loop_edge_relocation_find_relation(
          state->context->placement, destination_ordinal, backedge_terminator);
  if (relation == NULL) {
    return iree_ok_status();
  }

  uint32_t destination_assignment_index = 0;
  const loom_low_allocation_assignment_t* destination_assignment = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_assignment_map_require_assignment_for_value(
          &state->assignment_map, destination_value_id,
          &destination_assignment_index, &destination_assignment));
  if (!loom_low_allocation_assignment_is_register_like(
          destination_assignment) ||
      destination_assignment->unit_count != 1 ||
      destination_assignment->location_count != 1 ||
      loom_low_allocation_target_constraints_fixed_value_for_value(
          state->context->target_constraints, destination_value_id) != NULL ||
      loom_low_allocation_storage_lease_state_value_has_records(
          state->context->storage_leases, state->context->liveness,
          destination_value_id)) {
    return iree_ok_status();
  }

  const loom_value_id_t source_value_id = loom_low_placement_value_id(
      state->context->placement, relation->source_ordinal);
  uint32_t source_assignment_index = 0;
  const loom_low_allocation_assignment_t* source_assignment = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_assignment_map_require_assignment_for_value(
          &state->assignment_map, source_value_id, &source_assignment_index,
          &source_assignment));
  if (!loom_low_allocation_assignment_is_register_like(source_assignment) ||
      source_assignment->unit_count != 1 ||
      source_assignment->location_count != 1 ||
      !loom_liveness_value_class_equal(destination_assignment->value_class,
                                       source_assignment->value_class) ||
      !loom_low_allocation_storage_assignment_classes_share(
          state->context->descriptor_set, destination_assignment,
          source_assignment)) {
    return iree_ok_status();
  }
  if (loom_low_allocation_storage_assignment_ranges_equal(
          state->context->descriptor_set, destination_assignment,
          source_assignment)) {
    return iree_ok_status();
  }

  const loom_liveness_interval_t* destination_interval =
      loom_liveness_interval_for_value_ordinal(state->context->liveness,
                                               destination_ordinal);
  if (destination_interval == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "loop header value has no liveness interval");
  }
  loom_low_allocation_class_capacity_t destination_capacity = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_interval_capacity(
      state->context->target_constraints, destination_interval,
      &destination_capacity));
  if (!loom_low_allocation_target_constraints_location_range_fits_capacity(
          &destination_capacity, source_assignment->location_kind,
          source_assignment->location_base,
          source_assignment->location_count)) {
    return iree_ok_status();
  }
  const uint32_t required_alignment =
      loom_low_allocation_live_range_interval_alignment(destination_interval);
  if (required_alignment == 0 ||
      source_assignment->location_base % required_alignment != 0) {
    return iree_ok_status();
  }
  const loom_low_allocation_edge_alias_context_t edge_alias_context = {
      .placement = state->context->placement,
      .consumption_query =
          loom_low_allocation_loop_edge_relocation_consumption_query,
      .user_data = state,
  };
  bool allows_overlap = false;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_edge_alias_allows_counterpart_overlap(
          &edge_alias_context, destination_interval, relation,
          source_assignment, /*destination_unit_offset=*/0,
          /*destination_unit_count=*/1, &allows_overlap));
  if (!allows_overlap) {
    return iree_ok_status();
  }

  *out_candidate = (loom_low_allocation_loop_edge_candidate_t){
      .destination_assignment_index = destination_assignment_index,
      .source_assignment_index = source_assignment_index,
      .assignment = *destination_assignment,
  };
  out_candidate->assignment.location_kind = source_assignment->location_kind;
  out_candidate->assignment.location_base = source_assignment->location_base;
  out_candidate->assignment.location_count = source_assignment->location_count;
  *out_candidate_found = true;
  return iree_ok_status();
}

static bool loom_low_allocation_loop_edge_relocation_candidates_are_disjoint(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_loop_edge_candidate_t* candidates,
    iree_host_size_t candidate_count) {
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    for (iree_host_size_t j = i + 1; j < candidate_count; ++j) {
      if (loom_low_allocation_storage_assignment_ranges_overlap(
              descriptor_set, &candidates[i].assignment,
              &candidates[j].assignment)) {
        return false;
      }
    }
  }
  return true;
}

static bool loom_low_allocation_loop_edge_relocation_candidate_target_conflicts(
    const loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_allocation_loop_edge_candidate_t* candidate) {
  const loom_low_allocation_loop_edge_relocation_context_t* context =
      state->context;
  const loom_value_id_t source_value_id =
      context->assignments[candidate->source_assignment_index].value_id;
  if (loom_low_allocation_target_constraints_fixed_value_conflicts(
          context->target_constraints, context->liveness,
          context->unit_liveness, &candidate->assignment, &source_value_id,
          /*ignored_value_count=*/1)) {
    return true;
  }
  if (loom_low_allocation_target_constraints_reserved_range_conflicts(
          context->target_constraints,
          candidate->assignment.descriptor_reg_class_id,
          candidate->assignment.location_kind,
          candidate->assignment.location_base,
          candidate->assignment.location_count)) {
    return true;
  }
  // Edge aliasing ends ordinary source liveness at the handoff, but a storage
  // lease may preserve asynchronous ownership beyond that point. The source
  // lease therefore remains a hard conflict even though the source assignment
  // and fixed-value records can be ignored as the intentional counterpart.
  return loom_low_allocation_storage_lease_state_conflicts(
      context->storage_leases, context->descriptor_set, context->liveness,
      &candidate->assignment, /*ignored_value_ids=*/NULL,
      /*ignored_value_count=*/0, LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN);
}

static bool loom_low_allocation_loop_edge_relocation_candidate_conflicts(
    const loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_allocation_loop_edge_candidate_t* candidate,
    const uint8_t* ignored_assignments) {
  const loom_low_allocation_loop_edge_relocation_context_t* context =
      state->context;
  for (iree_host_size_t i = 0; i < context->assignment_count; ++i) {
    if (ignored_assignments[i]) {
      continue;
    }
    if (loom_low_allocation_live_range_assignments_conflict(
            context->descriptor_set, context->liveness,
            context->unit_liveness->start_points,
            context->unit_liveness->end_points,
            context->unit_liveness->point_count, &candidate->assignment,
            &context->assignments[i])) {
      return true;
    }
  }
  return loom_low_allocation_loop_edge_relocation_candidate_target_conflicts(
      state, candidate);
}

static bool
loom_low_allocation_loop_edge_relocation_value_has_placement_relations(
    const loom_low_placement_table_t* placement,
    loom_value_ordinal_t value_ordinal) {
  return loom_low_placement_relation_range_for_value_ordinal(placement,
                                                             value_ordinal)
                 .count != 0 ||
         loom_low_placement_relation_range_for_source_value_ordinal(
             placement, value_ordinal)
                 .count != 0;
}

static iree_status_t loom_low_allocation_loop_edge_relocation_prepare_eviction(
    const loom_low_allocation_loop_edge_relocation_state_t* state,
    uint32_t assignment_index,
    loom_low_allocation_loop_edge_eviction_t* out_eviction,
    bool* out_supported) {
  *out_eviction = (loom_low_allocation_loop_edge_eviction_t){0};
  *out_supported = false;
  const loom_low_allocation_loop_edge_relocation_context_t* context =
      state->context;
  const loom_low_allocation_assignment_t* assignment =
      &context->assignments[assignment_index];
  if (!loom_low_allocation_assignment_is_register_like(assignment) ||
      assignment->unit_count != 1 || assignment->location_count != 1 ||
      loom_low_allocation_target_constraints_fixed_value_for_value(
          context->target_constraints, assignment->value_id) != NULL ||
      loom_low_allocation_storage_lease_state_value_has_records(
          context->storage_leases, context->liveness, assignment->value_id)) {
    return iree_ok_status();
  }
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_assignment_map_value_ordinal_for_value(
          &state->assignment_map, assignment->value_id, &value_ordinal)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "loop edge recoloring value %u is outside the allocation value domain",
        (unsigned)assignment->value_id);
  }
  if (loom_low_allocation_loop_edge_relocation_value_has_placement_relations(
          context->placement, value_ordinal)) {
    return iree_ok_status();
  }
  const loom_liveness_interval_t* interval =
      loom_liveness_interval_for_value_ordinal(context->liveness,
                                               value_ordinal);
  if (interval == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "loop edge recoloring value has no live interval");
  }
  loom_low_allocation_class_capacity_t capacity = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_interval_capacity(
      context->target_constraints, interval, &capacity));
  *out_eviction = (loom_low_allocation_loop_edge_eviction_t){
      .assignment_index = assignment_index,
      .assignment = *assignment,
      .interval = interval,
      .capacity = capacity,
  };
  *out_supported = true;
  return iree_ok_status();
}

static bool loom_low_allocation_loop_edge_relocation_eviction_location_is_legal(
    const loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_allocation_loop_edge_candidate_t* candidates,
    iree_host_size_t candidate_count,
    const loom_low_allocation_loop_edge_eviction_t* eviction,
    const loom_low_allocation_loop_edge_candidate_t* vacancy,
    const uint8_t* ignored_assignments,
    loom_low_allocation_assignment_t* out_assignment) {
  const loom_low_allocation_loop_edge_relocation_context_t* context =
      state->context;
  const loom_low_allocation_assignment_t* vacated_assignment =
      &context->assignments[vacancy->destination_assignment_index];
  if (eviction->assignment.descriptor_reg_class_id !=
          vacated_assignment->descriptor_reg_class_id ||
      eviction->assignment.location_kind != vacated_assignment->location_kind ||
      eviction->assignment.location_count !=
          vacated_assignment->location_count) {
    return false;
  }
  loom_low_allocation_assignment_t assignment = eviction->assignment;
  assignment.location_base = vacated_assignment->location_base;
  if (!loom_low_allocation_target_constraints_location_range_fits_capacity(
          &eviction->capacity, assignment.location_kind,
          assignment.location_base, assignment.location_count)) {
    return false;
  }
  const uint32_t required_alignment =
      loom_low_allocation_live_range_interval_alignment(eviction->interval);
  if (required_alignment == 0 ||
      assignment.location_base % required_alignment != 0) {
    return false;
  }
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    if (loom_low_allocation_storage_assignment_ranges_overlap(
            context->descriptor_set, &assignment, &candidates[i].assignment)) {
      return false;
    }
  }
  for (iree_host_size_t i = 0; i < context->assignment_count; ++i) {
    if (ignored_assignments[i]) {
      continue;
    }
    if (loom_low_allocation_live_range_assignments_conflict(
            context->descriptor_set, context->liveness,
            context->unit_liveness->start_points,
            context->unit_liveness->end_points,
            context->unit_liveness->point_count, &assignment,
            &context->assignments[i])) {
      return false;
    }
  }
  if (loom_low_allocation_target_constraints_fixed_value_conflicts(
          context->target_constraints, context->liveness,
          context->unit_liveness, &assignment,
          /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0) ||
      loom_low_allocation_target_constraints_reserved_range_conflicts(
          context->target_constraints, assignment.descriptor_reg_class_id,
          assignment.location_kind, assignment.location_base,
          assignment.location_count) ||
      loom_low_allocation_storage_lease_state_conflicts(
          context->storage_leases, context->descriptor_set, context->liveness,
          &assignment, /*ignored_value_ids=*/NULL,
          /*ignored_value_count=*/0,
          LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN)) {
    return false;
  }
  *out_assignment = assignment;
  return true;
}

typedef struct loom_low_allocation_loop_edge_matching_t {
  // Loop relocation state owning assignments and target constraints.
  const loom_low_allocation_loop_edge_relocation_state_t* state;
  // Complete loop-header candidate group.
  const loom_low_allocation_loop_edge_candidate_t* candidates;
  // Number of entries in candidates.
  iree_host_size_t candidate_count;
  // External values that must leave candidate target locations.
  const loom_low_allocation_loop_edge_eviction_t* evictions;
  // Number of entries in evictions.
  iree_host_size_t eviction_count;
  // Storage-color group indexed by eviction.
  const uint32_t* group_indices_by_eviction;
  // Number of distinct storage-color groups.
  iree_host_size_t group_count;
  // Candidate indices whose old locations are absent from the new layout.
  const uint32_t* vacancy_candidate_indices;
  // Number of entries in vacancy_candidate_indices.
  iree_host_size_t vacancy_count;
  // Assignments omitted while checking proposed eviction locations.
  const uint8_t* ignored_assignments;
  // Matched storage-color group for each vacancy, or UINT32_MAX.
  uint32_t* group_indices_by_vacancy;
} loom_low_allocation_loop_edge_matching_t;

static bool loom_low_allocation_loop_edge_relocation_match_eviction(
    const loom_low_allocation_loop_edge_matching_t* matching,
    uint32_t group_index, uint8_t* visited_vacancies) {
  IREE_ASSERT_LT(group_index, matching->group_count);
  for (iree_host_size_t i = 0; i < matching->vacancy_count; ++i) {
    if (visited_vacancies[i]) {
      continue;
    }
    bool group_is_legal = true;
    loom_low_allocation_assignment_t proposed_assignment = {0};
    for (iree_host_size_t j = 0; j < matching->eviction_count; ++j) {
      if (matching->group_indices_by_eviction[j] != group_index) {
        continue;
      }
      if (!loom_low_allocation_loop_edge_relocation_eviction_location_is_legal(
              matching->state, matching->candidates, matching->candidate_count,
              &matching->evictions[j],
              &matching->candidates[matching->vacancy_candidate_indices[i]],
              matching->ignored_assignments, &proposed_assignment)) {
        group_is_legal = false;
        break;
      }
    }
    if (!group_is_legal) {
      continue;
    }
    visited_vacancies[i] = 1;
    const uint32_t previous_group_index = matching->group_indices_by_vacancy[i];
    if (previous_group_index != UINT32_MAX &&
        !loom_low_allocation_loop_edge_relocation_match_eviction(
            matching, previous_group_index, visited_vacancies)) {
      continue;
    }
    matching->group_indices_by_vacancy[i] = group_index;
    return true;
  }
  return false;
}

static iree_status_t loom_low_allocation_loop_edge_relocation_try_full_group(
    loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_allocation_loop_edge_candidate_t* candidates,
    iree_host_size_t candidate_count, const uint8_t* ignored_assignments,
    bool* out_applied, iree_host_size_t* out_recolored_value_count) {
  *out_applied = false;
  *out_recolored_value_count = 0;
  const loom_low_allocation_loop_edge_relocation_context_t* context =
      state->context;
  if (context->assignment_count > UINT32_MAX) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    if (loom_low_allocation_loop_edge_relocation_candidate_target_conflicts(
            state, &candidates[i])) {
      return iree_ok_status();
    }
  }

  uint8_t* conflict_assignments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, context->assignment_count, sizeof(*conflict_assignments),
      (void**)&conflict_assignments));
  memset(conflict_assignments, 0,
         context->assignment_count * sizeof(*conflict_assignments));
  iree_host_size_t conflict_count = 0;
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    for (iree_host_size_t j = 0; j < context->assignment_count; ++j) {
      if (ignored_assignments[j] || conflict_assignments[j] ||
          !loom_low_allocation_live_range_assignments_conflict(
              context->descriptor_set, context->liveness,
              context->unit_liveness->start_points,
              context->unit_liveness->end_points,
              context->unit_liveness->point_count, &candidates[i].assignment,
              &context->assignments[j])) {
        continue;
      }
      conflict_assignments[j] = 1;
      ++conflict_count;
    }
  }

  uint32_t* vacancy_candidate_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, candidate_count, sizeof(*vacancy_candidate_indices),
      (void**)&vacancy_candidate_indices));
  iree_host_size_t vacancy_count = 0;
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    const loom_low_allocation_assignment_t* old_destination =
        &context->assignments[candidates[i].destination_assignment_index];
    bool location_remains_occupied = false;
    for (iree_host_size_t j = 0; j < candidate_count; ++j) {
      if (loom_low_allocation_storage_assignment_ranges_overlap(
              context->descriptor_set, old_destination,
              &candidates[j].assignment)) {
        location_remains_occupied = true;
        break;
      }
    }
    if (!location_remains_occupied) {
      // A physical location can host at most one eviction color. Header values
      // may share storage when their unit liveness is disjoint, so reject the
      // full recoloring path when two apparent vacancies alias. The subset
      // relocation path below can still apply moves that need no eviction.
      for (iree_host_size_t j = 0; j < vacancy_count; ++j) {
        const loom_low_allocation_assignment_t* existing_vacancy =
            &context->assignments[candidates[vacancy_candidate_indices[j]]
                                      .destination_assignment_index];
        if (loom_low_allocation_storage_assignment_ranges_overlap(
                context->descriptor_set, old_destination, existing_vacancy)) {
          return iree_ok_status();
        }
      }
      vacancy_candidate_indices[vacancy_count++] = (uint32_t)i;
    }
  }
  if (conflict_count > UINT32_MAX) {
    return iree_ok_status();
  }

  loom_low_allocation_loop_edge_eviction_t* evictions = NULL;
  if (conflict_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(context->arena, conflict_count,
                                  sizeof(*evictions), (void**)&evictions));
  }
  iree_host_size_t eviction_count = 0;
  for (iree_host_size_t i = 0; i < context->assignment_count; ++i) {
    if (!conflict_assignments[i]) {
      continue;
    }
    bool supported = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_loop_edge_relocation_prepare_eviction(
            state, (uint32_t)i, &evictions[eviction_count], &supported));
    if (!supported) {
      return iree_ok_status();
    }
    ++eviction_count;
  }
  IREE_ASSERT_EQ(eviction_count, conflict_count);

  uint32_t* group_indices_by_eviction = NULL;
  if (conflict_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->arena, conflict_count, sizeof(*group_indices_by_eviction),
        (void**)&group_indices_by_eviction));
  }
  iree_host_size_t group_count = 0;
  for (iree_host_size_t i = 0; i < conflict_count; ++i) {
    uint32_t group_index = UINT32_MAX;
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (evictions[i].assignment.descriptor_reg_class_id ==
              evictions[j].assignment.descriptor_reg_class_id &&
          loom_low_allocation_storage_assignment_ranges_equal(
              context->descriptor_set, &evictions[i].assignment,
              &evictions[j].assignment)) {
        group_index = group_indices_by_eviction[j];
        break;
      }
    }
    if (group_index == UINT32_MAX) {
      group_index = (uint32_t)group_count++;
    }
    group_indices_by_eviction[i] = group_index;
  }
  if (group_count > vacancy_count) {
    return iree_ok_status();
  }

  uint8_t* eviction_ignored_assignments = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(context->arena, context->assignment_count,
                                sizeof(*eviction_ignored_assignments),
                                (void**)&eviction_ignored_assignments));
  memset(eviction_ignored_assignments, 0,
         context->assignment_count * sizeof(*eviction_ignored_assignments));
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    eviction_ignored_assignments[candidates[i].destination_assignment_index] =
        1;
  }
  for (iree_host_size_t i = 0; i < conflict_count; ++i) {
    eviction_ignored_assignments[evictions[i].assignment_index] = 1;
  }

  uint32_t* group_indices_by_vacancy = NULL;
  loom_low_allocation_assignment_t* assignments_by_eviction = NULL;
  uint8_t* visited_vacancies = NULL;
  if (conflict_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->arena, vacancy_count, sizeof(*group_indices_by_vacancy),
        (void**)&group_indices_by_vacancy));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->arena, conflict_count, sizeof(*assignments_by_eviction),
        (void**)&assignments_by_eviction));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->arena, vacancy_count, sizeof(*visited_vacancies),
        (void**)&visited_vacancies));
    for (iree_host_size_t i = 0; i < vacancy_count; ++i) {
      group_indices_by_vacancy[i] = UINT32_MAX;
    }
    const loom_low_allocation_loop_edge_matching_t matching = {
        .state = state,
        .candidates = candidates,
        .candidate_count = candidate_count,
        .evictions = evictions,
        .eviction_count = eviction_count,
        .group_indices_by_eviction = group_indices_by_eviction,
        .group_count = group_count,
        .vacancy_candidate_indices = vacancy_candidate_indices,
        .vacancy_count = vacancy_count,
        .ignored_assignments = eviction_ignored_assignments,
        .group_indices_by_vacancy = group_indices_by_vacancy,
    };
    for (uint32_t i = 0; i < (uint32_t)group_count; ++i) {
      memset(visited_vacancies, 0, vacancy_count * sizeof(*visited_vacancies));
      if (!loom_low_allocation_loop_edge_relocation_match_eviction(
              &matching, i, visited_vacancies)) {
        return iree_ok_status();
      }
    }
    // Materialize assignments only after every augmenting path has settled.
    // Writing locations while searching would leave stale locations behind
    // when a recursive match backtracks to a prior vacancy.
    for (iree_host_size_t i = 0; i < vacancy_count; ++i) {
      const uint32_t group_index = group_indices_by_vacancy[i];
      if (group_index == UINT32_MAX) {
        continue;
      }
      for (iree_host_size_t j = 0; j < eviction_count; ++j) {
        if (group_indices_by_eviction[j] != group_index) {
          continue;
        }
        const bool assignment_is_legal =
            loom_low_allocation_loop_edge_relocation_eviction_location_is_legal(
                state, candidates, candidate_count, &evictions[j],
                &candidates[vacancy_candidate_indices[i]],
                eviction_ignored_assignments, &assignments_by_eviction[j]);
        if (!assignment_is_legal) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "matched loop-edge eviction location is no longer legal");
        }
      }
    }
  }

  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    loom_low_allocation_assignment_t* destination =
        &context->assignments[candidates[i].destination_assignment_index];
    destination->location_kind = candidates[i].assignment.location_kind;
    destination->location_base = candidates[i].assignment.location_base;
    destination->location_count = candidates[i].assignment.location_count;
  }
  for (iree_host_size_t i = 0; i < conflict_count; ++i) {
    loom_low_allocation_assignment_t* assignment =
        &context->assignments[evictions[i].assignment_index];
    assignment->location_kind = assignments_by_eviction[i].location_kind;
    assignment->location_base = assignments_by_eviction[i].location_base;
    assignment->location_count = assignments_by_eviction[i].location_count;
  }
  *out_applied = true;
  *out_recolored_value_count = conflict_count;
  return iree_ok_status();
}

static bool
loom_low_allocation_loop_edge_relocation_candidate_depends_on_destination(
    const loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_allocation_loop_edge_candidate_t* candidate,
    uint32_t destination_assignment_index) {
  const loom_low_allocation_loop_edge_relocation_context_t* context =
      state->context;
  return loom_low_allocation_live_range_assignments_conflict(
      context->descriptor_set, context->liveness,
      context->unit_liveness->start_points, context->unit_liveness->end_points,
      context->unit_liveness->point_count, &candidate->assignment,
      &context->assignments[destination_assignment_index]);
}

static iree_status_t loom_low_allocation_loop_edge_relocation_try_header(
    loom_low_allocation_loop_edge_relocation_state_t* state,
    uint16_t header_index, iree_host_size_t* out_relocated_value_count,
    iree_host_size_t* out_recolored_value_count) {
  *out_relocated_value_count = 0;
  *out_recolored_value_count = 0;
  const loom_cfg_edge_info_t* backedge = NULL;
  if (!loom_low_allocation_loop_edge_relocation_find_backedge(
          state->context->cfg_graph, header_index, &backedge)) {
    return iree_ok_status();
  }
  const loom_block_t* header =
      state->context->cfg_graph->blocks[header_index].block;
  if (header == NULL || header->arg_count == 0) {
    return iree_ok_status();
  }

  loom_low_allocation_loop_edge_candidate_t* candidates = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->context->arena, header->arg_count,
                                sizeof(*candidates), (void**)&candidates));
  iree_host_size_t candidate_count = 0;
  for (uint16_t i = 0; i < header->arg_count; ++i) {
    bool candidate_found = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_loop_edge_relocation_collect_candidate(
            state, loom_block_arg_id(header, i), backedge->terminator,
            &candidates[candidate_count], &candidate_found));
    if (candidate_found) {
      ++candidate_count;
    }
  }
  if (candidate_count == 0 ||
      !loom_low_allocation_loop_edge_relocation_candidates_are_disjoint(
          state->context->descriptor_set, candidates, candidate_count)) {
    return iree_ok_status();
  }

  uint8_t* ignored_assignments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->context->arena, state->context->assignment_count,
      sizeof(*ignored_assignments), (void**)&ignored_assignments));
  memset(ignored_assignments, 0,
         state->context->assignment_count * sizeof(*ignored_assignments));
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    ignored_assignments[candidates[i].destination_assignment_index] = 1;
    ignored_assignments[candidates[i].source_assignment_index] = 1;
  }

  bool full_group_applied = false;
  IREE_RETURN_IF_ERROR(loom_low_allocation_loop_edge_relocation_try_full_group(
      state, candidates, candidate_count, ignored_assignments,
      &full_group_applied, out_recolored_value_count));
  if (full_group_applied) {
    *out_relocated_value_count = candidate_count;
    return iree_ok_status();
  }

  uint8_t* candidate_enabled = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->context->arena, candidate_count, sizeof(*candidate_enabled),
      (void**)&candidate_enabled));
  memset(candidate_enabled, 1, candidate_count * sizeof(*candidate_enabled));
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    if (loom_low_allocation_loop_edge_relocation_candidate_conflicts(
            state, &candidates[i], ignored_assignments)) {
      candidate_enabled[i] = 0;
    }
  }

  // An enabled candidate may only ignore the old destination storage of
  // another candidate when that destination will itself move. Repeatedly
  // remove candidates that depend on a rejected destination until the
  // selected subset is closed under that requirement.
  bool selection_changed = false;
  do {
    selection_changed = false;
    for (iree_host_size_t i = 0; i < candidate_count; ++i) {
      if (!candidate_enabled[i]) {
        continue;
      }
      for (iree_host_size_t j = 0; j < candidate_count; ++j) {
        if (candidate_enabled[j] ||
            !loom_low_allocation_loop_edge_relocation_candidate_depends_on_destination(
                state, &candidates[i],
                candidates[j].destination_assignment_index)) {
          continue;
        }
        candidate_enabled[i] = 0;
        selection_changed = true;
        break;
      }
    }
  } while (selection_changed);

  iree_host_size_t relocated_value_count = 0;
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    if (!candidate_enabled[i]) {
      continue;
    }
    state->context->assignments[candidates[i].destination_assignment_index]
        .location_kind = candidates[i].assignment.location_kind;
    state->context->assignments[candidates[i].destination_assignment_index]
        .location_base = candidates[i].assignment.location_base;
    state->context->assignments[candidates[i].destination_assignment_index]
        .location_count = candidates[i].assignment.location_count;
    ++relocated_value_count;
  }
  *out_relocated_value_count = relocated_value_count;
  return iree_ok_status();
}

iree_status_t loom_low_allocation_loop_edge_relocate(
    const loom_low_allocation_loop_edge_relocation_context_t* context,
    loom_low_allocation_loop_edge_relocation_result_t* out_result) {
  *out_result = (loom_low_allocation_loop_edge_relocation_result_t){0};
  // Straight-line functions use the function model's lightweight graph
  // identity without materializing adjacency storage.
  if (context->assignment_count == 0 || context->cfg_graph->block_count < 2 ||
      context->cfg_graph->blocks == NULL || context->cfg_graph->malformed) {
    return iree_ok_status();
  }
  loom_low_allocation_loop_edge_relocation_state_t state = {
      .context = context,
      .assignment_map =
          {
              .module = context->module,
              .liveness = context->liveness,
              .assignments = context->assignments,
              .assignment_count = context->assignment_count,
              .assignment_indices_by_value_ordinal =
                  context->assignment_indices_by_value_ordinal,
          },
  };
  loom_consumption_region_query_initialize_with_cfg_graph(
      context->module, context->body, context->cfg_graph, context->arena,
      &state.consumption_query);

  for (uint16_t header_index = 0;
       header_index < context->cfg_graph->block_count; ++header_index) {
    if (!loom_cfg_graph_block_is_reachable(context->cfg_graph, header_index)) {
      continue;
    }
    iree_host_size_t relocated_value_count = 0;
    iree_host_size_t recolored_value_count = 0;
    IREE_RETURN_IF_ERROR(loom_low_allocation_loop_edge_relocation_try_header(
        &state, header_index, &relocated_value_count, &recolored_value_count));
    if (relocated_value_count == 0) {
      continue;
    }
    out_result->relocated_value_count += relocated_value_count;
    out_result->recolored_value_count += recolored_value_count;
    ++out_result->relocated_header_count;
  }
  if (out_result->relocated_value_count != 0) {
    loom_low_allocation_target_constraints_rebuild_assignment_location_ends(
        context->target_constraints, context->assignments,
        context->assignment_count);
  }
  return iree_ok_status();
}
