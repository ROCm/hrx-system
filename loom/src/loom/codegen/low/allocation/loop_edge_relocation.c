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
#include "loom/codegen/low/allocation/relocation_group.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/allocation/write_interference.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/util/adaptive_sort.h"

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
  // Header-local scratch reset after each relocation attempt.
  iree_arena_allocator_t* scratch_arena;
  // Assignment lookup over context assignments.
  loom_low_allocation_assignment_map_t assignment_map;
  // Reusable path-sensitive consumption query for the function body.
  loom_consumption_region_query_t consumption_query;
  // Coalesced non-edge components retained before any header relocation.
  loom_low_allocation_relocation_groups_t groups;
  // Complete simultaneous location proposal checked before committing writes.
  loom_low_allocation_write_proposal_t write_proposal;
} loom_low_allocation_loop_edge_relocation_state_t;

static loom_low_allocation_storage_origin_t
loom_low_allocation_loop_edge_write_origin(
    const loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_allocation_assignment_t* assignment) {
  const loom_value_ordinal_t value_ordinal =
      loom_module_value_ordinal_scratch_lookup(state->context->module,
                                               assignment->value_id);
  return loom_low_allocation_write_interference_storage_origin_for_value_ordinal(
      state->context->unit_liveness->write_interference, value_ordinal);
}

static bool loom_low_allocation_loop_edge_write_conflicts(
    loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_allocation_loop_edge_candidate_t* candidates,
    iree_host_size_t candidate_count, const uint8_t* enabled,
    const loom_low_allocation_assignment_t* evictions,
    iree_host_size_t eviction_count) {
  if (state->write_proposal.bases == NULL) {
    return false;
  }
  loom_low_allocation_write_interference_t* interference =
      state->context->unit_liveness->write_interference;
  const loom_low_allocation_write_assignment_map_t write_assignments =
      loom_low_allocation_write_assignment_map_from_value_map(
          &state->assignment_map);
  loom_low_allocation_write_proposal_reset(&state->write_proposal);
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    if (enabled != NULL && !enabled[i]) {
      continue;
    }
    const uint32_t destination = candidates[i].destination_assignment_index;
    const int64_t offset =
        (int64_t)candidates[i].assignment.location_base -
        state->context->assignments[destination].location_base;
    uint32_t member = destination;
    do {
      const loom_low_allocation_assignment_t* assignment =
          &state->context->assignments[member];
      loom_low_allocation_write_proposal_add(
          interference,
          loom_low_allocation_loop_edge_write_origin(state, assignment),
          assignment, (uint32_t)((int64_t)assignment->location_base + offset),
          &state->write_proposal);
      member = state->groups.next_members[member];
    } while (member != destination);
  }
  for (iree_host_size_t i = 0; i < eviction_count; ++i) {
    loom_low_allocation_write_proposal_add(
        interference,
        loom_low_allocation_loop_edge_write_origin(state, &evictions[i]),
        &evictions[i], evictions[i].location_base, &state->write_proposal);
  }
  return loom_low_allocation_write_proposal_conflicts(
      interference, &write_assignments, &state->write_proposal);
}

// Original storage color of one eviction assignment.
typedef struct loom_low_allocation_loop_edge_eviction_color_t {
  // Target-visible storage kind containing the assignment.
  loom_low_allocation_location_kind_t location_kind;
  // Exact descriptor register class required by a matching vacancy.
  uint32_t descriptor_reg_class_id;
  // Base location within the register class.
  uint32_t location;
  // Eviction index owning the color.
  uint32_t owner_index;
} loom_low_allocation_loop_edge_eviction_color_t;

// Contiguous equal-color range in the sorted eviction color table.
typedef struct loom_low_allocation_loop_edge_eviction_group_t {
  // First entry in the sorted eviction color table.
  uint32_t color_start;
  // Number of equal-color eviction entries.
  uint32_t color_count;
} loom_low_allocation_loop_edge_eviction_group_t;

static bool loom_low_allocation_loop_edge_eviction_color_less(
    const loom_low_allocation_loop_edge_eviction_color_t* lhs,
    const loom_low_allocation_loop_edge_eviction_color_t* rhs) {
  if (lhs->location_kind != rhs->location_kind) {
    return lhs->location_kind < rhs->location_kind;
  }
  if (lhs->descriptor_reg_class_id != rhs->descriptor_reg_class_id) {
    return lhs->descriptor_reg_class_id < rhs->descriptor_reg_class_id;
  }
  if (lhs->location != rhs->location) {
    return lhs->location < rhs->location;
  }
  return lhs->owner_index < rhs->owner_index;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_low_allocation_loop_edge_eviction_color_sort,
                          loom_low_allocation_loop_edge_eviction_color_t,
                          loom_low_allocation_loop_edge_eviction_color_less)

static int loom_low_allocation_loop_edge_eviction_color_key_compare(
    const loom_low_allocation_loop_edge_eviction_color_t* lhs,
    const loom_low_allocation_loop_edge_eviction_color_t* rhs) {
  if (lhs->location_kind != rhs->location_kind) {
    return lhs->location_kind < rhs->location_kind ? -1 : 1;
  }
  if (lhs->descriptor_reg_class_id != rhs->descriptor_reg_class_id) {
    return lhs->descriptor_reg_class_id < rhs->descriptor_reg_class_id ? -1 : 1;
  }
  if (lhs->location != rhs->location) {
    return lhs->location < rhs->location ? -1 : 1;
  }
  return 0;
}

typedef struct loom_low_allocation_loop_edge_query_set_t {
  // Allocation state supplying current assignments to queries.
  const loom_low_allocation_loop_edge_relocation_context_t* context;
  // Assignments intersecting a queried physical unit or linear envelope.
  uint32_t* assignment_indices;
  // Number of entries in |assignment_indices|.
  iree_host_size_t assignment_count;
} loom_low_allocation_loop_edge_query_set_t;

// Coarse span covering every queried range in one linear storage namespace.
typedef struct loom_low_allocation_loop_edge_query_envelope_t {
  // Target-visible storage kind containing the span.
  loom_low_allocation_location_kind_t location_kind;
  // Descriptor-defined linear storage namespace containing the span.
  uint32_t storage_key;
  // Smallest queried location in the namespace.
  uint32_t minimum_location;
  // One-past-largest queried location in the namespace.
  uint64_t exclusive_maximum_location;
} loom_low_allocation_loop_edge_query_envelope_t;

// Unique assignment owners overlapping one storage assignment.
typedef struct loom_low_allocation_loop_edge_assignment_query_t {
  // Header-local relevant assignment set being queried.
  const loom_low_allocation_loop_edge_query_set_t* set;
  // Descriptor set defining storage overlap.
  const loom_low_descriptor_set_t* descriptor_set;
  // Target or vacated-destination storage queried by the caller.
  const loom_low_allocation_assignment_t* assignment;
  // Current position in the relevant owner list.
  iree_host_size_t position;
} loom_low_allocation_loop_edge_assignment_query_t;

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
    loom_value_ordinal_t destination_ordinal, uint32_t unit_count) {
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
        relation->source_unit_offset != 0 ||
        relation->unit_count != unit_count) {
      return false;
    }
  }
  return true;
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

static loom_low_allocation_assignment_t
loom_low_allocation_loop_edge_relocation_member_assignment(
    const loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_allocation_loop_edge_candidate_t* candidate,
    uint32_t member_index) {
  const loom_low_allocation_assignment_t* destination =
      &state->context->assignments[candidate->destination_assignment_index];
  loom_low_allocation_assignment_t member =
      state->context->assignments[member_index];
  member.location_base = candidate->assignment.location_base +
                         (member.location_base - destination->location_base);
  return member;
}

static bool loom_low_allocation_loop_edge_relocation_hard_relation_supported(
    const loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_placement_relation_t* relation) {
  if (!iree_any_bit_set(relation->flags,
                        LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD)) {
    return true;
  }
  if (!loom_low_placement_relation_can_alias(relation) ||
      loom_low_placement_cause_is_edge(relation->cause)) {
    return false;
  }
  const uint32_t result_index =
      state->context
          ->assignment_indices_by_value_ordinal[relation->result_ordinal];
  const uint32_t source_index =
      state->context
          ->assignment_indices_by_value_ordinal[relation->source_ordinal];
  return state->groups.representatives[result_index] ==
         state->groups.representatives[source_index];
}

static iree_status_t loom_low_allocation_loop_edge_relocation_group_supported(
    loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_allocation_loop_edge_candidate_t* candidate,
    bool* out_supported) {
  *out_supported = false;
  const loom_low_allocation_loop_edge_relocation_context_t* context =
      state->context;
  const loom_low_allocation_assignment_t* destination =
      &context->assignments[candidate->destination_assignment_index];
  const loom_low_allocation_assignment_t* source =
      &context->assignments[candidate->source_assignment_index];
  uint32_t member_index = candidate->destination_assignment_index;
  do {
    const loom_low_allocation_assignment_t* member =
        &context->assignments[member_index];
    // The backedge supplies storage for the header's window. An alias outside
    // that window would require additional storage with no handoff proof.
    if (member->location_base < destination->location_base ||
        (uint64_t)member->location_base + member->location_count >
            (uint64_t)destination->location_base +
                destination->location_count ||
        loom_low_allocation_target_constraints_fixed_value_for_value(
            context->target_constraints, member->value_id) != NULL ||
        loom_low_allocation_storage_lease_state_value_has_records(
            context->storage_leases, context->liveness, member->value_id)) {
      return iree_ok_status();
    }
    const loom_value_ordinal_t ordinal =
        loom_module_value_ordinal_scratch_lookup(context->module,
                                                 member->value_id);
    const loom_low_placement_relation_range_t result_range =
        loom_low_placement_relation_range_for_value_ordinal(context->placement,
                                                            ordinal);
    const loom_low_placement_relation_range_t source_range =
        loom_low_placement_relation_range_for_source_value_ordinal(
            context->placement, ordinal);
    for (uint32_t i = 0; i < result_range.count; ++i) {
      if (!loom_low_allocation_loop_edge_relocation_hard_relation_supported(
              state, &context->placement->relations[result_range.start + i])) {
        return iree_ok_status();
      }
    }
    for (uint32_t i = 0; i < source_range.count; ++i) {
      const uint32_t relation_index =
          context->placement
              ->relation_indices_by_source_ordinal[source_range.start + i];
      if (!loom_low_allocation_loop_edge_relocation_hard_relation_supported(
              state, &context->placement->relations[relation_index])) {
        return iree_ok_status();
      }
    }
    const loom_low_allocation_assignment_t proposed =
        loom_low_allocation_loop_edge_relocation_member_assignment(
            state, candidate, member_index);
    const loom_liveness_interval_t* interval =
        loom_liveness_interval_for_value_ordinal(context->liveness, ordinal);
    loom_low_allocation_class_capacity_t capacity = {0};
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_target_constraints_interval_capacity(
            context->target_constraints, context->liveness, context->placement,
            interval, &capacity));
    if (!loom_low_allocation_target_constraints_location_range_fits_capacity(
            context->descriptor_set, &capacity, proposed.location_kind,
            proposed.location_base, proposed.location_count) ||
        proposed.location_base %
                loom_low_allocation_live_range_interval_alignment(
                    context->descriptor_set, context->liveness,
                    context->placement->operand_constraints_by_interval,
                    interval) !=
            0) {
      return iree_ok_status();
    }
    // The header's edge proof covers its own uses. A coalesced slice can
    // outlive that header use, so its old payload must independently die
    // before the backedge source replaces it.
    if (member_index != candidate->destination_assignment_index &&
        loom_low_allocation_live_range_assignments_conflict(
            context->descriptor_set,
            context->unit_liveness->storage_segments.entries,
            context->unit_liveness->start_points,
            context->unit_liveness->end_points,
            context->unit_liveness->point_count, &proposed, source)) {
      return iree_ok_status();
    }
    member_index = state->groups.next_members[member_index];
  } while (member_index != candidate->destination_assignment_index);
  *out_supported = true;
  return iree_ok_status();
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
  if (!loom_low_allocation_loop_edge_relocation_destination_relations_supported(
          state->context->placement, destination_ordinal,
          destination_assignment->unit_count)) {
    return iree_ok_status();
  }
  if (!loom_low_allocation_assignment_is_register_like(
          destination_assignment) ||
      destination_assignment->unit_count == 0 ||
      destination_assignment->location_count !=
          destination_assignment->unit_count) {
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
      source_assignment->unit_count != destination_assignment->unit_count ||
      source_assignment->location_count != destination_assignment->unit_count ||
      destination_assignment->descriptor_reg_class_id !=
          source_assignment->descriptor_reg_class_id ||
      !loom_low_allocation_storage_assignment_classes_share(
          state->context->descriptor_set, destination_assignment,
          source_assignment)) {
    return iree_ok_status();
  }
  if (loom_low_allocation_storage_assignment_locations_correspond(
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
  const loom_low_allocation_edge_alias_context_t edge_alias_context = {
      .placement = state->context->placement,
      .liveness = state->context->liveness,
      .unit_liveness = state->context->unit_liveness,
      .consumption_query =
          loom_low_allocation_loop_edge_relocation_consumption_query,
      .user_data = state,
  };
  bool allows_overlap = false;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_edge_alias_allows_counterpart_overlap(
          &edge_alias_context, destination_interval, relation,
          source_assignment, /*destination_unit_offset=*/0,
          destination_assignment->unit_count, &allows_overlap));
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
  return loom_low_allocation_loop_edge_relocation_group_supported(
      state, out_candidate, out_candidate_found);
}

static bool loom_low_allocation_loop_edge_relocation_candidates_are_disjoint(
    const loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_allocation_loop_edge_candidate_t* candidates,
    iree_host_size_t candidate_count, uint8_t* component_markers) {
  const loom_low_allocation_loop_edge_relocation_context_t* context =
      state->context;

  // Each coalesced component can receive only one translation. Marking the
  // dense representative index bounds this check independently of the number
  // of header candidates.
  iree_host_size_t marked_candidate_count = 0;
  for (; marked_candidate_count < candidate_count; ++marked_candidate_count) {
    const uint32_t representative =
        state->groups.representatives[candidates[marked_candidate_count]
                                          .destination_assignment_index];
    if (component_markers[representative]) {
      break;
    }
    component_markers[representative] = 1;
  }
  for (iree_host_size_t i = 0; i < marked_candidate_count; ++i) {
    const uint32_t representative =
        state->groups
            .representatives[candidates[i].destination_assignment_index];
    component_markers[representative] = 0;
  }
  if (marked_candidate_count != candidate_count) {
    return false;
  }

  // Candidate count is bounded by the loop header's argument count. Compare
  // their ranges directly.
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (loom_low_allocation_storage_assignment_ranges_overlap(
              context->descriptor_set, &candidates[i].assignment,
              &candidates[j].assignment)) {
        return false;
      }
    }
  }
  return true;
}

static bool loom_low_allocation_loop_edge_candidates_overlap_assignment(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_loop_edge_candidate_t* candidates,
    iree_host_size_t candidate_count,
    const loom_low_allocation_assignment_t* assignment) {
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    if (loom_low_allocation_storage_assignment_ranges_overlap(
            descriptor_set, &candidates[i].assignment, assignment)) {
      return true;
    }
  }
  return false;
}

static void loom_low_allocation_loop_edge_include_query_range(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignment,
    uint8_t* queried_physical_units,
    loom_low_allocation_loop_edge_query_envelope_t* envelopes,
    iree_host_size_t* inout_envelope_count) {
  if (loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, assignment)) {
    const uint32_t unit_count =
        loom_low_allocation_storage_assignment_atomic_unit_count(descriptor_set,
                                                                 assignment);
    for (uint32_t i = 0; i < unit_count; ++i) {
      uint32_t unused_storage_key = 0;
      uint32_t physical_unit = 0;
      loom_low_allocation_storage_assignment_atomic_unit(
          descriptor_set, assignment, i, &unused_storage_key, &physical_unit);
      queried_physical_units[physical_unit] = 1;
    }
    return;
  }

  const uint32_t storage_key = loom_low_reg_class_storage_key(
      descriptor_set, assignment->descriptor_reg_class_id);
  const uint64_t exclusive_maximum_location =
      (uint64_t)assignment->location_base + assignment->location_count;
  for (iree_host_size_t i = 0; i < *inout_envelope_count; ++i) {
    loom_low_allocation_loop_edge_query_envelope_t* envelope = &envelopes[i];
    if (envelope->location_kind != assignment->location_kind ||
        envelope->storage_key != storage_key) {
      continue;
    }
    envelope->minimum_location =
        iree_min(envelope->minimum_location, assignment->location_base);
    envelope->exclusive_maximum_location = iree_max(
        envelope->exclusive_maximum_location, exclusive_maximum_location);
    return;
  }
  envelopes[(*inout_envelope_count)++] =
      (loom_low_allocation_loop_edge_query_envelope_t){
          .location_kind = assignment->location_kind,
          .storage_key = storage_key,
          .minimum_location = assignment->location_base,
          .exclusive_maximum_location = exclusive_maximum_location,
      };
}

// Retains assignments that may be reachable by one header's queries: targets
// and the destinations they may vacate. Candidate count is bounded by the loop
// header's argument count. Descriptor-owned physical-unit ordinals and coarse
// linear-storage envelopes reject unrelated assignments in one table pass.
static iree_status_t
loom_low_allocation_loop_edge_relocation_query_set_initialize(
    const loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_allocation_loop_edge_candidate_t* candidates,
    iree_host_size_t candidate_count, const uint8_t* ignored_assignments,
    loom_low_allocation_loop_edge_query_set_t* out_set) {
  *out_set = (loom_low_allocation_loop_edge_query_set_t){0};
  const loom_low_allocation_loop_edge_relocation_context_t* context =
      state->context;
  const loom_low_descriptor_set_t* descriptor_set = context->descriptor_set;

  uint8_t* queried_physical_units = NULL;
  if (descriptor_set->physical_register_unit_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, descriptor_set->physical_register_unit_count,
        sizeof(*queried_physical_units), (void**)&queried_physical_units));
    memset(queried_physical_units, 0,
           descriptor_set->physical_register_unit_count *
               sizeof(*queried_physical_units));
  }
  loom_low_allocation_loop_edge_query_envelope_t* envelopes = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->scratch_arena, candidate_count * 2,
                                sizeof(*envelopes), (void**)&envelopes));
  iree_host_size_t envelope_count = 0;
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    loom_low_allocation_loop_edge_include_query_range(
        descriptor_set, &candidates[i].assignment, queried_physical_units,
        envelopes, &envelope_count);
    loom_low_allocation_loop_edge_include_query_range(
        descriptor_set,
        &context->assignments[candidates[i].destination_assignment_index],
        queried_physical_units, envelopes, &envelope_count);
  }

  uint32_t* assignment_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, context->assignment_count,
      sizeof(*assignment_indices), (void**)&assignment_indices));
  iree_host_size_t assignment_count = 0;
  for (iree_host_size_t i = 0; i < context->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &context->assignments[i];
    if (ignored_assignments[i] ||
        !loom_low_allocation_assignment_is_register_like(assignment)) {
      continue;
    }
    bool is_relevant = false;
    if (loom_low_allocation_storage_assignment_uses_explicit_physical_register(
            descriptor_set, assignment)) {
      const uint32_t unit_count =
          loom_low_allocation_storage_assignment_atomic_unit_count(
              descriptor_set, assignment);
      for (uint32_t j = 0; j < unit_count; ++j) {
        uint32_t unused_storage_key = 0;
        uint32_t physical_unit = 0;
        loom_low_allocation_storage_assignment_atomic_unit(
            descriptor_set, assignment, j, &unused_storage_key, &physical_unit);
        if (queried_physical_units[physical_unit]) {
          is_relevant = true;
          break;
        }
      }
    } else {
      const uint32_t storage_key = loom_low_reg_class_storage_key(
          descriptor_set, assignment->descriptor_reg_class_id);
      const uint64_t assignment_begin = assignment->location_base;
      const uint64_t assignment_end =
          assignment_begin + assignment->location_count;
      for (iree_host_size_t j = 0; j < envelope_count; ++j) {
        const loom_low_allocation_loop_edge_query_envelope_t* envelope =
            &envelopes[j];
        if (assignment->location_kind == envelope->location_kind &&
            storage_key == envelope->storage_key &&
            assignment_begin < envelope->exclusive_maximum_location &&
            envelope->minimum_location < assignment_end) {
          is_relevant = true;
          break;
        }
      }
    }
    if (is_relevant) {
      assignment_indices[assignment_count++] = (uint32_t)i;
    }
  }

  *out_set = (loom_low_allocation_loop_edge_query_set_t){
      .context = context,
      .assignment_indices = assignment_indices,
      .assignment_count = assignment_count,
  };
  return iree_ok_status();
}

static void loom_low_allocation_loop_edge_assignment_query_initialize(
    const loom_low_allocation_loop_edge_query_set_t* set,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignment,
    loom_low_allocation_loop_edge_assignment_query_t* out_query) {
  *out_query = (loom_low_allocation_loop_edge_assignment_query_t){
      .set = set,
      .descriptor_set = descriptor_set,
      .assignment = assignment,
  };
}

static bool loom_low_allocation_loop_edge_assignment_query_next(
    loom_low_allocation_loop_edge_assignment_query_t* query,
    uint32_t* out_assignment_index) {
  while (query->position < query->set->assignment_count) {
    const uint32_t assignment_index =
        query->set->assignment_indices[query->position++];
    if (loom_low_allocation_storage_assignment_ranges_overlap(
            query->descriptor_set, query->assignment,
            &query->set->context->assignments[assignment_index])) {
      *out_assignment_index = assignment_index;
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_loop_edge_relocation_candidate_target_conflicts(
    const loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_allocation_loop_edge_candidate_t* candidate) {
  const loom_low_allocation_loop_edge_relocation_context_t* context =
      state->context;
  const loom_value_id_t source_value_id =
      context->assignments[candidate->source_assignment_index].value_id;
  uint32_t member_index = candidate->destination_assignment_index;
  do {
    const loom_low_allocation_assignment_t assignment =
        loom_low_allocation_loop_edge_relocation_member_assignment(
            state, candidate, member_index);
    if (loom_low_allocation_target_constraints_fixed_storage_conflicts(
            context->target_constraints, context->unit_liveness, &assignment,
            &source_value_id,
            /*ignored_value_count=*/1) ||
        loom_low_allocation_target_constraints_reserved_range_conflicts(
            context->target_constraints, assignment.descriptor_reg_class_id,
            assignment.location_kind, assignment.location_base,
            assignment.location_count)) {
      return true;
    }
    // Semantic handoff does not release asynchronous ownership. Keep every
    // lease, including the counterpart's, and intersect its full lifetime
    // with each member's retained storage segments.
    if (loom_low_allocation_storage_lease_state_conflicts(
            context->storage_leases, context->descriptor_set, context->liveness,
            &assignment, /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0,
            LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN)) {
      return true;
    }
    member_index = state->groups.next_members[member_index];
  } while (member_index != candidate->destination_assignment_index);
  return false;
}

static bool loom_low_allocation_loop_edge_relocation_assignment_conflicts(
    const loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_allocation_loop_edge_candidate_t* candidate,
    const loom_low_allocation_assignment_t* assignment) {
  const loom_low_allocation_loop_edge_relocation_context_t* context =
      state->context;
  uint32_t member_index = candidate->destination_assignment_index;
  do {
    const loom_low_allocation_assignment_t member =
        loom_low_allocation_loop_edge_relocation_member_assignment(
            state, candidate, member_index);
    if (loom_low_allocation_live_range_assignments_conflict(
            context->descriptor_set,
            context->unit_liveness->storage_segments.entries,
            context->unit_liveness->start_points,
            context->unit_liveness->end_points,
            context->unit_liveness->point_count, &member, assignment)) {
      return true;
    }
    member_index = state->groups.next_members[member_index];
  } while (member_index != candidate->destination_assignment_index);
  return false;
}

static bool loom_low_allocation_loop_edge_relocation_candidate_has_conflict(
    const loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_allocation_loop_edge_candidate_t* candidate,
    const loom_low_allocation_loop_edge_query_set_t* query_set) {
  const loom_low_allocation_loop_edge_relocation_context_t* context =
      state->context;
  loom_low_allocation_loop_edge_assignment_query_t query;
  loom_low_allocation_loop_edge_assignment_query_initialize(
      query_set, context->descriptor_set, &candidate->assignment, &query);
  uint32_t existing_assignment_index = 0;
  while (loom_low_allocation_loop_edge_assignment_query_next(
      &query, &existing_assignment_index)) {
    if (loom_low_allocation_loop_edge_relocation_assignment_conflicts(
            state, candidate,
            &context->assignments[existing_assignment_index])) {
      return true;
    }
  }
  return false;
}

static void
loom_low_allocation_loop_edge_relocation_collect_conflict_assignments(
    const loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_allocation_loop_edge_candidate_t* candidates,
    iree_host_size_t candidate_count,
    const loom_low_allocation_loop_edge_query_set_t* query_set,
    uint8_t* conflict_assignments, iree_host_size_t* out_conflict_count) {
  *out_conflict_count = 0;
  const loom_low_allocation_loop_edge_relocation_context_t* context =
      state->context;
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    loom_low_allocation_loop_edge_assignment_query_t query;
    loom_low_allocation_loop_edge_assignment_query_initialize(
        query_set, context->descriptor_set, &candidates[i].assignment, &query);
    uint32_t existing_assignment_index = 0;
    while (loom_low_allocation_loop_edge_assignment_query_next(
        &query, &existing_assignment_index)) {
      if (conflict_assignments[existing_assignment_index] ||
          !loom_low_allocation_loop_edge_relocation_assignment_conflicts(
              state, &candidates[i],
              &context->assignments[existing_assignment_index])) {
        continue;
      }
      conflict_assignments[existing_assignment_index] = 1;
      ++*out_conflict_count;
    }
  }
}

static void loom_low_allocation_loop_edge_relocation_ignore_group(
    const loom_low_allocation_loop_edge_relocation_state_t* state,
    uint32_t assignment_index, uint8_t* ignored_assignments) {
  uint32_t member_index = assignment_index;
  do {
    ignored_assignments[member_index] = 1;
    member_index = state->groups.next_members[member_index];
  } while (member_index != assignment_index);
}

static void loom_low_allocation_loop_edge_relocation_apply_candidate(
    const loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_allocation_loop_edge_candidate_t* candidate) {
  const int64_t offset =
      (int64_t)candidate->assignment.location_base -
      state->context->assignments[candidate->destination_assignment_index]
          .location_base;
  uint32_t member_index = candidate->destination_assignment_index;
  do {
    loom_low_allocation_assignment_t* member =
        &state->context->assignments[member_index];
    member->location_base = (uint32_t)((int64_t)member->location_base + offset);
    member_index = state->groups.next_members[member_index];
  } while (member_index != candidate->destination_assignment_index);
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
      context->target_constraints, context->liveness, context->placement,
      interval, &capacity));
  *out_eviction = (loom_low_allocation_loop_edge_eviction_t){
      .assignment_index = assignment_index,
      .assignment = *assignment,
      .interval = interval,
      .capacity = capacity,
  };
  *out_supported = true;
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_loop_edge_relocation_initialize_eviction_groups(
    const loom_low_allocation_loop_edge_relocation_context_t* context,
    iree_arena_allocator_t* scratch_arena,
    const loom_low_allocation_loop_edge_eviction_t* evictions,
    iree_host_size_t eviction_count,
    loom_low_allocation_loop_edge_eviction_color_t** out_colors,
    loom_low_allocation_loop_edge_eviction_group_t** out_groups,
    iree_host_size_t* out_group_count) {
  *out_colors = NULL;
  *out_groups = NULL;
  *out_group_count = 0;
  if (eviction_count == 0) {
    return iree_ok_status();
  }

  loom_low_allocation_loop_edge_eviction_color_t* colors = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, eviction_count, sizeof(*colors), (void**)&colors));
  for (iree_host_size_t i = 0; i < eviction_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &evictions[i].assignment;
    colors[i] = (loom_low_allocation_loop_edge_eviction_color_t){
        .location_kind = assignment->location_kind,
        .descriptor_reg_class_id = assignment->descriptor_reg_class_id,
        .location = assignment->location_base,
        .owner_index = (uint32_t)i,
    };
  }
  loom_low_allocation_loop_edge_eviction_color_sort(colors, eviction_count);

  loom_low_allocation_loop_edge_eviction_group_t* groups = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, eviction_count, sizeof(*groups), (void**)&groups));
  memset(groups, 0, eviction_count * sizeof(*groups));
  for (iree_host_size_t group_start = 0; group_start < eviction_count;) {
    iree_host_size_t group_end = group_start + 1;
    while (group_end < eviction_count &&
           loom_low_allocation_loop_edge_eviction_color_key_compare(
               &colors[group_start], &colors[group_end]) == 0) {
      ++group_end;
    }
    const uint32_t first_eviction_index = colors[group_start].owner_index;
    groups[first_eviction_index] =
        (loom_low_allocation_loop_edge_eviction_group_t){
            .color_start = (uint32_t)group_start,
            .color_count = (uint32_t)(group_end - group_start),
        };
    group_start = group_end;
  }
  // Preserve first-encounter group priority. Equal storage units are ordered
  // by owner index, so the first unit in each range identifies that order
  // without a second sort.
  iree_host_size_t group_count = 0;
  for (iree_host_size_t i = 0; i < eviction_count; ++i) {
    if (groups[i].color_count != 0) {
      groups[group_count++] = groups[i];
    }
  }
  *out_colors = colors;
  *out_groups = groups;
  *out_group_count = group_count;
  return iree_ok_status();
}

static bool loom_low_allocation_loop_edge_query_set_conflicts(
    const loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_allocation_loop_edge_query_set_t* query_set,
    const loom_low_allocation_assignment_t* candidate,
    const uint8_t* ignored_assignments) {
  const loom_low_allocation_loop_edge_relocation_context_t* context =
      state->context;
  loom_low_allocation_loop_edge_assignment_query_t query;
  loom_low_allocation_loop_edge_assignment_query_initialize(
      query_set, context->descriptor_set, candidate, &query);
  uint32_t existing_assignment_index = 0;
  while (loom_low_allocation_loop_edge_assignment_query_next(
      &query, &existing_assignment_index)) {
    if (ignored_assignments[existing_assignment_index]) {
      continue;
    }
    if (loom_low_allocation_live_range_assignments_conflict(
            context->descriptor_set,
            context->unit_liveness->storage_segments.entries,
            context->unit_liveness->start_points,
            context->unit_liveness->end_points,
            context->unit_liveness->point_count, candidate,
            &context->assignments[existing_assignment_index])) {
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_loop_edge_relocation_eviction_location_is_legal(
    const loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_allocation_loop_edge_candidate_t* candidates,
    iree_host_size_t candidate_count,
    const loom_low_allocation_loop_edge_query_set_t* query_set,
    const loom_low_allocation_loop_edge_eviction_t* eviction,
    uint32_t vacancy_candidate_index, const uint8_t* ignored_assignments) {
  const loom_low_allocation_loop_edge_relocation_context_t* context =
      state->context;
  const loom_low_allocation_loop_edge_candidate_t* vacancy =
      &candidates[vacancy_candidate_index];
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
          state->context->descriptor_set, &eviction->capacity,
          assignment.location_kind, assignment.location_base,
          assignment.location_count)) {
    return false;
  }
  const uint32_t required_alignment =
      loom_low_allocation_live_range_interval_alignment(
          context->descriptor_set, context->liveness,
          context->placement->operand_constraints_by_interval,
          eviction->interval);
  if (assignment.location_base % required_alignment != 0) {
    return false;
  }
  if (loom_low_allocation_loop_edge_candidates_overlap_assignment(
          context->descriptor_set, candidates, candidate_count, &assignment) ||
      loom_low_allocation_loop_edge_query_set_conflicts(
          state, query_set, &assignment, ignored_assignments)) {
    return false;
  }
  if (loom_low_allocation_target_constraints_fixed_storage_conflicts(
          context->target_constraints, context->unit_liveness, &assignment,
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
  return true;
}

typedef struct loom_low_allocation_loop_edge_matching_t {
  // Loop relocation state owning assignments and target constraints.
  const loom_low_allocation_loop_edge_relocation_state_t* state;
  // Relevant original assignments used by legality queries.
  const loom_low_allocation_loop_edge_query_set_t* query_set;
  // Complete loop-header candidate group.
  const loom_low_allocation_loop_edge_candidate_t* candidates;
  // Number of entries in |candidates|.
  iree_host_size_t candidate_count;
  // External values that must leave candidate target locations.
  const loom_low_allocation_loop_edge_eviction_t* evictions;
  // Evictions ordered by their original storage color.
  const loom_low_allocation_loop_edge_eviction_color_t* eviction_colors;
  // Equal-color eviction spans in matching-policy order.
  const loom_low_allocation_loop_edge_eviction_group_t* eviction_groups;
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
  // Disjoint-set successor links for vacancies that remain unmatched. Entry
  // vacancy_count is the terminal sentinel.
  uint32_t* next_free_vacancy_indices;
  // Disjoint-set successor links for vacancies not yet visited by the current
  // augmenting path. Entry vacancy_count is the terminal sentinel.
  uint32_t* next_unvisited_vacancy_indices;
  // Search generation owning each next_unvisited_vacancy_indices entry.
  uint32_t* unvisited_vacancy_generations;
  // Current nonzero augmenting-path search generation.
  uint32_t unvisited_vacancy_generation;
} loom_low_allocation_loop_edge_matching_t;

static uint32_t loom_low_allocation_loop_edge_relocation_successor_find(
    uint32_t* successor_indices, uint32_t vacancy_index) {
  uint32_t next_index = vacancy_index;
  while (successor_indices[next_index] != next_index) {
    next_index = successor_indices[next_index];
  }
  while (vacancy_index != next_index) {
    const uint32_t successor = successor_indices[vacancy_index];
    successor_indices[vacancy_index] = next_index;
    vacancy_index = successor;
  }
  return next_index;
}

static uint32_t
loom_low_allocation_loop_edge_relocation_generation_successor_find(
    uint32_t* successor_indices, uint32_t* generations, uint32_t generation,
    uint32_t vacancy_index) {
  if (generations[vacancy_index] != generation) {
    generations[vacancy_index] = generation;
    successor_indices[vacancy_index] = vacancy_index;
    return vacancy_index;
  }
  uint32_t next_index = vacancy_index;
  while (successor_indices[next_index] != next_index) {
    next_index = successor_indices[next_index];
  }
  while (vacancy_index != next_index) {
    const uint32_t successor = successor_indices[vacancy_index];
    successor_indices[vacancy_index] = next_index;
    vacancy_index = successor;
  }
  return next_index;
}

static bool loom_low_allocation_loop_edge_relocation_eviction_group_is_legal(
    const loom_low_allocation_loop_edge_matching_t* matching,
    uint32_t group_index, uint32_t vacancy_index) {
  const loom_low_allocation_loop_edge_eviction_group_t* group =
      &matching->eviction_groups[group_index];
  for (uint32_t i = 0; i < group->color_count; ++i) {
    const uint32_t eviction_index =
        matching->eviction_colors[group->color_start + i].owner_index;
    const uint32_t vacancy_candidate_index =
        matching->vacancy_candidate_indices[vacancy_index];
    if (!loom_low_allocation_loop_edge_relocation_eviction_location_is_legal(
            matching->state, matching->candidates, matching->candidate_count,
            matching->query_set, &matching->evictions[eviction_index],
            vacancy_candidate_index, matching->ignored_assignments)) {
      return false;
    }
  }
  return true;
}

static bool loom_low_allocation_loop_edge_relocation_match_eviction(
    loom_low_allocation_loop_edge_matching_t* matching, uint32_t group_index,
    uint32_t* out_consumed_free_vacancy_index) {
  IREE_ASSERT_LT(group_index, matching->group_count);
  uint32_t vacancy_index =
      loom_low_allocation_loop_edge_relocation_successor_find(
          matching->next_free_vacancy_indices, 0);
  while (vacancy_index < matching->vacancy_count) {
    if (loom_low_allocation_loop_edge_relocation_eviction_group_is_legal(
            matching, group_index, vacancy_index)) {
      matching->group_indices_by_vacancy[vacancy_index] = group_index;
      *out_consumed_free_vacancy_index = vacancy_index;
      return true;
    }
    vacancy_index = loom_low_allocation_loop_edge_relocation_successor_find(
        matching->next_free_vacancy_indices, vacancy_index + 1);
  }

  vacancy_index =
      loom_low_allocation_loop_edge_relocation_generation_successor_find(
          matching->next_unvisited_vacancy_indices,
          matching->unvisited_vacancy_generations,
          matching->unvisited_vacancy_generation, 0);
  while (vacancy_index < matching->vacancy_count) {
    if (!loom_low_allocation_loop_edge_relocation_eviction_group_is_legal(
            matching, group_index, vacancy_index)) {
      vacancy_index =
          loom_low_allocation_loop_edge_relocation_generation_successor_find(
              matching->next_unvisited_vacancy_indices,
              matching->unvisited_vacancy_generations,
              matching->unvisited_vacancy_generation, vacancy_index + 1);
      continue;
    }
    matching->next_unvisited_vacancy_indices[vacancy_index] =
        loom_low_allocation_loop_edge_relocation_generation_successor_find(
            matching->next_unvisited_vacancy_indices,
            matching->unvisited_vacancy_generations,
            matching->unvisited_vacancy_generation, vacancy_index + 1);
    const uint32_t previous_group_index =
        matching->group_indices_by_vacancy[vacancy_index];
    IREE_ASSERT_NE(previous_group_index, UINT32_MAX);
    if (!loom_low_allocation_loop_edge_relocation_match_eviction(
            matching, previous_group_index, out_consumed_free_vacancy_index)) {
      vacancy_index =
          loom_low_allocation_loop_edge_relocation_generation_successor_find(
              matching->next_unvisited_vacancy_indices,
              matching->unvisited_vacancy_generations,
              matching->unvisited_vacancy_generation, vacancy_index);
      continue;
    }
    matching->group_indices_by_vacancy[vacancy_index] = group_index;
    return true;
  }
  return false;
}

static iree_status_t loom_low_allocation_loop_edge_relocation_try_full_group(
    loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_allocation_loop_edge_candidate_t* candidates,
    iree_host_size_t candidate_count,
    const loom_low_allocation_loop_edge_query_set_t* query_set,
    const uint8_t* ignored_assignments, bool* out_applied,
    iree_host_size_t* out_recolored_value_count) {
  *out_applied = false;
  *out_recolored_value_count = 0;
  const loom_low_allocation_loop_edge_relocation_context_t* context =
      state->context;
  if (context->assignment_count > UINT32_MAX) {
    return iree_ok_status();
  }
  uint8_t* conflict_assignments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, context->assignment_count,
      sizeof(*conflict_assignments), (void**)&conflict_assignments));
  memset(conflict_assignments, 0,
         context->assignment_count * sizeof(*conflict_assignments));
  iree_host_size_t conflict_count = 0;
  loom_low_allocation_loop_edge_relocation_collect_conflict_assignments(
      state, candidates, candidate_count, query_set, conflict_assignments,
      &conflict_count);

  uint32_t* vacancy_candidate_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, candidate_count, sizeof(*vacancy_candidate_indices),
      (void**)&vacancy_candidate_indices));
  iree_host_size_t vacancy_count = 0;
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    const loom_low_allocation_assignment_t* old_destination =
        &context->assignments[candidates[i].destination_assignment_index];
    const bool location_remains_occupied =
        loom_low_allocation_loop_edge_candidates_overlap_assignment(
            context->descriptor_set, candidates, candidate_count,
            old_destination);
    if (!location_remains_occupied) {
      vacancy_candidate_indices[vacancy_count++] = (uint32_t)i;
    }
  }
  for (iree_host_size_t i = 0; i < vacancy_count; ++i) {
    const loom_low_allocation_assignment_t* lhs =
        &context->assignments[candidates[vacancy_candidate_indices[i]]
                                  .destination_assignment_index];
    for (iree_host_size_t j = 0; j < i; ++j) {
      const loom_low_allocation_assignment_t* rhs =
          &context->assignments[candidates[vacancy_candidate_indices[j]]
                                    .destination_assignment_index];
      // A physical location can host at most one eviction color. Header values
      // may share storage when their unit liveness is disjoint, so reject the
      // full recoloring path when two apparent vacancies alias. The subset
      // relocation path below can still apply moves that need no eviction.
      if (loom_low_allocation_storage_assignment_ranges_overlap(
              context->descriptor_set, lhs, rhs)) {
        return iree_ok_status();
      }
    }
  }
  if (conflict_count > UINT32_MAX) {
    return iree_ok_status();
  }

  loom_low_allocation_loop_edge_eviction_t* evictions = NULL;
  if (conflict_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->scratch_arena, conflict_count,
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

  loom_low_allocation_loop_edge_eviction_color_t* eviction_colors = NULL;
  loom_low_allocation_loop_edge_eviction_group_t* eviction_groups = NULL;
  iree_host_size_t group_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_loop_edge_relocation_initialize_eviction_groups(
          context, state->scratch_arena, evictions, eviction_count,
          &eviction_colors, &eviction_groups, &group_count));
  if (group_count > vacancy_count) {
    return iree_ok_status();
  }

  uint8_t* eviction_ignored_assignments = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->scratch_arena, context->assignment_count,
                                sizeof(*eviction_ignored_assignments),
                                (void**)&eviction_ignored_assignments));
  memset(eviction_ignored_assignments, 0,
         context->assignment_count * sizeof(*eviction_ignored_assignments));
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    loom_low_allocation_loop_edge_relocation_ignore_group(
        state, candidates[i].destination_assignment_index,
        eviction_ignored_assignments);
  }
  for (iree_host_size_t i = 0; i < conflict_count; ++i) {
    eviction_ignored_assignments[evictions[i].assignment_index] = 1;
  }

  uint32_t* group_indices_by_vacancy = NULL;
  loom_low_allocation_assignment_t* assignments_by_eviction = NULL;
  uint32_t* next_free_vacancy_indices = NULL;
  uint32_t* next_unvisited_vacancy_indices = NULL;
  uint32_t* unvisited_vacancy_generations = NULL;
  if (conflict_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, vacancy_count, sizeof(*group_indices_by_vacancy),
        (void**)&group_indices_by_vacancy));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, conflict_count, sizeof(*assignments_by_eviction),
        (void**)&assignments_by_eviction));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->scratch_arena, vacancy_count + 1,
                                  sizeof(*next_free_vacancy_indices),
                                  (void**)&next_free_vacancy_indices));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->scratch_arena, vacancy_count + 1,
                                  sizeof(*next_unvisited_vacancy_indices),
                                  (void**)&next_unvisited_vacancy_indices));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->scratch_arena, vacancy_count + 1,
                                  sizeof(*unvisited_vacancy_generations),
                                  (void**)&unvisited_vacancy_generations));
    memset(unvisited_vacancy_generations, 0,
           (vacancy_count + 1) * sizeof(*unvisited_vacancy_generations));
    for (iree_host_size_t i = 0; i < vacancy_count; ++i) {
      group_indices_by_vacancy[i] = UINT32_MAX;
    }
    for (uint32_t i = 0; i <= (uint32_t)vacancy_count; ++i) {
      next_free_vacancy_indices[i] = i;
    }
    loom_low_allocation_loop_edge_matching_t matching = {
        .state = state,
        .query_set = query_set,
        .candidates = candidates,
        .candidate_count = candidate_count,
        .evictions = evictions,
        .eviction_colors = eviction_colors,
        .eviction_groups = eviction_groups,
        .group_count = group_count,
        .vacancy_candidate_indices = vacancy_candidate_indices,
        .vacancy_count = vacancy_count,
        .ignored_assignments = eviction_ignored_assignments,
        .group_indices_by_vacancy = group_indices_by_vacancy,
        .next_free_vacancy_indices = next_free_vacancy_indices,
        .next_unvisited_vacancy_indices = next_unvisited_vacancy_indices,
        .unvisited_vacancy_generations = unvisited_vacancy_generations,
    };
    for (uint32_t i = 0; i < (uint32_t)group_count; ++i) {
      matching.unvisited_vacancy_generation = i + 1;
      uint32_t consumed_free_vacancy_index = UINT32_MAX;
      if (!loom_low_allocation_loop_edge_relocation_match_eviction(
              &matching, i, &consumed_free_vacancy_index)) {
        return iree_ok_status();
      }
      IREE_ASSERT_LT(consumed_free_vacancy_index, vacancy_count);
      next_free_vacancy_indices[consumed_free_vacancy_index] =
          loom_low_allocation_loop_edge_relocation_successor_find(
              next_free_vacancy_indices, consumed_free_vacancy_index + 1);
    }
    // Materialize assignments only after every augmenting path has settled.
    // Writing locations while searching would leave stale locations behind
    // when a recursive match backtracks to a prior vacancy.
    for (iree_host_size_t i = 0; i < vacancy_count; ++i) {
      const uint32_t group_index = group_indices_by_vacancy[i];
      if (group_index == UINT32_MAX) {
        continue;
      }
      const loom_low_allocation_loop_edge_eviction_group_t* group =
          &eviction_groups[group_index];
      const loom_low_allocation_assignment_t* vacancy =
          &context->assignments[candidates[vacancy_candidate_indices[i]]
                                    .destination_assignment_index];
      for (uint32_t j = 0; j < group->color_count; ++j) {
        const uint32_t eviction_index =
            eviction_colors[group->color_start + j].owner_index;
        assignments_by_eviction[eviction_index] =
            evictions[eviction_index].assignment;
        assignments_by_eviction[eviction_index].location_base =
            vacancy->location_base;
      }
    }
  }

  if (loom_low_allocation_loop_edge_write_conflicts(
          state, candidates, candidate_count, NULL, assignments_by_eviction,
          conflict_count)) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    loom_low_allocation_loop_edge_relocation_apply_candidate(state,
                                                             &candidates[i]);
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
  uint32_t member_index = destination_assignment_index;
  do {
    if (loom_low_allocation_loop_edge_relocation_assignment_conflicts(
            state, candidate, &state->context->assignments[member_index])) {
      return true;
    }
    member_index = state->groups.next_members[member_index];
  } while (member_index != destination_assignment_index);
  return false;
}

static void
loom_low_allocation_loop_edge_relocation_disable_candidate_dependents(
    const loom_low_allocation_loop_edge_relocation_state_t* state,
    const loom_low_allocation_loop_edge_candidate_t* candidates,
    iree_host_size_t candidate_count, uint32_t required_candidate_index,
    uint8_t* candidate_enabled, uint32_t* disabled_candidate_indices,
    iree_host_size_t* inout_disabled_candidate_count) {
  const uint32_t destination_assignment_index =
      candidates[required_candidate_index].destination_assignment_index;
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    if (!candidate_enabled[i] ||
        !loom_low_allocation_loop_edge_relocation_candidate_depends_on_destination(
            state, &candidates[i], destination_assignment_index)) {
      continue;
    }
    candidate_enabled[i] = 0;
    disabled_candidate_indices[(*inout_disabled_candidate_count)++] =
        (uint32_t)i;
  }
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
  if (state->groups.representatives == NULL) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_relocation_groups_initialize(
        state->context->descriptor_set, state->context->placement,
        &state->assignment_map, state->context->arena, &state->groups));
    IREE_RETURN_IF_ERROR(loom_low_allocation_write_proposal_initialize(
        state->context->unit_liveness->write_interference,
        state->context->arena, &state->write_proposal));
  }

  loom_low_allocation_loop_edge_candidate_t* candidates = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->scratch_arena, header->arg_count,
                                sizeof(*candidates), (void**)&candidates));
  iree_host_size_t candidate_count = 0;
  for (uint16_t i = 0; i < header->arg_count; ++i) {
    bool candidate_found = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_loop_edge_relocation_collect_candidate(
            state, loom_block_arg_id(header, i), backedge->terminator,
            &candidates[candidate_count], &candidate_found));
    // A hard target or lease conflict cannot be repaired by evicting ordinary
    // assignments. Remove it before considering the joint relocation.
    if (candidate_found &&
        !loom_low_allocation_loop_edge_relocation_candidate_target_conflicts(
            state, &candidates[candidate_count])) {
      ++candidate_count;
    }
  }
  if (candidate_count == 0) {
    return iree_ok_status();
  }

  uint8_t* ignored_assignments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, state->context->assignment_count,
      sizeof(*ignored_assignments), (void**)&ignored_assignments));
  memset(ignored_assignments, 0,
         state->context->assignment_count * sizeof(*ignored_assignments));
  if (!loom_low_allocation_loop_edge_relocation_candidates_are_disjoint(
          state, candidates, candidate_count, ignored_assignments)) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    loom_low_allocation_loop_edge_relocation_ignore_group(
        state, candidates[i].destination_assignment_index, ignored_assignments);
    ignored_assignments[candidates[i].source_assignment_index] = 1;
  }

  loom_low_allocation_loop_edge_query_set_t query_set;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_loop_edge_relocation_query_set_initialize(
          state, candidates, candidate_count, ignored_assignments, &query_set));

  bool full_group_applied = false;
  IREE_RETURN_IF_ERROR(loom_low_allocation_loop_edge_relocation_try_full_group(
      state, candidates, candidate_count, &query_set, ignored_assignments,
      &full_group_applied, out_recolored_value_count));
  if (full_group_applied) {
    *out_relocated_value_count = candidate_count;
    return iree_ok_status();
  }

  uint8_t* candidate_enabled = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, candidate_count, sizeof(*candidate_enabled),
      (void**)&candidate_enabled));
  memset(candidate_enabled, 1, candidate_count * sizeof(*candidate_enabled));
  uint32_t* disabled_candidate_indices = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->scratch_arena, candidate_count,
                                sizeof(*disabled_candidate_indices),
                                (void**)&disabled_candidate_indices));
  iree_host_size_t disabled_candidate_count = 0;
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    if (loom_low_allocation_loop_edge_relocation_candidate_has_conflict(
            state, &candidates[i], &query_set)) {
      candidate_enabled[i] = 0;
      disabled_candidate_indices[disabled_candidate_count++] = (uint32_t)i;
    }
  }

  // An enabled candidate may only ignore the old destination storage of
  // another candidate when that destination will itself move. Candidate count
  // is bounded by header arguments, so carry candidate identity directly while
  // computing the dependency closure.
  iree_host_size_t disabled_candidate_position = 0;
  while (disabled_candidate_position < disabled_candidate_count) {
    const uint32_t required_candidate_index =
        disabled_candidate_indices[disabled_candidate_position++];
    loom_low_allocation_loop_edge_relocation_disable_candidate_dependents(
        state, candidates, candidate_count, required_candidate_index,
        candidate_enabled, disabled_candidate_indices,
        &disabled_candidate_count);
  }

  if (loom_low_allocation_loop_edge_write_conflicts(
          state, candidates, candidate_count, candidate_enabled, NULL, 0)) {
    return iree_ok_status();
  }
  iree_host_size_t relocated_value_count = 0;
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    if (!candidate_enabled[i]) {
      continue;
    }
    loom_low_allocation_loop_edge_relocation_apply_candidate(state,
                                                             &candidates[i]);
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
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(context->arena->block_pool, &scratch_arena);
  loom_low_allocation_loop_edge_relocation_state_t state = {
      .context = context,
      .scratch_arena = &scratch_arena,
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
      context->module, context->body, context->cfg_graph, context->liveness,
      context->value_domain, context->arena, &state.consumption_query);

  iree_status_t status = iree_ok_status();
  for (uint16_t header_index = 0;
       iree_status_is_ok(status) &&
       header_index < context->cfg_graph->block_count;
       ++header_index) {
    if (!loom_cfg_graph_block_is_reachable(context->cfg_graph, header_index)) {
      continue;
    }
    iree_host_size_t relocated_value_count = 0;
    iree_host_size_t recolored_value_count = 0;
    status = loom_low_allocation_loop_edge_relocation_try_header(
        &state, header_index, &relocated_value_count, &recolored_value_count);
    iree_arena_reset(&scratch_arena);
    if (!iree_status_is_ok(status) || relocated_value_count == 0) {
      continue;
    }
    out_result->relocated_value_count += relocated_value_count;
    out_result->recolored_value_count += recolored_value_count;
    ++out_result->relocated_header_count;
  }
  iree_arena_deinitialize(&scratch_arena);
  if (iree_status_is_ok(status) && out_result->relocated_value_count != 0) {
    loom_low_allocation_target_constraints_rebuild_assignment_location_ends(
        context->target_constraints, context->assignments,
        context->assignment_count);
  }
  return status;
}
