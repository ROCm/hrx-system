// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/preference.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/allocation/storage.h"

struct loom_low_allocation_preference_memo_entry_t {
  // Both independently varying masked candidate bases, packed into one tag.
  uint64_t locations;
  // Exact objective value for this tag in the current prepared query.
  uint32_t penalty;
  // One plus the candidate-presence bitset; zero marks an empty entry.
  uint32_t presence;
};

iree_status_t loom_low_allocation_preference_workspace_initialize(
    const loom_low_placement_preference_index_t* index,
    iree_arena_allocator_t* arena,
    loom_low_allocation_preference_workspace_t* out_workspace) {
  *out_workspace = (loom_low_allocation_preference_workspace_t){0};
  if (index == NULL || index->use_count == index->instruction_use_count) {
    return iree_ok_status();
  }
  // A concat query merges two origin lists; neither merged span can exceed
  // the total index. Use wide arithmetic for the producer-retained bounds.
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena,
      iree_min((uint64_t)index->max_incident_use_count * 2, index->use_count),
      sizeof(*out_workspace->use_indices),
      (void**)&out_workspace->use_indices));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena,
      iree_min((uint64_t)index->max_incident_binding_count * 2,
               index->binding_count),
      sizeof(*out_workspace->locations), (void**)&out_workspace->locations));
  if (index->max_memo_entry_count == 0) {
    return iree_ok_status();
  }
  return iree_arena_allocate_array(arena, index->max_memo_entry_count,
                                   sizeof(*out_workspace->memo_entries),
                                   (void**)&out_workspace->memo_entries);
}

static loom_value_ordinal_t loom_low_allocation_preference_origin(
    const loom_low_placement_table_t* placement, loom_value_ordinal_t ordinal) {
  return placement->tied_storage_origins_by_value_ordinal
             ? placement->tied_storage_origins_by_value_ordinal[ordinal]
             : ordinal;
}

bool loom_low_allocation_preference_has_uses(
    const loom_low_placement_preference_index_t* index,
    const loom_low_placement_table_t* placement, loom_value_ordinal_t ordinal) {
  if (index == NULL || index->use_count == index->instruction_use_count) {
    return false;
  }
  const loom_value_ordinal_t origin =
      loom_low_allocation_preference_origin(placement, ordinal);
  return index->offsets_by_origin[origin] !=
         index->offsets_by_origin[origin + 1];
}

static const loom_low_allocation_assignment_t*
loom_low_allocation_preference_assignment(
    const loom_low_allocation_assignment_map_t* assignments,
    const loom_low_allocation_target_constraints_t* constraints,
    loom_value_ordinal_t ordinal) {
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_assignment_map_assignment_for_value_ordinal(
          assignments, ordinal, NULL);
  if (assignment != NULL) {
    return assignment;
  }
  const loom_low_allocation_resolved_fixed_value_t* fixed =
      loom_low_allocation_target_constraints_fixed_value_for_value_ordinal(
          constraints, ordinal);
  return fixed ? &fixed->assignment : NULL;
}

static const loom_low_allocation_assignment_t*
loom_low_allocation_preference_copy_prediction(
    const loom_low_placement_table_t* placement,
    const loom_low_allocation_assignment_map_t* assignments,
    const loom_low_allocation_target_constraints_t* constraints,
    loom_value_ordinal_t ordinal) {
  const loom_low_placement_relation_t* transfer =
      loom_low_placement_defining_transfer_for_value_ordinal(placement,
                                                             ordinal);
  if (transfer == NULL ||
      transfer->kind != LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE ||
      !loom_low_placement_relation_can_alias(transfer) ||
      transfer->result_unit_offset != 0 || transfer->source_unit_offset != 0) {
    return NULL;
  }
  const loom_liveness_interval_t* destination =
      loom_liveness_interval_for_value_ordinal(assignments->liveness, ordinal);
  const loom_low_allocation_assignment_t* source =
      loom_low_allocation_preference_assignment(assignments, constraints,
                                                transfer->source_ordinal);
  if (source == NULL ||
      !loom_low_allocation_assignment_is_register_like(source) ||
      transfer->unit_count != destination->unit_count ||
      transfer->unit_count != source->unit_count ||
      !loom_low_allocation_storage_reg_classes_share(
          constraints->target->descriptor_set,
          destination->value_class.register_class_id,
          source->descriptor_reg_class_id)) {
    return NULL;
  }
  return source;
}

static void loom_low_allocation_preference_prepare_use(
    const loom_low_placement_preference_index_t* index, uint32_t use_index,
    const loom_low_placement_table_t* placement,
    const loom_low_allocation_assignment_map_t* assignments,
    const loom_low_allocation_target_constraints_t* constraints,
    loom_value_ordinal_t primary_origin, loom_value_ordinal_t secondary_origin,
    loom_low_allocation_preference_location_t* locations) {
  const loom_low_placement_preference_use_t* use = &index->uses[use_index];
  const loom_low_placement_preference_binding_t* bindings =
      &index->bindings[use->binding_start];
  // Representatives are first in binding order. Collect every concrete member
  // before predicting any peer, so an assigned spill also blocks stale copies.
  for (uint16_t i = 0; i < use->preference->value_count; ++i) {
    const loom_value_ordinal_t ordinal = bindings[i].value_ordinal;
    const loom_value_ordinal_t origin =
        loom_low_allocation_preference_origin(placement, ordinal);
    locations[i] = (loom_low_allocation_preference_location_t){
        .representative = bindings[i].representative,
        .candidate_index = origin == primary_origin     ? 1
                           : origin == secondary_origin ? 2
                                                        : 0,
    };
    if (locations[i].candidate_index == 0) {
      locations[i].assignment = loom_low_allocation_preference_assignment(
          assignments, constraints, ordinal);
    }
    loom_low_allocation_preference_location_t* representative =
        &locations[bindings[i].representative];
    if (representative->assignment == NULL && locations[i].assignment != NULL) {
      representative->assignment = locations[i].assignment;
      representative->candidate_index = 0;
    }
  }
  for (uint16_t i = 0; i < use->preference->value_count; ++i) {
    if (bindings[i].representative != i || locations[i].assignment != NULL ||
        locations[i].candidate_index != 0) {
      continue;
    }
    const loom_value_ordinal_t origin = loom_low_allocation_preference_origin(
        placement, bindings[i].value_ordinal);
    if (origin != bindings[i].value_ordinal) {
      locations[i].assignment = loom_low_allocation_preference_assignment(
          assignments, constraints, origin);
    }
    if (locations[i].assignment == NULL) {
      locations[i].assignment = loom_low_allocation_preference_copy_prediction(
          placement, assignments, constraints, origin);
    }
  }
  for (uint16_t i = 0; i < use->preference->value_count; ++i) {
    if (locations[i].assignment == NULL) {
      const loom_low_allocation_preference_location_t* representative =
          &locations[bindings[i].representative];
      locations[i].assignment = representative->assignment;
      locations[i].candidate_index = representative->candidate_index;
    }
  }
}

static bool loom_low_allocation_preference_is_structural(
    const loom_low_placement_relation_t* relation) {
  if (relation->cause == LOOM_LOW_PLACEMENT_CAUSE_SCHEDULE_PAIR_AFFINITY) {
    return false;
  }
  return (relation->kind == LOOM_LOW_PLACEMENT_RELATION_SAME_REGISTER_ORDINAL &&
          iree_any_bit_set(relation->flags,
                           LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED)) ||
         relation->kind == LOOM_LOW_PLACEMENT_RELATION_DISJOINT_STORAGE ||
         (relation->kind == LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE &&
          (relation->cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY ||
           relation->cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_MOVE));
}

static bool loom_low_allocation_preference_is_hard_register_ordinal(
    const loom_low_placement_relation_t* relation) {
  return relation->kind == LOOM_LOW_PLACEMENT_RELATION_SAME_REGISTER_ORDINAL &&
         iree_any_bit_set(relation->flags,
                          LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD);
}

static const loom_low_allocation_assignment_t*
loom_low_allocation_preference_structural_counterpart(
    const loom_low_allocation_preference_query_t* query,
    const loom_low_placement_relation_t* relation,
    loom_value_ordinal_t ordinal) {
  const loom_low_allocation_assignment_t* assigned =
      loom_low_allocation_assignment_map_assignment_for_value_ordinal(
          query->structural.assignments, ordinal, NULL);
  // Ordinary copies are already handled by coalescing. Their search preference
  // matters only when the counterpart has a future fixed location.
  if (assigned != NULL) {
    return relation->kind == LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE
               ? NULL
               : assigned;
  }
  const loom_low_allocation_resolved_fixed_value_t* fixed =
      loom_low_allocation_target_constraints_fixed_value_for_value_ordinal(
          query->structural.constraints, ordinal);
  return fixed ? &fixed->assignment : NULL;
}

// Projects the concrete source of a permitted defining copy into the copy
// result's class. The source and result classes may encode the same physical
// storage with different candidate orders or location representations.
static bool loom_low_allocation_preference_hard_ordinal_copy_prediction(
    const loom_low_allocation_preference_query_t* query,
    const loom_low_placement_table_t* placement, loom_value_ordinal_t ordinal,
    loom_low_allocation_assignment_t* out_prediction) {
  ordinal = loom_low_allocation_preference_origin(placement, ordinal);
  if (loom_low_allocation_preference_assignment(query->structural.assignments,
                                                query->structural.constraints,
                                                ordinal) != NULL) {
    return false;
  }
  const loom_low_allocation_assignment_t* source =
      loom_low_allocation_preference_copy_prediction(
          placement, query->structural.assignments,
          query->structural.constraints, ordinal);
  if (source == NULL ||
      source->location_kind != LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
      source->location_count != 1) {
    return false;
  }
  const loom_liveness_analysis_t* liveness =
      query->structural.assignments->liveness;
  const loom_liveness_interval_t* destination =
      loom_liveness_interval_for_value_ordinal(liveness, ordinal);
  if (destination->unit_count != 1) {
    return false;
  }
  const uint16_t destination_class = destination->value_class.register_class_id;
  uint32_t destination_location = 0;
  if (!loom_low_allocation_storage_find_subrange_alias_location(
          query->structural.constraints->target->descriptor_set,
          destination_class, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          /*candidate_location_count=*/1, /*candidate_unit_start=*/0, source,
          /*reference_unit_start=*/0, /*unit_count=*/1,
          &destination_location)) {
    return false;
  }
  *out_prediction = (loom_low_allocation_assignment_t){
      .value_id = liveness->value_ids[ordinal],
      .descriptor_reg_class_id = destination_class,
      .unit_count = 1,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = destination_location,
      .location_count = 1,
  };
  return true;
}

static const loom_low_allocation_assignment_t*
loom_low_allocation_preference_structural_scoring_counterpart(
    const loom_low_allocation_preference_query_t* query,
    const loom_low_placement_table_t* placement,
    const loom_low_placement_relation_t* relation, loom_value_ordinal_t ordinal,
    loom_low_allocation_assignment_t* out_prediction) {
  if (loom_low_allocation_preference_is_structural(relation)) {
    const loom_low_allocation_assignment_t* counterpart =
        loom_low_allocation_preference_structural_counterpart(query, relation,
                                                              ordinal);
    if (counterpart != NULL) {
      return counterpart;
    }
  }
  if (loom_low_allocation_preference_is_hard_register_ordinal(relation) &&
      loom_low_allocation_preference_hard_ordinal_copy_prediction(
          query, placement, ordinal, out_prediction)) {
    return out_prediction;
  }
  return NULL;
}

static bool loom_low_allocation_preference_structural_is_actionable(
    const loom_low_allocation_preference_query_t* query,
    const loom_low_placement_table_t* placement,
    const loom_low_placement_relation_t* relation, uint32_t candidate_index,
    bool candidate_is_result) {
  const bool is_structural =
      loom_low_allocation_preference_is_structural(relation);
  if (!is_structural &&
      !loom_low_allocation_preference_is_hard_register_ordinal(relation)) {
    return false;
  }
  const loom_value_ordinal_t candidate =
      query->structural.candidates[candidate_index].ordinal;
  const loom_value_ordinal_t counterpart =
      candidate_is_result ? relation->source_ordinal : relation->result_ordinal;
  const loom_liveness_analysis_t* liveness =
      query->structural.assignments->liveness;
  const uint16_t candidate_class =
      loom_liveness_interval_for_value_ordinal(liveness, candidate)
          ->value_class.register_class_id;
  uint16_t counterpart_class;
  if (counterpart ==
      query->structural.candidates[1 - candidate_index].ordinal) {
    if (!is_structural ||
        relation->kind == LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE) {
      return false;
    }
    counterpart_class =
        loom_liveness_interval_for_value_ordinal(liveness, counterpart)
            ->value_class.register_class_id;
  } else {
    loom_low_allocation_assignment_t prediction;
    const loom_low_allocation_assignment_t* assigned =
        loom_low_allocation_preference_structural_scoring_counterpart(
            query, placement, relation, counterpart, &prediction);
    if (assigned == NULL ||
        !loom_low_allocation_assignment_is_register_like(assigned)) {
      return false;
    }
    counterpart_class = assigned->descriptor_reg_class_id;
  }
  return relation->kind == LOOM_LOW_PLACEMENT_RELATION_SAME_REGISTER_ORDINAL ||
         loom_low_allocation_storage_reg_classes_share(
             query->structural.constraints->target->descriptor_set,
             candidate_class, counterpart_class);
}

static void loom_low_allocation_preference_prepare_structural(
    const loom_low_placement_table_t* placement,
    const loom_low_allocation_assignment_map_t* assignments,
    const loom_low_allocation_target_constraints_t* constraints,
    loom_value_ordinal_t primary, loom_value_ordinal_t secondary,
    loom_low_allocation_preference_query_t* query) {
  if (placement->location_relation_count == 0 &&
      constraints->fixed_value_count == 0) {
    return;
  }
  query->structural.assignments = assignments;
  query->structural.constraints = constraints;
  query->structural.candidates[0].ordinal = primary;
  query->structural.candidates[1].ordinal = secondary;
  for (uint32_t i = 0; i < 2; ++i) {
    const loom_value_ordinal_t ordinal =
        query->structural.candidates[i].ordinal;
    if (ordinal == LOOM_VALUE_ORDINAL_INVALID) {
      continue;
    }
    query->structural.candidates[i].results =
        placement->ranges_by_result_ordinal[ordinal];
    query->structural.candidates[i].sources =
        placement->ranges_by_source_ordinal[ordinal];
    const loom_low_placement_relation_range_t results =
        query->structural.candidates[i].results;
    const loom_low_placement_relation_range_t sources =
        query->structural.candidates[i].sources;
    for (uint32_t j = 0; j < results.count + sources.count &&
                         query->structural.placement == NULL;
         ++j) {
      const uint32_t relation_index =
          j < results.count
              ? results.start + j
              : placement
                    ->relation_indices_by_source_ordinal[sources.start + j -
                                                         results.count];
      if (loom_low_allocation_preference_structural_is_actionable(
              query, placement, &placement->relations[relation_index], i,
              j < results.count)) {
        query->structural.placement = placement;
      }
    }
  }
}

loom_low_allocation_preference_query_t loom_low_allocation_preference_prepare(
    const loom_low_placement_preference_index_t* index,
    const loom_low_placement_table_t* placement,
    const loom_low_allocation_assignment_map_t* assignments,
    const loom_low_allocation_target_constraints_t* constraints,
    loom_value_ordinal_t primary_ordinal,
    loom_value_ordinal_t secondary_ordinal,
    loom_low_allocation_preference_workspace_t* workspace) {
  loom_low_allocation_preference_query_t query = {0};
  loom_low_allocation_preference_prepare_structural(
      placement, assignments, constraints, primary_ordinal, secondary_ordinal,
      &query);
  if (index == NULL || index->use_count == index->instruction_use_count) {
    return query;
  }
  const loom_value_ordinal_t primary_origin =
      loom_low_allocation_preference_origin(placement, primary_ordinal);
  const loom_value_ordinal_t secondary_origin =
      secondary_ordinal == LOOM_VALUE_ORDINAL_INVALID
          ? LOOM_VALUE_ORDINAL_INVALID
          : loom_low_allocation_preference_origin(placement, secondary_ordinal);
  uint32_t primary = index->offsets_by_origin[primary_origin];
  const uint32_t primary_end = index->offsets_by_origin[primary_origin + 1];
  uint32_t secondary = 0;
  uint32_t secondary_end = 0;
  if (secondary_origin != LOOM_VALUE_ORDINAL_INVALID &&
      secondary_origin != primary_origin) {
    secondary = index->offsets_by_origin[secondary_origin];
    secondary_end = index->offsets_by_origin[secondary_origin + 1];
  }
  query.index = index;
  query.use_indices = workspace->use_indices;
  query.locations = workspace->locations;
  uint32_t binding_start = 0;
  uint8_t location_bit_count = 0;
  uint8_t index_bit_count = 0;
  bool memo_applicable = query.structural.placement == NULL;
  while (primary < primary_end || secondary < secondary_end) {
    const uint32_t primary_use =
        primary < primary_end ? index->use_indices[primary] : UINT32_MAX;
    const uint32_t secondary_use =
        secondary < secondary_end ? index->use_indices[secondary] : UINT32_MAX;
    const uint32_t use_index = iree_min(primary_use, secondary_use);
    primary += primary_use == use_index;
    secondary += secondary_use == use_index;
    workspace->use_indices[query.use_count++] = use_index;
    const loom_low_placement_preference_use_t* use = &index->uses[use_index];
    location_bit_count =
        iree_max(location_bit_count, use->memo.location_bit_count);
    if (use->memo.index_bit_count_plus_one != 0) {
      index_bit_count =
          iree_max(index_bit_count, use->memo.index_bit_count_plus_one - 1);
    } else {
      memo_applicable = false;
    }
    loom_low_allocation_preference_prepare_use(
        index, use_index, placement, assignments, constraints, primary_origin,
        secondary_origin, &workspace->locations[binding_start]);
    binding_start += use->preference->value_count;
  }
  if (memo_applicable && location_bit_count != 0) {
    const uint32_t memo_entry_count = UINT32_C(1) << index_bit_count;
    query.memo.entries = workspace->memo_entries;
    query.memo.location_mask = UINT32_MAX >> (32 - location_bit_count);
    query.memo.index_mask = memo_entry_count - 1;
    memset(query.memo.entries, 0,
           memo_entry_count * sizeof(*query.memo.entries));
  }
  return query;
}

static bool loom_low_allocation_preference_predicate_established(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_placement_predicate_t* predicate,
    const loom_low_allocation_preference_location_t* locations,
    const loom_low_allocation_assignment_t* const* candidates) {
  const loom_low_allocation_preference_location_t* result =
      &locations[predicate->result];
  const loom_low_allocation_preference_location_t* source =
      &locations[predicate->source];
  const loom_low_allocation_assignment_t* result_assignment =
      result->candidate_index ? candidates[result->candidate_index - 1]
                              : result->assignment;
  const loom_low_allocation_assignment_t* source_assignment =
      source->candidate_index ? candidates[source->candidate_index - 1]
                              : source->assignment;
  if (result_assignment != NULL && source_assignment != NULL) {
    if (!loom_low_allocation_assignment_is_register_like(result_assignment) ||
        !loom_low_allocation_assignment_is_register_like(source_assignment) ||
        !loom_low_allocation_storage_assignment_classes_share(
            descriptor_set, result_assignment, source_assignment)) {
      return false;
    }
    return !loom_low_allocation_storage_relation_satisfied(
        descriptor_set, predicate->kind, predicate->result_unit_offset,
        predicate->source_unit_offset, predicate->unit_count,
        predicate->location_mask, result_assignment, source_assignment);
  }
  if (result_assignment != NULL || source_assignment != NULL ||
      result->representative != source->representative) {
    return false;
  }
  // For a contiguous low-bit mask, equal residues of two offsets prove equal
  // masked locations for every base. Arbitrary masks require concrete bases.
  if (predicate->kind ==
      LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION) {
    const uint32_t mask = predicate->location_mask;
    return (mask & (mask + 1)) == 0 &&
           ((predicate->result_unit_offset ^ predicate->source_unit_offset) &
            mask) == 0;
  }
  if (predicate->kind == LOOM_LOW_PLACEMENT_RELATION_DISJOINT_STORAGE) {
    return predicate->result_unit_offset <
               (uint32_t)predicate->source_unit_offset +
                   predicate->unit_count &&
           predicate->source_unit_offset <
               (uint32_t)predicate->result_unit_offset + predicate->unit_count;
  }
  return false;
}

static uint32_t loom_low_allocation_preference_structural_penalty(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_preference_query_t* query,
    const loom_low_allocation_assignment_t* const* candidates) {
  const loom_low_placement_table_t* placement = query->structural.placement;
  if (placement == NULL) {
    return 0;
  }
  uint32_t penalty = 0;
  for (uint32_t i = 0; i < 2; ++i) {
    const loom_value_ordinal_t ordinal =
        query->structural.candidates[i].ordinal;
    if (ordinal == LOOM_VALUE_ORDINAL_INVALID) {
      continue;
    }
    const loom_low_placement_relation_range_t results =
        query->structural.candidates[i].results;
    const loom_low_placement_relation_range_t sources =
        query->structural.candidates[i].sources;
    for (uint32_t j = 0; j < results.count + sources.count; ++j) {
      const uint32_t relation_index =
          j < results.count
              ? results.start + j
              : placement
                    ->relation_indices_by_source_ordinal[sources.start + j -
                                                         results.count];
      const loom_low_placement_relation_t* relation =
          &placement->relations[relation_index];
      const bool is_structural =
          loom_low_allocation_preference_is_structural(relation);
      if (!is_structural &&
          !loom_low_allocation_preference_is_hard_register_ordinal(relation)) {
        continue;
      }
      const loom_value_ordinal_t counterpart_ordinal =
          j < results.count ? relation->source_ordinal
                            : relation->result_ordinal;
      // A relation joining the two candidates was already scored from the
      // first candidate. A self-relation is visited only in its result range.
      if ((i != 0 &&
           counterpart_ordinal == query->structural.candidates[0].ordinal) ||
          (j >= results.count && counterpart_ordinal == ordinal)) {
        continue;
      }
      loom_low_allocation_assignment_t prediction;
      const loom_low_allocation_assignment_t* counterpart;
      if (counterpart_ordinal == query->structural.candidates[1 - i].ordinal) {
        if (!is_structural ||
            relation->kind == LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE) {
          continue;
        }
        counterpart = candidates[1 - i];
      } else {
        counterpart =
            loom_low_allocation_preference_structural_scoring_counterpart(
                query, placement, relation, counterpart_ordinal, &prediction);
      }
      if (counterpart == NULL ||
          (relation->kind !=
               LOOM_LOW_PLACEMENT_RELATION_SAME_REGISTER_ORDINAL &&
           !loom_low_allocation_storage_assignment_classes_share(
               descriptor_set, candidates[i], counterpart))) {
        continue;
      }
      if (!loom_low_allocation_storage_placement_relation_satisfied(
              descriptor_set, relation,
              j < results.count ? candidates[i] : counterpart,
              j < results.count ? counterpart : candidates[i])) {
        penalty = iree_math_saturating_add_u32(
            penalty, iree_max(1u, (uint32_t)relation->priority));
      }
    }
  }
  return penalty;
}

static uint32_t loom_low_allocation_preference_evaluate(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_preference_query_t* query,
    const loom_low_allocation_assignment_t* primary,
    const loom_low_allocation_assignment_t* secondary) {
  const loom_low_allocation_assignment_t* candidates[] = {primary, secondary};
  uint32_t total = loom_low_allocation_preference_structural_penalty(
      descriptor_set, query, candidates);
  uint32_t binding_start = 0;
  for (uint32_t i = 0; i < query->use_count; ++i) {
    const loom_low_placement_preference_use_t* use =
        &query->index->uses[query->use_indices[i]];
    const loom_low_placement_preference_t* preference = use->preference;
    const loom_low_allocation_preference_location_t* locations =
        &query->locations[binding_start];
    for (uint16_t j = 0; j < preference->clause_count; ++j) {
      const loom_low_placement_clause_t* clause = &preference->clauses[j];
      const bool require_all = clause->kind == LOOM_LOW_PLACEMENT_CLAUSE_ALL;
      bool established = require_all;
      for (uint16_t k = 0; k < clause->predicate_count; ++k) {
        const bool predicate_established =
            loom_low_allocation_preference_predicate_established(
                descriptor_set,
                &preference->predicates[clause->predicate_start + k], locations,
                candidates);
        if (predicate_established != require_all) {
          established = predicate_established;
          break;
        }
      }
      if (established) {
        total = iree_math_saturating_add_u32(
            total, (uint32_t)iree_min((uint64_t)clause->weight * use->priority,
                                      UINT32_MAX));
      }
    }
    binding_start += preference->value_count;
  }
  return total;
}

uint32_t loom_low_allocation_preference_penalty(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_preference_query_t* query,
    const loom_low_allocation_assignment_t* primary,
    const loom_low_allocation_assignment_t* secondary) {
  if (query->memo.entries == NULL) {
    return loom_low_allocation_preference_evaluate(descriptor_set, query,
                                                   primary, secondary);
  }
  const uint32_t first = primary ? primary->location_base : 0;
  const uint32_t second = secondary ? secondary->location_base : 0;
  const uint64_t locations =
      (uint64_t)(first & query->memo.location_mask) |
      ((uint64_t)(second & query->memo.location_mask) << 32);
  const uint32_t presence = 1 + (primary != NULL) + 2 * (secondary != NULL);
  loom_low_allocation_preference_memo_entry_t* entry =
      &query->memo.entries[first & query->memo.index_mask];
  if (entry->presence != presence || entry->locations != locations) {
    entry->penalty = loom_low_allocation_preference_evaluate(
        descriptor_set, query, primary, secondary);
    entry->locations = locations;
    entry->presence = presence;
  }
  return entry->penalty;
}
