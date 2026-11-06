// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/numbering.h"

#include <string.h>

#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/allocation/storage_lease_index.h"

typedef struct loom_low_numbering_space_t {
  // Beginning in the pass-local concatenation of physical storage spaces.
  uint32_t start;
  // Original physical extent, or zero for an inactive storage space.
  uint32_t count;
  // An instruction preference requests numbering in this storage space.
  bool active;
} loom_low_numbering_space_t;

typedef struct loom_low_numbering_block_t {
  // Original beginning in the concatenated physical coordinate space.
  uint32_t base;
  // Number of units in this rigid overlap-connected component.
  uint32_t count;
  // Beginning of the owning storage space, subtracted for target costs.
  uint32_t space_start;
  // Inclusive lower bound on translation from the original base.
  int32_t minimum_delta;
  // Inclusive upper bound on translation from the original base.
  int32_t maximum_delta;
  // Required power-of-two translation alignment.
  uint32_t alignment;
  // Carry-closed low bits affecting incident preferences.
  uint32_t period_mask;
} loom_low_numbering_block_t;

typedef struct loom_low_numbering_state_t {
  // Borrowed mutable allocation and immutable producer facts.
  const loom_low_allocation_numbering_context_t* context;
  // Descriptor storage and preference semantics.
  const loom_low_descriptor_set_t* descriptors;
  // Dense alias-set prefix followed by class-local storage spaces.
  loom_low_numbering_space_t* spaces;
  // Number of entries in spaces.
  uint32_t space_count;
  // Total active physical extent across storage spaces.
  uint32_t extent;
  // Rigid physical components, with capacity equal to extent.
  loom_low_numbering_block_t* blocks;
  // Number of initialized components.
  uint32_t block_count;
  // Physical-unit to component index.
  uint32_t* unit_blocks;
  // Component order in the accepted numbering.
  uint32_t* order;
  // Accepted component bases.
  uint32_t* bases;
  // Trial component bases while crossing adjacent blocks.
  uint32_t* trial_bases;
  // Construction reach/cursor storage, reused for accepted rotations.
  uint32_t* scratch;
  // Block-to-preference CSR offsets.
  uint32_t* offsets;
  // Incident use indexes, at most one entry per bound value.
  uint32_t* incidences;
  // Original concatenated unit per binding, or UINT32_MAX outside the search.
  uint32_t* locations;
  // Original costs bounding every accepted preference use.
  uint32_t* limits;
  // Accepted per-use costs.
  uint32_t* scores;
  // Trial per-use costs.
  uint32_t* trial_scores;
  // Per-crossing deduplication marks, cleared while scoring the crossing.
  uint32_t* queued;
  // Distinct use indexes affected by the current crossing.
  uint32_t* affected;
  // Number of initialized entries in affected.
  uint32_t affected_count;
  // Number of trial components violating their retained domain.
  uint32_t illegal_blocks;
  // Number of trial uses exceeding their original cost.
  uint32_t regressing_uses;
  // Largest retained location period minus one, bounding moved run widths.
  uint32_t maximum_move_width;
  // Accepted weighted objective, summed without per-use saturation overflow.
  uint64_t score;
  // Current trial weighted objective.
  uint64_t trial_score;
  // Counted search work, including snapshots and accepted refreshes.
  uint64_t work;
  // Input-proportional work limit; one final crossing/refresh may overrun it.
  uint64_t work_limit;
} loom_low_numbering_state_t;

// Alias-set IDs are dense and one-based. Class-local spaces follow them so
// distinct register classes never share coordinates accidentally.
static uint32_t loom_low_numbering_space_index(
    const loom_low_descriptor_set_t* descriptors, uint16_t class_id) {
  const uint16_t alias_set = descriptors->reg_classes[class_id].alias_set_id;
  return alias_set != 0 ? alias_set - 1
                        : (uint32_t)descriptors->reg_class_count + class_id;
}

static void loom_low_numbering_mark_storage_space_classes(
    const loom_low_numbering_state_t* state, uint16_t class_id,
    iree_bitmap_t classes) {
  const uint32_t space_index =
      loom_low_numbering_space_index(state->descriptors, class_id);
  for (uint16_t i = 0; i < state->descriptors->reg_class_count; ++i) {
    if (loom_low_numbering_space_index(state->descriptors, i) == space_index) {
      iree_bitmap_set(classes, i);
    }
  }
}

static const loom_low_allocation_assignment_t* loom_low_numbering_binding(
    const loom_low_numbering_state_t* state, uint32_t binding_index) {
  const loom_low_allocation_interval_assignment_result_t* allocation =
      state->context->interval_assignment;
  const loom_value_ordinal_t ordinal =
      state->context->preferences->bindings[binding_index].value_ordinal;
  return &allocation->assignments
              [allocation->assignment_indices_by_value_ordinal[ordinal]];
}

static uint32_t loom_low_numbering_unit(
    const loom_low_numbering_state_t* state, uint16_t class_id,
    loom_low_allocation_location_kind_t kind, uint32_t location) {
  if (kind != LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
      loom_low_reg_class_uses_explicit_physical_registers(
          &state->descriptors->reg_classes[class_id])) {
    return UINT32_MAX;
  }
  const loom_low_numbering_space_t* space =
      &state->spaces[loom_low_numbering_space_index(state->descriptors,
                                                    class_id)];
  return location < space->count ? space->start + location : UINT32_MAX;
}

static void loom_low_numbering_retain_range(
    loom_low_numbering_state_t* state, uint16_t class_id,
    loom_low_allocation_location_kind_t kind, uint32_t base, uint32_t count) {
  const uint32_t unit = loom_low_numbering_unit(state, class_id, kind, base);
  if (unit == UINT32_MAX) {
    return;
  }
  const loom_low_numbering_space_t* space =
      &state->spaces[loom_low_numbering_space_index(state->descriptors,
                                                    class_id)];
  const uint32_t end = unit + iree_min(count, space->count - base);
  state->scratch[unit] = iree_max(state->scratch[unit], end);
}

static void loom_low_numbering_build_blocks(loom_low_numbering_state_t* state) {
  const loom_low_allocation_numbering_context_t* context = state->context;
  for (uint32_t i = 0; i < state->extent; ++i) {
    state->scratch[i] = i + 1;
  }
  for (iree_host_size_t i = 0;
       i < context->interval_assignment->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &context->interval_assignment->assignments[i];
    loom_low_numbering_retain_range(
        state, assignment->descriptor_reg_class_id, assignment->location_kind,
        assignment->location_base, assignment->location_count);
  }
  for (iree_host_size_t i = 0; i < context->storage_leases->instance_count;
       ++i) {
    const loom_low_allocation_storage_lease_t* lease =
        &context->storage_leases->instances[i];
    loom_low_numbering_retain_range(state, lease->descriptor_reg_class_id,
                                    lease->location_kind, lease->location_base,
                                    lease->location_count);
  }
  for (uint32_t space_index = 0; space_index < state->space_count;
       ++space_index) {
    const loom_low_numbering_space_t* space = &state->spaces[space_index];
    for (uint32_t begin = space->start; begin < space->start + space->count;) {
      uint32_t end = state->scratch[begin];
      for (uint32_t i = begin + 1; i < end; ++i) {
        end = iree_max(end, state->scratch[i]);
      }
      const uint32_t index = state->block_count++;
      state->blocks[index] = (loom_low_numbering_block_t){
          .base = begin,
          .count = end - begin,
          .space_start = space->start,
          .minimum_delta = -(int32_t)(begin - space->start),
          .maximum_delta = (int32_t)(space->start + space->count - end),
          .alignment = 1,
      };
      state->order[index] = index;
      state->bases[index] = begin;
      for (uint32_t i = begin; i < end; ++i) {
        state->unit_blocks[i] = index;
      }
      begin = end;
    }
  }
}

static void loom_low_numbering_anchor_range(
    loom_low_numbering_state_t* state, uint16_t class_id,
    loom_low_allocation_location_kind_t kind, uint32_t base, uint32_t count) {
  const uint32_t unit = loom_low_numbering_unit(state, class_id, kind, base);
  if (unit == UINT32_MAX) {
    return;
  }
  const loom_low_numbering_space_t* space =
      &state->spaces[loom_low_numbering_space_index(state->descriptors,
                                                    class_id)];
  const uint32_t end = unit + iree_min(count, space->count - base);
  for (uint32_t cursor = unit; cursor < end;) {
    loom_low_numbering_block_t* block =
        &state->blocks[state->unit_blocks[cursor]];
    block->minimum_delta = block->maximum_delta = 0;
    cursor = block->base + block->count;
  }
}

static void loom_low_numbering_constrain_extent(
    loom_low_numbering_state_t* state, uint16_t class_id,
    loom_low_allocation_location_kind_t kind, uint32_t base, uint32_t count,
    uint32_t extent) {
  const uint32_t unit = loom_low_numbering_unit(state, class_id, kind, base);
  if (unit == UINT32_MAX) {
    return;
  }
  loom_low_numbering_block_t* block = &state->blocks[state->unit_blocks[unit]];
  if (loom_low_reg_class_fixed_location_range_contains(
          &state->descriptors->reg_classes[class_id], base, count)) {
    block->minimum_delta = block->maximum_delta = 0;
  } else {
    block->maximum_delta =
        iree_min(block->maximum_delta, (int32_t)(extent - base - count));
  }
}

static void loom_low_numbering_constrain_blocks(
    loom_low_numbering_state_t* state) {
  const loom_low_allocation_numbering_context_t* context = state->context;
  const loom_low_allocation_interval_assignment_result_t* allocation =
      context->interval_assignment;
  const loom_liveness_analysis_t* liveness =
      allocation->assignment_map.liveness;
  const loom_low_allocation_target_constraints_t* target =
      context->target_constraints;
  for (loom_value_ordinal_t ordinal = 0; ordinal < liveness->value_count;
       ++ordinal) {
    const uint32_t assignment_index =
        allocation->assignment_indices_by_value_ordinal[ordinal];
    if (assignment_index == UINT32_MAX) {
      continue;
    }
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[assignment_index];
    const uint16_t class_id = assignment->descriptor_reg_class_id;
    const uint32_t unit = loom_low_numbering_unit(
        state, class_id, assignment->location_kind, assignment->location_base);
    if (unit == UINT32_MAX) {
      continue;
    }
    loom_low_numbering_block_t* block =
        &state->blocks[state->unit_blocks[unit]];
    const loom_low_placement_operand_constraints_t operand =
        context->placement->operand_constraints_by_interval
            ? context->placement->operand_constraints_by_interval
                  [liveness->value_interval_indices[ordinal]]
            : (loom_low_placement_operand_constraints_t){0};
    uint32_t extent = target->max_assigned_location_end_by_reg_class[class_id];
    if (operand.addressable_unit_count != 0) {
      extent = iree_min(extent, operand.addressable_unit_count);
    }
    loom_low_numbering_constrain_extent(
        state, class_id, assignment->location_kind, assignment->location_base,
        assignment->location_count, extent);
    block->alignment =
        iree_max(block->alignment,
                 iree_max(UINT32_C(1) << operand.unit_alignment_log2,
                          loom_low_reg_class_unit_alignment(
                              &state->descriptors->reg_classes[class_id],
                              assignment->unit_count)));
    if (operand.has_target_address_state) {
      block->minimum_delta = block->maximum_delta = 0;
    }
  }
  for (iree_host_size_t i = 0; i < context->move_count; ++i) {
    const loom_low_move_location_t* endpoints[] = {
        &context->moves[i].source, &context->moves[i].destination};
    for (uint32_t j = 0; j < 2; ++j) {
      const loom_low_move_location_t* endpoint = endpoints[j];
      loom_low_numbering_constrain_extent(
          state, endpoint->descriptor_reg_class_id, endpoint->location_kind,
          endpoint->location, 1,
          target->max_assigned_location_end_by_reg_class
              [endpoint->descriptor_reg_class_id]);
    }
  }
  for (iree_host_size_t i = 0; i < target->fixed_value_count; ++i) {
    const loom_low_allocation_assignment_t* fixed =
        &target->fixed_values[i].assignment;
    loom_low_numbering_anchor_range(state, fixed->descriptor_reg_class_id,
                                    fixed->location_kind, fixed->location_base,
                                    fixed->location_count);
  }
  for (iree_host_size_t i = 0; i < context->entry_location_count; ++i) {
    const loom_low_allocation_abi_location_t* entry =
        &context->entry_locations[i];
    const uint32_t assignment_index =
        allocation->assignment_indices_by_value_ordinal[i];
    if (entry->location_kind == LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED ||
        assignment_index == UINT32_MAX) {
      continue;
    }
    const loom_low_allocation_assignment_t* argument =
        &allocation->assignments[assignment_index];
    loom_low_numbering_anchor_range(state, entry->descriptor_reg_class_id,
                                    entry->location_kind, entry->location_base,
                                    argument->location_count);
  }
  for (iree_host_size_t i = 0; i < context->boundary_anchor_count; ++i) {
    const loom_low_allocation_numbering_anchor_t* anchor =
        &context->boundary_anchors[i];
    loom_low_numbering_anchor_range(
        state, anchor->location.descriptor_reg_class_id,
        anchor->location.location_kind, anchor->location.location_base,
        anchor->unit_count);
  }
  for (iree_host_size_t i = 0; i < target->reserved_range_count; ++i) {
    const loom_low_allocation_resolved_reserved_range_t* range =
        &target->reserved_ranges[i];
    loom_low_numbering_anchor_range(state, range->descriptor_reg_class_id,
                                    range->location_kind, range->location_base,
                                    range->location_count);
  }
  if (context->unit_liveness->implicit_location_counts_by_reg_class != NULL) {
    for (uint16_t i = 0; i < state->descriptors->reg_class_count; ++i) {
      loom_low_numbering_anchor_range(
          state, i, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, 0,
          context->unit_liveness->implicit_location_counts_by_reg_class[i]);
    }
  }
  for (iree_host_size_t i = 0; i < context->placement->relation_count; ++i) {
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[i];
    if (!iree_any_bit_set(relation->flags,
                          LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD) ||
        (relation->kind != LOOM_LOW_PLACEMENT_RELATION_SAME_REGISTER_ORDINAL &&
         relation->kind !=
             LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION)) {
      continue;
    }
    const loom_value_ordinal_t ordinals[] = {relation->result_ordinal,
                                             relation->source_ordinal};
    for (uint32_t j = 0; j < 2; ++j) {
      const loom_low_allocation_assignment_t* assignment =
          &allocation->assignments
               [allocation->assignment_indices_by_value_ordinal[ordinals[j]]];
      loom_low_numbering_anchor_range(
          state, assignment->descriptor_reg_class_id, assignment->location_kind,
          assignment->location_base, assignment->location_count);
    }
  }
}

static uint32_t loom_low_numbering_location(
    const loom_low_numbering_state_t* state, uint32_t binding_index,
    const uint32_t* bases) {
  const uint32_t unit = state->locations[binding_index];
  if (unit == UINT32_MAX) {
    return loom_low_numbering_binding(state, binding_index)->location_base;
  }
  const uint32_t block_index = state->unit_blocks[unit];
  const loom_low_numbering_block_t* block = &state->blocks[block_index];
  return bases[block_index] - block->space_start + unit - block->base;
}

static bool loom_low_numbering_predicate_violated(
    const loom_low_numbering_state_t* state,
    const loom_low_placement_predicate_t* predicate, uint32_t result_index,
    uint32_t source_index, const uint32_t* bases) {
  const uint32_t result_unit = state->locations[result_index];
  const uint32_t source_unit = state->locations[source_index];
  if (predicate->kind ==
          LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION &&
      result_unit != UINT32_MAX && source_unit != UINT32_MAX) {
    const uint32_t result_block_index = state->unit_blocks[result_unit];
    const uint32_t source_block_index = state->unit_blocks[source_unit];
    const loom_low_numbering_block_t* result_block =
        &state->blocks[result_block_index];
    const loom_low_numbering_block_t* source_block =
        &state->blocks[source_block_index];
    // Active coordinates already identify their physical inventory. Aliased
    // classes share one space; independent classes cannot conflict.
    if (result_block->space_start != source_block->space_start) {
      return false;
    }
    const uint32_t result_location =
        bases[result_block_index] - result_block->space_start + result_unit -
        result_block->base + predicate->result_unit_offset;
    const uint32_t source_location =
        bases[source_block_index] - source_block->space_start + source_unit -
        source_block->base + predicate->source_unit_offset;
    return ((result_location ^ source_location) & predicate->location_mask) ==
           0;
  }
  const loom_low_allocation_assignment_t* result =
      loom_low_numbering_binding(state, result_index);
  const loom_low_allocation_assignment_t* source =
      loom_low_numbering_binding(state, source_index);
  if (!loom_low_allocation_storage_assignment_classes_share(state->descriptors,
                                                            result, source)) {
    return false;
  }
  if (predicate->kind ==
      LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION) {
    const uint32_t result_location =
        loom_low_numbering_location(state, result_index, bases) +
        predicate->result_unit_offset;
    const uint32_t source_location =
        loom_low_numbering_location(state, source_index, bases) +
        predicate->source_unit_offset;
    return ((result_location ^ source_location) & predicate->location_mask) ==
           0;
  }
  // A rigid bijection preserves structural relations. Evaluate these against
  // the unchanged assignments rather than reconstructing them.
  return !loom_low_allocation_storage_relation_satisfied(
      state->descriptors, predicate->kind, predicate->result_unit_offset,
      predicate->source_unit_offset, predicate->unit_count,
      predicate->location_mask, result, source);
}

// Returns true only when a predicate violation survives every legal rigid
// block permutation. Predicates spanning independent movable blocks may still
// be impossible for other reasons, but treating them as unknown keeps this a
// conservative lower bound instead of rediscovering the complete search.
static bool loom_low_numbering_predicate_unavoidably_violated(
    const loom_low_numbering_state_t* state,
    const loom_low_placement_predicate_t* predicate, uint32_t result_index,
    uint32_t source_index) {
  if (predicate->kind !=
      LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION) {
    return loom_low_numbering_predicate_violated(state, predicate, result_index,
                                                 source_index, state->bases);
  }
  const uint32_t result_unit = state->locations[result_index];
  const uint32_t source_unit = state->locations[source_index];
  if (result_unit == UINT32_MAX || source_unit == UINT32_MAX) {
    return result_unit == source_unit &&
           loom_low_numbering_predicate_violated(state, predicate, result_index,
                                                 source_index, state->bases);
  }
  if (state->unit_blocks[result_unit] != state->unit_blocks[source_unit]) {
    return false;
  }
  return loom_low_numbering_predicate_violated(state, predicate, result_index,
                                               source_index, state->bases);
}

static uint32_t loom_low_numbering_unavoidable_use_cost(
    const loom_low_numbering_state_t* state, uint32_t use_index,
    iree_bitmap_t unavoidable_reg_classes) {
  const loom_low_placement_preference_use_t* use =
      &state->context->preferences->uses[use_index];
  const loom_low_placement_preference_t* preference = use->preference;
  uint32_t score = 0;
  for (uint16_t i = 0; i < preference->clause_count; ++i) {
    const loom_low_placement_clause_t* clause = &preference->clauses[i];
    bool established = clause->kind == LOOM_LOW_PLACEMENT_CLAUSE_ALL;
    for (uint16_t j = 0; j < clause->predicate_count; ++j) {
      const loom_low_placement_predicate_t* predicate =
          &preference->predicates[clause->predicate_start + j];
      const bool unavoidable =
          loom_low_numbering_predicate_unavoidably_violated(
              state, predicate, use->binding_start + predicate->result,
              use->binding_start + predicate->source);
      if (clause->kind == LOOM_LOW_PLACEMENT_CLAUSE_ALL) {
        established &= unavoidable;
        if (!established) {
          break;
        }
      } else {
        established |= unavoidable;
        if (established) {
          break;
        }
      }
    }
    if (established) {
      const uint32_t contribution = (uint32_t)clause->weight * use->priority;
      score =
          contribution > UINT32_MAX - score ? UINT32_MAX : score + contribution;
      if (unavoidable_reg_classes.bit_count != 0) {
        for (uint16_t j = 0; j < clause->predicate_count; ++j) {
          const loom_low_placement_predicate_t* predicate =
              &preference->predicates[clause->predicate_start + j];
          const uint32_t result_index = use->binding_start + predicate->result;
          const uint32_t source_index = use->binding_start + predicate->source;
          if (!loom_low_numbering_predicate_unavoidably_violated(
                  state, predicate, result_index, source_index)) {
            continue;
          }
          loom_low_numbering_mark_storage_space_classes(
              state,
              loom_low_numbering_binding(state, result_index)
                  ->descriptor_reg_class_id,
              unavoidable_reg_classes);
          loom_low_numbering_mark_storage_space_classes(
              state,
              loom_low_numbering_binding(state, source_index)
                  ->descriptor_reg_class_id,
              unavoidable_reg_classes);
        }
      }
    }
  }
  return score;
}

static uint32_t loom_low_numbering_score_use(loom_low_numbering_state_t* state,
                                             uint32_t use_index,
                                             const uint32_t* bases) {
  const loom_low_placement_preference_use_t* use =
      &state->context->preferences->uses[use_index];
  const loom_low_placement_preference_t* preference = use->preference;
  uint32_t score = 0;
  for (uint16_t i = 0; i < preference->clause_count; ++i) {
    const loom_low_placement_clause_t* clause = &preference->clauses[i];
    bool established = clause->kind == LOOM_LOW_PLACEMENT_CLAUSE_ALL;
    for (uint16_t j = 0; j < clause->predicate_count; ++j) {
      const loom_low_placement_predicate_t* predicate =
          &preference->predicates[clause->predicate_start + j];
      const uint32_t result_index = use->binding_start + predicate->result;
      const uint32_t source_index = use->binding_start + predicate->source;
      const bool violation = loom_low_numbering_predicate_violated(
          state, predicate, result_index, source_index, bases);
      ++state->work;
      if (clause->kind == LOOM_LOW_PLACEMENT_CLAUSE_ALL) {
        established &= violation;
        if (!established) {
          break;
        }
      } else {
        established |= violation;
        if (established) {
          break;
        }
      }
    }
    if (established) {
      const uint32_t contribution = (uint32_t)clause->weight * use->priority;
      score =
          contribution > UINT32_MAX - score ? UINT32_MAX : score + contribution;
    }
  }
  return score;
}

static void loom_low_numbering_index_preferences(
    loom_low_numbering_state_t* state) {
  const loom_low_placement_preference_index_t* preferences =
      state->context->preferences;
  for (uint32_t i = 0; i < preferences->binding_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_numbering_binding(state, i);
    state->locations[i] = loom_low_numbering_unit(
        state, assignment->descriptor_reg_class_id, assignment->location_kind,
        assignment->location_base);
  }
  uint64_t predicate_terms = 0;
  for (uint32_t i = 0; i < preferences->use_count; ++i) {
    const loom_low_placement_preference_use_t* use = &preferences->uses[i];
    const uint32_t period_mask =
        use->memo.location_bit_count == 0
            ? 0
            : UINT32_MAX >> (32 - use->memo.location_bit_count);
    state->maximum_move_width =
        iree_max(state->maximum_move_width, period_mask);
    for (uint16_t j = 0; j < use->preference->clause_count; ++j) {
      predicate_terms += use->preference->clauses[j].predicate_count;
    }
    for (uint16_t j = 0; j < use->preference->value_count; ++j) {
      const uint32_t unit = state->locations[use->binding_start + j];
      if (unit == UINT32_MAX) {
        continue;
      }
      const uint32_t block = state->unit_blocks[unit];
      ++state->offsets[block + 1];
      state->blocks[block].period_mask |= period_mask;
    }
    state->scores[i] = state->limits[i] =
        loom_low_numbering_score_use(state, i, state->bases);
    state->score += state->scores[i];
  }
  for (uint32_t i = 0; i < state->block_count; ++i) {
    state->offsets[i + 1] += state->offsets[i];
    state->scratch[i] = state->offsets[i];
  }
  for (uint32_t i = 0; i < preferences->use_count; ++i) {
    const loom_low_placement_preference_use_t* use = &preferences->uses[i];
    for (uint16_t j = 0; j < use->preference->value_count; ++j) {
      const uint32_t unit = state->locations[use->binding_start + j];
      if (unit == UINT32_MAX) {
        continue;
      }
      state->incidences[state->scratch[state->unit_blocks[unit]]++] = i;
    }
  }
  // Charge work rather than passes: high fanout and rejected candidates spend
  // the same finite budget as successful moves. Finishing one in-flight
  // crossing and accepted refresh adds at most another linear amount of work.
  state->work_limit =
      16 *
      (state->extent + (uint64_t)preferences->binding_count + predicate_terms);
}

static bool loom_low_numbering_block_is_legal(
    const loom_low_numbering_block_t* block, uint32_t base) {
  const int32_t delta = (int32_t)(base - block->base);
  return delta >= block->minimum_delta && delta <= block->maximum_delta &&
         ((uint32_t)delta & (block->alignment - 1)) == 0;
}

static void loom_low_numbering_move_block(loom_low_numbering_state_t* state,
                                          uint32_t block_index, int32_t delta) {
  const loom_low_numbering_block_t* block = &state->blocks[block_index];
  const uint32_t before = state->trial_bases[block_index];
  const uint32_t after = before + delta;
  state->illegal_blocks -= !loom_low_numbering_block_is_legal(block, before);
  state->illegal_blocks += !loom_low_numbering_block_is_legal(block, after);
  state->trial_bases[block_index] = after;
  ++state->work;
  if (((before ^ after) & block->period_mask) == 0) {
    return;
  }
  for (uint32_t i = state->offsets[block_index];
       i < state->offsets[block_index + 1]; ++i) {
    const uint32_t use_index = state->incidences[i];
    ++state->work;
    if (state->queued[use_index]) {
      continue;
    }
    state->queued[use_index] = 1;
    state->affected[state->affected_count++] = use_index;
  }
}

static void loom_low_numbering_score_crossing(
    loom_low_numbering_state_t* state) {
  for (uint32_t i = 0; i < state->affected_count; ++i) {
    const uint32_t use_index = state->affected[i];
    state->queued[use_index] = 0;
    const uint32_t before = state->trial_scores[use_index];
    const uint32_t after =
        loom_low_numbering_score_use(state, use_index, state->trial_bases);
    state->trial_scores[use_index] = after;
    state->trial_score = state->trial_score - before + after;
    state->regressing_uses -= before > state->limits[use_index];
    state->regressing_uses += after > state->limits[use_index];
  }
}

static bool loom_low_numbering_search(loom_low_numbering_state_t* state) {
  const uint32_t block_count = state->block_count;
  const uint32_t use_count = state->context->preferences->use_count;
  bool changed = false;
  while (state->score != 0 && state->work < state->work_limit) {
    uint64_t best_score = state->score;
    uint32_t best_begin = 0, best_end = 0, best_destination = 0;
    for (uint32_t begin = 0;
         begin < block_count && state->work < state->work_limit; ++begin) {
      uint32_t width = 0;
      for (uint32_t end = begin + 1;
           end <= block_count && state->work < state->work_limit; ++end) {
        ++state->work;
        const loom_low_numbering_block_t* tail =
            &state->blocks[state->order[end - 1]];
        if (tail->minimum_delta == 0 && tail->maximum_delta == 0) {
          break;
        }
        width += tail->count;
        if (width > state->maximum_move_width) {
          break;
        }
        for (uint32_t direction = 0;
             direction < 2 && state->work < state->work_limit; ++direction) {
          memcpy(state->trial_bases, state->bases,
                 block_count * sizeof(uint32_t));
          memcpy(state->trial_scores, state->scores,
                 use_count * sizeof(uint32_t));
          state->work += block_count + use_count;
          state->trial_score = state->score;
          state->illegal_blocks = state->regressing_uses = 0;
          const uint32_t distance = direction == 0 ? begin : block_count - end;
          for (uint32_t step = 0;
               step < distance && state->work < state->work_limit; ++step) {
            ++state->work;
            state->affected_count = 0;
            const uint32_t crossed_position =
                direction == 0 ? begin - 1 - step : end + step;
            const uint32_t crossed = state->order[crossed_position];
            const int32_t shift = direction == 0
                                      ? -(int32_t)state->blocks[crossed].count
                                      : (int32_t)state->blocks[crossed].count;
            for (uint32_t i = begin; i < end; ++i) {
              loom_low_numbering_move_block(state, state->order[i], shift);
            }
            loom_low_numbering_move_block(
                state, crossed,
                direction == 0 ? (int32_t)width : -(int32_t)width);
            loom_low_numbering_score_crossing(state);
            // This crossed block stays shifted for the rest of the walk;
            // only the moved run can leave and later reenter its legal domain.
            if (!loom_low_numbering_block_is_legal(
                    &state->blocks[crossed], state->trial_bases[crossed])) {
              break;
            }
            if (state->illegal_blocks != 0 || state->regressing_uses != 0) {
              continue;
            }
            const uint32_t destination =
                direction == 0 ? begin - 1 - step : begin + 1 + step;
            if (state->trial_score < best_score ||
                (best_end != 0 && state->trial_score == best_score &&
                 begin == best_begin && end == best_end &&
                 destination < best_destination)) {
              best_score = state->trial_score;
              best_begin = begin;
              best_end = end;
              best_destination = destination;
            }
          }
        }
      }
      // Complete all small runs at the first improving origin, not another
      // whole-function sweep after a useful local insertion is already known.
      if (best_end != 0) {
        break;
      }
    }
    if (best_end == 0) {
      break;
    }
    const uint32_t run_count = best_end - best_begin;
    memcpy(state->scratch, state->order + best_begin,
           run_count * sizeof(uint32_t));
    if (best_destination < best_begin) {
      memmove(state->order + best_destination + run_count,
              state->order + best_destination,
              (best_begin - best_destination) * sizeof(uint32_t));
    } else {
      memmove(state->order + best_begin, state->order + best_end,
              (best_destination - best_begin) * sizeof(uint32_t));
    }
    memcpy(state->order + best_destination, state->scratch,
           run_count * sizeof(uint32_t));
    uint32_t base = 0;
    for (uint32_t i = 0; i < block_count; ++i) {
      const uint32_t block = state->order[i];
      state->bases[block] = base;
      base += state->blocks[block].count;
    }
    // Accepted maps refresh the objective once. Candidate crossings update
    // only their incident uses, including originally satisfied packet pairs.
    state->score = 0;
    for (uint32_t i = 0; i < use_count; ++i) {
      state->scores[i] = loom_low_numbering_score_use(state, i, state->bases);
      state->score += state->scores[i];
    }
    state->work += block_count;
    changed = true;
  }
  return changed;
}

static uint32_t loom_low_numbering_translate(
    const loom_low_numbering_state_t* state, uint16_t class_id,
    loom_low_allocation_location_kind_t kind, uint32_t location) {
  const uint32_t unit =
      loom_low_numbering_unit(state, class_id, kind, location);
  if (unit == UINT32_MAX) {
    return location;
  }
  const uint32_t block_index = state->unit_blocks[unit];
  return location + state->bases[block_index] - state->blocks[block_index].base;
}

static void loom_low_numbering_publish(
    const loom_low_numbering_state_t* state) {
  const loom_low_allocation_numbering_context_t* context = state->context;
  for (iree_host_size_t i = 0;
       i < context->interval_assignment->assignment_count; ++i) {
    loom_low_allocation_assignment_t* assignment =
        &context->interval_assignment->assignments[i];
    assignment->location_base = loom_low_numbering_translate(
        state, assignment->descriptor_reg_class_id, assignment->location_kind,
        assignment->location_base);
  }
  for (iree_host_size_t i = 0; i < context->storage_leases->instance_count;
       ++i) {
    loom_low_allocation_storage_lease_t* lease =
        &context->storage_leases->instances[i];
    lease->location_base = loom_low_numbering_translate(
        state, lease->descriptor_reg_class_id, lease->location_kind,
        lease->location_base);
  }
  for (iree_host_size_t i = 0; i < context->move_count; ++i) {
    loom_low_move_location_t* endpoints[] = {&context->moves[i].source,
                                             &context->moves[i].destination};
    for (uint32_t j = 0; j < 2; ++j) {
      loom_low_move_location_t* endpoint = endpoints[j];
      endpoint->location = loom_low_numbering_translate(
          state, endpoint->descriptor_reg_class_id, endpoint->location_kind,
          endpoint->location);
    }
  }
  if (loom_low_allocation_storage_lease_unit_index_is_enabled(
          context->storage_leases->unit_index)) {
    loom_low_allocation_storage_lease_unit_index_rebuild(
        context->storage_leases->unit_index, state->descriptors,
        context->storage_leases->instance_count);
  }
}

static iree_status_t loom_low_numbering_initialize_spaces(
    loom_low_numbering_state_t* state, iree_arena_allocator_t* arena) {
  const loom_low_placement_preference_index_t* preferences =
      state->context->preferences;
  state->space_count = (uint32_t)state->descriptors->reg_class_count * 2;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, state->space_count,
                                                 sizeof(*state->spaces),
                                                 (void**)&state->spaces));
  memset(state->spaces, 0, state->space_count * sizeof(*state->spaces));
  for (uint32_t i = 0; i < preferences->instruction_use_count; ++i) {
    const loom_low_placement_preference_use_t* use = &preferences->uses[i];
    if (use->memo.location_bit_count == 0) {
      continue;
    }
    for (uint16_t j = 0; j < use->preference->value_count; ++j) {
      const loom_low_allocation_assignment_t* assignment =
          loom_low_numbering_binding(state, use->binding_start + j);
      const uint16_t class_id = assignment->descriptor_reg_class_id;
      const loom_low_reg_class_t* reg_class =
          &state->descriptors->reg_classes[class_id];
      if (assignment->location_kind ==
              LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER &&
          reg_class->allocatable_count != 0 &&
          !loom_low_reg_class_uses_explicit_physical_registers(reg_class)) {
        state
            ->spaces[loom_low_numbering_space_index(state->descriptors,
                                                    class_id)]
            .active = true;
      }
    }
  }
  // Only finite linear inventories define a bounded numbering domain. An
  // alias set containing an unbounded physical class remains unchanged.
  for (uint16_t i = 0; i < state->descriptors->reg_class_count; ++i) {
    const loom_low_reg_class_t* reg_class = &state->descriptors->reg_classes[i];
    if (reg_class->allocatable_count == 0 &&
        iree_any_bit_set(reg_class->flags, LOOM_LOW_REG_CLASS_FLAG_PHYSICAL)) {
      state->spaces[loom_low_numbering_space_index(state->descriptors, i)]
          .active = false;
    }
  }
  for (uint16_t i = 0; i < state->descriptors->reg_class_count; ++i) {
    loom_low_numbering_space_t* space =
        &state->spaces[loom_low_numbering_space_index(state->descriptors, i)];
    if (space->active) {
      space->count = iree_max(space->count,
                              state->context->target_constraints
                                  ->max_assigned_location_end_by_reg_class[i]);
    }
  }
  for (uint32_t i = 0; i < state->space_count; ++i) {
    state->spaces[i].start = state->extent;
    state->extent += state->spaces[i].count;
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_numbering_analyze(
    const loom_low_allocation_numbering_context_t* context,
    iree_bitmap_t unavoidable_instruction_reg_classes,
    iree_arena_allocator_t* scratch_arena,
    loom_low_allocation_numbering_analysis_t* out_analysis) {
  *out_analysis = (loom_low_allocation_numbering_analysis_t){0};
  if (unavoidable_instruction_reg_classes.bit_count != 0) {
    IREE_ASSERT_EQ(
        unavoidable_instruction_reg_classes.bit_count,
        context->target_constraints->target->descriptor_set->reg_class_count);
    iree_bitmap_reset_all(unavoidable_instruction_reg_classes);
  }
  const loom_low_placement_preference_index_t* preferences =
      context->preferences;
  if (preferences->instruction_use_count == 0) {
    return iree_ok_status();
  }

  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(scratch_arena);
  loom_low_numbering_state_t state = {
      .context = context,
      .descriptors = context->target_constraints->target->descriptor_set,
  };
  iree_status_t status =
      loom_low_numbering_initialize_spaces(&state, scratch_arena);
  if (iree_status_is_ok(status) && state.extent != 0) {
    const uint64_t word_count =
        (uint64_t)state.extent * 4 + preferences->binding_count;
    const uint64_t workspace_size =
        (uint64_t)state.extent * sizeof(*state.blocks) +
        word_count * sizeof(uint32_t);
    if (workspace_size > IREE_HOST_SIZE_MAX) {
      status =
          iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                           "register analysis workspace exceeds host size");
    }
    void* workspace = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate(
          scratch_arena, (iree_host_size_t)workspace_size, &workspace);
    }
    if (iree_status_is_ok(status)) {
      memset(workspace, 0, (iree_host_size_t)workspace_size);
      state.blocks = workspace;
      uint32_t* cursor = (uint32_t*)(state.blocks + state.extent);
      state.scratch = cursor;
      cursor += state.extent;
      state.unit_blocks = cursor;
      cursor += state.extent;
      state.order = cursor;
      cursor += state.extent;
      state.bases = cursor;
      cursor += state.extent;
      state.locations = cursor;
      loom_low_numbering_build_blocks(&state);
      for (uint32_t i = 0; i < preferences->binding_count; ++i) {
        const loom_low_allocation_assignment_t* assignment =
            loom_low_numbering_binding(&state, i);
        state.locations[i] = loom_low_numbering_unit(
            &state, assignment->descriptor_reg_class_id,
            assignment->location_kind, assignment->location_base);
      }
      for (uint32_t i = 0; i < preferences->use_count; ++i) {
        const uint64_t cost =
            loom_low_numbering_score_use(&state, i, state.bases);
        const uint64_t unavoidable_cost =
            loom_low_numbering_unavoidable_use_cost(
                &state, i,
                i < preferences->instruction_use_count
                    ? unavoidable_instruction_reg_classes
                    : (iree_bitmap_t){0});
        if (i < preferences->instruction_use_count) {
          out_analysis->instruction_cost += cost;
          out_analysis->unavoidable_instruction_cost += unavoidable_cost;
        } else {
          out_analysis->scheduled_pair_cost += cost;
          out_analysis->unavoidable_scheduled_pair_cost += unavoidable_cost;
        }
      }
    }
  }
  iree_arena_checkpoint_restore(&checkpoint);
  return status;
}

iree_status_t loom_low_allocation_number_registers(
    const loom_low_allocation_numbering_context_t* context,
    iree_arena_allocator_t* scratch_arena) {
  const loom_low_placement_preference_index_t* preferences =
      context->preferences;
  if (preferences->instruction_use_count == 0) {
    return iree_ok_status();
  }
  loom_low_numbering_state_t state = {
      .context = context,
      .descriptors = context->target_constraints->target->descriptor_set,
  };
  IREE_RETURN_IF_ERROR(
      loom_low_numbering_initialize_spaces(&state, scratch_arena));
  if (state.extent == 0) {
    return iree_ok_status();
  }
  const uint64_t word_count = (uint64_t)state.extent * 6 + 1 +
                              (uint64_t)preferences->binding_count * 2 +
                              (uint64_t)preferences->use_count * 5;
  const uint64_t workspace_size =
      (uint64_t)state.extent * sizeof(*state.blocks) +
      word_count * sizeof(uint32_t);
  if (workspace_size > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "register numbering workspace exceeds host size");
  }
  void* workspace = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      scratch_arena, (iree_host_size_t)workspace_size, &workspace));
  memset(workspace, 0, (iree_host_size_t)workspace_size);
  state.blocks = workspace;
  uint32_t* cursor = (uint32_t*)(state.blocks + state.extent);
  state.scratch = cursor;
  cursor += state.extent;
  state.unit_blocks = cursor;
  cursor += state.extent;
  state.order = cursor;
  cursor += state.extent;
  state.bases = cursor;
  cursor += state.extent;
  state.trial_bases = cursor;
  cursor += state.extent;
  state.offsets = cursor;
  cursor += state.extent + 1;
  state.incidences = cursor;
  cursor += preferences->binding_count;
  state.locations = cursor;
  cursor += preferences->binding_count;
  state.limits = cursor;
  cursor += preferences->use_count;
  state.scores = cursor;
  cursor += preferences->use_count;
  state.trial_scores = cursor;
  cursor += preferences->use_count;
  state.queued = cursor;
  cursor += preferences->use_count;
  state.affected = cursor;
  loom_low_numbering_build_blocks(&state);
  loom_low_numbering_constrain_blocks(&state);
  loom_low_numbering_index_preferences(&state);
  if (loom_low_numbering_search(&state)) {
    loom_low_numbering_publish(&state);
  }
  return iree_ok_status();
}
