// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/edge_alias.h"

#include "loom/codegen/low/storage_relation.h"

static bool loom_low_allocation_edge_alias_unit_ranges_overlap(
    uint32_t lhs_offset, uint32_t lhs_count, uint32_t rhs_offset,
    uint32_t rhs_count, uint32_t* out_overlap_offset) {
  const uint64_t lhs_end = (uint64_t)lhs_offset + lhs_count;
  const uint64_t rhs_end = (uint64_t)rhs_offset + rhs_count;
  const uint32_t overlap_offset =
      lhs_offset > rhs_offset ? lhs_offset : rhs_offset;
  const uint64_t overlap_end = lhs_end < rhs_end ? lhs_end : rhs_end;
  if ((uint64_t)overlap_offset >= overlap_end) {
    return false;
  }
  *out_overlap_offset = overlap_offset;
  return true;
}

static bool
loom_low_allocation_edge_alias_destination_range_has_distinct_source(
    const loom_low_allocation_edge_alias_context_t* context,
    const loom_low_placement_relation_t* relation,
    uint32_t destination_unit_offset, uint32_t destination_unit_count) {
  const loom_low_placement_relation_range_t range =
      loom_low_placement_relation_range_for_value_ordinal(
          context->placement, relation->result_ordinal);
  for (uint32_t i = 0; i < range.count; ++i) {
    const loom_low_placement_relation_t* other_relation =
        &context->placement->relations[range.start + i];
    if (other_relation == relation ||
        other_relation->kind != LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE ||
        !loom_low_placement_cause_is_edge(other_relation->cause)) {
      continue;
    }
    uint32_t overlap_offset = 0;
    if (!loom_low_allocation_edge_alias_unit_ranges_overlap(
            destination_unit_offset, destination_unit_count,
            other_relation->result_unit_offset, other_relation->unit_count,
            &overlap_offset)) {
      continue;
    }
    const uint32_t relation_source_unit =
        relation->source_unit_offset +
        (overlap_offset - relation->result_unit_offset);
    const uint32_t other_source_unit =
        other_relation->source_unit_offset +
        (overlap_offset - other_relation->result_unit_offset);
    if (other_relation->source_ordinal != relation->source_ordinal ||
        other_source_unit != relation_source_unit) {
      return true;
    }
  }
  return false;
}

static loom_value_ordinal_t loom_low_allocation_edge_alias_storage_origin(
    const loom_low_allocation_edge_alias_context_t* context,
    loom_value_ordinal_t value_ordinal) {
  const loom_value_ordinal_t* origins =
      context->placement->tied_storage_origins_by_value_ordinal;
  return origins != NULL ? origins[value_ordinal] : value_ordinal;
}

static bool loom_low_allocation_edge_alias_has_indirect_observations(
    const loom_low_allocation_edge_alias_context_t* context,
    loom_value_ordinal_t storage_ordinal) {
  const loom_low_allocation_unit_liveness_t* unit_liveness =
      context->unit_liveness;
  return unit_liveness->observations.value_links != NULL &&
         unit_liveness->observations.value_links[storage_ordinal] != 0;
}

static iree_status_t
loom_low_allocation_edge_alias_storage_range_used_after_boundary(
    const loom_low_allocation_edge_alias_context_t* context,
    const loom_block_t* origin_block, uint64_t first_observation_ordinal,
    loom_value_id_t value_id, uint32_t unit_offset, uint32_t unit_count,
    bool* out_used_after) {
  *out_used_after = false;
  loom_consumption_region_query_t* region_query = NULL;
  IREE_RETURN_IF_ERROR(context->consumption_query(
      context->user_data, origin_block->parent_region, &region_query));
  const loom_value_ordinal_t ordinal = loom_module_value_ordinal_scratch_lookup(
      context->placement->module, value_id);
  const loom_value_ordinal_t storage_ordinal =
      loom_low_allocation_edge_alias_storage_origin(context, ordinal);
  const loom_value_id_t storage_value_id =
      loom_low_placement_value_id(context->placement, storage_ordinal);
  loom_consumption_use_after_query_t use_after_query = {0};
  IREE_RETURN_IF_ERROR(loom_consumption_use_after_query_prepare(
      region_query, origin_block, first_observation_ordinal, storage_value_id,
      loom_low_allocation_edge_alias_has_indirect_observations(context,
                                                               storage_ordinal)
          ? LOOM_CONSUMPTION_QUERY_FLAG_INDIRECT_OBSERVATIONS
          : 0,
      &use_after_query));
  const loom_low_allocation_unit_liveness_t* unit_liveness =
      context->unit_liveness;
  const uint32_t value_count = context->placement->value_count;
  for (uint32_t node = storage_ordinal + 1; node != 0;) {
    if (node > value_count) {
      const loom_low_allocation_decomposed_use_t* observation =
          &unit_liveness->observations.entries[node - value_count - 1];
      node = observation->next_node;
      if (loom_consumption_use_after_query_observes_operation(
              &use_after_query,
              loom_liveness_operation_at(context->liveness,
                                         observation->operation_index)
                  ->op)) {
        *out_used_after = true;
        return iree_ok_status();
      }
      continue;
    }
    const loom_value_ordinal_t member = node - 1;
    node = unit_liveness->observations.value_links != NULL
               ? unit_liveness->observations.value_links[member]
               : 0;
    const loom_value_t* value = loom_module_value(
        context->placement->module,
        loom_low_placement_value_id(context->placement, member));
    const loom_use_t* use = NULL;
    loom_value_for_each_use(value, use) {
      const loom_op_t* use_op = loom_use_user_op(*use);
      const uint16_t operand_index = loom_use_operand_index(*use);
      if (loom_low_storage_operand_may_read_unit_range(
              context->placement->module, use_op, operand_index, unit_offset,
              unit_count) &&
          loom_consumption_use_after_query_observes_operation(&use_after_query,
                                                              use_op)) {
        *out_used_after = true;
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_edge_alias_storage_range_used_after_operation(
    const loom_low_allocation_edge_alias_context_t* context,
    const loom_op_t* operation, loom_value_ordinal_t value_ordinal,
    uint32_t unit_offset, uint32_t unit_count, bool* out_used_after) {
  return loom_low_allocation_edge_alias_storage_range_used_after_boundary(
      context, operation->parent_block, operation->block_ordinal + 1,
      loom_low_placement_value_id(context->placement, value_ordinal),
      unit_offset, unit_count, out_used_after);
}

static iree_status_t
loom_low_allocation_edge_alias_destination_may_survive_candidate_definition(
    const loom_low_allocation_edge_alias_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* counterpart,
    uint32_t destination_unit_offset, uint32_t destination_unit_count,
    bool* out_may_survive) {
  *out_may_survive = false;
  const loom_value_id_t destination_value_id =
      loom_low_placement_value_id(context->placement, relation->result_ordinal);
  const loom_value_t* destination_value =
      loom_module_value(context->placement->module, destination_value_id);
  if (!loom_value_is_block_arg(destination_value)) {
    return iree_ok_status();
  }
  const loom_value_id_t candidate_value_id =
      interval->value_id == destination_value_id ? counterpart->value_id
                                                 : interval->value_id;
  const loom_value_t* candidate_value =
      loom_module_value(context->placement->module, candidate_value_id);
  if (loom_value_is_block_arg(candidate_value)) {
    // A join defines its argument before any operation in that block. Its
    // eventual backedge cannot overwrite an old destination value still live
    // at this definition point. Structured regions retain their edge handoffs.
    if (candidate_value_id != destination_value_id &&
        loom_value_def_block(candidate_value)->parent_region ==
            loom_value_def_block(destination_value)->parent_region) {
      const loom_liveness_interval_t* candidate_interval =
          interval->value_id == destination_value_id
              ? loom_liveness_interval_for_value_ordinal(
                    context->liveness, relation->source_ordinal)
              : interval;
      const uint32_t definition_point = candidate_interval->definition_point;
      const loom_liveness_segment_range_t segments =
          loom_liveness_segment_range_for_value_ordinal(
              context->liveness, relation->result_ordinal);
      *out_may_survive = loom_liveness_segment_range_contains(
          context->liveness->segments, segments, definition_point);
      if (!*out_may_survive &&
          loom_low_allocation_edge_alias_has_indirect_observations(
              context, loom_low_allocation_edge_alias_storage_origin(
                           context, relation->result_ordinal))) {
        return loom_low_allocation_edge_alias_storage_range_used_after_boundary(
            context, loom_value_def_block(candidate_value),
            /*first_observation_ordinal=*/0, destination_value_id,
            destination_unit_offset, destination_unit_count, out_may_survive);
      }
    }
    return iree_ok_status();
  }
  const loom_op_t* candidate_op = loom_value_def_op(candidate_value);
  if (candidate_op == NULL || candidate_op->parent_block == NULL) {
    return iree_ok_status();
  }
  return loom_low_allocation_edge_alias_storage_range_used_after_boundary(
      context, candidate_op->parent_block, candidate_op->block_ordinal + 1,
      destination_value_id, destination_unit_offset, destination_unit_count,
      out_may_survive);
}

iree_status_t loom_low_allocation_edge_alias_allows_counterpart_overlap(
    const loom_low_allocation_edge_alias_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* counterpart,
    uint32_t destination_unit_offset, uint32_t destination_unit_count,
    bool* out_allows_overlap) {
  *out_allows_overlap = false;
  if (iree_any_bit_set(relation->flags,
                       LOOM_LOW_PLACEMENT_RELATION_FLAG_IDENTITY_EDGE)) {
    *out_allows_overlap = true;
    return iree_ok_status();
  }
  if (relation->kind != LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE ||
      !loom_low_placement_relation_can_alias(relation) ||
      !loom_low_placement_cause_is_edge(relation->cause)) {
    return iree_ok_status();
  }
  bool destination_may_survive_candidate_definition = false;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_edge_alias_destination_may_survive_candidate_definition(
          context, interval, relation, counterpart, destination_unit_offset,
          destination_unit_count,
          &destination_may_survive_candidate_definition));
  if (destination_may_survive_candidate_definition) {
    return iree_ok_status();
  }
  switch (relation->cause) {
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_YIELD:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_CONDITION:
      *out_allows_overlap = true;
      return iree_ok_status();
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_LOOP_ENTRY: {
      if (!loom_low_allocation_edge_alias_destination_range_has_distinct_source(
              context, relation, destination_unit_offset,
              destination_unit_count)) {
        *out_allows_overlap = true;
        return iree_ok_status();
      }
      bool source_used_after = false;
      const loom_value_id_t source_value_id = loom_low_placement_value_id(
          context->placement, relation->source_ordinal);
      const uint32_t source_unit_offset =
          relation->source_unit_offset +
          (destination_unit_offset - relation->result_unit_offset);
      if (relation->cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_LOOP_ENTRY) {
        // Initial loop operands are defined outside the entered regions. The
        // physical lifetime at entry already includes nested captures and
        // implicit upper-bound/step reads, but excludes entry-only operands.
        // A query after the enclosing operation would miss all those reads.
        source_used_after =
            loom_low_allocation_unit_liveness_storage_component_live_at_point(
                context->unit_liveness, context->liveness, context->placement,
                relation->source_ordinal, source_unit_offset,
                destination_unit_count, relation->write_point);
      } else {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_edge_alias_storage_range_used_after_boundary(
                context, relation->op->parent_block,
                relation->op->block_ordinal + 1, source_value_id,
                source_unit_offset, destination_unit_count,
                &source_used_after));
      }
      *out_allows_overlap = !source_used_after;
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}
