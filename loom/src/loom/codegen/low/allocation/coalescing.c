// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/coalescing.h"

#include "loom/codegen/low/allocation/edge_alias.h"
#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/codegen/low/storage_relation.h"

static bool loom_low_allocation_coalescing_value_ordinal_for_value(
    const loom_low_allocation_coalescing_context_t* context,
    loom_value_id_t value_id, loom_value_ordinal_t* out_value_ordinal) {
  return loom_low_allocation_assignment_map_value_ordinal_for_value(
      context->assignment_map, value_id, out_value_ordinal);
}

static const loom_low_allocation_assignment_t*
loom_low_allocation_coalescing_current_assignment_for_value_ordinal(
    const loom_low_allocation_coalescing_context_t* context,
    loom_value_ordinal_t value_ordinal) {
  return loom_low_allocation_assignment_map_assignment_for_value_ordinal(
      context->assignment_map, value_ordinal, NULL);
}

static iree_status_t loom_low_allocation_coalescing_assignment_index_for_value(
    const loom_low_allocation_coalescing_context_t* context,
    loom_value_id_t value_id, uint32_t* out_assignment_index) {
  return loom_low_allocation_assignment_map_require_assignment_for_value(
      context->assignment_map, value_id, out_assignment_index, NULL);
}

static iree_status_t loom_low_allocation_coalescing_value_ordinal_for_interval(
    const loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval,
    loom_value_ordinal_t* out_value_ordinal) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_coalescing_value_ordinal_for_value(
          context, interval->value_id, &value_ordinal)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation interval value %u is outside the local value domain",
        (unsigned)interval->value_id);
  }
  *out_value_ordinal = value_ordinal;
  return iree_ok_status();
}

static const loom_low_placement_relation_t*
loom_low_allocation_coalescing_first_placement_relation(
    const loom_low_allocation_coalescing_context_t* context,
    loom_value_ordinal_t result_ordinal, loom_low_placement_cause_t cause) {
  const loom_low_placement_relation_range_t range =
      loom_low_placement_relation_range_for_value_ordinal(context->placement,
                                                          result_ordinal);
  for (uint32_t i = 0; i < range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[range.start + i];
    if (relation->cause == cause) {
      return relation;
    }
  }
  return NULL;
}

static bool loom_low_allocation_coalescing_unit_ranges_overlap(
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

static bool loom_low_allocation_coalescing_compose_tied_concat_source_relation(
    const loom_low_placement_relation_t* tied_relation,
    const loom_low_placement_relation_t* concat_relation,
    loom_low_placement_relation_t* out_relation) {
  // Carries a concat source slice backward through an exact tied-result alias
  // so the operand can reserve the eventual aligned aggregate.
  if (tied_relation->cause != LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT ||
      concat_relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT ||
      !iree_all_bits_set(
          tied_relation->flags,
          LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD |
              LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE)) {
    return false;
  }
  uint32_t overlap_offset = 0;
  if (!loom_low_allocation_coalescing_unit_ranges_overlap(
          tied_relation->result_unit_offset, tied_relation->unit_count,
          concat_relation->source_unit_offset, concat_relation->unit_count,
          &overlap_offset)) {
    return false;
  }
  const uint64_t tied_end =
      (uint64_t)tied_relation->result_unit_offset + tied_relation->unit_count;
  const uint64_t concat_end = (uint64_t)concat_relation->source_unit_offset +
                              concat_relation->unit_count;
  const uint32_t overlap_count =
      (uint32_t)((tied_end < concat_end ? tied_end : concat_end) -
                 overlap_offset);
  *out_relation = *concat_relation;
  out_relation->source_ordinal = tied_relation->source_ordinal;
  out_relation->source_unit_offset =
      tied_relation->source_unit_offset +
      (overlap_offset - tied_relation->result_unit_offset);
  out_relation->result_unit_offset =
      concat_relation->result_unit_offset +
      (overlap_offset - concat_relation->source_unit_offset);
  out_relation->unit_count = overlap_count;
  return true;
}

// Returns true when any unit in |unit_offset, unit_count| retains concrete
// storage across |program_point| in the scheduled allocation order.
static bool loom_low_allocation_coalescing_value_units_live_at_point(
    const loom_low_allocation_coalescing_context_t* context,
    loom_value_ordinal_t value_ordinal, uint32_t unit_offset,
    uint32_t unit_count, uint32_t program_point) {
  const loom_liveness_interval_t* interval =
      loom_liveness_interval_for_value_ordinal(context->liveness,
                                               value_ordinal);
  if (interval == NULL || unit_count == 0) {
    return false;
  }
  IREE_ASSERT_LE(unit_offset, interval->unit_count);
  IREE_ASSERT_LE(unit_count, interval->unit_count - unit_offset);

  bool value_live_at_point = false;
  const loom_low_allocation_unit_liveness_t* unit_liveness =
      context->search_context->unit_liveness;
  const loom_liveness_segment_range_t segments =
      loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
          unit_liveness, context->liveness, value_ordinal);
  if (segments.count == 0) {
    // Empty storage segments require the conservative linear check below. The
    // per-unit end points include storage continuation through tied results
    // and decomposed edge payloads that semantic SSA segments do not encode.
    value_live_at_point = interval->start_point <= program_point;
  } else {
    IREE_ASSERT_LE((uint64_t)segments.start + segments.count,
                   context->liveness->segment_count);
    for (uint32_t i = 0; i < segments.count; ++i) {
      const loom_liveness_segment_t* segment =
          &context->liveness->segments[segments.start + i];
      if (segment->start_point <= program_point &&
          program_point < segment->end_point) {
        value_live_at_point = true;
        break;
      }
      if (segment->start_point > program_point) {
        break;
      }
    }
  }
  if (value_live_at_point) {
    const uint32_t end_point_start =
        loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
            unit_liveness, context->liveness, value_ordinal);
    IREE_ASSERT_NE(end_point_start, UINT32_MAX);
    IREE_ASSERT_LE((uint64_t)end_point_start + unit_offset + unit_count,
                   unit_liveness->point_count);
    for (uint32_t i = 0; i < unit_count; ++i) {
      if (unit_liveness->end_points[end_point_start + unit_offset + i] >
          program_point) {
        return true;
      }
    }
  }

  // A tied result continues to own its operand's concrete units after the
  // operand SSA value is consumed. Follow those unit mappings so copy and
  // structural coalescing cannot classify continued storage as a dead alias.
  // Tied-result relations point from an operand definition to its newer SSA
  // result definition, making this recursive traversal acyclic.
  const loom_low_placement_relation_range_t tied_range =
      loom_low_placement_relation_range_for_source_value_ordinal(
          context->placement, value_ordinal);
  for (uint32_t i = 0; i < tied_range.count; ++i) {
    const uint32_t relation_index =
        context->placement
            ->relation_indices_by_source_ordinal[tied_range.start + i];
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[relation_index];
    if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT) {
      continue;
    }
    uint32_t overlap_offset = 0;
    if (!loom_low_allocation_coalescing_unit_ranges_overlap(
            unit_offset, unit_count, relation->source_unit_offset,
            relation->unit_count, &overlap_offset)) {
      continue;
    }
    const uint64_t query_end = (uint64_t)unit_offset + unit_count;
    const uint64_t relation_end =
        (uint64_t)relation->source_unit_offset + relation->unit_count;
    const uint32_t overlap_count =
        (uint32_t)((query_end < relation_end ? query_end : relation_end) -
                   overlap_offset);
    const uint32_t result_unit_offset =
        relation->result_unit_offset +
        (overlap_offset - relation->source_unit_offset);
    if (loom_low_allocation_coalescing_value_units_live_at_point(
            context, relation->result_ordinal, result_unit_offset,
            overlap_count, program_point)) {
      return true;
    }
  }
  return false;
}

// Edge placement may make counterpart values overlap in the linear interval
// space even when the edge handoff makes them mutually exclusive. Reuse is safe
// for yield edges, where the source is consumed by the edge itself. Branch and
// initial scf.for iter_arg edges can forward an outer value that remains live
// after the destination block/loop argument. In those cases overlap is only
// ignorable when the source has no use reachable after the edge handoff.
static iree_status_t
loom_low_allocation_coalescing_can_ignore_relation_counterpart_conflict(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* counterpart,
    uint32_t destination_unit_offset, uint32_t destination_unit_count,
    bool* out_can_ignore) {
  *out_can_ignore = false;
  if (!loom_low_allocation_live_range_assignment_overlaps_interval(counterpart,
                                                                   interval)) {
    return iree_ok_status();
  }
  const loom_low_allocation_edge_alias_context_t edge_alias_context = {
      .placement = context->placement,
      .consumption_query = context->consumption_query,
      .user_data = context->user_data,
  };
  bool edge_allows_overlap = false;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_edge_alias_allows_counterpart_overlap(
          &edge_alias_context, interval, relation, counterpart,
          destination_unit_offset, destination_unit_count,
          &edge_allows_overlap));
  if (edge_allows_overlap) {
    *out_can_ignore = true;
    return iree_ok_status();
  }
  *out_can_ignore = !loom_low_allocation_live_range_values_overlap(
      context->liveness, interval->value_id, interval->start_point,
      interval->end_point, counterpart->value_id, counterpart->start_point,
      counterpart->end_point);
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_coalescing_select_ignored_counterpart_value(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* counterpart,
    const loom_value_id_t* counterpart_value_id,
    uint32_t destination_unit_offset, uint32_t destination_unit_count,
    const loom_value_id_t** out_ignored_value_ids,
    uint16_t* out_ignored_value_count) {
  *out_ignored_value_ids = NULL;
  *out_ignored_value_count = 0;
  bool can_ignore_counterpart = false;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_coalescing_can_ignore_relation_counterpart_conflict(
          context, interval, relation, counterpart, destination_unit_offset,
          destination_unit_count, &can_ignore_counterpart));
  if (can_ignore_counterpart) {
    *out_ignored_value_ids = counterpart_value_id;
    *out_ignored_value_count = 1;
  }
  return iree_ok_status();
}

static const loom_low_placement_relation_t*
loom_low_allocation_coalescing_transfer_relation_for_result_ordinal(
    const loom_low_allocation_coalescing_context_t* context,
    loom_value_ordinal_t result_ordinal) {
  const loom_low_placement_relation_range_t range =
      loom_low_placement_relation_range_for_value_ordinal(context->placement,
                                                          result_ordinal);
  for (uint32_t i = 0; i < range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[range.start + i];
    if (relation->cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY ||
        relation->cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_MOVE) {
      return relation;
    }
  }
  return NULL;
}

// Returns whether the semantic SSA value itself has a direct use after
// |consuming_op|. Tied-result successors are distinct SSA identities and are
// intentionally not followed.
static iree_status_t
loom_low_allocation_coalescing_value_units_have_direct_use_after_clobber(
    const loom_low_allocation_coalescing_context_t* context,
    const loom_op_t* consuming_op, loom_value_ordinal_t value_ordinal,
    uint32_t unit_offset, uint32_t unit_count, bool* out_has_use) {
  *out_has_use = false;
  const loom_value_id_t value_id =
      loom_low_placement_value_id(context->placement, value_ordinal);
  const loom_low_schedule_node_t* consuming_node =
      context->schedule != NULL
          ? loom_low_schedule_node_for_op(context->schedule, consuming_op)
          : NULL;
  IREE_ASSERT(context->schedule == NULL || consuming_node != NULL);
  loom_consumption_region_query_t* region_query = NULL;
  loom_consumption_use_after_query_t use_after_query = {0};
  bool use_after_query_ready = false;
  const loom_value_t* value =
      loom_module_value(context->placement->module, value_id);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    const loom_op_t* use_op = loom_use_user_op(*use);
    const uint16_t operand_index = loom_use_operand_index(*use);
    if (!loom_low_storage_operand_may_read_unit_range(
            context->placement->module, use_op, operand_index, unit_offset,
            unit_count)) {
      continue;
    }

    // Pressure scheduling can reorder independent operations within a block,
    // so same-block source order does not establish whether a use observes the
    // pre-clobber value. Cross-block order remains a dynamic CFG question.
    bool use_after_clobber = use_op == consuming_op;
    if (!use_after_clobber &&
        use_op->parent_block == consuming_op->parent_block) {
      if (context->schedule != NULL) {
        const loom_low_schedule_node_t* use_node =
            loom_low_schedule_node_for_op(context->schedule, use_op);
        IREE_ASSERT(use_node != NULL);
        use_after_clobber =
            use_node->scheduled_ordinal >= consuming_node->scheduled_ordinal;
      } else {
        use_after_clobber =
            use_op->block_ordinal >= consuming_op->block_ordinal;
      }
    } else if (!use_after_clobber) {
      if (!use_after_query_ready) {
        IREE_RETURN_IF_ERROR(context->consumption_query(
            context->user_data, consuming_op->parent_block->parent_region,
            &region_query));
        IREE_RETURN_IF_ERROR(loom_consumption_use_after_query_prepare(
            region_query, consuming_op, value_id, &use_after_query));
        use_after_query_ready = true;
      }
      use_after_clobber =
          loom_consumption_use_after_query_contains(&use_after_query, *use);
    }
    if (use_after_clobber) {
      *out_has_use = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

// Returns whether the concrete units owned by |value_ordinal| have a use after
// |consuming_op|. Tied results continue the same storage under a new SSA
// identity and are followed transitively.
static iree_status_t
loom_low_allocation_coalescing_storage_units_have_use_after_clobber(
    const loom_low_allocation_coalescing_context_t* context,
    const loom_op_t* consuming_op, loom_value_ordinal_t value_ordinal,
    uint32_t unit_offset, uint32_t unit_count, bool* out_has_use) {
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_coalescing_value_units_have_direct_use_after_clobber(
          context, consuming_op, value_ordinal, unit_offset, unit_count,
          out_has_use));
  if (*out_has_use) {
    return iree_ok_status();
  }

  // A tied result continues to own its operand's concrete units. Follow exact
  // unit mappings so a later use of the newer SSA value still preserves the
  // original copy source. Tied-result relations follow SSA definition order
  // and cannot form cycles.
  const loom_low_placement_relation_range_t tied_range =
      loom_low_placement_relation_range_for_source_value_ordinal(
          context->placement, value_ordinal);
  for (uint32_t i = 0; i < tied_range.count; ++i) {
    const uint32_t relation_index =
        context->placement
            ->relation_indices_by_source_ordinal[tied_range.start + i];
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[relation_index];
    if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT) {
      continue;
    }
    uint32_t overlap_offset = 0;
    if (!loom_low_allocation_coalescing_unit_ranges_overlap(
            unit_offset, unit_count, relation->source_unit_offset,
            relation->unit_count, &overlap_offset)) {
      continue;
    }
    const uint64_t query_end = (uint64_t)unit_offset + unit_count;
    const uint64_t relation_end =
        (uint64_t)relation->source_unit_offset + relation->unit_count;
    const uint32_t overlap_count =
        (uint32_t)((query_end < relation_end ? query_end : relation_end) -
                   overlap_offset);
    const uint32_t result_unit_offset =
        relation->result_unit_offset +
        (overlap_offset - relation->source_unit_offset);
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_storage_units_have_use_after_clobber(
            context, consuming_op, relation->result_ordinal, result_unit_offset,
            overlap_count, out_has_use));
    if (*out_has_use) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_coalescing_transfer_source_live_at_tied_definition(
    const loom_low_allocation_coalescing_context_t* context,
    const loom_low_placement_relation_t* tied_relation,
    const loom_low_placement_relation_t* copy_relation,
    loom_value_id_t* out_copy_source_id, bool* out_copy_source_live) {
  *out_copy_source_id = LOOM_VALUE_ID_INVALID;
  *out_copy_source_live = false;
  if (!copy_relation) {
    return iree_ok_status();
  }
  *out_copy_source_id = loom_low_placement_value_id(
      context->placement, copy_relation->source_ordinal);

  uint32_t overlap_offset = 0;
  if (!loom_low_allocation_coalescing_unit_ranges_overlap(
          copy_relation->result_unit_offset, copy_relation->unit_count,
          tied_relation->source_unit_offset, tied_relation->unit_count,
          &overlap_offset)) {
    return iree_ok_status();
  }
  const uint64_t copy_end =
      (uint64_t)copy_relation->result_unit_offset + copy_relation->unit_count;
  const uint64_t tied_end =
      (uint64_t)tied_relation->source_unit_offset + tied_relation->unit_count;
  const uint32_t overlap_count =
      (uint32_t)((copy_end < tied_end ? copy_end : tied_end) - overlap_offset);
  const uint32_t source_unit_offset =
      copy_relation->source_unit_offset +
      (overlap_offset - copy_relation->result_unit_offset);
  const loom_liveness_interval_t* tied_result_interval =
      loom_liveness_interval_for_value_ordinal(context->liveness,
                                               tied_relation->result_ordinal);
  IREE_ASSERT_ARGUMENT(tied_result_interval);
  if (!loom_low_allocation_coalescing_value_units_live_at_point(
          context, copy_relation->source_ordinal, source_unit_offset,
          overlap_count, tied_result_interval->start_point)) {
    return iree_ok_status();
  }
  return loom_low_allocation_coalescing_storage_units_have_use_after_clobber(
      context, tied_relation->op, copy_relation->source_ordinal,
      source_unit_offset, overlap_count, out_copy_source_live);
}

static bool loom_low_allocation_coalescing_storage_alias_relation(
    const loom_low_placement_relation_t* relation) {
  if (!loom_low_placement_relation_can_alias(relation)) {
    return false;
  }
  switch (relation->cause) {
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_MOVE:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT:
      return true;
    default:
      return false;
  }
}

static bool loom_low_allocation_coalescing_value_id_is_listed(
    const loom_value_id_t* value_ids, uint16_t value_count,
    loom_value_id_t value_id) {
  for (uint16_t i = 0; i < value_count; ++i) {
    if (value_ids[i] == value_id) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_allocation_coalescing_append_unique_value_id(
    loom_value_id_t* value_ids, uint16_t value_capacity, uint16_t* value_count,
    loom_value_id_t value_id) {
  if (loom_low_allocation_coalescing_value_id_is_listed(value_ids, *value_count,
                                                        value_id)) {
    return iree_ok_status();
  }
  if (*value_count >= value_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low allocation ignored-value list exhausted");
  }
  value_ids[(*value_count)++] = value_id;
  return iree_ok_status();
}

static bool loom_low_allocation_coalescing_try_append_unique_value_id(
    loom_value_id_t* value_ids, uint16_t value_capacity, uint16_t* value_count,
    loom_value_id_t value_id) {
  if (loom_low_allocation_coalescing_value_id_is_listed(value_ids, *value_count,
                                                        value_id)) {
    return true;
  }
  if (*value_count >= value_capacity) {
    return false;
  }
  value_ids[(*value_count)++] = value_id;
  return true;
}

static iree_status_t
loom_low_allocation_coalescing_append_ignored_counterpart_value(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* counterpart,
    const loom_value_id_t* counterpart_value_id,
    uint32_t destination_unit_offset, uint32_t destination_unit_count,
    loom_value_id_t* ignored_value_ids, uint16_t ignored_value_capacity,
    uint16_t* ignored_value_count) {
  const loom_value_id_t* selected_value_ids = NULL;
  uint16_t selected_value_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_coalescing_select_ignored_counterpart_value(
          context, interval, relation, counterpart, counterpart_value_id,
          destination_unit_offset, destination_unit_count, &selected_value_ids,
          &selected_value_count));
  for (uint16_t i = 0; i < selected_value_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_coalescing_append_unique_value_id(
        ignored_value_ids, ignored_value_capacity, ignored_value_count,
        selected_value_ids[i]));
  }
  return iree_ok_status();
}

static bool loom_low_allocation_coalescing_exact_storage_alias_relation(
    const loom_low_allocation_coalescing_context_t* context,
    const loom_low_placement_relation_t* relation) {
  if (!loom_low_placement_relation_can_alias(relation) ||
      relation->kind != LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE ||
      relation->result_unit_offset != 0 || relation->source_unit_offset != 0) {
    return false;
  }
  const loom_liveness_interval_t* result_interval =
      loom_liveness_interval_for_value_ordinal(context->liveness,
                                               relation->result_ordinal);
  const loom_liveness_interval_t* source_interval =
      loom_liveness_interval_for_value_ordinal(context->liveness,
                                               relation->source_ordinal);
  return result_interval && source_interval &&
         loom_liveness_value_class_equal(result_interval->value_class,
                                         source_interval->value_class) &&
         relation->unit_count == result_interval->unit_count &&
         relation->unit_count == source_interval->unit_count;
}

static iree_status_t
loom_low_allocation_coalescing_try_append_dead_exact_storage_alias(
    loom_low_allocation_coalescing_context_t* context,
    const loom_op_t* anchor_op, uint32_t clobber_point,
    loom_value_ordinal_t alias_ordinal, loom_value_id_t* ignored_value_ids,
    uint16_t ignored_value_capacity, uint16_t* ignored_value_count) {
  const loom_value_id_t alias_id =
      loom_low_placement_value_id(context->placement, alias_ordinal);
  if (loom_low_allocation_coalescing_value_id_is_listed(
          ignored_value_ids, *ignored_value_count, alias_id)) {
    return iree_ok_status();
  }
  const loom_liveness_interval_t* alias_interval =
      loom_liveness_interval_for_value_ordinal(context->liveness,
                                               alias_ordinal);
  if (alias_interval == NULL) {
    return iree_ok_status();
  }
  bool has_use_after_clobber = false;
  if (loom_low_allocation_coalescing_value_units_live_at_point(
          context, alias_ordinal, 0, alias_interval->unit_count,
          clobber_point)) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_value_units_have_direct_use_after_clobber(
            context, anchor_op, alias_ordinal, 0, alias_interval->unit_count,
            &has_use_after_clobber));
  }
  if (!has_use_after_clobber) {
    loom_low_allocation_coalescing_try_append_unique_value_id(
        ignored_value_ids, ignored_value_capacity, ignored_value_count,
        alias_id);
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_coalescing_collect_dead_exact_storage_aliases(
    loom_low_allocation_coalescing_context_t* context,
    const loom_op_t* anchor_op, uint32_t clobber_point,
    loom_value_id_t* ignored_value_ids, uint16_t ignored_value_capacity,
    uint16_t* ignored_value_count) {
  for (uint16_t i = 0; i < *ignored_value_count; ++i) {
    loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
    if (!loom_low_allocation_coalescing_value_ordinal_for_value(
            context, ignored_value_ids[i], &value_ordinal)) {
      continue;
    }

    const loom_low_placement_relation_range_t result_range =
        loom_low_placement_relation_range_for_value_ordinal(context->placement,
                                                            value_ordinal);
    for (uint32_t j = 0; j < result_range.count; ++j) {
      const loom_low_placement_relation_t* relation =
          &context->placement->relations[result_range.start + j];
      if (!loom_low_allocation_coalescing_exact_storage_alias_relation(
              context, relation)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_coalescing_try_append_dead_exact_storage_alias(
              context, anchor_op, clobber_point, relation->source_ordinal,
              ignored_value_ids, ignored_value_capacity, ignored_value_count));
    }

    const loom_low_placement_relation_range_t source_range =
        loom_low_placement_relation_range_for_source_value_ordinal(
            context->placement, value_ordinal);
    for (uint32_t j = 0; j < source_range.count; ++j) {
      const uint32_t relation_index =
          context->placement
              ->relation_indices_by_source_ordinal[source_range.start + j];
      const loom_low_placement_relation_t* relation =
          &context->placement->relations[relation_index];
      if (!loom_low_allocation_coalescing_exact_storage_alias_relation(
              context, relation)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_coalescing_try_append_dead_exact_storage_alias(
              context, anchor_op, clobber_point, relation->result_ordinal,
              ignored_value_ids, ignored_value_capacity, ignored_value_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_coalescing_collect_tied_storage_aliases(
    loom_low_allocation_coalescing_context_t* context,
    const loom_low_placement_relation_t* tied_relation,
    loom_value_ordinal_t tied_operand_ordinal,
    loom_value_id_t* ignored_value_ids, uint16_t ignored_value_capacity,
    uint16_t* ignored_value_count) {
  const loom_low_placement_relation_range_t range =
      loom_low_placement_relation_range_for_value_ordinal(context->placement,
                                                          tied_operand_ordinal);
  const loom_liveness_interval_t* tied_result_interval =
      loom_liveness_interval_for_value_ordinal(context->liveness,
                                               tied_relation->result_ordinal);
  IREE_ASSERT_ARGUMENT(tied_result_interval);
  for (uint32_t i = 0; i < range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[range.start + i];
    if (!loom_low_allocation_coalescing_storage_alias_relation(relation)) {
      continue;
    }
    const loom_value_id_t source_value_id = loom_low_placement_value_id(
        context->placement, relation->source_ordinal);
    const loom_liveness_interval_t* source_interval =
        loom_liveness_interval_for_value_ordinal(context->liveness,
                                                 relation->source_ordinal);
    if (source_interval != NULL &&
        !loom_low_allocation_coalescing_value_units_live_at_point(
            context, relation->source_ordinal, 0, source_interval->unit_count,
            tied_result_interval->start_point)) {
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_coalescing_append_unique_value_id(
              ignored_value_ids, ignored_value_capacity, ignored_value_count,
              source_value_id));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_coalescing_collect_tied_concat_reservations(
    loom_low_allocation_coalescing_context_t* context,
    const loom_low_placement_relation_t* tied_relation,
    const loom_low_allocation_assignment_t* tied_operand_assignment,
    loom_value_id_t* ignored_value_ids, uint16_t ignored_value_capacity,
    uint16_t* ignored_value_count) {
  const loom_low_placement_relation_range_t source_range =
      loom_low_placement_relation_range_for_source_value_ordinal(
          context->placement, tied_relation->result_ordinal);
  for (uint32_t i = 0; i < source_range.count; ++i) {
    const uint32_t relation_index =
        context->placement
            ->relation_indices_by_source_ordinal[source_range.start + i];
    const loom_low_placement_relation_t* concat_relation =
        &context->placement->relations[relation_index];
    loom_low_placement_relation_t composed_relation;
    if (!loom_low_allocation_coalescing_compose_tied_concat_source_relation(
            tied_relation, concat_relation, &composed_relation)) {
      continue;
    }
    const loom_low_allocation_assignment_t* concat_assignment =
        loom_low_allocation_coalescing_current_assignment_for_value_ordinal(
            context, concat_relation->result_ordinal);
    if (concat_assignment == NULL ||
        !loom_low_allocation_storage_placement_relation_satisfied(
            context->search_context->descriptor_set, &composed_relation,
            concat_assignment, tied_operand_assignment)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_allocation_coalescing_append_unique_value_id(
        ignored_value_ids, ignored_value_capacity, ignored_value_count,
        concat_assignment->value_id));
  }
  return iree_ok_status();
}

static bool loom_low_allocation_coalescing_assignment_unit_span_fits(
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_offset,
    uint32_t unit_count) {
  return unit_offset <= assignment->location_count &&
         unit_count <= assignment->location_count - unit_offset &&
         assignment->location_base <= UINT32_MAX - unit_offset;
}

static bool loom_low_allocation_coalescing_align_up_u32(uint32_t value,
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

static iree_status_t loom_low_allocation_coalescing_append_interval_at_location(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval, uint16_t descriptor_reg_class_id,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t unit_count, const loom_value_id_t* ignored_value_ids,
    uint16_t ignored_value_count,
    const loom_value_id_t* ignored_storage_lease_value_ids,
    uint16_t ignored_storage_lease_value_count, bool* out_assigned) {
  *out_assigned = false;
  if (location_base > UINT32_MAX - unit_count) {
    return iree_ok_status();
  }
  loom_low_allocation_class_capacity_t capacity = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_interval_capacity(
      context->target_constraints, interval, &capacity));
  if (!loom_low_allocation_storage_reg_classes_share(
          context->search_context->descriptor_set,
          capacity.descriptor_reg_class_id, descriptor_reg_class_id)) {
    return iree_ok_status();
  }
  if (!loom_low_allocation_target_constraints_location_range_fits_capacity(
          &capacity, location_kind, location_base, unit_count)) {
    return iree_ok_status();
  }
  const uint32_t alignment =
      loom_low_allocation_live_range_interval_alignment(interval);
  if (location_base % alignment != 0) {
    return iree_ok_status();
  }
  // Coalesced assignments commit through the normal append path, which records
  // release actions for storage leases that can be legally released here.
  if (loom_low_allocation_search_location_conflicts(
          context->search_context, interval, descriptor_reg_class_id,
          location_kind, location_base, unit_count, ignored_value_ids,
          ignored_value_count, ignored_storage_lease_value_ids,
          ignored_storage_lease_value_count,
          LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED)) {
    return iree_ok_status();
  }

  const loom_low_allocation_assignment_t assignment = {
      .value_id = interval->value_id,
      .value_class = interval->value_class,
      .descriptor_reg_class_id = descriptor_reg_class_id,
      .start_point = interval->start_point,
      .end_point =
          loom_low_allocation_live_range_interval_storage_end_point(interval),
      .unit_count = interval->unit_count,
      .location_kind = location_kind,
      .location_base = location_base,
      .location_count = unit_count,
  };
  IREE_RETURN_IF_ERROR(context->append_assignment(
      context->user_data, &assignment, ignored_storage_lease_value_ids,
      ignored_storage_lease_value_count, NULL));
  *out_assigned = true;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_coalescing_append_relation_interval(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_t* relation,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count,
    bool* out_assigned) {
  *out_assigned = false;
  if (relation->result_unit_offset > interval->unit_count ||
      relation->unit_count >
          interval->unit_count - relation->result_unit_offset) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation placement relation exceeds result "
                            "interval units");
  }
  uint32_t source_assignment_index = 0;
  const loom_value_id_t source_value_id =
      loom_low_placement_value_id(context->placement, relation->source_ordinal);
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_coalescing_assignment_index_for_value(
          context, source_value_id, &source_assignment_index));
  const loom_low_allocation_assignment_t* source_assignment =
      &context->assignment_map->assignments[source_assignment_index];
  if (!loom_low_allocation_assignment_is_register_like(source_assignment)) {
    return iree_ok_status();
  }
  if (!loom_low_allocation_coalescing_assignment_unit_span_fits(
          source_assignment, relation->source_unit_offset,
          relation->unit_count)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation placement relation exceeds source "
                            "assignment units");
  }
  const uint32_t source_unit_location =
      source_assignment->location_base + relation->source_unit_offset;
  if (source_unit_location < relation->result_unit_offset) {
    return iree_ok_status();
  }
  const uint32_t result_location_base =
      source_unit_location - relation->result_unit_offset;
  uint16_t interval_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_resolve_reg_class(
      context->target_constraints, interval->value_class,
      &interval_reg_class_id, NULL));
  const loom_value_id_t* ignored_storage_lease_value_ids = NULL;
  uint16_t ignored_storage_lease_value_count = 0;
  if (loom_low_allocation_coalescing_storage_alias_relation(relation)) {
    ignored_storage_lease_value_ids = ignored_value_ids;
    ignored_storage_lease_value_count = ignored_value_count;
  }
  return loom_low_allocation_coalescing_append_interval_at_location(
      context, interval, interval_reg_class_id,
      source_assignment->location_kind, result_location_base,
      interval->unit_count, ignored_value_ids, ignored_value_count,
      ignored_storage_lease_value_ids, ignored_storage_lease_value_count,
      out_assigned);
}

static iree_status_t
loom_low_allocation_coalescing_append_relation_interval_if_source_assigned(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_t* relation, bool* out_assigned) {
  *out_assigned = false;
  if (relation->result_unit_offset > interval->unit_count ||
      relation->unit_count >
          interval->unit_count - relation->result_unit_offset) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation placement relation exceeds result "
                            "interval units");
  }
  const loom_low_allocation_assignment_t* source_assignment =
      loom_low_allocation_coalescing_current_assignment_for_value_ordinal(
          context, relation->source_ordinal);
  if (!source_assignment) {
    return iree_ok_status();
  }
  if (!loom_low_allocation_assignment_is_register_like(source_assignment)) {
    return iree_ok_status();
  }
  if (!loom_low_allocation_coalescing_assignment_unit_span_fits(
          source_assignment, relation->source_unit_offset,
          relation->unit_count)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation placement relation exceeds source "
                            "assignment units");
  }
  const uint32_t source_unit_location =
      source_assignment->location_base + relation->source_unit_offset;
  if (source_unit_location < relation->result_unit_offset) {
    return iree_ok_status();
  }
  const uint32_t result_location_base =
      source_unit_location - relation->result_unit_offset;
  const loom_value_id_t* ignored_value_ids = NULL;
  uint16_t ignored_value_count = 0;
  const loom_value_id_t source_value_id = source_assignment->value_id;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_coalescing_select_ignored_counterpart_value(
          context, interval, relation, source_assignment, &source_value_id,
          relation->result_unit_offset, relation->unit_count,
          &ignored_value_ids, &ignored_value_count));
  uint16_t interval_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_resolve_reg_class(
      context->target_constraints, interval->value_class,
      &interval_reg_class_id, NULL));
  return loom_low_allocation_coalescing_append_interval_at_location(
      context, interval, interval_reg_class_id,
      source_assignment->location_kind, result_location_base,
      interval->unit_count, ignored_value_ids, ignored_value_count,
      /*ignored_storage_lease_value_ids=*/NULL,
      /*ignored_storage_lease_value_count=*/0, out_assigned);
}

static iree_status_t loom_low_allocation_coalescing_assign_relation_interval(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_t* relation, bool* out_assigned) {
  *out_assigned = false;
  const loom_value_id_t source_value_id =
      loom_low_placement_value_id(context->placement, relation->source_ordinal);
  return loom_low_allocation_coalescing_append_relation_interval(
      context, interval, relation, &source_value_id,
      /*ignored_value_count=*/1, out_assigned);
}

static iree_status_t
loom_low_allocation_coalescing_transfer_ignored_aliases_for_tied_consume(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* copy_interval,
    const loom_low_placement_relation_t* copy_relation,
    loom_value_id_t* ignored_value_ids, uint16_t ignored_value_capacity,
    uint16_t* ignored_value_count, bool* out_requires_materialized_storage) {
  *out_requires_materialized_storage = false;
  const loom_value_id_t source_value_id = loom_low_placement_value_id(
      context->placement, copy_relation->source_ordinal);
  IREE_ASSERT_GT(ignored_value_capacity, 0);
  ignored_value_ids[0] = source_value_id;
  *ignored_value_count = 1;

  const loom_low_placement_relation_range_t tied_range =
      loom_low_placement_relation_range_for_source_value_ordinal(
          context->placement, copy_relation->result_ordinal);
  bool has_tied_consume = false;
  for (uint32_t i = 0; i < tied_range.count; ++i) {
    const uint32_t relation_index =
        context->placement
            ->relation_indices_by_source_ordinal[tied_range.start + i];
    const loom_low_placement_relation_t* tied_relation =
        &context->placement->relations[relation_index];
    if (tied_relation->cause != LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT) {
      continue;
    }
    const loom_low_placement_relation_t* tied_operand_copy_relation =
        loom_low_allocation_coalescing_transfer_relation_for_result_ordinal(
            context, tied_relation->source_ordinal);
    if (tied_operand_copy_relation != copy_relation) {
      continue;
    }
    has_tied_consume = true;
    loom_value_id_t copy_source_id = LOOM_VALUE_ID_INVALID;
    bool copy_source_live = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_transfer_source_live_at_tied_definition(
            context, tied_relation, copy_relation, &copy_source_id,
            &copy_source_live));
    if (copy_source_live) {
      *out_requires_materialized_storage = true;
      return iree_ok_status();
    }
  }
  if (has_tied_consume) {
    // Ignoring an alias permits the copy to share its location for the copy's
    // entire lifetime. Qualify aliases at the copy definition, before any
    // tied consume can clobber storage needed by another live copy.
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_collect_dead_exact_storage_aliases(
            context, copy_relation->op, copy_interval->start_point,
            ignored_value_ids, ignored_value_capacity, ignored_value_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_coalescing_assign_transfer_interval(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_t* relation, bool* out_assigned) {
  *out_assigned = false;
  loom_value_id_t ignored_value_ids[16];
  uint16_t ignored_value_count = 0;
  bool requires_materialized_storage = false;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_coalescing_transfer_ignored_aliases_for_tied_consume(
          context, interval, relation, ignored_value_ids,
          (uint16_t)IREE_ARRAYSIZE(ignored_value_ids), &ignored_value_count,
          &requires_materialized_storage));
  if (requires_materialized_storage) {
    return iree_ok_status();
  }
  return loom_low_allocation_coalescing_append_relation_interval(
      context, interval, relation, ignored_value_ids, ignored_value_count,
      out_assigned);
}

static iree_status_t
loom_low_allocation_coalescing_assign_edge_destination_interval(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_t* relation, bool* out_assigned) {
  return loom_low_allocation_coalescing_append_relation_interval_if_source_assigned(
      context, interval, relation, out_assigned);
}

static iree_status_t loom_low_allocation_coalescing_assign_concat_interval(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval,
    loom_value_ordinal_t result_ordinal,
    const loom_low_placement_relation_range_t* range, bool* out_assigned) {
  *out_assigned = false;
  if (range->count == 0) {
    return iree_ok_status();
  }

  uint16_t concat_source_count = 0;
  for (uint32_t i = 0; i < range->count; ++i) {
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[range->start + i];
    if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT) {
      continue;
    }
    if (concat_source_count == UINT16_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low.concat placement source count exceeds "
                              "uint16_t");
    }
    ++concat_source_count;
  }
  if (concat_source_count == 0) {
    return iree_ok_status();
  }
  const loom_low_placement_relation_range_t edge_source_range =
      loom_low_placement_relation_range_for_source_value_ordinal(
          context->placement, result_ordinal);
  if (edge_source_range.count > UINT16_MAX - concat_source_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low.concat edge counterpart count exceeds "
                            "uint16_t");
  }
  const uint16_t ignored_value_capacity =
      concat_source_count + (uint16_t)edge_source_range.count;
  loom_value_id_t inline_ignored_value_ids[8];
  loom_value_id_t* ignored_value_ids = inline_ignored_value_ids;
  if (ignored_value_capacity > IREE_ARRAYSIZE(inline_ignored_value_ids)) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->arena, ignored_value_capacity, sizeof(*ignored_value_ids),
        (void**)&ignored_value_ids));
  }

  uint32_t result_location_base = 0;
  uint32_t coalesced_unit_count = 0;
  uint16_t ignored_value_count = 0;
  bool has_result_location_base = false;
  for (uint32_t i = 0; i < range->count; ++i) {
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[range->start + i];
    if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT) {
      continue;
    }
    uint32_t source_assignment_index = 0;
    const loom_value_id_t source_value_id = loom_low_placement_value_id(
        context->placement, relation->source_ordinal);
    IREE_RETURN_IF_ERROR(loom_low_allocation_coalescing_append_unique_value_id(
        ignored_value_ids, ignored_value_capacity, &ignored_value_count,
        source_value_id));
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_assignment_index_for_value(
            context, source_value_id, &source_assignment_index));
    const loom_low_allocation_assignment_t* source_assignment =
        &context->assignment_map->assignments[source_assignment_index];
    if (!loom_low_allocation_assignment_is_register_like(source_assignment)) {
      return iree_ok_status();
    }
    if (!loom_low_allocation_coalescing_assignment_unit_span_fits(
            source_assignment, relation->source_unit_offset,
            relation->unit_count)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "low.concat placement relation exceeds source "
                              "assignment units");
    }
    const uint32_t source_unit_location =
        source_assignment->location_base + relation->source_unit_offset;
    if (source_unit_location < relation->result_unit_offset) {
      return iree_ok_status();
    }
    const uint32_t candidate_base =
        source_unit_location - relation->result_unit_offset;
    if (!has_result_location_base) {
      result_location_base = candidate_base;
      has_result_location_base = true;
    } else if (result_location_base != candidate_base) {
      return iree_ok_status();
    }
    if (relation->unit_count > UINT32_MAX - coalesced_unit_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low.concat coalesced unit count exceeds u32");
    }
    coalesced_unit_count += relation->unit_count;
  }
  if (coalesced_unit_count != interval->unit_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low.concat placement relations do not cover the "
                            "result interval");
  }
  const loom_value_id_t first_source_value_id = ignored_value_ids[0];
  uint32_t first_assignment_index = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_coalescing_assignment_index_for_value(
          context, first_source_value_id, &first_assignment_index));
  const loom_low_allocation_assignment_t* first_assignment =
      &context->assignment_map->assignments[first_assignment_index];
  const uint16_t ignored_storage_lease_value_count = ignored_value_count;
  for (uint32_t i = 0; i < edge_source_range.count; ++i) {
    const uint32_t relation_index =
        context->placement
            ->relation_indices_by_source_ordinal[edge_source_range.start + i];
    const loom_low_placement_relation_t* edge_relation =
        &context->placement->relations[relation_index];
    if (edge_relation->kind != LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE ||
        !loom_low_placement_cause_is_edge(edge_relation->cause) ||
        edge_relation->source_unit_offset != 0 ||
        edge_relation->unit_count != interval->unit_count) {
      continue;
    }
    const loom_low_allocation_assignment_t* destination_assignment =
        loom_low_allocation_coalescing_current_assignment_for_value_ordinal(
            context, edge_relation->result_ordinal);
    if (!destination_assignment ||
        !loom_low_allocation_assignment_is_register_like(
            destination_assignment) ||
        destination_assignment->location_kind !=
            first_assignment->location_kind ||
        !loom_liveness_value_class_equal(destination_assignment->value_class,
                                         interval->value_class) ||
        edge_relation->result_unit_offset != 0 ||
        edge_relation->unit_count != destination_assignment->location_count ||
        !loom_low_allocation_coalescing_assignment_unit_span_fits(
            destination_assignment, edge_relation->result_unit_offset,
            edge_relation->unit_count)) {
      continue;
    }
    if (result_location_base > UINT32_MAX - edge_relation->source_unit_offset) {
      continue;
    }
    const uint32_t source_unit_location =
        result_location_base + edge_relation->source_unit_offset;
    const uint32_t destination_unit_location =
        destination_assignment->location_base +
        edge_relation->result_unit_offset;
    if (source_unit_location != destination_unit_location) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_append_ignored_counterpart_value(
            context, interval, edge_relation, destination_assignment,
            &destination_assignment->value_id,
            edge_relation->result_unit_offset, edge_relation->unit_count,
            ignored_value_ids, ignored_value_capacity, &ignored_value_count));
  }
  uint16_t interval_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_resolve_reg_class(
      context->target_constraints, interval->value_class,
      &interval_reg_class_id, NULL));
  return loom_low_allocation_coalescing_append_interval_at_location(
      context, interval, interval_reg_class_id, first_assignment->location_kind,
      result_location_base, interval->unit_count, ignored_value_ids,
      ignored_value_count, ignored_value_ids, ignored_storage_lease_value_count,
      out_assigned);
}

static bool loom_low_allocation_coalescing_relation_source_matches_interval(
    const loom_low_placement_relation_t* relation,
    const loom_liveness_interval_t* interval) {
  if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT) {
    return false;
  }
  return relation->source_unit_offset <= interval->unit_count &&
         relation->unit_count <=
             interval->unit_count - relation->source_unit_offset;
}

static bool loom_low_allocation_coalescing_candidate_location_for_concat_source(
    const loom_low_placement_relation_t* relation,
    const loom_low_placement_relation_t* sibling_relation,
    const loom_low_allocation_assignment_t* sibling_assignment,
    uint32_t* out_location_base) {
  if (sibling_assignment->location_base >
      UINT32_MAX - sibling_relation->source_unit_offset) {
    return false;
  }
  const uint32_t sibling_source_location =
      sibling_assignment->location_base + sibling_relation->source_unit_offset;
  if (sibling_source_location < sibling_relation->result_unit_offset) {
    return false;
  }
  const uint32_t result_location_base =
      sibling_source_location - sibling_relation->result_unit_offset;
  if (result_location_base > UINT32_MAX - relation->result_unit_offset) {
    return false;
  }
  const uint32_t source_location =
      result_location_base + relation->result_unit_offset;
  if (source_location < relation->source_unit_offset) {
    return false;
  }
  *out_location_base = source_location - relation->source_unit_offset;
  return true;
}

static bool loom_low_allocation_coalescing_source_location_for_concat_result(
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* result_assignment,
    uint32_t* out_location_base) {
  if (!loom_low_allocation_coalescing_assignment_unit_span_fits(
          result_assignment, relation->result_unit_offset,
          relation->unit_count)) {
    return false;
  }
  if (result_assignment->location_base >
      UINT32_MAX - relation->result_unit_offset) {
    return false;
  }
  const uint32_t source_unit_location =
      result_assignment->location_base + relation->result_unit_offset;
  if (source_unit_location < relation->source_unit_offset) {
    return false;
  }
  *out_location_base = source_unit_location - relation->source_unit_offset;
  return true;
}

static iree_status_t
loom_low_allocation_coalescing_assign_concat_source_from_result(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* result_assignment,
    bool* out_assigned) {
  *out_assigned = false;
  if (!loom_low_allocation_assignment_is_register_like(result_assignment) ||
      !loom_liveness_value_class_equal(result_assignment->value_class,
                                       interval->value_class)) {
    return iree_ok_status();
  }
  uint32_t location_base = 0;
  if (!loom_low_allocation_coalescing_source_location_for_concat_result(
          relation, result_assignment, &location_base)) {
    return iree_ok_status();
  }
  const loom_value_id_t result_value_id =
      loom_low_placement_value_id(context->placement, relation->result_ordinal);
  uint16_t interval_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_resolve_reg_class(
      context->target_constraints, interval->value_class,
      &interval_reg_class_id, NULL));
  return loom_low_allocation_coalescing_append_interval_at_location(
      context, interval, interval_reg_class_id,
      result_assignment->location_kind, location_base, interval->unit_count,
      &result_value_id,
      /*ignored_value_count=*/1, &result_value_id,
      /*ignored_storage_lease_value_count=*/1, out_assigned);
}

static iree_status_t loom_low_allocation_coalescing_concat_ignored_sources(
    loom_low_allocation_coalescing_context_t* context,
    const loom_low_placement_relation_range_t* range,
    loom_value_id_t inline_ignored_value_ids[8],
    iree_host_size_t inline_ignored_value_capacity,
    loom_value_id_t** out_ignored_value_ids,
    uint16_t* out_ignored_value_count) {
  *out_ignored_value_ids = inline_ignored_value_ids;
  *out_ignored_value_count = 0;
  for (uint32_t i = 0; i < range->count; ++i) {
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[range->start + i];
    if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT) {
      continue;
    }
    if (*out_ignored_value_count == UINT16_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low.concat placement source count exceeds "
                              "uint16_t");
    }
    ++*out_ignored_value_count;
  }
  if (*out_ignored_value_count > inline_ignored_value_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->arena, *out_ignored_value_count,
        sizeof(**out_ignored_value_ids), (void**)out_ignored_value_ids));
  }

  uint16_t ignored_value_index = 0;
  for (uint32_t i = 0; i < range->count; ++i) {
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[range->start + i];
    if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT) {
      continue;
    }
    (*out_ignored_value_ids)[ignored_value_index++] =
        loom_low_placement_value_id(context->placement,
                                    relation->source_ordinal);
  }
  return iree_ok_status();
}

// Returns whether all concat sources are produced in one compact assembly
// window immediately before the result. Reserving a long-lived result for an
// early source carries its transient pressure through unrelated packets; a
// compact source cluster instead has no useful placement lifetime of its own.
static bool loom_low_allocation_coalescing_concat_sources_form_compact_assembly(
    const loom_low_allocation_coalescing_context_t* context,
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
loom_low_allocation_coalescing_concat_has_fragmented_tied_source_assembly(
    const loom_low_allocation_coalescing_context_t* context,
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
      if (!loom_low_allocation_coalescing_compose_tied_concat_source_relation(
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
static bool loom_low_allocation_coalescing_concat_has_assigned_source(
    const loom_low_allocation_coalescing_context_t* context,
    const loom_low_placement_relation_range_t* result_range) {
  for (uint32_t i = 0; i < result_range->count; ++i) {
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[result_range->start + i];
    if (relation->cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT &&
        loom_low_allocation_coalescing_current_assignment_for_value_ordinal(
            context, relation->source_ordinal) != NULL) {
      return true;
    }
  }
  return false;
}

// Returns whether normal placement of the current source can be extended to
// every sibling source without a materialized concat. This predicts the same
// locations that sibling coalescing will request later; reserving the result
// when they are all available would only perturb an already-valid allocation.
static iree_status_t
loom_low_allocation_coalescing_default_concat_source_assembles_result(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* source_interval,
    const loom_low_placement_relation_t* relation,
    const loom_liveness_interval_t* result_interval,
    loom_low_allocation_class_capacity_t result_capacity,
    const loom_low_placement_relation_range_t* result_range,
    bool* out_assembles_result) {
  *out_assembles_result = false;

  loom_low_allocation_class_capacity_t source_capacity = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_interval_capacity(
      context->target_constraints, source_interval, &source_capacity));
  uint32_t source_location_base = 0;
  if (!loom_low_allocation_search_find_free_location(
          context->search_context, source_interval, source_capacity,
          &source_location_base) ||
      source_location_base > UINT32_MAX - relation->source_unit_offset) {
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
      loom_low_allocation_live_range_interval_alignment(result_interval);
  if (result_location_base % result_alignment != 0 ||
      !loom_low_allocation_storage_reg_classes_share(
          context->search_context->descriptor_set,
          source_capacity.descriptor_reg_class_id,
          result_capacity.descriptor_reg_class_id) ||
      !loom_low_allocation_target_constraints_location_range_fits_capacity(
          &result_capacity, source_capacity.location_kind, result_location_base,
          result_interval->unit_count)) {
    return iree_ok_status();
  }

  for (uint32_t i = 0; i < result_range->count; ++i) {
    const loom_low_placement_relation_t* sibling_relation =
        &context->placement->relations[result_range->start + i];
    if (sibling_relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT) {
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
            context->target_constraints, sibling_interval, &sibling_capacity));
    const uint32_t sibling_alignment =
        loom_low_allocation_live_range_interval_alignment(sibling_interval);
    if (!loom_low_allocation_storage_reg_classes_share(
            context->search_context->descriptor_set,
            source_capacity.descriptor_reg_class_id,
            sibling_capacity.descriptor_reg_class_id) ||
        !loom_low_allocation_target_constraints_location_range_fits_capacity(
            &sibling_capacity, source_capacity.location_kind,
            sibling_location_base, sibling_interval->unit_count) ||
        sibling_location_base % sibling_alignment != 0 ||
        loom_low_allocation_search_location_conflicts(
            context->search_context, sibling_interval,
            sibling_capacity.descriptor_reg_class_id,
            source_capacity.location_kind, sibling_location_base,
            sibling_interval->unit_count,
            /*ignored_value_ids=*/NULL,
            /*ignored_value_count=*/0,
            /*ignored_storage_lease_value_ids=*/NULL,
            /*ignored_storage_lease_value_count=*/0,
            LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED)) {
      return iree_ok_status();
    }
  }

  *out_assembles_result = true;
  return iree_ok_status();
}

// Chooses a concat result span that can also accept the current source slice.
// Scheduled allocation may see scalar concat sources long before the concat op,
// so selecting only for the future result interval can reserve a span that the
// current source cannot occupy without a packet-local move.
static bool
loom_low_allocation_coalescing_find_concat_result_location_for_source(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* source_interval,
    const loom_low_placement_relation_t* relation,
    const loom_liveness_interval_t* result_interval,
    loom_low_allocation_class_capacity_t capacity,
    uint32_t reservation_start_point,
    loom_low_allocation_assignment_flags_t reservation_flags,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count,
    uint32_t* out_result_location_base) {
  if (result_interval->unit_count == 0 ||
      (capacity.is_bounded &&
       result_interval->unit_count > capacity.max_units)) {
    return false;
  }

  const uint32_t result_alignment =
      loom_low_allocation_live_range_interval_alignment(result_interval);
  const uint32_t source_alignment =
      loom_low_allocation_live_range_interval_alignment(source_interval);
  const uint32_t assigned_limit =
      loom_low_allocation_target_constraints_assigned_location_search_limit(
          context->target_constraints, capacity.descriptor_reg_class_id,
          capacity.location_kind);

  uint32_t last_base = 0;
  if (capacity.is_bounded) {
    last_base = capacity.max_units - result_interval->unit_count;
  } else if (!loom_low_allocation_coalescing_align_up_u32(
                 assigned_limit, result_alignment, &last_base)) {
    return false;
  }

  loom_low_allocation_assignment_t reservation = {
      .value_id = result_interval->value_id,
      .value_class = result_interval->value_class,
      .descriptor_reg_class_id = capacity.descriptor_reg_class_id,
      .start_point = reservation_start_point,
      .end_point = loom_low_allocation_live_range_interval_storage_end_point(
          result_interval),
      .unit_count = result_interval->unit_count,
      .location_kind = capacity.location_kind,
      .location_count = result_interval->unit_count,
      .unit_point_start =
          loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
              context->search_context->unit_liveness, context->liveness,
              relation->result_ordinal),
      .flags = reservation_flags,
  };
  reservation.end_point =
      loom_low_allocation_live_range_assignment_max_unit_end_point(
          context->search_context->unit_liveness->end_points,
          context->search_context->unit_liveness->point_count, &reservation);
  for (uint32_t base = 0; base <= last_base;) {
    reservation.location_base = base;
    if (!loom_low_allocation_search_assignment_conflicts(
            context->search_context, &reservation, ignored_value_ids,
            ignored_value_count, ignored_value_ids, ignored_value_count,
            LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED)) {
      bool source_location_ok = false;
      uint32_t source_location_base = 0;
      if (base <= UINT32_MAX - relation->result_unit_offset) {
        const uint32_t source_unit_location =
            base + relation->result_unit_offset;
        if (source_unit_location >= relation->source_unit_offset) {
          source_location_base =
              source_unit_location - relation->source_unit_offset;
          source_location_ok =
              source_location_base % source_alignment == 0 &&
              source_location_base <= UINT32_MAX - source_interval->unit_count;
        }
      }
      if (source_location_ok &&
          !(capacity.is_bounded &&
            source_location_base + source_interval->unit_count >
                capacity.max_units) &&
          !loom_low_allocation_search_location_conflicts(
              context->search_context, source_interval,
              capacity.descriptor_reg_class_id, capacity.location_kind,
              source_location_base, source_interval->unit_count,
              /*ignored_value_ids=*/NULL,
              /*ignored_value_count=*/0,
              /*ignored_storage_lease_value_ids=*/NULL,
              /*ignored_storage_lease_value_count=*/0,
              LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED)) {
        *out_result_location_base = base;
        return true;
      }
    }
    if (base > UINT32_MAX - result_alignment) {
      break;
    }
    base += result_alignment;
  }
  return false;
}

typedef enum loom_low_allocation_concat_assignment_e {
  LOOM_LOW_ALLOCATION_CONCAT_ASSIGNMENT_NONE = 0,
  LOOM_LOW_ALLOCATION_CONCAT_ASSIGNMENT_SOURCE = 1,
  LOOM_LOW_ALLOCATION_CONCAT_ASSIGNMENT_RESULT = 2,
} loom_low_allocation_concat_assignment_t;

static iree_status_t
loom_low_allocation_coalescing_assign_concat_source_or_result(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* source_interval,
    const loom_low_placement_relation_t* relation,
    const loom_liveness_interval_t* result_interval,
    const loom_low_placement_relation_range_t* result_range,
    loom_low_allocation_concat_assignment_t* out_assignment) {
  *out_assignment = LOOM_LOW_ALLOCATION_CONCAT_ASSIGNMENT_NONE;
  const uint32_t result_lifetime =
      result_interval->end_point - result_interval->start_point;
  loom_low_allocation_class_capacity_t capacity = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_interval_capacity(
      context->target_constraints, result_interval, &capacity));

  loom_value_id_t inline_ignored_value_ids[8];
  loom_value_id_t* ignored_value_ids = inline_ignored_value_ids;
  uint16_t ignored_value_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_coalescing_concat_ignored_sources(
      context, result_range, inline_ignored_value_ids,
      IREE_ARRAYSIZE(inline_ignored_value_ids), &ignored_value_ids,
      &ignored_value_count));
  if (ignored_value_count == 0) {
    return iree_ok_status();
  }

  const bool sources_form_compact_assembly =
      loom_low_allocation_coalescing_concat_sources_form_compact_assembly(
          context, result_interval, result_range);
  const uint32_t* result_unit_start_points =
      loom_low_allocation_unit_liveness_start_points_for_value_ordinal(
          context->search_context->unit_liveness, context->liveness,
          relation->result_ordinal);
  IREE_ASSERT(result_unit_start_points != NULL);
  const bool has_fragmented_tied_source_assembly =
      loom_low_allocation_coalescing_concat_has_fragmented_tied_source_assembly(
          context, result_range, result_unit_start_points);
  bool favor_result_reservation = true;
  // Scalar sources can cheaply reserve a future aggregate, and packet-local
  // or tied in-place sources must reserve before independently allocated
  // temporaries fragment their required span.
  if (source_interval->unit_count != 1 && result_lifetime != 1) {
    favor_result_reservation =
        sources_form_compact_assembly || has_fragmented_tied_source_assembly;
    if (loom_low_allocation_coalescing_concat_has_assigned_source(
            context, result_range)) {
      return iree_ok_status();
    }
    if (favor_result_reservation) {
      bool default_source_assembles_result = false;
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_coalescing_default_concat_source_assembles_result(
              context, source_interval, relation, result_interval, capacity,
              result_range, &default_source_assembles_result));
      if (default_source_assembles_result) {
        return iree_ok_status();
      }
    } else if (loom_target_residency_model_is_empty(
                   context->search_context->residency_model)) {
      return iree_ok_status();
    }
  }

  uint32_t reservation_start_point = result_interval->start_point;
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
  if (!loom_low_allocation_coalescing_find_concat_result_location_for_source(
          context, source_interval, relation, result_interval, capacity,
          reservation_start_point, reservation_flags, ignored_value_ids,
          ignored_value_count, &result_location_base)) {
    return iree_ok_status();
  }

  if (!favor_result_reservation) {
    uint32_t source_location_base = 0;
    if (loom_low_allocation_search_find_free_location(context->search_context,
                                                      source_interval, capacity,
                                                      &source_location_base)) {
      const uint16_t reg_class_id = capacity.descriptor_reg_class_id;
      const uint32_t current_location_end =
          context->target_constraints
              ->max_assigned_location_end_by_reg_class[reg_class_id];
      const uint32_t source_location_end =
          iree_max(current_location_end,
                   source_location_base + source_interval->unit_count);
      const uint32_t result_location_end =
          iree_max(current_location_end,
                   result_location_base + result_interval->unit_count);
      if (result_location_end > source_location_end) {
        const loom_target_residency_model_t* residency_model =
            context->search_context->residency_model;
        const uint32_t* current_units_by_reg_class =
            context->target_constraints->max_assigned_location_end_by_reg_class;
        const uint32_t source_tier =
            loom_target_residency_evaluate_tier_with_direct_resource_override(
                residency_model, current_units_by_reg_class, reg_class_id,
                source_location_end);
        const uint32_t result_tier =
            loom_target_residency_evaluate_tier_with_direct_resource_override(
                residency_model, current_units_by_reg_class, reg_class_id,
                result_location_end);
        if (result_tier < source_tier) {
          bool source_assigned = false;
          IREE_RETURN_IF_ERROR(
              loom_low_allocation_coalescing_append_interval_at_location(
                  context, source_interval, reg_class_id,
                  capacity.location_kind, source_location_base,
                  source_interval->unit_count,
                  /*ignored_value_ids=*/NULL,
                  /*ignored_value_count=*/0,
                  /*ignored_storage_lease_value_ids=*/NULL,
                  /*ignored_storage_lease_value_count=*/0, &source_assigned));
          IREE_ASSERT_TRUE(source_assigned);
          *out_assignment = LOOM_LOW_ALLOCATION_CONCAT_ASSIGNMENT_SOURCE;
          return iree_ok_status();
        }
      }
    }
  }

  const loom_low_allocation_assignment_t result_assignment = {
      .value_id = result_interval->value_id,
      .value_class = result_interval->value_class,
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
  IREE_RETURN_IF_ERROR(
      context->append_assignment(context->user_data, &result_assignment,
                                 ignored_value_ids, ignored_value_count, NULL));
  *out_assignment = LOOM_LOW_ALLOCATION_CONCAT_ASSIGNMENT_RESULT;
  return iree_ok_status();
}

static bool loom_low_allocation_coalescing_edge_relation_covers_concat_source(
    const loom_low_placement_relation_t* edge_relation,
    const loom_low_placement_relation_t* concat_relation) {
  if (!loom_low_placement_cause_is_edge(edge_relation->cause) ||
      edge_relation->source_ordinal != concat_relation->result_ordinal) {
    return false;
  }
  if (concat_relation->result_unit_offset < edge_relation->source_unit_offset) {
    return false;
  }
  if (concat_relation->unit_count >
      UINT32_MAX - concat_relation->result_unit_offset) {
    return false;
  }
  if (edge_relation->unit_count >
      UINT32_MAX - edge_relation->source_unit_offset) {
    return false;
  }
  const uint32_t concat_source_end =
      concat_relation->result_unit_offset + concat_relation->unit_count;
  const uint32_t edge_source_end =
      edge_relation->source_unit_offset + edge_relation->unit_count;
  return concat_source_end <= edge_source_end;
}

static iree_status_t
loom_low_allocation_coalescing_assign_concat_source_from_edge_destination(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_t* concat_relation, bool* out_assigned) {
  *out_assigned = false;
  const loom_low_placement_relation_range_t source_range =
      loom_low_placement_relation_range_for_source_value_ordinal(
          context->placement, concat_relation->result_ordinal);
  for (uint32_t i = 0; i < source_range.count; ++i) {
    const uint32_t relation_index =
        context->placement
            ->relation_indices_by_source_ordinal[source_range.start + i];
    const loom_low_placement_relation_t* edge_relation =
        &context->placement->relations[relation_index];
    if (!loom_low_allocation_coalescing_edge_relation_covers_concat_source(
            edge_relation, concat_relation)) {
      continue;
    }

    const loom_low_allocation_assignment_t* destination_assignment =
        loom_low_allocation_coalescing_current_assignment_for_value_ordinal(
            context, edge_relation->result_ordinal);
    if (!destination_assignment ||
        !loom_low_allocation_assignment_is_register_like(
            destination_assignment) ||
        !loom_liveness_value_class_equal(destination_assignment->value_class,
                                         interval->value_class)) {
      continue;
    }

    const uint32_t concat_to_edge_unit_offset =
        concat_relation->result_unit_offset - edge_relation->source_unit_offset;
    if (edge_relation->result_unit_offset >
        UINT32_MAX - concat_to_edge_unit_offset) {
      continue;
    }
    const uint32_t destination_unit_offset =
        edge_relation->result_unit_offset + concat_to_edge_unit_offset;
    if (!loom_low_allocation_coalescing_assignment_unit_span_fits(
            destination_assignment, destination_unit_offset,
            concat_relation->unit_count)) {
      continue;
    }
    const uint32_t destination_unit_location =
        destination_assignment->location_base + destination_unit_offset;
    if (destination_unit_location < concat_relation->source_unit_offset) {
      continue;
    }
    const uint32_t source_location_base =
        destination_unit_location - concat_relation->source_unit_offset;
    const loom_value_id_t* ignored_value_ids = NULL;
    uint16_t ignored_value_count = 0;
    const loom_value_id_t destination_value_id =
        destination_assignment->value_id;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_select_ignored_counterpart_value(
            context, interval, edge_relation, destination_assignment,
            &destination_value_id, destination_unit_offset,
            concat_relation->unit_count, &ignored_value_ids,
            &ignored_value_count));
    uint16_t interval_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_target_constraints_resolve_reg_class(
            context->target_constraints, interval->value_class,
            &interval_reg_class_id, NULL));
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_append_interval_at_location(
            context, interval, interval_reg_class_id,
            destination_assignment->location_kind, source_location_base,
            interval->unit_count, ignored_value_ids, ignored_value_count,
            /*ignored_storage_lease_value_ids=*/NULL,
            /*ignored_storage_lease_value_count=*/0, out_assigned));
    if (*out_assigned) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static bool
loom_low_allocation_coalescing_relation_source_matches_edge_interval(
    const loom_low_placement_relation_t* relation,
    const loom_liveness_interval_t* interval) {
  if (!loom_low_placement_cause_is_edge(relation->cause)) {
    return false;
  }
  return relation->source_unit_offset <= interval->unit_count &&
         relation->unit_count <=
             interval->unit_count - relation->source_unit_offset;
}

iree_status_t loom_low_allocation_coalescing_assign_tied_interval(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval, bool* out_assigned) {
  *out_assigned = false;
  loom_value_ordinal_t result_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_coalescing_value_ordinal_for_interval(
          context, interval, &result_ordinal));
  const loom_low_placement_relation_t* relation =
      loom_low_allocation_coalescing_first_placement_relation(
          context, result_ordinal, LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT);
  if (!relation) {
    return iree_ok_status();
  }

  const loom_value_id_t tied_operand_id =
      loom_low_placement_value_id(context->placement, relation->source_ordinal);
  uint32_t operand_assignment_index = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_coalescing_assignment_index_for_value(
          context, tied_operand_id, &operand_assignment_index));
  const loom_low_allocation_assignment_t* operand_assignment =
      &context->assignment_map->assignments[operand_assignment_index];
  if (!loom_low_allocation_assignment_is_register_like(operand_assignment)) {
    return iree_ok_status();
  }
  uint16_t interval_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_resolve_reg_class(
      context->target_constraints, interval->value_class,
      &interval_reg_class_id, NULL));
  if (!loom_liveness_value_class_equal(operand_assignment->value_class,
                                       interval->value_class) ||
      operand_assignment->location_count != interval->unit_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low tied result does not match operand allocation class");
  }

  loom_value_id_t ignored_value_ids[16] = {tied_operand_id,
                                           LOOM_VALUE_ID_INVALID};
  const uint16_t ignored_value_capacity =
      (uint16_t)IREE_ARRAYSIZE(ignored_value_ids);
  uint16_t ignored_value_count = 1;
  const loom_low_placement_relation_t* tied_operand_copy_relation =
      loom_low_allocation_coalescing_transfer_relation_for_result_ordinal(
          context, relation->source_ordinal);
  loom_value_id_t copy_source_id = LOOM_VALUE_ID_INVALID;
  bool copy_source_live = false;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_coalescing_transfer_source_live_at_tied_definition(
          context, relation, tied_operand_copy_relation, &copy_source_id,
          &copy_source_live));
  if (copy_source_id != LOOM_VALUE_ID_INVALID && !copy_source_live) {
    ignored_value_ids[ignored_value_count++] = copy_source_id;
  }
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_coalescing_collect_dead_exact_storage_aliases(
          context, relation->op, interval->start_point, ignored_value_ids,
          ignored_value_capacity, &ignored_value_count));
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_coalescing_collect_tied_concat_reservations(
          context, relation, operand_assignment, ignored_value_ids,
          ignored_value_capacity, &ignored_value_count));

  loom_value_id_t inline_storage_lease_ignored_value_ids[16];
  loom_value_id_t* storage_lease_ignored_value_ids =
      inline_storage_lease_ignored_value_ids;
  uint16_t storage_lease_ignored_value_capacity =
      (uint16_t)IREE_ARRAYSIZE(inline_storage_lease_ignored_value_ids);
  uint16_t storage_lease_ignored_value_count = 0;
  for (uint16_t i = 0; i < ignored_value_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_coalescing_append_unique_value_id(
        storage_lease_ignored_value_ids, storage_lease_ignored_value_capacity,
        &storage_lease_ignored_value_count, ignored_value_ids[i]));
  }
  const loom_low_placement_relation_range_t tied_operand_relation_range =
      loom_low_placement_relation_range_for_value_ordinal(
          context->placement, relation->source_ordinal);
  if (tied_operand_relation_range.count >
      storage_lease_ignored_value_capacity -
          storage_lease_ignored_value_count) {
    const uint32_t required_capacity =
        tied_operand_relation_range.count + storage_lease_ignored_value_count;
    if (required_capacity > UINT16_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low tied operand alias count exceeds uint16_t");
    }
    storage_lease_ignored_value_capacity = (uint16_t)required_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->arena, storage_lease_ignored_value_capacity,
        sizeof(*storage_lease_ignored_value_ids),
        (void**)&storage_lease_ignored_value_ids));
    for (uint16_t i = 0; i < ignored_value_count; ++i) {
      storage_lease_ignored_value_ids[i] = ignored_value_ids[i];
    }
    storage_lease_ignored_value_count = ignored_value_count;
  }
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_coalescing_collect_tied_storage_aliases(
          context, relation, relation->source_ordinal,
          storage_lease_ignored_value_ids, storage_lease_ignored_value_capacity,
          &storage_lease_ignored_value_count));
  if (loom_low_allocation_search_location_conflicts(
          context->search_context, interval, interval_reg_class_id,
          operand_assignment->location_kind, operand_assignment->location_base,
          operand_assignment->location_count, ignored_value_ids,
          ignored_value_count, storage_lease_ignored_value_ids,
          storage_lease_ignored_value_count,
          LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low tied result cannot share the operand location without "
        "overlapping another live interval");
  }

  const loom_low_allocation_assignment_t assignment = {
      .value_id = interval->value_id,
      .value_class = interval->value_class,
      .descriptor_reg_class_id = interval_reg_class_id,
      .start_point = interval->start_point,
      .end_point =
          loom_low_allocation_live_range_interval_storage_end_point(interval),
      .unit_count = interval->unit_count,
      .location_kind = operand_assignment->location_kind,
      .location_base = operand_assignment->location_base,
      .location_count = operand_assignment->location_count,
  };
  IREE_RETURN_IF_ERROR(context->append_assignment(
      context->user_data, &assignment, storage_lease_ignored_value_ids,
      storage_lease_ignored_value_count, NULL));
  *out_assigned = true;
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_coalescing_assign_concat_source_relation(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_t* relation, bool* out_assigned) {
  *out_assigned = false;
  if (!loom_low_allocation_coalescing_relation_source_matches_interval(
          relation, interval)) {
    return iree_ok_status();
  }

  const loom_liveness_interval_t* result_interval =
      loom_liveness_interval_for_value_ordinal(context->liveness,
                                               relation->result_ordinal);
  if (!result_interval ||
      !loom_liveness_value_class_equal(result_interval->value_class,
                                       interval->value_class)) {
    return iree_ok_status();
  }
  const loom_low_placement_relation_range_t result_range =
      loom_low_placement_relation_range_for_value_ordinal(
          context->placement, relation->result_ordinal);

  const loom_low_allocation_assignment_t* result_assignment =
      loom_low_allocation_coalescing_current_assignment_for_value_ordinal(
          context, relation->result_ordinal);
  if (result_assignment) {
    return loom_low_allocation_coalescing_assign_concat_source_from_result(
        context, interval, relation, result_assignment, out_assigned);
  }

  IREE_RETURN_IF_ERROR(
      loom_low_allocation_coalescing_assign_concat_source_from_edge_destination(
          context, interval, relation, out_assigned));
  if (*out_assigned) {
    return iree_ok_status();
  }

  for (uint32_t result_index = 0; result_index < result_range.count;
       ++result_index) {
    const loom_low_placement_relation_t* sibling_relation =
        &context->placement->relations[result_range.start + result_index];
    if (sibling_relation == relation ||
        sibling_relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT ||
        sibling_relation->source_ordinal == relation->source_ordinal) {
      continue;
    }
    const loom_low_allocation_assignment_t* sibling_assignment =
        loom_low_allocation_coalescing_current_assignment_for_value_ordinal(
            context, sibling_relation->source_ordinal);
    if (!sibling_assignment ||
        !loom_low_allocation_assignment_is_register_like(sibling_assignment) ||
        !loom_liveness_value_class_equal(sibling_assignment->value_class,
                                         interval->value_class) ||
        !loom_low_allocation_coalescing_assignment_unit_span_fits(
            sibling_assignment, sibling_relation->source_unit_offset,
            sibling_relation->unit_count)) {
      continue;
    }

    uint32_t location_base = 0;
    if (!loom_low_allocation_coalescing_candidate_location_for_concat_source(
            relation, sibling_relation, sibling_assignment, &location_base)) {
      continue;
    }
    uint16_t interval_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_target_constraints_resolve_reg_class(
            context->target_constraints, interval->value_class,
            &interval_reg_class_id, NULL));
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_append_interval_at_location(
            context, interval, interval_reg_class_id,
            sibling_assignment->location_kind, location_base,
            interval->unit_count, /*ignored_value_ids=*/NULL,
            /*ignored_value_count=*/0,
            /*ignored_storage_lease_value_ids=*/NULL,
            /*ignored_storage_lease_value_count=*/0, out_assigned));
    if (*out_assigned) {
      return iree_ok_status();
    }
  }

  loom_low_allocation_concat_assignment_t concat_assignment =
      LOOM_LOW_ALLOCATION_CONCAT_ASSIGNMENT_NONE;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_coalescing_assign_concat_source_or_result(
          context, interval, relation, result_interval, &result_range,
          &concat_assignment));
  if (concat_assignment == LOOM_LOW_ALLOCATION_CONCAT_ASSIGNMENT_NONE) {
    return iree_ok_status();
  }
  if (concat_assignment == LOOM_LOW_ALLOCATION_CONCAT_ASSIGNMENT_SOURCE) {
    *out_assigned = true;
    return iree_ok_status();
  }
  result_assignment =
      loom_low_allocation_coalescing_current_assignment_for_value_ordinal(
          context, relation->result_ordinal);
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_coalescing_assign_concat_source_from_result(
          context, interval, relation, result_assignment, out_assigned));
  return iree_ok_status();
}

iree_status_t loom_low_allocation_coalescing_assign_concat_source_interval(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval, bool* out_assigned) {
  *out_assigned = false;
  loom_value_ordinal_t source_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_coalescing_value_ordinal_for_interval(
          context, interval, &source_ordinal));
  const loom_low_placement_relation_range_t source_range =
      loom_low_placement_relation_range_for_source_value_ordinal(
          context->placement, source_ordinal);
  for (uint32_t source_index = 0; source_index < source_range.count;
       ++source_index) {
    const uint32_t relation_index =
        context->placement
            ->relation_indices_by_source_ordinal[source_range.start +
                                                 source_index];
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[relation_index];
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_assign_concat_source_relation(
            context, interval, relation, out_assigned));
    if (*out_assigned) {
      return iree_ok_status();
    }
    if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT) {
      continue;
    }
    const loom_low_placement_relation_range_t tied_result_source_range =
        loom_low_placement_relation_range_for_source_value_ordinal(
            context->placement, relation->result_ordinal);
    for (uint32_t tied_source_index = 0;
         tied_source_index < tied_result_source_range.count;
         ++tied_source_index) {
      const uint32_t tied_source_relation_index =
          context->placement->relation_indices_by_source_ordinal
              [tied_result_source_range.start + tied_source_index];
      const loom_low_placement_relation_t* tied_source_relation =
          &context->placement->relations[tied_source_relation_index];
      loom_low_placement_relation_t composed_relation;
      if (!loom_low_allocation_coalescing_compose_tied_concat_source_relation(
              relation, tied_source_relation, &composed_relation)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_coalescing_assign_concat_source_relation(
              context, interval, &composed_relation, out_assigned));
      if (*out_assigned) {
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_coalescing_assign_structural_interval(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval, bool* out_assigned) {
  *out_assigned = false;
  loom_value_ordinal_t result_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_coalescing_value_ordinal_for_interval(
          context, interval, &result_ordinal));
  const loom_low_placement_relation_range_t range =
      loom_low_placement_relation_range_for_value_ordinal(context->placement,
                                                          result_ordinal);
  for (uint32_t i = 0; i < range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[range.start + i];
    if (!loom_low_placement_relation_can_alias(relation)) {
      continue;
    }
    switch (relation->cause) {
      case LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY: {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_coalescing_assign_transfer_interval(
                context, interval, relation, out_assigned));
        if (*out_assigned) {
          return iree_ok_status();
        }
        break;
      }
      case LOOM_LOW_PLACEMENT_CAUSE_LOW_MOVE: {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_coalescing_assign_transfer_interval(
                context, interval, relation, out_assigned));
        if (*out_assigned) {
          return iree_ok_status();
        }
        break;
      }
      case LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE: {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_coalescing_assign_relation_interval(
                context, interval, relation, out_assigned));
        if (*out_assigned) {
          return iree_ok_status();
        }
        break;
      }
      case LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT: {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_coalescing_assign_concat_interval(
                context, interval, result_ordinal, &range, out_assigned));
        return iree_ok_status();
      }
      case LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH: {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_coalescing_assign_edge_destination_interval(
                context, interval, relation, out_assigned));
        if (*out_assigned) {
          return iree_ok_status();
        }
        break;
      }
      case LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_LOOP_ENTRY:
      case LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_YIELD:
      case LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_CONDITION: {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_coalescing_assign_edge_destination_interval(
                context, interval, relation, out_assigned));
        if (*out_assigned) {
          return iree_ok_status();
        }
        break;
      }
      default:
        break;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_coalescing_assign_edge_source_interval(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval, bool* out_assigned) {
  *out_assigned = false;
  loom_value_ordinal_t source_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_coalescing_value_ordinal_for_interval(
          context, interval, &source_ordinal));
  const loom_low_placement_relation_range_t source_range =
      loom_low_placement_relation_range_for_source_value_ordinal(
          context->placement, source_ordinal);
  for (uint32_t source_index = 0; source_index < source_range.count;
       ++source_index) {
    const uint32_t relation_index =
        context->placement
            ->relation_indices_by_source_ordinal[source_range.start +
                                                 source_index];
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[relation_index];
    if (!loom_low_allocation_coalescing_relation_source_matches_edge_interval(
            relation, interval)) {
      continue;
    }

    const loom_low_allocation_assignment_t* destination_assignment =
        loom_low_allocation_coalescing_current_assignment_for_value_ordinal(
            context, relation->result_ordinal);
    if (!destination_assignment ||
        !loom_low_allocation_assignment_is_register_like(
            destination_assignment) ||
        !loom_liveness_value_class_equal(destination_assignment->value_class,
                                         interval->value_class) ||
        !loom_low_allocation_coalescing_assignment_unit_span_fits(
            destination_assignment, relation->result_unit_offset,
            relation->unit_count)) {
      continue;
    }

    const uint32_t destination_unit_location =
        destination_assignment->location_base + relation->result_unit_offset;
    if (destination_unit_location < relation->source_unit_offset) {
      continue;
    }
    const uint32_t source_location_base =
        destination_unit_location - relation->source_unit_offset;
    const loom_value_id_t* ignored_value_ids = NULL;
    uint16_t ignored_value_count = 0;
    const loom_value_id_t destination_value_id =
        destination_assignment->value_id;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_select_ignored_counterpart_value(
            context, interval, relation, destination_assignment,
            &destination_value_id, relation->result_unit_offset,
            relation->unit_count, &ignored_value_ids, &ignored_value_count));
    uint16_t interval_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_target_constraints_resolve_reg_class(
            context->target_constraints, interval->value_class,
            &interval_reg_class_id, NULL));
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_append_interval_at_location(
            context, interval, interval_reg_class_id,
            destination_assignment->location_kind, source_location_base,
            interval->unit_count, ignored_value_ids, ignored_value_count,
            /*ignored_storage_lease_value_ids=*/NULL,
            /*ignored_storage_lease_value_count=*/0, out_assigned));
    if (*out_assigned) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}
