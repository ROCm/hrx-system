// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/packet_move.h"

#include "loom/codegen/low/allocation/move_topology.h"
#include "loom/codegen/low/allocation/unit_location.h"
#include "loom/ops/low/ops.h"

typedef struct loom_low_allocation_packet_move_builder_t {
  // Immutable allocation facts used to construct packet-local moves.
  const loom_low_allocation_packet_move_context_t* context;
  // Plan receiving final move groups.
  loom_low_allocation_packet_move_plan_t plan;
} loom_low_allocation_packet_move_builder_t;

static const loom_low_allocation_assignment_t*
loom_low_allocation_packet_move_assignment(
    const loom_low_allocation_assignment_map_t* map,
    loom_value_ordinal_t value_ordinal) {
  const uint32_t assignment_index =
      map->assignment_indices_by_value_ordinal[value_ordinal];
  IREE_ASSERT_NE(assignment_index, UINT32_MAX,
                 "allocated packet-move value must have an assignment");
  return &map->assignments[assignment_index];
}

static loom_value_id_t loom_low_allocation_packet_move_result(
    const loom_op_t* op,
    loom_low_allocation_packet_move_op_kind_t packet_move_kind) {
  switch (packet_move_kind) {
    case LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_COPY:
      return loom_low_copy_result(op);
    case LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_MOVE:
      return loom_low_move_result(op);
    case LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_SLICE:
      return loom_low_slice_result(op);
    case LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_CONCAT:
      return loom_low_concat_result(op);
    case LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_NONE:
      IREE_ASSERT_UNREACHABLE("packet-move operation must have a result");
      return LOOM_VALUE_ID_INVALID;
  }
  IREE_ASSERT_UNREACHABLE("unknown packet-move operation kind");
  return LOOM_VALUE_ID_INVALID;
}

static loom_low_placement_cause_t loom_low_allocation_packet_move_cause(
    loom_low_allocation_packet_move_op_kind_t packet_move_kind) {
  switch (packet_move_kind) {
    case LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_COPY:
      return LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY;
    case LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_MOVE:
      return LOOM_LOW_PLACEMENT_CAUSE_LOW_MOVE;
    case LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_SLICE:
      return LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE;
    case LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_CONCAT:
      return LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT;
    case LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_NONE:
      IREE_ASSERT_UNREACHABLE("packet-move group must have a placement cause");
      return LOOM_LOW_PLACEMENT_CAUSE_UNKNOWN;
  }
  IREE_ASSERT_UNREACHABLE("unknown packet-move operation kind");
  return LOOM_LOW_PLACEMENT_CAUSE_UNKNOWN;
}

static iree_status_t loom_low_allocation_packet_move_record_group(
    loom_low_allocation_packet_move_builder_t* builder, const loom_op_t* op,
    uint32_t source_ordinal,
    loom_low_allocation_packet_move_op_kind_t packet_move_kind) {
  const loom_low_allocation_packet_move_context_t* context = builder->context;
  loom_value_ordinal_t result_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  const bool result_found =
      loom_low_allocation_assignment_map_value_ordinal_for_value(
          &context->move_plan->context.assignment_map,
          loom_low_allocation_packet_move_result(op, packet_move_kind),
          &result_ordinal);
  IREE_ASSERT(result_found,
              "packet-move result must belong to allocation liveness");
  const loom_low_placement_cause_t cause =
      loom_low_allocation_packet_move_cause(packet_move_kind);

  loom_low_move_t* raw_moves =
      loom_low_allocation_move_plan_raw_moves(context->move_plan);
  iree_host_size_t raw_move_count = 0;
  const bool omit_concat =
      packet_move_kind == LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_CONCAT &&
      !loom_low_allocation_move_topology_concat_requires_packet_materialization_for_module(
          context->move_plan->context.assignment_map.module, op);
  if (!omit_concat) {
    const loom_low_placement_relation_range_t range =
        loom_low_placement_relation_range_for_value_ordinal(context->placement,
                                                            result_ordinal);
    for (uint32_t i = 0; i < range.count; ++i) {
      const loom_low_placement_relation_t* relation =
          &context->placement->relations[range.start + i];
      if (relation->op != op || relation->cause != cause) {
        continue;
      }
      const loom_low_allocation_assignment_t* destination_assignment =
          loom_low_allocation_packet_move_assignment(
              &context->move_plan->context.assignment_map,
              relation->result_ordinal);
      const loom_low_allocation_assignment_t* source_assignment =
          loom_low_allocation_packet_move_assignment(
              &context->move_plan->context.assignment_map,
              relation->source_ordinal);
      for (uint32_t unit_index = 0; unit_index < relation->unit_count;
           ++unit_index) {
        raw_moves[raw_move_count++] = (loom_low_move_t){
            .destination = loom_low_allocation_assignment_unit_location(
                destination_assignment,
                relation->result_unit_offset + unit_index),
            .source = loom_low_allocation_assignment_unit_location(
                source_assignment, relation->source_unit_offset + unit_index),
        };
      }
    }
  }

  loom_low_move_group_t move_group = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_move_plan_append_group(
      context->move_plan, op, raw_move_count, &move_group));
  builder->plan.move_count += move_group.moves.count;
  if (move_group.moves.count != 0) {
    builder->plan.groups[builder->plan.group_count++] =
        (loom_low_allocation_packet_move_group_t){
            .source_ordinal = source_ordinal,
            .cause = cause,
            .move_group = move_group,
        };
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_packet_move_record_region(
    loom_low_allocation_packet_move_builder_t* builder,
    const loom_region_t* region, uint32_t* inout_source_ordinal) {
  const loom_low_allocation_move_plan_context_t* move_context =
      &builder->context->move_plan->context;
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_low_allocation_packet_move_op_kind_t packet_move_kind =
          loom_low_allocation_move_topology_packet_move_op_kind(op);
      const bool has_move_relations =
          packet_move_kind != LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_NONE &&
          (packet_move_kind != LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_CONCAT ||
           loom_low_concat_sources(op).count != 0);
      if (has_move_relations) {
        IREE_RETURN_IF_ERROR(loom_low_allocation_packet_move_record_group(
            builder, op, *inout_source_ordinal, packet_move_kind));
        if (move_context->target_constraints->error_count != 0) {
          return iree_ok_status();
        }
      }
      ++*inout_source_ordinal;
      if (!loom_liveness_analysis_includes_region_tree(
              move_context->assignment_map.liveness)) {
        continue;
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (regions[i] == NULL) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_low_allocation_packet_move_record_region(
            builder, regions[i], inout_source_ordinal));
        if (move_context->target_constraints->error_count != 0) {
          return iree_ok_status();
        }
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_packet_move_plan_build(
    const loom_low_allocation_packet_move_context_t* context,
    iree_arena_allocator_t* arena,
    loom_low_allocation_packet_move_plan_t* out_plan) {
  *out_plan = (loom_low_allocation_packet_move_plan_t){0};
  const iree_host_size_t group_capacity =
      context->placement->packet_move_group_count;
  if (group_capacity == 0) {
    return iree_ok_status();
  }
  loom_low_allocation_packet_move_builder_t builder = {
      .context = context,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, group_capacity,
                                                 sizeof(*builder.plan.groups),
                                                 (void**)&builder.plan.groups));
  uint32_t source_ordinal = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_packet_move_record_region(
      &builder, context->move_plan->context.assignment_map.liveness->region,
      &source_ordinal));
  *out_plan = builder.plan;
  return iree_ok_status();
}
