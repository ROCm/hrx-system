// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/packet_move.h"

#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/move_topology.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/ops/low/ops.h"

typedef enum loom_low_allocation_packet_transfer_kind_e {
  LOOM_LOW_ALLOCATION_PACKET_TRANSFER_DEAD = 0,
  LOOM_LOW_ALLOCATION_PACKET_TRANSFER_FORWARDED = 1,
  LOOM_LOW_ALLOCATION_PACKET_TRANSFER_MATERIALIZED = 2,
} loom_low_allocation_packet_transfer_kind_t;

enum loom_low_allocation_packet_transfer_presence_bit_e {
  LOOM_LOW_ALLOCATION_PACKET_TRANSFER_PRESENCE_DEAD = 1u << 0,
  LOOM_LOW_ALLOCATION_PACKET_TRANSFER_PRESENCE_FORWARDED = 1u << 1,
  LOOM_LOW_ALLOCATION_PACKET_TRANSFER_PRESENCE_MATERIALIZED = 1u << 2,
};
typedef uint8_t loom_low_allocation_packet_transfer_presence_t;

typedef enum loom_low_allocation_packet_transfer_phase_e {
  LOOM_LOW_ALLOCATION_PACKET_TRANSFER_PHASE_COUNT = 0,
  LOOM_LOW_ALLOCATION_PACKET_TRANSFER_PHASE_FILL = 1,
} loom_low_allocation_packet_transfer_phase_t;

typedef struct loom_low_allocation_packet_move_builder_t {
  // Immutable allocation facts used to construct packet-local moves.
  const loom_low_allocation_packet_move_context_t* context;
  // Plan receiving final move groups.
  loom_low_allocation_packet_move_plan_t plan;
  // Number of group records available in |plan.groups|.
  iree_host_size_t group_capacity;
  // Next completed group being populated with exact transfer rows.
  iree_host_size_t fill_group_index;
  // Next exact transfer row being populated.
  iree_host_size_t transfer_index;
  // Number of raw physical moves in the current group.
  iree_host_size_t raw_move_count;
  // Source-preorder position shared by the construction region walk.
  loom_low_allocation_move_cursor_t cursor;
} loom_low_allocation_packet_move_builder_t;

static const loom_low_allocation_assignment_t*
loom_low_allocation_packet_move_try_assignment(
    const loom_low_allocation_assignment_map_t* map,
    loom_value_ordinal_t value_ordinal) {
  const uint32_t assignment_index =
      map->assignment_indices_by_value_ordinal[value_ordinal];
  return assignment_index != UINT32_MAX ? &map->assignments[assignment_index]
                                        : NULL;
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

static bool loom_low_allocation_packet_move_has_relations(
    const loom_op_t* op,
    loom_low_allocation_packet_move_op_kind_t packet_move_kind) {
  return packet_move_kind != LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_NONE &&
         (packet_move_kind != LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_CONCAT ||
          loom_low_concat_sources(op).count != 0);
}

static loom_low_placement_relation_range_t
loom_low_allocation_packet_move_relation_range(
    const loom_low_allocation_packet_move_context_t* context,
    const loom_op_t* op,
    loom_low_allocation_packet_move_op_kind_t packet_move_kind) {
  loom_value_ordinal_t result_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  const bool result_found =
      loom_low_allocation_assignment_map_value_ordinal_for_value(
          &context->move_plan->context.assignment_map,
          loom_low_allocation_packet_move_result(op, packet_move_kind),
          &result_ordinal);
  IREE_ASSERT(result_found,
              "packet-move result must belong to allocation liveness");
  return loom_low_placement_relation_range_for_value_ordinal(context->placement,
                                                             result_ordinal);
}

static bool loom_low_allocation_packet_move_unit_is_live(
    const loom_low_allocation_packet_move_context_t* context,
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_offset) {
  const loom_low_allocation_unit_liveness_t* unit_liveness =
      context->move_plan->context.unit_liveness;
  const uint32_t start_point =
      loom_low_allocation_live_range_assignment_unit_start_point(
          unit_liveness->start_points, unit_liveness->point_count, assignment,
          unit_offset);
  const uint32_t end_point =
      loom_low_allocation_live_range_assignment_unit_end_point(
          unit_liveness->end_points, unit_liveness->point_count, assignment,
          unit_offset);
  return start_point < end_point;
}

static loom_low_allocation_packet_transfer_kind_t
loom_low_allocation_packet_move_classify_unit(
    const loom_low_allocation_packet_move_context_t* context,
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* source_assignment,
    const loom_low_allocation_assignment_t* destination_assignment,
    uint32_t relation_unit_offset) {
  const uint32_t destination_unit =
      relation->result_unit_offset + relation_unit_offset;
  if (!loom_low_allocation_packet_move_unit_is_live(
          context, destination_assignment, destination_unit)) {
    return LOOM_LOW_ALLOCATION_PACKET_TRANSFER_DEAD;
  }
  IREE_ASSERT(source_assignment != NULL,
              "live packet transfer source must have an assignment");
  if (relation->cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT &&
      !iree_any_bit_set(relation->flags,
                        LOOM_LOW_PLACEMENT_RELATION_FLAG_MATERIALIZE_PART)) {
    return LOOM_LOW_ALLOCATION_PACKET_TRANSFER_FORWARDED;
  }
  return loom_low_allocation_storage_assignment_subranges_correspond(
             context->move_plan->context.descriptor_set, source_assignment,
             relation->source_unit_offset + relation_unit_offset,
             destination_assignment, destination_unit, 1)
             ? LOOM_LOW_ALLOCATION_PACKET_TRANSFER_FORWARDED
             : LOOM_LOW_ALLOCATION_PACKET_TRANSFER_MATERIALIZED;
}

IREE_ATTRIBUTE_NOINLINE static loom_low_allocation_packet_transfer_presence_t
loom_low_allocation_packet_move_append_relation(
    loom_low_allocation_packet_move_builder_t* builder,
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* source_assignment,
    const loom_low_allocation_assignment_t* destination_assignment) {
  if (relation->unit_count == 0 || destination_assignment == NULL) {
    return LOOM_LOW_ALLOCATION_PACKET_TRANSFER_PRESENCE_DEAD;
  }
  if (relation->cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT &&
      !iree_any_bit_set(relation->flags,
                        LOOM_LOW_PLACEMENT_RELATION_FLAG_MATERIALIZE_PART)) {
    return LOOM_LOW_ALLOCATION_PACKET_TRANSFER_PRESENCE_FORWARDED;
  }
  if (source_assignment == NULL) {
    for (uint32_t unit = 0; unit < relation->unit_count; ++unit) {
      IREE_ASSERT(!loom_low_allocation_packet_move_unit_is_live(
                      builder->context, destination_assignment,
                      relation->result_unit_offset + unit),
                  "live packet transfer source must have an assignment");
    }
    return LOOM_LOW_ALLOCATION_PACKET_TRANSFER_PRESENCE_DEAD;
  }

  const iree_host_size_t move_start = builder->raw_move_count;
  loom_low_allocation_move_plan_append_assignment(
      builder->context->move_plan, source_assignment,
      relation->source_unit_offset, destination_assignment,
      relation->result_unit_offset, relation->unit_count,
      &builder->raw_move_count);
  const iree_host_size_t live_unit_count = builder->raw_move_count - move_start;
  return live_unit_count == relation->unit_count
             ? 0
             : LOOM_LOW_ALLOCATION_PACKET_TRANSFER_PRESENCE_DEAD;
}

static void loom_low_allocation_packet_move_record_exact_transfer(
    loom_low_allocation_packet_move_builder_t* builder, uint32_t relation_index,
    loom_low_allocation_packet_transfer_kind_t kind,
    uint32_t relation_unit_offset, uint32_t unit_count,
    loom_low_allocation_packet_transfer_phase_t phase) {
  IREE_ASSERT(kind != LOOM_LOW_ALLOCATION_PACKET_TRANSFER_DEAD);
  IREE_ASSERT_NE(unit_count, 0u);
  if (phase == LOOM_LOW_ALLOCATION_PACKET_TRANSFER_PHASE_COUNT) {
    ++builder->plan.transfer_count;
    return;
  }

  IREE_ASSERT_LT(builder->transfer_index, builder->plan.transfer_count);
  builder->plan.transfers[builder->transfer_index++] =
      (loom_low_allocation_packet_transfer_t){
          .encoded_relation_index =
              relation_index |
              (kind == LOOM_LOW_ALLOCATION_PACKET_TRANSFER_MATERIALIZED
                   ? LOOM_LOW_ALLOCATION_PACKET_TRANSFER_MATERIALIZED_BIT
                   : 0),
          .relation_unit_offset = relation_unit_offset,
          .unit_count = unit_count,
      };
}

static void loom_low_allocation_packet_move_scan_exact_relation(
    loom_low_allocation_packet_move_builder_t* builder, uint32_t relation_index,
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* source_assignment,
    const loom_low_allocation_assignment_t* destination_assignment,
    loom_low_allocation_packet_transfer_phase_t phase) {
  if (relation->unit_count == 0 || destination_assignment == NULL) {
    return;
  }
  loom_low_allocation_packet_transfer_kind_t run_kind =
      loom_low_allocation_packet_move_classify_unit(builder->context, relation,
                                                    source_assignment,
                                                    destination_assignment, 0);
  uint32_t run_start = 0;
  for (uint32_t unit = 1; unit < relation->unit_count; ++unit) {
    const loom_low_allocation_packet_transfer_kind_t kind =
        loom_low_allocation_packet_move_classify_unit(
            builder->context, relation, source_assignment,
            destination_assignment, unit);
    if (kind == run_kind) {
      continue;
    }
    if (run_kind != LOOM_LOW_ALLOCATION_PACKET_TRANSFER_DEAD) {
      loom_low_allocation_packet_move_record_exact_transfer(
          builder, relation_index, run_kind, run_start, unit - run_start,
          phase);
    }
    run_kind = kind;
    run_start = unit;
  }
  if (run_kind != LOOM_LOW_ALLOCATION_PACKET_TRANSFER_DEAD) {
    loom_low_allocation_packet_move_record_exact_transfer(
        builder, relation_index, run_kind, run_start,
        relation->unit_count - run_start, phase);
  }
}

IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static void
loom_low_allocation_packet_move_scan_exact_group(
    loom_low_allocation_packet_move_builder_t* builder,
    const loom_low_allocation_packet_move_group_t* group, const loom_op_t* op,
    loom_low_allocation_packet_move_op_kind_t packet_move_kind,
    loom_low_allocation_packet_transfer_phase_t phase) {
  const loom_low_allocation_packet_move_context_t* context = builder->context;
  const loom_low_placement_relation_range_t range =
      loom_low_allocation_packet_move_relation_range(context, op,
                                                     packet_move_kind);
  const loom_low_allocation_assignment_map_t* assignment_map =
      &context->move_plan->context.assignment_map;
  for (uint32_t i = 0; i < range.count; ++i) {
    const uint32_t relation_index = range.start + i;
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[relation_index];
    if (relation->op != op || relation->cause != group->cause) {
      continue;
    }
    const loom_low_allocation_assignment_t* destination_assignment =
        loom_low_allocation_packet_move_try_assignment(
            assignment_map, relation->result_ordinal);
    const loom_low_allocation_assignment_t* source_assignment =
        loom_low_allocation_packet_move_try_assignment(
            assignment_map, relation->source_ordinal);
    loom_low_allocation_packet_move_scan_exact_relation(
        builder, relation_index, relation, source_assignment,
        destination_assignment, phase);
  }
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_low_allocation_packet_move_build_group(
    loom_low_allocation_packet_move_builder_t* builder,
    const loom_liveness_operation_point_t* operation_point,
    uint32_t source_ordinal,
    loom_low_allocation_packet_move_op_kind_t packet_move_kind) {
  IREE_ASSERT_LT(builder->plan.group_count, builder->group_capacity);
  const loom_op_t* op = operation_point->op;
  loom_low_allocation_packet_move_group_t* group =
      &builder->plan.groups[builder->plan.group_count++];
  *group = (loom_low_allocation_packet_move_group_t){
      .source_ordinal = source_ordinal,
      .cause = loom_low_allocation_packet_move_cause(packet_move_kind),
  };

  const loom_low_allocation_packet_move_context_t* context = builder->context;
  const loom_low_placement_relation_range_t range =
      loom_low_allocation_packet_move_relation_range(context, op,
                                                     packet_move_kind);
  const loom_low_allocation_assignment_map_t* assignment_map =
      &context->move_plan->context.assignment_map;
  loom_low_allocation_packet_transfer_presence_t presence = 0;
  builder->raw_move_count = 0;
  for (uint32_t i = 0; i < range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[range.start + i];
    if (relation->op != op || relation->cause != group->cause) {
      continue;
    }
    const loom_low_allocation_assignment_t* destination_assignment =
        loom_low_allocation_packet_move_try_assignment(
            assignment_map, relation->result_ordinal);
    const loom_low_allocation_assignment_t* source_assignment =
        loom_low_allocation_packet_move_try_assignment(
            assignment_map, relation->source_ordinal);
    presence |= loom_low_allocation_packet_move_append_relation(
        builder, relation, source_assignment, destination_assignment);
  }

  loom_low_move_sequence_input_flags_t input_flags = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_move_plan_append_group(
      context->move_plan, op, operation_point->start_point,
      operation_point->end_point, builder->raw_move_count, &group->move_group,
      &input_flags));
  builder->plan.move_count += group->move_group.moves.count;
  if (context->move_plan->context.target_constraints->error_count != 0) {
    return iree_ok_status();
  }
  if (iree_any_bit_set(input_flags,
                       LOOM_LOW_MOVE_SEQUENCE_INPUT_FLAG_IDENTITY)) {
    presence |= LOOM_LOW_ALLOCATION_PACKET_TRANSFER_PRESENCE_FORWARDED;
  }
  if (iree_any_bit_set(input_flags, LOOM_LOW_MOVE_SEQUENCE_INPUT_FLAG_ACTIVE)) {
    presence |= LOOM_LOW_ALLOCATION_PACKET_TRANSFER_PRESENCE_MATERIALIZED;
  }
  if (iree_any_bit_set(
          presence, LOOM_LOW_ALLOCATION_PACKET_TRANSFER_PRESENCE_FORWARDED)) {
    group->transfer_flags |=
        LOOM_LOW_ALLOCATION_PACKET_TRANSFER_GROUP_FLAG_FORWARDED;
  }
  if (iree_any_bit_set(
          presence,
          LOOM_LOW_ALLOCATION_PACKET_TRANSFER_PRESENCE_MATERIALIZED)) {
    group->transfer_flags |=
        LOOM_LOW_ALLOCATION_PACKET_TRANSFER_GROUP_FLAG_MATERIALIZED;
  }
  const bool requires_exact_ranges =
      iree_any_bit_set(
          presence,
          LOOM_LOW_ALLOCATION_PACKET_TRANSFER_PRESENCE_MATERIALIZED) &&
      iree_any_bit_set(
          presence, LOOM_LOW_ALLOCATION_PACKET_TRANSFER_PRESENCE_DEAD |
                        LOOM_LOW_ALLOCATION_PACKET_TRANSFER_PRESENCE_FORWARDED);
  if (requires_exact_ranges) {
    group->transfer_flags |=
        LOOM_LOW_ALLOCATION_PACKET_TRANSFER_GROUP_FLAG_EXACT;
    const iree_host_size_t transfer_start = builder->plan.transfer_count;
    loom_low_allocation_packet_move_scan_exact_group(
        builder, group, op, packet_move_kind,
        LOOM_LOW_ALLOCATION_PACKET_TRANSFER_PHASE_COUNT);
    const iree_host_size_t transfer_count =
        builder->plan.transfer_count - transfer_start;
    if (builder->plan.transfer_count > UINT32_MAX ||
        transfer_count > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "packet transfer run count exceeds the compact index domain");
    }
    group->transfer_start = (uint32_t)transfer_start;
    group->transfer_count = (uint32_t)transfer_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_packet_move_build_region(
    loom_low_allocation_packet_move_builder_t* builder,
    const loom_region_t* region, uint32_t* inout_source_ordinal) {
  const loom_low_allocation_move_plan_context_t* move_context =
      &builder->context->move_plan->context;
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_liveness_operation_point_t* operation_point =
          loom_low_allocation_move_plan_next_operation(
              builder->context->move_plan, op, &builder->cursor);
      const loom_low_allocation_packet_move_op_kind_t packet_move_kind =
          loom_low_allocation_move_topology_packet_move_op_kind(op);
      if (loom_low_allocation_packet_move_has_relations(op, packet_move_kind)) {
        IREE_RETURN_IF_ERROR(loom_low_allocation_packet_move_build_group(
            builder, operation_point, *inout_source_ordinal, packet_move_kind));
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
        IREE_RETURN_IF_ERROR(loom_low_allocation_packet_move_build_region(
            builder, regions[i], inout_source_ordinal));
        if (move_context->target_constraints->error_count != 0) {
          return iree_ok_status();
        }
      }
    }
  }
  return iree_ok_status();
}

IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static void
loom_low_allocation_packet_move_fill_region(
    loom_low_allocation_packet_move_builder_t* builder,
    const loom_region_t* region, uint32_t* inout_source_ordinal) {
  const loom_liveness_analysis_t* liveness =
      builder->context->move_plan->context.assignment_map.liveness;
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_low_allocation_packet_move_op_kind_t packet_move_kind =
          loom_low_allocation_move_topology_packet_move_op_kind(op);
      if (loom_low_allocation_packet_move_has_relations(op, packet_move_kind)) {
        IREE_ASSERT_LT(builder->fill_group_index, builder->plan.group_count);
        const loom_low_allocation_packet_move_group_t* group =
            &builder->plan.groups[builder->fill_group_index++];
        IREE_ASSERT_EQ(group->source_ordinal, *inout_source_ordinal);
        IREE_ASSERT_EQ(group->cause,
                       loom_low_allocation_packet_move_cause(packet_move_kind));
        if (iree_any_bit_set(
                group->transfer_flags,
                LOOM_LOW_ALLOCATION_PACKET_TRANSFER_GROUP_FLAG_EXACT)) {
          builder->transfer_index = group->transfer_start;
          loom_low_allocation_packet_move_scan_exact_group(
              builder, group, op, packet_move_kind,
              LOOM_LOW_ALLOCATION_PACKET_TRANSFER_PHASE_FILL);
          IREE_ASSERT_EQ(builder->transfer_index,
                         group->transfer_start + group->transfer_count);
        }
      }
      ++*inout_source_ordinal;
      if (!loom_liveness_analysis_includes_region_tree(liveness)) {
        continue;
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (regions[i] != NULL) {
          loom_low_allocation_packet_move_fill_region(builder, regions[i],
                                                      inout_source_ordinal);
        }
      }
    }
  }
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
  if ((uint64_t)context->placement->relation_count >
      (uint64_t)LOOM_LOW_ALLOCATION_PACKET_TRANSFER_RELATION_INDEX_MASK + 1) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "placement relation count exceeds the compact packet transfer domain");
  }

  loom_low_allocation_packet_move_builder_t builder = {
      .context = context,
      .group_capacity = group_capacity,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, group_capacity,
                                                 sizeof(*builder.plan.groups),
                                                 (void**)&builder.plan.groups));
  uint32_t source_ordinal = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_packet_move_build_region(
      &builder, context->move_plan->context.assignment_map.liveness->region,
      &source_ordinal));
  if (context->move_plan->context.target_constraints->error_count != 0) {
    *out_plan = builder.plan;
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(builder.plan.group_count, group_capacity);
  if (builder.plan.transfer_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, builder.plan.transfer_count, sizeof(*builder.plan.transfers),
        (void**)&builder.plan.transfers));
    source_ordinal = 0;
    loom_low_allocation_packet_move_fill_region(
        &builder, context->move_plan->context.assignment_map.liveness->region,
        &source_ordinal);
    IREE_ASSERT_EQ(builder.fill_group_index, builder.plan.group_count);
  }
  *out_plan = builder.plan;
  return iree_ok_status();
}
