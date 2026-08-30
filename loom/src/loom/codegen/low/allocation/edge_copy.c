// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/edge_copy.h"

#include "loom/codegen/low/allocation/unit_location.h"
#include "loom/ops/low/ops.h"

typedef struct loom_low_allocation_edge_copy_builder_t {
  // Immutable allocation facts used to construct edge-copy rows.
  const loom_low_allocation_edge_copy_context_t* context;
  // Plan receiving semantic segments and final move groups.
  loom_low_allocation_edge_copy_plan_t plan;
  // Number of raw rows populated for the current branch group.
  iree_host_size_t raw_move_count;
} loom_low_allocation_edge_copy_builder_t;

static loom_value_ordinal_t loom_low_allocation_edge_copy_value_ordinal(
    const loom_low_allocation_edge_copy_context_t* context,
    loom_value_id_t value_id) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  const bool found = loom_low_allocation_assignment_map_value_ordinal_for_value(
      &context->move_plan->context.assignment_map, value_id, &value_ordinal);
  IREE_ASSERT(found,
              "verified edge-copy value must belong to allocation liveness");
  return value_ordinal;
}

static const loom_low_placement_relation_t*
loom_low_allocation_edge_copy_branch_relation(
    const loom_low_allocation_edge_copy_context_t* context,
    loom_value_id_t payload_value_id, loom_value_id_t destination_value_id) {
  const loom_value_ordinal_t payload_ordinal =
      loom_low_allocation_edge_copy_value_ordinal(context, payload_value_id);
  const loom_value_ordinal_t destination_ordinal =
      loom_low_allocation_edge_copy_value_ordinal(context,
                                                  destination_value_id);
  const loom_low_placement_relation_range_t range =
      loom_low_placement_relation_range_for_value_ordinal(context->placement,
                                                          destination_ordinal);
  for (uint32_t i = 0; i < range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[range.start + i];
    if (relation->cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH &&
        relation->source_ordinal == payload_ordinal) {
      return relation;
    }
  }
  IREE_ASSERT_UNREACHABLE(
      "verified low.br payload must have a placement relation");
  return NULL;
}

static const loom_low_allocation_assignment_t*
loom_low_allocation_edge_copy_assignment(
    const loom_low_allocation_assignment_map_t* map,
    loom_value_ordinal_t value_ordinal, uint32_t* out_assignment_index) {
  const uint32_t assignment_index =
      map->assignment_indices_by_value_ordinal[value_ordinal];
  IREE_ASSERT_NE(assignment_index, UINT32_MAX,
                 "allocated edge-copy value must have an assignment");
  *out_assignment_index = assignment_index;
  return &map->assignments[assignment_index];
}

static bool loom_low_allocation_edge_copy_concat_relation_covers_branch_source(
    const loom_low_placement_relation_t* concat_relation,
    const loom_low_placement_relation_t* branch_relation) {
  if (concat_relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT ||
      concat_relation->result_ordinal != branch_relation->source_ordinal ||
      concat_relation->result_unit_offset <
          branch_relation->source_unit_offset) {
    return false;
  }
  const uint64_t concat_source_end =
      (uint64_t)concat_relation->result_unit_offset +
      concat_relation->unit_count;
  const uint64_t branch_source_end =
      (uint64_t)branch_relation->source_unit_offset +
      branch_relation->unit_count;
  return concat_source_end <= branch_source_end;
}

static void loom_low_allocation_edge_copy_record_segment(
    loom_low_allocation_edge_copy_builder_t* builder, uint16_t payload_index,
    loom_value_ordinal_t source_ordinal,
    loom_value_ordinal_t destination_ordinal, uint32_t source_unit_offset,
    uint32_t destination_unit_offset, uint32_t unit_count) {
  const loom_low_allocation_edge_copy_context_t* context = builder->context;
  uint32_t source_assignment_index = 0;
  const loom_low_allocation_assignment_t* source_assignment =
      loom_low_allocation_edge_copy_assignment(
          &context->move_plan->context.assignment_map, source_ordinal,
          &source_assignment_index);
  uint32_t destination_assignment_index = 0;
  const loom_low_allocation_assignment_t* destination_assignment =
      loom_low_allocation_edge_copy_assignment(
          &context->move_plan->context.assignment_map, destination_ordinal,
          &destination_assignment_index);

  builder->plan.copies[builder->plan.copy_count++] =
      (loom_low_allocation_edge_copy_t){
          .payload_index = payload_index,
          .source_value_id =
              loom_low_placement_value_id(context->placement, source_ordinal),
          .destination_value_id = loom_low_placement_value_id(
              context->placement, destination_ordinal),
          .source_assignment_index = source_assignment_index,
          .destination_assignment_index = destination_assignment_index,
          .source_unit_offset = source_unit_offset,
          .destination_unit_offset = destination_unit_offset,
          .unit_count = unit_count,
      };

  loom_low_move_t* raw_moves =
      loom_low_allocation_move_plan_raw_moves(context->move_plan);
  for (uint32_t i = 0; i < unit_count; ++i) {
    raw_moves[builder->raw_move_count++] = (loom_low_move_t){
        .destination = loom_low_allocation_assignment_unit_location(
            context->move_plan->context.descriptor_set, destination_assignment,
            destination_unit_offset + i),
        .source = loom_low_allocation_assignment_unit_location(
            context->move_plan->context.descriptor_set, source_assignment,
            source_unit_offset + i),
    };
  }
}

static void loom_low_allocation_edge_copy_record_branch_payload_segments(
    loom_low_allocation_edge_copy_builder_t* builder, uint16_t payload_index,
    loom_value_id_t payload_value_id, loom_value_id_t destination_value_id) {
  const loom_low_allocation_edge_copy_context_t* context = builder->context;
  const loom_low_placement_relation_t* branch_relation =
      loom_low_allocation_edge_copy_branch_relation(context, payload_value_id,
                                                    destination_value_id);
  const loom_low_placement_relation_range_t range =
      loom_low_placement_relation_range_for_value_ordinal(
          context->placement, branch_relation->source_ordinal);
  uint64_t covered_unit_count = 0;
  for (uint32_t i = 0; i < range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[range.start + i];
    if (!loom_low_allocation_edge_copy_concat_relation_covers_branch_source(
            relation, branch_relation)) {
      continue;
    }
    const uint32_t branch_source_delta =
        relation->result_unit_offset - branch_relation->source_unit_offset;
    loom_low_allocation_edge_copy_record_segment(
        builder, payload_index, relation->source_ordinal,
        branch_relation->result_ordinal, relation->source_unit_offset,
        branch_relation->result_unit_offset + branch_source_delta,
        relation->unit_count);
    covered_unit_count += relation->unit_count;
  }
  if (covered_unit_count != 0) {
    IREE_ASSERT_EQ(covered_unit_count, branch_relation->unit_count,
                   "concat placement must cover the full branch payload");
    return;
  }
  loom_low_allocation_edge_copy_record_segment(
      builder, payload_index, branch_relation->source_ordinal,
      branch_relation->result_ordinal, branch_relation->source_unit_offset,
      branch_relation->result_unit_offset, branch_relation->unit_count);
}

static iree_status_t loom_low_allocation_edge_copy_record_group(
    loom_low_allocation_edge_copy_builder_t* builder, const loom_op_t* op,
    uint32_t source_ordinal) {
  const loom_value_slice_t args = loom_low_br_args(op);
  if (args.count == 0) {
    return iree_ok_status();
  }
  const loom_block_t* destination_block = loom_low_br_dest(op);
  IREE_ASSERT_EQ(args.count, destination_block->arg_count,
                 "verified low.br payload must match destination arguments");
  loom_low_allocation_edge_copy_group_t* group =
      &builder->plan.groups[builder->plan.group_count++];
  *group = (loom_low_allocation_edge_copy_group_t){
      .terminator_op = op,
      .source_ordinal = source_ordinal,
      .copy_start = builder->plan.copy_count,
  };

  const iree_host_size_t copy_start = builder->plan.copy_count;
  builder->raw_move_count = 0;
  for (uint16_t i = 0; i < destination_block->arg_count; ++i) {
    loom_low_allocation_edge_copy_record_branch_payload_segments(
        builder, i, args.values[i], destination_block->arg_ids[i]);
  }
  group->copy_count = builder->plan.copy_count - copy_start;

  const loom_low_allocation_edge_copy_context_t* context = builder->context;
  return loom_low_allocation_move_plan_append_group(
      context->move_plan, op, builder->raw_move_count, &group->move_group);
}

static iree_status_t loom_low_allocation_edge_copy_record_region(
    loom_low_allocation_edge_copy_builder_t* builder,
    const loom_region_t* region, uint32_t* inout_source_ordinal) {
  const loom_low_allocation_move_plan_context_t* move_context =
      &builder->context->move_plan->context;
  loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_br_isa(op)) {
        IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_record_group(
            builder, op, *inout_source_ordinal));
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
        IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_record_region(
            builder, regions[i], inout_source_ordinal));
        if (move_context->target_constraints->error_count != 0) {
          return iree_ok_status();
        }
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_edge_copy_plan_build(
    const loom_low_allocation_edge_copy_context_t* context,
    iree_arena_allocator_t* arena,
    loom_low_allocation_edge_copy_plan_t* out_plan) {
  *out_plan = (loom_low_allocation_edge_copy_plan_t){0};
  const iree_host_size_t group_capacity =
      context->placement->edge_copy_group_count;
  if (group_capacity == 0) {
    return iree_ok_status();
  }
  const iree_host_size_t copy_capacity = context->placement->branch_unit_count;
  loom_low_allocation_edge_copy_builder_t builder = {
      .context = context,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, copy_capacity,
                                                 sizeof(*builder.plan.copies),
                                                 (void**)&builder.plan.copies));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, group_capacity,
                                                 sizeof(*builder.plan.groups),
                                                 (void**)&builder.plan.groups));
  uint32_t source_ordinal = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_record_region(
      &builder, context->move_plan->context.assignment_map.liveness->region,
      &source_ordinal));
  *out_plan = builder.plan;
  return iree_ok_status();
}
