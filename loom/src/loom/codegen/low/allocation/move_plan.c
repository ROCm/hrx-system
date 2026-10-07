// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/move_plan.h"

#include <string.h>

#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/allocation/unit_location.h"
#include "loom/codegen/low/allocation/write_interference.h"
#include "loom/codegen/low/schedule/types.h"

typedef struct loom_low_allocation_move_plan_group_context_t {
  // Plan owning shared allocation facts and persistent output rows.
  loom_low_allocation_move_plan_t* plan;
  // Diagnostic owner: the function for entry transport, otherwise the move op.
  const loom_op_t* owner_op;
  // Accepted liveness position where the parallel transfer reads its inputs.
  uint32_t read_point;
  // Accepted write position used by retained-read interference constraints.
  uint32_t write_point;
  // Global move-row index corresponding to local output row zero.
  iree_host_size_t move_start;
} loom_low_allocation_move_plan_group_context_t;

static iree_status_t loom_low_allocation_move_plan_record_scratch(
    void* user_data, iree_host_size_t move_index) {
  loom_low_allocation_move_plan_group_context_t* group_context =
      (loom_low_allocation_move_plan_group_context_t*)user_data;
  loom_low_allocation_move_plan_t* plan = group_context->plan;
  if (!plan->scratch_move_indices) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->output_arena, plan->scratch_move_index_capacity,
        sizeof(*plan->scratch_move_indices),
        (void**)&plan->scratch_move_indices));
  }
  plan->scratch_move_indices[plan->scratch_move_index_count++] =
      group_context->move_start + move_index;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_move_plan_resolve_temporary(
    void* user_data, const loom_low_move_location_t* storage_class,
    const loom_low_move_sequence_location_set_t* occupied_locations,
    loom_low_move_location_t* out_temporary, bool* out_resolved) {
  const loom_low_allocation_move_plan_group_context_t* group_context =
      (const loom_low_allocation_move_plan_group_context_t*)user_data;
  const loom_low_allocation_move_plan_context_t* context =
      &group_context->plan->context;
  const loom_op_t* op = group_context->owner_op;
  *out_temporary = (loom_low_move_location_t){0};
  *out_resolved = false;
  if (!loom_low_allocation_location_kind_is_register_like(
          storage_class->location_kind)) {
    loom_low_allocation_target_constraints_record_move_failure(
        context->target_constraints, op, storage_class->descriptor_reg_class_id,
        0, 1, IREE_SV("parallel-move-non-register-storage"));
    return iree_ok_status();
  }

  loom_low_allocation_class_capacity_t capacity = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_target_constraints_reg_class_capacity(
          context->target_constraints, storage_class->descriptor_reg_class_id,
          &capacity));
  if (capacity.location_kind != storage_class->location_kind) {
    loom_low_allocation_target_constraints_record_move_failure(
        context->target_constraints, op, storage_class->descriptor_reg_class_id,
        capacity.is_bounded ? capacity.max_units : UINT32_MAX, 1,
        IREE_SV("parallel-move-storage-kind-mismatch"));
    return iree_ok_status();
  }

  const loom_low_reg_class_t* reg_class =
      &context->descriptor_set
           ->reg_classes[storage_class->descriptor_reg_class_id];
  const bool uses_explicit_physical_registers =
      loom_low_reg_class_uses_explicit_physical_registers(reg_class);

  uint32_t last_location = 0;
  if (capacity.is_bounded) {
    if (capacity.max_units == 0) {
      loom_low_allocation_target_constraints_record_move_failure(
          context->target_constraints, op,
          storage_class->descriptor_reg_class_id, capacity.max_units, 1,
          IREE_SV("parallel-move-empty-budget"));
      return iree_ok_status();
    }
    last_location = capacity.max_units - 1u;
  } else {
    last_location =
        loom_low_allocation_target_constraints_assigned_location_search_limit(
            context->target_constraints, storage_class->descriptor_reg_class_id,
            storage_class->location_kind);
    if (last_location == UINT32_MAX) {
      loom_low_allocation_target_constraints_record_move_failure(
          context->target_constraints, op,
          storage_class->descriptor_reg_class_id, UINT32_MAX, 1,
          IREE_SV("parallel-move-location-range-overflow"));
      return iree_ok_status();
    }
  }

  const uint32_t candidate_count =
      uses_explicit_physical_registers
          ? iree_min((uint32_t)reg_class->allocatable_count,
                     capacity.is_bounded ? capacity.max_units : UINT32_MAX)
          : last_location + 1u;
  loom_low_allocation_move_plan_t* plan = group_context->plan;
  if (plan->storage_liveness_index.descriptor_set == NULL) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_storage_liveness_index_initialize(
        context->descriptor_set, context->assignment_map.assignments,
        context->assignment_map.assignment_count, context->unit_liveness,
        plan->scratch_arena, &plan->storage_liveness_index));
  }
  for (uint32_t candidate_ordinal = 0; candidate_ordinal < candidate_count;
       ++candidate_ordinal) {
    const uint32_t location =
        uses_explicit_physical_registers
            ? loom_low_descriptor_set_physical_register_candidate(
                  context->descriptor_set,
                  storage_class->descriptor_reg_class_id,
                  (uint16_t)candidate_ordinal)
            : candidate_ordinal;
    const loom_low_move_location_t temporary = {
        .location_kind = storage_class->location_kind,
        .descriptor_reg_class_id = storage_class->descriptor_reg_class_id,
        .location = location,
    };
    if (loom_low_allocation_target_constraints_reserved_range_conflicts(
            context->target_constraints, temporary.descriptor_reg_class_id,
            temporary.location_kind, temporary.location, 1) ||
        loom_low_allocation_storage_liveness_index_is_live_at_point(
            &plan->storage_liveness_index, &temporary,
            group_context->read_point) ||
        loom_low_move_sequence_location_set_contains(occupied_locations,
                                                     &temporary) ||
        loom_low_allocation_write_interference_temporary_conflicts(
            context->unit_liveness->write_interference,
            &context->assignment_map, group_context->write_point, &temporary)) {
      continue;
    }
    *out_temporary = temporary;
    *out_resolved = true;
    loom_low_allocation_target_constraints_record_location_extent(
        context->target_constraints, temporary.descriptor_reg_class_id,
        temporary.location_kind, temporary.location, 1);
    return iree_ok_status();
  }

  if (capacity.is_spillable && context->storage_spaces &&
      loom_low_storage_space_set_contains(
          context->storage_spaces,
          loom_low_allocation_storage_space_for_spill_slot(
              reg_class->spill_slot_space))) {
    if (plan->storage_indices_by_class == NULL) {
      const iree_host_size_t bytes =
          context->descriptor_set->reg_class_count * sizeof(uint32_t);
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          plan->scratch_arena, bytes, (void**)&plan->storage_indices_by_class));
      memset(plan->storage_indices_by_class, 0xff, bytes);
    }
    uint32_t* index =
        &plan->storage_indices_by_class[storage_class->descriptor_reg_class_id];
    if (*index == UINT32_MAX) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          plan->output_arena, plan->storage_count, plan->storage_count + 1,
          sizeof(*plan->storage), &plan->storage_capacity,
          (void**)&plan->storage));
      *index = (uint32_t)plan->storage_count++;
      const uint32_t byte_length =
          ((uint32_t)reg_class->alloc_unit_bits + 7) / 8;
      plan->storage[*index] = (loom_low_move_storage_t){
          .register_class = storage_class->descriptor_reg_class_id,
          .space = loom_low_allocation_storage_space_for_spill_slot(
              reg_class->spill_slot_space),
          .byte_length = byte_length,
          .byte_alignment = iree_math_round_up_to_pow2_u32(byte_length),
      };
    }
    *out_temporary = (loom_low_move_location_t){
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_MOVE_STORAGE,
        .descriptor_reg_class_id = storage_class->descriptor_reg_class_id,
        .location = *index,
    };
    *out_resolved = true;
    return iree_ok_status();
  }
  loom_low_allocation_target_constraints_record_move_failure(
      context->target_constraints, op, storage_class->descriptor_reg_class_id,
      capacity.is_bounded ? capacity.max_units : UINT32_MAX, 1,
      IREE_SV("parallel-move-no-scratch-unit"));

  return iree_ok_status();
}

iree_status_t loom_low_allocation_move_plan_initialize(
    const loom_low_allocation_move_plan_context_t* context,
    iree_host_size_t move_input_capacity, iree_host_size_t raw_group_capacity,
    iree_arena_allocator_t* output_arena, iree_arena_allocator_t* scratch_arena,
    loom_low_allocation_move_plan_t* out_plan) {
  *out_plan = (loom_low_allocation_move_plan_t){
      .context = *context,
      .output_arena = output_arena,
      .scratch_arena = scratch_arena,
      .scratch_move_index_capacity = move_input_capacity / 2,
  };
  const loom_low_schedule_table_t* schedule = context->schedule;
  if (schedule != NULL && schedule->node_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, schedule->node_count,
        sizeof(*out_plan->operation_indices_by_source_node),
        (void**)&out_plan->operation_indices_by_source_node));
    const loom_liveness_analysis_t* liveness = context->assignment_map.liveness;
    uint32_t scheduled_index = 0;
    for (uint32_t i = 0; i < liveness->operation_count; ++i) {
      const loom_liveness_operation_point_t* point =
          loom_liveness_operation_at(liveness, i);
      if (point->parent_operation_index != UINT32_MAX) {
        continue;
      }
      const uint32_t node_index =
          schedule->scheduled_node_indices[scheduled_index++];
      IREE_ASSERT_EQ(schedule->nodes[node_index].op, point->op);
      out_plan->operation_indices_by_source_node[node_index] = i;
    }
    IREE_ASSERT_EQ(scheduled_index, schedule->scheduled_node_count);
  }
  if (move_input_capacity == 0) {
    return iree_ok_status();
  }
  iree_host_size_t move_capacity = 0;
  if (!iree_host_size_checked_add(move_input_capacity, move_input_capacity / 2,
                                  &move_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parallel move output capacity exceeds host size");
  }
  out_plan->move_capacity = move_capacity;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(output_arena, move_capacity,
                                                 sizeof(*out_plan->moves),
                                                 (void**)&out_plan->moves));
  return loom_low_move_sequence_scratch_initialize(
      scratch_arena, raw_group_capacity, &out_plan->sequence_scratch);
}

const loom_liveness_operation_point_t*
loom_low_allocation_move_plan_next_operation(
    const loom_low_allocation_move_plan_t* plan, const loom_op_t* op,
    loom_low_allocation_move_cursor_t* cursor) {
  const loom_liveness_analysis_t* liveness =
      plan->context.assignment_map.liveness;
  if (plan->operation_indices_by_source_node != NULL &&
      op->parent_block->parent_region == liveness->region) {
    cursor->operation_index =
        plan->operation_indices_by_source_node[cursor->source_node_index++];
  }
  const loom_liveness_operation_point_t* point =
      loom_liveness_operation_at(liveness, cursor->operation_index++);
  IREE_ASSERT_EQ(point->op, op,
                 "move traversal must match the accepted liveness subtree");
  return point;
}

void loom_low_allocation_move_plan_append_assignment(
    loom_low_allocation_move_plan_t* plan,
    const loom_low_allocation_assignment_t* source, uint32_t source_unit_offset,
    const loom_low_allocation_assignment_t* destination,
    uint32_t destination_unit_offset, uint32_t unit_count,
    iree_host_size_t* inout_raw_move_count) {
  const loom_low_allocation_unit_liveness_t* unit_liveness =
      plan->context.unit_liveness;
  for (uint32_t unit = 0; unit < unit_count; ++unit) {
    const uint32_t destination_unit = destination_unit_offset + unit;
    const uint32_t start_point =
        loom_low_allocation_live_range_assignment_unit_start_point(
            unit_liveness->start_points, unit_liveness->point_count,
            destination, destination_unit);
    const uint32_t end_point =
        loom_low_allocation_live_range_assignment_unit_end_point(
            unit_liveness->end_points, unit_liveness->point_count, destination,
            destination_unit);
    if (start_point >= end_point) {
      continue;
    }
    plan->sequence_scratch.moves[(*inout_raw_move_count)++] = (loom_low_move_t){
        .source = loom_low_allocation_assignment_unit_location(
            plan->context.descriptor_set, source, source_unit_offset + unit),
        .destination = loom_low_allocation_assignment_unit_location(
            plan->context.descriptor_set, destination, destination_unit),
    };
  }
}

iree_status_t loom_low_allocation_move_plan_append_group(
    loom_low_allocation_move_plan_t* plan, const loom_op_t* owner_op,
    uint32_t read_point, uint32_t write_point, iree_host_size_t raw_move_count,
    loom_low_move_group_t* out_group) {
  *out_group = (loom_low_move_group_t){
      .moves.start = plan->move_count,
      .scratch_move_index_start = plan->scratch_move_index_count,
  };
  if (raw_move_count == 0) {
    return iree_ok_status();
  }
  loom_low_allocation_move_plan_group_context_t group_context = {
      .plan = plan,
      .owner_op = owner_op,
      .read_point = read_point,
      .write_point = write_point,
      .move_start = plan->move_count,
  };
  const loom_low_move_sequence_options_t options = {
      .descriptor_set = plan->context.descriptor_set,
      .resolve_temporary =
          {
              .fn = loom_low_allocation_move_plan_resolve_temporary,
              .user_data = &group_context,
          },
      .record_scratch =
          {
              .fn = loom_low_allocation_move_plan_record_scratch,
              .user_data = &group_context,
          },
  };
  iree_host_size_t move_count = 0;
  bool complete = false;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_resolve(
      &plan->sequence_scratch, raw_move_count, &options,
      plan->move_capacity - plan->move_count, &plan->moves[plan->move_count],
      &move_count, &complete));
  if (!complete) {
    return iree_ok_status();
  }
  plan->move_count += move_count;
  out_group->moves.count = move_count;
  out_group->scratch_move_index_count =
      plan->scratch_move_index_count - out_group->scratch_move_index_start;
  return iree_ok_status();
}
