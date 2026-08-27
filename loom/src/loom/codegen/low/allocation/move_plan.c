// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/move_plan.h"

#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/allocation/unit_location.h"

typedef struct loom_low_allocation_move_plan_group_context_t {
  // Plan owning shared allocation facts and persistent output rows.
  loom_low_allocation_move_plan_t* plan;
  // Operation that owns the current move group.
  const loom_op_t* op;
  // Global move-row index corresponding to local output row zero.
  iree_host_size_t move_start;
} loom_low_allocation_move_plan_group_context_t;

static uint32_t loom_low_allocation_move_group_program_point(
    const loom_low_allocation_move_plan_group_context_t* group_context) {
  const loom_liveness_analysis_t* liveness =
      group_context->plan->context.assignment_map.liveness;
  for (iree_host_size_t i = 0; i < liveness->operation_count; ++i) {
    const loom_liveness_operation_point_t* point =
        &liveness->operation_points[i];
    if (point->op == group_context->op) {
      return point->start_point;
    }
  }
  IREE_ASSERT_UNREACHABLE(
      "allocation move operation must have a liveness program point");
  return 0;
}

static bool loom_low_allocation_move_group_uses_location(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_move_t* moves, iree_host_size_t move_count,
    const loom_low_move_location_t* location) {
  for (iree_host_size_t i = 0; i < move_count; ++i) {
    const loom_low_move_location_t* source = &moves[i].source;
    const loom_low_move_location_t* destination = &moves[i].destination;
    if (source->location_kind == location->location_kind &&
        source->location == location->location &&
        loom_low_allocation_storage_reg_classes_share(
            descriptor_set, source->descriptor_reg_class_id,
            location->descriptor_reg_class_id)) {
      return true;
    }
    if (destination->location_kind == location->location_kind &&
        destination->location == location->location &&
        loom_low_allocation_storage_reg_classes_share(
            descriptor_set, destination->descriptor_reg_class_id,
            location->descriptor_reg_class_id)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_allocation_move_plan_record_scratch(
    void* user_data, iree_host_size_t move_index) {
  loom_low_allocation_move_plan_group_context_t* group_context =
      (loom_low_allocation_move_plan_group_context_t*)user_data;
  loom_low_allocation_move_plan_t* plan = group_context->plan;
  if (plan->scratch_move_index_count == plan->scratch_move_index_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->sequence_scratch.arena, plan->scratch_move_index_count,
        plan->scratch_move_index_count + 1, sizeof(*plan->scratch_move_indices),
        &plan->scratch_move_index_capacity,
        (void**)&plan->scratch_move_indices));
  }
  plan->scratch_move_indices[plan->scratch_move_index_count++] =
      group_context->move_start + move_index;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_move_plan_resolve_temporary(
    void* user_data, const loom_low_move_location_t* storage_class,
    const loom_low_move_t* moves, iree_host_size_t move_count,
    loom_low_move_location_t* out_temporary, bool* out_resolved) {
  const loom_low_allocation_move_plan_group_context_t* group_context =
      (const loom_low_allocation_move_plan_group_context_t*)user_data;
  const loom_low_allocation_move_plan_context_t* context =
      &group_context->plan->context;
  *out_temporary = (loom_low_move_location_t){0};
  *out_resolved = false;
  if (!loom_low_allocation_location_kind_is_register_like(
          storage_class->location_kind)) {
    return loom_low_allocation_target_constraints_emit_failure(
        context->target_constraints, group_context->op,
        storage_class->value_class, 0, 1,
        IREE_SV("parallel-move-non-register-storage"));
  }

  loom_low_allocation_class_capacity_t capacity = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_class_capacity(
      context->target_constraints, storage_class->value_class, &capacity));
  if (capacity.location_kind != storage_class->location_kind) {
    return loom_low_allocation_target_constraints_emit_failure(
        context->target_constraints, group_context->op,
        storage_class->value_class,
        capacity.is_bounded ? capacity.max_units : UINT32_MAX, 1,
        IREE_SV("parallel-move-storage-kind-mismatch"));
  }

  uint32_t last_location = 0;
  if (capacity.is_bounded) {
    if (capacity.max_units == 0) {
      return loom_low_allocation_target_constraints_emit_failure(
          context->target_constraints, group_context->op,
          storage_class->value_class, capacity.max_units, 1,
          IREE_SV("parallel-move-empty-budget"));
    }
    last_location = capacity.max_units - 1u;
  } else {
    last_location =
        loom_low_allocation_target_constraints_assigned_location_search_limit(
            context->target_constraints, storage_class->descriptor_reg_class_id,
            storage_class->location_kind);
    if (last_location == UINT32_MAX) {
      return loom_low_allocation_target_constraints_emit_failure(
          context->target_constraints, group_context->op,
          storage_class->value_class, UINT32_MAX, 1,
          IREE_SV("parallel-move-location-range-overflow"));
    }
  }

  const uint32_t program_point =
      loom_low_allocation_move_group_program_point(group_context);
  for (uint32_t location = 0; location <= last_location; ++location) {
    const loom_low_move_location_t temporary = {
        .location_kind = storage_class->location_kind,
        .value_class = storage_class->value_class,
        .descriptor_reg_class_id = storage_class->descriptor_reg_class_id,
        .location = location,
    };
    if (loom_low_allocation_target_constraints_reserved_range_conflicts(
            context->target_constraints, temporary.descriptor_reg_class_id,
            temporary.location_kind, temporary.location, 1) ||
        loom_low_allocation_unit_location_is_live_at_point(
            context->descriptor_set, context->assignment_map.assignments,
            context->assignment_map.assignment_count, context->unit_liveness,
            &temporary, program_point) ||
        loom_low_allocation_move_group_uses_location(
            context->descriptor_set, moves, move_count, &temporary)) {
      if (location == UINT32_MAX) {
        break;
      }
      continue;
    }
    *out_temporary = temporary;
    *out_resolved = true;
    loom_low_allocation_target_constraints_record_location_extent(
        context->target_constraints, temporary.descriptor_reg_class_id,
        temporary.location_kind, temporary.location, 1);
    return iree_ok_status();
  }

  return loom_low_allocation_target_constraints_emit_failure(
      context->target_constraints, group_context->op,
      storage_class->value_class,
      capacity.is_bounded ? capacity.max_units : UINT32_MAX, 1,
      IREE_SV("parallel-move-no-scratch-unit"));
}

iree_status_t loom_low_allocation_move_plan_initialize(
    const loom_low_allocation_move_plan_context_t* context,
    iree_arena_allocator_t* arena, iree_host_size_t move_input_capacity,
    iree_host_size_t raw_group_capacity,
    loom_low_allocation_move_plan_t* out_plan) {
  *out_plan = (loom_low_allocation_move_plan_t){
      .context = *context,
  };
  if (move_input_capacity == 0) {
    return iree_ok_status();
  }
  const iree_host_size_t move_capacity =
      move_input_capacity + move_input_capacity / 2;
  out_plan->move_capacity = move_capacity;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, move_capacity,
                                                 sizeof(*out_plan->moves),
                                                 (void**)&out_plan->moves));
  return loom_low_move_sequence_scratch_initialize(arena, raw_group_capacity,
                                                   &out_plan->sequence_scratch);
}

loom_low_move_t* loom_low_allocation_move_plan_raw_moves(
    loom_low_allocation_move_plan_t* plan) {
  return plan->sequence_scratch.moves;
}

iree_status_t loom_low_allocation_move_plan_append_group(
    loom_low_allocation_move_plan_t* plan, const loom_op_t* op,
    iree_host_size_t raw_move_count, loom_low_move_group_t* out_group) {
  *out_group = (loom_low_move_group_t){
      .moves.start = plan->move_count,
      .scratch_move_index_start = plan->scratch_move_index_count,
  };
  if (raw_move_count == 0) {
    return iree_ok_status();
  }
  loom_low_allocation_move_plan_group_context_t group_context = {
      .plan = plan,
      .op = op,
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
