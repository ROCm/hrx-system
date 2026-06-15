// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/packet_move.h"

#include <string.h>

#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/move_topology.h"
#include "loom/codegen/low/allocation/unit_location.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

typedef struct loom_low_allocation_packet_unit_move_t {
  // Unit overwritten by the packet-local move.
  loom_low_allocation_unit_location_t destination;
  // Unit read by the packet-local move.
  loom_low_allocation_unit_location_t source;
} loom_low_allocation_packet_unit_move_t;

static iree_status_t loom_low_allocation_packet_move_append_unit_move(
    const loom_low_allocation_assignment_t* source_assignment,
    uint32_t source_unit_index,
    const loom_low_allocation_assignment_t* destination_assignment,
    uint32_t destination_unit_index,
    loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t move_capacity, iree_host_size_t* inout_move_count) {
  if (*inout_move_count >= move_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "packet-local move count exceeds reserved capacity");
  }
  loom_low_allocation_packet_unit_move_t* move =
      moves ? &moves[*inout_move_count] : NULL;
  if (move) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_unit_location(
        source_assignment, source_unit_index, &move->source));
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_unit_location(
        destination_assignment, destination_unit_index, &move->destination));
  }
  ++*inout_move_count;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_packet_moves_for_copy(
    const loom_low_allocation_packet_move_context_t* context,
    const loom_op_t* op, loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t move_capacity, iree_host_size_t* out_move_count) {
  *out_move_count = 0;
  const loom_low_allocation_assignment_t* source_assignment = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_assignment_map_require_assignment_for_value(
          &context->assignment_map, loom_low_copy_source(op), NULL,
          &source_assignment));
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_assignment_map_require_assignment_for_value(
          &context->assignment_map, loom_low_copy_result(op), NULL,
          &result_assignment));
  if (source_assignment->location_count != result_assignment->location_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low.copy packet-local move requires matching location counts");
  }
  for (uint32_t i = 0; i < source_assignment->location_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_packet_move_append_unit_move(
        source_assignment, i, result_assignment, i, moves, move_capacity,
        out_move_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_packet_moves_for_slice(
    const loom_low_allocation_packet_move_context_t* context,
    const loom_op_t* op, loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t move_capacity, iree_host_size_t* out_move_count) {
  *out_move_count = 0;
  const int64_t offset = loom_low_slice_offset(op);
  if (offset < 0 || offset > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low.slice offset is outside uint32_t range");
  }
  const loom_low_allocation_assignment_t* source_assignment = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_assignment_map_require_assignment_for_value(
          &context->assignment_map, loom_low_slice_source(op), NULL,
          &source_assignment));
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_assignment_map_require_assignment_for_value(
          &context->assignment_map, loom_low_slice_result(op), NULL,
          &result_assignment));
  const uint32_t source_offset = (uint32_t)offset;
  if (source_offset > source_assignment->location_count ||
      result_assignment->location_count >
          source_assignment->location_count - source_offset) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low.slice packet-local move range exceeds source assignment");
  }
  for (uint32_t i = 0; i < result_assignment->location_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_packet_move_append_unit_move(
        source_assignment, source_offset + i, result_assignment, i, moves,
        move_capacity, out_move_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_packet_moves_for_concat(
    const loom_low_allocation_packet_move_context_t* context,
    const loom_op_t* op, loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t move_capacity, iree_host_size_t* out_move_count) {
  *out_move_count = 0;
  if (!loom_low_allocation_move_topology_concat_requires_packet_materialization_for_module(
          context->module, op)) {
    return iree_ok_status();
  }
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_assignment_map_require_assignment_for_value(
          &context->assignment_map, loom_low_concat_result(op), NULL,
          &result_assignment));
  uint32_t result_offset = 0;
  loom_value_slice_t sources = loom_low_concat_sources(op);
  for (uint16_t i = 0; i < sources.count; ++i) {
    const loom_low_allocation_assignment_t* source_assignment = NULL;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_assignment_map_require_assignment_for_value(
            &context->assignment_map, sources.values[i], NULL,
            &source_assignment));
    if (result_offset > result_assignment->location_count ||
        source_assignment->location_count >
            result_assignment->location_count - result_offset) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low.concat packet-local move source ranges exceed result "
          "assignment");
    }
    for (uint32_t source_unit = 0;
         source_unit < source_assignment->location_count; ++source_unit) {
      IREE_RETURN_IF_ERROR(loom_low_allocation_packet_move_append_unit_move(
          source_assignment, source_unit, result_assignment,
          result_offset + source_unit, moves, move_capacity, out_move_count));
    }
    result_offset += source_assignment->location_count;
  }
  if (result_offset != result_assignment->location_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low.concat packet-local move sources do not fill result assignment");
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_packet_moves_for_op(
    const loom_low_allocation_packet_move_context_t* context,
    const loom_op_t* op, loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t move_capacity, iree_host_size_t* out_move_count) {
  if (loom_low_copy_isa(op)) {
    return loom_low_allocation_packet_moves_for_copy(
        context, op, moves, move_capacity, out_move_count);
  }
  if (loom_low_slice_isa(op)) {
    return loom_low_allocation_packet_moves_for_slice(
        context, op, moves, move_capacity, out_move_count);
  }
  if (loom_low_concat_isa(op)) {
    return loom_low_allocation_packet_moves_for_concat(
        context, op, moves, move_capacity, out_move_count);
  }
  *out_move_count = 0;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_packet_move_count_region_capacity(
    const loom_low_allocation_packet_move_context_t* context,
    const loom_region_t* region, iree_host_size_t* out_group_capacity,
    iree_host_size_t* out_temporary_capacity) {
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_allocation_move_topology_op_has_packet_moves(op)) {
        iree_host_size_t move_count = 0;
        IREE_RETURN_IF_ERROR(loom_low_allocation_packet_moves_for_op(
            context, op, /*moves=*/NULL, /*move_capacity=*/IREE_HOST_SIZE_MAX,
            &move_count));
        if (move_count != 0) {
          if (*out_group_capacity == IREE_HOST_SIZE_MAX ||
              move_count > IREE_HOST_SIZE_MAX - *out_temporary_capacity) {
            return iree_make_status(
                IREE_STATUS_OUT_OF_RANGE,
                "packet-local move temporary capacity exceeds host size");
          }
          ++*out_group_capacity;
          *out_temporary_capacity += move_count;
        }
      }
      if (!loom_liveness_analysis_includes_region_tree(
              context->assignment_map.liveness)) {
        continue;
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (regions[i] == NULL) {
          continue;
        }
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_packet_move_count_region_capacity(
                context, regions[i], out_group_capacity,
                out_temporary_capacity));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_packet_move_count_capacity(
    const loom_low_allocation_packet_move_context_t* context,
    iree_host_size_t* out_group_capacity,
    iree_host_size_t* out_temporary_capacity) {
  *out_group_capacity = 0;
  *out_temporary_capacity = 0;
  return loom_low_allocation_packet_move_count_region_capacity(
      context, context->body, out_group_capacity, out_temporary_capacity);
}

static bool loom_low_allocation_packet_move_uses_location(
    const loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t move_count,
    const loom_low_allocation_unit_location_t* location) {
  for (iree_host_size_t i = 0; i < move_count; ++i) {
    if (!loom_low_allocation_unit_locations_form_register_move(
            &moves[i].source, &moves[i].destination)) {
      continue;
    }
    if (loom_low_allocation_unit_locations_equal(location, &moves[i].source) ||
        loom_low_allocation_unit_locations_equal(location,
                                                 &moves[i].destination)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_allocation_packet_move_find_destination(
    const loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t move_count,
    const loom_low_allocation_unit_location_t* destination, bool* out_found,
    loom_low_allocation_unit_location_t* out_source) {
  *out_found = false;
  *out_source = (loom_low_allocation_unit_location_t){0};
  for (iree_host_size_t i = 0; i < move_count; ++i) {
    const loom_low_allocation_packet_unit_move_t* move = &moves[i];
    if (!loom_low_allocation_unit_locations_form_register_move(
            &move->source, &move->destination)) {
      continue;
    }
    if (!loom_low_allocation_unit_storage_classes_equal(destination,
                                                        &move->destination)) {
      continue;
    }
    if (!loom_low_allocation_unit_locations_equal(destination,
                                                  &move->destination)) {
      continue;
    }
    if (*out_found) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "packet-local move set has duplicate destinations");
    }
    *out_found = true;
    *out_source = move->source;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_packet_move_starts_cycle(
    const loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t move_count,
    const loom_low_allocation_unit_location_t* destination,
    const loom_low_allocation_unit_location_t* source, bool* out_has_cycle) {
  *out_has_cycle = false;
  loom_low_allocation_unit_location_t next_destination = *source;
  for (iree_host_size_t step = 0; step < move_count; ++step) {
    if (loom_low_allocation_unit_locations_equal(&next_destination,
                                                 destination)) {
      *out_has_cycle = true;
      return iree_ok_status();
    }
    bool found = false;
    loom_low_allocation_unit_location_t next_source = {0};
    IREE_RETURN_IF_ERROR(loom_low_allocation_packet_move_find_destination(
        moves, move_count, &next_destination, &found, &next_source));
    if (!found) {
      return iree_ok_status();
    }
    next_destination = next_source;
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "packet-local move cycle is malformed");
}

static iree_status_t loom_low_allocation_packet_move_class_has_cycle(
    const loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t move_count,
    const loom_low_allocation_unit_location_t* storage_class,
    bool* out_has_cycle) {
  *out_has_cycle = false;
  for (iree_host_size_t i = 0; i < move_count; ++i) {
    const loom_low_allocation_packet_unit_move_t* move = &moves[i];
    if (!loom_low_allocation_unit_storage_classes_equal(storage_class,
                                                        &move->destination) ||
        !loom_low_allocation_unit_locations_form_register_move(
            &move->source, &move->destination)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_allocation_packet_move_starts_cycle(
        moves, move_count, &move->destination, &move->source, out_has_cycle));
    if (*out_has_cycle) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static bool loom_low_allocation_packet_move_class_seen_before(
    const loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t stop_move_index,
    const loom_low_allocation_unit_location_t* storage_class) {
  for (iree_host_size_t i = 0; i < stop_move_index; ++i) {
    if (!loom_low_allocation_unit_locations_form_register_move(
            &moves[i].source, &moves[i].destination)) {
      continue;
    }
    if (loom_low_allocation_unit_storage_classes_equal(storage_class,
                                                       &moves[i].destination)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_allocation_packet_move_find_temporary(
    const loom_low_allocation_packet_move_context_t* context,
    const loom_op_t* op, uint32_t program_point,
    const loom_low_allocation_unit_location_t* storage_class,
    const loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t move_count,
    loom_low_allocation_unit_location_t* out_temporary) {
  *out_temporary = (loom_low_allocation_unit_location_t){0};
  if (!loom_low_allocation_location_kind_is_register_like(
          storage_class->location_kind)) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_emit_failure(
        context->target_constraints, op, storage_class->value_class, 0, 1,
        IREE_SV("packet-move-non-register-storage")));
    return iree_ok_status();
  }

  loom_low_allocation_class_capacity_t capacity = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_class_capacity(
      context->target_constraints, storage_class->value_class, &capacity));
  if (capacity.location_kind != storage_class->location_kind) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_emit_failure(
        context->target_constraints, op, storage_class->value_class,
        capacity.is_bounded ? capacity.max_units : UINT32_MAX, 1,
        IREE_SV("packet-move-storage-kind-mismatch")));
    return iree_ok_status();
  }

  uint32_t last_location = 0;
  if (capacity.is_bounded) {
    if (capacity.max_units == 0) {
      IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_emit_failure(
          context->target_constraints, op, storage_class->value_class,
          capacity.max_units, 1, IREE_SV("packet-move-empty-budget")));
      return iree_ok_status();
    }
    last_location = capacity.max_units - 1u;
  } else {
    last_location =
        loom_low_allocation_target_constraints_assigned_location_search_limit(
            context->target_constraints, storage_class->descriptor_reg_class_id,
            storage_class->location_kind);
    if (last_location == UINT32_MAX) {
      IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_emit_failure(
          context->target_constraints, op, storage_class->value_class,
          UINT32_MAX, 1, IREE_SV("packet-move-location-range-overflow")));
      return iree_ok_status();
    }
  }

  for (uint32_t location = 0; location <= last_location; ++location) {
    loom_low_allocation_unit_location_t temporary = {
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
            context->assignment_map.assignment_count,
            context->unit_liveness->end_points,
            context->unit_liveness->end_point_count, &temporary,
            program_point) ||
        loom_low_allocation_packet_move_uses_location(moves, move_count,
                                                      &temporary)) {
      if (location == UINT32_MAX) {
        break;
      }
      continue;
    }
    *out_temporary = temporary;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_emit_failure(
      context->target_constraints, op, storage_class->value_class,
      capacity.is_bounded ? capacity.max_units : UINT32_MAX, 1,
      IREE_SV("packet-move-no-scratch-unit")));
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_packet_move_op_program_point(
    const loom_low_allocation_packet_move_context_t* context,
    const loom_op_t* op, uint32_t* out_program_point) {
  return loom_low_allocation_live_range_ordered_op_program_point(
      context->assignment_map.liveness, context->body, context->liveness_order,
      op, out_program_point);
}

static iree_status_t
loom_low_allocation_packet_move_plan_record_temporaries_for_op(
    const loom_low_allocation_packet_move_context_t* context,
    iree_arena_allocator_t* arena, const loom_op_t* op, uint32_t source_ordinal,
    loom_low_allocation_packet_move_plan_t* plan) {
  iree_host_size_t move_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_packet_moves_for_op(
      context, op, /*moves=*/NULL, IREE_HOST_SIZE_MAX, &move_capacity));
  if (move_capacity == 0) {
    return iree_ok_status();
  }
  loom_low_allocation_packet_unit_move_t* moves = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, move_capacity, sizeof(*moves), (void**)&moves));
  iree_host_size_t move_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_packet_moves_for_op(
      context, op, moves, move_capacity, &move_count));

  uint32_t program_point = UINT32_MAX;
  IREE_RETURN_IF_ERROR(loom_low_allocation_packet_move_op_program_point(
      context, op, &program_point));
  const iree_host_size_t temporary_start = plan->temporary_count;

  for (iree_host_size_t i = 0; i < move_count; ++i) {
    const loom_low_allocation_packet_unit_move_t* move = &moves[i];
    if (!loom_low_allocation_unit_locations_form_register_move(
            &move->source, &move->destination) ||
        loom_low_allocation_packet_move_class_seen_before(moves, i,
                                                          &move->destination)) {
      continue;
    }
    bool has_cycle = false;
    IREE_RETURN_IF_ERROR(loom_low_allocation_packet_move_class_has_cycle(
        moves, move_count, &move->destination, &has_cycle));
    if (!has_cycle) {
      continue;
    }
    if (plan->temporary_count >= UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "packet-local move temporary table exceeds u32 range");
    }
    loom_low_allocation_unit_location_t temporary = {0};
    IREE_RETURN_IF_ERROR(loom_low_allocation_packet_move_find_temporary(
        context, op, program_point, &move->destination, moves, move_count,
        &temporary));
    if (context->target_constraints->error_count != 0) {
      return iree_ok_status();
    }
    plan->temporaries[plan->temporary_count++] =
        (loom_low_allocation_packet_move_temporary_t){
            .value_class = temporary.value_class,
            .descriptor_reg_class_id = temporary.descriptor_reg_class_id,
            .location_kind = temporary.location_kind,
            .location = temporary.location,
        };
  }

  if (plan->temporary_count == temporary_start) {
    return iree_ok_status();
  }
  if (plan->group_count >= UINT32_MAX || temporary_start > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "packet-local move temporary group exceeds u32 range");
  }
  plan->groups[plan->group_count++] =
      (loom_low_allocation_packet_move_temporary_group_t){
          .op = op,
          .source_ordinal = source_ordinal,
          .program_point = program_point,
          .temporary_start = (uint32_t)temporary_start,
          .temporary_count =
              (uint32_t)(plan->temporary_count - temporary_start),
      };
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_packet_move_plan_record_temporaries_for_region(
    const loom_low_allocation_packet_move_context_t* context,
    iree_arena_allocator_t* arena, const loom_region_t* region,
    uint32_t* inout_source_ordinal,
    loom_low_allocation_packet_move_plan_t* plan) {
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_allocation_move_topology_op_has_packet_moves(op)) {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_packet_move_plan_record_temporaries_for_op(
                context, arena, op, *inout_source_ordinal, plan));
        if (context->target_constraints->error_count != 0) {
          return iree_ok_status();
        }
      }
      if (*inout_source_ordinal == UINT32_MAX) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "packet-local move source ordinal exceeds uint32_t");
      }
      ++*inout_source_ordinal;
      if (!loom_liveness_analysis_includes_region_tree(
              context->assignment_map.liveness)) {
        continue;
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (regions[i] == NULL) {
          continue;
        }
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_packet_move_plan_record_temporaries_for_region(
                context, arena, regions[i], inout_source_ordinal, plan));
        if (context->target_constraints->error_count != 0) {
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
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_low_allocation_packet_move_plan_t){0};

  iree_host_size_t group_capacity = 0;
  iree_host_size_t temporary_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_packet_move_count_capacity(
      context, &group_capacity, &temporary_capacity));
  if (temporary_capacity == 0) {
    return iree_ok_status();
  }

  loom_low_allocation_packet_move_plan_t plan = {0};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, group_capacity, sizeof(*plan.groups), (void**)&plan.groups));
  memset(plan.groups, 0, group_capacity * sizeof(*plan.groups));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, temporary_capacity,
                                                 sizeof(*plan.temporaries),
                                                 (void**)&plan.temporaries));
  memset(plan.temporaries, 0, temporary_capacity * sizeof(*plan.temporaries));

  uint32_t source_ordinal = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_packet_move_plan_record_temporaries_for_region(
          context, arena, context->body, &source_ordinal, &plan));
  if (context->target_constraints->error_count != 0) {
    *out_plan = plan;
    return iree_ok_status();
  }
  *out_plan = plan;
  return iree_ok_status();
}
