// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/testing/allocation_checker.h"

#include <string.h>

#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/placement.h"
#include "loom/codegen/low/storage_lease.h"
#include "loom/target/registers.h"

typedef struct loom_low_allocation_checker_t {
  // Emission frame being checked.
  const loom_low_emission_frame_t* frame;
  // Schedule sidecar being checked.
  const loom_low_schedule_table_t* schedule;
  // Allocation sidecar being checked.
  const loom_low_allocation_table_t* allocation;
  // Descriptor set defining physical storage aliases.
  const loom_low_descriptor_set_t* descriptor_set;
  // Scratch arena used only by this checker run.
  iree_arena_allocator_t* arena;
  // Assignment ordinal indexed by assignment index.
  loom_value_ordinal_t* assignment_ordinals;
  // Program point indexed by schedule node index.
  uint32_t* node_program_points;
  // Mutable aggregate result.
  loom_low_allocation_check_result_t* result;
} loom_low_allocation_checker_t;

static void loom_low_allocation_checker_record(
    loom_low_allocation_checker_t* checker,
    loom_low_allocation_check_violation_kind_t kind, uint32_t primary_index,
    uint32_t secondary_index, loom_value_id_t value_id,
    loom_value_id_t related_value_id, uint32_t program_point) {
  loom_low_allocation_check_result_t* result = checker->result;
  if (result->violation_count == 0) {
    result->first_violation = (loom_low_allocation_check_violation_t){
        .kind = kind,
        .primary_index = primary_index,
        .secondary_index = secondary_index,
        .value_id = value_id,
        .related_value_id = related_value_id,
        .program_point = program_point,
    };
  }
  if (result->violation_count != UINT32_MAX) {
    ++result->violation_count;
  }
}

static bool loom_low_allocation_checker_assignment_start_shape_is_valid(
    const loom_low_allocation_assignment_t* assignment,
    const loom_liveness_interval_t* interval) {
  const loom_low_allocation_assignment_flags_t unknown_flags =
      assignment->flags &
      (loom_low_allocation_assignment_flags_t)~LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS;
  if (unknown_flags != 0) {
    return false;
  }
  if (!iree_any_bit_set(
          assignment->flags,
          LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS)) {
    return assignment->start_point == interval->start_point;
  }
  return assignment->unit_count != 0 &&
         assignment->start_point <= interval->start_point;
}

iree_string_view_t loom_low_allocation_check_violation_kind_name(
    loom_low_allocation_check_violation_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_ALLOCATION_CHECK_VIOLATION_NONE:
      return IREE_SV("none");
    case LOOM_LOW_ALLOCATION_CHECK_VIOLATION_FRAME_IDENTITY:
      return IREE_SV("frame-identity");
    case LOOM_LOW_ALLOCATION_CHECK_VIOLATION_SCHEDULE_STRUCTURE:
      return IREE_SV("schedule-structure");
    case LOOM_LOW_ALLOCATION_CHECK_VIOLATION_VALUE_DOMAIN:
      return IREE_SV("value-domain");
    case LOOM_LOW_ALLOCATION_CHECK_VIOLATION_ASSIGNMENT_INDEX:
      return IREE_SV("assignment-index");
    case LOOM_LOW_ALLOCATION_CHECK_VIOLATION_ASSIGNMENT_SHAPE:
      return IREE_SV("assignment-shape");
    case LOOM_LOW_ALLOCATION_CHECK_VIOLATION_FIXED_LOCATION:
      return IREE_SV("fixed-location");
    case LOOM_LOW_ALLOCATION_CHECK_VIOLATION_RESERVED_LOCATION:
      return IREE_SV("reserved-location");
    case LOOM_LOW_ALLOCATION_CHECK_VIOLATION_PLACEMENT_RELATION:
      return IREE_SV("placement-relation");
    case LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_CONFLICT:
      return IREE_SV("storage-conflict");
    case LOOM_LOW_ALLOCATION_CHECK_VIOLATION_EARLY_CLOBBER:
      return IREE_SV("early-clobber");
    case LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_LEASE:
      return IREE_SV("storage-lease");
    default:
      return IREE_SV("unknown");
  }
}

static bool loom_low_allocation_checker_frame_identity(
    loom_low_allocation_checker_t* checker) {
  const loom_low_emission_frame_t* frame = checker->frame;
  const loom_low_schedule_table_t* schedule = checker->schedule;
  const loom_low_allocation_table_t* allocation = checker->allocation;
  const bool matches =
      frame->module == schedule->module &&
      frame->module == allocation->module &&
      frame->function_op == schedule->function_op &&
      frame->function_op == allocation->function_op &&
      frame->target.descriptor_set == schedule->target.descriptor_set &&
      frame->target.descriptor_set == allocation->target.descriptor_set;
  if (!matches) {
    loom_low_allocation_checker_record(
        checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_FRAME_IDENTITY, UINT32_MAX,
        UINT32_MAX, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, UINT32_MAX);
  }
  return matches;
}

static iree_status_t loom_low_allocation_checker_schedule(
    loom_low_allocation_checker_t* checker) {
  const loom_low_schedule_table_t* schedule = checker->schedule;
  const loom_liveness_analysis_t* liveness = &checker->allocation->liveness;
  if (schedule->block_count != liveness->block_count ||
      schedule->scheduled_node_count != schedule->node_count) {
    loom_low_allocation_checker_record(
        checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_SCHEDULE_STRUCTURE,
        UINT32_MAX, UINT32_MAX, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
        UINT32_MAX);
  }
  if (schedule->node_count == 0) {
    return iree_ok_status();
  }

  uint8_t* seen = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      checker->arena, schedule->node_count, sizeof(*seen), (void**)&seen));
  memset(seen, 0, schedule->node_count * sizeof(*seen));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(checker->arena, schedule->node_count,
                                sizeof(*checker->node_program_points),
                                (void**)&checker->node_program_points));
  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    checker->node_program_points[i] = UINT32_MAX;
  }

  const iree_host_size_t block_count =
      iree_min(schedule->block_count, liveness->block_count);
  for (iree_host_size_t block_index = 0; block_index < block_count;
       ++block_index) {
    const loom_low_schedule_block_t* schedule_block =
        &schedule->blocks[block_index];
    const loom_liveness_block_info_t* liveness_block =
        &liveness->blocks[block_index];
    if (schedule_block->block != liveness_block->block ||
        schedule_block->scheduled_node_count != schedule_block->node_count ||
        schedule_block->scheduled_node_start > schedule->scheduled_node_count ||
        schedule_block->scheduled_node_count >
            schedule->scheduled_node_count -
                schedule_block->scheduled_node_start) {
      loom_low_allocation_checker_record(
          checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_SCHEDULE_STRUCTURE,
          (uint32_t)block_index, UINT32_MAX, LOOM_VALUE_ID_INVALID,
          LOOM_VALUE_ID_INVALID, UINT32_MAX);
      continue;
    }

    uint32_t operation_index = liveness_block->operation_start;
    const uint32_t operation_end =
        operation_index + liveness_block->operation_count;
    for (uint32_t ordinal = 0; ordinal < schedule_block->scheduled_node_count;
         ++ordinal) {
      while (
          operation_index < operation_end &&
          liveness->operation_points[operation_index].parent_operation_index !=
              UINT32_MAX) {
        ++operation_index;
      }
      const loom_liveness_operation_point_t* operation_point =
          operation_index < operation_end
              ? &liveness->operation_points[operation_index++]
              : NULL;
      const uint32_t program_point =
          operation_point != NULL ? operation_point->start_point : UINT32_MAX;
      const iree_host_size_t scheduled_index =
          schedule_block->scheduled_node_start + ordinal;
      const uint32_t node_index =
          schedule->scheduled_node_indices[scheduled_index];
      if (node_index >= schedule->node_count) {
        loom_low_allocation_checker_record(
            checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_SCHEDULE_STRUCTURE,
            (uint32_t)scheduled_index, node_index, LOOM_VALUE_ID_INVALID,
            LOOM_VALUE_ID_INVALID, program_point);
        continue;
      }
      const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
      if (seen[node_index] || node->block_index != block_index ||
          node->scheduled_ordinal != ordinal ||
          node->block != schedule_block->block || operation_point == NULL ||
          operation_point->op != node->op) {
        loom_low_allocation_checker_record(
            checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_SCHEDULE_STRUCTURE,
            node_index, (uint32_t)scheduled_index, LOOM_VALUE_ID_INVALID,
            LOOM_VALUE_ID_INVALID, program_point);
      }
      seen[node_index] = 1;
      checker->node_program_points[node_index] = program_point;
    }
    while (operation_index < operation_end &&
           liveness->operation_points[operation_index].parent_operation_index !=
               UINT32_MAX) {
      ++operation_index;
    }
    if (operation_index != operation_end) {
      loom_low_allocation_checker_record(
          checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_SCHEDULE_STRUCTURE,
          (uint32_t)block_index, UINT32_MAX, LOOM_VALUE_ID_INVALID,
          LOOM_VALUE_ID_INVALID, liveness_block->end_point);
    }
  }
  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    if (!seen[i]) {
      loom_low_allocation_checker_record(
          checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_SCHEDULE_STRUCTURE,
          (uint32_t)i, UINT32_MAX, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
          UINT32_MAX);
    }
  }
  return iree_ok_status();
}

static bool loom_low_allocation_checker_interval_is_allocatable(
    const loom_liveness_interval_t* interval) {
  return interval->value_class.type_kind == LOOM_TYPE_REGISTER &&
         interval->unit_count != 0;
}

static uint32_t loom_low_allocation_checker_storage_end_point(
    const loom_liveness_interval_t* interval) {
  if (interval->end_point > interval->start_point) {
    return interval->end_point;
  }
  return interval->start_point == UINT32_MAX ? UINT32_MAX
                                             : interval->start_point + 1u;
}

static iree_status_t loom_low_allocation_checker_assignments(
    loom_low_allocation_checker_t* checker) {
  const loom_low_allocation_table_t* allocation = checker->allocation;
  const loom_liveness_analysis_t* liveness = &allocation->liveness;
  if (liveness->value_count != checker->schedule->value_count ||
      liveness->value_count != allocation->placement.value_count ||
      (liveness->value_count != 0 &&
       (liveness->value_ids == NULL || checker->schedule->value_ids == NULL ||
        allocation->placement.value_ids == NULL ||
        allocation->assignment_indices_by_value_ordinal == NULL))) {
    loom_low_allocation_checker_record(
        checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_VALUE_DOMAIN, UINT32_MAX,
        UINT32_MAX, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, UINT32_MAX);
    return iree_ok_status();
  }
  for (loom_value_ordinal_t ordinal = 0; ordinal < liveness->value_count;
       ++ordinal) {
    if (checker->schedule->value_ids[ordinal] != liveness->value_ids[ordinal] ||
        allocation->placement.value_ids[ordinal] !=
            liveness->value_ids[ordinal]) {
      loom_low_allocation_checker_record(
          checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_VALUE_DOMAIN, ordinal,
          UINT32_MAX, liveness->value_ids[ordinal],
          checker->schedule->value_ids[ordinal], UINT32_MAX);
    }
  }
  if (allocation->assignment_count == 0) {
    for (iree_host_size_t i = 0; i < liveness->interval_count; ++i) {
      if (loom_low_allocation_checker_interval_is_allocatable(
              &liveness->intervals[i])) {
        loom_low_allocation_checker_record(
            checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_ASSIGNMENT_INDEX,
            UINT32_MAX, (uint32_t)i, liveness->intervals[i].value_id,
            LOOM_VALUE_ID_INVALID, liveness->intervals[i].start_point);
      }
    }
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(checker->arena, allocation->assignment_count,
                                sizeof(*checker->assignment_ordinals),
                                (void**)&checker->assignment_ordinals));
  uint8_t* seen = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(checker->arena,
                                                 allocation->assignment_count,
                                                 sizeof(*seen), (void**)&seen));
  memset(seen, 0, allocation->assignment_count * sizeof(*seen));
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    checker->assignment_ordinals[i] = LOOM_VALUE_ORDINAL_INVALID;
  }

  for (loom_value_ordinal_t ordinal = 0; ordinal < liveness->value_count;
       ++ordinal) {
    const uint32_t assignment_index =
        allocation->assignment_indices_by_value_ordinal[ordinal];
    const loom_liveness_interval_t* interval =
        loom_liveness_interval_for_value_ordinal(liveness, ordinal);
    if (assignment_index == UINT32_MAX) {
      if (interval &&
          loom_low_allocation_checker_interval_is_allocatable(interval)) {
        loom_low_allocation_checker_record(
            checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_ASSIGNMENT_INDEX,
            ordinal, UINT32_MAX, interval->value_id, LOOM_VALUE_ID_INVALID,
            interval->start_point);
      }
      continue;
    }
    if (assignment_index >= allocation->assignment_count) {
      loom_low_allocation_checker_record(
          checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_ASSIGNMENT_INDEX,
          ordinal, assignment_index, liveness->value_ids[ordinal],
          LOOM_VALUE_ID_INVALID, UINT32_MAX);
      continue;
    }
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[assignment_index];
    if (seen[assignment_index] ||
        assignment->value_id != liveness->value_ids[ordinal]) {
      loom_low_allocation_checker_record(
          checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_ASSIGNMENT_INDEX,
          assignment_index, ordinal, assignment->value_id,
          liveness->value_ids[ordinal], assignment->start_point);
    }
    seen[assignment_index] = 1;
    checker->assignment_ordinals[assignment_index] = ordinal;
    if (interval == NULL ||
        !loom_low_allocation_checker_assignment_start_shape_is_valid(
            assignment, interval) ||
        assignment->end_point <
            loom_low_allocation_checker_storage_end_point(interval) ||
        assignment->unit_count != interval->unit_count ||
        assignment->location_count != assignment->unit_count ||
        !loom_liveness_value_class_equal(assignment->value_class,
                                         interval->value_class) ||
        assignment->descriptor_reg_class_id !=
            interval->value_class.register_class_id ||
        !loom_low_allocation_location_kind_is_known(
            assignment->location_kind) ||
        assignment->location_kind == LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED) {
      loom_low_allocation_checker_record(
          checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_ASSIGNMENT_SHAPE,
          assignment_index, ordinal, assignment->value_id,
          LOOM_VALUE_ID_INVALID, assignment->start_point);
    }
    const uint64_t unit_end =
        (uint64_t)assignment->unit_point_start + assignment->unit_count;
    if (unit_end > allocation->unit_point_count) {
      loom_low_allocation_checker_record(
          checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_ASSIGNMENT_SHAPE,
          assignment_index, ordinal, assignment->value_id,
          LOOM_VALUE_ID_INVALID, assignment->start_point);
    } else {
      uint32_t minimum_start_point = UINT32_MAX;
      for (uint32_t unit = 0; unit < assignment->unit_count; ++unit) {
        const uint32_t start_point =
            loom_low_allocation_live_range_assignment_unit_start_point(
                allocation->unit_start_points, allocation->unit_point_count,
                assignment, unit);
        minimum_start_point = iree_min(minimum_start_point, start_point);
        const uint32_t end_point =
            allocation->unit_end_points[assignment->unit_point_start + unit];
        if (start_point < assignment->start_point ||
            start_point > interval->start_point || end_point < start_point ||
            end_point > assignment->end_point) {
          loom_low_allocation_checker_record(
              checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_ASSIGNMENT_SHAPE,
              assignment_index, unit, assignment->value_id,
              LOOM_VALUE_ID_INVALID, end_point);
        }
      }
      if (iree_any_bit_set(
              assignment->flags,
              LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS) &&
          assignment->start_point != minimum_start_point) {
        loom_low_allocation_checker_record(
            checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_ASSIGNMENT_SHAPE,
            assignment_index, ordinal, assignment->value_id,
            LOOM_VALUE_ID_INVALID, assignment->start_point);
      }
    }
  }
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    if (!seen[i]) {
      loom_low_allocation_checker_record(
          checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_ASSIGNMENT_INDEX,
          (uint32_t)i, UINT32_MAX, allocation->assignments[i].value_id,
          LOOM_VALUE_ID_INVALID, allocation->assignments[i].start_point);
    }
  }
  return iree_ok_status();
}

static const loom_low_allocation_assignment_t*
loom_low_allocation_checker_assignment_for_ordinal(
    const loom_low_allocation_checker_t* checker, loom_value_ordinal_t ordinal,
    uint32_t* out_assignment_index) {
  *out_assignment_index = UINT32_MAX;
  if (ordinal >= checker->allocation->liveness.value_count) {
    return NULL;
  }
  const uint32_t assignment_index =
      checker->allocation->assignment_indices_by_value_ordinal[ordinal];
  if (assignment_index >= checker->allocation->assignment_count) {
    return NULL;
  }
  *out_assignment_index = assignment_index;
  return &checker->allocation->assignments[assignment_index];
}

static bool loom_low_allocation_checker_relation_range_fits(
    const loom_low_allocation_assignment_t* assignment, uint32_t offset,
    uint32_t count) {
  return offset <= assignment->unit_count &&
         count <= assignment->unit_count - offset;
}

static bool loom_low_allocation_checker_relation_satisfied(
    const loom_low_allocation_checker_t* checker,
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* result,
    const loom_low_allocation_assignment_t* source) {
  if (!loom_low_allocation_checker_relation_range_fits(
          result, relation->result_unit_offset, relation->unit_count) ||
      !loom_low_allocation_checker_relation_range_fits(
          source, relation->source_unit_offset, relation->unit_count)) {
    return false;
  }
  return loom_low_allocation_storage_placement_relation_satisfied(
      checker->descriptor_set, relation, result, source);
}

static void loom_low_allocation_checker_constraints(
    loom_low_allocation_checker_t* checker) {
  const loom_low_allocation_table_t* allocation = checker->allocation;
  for (iree_host_size_t i = 0; i < allocation->fixed_value_count; ++i) {
    const loom_low_allocation_resolved_fixed_value_t* fixed =
        &allocation->fixed_values[i];
    uint32_t assignment_index = UINT32_MAX;
    const loom_low_allocation_assignment_t* assignment =
        loom_low_allocation_checker_assignment_for_ordinal(
            checker, fixed->value_ordinal, &assignment_index);
    const loom_low_allocation_assignment_t* required = &fixed->assignment;
    if (assignment == NULL || assignment->value_id != required->value_id ||
        assignment->descriptor_reg_class_id !=
            required->descriptor_reg_class_id ||
        assignment->location_kind != required->location_kind ||
        assignment->location_base != required->location_base ||
        assignment->location_count != required->location_count) {
      loom_low_allocation_checker_record(
          checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_FIXED_LOCATION,
          (uint32_t)i, assignment_index, required->value_id,
          LOOM_VALUE_ID_INVALID, required->start_point);
    }
  }

  for (iree_host_size_t i = 0; i < allocation->reserved_range_count; ++i) {
    const loom_low_allocation_resolved_reserved_range_t* reserved =
        &allocation->reserved_ranges[i];
    const loom_low_allocation_assignment_t reserved_assignment = {
        .descriptor_reg_class_id = reserved->descriptor_reg_class_id,
        .location_kind = reserved->location_kind,
        .location_base = reserved->location_base,
        .location_count = reserved->location_count,
    };
    for (iree_host_size_t j = 0; j < allocation->assignment_count; ++j) {
      const loom_low_allocation_assignment_t* assignment =
          &allocation->assignments[j];
      if (loom_low_allocation_storage_assignment_ranges_overlap(
              checker->descriptor_set, assignment, &reserved_assignment)) {
        loom_low_allocation_checker_record(
            checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_RESERVED_LOCATION,
            (uint32_t)i, (uint32_t)j, assignment->value_id,
            LOOM_VALUE_ID_INVALID, assignment->start_point);
      }
    }
  }

  for (iree_host_size_t i = 0; i < allocation->placement.relation_count; ++i) {
    const loom_low_placement_relation_t* relation =
        &allocation->placement.relations[i];
    if (!iree_any_bit_set(relation->flags,
                          LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD)) {
      continue;
    }
    uint32_t result_assignment_index = UINT32_MAX;
    uint32_t source_assignment_index = UINT32_MAX;
    const loom_low_allocation_assignment_t* result =
        loom_low_allocation_checker_assignment_for_ordinal(
            checker, relation->result_ordinal, &result_assignment_index);
    const loom_low_allocation_assignment_t* source =
        loom_low_allocation_checker_assignment_for_ordinal(
            checker, relation->source_ordinal, &source_assignment_index);
    if (result == NULL || source == NULL ||
        !loom_low_allocation_checker_relation_satisfied(checker, relation,
                                                        result, source)) {
      loom_low_allocation_checker_record(
          checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_PLACEMENT_RELATION,
          (uint32_t)i, result_assignment_index,
          result ? result->value_id : LOOM_VALUE_ID_INVALID,
          source ? source->value_id : LOOM_VALUE_ID_INVALID, UINT32_MAX);
    }
  }
}

static bool loom_low_allocation_checker_segments_overlap(
    const loom_liveness_analysis_t* liveness,
    loom_liveness_segment_range_t lhs_range, uint32_t lhs_end,
    loom_liveness_segment_range_t rhs_range, uint32_t rhs_end) {
  if (lhs_range.count == 0 || rhs_range.count == 0) {
    return true;
  }
  if ((uint64_t)lhs_range.start + lhs_range.count > liveness->segment_count ||
      (uint64_t)rhs_range.start + rhs_range.count > liveness->segment_count) {
    return true;
  }
  uint32_t lhs_index = 0;
  uint32_t rhs_index = 0;
  while (lhs_index < lhs_range.count && rhs_index < rhs_range.count) {
    const loom_liveness_segment_t* lhs =
        &liveness->segments[lhs_range.start + lhs_index];
    const loom_liveness_segment_t* rhs =
        &liveness->segments[rhs_range.start + rhs_index];
    const uint32_t begin = iree_max(lhs->start_point, rhs->start_point);
    const uint32_t end = iree_min(iree_min(lhs->end_point, rhs->end_point),
                                  iree_min(lhs_end, rhs_end));
    if (begin < end) {
      return true;
    }
    if (lhs->end_point < rhs->end_point) {
      ++lhs_index;
    } else {
      ++rhs_index;
    }
  }
  return false;
}

static uint32_t loom_low_allocation_checker_unit_end_point(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_offset) {
  const uint64_t index = (uint64_t)assignment->unit_point_start + unit_offset;
  if (unit_offset >= assignment->unit_count ||
      index >= allocation->unit_point_count) {
    return assignment->end_point;
  }
  return allocation->unit_end_points[index];
}

static bool loom_low_allocation_checker_unit_lifetimes_overlap(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* lhs, uint32_t lhs_unit,
    const loom_low_allocation_assignment_t* rhs, uint32_t rhs_unit) {
  const uint32_t lhs_end =
      loom_low_allocation_checker_unit_end_point(allocation, lhs, lhs_unit);
  const uint32_t rhs_end =
      loom_low_allocation_checker_unit_end_point(allocation, rhs, rhs_unit);
  const uint32_t lhs_start =
      loom_low_allocation_live_range_assignment_unit_start_point(
          allocation->unit_start_points, allocation->unit_point_count, lhs,
          lhs_unit);
  const uint32_t rhs_start =
      loom_low_allocation_live_range_assignment_unit_start_point(
          allocation->unit_start_points, allocation->unit_point_count, rhs,
          rhs_unit);
  if (lhs_start >= rhs_end || rhs_start >= lhs_end) {
    return false;
  }
  return loom_low_allocation_checker_segments_overlap(
      &allocation->liveness, lhs->liveness_segments, lhs_end,
      rhs->liveness_segments, rhs_end);
}

static bool loom_low_allocation_checker_unit_alias_is_authorized(
    const loom_low_allocation_checker_t* checker,
    loom_value_ordinal_t lhs_ordinal, uint32_t lhs_unit,
    loom_value_ordinal_t rhs_ordinal, uint32_t rhs_unit) {
  const loom_low_placement_table_t* placement = &checker->allocation->placement;
  for (iree_host_size_t i = 0; i < placement->relation_count; ++i) {
    const loom_low_placement_relation_t* relation = &placement->relations[i];
    if (!loom_low_placement_relation_can_alias(relation)) {
      continue;
    }
    uint32_t result_unit = lhs_unit;
    uint32_t source_unit = rhs_unit;
    if (relation->result_ordinal == rhs_ordinal &&
        relation->source_ordinal == lhs_ordinal) {
      result_unit = rhs_unit;
      source_unit = lhs_unit;
    } else if (relation->result_ordinal != lhs_ordinal ||
               relation->source_ordinal != rhs_ordinal) {
      continue;
    }
    if (result_unit < relation->result_unit_offset ||
        source_unit < relation->source_unit_offset) {
      continue;
    }
    const uint32_t result_relative_unit =
        result_unit - relation->result_unit_offset;
    const uint32_t source_relative_unit =
        source_unit - relation->source_unit_offset;
    if (result_relative_unit == source_relative_unit &&
        result_relative_unit < relation->unit_count) {
      return true;
    }
  }
  return false;
}

static uint32_t loom_low_allocation_checker_storage_key(
    const loom_low_allocation_checker_t* checker,
    const loom_low_allocation_assignment_t* assignment) {
  if (assignment->location_kind == LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT &&
      assignment->descriptor_reg_class_id <
          checker->descriptor_set->reg_class_count) {
    return checker->descriptor_set
        ->reg_classes[assignment->descriptor_reg_class_id]
        .spill_slot_space;
  }
  return loom_low_reg_class_storage_key(checker->descriptor_set,
                                        assignment->descriptor_reg_class_id);
}

static void loom_low_allocation_checker_storage_conflicts(
    loom_low_allocation_checker_t* checker) {
  const loom_low_allocation_table_t* allocation = checker->allocation;
  if (checker->assignment_ordinals == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* lhs = &allocation->assignments[i];
    if (lhs->location_count == 0) {
      continue;
    }
    const bool lhs_is_explicit =
        loom_low_allocation_storage_assignment_uses_explicit_physical_register(
            checker->descriptor_set, lhs);
    const uint64_t lhs_end = (uint64_t)lhs->location_base + lhs->location_count;
    for (iree_host_size_t j = i + 1; j < allocation->assignment_count; ++j) {
      const loom_low_allocation_assignment_t* rhs = &allocation->assignments[j];
      const bool rhs_is_explicit =
          loom_low_allocation_storage_assignment_uses_explicit_physical_register(
              checker->descriptor_set, rhs);
      if (lhs_is_explicit || rhs_is_explicit) {
        if (!lhs_is_explicit || !rhs_is_explicit ||
            !loom_low_allocation_storage_assignment_ranges_overlap(
                checker->descriptor_set, lhs, rhs)) {
          continue;
        }
        bool conflict = false;
        for (uint32_t lhs_unit = 0; lhs_unit < lhs->location_count && !conflict;
             ++lhs_unit) {
          for (uint32_t rhs_unit = 0; rhs_unit < rhs->location_count;
               ++rhs_unit) {
            if (!loom_low_allocation_storage_assignment_subranges_overlap(
                    checker->descriptor_set, lhs, lhs_unit, rhs, rhs_unit,
                    /*unit_count=*/1) ||
                !loom_low_allocation_checker_unit_lifetimes_overlap(
                    allocation, lhs, lhs_unit, rhs, rhs_unit) ||
                loom_low_allocation_checker_unit_alias_is_authorized(
                    checker, checker->assignment_ordinals[i], lhs_unit,
                    checker->assignment_ordinals[j], rhs_unit)) {
              continue;
            }
            conflict = true;
            break;
          }
        }
        if (conflict) {
          loom_low_allocation_checker_record(
              checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_CONFLICT,
              (uint32_t)i, (uint32_t)j, lhs->value_id, rhs->value_id,
              iree_max(lhs->start_point, rhs->start_point));
        }
        continue;
      }
      if (lhs->location_kind != rhs->location_kind ||
          loom_low_allocation_checker_storage_key(checker, lhs) !=
              loom_low_allocation_checker_storage_key(checker, rhs) ||
          rhs->location_count == 0) {
        continue;
      }
      const uint64_t rhs_end =
          (uint64_t)rhs->location_base + rhs->location_count;
      const uint64_t overlap_begin =
          iree_max((uint64_t)lhs->location_base, (uint64_t)rhs->location_base);
      const uint64_t overlap_end = iree_min(lhs_end, rhs_end);
      for (uint64_t location = overlap_begin; location < overlap_end;
           ++location) {
        const uint32_t lhs_unit = (uint32_t)(location - lhs->location_base);
        const uint32_t rhs_unit = (uint32_t)(location - rhs->location_base);
        if (!loom_low_allocation_checker_unit_lifetimes_overlap(
                allocation, lhs, lhs_unit, rhs, rhs_unit) ||
            loom_low_allocation_checker_unit_alias_is_authorized(
                checker, checker->assignment_ordinals[i], lhs_unit,
                checker->assignment_ordinals[j], rhs_unit)) {
          continue;
        }
        loom_low_allocation_checker_record(
            checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_CONFLICT,
            (uint32_t)i, (uint32_t)j, lhs->value_id, rhs->value_id,
            iree_max(lhs->start_point, rhs->start_point));
        break;
      }
    }
  }
}

static bool loom_low_allocation_checker_units_overlap(
    const loom_low_allocation_checker_t* checker,
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  return loom_low_allocation_storage_assignment_ranges_overlap(
      checker->descriptor_set, lhs, rhs);
}

static void loom_low_allocation_checker_early_clobbers(
    loom_low_allocation_checker_t* checker) {
  const loom_low_schedule_table_t* schedule = checker->schedule;
  for (iree_host_size_t node_index = 0; node_index < schedule->node_count;
       ++node_index) {
    const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
    if (!iree_any_bit_set(node->flags,
                          LOOM_LOW_SCHEDULE_NODE_FLAG_EARLY_CLOBBER)) {
      continue;
    }
    const loom_value_ordinal_t* operands =
        loom_low_schedule_node_const_operand_ordinals(node);
    const loom_value_ordinal_t* results =
        loom_low_schedule_node_const_result_ordinals(node);
    for (uint16_t result_index = 0; result_index < node->result_count;
         ++result_index) {
      uint32_t result_assignment_index = UINT32_MAX;
      const loom_low_allocation_assignment_t* result =
          loom_low_allocation_checker_assignment_for_ordinal(
              checker, results[result_index], &result_assignment_index);
      if (result == NULL) {
        continue;
      }
      for (uint16_t operand_index = 0; operand_index < node->operand_count;
           ++operand_index) {
        uint32_t operand_assignment_index = UINT32_MAX;
        const loom_low_allocation_assignment_t* operand =
            loom_low_allocation_checker_assignment_for_ordinal(
                checker, operands[operand_index], &operand_assignment_index);
        if (operand == NULL || !loom_low_allocation_checker_units_overlap(
                                   checker, result, operand)) {
          continue;
        }
        bool tied = false;
        for (iree_host_size_t i = 0;
             i < checker->allocation->placement.relation_count; ++i) {
          const loom_low_placement_relation_t* relation =
              &checker->allocation->placement.relations[i];
          if (relation->op == node->op &&
              relation->cause == LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT &&
              iree_any_bit_set(relation->flags,
                               LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD) &&
              relation->result_ordinal == results[result_index] &&
              relation->source_ordinal == operands[operand_index] &&
              loom_low_allocation_checker_relation_satisfied(checker, relation,
                                                             result, operand)) {
            tied = true;
            break;
          }
        }
        if (!tied) {
          loom_low_allocation_checker_record(
              checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_EARLY_CLOBBER,
              (uint32_t)node_index, result_assignment_index, result->value_id,
              operand->value_id,
              checker->node_program_points
                  ? checker->node_program_points[node_index]
                  : UINT32_MAX);
        }
      }
    }
  }
}

static uint32_t loom_low_allocation_checker_release_program_point(
    const loom_low_allocation_checker_t* checker,
    const loom_low_storage_release_action_t* action) {
  if (action->insertion_node_index >= checker->schedule->node_count ||
      checker->node_program_points == NULL) {
    return UINT32_MAX;
  }
  return checker->node_program_points[action->insertion_node_index];
}

static void loom_low_allocation_checker_storage_leases(
    loom_low_allocation_checker_t* checker) {
  const loom_low_allocation_table_t* allocation = checker->allocation;
  if (allocation->storage_lease_instance_count !=
      allocation->storage_leases.record_count) {
    loom_low_allocation_checker_record(
        checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_LEASE, UINT32_MAX,
        UINT32_MAX, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, UINT32_MAX);
  }
  for (iree_host_size_t i = 0; i < allocation->storage_lease_instance_count;
       ++i) {
    const loom_low_allocation_storage_lease_t* lease =
        &allocation->storage_lease_instances[i];
    if (lease->assignment_index >= allocation->assignment_count ||
        lease->lease_record_index >= allocation->storage_leases.record_count) {
      loom_low_allocation_checker_record(
          checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_LEASE,
          (uint32_t)i, lease->assignment_index, lease->value_id,
          LOOM_VALUE_ID_INVALID, lease->start_point);
      continue;
    }
    const loom_low_allocation_assignment_t* owner =
        &allocation->assignments[lease->assignment_index];
    const uint64_t lease_end =
        (uint64_t)lease->location_base + lease->location_count;
    const uint64_t owner_end =
        (uint64_t)owner->location_base + owner->location_count;
    if (owner->value_id != lease->value_id ||
        owner->location_kind != lease->location_kind ||
        loom_low_allocation_checker_storage_key(checker, owner) !=
            loom_low_reg_class_storage_key(checker->descriptor_set,
                                           lease->descriptor_reg_class_id) ||
        lease->location_base < owner->location_base || lease_end > owner_end ||
        lease->start_point > lease->end_point) {
      loom_low_allocation_checker_record(
          checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_LEASE,
          (uint32_t)i, lease->assignment_index, lease->value_id,
          LOOM_VALUE_ID_INVALID, lease->start_point);
    }

    uint32_t effective_end = lease->end_point;
    if (lease->release_action_index !=
        LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE) {
      if (lease->release_action_index >=
          allocation->storage_release_action_count) {
        loom_low_allocation_checker_record(
            checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_LEASE,
            (uint32_t)i, lease->release_action_index, lease->value_id,
            LOOM_VALUE_ID_INVALID, lease->start_point);
        continue;
      }
      const loom_low_storage_release_action_t* action =
          &allocation->storage_release_actions[lease->release_action_index];
      effective_end =
          loom_low_allocation_checker_release_program_point(checker, action);
      if (action->lease_record_index != lease->lease_record_index ||
          effective_end == UINT32_MAX || effective_end < lease->start_point ||
          effective_end > lease->end_point) {
        loom_low_allocation_checker_record(
            checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_LEASE,
            (uint32_t)i, lease->release_action_index, lease->value_id,
            LOOM_VALUE_ID_INVALID, effective_end);
      }
    }

    for (iree_host_size_t j = 0; j < allocation->assignment_count; ++j) {
      if (j == lease->assignment_index) {
        continue;
      }
      const loom_low_allocation_assignment_t* candidate =
          &allocation->assignments[j];
      if (candidate->start_point >= effective_end ||
          candidate->end_point <= lease->start_point ||
          candidate->location_kind != lease->location_kind ||
          loom_low_allocation_checker_storage_key(checker, candidate) !=
              loom_low_reg_class_storage_key(checker->descriptor_set,
                                             lease->descriptor_reg_class_id)) {
        continue;
      }
      const loom_low_allocation_assignment_t lease_assignment = {
          .descriptor_reg_class_id = lease->descriptor_reg_class_id,
          .location_kind = lease->location_kind,
          .location_base = lease->location_base,
          .location_count = lease->location_count,
      };
      if (loom_low_allocation_storage_assignment_ranges_overlap(
              checker->descriptor_set, candidate, &lease_assignment)) {
        loom_low_allocation_checker_record(
            checker, LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_LEASE,
            (uint32_t)i, (uint32_t)j, lease->value_id, candidate->value_id,
            iree_max(lease->start_point, candidate->start_point));
      }
    }
  }
}

iree_status_t loom_low_allocation_check_frame(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_low_allocation_check_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(frame);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_result);
  *out_result = (loom_low_allocation_check_result_t){
      .first_violation =
          {
              .kind = LOOM_LOW_ALLOCATION_CHECK_VIOLATION_NONE,
              .primary_index = UINT32_MAX,
              .secondary_index = UINT32_MAX,
              .value_id = LOOM_VALUE_ID_INVALID,
              .related_value_id = LOOM_VALUE_ID_INVALID,
              .program_point = UINT32_MAX,
          },
  };
  loom_low_allocation_checker_t checker = {
      .frame = frame,
      .schedule = &frame->schedule,
      .allocation = &frame->allocation,
      .descriptor_set = frame->target.descriptor_set,
      .arena = arena,
      .result = out_result,
  };
  if (!loom_low_allocation_checker_frame_identity(&checker) ||
      checker.descriptor_set == NULL || frame->schedule.error_count != 0 ||
      frame->allocation.error_count != 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_allocation_checker_schedule(&checker));
  IREE_RETURN_IF_ERROR(loom_low_allocation_checker_assignments(&checker));
  loom_low_allocation_checker_constraints(&checker);
  loom_low_allocation_checker_storage_conflicts(&checker);
  loom_low_allocation_checker_early_clobbers(&checker);
  loom_low_allocation_checker_storage_leases(&checker);
  return iree_ok_status();
}
