// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/storage_lease.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/placement.h"

static iree_status_t loom_low_allocation_validate_storage_lease_table(
    const loom_low_storage_lease_table_t* lease_table,
    const loom_module_t* module, const loom_op_t* function_op) {
  if (lease_table->record_count == 0) {
    return iree_ok_status();
  }
  if (lease_table->schedule == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocation storage leases have records but no schedule");
  }
  if (lease_table->records == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocation storage leases have a record count but no records");
  }
  if (lease_table->schedule->module != module ||
      lease_table->schedule->function_op != function_op) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocation storage leases must describe the allocated low function");
  }
  if (lease_table->record_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation storage lease count exceeds u32 "
                            "range");
  }
  return iree_ok_status();
}

static bool loom_low_allocation_value_ordinal_for_liveness_value(
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, loom_value_id_t value_id,
    loom_value_ordinal_t* out_value_ordinal) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(value_domain, value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      value_ordinal >= liveness->value_count ||
      liveness->value_ids[value_ordinal] != value_id) {
    return false;
  }
  *out_value_ordinal = value_ordinal;
  return true;
}

static bool loom_low_allocation_storage_relation_preserves_identity(
    const loom_low_placement_relation_t* relation) {
  return relation->cause >= LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT &&
         relation->cause <= LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT &&
         loom_low_placement_relation_can_alias(relation) &&
         !iree_any_bit_set(relation->flags,
                           LOOM_LOW_PLACEMENT_RELATION_FLAG_WRITES_STORAGE |
                               LOOM_LOW_PLACEMENT_RELATION_FLAG_CAPTURED_PART);
}

iree_status_t loom_low_allocation_storage_identity_initialize(
    const loom_low_storage_lease_table_t* lease_table,
    const loom_low_placement_table_t* placement,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    iree_arena_allocator_t* arena,
    loom_low_allocation_storage_identity_t* out_identity) {
  IREE_ASSERT_ARGUMENT(lease_table);
  IREE_ASSERT_ARGUMENT(placement);
  IREE_ASSERT_ARGUMENT(unit_liveness);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_identity);
  *out_identity = (loom_low_allocation_storage_identity_t){0};
  if (lease_table->record_count == 0 ||
      !iree_any_bit_set(placement->storage.flags,
                        LOOM_LOW_PLACEMENT_STORAGE_FLAG_OPTIONAL_ALIASES |
                            LOOM_LOW_PLACEMENT_STORAGE_FLAG_IDENTITY_ALIASES) ||
      placement->storage_value_order_count == 0 ||
      unit_liveness->point_count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(placement->storage_value_order_count, placement->value_count);
  uint32_t* origins = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, unit_liveness->point_count, sizeof(*origins), (void**)&origins));
  for (iree_host_size_t i = 0; i < unit_liveness->point_count; ++i) {
    origins[i] = i;
  }
  // The retained order visits users before sources, so reverse traversal
  // assigns each user from an already canonical source root.
  for (loom_value_ordinal_t cursor = placement->storage_value_order_count;
       cursor > 0; --cursor) {
    const loom_value_ordinal_t result_ordinal =
        placement->storage_value_order[cursor - 1];
    const uint32_t result_start =
        unit_liveness->values[result_ordinal].unit_point_start;
    if (result_start == UINT32_MAX) {
      continue;
    }
    const loom_low_placement_relation_range_t range =
        placement->ranges_by_result_ordinal[result_ordinal];
    for (uint32_t i = 0; i < range.count; ++i) {
      const loom_low_placement_relation_t* relation =
          &placement->relations[range.start + i];
      if (!loom_low_allocation_storage_relation_preserves_identity(relation)) {
        continue;
      }
      const uint32_t source_start =
          unit_liveness->values[relation->source_ordinal].unit_point_start;
      if (source_start == UINT32_MAX) {
        continue;
      }
      IREE_ASSERT_LE((uint64_t)source_start + relation->source_unit_offset +
                         relation->unit_count,
                     unit_liveness->point_count);
      IREE_ASSERT_LE((uint64_t)result_start + relation->result_unit_offset +
                         relation->unit_count,
                     unit_liveness->point_count);
      for (uint32_t unit = 0; unit < relation->unit_count; ++unit) {
        origins[result_start + relation->result_unit_offset + unit] =
            origins[source_start + relation->source_unit_offset + unit];
      }
    }
  }
  out_identity->origins = origins;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_storage_lease_value_id(
    const loom_low_storage_lease_table_t* lease_table,
    const loom_low_storage_lease_record_t* record,
    loom_value_id_t* out_value_id) {
  const loom_low_schedule_table_t* schedule = lease_table->schedule;
  if (record->node_index >= schedule->node_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "storage lease references schedule node %" PRIu32
                            " outside the schedule",
                            record->node_index);
  }
  const loom_low_schedule_node_t* node = &schedule->nodes[record->node_index];
  if (node->block_index != record->block_index ||
      node->scheduled_ordinal != record->scheduled_ordinal) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "storage lease record no longer matches its schedule node");
  }

  const loom_value_ordinal_t* value_ordinals = NULL;
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (record->attachment == LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND) {
    if (record->attachment_index >= node->operand_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "storage lease operand index exceeds schedule "
                              "node operand count");
    }
    value_ordinals = loom_low_schedule_node_const_operand_ordinals(node);
    value_ordinal = value_ordinals[record->attachment_index];
  } else if (record->attachment == LOOM_LOW_STORAGE_LEASE_ATTACHMENT_RESULT) {
    if (record->attachment_index >= node->result_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "storage lease result index exceeds schedule "
                              "node result count");
    }
    value_ordinals = loom_low_schedule_node_const_result_ordinals(node);
    value_ordinal = value_ordinals[record->attachment_index];
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "storage lease record has invalid attachment kind %u",
        (unsigned)record->attachment);
  }
  if (value_ordinal >= schedule->value_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "storage lease references a value outside schedule value domain");
  }
  *out_value_id = schedule->value_ids[value_ordinal];
  return iree_ok_status();
}

static bool loom_low_allocation_storage_lease_overlaps_liveness(
    const loom_liveness_segment_t* storage_segments,
    const loom_low_allocation_storage_lease_t* lease,
    const loom_low_allocation_assignment_t* candidate) {
  if (lease->end_point <= candidate->start_point ||
      lease->start_point >= candidate->end_point) {
    return false;
  }
  if (candidate->liveness_segments.count == 0) {
    return true;
  }
  // Allocation retains these segments only when they cover storage liveness.
  // The lease keeps its full asynchronous lifetime; only the candidate's holes
  // are excluded.
  for (uint32_t i = 0; i < candidate->liveness_segments.count; ++i) {
    const loom_liveness_segment_t* segment =
        &storage_segments[candidate->liveness_segments.start + i];
    if (segment->start_point >= lease->end_point) {
      return false;
    }
    if (lease->start_point < segment->end_point) {
      return true;
    }
  }
  return false;
}

// Candidate units with no demand occupy no storage, even when another unit
// keeps the aggregate live. The lease retains every unit for its full lifetime.
static bool loom_low_allocation_storage_lease_overlaps_unit_liveness(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_allocation_storage_lease_t* lease,
    const loom_low_allocation_assignment_t* candidate,
    uint32_t candidate_unit) {
  const uint32_t start_point =
      loom_low_allocation_live_range_assignment_unit_start_point(
          unit_liveness->start_points, unit_liveness->point_count, candidate,
          candidate_unit);
  const uint32_t end_point =
      loom_low_allocation_live_range_assignment_unit_end_point(
          unit_liveness->end_points, unit_liveness->point_count, candidate,
          candidate_unit);
  const uint32_t overlap_start = iree_max(start_point, lease->start_point);
  return overlap_start < end_point && overlap_start < lease->end_point;
}

static bool loom_low_allocation_storage_lease_instance_conflicts(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t lease_record_index,
    const loom_low_allocation_storage_lease_t* lease,
    const loom_low_allocation_assignment_t* candidate) {
  if (!loom_low_allocation_storage_lease_overlaps_liveness(
          state->unit_liveness->storage_segments.entries, lease, candidate)) {
    return false;
  }
  if (lease->location_kind != candidate->location_kind) {
    return false;
  }
  if (!loom_low_allocation_storage_reg_classes_share(
          descriptor_set, lease->descriptor_reg_class_id,
          candidate->descriptor_reg_class_id)) {
    return false;
  }
  const bool uses_explicit_registers =
      loom_low_reg_class_uses_explicit_physical_registers(
          &descriptor_set->reg_classes[lease->descriptor_reg_class_id]) ||
      loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, candidate);
  if (uses_explicit_registers) {
    const loom_low_storage_lease_record_t* record =
        &state->lease_table->records[lease_record_index];
    const loom_low_allocation_assignment_t* lease_assignment =
        &state->assignments[lease->assignment_index];
    for (uint32_t candidate_unit = 0;
         candidate_unit < candidate->location_count; ++candidate_unit) {
      if (!loom_low_allocation_storage_lease_overlaps_unit_liveness(
              state->unit_liveness, lease, candidate, candidate_unit)) {
        continue;
      }
      for (uint32_t lease_unit = 0; lease_unit < record->unit_count;
           ++lease_unit) {
        if (loom_low_allocation_storage_assignment_subranges_overlap(
                descriptor_set, candidate, candidate_unit, lease_assignment,
                record->unit_offset + lease_unit, 1)) {
          return true;
        }
      }
    }
    return false;
  }
  const uint64_t lease_begin = lease->location_base;
  const uint64_t lease_end = lease_begin + lease->location_count;
  const uint64_t candidate_begin = candidate->location_base;
  const uint64_t candidate_end = candidate_begin + candidate->location_count;
  const uint64_t overlap_begin = iree_max(lease_begin, candidate_begin);
  const uint64_t overlap_end = iree_min(lease_end, candidate_end);
  for (uint64_t location = overlap_begin; location < overlap_end; ++location) {
    if (loom_low_allocation_storage_lease_overlaps_unit_liveness(
            state->unit_liveness, lease, candidate,
            (uint32_t)(location - candidate_begin))) {
      return true;
    }
  }
  return false;
}

static uint32_t loom_low_allocation_storage_identity_unit_start(
    const loom_low_allocation_storage_lease_state_t* state,
    loom_value_id_t value_id) {
  if (state->identity_origins == NULL) {
    return UINT32_MAX;
  }
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_ordinal(state->value_domain, value_id);
  const uint32_t unit_start =
      state->unit_liveness->values[value_ordinal].unit_point_start;
  IREE_ASSERT_NE(unit_start, UINT32_MAX);
  return unit_start;
}

// Returns true only when every live overlapping semantic unit carries the same
// structural content. Partial physical aliases remain conflicts: a lease owns
// every atomic part of its original allocation unit.
static bool loom_low_allocation_storage_lease_overlap_preserves_identity(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t lease_record_index,
    const loom_low_allocation_storage_lease_t* lease,
    const loom_low_allocation_assignment_t* candidate,
    uint32_t candidate_identity_start) {
  if (candidate_identity_start == UINT32_MAX) {
    return false;
  }
  const uint32_t lease_identity_start =
      loom_low_allocation_storage_identity_unit_start(state, lease->value_id);

  const loom_low_storage_lease_record_t* record =
      &state->lease_table->records[lease_record_index];
  const loom_low_allocation_assignment_t* lease_assignment =
      &state->assignments[lease->assignment_index];
  IREE_ASSERT_LE(record->unit_offset, lease_assignment->location_count);
  IREE_ASSERT_LE(record->unit_count,
                 lease_assignment->location_count - record->unit_offset);
  IREE_ASSERT_LE(candidate->location_count, candidate->unit_count);

  const bool uses_explicit_registers =
      loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, lease_assignment) ||
      loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, candidate);
  if (uses_explicit_registers) {
    bool overlaps = false;
    for (uint32_t candidate_unit = 0;
         candidate_unit < candidate->location_count; ++candidate_unit) {
      if (!loom_low_allocation_storage_lease_overlaps_unit_liveness(
              state->unit_liveness, lease, candidate, candidate_unit)) {
        continue;
      }
      for (uint32_t lease_unit = 0; lease_unit < record->unit_count;
           ++lease_unit) {
        const uint32_t source_unit = record->unit_offset + lease_unit;
        if (!loom_low_allocation_storage_assignment_subranges_overlap(
                descriptor_set, candidate, candidate_unit, lease_assignment,
                source_unit, 1)) {
          continue;
        }
        overlaps = true;
        if (!loom_low_allocation_storage_assignment_subranges_equal(
                descriptor_set, candidate, candidate_unit, lease_assignment,
                source_unit, 1) ||
            state->identity_origins[candidate_identity_start +
                                    candidate_unit] !=
                state->identity_origins[lease_identity_start + source_unit]) {
          return false;
        }
      }
    }
    return overlaps;
  }

  const uint64_t lease_begin =
      (uint64_t)lease_assignment->location_base + record->unit_offset;
  const uint64_t lease_end = lease_begin + record->unit_count;
  const uint64_t candidate_begin = candidate->location_base;
  const uint64_t candidate_end = candidate_begin + candidate->location_count;
  const uint64_t overlap_begin = iree_max(lease_begin, candidate_begin);
  const uint64_t overlap_end = iree_min(lease_end, candidate_end);
  if (overlap_begin >= overlap_end) {
    return false;
  }
  for (uint64_t location = overlap_begin; location < overlap_end; ++location) {
    const uint32_t candidate_unit = (uint32_t)(location - candidate_begin);
    if (!loom_low_allocation_storage_lease_overlaps_unit_liveness(
            state->unit_liveness, lease, candidate, candidate_unit)) {
      continue;
    }
    const uint32_t lease_unit =
        record->unit_offset + (uint32_t)(location - lease_begin);
    if (state->identity_origins[candidate_identity_start + candidate_unit] !=
        state->identity_origins[lease_identity_start + lease_unit]) {
      return false;
    }
  }
  return true;
}

static bool loom_low_allocation_storage_lease_value_is_ignored(
    const loom_low_allocation_storage_lease_t* lease,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count) {
  if (ignored_value_ids == NULL) {
    return false;
  }
  for (uint16_t i = 0; i < ignored_value_count; ++i) {
    if (lease->value_id == ignored_value_ids[i]) {
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_try_packet_at_program_point(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_liveness_analysis_t* liveness, uint32_t program_point,
    iree_host_size_t* out_packet_index, uint32_t* out_node_index,
    uint32_t* out_block_index, uint32_t* out_scheduled_ordinal) {
  *out_packet_index = LOOM_LOW_STORAGE_LEASE_PACKET_NONE;
  *out_node_index = LOOM_LOW_STORAGE_LEASE_NODE_NONE;
  *out_block_index = UINT32_MAX;
  *out_scheduled_ordinal = LOOM_LOW_STORAGE_LEASE_ORDINAL_NONE;

  const loom_low_schedule_table_t* schedule = state->lease_table->schedule;
  // Liveness retains ordered, disjoint block extents. Locate the first extent
  // ending after the point, then exclude the gap before its start. This also
  // excludes empty extents without scanning preceding blocks for every lease.
  iree_host_size_t lower = 0;
  iree_host_size_t upper = liveness->block_count;
  while (lower < upper) {
    const iree_host_size_t middle = lower + (upper - lower) / 2;
    if (liveness->blocks[middle].end_point <= program_point) {
      lower = middle + 1;
    } else {
      upper = middle;
    }
  }
  if (lower < liveness->block_count) {
    const loom_liveness_block_info_t* block_info = &liveness->blocks[lower];
    if (program_point < block_info->start_point) {
      return false;
    }
    const uint32_t scheduled_ordinal = program_point - block_info->start_point;
    if (scheduled_ordinal >= schedule->blocks[lower].scheduled_node_count) {
      return false;
    }
    const uint64_t packet_index =
        (uint64_t)schedule->blocks[lower].scheduled_node_start +
        scheduled_ordinal;
    if (packet_index >= schedule->scheduled_node_count ||
        packet_index > IREE_HOST_SIZE_MAX) {
      return false;
    }
    const uint32_t node_index =
        schedule->scheduled_node_indices[(iree_host_size_t)packet_index];
    if (node_index >= schedule->node_count) {
      return false;
    }
    *out_packet_index = (iree_host_size_t)packet_index;
    *out_node_index = node_index;
    *out_block_index = (uint32_t)lower;
    *out_scheduled_ordinal = scheduled_ordinal;
    return true;
  }
  return false;
}

static iree_status_t loom_low_allocation_packet_at_program_point(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_liveness_analysis_t* liveness, uint32_t program_point,
    iree_host_size_t* out_packet_index, uint32_t* out_node_index,
    uint32_t* out_block_index, uint32_t* out_scheduled_ordinal) {
  if (loom_low_allocation_try_packet_at_program_point(
          state, liveness, program_point, out_packet_index, out_node_index,
          out_block_index, out_scheduled_ordinal)) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "storage release insertion point does not map to a "
                          "scheduled packet");
}

static bool loom_low_allocation_try_packet_for_node(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_liveness_analysis_t* liveness, uint32_t node_index,
    iree_host_size_t* out_packet_index, uint32_t* out_block_index,
    uint32_t* out_scheduled_ordinal, uint32_t* out_program_point) {
  const loom_low_schedule_table_t* schedule = state->lease_table->schedule;
  if (node_index >= schedule->node_count) {
    return false;
  }
  const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
  const uint32_t block_index = node->block_index;
  if (block_index >= schedule->block_count ||
      block_index >= liveness->block_count) {
    return false;
  }
  const loom_low_schedule_block_t* schedule_block =
      &schedule->blocks[block_index];
  const loom_liveness_block_info_t* liveness_block =
      &liveness->blocks[block_index];
  if (node->scheduled_ordinal >= schedule_block->scheduled_node_count ||
      node->scheduled_ordinal > UINT32_MAX - liveness_block->start_point) {
    return false;
  }
  const iree_host_size_t packet_index =
      (iree_host_size_t)schedule_block->scheduled_node_start +
      node->scheduled_ordinal;
  if (packet_index >= schedule->scheduled_node_count ||
      schedule->scheduled_node_indices[packet_index] != node_index) {
    return false;
  }
  *out_packet_index = packet_index;
  *out_block_index = block_index;
  *out_scheduled_ordinal = node->scheduled_ordinal;
  *out_program_point = liveness_block->start_point + node->scheduled_ordinal;
  return true;
}

static bool loom_low_allocation_try_candidate_definition_node(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_assignment_t* candidate,
    uint32_t* out_node_index) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_value_ordinal_for_liveness_value(
          state->value_domain, liveness, candidate->value_id, &value_ordinal)) {
    return false;
  }
  const uint32_t node_index = loom_low_schedule_value_producer_node(
      state->lease_table->schedule, value_ordinal);
  if (node_index == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return false;
  }
  *out_node_index = node_index;
  return true;
}

static iree_status_t loom_low_allocation_release_packet_for_candidate(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_assignment_t* candidate,
    iree_host_size_t* out_packet_index, uint32_t* out_node_index,
    uint32_t* out_block_index, uint32_t* out_scheduled_ordinal,
    uint32_t* out_program_point) {
  uint32_t definition_node_index = LOOM_LOW_STORAGE_LEASE_NODE_NONE;
  if (loom_low_allocation_try_candidate_definition_node(
          state, liveness, candidate, &definition_node_index)) {
    if (loom_low_allocation_try_packet_for_node(
            state, liveness, definition_node_index, out_packet_index,
            out_block_index, out_scheduled_ordinal, out_program_point)) {
      *out_node_index = definition_node_index;
      return iree_ok_status();
    }
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "storage release definition node does not map to a scheduled packet");
  }
  IREE_RETURN_IF_ERROR(loom_low_allocation_packet_at_program_point(
      state, liveness, candidate->start_point, out_packet_index, out_node_index,
      out_block_index, out_scheduled_ordinal));
  *out_program_point = candidate->start_point;
  return iree_ok_status();
}

static bool loom_low_allocation_storage_lease_can_release_before(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_storage_lease_t* lease,
    const loom_low_allocation_assignment_t* candidate) {
  const loom_low_storage_lease_record_t* record =
      &state->lease_table->records[lease->lease_record_index];
  if (record->release_scope !=
      LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS) {
    return false;
  }
  if (candidate->start_point <= lease->start_point) {
    return false;
  }
  iree_host_size_t packet_index = LOOM_LOW_STORAGE_LEASE_PACKET_NONE;
  uint32_t node_index = LOOM_LOW_STORAGE_LEASE_NODE_NONE;
  uint32_t block_index = UINT32_MAX;
  uint32_t scheduled_ordinal = LOOM_LOW_STORAGE_LEASE_ORDINAL_NONE;
  uint32_t program_point = UINT32_MAX;
  if (loom_low_allocation_try_candidate_definition_node(
          state, liveness, candidate, &node_index)) {
    return loom_low_allocation_try_packet_for_node(
               state, liveness, node_index, &packet_index, &block_index,
               &scheduled_ordinal, &program_point) &&
           packet_index != LOOM_LOW_STORAGE_LEASE_PACKET_NONE &&
           packet_index > record->packet_index;
  }
  if (!loom_low_allocation_try_packet_at_program_point(
          state, liveness, candidate->start_point, &packet_index, &node_index,
          &block_index, &scheduled_ordinal)) {
    return false;
  }
  return packet_index != LOOM_LOW_STORAGE_LEASE_PACKET_NONE &&
         packet_index > record->packet_index;
}

static bool loom_low_allocation_storage_release_policy_allows_record(
    loom_low_allocation_storage_release_policy_t policy,
    const loom_low_storage_lease_record_t* record) {
  switch (policy) {
    case LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN:
      return false;
    case LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FOR_PRESSURE:
      return iree_all_bits_set(
          record->flags, LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_FOR_PRESSURE);
    case LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED:
      return true;
  }
  return false;
}

static bool loom_low_allocation_storage_lease_scan_conflicts(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count,
    loom_low_allocation_storage_release_policy_t policy) {
  const uint32_t candidate_identity_start =
      loom_low_allocation_storage_identity_unit_start(state,
                                                      candidate->value_id);
  const iree_host_size_t record_count = state->lease_table->record_count;
  for (iree_host_size_t i = 0; i < record_count; ++i) {
    if (state->instance_written[i] == 0) {
      continue;
    }
    const loom_low_allocation_storage_lease_t* lease = &state->instances[i];
    if (loom_low_allocation_storage_lease_value_is_ignored(
            lease, ignored_value_ids, ignored_value_count)) {
      continue;
    }
    if (loom_low_allocation_storage_lease_instance_conflicts(
            state, descriptor_set, (uint32_t)i, lease, candidate)) {
      if (loom_low_allocation_storage_lease_overlap_preserves_identity(
              state, descriptor_set, (uint32_t)i, lease, candidate,
              candidate_identity_start)) {
        continue;
      }
      const loom_low_storage_lease_record_t* record =
          &state->lease_table->records[i];
      if (policy != LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN &&
          loom_low_allocation_storage_release_policy_allows_record(policy,
                                                                   record) &&
          loom_low_allocation_storage_lease_can_release_before(
              state, liveness, lease, candidate)) {
        continue;
      }
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_storage_lease_index_conflicts(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count,
    loom_low_allocation_storage_release_policy_t policy) {
  const uint32_t candidate_identity_start =
      loom_low_allocation_storage_identity_unit_start(state,
                                                      candidate->value_id);
  loom_low_allocation_storage_lease_unit_query_t query;
  loom_low_allocation_storage_lease_unit_query_initialize(
      state->unit_index, descriptor_set, candidate->descriptor_reg_class_id,
      candidate->location_kind, candidate->location_base,
      candidate->location_count, (uint64_t)candidate->start_point + 1u,
      candidate->end_point,
      /*selection=*/NULL, &query);
  uint32_t storage_lease_index = 0;
  while (loom_low_allocation_storage_lease_unit_query_next(
      &query, &storage_lease_index)) {
    const loom_low_allocation_storage_lease_t* lease =
        &state->instances[storage_lease_index];
    if (!loom_low_allocation_storage_lease_instance_conflicts(
            state, descriptor_set, storage_lease_index, lease, candidate)) {
      continue;
    }
    if (loom_low_allocation_storage_lease_value_is_ignored(
            lease, ignored_value_ids, ignored_value_count)) {
      continue;
    }
    if (loom_low_allocation_storage_lease_overlap_preserves_identity(
            state, descriptor_set, storage_lease_index, lease, candidate,
            candidate_identity_start)) {
      continue;
    }
    const loom_low_storage_lease_record_t* record =
        &state->lease_table->records[storage_lease_index];
    if (policy != LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN &&
        loom_low_allocation_storage_release_policy_allows_record(policy,
                                                                 record) &&
        loom_low_allocation_storage_lease_can_release_before(
            state, liveness, lease, candidate)) {
      continue;
    }
    return true;
  }
  return false;
}

// Assignment capacity and ABI-fixed windows bound distinct physical units.
// Aliasing classes may overcount this bound; unbounded classes can use a
// distinct location for every materialized unit. Neither case needs a scan of
// assignments or an additional per-lease class-membership table.
static iree_host_size_t
loom_low_allocation_storage_lease_distinct_unit_capacity(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_host_size_t lease_unit_capacity) {
  iree_host_size_t capacity = 0;
  for (iree_host_size_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    const loom_low_reg_class_t* reg_class = &descriptor_set->reg_classes[i];
    if (reg_class->allocatable_count == 0) {
      return lease_unit_capacity;
    }
    const uint32_t class_capacity = (uint32_t)reg_class->allocatable_count +
                                    reg_class->fixed_location_count;
    if (class_capacity >= lease_unit_capacity - capacity) {
      return lease_unit_capacity;
    }
    capacity += class_capacity;
  }
  return capacity;
}

enum {
  // Small lease tables are cheaper to probe through the exact temporal index
  // than to retain and update a second ordered frontier.
  LOOM_LOW_ALLOCATION_STORAGE_LEASE_AVAILABILITY_MIN_UNITS = 32,
};

static bool loom_low_allocation_storage_lease_expiration_less(
    const loom_low_allocation_storage_lease_state_t* state, uint32_t lhs,
    uint32_t rhs) {
  const uint32_t lhs_end = state->instances[lhs].end_point;
  const uint32_t rhs_end = state->instances[rhs].end_point;
  return lhs_end < rhs_end || (lhs_end == rhs_end && lhs < rhs);
}

static void loom_low_allocation_storage_lease_expiration_swap(
    loom_low_allocation_storage_lease_state_t* state, uint32_t lhs_position,
    uint32_t rhs_position) {
  uint32_t* heap = state->availability_expiration_heap;
  const uint32_t lhs = heap[lhs_position];
  const uint32_t rhs = heap[rhs_position];
  heap[lhs_position] = rhs;
  heap[rhs_position] = lhs;
  state->availability_expiration_positions[lhs] = rhs_position;
  state->availability_expiration_positions[rhs] = lhs_position;
}

static void loom_low_allocation_storage_lease_expiration_sift_up(
    loom_low_allocation_storage_lease_state_t* state, uint32_t position) {
  while (position != 0) {
    const uint32_t parent = (position - 1u) / 2u;
    if (!loom_low_allocation_storage_lease_expiration_less(
            state, state->availability_expiration_heap[position],
            state->availability_expiration_heap[parent])) {
      return;
    }
    loom_low_allocation_storage_lease_expiration_swap(state, position, parent);
    position = parent;
  }
}

static void loom_low_allocation_storage_lease_expiration_sift_down(
    loom_low_allocation_storage_lease_state_t* state, uint32_t position) {
  while (true) {
    const uint32_t left = position * 2u + 1u;
    if (left >= state->availability_expiration_count) {
      return;
    }
    const uint32_t right = left + 1u;
    uint32_t selected = left;
    if (right < state->availability_expiration_count &&
        loom_low_allocation_storage_lease_expiration_less(
            state, state->availability_expiration_heap[right],
            state->availability_expiration_heap[left])) {
      selected = right;
    }
    if (!loom_low_allocation_storage_lease_expiration_less(
            state, state->availability_expiration_heap[selected],
            state->availability_expiration_heap[position])) {
      return;
    }
    loom_low_allocation_storage_lease_expiration_swap(state, position,
                                                      selected);
    position = selected;
  }
}

static void loom_low_allocation_storage_lease_expiration_insert(
    loom_low_allocation_storage_lease_state_t* state,
    uint32_t storage_lease_index) {
  IREE_ASSERT_EQ(state->availability_expiration_positions[storage_lease_index],
                 UINT32_MAX);
  const uint32_t position = state->availability_expiration_count++;
  state->availability_expiration_heap[position] = storage_lease_index;
  state->availability_expiration_positions[storage_lease_index] = position;
  loom_low_allocation_storage_lease_expiration_sift_up(state, position);
}

static void loom_low_allocation_storage_lease_expiration_remove(
    loom_low_allocation_storage_lease_state_t* state,
    uint32_t storage_lease_index) {
  const uint32_t position =
      state->availability_expiration_positions[storage_lease_index];
  IREE_ASSERT_NE(position, UINT32_MAX);
  state->availability_expiration_positions[storage_lease_index] = UINT32_MAX;
  --state->availability_expiration_count;
  if (position == state->availability_expiration_count) {
    return;
  }
  const uint32_t replacement =
      state->availability_expiration_heap[state->availability_expiration_count];
  state->availability_expiration_heap[position] = replacement;
  state->availability_expiration_positions[replacement] = position;
  if (position != 0 &&
      loom_low_allocation_storage_lease_expiration_less(
          state, replacement,
          state->availability_expiration_heap[(position - 1u) / 2u])) {
    loom_low_allocation_storage_lease_expiration_sift_up(state, position);
  } else {
    loom_low_allocation_storage_lease_expiration_sift_down(state, position);
  }
}

iree_status_t loom_low_allocation_storage_lease_state_initialize(
    const loom_low_storage_lease_table_t* lease_table,
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_storage_identity_t* storage_identity,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    iree_arena_allocator_t* arena,
    loom_low_allocation_storage_lease_state_t* out_state) {
  IREE_ASSERT_ARGUMENT(lease_table);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(function_op);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(storage_identity);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_state);
  *out_state = (loom_low_allocation_storage_lease_state_t){0};
  out_state->lease_table = lease_table;
  out_state->value_domain = value_domain;
  out_state->unit_liveness = unit_liveness;
  IREE_RETURN_IF_ERROR(loom_low_allocation_validate_storage_lease_table(
      lease_table, module, function_op));
  if (lease_table->record_count == 0) {
    return iree_ok_status();
  }
  out_state->identity_origins = storage_identity->origins;
  const loom_low_schedule_table_t* schedule = lease_table->schedule;
  IREE_ASSERT(schedule != NULL);
  IREE_ASSERT(schedule->value_count == 0 ||
              schedule->value_producer_nodes != NULL);

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, lease_table->record_count, sizeof(*out_state->instances),
      (void**)&out_state->instances));
  memset(out_state->instances, 0,
         lease_table->record_count * sizeof(*out_state->instances));
  for (iree_host_size_t i = 0; i < lease_table->record_count; ++i) {
    out_state->instances[i].release_action_index =
        LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE;
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, lease_table->record_count, sizeof(*out_state->release_actions),
      (void**)&out_state->release_actions));
  memset(out_state->release_actions, 0,
         lease_table->record_count * sizeof(*out_state->release_actions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, lease_table->record_count, sizeof(*out_state->instance_written),
      (void**)&out_state->instance_written));
  memset(out_state->instance_written, 0,
         lease_table->record_count * sizeof(*out_state->instance_written));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, lease_table->record_count, sizeof(*out_state->next_record_indices),
      (void**)&out_state->next_record_indices));
  for (iree_host_size_t i = 0; i < lease_table->record_count; ++i) {
    out_state->next_record_indices[i] = UINT32_MAX;
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, liveness->value_count,
      sizeof(*out_state->record_heads_by_value_ordinal),
      (void**)&out_state->record_heads_by_value_ordinal));
  for (iree_host_size_t i = 0; i < liveness->value_count; ++i) {
    out_state->record_heads_by_value_ordinal[i] = UINT32_MAX;
  }
  iree_host_size_t lease_unit_capacity = 0;
  for (iree_host_size_t i = 0; i < lease_table->record_count; ++i) {
    const loom_low_storage_lease_record_t* record = &lease_table->records[i];
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_allocation_storage_lease_value_id(
        lease_table, record, &value_id));
    loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
    if (!loom_low_allocation_value_ordinal_for_liveness_value(
            value_domain, liveness, value_id, &value_ordinal)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "storage lease references value outside allocation liveness");
    }
    out_state->next_record_indices[i] =
        out_state->record_heads_by_value_ordinal[value_ordinal];
    out_state->record_heads_by_value_ordinal[value_ordinal] = (uint32_t)i;
    if (iree_all_bits_set(record->flags,
                          LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_FOR_PRESSURE)) {
      ++out_state->pressure_release_record_count;
    }
    if (!iree_host_size_checked_add(lease_unit_capacity, record->unit_count,
                                    &lease_unit_capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "storage lease unit count exceeds host size");
    }
  }
  if (lease_unit_capacity == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      arena, sizeof(*out_state->unit_index), (void**)&out_state->unit_index));
  IREE_RETURN_IF_ERROR(loom_low_allocation_storage_lease_unit_index_initialize(
      out_state->unit_index, out_state->instances, lease_table->record_count,
      lease_unit_capacity,
      loom_low_allocation_storage_lease_distinct_unit_capacity(
          schedule->target.descriptor_set, lease_unit_capacity),
      arena));
  if (lease_unit_capacity <
      LOOM_LOW_ALLOCATION_STORAGE_LEASE_AVAILABILITY_MIN_UNITS) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, lease_table->record_count,
      sizeof(*out_state->availability_expiration_heap),
      (void**)&out_state->availability_expiration_heap));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, lease_table->record_count,
      sizeof(*out_state->availability_expiration_positions),
      (void**)&out_state->availability_expiration_positions));
  for (iree_host_size_t i = 0; i < lease_table->record_count; ++i) {
    out_state->availability_expiration_positions[i] = UINT32_MAX;
  }
  return iree_ok_status();
}

bool loom_low_allocation_storage_lease_state_conflicts(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count,
    loom_low_allocation_storage_release_policy_t policy) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(candidate);
  if (state->lease_table == NULL || state->lease_table->record_count == 0) {
    return false;
  }
  if (!loom_low_allocation_location_kind_is_register_like(
          candidate->location_kind)) {
    return false;
  }
  if (loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, candidate)) {
    return loom_low_allocation_storage_lease_scan_conflicts(
        state, descriptor_set, liveness, candidate, ignored_value_ids,
        ignored_value_count, policy);
  }
  if (!loom_low_allocation_storage_lease_unit_index_is_enabled(
          state->unit_index)) {
    return loom_low_allocation_storage_lease_scan_conflicts(
        state, descriptor_set, liveness, candidate, ignored_value_ids,
        ignored_value_count, policy);
  }
  return loom_low_allocation_storage_lease_index_conflicts(
      state, descriptor_set, liveness, candidate, ignored_value_ids,
      ignored_value_count, policy);
}

static void loom_low_allocation_storage_lease_state_advance_availability(
    loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set, uint32_t start_point) {
  IREE_ASSERT_GE(start_point, state->availability_start_point);
  state->availability_start_point = start_point;
  while (state->availability_expiration_count != 0) {
    const uint32_t storage_lease_index = state->availability_expiration_heap[0];
    if (state->instances[storage_lease_index].end_point > start_point) {
      break;
    }
    loom_low_allocation_storage_lease_expiration_remove(state,
                                                        storage_lease_index);
    loom_low_allocation_storage_lease_unit_index_refresh_availability(
        state->unit_index, descriptor_set, storage_lease_index, start_point);
  }
}

static bool loom_low_allocation_storage_lease_candidate_is_continuous(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_low_allocation_assignment_t* candidate) {
  if (candidate->liveness_segments.count == 0) {
    return true;
  }
  if (candidate->liveness_segments.count != 1) {
    return false;
  }
  const loom_liveness_segment_t* segment =
      &state->unit_liveness->storage_segments
           .entries[candidate->liveness_segments.start];
  return segment->start_point == candidate->start_point &&
         segment->end_point == candidate->end_point;
}

static bool loom_low_allocation_storage_lease_candidate_forwards_identity(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_low_allocation_assignment_t* candidate) {
  const uint32_t unit_start = loom_low_allocation_storage_identity_unit_start(
      state, candidate->value_id);
  if (unit_start == UINT32_MAX) {
    return false;
  }
  return state->identity_origins[unit_start] != unit_start;
}

bool loom_low_allocation_storage_lease_state_can_order_candidate(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* candidate,
    loom_low_allocation_storage_release_policy_t policy) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(candidate);
  return state->availability_expiration_heap != NULL &&
         policy != LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED &&
         candidate->descriptor_reg_class_id < descriptor_set->reg_class_count &&
         loom_low_allocation_assignment_is_register_like(candidate) &&
         candidate->unit_count == 1 && candidate->location_count == 1 &&
         candidate->start_point < candidate->end_point &&
         loom_low_allocation_live_range_assignment_unit_end_point(
             state->unit_liveness->end_points,
             state->unit_liveness->point_count, candidate,
             0) == candidate->end_point &&
         !loom_low_allocation_storage_lease_candidate_forwards_identity(
             state, candidate) &&
         loom_low_allocation_storage_lease_candidate_is_continuous(state,
                                                                   candidate) &&
         !iree_any_bit_set(
             candidate->flags,
             LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS) &&
         !loom_low_allocation_storage_assignment_uses_explicit_physical_register(
             descriptor_set, candidate);
}

static loom_low_allocation_storage_lease_conflict_class_t
loom_low_allocation_storage_lease_conflict_class_for_policy(
    loom_low_allocation_storage_release_policy_t policy) {
  IREE_ASSERT_NE(policy, LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED);
  return policy == LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN
             ? LOOM_LOW_ALLOCATION_STORAGE_LEASE_CONFLICT_ALL
             : LOOM_LOW_ALLOCATION_STORAGE_LEASE_CONFLICT_NON_PRESSURE;
}

bool loom_low_allocation_storage_lease_state_find_next_available_location(
    loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* candidate,
    loom_low_allocation_storage_release_policy_t policy, uint32_t minimum_base,
    uint32_t maximum_base, uint32_t* out_base) {
  IREE_ASSERT_TRUE(loom_low_allocation_storage_lease_state_can_order_candidate(
      state, descriptor_set, candidate, policy));
  loom_low_allocation_storage_lease_state_advance_availability(
      state, descriptor_set, candidate->start_point);
  return loom_low_allocation_storage_lease_unit_index_find_next_available_location(
      state->unit_index, descriptor_set, candidate->descriptor_reg_class_id,
      candidate->location_kind, candidate->end_point,
      loom_low_allocation_storage_lease_conflict_class_for_policy(policy),
      minimum_base, maximum_base, out_base);
}

bool loom_low_allocation_storage_lease_state_find_previous_available_location(
    loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* candidate,
    loom_low_allocation_storage_release_policy_t policy, uint32_t minimum_base,
    uint32_t maximum_base, uint32_t* out_base) {
  IREE_ASSERT_TRUE(loom_low_allocation_storage_lease_state_can_order_candidate(
      state, descriptor_set, candidate, policy));
  loom_low_allocation_storage_lease_state_advance_availability(
      state, descriptor_set, candidate->start_point);
  return loom_low_allocation_storage_lease_unit_index_find_previous_available_location(
      state->unit_index, descriptor_set, candidate->descriptor_reg_class_id,
      candidate->location_kind, candidate->end_point,
      loom_low_allocation_storage_lease_conflict_class_for_policy(policy),
      minimum_base, maximum_base, out_base);
}

bool loom_low_allocation_storage_lease_state_value_has_records(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_liveness_analysis_t* liveness, loom_value_id_t value_id) {
  if (!state || state->record_heads_by_value_ordinal == NULL) {
    return false;
  }
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_value_ordinal_for_liveness_value(
          state->value_domain, liveness, value_id, &value_ordinal)) {
    return false;
  }
  return state->record_heads_by_value_ordinal[value_ordinal] != UINT32_MAX;
}

static iree_status_t
loom_low_allocation_storage_lease_state_record_release_action(
    loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_assignment_t* candidate,
    uint32_t lease_record_index) {
  loom_low_allocation_storage_lease_t* lease =
      &state->instances[lease_record_index];
  const loom_low_storage_lease_record_t* record =
      &state->lease_table->records[lease_record_index];
  if (!loom_low_allocation_storage_lease_can_release_before(state, liveness,
                                                            lease, candidate)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "storage lease conflict has no legal release before assignment");
  }

  iree_host_size_t insertion_packet_index = LOOM_LOW_STORAGE_LEASE_PACKET_NONE;
  uint32_t insertion_node_index = LOOM_LOW_STORAGE_LEASE_NODE_NONE;
  uint32_t block_index = UINT32_MAX;
  uint32_t scheduled_ordinal = LOOM_LOW_STORAGE_LEASE_ORDINAL_NONE;
  uint32_t release_program_point = UINT32_MAX;
  IREE_RETURN_IF_ERROR(loom_low_allocation_release_packet_for_candidate(
      state, liveness, candidate, &insertion_packet_index,
      &insertion_node_index, &block_index, &scheduled_ordinal,
      &release_program_point));

  // Aggregate reservations can be assigned before the scalar writers that
  // populate them. Retain the earliest conflicting write even when a later
  // reservation already requested release of this lease.
  uint32_t release_action_index = lease->release_action_index;
  if (release_action_index == LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE) {
    IREE_ASSERT_LT(state->release_action_count,
                   state->lease_table->record_count);
    release_action_index = (uint32_t)state->release_action_count++;
  }
  state->release_actions[release_action_index] =
      (loom_low_storage_release_action_t){
          .insertion_packet_index = insertion_packet_index,
          .insertion_node_index = insertion_node_index,
          .block_index = block_index,
          .scheduled_ordinal = scheduled_ordinal,
          .release_class_id = record->release_class_id,
          .release_class_name = record->release_class_name,
          .release_action_id = record->release_action_id,
          .release_action_name = record->release_action_name,
          .release_reason_id = record->release_reason_id,
          .release_reason_name = record->release_reason_name,
          .required_progress = 1,
          .lease_record_index = lease_record_index,
      };
  lease->release_action_index = release_action_index;
  lease->end_point = release_program_point;
  loom_low_allocation_storage_lease_unit_index_update(state->unit_index,
                                                      lease_record_index);
  if (state->availability_expiration_heap != NULL) {
    const uint32_t position =
        state->availability_expiration_positions[lease_record_index];
    if (position != UINT32_MAX) {
      if (lease->end_point <= state->availability_start_point) {
        loom_low_allocation_storage_lease_expiration_remove(state,
                                                            lease_record_index);
        loom_low_allocation_storage_lease_unit_index_refresh_availability(
            state->unit_index, descriptor_set, lease_record_index,
            state->availability_start_point);
      } else {
        loom_low_allocation_storage_lease_expiration_sift_up(state, position);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_storage_lease_state_scan_release_actions(
    loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count) {
  const uint32_t candidate_identity_start =
      loom_low_allocation_storage_identity_unit_start(state,
                                                      candidate->value_id);
  const iree_host_size_t record_count = state->lease_table->record_count;
  for (iree_host_size_t i = 0; i < record_count; ++i) {
    if (state->instance_written[i] == 0) {
      continue;
    }
    loom_low_allocation_storage_lease_t* lease = &state->instances[i];
    if (loom_low_allocation_storage_lease_value_is_ignored(
            lease, ignored_value_ids, ignored_value_count)) {
      continue;
    }
    if (!loom_low_allocation_storage_lease_instance_conflicts(
            state, descriptor_set, (uint32_t)i, lease, candidate)) {
      continue;
    }
    if (loom_low_allocation_storage_lease_overlap_preserves_identity(
            state, descriptor_set, (uint32_t)i, lease, candidate,
            candidate_identity_start)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_storage_lease_state_record_release_action(
            state, descriptor_set, liveness, candidate, (uint32_t)i));
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_storage_lease_state_record_release_actions(
    loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(candidate);
  if (state->instances == NULL ||
      !loom_low_allocation_location_kind_is_register_like(
          candidate->location_kind)) {
    return iree_ok_status();
  }
  if (loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, candidate)) {
    return loom_low_allocation_storage_lease_state_scan_release_actions(
        state, descriptor_set, liveness, candidate, ignored_value_ids,
        ignored_value_count);
  }
  if (!loom_low_allocation_storage_lease_unit_index_is_enabled(
          state->unit_index)) {
    return loom_low_allocation_storage_lease_state_scan_release_actions(
        state, descriptor_set, liveness, candidate, ignored_value_ids,
        ignored_value_count);
  }

  const uint32_t candidate_identity_start =
      loom_low_allocation_storage_identity_unit_start(state,
                                                      candidate->value_id);
  loom_low_allocation_storage_lease_unit_query_t query;
  loom_low_allocation_storage_lease_unit_query_initialize(
      state->unit_index, descriptor_set, candidate->descriptor_reg_class_id,
      candidate->location_kind, candidate->location_base,
      candidate->location_count, (uint64_t)candidate->start_point + 1u,
      candidate->end_point,
      /*selection=*/NULL, &query);
  uint32_t storage_lease_index = 0;
  while (loom_low_allocation_storage_lease_unit_query_next(
      &query, &storage_lease_index)) {
    loom_low_allocation_storage_lease_t* lease =
        &state->instances[storage_lease_index];
    if (!loom_low_allocation_storage_lease_instance_conflicts(
            state, descriptor_set, storage_lease_index, lease, candidate)) {
      continue;
    }
    if (loom_low_allocation_storage_lease_value_is_ignored(
            lease, ignored_value_ids, ignored_value_count)) {
      continue;
    }
    if (loom_low_allocation_storage_lease_overlap_preserves_identity(
            state, descriptor_set, storage_lease_index, lease, candidate,
            candidate_identity_start)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_storage_lease_state_record_release_action(
            state, descriptor_set, liveness, candidate, storage_lease_index));
  }
  return iree_ok_status();
}

void loom_low_allocation_storage_lease_state_record_assignment(
    loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_assignment_t* assignments,
    uint32_t assignment_index, loom_value_ordinal_t value_ordinal) {
  if (state->record_heads_by_value_ordinal == NULL) {
    return;
  }
  if (state->assignments == NULL) {
    state->assignments = assignments;
  } else {
    IREE_ASSERT_EQ(state->assignments, assignments);
  }
  const loom_low_allocation_assignment_t* assignment =
      &assignments[assignment_index];
  uint32_t lease_record_index =
      state->record_heads_by_value_ordinal[value_ordinal];
  while (lease_record_index != UINT32_MAX) {
    const loom_low_storage_lease_record_t* record =
        &state->lease_table->records[lease_record_index];
    const loom_liveness_block_info_t* block_info =
        &liveness->blocks[record->block_index];
    const uint32_t end_point =
        iree_any_bit_set(record->flags,
                         LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_BEFORE_BOUNDARY)
            ? block_info->end_point
            : liveness->blocks[liveness->block_count - 1u].end_point;
    // The temporal index reserves nodes for exactly one insertion per lease.
    IREE_ASSERT_EQ(state->instance_written[lease_record_index], 0u);
    state->instances[lease_record_index] =
        (loom_low_allocation_storage_lease_t){
            .lease_record_index = lease_record_index,
            .assignment_index = assignment_index,
            .value_id = assignment->value_id,
            .start_point = block_info->start_point + record->scheduled_ordinal,
            .end_point = end_point,
            .release_action_index = LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE,
            .descriptor_reg_class_id = assignment->descriptor_reg_class_id,
            .location_kind = assignment->location_kind,
            .location_base = assignment->location_base + record->unit_offset,
            .location_count = record->unit_count,
        };
    state->instance_written[lease_record_index] = 1;
    ++state->instance_count;
    loom_low_allocation_storage_lease_unit_index_insert(
        state->unit_index, descriptor_set, lease_record_index, record->flags);
    if (state->availability_expiration_heap != NULL &&
        !loom_low_reg_class_uses_explicit_physical_registers(
            &descriptor_set
                 ->reg_classes[assignment->descriptor_reg_class_id])) {
      if (end_point > state->availability_start_point) {
        loom_low_allocation_storage_lease_expiration_insert(state,
                                                            lease_record_index);
      }
      loom_low_allocation_storage_lease_unit_index_refresh_availability(
          state->unit_index, descriptor_set, lease_record_index,
          state->availability_start_point);
    }
    lease_record_index = state->next_record_indices[lease_record_index];
  }
}

iree_status_t loom_low_allocation_storage_lease_state_finalize(
    const loom_low_allocation_storage_lease_state_t* state) {
  IREE_ASSERT_ARGUMENT(state);
  if (state->lease_table == NULL || state->lease_table->record_count == 0) {
    return iree_ok_status();
  }
  if (state->instance_count != state->lease_table->record_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation materialized %" PRIhsz
        " storage lease instance(s) for %" PRIhsz " storage lease record(s)",
        state->instance_count, state->lease_table->record_count);
  }
  for (iree_host_size_t i = 0; i < state->lease_table->record_count; ++i) {
    if (state->instance_written[i] == 0) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "storage lease record %" PRIhsz
                              " has no assigned physical storage",
                              i);
    }
  }
  return iree_ok_status();
}
