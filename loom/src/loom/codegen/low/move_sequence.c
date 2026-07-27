// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/move_sequence.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/allocation/move_topology.h"
#include "loom/ops/low/ops.h"

bool loom_low_move_locations_equal(const loom_low_move_location_t* lhs,
                                   const loom_low_move_location_t* rhs) {
  return lhs->location_kind == rhs->location_kind &&
         lhs->location == rhs->location &&
         lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id;
}

static bool loom_low_move_locations_share_target_storage(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_move_location_t* lhs, const loom_low_move_location_t* rhs) {
  if (lhs->location_kind != rhs->location_kind ||
      lhs->location != rhs->location) {
    return false;
  }
  if (descriptor_set == NULL) {
    return lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id;
  }
  return loom_low_allocation_storage_reg_classes_share(
      descriptor_set, lhs->descriptor_reg_class_id,
      rhs->descriptor_reg_class_id);
}

static bool loom_low_move_locations_share_storage_class(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_move_location_t* lhs, const loom_low_move_location_t* rhs) {
  if (lhs->location_kind != rhs->location_kind) {
    return false;
  }
  const bool classes_share_storage =
      descriptor_set == NULL
          ? lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id
          : loom_low_allocation_storage_reg_classes_share(
                descriptor_set, lhs->descriptor_reg_class_id,
                rhs->descriptor_reg_class_id);
  return classes_share_storage &&
         loom_liveness_value_class_equal(lhs->value_class, rhs->value_class);
}

#define LOOM_LOW_MOVE_SEQUENCE_INDEX_NONE IREE_HOST_SIZE_MAX

typedef enum loom_low_move_sequence_node_flag_bits_e {
  LOOM_LOW_MOVE_SEQUENCE_NODE_FLAG_ACTIVE = 1u << 0,
} loom_low_move_sequence_node_flag_bits_t;
typedef uint8_t loom_low_move_sequence_node_flags_t;

struct loom_low_move_sequence_node_t {
  // Active-state flags for the corresponding move row.
  loom_low_move_sequence_node_flags_t flags;
};

typedef enum loom_low_move_sequence_location_flag_bits_e {
  LOOM_LOW_MOVE_SEQUENCE_LOCATION_FLAG_OCCUPIED = 1u << 0,
  LOOM_LOW_MOVE_SEQUENCE_LOCATION_FLAG_TEMPORARY = 1u << 1,
} loom_low_move_sequence_location_flag_bits_t;
typedef uint8_t loom_low_move_sequence_location_flags_t;

struct loom_low_move_sequence_location_entry_t {
  // Target-visible unit keyed by this hash entry.
  loom_low_move_location_t location;
  // Move row that writes |location|, or INDEX_NONE when none does.
  iree_host_size_t destination_move_index;
  // Number of active moves that read |location|.
  iree_host_size_t source_use_count;
  // Occupancy and temporary-state flags.
  loom_low_move_sequence_location_flags_t flags;
};

void loom_low_move_sequence_scratch_initialize(
    iree_arena_allocator_t* arena,
    loom_low_move_sequence_scratch_t* out_scratch) {
  *out_scratch = (loom_low_move_sequence_scratch_t){
      .arena = arena,
  };
}

static iree_status_t loom_low_move_sequence_scratch_reserve_array(
    loom_low_move_sequence_scratch_t* scratch,
    iree_host_size_t minimum_capacity, iree_host_size_t element_size,
    iree_host_size_t* inout_capacity, void** inout_ptr) {
  if (*inout_capacity >= minimum_capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity = *inout_capacity ? *inout_capacity : 16;
  while (new_capacity < minimum_capacity) {
    if (new_capacity > IREE_HOST_SIZE_MAX / 2) {
      new_capacity = minimum_capacity;
      break;
    }
    new_capacity *= 2;
  }
  void* new_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch->arena, new_capacity,
                                                 element_size, &new_ptr));
  *inout_capacity = new_capacity;
  *inout_ptr = new_ptr;
  return iree_ok_status();
}

iree_status_t loom_low_move_sequence_scratch_reserve_moves(
    loom_low_move_sequence_scratch_t* scratch, iree_host_size_t move_count,
    loom_low_move_t** out_moves) {
  *out_moves = NULL;
  if (move_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_array(
      scratch, move_count, sizeof(*scratch->moves), &scratch->move_capacity,
      (void**)&scratch->moves));
  *out_moves = scratch->moves;
  return iree_ok_status();
}

iree_status_t loom_low_move_sequence_scratch_reserve_temporaries(
    loom_low_move_sequence_scratch_t* scratch,
    iree_host_size_t temporary_location_count,
    loom_low_move_location_t** out_temporary_locations) {
  *out_temporary_locations = NULL;
  if (temporary_location_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_array(
      scratch, temporary_location_count, sizeof(*scratch->temporary_locations),
      &scratch->temporary_location_capacity,
      (void**)&scratch->temporary_locations));
  *out_temporary_locations = scratch->temporary_locations;
  return iree_ok_status();
}

iree_status_t loom_low_move_location_from_assignment_unit(
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_index,
    loom_low_move_location_t* out_location) {
  *out_location = (loom_low_move_location_t){0};
  if (unit_index >= assignment->location_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "move location unit %" PRIu32
                            " exceeds assignment range of %" PRIu32,
                            unit_index, assignment->location_count);
  }
  if (assignment->location_base > UINT32_MAX - unit_index) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "move location unit exceeds uint32_t");
  }
  *out_location = (loom_low_move_location_t){
      .location_kind = assignment->location_kind,
      .value_class = assignment->value_class,
      .descriptor_reg_class_id = assignment->descriptor_reg_class_id,
      .location = assignment->location_base + unit_index,
  };
  return iree_ok_status();
}

static void loom_low_move_location_from_edge_copy_temporary(
    const loom_low_allocation_edge_copy_temporary_t* temporary,
    loom_low_move_location_t* out_location) {
  *out_location = (loom_low_move_location_t){
      .location_kind = temporary->location_kind,
      .value_class = temporary->value_class,
      .descriptor_reg_class_id = temporary->descriptor_reg_class_id,
      .location = temporary->location,
  };
}

static void loom_low_move_location_from_packet_move_temporary(
    const loom_low_allocation_packet_move_temporary_t* temporary,
    loom_low_move_location_t* out_location) {
  *out_location = (loom_low_move_location_t){
      .location_kind = temporary->location_kind,
      .value_class = temporary->value_class,
      .descriptor_reg_class_id = temporary->descriptor_reg_class_id,
      .location = temporary->location,
  };
}

static const loom_low_allocation_assignment_t*
loom_low_move_sequence_assignment(const loom_low_allocation_table_t* allocation,
                                  loom_value_id_t value_id) {
  return loom_low_allocation_map_active_value_assignment(allocation, value_id,
                                                         NULL);
}

static iree_status_t loom_low_move_sequence_require_same_storage_class(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs,
    iree_string_view_t structural_op) {
  if (!loom_low_allocation_storage_assignment_classes_share(descriptor_set, lhs,
                                                            rhs)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "%.*s structural move crosses allocation storage classes",
        (int)structural_op.size, structural_op.data);
  }
  return iree_ok_status();
}

iree_status_t loom_low_move_sequence_count_edge_copy_units(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_edge_copy_group_t* group,
    iree_host_size_t* out_move_count) {
  *out_move_count = 0;
  if (group->copy_start > allocation->edge_copy_count ||
      group->copy_count > allocation->edge_copy_count - group->copy_start) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "edge-copy group range is outside allocation");
  }
  for (uint32_t i = 0; i < group->copy_count; ++i) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &allocation->edge_copies[group->copy_start + i];
    if (edge_copy->source_assignment_index >= allocation->assignment_count ||
        edge_copy->destination_assignment_index >=
            allocation->assignment_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "edge-copy references an assignment outside allocation");
    }
    const loom_low_allocation_assignment_t* source_assignment =
        &allocation->assignments[edge_copy->source_assignment_index];
    const loom_low_allocation_assignment_t* destination_assignment =
        &allocation->assignments[edge_copy->destination_assignment_index];
    if (edge_copy->source_unit_offset > source_assignment->location_count ||
        edge_copy->unit_count >
            source_assignment->location_count - edge_copy->source_unit_offset ||
        edge_copy->destination_unit_offset >
            destination_assignment->location_count ||
        edge_copy->unit_count > destination_assignment->location_count -
                                    edge_copy->destination_unit_offset) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "edge-copy segment exceeds source or destination assignment");
    }
    if (edge_copy->unit_count > IREE_HOST_SIZE_MAX - *out_move_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "edge-copy unit move count exceeds host size");
    }
    *out_move_count += edge_copy->unit_count;
  }
  return iree_ok_status();
}

iree_status_t loom_low_move_sequence_populate_edge_copy_units(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_edge_copy_group_t* group, loom_low_move_t* moves,
    iree_host_size_t move_count) {
  iree_host_size_t expected_move_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_edge_copy_units(
      allocation, group, &expected_move_count));
  if (move_count != expected_move_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "edge-copy move storage has %zu entries but needs "
                            "%zu",
                            move_count, expected_move_count);
  }
  iree_host_size_t move_index = 0;
  for (uint32_t i = 0; i < group->copy_count; ++i) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &allocation->edge_copies[group->copy_start + i];
    const loom_low_allocation_assignment_t* source_assignment =
        &allocation->assignments[edge_copy->source_assignment_index];
    const loom_low_allocation_assignment_t* destination_assignment =
        &allocation->assignments[edge_copy->destination_assignment_index];
    for (uint32_t unit_index = 0; unit_index < edge_copy->unit_count;
         ++unit_index) {
      IREE_RETURN_IF_ERROR(loom_low_move_location_from_assignment_unit(
          destination_assignment,
          edge_copy->destination_unit_offset + unit_index,
          &moves[move_index].destination));
      IREE_RETURN_IF_ERROR(loom_low_move_location_from_assignment_unit(
          source_assignment, edge_copy->source_unit_offset + unit_index,
          &moves[move_index].source));
      ++move_index;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_move_sequence_populate_edge_copy_temporaries(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_edge_copy_group_t* group,
    loom_low_move_location_t* temporary_locations,
    iree_host_size_t temporary_location_count) {
  if (temporary_location_count != group->temporary_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "edge-copy temporary storage has %zu entries but needs %" PRIu32,
        temporary_location_count, group->temporary_count);
  }
  if (group->temporary_start > allocation->edge_copy_temporary_count ||
      group->temporary_count >
          allocation->edge_copy_temporary_count - group->temporary_start) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "edge-copy temporary group range is outside allocation");
  }
  for (uint32_t i = 0; i < group->temporary_count; ++i) {
    loom_low_move_location_from_edge_copy_temporary(
        &allocation->edge_copy_temporaries[group->temporary_start + i],
        &temporary_locations[i]);
  }
  return iree_ok_status();
}

iree_status_t loom_low_move_sequence_populate_packet_move_temporaries(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_packet_move_temporary_group_t* group,
    loom_low_move_location_t* temporary_locations,
    iree_host_size_t temporary_location_count) {
  if (temporary_location_count != group->temporary_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "packet move temporary storage has %zu entries but needs %" PRIu32,
        temporary_location_count, group->temporary_count);
  }
  if (group->temporary_start > allocation->packet_move_temporary_count ||
      group->temporary_count >
          allocation->packet_move_temporary_count - group->temporary_start) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "packet move temporary group range is outside allocation");
  }
  for (uint32_t i = 0; i < group->temporary_count; ++i) {
    loom_low_move_location_from_packet_move_temporary(
        &allocation->packet_move_temporaries[group->temporary_start + i],
        &temporary_locations[i]);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_move_sequence_slice_assignments(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op,
    const loom_low_allocation_assignment_t** out_source_assignment,
    const loom_low_allocation_assignment_t** out_result_assignment,
    uint32_t* out_source_offset) {
  *out_source_assignment = NULL;
  *out_result_assignment = NULL;
  *out_source_offset = 0;
  if (!loom_low_slice_isa(op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "expected low.slice");
  }
  const int64_t offset = loom_low_slice_offset(op);
  if (offset < 0 || offset > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low.slice offset is outside uint32_t range");
  }
  *out_source_assignment =
      loom_low_move_sequence_assignment(allocation, loom_low_slice_source(op));
  *out_result_assignment =
      loom_low_move_sequence_assignment(allocation, loom_low_slice_result(op));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_require_same_storage_class(
      allocation->target.descriptor_set, *out_source_assignment,
      *out_result_assignment, IREE_SV("low.slice")));
  const uint32_t source_offset = (uint32_t)offset;
  if (source_offset > (*out_source_assignment)->location_count ||
      (*out_result_assignment)->location_count >
          (*out_source_assignment)->location_count - source_offset) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low.slice structural move range exceeds source assignment");
  }
  *out_source_offset = source_offset;
  return iree_ok_status();
}

iree_status_t loom_low_move_sequence_count_slice_units(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op,
    iree_host_size_t* out_move_count) {
  *out_move_count = 0;
  const loom_low_allocation_assignment_t* source_assignment = NULL;
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  uint32_t source_offset = 0;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_slice_assignments(
      allocation, op, &source_assignment, &result_assignment, &source_offset));
  if (loom_low_allocation_storage_assignment_subranges_equal(
          allocation->target.descriptor_set, result_assignment, /*lhs_start=*/0,
          source_assignment, source_offset,
          result_assignment->location_count)) {
    return iree_ok_status();
  }
  *out_move_count = result_assignment->location_count;
  return iree_ok_status();
}

iree_status_t loom_low_move_sequence_populate_slice_units(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op,
    loom_low_move_t* moves, iree_host_size_t move_count) {
  const loom_low_allocation_assignment_t* source_assignment = NULL;
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  uint32_t source_offset = 0;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_slice_assignments(
      allocation, op, &source_assignment, &result_assignment, &source_offset));
  const bool coalesced = loom_low_allocation_storage_assignment_subranges_equal(
      allocation->target.descriptor_set, result_assignment, /*lhs_start=*/0,
      source_assignment, source_offset, result_assignment->location_count);
  const iree_host_size_t expected_move_count =
      coalesced ? 0 : result_assignment->location_count;
  if (move_count != expected_move_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low.slice move storage has %zu entries but needs "
                            "%zu",
                            move_count, expected_move_count);
  }
  if (move_count == 0) {
    return iree_ok_status();
  }
  for (uint32_t i = 0; i < result_assignment->location_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_move_location_from_assignment_unit(
        result_assignment, i, &moves[i].destination));
    IREE_RETURN_IF_ERROR(loom_low_move_location_from_assignment_unit(
        source_assignment, source_offset + i, &moves[i].source));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_move_sequence_concat_assignments(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op,
    const loom_low_allocation_assignment_t** out_result_assignment,
    bool* out_coalesced, iree_host_size_t* out_move_count) {
  *out_result_assignment = NULL;
  *out_coalesced = true;
  *out_move_count = 0;
  if (!loom_low_concat_isa(op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected low.concat");
  }
  if (!loom_low_allocation_move_topology_concat_requires_packet_materialization(
          allocation, op)) {
    return iree_ok_status();
  }
  *out_result_assignment =
      loom_low_move_sequence_assignment(allocation, loom_low_concat_result(op));
  uint32_t result_offset = 0;
  loom_value_slice_t sources = loom_low_concat_sources(op);
  for (uint16_t i = 0; i < sources.count; ++i) {
    const loom_low_allocation_assignment_t* source_assignment = NULL;
    source_assignment =
        loom_low_move_sequence_assignment(allocation, sources.values[i]);
    IREE_RETURN_IF_ERROR(loom_low_move_sequence_require_same_storage_class(
        allocation->target.descriptor_set, *out_result_assignment,
        source_assignment, IREE_SV("low.concat")));
    if (result_offset > (*out_result_assignment)->location_count ||
        source_assignment->location_count >
            (*out_result_assignment)->location_count - result_offset) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low.concat structural move source ranges exceed result assignment");
    }
    if (!loom_low_allocation_storage_assignment_subranges_equal(
            allocation->target.descriptor_set, *out_result_assignment,
            result_offset, source_assignment, /*rhs_start=*/0,
            source_assignment->location_count)) {
      *out_coalesced = false;
    }
    if (source_assignment->location_count >
        IREE_HOST_SIZE_MAX - *out_move_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low.concat move count exceeds host size");
    }
    *out_move_count += source_assignment->location_count;
    result_offset += source_assignment->location_count;
  }
  if (result_offset != (*out_result_assignment)->location_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low.concat structural move sources do not fill result assignment");
  }
  return iree_ok_status();
}

iree_status_t loom_low_move_sequence_count_concat_units(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op,
    iree_host_size_t* out_move_count) {
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  bool coalesced = false;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_concat_assignments(
      allocation, op, &result_assignment, &coalesced, out_move_count));
  if (coalesced) {
    *out_move_count = 0;
  }
  return iree_ok_status();
}

iree_status_t loom_low_move_sequence_populate_concat_units(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op,
    loom_low_move_t* moves, iree_host_size_t move_count) {
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  bool coalesced = false;
  iree_host_size_t expected_move_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_concat_assignments(
      allocation, op, &result_assignment, &coalesced, &expected_move_count));
  if (coalesced) {
    expected_move_count = 0;
  }
  if (move_count != expected_move_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low.concat move storage has %zu entries but needs %zu", move_count,
        expected_move_count);
  }
  if (move_count == 0) {
    return iree_ok_status();
  }
  uint32_t result_offset = 0;
  iree_host_size_t move_index = 0;
  loom_value_slice_t sources = loom_low_concat_sources(op);
  for (uint16_t i = 0; i < sources.count; ++i) {
    const loom_low_allocation_assignment_t* source_assignment = NULL;
    source_assignment =
        loom_low_move_sequence_assignment(allocation, sources.values[i]);
    for (uint32_t source_unit = 0;
         source_unit < source_assignment->location_count; ++source_unit) {
      IREE_RETURN_IF_ERROR(loom_low_move_location_from_assignment_unit(
          result_assignment, result_offset, &moves[move_index].destination));
      IREE_RETURN_IF_ERROR(loom_low_move_location_from_assignment_unit(
          source_assignment, source_unit, &moves[move_index].source));
      ++result_offset;
      ++move_index;
    }
  }
  return iree_ok_status();
}

typedef struct loom_low_move_sequence_state_t {
  // Scratch storage owned by the caller's arena.
  loom_low_move_sequence_scratch_t* scratch;
  // Options controlling temporary selection and move emission.
  const loom_low_move_sequence_options_t* options;
  // Number of active, non-identity moves in |scratch->moves|.
  iree_host_size_t move_count;
  // Number of moves that have not been emitted yet.
  iree_host_size_t active_count;
  // Power-of-two location-table entries used for this move set.
  iree_host_size_t location_entry_count;
  // Current read position in |scratch->ready_queue|.
  iree_host_size_t ready_head;
  // Current write position in |scratch->ready_queue|.
  iree_host_size_t ready_tail;
  // Cursor used to find the next active move when only cycles remain.
  iree_host_size_t active_cursor;
} loom_low_move_sequence_state_t;

static uint64_t loom_low_move_sequence_mix64(uint64_t value) {
  value ^= value >> 33;
  value *= UINT64_C(0xff51afd7ed558ccd);
  value ^= value >> 33;
  value *= UINT64_C(0xc4ceb9fe1a85ec53);
  value ^= value >> 33;
  return value;
}

static uint64_t loom_low_move_sequence_location_hash(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_move_location_t* location) {
  uint64_t value = (uint64_t)location->location;
  value ^= (uint64_t)location->location_kind << 32;
  value ^= (uint64_t)loom_low_reg_class_storage_key(
               descriptor_set, location->descriptor_reg_class_id)
           << 40;
  return loom_low_move_sequence_mix64(value);
}

static iree_status_t loom_low_move_sequence_next_power_of_two(
    iree_host_size_t minimum_capacity, iree_host_size_t* out_capacity) {
  iree_host_size_t capacity = 16;
  while (capacity < minimum_capacity) {
    if (capacity > IREE_HOST_SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "parallel move location table is too large");
    }
    capacity *= 2;
  }
  *out_capacity = capacity;
  return iree_ok_status();
}

static iree_status_t loom_low_move_sequence_reserve_solver_scratch(
    loom_low_move_sequence_scratch_t* scratch, iree_host_size_t move_count,
    iree_host_size_t temporary_location_count,
    iree_host_size_t* out_location_entry_count) {
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_array(
      scratch, move_count, sizeof(*scratch->nodes), &scratch->node_capacity,
      (void**)&scratch->nodes));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_array(
      scratch, move_count, sizeof(*scratch->ready_queue),
      &scratch->ready_queue_capacity, (void**)&scratch->ready_queue));
  iree_host_size_t minimum_location_count = 0;
  if (move_count > (IREE_HOST_SIZE_MAX - temporary_location_count) / 2) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parallel move location count exceeds host size");
  }
  minimum_location_count = move_count * 2 + temporary_location_count;
  if (minimum_location_count > IREE_HOST_SIZE_MAX / 2) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parallel move location table exceeds host size");
  }
  iree_host_size_t location_entry_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_next_power_of_two(
      minimum_location_count * 2, &location_entry_capacity));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_array(
      scratch, location_entry_capacity, sizeof(*scratch->location_entries),
      &scratch->location_entry_capacity, (void**)&scratch->location_entries));
  *out_location_entry_count = location_entry_capacity;
  return iree_ok_status();
}

static loom_low_move_sequence_location_entry_t*
loom_low_move_sequence_lookup_location(
    loom_low_move_sequence_state_t* state,
    const loom_low_move_location_t* location) {
  loom_low_move_sequence_scratch_t* scratch = state->scratch;
  const iree_host_size_t mask = state->location_entry_count - 1;
  iree_host_size_t index =
      (iree_host_size_t)loom_low_move_sequence_location_hash(
          state->options->descriptor_set, location) &
      mask;
  for (iree_host_size_t probe = 0; probe < state->location_entry_count;
       ++probe) {
    loom_low_move_sequence_location_entry_t* entry =
        &scratch->location_entries[index];
    if (!iree_any_bit_set(entry->flags,
                          LOOM_LOW_MOVE_SEQUENCE_LOCATION_FLAG_OCCUPIED)) {
      return NULL;
    }
    if (loom_low_move_locations_share_target_storage(
            state->options->descriptor_set, &entry->location, location)) {
      return entry;
    }
    index = (index + 1) & mask;
  }
  return NULL;
}

static loom_low_move_sequence_location_entry_t*
loom_low_move_sequence_insert_location(
    loom_low_move_sequence_state_t* state,
    const loom_low_move_location_t* location) {
  loom_low_move_sequence_scratch_t* scratch = state->scratch;
  const iree_host_size_t mask = state->location_entry_count - 1;
  iree_host_size_t index =
      (iree_host_size_t)loom_low_move_sequence_location_hash(
          state->options->descriptor_set, location) &
      mask;
  for (iree_host_size_t probe = 0; probe < state->location_entry_count;
       ++probe) {
    loom_low_move_sequence_location_entry_t* entry =
        &scratch->location_entries[index];
    if (!iree_any_bit_set(entry->flags,
                          LOOM_LOW_MOVE_SEQUENCE_LOCATION_FLAG_OCCUPIED)) {
      *entry = (loom_low_move_sequence_location_entry_t){
          .location = *location,
          .destination_move_index = LOOM_LOW_MOVE_SEQUENCE_INDEX_NONE,
          .flags = LOOM_LOW_MOVE_SEQUENCE_LOCATION_FLAG_OCCUPIED,
      };
      return entry;
    }
    if (loom_low_move_locations_share_target_storage(
            state->options->descriptor_set, &entry->location, location)) {
      return entry;
    }
    index = (index + 1) & mask;
  }
  IREE_ASSERT_UNREACHABLE(
      "parallel move location table exhausted after reserve");
  return NULL;
}

static loom_low_move_sequence_location_entry_t*
loom_low_move_sequence_require_location(
    loom_low_move_sequence_state_t* state,
    const loom_low_move_location_t* location) {
  loom_low_move_sequence_location_entry_t* entry =
      loom_low_move_sequence_lookup_location(state, location);
  IREE_ASSERT(entry != NULL, "parallel move location must be present");
  return entry;
}

static void loom_low_move_sequence_enqueue_ready(
    loom_low_move_sequence_state_t* state, iree_host_size_t move_index) {
  IREE_ASSERT_LT(state->ready_tail, state->move_count,
                 "ready queue must have one entry per move");
  state->scratch->ready_queue[state->ready_tail++] = move_index;
}

static bool loom_low_move_sequence_node_is_active(
    const loom_low_move_sequence_state_t* state, iree_host_size_t move_index) {
  return iree_any_bit_set(state->scratch->nodes[move_index].flags,
                          LOOM_LOW_MOVE_SEQUENCE_NODE_FLAG_ACTIVE);
}

static iree_status_t loom_low_move_sequence_deactivate_move(
    loom_low_move_sequence_state_t* state, iree_host_size_t move_index) {
  IREE_ASSERT(loom_low_move_sequence_node_is_active(state, move_index),
              "parallel move row must be active before deactivation");
  state->scratch->nodes[move_index].flags = 0;
  --state->active_count;
  loom_low_move_sequence_location_entry_t* source_entry =
      loom_low_move_sequence_require_location(
          state, &state->scratch->moves[move_index].source);
  IREE_ASSERT_NE(source_entry->source_use_count, 0,
                 "parallel move source-use count must be positive");
  --source_entry->source_use_count;
  if (source_entry->source_use_count == 0 &&
      source_entry->destination_move_index !=
          LOOM_LOW_MOVE_SEQUENCE_INDEX_NONE &&
      loom_low_move_sequence_node_is_active(
          state, source_entry->destination_move_index)) {
    loom_low_move_sequence_enqueue_ready(state,
                                         source_entry->destination_move_index);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_move_sequence_find_temporary(
    const loom_low_move_sequence_options_t* options,
    const loom_low_move_location_t* storage_class,
    const loom_low_move_location_t** out_temporary_location) {
  *out_temporary_location = NULL;
  for (iree_host_size_t i = 0; i < options->temporary_location_count; ++i) {
    const loom_low_move_location_t* temporary =
        &options->temporary_locations[i];
    if (!loom_low_move_locations_share_storage_class(
            options->descriptor_set, temporary, storage_class)) {
      continue;
    }
    *out_temporary_location = temporary;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "parallel move cycle requires a temporary "
                          "location");
}

static iree_status_t loom_low_move_sequence_prepare(
    loom_low_move_sequence_state_t* state) {
  loom_low_move_sequence_scratch_t* scratch = state->scratch;
  IREE_ASSERT_GE(scratch->move_capacity, state->move_count,
                 "move scratch must be reserved before sequencing");
  iree_host_size_t active_move_count = 0;
  for (iree_host_size_t i = 0; i < state->move_count; ++i) {
    const loom_low_move_t move = scratch->moves[i];
    if (loom_low_move_locations_share_target_storage(
            state->options->descriptor_set, &move.destination, &move.source)) {
      continue;
    }
    scratch->moves[active_move_count++] = move;
  }
  state->move_count = active_move_count;
  state->active_count = active_move_count;
  if (active_move_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_low_move_sequence_reserve_solver_scratch(
      scratch, active_move_count, state->options->temporary_location_count,
      &state->location_entry_count));
  memset(scratch->nodes, 0, active_move_count * sizeof(*scratch->nodes));
  memset(scratch->location_entries, 0,
         state->location_entry_count * sizeof(*scratch->location_entries));

  for (iree_host_size_t i = 0; i < active_move_count; ++i) {
    scratch->nodes[i].flags = LOOM_LOW_MOVE_SEQUENCE_NODE_FLAG_ACTIVE;
    loom_low_move_sequence_location_entry_t* destination_entry =
        loom_low_move_sequence_insert_location(state,
                                               &scratch->moves[i].destination);
    if (destination_entry->destination_move_index !=
        LOOM_LOW_MOVE_SEQUENCE_INDEX_NONE) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parallel move rows contain duplicate destinations");
    }
    destination_entry->destination_move_index = i;
    loom_low_move_sequence_location_entry_t* source_entry =
        loom_low_move_sequence_insert_location(state,
                                               &scratch->moves[i].source);
    ++source_entry->source_use_count;
  }

  for (iree_host_size_t i = 0; i < state->options->temporary_location_count;
       ++i) {
    const loom_low_move_location_t* temporary =
        &state->options->temporary_locations[i];
    loom_low_move_sequence_location_entry_t* temporary_entry =
        loom_low_move_sequence_insert_location(state, temporary);
    if (iree_any_bit_set(temporary_entry->flags,
                         LOOM_LOW_MOVE_SEQUENCE_LOCATION_FLAG_TEMPORARY)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parallel move temporary locations contain duplicates");
    }
    if (temporary_entry->destination_move_index !=
            LOOM_LOW_MOVE_SEQUENCE_INDEX_NONE ||
        temporary_entry->source_use_count != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parallel move temporary location aliases move storage");
    }
    temporary_entry->flags |= LOOM_LOW_MOVE_SEQUENCE_LOCATION_FLAG_TEMPORARY;
  }

  for (iree_host_size_t i = 0; i < active_move_count; ++i) {
    loom_low_move_sequence_location_entry_t* destination_entry =
        loom_low_move_sequence_require_location(state,
                                                &scratch->moves[i].destination);
    if (destination_entry->source_use_count == 0) {
      loom_low_move_sequence_enqueue_ready(state, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_move_sequence_emit_one(
    const loom_low_move_sequence_options_t* options,
    const loom_low_move_location_t* destination,
    const loom_low_move_location_t* source) {
  if (loom_low_move_locations_share_target_storage(options->descriptor_set,
                                                   destination, source)) {
    return iree_ok_status();
  }
  return options->emit_move.fn(options->emit_move.user_data, destination,
                               source);
}

static iree_status_t loom_low_move_sequence_emit_ready_move(
    loom_low_move_sequence_state_t* state, iree_host_size_t move_index) {
  const loom_low_move_t move = state->scratch->moves[move_index];
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_emit_one(
      state->options, &move.destination, &move.source));
  return loom_low_move_sequence_deactivate_move(state, move_index);
}

static iree_status_t loom_low_move_sequence_next_active_move(
    loom_low_move_sequence_state_t* state, iree_host_size_t* out_move_index) {
  while (state->active_cursor < state->move_count) {
    const iree_host_size_t move_index = state->active_cursor++;
    if (loom_low_move_sequence_node_is_active(state, move_index)) {
      *out_move_index = move_index;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "parallel move active set is inconsistent");
}

static iree_status_t loom_low_move_sequence_emit_cycle(
    loom_low_move_sequence_state_t* state, iree_host_size_t first_move_index) {
  loom_low_move_sequence_scratch_t* scratch = state->scratch;
  const loom_low_move_location_t saved_location =
      scratch->moves[first_move_index].destination;
  const loom_low_move_location_t* temporary_location = NULL;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_find_temporary(
      state->options, &saved_location, &temporary_location));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_emit_one(
      state->options, temporary_location, &saved_location));

  loom_low_move_location_t destination =
      scratch->moves[first_move_index].destination;
  loom_low_move_location_t source = scratch->moves[first_move_index].source;
  for (;;) {
    loom_low_move_sequence_location_entry_t* destination_entry =
        loom_low_move_sequence_lookup_location(state, &destination);
    if (destination_entry == NULL ||
        destination_entry->destination_move_index ==
            LOOM_LOW_MOVE_SEQUENCE_INDEX_NONE ||
        !loom_low_move_sequence_node_is_active(
            state, destination_entry->destination_move_index)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "parallel move cycle is malformed");
    }
    const iree_host_size_t move_index =
        destination_entry->destination_move_index;
    if (loom_low_move_locations_share_target_storage(
            state->options->descriptor_set, &source, &saved_location)) {
      IREE_RETURN_IF_ERROR(loom_low_move_sequence_emit_one(
          state->options, &destination, temporary_location));
      return loom_low_move_sequence_deactivate_move(state, move_index);
    }

    IREE_RETURN_IF_ERROR(
        loom_low_move_sequence_emit_one(state->options, &destination, &source));
    IREE_RETURN_IF_ERROR(
        loom_low_move_sequence_deactivate_move(state, move_index));
    destination = source;
    destination_entry =
        loom_low_move_sequence_lookup_location(state, &destination);
    if (destination_entry == NULL ||
        destination_entry->destination_move_index ==
            LOOM_LOW_MOVE_SEQUENCE_INDEX_NONE) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "parallel move cycle is malformed");
    }
    source = scratch->moves[destination_entry->destination_move_index].source;
  }
}

iree_status_t loom_low_move_sequence_emit(
    loom_low_move_sequence_scratch_t* scratch, iree_host_size_t move_count,
    const loom_low_move_sequence_options_t* options) {
  loom_low_move_sequence_state_t state = {
      .scratch = scratch,
      .options = options,
      .move_count = move_count,
  };
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_prepare(&state));
  while (state.active_count != 0) {
    while (state.ready_head != state.ready_tail) {
      const iree_host_size_t move_index =
          scratch->ready_queue[state.ready_head++];
      if (!loom_low_move_sequence_node_is_active(&state, move_index)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_low_move_sequence_emit_ready_move(&state, move_index));
    }
    if (state.active_count == 0) {
      break;
    }
    iree_host_size_t first_move_index = 0;
    IREE_RETURN_IF_ERROR(
        loom_low_move_sequence_next_active_move(&state, &first_move_index));
    IREE_RETURN_IF_ERROR(
        loom_low_move_sequence_emit_cycle(&state, first_move_index));
  }
  return iree_ok_status();
}
