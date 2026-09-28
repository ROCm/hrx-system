// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/unit_liveness.h"

#include <string.h>

#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/representation_binding.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/util/adaptive_sort.h"

struct loom_low_allocation_clobber_t {
  // Shared register storage identity; explicit physical units use key zero.
  uint32_t storage_key;
  // Atomic physical unit or linear register location within storage_key.
  uint32_t location;
  // Program point overwritten by the implicit instruction output.
  uint32_t point;
};

static bool loom_low_allocation_clobber_less(
    const loom_low_allocation_clobber_t* lhs,
    const loom_low_allocation_clobber_t* rhs) {
  if (lhs->storage_key != rhs->storage_key) {
    return lhs->storage_key < rhs->storage_key;
  }
  if (lhs->location != rhs->location) {
    return lhs->location < rhs->location;
  }
  return lhs->point < rhs->point;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_low_allocation_clobber_sort,
                          loom_low_allocation_clobber_t,
                          loom_low_allocation_clobber_less)

static iree_status_t loom_low_allocation_unit_liveness_note_clobber(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_descriptor_set_t* descriptor_set, uint16_t reg_class_id,
    uint32_t point, iree_arena_allocator_t* arena) {
  const loom_low_reg_class_t* reg_class =
      &descriptor_set->reg_classes[reg_class_id];
  if (!iree_any_bit_set(reg_class->flags, LOOM_LOW_REG_CLASS_FLAG_PHYSICAL)) {
    return iree_ok_status();
  }
  const uint16_t* atomic_units = NULL;
  uint16_t atomic_unit_count = 1;
  uint32_t storage_key =
      loom_low_reg_class_storage_key(descriptor_set, reg_class_id);
  if (loom_low_reg_class_uses_explicit_physical_registers(reg_class)) {
    const uint32_t physical_register_id =
        loom_low_descriptor_set_physical_register_candidate(descriptor_set,
                                                            reg_class_id, 0);
    atomic_units = loom_low_descriptor_set_physical_register_atomic_units(
        descriptor_set, physical_register_id, &atomic_unit_count);
    storage_key = 0;
    unit_liveness->clobbers.atomic_unit_begin =
        unit_liveness->clobbers.atomic_unit_end == 0
            ? atomic_units[0]
            : iree_min(unit_liveness->clobbers.atomic_unit_begin,
                       atomic_units[0]);
    unit_liveness->clobbers.atomic_unit_end =
        iree_max(unit_liveness->clobbers.atomic_unit_end,
                 (uint32_t)atomic_units[atomic_unit_count - 1] + 1);
  }
  if (unit_liveness->clobbers.count + atomic_unit_count >
      unit_liveness->clobbers.capacity) {
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(arena, unit_liveness->clobbers.count,
                              unit_liveness->clobbers.count + atomic_unit_count,
                              sizeof(*unit_liveness->clobbers.entries),
                              &unit_liveness->clobbers.capacity,
                              (void**)&unit_liveness->clobbers.entries));
  }
  for (uint16_t i = 0; i < atomic_unit_count; ++i) {
    unit_liveness->clobbers.entries[unit_liveness->clobbers.count++] =
        (loom_low_allocation_clobber_t){
            .storage_key = storage_key,
            .location = atomic_units != NULL ? atomic_units[i] : 0,
            .point = point,
        };
  }
  return iree_ok_status();
}

static bool loom_low_allocation_unit_liveness_unit_is_clobbered(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    uint32_t storage_key, uint32_t location, uint32_t start_point,
    uint32_t end_point) {
  const loom_low_allocation_clobber_t key = {
      .storage_key = storage_key, .location = location, .point = start_point};
  iree_host_size_t begin = 0;
  iree_host_size_t end = unit_liveness->clobbers.count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (loom_low_allocation_clobber_less(
            &unit_liveness->clobbers.entries[middle], &key)) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  if (begin == unit_liveness->clobbers.count) {
    return false;
  }
  const loom_low_allocation_clobber_t* clobber =
      &unit_liveness->clobbers.entries[begin];
  return clobber->storage_key == storage_key && clobber->location == location &&
         clobber->point < end_point;
}

bool loom_low_allocation_unit_liveness_clobber_conflicts(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* candidate) {
  if (unit_liveness->clobbers.count == 0 ||
      candidate->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    return false;
  }
  const bool is_explicit =
      loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, candidate);
  if (is_explicit) {
    const loom_low_physical_register_t* physical_register =
        &descriptor_set->physical_registers[candidate->location_base];
    const uint16_t* atomic_units =
        &descriptor_set->physical_register_atomic_units
             [physical_register->atomic_unit_start];
    if (atomic_units[0] >= unit_liveness->clobbers.atomic_unit_end ||
        atomic_units[physical_register->atomic_unit_count - 1] <
            unit_liveness->clobbers.atomic_unit_begin) {
      return false;
    }
  }
  for (uint32_t unit = 0; unit < candidate->location_count; ++unit) {
    const uint32_t start_point =
        loom_low_allocation_live_range_assignment_unit_start_point(
            unit_liveness->start_points, unit_liveness->point_count, candidate,
            unit);
    const uint32_t end_point =
        loom_low_allocation_live_range_assignment_unit_end_point(
            unit_liveness->end_points, unit_liveness->point_count, candidate,
            unit);
    const uint16_t* atomic_units = NULL;
    uint16_t atomic_unit_count = 1;
    uint32_t storage_key = loom_low_reg_class_storage_key(
        descriptor_set, candidate->descriptor_reg_class_id);
    if (is_explicit) {
      uint32_t physical_register_id = 0;
      const bool resolved =
          loom_low_allocation_storage_assignment_unit_physical_register(
              descriptor_set, candidate, unit, &physical_register_id);
      IREE_ASSERT(resolved, "accepted assignment must name its physical units");
      atomic_units = loom_low_descriptor_set_physical_register_atomic_units(
          descriptor_set, physical_register_id, &atomic_unit_count);
      storage_key = 0;
    }
    for (uint16_t atomic_unit = 0; atomic_unit < atomic_unit_count;
         ++atomic_unit) {
      const uint32_t location = atomic_units != NULL
                                    ? atomic_units[atomic_unit]
                                    : candidate->location_base + unit;
      if (candidate->liveness_segments.count == 0) {
        if (loom_low_allocation_unit_liveness_unit_is_clobbered(
                unit_liveness, storage_key, location, start_point, end_point)) {
          return true;
        }
        continue;
      }
      for (uint32_t segment_index = 0;
           segment_index < candidate->liveness_segments.count;
           ++segment_index) {
        const loom_liveness_segment_t* segment =
            &unit_liveness->storage_segments
                 .entries[candidate->liveness_segments.start + segment_index];
        if (loom_low_allocation_unit_liveness_unit_is_clobbered(
                unit_liveness, storage_key, location,
                iree_max(start_point, segment->start_point),
                iree_min(end_point, segment->end_point))) {
          return true;
        }
      }
    }
  }
  return false;
}

static bool loom_low_allocation_unit_liveness_value_ordinal_for_value(
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, loom_value_id_t value_id,
    loom_value_ordinal_t* out_value_ordinal) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(value_domain, value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      value_ordinal >= liveness->value_count) {
    return false;
  }
  *out_value_ordinal = value_ordinal;
  return true;
}

static iree_status_t loom_low_allocation_unit_liveness_note_unit_use_at_point(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_value_ordinal_t value_ordinal, uint32_t unit_offset,
    uint32_t unit_count, uint32_t point) {
  if (unit_count == 0) {
    return iree_ok_status();
  }
  if (value_ordinal >= liveness->value_count) {
    return iree_ok_status();
  }
  const uint32_t unit_point_start =
      loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
          unit_liveness, liveness, value_ordinal);
  if (unit_point_start == UINT32_MAX) {
    return iree_ok_status();
  }
  const loom_liveness_interval_t* interval =
      loom_liveness_interval_for_value_ordinal(liveness, value_ordinal);
  if (!interval || unit_offset > interval->unit_count ||
      unit_count > interval->unit_count - unit_offset) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation unit liveness use exceeds value unit count");
  }
  if (point == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low allocation unit use point exceeds u32 range");
  }
  const uint32_t end_point = point + 1u;
  if (end_point > interval->end_point) {
    // This storage use extends beyond the value's semantic SSA interval, so
    // its sparse semantic segments no longer fully describe storage
    // conflicts. Loop-carried storage can outlive its old SSA value while
    // awaiting the parallel move that starts the next iteration.
    iree_bitmap_set(unit_liveness->values_with_incomplete_storage_segments,
                    value_ordinal);
  }
  for (uint32_t i = 0; i < unit_count; ++i) {
    const iree_host_size_t unit_end_point_index =
        (iree_host_size_t)unit_point_start + unit_offset + i;
    uint32_t* unit_end_point = &unit_liveness->end_points[unit_end_point_index];
    if (*unit_end_point < end_point) {
      *unit_end_point = end_point;
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_unit_liveness_note_value_ordinal_use_at_point(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_value_ordinal_t value_ordinal, uint32_t point) {
  if (value_ordinal >= liveness->value_count) {
    return iree_ok_status();
  }
  const loom_liveness_interval_t* interval =
      loom_liveness_interval_for_value_ordinal(liveness, value_ordinal);
  if (!interval ||
      !loom_low_allocation_live_range_interval_is_allocatable(interval)) {
    return iree_ok_status();
  }
  return loom_low_allocation_unit_liveness_note_unit_use_at_point(
      unit_liveness, liveness, value_ordinal, /*unit_offset=*/0,
      interval->unit_count, point);
}

static iree_status_t loom_low_allocation_unit_liveness_note_value_use_at_point(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, loom_value_id_t value_id,
    uint32_t point) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_unit_liveness_value_ordinal_for_value(
          value_domain, liveness, value_id, &value_ordinal)) {
    return iree_ok_status();
  }
  return loom_low_allocation_unit_liveness_note_value_ordinal_use_at_point(
      unit_liveness, liveness, value_ordinal, point);
}

static iree_status_t
loom_low_allocation_unit_liveness_note_contiguous_part_uses_at_point(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement,
    loom_value_ordinal_t aggregate_ordinal, uint32_t unit_offset,
    uint32_t unit_count, uint32_t point) {
  const uint64_t query_begin = unit_offset;
  const uint64_t query_end = query_begin + unit_count;
  const loom_low_placement_relation_range_t range =
      loom_low_placement_relation_range_for_value_ordinal(placement,
                                                          aggregate_ordinal);
  for (uint32_t i = 0; i < range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &placement->relations[range.start + i];
    if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT ||
        relation->kind != LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART) {
      continue;
    }
    const uint64_t relation_begin = relation->result_unit_offset;
    const uint64_t relation_end = relation_begin + relation->unit_count;
    const uint64_t intersection_begin =
        query_begin > relation_begin ? query_begin : relation_begin;
    const uint64_t intersection_end =
        query_end < relation_end ? query_end : relation_end;
    if (intersection_begin >= intersection_end) {
      continue;
    }
    const uint32_t source_unit_offset =
        relation->source_unit_offset +
        (uint32_t)(intersection_begin - relation_begin);
    const uint32_t source_unit_count =
        (uint32_t)(intersection_end - intersection_begin);
    iree_bitmap_set(unit_liveness->values_with_incomplete_storage_segments,
                    relation->source_ordinal);
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_unit_use_at_point(
            unit_liveness, liveness, relation->source_ordinal,
            source_unit_offset, source_unit_count, point));
  }
  return iree_ok_status();
}

static bool loom_low_allocation_unit_liveness_contiguous_parts_cover_unit_range(
    const loom_low_placement_table_t* placement,
    loom_value_ordinal_t aggregate_ordinal, uint32_t unit_offset,
    uint32_t unit_count) {
  if (unit_count == 0) {
    return true;
  }
  const uint64_t query_begin = unit_offset;
  const uint64_t query_end = query_begin + unit_count;
  uint64_t covered_units = 0;
  const loom_low_placement_relation_range_t range =
      loom_low_placement_relation_range_for_value_ordinal(placement,
                                                          aggregate_ordinal);
  for (uint32_t i = 0; i < range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &placement->relations[range.start + i];
    if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT ||
        relation->kind != LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART) {
      continue;
    }
    const uint64_t relation_begin = relation->result_unit_offset;
    const uint64_t relation_end = relation_begin + relation->unit_count;
    const uint64_t intersection_begin =
        query_begin > relation_begin ? query_begin : relation_begin;
    const uint64_t intersection_end =
        query_end < relation_end ? query_end : relation_end;
    if (intersection_begin >= intersection_end) {
      continue;
    }
    covered_units += intersection_end - intersection_begin;
    if (covered_units > unit_count) {
      return false;
    }
  }
  return covered_units == unit_count;
}

static bool loom_low_allocation_unit_liveness_relation_is_edge_payload(
    const loom_low_placement_relation_t* relation) {
  return loom_low_placement_cause_is_edge(relation->cause) &&
         relation->kind == LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE;
}

static const loom_low_placement_relation_t*
loom_low_allocation_unit_liveness_decomposable_edge_payload_relation(
    const loom_low_placement_table_t* placement, const loom_op_t* op,
    loom_value_ordinal_t source_ordinal) {
  if (placement == NULL) {
    return NULL;
  }
  const loom_low_placement_relation_range_t range =
      loom_low_placement_relation_range_for_source_value_ordinal(
          placement, source_ordinal);
  for (uint32_t i = 0; i < range.count; ++i) {
    const uint32_t relation_index =
        placement->relation_indices_by_source_ordinal[range.start + i];
    const loom_low_placement_relation_t* relation =
        &placement->relations[relation_index];
    if (relation->op != op ||
        !loom_low_allocation_unit_liveness_relation_is_edge_payload(relation)) {
      continue;
    }
    if (loom_low_allocation_unit_liveness_contiguous_parts_cover_unit_range(
            placement, relation->source_ordinal, relation->source_unit_offset,
            relation->unit_count)) {
      return relation;
    }
  }
  return NULL;
}

static iree_status_t
loom_low_allocation_unit_liveness_note_operation_direct_unit_uses(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement,
    const loom_liveness_operation_point_t* operation_point) {
  for (uint32_t i = 0; i < operation_point->direct_use_count; ++i) {
    const loom_value_ordinal_t value_ordinal =
        loom_liveness_operation_use_ordinal(liveness,
                                            operation_point->use_start + i);
    const loom_low_placement_relation_t* edge_relation =
        loom_low_allocation_unit_liveness_decomposable_edge_payload_relation(
            placement, operation_point->op, value_ordinal);
    if (edge_relation != NULL) {
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_unit_liveness_note_contiguous_part_uses_at_point(
              unit_liveness, liveness, placement, edge_relation->source_ordinal,
              edge_relation->source_unit_offset, edge_relation->unit_count,
              operation_point->start_point));
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_value_ordinal_use_at_point(
            unit_liveness, liveness, value_ordinal,
            operation_point->start_point));
  }
  return iree_ok_status();
}

static bool loom_low_allocation_unit_liveness_op_ties_result_to_operand(
    const loom_op_t* op, uint16_t result_index, uint16_t operand_index) {
  const loom_tied_result_t* tied_results = loom_op_tied_results(op);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    if (tied_results[i].result_index == result_index &&
        tied_results[i].operand_index == operand_index) {
      return true;
    }
  }
  return false;
}

static iree_status_t
loom_low_allocation_unit_liveness_note_early_clobber_operand_uses(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, const loom_op_t* op,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    uint16_t early_clobber_result_index, uint32_t clobber_point) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    if (!loom_low_descriptor_operand_maps_to_packet_operand(descriptor_set,
                                                            descriptor, i)) {
      continue;
    }
    const loom_low_operand_t* descriptor_operand =
        &descriptor_set->operands[descriptor->operand_start + i];
    const uint16_t operand_index = descriptor_operand->source_value_index;
    if (operand_index >= op->operand_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low allocation early-clobber operand index exceeds packet operand "
          "count");
    }
    if (loom_low_descriptor_operands_are_tied(descriptor_set, descriptor,
                                              early_clobber_result_index, i) ||
        loom_low_allocation_unit_liveness_op_ties_result_to_operand(
            op, early_clobber_result_index, operand_index)) {
      continue;
    }
    loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
    if (!loom_low_allocation_unit_liveness_value_ordinal_for_value(
            value_domain, liveness, operands[operand_index], &value_ordinal)) {
      continue;
    }
    iree_bitmap_set(unit_liveness->values_with_incomplete_storage_segments,
                    value_ordinal);
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_value_ordinal_use_at_point(
            unit_liveness, liveness, value_ordinal, clobber_point));
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_unit_liveness_note_descriptor_unit_uses(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_resolved_target_t* target,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, const loom_op_t* op,
    uint32_t point, iree_arena_allocator_t* arena) {
  if (!loom_low_op_isa(op) && !loom_low_const_isa(op)) {
    return iree_ok_status();
  }
  if (point >= UINT32_MAX - 1u) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low allocation descriptor operation point exceeds u32 range");
  }
  const uint32_t clobber_point = point + 1u;
  loom_low_descriptor_packet_t packet = {0};
  loom_low_descriptor_packet_initialize(target->descriptor_set, op, &packet);
  if (packet.kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    return iree_ok_status();
  }

  const loom_low_descriptor_set_t* descriptor_set = target->descriptor_set;
  const loom_low_descriptor_t* descriptor = packet.descriptor;
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + i];
    if (operand->source_value_index != LOOM_LOW_ID_NONE ||
        !iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_STATE_WRITE)) {
      continue;
    }
    const uint16_t reg_class_id =
        descriptor_set->reg_class_alts[operand->reg_class_alt_start]
            .reg_class_id;
    IREE_RETURN_IF_ERROR(loom_low_allocation_unit_liveness_note_clobber(
        unit_liveness, descriptor_set, reg_class_id, clobber_point, arena));
  }
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_EARLY_CLOBBER) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_early_clobber_operand_uses(
            unit_liveness, value_domain, liveness, op, descriptor_set,
            descriptor, constraint->lhs_operand_index, clobber_point));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_unit_liveness_note_slice_unit_uses(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, const loom_op_t* op,
    uint32_t point) {
  const int64_t offset = loom_low_slice_offset(op);
  if (offset < 0 || offset > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation unit liveness saw malformed low.slice offset");
  }
  loom_value_ordinal_t result_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_unit_liveness_value_ordinal_for_value(
          value_domain, liveness, loom_low_slice_result(op), &result_ordinal)) {
    return iree_ok_status();
  }
  const loom_liveness_interval_t* result_interval =
      loom_liveness_interval_for_value_ordinal(liveness, result_ordinal);
  if (!result_interval ||
      !loom_low_allocation_live_range_interval_is_allocatable(
          result_interval)) {
    return iree_ok_status();
  }
  loom_value_ordinal_t source_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_unit_liveness_value_ordinal_for_value(
          value_domain, liveness, loom_low_slice_source(op), &source_ordinal)) {
    return iree_ok_status();
  }
  return loom_low_allocation_unit_liveness_note_unit_use_at_point(
      unit_liveness, liveness, source_ordinal, (uint32_t)offset,
      result_interval->unit_count, point);
}

static iree_status_t
loom_low_allocation_unit_liveness_note_operation_nested_uses_at_point(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_liveness_operation_point_t* operation_point, uint32_t point) {
  for (uint32_t i = operation_point->direct_use_count;
       i < operation_point->use_count; ++i) {
    const loom_value_ordinal_t value_ordinal =
        loom_liveness_operation_use_ordinal(liveness,
                                            operation_point->use_start + i);
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_value_ordinal_use_at_point(
            unit_liveness, liveness, value_ordinal, point));
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_unit_liveness_note_low_scf_loop_backedge_uses(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness,
    const loom_liveness_operation_point_t* loop_point,
    uint32_t backedge_point) {
  const loom_op_t* loop_op = loop_point->op;
  const bool is_for = loom_low_scf_for_isa(loop_op);
  IREE_ASSERT(is_for || loom_low_scf_while_isa(loop_op),
              "structured-loop backedge must belong to for or while");
  const loom_region_t* loop_body = is_for ? loom_low_scf_for_body(loop_op)
                                          : loom_low_scf_while_after(loop_op);
  const loom_block_t* body_block = loom_region_const_entry_block(loop_body);
  IREE_ASSERT(body_block != NULL && (!is_for || body_block->arg_count > 0),
              "verified structured loop must have a valid body block");
  const loom_op_t* yield = body_block->last_op;
  IREE_ASSERT(yield != NULL && loom_low_scf_yield_isa(yield),
              "verified structured loop body must end in low.scf.yield");

  const loom_value_slice_t iter_args =
      is_for ? loom_low_scf_for_iter_args(loop_op)
             : loom_low_scf_while_iter_args(loop_op);
  const uint16_t implicit_arg_count = is_for ? 1 : 0;
  IREE_ASSERT_EQ(body_block->arg_count, iter_args.count + implicit_arg_count,
                 "verified structured loop body args must match iter args");

  // Structured loop lowering reuses captures, control values, and loop-carried
  // body arguments after the body has executed to start the next iteration or
  // move final results.
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_unit_liveness_note_operation_nested_uses_at_point(
          unit_liveness, liveness, loop_point, backedge_point));
  if (is_for) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_value_use_at_point(
            unit_liveness, value_domain, liveness,
            loom_block_arg_id(body_block, 0), backedge_point));
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_value_use_at_point(
            unit_liveness, value_domain, liveness,
            loom_low_scf_for_upper_bound(loop_op), backedge_point));
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_value_use_at_point(
            unit_liveness, value_domain, liveness,
            loom_low_scf_for_step(loop_op), backedge_point));
  }
  const loom_block_t* carried_block =
      is_for
          ? body_block
          : loom_region_const_entry_block(loom_low_scf_while_before(loop_op));
  IREE_ASSERT(carried_block != NULL,
              "verified structured loop must have a carried-value block");
  IREE_ASSERT_EQ(carried_block->arg_count, iter_args.count + implicit_arg_count,
                 "verified structured loop carried args must match iter args");
  for (uint16_t i = 0; i < iter_args.count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_value_use_at_point(
            unit_liveness, value_domain, liveness,
            loom_block_arg_id(carried_block,
                              (uint16_t)(i + implicit_arg_count)),
            backedge_point));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_unit_liveness_note_operation_unit_uses(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_module_t* module, const loom_low_resolved_target_t* target,
    const loom_low_placement_table_t* placement,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, uint32_t operation_index,
    iree_arena_allocator_t* arena) {
  const loom_liveness_operation_point_t* operation_point =
      &liveness->operation_points[operation_index];
  const loom_op_t* op = operation_point->op;
  if (loom_low_slice_isa(op)) {
    return loom_low_allocation_unit_liveness_note_slice_unit_uses(
        unit_liveness, value_domain, liveness, op,
        operation_point->start_point);
  }
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_unit_liveness_note_operation_direct_unit_uses(
          unit_liveness, liveness, placement, operation_point));
  if (loom_low_scf_yield_isa(op) &&
      operation_point->parent_operation_index != UINT32_MAX) {
    const uint32_t parent_operation_index =
        operation_point->parent_operation_index;
    IREE_ASSERT_LT(parent_operation_index, operation_index);
    const loom_liveness_operation_point_t* parent_point =
        &liveness->operation_points[parent_operation_index];
    if (loom_low_scf_for_isa(parent_point->op) ||
        loom_low_scf_while_isa(parent_point->op)) {
      const loom_region_t* loop_body =
          loom_low_scf_for_isa(parent_point->op)
              ? loom_low_scf_for_body(parent_point->op)
              : loom_low_scf_while_after(parent_point->op);
      const loom_block_t* body_block = loom_region_const_entry_block(loop_body);
      if (body_block != NULL && loom_block_const_last_op(body_block) == op) {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_unit_liveness_note_low_scf_loop_backedge_uses(
                unit_liveness, value_domain, liveness, parent_point,
                parent_point->end_point - 1));
      }
    }
  }
  return loom_low_allocation_unit_liveness_note_descriptor_unit_uses(
      unit_liveness, target, value_domain, liveness, op,
      operation_point->start_point, arena);
}

static iree_status_t
loom_low_allocation_unit_liveness_note_value_unit_uses_at_point(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, const loom_value_id_t* values,
    iree_host_size_t value_count, uint32_t point) {
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_value_use_at_point(
            unit_liveness, value_domain, liveness, values[i], point));
  }
  return iree_ok_status();
}

static bool
loom_low_allocation_unit_liveness_value_is_decomposable_live_out_edge(
    const loom_low_placement_table_t* placement,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness,
    const loom_liveness_block_info_t* block_info, loom_value_id_t value_id) {
  const loom_op_t* terminator = loom_block_const_last_op(block_info->block);
  if (terminator == NULL) {
    return false;
  }
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_unit_liveness_value_ordinal_for_value(
          value_domain, liveness, value_id, &value_ordinal)) {
    return false;
  }
  return loom_low_allocation_unit_liveness_decomposable_edge_payload_relation(
             placement, terminator, value_ordinal) != NULL;
}

static iree_status_t loom_low_allocation_unit_liveness_note_block_boundary_uses(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_placement_table_t* placement,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness) {
  for (iree_host_size_t i = 0; i < liveness->block_count; ++i) {
    const loom_liveness_block_info_t* block_info = &liveness->blocks[i];
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_value_unit_uses_at_point(
            unit_liveness, value_domain, liveness, block_info->live_in_values,
            block_info->live_in_count, block_info->start_point));
    for (iree_host_size_t j = 0; j < block_info->live_out_count; ++j) {
      const loom_value_id_t value_id = block_info->live_out_values[j];
      if (loom_low_allocation_unit_liveness_value_is_decomposable_live_out_edge(
              placement, value_domain, liveness, block_info, value_id)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_unit_liveness_note_value_use_at_point(
              unit_liveness, value_domain, liveness, value_id,
              block_info->end_point));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_unit_liveness_note_body_op_unit_uses(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_module_t* module, const loom_low_resolved_target_t* target,
    const loom_low_placement_table_t* placement,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, iree_arena_allocator_t* arena) {
  for (iree_host_size_t i = 0; i < liveness->operation_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_operation_unit_uses(
            unit_liveness, module, target, placement, value_domain, liveness,
            (uint32_t)i, arena));
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_unit_liveness_initialize(
    const loom_module_t* module, const loom_low_resolved_target_t* target,
    const loom_low_placement_table_t* placement,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, iree_arena_allocator_t* arena,
    loom_low_allocation_unit_liveness_t* out_unit_liveness) {
  IREE_ASSERT_ARGUMENT(out_unit_liveness);
  *out_unit_liveness = (loom_low_allocation_unit_liveness_t){0};
  out_unit_liveness->storage_segments.entries = liveness->segments;

  if (liveness->value_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, liveness->value_count,
        sizeof(*out_unit_liveness->point_starts_by_value_ordinal),
        (void**)&out_unit_liveness->point_starts_by_value_ordinal));
    for (iree_host_size_t i = 0; i < liveness->value_count; ++i) {
      out_unit_liveness->point_starts_by_value_ordinal[i] = UINT32_MAX;
    }
    const iree_host_size_t incomplete_segment_word_count =
        iree_bitmap_calculate_words(liveness->value_count);
    uint64_t* incomplete_segment_words = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, incomplete_segment_word_count, sizeof(*incomplete_segment_words),
        (void**)&incomplete_segment_words));
    memset(incomplete_segment_words, 0,
           incomplete_segment_word_count * sizeof(*incomplete_segment_words));
    out_unit_liveness->values_with_incomplete_storage_segments =
        (iree_bitmap_t){
            .bit_count = liveness->value_count,
            .words = incomplete_segment_words,
        };
  }

  iree_host_size_t unit_point_count = 0;
  for (iree_host_size_t i = 0; i < liveness->value_count; ++i) {
    const loom_liveness_interval_t* interval =
        loom_liveness_interval_for_value_ordinal(liveness,
                                                 (loom_value_ordinal_t)i);
    if (!interval ||
        !loom_low_allocation_live_range_interval_is_allocatable(interval)) {
      continue;
    }
    if (interval->unit_count > IREE_HOST_SIZE_MAX - unit_point_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low allocation unit liveness count exceeds host size");
    }
    unit_point_count += interval->unit_count;
  }
  if (unit_point_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, unit_point_count, sizeof(*out_unit_liveness->start_points),
      (void**)&out_unit_liveness->start_points));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, unit_point_count, sizeof(*out_unit_liveness->end_points),
      (void**)&out_unit_liveness->end_points));
  out_unit_liveness->point_count = unit_point_count;

  iree_host_size_t unit_point_start = 0;
  for (iree_host_size_t i = 0; i < liveness->value_count; ++i) {
    const loom_liveness_interval_t* interval =
        loom_liveness_interval_for_value_ordinal(liveness,
                                                 (loom_value_ordinal_t)i);
    if (!interval ||
        !loom_low_allocation_live_range_interval_is_allocatable(interval)) {
      continue;
    }
    if (unit_point_start > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low allocation unit liveness start exceeds u32 range");
    }
    out_unit_liveness->point_starts_by_value_ordinal[i] =
        (uint32_t)unit_point_start;
    for (uint32_t unit_index = 0; unit_index < interval->unit_count;
         ++unit_index) {
      out_unit_liveness->start_points[unit_point_start + unit_index] =
          interval->start_point;
      out_unit_liveness->end_points[unit_point_start + unit_index] =
          loom_low_allocation_live_range_interval_initial_unit_end_point(
              interval);
    }
    unit_point_start += interval->unit_count;
  }

  // Unit liveness refines the value-granular analysis inside blocks so
  // operations like low.slice can release dead units independently. CFG
  // boundaries are still value-granular: every unit of a block live-in/out
  // value must stay reserved across the boundary until a per-unit dataflow
  // analysis can prove otherwise.
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_unit_liveness_note_block_boundary_uses(
          out_unit_liveness, placement, value_domain, liveness));

  IREE_RETURN_IF_ERROR(loom_low_allocation_unit_liveness_note_body_op_unit_uses(
      out_unit_liveness, module, target, placement, value_domain, liveness,
      arena));
  loom_low_allocation_clobber_sort(out_unit_liveness->clobbers.entries,
                                   out_unit_liveness->clobbers.count);
  return iree_ok_status();
}

uint32_t loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_value_ordinal_t value_ordinal) {
  IREE_ASSERT_ARGUMENT(unit_liveness);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_LT(value_ordinal, liveness->value_count);
  if (unit_liveness->point_starts_by_value_ordinal == NULL) {
    return UINT32_MAX;
  }
  return unit_liveness->point_starts_by_value_ordinal[value_ordinal];
}

const uint32_t*
loom_low_allocation_unit_liveness_start_points_for_value_ordinal(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_value_ordinal_t value_ordinal) {
  const uint32_t start =
      loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
          unit_liveness, liveness, value_ordinal);
  return start == UINT32_MAX ? NULL : &unit_liveness->start_points[start];
}

loom_liveness_segment_range_t
loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_value_ordinal_t value_ordinal) {
  IREE_ASSERT_ARGUMENT(unit_liveness);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_LT(value_ordinal, liveness->value_count);
  if (iree_bitmap_test(unit_liveness->values_with_incomplete_storage_segments,
                       value_ordinal)) {
    return unit_liveness->storage_segments.tied_sources != NULL
               ? unit_liveness->storage_segments.tied_sources[value_ordinal]
               : (loom_liveness_segment_range_t){0};
  }
  return loom_liveness_segment_range_for_value_ordinal(liveness, value_ordinal);
}

bool loom_low_allocation_unit_liveness_storage_component_live_at_point(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement,
    loom_value_ordinal_t value_ordinal, uint32_t unit_offset,
    uint32_t unit_count, uint32_t program_point) {
  const loom_liveness_interval_t* value_interval =
      loom_liveness_interval_for_value_ordinal(liveness, value_ordinal);
  if (value_interval == NULL || unit_count == 0) {
    return false;
  }
  IREE_ASSERT_LE(unit_offset, value_interval->unit_count);
  IREE_ASSERT_LE(unit_count, value_interval->unit_count - unit_offset);

  const loom_value_ordinal_t storage_ordinal =
      placement->tied_storage_origins_by_value_ordinal == NULL
          ? value_ordinal
          : placement->tied_storage_origins_by_value_ordinal[value_ordinal];
  const loom_liveness_interval_t* storage_interval =
      loom_liveness_interval_for_value_ordinal(liveness, storage_ordinal);
  IREE_ASSERT_ARGUMENT(storage_interval);
  IREE_ASSERT_EQ(storage_interval->unit_count, value_interval->unit_count);

  const loom_liveness_segment_range_t segments =
      loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
          unit_liveness, liveness, storage_ordinal);
  const bool storage_live_at_point =
      segments.count == 0 ? storage_interval->start_point <= program_point
                          : loom_liveness_segment_range_contains(
                                unit_liveness->storage_segments.entries,
                                segments, program_point);
  if (!storage_live_at_point) {
    return false;
  }

  const uint32_t point_start =
      loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
          unit_liveness, liveness, storage_ordinal);
  IREE_ASSERT_NE(point_start, UINT32_MAX);
  IREE_ASSERT_LE((uint64_t)point_start + unit_offset + unit_count,
                 unit_liveness->point_count);
  for (uint32_t i = 0; i < unit_count; ++i) {
    if (unit_liveness->end_points[point_start + unit_offset + i] >
        program_point) {
      return true;
    }
  }
  return false;
}

// A contribution starts on one program-point bucket and is relinked into its
// source's merged reservation during the ascending point sweep.
typedef struct loom_low_allocation_storage_segment_contribution_t {
  // Source whose physical location must remain available for this lifetime.
  loom_value_ordinal_t source_ordinal;
  // Next contribution in the point bucket, then next merged source segment.
  uint32_t next;
  // Half-open physical reservation.
  loom_liveness_segment_t segment;
} loom_low_allocation_storage_segment_contribution_t;

typedef struct loom_low_allocation_storage_segment_chain_t {
  // First merged contribution, or UINT32_MAX for an empty chain.
  uint32_t head;
  // Last merged contribution, or UINT32_MAX for an empty chain.
  uint32_t tail;
  // Number of merged segments.
  uint32_t count;
  // True when this value is the origin of a tied component.
  bool needs_reservation;
} loom_low_allocation_storage_segment_chain_t;

static bool loom_low_allocation_unit_liveness_can_refine_tie(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_placement_relation_t* relation) {
  return relation->cause == LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT &&
         unit_liveness
                 ->point_starts_by_value_ordinal[relation->source_ordinal] !=
             UINT32_MAX &&
         unit_liveness
                 ->point_starts_by_value_ordinal[relation->result_ordinal] !=
             UINT32_MAX;
}

static void loom_low_allocation_unit_liveness_contribute_segments(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_value_ordinal_t source_ordinal, loom_value_ordinal_t value_ordinal,
    uint32_t* point_heads,
    loom_low_allocation_storage_segment_contribution_t* contributions,
    uint32_t* contribution_count) {
  const loom_liveness_segment_range_t range =
      loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
          unit_liveness, liveness, value_ordinal);
  loom_liveness_segment_t contiguous = {0};
  const loom_liveness_segment_t* segments = NULL;
  uint32_t count = range.count;
  if (count != 0) {
    segments = &liveness->segments[range.start];
  } else {
    // A result with early-clobber/edge storage or no semantic uses still
    // reserves its concrete writes. Preserve its initial per-unit hull.
    const loom_liveness_interval_t* interval =
        loom_liveness_interval_for_value_ordinal(liveness, value_ordinal);
    const uint32_t unit_start =
        unit_liveness->point_starts_by_value_ordinal[value_ordinal];
    contiguous.start_point = interval->start_point;
    for (uint32_t i = 0; i < interval->unit_count; ++i) {
      contiguous.end_point = iree_max(
          contiguous.end_point, unit_liveness->end_points[unit_start + i]);
    }
    segments = &contiguous;
    count = 1;
  }
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t index = (*contribution_count)++;
    const uint32_t point = segments[i].start_point;
    contributions[index] = (loom_low_allocation_storage_segment_contribution_t){
        .source_ordinal = source_ordinal,
        .next = point_heads[point],
        .segment = segments[i],
    };
    point_heads[point] = index;
  }
}

// Each value in a mandatory tied-storage component contributes its physical
// lifetime once. Program-point buckets merge every component in one ordered
// sweep without sorting, repeated transitive unions, or chain-length growth.
static iree_status_t loom_low_allocation_unit_liveness_build_storage_segments(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement,
    iree_arena_allocator_t* scratch_arena, iree_arena_allocator_t* arena) {
  loom_low_allocation_storage_segment_chain_t* chains = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, liveness->value_count, sizeof(*chains), (void**)&chains));
  for (iree_host_size_t i = 0; i < liveness->value_count; ++i) {
    chains[i] = (loom_low_allocation_storage_segment_chain_t){
        .head = UINT32_MAX, .tail = UINT32_MAX};
  }
  for (iree_host_size_t i = 0; i < placement->relation_count; ++i) {
    const loom_low_placement_relation_t* relation = &placement->relations[i];
    if (!loom_low_allocation_unit_liveness_can_refine_tie(unit_liveness,
                                                          relation)) {
      continue;
    }
    const loom_value_ordinal_t origin =
        placement
            ->tied_storage_origins_by_value_ordinal[relation->source_ordinal];
    chains[origin].needs_reservation = true;
  }

  uint64_t capacity = 0;
  for (loom_value_ordinal_t i = 0; i < liveness->value_count; ++i) {
    const loom_value_ordinal_t origin =
        placement->tied_storage_origins_by_value_ordinal[i];
    if (!chains[origin].needs_reservation) {
      continue;
    }
    capacity += iree_max(
        loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
            unit_liveness, liveness, i)
            .count,
        1u);
  }
  if (capacity > UINT32_MAX - liveness->segment_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low storage segment count exceeds u32 range");
  }
  loom_low_allocation_storage_segment_contribution_t* contributions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, (iree_host_size_t)capacity, sizeof(*contributions),
      (void**)&contributions));
  const uint64_t point_capacity =
      (uint64_t)liveness->blocks[liveness->block_count - 1].end_point + 1;
  if (point_capacity > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low storage point count exceeds host size");
  }
  const iree_host_size_t point_count = (iree_host_size_t)point_capacity;
  uint32_t* point_heads = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, point_count, sizeof(*point_heads), (void**)&point_heads));
  memset(point_heads, 0xFF, point_count * sizeof(*point_heads));
  uint32_t contribution_count = 0;
  for (loom_value_ordinal_t i = 0; i < liveness->value_count; ++i) {
    const loom_value_ordinal_t origin =
        placement->tied_storage_origins_by_value_ordinal[i];
    if (!chains[origin].needs_reservation) {
      continue;
    }
    loom_low_allocation_unit_liveness_contribute_segments(
        unit_liveness, liveness, origin, i, point_heads, contributions,
        &contribution_count);
  }
  IREE_ASSERT_EQ(contribution_count, (uint32_t)capacity);
  uint32_t segment_count = (uint32_t)liveness->segment_count;
  for (iree_host_size_t point = 0; point < point_count; ++point) {
    uint32_t index = point_heads[point];
    while (index != UINT32_MAX) {
      loom_low_allocation_storage_segment_contribution_t* contribution =
          &contributions[index];
      const uint32_t next = contribution->next;
      loom_low_allocation_storage_segment_chain_t* chain =
          &chains[contribution->source_ordinal];
      if (chain->tail != UINT32_MAX &&
          contribution->segment.start_point <=
              contributions[chain->tail].segment.end_point) {
        contributions[chain->tail].segment.end_point =
            iree_max(contributions[chain->tail].segment.end_point,
                     contribution->segment.end_point);
      } else {
        if (chain->tail == UINT32_MAX) {
          chain->head = index;
        } else {
          contributions[chain->tail].next = index;
        }
        contribution->next = UINT32_MAX;
        chain->tail = index;
        ++chain->count;
        ++segment_count;
      }
      index = next;
    }
  }
  loom_liveness_segment_t* segments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, segment_count, sizeof(*segments), (void**)&segments));
  memcpy(segments, liveness->segments,
         liveness->segment_count * sizeof(*segments));
  loom_liveness_segment_range_t* ranges = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, liveness->value_count, sizeof(*ranges), (void**)&ranges));
  uint32_t segment_index = (uint32_t)liveness->segment_count;
  for (loom_value_ordinal_t i = 0; i < liveness->value_count; ++i) {
    ranges[i] = (loom_liveness_segment_range_t){0};
    if (!chains[i].needs_reservation) {
      continue;
    }
    ranges[i] = (loom_liveness_segment_range_t){.start = segment_index,
                                                .count = chains[i].count};
    for (uint32_t index = chains[i].head; index != UINT32_MAX;
         index = contributions[index].next) {
      segments[segment_index++] = contributions[index].segment;
    }
  }
  for (iree_host_size_t i = 0; i < placement->relation_count; ++i) {
    const loom_low_placement_relation_t* relation = &placement->relations[i];
    if (!loom_low_allocation_unit_liveness_can_refine_tie(unit_liveness,
                                                          relation)) {
      continue;
    }
    const loom_value_ordinal_t origin =
        placement
            ->tied_storage_origins_by_value_ordinal[relation->source_ordinal];
    ranges[relation->source_ordinal] = ranges[origin];
  }
  IREE_ASSERT_EQ(segment_index, segment_count);
  unit_liveness->storage_segments.entries = segments;
  unit_liveness->storage_segments.tied_sources = ranges;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_unit_liveness_refine_storage_segments(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement,
    iree_arena_allocator_t* arena) {
  if (liveness->block_count < 2) {
    return iree_ok_status();
  }
  bool has_sparse_tie = false;
  for (iree_host_size_t i = 0; i < placement->relation_count; ++i) {
    const loom_low_placement_relation_t* relation = &placement->relations[i];
    if (loom_low_allocation_unit_liveness_can_refine_tie(unit_liveness,
                                                         relation) &&
        (liveness->value_segment_ranges[relation->source_ordinal].count > 1 ||
         liveness->value_segment_ranges[relation->result_ordinal].count > 1)) {
      has_sparse_tie = true;
      break;
    }
  }
  if (!has_sparse_tie) {
    return iree_ok_status();
  }
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  iree_status_t status =
      loom_low_allocation_unit_liveness_build_storage_segments(
          unit_liveness, liveness, placement, &scratch_arena, arena);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

// One origin assignment retains each tied component's complete physical
// lifetime. Required ties force every later member to that location, so
// extending every intermediate alias would duplicate the reservation and grow
// the active set with chain depth.
static void loom_low_allocation_unit_liveness_retain_tied_component_ends(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_placement_table_t* placement) {
  for (iree_host_size_t i = 0; i < placement->relation_count; ++i) {
    const loom_low_placement_relation_t* relation = &placement->relations[i];
    if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT) {
      continue;
    }
    const loom_value_ordinal_t origin_ordinal =
        placement
            ->tied_storage_origins_by_value_ordinal[relation->source_ordinal];
    const uint32_t origin_unit_point_start =
        unit_liveness->point_starts_by_value_ordinal[origin_ordinal];
    const uint32_t result_unit_point_start =
        unit_liveness->point_starts_by_value_ordinal[relation->result_ordinal];
    if (origin_unit_point_start == UINT32_MAX ||
        result_unit_point_start == UINT32_MAX) {
      continue;
    }
    for (uint32_t unit_index = 0; unit_index < relation->unit_count;
         ++unit_index) {
      uint32_t* origin_end_point =
          &unit_liveness->end_points[origin_unit_point_start +
                                     relation->source_unit_offset + unit_index];
      const uint32_t result_end_point =
          unit_liveness->end_points[result_unit_point_start +
                                    relation->result_unit_offset + unit_index];
      *origin_end_point = iree_max(*origin_end_point, result_end_point);
    }
  }
}

iree_status_t loom_low_allocation_unit_liveness_propagate_storage_relations(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement,
    iree_arena_allocator_t* arena) {
  IREE_ASSERT_ARGUMENT(unit_liveness);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(placement);
  if (unit_liveness->end_points == NULL) {
    return iree_ok_status();
  }
  const loom_value_ordinal_t* order = placement->storage_value_order;
  const loom_value_ordinal_t order_count = placement->storage_value_order_count;
  if (order_count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(order_count, placement->value_count);

  loom_low_allocation_unit_liveness_retain_tied_component_ends(unit_liveness,
                                                               placement);

  IREE_RETURN_IF_ERROR(
      loom_low_allocation_unit_liveness_refine_storage_segments(
          unit_liveness, liveness, placement, arena));

  // Starts flow from sources to users. Reverse the same retained order so
  // tied-result starts reach any eventual concat reservation transitively.
  for (loom_value_ordinal_t cursor = order_count; cursor > 0; --cursor) {
    const loom_low_placement_relation_range_t range =
        placement->ranges_by_result_ordinal[order[cursor - 1]];
    for (uint32_t i = 0; i < range.count; ++i) {
      const loom_low_placement_relation_t* relation =
          &placement->relations[range.start + i];
      const bool is_tied_result =
          relation->cause == LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT;
      const bool is_concat_part =
          relation->cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT &&
          loom_low_placement_relation_can_alias(relation);
      if (!is_tied_result && !is_concat_part) {
        continue;
      }
      const uint32_t source_unit_point_start =
          unit_liveness
              ->point_starts_by_value_ordinal[relation->source_ordinal];
      const uint32_t result_unit_point_start =
          unit_liveness
              ->point_starts_by_value_ordinal[relation->result_ordinal];
      if (source_unit_point_start == UINT32_MAX ||
          result_unit_point_start == UINT32_MAX) {
        continue;
      }
      if (is_tied_result) {
        iree_bitmap_set(unit_liveness->values_with_incomplete_storage_segments,
                        relation->source_ordinal);
      }
      for (uint32_t unit_index = 0; unit_index < relation->unit_count;
           ++unit_index) {
        const iree_host_size_t source_unit_index =
            (iree_host_size_t)source_unit_point_start +
            relation->source_unit_offset + unit_index;
        const iree_host_size_t result_unit_index =
            (iree_host_size_t)result_unit_point_start +
            relation->result_unit_offset + unit_index;
        uint32_t* result_start_point =
            &unit_liveness->start_points[result_unit_index];
        *result_start_point =
            iree_min(*result_start_point,
                     unit_liveness->start_points[source_unit_index]);
      }
    }
  }
  return iree_ok_status();
}
