// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/unit_liveness.h"

#include <string.h>

#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

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

// Records a storage use synthesized by allocation lowering rather than
// semantic liveness. Sparse semantic segments cannot prove storage conflicts
// once the allocation lifetime has been extended this way.
static iree_status_t
loom_low_allocation_unit_liveness_note_synthetic_unit_use_at_point(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_value_ordinal_t value_ordinal, uint32_t unit_offset,
    uint32_t unit_count, uint32_t point) {
  if (unit_count != 0 && value_ordinal < liveness->value_count) {
    iree_bitmap_set(unit_liveness->values_with_incomplete_storage_segments,
                    value_ordinal);
  }
  return loom_low_allocation_unit_liveness_note_unit_use_at_point(
      unit_liveness, liveness, value_ordinal, unit_offset, unit_count, point);
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
loom_low_allocation_unit_liveness_note_synthetic_value_ordinal_use_at_point(
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
  return loom_low_allocation_unit_liveness_note_synthetic_unit_use_at_point(
      unit_liveness, liveness, value_ordinal, /*unit_offset=*/0,
      interval->unit_count, point);
}

static iree_status_t
loom_low_allocation_unit_liveness_note_synthetic_value_use_at_point(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, loom_value_id_t value_id,
    uint32_t point) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_unit_liveness_value_ordinal_for_value(
          value_domain, liveness, value_id, &value_ordinal)) {
    return iree_ok_status();
  }
  return loom_low_allocation_unit_liveness_note_synthetic_value_ordinal_use_at_point(
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
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_synthetic_unit_use_at_point(
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
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_synthetic_value_ordinal_use_at_point(
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
    uint32_t point) {
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
loom_low_allocation_unit_liveness_note_operation_nested_synthetic_uses_at_point(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_liveness_operation_point_t* operation_point, uint32_t point) {
  for (uint32_t i = operation_point->direct_use_count;
       i < operation_point->use_count; ++i) {
    const loom_value_ordinal_t value_ordinal =
        loom_liveness_operation_use_ordinal(liveness,
                                            operation_point->use_start + i);
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_synthetic_value_ordinal_use_at_point(
            unit_liveness, liveness, value_ordinal, point));
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_unit_liveness_note_low_scf_for_backedge_uses(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness,
    const loom_liveness_operation_point_t* loop_point,
    uint32_t backedge_point) {
  const loom_op_t* loop_op = loop_point->op;
  const loom_region_t* loop_body = loom_low_scf_for_body(loop_op);
  const loom_block_t* body_block = loom_region_const_entry_block(loop_body);
  if (body_block == NULL || body_block->arg_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low allocation saw malformed low.scf.for body");
  }
  const loom_op_t* yield = body_block->last_op;
  if (yield == NULL || !loom_low_scf_yield_isa(yield)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation saw low.scf.for without low.scf.yield terminator");
  }

  const loom_value_slice_t iter_args = loom_low_scf_for_iter_args(loop_op);
  if (body_block->arg_count != iter_args.count + 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation saw malformed low.scf.for body arguments");
  }

  // Structured loop lowering reuses captures, control values, and loop-carried
  // body arguments after the body has executed to start the next iteration or
  // move final results.
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_unit_liveness_note_operation_nested_synthetic_uses_at_point(
          unit_liveness, liveness, loop_point, backedge_point));
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_unit_liveness_note_synthetic_value_use_at_point(
          unit_liveness, value_domain, liveness,
          loom_block_arg_id(body_block, 0), backedge_point));
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_unit_liveness_note_synthetic_value_use_at_point(
          unit_liveness, value_domain, liveness,
          loom_low_scf_for_upper_bound(loop_op), backedge_point));
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_unit_liveness_note_synthetic_value_use_at_point(
          unit_liveness, value_domain, liveness, loom_low_scf_for_step(loop_op),
          backedge_point));
  for (uint16_t i = 0; i < iter_args.count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_synthetic_value_use_at_point(
            unit_liveness, value_domain, liveness,
            loom_block_arg_id(body_block, i + 1), backedge_point));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_unit_liveness_note_operation_unit_uses(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_module_t* module, const loom_low_resolved_target_t* target,
    const loom_low_placement_table_t* placement,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, uint32_t operation_index) {
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
    if (loom_low_scf_for_isa(parent_point->op)) {
      const loom_region_t* loop_body = loom_low_scf_for_body(parent_point->op);
      const loom_block_t* body_block = loom_region_const_entry_block(loop_body);
      if (body_block != NULL && loom_block_const_last_op(body_block) == op) {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_unit_liveness_note_low_scf_for_backedge_uses(
                unit_liveness, value_domain, liveness, parent_point,
                operation_point->start_point));
      }
    }
  }
  return loom_low_allocation_unit_liveness_note_descriptor_unit_uses(
      unit_liveness, target, value_domain, liveness, op,
      operation_point->start_point);
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
    const loom_liveness_analysis_t* liveness) {
  for (iree_host_size_t i = 0; i < liveness->operation_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_operation_unit_uses(
            unit_liveness, module, target, placement, value_domain, liveness,
            (uint32_t)i));
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

  return loom_low_allocation_unit_liveness_note_body_op_unit_uses(
      out_unit_liveness, module, target, placement, value_domain, liveness);
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
    return (loom_liveness_segment_range_t){0};
  }
  return loom_liveness_segment_range_for_value_ordinal(liveness, value_ordinal);
}

iree_status_t loom_low_allocation_unit_liveness_propagate_storage_relations(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement) {
  IREE_ASSERT_ARGUMENT(unit_liveness);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(placement);
  if (unit_liveness->end_points == NULL) {
    return iree_ok_status();
  }
  // Placement relations are grouped by result value ordinal, and local value
  // ordinals follow SSA definition order. A single forward traversal therefore
  // propagates starts through tied-result and concat chains transitively.
  for (iree_host_size_t i = 0; i < placement->relation_count; ++i) {
    const loom_low_placement_relation_t* relation = &placement->relations[i];
    const bool is_tied_result =
        relation->cause == LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT;
    const bool is_concat_part =
        relation->cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT;
    if (!is_tied_result && !is_concat_part) {
      continue;
    }

    const loom_liveness_interval_t* source_interval =
        loom_liveness_interval_for_value_ordinal(liveness,
                                                 relation->source_ordinal);
    const loom_liveness_interval_t* result_interval =
        loom_liveness_interval_for_value_ordinal(liveness,
                                                 relation->result_ordinal);
    if (!source_interval || !result_interval ||
        !loom_low_allocation_live_range_interval_is_allocatable(
            source_interval) ||
        !loom_low_allocation_live_range_interval_is_allocatable(
            result_interval)) {
      continue;
    }
    if (relation->source_unit_offset > source_interval->unit_count ||
        relation->unit_count >
            source_interval->unit_count - relation->source_unit_offset ||
        relation->result_unit_offset > result_interval->unit_count ||
        relation->unit_count >
            result_interval->unit_count - relation->result_unit_offset) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low structural placement relation exceeds allocation units");
    }

    const uint32_t source_unit_point_start =
        loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
            unit_liveness, liveness, relation->source_ordinal);
    const uint32_t result_unit_point_start =
        loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
            unit_liveness, liveness, relation->result_ordinal);
    if (source_unit_point_start == UINT32_MAX ||
        result_unit_point_start == UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low structural placement relation references a value without "
          "allocation unit liveness");
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
      const uint32_t source_start_point =
          unit_liveness->start_points[source_unit_index];
      if (source_start_point < *result_start_point) {
        *result_start_point = source_start_point;
      }
      if (is_tied_result) {
        uint32_t* source_end_point =
            &unit_liveness->end_points[source_unit_index];
        const uint32_t result_end_point =
            unit_liveness->end_points[result_unit_index];
        if (*source_end_point < result_end_point) {
          *source_end_point = result_end_point;
        }
      }
    }
  }
  return iree_ok_status();
}
