// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/unit_liveness.h"

#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

static bool loom_low_allocation_unit_liveness_value_ordinal_for_value(
    const loom_liveness_analysis_t* liveness, loom_value_id_t value_id,
    loom_value_ordinal_t* out_value_ordinal) {
  for (iree_host_size_t i = 0; i < liveness->value_count; ++i) {
    if (liveness->value_ids[i] == value_id) {
      *out_value_ordinal = (loom_value_ordinal_t)i;
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_allocation_unit_liveness_note_unit_use_at_point(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness, loom_value_id_t value_id,
    uint32_t unit_offset, uint32_t unit_count, uint32_t point) {
  if (unit_count == 0) {
    return iree_ok_status();
  }
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_unit_liveness_value_ordinal_for_value(
          liveness, value_id, &value_ordinal)) {
    return iree_ok_status();
  }
  const uint32_t unit_end_point_start =
      loom_low_allocation_unit_liveness_end_point_start_for_value_ordinal(
          unit_liveness, liveness, value_ordinal);
  if (unit_end_point_start == UINT32_MAX) {
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
        (iree_host_size_t)unit_end_point_start + unit_offset + i;
    uint32_t* unit_end_point = &unit_liveness->end_points[unit_end_point_index];
    if (*unit_end_point < end_point) {
      *unit_end_point = end_point;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_unit_liveness_note_value_use_at_point(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness, loom_value_id_t value_id,
    uint32_t point) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_unit_liveness_value_ordinal_for_value(
          liveness, value_id, &value_ordinal)) {
    return iree_ok_status();
  }
  const loom_liveness_interval_t* interval =
      loom_liveness_interval_for_value_ordinal(liveness, value_ordinal);
  if (!interval ||
      !loom_low_allocation_live_range_interval_is_allocatable(interval)) {
    return iree_ok_status();
  }
  return loom_low_allocation_unit_liveness_note_unit_use_at_point(
      unit_liveness, liveness, value_id, /*unit_offset=*/0,
      interval->unit_count, point);
}

static iree_status_t
loom_low_allocation_unit_liveness_note_generic_op_unit_uses(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness, const loom_op_t* op,
    uint32_t point) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_value_use_at_point(
            unit_liveness, liveness, operands[i], point));
  }
  return iree_ok_status();
}

static bool
loom_low_allocation_unit_liveness_descriptor_operand_is_explicit_packet_value(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    uint16_t descriptor_operand_index) {
  IREE_ASSERT_LT(descriptor_operand_index, descriptor->operand_count);
  const loom_low_operand_t* descriptor_operand =
      &descriptor_set
           ->operands[descriptor->operand_start + descriptor_operand_index];
  return loom_low_operand_role_is_packet_operand(descriptor_operand->role) &&
         !iree_any_bit_set(descriptor_operand->flags,
                           LOOM_LOW_OPERAND_FLAG_IMPLICIT);
}

static uint16_t
loom_low_allocation_unit_liveness_descriptor_packet_operand_index(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    uint16_t descriptor_operand_index) {
  uint16_t packet_operand_index = 0;
  for (uint16_t i = descriptor->result_count; i < descriptor_operand_index;
       ++i) {
    if (loom_low_allocation_unit_liveness_descriptor_operand_is_explicit_packet_value(
            descriptor_set, descriptor, i)) {
      ++packet_operand_index;
    }
  }
  return packet_operand_index;
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
    const loom_liveness_analysis_t* liveness, const loom_op_t* op,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    uint16_t early_clobber_result_index, uint32_t clobber_point) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    if (!loom_low_allocation_unit_liveness_descriptor_operand_is_explicit_packet_value(
            descriptor_set, descriptor, i)) {
      continue;
    }
    const uint16_t operand_index =
        loom_low_allocation_unit_liveness_descriptor_packet_operand_index(
            descriptor_set, descriptor, i);
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
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_value_use_at_point(
            unit_liveness, liveness, operands[operand_index], clobber_point));
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_unit_liveness_note_descriptor_unit_uses(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_module_t* module, const loom_low_resolved_target_t* target,
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
  loom_low_resolved_descriptor_packet_t packet = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_resolve_descriptor_packet(module, target, op, &packet));
  if (packet.descriptor == NULL) {
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
            unit_liveness, liveness, op, descriptor_set, descriptor,
            constraint->lhs_operand_index, clobber_point));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_unit_liveness_note_slice_unit_uses(
    loom_low_allocation_unit_liveness_t* unit_liveness,
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
          liveness, loom_low_slice_result(op), &result_ordinal)) {
    return iree_ok_status();
  }
  const loom_liveness_interval_t* result_interval =
      loom_liveness_interval_for_value_ordinal(liveness, result_ordinal);
  if (!result_interval ||
      !loom_low_allocation_live_range_interval_is_allocatable(
          result_interval)) {
    return iree_ok_status();
  }
  return loom_low_allocation_unit_liveness_note_unit_use_at_point(
      unit_liveness, liveness, loom_low_slice_source(op), (uint32_t)offset,
      result_interval->unit_count, point);
}

static bool loom_low_allocation_unit_liveness_region_is_nested_in_op(
    const loom_op_t* owner_op, const loom_region_t* query_region) {
  if (owner_op == NULL || query_region == NULL) {
    return false;
  }
  loom_region_t* const* regions = loom_op_regions(owner_op);
  for (uint8_t i = 0; i < owner_op->region_count; ++i) {
    if (regions[i] == query_region) {
      return true;
    }
    const loom_block_t* block = NULL;
    loom_region_for_each_block(regions[i], block) {
      const loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        if (loom_low_allocation_unit_liveness_region_is_nested_in_op(
                op, query_region)) {
          return true;
        }
      }
    }
  }
  return false;
}

static bool loom_low_allocation_unit_liveness_block_is_nested_in_op(
    const loom_op_t* owner_op, const loom_block_t* query_block) {
  return query_block != NULL &&
         loom_low_allocation_unit_liveness_region_is_nested_in_op(
             owner_op, query_block->parent_region);
}

static bool loom_low_allocation_unit_liveness_value_is_defined_inside_op(
    const loom_module_t* module, const loom_op_t* owner_op,
    loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return loom_low_allocation_unit_liveness_block_is_nested_in_op(
        owner_op, loom_value_def_block(value));
  }
  const loom_op_t* def_op = loom_value_def_op(value);
  while (def_op != NULL) {
    if (def_op == owner_op) {
      return true;
    }
    def_op = def_op->parent_op;
  }
  return false;
}

static iree_status_t
loom_low_allocation_unit_liveness_note_low_scf_for_capture_uses(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_module_t* module, const loom_liveness_analysis_t* liveness,
    const loom_op_t* loop_op, const loom_region_t* region,
    uint32_t backedge_point) {
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_value_id_t* operands = loom_op_const_operands(op);
      for (uint16_t i = 0; i < op->operand_count; ++i) {
        if (loom_low_allocation_unit_liveness_value_is_defined_inside_op(
                module, loop_op, operands[i])) {
          continue;
        }
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_unit_liveness_note_value_use_at_point(
                unit_liveness, liveness, operands[i], backedge_point));
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_unit_liveness_note_low_scf_for_capture_uses(
                unit_liveness, module, liveness, loop_op, regions[i],
                backedge_point));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_unit_liveness_note_low_scf_for_backedge_uses(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_module_t* module, const loom_liveness_analysis_t* liveness,
    const loom_region_t* body, loom_liveness_order_t liveness_order,
    const loom_op_t* op) {
  if (!loom_liveness_analysis_includes_region_tree(liveness)) {
    return iree_ok_status();
  }
  const loom_region_t* loop_body = loom_low_scf_for_body(op);
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

  const loom_value_slice_t iter_args = loom_low_scf_for_iter_args(op);
  if (body_block->arg_count != iter_args.count + 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation saw malformed low.scf.for body arguments");
  }

  uint32_t backedge_point = UINT32_MAX;
  IREE_RETURN_IF_ERROR(loom_low_allocation_live_range_ordered_op_program_point(
      liveness, body, liveness_order, yield, &backedge_point));

  // Structured loop lowering reuses captures, control values, and loop-carried
  // body arguments after the body has executed to start the next iteration or
  // move final results.
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_unit_liveness_note_low_scf_for_capture_uses(
          unit_liveness, module, liveness, op, loop_body, backedge_point));
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_unit_liveness_note_value_use_at_point(
          unit_liveness, liveness, loom_block_arg_id(body_block, 0),
          backedge_point));
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_unit_liveness_note_value_use_at_point(
          unit_liveness, liveness, loom_low_scf_for_upper_bound(op),
          backedge_point));
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_unit_liveness_note_value_use_at_point(
          unit_liveness, liveness, loom_low_scf_for_step(op), backedge_point));
  for (uint16_t i = 0; i < iter_args.count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_value_use_at_point(
            unit_liveness, liveness, loom_block_arg_id(body_block, i + 1),
            backedge_point));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_unit_liveness_note_op_unit_uses(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_module_t* module, loom_region_t* body,
    const loom_low_resolved_target_t* target,
    loom_liveness_order_t liveness_order,
    const loom_liveness_analysis_t* liveness, const loom_op_t* op) {
  uint32_t point = UINT32_MAX;
  IREE_RETURN_IF_ERROR(loom_low_allocation_live_range_ordered_op_program_point(
      liveness, body, liveness_order, op, &point));
  if (loom_low_slice_isa(op)) {
    return loom_low_allocation_unit_liveness_note_slice_unit_uses(
        unit_liveness, liveness, op, point);
  }
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_unit_liveness_note_generic_op_unit_uses(
          unit_liveness, liveness, op, point));
  if (loom_low_scf_for_isa(op)) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_low_scf_for_backedge_uses(
            unit_liveness, module, liveness, body, liveness_order, op));
  }
  return loom_low_allocation_unit_liveness_note_descriptor_unit_uses(
      unit_liveness, module, target, liveness, op, point);
}

static iree_status_t
loom_low_allocation_unit_liveness_note_value_unit_uses_at_point(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness, const loom_value_id_t* values,
    iree_host_size_t value_count, uint32_t point) {
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_value_use_at_point(
            unit_liveness, liveness, values[i], point));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_unit_liveness_note_block_boundary_uses(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness) {
  for (iree_host_size_t i = 0; i < liveness->block_count; ++i) {
    const loom_liveness_block_info_t* block_info = &liveness->blocks[i];
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_value_unit_uses_at_point(
            unit_liveness, liveness, block_info->live_in_values,
            block_info->live_in_count, block_info->start_point));
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_unit_liveness_note_value_unit_uses_at_point(
            unit_liveness, liveness, block_info->live_out_values,
            block_info->live_out_count, block_info->end_point));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_unit_liveness_note_region_op_unit_uses(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_module_t* module, loom_region_t* body,
    const loom_low_resolved_target_t* target,
    loom_liveness_order_t liveness_order,
    const loom_liveness_analysis_t* liveness, const loom_region_t* region) {
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(loom_low_allocation_unit_liveness_note_op_unit_uses(
          unit_liveness, module, body, target, liveness_order, liveness, op));
      if (!loom_liveness_analysis_includes_region_tree(liveness)) {
        continue;
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (regions[i] == NULL) {
          continue;
        }
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_unit_liveness_note_region_op_unit_uses(
                unit_liveness, module, body, target, liveness_order, liveness,
                regions[i]));
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_unit_liveness_initialize(
    const loom_module_t* module, loom_region_t* body,
    const loom_low_resolved_target_t* target,
    loom_liveness_order_t liveness_order,
    const loom_liveness_analysis_t* liveness, iree_arena_allocator_t* arena,
    loom_low_allocation_unit_liveness_t* out_unit_liveness) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(body);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_unit_liveness);
  *out_unit_liveness = (loom_low_allocation_unit_liveness_t){0};

  if (liveness->value_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, liveness->value_count,
        sizeof(*out_unit_liveness->end_point_starts_by_value_ordinal),
        (void**)&out_unit_liveness->end_point_starts_by_value_ordinal));
    for (iree_host_size_t i = 0; i < liveness->value_count; ++i) {
      out_unit_liveness->end_point_starts_by_value_ordinal[i] = UINT32_MAX;
    }
  }

  iree_host_size_t unit_end_point_count = 0;
  for (iree_host_size_t i = 0; i < liveness->value_count; ++i) {
    const loom_liveness_interval_t* interval =
        loom_liveness_interval_for_value_ordinal(liveness,
                                                 (loom_value_ordinal_t)i);
    if (!interval ||
        !loom_low_allocation_live_range_interval_is_allocatable(interval)) {
      continue;
    }
    if (interval->unit_count > IREE_HOST_SIZE_MAX - unit_end_point_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low allocation unit liveness count exceeds host size");
    }
    unit_end_point_count += interval->unit_count;
  }
  if (unit_end_point_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, unit_end_point_count, sizeof(*out_unit_liveness->end_points),
      (void**)&out_unit_liveness->end_points));
  out_unit_liveness->end_point_count = unit_end_point_count;

  iree_host_size_t unit_end_point_start = 0;
  for (iree_host_size_t i = 0; i < liveness->value_count; ++i) {
    const loom_liveness_interval_t* interval =
        loom_liveness_interval_for_value_ordinal(liveness,
                                                 (loom_value_ordinal_t)i);
    if (!interval ||
        !loom_low_allocation_live_range_interval_is_allocatable(interval)) {
      continue;
    }
    if (unit_end_point_start > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low allocation unit liveness start exceeds u32 range");
    }
    out_unit_liveness->end_point_starts_by_value_ordinal[i] =
        (uint32_t)unit_end_point_start;
    for (uint32_t unit_index = 0; unit_index < interval->unit_count;
         ++unit_index) {
      out_unit_liveness->end_points[unit_end_point_start + unit_index] =
          loom_low_allocation_live_range_interval_initial_unit_end_point(
              interval);
    }
    unit_end_point_start += interval->unit_count;
  }

  // Unit liveness refines the value-granular analysis inside blocks so
  // operations like low.slice can release dead units independently. CFG
  // boundaries are still value-granular: every unit of a block live-in/out
  // value must stay reserved across the boundary until a per-unit dataflow
  // analysis can prove otherwise.
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_unit_liveness_note_block_boundary_uses(
          out_unit_liveness, liveness));

  return loom_low_allocation_unit_liveness_note_region_op_unit_uses(
      out_unit_liveness, module, body, target, liveness_order, liveness, body);
}

uint32_t loom_low_allocation_unit_liveness_end_point_start_for_value_ordinal(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_value_ordinal_t value_ordinal) {
  IREE_ASSERT_ARGUMENT(unit_liveness);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_LT(value_ordinal, liveness->value_count);
  if (unit_liveness->end_point_starts_by_value_ordinal == NULL) {
    return UINT32_MAX;
  }
  return unit_liveness->end_point_starts_by_value_ordinal[value_ordinal];
}

iree_status_t loom_low_allocation_unit_liveness_extend_for_tied_results(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement) {
  IREE_ASSERT_ARGUMENT(unit_liveness);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(placement);
  if (unit_liveness->end_points == NULL) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < placement->relation_count; ++i) {
    const loom_low_placement_relation_t* relation = &placement->relations[i];
    if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT) {
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
          "low tied-result placement relation exceeds allocation units");
    }

    const uint32_t source_unit_end_point_start =
        loom_low_allocation_unit_liveness_end_point_start_for_value_ordinal(
            unit_liveness, liveness, relation->source_ordinal);
    const uint32_t result_unit_end_point_start =
        loom_low_allocation_unit_liveness_end_point_start_for_value_ordinal(
            unit_liveness, liveness, relation->result_ordinal);
    if (source_unit_end_point_start == UINT32_MAX ||
        result_unit_end_point_start == UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low tied-result placement relation references a value without "
          "allocation unit liveness");
    }

    for (uint32_t unit_index = 0; unit_index < relation->unit_count;
         ++unit_index) {
      const iree_host_size_t source_unit_index =
          (iree_host_size_t)source_unit_end_point_start +
          relation->source_unit_offset + unit_index;
      const iree_host_size_t result_unit_index =
          (iree_host_size_t)result_unit_end_point_start +
          relation->result_unit_offset + unit_index;
      uint32_t* source_end_point =
          &unit_liveness->end_points[source_unit_index];
      const uint32_t result_end_point =
          unit_liveness->end_points[result_unit_index];
      if (*source_end_point < result_end_point) {
        *source_end_point = result_end_point;
      }
    }
  }
  return iree_ok_status();
}
