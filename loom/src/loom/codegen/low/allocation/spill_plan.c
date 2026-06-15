// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/spill_plan.h"

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

static uint32_t loom_low_allocation_spill_plan_round_up_to_power_of_two_u32(
    uint32_t value) {
  if (value <= 1) {
    return 1;
  }
  --value;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  return value == UINT32_MAX ? 0 : value + 1u;
}

iree_status_t loom_low_allocation_spill_plan_layout(
    const loom_low_allocation_assignment_t* assignment,
    uint16_t alloc_unit_bits, uint32_t* out_byte_size,
    uint32_t* out_byte_alignment) {
  IREE_ASSERT_ARGUMENT(assignment);
  IREE_ASSERT_ARGUMENT(out_byte_size);
  IREE_ASSERT_ARGUMENT(out_byte_alignment);
  if (alloc_unit_bits == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot plan spill for register class with zero allocation unit bits");
  }
  uint64_t bit_size = (uint64_t)assignment->unit_count * alloc_unit_bits;
  uint64_t byte_size = (bit_size + 7u) / 8u;
  if (byte_size > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "spill slot byte size exceeds uint32_t");
  }
  uint32_t unit_byte_size = ((uint32_t)alloc_unit_bits + 7u) / 8u;
  uint32_t byte_alignment =
      loom_low_allocation_spill_plan_round_up_to_power_of_two_u32(
          unit_byte_size);
  if (byte_alignment == 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "spill slot byte alignment exceeds uint32_t");
  }
  *out_byte_size = (uint32_t)byte_size;
  *out_byte_alignment = byte_alignment;
  return iree_ok_status();
}

static bool loom_low_allocation_spill_plan_use_is_removed_block_arg_edge(
    loom_use_t use, const loom_block_t* block, uint16_t arg_index) {
  const loom_op_t* user_op = loom_use_user_op(use);
  return user_op && loom_low_br_isa(user_op) &&
         loom_low_br_dest(user_op) == block &&
         loom_use_operand_index(use) == arg_index;
}

static uint32_t loom_low_allocation_spill_plan_reload_count(
    const loom_value_t* value) {
  if (!loom_value_is_block_arg(value)) {
    return value->use_count;
  }

  const loom_block_t* block = loom_value_def_block(value);
  const uint16_t arg_index = loom_value_def_index(value);
  uint32_t reload_count = 0;
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    if (!loom_low_allocation_spill_plan_use_is_removed_block_arg_edge(
            uses[i], block, arg_index)) {
      ++reload_count;
    }
  }
  return reload_count;
}

static iree_status_t loom_low_allocation_spill_plan_store_count(
    const loom_module_t* module, loom_region_t* body, loom_value_id_t value_id,
    const loom_value_t* value, uint32_t reload_count,
    uint32_t* out_store_count) {
  *out_store_count = 0;
  if (reload_count == 0) {
    return iree_ok_status();
  }
  if (!loom_value_is_block_arg(value)) {
    *out_store_count = 1;
    return iree_ok_status();
  }

  const loom_block_t* block = loom_value_def_block(value);
  const uint16_t arg_index = loom_value_def_index(value);
  if (block == loom_region_entry_block(body)) {
    *out_store_count = 1;
    return iree_ok_status();
  }

  uint32_t store_count = 0;
  loom_block_t* predecessor_block = NULL;
  loom_region_for_each_block(body, predecessor_block) {
    const loom_op_t* terminator = loom_block_const_last_op(predecessor_block);
    if (!terminator || !loom_low_br_isa(terminator) ||
        loom_low_br_dest(terminator) != block) {
      continue;
    }
    if (arg_index >= terminator->operand_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low.br predecessor payload count is stale for spilled block "
          "argument");
    }
    const loom_value_id_t payload =
        loom_op_const_operands(terminator)[arg_index];
    if (payload == LOOM_VALUE_ID_INVALID || payload >= module->values.count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low.br predecessor payload is invalid for spilled block argument");
    }
    if (payload == value_id) {
      continue;
    }
    if (store_count == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "spill store count overflow");
    }
    ++store_count;
  }
  if (store_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "spilled non-entry block argument has reloads but no incoming value "
        "to store");
  }
  *out_store_count = store_count;
  return iree_ok_status();
}

iree_status_t loom_low_allocation_spill_plan_record(
    const loom_module_t* module, loom_region_t* body,
    const loom_low_allocation_assignment_t* assignment,
    uint32_t assignment_index, uint16_t alloc_unit_bits,
    loom_low_spill_slot_space_t spill_slot_space,
    loom_low_allocation_spill_plan_t* spill_plans,
    iree_host_size_t* inout_spill_plan_count) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(body);
  IREE_ASSERT_ARGUMENT(assignment);
  IREE_ASSERT_ARGUMENT(spill_plans);
  IREE_ASSERT_ARGUMENT(inout_spill_plan_count);
  uint32_t byte_size = 0;
  uint32_t byte_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_spill_plan_layout(
      assignment, alloc_unit_bits, &byte_size, &byte_alignment));

  if (assignment->value_id >= module->values.count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot plan spill for out-of-range value %u",
                            (unsigned)assignment->value_id);
  }
  const loom_value_t* value = loom_module_value(module, assignment->value_id);
  uint32_t reload_count = loom_low_allocation_spill_plan_reload_count(value);
  uint32_t store_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_spill_plan_store_count(
      module, body, assignment->value_id, value, reload_count, &store_count));
  spill_plans[(*inout_spill_plan_count)++] = (loom_low_allocation_spill_plan_t){
      .value_id = assignment->value_id,
      .assignment_index = assignment_index,
      .slot_index = assignment->location_base,
      .slot_space = spill_slot_space,
      .byte_size = byte_size,
      .byte_alignment = byte_alignment,
      .store_count = store_count,
      .reload_count = reload_count,
  };
  return iree_ok_status();
}

void loom_low_allocation_spill_remark_record(
    loom_low_allocation_remark_t* remarks, iree_host_size_t* inout_remark_count,
    uint32_t assignment_index, uint32_t budget_units, uint32_t required_units) {
  IREE_ASSERT_ARGUMENT(remarks);
  IREE_ASSERT_ARGUMENT(inout_remark_count);
  remarks[(*inout_remark_count)++] = (loom_low_allocation_remark_t){
      .kind = LOOM_LOW_ALLOCATION_REMARK_SPILL,
      .assignment_index = assignment_index,
      .budget_units = budget_units,
      .required_units = required_units,
  };
}
