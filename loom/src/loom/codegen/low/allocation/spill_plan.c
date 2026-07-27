// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/spill_plan.h"

#include "iree/base/internal/math.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

#define LOOM_LOW_ALLOCATION_SPILL_PLAN_MAX_NATURAL_ALIGNMENT 16u

static uint32_t loom_low_allocation_spill_plan_natural_chunk_units(
    uint32_t unit_count) {
  if (unit_count >= 4) {
    return 4;
  }
  if (unit_count >= 2) {
    return 2;
  }
  return 1;
}

iree_status_t loom_low_allocation_spill_plan_layout(
    const loom_low_allocation_assignment_t* assignment,
    uint16_t alloc_unit_bits, uint32_t* out_byte_size,
    uint32_t* out_byte_alignment) {
  uint64_t bit_size = (uint64_t)assignment->unit_count * alloc_unit_bits;
  uint64_t byte_size = (bit_size + 7u) / 8u;
  if (byte_size > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "spill slot byte size exceeds uint32_t");
  }
  uint32_t unit_byte_size = ((uint32_t)alloc_unit_bits + 7u) / 8u;
  const uint32_t chunk_units =
      loom_low_allocation_spill_plan_natural_chunk_units(
          assignment->unit_count);
  uint64_t natural_alignment = (uint64_t)unit_byte_size * chunk_units;
  if (natural_alignment >
      LOOM_LOW_ALLOCATION_SPILL_PLAN_MAX_NATURAL_ALIGNMENT) {
    natural_alignment = LOOM_LOW_ALLOCATION_SPILL_PLAN_MAX_NATURAL_ALIGNMENT;
  }
  const uint32_t byte_alignment =
      iree_math_round_up_to_pow2_u32((uint32_t)natural_alignment);
  *out_byte_size = (uint32_t)byte_size;
  *out_byte_alignment = byte_alignment;
  return iree_ok_status();
}

bool loom_low_allocation_spill_plan_slice_reload_byte_offset(
    const loom_low_allocation_assignment_t* assignment,
    uint32_t spill_byte_size, const loom_op_t* slice_op, uint16_t operand_index,
    uint32_t* out_unit_byte_size, int64_t* out_reload_offset) {
  *out_unit_byte_size = 0;
  *out_reload_offset = 0;
  if (!loom_low_slice_isa(slice_op) || operand_index != 0) {
    return false;
  }
  if (assignment->unit_count == 0 ||
      spill_byte_size % assignment->unit_count != 0) {
    return false;
  }
  const uint32_t unit_byte_size = spill_byte_size / assignment->unit_count;
  if (unit_byte_size == 0) {
    return false;
  }
  const int64_t slice_offset = loom_low_slice_offset(slice_op);
  if (slice_offset < 0 || (uint64_t)slice_offset >= assignment->unit_count) {
    return false;
  }
  int64_t reload_offset = 0;
  if (!iree_checked_mul_i64(slice_offset, (int64_t)unit_byte_size,
                            &reload_offset)) {
    return false;
  }
  *out_unit_byte_size = unit_byte_size;
  *out_reload_offset = reload_offset;
  return true;
}

bool loom_low_allocation_spill_plan_use_full_slice_reload(
    uint32_t slice_count, uint64_t narrow_reload_bytes,
    uint32_t spill_byte_size) {
  return slice_count > 1 &&
         (narrow_reload_bytes > spill_byte_size ||
          slice_count >=
              LOOM_LOW_ALLOCATION_DENSE_SLICE_RELOAD_MIN_SLICE_COUNT);
}

static bool loom_low_allocation_spill_plan_use_is_removed_block_arg_edge(
    loom_use_t use, const loom_block_t* block, uint16_t arg_index) {
  const loom_op_t* user_op = loom_use_user_op(use);
  return block && loom_low_br_isa(user_op) &&
         loom_low_br_dest(user_op) == block &&
         loom_use_operand_index(use) == arg_index;
}

static void loom_low_allocation_spill_plan_value_reload_traffic(
    const loom_low_allocation_assignment_t* assignment,
    uint32_t spill_byte_size, const loom_value_t* value,
    const loom_region_t* body, uint32_t* out_reload_count,
    uint64_t* out_reload_bytes) {
  uint32_t slice_counts[64];
  uint64_t narrow_reload_bytes[64];
  uint16_t touched_block_indices[64];
  uint16_t touched_block_count = 0;
  uint64_t touched_block_mask = 0;

  const loom_block_t* block = NULL;
  uint16_t arg_index = 0;
  if (loom_value_is_block_arg(value)) {
    block = loom_value_def_block(value);
    arg_index = loom_value_def_index(value);
  }

  uint32_t reload_count = 0;
  uint64_t reload_bytes = 0;
  const bool can_group_slices = body->block_count <= 64 && value->use_count > 1;
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    if (loom_low_allocation_spill_plan_use_is_removed_block_arg_edge(
            uses[i], block, arg_index)) {
      continue;
    }

    const loom_op_t* user_op = loom_use_user_op(uses[i]);
    const uint16_t operand_index = loom_use_operand_index(uses[i]);
    uint32_t unit_byte_size = 0;
    int64_t reload_offset = 0;
    const bool is_slice_reload =
        loom_low_allocation_spill_plan_slice_reload_byte_offset(
            assignment, spill_byte_size, user_op, operand_index,
            &unit_byte_size, &reload_offset);
    (void)reload_offset;

    uint16_t use_block_index = 0;
    if (!is_slice_reload || !can_group_slices ||
        !loom_region_try_block_index(body, user_op->parent_block,
                                     &use_block_index)) {
      ++reload_count;
      reload_bytes += is_slice_reload ? unit_byte_size : spill_byte_size;
      continue;
    }

    const uint64_t use_block_bit = UINT64_C(1) << use_block_index;
    if (!iree_all_bits_set(touched_block_mask, use_block_bit)) {
      touched_block_mask |= use_block_bit;
      touched_block_indices[touched_block_count++] = use_block_index;
      slice_counts[use_block_index] = 0;
      narrow_reload_bytes[use_block_index] = 0;
    }
    ++slice_counts[use_block_index];
    narrow_reload_bytes[use_block_index] += unit_byte_size;
  }

  for (uint16_t i = 0; i < touched_block_count; ++i) {
    const uint16_t block_index = touched_block_indices[i];
    if (loom_low_allocation_spill_plan_use_full_slice_reload(
            slice_counts[block_index], narrow_reload_bytes[block_index],
            spill_byte_size)) {
      ++reload_count;
      reload_bytes += spill_byte_size;
    } else {
      reload_count += slice_counts[block_index];
      reload_bytes += narrow_reload_bytes[block_index];
    }
  }
  *out_reload_count = reload_count;
  *out_reload_bytes = reload_bytes;
}

static iree_status_t loom_low_allocation_spill_plan_value_store_count(
    const loom_cfg_graph_t* cfg_graph, loom_value_id_t value_id,
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
  if (block == loom_region_const_entry_block(cfg_graph->region)) {
    *out_store_count = 1;
    return iree_ok_status();
  }

  uint32_t store_count = 0;
  const loom_cfg_edge_index_span_t predecessor_edges =
      loom_cfg_graph_predecessor_edges(cfg_graph,
                                       loom_block_region_index(block));
  for (iree_host_size_t i = 0; i < predecessor_edges.count; ++i) {
    const loom_cfg_edge_info_t* edge =
        loom_cfg_graph_edge(cfg_graph, predecessor_edges.values[i]);
    const loom_op_t* terminator = edge->terminator;
    const loom_value_id_t payload =
        loom_op_const_operands(terminator)[arg_index];
    if (payload == value_id) {
      continue;
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

static iree_status_t loom_low_allocation_spill_plan_traffic_for_layout(
    const loom_module_t* module, const loom_cfg_graph_t* cfg_graph,
    const loom_low_allocation_assignment_t* assignment, uint32_t byte_size,
    loom_low_allocation_spill_plan_traffic_t* out_traffic) {
  *out_traffic = (loom_low_allocation_spill_plan_traffic_t){0};
  const loom_value_id_t value_id = assignment->value_id;
  const loom_value_t* value = loom_module_value(module, value_id);
  uint32_t reload_count = 0;
  uint64_t reload_bytes = 0;
  loom_low_allocation_spill_plan_value_reload_traffic(
      assignment, byte_size, value, cfg_graph->region, &reload_count,
      &reload_bytes);
  uint32_t store_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_spill_plan_value_store_count(
      cfg_graph, value_id, value, reload_count, &store_count));
  *out_traffic = (loom_low_allocation_spill_plan_traffic_t){
      .store_count = store_count,
      .store_bytes = (uint64_t)store_count * byte_size,
      .reload_count = reload_count,
      .reload_bytes = reload_bytes,
  };
  return iree_ok_status();
}

iree_status_t loom_low_allocation_spill_plan_traffic(
    const loom_module_t* module, const loom_cfg_graph_t* cfg_graph,
    const loom_low_allocation_assignment_t* assignment,
    uint16_t alloc_unit_bits,
    loom_low_allocation_spill_plan_traffic_t* out_traffic) {
  uint32_t byte_size = 0;
  uint32_t byte_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_spill_plan_layout(
      assignment, alloc_unit_bits, &byte_size, &byte_alignment));
  return loom_low_allocation_spill_plan_traffic_for_layout(
      module, cfg_graph, assignment, byte_size, out_traffic);
}

iree_status_t loom_low_allocation_spill_plan_record(
    const loom_module_t* module, const loom_cfg_graph_t* cfg_graph,
    const loom_low_allocation_assignment_t* assignment,
    uint32_t assignment_index, uint16_t alloc_unit_bits,
    loom_low_spill_slot_space_t spill_slot_space,
    loom_low_allocation_spill_plan_t* spill_plans,
    iree_host_size_t* inout_spill_plan_count) {
  uint32_t byte_size = 0;
  uint32_t byte_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_spill_plan_layout(
      assignment, alloc_unit_bits, &byte_size, &byte_alignment));

  loom_low_allocation_spill_plan_traffic_t traffic = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_spill_plan_traffic_for_layout(
      module, cfg_graph, assignment, byte_size, &traffic));
  spill_plans[(*inout_spill_plan_count)++] = (loom_low_allocation_spill_plan_t){
      .value_id = assignment->value_id,
      .assignment_index = assignment_index,
      .slot_index = assignment->location_base,
      .slot_space = spill_slot_space,
      .byte_size = byte_size,
      .byte_alignment = byte_alignment,
      .store_count = traffic.store_count,
      .reload_count = traffic.reload_count,
  };
  return iree_ok_status();
}

void loom_low_allocation_spill_remark_record(
    loom_low_allocation_remark_t* remarks, iree_host_size_t* inout_remark_count,
    uint32_t assignment_index, uint32_t budget_units, uint32_t required_units) {
  remarks[(*inout_remark_count)++] = (loom_low_allocation_remark_t){
      .kind = LOOM_LOW_ALLOCATION_REMARK_SPILL,
      .assignment_index = assignment_index,
      .budget_units = budget_units,
      .required_units = required_units,
  };
}
