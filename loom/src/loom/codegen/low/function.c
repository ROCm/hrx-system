// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/function.h"

#include "loom/ops/low/ops.h"
#include "loom/ops/low/storage.h"

bool loom_low_function_def_isa(const loom_op_t* op) {
  return loom_low_func_def_isa(op) || loom_low_kernel_def_isa(op);
}

loom_symbol_ref_t loom_low_function_callee(const loom_op_t* function_op) {
  if (loom_low_func_def_isa(function_op)) {
    return loom_low_func_def_callee(function_op);
  }
  if (loom_low_kernel_def_isa(function_op)) {
    return loom_low_kernel_def_callee(function_op);
  }
  return loom_symbol_ref_null();
}

loom_symbol_ref_t loom_low_function_target(const loom_op_t* function_op) {
  if (loom_low_func_def_isa(function_op)) {
    return loom_low_func_def_target(function_op);
  }
  if (loom_low_kernel_def_isa(function_op)) {
    return loom_low_kernel_def_target(function_op);
  }
  return loom_symbol_ref_null();
}

uint8_t loom_low_function_allocation(const loom_op_t* function_op) {
  if (loom_low_func_def_isa(function_op)) {
    return loom_low_func_def_allocation(function_op);
  }
  if (loom_low_kernel_def_isa(function_op)) {
    return loom_low_kernel_def_allocation(function_op);
  }
  return 0;
}

uint8_t loom_low_function_schedule(const loom_op_t* function_op) {
  if (loom_low_func_def_isa(function_op)) {
    return loom_low_func_def_schedule(function_op);
  }
  if (loom_low_kernel_def_isa(function_op)) {
    return loom_low_kernel_def_schedule(function_op);
  }
  return 0;
}

loom_named_attr_slice_t loom_low_function_abi_layout(
    const loom_op_t* function_op) {
  if (loom_low_func_def_isa(function_op)) {
    return loom_low_func_def_abi_layout(function_op);
  }
  if (loom_low_kernel_def_isa(function_op)) {
    return loom_low_kernel_def_abi_layout(function_op);
  }
  return loom_named_attr_slice_empty();
}

loom_region_t* loom_low_function_body(loom_op_t* function_op) {
  if (loom_low_func_def_isa(function_op)) {
    return loom_low_func_def_body(function_op);
  }
  if (loom_low_kernel_def_isa(function_op)) {
    return loom_low_kernel_def_body(function_op);
  }
  return NULL;
}

const loom_region_t* loom_low_function_const_body(
    const loom_op_t* function_op) {
  if (loom_low_func_def_isa(function_op)) {
    return loom_low_func_def_body(function_op);
  }
  if (loom_low_kernel_def_isa(function_op)) {
    return loom_low_kernel_def_body(function_op);
  }
  return NULL;
}

bool loom_low_function_entry_preamble_op(const loom_op_t* op) {
  return loom_low_live_in_isa(op) || loom_low_resource_isa(op);
}

bool loom_low_function_entry_traffic_op(const loom_op_t* op) {
  return loom_low_function_entry_preamble_op(op) ||
         loom_low_storage_store_isa(op);
}

iree_status_t loom_low_function_materialize_schedule(
    loom_builder_t* builder, uint8_t schedule, loom_block_t* const* blocks,
    uint16_t block_count) {
  if (schedule != LOOM_LOW_SCHEDULE_LOCKED &&
      schedule != LOOM_LOW_SCHEDULE_PHASED) {
    return iree_ok_status();
  }

  loom_op_t* entry_first_op = blocks[0]->first_op;
  while (loom_low_function_entry_preamble_op(entry_first_op)) {
    entry_first_op = entry_first_op->next_op;
  }
  loom_op_t* control_op = NULL;
  if (schedule == LOOM_LOW_SCHEDULE_PHASED) {
    loom_builder_set_before(builder, entry_first_op);
    IREE_RETURN_IF_ERROR(loom_low_schedule_begin_build(
        builder, entry_first_op->location, &control_op));
    for (uint16_t block_index = 0; block_index < block_count; ++block_index) {
      loom_op_t* terminator = blocks[block_index]->last_op;
      if (!loom_low_return_isa(terminator)) {
        continue;
      }
      loom_builder_set_before(builder, terminator);
      IREE_RETURN_IF_ERROR(loom_low_schedule_end_build(
          builder, terminator->location, &control_op));
    }
    return iree_ok_status();
  }

  for (uint16_t block_index = 0; block_index < block_count; ++block_index) {
    loom_block_t* block = blocks[block_index];
    loom_op_t* terminator = block->last_op;
    loom_op_t* first_op = block_index == 0 ? entry_first_op : block->first_op;
    if (first_op == terminator) {
      continue;
    }
    loom_builder_set_before(builder, first_op);
    IREE_RETURN_IF_ERROR(loom_low_schedule_fence_build(
        builder, first_op->location, &control_op));
    for (loom_op_t* op = first_op; op != terminator;) {
      loom_op_t* next_op = op->next_op;
      loom_builder_set_after(builder, op);
      IREE_RETURN_IF_ERROR(
          loom_low_schedule_fence_build(builder, op->location, &control_op));
      op = next_op;
    }
  }
  return iree_ok_status();
}
