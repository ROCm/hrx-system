// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/move_topology.h"

#include "loom/ops/low/ops.h"

iree_host_size_t loom_low_allocation_move_topology_count_copy_ops(
    const loom_region_t* body) {
  IREE_ASSERT_ARGUMENT(body);
  iree_host_size_t copy_count = 0;
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_copy_isa(op)) {
        ++copy_count;
      }
    }
  }
  return copy_count;
}

bool loom_low_allocation_move_topology_concat_requires_packet_materialization_for_module(
    const loom_module_t* module, const loom_op_t* op) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(op);
  if (!loom_low_concat_isa(op)) {
    return true;
  }
  const loom_value_t* result =
      loom_module_value(module, loom_low_concat_result(op));
  const loom_use_t* use = NULL;
  loom_value_for_each_use(result, use) {
    if (!loom_low_br_isa(loom_use_user_op(*use))) {
      return true;
    }
  }
  return false;
}

bool loom_low_allocation_move_topology_concat_requires_packet_materialization(
    const loom_low_allocation_table_t* table, const loom_op_t* op) {
  IREE_ASSERT_ARGUMENT(table);
  return loom_low_allocation_move_topology_concat_requires_packet_materialization_for_module(
      table->module, op);
}

bool loom_low_allocation_move_topology_op_has_packet_moves(
    const loom_op_t* op) {
  IREE_ASSERT_ARGUMENT(op);
  return loom_low_copy_isa(op) || loom_low_slice_isa(op) ||
         loom_low_concat_isa(op);
}
