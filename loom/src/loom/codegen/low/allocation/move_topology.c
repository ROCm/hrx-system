// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/move_topology.h"

#include "loom/ops/low/ops.h"

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

loom_low_allocation_packet_move_op_kind_t
loom_low_allocation_move_topology_packet_move_op_kind(const loom_op_t* op) {
  IREE_ASSERT_ARGUMENT(op);
  if (loom_low_copy_isa(op)) {
    return LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_COPY;
  }
  if (loom_low_move_isa(op)) {
    return LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_MOVE;
  }
  if (loom_low_slice_isa(op)) {
    return LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_SLICE;
  }
  if (loom_low_concat_isa(op)) {
    return LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_CONCAT;
  }
  return LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_NONE;
}

bool loom_low_allocation_move_topology_op_has_packet_moves(
    const loom_op_t* op) {
  return loom_low_allocation_move_topology_packet_move_op_kind(op) !=
         LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_NONE;
}
