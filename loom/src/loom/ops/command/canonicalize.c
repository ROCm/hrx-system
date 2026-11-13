// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/command/composition.h"
#include "loom/rewrite/rewriter.h"

iree_status_t loom_command_schedule_canonicalize(loom_op_t* op,
                                                 loom_rewriter_t* rewriter) {
  if (!op->parent_op) {
    return iree_ok_status();
  }
  if (loom_command_composition_mode(op->parent_op) !=
      loom_command_composition_schedule_mode(op)) {
    return iree_ok_status();
  }

  // Equal composition rules are associative. Move the existing payload in
  // order, retaining SSA identities and allocation definition points. Unlike
  // a serial child of a concurrent parent, this region adds no ordering edge
  // or execution-frame lifetime boundary.
  loom_block_t* block = loom_region_entry_block(loom_op_regions(op)[0]);
  loom_op_t* terminator = block->last_op;
  iree_status_t status = iree_ok_status();
  for (loom_op_t* child = block->first_op;
       child != terminator && iree_status_is_ok(status);) {
    loom_op_t* next = child->next_op;
    status = loom_rewriter_move_before(rewriter, child, op);
    child = next;
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_erase(rewriter, op);
  }
  return status;
}
