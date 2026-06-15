// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/facts.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/rewrite/rewriter.h"

static loom_op_t* loom_kernel_region_return(loom_region_t* region) {
  if (!region || region->block_count != 1) return NULL;
  loom_block_t* block = loom_region_entry_block(region);
  if (!block || !block->last_op || !loom_kernel_return_isa(block->last_op)) {
    return NULL;
  }
  return block->last_op;
}

static iree_status_t loom_kernel_move_region_body_before_op(
    loom_rewriter_t* rewriter, loom_region_t* region, loom_op_t* old_return,
    loom_op_t* before_op) {
  loom_block_t* block = loom_region_entry_block(region);
  if (!block) return iree_ok_status();
  loom_op_t* child_op = block->first_op;
  while (child_op && child_op != old_return) {
    loom_op_t* next_child_op = child_op->next_op;
    IREE_RETURN_IF_ERROR(
        loom_rewriter_move_before(rewriter, child_op, before_op));
    child_op = next_child_op;
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_move_tail_before_op(loom_rewriter_t* rewriter,
                                                     loom_op_t* exit_op,
                                                     loom_op_t* final_return,
                                                     loom_op_t* before_op) {
  loom_op_t* tail_op = exit_op->next_op;
  while (tail_op && tail_op != final_return) {
    loom_op_t* next_tail_op = tail_op->next_op;
    IREE_RETURN_IF_ERROR(
        loom_rewriter_move_before(rewriter, tail_op, before_op));
    tail_op = next_tail_op;
  }
  return iree_ok_status();
}

iree_status_t loom_kernel_exit_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  bool condition = false;
  if (loom_value_facts_as_exact_bool(
          loom_rewriter_value_facts(rewriter, loom_kernel_exit_condition(op)),
          &condition) &&
      !condition) {
    return loom_rewriter_erase(rewriter, op);
  }

  if (!op->parent_block || !op->parent_block->last_op ||
      !loom_kernel_return_isa(op->parent_block->last_op)) {
    return iree_ok_status();
  }
  loom_op_t* final_return = op->parent_block->last_op;
  if (op == final_return) return iree_ok_status();

  loom_region_t* body = loom_kernel_exit_body(op);
  loom_op_t* body_return = NULL;
  if (body) {
    body_return = loom_kernel_region_return(body);
    if (!body_return) return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_op_t* if_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_if_build(
      &rewriter->builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
      loom_kernel_exit_condition(op), NULL, 0, NULL, 0, op->location, &if_op));

  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &rewriter->builder, if_op, loom_scf_if_then_region(if_op));
  loom_op_t* then_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&rewriter->builder, NULL, 0,
                                            op->location, &then_yield));
  loom_builder_restore(&rewriter->builder, saved_ip);

  saved_ip = loom_builder_enter_region(&rewriter->builder, if_op,
                                       loom_scf_if_else_region(if_op));
  loom_op_t* else_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&rewriter->builder, NULL, 0,
                                            op->location, &else_yield));
  loom_builder_restore(&rewriter->builder, saved_ip);

  if (body) {
    IREE_RETURN_IF_ERROR(loom_kernel_move_region_body_before_op(
        rewriter, body, body_return, then_yield));
  }
  IREE_RETURN_IF_ERROR(
      loom_kernel_move_tail_before_op(rewriter, op, final_return, else_yield));
  return loom_rewriter_erase(rewriter, op);
}
