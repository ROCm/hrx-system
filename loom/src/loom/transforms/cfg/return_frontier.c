// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cfg/return_frontier.h"

#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/op_defs.h"

iree_status_t loom_cfg_converge_returns(
    loom_rewriter_t* rewriter, loom_func_like_t function,
    const loom_value_fact_cfg_region_t* facts, bool* out_changed) {
  *out_changed = false;
  const loom_cfg_graph_t* graph = &facts->graph;
  const loom_region_descriptor_t* body_descriptor =
      loom_func_like_body_region_descriptor(rewriter->module, function);
  const loom_op_kind_t return_kind = body_descriptor->terminator;
  loom_op_t* first_return = NULL;
  iree_host_size_t return_count = 0;
  bool divergent = false;
  for (iree_host_size_t i = 0; i < graph->block_count; ++i) {
    if (!graph->blocks[i].reachable) {
      continue;
    }
    loom_op_t* terminator =
        (loom_op_t*)loom_block_const_last_op(graph->blocks[i].block);
    if (terminator->kind != return_kind) {
      continue;
    }
    if (!first_return) {
      first_return = terminator;
    }
    ++return_count;
    divergent |= !loom_value_facts_is_uniform_at_scope(
        loom_value_fact_control_execution(facts->control, (uint16_t)i),
        LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP);
  }
  if (return_count < 2 || !divergent) {
    return iree_ok_status();
  }

  loom_module_t* module = rewriter->module;
  loom_region_t* body = loom_func_like_body(function);
  loom_block_t* continuation = NULL;
  IREE_RETURN_IF_ERROR(
      loom_region_insert_block(module, body, body->block_count, &continuation));
  loom_builder_t* builder = &rewriter->builder;
  builder->ip.parent_op = function.op;
  loom_builder_set_block(builder, continuation);
  // Result definitions own the positional type scheme. Return operands may
  // repeat values and cannot define a one-to-one substitution for that scheme.
  const loom_value_id_t* values = loom_op_const_results(function.op);
  for (uint16_t i = 0; i < first_return->operand_count; ++i) {
    loom_value_id_t argument = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
        builder, continuation, loom_module_value_type(module, values[i]),
        &argument));
  }
  IREE_RETURN_IF_ERROR(loom_module_assign_value_types(
      module, values, continuation->arg_ids, first_return->operand_count));

  // Keep the existing return operation and its source location. Every source
  // return publishes its payload before its operation moves or is erased.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < graph->block_count && iree_status_is_ok(status); ++i) {
    if (!graph->blocks[i].reachable) {
      continue;
    }
    loom_op_t* terminator =
        (loom_op_t*)loom_block_const_last_op(graph->blocks[i].block);
    if (terminator->kind != return_kind) {
      continue;
    }
    loom_builder_set_before(builder, terminator);
    loom_op_t* branch = NULL;
    status = loom_cfg_br_build(
        builder, continuation, loom_op_const_operands(terminator),
        terminator->operand_count, terminator->location, &branch);
    if (iree_status_is_ok(status) && terminator == first_return) {
      status = loom_rewriter_move_to_block_end(rewriter, terminator,
                                               continuation, function.op);
      if (iree_status_is_ok(status)) {
        status = loom_rewriter_set_operands(rewriter, terminator,
                                            continuation->arg_ids);
      }
    } else if (iree_status_is_ok(status)) {
      status = loom_rewriter_erase(rewriter, terminator);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_refresh_cfg_facts(rewriter, body);
  }
  *out_changed = iree_status_is_ok(status);
  return status;
}
