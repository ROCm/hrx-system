// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/facts.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/rewrite/rewriter.h"

static const loom_op_t* loom_kernel_enclosing_def(const loom_op_t* op) {
  for (const loom_op_t* ancestor = op ? op->parent_op : NULL; ancestor;
       ancestor = ancestor->parent_op) {
    if (loom_kernel_def_isa(ancestor)) return ancestor;
  }
  return NULL;
}

static bool loom_kernel_has_trivial_workgroup_cluster(
    const loom_op_t* op, const loom_rewriter_t* rewriter) {
  const loom_op_t* kernel_op = loom_kernel_enclosing_def(op);
  if (!kernel_op) return false;
  const loom_op_t* launch_config = loom_kernel_def_launch_config_op(kernel_op);
  if (!launch_config) return false;
  if (!loom_kernel_launch_config_has_workgroup_cluster_size(launch_config)) {
    return true;
  }

  loom_target_workgroup_cluster_size_t cluster_size = {0};
  return loom_kernel_def_static_workgroup_cluster_size_from_facts(
             rewriter->module, kernel_op, rewriter->fact_table,
             &cluster_size) &&
         cluster_size.x == 1 && cluster_size.y == 1 && cluster_size.z == 1;
}

static iree_status_t loom_kernel_replace_single_result(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t replacement,
    loom_value_id_t value_checkpoint) {
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static iree_status_t loom_kernel_replace_single_result_with_index_constant(
    loom_op_t* op, loom_rewriter_t* rewriter, int64_t value) {
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_op_const_results(op)[0]);
  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(&rewriter->builder, loom_attr_i64(value),
                                result_type, op->location, &constant_op));
  return loom_kernel_replace_single_result(
      op, rewriter, loom_index_constant_result(constant_op), value_checkpoint);
}

static iree_status_t loom_kernel_replace_cluster_id_with_workgroup_id(
    loom_op_t* op, loom_rewriter_t* rewriter,
    loom_kernel_dimension_t dimension) {
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_op_const_results(op)[0]);
  loom_op_t* workgroup_id_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_kernel_workgroup_id_build(&rewriter->builder, dimension, result_type,
                                     op->location, &workgroup_id_op));
  return loom_kernel_replace_single_result(
      op, rewriter, loom_kernel_workgroup_id_result(workgroup_id_op),
      value_checkpoint);
}

static iree_status_t loom_kernel_replace_cluster_count_with_workgroup_count(
    loom_op_t* op, loom_rewriter_t* rewriter,
    loom_kernel_dimension_t dimension) {
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_op_const_results(op)[0]);
  loom_op_t* workgroup_count_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_workgroup_count_build(
      &rewriter->builder, dimension, result_type, op->location,
      &workgroup_count_op));
  return loom_kernel_replace_single_result(
      op, rewriter, loom_kernel_workgroup_count_result(workgroup_count_op),
      value_checkpoint);
}

iree_status_t loom_kernel_cluster_id_canonicalize(loom_op_t* op,
                                                  loom_rewriter_t* rewriter) {
  if (!loom_kernel_has_trivial_workgroup_cluster(op, rewriter)) {
    return iree_ok_status();
  }
  return loom_kernel_replace_cluster_id_with_workgroup_id(
      op, rewriter, loom_kernel_cluster_id_dimension(op));
}

iree_status_t loom_kernel_cluster_workgroup_id_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  if (!loom_kernel_has_trivial_workgroup_cluster(op, rewriter)) {
    return iree_ok_status();
  }
  return loom_kernel_replace_single_result_with_index_constant(op, rewriter, 0);
}

iree_status_t loom_kernel_cluster_workgroup_flat_id_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  if (!loom_kernel_has_trivial_workgroup_cluster(op, rewriter)) {
    return iree_ok_status();
  }
  return loom_kernel_replace_single_result_with_index_constant(op, rewriter, 0);
}

iree_status_t loom_kernel_cluster_size_canonicalize(loom_op_t* op,
                                                    loom_rewriter_t* rewriter) {
  if (!loom_kernel_has_trivial_workgroup_cluster(op, rewriter)) {
    return iree_ok_status();
  }
  return loom_kernel_replace_single_result_with_index_constant(op, rewriter, 1);
}

iree_status_t loom_kernel_cluster_count_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  if (!loom_kernel_has_trivial_workgroup_cluster(op, rewriter)) {
    return iree_ok_status();
  }
  return loom_kernel_replace_cluster_count_with_workgroup_count(
      op, rewriter, loom_kernel_cluster_count_dimension(op));
}

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
