// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cfg/branch_folding.h"

#include "loom/analysis/condition_fact_scope.h"
#include "loom/ops/cfg/ops.h"

typedef struct loom_cfg_branch_fold_t {
  // Original conditional terminator, retained until the batch applies.
  loom_op_t* branch;
  // Existing successor selected by the original analysis snapshot.
  loom_block_t* destination;
} loom_cfg_branch_fold_t;

typedef struct loom_cfg_branch_fold_plan_t {
  // Arena-owned replacements in block order.
  loom_cfg_branch_fold_t* folds;
  // Number of selected replacements.
  iree_host_size_t count;
  // Allocated replacement capacity, zero until the first proved fold.
  iree_host_size_t capacity;
} loom_cfg_branch_fold_plan_t;

static iree_status_t loom_cfg_branch_fold_plan_append(
    loom_cfg_branch_fold_plan_t* plan, loom_op_t* branch,
    loom_block_t* destination, iree_arena_allocator_t* arena) {
  if (plan->count == plan->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, plan->count, plan->count + 1, sizeof(*plan->folds),
        &plan->capacity, (void**)&plan->folds));
  }
  plan->folds[plan->count++] = (loom_cfg_branch_fold_t){
      .branch = branch,
      .destination = destination,
  };
  return iree_ok_status();
}

static iree_status_t loom_cfg_branch_fold_plan_apply(
    loom_rewriter_t* rewriter, const loom_cfg_branch_fold_plan_t* plan,
    uint16_t* out_folded_count) {
  loom_builder_ip_t saved_ip = loom_builder_save(&rewriter->builder);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < plan->count && iree_status_is_ok(status);
       ++i) {
    const loom_cfg_branch_fold_t* fold = &plan->folds[i];
    loom_builder_set_before(&rewriter->builder, fold->branch);
    loom_op_t* new_branch = NULL;
    status = loom_cfg_br_build(&rewriter->builder, fold->destination, NULL, 0,
                               fold->branch->location, &new_branch);
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_erase(rewriter, fold->branch);
    }
  }
  loom_builder_restore(&rewriter->builder, saved_ip);
  if (iree_status_is_ok(status)) {
    *out_folded_count = (uint16_t)plan->count;
  }
  return status;
}

iree_status_t loom_cfg_fold_constant_branches(loom_rewriter_t* rewriter,
                                              loom_region_t* region,
                                              iree_arena_allocator_t* arena,
                                              uint16_t* out_folded_count) {
  *out_folded_count = 0;
  loom_cfg_branch_fold_plan_t plan = {0};
  loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    loom_op_t* branch = block->last_op;
    if (!branch || !loom_cfg_cond_br_isa(branch)) {
      continue;
    }
    loom_block_t* true_destination = loom_cfg_cond_br_true_dest(branch);
    loom_block_t* false_destination = loom_cfg_cond_br_false_dest(branch);
    loom_block_t* destination = true_destination;
    if (true_destination != false_destination) {
      bool condition = false;
      loom_value_facts_t facts = loom_value_fact_table_lookup(
          rewriter->fact_table, loom_cfg_cond_br_condition(branch));
      if (!loom_value_facts_as_exact_bool(facts, &condition)) {
        continue;
      }
      destination = condition ? true_destination : false_destination;
    }
    if (destination->arg_count != 0) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_cfg_branch_fold_plan_append(&plan, branch, destination, arena));
  }
  return loom_cfg_branch_fold_plan_apply(rewriter, &plan, out_folded_count);
}

iree_status_t loom_cfg_fold_path_sensitive_branches(
    loom_rewriter_t* rewriter, const loom_cfg_graph_t* graph,
    const loom_cfg_condition_relation_table_t* table,
    loom_condition_query_t* query, iree_arena_allocator_t* arena,
    uint16_t* out_folded_count) {
  *out_folded_count = 0;
  loom_cfg_branch_fold_plan_t plan = {0};
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    if (!loom_cfg_graph_block_is_reachable(graph, block_index)) {
      continue;
    }
    loom_op_t* branch = graph->blocks[block_index].block->last_op;
    if (!branch || !loom_cfg_cond_br_isa(branch)) {
      continue;
    }
    loom_condition_fact_scope_t scope = {0};
    loom_condition_fact_scope_initialize_indexed(
        NULL, table,
        loom_cfg_condition_relation_table_block(table, block_index), &scope);
    bool condition = false;
    bool proven = false;
    IREE_RETURN_IF_ERROR(loom_condition_fact_scope_proves_condition(
        query, rewriter->fact_table, &scope, loom_cfg_cond_br_condition(branch),
        &condition, &proven));
    if (!proven) {
      continue;
    }
    loom_block_t* destination = condition ? loom_cfg_cond_br_true_dest(branch)
                                          : loom_cfg_cond_br_false_dest(branch);
    if (destination->arg_count != 0) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_cfg_branch_fold_plan_append(&plan, branch, destination, arena));
  }
  return loom_cfg_branch_fold_plan_apply(rewriter, &plan, out_folded_count);
}
