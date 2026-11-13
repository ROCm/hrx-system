// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/ownership_cleanup.h"

#include "loom/ir/module.h"

typedef struct loom_ownership_cleanup_split_edge_t {
  // Terminator whose successor is redirected through |block|.
  loom_op_t* terminator;
  // Successor ordinal redirected through |block|.
  uint16_t successor_index;
  // Split block containing edge-local lifetime operations.
  loom_block_t* block;
  // Branch from |block| to the original successor.
  loom_op_t* branch;
} loom_ownership_cleanup_split_edge_t;

typedef struct loom_ownership_cleanup_context_t {
  // Mutable module receiving executable releases and split blocks.
  loom_module_t* module;
  // Function whose source snapshot produced this plan.
  loom_func_like_t function;
  // Release and branch builders selected by the caller.
  const loom_ownership_lifetime_options_t* options;
  // Accumulated materialization counters.
  loom_ownership_lifetime_result_t* result;
} loom_ownership_cleanup_context_t;

static iree_status_t loom_ownership_cleanup_build_release(
    loom_ownership_cleanup_context_t* state, loom_builder_t* builder,
    const loom_ownership_cleanup_action_t* action, loom_op_t** out_op) {
  const loom_ownership_lifetime_policy_t* policy =
      &state->options->policies[action->policy_index];
  return policy->build_release(builder, action->value_id, action->location,
                               policy->user_data, out_op);
}

static iree_status_t loom_ownership_cleanup_start_split_edge(
    loom_ownership_cleanup_context_t* state,
    const loom_ownership_cleanup_action_t* action, loom_builder_t* builder,
    loom_ownership_cleanup_split_edge_t* split_edge) {
  loom_block_t** successors = loom_op_successors(action->terminator);
  loom_block_t* original_successor = successors[action->successor_index];
  loom_region_t* region = action->terminator->parent_block->parent_region;
  loom_block_t* split_block = NULL;
  IREE_RETURN_IF_ERROR(
      loom_region_append_block(state->module, region, &split_block));
  successors[action->successor_index] = split_block;

  loom_builder_set_block(builder, split_block);
  builder->ip.parent_op = action->terminator->parent_op;
  loom_op_t* branch = NULL;
  IREE_RETURN_IF_ERROR(state->options->build_branch(builder, original_successor,
                                                    action->location, &branch));
  *split_edge = (loom_ownership_cleanup_split_edge_t){
      .terminator = action->terminator,
      .successor_index = action->successor_index,
      .block = split_block,
      .branch = branch,
  };
  ++state->result->edges_split;
  return iree_ok_status();
}

static bool loom_ownership_cleanup_split_edge_matches(
    const loom_ownership_cleanup_split_edge_t* split_edge,
    const loom_ownership_cleanup_action_t* action) {
  return split_edge->terminator == action->terminator &&
         split_edge->successor_index == action->successor_index;
}

static iree_status_t loom_ownership_cleanup_apply_action(
    loom_ownership_cleanup_context_t* state, loom_builder_t* builder,
    const loom_ownership_cleanup_action_t* action,
    loom_ownership_cleanup_split_edge_t* current_split_edge) {
  if (action->successor_index != UINT16_MAX) {
    if (action->terminator->successor_count > 1) {
      if (!loom_ownership_cleanup_split_edge_matches(current_split_edge,
                                                     action)) {
        IREE_RETURN_IF_ERROR(loom_ownership_cleanup_start_split_edge(
            state, action, builder, current_split_edge));
      }
      loom_builder_set_before(builder, current_split_edge->branch);
    } else {
      loom_builder_set_before(builder, action->terminator);
    }
  } else if (action->before_op) {
    loom_builder_set_before(builder, action->before_op);
  } else {
    loom_builder_set_block(builder, action->block);
    builder->ip.parent_op = state->function.op;
  }

  loom_op_t* materialized_op = NULL;
  IREE_RETURN_IF_ERROR(loom_ownership_cleanup_build_release(
      state, builder, action, &materialized_op));
  ++state->result->releases_inserted;
  return iree_ok_status();
}

iree_status_t loom_ownership_cleanup_apply(
    loom_module_t* module, loom_func_like_t function,
    const loom_ownership_lifetime_options_t* options,
    const loom_ownership_cleanup_action_t* actions,
    iree_host_size_t action_count, loom_ownership_lifetime_result_t* result) {
  loom_ownership_cleanup_context_t context = {module, function, options,
                                              result};
  loom_builder_t builder = {0};
  loom_builder_initialize(
      module, &module->arena,
      loom_region_entry_block(loom_func_like_body(function)), &builder);
  builder.ip.parent_op = function.op;
  loom_ownership_cleanup_split_edge_t current_split_edge = {0};
  for (iree_host_size_t i = 0; i < action_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ownership_cleanup_apply_action(
        &context, &builder, &actions[i], &current_split_edge));
  }
  return iree_ok_status();
}
