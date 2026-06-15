// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cfg/branch_sink.h"

#include "loom/analysis/motion.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/rewriter.h"

#define LOOM_BRANCH_SINK_STATISTICS(V, statistics_type)                  \
  V(statistics_type, branches_visited, "branches-visited",               \
    "Number of branch operations inspected.")                            \
  V(statistics_type, ops_sunk, "ops-sunk", "Number of operations sunk.") \
  V(statistics_type, selectors_sunk, "selectors-sunk",                   \
    "Number of branch selector operations sunk.")

LOOM_PASS_STATISTICS_DEFINE(loom_branch_sink_statistics,
                            loom_branch_sink_statistics_t,
                            LOOM_BRANCH_SINK_STATISTICS)

static const loom_pass_info_t loom_branch_sink_pass_info_storage = {
    .name = IREE_SVL("branch-sink"),
    .description = IREE_SVL("Sink branch-local pure producers into branches."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_branch_sink_statistics_layout,
};

const loom_pass_info_t* loom_branch_sink_pass_info(void) {
  return &loom_branch_sink_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Scratch stacks
//===----------------------------------------------------------------------===//

#define LOOM_BRANCH_SINK_INITIAL_REGION_STACK_CAPACITY 16

typedef struct loom_branch_sink_region_stack_t {
  // Region pointers waiting to be processed.
  loom_region_t** regions;
  // Number of queued region pointers.
  iree_host_size_t count;
  // Allocated region pointer capacity.
  iree_host_size_t capacity;
} loom_branch_sink_region_stack_t;

static iree_status_t loom_branch_sink_region_stack_initialize(
    iree_arena_allocator_t* arena, loom_branch_sink_region_stack_t* stack) {
  stack->count = 0;
  stack->capacity = LOOM_BRANCH_SINK_INITIAL_REGION_STACK_CAPACITY;
  return iree_arena_allocate_array(
      arena, stack->capacity, sizeof(loom_region_t*), (void**)&stack->regions);
}

static iree_status_t loom_branch_sink_region_stack_push(
    iree_arena_allocator_t* arena, loom_branch_sink_region_stack_t* stack,
    loom_region_t* region) {
  if (!region || region->block_count == 0) return iree_ok_status();
  if (stack->count >= stack->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, stack->count, stack->count + 1, sizeof(loom_region_t*),
        &stack->capacity, (void**)&stack->regions));
  }
  stack->regions[stack->count++] = region;
  return iree_ok_status();
}

static loom_region_t* loom_branch_sink_region_stack_pop(
    loom_branch_sink_region_stack_t* stack) {
  return stack->count > 0 ? stack->regions[--stack->count] : NULL;
}

typedef struct loom_branch_sink_context_t {
  // Pass instance owning the scratch arena.
  loom_pass_t* pass;
  // Typed statistics storage for the current pass invocation.
  loom_branch_sink_statistics_t* statistics;
  // Module being transformed.
  loom_module_t* module;
  // Rewriter used for IR movement.
  loom_rewriter_t* rewriter;
  // Shared motion legality analysis.
  loom_motion_analysis_t motion;
  // Region DFS stack for finding region-branch ops in a function.
  loom_branch_sink_region_stack_t region_stack;
} loom_branch_sink_context_t;

//===----------------------------------------------------------------------===//
// Use classification
//===----------------------------------------------------------------------===//

#define LOOM_BRANCH_SINK_REGION_INDEX_NONE UINT8_MAX

static uint8_t loom_branch_sink_direct_region_index(
    loom_region_branch_t branch, const loom_region_t* region) {
  if (!region) return LOOM_BRANCH_SINK_REGION_INDEX_NONE;
  loom_region_t** regions = loom_op_regions(branch.op);
  for (uint8_t i = 0; i < branch.op->region_count; ++i) {
    if (regions[i] == region) return i;
  }
  return LOOM_BRANCH_SINK_REGION_INDEX_NONE;
}

static uint8_t loom_branch_sink_op_region_index(loom_region_branch_t branch,
                                                const loom_op_t* op) {
  for (const loom_op_t* current = op; current; current = current->parent_op) {
    if (current->parent_op != branch.op) continue;
    return loom_branch_sink_direct_region_index(
        branch,
        current->parent_block ? current->parent_block->parent_region : NULL);
  }
  return LOOM_BRANCH_SINK_REGION_INDEX_NONE;
}

static uint8_t loom_branch_sink_block_arg_region_index(
    loom_region_branch_t branch, const loom_block_t* block) {
  if (!block) return LOOM_BRANCH_SINK_REGION_INDEX_NONE;
  uint8_t direct_region_index =
      loom_branch_sink_direct_region_index(branch, block->parent_region);
  if (direct_region_index != LOOM_BRANCH_SINK_REGION_INDEX_NONE) {
    return direct_region_index;
  }
  if (!block->first_op || !block->first_op->parent_op) {
    return LOOM_BRANCH_SINK_REGION_INDEX_NONE;
  }
  return loom_branch_sink_op_region_index(branch, block->first_op->parent_op);
}

static bool loom_branch_sink_merge_region_index(uint8_t region_index,
                                                uint8_t* target_region_index,
                                                bool* has_use) {
  if (region_index == LOOM_BRANCH_SINK_REGION_INDEX_NONE) return false;
  if (!*has_use) {
    *has_use = true;
    *target_region_index = region_index;
    return true;
  }
  return *target_region_index == region_index;
}

static bool loom_branch_sink_value_uses_target_one_region(
    const loom_module_t* module, loom_region_branch_t branch,
    loom_value_id_t value_id, uint8_t* target_region_index, bool* has_use) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return false;
  }

  const loom_value_t* value = loom_module_value(module, value_id);
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    loom_op_t* user_op = loom_use_user_op(uses[i]);
    uint8_t region_index = loom_branch_sink_op_region_index(branch, user_op);
    if (!loom_branch_sink_merge_region_index(region_index, target_region_index,
                                             has_use)) {
      return false;
    }
  }

  if (value_id >= module->type_uses.value_capacity) return true;
  loom_type_use_id_t use_id =
      module->type_uses.value_heads[value_id].first_incoming_use_id;
  while (use_id != LOOM_TYPE_USE_ID_INVALID) {
    const loom_type_use_t* type_use = &module->type_uses.records[use_id];
    if (type_use->user_value_id >= module->values.count) return false;

    const loom_value_t* user_value =
        loom_module_value(module, type_use->user_value_id);
    uint8_t region_index = LOOM_BRANCH_SINK_REGION_INDEX_NONE;
    if (loom_value_is_block_arg(user_value)) {
      region_index = loom_branch_sink_block_arg_region_index(
          branch, loom_value_def_block(user_value));
    } else {
      region_index = loom_branch_sink_op_region_index(
          branch, loom_value_def_op(user_value));
    }
    if (!loom_branch_sink_merge_region_index(region_index, target_region_index,
                                             has_use)) {
      return false;
    }

    use_id = type_use->next_incoming_use_id;
  }

  return true;
}

static bool loom_branch_sink_results_target_one_region(
    const loom_module_t* module, loom_region_branch_t branch,
    const loom_op_t* candidate_op, uint8_t* out_region_index) {
  bool has_use = false;
  uint8_t region_index = LOOM_BRANCH_SINK_REGION_INDEX_NONE;
  const loom_value_id_t* results = loom_op_const_results(candidate_op);
  for (uint16_t i = 0; i < candidate_op->result_count; ++i) {
    if (!loom_branch_sink_value_uses_target_one_region(
            module, branch, results[i], &region_index, &has_use)) {
      return false;
    }
  }
  if (!has_use) return false;
  *out_region_index = region_index;
  return true;
}

static bool loom_branch_sink_value_uses_only_op(const loom_module_t* module,
                                                loom_value_id_t value_id,
                                                const loom_op_t* target_op,
                                                bool* has_use) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return false;
  }

  const loom_value_t* value = loom_module_value(module, value_id);
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    if (loom_use_user_op(uses[i]) != target_op) return false;
    *has_use = true;
  }

  if (value_id >= module->type_uses.value_capacity) return true;
  loom_type_use_id_t use_id =
      module->type_uses.value_heads[value_id].first_incoming_use_id;
  while (use_id != LOOM_TYPE_USE_ID_INVALID) {
    const loom_type_use_t* type_use = &module->type_uses.records[use_id];
    if (type_use->user_value_id >= module->values.count) return false;

    const loom_value_t* user_value =
        loom_module_value(module, type_use->user_value_id);
    if (loom_value_is_block_arg(user_value)) return false;
    if (loom_value_def_op(user_value) != target_op) return false;
    *has_use = true;

    use_id = type_use->next_incoming_use_id;
  }

  return true;
}

static bool loom_branch_sink_results_use_only_op(const loom_module_t* module,
                                                 const loom_op_t* candidate_op,
                                                 const loom_op_t* target_op,
                                                 bool* out_has_use) {
  *out_has_use = false;
  const loom_value_id_t* results = loom_op_const_results(candidate_op);
  for (uint16_t i = 0; i < candidate_op->result_count; ++i) {
    if (!loom_branch_sink_value_uses_only_op(module, results[i], target_op,
                                             out_has_use)) {
      return false;
    }
  }
  return true;
}

static loom_op_t* loom_branch_sink_value_producer(const loom_module_t* module,
                                                  loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return NULL;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  return loom_value_is_block_arg(value) ? NULL : loom_value_def_op(value);
}

//===----------------------------------------------------------------------===//
// Sinking
//===----------------------------------------------------------------------===//

static iree_status_t loom_branch_sink_push_child_regions(
    loom_branch_sink_context_t* context, loom_op_t* op) {
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_branch_sink_region_stack_push(
        context->pass->arena, &context->region_stack, regions[i]));
  }
  return iree_ok_status();
}

static loom_op_t* loom_branch_sink_region_insertion_op(
    loom_region_branch_t branch, uint8_t region_index) {
  if (region_index >= branch.op->region_count) return NULL;
  loom_region_t* region = loom_op_regions(branch.op)[region_index];
  if (!region || region->block_count == 0) return NULL;
  loom_block_t* entry_block = loom_region_entry_block(region);
  return entry_block ? entry_block->first_op : NULL;
}

static iree_status_t loom_branch_sink_try_selector(
    loom_branch_sink_context_t* context, loom_op_t* branch_op,
    loom_value_id_t selector, bool* out_sunk) {
  *out_sunk = false;

  loom_op_t* candidate_op =
      loom_branch_sink_value_producer(context->module, selector);
  if (!candidate_op || candidate_op->result_count == 0) {
    return iree_ok_status();
  }
  if (candidate_op->parent_block == branch_op->parent_block &&
      candidate_op->next_op == branch_op) {
    return iree_ok_status();
  }

  bool has_use = false;
  if (!loom_branch_sink_results_use_only_op(context->module, candidate_op,
                                            branch_op, &has_use) ||
      !has_use) {
    return iree_ok_status();
  }

  bool can_relocate = false;
  IREE_RETURN_IF_ERROR(loom_motion_subtree_can_relocate_before(
      &context->motion, candidate_op, branch_op, &can_relocate));
  if (!can_relocate) return iree_ok_status();

  IREE_RETURN_IF_ERROR(
      loom_rewriter_move_before(context->rewriter, candidate_op, branch_op));
  *out_sunk = true;
  ++context->statistics->selectors_sunk;
  ++context->statistics->ops_sunk;
  return iree_ok_status();
}

static iree_status_t loom_branch_sink_try_candidate(
    loom_branch_sink_context_t* context, loom_region_branch_t branch,
    loom_op_t* candidate_op, bool* out_sunk) {
  *out_sunk = false;
  if (candidate_op->result_count == 0) return iree_ok_status();

  uint8_t target_region_index = LOOM_BRANCH_SINK_REGION_INDEX_NONE;
  if (!loom_branch_sink_results_target_one_region(
          context->module, branch, candidate_op, &target_region_index)) {
    return iree_ok_status();
  }

  loom_op_t* insertion_op =
      loom_branch_sink_region_insertion_op(branch, target_region_index);
  if (!insertion_op) return iree_ok_status();

  bool can_relocate = false;
  IREE_RETURN_IF_ERROR(loom_motion_subtree_can_relocate_before(
      &context->motion, candidate_op, insertion_op, &can_relocate));
  if (!can_relocate) return iree_ok_status();

  IREE_RETURN_IF_ERROR(
      loom_rewriter_move_before(context->rewriter, candidate_op, insertion_op));
  *out_sunk = true;
  ++context->statistics->ops_sunk;
  return iree_ok_status();
}

static iree_status_t loom_branch_sink_process_branch(
    loom_branch_sink_context_t* context, loom_region_branch_t branch,
    bool* out_changed) {
  ++context->statistics->branches_visited;

  bool selector_sunk = false;
  IREE_RETURN_IF_ERROR(loom_branch_sink_try_selector(
      context, branch.op, loom_region_branch_selector(branch), &selector_sunk));
  *out_changed |= selector_sunk;

  loom_op_t* candidate_op = branch.op->prev_op;
  while (candidate_op) {
    loom_op_t* previous_op = candidate_op->prev_op;
    bool sunk = false;
    IREE_RETURN_IF_ERROR(
        loom_branch_sink_try_candidate(context, branch, candidate_op, &sunk));
    if (!sunk) break;
    *out_changed = true;
    candidate_op = previous_op;
  }
  return iree_ok_status();
}

static iree_status_t loom_branch_sink_process_cfg_cond_br(
    loom_branch_sink_context_t* context, loom_op_t* op, bool* out_changed) {
  ++context->statistics->branches_visited;

  bool selector_sunk = false;
  IREE_RETURN_IF_ERROR(loom_branch_sink_try_selector(
      context, op, loom_cfg_cond_br_condition(op), &selector_sunk));
  *out_changed |= selector_sunk;
  return iree_ok_status();
}

static iree_status_t loom_branch_sink_process_function_once(
    loom_branch_sink_context_t* context, loom_func_like_t function,
    bool* out_changed) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) return iree_ok_status();

  context->region_stack.count = 0;
  IREE_RETURN_IF_ERROR(loom_branch_sink_region_stack_push(
      context->pass->arena, &context->region_stack, body));

  while (true) {
    loom_region_t* region =
        loom_branch_sink_region_stack_pop(&context->region_stack);
    if (!region) break;

    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* op = block->first_op;
      while (op) {
        loom_op_t* next_op = op->next_op;
        if (op->flags & LOOM_OP_FLAG_DEAD) {
          op = next_op;
          continue;
        }

        loom_region_branch_t branch =
            loom_region_branch_cast(context->module, op);
        if (loom_region_branch_isa(branch)) {
          IREE_RETURN_IF_ERROR(
              loom_branch_sink_process_branch(context, branch, out_changed));
        } else if (loom_cfg_cond_br_isa(op)) {
          IREE_RETURN_IF_ERROR(
              loom_branch_sink_process_cfg_cond_br(context, op, out_changed));
        }

        IREE_RETURN_IF_ERROR(loom_branch_sink_push_child_regions(context, op));
        op = next_op;
      }
    }
  }

  return iree_ok_status();
}

iree_status_t loom_branch_sink_run(loom_pass_t* pass, loom_module_t* module,
                                   loom_func_like_t function) {
  if (!loom_func_like_body(function)) return iree_ok_status();

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));

  loom_branch_sink_context_t context = {
      .pass = pass,
      .statistics = loom_branch_sink_statistics(pass),
      .module = module,
      .rewriter = &rewriter,
  };
  iree_status_t status = loom_branch_sink_region_stack_initialize(
      pass->arena, &context.region_stack);
  if (iree_status_is_ok(status)) {
    status =
        loom_motion_analysis_initialize(module, pass->arena, &context.motion);
  }

  bool changed = true;
  bool any_changed = false;
  while (iree_status_is_ok(status) && changed) {
    changed = false;
    status =
        loom_branch_sink_process_function_once(&context, function, &changed);
    any_changed |= changed;
  }

  if (iree_status_is_ok(status) && any_changed) {
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
