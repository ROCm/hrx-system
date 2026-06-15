// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/loop/licm.h"

#include "loom/analysis/motion.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/rewriter.h"

#define LOOM_LICM_STATISTICS(V, statistics_type)     \
  V(statistics_type, loops_visited, "loops-visited", \
    "Number of loop-like operations inspected.")     \
  V(statistics_type, ops_hoisted, "ops-hoisted",     \
    "Number of loop-invariant operations hoisted.")

LOOM_PASS_STATISTICS_DEFINE(loom_licm_statistics, loom_licm_statistics_t,
                            LOOM_LICM_STATISTICS)

static const loom_pass_info_t loom_licm_pass_info_storage = {
    .name = IREE_SVL("licm"),
    .description = IREE_SVL("Hoist loop-invariant pure operations."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_licm_statistics_layout,
};

const loom_pass_info_t* loom_licm_pass_info(void) {
  return &loom_licm_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Scratch stacks
//===----------------------------------------------------------------------===//

#define LOOM_LICM_INITIAL_REGION_STACK_CAPACITY 16

typedef struct loom_licm_region_stack_t {
  // Region pointers waiting to be processed.
  loom_region_t** regions;
  // Number of queued region pointers.
  iree_host_size_t count;
  // Allocated region pointer capacity.
  iree_host_size_t capacity;
} loom_licm_region_stack_t;

static iree_status_t loom_licm_region_stack_initialize(
    iree_arena_allocator_t* arena, loom_licm_region_stack_t* stack) {
  stack->count = 0;
  stack->capacity = LOOM_LICM_INITIAL_REGION_STACK_CAPACITY;
  return iree_arena_allocate_array(
      arena, stack->capacity, sizeof(loom_region_t*), (void**)&stack->regions);
}

static iree_status_t loom_licm_region_stack_push(
    iree_arena_allocator_t* arena, loom_licm_region_stack_t* stack,
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

static loom_region_t* loom_licm_region_stack_pop(
    loom_licm_region_stack_t* stack) {
  return stack->count > 0 ? stack->regions[--stack->count] : NULL;
}

typedef struct loom_licm_context_t {
  // Pass instance owning the scratch arena.
  loom_pass_t* pass;
  // Typed statistics storage for the current pass invocation.
  loom_licm_statistics_t* statistics;
  // Module being transformed.
  loom_module_t* module;
  // Rewriter used for all IR movement.
  loom_rewriter_t* rewriter;
  // Shared motion legality analysis.
  loom_motion_analysis_t motion;
  // Region DFS stack for finding loop-like ops in the function.
  loom_licm_region_stack_t function_stack;
  // Region DFS stack for scanning one loop body for hoistable ops.
  loom_licm_region_stack_t loop_stack;
} loom_licm_context_t;

//===----------------------------------------------------------------------===//
// Hoisting
//===----------------------------------------------------------------------===//

static iree_status_t loom_licm_op_is_hoistable(loom_licm_context_t* context,
                                               loom_op_t* loop_op,
                                               loom_op_t* candidate_op,
                                               bool* out_hoistable) {
  return loom_motion_subtree_can_relocate_before(&context->motion, candidate_op,
                                                 loop_op, out_hoistable);
}

static iree_status_t loom_licm_push_child_regions(
    loom_licm_context_t* context, loom_licm_region_stack_t* stack,
    loom_op_t* op) {
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_licm_region_stack_push(context->pass->arena, stack, regions[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_licm_hoist_from_loop_body(
    loom_licm_context_t* context, loom_loop_like_t loop, bool* out_changed) {
  loom_region_t* body = loom_loop_like_body(loop);
  if (!body) return iree_ok_status();

  context->loop_stack.count = 0;
  IREE_RETURN_IF_ERROR(loom_licm_region_stack_push(context->pass->arena,
                                                   &context->loop_stack, body));

  while (true) {
    loom_region_t* region = loom_licm_region_stack_pop(&context->loop_stack);
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

        bool hoistable = false;
        IREE_RETURN_IF_ERROR(
            loom_licm_op_is_hoistable(context, loop.op, op, &hoistable));
        if (hoistable) {
          IREE_RETURN_IF_ERROR(
              loom_rewriter_move_before(context->rewriter, op, loop.op));
          *out_changed = true;
          ++context->statistics->ops_hoisted;
        } else {
          IREE_RETURN_IF_ERROR(
              loom_licm_push_child_regions(context, &context->loop_stack, op));
        }

        op = next_op;
      }
    }
  }

  return iree_ok_status();
}

static iree_status_t loom_licm_process_function_once(
    loom_licm_context_t* context, loom_func_like_t function,
    bool* out_changed) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) return iree_ok_status();

  context->function_stack.count = 0;
  IREE_RETURN_IF_ERROR(loom_licm_region_stack_push(
      context->pass->arena, &context->function_stack, body));

  while (true) {
    loom_region_t* region =
        loom_licm_region_stack_pop(&context->function_stack);
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

        loom_loop_like_t loop = loom_loop_like_cast(context->module, op);
        if (loom_loop_like_isa(loop)) {
          ++context->statistics->loops_visited;
          IREE_RETURN_IF_ERROR(
              loom_licm_hoist_from_loop_body(context, loop, out_changed));
        }

        IREE_RETURN_IF_ERROR(loom_licm_push_child_regions(
            context, &context->function_stack, op));
        op = next_op;
      }
    }
  }

  return iree_ok_status();
}

iree_status_t loom_licm_run(loom_pass_t* pass, loom_module_t* module,
                            loom_func_like_t function) {
  if (!loom_func_like_body(function)) return iree_ok_status();

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));

  loom_licm_context_t context = {
      .pass = pass,
      .statistics = loom_licm_statistics(pass),
      .module = module,
      .rewriter = &rewriter,
  };
  iree_status_t status =
      loom_licm_region_stack_initialize(pass->arena, &context.function_stack);
  if (iree_status_is_ok(status)) {
    status =
        loom_licm_region_stack_initialize(pass->arena, &context.loop_stack);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_motion_analysis_initialize(module, pass->arena, &context.motion);
  }

  bool changed = true;
  bool any_changed = false;
  while (iree_status_is_ok(status) && changed) {
    changed = false;
    status = loom_licm_process_function_once(&context, function, &changed);
    any_changed |= changed;
  }

  if (iree_status_is_ok(status) && any_changed) {
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
