// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/loop/licm.h"

#include "loom/analysis/motion.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/function_version.h"
#include "loom/util/walk.h"

#define LOOM_LICM_STATISTICS(V, statistics_type)     \
  V(statistics_type, loops_visited, "loops-visited", \
    "Number of loop-like operations inspected.")     \
  V(statistics_type, ops_hoisted, "ops-hoisted",     \
    "Number of loop-invariant operations hoisted.")

LOOM_PASS_STATISTICS_DEFINE(loom_licm_statistics, loom_licm_statistics_t,
                            LOOM_LICM_STATISTICS)

static const loom_pass_info_t loom_licm_pass_info_storage = {
    .name = IREE_SVL("licm"),
    .description =
        IREE_SVL("Hoist loop-invariant operations with proven legality."),
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
  if (!region || region->block_count == 0) {
    return iree_ok_status();
  }
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
  // Function owning the lazily acquired analysis scope.
  loom_func_like_t function;
  // Rewriter used for all IR movement.
  loom_rewriter_t* rewriter;
  // Shared motion legality analysis.
  loom_motion_analysis_t motion;
  // Function-local value domain backing symbolic movement analysis.
  loom_local_value_domain_t value_domain;
  // Region DFS stack for scanning one loop body for hoistable ops.
  loom_licm_region_stack_t loop_stack;
  // Whether the current traversal moved any operation.
  bool changed;
} loom_licm_context_t;

//===----------------------------------------------------------------------===//
// Hoisting
//===----------------------------------------------------------------------===//

static iree_status_t loom_licm_push_child_regions(loom_licm_context_t* context,
                                                  loom_op_t* op) {
  if (op->region_count == 0) {
    return iree_ok_status();
  }
  const loom_loop_like_t nested_loop = loom_loop_like_cast(context->module, op);
  loom_region_t* nested_body =
      loom_loop_like_isa(nested_loop) ? loom_loop_like_body(nested_loop) : NULL;
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    // The child loop already inspected its body in this round. Its hoisted
    // work is now in the enclosing block. Keep visiting other regions, such
    // as a while condition, which do not belong to that body's traversal.
    if (regions[i] == nested_body) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_licm_region_stack_push(
        context->pass->arena, &context->loop_stack, regions[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_licm_hoist_from_loop_body(
    loom_licm_context_t* context, loom_loop_like_t loop) {
  loom_region_t* body = loom_loop_like_body(loop);
  if (!body) {
    return iree_ok_status();
  }

  context->loop_stack.count = 0;
  IREE_RETURN_IF_ERROR(loom_licm_region_stack_push(context->pass->arena,
                                                   &context->loop_stack, body));

  while (true) {
    loom_region_t* region = loom_licm_region_stack_pop(&context->loop_stack);
    if (!region) {
      break;
    }

    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      bool in_entry_prefix =
          region == body && block == loom_region_entry_block(body);
      loom_op_t* op = block->first_op;
      while (op) {
        loom_op_t* next_op = op->next_op;
        if (op->flags & LOOM_OP_FLAG_DEAD) {
          op = next_op;
          continue;
        }

        loom_motion_loop_hoist_result_t result;
        IREE_RETURN_IF_ERROR(loom_motion_subtree_evaluate_hoist_before_loop(
            &context->motion, loop, op, in_entry_prefix, &result));
        // A fact identity in this prefix executes on every entry into a
        // nonempty loop. Effects, control regions, or other non-speculatable
        // work end that proof even when a later identity is invariant.
        in_entry_prefix =
            in_entry_prefix && op->region_count == 0 &&
            loom_motion_traits_are_speculation_compatible(
                loom_op_effective_traits(context->module, op),
                /*is_root_op=*/true) &&
            !iree_any_bit_set(
                result.rejection_bits,
                LOOM_MOTION_LOOP_HOIST_REJECTION_CANDIDATE_SEMANTICS);
        if (loom_motion_loop_hoist_result_is_legal(&result)) {
          IREE_RETURN_IF_ERROR(
              loom_rewriter_move_before(context->rewriter, op, loop.op));
          context->changed = true;
          ++context->statistics->ops_hoisted;
        } else {
          IREE_RETURN_IF_ERROR(loom_licm_push_child_regions(context, op));
        }

        op = next_op;
      }
    }
  }

  return iree_ok_status();
}

static iree_status_t loom_licm_initialize_motion(loom_licm_context_t* context) {
  loom_pass_t* pass = context->pass;
  loom_region_t* body = loom_func_like_body(context->function);
  IREE_RETURN_IF_ERROR(
      loom_licm_region_stack_initialize(pass->arena, &context->loop_stack));
  IREE_RETURN_IF_ERROR(loom_local_value_domain_acquire_for_region_tree(
      context->module, body, pass->arena, &context->value_domain));
  loom_value_fact_table_t* fact_table = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
      pass, context->module,
      loom_pass_value_fact_scope_function_for_target(
          context->function,
          loom_target_function_version_target_facts(pass->function_version)),
      &fact_table));
  return loom_motion_analysis_initialize_region(
      context->module, body, fact_table, &context->value_domain, pass->arena,
      &context->motion);
}

static iree_status_t loom_licm_process_loop(
    void* user_data, loom_op_t* op, const loom_walk_context_t* walk_context,
    loom_walk_result_t* out_result) {
  (void)walk_context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_licm_context_t* context = (loom_licm_context_t*)user_data;
  const loom_loop_like_t loop = loom_loop_like_cast(context->module, op);
  if (loom_loop_like_isa(loop)) {
    // No mutation precedes the first loop visit. Acquire the shared snapshot
    // here so loop-free functions do not construct unused analysis state.
    if (!context->motion.module) {
      IREE_RETURN_IF_ERROR(loom_licm_initialize_motion(context));
    }
    ++context->statistics->loops_visited;
    return loom_licm_hoist_from_loop_body(context, loop);
  }
  return iree_ok_status();
}

iree_status_t loom_licm_run(loom_pass_t* pass, loom_module_t* module,
                            loom_func_like_t function) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) {
    return iree_ok_status();
  }

  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, pass->arena);

  loom_licm_context_t context = {
      .pass = pass,
      .statistics = loom_licm_statistics(pass),
      .module = module,
      .function = function,
      .rewriter = &rewriter,
  };
  iree_status_t status = iree_ok_status();
  bool any_changed = false;
  context.changed = true;
  while (iree_status_is_ok(status) && context.changed) {
    context.changed = false;
    loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
    // Visit children before parents so an enclosing loop immediately sees
    // operations hoisted from its nested loops.
    status =
        loom_walk_region(module, body, LOOM_WALK_POST_ORDER,
                         (loom_walk_callback_t){.fn = loom_licm_process_loop,
                                                .user_data = &context},
                         &walk_result);
    any_changed |= context.changed;
  }

  if (iree_status_is_ok(status) && any_changed) {
    loom_pass_mark_changed(pass);
  }
  loom_local_value_domain_release(&context.value_domain);
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
