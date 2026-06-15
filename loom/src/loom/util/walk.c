// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/walk.h"

#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// DFS stack
//===----------------------------------------------------------------------===//

typedef struct loom_walk_frame_t {
  loom_block_t* block;
  loom_region_t* region;
  loom_op_t* parent_op;
  loom_op_t* next_op;
  uint16_t depth;
  // For post-order: the op whose children have just been visited.
  // When non-NULL, the callback fires for this op before advancing
  // to the next op in the block.
  loom_op_t* deferred_post_op;
} loom_walk_frame_t;

typedef struct loom_walk_stack_t {
  loom_walk_frame_t* frames;
  iree_host_size_t count;
  iree_host_size_t capacity;
} loom_walk_stack_t;

#define LOOM_WALK_INITIAL_STACK_CAPACITY 32

static iree_status_t loom_walk_stack_initialize(iree_arena_allocator_t* arena,
                                                loom_walk_stack_t* stack) {
  stack->count = 0;
  stack->capacity = LOOM_WALK_INITIAL_STACK_CAPACITY;
  return iree_arena_allocate_array(arena, stack->capacity,
                                   sizeof(loom_walk_frame_t),
                                   (void**)&stack->frames);
}

static iree_status_t loom_walk_stack_reserve(loom_walk_stack_t* stack,
                                             iree_arena_allocator_t* arena,
                                             iree_host_size_t additional) {
  iree_host_size_t required = stack->count + additional;
  if (required <= stack->capacity) return iree_ok_status();
  return iree_arena_grow_array(arena, stack->count, required,
                               sizeof(loom_walk_frame_t), &stack->capacity,
                               (void**)&stack->frames);
}

static void loom_walk_stack_push(loom_walk_stack_t* stack, loom_block_t* block,
                                 loom_region_t* region, loom_op_t* parent_op,
                                 uint16_t depth) {
  IREE_ASSERT(stack->count < stack->capacity);
  stack->frames[stack->count++] = (loom_walk_frame_t){
      .block = block,
      .region = region,
      .parent_op = parent_op,
      .next_op = block->first_op,
      .depth = depth,
      .deferred_post_op = NULL,
  };
}

//===----------------------------------------------------------------------===//
// Region child frame pushing
//===----------------------------------------------------------------------===//

// Counts the total number of blocks across all regions of an op.
static iree_host_size_t loom_walk_total_block_count(const loom_op_t* op) {
  iree_host_size_t total = 0;
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t r = 0; r < op->region_count; ++r) {
    if (regions[r]) total += regions[r]->block_count;
  }
  return total;
}

// Pushes child frames for all nested regions of an op.
//
// For single-block regions, pushes one frame.
//
// For multi-block regions, uses entry block dominance: block 0
// is pushed on top of the stack so it is processed first. Sibling
// blocks are pushed below in reverse order.
static void loom_walk_push_region_frames(loom_walk_stack_t* stack,
                                         const loom_op_t* op,
                                         uint16_t child_depth) {
  loom_region_t** regions = loom_op_regions(op);
  // Push regions in reverse order so the first region's first block
  // is on top of the stack and processed first.
  for (int32_t r = (int32_t)op->region_count - 1; r >= 0; --r) {
    loom_region_t* region = regions[r];
    if (!region || region->block_count == 0) continue;
    if (region->block_count == 1) {
      loom_walk_stack_push(stack, loom_region_entry_block(region), region,
                           (loom_op_t*)op, child_depth);
    } else {
      // Push non-entry blocks in reverse order.
      for (int32_t b = (int32_t)region->block_count - 1; b >= 1; --b) {
        loom_walk_stack_push(stack, loom_region_block(region, (uint16_t)b),
                             region, (loom_op_t*)op, child_depth);
      }
      // Push entry block on top — processed first.
      loom_walk_stack_push(stack, loom_region_entry_block(region), region,
                           (loom_op_t*)op, child_depth);
    }
  }
}

//===----------------------------------------------------------------------===//
// Walk implementation
//===----------------------------------------------------------------------===//

iree_status_t loom_walk_region(const loom_module_t* module,
                               loom_region_t* region, loom_walk_order_t order,
                               loom_walk_callback_t callback,
                               iree_arena_allocator_t* arena,
                               loom_walk_result_t* out_result) {
  *out_result = LOOM_WALK_CONTINUE;
  if (!region || region->block_count == 0) return iree_ok_status();

  loom_walk_stack_t stack;
  IREE_RETURN_IF_ERROR(loom_walk_stack_initialize(arena, &stack));

  // Push the region's blocks onto the stack. For multi-block regions,
  // entry block goes on top (processed first).
  IREE_RETURN_IF_ERROR(
      loom_walk_stack_reserve(&stack, arena, region->block_count));
  if (region->block_count == 1) {
    loom_walk_stack_push(&stack, loom_region_entry_block(region), region, NULL,
                         0);
  } else {
    for (int32_t b = (int32_t)region->block_count - 1; b >= 1; --b) {
      loom_walk_stack_push(&stack, loom_region_block(region, (uint16_t)b),
                           region, NULL, 0);
    }
    loom_walk_stack_push(&stack, loom_region_entry_block(region), region, NULL,
                         0);
  }

  while (stack.count > 0) {
    loom_walk_frame_t* frame = &stack.frames[stack.count - 1];

    // Handle deferred post-order callback: the op's children have
    // been fully visited, now fire the callback for the op itself.
    if (frame->deferred_post_op) {
      loom_walk_context_t context = {
          .block = frame->block,
          .region = frame->region,
          .parent_op = frame->parent_op,
          .depth = frame->depth,
      };
      loom_walk_result_t result = LOOM_WALK_CONTINUE;
      IREE_RETURN_IF_ERROR(callback.fn(
          callback.user_data, frame->deferred_post_op, &context, &result));
      frame->deferred_post_op = NULL;
      if (result == LOOM_WALK_ABORT) {
        *out_result = LOOM_WALK_ABORT;
        return iree_ok_status();
      }
      // SKIP in post-order is a no-op (children already visited).
      continue;
    }

    // Block done — pop frame.
    if (!frame->next_op) {
      --stack.count;
      continue;
    }

    loom_op_t* op = frame->next_op;
    frame->next_op = op->next_op;
    if (op->flags & LOOM_OP_FLAG_DEAD) continue;

    loom_walk_context_t context = {
        .block = frame->block,
        .region = frame->region,
        .parent_op = frame->parent_op,
        .depth = frame->depth,
    };

    bool skip_regions = false;

    if (order == LOOM_WALK_PRE_ORDER) {
      loom_walk_result_t result = LOOM_WALK_CONTINUE;
      IREE_RETURN_IF_ERROR(
          callback.fn(callback.user_data, op, &context, &result));
      if (result == LOOM_WALK_ABORT) {
        *out_result = LOOM_WALK_ABORT;
        return iree_ok_status();
      }
      if (result == LOOM_WALK_SKIP) {
        skip_regions = true;
      }
    }

    // Push child frames for nested regions.
    if (op->region_count > 0 && !skip_regions) {
      iree_host_size_t child_block_count = loom_walk_total_block_count(op);
      if (child_block_count > 0) {
        IREE_RETURN_IF_ERROR(
            loom_walk_stack_reserve(&stack, arena, child_block_count));
        // Re-fetch frame pointer — reserve may have reallocated.
        frame = &stack.frames[stack.count - 1];

        if (order == LOOM_WALK_POST_ORDER) {
          // Defer the callback until after children are processed.
          frame->deferred_post_op = op;
        }

        uint16_t child_depth = (uint16_t)(frame->depth + 1);
        loom_walk_push_region_frames(&stack, op, child_depth);
      }
    } else if (order == LOOM_WALK_POST_ORDER) {
      // No regions (or skipped): fire callback immediately.
      loom_walk_result_t result = LOOM_WALK_CONTINUE;
      IREE_RETURN_IF_ERROR(
          callback.fn(callback.user_data, op, &context, &result));
      if (result == LOOM_WALK_ABORT) {
        *out_result = LOOM_WALK_ABORT;
        return iree_ok_status();
      }
    }
  }

  return iree_ok_status();
}

iree_status_t loom_walk_function(const loom_module_t* module,
                                 loom_func_like_t function,
                                 loom_walk_order_t order,
                                 loom_walk_callback_t callback,
                                 iree_arena_allocator_t* arena,
                                 loom_walk_result_t* out_result) {
  *out_result = LOOM_WALK_CONTINUE;
  for (uint8_t i = 0; i < loom_func_like_region_count(function); ++i) {
    loom_region_t* region = loom_func_like_region(function, i);
    if (!region) continue;
    IREE_RETURN_IF_ERROR(
        loom_walk_region(module, region, order, callback, arena, out_result));
    if (*out_result == LOOM_WALK_ABORT) break;
  }
  return iree_ok_status();
}
