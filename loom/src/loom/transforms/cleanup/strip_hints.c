// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cleanup/strip_hints.h"

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

#define LOOM_STRIP_HINTS_STATISTICS(V, statistics_type) \
  V(statistics_type, hints_stripped, "hints-stripped",  \
    "Number of hint ops removed.")

LOOM_PASS_STATISTICS_DEFINE(loom_strip_hints_statistics,
                            loom_strip_hints_statistics_t,
                            LOOM_STRIP_HINTS_STATISTICS)

static const loom_pass_info_t loom_strip_hints_pass_info_storage = {
    .name = IREE_SVL("strip-hints"),
    .description = IREE_SVL("Remove compiler hint operations."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_strip_hints_statistics_layout,
};

const loom_pass_info_t* loom_strip_hints_pass_info(void) {
  return &loom_strip_hints_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Block collection
//===----------------------------------------------------------------------===//

#define LOOM_STRIP_HINTS_INITIAL_CAPACITY 32

static iree_status_t loom_strip_hints_collect_blocks(
    iree_arena_allocator_t* arena, loom_region_t* body,
    loom_block_t*** out_blocks, iree_host_size_t* out_count) {
  loom_region_t** region_stack = NULL;
  iree_host_size_t stack_count = 0;
  iree_host_size_t stack_capacity = LOOM_STRIP_HINTS_INITIAL_CAPACITY;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, stack_capacity, sizeof(loom_region_t*), (void**)&region_stack));

  loom_block_t** blocks = NULL;
  iree_host_size_t block_count = 0;
  iree_host_size_t block_capacity = LOOM_STRIP_HINTS_INITIAL_CAPACITY;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, block_capacity, sizeof(loom_block_t*), (void**)&blocks));

  region_stack[stack_count++] = body;
  while (stack_count > 0) {
    loom_region_t* region = region_stack[--stack_count];

    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      if (block_count >= block_capacity) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            arena, block_count, block_count + 1, sizeof(loom_block_t*),
            &block_capacity, (void**)&blocks));
      }
      blocks[block_count++] = block;

      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        if (op->region_count == 0) continue;
        loom_region_t** regions = loom_op_regions(op);
        for (uint8_t r = 0; r < op->region_count; ++r) {
          if (!regions[r] || regions[r]->block_count == 0) continue;
          if (stack_count >= stack_capacity) {
            IREE_RETURN_IF_ERROR(iree_arena_grow_array(
                arena, stack_count, stack_count + 1, sizeof(loom_region_t*),
                &stack_capacity, (void**)&region_stack));
          }
          region_stack[stack_count++] = regions[r];
        }
      }
    }
  }

  *out_blocks = blocks;
  *out_count = block_count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

iree_status_t loom_strip_hints_run(loom_pass_t* pass, loom_module_t* module,
                                   loom_func_like_t function) {
  if (!loom_func_like_body(function)) return iree_ok_status();
  loom_strip_hints_statistics_t* statistics = loom_strip_hints_statistics(pass);

  for (uint8_t region_index = 0;
       region_index < loom_func_like_region_count(function); ++region_index) {
    loom_region_t* region = loom_func_like_region(function, region_index);
    if (!region) continue;

    loom_block_t** all_blocks = NULL;
    iree_host_size_t block_count = 0;
    IREE_RETURN_IF_ERROR(loom_strip_hints_collect_blocks(
        pass->arena, region, &all_blocks, &block_count));

    for (iree_host_size_t b = 0; b < block_count; ++b) {
      loom_block_t* block = all_blocks[b];
      for (loom_op_t* op = block->last_op; op;) {
        loom_op_t* prev_op = op->prev_op;
        if (!iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
          loom_trait_flags_t traits = loom_op_effective_traits(module, op);
          if (iree_any_bit_set(traits, LOOM_TRAIT_HINT)) {
            IREE_RETURN_IF_ERROR(loom_op_erase(module, op));
            loom_pass_mark_changed(pass);
            ++statistics->hints_stripped;
          }
        }
        op = prev_op;
      }
    }
  }
  return iree_ok_status();
}
