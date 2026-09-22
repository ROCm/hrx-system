// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/read_motion.h"

#include "loom/analysis/motion.h"
#include "loom/util/adaptive_sort.h"
#include "loom/util/walk.h"

typedef struct loom_read_motion_builder_t {
  // Analysis receiving the retained ordering boundaries.
  loom_read_motion_t* analysis;
  // Arena owning the boundary index and traversal stack.
  iree_arena_allocator_t* arena;
  // Allocated boundary entries.
  iree_host_size_t capacity;
} loom_read_motion_builder_t;

static iree_status_t loom_read_motion_visit_op(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_read_motion_builder_t* builder = user_data;
  loom_read_motion_t* analysis = builder->analysis;
  if (loom_motion_read_can_cross_op(analysis->module, op)) {
    return iree_ok_status();
  }
  if (analysis->barrier_count == builder->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        builder->arena, analysis->barrier_count, analysis->barrier_count + 1,
        sizeof(*analysis->barriers), &builder->capacity,
        (void**)&analysis->barriers));
  }
  analysis->barriers[analysis->barrier_count++].op = op;
  return iree_ok_status();
}

static bool loom_read_motion_barrier_less(
    const loom_read_motion_barrier_t* lhs,
    const loom_read_motion_barrier_t* rhs) {
  if (lhs->op->parent_block != rhs->op->parent_block) {
    return (uintptr_t)lhs->op->parent_block < (uintptr_t)rhs->op->parent_block;
  }
  return lhs->op->block_ordinal < rhs->op->block_ordinal;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_read_motion_sort_barriers,
                          loom_read_motion_barrier_t,
                          loom_read_motion_barrier_less)

iree_status_t loom_read_motion_analyze_region(
    const loom_module_t* module, loom_region_t* region,
    iree_arena_allocator_t* arena, loom_read_motion_t* out_analysis) {
  *out_analysis = (loom_read_motion_t){.module = module};
  loom_read_motion_builder_t builder = {
      .analysis = out_analysis,
      .arena = arena,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(loom_walk_region(
      module, region, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_read_motion_visit_op, &builder}, arena,
      &walk_result));
  loom_read_motion_sort_barriers(out_analysis->barriers,
                                 out_analysis->barrier_count);
  return iree_ok_status();
}

bool loom_read_motion_can_sink_before(const loom_read_motion_t* analysis,
                                      const loom_op_t* read_op,
                                      const loom_op_t* before_op) {
  if (!loom_motion_op_is_ordinary_load(analysis->module, read_op) ||
      read_op->parent_block != before_op->parent_block ||
      read_op->block_ordinal >= before_op->block_ordinal) {
    return false;
  }
  const uintptr_t block = (uintptr_t)read_op->parent_block;
  iree_host_size_t begin = 0;
  iree_host_size_t end = analysis->barrier_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    const loom_op_t* barrier = analysis->barriers[middle].op;
    if ((uintptr_t)barrier->parent_block < block ||
        ((uintptr_t)barrier->parent_block == block &&
         barrier->block_ordinal <= read_op->block_ordinal)) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin == analysis->barrier_count ||
         analysis->barriers[begin].op->parent_block != read_op->parent_block ||
         analysis->barriers[begin].op->block_ordinal >=
             before_op->block_ordinal;
}
