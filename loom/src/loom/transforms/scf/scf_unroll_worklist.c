// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scf/scf_unroll_worklist.h"

#include "loom/ops/scf/ops.h"
#include "loom/util/walk.h"

typedef struct loom_scf_unroll_worklist_builder_t {
  // Scratch arena owning the collected source handles.
  iree_arena_allocator_t* arena;
  // Result populated in source preorder.
  loom_scf_unroll_worklist_t* worklist;
  // Allocated item slots.
  iree_host_size_t capacity;
  // Outermost enclosing recurrence's depth, or UINT32_MAX outside recurrence.
  uint32_t recurrence_depth;
} loom_scf_unroll_worklist_builder_t;

bool loom_scf_unroll_policy_present(const loom_op_t* op) {
  return loom_scf_for_unroll_factor_is_present(op) ||
         loom_scf_for_has_unroll_policy(op) ||
         loom_scf_for_has_unroll_schedule(op);
}

static iree_status_t loom_scf_unroll_worklist_collect(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  *out_result = LOOM_WALK_CONTINUE;
  loom_scf_unroll_worklist_builder_t* builder = user_data;
  if (context->depth <= builder->recurrence_depth) {
    builder->recurrence_depth = UINT32_MAX;
  }
  if (!loom_scf_for_isa(op)) {
    return iree_ok_status();
  }
  loom_scf_unroll_worklist_t* worklist = builder->worklist;
  const bool has_policy = loom_scf_unroll_policy_present(op);
  const bool prepare = has_policy && builder->recurrence_depth != UINT32_MAX;
  worklist->has_policy |= has_policy;
  worklist->preparation_candidate_count += prepare;
  if (worklist->count == builder->capacity) {
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(builder->arena, worklist->count,
                              worklist->count + 1, sizeof(*worklist->items),
                              &builder->capacity, (void**)&worklist->items));
  }
  worklist->items[worklist->count++] =
      (loom_scf_unroll_work_item_t){.op = op, .prepare = prepare};
  if (builder->recurrence_depth == UINT32_MAX &&
      loom_scf_for_has_unroll_schedule(op) &&
      loom_scf_for_unroll_schedule(op) ==
          LOOM_SCF_FOR_UNROLL_SCHEDULE_RECURRENCE) {
    builder->recurrence_depth = context->depth;
  }
  return iree_ok_status();
}

iree_status_t loom_scf_unroll_worklist_build(
    loom_module_t* module, loom_func_like_t function,
    iree_arena_allocator_t* arena, loom_scf_unroll_worklist_t* out_worklist) {
  *out_worklist = (loom_scf_unroll_worklist_t){0};
  loom_scf_unroll_worklist_builder_t builder = {
      .arena = arena,
      .worklist = out_worklist,
      .recurrence_depth = UINT32_MAX,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  return loom_walk_function(
      module, function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){.fn = loom_scf_unroll_worklist_collect,
                             .user_data = &builder},
      &walk_result);
}
