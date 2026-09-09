// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cleanup/cse.h"

#include "loom/ops/op_defs.h"
#include "loom/transforms/cleanup/expression_scope.h"

#define LOOM_CSE_STATISTICS(V, statistics_type)                        \
  V(statistics_type, expressions_eliminated, "expressions-eliminated", \
    "Number of redundant expressions removed.")

LOOM_PASS_STATISTICS_DEFINE(loom_cse_statistics, loom_cse_statistics_t,
                            LOOM_CSE_STATISTICS)

static const loom_pass_info_t loom_cse_pass_info_storage = {
    .name = IREE_SVL("cse"),
    .description = IREE_SVL("Eliminate common subexpressions."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_cse_statistics_layout,
};

const loom_pass_info_t* loom_cse_pass_info(void) {
  return &loom_cse_pass_info_storage;
}

iree_status_t loom_cse_run(loom_pass_t* pass, loom_module_t* module,
                           loom_func_like_t function) {
  // Low definitions have a separate descriptor-backed identity policy.
  if (!loom_func_like_body(function) ||
      loom_op_dialect_id(function.op->kind) == LOOM_DIALECT_LOW) {
    return iree_ok_status();
  }
  loom_cse_statistics_t* statistics = loom_cse_statistics(pass);
  iree_arena_allocator_t arena;
  iree_arena_initialize(pass->arena->block_pool, &arena);
  iree_status_t status = iree_ok_status();
  for (uint8_t i = 0;
       i < loom_func_like_region_count(function) && iree_status_is_ok(status);
       ++i) {
    loom_region_t* region = loom_func_like_region(function, i);
    if (!region) continue;
    iree_arena_reset(&arena);
    loom_expression_walk_t* walk = NULL;
    status = loom_expression_walk_initialize(module, region, &arena, &walk);
    loom_expression_barriers_t barriers = {0};
    while (iree_status_is_ok(status)) {
      loom_expression_cursor_t cursor;
      status = loom_expression_walk_next(walk, &cursor);
      if (!iree_status_is_ok(status) || !cursor.op) break;
      const uint32_t minimum_epoch =
          loom_expression_observe_barriers(&cursor, &barriers);
      if (!loom_expression_is_reusable(&cursor)) continue;
      const uint32_t hash = loom_expression_hash(module, cursor.op);
      const loom_expression_lookup_flags_t flags =
          iree_any_bit_set(cursor.traits, LOOM_TRAIT_PURE)
              ? 0
              : LOOM_EXPRESSION_LOOKUP_FLAG_STATEFUL;
      loom_expression_entry_t* existing =
          loom_expression_scope_find(&cursor, hash, flags);
      if (existing && existing->epoch >= minimum_epoch) {
        status = loom_expression_replace(module, cursor.op, existing->op);
        if (iree_status_is_ok(status)) {
          loom_pass_mark_changed(pass);
          ++statistics->expressions_eliminated;
        }
      } else {
        loom_expression_scope_insert(&cursor, (loom_expression_entry_t){
                                                  .op = cursor.op,
                                                  .hash = hash,
                                                  .epoch = cursor.epoch,
                                              });
      }
    }
  }
  iree_arena_deinitialize(&arena);
  return status;
}
