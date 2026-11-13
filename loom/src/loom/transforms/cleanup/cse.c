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
    if (!region) {
      continue;
    }
    iree_arena_reset(&arena);
    loom_expression_walk_t* walk = NULL;
    status =
        loom_expression_walk_allocate(module, region,
                                      /*identity_policy=*/NULL,
                                      /*maximum_entry_count=*/0, &arena, &walk);
    loom_expression_barriers_t barriers = {0};
    loom_expression_hash_cache_t hash_cache;
    loom_expression_hash_cache_initialize(&hash_cache);
    while (iree_status_is_ok(status)) {
      loom_expression_cursor_t cursor;
      status = loom_expression_walk_next(walk, &cursor);
      if (!iree_status_is_ok(status) || !cursor.op) {
        break;
      }
      uint32_t minimum_epoch =
          loom_expression_observe_barriers(&cursor, &barriers);
      // Source SSA values retain their meaning across collective execution.
      // Convergence constrains the collective itself, not a dominating pure
      // expression. Masked machine definitions have a separate Low CSE policy.
      if (iree_any_bit_set(cursor.traits, LOOM_TRAIT_PURE)) {
        minimum_epoch = 0;
      }
      bool eliminated = false;
      if (loom_expression_op_is_reusable(module, cursor.op, cursor.traits) &&
          !loom_op_requires_source_context(module, cursor.op)) {
        const uint32_t hash =
            loom_expression_hash(module, cursor.op, &hash_cache);
        const loom_expression_lookup_flags_t flags =
            iree_any_bit_set(cursor.traits, LOOM_TRAIT_PURE)
                ? 0
                : LOOM_EXPRESSION_LOOKUP_FLAG_STATEFUL;
        const loom_expression_probe_t probe =
            loom_expression_scope_probe(&cursor, hash, flags);
        loom_expression_entry_t* existing = probe.entry;
        if (existing && existing->metadata >= minimum_epoch) {
          status = loom_expression_replace(&cursor, cursor.op, existing->op);
          if (iree_status_is_ok(status)) {
            eliminated = true;
            loom_pass_mark_changed(pass);
            ++statistics->expressions_eliminated;
          }
        } else {
          loom_expression_scope_insert(&cursor, probe,
                                       (loom_expression_entry_t){
                                           .op = cursor.op,
                                           .hash = hash,
                                           .metadata = cursor.epoch,
                                       });
        }
      }
      if (!eliminated && iree_status_is_ok(status) &&
          loom_traits_are_fact_identity(cursor.traits)) {
        status = loom_expression_scope_record_fact_identity(&cursor);
      }
    }
  }
  iree_arena_deinitialize(&arena);
  return status;
}
