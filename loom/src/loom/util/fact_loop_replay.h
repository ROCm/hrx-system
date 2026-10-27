// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Complete nested-loop solves retained within one immutable outer solve.

#ifndef LOOM_UTIL_FACT_LOOP_REPLAY_H_
#define LOOM_UTIL_FACT_LOOP_REPLAY_H_

#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_value_fact_loop_replay_index_t
    loom_value_fact_loop_replay_index_t;
typedef struct loom_value_fact_loop_replay_record_t
    loom_value_fact_loop_replay_record_t;

// Stack-owned solve boundary. Only the outermost frame owns an index; flat
// loops never construct it. Only root definitions may change during a solve;
// captured definitions and the IR remain immutable until the root returns.
// Published facts retain their normal table lifetime.
struct loom_value_fact_loop_replay_t {
  // Enclosing solve on this table, or NULL at the root.
  loom_value_fact_loop_replay_t* parent;
  // Outermost solve whose arena owns all reuse state.
  loom_value_fact_loop_replay_t* root;
  // Loop whose complete body and result publication belongs to this frame.
  loom_op_t* op;
  // Borrowed solve arena; only the root's arena owns retained dependencies.
  iree_arena_allocator_t* arena;
  // Lazily built ownership and dependency index, populated only on the root.
  loom_value_fact_loop_replay_index_t* index;
  // Current nested-loop record, or NULL for a root or a reused solve.
  loom_value_fact_loop_replay_record_t* record;
  // Observation collector suspended while this solve runs.
  loom_value_fact_loop_reads_t* parent_reads;
};

// Enters an immutable solve, observing its fact dependencies unless its last
// complete publication is reusable. Operand-refinement scopes retain ordinary
// solving because their cached constraints are owned outside this boundary.
// Failure leaves the caller's frame active.
iree_status_t loom_value_fact_loop_replay_begin(
    loom_value_fact_table_t* table, const loom_module_t* module, loom_op_t* op,
    iree_arena_allocator_t* arena, loom_value_fact_loop_replay_t* replay,
    bool* out_reused);

// Publishes successful observations and restores the parent on every path.
// The root removes all region bindings before its caller releases the arena.
iree_status_t loom_value_fact_loop_replay_end(
    loom_value_fact_table_t* table, loom_value_fact_loop_replay_t* replay,
    bool completed);

// A reset or withdrawal may erase a previously solved body's facts. Expire
// completed records even when their external inputs are unchanged.
void loom_value_fact_loop_replay_invalidate(loom_value_fact_table_t* table);

// Bindings borrow the active root's arena and use the table's existing region
// index. They are cleared at the same boundary on success and failure.
iree_status_t loom_value_fact_table_set_loop_replay(
    loom_value_fact_table_t* table, const loom_region_t* body,
    loom_value_fact_loop_replay_record_t* record);
loom_value_fact_loop_replay_record_t* loom_value_fact_table_lookup_loop_replay(
    const loom_value_fact_table_t* table, const loom_region_t* body);
void loom_value_fact_table_clear_loop_replay(loom_value_fact_table_t* table,
                                             const loom_region_t* body);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_FACT_LOOP_REPLAY_H_
