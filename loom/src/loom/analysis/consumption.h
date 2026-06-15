// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Queries for linear value consumption.
//
// Tied low results consume their tied operands: after the consuming op executes
// along one dynamic path, later operations on that same path must observe the
// tied result, not the consumed value. CFG regions make that a path-sensitive
// question because a value can be re-created by a block argument or an earlier
// same-block definition on a later dynamic entry.

#ifndef LOOM_ANALYSIS_CONSUMPTION_H_
#define LOOM_ANALYSIS_CONSUMPTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/util/cfg_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

// Operand occurrence that observes a consumed value.
typedef struct loom_consumption_use_t {
  // Operation containing the observing operand.
  const loom_op_t* op;
  // Operand index on |op| that observes the value.
  uint16_t operand_index;
} loom_consumption_use_t;

// Reusable per-region query state for consumed-value checks.
typedef struct loom_consumption_region_query_t {
  // Module containing the queried region.
  const loom_module_t* module;
  // Region whose dynamic paths are queried.
  const loom_region_t* region;
  // Arena used for the lazily built CFG graph and reusable DFS scratch.
  iree_arena_allocator_t* arena;
  // Lazily built CFG graph for region when region is CFG-shaped.
  loom_cfg_graph_t cfg_graph;
  // True once cfg_graph has been initialized.
  bool cfg_graph_ready;
  // Reusable visited bitset for CFG searches.
  uint64_t* visited_bits;
  // Allocated word capacity of visited_bits.
  iree_host_size_t visited_word_capacity;
  // Reusable DFS stack of dense CFG block indices.
  uint16_t* block_stack;
  // Allocated element capacity of block_stack.
  iree_host_size_t block_stack_capacity;
} loom_consumption_region_query_t;

// Initializes reusable consumption query state for |region|. CFG extraction is
// lazy: regions without tied-result consumption do not pay graph construction.
iree_status_t loom_consumption_region_query_initialize(
    const loom_module_t* module, const loom_region_t* region,
    iree_arena_allocator_t* arena, loom_consumption_region_query_t* out_query);

// Finds a use of |value_id| that can dynamically execute after |consuming_op|.
// |query| must describe |consuming_op|'s parent region. Non-CFG regions only
// check later uses in the same block. CFG regions reuse the query's graph and
// DFS scratch across calls.
iree_status_t loom_consumption_find_use_after(
    loom_consumption_region_query_t* query, const loom_op_t* consuming_op,
    loom_value_id_t value_id, loom_consumption_use_t* out_use, bool* out_found);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_CONSUMPTION_H_
