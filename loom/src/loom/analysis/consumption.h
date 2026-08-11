// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Queries for linear value consumption.
//
// Tied and moved results consume their source operands: after the consuming op
// executes along one dynamic path, later operations on that same path must
// observe the result, not the consumed value. CFG regions make that a
// path-sensitive question because a value can be re-created by a block argument
// or an earlier same-block definition on a later dynamic entry.

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
  // Arena used for owned CFG extraction and reusable DFS scratch.
  iree_arena_allocator_t* arena;
  // CFG graph for the region, built lazily or copied from shared analysis.
  loom_cfg_graph_t cfg_graph;
  // True once cfg_graph has been initialized.
  bool cfg_graph_ready;
  // Reusable visited bitset for CFG searches.
  uint64_t* visited_bits;
  // Allocated word capacity of visited_bits.
  iree_host_size_t visited_word_capacity;
  // Reusable bitset of blocks reachable before |value_id| is recreated.
  uint64_t* reachable_bits;
  // Allocated word capacity of reachable_bits.
  iree_host_size_t reachable_word_capacity;
  // Reusable DFS stack of dense CFG block indices.
  uint16_t* block_stack;
  // Allocated element capacity of block_stack.
  iree_host_size_t block_stack_capacity;
} loom_consumption_region_query_t;

// Prepared dynamic-path query for uses after one consuming operation.
//
// The query borrows scratch from |region_query| and remains valid until the
// next use-after query is prepared from that region query.
typedef struct loom_consumption_use_after_query_t {
  // Reusable region query owning CFG and reachability scratch.
  const loom_consumption_region_query_t* region_query;
  // Operation after which uses are queried.
  const loom_op_t* consuming_op;
  // Value whose dynamic instance is being followed.
  loom_value_id_t value_id;
  // Number of words populated in region_query->reachable_bits.
  iree_host_size_t reachable_word_count;
} loom_consumption_use_after_query_t;

// Initializes reusable consumption query state for |region|. CFG extraction is
// lazy: regions without consumed values do not pay graph construction.
void loom_consumption_region_query_initialize(
    const loom_module_t* module, const loom_region_t* region,
    iree_arena_allocator_t* arena, loom_consumption_region_query_t* out_query);

// Initializes reusable consumption query state with a prebuilt CFG graph.
//
// The query copies the graph view and borrows its arena-owned arrays. The graph
// must describe |region| and remain immutable while the query is used.
void loom_consumption_region_query_initialize_with_cfg_graph(
    const loom_module_t* module, const loom_region_t* region,
    const loom_cfg_graph_t* cfg_graph, iree_arena_allocator_t* arena,
    loom_consumption_region_query_t* out_query);

// Prepares a reusable path query for uses of |value_id| that can dynamically
// execute after |consuming_op|. This walks CFG edges once and does not scan IR
// operations; individual uses can then be tested in constant time.
iree_status_t loom_consumption_use_after_query_prepare(
    loom_consumption_region_query_t* region_query,
    const loom_op_t* consuming_op, loom_value_id_t value_id,
    loom_consumption_use_after_query_t* out_query);

// Returns true when |use| can dynamically execute after the consuming
// operation represented by |query|. |use| must belong to the queried value.
bool loom_consumption_use_after_query_contains(
    const loom_consumption_use_after_query_t* query, loom_use_t use);

// Finds a use of |value_id| that can dynamically execute after |consuming_op|.
// |query| must describe |consuming_op|'s parent region. The value's use list
// supplies candidates; non-CFG regions only check later uses in the same block,
// while CFG regions reuse graph and reachability scratch across calls.
iree_status_t loom_consumption_find_use_after(
    loom_consumption_region_query_t* query, const loom_op_t* consuming_op,
    loom_value_id_t value_id, loom_consumption_use_t* out_use, bool* out_found);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_CONSUMPTION_H_
