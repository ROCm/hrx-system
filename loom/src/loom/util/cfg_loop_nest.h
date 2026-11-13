// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_UTIL_CFG_LOOP_NEST_H_
#define LOOM_UTIL_CFG_LOOP_NEST_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/util/cfg_dominance.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_CFG_LOOP_NEST_NONE UINT16_MAX
#define LOOM_CFG_LOOP_CONTINUATION_NONE UINT16_MAX

// Compact summary of one class of loop edges, including parallel edges.
typedef struct loom_cfg_loop_edge_summary_t {
  // Number of reachable edges in this class.
  uint32_t count;
  // Graph edge index when count is one, otherwise LOOM_CFG_EDGE_INDEX_INVALID.
  loom_cfg_edge_index_t unique_index;
} loom_cfg_loop_edge_summary_t;

// A natural loop is the union of cycles returning to one dominating header.
// Its entry and exit edges need not be adjacent to each other in the CFG or
// in block layout. Multiple latches and exits remain represented explicitly.
typedef struct loom_cfg_natural_loop_t {
  // Dense index of the header dominating every loop member.
  uint16_t header_index;
  // Immediately enclosing natural loop, or LOOM_CFG_LOOP_NEST_NONE.
  uint16_t parent_loop_index;
  // Packed inclusive loop-tree preorder range: first low, last high 16 bits.
  uint32_t interval;
  // Number of exits sourced by blocks that belong directly to this loop.
  // Exits propagated from nested loops are excluded.
  uint32_t direct_exit_count;
  // Shared destination of every exit, or CONTINUATION_NONE when exits have
  // distinct destinations or the loop has no exits.
  uint16_t continuation_index;
  // Edges entering the header from outside the loop.
  loom_cfg_loop_edge_summary_t entries;
  // Edges returning to the header from inside the loop.
  loom_cfg_loop_edge_summary_t backedges;
  // Edges from loop members to blocks outside the loop.
  loom_cfg_loop_edge_summary_t exits;
} loom_cfg_natural_loop_t;

// Immutable semantic loop structure for one graph snapshot. Unlike canonical
// layout intervals, this nesting is independent of textual block order.
typedef struct loom_cfg_loop_nest_t {
  // Borrowed graph; it must outlive the nest and retain its topology/indices.
  const loom_cfg_graph_t* graph;
  // Natural loops in child-before-parent order, owned by the caller's arena.
  const loom_cfg_natural_loop_t* loops;
  // Innermost natural loop per block, or NONE. NULL when loop_count is zero.
  const uint16_t* innermost_loop_indices;
  // Number of entries in loops.
  iree_host_size_t loop_count;
  // True when every reachable cycle has a dominating header. Irreducible
  // graphs still retain their natural subloops; malformed graphs are false.
  bool reducible;
} loom_cfg_loop_nest_t;

// Builds loop nesting from retained adjacency, DFS and dominance facts.
// Discovery uses dominance and path-compressed subloop contraction, taking
// O((B+E) log B) time and O(B+E) space without inclusive per-loop block lists.
// Entry/exit summaries, common exit destinations, and loop-tree intervals take
// O(B+E) time. Scratch storage is released before returning; retained storage
// is O(B+L).
// The graph must be produced by loom_cfg_graph_build. Rebuild after topology
// changes. Acyclic graphs require no traversal or allocation.
iree_status_t loom_cfg_loop_nest_build(const loom_cfg_graph_t* graph,
                                       const loom_cfg_dominance_t* dominance,
                                       iree_arena_allocator_t* arena,
                                       loom_cfg_loop_nest_t* out_nest);

// Returns the innermost loop for a valid dense block index, or NONE.
static inline uint16_t loom_cfg_loop_nest_innermost(
    const loom_cfg_loop_nest_t* nest, uint16_t block_index) {
  return nest->loop_count ? nest->innermost_loop_indices[block_index]
                          : LOOM_CFG_LOOP_NEST_NONE;
}

// Constant-time membership query for a loop and valid dense block index.
static inline bool loom_cfg_loop_nest_contains(const loom_cfg_loop_nest_t* nest,
                                               uint16_t loop_index,
                                               uint16_t block_index) {
  uint16_t inner = loom_cfg_loop_nest_innermost(nest, block_index);
  if (loop_index == LOOM_CFG_LOOP_NEST_NONE ||
      inner == LOOM_CFG_LOOP_NEST_NONE) {
    return false;
  }
  uint32_t interval = nest->loops[loop_index].interval;
  uint16_t position = (uint16_t)nest->loops[inner].interval;
  return (uint16_t)interval <= position && position <= (interval >> 16);
}

// Expands exact trip counts for header-tested, single-entry/single-backedge
// loops into block execution counts in O(B+L) time without extra storage.
// Returns false for irreducible cycles, non-header exits, conditional paths
// outside modeled loop headers, or overflow. A header executes trip_count+1
// times per entry; other loop blocks execute trip_count times. Nested counts
// are multiplied.
bool loom_cfg_loop_nest_calculate_block_execution_counts(
    const loom_cfg_loop_nest_t* nest, const uint64_t* trip_counts,
    uint64_t* out_block_counts);

// Computes bounds and reports single-path structure in one retained-graph pass.
// Returns whether all bounds are finite. out_exact distinguishes modeled loop
// headers from conditional paths independently of boundedness. Output storage
// is initialized even for partially bounded or irreducible control.
bool loom_cfg_loop_nest_calculate_bounds(const loom_cfg_loop_nest_t* nest,
                                         const uint64_t* trip_bounds,
                                         uint64_t* out_block_bounds,
                                         bool* out_exact);

// Retains an upper bound for each block even when another path has an unknown
// count. UINT64_MAX denotes an unknown or overflowing trip/block bound.
// Conditional paths receive their enclosing loop multiplier; unreachable
// blocks receive zero. Unsupported cycles remain unbounded without discarding
// bounds outside them. Takes O(B+L) time without additional storage.
// Returns whether every output is exact, provided finite trip_bounds are exact.
// Outputs are always initialized, including when the return value is false.
bool loom_cfg_loop_nest_calculate_block_execution_bounds(
    const loom_cfg_loop_nest_t* nest, const uint64_t* trip_bounds,
    uint64_t* out_block_bounds);

// Orders each reachable block once, visiting complete loop bodies before their
// continuations. Requires reducible control flow. Conditional alternatives
// retain reverse postorder within their loop scope; they are not a promise of
// execution. The output has graph->reverse_postorder.count entries. Consumers
// retain loop scopes/counts when interpreting repeated effects.
// Takes O(B+L) time and temporary storage, released before returning. Acyclic
// graphs copy retained reverse postorder without allocating.
iree_status_t loom_cfg_loop_nest_order_blocks(const loom_cfg_loop_nest_t* nest,
                                              iree_arena_allocator_t* arena,
                                              uint16_t* out_blocks);

// Expands exact loop trip counts into the execution multiplier contributed by
// enclosing loops for each block. Unlike exact execution counts, multipliers
// remain useful in a CFG with data-dependent alternatives: a separate control
// analysis can mark branch-local blocks unknown while retaining the multiplier
// for blocks after reconvergence. Returns false for unsupported loop structure
// or arithmetic overflow. Expansion takes O(B+L) time without extra storage.
bool loom_cfg_loop_nest_calculate_block_multipliers(
    const loom_cfg_loop_nest_t* nest, const uint64_t* trip_counts,
    uint64_t* out_block_multipliers);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_UTIL_CFG_LOOP_NEST_H_
