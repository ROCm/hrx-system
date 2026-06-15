// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Arena-backed control-flow graph extraction for Loom regions.
//
// The graph is a dense view over a single region's ordered block table.
// Successor edges are read from successor-bearing terminators. Malformed edges
// such as NULL targets, targets outside the region, or successor-bearing ops
// before the block terminator set graph->malformed and are omitted from the
// adjacency lists so analyses can answer conservatively while the verifier owns
// user-facing diagnostics.

#ifndef LOOM_UTIL_CFG_GRAPH_H_
#define LOOM_UTIL_CFG_GRAPH_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// A read-only span of dense block indices.
typedef struct loom_cfg_block_index_span_t {
  // Dense block indices in region block order.
  const uint16_t* values;
  // Number of entries in values.
  iree_host_size_t count;
} loom_cfg_block_index_span_t;

typedef uint32_t loom_cfg_edge_index_t;
#define LOOM_CFG_EDGE_INDEX_INVALID ((loom_cfg_edge_index_t)UINT32_MAX)

// A read-only span of dense CFG edge indices.
typedef struct loom_cfg_edge_index_span_t {
  // Dense edge indices in graph edge order.
  const loom_cfg_edge_index_t* values;
  // Number of entries in values.
  iree_host_size_t count;
} loom_cfg_edge_index_span_t;

// Per-edge CFG metadata. Edges are stable for the lifetime of the graph and
// record the originating terminator plus the concrete successor ordinal so
// analyses can ask edge-local questions without re-walking blocks.
typedef struct loom_cfg_edge_info_t {
  // Terminator op that owns this successor edge.
  const loom_op_t* terminator;
  // Dense source block index in graph block order.
  uint16_t source_block_index;
  // Dense target block index in graph block order.
  uint16_t target_block_index;
  // Successor ordinal on terminator.
  uint16_t successor_index;
  // Selector value for successor alternatives, or LOOM_VALUE_ID_INVALID when
  // the terminator has no declared selector.
  loom_value_id_t selector_value_id;
} loom_cfg_edge_info_t;

// Per-block adjacency metadata inside a CFG graph.
typedef struct loom_cfg_block_info_t {
  // Region block represented by this dense graph node.
  const loom_block_t* block;
  // Offset into loom_cfg_graph_t::successor_indices.
  iree_host_size_t successor_start;
  // Number of outgoing successor indices for this block.
  iree_host_size_t successor_count;
  // Offset into loom_cfg_graph_t::successor_edge_indices.
  iree_host_size_t successor_edge_start;
  // Offset into loom_cfg_graph_t::predecessor_indices.
  iree_host_size_t predecessor_start;
  // Number of incoming predecessor indices for this block.
  iree_host_size_t predecessor_count;
  // Offset into loom_cfg_graph_t::predecessor_edge_indices.
  iree_host_size_t predecessor_edge_start;
  // True when the block is reachable from the region entry block.
  bool reachable;
} loom_cfg_block_info_t;

// Dense CFG adjacency for one region.
typedef struct loom_cfg_graph_t {
  // Module containing region.
  const loom_module_t* module;
  // Region whose block table defines the dense graph node order.
  const loom_region_t* region;
  // Per-block adjacency metadata with block_count entries.
  loom_cfg_block_info_t* blocks;
  // Dense edge metadata with edge_count entries.
  loom_cfg_edge_info_t* edges;
  // Dense target block indices for all outgoing edges.
  uint16_t* successor_indices;
  // Dense edge indices for all outgoing edges.
  loom_cfg_edge_index_t* successor_edge_indices;
  // Dense source block indices for all incoming edges.
  uint16_t* predecessor_indices;
  // Dense edge indices for all incoming edges.
  loom_cfg_edge_index_t* predecessor_edge_indices;
  // Number of dense block nodes.
  iree_host_size_t block_count;
  // Number of valid in-region successor edges.
  iree_host_size_t edge_count;
  // True when malformed successor structure was seen while building the graph.
  bool malformed;
} loom_cfg_graph_t;

// Builds a dense CFG graph for |region| and stores all result memory in
// |arena|. Passing NULL arguments is an API error. Malformed IR is represented
// by graph->malformed rather than a failing status so callers can stay
// conservative and let verification produce structured diagnostics.
iree_status_t loom_cfg_graph_build(const loom_module_t* module,
                                   const loom_region_t* region,
                                   iree_arena_allocator_t* arena,
                                   loom_cfg_graph_t* out_graph);

// Returns the dense index of |block| in |graph|, or IREE_HOST_SIZE_MAX when the
// block is not owned by the graph's region.
iree_host_size_t loom_cfg_graph_block_index(const loom_cfg_graph_t* graph,
                                            const loom_block_t* block);

// Returns outgoing successor block indices for |block_index|.
loom_cfg_block_index_span_t loom_cfg_graph_successors(
    const loom_cfg_graph_t* graph, uint16_t block_index);

// Returns outgoing successor edge indices for |block_index|.
loom_cfg_edge_index_span_t loom_cfg_graph_successor_edges(
    const loom_cfg_graph_t* graph, uint16_t block_index);

// Returns incoming predecessor block indices for |block_index|.
loom_cfg_block_index_span_t loom_cfg_graph_predecessors(
    const loom_cfg_graph_t* graph, uint16_t block_index);

// Returns incoming predecessor edge indices for |block_index|.
loom_cfg_edge_index_span_t loom_cfg_graph_predecessor_edges(
    const loom_cfg_graph_t* graph, uint16_t block_index);

// Returns edge metadata for |edge_index|, or NULL when out of range.
const loom_cfg_edge_info_t* loom_cfg_graph_edge(
    const loom_cfg_graph_t* graph, loom_cfg_edge_index_t edge_index);

// Returns true when |block_index| is reachable from the region entry block.
bool loom_cfg_graph_block_is_reachable(const loom_cfg_graph_t* graph,
                                       uint16_t block_index);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_CFG_GRAPH_H_
