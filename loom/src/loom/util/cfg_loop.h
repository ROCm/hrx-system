// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical loop intervals preserved from a dense CFG graph.

#ifndef LOOM_UTIL_CFG_LOOP_H_
#define LOOM_UTIL_CFG_LOOP_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/util/cfg_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel for the absence of a loop interval.
#define LOOM_CFG_LOOP_NONE UINT32_MAX

// One candidate single-entry loop interval in deterministic block order.
typedef struct loom_cfg_loop_interval_t {
  // Dense block index of the loop header.
  uint16_t header_index;
  // Dense block index of the unique backward-edge source.
  uint16_t latch_index;
  // Dense block index of the dedicated entry predecessor.
  uint16_t preheader_index;
  // Immediately enclosing interval, or LOOM_CFG_LOOP_NONE.
  uint32_t parent_loop_index;
  // CFG edge entering the header from |preheader_index|.
  loom_cfg_edge_index_t entry_edge_index;
  // CFG edge entering the header from |latch_index|.
  loom_cfg_edge_index_t backedge_edge_index;
  // True when the interval is properly nested and has no side entry.
  bool is_canonical;
} loom_cfg_loop_interval_t;

// Immutable canonical loop forest derived from one CFG graph.
typedef struct loom_cfg_loop_forest_t {
  // CFG graph whose block and edge indices are referenced by the forest.
  const loom_cfg_graph_t* graph;
  // Loop intervals in increasing header-block order.
  const loom_cfg_loop_interval_t* intervals;
  // Innermost candidate interval per CFG block, or LOOM_CFG_LOOP_NONE.
  const uint32_t* innermost_loop_indices;
  // Number of entries in |intervals|.
  iree_host_size_t interval_count;
} loom_cfg_loop_forest_t;

// Builds canonical loop intervals for |graph| in |arena|.
//
// Construction takes O(B+E+L) time for B blocks, E edges, and L candidate
// intervals. Graphs without backward edges perform no traversal or allocation.
// Crossing intervals conservatively produce an empty forest. Noncanonical
// nested intervals remain represented with |is_canonical| false so consumers
// can continue with an enclosing canonical interval.
iree_status_t loom_cfg_loop_forest_build(const loom_cfg_graph_t* graph,
                                         iree_arena_allocator_t* arena,
                                         loom_cfg_loop_forest_t* out_forest);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_UTIL_CFG_LOOP_H_
