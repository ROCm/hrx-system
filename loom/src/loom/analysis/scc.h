// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Strongly connected component analysis for caller-provided graphs.
//
// The utility is deliberately independent of functions, symbols, and Loom IR
// containers. Callers adapt their graph to node ordinals and a successor
// iterator, then consume SCCs in successor-before-predecessor order. That order
// is the natural bottom-up direction for call graph summaries and fixed-point
// propagation.

#ifndef LOOM_ANALYSIS_SCC_H_
#define LOOM_ANALYSIS_SCC_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

// Visits one successor node ordinal.
typedef iree_status_t (*loom_scc_successor_fn_t)(
    void* user_data, iree_host_size_t successor_node);

typedef struct loom_scc_successor_callback_t {
  // Callback invoked once for each outgoing successor node ordinal.
  loom_scc_successor_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_scc_successor_callback_t;

// Returns a successor callback wrapping |fn| and |user_data|.
static inline loom_scc_successor_callback_t loom_scc_successor_callback_make(
    loom_scc_successor_fn_t fn, void* user_data) {
  return (loom_scc_successor_callback_t){
      /*.fn=*/fn,
      /*.user_data=*/user_data,
  };
}

// Iterates the outgoing successors of one node. The callback must invoke
// |successor| once for each successor and return the first successor failure.
typedef iree_status_t (*loom_scc_visit_successors_fn_t)(
    void* user_data, iree_host_size_t node,
    loom_scc_successor_callback_t successor);

typedef struct loom_scc_visit_successors_callback_t {
  // Callback used to enumerate outgoing edges from one node.
  loom_scc_visit_successors_fn_t fn;
  // Caller-owned graph adapter payload passed to |fn|.
  void* user_data;
} loom_scc_visit_successors_callback_t;

// Returns a graph successor-iteration callback wrapping |fn| and |user_data|.
static inline loom_scc_visit_successors_callback_t
loom_scc_visit_successors_callback_make(loom_scc_visit_successors_fn_t fn,
                                        void* user_data) {
  return (loom_scc_visit_successors_callback_t){
      /*.fn=*/fn,
      /*.user_data=*/user_data,
  };
}

// Graph adapter consumed by the SCC utility.
typedef struct loom_scc_graph_t {
  // Number of dense node ordinals in the graph.
  iree_host_size_t node_count;

  // Callback used to enumerate outgoing edges from each node.
  loom_scc_visit_successors_callback_t visit_successors;
} loom_scc_graph_t;

// Optional root filter for SCC computation.
typedef struct loom_scc_options_t {
  // Root nodes to traverse. NULL means every graph node is a root only when
  // root_count is zero.
  const iree_host_size_t* root_nodes;

  // Number of entries in root_nodes. Zero with non-NULL root_nodes produces an
  // empty result.
  iree_host_size_t root_count;
} loom_scc_options_t;

// One strongly connected component.
typedef struct loom_scc_t {
  // Component node ordinals. The order inside one component is not semantic.
  const iree_host_size_t* nodes;

  // Number of entries in nodes.
  iree_host_size_t node_count;

  // True when the component has a real cycle: either multiple nodes or one node
  // with a self-edge.
  bool is_cycle;
} loom_scc_t;

// Computed SCC list.
typedef struct loom_scc_list_t {
  // Components in successor-before-predecessor order.
  const loom_scc_t* values;

  // Number of entries in values.
  iree_host_size_t count;
} loom_scc_list_t;

// Computes SCCs for |graph| and stores all transient/result memory in |arena|.
//
// When |options| is NULL, all graph nodes are traversed. When options provides
// roots, only root-reachable nodes appear in the result. A nonzero root_count
// requires a non-NULL root_nodes array. Edges to node ordinals outside the
// graph are API-contract errors because they make the caller's graph adapter
// malformed.
iree_status_t loom_scc_compute(const loom_scc_graph_t* graph,
                               const loom_scc_options_t* options,
                               iree_arena_allocator_t* arena,
                               loom_scc_list_t* out_sccs);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_SCC_H_
