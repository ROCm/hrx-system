// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Call graph construction and SCC analysis.
//
// Builds a call graph from direct call-like ops in a module, then computes
// strongly connected components (SCCs) using Tarjan's algorithm. SCC IDs are
// assigned in reverse topological order (callees before callers), which is the
// right iteration order for bottom-up analyses like effect inference, cost
// modeling, and inlining decisions. Nodes remain in module symbol order.
//
// Construction: walks every root region of each function-like symbol looking
// for direct semantic, target-low-internal, explicit low-invoke, and command
// program call-like ops, then records (caller, callee) edges. Total cost:
// O(S+N), where S is the symbol domain and N is the total number of ops across
// all function root regions. Callee membership is constant-time per callsite.
//
// SCC: Tarjan's iterative algorithm, O(V+E) where V = functions and
// E = call edges. Typically trivial (< 100 nodes for ML models).
//
// All memory is arena-allocated. The call graph is valid for the
// lifetime of the arena.
//
#ifndef LOOM_UTIL_CALL_GRAPH_H_
#define LOOM_UTIL_CALL_GRAPH_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Call graph types
//===----------------------------------------------------------------------===//

// Per-function node in the call graph.
typedef struct loom_call_graph_node_t {
  // Symbol ID of the function in the module's symbol table.
  loom_symbol_id_t symbol_id;
  // Number of distinct callees.
  uint16_t callee_count;
  // Arena-allocated callee symbol IDs, deduplicated in first-callsite order.
  loom_symbol_id_t* callees;
  // SCC ID after loom_call_graph_build completes. Functions in the
  // same SCC share the same scc_id. UINT16_MAX before SCC analysis.
  uint16_t scc_id;
  // Number of distinct callers, including a self-edge for recursion.
  uint16_t caller_count;
} loom_call_graph_node_t;

// Call graph for all functions in a module.
typedef struct loom_call_graph_t {
  // Immutable source module supplying symbol identities.
  const loom_module_t* module;
  // Number of function-like symbols with bodies.
  uint16_t node_count;
  // Bit (1 << loom_call_like_kind_t) for each observed tracked call kind,
  // including calls to declarations and duplicate edges of different kinds.
  uint8_t call_kinds;
  // One node per definition, in module symbol order.
  loom_call_graph_node_t* nodes;
  // SCC count after analysis. SCCs are numbered 0..scc_count-1.
  uint16_t scc_count;
  // Per-SCC: true if the SCC is recursive (>1 member or self-call).
  bool* scc_recursive;
  // Maps symbol_id → node index (UINT16_MAX if not in the graph).
  // Enables O(1) lookup from symbol ID.
  uint16_t* symbol_to_node;
  // Symbol domain covered by symbol_to_node.
  uint16_t symbol_to_node_count;
} loom_call_graph_t;

//===----------------------------------------------------------------------===//
// Call graph construction
//===----------------------------------------------------------------------===//

// Returns true when |kind| contributes an edge to the call graph. Analyses
// consuming the graph use this query to classify the same call boundary once.
bool loom_call_graph_tracks_call_kind(loom_call_like_kind_t kind);

// Builds the call graph for all function-like symbols in |module|. Walks every
// root region looking for tracked direct call-like ops, records edges,
// deduplicates callee lists, and computes SCCs. All memory is allocated from
// |arena|.
iree_status_t loom_call_graph_build(const loom_module_t* module,
                                    iree_arena_allocator_t* arena,
                                    loom_call_graph_t* out_graph);

// Optional consumer of callsites discovered during graph construction. The
// visitor retains callsite-specific state without a second source traversal.
typedef struct loom_call_graph_call_visitor_t {
  // Owner of the retained callsite information.
  void* user_data;
  // Receives every tracked call in walk order, including repeated edges.
  iree_status_t (*fn)(void* user_data, loom_symbol_id_t caller,
                      loom_call_like_t call);
} loom_call_graph_call_visitor_t;

iree_status_t loom_call_graph_build_with_calls(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_call_graph_call_visitor_t visitor, loom_call_graph_t* out_graph);

//===----------------------------------------------------------------------===//
// Queries
//===----------------------------------------------------------------------===//

// Returns whether construction observed a tracked call of |kind|. This summary
// includes declaration targets, which have no graph node or caller count.
static inline bool loom_call_graph_has_call_kind(const loom_call_graph_t* graph,
                                                 loom_call_like_kind_t kind) {
  return (graph->call_kinds & (1u << kind)) != 0;
}

// Returns true if the function at |symbol_id| is (mutually) recursive.
// A function is recursive if its SCC has more than one member or if
// it has a direct self-call.
bool loom_call_graph_is_recursive(const loom_call_graph_t* graph,
                                  loom_symbol_id_t symbol_id);

// Returns the call graph node for |symbol_id|, or NULL if the
// function is not in the graph (e.g., declaration without a body).
const loom_call_graph_node_t* loom_call_graph_node(
    const loom_call_graph_t* graph, loom_symbol_id_t symbol_id);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_CALL_GRAPH_H_
