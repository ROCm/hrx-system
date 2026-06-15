// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Call graph construction and SCC analysis.
//
// Builds a call graph from direct call-like ops in a module, then computes
// strongly connected components (SCCs) using Tarjan's algorithm. SCCs are
// returned in reverse topological order (callees before callers), which is the
// right iteration order for bottom-up analyses like purity inference, cost
// modeling, and inlining decisions.
//
// Construction: walks each function body looking for direct semantic,
// target-low-internal, and explicit low-invoke call-like ops, then records
// (caller, callee) edges. Total cost: O(N) where N is the total number of ops
// across all function bodies.
//
// SCC: Tarjan's iterative algorithm, O(V+E) where V = functions and
// E = call edges. Typically trivial (< 100 nodes for ML models).
//
// All memory is arena-allocated. The call graph is valid for the
// lifetime of the arena.
//
// Usage:
//
//   loom_call_graph_t graph;
//   IREE_RETURN_IF_ERROR(loom_call_graph_build(module, &arena, &graph));
//
//   // Iterate functions in bottom-up order (callees first).
//   for (uint16_t i = 0; i < graph.node_count; ++i) {
//     const loom_call_graph_node_t* node = &graph.nodes[i];
//     if (loom_call_graph_is_recursive(&graph, node->symbol_id)) {
//       // Handle recursive function.
//     }
//   }

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
  // Arena-allocated array of callee symbol IDs. Deduplicated.
  loom_symbol_id_t* callees;
  // SCC ID after loom_call_graph_build completes. Functions in the
  // same SCC share the same scc_id. UINT16_MAX before SCC analysis.
  uint16_t scc_id;
} loom_call_graph_node_t;

// Call graph for all functions in a module.
typedef struct loom_call_graph_t {
  const loom_module_t* module;
  // One node per function-like symbol with a body.
  uint16_t node_count;
  loom_call_graph_node_t* nodes;
  // SCC count after analysis. SCCs are numbered 0..scc_count-1.
  uint16_t scc_count;
  // Per-SCC: true if the SCC is recursive (>1 member or self-call).
  bool* scc_recursive;
  // Maps symbol_id → node index (UINT16_MAX if not in the graph).
  // Enables O(1) lookup from symbol ID.
  uint16_t* symbol_to_node;
  uint16_t symbol_to_node_count;
} loom_call_graph_t;

//===----------------------------------------------------------------------===//
// Call graph construction
//===----------------------------------------------------------------------===//

// Builds the call graph for all function-like symbols in |module|.
// Walks each function body looking for direct call-like ops, records edges,
// deduplicates callee lists, and computes SCCs. All memory is allocated from
// |arena|.
iree_status_t loom_call_graph_build(const loom_module_t* module,
                                    iree_arena_allocator_t* arena,
                                    loom_call_graph_t* out_graph);

//===----------------------------------------------------------------------===//
// Queries
//===----------------------------------------------------------------------===//

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
