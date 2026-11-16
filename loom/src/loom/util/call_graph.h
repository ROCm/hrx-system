// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Direct-call discovery and local collection over an immutable source module.
//
// An explicit root set admits only reachable function definitions. Omitting
// roots includes every definition in module symbol order. Each admitted body
// is walked once; consumers collect local summaries and callsites during that
// walk. Distinct edges feed shared SCC analysis in callee-before-caller order.
//
// Rooted graph storage scales with admitted functions and distinct edges,
// independent of the module's unused symbol domain. Whole-module discovery
// uses compact direct indexing over its already-selected symbol domain.
// The source module must have complete, resolved local call references and
// remain unchanged throughout discovery.
// Results borrow source identities and live in the caller's arena. The graph
// remains at its original address and expires when the source is rewritten.
//
#ifndef LOOM_UTIL_CALL_GRAPH_H_
#define LOOM_UTIL_CALL_GRAPH_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/scc.h"
#include "loom/ir/ir.h"
#include "loom/util/ordinal_map.h"
#include "loom/util/walk.h"

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
  // Index into graph.components, assigned after discovery completes.
  uint16_t scc_id;
  // Number of distinct callers, including a self-edge for recursion.
  uint16_t caller_count;
} loom_call_graph_node_t;

// Call graph for admitted function definitions in a module.
typedef struct loom_call_graph_t {
  // Immutable source module supplying symbol identities.
  const loom_module_t* module;
  // Number of function-like symbols with bodies.
  uint16_t node_count;
  // Bit (1 << loom_call_like_kind_t) for each observed tracked call kind,
  // including calls to declarations and duplicate edges of different kinds.
  uint8_t call_kinds;
  // One node per admitted definition, in deterministic discovery order.
  loom_call_graph_node_t* nodes;
  // Callee-first components retaining their member nodes and recursion state.
  loom_scc_list_t components;
  // Direct symbol-to-node index for whole-module discovery. NULL for rooted
  // discovery; entries without a definition contain UINT16_MAX.
  uint16_t* symbol_to_node;
  // Symbol domain covered by the direct index at discovery time.
  uint16_t symbol_to_node_count;
  // Sparse symbol-to-node index used only for explicit-root discovery.
  loom_ordinal_map_t reached_symbols;
} loom_call_graph_t;

//===----------------------------------------------------------------------===//
// Call graph construction
//===----------------------------------------------------------------------===//

// Returns true when |kind| contributes an edge to the call graph. Analyses
// consuming the graph use this query to classify the same call boundary once.
bool loom_call_graph_tracks_call_kind(loom_call_like_kind_t kind);

// Local collection callbacks share discovery without owning traversal. They
// may allocate consumer state but must not mutate the source or graph. Node
// ordinals remain stable; node pointers may move while discovery admits
// callees.
typedef struct loom_call_graph_visitor_t {
  // Owner of the local summaries and retained callsite information.
  void* user_data;
  // Receives each admitted definition before its body or an incoming callsite
  // is visited, including functions with empty bodies.
  iree_status_t (*function)(void* user_data, uint16_t node_index,
                            loom_func_like_t function);
  // Receives every body operation in preorder. |call| is populated only for a
  // tracked call, avoiding a second interface lookup by local classification.
  iree_status_t (*operation)(void* user_data, uint16_t node_index,
                             loom_op_t* op, loom_call_like_t call,
                             const loom_walk_context_t* context);
  // Receives every tracked call in walk order, including repeated edges.
  iree_status_t (*call)(void* user_data, loom_symbol_id_t caller,
                        loom_call_like_t call);
} loom_call_graph_visitor_t;

typedef struct loom_call_graph_options_t {
  // Callable symbols whose transitive definitions are admitted. NULL selects
  // all definitions; a non-NULL empty array selects none. Declarations have no
  // graph node. Explicit roots precede transitively discovered definitions.
  const loom_symbol_id_t* roots;
  // Number of entries in |roots|.
  iree_host_size_t root_count;
  // Optional local collectors invoked during the owning body walk.
  loom_call_graph_visitor_t visitor;
} loom_call_graph_options_t;

// Walks every root region of each admitted definition, records distinct
// edges, and computes SCCs. NULL |options| selects all definitions without
// local collection. Rooted discovery never scans unused symbol entries/bodies.
// Discovery costs O(N + C), for admitted body operations and callsites; SCC
// analysis costs O(V + E), for admitted definitions and distinct call edges.
// Unrooted construction additionally enumerates the module's symbol table.
// All output and temporary storage comes from |arena| and its reusable pool.
iree_status_t loom_call_graph_build(const loom_module_t* module,
                                    const loom_call_graph_options_t* options,
                                    iree_arena_allocator_t* arena,
                                    loom_call_graph_t* out_graph);

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

// Returns the call graph node for |symbol_id|, or NULL if the symbol is not
// admitted (including declarations and identities outside the snapshot).
const loom_call_graph_node_t* loom_call_graph_node(
    const loom_call_graph_t* graph, loom_symbol_id_t symbol_id);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_CALL_GRAPH_H_
