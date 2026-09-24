// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_SYMBOL_BOUNDARY_GRAPH_H_
#define LOOM_TRANSFORMS_SYMBOL_BOUNDARY_GRAPH_H_

#include "loom/analysis/scc.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Return forwarding summaries distinguish unobserved returns from conflicting
// alternatives with no common forwarded value.
#define LOOM_REFINE_BOUNDARIES_FORWARD_UNSEEN ((int32_t)-2)
#define LOOM_REFINE_BOUNDARIES_FORWARD_NONE ((int32_t)-1)

// Logical function arguments projected into one root region.
typedef struct loom_refine_boundaries_argument_projection_t {
  // Root region projecting the logical function argument list.
  loom_region_t* region;

  // Entry block containing the projected argument values.
  loom_block_t* entry_block;
} loom_refine_boundaries_argument_projection_t;

// One function's signature and per-round return summaries.
typedef struct loom_refine_boundaries_function_t {
  // Function-like wrapper for the bodyful definition.
  loom_func_like_t function;

  // Function body region.
  loom_region_t* body;

  // Declared exit operation kind for direct blocks in |body|.
  loom_op_kind_t body_exit_kind;

  // Canonical logical argument ids from the body entry block.
  const loom_value_id_t* argument_ids;

  // Number of logical arguments.
  uint16_t argument_count;

  // Root-region projections of the logical arguments.
  loom_refine_boundaries_argument_projection_t* argument_projections;

  // Number of entries in argument_projections.
  uint8_t argument_projection_count;

  // Number of function result slots.
  uint16_t result_count;

  // True for a private source callable whose signature and callers are owned
  // by this semantic-call graph. Template, command and target-Low boundaries
  // retain their independently owned contracts.
  bool can_refine_boundary;

  // True when caller facts can refine arguments without recursive feedback.
  // Cyclic functions retain their declared argument contracts.
  bool can_refine_argument_facts;

  // True after return facts have been computed in the current round.
  bool has_return_facts;

  // Joined return facts for each result slot in the current round. Extension
  // IDs are owned by the next boundary fact table for that round.
  loom_value_facts_t* return_facts;

  // Per-result bit saying whether return_facts has an observed return value.
  bool* return_fact_defined;

  // Forwarded argument index for each result slot, or a negative sentinel.
  int32_t* return_forward_argument_indices;

  // Forwarded earlier result index for each result slot, or a negative
  // sentinel.
  int32_t* return_forward_result_indices;
} loom_refine_boundaries_function_t;

// Per-iteration graph and return summaries for direct-call refinement. The
// graph borrows the module and its signature identities; signature mutations
// require rebuilding the graph before the next fixed-point round.
typedef struct loom_refine_boundaries_graph_t {
  // Module being refined.
  loom_module_t* module;

  // Reusable scratch for graph walks and per-function boundary application.
  // Nested successor walks preserve live caller frames through stack-ordered
  // arena checkpoints.
  iree_arena_allocator_t* walk_arena;

  // Dense function nodes.
  loom_refine_boundaries_function_t* functions;

  // Number of function nodes.
  iree_host_size_t function_count;

  // Symbol-id to function-node map.
  iree_host_size_t* symbol_to_node;

  // Number of entries in symbol_to_node.
  iree_host_size_t symbol_to_node_count;
} loom_refine_boundaries_graph_t;

// Reads semantic direct-call boundaries participating in refinement. Returns
// false for other operations and call kinds. |out_call| may be NULL.
bool loom_refine_boundaries_read_call(const loom_module_t* module,
                                      loom_op_t* op, loom_call_like_t* out_call,
                                      loom_symbol_ref_t* out_callee,
                                      loom_value_slice_t* out_operands,
                                      loom_value_slice_t* out_results);

// Resolves a local bodyful callee through the graph's symbol index.
bool loom_refine_boundaries_callee_node(
    const loom_refine_boundaries_graph_t* graph, loom_symbol_ref_t callee,
    iree_host_size_t* out_node);

// Builds graph nodes, argument projections and bottom-up SCC order in |arena|.
// |walk_arena| is borrowed scratch for graph walks and must outlive the graph.
iree_status_t loom_refine_boundaries_build_graph(
    loom_module_t* module, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* walk_arena,
    loom_refine_boundaries_graph_t* out_graph, loom_scc_list_t* out_sccs);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_BOUNDARY_GRAPH_H_
