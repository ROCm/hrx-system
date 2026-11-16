// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_CALLABLE_INPUTS_H_
#define LOOM_ANALYSIS_CALLABLE_INPUTS_H_

#include "loom/analysis/scc.h"

#ifdef __cplusplus
extern "C" {
#endif

// Direct requirement published by the owner of a callable's local summary.
typedef struct loom_callable_input_requirement_t {
  // Callable ordinal in the admitted graph.
  iree_host_size_t node;
  // Typed input identity in the caller-owned key table.
  iree_host_size_t input;
} loom_callable_input_requirement_t;

// Transitive requirements over one immutable admitted callable graph. Edges
// point from callers to callees and exclude invocation boundaries that supply
// inputs locally. Source discovery and selected lowering each own their graph
// adapter and keys; this analysis neither walks IR nor chooses suppliers.
typedef struct loom_callable_inputs_t {
  // Row-major input membership, borrowed from the analysis arena.
  const uint64_t* words;
  // Number of words in each callable's row; zero for an empty input universe.
  iree_host_size_t words_per_node;
} loom_callable_inputs_t;

// Joins local requirements through producer-owned callee-first components.
// The graph and components describe the same admitted nodes; every requirement
// names a valid node and an input below input_count. These are trusted analysis
// products, not a second user-input validation boundary.
//
// O((V + E) * ceil(input_count / 64)) work and O(V * ceil(input_count / 64))
// arena storage. V is the affected closure selected by the caller, not the
// module's symbol domain. Zero inputs allocate nothing and visit no edges.
// Result membership is independent of discovery and SCC member order.
iree_status_t loom_callable_inputs_compute(
    const loom_scc_graph_t* graph, const loom_scc_list_t* components,
    iree_host_size_t input_count,
    const loom_callable_input_requirement_t* requirements,
    iree_host_size_t requirement_count, iree_arena_allocator_t* arena,
    loom_callable_inputs_t* out_inputs);

// Tests one valid node/input pair in the completed analysis.
static inline bool loom_callable_inputs_contains(
    const loom_callable_inputs_t* inputs, iree_host_size_t node,
    iree_host_size_t input) {
  return (inputs->words[node * inputs->words_per_node + input / 64] &
          (UINT64_C(1) << (input % 64))) != 0;
}

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_CALLABLE_INPUTS_H_
