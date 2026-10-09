// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_SYMBOL_BOUNDARY_PRUNING_H_
#define LOOM_TRANSFORMS_SYMBOL_BOUNDARY_PRUNING_H_

#include "loom/transforms/symbol/boundary_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

// Removes unused internal arguments and results after checking every call.
// Plans live in |arena|; |scratch_arena| holds temporary call-rewrite data.
// Rewrites calls and returns before changing definitions. Signature changes
// invalidate the graph and its borrowed argument projections.
iree_status_t loom_refine_boundaries_prune_internal_boundaries(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch_arena,
    int64_t* out_pruned_argument_count, int64_t* out_pruned_result_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_BOUNDARY_PRUNING_H_
