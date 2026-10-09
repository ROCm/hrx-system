// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_SYMBOL_BOUNDARY_SPECIALIZATION_H_
#define LOOM_TRANSFORMS_SYMBOL_BOUNDARY_SPECIALIZATION_H_

#include "loom/transforms/symbol/boundary_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

// Groups calls to internal functions by concrete result types, clones each
// needed specialization and retargets its calls. Plans live in |arena|. New
// definitions and changed calls invalidate the graph.
iree_status_t loom_refine_boundaries_specialize_internal_boundaries(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    iree_arena_allocator_t* arena, int64_t* out_specialization_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_BOUNDARY_SPECIALIZATION_H_
