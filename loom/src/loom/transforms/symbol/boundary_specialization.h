// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_SYMBOL_BOUNDARY_SPECIALIZATION_H_
#define LOOM_TRANSFORMS_SYMBOL_BOUNDARY_SPECIALIZATION_H_

#include "loom/transforms/symbol/boundary_graph.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_refine_boundaries_specialization_plans_t
    loom_refine_boundaries_specialization_plans_t;

// Allocates empty specialization plans indexed by |graph| function node.
iree_status_t loom_refine_boundaries_specialization_plans_allocate(
    const loom_refine_boundaries_graph_t* graph, bool encodings_only,
    iree_arena_allocator_t* arena,
    loom_refine_boundaries_specialization_plans_t** out_plans);

// Records |op| when it is a specializable call using the caller's
// already-computed local value facts. Boundary types are normalized to callee
// identities; reference argument memory spaces distinguish receiving bodies
// without splitting unrelated numeric facts into separate specializations.
iree_status_t loom_refine_boundaries_specialization_plans_collect_call(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    loom_refine_boundaries_specialization_plans_t* plans,
    const loom_value_fact_table_t* function_facts, loom_op_t* op);

// Clones each required internal specialization and retargets its calls. New
// definitions and changed calls invalidate |graph| and |plans|.
iree_status_t loom_refine_boundaries_specialization_plans_apply(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    loom_refine_boundaries_specialization_plans_t* plans,
    int64_t* out_specialization_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_BOUNDARY_SPECIALIZATION_H_
