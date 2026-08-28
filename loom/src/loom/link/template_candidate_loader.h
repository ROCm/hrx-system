// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lazy template-provider header loading for specializing links.

#ifndef LOOM_LINK_TEMPLATE_CANDIDATE_LOADER_H_
#define LOOM_LINK_TEMPLATE_CANDIDATE_LOADER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/template_provider_catalog.h"
#include "loom/link/plan_materializer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_link_template_candidate_loader_t
    loom_link_template_candidate_loader_t;

// Allocates a lazy candidate-header loader over |index|.
//
// The loader borrows |index| and |environment| for its lifetime. Materialized
// source modules are summarized once. Bytecode provider headers and the shared
// table entries they reach are decoded at most once and implementation bodies
// are never read.
iree_status_t loom_link_template_candidate_loader_allocate(
    const loom_link_module_index_t* index,
    const loom_link_plan_materialization_environment_t* environment,
    loom_link_template_candidate_loader_t** out_loader);

// Frees |loader| and all persistent source-summary caches.
void loom_link_template_candidate_loader_free(
    loom_link_template_candidate_loader_t* loader);

// Binds unselected candidates for families demanded by |plan|.
//
// The plan's existing reachability bitmap identifies providers already present
// in |materialization|. Each remaining candidate borrows its source-owned
// applicability facts while using the already-linked family signature in
// materialization->module. This never adds values, symbols, types, or
// operations to the materialized module. The returned summaries belong to
// |arena| and are suitable as the external catalog overlay for one selection
// query.
iree_status_t loom_link_template_candidate_loader_load(
    loom_link_template_candidate_loader_t* loader, const loom_link_plan_t* plan,
    loom_link_plan_materialization_t* materialization,
    iree_arena_allocator_t* arena,
    loom_template_provider_slice_t* out_candidates);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_TEMPLATE_CANDIDATE_LOADER_H_
