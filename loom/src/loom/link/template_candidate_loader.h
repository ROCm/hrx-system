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

// Dense membership for provider symbols already selected by the link.
typedef struct loom_link_template_provider_membership_t {
  // Packed membership bits indexed by index-wide symbol ordinal.
  const uint64_t* words;
  // Number of index-wide symbol ordinals represented by words.
  iree_host_size_t symbol_count;
} loom_link_template_provider_membership_t;

// Creates a lazy candidate-header loader over |index|.
//
// The loader borrows |index| and |environment| for its lifetime. Materialized
// source modules are summarized once. Bytecode provider headers and the shared
// table entries they reach are decoded at most once and implementation bodies
// are never read.
iree_status_t loom_link_template_candidate_loader_create(
    const loom_link_module_index_t* index,
    const loom_link_plan_materialization_environment_t* environment,
    loom_link_template_candidate_loader_t** out_loader);

// Releases |loader| and all persistent source-summary caches.
void loom_link_template_candidate_loader_free(
    loom_link_template_candidate_loader_t* loader);

// Projects unselected candidates for families demanded by |plan|.
//
// Source summaries are projected into materialization->module. Missing
// provider-owned symbol references become disposable analysis placeholders;
// they never enter the final linked artifact. The returned summaries belong to
// |arena| and are suitable as the external catalog overlay for one selection
// query.
iree_status_t loom_link_template_candidate_loader_project(
    loom_link_template_candidate_loader_t* loader, const loom_link_plan_t* plan,
    loom_link_plan_materialization_t* materialization,
    loom_link_template_provider_membership_t selected_providers,
    iree_arena_allocator_t* arena,
    loom_template_provider_slice_t* out_candidates);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_TEMPLATE_CANDIDATE_LOADER_H_
