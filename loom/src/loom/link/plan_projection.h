// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exact module-local projection of a metadata link plan.

#ifndef LOOM_LINK_PLAN_PROJECTION_H_
#define LOOM_LINK_PLAN_PROJECTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/link/planner.h"

#ifdef __cplusplus
extern "C" {
#endif

// One selected symbol projected into its source module's ordinal domain.
typedef struct loom_link_plan_module_symbol_t {
  // Authoritative live-symbol selection in the source plan.
  const loom_link_plan_symbol_t* plan_symbol;
  // Indexed source symbol consumed by materialization.
  const loom_link_module_index_symbol_t* source_symbol;
  // Symbol ordinal used by the module presented to the incremental linker.
  // Already-materialized complete modules retain sparse source ordinals;
  // compact bytecode and facet-projected modules use dense projected ordinals.
  uint32_t materialized_symbol_ordinal;
} loom_link_plan_module_symbol_t;

// Exact selected-symbol slice owned by one source module.
typedef struct loom_link_plan_module_selection_t {
  // Indexed source module owning every symbol in this slice.
  const loom_link_module_index_module_t* source_module;
  // Selected symbols in increasing module-local source ordinal order.
  struct {
    // Arena-owned symbol selection array.
    loom_link_plan_module_symbol_t* values;
    // Number of selected source symbols.
    iree_host_size_t count;
  } symbols;
  // Dense symbol count produced when at least one selected source symbol must
  // be reconstructed from a strict subset of its semantic facets. Includes one
  // helper for every partial source symbol; zero selects ordinary
  // materialization.
  iree_host_size_t projected_symbol_count;
} loom_link_plan_module_selection_t;

static inline bool loom_link_plan_module_requires_symbol_projection(
    const loom_link_plan_module_selection_t* selection) {
  return selection->projected_symbol_count != 0;
}

// Module-local partition of one authoritative metadata link plan.
typedef struct loom_link_plan_module_projection_t {
  // Largest symbol domain presented by any projected source module.
  iree_host_size_t maximum_materialized_symbol_count;
  // Number of synthetic helper symbols added by semantic facet projection.
  iree_host_size_t synthetic_symbol_count;
  // Source modules in increasing index ordinal order. Merge projections include
  // symbol-empty INPUT modules so their module-level metadata is retained.
  struct {
    // Arena-owned module selection array.
    loom_link_plan_module_selection_t* values;
    // Number of projected source modules.
    iree_host_size_t count;
  } modules;
  // Flat storage sliced by modules[].symbols.
  struct {
    // Arena-owned symbol selection array.
    loom_link_plan_module_symbol_t* values;
    // Number of selected symbols in the source plan.
    iree_host_size_t count;
  } symbols;
} loom_link_plan_module_projection_t;

// Partitions |plan| into deterministic module-local source-symbol slices.
//
// Construction performs no name lookup or reachability analysis. Merge
// projections own every INPUT module, including modules without symbols; link
// projections contain only modules with selected symbols. Work and storage
// scale with selected symbols. Arrays borrow the plan's index and
// remain valid until |arena| is reset or deinitialized.
iree_status_t loom_link_plan_project_modules(
    const loom_link_plan_t* plan, iree_arena_allocator_t* arena,
    loom_link_plan_module_projection_t* out_projection);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_PLAN_PROJECTION_H_
