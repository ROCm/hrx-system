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
} loom_link_plan_module_selection_t;

// Selected-only partition of one authoritative metadata link plan.
typedef struct loom_link_plan_module_projection_t {
  // Source modules in increasing index ordinal order.
  struct {
    // Arena-owned module selection array.
    loom_link_plan_module_selection_t* values;
    // Number of source modules containing selected symbols.
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
// Construction performs no name lookup or reachability analysis. Storage and
// work scale only with selected symbols and modules. Arrays borrow the plan's
// index and remain valid until |arena| is reset or deinitialized.
iree_status_t loom_link_plan_project_modules(
    const loom_link_plan_t* plan, iree_arena_allocator_t* arena,
    loom_link_plan_module_projection_t* out_projection);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_PLAN_PROJECTION_H_
