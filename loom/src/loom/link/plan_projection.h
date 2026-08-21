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
#include "loom/link/linker.h"
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

// One retained provider import projected into a source module's ordinal
// domain.
typedef struct loom_link_plan_module_provider_import_t {
  // Provider-import ordinal in the indexed source module.
  uint32_t source_import_ordinal;
  // Live unresolved anchors in canonical source order.
  struct {
    // First entry in the owning module selection's flat anchor slice.
    uint32_t first;
    // Number of retained anchors.
    uint32_t count;
  } anchors;
} loom_link_plan_module_provider_import_t;

// Exact selected-symbol and retained-import slices owned by one source module.
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
  // Provider imports containing live anchors not consumed by this plan.
  struct {
    // Arena-owned retained provider-import array.
    loom_link_plan_module_provider_import_t* values;
    // Number of retained provider imports.
    iree_host_size_t count;
  } provider_imports;
  // Flat module-local source symbol ordinals sliced by provider imports.
  struct {
    // Arena-owned retained anchor array.
    uint32_t* values;
    // Number of retained anchors in this module.
    iree_host_size_t count;
  } provider_import_anchors;
} loom_link_plan_module_selection_t;

// Module-local partition of one authoritative metadata link plan.
typedef struct loom_link_plan_module_projection_t {
  // Source modules in increasing index ordinal order. Archive projections
  // include symbol-empty modules so their module-level metadata is retained.
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
  // Flat storage sliced by modules[].provider_imports.
  struct {
    // Arena-owned retained provider-import array.
    loom_link_plan_module_provider_import_t* values;
    // Number of retained provider imports.
    iree_host_size_t count;
  } provider_imports;
  // Flat storage sliced by provider_imports[].anchors.
  struct {
    // Arena-owned module-local source symbol ordinals.
    uint32_t* values;
    // Number of retained provider-import anchors.
    iree_host_size_t count;
  } provider_import_anchors;
} loom_link_plan_module_projection_t;

// Linker-ready provider imports aligned one-to-one with projected modules.
typedef struct loom_link_plan_linker_import_projection_t {
  // Per-module import lists in module-projection order.
  struct {
    // Arena-owned lists aligned with
    // loom_link_plan_module_projection_t.modules.
    loom_linker_source_provider_import_list_t* values;
    // Number of per-module lists.
    iree_host_size_t count;
  } modules;
  // Flat provider-import rows sliced by modules[].
  struct {
    // Arena-owned provider-import rows.
    loom_linker_source_provider_import_t* values;
    // Number of retained provider-import rows.
    iree_host_size_t count;
  } provider_imports;
  // Flat linker-domain symbol ordinals sliced by provider-import rows.
  struct {
    // Arena-owned symbol ordinal array.
    uint32_t* values;
    // Number of retained provider-import anchors.
    iree_host_size_t count;
  } provider_import_anchors;
} loom_link_plan_linker_import_projection_t;

// Partitions |plan| into deterministic module-local source-symbol slices.
//
// Construction performs no name lookup or reachability analysis. Dead anchors
// are pruned, exact declarations resolved by this plan are consumed, and live
// unresolved or concrete availability anchors remain. Archive projections own
// every indexed module, including modules without symbols; selective
// projections contain only modules with selected symbols. Work scales with the
// selected symbols and provider-import metadata of projected source modules;
// storage scales only with retained output. Arrays borrow the plan's index and
// remain valid until |arena| is reset or deinitialized.
iree_status_t loom_link_plan_project_modules(
    const loom_link_plan_t* plan, iree_arena_allocator_t* arena,
    loom_link_plan_module_projection_t* out_projection);

// Projects retained provider imports into each selected module's linker domain.
//
// Already-materialized modules retain their original sparse symbol ordinals.
// Bytecode modules use the dense ordinals assigned by selected materialization.
// All views borrow canonical provider semantics from |index| and remain valid
// until |arena| is reset or deinitialized.
iree_status_t loom_link_plan_project_linker_imports(
    const loom_link_module_index_t* index,
    const loom_link_plan_module_projection_t* module_projection,
    iree_arena_allocator_t* arena,
    loom_link_plan_linker_import_projection_t* out_projection);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_PLAN_PROJECTION_H_
