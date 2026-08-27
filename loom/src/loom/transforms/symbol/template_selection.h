// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_SYMBOL_TEMPLATE_SELECTION_H_
#define LOOM_TRANSFORMS_SYMBOL_TEMPLATE_SELECTION_H_

#include "iree/base/internal/arena.h"
#include "loom/analysis/template_provider_catalog.h"
#include "loom/pass/types.h"
#include "loom/target/function_version.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selection behavior at an incremental or closed specialization boundary.
typedef enum loom_template_selection_mode_e {
  // Preserves applications whose best provider is not yet provable.
  LOOM_TEMPLATE_SELECTION_MODE_EARLY = 0,
  // Requires every reachable application to select one exact provider.
  LOOM_TEMPLATE_SELECTION_MODE_FINAL = 1,
} loom_template_selection_mode_t;

// Options for read-only provider selection over one materialized module.
typedef struct loom_template_selection_query_options_t {
  // Early or final selection behavior.
  loom_template_selection_mode_t mode;
  // Provider catalog in |module|'s family-symbol and value domains.
  const loom_template_provider_catalog_t* catalog;
  // Optional compiler-owned function versions for target-refined functions.
  const loom_target_function_version_snapshot_t* function_versions;
  // Exclusive upper bound for external provider origin ordinals.
  iree_host_size_t origin_count;
} loom_template_selection_query_options_t;

// Read-only provider selection result borrowing caller-owned arena storage.
typedef struct loom_template_selection_query_result_t {
  // Exact external provider origins selected by reachable applications.
  struct {
    // Dense origins in first-selection order.
    const iree_host_size_t* values;
    // Number of unique selected origins.
    iree_host_size_t count;
  } selected_origins;
  // Number of reachable applications left unresolved.
  iree_host_size_t unresolved_site_count;
} loom_template_selection_query_result_t;

// Selects providers without rewriting |module|.
//
// Local providers in |options->catalog| participate in liveness so nested
// applications in already-materialized selected bodies are visited in the
// same fixed point. External providers carry an origin ordinal and are
// returned for selective body materialization by the caller.
iree_status_t loom_template_selection_query(
    loom_module_t* module,
    const loom_template_selection_query_options_t* options,
    iree_arena_block_pool_t* block_pool, iree_arena_allocator_t* arena,
    loom_template_selection_query_result_t* out_result);

// Returns the template selection pass metadata.
const loom_pass_info_t* loom_template_selection_pass_info(void);

// Creates a template selection pass invocation.
iree_status_t loom_template_selection_create(loom_pass_t* pass,
                                             iree_string_view_t options);

// Runs module-level template provider selection.
iree_status_t loom_template_selection_run(loom_pass_t* pass,
                                          loom_module_t* module);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_TEMPLATE_SELECTION_H_
