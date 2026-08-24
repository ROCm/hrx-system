// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Materializes one metadata link plan into standalone in-memory Loom IR.

#ifndef LOOM_LINK_PLAN_MATERIALIZER_H_
#define LOOM_LINK_PLAN_MATERIALIZER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/diagnostic.h"
#include "loom/format/low_repr.h"
#include "loom/link/planner.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the diagnostic sink for one indexed bytecode provider.
typedef loom_diagnostic_sink_t (*loom_link_plan_diagnostic_sink_fn_t)(
    void* user_data, const loom_link_module_index_provider_t* provider);

// Applies caller-owned specialization to a newly linked module.
typedef iree_status_t (*loom_link_plan_prepare_module_fn_t)(
    void* user_data, loom_module_t* module);

// Environment shared by every source materialized for one plan.
typedef struct loom_link_plan_materialization_environment_t {
  // Finalized context shared by the index and output module.
  loom_context_t* context;
  // Block pool backing transient and output module arenas.
  iree_arena_block_pool_t* block_pool;
  // Stable-key codec used while decoding low bytecode representations.
  loom_low_repr_environment_t low_repr_environment;
  // Optional per-provider diagnostic sink resolver.
  loom_link_plan_diagnostic_sink_fn_t diagnostic_sink;
  // Optional caller-owned specialization applied to the linked output.
  loom_link_plan_prepare_module_fn_t prepare_module;
  // User data passed to diagnostic_sink and prepare_module.
  void* user_data;
  // Host allocator for transient readers, link state, and the output module.
  iree_allocator_t allocator;
} loom_link_plan_materialization_environment_t;

// Result of materializing one exact plan.
typedef struct loom_link_plan_materialization_t {
  // Standalone linked module owned by the caller.
  loom_module_t* module;
  // Dense target refs indexed by source index symbol ordinal. Unselected
  // symbols contain null refs. Storage belongs to the caller's arena.
  struct {
    // Arena-owned source-to-target projection.
    loom_symbol_ref_t* values;
    // Number of index-wide source symbol slots.
    iree_host_size_t count;
  } target_symbols;
  // Dense target refs indexed by template-family ordinal. Each demanded family
  // resolves through whichever equivalent declaration was selected, rather
  // than depending on one canonical source declaration being live.
  struct {
    // Arena-owned family-to-target projection.
    loom_symbol_ref_t* values;
    // Number of index-wide template-family slots.
    iree_host_size_t count;
  } target_template_families;
} loom_link_plan_materialization_t;

// Materializes |plan| through the incremental linker.
//
// Bytecode bodies are decoded only for symbols selected by the plan. The
// plan's root/dependency reasons authoritatively determine the output surface;
// no second root list is resolved during materialization. The
// returned target-symbol projection preserves exact indexed source identity
// across private names and duplicate global declarations.
iree_status_t loom_link_plan_materialize(
    const loom_link_plan_t* plan,
    const loom_link_plan_materialization_environment_t* environment,
    iree_string_view_t module_name, iree_arena_allocator_t* arena,
    loom_link_plan_materialization_t* out_materialization);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_PLAN_MATERIALIZER_H_
