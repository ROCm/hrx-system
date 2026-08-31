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
#include "loom/link/materialization_environment.h"
#include "loom/link/planner.h"

#ifdef __cplusplus
extern "C" {
#endif

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
  // Dense concrete source-definition ordinals indexed by target module symbol
  // ID at the end of linking. A target assembled from a declaration and
  // definition names the definition selected by the plan. Declarations
  // without an indexed provider and synthetic symbols contain
  // LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL. Caller-owned preparation may
  // append symbols while preserving this stable linked prefix. Storage belongs
  // to the caller's arena.
  struct {
    // Arena-owned target-to-source definition projection.
    iree_host_size_t* values;
    // Number of target module symbol slots present before caller preparation.
    iree_host_size_t count;
  } target_source_definitions;
  // Dense target refs indexed by template-family ordinal. Each demanded family
  // resolves through whichever equivalent declaration was selected, rather
  // than depending on one canonical source declaration being live.
  struct {
    // Arena-owned family-to-target projection.
    loom_symbol_ref_t* values;
    // Number of index-wide template-family slots.
    iree_host_size_t count;
  } target_template_families;
  // Dense configuration-function refs indexed by target module symbol ID at
  // the end of linking. Only partially projected logical kernels contain valid
  // refs. Caller-owned preparation may append symbols while preserving this
  // stable linked prefix. This direct target-domain projection lets downstream
  // consumers resolve a linked kernel callee without rebuilding an inverse
  // source-symbol map. Storage belongs to the caller's arena.
  struct {
    // Arena-owned target-kernel-to-configuration-function projection.
    loom_symbol_ref_t* values;
    // Number of target module symbol slots present before caller preparation.
    iree_host_size_t count;
  } target_kernel_configurations;
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
