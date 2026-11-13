// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Body-blind launch contract and kernel configuration facet projection.

#ifndef LOOM_LINK_KERNEL_CONFIG_MATERIALIZER_H_
#define LOOM_LINK_KERNEL_CONFIG_MATERIALIZER_H_

#include "loom/link/materialization_environment.h"
#include "loom/link/plan_projection.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compact ordinary IR reconstructed for one source module containing partial
// kernel entry, kernel configuration or pipeline contract selections.
typedef struct loom_link_kernel_config_module_projection_t {
  // Standalone compact module owned by the caller.
  loom_module_t* module;
  // Arena-owned original-to-projected source IDs for materialized input.
  // Bytecode-only inputs have no external source snapshots and leave this NULL.
  const loom_source_id_t* target_sources;
  // Configuration functions aligned with |selection.symbols|. Complete source
  // symbols and entry-only projections contain null refs; selected kernel
  // configurations name their private pure or deterministic read-only
  // workload-to-count function.
  // Storage belongs to the caller's arena.
  struct {
    // Arena-owned compact-module symbol refs.
    loom_symbol_ref_t* values;
    // Number of entries aligned with the source selection.
    iree_host_size_t count;
  } configuration_functions;
} loom_link_kernel_config_module_projection_t;

// Projects every partial launch contract selected from one source module.
//
// Complete dependency symbols are materialized once in canonical source order.
// A configuration selection contributes an ordinary private kernel.decl and
// inline func.def with the source configuration's proven effect contract. An
// entry-only selection contributes kernel.entry.decl without opening its
// configuration or implementation regions. A pipeline contract contributes
// pipeline.decl with its full staged and run signature without opening the
// implementation. Source symbols use the exact compact ordinals assigned by
// |selection|, allowing the
// generic plan materializer to pass this module directly to the incremental
// linker without name lookup or a second reachability pass.
//
// Diagnosed input rejection leaves out_projection->module NULL independently
// of installed callbacks. Statuses carry infrastructure failures only.
iree_status_t loom_link_plan_project_kernel_config_module(
    const loom_link_plan_t* plan,
    const loom_link_plan_module_selection_t* selection,
    const loom_link_plan_materialization_environment_t* environment,
    iree_string_view_t module_name, iree_arena_allocator_t* arena,
    loom_link_kernel_config_module_projection_t* out_projection);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_KERNEL_CONFIG_MATERIALIZER_H_
