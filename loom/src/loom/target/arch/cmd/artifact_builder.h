// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Indexed construction of portable command-program artifact sets.

#ifndef LOOM_TARGET_ARCH_CMD_ARTIFACT_BUILDER_H_
#define LOOM_TARGET_ARCH_CMD_ARTIFACT_BUILDER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/link/materialization_environment.h"
#include "loom/link/module_index.h"
#include "loom/pass/registry.h"
#include "loom/target/arch/cmd/artifact_set.h"
#include "loom/target/arch/cmd/lower/program_plan_index.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compiler services used while constructing one command artifact set.
typedef struct loom_cmd_program_artifact_builder_options_t {
  // Optional indexed-planning behavior such as kernel request publication.
  const loom_cmd_program_plan_index_options_t* plan_options;

  // Pass registry used to specialize linked command and configuration IR.
  const loom_pass_registry_t* pass_registry;

  // Destination for authored command contract diagnostics.
  iree_diagnostic_emitter_t diagnostic_emitter;

  // Provider materialization and specialization environment.
  const loom_link_plan_materialization_environment_t*
      materialization_environment;
} loom_cmd_program_artifact_builder_options_t;

// Selectively prepares indexed command roots and serializes their artifacts.
//
// This is the single index-to-product compiler boundary shared by LoomC and
// command-line tooling. Source contract failures emit diagnostics, leave
// |out_valid| false, and return OK. Infrastructure and callback failures return
// a non-OK status. A valid returned artifact set owns all of its storage and
// must be released with loom_cmd_program_artifact_set_deinitialize.
iree_status_t loom_cmd_program_artifact_set_build_from_index(
    const loom_link_module_index_t* index,
    const iree_host_size_t* root_symbol_ordinals,
    iree_host_size_t root_symbol_count,
    const loom_cmd_program_artifact_builder_options_t* options,
    iree_arena_allocator_t* scratch_arena, bool* out_valid,
    loom_cmd_program_artifact_set_t* out_artifact_set,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_ARTIFACT_BUILDER_H_
