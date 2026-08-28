// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Selective command-program preparation from an indexed source universe.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_INDEX_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_INDEX_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/link/materialization_environment.h"
#include "loom/link/module_index.h"
#include "loom/pass/registry.h"
#include "loom/target/arch/cmd/lower/program_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Optional indexed command-plan preparation behavior.
typedef struct loom_cmd_program_plan_index_options_t {
  // Sink enabling source-backed kernel request publication. A NULL callback
  // leaves command planning body-blind and publishes only binding requirements.
  loom_cmd_program_kernel_request_sink_t kernel_request_sink;

  // Bounded kernel semantic class collection policy.
  loom_kernel_class_collection_options_t kernel_class_collection;
} loom_cmd_program_plan_index_options_t;

// Initializes command-plan index options to body-blind preparation defaults.
static inline void loom_cmd_program_plan_index_options_initialize(
    loom_cmd_program_plan_index_options_t* options) {
  *options = (loom_cmd_program_plan_index_options_t){
      /*.kernel_request_sink=*/{},
      /*.kernel_class_collection=*/
      loom_kernel_class_collection_options_default(),
  };
}

// Selectively materializes and prepares command-program roots from |index|.
//
// |program_symbol_ordinals| contains identity-resolved index-wide symbol
// ordinals in caller order. Each symbol must expose a command implementation
// facet. Planning retains only command implementations, logical kernel
// contracts and configuration functions, and executable-entry declarations;
// kernel implementation bodies remain unopened.
//
// |scratch_arena| owns the temporary plan, source-to-target projections, and
// root tables used during this call. The returned plan owns every artifact
// needed after the call and remains valid after the index and scratch arena are
// released.
//
// |materialization_environment| is shared by command/configuration projection
// and optional kernel request production. Bytecode providers therefore use the
// same Low descriptor codec, diagnostics, and caller preparation policy at
// both product boundaries.
//
// Unsupported portable mappings and infrastructure failures return a non-OK
// status. Source contract violations emit diagnostics, set |out_valid| to
// false, and return OK.
iree_status_t loom_cmd_program_plan_prepare_index(
    const loom_link_module_index_t* index,
    const iree_host_size_t* program_symbol_ordinals,
    iree_host_size_t program_count,
    const loom_cmd_program_plan_index_options_t* options,
    const loom_pass_registry_t* pass_registry,
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_link_plan_materialization_environment_t*
        materialization_environment,
    iree_arena_allocator_t* scratch_arena, bool* out_valid,
    loom_cmd_program_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_INDEX_H_
