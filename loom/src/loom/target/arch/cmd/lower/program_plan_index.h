// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Selective command-program plan construction from an indexed source universe.

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

// Optional indexed command-plan construction behavior.
typedef struct loom_cmd_program_plan_index_options_t {
  // Source-boundary verification after selective linking and specialization,
  // before command normalization or request publication. Internal callers of
  // already-verified sources may leave this empty. The callback borrows the
  // linked module and reports source errors through its own diagnostics and
  // |out_valid|; a non-OK status identifies an infrastructure failure.
  struct {
    // Invoked exactly once on the final linked command module.
    iree_status_t (*fn)(void* user_data, const loom_module_t* module,
                        bool* out_valid);
    // Caller-owned verification and diagnostic context.
    void* user_data;
  } verification;

  // Optional validation of the exact kernel entries retained by command
  // normalization. This runs before any kernel request is published.
  loom_cmd_program_live_kernel_entry_validator_t live_kernel_entry_validator;

  // Sink enabling indexed and generated kernel request publication. A NULL
  // callback permits body-blind plans whose launch counts are already directly
  // or statically representable; storage-backed launch arithmetic requires a
  // sink for its generated device producer.
  loom_cmd_program_kernel_request_sink_t kernel_request_sink;

  // Optional source preparation owner. Empty selects invocation-local source
  // preparation using materialization_environment. This provider is called only
  // with a request sink and a surviving indexed kernel's complete site set.
  loom_kernel_request_provider_t kernel_request_provider;

  // Bounded kernel semantic class collection policy.
  loom_kernel_class_collection_options_t kernel_class_collection;
} loom_cmd_program_plan_index_options_t;

// Initializes command-plan index options to body-blind planning defaults.
static inline void loom_cmd_program_plan_index_options_initialize(
    loom_cmd_program_plan_index_options_t* options) {
  *options = (loom_cmd_program_plan_index_options_t){
      /*.verification=*/{},
      /*.live_kernel_entry_validator=*/{},
      /*.kernel_request_sink=*/{},
      /*.kernel_request_provider=*/{},
      /*.kernel_class_collection=*/
      loom_kernel_class_collection_options_default(),
  };
}

// Builds a command-program plan from selectively materialized |index| roots.
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
// |pass_registry| and |cleanup_pattern_provider_set| select the nested
// target-neutral compiler used to normalize linked command and configuration
// IR before lowering.
//
// |materialization_environment| is shared by command/configuration projection
// and optional kernel request production. Bytecode providers therefore use the
// same Low descriptor codec, diagnostics, and caller planning policy at
// both product boundaries.
//
// Unsupported portable mappings and infrastructure failures return a non-OK
// status. Source contract violations emit diagnostics, set |out_valid| to
// false, and return OK.
iree_status_t loom_cmd_program_plan_build_from_index(
    const loom_link_module_index_t* index,
    const iree_host_size_t* program_symbol_ordinals,
    iree_host_size_t program_count,
    const loom_cmd_program_plan_index_options_t* options,
    const loom_pass_registry_t* pass_registry,
    const loom_cleanup_pattern_provider_set_t* cleanup_pattern_provider_set,
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_link_plan_materialization_environment_t*
        materialization_environment,
    iree_arena_allocator_t* scratch_arena, bool* out_valid,
    loom_cmd_program_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_INDEX_H_
