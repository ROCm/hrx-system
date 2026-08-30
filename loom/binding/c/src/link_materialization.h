// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_LINK_MATERIALIZATION_H_
#define LOOMC_LINK_MATERIALIZATION_H_

#include "loom/link/materialization_environment.h"
#include "loomc/config.h"
#include "loomc/context.h"
#include "loomc/link_index.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/target.h"
#include "loomc/workspace.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Shared invocation state for selective provider materialization.
//
// The state borrows every referenced public handle and descriptor. It must
// remain live while an environment returned by
// loomc_link_materialization_state_environment is in use.
typedef struct loomc_link_materialization_state_t {
  // Public composition objects borrowed by the invocation.
  struct {
    // Context shared by the index and all materialized modules.
    loomc_context_t* context;

    // Workspace block pool backing transient and returned modules.
    loomc_workspace_t* workspace;

    // Provider-to-source map used for diagnostic identity.
    struct {
      // Frozen index supplying the base provider source domain, or NULL.
      const loomc_link_index_t* link_index;

      // One appended provider not owned by |link_index|.
      struct {
        // Borrowed source associated with the appended provider, or NULL.
        const loomc_source_t* source;

        // Internal index ordinal associated with |source|.
        iree_host_size_t provider_ordinal;
      } appended;
    } sources;
  } composition;

  // Per-invocation specialization inputs.
  struct {
    // Configuration bindings applied to each materialized source module.
    const loomc_config_options_t* config;

    // Optional target specialization applied after configuration.
    const loomc_target_specialization_options_t* target;
  } specialization;

  // Synchronous diagnostic destination and current provider source.
  struct {
    // Result receiving materialization diagnostics.
    loomc_result_t* result;

    // Source associated with the provider currently being decoded.
    const loomc_source_t* source;
  } diagnostics;

  // Host allocator used for transient specialization objects.
  loomc_allocator_t allocator;
} loomc_link_materialization_state_t;

// Initializes borrowed state shared by one selective materialization.
LOOMC_API_PRIVATE void loomc_link_materialization_state_initialize(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_link_index_t* link_index, const loomc_config_options_t* config,
    const loomc_target_specialization_options_t* target_specialization,
    loomc_result_t* result, loomc_allocator_t allocator,
    loomc_link_materialization_state_t* out_state);

// Initializes borrowed materialization state for an immutable library index
// overlaid by one invocation-local source provider.
LOOMC_API_PRIVATE void loomc_link_materialization_state_initialize_overlay(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_link_index_t* library_index,
    iree_host_size_t appended_provider_ordinal,
    const loomc_source_t* appended_source, const loomc_config_options_t* config,
    const loomc_target_specialization_options_t* target_specialization,
    loomc_result_t* result, loomc_allocator_t allocator,
    loomc_link_materialization_state_t* out_state);

// Returns an internal materialization environment borrowing |state|.
LOOMC_API_PRIVATE loom_link_plan_materialization_environment_t
loomc_link_materialization_state_environment(
    loomc_link_materialization_state_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_LINK_MATERIALIZATION_H_
