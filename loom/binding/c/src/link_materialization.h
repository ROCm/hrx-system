// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_LINK_MATERIALIZATION_H_
#define LOOMC_LINK_MATERIALIZATION_H_

#include "loom/config/text_binding.h"
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

typedef struct loomc_link_target_specializations_t
    loomc_link_target_specializations_t;

// Homogeneous deployment specialization phase for one materialization.
typedef enum loomc_link_deployment_specialization_phase_e {
  // Final linked products reject every incompatible selected entry.
  LOOMC_LINK_DEPLOYMENT_SPECIALIZATION_PHASE_FINAL = 0,

  // Command projection preserves incompatible concrete kernel alternatives
  // until configuration and command normalization establish live entries.
  LOOMC_LINK_DEPLOYMENT_SPECIALIZATION_PHASE_COMMAND_PROJECTION = 1,
} loomc_link_deployment_specialization_phase_t;

// Shared invocation state for selective provider materialization.
//
// The state borrows every referenced public handle and target descriptor;
// normalized configuration is owned until state deinitialization. The state
// must remain live while an environment returned by
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
    // Owned bindings shared by every materialized source module.
    loom_config_text_binding_set_t config_bindings;

    // Resolution policy applied separately to each materialized module.
    loomc_config_policy_flags_t config_policy_flags;

    // Optional target specialization applied after configuration.
    const loomc_target_specialization_options_t* target;

    // Resolved input identities retained after the first materialization.
    const loomc_link_target_specializations_t* indexed;

    // Optional homogeneous deployment profile applied after configuration to
    // selected kernel entries and declaration-targeted functions. Mutually
    // exclusive with nonempty explicit target assignments.
    const loomc_target_profile_t* deployment_profile;

    // Phase governing concrete kernel compatibility for deployment_profile.
    loomc_link_deployment_specialization_phase_t deployment_phase;
  } specialization;

  // Synchronous diagnostic destination and current provider source.
  struct {
    // Result receiving materialization diagnostics.
    loomc_result_t* result;

    // Source associated with the provider currently being decoded.
    const loomc_source_t* source;
  } diagnostics;

  // Invocation-owned storage for target assignments shared by materializations.
  iree_arena_allocator_t* arena;

  // Host allocator used for transient specialization objects.
  loomc_allocator_t allocator;
} loomc_link_materialization_state_t;

// Validates the mutually exclusive homogeneous and explicit target policies at
// the public invocation boundary, including profile/environment compatibility.
LOOMC_API_PRIVATE loomc_status_t loomc_link_materialization_validate_targets(
    const loomc_target_profile_t* deployment_profile,
    const loomc_target_specialization_options_t* target_specialization,
    const loomc_target_environment_t* target_environment);

// Initializes invocation state and normalizes configuration once. Configuration
// syntax errors are recorded in |result|; allocation failures return a status.
LOOMC_API_PRIVATE loomc_status_t loomc_link_materialization_state_initialize(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_link_index_t* link_index, const loomc_config_options_t* config,
    const loomc_target_specialization_options_t* target_specialization,
    loomc_result_t* result, iree_arena_allocator_t* arena,
    loomc_allocator_t allocator, loomc_link_materialization_state_t* out_state);

// Initializes borrowed materialization state for an immutable library index
// overlaid by one invocation-local source provider.
LOOMC_API_PRIVATE loomc_status_t
loomc_link_materialization_state_initialize_overlay(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_link_index_t* library_index,
    iree_host_size_t appended_provider_ordinal,
    const loomc_source_t* appended_source, const loomc_config_options_t* config,
    const loomc_target_specialization_options_t* target_specialization,
    loomc_result_t* result, iree_arena_allocator_t* arena,
    loomc_allocator_t allocator, loomc_link_materialization_state_t* out_state);

// Releases invocation-owned normalized configuration, including after failed
// initialization. Public handles, target descriptors, and the arena are
// borrowed.
LOOMC_API_PRIVATE void loomc_link_materialization_state_deinitialize(
    loomc_link_materialization_state_t* state);

// Returns an internal materialization environment borrowing |state|.
LOOMC_API_PRIVATE loom_link_plan_materialization_environment_t
loomc_link_materialization_state_environment(
    loomc_link_materialization_state_t* state);

// Validates the normalized live command entries retained after a command
// projection materialization. This matches
// loom_cmd_program_live_kernel_entry_validator_t without introducing a
// command-target dependency into the shared binding library.
LOOMC_API_PRIVATE iree_status_t
loomc_link_materialization_validate_command_kernel_entries(
    void* user_data, loom_module_t* module,
    const loom_symbol_ref_t* kernel_refs, iree_host_size_t kernel_count,
    bool* out_valid);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_LINK_MATERIALIZATION_H_
