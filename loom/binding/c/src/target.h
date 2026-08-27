// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_STORAGE_H_
#define LOOMC_TARGET_STORAGE_H_

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/pass/registry.h"
#include "loom/target/function_version.h"
#include "loom/target/profile.h"
#include "loom/target/provider.h"
#include "loom/target/specialization.h"
#include "loom/target/types.h"
#include "loomc/target.h"
#include "visibility.h"

typedef struct loom_text_low_asm_environment_t loom_text_low_asm_environment_t;
typedef struct loom_low_repr_environment_t loom_low_repr_environment_t;

#ifdef __cplusplus
extern "C" {
#endif

// Releases a target-family profile owned by a public profile handle.
typedef void (*loomc_target_profile_deinitialize_fn_t)(
    loom_target_profile_t* profile, loomc_allocator_t allocator);

// Prepared target pass capability tables derived from a public target
// environment.
typedef struct loomc_target_pass_environment_t {
  // Target providers owning compiler semantics used by target-aware passes.
  const loom_target_environment_t* target_environment;

  // Target-low descriptor registry used by target-low passes and emitters.
  loom_target_low_descriptor_registry_t low_descriptor_registry;

  // Source-to-target-low lowering policy registry used by source-to-low.
  loom_low_lower_policy_registry_t low_lower_policy_registry;

  // Target math policy registry used by math legalization passes.
  loom_target_math_policy_registry_t math_policy_registry;

  // Target-low source legality providers linked into this environment.
  loom_target_low_legality_provider_list_t low_legality_provider_list;

  // Target source legalizer providers linked into this environment.
  loom_target_legalizer_provider_list_t legalizer_provider_list;
} loomc_target_pass_environment_t;

// Creates a public target environment from an internal provider set.
LOOMC_API_PRIVATE loomc_status_t
loomc_target_environment_create_from_provider_set(
    const loom_target_provider_set_t* provider_set, loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment);

// Returns the internal target environment owned by the public handle.
LOOMC_API_PRIVATE const loom_target_environment_t*
loomc_target_environment_loom_target_environment(
    const loomc_target_environment_t* target_environment);

// Returns prepared pass capability tables owned by the target environment, or
// NULL when target_environment is NULL.
LOOMC_API_PRIVATE const loomc_target_pass_environment_t*
loomc_target_environment_pass_environment(
    const loomc_target_environment_t* target_environment);

// Registers target dialects with a not-yet-finalized Loom context.
LOOMC_API_PRIVATE loomc_status_t loomc_target_environment_register_context(
    const loomc_target_environment_t* target_environment,
    loom_context_t* context);

// Resolves target-related context option extensions.
LOOMC_API_PRIVATE loomc_status_t loomc_context_target_options_resolve(
    const loomc_context_options_t* options,
    loomc_target_environment_t** out_target_environment);

// Resolves the target-specialization compile option extension.
LOOMC_API_PRIVATE loomc_status_t loomc_target_specialization_options_resolve(
    const void* next,
    const loomc_target_specialization_options_t** out_options);

// Validates one target-specialization option descriptor.
LOOMC_API_PRIVATE loomc_status_t loomc_target_specialization_options_validate(
    const loomc_target_specialization_options_t* options);

// Validates every specialization profile against a target environment.
LOOMC_API_PRIVATE loomc_status_t
loomc_target_specialization_options_validate_environment(
    const loomc_target_specialization_options_t* options,
    const loomc_target_environment_t* target_environment);

// Materializes internal function requests and target declaration bindings from
// public options. Returned lists are owned by |arena| until it is reset.
LOOMC_API_PRIVATE loomc_status_t loomc_target_specialization_options_make_lists(
    const loomc_target_specialization_options_t* options,
    iree_arena_allocator_t* arena,
    loom_target_specialization_request_list_t* out_requests,
    loom_target_declaration_binding_list_t* out_bindings);

// Creates a public target profile from a prepared target-family profile. Takes
// ownership of |target_profile| on entry and calls |deinitialize| on failure or
// when the final handle reference is released. A NULL deinitializer denotes
// profile storage with process lifetime.
LOOMC_API_PRIVATE loomc_status_t loomc_target_profile_create(
    loomc_target_environment_t* target_environment,
    loomc_string_view_t identifier, loom_target_profile_t* target_profile,
    loomc_target_profile_deinitialize_fn_t deinitialize,
    loomc_allocator_t allocator, loomc_target_profile_t** out_profile);

// Returns the typed target-family profile owned by a public profile.
LOOMC_API_PRIVATE const loom_target_profile_t*
loomc_target_profile_loom_target_profile(const loomc_target_profile_t* profile);

// Returns the target environment retained by a public profile.
LOOMC_API_PRIVATE loomc_target_environment_t*
loomc_target_profile_target_environment(const loomc_target_profile_t* profile);

// Returns the stable diagnostic identifier owned by a public profile.
LOOMC_API_PRIVATE loomc_string_view_t
loomc_target_profile_identifier(const loomc_target_profile_t* profile);

// Initializes a stable pass registry combining builtin and target-owned pass
// descriptors. The returned registry points into out_storage.
LOOMC_API_PRIVATE loomc_status_t loomc_target_pass_registry_initialize(
    const loomc_target_environment_t* target_environment,
    loom_pass_registry_storage_t* out_storage,
    const loom_pass_registry_t** out_registry);

// Creates a borrowed pass environment view over prepared target pass tables.
LOOMC_API_PRIVATE loom_pass_environment_t
loomc_target_pass_environment_make_loom_pass_environment(
    const loomc_target_pass_environment_t* environment,
    loom_function_version_owner_t* function_version_owner,
    loom_low_pass_environment_storage_t* out_storage);

// Initializes a target-aware text low-asm environment over prepared target
// descriptor tables.
LOOMC_API_PRIVATE void
loomc_target_pass_environment_initialize_text_asm_environment(
    const loomc_target_pass_environment_t* environment,
    loom_text_low_asm_environment_t* out_environment);

// Initializes the stable Low representation codec over prepared target
// descriptor tables. A target-free context produces an empty codec suitable
// only for formats that do not materialize scoped Low representation values.
LOOMC_API_PRIVATE void
loomc_target_pass_environment_initialize_low_repr_environment(
    const loomc_target_pass_environment_t* environment,
    loom_low_repr_environment_t* out_environment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_STORAGE_H_
