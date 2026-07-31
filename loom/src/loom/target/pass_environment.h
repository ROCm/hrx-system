// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target compiler pass capability.
//
// The target environment supplies provider-owned compiler semantics without
// selecting a target for the module. Concrete compiler function versions carry
// invocation-refined facts without changing authored target IR. An optional
// specialization context supplies per-function profile payloads.

#ifndef LOOM_TARGET_PASS_ENVIRONMENT_H_
#define LOOM_TARGET_PASS_ENVIRONMENT_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/ir/module.h"
#include "loom/pass/environment.h"
#include "loom/pass/types.h"
#include "loom/target/facts.h"
#include "loom/target/function_version.h"
#include "loom/target/specialization.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_environment_t loom_target_environment_t;

// Capability type for loom_target_pass_capability_t.
extern const loom_pass_environment_capability_type_t
    loom_target_pass_capability_type;

typedef struct loom_target_pass_capability_t {
  // Base capability header. Must remain the first field.
  loom_pass_environment_capability_t base;

  // Target providers linked into the compiler session, or NULL.
  const loom_target_environment_t* target_environment;

  // Invocation-local supplemental per-function profiles, or NULL.
  const loom_target_specialization_context_t* specialization_context;

  // Invocation-local concrete function versions, or NULL.
  const loom_function_version_list_t* function_versions;
} loom_target_pass_capability_t;

// Creates a borrowed target pass capability.
loom_target_pass_capability_t loom_target_pass_capability_make(
    const loom_target_environment_t* target_environment,
    const loom_target_specialization_context_t* specialization_context,
    const loom_function_version_list_t* function_versions);

// Looks up the target capability from |environment|. Returns NULL when absent.
const loom_target_pass_capability_t*
loom_target_pass_capability_from_environment(
    const loom_pass_environment_t* environment);

// Looks up the target capability from |pass->environment|. Returns NULL when
// absent.
const loom_target_pass_capability_t* loom_target_pass_capability_from_pass(
    const loom_pass_t* pass);

// Returns the target providers linked into the compiler session, or NULL.
const loom_target_environment_t* loom_target_pass_capability_target_environment(
    const loom_target_pass_capability_t* capability);

// Returns the invocation-local per-function specialization context, or NULL.
const loom_target_specialization_context_t*
loom_target_pass_capability_specialization_context(
    const loom_target_pass_capability_t* capability);

// Returns the invocation-local concrete function versions, or NULL.
const loom_function_version_list_t*
loom_target_pass_capability_function_versions(
    const loom_target_pass_capability_t* capability);

// Returns the supplemental specialization profile for |function|, or NULL.
const loom_target_profile_t* loom_target_pass_capability_specialization_profile(
    const loom_target_pass_capability_t* capability,
    const loom_module_t* module, loom_func_like_t function);

// Resolves the immutable effective target facts for the active function pass.
//
// A concrete target-refined function version supplies its facts directly from
// the pass frame. An unrefined function projects its authored target contract
// into |pass->arena|. Returns OK with |out_resolved| false when |function| has
// no target contract.
iree_status_t loom_target_pass_resolve_function_facts(
    const loom_pass_t* pass, const loom_module_t* module,
    loom_func_like_t function, bool* out_resolved,
    const loom_target_facts_t** out_facts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_PASS_ENVIRONMENT_H_
