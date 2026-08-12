// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target compiler pass capability.
//
// The target environment supplies provider-owned compiler semantics without
// selecting a target for the module. Concrete compiler function versions carry
// invocation-refined facts without changing authored target IR.

#ifndef LOOM_TARGET_PASS_ENVIRONMENT_H_
#define LOOM_TARGET_PASS_ENVIRONMENT_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/ir/module.h"
#include "loom/pass/environment.h"
#include "loom/pass/types.h"
#include "loom/target/function_version.h"

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

  // Invocation-local concrete function versions, or NULL.
  const loom_function_version_list_t* function_versions;

  // Mutable owner of |function_versions|, or NULL for read-only capabilities.
  loom_function_version_owner_t* function_version_owner;

  // True when pass-program executions supply a mutable function-version
  // owner. The owner may be NULL while compiling a pass program before an
  // invocation exists.
  bool supports_mutable_function_versions;
} loom_target_pass_capability_t;

// Creates a borrowed target pass capability.
loom_target_pass_capability_t loom_target_pass_capability_make(
    const loom_target_environment_t* target_environment,
    const loom_function_version_list_t* function_versions);

// Creates a borrowed target pass capability whose function versions may be
// extended by module passes. |function_version_owner| may be NULL.
loom_target_pass_capability_t loom_target_pass_capability_make_mutable(
    const loom_target_environment_t* target_environment,
    loom_function_version_owner_t* function_version_owner);

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

// Returns the invocation-local concrete function versions, or NULL.
const loom_function_version_list_t*
loom_target_pass_capability_function_versions(
    const loom_target_pass_capability_t* capability);

// Returns the mutable invocation-local function-version owner, or NULL when
// the capability exposes only a read-only list.
loom_function_version_owner_t*
loom_target_pass_capability_function_version_owner(
    const loom_target_pass_capability_t* capability);

// Resolves immutable function target facts for the active function pass.
//
// A concrete target-refined function version supplies its facts directly from
// the pass frame. An unrefined function projects its authored target contract
// into |pass->instance_arena| so the facts survive resets of the current run's
// scratch arena. Returns OK with |out_resolved| false when |function| has no
// target contract.
iree_status_t loom_target_pass_resolve_function_facts(
    const loom_pass_t* pass, const loom_module_t* module,
    loom_func_like_t function, bool* out_resolved,
    const loom_target_facts_t** out_facts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_PASS_ENVIRONMENT_H_
