// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cleanup pass environment capability.

#ifndef LOOM_TRANSFORMS_CLEANUP_PASS_ENVIRONMENT_H_
#define LOOM_TRANSFORMS_CLEANUP_PASS_ENVIRONMENT_H_

#include "loom/pass/environment.h"
#include "loom/pass/types.h"
#include "loom/transforms/cleanup/patterns.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_facts_t loom_target_facts_t;
typedef struct loom_target_math_policy_t loom_target_math_policy_t;

typedef uint32_t loom_cleanup_context_flags_t;
enum loom_cleanup_context_flag_bits_e {
  // Target facts outlive the pass invocation and may remain in its shared
  // value-fact workspace. Otherwise that workspace must expire with the pass.
  LOOM_CLEANUP_CONTEXT_PERSISTENT_TARGET_FACTS = 1u << 0,
};

// Function-scoped semantics resolved by the environment that owns them.
typedef struct loom_cleanup_canonicalizer_context_t {
  // Ownership guarantees supplied by the context resolver.
  loom_cleanup_context_flags_t flags;
  // Immutable target facts used by target-sensitive value inference.
  const loom_target_facts_t* target_facts;
  // Target math policy used for optional arithmetic rewrites.
  const loom_target_math_policy_t* math_policy;
} loom_cleanup_canonicalizer_context_t;

typedef iree_status_t (*loom_cleanup_resolve_canonicalizer_context_fn_t)(
    void* user_data, const loom_pass_t* pass, const loom_module_t* module,
    loom_func_like_t function,
    loom_cleanup_canonicalizer_context_t* out_context);

typedef struct loom_cleanup_canonicalizer_context_resolver_t {
  // Environment-owned resolution callback, or NULL for target-neutral runs.
  loom_cleanup_resolve_canonicalizer_context_fn_t fn;
  // Opaque environment-owned state passed to |fn|.
  void* user_data;
} loom_cleanup_canonicalizer_context_resolver_t;

// Capability type for loom_cleanup_pass_capability_t.
extern const loom_pass_environment_capability_type_t
    loom_cleanup_pass_capability_type;

typedef struct loom_cleanup_pass_capability_t {
  // Base capability header. Must remain the first field.
  loom_pass_environment_capability_t base;
  // Cleanup pattern registries prepared for this compiler invocation.
  const loom_cleanup_pattern_registry_t* pattern_registry;
  // Resolver for function-scoped canonicalizer semantics.
  loom_cleanup_canonicalizer_context_resolver_t context_resolver;
} loom_cleanup_pass_capability_t;

// Creates a borrowed cleanup pass capability.
loom_cleanup_pass_capability_t loom_cleanup_pass_capability_make(
    const loom_cleanup_pattern_registry_t* pattern_registry,
    loom_cleanup_canonicalizer_context_resolver_t context_resolver);

// Looks up the cleanup capability from |environment|. Returns NULL when absent.
const loom_cleanup_pass_capability_t*
loom_cleanup_pass_capability_from_environment(
    const loom_pass_environment_t* environment);

// Looks up the cleanup capability from |pass->environment|. Returns NULL when
// absent.
const loom_cleanup_pass_capability_t* loom_cleanup_pass_capability_from_pass(
    const loom_pass_t* pass);

// Returns the pattern registries selected by |capability|, or NULL.
const loom_cleanup_pattern_registry_t*
loom_cleanup_pass_capability_pattern_registry(
    const loom_cleanup_pass_capability_t* capability);

// Resolves function-scoped canonicalizer semantics through |capability|.
// Missing capabilities and resolvers produce an empty target-neutral context.
iree_status_t loom_cleanup_pass_capability_resolve_canonicalizer_context(
    const loom_cleanup_pass_capability_t* capability, const loom_pass_t* pass,
    const loom_module_t* module, loom_func_like_t function,
    loom_cleanup_canonicalizer_context_t* out_context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_CLEANUP_PASS_ENVIRONMENT_H_
