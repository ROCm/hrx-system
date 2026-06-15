// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Explicit LLVM target profile preset registries.
//
// Target providers own their preset rows. Developer tools and embedders
// assemble a registry from the providers they intentionally link, keeping the
// generic LLVMIR infrastructure free of a process-wide target catalog.

#ifndef LOOM_TARGET_LLVMIR_TARGET_PRESETS_H_
#define LOOM_TARGET_LLVMIR_TARGET_PRESETS_H_

#include "loom/target/emit/llvmir/target_env.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_llvmir_target_profile_projection_request_t {
  // Generic Loom target bundle resolved for the function being emitted.
  const loom_target_bundle_t* bundle;
  // LLVM target triple requested by llvmir.target, or empty when unspecified.
  iree_string_view_t target_triple;
} loom_llvmir_target_profile_projection_request_t;

typedef bool(
    IREE_API_PTR* loom_llvmir_target_profile_provider_project_bundle_fn_t)(
    const loom_llvmir_target_profile_projection_request_t* request,
    const loom_llvmir_target_profile_t** out_profile);

typedef struct loom_llvmir_target_profile_provider_t {
  // Stable provider name used for diagnostics.
  iree_string_view_t name;
  // Static profile rows owned by the provider.
  const loom_llvmir_target_profile_t* const* profiles;
  // Number of profile pointers in |profiles|.
  iree_host_size_t profile_count;
  // Target name reported by `llc --version`, or empty when not applicable.
  iree_string_view_t llc_target_name;
  // Projects a Loom target bundle to this provider's LLVM profile row.
  loom_llvmir_target_profile_provider_project_bundle_fn_t project_bundle;
} loom_llvmir_target_profile_provider_t;

typedef struct loom_llvmir_target_profile_registry_t {
  // Default profile used when a caller supplies an empty profile name.
  const loom_llvmir_target_profile_t* default_profile;
  // Provider tables explicitly linked into this registry.
  const loom_llvmir_target_profile_provider_t* const* providers;
  // Number of provider pointers in |providers|.
  iree_host_size_t provider_count;
} loom_llvmir_target_profile_registry_t;

// Looks up a target profile by name in |registry|. Empty names resolve to the
// registry default profile. Returns false when no linked profile matches.
bool loom_llvmir_target_profile_registry_lookup(
    const loom_llvmir_target_profile_registry_t* registry,
    iree_string_view_t profile_name,
    const loom_llvmir_target_profile_t** out_profile);

// Projects a generic Loom target bundle to a linked LLVM target profile.
// Returns false when no provider in |registry| supports the requested bundle
// and LLVM target triple combination.
bool loom_llvmir_target_profile_registry_project_bundle(
    const loom_llvmir_target_profile_registry_t* registry,
    const loom_llvmir_target_profile_projection_request_t* request,
    const loom_llvmir_target_profile_t** out_profile);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_TARGET_PRESETS_H_
