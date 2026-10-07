// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Code generation pass environment composition.
//
// A code generation invocation presents independently owned target, low, and
// math capabilities through one borrowed pass environment. This layer owns the
// composition without making any individual capability responsible for its
// peers.

#ifndef LOOM_CODEGEN_PASS_ENVIRONMENT_H_
#define LOOM_CODEGEN_PASS_ENVIRONMENT_H_

#include "iree/base/api.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/pass/environment.h"
#include "loom/target/math_policy.h"
#include "loom/target/pass_environment.h"
#include "loom/transforms/cleanup/pass_environment.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_codegen_pass_environment_options_t {
  // Linked target-low descriptor registry.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Linked source-to-target-low lowering policy registry.
  const loom_low_lower_policy_registry_t* lower_policy_registry;
  // Optional target-specific source legality provider list.
  const loom_target_low_legality_provider_list_t* legality_provider_list;
  // Dense source legalizer registry prepared for this invocation.
  const loom_target_legalizer_registry_t* legalizer_registry;
  // Target math policy registry prepared for this invocation.
  const loom_target_math_policy_registry_t* math_policy_registry;
  // Optional compile report receiving target feedback.
  loom_target_compile_report_t* compile_report;
  // Target providers linked into the compiler session.
  const loom_target_environment_t* target_environment;
  // Cleanup pattern registries prepared for this compiler invocation.
  const loom_cleanup_pattern_registry_t* cleanup_pattern_registry;
  // Optional launch-config product capability populated by source-to-Low.
  const loom_pass_environment_capability_t* launch_config_capability;
} loom_codegen_pass_environment_options_t;

typedef struct loom_codegen_pass_environment_storage_t {
  // Target compiler capability entry stored for the borrowed environment view.
  loom_target_pass_capability_t target_capability;
  // Low capability entry stored for the borrowed environment view.
  loom_low_pass_capability_t low_capability;
  // Target math capability entry stored for the borrowed environment view.
  loom_target_math_pass_capability_t math_capability;
  // Cleanup capability entry stored for the borrowed environment view.
  loom_cleanup_pass_capability_t cleanup_capability;
  // Pointer table borrowed by |environment|.
  const loom_pass_environment_capability_t* capabilities[5];
  // Pass environment view over |capabilities|.
  loom_pass_environment_t environment;
} loom_codegen_pass_environment_storage_t;

// Initializes stack storage for a code generation pass environment. The
// returned environment must not outlive |out_storage| or |function_versions|.
loom_pass_environment_t loom_codegen_pass_environment_storage_initialize(
    const loom_codegen_pass_environment_options_t* options,
    const loom_function_version_list_t* function_versions,
    loom_codegen_pass_environment_storage_t* out_storage);

// Initializes stack storage for a code generation pass environment whose
// function-version owner may be extended by module passes. The returned
// environment must not outlive |out_storage| or |function_version_owner|.
loom_pass_environment_t
loom_codegen_pass_environment_storage_initialize_mutable(
    const loom_codegen_pass_environment_options_t* options,
    loom_function_version_owner_t* function_version_owner,
    loom_codegen_pass_environment_storage_t* out_storage);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_PASS_ENVIRONMENT_H_
