// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-low pass environment capability.
//
// This capability exposes the target-low tables linked into the current
// compiler session without selecting a target for the whole pipeline. Passes
// use the registry and policy tables here, then resolve each function's
// concrete target record from IR facts.

#ifndef LOOM_CODEGEN_LOW_PIPELINE_PASS_ENVIRONMENT_H_
#define LOOM_CODEGEN_LOW_PIPELINE_PASS_ENVIRONMENT_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/pass/environment.h"
#include "loom/pass/types.h"
#include "loom/target/math_policy.h"
#include "loom/target/selection.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Capability type for loom_low_pass_capability_t.
extern const loom_pass_environment_capability_type_t
    loom_low_pass_capability_type;

typedef struct loom_low_lower_policy_registry_t
    loom_low_lower_policy_registry_t;
typedef struct loom_target_low_legality_provider_list_t
    loom_target_low_legality_provider_list_t;
typedef struct loom_target_legalizer_provider_list_t
    loom_target_legalizer_provider_list_t;
typedef struct loom_target_compile_report_t loom_target_compile_report_t;

typedef struct loom_low_pass_capability_t {
  // Base capability header. Must remain the first field.
  loom_pass_environment_capability_t base;
  // Linked target-low descriptor registry.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Linked source-to-target-low lowering policy registry.
  const loom_low_lower_policy_registry_t* lower_policy_registry;
  // Optional target-specific source legality providers linked into this
  // compiler.
  const loom_target_low_legality_provider_list_t* legality_provider_list;
  // Optional target-specific source legalizer providers linked into this
  // compiler.
  const loom_target_legalizer_provider_list_t* legalizer_provider_list;
  // Optional caller-owned compile report receiving pass-level target feedback.
  loom_target_compile_report_t* compile_report;
} loom_low_pass_capability_t;

typedef struct loom_low_pass_environment_storage_t {
  // Target invocation capability entry stored for the borrowed environment
  // view.
  loom_target_pass_capability_t target_capability;
  // Low capability entry stored for the borrowed environment view.
  loom_low_pass_capability_t low_capability;
  // Target math capability entry stored for the borrowed environment view.
  loom_target_math_pass_capability_t math_capability;
  // Pointer table borrowed by |environment|.
  const loom_pass_environment_capability_t* capabilities[3];
  // Pass environment view over |capabilities|.
  loom_pass_environment_t environment;
} loom_low_pass_environment_storage_t;

// Creates a borrowed low pass capability.
loom_low_pass_capability_t loom_low_pass_capability_make(
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_low_lower_policy_registry_t* lower_policy_registry,
    const loom_target_low_legality_provider_list_t* legality_provider_list,
    const loom_target_legalizer_provider_list_t* legalizer_provider_list,
    loom_target_compile_report_t* compile_report);

// Initializes stack storage for a pass environment containing one low
// capability. The returned environment must not outlive |out_storage|.
loom_pass_environment_t loom_low_pass_environment_storage_initialize(
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_low_lower_policy_registry_t* lower_policy_registry,
    const loom_target_low_legality_provider_list_t* legality_provider_list,
    const loom_target_legalizer_provider_list_t* legalizer_provider_list,
    const loom_target_math_policy_registry_t* math_policy_registry,
    loom_target_compile_report_t* compile_report,
    loom_target_selection_t target_selection, loom_symbol_ref_t target_ref,
    loom_low_pass_environment_storage_t* out_storage);

// Looks up the low capability from |environment|. Returns NULL when absent.
const loom_low_pass_capability_t* loom_low_pass_capability_from_environment(
    const loom_pass_environment_t* environment);

// Looks up the low capability from |pass->environment|. Returns NULL when
// absent.
const loom_low_pass_capability_t* loom_low_pass_capability_from_pass(
    const loom_pass_t* pass);

// Returns the descriptor registry selected by |capability|, or NULL.
const loom_low_descriptor_registry_t*
loom_low_pass_capability_descriptor_registry(
    const loom_low_pass_capability_t* capability);

// Returns the lowering policy registry selected by |capability|, or NULL.
const loom_low_lower_policy_registry_t*
loom_low_pass_capability_lower_policy_registry(
    const loom_low_pass_capability_t* capability);

// Returns the legality provider list selected by |capability|, or NULL.
const loom_target_low_legality_provider_list_t*
loom_low_pass_capability_legality_provider_list(
    const loom_low_pass_capability_t* capability);

// Returns the legalizer provider list selected by |capability|, or NULL.
const loom_target_legalizer_provider_list_t*
loom_low_pass_capability_legalizer_provider_list(
    const loom_low_pass_capability_t* capability);

// Returns the optional compile report selected by |capability|, or NULL.
loom_target_compile_report_t* loom_low_pass_capability_compile_report(
    const loom_low_pass_capability_t* capability);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_PIPELINE_PASS_ENVIRONMENT_H_
