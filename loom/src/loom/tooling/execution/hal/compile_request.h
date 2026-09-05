// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Live-device resolution for Loom kernel compile requests.

#ifndef LOOM_TOOLING_EXECUTION_HAL_COMPILE_REQUEST_H_
#define LOOM_TOOLING_EXECUTION_HAL_COMPILE_REQUEST_H_

#include "iree/base/api.h"
#include "loom/tooling/compile/request.h"
#include "loom/tooling/execution/hal/device_provider.h"
#include "loom/tooling/execution/hal/runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

// Inputs used to resolve one kernel compile request for a live HAL device.
typedef struct loom_run_hal_compile_resolve_options_t {
  // Module containing the selected kernel roots.
  const loom_module_t* module;
  // User constraints and selected kernel roots.
  loom_compile_request_options_t compile;
  // Authored target facts constraining the selected device target.
  const loom_target_facts_t* target_requirement;
  // Configured target environment used for profile selection.
  const loom_target_environment_t* target_environment;
  // Provider selected by the standard --device URI.
  const loom_device_provider_t* device_provider;
  // Active HAL runtime exposing immutable device facts.
  const loom_run_hal_runtime_t* runtime;
} loom_run_hal_compile_resolve_options_t;

// Fully resolved compile request and its selected device target.
typedef struct loom_run_hal_compile_request_t {
  // Product, format, producer, and optional explicit target resolution.
  loom_compile_request_t compile;
  // Exact executable target used for specialization, emission, and loading.
  loom_device_target_t device_target;
} loom_run_hal_compile_request_t;

// Resolves one kernel request against the provider selected by --device.
//
// Resolution performs no allocation and compiles no candidates. An explicit
// --target must be advertised verbatim by the active device. With no explicit
// target, the device provider selects its best target satisfying the authored
// requirement. The resolved format must be the artifact format loadable by
// the selected device provider.
iree_status_t loom_run_hal_compile_request_resolve(
    const loom_run_hal_compile_resolve_options_t* options,
    loom_run_hal_compile_request_t* out_request);

// Returns the profile used to specialize |request|'s selected kernel roots.
//
// Explicit requests borrow their configured process-lifetime profile. An
// implicit live-device request initializes |out_device_profile| as stack-only
// adapter storage and returns its base profile. The returned profile remains
// valid until |out_device_profile| leaves scope.
iree_status_t loom_run_hal_compile_request_target_profile(
    const loom_run_hal_compile_request_t* request,
    const loom_device_provider_t* device_provider,
    const loom_run_hal_runtime_t* runtime,
    loom_device_target_profile_t* out_device_profile,
    const loom_target_profile_t** out_profile);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_COMPILE_REQUEST_H_
