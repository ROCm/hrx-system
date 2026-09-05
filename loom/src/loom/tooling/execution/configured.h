// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Execution providers selected by the Loom build configuration.

#ifndef LOOM_TOOLING_EXECUTION_CONFIGURED_H_
#define LOOM_TOOLING_EXECUTION_CONFIGURED_H_

#include "loom/tooling/execution/execution_provider.h"
#include "loom/tooling/execution/hal/device_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Process-lifetime execution provider tables selected by the build.
typedef struct loom_tooling_execution_providers_t {
  // Target and execution backend contributions.
  loom_run_execution_provider_set_t execution_provider_set;
  // Live HAL device adapters corresponding to the execution providers.
  loom_device_provider_registry_t device_provider_registry;
} loom_tooling_execution_providers_t;

// Returns the immutable configured execution provider tables.
const loom_tooling_execution_providers_t*
loom_tooling_configured_execution_providers(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_CONFIGURED_H_
