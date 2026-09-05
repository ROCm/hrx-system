// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-linked benchmark runner configuration.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_CONFIGURATION_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_CONFIGURATION_H_

#include "iree/base/api.h"
#include "loom/tooling/execution/hal/testbench_requirement_provider.h"
#include "loom/tooling/execution/session.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_device_provider_registry_t loom_device_provider_registry_t;
typedef struct loom_target_environment_t loom_target_environment_t;

typedef struct iree_benchmark_loom_configuration_t {
  // Null-terminated executable name used in help and diagnostics.
  const char* tool_name;
  // Target-owned dialect and interface registration callback.
  loom_run_register_context_callback_t register_context;
  // Target environment linked into this runner.
  const loom_target_environment_t* target_environment;
  // Device provider registry linked into this runner.
  const loom_device_provider_registry_t* device_provider_registry;
  // Target-specific requirement provider initializers linked into this runner.
  const loom_run_hal_testbench_requirement_initializer_set_t*
      requirement_provider_initializers;
  // Target-low descriptor registry package linked into this runner.
  loom_run_initialize_low_descriptor_registry_callback_t
      initialize_low_descriptor_registry;
} iree_benchmark_loom_configuration_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_CONFIGURATION_H_
