// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared iree-run-loom command-line implementation.

#ifndef LOOM_TOOLS_IREE_RUN_LOOM_MAIN_H_
#define LOOM_TOOLS_IREE_RUN_LOOM_MAIN_H_

#include "iree/base/api.h"
#include "loom/tooling/execution/execution_backend.h"
#include "loom/tooling/execution/session.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_environment_t loom_target_environment_t;

typedef struct iree_run_loom_configuration_t {
  // Null-terminated executable name used in help and diagnostics.
  const char* tool_name;
  // Target-owned dialect registration selected by linked providers.
  loom_run_register_context_callback_t register_context;
  // Target-low descriptor registry package linked into this runner.
  loom_run_initialize_low_descriptor_registry_callback_t
      initialize_low_descriptor_registry;
  // Target environment linked into this runner and used for pass pipelines.
  const loom_target_environment_t* target_environment;
  // Execution backends linked into this runner.
  loom_run_execution_backend_registry_t execution_backend_registry;
} iree_run_loom_configuration_t;

// Runs the configured iree-run-loom command-line tool.
int iree_run_loom_main(int argc, char** argv,
                       const iree_run_loom_configuration_t* configuration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_RUN_LOOM_MAIN_H_
