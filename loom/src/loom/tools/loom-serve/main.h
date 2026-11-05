// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TOOLS_LOOM_SERVE_MAIN_H_
#define LOOM_TOOLS_LOOM_SERVE_MAIN_H_

#include "loom/tools/loom-serve/build_identity.h"
#include "loom/tools/loom-serve/options.h"

#ifdef __cplusplus
extern "C" {
#endif

// Exact compiler and runtime-driver composition for one serving application.
// The pointed-to identity strings, compiler options, and driver registry only
// need to remain valid for the duration of loom_serve_main.
typedef struct loom_serve_configuration_t {
  // Product version and source revision reported by help and --version.
  loom_serve_build_identity_t build_identity;
  // Compiler target package, or NULL for a runtime-only application.
  const loom_runtime_compiler_options_t* compiler_options;
  // Exact linked non-task runtime drivers, or NULL for a task-only application.
  iree_hal_driver_registry_t* driver_registry;
} loom_serve_configuration_t;

// Runs one serving application with an exact linked target composition.
int loom_serve_main(int argc, char** argv,
                    const loom_serve_configuration_t* configuration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_SERVE_MAIN_H_
