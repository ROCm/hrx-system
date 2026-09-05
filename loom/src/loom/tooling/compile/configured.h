// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Offline compiler providers selected by the Loom build configuration.

#ifndef LOOM_TOOLING_COMPILE_CONFIGURED_H_
#define LOOM_TOOLING_COMPILE_CONFIGURED_H_

#include "loom/target/provider.h"
#include "loom/tooling/compile/artifact.h"

#ifdef __cplusplus
extern "C" {
#endif

// Borrowed process-lifetime view of configured offline compiler providers.
typedef struct loom_tooling_compile_environment_t {
  // Target compiler environment, including portable command descriptors.
  const loom_target_environment_t* target_environment;
  // Loadable artifact providers selected by the build configuration.
  const loom_artifact_provider_registry_t* artifact_provider_registry;
} loom_tooling_compile_environment_t;

// Returns the immutable configured offline compiler environment.
const loom_tooling_compile_environment_t*
loom_tooling_configured_compile_environment(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_COMPILE_CONFIGURED_H_
