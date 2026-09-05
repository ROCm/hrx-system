// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Execution providers selected by the Loom build configuration.

#ifndef LOOM_TOOLING_EXECUTION_CONFIGURED_H_
#define LOOM_TOOLING_EXECUTION_CONFIGURED_H_

#include "loom/tooling/execution/execution_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the immutable execution provider set selected by the build.
const loom_run_execution_provider_set_t*
loom_tooling_configured_execution_providers(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_CONFIGURED_H_
