// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared loom-check command-line entry point.
//
// Tool binaries provide a loom_check_environment_t that selects the dialects,
// target environment, descriptor package, and optional test-only providers
// linked into that binary. The shared entry point owns flag parsing, context
// setup, and process exit behavior.

#ifndef LOOM_TOOLS_LOOM_CHECK_MAIN_H_
#define LOOM_TOOLS_LOOM_CHECK_MAIN_H_

#include "iree/base/api.h"
#include "loom/tools/loom-check/execute.h"

#ifdef __cplusplus
extern "C" {
#endif

// Runs loom-check using |environment| as the linked tool environment.
int loom_check_main(int argc, char** argv,
                    const loom_check_environment_t* environment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_MAIN_H_
