// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Optional exact-build synthetic IR support for compiler tools.

#ifndef LOOM_TESTING_TARGET_ENVIRONMENT_H_
#define LOOM_TESTING_TARGET_ENVIRONMENT_H_

#include "loomc/target.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates an environment containing synthetic test dialects and target IR.
// Contributes no test passes or lowering pipelines. Compose with configured IR
// using loomc_target_environment_create. The context retains the composed
// environment; callers release their references normally.
//
// This optional component belongs to the compiler owner in shared tool bundles.
// The header is an exact-build interface and is not part of the installed SDK.
LOOMC_API_EXPORT loomc_status_t loomc_target_environment_create_test_ir(
    loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TESTING_TARGET_ENVIRONMENT_H_
