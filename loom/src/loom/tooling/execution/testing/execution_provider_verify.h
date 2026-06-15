// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test-only execution provider table audits.
//
// Execution provider sets are linked static tables. Production run/benchmark
// tools trust them as executable construction invariants; this verifier lets
// tests audit provider packages without putting duplicate scans on startup.

#ifndef LOOM_TOOLING_EXECUTION_TESTING_EXECUTION_PROVIDER_VERIFY_H_
#define LOOM_TOOLING_EXECUTION_TESTING_EXECUTION_PROVIDER_VERIFY_H_

#include "iree/base/api.h"
#include "loom/tooling/execution/execution_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Verifies execution provider and execution backend table shape.
iree_status_t loom_run_execution_provider_set_verify(
    const loom_run_execution_provider_set_t* provider_set);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_TESTING_EXECUTION_PROVIDER_VERIFY_H_
