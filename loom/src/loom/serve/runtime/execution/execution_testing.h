// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_EXECUTION_EXECUTION_TESTING_H_
#define LOOM_SERVE_RUNTIME_EXECUTION_EXECUTION_TESTING_H_

#include "loom/serve/runtime/execution/execution.h"

#ifdef __cplusplus
extern "C" {
#endif

// Test-only: host-signals an idle work timeline to |value| and records it as
// the accepted frontier, so timeline exhaustion is reachable without ~2^31
// submissions. Fails unless every accepted work value has already signaled.
iree_status_t loom_serve_execution_advance_for_testing(
    loom_serve_execution_t* execution, uint64_t value);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_RUNTIME_EXECUTION_EXECUTION_TESTING_H_
