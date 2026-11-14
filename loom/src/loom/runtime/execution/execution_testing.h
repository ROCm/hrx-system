// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_EXECUTION_EXECUTION_TESTING_H_
#define LOOM_RUNTIME_EXECUTION_EXECUTION_TESTING_H_

#include "loom/runtime/execution/execution.h"

#ifdef __cplusplus
extern "C" {
#endif

// Test-only: host-signals an idle work timeline to |value| and records it as
// the accepted frontier, so timeline exhaustion is reachable without ~2^31
// submissions. Fails unless every accepted work value has already signaled.
iree_status_t loom_runtime_execution_advance_for_testing(
    loom_runtime_execution_t* execution, uint64_t value);

// Test-only: makes the next group submission reject its node at |index| as a
// driver would after HAL validation, after accepting every earlier node. The
// injection is consumed by the submission that reaches |index|.
iree_status_t loom_runtime_execution_reject_group_node_for_testing(
    loom_runtime_execution_t* execution, iree_host_size_t index);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_RUNTIME_EXECUTION_EXECUTION_TESTING_H_
