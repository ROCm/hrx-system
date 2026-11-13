// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/execution/wait.h"

iree_status_t loom_runtime_execution_await(
    loom_runtime_readiness_t* readiness,
    const loom_runtime_execution_frontier_t* frontier,
    loom_runtime_execution_wait_t* wait) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(wait->semaphores); ++i) {
    wait->semaphores[i] = (iree_async_semaphore_t*)frontier->semaphores[i];
    wait->values[i] = frontier->values[i];
  }
  return loom_runtime_readiness_await(
      readiness, IREE_ARRAYSIZE(wait->semaphores), wait->semaphores,
      wait->values, &wait->wait);
}
