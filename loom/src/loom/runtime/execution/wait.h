// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_EXECUTION_WAIT_H_
#define LOOM_RUNTIME_EXECUTION_WAIT_H_

#include "loom/runtime/execution/execution.h"
#include "loom/runtime/host/readiness.h"

#ifdef __cplusplus
extern "C" {
#endif

// One captured work/feedback frontier. Later submissions cannot change the
// accepted timepoints. Storage remains stable until the embedded wait result
// is consumed.
typedef struct loom_runtime_execution_wait_t {
  // Independent asynchronous readiness, including the final callback outcome.
  loom_runtime_readiness_wait_t wait;
  // Borrowed toll-free async handles of the execution domain's semaphores.
  iree_async_semaphore_t* semaphores[2];
  // Accepted values, captured before another domain submission may advance.
  uint64_t values[2];
} loom_runtime_execution_wait_t;

// Captures frontier and accepts an ALL wait without blocking. The frontier
// descriptor is borrowed only during this call; its semaphores remain borrowed
// through consumption of the embedded wait result. Failure owes no readiness
// callback.
// Readiness, including failure, does not replace native capture retirement.
iree_status_t loom_runtime_execution_await(
    loom_runtime_readiness_t* readiness,
    const loom_runtime_execution_frontier_t* frontier,
    loom_runtime_execution_wait_t* wait);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_EXECUTION_WAIT_H_
