// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_EXECUTION_AWAIT_H_
#define LOOM_SERVE_RUNTIME_EXECUTION_AWAIT_H_

#include "iree/async/operations/semaphore.h"
#include "iree/async/proactor.h"
#include "loom/serve/runtime/execution/execution.h"

#ifdef __cplusplus
extern "C" {
#endif

// Caller-owned storage for one captured model readiness frontier. Successful
// submission borrows this record and both semaphores through completion.
// Failure readiness does not establish native resource retirement.
typedef struct loom_serve_execution_await_t {
  // Intrusive wait retained through the callback's final access.
  iree_async_semaphore_wait_operation_t operation;
  // Borrowed work and feedback semaphores from the captured execution.
  iree_async_semaphore_t* semaphores[2];
  // Copied values, independent of subsequent execution submissions.
  uint64_t values[2];
} loom_serve_execution_await_t;

// Submits an ALL wait using caller storage without blocking a host worker. The
// caller joins the completion callback before reusing or freeing storage. A
// rejected submission owes no callback and releases the borrow immediately.
iree_status_t loom_serve_execution_await(
    iree_async_proactor_t* proactor,
    const loom_serve_execution_frontier_t* frontier,
    iree_async_completion_fn_t completion, void* user_data,
    loom_serve_execution_await_t* storage);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_EXECUTION_AWAIT_H_
