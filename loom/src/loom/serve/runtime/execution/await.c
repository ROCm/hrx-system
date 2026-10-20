// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/execution/await.h"

iree_status_t loom_serve_execution_await(
    iree_async_proactor_t* proactor,
    const loom_serve_execution_frontier_t* frontier,
    iree_async_completion_fn_t completion, void* user_data,
    loom_serve_execution_await_t* storage) {
  *storage = (loom_serve_execution_await_t){0};
  iree_async_semaphore_wait_operation_t* wait = &storage->operation;
  iree_async_operation_initialize(
      &wait->base, IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE, completion, user_data);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(storage->values); ++i) {
    storage->semaphores[i] = (iree_async_semaphore_t*)frontier->semaphores[i];
    storage->values[i] = frontier->values[i];
  }
  wait->semaphores = storage->semaphores;
  wait->values = storage->values;
  wait->count = IREE_ARRAYSIZE(storage->values);
  wait->mode = IREE_ASYNC_WAIT_MODE_ALL;
  return iree_async_proactor_submit_one(proactor, &wait->base);
}
