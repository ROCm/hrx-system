// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/operation_completion.h"

#include "iree/async/util/operation_pool.h"

iree_host_size_t iree_async_operation_complete(
    iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  const bool is_final =
      !iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE);
  iree_async_operation_pool_t* pool = is_final ? operation->pool : NULL;
  status = iree_async_operation_resolve_completion(operation, status, &flags);

  iree_host_size_t callback_count = 0;
  if (operation->completion_fn) {
    operation->completion_fn(operation->user_data, operation, status, flags);
    callback_count = 1;
  } else {
    iree_status_free(status);
  }

  if (pool) {
    iree_async_operation_pool_release(pool, operation);
  }
  return callback_count;
}
