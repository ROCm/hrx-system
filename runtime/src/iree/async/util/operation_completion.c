// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/operation_completion.h"

#include "iree/async/operations/net.h"
#include "iree/async/region.h"
#include "iree/async/util/operation_pool.h"

iree_host_size_t iree_async_operation_complete(
    iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, operation->type);
  const bool is_final =
      !iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE);
  iree_async_operation_pool_t* pool = is_final ? operation->pool : NULL;
  status = iree_async_operation_resolve_completion(operation, status, &flags);

  iree_async_region_t*
      retained_regions[IREE_ASYNC_SOCKET_SCATTER_GATHER_MAX_BUFFERS];
  const uint8_t retained_region_count =
      is_final
          ? iree_async_operation_release_resources(operation, retained_regions)
          : 0;

  // Final completion returns operation ownership to the caller. Clear all
  // backend-private state before the callback so it may immediately reuse or
  // resubmit the operation without carrying cancellation or iteration state
  // from the completed execution. Multishot operations retain their state
  // until the final completion.
  if (is_final) {
    iree_async_operation_clear_internal_flags(operation);
  }

  iree_host_size_t callback_count = 0;
  if (operation->completion_fn) {
    operation->completion_fn(operation->user_data, operation, status, flags);
    callback_count = 1;
  } else {
    iree_status_free(status);
  }

  for (uint8_t i = 0; i < retained_region_count; ++i) {
    iree_async_region_release(retained_regions[i]);
  }
  if (pool) {
    iree_async_operation_pool_release(pool, operation);
  }
  IREE_TRACE_ZONE_END(z0);
  return callback_count;
}
