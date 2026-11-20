// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/completion_pool.h"

#include <stddef.h>

void iree_async_posix_completion_pool_initialize_with_storage(
    iree_host_size_t capacity, iree_async_posix_completion_t* entries,
    iree_async_posix_completion_pool_t* out_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_pool, 0, sizeof(*out_pool));
  out_pool->capacity = capacity;

  iree_atomic_slist_initialize(&out_pool->free_list);
  for (iree_host_size_t i = 0; i < capacity; ++i) {
    entries[i].worker_reserved = false;
    iree_atomic_slist_push(&out_pool->free_list, &entries[i].slist_entry);
  }

  IREE_TRACE_ZONE_END(z0);
}

void iree_async_posix_completion_pool_deinitialize(
    iree_async_posix_completion_pool_t* pool) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_atomic_slist_deinitialize(&pool->free_list);
  memset(pool, 0, sizeof(*pool));

  IREE_TRACE_ZONE_END(z0);
}

iree_async_posix_completion_t* iree_async_posix_completion_pool_acquire(
    iree_async_posix_completion_pool_t* pool) {
  iree_atomic_slist_entry_t* entry = iree_atomic_slist_pop(&pool->free_list);
  if (!entry) {
    return NULL;
  }

  iree_async_posix_completion_t* completion =
      iree_containerof(entry, iree_async_posix_completion_t, slist_entry);

  // Clear the entry for reuse.
  completion->operation = NULL;
  completion->status = iree_ok_status();
  completion->flags = IREE_ASYNC_COMPLETION_FLAG_NONE;

  return completion;
}

void iree_async_posix_completion_pool_release(
    iree_async_posix_completion_pool_t* pool,
    iree_async_posix_completion_t* completion) {
  // Callers transfer ownership of the status before returning the slot.
  IREE_ASSERT(iree_status_is_ok(completion->status),
              "completion released with unconsumed status");

  completion->operation = NULL;
  completion->status = iree_ok_status();
  completion->flags = IREE_ASYNC_COMPLETION_FLAG_NONE;

  iree_atomic_slist_push(&pool->free_list, &completion->slist_entry);
}
