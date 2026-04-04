// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_internal.h"

#include <stdlib.h>

pyre_status_t pyre_fence_create(size_t capacity, pyre_fence_t* fence) {
  if (!fence) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "fence is NULL");
  }
  *fence = NULL;

  pyre_fence_t created = (pyre_fence_t)calloc(1, sizeof(*created));
  if (!created) {
    return pyre_make_status(PYRE_STATUS_OUT_OF_MEMORY,
                            "failed to allocate fence");
  }

  iree_status_t status = iree_hal_fence_create(
      (iree_host_size_t)capacity, iree_allocator_system(),
      &created->hal_fence);
  if (!iree_status_is_ok(status)) {
    free(created);
    return pyre_status_from_iree(status);
  }

  iree_atomic_ref_count_init(&created->ref_count);
  *fence = created;
  return pyre_ok_status();
}

pyre_status_t pyre_fence_create_at(pyre_semaphore_t semaphore,
                                   uint64_t value,
                                   pyre_fence_t* fence) {
  if (!semaphore || !fence) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "semaphore or fence is NULL");
  }
  *fence = NULL;

  pyre_fence_t created = (pyre_fence_t)calloc(1, sizeof(*created));
  if (!created) {
    return pyre_make_status(PYRE_STATUS_OUT_OF_MEMORY,
                            "failed to allocate fence");
  }

  iree_status_t status = iree_hal_fence_create_at(
      semaphore->hal_semaphore, value,
      iree_allocator_system(), &created->hal_fence);
  if (!iree_status_is_ok(status)) {
    free(created);
    return pyre_status_from_iree(status);
  }

  iree_atomic_ref_count_init(&created->ref_count);
  *fence = created;
  return pyre_ok_status();
}

void pyre_fence_retain(pyre_fence_t fence) {
  iree_hal_fence_retain(fence->hal_fence);
  iree_atomic_ref_count_inc(&fence->ref_count);
}

void pyre_fence_release(pyre_fence_t fence) {
  iree_hal_fence_release(fence->hal_fence);
  if (iree_atomic_ref_count_dec(&fence->ref_count) == 1) {
    free(fence);
  }
}

pyre_status_t pyre_fence_insert(pyre_fence_t fence,
                                pyre_semaphore_t semaphore,
                                uint64_t value) {
  if (!fence || !semaphore) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "fence or semaphore is NULL");
  }
  return pyre_status_from_iree(
      iree_hal_fence_insert(fence->hal_fence, semaphore->hal_semaphore,
                            value));
}

pyre_status_t pyre_fence_extend(pyre_fence_t into_fence,
                                pyre_fence_t from_fence) {
  if (!into_fence || !from_fence) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "into_fence or from_fence is NULL");
  }
  return pyre_status_from_iree(
      iree_hal_fence_extend(into_fence->hal_fence,
                            from_fence->hal_fence));
}

pyre_status_t pyre_fence_signal(pyre_fence_t fence) {
  if (!fence) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "fence is NULL");
  }
  return pyre_status_from_iree(iree_hal_fence_signal(fence->hal_fence));
}

pyre_status_t pyre_fence_wait(pyre_fence_t fence, uint64_t timeout_ns) {
  if (!fence) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "fence is NULL");
  }

  iree_timeout_t timeout = iree_make_timeout_ns(timeout_ns);
  if (timeout_ns == UINT64_MAX) {
    timeout = iree_infinite_timeout();
  } else if (timeout_ns == 0) {
    timeout = iree_immediate_timeout();
  }

  return pyre_status_from_iree(
      iree_hal_fence_wait(fence->hal_fence, timeout, 0));
}
