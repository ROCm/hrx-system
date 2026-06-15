// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <stdlib.h>

#include "hrx_internal.h"

hrx_status_t hrx_fence_create(size_t capacity, hrx_fence_t* fence) {
  if (!fence) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "fence is NULL");
  }
  *fence = NULL;

  hrx_fence_t created = (hrx_fence_t)calloc(1, sizeof(*created));
  if (!created) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "failed to allocate fence");
  }

  iree_status_t status = iree_hal_fence_create(
      (iree_host_size_t)capacity, iree_allocator_system(), &created->hal_fence);
  if (!iree_status_is_ok(status)) {
    free(created);
    return hrx_status_from_iree(status);
  }

  iree_atomic_ref_count_init(&created->ref_count);
  *fence = created;
  return hrx_ok_status();
}

hrx_status_t hrx_fence_create_at(hrx_semaphore_t semaphore, uint64_t value,
                                 hrx_fence_t* fence) {
  if (!semaphore || !fence) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "semaphore or fence is NULL");
  }
  *fence = NULL;

  hrx_fence_t created = (hrx_fence_t)calloc(1, sizeof(*created));
  if (!created) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "failed to allocate fence");
  }

  iree_status_t status =
      iree_hal_fence_create_at(semaphore->hal_semaphore, value,
                               iree_allocator_system(), &created->hal_fence);
  if (!iree_status_is_ok(status)) {
    free(created);
    return hrx_status_from_iree(status);
  }

  iree_atomic_ref_count_init(&created->ref_count);
  *fence = created;
  return hrx_ok_status();
}

void hrx_fence_retain(hrx_fence_t fence) {
  if (!fence) return;
  iree_hal_fence_retain(fence->hal_fence);
  iree_atomic_ref_count_inc(&fence->ref_count);
}

void hrx_fence_release(hrx_fence_t fence) {
  if (!fence) return;
  iree_hal_fence_release(fence->hal_fence);
  if (iree_atomic_ref_count_dec(&fence->ref_count) == 1) {
    free(fence);
  }
}

hrx_status_t hrx_fence_insert(hrx_fence_t fence, hrx_semaphore_t semaphore,
                              uint64_t value) {
  if (!fence || !semaphore) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "fence or semaphore is NULL");
  }
  return hrx_status_from_iree(
      iree_hal_fence_insert(fence->hal_fence, semaphore->hal_semaphore, value));
}

hrx_status_t hrx_fence_extend(hrx_fence_t into_fence, hrx_fence_t from_fence) {
  if (!into_fence || !from_fence) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "into_fence or from_fence is NULL");
  }
  return hrx_status_from_iree(
      iree_hal_fence_extend(into_fence->hal_fence, from_fence->hal_fence));
}

hrx_status_t hrx_fence_signal(hrx_fence_t fence) {
  if (!fence) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "fence is NULL");
  }
  return hrx_status_from_iree(iree_hal_fence_signal(fence->hal_fence));
}

hrx_status_t hrx_fence_wait(hrx_fence_t fence, uint64_t timeout_ns) {
  if (!fence) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "fence is NULL");
  }

  iree_timeout_t timeout = iree_make_timeout_ns(timeout_ns);
  if (timeout_ns == UINT64_MAX) {
    timeout = iree_infinite_timeout();
  } else if (timeout_ns == 0) {
    timeout = iree_immediate_timeout();
  }

  return hrx_status_from_iree(
      iree_hal_fence_wait(fence->hal_fence, timeout, 0));
}
