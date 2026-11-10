// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <stdlib.h>

#include "hrx_internal.h"

// The native fence appends unique semaphores and only updates values when an
// existing semaphore is inserted. Its count/order also indexes these owners.
static hrx_semaphore_t* hrx_fence_semaphore_owners(hrx_fence_t fence) {
  return (hrx_semaphore_t*)(fence + 1);
}

hrx_status_t hrx_fence_create(size_t capacity, hrx_fence_t* fence) {
  if (!fence) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "fence is NULL");
  }
  *fence = NULL;

  // Native construction bounds capacity before sizing the owner array.
  iree_hal_fence_t* hal_fence = NULL;
  HRX_RETURN_IF_IREE_ERROR(iree_hal_fence_create(
      (iree_host_size_t)capacity, iree_allocator_system(), &hal_fence));
  hrx_fence_t created = (hrx_fence_t)calloc(
      1, sizeof(*created) + capacity * sizeof(hrx_semaphore_t));
  if (!created) {
    iree_hal_fence_release(hal_fence);
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "failed to allocate fence");
  }

  iree_atomic_ref_count_init(&created->ref_count);
  created->hal_fence = hal_fence;
  *fence = created;
  return hrx_ok_status();
}

static iree_status_t hrx_fence_insert_timepoint(hrx_fence_t fence,
                                                hrx_semaphore_t semaphore,
                                                uint64_t value) {
  const iree_host_size_t previous_count =
      iree_hal_fence_timepoint_count(fence->hal_fence);
  iree_status_t status =
      iree_hal_fence_insert(fence->hal_fence, semaphore->hal_semaphore, value);
  if (iree_hal_fence_timepoint_count(fence->hal_fence) != previous_count) {
    hrx_semaphore_retain(semaphore);
    hrx_fence_semaphore_owners(fence)[previous_count] = semaphore;
  }
  return status;
}

hrx_status_t hrx_fence_create_at(hrx_semaphore_t semaphore, uint64_t value,
                                 hrx_fence_t* fence) {
  if (!semaphore || !fence) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "semaphore or fence is NULL");
  }
  *fence = NULL;

  hrx_fence_t created = NULL;
  hrx_status_t status = hrx_fence_create(1, &created);
  if (!hrx_status_is_ok(status)) {
    return status;
  }
  status = hrx_status_from_iree(
      hrx_fence_insert_timepoint(created, semaphore, value));
  if (hrx_status_is_ok(status)) {
    *fence = created;
  } else {
    hrx_fence_release(created);
  }
  return status;
}

void hrx_fence_retain(hrx_fence_t fence) {
  if (!fence) {
    return;
  }
  iree_atomic_ref_count_inc(&fence->ref_count);
}

void hrx_fence_release(hrx_fence_t fence) {
  if (!fence) {
    return;
  }
  if (iree_atomic_ref_count_dec(&fence->ref_count) != 1) {
    return;
  }
  const iree_host_size_t count =
      iree_hal_fence_timepoint_count(fence->hal_fence);
  iree_hal_fence_release(fence->hal_fence);
  hrx_semaphore_t* owners = hrx_fence_semaphore_owners(fence);
  for (iree_host_size_t i = 0; i < count; ++i) {
    hrx_semaphore_release(owners[i]);
  }
  free(fence);
}

hrx_status_t hrx_fence_insert(hrx_fence_t fence, hrx_semaphore_t semaphore,
                              uint64_t value) {
  if (!fence || !semaphore) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "fence or semaphore is NULL");
  }
  return hrx_status_from_iree(
      hrx_fence_insert_timepoint(fence, semaphore, value));
}

hrx_status_t hrx_fence_extend(hrx_fence_t into_fence, hrx_fence_t from_fence) {
  if (!into_fence || !from_fence) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "into_fence or from_fence is NULL");
  }
  const iree_hal_semaphore_list_t list =
      iree_hal_fence_semaphore_list(from_fence->hal_fence);
  hrx_semaphore_t* owners = hrx_fence_semaphore_owners(from_fence);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < list.count && iree_status_is_ok(status);
       ++i) {
    status = hrx_fence_insert_timepoint(into_fence, owners[i],
                                        list.payload_values[i]);
  }
  return hrx_status_from_iree(status);
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
