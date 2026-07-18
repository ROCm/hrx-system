// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Timeline semaphore operations.

#include <stdlib.h>

#include "hrx_internal.h"

hrx_status_t hrx_semaphore_create(hrx_device_t device, uint64_t initial_value,
                                  hrx_semaphore_t* semaphore) {
  if (!device || !semaphore) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "device or semaphore is NULL");
  }

  hrx_semaphore_s* sem = (hrx_semaphore_s*)calloc(1, sizeof(hrx_semaphore_s));
  if (!sem) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "failed to allocate semaphore");
  }

  iree_status_t status = iree_hal_semaphore_create(
      device->hal_device, IREE_HAL_QUEUE_AFFINITY_ANY, initial_value,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &sem->hal_semaphore);
  if (!iree_status_is_ok(status)) {
    free(sem);
    return hrx_status_from_iree(status);
  }

  iree_atomic_ref_count_init(&sem->ref_count);
  sem->device = device;
  hrx_device_retain(sem->device);
  *semaphore = sem;
  return hrx_ok_status();
}

void hrx_semaphore_retain(hrx_semaphore_t semaphore) {
  if (!semaphore) return;
  iree_hal_semaphore_retain(semaphore->hal_semaphore);
  hrx_device_retain(semaphore->device);
  iree_atomic_ref_count_inc(&semaphore->ref_count);
}

void hrx_semaphore_release(hrx_semaphore_t semaphore) {
  if (!semaphore) return;
  iree_hal_semaphore_release(semaphore->hal_semaphore);
  hrx_device_release(semaphore->device);
  if (iree_atomic_ref_count_dec(&semaphore->ref_count) == 1) {
    free(semaphore);
  }
}

hrx_status_t hrx_semaphore_query(hrx_semaphore_t semaphore, uint64_t* value) {
  if (!semaphore || !value) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "semaphore or value is NULL");
  }
  iree_status_t status =
      iree_hal_semaphore_query(semaphore->hal_semaphore, value);
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_semaphore_wait(hrx_semaphore_t semaphore, uint64_t value,
                                uint64_t timeout_ns) {
  if (!semaphore) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "semaphore is NULL");
  }
  iree_timeout_t timeout;
  if (timeout_ns == UINT64_MAX) {
    timeout = iree_infinite_timeout();
  } else if (timeout_ns == 0) {
    timeout = iree_immediate_timeout();
  } else {
    timeout = iree_make_timeout_ns(timeout_ns);
  }

  iree_status_t status =
      iree_hal_semaphore_wait(semaphore->hal_semaphore, value, timeout,
                              /*flags=*/0);
  if (!iree_status_is_ok(status)) {
    // iree_async_semaphore_multi_wait reports failures as
    // iree_status_from_code(code), dropping the annotation; the semaphore still
    // owns the original. Query it so callers see the producer's message (e.g.
    // "amdxdna ... did not complete: ert state N (error_index M)") instead of a
    // bare code. Keep the wait status if the semaphore has nothing better.
    uint64_t current_value = 0;
    iree_status_t detail =
        iree_hal_semaphore_query(semaphore->hal_semaphore, &current_value);
    if (!iree_status_is_ok(detail)) {
      iree_status_ignore(status);
      status = detail;
    } else {
      iree_status_ignore(detail);
    }
  }
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_semaphore_signal(hrx_semaphore_t semaphore, uint64_t value) {
  if (!semaphore) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "semaphore is NULL");
  }
  iree_status_t status =
      iree_hal_semaphore_signal(semaphore->hal_semaphore, value,
                                /*frontier=*/NULL);
  return hrx_status_from_iree(status);
}
