// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Timeline semaphore operations.

#include "hrx_internal.h"

#include <stdlib.h>

hrx_status_t hrx_semaphore_create(hrx_device_t device,
                                    uint64_t initial_value,
                                    hrx_semaphore_t* semaphore) {
  if (!device || !semaphore) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "device or semaphore is NULL");
  }

  hrx_semaphore_s* sem =
      (hrx_semaphore_s*)calloc(1, sizeof(hrx_semaphore_s));
  if (!sem) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                            "failed to allocate semaphore");
  }

  iree_status_t status = iree_hal_semaphore_create(
      device->hal_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      initial_value, IREE_HAL_SEMAPHORE_FLAG_NONE, &sem->hal_semaphore);
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
  iree_hal_semaphore_retain(semaphore->hal_semaphore);
  hrx_device_retain(semaphore->device);
  iree_atomic_ref_count_inc(&semaphore->ref_count);
}

void hrx_semaphore_release(hrx_semaphore_t semaphore) {
  iree_hal_semaphore_release(semaphore->hal_semaphore);
  hrx_device_release(semaphore->device);
  if (iree_atomic_ref_count_dec(&semaphore->ref_count) == 1) {
    free(semaphore);
  }
}

hrx_status_t hrx_semaphore_query(hrx_semaphore_t semaphore,
                                   uint64_t* value) {
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
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "semaphore is NULL");
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
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_semaphore_signal(hrx_semaphore_t semaphore,
                                    uint64_t value) {
  if (!semaphore) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "semaphore is NULL");
  }
  iree_status_t status =
      iree_hal_semaphore_signal(semaphore->hal_semaphore, value);
  return hrx_status_from_iree(status);
}
