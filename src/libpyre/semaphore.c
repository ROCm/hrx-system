// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// Timeline semaphore operations.

#include "pyre_internal.h"

#include <stdlib.h>

pyre_status_t pyre_semaphore_create(pyre_device_t device,
                                    uint64_t initial_value,
                                    pyre_semaphore_t* semaphore) {
  if (!device || !semaphore) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "device or semaphore is NULL");
  }

  pyre_semaphore_s* sem =
      (pyre_semaphore_s*)calloc(1, sizeof(pyre_semaphore_s));
  if (!sem) {
    return pyre_make_status(PYRE_STATUS_OUT_OF_MEMORY,
                            "failed to allocate semaphore");
  }

  iree_status_t status = iree_hal_semaphore_create(
      device->hal_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      initial_value, IREE_HAL_SEMAPHORE_FLAG_NONE, &sem->hal_semaphore);
  if (!iree_status_is_ok(status)) {
    free(sem);
    return pyre_status_from_iree(status);
  }

  iree_atomic_ref_count_init(&sem->ref_count);
  sem->device = device;
  *semaphore = sem;
  return pyre_ok_status();
}

pyre_status_t pyre_semaphore_retain(pyre_semaphore_t semaphore) {
  if (!semaphore) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "semaphore is NULL");
  }
  iree_atomic_ref_count_inc(&semaphore->ref_count);
  return pyre_ok_status();
}

pyre_status_t pyre_semaphore_release(pyre_semaphore_t semaphore) {
  if (!semaphore) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "semaphore is NULL");
  }
  if (iree_atomic_ref_count_dec(&semaphore->ref_count) == 1) {
    if (semaphore->hal_semaphore) {
      iree_hal_semaphore_release(semaphore->hal_semaphore);
    }
    free(semaphore);
  }
  return pyre_ok_status();
}

pyre_status_t pyre_semaphore_query(pyre_semaphore_t semaphore,
                                   uint64_t* value) {
  if (!semaphore || !value) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "semaphore or value is NULL");
  }
  iree_status_t status =
      iree_hal_semaphore_query(semaphore->hal_semaphore, value);
  return pyre_status_from_iree(status);
}

pyre_status_t pyre_semaphore_wait(pyre_semaphore_t semaphore, uint64_t value,
                                  uint64_t timeout_ns) {
  if (!semaphore) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
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
  return pyre_status_from_iree(status);
}

pyre_status_t pyre_semaphore_signal(pyre_semaphore_t semaphore,
                                    uint64_t value) {
  if (!semaphore) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "semaphore is NULL");
  }
  iree_status_t status =
      iree_hal_semaphore_signal(semaphore->hal_semaphore, value);
  return pyre_status_from_iree(status);
}
