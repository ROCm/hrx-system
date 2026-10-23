// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/testing/device.h"

#include "iree/base/threading/call_once.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/pool_wait.h"
#include "loom/serve/runtime/execution/workspace.h"

static iree_once_flag loom_serve_test_driver_once = IREE_ONCE_FLAG_INIT;
static iree_status_t loom_serve_test_driver_registration_status = 0;

static void loom_serve_test_driver_register(void) {
  loom_serve_test_driver_registration_status =
      loom_serve_test_device()->driver_register(
          iree_hal_driver_registry_default());
}

iree_status_t loom_serve_test_driver_acquire(loom_serve_host_t* host,
                                             iree_hal_driver_t** out_driver) {
  const loom_serve_test_device_t* device = loom_serve_test_device();
  *out_driver = NULL;
  if (!device->driver_register) {
    *out_driver = loom_serve_host_task_driver(host);
    iree_hal_driver_retain(*out_driver);
    return iree_ok_status();
  }
  // Native process contexts remain shared across independently created groups.
  // A fixture may release every device without ending that process lifetime.
  iree_call_once(&loom_serve_test_driver_once, loom_serve_test_driver_register);
  IREE_RETURN_IF_ERROR(
      iree_status_clone(loom_serve_test_driver_registration_status));
  return iree_hal_driver_registry_acquire_driver(
      iree_hal_driver_registry_default(), device->driver_name, out_driver);
}

typedef struct loom_serve_test_capacity_wait_t {
  // Joins the callback's last access before stack ownership ends.
  iree_slim_mutex_t mutex;
  // Wakes the outer test thread after a native capacity event.
  iree_notification_t notification;
  // Protected terminal callback result.
  iree_status_t status;
  // True after the current observation has retired.
  bool complete;
} loom_serve_test_capacity_wait_t;

static void loom_serve_test_capacity_ready(void* user_data,
                                           iree_status_t status) {
  loom_serve_test_capacity_wait_t* wait = user_data;
  iree_slim_mutex_lock(&wait->mutex);
  wait->status = status;
  wait->complete = true;
  iree_notification_post(&wait->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&wait->mutex);
}

static bool loom_serve_test_capacity_complete(void* user_data) {
  loom_serve_test_capacity_wait_t* wait = user_data;
  iree_slim_mutex_lock(&wait->mutex);
  const bool complete = wait->complete;
  iree_slim_mutex_unlock(&wait->mutex);
  return complete;
}

iree_status_t loom_serve_test_device_trim_and_drain(loom_serve_device_t* device,
                                                    uint64_t target_bytes) {
  IREE_RETURN_IF_ERROR(loom_serve_device_trim(device, target_bytes));
  loom_serve_memory_pool_t* memory = loom_serve_device_memory_pool(device);
  // A retained admission failure does not prevent already-accepted returns
  // from retiring. Join the native source before returning either diagnosis.
  iree_status_t status =
      memory ? loom_serve_memory_pool_await(memory) : iree_ok_status();
  status = iree_status_join(status, loom_serve_workspace_advance(
                                        loom_serve_device_workspace(device)));
  iree_hal_pool_t* pool = loom_serve_device_storage_pool(device);
  iree_hal_pool_wait_t* capacity = NULL;
  iree_status_t capacity_status =
      iree_hal_pool_wait_allocate(pool, iree_allocator_system(), &capacity);
  if (!iree_status_is_ok(capacity_status)) {
    return iree_status_join(status, capacity_status);
  }
  loom_serve_test_capacity_wait_t wait = {0};
  iree_slim_mutex_initialize(&wait.mutex);
  iree_notification_initialize(&wait.notification);
  while (iree_status_is_ok(wait.status)) {
    iree_hal_pool_wait_prepare(capacity);
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(pool, &stats);
    if (stats.bytes_committed <= stats.bytes_reserved + stats.bytes_abandoned) {
      iree_hal_pool_wait_abort(capacity);
      wait.status = iree_hal_pool_query_abandonment_status(pool);
      break;
    }
    wait.complete = false;
    iree_hal_pool_wait_commit(
        capacity, iree_infinite_timeout(),
        (iree_hal_pool_wait_callback_t){loom_serve_test_capacity_ready, &wait});
    iree_notification_await(&wait.notification,
                            loom_serve_test_capacity_complete, &wait,
                            iree_infinite_timeout());
  }
  iree_notification_deinitialize(&wait.notification);
  iree_slim_mutex_deinitialize(&wait.mutex);
  iree_hal_pool_wait_free(capacity);
  return iree_status_join(status, wait.status);
}
