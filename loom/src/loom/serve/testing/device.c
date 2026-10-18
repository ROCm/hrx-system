// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/testing/device.h"

#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/pool_wait.h"

iree_status_t loom_serve_test_driver_create(loom_serve_host_t* host,
                                            iree_allocator_t host_allocator,
                                            iree_hal_driver_t** out_driver) {
  const loom_serve_test_device_t* device = loom_serve_test_device();
  *out_driver = NULL;
  if (!device->driver_register) {
    *out_driver = loom_serve_host_task_driver(host);
    iree_hal_driver_retain(*out_driver);
    return iree_ok_status();
  }
  iree_hal_driver_registry_t* registry = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_driver_registry_allocate(host_allocator, &registry));
  iree_status_t status = device->driver_register(registry);
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_acquire_driver(
        registry, device->driver_name, out_driver);
  }
  iree_hal_driver_registry_free(registry);
  return status;
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
  if (memory) {
    IREE_RETURN_IF_ERROR(loom_serve_memory_pool_await(memory));
  }
  iree_hal_pool_t* pool = loom_serve_device_storage_pool(device);
  iree_hal_pool_wait_t* capacity = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_pool_wait_allocate(pool, iree_allocator_system(), &capacity));
  loom_serve_test_capacity_wait_t wait = {0};
  iree_slim_mutex_initialize(&wait.mutex);
  iree_notification_initialize(&wait.notification);
  while (iree_status_is_ok(wait.status)) {
    iree_hal_pool_wait_prepare(capacity);
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(pool, &stats);
    if (stats.bytes_committed <= stats.bytes_reserved) {
      iree_hal_pool_wait_abort(capacity);
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
  return wait.status;
}
