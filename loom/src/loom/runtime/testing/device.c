// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/testing/device.h"

#include "iree/base/threading/call_once.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/pool_wait.h"
#include "loom/runtime/execution/workspace.h"

static iree_once_flag loom_runtime_test_driver_once = IREE_ONCE_FLAG_INIT;
static iree_status_t loom_runtime_test_driver_registration_status = 0;

static void loom_runtime_test_driver_register(void) {
  loom_runtime_test_driver_registration_status =
      loom_runtime_test_device()->driver_register(
          iree_hal_driver_registry_default());
}

iree_status_t loom_runtime_test_driver_acquire(loom_runtime_host_t* host,
                                               iree_hal_driver_t** out_driver) {
  const loom_runtime_test_device_t* device = loom_runtime_test_device();
  *out_driver = NULL;
  if (!device->driver_register) {
    *out_driver = loom_runtime_host_task_driver(host);
    iree_hal_driver_retain(*out_driver);
    return iree_ok_status();
  }
  // Native process contexts remain shared across independently created groups.
  // A fixture may release every device without ending that process lifetime.
  iree_call_once(&loom_runtime_test_driver_once,
                 loom_runtime_test_driver_register);
  IREE_RETURN_IF_ERROR(
      iree_status_clone(loom_runtime_test_driver_registration_status));
  return iree_hal_driver_registry_acquire_driver(
      iree_hal_driver_registry_default(), device->driver_name, out_driver);
}

iree_status_t loom_runtime_test_workspace_await_idle(
    loom_runtime_workspace_t* workspace) {
  // Use the shortest enabled idle interval. Readiness is the last raw root's
  // retirement, not elapsed time or the model's completion semaphore.
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         loom_runtime_workspace_trim_idle(workspace, iree_time_now(), 1)) {
    status = loom_runtime_workspace_advance(workspace);
    iree_thread_yield();
  }
  return status;
}

typedef struct loom_runtime_test_capacity_wait_t {
  // Joins the callback's last access before stack ownership ends.
  iree_slim_mutex_t mutex;
  // Wakes the outer test thread after a native capacity event.
  iree_notification_t notification;
  // Protected terminal callback result.
  iree_status_t status;
  // True after the current observation has retired.
  bool complete;
} loom_runtime_test_capacity_wait_t;

static void loom_runtime_test_capacity_ready(void* user_data,
                                             iree_status_t status) {
  loom_runtime_test_capacity_wait_t* wait = user_data;
  iree_slim_mutex_lock(&wait->mutex);
  wait->status = status;
  wait->complete = true;
  iree_notification_post(&wait->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&wait->mutex);
}

static bool loom_runtime_test_capacity_complete(void* user_data) {
  loom_runtime_test_capacity_wait_t* wait = user_data;
  iree_slim_mutex_lock(&wait->mutex);
  const bool complete = wait->complete;
  iree_slim_mutex_unlock(&wait->mutex);
  return complete;
}

static bool loom_runtime_test_pool_trim_complete(void* user_data) {
  return !loom_runtime_workspace_is_reclaiming(user_data);
}

iree_status_t loom_runtime_test_pool_trim_and_drain(loom_runtime_pool_t* owner,
                                                    uint64_t target_bytes) {
  IREE_RETURN_IF_ERROR(loom_runtime_pool_trim(owner, target_bytes));
  loom_runtime_memory_pool_t* memory = loom_runtime_pool_memory(owner);
  if (memory) {
    // Elastic allocators pack live buffers into partially occupied slabs.
    // Join accepted memory work and advance its cold-owner publication before
    // taking the final accounting snapshot. The first trim may observe zero
    // committed bytes while an accepted workspace acquisition has not
    // published its physical slab yet. Unrelated capacity-blocked acquisitions
    // remain pending and cannot prevent reclamation from completing.
    loom_runtime_workspace_t* workspace = loom_runtime_pool_workspace(owner);
    iree_notification_t notification;
    iree_notification_initialize(&notification);
    iree_status_t status = loom_runtime_workspace_await(
        workspace, &notification, loom_runtime_test_pool_trim_complete,
        workspace);
    status = iree_status_join(status, loom_runtime_memory_pool_await(memory));
    status =
        iree_status_join(status, loom_runtime_workspace_advance(workspace));
    if (iree_status_is_ok(status)) {
      status = loom_runtime_pool_trim(owner, target_bytes);
    }
    if (iree_status_is_ok(status)) {
      status = loom_runtime_workspace_await(
          workspace, &notification, loom_runtime_test_pool_trim_complete,
          workspace);
    }
    iree_notification_deinitialize(&notification);
    status = iree_status_join(status, loom_runtime_memory_pool_await(memory));
    loom_runtime_pool_advance(owner);
    return status;
  }
  // A retained admission failure does not prevent already-accepted returns
  // from retiring. Join the native source before returning either diagnosis.
  iree_status_t status =
      loom_runtime_workspace_advance(loom_runtime_pool_workspace(owner));
  iree_hal_pool_t* pool = loom_runtime_pool_storage(owner);
  iree_hal_pool_wait_t* capacity = NULL;
  iree_status_t capacity_status =
      iree_hal_pool_wait_allocate(pool, iree_allocator_system(), &capacity);
  if (!iree_status_is_ok(capacity_status)) {
    return iree_status_join(status, capacity_status);
  }
  loom_runtime_test_capacity_wait_t wait = {0};
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
    iree_hal_pool_wait_commit(capacity, iree_infinite_timeout(),
                              (iree_hal_pool_wait_callback_t){
                                  loom_runtime_test_capacity_ready, &wait});
    iree_notification_await(&wait.notification,
                            loom_runtime_test_capacity_complete, &wait,
                            iree_infinite_timeout());
  }
  iree_notification_deinitialize(&wait.notification);
  iree_slim_mutex_deinitialize(&wait.mutex);
  iree_hal_pool_wait_free(capacity);
  status = iree_status_join(status, wait.status);
  loom_runtime_pool_advance(owner);
  return status;
}
