// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/owner.h"

static iree_status_t iree_task_owner_drain(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
  iree_task_owner_t* owner =
      iree_containerof(process, iree_task_owner_t, process);
  if (!iree_slim_mutex_try_lock(&owner->mutex)) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!owner->completed) {
    owner->advance(process->user_data, context, result);
    owner->completed = result->completed;
  }
  result->completed = owner->completed;
  iree_slim_mutex_unlock(&owner->mutex);
  IREE_TRACE_ZONE_END(z0);
  // Failure policy belongs to the application; it must join accepted borrowers.
  return iree_ok_status();
}

static void iree_task_owner_released(iree_task_process_t* process) {
  iree_task_owner_t* owner =
      iree_containerof(process, iree_task_owner_t, process);
  iree_slim_mutex_lock(&owner->retirement.mutex);
  owner->retirement.released = true;
  iree_notification_post(&owner->retirement.notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&owner->retirement.mutex);
}

static bool iree_task_owner_is_released(void* user_data) {
  iree_task_owner_t* owner = user_data;
  iree_slim_mutex_lock(&owner->retirement.mutex);
  const bool released = owner->retirement.released;
  iree_slim_mutex_unlock(&owner->retirement.mutex);
  return released;
}

void iree_task_owner_initialize(iree_task_executor_t* executor,
                                iree_task_owner_advance_fn_t advance,
                                void* user_data, iree_task_owner_t* out_owner) {
  out_owner->executor = executor;
  out_owner->advance = advance;
  out_owner->completed = false;
  out_owner->retirement.released = false;
  iree_slim_mutex_initialize(&out_owner->mutex);
  iree_slim_mutex_initialize(&out_owner->retirement.mutex);
  iree_notification_initialize(&out_owner->retirement.notification);
  iree_task_process_initialize(iree_task_owner_drain, 0, 1,
                               &out_owner->process);
  iree_task_process_set_flags(&out_owner->process,
                              IREE_TASK_PROCESS_FLAG_COMPUTE_SLOT);
  out_owner->process.user_data = user_data;
  out_owner->process.release_fn = iree_task_owner_released;
}

void iree_task_owner_deinitialize(iree_task_owner_t* owner) {
  iree_notification_deinitialize(&owner->retirement.notification);
  iree_slim_mutex_deinitialize(&owner->retirement.mutex);
  iree_slim_mutex_deinitialize(&owner->mutex);
}

void iree_task_owner_wake(iree_task_owner_t* owner) {
  iree_task_executor_schedule_process(owner->executor, &owner->process);
}

void iree_task_owner_call(iree_task_owner_t* owner, void (*call)(void*),
                          void* user_data) {
  iree_slim_mutex_lock(&owner->mutex);
  call(user_data);
  iree_slim_mutex_unlock(&owner->mutex);
  iree_task_owner_wake(owner);
}

void iree_task_owner_await(iree_task_owner_t* owner) {
  iree_notification_await(&owner->retirement.notification,
                          iree_task_owner_is_released, owner,
                          iree_infinite_timeout());
}
