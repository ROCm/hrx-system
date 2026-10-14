// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/continuation.h"

#include "iree/base/threading/notification.h"

enum {
  IREE_HAL_AMD_GPU_CONTINUATION_SCHEDULED = 1u << 0,
  IREE_HAL_AMD_GPU_CONTINUATION_NOTIFIED = 1u << 1,
};

static void iree_hal_amd_gpu_continuation_run(
    iree_hal_memory_maintenance_entry_t* entry) {
  iree_hal_amd_gpu_continuation_t* continuation =
      (iree_hal_amd_gpu_continuation_t*)entry;
  iree_slim_mutex_lock(&continuation->mutex);
  continuation->state &= ~IREE_HAL_AMD_GPU_CONTINUATION_NOTIFIED;
  iree_slim_mutex_unlock(&continuation->mutex);

  IREE_TRACE_ZONE_BEGIN(z0);
  const bool ready = continuation->fn(continuation->user_data);
  IREE_TRACE_ZONE_END(z0);

  iree_slim_mutex_lock(&continuation->mutex);
  const bool reschedule =
      ready || (continuation->state & IREE_HAL_AMD_GPU_CONTINUATION_NOTIFIED);
  if (!reschedule) {
    continuation->state = 0;
    if (continuation->join_notification) {
      iree_notification_post(continuation->join_notification, IREE_ALL_WAITERS);
    }
  }
  // An idle result permits deinitialization immediately after this unlock.
  iree_slim_mutex_unlock(&continuation->mutex);
  if (reschedule) {
    // Scheduling remains claimed across the FIFO handoff, including teardown.
    // The current executor cannot enter the next callback until this returns.
    iree_hal_memory_maintenance_enqueue(continuation->maintenance, entry);
  }
}

void iree_hal_amd_gpu_continuation_initialize(
    iree_hal_memory_maintenance_t* maintenance,
    iree_hal_amd_gpu_continuation_fn_t fn, void* user_data,
    iree_hal_amd_gpu_continuation_t* out_continuation) {
  memset(out_continuation, 0, sizeof(*out_continuation));
  out_continuation->entry.fn = iree_hal_amd_gpu_continuation_run;
  out_continuation->maintenance = maintenance;
  out_continuation->fn = fn;
  out_continuation->user_data = user_data;
  iree_slim_mutex_initialize(&out_continuation->mutex);
}

static bool iree_hal_amd_gpu_continuation_is_idle(void* user_data) {
  iree_hal_amd_gpu_continuation_t* continuation = user_data;
  iree_slim_mutex_lock(&continuation->mutex);
  const bool idle =
      !(continuation->state & IREE_HAL_AMD_GPU_CONTINUATION_SCHEDULED);
  iree_slim_mutex_unlock(&continuation->mutex);
  return idle;
}

void iree_hal_amd_gpu_continuation_deinitialize(
    iree_hal_amd_gpu_continuation_t* continuation) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  iree_slim_mutex_lock(&continuation->mutex);
  continuation->join_notification = &notification;
  iree_slim_mutex_unlock(&continuation->mutex);
  iree_notification_await(&notification, iree_hal_amd_gpu_continuation_is_idle,
                          continuation, iree_infinite_timeout());
  iree_notification_deinitialize(&notification);
  iree_slim_mutex_deinitialize(&continuation->mutex);
  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_amd_gpu_continuation_notify(
    iree_hal_amd_gpu_continuation_t* continuation) {
  iree_slim_mutex_lock(&continuation->mutex);
  const bool schedule =
      !(continuation->state & IREE_HAL_AMD_GPU_CONTINUATION_SCHEDULED);
  continuation->state |= IREE_HAL_AMD_GPU_CONTINUATION_SCHEDULED |
                         IREE_HAL_AMD_GPU_CONTINUATION_NOTIFIED;
  iree_slim_mutex_unlock(&continuation->mutex);
  if (schedule) {
    iree_hal_memory_maintenance_enqueue(continuation->maintenance,
                                        &continuation->entry);
  }
}
