// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/host_wait.h"

static void iree_hal_amd_gpu_host_wait_run(
    iree_hal_memory_maintenance_entry_t* entry) {
  iree_hal_amd_gpu_host_wait_t* wait = (iree_hal_amd_gpu_host_wait_t*)entry;
  if (!iree_async_semaphore_wait_tracker_try_prepare_completion(
          wait->tracker)) {
    // A detached callback owns the next enqueue. No sidecar access may follow.
    return;
  }
  iree_async_semaphore_wait_completion_t completion;
  iree_async_semaphore_wait_tracker_finalize(wait->tracker, &completion);
  // The terminal continuation may return the entire capture to reusable
  // storage.
  wait->fn(wait->user_data, completion.status);
}

static void iree_hal_amd_gpu_host_wait_enqueue(
    void* user_data, iree_atomic_slist_entry_t* entry) {
  (void)entry;
  iree_hal_amd_gpu_host_wait_t* wait = user_data;
  // The tracker grants one queue owner; the single service executor joins the
  // preceding callback before entering a requeued turn of the same sidecar.
  iree_hal_memory_maintenance_enqueue(wait->service, &wait->entry);
}

iree_status_t iree_hal_amd_gpu_host_wait_begin(
    iree_async_semaphore_wait_context_t* context,
    iree_hal_memory_maintenance_t* service, iree_hal_semaphore_list_t waits,
    iree_hal_amd_gpu_host_wait_fn_t fn, void* user_data,
    iree_allocator_t host_allocator, iree_hal_amd_gpu_host_wait_t* wait) {
  *wait = (iree_hal_amd_gpu_host_wait_t){
      .entry = {.fn = iree_hal_amd_gpu_host_wait_run},
      .service = service,
      .context = context,
      .operation =
          {
              .semaphores = (iree_async_semaphore_t**)waits.semaphores,
              .values = waits.payload_values,
              .count = waits.count,
              .mode = IREE_ASYNC_WAIT_MODE_ALL,
          },
      .fn = fn,
      .user_data = user_data,
  };
  IREE_RETURN_IF_ERROR(iree_async_semaphore_wait_tracker_allocate(
      context, &wait->operation,
      (iree_async_semaphore_wait_enqueue_callback_t){
          .fn = iree_hal_amd_gpu_host_wait_enqueue, .user_data = wait},
      host_allocator, &wait->tracker));
  // Registration owns its final access even when timepoints resolve inline.
  iree_async_semaphore_wait_tracker_register_timepoints(wait->tracker);
  return iree_ok_status();
}

void iree_hal_amd_gpu_host_wait_cancel(iree_hal_amd_gpu_host_wait_t* wait) {
  iree_async_semaphore_wait_context_request_cancellation(wait->context,
                                                         &wait->operation);
}
