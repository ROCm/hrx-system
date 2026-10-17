// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/host_wait.h"

static void iree_hal_amd_gpu_host_wait_complete(
    void* user_data, iree_atomic_slist_entry_t* entry) {
  (void)entry;
  iree_hal_amd_gpu_host_wait_t* wait = user_data;
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

iree_status_t iree_hal_amd_gpu_host_wait_prepare(
    iree_async_semaphore_wait_context_t* context,
    iree_hal_semaphore_list_t waits, iree_hal_amd_gpu_host_wait_fn_t fn,
    void* user_data, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_host_wait_t* wait) {
  *wait = (iree_hal_amd_gpu_host_wait_t){
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
  return iree_async_semaphore_wait_tracker_allocate(
      context, &wait->operation,
      (iree_async_semaphore_wait_enqueue_callback_t){
          .fn = iree_hal_amd_gpu_host_wait_complete, .user_data = wait},
      host_allocator, &wait->tracker);
}

void iree_hal_amd_gpu_host_wait_register(iree_hal_amd_gpu_host_wait_t* wait) {
  // Registration owns its final access even when timepoints resolve inline.
  iree_async_semaphore_wait_tracker_register_timepoints(wait->tracker);
}

iree_status_t iree_hal_amd_gpu_host_wait_begin(
    iree_async_semaphore_wait_context_t* context,
    iree_hal_semaphore_list_t waits, iree_hal_amd_gpu_host_wait_fn_t fn,
    void* user_data, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_host_wait_t* wait) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_host_wait_prepare(
      context, waits, fn, user_data, host_allocator, wait));
  iree_hal_amd_gpu_host_wait_register(wait);
  return iree_ok_status();
}

void iree_hal_amd_gpu_host_wait_cancel(iree_hal_amd_gpu_host_wait_t* wait) {
  iree_async_semaphore_wait_context_request_cancellation(wait->context,
                                                         &wait->operation);
}

void iree_hal_amd_gpu_host_wait_set_initialize(
    iree_async_semaphore_wait_context_t* context,
    iree_hal_amd_gpu_host_wait_set_t* out_set) {
  *out_set = (iree_hal_amd_gpu_host_wait_set_t){.context = context};
  iree_slim_mutex_initialize(&out_set->mutex);
}

void iree_hal_amd_gpu_host_wait_set_deinitialize(
    iree_hal_amd_gpu_host_wait_set_t* set) {
  iree_slim_mutex_deinitialize(&set->mutex);
}

static void iree_hal_amd_gpu_host_wait_entry_release(
    iree_hal_amd_gpu_host_wait_entry_t* entry) {
  if (iree_atomic_fetch_sub(&entry->borrowers, 1, iree_memory_order_acq_rel) ==
      1) {
    entry->fn(entry->user_data, entry->status);
  }
}

static void iree_hal_amd_gpu_host_wait_entry_complete(void* user_data,
                                                      iree_status_t status) {
  iree_hal_amd_gpu_host_wait_entry_t* entry = user_data;
  iree_hal_amd_gpu_host_wait_set_t* set = entry->set;
  iree_slim_mutex_lock(&set->mutex);
  if (entry->linked) {
    if (entry->previous) {
      entry->previous->next = entry->next;
    } else {
      set->head = entry->next;
    }
    if (entry->next) {
      entry->next->previous = entry->previous;
    }
    entry->linked = false;
  }
  iree_slim_mutex_unlock(&set->mutex);
  entry->status = status;
  iree_hal_amd_gpu_host_wait_entry_release(entry);
}

iree_status_t iree_hal_amd_gpu_host_wait_set_begin(
    iree_hal_amd_gpu_host_wait_set_t* set, iree_hal_semaphore_list_t waits,
    iree_hal_amd_gpu_host_wait_fn_t fn, void* user_data,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_host_wait_entry_t* entry) {
  *entry = (iree_hal_amd_gpu_host_wait_entry_t){
      .set = set, .fn = fn, .user_data = user_data};
  iree_atomic_store(&entry->borrowers, 1, iree_memory_order_relaxed);
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_host_wait_prepare(
      set->context, waits, iree_hal_amd_gpu_host_wait_entry_complete, entry,
      host_allocator, &entry->wait));
  iree_slim_mutex_lock(&set->mutex);
  const bool cancelled = set->cancelled;
  if (!cancelled) {
    entry->next = set->head;
    if (entry->next) {
      entry->next->previous = entry;
    }
    set->head = entry;
    entry->linked = true;
  }
  iree_slim_mutex_unlock(&set->mutex);
  if (cancelled) {
    iree_hal_amd_gpu_host_wait_cancel(&entry->wait);
  }
  // Even a cancelled prepared tracker requires registration to return its
  // initial borrower. That last access may complete and free entry inline.
  iree_hal_amd_gpu_host_wait_register(&entry->wait);
  return iree_ok_status();
}

void iree_hal_amd_gpu_host_wait_set_cancel(
    iree_hal_amd_gpu_host_wait_set_t* set) {
  iree_slim_mutex_lock(&set->mutex);
  set->cancelled = true;
  iree_hal_amd_gpu_host_wait_entry_t* entry = set->head;
  set->head = NULL;
  for (iree_hal_amd_gpu_host_wait_entry_t* current = entry; current;
       current = current->next) {
    iree_atomic_fetch_add(&current->borrowers, 1, iree_memory_order_relaxed);
    current->linked = false;
  }
  iree_slim_mutex_unlock(&set->mutex);
  while (entry) {
    // Detached next links are now exclusively owned by this cancellation.
    // A concurrent terminal callback can no longer unlink or free the entry.
    iree_hal_amd_gpu_host_wait_entry_t* next = entry->next;
    iree_hal_amd_gpu_host_wait_cancel(&entry->wait);
    iree_hal_amd_gpu_host_wait_entry_release(entry);
    entry = next;
  }
}
