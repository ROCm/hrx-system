// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Host-side timeline semaphore aligned with
// iree/hal/drivers/local_sync/sync_semaphore.c (no GPU primitive).

#include "iree/hal/drivers/amdxdna/semaphore.h"

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amdxdna/completion_queue.h"
#include "iree/hal/semaphore.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdxdna_semaphore_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdxdna_semaphore_t {
  iree_async_semaphore_t async;
  iree_slim_mutex_t mutex;
  iree_hal_amdxdna_completion_batch_t* native_signal_batch;
  uint64_t native_signal_value;
  iree_allocator_t host_allocator;
} iree_hal_amdxdna_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_amdxdna_semaphore_vtable;

static iree_hal_amdxdna_completion_batch_t*
iree_hal_amdxdna_semaphore_try_retain_native_signal(
    iree_hal_amdxdna_semaphore_t* semaphore, uint64_t value) {
  iree_hal_amdxdna_completion_batch_t* native_signal_batch = NULL;
  iree_slim_mutex_lock(&semaphore->mutex);
  if (semaphore->native_signal_batch &&
      semaphore->native_signal_value >= value) {
    native_signal_batch = iree_hal_amdxdna_completion_batch_retain(
        semaphore->native_signal_batch);
  }
  iree_slim_mutex_unlock(&semaphore->mutex);
  return native_signal_batch;
}

static iree_hal_amdxdna_semaphore_t* iree_hal_amdxdna_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdxdna_semaphore_vtable);
  return (iree_hal_amdxdna_semaphore_t*)base_value;
}

iree_status_t iree_hal_amdxdna_semaphore_create(
    iree_async_proactor_t* proactor, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)queue_affinity;
  (void)flags;

  iree_hal_amdxdna_semaphore_t* semaphore = NULL;
  iree_host_size_t frontier_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_semaphore_layout(sizeof(*semaphore), 0, &frontier_offset,
                                      &total_size));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&semaphore));
  iree_async_semaphore_initialize(
      (const iree_async_semaphore_vtable_t*)&iree_hal_amdxdna_semaphore_vtable,
      proactor, initial_value, frontier_offset, 0, &semaphore->async);
  iree_slim_mutex_initialize(&semaphore->mutex);
  semaphore->native_signal_batch = NULL;
  semaphore->native_signal_value = 0;
  semaphore->host_allocator = host_allocator;
  *out_semaphore = iree_hal_semaphore_cast(&semaphore->async);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_amdxdna_semaphore_destroy(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_amdxdna_semaphore_t* semaphore =
      iree_hal_amdxdna_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_completion_batch_t* native_signal_batch = NULL;
  iree_slim_mutex_lock(&semaphore->mutex);
  native_signal_batch = semaphore->native_signal_batch;
  semaphore->native_signal_batch = NULL;
  semaphore->native_signal_value = 0;
  iree_slim_mutex_unlock(&semaphore->mutex);
  iree_hal_amdxdna_completion_batch_destroy(native_signal_batch);
  iree_slim_mutex_deinitialize(&semaphore->mutex);
  iree_async_semaphore_deinitialize(&semaphore->async);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_amdxdna_semaphore_isa(iree_hal_semaphore_t* semaphore) {
  return iree_hal_resource_is((const iree_hal_resource_t*)semaphore,
                              &iree_hal_amdxdna_semaphore_vtable);
}

void iree_hal_amdxdna_semaphore_record_native_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_amdxdna_completion_batch_t* batch) {
  if (!iree_hal_amdxdna_semaphore_isa(base_semaphore) || !batch) return;
  iree_hal_amdxdna_semaphore_t* semaphore =
      iree_hal_amdxdna_semaphore_cast(base_semaphore);
  iree_hal_amdxdna_completion_batch_t* old_batch = NULL;
  iree_hal_amdxdna_completion_batch_retain(batch);
  iree_slim_mutex_lock(&semaphore->mutex);
  old_batch = semaphore->native_signal_batch;
  semaphore->native_signal_batch = batch;
  semaphore->native_signal_value = value;
  iree_slim_mutex_unlock(&semaphore->mutex);
  iree_hal_amdxdna_completion_batch_destroy(old_batch);
}

void iree_hal_amdxdna_semaphore_clear_native_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_amdxdna_completion_batch_t* batch) {
  if (!iree_hal_amdxdna_semaphore_isa(base_semaphore) || !batch) return;
  iree_hal_amdxdna_semaphore_t* semaphore =
      iree_hal_amdxdna_semaphore_cast(base_semaphore);
  iree_hal_amdxdna_completion_batch_t* old_batch = NULL;
  iree_slim_mutex_lock(&semaphore->mutex);
  if (semaphore->native_signal_batch == batch &&
      semaphore->native_signal_value == value) {
    old_batch = semaphore->native_signal_batch;
    semaphore->native_signal_batch = NULL;
    semaphore->native_signal_value = 0;
  }
  iree_slim_mutex_unlock(&semaphore->mutex);
  iree_hal_amdxdna_completion_batch_destroy(old_batch);
}

static uint64_t iree_hal_amdxdna_semaphore_query(
    iree_async_semaphore_t* base_semaphore) {
  // Same as iree_hal_sync_semaphore_query: failure_status is stored as intptr.
  iree_status_t failure = (iree_status_t)(uintptr_t)iree_atomic_load(
      &base_semaphore->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    return iree_hal_status_as_semaphore_failure(failure);
  }
  return (uint64_t)iree_atomic_load(&base_semaphore->timeline_value,
                                    iree_memory_order_acquire);
}

static iree_status_t iree_hal_amdxdna_semaphore_signal(
    iree_async_semaphore_t* base_semaphore, uint64_t new_value,
    const iree_async_frontier_t* frontier) {
  iree_status_t status = iree_async_semaphore_advance_timeline(
      base_semaphore, new_value, frontier);
  if (!iree_status_is_ok(status)) return status;
  iree_async_semaphore_dispatch_timepoints(base_semaphore, new_value);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout, iree_async_wait_flags_t flags) {
  iree_hal_amdxdna_semaphore_t* semaphore =
      iree_hal_amdxdna_semaphore_cast(base_semaphore);
  uint64_t current = iree_async_semaphore_query(&semaphore->async);
  if (current >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
    return iree_hal_semaphore_failure_as_status(current);
  }
  if (current >= value) {
    return iree_ok_status();
  }

  iree_hal_amdxdna_completion_batch_t* native_signal_batch = NULL;
  if (iree_timeout_is_infinite(timeout)) {
    native_signal_batch =
        iree_hal_amdxdna_semaphore_try_retain_native_signal(semaphore, value);
  }
  if (native_signal_batch) {
    iree_status_t status = iree_hal_amdxdna_completion_batch_wait(
        native_signal_batch, timeout, flags);
    iree_hal_amdxdna_completion_batch_destroy(native_signal_batch);
    if (!iree_status_is_ok(status)) {
      return status;
    }

    current = iree_async_semaphore_query(&semaphore->async);
    if (current >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
      return iree_hal_semaphore_failure_as_status(current);
    }
    if (current >= value) {
      return iree_ok_status();
    }
  }

  return iree_async_semaphore_multi_wait(
      IREE_ASYNC_WAIT_MODE_ALL, (iree_async_semaphore_t**)&base_semaphore,
      &value, 1, timeout, flags, iree_allocator_system());
}

static iree_status_t iree_hal_amdxdna_semaphore_import_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_t external_timepoint) {
  (void)base_semaphore;
  (void)value;
  (void)queue_affinity;
  (void)external_timepoint;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint import is not yet implemented");
}

static iree_status_t iree_hal_amdxdna_semaphore_export_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_type_t requested_type,
    iree_hal_external_timepoint_flags_t requested_flags,
    iree_hal_external_timepoint_t* IREE_RESTRICT out_external_timepoint) {
  (void)base_semaphore;
  (void)value;
  (void)queue_affinity;
  (void)requested_type;
  (void)requested_flags;
  (void)out_external_timepoint;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint export is not yet implemented");
}

static const iree_hal_semaphore_vtable_t iree_hal_amdxdna_semaphore_vtable = {
    .async =
        {
            .destroy = iree_hal_amdxdna_semaphore_destroy,
            .query = iree_hal_amdxdna_semaphore_query,
            .signal = iree_hal_amdxdna_semaphore_signal,
            .on_fail = NULL,
        },
    .wait = iree_hal_amdxdna_semaphore_wait,
    .import_timepoint = iree_hal_amdxdna_semaphore_import_timepoint,
    .export_timepoint = iree_hal_amdxdna_semaphore_export_timepoint,
};
