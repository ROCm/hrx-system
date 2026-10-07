// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/semaphore.h"

#include <string.h>

typedef struct iree_hal_amd_xdna_semaphore_t {
  // Timeline, failure, frontier, and asynchronous wait state.
  iree_async_semaphore_t async;

  // Allocator owning the semaphore and its inline frontier storage.
  iree_allocator_t host_allocator;

  // Exact creating device identity, borrowed for type discrimination.
  iree_hal_device_t* device;

  // Creation flags controlling synchronization behavior.
  iree_hal_semaphore_flags_t flags;

  // Complete set of queue families that may use the semaphore.
  iree_hal_queue_family_affinity_t queue_family_affinity;

  // Seqlock-protected metadata for the most recently submitted signal.
  iree_hal_submitted_signal_t submitted_signal;
} iree_hal_amd_xdna_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_amd_xdna_semaphore_vtable;

static iree_hal_amd_xdna_semaphore_t* iree_hal_amd_xdna_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amd_xdna_semaphore_vtable);
  return (iree_hal_amd_xdna_semaphore_t*)base_value;
}

iree_status_t iree_hal_amd_xdna_semaphore_create(
    iree_hal_device_t* device, iree_async_proactor_t* proactor,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amd_xdna_semaphore_t* semaphore = NULL;
  iree_host_size_t frontier_offset = 0;
  iree_host_size_t total_size = 0;
  iree_status_t status = iree_async_semaphore_layout(
      sizeof(*semaphore), IREE_HAL_AMD_XDNA_FRONTIER_CAPACITY, &frontier_offset,
      &total_size);
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&semaphore);
  }
  if (iree_status_is_ok(status)) {
    iree_async_semaphore_initialize(
        (const iree_async_semaphore_vtable_t*)&iree_hal_amd_xdna_semaphore_vtable,
        proactor, initial_value, frontier_offset,
        IREE_HAL_AMD_XDNA_FRONTIER_CAPACITY, &semaphore->async);
    semaphore->host_allocator = host_allocator;
    semaphore->device = device;
    semaphore->flags = flags;
    semaphore->queue_family_affinity = queue_family_affinity;
    memset(&semaphore->submitted_signal, 0,
           sizeof(semaphore->submitted_signal));
    *out_semaphore = iree_hal_semaphore_cast(&semaphore->async);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amd_xdna_semaphore_destroy(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_amd_xdna_semaphore_t* semaphore =
      iree_hal_amd_xdna_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_semaphore_deinitialize(&semaphore->async);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_amd_xdna_semaphore_isa(iree_hal_semaphore_t* semaphore) {
  return iree_hal_resource_is((const iree_hal_resource_t*)semaphore,
                              &iree_hal_amd_xdna_semaphore_vtable);
}

bool iree_hal_amd_xdna_semaphore_is_local(iree_hal_semaphore_t* semaphore,
                                          const iree_hal_device_t* device) {
  return iree_hal_amd_xdna_semaphore_isa(semaphore) &&
         ((const iree_hal_amd_xdna_semaphore_t*)semaphore)->device == device;
}

iree_hal_semaphore_flags_t iree_hal_amd_xdna_semaphore_flags(
    iree_hal_semaphore_t* semaphore) {
  return iree_hal_amd_xdna_semaphore_cast(semaphore)->flags;
}

iree_hal_queue_family_affinity_t
iree_hal_amd_xdna_semaphore_queue_family_affinity(
    iree_hal_semaphore_t* semaphore) {
  return iree_hal_amd_xdna_semaphore_cast(semaphore)->queue_family_affinity;
}

iree_hal_submitted_signal_t* iree_hal_amd_xdna_semaphore_submitted_signal(
    iree_hal_semaphore_t* semaphore) {
  return &iree_hal_amd_xdna_semaphore_cast(semaphore)->submitted_signal;
}

bool iree_hal_amd_xdna_semaphore_publish_signal(
    iree_hal_semaphore_t* base_semaphore, iree_async_axis_t producer_axis,
    const iree_async_frontier_t* producer_frontier,
    bool producer_frontier_exact, uint64_t producer_epoch,
    uint64_t producer_value) {
  IREE_ASSERT_ARGUMENT(producer_frontier);
  iree_hal_amd_xdna_semaphore_t* semaphore =
      iree_hal_amd_xdna_semaphore_cast(base_semaphore);

  iree_hal_submitted_signal_flags_t flags =
      IREE_HAL_SUBMITTED_SIGNAL_FLAG_VALID;
  bool source_dominates_frontier = false;
  iree_slim_mutex_lock(&semaphore->async.mutex);
  const bool merged = iree_async_frontier_merge_and_test_source_dominance(
      semaphore->async.frontier, semaphore->async.frontier_capacity,
      producer_frontier, &source_dominates_frontier);
  if (merged && producer_frontier_exact && source_dominates_frontier) {
    flags |= IREE_HAL_SUBMITTED_SIGNAL_FLAG_PRODUCER_FRONTIER_EXACT;
  }
  iree_hal_submitted_signal_store(
      &semaphore->submitted_signal,
      merged ? flags : IREE_HAL_SUBMITTED_SIGNAL_FLAG_NONE,
      merged ? producer_axis : (iree_async_axis_t)0,
      merged ? producer_epoch : 0, merged ? producer_value : 0);
  iree_slim_mutex_unlock(&semaphore->async.mutex);
  return merged;
}

void iree_hal_amd_xdna_semaphore_clear_submitted_signal(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_amd_xdna_semaphore_t* semaphore =
      iree_hal_amd_xdna_semaphore_cast(base_semaphore);
  iree_slim_mutex_lock(&semaphore->async.mutex);
  iree_hal_submitted_signal_store(&semaphore->submitted_signal,
                                  IREE_HAL_SUBMITTED_SIGNAL_FLAG_NONE,
                                  (iree_async_axis_t)0, 0, 0);
  iree_slim_mutex_unlock(&semaphore->async.mutex);
}

static uint64_t iree_hal_amd_xdna_semaphore_query(
    iree_async_semaphore_t* base_semaphore) {
  const iree_status_t failure = (iree_status_t)iree_atomic_load(
      &base_semaphore->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    return iree_hal_status_as_semaphore_failure(failure);
  }
  return (uint64_t)iree_atomic_load(&base_semaphore->timeline_value,
                                    iree_memory_order_acquire);
}

static iree_status_t iree_hal_amd_xdna_semaphore_signal(
    iree_async_semaphore_t* base_semaphore, uint64_t new_value,
    const iree_async_frontier_t* frontier) {
  IREE_RETURN_IF_ERROR(iree_async_semaphore_advance_timeline(
      base_semaphore, new_value, frontier));
  iree_async_semaphore_dispatch_timepoints(base_semaphore, new_value);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout, iree_async_wait_flags_t flags) {
  return iree_async_semaphore_multi_wait(
      IREE_ASYNC_WAIT_MODE_ALL, (iree_async_semaphore_t**)&base_semaphore,
      &value, 1, timeout, flags, iree_allocator_system());
}

static const iree_hal_semaphore_vtable_t iree_hal_amd_xdna_semaphore_vtable = {
    .async =
        {
            .destroy = iree_hal_amd_xdna_semaphore_destroy,
            .query = iree_hal_amd_xdna_semaphore_query,
            .signal = iree_hal_amd_xdna_semaphore_signal,
        },
    .wait = iree_hal_amd_xdna_semaphore_wait,
};
