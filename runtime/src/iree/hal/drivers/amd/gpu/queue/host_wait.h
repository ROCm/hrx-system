// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_HOST_WAIT_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_HOST_WAIT_H_

#include "iree/async/util/semaphore_wait.h"
#include "iree/hal/memory/maintenance.h"
#include "iree/hal/semaphore.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Receives the terminal wait status on the device's service executor after all
// semaphore callbacks have relinquished their storage. Takes ownership of the
// status and may free the enclosing capture. Success permits causal admission
// to resume; it does not establish native memory visibility by itself.
typedef void(IREE_API_PTR* iree_hal_amd_gpu_host_wait_fn_t)(
    void* user_data, iree_status_t status);

// Sidecar for waits that cannot yet be admitted as native dependencies.
// Ready submissions do not allocate or initialize this state. The existing
// async tracker owns timepoint registration, failure, cancellation, and
// callback joining; the existing device service owns finalization and
// resumption.
typedef struct iree_hal_amd_gpu_host_wait_t {
  // Service entry owned by the tracker enqueue/prepare-completion protocol.
  iree_hal_memory_maintenance_entry_t entry;
  // Borrowed thread-backed service, live through the final callback.
  iree_hal_memory_maintenance_t* service;
  // Borrowed queue/device association context, live through finalization.
  iree_async_semaphore_wait_context_t* context;
  // Captured wait descriptor borrowing the submission's stable arrays.
  iree_async_semaphore_wait_operation_t operation;
  // Owned tracker, freed by finalization on the service executor.
  iree_async_semaphore_wait_tracker_t* tracker;
  // Terminal continuation, invoked exactly once after successful acceptance.
  iree_hal_amd_gpu_host_wait_fn_t fn;
  // Borrowed continuation context, live through the terminal callback.
  void* user_data;
} iree_hal_amd_gpu_host_wait_t;

// Registers a nonempty, already captured ALL wait. The tracker retains the
// semaphores; the caller keeps the arrays, sidecar, context, and service live
// through the terminal callback. The service must be thread-backed, so cleanup
// never runs inline on a signaling thread. No additional worker is created.
//
// Failure returns before acceptance, retains nothing, and invokes no callback.
// Success transfers storage to the callback, which may run and free it before
// this call returns. All enclosing capture state must be initialized first;
// the caller makes no further access after successful acceptance. Allocation
// uses host_allocator only for the tracker and its timepoints. Fixed-capacity
// callers may supply an inline arena sized by the tracker's storage-size query;
// finalization frees that allocation before invoking the terminal callback.
iree_status_t iree_hal_amd_gpu_host_wait_begin(
    iree_async_semaphore_wait_context_t* context,
    iree_hal_memory_maintenance_t* service, iree_hal_semaphore_list_t waits,
    iree_hal_amd_gpu_host_wait_fn_t fn, void* user_data,
    iree_allocator_t host_allocator, iree_hal_amd_gpu_host_wait_t* wait);

// Requests terminal cancellation, racing satisfaction/failure under the async
// tracker's first-terminal-event contract. The callback still runs on service
// after every detached semaphore callback has joined. The caller must keep
// |wait| alive through this call, independently of its terminal callback, and
// must have joined begin before cancelling. This does not synchronously wait.
void iree_hal_amd_gpu_host_wait_cancel(iree_hal_amd_gpu_host_wait_t* wait);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_HOST_WAIT_H_
