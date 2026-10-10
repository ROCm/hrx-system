// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_SERVICE_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_SERVICE_H_

#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_amd_xdna_queue_service_t
    iree_hal_amd_xdna_queue_service_t;

// Intrusive storage carried by each operation crossing a progress service.
typedef struct iree_hal_amd_xdna_queue_service_item_t {
  // Placed handoff returning worker results to the proactor owner.
  iree_async_operation_t handoff;
  // Next item in service admission order, or NULL.
  struct iree_hal_amd_xdna_queue_service_item_t* next;
} iree_hal_amd_xdna_queue_service_item_t;

// Executes one item on the service thread. Results remain in item-owned
// storage.
typedef void (*iree_hal_amd_xdna_queue_service_execute_fn_t)(
    void* user_data, iree_hal_amd_xdna_queue_service_item_t* item);

// Consumes one worker result on the proactor thread. The callback must call
// iree_hal_amd_xdna_queue_service_acknowledge before it releases ownership that
// may dominate |service|. Takes ownership of |status|.
typedef void (*iree_hal_amd_xdna_queue_service_complete_fn_t)(
    void* user_data, iree_hal_amd_xdna_queue_service_t* service,
    iree_hal_amd_xdna_queue_service_item_t* item, iree_status_t status);

// Handles a result that cannot be returned to the proactor. Runs on the service
// thread and takes ownership of |status|. The item and its owner graph must
// remain live because terminal reclamation is no longer serialized.
typedef void (*iree_hal_amd_xdna_queue_service_abandon_fn_t)(
    void* user_data, iree_hal_amd_xdna_queue_service_item_t* item,
    iree_status_t status);

// Immutable callbacks used by one service thread.
typedef struct iree_hal_amd_xdna_queue_service_callbacks_t {
  // Potentially blocking worker function.
  iree_hal_amd_xdna_queue_service_execute_fn_t execute;
  // Proactor-owned result callback.
  iree_hal_amd_xdna_queue_service_complete_fn_t complete;
  // Worker-owned failed-handoff callback.
  iree_hal_amd_xdna_queue_service_abandon_fn_t abandon;
} iree_hal_amd_xdna_queue_service_callbacks_t;

// One cold-created execution owner with ordered proactor acknowledgement.
struct iree_hal_amd_xdna_queue_service_t {
  // Borrowed proactor receiving placed result handoffs.
  iree_async_proactor_t* proactor;
  // Borrowed callback context retained by the service owner.
  void* user_data;
  // Immutable execution and ownership callbacks.
  iree_hal_amd_xdna_queue_service_callbacks_t callbacks;
  // Protects admitted and active item state.
  iree_slim_mutex_t mutex;
  // Wakes the worker for admission, acknowledgement, and shutdown.
  iree_notification_t notification;
  // Set before final worker release or an unreturnable handoff.
  iree_atomic_int32_t stopping;
  // Owned worker thread, or NULL before successful initialization.
  iree_thread_t* thread;
  // Oldest admitted item awaiting execution, or NULL.
  iree_hal_amd_xdna_queue_service_item_t* ready_head;
  // Newest admitted item awaiting execution, or NULL.
  iree_hal_amd_xdna_queue_service_item_t* ready_tail;
  // Item awaiting proactor acknowledgement, or NULL.
  iree_hal_amd_xdna_queue_service_item_t* active;
};

// Starts one progress service. Thread creation is the only steady owner cost.
iree_status_t iree_hal_amd_xdna_queue_service_initialize(
    iree_string_view_t name, iree_async_proactor_t* proactor,
    iree_hal_amd_xdna_queue_service_callbacks_t callbacks, void* user_data,
    iree_allocator_t host_allocator,
    iree_hal_amd_xdna_queue_service_t* out_service);

// Stops and joins an idle service and releases its cold synchronization state.
// The service owner guarantees that no item is admitted or active.
void iree_hal_amd_xdna_queue_service_deinitialize(
    iree_hal_amd_xdna_queue_service_t* service);

// Admits one placed item in call order and wakes the service when idle.
void iree_hal_amd_xdna_queue_service_enqueue(
    iree_hal_amd_xdna_queue_service_t* service,
    iree_hal_amd_xdna_queue_service_item_t* item);

// Acknowledges the active item after all proactor-owned state is current. The
// next admitted item may begin before this function returns.
void iree_hal_amd_xdna_queue_service_acknowledge(
    iree_hal_amd_xdna_queue_service_t* service,
    iree_hal_amd_xdna_queue_service_item_t* item);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_SERVICE_H_
