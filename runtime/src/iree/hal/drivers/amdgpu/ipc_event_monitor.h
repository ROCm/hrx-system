// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_IPC_EVENT_MONITOR_H_
#define IREE_HAL_DRIVERS_AMDGPU_IPC_EVENT_MONITOR_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_ipc_event_monitor_operation_t
    iree_hal_amdgpu_ipc_event_monitor_operation_t;

// First retry delay for operations that require periodic polling.
#define IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_INITIAL_POLL_DELAY_NS 100000

// Maximum delay between polls of an external IPC signal.
#define IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_EXTERNAL_MAX_POLL_DELAY_NS 1000000

// Polls one committed IPC event operation on the monitor worker. Returns true
// after resolving, cancelling, or abandoning the operation. A completed poll
// may release all resources and free the storage containing |operation|. Every
// poll samples normal terminal state before allowing |is_shutting_down| to
// override a still-pending outcome. A shutdown poll must not keep waiting on a
// condition that can remain pending indefinitely. |was_requested| is true when
// an external callback explicitly requested this poll and establishes acquire
// visibility of everything preceding that request.
// Returning false from a shutdown poll is permitted only when a callback has
// already taken ownership and is guaranteed to request another poll.
typedef bool (*iree_hal_amdgpu_ipc_event_monitor_poll_fn_t)(
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation,
    bool is_shutting_down, bool was_requested);

// Intrusive operation transferred to the process-wide IPC event monitor after
// its associated queue transaction commits.
struct iree_hal_amdgpu_ipc_event_monitor_operation_t {
  // Intrusive link used only on the publication MPSC stack.
  iree_hal_amdgpu_ipc_event_monitor_operation_t* pending_next;
  // Next operation in a scheduler list, protected by the schedule mutex.
  iree_hal_amdgpu_ipc_event_monitor_operation_t* schedule_next;
  // Previous operation in a scheduler list, protected by the schedule mutex.
  iree_hal_amdgpu_ipc_event_monitor_operation_t* schedule_previous;
  // Internal lifecycle state asserting the single ownership transfer.
  iree_atomic_int32_t state;
  // Set by a callback to request an immediate worker poll.
  iree_atomic_int32_t poll_requested;
  // Maximum periodic poll delay, or IREE_DURATION_INFINITE for notification-
  // driven operations. Immutable after initialization.
  iree_duration_t maximum_poll_delay_ns;
  // Current retry delay, owned by the monitor worker after publication.
  iree_duration_t current_poll_delay_ns;
  // Absolute deadline for the next periodic poll, owned by the worker.
  iree_time_t next_poll_deadline_ns;
  // Scheduler list identity, protected by the schedule mutex.
  uint8_t schedule_kind;
  // True after the monitor has invoked |poll| at least once.
  bool has_been_polled;
  // Type-specific polling and shutdown-settlement routine.
  iree_hal_amdgpu_ipc_event_monitor_poll_fn_t poll;
};

// Initializes an operation before admission or queue visibility.
void iree_hal_amdgpu_ipc_event_monitor_operation_initialize(
    iree_hal_amdgpu_ipc_event_monitor_poll_fn_t poll,
    iree_duration_t maximum_poll_delay_ns,
    iree_hal_amdgpu_ipc_event_monitor_operation_t* out_operation);

// Admits |operation| and lazily starts the process-wide monitor. Admission is
// the only fallible monitor operation and must complete before a queue can
// accept work whose completion depends on the operation.
iree_status_t iree_hal_amdgpu_ipc_event_monitor_operation_admit(
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation);

// Transfers a committed operation to the monitor. Publication allocates no
// memory. It is the caller's final operation access; exactly one publication
// or pre-commit cancellation consumes each admission.
void iree_hal_amdgpu_ipc_event_monitor_operation_publish(
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation);

// Requests an immediate poll from an external callback. The request's final
// locked publication is the caller's final access to |operation|; once it
// completes the monitor may poll and free the containing allocation. Safe
// before or after transaction publication.
void iree_hal_amdgpu_ipc_event_monitor_operation_request_poll(
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation);

// Waits until an external callback requests a poll or |timeout| expires.
// Used only while a transaction still owns an unpublished operation whose
// callback could not be cancelled.
bool iree_hal_amdgpu_ipc_event_monitor_operation_await_poll_request(
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation,
    iree_timeout_t timeout);

// Cancels an admitted operation before it is published. The caller retains
// ownership and may free the containing allocation after this returns.
void iree_hal_amdgpu_ipc_event_monitor_operation_cancel(
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation);

// Waits until monitor shutdown is requested or |timeout| expires. Returns true
// when admission has closed. Admitted preparation paths use this to make local
// waits cancellable without manufacturing transport completion.
bool iree_hal_amdgpu_ipc_event_monitor_await_shutdown(iree_timeout_t timeout);

// Closes admission, asks every committed operation to take its type-specific
// shutdown path, and joins the monitor. Operations admitted before closure
// must still be committed or aborted by their transaction owners; shutdown
// waits for that handoff. Callers must first stop new API transaction
// preparation, then call this before releasing any context or device backing
// an admitted operation. A later admission may restart the monitor. This must
// not be called from the monitor worker.
void iree_hal_amdgpu_ipc_event_monitor_shutdown(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_IPC_EVENT_MONITOR_H_
