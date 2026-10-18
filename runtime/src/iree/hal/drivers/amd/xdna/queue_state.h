// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Queue publication, progress and causal state shared with admission.
#ifndef IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_STATE_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_STATE_H_

#include "iree/async/event.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amd/xdna/profile.h"
#include "iree/hal/drivers/amd/xdna/queue.h"
#include "iree/hal/drivers/amd/xdna/queue_operation.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Native FIFO positions, distinct from host operation readiness.
typedef struct iree_hal_amd_xdna_pending_t {
  // Accepted invocation retaining its command, buffers and executable.
  iree_hal_amd_xdna_operation_t* operation;
  // Exclusive invocation storage returned after checked retirement.
  iree_hal_amd_xdna_invocation_t* invocation;
  // Actual opaque point returned by native acceptance.
  uint64_t submission;
  // Native-only causal epoch assigned after acceptance.
  uint64_t epoch;
} iree_hal_amd_xdna_pending_t;

// Intrusive ready work; submissions with unsatisfied waits are not linked here.
typedef struct iree_hal_amd_xdna_ready_list_t {
  // Oldest eligible operation, or NULL.
  iree_hal_amd_xdna_operation_t* head;
  // Newest eligible operation, or NULL.
  iree_hal_amd_xdna_operation_t* tail;
} iree_hal_amd_xdna_ready_list_t;

// Cold persistent-observer lifecycle, serialized by the proactor poll owner.
typedef enum iree_hal_amd_xdna_observer_state_e {
  // No registration handoff has been accepted.
  IREE_HAL_AMD_XDNA_OBSERVER_STATE_UNINITIALIZED = 0,
  // The placed registration handoff is pending or executing.
  IREE_HAL_AMD_XDNA_OBSERVER_STATE_REGISTERING,
  // The persistent source owns the event's borrowed native primitive.
  IREE_HAL_AMD_XDNA_OBSERVER_STATE_REGISTERED,
  // Registration failed without creating a source.
  IREE_HAL_AMD_XDNA_OBSERVER_STATE_UNAVAILABLE,
  // Terminal unregistration owns source retirement.
  IREE_HAL_AMD_XDNA_OBSERVER_STATE_UNREGISTERING,
  // Terminal unregistration returned borrowed event ownership.
  IREE_HAL_AMD_XDNA_OBSERVER_STATE_RETIRED,
  // Terminal unregistration failed and retained native-reachable ownership.
  IREE_HAL_AMD_XDNA_OBSERVER_STATE_RETAINED,
} iree_hal_amd_xdna_observer_state_t;

struct iree_hal_amd_xdna_queue_t {
  // HAL identity and borrowed family.
  iree_hal_queue_t base;
  // Allocator owning the queue and cold native ring storage.
  iree_allocator_t host_allocator;
  // Reusable operation metadata and captured payload blocks.
  iree_hal_amd_xdna_queue_storage_t storage;
  // Session recorder owned by this queue under the idle begin/end contract.
  iree_hal_amd_xdna_profile_t* profile;
  // Monotonically increasing session identity, independent of native epochs.
  uint64_t next_profile_session_id;
  // Borrowed device owner dominating all native children.
  iree_hal_amd_xdna_context_t* context;
  // Borrowed shared completion owner.
  iree_async_proactor_t* proactor;
  // Native queue, released after accepted resources retire.
  amdf_kernel_queue_t* handle;
  // Maximum commands in one native array, bounded by pending ring capacity.
  uint32_t maximum_command_count;
  // Owned native wake destination, live through every requested notification.
  iree_async_event_t* event;
  // Borrowed native descriptor naming event's signal primitive.
  amdf_native_event_t native_event;
  // Persistent proactor source borrowing the event's wait primitive.
  iree_async_event_source_t* event_source;
  // Placed cold registration handoff to the proactor owner.
  iree_async_operation_t observer_registration;
  // Placed first-stage terminal shutdown handoff.
  iree_async_operation_t shutdown_operation;
  // Placed final handoff after terminal source unregistration.
  iree_async_operation_t finalization_operation;
  // Device-owned terminal callback retained through finalization.
  iree_hal_amd_xdna_queue_shutdown_callback_t shutdown_callback;
  // Poll-owned persistent observer lifecycle.
  iree_hal_amd_xdna_observer_state_t observer_state;
  // True when shutdown arrived before observer registration completed.
  bool shutdown_waiting_for_registration;
  // Eligible invocations waiting for native capacity.
  iree_hal_amd_xdna_ready_list_t dispatch_ready;
  // Deferred consumers waiting for the caller-owned direct publisher.
  iree_hal_amd_xdna_ready_list_t acceptance_waiters;
  // Serialized lookup of signals awaiting native acceptance.
  iree_hal_amd_xdna_queue_producer_index_t producer_index;
  // Serializes queue causal state shared by callers and the proactor owner.
  iree_slim_mutex_t state_mutex;
  // Serializes ordered completion publication after pending-ring detachment.
  iree_slim_mutex_t completion_drain_mutex;
  // Dispatch currently inside native preparation/publication, or NULL.
  // Published atomically because a direct caller may own publication while
  // the proactor resolves a dependent operation.
  iree_atomic_intptr_t publisher_operation;
  // Nonzero while one caller or the native service owns publication order.
  iree_atomic_int32_t publisher_claim;
  // Nonzero while the persistent observer permits new native admission.
  iree_atomic_int32_t native_admission_open;
  // Nonzero while a BUSY publication waits for checked native progress.
  iree_atomic_int32_t native_publication_blocked;
  // Potentially blocking native preparation and publication owner.
  iree_hal_amd_xdna_queue_service_t publisher;
  // Potentially blocking memory, transfer, and callback execution owner.
  iree_hal_amd_xdna_queue_service_t host_worker;
  // True after both progress services have been stopped and joined.
  bool services_shutdown;
  // True only while the device may release the queue and its parent graph.
  bool shutdown_complete;
  // Placed native pending ring, sized to the provider's prepared capacity.
  struct {
    // Accepted invocations in native FIFO order.
    iree_hal_amd_xdna_pending_t* entries;
    // Number of placed pending entries.
    uint32_t capacity;
    // Index of the oldest accepted native invocation.
    uint32_t head;
    // Number of accepted invocations awaiting checked retirement.
    uint32_t count;
    // Atomic mirror of |count| for optimistic caller-side capacity checks.
    iree_atomic_int32_t visible_count;
  } pending;
  // True while libamdf owes a wake for the current oldest native point.
  bool notification_pending;
  // Direct-wait ownership and native-event handoff flags.
  iree_atomic_int32_t direct_wait_flags;
  // First native failure, published once and immutable until queue destruction.
  iree_atomic_intptr_t failure_status;
  // Retained topology tracker; NULL before group assignment.
  iree_async_frontier_tracker_t* tracker;
  // Queue axis assigned by its group.
  iree_async_axis_t axis;
  // Last accepted native epoch, never advanced by host work or unresolved
  // waits.
  uint64_t epoch;
  // Complete causal lower bound of the accepted native FIFO.
  iree_hal_amd_xdna_frontier_state_t accepted_frontier;
};

// Borrows the immutable first failure while the caller retains queue/device
// lifetime. Semaphore callbacks can query this while their signal producer
// owns the state mutex; no lock or mutable status annotation is involved.
static inline iree_status_t iree_hal_amd_xdna_queue_failure_status(
    const iree_hal_amd_xdna_queue_t* queue) {
  return (iree_status_t)iree_atomic_load(&queue->failure_status,
                                         iree_memory_order_acquire);
}

// Publishes a driver failure event without consuming a status.
void iree_hal_amd_xdna_queue_report(iree_hal_amd_xdna_queue_t* queue,
                                    iree_status_code_t status_code,
                                    iree_string_view_t message);

// Returns the current publisher under its atomic lifetime handoff. The caller
// holds the queue state mutex while attaching acceptance waiters.
iree_hal_amd_xdna_operation_t* iree_hal_amd_xdna_queue_publisher_operation(
    const iree_hal_amd_xdna_queue_t* queue);

// Starts eligible native work while the queue state mutex is held.
void iree_hal_amd_xdna_queue_pump_dispatch(iree_hal_amd_xdna_queue_t* queue);

// Transfers an admitted operation to its execution owner, or completes a
// rejected/empty operation. The queue state mutex is held.
void iree_hal_amd_xdna_queue_enqueue_ready(
    iree_hal_amd_xdna_operation_t* operation, iree_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_STATE_H_
