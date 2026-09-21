// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lightweight async signaling primitive.
//
// An iree_async_event_t provides cross-thread signaling that integrates with
// the proactor's event loop. Events are the building block for proactor wake(),
// cross-thread notification, and user-facing synchronization.
//
// Semantics: set() makes the event signaled (thread-safe, may be called from
// any context). The proactor detects the signaled state during poll() and
// delivers the completion. One-shot event waits consume their signal before
// completion; persistent source callbacks call consume() before inspecting
// the state whose change the event announces.
//
// Platform mapping:
//   Linux:   eventfd (write to signal, read to reset)
//   macOS:   pipe (write end signals, read end monitored by kqueue)
//   Windows: auto-reset Win32 event (SetEvent to signal)

#ifndef IREE_ASYNC_EVENT_H_
#define IREE_ASYNC_EVENT_H_

#include "iree/async/primitive.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_event_pool_t iree_async_event_pool_t;
typedef struct iree_async_proactor_t iree_async_proactor_t;

//===----------------------------------------------------------------------===//
// Native event storage
//===----------------------------------------------------------------------===//

// Owned native wait/signal resources, independent of any proactor. No
// allocation or reference count is required for the container. Keep it alive
// until every borrower and concurrent signal call has retired; copying it does
// not duplicate ownership. A zero-initialized instance is empty and may be
// deinitialized.
//
// Linux/Android use one coalescing eventfd; Windows uses one auto-reset Event.
// Both primitive fields refer to the same owned handle on these platforms.
// macOS/BSD use a pipe pair. Retaining both pipe ends keeps signaling valid
// even after a peer closes its handles; an unpolled read end is a lifetime
// guard, not a second consumer. Only one consumer may drain each native event.
//
// Shared notifications can borrow these primitives while a separate owner
// retains them beyond notification/proactor teardown. Setting the native event
// does not advance a notification epoch or wake synchronous address waiters.
typedef struct iree_async_event_native_t {
  // Borrowed by the single native polling/draining consumer.
  iree_async_primitive_t wait_primitive;
  // Written by set(); may alias wait_primitive's owned handle.
  iree_async_primitive_t signal_primitive;
} iree_async_event_native_t;

// Creates an initially unsignaled native event with nonblocking,
// noninheritable handles. On failure, leaves |out_event| empty.
IREE_API_EXPORT iree_status_t
iree_async_event_native_initialize(iree_async_event_native_t* out_event);

// Closes the owned handle(s) exactly once and leaves |event| empty. All native
// waits and signal calls must have retired before deinitialization.
IREE_API_EXPORT void iree_async_event_native_deinitialize(
    iree_async_event_native_t* event);

// Signals an initialized native event without accessing a proactor. Thread-safe
// against other signal calls, but not deinitialization. Multiple signals may
// coalesce; saturation of a nonblocking native counter/pipe is already signaled
// and needs no additional wake. Publication is infallible and does not
// acknowledge observer progress. The owner must retain the native resources
// until every concurrent signal call and borrowed wait has retired.
IREE_API_EXPORT void iree_async_event_native_set(
    const iree_async_event_native_t* event);

// Consumes a delivered readiness hint in a persistent event-source callback.
//
// Register event->wait_primitive with
// iree_async_proactor_register_event_source(), then call this at the start of
// each callback, before inspecting authoritative state and requesting another
// notification from an external producer. The event must remain live through
// terminal source unregistration. Use one serialized consumer; do not
// concurrently submit EVENT_WAIT operations on the same event.
//
// Linux drains the nonblocking eventfd counter with one read; pipe-backed
// platforms drain available bytes. Already-drained readiness is successful.
// Windows performs no work: the native wait already consumed the delivered
// auto-reset signal, and resetting again could clear a newer signal. Signals
// may coalesce during consumption; the caller must check authoritative state
// rather than infer progress from a callback or a count.
//
// This is not a general event reset or cancellation operation. It does not
// consume a Windows event outside a delivered callback, suppress queued
// callbacks, or change the persistent registration. EVENT_WAIT operations
// already consume their signal and do not require this call.
//
// No allocation, locking, lazy initialization, or blocking wait occurs. POSIX
// read failures propagate; an empty nonblocking read is ordinary success.
IREE_API_EXPORT iree_status_t
iree_async_event_native_consume(const iree_async_event_native_t* event);

//===----------------------------------------------------------------------===//
// Managed event
//===----------------------------------------------------------------------===//

// A signaling primitive bound to one proactor for asynchronous waits. Created
// via iree_async_event_create(); the proactor must outlive the managed event.
typedef struct iree_async_event_t {
  // References to this managed event, independent of pool ownership.
  iree_atomic_ref_count_t ref_count;

  // The proactor this event is bound to. Not retained.
  iree_async_proactor_t* proactor;

  // Native resources owned until this managed event is destroyed.
  iree_async_event_native_t native;

  // io_uring fixed file index (-1 if not registered).
  int32_t fixed_file_index;

  // Buffer for linked READ operations that drain the eventfd.
  // Used by io_uring's linked POLL_ADD + READ to auto-reset the event.
  uint64_t drain_buffer;

  // Pool support: home pool for release routing (NULL if not pooled).
  iree_async_event_pool_t* pool;

  // Pool support: intrusive list linkage for acquire_stack and return_stack.
  // This field is used for the LIFO stacks that manage available events.
  struct iree_async_event_t* pool_next;

  // Pool support: intrusive list linkage for all_events cleanup list.
  // This separate field tracks ALL events ever created by the pool,
  // independent of their current stack location, for cleanup during deinit.
  struct iree_async_event_t* pool_all_next;
} iree_async_event_t;

// Creates a new event for cross-thread signaling.
//
// Events are lightweight, waitable objects that can be signaled from any
// thread. Use iree_async_event_wait_operation_t to wait on them asynchronously.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Implementation:
//   io_uring: eventfd (IORING_OP_POLL_ADD for waits)
//   IOCP: Event object (SetEvent/WaitForSingleObject)
//   kqueue: pipe + EVFILT_READ
//   generic: eventfd or pipe + poll
//
// Note: For cross-platform notification semantics with richer features,
// consider using iree_async_notification_t (when available) which provides
// a higher-level abstraction over events and futexes.
//
// Returns:
//   IREE_STATUS_OK: Event created successfully.
//   IREE_STATUS_RESOURCE_EXHAUSTED: System resource limit reached.
IREE_API_EXPORT iree_status_t iree_async_event_create(
    iree_async_proactor_t* proactor, iree_async_event_t** out_event);

// Increments the reference count.
IREE_API_EXPORT void iree_async_event_retain(iree_async_event_t* event);

// Decrements the reference count and destroys if it reaches zero.
IREE_API_EXPORT void iree_async_event_release(iree_async_event_t* event);

// Signals the event. Thread-safe.
// Wakes the proactor's poll() if it is monitoring this event.
// Idempotent: multiple calls before the wait completes are coalesced.
// Publication is infallible; completion is observed through accepted waits.
IREE_API_EXPORT void iree_async_event_set(iree_async_event_t* event);

// Consumes a managed event's delivered persistent-source readiness hint using
// iree_async_event_native_consume(). Register event->native.wait_primitive and
// retain the event through terminal source unregistration. This is not a reset
// or cancellation operation; one-shot EVENT_WAIT operations already consume
// their signal.
IREE_API_EXPORT iree_status_t
iree_async_event_consume(iree_async_event_t* event);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_EVENT_H_
