// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_FRONTIER_TRACKER_H_
#define IREE_ASYNC_FRONTIER_TRACKER_H_

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_semaphore_t iree_async_semaphore_t;
typedef struct iree_async_frontier_tracker_t iree_async_frontier_tracker_t;

//===----------------------------------------------------------------------===//
// Frontier waiter
//===----------------------------------------------------------------------===//

// Callback invoked when a frontier wait is satisfied (or fails).
// |status| is OK if all entries reached their target epochs, or an error if
// any axis failed (e.g., device lost → semaphore failure propagates here).
typedef void (*iree_async_frontier_waiter_fn_t)(void* user_data,
                                                iree_status_t status);

// Caller-owned storage for a pending frontier wait. Indexed by the next pending
// axis and epoch while active. The caller must not modify or free this storage
// until the callback fires.
//
// Usage:
//   iree_async_frontier_waiter_t waiter;
//   iree_async_frontier_tracker_wait(tracker, &frontier, on_ready, ctx,
//   &waiter);
//   // ... waiter is now pending. callback fires when frontier is satisfied.
//   // After callback fires (or cancel returns true), storage is yours again.
typedef struct iree_async_frontier_waiter_t {
  // Next pending waiter, or next detached callback awaiting dispatch.
  struct iree_async_frontier_waiter_t* next;

  // Previous pending waiter, used for constant-time removal on terminal paths.
  struct iree_async_frontier_waiter_t* previous;

  // The frontier being waited on. Must remain valid while the waiter is
  // pending. The tracker reads entries during advance() to check satisfaction.
  const iree_async_frontier_t* frontier;

  // Callback invoked when the frontier becomes ready or failed.
  iree_async_frontier_waiter_fn_t callback;

  // User data passed through to |callback|.
  void* user_data;

  // Scratch field used during dispatch: holds the status to pass to the
  // callback after the waiter has been unlinked from the tracker list.
  // Only valid between unlinking and callback invocation.
  iree_status_t dispatch_status;

  // Tracker-owned ordered index for this waiter's next pending dependency.
  struct {
    // Smaller pending epoch, or previous peer for a duplicate epoch.
    struct iree_async_frontier_waiter_t* left;
    // Larger pending epoch, or next peer for a duplicate epoch.
    struct iree_async_frontier_waiter_t* right;
    // Parent in the axis's intrusive tree, or NULL at its root.
    struct iree_async_frontier_waiter_t* parent;
    // First same-epoch peer of a tree representative, or NULL.
    struct iree_async_frontier_waiter_t* peers;
    // Cached target epoch, avoiding frontier indirection during tree updates.
    uint64_t epoch;
    // Stable tracker table slot owning this index entry.
    uint32_t axis_slot;
    // Frontier entry currently pending; earlier entries have completed.
    uint8_t frontier_ordinal;
    // Tree height, UINT8_MAX for a same-epoch peer, or zero when not pending.
    uint8_t height;
  } index;
} iree_async_frontier_waiter_t;

//===----------------------------------------------------------------------===//
// Frontier tracker
//===----------------------------------------------------------------------===//

// Tracks axis progress and dispatches frontier waiters when satisfied.
//
// The frontier tracker is the runtime component that makes frontiers
// actionable. It answers two questions:
//   1. "Has this frontier been reached yet?" (is_satisfied / wait)
//   2. "An axis just advanced; who should be woken up?" (advance)
//
// ## How it connects to the rest of the system
//
//   Participant completes work
//     → participant timeline signals epoch N
//     → Semaphore timepoint fires
//     → iree_async_frontier_tracker_advance(tracker, axis, N)
//     → Tracker visits waits whose next axis/epoch became ready
//     → Waiters whose frontiers are now fully satisfied get their callbacks
//     → Callbacks submit dependent operations to their scheduler
//
//   Network receives frontier update from remote machine
//     → Protocol decoder reads {axis, epoch} pairs
//     → iree_async_frontier_tracker_advance(tracker, remote_axis, epoch)
//     → Same flow as above; local waiters on remote axes get woken
//
// ## Thread safety
//
//   register_axis(): Thread-safe. Registration may run concurrently with other
//                    registrations and operations on previously registered
//                    axes.
//
//   advance():     Thread-safe. Multiple threads can advance different axes
//                  concurrently. Same axis from multiple threads is safe
//                  (monotonic: max wins, lower values are no-ops).
//
//   wait():        Thread-safe. Can be called from any thread.
//
//   cancel_wait(): Thread-safe. Reports whether the waiter was unlinked before
//                  dispatch. Callers that need to wait for an
//                  already-dispatched callback must use their own completion
//                  marker.
//
// Waiter indexes are protected by a slim mutex. The advance() hot path updates
// its axis and checks that axis's earliest pending epoch before taking the
// lock. Unrelated waits and later epochs require no traversal or locking. Each
// wait is indexed on its first unresolved entry; satisfied prefixes are not
// revisited. Terminal failure handling visits the full pending set so failure
// of any referenced axis is reported even while a different axis is pending.
//
// ## Lifecycle
//
// Per-causal-domain object. Created when a distributed/local execution context
// establishes its axis namespace and destroyed after all participants using
// that namespace have retired. Axis storage has fixed capacity. New axes may be
// registered while other axes are active, such as when acquiring another queue
// on an existing device. Registered identities remain reserved after
// retirement.
//
//   iree_async_frontier_tracker_create(options, allocator, &tracker);
//   // ... register each axis before publishing its participant ...
//   // ... advance()/wait() and new registrations from multiple threads ...
//   iree_async_frontier_tracker_release(tracker);
//
// ## Axis lookup shape
//
// Each axis is registered once, then looked up by advance(), wait(), and
// query_epoch(). The implementation stores them in tracker-owned hash
// slots so steady-state operations do not scan all registered axes:
//
//   ┌───────────────────────────────┐
//   │ iree_async_axis_t             │
//   │ session/machine/domain/ordinal│
//   └──────────────┬────────────────┘
//                  │ hash
//                  ▼
//   ┌──────┬──────┬──────┬──────┬──────┐
//   │ slot │ slot │ slot │ slot │ slot │
//   └──────┴──────┴──────┴──────┴──────┘
//                         │
//                         ▼
//              {epoch, bridge semaphore, failure}
//
// The capacity option below is the maximum number of registered axes. The
// tracker may allocate additional internal slots to keep probes short.
//
// Options for creating a frontier tracker.
typedef struct iree_async_frontier_tracker_options_t {
  // Maximum number of axes that may be registered for the causal domain.
  uint32_t axis_table_capacity;

  // Generation value for the causal domain. This is embedded into axes to
  // prevent stale axis IDs from a previous execution context from comparing
  // equal to axes in a later context that reused the same machine/domain/
  // ordinal values.
  uint8_t session_epoch;

  // Local machine ordinal within the causal domain. This is embedded into axes
  // so frontiers can distinguish participants with the same domain/ordinal
  // values on different machines.
  uint8_t machine_index;
} iree_async_frontier_tracker_options_t;

static inline iree_async_frontier_tracker_options_t
iree_async_frontier_tracker_options_default(void) {
  iree_async_frontier_tracker_options_t options;
  memset(&options, 0, sizeof(options));
  options.axis_table_capacity = 256;
  options.session_epoch = 1;
  options.machine_index = 0;
  return options;
}

// Creates a ref-counted frontier tracker with tracker-owned axis storage.
iree_status_t iree_async_frontier_tracker_create(
    iree_async_frontier_tracker_options_t options, iree_allocator_t allocator,
    iree_async_frontier_tracker_t** out_tracker);

// Retains the given |tracker| for the caller.
void iree_async_frontier_tracker_retain(iree_async_frontier_tracker_t* tracker);

// Releases the given |tracker| from the caller. When the final reference is
// released, all pending waiters are cancelled with IREE_STATUS_CANCELLED.
void iree_async_frontier_tracker_release(
    iree_async_frontier_tracker_t* tracker);

// Returns the local session epoch used for deterministic axis construction.
uint8_t iree_async_frontier_tracker_session_epoch(
    const iree_async_frontier_tracker_t* tracker);

// Returns the local machine index used for deterministic axis construction.
uint8_t iree_async_frontier_tracker_machine_index(
    const iree_async_frontier_tracker_t* tracker);

// Registers an axis for the lifetime of the tracker. Axis IDs are never reused:
// once registered, an axis remains reserved even after retirement.
//
// Thread-safe with other registrations and operations on existing axes. The
// caller must finish registration before publishing the axis to its users.
// For a new ID with available capacity, exactly one concurrent registration
// succeeds; the others return ALREADY_EXISTS and consume no additional entry.
//
// |semaphore| is a borrowed bridge pointer. The registering participant must
// retire the axis before destroying the borrowed semaphore. The tracker owns
// bridge signaling; other producers must not independently signal it.
iree_status_t iree_async_frontier_tracker_register_axis(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    iree_async_semaphore_t* semaphore);

// Retires an axis and clears any borrowed bridge semaphore. This marks the
// axis permanently failed/cancelled, dispatches pending waiters that reference
// it, and causes future waits on the axis to fail immediately.
//
// The participant owning the axis must quiesce all concurrent calls to
// iree_async_frontier_tracker_advance() for that axis before retiring it.
// Takes ownership of |status|, which must be a non-OK status describing why
// the axis is being retired (e.g. CANCELLED for clean teardown).
void iree_async_frontier_tracker_retire_axis(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    iree_status_t status);

// Returns true if |axis| has reached at least |epoch|. Unknown axes return
// false. This is a lock-free progress query for pool/death-frontier dominance;
// it does not treat axis failure as success.
bool iree_async_frontier_tracker_query_epoch(
    const iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    uint64_t epoch);

// Queries whether every entry in |frontier| has reached its epoch without
// registering a waiter. An empty frontier is satisfied. Pending work returns
// OK with |out_satisfied| false; any failed axis propagates its terminal error
// even if another axis is pending. Unknown axes return NOT_FOUND. On error
// |out_satisfied| is unchanged.
//
// Takes the tracker's metadata lock to observe axis failures consistently.
// No callback storage, notification, or completion wait is created.
iree_status_t iree_async_frontier_tracker_query(
    iree_async_frontier_tracker_t* tracker,
    const iree_async_frontier_t* frontier, bool* out_satisfied);

// Advances an axis to a new epoch value. Thread-safe.
//
// If |epoch| is greater than the axis's current epoch, the axis is updated
// and any waiters whose frontiers are now fully satisfied are dispatched
// (callbacks fire from within this call, on the calling thread).
//
// If |epoch| is less than or equal to the current epoch, this is a no-op
// (monotonic timelines never go backwards).
//
// If the axis has an associated semaphore, the semaphore is also signaled
// to |epoch| (bridging frontier advancement to the proactor's semaphore wait
// infrastructure). Concurrent or reentrant advances may coalesce bridge
// signals: the caller currently publishing to the bridge drains the latest
// epoch while other callers continue without waiting for that publication.
//
// Returns the number of waiters that were dispatched.
iree_host_size_t iree_async_frontier_tracker_advance(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    uint64_t epoch);

// Registers a waiter for a frontier to be satisfied. Thread-safe.
//
// The |callback| fires (from an advance() call) when every entry in |frontier|
// has its axis at or beyond the target epoch. If the frontier is already
// satisfied at the time of this call, the callback fires immediately (before
// this function returns).
//
// |waiter| is caller-owned storage that must remain valid until the callback
// fires or the wait is cancelled. |frontier| must also remain valid for the
// same duration.
//
// Returns IREE_STATUS_NOT_FOUND if any axis in the frontier is not in the
// tracker's axis table. Registration must complete before waiting on an axis.
iree_status_t iree_async_frontier_tracker_wait(
    iree_async_frontier_tracker_t* tracker,
    const iree_async_frontier_t* frontier,
    iree_async_frontier_waiter_fn_t callback, void* user_data,
    iree_async_frontier_waiter_t* waiter);

// Cancels a pending wait.
//
// |waiter| must have been successfully registered with this tracker. Returns
// true if it was pending and removed, meaning the callback will not fire.
// Returns false after cancellation or dispatch; a dispatched callback may
// still be executing. Caller-owned storage must remain valid through this call.
bool iree_async_frontier_tracker_cancel_wait(
    iree_async_frontier_tracker_t* tracker,
    iree_async_frontier_waiter_t* waiter);

// Fails an axis permanently. All pending waiters that reference this axis are
// dispatched with |status|. Future waits on this axis will fail immediately.
// Takes ownership of |status|, which must be a non-OK status describing the
// failure. This API is explicitly for the error path, never for clean
// dispositions.
//
// This propagates device-lost and similar fatal errors through the frontier
// system: a failed GPU queue axis causes all dependent operations to fail,
// regardless of which machine they're on.
void iree_async_frontier_tracker_fail_axis(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    iree_status_t status);

//===----------------------------------------------------------------------===//
// Logical joins
//===----------------------------------------------------------------------===//

typedef struct iree_async_frontier_timeline_t iree_async_frontier_timeline_t;
typedef struct iree_async_frontier_join_wait_t iree_async_frontier_join_wait_t;

// Acquires a retained logical timeline. Healthy idle timelines reuse their
// registered axis and continue its monotonically increasing epochs; completed
// points remain valid across leases. Failed timelines are never reused.
// The tracker owns allocation and identity storage. No thread or semaphore is
// created. Exhaustion concerns concurrently live or failed timelines, not the
// number of successful acquire/release cycles.
iree_status_t iree_async_frontier_tracker_acquire_timeline(
    iree_async_frontier_tracker_t* tracker,
    iree_async_frontier_timeline_t** out_timeline);
void iree_async_frontier_timeline_retain(
    iree_async_frontier_timeline_t* timeline);
void iree_async_frontier_timeline_release(
    iree_async_frontier_timeline_t* timeline);

// Presecured mutable capture of registered-axis dependencies. Storage is sized
// to the tracker capacity, independently of the compact public frontier width.
// Only |next| is caller-owned; the remaining fields belong to the join API.
// The owner serializes begin/merge with every other capture on its timeline.
// A returned point may escape only after its input set has become immutable.
typedef struct iree_async_frontier_join_t {
  // Caller-owned linkage before publish, allowing allocation-free reserve
  // lists.
  struct iree_async_frontier_join_t* next;
  // Retained timeline lease, independent of the caller's metadata lifetime.
  iree_async_frontier_timeline_t* timeline;
  // Published proof point, or epoch zero before begin.
  iree_async_frontier_entry_t point;
  // Sorted unique input entries in the trailing storage before publication.
  iree_async_frontier_entry_t* entries;
  // Number of captured inputs, including the preceding timeline point.
  uint32_t entry_count;
  // Maximum distinct inputs established by the tracker capacity.
  uint32_t entry_capacity;
  // Presecured ordinary frontier wait records in the trailing storage.
  iree_async_frontier_join_wait_t* waits;
  // Number of wait records being published.
  uint32_t wait_count;
  // Submission and wait references preserving all callback storage.
  iree_atomic_int32_t pending_count;
  // First owned input failure, or zero while healthy.
  iree_atomic_intptr_t failure_status;
  // Set after every wait has either registered or completed inline.
  iree_atomic_int32_t published;
  // Zero before failure, one when requested, two when cancellation is owned.
  iree_atomic_int32_t cancellation;
} iree_async_frontier_join_t;

// Allocates capture storage using the tracker's allocator and pins its lease.
// The causal domain outlives unpublished captures; domain destruction cancels
// published waits and reclaims them without an internal reference cycle.
// This is the fallible preparation step; begin/merge/publish never allocate.
iree_status_t iree_async_frontier_join_allocate(
    iree_async_frontier_timeline_t* timeline,
    iree_async_frontier_join_t** out_join);
// Releases an unused capture that has never begun.
void iree_async_frontier_join_free(iree_async_frontier_join_t* join);
// Begins one immutable-in-order proof and includes its preceding timeline
// point.
iree_async_frontier_entry_t iree_async_frontier_join_begin(
    iree_async_frontier_join_t* join);
// Captures epoch maxima from a well-formed registered-axis frontier. References
// to this open proof are already represented by its inputs and are ignored.
void iree_async_frontier_join_merge(iree_async_frontier_join_t* join,
                                    const iree_async_frontier_t* frontier);
// Consumes the sealed capture. Exact waits advance/fail its point inline and
// reclaim storage after their final callback. Must run outside caller locks.
void iree_async_frontier_join_publish(iree_async_frontier_join_t* join);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_FRONTIER_TRACKER_H_
