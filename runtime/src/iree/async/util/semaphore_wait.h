// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared lifecycle tracking for asynchronous semaphore wait operations.

#ifndef IREE_ASYNC_UTIL_SEMAPHORE_WAIT_H_
#define IREE_ASYNC_UTIL_SEMAPHORE_WAIT_H_

#include "iree/async/operations/semaphore.h"
#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomic_slist.h"
#include "iree/base/threading/mutex.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_semaphore_wait_tracker_t
    iree_async_semaphore_wait_tracker_t;

// Serializes the association between a semaphore wait operation and its
// tracker. One context is embedded in each proactor and shared by its waits.
typedef struct iree_async_semaphore_wait_context_t {
  // Guards operation tracker association, cancellation lookup, and detachment.
  iree_slim_mutex_t association_mutex;
} iree_async_semaphore_wait_context_t;

// Initializes |out_context| for use by a proactor.
void iree_async_semaphore_wait_context_initialize(
    iree_async_semaphore_wait_context_t* out_context);

// Deinitializes |context| after all of its wait operations have completed.
void iree_async_semaphore_wait_context_deinitialize(
    iree_async_semaphore_wait_context_t* context);

// Transfers a terminal semaphore wait tracker to its consumer. The callback
// takes ownership of |entry| and may process it inline or enqueue it for later
// processing. It must use try_prepare_completion before finalizing the tracker.
// Successful preparation joins the producer's final access. Inline consumers
// may then finalize and reuse or release their enclosing operation.
typedef void (*iree_async_semaphore_wait_enqueue_fn_t)(
    void* user_data, iree_atomic_slist_entry_t* entry);

// Bundles the backend enqueue callback and its context.
typedef struct iree_async_semaphore_wait_enqueue_callback_t {
  // Callback invoked when the tracker is ready for terminal processing.
  iree_async_semaphore_wait_enqueue_fn_t fn;

  // Backend-defined context passed to |fn|.
  void* user_data;
} iree_async_semaphore_wait_enqueue_callback_t;

// Detached state returned when a wait tracker is finalized.
typedef struct iree_async_semaphore_wait_completion_t {
  // Completed wait operation with its tracker backpointer cleared.
  iree_async_semaphore_wait_operation_t* operation;

  // LINKED continuation chain transferred from the tracker, if present.
  iree_async_operation_t* continuation_head;

  // Owned terminal status to transfer to the operation callback.
  iree_status_t status;
} iree_async_semaphore_wait_completion_t;

// Returns the exact allocation size for a tracker of |count| timepoints.
// Tracker allocation uses one contiguous region with normal allocator
// alignment, allowing callers to provide fixed storage through an inline arena
// allocator. Returns OUT_OF_RANGE if the count or allocation size cannot be
// represented.
iree_status_t iree_async_semaphore_wait_tracker_storage_size(
    iree_host_size_t count, iree_host_size_t* out_size);

// Allocates and initializes a tracker for |operation|. On success, associates
// the tracker through |context|, retains each referenced semaphore, and
// transfers the operation's LINKED continuation to it. Registration has not
// begun when this returns.
iree_status_t iree_async_semaphore_wait_tracker_allocate(
    iree_async_semaphore_wait_context_t* context,
    iree_async_semaphore_wait_operation_t* operation,
    iree_async_semaphore_wait_enqueue_callback_t enqueue_callback,
    iree_allocator_t allocator,
    iree_async_semaphore_wait_tracker_t** out_tracker);

// Registers all timepoints represented by |tracker|. Synchronous timepoint
// callbacks may run from within this call. Registration failures become the
// operation's asynchronous terminal status so callbacks already in flight can
// quiesce through the normal completion path.
void iree_async_semaphore_wait_tracker_register_timepoints(
    iree_async_semaphore_wait_tracker_t* tracker);

// Requests terminal cancellation for |operation| if it still has a tracker
// associated with |context|. The first terminal event wins against semaphore
// satisfaction and failure. The caller must retain |operation| through this
// call, as required by the proactor cancellation contract.
void iree_async_semaphore_wait_context_request_cancellation(
    iree_async_semaphore_wait_context_t* context,
    iree_async_semaphore_wait_operation_t* operation);

// Cancels timepoints still owned by semaphores and tests whether every
// detached callback has returned. Returns true when the caller may finalize
// the tracker. Success also detaches the operation from cancellation,
// preventing new borrowers before finalization. A false result transfers the
// tracker back to its enqueue path; the caller must make no further accesses to
// it.
bool iree_async_semaphore_wait_tracker_try_prepare_completion(
    iree_async_semaphore_wait_tracker_t* tracker);

// Returns the detached operation, status, and continuation; releases the
// retained semaphores; and frees |tracker|.
// Must only be called after try_prepare_completion returns true.
void iree_async_semaphore_wait_tracker_finalize(
    iree_async_semaphore_wait_tracker_t* tracker,
    iree_async_semaphore_wait_completion_t* out_completion);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_UTIL_SEMAPHORE_WAIT_H_
