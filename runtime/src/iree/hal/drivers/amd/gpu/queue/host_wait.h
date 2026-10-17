// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_HOST_WAIT_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_HOST_WAIT_H_

#include "iree/async/util/semaphore_wait.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/semaphore.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Receives the terminal wait status on the last
// registering/signaling/cancelling thread after all tracker borrowers
// relinquish their storage. Takes ownership of the status and may free the
// enclosing capture. Success permits causal admission to resume; it does not
// establish native memory visibility itself.
typedef void(IREE_API_PTR* iree_hal_amd_gpu_host_wait_fn_t)(
    void* user_data, iree_status_t status);

// Sidecar for waits that cannot yet be admitted as native dependencies.
// Ready submissions do not allocate or initialize this state. The existing
// async tracker owns timepoint registration, failure, cancellation, and
// callback joining. Finalization and resumption require no executor handoff.
typedef struct iree_hal_amd_gpu_host_wait_t {
  // Borrowed queue/device association context, live through finalization.
  iree_async_semaphore_wait_context_t* context;
  // Captured wait descriptor borrowing the submission's stable arrays.
  iree_async_semaphore_wait_operation_t operation;
  // Owned tracker, freed before invoking terminal completion.
  iree_async_semaphore_wait_tracker_t* tracker;
  // Terminal continuation, invoked exactly once after successful acceptance.
  iree_hal_amd_gpu_host_wait_fn_t fn;
  // Borrowed continuation context, live through the terminal callback.
  void* user_data;
} iree_hal_amd_gpu_host_wait_t;

// Prepares a nonempty, already captured ALL wait. The tracker retains the
// semaphores; arrays, sidecar, and context remain live through the terminal
// callback. No callback runs until register. This split lets an owner link the
// prepared wait into its cancellation set before registration can complete.
//
// Failure returns before acceptance, retains nothing, and invokes no callback.
// Success requires exactly one register, even if cancellation arrives first.
// Allocation uses host_allocator only for the tracker and its timepoints.
// Fixed-capacity callers may supply an inline arena sized by the tracker's
// storage-size query; finalization frees that allocation before invoking the
// terminal callback.
iree_status_t iree_hal_amd_gpu_host_wait_prepare(
    iree_async_semaphore_wait_context_t* context,
    iree_hal_semaphore_list_t waits, iree_hal_amd_gpu_host_wait_fn_t fn,
    void* user_data, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_host_wait_t* wait);

// Transfers the prepared sidecar to terminal completion, which may run and free
// or reuse it before this returns. Registration performs its final access
// before that handoff. Enclosing owners initialize all callback state first.
void iree_hal_amd_gpu_host_wait_register(iree_hal_amd_gpu_host_wait_t* wait);

// Prepares and registers a captured wait when no external cancellation set
// needs publication between those steps. Failure retains nothing and invokes
// no callback; success transfers storage as register does.
iree_status_t iree_hal_amd_gpu_host_wait_begin(
    iree_async_semaphore_wait_context_t* context,
    iree_hal_semaphore_list_t waits, iree_hal_amd_gpu_host_wait_fn_t fn,
    void* user_data, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_host_wait_t* wait);

// Requests terminal cancellation, racing satisfaction/failure under the async
// tracker's first-terminal-event contract. Completion runs after every detached
// semaphore callback has joined. The caller must keep
// |wait| alive through this call, independently of its terminal callback, and
// must have joined prepare before cancelling. This does not synchronously wait.
void iree_hal_amd_gpu_host_wait_cancel(iree_hal_amd_gpu_host_wait_t* wait);

typedef struct iree_hal_amd_gpu_host_wait_entry_t
    iree_hal_amd_gpu_host_wait_entry_t;

// Failure cancellation for a queue's unpublished dependency waits. Only list
// membership is locked; registration, cancellation, and terminal callbacks run
// outside the lock. Ready submissions never access this set.
typedef struct iree_hal_amd_gpu_host_wait_set_t {
  // Borrowed association context, live through every enrolled wait.
  iree_async_semaphore_wait_context_t* context;
  // Protects head, linked membership, and the permanent cancellation latch.
  iree_slim_mutex_t mutex;
  // Intrusive registered waits; no scan occurs during normal admission.
  iree_hal_amd_gpu_host_wait_entry_t* head;
  // Terminal queue failure rejects subsequent enrollment.
  bool cancelled;
} iree_hal_amd_gpu_host_wait_set_t;

// Placed in the owning capture. The last completion/cancellation borrower
// invokes fn; the callback may immediately reuse or free the entire capture.
struct iree_hal_amd_gpu_host_wait_entry_t {
  // Borrowed cancellation owner, live until this entry's callback.
  iree_hal_amd_gpu_host_wait_set_t* set;
  // Next linked entry, or next cancellation-owned entry after detachment.
  iree_hal_amd_gpu_host_wait_entry_t* next;
  // Previous linked entry, protected by the set mutex.
  iree_hal_amd_gpu_host_wait_entry_t* previous;
  // Membership in the set, cleared before handing the list to cancellation.
  bool linked;
  // One terminal callback and an optional detached cancellation borrower.
  iree_atomic_int32_t borrowers;
  // Owned terminal result, published before the completion borrower returns.
  iree_status_t status;
  // Final capture continuation, invoked outside the set lock.
  iree_hal_amd_gpu_host_wait_fn_t fn;
  // Borrowed capture context dominated by this entry's ownership.
  void* user_data;
  // Prepared tracker with its independent semaphore callback join.
  iree_hal_amd_gpu_host_wait_t wait;
};

void iree_hal_amd_gpu_host_wait_set_initialize(
    iree_async_semaphore_wait_context_t* context,
    iree_hal_amd_gpu_host_wait_set_t* out_set);

// Every accepted entry and concurrent cancellation must have joined first.
void iree_hal_amd_gpu_host_wait_set_deinitialize(
    iree_hal_amd_gpu_host_wait_set_t* set);

// Atomically enrolls a prepared wait before registration. Failure retains
// nothing and invokes no callback. Success transfers entry and the borrowed
// arrays through fn, which may run before begin returns. A cancelled set
// completes this entry with cancellation without registering dependency work.
iree_status_t iree_hal_amd_gpu_host_wait_set_begin(
    iree_hal_amd_gpu_host_wait_set_t* set, iree_hal_semaphore_list_t waits,
    iree_hal_amd_gpu_host_wait_fn_t fn, void* user_data,
    iree_allocator_t host_allocator, iree_hal_amd_gpu_host_wait_entry_t* entry);

// Detaches and borrows all entries under the membership lock, then requests
// cancellation outside it. This neither waits for callbacks nor loses entries
// racing enrollment. May be called concurrently; the terminal latch is sticky.
// The caller keeps set alive until cancel returns, independently of entries.
void iree_hal_amd_gpu_host_wait_set_cancel(
    iree_hal_amd_gpu_host_wait_set_t* set);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_HOST_WAIT_H_
