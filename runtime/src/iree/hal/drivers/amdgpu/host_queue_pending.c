// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/operations/scheduling.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/processor.h"
#include "iree/hal/drivers/amdgpu/host_queue_memory.h"
#include "iree/hal/drivers/amdgpu/host_queue_pending_operation.h"
#include "iree/hal/drivers/amdgpu/host_queue_waits.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/transient_buffer.h"
#include "iree/hal/pool_wait.h"

//===----------------------------------------------------------------------===//
// Pending operations (deferred submission)
//===----------------------------------------------------------------------===//
//
// LOCKING PROTOCOL
//
// submission_mutex protects all submission-path state: AQL ring reservation,
// kernarg allocation, packet emission, commit_signals, frontier mutation,
// notification ring push, and the pending list (link/unlink).
//
// The completion thread (drain, error check) does NOT acquire
// submission_mutex. It reads the notification ring (SPSC consumer) and the
// atomic error_status.
//
// Deferred operations use a two-phase protocol:
//
//   Phase 1 (under submission_mutex): resolve_waits, allocate pending_op,
//     capture operation parameters, link to pending list.
//
//   Phase 2 (WITHOUT submission_mutex): register timepoints via enqueue_waits.
//     Timepoint callbacks may fire synchronously during acquire_timepoint
//     (when the semaphore value is already reached or the semaphore is already
//     failed). The last callback to fire calls pending_op_issue or
//     pending_op_fail, both of which acquire submission_mutex internally.
//     This is safe because Phase 1 released the mutex before Phase 2 began.
//
// pending_op_issue: acquires submission_mutex to emit AQL packets, transfer
//   retained resources to the reclaim ring, commit signals, and unlink.
//
// pending_op_fail: acquires submission_mutex to unlink. Semaphore failure
//   and resource release happen outside the lock.
//
// pending_op_discard_under_lock: for capture-time failures (arena allocation
//   errors after pending_op_allocate). Caller already holds submission_mutex.
//   Does NOT re-acquire; unlinks and cleans up directly.

// A result is visible before notification posting finishes. Retirement is the
// final storage handoff; joining a result also joins that brief final access.
typedef enum iree_hal_amdgpu_callback_state_e {
  IREE_HAL_AMDGPU_CALLBACK_PENDING = 0,
  IREE_HAL_AMDGPU_CALLBACK_NOTIFYING = 1,
  IREE_HAL_AMDGPU_CALLBACK_RETIRED = 2,
} iree_hal_amdgpu_callback_state_t;

// Per-wait timepoint entry, arena-allocated one per unsatisfied wait. The
// timepoint callback decrements the operation's atomic wait counter; the last
// callback to fire issues or fails the operation.
struct iree_hal_amdgpu_wait_entry_t {
  // Async semaphore timepoint registration owned by this wait entry.
  iree_async_semaphore_timepoint_t timepoint;
  // Pending operation whose wait_count is decremented by this callback.
  iree_hal_amdgpu_pending_op_t* operation;
  // Callback publication and final notification-access retirement state.
  iree_atomic_int32_t callback_state;
};

typedef enum iree_hal_amdgpu_alloca_memory_wait_kind_e {
  // No active memory wait or held reservation transaction.
  IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE = 0,
  // Waiting for a copied pool death frontier while holding reservations.
  IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER = 1,
  // Awaiting a pool acquisition callback outside submission_mutex.
  IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_ACQUISITION = 2,
  // Waiting for local or backing capacity before retrying reservation.
  IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_CAPACITY = 3,
} iree_hal_amdgpu_alloca_memory_wait_kind_t;

// Cold-path alloca memory-readiness wait. Allocated inside a pending op's arena
// after user semaphore waits resolve, before the pool acquisition begins.
struct iree_hal_amdgpu_alloca_memory_wait_t {
  // Active wait source.
  iree_hal_amdgpu_alloca_memory_wait_kind_t kind;

  // Memory callback publication and final notification-access retirement.
  iree_atomic_int32_t callback_state;

  // Held-reservation wait state blocked on a pool death frontier.
  struct {
    // Tracker waiter storage for the transaction's merged wait frontier.
    iree_async_frontier_waiter_t waiter;
  } frontier;

  // Immutable acquisition inputs captured under submission_mutex.
  struct {
    // Queue-order frontier snapshot used by the asynchronous acquisition.
    iree_hal_amdgpu_fixed_frontier_t requester_frontier;

  } acquisition;

  // Cold capacity retry state for reservation attempts.
  struct {
    // Owned helper joining local and backing notification sources.
    iree_hal_pool_wait_t* wait;
    // Observations held between a retry and commit or abort.
    bool prepared;
  } capacity;
};

static void iree_hal_amdgpu_pending_op_issue(iree_hal_amdgpu_pending_op_t* op);
static void iree_hal_amdgpu_pending_op_capacity_post_drain(void* user_data);
static void iree_hal_amdgpu_pending_op_fail(iree_hal_amdgpu_pending_op_t* op,
                                            iree_status_t status);
// Links a pending op into the queue's pending list. Caller must hold
// submission_mutex.
static void iree_hal_amdgpu_pending_op_link(iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  op->next = queue->pending_head;
  op->prev_next = &queue->pending_head;
  if (queue->pending_head) {
    queue->pending_head->prev_next = &op->next;
  }
  queue->pending_head = op;
}

// Unlinks a pending op from the queue's pending list. Caller must hold
// submission_mutex.
static void iree_hal_amdgpu_pending_op_unlink(
    iree_hal_amdgpu_pending_op_t* op) {
  *op->prev_next = op->next;
  if (op->next) {
    op->next->prev_next = op->prev_next;
  }
  op->next = NULL;
  op->prev_next = NULL;
}

// Retains a resource and appends it to the pending op's retained_resources
// array. The caller must have allocated sufficient capacity in the array
// via the max_resource_count parameter to pending_op_allocate.
void iree_hal_amdgpu_pending_op_retain(iree_hal_amdgpu_pending_op_t* op,
                                       iree_hal_resource_t* resource) {
  if (IREE_LIKELY(resource)) {
    iree_hal_resource_retain(resource);
    op->retained_resources[op->retained_resource_count++] = resource;
  }
}

// Releases all retained HAL resources in the flat array. Used on failure,
// cancellation, and success paths where the submit helper retained the
// resources it needs instead of consuming this pending op's refs.
void iree_hal_amdgpu_pending_op_release_retained(
    iree_hal_amdgpu_pending_op_t* op) {
  for (uint16_t i = 0; i < op->retained_resource_count; ++i) {
    iree_hal_resource_release(op->retained_resources[i]);
  }
  op->retained_resource_count = 0;
}

static void iree_hal_amdgpu_pending_op_release_execute_binding_resource_set(
    iree_hal_amdgpu_pending_op_t* op) {
  if (op->type == IREE_HAL_AMDGPU_PENDING_OP_EXECUTE) {
    iree_hal_resource_set_free(op->execute.binding_resource_set);
    op->execute.binding_resource_set = NULL;
  }
}

static void iree_hal_amdgpu_pending_op_fail_host_action(
    iree_hal_amdgpu_pending_op_t* op, const iree_status_t status) {
  if (op->type != IREE_HAL_AMDGPU_PENDING_OP_HOST_ACTION ||
      !op->host_action.action.fn) {
    return;
  }
  op->host_action.action.fn(/*entry=*/NULL, op->host_action.action.user_data,
                            status);
  op->host_action.action.fn = NULL;
  op->host_action.action.user_data = NULL;
}

// Releases cold wait storage and any queue-owned alloca reservation. A
// successful submission has already transferred its reservation into the
// transient buffer; failure returns the still-owned transaction.
static void iree_hal_amdgpu_pending_op_release_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  if (op->type != IREE_HAL_AMDGPU_PENDING_OP_ALLOCA) {
    return;
  }
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  if (wait) {
    wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
    if (wait->capacity.prepared) {
      iree_hal_pool_wait_abort(wait->capacity.wait);
      wait->capacity.prepared = false;
    }
    iree_hal_pool_wait_free(wait->capacity.wait);
    wait->capacity.wait = NULL;
  }
  iree_hal_amdgpu_host_queue_release_alloca_transaction(
      op->alloca_op.pool, &op->alloca_op.transaction);
}

// Clears the queued marker for a deferred dealloca that never published a
// completion epoch. Successful deallocas transfer ownership to the reclaim ring
// and must not call this.
static void iree_hal_amdgpu_pending_op_abort_unsubmitted_dealloca(
    iree_hal_amdgpu_pending_op_t* op) {
  if (op->type != IREE_HAL_AMDGPU_PENDING_OP_DEALLOCA) {
    return;
  }
  for (iree_host_size_t i = 0; i < op->dealloca.transaction.buffer_count; ++i) {
    iree_hal_buffer_allocation_abort_dealloca(
        op->dealloca.transaction.buffers[i]);
  }
}

static bool iree_hal_amdgpu_callback_is_resolved(void* user_data) {
  iree_atomic_int32_t* state = user_data;
  return iree_atomic_load(state, iree_memory_order_acquire) !=
         IREE_HAL_AMDGPU_CALLBACK_PENDING;
}

static void iree_hal_amdgpu_publish_callback_complete(
    iree_notification_t* notification, iree_atomic_int32_t* state) {
  iree_atomic_store(state, IREE_HAL_AMDGPU_CALLBACK_NOTIFYING,
                    iree_memory_order_release);
  iree_notification_post(notification, IREE_ALL_WAITERS);
  // No further access to callback-owned storage after this handoff.
  iree_atomic_store(state, IREE_HAL_AMDGPU_CALLBACK_RETIRED,
                    iree_memory_order_release);
}

static void iree_hal_amdgpu_join_callback(iree_notification_t* notification,
                                          iree_atomic_int32_t* state) {
  iree_notification_await(notification, iree_hal_amdgpu_callback_is_resolved,
                          state, iree_infinite_timeout());
  while (iree_atomic_load(state, iree_memory_order_acquire) !=
         IREE_HAL_AMDGPU_CALLBACK_RETIRED) {
    iree_processor_yield();
  }
}

static void iree_hal_amdgpu_alloca_memory_wait_publish_callback_complete(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_publish_callback_complete(
      &op->callback_notification, &op->alloca_op.memory_wait->callback_state);
}

static void iree_hal_amdgpu_join_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_join_callback(&op->callback_notification,
                                &op->alloca_op.memory_wait->callback_state);
}

// Publishes a prepared memory-readiness wait as ARMING. The release store on
// lifecycle_state makes the initialized sidecar fields visible to the callback
// or cancellation path that observes the state transition.
static void iree_hal_amdgpu_pending_op_begin_alloca_memory_wait_arming(
    iree_hal_amdgpu_pending_op_t* op,
    iree_hal_amdgpu_alloca_memory_wait_t* wait,
    iree_hal_amdgpu_alloca_memory_wait_kind_t kind) {
  wait->kind = kind;
  iree_atomic_store(&wait->callback_state, IREE_HAL_AMDGPU_CALLBACK_RETIRED,
                    iree_memory_order_relaxed);
  iree_atomic_store(&op->lifecycle_state,
                    IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING,
                    iree_memory_order_release);
}

static iree_status_t iree_hal_amdgpu_pending_op_ensure_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op,
    iree_hal_amdgpu_alloca_memory_wait_t** out_wait) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  if (!wait) {
    IREE_TRACE_ZONE_BEGIN(z0);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(&op->arena, sizeof(*wait), (void**)&wait));
    memset(wait, 0, sizeof(*wait));
    iree_atomic_store(&wait->callback_state, IREE_HAL_AMDGPU_CALLBACK_RETIRED,
                      iree_memory_order_relaxed);
    op->alloca_op.memory_wait = wait;
    IREE_TRACE_ZONE_END(z0);
  }
  *out_wait = wait;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pending_op_prepare_alloca_frontier_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  if (IREE_UNLIKELY(op->alloca_op.transaction.wait_frontier->entry_count ==
                    0)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "queue_alloca waitable pool reservation did not provide a frontier");
  }

  iree_hal_amdgpu_alloca_memory_wait_t* wait = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pending_op_ensure_alloca_memory_wait(op, &wait));
  iree_hal_amdgpu_pending_op_begin_alloca_memory_wait_arming(
      op, wait, IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pending_op_prepare_alloca_acquisition(
    iree_hal_amdgpu_pending_op_t* op,
    const iree_async_frontier_t* requester_frontier) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pending_op_ensure_alloca_memory_wait(op, &wait));
  iree_async_frontier_t* acquisition_frontier =
      iree_hal_amdgpu_fixed_frontier_as_frontier(
          &wait->acquisition.requester_frontier);
  iree_async_frontier_initialize(acquisition_frontier,
                                 requester_frontier->entry_count);
  memcpy(
      acquisition_frontier->entries, requester_frontier->entries,
      requester_frontier->entry_count * sizeof(requester_frontier->entries[0]));
  op->alloca_op.transaction.acquire_result = IREE_HAL_POOL_ACQUIRE_NONE;
  if (wait->capacity.wait && !wait->capacity.prepared) {
    iree_hal_pool_wait_prepare(wait->capacity.wait);
    wait->capacity.prepared = true;
  }
  iree_hal_amdgpu_pending_op_begin_alloca_memory_wait_arming(
      op, wait, IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_ACQUISITION);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pending_op_prepare_alloca_capacity_wait(
    iree_hal_amdgpu_pending_op_t* op, iree_hal_pool_wait_t* capacity_wait) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pending_op_ensure_alloca_memory_wait(op, &wait));
  wait->capacity.wait = capacity_wait;
  wait->capacity.prepared = true;
  iree_hal_amdgpu_pending_op_begin_alloca_memory_wait_arming(
      op, wait, IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_CAPACITY);
  return iree_ok_status();
}

static void iree_hal_amdgpu_alloca_capacity_end_observe(
    iree_hal_amdgpu_alloca_memory_wait_t* wait) {
  if (wait->capacity.prepared) {
    wait->capacity.prepared = false;
    iree_hal_pool_wait_abort(wait->capacity.wait);
  }
}

// Cancels any active alloca memory-readiness wait before destroying the op.
static void iree_hal_amdgpu_pending_op_cancel_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  if (op->type != IREE_HAL_AMDGPU_PENDING_OP_ALLOCA) {
    return;
  }
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  if (!wait) {
    return;
  }

  switch (wait->kind) {
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER: {
      const bool cancelled = iree_async_frontier_tracker_cancel_wait(
          op->queue->frontier_tracker, &wait->frontier.waiter);
      if (!cancelled) {
        iree_hal_amdgpu_join_alloca_memory_wait(op);
      }
      break;
    }
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_CAPACITY: {
      iree_hal_amdgpu_alloca_capacity_end_observe(wait);
      iree_hal_pool_wait_cancel(wait->capacity.wait);
      iree_hal_amdgpu_join_alloca_memory_wait(op);
      break;
    }
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_ACQUISITION:
      iree_hal_amdgpu_join_alloca_memory_wait(op);
      iree_hal_amdgpu_host_queue_release_alloca_transaction(
          op->alloca_op.pool, &op->alloca_op.transaction);
      wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
      break;
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE:
      break;
  }
}

static bool iree_hal_amdgpu_wait_entry_callback_is_complete(
    iree_hal_amdgpu_wait_entry_t* entry) {
  return iree_atomic_load(&entry->callback_state, iree_memory_order_acquire) ==
         IREE_HAL_AMDGPU_CALLBACK_RETIRED;
}

static void iree_hal_amdgpu_wait_entry_publish_callback_complete(
    iree_hal_amdgpu_wait_entry_t* entry) {
  iree_hal_amdgpu_publish_callback_complete(
      &entry->operation->callback_notification, &entry->callback_state);
}

static void iree_hal_amdgpu_pending_op_join_wait_callbacks(
    iree_hal_amdgpu_pending_op_t* op) {
  for (iree_host_size_t i = 0; i < op->wait_semaphore_list.count; ++i) {
    iree_hal_amdgpu_join_callback(&op->callback_notification,
                                  &op->wait_entries[i].callback_state);
  }
}

// Records the first asynchronous wait failure. Takes ownership of |status|,
// storing it for the completion owner or dropping it if another failure won.
static void iree_hal_amdgpu_pending_op_record_error_status(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return;
  }
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &op->error_status, &expected, (intptr_t)status,
          iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
    iree_status_free(status);
  }
}

static bool iree_hal_amdgpu_pending_op_mark_waits_resolved(
    iree_hal_amdgpu_pending_op_t* op) {
  int32_t expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING;
  if (iree_atomic_compare_exchange_strong(
          &op->lifecycle_state, &expected_state,
          IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    // The registering producer joins every callback before completing the op.
    return false;
  }
  expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING;
  return iree_atomic_compare_exchange_strong(
      &op->lifecycle_state, &expected_state,
      IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
      iree_memory_order_acq_rel, iree_memory_order_acquire);
}

static void iree_hal_amdgpu_pending_op_complete_resolved_waits(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_pending_op_join_wait_callbacks(op);
  iree_status_t error = (iree_status_t)iree_atomic_exchange(
      &op->error_status, 0, iree_memory_order_acquire);
  if (!iree_status_is_ok(error)) {
    iree_hal_amdgpu_pending_op_fail(op, error);
  } else {
    iree_hal_amdgpu_pending_op_issue(op);
  }
}

// Discards a pending operation that failed during capture. Caller MUST hold
// submission_mutex; the op is linked to the pending list by allocate and needs
// the mutex for unlinking. Signal semaphores remain untouched because the
// operation was never successfully captured.
//
// Unlike pending_op_fail (which acquires submission_mutex internally), this
// function assumes the caller already holds it. This is necessary because the
// capture phase runs under the mutex (Phase 1 of the two-phase protocol).
void iree_hal_amdgpu_pending_op_discard_under_lock(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_pending_op_abort_unsubmitted_dealloca(op);
  // Release any queue-owned memory reservation before releasing op resources.
  iree_hal_amdgpu_pending_op_release_alloca_memory_wait(op);
  iree_hal_amdgpu_pending_op_release_execute_binding_resource_set(op);
  // Release all retained resources (signal semaphores + op resources).
  iree_hal_amdgpu_pending_op_release_retained(op);
  // Release wait semaphores (separately retained by the clone).
  iree_hal_semaphore_list_release(op->wait_semaphore_list);
  // Unlink from the pending list (caller holds submission_mutex).
  iree_hal_amdgpu_pending_op_unlink(op);
  // Tear down callback wake state before returning arena blocks to the pool.
  iree_notification_deinitialize(&op->callback_notification);
  // Return arena blocks to the pool.
  iree_arena_deinitialize(&op->arena);
}

// Timepoint callback fired when a wait semaphore reaches its target value or
// fails. The last resolved wait claims completion, waits for all callbacks to
// finish touching arena-owned entries, and then issues or fails the operation.
static void iree_hal_amdgpu_wait_entry_resolved(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_amdgpu_wait_entry_t* entry =
      (iree_hal_amdgpu_wait_entry_t*)user_data;
  iree_hal_amdgpu_pending_op_t* op = entry->operation;

  iree_hal_amdgpu_pending_op_record_error_status(op, status);

  int32_t previous_count =
      iree_atomic_fetch_sub(&op->wait_count, 1, iree_memory_order_acq_rel);
  bool owns_completion = false;
  if (previous_count == 1) {
    owns_completion = iree_hal_amdgpu_pending_op_mark_waits_resolved(op);
  }

  iree_hal_amdgpu_wait_entry_publish_callback_complete(entry);
  if (owns_completion) {
    iree_hal_amdgpu_pending_op_complete_resolved_waits(op);
  }
}

// Registers timepoints for all waits in the operation's wait semaphore list.
// Sets wait_count and registers one timepoint per wait; callbacks may fire
// synchronously during registration. When all waits are satisfied (the last
// callback fires), the operation is issued or failed.
//
// The wait_semaphore_list on the op (cloned into the arena by allocate)
// retains all semaphores for the lifetime of the op. Wait entries do not
// independently retain semaphores.
static iree_status_t iree_hal_amdgpu_pending_op_enqueue_waits(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_semaphore_list_t wait_semaphores = op->wait_semaphore_list;
  if (wait_semaphores.count == 0) {
    iree_hal_amdgpu_pending_op_issue(op);
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, wait_semaphores.count);

  iree_host_size_t wait_entry_bytes = 0;
  iree_status_t status =
      IREE_STRUCT_LAYOUT(0, &wait_entry_bytes,
                         IREE_STRUCT_FIELD(wait_semaphores.count,
                                           iree_hal_amdgpu_wait_entry_t, NULL));
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_fail(op, status);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  status = iree_arena_allocate(&op->arena, wait_entry_bytes,
                               (void**)&op->wait_entries);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_fail(op, status);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  memset(op->wait_entries, 0, wait_entry_bytes);
  // Unregistered entries never receive callbacks, so they start complete.
  // Active registrations flip their entry incomplete until the callback exits.
  for (iree_host_size_t i = 0; i < wait_semaphores.count; ++i) {
    iree_atomic_store(&op->wait_entries[i].callback_state,
                      IREE_HAL_AMDGPU_CALLBACK_RETIRED,
                      iree_memory_order_relaxed);
  }

  // Set wait_count before registering any timepoints. A timepoint callback
  // may fire synchronously during acquire_timepoint.
  iree_atomic_store(&op->wait_count, (int32_t)wait_semaphores.count,
                    iree_memory_order_release);

  for (iree_host_size_t i = 0;
       i < wait_semaphores.count && iree_status_is_ok(status); ++i) {
    iree_hal_amdgpu_wait_entry_t* entry = &op->wait_entries[i];
    entry->operation = op;
    iree_atomic_store(&entry->callback_state, IREE_HAL_AMDGPU_CALLBACK_PENDING,
                      iree_memory_order_relaxed);
    entry->timepoint.callback = iree_hal_amdgpu_wait_entry_resolved;
    entry->timepoint.user_data = entry;
    status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)wait_semaphores.semaphores[i],
        wait_semaphores.payload_values[i], &entry->timepoint);

    if (!iree_status_is_ok(status)) {
      // Registration failed at index i. Timepoints 0..i-1 are already
      // registered and their callbacks will fire asynchronously; we cannot
      // destroy the op here. Record the error and subtract the unregistered
      // count so the existing callbacks drain and destroy the op.
      iree_hal_amdgpu_pending_op_record_error_status(op, status);
      int32_t unregistered = (int32_t)(wait_semaphores.count - i);
      iree_atomic_store(&entry->callback_state,
                        IREE_HAL_AMDGPU_CALLBACK_RETIRED,
                        iree_memory_order_release);
      int32_t previous_count = iree_atomic_fetch_sub(
          &op->wait_count, unregistered, iree_memory_order_acq_rel);
      if (previous_count == unregistered) {
        iree_atomic_store(&op->lifecycle_state,
                          IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
                          iree_memory_order_release);
      }
    }
  }

  int32_t expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING;
  if (!iree_atomic_compare_exchange_strong(
          &op->lifecycle_state, &expected_state,
          IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    IREE_ASSERT(expected_state ==
                IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING);
    iree_hal_amdgpu_pending_op_complete_resolved_waits(op);
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_amdgpu_alloca_memory_wait_resolved(void* user_data,
                                                        iree_status_t status) {
  iree_hal_amdgpu_pending_op_t* op = user_data;
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  const bool resolved_successfully = iree_status_is_ok(status);

  int32_t expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING;
  if (iree_atomic_compare_exchange_strong(
          &op->lifecycle_state, &expected_state,
          IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    iree_hal_amdgpu_pending_op_record_error_status(op, status);
    if (resolved_successfully &&
        wait->kind == IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_CAPACITY) {
      wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
    }
    iree_hal_amdgpu_alloca_memory_wait_publish_callback_complete(op);
    return;
  }

  expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING;
  if (iree_atomic_compare_exchange_strong(
          &op->lifecycle_state, &expected_state,
          IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    if (resolved_successfully &&
        wait->kind == IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_CAPACITY) {
      wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
    }
    iree_hal_amdgpu_alloca_memory_wait_publish_callback_complete(op);
    if (!resolved_successfully) {
      iree_hal_amdgpu_pending_op_fail(op, status);
    } else {
      iree_hal_amdgpu_pending_op_issue(op);
    }
    return;
  }

  iree_hal_amdgpu_pending_op_record_error_status(op, status);
  iree_hal_amdgpu_alloca_memory_wait_publish_callback_complete(op);
}

static void iree_hal_amdgpu_pending_op_finish_alloca_memory_wait_enqueue(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  int32_t expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING;
  if (iree_status_is_ok(status)) {
    if (iree_atomic_compare_exchange_strong(
            &op->lifecycle_state, &expected_state,
            IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      return;
    }
    if (expected_state == IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING) {
      iree_hal_amdgpu_join_alloca_memory_wait(op);
      iree_status_t error = (iree_status_t)iree_atomic_exchange(
          &op->error_status, 0, iree_memory_order_acquire);
      if (!iree_status_is_ok(error)) {
        iree_hal_amdgpu_pending_op_fail(op, error);
      } else {
        iree_hal_amdgpu_pending_op_issue(op);
      }
    }
    return;
  }

  if (iree_atomic_compare_exchange_strong(
          &op->lifecycle_state, &expected_state,
          IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    iree_hal_amdgpu_pending_op_fail(op, status);
    return;
  }
  iree_hal_amdgpu_pending_op_record_error_status(op, status);
}

static void iree_hal_amdgpu_pending_op_enqueue_alloca_frontier_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  iree_atomic_store(&wait->callback_state, IREE_HAL_AMDGPU_CALLBACK_PENDING,
                    iree_memory_order_relaxed);
  iree_status_t status = iree_async_frontier_tracker_wait(
      queue->frontier_tracker, op->alloca_op.transaction.wait_frontier,
      iree_hal_amdgpu_alloca_memory_wait_resolved, op, &wait->frontier.waiter);
  iree_hal_amdgpu_pending_op_finish_alloca_memory_wait_enqueue(op, status);
}

static void iree_hal_amdgpu_pending_op_enqueue_alloca_capacity_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  iree_atomic_store(&wait->callback_state, IREE_HAL_AMDGPU_CALLBACK_PENDING,
                    iree_memory_order_relaxed);
  wait->capacity.prepared = false;
  iree_hal_pool_wait_commit(
      wait->capacity.wait, iree_infinite_timeout(),
      (iree_hal_pool_wait_callback_t){
          .fn = iree_hal_amdgpu_alloca_memory_wait_resolved,
          .user_data = op,
      });
  iree_hal_amdgpu_pending_op_finish_alloca_memory_wait_enqueue(
      op, iree_ok_status());
}

static void iree_hal_amdgpu_pending_op_alloca_acquired(
    void* user_data, iree_status_t status,
    iree_hal_pool_acquire_result_t result) {
  iree_hal_amdgpu_pending_op_t* op = user_data;
  iree_hal_amdgpu_alloca_transaction_t* transaction =
      &op->alloca_op.transaction;
  transaction->acquire_result = result;
  transaction->reservations_held =
      iree_status_is_ok(status) &&
      (result == IREE_HAL_POOL_ACQUIRE_OK ||
       result == IREE_HAL_POOL_ACQUIRE_OK_FRESH ||
       result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  iree_hal_amdgpu_alloca_memory_wait_resolved(op, status);
}

static void iree_hal_amdgpu_pending_op_enqueue_alloca_acquisition(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  iree_hal_amdgpu_alloca_transaction_t* transaction =
      &op->alloca_op.transaction;
  iree_atomic_store(&wait->callback_state, IREE_HAL_AMDGPU_CALLBACK_PENDING,
                    iree_memory_order_relaxed);
  iree_status_t status = iree_hal_pool_acquire_reservations(
      op->alloca_op.pool, transaction->request_count, transaction->requests,
      iree_hal_pool_requires_asan_advice(op->alloca_op.pool)
          ? NULL
          : iree_hal_amdgpu_fixed_frontier_as_frontier(
                &wait->acquisition.requester_frontier),
      transaction->reservations, transaction->acquire_infos,
      (iree_hal_pool_acquire_callback_t){
          .fn = iree_hal_amdgpu_pending_op_alloca_acquired,
          .user_data = op,
      },
      op->alloca_op.reserve_flags);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_alloca_memory_wait_resolved(op, status);
  }
  iree_hal_amdgpu_pending_op_finish_alloca_memory_wait_enqueue(
      op, iree_ok_status());
}

void iree_hal_amdgpu_pending_op_enqueue_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  switch (wait->kind) {
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER:
      iree_hal_amdgpu_pending_op_enqueue_alloca_frontier_wait(op);
      break;
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_ACQUISITION:
      iree_hal_amdgpu_pending_op_enqueue_alloca_acquisition(op);
      break;
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_CAPACITY:
      iree_hal_amdgpu_pending_op_enqueue_alloca_capacity_wait(op);
      break;
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE:
      iree_hal_amdgpu_pending_op_fail(
          op, iree_make_status(IREE_STATUS_INTERNAL,
                               "pending alloca has no memory wait to enqueue"));
      break;
  }
}

static void iree_hal_amdgpu_pending_op_enqueue_capacity_retry(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_atomic_store(&op->lifecycle_state,
                    IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
                    iree_memory_order_release);
  iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
      op->queue, &op->capacity_retry,
      iree_hal_amdgpu_pending_op_capacity_post_drain, op);
}

static void iree_hal_amdgpu_pending_op_capacity_post_drain(void* user_data) {
  iree_hal_amdgpu_pending_op_issue((iree_hal_amdgpu_pending_op_t*)user_data);
}

iree_status_t iree_hal_amdgpu_pending_op_start(iree_hal_amdgpu_pending_op_t* op,
                                               bool wait_for_capacity) {
  if (wait_for_capacity) {
    iree_hal_amdgpu_pending_op_enqueue_capacity_retry(op);
    return iree_ok_status();
  }
  return iree_hal_amdgpu_pending_op_enqueue_waits(op);
}

// Allocates and initializes a pending operation from a fresh arena.
// Clones the wait semaphore list. Allocates the retained_resources array
// with |max_resource_count| capacity and populates the first entries with
// signal semaphores (retained). The signal_semaphore_list.semaphores pointer
// aliases into retained_resources so that commit_signals and
// semaphore_list_fail can use it directly.
//
// The caller must push operation-specific resources into retained_resources
// (via the returned op) before calling enqueue_waits.
//
// On failure, the arena is cleaned up and *out_op is set to NULL.
iree_status_t iree_hal_amdgpu_pending_op_allocate(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_amdgpu_pending_op_type_t type, uint16_t max_resource_count,
    iree_hal_amdgpu_pending_op_t** out_op) {
  IREE_ASSERT_ARGUMENT(out_op);
  *out_op = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, type);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, max_resource_count);

  iree_arena_allocator_t arena;
  iree_arena_initialize(queue->block_pool, &arena);

  iree_hal_amdgpu_pending_op_t* op = NULL;
  iree_status_t status = iree_arena_allocate(&arena, sizeof(*op), (void**)&op);
  if (!iree_status_is_ok(status)) {
    iree_arena_deinitialize(&arena);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  memset(op, 0, sizeof(*op));
  memcpy(&op->arena, &arena, sizeof(arena));
  op->queue = queue;
  op->type = type;
  iree_atomic_store(&op->wait_count, 0, iree_memory_order_relaxed);
  iree_atomic_store(&op->lifecycle_state,
                    IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING,
                    iree_memory_order_relaxed);
  iree_atomic_store(&op->error_status, 0, iree_memory_order_relaxed);
  iree_notification_initialize(&op->callback_notification);

  iree_allocator_t arena_allocator = iree_arena_allocator(&op->arena);

  // Clone the wait semaphore list (retains each wait semaphore).
  status = iree_hal_semaphore_list_clone(wait_semaphore_list, arena_allocator,
                                         &op->wait_semaphore_list);

  // Allocate the retained resources array and the signal payload values.
  if (iree_status_is_ok(status) && max_resource_count > 0) {
    iree_host_size_t retained_resource_size = 0;
    status = IREE_STRUCT_LAYOUT(
        0, &retained_resource_size,
        IREE_STRUCT_FIELD(max_resource_count, iree_hal_resource_t*, NULL));
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate(&op->arena, retained_resource_size,
                                   (void**)&op->retained_resources);
    }
  }
  uint64_t* signal_payload_values = NULL;
  if (iree_status_is_ok(status) && signal_semaphore_list->count > 0) {
    iree_host_size_t signal_payload_size = 0;
    status = IREE_STRUCT_LAYOUT(
        0, &signal_payload_size,
        IREE_STRUCT_FIELD(signal_semaphore_list->count, uint64_t, NULL));
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate(&op->arena, signal_payload_size,
                                   (void**)&signal_payload_values);
    }
  }

  if (iree_status_is_ok(status)) {
    // Signal semaphores occupy the first entries of retained_resources.
    // The signal_semaphore_list.semaphores pointer aliases this region.
    for (iree_host_size_t i = 0; i < signal_semaphore_list->count; ++i) {
      op->retained_resources[i] =
          (iree_hal_resource_t*)signal_semaphore_list->semaphores[i];
      iree_hal_resource_retain(op->retained_resources[i]);
      signal_payload_values[i] = signal_semaphore_list->payload_values[i];
    }
    op->retained_resource_count = (uint16_t)signal_semaphore_list->count;
    op->signal_semaphore_list.count = signal_semaphore_list->count;
    op->signal_semaphore_list.semaphores =
        (iree_hal_semaphore_t**)op->retained_resources;
    op->signal_semaphore_list.payload_values = signal_payload_values;

    iree_hal_amdgpu_pending_op_link(op);
    *out_op = op;
  } else {
    iree_hal_semaphore_list_release(op->wait_semaphore_list);
    iree_notification_deinitialize(&op->callback_notification);
    iree_arena_deinitialize(&op->arena);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Issues a deferred operation after all waits are satisfied. All waits are
// tier 0 (timeline_value >= waited_value); the GPU work producing those
// values has completed. No barriers are needed.
//
// Called from the last wait_entry callback (any thread). Acquires
// submission_mutex to emit AQL packets and commit signals.
static void iree_hal_amdgpu_pending_op_issue(iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  iree_slim_mutex_lock(&queue->locks.submission_mutex);

  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pending_op_payload_issue_t issue = {
      .ready = true,
      .memory_wait_op = NULL,
  };
  // A recorded failure outranks closed admission. The terminal transition
  // closes admission and then flushes the post-drain retries that resume
  // capacity-parked operations, so consulting the flag first would report the
  // shutdown to an operation resumed there while every operation the same
  // transition cancels reports the fault that caused it.
  status = iree_hal_amdgpu_host_queue_clone_error_status(queue);
  if (iree_status_is_ok(status) && queue->is_shutting_down) {
    status = iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }
  if (iree_status_is_ok(status)) {
    // All waits are tier 0; emit operation packets with no dependency
    // barriers.
    iree_hal_amdgpu_wait_resolution_t resolution;
    resolution.barrier_count = 0;
    resolution.payload_barriers = op->payload_barriers;
    resolution.needs_deferral = false;
    memset(resolution.reserved, 0, sizeof(resolution.reserved));
    resolution.wait_count = op->wait_semaphore_list.count > UINT32_MAX
                                ? UINT32_MAX
                                : (uint32_t)op->wait_semaphore_list.count;
    resolution.profile_event_flags =
        IREE_HAL_PROFILE_QUEUE_EVENT_FLAG_SOFTWARE_DEFERRED;
    resolution.inline_acquire_scope = op->wait_semaphore_list.count > 0
                                          ? IREE_HSA_FENCE_SCOPE_SYSTEM
                                          : IREE_HSA_FENCE_SCOPE_NONE;
    resolution.barrier_acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;
    status = iree_hal_amdgpu_pending_op_issue_payload(op, &resolution, &issue);
    if (iree_status_is_ok(status) && issue.memory_wait_op) {
      iree_slim_mutex_unlock(&queue->locks.submission_mutex);
      iree_hal_amdgpu_pending_op_enqueue_alloca_memory_wait(
          issue.memory_wait_op);
      return;
    }
  }

  if (iree_status_is_ok(status) && !issue.ready) {
    iree_hal_amdgpu_pending_op_enqueue_capacity_retry(op);
    iree_slim_mutex_unlock(&queue->locks.submission_mutex);
    return;
  }

  if (!iree_status_is_ok(status)) {
    // Failing a semaphore may synchronously complete a dependent operation
    // on this queue. Its terminal path acquires submission_mutex to unlink.
    // Completion ownership keeps this op alive until fail retires it.
    iree_slim_mutex_unlock(&queue->locks.submission_mutex);
    iree_hal_amdgpu_pending_op_fail(op, status);
    return;
  }
  iree_hal_amdgpu_pending_op_release_alloca_memory_wait(op);

  // Clean up the pending op. Wait semaphore list is released (the clone holds
  // separate retains). Remaining retained_resources entries are either
  // transferred to reclaim or were released by the success path above.
  iree_hal_semaphore_list_release(op->wait_semaphore_list);
  iree_hal_amdgpu_pending_op_unlink(op);
  iree_notification_deinitialize(&op->callback_notification);
  iree_arena_deinitialize(&op->arena);

  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
}

// Fails a deferred operation. Propagates the error to all signal semaphores
// so downstream waiters receive the failure instead of hanging. Takes
// ownership of |status|.
static void iree_hal_amdgpu_pending_op_fail(iree_hal_amdgpu_pending_op_t* op,
                                            iree_status_t status) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  iree_hal_amdgpu_pending_op_fail_host_action(op, status);
  // Fail signal semaphores (records error, does not release our retains).
  iree_hal_semaphore_list_fail(op->signal_semaphore_list, status);
  iree_hal_amdgpu_pending_op_abort_unsubmitted_dealloca(op);
  // Release any queue-owned memory reservation before releasing op resources.
  iree_hal_amdgpu_pending_op_release_alloca_memory_wait(op);
  iree_hal_amdgpu_pending_op_release_execute_binding_resource_set(op);
  // Release all retained resources (signal semaphores + op resources).
  iree_hal_amdgpu_pending_op_release_retained(op);
  // Release wait semaphores (separately retained by the clone).
  iree_hal_semaphore_list_release(op->wait_semaphore_list);
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  iree_hal_amdgpu_pending_op_unlink(op);
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  iree_notification_deinitialize(&op->callback_notification);
  iree_arena_deinitialize(&op->arena);
}

void iree_hal_amdgpu_host_queue_cancel_pending(
    iree_hal_amdgpu_host_queue_t* queue, const iree_status_t failure_status) {
  for (;;) {
    iree_hal_amdgpu_pending_op_t* op = NULL;
    iree_slim_mutex_lock(&queue->locks.submission_mutex);
    for (iree_hal_amdgpu_pending_op_t* candidate = queue->pending_head;
         candidate != NULL; candidate = candidate->next) {
      int32_t expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING;
      if (iree_atomic_compare_exchange_strong(
              &candidate->lifecycle_state, &expected_state,
              IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_CANCELLING,
              iree_memory_order_acq_rel, iree_memory_order_acquire)) {
        iree_hal_amdgpu_pending_op_unlink(candidate);
        op = candidate;
        break;
      }
    }
    bool has_pending_ops = queue->pending_head != NULL;
    iree_slim_mutex_unlock(&queue->locks.submission_mutex);

    if (op == NULL) {
      if (!has_pending_ops) {
        break;
      }
      iree_thread_yield();
      continue;
    }

    for (iree_host_size_t i = 0; i < op->wait_semaphore_list.count; ++i) {
      iree_hal_amdgpu_wait_entry_t* entry = &op->wait_entries[i];
      if (iree_hal_amdgpu_wait_entry_callback_is_complete(entry)) {
        continue;
      }
      if (iree_async_semaphore_cancel_timepoint(entry->timepoint.semaphore,
                                                &entry->timepoint)) {
        continue;
      }
      iree_hal_amdgpu_join_callback(&op->callback_notification,
                                    &entry->callback_state);
    }
    iree_hal_amdgpu_pending_op_cancel_alloca_memory_wait(op);

    iree_status_t op_status = (iree_status_t)iree_atomic_exchange(
        &op->error_status, 0, iree_memory_order_acquire);
    if (iree_status_is_ok(op_status)) {
      op_status = iree_status_clone(failure_status);
    }
    iree_hal_amdgpu_pending_op_fail_host_action(op, op_status);
    iree_hal_semaphore_list_fail(op->signal_semaphore_list, op_status);
    iree_hal_amdgpu_pending_op_abort_unsubmitted_dealloca(op);
    iree_hal_amdgpu_pending_op_release_alloca_memory_wait(op);
    iree_hal_amdgpu_pending_op_release_execute_binding_resource_set(op);
    iree_hal_amdgpu_pending_op_release_retained(op);
    iree_hal_semaphore_list_release(op->wait_semaphore_list);
    iree_notification_deinitialize(&op->callback_notification);
    iree_arena_deinitialize(&op->arena);
  }
}

//===----------------------------------------------------------------------===//
// Alloca memory-readiness waits
//===----------------------------------------------------------------------===//

static iree_status_t
iree_hal_amdgpu_host_queue_submit_alloca_held_frontier_wait(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_alloca_transaction_t* transaction,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_alloca_memory_wait_t* memory_wait, bool* out_ready) {
  transaction->readiness = IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY;
  transaction->wait_resolution = *resolution;
  memory_wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
  return iree_hal_amdgpu_host_queue_submit_alloca_transaction(
      queue, transaction, signal_semaphore_list, allocation_pool,
      submission_flags, out_ready);
}

static iree_status_t iree_hal_amdgpu_host_queue_get_alloca_memory_wait_op(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_alloca_transaction_t* transaction,
    iree_hal_pool_reserve_flags_t reserve_flags,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op) {
  *out_memory_wait_op = NULL;
  if (pending_op) {
    *out_memory_wait_op = pending_op;
    return iree_ok_status();
  }

  iree_hal_semaphore_list_t empty_wait_list = iree_hal_semaphore_list_empty();
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_defer_alloca(
      queue, &empty_wait_list, &signal_semaphore_list, allocation_pool,
      transaction->request_count, transaction->requests, transaction->buffers,
      reserve_flags, out_memory_wait_op));
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_move_alloca_acquisition(
    iree_hal_amdgpu_alloca_transaction_t* source,
    iree_hal_amdgpu_alloca_transaction_t* target) {
  if (source == target) {
    return iree_ok_status();
  }
  IREE_ASSERT_TRUE(source->request_count == target->request_count);
  IREE_ASSERT_TRUE(source->reservations_held);
  memcpy(target->reservations, source->reservations,
         source->request_count * sizeof(*source->reservations));
  memcpy(target->acquire_infos, source->acquire_infos,
         source->request_count * sizeof(*source->acquire_infos));
  iree_host_size_t frontier_size = 0;
  IREE_RETURN_IF_ERROR(iree_async_frontier_size(
      source->wait_frontier->entry_count, &frontier_size));
  memcpy(target->wait_frontier, source->wait_frontier, frontier_size);
  target->readiness = source->readiness;
  target->acquire_result = source->acquire_result;
  target->wait_resolution = source->wait_resolution;
  target->reservations_held = true;
  source->reservations_held = false;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_defer_alloca_frontier_wait(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_alloca_transaction_t* transaction,
    iree_hal_pool_reserve_flags_t reserve_flags,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op) {
  iree_hal_amdgpu_pending_op_t* memory_wait_op = pending_op;
  iree_status_t status = iree_hal_amdgpu_host_queue_get_alloca_memory_wait_op(
      queue, signal_semaphore_list, allocation_pool, transaction, reserve_flags,
      pending_op, &memory_wait_op);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_move_alloca_acquisition(
        transaction, &memory_wait_op->alloca_op.transaction);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amdgpu_pending_op_prepare_alloca_frontier_wait(memory_wait_op);
  }
  if (iree_status_is_ok(status)) {
    *out_memory_wait_op = memory_wait_op;
  } else {
    if (!pending_op && memory_wait_op) {
      iree_hal_amdgpu_pending_op_discard_under_lock(memory_wait_op);
    }
  }

  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_defer_alloca_acquisition(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_alloca_transaction_t* transaction,
    iree_hal_pool_reserve_flags_t reserve_flags,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op) {
  iree_hal_amdgpu_pending_op_t* memory_wait_op = pending_op;
  iree_status_t status = iree_hal_amdgpu_host_queue_get_alloca_memory_wait_op(
      queue, signal_semaphore_list, allocation_pool, transaction, reserve_flags,
      pending_op, &memory_wait_op);
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_fixed_frontier_t requester_frontier_storage;
    const iree_async_frontier_t* requester_frontier =
        iree_hal_amdgpu_host_queue_pool_requester_frontier(
            queue, resolution, &requester_frontier_storage);
    status = iree_hal_amdgpu_pending_op_prepare_alloca_acquisition(
        memory_wait_op, requester_frontier);
  }
  if (iree_status_is_ok(status)) {
    *out_memory_wait_op = memory_wait_op;
  } else if (!pending_op && memory_wait_op) {
    iree_hal_amdgpu_pending_op_discard_under_lock(memory_wait_op);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_host_queue_submit_alloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_alloca_transaction_t* transaction,
    iree_hal_pool_reserve_flags_t reserve_flags,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op, bool* out_ready) {
  IREE_ASSERT_ARGUMENT(out_ready);
  *out_ready = false;
  *out_memory_wait_op = NULL;
  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }

  iree_hal_amdgpu_alloca_memory_wait_t* memory_wait =
      pending_op ? pending_op->alloca_op.memory_wait : NULL;
  if (memory_wait &&
      memory_wait->kind == IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER) {
    return iree_hal_amdgpu_host_queue_submit_alloca_held_frontier_wait(
        queue, resolution, signal_semaphore_list, allocation_pool, transaction,
        submission_flags, memory_wait, out_ready);
  }
  if (!memory_wait ||
      memory_wait->kind != IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_ACQUISITION ||
      (!transaction->reservations_held &&
       transaction->acquire_result != IREE_HAL_POOL_ACQUIRE_EXHAUSTED &&
       transaction->acquire_result != IREE_HAL_POOL_ACQUIRE_OVER_BUDGET)) {
    return iree_hal_amdgpu_host_queue_defer_alloca_acquisition(
        queue, resolution, signal_semaphore_list, allocation_pool, transaction,
        reserve_flags, pending_op, out_memory_wait_op);
  }

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_resolve_alloca_transaction(
      queue, resolution, allocation_pool, transaction));
  if (transaction->readiness !=
      IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_POOL_NOTIFICATION) {
    iree_hal_amdgpu_alloca_capacity_end_observe(memory_wait);
  }
  switch (transaction->readiness) {
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY:
      return iree_hal_amdgpu_host_queue_submit_alloca_transaction(
          queue, transaction, signal_semaphore_list, allocation_pool,
          submission_flags, out_ready);
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_FRONTIER_WAIT:
      return iree_hal_amdgpu_host_queue_defer_alloca_frontier_wait(
          queue, signal_semaphore_list, allocation_pool, transaction,
          reserve_flags, pending_op, out_memory_wait_op);
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_POOL_NOTIFICATION:
      if (!memory_wait->capacity.prepared) {
        if (!memory_wait->capacity.wait) {
          IREE_RETURN_IF_ERROR(iree_hal_pool_wait_allocate(
              allocation_pool, queue->host_allocator,
              &memory_wait->capacity.wait));
        }
        return iree_hal_amdgpu_host_queue_defer_alloca_acquisition(
            queue, resolution, signal_semaphore_list, allocation_pool,
            transaction, reserve_flags, pending_op, out_memory_wait_op);
      }
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_pending_op_prepare_alloca_capacity_wait(
              pending_op, memory_wait->capacity.wait));
      *out_memory_wait_op = pending_op;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unrecognized alloca reservation readiness %u",
                              transaction->readiness);
  }
}
