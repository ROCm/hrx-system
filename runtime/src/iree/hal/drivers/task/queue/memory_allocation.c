// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/frontier_tracker.h"
#include "iree/hal/drivers/task/queue/queue.h"
#include "iree/hal/drivers/task/transient_buffer.h"
#include "iree/hal/pool_wait.h"

//===----------------------------------------------------------------------===//
// Alloca memory readiness
//===----------------------------------------------------------------------===//

typedef enum iree_hal_task_queue_alloca_memory_wait_kind_e {
  IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_NONE = 0,
  IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_FRONTIER = 1,
  IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_CAPACITY = 2,
  IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_CAPACITY_PREPARED = 3,
} iree_hal_task_queue_alloca_memory_wait_kind_t;

// Cold-path alloca memory-readiness wait. Allocated inside the queue
// operation's arena only after user semaphore waits have resolved and the pool
// cannot produce immediately-usable bytes.
struct iree_hal_task_queue_alloca_memory_wait_t {
  // Active wait source.
  iree_hal_task_queue_alloca_memory_wait_kind_t kind;

  // State for a held reservation transaction blocked on pool death frontiers.
  struct {
    // Arena-owned union of all reservation death frontiers in the transaction.
    iree_async_frontier_t* wait_frontier;

    // Tracker waiter storage for |wait_frontier|.
    iree_async_frontier_waiter_t waiter;
  } frontier;

  // Arena-owned helper joining local and backing capacity notifications.
  iree_hal_pool_wait_t* capacity_wait;
};

static iree_status_t iree_hal_task_queue_alloca_memory_wait_ensure(
    iree_hal_task_queue_op_t* operation,
    iree_hal_task_queue_alloca_memory_wait_t** out_wait) {
  iree_hal_task_queue_alloca_memory_wait_t* wait =
      operation->alloca.memory_wait;
  if (!wait) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(&operation->arena, sizeof(*wait), (void**)&wait));
    memset(wait, 0, sizeof(*wait));
    operation->alloca.memory_wait = wait;
  }
  *out_wait = wait;
  return iree_ok_status();
}

// Returns unused reservations with their exact prerequisites, including those
// already covered when acquired. Allocation failure does not complete prior
// use.
static void iree_hal_task_queue_alloca_release_reservations(
    iree_hal_task_queue_op_t* operation) {
  if (!operation->alloca.reservations_held) {
    return;
  }
  for (iree_host_size_t i = 0; i < operation->alloca.request_count; ++i) {
    iree_hal_pool_release_reservations(
        operation->alloca.pool, 1, &operation->alloca.reservations[i],
        operation->alloca.acquire_infos[i].reuse_frontier);
  }
  operation->alloca.reservations_held = false;
}

void iree_hal_task_queue_alloca_cleanup(iree_hal_task_queue_op_t* operation) {
  if (operation->type != IREE_HAL_TASK_QUEUE_OP_ALLOCA) {
    return;
  }
  iree_hal_task_queue_alloca_memory_wait_t* wait =
      operation->alloca.memory_wait;
  if (!wait) {
    iree_hal_task_queue_alloca_release_reservations(operation);
    return;
  }

  switch (wait->kind) {
    case IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_FRONTIER:
      iree_hal_task_queue_alloca_release_reservations(operation);
      wait->kind = IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
      break;
    case IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_CAPACITY_PREPARED:
      iree_hal_pool_wait_abort(wait->capacity_wait);
      wait->kind = IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
      break;
    case IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_CAPACITY:
      wait->kind = IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
      break;
    case IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_NONE:
      iree_hal_task_queue_alloca_release_reservations(operation);
      break;
  }
  iree_hal_pool_wait_free(wait->capacity_wait);
  wait->capacity_wait = NULL;
}

static void iree_hal_task_queue_alloca_memory_wait_resolved(
    void* user_data, iree_status_t status) {
  iree_hal_task_queue_op_t* operation = user_data;
  if (!iree_status_is_ok(status)) {
    iree_hal_task_queue_profile_record_failed_before_ready(operation);
    iree_hal_task_queue_op_fail(operation, status);
    return;
  }

  if (iree_hal_task_queue_is_shutting_down(operation->queue)) {
    iree_hal_task_queue_profile_record_failed_before_ready(operation);
    iree_hal_task_queue_op_destroy(operation,
                                   iree_hal_task_queue_make_shutdown_status());
    return;
  }

  iree_hal_task_queue_alloca_memory_wait_t* wait =
      operation->alloca.memory_wait;
  if (wait && wait->kind == IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_CAPACITY) {
    wait->kind = IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
  }
  iree_hal_task_queue_schedule_ready(operation);
}

static iree_status_t iree_hal_task_queue_alloca_wait_for_frontier(
    iree_hal_task_queue_op_t* operation) {
  iree_hal_task_queue_alloca_memory_wait_t* wait = NULL;
  iree_status_t status =
      iree_hal_task_queue_alloca_memory_wait_ensure(operation, &wait);
  if (iree_status_is_ok(status)) {
    iree_host_size_t frontier_size = 0;
    status = iree_async_frontier_size(UINT8_MAX, &frontier_size);
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate(&operation->arena, frontier_size,
                                   (void**)&wait->frontier.wait_frontier);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_async_frontier_initialize(wait->frontier.wait_frontier,
                                   /*entry_count=*/0);
    for (iree_host_size_t i = 0;
         i < operation->alloca.request_count && iree_status_is_ok(status);
         ++i) {
      const iree_async_frontier_t* request_frontier =
          operation->alloca.acquire_infos[i].reuse_frontier;
      if (operation->alloca.acquire_infos[i].result !=
          IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT) {
        continue;
      }
      if (!iree_async_frontier_merge(wait->frontier.wait_frontier, UINT8_MAX,
                                     request_frontier)) {
        status = iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "allocation transaction dependency frontier exceeds %u axes",
            UINT8_MAX);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    wait->kind = IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_FRONTIER;
    // The tracker may invoke the callback inline and transfer ownership of the
    // operation before returning. Record profiling state before registration
    // and never access the operation after a successful call.
    iree_hal_task_queue_profile_force_software_defer(operation);
    iree_hal_task_queue_profile_record_memory_event(
        operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION |
            IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_FRONTIER,
        IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT, operation->alloca.reservations,
        wait->frontier.wait_frontier->entry_count);
    status = iree_async_frontier_tracker_wait(
        operation->frontier_tracker, wait->frontier.wait_frontier,
        iree_hal_task_queue_alloca_memory_wait_resolved, operation,
        &wait->frontier.waiter);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_task_queue_alloca_release_reservations(operation);
    if (wait) {
      wait->kind = IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
    }
  }
  return status;
}

static void iree_hal_task_queue_alloca_wait_for_capacity(
    iree_hal_task_queue_op_t* operation,
    iree_hal_pool_acquire_result_t acquire_result) {
  iree_hal_task_queue_alloca_memory_wait_t* wait =
      operation->alloca.memory_wait;
  wait->kind = IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_CAPACITY;

  // Commit may dispatch before returning, including on admission failure.
  // Finish profiling before handing off the operation to the helper.
  iree_hal_task_queue_profile_force_software_defer(operation);
  iree_hal_task_queue_profile_record_memory_event(
      operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT,
      IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION |
          IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_NOTIFICATION,
      acquire_result, /*reservations=*/NULL, /*frontier_entry_count=*/0);
  iree_hal_pool_wait_commit(
      wait->capacity_wait, iree_infinite_timeout(),
      (iree_hal_pool_wait_callback_t){
          .fn = iree_hal_task_queue_alloca_memory_wait_resolved,
          .user_data = operation,
      });
}

static iree_status_t iree_hal_task_queue_drain_alloca_submit_reservations(
    iree_hal_task_queue_op_t* operation,
    iree_hal_pool_acquire_result_t acquire_result,
    const iree_async_frontier_t* reservation_failure_frontier) {
  iree_hal_task_queue_profile_record_queue_event(operation, iree_time_now());
  iree_hal_task_queue_profile_start_host_execution(operation);
  iree_status_t status = iree_hal_pool_materialize_reservations(
      operation->alloca.pool, operation->alloca.request_count,
      operation->alloca.requests, operation->alloca.reservations,
      IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, operation->alloca.backing_buffers);
  // Unscoped pools are caller-supplied and require native CPU qualification.
  // Scoped sources already establish the complete representation at creation.
  for (iree_host_size_t i = 0;
       !operation->alloca.pool->memory_contract &&
       i < operation->alloca.request_count && iree_status_is_ok(status);
       ++i) {
    iree_byte_span_t span;
    status = iree_hal_buffer_native_host_span(
        operation->alloca.backing_buffers[i], 0,
        iree_hal_buffer_byte_length(operation->alloca.transient_buffers[i]),
        &span);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_pool_advise_asan_reservations(
        operation->alloca.pool, operation->alloca.request_count,
        operation->alloca.reservations,
        IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED);
    for (iree_host_size_t i = 0; i < operation->alloca.request_count; ++i) {
      iree_hal_task_transient_buffer_attach_reservation(
          operation->alloca.transient_buffers[i], operation->alloca.pool,
          &operation->alloca.reservations[i]);
    }
    operation->alloca.reservations_held = false;
    for (iree_host_size_t i = 0; i < operation->alloca.request_count; ++i) {
      iree_hal_task_transient_buffer_stage_backing(
          operation->alloca.transient_buffers[i],
          operation->alloca.backing_buffers[i]);
      iree_hal_buffer_release(operation->alloca.backing_buffers[i]);
      operation->alloca.backing_buffers[i] = NULL;
    }
    iree_hal_task_queue_profile_record_memory_event(
        operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION, acquire_result,
        operation->alloca.reservations,
        reservation_failure_frontier ? reservation_failure_frontier->entry_count
                                     : 0);
  }

  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < operation->alloca.request_count; ++i) {
      iree_hal_task_transient_buffer_commit(
          operation->alloca.transient_buffers[i]);
    }
    iree_hal_task_queue_profile_record_memory_event(
        operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION, acquire_result,
        operation->alloca.reservations,
        reservation_failure_frontier ? reservation_failure_frontier->entry_count
                                     : 0);
    iree_hal_task_queue_op_complete(operation);
  } else {
    for (iree_host_size_t i = 0; i < operation->alloca.request_count; ++i) {
      iree_hal_buffer_release(operation->alloca.backing_buffers[i]);
      operation->alloca.backing_buffers[i] = NULL;
    }
    iree_hal_task_queue_alloca_release_reservations(operation);
    iree_hal_task_queue_profile_record_memory_event(
        operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION,
        iree_status_code(status), operation->alloca.reservations,
        reservation_failure_frontier ? reservation_failure_frontier->entry_count
                                     : 0);
  }
  return status;
}

// Acquisition can complete inline or on a native provider context. The second
// claim protects the submission's final accesses from an early callback.
static iree_status_t iree_hal_task_queue_drain_alloca_acquire(
    iree_hal_task_queue_op_t* operation);

static iree_status_t iree_hal_task_queue_drain_alloca_acquired(
    iree_hal_task_queue_op_t* operation) {
  iree_status_t status = operation->alloca.acquire.status;
  const iree_hal_pool_acquire_result_t acquire_result =
      operation->alloca.acquire.result;
  iree_hal_task_queue_alloca_memory_wait_t* wait =
      operation->alloca.memory_wait;
  const bool capacity_prepared =
      wait &&
      wait->kind == IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_CAPACITY_PREPARED;
  const bool needs_capacity =
      iree_status_is_ok(status) &&
      (acquire_result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED ||
       acquire_result == IREE_HAL_POOL_ACQUIRE_OVER_BUDGET);
  if (capacity_prepared && !needs_capacity) {
    iree_hal_pool_wait_abort(wait->capacity_wait);
    wait->kind = IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
  }
  if (!iree_status_is_ok(status)) {
    return status;
  }

  operation->alloca.reservations_held = !needs_capacity;
  iree_hal_profile_memory_event_flags_t flags =
      IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION;
  if (acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT) {
    flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_FRONTIER;
  } else if (needs_capacity) {
    flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_NOTIFICATION;
  }
  iree_hal_task_queue_profile_record_memory_event(
      operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE, flags,
      acquire_result, needs_capacity ? NULL : operation->alloca.reservations,
      !needs_capacity && operation->alloca.acquire_infos[0].reuse_frontier
          ? operation->alloca.acquire_infos[0].reuse_frontier->entry_count
          : 0);

  switch (acquire_result) {
    case IREE_HAL_POOL_ACQUIRE_OK:
    case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
      return iree_hal_task_queue_drain_alloca_submit_reservations(
          operation, acquire_result, /*reservation_failure_frontier=*/NULL);
    case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
      return iree_hal_task_queue_alloca_wait_for_frontier(operation);
    case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
    case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
      if (capacity_prepared) {
        iree_hal_task_queue_alloca_wait_for_capacity(operation, acquire_result);
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(
          iree_hal_task_queue_alloca_memory_wait_ensure(operation, &wait));
      if (!wait->capacity_wait) {
        IREE_RETURN_IF_ERROR(iree_hal_pool_wait_allocate(
            operation->alloca.pool, iree_arena_allocator(&operation->arena),
            &wait->capacity_wait));
      }
      iree_hal_pool_wait_prepare(wait->capacity_wait);
      wait->kind = IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_CAPACITY_PREPARED;
      return iree_hal_task_queue_drain_alloca_acquire(operation);
    default:
      IREE_CHECK_UNREACHABLE("pool acquisition must produce a terminal result");
  }
}

static void iree_hal_task_queue_alloca_acquire_completed(
    void* user_data, iree_status_t status,
    iree_hal_pool_acquire_result_t result) {
  iree_hal_task_queue_op_t* operation = user_data;
  operation->alloca.acquire.status = status;
  operation->alloca.acquire.result = result;
  if (iree_atomic_fetch_sub(&operation->alloca.acquire.pending, 1,
                            iree_memory_order_acq_rel) != 1) {
    return;
  }
  status = iree_hal_task_queue_drain_alloca_acquired(operation);
  if (!iree_status_is_ok(status)) {
    iree_hal_task_queue_profile_record_failed_before_ready(operation);
    iree_hal_task_queue_op_fail(operation, status);
  }
}

static iree_status_t iree_hal_task_queue_drain_alloca_acquire(
    iree_hal_task_queue_op_t* operation) {
  iree_atomic_store(&operation->alloca.acquire.pending, 2,
                    iree_memory_order_relaxed);
  iree_status_t status = iree_hal_pool_acquire_reservations(
      operation->alloca.pool, operation->alloca.request_count,
      operation->alloca.requests, /*requester_frontier=*/NULL,
      operation->alloca.reservations, operation->alloca.acquire_infos,
      (iree_hal_pool_acquire_callback_t){
          .fn = iree_hal_task_queue_alloca_acquire_completed,
          .user_data = operation,
      },
      operation->alloca.reserve_flags);
  if (!iree_status_is_ok(status)) {
    // Rejected capture did not invoke completion. Consume the result locally
    // so a prepared capacity probe is aborted before the operation is retired.
    operation->alloca.acquire.status = status;
    operation->alloca.acquire.result = IREE_HAL_POOL_ACQUIRE_NONE;
    return iree_hal_task_queue_drain_alloca_acquired(operation);
  }
  if (iree_atomic_load(&operation->alloca.acquire.pending,
                       iree_memory_order_acquire) == 2) {
    iree_hal_task_queue_profile_force_software_defer(operation);
  }
  if (iree_atomic_fetch_sub(&operation->alloca.acquire.pending, 1,
                            iree_memory_order_acq_rel) == 1) {
    return iree_hal_task_queue_drain_alloca_acquired(operation);
  }
  return iree_ok_status();
}

// Handles an ALLOCA operation: acquires pool memory after user waits have
// resolved, parks on hidden memory-readiness waits when needed, and publishes
// the transient buffer backing only once the bytes are safe to use.
iree_status_t iree_hal_task_queue_issue_alloca(
    iree_hal_task_queue_op_t* operation) {
  iree_hal_task_queue_alloca_memory_wait_t* wait =
      operation->alloca.memory_wait;
  if (wait && wait->kind == IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_FRONTIER) {
    const iree_async_frontier_t* wait_frontier = wait->frontier.wait_frontier;
    wait->kind = IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
    return iree_hal_task_queue_drain_alloca_submit_reservations(
        operation, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT,
        /*reservation_failure_frontier=*/wait_frontier);
  }

  return iree_hal_task_queue_drain_alloca_acquire(operation);
}

// Handles a DEALLOCA operation after all wait semaphores are satisfied:
// decommits target-visible backing, returns immediately reusable reservations,
// then publishes dealloca completion.
void iree_hal_task_queue_issue_dealloca(iree_hal_task_queue_op_t* operation) {
  for (iree_host_size_t i = 0; i < operation->dealloca.buffer_count; ++i) {
    iree_hal_pool_t* source_pool = NULL;
    iree_hal_buffer_allocation_take_dealloca_reservation(
        operation->dealloca.transient_buffers[i], &source_pool,
        &operation->dealloca.reservations[i]);
    IREE_ASSERT_TRUE(source_pool == operation->dealloca.pool);
  }
  operation->dealloca.marks_owned = false;
  for (iree_host_size_t i = 0; i < operation->dealloca.buffer_count; ++i) {
    iree_hal_buffer_allocation_decommit(
        operation->dealloca.transient_buffers[i]);
  }
  // The waits and decommit have completed, so no prior access or target-side
  // deallocation effect remains. Publish usable capacity without a dependency
  // on this operation's later completion bookkeeping.
  iree_hal_pool_advise_asan_reservations(
      operation->dealloca.pool, operation->dealloca.buffer_count,
      operation->dealloca.reservations,
      IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED);
  iree_hal_pool_release_reservations(
      operation->dealloca.pool, operation->dealloca.buffer_count,
      operation->dealloca.reservations, /*death_frontier=*/NULL);
  iree_hal_task_queue_profile_record_memory_event(
      operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA,
      IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION, UINT32_MAX,
      operation->dealloca.reservations,
      /*frontier_entry_count=*/0);
  iree_hal_task_queue_profile_record_memory_event(
      operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
      IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION, UINT32_MAX,
      operation->dealloca.reservations,
      /*frontier_entry_count=*/0);
  iree_hal_task_queue_op_complete(operation);
}
