// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/drivers/vulkan/queue.h"
#include "iree/hal/drivers/vulkan/transient_buffer.h"
#include "iree/hal/pool_wait.h"

// Logical memory operations have no native FIFO entry. Each dependency hands
// ownership directly to its continuation; only the slab provider can defer a
// native allocation. A submitting claim prevents an inline callback from
// consuming captured arrays before the producing API returns.
typedef enum iree_hal_vulkan_memory_stage_e {
  IREE_HAL_VULKAN_MEMORY_STAGE_CAPTURED = 0,
  IREE_HAL_VULKAN_MEMORY_STAGE_DEPENDENCIES,
  IREE_HAL_VULKAN_MEMORY_STAGE_ACQUIRE,
  IREE_HAL_VULKAN_MEMORY_STAGE_ACQUIRED,
  IREE_HAL_VULKAN_MEMORY_STAGE_REUSE,
} iree_hal_vulkan_memory_stage_t;

typedef enum iree_hal_vulkan_memory_wait_e {
  IREE_HAL_VULKAN_MEMORY_WAIT_NONE = 0,
  IREE_HAL_VULKAN_MEMORY_WAIT_DEPENDENCIES,
  IREE_HAL_VULKAN_MEMORY_WAIT_CAPACITY,
  IREE_HAL_VULKAN_MEMORY_WAIT_REUSE,
} iree_hal_vulkan_memory_wait_t;

typedef struct iree_hal_vulkan_memory_operation_t {
  // Capture, accepted-list, and currently executing continuation ownership.
  iree_hal_resource_t resource;
  // Allocator copied so final capture destruction never dereferences the queue.
  iree_allocator_t host_allocator;
  // Previous accepted operation, protected by queue->memory_mutex.
  struct iree_hal_vulkan_memory_operation_t* previous;
  // Next accepted operation, protected by queue->memory_mutex.
  struct iree_hal_vulkan_memory_operation_t* next;
  // Queue and its native namespace, borrowed through terminal publication.
  iree_hal_vulkan_queue_t* queue;
  // Exact caller-selected pool, borrowed through the allocation epochs.
  iree_hal_pool_t* pool;
  // Captured prerequisites retained through completion.
  iree_hal_semaphore_list_t waits;
  // Captured result timepoints retained through publication.
  iree_hal_semaphore_list_t signals;
  // Timepoint storage registered once for each captured prerequisite.
  iree_async_semaphore_timepoint_t* timepoints;
  // Caller and callback claims on the current asynchronous boundary.
  iree_atomic_int32_t pending_count;
  // Coalesces inline callbacks/cancellation into one directly executing pump.
  iree_atomic_int32_t run_count;
  // Requested by queue failure/teardown; cancellation runs in the pump.
  iree_atomic_int32_t cancel_requested;
  // Active cancellable boundary. Native allocation itself is not cancellable.
  iree_hal_vulkan_memory_wait_t wait_kind;
  // Terminal operations ignore a final coalesced callback/cancellation request.
  bool finished;
  // First dependency or acquisition failure, owned until terminal publication.
  iree_atomic_intptr_t failure;
  // Next continuation after the active asynchronous boundary.
  iree_hal_vulkan_memory_stage_t stage;
  // True for a logical return, false for acquisition.
  bool releasing;
  // Number of captured buffers and reservation records.
  iree_host_size_t count;
  // Retained transient allocation roots.
  iree_hal_buffer_t** buffers;
  // Canonical native allocation requests, absent for logical return.
  iree_hal_pool_reservation_request_t* requests;
  // Reservation ownership transferred at commit or logical return.
  iree_hal_pool_reservation_t* reservations;
  // Exact acquired histories retained through commitment or rollback.
  iree_hal_pool_acquire_info_t* infos;
  // Temporary materialized views released before terminal publication.
  iree_hal_buffer_t** backing_buffers;
  // Number of deallocation marks held by this transaction.
  iree_host_size_t marked_count;
  // The operation owns acquired reservations until attachment or rollback.
  bool reservations_held;
  // Result supplied by the last acquisition callback.
  iree_hal_pool_acquire_result_t acquire_result;
  // Exact observed prerequisite history, when representable.
  iree_hal_vulkan_queue_frontier_t frontier;
  // False when observed history cannot be used to prove early reuse.
  bool frontier_exact;
  // Cold capacity subscription, allocated only after a capacity miss.
  iree_hal_pool_wait_t* capacity_wait;
  // Capacity was observed before the acquisition currently in progress.
  bool capacity_prepared;
  // Cold union of dependencies on acquired but not yet reusable storage.
  iree_async_frontier_t* reuse_frontier;
  // Embedded subscription to the acquired storage's exact reuse history.
  iree_async_frontier_waiter_t reuse_waiter;
  // Borrowed recorder enclosed by the application's profiling session.
  iree_hal_vulkan_profile_recorder_t* recorder;
  // Host queue event captured independently of native FIFO epochs.
  iree_hal_vulkan_profile_queue_event_info_t event;
} iree_hal_vulkan_memory_operation_t;

static void iree_hal_vulkan_memory_profile(
    iree_hal_vulkan_memory_operation_t* operation,
    iree_hal_profile_memory_event_type_t type,
    iree_hal_profile_memory_event_flags_t flags, uint32_t result) {
  if (!iree_hal_vulkan_profile_recorder_is_enabled(
          operation->recorder, IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS)) {
    return;
  }
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(operation->pool, &stats);
  for (iree_host_size_t i = 0; i < operation->count; ++i) {
    iree_hal_profile_memory_event_t event =
        iree_hal_profile_memory_event_default();
    event.type = type;
    event.flags = IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION |
                  IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_STATS | flags;
    event.result = result;
    event.allocation_id =
        iree_hal_buffer_allocation_profile(operation->buffers[i]).id;
    event.pool_id = (uint64_t)(uintptr_t)operation->pool;
    event.submission_id = operation->event.submission_id;
    event.physical_device_ordinal =
        operation->event.scope.physical_device_ordinal;
    event.queue_ordinal = operation->event.scope.queue_ordinal;
    event.memory_type = iree_hal_buffer_memory_type(operation->buffers[i]);
    event.buffer_usage = iree_hal_buffer_allowed_usage(operation->buffers[i]);
    event.length = iree_hal_buffer_allocation_size(operation->buffers[i]);
    event.alignment = operation->releasing
                          ? 1
                          : iree_max(operation->requests[i].min_alignment, 1);
    if (operation->reuse_frontier) {
      event.frontier_entry_count = operation->reuse_frontier->entry_count;
    }
    if (operation->reservations_held) {
      event.flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_RESERVATION;
      event.backing_id = operation->reservations[i].block_handle;
      event.offset = operation->reservations[i].offset;
      if (type != IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA &&
          type != IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA) {
        event.length = operation->reservations[i].byte_length;
      }
    }
    event.pool_bytes_committed = stats.bytes_committed;
    event.pool_bytes_reserved = stats.bytes_reserved;
    event.pool_bytes_free = stats.bytes_free;
    event.pool_budget_limit = stats.budget_limit;
    event.pool_reservation_count = stats.reservation_count;
    event.pool_slab_count = stats.slab_count;
    iree_hal_vulkan_profile_recorder_append_memory_event(operation->recorder,
                                                         &event, NULL);
  }
}

static void iree_hal_vulkan_memory_operation_destroy(
    iree_hal_resource_t* base_resource) {
  iree_hal_vulkan_memory_operation_t* operation =
      (iree_hal_vulkan_memory_operation_t*)base_resource;
  iree_hal_vulkan_queue_t* queue = operation->queue;
  iree_hal_pool_wait_free(operation->capacity_wait);
  iree_allocator_free(operation->host_allocator, operation->reuse_frontier);
  iree_hal_semaphore_list_release(operation->waits);
  iree_hal_semaphore_list_release(operation->signals);
  iree_allocator_free(operation->host_allocator, operation);
  iree_slim_mutex_lock(&queue->memory_mutex);
  if (--queue->memory_operation_count == 0) {
    // Teardown reacquires this mutex before destroying the notification.
    iree_notification_post(&queue->memory_notification, IREE_ALL_WAITERS);
  }
  iree_slim_mutex_unlock(&queue->memory_mutex);
}

static const iree_hal_resource_vtable_t
    iree_hal_vulkan_memory_operation_vtable = {
        .destroy = iree_hal_vulkan_memory_operation_destroy,
};

static void iree_hal_vulkan_memory_complete(
    iree_hal_vulkan_memory_operation_t* operation, iree_status_t status) {
  if (operation->recorder && !operation->event.ready_host_time_ns) {
    iree_hal_vulkan_profile_recorder_append_queue_event(
        operation->recorder, &operation->event, NULL);
  }
  if (operation->capacity_prepared) {
    iree_hal_pool_wait_abort(operation->capacity_wait);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_memory_profile(
        operation,
        operation->releasing ? IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA
                             : IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA,
        0, iree_status_code(status));
  }
  if (operation->reservations_held) {
    iree_hal_vulkan_memory_profile(
        operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE, 0,
        iree_status_code(status));
    for (iree_host_size_t i = 0; i < operation->count; ++i) {
      iree_hal_pool_release_reservations(operation->pool, 1,
                                         &operation->reservations[i],
                                         operation->infos[i].reuse_frontier);
    }
  }
  for (iree_host_size_t i = 0; i < operation->marked_count; ++i) {
    iree_hal_buffer_allocation_abort_dealloca(operation->buffers[i]);
  }
  if (operation->backing_buffers) {
    for (iree_host_size_t i = 0; i < operation->count; ++i) {
      iree_hal_buffer_release(operation->backing_buffers[i]);
    }
  }
  for (iree_host_size_t i = 0; i < operation->count; ++i) {
    iree_hal_buffer_release(operation->buffers[i]);
  }
  operation->finished = true;
  iree_hal_vulkan_queue_t* queue = operation->queue;
  iree_slim_mutex_lock(&queue->memory_mutex);
  if (operation->previous) {
    operation->previous->next = operation->next;
  } else {
    queue->memory_head = operation->next;
  }
  if (operation->next) {
    operation->next->previous = operation->previous;
  }
  iree_slim_mutex_unlock(&queue->memory_mutex);
  // A signal callback may immediately destroy the source pool. All buffer,
  // reservation, capacity, and profiling accesses precede that publication.
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(
        operation->signals,
        operation->frontier_exact
            ? iree_async_fixed_frontier_as_const_frontier(&operation->frontier)
            : NULL);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(operation->signals, status);
  }
  iree_hal_resource_release(operation);
}

static void iree_hal_vulkan_memory_record_failure(
    iree_hal_vulkan_memory_operation_t* operation, iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    intptr_t expected = (intptr_t)iree_ok_status();
    if (!iree_atomic_compare_exchange_strong(
            &operation->failure, &expected, (intptr_t)status,
            iree_memory_order_release, iree_memory_order_relaxed)) {
      // The first dependency failure is the operation's terminal error.
      iree_status_free(status);
    }
  }
}

// Inline completion and cancellation request another iteration of the current
// pump; they never recursively execute a continuation or free its capture.
// Each caller owns an operation reference through the entire pump, including
// terminal publication that releases the accepted-list reference.
static void iree_hal_vulkan_memory_resume(
    iree_hal_vulkan_memory_operation_t* operation);

static void iree_hal_vulkan_memory_schedule(
    iree_hal_vulkan_memory_operation_t* operation) {
  if (iree_atomic_fetch_add(&operation->run_count, 1,
                            iree_memory_order_acq_rel) == 0) {
    do {
      iree_hal_vulkan_memory_resume(operation);
    } while (iree_atomic_fetch_sub(&operation->run_count, 1,
                                   iree_memory_order_acq_rel) != 1);
  }
}

static void iree_hal_vulkan_memory_ready(void* user_data,
                                         iree_status_t status) {
  iree_hal_vulkan_memory_operation_t* operation = user_data;
  iree_hal_resource_retain(operation);
  const bool failed = !iree_status_is_ok(status);
  iree_hal_vulkan_memory_record_failure(operation, status);
  if (iree_atomic_fetch_sub(&operation->pending_count, 1,
                            iree_memory_order_acq_rel) == 1 ||
      failed) {
    iree_hal_vulkan_memory_schedule(operation);
  }
  iree_hal_resource_release(operation);
}

static void iree_hal_vulkan_memory_acquired(
    void* user_data, iree_status_t status,
    iree_hal_pool_acquire_result_t result) {
  iree_hal_vulkan_memory_operation_t* operation = user_data;
  operation->acquire_result = result;
  operation->reservations_held =
      iree_status_is_ok(status) &&
      (result == IREE_HAL_POOL_ACQUIRE_OK ||
       result == IREE_HAL_POOL_ACQUIRE_OK_FRESH ||
       result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  iree_hal_vulkan_memory_ready(operation, status);
}

static bool iree_hal_vulkan_memory_yield(
    iree_hal_vulkan_memory_operation_t* operation) {
  return iree_atomic_fetch_sub(&operation->pending_count, 1,
                               iree_memory_order_acq_rel) != 1;
}

static void iree_hal_vulkan_memory_resolve_frontier(
    iree_hal_vulkan_memory_operation_t* operation) {
  operation->frontier_exact = true;
  iree_async_frontier_t* target =
      iree_async_fixed_frontier_as_frontier(&operation->frontier);
  iree_async_frontier_initialize(target, 0);
  for (iree_host_size_t i = 0; i < operation->waits.count; ++i) {
    iree_async_semaphore_t* semaphore =
        (iree_async_semaphore_t*)operation->waits.semaphores[i];
    iree_hal_vulkan_queue_frontier_t source_storage;
    iree_async_frontier_t* source =
        iree_async_fixed_frontier_as_frontier(&source_storage);
    const uint8_t count = iree_async_semaphore_query_frontier(
        semaphore, source, IREE_HAL_VULKAN_QUEUE_FRONTIER_CAPACITY);
    if (count > IREE_HAL_VULKAN_QUEUE_FRONTIER_CAPACITY ||
        iree_async_semaphore_is_value_tainted(
            semaphore, operation->waits.payload_values[i]) ||
        !iree_async_frontier_merge(
            target, IREE_HAL_VULKAN_QUEUE_FRONTIER_CAPACITY, source)) {
      operation->frontier_exact = false;
    }
  }
}

static iree_status_t iree_hal_vulkan_memory_commit(
    iree_hal_vulkan_memory_operation_t* operation) {
  iree_status_t status = iree_hal_pool_materialize_reservations(
      operation->pool, operation->count, operation->requests,
      operation->reservations, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
      operation->backing_buffers);
  // Legacy sources carry no prepared contract; qualify native ownership once
  // before publishing their bindings. Prepared contracts already prove it.
  for (iree_host_size_t i = 0;
       !operation->pool->memory_contract && i < operation->count &&
       iree_status_is_ok(status);
       ++i) {
    iree_hal_buffer_t* backing = NULL;
    status = iree_hal_vulkan_buffer_resolve_backing(
        operation->backing_buffers[i], &backing);
    if (iree_status_is_ok(status) &&
        iree_hal_buffer_allocation_placement(backing).device !=
            (iree_hal_device_t*)operation->queue->device) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "allocation pool belongs to another Vulkan device");
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_memory_profile(
        operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE, 0,
        operation->acquire_result);
    iree_hal_pool_advise_asan_reservations(
        operation->pool, operation->count, operation->reservations,
        IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED);
    for (iree_host_size_t i = 0; i < operation->count; ++i) {
      iree_hal_vulkan_transient_buffer_attach_reservation(
          operation->buffers[i], operation->pool, &operation->reservations[i]);
      iree_hal_vulkan_transient_buffer_stage_backing(
          operation->buffers[i], operation->backing_buffers[i]);
      iree_hal_vulkan_transient_buffer_commit(operation->buffers[i]);
      const iree_async_frontier_t* history = operation->infos[i].reuse_frontier;
      if (history &&
          !iree_async_frontier_merge(
              iree_async_fixed_frontier_as_frontier(&operation->frontier),
              IREE_HAL_VULKAN_QUEUE_FRONTIER_CAPACITY, history)) {
        operation->frontier_exact = false;
      }
    }
    iree_hal_vulkan_memory_profile(
        operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA, 0,
        UINT32_MAX);
    operation->reservations_held = false;
  }
  return status;
}

static void iree_hal_vulkan_memory_return(
    iree_hal_vulkan_memory_operation_t* operation) {
  for (iree_host_size_t i = 0; i < operation->count; ++i) {
    iree_hal_pool_t* source_pool = NULL;
    iree_hal_buffer_allocation_take_dealloca_reservation(
        operation->buffers[i], &source_pool, &operation->reservations[i]);
    iree_hal_buffer_allocation_decommit(operation->buffers[i]);
  }
  operation->marked_count = 0;
  iree_hal_pool_advise_asan_reservations(
      operation->pool, operation->count, operation->reservations,
      IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED);
  operation->reservations_held = true;
  iree_hal_vulkan_memory_profile(
      operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA, 0,
      UINT32_MAX);
  iree_hal_vulkan_memory_profile(
      operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE, 0,
      UINT32_MAX);
  // Every explicit predecessor has completed; no native FIFO epoch is needed
  // for these already-retired ranges.
  iree_hal_pool_release_reservations(operation->pool, operation->count,
                                     operation->reservations, NULL);
  operation->reservations_held = false;
}

static iree_status_t iree_hal_vulkan_memory_prepare_reuse(
    iree_hal_vulkan_memory_operation_t* operation) {
  iree_host_size_t storage_size = 0;
  IREE_RETURN_IF_ERROR(iree_async_frontier_size(UINT8_MAX, &storage_size));
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(operation->queue->host_allocator, storage_size,
                            (void**)&operation->reuse_frontier));
  iree_async_frontier_initialize(operation->reuse_frontier, 0);
  for (iree_host_size_t i = 0; i < operation->count; ++i) {
    if (operation->infos[i].result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT &&
        !iree_async_frontier_merge(operation->reuse_frontier, UINT8_MAX,
                                   operation->infos[i].reuse_frontier)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "Vulkan allocation reuse exceeds %u axes",
                              UINT8_MAX);
    }
  }
  return iree_ok_status();
}

static void iree_hal_vulkan_memory_arm_dependencies(
    iree_hal_vulkan_memory_operation_t* operation);

// The pump owns registration, cancellation, and reuse of every wait descriptor.
// Callbacks only publish their result and request another pump iteration.
static void iree_hal_vulkan_memory_cancel_wait(
    iree_hal_vulkan_memory_operation_t* operation) {
  const iree_hal_vulkan_memory_wait_t kind = operation->wait_kind;
  operation->wait_kind = IREE_HAL_VULKAN_MEMORY_WAIT_NONE;
  switch (kind) {
    case IREE_HAL_VULKAN_MEMORY_WAIT_NONE:
      break;
    case IREE_HAL_VULKAN_MEMORY_WAIT_DEPENDENCIES:
      for (iree_host_size_t i = 0; i < operation->waits.count; ++i) {
        if (iree_async_semaphore_cancel_timepoint(
                (iree_async_semaphore_t*)operation->waits.semaphores[i],
                &operation->timepoints[i])) {
          iree_hal_vulkan_memory_ready(operation, iree_ok_status());
        }
      }
      break;
    case IREE_HAL_VULKAN_MEMORY_WAIT_CAPACITY:
      iree_hal_pool_wait_cancel(operation->capacity_wait);
      break;
    case IREE_HAL_VULKAN_MEMORY_WAIT_REUSE:
      if (iree_async_frontier_tracker_cancel_wait(
              operation->pool->frontier_tracker, &operation->reuse_waiter)) {
        iree_hal_vulkan_memory_ready(operation, iree_ok_status());
      }
      break;
  }
}

static void iree_hal_vulkan_memory_resume(
    iree_hal_vulkan_memory_operation_t* operation) {
  if (operation->finished) {
    return;
  }
  for (;;) {
    const bool cancelled = iree_atomic_load(&operation->cancel_requested,
                                            iree_memory_order_acquire) != 0;
    if (cancelled ||
        iree_atomic_load(&operation->failure, iree_memory_order_acquire)) {
      iree_hal_vulkan_memory_cancel_wait(operation);
    }
    if (iree_atomic_load(&operation->pending_count,
                         iree_memory_order_acquire) != 0) {
      return;
    }
    operation->wait_kind = IREE_HAL_VULKAN_MEMORY_WAIT_NONE;
    iree_status_t status = (iree_status_t)iree_atomic_exchange(
        &operation->failure, (intptr_t)iree_ok_status(),
        iree_memory_order_acq_rel);
    if (iree_status_is_ok(status)) {
      status = iree_status_clone((iree_status_t)iree_atomic_load(
          &operation->queue->failure_status, iree_memory_order_acquire));
    }
    if (iree_status_is_ok(status) && cancelled) {
      status = iree_status_from_code(IREE_STATUS_CANCELLED);
    }
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_memory_complete(operation, status);
      return;
    }
    switch (operation->stage) {
      case IREE_HAL_VULKAN_MEMORY_STAGE_CAPTURED:
        operation->stage = IREE_HAL_VULKAN_MEMORY_STAGE_DEPENDENCIES;
        operation->wait_kind = IREE_HAL_VULKAN_MEMORY_WAIT_DEPENDENCIES;
        iree_hal_vulkan_memory_arm_dependencies(operation);
        break;
      case IREE_HAL_VULKAN_MEMORY_STAGE_DEPENDENCIES:
        iree_hal_vulkan_memory_resolve_frontier(operation);
        if (operation->recorder) {
          operation->event.ready_host_time_ns = iree_time_now();
          iree_hal_vulkan_profile_recorder_append_queue_event(
              operation->recorder, &operation->event, NULL);
        }
        if (operation->releasing) {
          iree_hal_vulkan_memory_return(operation);
          iree_hal_vulkan_memory_complete(operation, iree_ok_status());
          return;
        }
        operation->stage = IREE_HAL_VULKAN_MEMORY_STAGE_ACQUIRE;
        break;
      case IREE_HAL_VULKAN_MEMORY_STAGE_ACQUIRE:
        operation->stage = IREE_HAL_VULKAN_MEMORY_STAGE_ACQUIRED;
        iree_atomic_store(&operation->pending_count, 2,
                          iree_memory_order_relaxed);
        status = iree_hal_pool_acquire_reservations(
            operation->pool, operation->count, operation->requests,
            operation->frontier_exact
                ? iree_async_fixed_frontier_as_const_frontier(
                      &operation->frontier)
                : NULL,
            operation->reservations, operation->infos,
            (iree_hal_pool_acquire_callback_t){
                .fn = iree_hal_vulkan_memory_acquired, .user_data = operation},
            IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER);
        if (!iree_status_is_ok(status)) {
          iree_hal_vulkan_memory_acquired(operation, status,
                                          IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
        }
        if (iree_hal_vulkan_memory_yield(operation)) {
          return;
        }
        break;
      case IREE_HAL_VULKAN_MEMORY_STAGE_ACQUIRED:
        iree_hal_vulkan_memory_profile(
            operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE, 0,
            operation->acquire_result);
        if (operation->reservations_held) {
          if (operation->capacity_prepared) {
            iree_hal_pool_wait_abort(operation->capacity_wait);
            operation->capacity_prepared = false;
          }
          if (operation->acquire_result !=
              IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT) {
            status = iree_hal_vulkan_memory_commit(operation);
            iree_hal_vulkan_memory_complete(operation, status);
            return;
          }
          status = iree_hal_vulkan_memory_prepare_reuse(operation);
          if (!iree_status_is_ok(status)) {
            iree_hal_vulkan_memory_complete(operation, status);
            return;
          }
          iree_hal_vulkan_memory_profile(
              operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT,
              IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_FRONTIER, UINT32_MAX);
          operation->stage = IREE_HAL_VULKAN_MEMORY_STAGE_REUSE;
          operation->wait_kind = IREE_HAL_VULKAN_MEMORY_WAIT_REUSE;
          iree_atomic_store(&operation->pending_count, 2,
                            iree_memory_order_relaxed);
          status = iree_async_frontier_tracker_wait(
              operation->pool->frontier_tracker, operation->reuse_frontier,
              iree_hal_vulkan_memory_ready, operation,
              &operation->reuse_waiter);
          if (!iree_status_is_ok(status)) {
            iree_hal_vulkan_memory_ready(operation, status);
          }
          if (iree_hal_vulkan_memory_yield(operation)) {
            return;
          }
        } else if (!operation->capacity_prepared) {
          if (!operation->capacity_wait) {
            status = iree_hal_pool_wait_allocate(
                operation->pool, operation->queue->host_allocator,
                &operation->capacity_wait);
          }
          if (!iree_status_is_ok(status)) {
            iree_hal_vulkan_memory_complete(operation, status);
            return;
          }
          iree_hal_pool_wait_prepare(operation->capacity_wait);
          operation->capacity_prepared = true;
          operation->stage = IREE_HAL_VULKAN_MEMORY_STAGE_ACQUIRE;
        } else {
          iree_hal_vulkan_memory_profile(
              operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT,
              IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_NOTIFICATION, UINT32_MAX);
          operation->capacity_prepared = false;
          operation->stage = IREE_HAL_VULKAN_MEMORY_STAGE_ACQUIRE;
          operation->wait_kind = IREE_HAL_VULKAN_MEMORY_WAIT_CAPACITY;
          iree_atomic_store(&operation->pending_count, 2,
                            iree_memory_order_relaxed);
          iree_hal_pool_wait_commit(
              operation->capacity_wait, iree_infinite_timeout(),
              (iree_hal_pool_wait_callback_t){
                  .fn = iree_hal_vulkan_memory_ready, .user_data = operation});
          if (iree_hal_vulkan_memory_yield(operation)) {
            return;
          }
        }
        break;
      case IREE_HAL_VULKAN_MEMORY_STAGE_REUSE:
        status = iree_hal_vulkan_memory_commit(operation);
        iree_hal_vulkan_memory_complete(operation, status);
        return;
    }
  }
}

static void iree_hal_vulkan_memory_dependency_ready(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_vulkan_memory_ready(user_data, status);
}

static void iree_hal_vulkan_memory_arm_dependencies(
    iree_hal_vulkan_memory_operation_t* operation) {
  iree_atomic_store(&operation->pending_count, operation->waits.count + 1,
                    iree_memory_order_relaxed);
  for (iree_host_size_t i = 0; i < operation->waits.count; ++i) {
    operation->timepoints[i].callback = iree_hal_vulkan_memory_dependency_ready;
    operation->timepoints[i].user_data = operation;
    // The semaphore dispatches an already-failed prerequisite inline and
    // returns OK after transferring the error to that callback.
    iree_status_t status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)operation->waits.semaphores[i],
        operation->waits.payload_values[i], &operation->timepoints[i]);
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_memory_ready(operation, status);
    }
  }
  if (iree_hal_vulkan_memory_yield(operation)) {
    operation->event.dependency_strategy =
        IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER;
    operation->event.flags |=
        IREE_HAL_PROFILE_QUEUE_EVENT_FLAG_SOFTWARE_DEFERRED;
  } else if (operation->waits.count) {
    operation->event.dependency_strategy =
        IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_INLINE;
  }
}

static void iree_hal_vulkan_memory_submit(
    iree_hal_vulkan_memory_operation_t* operation) {
  iree_hal_vulkan_queue_t* queue = operation->queue;
  iree_slim_mutex_lock(&queue->memory_mutex);
  iree_hal_resource_retain(operation);
  operation->next = queue->memory_head;
  if (queue->memory_head) {
    queue->memory_head->previous = operation;
  }
  queue->memory_head = operation;
  if (iree_atomic_load(&queue->stop_requested, iree_memory_order_acquire) ||
      !iree_status_is_ok((iree_status_t)iree_atomic_load(
          &queue->failure_status, iree_memory_order_acquire))) {
    iree_atomic_store(&operation->cancel_requested, 1,
                      iree_memory_order_release);
  }
  iree_slim_mutex_unlock(&queue->memory_mutex);
  iree_hal_vulkan_memory_schedule(operation);
  iree_hal_resource_release(operation);
}

void iree_hal_vulkan_queue_cancel_memory(iree_hal_vulkan_queue_t* queue) {
  // Teardown/fatal failure is cold. Take one operation at a time without an
  // allocation or a callback under the linkage lock.
  for (;;) {
    iree_slim_mutex_lock(&queue->memory_mutex);
    iree_hal_vulkan_memory_operation_t* operation = queue->memory_head;
    while (operation && iree_atomic_load(&operation->cancel_requested,
                                         iree_memory_order_relaxed)) {
      operation = operation->next;
    }
    if (operation) {
      iree_hal_resource_retain(operation);
      iree_atomic_store(&operation->cancel_requested, 1,
                        iree_memory_order_release);
    }
    iree_slim_mutex_unlock(&queue->memory_mutex);
    if (!operation) {
      break;
    }
    iree_hal_vulkan_memory_schedule(operation);
    iree_hal_resource_release(operation);
  }
}

static bool iree_hal_vulkan_queue_memory_is_idle(void* user_data) {
  iree_hal_vulkan_queue_t* queue = user_data;
  iree_slim_mutex_lock(&queue->memory_mutex);
  const bool idle = queue->memory_operation_count == 0;
  iree_slim_mutex_unlock(&queue->memory_mutex);
  return idle;
}

void iree_hal_vulkan_queue_join_memory(iree_hal_vulkan_queue_t* queue) {
  iree_notification_await(&queue->memory_notification,
                          iree_hal_vulkan_queue_memory_is_idle, queue,
                          iree_infinite_timeout());
}

static iree_status_t iree_hal_vulkan_memory_operation_create(
    iree_hal_vulkan_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count, bool releasing,
    iree_hal_vulkan_memory_operation_t** out_operation) {
  IREE_RETURN_IF_ERROR(iree_status_clone((iree_status_t)iree_atomic_load(
      &queue->failure_status, iree_memory_order_acquire)));
  if (!queue->frontier_tracker) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan queue frontier is not assigned");
  }
  if (iree_atomic_load(&queue->stop_requested, iree_memory_order_acquire)) {
    return iree_make_status(IREE_STATUS_CANCELLED,
                            "Vulkan queue is shutting down");
  }
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_validate_semaphore_list(
      queue, waits, IREE_SV("wait")));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_validate_semaphore_list(
      queue, signals, IREE_SV("signal")));
  if (waits.count >= INT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many allocation prerequisites");
  }
  iree_host_size_t total_size = 0, waits_offset = 0, wait_values_offset = 0,
                   signals_offset = 0, signal_values_offset = 0,
                   timepoints_offset = 0, buffers_offset = 0,
                   requests_offset = 0, reservations_offset = 0,
                   infos_offset = 0, backing_offset = 0;
  const iree_host_size_t acquire_count = releasing ? 0 : count;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_vulkan_memory_operation_t), &total_size,
      IREE_STRUCT_FIELD(waits.count, iree_hal_semaphore_t*, &waits_offset),
      IREE_STRUCT_FIELD(waits.count, uint64_t, &wait_values_offset),
      IREE_STRUCT_FIELD(signals.count, iree_hal_semaphore_t*, &signals_offset),
      IREE_STRUCT_FIELD(signals.count, uint64_t, &signal_values_offset),
      IREE_STRUCT_FIELD(waits.count, iree_async_semaphore_timepoint_t,
                        &timepoints_offset),
      IREE_STRUCT_FIELD(count, iree_hal_buffer_t*, &buffers_offset),
      IREE_STRUCT_FIELD(acquire_count, iree_hal_pool_reservation_request_t,
                        &requests_offset),
      IREE_STRUCT_FIELD(count, iree_hal_pool_reservation_t,
                        &reservations_offset),
      IREE_STRUCT_FIELD(acquire_count, iree_hal_pool_acquire_info_t,
                        &infos_offset),
      IREE_STRUCT_FIELD(acquire_count, iree_hal_buffer_t*, &backing_offset)));
  iree_hal_vulkan_memory_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(queue->host_allocator, total_size,
                                             (void**)&operation));
  iree_hal_resource_initialize(&iree_hal_vulkan_memory_operation_vtable,
                               &operation->resource);
  operation->host_allocator = queue->host_allocator;
  uint8_t* storage = (uint8_t*)operation;
  operation->queue = queue;
  iree_slim_mutex_lock(&queue->memory_mutex);
  ++queue->memory_operation_count;
  iree_slim_mutex_unlock(&queue->memory_mutex);
  operation->count = count;
  operation->releasing = releasing;
  operation->waits = (iree_hal_semaphore_list_t){
      waits.count, (iree_hal_semaphore_t**)(storage + waits_offset),
      (uint64_t*)(storage + wait_values_offset)};
  operation->signals = (iree_hal_semaphore_list_t){
      signals.count, (iree_hal_semaphore_t**)(storage + signals_offset),
      (uint64_t*)(storage + signal_values_offset)};
  for (iree_host_size_t i = 0; i < waits.count; ++i) {
    operation->waits.semaphores[i] = waits.semaphores[i];
    operation->waits.payload_values[i] = waits.payload_values[i];
    iree_hal_semaphore_retain(waits.semaphores[i]);
  }
  for (iree_host_size_t i = 0; i < signals.count; ++i) {
    operation->signals.semaphores[i] = signals.semaphores[i];
    operation->signals.payload_values[i] = signals.payload_values[i];
    iree_hal_semaphore_retain(signals.semaphores[i]);
  }
  operation->timepoints =
      (iree_async_semaphore_timepoint_t*)(storage + timepoints_offset);
  operation->buffers = (iree_hal_buffer_t**)(storage + buffers_offset);
  operation->reservations =
      (iree_hal_pool_reservation_t*)(storage + reservations_offset);
  if (!releasing) {
    operation->requests =
        (iree_hal_pool_reservation_request_t*)(storage + requests_offset);
    operation->infos = (iree_hal_pool_acquire_info_t*)(storage + infos_offset);
    operation->backing_buffers =
        (iree_hal_buffer_t**)(storage + backing_offset);
  }
  operation->recorder = queue->profile_recorder;
  operation->event = iree_hal_vulkan_profile_queue_event_info_default();
  operation->event.scope = queue->profile_scope;
  operation->event.wait_count = (uint32_t)iree_min(waits.count, UINT32_MAX);
  operation->event.signal_count = (uint32_t)iree_min(signals.count, UINT32_MAX);
  operation->event.operation_count = (uint32_t)iree_min(count, UINT32_MAX);
  if (operation->recorder) {
    operation->event.host_time_ns = iree_time_now();
    if (queue->profile_submission_counter) {
      operation->event.submission_id = (uint64_t)iree_atomic_fetch_add(
          queue->profile_submission_counter, 1, iree_memory_order_relaxed);
    }
  }
  *out_operation = operation;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_create_transient_buffer(
    iree_hal_vulkan_queue_t* queue, iree_hal_pool_t* pool,
    iree_host_size_t request_index,
    const iree_hal_pool_reservation_request_t* request,
    iree_hal_pool_reservation_request_t* out_canonical_request,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_pool_capabilities_t pool_capabilities;
  iree_hal_pool_query_capabilities(pool, &pool_capabilities);
  iree_hal_pool_reservation_request_t canonical_request = *request;
  if (!iree_device_size_checked_align(request->allocation_size, 4,
                                      &canonical_request.allocation_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan allocation size overflows dword alignment");
  }
  if (IREE_UNLIKELY(pool_capabilities.max_allocation_size != 0 &&
                    canonical_request.allocation_size >
                        pool_capabilities.max_allocation_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "Vulkan allocation request %" PRIhsz
                            " exceeds the source pool maximum allocation size",
                            request_index);
  }
  if (IREE_UNLIKELY(
          !iree_device_size_is_valid_alignment(request->min_alignment) ||
          request->min_alignment >
              pool_capabilities.max_allocation_alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan allocation request %" PRIhsz
                            " alignment exceeds the source pool maximum",
                            request_index);
  }

  iree_hal_buffer_placement_t placement = {
      .device = (iree_hal_device_t*)queue->device,
      .queue_family_affinity = pool_capabilities.queue_family_affinity,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS,
  };
  const iree_hal_memory_access_t allowed_access =
      pool->memory_contract ? pool->memory_contract->allowed_access
                            : pool_capabilities.allowed_access;
  const iree_hal_buffer_usage_t allowed_usage =
      pool->memory_contract ? pool->memory_contract->allowed_usage
                            : pool_capabilities.supported_usage;
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_transient_buffer_create(
      placement, pool_capabilities.memory_type, allowed_access, allowed_usage,
      canonical_request.allocation_size, request->allocation_size, pool,
      queue->host_allocator, &buffer));
  *out_canonical_request = canonical_request;
  *out_buffer = buffer;
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_queue_memory_alloca(
    iree_hal_queue_t* base_queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_pool_t* pool,
    iree_host_size_t count, const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** IREE_RESTRICT out_buffers) {
  iree_hal_vulkan_queue_t* queue = (iree_hal_vulkan_queue_t*)base_queue;
  iree_hal_vulkan_memory_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_memory_operation_create(
      queue, waits, signals, count, false, &operation));
  operation->pool = pool;
  operation->event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA;
  iree_status_t status = iree_ok_status();
  iree_host_size_t prepared_count = 0;
  for (; prepared_count < count && iree_status_is_ok(status);
       ++prepared_count) {
    status = iree_hal_vulkan_queue_create_transient_buffer(
        queue, pool, prepared_count, &requests[prepared_count],
        &operation->requests[prepared_count],
        &operation->buffers[prepared_count]);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      out_buffers[i] = operation->buffers[i];
      iree_hal_buffer_retain(out_buffers[i]);
      operation->event.payload_length =
          iree_min(UINT64_MAX - requests[i].allocation_size,
                   operation->event.payload_length) +
          requests[i].allocation_size;
    }
    operation->event.allocation_id =
        count == 1 ? iree_hal_buffer_allocation_profile(out_buffers[0]).id : 0;
    iree_hal_vulkan_memory_submit(operation);
  } else {
    for (iree_host_size_t i = 0; i < prepared_count; ++i) {
      iree_hal_buffer_release(operation->buffers[i]);
    }
    iree_hal_resource_release(operation);
  }
  return status;
}

iree_status_t iree_hal_vulkan_queue_memory_dealloca(
    iree_hal_queue_t* base_queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    iree_hal_buffer_t* const* buffers) {
  iree_hal_vulkan_queue_t* queue = (iree_hal_vulkan_queue_t*)base_queue;
  iree_hal_vulkan_memory_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_memory_operation_create(
      queue, waits, signals, count, true, &operation));
  operation->event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count; ++i) {
    operation->buffers[i] = buffers[i];
    iree_hal_buffer_retain(buffers[i]);
    const uint64_t size = iree_hal_buffer_allocation_size(buffers[i]);
    operation->event.payload_length =
        iree_min(UINT64_MAX - size, operation->event.payload_length) + size;
  }
  operation->event.allocation_id =
      count == 1 ? iree_hal_buffer_allocation_profile(buffers[0]).id : 0;
  while (operation->marked_count < count && iree_status_is_ok(status)) {
    iree_hal_pool_t* source_pool = NULL;
    status = iree_hal_buffer_allocation_begin_dealloca(
        buffers[operation->marked_count], &source_pool);
    if (iree_status_is_ok(status)) {
      ++operation->marked_count;
      if (!operation->pool) {
        operation->pool = source_pool;
      }
      if (source_pool != operation->pool) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "Vulkan deallocation spans multiple source pools");
      }
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_memory_submit(operation);
  } else {
    for (iree_host_size_t i = 0; i < operation->marked_count; ++i) {
      iree_hal_buffer_allocation_abort_dealloca(buffers[i]);
    }
    for (iree_host_size_t i = 0; i < count; ++i) {
      iree_hal_buffer_release(buffers[i]);
    }
    iree_hal_resource_release(operation);
  }
  return status;
}
