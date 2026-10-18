// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/queue_admission.h"

#include "iree/async/operations/scheduling.h"
#include "iree/async/semaphore.h"
#include "iree/hal/drivers/amd/xdna/queue_state.h"
#include "iree/hal/drivers/amd/xdna/semaphore.h"

// Arena-owned registration for one deferred semaphore wait.
typedef struct iree_hal_amd_xdna_wait_entry_t {
  // Intrusive timepoint owned by the semaphore until its callback fires.
  iree_async_semaphore_timepoint_t timepoint;
  // Operation awaiting all registered timepoints.
  iree_hal_amd_xdna_operation_t* operation;
  // Retained semaphore kept live until this callback returns.
  iree_hal_semaphore_t* semaphore;
} iree_hal_amd_xdna_wait_entry_t;

// Returns both arenas before releasing the device that owns their pools.
static void iree_hal_amd_xdna_operation_deallocate(
    iree_hal_amd_xdna_operation_t* operation) {
  iree_hal_device_t* device = operation->device;
  iree_hal_amd_xdna_queue_capture_t capture = operation->capture;
  operation = NULL;
  iree_hal_amd_xdna_queue_capture_deinitialize(&capture);
  iree_hal_device_release(device);
}

void iree_hal_amd_xdna_operation_register_producers(
    iree_hal_amd_xdna_operation_t* operation) {
  if ((operation->kind != IREE_HAL_AMD_XDNA_OPERATION_DISPATCH &&
       operation->kind != IREE_HAL_AMD_XDNA_OPERATION_EXECUTE) ||
      operation->native.producers_indexed) {
    return;
  }
  for (iree_hal_amd_xdna_producer_entry_t* entry =
           operation->native.producer_entries;
       entry; entry = entry->next) {
    iree_hal_amd_xdna_queue_producer_index_insert(
        &operation->queue->producer_index, &entry->index_node);
  }
  operation->native.producers_indexed = true;
}

// Removes every signal owned by |operation| and returns its targeted consumer
// list. The returned operations remain owned by their captured submissions.
iree_hal_amd_xdna_operation_t* iree_hal_amd_xdna_operation_unregister_producers(
    iree_hal_amd_xdna_operation_t* operation) {
  if ((operation->kind != IREE_HAL_AMD_XDNA_OPERATION_DISPATCH &&
       operation->kind != IREE_HAL_AMD_XDNA_OPERATION_EXECUTE) ||
      !operation->native.producers_indexed) {
    return NULL;
  }
  for (iree_hal_amd_xdna_producer_entry_t* entry =
           operation->native.producer_entries;
       entry; entry = entry->next) {
    iree_hal_amd_xdna_queue_producer_index_erase(
        &operation->queue->producer_index, &entry->index_node);
  }
  operation->native.producers_indexed = false;
  iree_hal_amd_xdna_operation_t* waiters_head =
      operation->native.acceptance_waiters_head;
  operation->native.acceptance_waiters_head = NULL;
  operation->native.acceptance_waiters_tail = NULL;
  return waiters_head;
}

void iree_hal_amd_xdna_operation_wake_acceptance_waiters(
    iree_hal_amd_xdna_operation_t* operation) {
  while (operation) {
    iree_hal_amd_xdna_operation_t* next = operation->next;
    operation->next = NULL;
    iree_hal_amd_xdna_operation_admit(operation, iree_ok_status());
    operation = next;
  }
}

iree_status_t iree_hal_amd_xdna_operation_signal(
    iree_hal_amd_xdna_operation_t* operation, iree_hal_device_t* queue_device) {
  const iree_async_frontier_t* frontier =
      iree_hal_amd_xdna_frontier_state_as_frontier(&operation->frontier);
  for (iree_host_size_t i = 0; i < operation->signals.count; ++i) {
    iree_hal_semaphore_t* semaphore = operation->signals.semaphores[i];
    const uint64_t value = operation->signals.payload_values[i];
    iree_status_t status = iree_ok_status();
    if (operation->frontier.exact &&
        iree_hal_amd_xdna_semaphore_is_local(semaphore, queue_device)) {
      status = iree_async_semaphore_publish_untainted(
          (iree_async_semaphore_t*)semaphore, value, frontier);
    } else {
      status = iree_hal_semaphore_signal(semaphore, value, frontier);
    }
    if (!iree_status_is_ok(status)) {
      return status;
    }
  }
  return iree_ok_status();
}

// The operation's device reference keeps queue storage live if a signal waiter
// releases its public device reference inline.
void iree_hal_amd_xdna_operation_complete(
    iree_hal_amd_xdna_operation_t* operation) {
  iree_hal_amd_xdna_operation_t* acceptance_waiters =
      iree_hal_amd_xdna_operation_unregister_producers(operation);
  if (!iree_status_is_ok(operation->status)) {
    iree_hal_amd_xdna_queue_report(operation->queue,
                                   iree_status_code(operation->status),
                                   iree_status_message(operation->status));
  }
  if (operation->profile) {
    iree_hal_amd_xdna_operation_profile_complete(operation);
  }
  iree_hal_amd_xdna_operation_release_resources(operation);
  if (iree_status_is_ok(operation->status)) {
    operation->status =
        iree_hal_amd_xdna_operation_signal(operation, operation->device);
  }
  if (!iree_status_is_ok(operation->status)) {
    iree_hal_semaphore_list_fail(operation->signals, operation->status);
  }
  iree_hal_semaphore_list_release(operation->signals);
  iree_hal_amd_xdna_operation_deallocate(operation);
  iree_hal_amd_xdna_operation_wake_acceptance_waiters(acceptance_waiters);
}

iree_status_t iree_hal_amd_xdna_operation_resolve_waits(
    iree_hal_amd_xdna_operation_t* operation,
    iree_hal_amd_xdna_wait_resolution_t* out_resolution,
    iree_host_size_t* out_deferred_wait_index) {
  iree_hal_amd_xdna_queue_t* queue = operation->queue;
  const bool is_dispatch = iree_hal_amd_xdna_operation_has_dispatch(operation);
  const iree_hal_amd_xdna_frontier_state_t* accepted_state =
      is_dispatch ? &queue->accepted_frontier : NULL;
  const iree_hal_amd_xdna_wait_resolution_flags_t flags =
      is_dispatch ? IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_FLAG_ALLOW_ACCEPTED_FIFO
                  : IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_FLAG_NONE;
  const iree_host_size_t wait_offset = operation->wait_resolution_offset;
  const iree_hal_semaphore_list_t remaining_waits = {
      .count = operation->waits.count - wait_offset,
      .semaphores = operation->waits.semaphores
                        ? operation->waits.semaphores + wait_offset
                        : NULL,
      .payload_values = operation->waits.payload_values
                            ? operation->waits.payload_values + wait_offset
                            : NULL,
  };
  iree_host_size_t deferred_wait_index = IREE_HOST_SIZE_MAX;
  iree_status_t status = iree_hal_amd_xdna_frontier_resolve_waits(
      iree_hal_queue_family_device(queue->base.queue_family), remaining_waits,
      accepted_state, &operation->frontier, flags, &operation->frontier,
      out_resolution, &deferred_wait_index);
  if (iree_status_is_ok(status)) {
    operation->wait_resolution_offset =
        deferred_wait_index == IREE_HOST_SIZE_MAX
            ? operation->waits.count
            : wait_offset + deferred_wait_index;
    if (out_deferred_wait_index) {
      *out_deferred_wait_index = deferred_wait_index == IREE_HOST_SIZE_MAX
                                     ? IREE_HOST_SIZE_MAX
                                     : operation->wait_resolution_offset;
    }
  }
  return status;
}

static void iree_hal_amd_xdna_operation_record_wait_status(
    iree_hal_amd_xdna_operation_t* operation, iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return;
  }
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &operation->wait_status, &expected, (intptr_t)status,
          iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
    iree_status_free(status);
  }
}

// Memory continuations own no FIFO state and consume the last wait directly.
// Native execution returns to its queue owner; a failed handoff cannot safely
// destroy that owner's active publication state on a semaphore callback thread.
static void iree_hal_amd_xdna_operation_schedule_wait_completion(
    iree_hal_amd_xdna_operation_t* operation) {
  if (operation->kind == IREE_HAL_AMD_XDNA_OPERATION_ALLOCA ||
      operation->kind == IREE_HAL_AMD_XDNA_OPERATION_DEALLOCA) {
    iree_status_t status = (iree_status_t)iree_atomic_exchange(
        &operation->wait_status, 0, iree_memory_order_acquire);
    iree_hal_amd_xdna_operation_admit_memory(operation, status);
    return;
  }
  iree_status_t status = iree_async_proactor_submit_one(
      operation->queue->proactor, &operation->wait_completion);
  if (iree_status_is_ok(status)) {
    return;
  }
  iree_status_t wait_status = (iree_status_t)iree_atomic_exchange(
      &operation->wait_status, 0, iree_memory_order_acquire);
  status = iree_status_join(wait_status, status);
  iree_hal_amd_xdna_queue_report(operation->queue, iree_status_code(status),
                                 iree_status_message(status));
  iree_hal_semaphore_list_fail(operation->signals, status);
}

static void iree_hal_amd_xdna_wait_entry_resolved(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_amd_xdna_wait_entry_t* entry = user_data;
  iree_hal_amd_xdna_operation_t* operation = entry->operation;
  iree_hal_semaphore_t* semaphore = entry->semaphore;
  iree_hal_amd_xdna_operation_record_wait_status(operation, status);
  const int32_t previous_count = iree_atomic_fetch_sub(
      &operation->wait_count, 1, iree_memory_order_acq_rel);
  if (previous_count == 1) {
    iree_hal_amd_xdna_operation_schedule_wait_completion(operation);
  }
  // No entry or operation access is permitted after the completion handoff.
  iree_hal_semaphore_release(semaphore);
}

// Registers arena-owned timepoints for every wait. The extra count sentinel
// prevents synchronous callbacks from completing the operation until all
// registrations have either succeeded or been accounted for.
static void iree_hal_amd_xdna_operation_register_waits(
    iree_hal_amd_xdna_operation_t* operation) {
  const iree_host_size_t wait_offset = operation->wait_resolution_offset;
  const iree_hal_semaphore_list_t waits = {
      .count = operation->waits.count - wait_offset,
      .semaphores = operation->waits.semaphores
                        ? operation->waits.semaphores + wait_offset
                        : NULL,
      .payload_values = operation->waits.payload_values
                            ? operation->waits.payload_values + wait_offset
                            : NULL,
  };
  iree_atomic_store(&operation->wait_count, (int32_t)waits.count + 1,
                    iree_memory_order_release);
  iree_status_t status = iree_ok_status();
  iree_host_size_t i = 0;
  while (i < waits.count && iree_status_is_ok(status)) {
    iree_hal_amd_xdna_wait_entry_t* entry = NULL;
    status = iree_hal_amd_xdna_queue_capture_allocate_metadata(
        &operation->capture, sizeof(*entry), (void**)&entry);
    if (iree_status_is_ok(status)) {
      memset(entry, 0, sizeof(*entry));
      entry->operation = operation;
      entry->semaphore = waits.semaphores[i];
      iree_hal_semaphore_retain(entry->semaphore);
      entry->timepoint.callback = iree_hal_amd_xdna_wait_entry_resolved;
      entry->timepoint.user_data = entry;
      status = iree_async_semaphore_acquire_timepoint(
          (iree_async_semaphore_t*)entry->semaphore, waits.payload_values[i],
          &entry->timepoint);
      if (!iree_status_is_ok(status)) {
        iree_hal_semaphore_release(entry->semaphore);
      }
    }
    if (iree_status_is_ok(status)) {
      ++i;
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_operation_record_wait_status(operation, status);
    const int32_t unregistered_count = (int32_t)(waits.count - i);
    iree_atomic_fetch_sub(&operation->wait_count, unregistered_count,
                          iree_memory_order_acq_rel);
  }
  const int32_t previous_count = iree_atomic_fetch_sub(
      &operation->wait_count, 1, iree_memory_order_acq_rel);
  if (previous_count == 1) {
    iree_hal_amd_xdna_operation_schedule_wait_completion(operation);
  }
}

void iree_hal_amd_xdna_operation_admit_memory(
    iree_hal_amd_xdna_operation_t* operation, iree_status_t status) {
  iree_hal_amd_xdna_wait_resolution_t resolution =
      IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_READY;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_operation_resolve_waits(
        operation, &resolution, /*out_deferred_wait_index=*/NULL);
  }
  if (iree_status_is_ok(status) &&
      resolution == IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_DEFER) {
    if (operation->profile) {
      operation->profile->event.flags |=
          IREE_HAL_PROFILE_QUEUE_EVENT_FLAG_SOFTWARE_DEFERRED;
      operation->profile->event.dependency_strategy =
          IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER;
    }
    iree_hal_amd_xdna_operation_register_waits(operation);
    return;
  }
  iree_hal_amd_xdna_operation_issue_memory(operation, status);
}

static bool iree_hal_amd_xdna_operation_signals_wait(
    const iree_hal_amd_xdna_operation_t* producer,
    iree_hal_semaphore_t* semaphore, uint64_t minimum_value) {
  if (!producer) {
    return false;
  }
  for (iree_host_size_t i = 0; i < producer->signals.count; ++i) {
    if (producer->signals.semaphores[i] == semaphore &&
        producer->signals.payload_values[i] >= minimum_value) {
      return true;
    }
  }
  return false;
}

// Links |operation| to the lowest safe queued producer of its first unresolved
// wait. The sole caller-owned publisher is handled by a small fallback list
// because it cannot enter the proactor-owned index during native submission.
static bool iree_hal_amd_xdna_operation_defer_for_acceptance(
    iree_hal_amd_xdna_operation_t* operation,
    iree_host_size_t deferred_wait_index) {
  iree_hal_amd_xdna_queue_t* queue = operation->queue;
  iree_hal_semaphore_t* semaphore =
      operation->waits.semaphores[deferred_wait_index];
  const uint64_t minimum_value =
      operation->waits.payload_values[deferred_wait_index];

  // An accepted signal may be unusable for native FIFO ordering when its
  // causal frontier is inexact, but it still guarantees eventual timeline
  // progress. Waiting for its retirement avoids attaching to a younger
  // unaccepted signal that may itself depend on this operation.
  if (iree_hal_amd_xdna_semaphore_is_local(semaphore, operation->device)) {
    iree_hal_submitted_signal_flags_t signal_flags =
        IREE_HAL_SUBMITTED_SIGNAL_FLAG_NONE;
    iree_async_axis_t producer_axis = 0;
    uint64_t producer_epoch = 0;
    uint64_t producer_value = 0;
    if (iree_hal_submitted_signal_load(
            iree_hal_amd_xdna_semaphore_submitted_signal(semaphore),
            &signal_flags, &producer_axis, &producer_epoch, &producer_value) &&
        producer_value >= minimum_value) {
      return false;
    }
  }
  iree_hal_amd_xdna_queue_producer_node_t* producer_node =
      iree_hal_amd_xdna_queue_producer_index_find(
          &queue->producer_index, semaphore, minimum_value, operation);
  if (producer_node) {
    iree_hal_amd_xdna_operation_t* producer = producer_node->producer;
    // A later signal is a valid timeline producer only when it cannot carry a
    // dependency back to this consumer. Exact-value edges are the user's
    // dependency graph; higher values are selected only from already-ready
    // producers.
    if (producer_node->value != minimum_value &&
        producer->wait_resolution_offset != producer->waits.count) {
      return false;
    }
    operation->next = NULL;
    if (producer->native.acceptance_waiters_tail) {
      producer->native.acceptance_waiters_tail->next = operation;
    } else {
      producer->native.acceptance_waiters_head = operation;
    }
    producer->native.acceptance_waiters_tail = operation;
    return true;
  } else {
    iree_hal_amd_xdna_operation_t* publisher =
        iree_hal_amd_xdna_queue_publisher_operation(queue);
    if (publisher == operation || !iree_hal_amd_xdna_operation_signals_wait(
                                      publisher, semaphore, minimum_value)) {
      return false;
    }
  }
  iree_hal_amd_xdna_ready_list_t* waiters = &queue->acceptance_waiters;
  operation->next = NULL;
  if (waiters->tail) {
    waiters->tail->next = operation;
  } else {
    waiters->head = operation;
  }
  waiters->tail = operation;
  return true;
}

// Resolves one admission on the proactor owner. A consumer whose producer is
// already in this queue's native-admission path waits for that acceptance fact;
// every unrelated unresolved wait registers its ordinary semaphore timepoint.
void iree_hal_amd_xdna_operation_admit(iree_hal_amd_xdna_operation_t* operation,
                                       iree_status_t status) {
  iree_hal_amd_xdna_queue_t* queue = operation->queue;
  if (iree_status_is_ok(status) &&
      !iree_status_is_ok(iree_hal_amd_xdna_queue_failure_status(queue))) {
    status = iree_status_clone(iree_hal_amd_xdna_queue_failure_status(queue));
  }
  iree_hal_amd_xdna_wait_resolution_t resolution =
      IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_READY;
  iree_host_size_t deferred_wait_index = IREE_HOST_SIZE_MAX;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_operation_resolve_waits(operation, &resolution,
                                                       &deferred_wait_index);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amd_xdna_operation_register_producers(operation);
  }
  if (iree_status_is_ok(status) &&
      resolution == IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_DEFER) {
    if (iree_hal_amd_xdna_operation_has_dispatch(operation) &&
        iree_hal_amd_xdna_operation_defer_for_acceptance(operation,
                                                         deferred_wait_index)) {
    } else {
      iree_hal_amd_xdna_operation_register_waits(operation);
    }
    return;
  }
  iree_hal_amd_xdna_queue_enqueue_ready(operation, status);
}

void iree_hal_amd_xdna_queue_retry_acceptance_waiters(
    iree_hal_amd_xdna_queue_t* queue) {
  iree_hal_amd_xdna_operation_t* operation = queue->acceptance_waiters.head;
  queue->acceptance_waiters.head = NULL;
  queue->acceptance_waiters.tail = NULL;
  while (operation) {
    iree_hal_amd_xdna_operation_t* next = operation->next;
    operation->next = NULL;
    iree_hal_amd_xdna_operation_admit(operation, iree_ok_status());
    operation = next;
  }
}

static void iree_hal_amd_xdna_queue_wait_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  iree_hal_amd_xdna_operation_t* operation = user_data;
  iree_hal_amd_xdna_queue_t* queue = operation->queue;
  iree_status_t wait_status = (iree_status_t)iree_atomic_exchange(
      &operation->wait_status, 0, iree_memory_order_acquire);
  status = iree_status_join(status, wait_status);
  iree_slim_mutex_lock(&queue->state_mutex);
  if (iree_status_is_ok(status)) {
    iree_hal_amd_xdna_wait_resolution_t resolution =
        IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_DEFER;
    status = iree_hal_amd_xdna_operation_resolve_waits(
        operation, &resolution, /*out_deferred_wait_index=*/NULL);
    if (iree_status_is_ok(status) &&
        resolution != IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_READY) {
      status =
          iree_make_status(IREE_STATUS_INTERNAL,
                           "completed XDNA software wait remained unresolved");
    }
  }
  // Enqueueing may complete the operation and recycle its capture storage.
  iree_hal_amd_xdna_queue_enqueue_ready(operation, status);
  iree_slim_mutex_unlock(&queue->state_mutex);
}

static void iree_hal_amd_xdna_queue_admit(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  iree_hal_amd_xdna_operation_t* operation = user_data;
  iree_hal_amd_xdna_queue_t* queue = operation->queue;
  iree_slim_mutex_lock(&queue->state_mutex);
  iree_hal_amd_xdna_operation_admit(operation, status);
  // A direct caller may have briefly owned and then released publication while
  // this admission was pending. Retry older ready work even when this operation
  // itself registered a software wait.
  iree_hal_amd_xdna_queue_pump_dispatch(queue);
  iree_slim_mutex_unlock(&queue->state_mutex);
}

typedef enum iree_hal_amd_xdna_semaphore_capture_kind_e {
  IREE_HAL_AMD_XDNA_SEMAPHORE_CAPTURE_WAITS = 0,
  IREE_HAL_AMD_XDNA_SEMAPHORE_CAPTURE_SIGNALS,
} iree_hal_amd_xdna_semaphore_capture_kind_t;

static iree_status_t iree_hal_amd_xdna_operation_capture_semaphores(
    iree_hal_amd_xdna_operation_t* operation, iree_hal_semaphore_list_t source,
    iree_hal_amd_xdna_semaphore_capture_kind_t capture_kind,
    iree_hal_semaphore_list_t* out_list) {
  *out_list = iree_hal_semaphore_list_empty();
  if (!source.count) {
    return iree_ok_status();
  }
  const bool captures_producers =
      capture_kind == IREE_HAL_AMD_XDNA_SEMAPHORE_CAPTURE_SIGNALS &&
      (operation->kind == IREE_HAL_AMD_XDNA_OPERATION_DISPATCH ||
       operation->kind == IREE_HAL_AMD_XDNA_OPERATION_EXECUTE);
  iree_host_size_t producer_count = 0;
  if (captures_producers) {
    for (iree_host_size_t i = 0; i < source.count; ++i) {
      producer_count += iree_hal_amd_xdna_semaphore_is_local(
          source.semaphores[i], operation->device);
    }
  }
  iree_host_size_t size = 0;
  iree_host_size_t semaphores_offset = 0;
  iree_host_size_t values_offset = 0;
  iree_host_size_t producers_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &size,
      IREE_STRUCT_FIELD_ALIGNED(source.count, iree_hal_semaphore_t*,
                                iree_alignof(iree_hal_semaphore_t*),
                                &semaphores_offset),
      IREE_STRUCT_FIELD_ALIGNED(source.count, uint64_t, iree_alignof(uint64_t),
                                &values_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          producer_count, iree_hal_amd_xdna_producer_entry_t,
          iree_alignof(iree_hal_amd_xdna_producer_entry_t),
          &producers_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_queue_capture_allocate_metadata(
      &operation->capture, size, (void**)&storage));
  memset(storage, 0, size);

  iree_hal_amd_xdna_producer_entry_t* producer_entries = NULL;
  iree_hal_amd_xdna_producer_entry_t* packed_entries =
      (iree_hal_amd_xdna_producer_entry_t*)(storage + producers_offset);
  for (iree_host_size_t i = 0; i < producer_count; ++i) {
    packed_entries[i].next = producer_entries;
    producer_entries = &packed_entries[i];
  }

  iree_hal_semaphore_list_t captured_list = {
      .count = source.count,
      .semaphores = (iree_hal_semaphore_t**)(storage + semaphores_offset),
      .payload_values = (uint64_t*)(storage + values_offset),
  };
  iree_hal_amd_xdna_producer_entry_t* producer_entry = producer_entries;
  for (iree_host_size_t i = 0; i < source.count; ++i) {
    captured_list.semaphores[i] = source.semaphores[i];
    captured_list.payload_values[i] = source.payload_values[i];
    if (captures_producers && iree_hal_amd_xdna_semaphore_is_local(
                                  source.semaphores[i], operation->device)) {
      producer_entry->index_node.semaphore = source.semaphores[i];
      producer_entry->index_node.value = source.payload_values[i];
      producer_entry->index_node.producer = operation;
      producer_entry = producer_entry->next;
    }
  }
  if (captures_producers) {
    operation->native.producer_entries = producer_entries;
  }
  iree_hal_semaphore_list_retain(captured_list);
  *out_list = captured_list;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_operation_allocate(
    iree_hal_amd_xdna_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_amd_xdna_operation_kind_t kind,
    iree_hal_amd_xdna_operation_t** out_operation) {
  *out_operation = NULL;
  iree_hal_amd_xdna_queue_capture_t capture;
  iree_hal_amd_xdna_queue_capture_initialize(&queue->storage, &capture);
  iree_hal_amd_xdna_operation_t* operation = NULL;
  iree_status_t status = iree_hal_amd_xdna_queue_capture_allocate_metadata(
      &capture, sizeof(*operation), (void**)&operation);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_queue_capture_deinitialize(&capture);
    return status;
  }
  memset(operation, 0, sizeof(*operation));
  operation->queue = queue;
  operation->device = iree_hal_queue_family_device(queue->base.queue_family);
  operation->metadata_block_pool = &queue->storage.metadata_block_pool;
  operation->capture = capture;
  operation->kind = kind;
  iree_hal_amd_xdna_frontier_state_initialize(&operation->frontier);
  iree_hal_device_retain(operation->device);
  status = iree_hal_amd_xdna_operation_capture_semaphores(
      operation, waits, IREE_HAL_AMD_XDNA_SEMAPHORE_CAPTURE_WAITS,
      &operation->waits);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_operation_capture_semaphores(
        operation, signals, IREE_HAL_AMD_XDNA_SEMAPHORE_CAPTURE_SIGNALS,
        &operation->signals);
  }
  if (iree_status_is_ok(status)) {
    iree_async_operation_initialize(&operation->admission,
                                    IREE_ASYNC_OPERATION_TYPE_NOP,
                                    IREE_ASYNC_OPERATION_FLAG_NONE,
                                    iree_hal_amd_xdna_queue_admit, operation);
    iree_async_operation_initialize(
        &operation->wait_completion, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_amd_xdna_queue_wait_complete,
        operation);
    iree_atomic_store(&operation->wait_count, 0, iree_memory_order_relaxed);
    iree_atomic_store(&operation->wait_status, 0, iree_memory_order_relaxed);
    iree_atomic_store(&operation->continuation_status, 0,
                      iree_memory_order_relaxed);
    *out_operation = operation;
  } else {
    iree_hal_semaphore_list_release(operation->signals);
    iree_hal_semaphore_list_release(operation->waits);
    iree_hal_amd_xdna_operation_deallocate(operation);
  }
  return status;
}

void iree_hal_amd_xdna_operation_discard(
    iree_hal_amd_xdna_operation_t* operation) {
  iree_hal_amd_xdna_operation_release_resources(operation);
  iree_hal_semaphore_list_release(operation->signals);
  iree_hal_amd_xdna_operation_deallocate(operation);
}
