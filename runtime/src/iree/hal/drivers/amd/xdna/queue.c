// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/queue.h"

#include "iree/async/event.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/semaphore.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/drivers/amd/xdna/barrier.h"
#include "iree/hal/drivers/amd/xdna/executable.h"
#include "iree/hal/drivers/amd/xdna/queue_frontier.h"
#include "iree/hal/drivers/amd/xdna/queue_operation.h"
#include "iree/hal/drivers/amd/xdna/queue_producer_index.h"
#include "iree/hal/drivers/amd/xdna/queue_service.h"
#include "iree/hal/drivers/amd/xdna/queue_storage.h"
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

// Native FIFO positions, distinct from host operation readiness.
typedef struct iree_hal_amd_xdna_pending_t {
  // Accepted invocation retaining its command, buffers and executable.
  iree_hal_amd_xdna_operation_t* operation;
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
  // Borrowed device owner dominating all native children.
  iree_hal_amd_xdna_context_t* context;
  // Borrowed shared completion owner.
  iree_async_proactor_t* proactor;
  // Native queue, released after accepted resources retire.
  amdf_kernel_queue_t* handle;
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
  // Potentially blocking mapped host transfer owner.
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
  // Sticky native observation/execution failure, owned by this queue.
  iree_status_t failure_status;
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

static const iree_hal_queue_vtable_t iree_hal_amd_xdna_queue_vtable;

static void iree_hal_amd_xdna_operation_admit(
    iree_hal_amd_xdna_operation_t* operation, iree_status_t status);

static void iree_hal_amd_xdna_queue_report(iree_hal_amd_xdna_queue_t* queue,
                                           iree_status_code_t status_code,
                                           iree_string_view_t message) {
  const iree_hal_device_driver_failure_event_t payload = {
      .record_length = sizeof(payload),
      .abi_version = IREE_HAL_DEVICE_DRIVER_FAILURE_EVENT_ABI_VERSION_0,
      .status_code = status_code,
      .message = message,
  };
  iree_hal_device_event_t event = iree_hal_device_event_default();
  event.type = IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE;
  event.severity = IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
  event.source.driver_id = IREE_SV("xdna");
  event.payload = iree_make_const_byte_span(&payload, sizeof(payload));
  iree_hal_device_event_sink_publish(queue->context->event_sink, &event);
}

// Reports and consumes a cold-path failure that has no public status channel.
static void iree_hal_amd_xdna_queue_report_status(
    iree_hal_amd_xdna_queue_t* queue, iree_status_t status) {
  iree_hal_amd_xdna_queue_report(queue, iree_status_code(status),
                                 iree_status_message(status));
  iree_status_free(status);
}

// Returns both arenas before releasing the device that owns their pools.
static void iree_hal_amd_xdna_operation_deallocate(
    iree_hal_amd_xdna_operation_t* operation) {
  iree_hal_device_t* device = operation->device;
  iree_hal_amd_xdna_queue_capture_t capture = operation->capture;
  operation = NULL;
  iree_hal_amd_xdna_queue_capture_deinitialize(&capture);
  iree_hal_device_release(device);
}

static void iree_hal_amd_xdna_operation_register_producers(
    iree_hal_amd_xdna_operation_t* operation) {
  if (operation->kind != IREE_HAL_AMD_XDNA_OPERATION_DISPATCH ||
      operation->dispatch.producers_indexed) {
    return;
  }
  for (iree_hal_amd_xdna_producer_entry_t* entry =
           operation->dispatch.producer_entries;
       entry; entry = entry->next) {
    iree_hal_amd_xdna_queue_producer_index_insert(
        &operation->queue->producer_index, &entry->index_node);
  }
  operation->dispatch.producers_indexed = true;
}

// Removes every signal owned by |operation| and returns its targeted consumer
// list. The returned operations remain owned by their captured submissions.
static iree_hal_amd_xdna_operation_t*
iree_hal_amd_xdna_operation_unregister_producers(
    iree_hal_amd_xdna_operation_t* operation) {
  if (operation->kind != IREE_HAL_AMD_XDNA_OPERATION_DISPATCH ||
      !operation->dispatch.producers_indexed) {
    return NULL;
  }
  for (iree_hal_amd_xdna_producer_entry_t* entry =
           operation->dispatch.producer_entries;
       entry; entry = entry->next) {
    iree_hal_amd_xdna_queue_producer_index_erase(
        &operation->queue->producer_index, &entry->index_node);
  }
  operation->dispatch.producers_indexed = false;
  iree_hal_amd_xdna_operation_t* waiters_head =
      operation->dispatch.acceptance_waiters_head;
  operation->dispatch.acceptance_waiters_head = NULL;
  operation->dispatch.acceptance_waiters_tail = NULL;
  return waiters_head;
}

static void iree_hal_amd_xdna_operation_wake_acceptance_waiters(
    iree_hal_amd_xdna_operation_t* operation) {
  while (operation) {
    iree_hal_amd_xdna_operation_t* next = operation->next;
    operation->next = NULL;
    iree_hal_amd_xdna_operation_admit(operation, iree_ok_status());
    operation = next;
  }
}

static iree_status_t iree_hal_amd_xdna_operation_signal(
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
static void iree_hal_amd_xdna_operation_complete(
    iree_hal_amd_xdna_operation_t* operation) {
  iree_hal_amd_xdna_operation_t* acceptance_waiters =
      iree_hal_amd_xdna_operation_unregister_producers(operation);
  if (!iree_status_is_ok(operation->status)) {
    iree_hal_amd_xdna_queue_report(operation->queue,
                                   iree_status_code(operation->status),
                                   iree_status_message(operation->status));
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

static iree_hal_amd_xdna_operation_t*
iree_hal_amd_xdna_operation_from_service_item(
    iree_hal_amd_xdna_queue_service_item_t* item) {
  return iree_containerof(item, iree_hal_amd_xdna_operation_t, service_item);
}

static iree_hal_amd_xdna_operation_t*
iree_hal_amd_xdna_queue_publisher_operation(
    const iree_hal_amd_xdna_queue_t* queue) {
  return (iree_hal_amd_xdna_operation_t*)iree_atomic_load(
      &queue->publisher_operation, iree_memory_order_acquire);
}

static bool iree_hal_amd_xdna_queue_try_claim_publisher(
    iree_hal_amd_xdna_queue_t* queue) {
  int32_t expected = 0;
  return iree_atomic_compare_exchange_strong(&queue->publisher_claim, &expected,
                                             1, iree_memory_order_acquire,
                                             iree_memory_order_relaxed);
}

static void iree_hal_amd_xdna_queue_release_publisher(
    iree_hal_amd_xdna_queue_t* queue) {
  iree_atomic_store(&queue->publisher_claim, 0, iree_memory_order_release);
}

static bool iree_hal_amd_xdna_queue_has_publishable_dispatch(
    const iree_hal_amd_xdna_queue_t* queue) {
  return iree_status_is_ok(queue->failure_status) &&
         queue->observer_state == IREE_HAL_AMD_XDNA_OBSERVER_STATE_REGISTERED &&
         !iree_atomic_load(&queue->native_publication_blocked,
                           iree_memory_order_acquire) &&
         queue->dispatch_ready.head &&
         queue->pending.count < queue->pending.capacity;
}

// Starts one deferred dispatch while retaining the existing publisher claim.
static void iree_hal_amd_xdna_queue_start_claimed_dispatch(
    iree_hal_amd_xdna_queue_t* queue) {
  iree_hal_amd_xdna_operation_t* operation = queue->dispatch_ready.head;
  queue->dispatch_ready.head = operation->next;
  if (!queue->dispatch_ready.head) {
    queue->dispatch_ready.tail = NULL;
  }
  operation->next = NULL;
  iree_atomic_store(&queue->publisher_operation, (intptr_t)operation,
                    iree_memory_order_release);
  iree_hal_amd_xdna_queue_service_enqueue(&queue->publisher,
                                          &operation->service_item);
}

// Admits at most one publisher item. Its proactor acknowledgement advances the
// accepted frontier before this function may admit the next physical command.
static void iree_hal_amd_xdna_queue_pump_dispatch(
    iree_hal_amd_xdna_queue_t* queue) {
  if (!iree_hal_amd_xdna_queue_has_publishable_dispatch(queue) ||
      !iree_hal_amd_xdna_queue_try_claim_publisher(queue)) {
    return;
  }
  iree_hal_amd_xdna_queue_start_claimed_dispatch(queue);
}

// Gives already-deferred work priority when one publication result returns.
// The existing claim transfers directly to the worker when capacity permits.
static void iree_hal_amd_xdna_queue_continue_or_release_publisher(
    iree_hal_amd_xdna_queue_t* queue) {
  if (iree_hal_amd_xdna_queue_has_publishable_dispatch(queue)) {
    iree_hal_amd_xdna_queue_start_claimed_dispatch(queue);
  } else {
    iree_hal_amd_xdna_queue_release_publisher(queue);
  }
}

// Fails software-ready dispatches after the native observer loses authority.
// The current or retained accepted operation keeps the queue live throughout.
static void iree_hal_amd_xdna_queue_fail_dispatch_ready(
    iree_hal_amd_xdna_queue_t* queue) {
  iree_hal_amd_xdna_operation_t* operation = queue->dispatch_ready.head;
  if (queue->dispatch_ready.tail) {
    queue->dispatch_ready.tail->next = queue->acceptance_waiters.head;
  } else {
    operation = queue->acceptance_waiters.head;
  }
  queue->dispatch_ready.head = NULL;
  queue->dispatch_ready.tail = NULL;
  queue->acceptance_waiters.head = NULL;
  queue->acceptance_waiters.tail = NULL;
  while (operation) {
    iree_hal_amd_xdna_operation_t* next = operation->next;
    operation->next = NULL;
    operation->status = iree_status_clone(queue->failure_status);
    iree_hal_amd_xdna_operation_complete(operation);
    operation = next;
  }
}

static void iree_hal_amd_xdna_queue_record_failure(
    iree_hal_amd_xdna_queue_t* queue, iree_status_t status) {
  iree_atomic_store(&queue->native_admission_open, 0,
                    iree_memory_order_release);
  queue->failure_status = iree_status_join(queue->failure_status, status);
}

// A failed observer cannot prove accepted device references have retired. Keep
// each remaining operation and its parent graph live, and fail completion
// edges. Native progress is never inferred from a failure signal.
static void iree_hal_amd_xdna_queue_abandon_pending(
    iree_hal_amd_xdna_queue_t* queue) {
  iree_hal_amd_xdna_queue_report(queue, iree_status_code(queue->failure_status),
                                 iree_status_message(queue->failure_status));
  while (queue->pending.count) {
    iree_hal_amd_xdna_pending_t* pending =
        &queue->pending.entries[queue->pending.head];
    iree_hal_semaphore_list_fail(pending->operation->signals,
                                 iree_status_clone(queue->failure_status));
    pending->operation = NULL;
    queue->pending.head = (queue->pending.head + 1) % queue->pending.capacity;
    --queue->pending.count;
  }
  iree_atomic_store(&queue->pending.visible_count,
                    (int32_t)queue->pending.count, iree_memory_order_release);
  iree_hal_amd_xdna_queue_fail_dispatch_ready(queue);
}

static void iree_hal_amd_xdna_queue_fail_pending(
    iree_hal_amd_xdna_queue_t* queue, iree_status_t status) {
  iree_hal_amd_xdna_queue_record_failure(queue, status);
  iree_hal_amd_xdna_queue_abandon_pending(queue);
}

static iree_status_t iree_hal_amd_xdna_queue_arm(
    iree_hal_amd_xdna_queue_t* queue) {
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      queue->context->api->kernel_queue_request_notification(
          queue->handle, queue->pending.entries[queue->pending.head].submission,
          &queue->native_event),
      "kernel_queue_request_notification"));
  queue->notification_pending = true;
  return iree_ok_status();
}

static void iree_hal_amd_xdna_queue_publish_signals(
    iree_hal_amd_xdna_queue_t* queue, iree_hal_amd_xdna_operation_t* operation,
    uint64_t epoch) {
  if (!queue->tracker) {
    return;
  }
  iree_hal_device_t* queue_device =
      iree_hal_queue_family_device(queue->base.queue_family);
  const iree_async_frontier_t* frontier =
      iree_hal_amd_xdna_frontier_state_as_frontier(&operation->frontier);
  for (iree_host_size_t i = 0; i < operation->signals.count; ++i) {
    iree_hal_semaphore_t* semaphore = operation->signals.semaphores[i];
    if (!iree_hal_amd_xdna_semaphore_is_local(semaphore, queue_device)) {
      continue;
    }
    iree_hal_amd_xdna_semaphore_publish_signal(
        semaphore, queue->axis, frontier, operation->frontier.exact, epoch,
        operation->signals.payload_values[i]);
  }
}

static void iree_hal_amd_xdna_publisher_execute(
    void* user_data, iree_hal_amd_xdna_queue_service_item_t* item) {
  iree_hal_amd_xdna_queue_t* queue = (iree_hal_amd_xdna_queue_t*)user_data;
  iree_hal_amd_xdna_operation_t* operation =
      iree_hal_amd_xdna_operation_from_service_item(item);
  operation->dispatch.publication_result =
      IREE_HAL_AMD_XDNA_PUBLICATION_RESULT_REJECTED;
  if (!operation->dispatch.invocation) {
    operation->status = iree_hal_amd_xdna_function_prepare(
        operation->dispatch.function, operation->dispatch.bindings,
        &operation->dispatch.invocation, &operation->dispatch.command);
  }
  if (iree_status_is_ok(operation->status)) {
    const amdf_xdna_kernel_queue_submission_info_t submission_info = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO,
        .structure_size = sizeof(submission_info),
        .command_count = 1,
        .commands = &operation->dispatch.command,
    };
    operation->dispatch.publication_had_pending =
        iree_atomic_load(&queue->pending.visible_count,
                         iree_memory_order_acquire) > 0;
    const amdf_status_t submit_status =
        queue->context->xdna->kernel_queue_submit(
            queue->handle, &submission_info, &operation->dispatch.submission);
    if (submit_status == amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
      operation->dispatch.publication_result =
          IREE_HAL_AMD_XDNA_PUBLICATION_RESULT_RETRY;
    } else {
      operation->status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          submit_status, "xdna.kernel_queue_submit");
      if (iree_status_is_ok(operation->status)) {
        operation->dispatch.publication_result =
            IREE_HAL_AMD_XDNA_PUBLICATION_RESULT_ACCEPTED;
      }
    }
  }
}

static void iree_hal_amd_xdna_host_worker_execute(
    void* user_data, iree_hal_amd_xdna_queue_service_item_t* item) {
  (void)user_data;
  iree_hal_amd_xdna_operation_execute_host(
      iree_hal_amd_xdna_operation_from_service_item(item));
}

static void iree_hal_amd_xdna_service_abandon(
    void* user_data, iree_hal_amd_xdna_queue_service_item_t* item,
    iree_status_t status) {
  iree_hal_amd_xdna_queue_t* queue = (iree_hal_amd_xdna_queue_t*)user_data;
  iree_atomic_store(&queue->native_admission_open, 0,
                    iree_memory_order_release);
  iree_hal_amd_xdna_operation_t* operation =
      iree_hal_amd_xdna_operation_from_service_item(item);
  status = iree_status_join(operation->status, status);
  operation->status = iree_ok_status();
  iree_hal_amd_xdna_queue_report(queue, iree_status_code(status),
                                 iree_status_message(status));
  iree_hal_semaphore_list_fail(operation->signals, status);
}

static void iree_hal_amd_xdna_queue_enqueue_ready(
    iree_hal_amd_xdna_operation_t* operation, iree_status_t status) {
  iree_hal_amd_xdna_queue_t* queue = operation->queue;
  operation->status = status;
  if (iree_status_is_ok(operation->status) &&
      !iree_status_is_ok(queue->failure_status)) {
    operation->status = iree_status_clone(queue->failure_status);
  }
  if (!iree_status_is_ok(operation->status) ||
      operation->kind == IREE_HAL_AMD_XDNA_OPERATION_BARRIER) {
    iree_hal_amd_xdna_operation_complete(operation);
    return;
  }
  if (operation->kind == IREE_HAL_AMD_XDNA_OPERATION_ALLOCA ||
      operation->kind == IREE_HAL_AMD_XDNA_OPERATION_DEALLOCA ||
      operation->kind == IREE_HAL_AMD_XDNA_OPERATION_TRANSFER) {
    iree_hal_amd_xdna_queue_service_enqueue(&queue->host_worker,
                                            &operation->service_item);
    return;
  }
  operation->next = NULL;
  if (queue->dispatch_ready.tail) {
    queue->dispatch_ready.tail->next = operation;
  } else {
    queue->dispatch_ready.head = operation;
  }
  queue->dispatch_ready.tail = operation;
  iree_hal_amd_xdna_queue_pump_dispatch(queue);
}

// Restores a prepared operation ahead of younger ready work without pumping.
static void iree_hal_amd_xdna_queue_prepend_ready(
    iree_hal_amd_xdna_queue_t* queue,
    iree_hal_amd_xdna_operation_t* operation) {
  operation->next = queue->dispatch_ready.head;
  queue->dispatch_ready.head = operation;
  if (!queue->dispatch_ready.tail) {
    queue->dispatch_ready.tail = operation;
  }
}

static iree_status_t iree_hal_amd_xdna_operation_resolve_waits(
    iree_hal_amd_xdna_operation_t* operation,
    iree_hal_amd_xdna_wait_resolution_t* out_resolution,
    iree_host_size_t* out_deferred_wait_index) {
  iree_hal_amd_xdna_queue_t* queue = operation->queue;
  const bool is_dispatch =
      operation->kind == IREE_HAL_AMD_XDNA_OPERATION_DISPATCH;
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

// Schedules the sole wait completion owner. A failed handoff cannot safely
// destroy queue-owned storage from an arbitrary semaphore callback thread.
static void iree_hal_amd_xdna_operation_schedule_wait_completion(
    iree_hal_amd_xdna_operation_t* operation) {
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
    if (producer->dispatch.acceptance_waiters_tail) {
      producer->dispatch.acceptance_waiters_tail->next = operation;
    } else {
      producer->dispatch.acceptance_waiters_head = operation;
    }
    producer->dispatch.acceptance_waiters_tail = operation;
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
static void iree_hal_amd_xdna_operation_admit(
    iree_hal_amd_xdna_operation_t* operation, iree_status_t status) {
  iree_hal_amd_xdna_queue_t* queue = operation->queue;
  if (iree_status_is_ok(status) && !iree_status_is_ok(queue->failure_status)) {
    status = iree_status_clone(queue->failure_status);
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
    if (operation->kind == IREE_HAL_AMD_XDNA_OPERATION_DISPATCH &&
        iree_hal_amd_xdna_operation_defer_for_acceptance(operation,
                                                         deferred_wait_index)) {
    } else {
      iree_hal_amd_xdna_operation_register_waits(operation);
    }
    return;
  }
  iree_hal_amd_xdna_queue_enqueue_ready(operation, status);
}

static void iree_hal_amd_xdna_queue_retry_acceptance_waiters(
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
  iree_status_t wait_status = (iree_status_t)iree_atomic_exchange(
      &operation->wait_status, 0, iree_memory_order_acquire);
  status = iree_status_join(status, wait_status);
  iree_slim_mutex_lock(&operation->queue->state_mutex);
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
  iree_hal_amd_xdna_queue_enqueue_ready(operation, status);
  iree_slim_mutex_unlock(&operation->queue->state_mutex);
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

// Commits one direct or deferred publication result under |state_mutex|.
// |service| and |item| are NULL when the caller performed native publication.
static void iree_hal_amd_xdna_queue_commit_publication(
    iree_hal_amd_xdna_queue_t* queue,
    iree_hal_amd_xdna_queue_service_t* service,
    iree_hal_amd_xdna_queue_service_item_t* item,
    iree_hal_amd_xdna_operation_t* operation, iree_status_t status) {
  const iree_hal_amd_xdna_publication_result_t publication_result =
      operation->dispatch.publication_result;
  const bool was_accepted =
      publication_result == IREE_HAL_AMD_XDNA_PUBLICATION_RESULT_ACCEPTED;
  iree_hal_amd_xdna_operation_t* targeted_waiters = NULL;

  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_queue_fail_pending(queue, iree_status_clone(status));
    operation->status = iree_status_join(operation->status, status);
  }

  if (publication_result == IREE_HAL_AMD_XDNA_PUBLICATION_RESULT_RETRY &&
      iree_status_is_ok(operation->status) &&
      iree_status_is_ok(queue->failure_status)) {
    if (!operation->dispatch.publication_had_pending) {
      operation->status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "XDNA native publication reported BUSY without pending progress");
      iree_hal_amd_xdna_queue_fail_pending(
          queue, iree_status_clone(operation->status));
    } else {
      // Preserve the prepared invocation at the head without another binding
      // patch/allocation cycle. Remaining accepted work supplies the next
      // checked progress event; if it retired while BUSY was returning then
      // the existing claim transfers immediately to one retry.
      iree_hal_amd_xdna_operation_register_producers(operation);
      iree_hal_amd_xdna_queue_prepend_ready(queue, operation);
      iree_atomic_store(&queue->publisher_operation, 0,
                        iree_memory_order_release);
      iree_hal_amd_xdna_queue_retry_acceptance_waiters(queue);
      if (service) {
        iree_hal_amd_xdna_queue_service_acknowledge(service, item);
      }
      if (queue->pending.count) {
        iree_atomic_store(&queue->native_publication_blocked, 1,
                          iree_memory_order_release);
        iree_hal_amd_xdna_queue_release_publisher(queue);
      } else {
        iree_hal_amd_xdna_queue_continue_or_release_publisher(queue);
      }
      return;
    }
  }

  if (was_accepted && iree_status_is_ok(operation->status) &&
      iree_status_is_ok(queue->failure_status)) {
    // Physical FIFO order includes every previously acknowledged acceptance.
    iree_hal_amd_xdna_frontier_state_merge(&operation->frontier,
                                           &queue->accepted_frontier);
    const uint64_t epoch = queue->epoch + 1;
    if (queue->tracker) {
      iree_hal_amd_xdna_frontier_state_advance(&operation->frontier,
                                               queue->axis, epoch);
    } else {
      operation->frontier.exact = false;
    }
    iree_hal_amd_xdna_queue_publish_signals(queue, operation, epoch);
    iree_hal_amd_xdna_frontier_state_copy(&operation->frontier,
                                          &queue->accepted_frontier);
    queue->epoch = epoch;
    targeted_waiters =
        iree_hal_amd_xdna_operation_unregister_producers(operation);
    const uint32_t tail =
        (queue->pending.head + queue->pending.count) % queue->pending.capacity;
    queue->pending.entries[tail] = (iree_hal_amd_xdna_pending_t){
        .operation = operation,
        .submission = operation->dispatch.submission,
        .epoch = epoch,
    };
    ++queue->pending.count;
    iree_atomic_store(&queue->pending.visible_count,
                      (int32_t)queue->pending.count, iree_memory_order_release);
    if (!queue->notification_pending) {
      iree_status_t arm_status = iree_hal_amd_xdna_queue_arm(queue);
      if (!iree_status_is_ok(arm_status)) {
        iree_hal_amd_xdna_queue_fail_pending(queue, arm_status);
      }
    }
    operation = NULL;
  } else if (was_accepted) {
    targeted_waiters =
        iree_hal_amd_xdna_operation_unregister_producers(operation);
    iree_hal_semaphore_list_fail(operation->signals,
                                 iree_status_clone(queue->failure_status));
    // Native acceptance is unobservable after the queue failure. Preserve the
    // operation and its device graph instead of guessing at retirement.
    operation = NULL;
  } else if (iree_status_is_ok(operation->status) &&
             !iree_status_is_ok(queue->failure_status)) {
    operation->status = iree_status_clone(queue->failure_status);
  }

  iree_atomic_store(&queue->publisher_operation, 0, iree_memory_order_release);
  iree_hal_amd_xdna_queue_retry_acceptance_waiters(queue);
  if (service) {
    iree_hal_amd_xdna_queue_service_acknowledge(service, item);
  }
  iree_hal_amd_xdna_queue_continue_or_release_publisher(queue);
  if (operation) {
    iree_hal_amd_xdna_operation_complete(operation);
  }
  iree_hal_amd_xdna_operation_wake_acceptance_waiters(targeted_waiters);
}

static void iree_hal_amd_xdna_publisher_complete(
    void* user_data, iree_hal_amd_xdna_queue_service_t* service,
    iree_hal_amd_xdna_queue_service_item_t* item, iree_status_t status) {
  iree_hal_amd_xdna_queue_t* queue = (iree_hal_amd_xdna_queue_t*)user_data;
  iree_hal_amd_xdna_operation_t* operation =
      iree_hal_amd_xdna_operation_from_service_item(item);
  iree_slim_mutex_lock(&queue->state_mutex);
  iree_hal_amd_xdna_queue_commit_publication(queue, service, item, operation,
                                             status);
  iree_slim_mutex_unlock(&queue->state_mutex);
}

static void iree_hal_amd_xdna_memory_wait_resolved(void* user_data,
                                                   iree_status_t status) {
  iree_hal_amd_xdna_operation_t* operation = user_data;
  if (!iree_status_is_ok(status)) {
    intptr_t expected = 0;
    if (!iree_atomic_compare_exchange_strong(
            &operation->memory_status, &expected, (intptr_t)status,
            iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
      iree_status_free(status);
    }
  }
  status = iree_async_proactor_submit_one(operation->queue->proactor,
                                          &operation->memory_completion);
  if (!iree_status_is_ok(status)) {
    iree_status_t memory_status = (iree_status_t)iree_atomic_exchange(
        &operation->memory_status, 0, iree_memory_order_acquire);
    status = iree_status_join(memory_status, status);
    iree_hal_amd_xdna_queue_report(operation->queue, iree_status_code(status),
                                   iree_status_message(status));
    iree_hal_semaphore_list_fail(operation->signals, status);
  }
}

static void iree_hal_amd_xdna_queue_memory_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  (void)base_operation;
  (void)flags;
  iree_hal_amd_xdna_operation_t* operation = user_data;
  iree_hal_amd_xdna_memory_wait_t* wait = operation->alloca.memory_wait;
  const iree_hal_amd_xdna_memory_wait_kind_t wait_kind = wait->kind;
  wait->kind = IREE_HAL_AMD_XDNA_MEMORY_WAIT_NONE;
  iree_status_t memory_status = (iree_status_t)iree_atomic_exchange(
      &operation->memory_status, 0, iree_memory_order_acquire);
  status = iree_status_join(status, memory_status);
  if (iree_status_is_ok(status) &&
      !iree_status_is_ok(operation->queue->failure_status)) {
    status = iree_status_clone(operation->queue->failure_status);
  }
  if (!iree_status_is_ok(status)) {
    operation->status = status;
    if (wait_kind == IREE_HAL_AMD_XDNA_MEMORY_WAIT_FRONTIER) {
      iree_hal_amd_xdna_operation_release_alloca_reservations(operation);
    }
    iree_hal_amd_xdna_operation_complete(operation);
    return;
  }
  if (IREE_UNLIKELY(wait_kind == IREE_HAL_AMD_XDNA_MEMORY_WAIT_NONE)) {
    operation->status = iree_make_status(
        IREE_STATUS_INTERNAL, "XDNA memory wait completed without an owner");
    iree_hal_amd_xdna_operation_complete(operation);
    return;
  }
  iree_hal_amd_xdna_queue_service_enqueue(&operation->queue->host_worker,
                                          &operation->service_item);
}

static void iree_hal_amd_xdna_host_worker_complete(
    void* user_data, iree_hal_amd_xdna_queue_service_t* service,
    iree_hal_amd_xdna_queue_service_item_t* item, iree_status_t status) {
  iree_hal_amd_xdna_queue_t* queue = (iree_hal_amd_xdna_queue_t*)user_data;
  iree_hal_amd_xdna_operation_t* operation =
      iree_hal_amd_xdna_operation_from_service_item(item);
  operation->status = iree_status_join(operation->status, status);
  if (iree_status_is_ok(operation->status) &&
      !iree_status_is_ok(queue->failure_status)) {
    operation->status = iree_status_clone(queue->failure_status);
  }
  iree_hal_amd_xdna_queue_service_acknowledge(service, item);

  if (operation->kind == IREE_HAL_AMD_XDNA_OPERATION_ALLOCA &&
      iree_status_is_ok(operation->status)) {
    iree_hal_amd_xdna_memory_wait_t* wait = operation->alloca.memory_wait;
    iree_async_operation_initialize(
        &operation->memory_completion, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_amd_xdna_queue_memory_complete,
        operation);
    switch (operation->alloca.acquire_result) {
      case IREE_HAL_POOL_ACQUIRE_OK:
      case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
        break;
      case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT: {
        wait->kind = IREE_HAL_AMD_XDNA_MEMORY_WAIT_FRONTIER;
        status = iree_async_frontier_tracker_wait(
            operation->alloca.pool->frontier_tracker, wait->frontier,
            iree_hal_amd_xdna_memory_wait_resolved, operation,
            &wait->frontier_waiter);
        if (iree_status_is_ok(status)) {
          return;
        }
        wait->kind = IREE_HAL_AMD_XDNA_MEMORY_WAIT_NONE;
        iree_hal_amd_xdna_operation_release_alloca_reservations(operation);
        operation->status = status;
        break;
      }
      case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
      case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
        wait->kind = IREE_HAL_AMD_XDNA_MEMORY_WAIT_CAPACITY;
        iree_hal_pool_wait_commit(
            wait->capacity_wait, iree_infinite_timeout(),
            (iree_hal_pool_wait_callback_t){
                .fn = iree_hal_amd_xdna_memory_wait_resolved,
                .user_data = operation,
            });
        return;
    }
  }
  iree_hal_amd_xdna_operation_complete(operation);
}

static void iree_hal_amd_xdna_queue_native_ready(
    void* user_data, iree_async_event_source_t* event_source,
    iree_async_poll_events_t events) {
  (void)event_source;
  iree_hal_amd_xdna_queue_t* queue = user_data;
  iree_status_t status = iree_async_event_consume(queue->event);
  iree_slim_mutex_lock(&queue->state_mutex);
  queue->notification_pending = false;
  if (iree_status_is_ok(status) && iree_async_poll_has_error(events)) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "XDNA native completion source reported poll events 0x%08X",
        (unsigned)events);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_queue_fail_pending(queue, status);
    iree_slim_mutex_unlock(&queue->state_mutex);
    return;
  }
  if (!iree_status_is_ok(queue->failure_status) || !queue->pending.count) {
    iree_slim_mutex_unlock(&queue->state_mutex);
    return;
  }
  amdf_kernel_queue_status_t checked = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
      .structure_size = sizeof(checked),
  };
  status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      queue->context->api->kernel_queue_refresh_status(queue->handle, &checked),
      "kernel_queue_refresh_status");
  iree_hal_amd_xdna_operation_t* retired_head = NULL;
  iree_hal_amd_xdna_operation_t* retired_tail = NULL;
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_queue_fail_pending(queue, status);
  } else {
    iree_atomic_store(&queue->native_publication_blocked, 0,
                      iree_memory_order_release);
    if (!amdf_status_is_ok(checked.terminal_status)) {
      iree_hal_amd_xdna_queue_record_failure(
          queue, IREE_HAL_AMD_STATUS_FROM_AMDF(checked.terminal_status,
                                               "XDNA command outcome"));
    } else if (checked.state != AMDF_QUEUE_STATE_ACTIVE) {
      iree_hal_amd_xdna_queue_record_failure(
          queue,
          iree_make_status(
              IREE_STATUS_INTERNAL,
              "XDNA queue became inactive without a terminal error (state=%d)",
              checked.state));
    }
    while (queue->pending.count &&
           queue->pending.entries[queue->pending.head].submission <=
               checked.retired_submission) {
      iree_hal_amd_xdna_pending_t* pending =
          &queue->pending.entries[queue->pending.head];
      iree_hal_amd_xdna_operation_t* operation = pending->operation;
      operation->status = iree_status_clone(queue->failure_status);
      if (queue->tracker && iree_status_is_ok(operation->status)) {
        iree_async_frontier_tracker_advance(queue->tracker, queue->axis,
                                            pending->epoch);
      }
      pending->operation = NULL;
      queue->pending.head = (queue->pending.head + 1) % queue->pending.capacity;
      --queue->pending.count;
      operation->next = NULL;
      if (retired_tail) {
        retired_tail->next = operation;
      } else {
        retired_head = operation;
      }
      retired_tail = operation;
    }
    iree_atomic_store(&queue->pending.visible_count,
                      (int32_t)queue->pending.count, iree_memory_order_release);
    if (queue->pending.count) {
      if (checked.state != AMDF_QUEUE_STATE_ACTIVE) {
        iree_hal_amd_xdna_queue_abandon_pending(queue);
      } else {
        status = iree_hal_amd_xdna_queue_arm(queue);
        if (!iree_status_is_ok(status)) {
          iree_hal_amd_xdna_queue_fail_pending(queue, status);
        }
      }
    }
  }
  if (!iree_status_is_ok(queue->failure_status)) {
    iree_hal_amd_xdna_queue_fail_dispatch_ready(queue);
  }
  const bool pump_after_publication =
      queue->dispatch_ready.head &&
      queue->pending.count < queue->pending.capacity &&
      iree_status_is_ok(queue->failure_status);
  iree_slim_mutex_unlock(&queue->state_mutex);
  while (retired_head) {
    iree_hal_amd_xdna_operation_t* operation = retired_head;
    retired_head = operation->next;
    operation->next = NULL;
    iree_hal_amd_xdna_operation_complete(operation);
  }
  if (pump_after_publication) {
    // The ready operation retains the device, so retirement publication cannot
    // destroy the queue before the newly available slot is handed off.
    iree_slim_mutex_lock(&queue->state_mutex);
    iree_hal_amd_xdna_queue_pump_dispatch(queue);
    iree_slim_mutex_unlock(&queue->state_mutex);
  }
}

// Releases the parent only from a poll callback after source unregistration has
// returned the event and a separate placed handoff has executed.
static void iree_hal_amd_xdna_queue_finalize(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  (void)base_operation;
  (void)flags;
  iree_hal_amd_xdna_queue_t* queue = user_data;
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_queue_report_status(queue, status);
    return;
  }
  queue->shutdown_complete = true;
  const iree_hal_amd_xdna_queue_shutdown_callback_t callback =
      queue->shutdown_callback;
  queue->shutdown_callback = (iree_hal_amd_xdna_queue_shutdown_callback_t){0};
  callback.fn(callback.user_data);
  // The callback releases |queue| and its proactor-owning parent.
}

static void iree_hal_amd_xdna_queue_schedule_finalization(
    iree_hal_amd_xdna_queue_t* queue) {
  iree_async_operation_initialize(
      &queue->finalization_operation, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_amd_xdna_queue_finalize, queue);
  iree_status_t status = iree_async_proactor_submit_one(
      queue->proactor, &queue->finalization_operation);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_queue_report_status(queue, status);
  }
}

static void iree_hal_amd_xdna_queue_observer_unregistered(
    void* user_data, iree_status_t status) {
  iree_hal_amd_xdna_queue_t* queue = user_data;
  if (!iree_status_is_ok(status)) {
    queue->observer_state = IREE_HAL_AMD_XDNA_OBSERVER_STATE_RETAINED;
    iree_hal_amd_xdna_queue_report_status(queue, status);
    return;
  }
  queue->observer_state = IREE_HAL_AMD_XDNA_OBSERVER_STATE_RETIRED;
  iree_hal_amd_xdna_queue_schedule_finalization(queue);
}

// Runs only on the poll owner and never from the persistent source callback.
static void iree_hal_amd_xdna_queue_shutdown_on_proactor(
    iree_hal_amd_xdna_queue_t* queue) {
  iree_atomic_store(&queue->native_admission_open, 0,
                    iree_memory_order_release);
  if (!queue->services_shutdown) {
    iree_hal_amd_xdna_queue_service_deinitialize(&queue->host_worker);
    iree_hal_amd_xdna_queue_service_deinitialize(&queue->publisher);
    queue->services_shutdown = true;
  }
  if (queue->handle) {
    amdf_status_t status =
        queue->context->api->kernel_queue_destroy(queue->handle);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_xdna_context_report(queue->context, status,
                                       "kernel_queue_destroy");
      // Failure does not prove that the native queue released its borrowed
      // notification target. Retain the complete reachable owner graph.
      return;
    }
    queue->handle = NULL;
  }

  if (queue->observer_state == IREE_HAL_AMD_XDNA_OBSERVER_STATE_UNAVAILABLE) {
    queue->observer_state = IREE_HAL_AMD_XDNA_OBSERVER_STATE_RETIRED;
    iree_hal_amd_xdna_queue_schedule_finalization(queue);
    return;
  }
  if (queue->observer_state != IREE_HAL_AMD_XDNA_OBSERVER_STATE_REGISTERED ||
      !queue->event_source) {
    iree_hal_amd_xdna_queue_report(
        queue, IREE_STATUS_INTERNAL,
        IREE_SV("XDNA queue shutdown lost persistent observer ownership"));
    return;
  }

  iree_async_event_source_t* event_source = queue->event_source;
  queue->event_source = NULL;
  queue->observer_state = IREE_HAL_AMD_XDNA_OBSERVER_STATE_UNREGISTERING;
  iree_async_proactor_unregister_event_source(
      queue->proactor, event_source,
      (iree_async_event_source_unregistered_callback_t){
          iree_hal_amd_xdna_queue_observer_unregistered, queue});
}

static void iree_hal_amd_xdna_queue_register_observer(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  (void)base_operation;
  (void)flags;
  iree_hal_amd_xdna_queue_t* queue = user_data;
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_register_event_source(
        queue->proactor, queue->event->native.wait_primitive,
        (iree_async_event_source_callback_t){
            iree_hal_amd_xdna_queue_native_ready, queue},
        &queue->event_source);
  }
  if (iree_status_is_ok(status)) {
    queue->observer_state = IREE_HAL_AMD_XDNA_OBSERVER_STATE_REGISTERED;
    iree_atomic_store(&queue->native_admission_open, 1,
                      iree_memory_order_release);
  } else {
    queue->observer_state = IREE_HAL_AMD_XDNA_OBSERVER_STATE_UNAVAILABLE;
    iree_hal_amd_xdna_queue_fail_pending(queue, status);
  }

  if (queue->shutdown_waiting_for_registration) {
    queue->shutdown_waiting_for_registration = false;
    iree_hal_amd_xdna_queue_shutdown_on_proactor(queue);
  } else if (queue->observer_state ==
             IREE_HAL_AMD_XDNA_OBSERVER_STATE_REGISTERED) {
    iree_hal_amd_xdna_queue_pump_dispatch(queue);
  }
}

static void iree_hal_amd_xdna_queue_shutdown_handoff(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  (void)base_operation;
  (void)flags;
  iree_hal_amd_xdna_queue_t* queue = user_data;
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_queue_report_status(queue, status);
    return;
  }
  if (queue->observer_state == IREE_HAL_AMD_XDNA_OBSERVER_STATE_REGISTERING) {
    queue->shutdown_waiting_for_registration = true;
    return;
  }
  iree_hal_amd_xdna_queue_shutdown_on_proactor(queue);
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
      operation->kind == IREE_HAL_AMD_XDNA_OPERATION_DISPATCH;
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
                                &values_offset)));
  const iree_host_size_t list_size = size;
  IREE_RETURN_IF_ERROR(
      IREE_STRUCT_LAYOUT(size, &size,
                         IREE_STRUCT_FIELD_ALIGNED(
                             producer_count, iree_hal_amd_xdna_producer_entry_t,
                             iree_alignof(iree_hal_amd_xdna_producer_entry_t),
                             &producers_offset)));
  const iree_host_size_t maximum_record_size =
      iree_arena_block_pool_max_allocation_size(operation->metadata_block_pool);
  const bool producers_are_packed = size <= maximum_record_size;
  if (!producers_are_packed) {
    size = list_size;
  }
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_queue_capture_allocate_metadata(
      &operation->capture, size, (void**)&storage));
  memset(storage, 0, size);

  iree_hal_amd_xdna_producer_entry_t* producer_entries = NULL;
  iree_host_size_t remaining_producer_count = producer_count;
  if (producers_are_packed) {
    iree_hal_amd_xdna_producer_entry_t* packed_entries =
        (iree_hal_amd_xdna_producer_entry_t*)(storage + producers_offset);
    for (iree_host_size_t i = 0; i < producer_count; ++i) {
      packed_entries[i].next = producer_entries;
      producer_entries = &packed_entries[i];
    }
    remaining_producer_count = 0;
  }
  const iree_host_size_t maximum_chunk_count =
      maximum_record_size / sizeof(iree_hal_amd_xdna_producer_entry_t);
  while (remaining_producer_count) {
    const iree_host_size_t chunk_count =
        iree_min(remaining_producer_count, maximum_chunk_count);
    iree_host_size_t chunk_size = 0;
    iree_host_size_t chunk_offset = 0;
    IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
        0, &chunk_size,
        IREE_STRUCT_FIELD_ALIGNED(
            chunk_count, iree_hal_amd_xdna_producer_entry_t,
            iree_alignof(iree_hal_amd_xdna_producer_entry_t), &chunk_offset)));
    uint8_t* chunk_storage = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_queue_capture_allocate_metadata(
        &operation->capture, chunk_size, (void**)&chunk_storage));
    memset(chunk_storage, 0, chunk_size);
    iree_hal_amd_xdna_producer_entry_t* chunk_entries =
        (iree_hal_amd_xdna_producer_entry_t*)(chunk_storage + chunk_offset);
    for (iree_host_size_t i = 0; i < chunk_count; ++i) {
      chunk_entries[i].next = producer_entries;
      producer_entries = &chunk_entries[i];
    }
    remaining_producer_count -= chunk_count;
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
    operation->dispatch.producer_entries = producer_entries;
  }
  iree_hal_semaphore_list_retain(captured_list);
  *out_list = captured_list;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_operation_create(
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
    iree_atomic_store(&operation->memory_status, 0, iree_memory_order_relaxed);
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

// Attempts caller-side native publication without waiting for ownership of the
// publisher. Native submission itself may still wait for device wake or native
// credits because libamdf does not expose a nonblocking try-submit contract.
// Returns true once ownership of |operation| has been consumed.
static bool iree_hal_amd_xdna_operation_try_direct_publish(
    iree_hal_amd_xdna_operation_t* operation,
    iree_status_t* out_submission_status) {
  *out_submission_status = iree_ok_status();
  if (operation->kind != IREE_HAL_AMD_XDNA_OPERATION_DISPATCH) {
    return false;
  }

  iree_hal_amd_xdna_queue_t* queue = operation->queue;
  if (!iree_slim_mutex_try_lock(&queue->state_mutex)) {
    return false;
  }
  if (!iree_atomic_load(&queue->native_admission_open,
                        iree_memory_order_acquire) ||
      iree_atomic_load(&queue->native_publication_blocked,
                       iree_memory_order_acquire) ||
      !iree_hal_amd_xdna_queue_try_claim_publisher(queue)) {
    iree_slim_mutex_unlock(&queue->state_mutex);
    return false;
  }

  const int32_t pending_count = iree_atomic_load(&queue->pending.visible_count,
                                                 iree_memory_order_acquire);
  if (!iree_atomic_load(&queue->native_admission_open,
                        iree_memory_order_acquire) ||
      iree_atomic_load(&queue->native_publication_blocked,
                       iree_memory_order_acquire) ||
      pending_count >= (int32_t)queue->pending.capacity) {
    iree_hal_amd_xdna_queue_release_publisher(queue);
    iree_slim_mutex_unlock(&queue->state_mutex);
    return false;
  }

  iree_hal_amd_xdna_wait_resolution_t resolution =
      IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_READY;
  iree_status_t wait_status = iree_hal_amd_xdna_operation_resolve_waits(
      operation, &resolution, /*out_deferred_wait_index=*/NULL);
  const bool should_publish = iree_status_is_ok(wait_status);
  if (!iree_status_is_ok(wait_status)) {
    operation->status = wait_status;
    iree_atomic_store(&queue->publisher_operation, (intptr_t)operation,
                      iree_memory_order_release);
  } else {
    if (resolution != IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_READY ||
        !operation->frontier.exact) {
      iree_hal_amd_xdna_queue_release_publisher(queue);
      iree_slim_mutex_unlock(&queue->state_mutex);
      return false;
    }
    iree_atomic_store(&queue->publisher_operation, (intptr_t)operation,
                      iree_memory_order_release);
  }
  iree_slim_mutex_unlock(&queue->state_mutex);

  // Native publication may block, so it runs without excluding proactor state
  // transitions. The publisher claim preserves physical acceptance order.
  // Commit the result on this caller after reacquiring queue-state ownership;
  // queued arrivals may have attached to the in-flight publisher meanwhile.
  if (should_publish) {
    iree_hal_amd_xdna_publisher_execute(queue, &operation->service_item);
  }
  iree_slim_mutex_lock(&queue->state_mutex);
  iree_hal_amd_xdna_queue_commit_publication(
      queue, /*service=*/NULL, /*item=*/NULL, operation, iree_ok_status());
  iree_slim_mutex_unlock(&queue->state_mutex);
  return true;
}

iree_status_t iree_hal_amd_xdna_operation_submit(
    iree_hal_amd_xdna_operation_t* operation) {
  iree_status_t status = iree_ok_status();
  if (iree_hal_amd_xdna_operation_try_direct_publish(operation, &status)) {
    return status;
  }
  status = iree_async_proactor_submit_one(operation->queue->proactor,
                                          &operation->admission);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_operation_discard(operation);
  }
  return status;
}

static iree_status_t iree_hal_amd_xdna_queue_barrier(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_queue_barrier_flags_t flags) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_queue_barriers_validate(barriers));
  iree_hal_amd_xdna_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_operation_create(
      (iree_hal_amd_xdna_queue_t*)base, waits, signals,
      IREE_HAL_AMD_XDNA_OPERATION_BARRIER, &operation));
  return iree_hal_amd_xdna_operation_submit(operation);
}

static iree_status_t iree_hal_amd_xdna_queue_dispatch(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_dispatch_flags_t flags) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_queue_barriers_validate(barriers));
  const iree_hal_dispatch_flags_t supported_flags =
      IREE_HAL_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION |
      IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES;
  if ((flags & ~supported_flags) || config.workgroup_count_ref.buffer ||
      config.dynamic_workgroup_local_memory || constants.data_length) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "XDNA dispatch requires direct buffer bindings");
  }
  for (uint32_t i = 0; i < 3; ++i) {
    if (config.workgroup_count[i] > 1 || config.workgroup_size[i] > 1) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "XDNA function is one compiled array invocation");
    }
  }
  if (!config.workgroup_count[0] || !config.workgroup_count[1] ||
      !config.workgroup_count[2]) {
    return iree_hal_amd_xdna_queue_barrier(base, waits, signals, barriers, 0);
  }
  iree_host_size_t binding_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &binding_length,
      IREE_STRUCT_FIELD_FAM(bindings.count,
                            iree_hal_amd_xdna_executable_binding_t)));
  iree_hal_amd_xdna_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_operation_create(
      (iree_hal_amd_xdna_queue_t*)base, waits, signals,
      IREE_HAL_AMD_XDNA_OPERATION_DISPATCH, &operation));
  void* binding_storage = NULL;
  iree_status_t status = iree_hal_amd_xdna_queue_capture_allocate_metadata(
      &operation->capture, binding_length, &binding_storage);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_executable_resolve(
        executable, function, bindings, binding_storage,
        &operation->dispatch.function);
  }
  if (iree_status_is_ok(status)) {
    operation->dispatch.executable = executable;
    iree_hal_executable_retain(executable);
    operation->dispatch.binding_count = bindings.count;
    operation->dispatch.bindings = binding_storage;
    for (iree_host_size_t i = 0; i < bindings.count; ++i) {
      iree_hal_buffer_retain(operation->dispatch.bindings[i].buffer_ref.buffer);
    }
    status = iree_hal_amd_xdna_operation_submit(operation);
  } else {
    iree_hal_amd_xdna_operation_discard(operation);
  }
  return status;
}

iree_status_t iree_hal_amd_xdna_queue_assign_frontier(
    iree_hal_queue_t* base, iree_async_frontier_tracker_t* tracker,
    iree_async_axis_t axis) {
  iree_hal_amd_xdna_queue_t* queue = (iree_hal_amd_xdna_queue_t*)base;
  if (tracker) {
    IREE_RETURN_IF_ERROR(
        iree_async_frontier_tracker_register_axis(tracker, axis, NULL));
    iree_async_frontier_tracker_retain(tracker);
  }
  if (queue->tracker) {
    iree_async_frontier_tracker_retire_axis(
        queue->tracker, queue->axis,
        iree_status_from_code(IREE_STATUS_CANCELLED));
    iree_async_frontier_tracker_release(queue->tracker);
  }
  queue->tracker = tracker;
  queue->axis = axis;
  queue->epoch = 0;
  iree_hal_amd_xdna_frontier_state_initialize(&queue->accepted_frontier);
  return iree_ok_status();
}

void iree_hal_amd_xdna_queue_begin_shutdown(
    iree_hal_queue_t* base,
    iree_hal_amd_xdna_queue_shutdown_callback_t callback) {
  iree_hal_amd_xdna_queue_t* queue = (iree_hal_amd_xdna_queue_t*)base;
  iree_atomic_store(&queue->native_admission_open, 0,
                    iree_memory_order_release);
  queue->shutdown_callback = callback;
  iree_async_operation_initialize(
      &queue->shutdown_operation, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_amd_xdna_queue_shutdown_handoff,
      queue);
  iree_status_t status = iree_async_proactor_submit_one(
      queue->proactor, &queue->shutdown_operation);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_queue_report_status(queue, status);
  }
}

static void iree_hal_amd_xdna_queue_destroy(iree_hal_queue_t* base) {
  iree_hal_amd_xdna_queue_t* queue = (iree_hal_amd_xdna_queue_t*)base;
  if (!queue->shutdown_complete &&
      queue->observer_state != IREE_HAL_AMD_XDNA_OBSERVER_STATE_UNINITIALIZED) {
    iree_hal_amd_xdna_queue_report(
        queue, IREE_STATUS_INTERNAL,
        IREE_SV("XDNA queue released before terminal observer shutdown"));
    return;
  }
  if (!queue->services_shutdown) {
    iree_hal_amd_xdna_queue_service_deinitialize(&queue->host_worker);
    iree_hal_amd_xdna_queue_service_deinitialize(&queue->publisher);
    queue->services_shutdown = true;
  }
  if (queue->handle) {
    amdf_status_t status =
        queue->context->api->kernel_queue_destroy(queue->handle);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_xdna_context_report(queue->context, status,
                                       "kernel_queue_destroy");
      return;
    }
    queue->handle = NULL;
  }
  if (queue->tracker) {
    iree_async_frontier_tracker_retire_axis(
        queue->tracker, queue->axis,
        iree_status_from_code(IREE_STATUS_CANCELLED));
    iree_async_frontier_tracker_release(queue->tracker);
  }
  iree_status_free(queue->failure_status);
  iree_async_event_release(queue->event);
  iree_allocator_free(queue->host_allocator, queue->pending.entries);
  iree_hal_amd_xdna_queue_storage_deinitialize(&queue->storage);
  iree_slim_mutex_deinitialize(&queue->state_mutex);
  iree_allocator_free(queue->host_allocator, queue);
}

iree_status_t iree_hal_amd_xdna_queue_create(
    const iree_hal_queue_family_t* family, iree_hal_amd_xdna_context_t* context,
    iree_async_proactor_t* proactor, iree_allocator_t host_allocator,
    iree_hal_queue_t** out_queue) {
  *out_queue = NULL;
  iree_hal_amd_xdna_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*queue), (void**)&queue));
  memset(queue, 0, sizeof(*queue));
  iree_hal_amd_xdna_queue_storage_initialize(host_allocator, &queue->storage);
  iree_hal_amd_xdna_queue_producer_index_initialize(&queue->producer_index);
  iree_slim_mutex_initialize(&queue->state_mutex);
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  iree_hal_queue_initialize(family, &params, &iree_hal_amd_xdna_queue_vtable,
                            &queue->base);
  queue->host_allocator = host_allocator;
  queue->context = context;
  queue->proactor = proactor;
  iree_atomic_store(&queue->publisher_operation, 0, iree_memory_order_relaxed);
  iree_atomic_store(&queue->publisher_claim, 0, iree_memory_order_relaxed);
  iree_atomic_store(&queue->native_admission_open, 0,
                    iree_memory_order_relaxed);
  iree_atomic_store(&queue->native_publication_blocked, 0,
                    iree_memory_order_relaxed);
  iree_atomic_store(&queue->pending.visible_count, 0,
                    iree_memory_order_relaxed);
  iree_hal_amd_xdna_frontier_state_initialize(&queue->accepted_frontier);
  const amdf_xdna_kernel_queue_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .queue_family_ordinal = context->queue_family_ordinal,
      .maximum_pending_submission_count = 0,
  };
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->xdna->kernel_queue_create(context->handle, &create_info,
                                         &queue->handle),
      "xdna.kernel_queue_create");
  amdf_kernel_queue_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO,
      .structure_size = sizeof(info),
  };
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        context->api->kernel_queue_query_info(queue->handle, &info),
        "kernel_queue_query_info");
  }
  if (iree_status_is_ok(status)) {
    queue->pending.capacity = info.maximum_pending_submission_count;
    if (!queue->pending.capacity || queue->pending.capacity > INT32_MAX) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "XDNA native pending capacity %u is outside the HAL range",
          queue->pending.capacity);
    }
    iree_host_size_t pending_size = 0;
    if (iree_status_is_ok(status)) {
      status = IREE_STRUCT_LAYOUT(
          0, &pending_size,
          IREE_STRUCT_FIELD_FAM(queue->pending.capacity,
                                iree_hal_amd_xdna_pending_t));
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(host_allocator, pending_size,
                                     (void**)&queue->pending.entries);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_event_create(proactor, &queue->event);
  }
  if (iree_status_is_ok(status)) {
#if defined(IREE_ASYNC_HAVE_EVENTFD)
    queue->native_event.type = AMDF_NATIVE_EVENT_TYPE_EVENTFD;
    queue->native_event.payload.file_descriptor =
        queue->event->native.signal_primitive.value.fd;
#elif defined(IREE_ASYNC_HAVE_WIN32_HANDLE)
    queue->native_event.type = AMDF_NATIVE_EVENT_TYPE_WIN32_EVENT;
    queue->native_event.payload.native_handle =
        (void*)queue->event->native.signal_primitive.value.win32_handle;
#endif
    if (!iree_any_bit_set(info.notification_types,
                          UINT64_C(1) << queue->native_event.type)) {
      status = iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "XDNA queue has no compatible native wake destination");
    }
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_amd_xdna_queue_service_callbacks_t publisher_callbacks = {
        .execute = iree_hal_amd_xdna_publisher_execute,
        .complete = iree_hal_amd_xdna_publisher_complete,
        .abandon = iree_hal_amd_xdna_service_abandon,
    };
    status = iree_hal_amd_xdna_queue_service_initialize(
        IREE_SV("xdna-publish"), proactor, publisher_callbacks, queue,
        host_allocator, &queue->publisher);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_amd_xdna_queue_service_callbacks_t host_callbacks = {
        .execute = iree_hal_amd_xdna_host_worker_execute,
        .complete = iree_hal_amd_xdna_host_worker_complete,
        .abandon = iree_hal_amd_xdna_service_abandon,
    };
    status = iree_hal_amd_xdna_queue_service_initialize(
        IREE_SV("xdna-host"), proactor, host_callbacks, queue, host_allocator,
        &queue->host_worker);
  }
  if (iree_status_is_ok(status)) {
    queue->observer_state = IREE_HAL_AMD_XDNA_OBSERVER_STATE_REGISTERING;
    iree_async_operation_initialize(
        &queue->observer_registration, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE,
        iree_hal_amd_xdna_queue_register_observer, queue);
    status =
        iree_async_proactor_submit_one(proactor, &queue->observer_registration);
    if (!iree_status_is_ok(status)) {
      queue->observer_state = IREE_HAL_AMD_XDNA_OBSERVER_STATE_UNINITIALIZED;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_queue = &queue->base;
  } else {
    iree_hal_queue_release(&queue->base);
  }
  return status;
}

void iree_hal_amd_xdna_queue_trim(iree_hal_queue_t* base) {
  iree_hal_amd_xdna_queue_t* queue = (iree_hal_amd_xdna_queue_t*)base;
  iree_hal_amd_xdna_queue_storage_trim(&queue->storage);
}

static iree_status_t iree_hal_amd_xdna_queue_flush(iree_hal_queue_t* queue) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_queue_execute(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_queue_execute_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "XDNA queue does not support execute");
}

static iree_status_t iree_hal_amd_xdna_queue_host_call(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "XDNA queue does not support host_call");
}

static iree_status_t iree_hal_amd_xdna_queue_query_dispatch_concurrency(
    iree_hal_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function,
    iree_hal_queue_dispatch_concurrency_params_t params,
    iree_hal_queue_dispatch_concurrency_flags_t flags,
    iree_hal_queue_dispatch_concurrency_t* out_concurrency) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "XDNA queue does not support query_dispatch_concurrency");
}

static iree_status_t iree_hal_amd_xdna_queue_atomic_wait(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_wait_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "XDNA queue does not support atomic_wait");
}

static iree_status_t iree_hal_amd_xdna_queue_atomic_store(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_store_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "XDNA queue does not support atomic_store");
}

static iree_status_t iree_hal_amd_xdna_queue_atomic_rmw(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_rmw_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "XDNA queue does not support atomic_rmw");
}

static iree_status_t iree_hal_amd_xdna_queue_timestamp(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_timestamp_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "XDNA queue does not support timestamp");
}

static iree_status_t iree_hal_amd_xdna_queue_read(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const iree_hal_queue_barriers_t* barriers,
    iree_hal_read_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "XDNA queue does not support read");
}

static iree_status_t iree_hal_amd_xdna_queue_write(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, const iree_hal_queue_barriers_t* barriers,
    iree_hal_write_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "XDNA queue does not support write");
}

static const iree_hal_queue_vtable_t iree_hal_amd_xdna_queue_vtable = {
    .destroy = iree_hal_amd_xdna_queue_destroy,
    .barrier = iree_hal_amd_xdna_queue_barrier,
    .execute = iree_hal_amd_xdna_queue_execute,
    .host_call = iree_hal_amd_xdna_queue_host_call,
    .query_dispatch_concurrency =
        iree_hal_amd_xdna_queue_query_dispatch_concurrency,
    .dispatch = iree_hal_amd_xdna_queue_dispatch,
    .atomic_wait = iree_hal_amd_xdna_queue_atomic_wait,
    .atomic_store = iree_hal_amd_xdna_queue_atomic_store,
    .atomic_rmw = iree_hal_amd_xdna_queue_atomic_rmw,
    .timestamp = iree_hal_amd_xdna_queue_timestamp,
    .flush = iree_hal_amd_xdna_queue_flush,
    .alloca = iree_hal_amd_xdna_queue_alloca,
    .dealloca = iree_hal_amd_xdna_queue_dealloca,
    .transfer = iree_hal_amd_xdna_queue_transfer,
    .read = iree_hal_amd_xdna_queue_read,
    .write = iree_hal_amd_xdna_queue_write,
};
