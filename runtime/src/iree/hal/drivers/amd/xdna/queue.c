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
#include "iree/hal/drivers/amd/fabric/event.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/drivers/amd/xdna/barrier.h"
#include "iree/hal/drivers/amd/xdna/executable.h"
#include "iree/hal/drivers/amd/xdna/queue_admission.h"
#include "iree/hal/drivers/amd/xdna/queue_frontier.h"
#include "iree/hal/drivers/amd/xdna/queue_operation.h"
#include "iree/hal/drivers/amd/xdna/queue_producer_index.h"
#include "iree/hal/drivers/amd/xdna/queue_service.h"
#include "iree/hal/drivers/amd/xdna/queue_state.h"
#include "iree/hal/drivers/amd/xdna/queue_storage.h"
#include "iree/hal/drivers/amd/xdna/semaphore.h"

// Ordered operations detached from the native pending ring for publication.
typedef struct iree_hal_amd_xdna_retired_batch_t {
  // Oldest detached operation.
  iree_hal_amd_xdna_operation_t* head;
  // Newest detached operation.
  iree_hal_amd_xdna_operation_t* tail;
} iree_hal_amd_xdna_retired_batch_t;

typedef uint32_t iree_hal_amd_xdna_direct_wait_flags_t;
enum iree_hal_amd_xdna_direct_wait_flag_bits_e {
  IREE_HAL_AMD_XDNA_DIRECT_WAIT_FLAG_NONE = 0u,
  // One exact host waiter owns checked native retirement.
  IREE_HAL_AMD_XDNA_DIRECT_WAIT_FLAG_ACTIVE = 1u << 0,
  // The waiter's outstanding native completion hint was consumed.
  IREE_HAL_AMD_XDNA_DIRECT_WAIT_FLAG_EVENT_OBSERVED = 1u << 1,
};

static const iree_hal_queue_vtable_t iree_hal_amd_xdna_queue_vtable;

void iree_hal_amd_xdna_queue_report(iree_hal_amd_xdna_queue_t* queue,
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

static iree_hal_amd_xdna_operation_t*
iree_hal_amd_xdna_operation_from_service_item(
    iree_hal_amd_xdna_queue_service_item_t* item) {
  return iree_containerof(item, iree_hal_amd_xdna_operation_t, service_item);
}

iree_hal_amd_xdna_operation_t* iree_hal_amd_xdna_queue_publisher_operation(
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
  return iree_status_is_ok(iree_hal_amd_xdna_queue_failure_status(queue)) &&
         queue->observer_state == IREE_HAL_AMD_XDNA_OBSERVER_STATE_REGISTERED &&
         !iree_atomic_load(&queue->native_publication_blocked,
                           iree_memory_order_acquire) &&
         queue->dispatch_ready.head &&
         queue->pending.count < queue->pending.capacity;
}

// Reserves the invisible free suffix while the publisher claim is held.
// Retirement can advance the head concurrently without changing this tail.
static void iree_hal_amd_xdna_queue_reserve_publication(
    iree_hal_amd_xdna_queue_t* queue,
    iree_hal_amd_xdna_operation_t* operation) {
  operation->native.publication.start =
      (queue->pending.head + queue->pending.count) % queue->pending.capacity;
  operation->native.publication.capacity =
      queue->pending.capacity - queue->pending.count;
  operation->native.publication.count = 0;
  operation->native.publication.status = iree_ok_status();
  operation->native.publication.result =
      IREE_HAL_AMD_XDNA_PUBLICATION_RESULT_REJECTED;
  operation->native.publication.had_pending = false;
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
  iree_hal_amd_xdna_queue_reserve_publication(queue, operation);
  iree_atomic_store(&queue->publisher_operation, (intptr_t)operation,
                    iree_memory_order_release);
  if (operation->profile) {
    operation->profile->flags |=
        IREE_HAL_AMD_XDNA_OPERATION_PROFILE_FLAG_DEFERRED;
  }
  iree_hal_amd_xdna_queue_service_enqueue(&queue->publisher,
                                          &operation->service_item);
}

// Admits at most one publisher item. Its proactor acknowledgement advances the
// accepted frontier before this function may admit the next physical command.
void iree_hal_amd_xdna_queue_pump_dispatch(iree_hal_amd_xdna_queue_t* queue) {
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
    if (operation->native.phase != IREE_HAL_AMD_XDNA_NATIVE_PHASE_ABANDONED) {
      operation->status = iree_status_join(
          operation->status,
          iree_status_clone(iree_hal_amd_xdna_queue_failure_status(queue)));
      operation->native.phase = IREE_HAL_AMD_XDNA_NATIVE_PHASE_COMPLETE;
      if (!operation->native.pending_count) {
        iree_hal_amd_xdna_operation_complete(operation);
      }
    }
    operation = next;
  }
}

static void iree_hal_amd_xdna_queue_record_failure(
    iree_hal_amd_xdna_queue_t* queue, iree_status_t status) {
  if (iree_status_is_ok(iree_hal_amd_xdna_queue_failure_status(queue))) {
    iree_atomic_store(&queue->failure_status, (intptr_t)status,
                      iree_memory_order_release);
  } else {
    iree_hal_amd_xdna_queue_report(queue, iree_status_code(status),
                                   iree_status_message(status));
    iree_status_free(status);
  }
  iree_atomic_store(&queue->native_admission_open, 0,
                    iree_memory_order_release);
}

// A failed observer cannot prove accepted device references have retired. Keep
// each remaining operation and its parent graph live, and fail completion
// edges. Native progress is never inferred from a failure signal.
static void iree_hal_amd_xdna_queue_abandon_pending(
    iree_hal_amd_xdna_queue_t* queue) {
  iree_hal_amd_xdna_queue_report(
      queue, iree_status_code(iree_hal_amd_xdna_queue_failure_status(queue)),
      iree_status_message(iree_hal_amd_xdna_queue_failure_status(queue)));
  while (queue->pending.count) {
    iree_hal_amd_xdna_pending_t* pending =
        &queue->pending.entries[queue->pending.head];
    iree_hal_amd_xdna_operation_t* operation = pending->operation;
    if (operation->native.phase != IREE_HAL_AMD_XDNA_NATIVE_PHASE_ABANDONED) {
      operation->native.phase = IREE_HAL_AMD_XDNA_NATIVE_PHASE_ABANDONED;
      iree_hal_amd_xdna_operation_t* waiters =
          iree_hal_amd_xdna_operation_unregister_producers(operation);
      iree_hal_semaphore_list_fail(
          operation->signals,
          iree_status_clone(iree_hal_amd_xdna_queue_failure_status(queue)));
      iree_hal_amd_xdna_operation_wake_acceptance_waiters(waiters);
    }
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

// Applies one checked libamdf snapshot while |state_mutex| is held. Detached
// operations remain owned by |out_retired| until their signals and resources
// are published under |completion_drain_mutex|.
static bool iree_hal_amd_xdna_queue_apply_checked_status(
    iree_hal_amd_xdna_queue_t* queue, const amdf_kernel_queue_status_t* checked,
    iree_hal_amd_xdna_retired_batch_t* out_retired) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, checked->retired_submission);
  IREE_TRACE(const uint32_t pending_count = queue->pending.count);
  iree_atomic_store(&queue->native_publication_blocked, 0,
                    iree_memory_order_release);
  if (!amdf_status_is_ok(checked->terminal_status)) {
    iree_hal_amd_xdna_queue_record_failure(
        queue, IREE_HAL_AMD_STATUS_FROM_AMDF(checked->terminal_status,
                                             "XDNA command outcome"));
  } else if (checked->state != AMDF_QUEUE_STATE_ACTIVE) {
    iree_hal_amd_xdna_queue_record_failure(
        queue,
        iree_make_status(
            IREE_STATUS_INTERNAL,
            "XDNA queue became inactive without a terminal error (state=%d)",
            checked->state));
  }

  while (queue->pending.count &&
         queue->pending.entries[queue->pending.head].submission <=
             checked->retired_submission) {
    iree_hal_amd_xdna_pending_t* pending =
        &queue->pending.entries[queue->pending.head];
    iree_hal_amd_xdna_operation_t* operation = pending->operation;
    if (iree_status_is_ok(operation->status)) {
      operation->status =
          iree_status_clone(iree_hal_amd_xdna_queue_failure_status(queue));
    }
    iree_hal_amd_xdna_invocation_release(pending->invocation);
    pending->invocation = NULL;
    --operation->native.pending_count;
    if (queue->tracker &&
        iree_status_is_ok(iree_hal_amd_xdna_queue_failure_status(queue))) {
      iree_async_frontier_tracker_advance(queue->tracker, queue->axis,
                                          pending->epoch);
    }
    pending->operation = NULL;
    queue->pending.head = (queue->pending.head + 1) % queue->pending.capacity;
    --queue->pending.count;
    if (operation->native.pending_count ||
        operation->native.phase == IREE_HAL_AMD_XDNA_NATIVE_PHASE_ISSUING ||
        operation->native.phase == IREE_HAL_AMD_XDNA_NATIVE_PHASE_ABANDONED) {
      continue;
    }
    if (operation->native.phase == IREE_HAL_AMD_XDNA_NATIVE_PHASE_DRAINING &&
        iree_status_is_ok(operation->status)) {
      iree_hal_amd_xdna_queue_service_enqueue(&queue->host_worker,
                                              &operation->service_item);
      continue;
    }
    iree_hal_amd_xdna_operation_wake_acceptance_waiters(
        iree_hal_amd_xdna_operation_unregister_producers(operation));
    operation->next = NULL;
    if (out_retired->tail) {
      out_retired->tail->next = operation;
    } else {
      out_retired->head = operation;
    }
    out_retired->tail = operation;
  }
  iree_atomic_store(&queue->pending.visible_count,
                    (int32_t)queue->pending.count, iree_memory_order_release);

  if (queue->pending.count) {
    if (checked->state != AMDF_QUEUE_STATE_ACTIVE) {
      iree_hal_amd_xdna_queue_abandon_pending(queue);
    } else if (!queue->notification_pending) {
      iree_status_t status = iree_hal_amd_xdna_queue_arm(queue);
      if (!iree_status_is_ok(status)) {
        iree_hal_amd_xdna_queue_fail_pending(queue, status);
      }
    }
  }
  if (!iree_status_is_ok(iree_hal_amd_xdna_queue_failure_status(queue))) {
    iree_hal_amd_xdna_queue_fail_dispatch_ready(queue);
  }
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, pending_count - queue->pending.count);
  IREE_TRACE_ZONE_END(z0);
  return queue->dispatch_ready.head &&
         queue->pending.count < queue->pending.capacity &&
         iree_status_is_ok(iree_hal_amd_xdna_queue_failure_status(queue));
}

// Publishes a detached FIFO prefix while |completion_drain_mutex| is held.
static void iree_hal_amd_xdna_queue_complete_retired_batch(
    iree_hal_amd_xdna_retired_batch_t* retired) {
  while (retired->head) {
    iree_hal_amd_xdna_operation_t* operation = retired->head;
    retired->head = operation->next;
    operation->next = NULL;
    iree_hal_amd_xdna_operation_complete(operation);
  }
  retired->tail = NULL;
}

static void iree_hal_amd_xdna_queue_pump_after_retirement(
    iree_hal_amd_xdna_queue_t* queue, bool should_pump) {
  if (!should_pump) {
    return;
  }
  // The ready operation retains the device, so retirement publication cannot
  // destroy the queue before the newly available slot is handed off.
  iree_slim_mutex_lock(&queue->state_mutex);
  iree_hal_amd_xdna_queue_pump_dispatch(queue);
  iree_slim_mutex_unlock(&queue->state_mutex);
}

static void iree_hal_amd_xdna_queue_publish_signals(
    iree_hal_amd_xdna_queue_t* queue, iree_hal_amd_xdna_operation_t* operation,
    uint64_t epoch) {
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
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_amd_xdna_queue_t* queue = user_data;
  iree_hal_amd_xdna_operation_t* operation =
      iree_hal_amd_xdna_operation_from_service_item(item);
  iree_time_t profile_start = 0;
  if (operation->profile) {
    iree_hal_amd_xdna_operation_profile_ready(operation);
    profile_start = iree_hal_amd_xdna_profile_host_begin(queue->profile);
  }
  while (operation->native.publication.count <
             operation->native.publication.capacity &&
         iree_status_is_ok(operation->native.publication.status) &&
         iree_atomic_load(&queue->native_admission_open,
                          iree_memory_order_acquire)) {
    uint32_t command_count = 1;
    amdf_xdna_kernel_command_t* commands = &operation->native.command;
    iree_hal_amd_xdna_invocation_t** invocations =
        &operation->native.invocation;
    if (operation->kind == IREE_HAL_AMD_XDNA_OPERATION_EXECUTE) {
      operation->native.publication.status =
          iree_hal_amd_xdna_execution_prepare_batch(
              operation->execution, operation->native.publication.capacity -
                                        operation->native.publication.count);
      command_count = operation->execution->batch.count;
      commands = operation->execution->batch.commands;
      invocations = operation->execution->batch.invocations;
    } else if (!operation->native.invocation) {
      operation->native.publication.status = iree_hal_amd_xdna_function_prepare(
          operation->dispatch.function, operation->dispatch.bindings,
          &operation->native.invocation, &operation->native.command);
    }
    if (!iree_status_is_ok(operation->native.publication.status)) {
      break;
    }
    if (operation->profile) {
      iree_hal_amd_xdna_operation_profile_prepare_native(
          operation, command_count, invocations, commands);
    }
    const amdf_xdna_kernel_queue_submission_info_t submission_info = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO,
        .structure_size = sizeof(submission_info),
        .command_count = command_count,
        .commands = commands,
    };
    operation->native.publication.had_pending =
        operation->native.publication.count ||
        iree_atomic_load(&queue->pending.visible_count,
                         iree_memory_order_acquire) > 0;
    uint64_t submission = 0;
    IREE_TRACE_ZONE_BEGIN_NAMED(z1, "iree_hal_amd_xdna_native_submit");
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z1, command_count);
    const amdf_status_t submit_status =
        queue->context->xdna->kernel_queue_submit(
            queue->handle, &submission_info, &submission);
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z1, submission);
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z1, submit_status);
    IREE_TRACE_ZONE_END(z1);
    if (submit_status == amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
      operation->native.publication.result =
          IREE_HAL_AMD_XDNA_PUBLICATION_RESULT_RETRY;
      break;
    }
    operation->native.publication.status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        submit_status, "xdna.kernel_queue_submit");
    if (!iree_status_is_ok(operation->native.publication.status)) {
      break;
    }
    if (operation->profile) {
      iree_hal_amd_xdna_operation_profile_accept_native(
          operation, command_count, invocations, commands);
    }
    // Each invocation retains its own immutable instruction range, while one
    // checked native point retires the complete accepted array.
    for (uint32_t i = 0; i < command_count; ++i) {
      const uint32_t index = (operation->native.publication.start +
                              operation->native.publication.count + i) %
                             queue->pending.capacity;
      queue->pending.entries[index] = (iree_hal_amd_xdna_pending_t){
          .operation = operation,
          .invocation = invocations[i],
          .submission = submission,
      };
    }
    operation->native.publication.count += command_count;
    operation->native.publication.result =
        IREE_HAL_AMD_XDNA_PUBLICATION_RESULT_ACCEPTED;
    if (operation->kind == IREE_HAL_AMD_XDNA_OPERATION_DISPATCH) {
      operation->native.invocation = NULL;
      break;
    }
    operation->execution->cursor = operation->execution->batch.next;
    operation->execution->batch.count = 0;
    if (!iree_hal_amd_xdna_operation_has_dispatch(operation)) {
      break;
    }
  }
  if (profile_start) {
    iree_hal_amd_xdna_operation_profile_host_end(
        operation, profile_start,
        iree_status_code(operation->native.publication.status),
        operation->native.publication.count);
  }
  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_amd_xdna_host_worker_execute(
    void* user_data, iree_hal_amd_xdna_queue_service_item_t* item) {
  iree_hal_amd_xdna_queue_t* queue = user_data;
  iree_hal_amd_xdna_operation_t* operation =
      iree_hal_amd_xdna_operation_from_service_item(item);
  iree_time_t start = 0;
  if (operation->profile) {
    operation->profile->flags |=
        IREE_HAL_AMD_XDNA_OPERATION_PROFILE_FLAG_DEFERRED;
    iree_hal_amd_xdna_operation_profile_ready(operation);
    start = iree_hal_amd_xdna_profile_host_begin(queue->profile);
  }
  const uint32_t count = iree_hal_amd_xdna_operation_execute_host(operation);
  if (start) {
    iree_hal_amd_xdna_operation_profile_host_end(
        operation, start, iree_status_code(operation->status), count);
  }
}

static void iree_hal_amd_xdna_service_abandon(
    void* user_data, iree_hal_amd_xdna_queue_service_item_t* item,
    iree_status_t status) {
  iree_hal_amd_xdna_queue_t* queue = (iree_hal_amd_xdna_queue_t*)user_data;
  iree_atomic_store(&queue->native_admission_open, 0,
                    iree_memory_order_release);
  iree_hal_amd_xdna_operation_t* operation =
      iree_hal_amd_xdna_operation_from_service_item(item);
  // Native replay may already have a visible prefix retiring on another thread.
  iree_slim_mutex_lock(&queue->state_mutex);
  status = iree_status_join(operation->status, status);
  status = iree_status_join(operation->native.publication.status, status);
  operation->native.publication.status = iree_ok_status();
  operation->native.phase = IREE_HAL_AMD_XDNA_NATIVE_PHASE_ABANDONED;
  operation->status = iree_ok_status();
  iree_hal_amd_xdna_queue_report(queue, iree_status_code(status),
                                 iree_status_message(status));
  iree_hal_semaphore_list_fail(operation->signals, status);
  iree_slim_mutex_unlock(&queue->state_mutex);
}

void iree_hal_amd_xdna_queue_enqueue_ready(
    iree_hal_amd_xdna_operation_t* operation, iree_status_t status) {
  iree_hal_amd_xdna_queue_t* queue = operation->queue;
  operation->status = status;
  if (iree_status_is_ok(operation->status) &&
      !iree_status_is_ok(iree_hal_amd_xdna_queue_failure_status(queue))) {
    operation->status =
        iree_status_clone(iree_hal_amd_xdna_queue_failure_status(queue));
  }
  if (iree_status_is_ok(operation->status) &&
      operation->kind == IREE_HAL_AMD_XDNA_OPERATION_HOST_CALL &&
      iree_any_bit_set(operation->host_call.flags,
                       IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING)) {
    // Publish after dependencies resolve and before invoking the callback.
    // The operation still retains callback state until invocation returns.
    operation->status =
        iree_hal_amd_xdna_operation_signal(operation, operation->device);
    if (iree_status_is_ok(operation->status)) {
      iree_hal_semaphore_list_release(operation->signals);
      operation->signals = iree_hal_semaphore_list_empty();
    }
  }
  if (!iree_status_is_ok(operation->status) ||
      operation->kind == IREE_HAL_AMD_XDNA_OPERATION_BARRIER ||
      (operation->kind == IREE_HAL_AMD_XDNA_OPERATION_EXECUTE &&
       !operation->execution->cursor)) {
    if (operation->profile && iree_status_is_ok(operation->status)) {
      iree_hal_amd_xdna_operation_profile_ready(operation);
    }
    iree_hal_amd_xdna_operation_complete(operation);
    return;
  }
  if (operation->kind == IREE_HAL_AMD_XDNA_OPERATION_TRANSFER ||
      operation->kind == IREE_HAL_AMD_XDNA_OPERATION_HOST_CALL ||
      operation->kind == IREE_HAL_AMD_XDNA_OPERATION_FILE ||
      (operation->kind == IREE_HAL_AMD_XDNA_OPERATION_EXECUTE &&
       !iree_hal_amd_xdna_operation_has_dispatch(operation))) {
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

// Commits one direct or deferred publication result under |state_mutex|.
// |service| and |item| are NULL when the caller performed native publication.
static void iree_hal_amd_xdna_queue_commit_publication(
    iree_hal_amd_xdna_queue_t* queue,
    iree_hal_amd_xdna_queue_service_t* service,
    iree_hal_amd_xdna_queue_service_item_t* item,
    iree_hal_amd_xdna_operation_t* operation, iree_status_t status) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, operation->native.publication.count);
  operation->status =
      iree_status_join(operation->status, operation->native.publication.status);
  operation->native.publication.status = iree_ok_status();
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_queue_fail_pending(queue, iree_status_clone(status));
    operation->status = iree_status_join(operation->status, status);
  }
  const uint32_t accepted_count = operation->native.publication.count;
  const bool retry = operation->native.publication.result ==
                     IREE_HAL_AMD_XDNA_PUBLICATION_RESULT_RETRY;
  if (retry && !operation->native.publication.had_pending &&
      iree_status_is_ok(iree_hal_amd_xdna_queue_failure_status(queue))) {
    iree_hal_amd_xdna_queue_fail_pending(
        queue,
        iree_make_status(
            IREE_STATUS_INTERNAL,
            "XDNA native publication reported BUSY without pending progress"));
  }
  if (iree_status_is_ok(operation->status)) {
    operation->status =
        iree_status_clone(iree_hal_amd_xdna_queue_failure_status(queue));
  }
  if (service) {
    iree_hal_amd_xdna_queue_service_acknowledge(service, item);
  }
  iree_atomic_store(&queue->publisher_operation, 0, iree_memory_order_release);

  iree_hal_amd_xdna_operation_t* targeted_waiters = NULL;
  bool complete = false;
  if (operation->native.phase == IREE_HAL_AMD_XDNA_NATIVE_PHASE_ABANDONED ||
      (accepted_count &&
       !iree_status_is_ok(iree_hal_amd_xdna_queue_failure_status(queue)))) {
    // A concurrent observer failure can precede publication acknowledgement.
    // The invisible accepted suffix is still native-reachable and must leak.
    if (operation->native.phase != IREE_HAL_AMD_XDNA_NATIVE_PHASE_ABANDONED) {
      operation->native.phase = IREE_HAL_AMD_XDNA_NATIVE_PHASE_ABANDONED;
      targeted_waiters =
          iree_hal_amd_xdna_operation_unregister_producers(operation);
      iree_hal_semaphore_list_fail(
          operation->signals,
          iree_status_clone(iree_hal_amd_xdna_queue_failure_status(queue)));
    }
  } else {
    if (accepted_count) {
      iree_hal_amd_xdna_frontier_state_merge(&operation->frontier,
                                             &queue->accepted_frontier);
      for (uint32_t i = 0; i < accepted_count; ++i) {
        const uint32_t index =
            (operation->native.publication.start + i) % queue->pending.capacity;
        queue->pending.entries[index].epoch = ++queue->epoch;
      }
      queue->pending.count += accepted_count;
      operation->native.pending_count += accepted_count;
      if (queue->tracker) {
        iree_hal_amd_xdna_frontier_state_advance(&operation->frontier,
                                                 queue->axis, queue->epoch);
      } else {
        operation->frontier.exact = false;
      }
      iree_hal_amd_xdna_frontier_state_copy(&operation->frontier,
                                            &queue->accepted_frontier);
      iree_atomic_store(&queue->pending.visible_count,
                        (int32_t)queue->pending.count,
                        iree_memory_order_release);
    }
    const bool has_suffix =
        operation->kind == IREE_HAL_AMD_XDNA_OPERATION_EXECUTE
            ? operation->execution->cursor != NULL
            : accepted_count == 0;
    if (!iree_status_is_ok(operation->status) || !has_suffix) {
      operation->native.phase = IREE_HAL_AMD_XDNA_NATIVE_PHASE_COMPLETE;
      targeted_waiters =
          iree_hal_amd_xdna_operation_unregister_producers(operation);
      if (iree_status_is_ok(operation->status) && accepted_count) {
        // Only the final native tail proves the entire recording's completion.
        iree_hal_amd_xdna_queue_publish_signals(queue, operation, queue->epoch);
      }
      complete = !operation->native.pending_count;
    } else if (!iree_hal_amd_xdna_operation_has_dispatch(operation)) {
      operation->native.phase = IREE_HAL_AMD_XDNA_NATIVE_PHASE_DRAINING;
      if (!operation->native.pending_count) {
        iree_hal_amd_xdna_queue_service_enqueue(&queue->host_worker,
                                                &operation->service_item);
      }
    } else {
      iree_hal_amd_xdna_operation_register_producers(operation);
      iree_hal_amd_xdna_queue_prepend_ready(queue, operation);
      if (retry && queue->pending.count) {
        iree_atomic_store(&queue->native_publication_blocked, 1,
                          iree_memory_order_release);
      }
    }
    if (queue->pending.count && !queue->notification_pending) {
      iree_status_t arm_status = iree_hal_amd_xdna_queue_arm(queue);
      if (!iree_status_is_ok(arm_status)) {
        iree_hal_amd_xdna_queue_fail_pending(queue, arm_status);
      }
    }
  }
  iree_hal_amd_xdna_queue_retry_acceptance_waiters(queue);
  iree_hal_amd_xdna_queue_continue_or_release_publisher(queue);
  if (complete) {
    iree_hal_amd_xdna_operation_complete(operation);
  }
  iree_hal_amd_xdna_operation_wake_acceptance_waiters(targeted_waiters);
  IREE_TRACE_ZONE_END(z0);
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

static void iree_hal_amd_xdna_operation_resume(void* user_data,
                                               iree_status_t status) {
  iree_hal_amd_xdna_operation_t* operation = user_data;
  // Exactly one pool wait or file request owns the continuation at a time.
  // Its terminal callback transfers status before publishing the handoff.
  iree_atomic_store(&operation->continuation_status, (intptr_t)status,
                    iree_memory_order_release);
  status = iree_async_proactor_submit_one(operation->queue->proactor,
                                          &operation->continuation);
  if (!iree_status_is_ok(status)) {
    iree_status_t continuation_status = (iree_status_t)iree_atomic_exchange(
        &operation->continuation_status, 0, iree_memory_order_acquire);
    status = iree_status_join(continuation_status, status);
    iree_hal_amd_xdna_queue_report(operation->queue, iree_status_code(status),
                                   iree_status_message(status));
    iree_hal_semaphore_list_fail(operation->signals, status);
  }
}

static void iree_hal_amd_xdna_queue_file_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  (void)base_operation;
  (void)flags;
  iree_hal_amd_xdna_operation_t* operation = user_data;
  iree_status_t io_status = (iree_status_t)iree_atomic_exchange(
      &operation->continuation_status, 0, iree_memory_order_acquire);
  operation->status = iree_status_join(status, io_status);
  if (iree_status_is_ok(operation->status) &&
      !iree_status_is_ok(
          iree_hal_amd_xdna_queue_failure_status(operation->queue))) {
    operation->status = iree_status_clone(
        iree_hal_amd_xdna_queue_failure_status(operation->queue));
  }
  operation->file.io_completed = true;
  const iree_hal_amd_fabric_file_transfer_t* transfer =
      operation->file.transfer;
  if (iree_status_is_ok(operation->status) &&
      (transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ ||
       transfer->async.staging.data ||
       transfer->async.bytes_transferred < transfer->length)) {
    iree_hal_amd_xdna_queue_service_enqueue(&operation->queue->host_worker,
                                            &operation->service_item);
  } else {
    iree_hal_amd_xdna_operation_complete(operation);
  }
}

static void iree_hal_amd_xdna_host_worker_complete(
    void* user_data, iree_hal_amd_xdna_queue_service_t* service,
    iree_hal_amd_xdna_queue_service_item_t* item, iree_status_t status) {
  iree_hal_amd_xdna_queue_t* queue = (iree_hal_amd_xdna_queue_t*)user_data;
  iree_hal_amd_xdna_operation_t* operation =
      iree_hal_amd_xdna_operation_from_service_item(item);
  if (operation->kind == IREE_HAL_AMD_XDNA_OPERATION_EXECUTE) {
    iree_slim_mutex_lock(&queue->state_mutex);
    operation->native.phase = IREE_HAL_AMD_XDNA_NATIVE_PHASE_ISSUING;
    status = iree_status_join(operation->status, status);
    operation->status = iree_ok_status();
    iree_hal_amd_xdna_queue_service_acknowledge(service, item);
    iree_hal_amd_xdna_queue_enqueue_ready(operation, status);
    iree_slim_mutex_unlock(&queue->state_mutex);
    return;
  }
  operation->status = iree_status_join(operation->status, status);
  if (iree_status_is_ok(operation->status) &&
      !iree_status_is_ok(iree_hal_amd_xdna_queue_failure_status(queue))) {
    operation->status =
        iree_status_clone(iree_hal_amd_xdna_queue_failure_status(queue));
  }
  iree_hal_amd_xdna_queue_service_acknowledge(service, item);

  if (operation->kind == IREE_HAL_AMD_XDNA_OPERATION_FILE &&
      iree_status_is_ok(operation->status) &&
      operation->file.transfer->async.file && !operation->file.io_completed) {
    iree_async_operation_initialize(
        &operation->continuation, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_amd_xdna_queue_file_complete,
        operation);
    operation->status = iree_hal_amd_fabric_file_transfer_submit(
        operation->file.transfer,
        (iree_hal_amd_fabric_file_transfer_callback_t){
            .fn = iree_hal_amd_xdna_operation_resume,
            .user_data = operation,
        });
    if (iree_status_is_ok(operation->status)) {
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
  iree_hal_amd_xdna_retired_batch_t retired = {0};
  bool should_pump = false;
  if (iree_status_is_ok(status) && iree_async_poll_has_error(events)) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "XDNA native completion source reported poll events 0x%08X",
        (unsigned)events);
  }
  int32_t direct_wait_flags =
      iree_atomic_load(&queue->direct_wait_flags, iree_memory_order_acquire);
  while (iree_status_is_ok(status) &&
         iree_any_bit_set(direct_wait_flags,
                          IREE_HAL_AMD_XDNA_DIRECT_WAIT_FLAG_ACTIVE)) {
    const int32_t desired_flags =
        direct_wait_flags | IREE_HAL_AMD_XDNA_DIRECT_WAIT_FLAG_EVENT_OBSERVED;
    if (iree_atomic_compare_exchange_weak(
            &queue->direct_wait_flags, &direct_wait_flags, desired_flags,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      // The caller owns checked retirement. Record that this one-shot hint was
      // consumed and return without contending on its publication locks.
      return;
    }
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&queue->completion_drain_mutex);
  iree_slim_mutex_lock(&queue->state_mutex);
  queue->notification_pending = false;
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_queue_fail_pending(queue, status);
  } else if (iree_any_bit_set(iree_atomic_load(&queue->direct_wait_flags,
                                               iree_memory_order_acquire),
                              IREE_HAL_AMD_XDNA_DIRECT_WAIT_FLAG_ACTIVE)) {
    // A waiter claimed retirement while this callback acquired the locks. The
    // consumed notification is already reflected in |notification_pending|;
    // the waiter will rearm any remaining prefix after checked retirement.
  } else if (iree_status_is_ok(iree_hal_amd_xdna_queue_failure_status(queue)) &&
             queue->pending.count) {
    amdf_kernel_queue_status_t checked = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
        .structure_size = sizeof(checked),
    };
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        queue->context->api->kernel_queue_refresh_status(queue->handle,
                                                         &checked),
        "kernel_queue_refresh_status");
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_xdna_queue_fail_pending(queue, status);
    } else {
      should_pump = iree_hal_amd_xdna_queue_apply_checked_status(
          queue, &checked, &retired);
    }
  }
  iree_slim_mutex_unlock(&queue->state_mutex);
  iree_hal_amd_xdna_queue_complete_retired_batch(&retired);
  iree_slim_mutex_unlock(&queue->completion_drain_mutex);
  iree_hal_amd_xdna_queue_pump_after_retirement(queue, should_pump);
  IREE_TRACE_ZONE_END(z0);
}

static uint64_t iree_hal_amd_xdna_queue_wait_timeout_ns(
    iree_timeout_t timeout) {
  if (iree_timeout_is_infinite(timeout)) {
    return AMDF_TIMEOUT_INFINITE;
  }
  const iree_duration_t duration_ns = iree_timeout_as_duration_ns(timeout);
  return duration_ns > 0 ? (uint64_t)duration_ns : 0;
}

static uint64_t iree_hal_amd_xdna_queue_wait_poll_duration_ns(
    uint64_t timeout_ns, iree_async_wait_flags_t flags) {
  if (iree_any_bit_set(flags, IREE_ASYNC_WAIT_FLAG_ACTIVE)) {
    return timeout_ns;
  }
  if (iree_any_bit_set(flags, IREE_ASYNC_WAIT_FLAG_YIELD)) {
    const uint64_t yield_duration_ns = 500;
    return timeout_ns == AMDF_TIMEOUT_INFINITE
               ? yield_duration_ns
               : iree_min(timeout_ns, yield_duration_ns);
  }
  return 0;
}

bool iree_hal_amd_xdna_queue_try_wait_epoch(iree_hal_queue_t* base,
                                            iree_async_axis_t producer_axis,
                                            uint64_t producer_epoch,
                                            iree_timeout_t timeout,
                                            iree_async_wait_flags_t flags,
                                            iree_status_t* out_status) {
  iree_hal_amd_xdna_queue_t* queue = (iree_hal_amd_xdna_queue_t*)base;

  // The exact producer epoch maps to a pending ring position in O(1) because
  // native epochs are assigned contiguously in physical acceptance order.
  if (!iree_slim_mutex_try_lock(&queue->state_mutex)) {
    return false;
  }
  uint64_t submission = 0;
  if (iree_status_is_ok(iree_hal_amd_xdna_queue_failure_status(queue)) &&
      queue->pending.count && queue->axis == producer_axis) {
    const uint64_t head_epoch =
        queue->pending.entries[queue->pending.head].epoch;
    if (producer_epoch >= head_epoch) {
      const uint64_t offset = producer_epoch - head_epoch;
      if (offset < queue->pending.count) {
        const uint32_t index =
            (queue->pending.head + (uint32_t)offset) % queue->pending.capacity;
        const iree_hal_amd_xdna_pending_t* pending =
            &queue->pending.entries[index];
        int32_t expected_flags = IREE_HAL_AMD_XDNA_DIRECT_WAIT_FLAG_NONE;
        if (pending->epoch == producer_epoch &&
            iree_atomic_compare_exchange_strong(
                &queue->direct_wait_flags, &expected_flags,
                IREE_HAL_AMD_XDNA_DIRECT_WAIT_FLAG_ACTIVE,
                iree_memory_order_acq_rel, iree_memory_order_acquire)) {
          submission = pending->submission;
        }
      }
    }
  }
  iree_slim_mutex_unlock(&queue->state_mutex);
  if (!submission) {
    return false;
  }

  const uint64_t timeout_ns = iree_hal_amd_xdna_queue_wait_timeout_ns(timeout);
  const uint64_t poll_duration_ns =
      iree_hal_amd_xdna_queue_wait_poll_duration_ns(timeout_ns, flags);
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      queue->context->api->kernel_queue_wait(queue->handle, submission,
                                             timeout_ns, poll_duration_ns),
      "kernel_queue_wait");

  // The wait performs checked retirement but returns no snapshot. Query the
  // established state without another native operation, then serialize HAL
  // ring detachment and user-visible completion publication.
  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_amd_xdna_wait_retire");
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, submission);
  amdf_kernel_queue_status_t checked = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
      .structure_size = sizeof(checked),
  };
  iree_status_t query_status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      queue->context->api->kernel_queue_query_status(queue->handle, &checked),
      "kernel_queue_query_status");
  iree_hal_amd_xdna_retired_batch_t retired = {0};
  bool should_pump = false;
  iree_slim_mutex_lock(&queue->completion_drain_mutex);
  iree_slim_mutex_lock(&queue->state_mutex);
  const iree_hal_amd_xdna_direct_wait_flags_t direct_wait_flags =
      (iree_hal_amd_xdna_direct_wait_flags_t)iree_atomic_exchange(
          &queue->direct_wait_flags, IREE_HAL_AMD_XDNA_DIRECT_WAIT_FLAG_NONE,
          iree_memory_order_acq_rel);
  if (iree_any_bit_set(direct_wait_flags,
                       IREE_HAL_AMD_XDNA_DIRECT_WAIT_FLAG_EVENT_OBSERVED)) {
    queue->notification_pending = false;
  }
  if (!iree_status_is_ok(query_status)) {
    iree_hal_amd_xdna_queue_fail_pending(queue,
                                         iree_status_clone(query_status));
    status = iree_status_join(status, query_status);
  } else if (iree_status_is_ok(status) &&
             checked.retired_submission < submission) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "libamdf reported successful wait for submission %" PRIu64
        " with checked retirement only through %" PRIu64,
        submission, checked.retired_submission);
    iree_hal_amd_xdna_queue_fail_pending(queue, iree_status_clone(status));
  } else {
    should_pump =
        iree_hal_amd_xdna_queue_apply_checked_status(queue, &checked, &retired);
    if (iree_status_is_ok(status) &&
        !iree_status_is_ok(iree_hal_amd_xdna_queue_failure_status(queue))) {
      status = iree_status_clone(iree_hal_amd_xdna_queue_failure_status(queue));
    }
  }
  iree_slim_mutex_unlock(&queue->state_mutex);
  iree_hal_amd_xdna_queue_complete_retired_batch(&retired);
  iree_slim_mutex_unlock(&queue->completion_drain_mutex);
  iree_hal_amd_xdna_queue_pump_after_retirement(queue, should_pump);
  *out_status = status;
  IREE_TRACE_ZONE_END(z0);
  return true;
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

// Attempts caller-side native publication without waiting for ownership of the
// publisher. Native submission itself may still wait for device wake or native
// credits because libamdf does not expose a nonblocking try-submit contract.
// Returns true once ownership of |operation| has been consumed.
static bool iree_hal_amd_xdna_operation_try_direct_publish(
    iree_hal_amd_xdna_operation_t* operation,
    iree_status_t* out_submission_status) {
  *out_submission_status = iree_ok_status();
  if (!iree_hal_amd_xdna_operation_has_dispatch(operation)) {
    return false;
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_amd_xdna_queue_t* queue = operation->queue;
  if (!iree_slim_mutex_try_lock(&queue->state_mutex)) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "queue state busy");
    IREE_TRACE_ZONE_END(z0);
    return false;
  }
  if (!iree_atomic_load(&queue->native_admission_open,
                        iree_memory_order_acquire)) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "native admission closed");
    iree_slim_mutex_unlock(&queue->state_mutex);
    IREE_TRACE_ZONE_END(z0);
    return false;
  }
  if (iree_atomic_load(&queue->native_publication_blocked,
                       iree_memory_order_acquire)) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "native credits busy");
    iree_slim_mutex_unlock(&queue->state_mutex);
    IREE_TRACE_ZONE_END(z0);
    return false;
  }
  if (!iree_hal_amd_xdna_queue_try_claim_publisher(queue)) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "publisher busy");
    iree_slim_mutex_unlock(&queue->state_mutex);
    IREE_TRACE_ZONE_END(z0);
    return false;
  }

  const int32_t pending_count = iree_atomic_load(&queue->pending.visible_count,
                                                 iree_memory_order_acquire);
  if (!iree_atomic_load(&queue->native_admission_open,
                        iree_memory_order_acquire) ||
      iree_atomic_load(&queue->native_publication_blocked,
                       iree_memory_order_acquire) ||
      pending_count >= (int32_t)queue->pending.capacity) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "native window unavailable");
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, pending_count);
    iree_hal_amd_xdna_queue_release_publisher(queue);
    iree_slim_mutex_unlock(&queue->state_mutex);
    IREE_TRACE_ZONE_END(z0);
    return false;
  }

  iree_hal_amd_xdna_queue_reserve_publication(queue, operation);
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
    if (resolution != IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_READY) {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "wait frontier unproven");
      iree_hal_amd_xdna_queue_release_publisher(queue);
      iree_slim_mutex_unlock(&queue->state_mutex);
      IREE_TRACE_ZONE_END(z0);
      return false;
    }
    iree_atomic_store(&queue->publisher_operation, (intptr_t)operation,
                      iree_memory_order_release);
  }
  iree_slim_mutex_unlock(&queue->state_mutex);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, "caller direct");
  IREE_TRACE_ZONE_END(z0);

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
  iree_status_t status =
      operation->queue->profile
          ? iree_hal_amd_xdna_operation_profile_capture(operation)
          : iree_ok_status();
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_operation_discard(operation);
    return status;
  }
  if (operation->kind == IREE_HAL_AMD_XDNA_OPERATION_ALLOCA ||
      operation->kind == IREE_HAL_AMD_XDNA_OPERATION_DEALLOCA) {
    iree_hal_amd_xdna_operation_admit_memory(operation, iree_ok_status());
    return iree_ok_status();
  }
  if (iree_hal_amd_xdna_operation_try_direct_publish(operation, &status)) {
    return status;
  }
  if (operation->profile) {
    operation->profile->event.flags |=
        IREE_HAL_PROFILE_QUEUE_EVENT_FLAG_SOFTWARE_DEFERRED;
    operation->profile->event.dependency_strategy =
        IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER;
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
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_operation_allocate(
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
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_operation_allocate(
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
  iree_status_t profile_status = iree_hal_amd_xdna_queue_profiling_end(base);
  if (!iree_status_is_ok(profile_status)) {
    iree_hal_amd_xdna_queue_report_status(queue, profile_status);
  }
  iree_status_free(iree_hal_amd_xdna_queue_failure_status(queue));
  iree_async_event_release(queue->event);
  iree_allocator_free(queue->host_allocator, queue->pending.entries);
  iree_hal_amd_xdna_queue_storage_deinitialize(&queue->storage);
  iree_slim_mutex_deinitialize(&queue->completion_drain_mutex);
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
  iree_slim_mutex_initialize(&queue->completion_drain_mutex);
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
  iree_atomic_store(&queue->direct_wait_flags,
                    IREE_HAL_AMD_XDNA_DIRECT_WAIT_FLAG_NONE,
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
  if (iree_status_is_unimplemented(status)) {
    status = iree_status_annotate(
        status,
        IREE_SV("XDNA requires native instruction submission and completion "
                "support; update the NPU driver and matching firmware. See "
                "https://github.com/ROCm/hrx-system/blob/main/libamdf/docs/"
                "xdna.md#native-requirements"));
  }
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
    queue->maximum_command_count =
        iree_min(info.maximum_command_count, queue->pending.capacity);
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
    queue->native_event =
        iree_hal_amd_fabric_event_native_handle(&queue->event->native);
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

uint32_t iree_hal_amd_xdna_queue_maximum_command_count(iree_hal_queue_t* base) {
  return ((iree_hal_amd_xdna_queue_t*)base)->maximum_command_count;
}

void iree_hal_amd_xdna_queue_trim(iree_hal_queue_t* base) {
  iree_hal_amd_xdna_queue_t* queue = (iree_hal_amd_xdna_queue_t*)base;
  iree_hal_amd_xdna_queue_storage_trim(&queue->storage);
}

static iree_status_t iree_hal_amd_xdna_queue_flush(iree_hal_queue_t* queue) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_queue_execute(
    iree_hal_queue_t* queue, const iree_hal_semaphore_list_t waits,
    const iree_hal_semaphore_list_t signals,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_queue_execute_flags_t flags) {
  iree_hal_amd_xdna_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_operation_allocate(
      (iree_hal_amd_xdna_queue_t*)queue, waits, signals,
      IREE_HAL_AMD_XDNA_OPERATION_EXECUTE, &operation));
  iree_status_t status = iree_hal_amd_xdna_command_buffer_acquire_execution(
      command_buffer, binding_table, flags, &operation->execution);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_operation_submit(operation);
  } else {
    iree_hal_amd_xdna_operation_discard(operation);
  }
  return status;
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
