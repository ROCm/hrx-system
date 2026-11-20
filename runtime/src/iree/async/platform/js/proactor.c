// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/js/proactor.h"

#include <string.h>

#include "iree/async/operations/scheduling.h"
#include "iree/async/platform/js/imports.h"
#include "iree/async/platform/js/token_table.h"
#include "iree/async/util/continuation.h"
#include "iree/async/util/operation_completion.h"
#include "iree/async/util/sequence_emulation.h"

// Forward-declare the vtable (defined at bottom of file).
static const iree_async_proactor_vtable_t iree_async_proactor_js_vtable;

//===----------------------------------------------------------------------===//
// Poll-owned pending queue
//===----------------------------------------------------------------------===//

// Pushes an operation onto the pending queue tail.
static void iree_async_proactor_js_pending_enqueue(
    iree_async_proactor_js_t* proactor, iree_async_operation_t* operation) {
  operation->next = NULL;
  if (proactor->pending_tail) {
    proactor->pending_tail->next = operation;
  } else {
    proactor->pending_head = operation;
  }
  proactor->pending_tail = operation;
}

// Pops an operation from the pending queue head. Returns NULL if empty.
static iree_async_operation_t* iree_async_proactor_js_pending_dequeue(
    iree_async_proactor_js_t* proactor) {
  iree_async_operation_t* operation = proactor->pending_head;
  if (operation) {
    proactor->pending_head = operation->next;
    if (!proactor->pending_head) {
      proactor->pending_tail = NULL;
    }
    operation->next = NULL;
  }
  return operation;
}

//===----------------------------------------------------------------------===//
// Submit
//===----------------------------------------------------------------------===//

// Validates one public operation before the batch reserves or mutates any
// backend state. Commit relies on these checks and cannot reject an operation.
static iree_status_t iree_async_proactor_js_validate_operation(
    const iree_async_operation_t* operation) {
  const iree_async_operation_flags_t known_operation_flags =
      IREE_ASYNC_OPERATION_FLAG_MULTISHOT | IREE_ASYNC_OPERATION_FLAG_LINKED |
      IREE_ASYNC_OPERATION_FLAG_CANCELLATION_IS_SUCCESS;
  if (operation->flags & ~known_operation_flags) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "operation has unknown flags 0x%08X",
                            operation->flags & ~known_operation_flags);
  }
  if (!operation->completion_fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "operation requires a completion callback");
  }
  if (iree_any_bit_set(operation->flags, IREE_ASYNC_OPERATION_FLAG_MULTISHOT)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "MULTISHOT is unsupported for JS operation type %d",
                            (int)operation->type);
  }

  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_NOP:
    case IREE_ASYNC_OPERATION_TYPE_TIMER:
      return iree_ok_status();
    case IREE_ASYNC_OPERATION_TYPE_SEQUENCE:
      return iree_async_sequence_validate(
          (const iree_async_sequence_operation_t*)operation);
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "JS proactor does not support operation type %d",
                              (int)operation->type);
  }
}

// Reserves the token slot needed by a future timer. Timer routing is frozen
// against one batch timestamp so commit cannot cross the expired/future
// boundary and introduce a new failure after earlier operations are visible.
static iree_status_t iree_async_proactor_js_reserve_operation(
    iree_async_proactor_js_t* proactor, iree_async_operation_t* operation,
    iree_time_t now_ns) {
  if (operation->type != IREE_ASYNC_OPERATION_TYPE_TIMER) {
    return iree_ok_status();
  }

  iree_async_timer_operation_t* timer =
      (iree_async_timer_operation_t*)operation;
  if (timer->deadline_ns <= now_ns) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_async_js_token_table_acquire(
      &proactor->token_table, operation, &timer->platform.js.token));
  timer->platform.js.is_token_active = true;
  return iree_ok_status();
}

static void iree_async_proactor_js_rollback_reservations(
    iree_async_proactor_js_t* proactor,
    iree_async_operation_list_t operations) {
  iree_async_continuation_chain_iterator_t iterator =
      iree_async_continuation_chain_iterator_make(operations);
  iree_async_operation_t* operation = NULL;
  while ((operation = iree_async_continuation_chain_iterator_next(&iterator)) !=
         NULL) {
    if (operation->type != IREE_ASYNC_OPERATION_TYPE_TIMER) {
      continue;
    }
    iree_async_timer_operation_t* timer =
        (iree_async_timer_operation_t*)operation;
    if (timer->platform.js.is_token_active) {
      iree_async_js_token_table_release(&proactor->token_table,
                                        timer->platform.js.token);
      timer->platform.js.is_token_active = false;
    }
  }
}

// Commits one fully validated and reserved operation. No branch can reject the
// operation after this point; sequence startup failures become completions.
static bool iree_async_proactor_js_commit_operation(
    iree_async_proactor_js_t* proactor, iree_async_operation_t* operation) {
  if (operation->type == IREE_ASYNC_OPERATION_TYPE_SEQUENCE) {
    iree_async_sequence_prepare_for_submission(
        (iree_async_sequence_operation_t*)operation);
  } else {
    iree_async_operation_clear_internal_flags(operation);
  }
  IREE_TRACE(operation->submit_time_ns = iree_time_now();)

  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_NOP:
    case IREE_ASYNC_OPERATION_TYPE_SEQUENCE:
      iree_async_proactor_js_pending_enqueue(proactor, operation);
      return true;
    case IREE_ASYNC_OPERATION_TYPE_TIMER: {
      iree_async_timer_operation_t* timer =
          (iree_async_timer_operation_t*)operation;
      if (timer->platform.js.is_token_active) {
        iree_async_js_import_timer_start(timer->platform.js.token,
                                         timer->deadline_ns);
        return false;
      }
      iree_async_proactor_js_pending_enqueue(proactor, operation);
      return true;
    }
    default:
      IREE_ASSERT_UNREACHABLE("operation type must be validated");
      IREE_BUILTIN_UNREACHABLE();
  }
}

// Submits a list whose LINKED topology has already been established. Public
// batches arrive here after continuation preparation; continuation dispatch
// supplies a single pre-linked chain head.
static iree_status_t iree_async_proactor_js_submit_prepared(
    iree_async_proactor_js_t* proactor,
    iree_async_operation_list_t operations) {
  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_async_proactor_js_validate_operation(operations.values[i]));
  }

  // Clear reservation scratch for active heads before any acquisition. Linked
  // successors remain owned by their predecessors and are admitted later.
  iree_async_continuation_chain_iterator_t clear_iterator =
      iree_async_continuation_chain_iterator_make(operations);
  iree_async_operation_t* operation = NULL;
  while ((operation = iree_async_continuation_chain_iterator_next(
              &clear_iterator)) != NULL) {
    operation->next = NULL;
    if (operation->type == IREE_ASYNC_OPERATION_TYPE_TIMER) {
      iree_async_timer_operation_t* timer =
          (iree_async_timer_operation_t*)operation;
      timer->platform.js.token = 0;
      timer->platform.js.is_token_active = false;
    }
  }

  iree_time_t now_ns = iree_time_now();
  iree_async_continuation_chain_iterator_t reserve_iterator =
      iree_async_continuation_chain_iterator_make(operations);
  while ((operation = iree_async_continuation_chain_iterator_next(
              &reserve_iterator)) != NULL) {
    iree_status_t status =
        iree_async_proactor_js_reserve_operation(proactor, operation, now_ns);
    if (!iree_status_is_ok(status)) {
      iree_async_proactor_js_rollback_reservations(proactor, operations);
      return status;
    }
  }

  // The complete list is now accepted. Acquire linked successors before any
  // active timer or software completion can make the chain observable.
  iree_async_operation_list_acquire_resources(operations);

  bool requires_drain = false;
  iree_async_continuation_chain_iterator_t commit_iterator =
      iree_async_continuation_chain_iterator_make(operations);
  while ((operation = iree_async_continuation_chain_iterator_next(
              &commit_iterator)) != NULL) {
    requires_drain |=
        iree_async_proactor_js_commit_operation(proactor, operation);
  }
  if (requires_drain) {
    iree_async_js_import_schedule_drain();
  }
  return iree_ok_status();
}

static iree_status_t iree_async_proactor_js_submit_continuation(
    void* user_data, iree_async_operation_t* chain_head) {
  iree_async_operation_t* operations[] = {chain_head};
  return iree_async_proactor_js_submit_prepared(
      (iree_async_proactor_js_t*)user_data,
      iree_async_operation_list_make(operations, IREE_ARRAYSIZE(operations)));
}

// Dispatches a final completion and any continuation callbacks from poll().
// Returns the exact number of user callbacks invoked.
static iree_host_size_t iree_async_proactor_js_complete(
    iree_async_proactor_js_t* proactor, iree_async_operation_t* operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_continuation_t continuation = {0};
  if (!iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE)) {
    iree_async_operation_t* chain_head =
        iree_async_continuation_take(operation);
    continuation = iree_async_continuation_begin(
        iree_async_proactor_js_submit_continuation, proactor, chain_head,
        iree_status_code(status));
  }
  iree_host_size_t completed_count =
      iree_async_operation_complete(operation, status, flags);
  completed_count += iree_async_continuation_finish(&continuation);
  IREE_TRACE_ZONE_END(z0);
  return completed_count;
}

static iree_status_t iree_async_proactor_js_submit(
    iree_async_proactor_t* proactor, iree_async_operation_list_t operations,
    iree_host_size_t* out_retry_batch_size) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_proactor_js_t* js_proactor = iree_async_proactor_js_cast(proactor);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_continuation_prepare_batch(operations));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_proactor_js_submit_prepared(js_proactor, operations));
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_async_proactor_js_submit_external(
    iree_async_proactor_t* proactor, iree_async_operation_t* operation,
    uint32_t* out_token) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(operation);
  IREE_ASSERT_ARGUMENT(out_token);
  *out_token = UINT32_MAX;

  iree_async_proactor_js_t* js_proactor = iree_async_proactor_js_cast(proactor);
  iree_async_operation_clear_internal_flags(operation);
  IREE_TRACE(operation->submit_time_ns = iree_time_now();)

  iree_status_t status = iree_async_js_token_table_acquire(
      &js_proactor->token_table, operation, out_token);
  if (iree_status_is_ok(status)) {
    iree_async_operation_acquire_resources(operation);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Poll
//===----------------------------------------------------------------------===//

// Drains completions from the JS ring and dispatches callbacks.
static iree_host_size_t iree_async_proactor_js_drain_ring(
    iree_async_proactor_js_t* proactor) {
  uint32_t entry_count = iree_async_js_import_ring_drain(
      proactor->completion_buffer, proactor->completion_buffer_capacity);
  iree_host_size_t completed_count = 0;

  for (uint32_t i = 0; i < entry_count; ++i) {
    iree_async_js_completion_entry_t* entry = &proactor->completion_buffer[i];
    iree_async_operation_t* operation =
        iree_async_js_token_table_lookup(&proactor->token_table, entry->token);
    if (!operation) {
      // Stale completion for a token that was already released (e.g., timer
      // fired after cancel). This is expected and harmless.
      continue;
    }

    // Release the token table slot before dispatching the callback, since the
    // callback may submit new operations that reuse this slot.
    iree_async_js_token_table_release(&proactor->token_table, entry->token);
    if (operation->type == IREE_ASYNC_OPERATION_TYPE_TIMER) {
      ((iree_async_timer_operation_t*)operation)->platform.js.is_token_active =
          false;
    }

    // Check if the operation was cancelled while in flight.
    iree_async_operation_internal_flags_t internal_flags =
        iree_atomic_load(&operation->internal_flags, iree_memory_order_relaxed);
    if (internal_flags & IREE_ASYNC_JS_OPERATION_INTERNAL_FLAG_CANCELLED) {
      completed_count += iree_async_proactor_js_complete(
          proactor, operation, iree_status_from_code(IREE_STATUS_CANCELLED),
          IREE_ASYNC_COMPLETION_FLAG_NONE);
    } else {
      iree_status_t status =
          entry->status_code == 0
              ? iree_ok_status()
              : iree_status_from_code((iree_status_code_t)entry->status_code);
      completed_count += iree_async_proactor_js_complete(
          proactor, operation, status, IREE_ASYNC_COMPLETION_FLAG_NONE);
    }
  }

  return completed_count;
}

// Drains immediate completions and starts accepted sequences on the poll
// owner. Sequence startup may submit child operations onto this same queue.
static iree_host_size_t iree_async_proactor_js_drain_pending(
    iree_async_proactor_js_t* proactor) {
  iree_host_size_t count = 0;
  iree_async_operation_t* operation;
  while ((operation = iree_async_proactor_js_pending_dequeue(proactor)) !=
         NULL) {
    if (operation->type == IREE_ASYNC_OPERATION_TYPE_SEQUENCE) {
      iree_async_sequence_operation_t* sequence =
          (iree_async_sequence_operation_t*)operation;
      bool is_cancelled =
          iree_any_bit_set(iree_async_operation_load_internal_flags(operation),
                           IREE_ASYNC_SEQUENCE_INTERNAL_CANCEL_REQUESTED);
      if (is_cancelled || sequence->step_count == 0) {
        iree_async_sequence_prepare_for_completion(sequence);
        count += iree_async_proactor_js_complete(
            proactor, operation,
            is_cancelled ? iree_status_from_code(IREE_STATUS_CANCELLED)
                         : iree_ok_status(),
            IREE_ASYNC_COMPLETION_FLAG_NONE);
        continue;
      }

      iree_status_t status =
          sequence->step_fn
              ? iree_async_sequence_emulation_begin(
                    &proactor->sequence_emulator, sequence)
              : iree_async_sequence_submit_as_linked(&proactor->base, sequence);
      if (!iree_status_is_ok(status)) {
        iree_async_sequence_prepare_for_completion(sequence);
        count += iree_async_proactor_js_complete(
            proactor, operation, status, IREE_ASYNC_COMPLETION_FLAG_NONE);
      }
      continue;
    }

    bool is_cancelled =
        iree_any_bit_set(iree_async_operation_load_internal_flags(operation),
                         IREE_ASYNC_JS_OPERATION_INTERNAL_FLAG_CANCELLED);
    if (is_cancelled) {
      count += iree_async_proactor_js_complete(
          proactor, operation, iree_status_from_code(IREE_STATUS_CANCELLED),
          IREE_ASYNC_COMPLETION_FLAG_NONE);
    } else {
      count +=
          iree_async_proactor_js_complete(proactor, operation, iree_ok_status(),
                                          IREE_ASYNC_COMPLETION_FLAG_NONE);
    }
  }
  return count;
}

static iree_status_t iree_async_proactor_js_poll(
    iree_async_proactor_t* proactor, iree_timeout_t timeout,
    iree_host_size_t* out_completed_count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_convert_timeout_to_absolute(&timeout);
  iree_async_proactor_js_t* js_proactor = iree_async_proactor_js_cast(proactor);
  iree_host_size_t completed_count = 0;

  // Run progress callbacks (shared infrastructure with other backends).
  iree_status_t progress_status =
      iree_async_proactor_run_progress(proactor, &completed_count);
  if (!iree_status_is_ok(progress_status)) {
    if (out_completed_count) {
      *out_completed_count = completed_count;
    }
    IREE_TRACE_ZONE_END(z0);
    return progress_status;
  }

  // Drain poll-owned completions and sequence startup work.
  completed_count += iree_async_proactor_js_drain_pending(js_proactor);

  // Drain completions from the JS ring.
  completed_count += iree_async_proactor_js_drain_ring(js_proactor);

  // If nothing completed and timeout is not immediate, block for completions.
  if (completed_count == 0 && !proactor->progress_list &&
      !iree_timeout_is_immediate(timeout)) {
    iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
    uint32_t wait_result = iree_async_js_import_poll_wait(deadline_ns);
    if (wait_result == 0) {
      // Data available: drain again.
      completed_count += iree_async_proactor_js_drain_ring(js_proactor);
    }
  }

  if (out_completed_count) {
    *out_completed_count = completed_count;
  }
  IREE_TRACE_ZONE_END(z0);

  // An inline or cooperative turn can yield before the caller's deadline.
  if (completed_count == 0 && iree_timeout_as_duration_ns(timeout) == 0) {
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Wake and cancel
//===----------------------------------------------------------------------===//

static void iree_async_proactor_js_wake(iree_async_proactor_t* proactor) {
  iree_async_js_import_wake();
}

static iree_status_t iree_async_proactor_js_cancel(
    iree_async_proactor_t* proactor, iree_async_operation_t* operation) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (operation->type == IREE_ASYNC_OPERATION_TYPE_SEQUENCE) {
    IREE_TRACE_ZONE_END(z0);
    return iree_async_sequence_cancel(
        proactor, (iree_async_sequence_operation_t*)operation);
  }
  if (operation->type != IREE_ASYNC_OPERATION_TYPE_NOP &&
      operation->type != IREE_ASYNC_OPERATION_TYPE_TIMER) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "JS proactor does not support cancelling operation type %u",
        operation->type);
  }

  iree_async_operation_internal_flags_t previous_flags =
      (iree_async_operation_internal_flags_t)iree_atomic_fetch_or(
          &operation->internal_flags,
          (int32_t)IREE_ASYNC_JS_OPERATION_INTERNAL_FLAG_CANCELLED,
          iree_memory_order_release);
  if (iree_any_bit_set(previous_flags,
                       IREE_ASYNC_JS_OPERATION_INTERNAL_FLAG_CANCELLED)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_TIMER: {
      iree_async_timer_operation_t* timer =
          (iree_async_timer_operation_t*)operation;
      if (!timer->platform.js.is_token_active) {
        // Expired timers are already pending on the poll owner. They never
        // acquire a token and cancellation only changes their terminal status.
        break;
      }
      uint32_t cancelled =
          iree_async_js_import_timer_cancel(timer->platform.js.token);
      if (cancelled) {
        // Timer was cancelled before firing. Release the token and queue it so
        // the callback still fires from poll().
        iree_async_js_token_table_release(
            &iree_async_proactor_js_cast(proactor)->token_table,
            timer->platform.js.token);
        timer->platform.js.is_token_active = false;
        iree_async_proactor_js_pending_enqueue(
            iree_async_proactor_js_cast(proactor), operation);
        iree_async_js_import_schedule_drain();
      }
      // If cancelled == 0, the timer already fired. The completion will arrive
      // through the ring and be dispatched with CANCELLED status because we set
      // the internal flag above.
      break;
    }

    case IREE_ASYNC_OPERATION_TYPE_NOP:
      // NOP is in the pending queue. The CANCELLED flag is checked when it's
      // dequeued during poll.
      break;
    default:
      IREE_ASSERT_UNREACHABLE("operation type must be validated");
      IREE_BUILTIN_UNREACHABLE();
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_async_proactor_capabilities_t
iree_async_proactor_js_query_capabilities(iree_async_proactor_t* proactor) {
  return iree_async_proactor_js_cast(proactor)->capabilities;
}

//===----------------------------------------------------------------------===//
// Unavailable vtable methods
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_js_create_socket(
    iree_async_proactor_t* proactor, iree_async_socket_type_t type,
    iree_async_socket_options_t options, iree_async_socket_t** out_socket) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support sockets");
}

static iree_status_t iree_async_proactor_js_import_socket(
    iree_async_proactor_t* proactor, iree_async_primitive_t primitive,
    iree_async_socket_type_t type, iree_async_socket_flags_t flags,
    iree_async_socket_t** out_socket) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support sockets");
}

static void iree_async_proactor_js_destroy_socket(
    iree_async_proactor_t* proactor, iree_async_socket_t* socket) {
  // Should never be called since create/import return UNAVAILABLE.
  IREE_ASSERT(false, "destroy_socket called on JS proactor");
}

static iree_status_t iree_async_proactor_js_import_file(
    iree_async_proactor_t* proactor, iree_async_primitive_t primitive,
    iree_async_file_t** out_file) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support files");
}

static void iree_async_proactor_js_destroy_file(iree_async_proactor_t* proactor,
                                                iree_async_file_t* file) {
  IREE_ASSERT(false, "destroy_file called on JS proactor");
}

static iree_status_t iree_async_proactor_js_create_event(
    iree_async_proactor_t* proactor, iree_async_event_t** out_event) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support events yet");
}

static void iree_async_proactor_js_destroy_event(
    iree_async_proactor_t* proactor, iree_async_event_t* event) {
  IREE_ASSERT(false, "destroy_event called on JS proactor");
}

static iree_status_t iree_async_proactor_js_register_event_source(
    iree_async_proactor_t* proactor, iree_async_primitive_t handle,
    iree_async_event_source_callback_t callback,
    iree_async_event_source_t** out_event_source) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support event sources");
}

static void iree_async_proactor_js_unregister_event_source(
    iree_async_proactor_t* proactor, iree_async_event_source_t* event_source,
    iree_async_event_source_unregistered_callback_t callback) {
  IREE_ASSERT(false, "unregister_event_source called on JS proactor");
}

static iree_status_t iree_async_proactor_js_create_notification(
    iree_async_proactor_t* proactor, iree_async_notification_flags_t flags,
    iree_async_notification_t** out_notification) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support notifications yet");
}

static iree_status_t iree_async_proactor_js_create_notification_shared(
    iree_async_proactor_t* proactor, iree_async_notification_native_t* native,
    iree_async_notification_t** out_notification) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support shared notifications");
}

static void iree_async_proactor_js_destroy_notification(
    iree_async_proactor_t* proactor, iree_async_notification_t* notification) {
  IREE_ASSERT(false, "destroy_notification called on JS proactor");
}

static void iree_async_proactor_js_notification_signal(
    iree_async_proactor_t* proactor, iree_async_notification_t* notification,
    int32_t wake_count) {
  IREE_ASSERT(false, "notification_signal called on JS proactor");
}

static bool iree_async_proactor_js_notification_wait(
    iree_async_proactor_t* proactor, iree_async_notification_t* notification,
    uint32_t wait_token, iree_timeout_t timeout) {
  IREE_ASSERT(false, "notification_wait called on JS proactor");
  return false;
}

static iree_status_t iree_async_proactor_js_register_relay(
    iree_async_proactor_t* proactor, iree_async_relay_source_t source,
    iree_async_relay_sink_t sink, iree_async_relay_flags_t flags,
    iree_async_relay_error_callback_t error_callback,
    iree_async_relay_t** out_relay) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support relays");
}

static void iree_async_proactor_js_unregister_relay(
    iree_async_proactor_t* proactor, iree_async_relay_t* relay,
    iree_async_relay_unregistered_callback_t callback) {
  IREE_ASSERT(false, "unregister_relay called on JS proactor");
}

static iree_status_t iree_async_proactor_js_register_buffer(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_state_t* state, iree_byte_span_t buffer,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support buffer registration");
}

static iree_status_t iree_async_proactor_js_register_dmabuf(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_state_t* state, int dmabuf_fd,
    uint64_t offset, iree_host_size_t length,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support dmabuf");
}

static void iree_async_proactor_js_unregister_buffer(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_entry_t* entry,
    iree_async_buffer_registration_state_t* state) {
  IREE_ASSERT(false, "unregister_buffer called on JS proactor");
}

static iree_status_t iree_async_proactor_js_register_slab(
    iree_async_proactor_t* proactor, iree_async_slab_t* slab,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_region_t** out_region) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support slabs");
}

static iree_status_t iree_async_proactor_js_import_fence(
    iree_async_proactor_t* proactor, iree_async_primitive_t fence,
    iree_async_semaphore_t* semaphore, uint64_t signal_value) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support fences");
}

static iree_status_t iree_async_proactor_js_export_fence(
    iree_async_proactor_t* proactor, iree_async_semaphore_t* semaphore,
    uint64_t wait_value, iree_async_primitive_t* out_fence) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support fences");
}

static void iree_async_proactor_js_set_message_callback(
    iree_async_proactor_t* proactor,
    iree_async_proactor_message_callback_t callback) {
  // Accept but discard: messaging is not supported.
}

static iree_status_t iree_async_proactor_js_send_message(
    iree_async_proactor_t* target, uint64_t message_data) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support messaging");
}

static iree_status_t iree_async_proactor_js_subscribe_signal(
    iree_async_proactor_t* proactor, iree_async_signal_t signal,
    iree_async_signal_callback_t callback,
    iree_async_signal_subscription_t** out_subscription) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support signals");
}

static void iree_async_proactor_js_unsubscribe_signal(
    iree_async_proactor_t* proactor,
    iree_async_signal_subscription_t* subscription) {
  IREE_ASSERT(false, "unsubscribe_signal called on JS proactor");
}

//===----------------------------------------------------------------------===//
// Lifecycle and vtable
//===----------------------------------------------------------------------===//

static void iree_async_proactor_js_destroy(iree_async_proactor_t* proactor) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_proactor_js_t* js_proactor = iree_async_proactor_js_cast(proactor);
  iree_allocator_t allocator = proactor->allocator;

  // Poll-owned work must be drained before destruction.
  IREE_ASSERT(js_proactor->pending_head == NULL,
              "JS proactor destroyed with pending operations");

  iree_async_js_token_table_deinitialize(&js_proactor->token_table);
  iree_allocator_free(allocator, js_proactor);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_async_proactor_create_js(
    iree_async_proactor_options_t options, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_proactor);
  *out_proactor = NULL;

  // Token table capacity from options, with default fallback.
  iree_host_size_t token_table_capacity = options.max_concurrent_operations;
  if (token_table_capacity == 0) {
    token_table_capacity = IREE_ASYNC_JS_DEFAULT_TOKEN_TABLE_CAPACITY;
  }

  iree_host_size_t completion_buffer_capacity =
      IREE_ASYNC_JS_DEFAULT_COMPLETION_BUFFER_CAPACITY;

  // Calculate allocation layout: [proactor | completion_buffer[]]
  iree_host_size_t total_size = 0;
  iree_host_size_t completion_buffer_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(sizeof(iree_async_proactor_js_t), &total_size,
                             IREE_STRUCT_FIELD(completion_buffer_capacity,
                                               iree_async_js_completion_entry_t,
                                               &completion_buffer_offset)));

  // Single allocation for proactor + completion buffer.
  iree_async_proactor_js_t* proactor = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&proactor));
  memset(proactor, 0, total_size);

  // Initialize base proactor.
  iree_async_proactor_initialize(&iree_async_proactor_js_vtable,
                                 options.debug_name, allocator,
                                 &proactor->base);

  // Initialize token table (separately allocated by the table itself).
  iree_status_t status = iree_async_js_token_table_initialize(
      token_table_capacity, allocator, &proactor->token_table);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, proactor);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Set up completion buffer pointer into trailing data.
  proactor->completion_buffer =
      (iree_async_js_completion_entry_t*)((uint8_t*)proactor +
                                          completion_buffer_offset);
  proactor->completion_buffer_capacity = completion_buffer_capacity;

  // Poll-owned queue starts empty.
  proactor->pending_head = NULL;
  proactor->pending_tail = NULL;

  // Initialize sequence emulator for SEQUENCE operation support. Uses the
  // public submit_one API which re-enters through the vtable submit path.
  iree_async_sequence_emulator_initialize(&proactor->sequence_emulator,
                                          &proactor->base,
                                          iree_async_proactor_submit_one);

  // Store capabilities. The JS proactor supports absolute timeouts natively
  // (JS setTimeout uses absolute deadlines internally). LINKED_OPERATIONS is
  // emulated in userspace via linked_next chains — the proactor builds a
  // linked list during submit and dispatches continuations on completion.
  iree_async_proactor_capabilities_t supported_capabilities =
      IREE_ASYNC_PROACTOR_CAPABILITY_ABSOLUTE_TIMEOUT |
      IREE_ASYNC_PROACTOR_CAPABILITY_LINKED_OPERATIONS;
  proactor->capabilities =
      options.allowed_capabilities & supported_capabilities;

  *out_proactor = &proactor->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static const iree_async_proactor_vtable_t iree_async_proactor_js_vtable = {
    .destroy = iree_async_proactor_js_destroy,
    .query_capabilities = iree_async_proactor_js_query_capabilities,
    .submit = iree_async_proactor_js_submit,
    .poll = iree_async_proactor_js_poll,
    .wake = iree_async_proactor_js_wake,
    .cancel = iree_async_proactor_js_cancel,
    .create_socket = iree_async_proactor_js_create_socket,
    .import_socket = iree_async_proactor_js_import_socket,
    .destroy_socket = iree_async_proactor_js_destroy_socket,
    .import_file = iree_async_proactor_js_import_file,
    .destroy_file = iree_async_proactor_js_destroy_file,
    .create_event = iree_async_proactor_js_create_event,
    .destroy_event = iree_async_proactor_js_destroy_event,
    .register_event_source = iree_async_proactor_js_register_event_source,
    .unregister_event_source = iree_async_proactor_js_unregister_event_source,
    .create_notification = iree_async_proactor_js_create_notification,
    .create_notification_shared =
        iree_async_proactor_js_create_notification_shared,
    .destroy_notification = iree_async_proactor_js_destroy_notification,
    .notification_signal = iree_async_proactor_js_notification_signal,
    .notification_wait = iree_async_proactor_js_notification_wait,
    .register_relay = iree_async_proactor_js_register_relay,
    .unregister_relay = iree_async_proactor_js_unregister_relay,
    .register_buffer = iree_async_proactor_js_register_buffer,
    .register_dmabuf = iree_async_proactor_js_register_dmabuf,
    .unregister_buffer = iree_async_proactor_js_unregister_buffer,
    .register_slab = iree_async_proactor_js_register_slab,
    .import_fence = iree_async_proactor_js_import_fence,
    .export_fence = iree_async_proactor_js_export_fence,
    .set_message_callback = iree_async_proactor_js_set_message_callback,
    .send_message = iree_async_proactor_js_send_message,
    .subscribe_signal = iree_async_proactor_js_subscribe_signal,
    .unsubscribe_signal = iree_async_proactor_js_unsubscribe_signal,
};
