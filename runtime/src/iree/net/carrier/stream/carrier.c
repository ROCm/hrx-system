// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/stream/carrier.h"

#include <string.h>

#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/base/alignment.h"
#include "iree/base/internal/math.h"
#include "iree/base/threading/mutex.h"

typedef struct iree_net_stream_carrier_t iree_net_stream_carrier_t;

// Send-slot ownership transitions are serialized by the carrier mutex. A
// submitted or cancelling slot belongs to the kernel; a completing slot belongs
// to the owning proactor callback.
typedef enum iree_net_stream_send_slot_state_e {
  IREE_NET_STREAM_SEND_SLOT_STATE_FREE = 0,
  IREE_NET_STREAM_SEND_SLOT_STATE_PREPARING = 1,
  IREE_NET_STREAM_SEND_SLOT_STATE_QUEUED = 2,
  IREE_NET_STREAM_SEND_SLOT_STATE_SUBMITTED = 3,
  IREE_NET_STREAM_SEND_SLOT_STATE_CANCELLING = 4,
  IREE_NET_STREAM_SEND_SLOT_STATE_COMPLETING = 5,
  IREE_NET_STREAM_SEND_SLOT_STATE_LOCAL_COMPLETION = 6,
  IREE_NET_STREAM_SEND_SLOT_STATE_DETACHED = 7,
  // All logical bytes accepted; native source-reuse notification outstanding.
  IREE_NET_STREAM_SEND_SLOT_STATE_RETIRING = 8,
} iree_net_stream_send_slot_state_t;

// One bounded logical send.
typedef struct iree_net_stream_send_slot_t {
  // Socket or local-completion operation; first for completion downcasting.
  iree_async_socket_send_operation_t operation;

  // Next slot in the send FIFO or a detached completion list.
  struct iree_net_stream_send_slot_t* next;

  // Stable descriptors retained across queueing and partial sends. One
  // transport-private prefix may precede the maximum caller span count.
  iree_async_span_t spans[IREE_ASYNC_SOCKET_SEND_MAX_BUFFERS + 1];

  // Completion callback installed before the send is published.
  iree_net_send_completion_callback_t completion_callback;

  // Optional carrier-owned oversized prefix.
  void* owned_buffer;

  // Owned status transferred through a local completion operation.
  iree_status_t local_completion_status;

  // Total logical payload length.
  iree_host_size_t total_length;

  // Payload bytes completed by prior socket operations.
  iree_host_size_t bytes_transferred;

  // Bytes represented by the currently submitted physical socket vector.
  iree_host_size_t submitted_length;

  // Number of valid descriptors in |spans|.
  iree_host_size_t span_count;

  // Index of the first descriptor not fully sent.
  iree_host_size_t first_span;

  // Current ownership state.
  iree_net_stream_send_slot_state_t state;

  // True while the slot owns one retain on each registered region.
  bool regions_retained;
} iree_net_stream_send_slot_t;

typedef enum iree_net_stream_receive_state_e {
  IREE_NET_STREAM_RECEIVE_STATE_RETIRED = 0,
  IREE_NET_STREAM_RECEIVE_STATE_SUBMITTED = 1,
  IREE_NET_STREAM_RECEIVE_STATE_CANCELLING = 2,
  IREE_NET_STREAM_RECEIVE_STATE_COMPLETING = 3,
} iree_net_stream_receive_state_t;

typedef enum iree_net_stream_receive_lease_state_e {
  IREE_NET_STREAM_RECEIVE_LEASE_STATE_RELEASED = 0,
  IREE_NET_STREAM_RECEIVE_LEASE_STATE_PENDING = 1,
  IREE_NET_STREAM_RECEIVE_LEASE_STATE_RETAINED = 2,
} iree_net_stream_receive_lease_state_t;

typedef enum iree_net_stream_carrier_flag_bits_e {
  IREE_NET_STREAM_CARRIER_FLAG_NONE = 0u,

  // One ordered write lane is submitted or owned by its progress callback.
  IREE_NET_STREAM_CARRIER_FLAG_SEND_DISPATCH_ACTIVE = 1u << 0,
  IREE_NET_STREAM_CARRIER_FLAG_SEND_SHUTDOWN_INITIATED = 1u << 1,
  IREE_NET_STREAM_CARRIER_FLAG_SOCKET_WRITE_SHUTDOWN_ISSUED = 1u << 2,
} iree_net_stream_carrier_flag_bits_t;
typedef uint32_t iree_net_stream_carrier_flags_t;

// Per-buffer wrapper accounting for consumer ownership of native storage.
typedef struct iree_net_stream_receive_lease_context_t {
  // Carrier retained only while the consumer owns the wrapped lease.
  iree_net_stream_carrier_t* carrier;

  // Original callback returning this buffer to its source.
  iree_async_buffer_recycle_callback_t recycle;

  // Current wrapper ownership state.
  iree_atomic_int32_t state;
} iree_net_stream_receive_lease_context_t;

struct iree_net_stream_carrier_t {
  // Base carrier; must be first for upcasting.
  iree_net_carrier_t base;

  // Serializes send admission, slot queues, and shutdown publication.
  iree_slim_mutex_t mutex;

  // Proactor owning socket operations. Retained.
  iree_async_proactor_t* proactor;

  // Connected stream socket. Retained.
  iree_async_socket_t* socket;

  // Pool supplying receive buffers. Retained.
  iree_async_buffer_pool_t* receive_pool;

  // Contiguous bounded send slot storage.
  iree_net_stream_send_slot_t* send_slots;

  // Number of entries in |send_slots|.
  uint32_t send_slot_count;

  // Number of sends currently owned.
  uint32_t send_slots_in_use;

  // Maximum generated-prefix bytes available per send slot.
  iree_host_size_t generated_prefix_capacity;

  // Minimum attempted native send extent for requesting kernel copy avoidance.
  iree_host_size_t zero_copy_min_send_size;

  // Cache-line-isolated byte stride between generated-prefix slices.
  iree_host_size_t generated_prefix_stride;

  // Preallocated generated-prefix storage indexed by send slot.
  uint8_t* generated_prefix_slab;

  // First committed send waiting for socket submission.
  iree_net_stream_send_slot_t* send_queue_head;

  // Last committed send waiting for socket submission.
  iree_net_stream_send_slot_t* send_queue_tail;

  // Mutex-protected dispatch and shutdown state.
  iree_net_stream_carrier_flags_t flags;

  // Single receive operation reused for the carrier lifetime.
  iree_async_socket_recv_pool_operation_t receive_operation;

  // Mutex-protected receive operation ownership state.
  iree_net_stream_receive_state_t receive_state;

  // Number of receive leases currently retained by consumers.
  iree_atomic_int32_t retained_receive_lease_count;

  // Per-buffer wrappers indexed by receive lease buffer index.
  iree_net_stream_receive_lease_context_t* receive_lease_contexts;

  // Number of entries in |receive_lease_contexts|.
  iree_host_size_t receive_lease_context_count;

  // Callback invoked after every accepted operation drains.
  struct {
    // Function invoked after transition to DEACTIVATED.
    iree_net_carrier_deactivate_callback_fn_t fn;

    // Opaque value passed to |fn|.
    void* user_data;
  } deactivate_callback;
};

static iree_net_stream_carrier_t* iree_net_stream_carrier_cast(
    iree_net_carrier_t* base_carrier) {
  return (iree_net_stream_carrier_t*)base_carrier;
}

static void iree_net_stream_carrier_maybe_complete_deactivation(
    iree_net_stream_carrier_t* carrier) {
  if (iree_atomic_load(&carrier->base.pending_operations,
                       iree_memory_order_acquire) != 0) {
    return;
  }
  if (!iree_net_carrier_try_transition_state(
          &carrier->base, IREE_NET_CARRIER_STATE_DRAINING,
          IREE_NET_CARRIER_STATE_DEACTIVATED)) {
    return;
  }

  iree_net_carrier_deactivate_callback_fn_t callback =
      carrier->deactivate_callback.fn;
  void* user_data = carrier->deactivate_callback.user_data;
  carrier->deactivate_callback.fn = NULL;
  carrier->deactivate_callback.user_data = NULL;
  callback(user_data);
}

// Retirement must be the caller's final carrier access.
static void iree_net_stream_carrier_retire_pending_operation(
    iree_net_stream_carrier_t* carrier) {
  if (iree_net_carrier_retire_pending_operation(&carrier->base)) {
    iree_net_stream_carrier_maybe_complete_deactivation(carrier);
  }
}

static iree_status_t iree_net_stream_validate_send_span(
    iree_async_span_t span) {
  if (span.length == 0) {
    return iree_ok_status();
  }
  if (!iree_async_span_is_cpu_accessible(span)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "stream send span is not CPU-accessible");
  }
  if (!span.region) {
    if (span.offset == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "stream send span has null storage");
    }
    return iree_ok_status();
  }
  if (!iree_any_bit_set(span.region->access_flags,
                        IREE_ASYNC_BUFFER_ACCESS_FLAG_READ)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream send region does not permit reads");
  }
  if (span.offset > span.region->length ||
      span.length > span.region->length - span.offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "stream send span range [%" PRIhsz ", %" PRIhsz
                            ") exceeds its registered region length %" PRIhsz,
                            span.offset, span.offset + span.length,
                            span.region->length);
  }
  return iree_ok_status();
}

static iree_net_stream_send_slot_t* iree_net_stream_find_free_send_slot_locked(
    iree_net_stream_carrier_t* carrier) {
  for (uint32_t i = 0; i < carrier->send_slot_count; ++i) {
    if (carrier->send_slots[i].state == IREE_NET_STREAM_SEND_SLOT_STATE_FREE) {
      return &carrier->send_slots[i];
    }
  }
  return NULL;
}

static uint8_t* iree_net_stream_send_slot_prefix_storage(
    iree_net_stream_carrier_t* carrier, iree_net_stream_send_slot_t* slot) {
  if (!carrier->generated_prefix_slab) {
    return NULL;
  }
  const iree_host_size_t slot_index = slot - carrier->send_slots;
  return carrier->generated_prefix_slab +
         slot_index * carrier->generated_prefix_stride;
}

static void iree_net_stream_enqueue_send_locked(
    iree_net_stream_carrier_t* carrier, iree_net_stream_send_slot_t* slot) {
  IREE_ASSERT(slot->state == IREE_NET_STREAM_SEND_SLOT_STATE_QUEUED);
  slot->next = NULL;
  if (carrier->send_queue_tail) {
    carrier->send_queue_tail->next = slot;
  } else {
    carrier->send_queue_head = slot;
  }
  carrier->send_queue_tail = slot;
}

static iree_net_stream_send_slot_t* iree_net_stream_pop_send_locked(
    iree_net_stream_carrier_t* carrier) {
  iree_net_stream_send_slot_t* slot = carrier->send_queue_head;
  if (!slot) {
    return NULL;
  }
  carrier->send_queue_head = slot->next;
  if (!carrier->send_queue_head) {
    carrier->send_queue_tail = NULL;
  }
  slot->next = NULL;
  slot->state = IREE_NET_STREAM_SEND_SLOT_STATE_SUBMITTED;
  return slot;
}

static iree_net_stream_send_slot_t* iree_net_stream_claim_send_dispatch_locked(
    iree_net_stream_carrier_t* carrier) {
  if (iree_any_bit_set(carrier->flags,
                       IREE_NET_STREAM_CARRIER_FLAG_SEND_DISPATCH_ACTIVE) ||
      !carrier->send_queue_head) {
    return NULL;
  }
  carrier->flags |= IREE_NET_STREAM_CARRIER_FLAG_SEND_DISPATCH_ACTIVE;
  return iree_net_stream_pop_send_locked(carrier);
}

static iree_net_stream_send_slot_t* iree_net_stream_find_submitted_send_locked(
    iree_net_stream_carrier_t* carrier) {
  iree_net_stream_send_slot_t* submitted_slot = NULL;
  for (uint32_t i = 0; i < carrier->send_slot_count; ++i) {
    iree_net_stream_send_slot_t* slot = &carrier->send_slots[i];
    if (slot->state != IREE_NET_STREAM_SEND_SLOT_STATE_SUBMITTED) {
      continue;
    }
    IREE_ASSERT(!submitted_slot,
                "stream carrier submitted multiple socket sends concurrently");
    submitted_slot = slot;
  }
  return submitted_slot;
}

static void iree_net_stream_send_slot_release_resources(
    iree_net_stream_carrier_t* carrier, iree_net_stream_send_slot_t* slot) {
  if (slot->regions_retained) {
    iree_async_span_list_release_regions(
        iree_async_span_list_make(slot->spans, slot->span_count));
    slot->regions_retained = false;
  }
  iree_allocator_free(carrier->base.host_allocator, slot->owned_buffer);
  slot->owned_buffer = NULL;
}

static void iree_net_stream_recycle_send_slot_locked(
    iree_net_stream_carrier_t* carrier, iree_net_stream_send_slot_t* slot) {
  IREE_ASSERT(slot->state != IREE_NET_STREAM_SEND_SLOT_STATE_FREE);
  IREE_ASSERT(carrier->send_slots_in_use > 0);
  IREE_ASSERT(iree_status_is_ok(slot->local_completion_status));
  slot->next = NULL;
  slot->completion_callback = (iree_net_send_completion_callback_t){0};
  slot->total_length = 0;
  slot->bytes_transferred = 0;
  slot->submitted_length = 0;
  slot->span_count = 0;
  slot->first_span = 0;
  slot->state = IREE_NET_STREAM_SEND_SLOT_STATE_FREE;
  --carrier->send_slots_in_use;
}

static iree_net_stream_send_slot_t* iree_net_stream_detach_send_queue_locked(
    iree_net_stream_carrier_t* carrier,
    iree_net_stream_send_slot_t** out_tail) {
  iree_net_stream_send_slot_t* head = carrier->send_queue_head;
  iree_net_stream_send_slot_t* tail = carrier->send_queue_tail;
  carrier->send_queue_head = NULL;
  carrier->send_queue_tail = NULL;

  for (iree_net_stream_send_slot_t* slot = head; slot; slot = slot->next) {
    IREE_ASSERT(slot->state == IREE_NET_STREAM_SEND_SLOT_STATE_QUEUED);
    slot->state = IREE_NET_STREAM_SEND_SLOT_STATE_DETACHED;
  }
  if (out_tail) {
    *out_tail = tail;
  }
  return head;
}

static iree_status_t iree_net_stream_cancel_operation_locked(
    iree_net_stream_carrier_t* carrier, iree_async_operation_t* operation) {
  iree_status_t status =
      iree_async_proactor_cancel(carrier->proactor, operation);
  if (iree_status_is_not_found(status)) {
    iree_status_free(status);
    return iree_ok_status();
  }
  return status;
}

static iree_status_t iree_net_stream_request_receive_cancellation_locked(
    iree_net_stream_carrier_t* carrier) {
  if (carrier->receive_state != IREE_NET_STREAM_RECEIVE_STATE_SUBMITTED) {
    return iree_ok_status();
  }
  iree_status_t status = iree_net_stream_cancel_operation_locked(
      carrier, &carrier->receive_operation.base);
  if (iree_status_is_ok(status)) {
    carrier->receive_state = IREE_NET_STREAM_RECEIVE_STATE_CANCELLING;
  }
  return status;
}

static iree_status_t iree_net_stream_request_send_cancellation_locked(
    iree_net_stream_carrier_t* carrier) {
  iree_net_stream_send_slot_t* submitted_slot =
      iree_net_stream_find_submitted_send_locked(carrier);
  if (!submitted_slot) {
    return iree_ok_status();
  }
  iree_status_t status = iree_net_stream_cancel_operation_locked(
      carrier, &submitted_slot->operation.base);
  if (iree_status_is_ok(status)) {
    submitted_slot->state = IREE_NET_STREAM_SEND_SLOT_STATE_CANCELLING;
  }
  return status;
}

static bool iree_net_stream_claim_write_shutdown_locked(
    iree_net_stream_carrier_t* carrier) {
  if (!iree_any_bit_set(carrier->flags,
                        IREE_NET_STREAM_CARRIER_FLAG_SEND_SHUTDOWN_INITIATED) ||
      iree_any_bit_set(
          carrier->flags,
          IREE_NET_STREAM_CARRIER_FLAG_SOCKET_WRITE_SHUTDOWN_ISSUED) ||
      carrier->send_slots_in_use != 0 ||
      iree_net_carrier_state(&carrier->base) != IREE_NET_CARRIER_STATE_ACTIVE) {
    return false;
  }
  carrier->flags |= IREE_NET_STREAM_CARRIER_FLAG_SOCKET_WRITE_SHUTDOWN_ISSUED;
  return true;
}

static void iree_net_stream_complete_detached_sends(
    iree_net_stream_carrier_t* carrier, iree_net_stream_send_slot_t* slot,
    iree_status_code_t status_code) {
  while (slot) {
    IREE_ASSERT(slot->state == IREE_NET_STREAM_SEND_SLOT_STATE_DETACHED);
    iree_net_stream_send_slot_t* next = slot->next;
    const iree_net_send_completion_callback_t completion_callback =
        slot->completion_callback;
    const iree_host_size_t bytes_transferred = slot->bytes_transferred;
    iree_net_stream_send_slot_release_resources(carrier, slot);

    iree_slim_mutex_lock(&carrier->mutex);
    iree_net_stream_recycle_send_slot_locked(carrier, slot);
    iree_slim_mutex_unlock(&carrier->mutex);

    if (completion_callback.fn) {
      completion_callback.fn(
          completion_callback.user_data,
          iree_make_status(
              status_code,
              "stream send did not complete before transport stop"),
          bytes_transferred);
    }
    iree_net_stream_carrier_retire_pending_operation(carrier);
    slot = next;
  }
}

// Publishes terminal failure. Callers retain an operation or lifetime
// reference.
static void iree_net_stream_publish_failure(iree_net_stream_carrier_t* carrier,
                                            iree_status_t status) {
  IREE_ASSERT(!iree_status_is_ok(status));
  iree_status_t receive_cancel_status = iree_ok_status();
  iree_status_t send_cancel_status = iree_ok_status();

  iree_slim_mutex_lock(&carrier->mutex);
  carrier->flags |= IREE_NET_STREAM_CARRIER_FLAG_SEND_SHUTDOWN_INITIATED |
                    IREE_NET_STREAM_CARRIER_FLAG_SOCKET_WRITE_SHUTDOWN_ISSUED;
  // Accepted sends stay on their single dispatch lane so only its proactor
  // callback can deliver their terminal completions.
  receive_cancel_status =
      iree_net_stream_request_receive_cancellation_locked(carrier);
  send_cancel_status =
      iree_net_stream_request_send_cancellation_locked(carrier);
  iree_slim_mutex_unlock(&carrier->mutex);

  status = iree_status_join(status, receive_cancel_status);
  status = iree_status_join(status, send_cancel_status);
  status = iree_status_join(
      status, iree_async_socket_shutdown(carrier->socket,
                                         IREE_ASYNC_SOCKET_SHUTDOWN_BOTH));
  iree_net_carrier_report_terminal_error(&carrier->base, status);
}

static void iree_net_stream_fail_without_operation(
    iree_net_stream_carrier_t* carrier, iree_status_t status) {
  iree_net_carrier_retain(&carrier->base);
  iree_net_stream_publish_failure(carrier, status);
  iree_net_carrier_release(&carrier->base);
}

static void iree_net_stream_issue_deferred_write_shutdown(
    iree_net_stream_carrier_t* carrier) {
  iree_status_t status = iree_async_socket_shutdown(
      carrier->socket, IREE_ASYNC_SOCKET_SHUTDOWN_WRITE);
  if (!iree_status_is_ok(status)) {
    iree_net_stream_fail_without_operation(carrier, status);
  }
}

static bool iree_net_stream_advance_send_slot(
    iree_net_stream_send_slot_t* slot, iree_host_size_t bytes_transferred) {
  iree_host_size_t remaining = bytes_transferred;
  while (remaining > 0 && slot->first_span < slot->span_count) {
    iree_async_span_t* span = &slot->spans[slot->first_span];
    if (remaining >= span->length) {
      remaining -= span->length;
      ++slot->first_span;
    } else {
      span->offset += remaining;
      span->length -= remaining;
      remaining = 0;
    }
  }
  if (remaining != 0) {
    return false;
  }
  slot->bytes_transferred += bytes_transferred;
  return true;
}

static void iree_net_stream_process_send_result(
    iree_net_stream_carrier_t* carrier, iree_net_stream_send_slot_t* slot,
    iree_status_t status, iree_host_size_t bytes_sent,
    iree_async_completion_flags_t flags);

static void iree_net_stream_advance_send_dispatch(
    iree_net_stream_carrier_t* carrier);

static void iree_net_stream_local_send_completed(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_stream_carrier_t* carrier = (iree_net_stream_carrier_t*)user_data;
  iree_net_stream_send_slot_t* slot = (iree_net_stream_send_slot_t*)operation;
  IREE_ASSERT(!iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE));

  iree_slim_mutex_lock(&carrier->mutex);
  IREE_ASSERT(slot->state == IREE_NET_STREAM_SEND_SLOT_STATE_LOCAL_COMPLETION,
              "stream local send completed from state %d", (int)slot->state);
  slot->state = IREE_NET_STREAM_SEND_SLOT_STATE_COMPLETING;
  iree_status_t completion_status = slot->local_completion_status;
  slot->local_completion_status = iree_ok_status();
  const iree_net_send_completion_callback_t completion_callback =
      slot->completion_callback;
  iree_slim_mutex_unlock(&carrier->mutex);

  completion_status = iree_status_join(completion_status, status);
  iree_net_stream_send_slot_release_resources(carrier, slot);

  iree_slim_mutex_lock(&carrier->mutex);
  iree_net_stream_recycle_send_slot_locked(carrier, slot);
  const bool issue_write_shutdown =
      iree_net_stream_claim_write_shutdown_locked(carrier);
  iree_slim_mutex_unlock(&carrier->mutex);

  completion_callback.fn(completion_callback.user_data, completion_status,
                         /*bytes_transferred=*/0);
  if (issue_write_shutdown) {
    iree_net_stream_issue_deferred_write_shutdown(carrier);
  }
  iree_net_stream_carrier_retire_pending_operation(carrier);
}

static void iree_net_stream_initial_send_submit_completed(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_stream_carrier_t* carrier = (iree_net_stream_carrier_t*)user_data;
  iree_net_stream_send_slot_t* slot = (iree_net_stream_send_slot_t*)operation;
  IREE_ASSERT(!iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE));

  iree_slim_mutex_lock(&carrier->mutex);
  IREE_ASSERT(slot->state == IREE_NET_STREAM_SEND_SLOT_STATE_LOCAL_COMPLETION,
              "stream initial send completion arrived from state %d",
              (int)slot->state);
  slot->state = IREE_NET_STREAM_SEND_SLOT_STATE_COMPLETING;
  iree_status_t completion_status = slot->local_completion_status;
  slot->local_completion_status = iree_ok_status();
  iree_slim_mutex_unlock(&carrier->mutex);

  completion_status = iree_status_join(completion_status, status);
  iree_net_stream_process_send_result(carrier, slot, completion_status,
                                      /*bytes_sent=*/0, flags);
}

static iree_status_t iree_net_stream_schedule_local_send_completion_locked(
    iree_net_stream_carrier_t* carrier, iree_net_stream_send_slot_t* slot,
    iree_status_t status, iree_async_completion_fn_t completion_fn) {
  IREE_ASSERT(slot->state == IREE_NET_STREAM_SEND_SLOT_STATE_LOCAL_COMPLETION);
  IREE_ASSERT(!iree_status_is_ok(status));
  IREE_ASSERT(iree_status_is_ok(slot->local_completion_status));
  slot->local_completion_status = status;
  iree_async_operation_zero(&slot->operation.base, sizeof(slot->operation));
  iree_async_operation_initialize(
      &slot->operation.base, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, completion_fn, carrier);
  iree_status_t submit_status =
      iree_async_proactor_submit_one(carrier->proactor, &slot->operation.base);
  if (iree_status_is_ok(submit_status)) {
    return iree_ok_status();
  }

  status = slot->local_completion_status;
  slot->local_completion_status = iree_ok_status();
  return iree_status_join(status, submit_status);
}

static void iree_net_stream_socket_send_completed(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_stream_carrier_t* carrier = (iree_net_stream_carrier_t*)user_data;
  iree_net_stream_send_slot_t* slot = (iree_net_stream_send_slot_t*)operation;
  iree_slim_mutex_lock(&carrier->mutex);
  IREE_ASSERT(slot->state == IREE_NET_STREAM_SEND_SLOT_STATE_SUBMITTED ||
                  slot->state == IREE_NET_STREAM_SEND_SLOT_STATE_CANCELLING ||
                  slot->state == IREE_NET_STREAM_SEND_SLOT_STATE_RETIRING,
              "stream send completed from state %d", (int)slot->state);
  const iree_host_size_t bytes_sent = slot->operation.bytes_sent;
  if (iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE)) {
    IREE_ASSERT(iree_status_is_ok(status));
    // A partial attempt keeps the write lane until its operation can be reused.
    // A complete logical frame relinquishes only the lane, not its source
    // storage or user completion. No later frame can interleave its bytes.
    const bool frame_accepted =
        bytes_sent == slot->submitted_length &&
        bytes_sent == slot->total_length - slot->bytes_transferred;
    if (frame_accepted) {
      slot->bytes_transferred = slot->total_length;
      slot->state = IREE_NET_STREAM_SEND_SLOT_STATE_RETIRING;
    }
    iree_slim_mutex_unlock(&carrier->mutex);
    iree_status_free(status);
    if (frame_accepted) {
      iree_net_stream_advance_send_dispatch(carrier);
    }
    return;
  }
  if (slot->state != IREE_NET_STREAM_SEND_SLOT_STATE_RETIRING) {
    slot->state = IREE_NET_STREAM_SEND_SLOT_STATE_COMPLETING;
  }
  iree_slim_mutex_unlock(&carrier->mutex);
  iree_net_stream_process_send_result(carrier, slot, status, bytes_sent, flags);
}

static iree_status_t iree_net_stream_submit_send_slot_locked(
    iree_net_stream_carrier_t* carrier, iree_net_stream_send_slot_t* slot) {
  IREE_ASSERT(slot->state == IREE_NET_STREAM_SEND_SLOT_STATE_SUBMITTED);
  IREE_ASSERT(slot->first_span < slot->span_count);
  iree_async_operation_zero(&slot->operation.base, sizeof(slot->operation));
  iree_async_operation_initialize(
      &slot->operation.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_stream_socket_send_completed,
      carrier);
  slot->operation.socket = carrier->socket;
  const iree_host_size_t submitted_span_count =
      iree_min(slot->span_count - slot->first_span,
               (iree_host_size_t)IREE_ASYNC_SOCKET_SEND_MAX_BUFFERS);
  slot->operation.buffers = iree_async_span_list_make(
      &slot->spans[slot->first_span], submitted_span_count);
  slot->submitted_length = 0;
  for (iree_host_size_t i = 0; i < submitted_span_count; ++i) {
    slot->submitted_length += slot->operation.buffers.values[i].length;
  }
  slot->operation.send_flags = IREE_ASYNC_SOCKET_SEND_FLAG_REPORT_PROGRESS;
  if (slot->submitted_length < carrier->zero_copy_min_send_size ||
      carrier->zero_copy_min_send_size == IREE_HOST_SIZE_MAX) {
    slot->operation.send_flags |= IREE_ASYNC_SOCKET_SEND_FLAG_NO_ZERO_COPY;
  }
  return iree_async_proactor_submit_one(carrier->proactor,
                                        &slot->operation.base);
}

static iree_status_t iree_net_stream_start_send_dispatch_locked(
    iree_net_stream_carrier_t* carrier,
    iree_net_stream_send_slot_t** out_failed_slot) {
  *out_failed_slot = NULL;
  iree_net_stream_send_slot_t* slot =
      iree_net_stream_claim_send_dispatch_locked(carrier);
  if (!slot) {
    return iree_ok_status();
  }
  iree_status_t status = iree_net_stream_submit_send_slot_locked(carrier, slot);
  if (!iree_status_is_ok(status)) {
    slot->state = IREE_NET_STREAM_SEND_SLOT_STATE_LOCAL_COMPLETION;
    *out_failed_slot = slot;
  }
  return status;
}

static void iree_net_stream_process_send_result(
    iree_net_stream_carrier_t* carrier, iree_net_stream_send_slot_t* slot,
    iree_status_t status, iree_host_size_t bytes_sent,
    iree_async_completion_flags_t flags) {
  const bool is_retiring =
      slot->state == IREE_NET_STREAM_SEND_SLOT_STATE_RETIRING;
  if (iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_CANCELLED) &&
      iree_status_is_ok(status)) {
    status = iree_make_status(IREE_STATUS_CANCELLED,
                              "stream socket send was cancelled");
  }

  const iree_host_size_t remaining_length =
      slot->total_length - slot->bytes_transferred;
  if (!is_retiring && iree_status_is_ok(status)) {
    if (bytes_sent == 0) {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "stream socket send made no forward progress");
    } else if (bytes_sent > remaining_length ||
               bytes_sent > slot->submitted_length ||
               !iree_net_stream_advance_send_slot(slot, bytes_sent)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "stream socket reported %" PRIhsz " bytes for a %" PRIhsz
          " byte submitted vector (%" PRIhsz " logical bytes remain)",
          bytes_sent, slot->submitted_length, remaining_length);
    }
  }

  if (iree_status_is_ok(status) &&
      slot->bytes_transferred < slot->total_length) {
    bool resubmitted = false;
    iree_status_t submit_status = iree_ok_status();
    iree_slim_mutex_lock(&carrier->mutex);
    if (iree_net_carrier_state(&carrier->base) ==
            IREE_NET_CARRIER_STATE_ACTIVE &&
        !iree_net_carrier_has_terminal_error(&carrier->base)) {
      slot->state = IREE_NET_STREAM_SEND_SLOT_STATE_SUBMITTED;
      submit_status = iree_net_stream_submit_send_slot_locked(carrier, slot);
      if (iree_status_is_ok(submit_status)) {
        resubmitted = true;
      } else {
        slot->state = IREE_NET_STREAM_SEND_SLOT_STATE_COMPLETING;
      }
    }
    iree_slim_mutex_unlock(&carrier->mutex);
    if (resubmitted) {
      iree_status_free(status);
      return;
    }
    iree_status_free(status);
    if (!iree_status_is_ok(submit_status)) {
      status = submit_status;
    } else {
      status = iree_net_carrier_clone_terminal_error(&carrier->base);
      if (iree_status_is_ok(status)) {
        status =
            iree_make_status(IREE_STATUS_CANCELLED,
                             "stream send interrupted by carrier deactivation");
      }
    }
  }

  const iree_net_send_completion_callback_t completion_callback =
      slot->completion_callback;
  const iree_host_size_t total_bytes_transferred = slot->bytes_transferred;
  const bool send_succeeded = iree_status_is_ok(status);
  iree_net_stream_send_slot_release_resources(carrier, slot);

  iree_slim_mutex_lock(&carrier->mutex);
  iree_net_stream_recycle_send_slot_locked(carrier, slot);
  if (!send_succeeded) {
    carrier->flags |= IREE_NET_STREAM_CARRIER_FLAG_SEND_SHUTDOWN_INITIATED;
  }
  iree_slim_mutex_unlock(&carrier->mutex);

  if (!send_succeeded &&
      iree_net_carrier_state(&carrier->base) == IREE_NET_CARRIER_STATE_ACTIVE &&
      !iree_net_carrier_has_terminal_error(&carrier->base)) {
    iree_net_stream_publish_failure(carrier, iree_status_clone(status));
  }

  completion_callback.fn(completion_callback.user_data, status,
                         total_bytes_transferred);

  if (!is_retiring) {
    iree_net_stream_advance_send_dispatch(carrier);
  } else {
    iree_slim_mutex_lock(&carrier->mutex);
    const bool issue_write_shutdown =
        iree_net_stream_claim_write_shutdown_locked(carrier);
    iree_slim_mutex_unlock(&carrier->mutex);
    if (issue_write_shutdown) {
      iree_net_stream_issue_deferred_write_shutdown(carrier);
    }
  }
  iree_net_stream_carrier_retire_pending_operation(carrier);
}

// Hands the single ordered write lane to its next frame. Prior full-frame
// writes may still own their independent slots pending native retirement.
static void iree_net_stream_advance_send_dispatch(
    iree_net_stream_carrier_t* carrier) {
  iree_net_stream_send_slot_t* next_slot = NULL;
  iree_net_stream_send_slot_t* detached_sends = NULL;
  iree_status_t next_submit_status = iree_ok_status();
  iree_slim_mutex_lock(&carrier->mutex);
  if (iree_net_carrier_state(&carrier->base) == IREE_NET_CARRIER_STATE_ACTIVE &&
      !iree_net_carrier_has_terminal_error(&carrier->base)) {
    next_slot = iree_net_stream_pop_send_locked(carrier);
    if (next_slot) {
      next_submit_status =
          iree_net_stream_submit_send_slot_locked(carrier, next_slot);
      if (!iree_status_is_ok(next_submit_status)) {
        next_slot->state = IREE_NET_STREAM_SEND_SLOT_STATE_COMPLETING;
      }
    }
  } else {
    detached_sends =
        iree_net_stream_detach_send_queue_locked(carrier, /*out_tail=*/NULL);
  }
  if (!next_slot) {
    carrier->flags &= ~IREE_NET_STREAM_CARRIER_FLAG_SEND_DISPATCH_ACTIVE;
  }
  const bool issue_write_shutdown =
      iree_net_stream_claim_write_shutdown_locked(carrier);
  iree_slim_mutex_unlock(&carrier->mutex);

  if (detached_sends) {
    iree_status_t terminal_status =
        iree_net_carrier_clone_terminal_error(&carrier->base);
    const iree_status_code_t status_code =
        iree_status_is_ok(terminal_status) ? IREE_STATUS_CANCELLED
                                           : iree_status_code(terminal_status);
    iree_status_free(terminal_status);
    iree_net_stream_complete_detached_sends(carrier, detached_sends,
                                            status_code);
  }
  if (next_slot && !iree_status_is_ok(next_submit_status)) {
    iree_net_stream_process_send_result(carrier, next_slot, next_submit_status,
                                        0, IREE_ASYNC_COMPLETION_FLAG_NONE);
  }
  if (issue_write_shutdown) {
    iree_net_stream_issue_deferred_write_shutdown(carrier);
  }
}

static void iree_net_stream_socket_receive_completed(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags);

static iree_status_t iree_net_stream_submit_receive_locked(
    iree_net_stream_carrier_t* carrier) {
  iree_async_socket_recv_pool_operation_t* operation =
      &carrier->receive_operation;
  iree_async_operation_zero(&operation->base, sizeof(*operation));
  iree_async_operation_initialize(
      &operation->base, IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_stream_socket_receive_completed,
      carrier);
  operation->socket = carrier->socket;
  operation->pool = carrier->receive_pool;
  return iree_async_proactor_submit_one(carrier->proactor, &operation->base);
}

static iree_status_t iree_net_stream_continue_receive_locked(
    iree_net_stream_carrier_t* carrier, bool* out_retire_receive) {
  *out_retire_receive = false;
  if (iree_net_carrier_state(&carrier->base) == IREE_NET_CARRIER_STATE_ACTIVE &&
      !iree_net_carrier_has_terminal_error(&carrier->base)) {
    carrier->receive_state = IREE_NET_STREAM_RECEIVE_STATE_SUBMITTED;
    iree_status_t status = iree_net_stream_submit_receive_locked(carrier);
    if (iree_status_is_ok(status)) {
      return status;
    }
    carrier->receive_state = IREE_NET_STREAM_RECEIVE_STATE_RETIRED;
    *out_retire_receive = true;
    return status;
  }
  carrier->receive_state = IREE_NET_STREAM_RECEIVE_STATE_RETIRED;
  *out_retire_receive = true;
  return iree_ok_status();
}

// Finishes a receive transition and must be the caller's final carrier access
// when |retire_receive| is true.
static void iree_net_stream_finish_receive_transition(
    iree_net_stream_carrier_t* carrier, iree_status_t status,
    bool retire_receive) {
  if (!iree_status_is_ok(status)) {
    iree_net_stream_publish_failure(carrier, status);
  }
  if (retire_receive) {
    iree_net_stream_carrier_retire_pending_operation(carrier);
  }
}

static void iree_net_stream_receive_lease_recycle(void* user_data,
                                                  uint32_t buffer_index) {
  iree_net_stream_receive_lease_context_t* context =
      (iree_net_stream_receive_lease_context_t*)user_data;
  iree_net_stream_carrier_t* carrier = context->carrier;
  const iree_async_buffer_recycle_callback_t recycle = context->recycle;
  const int32_t old_state = iree_atomic_exchange(
      &context->state, IREE_NET_STREAM_RECEIVE_LEASE_STATE_RELEASED,
      iree_memory_order_acq_rel);
  IREE_ASSERT(old_state != IREE_NET_STREAM_RECEIVE_LEASE_STATE_RELEASED,
              "stream receive lease recycled more than once");

  recycle.fn(recycle.user_data, buffer_index);
  if (old_state == IREE_NET_STREAM_RECEIVE_LEASE_STATE_RETAINED) {
    int32_t previous_count = iree_atomic_fetch_sub(
        &carrier->retained_receive_lease_count, 1, iree_memory_order_acq_rel);
    IREE_ASSERT(previous_count > 0,
                "stream carrier lost a retained receive lease");
    iree_net_carrier_release(&carrier->base);
  }
}

static iree_net_stream_receive_lease_context_t*
iree_net_stream_prepare_receive_lease(iree_net_stream_carrier_t* carrier,
                                      iree_async_buffer_lease_t* lease) {
  IREE_ASSERT(lease->release.fn,
              "pool-backed stream receive produced an unrecyclable lease");
  IREE_ASSERT(lease->buffer_index < carrier->receive_lease_context_count,
              "stream receive lease buffer index is out of range");
  iree_net_stream_receive_lease_context_t* context =
      &carrier->receive_lease_contexts[lease->buffer_index];
  IREE_ASSERT(
      iree_atomic_load(&context->state, iree_memory_order_acquire) ==
          IREE_NET_STREAM_RECEIVE_LEASE_STATE_RELEASED,
      "stream receive buffer was reused before its prior lease returned");
  context->carrier = carrier;
  context->recycle = lease->release;
  iree_atomic_store(&context->state,
                    IREE_NET_STREAM_RECEIVE_LEASE_STATE_PENDING,
                    iree_memory_order_release);
  lease->release.fn = iree_net_stream_receive_lease_recycle;
  lease->release.user_data = context;
  return context;
}

static void iree_net_stream_retain_moved_receive_lease(
    iree_net_stream_carrier_t* carrier,
    iree_net_stream_receive_lease_context_t* context,
    iree_async_buffer_lease_t* callback_lease) {
  if (callback_lease->release.fn) {
    return;
  }

  iree_net_carrier_retain(&carrier->base);
  iree_atomic_fetch_add(&carrier->retained_receive_lease_count, 1,
                        iree_memory_order_acq_rel);
  int32_t expected_state = IREE_NET_STREAM_RECEIVE_LEASE_STATE_PENDING;
  if (!iree_atomic_compare_exchange_strong(
          &context->state, &expected_state,
          IREE_NET_STREAM_RECEIVE_LEASE_STATE_RETAINED,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    int32_t previous_count = iree_atomic_fetch_sub(
        &carrier->retained_receive_lease_count, 1, iree_memory_order_acq_rel);
    IREE_ASSERT(previous_count > 0,
                "stream carrier lost a retained receive lease");
    iree_net_carrier_release(&carrier->base);
  }
}

static void iree_net_stream_retire_completing_receive(
    iree_net_stream_carrier_t* carrier) {
  iree_slim_mutex_lock(&carrier->mutex);
  IREE_ASSERT(
      carrier->receive_state == IREE_NET_STREAM_RECEIVE_STATE_COMPLETING,
      "stream receive retired from state %d", (int)carrier->receive_state);
  carrier->receive_state = IREE_NET_STREAM_RECEIVE_STATE_RETIRED;
  iree_slim_mutex_unlock(&carrier->mutex);
  iree_net_stream_carrier_retire_pending_operation(carrier);
}

static void iree_net_stream_socket_receive_completed(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_stream_carrier_t* carrier = (iree_net_stream_carrier_t*)user_data;
  iree_async_socket_recv_pool_operation_t* receive_operation =
      (iree_async_socket_recv_pool_operation_t*)operation;

  iree_slim_mutex_lock(&carrier->mutex);
  IREE_ASSERT(
      carrier->receive_state == IREE_NET_STREAM_RECEIVE_STATE_SUBMITTED ||
          carrier->receive_state == IREE_NET_STREAM_RECEIVE_STATE_CANCELLING,
      "stream receive completed from state %d", (int)carrier->receive_state);
  carrier->receive_state = IREE_NET_STREAM_RECEIVE_STATE_COMPLETING;
  iree_slim_mutex_unlock(&carrier->mutex);

  if (iree_net_carrier_state(&carrier->base) != IREE_NET_CARRIER_STATE_ACTIVE) {
    iree_status_free(status);
    iree_async_buffer_lease_release(&receive_operation->lease);
    iree_net_stream_retire_completing_receive(carrier);
    return;
  }

  if (iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_CANCELLED) &&
      iree_status_is_ok(status)) {
    status = iree_make_status(IREE_STATUS_CANCELLED,
                              "stream socket receive was cancelled");
  }
  if (!iree_status_is_ok(status)) {
    iree_async_buffer_lease_release(&receive_operation->lease);
    iree_net_stream_publish_failure(carrier, status);
    iree_net_stream_retire_completing_receive(carrier);
    return;
  }
  iree_status_free(status);

  if (receive_operation->bytes_received == 0) {
    iree_async_buffer_lease_release(&receive_operation->lease);
    iree_status_t handler_status = carrier->base.handlers.on_receive(
        carrier->base.handlers.user_data, iree_async_span_empty(), NULL);
    if (!iree_status_is_ok(handler_status)) {
      iree_net_stream_publish_failure(carrier, handler_status);
    }
    iree_net_stream_retire_completing_receive(carrier);
    return;
  }

  IREE_ASSERT(
      receive_operation->bytes_received <= receive_operation->lease.span.length,
      "stream receive exceeds its leased buffer");
  iree_async_span_t data = iree_async_span_make(
      receive_operation->lease.span.region,
      receive_operation->lease.span.offset, receive_operation->bytes_received);
  // Only this serialized callback can increase the retained count. Concurrent
  // returns can lower it, conservatively leaving this delivery borrowed. Keep
  // one native buffer available for later control/release-trigger messages;
  // framing already gives borrowed frames independent ownership when needed.
  const int32_t retained_lease_count = iree_atomic_load(
      &carrier->retained_receive_lease_count, iree_memory_order_acquire);
  iree_net_stream_receive_lease_context_t* lease_context = NULL;
  if ((iree_host_size_t)retained_lease_count <
      carrier->receive_lease_context_count - 1) {
    lease_context = iree_net_stream_prepare_receive_lease(
        carrier, &receive_operation->lease);
  }
  iree_status_t handler_status = carrier->base.handlers.on_receive(
      carrier->base.handlers.user_data, data,
      lease_context ? &receive_operation->lease : NULL);
  if (lease_context) {
    iree_net_stream_retain_moved_receive_lease(carrier, lease_context,
                                               &receive_operation->lease);
  }
  iree_async_buffer_lease_release(&receive_operation->lease);

  if (!iree_status_is_ok(handler_status)) {
    iree_net_stream_publish_failure(carrier, handler_status);
    iree_net_stream_retire_completing_receive(carrier);
    return;
  }

  iree_status_t submit_status = iree_ok_status();
  bool retire_receive = false;
  iree_slim_mutex_lock(&carrier->mutex);
  IREE_ASSERT(
      carrier->receive_state == IREE_NET_STREAM_RECEIVE_STATE_COMPLETING,
      "stream receive resumed from state %d", (int)carrier->receive_state);
  submit_status =
      iree_net_stream_continue_receive_locked(carrier, &retire_receive);
  iree_slim_mutex_unlock(&carrier->mutex);
  iree_net_stream_finish_receive_transition(carrier, submit_status,
                                            retire_receive);
}

static void iree_net_stream_carrier_destroy(iree_net_carrier_t* base_carrier) {
  iree_net_stream_carrier_t* carrier =
      iree_net_stream_carrier_cast(base_carrier);
  iree_allocator_t host_allocator = base_carrier->host_allocator;
  const iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  IREE_ASSERT(state == IREE_NET_CARRIER_STATE_CREATED ||
              state == IREE_NET_CARRIER_STATE_DEACTIVATED);
  IREE_ASSERT(iree_atomic_load(&base_carrier->pending_operations,
                               iree_memory_order_acquire) == 0);
  IREE_ASSERT(carrier->send_slots_in_use == 0);
  IREE_ASSERT(!carrier->send_queue_head);
  IREE_ASSERT(iree_atomic_load(&carrier->retained_receive_lease_count,
                               iree_memory_order_acquire) == 0);

  iree_async_socket_release(carrier->socket);
  iree_async_buffer_pool_release(carrier->receive_pool);
  iree_async_proactor_release(carrier->proactor);
  iree_slim_mutex_deinitialize(&carrier->mutex);
  iree_net_carrier_deinitialize(base_carrier);
  if (carrier->generated_prefix_slab) {
    iree_allocator_free_aligned(host_allocator, carrier);
  } else {
    iree_allocator_free(host_allocator, carrier);
  }
}

static iree_status_t iree_net_stream_carrier_activate(
    iree_net_carrier_t* base_carrier) {
  iree_net_stream_carrier_t* carrier =
      iree_net_stream_carrier_cast(base_carrier);
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&carrier->mutex);
  if (!iree_net_carrier_try_transition_state(base_carrier,
                                             IREE_NET_CARRIER_STATE_CREATED,
                                             IREE_NET_CARRIER_STATE_ACTIVE)) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "stream carrier is not in CREATED state");
  } else {
    carrier->receive_state = IREE_NET_STREAM_RECEIVE_STATE_SUBMITTED;
    iree_atomic_fetch_add(&base_carrier->pending_operations, 1,
                          iree_memory_order_acq_rel);
    status = iree_net_stream_submit_receive_locked(carrier);
  }
  if (!iree_status_is_ok(status)) {
    if (carrier->receive_state == IREE_NET_STREAM_RECEIVE_STATE_SUBMITTED) {
      carrier->receive_state = IREE_NET_STREAM_RECEIVE_STATE_RETIRED;
      bool was_final = iree_net_carrier_retire_pending_operation(base_carrier);
      IREE_ASSERT(was_final,
                  "failed stream activation left pending operations");
      iree_net_carrier_set_state(base_carrier, IREE_NET_CARRIER_STATE_CREATED);
    }
  }
  iree_slim_mutex_unlock(&carrier->mutex);
  return status;
}

static void iree_net_stream_carrier_deactivate(
    iree_net_carrier_t* base_carrier,
    iree_net_carrier_deactivate_callback_fn_t callback, void* user_data) {
  iree_net_stream_carrier_t* carrier =
      iree_net_stream_carrier_cast(base_carrier);
  iree_status_t cleanup_status = iree_ok_status();
  bool shutdown_socket = false;
  bool valid_request = false;

  iree_slim_mutex_lock(&carrier->mutex);
  const iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state == IREE_NET_CARRIER_STATE_CREATED ||
      state == IREE_NET_CARRIER_STATE_ACTIVE) {
    valid_request = true;
    shutdown_socket = state == IREE_NET_CARRIER_STATE_ACTIVE;
    iree_atomic_fetch_add(&base_carrier->pending_operations, 1,
                          iree_memory_order_acq_rel);
    carrier->deactivate_callback.fn = callback;
    carrier->deactivate_callback.user_data = user_data;
    carrier->flags |= IREE_NET_STREAM_CARRIER_FLAG_SEND_SHUTDOWN_INITIATED |
                      IREE_NET_STREAM_CARRIER_FLAG_SOCKET_WRITE_SHUTDOWN_ISSUED;
    iree_net_carrier_set_state(base_carrier, IREE_NET_CARRIER_STATE_DRAINING);
    cleanup_status =
        iree_net_stream_request_receive_cancellation_locked(carrier);
    cleanup_status = iree_status_join(
        cleanup_status,
        iree_net_stream_request_send_cancellation_locked(carrier));
  }
  iree_slim_mutex_unlock(&carrier->mutex);

  IREE_ASSERT(valid_request, "stream carrier deactivated more than once");
  if (!valid_request) {
    return;
  }

  if (shutdown_socket) {
    cleanup_status = iree_status_join(
        cleanup_status, iree_async_socket_shutdown(
                            carrier->socket, IREE_ASYNC_SOCKET_SHUTDOWN_BOTH));
  }
  if (!iree_status_is_ok(cleanup_status)) {
    iree_net_carrier_report_terminal_error(base_carrier, cleanup_status);
  }
  iree_net_stream_carrier_retire_pending_operation(carrier);
}

static iree_net_carrier_send_budget_t iree_net_stream_carrier_query_send_budget(
    iree_net_carrier_t* base_carrier) {
  iree_net_stream_carrier_t* carrier =
      iree_net_stream_carrier_cast(base_carrier);
  iree_net_carrier_send_budget_t budget = {0};
  iree_slim_mutex_lock(&carrier->mutex);
  if (iree_net_carrier_state(base_carrier) == IREE_NET_CARRIER_STATE_ACTIVE &&
      !iree_any_bit_set(carrier->flags,
                        IREE_NET_STREAM_CARRIER_FLAG_SEND_SHUTDOWN_INITIATED) &&
      !iree_net_carrier_has_terminal_error(base_carrier)) {
    budget.bytes = IREE_HOST_SIZE_MAX;
    budget.slots = carrier->send_slot_count - carrier->send_slots_in_use;
  }
  iree_slim_mutex_unlock(&carrier->mutex);
  return budget;
}

static iree_status_t iree_net_stream_check_send_admission_locked(
    iree_net_stream_carrier_t* carrier,
    iree_net_stream_send_slot_t** out_slot) {
  *out_slot = NULL;
  if (iree_net_carrier_state(&carrier->base) != IREE_NET_CARRIER_STATE_ACTIVE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream carrier is not active");
  }
  IREE_RETURN_IF_ERROR(iree_net_carrier_clone_terminal_error(&carrier->base));
  if (iree_any_bit_set(carrier->flags,
                       IREE_NET_STREAM_CARRIER_FLAG_SEND_SHUTDOWN_INITIATED)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream carrier send direction is shut down");
  }
  iree_net_stream_send_slot_t* slot =
      iree_net_stream_find_free_send_slot_locked(carrier);
  if (!slot) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "stream send operation slots are exhausted");
  }
  *out_slot = slot;
  return iree_ok_status();
}

static iree_status_t iree_net_stream_check_send_publication_locked(
    iree_net_stream_carrier_t* carrier) {
  if (iree_net_carrier_state(&carrier->base) != IREE_NET_CARRIER_STATE_ACTIVE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream carrier deactivated during send");
  }
  IREE_RETURN_IF_ERROR(iree_net_carrier_clone_terminal_error(&carrier->base));
  if (iree_any_bit_set(carrier->flags,
                       IREE_NET_STREAM_CARRIER_FLAG_SEND_SHUTDOWN_INITIATED)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream carrier send direction is shut down");
  }
  return iree_ok_status();
}

static void iree_net_stream_reject_preparing_send(
    iree_net_stream_carrier_t* carrier, iree_net_stream_send_slot_t* slot) {
  iree_net_stream_send_slot_release_resources(carrier, slot);
  bool issue_write_shutdown = false;
  iree_slim_mutex_lock(&carrier->mutex);
  IREE_ASSERT(slot->state == IREE_NET_STREAM_SEND_SLOT_STATE_PREPARING);
  iree_net_stream_recycle_send_slot_locked(carrier, slot);
  issue_write_shutdown = iree_net_stream_claim_write_shutdown_locked(carrier);
  iree_slim_mutex_unlock(&carrier->mutex);
  if (issue_write_shutdown) {
    iree_net_stream_issue_deferred_write_shutdown(carrier);
  }
  iree_net_stream_carrier_retire_pending_operation(carrier);
}

static iree_status_t iree_net_stream_carrier_send(
    iree_net_carrier_t* base_carrier, const iree_net_send_params_t* params) {
  iree_net_stream_carrier_t* carrier =
      iree_net_stream_carrier_cast(base_carrier);

  iree_host_size_t total_length = params->generated_prefix.length;
  for (iree_host_size_t i = 0; i < params->data.count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_net_stream_validate_send_span(params->data.values[i]));
    if (!iree_host_size_checked_add(total_length, params->data.values[i].length,
                                    &total_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "stream send payload length overflow");
    }
  }

  const bool has_generated_prefix = params->generated_prefix.length > 0;
  void* owned_buffer = NULL;
  if (params->generated_prefix.length > carrier->generated_prefix_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
        base_carrier->host_allocator, params->generated_prefix.length,
        &owned_buffer));
  }

  iree_net_stream_send_slot_t* slot = NULL;
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&carrier->mutex);
  status = iree_net_stream_check_send_admission_locked(carrier, &slot);
  if (iree_status_is_ok(status)) {
    slot->completion_callback = params->completion_callback;
    slot->owned_buffer = owned_buffer;
    owned_buffer = NULL;
    slot->total_length = total_length;
    slot->bytes_transferred = 0;
    slot->submitted_length = 0;
    slot->span_count = 0;
    slot->first_span = 0;
    slot->state = IREE_NET_STREAM_SEND_SLOT_STATE_PREPARING;
    ++carrier->send_slots_in_use;
    iree_atomic_fetch_add(&base_carrier->pending_operations, 1,
                          iree_memory_order_acq_rel);
  }
  iree_slim_mutex_unlock(&carrier->mutex);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(base_carrier->host_allocator, owned_buffer);
    return status;
  }

  iree_byte_span_t prefix_storage = iree_byte_span_empty();
  if (has_generated_prefix) {
    uint8_t* prefix_data =
        slot->owned_buffer
            ? (uint8_t*)slot->owned_buffer
            : iree_net_stream_send_slot_prefix_storage(carrier, slot);
    prefix_storage =
        iree_make_byte_span(prefix_data, params->generated_prefix.length);
    slot->spans[slot->span_count++] =
        iree_async_span_from_ptr(prefix_data, params->generated_prefix.length);
  }
  for (iree_host_size_t i = 0; i < params->data.count; ++i) {
    if (params->data.values[i].length == 0) {
      continue;
    }
    slot->spans[slot->span_count++] = params->data.values[i];
  }

  if (has_generated_prefix) {
    status = params->generated_prefix.write(params->generated_prefix.user_data,
                                            prefix_storage);
  }
  if (iree_status_is_ok(status)) {
    iree_async_span_list_retain_regions(
        iree_async_span_list_make(slot->spans, slot->span_count));
    slot->regions_retained = true;
  }

  iree_net_stream_send_slot_t* rejected_slot = NULL;
  iree_slim_mutex_lock(&carrier->mutex);
  if (iree_status_is_ok(status)) {
    status = iree_net_stream_check_send_publication_locked(carrier);
  }
  if (iree_status_is_ok(status)) {
    slot->state = IREE_NET_STREAM_SEND_SLOT_STATE_QUEUED;
    iree_net_stream_enqueue_send_locked(carrier, slot);
    iree_net_stream_send_slot_t* failed_slot = NULL;
    iree_status_t submit_status =
        iree_net_stream_start_send_dispatch_locked(carrier, &failed_slot);
    if (failed_slot) {
      status = iree_net_stream_schedule_local_send_completion_locked(
          carrier, failed_slot, submit_status,
          iree_net_stream_initial_send_submit_completed);
      if (!iree_status_is_ok(status)) {
        IREE_ASSERT(failed_slot == slot,
                    "initial stream dispatch failure did not own current send");
        IREE_ASSERT(!carrier->send_queue_head,
                    "initial stream dispatch failure left queued sends");
        carrier->flags &= ~IREE_NET_STREAM_CARRIER_FLAG_SEND_DISPATCH_ACTIVE;
        failed_slot->state = IREE_NET_STREAM_SEND_SLOT_STATE_PREPARING;
        rejected_slot = failed_slot;
      }
    }
  } else {
    slot->state = IREE_NET_STREAM_SEND_SLOT_STATE_LOCAL_COMPLETION;
    status = iree_net_stream_schedule_local_send_completion_locked(
        carrier, slot, status, iree_net_stream_local_send_completed);
    if (!iree_status_is_ok(status)) {
      slot->state = IREE_NET_STREAM_SEND_SLOT_STATE_PREPARING;
      rejected_slot = slot;
    }
  }
  iree_slim_mutex_unlock(&carrier->mutex);

  if (rejected_slot) {
    iree_net_stream_reject_preparing_send(carrier, rejected_slot);
  }
  return status;
}

static iree_status_t iree_net_stream_carrier_shutdown(
    iree_net_carrier_t* base_carrier) {
  iree_net_stream_carrier_t* carrier =
      iree_net_stream_carrier_cast(base_carrier);
  bool issue_write_shutdown = false;
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&carrier->mutex);
  if (iree_net_carrier_state(base_carrier) != IREE_NET_CARRIER_STATE_ACTIVE) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "stream carrier is not active");
  } else if (iree_any_bit_set(
                 carrier->flags,
                 IREE_NET_STREAM_CARRIER_FLAG_SEND_SHUTDOWN_INITIATED)) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "stream carrier send direction is already shut down");
  } else {
    carrier->flags |= IREE_NET_STREAM_CARRIER_FLAG_SEND_SHUTDOWN_INITIATED;
    issue_write_shutdown = iree_net_stream_claim_write_shutdown_locked(carrier);
  }
  iree_slim_mutex_unlock(&carrier->mutex);

  if (issue_write_shutdown) {
    status = iree_async_socket_shutdown(carrier->socket,
                                        IREE_ASYNC_SOCKET_SHUTDOWN_WRITE);
    if (!iree_status_is_ok(status)) {
      iree_status_t caller_status = iree_status_clone(status);
      iree_net_stream_fail_without_operation(carrier, status);
      status = caller_status;
    }
  }
  return status;
}

static const iree_net_carrier_vtable_t iree_net_stream_carrier_vtable = {
    .destroy = iree_net_stream_carrier_destroy,
    .activate = iree_net_stream_carrier_activate,
    .deactivate = iree_net_stream_carrier_deactivate,
    .query_send_budget = iree_net_stream_carrier_query_send_budget,
    .send = iree_net_stream_carrier_send,
    .shutdown = iree_net_stream_carrier_shutdown,
};

IREE_API_EXPORT iree_status_t iree_net_stream_carrier_create(
    iree_async_proactor_t* proactor, iree_async_socket_t* socket,
    iree_async_buffer_pool_t* receive_pool,
    const iree_net_stream_carrier_options_t* options,
    iree_allocator_t host_allocator, iree_net_carrier_t** out_carrier) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(socket);
  IREE_ASSERT_ARGUMENT(receive_pool);
  IREE_ASSERT_ARGUMENT(out_carrier);
  *out_carrier = NULL;

  iree_net_stream_carrier_options_t default_options =
      iree_net_stream_carrier_options_default();
  if (!options) {
    options = &default_options;
  }
  if (options->max_send_operations == 0 ||
      options->max_send_operations == UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "stream max send operations must be in [1, UINT32_MAX)");
  }
  if (socket->proactor != proactor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "stream socket belongs to a different proactor");
  }
  if (socket->type != IREE_ASYNC_SOCKET_TYPE_TCP &&
      socket->type != IREE_ASYNC_SOCKET_TYPE_TCP6 &&
      socket->type != IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "stream carrier requires TCP, TCP6, or UNIX_STREAM");
  }
  if (iree_async_socket_query_state(socket) !=
      IREE_ASYNC_SOCKET_STATE_CONNECTED) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream carrier requires a connected socket");
  }

  iree_async_region_t* receive_region =
      iree_async_buffer_pool_region(receive_pool);
  if (receive_region->proactor != proactor) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "stream receive pool was registered with a different proactor");
  }
  if (!iree_any_bit_set(receive_region->access_flags,
                        IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream receive pool does not permit writes");
  }
  const iree_host_size_t receive_buffer_count =
      iree_async_buffer_pool_capacity(receive_pool);
  if (receive_buffer_count == 0 || receive_buffer_count > INT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "stream receive pool capacity must be in [1, INT32_MAX]");
  }

  iree_host_size_t generated_prefix_stride = 0;
  iree_host_size_t generated_prefix_slab_size = 0;
  if (options->generated_prefix_capacity > 0 &&
      (!iree_host_size_checked_align(
           options->generated_prefix_capacity,
           iree_hardware_destructive_interference_size,
           &generated_prefix_stride) ||
       !iree_host_size_checked_mul(options->max_send_operations,
                                   generated_prefix_stride,
                                   &generated_prefix_slab_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "stream generated-prefix slab size overflow");
  }

  iree_host_size_t total_size = 0;
  iree_host_size_t send_slots_offset = 0;
  iree_host_size_t receive_contexts_offset = 0;
  iree_host_size_t generated_prefix_slab_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_net_stream_carrier_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(
          options->max_send_operations, iree_net_stream_send_slot_t,
          iree_alignof(iree_net_stream_send_slot_t), &send_slots_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          receive_buffer_count, iree_net_stream_receive_lease_context_t,
          iree_alignof(iree_net_stream_receive_lease_context_t),
          &receive_contexts_offset),
      IREE_STRUCT_FIELD(generated_prefix_slab_size, uint8_t,
                        &generated_prefix_slab_offset)));

  iree_net_stream_carrier_t* carrier = NULL;
  if (generated_prefix_slab_size > 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
        host_allocator, total_size, iree_hardware_destructive_interference_size,
        generated_prefix_slab_offset, (void**)&carrier));
  } else {
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(host_allocator, total_size, (void**)&carrier));
  }
  memset(carrier, 0, total_size);
  iree_slim_mutex_initialize(&carrier->mutex);
  carrier->proactor = proactor;
  iree_async_proactor_retain(proactor);
  carrier->socket = socket;
  iree_async_socket_retain(socket);
  carrier->receive_pool = receive_pool;
  iree_async_buffer_pool_retain(receive_pool);
  carrier->send_slots =
      (iree_net_stream_send_slot_t*)((uint8_t*)carrier + send_slots_offset);
  carrier->send_slot_count = options->max_send_operations;
  carrier->generated_prefix_capacity = options->generated_prefix_capacity;
  carrier->zero_copy_min_send_size = options->zero_copy_min_send_size;
  carrier->generated_prefix_stride = generated_prefix_stride;
  if (generated_prefix_slab_size > 0) {
    carrier->generated_prefix_slab =
        (uint8_t*)carrier + generated_prefix_slab_offset;
  }
  carrier->receive_lease_contexts =
      (iree_net_stream_receive_lease_context_t*)((uint8_t*)carrier +
                                                 receive_contexts_offset);
  carrier->receive_lease_context_count = receive_buffer_count;
  carrier->receive_state = IREE_NET_STREAM_RECEIVE_STATE_RETIRED;
  iree_atomic_store(&carrier->retained_receive_lease_count, 0,
                    iree_memory_order_relaxed);
  for (iree_host_size_t i = 0; i < receive_buffer_count; ++i) {
    iree_atomic_store(&carrier->receive_lease_contexts[i].state,
                      IREE_NET_STREAM_RECEIVE_LEASE_STATE_RELEASED,
                      iree_memory_order_relaxed);
  }

  iree_net_carrier_capabilities_t capabilities =
      IREE_NET_CARRIER_CAPABILITY_RELIABLE |
      IREE_NET_CARRIER_CAPABILITY_ORDERED |
      IREE_NET_CARRIER_CAPABILITY_ZERO_COPY_RX;
  const iree_async_proactor_capabilities_t proactor_capabilities =
      iree_async_proactor_query_capabilities(proactor);
  if (options->zero_copy_min_send_size != IREE_HOST_SIZE_MAX &&
      iree_any_bit_set(socket->flags, IREE_ASYNC_SOCKET_FLAG_ZERO_COPY) &&
      iree_any_bit_set(proactor_capabilities,
                       IREE_ASYNC_PROACTOR_CAPABILITY_ZERO_COPY_SEND)) {
    capabilities |= IREE_NET_CARRIER_CAPABILITY_ZERO_COPY_TX;
  }
  iree_net_carrier_initialize(&iree_net_stream_carrier_vtable, capabilities,
                              IREE_ASYNC_SOCKET_SEND_MAX_BUFFERS,
                              host_allocator, &carrier->base);
  *out_carrier = &carrier->base;
  return iree_ok_status();
}
