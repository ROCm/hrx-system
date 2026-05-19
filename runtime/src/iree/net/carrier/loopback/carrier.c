// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/loopback/carrier.h"

#include "iree/async/operations/scheduling.h"
#include "iree/base/threading/mutex.h"

static_assert(sizeof(uintptr_t) <= sizeof(iree_net_carrier_send_handle_t),
              "send handle cannot round-trip pending send pointers");

typedef struct iree_net_loopback_file_transfer_payload_t {
  // Opaque transfer ID resolved by the peer carrier.
  uint64_t id;
} iree_net_loopback_file_transfer_payload_t;
static_assert(sizeof(iree_net_loopback_file_transfer_payload_t) == 8, "");

typedef struct iree_net_loopback_file_transfer_t {
  // Next transfer in the carrier-local pending transfer list.
  struct iree_net_loopback_file_transfer_t* next;

  // Opaque transfer ID serialized in the control payload.
  uint64_t id;

  // Retained file handle transferred to the peer on import.
  iree_io_file_handle_t* file_handle;
} iree_net_loopback_file_transfer_t;

// A pending send operation awaiting delivery during the next poll() cycle. The
// trailing storage contains the payload copied from send() or written by
// begin_send() before commit_send().
typedef struct iree_net_loopback_pending_send_t {
  // Next committed send in the carrier-local FIFO queue.
  struct iree_net_loopback_pending_send_t* next;

  // Carrier that owns the send. Retained until the send completes or aborts.
  struct iree_net_loopback_carrier_t* carrier;

  // Data to deliver to the peer's recv handler when the drain NOP completes.
  // Points into trailing storage.
  iree_async_span_t delivery_span;

  // Total byte count for statistics and completion callback.
  iree_host_size_t total_size;

  // User data from iree_net_send_params_t, echoed to send completion callback.
  uint64_t user_data;

  // True when completion should invoke the carrier callback.
  bool notify_completion;

  // Payload bytes delivered to the peer.
  iree_alignas(IREE_NET_MESSAGE_ALIGNMENT) uint8_t storage[];
} iree_net_loopback_pending_send_t;
static_assert(offsetof(iree_net_loopback_pending_send_t, storage) %
                      IREE_NET_MESSAGE_ALIGNMENT ==
                  0,
              "loopback message storage must satisfy carrier alignment");

typedef struct iree_net_loopback_carrier_t {
  // Base carrier (must be first for safe upcasting).
  iree_net_carrier_t base;

  // Proactor for async delivery via NOP operations. Retained.
  iree_async_proactor_t* proactor;

  // Guards send_queue_head, send_queue_tail, and send_drain_scheduled.
  iree_slim_mutex_t send_queue_mutex;

  // NOP operation submitted while queued sends need poll-thread delivery.
  iree_async_nop_operation_t send_drain_nop;

  // Head of the FIFO committed-send queue awaiting delivery.
  iree_net_loopback_pending_send_t* send_queue_head;

  // Tail of the FIFO committed-send queue awaiting delivery.
  iree_net_loopback_pending_send_t* send_queue_tail;

  // True while send_drain_nop is submitted or its callback is draining sends.
  // The carrier is retained for this interval.
  bool send_drain_scheduled;

  // Peer carrier (the other end of the pair). Not retained.
  // Set at pair creation, cleared on deactivate or peer destruction.
  struct iree_net_loopback_carrier_t* peer;

  // True if shutdown() was called (future sends fail).
  bool shutdown_initiated;

  // Number of committed or reserved sends awaiting completion/abort.
  iree_atomic_int32_t sends_in_flight;

  // Deactivation callback state stored during deactivate.
  struct {
    // Function invoked when all pending operations drain.
    iree_net_carrier_deactivate_callback_fn_t fn;

    // User data passed to fn when deactivation completes.
    void* user_data;
  } deactivate_callback;

  // Handler invoked when the peer carrier disconnects (deactivates or is
  // destroyed). Set by the endpoint adapter to propagate transport errors.
  // This provides the loopback equivalent of TCP's ECONNRESET notification.
  iree_net_loopback_carrier_disconnect_handler_t peer_disconnect_handler;

  // Guards pending_file_transfers and next_file_transfer_id.
  iree_slim_mutex_t file_transfer_mutex;

  // Next opaque file transfer ID to allocate.
  uint64_t next_file_transfer_id;

  // Pending file handles exported by this carrier and not yet imported.
  iree_net_loopback_file_transfer_t* pending_file_transfers;
} iree_net_loopback_carrier_t;

static inline iree_net_loopback_carrier_t* iree_net_loopback_carrier_cast(
    iree_net_carrier_t* base_carrier) {
  return (iree_net_loopback_carrier_t*)base_carrier;
}

static iree_net_file_handle_transfer_type_t
iree_net_loopback_file_transfer_type(void) {
#if defined(IREE_PLATFORM_WINDOWS)
  return IREE_NET_FILE_HANDLE_TRANSFER_TYPE_WIN32_HANDLE;
#else
  return IREE_NET_FILE_HANDLE_TRANSFER_TYPE_POSIX_FD;
#endif  // IREE_PLATFORM_WINDOWS
}

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

static void iree_net_loopback_carrier_send_drain_completion(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags);

// Checks if deactivation has completed and invokes callback if so.
// Called after every pending_operations decrement.
static void iree_net_loopback_carrier_maybe_complete_deactivation(
    iree_net_loopback_carrier_t* carrier) {
  iree_net_carrier_state_t state = iree_net_carrier_state(&carrier->base);
  if (state != IREE_NET_CARRIER_STATE_DRAINING) return;

  int32_t pending = iree_atomic_load(&carrier->base.pending_operations,
                                     iree_memory_order_acquire);
  if (pending > 0) return;

  iree_net_carrier_set_state(&carrier->base,
                             IREE_NET_CARRIER_STATE_DEACTIVATED);
  if (carrier->deactivate_callback.fn) {
    carrier->deactivate_callback.fn(carrier->deactivate_callback.user_data);
  }
}

static void iree_net_loopback_carrier_release_file_transfers(
    iree_net_loopback_carrier_t* carrier) {
  iree_slim_mutex_lock(&carrier->file_transfer_mutex);
  iree_net_loopback_file_transfer_t* transfer = carrier->pending_file_transfers;
  carrier->pending_file_transfers = NULL;
  iree_slim_mutex_unlock(&carrier->file_transfer_mutex);

  while (transfer) {
    iree_net_loopback_file_transfer_t* next = transfer->next;
    iree_io_file_handle_release(transfer->file_handle);
    iree_allocator_free(carrier->base.host_allocator, transfer);
    transfer = next;
  }
}

static iree_net_loopback_file_transfer_t*
iree_net_loopback_carrier_take_file_transfer(
    iree_net_loopback_carrier_t* carrier, uint64_t id) {
  iree_slim_mutex_lock(&carrier->file_transfer_mutex);
  iree_net_loopback_file_transfer_t** previous_next =
      &carrier->pending_file_transfers;
  iree_net_loopback_file_transfer_t* transfer = carrier->pending_file_transfers;
  while (transfer && transfer->id != id) {
    previous_next = &transfer->next;
    transfer = transfer->next;
  }
  if (transfer) {
    *previous_next = transfer->next;
    transfer->next = NULL;
  }
  iree_slim_mutex_unlock(&carrier->file_transfer_mutex);
  return transfer;
}

// Deferred notification delivered to the surviving peer when the other side
// of a loopback pair deactivates or is destroyed. The NOP fires on the next
// proactor poll cycle, invoking the peer's disconnect handler.
typedef struct iree_net_loopback_disconnect_notify_t {
  iree_async_nop_operation_t nop;
  iree_net_loopback_carrier_t* peer;  // Retained.
} iree_net_loopback_disconnect_notify_t;

static void iree_net_loopback_carrier_disconnect_notify_completion(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)user_data;
  (void)flags;
  iree_status_ignore(status);
  iree_net_loopback_disconnect_notify_t* notify =
      (iree_net_loopback_disconnect_notify_t*)operation;
  iree_net_loopback_carrier_t* peer = notify->peer;
  iree_allocator_free(peer->base.host_allocator, notify);

  // Fire the handler if the peer hasn't already been deactivated.
  iree_net_carrier_state_t state = iree_net_carrier_state(&peer->base);
  if (state != IREE_NET_CARRIER_STATE_DEACTIVATED &&
      peer->peer_disconnect_handler.fn) {
    peer->peer_disconnect_handler.fn(
        peer->peer_disconnect_handler.user_data,
        iree_make_status(IREE_STATUS_UNAVAILABLE, "peer disconnected"));
  }

  iree_net_carrier_release(&peer->base);
}

// Notifies the surviving peer that this carrier is departing. Submits a NOP
// to deliver the notification asynchronously. Must be called BEFORE clearing
// the peer link (carrier->peer).
static void iree_net_loopback_carrier_notify_peer_disconnect(
    iree_net_loopback_carrier_t* carrier) {
  iree_net_loopback_carrier_t* peer = carrier->peer;
  if (!peer || !peer->peer_disconnect_handler.fn) return;

  // Don't notify if peer is already shutting down.
  iree_net_carrier_state_t peer_state = iree_net_carrier_state(&peer->base);
  if (peer_state == IREE_NET_CARRIER_STATE_DEACTIVATED) return;

  iree_net_loopback_disconnect_notify_t* notify = NULL;
  iree_status_t status = iree_allocator_malloc(
      peer->base.host_allocator, sizeof(*notify), (void**)&notify);
  if (iree_status_is_ok(status)) {
    memset(notify, 0, sizeof(*notify));
    notify->peer = peer;
    iree_net_carrier_retain(&peer->base);
    iree_async_operation_initialize(
        &notify->nop.base, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE,
        iree_net_loopback_carrier_disconnect_notify_completion, notify);
    status = iree_async_proactor_submit_one(peer->proactor, &notify->nop.base);
    if (!iree_status_is_ok(status)) {
      iree_net_carrier_release(&peer->base);
      iree_allocator_free(peer->base.host_allocator, notify);
    }
  }
  if (!iree_status_is_ok(status)) {
    // Synchronous fallback on OOM or submit failure. Safe because the handler
    // operates on the surviving peer, not the carrier being torn down.
    iree_status_ignore(status);
    peer->peer_disconnect_handler.fn(
        peer->peer_disconnect_handler.user_data,
        iree_make_status(IREE_STATUS_UNAVAILABLE, "peer disconnected"));
  }
}

static iree_status_t iree_net_loopback_pending_send_allocate(
    iree_net_loopback_carrier_t* carrier, iree_host_size_t total_size,
    iree_net_loopback_pending_send_t** out_pending_send) {
  *out_pending_send = NULL;

  iree_host_size_t allocation_size = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_net_loopback_pending_send_t), &allocation_size,
      IREE_STRUCT_FIELD_FAM(total_size, uint8_t));

  iree_net_loopback_pending_send_t* pending_send = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(carrier->base.host_allocator,
                                   allocation_size, (void**)&pending_send);
  }
  if (iree_status_is_ok(status)) {
    memset(pending_send, 0, allocation_size);
    pending_send->carrier = carrier;
    iree_net_carrier_retain(&carrier->base);
    pending_send->delivery_span =
        iree_async_span_from_ptr(pending_send->storage, total_size);
    pending_send->total_size = total_size;
    *out_pending_send = pending_send;
  }
  return status;
}

static void iree_net_loopback_pending_send_free(
    iree_net_loopback_pending_send_t* pending_send) {
  if (!pending_send) return;
  iree_net_loopback_carrier_t* carrier = pending_send->carrier;
  iree_allocator_t host_allocator = carrier->base.host_allocator;
  iree_allocator_free(host_allocator, pending_send);
  iree_net_carrier_release(&carrier->base);
}

static iree_status_t iree_net_loopback_carrier_begin_send_operation(
    iree_net_loopback_carrier_t* carrier) {
  iree_net_carrier_t* base_carrier = &carrier->base;
  iree_atomic_fetch_add(&base_carrier->pending_operations, 1,
                        iree_memory_order_acq_rel);

  iree_status_t status = iree_ok_status();
  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "carrier must be in ACTIVE state to send");
  }
  if (iree_status_is_ok(status) && carrier->shutdown_initiated) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "carrier has been shut down for sending");
  }
  if (iree_status_is_ok(status) && !carrier->peer) {
    status = iree_make_status(IREE_STATUS_UNAVAILABLE, "peer disconnected");
  }
  if (iree_status_is_ok(status)) {
    iree_atomic_fetch_add(&carrier->sends_in_flight, 1,
                          iree_memory_order_acq_rel);
  }

  if (!iree_status_is_ok(status)) {
    iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                          iree_memory_order_release);
    iree_net_loopback_carrier_maybe_complete_deactivation(carrier);
  }
  return status;
}

static void iree_net_loopback_carrier_end_send_operation(
    iree_net_loopback_carrier_t* carrier) {
  iree_atomic_fetch_sub(&carrier->sends_in_flight, 1,
                        iree_memory_order_release);
  iree_atomic_fetch_sub(&carrier->base.pending_operations, 1,
                        iree_memory_order_release);
  iree_net_loopback_carrier_maybe_complete_deactivation(carrier);
}

static iree_status_t iree_net_loopback_pending_send_enqueue(
    iree_net_loopback_pending_send_t* pending_send) {
  iree_net_loopback_carrier_t* carrier = pending_send->carrier;
  pending_send->next = NULL;

  iree_slim_mutex_lock(&carrier->send_queue_mutex);

  iree_status_t status = iree_ok_status();
  if (!carrier->send_drain_scheduled) {
    iree_async_operation_zero(&carrier->send_drain_nop.base,
                              sizeof(carrier->send_drain_nop));
    iree_async_operation_initialize(
        &carrier->send_drain_nop.base, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE,
        iree_net_loopback_carrier_send_drain_completion, carrier);
    iree_net_carrier_retain(&carrier->base);
    status = iree_async_proactor_submit_one(carrier->proactor,
                                            &carrier->send_drain_nop.base);
    if (iree_status_is_ok(status)) {
      carrier->send_drain_scheduled = true;
    } else {
      iree_net_carrier_release(&carrier->base);
    }
  }

  if (iree_status_is_ok(status)) {
    if (carrier->send_queue_tail) {
      carrier->send_queue_tail->next = pending_send;
    } else {
      carrier->send_queue_head = pending_send;
    }
    carrier->send_queue_tail = pending_send;
  }

  iree_slim_mutex_unlock(&carrier->send_queue_mutex);
  return status;
}

static iree_net_loopback_pending_send_t* iree_net_loopback_pending_send_pop(
    iree_net_loopback_carrier_t* carrier) {
  iree_slim_mutex_lock(&carrier->send_queue_mutex);
  iree_net_loopback_pending_send_t* pending_send = carrier->send_queue_head;
  if (pending_send) {
    carrier->send_queue_head = pending_send->next;
    if (!carrier->send_queue_head) {
      carrier->send_queue_tail = NULL;
    }
    pending_send->next = NULL;
  } else {
    carrier->send_drain_scheduled = false;
  }
  iree_slim_mutex_unlock(&carrier->send_queue_mutex);
  return pending_send;
}

static void iree_net_loopback_pending_send_deliver(
    iree_net_loopback_pending_send_t* pending_send) {
  iree_net_loopback_carrier_t* carrier = pending_send->carrier;
  // Deliver data to peer's recv handler if peer is still alive and active.
  // If the peer departed between send() and this completion (deactivated or
  // destroyed while the send was queued), report an error through the sender's
  // completion callback. This mirrors TCP/SHM carriers where the OS reports
  // EPIPE/ECONNRESET when the peer closes the connection.
  //
  // The recv handler may trigger peer destruction (e.g., GOAWAY processing
  // destroys the peer's session, which deactivates and releases the peer
  // carrier). Retain the peer so the statistics update below doesn't access
  // freed memory.
  iree_status_t delivery_status = iree_ok_status();
  iree_net_loopback_carrier_t* peer = carrier->peer;
  if (peer) iree_net_carrier_retain(&peer->base);
  if (peer &&
      iree_net_carrier_state(&peer->base) == IREE_NET_CARRIER_STATE_ACTIVE &&
      peer->base.recv_handler.fn) {
    delivery_status = peer->base.recv_handler.fn(
        peer->base.recv_handler.user_data, pending_send->delivery_span, NULL);
  } else {
    delivery_status =
        iree_make_status(IREE_STATUS_UNAVAILABLE, "peer disconnected");
  }

  // Update statistics on successful delivery.
  if (iree_status_is_ok(delivery_status)) {
    iree_atomic_fetch_add(&carrier->base.bytes_sent,
                          (int64_t)pending_send->total_size,
                          iree_memory_order_relaxed);
    if (peer) {
      iree_atomic_fetch_add(&peer->base.bytes_received,
                            (int64_t)pending_send->total_size,
                            iree_memory_order_relaxed);
    }
  }

  // Release peer ref acquired above.
  if (peer) iree_net_carrier_release(&peer->base);

  // Fire sender's send completion callback if set.
  if (pending_send->notify_completion && carrier->base.callback.fn) {
    carrier->base.callback.fn(carrier->base.callback.user_data,
                              pending_send->user_data, delivery_status,
                              pending_send->total_size, NULL);
  } else {
    iree_status_ignore(delivery_status);
  }

  iree_net_loopback_carrier_end_send_operation(carrier);
  iree_net_loopback_pending_send_free(pending_send);
}

static void iree_net_loopback_pending_send_fail(
    iree_net_loopback_pending_send_t* pending_send, iree_status_t status) {
  iree_net_loopback_carrier_t* carrier = pending_send->carrier;
  iree_status_t completion_status = iree_status_clone(status);
  if (pending_send->notify_completion && carrier->base.callback.fn) {
    carrier->base.callback.fn(carrier->base.callback.user_data,
                              pending_send->user_data, completion_status,
                              pending_send->total_size, NULL);
  } else {
    iree_status_ignore(completion_status);
  }
  iree_net_loopback_carrier_end_send_operation(carrier);
  iree_net_loopback_pending_send_free(pending_send);
}

// Fires from within iree_async_proactor_poll() on the proactor thread. Drains
// queued committed sends, delivers data to peer recv handlers, and fires
// sender send-completion callbacks for send() calls.
static void iree_net_loopback_carrier_send_drain_completion(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;  // NOP is never multishot.
  iree_net_loopback_carrier_t* carrier =
      (iree_net_loopback_carrier_t*)user_data;

  if (iree_status_is_ok(status)) {
    iree_net_loopback_pending_send_t* pending_send = NULL;
    while ((pending_send = iree_net_loopback_pending_send_pop(carrier)) !=
           NULL) {
      iree_net_loopback_pending_send_deliver(pending_send);
    }
  } else {
    iree_net_loopback_pending_send_t* pending_send = NULL;
    while ((pending_send = iree_net_loopback_pending_send_pop(carrier)) !=
           NULL) {
      iree_net_loopback_pending_send_fail(pending_send, status);
    }
  }
  iree_status_ignore(status);
  iree_net_carrier_release(&carrier->base);
}

//===----------------------------------------------------------------------===//
// Carrier interface implementation
//===----------------------------------------------------------------------===//

static void iree_net_loopback_carrier_destroy(
    iree_net_carrier_t* base_carrier) {
  iree_net_loopback_carrier_t* carrier =
      iree_net_loopback_carrier_cast(base_carrier);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Assert state is DEACTIVATED or CREATED (never activated).
  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  IREE_ASSERT(state == IREE_NET_CARRIER_STATE_DEACTIVATED ||
              state == IREE_NET_CARRIER_STATE_CREATED);

  // Notify surviving peer before clearing the link. This delivers a transport
  // error to the peer's session so it can transition to ERROR state and be
  // cleaned up by the server.
  iree_net_loopback_carrier_notify_peer_disconnect(carrier);

  // Clear peer link if still set (peer may have already been destroyed).
  if (carrier->peer) {
    carrier->peer->peer = NULL;
    carrier->peer = NULL;
  }

  // Release proactor reference.
  iree_async_proactor_release(carrier->proactor);

  iree_slim_mutex_deinitialize(&carrier->send_queue_mutex);
  iree_net_loopback_carrier_release_file_transfers(carrier);
  iree_slim_mutex_deinitialize(&carrier->file_transfer_mutex);

  // Free carrier memory.
  iree_allocator_t allocator = carrier->base.host_allocator;
  iree_allocator_free(allocator, carrier);
  IREE_TRACE_ZONE_END(z0);
}

static void iree_net_loopback_carrier_set_recv_handler(
    iree_net_carrier_t* base_carrier, iree_net_carrier_recv_handler_t handler) {
  base_carrier->recv_handler = handler;
}

static iree_status_t iree_net_loopback_carrier_activate(
    iree_net_carrier_t* base_carrier) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Verify state is CREATED.
  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_CREATED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier must be in CREATED state to activate");
  }

  // Verify recv handler is set.
  if (!base_carrier->recv_handler.fn) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "recv handler must be set before activation");
  }

  // Transition to ACTIVE state.
  iree_net_carrier_set_state(base_carrier, IREE_NET_CARRIER_STATE_ACTIVE);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_net_loopback_carrier_deactivate(
    iree_net_carrier_t* base_carrier,
    iree_net_carrier_deactivate_callback_fn_t callback, void* user_data) {
  iree_net_loopback_carrier_t* carrier =
      iree_net_loopback_carrier_cast(base_carrier);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Verify state is ACTIVE or CREATED.
  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE &&
      state != IREE_NET_CARRIER_STATE_CREATED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "carrier must be in ACTIVE or CREATED state to deactivate");
  }

  // Store callback for deferred invocation when all operations drain.
  carrier->deactivate_callback.fn = callback;
  carrier->deactivate_callback.user_data = user_data;

  // Notify surviving peer before clearing the link. This delivers a transport
  // error so the peer's session can detect the disconnect.
  iree_net_loopback_carrier_notify_peer_disconnect(carrier);

  // Clear peer link so in-flight NOP completions skip delivery and new sends
  // fail immediately.
  if (carrier->peer) {
    carrier->peer->peer = NULL;
    carrier->peer = NULL;
  }

  // If never activated, transition directly to DEACTIVATED.
  if (state == IREE_NET_CARRIER_STATE_CREATED) {
    iree_net_carrier_set_state(base_carrier,
                               IREE_NET_CARRIER_STATE_DEACTIVATED);
    if (callback) callback(user_data);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Transition to DRAINING. In-flight NOP completions will check this state.
  iree_net_carrier_set_state(base_carrier, IREE_NET_CARRIER_STATE_DRAINING);

  // If no pending operations, complete immediately.
  iree_net_loopback_carrier_maybe_complete_deactivation(carrier);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_net_carrier_send_budget_t
iree_net_loopback_carrier_query_send_budget(iree_net_carrier_t* base_carrier) {
  iree_net_loopback_carrier_t* carrier =
      iree_net_loopback_carrier_cast(base_carrier);

  // If peer is gone, no budget available.
  if (!carrier->peer) {
    iree_net_carrier_send_budget_t budget = {0, 0};
    return budget;
  }

  iree_net_carrier_send_budget_t budget;
  budget.slots = UINT32_MAX;
  budget.bytes = IREE_HOST_SIZE_MAX;
  return budget;
}

static iree_status_t iree_net_loopback_carrier_send(
    iree_net_carrier_t* base_carrier, const iree_net_send_params_t* params) {
  iree_net_loopback_carrier_t* carrier =
      iree_net_loopback_carrier_cast(base_carrier);

  // Calculate total size and reject empty sends (before pending_operations
  // increment to avoid unnecessary rollback).
  iree_host_size_t total_size = 0;
  for (iree_host_size_t i = 0; i < params->data.count; ++i) {
    if (!iree_host_size_checked_add(total_size, params->data.values[i].length,
                                    &total_size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "send payload size overflow");
    }
  }
  if (total_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty sends are not allowed");
  }

  iree_status_t status =
      iree_net_loopback_carrier_begin_send_operation(carrier);
  bool send_operation_active = iree_status_is_ok(status);

  iree_net_loopback_pending_send_t* pending_send = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_net_loopback_pending_send_allocate(carrier, total_size,
                                                     &pending_send);
  }
  if (iree_status_is_ok(status)) {
    uint8_t* write_ptr = pending_send->storage;
    for (iree_host_size_t i = 0; i < params->data.count; ++i) {
      memcpy(write_ptr, iree_async_span_ptr(params->data.values[i]),
             params->data.values[i].length);
      write_ptr += params->data.values[i].length;
    }
    pending_send->user_data = params->user_data;
    pending_send->notify_completion = true;
    status = iree_net_loopback_pending_send_enqueue(pending_send);
    if (iree_status_is_ok(status)) {
      pending_send = NULL;
      send_operation_active = false;
    }
  }

  if (!iree_status_is_ok(status)) {
    if (send_operation_active) {
      iree_net_loopback_carrier_end_send_operation(carrier);
    }
    iree_net_loopback_pending_send_free(pending_send);
  }
  return status;
}

static iree_status_t iree_net_loopback_carrier_begin_send(
    iree_net_carrier_t* base_carrier, iree_host_size_t size, void** out_ptr,
    iree_net_carrier_send_handle_t* out_handle) {
  iree_net_loopback_carrier_t* carrier =
      iree_net_loopback_carrier_cast(base_carrier);

  // Reject empty sends.
  if (size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty sends are not allowed");
  }

  iree_status_t status =
      iree_net_loopback_carrier_begin_send_operation(carrier);
  bool send_operation_active = iree_status_is_ok(status);

  iree_net_loopback_pending_send_t* pending_send = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_net_loopback_pending_send_allocate(carrier, size, &pending_send);
  }
  if (iree_status_is_ok(status)) {
    *out_ptr = pending_send->storage;
    *out_handle = (iree_net_carrier_send_handle_t)(uintptr_t)pending_send;
    pending_send = NULL;
    send_operation_active = false;
  }

  if (!iree_status_is_ok(status)) {
    if (send_operation_active) {
      iree_net_loopback_carrier_end_send_operation(carrier);
    }
    iree_net_loopback_pending_send_free(pending_send);
  }
  return status;
}

static iree_status_t iree_net_loopback_carrier_commit_send(
    iree_net_carrier_t* base_carrier, iree_net_carrier_send_handle_t handle) {
  (void)base_carrier;
  iree_net_loopback_pending_send_t* pending_send =
      (iree_net_loopback_pending_send_t*)(uintptr_t)handle;
  iree_net_loopback_carrier_t* carrier = pending_send->carrier;
  iree_status_t status = iree_net_loopback_pending_send_enqueue(pending_send);
  if (!iree_status_is_ok(status)) {
    iree_net_loopback_carrier_end_send_operation(carrier);
    iree_net_loopback_pending_send_free(pending_send);
  }
  return status;
}

static void iree_net_loopback_carrier_abort_send(
    iree_net_carrier_t* base_carrier, iree_net_carrier_send_handle_t handle) {
  (void)base_carrier;
  iree_net_loopback_pending_send_t* pending_send =
      (iree_net_loopback_pending_send_t*)(uintptr_t)handle;
  iree_net_loopback_carrier_t* carrier = pending_send->carrier;
  iree_net_loopback_carrier_end_send_operation(carrier);
  iree_net_loopback_pending_send_free(pending_send);
}

static iree_status_t iree_net_loopback_carrier_direct_write(
    iree_net_carrier_t* base_carrier,
    const iree_net_direct_write_params_t* params) {
  (void)base_carrier;
  (void)params;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "loopback carrier does not support direct_write");
}

static iree_status_t iree_net_loopback_carrier_direct_read(
    iree_net_carrier_t* base_carrier,
    const iree_net_direct_read_params_t* params) {
  (void)base_carrier;
  (void)params;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "loopback carrier does not support direct_read");
}

static iree_status_t iree_net_loopback_carrier_register_buffer(
    iree_net_carrier_t* base_carrier, iree_async_region_t* region,
    iree_net_remote_handle_t* out_handle) {
  (void)base_carrier;
  (void)region;
  (void)out_handle;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "loopback carrier does not support register_buffer");
}

static void iree_net_loopback_carrier_unregister_buffer(
    iree_net_carrier_t* base_carrier, iree_net_remote_handle_t handle) {
  (void)base_carrier;
  (void)handle;
}

static iree_status_t iree_net_loopback_carrier_query_file_handle_transfer(
    iree_net_carrier_t* base_carrier, iree_io_file_handle_t* file_handle,
    iree_net_file_handle_transfer_type_t* out_transfer_type,
    iree_host_size_t* out_payload_length) {
  IREE_ASSERT_ARGUMENT(file_handle);
  IREE_ASSERT_ARGUMENT(out_transfer_type);
  IREE_ASSERT_ARGUMENT(out_payload_length);

  *out_transfer_type = IREE_NET_FILE_HANDLE_TRANSFER_TYPE_NONE;
  *out_payload_length = 0;

  iree_status_t status = iree_ok_status();
  iree_net_loopback_carrier_t* carrier =
      iree_net_loopback_carrier_cast(base_carrier);
#if !IREE_FILE_IO_ENABLE
  status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "file support has been compiled out of this "
                            "binary; set IREE_FILE_IO_ENABLE=1 to include it");
#endif  // !IREE_FILE_IO_ENABLE
  if (iree_status_is_ok(status) && !carrier->peer) {
    status = iree_make_status(IREE_STATUS_UNAVAILABLE, "peer disconnected");
  }
  if (iree_status_is_ok(status) &&
      iree_io_file_handle_type(file_handle) != IREE_IO_FILE_HANDLE_TYPE_FD) {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "loopback carrier can only transfer descriptor-backed file handles");
  }
  if (iree_status_is_ok(status)) {
    *out_transfer_type = iree_net_loopback_file_transfer_type();
    *out_payload_length = sizeof(iree_net_loopback_file_transfer_payload_t);
  }
  return status;
}

static iree_status_t iree_net_loopback_carrier_export_file_handle(
    iree_net_carrier_t* base_carrier, iree_io_file_handle_t* file_handle,
    iree_net_file_handle_transfer_type_t transfer_type,
    iree_byte_span_t transfer_payload) {
  IREE_ASSERT_ARGUMENT(file_handle);

  iree_net_loopback_carrier_t* carrier =
      iree_net_loopback_carrier_cast(base_carrier);
  iree_status_t status = iree_ok_status();
#if !IREE_FILE_IO_ENABLE
  status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "file support has been compiled out of this "
                            "binary; set IREE_FILE_IO_ENABLE=1 to include it");
#endif  // !IREE_FILE_IO_ENABLE
  if (iree_status_is_ok(status) &&
      transfer_type != iree_net_loopback_file_transfer_type()) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported loopback file transfer type %u",
                              (uint32_t)transfer_type);
  }
  if (iree_status_is_ok(status) &&
      transfer_payload.data_length !=
          sizeof(iree_net_loopback_file_transfer_payload_t)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "loopback file transfer payload length must be %" PRIhsz " bytes",
        (iree_host_size_t)sizeof(iree_net_loopback_file_transfer_payload_t));
  }
  if (iree_status_is_ok(status) && !carrier->peer) {
    status = iree_make_status(IREE_STATUS_UNAVAILABLE, "peer disconnected");
  }
  if (iree_status_is_ok(status) &&
      iree_io_file_handle_type(file_handle) != IREE_IO_FILE_HANDLE_TYPE_FD) {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "loopback carrier can only transfer descriptor-backed file handles");
  }

  iree_net_loopback_file_transfer_t* transfer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(carrier->base.host_allocator,
                                   sizeof(*transfer), (void**)&transfer);
  }
  if (iree_status_is_ok(status)) {
    memset(transfer, 0, sizeof(*transfer));
    transfer->file_handle = file_handle;
    iree_io_file_handle_retain(transfer->file_handle);

    iree_slim_mutex_lock(&carrier->file_transfer_mutex);
    transfer->id = carrier->next_file_transfer_id++;
    if (carrier->next_file_transfer_id == 0) {
      carrier->next_file_transfer_id = 1;
    }
    transfer->next = carrier->pending_file_transfers;
    carrier->pending_file_transfers = transfer;
    iree_slim_mutex_unlock(&carrier->file_transfer_mutex);

    iree_net_loopback_file_transfer_payload_t payload = {
        .id = transfer->id,
    };
    memcpy(transfer_payload.data, &payload, sizeof(payload));
  }
  return status;
}

static iree_status_t iree_net_loopback_carrier_import_file_handle(
    iree_net_carrier_t* base_carrier,
    iree_net_file_handle_transfer_type_t transfer_type,
    iree_const_byte_span_t transfer_payload, iree_allocator_t host_allocator,
    iree_io_file_handle_t** out_file_handle) {
  IREE_ASSERT_ARGUMENT(out_file_handle);
  *out_file_handle = NULL;
  (void)host_allocator;

  iree_net_loopback_carrier_t* carrier =
      iree_net_loopback_carrier_cast(base_carrier);
  iree_net_loopback_file_transfer_payload_t payload = {0};
  iree_status_t status = iree_ok_status();
#if !IREE_FILE_IO_ENABLE
  status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "file support has been compiled out of this "
                            "binary; set IREE_FILE_IO_ENABLE=1 to include it");
#endif  // !IREE_FILE_IO_ENABLE
  if (iree_status_is_ok(status) &&
      transfer_type != iree_net_loopback_file_transfer_type()) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported loopback file transfer type %u",
                              (uint32_t)transfer_type);
  }
  if (iree_status_is_ok(status) &&
      transfer_payload.data_length != sizeof(payload)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "loopback file transfer payload length must be %" PRIhsz " bytes",
        (iree_host_size_t)sizeof(payload));
  }
  if (iree_status_is_ok(status) && !carrier->peer) {
    status = iree_make_status(IREE_STATUS_UNAVAILABLE, "peer disconnected");
  }
  if (iree_status_is_ok(status)) {
    memcpy(&payload, transfer_payload.data, sizeof(payload));
  }

  iree_net_loopback_file_transfer_t* transfer = NULL;
  iree_allocator_t transfer_allocator = carrier->base.host_allocator;
  if (iree_status_is_ok(status)) {
    transfer_allocator = carrier->peer->base.host_allocator;
    transfer =
        iree_net_loopback_carrier_take_file_transfer(carrier->peer, payload.id);
    if (!transfer) {
      status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                "loopback file transfer token not found");
    }
  }
  if (iree_status_is_ok(status)) {
    *out_file_handle = transfer->file_handle;
    transfer->file_handle = NULL;
  }
  if (transfer) {
    iree_io_file_handle_release(transfer->file_handle);
    iree_allocator_free(transfer_allocator, transfer);
  }
  return status;
}

static iree_status_t iree_net_loopback_carrier_release_file_handle_transfer(
    iree_net_carrier_t* base_carrier,
    iree_net_file_handle_transfer_type_t transfer_type,
    iree_const_byte_span_t transfer_payload) {
  iree_net_loopback_carrier_t* carrier =
      iree_net_loopback_carrier_cast(base_carrier);
  iree_net_loopback_file_transfer_payload_t payload = {0};
  if (transfer_type != iree_net_loopback_file_transfer_type()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported loopback file transfer type %u",
                            (uint32_t)transfer_type);
  }
  if (transfer_payload.data_length != sizeof(payload)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "loopback file transfer payload length must be %" PRIhsz " bytes",
        (iree_host_size_t)sizeof(payload));
  }
  memcpy(&payload, transfer_payload.data, sizeof(payload));
  iree_net_loopback_file_transfer_t* transfer =
      iree_net_loopback_carrier_take_file_transfer(carrier, payload.id);
  if (transfer) {
    iree_io_file_handle_release(transfer->file_handle);
    iree_allocator_free(carrier->base.host_allocator, transfer);
  }
  return iree_ok_status();
}

static iree_status_t iree_net_loopback_carrier_shutdown(
    iree_net_carrier_t* base_carrier) {
  iree_net_loopback_carrier_t* carrier =
      iree_net_loopback_carrier_cast(base_carrier);

  // Verify state is ACTIVE.
  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier must be in ACTIVE state to shutdown");
  }

  // Mark shutdown initiated — future sends will fail.
  carrier->shutdown_initiated = true;

  return iree_ok_status();
}

static const iree_net_carrier_vtable_t iree_net_loopback_carrier_vtable = {
    .destroy = iree_net_loopback_carrier_destroy,
    .set_recv_handler = iree_net_loopback_carrier_set_recv_handler,
    .activate = iree_net_loopback_carrier_activate,
    .deactivate = iree_net_loopback_carrier_deactivate,
    .query_send_budget = iree_net_loopback_carrier_query_send_budget,
    .send = iree_net_loopback_carrier_send,
    .begin_send = iree_net_loopback_carrier_begin_send,
    .commit_send = iree_net_loopback_carrier_commit_send,
    .abort_send = iree_net_loopback_carrier_abort_send,
    .shutdown = iree_net_loopback_carrier_shutdown,
    .direct_write = iree_net_loopback_carrier_direct_write,
    .direct_read = iree_net_loopback_carrier_direct_read,
    .register_buffer = iree_net_loopback_carrier_register_buffer,
    .unregister_buffer = iree_net_loopback_carrier_unregister_buffer,
    .query_file_handle_transfer =
        iree_net_loopback_carrier_query_file_handle_transfer,
    .export_file_handle = iree_net_loopback_carrier_export_file_handle,
    .import_file_handle = iree_net_loopback_carrier_import_file_handle,
    .release_file_handle_transfer =
        iree_net_loopback_carrier_release_file_handle_transfer,
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

// Initializes a single loopback carrier.
static void iree_net_loopback_carrier_init(
    iree_net_loopback_carrier_t* carrier, iree_async_proactor_t* proactor,
    iree_net_carrier_capabilities_t capabilities,
    iree_net_carrier_callback_t callback, iree_allocator_t host_allocator) {
  memset(carrier, 0, sizeof(*carrier));
  iree_net_carrier_initialize(&iree_net_loopback_carrier_vtable, capabilities,
                              0,         // No MTU (stream-like).
                              SIZE_MAX,  // Unlimited scatter-gather.
                              callback, host_allocator, &carrier->base);
  carrier->proactor = proactor;
  iree_async_proactor_retain(proactor);
  carrier->shutdown_initiated = false;
  iree_slim_mutex_initialize(&carrier->send_queue_mutex);
  iree_slim_mutex_initialize(&carrier->file_transfer_mutex);
  carrier->next_file_transfer_id = 1;
}

IREE_API_EXPORT void iree_net_loopback_carrier_set_peer_disconnect_handler(
    iree_net_carrier_t* base_carrier,
    iree_net_loopback_carrier_disconnect_handler_t handler) {
  iree_net_loopback_carrier_t* carrier =
      iree_net_loopback_carrier_cast(base_carrier);
  carrier->peer_disconnect_handler = handler;
}

IREE_API_EXPORT iree_status_t iree_net_loopback_carrier_create_pair(
    iree_async_proactor_t* proactor, iree_net_carrier_callback_t callback,
    iree_allocator_t host_allocator, iree_net_carrier_t** out_client,
    iree_net_carrier_t** out_server) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_client);
  IREE_ASSERT_ARGUMENT(out_server);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_client = NULL;
  *out_server = NULL;

  // Allocate both carriers.
  iree_net_loopback_carrier_t* client = NULL;
  iree_net_loopback_carrier_t* server = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*client), (void**)&client);
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, sizeof(*server), (void**)&server);
  }

  if (iree_status_is_ok(status)) {
    // Capabilities: reliable, ordered. Data is copied during send() (no
    // zero-copy TX) to match real carrier behavior where the sender's buffer
    // can be freed immediately after send() returns.
    iree_net_carrier_capabilities_t capabilities =
        IREE_NET_CARRIER_CAPABILITY_RELIABLE |
        IREE_NET_CARRIER_CAPABILITY_ORDERED;
#if IREE_FILE_IO_ENABLE
#if defined(IREE_PLATFORM_WINDOWS)
    capabilities |= IREE_NET_CARRIER_CAPABILITY_WIN32_HANDLE_TRANSFER;
#else
    capabilities |= IREE_NET_CARRIER_CAPABILITY_POSIX_FD_TRANSFER;
#endif  // IREE_PLATFORM_WINDOWS
#endif  // IREE_FILE_IO_ENABLE

    // Initialize both carriers.
    iree_net_loopback_carrier_init(client, proactor, capabilities, callback,
                                   host_allocator);
    iree_net_loopback_carrier_init(server, proactor, capabilities, callback,
                                   host_allocator);

    // Set up peer links.
    client->peer = server;
    server->peer = client;

    *out_client = &client->base;
    *out_server = &server->base;
  } else {
    // Cleanup on allocation failure. Carriers are initialized only after both
    // allocations succeed, so there is no proactor reference to release here.
    iree_allocator_free(host_allocator, client);
    iree_allocator_free(host_allocator, server);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
