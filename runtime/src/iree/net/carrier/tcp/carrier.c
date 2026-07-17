// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/tcp/carrier.h"

#include "iree/async/operations/net.h"
#include "iree/async/socket.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/internal/math.h"
#include "iree/base/threading/mutex.h"

//===----------------------------------------------------------------------===//
// Send slot management
//===----------------------------------------------------------------------===//

// Inline storage capacity for unregistered span data in each send slot.
// Covers TCP frame headers (16 bytes) and small metadata. Unregistered spans
// (region == NULL) use raw pointers that may reference caller stack frames.
// The io_uring backend defers kernel data reads until io_uring_enter, so data
// must survive beyond the submit call. This buffer provides stable storage.
#define IREE_NET_TCP_SEND_SLOT_INLINE_DATA_CAPACITY 64

typedef uint8_t iree_net_tcp_send_slot_flags_t;
enum iree_net_tcp_send_slot_flag_bits_e {
  IREE_NET_TCP_SEND_SLOT_FLAG_NONE = 0u,
  IREE_NET_TCP_SEND_SLOT_FLAG_QUEUED_RESOURCES_RETAINED = 1u << 0,
};

// Pre-allocated send operation slot.
// Each slot tracks one in-flight send operation and provides inline storage
// for the span list and small unregistered span data. This ensures all data
// referenced by io_uring SQEs survives until the kernel processes them.
typedef struct iree_net_tcp_send_slot_t {
  // The async send operation submitted to the proactor.
  iree_async_socket_send_operation_t operation;

  // Next committed send slot waiting for proactor submission.
  struct iree_net_tcp_send_slot_t* pending_next;

  // Flags tracking carrier-owned send slot state.
  iree_net_tcp_send_slot_flags_t flags;

  // User data from iree_net_send_params_t, echoed to completion callback.
  uint64_t user_data;

  // Slot-local copy of the span list values. The caller's span array may be
  // stack-allocated, so we copy it here to ensure operation.buffers.values
  // references stable memory.
  iree_async_span_t inline_spans[IREE_ASYNC_SOCKET_SEND_MAX_BUFFERS];

  // Inline data storage for unregistered span payloads. When a span has
  // region == NULL, its data pointer may reference transient memory (e.g. a
  // stack-allocated frame header). Small payloads are copied here so that
  // iovec data pointers in the platform storage reference heap memory.
  uint8_t inline_data[IREE_NET_TCP_SEND_SLOT_INLINE_DATA_CAPACITY];

  // Buffer allocated by begin_send. Freed in send completion. NULL for
  // regular send() operations (which use caller-owned scatter-gather buffers).
  void* begin_send_buffer;

  // Size of the begin_send buffer, stored here so the handle carries only the
  // slot index (avoiding truncation for sizes > UINT32_MAX).
  iree_host_size_t begin_send_size;
} iree_net_tcp_send_slot_t;

// State of a single-shot receive slot.
typedef enum iree_net_tcp_recv_slot_state_e {
  IREE_NET_TCP_RECV_SLOT_RETIRED = 0,
  IREE_NET_TCP_RECV_SLOT_SUBMITTED = 1,
  IREE_NET_TCP_RECV_SLOT_PAUSED = 2,
} iree_net_tcp_recv_slot_state_t;

// Pre-allocated single-shot receive operation slot.
typedef struct iree_net_tcp_recv_slot_t {
  // Async receive operation. Must be first for callback downcasting.
  iree_async_socket_recv_pool_operation_t operation;

  // Current slot state; see iree_net_tcp_recv_slot_state_t.
  iree_atomic_int32_t state;
} iree_net_tcp_recv_slot_t;

// Per-buffer release wrapper state for retained receive leases.
typedef struct iree_net_tcp_recv_lease_context_t {
  // Carrier to resume when this buffer is returned.
  struct iree_net_tcp_carrier_t* carrier;

  // Original release callback that returns the buffer to its source.
  iree_async_buffer_recycle_callback_t release;

  // Lease release state: 0 pending, 1 released, 2 retained past callback.
  iree_atomic_int32_t state;
} iree_net_tcp_recv_lease_context_t;

enum {
  IREE_NET_TCP_RECV_LEASE_PENDING = 0,
  IREE_NET_TCP_RECV_LEASE_RELEASED = 1,
  IREE_NET_TCP_RECV_LEASE_RETAINED = 2,
};

//===----------------------------------------------------------------------===//
// Deactivate callback storage
//===----------------------------------------------------------------------===//

// Bundles deactivate callback function with user data.
typedef struct iree_net_tcp_deactivate_callback_t {
  iree_net_carrier_deactivate_callback_fn_t fn;
  void* user_data;
} iree_net_tcp_deactivate_callback_t;

//===----------------------------------------------------------------------===//
// TCP carrier structure
//===----------------------------------------------------------------------===//

typedef struct iree_net_tcp_carrier_t {
  // Base carrier (must be first for safe upcasting).
  iree_net_carrier_t base;

  // Proactor this carrier is bound to. Not owned.
  iree_async_proactor_t* proactor;

  // Socket owned by this carrier.
  iree_async_socket_t* socket;

  // Buffer pool for receive operations. Not owned - caller must ensure it
  // outlives the carrier.
  iree_async_buffer_pool_t* recv_pool;

  // Per-buffer release wrappers used to resume paused receive operations.
  iree_net_tcp_recv_lease_context_t* recv_lease_contexts;

  // Number of entries in recv_lease_contexts.
  iree_host_size_t recv_lease_context_count;

  // Send slot bitmap. Bit i set = slot i is free.
  // Claim: find first set bit (ctz), CAS-clear it.
  // Release: atomic OR to set bit.
  // No ordering dependency between slots — out-of-order completion is correct.
  struct {
    // Guards pending_head, pending_tail, and submitted_count.
    iree_slim_mutex_t pending_mutex;

    // Head of the FIFO committed-send queue awaiting proactor submission.
    iree_net_tcp_send_slot_t* pending_head;

    // Tail of the FIFO committed-send queue awaiting proactor submission.
    iree_net_tcp_send_slot_t* pending_tail;

    // Number of send slots currently submitted to the proactor.
    uint32_t submitted_count;

    // Number of allocated send slots.
    uint32_t slot_count;

    // Bitmap of free send slots; bit i set means slot i is free.
    iree_atomic_uint32_t free_bitmap;

    // Contiguous storage for slot_count send slots.
    iree_net_tcp_send_slot_t* slots;
  } send;

  // Receive operations.
  struct {
    // True if using multishot recv with PBUF_RING (io_uring 5.19+).
    bool multishot_enabled;

    // Incremented whenever a receive buffer lease returns to its source.
    iree_atomic_int32_t returned_buffer_epoch;

    // For multishot: single operation that stays posted.
    // For single-shot: ring of recv operations.
    union {
      struct {
        // Multishot receive operation.
        iree_async_socket_recv_pool_operation_t operation;

        // True when the operation is paused waiting for a returned buffer.
        iree_atomic_int32_t paused;
      } multishot;
      struct {
        // Number of single-shot receive slots (power of 2).
        uint32_t slot_count;

        // Receive slots, re-posted after each completion.
        iree_net_tcp_recv_slot_t* slots;

        // Number of logical receive slots that have not terminated.
        iree_atomic_int32_t live_count;

        // Number of receive slots paused waiting for buffer availability.
        iree_atomic_int32_t paused_count;
      } single_shot;
    };
  } recv;

  // Sticky failure status. Initially NULL (no error).
  // Set via atomic CAS - first error wins, subsequent errors are ignored.
  iree_atomic_intptr_t failure_status;

  // Callback to invoke when deactivation completes.
  iree_net_tcp_deactivate_callback_t deactivate_callback;
} iree_net_tcp_carrier_t;

// Casts from base carrier to TCP carrier.
static inline iree_net_tcp_carrier_t* iree_net_tcp_carrier_cast(
    iree_net_carrier_t* base_carrier) {
  return (iree_net_tcp_carrier_t*)base_carrier;
}

//===----------------------------------------------------------------------===//
// Sticky failure status helpers
//===----------------------------------------------------------------------===//

// Sets the sticky failure status (first error wins).
static void iree_net_tcp_carrier_set_failure_status(
    iree_net_tcp_carrier_t* carrier, iree_status_t status) {
  intptr_t expected = 0;
  intptr_t desired = (intptr_t)status;
  if (!iree_atomic_compare_exchange_strong(&carrier->failure_status, &expected,
                                           desired, iree_memory_order_release,
                                           iree_memory_order_relaxed)) {
    // Another error was already captured.
    iree_status_free(status);
  }
}

// Returns a CLONE of the sticky failure status, or iree_ok_status() if none.
// Caller takes ownership of the returned status.
static iree_status_t iree_net_tcp_carrier_get_failure_status(
    iree_net_tcp_carrier_t* carrier) {
  intptr_t stored =
      iree_atomic_load(&carrier->failure_status, iree_memory_order_acquire);
  if (stored == 0) {
    return iree_ok_status();
  }
  // Clone to avoid double-free: the stored status remains owned by carrier.
  return iree_status_clone((iree_status_t)stored);
}

// Consumes the sticky failure status (called once during destruction).
static void iree_net_tcp_carrier_consume_failure_status(
    iree_net_tcp_carrier_t* carrier) {
  intptr_t stored = iree_atomic_exchange(&carrier->failure_status, 0,
                                         iree_memory_order_acq_rel);
  if (stored != 0) {
    iree_status_free((iree_status_t)stored);
  }
}

// Requests cancellation of a submitted operation. NOT_FOUND means its
// completion won the race and owns terminal retirement. Any other failure
// would leave the carrier unable to prove that deactivation can complete.
static void iree_net_tcp_carrier_cancel_operation(
    iree_net_tcp_carrier_t* carrier, iree_async_operation_t* operation) {
  iree_status_t status =
      iree_async_proactor_cancel(carrier->proactor, operation);
  if (iree_status_is_ok(status)) return;
  if (iree_status_is_not_found(status)) {
    iree_status_free(status);
    return;
  }
  iree_status_abort(status);
}

//===----------------------------------------------------------------------===//
// Deactivation completion check
//===----------------------------------------------------------------------===//

// Checks if deactivation has completed and invokes callback if so.
static void iree_net_tcp_carrier_maybe_complete_deactivation(
    iree_net_tcp_carrier_t* carrier) {
  // Check all pending operations have completed.
  int32_t pending = iree_atomic_load(&carrier->base.pending_operations,
                                     iree_memory_order_acquire);
  if (pending > 0) return;

  // Several operation callbacks may observe zero pending work concurrently.
  // Only the state-transition winner owns terminal callback delivery.
  if (!iree_net_carrier_try_transition_state(
          &carrier->base, IREE_NET_CARRIER_STATE_DRAINING,
          IREE_NET_CARRIER_STATE_DEACTIVATED)) {
    return;
  }

  iree_net_carrier_deactivate_callback_fn_t callback =
      carrier->deactivate_callback.fn;
  void* callback_user_data = carrier->deactivate_callback.user_data;
  carrier->deactivate_callback.fn = NULL;
  carrier->deactivate_callback.user_data = NULL;
  if (callback) callback(callback_user_data);
}

// Retires one operation and attempts terminal callback delivery only when this
// caller owned the final count. No carrier access is permitted after this call.
static void iree_net_tcp_carrier_retire_pending_operation(
    iree_net_tcp_carrier_t* carrier) {
  if (iree_net_carrier_retire_pending_operation(&carrier->base)) {
    iree_net_tcp_carrier_maybe_complete_deactivation(carrier);
  }
}

//===----------------------------------------------------------------------===//
// Recv pause/resume helpers
//===----------------------------------------------------------------------===//

static iree_status_t iree_net_tcp_carrier_submit_multishot_recv(
    iree_net_tcp_carrier_t* carrier,
    iree_async_socket_recv_pool_operation_t* recv_op);

static iree_status_t iree_net_tcp_carrier_submit_single_shot_recv(
    iree_net_tcp_carrier_t* carrier, iree_net_tcp_recv_slot_t* slot);

static void iree_net_tcp_carrier_resume_one_paused_recv(
    iree_net_tcp_carrier_t* carrier);

static void iree_net_tcp_carrier_send_completion(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags);

static void iree_net_tcp_carrier_terminate_single_shot_recv(
    iree_net_tcp_carrier_t* carrier, iree_net_tcp_recv_slot_t* slot) {
  int32_t old_state = iree_atomic_exchange(
      &slot->state, IREE_NET_TCP_RECV_SLOT_RETIRED, iree_memory_order_acq_rel);
  if (old_state == IREE_NET_TCP_RECV_SLOT_RETIRED) return;

  if (old_state == IREE_NET_TCP_RECV_SLOT_PAUSED) {
    iree_atomic_fetch_sub(&carrier->recv.single_shot.paused_count, 1,
                          iree_memory_order_release);
  }

  int32_t old_live_count = iree_atomic_fetch_sub(
      &carrier->recv.single_shot.live_count, 1, iree_memory_order_release);

  if (old_live_count == 1) {
    iree_net_carrier_state_t state = iree_net_carrier_state(&carrier->base);
    if (state == IREE_NET_CARRIER_STATE_ACTIVE) {
      iree_status_t status = iree_async_socket_shutdown(
          carrier->socket, IREE_ASYNC_SOCKET_SHUTDOWN_READ);
      if (!iree_status_is_ok(status)) {
        iree_net_tcp_carrier_set_failure_status(carrier, status);
      }
    }
  }

  iree_net_tcp_carrier_retire_pending_operation(carrier);
}

static void iree_net_tcp_carrier_pause_single_shot_recv(
    iree_net_tcp_carrier_t* carrier, iree_net_tcp_recv_slot_t* slot,
    int32_t observed_returned_buffer_epoch) {
  iree_atomic_store(&slot->state, IREE_NET_TCP_RECV_SLOT_PAUSED,
                    iree_memory_order_release);
  iree_atomic_fetch_add(&carrier->recv.single_shot.paused_count, 1,
                        iree_memory_order_release);

  // A buffer may have been returned concurrently with the transition to paused.
  // Resume only if we observed such a return; otherwise wait for the next
  // release callback to avoid spinning while the pool remains exhausted.
  int32_t current_returned_buffer_epoch = iree_atomic_load(
      &carrier->recv.returned_buffer_epoch, iree_memory_order_acquire);
  if (current_returned_buffer_epoch != observed_returned_buffer_epoch) {
    iree_net_tcp_carrier_resume_one_paused_recv(carrier);
  }
}

static void iree_net_tcp_carrier_pause_multishot_recv(
    iree_net_tcp_carrier_t* carrier, int32_t observed_returned_buffer_epoch) {
  iree_atomic_store(&carrier->recv.multishot.paused, 1,
                    iree_memory_order_release);

  // A buffer may have been returned concurrently with the transition to paused.
  // Resume only if we observed such a return; otherwise wait for the next
  // release callback to avoid spinning while the ring remains exhausted.
  int32_t current_returned_buffer_epoch = iree_atomic_load(
      &carrier->recv.returned_buffer_epoch, iree_memory_order_acquire);
  if (current_returned_buffer_epoch != observed_returned_buffer_epoch) {
    iree_net_tcp_carrier_resume_one_paused_recv(carrier);
  }
}

static void iree_net_tcp_carrier_recv_lease_release(void* user_data,
                                                    uint32_t buffer_index) {
  iree_net_tcp_recv_lease_context_t* context =
      (iree_net_tcp_recv_lease_context_t*)user_data;
  iree_net_tcp_carrier_t* carrier = context->carrier;
  iree_async_buffer_recycle_callback_t release = context->release;
  int32_t old_state =
      iree_atomic_exchange(&context->state, IREE_NET_TCP_RECV_LEASE_RELEASED,
                           iree_memory_order_acq_rel);

  if (release.fn) {
    release.fn(release.user_data, buffer_index);
  }
  iree_atomic_fetch_add(&carrier->recv.returned_buffer_epoch, 1,
                        iree_memory_order_release);
  iree_net_tcp_carrier_resume_one_paused_recv(carrier);

  if (old_state == IREE_NET_TCP_RECV_LEASE_RETAINED) {
    iree_net_carrier_release(&carrier->base);
  }
}

static iree_net_tcp_recv_lease_context_t*
iree_net_tcp_carrier_prepare_recv_lease(iree_net_tcp_carrier_t* carrier,
                                        iree_async_buffer_lease_t* lease) {
  if (!lease || !lease->release.fn) return NULL;
  if (IREE_UNLIKELY(lease->buffer_index >= carrier->recv_lease_context_count)) {
    IREE_ASSERT(false, "recv lease buffer index out of range");
    return NULL;
  }

  iree_net_tcp_recv_lease_context_t* context =
      &carrier->recv_lease_contexts[lease->buffer_index];
  context->carrier = carrier;
  context->release = lease->release;
  iree_atomic_store(&context->state, IREE_NET_TCP_RECV_LEASE_PENDING,
                    iree_memory_order_release);
  lease->release.fn = iree_net_tcp_carrier_recv_lease_release;
  lease->release.user_data = context;
  return context;
}

static void iree_net_tcp_carrier_maybe_retain_stolen_recv_lease(
    iree_net_tcp_carrier_t* carrier, iree_net_tcp_recv_lease_context_t* context,
    iree_async_buffer_lease_t* lease) {
  if (!context || !lease || lease->release.fn) return;
  int32_t expected = IREE_NET_TCP_RECV_LEASE_PENDING;
  iree_net_carrier_retain(&carrier->base);
  if (iree_atomic_compare_exchange_strong(
          &context->state, &expected, IREE_NET_TCP_RECV_LEASE_RETAINED,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    return;
  }
  iree_net_carrier_release(&carrier->base);
}

static void iree_net_tcp_carrier_resume_one_paused_recv(
    iree_net_tcp_carrier_t* carrier) {
  if (iree_net_carrier_state(&carrier->base) != IREE_NET_CARRIER_STATE_ACTIVE) {
    return;
  }

  if (carrier->recv.multishot_enabled) {
    int32_t expected = 1;
    bool was_paused = iree_atomic_compare_exchange_strong(
        &carrier->recv.multishot.paused, &expected, 0,
        iree_memory_order_acq_rel, iree_memory_order_acquire);
    if (!was_paused) return;

    iree_status_t status = iree_net_tcp_carrier_submit_multishot_recv(
        carrier, &carrier->recv.multishot.operation);
    if (!iree_status_is_ok(status)) {
      iree_net_tcp_carrier_set_failure_status(carrier, status);
      iree_net_tcp_carrier_retire_pending_operation(carrier);
    }
    return;
  }

  int32_t paused_count = iree_atomic_load(
      &carrier->recv.single_shot.paused_count, iree_memory_order_acquire);
  if (paused_count <= 0) return;

  for (uint32_t i = 0; i < carrier->recv.single_shot.slot_count; ++i) {
    iree_net_tcp_recv_slot_t* slot = &carrier->recv.single_shot.slots[i];
    int32_t expected = IREE_NET_TCP_RECV_SLOT_PAUSED;
    bool claimed = iree_atomic_compare_exchange_strong(
        &slot->state, &expected, IREE_NET_TCP_RECV_SLOT_SUBMITTED,
        iree_memory_order_acq_rel, iree_memory_order_acquire);
    if (!claimed) continue;

    iree_status_t status =
        iree_net_tcp_carrier_submit_single_shot_recv(carrier, slot);
    if (iree_status_is_ok(status)) {
      iree_atomic_fetch_sub(&carrier->recv.single_shot.paused_count, 1,
                            iree_memory_order_release);
    } else if (iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
      iree_status_free(status);
      iree_atomic_store(&slot->state, IREE_NET_TCP_RECV_SLOT_PAUSED,
                        iree_memory_order_release);
    } else {
      iree_net_tcp_carrier_set_failure_status(carrier, status);
      iree_net_tcp_carrier_terminate_single_shot_recv(carrier, slot);
    }
    return;
  }
}

//===----------------------------------------------------------------------===//
// Recv completion handlers
//===----------------------------------------------------------------------===//

// Processes received data by invoking the recv handler and handling errors.
// Returns true if the carrier should continue receiving, false if receiving
// should stop (EOF, error, or deactivating).
static bool iree_net_tcp_carrier_process_recv(
    iree_net_tcp_carrier_t* carrier, iree_status_t status,
    iree_host_size_t bytes_received, iree_async_buffer_lease_t* lease) {
  // Check for errors from the recv operation itself.
  if (!iree_status_is_ok(status)) {
    iree_net_tcp_carrier_set_failure_status(carrier, status);
    return false;
  }

  // EOF: bytes_received == 0 with OK status means graceful close.
  if (bytes_received == 0) {
    if (carrier->base.recv_handler.fn) {
      iree_async_span_t empty_span = iree_async_span_make(NULL, 0, 0);
      iree_status_t handler_status = carrier->base.recv_handler.fn(
          carrier->base.recv_handler.user_data, empty_span, NULL);
      if (!iree_status_is_ok(handler_status)) {
        iree_net_tcp_carrier_set_failure_status(carrier, handler_status);
      }
    }
    return false;
  }

  // Build span from the lease, adjusted for actual bytes received.
  // The lease's span covers the full buffer; we narrow to received bytes.
  iree_async_span_t data = iree_async_span_make(
      lease->span.region, lease->span.offset, bytes_received);

  // Invoke the recv handler.
  iree_status_t handler_status = iree_ok_status();
  if (carrier->base.recv_handler.fn) {
    iree_net_tcp_recv_lease_context_t* lease_context =
        iree_net_tcp_carrier_prepare_recv_lease(carrier, lease);
    handler_status = carrier->base.recv_handler.fn(
        carrier->base.recv_handler.user_data, data, lease);
    iree_net_tcp_carrier_maybe_retain_stolen_recv_lease(carrier, lease_context,
                                                        lease);
  }

  // Capture handler errors as sticky failure status.
  if (!iree_status_is_ok(handler_status)) {
    iree_net_tcp_carrier_set_failure_status(carrier, handler_status);
    return false;
  }

  // Check if we're still active (handler might have triggered deactivation).
  iree_net_carrier_state_t state = iree_net_carrier_state(&carrier->base);
  return state == IREE_NET_CARRIER_STATE_ACTIVE;
}

// Completion callback for multishot recv operations.
// Fires repeatedly until EOF, error, or cancellation.
static void iree_net_tcp_carrier_recv_completion_multishot(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags);

static iree_status_t iree_net_tcp_carrier_submit_multishot_recv(
    iree_net_tcp_carrier_t* carrier,
    iree_async_socket_recv_pool_operation_t* recv_op) {
  memset(recv_op, 0, sizeof(*recv_op));
  iree_async_operation_initialize(
      &recv_op->base, IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL,
      IREE_ASYNC_OPERATION_FLAG_MULTISHOT |
          IREE_ASYNC_OPERATION_FLAG_CANCELLATION_IS_SUCCESS,
      iree_net_tcp_carrier_recv_completion_multishot, carrier);
  recv_op->socket = carrier->socket;
  recv_op->pool = carrier->recv_pool;
  return iree_async_proactor_submit_one(carrier->proactor, &recv_op->base);
}

static void iree_net_tcp_carrier_recv_completion_multishot(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_tcp_carrier_t* carrier = (iree_net_tcp_carrier_t*)user_data;
  iree_async_socket_recv_pool_operation_t* recv_op =
      (iree_async_socket_recv_pool_operation_t*)operation;

  bool is_final = !iree_all_bits_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE);
  if (iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_CANCELLED)) {
    IREE_ASSERT(is_final, "cancelled multishot receive must be terminal");
    if (!iree_status_is_ok(status)) iree_status_abort(status);
    iree_async_buffer_lease_release(&recv_op->lease);
    iree_net_tcp_carrier_retire_pending_operation(carrier);
    return;
  }

  // Terminal resource exhaustion means the provided buffer ring was empty when
  // data arrived. Leave the logical receive pending and resume it when an
  // outstanding lease returns a buffer.
  iree_status_code_t status_code = iree_status_code(status);
  bool should_pause =
      is_final && status_code == IREE_STATUS_RESOURCE_EXHAUSTED &&
      iree_net_carrier_state(&carrier->base) == IREE_NET_CARRIER_STATE_ACTIVE;
  if (should_pause) {
    int32_t returned_buffer_epoch = iree_atomic_load(
        &carrier->recv.returned_buffer_epoch, iree_memory_order_acquire);
    iree_status_free(status);
    iree_net_tcp_carrier_pause_multishot_recv(carrier, returned_buffer_epoch);
    return;
  } else if (is_final && status_code == IREE_STATUS_RESOURCE_EXHAUSTED) {
    iree_status_free(status);
    iree_net_tcp_carrier_retire_pending_operation(carrier);
    return;
  }

  // Terminal deferred status means the kernel reported EAGAIN without
  // consuming data. Re-arm immediately; no buffer return will arrive to wake
  // us.
  bool should_rearm =
      is_final && status_code == IREE_STATUS_DEFERRED &&
      iree_net_carrier_state(&carrier->base) == IREE_NET_CARRIER_STATE_ACTIVE;
  if (should_rearm) {
    iree_status_free(status);
    iree_status_t rearm_status =
        iree_net_tcp_carrier_submit_multishot_recv(carrier, recv_op);
    if (!iree_status_is_ok(rearm_status)) {
      iree_net_tcp_carrier_set_failure_status(carrier, rearm_status);
      iree_net_tcp_carrier_retire_pending_operation(carrier);
    }
    return;
  }

  // Save status info before process_recv. process_recv takes ownership of
  // |status| when it is non-OK (passes to set_failure_status), so |status|
  // must not be read after this call.
  bool status_ok = iree_status_is_ok(status);

  // Process the received data.
  bool continue_recv = iree_net_tcp_carrier_process_recv(
      carrier, status, recv_op->bytes_received, &recv_op->lease);
  // NOTE: |status| is consumed and must not be used after this point.

  // Release the lease (we've either copied the data or passed it to handler).
  iree_async_buffer_lease_release(&recv_op->lease);

  // Update statistics on success.
  if (status_ok && recv_op->bytes_received > 0) {
    iree_atomic_fetch_add(&carrier->base.bytes_received,
                          (int64_t)recv_op->bytes_received,
                          iree_memory_order_relaxed);
  }

  if (is_final) {
    // Permanent termination (connection reset, EOF, cancelled, etc.).
    iree_net_tcp_carrier_retire_pending_operation(carrier);
  } else if (!continue_recv) {
    // Handler signaled stop (error or application request) while the multishot
    // operation is still armed. Its terminal cancellation completion owns the
    // pending-operation retirement.
    iree_net_tcp_carrier_cancel_operation(carrier, &recv_op->base);
  }
}

// Completion callback for single-shot recv operations.
// Re-submits the operation after processing unless stopping.
static void iree_net_tcp_carrier_recv_completion_single_shot(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_tcp_carrier_t* carrier = (iree_net_tcp_carrier_t*)user_data;
  iree_net_tcp_recv_slot_t* slot = (iree_net_tcp_recv_slot_t*)operation;
  iree_async_socket_recv_pool_operation_t* recv_op =
      (iree_async_socket_recv_pool_operation_t*)operation;

  if (iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_CANCELLED)) {
    if (!iree_status_is_ok(status)) iree_status_abort(status);
    iree_async_buffer_lease_release(&recv_op->lease);
    iree_net_tcp_carrier_terminate_single_shot_recv(carrier, slot);
    return;
  }

  // Pool exhaustion is receive-side backpressure, not a stream failure. Leave
  // this logical receive pending and resume it when a lease returns a buffer.
  iree_status_code_t status_code = iree_status_code(status);
  bool should_pause =
      status_code == IREE_STATUS_RESOURCE_EXHAUSTED &&
      iree_net_carrier_state(&carrier->base) == IREE_NET_CARRIER_STATE_ACTIVE;
  if (should_pause) {
    int32_t returned_buffer_epoch = iree_atomic_load(
        &carrier->recv.returned_buffer_epoch, iree_memory_order_acquire);
    iree_status_free(status);
    iree_net_tcp_carrier_pause_single_shot_recv(carrier, slot,
                                                returned_buffer_epoch);
    return;
  } else if (status_code == IREE_STATUS_RESOURCE_EXHAUSTED) {
    iree_status_free(status);
    iree_net_tcp_carrier_terminate_single_shot_recv(carrier, slot);
    return;
  }

  // Save status info before process_recv. process_recv takes ownership of
  // |status| when it is non-OK (passes to set_failure_status), so |status|
  // must not be read after this call.
  bool status_ok = iree_status_is_ok(status);

  // Process the received data.
  bool continue_recv = iree_net_tcp_carrier_process_recv(
      carrier, status, recv_op->bytes_received, &recv_op->lease);
  // NOTE: |status| is consumed and must not be used after this point.

  // Release the lease.
  iree_async_buffer_lease_release(&recv_op->lease);

  // Update statistics on success.
  if (status_ok && recv_op->bytes_received > 0) {
    iree_atomic_fetch_add(&carrier->base.bytes_received,
                          (int64_t)recv_op->bytes_received,
                          iree_memory_order_relaxed);
  }

  if (continue_recv) {
    // Re-submit the recv operation.
    int32_t returned_buffer_epoch = iree_atomic_load(
        &carrier->recv.returned_buffer_epoch, iree_memory_order_acquire);
    iree_status_t submit_status =
        iree_net_tcp_carrier_submit_single_shot_recv(carrier, slot);
    if (!iree_status_is_ok(submit_status)) {
      if (iree_status_code(submit_status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
        iree_status_free(submit_status);
        iree_net_tcp_carrier_pause_single_shot_recv(carrier, slot,
                                                    returned_buffer_epoch);
      } else {
        iree_net_tcp_carrier_set_failure_status(carrier, submit_status);
        iree_net_tcp_carrier_terminate_single_shot_recv(carrier, slot);
      }
    }
    // On successful resubmit, pending_operations stays the same.
  } else {
    iree_net_tcp_carrier_terminate_single_shot_recv(carrier, slot);
  }
}

// Submits a single-shot recv operation (implementation).
static iree_status_t iree_net_tcp_carrier_submit_single_shot_recv(
    iree_net_tcp_carrier_t* carrier, iree_net_tcp_recv_slot_t* slot) {
  iree_async_socket_recv_pool_operation_t* recv_op = &slot->operation;

  // Initialize the operation for reuse.
  memset(recv_op, 0, sizeof(*recv_op));
  iree_async_operation_initialize(
      &recv_op->base, IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL,
      IREE_ASYNC_OPERATION_FLAG_CANCELLATION_IS_SUCCESS,
      iree_net_tcp_carrier_recv_completion_single_shot, carrier);
  recv_op->socket = carrier->socket;
  recv_op->pool = carrier->recv_pool;

  return iree_async_proactor_submit_one(carrier->proactor, &recv_op->base);
}

//===----------------------------------------------------------------------===//
// Send completion handler
//===----------------------------------------------------------------------===//

static uint32_t iree_net_tcp_send_slot_index(iree_net_tcp_carrier_t* carrier,
                                             iree_net_tcp_send_slot_t* slot) {
  return (uint32_t)(slot - carrier->send.slots);
}

static void iree_net_tcp_carrier_complete_send_slot(
    iree_net_tcp_carrier_t* carrier, iree_net_tcp_send_slot_t* slot,
    iree_status_t status, iree_host_size_t bytes_sent) {
  // Update statistics on success.
  if (iree_status_is_ok(status)) {
    iree_atomic_fetch_add(&carrier->base.bytes_sent, (int64_t)bytes_sent,
                          iree_memory_order_relaxed);
  } else {
    // Capture send errors as sticky failure status.
    iree_net_tcp_carrier_set_failure_status(carrier, iree_status_clone(status));
  }

  // Free begin_send buffer if this was a begin_send/commit_send operation.
  // Save the flag before clearing so we can skip the user callback below.
  bool is_begin_send = slot->begin_send_buffer != NULL;
  if (is_begin_send) {
    iree_allocator_free(carrier->base.host_allocator, slot->begin_send_buffer);
    slot->begin_send_buffer = NULL;
    slot->begin_send_size = 0;
  }

  if (iree_any_bit_set(slot->flags,
                       IREE_NET_TCP_SEND_SLOT_FLAG_QUEUED_RESOURCES_RETAINED)) {
    iree_async_span_list_release_regions(slot->operation.buffers);
    iree_async_operation_release_resources(&slot->operation.base);
    slot->flags &= ~IREE_NET_TCP_SEND_SLOT_FLAG_QUEUED_RESOURCES_RETAINED;
  }

  // Invoke user callback if set. begin_send/commit_send operations do not fire
  // user callbacks (the data is fully consumed on commit).
  if (!is_begin_send && carrier->base.callback.fn) {
    carrier->base.callback.fn(carrier->base.callback.user_data,
                              IREE_NET_CARRIER_COMPLETION_SEND, slot->user_data,
                              status, bytes_sent, NULL);
  } else {
    // No callback or begin_send operation - we must consume the status.
    iree_status_free(status);
  }
  slot->user_data = 0;
  slot->pending_next = NULL;
  slot->flags = IREE_NET_TCP_SEND_SLOT_FLAG_NONE;

  // Release the slot back to the free bitmap.
  // Release ordering ensures all slot cleanup (buffer free, field zeroing) is
  // visible to threads that acquire-load the bitmap in begin_send/send.
  uint32_t slot_index = iree_net_tcp_send_slot_index(carrier, slot);
  iree_atomic_fetch_or(&carrier->send.free_bitmap, (uint32_t)1 << slot_index,
                       iree_memory_order_release);

  iree_net_tcp_carrier_retire_pending_operation(carrier);
}

static iree_status_t iree_net_tcp_carrier_submit_send_slot(
    iree_net_tcp_carrier_t* carrier, iree_net_tcp_send_slot_t* slot) {
  return iree_async_proactor_submit_one(carrier->proactor,
                                        &slot->operation.base);
}

static void iree_net_tcp_carrier_enqueue_send_slot_locked(
    iree_net_tcp_carrier_t* carrier, iree_net_tcp_send_slot_t* slot) {
  slot->pending_next = NULL;
  iree_async_operation_retain_resources(&slot->operation.base);
  iree_async_span_list_retain_regions(slot->operation.buffers);
  slot->flags |= IREE_NET_TCP_SEND_SLOT_FLAG_QUEUED_RESOURCES_RETAINED;
  if (carrier->send.pending_tail) {
    carrier->send.pending_tail->pending_next = slot;
  } else {
    carrier->send.pending_head = slot;
  }
  carrier->send.pending_tail = slot;
}

static iree_net_tcp_send_slot_t*
iree_net_tcp_carrier_pop_pending_send_slot_locked(
    iree_net_tcp_carrier_t* carrier) {
  iree_net_tcp_send_slot_t* slot = carrier->send.pending_head;
  if (slot) {
    carrier->send.pending_head = slot->pending_next;
    if (!carrier->send.pending_head) {
      carrier->send.pending_tail = NULL;
    }
    slot->pending_next = NULL;
  }
  return slot;
}

static iree_status_t iree_net_tcp_carrier_enqueue_or_submit_send_slot(
    iree_net_tcp_carrier_t* carrier, iree_net_tcp_send_slot_t* slot) {
  iree_slim_mutex_lock(&carrier->send.pending_mutex);

  iree_status_t status = iree_ok_status();
  if (carrier->send.pending_head) {
    iree_net_tcp_carrier_enqueue_send_slot_locked(carrier, slot);
  } else {
    status = iree_net_tcp_carrier_submit_send_slot(carrier, slot);
    if (iree_status_is_ok(status)) {
      ++carrier->send.submitted_count;
    } else if (iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED &&
               carrier->send.submitted_count > 0) {
      iree_status_free(status);
      iree_net_tcp_carrier_enqueue_send_slot_locked(carrier, slot);
      status = iree_ok_status();
    }
  }

  iree_slim_mutex_unlock(&carrier->send.pending_mutex);
  return status;
}

static void iree_net_tcp_carrier_flush_pending_send_slots(
    iree_net_tcp_carrier_t* carrier) {
  for (;;) {
    iree_slim_mutex_lock(&carrier->send.pending_mutex);
    iree_net_tcp_send_slot_t* slot = carrier->send.pending_head;
    if (!slot) {
      iree_slim_mutex_unlock(&carrier->send.pending_mutex);
      return;
    }

    iree_status_t status = iree_net_tcp_carrier_submit_send_slot(carrier, slot);
    if (iree_status_is_ok(status)) {
      (void)iree_net_tcp_carrier_pop_pending_send_slot_locked(carrier);
      ++carrier->send.submitted_count;
      iree_slim_mutex_unlock(&carrier->send.pending_mutex);
    } else if (iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
      if (carrier->send.submitted_count > 0) {
        // Submission capacity is still exhausted. The head slot remains queued
        // and another send completion will retry it.
        iree_status_free(status);
        iree_slim_mutex_unlock(&carrier->send.pending_mutex);
        return;
      }
      // There is no carrier-owned send completion left to trigger a retry.
      // Fail the accepted slot instead of leaving deactivation permanently
      // blocked behind a queued operation that nobody can service.
      (void)iree_net_tcp_carrier_pop_pending_send_slot_locked(carrier);
      iree_slim_mutex_unlock(&carrier->send.pending_mutex);
      iree_net_tcp_carrier_complete_send_slot(carrier, slot, status, 0);
    } else {
      (void)iree_net_tcp_carrier_pop_pending_send_slot_locked(carrier);
      iree_slim_mutex_unlock(&carrier->send.pending_mutex);
      iree_net_tcp_carrier_complete_send_slot(carrier, slot, status, 0);
    }
  }
}

// Completion callback for send operations.
static void iree_net_tcp_carrier_send_completion(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)flags;
  iree_net_tcp_carrier_t* carrier = (iree_net_tcp_carrier_t*)user_data;
  iree_async_socket_send_operation_t* send_op =
      (iree_async_socket_send_operation_t*)operation;

  // The slot's operation is the first field.
  iree_net_tcp_send_slot_t* slot = (iree_net_tcp_send_slot_t*)send_op;

  iree_slim_mutex_lock(&carrier->send.pending_mutex);
  IREE_ASSERT(carrier->send.submitted_count > 0);
  --carrier->send.submitted_count;
  iree_slim_mutex_unlock(&carrier->send.pending_mutex);

  iree_net_tcp_carrier_flush_pending_send_slots(carrier);
  iree_net_tcp_carrier_complete_send_slot(carrier, slot, status,
                                          send_op->bytes_sent);
}

//===----------------------------------------------------------------------===//
// Vtable implementations
//===----------------------------------------------------------------------===//

static iree_status_t iree_net_tcp_carrier_deactivate(
    iree_net_carrier_t* base_carrier,
    iree_net_carrier_deactivate_callback_fn_t callback, void* user_data);

// Final free after all io_uring operations have been cancelled and completed.
static void iree_net_tcp_carrier_free(iree_net_tcp_carrier_t* carrier) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Consume sticky failure status.
  iree_net_tcp_carrier_consume_failure_status(carrier);

  // Free any outstanding begin_send buffers. A slot may hold an allocated
  // buffer if begin_send was called but commit_send/abort_send never completed
  // (e.g. carrier destroyed during concurrent sends or after deactivation
  // cancelled in-flight operations).
  for (uint32_t i = 0; i < carrier->send.slot_count; ++i) {
    if (carrier->send.slots[i].begin_send_buffer) {
      iree_allocator_free(carrier->base.host_allocator,
                          carrier->send.slots[i].begin_send_buffer);
    }
  }

  // Release socket - null-safe.
  iree_async_socket_release(carrier->socket);

  // NOTE: recv_pool is not owned by carrier. Caller must ensure it outlives us.

  iree_slim_mutex_deinitialize(&carrier->send.pending_mutex);

  // Free carrier memory.
  iree_allocator_t allocator = carrier->base.host_allocator;
  iree_allocator_free(allocator, carrier);
  IREE_TRACE_ZONE_END(z0);
}

// Deactivation callback for deferred destroy: called when all pending io_uring
// operations have completed their cancellation CQEs.
static void iree_net_tcp_carrier_deferred_destroy(void* user_data) {
  iree_net_tcp_carrier_t* carrier = (iree_net_tcp_carrier_t*)user_data;
  iree_net_tcp_carrier_free(carrier);
}

static void iree_net_tcp_carrier_destroy(iree_net_carrier_t* base_carrier) {
  iree_net_tcp_carrier_t* carrier = iree_net_tcp_carrier_cast(base_carrier);

  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);

  if (state == IREE_NET_CARRIER_STATE_ACTIVE) {
    // Carrier was never properly deactivated (caller released without
    // shutting down). Start async deactivation to cancel pending io_uring
    // operations (multishot recv, in-flight sends). The carrier stays alive
    // until all cancellation CQEs have been processed, at which point
    // iree_net_tcp_carrier_deferred_destroy fires the actual free.
    //
    // Clear recv handler to prevent delivery of stale data during drain.
    base_carrier->recv_handler.fn = NULL;
    base_carrier->recv_handler.user_data = NULL;
    iree_status_t status = iree_net_tcp_carrier_deactivate(
        base_carrier, iree_net_tcp_carrier_deferred_destroy, carrier);
    if (!iree_status_is_ok(status)) iree_status_abort(status);
    return;
  }

  if (state == IREE_NET_CARRIER_STATE_DRAINING) {
    // Already deactivating — replace the callback so we get notified when
    // draining completes. The previous callback holder has already released.
    carrier->deactivate_callback.fn = iree_net_tcp_carrier_deferred_destroy;
    carrier->deactivate_callback.user_data = carrier;
    return;
  }

  // DEACTIVATED or CREATED — safe to free immediately.
  IREE_ASSERT(state == IREE_NET_CARRIER_STATE_DEACTIVATED ||
              state == IREE_NET_CARRIER_STATE_CREATED);
  iree_net_tcp_carrier_free(carrier);
}

static void iree_net_tcp_carrier_set_recv_handler(
    iree_net_carrier_t* base_carrier, iree_net_carrier_recv_handler_t handler) {
  base_carrier->recv_handler = handler;
}

static iree_status_t iree_net_tcp_carrier_activate(
    iree_net_carrier_t* base_carrier) {
  iree_net_tcp_carrier_t* carrier = iree_net_tcp_carrier_cast(base_carrier);
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

  iree_status_t status = iree_ok_status();

  if (carrier->recv.multishot_enabled) {
    // Multishot: submit single recv_pool operation with MULTISHOT flag.
    iree_async_socket_recv_pool_operation_t* recv_op =
        &carrier->recv.multishot.operation;

    // Track pending operation before submit.
    iree_atomic_fetch_add(&base_carrier->pending_operations, 1,
                          iree_memory_order_relaxed);

    status = iree_net_tcp_carrier_submit_multishot_recv(carrier, recv_op);
    if (!iree_status_is_ok(status)) {
      // Rollback pending_operations on submit failure.
      iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                            iree_memory_order_relaxed);
    }
  } else {
    // Single-shot: submit all recv operations.
    uint32_t slot_count = carrier->recv.single_shot.slot_count;
    for (uint32_t i = 0; i < slot_count && iree_status_is_ok(status); ++i) {
      iree_net_tcp_recv_slot_t* slot = &carrier->recv.single_shot.slots[i];

      // Track pending operation before submit.
      iree_atomic_fetch_add(&base_carrier->pending_operations, 1,
                            iree_memory_order_relaxed);
      iree_atomic_fetch_add(&carrier->recv.single_shot.live_count, 1,
                            iree_memory_order_relaxed);
      iree_atomic_store(&slot->state, IREE_NET_TCP_RECV_SLOT_SUBMITTED,
                        iree_memory_order_relaxed);

      status = iree_net_tcp_carrier_submit_single_shot_recv(carrier, slot);
      if (!iree_status_is_ok(status)) {
        // Rollback counts on submit failure.
        iree_atomic_store(&slot->state, IREE_NET_TCP_RECV_SLOT_RETIRED,
                          iree_memory_order_relaxed);
        iree_atomic_fetch_sub(&carrier->recv.single_shot.live_count, 1,
                              iree_memory_order_relaxed);
        iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                              iree_memory_order_relaxed);
      }
    }
  }

  // On failure, handle partial activation state.
  if (!iree_status_is_ok(status)) {
    int32_t pending = iree_atomic_load(&base_carrier->pending_operations,
                                       iree_memory_order_acquire);
    if (pending > 0) {
      // Some operations were successfully submitted before failure.
      // Stay in ACTIVE state and shut down socket to trigger their completion.
      // Caller MUST call deactivate() before destroy() to drain operations.
      // Set sticky failure status so subsequent send() calls fail fast.
      iree_net_tcp_carrier_set_failure_status(carrier,
                                              iree_status_clone(status));
      status = iree_status_join(
          status, iree_async_socket_shutdown(carrier->socket,
                                             IREE_ASYNC_SOCKET_SHUTDOWN_READ));
    } else {
      // No operations were submitted - safe to reset to CREATED.
      iree_net_carrier_set_state(base_carrier, IREE_NET_CARRIER_STATE_CREATED);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_net_tcp_carrier_deactivate(
    iree_net_carrier_t* base_carrier,
    iree_net_carrier_deactivate_callback_fn_t callback, void* user_data) {
  iree_net_tcp_carrier_t* carrier = iree_net_tcp_carrier_cast(base_carrier);
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

  // Store callback for when deactivation completes.
  carrier->deactivate_callback.fn = callback;
  carrier->deactivate_callback.user_data = user_data;

  // If never activated, skip directly to DEACTIVATED (no async work to drain).
  if (state == IREE_NET_CARRIER_STATE_CREATED) {
    iree_net_carrier_set_state(base_carrier,
                               IREE_NET_CARRIER_STATE_DEACTIVATED);
    if (callback) {
      callback(user_data);
    }
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Keep one pending count owned by this call until state publication and
  // cancellation submission are complete. The final retirement is then the
  // only path allowed to deliver the terminal callback.
  iree_atomic_fetch_add(&base_carrier->pending_operations, 1,
                        iree_memory_order_acq_rel);

  // Transition to DRAINING.
  iree_net_carrier_set_state(base_carrier, IREE_NET_CARRIER_STATE_DRAINING);

  // Explicitly cancel pending receive operations. Their terminal callbacks own
  // pending-operation retirement; NOT_FOUND means a completion won the race.
  if (carrier->recv.multishot_enabled) {
    int32_t expected = 1;
    bool was_paused = iree_atomic_compare_exchange_strong(
        &carrier->recv.multishot.paused, &expected, 0,
        iree_memory_order_acq_rel, iree_memory_order_acquire);
    if (was_paused) {
      iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                            iree_memory_order_release);
    } else {
      iree_net_tcp_carrier_cancel_operation(
          carrier, &carrier->recv.multishot.operation.base);
    }
  } else {
    uint32_t slot_count = carrier->recv.single_shot.slot_count;
    for (uint32_t i = 0; i < slot_count; ++i) {
      iree_net_tcp_recv_slot_t* slot = &carrier->recv.single_shot.slots[i];
      int32_t expected = IREE_NET_TCP_RECV_SLOT_PAUSED;
      bool was_paused = iree_atomic_compare_exchange_strong(
          &slot->state, &expected, IREE_NET_TCP_RECV_SLOT_RETIRED,
          iree_memory_order_acq_rel, iree_memory_order_acquire);
      if (was_paused) {
        iree_atomic_fetch_sub(&carrier->recv.single_shot.paused_count, 1,
                              iree_memory_order_release);
        iree_atomic_fetch_sub(&carrier->recv.single_shot.live_count, 1,
                              iree_memory_order_release);
        iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                              iree_memory_order_release);
      } else if (iree_atomic_load(&slot->state, iree_memory_order_acquire) ==
                 IREE_NET_TCP_RECV_SLOT_SUBMITTED) {
        iree_net_tcp_carrier_cancel_operation(carrier, &slot->operation.base);
      }
    }
  }

  IREE_TRACE_ZONE_END(z0);
  iree_net_tcp_carrier_retire_pending_operation(carrier);
  return iree_ok_status();
}

static iree_net_carrier_send_budget_t iree_net_tcp_carrier_query_send_budget(
    iree_net_carrier_t* base_carrier) {
  iree_net_tcp_carrier_t* carrier = iree_net_tcp_carrier_cast(base_carrier);

  // Count free slots via popcount on the bitmap.
  uint32_t bitmap =
      iree_atomic_load(&carrier->send.free_bitmap, iree_memory_order_acquire);
  uint32_t available = iree_math_count_ones_u32(bitmap);

  iree_net_carrier_send_budget_t budget;
  budget.slots = available;
  // For TCP, byte budget is effectively unlimited (kernel handles buffering).
  // Use SIZE_MAX to indicate "no byte limit, limited by slots only".
  budget.bytes = IREE_HOST_SIZE_MAX;
  return budget;
}

static iree_status_t iree_net_tcp_carrier_send(
    iree_net_carrier_t* base_carrier, const iree_net_send_params_t* params) {
  iree_net_tcp_carrier_t* carrier = iree_net_tcp_carrier_cast(base_carrier);

  // Validate scatter-gather count early (doesn't need pending_operations).
  if (params->data.count > IREE_ASYNC_SOCKET_SEND_MAX_BUFFERS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "too many scatter-gather buffers: %" PRIhsz " > %d",
                            params->data.count,
                            IREE_ASYNC_SOCKET_SEND_MAX_BUFFERS);
  }

  // Calculate total size and reject empty sends.
  iree_host_size_t total_size = 0;
  for (iree_host_size_t i = 0; i < params->data.count; ++i) {
    total_size += params->data.values[i].length;
  }
  if (total_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty sends are not allowed");
  }

  // Increment pending_operations FIRST to prevent TOCTOU race with deactivate.
  // This ensures deactivate sees our operation before completing.
  iree_atomic_fetch_add(&base_carrier->pending_operations, 1,
                        iree_memory_order_acq_rel);

  // Now verify state is ACTIVE. If not, rollback and return error.
  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE) {
    iree_net_tcp_carrier_retire_pending_operation(carrier);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier must be in ACTIVE state to send");
  }

  // Check for sticky failure status.
  iree_status_t failure = iree_net_tcp_carrier_get_failure_status(carrier);
  if (!iree_status_is_ok(failure)) {
    iree_net_tcp_carrier_retire_pending_operation(carrier);
    return failure;
  }

  // Claim a send slot using bitmap CAS loop.
  // Find first free slot (set bit), CAS-clear it to claim.
  uint32_t slot_index;
  uint32_t bitmap =
      iree_atomic_load(&carrier->send.free_bitmap, iree_memory_order_acquire);
  for (;;) {
    if (bitmap == 0) {
      // No free slots available.
      iree_net_tcp_carrier_retire_pending_operation(carrier);
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "no send slots available");
    }
    slot_index = (uint32_t)iree_math_count_trailing_zeros_u32(bitmap);
    uint32_t cleared = bitmap & ~((uint32_t)1 << slot_index);
    if (iree_atomic_compare_exchange_weak(&carrier->send.free_bitmap, &bitmap,
                                          cleared, iree_memory_order_acq_rel,
                                          iree_memory_order_acquire)) {
      break;
    }
  }

  // Get the slot and initialize the operation.
  iree_net_tcp_send_slot_t* slot = &carrier->send.slots[slot_index];

  // Store user data for completion callback.
  slot->user_data = params->user_data;

  // Initialize the send operation. Uses iree_async_operation_zero instead of
  // memset to avoid non-atomic writes to the atomic fields in the base struct.
  iree_async_socket_send_operation_t* send_op = &slot->operation;
  iree_async_operation_zero(&send_op->base, sizeof(*send_op));
  iree_async_operation_initialize(
      &send_op->base, IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_tcp_carrier_send_completion,
      carrier);
  send_op->socket = carrier->socket;
  send_op->send_flags = IREE_ASYNC_SOCKET_SEND_FLAG_NONE;

  // Copy span list into slot-local storage. The caller's span array and the
  // data behind unregistered spans may be stack-allocated. Under io_uring,
  // the kernel reads iovec data at io_uring_enter time (during poll), which
  // is after the caller returns. We must ensure all references are stable.
  memcpy(slot->inline_spans, params->data.values,
         params->data.count * sizeof(iree_async_span_t));
  send_op->buffers.values = slot->inline_spans;
  send_op->buffers.count = params->data.count;

  // Copy small unregistered span data into slot-local inline storage.
  // Unregistered spans (region == NULL) store raw pointers via
  // iree_async_span_from_ptr that may reference transient caller memory.
  // Registered spans have managed lifetime through the region and are
  // left as-is.
  iree_host_size_t inline_data_offset = 0;
  for (iree_host_size_t i = 0; i < send_op->buffers.count; ++i) {
    if (slot->inline_spans[i].region != NULL) continue;
    iree_host_size_t length = slot->inline_spans[i].length;
    if (length == 0) continue;
    if (inline_data_offset + length >
        IREE_NET_TCP_SEND_SLOT_INLINE_DATA_CAPACITY) {
      // Data exceeds inline capacity. The caller is responsible for keeping
      // unregistered data alive until send completion. This is only safe when
      // the data is in stable memory (heap, static, or a stack frame that
      // won't unwind before poll). Large stack-allocated sends are a bug.
      break;
    }
    void* source = iree_async_span_ptr(slot->inline_spans[i]);
    memcpy(slot->inline_data + inline_data_offset, source, length);
    slot->inline_spans[i] = iree_async_span_from_ptr(
        slot->inline_data + inline_data_offset, length);
    inline_data_offset += length;
  }

  iree_status_t status =
      iree_net_tcp_carrier_enqueue_or_submit_send_slot(carrier, slot);

  if (!iree_status_is_ok(status)) {
    // Rollback: release slot back to free bitmap.
    iree_atomic_fetch_or(&carrier->send.free_bitmap, (uint32_t)1 << slot_index,
                         iree_memory_order_release);
    iree_net_tcp_carrier_retire_pending_operation(carrier);
    return status;
  }

  return iree_ok_status();
}

static iree_status_t iree_net_tcp_carrier_begin_send(
    iree_net_carrier_t* base_carrier, iree_host_size_t size, void** out_ptr,
    iree_net_carrier_send_handle_t* out_handle) {
  iree_net_tcp_carrier_t* carrier = iree_net_tcp_carrier_cast(base_carrier);

  if (size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty sends are not allowed");
  }

  // Increment pending_operations FIRST to prevent TOCTOU race with deactivate.
  iree_atomic_fetch_add(&base_carrier->pending_operations, 1,
                        iree_memory_order_acq_rel);

  // Verify state is ACTIVE. If not, rollback and return error.
  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE) {
    iree_net_tcp_carrier_retire_pending_operation(carrier);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier must be in ACTIVE state to send");
  }

  // Check for sticky failure status.
  iree_status_t failure = iree_net_tcp_carrier_get_failure_status(carrier);
  if (!iree_status_is_ok(failure)) {
    iree_net_tcp_carrier_retire_pending_operation(carrier);
    return failure;
  }

  // Claim a send slot using bitmap CAS loop.
  uint32_t slot_index;
  uint32_t bitmap =
      iree_atomic_load(&carrier->send.free_bitmap, iree_memory_order_acquire);
  for (;;) {
    if (bitmap == 0) {
      iree_net_tcp_carrier_retire_pending_operation(carrier);
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "no send slots available");
    }
    slot_index = (uint32_t)iree_math_count_trailing_zeros_u32(bitmap);
    uint32_t cleared = bitmap & ~((uint32_t)1 << slot_index);
    if (iree_atomic_compare_exchange_weak(&carrier->send.free_bitmap, &bitmap,
                                          cleared, iree_memory_order_acq_rel,
                                          iree_memory_order_acquire)) {
      break;
    }
  }

  // Get the slot.
  iree_net_tcp_send_slot_t* slot = &carrier->send.slots[slot_index];

  // Allocate the send buffer.
  void* buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(carrier->base.host_allocator, size, &buffer);
  if (!iree_status_is_ok(status)) {
    // Rollback: release slot back to free bitmap.
    iree_atomic_fetch_or(&carrier->send.free_bitmap, (uint32_t)1 << slot_index,
                         iree_memory_order_release);
    iree_net_tcp_carrier_retire_pending_operation(carrier);
    return status;
  }

  slot->begin_send_buffer = buffer;
  slot->begin_send_size = size;
  *out_ptr = buffer;
  *out_handle = (iree_net_carrier_send_handle_t)slot_index;
  return iree_ok_status();
}

static iree_status_t iree_net_tcp_carrier_commit_send(
    iree_net_carrier_t* base_carrier, iree_net_carrier_send_handle_t handle) {
  iree_net_tcp_carrier_t* carrier = iree_net_tcp_carrier_cast(base_carrier);

  uint32_t slot_index = (uint32_t)handle;

  iree_net_tcp_send_slot_t* slot = &carrier->send.slots[slot_index];
  iree_host_size_t size = slot->begin_send_size;
  slot->user_data = 0;

  // Initialize the send operation. Uses iree_async_operation_zero instead of
  // memset to avoid non-atomic writes to the atomic fields in the base struct.
  iree_async_socket_send_operation_t* send_op = &slot->operation;
  iree_async_operation_zero(&send_op->base, sizeof(*send_op));
  iree_async_operation_initialize(
      &send_op->base, IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_tcp_carrier_send_completion,
      carrier);
  send_op->socket = carrier->socket;

  // Set up single-buffer scatter-gather pointing to the begin_send buffer.
  // Uses inline_spans[0] since the slot already has persistent span storage.
  slot->inline_spans[0] =
      iree_async_span_from_ptr(slot->begin_send_buffer, size);
  send_op->buffers.values = slot->inline_spans;
  send_op->buffers.count = 1;
  send_op->send_flags = IREE_ASYNC_SOCKET_SEND_FLAG_NONE;

  iree_status_t status =
      iree_net_tcp_carrier_enqueue_or_submit_send_slot(carrier, slot);

  if (!iree_status_is_ok(status)) {
    // Rollback: free buffer, release slot back to free bitmap.
    iree_allocator_free(carrier->base.host_allocator, slot->begin_send_buffer);
    slot->begin_send_buffer = NULL;
    iree_atomic_fetch_or(&carrier->send.free_bitmap, (uint32_t)1 << slot_index,
                         iree_memory_order_release);
    iree_net_tcp_carrier_retire_pending_operation(carrier);
    return status;
  }

  return iree_ok_status();
}

static void iree_net_tcp_carrier_abort_send(
    iree_net_carrier_t* base_carrier, iree_net_carrier_send_handle_t handle) {
  iree_net_tcp_carrier_t* carrier = iree_net_tcp_carrier_cast(base_carrier);

  uint32_t slot_index = (uint32_t)handle;

  iree_net_tcp_send_slot_t* slot = &carrier->send.slots[slot_index];

  // Free the allocated buffer.
  iree_allocator_free(carrier->base.host_allocator, slot->begin_send_buffer);
  slot->begin_send_buffer = NULL;
  slot->begin_send_size = 0;

  // Release the slot back to the free bitmap.
  iree_atomic_fetch_or(&carrier->send.free_bitmap, (uint32_t)1 << slot_index,
                       iree_memory_order_release);

  iree_net_tcp_carrier_retire_pending_operation(carrier);
}

static iree_status_t iree_net_tcp_carrier_shutdown(
    iree_net_carrier_t* base_carrier) {
  iree_net_tcp_carrier_t* carrier = iree_net_tcp_carrier_cast(base_carrier);

  // Shutdown is valid in ACTIVE or DRAINING state.
  // In ACTIVE: normal graceful close initiation.
  // In DRAINING: shutdown write side while waiting for recv to drain.
  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE &&
      state != IREE_NET_CARRIER_STATE_DRAINING) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier must be in ACTIVE or DRAINING state to "
                            "shutdown");
  }

  // Shut down the write side of the socket. This sends FIN to the peer,
  // causing their recv to return EOF. The carrier can still receive data
  // until deactivate is called.
  return iree_async_socket_shutdown(carrier->socket,
                                    IREE_ASYNC_SOCKET_SHUTDOWN_WRITE);
}

static iree_status_t iree_net_tcp_carrier_direct_write(
    iree_net_carrier_t* base_carrier,
    const iree_net_direct_write_params_t* params) {
  (void)base_carrier;
  (void)params;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "TCP carrier does not support direct_write");
}

static iree_status_t iree_net_tcp_carrier_direct_read(
    iree_net_carrier_t* base_carrier,
    const iree_net_direct_read_params_t* params) {
  (void)base_carrier;
  (void)params;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "TCP carrier does not support direct_read");
}

static iree_status_t iree_net_tcp_carrier_register_buffer(
    iree_net_carrier_t* base_carrier, iree_async_region_t* region,
    iree_net_remote_handle_t* out_handle) {
  (void)base_carrier;
  (void)region;
  (void)out_handle;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "TCP carrier does not support register_buffer");
}

static void iree_net_tcp_carrier_unregister_buffer(
    iree_net_carrier_t* base_carrier, iree_net_remote_handle_t handle) {
  (void)base_carrier;
  (void)handle;
}

// Vtable for TCP carrier. RDMA operations are not supported.
static const iree_net_carrier_vtable_t iree_net_tcp_carrier_vtable = {
    .destroy = iree_net_tcp_carrier_destroy,
    .set_recv_handler = iree_net_tcp_carrier_set_recv_handler,
    .activate = iree_net_tcp_carrier_activate,
    .deactivate = iree_net_tcp_carrier_deactivate,
    .query_send_budget = iree_net_tcp_carrier_query_send_budget,
    .send = iree_net_tcp_carrier_send,
    .begin_send = iree_net_tcp_carrier_begin_send,
    .commit_send = iree_net_tcp_carrier_commit_send,
    .abort_send = iree_net_tcp_carrier_abort_send,
    .shutdown = iree_net_tcp_carrier_shutdown,
    .direct_write = iree_net_tcp_carrier_direct_write,
    .direct_read = iree_net_tcp_carrier_direct_read,
    .register_buffer = iree_net_tcp_carrier_register_buffer,
    .unregister_buffer = iree_net_tcp_carrier_unregister_buffer,
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_net_tcp_carrier_create(
    iree_async_proactor_t* proactor, iree_async_socket_t* socket,
    iree_async_buffer_pool_t* recv_pool, iree_net_tcp_carrier_options_t options,
    iree_net_carrier_callback_t callback, iree_allocator_t host_allocator,
    iree_net_carrier_t** out_carrier) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(socket);
  IREE_ASSERT_ARGUMENT(recv_pool);
  IREE_ASSERT_ARGUMENT(out_carrier);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_carrier = NULL;

  // Validate options.
  if (!iree_is_power_of_two_uint64(options.send_slot_count)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "send_slot_count must be power of 2, got %" PRIu32,
                            options.send_slot_count);
  }
  if (options.send_slot_count > 32) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "send_slot_count must be at most 32, got %" PRIu32,
                            options.send_slot_count);
  }
  if (!iree_is_power_of_two_uint64(options.single_shot_recv_count)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "single_shot_recv_count must be power of 2, got %" PRIu32,
        options.single_shot_recv_count);
  }

  // Query proactor capabilities.
  iree_async_proactor_capabilities_t capabilities =
      iree_async_proactor_query_capabilities(proactor);
  bool use_multishot =
      options.prefer_multishot_recv &&
      iree_any_bit_set(capabilities, IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT);
  bool use_zero_copy_send =
      options.prefer_zero_copy_send &&
      iree_any_bit_set(capabilities,
                       IREE_ASYNC_PROACTOR_CAPABILITY_ZERO_COPY_SEND);
  iree_host_size_t recv_lease_context_count =
      iree_async_buffer_pool_capacity(recv_pool);
  if (!use_multishot &&
      options.single_shot_recv_count > recv_lease_context_count) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "single_shot_recv_count (%" PRIu32
        ") must not exceed recv pool capacity (%" PRIhsz ")",
        options.single_shot_recv_count, recv_lease_context_count);
  }

  // Compute allocation size with overflow checking.
  // For multishot, recv_slot_count is 0 so that array contributes nothing.
  iree_host_size_t recv_slot_count =
      use_multishot ? 0 : options.single_shot_recv_count;
  iree_host_size_t total_size = 0;
  iree_host_size_t send_slots_offset = 0;
  iree_host_size_t recv_slots_offset = 0;
  iree_host_size_t recv_lease_contexts_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(iree_net_tcp_carrier_t), &total_size,
              IREE_STRUCT_FIELD_ALIGNED(
                  options.send_slot_count, iree_net_tcp_send_slot_t,
                  iree_alignof(iree_net_tcp_send_slot_t), &send_slots_offset),
              IREE_STRUCT_FIELD_ALIGNED(
                  recv_slot_count, iree_net_tcp_recv_slot_t,
                  iree_alignof(iree_net_tcp_recv_slot_t), &recv_slots_offset),
              IREE_STRUCT_FIELD_ALIGNED(
                  recv_lease_context_count, iree_net_tcp_recv_lease_context_t,
                  iree_alignof(iree_net_tcp_recv_lease_context_t),
                  &recv_lease_contexts_offset)));

  // Allocate carrier.
  iree_net_tcp_carrier_t* carrier = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&carrier));
  memset(carrier, 0, total_size);

  // Compute capabilities.
  iree_net_carrier_capabilities_t carrier_capabilities =
      IREE_NET_CARRIER_CAPABILITY_RELIABLE |
      IREE_NET_CARRIER_CAPABILITY_ORDERED;
  if (use_zero_copy_send) {
    carrier_capabilities |= IREE_NET_CARRIER_CAPABILITY_ZERO_COPY_TX;
  }
  if (use_multishot) {
    carrier_capabilities |= IREE_NET_CARRIER_CAPABILITY_ZERO_COPY_RX;
  }

  // Initialize base carrier.
  iree_net_carrier_initialize(&iree_net_tcp_carrier_vtable,
                              carrier_capabilities,
                              /*mtu=*/0,  // Stream carrier, no MTU.
                              IREE_ASYNC_SOCKET_SEND_MAX_BUFFERS, callback,
                              host_allocator, &carrier->base);

  // Initialize TCP carrier fields.
  carrier->proactor = proactor;
  carrier->socket = socket;
  carrier->recv_pool = recv_pool;
  carrier->recv_lease_contexts =
      (iree_net_tcp_recv_lease_context_t*)((uint8_t*)carrier +
                                           recv_lease_contexts_offset);
  carrier->recv_lease_context_count = recv_lease_context_count;
  // NOTE: recv_pool is not ref-counted. Caller must ensure it outlives carrier.

  // Initialize send slot bitmap with all slots free.
  iree_slim_mutex_initialize(&carrier->send.pending_mutex);
  carrier->send.slot_count = options.send_slot_count;
  uint32_t all_free = (options.send_slot_count == 32)
                          ? UINT32_MAX
                          : ((uint32_t)1 << options.send_slot_count) - 1;
  iree_atomic_store(&carrier->send.free_bitmap, all_free,
                    iree_memory_order_relaxed);

  // Point to trailing send slot storage.
  carrier->send.slots =
      (iree_net_tcp_send_slot_t*)((uint8_t*)carrier + send_slots_offset);

  // Initialize recv operations.
  carrier->recv.multishot_enabled = use_multishot;
  iree_atomic_store(&carrier->recv.returned_buffer_epoch, 0,
                    iree_memory_order_relaxed);
  if (use_multishot) {
    iree_atomic_store(&carrier->recv.multishot.paused, 0,
                      iree_memory_order_relaxed);
  } else {
    carrier->recv.single_shot.slot_count = options.single_shot_recv_count;
    carrier->recv.single_shot.slots =
        (iree_net_tcp_recv_slot_t*)((uint8_t*)carrier + recv_slots_offset);
    iree_atomic_store(&carrier->recv.single_shot.live_count, 0,
                      iree_memory_order_relaxed);
    iree_atomic_store(&carrier->recv.single_shot.paused_count, 0,
                      iree_memory_order_relaxed);
  }

  // Initialize error handling.
  iree_atomic_store(&carrier->failure_status, 0, iree_memory_order_relaxed);

  // Initialize deactivate callback to empty.
  carrier->deactivate_callback.fn = NULL;
  carrier->deactivate_callback.user_data = NULL;

  *out_carrier = &carrier->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
