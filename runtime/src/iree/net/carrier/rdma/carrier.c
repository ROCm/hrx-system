// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/carrier.h"

#include <arpa/inet.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>

#include "iree/async/slab.h"
#include "iree/base/threading/mutex.h"
#include "iree/net/carrier/rdma/completion_queue.h"
#include "iree/net/carrier/rdma/credit_memory.h"
#include "iree/net/carrier/rdma/memory_window.h"
#include "iree/net/carrier/rdma/queue_pair.h"
#include "iree/net/carrier/rdma/receive_queue.h"
#include "iree/net/carrier/rdma/region.h"
#include "iree/net/carrier/rdma/send_reservation_table.h"
#include "iree/net/carrier/rdma/send_window.h"
#include "iree/net/carrier/rdma/sge.h"
#include "iree/net/carrier/rdma/work_request_table.h"

#define IREE_NET_RDMA_CARRIER_CREDIT_WAKE_INTERVAL_NS iree_make_duration_ms(1)

typedef uint16_t iree_net_rdma_carrier_flags_t;
enum iree_net_rdma_carrier_flag_bits_e {
  IREE_NET_RDMA_CARRIER_FLAG_OWNS_CONNECTION_ID = 1u << 0,
  IREE_NET_RDMA_CARRIER_FLAG_REMOTE_CONNECTION_DATA_APPLIED = 1u << 1,
  IREE_NET_RDMA_CARRIER_FLAG_QUEUE_PAIR_ERROR_REQUESTED = 1u << 2,
  IREE_NET_RDMA_CARRIER_FLAG_CREDIT_GRANT_IN_FLIGHT = 1u << 3,
  IREE_NET_RDMA_CARRIER_FLAG_TERMINAL_FAILURE_HANDLED = 1u << 4,
  IREE_NET_RDMA_CARRIER_FLAG_OWNS_RECV_POOL = 1u << 5,
  IREE_NET_RDMA_CARRIER_FLAG_CREDIT_WAKE_TIMER_IN_FLIGHT = 1u << 6,
  IREE_NET_RDMA_CARRIER_FLAG_SHUTDOWN_REQUESTED = 1u << 7,
  IREE_NET_RDMA_CARRIER_FLAG_DEACTIVATE_CALLBACK_DEFERRED = 1u << 8,
  IREE_NET_RDMA_CARRIER_FLAG_RECV_REPLENISH_DEFERRED = 1u << 9,
  IREE_NET_RDMA_CARRIER_FLAG_CREDIT_WAKE_TIMER_CANCEL_REQUESTED = 1u << 10,
};

typedef struct iree_net_rdma_carrier_memory_export_t
    iree_net_rdma_carrier_memory_export_t;
struct iree_net_rdma_carrier_t {
  // Base carrier; must be first for upcasting.
  iree_net_carrier_t base;

  // RDMA context retained by the carrier.
  iree_net_rdma_context_t* context;

  // Proactor retained while completion queues are registered.
  iree_async_proactor_t* proactor;

  // Receive buffer pool supplying posted receives; borrowed unless owned.
  iree_async_buffer_pool_t* recv_pool;

  // Serializes native QP posting, fixed queue bookkeeping, peer connection
  // data, and mutable flags after the carrier is published.
  iree_slim_mutex_t queue_mutex;

  // Serializes exported memory-window handle bookkeeping.
  iree_slim_mutex_t memory_export_mutex;

  // Callback invoked after DRAINING retires all pending operations.
  iree_net_carrier_deactivate_callback_fn_t deactivate_callback;

  // Opaque user data passed to deactivate_callback.
  void* deactivate_user_data;

  // NOP used to defer deactivation callbacks out of CQ callback stacks.
  iree_async_nop_operation_t deactivate_operation;

  // NOP used to replenish receives after an externally held lease returns.
  iree_async_nop_operation_t recv_replenish_operation;

  // Owned RDMA-registered pool for staging non-RDMA CPU send spans.
  iree_async_buffer_pool_t* send_staging_pool;

  // Active memory-window exports returned from register_buffer.
  iree_net_rdma_carrier_memory_export_t* memory_exports;

  // Reservations and pending-post FIFO for staged SEND leases.
  iree_net_rdma_send_reservation_table_t send_reservation_table;

  // rdma_cm ID owned after successful carrier creation.
  struct rdma_cm_id* connection_id;

  // Send completion queue owned by this carrier.
  iree_net_rdma_completion_queue_t* send_completion_queue;

  // Receive completion queue owned by this carrier.
  iree_net_rdma_completion_queue_t* recv_completion_queue;

  // Queue pair attached to connection_id.
  iree_net_rdma_queue_pair_t queue_pair;

  // WR correlation table shared across send and receive completions.
  iree_net_rdma_work_request_table_t work_request_table;

  // Receive WQE posting state.
  iree_net_rdma_receive_queue_t receive_queue;

  // Send SQ and peer receive-credit admission state.
  iree_net_rdma_send_window_t send_window;

  // Peer-writable memory where remote receive-credit grants arrive.
  iree_net_rdma_credit_memory_t* credit_memory;

  // Local registered scratch used as the source for receive-credit grants.
  iree_net_rdma_credit_memory_t* credit_grant_memory;

  // Cumulative local receive credits made available by posted receive WQEs.
  uint32_t local_recv_credit_limit;

  // Cumulative local receive credits most recently submitted to the peer.
  uint32_t local_recv_credit_submitted;

  // Cumulative local receive credits known complete and visible to the peer.
  uint32_t local_recv_credit_published;

  // Proactor timers used to re-check peer-written credit memory after stalls.
  iree_async_timer_operation_t credit_wake_timers[2];

  // Credit-wake timer currently owned by the proactor, or NULL when idle.
  // Protected by |queue_mutex| and counted in |base.pending_operations|.
  iree_async_timer_operation_t* credit_wake_timer;

  // Timer slot selected for the next credit-wake arm.
  uint8_t credit_wake_next_timer_index;

  // First asynchronous failure observed by the carrier; NULL while healthy.
  iree_atomic_intptr_t failure_status;

  // Original receive-pool recycle callback wrapped for delivered leases.
  iree_async_buffer_recycle_callback_t recv_pool_recycle;

  // Local connection data serialized during carrier setup.
  iree_net_rdma_connection_data_t local_connection_data;

  // Peer connection data applied after carrier setup.
  iree_net_rdma_connection_data_t remote_connection_data;

  // Normalized queue and inline-data options.
  iree_net_rdma_carrier_options_t options;

  // Bitfield of iree_net_rdma_carrier_flag_bits_e values.
  iree_net_rdma_carrier_flags_t flags;
};

typedef struct iree_net_rdma_carrier_pending_send_failure_t {
  // Whether reservation contains an accepted send that needs a failure
  // callback.
  bool has_reservation;

  // User-visible reservation metadata captured after a deferred post failed.
  iree_net_rdma_send_reservation_t reservation;
} iree_net_rdma_carrier_pending_send_failure_t;

struct iree_net_rdma_carrier_memory_export_t {
  // Next active memory-window export in the carrier list.
  struct iree_net_rdma_carrier_memory_export_t* next;

  // Memory window owning the exported remote handle.
  iree_net_rdma_memory_window_t* memory_window;
};

typedef struct iree_net_rdma_carrier_memory_window_bind_state_t {
  // Completion status cloned from the bind work completion.
  iree_status_t status;

  // Set to 1 after status has been recorded.
  iree_atomic_int32_t completed;
} iree_net_rdma_carrier_memory_window_bind_state_t;

static const iree_net_carrier_vtable_t iree_net_rdma_carrier_vtable;

static iree_status_t iree_net_rdma_carrier_arm_credit_wake_locked(
    iree_net_rdma_carrier_t* carrier);

static iree_status_t
iree_net_rdma_carrier_try_post_pending_committed_sends_locked(
    iree_net_rdma_carrier_t* carrier, uint32_t* out_posted_count,
    iree_net_rdma_carrier_pending_send_failure_t* out_failed_send,
    iree_net_carrier_deactivate_callback_fn_t* inout_deactivate_callback,
    void** inout_deactivate_user_data);

static iree_status_t iree_net_rdma_carrier_try_post_credit_grant_locked(
    iree_net_rdma_carrier_t* carrier);

static bool iree_net_rdma_carrier_terminal_failure_handled_locked(
    iree_net_rdma_carrier_t* carrier);

static void iree_net_rdma_carrier_recycle_receive_buffer(void* user_data,
                                                         uint32_t buffer_index);

static void iree_net_rdma_carrier_defer_receive_replenish(
    iree_net_rdma_carrier_t* carrier);

static iree_status_t iree_net_rdma_carrier_try_replenish_receives_locked(
    iree_net_rdma_carrier_t* carrier,
    iree_net_rdma_carrier_pending_send_failure_t* out_failed_send,
    iree_net_carrier_deactivate_callback_fn_t* inout_deactivate_callback,
    void** inout_deactivate_user_data);

static void iree_net_rdma_carrier_notify_pending_send_failure(
    iree_net_rdma_carrier_t* carrier,
    iree_net_rdma_carrier_pending_send_failure_t* failure,
    const iree_status_t failure_status);

static void iree_net_rdma_carrier_free_memory_exports(
    iree_net_rdma_carrier_t* carrier);

static void iree_net_rdma_carrier_retire_created_receives_locked(
    iree_net_rdma_carrier_t* carrier,
    iree_net_carrier_deactivate_callback_fn_t* inout_deactivate_callback,
    void** inout_deactivate_user_data);

static iree_net_rdma_carrier_t* iree_net_rdma_carrier_from_base(
    iree_net_carrier_t* base_carrier) {
  return (iree_net_rdma_carrier_t*)base_carrier;
}

static iree_status_t iree_net_rdma_carrier_status_from_errno_required(
    const char* file, uint32_t line, int error, const char* call) {
  return iree_status_from_errno(file, line, error != 0 ? error : EIO, call);
}

static iree_status_t iree_net_rdma_carrier_validate_options(
    iree_net_rdma_carrier_options_t options) {
  if (options.send_queue_depth == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "send_queue_depth must be non-zero");
  }
  if (options.recv_queue_depth == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "recv_queue_depth must be non-zero");
  }
  if (options.recv_buffer_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "recv_buffer_size must be non-zero");
  }
  if (options.max_send_sge == 0 ||
      options.max_send_sge > IREE_NET_RDMA_CARRIER_MAX_SEND_SGE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "max_send_sge %u must be in [1, %u]",
        options.max_send_sge, IREE_NET_RDMA_CARRIER_MAX_SEND_SGE);
  }
  if (options.max_recv_sge == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "max_recv_sge must be non-zero");
  }
  if (options.send_staging_buffer_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "send_staging_buffer_size must be non-zero");
  }
  if (options.send_queue_depth > (uint32_t)INT_MAX ||
      options.recv_queue_depth > (uint32_t)INT_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue depths must fit in int");
  }
  if (options.send_queue_depth > UINT32_MAX - options.recv_queue_depth) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "combined queue depths overflow uint32_t");
  }
  return iree_ok_status();
}

static iree_net_rdma_carrier_options_t iree_net_rdma_carrier_resolve_options(
    iree_net_rdma_carrier_options_t options) {
  iree_net_rdma_carrier_options_t defaults =
      iree_net_rdma_carrier_options_default();
  if (options.send_queue_depth == 0) {
    options.send_queue_depth = defaults.send_queue_depth;
  }
  if (options.recv_queue_depth == 0) {
    options.recv_queue_depth = defaults.recv_queue_depth;
  }
  if (options.recv_buffer_size == 0) {
    options.recv_buffer_size = defaults.recv_buffer_size;
  }
  if (options.max_send_sge == 0) {
    options.max_send_sge = defaults.max_send_sge;
  }
  if (options.max_recv_sge == 0) {
    options.max_recv_sge = defaults.max_recv_sge;
  }
  if (options.max_inline_data == 0) {
    options.max_inline_data = defaults.max_inline_data;
  }
  if (options.send_staging_buffer_size == 0) {
    options.send_staging_buffer_size = defaults.send_staging_buffer_size;
  }
  return options;
}

static iree_status_t iree_net_rdma_carrier_validate_params(
    iree_net_rdma_carrier_create_params_t params) {
  if (!params.context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "context must not be NULL");
  }
  if (!params.proactor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "proactor must not be NULL");
  }
  if (!params.connection_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "connection_id must not be NULL");
  }
  if (params.connection_id->qp) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "connection_id already has a QP");
  }
  if (!params.callback.fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "callback.fn must not be NULL");
  }
  return iree_ok_status();
}

static iree_net_carrier_capabilities_t iree_net_rdma_carrier_capabilities(
    iree_net_rdma_context_t* context) {
  iree_net_carrier_capabilities_t capabilities =
      IREE_NET_CARRIER_CAPABILITY_RELIABLE |
      IREE_NET_CARRIER_CAPABILITY_ORDERED |
      IREE_NET_CARRIER_CAPABILITY_ZERO_COPY_TX |
      IREE_NET_CARRIER_CAPABILITY_ZERO_COPY_RX |
      IREE_NET_CARRIER_CAPABILITY_REGISTERED_REGIONS;
  if (iree_net_rdma_context_supports_memory_windows(context)) {
    capabilities |= IREE_NET_CARRIER_CAPABILITY_DIRECT_WRITE |
                    IREE_NET_CARRIER_CAPABILITY_DIRECT_READ;
  }
  return capabilities;
}

static void iree_net_rdma_carrier_set_failure_status(
    iree_net_rdma_carrier_t* carrier, iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  intptr_t expected = 0;
  intptr_t desired = (intptr_t)status;
  if (!iree_atomic_compare_exchange_strong(&carrier->failure_status, &expected,
                                           desired, iree_memory_order_release,
                                           iree_memory_order_relaxed)) {
    iree_status_free(status);
  }
}

static bool iree_net_rdma_carrier_has_failure_status(
    iree_net_rdma_carrier_t* carrier) {
  return iree_atomic_load(&carrier->failure_status,
                          iree_memory_order_acquire) != 0;
}

static iree_status_t iree_net_rdma_carrier_get_failure_status(
    iree_net_rdma_carrier_t* carrier) {
  intptr_t stored =
      iree_atomic_load(&carrier->failure_status, iree_memory_order_acquire);
  return stored ? iree_status_clone((iree_status_t)stored) : iree_ok_status();
}

static void iree_net_rdma_carrier_consume_failure_status(
    iree_net_rdma_carrier_t* carrier) {
  intptr_t stored = iree_atomic_exchange(&carrier->failure_status, 0,
                                         iree_memory_order_acq_rel);
  iree_status_free((iree_status_t)stored);
}

IREE_API_EXPORT void iree_net_rdma_carrier_record_failure(
    iree_net_rdma_carrier_t* carrier, iree_status_t status) {
  if (carrier) {
    iree_net_rdma_carrier_set_failure_status(carrier, status);
  } else {
    iree_status_free(status);
  }
}

static void iree_net_rdma_carrier_notify_error(iree_net_rdma_carrier_t* carrier,
                                               iree_status_t status) {
  if (carrier->base.callback.fn) {
    iree_net_rdma_carrier_record_failure(carrier, iree_status_clone(status));
    carrier->base.callback.fn(carrier->base.callback.user_data,
                              IREE_NET_CARRIER_COMPLETION_ERROR,
                              /*operation_user_data=*/0, status,
                              /*bytes_transferred=*/0, /*recv_lease=*/NULL);
  } else {
    iree_net_rdma_carrier_record_failure(carrier, status);
  }
}

static void iree_net_rdma_carrier_memory_window_bind_state_initialize(
    iree_net_rdma_carrier_memory_window_bind_state_t* state) {
  state->status = iree_ok_status();
  iree_atomic_store(&state->completed, 0, iree_memory_order_relaxed);
}

static void iree_net_rdma_carrier_memory_window_bind_state_complete(
    iree_net_rdma_carrier_memory_window_bind_state_t* state,
    const iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    state->status = iree_status_clone(status);
  }
  iree_atomic_store(&state->completed, 1, iree_memory_order_release);
}

static bool iree_net_rdma_carrier_memory_window_bind_state_completed(
    iree_net_rdma_carrier_memory_window_bind_state_t* state) {
  return iree_atomic_load(&state->completed, iree_memory_order_acquire) != 0;
}

static iree_status_t iree_net_rdma_carrier_memory_window_bind_state_consume(
    iree_net_rdma_carrier_memory_window_bind_state_t* state) {
  iree_status_t status = state->status;
  state->status = iree_ok_status();
  return status;
}

static void iree_net_rdma_carrier_memory_window_bind_state_deinitialize(
    iree_net_rdma_carrier_memory_window_bind_state_t* state) {
  iree_status_free(state->status);
  state->status = iree_ok_status();
}

static void iree_net_rdma_carrier_add_pending_operations(
    iree_net_rdma_carrier_t* carrier, uint32_t count) {
  if (count == 0) return;
  iree_atomic_fetch_add(&carrier->base.pending_operations, (int32_t)count,
                        iree_memory_order_release);
}

static void iree_net_rdma_carrier_maybe_complete_deactivation_locked(
    iree_net_rdma_carrier_t* carrier,
    iree_net_carrier_deactivate_callback_fn_t* out_callback,
    void** out_user_data) {
  *out_callback = NULL;
  *out_user_data = NULL;

  iree_net_carrier_state_t state = iree_net_carrier_state(&carrier->base);
  if (state != IREE_NET_CARRIER_STATE_DRAINING) return;

  int32_t pending = iree_atomic_load(&carrier->base.pending_operations,
                                     iree_memory_order_acquire);
  if (pending > 0) return;

  iree_net_carrier_set_state(&carrier->base,
                             IREE_NET_CARRIER_STATE_DEACTIVATED);
  *out_callback = carrier->deactivate_callback;
  *out_user_data = carrier->deactivate_user_data;
  carrier->deactivate_callback = NULL;
  carrier->deactivate_user_data = NULL;
}

static void iree_net_rdma_carrier_drop_pending_operations_locked(
    iree_net_rdma_carrier_t* carrier, uint32_t count,
    iree_net_carrier_deactivate_callback_fn_t* out_callback,
    void** out_user_data) {
  *out_callback = NULL;
  *out_user_data = NULL;
  if (count == 0) return;

  int32_t old_count =
      iree_atomic_fetch_sub(&carrier->base.pending_operations, (int32_t)count,
                            iree_memory_order_release);
  if (old_count < (int32_t)count) {
    iree_status_abort(iree_make_status(
        IREE_STATUS_INTERNAL,
        "RDMA carrier pending operation count underflow: old=%d drop=%u",
        old_count, count));
  }

  iree_net_rdma_carrier_maybe_complete_deactivation_locked(
      carrier, out_callback, out_user_data);
}

static void iree_net_rdma_carrier_invoke_deactivate_callback(
    iree_net_carrier_deactivate_callback_fn_t callback, void* user_data) {
  if (callback) callback(user_data);
}

static void iree_net_rdma_carrier_deferred_deactivate_callback(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_net_rdma_carrier_t* carrier = (iree_net_rdma_carrier_t*)user_data;
  if (!iree_status_is_ok(status)) {
    iree_status_abort(status);
  }

  iree_net_carrier_deactivate_callback_fn_t callback = NULL;
  void* callback_user_data = NULL;
  iree_slim_mutex_lock(&carrier->queue_mutex);
  if (!iree_all_bits_set(
          carrier->flags,
          IREE_NET_RDMA_CARRIER_FLAG_DEACTIVATE_CALLBACK_DEFERRED)) {
    iree_slim_mutex_unlock(&carrier->queue_mutex);
    iree_status_abort(iree_make_status(
        IREE_STATUS_INTERNAL,
        "RDMA carrier deactivation callback NOP completed with no callback"));
  }
  carrier->flags &= ~IREE_NET_RDMA_CARRIER_FLAG_DEACTIVATE_CALLBACK_DEFERRED;
  callback = carrier->deactivate_callback;
  callback_user_data = carrier->deactivate_user_data;
  carrier->deactivate_callback = NULL;
  carrier->deactivate_user_data = NULL;
  iree_slim_mutex_unlock(&carrier->queue_mutex);

  // Drop the NOP retain before invoking the public callback. Connection
  // teardown must see only connection-owned carrier references so carrier-owned
  // rdma_cm IDs are destroyed before the connection releases its CM channel.
  iree_net_carrier_release(&carrier->base);
  iree_net_rdma_carrier_invoke_deactivate_callback(callback,
                                                   callback_user_data);
}

static void iree_net_rdma_carrier_defer_deactivate_callback(
    iree_net_rdma_carrier_t* carrier,
    iree_net_carrier_deactivate_callback_fn_t callback, void* user_data) {
  if (!callback) return;

  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&carrier->queue_mutex);
  bool callback_in_flight =
      iree_any_bit_set(
          carrier->flags,
          IREE_NET_RDMA_CARRIER_FLAG_DEACTIVATE_CALLBACK_DEFERRED) ||
      carrier->deactivate_callback != NULL;
  if (callback_in_flight) {
    status =
        iree_make_status(IREE_STATUS_INTERNAL,
                         "RDMA carrier observed multiple deactivate callbacks");
  } else {
    carrier->deactivate_callback = callback;
    carrier->deactivate_user_data = user_data;
    carrier->flags |= IREE_NET_RDMA_CARRIER_FLAG_DEACTIVATE_CALLBACK_DEFERRED;
    iree_async_operation_zero(&carrier->deactivate_operation.base,
                              sizeof(carrier->deactivate_operation));
    iree_async_operation_initialize(
        &carrier->deactivate_operation.base, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE,
        iree_net_rdma_carrier_deferred_deactivate_callback, carrier);
    iree_net_carrier_retain(&carrier->base);
  }
  iree_slim_mutex_unlock(&carrier->queue_mutex);

  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_submit_one(
        carrier->proactor, &carrier->deactivate_operation.base);
  }
  if (!iree_status_is_ok(status)) {
    bool release_callback_retain = false;
    iree_slim_mutex_lock(&carrier->queue_mutex);
    if (iree_any_bit_set(
            carrier->flags,
            IREE_NET_RDMA_CARRIER_FLAG_DEACTIVATE_CALLBACK_DEFERRED)) {
      carrier->flags &=
          ~IREE_NET_RDMA_CARRIER_FLAG_DEACTIVATE_CALLBACK_DEFERRED;
      carrier->deactivate_callback = NULL;
      carrier->deactivate_user_data = NULL;
      release_callback_retain = true;
    }
    iree_slim_mutex_unlock(&carrier->queue_mutex);
    if (release_callback_retain) {
      iree_net_carrier_release(&carrier->base);
    }
    iree_status_abort(status);
  }
}

static void iree_net_rdma_carrier_capture_deactivate_callback(
    iree_net_carrier_deactivate_callback_fn_t new_callback, void* new_user_data,
    iree_net_carrier_deactivate_callback_fn_t* inout_callback,
    void** inout_user_data) {
  if (!new_callback) return;
  if (*inout_callback) {
    iree_status_abort(iree_make_status(
        IREE_STATUS_INTERNAL,
        "RDMA carrier observed multiple deactivate callbacks"));
  }
  *inout_callback = new_callback;
  *inout_user_data = new_user_data;
}

static iree_status_t iree_net_rdma_carrier_status_from_work_completion(
    const struct ibv_wc* completion, const char* queue_name) {
  if (completion->status == IBV_WC_SUCCESS) return iree_ok_status();
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE, "RDMA %s completion failed: status=%u vendor=%u",
      queue_name, (uint32_t)completion->status, completion->vendor_err);
}

static bool iree_net_rdma_carrier_work_completion_is_draining_flush(
    iree_net_rdma_carrier_t* carrier, const struct ibv_wc* completion) {
  if (completion->status != IBV_WC_WR_FLUSH_ERR) return false;
  iree_net_carrier_state_t state = iree_net_carrier_state(&carrier->base);
  return state == IREE_NET_CARRIER_STATE_DRAINING ||
         state == IREE_NET_CARRIER_STATE_DEACTIVATED;
}

static void iree_net_rdma_carrier_refresh_remote_recv_credits_locked(
    iree_net_rdma_carrier_t* carrier) {
  if (iree_any_bit_set(
          carrier->flags,
          IREE_NET_RDMA_CARRIER_FLAG_REMOTE_CONNECTION_DATA_APPLIED)) {
    uint32_t remote_recv_credit_limit =
        iree_net_rdma_credit_memory_load(carrier->credit_memory);
    iree_net_rdma_send_window_refresh_remote_credits(&carrier->send_window,
                                                     remote_recv_credit_limit);
  }
}

static bool iree_net_rdma_carrier_remote_recv_credits_exhausted_locked(
    iree_net_rdma_carrier_t* carrier) {
  if (!iree_any_bit_set(
          carrier->flags,
          IREE_NET_RDMA_CARRIER_FLAG_REMOTE_CONNECTION_DATA_APPLIED)) {
    return false;
  }
  return iree_net_rdma_send_window_available_recv_credits(
             &carrier->send_window) == 0;
}

static bool iree_net_rdma_carrier_acquire_uses_remote_recv_credit(
    iree_net_rdma_send_window_acquire_flags_t acquire_flags) {
  return iree_any_bit_set(
      acquire_flags, IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT);
}

static bool iree_net_rdma_carrier_can_inline_credit_grant(
    iree_net_rdma_carrier_t* carrier) {
  return carrier->local_connection_data.max_inline_data >= sizeof(uint32_t);
}

// Cancels the active credit-wake timer exactly once. The timer remains counted
// until its completion callback retires the operation. The caller must hold
// |queue_mutex|.
static void iree_net_rdma_carrier_cancel_credit_wake_locked(
    iree_net_rdma_carrier_t* carrier) {
  if (!carrier->credit_wake_timer ||
      iree_any_bit_set(
          carrier->flags,
          IREE_NET_RDMA_CARRIER_FLAG_CREDIT_WAKE_TIMER_CANCEL_REQUESTED)) {
    return;
  }

  iree_status_t status = iree_async_proactor_cancel(
      carrier->proactor, &carrier->credit_wake_timer->base);
  if (iree_status_is_not_found(status)) {
    // Timer completion already owns retirement and will drop the counted
    // operation.
    iree_status_free(status);
  } else if (!iree_status_is_ok(status)) {
    iree_status_abort(status);
  }
  carrier->flags |=
      IREE_NET_RDMA_CARRIER_FLAG_CREDIT_WAKE_TIMER_CANCEL_REQUESTED;
}

static void iree_net_rdma_carrier_on_credit_wake_timer(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_rdma_carrier_t* carrier = (iree_net_rdma_carrier_t*)user_data;

  bool notify_send_ready = false;
  iree_net_carrier_deactivate_callback_fn_t deactivate_callback = NULL;
  void* deactivate_user_data = NULL;
  iree_net_rdma_carrier_pending_send_failure_t pending_send_failure;
  memset(&pending_send_failure, 0, sizeof(pending_send_failure));
  iree_status_t rearm_status = iree_ok_status();
  bool cancelled =
      iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_CANCELLED);
  iree_slim_mutex_lock(&carrier->queue_mutex);
  IREE_ASSERT(carrier->credit_wake_timer ==
              (iree_async_timer_operation_t*)operation);
  carrier->credit_wake_timer = NULL;
  carrier->flags &=
      ~(IREE_NET_RDMA_CARRIER_FLAG_CREDIT_WAKE_TIMER_IN_FLIGHT |
        IREE_NET_RDMA_CARRIER_FLAG_CREDIT_WAKE_TIMER_CANCEL_REQUESTED);
  iree_net_rdma_carrier_drop_pending_operations_locked(
      carrier, 1, &deactivate_callback, &deactivate_user_data);
  if (iree_status_is_ok(status) && !cancelled &&
      iree_net_carrier_state(&carrier->base) == IREE_NET_CARRIER_STATE_ACTIVE) {
    iree_net_rdma_carrier_refresh_remote_recv_credits_locked(carrier);
    rearm_status =
        iree_net_rdma_carrier_try_post_pending_committed_sends_locked(
            carrier, /*out_posted_count=*/NULL, &pending_send_failure,
            &deactivate_callback, &deactivate_user_data);
    if (iree_status_is_ok(rearm_status) &&
        iree_net_rdma_carrier_remote_recv_credits_exhausted_locked(carrier)) {
      rearm_status = iree_net_rdma_carrier_arm_credit_wake_locked(carrier);
    } else if (iree_status_is_ok(rearm_status)) {
      notify_send_ready = true;
    }
  }
  iree_slim_mutex_unlock(&carrier->queue_mutex);

  if (!iree_status_is_ok(status)) {
    iree_net_rdma_carrier_notify_error(carrier, status);
  } else if (!iree_status_is_ok(rearm_status)) {
    iree_net_rdma_carrier_notify_pending_send_failure(
        carrier, &pending_send_failure, rearm_status);
    iree_net_rdma_carrier_notify_error(carrier, rearm_status);
  } else if (notify_send_ready) {
    carrier->base.callback.fn(carrier->base.callback.user_data,
                              IREE_NET_CARRIER_COMPLETION_SEND_READY,
                              /*operation_user_data=*/0, iree_ok_status(),
                              /*bytes_transferred=*/0, /*recv_lease=*/NULL);
  }
  iree_net_rdma_carrier_defer_deactivate_callback(carrier, deactivate_callback,
                                                  deactivate_user_data);
  iree_net_carrier_release(&carrier->base);
}

static iree_status_t iree_net_rdma_carrier_arm_credit_wake_locked(
    iree_net_rdma_carrier_t* carrier) {
  if (iree_net_carrier_state(&carrier->base) != IREE_NET_CARRIER_STATE_ACTIVE) {
    return iree_ok_status();
  }
  if (!iree_net_rdma_carrier_remote_recv_credits_exhausted_locked(carrier)) {
    return iree_ok_status();
  }
  if (iree_any_bit_set(
          carrier->flags,
          IREE_NET_RDMA_CARRIER_FLAG_CREDIT_WAKE_TIMER_IN_FLIGHT)) {
    return iree_ok_status();
  }

  uint8_t timer_index = carrier->credit_wake_next_timer_index;
  carrier->credit_wake_next_timer_index ^= 1u;
  iree_async_timer_operation_t* timer =
      &carrier->credit_wake_timers[timer_index];
  iree_async_operation_zero(&timer->base, sizeof(*timer));
  iree_async_operation_initialize(
      &timer->base, IREE_ASYNC_OPERATION_TYPE_TIMER,
      IREE_ASYNC_OPERATION_FLAG_CANCELLATION_IS_SUCCESS,
      iree_net_rdma_carrier_on_credit_wake_timer, carrier);
  timer->deadline_ns =
      iree_time_now() + IREE_NET_RDMA_CARRIER_CREDIT_WAKE_INTERVAL_NS;

  carrier->credit_wake_timer = timer;
  carrier->flags &=
      ~IREE_NET_RDMA_CARRIER_FLAG_CREDIT_WAKE_TIMER_CANCEL_REQUESTED;
  carrier->flags |= IREE_NET_RDMA_CARRIER_FLAG_CREDIT_WAKE_TIMER_IN_FLIGHT;
  iree_net_rdma_carrier_add_pending_operations(carrier, 1);
  iree_net_carrier_retain(&carrier->base);
  iree_status_t status =
      iree_async_proactor_submit_one(carrier->proactor, &timer->base);
  if (!iree_status_is_ok(status)) {
    carrier->credit_wake_timer = NULL;
    carrier->flags &=
        ~(IREE_NET_RDMA_CARRIER_FLAG_CREDIT_WAKE_TIMER_IN_FLIGHT |
          IREE_NET_RDMA_CARRIER_FLAG_CREDIT_WAKE_TIMER_CANCEL_REQUESTED);
    iree_net_carrier_deactivate_callback_fn_t deactivate_callback = NULL;
    void* deactivate_user_data = NULL;
    iree_net_rdma_carrier_drop_pending_operations_locked(
        carrier, 1, &deactivate_callback, &deactivate_user_data);
    IREE_ASSERT(!deactivate_callback && !deactivate_user_data);
    iree_net_carrier_release(&carrier->base);
  }
  return status;
}

static void iree_net_rdma_carrier_on_deferred_receive_replenish(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_net_rdma_carrier_t* carrier = (iree_net_rdma_carrier_t*)user_data;

  iree_net_carrier_deactivate_callback_fn_t deactivate_callback = NULL;
  void* deactivate_user_data = NULL;
  iree_net_rdma_carrier_pending_send_failure_t pending_send_failure;
  memset(&pending_send_failure, 0, sizeof(pending_send_failure));
  iree_slim_mutex_lock(&carrier->queue_mutex);
  carrier->flags &= ~IREE_NET_RDMA_CARRIER_FLAG_RECV_REPLENISH_DEFERRED;
  if (iree_status_is_ok(status) &&
      iree_net_carrier_state(&carrier->base) == IREE_NET_CARRIER_STATE_ACTIVE &&
      !iree_net_rdma_carrier_terminal_failure_handled_locked(carrier)) {
    status = iree_net_rdma_carrier_try_replenish_receives_locked(
        carrier, &pending_send_failure, &deactivate_callback,
        &deactivate_user_data);
  }
  iree_slim_mutex_unlock(&carrier->queue_mutex);

  if (!iree_status_is_ok(status)) {
    iree_net_rdma_carrier_notify_pending_send_failure(
        carrier, &pending_send_failure, status);
    iree_net_rdma_carrier_notify_error(carrier, status);
  }
  iree_net_rdma_carrier_defer_deactivate_callback(carrier, deactivate_callback,
                                                  deactivate_user_data);
  iree_net_carrier_release(&carrier->base);
}

static void iree_net_rdma_carrier_defer_receive_replenish(
    iree_net_rdma_carrier_t* carrier) {
  iree_status_t status = iree_ok_status();
  bool submit_operation = false;
  iree_slim_mutex_lock(&carrier->queue_mutex);
  if (iree_net_carrier_state(&carrier->base) == IREE_NET_CARRIER_STATE_ACTIVE &&
      !iree_net_rdma_carrier_terminal_failure_handled_locked(carrier) &&
      !iree_any_bit_set(carrier->flags,
                        IREE_NET_RDMA_CARRIER_FLAG_RECV_REPLENISH_DEFERRED)) {
    carrier->flags |= IREE_NET_RDMA_CARRIER_FLAG_RECV_REPLENISH_DEFERRED;
    iree_async_operation_zero(&carrier->recv_replenish_operation.base,
                              sizeof(carrier->recv_replenish_operation));
    iree_async_operation_initialize(
        &carrier->recv_replenish_operation.base, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE,
        iree_net_rdma_carrier_on_deferred_receive_replenish, carrier);
    iree_net_carrier_retain(&carrier->base);
    submit_operation = true;
  }
  iree_slim_mutex_unlock(&carrier->queue_mutex);

  if (submit_operation) {
    status = iree_async_proactor_submit_one(
        carrier->proactor, &carrier->recv_replenish_operation.base);
  }
  if (!iree_status_is_ok(status)) {
    bool release_operation_retain = false;
    iree_slim_mutex_lock(&carrier->queue_mutex);
    if (iree_any_bit_set(carrier->flags,
                         IREE_NET_RDMA_CARRIER_FLAG_RECV_REPLENISH_DEFERRED)) {
      carrier->flags &= ~IREE_NET_RDMA_CARRIER_FLAG_RECV_REPLENISH_DEFERRED;
      release_operation_retain = true;
    }
    iree_slim_mutex_unlock(&carrier->queue_mutex);
    if (release_operation_retain) {
      iree_net_carrier_release(&carrier->base);
    }
    iree_status_abort(status);
  }
}

static bool iree_net_rdma_carrier_receive_lease_is_wrapped(
    iree_net_rdma_carrier_t* carrier, iree_async_buffer_lease_t* lease) {
  return lease->release.fn == iree_net_rdma_carrier_recycle_receive_buffer &&
         lease->release.user_data == carrier;
}

static void iree_net_rdma_carrier_wrap_receive_lease_locked(
    iree_net_rdma_carrier_t* carrier, iree_async_buffer_lease_t* lease) {
  if (!lease->release.fn) return;
  if (!carrier->recv_pool_recycle.fn) {
    carrier->recv_pool_recycle = lease->release;
  } else if (carrier->recv_pool_recycle.fn != lease->release.fn ||
             carrier->recv_pool_recycle.user_data != lease->release.user_data) {
    iree_status_abort(iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "RDMA receive pool recycle callback changed unexpectedly"));
  }

  lease->release = (iree_async_buffer_recycle_callback_t){
      iree_net_rdma_carrier_recycle_receive_buffer,
      carrier,
  };
  iree_net_carrier_retain(&carrier->base);
}

static iree_status_t iree_net_rdma_carrier_release_receive_lease(
    iree_net_rdma_carrier_t* carrier, iree_async_buffer_lease_t* lease,
    iree_net_rdma_carrier_pending_send_failure_t* out_failed_send,
    iree_net_carrier_deactivate_callback_fn_t* inout_deactivate_callback,
    void** inout_deactivate_user_data) {
  if (!iree_net_rdma_carrier_receive_lease_is_wrapped(carrier, lease)) {
    iree_async_buffer_lease_release(lease);
    return iree_ok_status();
  }

  iree_async_buffer_recycle_callback_t recycle = carrier->recv_pool_recycle;
  uint32_t buffer_index = lease->buffer_index;
  lease->release = iree_async_buffer_recycle_callback_null();
  recycle.fn(recycle.user_data, buffer_index);

  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&carrier->queue_mutex);
  if (iree_net_carrier_state(&carrier->base) == IREE_NET_CARRIER_STATE_ACTIVE &&
      !iree_net_rdma_carrier_terminal_failure_handled_locked(carrier)) {
    status = iree_net_rdma_carrier_try_replenish_receives_locked(
        carrier, out_failed_send, inout_deactivate_callback,
        inout_deactivate_user_data);
  }
  iree_slim_mutex_unlock(&carrier->queue_mutex);
  iree_net_carrier_release(&carrier->base);
  return status;
}

static void iree_net_rdma_carrier_discard_receive_lease(
    iree_net_rdma_carrier_t* carrier, iree_async_buffer_lease_t* lease) {
  if (!iree_net_rdma_carrier_receive_lease_is_wrapped(carrier, lease)) {
    iree_async_buffer_lease_release(lease);
    return;
  }

  iree_async_buffer_recycle_callback_t recycle = carrier->recv_pool_recycle;
  uint32_t buffer_index = lease->buffer_index;
  lease->release = iree_async_buffer_recycle_callback_null();
  recycle.fn(recycle.user_data, buffer_index);
  iree_net_carrier_release(&carrier->base);
}

static void iree_net_rdma_carrier_recycle_receive_buffer(
    void* user_data, uint32_t buffer_index) {
  iree_net_rdma_carrier_t* carrier = (iree_net_rdma_carrier_t*)user_data;
  iree_async_buffer_recycle_callback_t recycle = carrier->recv_pool_recycle;
  recycle.fn(recycle.user_data, buffer_index);

  iree_net_rdma_carrier_defer_receive_replenish(carrier);
  iree_net_carrier_release(&carrier->base);
}

static iree_status_t iree_net_rdma_carrier_completion_kind_from_operation(
    iree_net_rdma_work_request_operation_t operation,
    iree_net_carrier_completion_kind_t* out_kind) {
  *out_kind = IREE_NET_CARRIER_COMPLETION_NONE;
  switch (operation) {
    case IREE_NET_RDMA_WORK_REQUEST_OPERATION_SEND:
      *out_kind = IREE_NET_CARRIER_COMPLETION_SEND;
      return iree_ok_status();
    case IREE_NET_RDMA_WORK_REQUEST_OPERATION_DIRECT_WRITE:
      *out_kind = IREE_NET_CARRIER_COMPLETION_DIRECT_WRITE;
      return iree_ok_status();
    case IREE_NET_RDMA_WORK_REQUEST_OPERATION_DIRECT_READ:
      *out_kind = IREE_NET_CARRIER_COMPLETION_DIRECT_READ;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "RDMA operation %u has no carrier completion",
                              operation);
  }
}

static bool iree_net_rdma_carrier_terminal_failure_handled_locked(
    iree_net_rdma_carrier_t* carrier) {
  return iree_any_bit_set(carrier->flags,
                          IREE_NET_RDMA_CARRIER_FLAG_TERMINAL_FAILURE_HANDLED);
}

static bool iree_net_rdma_carrier_operation_has_user_completion(
    iree_net_rdma_work_request_operation_t operation) {
  switch (operation) {
    case IREE_NET_RDMA_WORK_REQUEST_OPERATION_SEND:
    case IREE_NET_RDMA_WORK_REQUEST_OPERATION_DIRECT_WRITE:
    case IREE_NET_RDMA_WORK_REQUEST_OPERATION_DIRECT_READ:
      return true;
    default:
      return false;
  }
}

static bool iree_net_rdma_carrier_operation_requires_open_send_direction(
    iree_net_rdma_work_request_operation_t operation) {
  switch (operation) {
    case IREE_NET_RDMA_WORK_REQUEST_OPERATION_SEND:
    case IREE_NET_RDMA_WORK_REQUEST_OPERATION_DIRECT_WRITE:
    case IREE_NET_RDMA_WORK_REQUEST_OPERATION_DIRECT_READ:
      return true;
    default:
      return false;
  }
}

static bool iree_net_rdma_carrier_operation_is_internal(
    iree_net_rdma_work_request_operation_t operation) {
  switch (operation) {
    case IREE_NET_RDMA_WORK_REQUEST_OPERATION_COMMITTED_SEND:
    case IREE_NET_RDMA_WORK_REQUEST_OPERATION_CREDIT_GRANT:
    case IREE_NET_RDMA_WORK_REQUEST_OPERATION_MEMORY_WINDOW_BIND:
    case IREE_NET_RDMA_WORK_REQUEST_OPERATION_BOOTSTRAP_SEND:
      return true;
    default:
      return false;
  }
}

static iree_status_t
iree_net_rdma_carrier_operation_from_send_reservation_completion(
    iree_net_rdma_send_reservation_completion_t completion,
    iree_net_rdma_work_request_operation_t* out_operation) {
  *out_operation = IREE_NET_RDMA_WORK_REQUEST_OPERATION_NONE;
  switch (completion) {
    case IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_INTERNAL:
      *out_operation = IREE_NET_RDMA_WORK_REQUEST_OPERATION_COMMITTED_SEND;
      return iree_ok_status();
    case IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_SEND:
      *out_operation = IREE_NET_RDMA_WORK_REQUEST_OPERATION_SEND;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "reservation completion mode %u is not valid",
                              (uint32_t)completion);
  }
}

static void iree_net_rdma_carrier_notify_work_request_failure(
    iree_net_rdma_carrier_t* carrier,
    iree_net_rdma_work_request_completion_t* completion,
    const iree_status_t failure_status) {
  iree_async_buffer_lease_release(&completion->retained_buffer_lease);
  if (completion->operation ==
      IREE_NET_RDMA_WORK_REQUEST_OPERATION_MEMORY_WINDOW_BIND) {
    iree_net_rdma_carrier_memory_window_bind_state_complete(
        (iree_net_rdma_carrier_memory_window_bind_state_t*)(uintptr_t)
            completion->user_data,
        failure_status);
    return;
  }
  if (!iree_net_rdma_carrier_operation_has_user_completion(
          completion->operation)) {
    return;
  }

  iree_net_carrier_completion_kind_t kind = IREE_NET_CARRIER_COMPLETION_NONE;
  iree_status_t kind_status =
      iree_net_rdma_carrier_completion_kind_from_operation(
          completion->operation, &kind);
  if (!iree_status_is_ok(kind_status)) {
    iree_status_abort(kind_status);
  }

  carrier->base.callback.fn(carrier->base.callback.user_data, kind,
                            completion->user_data,
                            iree_status_clone(failure_status),
                            /*bytes_transferred=*/0,
                            /*recv_lease=*/NULL);
}

static void iree_net_rdma_carrier_notify_send_reservation_failure(
    iree_net_rdma_carrier_t* carrier,
    iree_net_rdma_send_reservation_t* reservation,
    const iree_status_t failure_status) {
  iree_async_buffer_lease_release(&reservation->buffer_lease);
  switch (reservation->completion) {
    case IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_INTERNAL:
      return;
    case IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_SEND:
      carrier->base.callback.fn(
          carrier->base.callback.user_data, IREE_NET_CARRIER_COMPLETION_SEND,
          reservation->user_data, iree_status_clone(failure_status),
          /*bytes_transferred=*/0,
          /*recv_lease=*/NULL);
      return;
    default:
      iree_status_abort(
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "reservation completion mode %u is not valid",
                           (uint32_t)reservation->completion));
      return;
  }
}

static void iree_net_rdma_carrier_capture_pending_send_failure(
    iree_net_rdma_send_reservation_t* reservation,
    iree_net_rdma_carrier_pending_send_failure_t* out_failure) {
  if (reservation->completion !=
      IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_SEND) {
    return;
  }
  if (out_failure->has_reservation) {
    iree_status_abort(iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "multiple deferred RDMA SEND failures captured in one drain"));
  }
  out_failure->has_reservation = true;
  out_failure->reservation = *reservation;
  memset(reservation, 0, sizeof(*reservation));
}

static void iree_net_rdma_carrier_notify_pending_send_failure(
    iree_net_rdma_carrier_t* carrier,
    iree_net_rdma_carrier_pending_send_failure_t* failure,
    const iree_status_t failure_status) {
  if (!failure->has_reservation) return;
  iree_net_rdma_carrier_notify_send_reservation_failure(
      carrier, &failure->reservation, failure_status);
  memset(failure, 0, sizeof(*failure));
}

// Resolves and fails every carrier-owned send reservation. Reservation slots
// are counted as pending operations from acquisition through native posting or
// cancellation, including committed sends waiting for peer receive credits.
// The caller owns |failure_status| and receives any deactivation callback made
// ready by the final retirement.
static void iree_net_rdma_carrier_fail_send_reservations(
    iree_net_rdma_carrier_t* carrier, const iree_status_t failure_status,
    iree_net_carrier_deactivate_callback_fn_t* inout_deactivate_callback,
    void** inout_deactivate_user_data) {
  uint32_t reservation_cursor = 0;
  while (true) {
    iree_net_rdma_send_reservation_t reservation;
    memset(&reservation, 0, sizeof(reservation));
    bool found_reservation = false;

    iree_slim_mutex_lock(&carrier->queue_mutex);
    iree_status_t resolve_status =
        iree_net_rdma_send_reservation_table_resolve_next(
            &carrier->send_reservation_table, &reservation_cursor, &reservation,
            &found_reservation);
    iree_net_carrier_deactivate_callback_fn_t drop_callback = NULL;
    void* drop_user_data = NULL;
    if (iree_status_is_ok(resolve_status) && found_reservation) {
      iree_net_rdma_carrier_drop_pending_operations_locked(
          carrier, 1, &drop_callback, &drop_user_data);
    } else if (iree_status_is_ok(resolve_status)) {
      iree_net_rdma_carrier_maybe_complete_deactivation_locked(
          carrier, &drop_callback, &drop_user_data);
    }
    iree_net_rdma_carrier_capture_deactivate_callback(
        drop_callback, drop_user_data, inout_deactivate_callback,
        inout_deactivate_user_data);
    iree_slim_mutex_unlock(&carrier->queue_mutex);

    if (!iree_status_is_ok(resolve_status)) {
      iree_async_buffer_lease_release(&reservation.buffer_lease);
      iree_status_abort(resolve_status);
    }
    if (!found_reservation) break;

    iree_net_rdma_carrier_notify_send_reservation_failure(carrier, &reservation,
                                                          failure_status);
  }
}

// Resolves and fails committed reservations still waiting in the carrier-owned
// post FIFO. Uncommitted begin_send reservations remain caller-owned and keep
// deactivation pending until the caller commits or aborts their handles.
static void iree_net_rdma_carrier_fail_pending_send_reservations(
    iree_net_rdma_carrier_t* carrier, const iree_status_t failure_status,
    iree_net_carrier_deactivate_callback_fn_t* inout_deactivate_callback,
    void** inout_deactivate_user_data) {
  while (true) {
    iree_net_rdma_send_reservation_t reservation;
    memset(&reservation, 0, sizeof(reservation));
    bool found_reservation = false;

    iree_slim_mutex_lock(&carrier->queue_mutex);
    found_reservation = iree_net_rdma_send_reservation_table_pending_count(
                            &carrier->send_reservation_table) > 0;
    iree_status_t resolve_status = iree_ok_status();
    if (found_reservation) {
      iree_net_carrier_send_handle_t handle = 0;
      resolve_status =
          iree_net_rdma_send_reservation_table_resolve_pending_front(
              &carrier->send_reservation_table, &handle, &reservation);
    }
    iree_net_carrier_deactivate_callback_fn_t drop_callback = NULL;
    void* drop_user_data = NULL;
    if (iree_status_is_ok(resolve_status) && found_reservation) {
      iree_net_rdma_carrier_drop_pending_operations_locked(
          carrier, 1, &drop_callback, &drop_user_data);
    } else if (iree_status_is_ok(resolve_status)) {
      iree_net_rdma_carrier_maybe_complete_deactivation_locked(
          carrier, &drop_callback, &drop_user_data);
    }
    iree_net_rdma_carrier_capture_deactivate_callback(
        drop_callback, drop_user_data, inout_deactivate_callback,
        inout_deactivate_user_data);
    iree_slim_mutex_unlock(&carrier->queue_mutex);

    if (!iree_status_is_ok(resolve_status)) {
      iree_async_buffer_lease_release(&reservation.buffer_lease);
      iree_status_abort(resolve_status);
    }
    if (!found_reservation) break;

    iree_net_rdma_carrier_notify_send_reservation_failure(carrier, &reservation,
                                                          failure_status);
  }
}

static void iree_net_rdma_carrier_fail_all(iree_net_rdma_carrier_t* carrier,
                                           iree_status_t failure_status) {
  bool should_retire_local_state = false;
  iree_slim_mutex_lock(&carrier->queue_mutex);
  if (!iree_net_rdma_carrier_terminal_failure_handled_locked(carrier)) {
    carrier->flags |= IREE_NET_RDMA_CARRIER_FLAG_TERMINAL_FAILURE_HANDLED;
    carrier->flags &= ~IREE_NET_RDMA_CARRIER_FLAG_CREDIT_GRANT_IN_FLIGHT;
    iree_net_carrier_state_t state = iree_net_carrier_state(&carrier->base);
    if (state == IREE_NET_CARRIER_STATE_ACTIVE ||
        state == IREE_NET_CARRIER_STATE_CREATED) {
      iree_net_carrier_set_state(&carrier->base,
                                 IREE_NET_CARRIER_STATE_DRAINING);
    }
    iree_net_rdma_carrier_cancel_credit_wake_locked(carrier);
    should_retire_local_state = true;
  }
  iree_slim_mutex_unlock(&carrier->queue_mutex);

  if (!should_retire_local_state) {
    // A sibling CQ may report the same terminal condition after local state has
    // already been retired. Surface it as a carrier error without touching the
    // already-failed work tables.
    iree_net_rdma_carrier_notify_error(carrier, failure_status);
    return;
  }

  iree_net_carrier_deactivate_callback_fn_t deactivate_callback = NULL;
  void* deactivate_user_data = NULL;
  uint32_t drain_cursor = 0;
  while (true) {
    iree_net_rdma_work_request_completion_t completion;
    memset(&completion, 0, sizeof(completion));
    iree_async_buffer_lease_t receive_lease;
    memset(&receive_lease, 0, sizeof(receive_lease));
    bool found_completion = false;

    iree_slim_mutex_lock(&carrier->queue_mutex);
    iree_status_t drain_status = iree_net_rdma_work_request_table_drain_next(
        &carrier->work_request_table, &drain_cursor, &completion,
        &found_completion);
    if (iree_status_is_ok(drain_status) && found_completion &&
        completion.operation == IREE_NET_RDMA_WORK_REQUEST_OPERATION_RECV) {
      drain_status = iree_net_rdma_receive_queue_complete(
          &carrier->receive_queue, completion, /*byte_length=*/0,
          &receive_lease);
    }
    iree_net_carrier_deactivate_callback_fn_t drop_callback = NULL;
    void* drop_user_data = NULL;
    if (iree_status_is_ok(drain_status) && found_completion) {
      iree_net_rdma_carrier_drop_pending_operations_locked(
          carrier, 1, &drop_callback, &drop_user_data);
    }
    iree_slim_mutex_unlock(&carrier->queue_mutex);
    iree_net_rdma_carrier_capture_deactivate_callback(
        drop_callback, drop_user_data, &deactivate_callback,
        &deactivate_user_data);

    if (!iree_status_is_ok(drain_status)) {
      iree_async_buffer_lease_release(&receive_lease);
      iree_async_buffer_lease_release(&completion.retained_buffer_lease);
      iree_status_abort(drain_status);
    }
    if (!found_completion) break;

    iree_async_buffer_lease_release(&receive_lease);
    iree_net_rdma_carrier_notify_work_request_failure(carrier, &completion,
                                                      failure_status);
  }

  iree_net_rdma_carrier_fail_send_reservations(
      carrier, failure_status, &deactivate_callback, &deactivate_user_data);

  iree_net_rdma_carrier_notify_error(carrier, failure_status);
  iree_net_rdma_carrier_defer_deactivate_callback(carrier, deactivate_callback,
                                                  deactivate_user_data);
}

static void iree_net_rdma_carrier_retire_created_receives_locked(
    iree_net_rdma_carrier_t* carrier,
    iree_net_carrier_deactivate_callback_fn_t* inout_deactivate_callback,
    void** inout_deactivate_user_data) {
  uint32_t drain_cursor = 0;
  bool found_completion = true;
  while (found_completion) {
    iree_net_rdma_work_request_completion_t completion;
    memset(&completion, 0, sizeof(completion));
    iree_async_buffer_lease_t receive_lease;
    memset(&receive_lease, 0, sizeof(receive_lease));
    iree_status_t drain_status = iree_net_rdma_work_request_table_drain_next(
        &carrier->work_request_table, &drain_cursor, &completion,
        &found_completion);
    if (iree_status_is_ok(drain_status) && found_completion &&
        completion.operation != IREE_NET_RDMA_WORK_REQUEST_OPERATION_RECV) {
      drain_status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "CREATED RDMA carrier contains non-receive operation %u",
          completion.operation);
    }
    if (iree_status_is_ok(drain_status) && found_completion) {
      drain_status = iree_net_rdma_receive_queue_complete(
          &carrier->receive_queue, completion, /*byte_length=*/0,
          &receive_lease);
    }
    iree_net_carrier_deactivate_callback_fn_t drop_callback = NULL;
    void* drop_user_data = NULL;
    if (iree_status_is_ok(drain_status) && found_completion) {
      iree_net_rdma_carrier_drop_pending_operations_locked(
          carrier, 1, &drop_callback, &drop_user_data);
      iree_net_rdma_carrier_capture_deactivate_callback(
          drop_callback, drop_user_data, inout_deactivate_callback,
          inout_deactivate_user_data);
    }
    iree_async_buffer_lease_release(&receive_lease);
    iree_async_buffer_lease_release(&completion.retained_buffer_lease);
    if (!iree_status_is_ok(drain_status)) {
      iree_status_abort(drain_status);
    }
  }
  iree_net_carrier_deactivate_callback_fn_t deactivate_callback = NULL;
  void* deactivate_user_data = NULL;
  iree_net_rdma_carrier_maybe_complete_deactivation_locked(
      carrier, &deactivate_callback, &deactivate_user_data);
  iree_net_rdma_carrier_capture_deactivate_callback(
      deactivate_callback, deactivate_user_data, inout_deactivate_callback,
      inout_deactivate_user_data);
}

static void iree_net_rdma_carrier_on_send_completions(
    void* user_data, iree_status_t status, const struct ibv_wc* completions,
    iree_host_size_t completion_count) {
  iree_net_rdma_carrier_t* carrier = (iree_net_rdma_carrier_t*)user_data;
  if (!iree_status_is_ok(status)) {
    iree_net_rdma_carrier_fail_all(carrier, status);
    return;
  }

  for (iree_host_size_t i = 0; i < completion_count; ++i) {
    const struct ibv_wc* completion = &completions[i];
    iree_net_rdma_work_request_completion_t work_request_completion;
    memset(&work_request_completion, 0, sizeof(work_request_completion));
    bool draining_flush =
        iree_net_rdma_carrier_work_completion_is_draining_flush(carrier,
                                                                completion);
    iree_status_t work_status = iree_ok_status();
    iree_net_carrier_deactivate_callback_fn_t deactivate_callback = NULL;
    void* deactivate_user_data = NULL;
    iree_net_rdma_carrier_pending_send_failure_t pending_send_failure;
    memset(&pending_send_failure, 0, sizeof(pending_send_failure));
    bool send_queue_completed = false;
    bool terminal_failure_handled = false;
    iree_status_t credit_wake_status = iree_ok_status();
    bool notify_send_ready = false;

    iree_slim_mutex_lock(&carrier->queue_mutex);
    terminal_failure_handled =
        iree_net_rdma_carrier_terminal_failure_handled_locked(carrier);
    iree_status_t cleanup_status = iree_ok_status();
    if (!terminal_failure_handled) {
      work_status = draining_flush
                        ? iree_ok_status()
                        : iree_net_rdma_carrier_status_from_work_completion(
                              completion, "send");
      cleanup_status = iree_net_rdma_work_request_table_complete(
          &carrier->work_request_table, completion->wr_id,
          &work_request_completion);
      if (iree_status_is_ok(cleanup_status) &&
          work_request_completion.operation ==
              IREE_NET_RDMA_WORK_REQUEST_OPERATION_RECV) {
        cleanup_status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                          "RDMA send CQ returned operation %u",
                                          work_request_completion.operation);
      }
      if (iree_status_is_ok(cleanup_status)) {
        cleanup_status =
            iree_net_rdma_send_window_complete(&carrier->send_window);
        send_queue_completed = iree_status_is_ok(cleanup_status);
      }
      if (iree_status_is_ok(cleanup_status) &&
          work_request_completion.operation ==
              IREE_NET_RDMA_WORK_REQUEST_OPERATION_CREDIT_GRANT) {
        carrier->flags &= ~IREE_NET_RDMA_CARRIER_FLAG_CREDIT_GRANT_IN_FLIGHT;
        if (!draining_flush && iree_status_is_ok(work_status)) {
          carrier->local_recv_credit_published =
              (uint32_t)work_request_completion.user_data;
        }
      }
      bool memory_window_bind =
          work_request_completion.operation ==
          IREE_NET_RDMA_WORK_REQUEST_OPERATION_MEMORY_WINDOW_BIND;
      if (!memory_window_bind && !draining_flush &&
          iree_status_is_ok(work_status) && iree_status_is_ok(cleanup_status)) {
        iree_status_t post_credit_grant_status =
            iree_net_rdma_carrier_try_post_credit_grant_locked(carrier);
        cleanup_status =
            iree_status_join(cleanup_status, post_credit_grant_status);
      }
      if (send_queue_completed) {
        iree_net_rdma_carrier_drop_pending_operations_locked(
            carrier, 1, &deactivate_callback, &deactivate_user_data);
      }
      if (!memory_window_bind && !draining_flush && send_queue_completed &&
          iree_status_is_ok(work_status) && iree_status_is_ok(cleanup_status)) {
        iree_net_rdma_carrier_refresh_remote_recv_credits_locked(carrier);
      }
      if (!memory_window_bind && !draining_flush && send_queue_completed &&
          iree_status_is_ok(work_status) && iree_status_is_ok(cleanup_status)) {
        iree_status_t post_pending_send_status =
            iree_net_rdma_carrier_try_post_pending_committed_sends_locked(
                carrier, /*out_posted_count=*/NULL, &pending_send_failure,
                &deactivate_callback, &deactivate_user_data);
        cleanup_status =
            iree_status_join(cleanup_status, post_pending_send_status);
      }
      if (!memory_window_bind && !draining_flush && send_queue_completed &&
          iree_status_is_ok(work_status) && iree_status_is_ok(cleanup_status) &&
          iree_net_rdma_carrier_remote_recv_credits_exhausted_locked(carrier)) {
        credit_wake_status =
            iree_net_rdma_carrier_arm_credit_wake_locked(carrier);
      }
    }
    iree_slim_mutex_unlock(&carrier->queue_mutex);

    // The terminal failure path already failed and released every local WR
    // table entry. Any later CQE is stale and must not complete an operation a
    // second time.
    if (terminal_failure_handled) continue;

    iree_async_buffer_lease_release(
        &work_request_completion.retained_buffer_lease);
    if (!iree_status_is_ok(cleanup_status)) {
      iree_net_rdma_carrier_notify_pending_send_failure(
          carrier, &pending_send_failure, cleanup_status);
    }
    if (!draining_flush && iree_status_is_ok(work_status) &&
        iree_status_is_ok(cleanup_status) &&
        !iree_net_rdma_carrier_operation_is_internal(
            work_request_completion.operation)) {
      if (work_request_completion.operation ==
          IREE_NET_RDMA_WORK_REQUEST_OPERATION_DIRECT_READ) {
        iree_atomic_fetch_add(&carrier->base.bytes_received,
                              (int64_t)work_request_completion.byte_length,
                              iree_memory_order_relaxed);
      } else {
        iree_atomic_fetch_add(&carrier->base.bytes_sent,
                              (int64_t)work_request_completion.byte_length,
                              iree_memory_order_relaxed);
      }
    }
    if (draining_flush && !iree_net_rdma_carrier_operation_is_internal(
                              work_request_completion.operation)) {
      work_status = iree_make_status(
          IREE_STATUS_CANCELLED, "RDMA send cancelled by carrier deactivation");
    }
    iree_status_t completion_status =
        iree_status_join(work_status, cleanup_status);
    // Non-draining verbs failures are terminal for the QP. Complete the current
    // CQE first, then retire the rest of the local carrier state.
    iree_status_t terminal_failure_status =
        !draining_flush && !iree_status_is_ok(work_status)
            ? iree_status_clone(completion_status)
            : iree_ok_status();
    iree_host_size_t bytes_transferred =
        iree_status_is_ok(completion_status)
            ? work_request_completion.byte_length
            : 0;
    if (work_request_completion.operation ==
        IREE_NET_RDMA_WORK_REQUEST_OPERATION_MEMORY_WINDOW_BIND) {
      iree_net_rdma_carrier_memory_window_bind_state_complete(
          (iree_net_rdma_carrier_memory_window_bind_state_t*)(uintptr_t)
              work_request_completion.user_data,
          completion_status);
    }
    bool internal_completion = iree_net_rdma_carrier_operation_is_internal(
        work_request_completion.operation);
    if (internal_completion) {
      if (!iree_status_is_ok(completion_status)) {
        if (iree_status_is_ok(terminal_failure_status)) {
          iree_net_rdma_carrier_notify_error(carrier, completion_status);
        } else {
          iree_status_free(completion_status);
        }
      } else if (!draining_flush &&
                 work_request_completion.operation !=
                     IREE_NET_RDMA_WORK_REQUEST_OPERATION_MEMORY_WINDOW_BIND &&
                 work_request_completion.operation !=
                     IREE_NET_RDMA_WORK_REQUEST_OPERATION_BOOTSTRAP_SEND) {
        notify_send_ready = true;
      }
    } else {
      iree_net_carrier_completion_kind_t kind =
          IREE_NET_CARRIER_COMPLETION_NONE;
      completion_status =
          iree_status_join(completion_status,
                           iree_net_rdma_carrier_completion_kind_from_operation(
                               work_request_completion.operation, &kind));
      if (!iree_status_is_ok(completion_status)) {
        bytes_transferred = 0;
      }
      carrier->base.callback.fn(carrier->base.callback.user_data, kind,
                                work_request_completion.user_data,
                                completion_status, bytes_transferred,
                                /*recv_lease=*/NULL);
    }
    if (!iree_status_is_ok(credit_wake_status)) {
      iree_net_rdma_carrier_notify_error(carrier, credit_wake_status);
    } else if (notify_send_ready) {
      carrier->base.callback.fn(carrier->base.callback.user_data,
                                IREE_NET_CARRIER_COMPLETION_SEND_READY,
                                /*operation_user_data=*/0, iree_ok_status(),
                                /*bytes_transferred=*/0, /*recv_lease=*/NULL);
    }
    iree_net_rdma_carrier_defer_deactivate_callback(
        carrier, deactivate_callback, deactivate_user_data);
    if (!iree_status_is_ok(terminal_failure_status)) {
      iree_net_rdma_carrier_fail_all(carrier, terminal_failure_status);
    }
  }
}

static iree_status_t iree_net_rdma_carrier_deliver_receive(
    iree_net_rdma_carrier_t* carrier, iree_async_buffer_lease_t* lease) {
  iree_status_t status = iree_ok_status();
  iree_net_carrier_state_t state = iree_net_carrier_state(&carrier->base);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE ||
      !carrier->base.recv_handler.fn) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "RDMA receive arrived before activation");
  }
  if (iree_status_is_ok(status)) {
    status = carrier->base.recv_handler.fn(carrier->base.recv_handler.user_data,
                                           lease->span, lease);
  }
  if (iree_status_is_ok(status)) {
    iree_atomic_fetch_add(&carrier->base.bytes_received,
                          (int64_t)lease->span.length,
                          iree_memory_order_relaxed);
  }
  return status;
}

static iree_status_t iree_net_rdma_carrier_deliver_signal(
    iree_net_rdma_carrier_t* carrier, uint32_t immediate) {
  iree_status_t status = iree_ok_status();
  iree_net_carrier_state_t state = iree_net_carrier_state(&carrier->base);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE ||
      !carrier->base.signal_handler.fn) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "RDMA direct_write signal arrived before activation");
  }
  if (iree_status_is_ok(status)) {
    status = carrier->base.signal_handler.fn(
        carrier->base.signal_handler.user_data, immediate);
  }
  return status;
}

static void iree_net_rdma_carrier_on_recv_completions(
    void* user_data, iree_status_t status, const struct ibv_wc* completions,
    iree_host_size_t completion_count) {
  iree_net_rdma_carrier_t* carrier = (iree_net_rdma_carrier_t*)user_data;
  if (!iree_status_is_ok(status)) {
    iree_net_rdma_carrier_fail_all(carrier, status);
    return;
  }

  for (iree_host_size_t i = 0; i < completion_count; ++i) {
    const struct ibv_wc* completion = &completions[i];
    iree_net_rdma_work_request_completion_t work_request_completion;
    memset(&work_request_completion, 0, sizeof(work_request_completion));
    bool draining_flush =
        iree_net_rdma_carrier_work_completion_is_draining_flush(carrier,
                                                                completion);
    iree_status_t work_status = iree_ok_status();
    iree_net_carrier_deactivate_callback_fn_t deactivate_callback = NULL;
    void* deactivate_user_data = NULL;
    iree_net_rdma_carrier_pending_send_failure_t pending_send_failure;
    memset(&pending_send_failure, 0, sizeof(pending_send_failure));
    bool receive_queue_completed = false;
    bool receive_is_signal = false;
    bool terminal_failure_handled = false;
    uint32_t receive_signal_immediate = 0;

    iree_slim_mutex_lock(&carrier->queue_mutex);
    iree_async_buffer_lease_t lease;
    memset(&lease, 0, sizeof(lease));
    terminal_failure_handled =
        iree_net_rdma_carrier_terminal_failure_handled_locked(carrier);
    iree_status_t cleanup_status = iree_ok_status();
    if (!terminal_failure_handled) {
      work_status = draining_flush
                        ? iree_ok_status()
                        : iree_net_rdma_carrier_status_from_work_completion(
                              completion, "recv");
      cleanup_status = iree_net_rdma_work_request_table_complete(
          &carrier->work_request_table, completion->wr_id,
          &work_request_completion);
      if (iree_status_is_ok(cleanup_status) &&
          work_request_completion.operation !=
              IREE_NET_RDMA_WORK_REQUEST_OPERATION_RECV) {
        cleanup_status =
            iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                             "RDMA receive CQ returned operation %u",
                             work_request_completion.operation);
      }

      if (iree_status_is_ok(cleanup_status)) {
        iree_host_size_t byte_length = 0;
        if (!draining_flush && iree_status_is_ok(work_status)) {
          switch (completion->opcode) {
            case IBV_WC_RECV:
              byte_length = (iree_host_size_t)completion->byte_len;
              break;
            case IBV_WC_RECV_RDMA_WITH_IMM:
              receive_is_signal = true;
              if (iree_any_bit_set(completion->wc_flags, IBV_WC_WITH_IMM)) {
                receive_signal_immediate = ntohl(completion->imm_data);
              } else {
                cleanup_status = iree_make_status(
                    IREE_STATUS_FAILED_PRECONDITION,
                    "RDMA direct_write signal completion missing immediate");
              }
              break;
            default:
              cleanup_status =
                  iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                   "RDMA receive CQ returned opcode %d",
                                   (int)completion->opcode);
              break;
          }
        }
        if (!iree_status_is_ok(work_status)) {
          receive_is_signal = false;
        }
        if (!iree_status_is_ok(cleanup_status)) {
          byte_length = 0;
        }
        iree_status_t receive_queue_status =
            iree_net_rdma_receive_queue_complete(&carrier->receive_queue,
                                                 work_request_completion,
                                                 byte_length, &lease);
        receive_queue_completed = iree_status_is_ok(receive_queue_status);
        if (receive_queue_completed) {
          iree_net_rdma_carrier_wrap_receive_lease_locked(carrier, &lease);
        }
        cleanup_status = iree_status_join(cleanup_status, receive_queue_status);
      }
    }
    iree_slim_mutex_unlock(&carrier->queue_mutex);

    // The terminal failure path already failed and released every local WR
    // table entry. Any later CQE is stale and must not complete an operation a
    // second time.
    if (terminal_failure_handled) continue;

    iree_net_carrier_state_t delivery_state =
        iree_net_carrier_state(&carrier->base);
    bool should_deliver =
        !draining_flush && delivery_state == IREE_NET_CARRIER_STATE_ACTIVE &&
        iree_status_is_ok(work_status) && iree_status_is_ok(cleanup_status);
    if (should_deliver) {
      work_status = receive_is_signal ? iree_net_rdma_carrier_deliver_signal(
                                            carrier, receive_signal_immediate)
                                      : iree_net_rdma_carrier_deliver_receive(
                                            carrier, &lease);
    }
    if (iree_status_is_ok(cleanup_status) &&
        (draining_flush || iree_status_is_ok(work_status))) {
      cleanup_status = iree_net_rdma_carrier_release_receive_lease(
          carrier, &lease, &pending_send_failure, &deactivate_callback,
          &deactivate_user_data);
    } else {
      iree_net_rdma_carrier_discard_receive_lease(carrier, &lease);
    }
    if (receive_queue_completed) {
      iree_slim_mutex_lock(&carrier->queue_mutex);
      iree_net_rdma_carrier_drop_pending_operations_locked(
          carrier, 1, &deactivate_callback, &deactivate_user_data);
      iree_slim_mutex_unlock(&carrier->queue_mutex);
    }

    iree_status_t completion_status =
        iree_status_join(work_status, cleanup_status);
    // Non-draining verbs failures are terminal for the QP. Complete the current
    // CQE first, then retire the rest of the local carrier state.
    iree_status_t terminal_failure_status =
        !draining_flush && !iree_status_is_ok(work_status)
            ? iree_status_clone(completion_status)
            : iree_ok_status();
    if (!iree_status_is_ok(completion_status)) {
      iree_net_rdma_carrier_notify_pending_send_failure(
          carrier, &pending_send_failure, completion_status);
    }
    if (!iree_status_is_ok(completion_status)) {
      if (iree_status_is_ok(terminal_failure_status)) {
        iree_net_rdma_carrier_notify_error(carrier, completion_status);
      } else {
        iree_status_free(completion_status);
      }
    }
    iree_net_rdma_carrier_defer_deactivate_callback(
        carrier, deactivate_callback, deactivate_user_data);
    if (!iree_status_is_ok(terminal_failure_status)) {
      iree_net_rdma_carrier_fail_all(carrier, terminal_failure_status);
    }
  }
}

static void iree_net_rdma_carrier_destroy(iree_net_carrier_t* base_carrier) {
  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_from_base(base_carrier);

  iree_net_rdma_carrier_free_memory_exports(carrier);
  iree_net_rdma_queue_pair_deinitialize(&carrier->queue_pair);
  iree_net_rdma_receive_queue_deinitialize(&carrier->receive_queue);
  iree_net_rdma_work_request_table_deinitialize(&carrier->work_request_table);
  iree_net_rdma_send_reservation_table_deinitialize(
      &carrier->send_reservation_table);
  if (iree_any_bit_set(carrier->flags,
                       IREE_NET_RDMA_CARRIER_FLAG_OWNS_RECV_POOL)) {
    iree_async_buffer_pool_release(carrier->recv_pool);
  }
  iree_async_buffer_pool_release(carrier->send_staging_pool);
  iree_net_rdma_carrier_consume_failure_status(carrier);
  iree_net_rdma_credit_memory_release(carrier->credit_grant_memory);
  iree_net_rdma_credit_memory_release(carrier->credit_memory);
  iree_net_rdma_completion_queue_release(carrier->recv_completion_queue);
  iree_net_rdma_completion_queue_release(carrier->send_completion_queue);
  iree_slim_mutex_deinitialize(&carrier->memory_export_mutex);
  iree_slim_mutex_deinitialize(&carrier->queue_mutex);
  if (carrier->connection_id &&
      iree_any_bit_set(carrier->flags,
                       IREE_NET_RDMA_CARRIER_FLAG_OWNS_CONNECTION_ID)) {
    const iree_net_librdmacm_t* librdmacm =
        iree_net_rdma_context_librdmacm(carrier->context);
    errno = 0;
    int result = librdmacm->rdma_destroy_id(carrier->connection_id);
    if (result != 0) {
      int error = result > 0 ? result : errno;
      iree_status_t status = iree_net_rdma_carrier_status_from_errno_required(
          __FILE__, __LINE__, error, "rdma_destroy_id");
      iree_status_abort(status);
    }
  }
  iree_async_proactor_release(carrier->proactor);
  iree_net_rdma_context_release(carrier->context);

  iree_allocator_t host_allocator = carrier->base.host_allocator;
  iree_allocator_free(host_allocator, carrier);
}

static void iree_net_rdma_carrier_set_recv_handler(
    iree_net_carrier_t* base_carrier, iree_net_carrier_recv_handler_t handler) {
  base_carrier->recv_handler = handler;
}

static iree_status_t iree_net_rdma_carrier_activate(
    iree_net_carrier_t* base_carrier) {
  if (iree_net_carrier_state(base_carrier) != IREE_NET_CARRIER_STATE_CREATED) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier is not in CREATED state");
  }
  if (!base_carrier->recv_handler.fn) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "recv handler must be set before activation");
  }
  iree_net_carrier_set_state(base_carrier, IREE_NET_CARRIER_STATE_ACTIVE);
  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_from_base(base_carrier);
  iree_status_t status =
      iree_net_rdma_completion_queue_activate(carrier->recv_completion_queue);
  if (!iree_status_is_ok(status)) {
    iree_net_carrier_set_state(base_carrier, IREE_NET_CARRIER_STATE_CREATED);
  }
  return status;
}

static iree_status_t iree_net_rdma_carrier_deactivate(
    iree_net_carrier_t* base_carrier,
    iree_net_carrier_deactivate_callback_fn_t callback, void* user_data) {
  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_from_base(base_carrier);
  iree_net_carrier_deactivate_callback_fn_t deactivate_callback = NULL;
  void* deactivate_user_data = NULL;
  iree_status_t status = iree_ok_status();
  int32_t pending = 0;
  bool drain_completion_queues = false;
  bool retire_created_receives = false;

  iree_slim_mutex_lock(&carrier->queue_mutex);
  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  bool already_deactivated = state == IREE_NET_CARRIER_STATE_DEACTIVATED;
  if (state == IREE_NET_CARRIER_STATE_DEACTIVATED) {
    deactivate_callback = callback;
    deactivate_user_data = user_data;
  } else if (state != IREE_NET_CARRIER_STATE_ACTIVE &&
             state != IREE_NET_CARRIER_STATE_CREATED) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "carrier must be in ACTIVE or CREATED state to deactivate");
  }

  if (!already_deactivated && iree_status_is_ok(status)) {
    carrier->deactivate_callback = callback;
    carrier->deactivate_user_data = user_data;
    iree_net_carrier_set_state(base_carrier, IREE_NET_CARRIER_STATE_DRAINING);

    iree_net_rdma_carrier_cancel_credit_wake_locked(carrier);
  }

  if (!already_deactivated && iree_status_is_ok(status)) {
    pending = iree_atomic_load(&base_carrier->pending_operations,
                               iree_memory_order_acquire);
    retire_created_receives =
        state == IREE_NET_CARRIER_STATE_CREATED && pending > 0;
    drain_completion_queues =
        state == IREE_NET_CARRIER_STATE_ACTIVE && pending > 0;
    if (pending > 0 &&
        !iree_any_bit_set(
            carrier->flags,
            IREE_NET_RDMA_CARRIER_FLAG_QUEUE_PAIR_ERROR_REQUESTED)) {
      status = iree_net_rdma_queue_pair_request_error(&carrier->queue_pair);
      if (iree_status_is_ok(status)) {
        carrier->flags |= IREE_NET_RDMA_CARRIER_FLAG_QUEUE_PAIR_ERROR_REQUESTED;
      } else {
        carrier->deactivate_callback = NULL;
        carrier->deactivate_user_data = NULL;
        iree_net_carrier_set_state(base_carrier, state);
      }
    }
  }

  if (!already_deactivated && iree_status_is_ok(status)) {
    if (retire_created_receives) {
      iree_net_rdma_carrier_retire_created_receives_locked(
          carrier, &deactivate_callback, &deactivate_user_data);
    } else {
      iree_net_rdma_carrier_maybe_complete_deactivation_locked(
          carrier, &deactivate_callback, &deactivate_user_data);
    }
  }
  iree_slim_mutex_unlock(&carrier->queue_mutex);

  if (!already_deactivated && iree_status_is_ok(status)) {
    iree_status_t cancellation_status =
        iree_status_from_code(IREE_STATUS_CANCELLED);
    iree_net_rdma_carrier_fail_pending_send_reservations(
        carrier, cancellation_status, &deactivate_callback,
        &deactivate_user_data);
    iree_status_free(cancellation_status);
  }

  if (iree_status_is_ok(status) && drain_completion_queues) {
    status =
        iree_net_rdma_completion_queue_drain(carrier->send_completion_queue);
  }
  if (iree_status_is_ok(status) && drain_completion_queues) {
    status =
        iree_net_rdma_completion_queue_drain(carrier->recv_completion_queue);
  }
  iree_net_rdma_carrier_invoke_deactivate_callback(deactivate_callback,
                                                   deactivate_user_data);
  return status;
}

static iree_net_carrier_send_budget_t iree_net_rdma_carrier_query_send_budget(
    iree_net_carrier_t* base_carrier) {
  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_from_base(base_carrier);
  iree_net_carrier_send_budget_t budget = {0};
  iree_slim_mutex_lock(&carrier->queue_mutex);
  if (iree_net_carrier_state(base_carrier) == IREE_NET_CARRIER_STATE_ACTIVE &&
      !iree_net_rdma_carrier_has_failure_status(carrier)) {
    if (!iree_any_bit_set(carrier->flags,
                          IREE_NET_RDMA_CARRIER_FLAG_SHUTDOWN_REQUESTED)) {
      iree_net_rdma_carrier_refresh_remote_recv_credits_locked(carrier);
      budget.bytes = carrier->remote_connection_data.recv_buffer_size;
      budget.slots = iree_net_rdma_send_window_available(
          &carrier->send_window,
          IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT);
      if (carrier->send_staging_pool) {
        iree_host_size_t staging_available =
            iree_async_buffer_pool_available(carrier->send_staging_pool);
        if (staging_available < budget.slots) {
          budget.slots = (uint32_t)staging_available;
        }
      }
    }
  }
  iree_slim_mutex_unlock(&carrier->queue_mutex);
  return budget;
}

static iree_status_t iree_net_rdma_carrier_validate_send_length_locked(
    iree_net_rdma_carrier_t* carrier, iree_host_size_t byte_length) {
  if (byte_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "send payload must not be empty");
  }
  if (!iree_any_bit_set(
          carrier->flags,
          IREE_NET_RDMA_CARRIER_FLAG_REMOTE_CONNECTION_DATA_APPLIED)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "RDMA remote connection data is not available");
  }
  uint32_t recv_buffer_size = carrier->remote_connection_data.recv_buffer_size;
  if (byte_length > recv_buffer_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "send payload size %" PRIhsz
                            " exceeds remote RDMA receive buffer size %" PRIu32,
                            byte_length, recv_buffer_size);
  }
  return iree_ok_status();
}

static iree_net_carrier_send_budget_t
iree_net_rdma_carrier_query_direct_write_budget(
    iree_net_carrier_t* base_carrier, iree_net_direct_write_flags_t flags) {
  iree_net_direct_write_flags_t known_flags =
      IREE_NET_DIRECT_WRITE_FLAG_SIGNAL_RECEIVER;
  iree_net_carrier_send_budget_t budget = {0};
  if (iree_any_bit_set(flags, ~known_flags)) return budget;
  if (!iree_all_bits_set(iree_net_carrier_capabilities(base_carrier),
                         IREE_NET_CARRIER_CAPABILITY_DIRECT_WRITE)) {
    return budget;
  }

  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_from_base(base_carrier);
  iree_net_rdma_send_window_acquire_flags_t acquire_flags =
      IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_NONE;
  if (iree_any_bit_set(flags, IREE_NET_DIRECT_WRITE_FLAG_SIGNAL_RECEIVER)) {
    acquire_flags = IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT;
  }

  iree_slim_mutex_lock(&carrier->queue_mutex);
  if (iree_net_carrier_state(base_carrier) == IREE_NET_CARRIER_STATE_ACTIVE &&
      !iree_net_rdma_carrier_has_failure_status(carrier)) {
    if (!iree_any_bit_set(carrier->flags,
                          IREE_NET_RDMA_CARRIER_FLAG_SHUTDOWN_REQUESTED)) {
      budget.bytes = IREE_HOST_SIZE_MAX;
      if (iree_any_bit_set(
              acquire_flags,
              IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT)) {
        iree_net_rdma_carrier_refresh_remote_recv_credits_locked(carrier);
      }
      budget.slots = iree_net_rdma_send_window_available(&carrier->send_window,
                                                         acquire_flags);
    }
  }
  iree_slim_mutex_unlock(&carrier->queue_mutex);
  return budget;
}

static iree_status_t iree_net_rdma_carrier_total_span_length(
    iree_async_span_list_t spans, iree_host_size_t* out_total_length) {
  *out_total_length = 0;
  iree_status_t status = iree_ok_status();
  if (spans.count > 0 && !spans.values) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "span list values must not be NULL");
  }
  for (iree_host_size_t i = 0; i < spans.count && iree_status_is_ok(status);
       ++i) {
    iree_host_size_t new_total = 0;
    if (iree_host_size_checked_add(*out_total_length, spans.values[i].length,
                                   &new_total)) {
      *out_total_length = new_total;
    } else {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "send span length total overflows host size");
    }
  }
  return status;
}

static bool iree_net_rdma_carrier_span_uses_rdma_region(
    iree_async_span_t span) {
  return span.region && span.region->type == IREE_ASYNC_REGION_TYPE_RDMA;
}

static iree_status_t iree_net_rdma_carrier_span_list_requires_staging(
    iree_async_span_list_t spans, bool* out_requires_staging) {
  *out_requires_staging = false;
  iree_status_t status = iree_ok_status();
  if (spans.count > 0 && !spans.values) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "span list values must not be NULL");
  }
  for (iree_host_size_t i = 0; i < spans.count && iree_status_is_ok(status);
       ++i) {
    if (!iree_net_rdma_carrier_span_uses_rdma_region(spans.values[i])) {
      *out_requires_staging = true;
      break;
    }
  }
  return status;
}

static iree_status_t iree_net_rdma_carrier_validate_staging_source_span(
    iree_async_span_t span, iree_host_size_t span_index) {
  if (!iree_async_span_is_cpu_accessible(span)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "send staging requires CPU-accessible span %" PRIhsz, span_index);
  }
  if (span.region) {
    bool in_range = span.offset <= span.region->length &&
                    span.length <= span.region->length - span.offset;
    if (!in_range) {
      iree_host_size_t span_end = span.length > SIZE_MAX - span.offset
                                      ? SIZE_MAX
                                      : span.offset + span.length;
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "send span %" PRIhsz " [%" PRIhsz ", %" PRIhsz
                              ") exceeds region length %" PRIhsz,
                              span_index, span.offset, span_end,
                              span.region->length);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_net_rdma_carrier_acquire_send_staging_buffer_locked(
    iree_net_rdma_carrier_t* carrier, iree_async_buffer_lease_t* out_lease) {
  memset(out_lease, 0, sizeof(*out_lease));
  return iree_async_buffer_pool_acquire(carrier->send_staging_pool, out_lease);
}

static iree_status_t iree_net_rdma_carrier_acquire_send_staging_buffer(
    iree_net_rdma_carrier_t* carrier, iree_async_buffer_lease_t* out_lease) {
  iree_slim_mutex_lock(&carrier->queue_mutex);
  iree_status_t status =
      iree_net_rdma_carrier_acquire_send_staging_buffer_locked(carrier,
                                                               out_lease);
  iree_slim_mutex_unlock(&carrier->queue_mutex);
  return status;
}

static iree_status_t iree_net_rdma_carrier_stage_send_data(
    iree_net_rdma_carrier_t* carrier, iree_async_span_list_t spans,
    iree_host_size_t total_length, iree_async_buffer_lease_t* out_lease,
    struct ibv_sge* out_scatter_gather_entry) {
  memset(out_lease, 0, sizeof(*out_lease));
  memset(out_scatter_gather_entry, 0, sizeof(*out_scatter_gather_entry));

  iree_status_t status = iree_ok_status();
  if (!carrier->send_staging_pool) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "RDMA send staging pool is not available");
  }

  iree_host_size_t buffer_size = 0;
  if (iree_status_is_ok(status)) {
    buffer_size =
        iree_async_buffer_pool_buffer_size(carrier->send_staging_pool);
    if (total_length > buffer_size) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "send payload size %" PRIhsz
                                " exceeds RDMA staging buffer size %" PRIhsz,
                                total_length, buffer_size);
    }
  }

  for (iree_host_size_t i = 0; i < spans.count && iree_status_is_ok(status);
       ++i) {
    status =
        iree_net_rdma_carrier_validate_staging_source_span(spans.values[i], i);
  }

  iree_async_buffer_lease_t lease;
  memset(&lease, 0, sizeof(lease));
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_acquire_send_staging_buffer(carrier, &lease);
  }

  if (iree_status_is_ok(status)) {
    uint8_t* target_ptr = iree_async_span_ptr(lease.span);
    for (iree_host_size_t i = 0; i < spans.count; ++i) {
      iree_async_span_t span = spans.values[i];
      if (span.length > 0) {
        memcpy(target_ptr, iree_async_span_ptr(span), span.length);
        target_ptr += span.length;
      }
    }

    iree_async_span_t staged_span = iree_async_span_make(
        lease.span.region, lease.span.offset, total_length);
    status = iree_net_rdma_sge_from_span(staged_span, out_scatter_gather_entry);
  }

  if (iree_status_is_ok(status)) {
    *out_lease = lease;
  } else {
    iree_async_buffer_lease_release(&lease);
  }
  return status;
}

// Posts one send-queue work request with queue_mutex held.
//
// |retained_buffer_lease|, when non-NULL, is consumed by this function: on
// successful post it is retained by the WR table until CQ completion, and on
// failed admission/post it is released before returning.
static iree_status_t iree_net_rdma_carrier_post_send_work_request_locked(
    iree_net_rdma_carrier_t* carrier,
    iree_net_rdma_work_request_operation_t operation,
    iree_net_rdma_send_window_acquire_flags_t acquire_flags, uint64_t user_data,
    iree_host_size_t byte_length, uint32_t pending_operation_delta,
    iree_async_buffer_lease_t* retained_buffer_lease,
    struct ibv_send_wr* work_request) {
  iree_status_t status = iree_ok_status();
  if (iree_net_carrier_state(&carrier->base) != IREE_NET_CARRIER_STATE_ACTIVE) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "carrier is not active");
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_get_failure_status(carrier);
  }
  if (iree_status_is_ok(status) &&
      iree_net_rdma_carrier_operation_requires_open_send_direction(operation) &&
      iree_any_bit_set(carrier->flags,
                       IREE_NET_RDMA_CARRIER_FLAG_SHUTDOWN_REQUESTED)) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "carrier send direction is shut down");
  }
  if (iree_status_is_ok(status)) {
    if (iree_any_bit_set(
            acquire_flags,
            IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT)) {
      iree_net_rdma_carrier_refresh_remote_recv_credits_locked(carrier);
    }
    status =
        iree_net_rdma_send_window_acquire(&carrier->send_window, acquire_flags);
    if (!iree_status_is_ok(status) &&
        iree_net_rdma_carrier_acquire_uses_remote_recv_credit(acquire_flags)) {
      status = iree_status_join(
          status, iree_net_rdma_carrier_arm_credit_wake_locked(carrier));
    }
  }
  bool send_window_acquired = iree_status_is_ok(status);

  uint64_t work_request_id = 0;
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_work_request_table_acquire(
        &carrier->work_request_table, operation, user_data, byte_length,
        retained_buffer_lease, &work_request_id);
  }

  if (iree_status_is_ok(status)) {
    work_request->wr_id = work_request_id;
    struct ibv_send_wr* bad_work_request = NULL;
    errno = 0;
    int result =
        ibv_post_send(iree_net_rdma_queue_pair_native_qp(&carrier->queue_pair),
                      work_request, &bad_work_request);
    if (result != 0) {
      int error = result > 0 ? result : errno;
      status = iree_net_rdma_carrier_status_from_errno_required(
          __FILE__, __LINE__, error, "ibv_post_send");
    }
  }
  if (iree_status_is_ok(status)) {
    iree_net_rdma_carrier_add_pending_operations(carrier,
                                                 pending_operation_delta);
  }

  if (!iree_status_is_ok(status)) {
    if (work_request_id != 0) {
      iree_net_rdma_work_request_completion_t completion;
      status = iree_status_join(
          status,
          iree_net_rdma_work_request_table_complete(
              &carrier->work_request_table, work_request_id, &completion));
      iree_async_buffer_lease_release(&completion.retained_buffer_lease);
    } else {
      iree_async_buffer_lease_release(retained_buffer_lease);
    }
    if (send_window_acquired) {
      status =
          iree_status_join(status, iree_net_rdma_send_window_abort(
                                       &carrier->send_window, acquire_flags));
    }
  }
  return status;
}

static iree_status_t iree_net_rdma_carrier_post_send_work_request(
    iree_net_rdma_carrier_t* carrier,
    iree_net_rdma_work_request_operation_t operation,
    iree_net_rdma_send_window_acquire_flags_t acquire_flags, uint64_t user_data,
    iree_host_size_t byte_length, uint32_t pending_operation_delta,
    iree_async_buffer_lease_t* retained_buffer_lease,
    struct ibv_send_wr* work_request) {
  iree_slim_mutex_lock(&carrier->queue_mutex);
  iree_status_t status = iree_net_rdma_carrier_post_send_work_request_locked(
      carrier, operation, acquire_flags, user_data, byte_length,
      pending_operation_delta, retained_buffer_lease, work_request);
  iree_slim_mutex_unlock(&carrier->queue_mutex);
  return status;
}

static bool iree_net_rdma_carrier_can_post_send_work_request_locked(
    iree_net_rdma_carrier_t* carrier,
    iree_net_rdma_send_window_acquire_flags_t acquire_flags) {
  if (iree_net_carrier_state(&carrier->base) != IREE_NET_CARRIER_STATE_ACTIVE) {
    return false;
  }
  if (iree_net_rdma_carrier_has_failure_status(carrier)) {
    return false;
  }
  if (iree_net_rdma_work_request_table_available_capacity(
          &carrier->work_request_table) == 0) {
    return false;
  }
  if (iree_net_rdma_carrier_acquire_uses_remote_recv_credit(acquire_flags)) {
    iree_net_rdma_carrier_refresh_remote_recv_credits_locked(carrier);
  }
  return iree_net_rdma_send_window_available(&carrier->send_window,
                                             acquire_flags) > 0;
}

static iree_status_t iree_net_rdma_carrier_sge_list_from_reservation(
    iree_net_rdma_send_reservation_t* reservation,
    iree_host_size_t scatter_gather_entry_capacity,
    struct ibv_sge* scatter_gather_entries,
    int* out_scatter_gather_entry_count) {
  *out_scatter_gather_entry_count = 0;
  switch (reservation->payload) {
    case IREE_NET_RDMA_SEND_RESERVATION_PAYLOAD_STAGED_BUFFER: {
      iree_async_span_t staged_span = iree_async_span_make(
          reservation->buffer_lease.span.region,
          reservation->buffer_lease.span.offset, reservation->byte_length);
      IREE_RETURN_IF_ERROR(
          iree_net_rdma_sge_from_span(staged_span, &scatter_gather_entries[0]));
      *out_scatter_gather_entry_count = 1;
      return iree_ok_status();
    }
    case IREE_NET_RDMA_SEND_RESERVATION_PAYLOAD_SPAN_LIST: {
      iree_async_span_list_t spans = iree_async_span_list_make(
          reservation->spans, reservation->span_count);
      return iree_net_rdma_sge_list_from_span_list(
          spans, scatter_gather_entry_capacity, scatter_gather_entries,
          out_scatter_gather_entry_count);
    }
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "reservation payload kind %u is not valid",
                              (uint32_t)reservation->payload);
  }
}

static iree_status_t iree_net_rdma_carrier_accept_staged_send(
    iree_net_rdma_carrier_t* carrier,
    iree_net_rdma_send_reservation_completion_t completion, uint64_t user_data,
    iree_host_size_t byte_length,
    iree_async_buffer_lease_t* retained_buffer_lease,
    struct ibv_sge* scatter_gather_entry) {
  iree_status_t status = iree_ok_status();
  iree_net_rdma_work_request_operation_t operation =
      IREE_NET_RDMA_WORK_REQUEST_OPERATION_NONE;
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_operation_from_send_reservation_completion(
        completion, &operation);
  }

  iree_status_t async_error_status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&carrier->queue_mutex);
    if (iree_net_carrier_state(&carrier->base) !=
        IREE_NET_CARRIER_STATE_ACTIVE) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "carrier is not active");
    }
    if (iree_status_is_ok(status) &&
        iree_any_bit_set(carrier->flags,
                         IREE_NET_RDMA_CARRIER_FLAG_SHUTDOWN_REQUESTED)) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "carrier send direction is shut down");
    }
    if (iree_status_is_ok(status)) {
      status = iree_net_rdma_carrier_get_failure_status(carrier);
    }
    if (iree_status_is_ok(status)) {
      status = iree_net_rdma_carrier_validate_send_length_locked(carrier,
                                                                 byte_length);
    }

    iree_net_rdma_send_window_acquire_flags_t acquire_flags =
        IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT;
    bool pending_fifo_empty =
        iree_net_rdma_send_reservation_table_pending_count(
            &carrier->send_reservation_table) == 0;
    bool can_post = iree_status_is_ok(status) && pending_fifo_empty &&
                    iree_net_rdma_carrier_can_post_send_work_request_locked(
                        carrier, acquire_flags);
    if (can_post) {
      struct ibv_send_wr work_request;
      memset(&work_request, 0, sizeof(work_request));
      work_request.sg_list = scatter_gather_entry;
      work_request.num_sge = 1;
      work_request.opcode = IBV_WR_SEND;
      work_request.send_flags = IBV_SEND_SIGNALED;

      status = iree_net_rdma_carrier_post_send_work_request_locked(
          carrier, operation, acquire_flags, user_data, byte_length,
          /*pending_operation_delta=*/1, retained_buffer_lease, &work_request);
    } else if (iree_status_is_ok(status)) {
      iree_net_carrier_send_handle_t handle = 0;
      status = iree_net_rdma_send_reservation_table_acquire(
          &carrier->send_reservation_table, retained_buffer_lease, byte_length,
          completion, user_data, &handle);
      bool reservation_acquired = iree_status_is_ok(status);
      if (iree_status_is_ok(status)) {
        status = iree_net_rdma_send_reservation_table_commit(
            &carrier->send_reservation_table, handle);
      }
      if (!iree_status_is_ok(status) && reservation_acquired) {
        iree_net_rdma_send_reservation_t reservation;
        memset(&reservation, 0, sizeof(reservation));
        iree_status_t resolve_status =
            iree_net_rdma_send_reservation_table_resolve(
                &carrier->send_reservation_table, handle, &reservation);
        status = iree_status_join(status, resolve_status);
        if (iree_status_is_ok(resolve_status)) {
          iree_async_buffer_lease_release(&reservation.buffer_lease);
        }
      }
      if (iree_status_is_ok(status)) {
        iree_net_rdma_carrier_add_pending_operations(carrier, 1);
        if (iree_net_rdma_carrier_remote_recv_credits_exhausted_locked(
                carrier)) {
          async_error_status =
              iree_net_rdma_carrier_arm_credit_wake_locked(carrier);
        }
      }
    }
    iree_slim_mutex_unlock(&carrier->queue_mutex);
  }

  if (!iree_status_is_ok(status)) {
    iree_async_buffer_lease_release(retained_buffer_lease);
  }
  if (!iree_status_is_ok(async_error_status)) {
    iree_net_rdma_carrier_notify_error(carrier, async_error_status);
  }
  return status;
}

static iree_status_t iree_net_rdma_carrier_accept_span_list_send(
    iree_net_rdma_carrier_t* carrier, iree_async_span_list_t spans,
    uint64_t user_data, iree_host_size_t byte_length,
    struct ibv_sge* scatter_gather_entries, int scatter_gather_entry_count) {
  iree_status_t async_error_status = iree_ok_status();
  iree_status_t status = iree_ok_status();

  iree_slim_mutex_lock(&carrier->queue_mutex);
  if (iree_net_carrier_state(&carrier->base) != IREE_NET_CARRIER_STATE_ACTIVE) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "carrier is not active");
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(carrier->flags,
                       IREE_NET_RDMA_CARRIER_FLAG_SHUTDOWN_REQUESTED)) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "carrier send direction is shut down");
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_get_failure_status(carrier);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_net_rdma_carrier_validate_send_length_locked(carrier, byte_length);
  }

  iree_net_rdma_send_window_acquire_flags_t acquire_flags =
      IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT;
  bool pending_fifo_empty = iree_net_rdma_send_reservation_table_pending_count(
                                &carrier->send_reservation_table) == 0;
  bool can_post = iree_status_is_ok(status) && pending_fifo_empty &&
                  iree_net_rdma_carrier_can_post_send_work_request_locked(
                      carrier, acquire_flags);
  if (can_post) {
    struct ibv_send_wr work_request;
    memset(&work_request, 0, sizeof(work_request));
    work_request.sg_list = scatter_gather_entries;
    work_request.num_sge = scatter_gather_entry_count;
    work_request.opcode = IBV_WR_SEND;
    work_request.send_flags = IBV_SEND_SIGNALED;

    status = iree_net_rdma_carrier_post_send_work_request_locked(
        carrier, IREE_NET_RDMA_WORK_REQUEST_OPERATION_SEND, acquire_flags,
        user_data, byte_length, /*pending_operation_delta=*/1,
        /*retained_buffer_lease=*/NULL, &work_request);
  } else if (iree_status_is_ok(status)) {
    iree_net_carrier_send_handle_t handle = 0;
    status = iree_net_rdma_send_reservation_table_acquire_span_list(
        &carrier->send_reservation_table, spans, byte_length,
        IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_SEND, user_data, &handle);
    bool reservation_acquired = iree_status_is_ok(status);
    if (iree_status_is_ok(status)) {
      status = iree_net_rdma_send_reservation_table_commit(
          &carrier->send_reservation_table, handle);
    }
    if (!iree_status_is_ok(status) && reservation_acquired) {
      iree_net_rdma_send_reservation_t reservation;
      memset(&reservation, 0, sizeof(reservation));
      iree_status_t resolve_status =
          iree_net_rdma_send_reservation_table_resolve(
              &carrier->send_reservation_table, handle, &reservation);
      status = iree_status_join(status, resolve_status);
      if (iree_status_is_ok(resolve_status)) {
        iree_async_buffer_lease_release(&reservation.buffer_lease);
      }
    }
    if (iree_status_is_ok(status)) {
      iree_net_rdma_carrier_add_pending_operations(carrier, 1);
      if (iree_net_rdma_carrier_remote_recv_credits_exhausted_locked(carrier)) {
        async_error_status =
            iree_net_rdma_carrier_arm_credit_wake_locked(carrier);
      }
    }
  }
  iree_slim_mutex_unlock(&carrier->queue_mutex);

  if (!iree_status_is_ok(async_error_status)) {
    iree_net_rdma_carrier_notify_error(carrier, async_error_status);
  }
  return status;
}

static iree_status_t
iree_net_rdma_carrier_try_post_pending_committed_sends_locked(
    iree_net_rdma_carrier_t* carrier, uint32_t* out_posted_count,
    iree_net_rdma_carrier_pending_send_failure_t* out_failed_send,
    iree_net_carrier_deactivate_callback_fn_t* inout_deactivate_callback,
    void** inout_deactivate_user_data) {
  if (out_posted_count) *out_posted_count = 0;
  iree_net_rdma_send_window_acquire_flags_t acquire_flags =
      IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT;

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         iree_net_rdma_send_reservation_table_pending_count(
             &carrier->send_reservation_table) > 0 &&
         iree_net_rdma_carrier_can_post_send_work_request_locked(
             carrier, acquire_flags)) {
    iree_net_carrier_send_handle_t pending_handle = 0;
    iree_net_rdma_send_reservation_t reservation_view;
    memset(&reservation_view, 0, sizeof(reservation_view));
    status = iree_net_rdma_send_reservation_table_peek_pending_front(
        &carrier->send_reservation_table, &pending_handle, &reservation_view);

    struct ibv_sge scatter_gather_entries[IREE_NET_RDMA_CARRIER_MAX_SEND_SGE];
    int scatter_gather_entry_count = 0;
    if (iree_status_is_ok(status)) {
      status = iree_net_rdma_carrier_sge_list_from_reservation(
          &reservation_view, IREE_ARRAYSIZE(scatter_gather_entries),
          scatter_gather_entries, &scatter_gather_entry_count);
    }
    if (iree_status_is_ok(status)) {
      status = iree_net_rdma_carrier_validate_send_length_locked(
          carrier, reservation_view.byte_length);
    }
    iree_net_rdma_work_request_operation_t operation =
        IREE_NET_RDMA_WORK_REQUEST_OPERATION_NONE;
    if (iree_status_is_ok(status)) {
      status = iree_net_rdma_carrier_operation_from_send_reservation_completion(
          reservation_view.completion, &operation);
    }

    iree_net_rdma_send_reservation_t reservation;
    memset(&reservation, 0, sizeof(reservation));
    if (iree_status_is_ok(status)) {
      iree_net_carrier_send_handle_t resolved_handle = 0;
      status = iree_net_rdma_send_reservation_table_resolve_pending_front(
          &carrier->send_reservation_table, &resolved_handle, &reservation);
      if (iree_status_is_ok(status) && resolved_handle != pending_handle) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "pending reservation FIFO resolved handle mismatch");
      }
    }
    if (iree_status_is_ok(status)) {
      struct ibv_send_wr work_request;
      memset(&work_request, 0, sizeof(work_request));
      work_request.sg_list = scatter_gather_entries;
      work_request.num_sge = scatter_gather_entry_count;
      work_request.opcode = IBV_WR_SEND;
      work_request.send_flags = IBV_SEND_SIGNALED;

      status = iree_net_rdma_carrier_post_send_work_request_locked(
          carrier, operation, acquire_flags, reservation.user_data,
          reservation.byte_length,
          /*pending_operation_delta=*/0, &reservation.buffer_lease,
          &work_request);
      if (iree_status_is_ok(status)) {
        if (out_posted_count) *out_posted_count += 1;
      } else {
        iree_net_rdma_carrier_capture_pending_send_failure(&reservation,
                                                           out_failed_send);
        iree_net_carrier_deactivate_callback_fn_t drop_callback = NULL;
        void* drop_user_data = NULL;
        iree_net_rdma_carrier_drop_pending_operations_locked(
            carrier, 1, &drop_callback, &drop_user_data);
        iree_net_rdma_carrier_capture_deactivate_callback(
            drop_callback, drop_user_data, inout_deactivate_callback,
            inout_deactivate_user_data);
      }
    } else {
      iree_async_buffer_lease_release(&reservation.buffer_lease);
    }
  }

  if (iree_status_is_ok(status) &&
      iree_net_rdma_send_reservation_table_pending_count(
          &carrier->send_reservation_table) > 0 &&
      iree_net_rdma_carrier_remote_recv_credits_exhausted_locked(carrier)) {
    status = iree_net_rdma_carrier_arm_credit_wake_locked(carrier);
  }
  return status;
}

static iree_status_t iree_net_rdma_carrier_try_replenish_receives_locked(
    iree_net_rdma_carrier_t* carrier,
    iree_net_rdma_carrier_pending_send_failure_t* out_failed_send,
    iree_net_carrier_deactivate_callback_fn_t* inout_deactivate_callback,
    void** inout_deactivate_user_data) {
  uint32_t posted_count = 0;
  iree_status_t status = iree_net_rdma_receive_queue_replenish(
      &carrier->receive_queue, carrier->options.recv_queue_depth,
      &posted_count);
  iree_net_rdma_carrier_add_pending_operations(carrier, posted_count);
  carrier->local_recv_credit_limit += posted_count;
  if (posted_count > 0 && iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_try_post_credit_grant_locked(carrier);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_try_post_pending_committed_sends_locked(
        carrier, /*out_posted_count=*/NULL, out_failed_send,
        inout_deactivate_callback, inout_deactivate_user_data);
  }
  return status;
}

// Posts at most one internal RDMA write publishing local receive credits.
static iree_status_t iree_net_rdma_carrier_try_post_credit_grant_locked(
    iree_net_rdma_carrier_t* carrier) {
  bool can_inline_credit_grant =
      iree_net_rdma_carrier_can_inline_credit_grant(carrier);
  bool should_post =
      iree_net_carrier_state(&carrier->base) == IREE_NET_CARRIER_STATE_ACTIVE &&
      iree_any_bit_set(
          carrier->flags,
          IREE_NET_RDMA_CARRIER_FLAG_REMOTE_CONNECTION_DATA_APPLIED);
  if (should_post && can_inline_credit_grant) {
    should_post = carrier->local_recv_credit_limit !=
                  carrier->local_recv_credit_submitted;
  } else if (should_post) {
    should_post =
        !iree_any_bit_set(carrier->flags,
                          IREE_NET_RDMA_CARRIER_FLAG_CREDIT_GRANT_IN_FLIGHT) &&
        carrier->local_recv_credit_limit !=
            carrier->local_recv_credit_published;
  }
  if (should_post) {
    should_post = iree_net_rdma_send_window_available(
                      &carrier->send_window,
                      IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_NONE) > 0;
  }
  if (should_post) {
    should_post = iree_net_rdma_work_request_table_available_capacity(
                      &carrier->work_request_table) > 0;
  }

  uint32_t credit_limit = 0;
  struct ibv_sge scatter_gather_entry;
  memset(&scatter_gather_entry, 0, sizeof(scatter_gather_entry));
  iree_status_t status = iree_ok_status();
  if (should_post) {
    credit_limit = carrier->local_recv_credit_limit;
    if (can_inline_credit_grant) {
      scatter_gather_entry.addr = (uint64_t)(uintptr_t)&credit_limit;
      scatter_gather_entry.length = sizeof(credit_limit);
      scatter_gather_entry.lkey = 0;
    } else {
      status = iree_net_rdma_credit_memory_store_sge(
          carrier->credit_grant_memory, credit_limit, &scatter_gather_entry);
    }
  }

  if (should_post && iree_status_is_ok(status)) {
    struct ibv_send_wr work_request;
    memset(&work_request, 0, sizeof(work_request));
    work_request.sg_list = &scatter_gather_entry;
    work_request.num_sge = 1;
    work_request.opcode = IBV_WR_RDMA_WRITE;
    work_request.send_flags = can_inline_credit_grant
                                  ? IBV_SEND_SIGNALED | IBV_SEND_INLINE
                                  : IBV_SEND_SIGNALED;
    work_request.wr.rdma.remote_addr =
        carrier->remote_connection_data.credit_memory.address;
    work_request.wr.rdma.rkey =
        carrier->remote_connection_data.credit_memory.rkey;

    status = iree_net_rdma_carrier_post_send_work_request_locked(
        carrier, IREE_NET_RDMA_WORK_REQUEST_OPERATION_CREDIT_GRANT,
        IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_NONE, credit_limit,
        sizeof(uint32_t), /*pending_operation_delta=*/1,
        /*retained_buffer_lease=*/NULL, &work_request);
  }

  if (should_post && iree_status_is_ok(status)) {
    carrier->local_recv_credit_submitted = credit_limit;
    if (!can_inline_credit_grant) {
      carrier->flags |= IREE_NET_RDMA_CARRIER_FLAG_CREDIT_GRANT_IN_FLIGHT;
    }
  }
  return status;
}

static iree_status_t iree_net_rdma_carrier_send(
    iree_net_carrier_t* base_carrier, const iree_net_send_params_t* params) {
  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_from_base(base_carrier);
  IREE_RETURN_IF_ERROR(iree_net_rdma_carrier_get_failure_status(carrier));
  if (iree_net_carrier_state(base_carrier) != IREE_NET_CARRIER_STATE_ACTIVE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier is not active");
  }
  if (!params) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "params must not be NULL");
  }
  if (iree_async_span_list_is_empty(params->data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "send data must not be empty");
  }
  iree_net_send_flags_t supported_flags =
      IREE_NET_SEND_FLAG_ZERO_COPY | IREE_NET_SEND_FLAG_END_OF_MESSAGE;
  if (iree_any_bit_set(params->flags, ~supported_flags)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "RDMA send flags 0x%08X are not supported",
                            params->flags & ~supported_flags);
  }
  if (params->data.count > base_carrier->max_iov) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "send requires %" PRIhsz
                            " spans but max_iov is %" PRIhsz,
                            params->data.count, base_carrier->max_iov);
  }

  bool requires_staging = false;
  iree_status_t status = iree_net_rdma_carrier_span_list_requires_staging(
      params->data, &requires_staging);

  iree_host_size_t total_length = 0;
  if (iree_status_is_ok(status)) {
    status =
        iree_net_rdma_carrier_total_span_length(params->data, &total_length);
  }
  if (iree_status_is_ok(status) && total_length == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "send payload must not be empty");
  }
  if (iree_status_is_ok(status) && requires_staging &&
      iree_any_bit_set(params->flags, IREE_NET_SEND_FLAG_ZERO_COPY)) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "RDMA ZERO_COPY send requires RDMA-registered spans");
  }

  struct ibv_sge scatter_gather_entries[IREE_NET_RDMA_CARRIER_MAX_SEND_SGE];
  int scatter_gather_entry_count = 0;
  iree_async_buffer_lease_t retained_buffer_lease;
  memset(&retained_buffer_lease, 0, sizeof(retained_buffer_lease));
  if (iree_status_is_ok(status) && requires_staging) {
    status = iree_net_rdma_carrier_stage_send_data(
        carrier, params->data, total_length, &retained_buffer_lease,
        &scatter_gather_entries[0]);
    if (iree_status_is_ok(status)) {
      scatter_gather_entry_count = 1;
    }
  } else if (iree_status_is_ok(status)) {
    status = iree_net_rdma_sge_list_from_span_list(
        params->data, IREE_ARRAYSIZE(scatter_gather_entries),
        scatter_gather_entries, &scatter_gather_entry_count);
  }

  if (iree_status_is_ok(status) && requires_staging) {
    status = iree_net_rdma_carrier_accept_staged_send(
        carrier, IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_SEND,
        params->user_data, total_length, &retained_buffer_lease,
        &scatter_gather_entries[0]);
  } else if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_accept_span_list_send(
        carrier, params->data, params->user_data, total_length,
        scatter_gather_entries, scatter_gather_entry_count);
  }
  if (!iree_status_is_ok(status)) {
    iree_async_buffer_lease_release(&retained_buffer_lease);
  }
  return status;
}

static iree_status_t iree_net_rdma_carrier_begin_send(
    iree_net_carrier_t* base_carrier, iree_host_size_t size, void** out_ptr,
    iree_net_carrier_send_handle_t* out_handle) {
  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_from_base(base_carrier);
  if (!out_ptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_ptr must not be NULL");
  }
  *out_ptr = NULL;
  if (!out_handle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_handle must not be NULL");
  }
  *out_handle = 0;
  if (iree_net_carrier_state(base_carrier) != IREE_NET_CARRIER_STATE_ACTIVE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier is not active");
  }
  IREE_RETURN_IF_ERROR(iree_net_rdma_carrier_get_failure_status(carrier));
  if (size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty sends are not allowed");
  }
  iree_host_size_t buffer_size =
      iree_async_buffer_pool_buffer_size(carrier->send_staging_pool);
  if (size > buffer_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "send payload size %" PRIhsz
                            " exceeds RDMA staging buffer size %" PRIhsz,
                            size, buffer_size);
  }
  iree_async_buffer_lease_t lease;
  memset(&lease, 0, sizeof(lease));
  iree_status_t status = iree_ok_status();
  void* buffer_ptr = NULL;
  iree_slim_mutex_lock(&carrier->queue_mutex);
  if (iree_net_carrier_state(base_carrier) != IREE_NET_CARRIER_STATE_ACTIVE) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "carrier is not active");
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(carrier->flags,
                       IREE_NET_RDMA_CARRIER_FLAG_SHUTDOWN_REQUESTED)) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "carrier send direction is shut down");
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_get_failure_status(carrier);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_validate_send_length_locked(carrier, size);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_acquire_send_staging_buffer_locked(carrier,
                                                                      &lease);
  }
  if (iree_status_is_ok(status)) {
    buffer_ptr = iree_async_span_ptr(lease.span);
    status = iree_net_rdma_send_reservation_table_acquire(
        &carrier->send_reservation_table, &lease, size,
        IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_INTERNAL,
        /*user_data=*/0, out_handle);
  }
  if (iree_status_is_ok(status)) {
    iree_net_rdma_carrier_add_pending_operations(carrier, 1);
  }
  iree_slim_mutex_unlock(&carrier->queue_mutex);
  if (iree_status_is_ok(status)) {
    *out_ptr = buffer_ptr;
  } else {
    iree_async_buffer_lease_release(&lease);
  }
  return status;
}

static iree_status_t iree_net_rdma_carrier_commit_send(
    iree_net_carrier_t* base_carrier, iree_net_carrier_send_handle_t handle) {
  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_from_base(base_carrier);

  iree_status_t async_error_status = iree_ok_status();
  iree_net_carrier_deactivate_callback_fn_t deactivate_callback = NULL;
  void* deactivate_user_data = NULL;
  iree_slim_mutex_lock(&carrier->queue_mutex);
  iree_net_rdma_send_reservation_t reservation_view;
  memset(&reservation_view, 0, sizeof(reservation_view));
  iree_status_t status = iree_net_rdma_send_reservation_table_peek(
      &carrier->send_reservation_table, handle, &reservation_view);
  bool handle_valid = iree_status_is_ok(status);
  bool reservation_consumed = false;
  struct ibv_sge scatter_gather_entry;
  if (iree_status_is_ok(status) &&
      iree_net_carrier_state(base_carrier) != IREE_NET_CARRIER_STATE_ACTIVE) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "carrier is not active");
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_get_failure_status(carrier);
  }
  if (iree_status_is_ok(status)) {
    iree_async_span_t staged_span =
        iree_async_span_make(reservation_view.buffer_lease.span.region,
                             reservation_view.buffer_lease.span.offset,
                             reservation_view.byte_length);
    status = iree_net_rdma_sge_from_span(staged_span, &scatter_gather_entry);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_validate_send_length_locked(
        carrier, reservation_view.byte_length);
  }

  if (iree_status_is_ok(status)) {
    iree_net_rdma_send_window_acquire_flags_t acquire_flags =
        IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT;
    bool pending_fifo_empty =
        iree_net_rdma_send_reservation_table_pending_count(
            &carrier->send_reservation_table) == 0;
    bool can_post = pending_fifo_empty &&
                    iree_net_rdma_carrier_can_post_send_work_request_locked(
                        carrier, acquire_flags);
    if (can_post) {
      iree_net_rdma_send_reservation_t reservation;
      memset(&reservation, 0, sizeof(reservation));
      status = iree_net_rdma_send_reservation_table_resolve(
          &carrier->send_reservation_table, handle, &reservation);
      if (iree_status_is_ok(status)) {
        reservation_consumed = true;
        struct ibv_send_wr work_request;
        memset(&work_request, 0, sizeof(work_request));
        work_request.sg_list = &scatter_gather_entry;
        work_request.num_sge = 1;
        work_request.opcode = IBV_WR_SEND;
        work_request.send_flags = IBV_SEND_SIGNALED;

        status = iree_net_rdma_carrier_post_send_work_request_locked(
            carrier, IREE_NET_RDMA_WORK_REQUEST_OPERATION_COMMITTED_SEND,
            acquire_flags, /*user_data=*/0, reservation.byte_length,
            /*pending_operation_delta=*/0, &reservation.buffer_lease,
            &work_request);
        if (!iree_status_is_ok(status)) {
          iree_net_rdma_carrier_drop_pending_operations_locked(
              carrier, 1, &deactivate_callback, &deactivate_user_data);
        }
      }
    } else {
      status = iree_net_rdma_send_reservation_table_commit(
          &carrier->send_reservation_table, handle);
      reservation_consumed = iree_status_is_ok(status);
      if (iree_status_is_ok(status) &&
          iree_net_rdma_carrier_remote_recv_credits_exhausted_locked(carrier)) {
        async_error_status =
            iree_net_rdma_carrier_arm_credit_wake_locked(carrier);
      }
    }
  }
  if (!iree_status_is_ok(status) && handle_valid && !reservation_consumed) {
    iree_net_rdma_send_reservation_t reservation;
    memset(&reservation, 0, sizeof(reservation));
    iree_status_t resolve_status = iree_net_rdma_send_reservation_table_resolve(
        &carrier->send_reservation_table, handle, &reservation);
    bool resolved_reservation = iree_status_is_ok(resolve_status);
    status = iree_status_join(status, resolve_status);
    if (resolved_reservation) {
      iree_async_buffer_lease_release(&reservation.buffer_lease);
      iree_net_rdma_carrier_drop_pending_operations_locked(
          carrier, 1, &deactivate_callback, &deactivate_user_data);
    }
  }
  iree_slim_mutex_unlock(&carrier->queue_mutex);
  if (!iree_status_is_ok(async_error_status)) {
    iree_net_rdma_carrier_notify_error(carrier, async_error_status);
  }
  iree_net_rdma_carrier_invoke_deactivate_callback(deactivate_callback,
                                                   deactivate_user_data);
  return status;
}

static void iree_net_rdma_carrier_abort_send(
    iree_net_carrier_t* base_carrier, iree_net_carrier_send_handle_t handle) {
  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_from_base(base_carrier);
  iree_net_carrier_deactivate_callback_fn_t deactivate_callback = NULL;
  void* deactivate_user_data = NULL;
  iree_slim_mutex_lock(&carrier->queue_mutex);
  iree_status_t status = iree_net_rdma_send_reservation_table_abort(
      &carrier->send_reservation_table, handle);
  if (iree_status_is_ok(status)) {
    iree_net_rdma_carrier_drop_pending_operations_locked(
        carrier, 1, &deactivate_callback, &deactivate_user_data);
  }
  iree_slim_mutex_unlock(&carrier->queue_mutex);
  if (!iree_status_is_ok(status)) {
    iree_status_abort(status);
  }
  iree_net_rdma_carrier_invoke_deactivate_callback(deactivate_callback,
                                                   deactivate_user_data);
}

static iree_status_t iree_net_rdma_carrier_shutdown(
    iree_net_carrier_t* base_carrier) {
  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_from_base(base_carrier);
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&carrier->queue_mutex);
  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE &&
      state != IREE_NET_CARRIER_STATE_DRAINING) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "carrier must be in ACTIVE or DRAINING state to shut down");
  }
  if (iree_status_is_ok(status)) {
    carrier->flags |= IREE_NET_RDMA_CARRIER_FLAG_SHUTDOWN_REQUESTED;
  }
  iree_slim_mutex_unlock(&carrier->queue_mutex);
  return status;
}

static iree_status_t iree_net_rdma_carrier_direct_write(
    iree_net_carrier_t* base_carrier,
    const iree_net_direct_write_params_t* params) {
  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_from_base(base_carrier);
  if (iree_net_carrier_state(base_carrier) != IREE_NET_CARRIER_STATE_ACTIVE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier is not active");
  }
  if (!iree_all_bits_set(iree_net_carrier_capabilities(base_carrier),
                         IREE_NET_CARRIER_CAPABILITY_DIRECT_WRITE)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "RDMA direct writes require memory windows");
  }
  if (!params) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "params must not be NULL");
  }
  iree_net_direct_write_flags_t known_flags =
      IREE_NET_DIRECT_WRITE_FLAG_SIGNAL_RECEIVER;
  if (iree_any_bit_set(params->flags, ~known_flags)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "RDMA direct_write flags 0x%08X are not supported",
                            params->flags & ~known_flags);
  }
  IREE_RETURN_IF_ERROR(iree_net_rdma_carrier_get_failure_status(carrier));
  if (iree_async_span_is_empty(params->local)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "direct_write local span must not be empty");
  }
  if (iree_net_remote_handle_is_null(params->remote)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "direct_write remote handle must not be null");
  }
  if (params->remote.opaque[0] > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "RDMA rkey 0x%016" PRIX64
                            " does not fit in uint32_t",
                            params->remote.opaque[0]);
  }

  struct ibv_sge scatter_gather_entry;
  iree_status_t status =
      iree_net_rdma_sge_from_span(params->local, &scatter_gather_entry);

  if (iree_status_is_ok(status)) {
    bool signal_receiver = iree_any_bit_set(
        params->flags, IREE_NET_DIRECT_WRITE_FLAG_SIGNAL_RECEIVER);
    struct ibv_send_wr work_request;
    memset(&work_request, 0, sizeof(work_request));
    work_request.sg_list = &scatter_gather_entry;
    work_request.num_sge = 1;
    work_request.opcode =
        signal_receiver ? IBV_WR_RDMA_WRITE_WITH_IMM : IBV_WR_RDMA_WRITE;
    work_request.send_flags = IBV_SEND_SIGNALED;
    work_request.imm_data = signal_receiver ? htonl(params->immediate) : 0;
    work_request.wr.rdma.remote_addr = params->remote.opaque[1];
    work_request.wr.rdma.rkey = (uint32_t)params->remote.opaque[0];

    iree_net_rdma_send_window_acquire_flags_t acquire_flags =
        signal_receiver
            ? IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT
            : IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_NONE;
    status = iree_net_rdma_carrier_post_send_work_request(
        carrier, IREE_NET_RDMA_WORK_REQUEST_OPERATION_DIRECT_WRITE,
        acquire_flags, params->user_data, params->local.length,
        /*pending_operation_delta=*/1,
        /*retained_buffer_lease=*/NULL, &work_request);
  }
  return status;
}

static iree_status_t iree_net_rdma_carrier_direct_read(
    iree_net_carrier_t* base_carrier,
    const iree_net_direct_read_params_t* params) {
  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_from_base(base_carrier);
  if (iree_net_carrier_state(base_carrier) != IREE_NET_CARRIER_STATE_ACTIVE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier is not active");
  }
  if (!iree_all_bits_set(iree_net_carrier_capabilities(base_carrier),
                         IREE_NET_CARRIER_CAPABILITY_DIRECT_READ)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "RDMA direct reads require memory windows");
  }
  if (!params) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "params must not be NULL");
  }
  IREE_RETURN_IF_ERROR(iree_net_rdma_carrier_get_failure_status(carrier));
  if (iree_async_span_is_empty(params->local)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "direct_read local span must not be empty");
  }
  if (iree_net_remote_handle_is_null(params->remote)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "direct_read remote handle must not be null");
  }
  if (params->remote.opaque[0] > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "RDMA rkey 0x%016" PRIX64
                            " does not fit in uint32_t",
                            params->remote.opaque[0]);
  }
  if (!params->local.region ||
      !iree_any_bit_set(params->local.region->access_flags,
                        IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE)) {
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "direct_read local span is not registered for local writes");
  }

  struct ibv_sge scatter_gather_entry;
  iree_status_t status =
      iree_net_rdma_sge_from_span(params->local, &scatter_gather_entry);

  if (iree_status_is_ok(status)) {
    struct ibv_send_wr work_request;
    memset(&work_request, 0, sizeof(work_request));
    work_request.sg_list = &scatter_gather_entry;
    work_request.num_sge = 1;
    work_request.opcode = IBV_WR_RDMA_READ;
    work_request.send_flags = IBV_SEND_SIGNALED;
    work_request.wr.rdma.remote_addr = params->remote.opaque[1];
    work_request.wr.rdma.rkey = (uint32_t)params->remote.opaque[0];

    status = iree_net_rdma_carrier_post_send_work_request(
        carrier, IREE_NET_RDMA_WORK_REQUEST_OPERATION_DIRECT_READ,
        IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_NONE, params->user_data,
        params->local.length, /*pending_operation_delta=*/1,
        /*retained_buffer_lease=*/NULL, &work_request);
  }
  return status;
}

static iree_status_t iree_net_rdma_carrier_allocate_memory_export(
    iree_net_rdma_carrier_t* carrier,
    iree_net_rdma_carrier_memory_export_t** out_memory_export) {
  *out_memory_export = NULL;
  iree_net_rdma_carrier_memory_export_t* memory_export = NULL;
  iree_status_t status =
      iree_allocator_malloc(carrier->base.host_allocator,
                            sizeof(*memory_export), (void**)&memory_export);
  if (iree_status_is_ok(status)) {
    memset(memory_export, 0, sizeof(*memory_export));
    status = iree_net_rdma_memory_window_allocate(
        carrier->context, carrier->base.host_allocator,
        &memory_export->memory_window);
  }
  if (iree_status_is_ok(status)) {
    *out_memory_export = memory_export;
  } else if (memory_export) {
    iree_net_rdma_memory_window_release(memory_export->memory_window);
    iree_allocator_free(carrier->base.host_allocator, memory_export);
  }
  return status;
}

static void iree_net_rdma_carrier_free_memory_export(
    iree_net_rdma_carrier_t* carrier,
    iree_net_rdma_carrier_memory_export_t* memory_export) {
  if (!memory_export) return;
  iree_net_rdma_memory_window_release(memory_export->memory_window);
  iree_allocator_free(carrier->base.host_allocator, memory_export);
}

static void iree_net_rdma_carrier_insert_memory_export(
    iree_net_rdma_carrier_t* carrier,
    iree_net_rdma_carrier_memory_export_t* memory_export) {
  iree_slim_mutex_lock(&carrier->memory_export_mutex);
  memory_export->next = carrier->memory_exports;
  carrier->memory_exports = memory_export;
  iree_slim_mutex_unlock(&carrier->memory_export_mutex);
}

static iree_net_rdma_carrier_memory_export_t*
iree_net_rdma_carrier_remove_memory_export(iree_net_rdma_carrier_t* carrier,
                                           iree_net_remote_handle_t handle) {
  iree_net_rdma_carrier_memory_export_t* removed_export = NULL;
  iree_slim_mutex_lock(&carrier->memory_export_mutex);
  iree_net_rdma_carrier_memory_export_t** link = &carrier->memory_exports;
  while (*link) {
    iree_net_rdma_carrier_memory_export_t* memory_export = *link;
    if (iree_net_rdma_memory_window_matches_handle(memory_export->memory_window,
                                                   handle)) {
      *link = memory_export->next;
      memory_export->next = NULL;
      removed_export = memory_export;
      break;
    }
    link = &memory_export->next;
  }
  iree_slim_mutex_unlock(&carrier->memory_export_mutex);
  return removed_export;
}

static void iree_net_rdma_carrier_free_memory_exports(
    iree_net_rdma_carrier_t* carrier) {
  iree_net_rdma_carrier_memory_export_t* memory_export =
      carrier->memory_exports;
  carrier->memory_exports = NULL;
  while (memory_export) {
    iree_net_rdma_carrier_memory_export_t* next = memory_export->next;
    iree_net_rdma_carrier_free_memory_export(carrier, memory_export);
    memory_export = next;
  }
}

static iree_status_t iree_net_rdma_carrier_post_memory_window_bind_locked(
    iree_net_rdma_carrier_t* carrier,
    iree_net_rdma_carrier_memory_export_t* memory_export,
    struct ibv_mr* memory_region, iree_async_region_t* region,
    iree_net_rdma_carrier_memory_window_bind_state_t* bind_state,
    iree_net_remote_handle_t* out_handle) {
  iree_status_t status = iree_ok_status();
  iree_net_carrier_state_t state = iree_net_carrier_state(&carrier->base);
  if (state != IREE_NET_CARRIER_STATE_CREATED &&
      state != IREE_NET_CARRIER_STATE_ACTIVE) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "carrier must be CREATED or ACTIVE for RDMA memory-window export");
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_get_failure_status(carrier);
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(carrier->flags,
                       IREE_NET_RDMA_CARRIER_FLAG_SHUTDOWN_REQUESTED)) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "carrier send direction is shut down");
  }

  bool send_window_acquired = false;
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_send_window_acquire(
        &carrier->send_window, IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_NONE);
    send_window_acquired = iree_status_is_ok(status);
  }

  uint64_t work_request_id = 0;
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_work_request_table_acquire(
        &carrier->work_request_table,
        IREE_NET_RDMA_WORK_REQUEST_OPERATION_MEMORY_WINDOW_BIND,
        (uint64_t)(uintptr_t)bind_state, /*byte_length=*/0,
        /*retained_buffer_lease=*/NULL, &work_request_id);
  }

  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_memory_window_post_bind(
        memory_export->memory_window, &carrier->queue_pair, memory_region,
        region->base_ptr, region->length, region->access_flags, work_request_id,
        out_handle);
  }
  if (iree_status_is_ok(status)) {
    iree_net_rdma_carrier_add_pending_operations(carrier, 1);
  }

  if (!iree_status_is_ok(status)) {
    if (work_request_id != 0) {
      iree_net_rdma_work_request_completion_t completion;
      status = iree_status_join(
          status,
          iree_net_rdma_work_request_table_complete(
              &carrier->work_request_table, work_request_id, &completion));
    }
    if (send_window_acquired) {
      status = iree_status_join(
          status, iree_net_rdma_send_window_abort(
                      &carrier->send_window,
                      IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_NONE));
    }
  }
  return status;
}

static iree_status_t iree_net_rdma_carrier_wait_for_memory_window_bind(
    iree_net_rdma_carrier_t* carrier,
    iree_net_rdma_carrier_memory_window_bind_state_t* bind_state) {
  iree_status_t status = iree_ok_status();
  while (
      !iree_net_rdma_carrier_memory_window_bind_state_completed(bind_state)) {
    iree_host_size_t completed_count = 0;
    iree_status_t poll_status = iree_async_proactor_poll(
        carrier->proactor, iree_infinite_timeout(), &completed_count);
    if (iree_status_is_deadline_exceeded(poll_status)) {
      iree_status_free(poll_status);
    } else if (!iree_status_is_ok(poll_status)) {
      status = iree_status_join(status, poll_status);
    }
  }
  status = iree_status_join(
      status,
      iree_net_rdma_carrier_memory_window_bind_state_consume(bind_state));
  return status;
}

static iree_status_t iree_net_rdma_carrier_register_buffer(
    iree_net_carrier_t* base_carrier, iree_async_region_t* region,
    iree_net_remote_handle_t* out_handle) {
  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_from_base(base_carrier);
  if (!out_handle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_handle must not be NULL");
  }
  *out_handle = iree_net_remote_handle_null();
  if (!region) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region must not be NULL");
  }
  IREE_RETURN_IF_ERROR(iree_net_rdma_carrier_get_failure_status(carrier));
  if (region->type != IREE_ASYNC_REGION_TYPE_RDMA) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region type %u is not RDMA",
                            (unsigned)region->type);
  }
  iree_async_buffer_access_flags_t remote_access_flags =
      IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_READ |
      IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_WRITE;
  if (!iree_any_bit_set(region->access_flags, remote_access_flags)) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "RDMA region is not registered for remote access");
  }
  if (!region->base_ptr) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "CPU-inaccessible RDMA regions need an explicit device IOVA");
  }
  if (region->length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region length must be non-zero");
  }
  iree_net_carrier_capabilities_t direct_capabilities =
      IREE_NET_CARRIER_CAPABILITY_DIRECT_WRITE |
      IREE_NET_CARRIER_CAPABILITY_DIRECT_READ;
  if (!iree_any_bit_set(iree_net_carrier_capabilities(base_carrier),
                        direct_capabilities)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "RDMA buffer registration requires memory windows");
  }

  struct ibv_mr* memory_region = (struct ibv_mr*)region->handles.rdma.mr;
  if (!memory_region) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "RDMA region has no memory registration");
  }
  if (memory_region->pd !=
      iree_net_rdma_context_protection_domain(carrier->context)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "RDMA region is not registered in the carrier protection domain");
  }

  iree_net_rdma_carrier_memory_export_t* memory_export = NULL;
  iree_status_t status =
      iree_net_rdma_carrier_allocate_memory_export(carrier, &memory_export);

  iree_net_rdma_carrier_memory_window_bind_state_t bind_state;
  iree_net_rdma_carrier_memory_window_bind_state_initialize(&bind_state);
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&carrier->queue_mutex);
    status = iree_net_rdma_carrier_post_memory_window_bind_locked(
        carrier, memory_export, memory_region, region, &bind_state, out_handle);
    iree_slim_mutex_unlock(&carrier->queue_mutex);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_net_rdma_carrier_wait_for_memory_window_bind(carrier, &bind_state);
  }
  if (iree_status_is_ok(status)) {
    iree_net_rdma_carrier_insert_memory_export(carrier, memory_export);
    memory_export = NULL;
  } else {
    *out_handle = iree_net_remote_handle_null();
  }
  iree_net_rdma_carrier_memory_window_bind_state_deinitialize(&bind_state);
  iree_net_rdma_carrier_free_memory_export(carrier, memory_export);
  return status;
}

static void iree_net_rdma_carrier_unregister_buffer(
    iree_net_carrier_t* base_carrier, iree_net_remote_handle_t handle) {
  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_from_base(base_carrier);
  iree_net_rdma_carrier_memory_export_t* memory_export =
      iree_net_rdma_carrier_remove_memory_export(carrier, handle);
  if (!memory_export) {
    iree_status_abort(iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                       "RDMA memory export handle not found"));
    return;
  }
  iree_net_rdma_carrier_free_memory_export(carrier, memory_export);
}

static const iree_net_carrier_vtable_t iree_net_rdma_carrier_vtable = {
    .destroy = iree_net_rdma_carrier_destroy,
    .set_recv_handler = iree_net_rdma_carrier_set_recv_handler,
    .activate = iree_net_rdma_carrier_activate,
    .deactivate = iree_net_rdma_carrier_deactivate,
    .query_send_budget = iree_net_rdma_carrier_query_send_budget,
    .send = iree_net_rdma_carrier_send,
    .begin_send = iree_net_rdma_carrier_begin_send,
    .commit_send = iree_net_rdma_carrier_commit_send,
    .abort_send = iree_net_rdma_carrier_abort_send,
    .shutdown = iree_net_rdma_carrier_shutdown,
    .direct_write = iree_net_rdma_carrier_direct_write,
    .direct_read = iree_net_rdma_carrier_direct_read,
    .register_buffer = iree_net_rdma_carrier_register_buffer,
    .unregister_buffer = iree_net_rdma_carrier_unregister_buffer,
    .query_direct_write_budget =
        iree_net_rdma_carrier_query_direct_write_budget,
};

static iree_status_t iree_net_rdma_carrier_create_send_staging_pool(
    iree_net_rdma_carrier_t* carrier) {
  iree_async_slab_options_t slab_options = iree_async_slab_options_default();
  slab_options.buffer_size = carrier->options.send_staging_buffer_size;
  slab_options.buffer_count = carrier->options.send_queue_depth;

  iree_async_slab_t* slab = NULL;
  iree_status_t status =
      iree_async_slab_create(slab_options, carrier->base.host_allocator, &slab);

  iree_async_region_t* region = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_region_register_slab(
        carrier->context, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ,
        carrier->base.host_allocator, &region);
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_buffer_pool_create(region, carrier->base.host_allocator,
                                           &carrier->send_staging_pool);
  }
  iree_async_region_release(region);
  iree_async_slab_release(slab);
  return status;
}

static iree_status_t iree_net_rdma_carrier_create_recv_pool(
    iree_net_rdma_carrier_t* carrier) {
  iree_async_slab_options_t slab_options = iree_async_slab_options_default();
  slab_options.buffer_size = carrier->options.recv_buffer_size;
  slab_options.buffer_count = carrier->options.recv_queue_depth;

  iree_async_slab_t* slab = NULL;
  iree_status_t status =
      iree_async_slab_create(slab_options, carrier->base.host_allocator, &slab);

  iree_async_region_t* region = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_region_register_slab(
        carrier->context, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE,
        carrier->base.host_allocator, &region);
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_buffer_pool_create(region, carrier->base.host_allocator,
                                           &carrier->recv_pool);
  }
  if (iree_status_is_ok(status)) {
    carrier->flags |= IREE_NET_RDMA_CARRIER_FLAG_OWNS_RECV_POOL;
  }
  iree_async_region_release(region);
  iree_async_slab_release(slab);
  return status;
}

static iree_status_t iree_net_rdma_carrier_validate_recv_pool(
    iree_net_rdma_carrier_t* carrier) {
  if (!carrier->recv_pool) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "recv_pool must not be NULL");
  }

  iree_async_region_t* region =
      iree_async_buffer_pool_region(carrier->recv_pool);
  if (!region) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "recv_pool region must not be NULL");
  }
  if (region->type != IREE_ASYNC_REGION_TYPE_RDMA) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "recv_pool region type %u is not RDMA",
                            (unsigned)region->type);
  }
  if (!iree_any_bit_set(region->access_flags,
                        IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE)) {
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "recv_pool RDMA region is not registered for local writes");
  }
  if (!region->base_ptr) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "CPU-inaccessible RDMA receive pools need an explicit message path");
  }
  iree_host_size_t buffer_count =
      iree_async_buffer_pool_capacity(carrier->recv_pool);
  if (buffer_count < carrier->options.recv_queue_depth) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "recv_pool capacity %" PRIhsz
        " is smaller than RDMA receive queue depth %" PRIu32,
        buffer_count, carrier->options.recv_queue_depth);
  }
  iree_host_size_t buffer_size =
      iree_async_buffer_pool_buffer_size(carrier->recv_pool);
  if (buffer_size == 0 || buffer_size > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "recv_pool buffer size %" PRIhsz
                            " is outside uint32_t range",
                            buffer_size);
  }

  struct ibv_mr* memory_region = (struct ibv_mr*)region->handles.rdma.mr;
  if (!memory_region) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "recv_pool RDMA region has no memory registration");
  }
  if (memory_region->pd !=
      iree_net_rdma_context_protection_domain(carrier->context)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "recv_pool RDMA region is not registered in the carrier protection "
        "domain");
  }
  return iree_ok_status();
}

static iree_status_t iree_net_rdma_carrier_create_completion_queues(
    iree_net_rdma_carrier_t* carrier) {
  iree_net_rdma_completion_queue_options_t send_options =
      iree_net_rdma_completion_queue_options_default();
  send_options.completion_capacity = (int)carrier->options.send_queue_depth;
  iree_net_rdma_completion_queue_callback_t send_callback = {
      iree_net_rdma_carrier_on_send_completions,
      carrier,
  };
  iree_status_t status = iree_net_rdma_completion_queue_create(
      carrier->context, carrier->proactor, send_options, send_callback,
      carrier->base.host_allocator, &carrier->send_completion_queue);

  if (iree_status_is_ok(status)) {
    iree_net_rdma_completion_queue_options_t recv_options =
        iree_net_rdma_completion_queue_options_default();
    recv_options.completion_capacity = (int)carrier->options.recv_queue_depth;
    recv_options.flags = IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_DEFER_ACTIVATION;
    iree_net_rdma_completion_queue_callback_t recv_callback = {
        iree_net_rdma_carrier_on_recv_completions,
        carrier,
    };
    status = iree_net_rdma_completion_queue_create(
        carrier->context, carrier->proactor, recv_options, recv_callback,
        carrier->base.host_allocator, &carrier->recv_completion_queue);
  }
  return status;
}

static iree_status_t iree_net_rdma_carrier_initialize_queue_pair(
    iree_net_rdma_carrier_t* carrier) {
  iree_net_rdma_queue_pair_options_t queue_pair_options = {
      .send_queue_depth = carrier->options.send_queue_depth,
      .recv_queue_depth = carrier->options.recv_queue_depth,
      .max_send_sge = carrier->options.max_send_sge,
      .max_recv_sge = carrier->options.max_recv_sge,
      .max_inline_data = carrier->options.max_inline_data,
      .signal_all_send_work_requests = true,
  };
  iree_net_rdma_queue_pair_initialize_params_t queue_pair_params = {
      .context = carrier->context,
      .connection_id = carrier->connection_id,
      .send_completion_queue = carrier->send_completion_queue,
      .recv_completion_queue = carrier->recv_completion_queue,
      .qp_context = carrier,
      .options = queue_pair_options,
  };
  return iree_net_rdma_queue_pair_initialize(queue_pair_params,
                                             &carrier->queue_pair);
}

static iree_status_t iree_net_rdma_carrier_initialize_local_connection_data(
    iree_net_rdma_carrier_t* carrier, uint32_t initial_recv_credits) {
  const struct ibv_qp_cap* capabilities =
      iree_net_rdma_queue_pair_capabilities(&carrier->queue_pair);
  if (!capabilities) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "queue pair capabilities are not available");
  }

  iree_net_rdma_connection_data_t connection_data;
  memset(&connection_data, 0, sizeof(connection_data));
  connection_data.send_queue_depth = capabilities->max_send_wr;
  connection_data.recv_queue_depth = capabilities->max_recv_wr;
  connection_data.recv_buffer_size =
      (uint32_t)iree_async_buffer_pool_buffer_size(carrier->recv_pool);
  connection_data.max_send_sge = capabilities->max_send_sge;
  if (connection_data.max_send_sge > IREE_NET_RDMA_CARRIER_MAX_SEND_SGE) {
    connection_data.max_send_sge = IREE_NET_RDMA_CARRIER_MAX_SEND_SGE;
  }
  connection_data.max_recv_sge = capabilities->max_recv_sge;
  connection_data.max_inline_data = capabilities->max_inline_data;
  connection_data.initial_recv_credits = initial_recv_credits;
  connection_data.credit_memory =
      iree_net_rdma_credit_memory_remote(carrier->credit_memory);
  carrier->local_connection_data = connection_data;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_net_rdma_carrier_create(
    iree_net_rdma_carrier_create_params_t params,
    iree_allocator_t host_allocator, iree_net_carrier_t** out_carrier) {
  if (!out_carrier) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_carrier must not be NULL");
  }
  *out_carrier = NULL;

  params.options = iree_net_rdma_carrier_resolve_options(params.options);
  iree_status_t status = iree_net_rdma_carrier_validate_options(params.options);
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_validate_params(params);
  }

  iree_net_rdma_carrier_t* carrier = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, sizeof(*carrier),
                                   (void**)&carrier);
  }

  if (iree_status_is_ok(status)) {
    memset(carrier, 0, sizeof(*carrier));
    iree_net_carrier_initialize(
        &iree_net_rdma_carrier_vtable,
        iree_net_rdma_carrier_capabilities(params.context),
        /*mtu=*/0, params.options.max_send_sge, params.callback, host_allocator,
        &carrier->base);
    carrier->context = params.context;
    iree_net_rdma_context_retain(params.context);
    carrier->proactor = params.proactor;
    iree_async_proactor_retain(params.proactor);
    carrier->recv_pool = params.recv_pool;
    iree_slim_mutex_initialize(&carrier->queue_mutex);
    iree_slim_mutex_initialize(&carrier->memory_export_mutex);
    carrier->connection_id = params.connection_id;
    carrier->options = params.options;
  }

  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_create_send_staging_pool(carrier);
  }
  if (iree_status_is_ok(status) && !carrier->recv_pool) {
    status = iree_net_rdma_carrier_create_recv_pool(carrier);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_validate_recv_pool(carrier);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_send_reservation_table_initialize(
        params.options.send_queue_depth, host_allocator,
        &carrier->send_reservation_table);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_create_completion_queues(carrier);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_initialize_queue_pair(carrier);
  }
  if (iree_status_is_ok(status)) {
    uint32_t work_request_capacity =
        params.options.send_queue_depth + params.options.recv_queue_depth;
    status = iree_net_rdma_work_request_table_initialize(
        work_request_capacity, host_allocator, &carrier->work_request_table);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_credit_memory_create(
        params.context, /*initial_credit_limit=*/0, host_allocator,
        &carrier->credit_memory);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_credit_memory_create(
        params.context, /*initial_credit_limit=*/0, host_allocator,
        &carrier->credit_grant_memory);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_receive_queue_initialize(
        &carrier->queue_pair, &carrier->work_request_table, carrier->recv_pool,
        params.options.recv_queue_depth, host_allocator,
        &carrier->receive_queue);
  }
  uint32_t posted_recv_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_receive_queue_replenish(
        &carrier->receive_queue, params.options.recv_queue_depth,
        &posted_recv_count);
    iree_net_rdma_carrier_add_pending_operations(carrier, posted_recv_count);
    carrier->local_recv_credit_limit += posted_recv_count;
    carrier->local_recv_credit_submitted = carrier->local_recv_credit_limit;
    carrier->local_recv_credit_published = carrier->local_recv_credit_limit;
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_send_window_initialize(
        params.options.send_queue_depth,
        /*initial_remote_recv_credits=*/0, &carrier->send_window);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_initialize_local_connection_data(
        carrier, carrier->local_recv_credit_limit);
  }

  if (iree_status_is_ok(status)) {
    carrier->flags |= IREE_NET_RDMA_CARRIER_FLAG_OWNS_CONNECTION_ID;
    *out_carrier = &carrier->base;
  } else if (carrier) {
    iree_net_rdma_carrier_destroy(&carrier->base);
  }
  return status;
}

IREE_API_EXPORT iree_net_rdma_carrier_t* iree_net_rdma_carrier_cast(
    iree_net_carrier_t* carrier) {
  return carrier ? iree_net_rdma_carrier_from_base(carrier) : NULL;
}

IREE_API_EXPORT iree_net_carrier_t* iree_net_rdma_carrier_as_generic(
    iree_net_rdma_carrier_t* carrier) {
  return carrier ? &carrier->base : NULL;
}

IREE_API_EXPORT struct rdma_cm_id* iree_net_rdma_carrier_connection_id(
    iree_net_rdma_carrier_t* carrier) {
  return carrier ? carrier->connection_id : NULL;
}

IREE_API_EXPORT iree_net_rdma_context_t* iree_net_rdma_carrier_context(
    iree_net_rdma_carrier_t* carrier) {
  return carrier ? carrier->context : NULL;
}

IREE_API_EXPORT iree_status_t iree_net_rdma_carrier_export_connection_data(
    iree_net_rdma_carrier_t* carrier,
    iree_net_rdma_connection_data_t* out_data) {
  if (!out_data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_data must not be NULL");
  }
  memset(out_data, 0, sizeof(*out_data));
  if (!carrier) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "carrier must not be NULL");
  }
  *out_data = carrier->local_connection_data;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_net_rdma_carrier_import_connection_data(
    iree_net_rdma_carrier_t* carrier,
    const iree_net_rdma_connection_data_t* data) {
  if (!carrier) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "carrier must not be NULL");
  }
  if (!data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "connection data must not be NULL");
  }
  IREE_RETURN_IF_ERROR(iree_net_rdma_connection_data_validate(data));

  iree_slim_mutex_lock(&carrier->queue_mutex);
  carrier->remote_connection_data = *data;
  iree_net_rdma_send_window_refresh_remote_credits(&carrier->send_window,
                                                   data->initial_recv_credits);
  iree_net_rdma_credit_memory_store(carrier->credit_memory,
                                    data->initial_recv_credits);
  carrier->flags |= IREE_NET_RDMA_CARRIER_FLAG_REMOTE_CONNECTION_DATA_APPLIED;
  iree_status_t status =
      iree_net_rdma_carrier_try_post_credit_grant_locked(carrier);
  iree_slim_mutex_unlock(&carrier->queue_mutex);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_net_rdma_carrier_send_bootstrap_connection_data(
    iree_net_rdma_carrier_t* carrier, uint32_t remote_recv_buffer_size,
    uint32_t remote_recv_credits) {
  if (!carrier) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "carrier must not be NULL");
  }
  if (iree_net_carrier_state(&carrier->base) != IREE_NET_CARRIER_STATE_ACTIVE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier is not active");
  }
  if (remote_recv_credits == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "remote_recv_credits must be non-zero");
  }
  if (remote_recv_buffer_size < IREE_NET_RDMA_CONNECTION_DATA_LENGTH) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "remote recv buffer size %" PRIu32
                            " is too small for connection data length %" PRIhsz,
                            remote_recv_buffer_size,
                            IREE_NET_RDMA_CONNECTION_DATA_LENGTH);
  }
  IREE_RETURN_IF_ERROR(iree_net_rdma_carrier_get_failure_status(carrier));

  uint8_t storage[IREE_NET_RDMA_CONNECTION_DATA_LENGTH];
  iree_host_size_t data_length = 0;
  IREE_RETURN_IF_ERROR(iree_net_rdma_carrier_serialize_local_connection_data(
      carrier, iree_make_byte_span(storage, sizeof(storage)), &data_length));

  iree_async_span_t span = iree_async_span_from_ptr(storage, data_length);
  iree_async_span_list_t span_list = iree_async_span_list_make(&span, 1);
  iree_async_buffer_lease_t retained_buffer_lease;
  memset(&retained_buffer_lease, 0, sizeof(retained_buffer_lease));
  struct ibv_sge scatter_gather_entry;
  iree_status_t status = iree_net_rdma_carrier_stage_send_data(
      carrier, span_list, data_length, &retained_buffer_lease,
      &scatter_gather_entry);

  bool retained_buffer_lease_submitted = false;
  if (iree_status_is_ok(status)) {
    struct ibv_send_wr work_request;
    memset(&work_request, 0, sizeof(work_request));
    work_request.sg_list = &scatter_gather_entry;
    work_request.num_sge = 1;
    work_request.opcode = IBV_WR_SEND;
    work_request.send_flags = IBV_SEND_SIGNALED;

    retained_buffer_lease_submitted = true;
    status = iree_net_rdma_carrier_post_send_work_request(
        carrier, IREE_NET_RDMA_WORK_REQUEST_OPERATION_BOOTSTRAP_SEND,
        IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_NONE, /*user_data=*/0,
        data_length, /*pending_operation_delta=*/1, &retained_buffer_lease,
        &work_request);
  }
  if (!iree_status_is_ok(status) && !retained_buffer_lease_submitted) {
    iree_async_buffer_lease_release(&retained_buffer_lease);
  }
  return status;
}

IREE_API_EXPORT iree_status_t
iree_net_rdma_carrier_complete_bootstrap_connection_data(
    iree_net_rdma_carrier_t* carrier) {
  if (!carrier) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "carrier must not be NULL");
  }
  if (iree_net_carrier_state(&carrier->base) != IREE_NET_CARRIER_STATE_ACTIVE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier is not active");
  }
  carrier->base.recv_handler = (iree_net_carrier_recv_handler_t){0};
  iree_net_carrier_set_state(&carrier->base, IREE_NET_CARRIER_STATE_CREATED);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_net_rdma_carrier_serialize_local_connection_data(
    iree_net_rdma_carrier_t* carrier, iree_byte_span_t target,
    iree_host_size_t* out_length) {
  if (!carrier) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "carrier must not be NULL");
  }
  iree_net_rdma_connection_data_t connection_data;
  IREE_RETURN_IF_ERROR(
      iree_net_rdma_carrier_export_connection_data(carrier, &connection_data));
  return iree_net_rdma_connection_data_serialize(&connection_data, target,
                                                 out_length);
}

IREE_API_EXPORT iree_status_t
iree_net_rdma_carrier_apply_remote_connection_data(
    iree_net_rdma_carrier_t* carrier, iree_const_byte_span_t source) {
  if (!carrier) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "carrier must not be NULL");
  }

  iree_net_rdma_connection_data_t connection_data;
  iree_status_t status =
      iree_net_rdma_connection_data_deserialize(source, &connection_data);
  if (iree_status_is_ok(status)) {
    status =
        iree_net_rdma_carrier_import_connection_data(carrier, &connection_data);
  }
  return status;
}
