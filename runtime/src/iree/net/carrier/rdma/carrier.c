// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/carrier.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>

#include "iree/async/slab.h"
#include "iree/base/threading/mutex.h"
#include "iree/net/carrier/rdma/completion_queue.h"
#include "iree/net/carrier/rdma/credit_memory.h"
#include "iree/net/carrier/rdma/queue_pair.h"
#include "iree/net/carrier/rdma/receive_queue.h"
#include "iree/net/carrier/rdma/region.h"
#include "iree/net/carrier/rdma/send_reservation_table.h"
#include "iree/net/carrier/rdma/send_window.h"
#include "iree/net/carrier/rdma/sge.h"
#include "iree/net/carrier/rdma/work_request_table.h"

typedef uint8_t iree_net_rdma_carrier_flags_t;
enum iree_net_rdma_carrier_flag_bits_e {
  IREE_NET_RDMA_CARRIER_FLAG_OWNS_CONNECTION_ID = 1u << 0,
  IREE_NET_RDMA_CARRIER_FLAG_REMOTE_CONNECTION_DATA_APPLIED = 1u << 1,
  IREE_NET_RDMA_CARRIER_FLAG_QUEUE_PAIR_ERROR_REQUESTED = 1u << 2,
  IREE_NET_RDMA_CARRIER_FLAG_CREDIT_GRANT_IN_FLIGHT = 1u << 3,
};

struct iree_net_rdma_carrier_t {
  // Base carrier; must be first for upcasting.
  iree_net_carrier_t base;

  // RDMA context retained by the carrier.
  iree_net_rdma_context_t* context;

  // Proactor retained while completion queues are registered.
  iree_async_proactor_t* proactor;

  // Borrowed receive buffer pool supplying posted receive buffers.
  iree_async_buffer_pool_t* recv_pool;

  // Serializes native QP posting and fixed queue bookkeeping.
  iree_slim_mutex_t queue_mutex;

  // Callback invoked after DRAINING retires all pending operations.
  iree_net_carrier_deactivate_callback_fn_t deactivate_callback;

  // Opaque user data passed to deactivate_callback.
  void* deactivate_user_data;

  // Owned RDMA-registered pool for staging non-RDMA CPU send spans.
  iree_async_buffer_pool_t* send_staging_pool;

  // Reservations of staging-pool leases between begin_send and commit/abort.
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

  // Local private data serialized during rdma_cm connect/accept.
  iree_net_rdma_connection_data_t local_connection_data;

  // Peer private data applied after rdma_cm connect/accept.
  iree_net_rdma_connection_data_t remote_connection_data;

  // Normalized queue and inline-data options.
  iree_net_rdma_carrier_options_t options;

  // Bitfield of iree_net_rdma_carrier_flag_bits_e values.
  iree_net_rdma_carrier_flags_t flags;
};

static const iree_net_carrier_vtable_t iree_net_rdma_carrier_vtable;

static iree_status_t iree_net_rdma_carrier_try_post_credit_grant_locked(
    iree_net_rdma_carrier_t* carrier);

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
  if (!params.recv_pool) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "recv_pool must not be NULL");
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
    void) {
  return IREE_NET_CARRIER_CAPABILITY_RELIABLE |
         IREE_NET_CARRIER_CAPABILITY_ORDERED |
         IREE_NET_CARRIER_CAPABILITY_ZERO_COPY_TX |
         IREE_NET_CARRIER_CAPABILITY_ZERO_COPY_RX |
         IREE_NET_CARRIER_CAPABILITY_REGISTERED_REGIONS |
         IREE_NET_CARRIER_CAPABILITY_DIRECT_WRITE |
         IREE_NET_CARRIER_CAPABILITY_DIRECT_READ;
}

static void iree_net_rdma_carrier_notify_error(iree_net_rdma_carrier_t* carrier,
                                               iree_status_t status) {
  carrier->base.callback.fn(carrier->base.callback.user_data,
                            IREE_NET_CARRIER_COMPLETION_ERROR,
                            /*operation_user_data=*/0, status,
                            /*bytes_transferred=*/0, /*recv_lease=*/NULL);
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

static void iree_net_rdma_carrier_on_send_completions(
    void* user_data, iree_status_t status, const struct ibv_wc* completions,
    iree_host_size_t completion_count) {
  iree_net_rdma_carrier_t* carrier = (iree_net_rdma_carrier_t*)user_data;
  if (!iree_status_is_ok(status)) {
    iree_net_rdma_carrier_notify_error(carrier, status);
    return;
  }

  for (iree_host_size_t i = 0; i < completion_count; ++i) {
    const struct ibv_wc* completion = &completions[i];
    iree_net_rdma_work_request_completion_t work_request_completion;
    memset(&work_request_completion, 0, sizeof(work_request_completion));
    bool draining_flush =
        iree_net_rdma_carrier_work_completion_is_draining_flush(carrier,
                                                                completion);
    iree_status_t work_status =
        draining_flush ? iree_ok_status()
                       : iree_net_rdma_carrier_status_from_work_completion(
                             completion, "send");
    iree_net_carrier_deactivate_callback_fn_t deactivate_callback = NULL;
    void* deactivate_user_data = NULL;
    bool send_queue_completed = false;

    iree_slim_mutex_lock(&carrier->queue_mutex);
    iree_status_t cleanup_status = iree_net_rdma_work_request_table_complete(
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
            carrier->local_recv_credit_submitted;
      }
    }
    if (!draining_flush && iree_status_is_ok(work_status) &&
        iree_status_is_ok(cleanup_status)) {
      cleanup_status =
          iree_net_rdma_carrier_try_post_credit_grant_locked(carrier);
    }
    if (send_queue_completed) {
      iree_net_rdma_carrier_drop_pending_operations_locked(
          carrier, 1, &deactivate_callback, &deactivate_user_data);
    }
    iree_slim_mutex_unlock(&carrier->queue_mutex);

    iree_async_buffer_lease_release(
        &work_request_completion.retained_buffer_lease);
    if (!draining_flush && iree_status_is_ok(work_status) &&
        iree_status_is_ok(cleanup_status) &&
        work_request_completion.operation !=
            IREE_NET_RDMA_WORK_REQUEST_OPERATION_CREDIT_GRANT) {
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
    if (draining_flush &&
        work_request_completion.operation !=
            IREE_NET_RDMA_WORK_REQUEST_OPERATION_COMMITTED_SEND &&
        work_request_completion.operation !=
            IREE_NET_RDMA_WORK_REQUEST_OPERATION_CREDIT_GRANT) {
      work_status = iree_make_status(
          IREE_STATUS_CANCELLED, "RDMA send cancelled by carrier deactivation");
    }
    iree_status_t completion_status =
        iree_status_join(work_status, cleanup_status);
    iree_host_size_t bytes_transferred =
        iree_status_is_ok(completion_status)
            ? work_request_completion.byte_length
            : 0;
    bool internal_completion =
        work_request_completion.operation ==
            IREE_NET_RDMA_WORK_REQUEST_OPERATION_COMMITTED_SEND ||
        work_request_completion.operation ==
            IREE_NET_RDMA_WORK_REQUEST_OPERATION_CREDIT_GRANT;
    if (internal_completion) {
      if (!iree_status_is_ok(completion_status)) {
        iree_net_rdma_carrier_notify_error(carrier, completion_status);
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
    iree_net_rdma_carrier_invoke_deactivate_callback(deactivate_callback,
                                                     deactivate_user_data);
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

static void iree_net_rdma_carrier_on_recv_completions(
    void* user_data, iree_status_t status, const struct ibv_wc* completions,
    iree_host_size_t completion_count) {
  iree_net_rdma_carrier_t* carrier = (iree_net_rdma_carrier_t*)user_data;
  if (!iree_status_is_ok(status)) {
    iree_net_rdma_carrier_notify_error(carrier, status);
    return;
  }

  for (iree_host_size_t i = 0; i < completion_count; ++i) {
    const struct ibv_wc* completion = &completions[i];
    iree_net_rdma_work_request_completion_t work_request_completion;
    memset(&work_request_completion, 0, sizeof(work_request_completion));
    bool draining_flush =
        iree_net_rdma_carrier_work_completion_is_draining_flush(carrier,
                                                                completion);
    iree_status_t work_status =
        draining_flush ? iree_ok_status()
                       : iree_net_rdma_carrier_status_from_work_completion(
                             completion, "recv");
    iree_net_carrier_deactivate_callback_fn_t deactivate_callback = NULL;
    void* deactivate_user_data = NULL;
    bool receive_queue_completed = false;

    iree_slim_mutex_lock(&carrier->queue_mutex);
    iree_status_t cleanup_status = iree_net_rdma_work_request_table_complete(
        &carrier->work_request_table, completion->wr_id,
        &work_request_completion);
    if (iree_status_is_ok(cleanup_status) &&
        work_request_completion.operation !=
            IREE_NET_RDMA_WORK_REQUEST_OPERATION_RECV) {
      cleanup_status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                        "RDMA receive CQ returned operation %u",
                                        work_request_completion.operation);
    }

    iree_async_buffer_lease_t lease;
    memset(&lease, 0, sizeof(lease));
    if (iree_status_is_ok(cleanup_status)) {
      iree_host_size_t byte_length =
          iree_status_is_ok(work_status)
              ? (iree_host_size_t)completion->byte_len
              : 0;
      cleanup_status = iree_net_rdma_receive_queue_complete(
          &carrier->receive_queue, work_request_completion, byte_length,
          &lease);
      receive_queue_completed = iree_status_is_ok(cleanup_status);
    }
    iree_slim_mutex_unlock(&carrier->queue_mutex);

    if (!draining_flush && iree_status_is_ok(work_status) &&
        iree_status_is_ok(cleanup_status)) {
      work_status = iree_net_rdma_carrier_deliver_receive(carrier, &lease);
    }
    iree_async_buffer_lease_release(&lease);

    if (!draining_flush && iree_status_is_ok(work_status) &&
        iree_status_is_ok(cleanup_status)) {
      uint32_t posted_count = 0;
      iree_slim_mutex_lock(&carrier->queue_mutex);
      cleanup_status = iree_net_rdma_receive_queue_replenish(
          &carrier->receive_queue, carrier->options.recv_queue_depth,
          &posted_count);
      iree_net_rdma_carrier_add_pending_operations(carrier, posted_count);
      carrier->local_recv_credit_limit += posted_count;
      if (posted_count > 0 && iree_status_is_ok(cleanup_status)) {
        cleanup_status =
            iree_net_rdma_carrier_try_post_credit_grant_locked(carrier);
      }
      iree_slim_mutex_unlock(&carrier->queue_mutex);
    }
    if (receive_queue_completed) {
      iree_slim_mutex_lock(&carrier->queue_mutex);
      iree_net_rdma_carrier_drop_pending_operations_locked(
          carrier, 1, &deactivate_callback, &deactivate_user_data);
      iree_slim_mutex_unlock(&carrier->queue_mutex);
    }

    iree_status_t completion_status =
        iree_status_join(work_status, cleanup_status);
    if (!iree_status_is_ok(completion_status)) {
      iree_net_rdma_carrier_notify_error(carrier, completion_status);
    }
    iree_net_rdma_carrier_invoke_deactivate_callback(deactivate_callback,
                                                     deactivate_user_data);
  }
}

static void iree_net_rdma_carrier_destroy(iree_net_carrier_t* base_carrier) {
  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_from_base(base_carrier);

  iree_net_rdma_queue_pair_deinitialize(&carrier->queue_pair);
  iree_net_rdma_receive_queue_deinitialize(&carrier->receive_queue);
  iree_net_rdma_work_request_table_deinitialize(&carrier->work_request_table);
  iree_net_rdma_send_reservation_table_deinitialize(
      &carrier->send_reservation_table);
  iree_async_buffer_pool_free(carrier->send_staging_pool);
  iree_net_rdma_credit_memory_release(carrier->credit_grant_memory);
  iree_net_rdma_credit_memory_release(carrier->credit_memory);
  iree_net_rdma_completion_queue_release(carrier->recv_completion_queue);
  iree_net_rdma_completion_queue_release(carrier->send_completion_queue);
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
  return iree_ok_status();
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
  }

  if (!already_deactivated && iree_status_is_ok(status)) {
    pending = iree_atomic_load(&base_carrier->pending_operations,
                               iree_memory_order_acquire);
    drain_completion_queues = pending > 0;
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
    iree_net_rdma_carrier_maybe_complete_deactivation_locked(
        carrier, &deactivate_callback, &deactivate_user_data);
  }
  iree_slim_mutex_unlock(&carrier->queue_mutex);

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
  iree_net_carrier_send_budget_t budget;
  budget.bytes = IREE_HOST_SIZE_MAX;
  budget.slots = 0;
  iree_slim_mutex_lock(&carrier->queue_mutex);
  if (iree_net_carrier_state(base_carrier) == IREE_NET_CARRIER_STATE_ACTIVE) {
    iree_net_rdma_carrier_refresh_remote_recv_credits_locked(carrier);
    budget.slots = iree_net_rdma_send_window_available(
        &carrier->send_window,
        IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT);
  }
  iree_slim_mutex_unlock(&carrier->queue_mutex);
  if (carrier->send_staging_pool) {
    iree_host_size_t staging_available =
        iree_async_buffer_pool_available(carrier->send_staging_pool);
    if (staging_available < budget.slots) {
      budget.slots = (uint32_t)staging_available;
    }
  }
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
    status = iree_async_buffer_pool_acquire(carrier->send_staging_pool, &lease);
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
    if (iree_any_bit_set(
            acquire_flags,
            IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT)) {
      iree_net_rdma_carrier_refresh_remote_recv_credits_locked(carrier);
    }
    status =
        iree_net_rdma_send_window_acquire(&carrier->send_window, acquire_flags);
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

// Posts at most one internal RDMA write publishing local receive credits.
static iree_status_t iree_net_rdma_carrier_try_post_credit_grant_locked(
    iree_net_rdma_carrier_t* carrier) {
  bool should_post =
      iree_net_carrier_state(&carrier->base) == IREE_NET_CARRIER_STATE_ACTIVE &&
      iree_any_bit_set(
          carrier->flags,
          IREE_NET_RDMA_CARRIER_FLAG_REMOTE_CONNECTION_DATA_APPLIED) &&
      !iree_any_bit_set(carrier->flags,
                        IREE_NET_RDMA_CARRIER_FLAG_CREDIT_GRANT_IN_FLIGHT) &&
      carrier->local_recv_credit_limit != carrier->local_recv_credit_published;
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
  iree_status_t status = iree_ok_status();
  if (should_post) {
    credit_limit = carrier->local_recv_credit_limit;
    status = iree_net_rdma_credit_memory_store_sge(
        carrier->credit_grant_memory, credit_limit, &scatter_gather_entry);
  }

  if (should_post && iree_status_is_ok(status)) {
    struct ibv_send_wr work_request;
    memset(&work_request, 0, sizeof(work_request));
    work_request.sg_list = &scatter_gather_entry;
    work_request.num_sge = 1;
    work_request.opcode = IBV_WR_RDMA_WRITE;
    work_request.send_flags = IBV_SEND_SIGNALED;
    work_request.wr.rdma.remote_addr =
        carrier->remote_connection_data.credit_memory.address;
    work_request.wr.rdma.rkey =
        carrier->remote_connection_data.credit_memory.rkey;

    status = iree_net_rdma_carrier_post_send_work_request_locked(
        carrier, IREE_NET_RDMA_WORK_REQUEST_OPERATION_CREDIT_GRANT,
        IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_NONE, /*user_data=*/0,
        sizeof(uint32_t), /*pending_operation_delta=*/1,
        /*retained_buffer_lease=*/NULL, &work_request);
  }

  if (should_post && iree_status_is_ok(status)) {
    carrier->local_recv_credit_submitted = credit_limit;
    carrier->flags |= IREE_NET_RDMA_CARRIER_FLAG_CREDIT_GRANT_IN_FLIGHT;
  }
  return status;
}

static iree_status_t iree_net_rdma_carrier_send(
    iree_net_carrier_t* base_carrier, const iree_net_send_params_t* params) {
  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_from_base(base_carrier);
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

  if (iree_status_is_ok(status)) {
    struct ibv_send_wr work_request;
    memset(&work_request, 0, sizeof(work_request));
    work_request.sg_list = scatter_gather_entries;
    work_request.num_sge = scatter_gather_entry_count;
    work_request.opcode = IBV_WR_SEND;
    work_request.send_flags = IBV_SEND_SIGNALED;

    status = iree_net_rdma_carrier_post_send_work_request(
        carrier, IREE_NET_RDMA_WORK_REQUEST_OPERATION_SEND,
        IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT,
        params->user_data, total_length, /*pending_operation_delta=*/1,
        &retained_buffer_lease, &work_request);
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
  iree_status_t status =
      iree_async_buffer_pool_acquire(carrier->send_staging_pool, &lease);
  void* buffer_ptr = NULL;
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&carrier->queue_mutex);
    if (iree_net_carrier_state(base_carrier) != IREE_NET_CARRIER_STATE_ACTIVE) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "carrier is not active");
    }
    if (iree_status_is_ok(status)) {
      buffer_ptr = iree_async_span_ptr(lease.span);
      status = iree_net_rdma_send_reservation_table_acquire(
          &carrier->send_reservation_table, &lease, size, out_handle);
    }
    if (iree_status_is_ok(status)) {
      iree_net_rdma_carrier_add_pending_operations(carrier, 1);
    }
    iree_slim_mutex_unlock(&carrier->queue_mutex);
  }
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

  iree_net_rdma_send_reservation_t reservation;
  memset(&reservation, 0, sizeof(reservation));

  bool reservation_resolved = false;
  iree_slim_mutex_lock(&carrier->queue_mutex);
  iree_status_t status = iree_net_rdma_send_reservation_table_resolve(
      &carrier->send_reservation_table, handle, &reservation);
  reservation_resolved = iree_status_is_ok(status);
  struct ibv_sge scatter_gather_entry;
  if (iree_status_is_ok(status) &&
      iree_net_carrier_state(base_carrier) != IREE_NET_CARRIER_STATE_ACTIVE) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "carrier is not active");
  }
  if (iree_status_is_ok(status)) {
    iree_async_span_t staged_span = iree_async_span_make(
        reservation.buffer_lease.span.region,
        reservation.buffer_lease.span.offset, reservation.byte_length);
    status = iree_net_rdma_sge_from_span(staged_span, &scatter_gather_entry);
  }

  if (iree_status_is_ok(status)) {
    struct ibv_send_wr work_request;
    memset(&work_request, 0, sizeof(work_request));
    work_request.sg_list = &scatter_gather_entry;
    work_request.num_sge = 1;
    work_request.opcode = IBV_WR_SEND;
    work_request.send_flags = IBV_SEND_SIGNALED;

    status = iree_net_rdma_carrier_post_send_work_request_locked(
        carrier, IREE_NET_RDMA_WORK_REQUEST_OPERATION_COMMITTED_SEND,
        IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT,
        /*user_data=*/0, reservation.byte_length,
        /*pending_operation_delta=*/0, &reservation.buffer_lease,
        &work_request);
  }
  iree_slim_mutex_unlock(&carrier->queue_mutex);
  if (!iree_status_is_ok(status)) {
    iree_async_buffer_lease_release(&reservation.buffer_lease);
    if (reservation_resolved) {
      iree_net_carrier_deactivate_callback_fn_t deactivate_callback = NULL;
      void* deactivate_user_data = NULL;
      iree_slim_mutex_lock(&carrier->queue_mutex);
      iree_net_rdma_carrier_drop_pending_operations_locked(
          carrier, 1, &deactivate_callback, &deactivate_user_data);
      iree_slim_mutex_unlock(&carrier->queue_mutex);
      iree_net_rdma_carrier_invoke_deactivate_callback(deactivate_callback,
                                                       deactivate_user_data);
    }
  }
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
  (void)base_carrier;
  return iree_ok_status();
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
  if (iree_any_bit_set(params->flags,
                       IREE_NET_DIRECT_WRITE_FLAG_SIGNAL_RECEIVER)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "RDMA direct_write receiver signaling is not implemented yet");
  }
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
    struct ibv_send_wr work_request;
    memset(&work_request, 0, sizeof(work_request));
    work_request.sg_list = &scatter_gather_entry;
    work_request.num_sge = 1;
    work_request.opcode = IBV_WR_RDMA_WRITE;
    work_request.send_flags = IBV_SEND_SIGNALED;
    work_request.wr.rdma.remote_addr = params->remote.opaque[1];
    work_request.wr.rdma.rkey = (uint32_t)params->remote.opaque[0];

    status = iree_net_rdma_carrier_post_send_work_request(
        carrier, IREE_NET_RDMA_WORK_REQUEST_OPERATION_DIRECT_WRITE,
        IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_NONE, params->user_data,
        params->local.length, /*pending_operation_delta=*/1,
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
  if (!params) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "params must not be NULL");
  }
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

  out_handle->opaque[0] = region->handles.rdma.rkey;
  out_handle->opaque[1] = (uint64_t)(uintptr_t)region->base_ptr;
  return iree_ok_status();
}

static void iree_net_rdma_carrier_unregister_buffer(
    iree_net_carrier_t* base_carrier, iree_net_remote_handle_t handle) {
  (void)base_carrier;
  (void)handle;
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
    status = iree_async_buffer_pool_allocate(
        region, carrier->base.host_allocator, &carrier->send_staging_pool);
  }
  iree_async_region_release(region);
  iree_async_slab_release(slab);
  return status;
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
        &iree_net_rdma_carrier_vtable, iree_net_rdma_carrier_capabilities(),
        /*mtu=*/0, params.options.max_send_sge, params.callback, host_allocator,
        &carrier->base);
    carrier->context = params.context;
    iree_net_rdma_context_retain(params.context);
    carrier->proactor = params.proactor;
    iree_async_proactor_retain(params.proactor);
    carrier->recv_pool = params.recv_pool;
    iree_slim_mutex_initialize(&carrier->queue_mutex);
    carrier->connection_id = params.connection_id;
    carrier->options = params.options;
  }

  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_create_send_staging_pool(carrier);
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
        &carrier->queue_pair, &carrier->work_request_table, params.recv_pool,
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

IREE_API_EXPORT iree_status_t
iree_net_rdma_carrier_serialize_local_connection_data(
    iree_net_rdma_carrier_t* carrier, iree_byte_span_t target,
    iree_host_size_t* out_length) {
  if (!carrier) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "carrier must not be NULL");
  }
  return iree_net_rdma_connection_data_serialize(
      &carrier->local_connection_data, target, out_length);
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
    iree_slim_mutex_lock(&carrier->queue_mutex);
    carrier->remote_connection_data = connection_data;
    iree_net_rdma_send_window_refresh_remote_credits(
        &carrier->send_window, connection_data.initial_recv_credits);
    iree_net_rdma_credit_memory_store(carrier->credit_memory,
                                      connection_data.initial_recv_credits);
    carrier->flags |= IREE_NET_RDMA_CARRIER_FLAG_REMOTE_CONNECTION_DATA_APPLIED;
    status = iree_net_rdma_carrier_try_post_credit_grant_locked(carrier);
    iree_slim_mutex_unlock(&carrier->queue_mutex);
  }
  return status;
}
