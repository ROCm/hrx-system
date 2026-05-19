// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/carrier.h"

#include <errno.h>
#include <limits.h>
#include <string.h>

#include "iree/net/carrier/rdma/completion_queue.h"
#include "iree/net/carrier/rdma/credit_memory.h"
#include "iree/net/carrier/rdma/queue_pair.h"
#include "iree/net/carrier/rdma/receive_queue.h"
#include "iree/net/carrier/rdma/send_window.h"
#include "iree/net/carrier/rdma/sge.h"
#include "iree/net/carrier/rdma/work_request_table.h"

typedef uint8_t iree_net_rdma_carrier_flags_t;
enum iree_net_rdma_carrier_flag_bits_e {
  IREE_NET_RDMA_CARRIER_FLAG_OWNS_CONNECTION_ID = 1u << 0,
  IREE_NET_RDMA_CARRIER_FLAG_REMOTE_CONNECTION_DATA_APPLIED = 1u << 1,
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
         IREE_NET_CARRIER_CAPABILITY_REGISTERED_REGIONS;
}

static void iree_net_rdma_carrier_notify_error(iree_net_rdma_carrier_t* carrier,
                                               iree_status_t status) {
  carrier->base.callback.fn(carrier->base.callback.user_data,
                            /*operation_user_data=*/0, status,
                            /*bytes_transferred=*/0, /*recv_lease=*/NULL);
}

static iree_status_t iree_net_rdma_carrier_status_from_work_completion(
    const struct ibv_wc* completion, const char* queue_name) {
  if (completion->status == IBV_WC_SUCCESS) return iree_ok_status();
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE, "RDMA %s completion failed: status=%u vendor=%u",
      queue_name, (uint32_t)completion->status, completion->vendor_err);
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
    iree_status_t completion_status =
        iree_net_rdma_carrier_status_from_work_completion(completion, "send");
    if (iree_status_is_ok(completion_status)) {
      completion_status = iree_net_rdma_work_request_table_complete(
          &carrier->work_request_table, completion->wr_id,
          &work_request_completion);
    }
    if (iree_status_is_ok(completion_status)) {
      completion_status =
          iree_net_rdma_send_window_complete(&carrier->send_window);
    }
    carrier->base.callback.fn(
        carrier->base.callback.user_data, work_request_completion.user_data,
        completion_status, work_request_completion.byte_length,
        /*recv_lease=*/NULL);
  }
}

static iree_status_t iree_net_rdma_carrier_deliver_receive(
    iree_net_rdma_carrier_t* carrier,
    iree_net_rdma_work_request_completion_t completion,
    const struct ibv_wc* work_completion) {
  iree_async_buffer_lease_t lease;
  iree_status_t status = iree_net_rdma_receive_queue_complete(
      &carrier->receive_queue, completion,
      (iree_host_size_t)work_completion->byte_len, &lease);

  if (iree_status_is_ok(status)) {
    iree_net_carrier_state_t state = iree_net_carrier_state(&carrier->base);
    if (state != IREE_NET_CARRIER_STATE_ACTIVE ||
        !carrier->base.recv_handler.fn) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "RDMA receive arrived before activation");
    }
  }
  if (iree_status_is_ok(status)) {
    status = carrier->base.recv_handler.fn(carrier->base.recv_handler.user_data,
                                           lease.span, &lease);
  }
  iree_async_buffer_lease_release(&lease);

  if (iree_status_is_ok(status)) {
    uint32_t posted_count = 0;
    status = iree_net_rdma_receive_queue_replenish(
        &carrier->receive_queue, carrier->options.recv_queue_depth,
        &posted_count);
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
    iree_status_t completion_status =
        iree_net_rdma_carrier_status_from_work_completion(completion, "recv");
    if (iree_status_is_ok(completion_status)) {
      completion_status = iree_net_rdma_work_request_table_complete(
          &carrier->work_request_table, completion->wr_id,
          &work_request_completion);
    }
    if (iree_status_is_ok(completion_status) &&
        work_request_completion.operation !=
            IREE_NET_RDMA_WORK_REQUEST_OPERATION_RECV) {
      completion_status =
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "RDMA receive CQ returned operation %u",
                           work_request_completion.operation);
    }
    if (iree_status_is_ok(completion_status)) {
      completion_status = iree_net_rdma_carrier_deliver_receive(
          carrier, work_request_completion, completion);
    }
    if (!iree_status_is_ok(completion_status)) {
      iree_net_rdma_carrier_notify_error(carrier, completion_status);
    }
  }
}

static void iree_net_rdma_carrier_destroy(iree_net_carrier_t* base_carrier) {
  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_from_base(base_carrier);

  iree_net_rdma_queue_pair_deinitialize(&carrier->queue_pair);
  iree_net_rdma_receive_queue_deinitialize(&carrier->receive_queue);
  iree_net_rdma_work_request_table_deinitialize(&carrier->work_request_table);
  iree_net_rdma_credit_memory_release(carrier->credit_memory);
  iree_net_rdma_completion_queue_release(carrier->recv_completion_queue);
  iree_net_rdma_completion_queue_release(carrier->send_completion_queue);
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
  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state == IREE_NET_CARRIER_STATE_DEACTIVATED) {
    if (callback) callback(user_data);
    return iree_ok_status();
  }
  iree_net_carrier_set_state(base_carrier, IREE_NET_CARRIER_STATE_DRAINING);
  iree_net_carrier_set_state(base_carrier, IREE_NET_CARRIER_STATE_DEACTIVATED);
  if (callback) callback(user_data);
  return iree_ok_status();
}

static iree_net_carrier_send_budget_t iree_net_rdma_carrier_query_send_budget(
    iree_net_carrier_t* base_carrier) {
  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_from_base(base_carrier);
  iree_net_carrier_send_budget_t budget;
  budget.bytes = IREE_HOST_SIZE_MAX;
  budget.slots = iree_net_rdma_send_window_available(
      &carrier->send_window,
      IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT);
  return budget;
}

static iree_status_t iree_net_rdma_carrier_unimplemented(
    const char* operation) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "RDMA carrier %s is not implemented yet", operation);
}

static iree_status_t iree_net_rdma_carrier_total_span_length(
    iree_async_span_list_t spans, iree_host_size_t* out_total_length) {
  *out_total_length = 0;
  iree_status_t status = iree_ok_status();
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

  struct ibv_sge scatter_gather_entries[IREE_NET_RDMA_CARRIER_MAX_SEND_SGE];
  int scatter_gather_entry_count = 0;
  iree_status_t status = iree_net_rdma_sge_list_from_span_list(
      params->data, IREE_ARRAYSIZE(scatter_gather_entries),
      scatter_gather_entries, &scatter_gather_entry_count);

  iree_host_size_t total_length = 0;
  if (iree_status_is_ok(status)) {
    status =
        iree_net_rdma_carrier_total_span_length(params->data, &total_length);
  }

  iree_net_rdma_send_window_acquire_flags_t acquire_flags =
      IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT;
  bool send_window_acquired = false;
  if (iree_status_is_ok(status)) {
    status =
        iree_net_rdma_send_window_acquire(&carrier->send_window, acquire_flags);
    send_window_acquired = iree_status_is_ok(status);
  }

  uint64_t work_request_id = 0;
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_work_request_table_acquire(
        &carrier->work_request_table, IREE_NET_RDMA_WORK_REQUEST_OPERATION_SEND,
        params->user_data, total_length, &work_request_id);
  }

  if (iree_status_is_ok(status)) {
    struct ibv_send_wr work_request;
    memset(&work_request, 0, sizeof(work_request));
    work_request.wr_id = work_request_id;
    work_request.sg_list = scatter_gather_entries;
    work_request.num_sge = scatter_gather_entry_count;
    work_request.opcode = IBV_WR_SEND;
    work_request.send_flags = IBV_SEND_SIGNALED;

    struct ibv_send_wr* bad_work_request = NULL;
    errno = 0;
    int result =
        ibv_post_send(iree_net_rdma_queue_pair_native_qp(&carrier->queue_pair),
                      &work_request, &bad_work_request);
    if (result != 0) {
      int error = result > 0 ? result : errno;
      status = iree_net_rdma_carrier_status_from_errno_required(
          __FILE__, __LINE__, error, "ibv_post_send");
    }
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
      status =
          iree_status_join(status, iree_net_rdma_send_window_abort(
                                       &carrier->send_window, acquire_flags));
    }
  }
  return status;
}

static iree_status_t iree_net_rdma_carrier_begin_send(
    iree_net_carrier_t* base_carrier, iree_host_size_t size, void** out_ptr,
    iree_net_carrier_send_handle_t* out_handle) {
  (void)base_carrier;
  (void)size;
  if (out_ptr) *out_ptr = NULL;
  if (out_handle) *out_handle = 0;
  return iree_net_rdma_carrier_unimplemented("begin_send");
}

static iree_status_t iree_net_rdma_carrier_commit_send(
    iree_net_carrier_t* base_carrier, iree_net_carrier_send_handle_t handle) {
  (void)base_carrier;
  (void)handle;
  return iree_net_rdma_carrier_unimplemented("commit_send");
}

static void iree_net_rdma_carrier_abort_send(
    iree_net_carrier_t* base_carrier, iree_net_carrier_send_handle_t handle) {
  (void)base_carrier;
  (void)handle;
}

static iree_status_t iree_net_rdma_carrier_shutdown(
    iree_net_carrier_t* base_carrier) {
  (void)base_carrier;
  return iree_ok_status();
}

static iree_status_t iree_net_rdma_carrier_direct_write(
    iree_net_carrier_t* base_carrier,
    const iree_net_direct_write_params_t* params) {
  (void)base_carrier;
  (void)params;
  return iree_net_rdma_carrier_unimplemented("direct_write");
}

static iree_status_t iree_net_rdma_carrier_direct_read(
    iree_net_carrier_t* base_carrier,
    const iree_net_direct_read_params_t* params) {
  (void)base_carrier;
  (void)params;
  return iree_net_rdma_carrier_unimplemented("direct_read");
}

static iree_status_t iree_net_rdma_carrier_register_buffer(
    iree_net_carrier_t* base_carrier, iree_async_region_t* region,
    iree_net_remote_handle_t* out_handle) {
  (void)base_carrier;
  (void)region;
  if (out_handle) *out_handle = iree_net_remote_handle_null();
  return iree_net_rdma_carrier_unimplemented("register_buffer");
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
    carrier->connection_id = params.connection_id;
    carrier->options = params.options;
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
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_send_window_initialize(
        params.options.send_queue_depth,
        /*initial_remote_recv_credits=*/0, &carrier->send_window);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_initialize_local_connection_data(
        carrier, posted_recv_count);
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
    carrier->remote_connection_data = connection_data;
    iree_net_rdma_send_window_refresh_remote_credits(
        &carrier->send_window, connection_data.initial_recv_credits);
    iree_net_rdma_credit_memory_store(carrier->credit_memory,
                                      connection_data.initial_recv_credits);
    carrier->flags |= IREE_NET_RDMA_CARRIER_FLAG_REMOTE_CONNECTION_DATA_APPLIED;
  }
  return status;
}
