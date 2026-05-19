// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/factory.h"

#include <errno.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/socket.h>

#include "iree/async/address.h"
#include "iree/async/operations/scheduling.h"
#include "iree/net/carrier/rdma/cm_channel.h"
#include "iree/net/channel/util/frame_sender.h"
#include "iree/net/connection.h"
#include "iree/net/message_endpoint.h"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static int iree_net_rdma_factory_error_from_result(int result) {
  if (result >= 0) return result;
  if (errno != 0) return errno;
  return result == -1 ? EIO : -result;
}

static iree_status_t iree_net_rdma_factory_status_from_result(
    const char* file, uint32_t line, int result, const char* call) {
  if (result == 0) return iree_ok_status();
  return iree_status_from_errno(
      file, line, iree_net_rdma_factory_error_from_result(result), call);
}

static iree_status_t iree_net_rdma_factory_status_from_cm_event(
    const iree_net_rdma_cm_event_t* event, const char* operation) {
  if (event->status == 0) return iree_ok_status();
  return iree_status_from_errno(__FILE__, __LINE__, event->status, operation);
}

static iree_status_t iree_net_rdma_factory_status_from_failed_cm_event(
    const iree_net_rdma_cm_event_t* event, const char* operation) {
  if (event->status != 0) {
    return iree_status_from_errno(__FILE__, __LINE__, event->status, operation);
  }
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "%s failed with RDMA CM event %d", operation,
                          (int)event->type);
}

static iree_status_t iree_net_rdma_factory_copy_sockaddr(
    const struct sockaddr* source, iree_async_address_t* out_address) {
  if (!source) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sockaddr must not be NULL");
  }

  socklen_t length = 0;
  switch (source->sa_family) {
    case AF_INET:
      length = sizeof(struct sockaddr_in);
      break;
    case AF_INET6:
      length = sizeof(struct sockaddr_in6);
      break;
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported RDMA address family %d",
                              (int)source->sa_family);
  }

  memset(out_address, 0, sizeof(*out_address));
  memcpy(out_address->storage, source, length);
  out_address->length = length;
  return iree_ok_status();
}

static iree_status_t iree_net_rdma_factory_initialize_conn_param(
    iree_const_byte_span_t private_data, struct rdma_conn_param* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  if (private_data.data_length > UINT8_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "RDMA private data length %" PRIhsz
                            " exceeds uint8_t capacity",
                            private_data.data_length);
  }
  out_params->private_data = private_data.data;
  out_params->private_data_len = (uint8_t)private_data.data_length;
  out_params->responder_resources = 1;
  out_params->initiator_depth = 1;
  out_params->retry_count = 7;
  out_params->rnr_retry_count = 7;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Factory type
//===----------------------------------------------------------------------===//

typedef struct iree_net_rdma_factory_t {
  // Base transport factory; must be first for vtable dispatch.
  iree_net_transport_factory_t base;

  // Parent context holding rdma-core libraries and expected device selection.
  iree_net_rdma_context_t* context;

  // Carrier options copied into each connection's carrier creation params.
  iree_net_rdma_carrier_options_t carrier_options;

  // Timeout in milliseconds for rdma_cm address and route resolution.
  int resolve_timeout_ms;

  // Backlog used by listeners.
  int listen_backlog;

  // Maximum endpoint count reported by created connections.
  uint32_t max_endpoint_count;

  // Host allocator used for factory-owned allocations.
  iree_allocator_t host_allocator;
} iree_net_rdma_factory_t;

static const iree_net_transport_factory_vtable_t iree_net_rdma_factory_vtable;

//===----------------------------------------------------------------------===//
// Endpoint
//===----------------------------------------------------------------------===//

typedef struct iree_net_rdma_connection_t iree_net_rdma_connection_t;

typedef uint8_t iree_net_rdma_endpoint_flags_t;
enum iree_net_rdma_endpoint_flag_bits_e {
  IREE_NET_RDMA_ENDPOINT_FLAG_ALLOCATED = 1u << 0,
  IREE_NET_RDMA_ENDPOINT_FLAG_ACTIVE = 1u << 1,
};

typedef struct iree_net_rdma_endpoint_t {
  // Back-pointer to the owning connection.
  iree_net_rdma_connection_t* connection;

  // Borrowed carrier owned by the connection.
  iree_net_carrier_t* carrier;

  // Message and error callbacks installed by the endpoint consumer.
  iree_net_message_endpoint_callbacks_t callbacks;

  // Deactivation callback stored while the carrier drains.
  iree_net_message_endpoint_deactivate_fn_t deactivate_callback;

  // User data passed to deactivate_callback.
  void* deactivate_user_data;

  // Bitfield of iree_net_rdma_endpoint_flag_bits_e values.
  iree_net_rdma_endpoint_flags_t flags;
} iree_net_rdma_endpoint_t;

static const iree_net_message_endpoint_vtable_t iree_net_rdma_endpoint_vtable;

static iree_status_t iree_net_rdma_endpoint_on_recv(
    void* user_data, iree_async_span_t data, iree_async_buffer_lease_t* lease) {
  iree_net_rdma_endpoint_t* endpoint = (iree_net_rdma_endpoint_t*)user_data;
  if (!endpoint->callbacks.on_message) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "RDMA endpoint has no message handler");
  }
  iree_const_byte_span_t message =
      iree_make_const_byte_span(iree_async_span_ptr(data), data.length);
  return endpoint->callbacks.on_message(endpoint->callbacks.user_data, message,
                                        lease);
}

static void iree_net_rdma_endpoint_carrier_deactivated(void* user_data) {
  iree_net_rdma_endpoint_t* endpoint = (iree_net_rdma_endpoint_t*)user_data;
  endpoint->flags &= ~IREE_NET_RDMA_ENDPOINT_FLAG_ACTIVE;
  if (endpoint->deactivate_callback) {
    endpoint->deactivate_callback(endpoint->deactivate_user_data);
  }
}

static void iree_net_rdma_endpoint_set_callbacks(
    void* self, iree_net_message_endpoint_callbacks_t callbacks) {
  iree_net_rdma_endpoint_t* endpoint = (iree_net_rdma_endpoint_t*)self;
  endpoint->callbacks = callbacks;
}

static iree_status_t iree_net_rdma_endpoint_activate(void* self) {
  iree_net_rdma_endpoint_t* endpoint = (iree_net_rdma_endpoint_t*)self;
  if (!endpoint->callbacks.on_message) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "callbacks must be set before activation");
  }

  iree_net_carrier_t* carrier = endpoint->carrier;
  iree_net_carrier_set_recv_handler(carrier,
                                    (iree_net_carrier_recv_handler_t){
                                        .fn = iree_net_rdma_endpoint_on_recv,
                                        .user_data = endpoint,
                                    });
  iree_status_t status = iree_net_carrier_activate(carrier);
  if (iree_status_is_ok(status)) {
    endpoint->flags |= IREE_NET_RDMA_ENDPOINT_FLAG_ACTIVE;
  }
  return status;
}

static iree_status_t iree_net_rdma_endpoint_deactivate(
    void* self, iree_net_message_endpoint_deactivate_fn_t callback,
    void* user_data) {
  iree_net_rdma_endpoint_t* endpoint = (iree_net_rdma_endpoint_t*)self;
  if (!iree_any_bit_set(endpoint->flags, IREE_NET_RDMA_ENDPOINT_FLAG_ACTIVE)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "endpoint not active");
  }
  endpoint->deactivate_callback = callback;
  endpoint->deactivate_user_data = user_data;
  return iree_net_carrier_deactivate(
      endpoint->carrier, iree_net_rdma_endpoint_carrier_deactivated, endpoint);
}

static iree_status_t iree_net_rdma_endpoint_send(
    void* self, const iree_net_message_endpoint_send_params_t* params) {
  iree_net_rdma_endpoint_t* endpoint = (iree_net_rdma_endpoint_t*)self;
  iree_net_send_params_t carrier_params = {
      .data = params->data,
      .flags = IREE_NET_SEND_FLAG_NONE,
      .user_data = params->user_data,
  };
  return iree_net_carrier_send(endpoint->carrier, &carrier_params);
}

static iree_net_carrier_send_budget_t iree_net_rdma_endpoint_query_send_budget(
    void* self) {
  iree_net_rdma_endpoint_t* endpoint = (iree_net_rdma_endpoint_t*)self;
  return iree_net_carrier_query_send_budget(endpoint->carrier);
}

static iree_status_t iree_net_rdma_endpoint_begin_send(
    void* self, iree_host_size_t size, void** out_ptr,
    iree_net_carrier_send_handle_t* out_handle) {
  iree_net_rdma_endpoint_t* endpoint = (iree_net_rdma_endpoint_t*)self;
  return iree_net_carrier_begin_send(endpoint->carrier, size, out_ptr,
                                     out_handle);
}

static iree_status_t iree_net_rdma_endpoint_commit_send(
    void* self, iree_net_carrier_send_handle_t handle) {
  iree_net_rdma_endpoint_t* endpoint = (iree_net_rdma_endpoint_t*)self;
  return iree_net_carrier_commit_send(endpoint->carrier, handle);
}

static void iree_net_rdma_endpoint_abort_send(
    void* self, iree_net_carrier_send_handle_t handle) {
  iree_net_rdma_endpoint_t* endpoint = (iree_net_rdma_endpoint_t*)self;
  iree_net_carrier_abort_send(endpoint->carrier, handle);
}

static const iree_net_message_endpoint_vtable_t iree_net_rdma_endpoint_vtable =
    {
        .set_callbacks = iree_net_rdma_endpoint_set_callbacks,
        .activate = iree_net_rdma_endpoint_activate,
        .deactivate = iree_net_rdma_endpoint_deactivate,
        .send = iree_net_rdma_endpoint_send,
        .query_send_budget = iree_net_rdma_endpoint_query_send_budget,
        .begin_send = iree_net_rdma_endpoint_begin_send,
        .commit_send = iree_net_rdma_endpoint_commit_send,
        .abort_send = iree_net_rdma_endpoint_abort_send,
};

//===----------------------------------------------------------------------===//
// Connection
//===----------------------------------------------------------------------===//

typedef struct iree_net_rdma_connection_t {
  // Base connection; must be first for vtable dispatch.
  iree_net_connection_t base;

  // Proactor used to deliver endpoint-ready NOPs. Retained by the connection.
  iree_async_proactor_t* proactor;

  // Receive pool borrowed by the carrier. Stored for diagnostics/future QPs.
  iree_async_buffer_pool_t* recv_pool;

  // Owned single-QP carrier.
  iree_net_carrier_t* carrier;

  // Embedded single endpoint view over carrier.
  iree_net_rdma_endpoint_t endpoint;

  // Endpoint-ready operation submitted by open_endpoint.
  iree_async_nop_operation_t endpoint_ready_operation;

  // Endpoint-ready callback for endpoint_ready_operation.
  iree_net_endpoint_ready_callback_t endpoint_ready_callback;

  // Connection-level deactivation callback.
  iree_net_connection_deactivate_callback_t deactivate_callback;

  // Number of endpoint slots already handed out.
  uint32_t allocated_endpoint_count;
} iree_net_rdma_connection_t;

static const iree_net_connection_vtable_t iree_net_rdma_connection_vtable;

static void iree_net_rdma_connection_notify_error(
    iree_net_rdma_connection_t* connection, iree_status_t status) {
  if (connection->endpoint.callbacks.on_error &&
      iree_any_bit_set(connection->endpoint.flags,
                       IREE_NET_RDMA_ENDPOINT_FLAG_ALLOCATED)) {
    connection->endpoint.callbacks.on_error(
        connection->endpoint.callbacks.user_data, status);
  } else if (!iree_status_is_ok(status)) {
    iree_status_abort(status);
  }
}

static void iree_net_rdma_connection_carrier_completion(
    void* callback_user_data, iree_net_carrier_completion_kind_t kind,
    uint64_t operation_user_data, iree_status_t status,
    iree_host_size_t bytes_transferred, iree_async_buffer_lease_t* recv_lease) {
  iree_net_rdma_connection_t* connection =
      (iree_net_rdma_connection_t*)callback_user_data;
  if (kind == IREE_NET_CARRIER_COMPLETION_SEND && operation_user_data != 0) {
    iree_status_t endpoint_status = iree_ok_status();
    if (!iree_status_is_ok(status)) endpoint_status = iree_status_clone(status);
    iree_net_frame_sender_dispatch_carrier_completion(
        NULL, kind, operation_user_data, status, bytes_transferred, recv_lease);
    if (!iree_status_is_ok(endpoint_status)) {
      iree_net_rdma_connection_notify_error(connection, endpoint_status);
    }
    return;
  }

  if (!iree_status_is_ok(status)) {
    iree_net_rdma_connection_notify_error(connection, status);
  }
}

static void iree_net_rdma_connection_deactivate_complete(void* user_data) {
  iree_net_rdma_connection_t* connection =
      (iree_net_rdma_connection_t*)user_data;
  connection->endpoint.flags &= ~IREE_NET_RDMA_ENDPOINT_FLAG_ACTIVE;
  connection->deactivate_callback.fn(connection->deactivate_callback.user_data);
  iree_net_connection_release(&connection->base);
}

static void iree_net_rdma_connection_deactivate(
    iree_net_connection_t* base_connection,
    iree_net_connection_deactivate_callback_t callback) {
  iree_net_rdma_connection_t* connection =
      (iree_net_rdma_connection_t*)base_connection;
  if (!iree_any_bit_set(connection->endpoint.flags,
                        IREE_NET_RDMA_ENDPOINT_FLAG_ACTIVE)) {
    callback.fn(callback.user_data);
    return;
  }

  connection->deactivate_callback = callback;
  iree_net_connection_retain(base_connection);
  iree_status_t status = iree_net_carrier_deactivate(
      connection->carrier, iree_net_rdma_connection_deactivate_complete,
      connection);
  if (!iree_status_is_ok(status)) {
    iree_net_connection_release(base_connection);
    iree_status_abort(status);
  }
}

static void iree_net_rdma_connection_destroy(
    iree_net_connection_t* base_connection) {
  iree_net_rdma_connection_t* connection =
      (iree_net_rdma_connection_t*)base_connection;
  IREE_ASSERT(!iree_any_bit_set(connection->endpoint.flags,
                                IREE_NET_RDMA_ENDPOINT_FLAG_ACTIVE),
              "connection destroyed with active endpoint; call "
              "iree_net_connection_deactivate before releasing");
  iree_allocator_t host_allocator = connection->base.host_allocator;
  iree_net_carrier_release(connection->carrier);
  iree_async_proactor_release(connection->proactor);
  iree_allocator_free(host_allocator, connection);
}

static void iree_net_rdma_endpoint_ready_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_rdma_connection_t* connection =
      (iree_net_rdma_connection_t*)user_data;
  if (iree_status_is_ok(status)) {
    iree_net_message_endpoint_t endpoint = {
        .self = &connection->endpoint,
        .vtable = &iree_net_rdma_endpoint_vtable,
    };
    connection->endpoint_ready_callback.fn(
        connection->endpoint_ready_callback.user_data, iree_ok_status(),
        endpoint);
  } else {
    connection->endpoint.flags &= ~IREE_NET_RDMA_ENDPOINT_FLAG_ALLOCATED;
    connection->endpoint_ready_callback.fn(
        connection->endpoint_ready_callback.user_data, status,
        (iree_net_message_endpoint_t){0});
  }
  iree_net_connection_release(&connection->base);
}

static iree_status_t iree_net_rdma_connection_open_endpoint(
    iree_net_connection_t* base_connection,
    iree_net_endpoint_ready_callback_t callback) {
  iree_net_rdma_connection_t* connection =
      (iree_net_rdma_connection_t*)base_connection;
  if (connection->allocated_endpoint_count >= 1) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "RDMA connection has no free endpoint slots");
  }

  connection->allocated_endpoint_count = 1;
  connection->endpoint.flags |= IREE_NET_RDMA_ENDPOINT_FLAG_ALLOCATED;
  connection->endpoint_ready_callback = callback;
  memset(&connection->endpoint_ready_operation, 0,
         sizeof(connection->endpoint_ready_operation));
  iree_async_operation_initialize(
      &connection->endpoint_ready_operation.base, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_rdma_endpoint_ready_complete,
      connection);

  iree_net_connection_retain(base_connection);
  iree_status_t status = iree_async_proactor_submit_one(
      connection->proactor, &connection->endpoint_ready_operation.base);
  if (!iree_status_is_ok(status)) {
    iree_net_connection_release(base_connection);
    connection->allocated_endpoint_count = 0;
    connection->endpoint.flags &= ~IREE_NET_RDMA_ENDPOINT_FLAG_ALLOCATED;
  }
  return status;
}

static iree_net_carrier_t* iree_net_rdma_connection_carrier(
    iree_net_connection_t* base_connection) {
  iree_net_rdma_connection_t* connection =
      (iree_net_rdma_connection_t*)base_connection;
  return connection->carrier;
}

static const iree_net_connection_vtable_t iree_net_rdma_connection_vtable = {
    .destroy = iree_net_rdma_connection_destroy,
    .deactivate = iree_net_rdma_connection_deactivate,
    .open_endpoint = iree_net_rdma_connection_open_endpoint,
    .carrier = iree_net_rdma_connection_carrier,
};

static iree_status_t iree_net_rdma_connection_create(
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_carrier_t* carrier, iree_allocator_t host_allocator,
    iree_net_connection_t** out_connection) {
  *out_connection = NULL;

  iree_net_rdma_connection_t* connection = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*connection), (void**)&connection);
  if (iree_status_is_ok(status)) {
    memset(connection, 0, sizeof(*connection));
    iree_net_connection_initialize(&iree_net_rdma_connection_vtable,
                                   host_allocator, /*max_endpoint_count=*/1,
                                   &connection->base);
    connection->proactor = proactor;
    iree_async_proactor_retain(proactor);
    connection->recv_pool = recv_pool;
    connection->carrier = carrier;
    carrier->callback.user_data = connection;
    connection->endpoint.connection = connection;
    connection->endpoint.carrier = carrier;
    *out_connection = &connection->base;
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Connect state machine
//===----------------------------------------------------------------------===//

typedef enum iree_net_rdma_connect_phase_e {
  IREE_NET_RDMA_CONNECT_PHASE_RESOLVING_ADDRESS = 0,
  IREE_NET_RDMA_CONNECT_PHASE_RESOLVING_ROUTE = 1,
  IREE_NET_RDMA_CONNECT_PHASE_CONNECTING = 2,
  IREE_NET_RDMA_CONNECT_PHASE_COMPLETE = 3,
} iree_net_rdma_connect_phase_t;

typedef struct iree_net_rdma_connect_state_t {
  // Cleanup operation used to release cm_channel outside its callback.
  iree_async_nop_operation_t cleanup_operation;

  // Factory retained while the connect is in flight.
  iree_net_rdma_factory_t* factory;

  // Proactor used for CM events and cleanup NOPs.
  iree_async_proactor_t* proactor;

  // Receive pool borrowed by the resulting carrier.
  iree_async_buffer_pool_t* recv_pool;

  // rdma_cm event channel wrapper.
  iree_net_rdma_cm_channel_t* cm_channel;

  // Raw connection ID before carrier creation transfers ownership.
  struct rdma_cm_id* connection_id;

  // Carrier created after address resolution.
  iree_net_carrier_t* carrier;

  // Child context matching connection_id->verbs.
  iree_net_rdma_context_t* connection_context;

  // User connect callback.
  iree_net_transport_connect_callback_t callback;

  // User data passed to callback.
  void* callback_user_data;

  // Serialized local RDMA private data sent by rdma_connect.
  uint8_t private_data[IREE_NET_RDMA_CONNECTION_DATA_LENGTH];

  // Number of bytes valid in private_data.
  iree_host_size_t private_data_length;

  // Current CM state-machine phase.
  iree_net_rdma_connect_phase_t phase;

  // True after remote private data has been applied.
  bool remote_data_applied;

  // Host allocator used for this state allocation.
  iree_allocator_t host_allocator;
} iree_net_rdma_connect_state_t;

static void iree_net_rdma_connect_state_cleanup(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_rdma_connect_state_t* state =
      (iree_net_rdma_connect_state_t*)user_data;
  if (!iree_status_is_ok(status)) iree_status_abort(status);

  iree_net_carrier_release(state->carrier);
  iree_net_rdma_context_release(state->connection_context);
  if (state->connection_id) {
    const iree_net_librdmacm_t* librdmacm =
        iree_net_rdma_context_librdmacm(state->factory->context);
    int result = librdmacm->rdma_destroy_id(state->connection_id);
    iree_status_t destroy_status = iree_net_rdma_factory_status_from_result(
        __FILE__, __LINE__, result, "rdma_destroy_id");
    if (!iree_status_is_ok(destroy_status)) iree_status_abort(destroy_status);
  }
  iree_net_rdma_cm_channel_release(state->cm_channel);
  iree_async_proactor_release(state->proactor);
  iree_net_transport_factory_release(&state->factory->base);
  iree_allocator_t host_allocator = state->host_allocator;
  iree_allocator_free(host_allocator, state);
}

static void iree_net_rdma_connect_state_submit_cleanup(
    iree_net_rdma_connect_state_t* state) {
  memset(&state->cleanup_operation, 0, sizeof(state->cleanup_operation));
  iree_async_operation_initialize(&state->cleanup_operation.base,
                                  IREE_ASYNC_OPERATION_TYPE_NOP,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  iree_net_rdma_connect_state_cleanup, state);
  iree_status_t status = iree_async_proactor_submit_one(
      state->proactor, &state->cleanup_operation.base);
  if (!iree_status_is_ok(status)) iree_status_abort(status);
}

static void iree_net_rdma_connect_state_fail(
    iree_net_rdma_connect_state_t* state, iree_status_t status) {
  state->phase = IREE_NET_RDMA_CONNECT_PHASE_COMPLETE;
  state->callback(state->callback_user_data, status, NULL);
  iree_net_rdma_connect_state_submit_cleanup(state);
}

static void iree_net_rdma_connect_state_succeed(
    iree_net_rdma_connect_state_t* state) {
  iree_net_connection_t* connection = NULL;
  iree_status_t status = iree_net_rdma_connection_create(
      state->proactor, state->recv_pool, state->carrier, state->host_allocator,
      &connection);
  if (iree_status_is_ok(status)) {
    state->carrier = NULL;
    state->phase = IREE_NET_RDMA_CONNECT_PHASE_COMPLETE;
    state->callback(state->callback_user_data, iree_ok_status(), connection);
  } else {
    state->phase = IREE_NET_RDMA_CONNECT_PHASE_COMPLETE;
    state->callback(state->callback_user_data, status, NULL);
  }
  iree_net_rdma_connect_state_submit_cleanup(state);
}

static iree_status_t iree_net_rdma_connect_state_create_carrier(
    iree_net_rdma_connect_state_t* state) {
  iree_status_t status = iree_net_rdma_context_create_for_cm_id(
      state->factory->context, state->connection_id, state->host_allocator,
      &state->connection_context);

  if (iree_status_is_ok(status)) {
    iree_net_carrier_callback_t callback = {
        .fn = iree_net_rdma_connection_carrier_completion,
        .user_data = NULL,
    };
    iree_net_rdma_carrier_create_params_t params = {
        .context = state->connection_context,
        .proactor = state->proactor,
        .recv_pool = state->recv_pool,
        .connection_id = state->connection_id,
        .callback = callback,
        .options = state->factory->carrier_options,
    };
    status = iree_net_rdma_carrier_create(params, state->host_allocator,
                                          &state->carrier);
  }

  if (iree_status_is_ok(status)) {
    state->connection_id = NULL;
    iree_net_rdma_carrier_t* rdma_carrier =
        iree_net_rdma_carrier_cast(state->carrier);
    status = iree_net_rdma_carrier_serialize_local_connection_data(
        rdma_carrier,
        iree_make_byte_span(state->private_data, sizeof(state->private_data)),
        &state->private_data_length);
  }
  return status;
}

static iree_status_t iree_net_rdma_connect_state_apply_remote_data(
    iree_net_rdma_connect_state_t* state,
    const iree_net_rdma_cm_event_t* event) {
  if (state->remote_data_applied) return iree_ok_status();
  if (event->private_data.data_length == 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "RDMA peer did not provide private data");
  }
  iree_net_rdma_carrier_t* carrier = iree_net_rdma_carrier_cast(state->carrier);
  iree_status_t status = iree_net_rdma_carrier_apply_remote_connection_data(
      carrier, event->private_data);
  if (iree_status_is_ok(status)) state->remote_data_applied = true;
  return status;
}

static void iree_net_rdma_connect_state_on_cm_event(
    void* user_data, iree_status_t status,
    const iree_net_rdma_cm_event_t* event) {
  iree_net_rdma_connect_state_t* state =
      (iree_net_rdma_connect_state_t*)user_data;
  if (state->phase == IREE_NET_RDMA_CONNECT_PHASE_COMPLETE) {
    if (!iree_status_is_ok(status)) iree_status_abort(status);
    return;
  }
  if (!iree_status_is_ok(status)) {
    iree_net_rdma_connect_state_fail(state, status);
    return;
  }

  const iree_net_librdmacm_t* librdmacm =
      iree_net_rdma_context_librdmacm(state->factory->context);
  switch (event->type) {
    case RDMA_CM_EVENT_ADDR_RESOLVED: {
      status = iree_net_rdma_factory_status_from_cm_event(event,
                                                          "rdma_resolve_addr");
      if (iree_status_is_ok(status)) {
        status = iree_net_rdma_connect_state_create_carrier(state);
      }
      if (iree_status_is_ok(status)) {
        state->phase = IREE_NET_RDMA_CONNECT_PHASE_RESOLVING_ROUTE;
        int result = librdmacm->rdma_resolve_route(
            iree_net_rdma_carrier_connection_id(
                iree_net_rdma_carrier_cast(state->carrier)),
            state->factory->resolve_timeout_ms);
        status = iree_net_rdma_factory_status_from_result(
            __FILE__, __LINE__, result, "rdma_resolve_route");
      }
      if (!iree_status_is_ok(status)) {
        iree_net_rdma_connect_state_fail(state, status);
      }
      break;
    }
    case RDMA_CM_EVENT_ROUTE_RESOLVED: {
      status = iree_net_rdma_factory_status_from_cm_event(event,
                                                          "rdma_resolve_route");
      struct rdma_conn_param conn_param;
      if (iree_status_is_ok(status)) {
        status = iree_net_rdma_factory_initialize_conn_param(
            iree_make_const_byte_span(state->private_data,
                                      state->private_data_length),
            &conn_param);
      }
      if (iree_status_is_ok(status)) {
        state->phase = IREE_NET_RDMA_CONNECT_PHASE_CONNECTING;
        int result = librdmacm->rdma_connect(
            iree_net_rdma_carrier_connection_id(
                iree_net_rdma_carrier_cast(state->carrier)),
            &conn_param);
        status = iree_net_rdma_factory_status_from_result(
            __FILE__, __LINE__, result, "rdma_connect");
      }
      if (!iree_status_is_ok(status)) {
        iree_net_rdma_connect_state_fail(state, status);
      }
      break;
    }
    case RDMA_CM_EVENT_CONNECT_RESPONSE:
      status = iree_net_rdma_connect_state_apply_remote_data(state, event);
      if (!iree_status_is_ok(status)) {
        iree_net_rdma_connect_state_fail(state, status);
      }
      break;
    case RDMA_CM_EVENT_ESTABLISHED:
      status =
          iree_net_rdma_factory_status_from_cm_event(event, "rdma_connect");
      if (iree_status_is_ok(status)) {
        status = iree_net_rdma_connect_state_apply_remote_data(state, event);
      }
      if (iree_status_is_ok(status)) {
        iree_net_rdma_connect_state_succeed(state);
      } else {
        iree_net_rdma_connect_state_fail(state, status);
      }
      break;
    case RDMA_CM_EVENT_REJECTED:
      iree_net_rdma_connect_state_fail(
          state, iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                                  "RDMA connection rejected by peer"));
      break;
    case RDMA_CM_EVENT_ADDR_ERROR:
    case RDMA_CM_EVENT_ROUTE_ERROR:
    case RDMA_CM_EVENT_CONNECT_ERROR:
    case RDMA_CM_EVENT_UNREACHABLE:
      iree_net_rdma_connect_state_fail(
          state, iree_net_rdma_factory_status_from_failed_cm_event(
                     event, "rdma_cm connect"));
      break;
    case RDMA_CM_EVENT_DISCONNECTED:
    case RDMA_CM_EVENT_DEVICE_REMOVAL:
      iree_net_rdma_connect_state_fail(
          state,
          iree_make_status(IREE_STATUS_UNAVAILABLE,
                           "RDMA connection closed before establishment"));
      break;
    default:
      iree_net_rdma_connect_state_fail(
          state, iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "unexpected RDMA CM event %d during connect",
                                  (int)event->type));
      break;
  }
}

//===----------------------------------------------------------------------===//
// Listener
//===----------------------------------------------------------------------===//

typedef enum iree_net_rdma_listener_state_e {
  IREE_NET_RDMA_LISTENER_STATE_LISTENING = 0,
  IREE_NET_RDMA_LISTENER_STATE_STOPPING = 1,
  IREE_NET_RDMA_LISTENER_STATE_STOPPED = 2,
} iree_net_rdma_listener_state_t;

typedef struct iree_net_rdma_accept_state_t iree_net_rdma_accept_state_t;

typedef struct iree_net_rdma_listener_t {
  // Base listener; must be first for vtable dispatch.
  iree_net_listener_t base;

  // Factory retained while the listener is alive.
  iree_net_rdma_factory_t* factory;

  // Proactor used for CM events and stopped callbacks. Retained.
  iree_async_proactor_t* proactor;

  // Receive pool borrowed by accepted carriers.
  iree_async_buffer_pool_t* recv_pool;

  // rdma_cm event channel wrapper.
  iree_net_rdma_cm_channel_t* cm_channel;

  // Listening rdma_cm ID.
  struct rdma_cm_id* listen_id;

  // Bound listener address after rdma_bind_addr.
  iree_async_address_t bound_address;

  // Application accept callback.
  iree_net_listener_accept_callback_t accept_callback;

  // User data passed to accept_callback.
  void* accept_user_data;

  // Listener stopped callback.
  iree_net_listener_stopped_callback_t stopped_callback;

  // Stop operation used when stop is requested from inside a CM callback.
  iree_async_nop_operation_t stop_operation;

  // Stop notification operation.
  iree_async_nop_operation_t stopped_operation;

  // Linked list of accepted IDs waiting for ESTABLISHED.
  iree_net_rdma_accept_state_t* pending_accepts;

  // Listener lifecycle state.
  iree_net_rdma_listener_state_t state;

  // True while executing the rdma_cm channel callback.
  bool in_callback;

  // Host allocator used for this listener allocation.
  iree_allocator_t host_allocator;
} iree_net_rdma_listener_t;

struct iree_net_rdma_accept_state_t {
  // Next pending accepted connection in listener list.
  iree_net_rdma_accept_state_t* next;

  // Listener owning this accept state.
  iree_net_rdma_listener_t* listener;

  // Raw connection ID before carrier creation transfers ownership.
  struct rdma_cm_id* connection_id;

  // Carrier created for the accepted connection.
  iree_net_carrier_t* carrier;

  // Child context matching connection_id->verbs.
  iree_net_rdma_context_t* connection_context;

  // Serialized local RDMA private data sent by rdma_accept.
  uint8_t private_data[IREE_NET_RDMA_CONNECTION_DATA_LENGTH];

  // Number of bytes valid in private_data.
  iree_host_size_t private_data_length;

  // Cleanup operation submitted after the CM event callback returns.
  iree_async_nop_operation_t cleanup_operation;

  // Host allocator used for this state allocation.
  iree_allocator_t host_allocator;
};

static const iree_net_listener_vtable_t iree_net_rdma_listener_vtable;

static void iree_net_rdma_accept_state_cleanup(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_rdma_accept_state_t* accept_state =
      (iree_net_rdma_accept_state_t*)user_data;
  if (!iree_status_is_ok(status)) iree_status_abort(status);

  iree_net_carrier_release(accept_state->carrier);
  iree_net_rdma_context_release(accept_state->connection_context);
  if (accept_state->connection_id) {
    const iree_net_librdmacm_t* librdmacm = iree_net_rdma_context_librdmacm(
        accept_state->listener->factory->context);
    int result = librdmacm->rdma_destroy_id(accept_state->connection_id);
    iree_status_t destroy_status = iree_net_rdma_factory_status_from_result(
        __FILE__, __LINE__, result, "rdma_destroy_id");
    if (!iree_status_is_ok(destroy_status)) iree_status_abort(destroy_status);
  }
  iree_allocator_t host_allocator = accept_state->host_allocator;
  iree_allocator_free(host_allocator, accept_state);
}

static void iree_net_rdma_accept_state_submit_cleanup(
    iree_net_rdma_accept_state_t* accept_state) {
  memset(&accept_state->cleanup_operation, 0,
         sizeof(accept_state->cleanup_operation));
  iree_async_operation_initialize(
      &accept_state->cleanup_operation.base, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_rdma_accept_state_cleanup,
      accept_state);
  iree_status_t status = iree_async_proactor_submit_one(
      accept_state->listener->proactor, &accept_state->cleanup_operation.base);
  if (!iree_status_is_ok(status)) iree_status_abort(status);
}

static void iree_net_rdma_listener_push_accept(
    iree_net_rdma_listener_t* listener,
    iree_net_rdma_accept_state_t* accept_state) {
  accept_state->next = listener->pending_accepts;
  listener->pending_accepts = accept_state;
}

static iree_net_rdma_accept_state_t* iree_net_rdma_listener_pop_accept(
    iree_net_rdma_listener_t* listener, struct rdma_cm_id* id) {
  iree_net_rdma_accept_state_t** link = &listener->pending_accepts;
  while (*link) {
    iree_net_rdma_accept_state_t* accept_state = *link;
    iree_net_rdma_carrier_t* carrier =
        iree_net_rdma_carrier_cast(accept_state->carrier);
    if (carrier && iree_net_rdma_carrier_connection_id(carrier) == id) {
      *link = accept_state->next;
      accept_state->next = NULL;
      return accept_state;
    }
    link = &accept_state->next;
  }
  return NULL;
}

static void iree_net_rdma_listener_deliver_accept(
    iree_net_rdma_listener_t* listener,
    iree_net_rdma_accept_state_t* accept_state) {
  iree_net_connection_t* connection = NULL;
  iree_status_t status = iree_net_rdma_connection_create(
      listener->proactor, listener->recv_pool, accept_state->carrier,
      accept_state->host_allocator, &connection);
  if (iree_status_is_ok(status)) {
    accept_state->carrier = NULL;
    listener->accept_callback(listener->accept_user_data, iree_ok_status(),
                              connection);
  } else {
    listener->accept_callback(listener->accept_user_data, status, NULL);
  }
  iree_net_rdma_accept_state_submit_cleanup(accept_state);
}

static iree_status_t iree_net_rdma_accept_state_create(
    iree_net_rdma_listener_t* listener, const iree_net_rdma_cm_event_t* event,
    iree_net_rdma_accept_state_t** out_accept_state) {
  *out_accept_state = NULL;

  iree_net_rdma_accept_state_t* accept_state = NULL;
  iree_status_t status = iree_allocator_malloc(
      listener->host_allocator, sizeof(*accept_state), (void**)&accept_state);
  if (iree_status_is_ok(status)) {
    memset(accept_state, 0, sizeof(*accept_state));
    accept_state->listener = listener;
    accept_state->connection_id = event->id;
    accept_state->host_allocator = listener->host_allocator;
    *out_accept_state = accept_state;
  }

  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_context_create_for_cm_id(
        listener->factory->context, accept_state->connection_id,
        listener->host_allocator, &accept_state->connection_context);
  }

  if (iree_status_is_ok(status)) {
    iree_net_carrier_callback_t callback = {
        .fn = iree_net_rdma_connection_carrier_completion,
        .user_data = NULL,
    };
    iree_net_rdma_carrier_create_params_t params = {
        .context = accept_state->connection_context,
        .proactor = listener->proactor,
        .recv_pool = listener->recv_pool,
        .connection_id = accept_state->connection_id,
        .callback = callback,
        .options = listener->factory->carrier_options,
    };
    status = iree_net_rdma_carrier_create(params, listener->host_allocator,
                                          &accept_state->carrier);
  }

  if (iree_status_is_ok(status)) {
    accept_state->connection_id = NULL;
    iree_net_rdma_carrier_t* carrier =
        iree_net_rdma_carrier_cast(accept_state->carrier);
    status = iree_net_rdma_carrier_apply_remote_connection_data(
        carrier, event->private_data);
  }

  if (iree_status_is_ok(status)) {
    iree_net_rdma_carrier_t* carrier =
        iree_net_rdma_carrier_cast(accept_state->carrier);
    status = iree_net_rdma_carrier_serialize_local_connection_data(
        carrier,
        iree_make_byte_span(accept_state->private_data,
                            sizeof(accept_state->private_data)),
        &accept_state->private_data_length);
  }

  return status;
}

static iree_status_t iree_net_rdma_listener_accept_request(
    iree_net_rdma_listener_t* listener, const iree_net_rdma_cm_event_t* event) {
  if (event->private_data.data_length == 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "RDMA peer did not provide private data");
  }

  iree_net_rdma_accept_state_t* accept_state = NULL;
  iree_status_t status =
      iree_net_rdma_accept_state_create(listener, event, &accept_state);

  struct rdma_conn_param conn_param;
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_factory_initialize_conn_param(
        iree_make_const_byte_span(accept_state->private_data,
                                  accept_state->private_data_length),
        &conn_param);
  }

  if (iree_status_is_ok(status)) {
    const iree_net_librdmacm_t* librdmacm =
        iree_net_rdma_context_librdmacm(listener->factory->context);
    int result = librdmacm->rdma_accept(
        iree_net_rdma_carrier_connection_id(
            iree_net_rdma_carrier_cast(accept_state->carrier)),
        &conn_param);
    status = iree_net_rdma_factory_status_from_result(__FILE__, __LINE__,
                                                      result, "rdma_accept");
  }

  if (iree_status_is_ok(status)) {
    iree_net_rdma_listener_push_accept(listener, accept_state);
  } else {
    const iree_net_librdmacm_t* librdmacm =
        iree_net_rdma_context_librdmacm(listener->factory->context);
    int result = librdmacm->rdma_reject(event->id, NULL, 0);
    status = iree_status_join(status,
                              iree_net_rdma_factory_status_from_result(
                                  __FILE__, __LINE__, result, "rdma_reject"));
    if (accept_state) {
      iree_net_rdma_accept_state_submit_cleanup(accept_state);
    } else {
      result = librdmacm->rdma_destroy_id(event->id);
      status = iree_status_join(
          status, iree_net_rdma_factory_status_from_result(
                      __FILE__, __LINE__, result, "rdma_destroy_id"));
    }
    listener->accept_callback(listener->accept_user_data, status, NULL);
  }
  return iree_ok_status();
}

static iree_status_t iree_net_rdma_listener_discard_stopping_event(
    iree_net_rdma_listener_t* listener, const iree_net_rdma_cm_event_t* event) {
  iree_status_t status = iree_ok_status();
  if (event && event->type == RDMA_CM_EVENT_CONNECT_REQUEST) {
    const iree_net_librdmacm_t* librdmacm =
        iree_net_rdma_context_librdmacm(listener->factory->context);
    int result = librdmacm->rdma_reject(event->id, NULL, 0);
    status = iree_net_rdma_factory_status_from_result(__FILE__, __LINE__,
                                                      result, "rdma_reject");
    result = librdmacm->rdma_destroy_id(event->id);
    status = iree_status_join(
        status, iree_net_rdma_factory_status_from_result(
                    __FILE__, __LINE__, result, "rdma_destroy_id"));
  }
  return status;
}

static void iree_net_rdma_listener_on_cm_event(
    void* user_data, iree_status_t status,
    const iree_net_rdma_cm_event_t* event) {
  iree_net_rdma_listener_t* listener = (iree_net_rdma_listener_t*)user_data;
  listener->in_callback = true;
  if (listener->state != IREE_NET_RDMA_LISTENER_STATE_LISTENING) {
    if (!iree_status_is_ok(status)) iree_status_abort(status);
    status = iree_net_rdma_listener_discard_stopping_event(listener, event);
    if (!iree_status_is_ok(status)) iree_status_abort(status);
    listener->in_callback = false;
    return;
  }
  if (!iree_status_is_ok(status)) {
    listener->accept_callback(listener->accept_user_data, status, NULL);
    listener->in_callback = false;
    return;
  }

  switch (event->type) {
    case RDMA_CM_EVENT_CONNECT_REQUEST: {
      status = iree_net_rdma_listener_accept_request(listener, event);
      if (!iree_status_is_ok(status)) iree_status_abort(status);
      break;
    }
    case RDMA_CM_EVENT_ESTABLISHED: {
      iree_net_rdma_accept_state_t* accept_state =
          iree_net_rdma_listener_pop_accept(listener, event->id);
      if (accept_state) {
        iree_net_rdma_listener_deliver_accept(listener, accept_state);
      } else {
        listener->accept_callback(
            listener->accept_user_data,
            iree_make_status(IREE_STATUS_NOT_FOUND,
                             "RDMA established event for unknown connection"),
            NULL);
      }
      break;
    }
    case RDMA_CM_EVENT_REJECTED:
    case RDMA_CM_EVENT_CONNECT_ERROR:
    case RDMA_CM_EVENT_UNREACHABLE:
    case RDMA_CM_EVENT_DISCONNECTED:
    case RDMA_CM_EVENT_DEVICE_REMOVAL: {
      iree_net_rdma_accept_state_t* accept_state =
          iree_net_rdma_listener_pop_accept(listener, event->id);
      if (accept_state) {
        iree_net_rdma_accept_state_submit_cleanup(accept_state);
      }
      listener->accept_callback(
          listener->accept_user_data,
          iree_net_rdma_factory_status_from_failed_cm_event(event,
                                                            "rdma_cm accept"),
          NULL);
      break;
    }
    default:
      listener->accept_callback(
          listener->accept_user_data,
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "unexpected RDMA CM event %d on listener",
                           (int)event->type),
          NULL);
      break;
  }
  listener->in_callback = false;
}

static void iree_net_rdma_listener_stopped_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_rdma_listener_t* listener = (iree_net_rdma_listener_t*)user_data;
  if (!iree_status_is_ok(status)) iree_status_abort(status);
  listener->state = IREE_NET_RDMA_LISTENER_STATE_STOPPED;
  listener->stopped_callback.fn(listener->stopped_callback.user_data);
}

static iree_status_t iree_net_rdma_listener_begin_stop(
    iree_net_rdma_listener_t* listener) {
  const iree_net_librdmacm_t* librdmacm =
      iree_net_rdma_context_librdmacm(listener->factory->context);

  iree_status_t status = iree_ok_status();
  if (listener->listen_id) {
    struct rdma_cm_id* listen_id = listener->listen_id;
    int result = librdmacm->rdma_destroy_id(listen_id);
    status = iree_net_rdma_factory_status_from_result(
        __FILE__, __LINE__, result, "rdma_destroy_id");
    if (iree_status_is_ok(status)) listener->listen_id = NULL;
  }
  if (iree_status_is_ok(status)) {
    iree_net_rdma_cm_channel_release(listener->cm_channel);
    listener->cm_channel = NULL;
  }
  if (iree_status_is_ok(status)) {
    memset(&listener->stopped_operation, 0,
           sizeof(listener->stopped_operation));
    iree_async_operation_initialize(
        &listener->stopped_operation.base, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_rdma_listener_stopped_complete,
        listener);
    status = iree_async_proactor_submit_one(listener->proactor,
                                            &listener->stopped_operation.base);
  }
  return status;
}

static void iree_net_rdma_listener_stop_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_rdma_listener_t* listener = (iree_net_rdma_listener_t*)user_data;
  if (!iree_status_is_ok(status)) iree_status_abort(status);
  status = iree_net_rdma_listener_begin_stop(listener);
  if (!iree_status_is_ok(status)) iree_status_abort(status);
}

static void iree_net_rdma_listener_free(iree_net_listener_t* base_listener) {
  iree_net_rdma_listener_t* listener = (iree_net_rdma_listener_t*)base_listener;
  IREE_ASSERT(listener->state == IREE_NET_RDMA_LISTENER_STATE_STOPPED,
              "listener must be stopped before free");
  iree_allocator_t host_allocator = listener->host_allocator;
  iree_async_proactor_release(listener->proactor);
  iree_net_transport_factory_release(&listener->factory->base);
  iree_allocator_free(host_allocator, listener);
}

static iree_status_t iree_net_rdma_listener_stop(
    iree_net_listener_t* base_listener,
    iree_net_listener_stopped_callback_t callback) {
  iree_net_rdma_listener_t* listener = (iree_net_rdma_listener_t*)base_listener;
  if (listener->state != IREE_NET_RDMA_LISTENER_STATE_LISTENING) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "listener is already stopping or stopped");
  }
  if (listener->pending_accepts) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "listener has pending RDMA accepts that have not established");
  }

  listener->state = IREE_NET_RDMA_LISTENER_STATE_STOPPING;
  listener->stopped_callback = callback;
  iree_status_t status = iree_ok_status();
  if (listener->in_callback) {
    memset(&listener->stop_operation, 0, sizeof(listener->stop_operation));
    iree_async_operation_initialize(
        &listener->stop_operation.base, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_rdma_listener_stop_complete,
        listener);
    status = iree_async_proactor_submit_one(listener->proactor,
                                            &listener->stop_operation.base);
  } else {
    status = iree_net_rdma_listener_begin_stop(listener);
  }
  if (!iree_status_is_ok(status)) {
    listener->state = IREE_NET_RDMA_LISTENER_STATE_LISTENING;
    listener->stopped_callback = (iree_net_listener_stopped_callback_t){0};
  }
  return status;
}

static iree_status_t iree_net_rdma_listener_query_bound_address(
    iree_net_listener_t* base_listener, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_address) {
  iree_net_rdma_listener_t* listener = (iree_net_rdma_listener_t*)base_listener;
  return iree_async_address_format(&listener->bound_address, buffer_capacity,
                                   buffer, out_address);
}

static const iree_net_listener_vtable_t iree_net_rdma_listener_vtable = {
    .free = iree_net_rdma_listener_free,
    .stop = iree_net_rdma_listener_stop,
    .query_bound_address = iree_net_rdma_listener_query_bound_address,
};

//===----------------------------------------------------------------------===//
// Factory
//===----------------------------------------------------------------------===//

static iree_net_rdma_factory_options_t iree_net_rdma_factory_resolve_options(
    iree_net_rdma_factory_options_t options) {
  iree_net_rdma_factory_options_t defaults =
      iree_net_rdma_factory_options_default();
  if (options.resolve_timeout_ms == 0) {
    options.resolve_timeout_ms = defaults.resolve_timeout_ms;
  }
  if (options.listen_backlog == 0) {
    options.listen_backlog = defaults.listen_backlog;
  }
  if (options.max_endpoint_count == 0) {
    options.max_endpoint_count = defaults.max_endpoint_count;
  }
  return options;
}

static iree_status_t iree_net_rdma_factory_validate_options(
    iree_net_rdma_factory_options_t options) {
  if (options.max_endpoint_count != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "RDMA factory currently supports exactly one endpoint per connection");
  }
  if (options.resolve_timeout_ms < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "resolve_timeout_ms must be non-negative");
  }
  if (options.listen_backlog < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "listen_backlog must be non-negative");
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_net_rdma_factory_create(
    iree_net_rdma_factory_options_t options, iree_allocator_t host_allocator,
    iree_net_transport_factory_t** out_factory) {
  IREE_ASSERT_ARGUMENT(out_factory);
  *out_factory = NULL;

  options = iree_net_rdma_factory_resolve_options(options);
  iree_status_t status = iree_net_rdma_factory_validate_options(options);

  iree_net_rdma_factory_t* factory = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, sizeof(*factory),
                                   (void**)&factory);
  }

  if (iree_status_is_ok(status)) {
    memset(factory, 0, sizeof(*factory));
    iree_atomic_ref_count_init(&factory->base.ref_count);
    factory->base.vtable = &iree_net_rdma_factory_vtable;
    factory->carrier_options = options.carrier_options;
    factory->resolve_timeout_ms = options.resolve_timeout_ms;
    factory->listen_backlog = options.listen_backlog;
    factory->max_endpoint_count = options.max_endpoint_count;
    factory->host_allocator = host_allocator;
  }

  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_context_create(options.context_options,
                                          host_allocator, &factory->context);
  }

  if (iree_status_is_ok(status)) {
    *out_factory = &factory->base;
  } else if (factory) {
    iree_net_transport_factory_release(&factory->base);
  }
  return status;
}

static void iree_net_rdma_factory_destroy(
    iree_net_transport_factory_t* base_factory) {
  iree_net_rdma_factory_t* factory = (iree_net_rdma_factory_t*)base_factory;
  iree_allocator_t host_allocator = factory->host_allocator;
  iree_net_rdma_context_release(factory->context);
  iree_allocator_free(host_allocator, factory);
}

static iree_net_transport_capabilities_t
iree_net_rdma_factory_query_capabilities(
    iree_net_transport_factory_t* base_factory) {
  (void)base_factory;
  return IREE_NET_TRANSPORT_CAPABILITY_RELIABLE |
         IREE_NET_TRANSPORT_CAPABILITY_ORDERED |
         IREE_NET_TRANSPORT_CAPABILITY_ZERO_COPY_TX |
         IREE_NET_TRANSPORT_CAPABILITY_ZERO_COPY_RX |
         IREE_NET_TRANSPORT_CAPABILITY_RDMA;
}

static iree_status_t iree_net_rdma_factory_connect(
    iree_net_transport_factory_t* base_factory, iree_string_view_t address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_transport_connect_callback_t callback, void* user_data) {
  iree_net_rdma_factory_t* factory = (iree_net_rdma_factory_t*)base_factory;

  iree_async_address_t remote_address;
  iree_status_t status =
      iree_async_address_from_string(address, &remote_address);

  iree_net_rdma_connect_state_t* state = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(factory->host_allocator, sizeof(*state),
                                   (void**)&state);
  }

  if (iree_status_is_ok(status)) {
    memset(state, 0, sizeof(*state));
    state->factory = factory;
    iree_net_transport_factory_retain(base_factory);
    state->proactor = proactor;
    iree_async_proactor_retain(proactor);
    state->recv_pool = recv_pool;
    state->callback = callback;
    state->callback_user_data = user_data;
    state->phase = IREE_NET_RDMA_CONNECT_PHASE_RESOLVING_ADDRESS;
    state->host_allocator = factory->host_allocator;
  }

  if (iree_status_is_ok(status)) {
    iree_net_rdma_cm_channel_callback_t cm_callback = {
        .fn = iree_net_rdma_connect_state_on_cm_event,
        .user_data = state,
    };
    status = iree_net_rdma_cm_channel_create(
        iree_net_rdma_context_librdmacm(factory->context), proactor,
        cm_callback, factory->host_allocator, &state->cm_channel);
  }

  const iree_net_librdmacm_t* librdmacm =
      iree_net_rdma_context_librdmacm(factory->context);
  if (iree_status_is_ok(status)) {
    struct rdma_event_channel* event_channel =
        iree_net_rdma_cm_channel_native_event_channel(state->cm_channel);
    int result = librdmacm->rdma_create_id(event_channel, &state->connection_id,
                                           state, RDMA_PS_TCP);
    status = iree_net_rdma_factory_status_from_result(__FILE__, __LINE__,
                                                      result, "rdma_create_id");
  }

  if (iree_status_is_ok(status)) {
    int result = librdmacm->rdma_resolve_addr(
        state->connection_id, NULL, (struct sockaddr*)remote_address.storage,
        factory->resolve_timeout_ms);
    status = iree_net_rdma_factory_status_from_result(
        __FILE__, __LINE__, result, "rdma_resolve_addr");
  }

  if (!iree_status_is_ok(status) && state) {
    iree_net_rdma_connect_state_submit_cleanup(state);
    state = NULL;
  }
  return status;
}

static iree_status_t iree_net_rdma_factory_create_listener(
    iree_net_transport_factory_t* base_factory, iree_string_view_t bind_address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_listener_accept_callback_t accept_callback, void* user_data,
    iree_allocator_t host_allocator, iree_net_listener_t** out_listener) {
  IREE_ASSERT_ARGUMENT(out_listener);
  *out_listener = NULL;
  iree_net_rdma_factory_t* factory = (iree_net_rdma_factory_t*)base_factory;

  iree_async_address_t address;
  iree_status_t status = iree_async_address_from_string(bind_address, &address);

  iree_net_rdma_listener_t* listener = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, sizeof(*listener),
                                   (void**)&listener);
  }

  if (iree_status_is_ok(status)) {
    memset(listener, 0, sizeof(*listener));
    listener->base.vtable = &iree_net_rdma_listener_vtable;
    listener->factory = factory;
    iree_net_transport_factory_retain(base_factory);
    listener->proactor = proactor;
    iree_async_proactor_retain(proactor);
    listener->recv_pool = recv_pool;
    listener->accept_callback = accept_callback;
    listener->accept_user_data = user_data;
    listener->state = IREE_NET_RDMA_LISTENER_STATE_LISTENING;
    listener->host_allocator = host_allocator;
  }

  if (iree_status_is_ok(status)) {
    iree_net_rdma_cm_channel_callback_t cm_callback = {
        .fn = iree_net_rdma_listener_on_cm_event,
        .user_data = listener,
    };
    status = iree_net_rdma_cm_channel_create(
        iree_net_rdma_context_librdmacm(factory->context), proactor,
        cm_callback, host_allocator, &listener->cm_channel);
  }

  const iree_net_librdmacm_t* librdmacm =
      iree_net_rdma_context_librdmacm(factory->context);
  if (iree_status_is_ok(status)) {
    struct rdma_event_channel* event_channel =
        iree_net_rdma_cm_channel_native_event_channel(listener->cm_channel);
    int result = librdmacm->rdma_create_id(event_channel, &listener->listen_id,
                                           listener, RDMA_PS_TCP);
    status = iree_net_rdma_factory_status_from_result(__FILE__, __LINE__,
                                                      result, "rdma_create_id");
  }

  if (iree_status_is_ok(status)) {
    int result = librdmacm->rdma_bind_addr(listener->listen_id,
                                           (struct sockaddr*)address.storage);
    status = iree_net_rdma_factory_status_from_result(__FILE__, __LINE__,
                                                      result, "rdma_bind_addr");
  }

  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_factory_copy_sockaddr(
        rdma_get_local_addr(listener->listen_id), &listener->bound_address);
  }

  if (iree_status_is_ok(status)) {
    int result =
        librdmacm->rdma_listen(listener->listen_id, factory->listen_backlog);
    status = iree_net_rdma_factory_status_from_result(__FILE__, __LINE__,
                                                      result, "rdma_listen");
  }

  if (iree_status_is_ok(status)) {
    *out_listener = &listener->base;
  } else if (listener) {
    if (listener->listen_id) {
      int result = librdmacm->rdma_destroy_id(listener->listen_id);
      status = iree_status_join(
          status, iree_net_rdma_factory_status_from_result(
                      __FILE__, __LINE__, result, "rdma_destroy_id"));
      listener->listen_id = NULL;
    }
    iree_net_rdma_cm_channel_release(listener->cm_channel);
    listener->cm_channel = NULL;
    iree_async_proactor_release(listener->proactor);
    iree_net_transport_factory_release(&listener->factory->base);
    iree_allocator_free(host_allocator, listener);
  }
  return status;
}

static const iree_net_transport_factory_vtable_t iree_net_rdma_factory_vtable =
    {
        .destroy = iree_net_rdma_factory_destroy,
        .query_capabilities = iree_net_rdma_factory_query_capabilities,
        .connect = iree_net_rdma_factory_connect,
        .create_listener = iree_net_rdma_factory_create_listener,
};
