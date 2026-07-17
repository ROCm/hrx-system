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
#include "iree/base/internal/csprng.h"
#include "iree/net/carrier/rdma/cm_channel.h"
#include "iree/net/carrier/rdma/endpoint_data.h"
#include "iree/net/channel/util/frame_sender.h"
#include "iree/net/connection.h"
#include "iree/net/endpoint_lifecycle.h"
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
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "%s failed with RDMA CM event %d status %d",
                            operation, (int)event->type, event->status);
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
    iree_const_byte_span_t bootstrap_data, struct rdma_conn_param* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  if (bootstrap_data.data_length > UINT8_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "RDMA CM bootstrap data length %" PRIhsz
                            " exceeds uint8_t capacity",
                            bootstrap_data.data_length);
  }
  out_params->private_data = bootstrap_data.data;
  out_params->private_data_len = (uint8_t)bootstrap_data.data_length;
  out_params->responder_resources = 1;
  out_params->initiator_depth = 1;
  out_params->retry_count = 7;
  out_params->rnr_retry_count = 7;
  return iree_ok_status();
}

static iree_status_t iree_net_rdma_factory_generate_group_id(
    uint64_t* out_group_id) {
  *out_group_id = 0;
  IREE_RETURN_IF_ERROR(iree_csprng_fill(
      iree_make_byte_span((uint8_t*)out_group_id, sizeof(*out_group_id))));
  if (*out_group_id == 0) {
    *out_group_id = 1;
  }
  return iree_ok_status();
}

static iree_status_t iree_net_rdma_factory_serialize_endpoint_data(
    iree_net_rdma_carrier_t* carrier, uint64_t group_id,
    uint16_t endpoint_index, uint16_t endpoint_count, iree_byte_span_t target,
    iree_host_size_t* out_length) {
  iree_net_rdma_endpoint_data_t endpoint_data;
  memset(&endpoint_data, 0, sizeof(endpoint_data));
  endpoint_data.group_id = group_id;
  endpoint_data.endpoint_index = endpoint_index;
  endpoint_data.endpoint_count = endpoint_count;
  iree_net_rdma_connection_data_t connection_data;
  IREE_RETURN_IF_ERROR(
      iree_net_rdma_carrier_export_connection_data(carrier, &connection_data));
  endpoint_data.bootstrap_recv_buffer_size = connection_data.recv_buffer_size;
  endpoint_data.bootstrap_recv_credits = connection_data.initial_recv_credits;
  return iree_net_rdma_endpoint_data_serialize(&endpoint_data, target,
                                               out_length);
}

static iree_status_t iree_net_rdma_factory_validate_endpoint_slot(
    const iree_net_rdma_endpoint_data_t* endpoint_data, uint16_t endpoint_index,
    uint16_t endpoint_count) {
  if (endpoint_data->endpoint_index != endpoint_index) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "RDMA endpoint_index %" PRIu16
                            " does not match expected %" PRIu16,
                            endpoint_data->endpoint_index, endpoint_index);
  }
  if (endpoint_data->endpoint_count != endpoint_count) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "RDMA endpoint_count %" PRIu16
                            " does not match expected %" PRIu16,
                            endpoint_data->endpoint_count, endpoint_count);
  }
  return iree_ok_status();
}

static iree_status_t iree_net_rdma_factory_validate_endpoint_data(
    const iree_net_rdma_endpoint_data_t* endpoint_data, uint64_t group_id,
    uint16_t endpoint_index, uint16_t endpoint_count) {
  if (endpoint_data->group_id != group_id) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "RDMA endpoint group_id mismatch");
  }
  return iree_net_rdma_factory_validate_endpoint_slot(
      endpoint_data, endpoint_index, endpoint_count);
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
};

typedef struct iree_net_rdma_endpoint_t {
  // Back-pointer to the owning connection.
  iree_net_rdma_connection_t* connection;

  // Borrowed carrier owned by the connection.
  iree_net_carrier_t* carrier;

  // Endpoint-ready operation submitted by open_endpoint.
  iree_async_nop_operation_t ready_operation;

  // Endpoint-ready callback invoked by ready_operation.
  iree_net_endpoint_ready_callback_t ready_callback;

  // Message and error callbacks installed by the endpoint consumer.
  iree_net_message_endpoint_callbacks_t callbacks;

  // Coordinates endpoint and connection deactivation requests.
  iree_net_endpoint_lifecycle_t lifecycle;

  // Zero-based endpoint slot index in the owning connection.
  uint32_t endpoint_index;

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
  iree_net_endpoint_lifecycle_complete_deactivation(&endpoint->lifecycle);
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
  IREE_RETURN_IF_ERROR(
      iree_net_endpoint_lifecycle_activate(&endpoint->lifecycle));

  iree_net_carrier_t* carrier = endpoint->carrier;
  iree_net_carrier_set_recv_handler(carrier,
                                    (iree_net_carrier_recv_handler_t){
                                        .fn = iree_net_rdma_endpoint_on_recv,
                                        .user_data = endpoint,
                                    });
  iree_net_carrier_state_t carrier_state = iree_net_carrier_state(carrier);
  iree_status_t status = iree_ok_status();
  if (carrier_state == IREE_NET_CARRIER_STATE_CREATED) {
    status = iree_net_carrier_activate(carrier);
  } else if (carrier_state != IREE_NET_CARRIER_STATE_ACTIVE) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "carrier is not activatable");
  }
  if (!iree_status_is_ok(status)) {
    iree_net_endpoint_lifecycle_rollback_activation(&endpoint->lifecycle);
  }
  return status;
}

static iree_status_t iree_net_rdma_endpoint_deactivate(
    void* self, iree_net_message_endpoint_deactivate_fn_t callback,
    void* user_data) {
  iree_net_rdma_endpoint_t* endpoint = (iree_net_rdma_endpoint_t*)self;
  iree_net_endpoint_lifecycle_actions_t actions =
      IREE_NET_ENDPOINT_LIFECYCLE_ACTION_NONE;
  IREE_RETURN_IF_ERROR(iree_net_endpoint_lifecycle_request_deactivation(
      &endpoint->lifecycle, callback, user_data, &actions));
  if (iree_any_bit_set(actions,
                       IREE_NET_ENDPOINT_LIFECYCLE_ACTION_BEGIN_DEACTIVATION)) {
    iree_status_t status = iree_net_carrier_deactivate(
        endpoint->carrier, iree_net_rdma_endpoint_carrier_deactivated,
        endpoint);
    if (!iree_status_is_ok(status)) iree_status_abort(status);
  }
  return iree_ok_status();
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

struct iree_net_rdma_connection_t {
  // Base connection; must be first for vtable dispatch.
  iree_net_connection_t base;

  // Proactor used to deliver endpoint-ready NOPs. Retained by the connection.
  iree_async_proactor_t* proactor;

  // CM event channel owning established endpoint rdma_cm_id event delivery.
  iree_net_rdma_cm_channel_t* cm_channel;

  // Maximum number of endpoint slots in endpoints.
  uint32_t max_endpoint_count;

  // Number of endpoint slots already reserved by open_endpoint.
  uint32_t allocated_endpoint_count;

  // Joins endpoint drains during connection deactivation.
  iree_net_endpoint_deactivation_barrier_t deactivation_barrier;

  // Flexible array of endpoint slots owned by this connection.
  iree_net_rdma_endpoint_t endpoints[];
};

static const iree_net_connection_vtable_t iree_net_rdma_connection_vtable;

static iree_net_rdma_endpoint_t* iree_net_rdma_connection_find_endpoint_by_id(
    iree_net_rdma_connection_t* connection, struct rdma_cm_id* id) {
  if (!connection || !id) return NULL;
  for (uint32_t i = 0; i < connection->max_endpoint_count; ++i) {
    iree_net_rdma_endpoint_t* endpoint = &connection->endpoints[i];
    iree_net_rdma_carrier_t* carrier =
        iree_net_rdma_carrier_cast(endpoint->carrier);
    if (carrier && iree_net_rdma_carrier_connection_id(carrier) == id) {
      return endpoint;
    }
  }
  return NULL;
}

static void iree_net_rdma_endpoint_notify_error(
    iree_net_rdma_endpoint_t* endpoint, iree_status_t status) {
  bool endpoint_allocated =
      iree_any_bit_set(endpoint->flags, IREE_NET_RDMA_ENDPOINT_FLAG_ALLOCATED);
  if (endpoint_allocated && endpoint->callbacks.on_error) {
    endpoint->callbacks.on_error(endpoint->callbacks.user_data, status);
  } else if (endpoint_allocated && !iree_status_is_ok(status)) {
    iree_status_abort(status);
  } else if (!iree_status_is_ok(status)) {
    iree_net_rdma_carrier_record_failure(
        iree_net_rdma_carrier_cast(endpoint->carrier), status);
  }
}

static void iree_net_rdma_connection_on_cm_event(
    void* user_data, iree_status_t status,
    const iree_net_rdma_cm_event_t* event) {
  iree_net_rdma_connection_t* connection =
      (iree_net_rdma_connection_t*)user_data;
  if (!iree_status_is_ok(status)) {
    iree_status_abort(status);
    return;
  }

  switch (event->type) {
    case RDMA_CM_EVENT_DISCONNECTED:
    case RDMA_CM_EVENT_TIMEWAIT_EXIT:
      break;
    case RDMA_CM_EVENT_CONNECT_ERROR:
    case RDMA_CM_EVENT_UNREACHABLE:
    case RDMA_CM_EVENT_REJECTED:
    case RDMA_CM_EVENT_DEVICE_REMOVAL: {
      iree_status_t event_status =
          iree_net_rdma_factory_status_from_failed_cm_event(
              event, "rdma_cm connection");
      iree_net_rdma_endpoint_t* endpoint =
          iree_net_rdma_connection_find_endpoint_by_id(connection, event->id);
      if (endpoint) {
        iree_net_rdma_endpoint_notify_error(endpoint, event_status);
      } else {
        iree_status_abort(event_status);
      }
      break;
    }
    default:
      iree_status_abort(iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "unexpected RDMA CM event %d on established connection",
          (int)event->type));
      break;
  }
}

static void iree_net_rdma_discard_cm_event(
    void* user_data, iree_status_t status,
    const iree_net_rdma_cm_event_t* event) {
  (void)user_data;
  (void)event;
  if (!iree_status_is_ok(status)) iree_status_abort(status);
}

static void iree_net_rdma_endpoint_notify_send_ready(
    iree_net_rdma_endpoint_t* endpoint) {
  if (endpoint->callbacks.on_send_ready &&
      iree_any_bit_set(endpoint->flags,
                       IREE_NET_RDMA_ENDPOINT_FLAG_ALLOCATED)) {
    endpoint->callbacks.on_send_ready(endpoint->callbacks.user_data);
  }
}

static bool iree_net_rdma_endpoint_completion_is_deactivation_cancel(
    iree_net_rdma_endpoint_t* endpoint, iree_net_carrier_completion_kind_t kind,
    iree_status_code_t status_code) {
  if (status_code != IREE_STATUS_CANCELLED) return false;
  switch (kind) {
    case IREE_NET_CARRIER_COMPLETION_SEND:
    case IREE_NET_CARRIER_COMPLETION_DIRECT_WRITE:
    case IREE_NET_CARRIER_COMPLETION_DIRECT_READ:
      break;
    default:
      return false;
  }

  iree_net_carrier_state_t state = iree_net_carrier_state(endpoint->carrier);
  return state == IREE_NET_CARRIER_STATE_DRAINING ||
         state == IREE_NET_CARRIER_STATE_DEACTIVATED;
}

static void iree_net_rdma_connection_carrier_completion(
    void* callback_user_data, iree_net_carrier_completion_kind_t kind,
    uint64_t operation_user_data, iree_status_t status,
    iree_host_size_t bytes_transferred, iree_async_buffer_lease_t* recv_lease) {
  iree_net_rdma_endpoint_t* endpoint =
      (iree_net_rdma_endpoint_t*)callback_user_data;
  if (kind == IREE_NET_CARRIER_COMPLETION_SEND_READY) {
    if (iree_status_is_ok(status)) {
      iree_net_rdma_endpoint_notify_send_ready(endpoint);
    } else {
      iree_net_rdma_endpoint_notify_error(endpoint, status);
    }
    return;
  }

  bool deactivation_cancel =
      iree_net_rdma_endpoint_completion_is_deactivation_cancel(
          endpoint, kind, iree_status_code(status));
  if (kind == IREE_NET_CARRIER_COMPLETION_SEND && operation_user_data != 0) {
    iree_status_t endpoint_status = iree_ok_status();
    if (!iree_status_is_ok(status) && !deactivation_cancel) {
      endpoint_status = iree_status_clone(status);
    }
    iree_net_frame_sender_dispatch_carrier_completion(
        NULL, kind, operation_user_data, status, bytes_transferred, recv_lease);
    if (!iree_status_is_ok(endpoint_status)) {
      iree_net_rdma_endpoint_notify_error(endpoint, endpoint_status);
    }
    return;
  }

  if (!iree_status_is_ok(status)) {
    if (deactivation_cancel) {
      (void)iree_status_consume_code(status);
    } else {
      iree_net_rdma_endpoint_notify_error(endpoint, status);
    }
  }
}

static void iree_net_rdma_connection_deactivate(
    iree_net_connection_t* base_connection,
    iree_net_connection_deactivate_callback_t callback) {
  iree_net_rdma_connection_t* connection =
      (iree_net_rdma_connection_t*)base_connection;
  iree_net_connection_retain(base_connection);
  iree_net_endpoint_deactivation_barrier_initialize(
      callback, &connection->deactivation_barrier);
  for (uint32_t i = 0; i < connection->max_endpoint_count; ++i) {
    iree_net_rdma_endpoint_t* endpoint = &connection->endpoints[i];
    iree_net_endpoint_lifecycle_actions_t actions =
        iree_net_endpoint_lifecycle_join_deactivation(
            &endpoint->lifecycle, &connection->deactivation_barrier);
    if (iree_any_bit_set(
            actions, IREE_NET_ENDPOINT_LIFECYCLE_ACTION_BEGIN_DEACTIVATION)) {
      iree_status_t status = iree_net_carrier_deactivate(
          endpoint->carrier, iree_net_rdma_endpoint_carrier_deactivated,
          endpoint);
      if (!iree_status_is_ok(status)) {
        iree_status_abort(status);
      }
    }
  }
  iree_net_endpoint_deactivation_barrier_commit(
      &connection->deactivation_barrier);
  iree_net_connection_release(base_connection);
}

static void iree_net_rdma_connection_destroy(
    iree_net_connection_t* base_connection) {
  iree_net_rdma_connection_t* connection =
      (iree_net_rdma_connection_t*)base_connection;
  iree_allocator_t host_allocator = connection->base.host_allocator;
  for (uint32_t i = 0; i < connection->max_endpoint_count; ++i) {
    iree_net_rdma_endpoint_t* endpoint = &connection->endpoints[i];
    iree_net_endpoint_lifecycle_deinitialize(&endpoint->lifecycle);
    iree_net_carrier_release(endpoint->carrier);
  }
  iree_net_rdma_cm_channel_release(connection->cm_channel);
  iree_async_proactor_release(connection->proactor);
  iree_allocator_free(host_allocator, connection);
}

static void iree_net_rdma_endpoint_ready_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_rdma_endpoint_t* endpoint = (iree_net_rdma_endpoint_t*)user_data;
  if (iree_status_is_ok(status)) {
    iree_net_message_endpoint_t message_endpoint = {
        .self = endpoint,
        .vtable = &iree_net_rdma_endpoint_vtable,
    };
    endpoint->ready_callback.fn(endpoint->ready_callback.user_data,
                                iree_ok_status(), message_endpoint);
  } else {
    endpoint->flags &= ~IREE_NET_RDMA_ENDPOINT_FLAG_ALLOCATED;
    endpoint->ready_callback.fn(endpoint->ready_callback.user_data, status,
                                (iree_net_message_endpoint_t){0});
  }
  iree_net_connection_release(&endpoint->connection->base);
}

static iree_status_t iree_net_rdma_connection_open_endpoint(
    iree_net_connection_t* base_connection,
    iree_net_endpoint_ready_callback_t callback) {
  iree_net_rdma_connection_t* connection =
      (iree_net_rdma_connection_t*)base_connection;
  if (connection->allocated_endpoint_count >= connection->max_endpoint_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "all %u RDMA endpoint slots allocated",
                            (unsigned)connection->max_endpoint_count);
  }

  uint32_t endpoint_index = connection->allocated_endpoint_count++;
  iree_net_rdma_endpoint_t* endpoint = &connection->endpoints[endpoint_index];
  endpoint->flags |= IREE_NET_RDMA_ENDPOINT_FLAG_ALLOCATED;
  endpoint->ready_callback = callback;
  iree_async_operation_zero(&endpoint->ready_operation.base,
                            sizeof(endpoint->ready_operation));
  iree_async_operation_initialize(
      &endpoint->ready_operation.base, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_rdma_endpoint_ready_complete,
      endpoint);

  iree_net_connection_retain(base_connection);
  iree_status_t status = iree_async_proactor_submit_one(
      connection->proactor, &endpoint->ready_operation.base);
  if (!iree_status_is_ok(status)) {
    iree_net_connection_release(base_connection);
    --connection->allocated_endpoint_count;
    endpoint->flags &= ~IREE_NET_RDMA_ENDPOINT_FLAG_ALLOCATED;
  }
  return status;
}

static iree_net_carrier_t* iree_net_rdma_connection_carrier(
    iree_net_connection_t* base_connection) {
  iree_net_rdma_connection_t* connection =
      (iree_net_rdma_connection_t*)base_connection;
  if (connection->max_endpoint_count == 0) return NULL;
  return connection->endpoints[0].carrier;
}

static const iree_net_connection_vtable_t iree_net_rdma_connection_vtable = {
    .destroy = iree_net_rdma_connection_destroy,
    .deactivate = iree_net_rdma_connection_deactivate,
    .open_endpoint = iree_net_rdma_connection_open_endpoint,
    .carrier = iree_net_rdma_connection_carrier,
};

static iree_status_t iree_net_rdma_connection_create(
    iree_async_proactor_t* proactor, uint32_t endpoint_count,
    iree_net_carrier_t** carriers, iree_net_rdma_cm_channel_t* cm_channel,
    bool install_cm_callback, iree_allocator_t host_allocator,
    iree_net_connection_t** out_connection) {
  *out_connection = NULL;

  iree_status_t status = iree_ok_status();
  if (endpoint_count == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "RDMA connection requires at least one endpoint");
  }
  for (uint32_t i = 0; i < endpoint_count && iree_status_is_ok(status); ++i) {
    if (!carriers || !carriers[i]) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "RDMA connection carrier %u must not be NULL",
                                (unsigned)i);
    }
  }

  iree_net_rdma_connection_t* connection = NULL;
  iree_host_size_t total_size = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(*connection), &total_size,
        IREE_STRUCT_FIELD_FAM(endpoint_count, iree_net_rdma_endpoint_t));
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&connection);
  }
  if (iree_status_is_ok(status)) {
    memset(connection, 0, total_size);
    iree_net_connection_initialize(&iree_net_rdma_connection_vtable,
                                   host_allocator, endpoint_count,
                                   &connection->base);
    connection->proactor = proactor;
    iree_async_proactor_retain(proactor);
    connection->cm_channel = cm_channel;
    iree_net_rdma_cm_channel_retain(connection->cm_channel);
    connection->max_endpoint_count = endpoint_count;
  }

  if (iree_status_is_ok(status) && connection->cm_channel &&
      install_cm_callback) {
    iree_net_rdma_cm_channel_callback_t cm_callback = {
        .fn = iree_net_rdma_connection_on_cm_event,
        .user_data = connection,
    };
    iree_net_rdma_cm_channel_set_callback(connection->cm_channel, cm_callback);
  }

  if (iree_status_is_ok(status)) {
    for (uint32_t i = 0; i < endpoint_count; ++i) {
      iree_net_rdma_endpoint_t* endpoint = &connection->endpoints[i];
      endpoint->connection = connection;
      endpoint->carrier = carriers[i];
      endpoint->endpoint_index = i;
      iree_net_endpoint_lifecycle_initialize(&endpoint->lifecycle);
      carriers[i]->callback.fn = iree_net_rdma_connection_carrier_completion;
      carriers[i]->callback.user_data = endpoint;
      carriers[i] = NULL;
    }
    *out_connection = &connection->base;
  } else if (connection) {
    iree_net_rdma_connection_destroy(&connection->base);
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

typedef struct iree_net_rdma_connect_state_t iree_net_rdma_connect_state_t;

typedef struct iree_net_rdma_connect_endpoint_t {
  // Owning logical connect state.
  iree_net_rdma_connect_state_t* state;

  // Remote endpoint bootstrap data received from rdma_cm.
  iree_net_rdma_endpoint_data_t remote_endpoint_data;

  // Raw connection ID before carrier creation transfers ownership.
  struct rdma_cm_id* connection_id;

  // Carrier created after address resolution.
  iree_net_carrier_t* carrier;

  // Child context matching connection_id->verbs.
  iree_net_rdma_context_t* connection_context;

  // Serialized local endpoint bootstrap data sent by rdma_connect.
  uint8_t bootstrap_data[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH];

  // Number of bytes valid in bootstrap_data.
  iree_host_size_t bootstrap_data_length;

  // Zero-based endpoint slot index.
  uint16_t endpoint_index;

  // Current per-endpoint CM state-machine phase.
  iree_net_rdma_connect_phase_t phase;

  // True after remote endpoint bootstrap data has been applied.
  bool remote_endpoint_data_applied;

  // True after remote connection data has been applied.
  bool remote_data_applied;

  // True after local connection data has been sent.
  bool local_data_sent;

  // True after the setup exchange has restored public carrier state.
  bool setup_complete;
} iree_net_rdma_connect_endpoint_t;

struct iree_net_rdma_connect_state_t {
  // Cleanup operation used to release cm_channel outside its callback.
  iree_async_nop_operation_t cleanup_operation;

  // Factory retained while the connect is in flight.
  iree_net_rdma_factory_t* factory;

  // Proactor used for CM events and cleanup NOPs.
  iree_async_proactor_t* proactor;

  // rdma_cm event channel wrapper.
  iree_net_rdma_cm_channel_t* cm_channel;

  // User connect callback.
  iree_net_transport_connect_callback_t callback;

  // User data passed to callback.
  void* callback_user_data;

  // Logical connection group identifier shared by endpoint QPs.
  uint64_t group_id;

  // Total endpoint slots in this logical connection.
  uint16_t endpoint_count;

  // Number of endpoint slots that have exchanged connection data.
  uint16_t setup_endpoint_count;

  // True after the user callback has been invoked.
  bool callback_issued;

  // Host allocator used for this state allocation.
  iree_allocator_t host_allocator;

  // Flexible array of endpoint connection substates.
  iree_net_rdma_connect_endpoint_t endpoints[];
};

static void iree_net_rdma_connect_state_cleanup(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_rdma_connect_state_t* state =
      (iree_net_rdma_connect_state_t*)user_data;
  if (!iree_status_is_ok(status)) iree_status_abort(status);

  const iree_net_librdmacm_t* librdmacm =
      iree_net_rdma_context_librdmacm(state->factory->context);
  for (uint16_t i = 0; i < state->endpoint_count; ++i) {
    iree_net_rdma_connect_endpoint_t* endpoint = &state->endpoints[i];
    iree_net_carrier_release(endpoint->carrier);
    iree_net_rdma_context_release(endpoint->connection_context);
    if (endpoint->connection_id) {
      int result = librdmacm->rdma_destroy_id(endpoint->connection_id);
      iree_status_t destroy_status = iree_net_rdma_factory_status_from_result(
          __FILE__, __LINE__, result, "rdma_destroy_id");
      if (!iree_status_is_ok(destroy_status)) iree_status_abort(destroy_status);
    }
  }
  iree_net_rdma_cm_channel_release(state->cm_channel);
  iree_async_proactor_release(state->proactor);
  iree_net_transport_factory_release(&state->factory->base);
  iree_allocator_t host_allocator = state->host_allocator;
  iree_allocator_free(host_allocator, state);
}

static void iree_net_rdma_connect_state_submit_cleanup(
    iree_net_rdma_connect_state_t* state) {
  iree_async_operation_zero(&state->cleanup_operation.base,
                            sizeof(state->cleanup_operation));
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
  state->callback_issued = true;
  state->callback(state->callback_user_data, status, NULL);
  iree_net_rdma_connect_state_submit_cleanup(state);
}

static void iree_net_rdma_connect_state_succeed(
    iree_net_rdma_connect_state_t* state) {
  iree_net_connection_t* connection = NULL;
  iree_net_carrier_t** carriers = NULL;
  iree_status_t status =
      iree_allocator_malloc_array(state->host_allocator, state->endpoint_count,
                                  sizeof(*carriers), (void**)&carriers);
  if (iree_status_is_ok(status)) {
    for (uint16_t i = 0; i < state->endpoint_count; ++i) {
      carriers[i] = state->endpoints[i].carrier;
    }
    status = iree_net_rdma_connection_create(
        state->proactor, state->endpoint_count, carriers, state->cm_channel,
        /*install_cm_callback=*/true, state->host_allocator, &connection);
  }
  if (iree_status_is_ok(status)) {
    for (uint16_t i = 0; i < state->endpoint_count; ++i) {
      state->endpoints[i].carrier = carriers[i];
    }
    state->callback_issued = true;
    state->callback(state->callback_user_data, iree_ok_status(), connection);
  } else {
    state->callback_issued = true;
    state->callback(state->callback_user_data, status, NULL);
  }
  iree_allocator_free(state->host_allocator, carriers);
  iree_net_rdma_connect_state_submit_cleanup(state);
}

static iree_status_t iree_net_rdma_connect_state_create_carrier(
    iree_net_rdma_connect_endpoint_t* endpoint) {
  iree_net_rdma_connect_state_t* state = endpoint->state;
  iree_status_t status = iree_net_rdma_context_create_for_cm_id(
      state->factory->context, endpoint->connection_id, state->host_allocator,
      &endpoint->connection_context);

  if (iree_status_is_ok(status)) {
    iree_net_carrier_callback_t callback = {
        .fn = iree_net_rdma_connection_carrier_completion,
        .user_data = NULL,
    };
    iree_net_rdma_carrier_create_params_t params = {
        .context = endpoint->connection_context,
        .proactor = state->proactor,
        .recv_pool = NULL,
        .connection_id = endpoint->connection_id,
        .callback = callback,
        .options = state->factory->carrier_options,
    };
    status = iree_net_rdma_carrier_create(params, state->host_allocator,
                                          &endpoint->carrier);
  }

  if (iree_status_is_ok(status)) {
    endpoint->connection_id = NULL;
    iree_net_rdma_carrier_t* rdma_carrier =
        iree_net_rdma_carrier_cast(endpoint->carrier);
    status = iree_net_rdma_factory_serialize_endpoint_data(
        rdma_carrier, state->group_id, endpoint->endpoint_index,
        state->endpoint_count,
        iree_make_byte_span(endpoint->bootstrap_data,
                            sizeof(endpoint->bootstrap_data)),
        &endpoint->bootstrap_data_length);
  }
  return status;
}

static iree_status_t iree_net_rdma_connect_state_apply_remote_endpoint_data(
    iree_net_rdma_connect_endpoint_t* endpoint,
    const iree_net_rdma_cm_event_t* event) {
  if (endpoint->remote_endpoint_data_applied) return iree_ok_status();
  if (event->private_data.data_length == 0) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "RDMA peer did not provide endpoint bootstrap data");
  }
  iree_net_rdma_connect_state_t* state = endpoint->state;
  iree_net_rdma_endpoint_data_t endpoint_data;
  IREE_RETURN_IF_ERROR(iree_net_rdma_endpoint_data_deserialize(
      event->private_data, &endpoint_data));
  IREE_RETURN_IF_ERROR(iree_net_rdma_factory_validate_endpoint_data(
      &endpoint_data, state->group_id, endpoint->endpoint_index,
      state->endpoint_count));

  endpoint->remote_endpoint_data = endpoint_data;
  endpoint->remote_endpoint_data_applied = true;
  return iree_ok_status();
}

static iree_status_t iree_net_rdma_connect_endpoint_maybe_complete_setup(
    iree_net_rdma_connect_endpoint_t* endpoint) {
  if (endpoint->setup_complete || !endpoint->remote_data_applied ||
      !endpoint->local_data_sent) {
    return iree_ok_status();
  }
  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_cast(endpoint->carrier);
  iree_status_t status =
      iree_net_rdma_carrier_complete_bootstrap_connection_data(carrier);
  if (iree_status_is_ok(status)) {
    endpoint->setup_complete = true;
    iree_net_rdma_connect_state_t* state = endpoint->state;
    ++state->setup_endpoint_count;
    if (state->setup_endpoint_count == state->endpoint_count) {
      iree_net_rdma_connect_state_succeed(state);
    }
  }
  return status;
}

static iree_status_t iree_net_rdma_connect_endpoint_on_setup_recv(
    void* user_data, iree_async_span_t data, iree_async_buffer_lease_t* lease) {
  (void)lease;
  iree_net_rdma_connect_endpoint_t* endpoint =
      (iree_net_rdma_connect_endpoint_t*)user_data;
  if (endpoint->remote_data_applied) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "duplicate RDMA connection-data setup message for connect endpoint");
  }

  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_cast(endpoint->carrier);
  iree_status_t status = iree_net_rdma_carrier_apply_remote_connection_data(
      carrier, iree_async_span_const_data(data));
  if (iree_status_is_ok(status)) {
    endpoint->remote_data_applied = true;
    status = iree_net_rdma_connect_endpoint_maybe_complete_setup(endpoint);
  }
  return status;
}

static void iree_net_rdma_connect_endpoint_carrier_completion(
    void* callback_user_data, iree_net_carrier_completion_kind_t kind,
    uint64_t operation_user_data, iree_status_t status,
    iree_host_size_t bytes_transferred, iree_async_buffer_lease_t* recv_lease) {
  (void)kind;
  (void)operation_user_data;
  (void)bytes_transferred;
  (void)recv_lease;
  iree_net_rdma_connect_endpoint_t* endpoint =
      (iree_net_rdma_connect_endpoint_t*)callback_user_data;
  iree_net_rdma_connect_state_t* state = endpoint->state;
  if (iree_status_is_ok(status)) return;
  if (state->callback_issued) {
    iree_status_abort(status);
    return;
  }
  iree_net_rdma_connect_state_fail(state, status);
}

static iree_status_t iree_net_rdma_connect_endpoint_start_setup(
    iree_net_rdma_connect_endpoint_t* endpoint) {
  if (!endpoint->remote_endpoint_data_applied) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "RDMA endpoint bootstrap data is missing");
  }
  iree_net_carrier_t* carrier = endpoint->carrier;
  carrier->callback.fn = iree_net_rdma_connect_endpoint_carrier_completion;
  carrier->callback.user_data = endpoint;
  iree_net_carrier_set_recv_handler(
      carrier, (iree_net_carrier_recv_handler_t){
                   .fn = iree_net_rdma_connect_endpoint_on_setup_recv,
                   .user_data = endpoint,
               });

  iree_status_t status = iree_net_carrier_activate(carrier);
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_send_bootstrap_connection_data(
        iree_net_rdma_carrier_cast(carrier),
        endpoint->remote_endpoint_data.bootstrap_recv_buffer_size,
        endpoint->remote_endpoint_data.bootstrap_recv_credits);
  }
  if (iree_status_is_ok(status)) {
    endpoint->local_data_sent = true;
    status = iree_net_rdma_connect_endpoint_maybe_complete_setup(endpoint);
  }
  return status;
}

static iree_status_t iree_net_rdma_connect_state_lookup_endpoint(
    iree_net_rdma_connect_state_t* state, const iree_net_rdma_cm_event_t* event,
    iree_net_rdma_connect_endpoint_t** out_endpoint) {
  *out_endpoint = NULL;
  if (!event || !event->id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "RDMA CM event must have a connection ID");
  }
  void* id_context = event->id->context;
  for (uint16_t i = 0; i < state->endpoint_count; ++i) {
    iree_net_rdma_connect_endpoint_t* endpoint = &state->endpoints[i];
    if (id_context == endpoint) {
      *out_endpoint = endpoint;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "RDMA CM event does not match a connect endpoint");
}

static void iree_net_rdma_connect_state_on_cm_event(
    void* user_data, iree_status_t status,
    const iree_net_rdma_cm_event_t* event) {
  iree_net_rdma_connect_state_t* state =
      (iree_net_rdma_connect_state_t*)user_data;
  if (state->callback_issued) {
    if (!iree_status_is_ok(status)) iree_status_abort(status);
    return;
  }
  if (!iree_status_is_ok(status)) {
    iree_net_rdma_connect_state_fail(state, status);
    return;
  }

  iree_net_rdma_connect_endpoint_t* endpoint = NULL;
  status = iree_net_rdma_connect_state_lookup_endpoint(state, event, &endpoint);
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
        status = iree_net_rdma_connect_state_create_carrier(endpoint);
      }
      if (iree_status_is_ok(status)) {
        endpoint->phase = IREE_NET_RDMA_CONNECT_PHASE_RESOLVING_ROUTE;
        int result = librdmacm->rdma_resolve_route(
            iree_net_rdma_carrier_connection_id(
                iree_net_rdma_carrier_cast(endpoint->carrier)),
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
            iree_make_const_byte_span(endpoint->bootstrap_data,
                                      endpoint->bootstrap_data_length),
            &conn_param);
      }
      if (iree_status_is_ok(status)) {
        endpoint->phase = IREE_NET_RDMA_CONNECT_PHASE_CONNECTING;
        int result = librdmacm->rdma_connect(
            iree_net_rdma_carrier_connection_id(
                iree_net_rdma_carrier_cast(endpoint->carrier)),
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
      status = iree_net_rdma_connect_state_apply_remote_endpoint_data(endpoint,
                                                                      event);
      if (!iree_status_is_ok(status)) {
        iree_net_rdma_connect_state_fail(state, status);
      }
      break;
    case RDMA_CM_EVENT_ESTABLISHED:
      if (endpoint->phase == IREE_NET_RDMA_CONNECT_PHASE_COMPLETE) {
        status = iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "duplicate RDMA ESTABLISHED event for connect endpoint");
      } else {
        status =
            iree_net_rdma_factory_status_from_cm_event(event, "rdma_connect");
      }
      if (iree_status_is_ok(status)) {
        status = iree_net_rdma_connect_state_apply_remote_endpoint_data(
            endpoint, event);
      }
      if (iree_status_is_ok(status)) {
        endpoint->phase = IREE_NET_RDMA_CONNECT_PHASE_COMPLETE;
        status = iree_net_rdma_connect_endpoint_start_setup(endpoint);
      }
      if (!iree_status_is_ok(status)) {
        iree_net_rdma_connect_state_fail(state, status);
      }
      break;
    case RDMA_CM_EVENT_REJECTED:
      iree_net_rdma_connect_state_fail(
          state, iree_net_rdma_factory_status_from_failed_cm_event(
                     event, "rdma_cm connect"));
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

typedef uint8_t iree_net_rdma_listener_flags_t;
enum iree_net_rdma_listener_flag_bits_e {
  IREE_NET_RDMA_LISTENER_FLAG_IN_CALLBACK = 1u << 0,
  IREE_NET_RDMA_LISTENER_FLAG_STOPPED_OPERATION_SUBMITTED = 1u << 1,
};

typedef struct iree_net_rdma_accept_state_t iree_net_rdma_accept_state_t;

typedef struct iree_net_rdma_listener_t {
  // Base listener; must be first for vtable dispatch.
  iree_net_listener_t base;

  // Factory retained while the listener is alive.
  iree_net_rdma_factory_t* factory;

  // Proactor used for CM events and stopped callbacks. Retained.
  iree_async_proactor_t* proactor;

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

  // Number of submitted accept-state cleanup operations still pending.
  uint32_t pending_cleanup_count;

  // Listener lifecycle state.
  iree_net_rdma_listener_state_t state;

  // Bitfield of iree_net_rdma_listener_flag_bits_e values.
  iree_net_rdma_listener_flags_t flags;

  // Host allocator used for this listener allocation.
  iree_allocator_t host_allocator;
} iree_net_rdma_listener_t;

typedef uint8_t iree_net_rdma_accept_endpoint_flags_t;
enum iree_net_rdma_accept_endpoint_flag_bits_e {
  IREE_NET_RDMA_ACCEPT_ENDPOINT_FLAG_REQUESTED = 1u << 0,
  IREE_NET_RDMA_ACCEPT_ENDPOINT_FLAG_ESTABLISHED = 1u << 1,
  IREE_NET_RDMA_ACCEPT_ENDPOINT_FLAG_REMOTE_DATA_APPLIED = 1u << 2,
  IREE_NET_RDMA_ACCEPT_ENDPOINT_FLAG_LOCAL_DATA_SENT = 1u << 3,
  IREE_NET_RDMA_ACCEPT_ENDPOINT_FLAG_SETUP_COMPLETE = 1u << 4,
};

typedef struct iree_net_rdma_accept_endpoint_t {
  // Owning accept state.
  iree_net_rdma_accept_state_t* accept_state;

  // Remote endpoint bootstrap data received in CONNECT_REQUEST.
  iree_net_rdma_endpoint_data_t remote_endpoint_data;

  // Raw connection ID before carrier creation transfers ownership.
  struct rdma_cm_id* connection_id;

  // Carrier created for the accepted endpoint.
  iree_net_carrier_t* carrier;

  // Child context matching connection_id->verbs.
  iree_net_rdma_context_t* connection_context;

  // Serialized local endpoint bootstrap data sent by rdma_accept.
  uint8_t bootstrap_data[IREE_NET_RDMA_ENDPOINT_DATA_LENGTH];

  // Number of bytes valid in bootstrap_data.
  iree_host_size_t bootstrap_data_length;

  // Bitfield of iree_net_rdma_accept_endpoint_flag_bits_e values.
  iree_net_rdma_accept_endpoint_flags_t flags;
} iree_net_rdma_accept_endpoint_t;

struct iree_net_rdma_accept_state_t {
  // Next pending accepted connection group in listener list.
  iree_net_rdma_accept_state_t* next;

  // Listener owning this accept state.
  iree_net_rdma_listener_t* listener;

  // Logical connection group identifier shared by endpoint QPs.
  uint64_t group_id;

  // Total endpoint slots expected in this logical connection.
  uint16_t endpoint_count;

  // Number of endpoint slots that have exchanged connection data.
  uint16_t setup_endpoint_count;

  // Cleanup operation submitted after the CM event callback returns.
  iree_async_nop_operation_t cleanup_operation;

  // Host allocator used for this state allocation.
  iree_allocator_t host_allocator;

  // Flexible array of accepted endpoint substates.
  iree_net_rdma_accept_endpoint_t endpoints[];
};

static const iree_net_listener_vtable_t iree_net_rdma_listener_vtable;

static void iree_net_rdma_listener_stopped_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_rdma_listener_t* listener = (iree_net_rdma_listener_t*)user_data;
  if (!iree_status_is_ok(status)) iree_status_abort(status);
  listener->state = IREE_NET_RDMA_LISTENER_STATE_STOPPED;
  listener->stopped_callback.fn(listener->stopped_callback.user_data);
}

static iree_status_t iree_net_rdma_listener_maybe_submit_stopped(
    iree_net_rdma_listener_t* listener) {
  if (listener->state != IREE_NET_RDMA_LISTENER_STATE_STOPPING ||
      iree_any_bit_set(listener->flags,
                       IREE_NET_RDMA_LISTENER_FLAG_STOPPED_OPERATION_SUBMITTED |
                           IREE_NET_RDMA_LISTENER_FLAG_IN_CALLBACK) ||
      listener->listen_id || listener->cm_channel ||
      listener->pending_accepts || listener->pending_cleanup_count != 0) {
    return iree_ok_status();
  }

  listener->flags |= IREE_NET_RDMA_LISTENER_FLAG_STOPPED_OPERATION_SUBMITTED;
  iree_async_operation_zero(&listener->stopped_operation.base,
                            sizeof(listener->stopped_operation));
  iree_async_operation_initialize(
      &listener->stopped_operation.base, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_rdma_listener_stopped_complete,
      listener);
  iree_status_t status = iree_async_proactor_submit_one(
      listener->proactor, &listener->stopped_operation.base);
  if (!iree_status_is_ok(status)) {
    listener->flags &= ~IREE_NET_RDMA_LISTENER_FLAG_STOPPED_OPERATION_SUBMITTED;
  }
  return status;
}

static void iree_net_rdma_accept_state_cleanup(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_rdma_accept_state_t* accept_state =
      (iree_net_rdma_accept_state_t*)user_data;
  if (!iree_status_is_ok(status)) iree_status_abort(status);

  iree_net_rdma_listener_t* listener = accept_state->listener;
  const iree_net_librdmacm_t* librdmacm =
      iree_net_rdma_context_librdmacm(listener->factory->context);
  for (uint16_t i = 0; i < accept_state->endpoint_count; ++i) {
    iree_net_rdma_accept_endpoint_t* endpoint = &accept_state->endpoints[i];
    iree_net_carrier_release(endpoint->carrier);
    iree_net_rdma_context_release(endpoint->connection_context);
    if (endpoint->connection_id) {
      int result = librdmacm->rdma_destroy_id(endpoint->connection_id);
      iree_status_t destroy_status = iree_net_rdma_factory_status_from_result(
          __FILE__, __LINE__, result, "rdma_destroy_id");
      if (!iree_status_is_ok(destroy_status)) iree_status_abort(destroy_status);
    }
  }
  iree_allocator_t host_allocator = accept_state->host_allocator;
  iree_allocator_free(host_allocator, accept_state);

  if (listener->pending_cleanup_count == 0) {
    iree_status_abort(iree_make_status(
        IREE_STATUS_INTERNAL, "RDMA listener accept cleanup count underflow"));
  }
  --listener->pending_cleanup_count;
  iree_status_t stopped_status =
      iree_net_rdma_listener_maybe_submit_stopped(listener);
  if (!iree_status_is_ok(stopped_status)) iree_status_abort(stopped_status);
}

static void iree_net_rdma_accept_state_submit_cleanup(
    iree_net_rdma_accept_state_t* accept_state) {
  ++accept_state->listener->pending_cleanup_count;
  iree_async_operation_zero(&accept_state->cleanup_operation.base,
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

static iree_net_rdma_accept_state_t* iree_net_rdma_listener_find_accept(
    iree_net_rdma_listener_t* listener, uint64_t group_id) {
  iree_net_rdma_accept_state_t* accept_state = listener->pending_accepts;
  while (accept_state) {
    if (accept_state->group_id == group_id) return accept_state;
    accept_state = accept_state->next;
  }
  return NULL;
}

static iree_net_rdma_accept_endpoint_t*
iree_net_rdma_accept_state_find_endpoint_by_id(
    iree_net_rdma_accept_state_t* accept_state, struct rdma_cm_id* id) {
  for (uint16_t i = 0; i < accept_state->endpoint_count; ++i) {
    iree_net_rdma_accept_endpoint_t* endpoint = &accept_state->endpoints[i];
    iree_net_rdma_carrier_t* carrier =
        iree_net_rdma_carrier_cast(endpoint->carrier);
    if ((carrier && iree_net_rdma_carrier_connection_id(carrier) == id) ||
        endpoint->connection_id == id) {
      return endpoint;
    }
  }
  return NULL;
}

static iree_net_rdma_accept_state_t* iree_net_rdma_listener_pop_accept_by_group(
    iree_net_rdma_listener_t* listener, uint64_t group_id) {
  iree_net_rdma_accept_state_t** link = &listener->pending_accepts;
  while (*link) {
    iree_net_rdma_accept_state_t* accept_state = *link;
    if (accept_state->group_id == group_id) {
      *link = accept_state->next;
      accept_state->next = NULL;
      return accept_state;
    }
    link = &accept_state->next;
  }
  return NULL;
}

static iree_net_rdma_accept_state_t* iree_net_rdma_listener_find_accept_by_id(
    iree_net_rdma_listener_t* listener, struct rdma_cm_id* id,
    iree_net_rdma_accept_endpoint_t** out_endpoint) {
  if (out_endpoint) *out_endpoint = NULL;
  iree_net_rdma_accept_state_t* accept_state = listener->pending_accepts;
  while (accept_state) {
    iree_net_rdma_accept_endpoint_t* endpoint =
        iree_net_rdma_accept_state_find_endpoint_by_id(accept_state, id);
    if (endpoint) {
      if (out_endpoint) *out_endpoint = endpoint;
      return accept_state;
    }
    accept_state = accept_state->next;
  }
  return NULL;
}

static iree_net_rdma_accept_state_t* iree_net_rdma_listener_pop_accept_by_id(
    iree_net_rdma_listener_t* listener, struct rdma_cm_id* id,
    iree_net_rdma_accept_endpoint_t** out_endpoint) {
  if (out_endpoint) *out_endpoint = NULL;
  iree_net_rdma_accept_state_t** link = &listener->pending_accepts;
  while (*link) {
    iree_net_rdma_accept_state_t* accept_state = *link;
    iree_net_rdma_accept_endpoint_t* endpoint =
        iree_net_rdma_accept_state_find_endpoint_by_id(accept_state, id);
    if (endpoint) {
      *link = accept_state->next;
      accept_state->next = NULL;
      if (out_endpoint) *out_endpoint = endpoint;
      return accept_state;
    }
    link = &accept_state->next;
  }
  return NULL;
}

static void iree_net_rdma_listener_cancel_pending_accepts(
    iree_net_rdma_listener_t* listener) {
  iree_net_rdma_accept_state_t* accept_state = listener->pending_accepts;
  listener->pending_accepts = NULL;
  while (accept_state) {
    iree_net_rdma_accept_state_t* next_accept_state = accept_state->next;
    accept_state->next = NULL;
    iree_net_rdma_accept_state_submit_cleanup(accept_state);
    accept_state = next_accept_state;
  }
}

static void iree_net_rdma_listener_deliver_accept(
    iree_net_rdma_listener_t* listener,
    iree_net_rdma_accept_state_t* accept_state) {
  iree_net_connection_t* connection = NULL;
  iree_net_carrier_t** carriers = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      accept_state->host_allocator, accept_state->endpoint_count,
      sizeof(*carriers), (void**)&carriers);
  if (iree_status_is_ok(status)) {
    for (uint16_t i = 0; i < accept_state->endpoint_count; ++i) {
      carriers[i] = accept_state->endpoints[i].carrier;
    }
    status = iree_net_rdma_connection_create(
        listener->proactor, accept_state->endpoint_count, carriers,
        listener->cm_channel, /*install_cm_callback=*/false,
        accept_state->host_allocator, &connection);
  }
  if (iree_status_is_ok(status)) {
    for (uint16_t i = 0; i < accept_state->endpoint_count; ++i) {
      accept_state->endpoints[i].carrier = carriers[i];
    }
    listener->accept_callback(listener->accept_user_data, iree_ok_status(),
                              connection);
  } else {
    listener->accept_callback(listener->accept_user_data, status, NULL);
  }
  iree_allocator_free(accept_state->host_allocator, carriers);
  iree_net_rdma_accept_state_submit_cleanup(accept_state);
}

static void iree_net_rdma_listener_fail_pending_accept(
    iree_net_rdma_listener_t* listener,
    iree_net_rdma_accept_state_t* accept_state, iree_status_t status) {
  iree_net_rdma_accept_state_t* popped_accept_state =
      iree_net_rdma_listener_pop_accept_by_group(listener,
                                                 accept_state->group_id);
  if (popped_accept_state) {
    listener->accept_callback(listener->accept_user_data, status, NULL);
    iree_net_rdma_accept_state_submit_cleanup(popped_accept_state);
  } else if (!iree_status_is_ok(status)) {
    iree_status_abort(status);
  }
}

static iree_status_t iree_net_rdma_accept_endpoint_maybe_complete_setup(
    iree_net_rdma_accept_endpoint_t* endpoint) {
  if (iree_any_bit_set(endpoint->flags,
                       IREE_NET_RDMA_ACCEPT_ENDPOINT_FLAG_SETUP_COMPLETE) ||
      !iree_all_bits_set(
          endpoint->flags,
          IREE_NET_RDMA_ACCEPT_ENDPOINT_FLAG_REMOTE_DATA_APPLIED |
              IREE_NET_RDMA_ACCEPT_ENDPOINT_FLAG_LOCAL_DATA_SENT)) {
    return iree_ok_status();
  }
  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_cast(endpoint->carrier);
  iree_status_t status =
      iree_net_rdma_carrier_complete_bootstrap_connection_data(carrier);
  if (iree_status_is_ok(status)) {
    endpoint->flags |= IREE_NET_RDMA_ACCEPT_ENDPOINT_FLAG_SETUP_COMPLETE;
    iree_net_rdma_accept_state_t* accept_state = endpoint->accept_state;
    ++accept_state->setup_endpoint_count;
    if (accept_state->setup_endpoint_count == accept_state->endpoint_count) {
      iree_net_rdma_listener_t* listener = accept_state->listener;
      iree_net_rdma_accept_state_t* popped_accept_state =
          iree_net_rdma_listener_pop_accept_by_group(listener,
                                                     accept_state->group_id);
      if (popped_accept_state) {
        iree_net_rdma_listener_deliver_accept(listener, popped_accept_state);
      } else {
        status =
            iree_make_status(IREE_STATUS_NOT_FOUND,
                             "RDMA setup endpoint group was already removed");
      }
    }
  }
  return status;
}

static iree_status_t iree_net_rdma_accept_endpoint_on_setup_recv(
    void* user_data, iree_async_span_t data, iree_async_buffer_lease_t* lease) {
  (void)lease;
  iree_net_rdma_accept_endpoint_t* endpoint =
      (iree_net_rdma_accept_endpoint_t*)user_data;
  if (iree_any_bit_set(
          endpoint->flags,
          IREE_NET_RDMA_ACCEPT_ENDPOINT_FLAG_REMOTE_DATA_APPLIED)) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "duplicate RDMA connection-data setup message for accepted endpoint");
  }

  iree_net_rdma_carrier_t* carrier =
      iree_net_rdma_carrier_cast(endpoint->carrier);
  iree_status_t status = iree_net_rdma_carrier_apply_remote_connection_data(
      carrier, iree_async_span_const_data(data));
  if (iree_status_is_ok(status)) {
    endpoint->flags |= IREE_NET_RDMA_ACCEPT_ENDPOINT_FLAG_REMOTE_DATA_APPLIED;
    status = iree_net_rdma_accept_endpoint_maybe_complete_setup(endpoint);
  }
  return status;
}

static void iree_net_rdma_accept_endpoint_carrier_completion(
    void* callback_user_data, iree_net_carrier_completion_kind_t kind,
    uint64_t operation_user_data, iree_status_t status,
    iree_host_size_t bytes_transferred, iree_async_buffer_lease_t* recv_lease) {
  (void)kind;
  (void)operation_user_data;
  (void)bytes_transferred;
  (void)recv_lease;
  if (iree_status_is_ok(status)) return;
  iree_net_rdma_accept_endpoint_t* endpoint =
      (iree_net_rdma_accept_endpoint_t*)callback_user_data;
  iree_net_rdma_accept_state_t* accept_state = endpoint->accept_state;
  iree_net_rdma_listener_fail_pending_accept(accept_state->listener,
                                             accept_state, status);
}

static iree_status_t iree_net_rdma_accept_endpoint_start_setup(
    iree_net_rdma_accept_endpoint_t* endpoint) {
  iree_net_carrier_t* carrier = endpoint->carrier;
  carrier->callback.fn = iree_net_rdma_accept_endpoint_carrier_completion;
  carrier->callback.user_data = endpoint;
  iree_net_carrier_set_recv_handler(
      carrier, (iree_net_carrier_recv_handler_t){
                   .fn = iree_net_rdma_accept_endpoint_on_setup_recv,
                   .user_data = endpoint,
               });

  iree_status_t status = iree_net_carrier_activate(carrier);
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_carrier_send_bootstrap_connection_data(
        iree_net_rdma_carrier_cast(carrier),
        endpoint->remote_endpoint_data.bootstrap_recv_buffer_size,
        endpoint->remote_endpoint_data.bootstrap_recv_credits);
  }
  if (iree_status_is_ok(status)) {
    endpoint->flags |= IREE_NET_RDMA_ACCEPT_ENDPOINT_FLAG_LOCAL_DATA_SENT;
    status = iree_net_rdma_accept_endpoint_maybe_complete_setup(endpoint);
  }
  return status;
}

static iree_status_t iree_net_rdma_accept_state_create(
    iree_net_rdma_listener_t* listener, uint64_t group_id,
    uint16_t endpoint_count, iree_net_rdma_accept_state_t** out_accept_state) {
  *out_accept_state = NULL;

  iree_host_size_t total_size = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_net_rdma_accept_state_t), &total_size,
      IREE_STRUCT_FIELD_FAM(endpoint_count, iree_net_rdma_accept_endpoint_t));

  iree_net_rdma_accept_state_t* accept_state = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(listener->host_allocator, total_size,
                                   (void**)&accept_state);
  }
  if (iree_status_is_ok(status)) {
    memset(accept_state, 0, total_size);
    accept_state->listener = listener;
    accept_state->group_id = group_id;
    accept_state->endpoint_count = endpoint_count;
    accept_state->host_allocator = listener->host_allocator;
    for (uint16_t i = 0; i < endpoint_count; ++i) {
      accept_state->endpoints[i].accept_state = accept_state;
    }
    *out_accept_state = accept_state;
  }
  return status;
}

static iree_status_t iree_net_rdma_accept_state_prepare_endpoint(
    iree_net_rdma_accept_state_t* accept_state,
    const iree_net_rdma_cm_event_t* event,
    const iree_net_rdma_endpoint_data_t* remote_endpoint_data) {
  iree_net_rdma_listener_t* listener = accept_state->listener;
  iree_net_rdma_accept_endpoint_t* endpoint =
      &accept_state->endpoints[remote_endpoint_data->endpoint_index];
  endpoint->remote_endpoint_data = *remote_endpoint_data;
  endpoint->connection_id = event->id;

  iree_status_t status = iree_net_rdma_context_create_for_cm_id(
      listener->factory->context, endpoint->connection_id,
      listener->host_allocator, &endpoint->connection_context);

  if (iree_status_is_ok(status)) {
    iree_net_carrier_callback_t callback = {
        .fn = iree_net_rdma_connection_carrier_completion,
        .user_data = NULL,
    };
    iree_net_rdma_carrier_create_params_t params = {
        .context = endpoint->connection_context,
        .proactor = listener->proactor,
        .recv_pool = NULL,
        .connection_id = endpoint->connection_id,
        .callback = callback,
        .options = listener->factory->carrier_options,
    };
    status = iree_net_rdma_carrier_create(params, listener->host_allocator,
                                          &endpoint->carrier);
  }

  if (iree_status_is_ok(status)) {
    endpoint->connection_id = NULL;
    iree_net_rdma_carrier_t* carrier =
        iree_net_rdma_carrier_cast(endpoint->carrier);
    status = iree_net_rdma_factory_serialize_endpoint_data(
        carrier, accept_state->group_id, remote_endpoint_data->endpoint_index,
        accept_state->endpoint_count,
        iree_make_byte_span(endpoint->bootstrap_data,
                            sizeof(endpoint->bootstrap_data)),
        &endpoint->bootstrap_data_length);
  }

  return status;
}

static iree_status_t iree_net_rdma_listener_accept_request(
    iree_net_rdma_listener_t* listener, const iree_net_rdma_cm_event_t* event) {
  iree_status_t status = iree_ok_status();
  if (event->private_data.data_length == 0) {
    status =
        iree_make_status(IREE_STATUS_DATA_LOSS,
                         "RDMA peer did not provide endpoint bootstrap data");
  }

  iree_net_rdma_endpoint_data_t remote_endpoint_data;
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_endpoint_data_deserialize(event->private_data,
                                                     &remote_endpoint_data);
  }
  if (iree_status_is_ok(status) && remote_endpoint_data.endpoint_count >
                                       listener->factory->max_endpoint_count) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "RDMA endpoint_count %" PRIu16
                              " exceeds listener capacity %" PRIu32,
                              remote_endpoint_data.endpoint_count,
                              listener->factory->max_endpoint_count);
  }

  iree_net_rdma_accept_state_t* accept_state = NULL;
  if (iree_status_is_ok(status)) {
    accept_state = iree_net_rdma_listener_find_accept(
        listener, remote_endpoint_data.group_id);
    if (accept_state &&
        accept_state->endpoint_count != remote_endpoint_data.endpoint_count) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "RDMA endpoint_count changed within connection group");
    }
  }

  if (iree_status_is_ok(status) && !accept_state) {
    status = iree_net_rdma_accept_state_create(
        listener, remote_endpoint_data.group_id,
        remote_endpoint_data.endpoint_count, &accept_state);
    if (iree_status_is_ok(status)) {
      iree_net_rdma_listener_push_accept(listener, accept_state);
    }
  }

  iree_net_rdma_accept_endpoint_t* endpoint = NULL;
  bool event_id_adopted = false;
  if (iree_status_is_ok(status)) {
    endpoint = &accept_state->endpoints[remote_endpoint_data.endpoint_index];
    if (iree_any_bit_set(endpoint->flags,
                         IREE_NET_RDMA_ACCEPT_ENDPOINT_FLAG_REQUESTED)) {
      status = iree_make_status(
          IREE_STATUS_ALREADY_EXISTS,
          "duplicate RDMA CONNECT_REQUEST for endpoint_index %" PRIu16,
          remote_endpoint_data.endpoint_index);
    }
  }

  if (iree_status_is_ok(status)) {
    event_id_adopted = true;
    status = iree_net_rdma_accept_state_prepare_endpoint(accept_state, event,
                                                         &remote_endpoint_data);
  }

  struct rdma_conn_param conn_param;
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_factory_initialize_conn_param(
        iree_make_const_byte_span(endpoint->bootstrap_data,
                                  endpoint->bootstrap_data_length),
        &conn_param);
  }

  if (iree_status_is_ok(status)) {
    const iree_net_librdmacm_t* librdmacm =
        iree_net_rdma_context_librdmacm(listener->factory->context);
    int result = librdmacm->rdma_accept(
        iree_net_rdma_carrier_connection_id(
            iree_net_rdma_carrier_cast(endpoint->carrier)),
        &conn_param);
    status = iree_net_rdma_factory_status_from_result(__FILE__, __LINE__,
                                                      result, "rdma_accept");
  }

  if (iree_status_is_ok(status)) {
    endpoint->flags |= IREE_NET_RDMA_ACCEPT_ENDPOINT_FLAG_REQUESTED;
  } else {
    const iree_net_librdmacm_t* librdmacm =
        iree_net_rdma_context_librdmacm(listener->factory->context);
    int result = librdmacm->rdma_reject(event->id, NULL, 0);
    status = iree_status_join(status,
                              iree_net_rdma_factory_status_from_result(
                                  __FILE__, __LINE__, result, "rdma_reject"));
    if (accept_state) {
      iree_net_rdma_accept_state_t* popped_accept_state =
          iree_net_rdma_listener_pop_accept_by_group(listener,
                                                     accept_state->group_id);
      if (popped_accept_state) {
        iree_net_rdma_accept_state_submit_cleanup(popped_accept_state);
      }
    }
    if (!event_id_adopted) {
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
  if (!event) return status;

  switch (event->type) {
    case RDMA_CM_EVENT_CONNECT_REQUEST: {
      const iree_net_librdmacm_t* librdmacm =
          iree_net_rdma_context_librdmacm(listener->factory->context);
      int result = librdmacm->rdma_reject(event->id, NULL, 0);
      status = iree_net_rdma_factory_status_from_result(__FILE__, __LINE__,
                                                        result, "rdma_reject");
      result = librdmacm->rdma_destroy_id(event->id);
      status = iree_status_join(
          status, iree_net_rdma_factory_status_from_result(
                      __FILE__, __LINE__, result, "rdma_destroy_id"));
      break;
    }
    case RDMA_CM_EVENT_ESTABLISHED:
    case RDMA_CM_EVENT_REJECTED:
    case RDMA_CM_EVENT_CONNECT_ERROR:
    case RDMA_CM_EVENT_UNREACHABLE:
    case RDMA_CM_EVENT_DISCONNECTED:
    case RDMA_CM_EVENT_DEVICE_REMOVAL: {
      iree_net_rdma_accept_state_t* accept_state =
          iree_net_rdma_listener_pop_accept_by_id(listener, event->id, NULL);
      if (accept_state) {
        iree_net_rdma_accept_state_submit_cleanup(accept_state);
      }
      break;
    }
    default:
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "unexpected RDMA CM event %d while stopping",
                                (int)event->type);
      break;
  }
  return status;
}

static iree_status_t iree_net_rdma_listener_drain_stopping_events(
    iree_net_rdma_listener_t* listener) {
  iree_status_t status = iree_ok_status();
  if (listener->cm_channel) {
    status = iree_net_rdma_cm_channel_drain(listener->cm_channel);
  }
  return status;
}

static iree_status_t iree_net_rdma_listener_destroy_listen_id(
    iree_net_rdma_listener_t* listener) {
  iree_status_t status = iree_ok_status();
  if (listener->listen_id) {
    const iree_net_librdmacm_t* librdmacm =
        iree_net_rdma_context_librdmacm(listener->factory->context);
    struct rdma_cm_id* listen_id = listener->listen_id;
    int result = librdmacm->rdma_destroy_id(listen_id);
    status = iree_net_rdma_factory_status_from_result(
        __FILE__, __LINE__, result, "rdma_destroy_id");
    if (iree_status_is_ok(status)) listener->listen_id = NULL;
  }
  return status;
}

static void iree_net_rdma_listener_on_cm_event(
    void* user_data, iree_status_t status,
    const iree_net_rdma_cm_event_t* event) {
  iree_net_rdma_listener_t* listener = (iree_net_rdma_listener_t*)user_data;
  listener->flags |= IREE_NET_RDMA_LISTENER_FLAG_IN_CALLBACK;
  if (listener->state != IREE_NET_RDMA_LISTENER_STATE_LISTENING) {
    if (!iree_status_is_ok(status)) iree_status_abort(status);
    status = iree_net_rdma_listener_discard_stopping_event(listener, event);
    if (!iree_status_is_ok(status)) iree_status_abort(status);
    listener->flags &= ~IREE_NET_RDMA_LISTENER_FLAG_IN_CALLBACK;
    return;
  }
  if (!iree_status_is_ok(status)) {
    listener->accept_callback(listener->accept_user_data, status, NULL);
    listener->flags &= ~IREE_NET_RDMA_LISTENER_FLAG_IN_CALLBACK;
    return;
  }

  switch (event->type) {
    case RDMA_CM_EVENT_CONNECT_REQUEST: {
      status = iree_net_rdma_listener_accept_request(listener, event);
      if (!iree_status_is_ok(status)) iree_status_abort(status);
      break;
    }
    case RDMA_CM_EVENT_ESTABLISHED: {
      iree_net_rdma_accept_endpoint_t* endpoint = NULL;
      iree_net_rdma_accept_state_t* accept_state =
          iree_net_rdma_listener_find_accept_by_id(listener, event->id,
                                                   &endpoint);
      if (accept_state && endpoint) {
        if (iree_any_bit_set(endpoint->flags,
                             IREE_NET_RDMA_ACCEPT_ENDPOINT_FLAG_ESTABLISHED)) {
          accept_state = iree_net_rdma_listener_pop_accept_by_group(
              listener, accept_state->group_id);
          if (accept_state) {
            iree_net_rdma_accept_state_submit_cleanup(accept_state);
          }
          listener->accept_callback(
              listener->accept_user_data,
              iree_make_status(
                  IREE_STATUS_ALREADY_EXISTS,
                  "duplicate RDMA ESTABLISHED event for accepted endpoint"),
              NULL);
        } else {
          endpoint->flags |= IREE_NET_RDMA_ACCEPT_ENDPOINT_FLAG_ESTABLISHED;
          status = iree_net_rdma_accept_endpoint_start_setup(endpoint);
          if (!iree_status_is_ok(status)) {
            iree_net_rdma_listener_fail_pending_accept(listener, accept_state,
                                                       status);
          }
        }
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
          iree_net_rdma_listener_pop_accept_by_id(listener, event->id, NULL);
      if (accept_state) {
        iree_net_rdma_accept_state_submit_cleanup(accept_state);
        listener->accept_callback(
            listener->accept_user_data,
            iree_net_rdma_factory_status_from_failed_cm_event(event,
                                                              "rdma_cm accept"),
            NULL);
      }
      break;
    }
    case RDMA_CM_EVENT_TIMEWAIT_EXIT:
      break;
    default:
      listener->accept_callback(
          listener->accept_user_data,
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "unexpected RDMA CM event %d on listener",
                           (int)event->type),
          NULL);
      break;
  }
  listener->flags &= ~IREE_NET_RDMA_LISTENER_FLAG_IN_CALLBACK;
}

static iree_status_t iree_net_rdma_listener_begin_stop(
    iree_net_rdma_listener_t* listener) {
  iree_status_t status = iree_net_rdma_listener_destroy_listen_id(listener);
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_listener_drain_stopping_events(listener);
  }
  if (iree_status_is_ok(status)) {
    iree_net_rdma_listener_cancel_pending_accepts(listener);
    iree_net_rdma_cm_channel_set_callback(
        listener->cm_channel, (iree_net_rdma_cm_channel_callback_t){
                                  .fn = iree_net_rdma_discard_cm_event,
                                  .user_data = NULL,
                              });
    iree_net_rdma_cm_channel_release(listener->cm_channel);
    listener->cm_channel = NULL;
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_listener_maybe_submit_stopped(listener);
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

  listener->state = IREE_NET_RDMA_LISTENER_STATE_STOPPING;
  listener->stopped_callback = callback;
  iree_status_t status = iree_ok_status();
  if (iree_all_bits_set(listener->flags,
                        IREE_NET_RDMA_LISTENER_FLAG_IN_CALLBACK)) {
    iree_async_operation_zero(&listener->stop_operation.base,
                              sizeof(listener->stop_operation));
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
  if (options.max_endpoint_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "max_endpoint_count must fit in uint16_t");
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
  (void)recv_pool;
  iree_net_rdma_factory_t* factory = (iree_net_rdma_factory_t*)base_factory;
  uint16_t endpoint_count = (uint16_t)factory->max_endpoint_count;

  iree_async_address_t remote_address;
  iree_status_t status =
      iree_async_address_from_string(address, &remote_address);

  iree_host_size_t total_size = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_net_rdma_connect_state_t), &total_size,
        IREE_STRUCT_FIELD_FAM(endpoint_count,
                              iree_net_rdma_connect_endpoint_t));
  }

  iree_net_rdma_connect_state_t* state = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(factory->host_allocator, total_size,
                                   (void**)&state);
  }

  if (iree_status_is_ok(status)) {
    memset(state, 0, total_size);
    state->factory = factory;
    iree_net_transport_factory_retain(base_factory);
    state->proactor = proactor;
    iree_async_proactor_retain(proactor);
    state->callback = callback;
    state->callback_user_data = user_data;
    state->endpoint_count = endpoint_count;
    state->host_allocator = factory->host_allocator;
    for (uint16_t i = 0; i < endpoint_count; ++i) {
      state->endpoints[i].state = state;
      state->endpoints[i].endpoint_index = i;
      state->endpoints[i].phase = IREE_NET_RDMA_CONNECT_PHASE_RESOLVING_ADDRESS;
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_factory_generate_group_id(&state->group_id);
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

  if (iree_status_is_ok(status)) {
    const iree_net_librdmacm_t* librdmacm =
        iree_net_rdma_context_librdmacm(factory->context);
    struct rdma_event_channel* event_channel =
        iree_net_rdma_cm_channel_native_event_channel(state->cm_channel);
    for (uint16_t i = 0; i < endpoint_count && iree_status_is_ok(status); ++i) {
      iree_net_rdma_connect_endpoint_t* endpoint = &state->endpoints[i];
      int result = librdmacm->rdma_create_id(
          event_channel, &endpoint->connection_id, endpoint, RDMA_PS_TCP);
      status = iree_net_rdma_factory_status_from_result(
          __FILE__, __LINE__, result, "rdma_create_id");
      if (iree_status_is_ok(status)) {
        result = librdmacm->rdma_resolve_addr(
            endpoint->connection_id, NULL,
            (struct sockaddr*)remote_address.storage,
            factory->resolve_timeout_ms);
        status = iree_net_rdma_factory_status_from_result(
            __FILE__, __LINE__, result, "rdma_resolve_addr");
      }
    }
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
  (void)recv_pool;
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
