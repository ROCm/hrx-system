// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/connection.h"

#include <limits.h>

#include "iree/async/operations/scheduling.h"
#include "iree/base/alignment.h"
#include "iree/net/framed_endpoint.h"
#include "iree/net/rdma/target.h"

#define IREE_NET_RDMA_HELLO_MAGIC UINT32_C(0x414d4452)
#define IREE_NET_RDMA_PROTOCOL_VERSION 1u
#define IREE_NET_RDMA_HELLO_SIZE 16u

typedef enum iree_net_rdma_endpoint_kind_e {
  IREE_NET_RDMA_ENDPOINT_NONE = 0,
  IREE_NET_RDMA_ENDPOINT_MESSAGE = 1,
  IREE_NET_RDMA_ENDPOINT_DIRECT = 2,
} iree_net_rdma_endpoint_kind_t;

// OPEN record kinds match endpoint kinds; CREDIT carries no endpoint setup.
#define IREE_NET_RDMA_RECORD_CREDIT 3u

typedef enum iree_net_rdma_connection_state_e {
  IREE_NET_RDMA_CONNECTION_CREATED = 0,
  IREE_NET_RDMA_CONNECTION_CONNECTING,
  IREE_NET_RDMA_CONNECTION_OPEN,
  IREE_NET_RDMA_CONNECTION_DRAINING,
  IREE_NET_RDMA_CONNECTION_DEACTIVATED,
} iree_net_rdma_connection_state_t;

enum iree_net_rdma_connection_flag_bits_e {
  IREE_NET_RDMA_CONNECTION_PROGRESS_QUEUED = 1u << 0,
  IREE_NET_RDMA_CONNECTION_CONTROL_STARTED = 1u << 1,
  IREE_NET_RDMA_CONNECTION_CONTROL_READY = 1u << 2,
  IREE_NET_RDMA_CONNECTION_ENDPOINTS_DRAINING = 1u << 3,
  IREE_NET_RDMA_CONNECTION_ENDPOINTS_DRAINED = 1u << 4,
  IREE_NET_RDMA_CONNECTION_CONTROL_DRAINING = 1u << 5,
  IREE_NET_RDMA_CONNECTION_CONTROL_DRAINED = 1u << 6,
};
typedef uint32_t iree_net_rdma_connection_flags_t;

enum iree_net_rdma_endpoint_flag_bits_e {
  IREE_NET_RDMA_ENDPOINT_OPEN_SENT = 1u << 0,
  IREE_NET_RDMA_ENDPOINT_CONNECTED = 1u << 1,
  IREE_NET_RDMA_ENDPOINT_CREDIT_PENDING = 1u << 2,
};
typedef uint32_t iree_net_rdma_endpoint_flags_t;

typedef struct iree_net_rdma_connection_endpoint_t {
  // Containing owner retained through native completion service retirement.
  iree_net_rdma_connection_t* connection;
  // Claimed once under the connection mutex; NONE means locally unopened.
  iree_net_rdma_endpoint_kind_t kind;
  // Accepted callback, protected by the mutex and cleared before delivery.
  union {
    // Message readiness for MESSAGE kind.
    iree_net_endpoint_ready_callback_t message;
    // Registered-placement readiness for DIRECT kind.
    iree_net_direct_endpoint_ready_callback_t direct;
  } ready;
  // Poll-owner-only setup and cumulative-publication obligations.
  iree_net_rdma_endpoint_flags_t flags;
  // Native owner, owned directly or borrowed from carrier below.
  iree_net_rdma_direct_endpoint_t* native;
  // Optional framing owner for MESSAGE kind; owns carrier and native.
  iree_net_framed_endpoint_t* framed;
  // Optional carrier borrowed from framed for failure and setup integration.
  iree_net_carrier_t* carrier;
  // Latest actual notification receive grant; coalesced while control is full.
  uint64_t credit;
  // Peer facts captured before or after the corresponding local open call.
  struct {
    // NONE until the unique peer OPEN arrives.
    iree_net_rdma_endpoint_kind_t kind;
    // Native 24-bit destination QPN.
    uint32_t queue_number;
    // Native 24-bit initial receive PSN.
    uint32_t sequence_number;
    // Peer's enforced notification receive window.
    uint32_t receive_count;
    // MESSAGE receive slot stride; absent for DIRECT kind.
    uint32_t chunk_capacity;
    // MESSAGE receive target description; absent for DIRECT kind.
    uint8_t target[IREE_NET_RDMA_TARGET_WIRE_SIZE];
  } peer;
} iree_net_rdma_connection_endpoint_t;

struct iree_net_rdma_connection_t {
  // Public base, with negotiated capacity fixed before publication.
  iree_net_connection_t base;
  // Retained native device/registration compatibility owner.
  iree_net_rdma_context_t* context;
  // Retained poll owner for setup, native progress and result callbacks.
  iree_async_proactor_t* proactor;
  // Immutable local geometry; bounds the trailing ordinal slots.
  iree_net_rdma_connection_options_t options;
  // Protects admission, cancellation, failure and owner handoff accounting.
  iree_slim_mutex_t mutex;
  // Public lifecycle, separate from failure-driven native retirement.
  iree_net_rdma_connection_state_t state;
  // Monotonic native lifecycle and coalesced dispatch obligations.
  iree_net_rdma_connection_flags_t flags;
  // Owned terminal failure, joined under mutex until final release.
  iree_status_t failure;
  // Number of monotonically claimed local ordinals, protected by mutex.
  uint32_t opened_count;
  // Queued/executing owner callbacks, including user callback bodies.
  uint32_t pending_progress_count;
  // Coalesced resource-free handoff for setup, opens, credits and retirement.
  iree_async_nop_operation_t progress;
  // Shared CM/control/CQ owner; endpoint metadata outlives its final join.
  iree_net_rdma_connection_control_t* control;
  // Peer native message limit, checked before endpoint creation.
  uint32_t peer_max_request_length;
  // One unpublished attempt, detached before its terminal callback.
  struct {
    // Optional outbound cancellation owner; lock order is operation then mutex.
    iree_net_transport_connect_operation_t* operation;
    // Captured outbound numeric address, unused for accepted attempts.
    iree_async_address_t address;
    // Transfers the initial connection reference on success.
    iree_net_transport_connect_callback_t callback;
  } setup;
  // Public drain observer; the active reference survives its callback body.
  iree_net_connection_deactivate_callback_t deactivated_callback;
  // Shared endpoint consumer/connection drain, including off-thread callers.
  iree_net_endpoint_deactivation_barrier_t endpoint_barrier;
  // Fixed protocol slots; endpoint resources are constructed only on open.
  iree_net_rdma_connection_endpoint_t endpoints[];
};

iree_net_rdma_connection_options_t iree_net_rdma_connection_options_default(
    void) {
  iree_net_rdma_connection_options_t options = {0};
  options.max_endpoint_count = 4;
  options.completion_mode = IREE_NET_RDMA_COMPLETION_QUEUE_MODE_READINESS;
  options.control.send_count = 16;
  options.control.receive_count = 16;
  options.control.service_batch_size = 32;
  options.control.resolution_timeout_ms = 10000;
  options.control.minimum_rnr_timer = 1;
  options.direct = (iree_net_rdma_direct_endpoint_options_t){
      .max_write_operations = 64,
      .max_write_entries = 256,
      .send_work_count = 128,
      .receive_work_count = 128,
      .post_batch_size = 32,
      .max_request_length = UINT32_MAX,
      .minimum_rnr_timer = 1,
  };
  options.message.direct = options.direct;
  options.message.direct.max_write_operations = 16;
  options.message.direct.max_write_entries = 1;
  options.message.direct.receive_work_count = 32;
  options.message.carrier = (iree_net_rdma_carrier_options_t){
      .max_send_operations = 64,
      .generated_prefix_capacity = 256,
      .chunk_capacity = 64 * 1024,
  };
  return options;
}

iree_status_t iree_net_rdma_connection_options_validate(
    const iree_net_rdma_connection_options_t* options) {
  if (!options || !options->max_endpoint_count ||
      options->max_endpoint_count >= (UINT32_C(1) << 30) ||
      options->completion_mode >
          IREE_NET_RDMA_COMPLETION_QUEUE_MODE_BUSY_POLL ||
      !options->control.send_count || !options->control.receive_count ||
      !options->control.service_batch_size ||
      !options->control.resolution_timeout_ms ||
      options->control.resolution_timeout_ms > INT_MAX ||
      options->control.minimum_rnr_timer > 31 ||
      !options->message.carrier.max_send_operations ||
      options->message.carrier.max_send_operations > INT32_MAX - 4 ||
      !options->message.carrier.chunk_capacity ||
      options->message.direct.max_write_entries != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid RDMA connection geometry");
  }
  const iree_net_rdma_direct_endpoint_options_t* data_options[] = {
      &options->direct, &options->message.direct};
  uint64_t work_capacity = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(data_options); ++i) {
    const iree_net_rdma_direct_endpoint_options_t* data = data_options[i];
    if (!data->max_write_operations || data->max_write_operations > INT_MAX ||
        !data->max_write_entries || !data->send_work_count ||
        data->send_work_count > INT_MAX || !data->receive_work_count ||
        data->receive_work_count > INT_MAX || !data->post_batch_size ||
        data->post_batch_size > data->send_work_count ||
        !data->max_request_length || data->minimum_rnr_timer > 31) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid RDMA endpoint geometry");
    }
    work_capacity = iree_max(work_capacity, (uint64_t)data->send_work_count +
                                                data->receive_work_count);
  }
  work_capacity = work_capacity * options->max_endpoint_count +
                  options->control.send_count + options->control.receive_count;
  if (work_capacity > INT_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "RDMA connection completion capacity overflow");
  }
  return iree_ok_status();
}

static void iree_net_rdma_connection_schedule_locked(
    iree_net_rdma_connection_t* connection) {
  if (iree_any_bit_set(connection->flags,
                       IREE_NET_RDMA_CONNECTION_PROGRESS_QUEUED)) {
    return;
  }
  connection->flags |= IREE_NET_RDMA_CONNECTION_PROGRESS_QUEUED;
  ++connection->pending_progress_count;
  iree_async_operation_t* operation = &connection->progress.base;
  iree_async_operation_initialize(operation, IREE_ASYNC_OPERATION_TYPE_NOP, 0,
                                  operation->completion_fn, connection);
  IREE_CHECK_OK(
      iree_async_proactor_submit_one(connection->proactor, operation));
}

static void iree_net_rdma_connection_fail(
    iree_net_rdma_connection_t* connection, iree_status_t status) {
  iree_slim_mutex_lock(&connection->mutex);
  connection->failure = iree_status_join(connection->failure, status);
  iree_net_rdma_connection_schedule_locked(connection);
  iree_slim_mutex_unlock(&connection->mutex);
}

static bool iree_net_rdma_connection_is_open(
    iree_net_rdma_connection_t* connection) {
  iree_slim_mutex_lock(&connection->mutex);
  bool open = connection->state == IREE_NET_RDMA_CONNECTION_OPEN &&
              iree_status_is_ok(connection->failure);
  iree_slim_mutex_unlock(&connection->mutex);
  return open;
}

// Called on the poll owner after every native/callback obligation can be
// joined. Outbound cancellation and result publication use the same lock order.
static void iree_net_rdma_connection_try_finish(
    iree_net_rdma_connection_t* connection) {
  iree_net_transport_connect_operation_t* operation =
      connection->setup.operation;
  if (operation) {
    iree_slim_mutex_lock(&operation->mutex);
  }
  iree_slim_mutex_lock(&connection->mutex);
  bool joined = !connection->pending_progress_count &&
                iree_any_bit_set(connection->flags,
                                 IREE_NET_RDMA_CONNECTION_CONTROL_DRAINED);
  iree_net_transport_connect_callback_t setup_callback = {0};
  iree_net_connection_deactivate_callback_t deactivated_callback = {0};
  iree_status_t status = iree_ok_status();
  if (joined && connection->state == IREE_NET_RDMA_CONNECTION_CONNECTING) {
    status = iree_status_clone(connection->failure);
    setup_callback = connection->setup.callback;
    connection->setup.callback = (iree_net_transport_connect_callback_t){0};
    connection->setup.operation = NULL;
    if (operation) {
      operation->binding.cancel_fn = NULL;
      operation->binding.user_data = NULL;
    }
    connection->state = IREE_NET_RDMA_CONNECTION_DEACTIVATED;
  } else if (joined && connection->state == IREE_NET_RDMA_CONNECTION_DRAINING) {
    deactivated_callback = connection->deactivated_callback;
    connection->deactivated_callback =
        (iree_net_connection_deactivate_callback_t){0};
    connection->state = IREE_NET_RDMA_CONNECTION_DEACTIVATED;
  }
  iree_slim_mutex_unlock(&connection->mutex);
  if (operation) {
    iree_slim_mutex_unlock(&operation->mutex);
  }
  if (setup_callback.fn) {
    setup_callback.fn(setup_callback.user_data, status, NULL);
    iree_net_connection_release(&connection->base);
  } else if (deactivated_callback.fn) {
    deactivated_callback.fn(deactivated_callback.user_data);
    iree_net_connection_release(&connection->base);
  }
}

static void iree_net_rdma_connection_control_drained(void* user_data,
                                                     iree_status_t status) {
  iree_net_rdma_connection_t* connection = user_data;
  if (!iree_status_is_ok(status)) {
    iree_net_rdma_connection_fail(connection, status);
    return;
  }
  iree_slim_mutex_lock(&connection->mutex);
  connection->flags |= IREE_NET_RDMA_CONNECTION_CONTROL_DRAINED;
  iree_slim_mutex_unlock(&connection->mutex);
  iree_net_rdma_connection_try_finish(connection);
}

static void iree_net_rdma_connection_endpoints_drained(void* user_data) {
  iree_net_rdma_connection_t* connection = user_data;
  iree_slim_mutex_lock(&connection->mutex);
  connection->flags |= IREE_NET_RDMA_CONNECTION_ENDPOINTS_DRAINED;
  // A concurrent submitter may release the final endpoint hold. Native control
  // retirement must return to the poll owner instead of running on that thread.
  iree_net_rdma_connection_schedule_locked(connection);
  iree_slim_mutex_unlock(&connection->mutex);
}

// Borrows status, cloning only for an outstanding result callback. A concurrent
// close can win before successful readiness publication.
static void iree_net_rdma_connection_endpoint_ready(
    iree_net_rdma_connection_endpoint_t* slot, iree_status_t status) {
  iree_net_rdma_connection_t* connection = slot->connection;
  iree_slim_mutex_lock(&connection->mutex);
  iree_net_endpoint_ready_callback_t message = {0};
  iree_net_direct_endpoint_ready_callback_t direct = {0};
  if (slot->kind == IREE_NET_RDMA_ENDPOINT_MESSAGE) {
    message = slot->ready.message;
    slot->ready.message = (iree_net_endpoint_ready_callback_t){0};
  } else if (slot->kind == IREE_NET_RDMA_ENDPOINT_DIRECT) {
    direct = slot->ready.direct;
    slot->ready.direct = (iree_net_direct_endpoint_ready_callback_t){0};
  }
  iree_status_t result = iree_ok_status();
  if (message.fn || direct.fn) {
    if (!iree_status_is_ok(status)) {
      result = iree_status_clone(status);
    } else if (!iree_status_is_ok(connection->failure)) {
      result = iree_status_clone(connection->failure);
    } else if (connection->state != IREE_NET_RDMA_CONNECTION_OPEN) {
      result = iree_status_from_code(IREE_STATUS_CANCELLED);
    }
  }
  iree_slim_mutex_unlock(&connection->mutex);
  if (message.fn) {
    iree_net_message_endpoint_t view = {0};
    if (iree_status_is_ok(result)) {
      view = iree_net_framed_endpoint_as_message_endpoint(slot->framed);
    }
    message.fn(message.user_data, result, view);
  } else if (direct.fn) {
    iree_net_direct_endpoint_t view = {0};
    if (iree_status_is_ok(result)) {
      view = iree_net_rdma_direct_endpoint_as_direct_endpoint(slot->native);
    }
    direct.fn(direct.user_data, result, view);
  }
}

static void iree_net_rdma_connection_retire(
    iree_net_rdma_connection_t* connection) {
  iree_slim_mutex_lock(&connection->mutex);
  bool closing = connection->state == IREE_NET_RDMA_CONNECTION_DRAINING ||
                 !iree_status_is_ok(connection->failure);
  bool begin_endpoints =
      closing && !iree_any_bit_set(connection->flags,
                                   IREE_NET_RDMA_CONNECTION_ENDPOINTS_DRAINING);
  if (begin_endpoints) {
    connection->flags |= IREE_NET_RDMA_CONNECTION_ENDPOINTS_DRAINING;
  }
  uint32_t count = connection->opened_count;
  iree_status_t failure = begin_endpoints
                              ? iree_status_clone(connection->failure)
                              : iree_ok_status();
  iree_slim_mutex_unlock(&connection->mutex);
  if (begin_endpoints) {
    for (uint32_t i = 0; i < count; ++i) {
      iree_net_rdma_connection_endpoint_t* slot = &connection->endpoints[i];
      iree_net_rdma_connection_endpoint_ready(
          slot, iree_status_is_ok(failure)
                    ? iree_status_from_code(IREE_STATUS_CANCELLED)
                    : failure);
      if (slot->framed) {
        if (!iree_status_is_ok(failure)) {
          iree_net_rdma_carrier_fail(slot->carrier, iree_status_clone(failure));
        }
        iree_net_framed_endpoint_join_deactivation(slot->framed);
      } else if (slot->native) {
        if (!iree_status_is_ok(failure)) {
          iree_net_rdma_direct_endpoint_fail(slot->native,
                                             iree_status_clone(failure));
        }
        iree_net_rdma_direct_endpoint_join_deactivation(slot->native);
      }
    }
    iree_net_endpoint_deactivation_barrier_commit(
        &connection->endpoint_barrier,
        (iree_net_connection_deactivate_callback_t){
            iree_net_rdma_connection_endpoints_drained, connection});
  }
  iree_status_free(failure);
  iree_slim_mutex_lock(&connection->mutex);
  bool begin_control =
      iree_any_bit_set(connection->flags,
                       IREE_NET_RDMA_CONNECTION_ENDPOINTS_DRAINED) &&
      !iree_any_bit_set(connection->flags,
                        IREE_NET_RDMA_CONNECTION_CONTROL_DRAINING);
  bool started = iree_any_bit_set(connection->flags,
                                  IREE_NET_RDMA_CONNECTION_CONTROL_STARTED);
  if (begin_control) {
    connection->flags |= IREE_NET_RDMA_CONNECTION_CONTROL_DRAINING;
  }
  iree_slim_mutex_unlock(&connection->mutex);
  if (begin_control) {
    for (uint32_t i = 0; i < count; ++i) {
      if (connection->endpoints[i].native) {
        // Retires created private message QPs that never acquired a barrier
        // hold.
        iree_net_rdma_direct_endpoint_join_deactivation(
            connection->endpoints[i].native);
      }
    }
    if (started) {
      iree_net_rdma_connection_control_deactivate(
          connection->control,
          (iree_async_event_source_unregistered_callback_t){
              iree_net_rdma_connection_control_drained, connection});
    } else {
      iree_net_rdma_connection_control_drained(connection, iree_ok_status());
    }
  }
}

static void iree_net_rdma_connection_publish(
    iree_net_rdma_connection_t* connection) {
  iree_net_transport_connect_operation_t* operation =
      connection->setup.operation;
  if (operation) {
    iree_slim_mutex_lock(&operation->mutex);
  }
  iree_slim_mutex_lock(&connection->mutex);
  iree_net_transport_connect_callback_t callback = {0};
  if (connection->state == IREE_NET_RDMA_CONNECTION_CONNECTING &&
      iree_status_is_ok(connection->failure) &&
      iree_any_bit_set(connection->flags,
                       IREE_NET_RDMA_CONNECTION_CONTROL_READY)) {
    connection->state = IREE_NET_RDMA_CONNECTION_OPEN;
    iree_net_connection_retain(&connection->base);
    callback = connection->setup.callback;
    connection->setup.callback = (iree_net_transport_connect_callback_t){0};
    connection->setup.operation = NULL;
    if (operation) {
      operation->binding.cancel_fn = NULL;
      operation->binding.user_data = NULL;
    }
  }
  iree_slim_mutex_unlock(&connection->mutex);
  if (operation) {
    iree_slim_mutex_unlock(&operation->mutex);
  }
  if (callback.fn) {
    callback.fn(callback.user_data, iree_ok_status(), &connection->base);
  }
}

static void iree_net_rdma_connection_credit(void* user_data,
                                            uint64_t posted_count) {
  iree_net_rdma_connection_endpoint_t* slot = user_data;
  slot->credit = posted_count;
  slot->flags |= IREE_NET_RDMA_ENDPOINT_CREDIT_PENDING;
  iree_slim_mutex_lock(&slot->connection->mutex);
  iree_net_rdma_connection_schedule_locked(slot->connection);
  iree_slim_mutex_unlock(&slot->connection->mutex);
}

static iree_status_t iree_net_rdma_connection_create_endpoint(
    iree_net_rdma_connection_endpoint_t* slot, uint32_t ordinal) {
  iree_net_rdma_connection_t* connection = slot->connection;
  const iree_net_rdma_connection_route_t* route =
      iree_net_rdma_connection_control_route(connection->control);
  // Reuse this connection incarnation's CM-generated PSN; QPNs distinguish
  // the independent native streams. No global sequence allocator is needed.
  uint32_t sequence_number = route->send.attributes.sq_psn;
  iree_net_rdma_direct_endpoint_options_t options =
      slot->kind == IREE_NET_RDMA_ENDPOINT_DIRECT
          ? connection->options.direct
          : connection->options.message.direct;
  options.max_request_length =
      iree_min(options.max_request_length, connection->peer_max_request_length);
  iree_net_rdma_direct_endpoint_callbacks_t callbacks = {
      iree_net_rdma_connection_credit, slot};
  if (slot->kind == IREE_NET_RDMA_ENDPOINT_DIRECT) {
    return iree_net_rdma_direct_endpoint_create(
        connection->context, connection->proactor, connection->control, ordinal,
        sequence_number, options, callbacks, &connection->endpoint_barrier,
        connection->base.host_allocator, &slot->native);
  }
  iree_net_carrier_t* carrier = NULL;
  IREE_RETURN_IF_ERROR(iree_net_rdma_carrier_create(
      connection->context, connection->proactor, connection->control, ordinal,
      sequence_number, options, connection->options.message.carrier, callbacks,
      connection->base.host_allocator, &carrier));
  iree_status_t status = iree_net_framed_endpoint_allocate(
      carrier, connection->proactor,
      connection->options.message.carrier.max_send_operations,
      &connection->endpoint_barrier, connection->base.host_allocator,
      &slot->framed);
  if (iree_status_is_ok(status)) {
    slot->carrier = carrier;
    slot->native = iree_net_rdma_carrier_direct_endpoint(carrier);
  } else {
    iree_net_carrier_release(carrier);
  }
  return status;
}

static iree_status_t iree_net_rdma_connection_connect_endpoint(
    iree_net_rdma_connection_endpoint_t* slot, uint32_t ordinal) {
  if (slot->native && slot->peer.kind != IREE_NET_RDMA_ENDPOINT_NONE &&
      iree_any_bit_set(slot->flags, IREE_NET_RDMA_ENDPOINT_OPEN_SENT) &&
      !iree_any_bit_set(slot->flags, IREE_NET_RDMA_ENDPOINT_CONNECTED)) {
    if (slot->kind != slot->peer.kind) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "RDMA endpoint %u kinds disagree", ordinal);
    }
    if (slot->kind == IREE_NET_RDMA_ENDPOINT_MESSAGE) {
      IREE_RETURN_IF_ERROR(iree_net_rdma_carrier_connect(
          slot->carrier, slot->peer.queue_number, slot->peer.sequence_number,
          slot->peer.receive_count, slot->peer.chunk_capacity,
          iree_make_const_byte_span(slot->peer.target,
                                    sizeof(slot->peer.target))));
    } else {
      IREE_RETURN_IF_ERROR(iree_net_rdma_direct_endpoint_connect(
          slot->native, slot->peer.queue_number, slot->peer.sequence_number,
          slot->peer.receive_count));
    }
    slot->flags |= IREE_NET_RDMA_ENDPOINT_CONNECTED;
  }
  return iree_ok_status();
}

static iree_status_t iree_net_rdma_connection_service_endpoint(
    iree_net_rdma_connection_endpoint_t* slot, uint32_t ordinal) {
  iree_net_rdma_connection_t* connection = slot->connection;
  if (!slot->native) {
    IREE_RETURN_IF_ERROR(
        iree_net_rdma_connection_create_endpoint(slot, ordinal));
  }
  if (!iree_any_bit_set(slot->flags, IREE_NET_RDMA_ENDPOINT_OPEN_SENT)) {
    uint8_t record[IREE_NET_RDMA_CONTROL_RECORD_SIZE] = {0};
    iree_unaligned_store_le_u32(record, (uint32_t)slot->kind);
    iree_unaligned_store_le_u32(record + 4, ordinal);
    iree_unaligned_store_le_u32(
        record + 8, iree_net_rdma_direct_endpoint_queue_number(slot->native));
    iree_unaligned_store_le_u32(
        record + 12, iree_net_rdma_connection_control_route(connection->control)
                         ->send.attributes.sq_psn);
    uint32_t receive_count = connection->options.direct.receive_work_count;
    if (slot->kind == IREE_NET_RDMA_ENDPOINT_MESSAGE) {
      receive_count = connection->options.message.direct.receive_work_count;
      iree_unaligned_store_le_u32(
          record + 20, connection->options.message.carrier.chunk_capacity);
      iree_host_size_t length = 0;
      IREE_RETURN_IF_ERROR(iree_net_rdma_carrier_export_receive(
          slot->carrier,
          iree_make_byte_span(record + 24, IREE_NET_RDMA_TARGET_WIRE_SIZE),
          &length));
    }
    iree_unaligned_store_le_u32(record + 16, receive_count);
    if (!iree_net_rdma_connection_control_try_send(connection->control,
                                                   record)) {
      return iree_ok_status();
    }
    slot->flags |= IREE_NET_RDMA_ENDPOINT_OPEN_SENT;
  }
  IREE_RETURN_IF_ERROR(
      iree_net_rdma_connection_connect_endpoint(slot, ordinal));
  if (iree_any_bit_set(slot->flags, IREE_NET_RDMA_ENDPOINT_CONNECTED)) {
    iree_net_rdma_connection_endpoint_ready(slot, iree_ok_status());
  }
  if (iree_any_bit_set(slot->flags, IREE_NET_RDMA_ENDPOINT_CREDIT_PENDING) &&
      iree_net_rdma_connection_is_open(connection)) {
    uint8_t record[IREE_NET_RDMA_CONTROL_RECORD_SIZE] = {0};
    iree_unaligned_store_le_u32(record, IREE_NET_RDMA_RECORD_CREDIT);
    iree_unaligned_store_le_u32(record + 4, ordinal);
    iree_unaligned_store_le_u64(record + 8, slot->credit);
    if (iree_net_rdma_connection_control_try_send(connection->control,
                                                  record)) {
      slot->flags &= ~IREE_NET_RDMA_ENDPOINT_CREDIT_PENDING;
    }
  }
  return iree_ok_status();
}

static void iree_net_rdma_connection_progress(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_net_rdma_connection_t* connection = user_data;
  if (!iree_status_is_ok(status)) {
    iree_net_rdma_connection_fail(connection, status);
  }
  iree_slim_mutex_lock(&connection->mutex);
  connection->flags &= ~IREE_NET_RDMA_CONNECTION_PROGRESS_QUEUED;
  bool start = connection->state == IREE_NET_RDMA_CONNECTION_CONNECTING &&
               iree_status_is_ok(connection->failure) &&
               !iree_any_bit_set(connection->flags,
                                 IREE_NET_RDMA_CONNECTION_CONTROL_STARTED);
  if (start) {
    connection->flags |= IREE_NET_RDMA_CONNECTION_CONTROL_STARTED;
  }
  iree_slim_mutex_unlock(&connection->mutex);
  if (start) {
    iree_net_rdma_connection_control_connect(connection->control,
                                             &connection->setup.address);
  }
  iree_net_rdma_connection_publish(connection);
  iree_slim_mutex_lock(&connection->mutex);
  uint32_t count = connection->opened_count;
  iree_slim_mutex_unlock(&connection->mutex);
  status = iree_ok_status();
  for (uint32_t i = 0; i < count && iree_status_is_ok(status) &&
                       iree_net_rdma_connection_is_open(connection);
       ++i) {
    status =
        iree_net_rdma_connection_service_endpoint(&connection->endpoints[i], i);
  }
  if (!iree_status_is_ok(status)) {
    iree_net_rdma_connection_fail(connection, status);
  }
  iree_net_rdma_connection_retire(connection);
  iree_slim_mutex_lock(&connection->mutex);
  --connection->pending_progress_count;
  iree_slim_mutex_unlock(&connection->mutex);
  iree_net_rdma_connection_try_finish(connection);
}

static void iree_net_rdma_connection_native_ready(void* user_data,
                                                  iree_const_byte_span_t data) {
  iree_net_rdma_connection_t* connection = user_data;
  if (data.data_length < IREE_NET_RDMA_HELLO_SIZE ||
      iree_unaligned_load_le_u32(data.data) != IREE_NET_RDMA_HELLO_MAGIC ||
      iree_unaligned_load_le_u32(data.data + 4) !=
          IREE_NET_RDMA_PROTOCOL_VERSION ||
      !iree_unaligned_load_le_u32(data.data + 8) ||
      !iree_unaligned_load_le_u32(data.data + 12)) {
    iree_net_rdma_connection_fail(
        connection, iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                     "incompatible RDMA connection hello"));
    return;
  }
  iree_slim_mutex_lock(&connection->mutex);
  connection->base.max_endpoint_count =
      iree_min(connection->options.max_endpoint_count,
               iree_unaligned_load_le_u32(data.data + 8));
  connection->peer_max_request_length =
      iree_unaligned_load_le_u32(data.data + 12);
  connection->flags |= IREE_NET_RDMA_CONNECTION_CONTROL_READY;
  iree_net_rdma_connection_schedule_locked(connection);
  iree_slim_mutex_unlock(&connection->mutex);
}

static iree_status_t iree_net_rdma_connection_record(
    void* user_data, iree_const_byte_span_t record) {
  iree_net_rdma_connection_t* connection = user_data;
  iree_slim_mutex_lock(&connection->mutex);
  bool retiring = !iree_status_is_ok(connection->failure) ||
                  connection->state >= IREE_NET_RDMA_CONNECTION_DRAINING;
  iree_slim_mutex_unlock(&connection->mutex);
  if (retiring) {
    return iree_ok_status();
  }
  uint32_t kind = iree_unaligned_load_le_u32(record.data);
  uint32_t ordinal = iree_unaligned_load_le_u32(record.data + 4);
  if (ordinal >= connection->base.max_endpoint_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "RDMA control endpoint ordinal exceeds capacity");
  }
  iree_net_rdma_connection_endpoint_t* slot = &connection->endpoints[ordinal];
  if (kind == IREE_NET_RDMA_RECORD_CREDIT) {
    if (!slot->native || slot->peer.kind == IREE_NET_RDMA_ENDPOINT_NONE ||
        !iree_any_bit_set(slot->flags, IREE_NET_RDMA_ENDPOINT_OPEN_SENT)) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "RDMA credit arrived before endpoint setup");
    }
    return iree_net_rdma_direct_endpoint_update_credit(
        slot->native, iree_unaligned_load_le_u64(record.data + 8));
  }
  uint32_t queue_number = iree_unaligned_load_le_u32(record.data + 8);
  uint32_t sequence_number = iree_unaligned_load_le_u32(record.data + 12);
  uint32_t receive_count = iree_unaligned_load_le_u32(record.data + 16);
  if ((kind != IREE_NET_RDMA_ENDPOINT_MESSAGE &&
       kind != IREE_NET_RDMA_ENDPOINT_DIRECT) ||
      slot->peer.kind != IREE_NET_RDMA_ENDPOINT_NONE || !queue_number ||
      queue_number > 0xffffffu || sequence_number > 0xffffffu ||
      !receive_count || receive_count > INT_MAX) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "invalid or duplicate RDMA endpoint OPEN");
  }
  slot->peer.kind = (iree_net_rdma_endpoint_kind_t)kind;
  slot->peer.queue_number = queue_number;
  slot->peer.sequence_number = sequence_number;
  slot->peer.receive_count = receive_count;
  if (kind == IREE_NET_RDMA_ENDPOINT_MESSAGE) {
    slot->peer.chunk_capacity = iree_unaligned_load_le_u32(record.data + 20);
    memcpy(slot->peer.target, record.data + 24, sizeof(slot->peer.target));
  }
  // OPEN and initial CREDIT can share a native receive batch. Install peer
  // geometry now if local setup exists; public readiness remains on the NOP.
  IREE_RETURN_IF_ERROR(
      iree_net_rdma_connection_connect_endpoint(slot, ordinal));
  iree_slim_mutex_lock(&connection->mutex);
  iree_net_rdma_connection_schedule_locked(connection);
  iree_slim_mutex_unlock(&connection->mutex);
  return iree_ok_status();
}

static void iree_net_rdma_connection_capacity(void* user_data) {
  iree_net_rdma_connection_t* connection = user_data;
  iree_slim_mutex_lock(&connection->mutex);
  if (connection->state == IREE_NET_RDMA_CONNECTION_OPEN &&
      iree_status_is_ok(connection->failure)) {
    iree_net_rdma_connection_schedule_locked(connection);
  }
  iree_slim_mutex_unlock(&connection->mutex);
}

static void iree_net_rdma_connection_completions(
    void* user_data, iree_host_size_t count, const struct ibv_wc* completions) {
  iree_net_rdma_connection_t* connection = user_data;
  for (iree_host_size_t i = 0; i < count; ++i) {
    // WR identity is constructed locally, including every unsignaled request.
    // It remains valid after QP retirement until this shared CQ joins.
    uint32_t ordinal = (uint32_t)(completions[i].wr_id >> 33);
    iree_net_rdma_direct_endpoint_complete(
        connection->endpoints[ordinal].native, &completions[i]);
  }
}

static void iree_net_rdma_connection_native_error(void* user_data,
                                                  iree_status_t status) {
  iree_net_rdma_connection_fail(user_data, status);
}

static void iree_net_rdma_connection_destroy(iree_net_connection_t* base) {
  iree_net_rdma_connection_t* connection = (iree_net_rdma_connection_t*)base;
  IREE_ASSERT(connection->state == IREE_NET_RDMA_CONNECTION_CREATED ||
                  connection->state == IREE_NET_RDMA_CONNECTION_DEACTIVATED,
              "RDMA connection released before native/callback retirement");
  for (uint32_t i = 0; i < connection->opened_count; ++i) {
    iree_net_rdma_connection_endpoint_t* slot = &connection->endpoints[i];
    if (slot->framed) {
      iree_net_framed_endpoint_free(slot->framed);
    } else {
      iree_net_rdma_direct_endpoint_destroy(slot->native);
    }
  }
  iree_net_rdma_connection_control_destroy(connection->control);
  iree_status_free(connection->failure);
  iree_slim_mutex_deinitialize(&connection->mutex);
  iree_async_proactor_release(connection->proactor);
  iree_net_rdma_context_release(connection->context);
  iree_allocator_free(base->host_allocator, connection);
}

static void iree_net_rdma_connection_deactivate(
    iree_net_connection_t* base,
    iree_net_connection_deactivate_callback_t callback) {
  iree_net_rdma_connection_t* connection = (iree_net_rdma_connection_t*)base;
  iree_slim_mutex_lock(&connection->mutex);
  IREE_ASSERT(connection->state == IREE_NET_RDMA_CONNECTION_OPEN,
              "RDMA deactivation requires a published connection");
  connection->state = IREE_NET_RDMA_CONNECTION_DRAINING;
  connection->deactivated_callback = callback;
  iree_net_rdma_connection_schedule_locked(connection);
  iree_slim_mutex_unlock(&connection->mutex);
}

static iree_status_t iree_net_rdma_connection_claim_locked(
    iree_net_rdma_connection_t* connection,
    iree_net_rdma_connection_endpoint_t** out_slot) {
  if (connection->state != IREE_NET_RDMA_CONNECTION_OPEN) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "RDMA connection is not open");
  }
  if (!iree_status_is_ok(connection->failure)) {
    return iree_status_clone(connection->failure);
  }
  if (connection->opened_count == connection->base.max_endpoint_count) {
    return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  }
  *out_slot = &connection->endpoints[connection->opened_count++];
  return iree_ok_status();
}

static iree_status_t iree_net_rdma_connection_open_endpoint(
    iree_net_connection_t* base, iree_net_endpoint_ready_callback_t callback) {
  iree_net_rdma_connection_t* connection = (iree_net_rdma_connection_t*)base;
  iree_slim_mutex_lock(&connection->mutex);
  iree_net_rdma_connection_endpoint_t* slot = NULL;
  iree_status_t status =
      iree_net_rdma_connection_claim_locked(connection, &slot);
  if (iree_status_is_ok(status)) {
    slot->kind = IREE_NET_RDMA_ENDPOINT_MESSAGE;
    slot->ready.message = callback;
    iree_net_rdma_connection_schedule_locked(connection);
  }
  iree_slim_mutex_unlock(&connection->mutex);
  return status;
}

static iree_status_t iree_net_rdma_connection_open_direct_endpoint(
    iree_net_connection_t* base,
    iree_net_direct_endpoint_ready_callback_t callback) {
  iree_net_rdma_connection_t* connection = (iree_net_rdma_connection_t*)base;
  iree_slim_mutex_lock(&connection->mutex);
  iree_net_rdma_connection_endpoint_t* slot = NULL;
  iree_status_t status =
      iree_net_rdma_connection_claim_locked(connection, &slot);
  if (iree_status_is_ok(status)) {
    slot->kind = IREE_NET_RDMA_ENDPOINT_DIRECT;
    slot->ready.direct = callback;
    iree_net_rdma_connection_schedule_locked(connection);
  }
  iree_slim_mutex_unlock(&connection->mutex);
  return status;
}

static iree_async_proactor_t* iree_net_rdma_connection_proactor(
    iree_net_connection_t* base) {
  return ((iree_net_rdma_connection_t*)base)->proactor;
}

static const iree_net_connection_vtable_t iree_net_rdma_connection_vtable = {
    .destroy = iree_net_rdma_connection_destroy,
    .deactivate = iree_net_rdma_connection_deactivate,
    .open_endpoint = iree_net_rdma_connection_open_endpoint,
    .open_direct_endpoint = iree_net_rdma_connection_open_direct_endpoint,
    .proactor = iree_net_rdma_connection_proactor,
};

iree_status_t iree_net_rdma_connection_create(
    iree_net_rdma_context_t* context, iree_async_proactor_t* proactor,
    const iree_net_rdma_connection_options_t* options,
    iree_allocator_t host_allocator,
    iree_net_rdma_connection_t** out_connection) {
  *out_connection = NULL;
  IREE_RETURN_IF_ERROR(iree_net_rdma_connection_options_validate(options));
  iree_host_size_t allocation_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_net_rdma_connection_t), &allocation_size,
      IREE_STRUCT_FIELD_FAM(options->max_endpoint_count,
                            iree_net_rdma_connection_endpoint_t)));
  iree_net_rdma_connection_t* connection = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, allocation_size,
                                             (void**)&connection));
  iree_net_connection_initialize(&iree_net_rdma_connection_vtable,
                                 host_allocator, options->max_endpoint_count,
                                 &connection->base);
  connection->context = context;
  iree_net_rdma_context_retain(context);
  connection->proactor = proactor;
  iree_async_proactor_retain(proactor);
  connection->options = *options;
  iree_slim_mutex_initialize(&connection->mutex);
  iree_net_endpoint_deactivation_barrier_initialize(
      &connection->endpoint_barrier);
  connection->progress.base.completion_fn = iree_net_rdma_connection_progress;
  for (uint32_t i = 0; i < options->max_endpoint_count; ++i) {
    connection->endpoints[i].connection = connection;
  }
  uint32_t data_work_capacity =
      options->max_endpoint_count *
      iree_max(
          options->direct.send_work_count + options->direct.receive_work_count,
          options->message.direct.send_work_count +
              options->message.direct.receive_work_count);
  iree_net_rdma_connection_control_options_t control_options = {
      .send_count = options->control.send_count,
      .receive_count = options->control.receive_count,
      .data_work_capacity = data_work_capacity,
      .service_batch_size = options->control.service_batch_size,
      .completion_mode = options->completion_mode,
      .resolution_timeout_ms = options->control.resolution_timeout_ms,
      .minimum_rnr_timer = options->control.minimum_rnr_timer,
  };
  uint8_t hello[IREE_NET_RDMA_HELLO_SIZE];
  iree_unaligned_store_le_u32(hello, IREE_NET_RDMA_HELLO_MAGIC);
  iree_unaligned_store_le_u32(hello + 4, IREE_NET_RDMA_PROTOCOL_VERSION);
  iree_unaligned_store_le_u32(hello + 8, options->max_endpoint_count);
  iree_unaligned_store_le_u32(
      hello + 12, iree_net_rdma_context_port_attributes(context)->max_msg_sz);
  iree_status_t status = iree_net_rdma_connection_control_create(
      context, proactor, control_options,
      iree_make_const_byte_span(hello, sizeof(hello)),
      (iree_net_rdma_connection_control_callbacks_t){
          .on_ready = iree_net_rdma_connection_native_ready,
          .on_record = iree_net_rdma_connection_record,
          .on_capacity = iree_net_rdma_connection_capacity,
          .on_completions = iree_net_rdma_connection_completions,
          .on_error = iree_net_rdma_connection_native_error,
          .user_data = connection,
      },
      host_allocator, &connection->control);
  if (iree_status_is_ok(status)) {
    *out_connection = connection;
  } else {
    iree_net_connection_release(&connection->base);
  }
  return status;
}

iree_net_connection_t* iree_net_rdma_connection_base(
    iree_net_rdma_connection_t* connection) {
  return &connection->base;
}

void iree_net_rdma_connection_cancel(iree_net_rdma_connection_t* connection) {
  iree_slim_mutex_lock(&connection->mutex);
  if (connection->state == IREE_NET_RDMA_CONNECTION_CONNECTING &&
      iree_status_is_ok(connection->failure)) {
    connection->failure = iree_status_from_code(IREE_STATUS_CANCELLED);
    iree_net_rdma_connection_schedule_locked(connection);
  }
  iree_slim_mutex_unlock(&connection->mutex);
}

static void iree_net_rdma_connection_cancel_bound(void* user_data) {
  iree_net_rdma_connection_cancel(user_data);
}

void iree_net_rdma_connection_connect(
    iree_net_rdma_connection_t* connection, const iree_async_address_t* address,
    iree_net_transport_connect_callback_t callback,
    iree_net_transport_connect_operation_t* operation) {
  connection->state = IREE_NET_RDMA_CONNECTION_CONNECTING;
  connection->setup.operation = operation;
  connection->setup.address = *address;
  connection->setup.callback = callback;
  operation->binding.cancel_fn = iree_net_rdma_connection_cancel_bound;
  operation->binding.user_data = connection;
  iree_slim_mutex_lock(&connection->mutex);
  iree_net_rdma_connection_schedule_locked(connection);
  iree_slim_mutex_unlock(&connection->mutex);
}

void iree_net_rdma_connection_accept(
    iree_net_rdma_connection_t* connection, struct rdma_cm_id* id,
    iree_const_byte_span_t private_data,
    iree_net_transport_connect_callback_t callback) {
  connection->state = IREE_NET_RDMA_CONNECTION_CONNECTING;
  connection->setup.callback = callback;
  connection->flags |= IREE_NET_RDMA_CONNECTION_CONTROL_STARTED;
  iree_net_rdma_connection_control_accept(connection->control, id,
                                          private_data);
}
