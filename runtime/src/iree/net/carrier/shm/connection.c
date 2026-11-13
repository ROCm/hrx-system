// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/shm/connection.h"

#include <string.h>

#include "iree/async/operations/scheduling.h"
#include "iree/base/threading/mutex.h"
#include "iree/net/carrier/shm/storage.h"
#include "iree/net/endpoint_lifecycle.h"
#include "iree/net/framed_endpoint.h"

typedef struct iree_net_shm_connection_t iree_net_shm_connection_t;

typedef enum iree_net_shm_connection_state_e {
  IREE_NET_SHM_CONNECTION_STATE_CREATED = 0,
  IREE_NET_SHM_CONNECTION_STATE_OPEN,
  IREE_NET_SHM_CONNECTION_STATE_DRAINING,
  IREE_NET_SHM_CONNECTION_STATE_DEACTIVATED,
} iree_net_shm_connection_state_t;

enum iree_net_shm_connection_flag_bits_e {
  IREE_NET_SHM_CONNECTION_FLAG_CONTROL_QUEUED = 1u << 0,
  IREE_NET_SHM_CONNECTION_FLAG_FAILED = 1u << 1,
  IREE_NET_SHM_CONNECTION_FLAG_FAILURE_PENDING = 1u << 2,
  IREE_NET_SHM_CONNECTION_FLAG_DETACHED = 1u << 3,
  IREE_NET_SHM_CONNECTION_FLAG_ENDPOINTS_DRAINING = 1u << 4,
  IREE_NET_SHM_CONNECTION_FLAG_ENDPOINTS_DRAINED = 1u << 5,
  IREE_NET_SHM_CONNECTION_FLAG_STREAM_DRAINING = 1u << 6,
  IREE_NET_SHM_CONNECTION_FLAG_STREAM_DRAINED = 1u << 7,
};
typedef uint32_t iree_net_shm_connection_flags_t;

typedef struct iree_net_shm_connection_endpoint_t {
  // Containing connection; its active hold joins every ready callback.
  iree_net_shm_connection_t* connection;
  // Owned framing adapter for this protocol ordinal.
  iree_net_framed_endpoint_t* endpoint;
  // Carrier borrowed from |endpoint| for connection-wide failure fanout.
  iree_net_carrier_t* carrier;
  // Preallocated one-shot endpoint-ready dispatch.
  iree_async_nop_operation_t ready_operation;
  // Accepted readiness callback, protected by the connection mutex.
  iree_net_endpoint_ready_callback_t ready_callback;
} iree_net_shm_connection_endpoint_t;

struct iree_net_shm_connection_t {
  // Public connection base; must be first.
  iree_net_connection_t base;
  // Serializes admission, weak failure handoff, and drain accounting.
  iree_slim_mutex_t mutex;
  // Retained callback executor, independent of detached receive leases.
  iree_async_proactor_t* proactor;
  // Retained mapping and native wake ownership, shared with receive leases.
  iree_net_shm_storage_t* storage;
  // Retained single poll consumer of this side's native wake.
  iree_async_notification_t* notification;
  // Lifecycle protected by |mutex|.
  iree_net_shm_connection_state_t state;
  // Independent monotonic failure/drain obligations, protected by |mutex|.
  iree_net_shm_connection_flags_t flags;
  // Number of monotonically claimed protocol endpoint ordinals.
  uint32_t opened_endpoint_count;
  // Ready callbacks still queued or executing, including user callback bodies.
  uint32_t pending_ready_count;
  // Control callbacks still queued or executing, including reentrant fanout.
  uint32_t pending_control_count;
  // Coalesced poll-owner failure/deactivation handoff.
  iree_async_nop_operation_t control_operation;
  // Owned bootstrap stream, retained after READY to observe peer departure.
  iree_net_shm_connection_channel_t channel;
  // Receive destination for EOF observation; any byte is a protocol error.
  uint8_t peer_byte;
  // Callback delivered only after every connection-owned callback has retired.
  iree_net_connection_deactivate_callback_t deactivate_callback;
  // Joins endpoint carrier and framing callbacks before connection teardown.
  iree_net_endpoint_deactivation_barrier_t deactivation_barrier;
  // Eagerly allocated endpoints, indexed by protocol ordinal.
  iree_net_shm_connection_endpoint_t endpoints[];
};

void iree_net_shm_connection_channel_deinitialize(
    iree_net_shm_connection_channel_t* channel) {
  iree_async_local_stream_free(channel->stream);
#if defined(IREE_PLATFORM_WINDOWS)
  iree_async_primitive_close(&channel->pipe);
#else
  iree_async_socket_release(channel->socket);
#endif
  memset(channel, 0, sizeof(*channel));
}

static void iree_net_shm_connection_destroy(iree_net_connection_t* base) {
  iree_net_shm_connection_t* connection = (iree_net_shm_connection_t*)base;
  IREE_ASSERT(
      connection->state == IREE_NET_SHM_CONNECTION_STATE_CREATED ||
          connection->state == IREE_NET_SHM_CONNECTION_STATE_DEACTIVATED,
      "published SHM connection released before deactivation");
  iree_allocator_t host_allocator = base->host_allocator;
  for (uint32_t i = 0; i < base->max_endpoint_count; ++i) {
    iree_net_framed_endpoint_free(connection->endpoints[i].endpoint);
  }
  iree_async_notification_release(connection->notification);
  iree_net_shm_storage_release(connection->storage);
  iree_async_proactor_release(connection->proactor);
  iree_slim_mutex_deinitialize(&connection->mutex);
  iree_allocator_free(host_allocator, connection);
}

// The active lifetime hold survives through the callback. Every caller must
// treat this as its final access because the callback may release the owner.
static void iree_net_shm_connection_try_complete_deactivation(
    iree_net_shm_connection_t* connection) {
  iree_net_connection_deactivate_callback_t callback = {0};
  iree_slim_mutex_lock(&connection->mutex);
  const iree_net_shm_connection_flags_t required_flags =
      IREE_NET_SHM_CONNECTION_FLAG_DETACHED |
      IREE_NET_SHM_CONNECTION_FLAG_ENDPOINTS_DRAINED |
      IREE_NET_SHM_CONNECTION_FLAG_STREAM_DRAINED;
  if (connection->state == IREE_NET_SHM_CONNECTION_STATE_DRAINING &&
      iree_all_bits_set(connection->flags, required_flags) &&
      connection->pending_control_count == 0 &&
      connection->pending_ready_count == 0) {
    connection->state = IREE_NET_SHM_CONNECTION_STATE_DEACTIVATED;
    callback = connection->deactivate_callback;
    connection->deactivate_callback =
        (iree_net_connection_deactivate_callback_t){0};
  }
  iree_slim_mutex_unlock(&connection->mutex);
  if (callback.fn) {
    callback.fn(callback.user_data);
    iree_net_connection_release(&connection->base);
  }
}

// Called under the connection mutex. Completion is installed during creation;
// NOP admission is resource-free and cannot run the callback inline.
static void iree_net_shm_connection_kick_locked(
    iree_net_shm_connection_t* connection) {
  if (iree_any_bit_set(connection->flags,
                       IREE_NET_SHM_CONNECTION_FLAG_CONTROL_QUEUED)) {
    return;
  }
  connection->flags |= IREE_NET_SHM_CONNECTION_FLAG_CONTROL_QUEUED;
  ++connection->pending_control_count;
  iree_async_operation_t* operation = &connection->control_operation.base;
  iree_async_operation_initialize(operation, IREE_ASYNC_OPERATION_TYPE_NOP,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  operation->completion_fn, connection);
  IREE_CHECK_OK(
      iree_async_proactor_submit_one(connection->proactor, operation));
}

// Storage holds its failure mutex through this weak handoff. Detach joins this
// admission; the counted NOP, not a refcount alone, joins callback lifetime.
static void iree_net_shm_connection_storage_failed(void* user_data) {
  iree_net_shm_connection_t* connection = user_data;
  iree_slim_mutex_lock(&connection->mutex);
  connection->flags |= IREE_NET_SHM_CONNECTION_FLAG_FAILED |
                       IREE_NET_SHM_CONNECTION_FLAG_FAILURE_PENDING;
  iree_net_shm_connection_kick_locked(connection);
  iree_slim_mutex_unlock(&connection->mutex);
}

static void iree_net_shm_connection_stream_drained(void* user_data) {
  iree_net_shm_connection_t* connection = user_data;
  iree_net_shm_connection_channel_deinitialize(&connection->channel);
  iree_slim_mutex_lock(&connection->mutex);
  connection->flags |= IREE_NET_SHM_CONNECTION_FLAG_STREAM_DRAINED;
  iree_slim_mutex_unlock(&connection->mutex);
  iree_net_shm_connection_try_complete_deactivation(connection);
}

static void iree_net_shm_connection_endpoints_drained(void* user_data) {
  iree_net_shm_connection_t* connection = user_data;
  iree_slim_mutex_lock(&connection->mutex);
  connection->flags |= IREE_NET_SHM_CONNECTION_FLAG_ENDPOINTS_DRAINED;
  iree_slim_mutex_unlock(&connection->mutex);
  iree_net_shm_connection_try_complete_deactivation(connection);
}

static void iree_net_shm_connection_control_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_net_shm_connection_t* connection = user_data;
  if (!iree_status_is_ok(status)) {
    iree_net_shm_storage_fail(connection->storage, status);
  }
  iree_slim_mutex_lock(&connection->mutex);
  connection->flags &= ~IREE_NET_SHM_CONNECTION_FLAG_CONTROL_QUEUED;
  bool detach = connection->state == IREE_NET_SHM_CONNECTION_STATE_DRAINING &&
                !iree_any_bit_set(connection->flags,
                                  IREE_NET_SHM_CONNECTION_FLAG_DETACHED);
  bool fail = iree_any_bit_set(connection->flags,
                               IREE_NET_SHM_CONNECTION_FLAG_FAILURE_PENDING);
  connection->flags &= ~IREE_NET_SHM_CONNECTION_FLAG_FAILURE_PENDING;
  bool drain_stream =
      (connection->state == IREE_NET_SHM_CONNECTION_STATE_DRAINING || fail) &&
      !iree_any_bit_set(connection->flags,
                        IREE_NET_SHM_CONNECTION_FLAG_STREAM_DRAINING);
  if (drain_stream) {
    connection->flags |= IREE_NET_SHM_CONNECTION_FLAG_STREAM_DRAINING;
  }
  bool drain_endpoints =
      connection->state == IREE_NET_SHM_CONNECTION_STATE_DRAINING &&
      connection->pending_ready_count == 0 &&
      !iree_any_bit_set(connection->flags,
                        IREE_NET_SHM_CONNECTION_FLAG_ENDPOINTS_DRAINING);
  if (drain_endpoints) {
    connection->flags |= IREE_NET_SHM_CONNECTION_FLAG_ENDPOINTS_DRAINING;
  }
  iree_slim_mutex_unlock(&connection->mutex);

  if (detach) {
    iree_net_shm_storage_set_failure_callback(
        connection->storage, (iree_net_shm_storage_failure_callback_t){0});
    iree_slim_mutex_lock(&connection->mutex);
    connection->flags |= IREE_NET_SHM_CONNECTION_FLAG_DETACHED;
    iree_slim_mutex_unlock(&connection->mutex);
  }
  if (fail) {
    for (uint32_t i = 0; i < connection->base.max_endpoint_count; ++i) {
      iree_net_shm_carrier_fail(
          connection->endpoints[i].carrier,
          iree_net_shm_storage_clone_failure(connection->storage));
    }
  }
  if (drain_stream) {
    IREE_CHECK_OK(iree_async_local_stream_deactivate(
        connection->channel.stream,
        (iree_async_local_stream_deactivated_callback_t){
            .fn = iree_net_shm_connection_stream_drained,
            .user_data = connection,
        }));
  }
  if (drain_endpoints) {
    for (uint32_t i = 0; i < connection->base.max_endpoint_count; ++i) {
      iree_net_framed_endpoint_join_deactivation(
          connection->endpoints[i].endpoint);
    }
    iree_net_endpoint_deactivation_barrier_commit(
        &connection->deactivation_barrier,
        (iree_net_connection_deactivate_callback_t){
            .fn = iree_net_shm_connection_endpoints_drained,
            .user_data = connection,
        });
  }

  iree_slim_mutex_lock(&connection->mutex);
  --connection->pending_control_count;
  iree_slim_mutex_unlock(&connection->mutex);
  iree_net_shm_connection_try_complete_deactivation(connection);
}

static void iree_net_shm_connection_deactivate(
    iree_net_connection_t* base,
    iree_net_connection_deactivate_callback_t callback) {
  iree_net_shm_connection_t* connection = (iree_net_shm_connection_t*)base;
  iree_slim_mutex_lock(&connection->mutex);
  IREE_ASSERT(connection->state == IREE_NET_SHM_CONNECTION_STATE_OPEN,
              "SHM connection deactivation requires a published connection");
  connection->state = IREE_NET_SHM_CONNECTION_STATE_DRAINING;
  connection->deactivate_callback = callback;
  iree_net_shm_connection_kick_locked(connection);
  iree_slim_mutex_unlock(&connection->mutex);
}

static void iree_net_shm_connection_endpoint_ready(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_net_shm_connection_endpoint_t* slot = user_data;
  iree_net_shm_connection_t* connection = slot->connection;
  iree_slim_mutex_lock(&connection->mutex);
  iree_net_endpoint_ready_callback_t callback = slot->ready_callback;
  slot->ready_callback = (iree_net_endpoint_ready_callback_t){0};
  bool open = connection->state == IREE_NET_SHM_CONNECTION_STATE_OPEN;
  bool failed =
      iree_any_bit_set(connection->flags, IREE_NET_SHM_CONNECTION_FLAG_FAILED);
  iree_slim_mutex_unlock(&connection->mutex);
  if (iree_status_is_ok(status)) {
    if (failed) {
      status = iree_net_shm_storage_clone_failure(connection->storage);
    } else if (!open) {
      status = iree_make_status(IREE_STATUS_CANCELLED,
                                "SHM connection deactivated before ready");
    }
  }
  iree_net_message_endpoint_t endpoint = {0};
  if (iree_status_is_ok(status)) {
    endpoint = iree_net_framed_endpoint_as_message_endpoint(slot->endpoint);
  }
  callback.fn(callback.user_data, status, endpoint);

  iree_slim_mutex_lock(&connection->mutex);
  --connection->pending_ready_count;
  if (connection->state == IREE_NET_SHM_CONNECTION_STATE_DRAINING &&
      connection->pending_ready_count == 0) {
    iree_net_shm_connection_kick_locked(connection);
  }
  iree_slim_mutex_unlock(&connection->mutex);
  iree_net_shm_connection_try_complete_deactivation(connection);
}

static iree_status_t iree_net_shm_connection_open_endpoint(
    iree_net_connection_t* base, iree_net_endpoint_ready_callback_t callback) {
  iree_net_shm_connection_t* connection = (iree_net_shm_connection_t*)base;
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&connection->mutex);
  if (connection->state != IREE_NET_SHM_CONNECTION_STATE_OPEN) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "SHM connection is not open");
  } else if (iree_any_bit_set(connection->flags,
                              IREE_NET_SHM_CONNECTION_FLAG_FAILED)) {
    // Failure is immutable. Clone outside this mutex to preserve the storage
    // failure -> connection admission lock order.
    iree_slim_mutex_unlock(&connection->mutex);
    return iree_net_shm_storage_clone_failure(connection->storage);
  } else if (connection->opened_endpoint_count >= base->max_endpoint_count) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "all %u SHM endpoint slots are claimed",
                              base->max_endpoint_count);
  } else {
    iree_net_shm_connection_endpoint_t* slot =
        &connection->endpoints[connection->opened_endpoint_count];
    slot->ready_callback = callback;
    iree_async_operation_initialize(
        &slot->ready_operation.base, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_shm_connection_endpoint_ready,
        slot);
    ++connection->pending_ready_count;
    status = iree_async_proactor_submit_one(connection->proactor,
                                            &slot->ready_operation.base);
    if (iree_status_is_ok(status)) {
      ++connection->opened_endpoint_count;
    } else {
      --connection->pending_ready_count;
      slot->ready_callback = (iree_net_endpoint_ready_callback_t){0};
    }
  }
  iree_slim_mutex_unlock(&connection->mutex);
  return status;
}

static iree_async_proactor_t* iree_net_shm_connection_proactor(
    iree_net_connection_t* base) {
  return ((iree_net_shm_connection_t*)base)->proactor;
}

static const iree_net_connection_vtable_t iree_net_shm_connection_vtable = {
    .destroy = iree_net_shm_connection_destroy,
    .deactivate = iree_net_shm_connection_deactivate,
    .open_endpoint = iree_net_shm_connection_open_endpoint,
    .open_direct_endpoint = NULL,
    .proactor = iree_net_shm_connection_proactor,
};

iree_status_t iree_net_shm_connection_create(
    iree_async_proactor_t* proactor, iree_net_shm_storage_t* storage,
    const iree_net_shm_carrier_options_t* options,
    iree_allocator_t host_allocator, iree_net_connection_t** out_connection) {
  *out_connection = NULL;
  uint32_t endpoint_count = storage->layout.options.endpoint_count;
  iree_host_size_t allocation_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_net_shm_connection_t), &allocation_size,
      IREE_STRUCT_FIELD_FAM(endpoint_count,
                            iree_net_shm_connection_endpoint_t)));
  iree_net_shm_connection_t* connection = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, allocation_size,
                                             (void**)&connection));
  memset(connection, 0, allocation_size);
  iree_net_connection_initialize(&iree_net_shm_connection_vtable,
                                 host_allocator, endpoint_count,
                                 &connection->base);
  iree_slim_mutex_initialize(&connection->mutex);
  iree_net_endpoint_deactivation_barrier_initialize(
      &connection->deactivation_barrier);
  connection->control_operation.base.completion_fn =
      iree_net_shm_connection_control_complete;
  connection->proactor = proactor;
  iree_async_proactor_retain(proactor);
  connection->storage = storage;
  iree_net_shm_storage_retain(storage);

  iree_status_t status = iree_async_notification_create_shared(
      proactor, &storage->wakes[storage->side], &connection->notification);
  for (uint32_t i = 0; i < endpoint_count && iree_status_is_ok(status); ++i) {
    iree_net_shm_connection_endpoint_t* slot = &connection->endpoints[i];
    slot->connection = connection;
    iree_net_carrier_t* carrier = NULL;
    status =
        iree_net_shm_carrier_create(proactor, storage, connection->notification,
                                    i, options, host_allocator, &carrier);
    if (iree_status_is_ok(status)) {
      status = iree_net_framed_endpoint_allocate(
          carrier, proactor, options->max_send_operations,
          &connection->deactivation_barrier, host_allocator, &slot->endpoint);
      if (iree_status_is_ok(status)) {
        slot->carrier = carrier;
        carrier = NULL;
      }
    }
    iree_net_carrier_release(carrier);
  }
  if (iree_status_is_ok(status)) {
    *out_connection = &connection->base;
  } else {
    iree_net_connection_release(&connection->base);
  }
  return status;
}

static void iree_net_shm_connection_peer_complete(void* user_data,
                                                  iree_status_t status) {
  iree_net_shm_connection_t* connection = user_data;
  iree_slim_mutex_lock(&connection->mutex);
  bool closing = iree_any_bit_set(connection->flags,
                                  IREE_NET_SHM_CONNECTION_FLAG_STREAM_DRAINING);
  iree_slim_mutex_unlock(&connection->mutex);
  if (closing && iree_status_is_cancelled(status)) {
    iree_status_free(status);  // Expected terminal result of our stream drain.
    return;
  }
  if (iree_status_is_ok(status)) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "unexpected SHM control byte after READY");
  } else if (iree_status_is_out_of_range(status)) {
    status = iree_status_annotate(status, IREE_SV("SHM peer disconnected"));
  }
  iree_net_shm_storage_fail(connection->storage, status);
}

void iree_net_shm_connection_publish(
    iree_net_connection_t* base, iree_net_shm_connection_channel_t* channel) {
  iree_net_shm_connection_t* connection = (iree_net_shm_connection_t*)base;
  IREE_ASSERT(connection->state == IREE_NET_SHM_CONNECTION_STATE_CREATED,
              "SHM connection published more than once");
  connection->channel = *channel;
  memset(channel, 0, sizeof(*channel));
  iree_net_connection_retain(base);
  connection->state = IREE_NET_SHM_CONNECTION_STATE_OPEN;
  iree_net_shm_storage_set_failure_callback(
      connection->storage, (iree_net_shm_storage_failure_callback_t){
                               .fn = iree_net_shm_connection_storage_failed,
                               .user_data = connection,
                           });
  IREE_CHECK_OK(iree_async_local_stream_receive(
      connection->channel.stream,
      iree_make_byte_span(&connection->peer_byte, 1), 0,
      (iree_async_local_stream_callback_t){
          .fn = iree_net_shm_connection_peer_complete,
          .user_data = connection,
      },
      NULL));
}
