// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/net/channel.h"

#include "iree/base/alignment.h"
#include "iree/hal/device.h"
#include "iree/hal/net/channel_peer.h"
#include "iree/hal/net/channel_transfer.h"

typedef struct iree_hal_net_channel_t iree_hal_net_channel_t;

// Stable bootstrap target, including before an accepted peer identifies itself.
typedef struct iree_hal_net_channel_connection_t {
  // Borrowed rank owner of this callback storage.
  iree_hal_net_channel_t* channel;
  // Canonical peer bound by local preparation or validated remote bootstrap.
  iree_hal_net_channel_peer_t* peer;
  // Owned session reference, retained until rank destruction.
  iree_net_session_t* session;
  // Accepted connection retained only if session adoption fails.
  iree_net_connection_t* accepted;
  // Expected remote rank for outgoing connections, or -1 for an accept.
  int32_t expected_rank;
  // Session or accepted-connection callbacks have not retired.
  bool active;
} iree_hal_net_channel_connection_t;

struct iree_hal_net_channel_t {
  // Public rank resource retained by prepared transfers.
  iree_hal_resource_t resource;
  // Allocator owning this channel and its cold metadata.
  iree_allocator_t host_allocator;
  // Retained local device qualifying queue and memory contracts.
  iree_hal_device_t* device;
  // Local queue families admitted by every prepared peer.
  iree_hal_queue_family_affinity_t queue_family_affinity;
  // Retained transport factory shared by this group's connections.
  iree_net_transport_factory_t* factory;
  // Retained externally driven progress service.
  iree_async_proactor_t* proactor;
  // Owned immutable membership, contacts and tag geometry.
  iree_hal_net_channel_options_t* options;
  // Encoded group identity, copied into each session's bootstrap.
  iree_byte_span_t metadata;
  // Owned peer state indexed by remote rank, allocated during cold setup.
  iree_hal_net_channel_peer_t** peers;
  // Stable callback slots and bounded accept accounting for this rank.
  struct {
    // Owned slots; incoming entries are assigned in accept order.
    iree_hal_net_channel_connection_t* entries;
    // Number of accepted connections from higher ranks.
    int32_t accepted_count;
    // Number of connections whose callbacks have not retired.
    iree_host_size_t active_count;
  } connections;
  // Number of peer retirement receipts still owed by allocated peer state.
  iree_host_size_t pending_peer_count;
  // Single listening endpoint and its native stop receipt.
  struct {
    // Owned listener accepting every higher rank.
    iree_net_listener_t* handle;
    // Stop has been requested successfully.
    bool stopping;
    // Stop has returned all native and callback borrows.
    bool stopped;
  } listener;
  // Protects cross-thread admission closure and terminal status.
  iree_slim_mutex_t mutex;
  // First terminal failure and subsequent drain errors.
  iree_status_t status;
  // Admission closed by failure or explicit deactivation.
  bool closing;
  // Explicit deactivation has captured its final receipt.
  bool deactivation_requested;
  // Group shutdown has visited all peers and incomplete handshakes.
  bool drain_started;
  // Guards recursive progress and final receipt during a service walk.
  bool servicing;
  // The cross-thread shutdown operation still borrows this object.
  bool shutdown_pending;
  // All channel-owned callbacks have retired and the receipt was published.
  bool drained;
  // NOP marshaling cross-thread deactivation onto the poll owner.
  iree_async_nop_operation_t shutdown_operation;
  // Caller-owned receipt target captured by accepted deactivation.
  iree_hal_channel_deactivated_callback_t deactivated_callback;
};

// Called with the rank mutex held. Publish admission closure and its terminal
// state together: native progress may run before the owner's shutdown NOP.
// Queue submissions need no second lock or group lookup.
static void iree_hal_net_channel_close_admission(
    iree_hal_net_channel_t* channel) {
  channel->closing = true;
  for (int32_t rank = 0; rank < channel->options->count; ++rank) {
    iree_hal_net_channel_peer_t* peer = channel->peers[rank];
    if (peer) {
      iree_slim_mutex_lock(&peer->mutex);
      peer->closing = true;
      if (iree_status_is_ok(peer->status)) {
        peer->status = iree_status_clone(channel->status);
      }
      iree_slim_mutex_unlock(&peer->mutex);
    }
  }
}

void iree_hal_net_channel_fail(iree_hal_channel_t* base_channel,
                               iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return;
  }
  iree_hal_net_channel_t* channel = (iree_hal_net_channel_t*)base_channel;
  iree_slim_mutex_lock(&channel->mutex);
  if (iree_status_is_ok(channel->status) &&
      !(channel->closing && iree_status_is_cancelled(status))) {
    channel->status = status;
  } else {
    iree_status_free(status);
  }
  iree_hal_net_channel_close_admission(channel);
  iree_slim_mutex_unlock(&channel->mutex);
}

static void iree_hal_net_channel_try_finish(iree_hal_net_channel_t* channel) {
  iree_slim_mutex_lock(&channel->mutex);
  const bool finished =
      channel->deactivation_requested && channel->drain_started &&
      !channel->servicing && !channel->shutdown_pending &&
      (!channel->listener.handle || channel->listener.stopped) &&
      !channel->connections.active_count && !channel->pending_peer_count &&
      !channel->drained;
  iree_status_t status = iree_ok_status();
  iree_hal_channel_deactivated_callback_t callback = {0};
  if (finished) {
    channel->drained = true;
    status = iree_status_clone(channel->status);
    callback = channel->deactivated_callback;
  }
  iree_slim_mutex_unlock(&channel->mutex);
  if (finished) {
    callback.fn(callback.user_data, status);
  }
}

static void iree_hal_net_channel_on_listener_stopped(void* user_data) {
  iree_hal_net_channel_t* channel = user_data;
  channel->listener.stopped = true;
  iree_hal_net_channel_try_finish(channel);
}

static void iree_hal_net_channel_stop_listener(
    iree_hal_net_channel_t* channel) {
  if (!channel->listener.handle || channel->listener.stopping) {
    return;
  }
  iree_status_t status = iree_net_listener_stop(
      channel->listener.handle,
      (iree_net_listener_stopped_callback_t){
          iree_hal_net_channel_on_listener_stopped, channel});
  if (iree_status_is_ok(status)) {
    channel->listener.stopping = true;
  }
  iree_hal_net_channel_fail((iree_hal_channel_t*)channel, status);
}

static void iree_hal_net_channel_on_peer_drained(void* user_data,
                                                 iree_status_t status) {
  iree_hal_net_channel_t* channel = user_data;
  --channel->pending_peer_count;
  iree_slim_mutex_lock(&channel->mutex);
  channel->status = iree_status_join(channel->status, status);
  iree_slim_mutex_unlock(&channel->mutex);
  iree_hal_net_channel_try_finish(channel);
}

void iree_hal_net_channel_service(iree_hal_channel_t* base_channel) {
  iree_hal_net_channel_t* channel = (iree_hal_net_channel_t*)base_channel;
  if (channel->servicing) {
    return;
  }
  channel->servicing = true;
  iree_slim_mutex_lock(&channel->mutex);
  const bool closing = channel->closing;
  iree_slim_mutex_unlock(&channel->mutex);
  if (closing && !channel->drain_started) {
    channel->drain_started = true;
    iree_hal_net_channel_stop_listener(channel);
    for (int32_t rank = 0; rank < channel->options->count; ++rank) {
      iree_hal_net_channel_peer_t* peer = channel->peers[rank];
      if (peer && !peer->drained) {
        iree_hal_net_channel_peer_service(peer);
      }
      iree_hal_net_channel_connection_t* connection =
          &channel->connections.entries[rank];
      if (connection->active && !connection->peer && connection->session) {
        iree_net_session_deactivate(connection->session);
      }
    }
  }
  channel->servicing = false;
  iree_hal_net_channel_try_finish(channel);
}

// Cold setup only. Both local preparation and accepted bootstrap converge on
// this canonical peer before any endpoint or transfer can reference it.
static iree_status_t iree_hal_net_channel_ensure_peer(
    iree_hal_net_channel_t* channel, int32_t rank,
    iree_hal_net_channel_peer_t** out_peer) {
  iree_hal_net_channel_peer_t** slot = &channel->peers[rank];
  if (!*slot) {
    const iree_hal_net_channel_peer_options_t options = {
        .rank = channel->options->rank,
        .peer_rank = rank,
        .tag_capacity = channel->options->tag_capacity,
        .address = rank < channel->options->rank
                       ? channel->options->peer_addresses.values[rank]
                       : channel->options->address};
    IREE_RETURN_IF_ERROR(iree_hal_net_channel_peer_allocate(
        (iree_hal_channel_t*)channel, channel->device,
        channel->queue_family_affinity, channel->proactor, &options,
        (iree_hal_channel_deactivated_callback_t){
            iree_hal_net_channel_on_peer_drained, channel},
        channel->host_allocator, slot));
    ++channel->pending_peer_count;
  }
  *out_peer = *slot;
  return iree_ok_status();
}

static void iree_hal_net_channel_on_ready(
    void* user_data, iree_net_session_t* session,
    const iree_net_bootstrap_peer_info_view_t* remote,
    iree_net_bootstrap_capabilities_t capabilities) {
  iree_hal_net_channel_connection_t* connection = user_data;
  iree_hal_net_channel_t* channel = connection->channel;
  const iree_const_byte_span_t metadata = remote->application_data;
  int32_t rank = -1;
  iree_status_t status = iree_ok_status();
  if (metadata.data_length != channel->metadata.data_length ||
      memcmp(metadata.data, channel->metadata.data, 4) ||
      memcmp(metadata.data + 8, channel->metadata.data + 8,
             metadata.data_length - 8)) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "channel identity or preparation geometry differs");
  } else {
    rank = (int32_t)iree_unaligned_load_le_u32(metadata.data + 4);
    if (rank < 0 || rank >= channel->options->count ||
        (connection->expected_rank >= 0 ? rank != connection->expected_rank
                                        : rank <= channel->options->rank)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unexpected channel peer rank %d", rank);
    }
  }
  iree_slim_mutex_lock(&channel->mutex);
  const bool closing = channel->closing;
  iree_hal_net_channel_peer_t* peer = NULL;
  if (iree_status_is_ok(status) && !closing) {
    status = iree_hal_net_channel_ensure_peer(channel, rank, &peer);
    if (iree_status_is_ok(status)) {
      if (peer->session && peer->session != session) {
        status = iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                                  "duplicate channel peer rank %d", rank);
      } else {
        connection->peer = peer;
        peer->session = session;
        peer->session_drained = false;
      }
    }
  }
  iree_slim_mutex_unlock(&channel->mutex);
  if (iree_status_is_ok(status) && peer) {
    iree_hal_net_channel_peer_open(peer);
  }
  iree_hal_net_channel_fail((iree_hal_channel_t*)channel, status);
  if (closing) {
    iree_net_session_deactivate(session);
  }
  iree_hal_net_channel_service((iree_hal_channel_t*)channel);
}

static iree_status_t iree_hal_net_channel_on_control(
    void* user_data, iree_net_session_t* session,
    iree_net_control_data_flags_t flags, iree_const_byte_span_t payload,
    iree_async_buffer_lease_t* lease) {
  return iree_make_status(IREE_STATUS_DATA_LOSS,
                          "queue channels do not define control data messages");
}

static void iree_hal_net_channel_on_goaway(void* user_data,
                                           iree_net_session_t* session,
                                           uint32_t reason) {
  iree_hal_net_channel_connection_t* connection = user_data;
  iree_hal_net_channel_t* channel = connection->channel;
  iree_slim_mutex_lock(&channel->mutex);
  iree_hal_net_channel_close_admission(channel);
  iree_slim_mutex_unlock(&channel->mutex);
  if (connection->peer) {
    iree_hal_net_channel_peer_on_goaway(connection->peer, reason);
  } else {
    iree_net_session_deactivate(session);
  }
}

static void iree_hal_net_channel_on_error(void* user_data,
                                          iree_net_session_t* session,
                                          iree_status_t status) {
  iree_hal_net_channel_connection_t* connection = user_data;
  iree_hal_net_channel_fail((iree_hal_channel_t*)connection->channel, status);
  iree_hal_net_channel_service((iree_hal_channel_t*)connection->channel);
}

static void iree_hal_net_channel_connection_retired(
    iree_hal_net_channel_connection_t* connection) {
  iree_hal_net_channel_t* channel = connection->channel;
  connection->active = false;
  --channel->connections.active_count;
  // The last peer receipt may return the caller's rank reference. Hold the
  // callback target until the connection's own final service has returned.
  iree_hal_channel_retain((iree_hal_channel_t*)channel);
  if (connection->peer) {
    connection->peer->session_drained = true;
    iree_hal_net_channel_peer_try_finish(connection->peer);
  }
  iree_hal_net_channel_try_finish(channel);
  iree_hal_channel_release((iree_hal_channel_t*)channel);
}

static void iree_hal_net_channel_on_session_drained(
    void* user_data, iree_net_session_t* session) {
  iree_hal_net_channel_connection_retired(user_data);
}

static void iree_hal_net_channel_on_accepted_drained(void* user_data) {
  iree_hal_net_channel_connection_retired(user_data);
}

static iree_net_session_callbacks_t iree_hal_net_channel_callbacks(
    iree_hal_net_channel_connection_t* connection) {
  return (iree_net_session_callbacks_t){
      .on_ready = iree_hal_net_channel_on_ready,
      .on_control_data = iree_hal_net_channel_on_control,
      .on_goaway = iree_hal_net_channel_on_goaway,
      .on_error = iree_hal_net_channel_on_error,
      .on_deactivated = iree_hal_net_channel_on_session_drained,
      .user_data = connection};
}

static iree_net_session_options_t iree_hal_net_channel_session_options(
    iree_hal_net_channel_t* channel) {
  iree_net_session_options_t options = iree_net_session_options_default();
  options.local_peer.application_endpoint_count = 1;
  options.local_peer.application_data = iree_make_const_byte_span(
      channel->metadata.data, channel->metadata.data_length);
  return options;
}

static iree_status_t iree_hal_net_channel_connect_peer(
    iree_hal_net_channel_t* channel, iree_hal_net_channel_peer_t* peer) {
  if (peer->session || peer->options.peer_rank > channel->options->rank) {
    return iree_ok_status();
  }
  iree_hal_net_channel_connection_t* connection =
      &channel->connections.entries[peer->options.peer_rank];
  connection->peer = peer;
  const iree_net_session_options_t options =
      iree_hal_net_channel_session_options(channel);
  iree_status_t status = iree_net_session_connect(
      channel->factory, peer->options.address, channel->proactor, NULL,
      &options, iree_hal_net_channel_callbacks(connection),
      channel->host_allocator, &connection->session);
  if (iree_status_is_ok(status)) {
    connection->active = true;
    ++channel->connections.active_count;
    peer->session = connection->session;
    peer->session_drained = false;
  }
  return status;
}

static void iree_hal_net_channel_on_accept(void* user_data,
                                           iree_status_t status,
                                           iree_net_connection_t* accepted) {
  iree_hal_net_channel_t* channel = user_data;
  if (iree_status_is_ok(status)) {
    // Listener stop on the poll owner prevents another successful publication
    // after the fixed incoming budget has been consumed.
    const int32_t index =
        channel->options->rank + 1 + channel->connections.accepted_count++;
    iree_hal_net_channel_connection_t* connection =
        &channel->connections.entries[index];
    connection->expected_rank = -1;
    connection->active = true;
    connection->accepted = accepted;
    ++channel->connections.active_count;
    if (index + 1 == channel->options->count) {
      iree_hal_net_channel_stop_listener(channel);
    }
    iree_slim_mutex_lock(&channel->mutex);
    const bool closing = channel->closing;
    iree_slim_mutex_unlock(&channel->mutex);
    if (!closing) {
      const iree_net_session_options_t options =
          iree_hal_net_channel_session_options(channel);
      status = iree_net_session_accept(
          accepted, &options, iree_hal_net_channel_callbacks(connection),
          channel->host_allocator, &connection->session);
    }
    if (!closing && iree_status_is_ok(status)) {
      iree_net_connection_release(connection->accepted);
      connection->accepted = NULL;
    } else {
      iree_net_connection_deactivate(
          accepted, (iree_net_connection_deactivate_callback_t){
                        iree_hal_net_channel_on_accepted_drained, connection});
    }
  }
  iree_hal_net_channel_fail((iree_hal_channel_t*)channel, status);
  iree_hal_net_channel_service((iree_hal_channel_t*)channel);
}

static void iree_hal_net_channel_destroy(iree_hal_channel_t* base_channel) {
  iree_hal_net_channel_t* channel = (iree_hal_net_channel_t*)base_channel;
  // Network callbacks borrow rank storage through explicit deactivation.
  IREE_ASSERT(channel->drained);
  if (channel->options) {
    for (int32_t rank = 0; rank < channel->options->count; ++rank) {
      if (channel->peers[rank]) {
        iree_hal_net_channel_peer_free(channel->peers[rank]);
      }
      iree_net_session_release(channel->connections.entries[rank].session);
      iree_net_connection_release(channel->connections.entries[rank].accepted);
    }
  }
  iree_net_listener_free(channel->listener.handle);
  iree_net_transport_factory_release(channel->factory);
  iree_async_proactor_release(channel->proactor);
  iree_hal_device_release(channel->device);
  iree_status_free(channel->status);
  iree_slim_mutex_deinitialize(&channel->mutex);
  iree_allocator_free(channel->host_allocator, channel->options);
  iree_allocator_free(channel->host_allocator, channel->metadata.data);
  iree_allocator_free(channel->host_allocator, channel);
}

static iree_status_t iree_hal_net_channel_split(
    iree_hal_channel_t* base_channel, int32_t color, int32_t key,
    iree_hal_channel_flags_t flags, iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "net channel split requires subgroup rendezvous");
}

static void iree_hal_net_channel_query_rank_and_count(
    const iree_hal_channel_t* base_channel, int32_t* out_rank,
    int32_t* out_count) {
  const iree_hal_net_channel_t* channel =
      (const iree_hal_net_channel_t*)base_channel;
  *out_rank = channel->options->rank;
  *out_count = channel->options->count;
}

static iree_status_t iree_hal_net_channel_prepare(
    iree_hal_channel_t* base_channel,
    const iree_hal_channel_transfer_params_t* params,
    iree_hal_channel_transfer_t** out_transfer) {
  iree_hal_net_channel_t* channel = (iree_hal_net_channel_t*)base_channel;
  iree_slim_mutex_lock(&channel->mutex);
  iree_status_t status =
      channel->closing
          ? (iree_status_is_ok(channel->status)
                 ? iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                    "channel admission is closed")
                 : iree_status_clone(channel->status))
          : iree_ok_status();
  iree_hal_net_channel_peer_t* peer = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_net_channel_ensure_peer(channel, params->peer_rank, &peer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_net_channel_connect_peer(channel, peer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_net_channel_prepare_transfer(peer, params, out_transfer);
  }
  iree_slim_mutex_unlock(&channel->mutex);
  return status;
}

static void iree_hal_net_channel_on_shutdown(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_hal_net_channel_t* channel = user_data;
  iree_slim_mutex_lock(&channel->mutex);
  channel->shutdown_pending = false;
  channel->status = iree_status_join(channel->status, status);
  iree_slim_mutex_unlock(&channel->mutex);
  iree_hal_net_channel_service((iree_hal_channel_t*)channel);
}

static iree_status_t iree_hal_net_channel_deactivate(
    iree_hal_channel_t* base_channel, iree_status_t reason,
    iree_hal_channel_deactivated_callback_t callback) {
  iree_hal_net_channel_t* channel = (iree_hal_net_channel_t*)base_channel;
  iree_slim_mutex_lock(&channel->mutex);
  iree_status_t status = iree_ok_status();
  if (channel->deactivation_requested) {
    status = iree_status_join(
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "channel deactivation already requested"),
        reason);
  } else {
    channel->shutdown_operation.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
    channel->shutdown_operation.base.completion_fn =
        iree_hal_net_channel_on_shutdown;
    channel->shutdown_operation.base.user_data = channel;
    status = iree_async_proactor_submit_one(channel->proactor,
                                            &channel->shutdown_operation.base);
    if (iree_status_is_ok(status)) {
      channel->deactivation_requested = true;
      channel->shutdown_pending = true;
      channel->deactivated_callback = callback;
      channel->status = iree_status_join(channel->status, reason);
      iree_hal_net_channel_close_admission(channel);
    } else {
      status = iree_status_join(status, reason);
    }
  }
  iree_slim_mutex_unlock(&channel->mutex);
  return status;
}

static const iree_hal_channel_vtable_t iree_hal_net_channel_vtable = {
    .destroy = iree_hal_net_channel_destroy,
    .split = iree_hal_net_channel_split,
    .query_rank_and_count = iree_hal_net_channel_query_rank_and_count,
    .prepare_transfer = iree_hal_net_channel_prepare,
    .deactivate = iree_hal_net_channel_deactivate,
};

bool iree_hal_net_channel_isa(const iree_hal_channel_t* channel) {
  return iree_hal_resource_is(channel, &iree_hal_net_channel_vtable);
}

iree_status_t iree_hal_net_channel_create(
    iree_hal_device_t* device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_channel_params_t params, iree_net_transport_factory_t* factory,
    iree_async_proactor_t* proactor,
    const iree_hal_net_channel_options_t* options,
    iree_allocator_t host_allocator, iree_hal_channel_t** out_channel) {
  *out_channel = NULL;
  if (!params.group.size || params.group.size > UINT32_MAX ||
      params.id.data_length > UINT32_MAX || !params.group.data ||
      (params.id.data_length && !params.id.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid channel group or identifier storage");
  }
  iree_host_size_t metadata_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      24, &metadata_length, IREE_STRUCT_FIELD_FAM(params.group.size, char),
      IREE_STRUCT_FIELD_FAM(params.id.data_length, uint8_t)));
  iree_host_size_t allocation_size = 0;
  iree_host_size_t peers_offset = 0;
  iree_host_size_t connections_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_net_channel_t), &allocation_size,
      IREE_STRUCT_FIELD(options->count, iree_hal_net_channel_peer_t*,
                        &peers_offset),
      IREE_STRUCT_FIELD(options->count, iree_hal_net_channel_connection_t,
                        &connections_offset)));
  iree_hal_net_channel_t* channel = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, allocation_size, (void**)&channel));
  iree_hal_resource_initialize(&iree_hal_net_channel_vtable,
                               &channel->resource);
  channel->host_allocator = host_allocator;
  channel->device = device;
  iree_hal_device_retain(device);
  channel->queue_family_affinity = queue_family_affinity;
  channel->factory = factory;
  iree_net_transport_factory_retain(factory);
  channel->proactor = proactor;
  iree_async_proactor_retain(proactor);
  channel->peers =
      (iree_hal_net_channel_peer_t**)((uint8_t*)channel + peers_offset);
  channel->connections.entries =
      (iree_hal_net_channel_connection_t*)((uint8_t*)channel +
                                           connections_offset);
  iree_slim_mutex_initialize(&channel->mutex);
  iree_status_t status = iree_hal_net_channel_options_clone(
      options, host_allocator, &channel->options);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, metadata_length,
                                   (void**)&channel->metadata.data);
  }
  if (iree_status_is_ok(status)) {
    channel->metadata.data_length = metadata_length;
    uint8_t* metadata = channel->metadata.data;
    iree_unaligned_store_le_u32(metadata, 0x48414C02u);
    iree_unaligned_store_le_u32(metadata + 4, params.rank);
    iree_unaligned_store_le_u32(metadata + 8, params.count);
    iree_unaligned_store_le_u32(metadata + 12, options->tag_capacity);
    iree_unaligned_store_le_u32(metadata + 16, params.group.size);
    iree_unaligned_store_le_u32(metadata + 20, params.id.data_length);
    memcpy(metadata + 24, params.group.data, params.group.size);
    if (params.id.data_length) {
      memcpy(metadata + 24 + params.group.size, params.id.data,
             params.id.data_length);
    }
    for (int32_t rank = 0; rank < options->count; ++rank) {
      channel->connections.entries[rank].channel = channel;
      channel->connections.entries[rank].expected_rank = rank;
    }
    if (options->rank + 1 < options->count) {
      status = iree_net_transport_factory_create_listener(
          factory, options->address, proactor, NULL,
          (iree_net_listener_accept_callback_t){iree_hal_net_channel_on_accept,
                                                channel},
          host_allocator, &channel->listener.handle);
    }
  }
  if (iree_status_is_ok(status) && options->rank > 0) {
    iree_hal_net_channel_peer_t* peer = NULL;
    iree_status_t connection_status = iree_hal_net_channel_ensure_peer(
        channel, (options->rank - 1) / 2, &peer);
    if (iree_status_is_ok(connection_status)) {
      connection_status = iree_hal_net_channel_connect_peer(channel, peer);
    }
    iree_hal_net_channel_fail((iree_hal_channel_t*)channel, connection_status);
    iree_hal_net_channel_service((iree_hal_channel_t*)channel);
  }
  if (iree_status_is_ok(status)) {
    *out_channel = (iree_hal_channel_t*)channel;
  } else {
    channel->drained = true;
    iree_hal_channel_release((iree_hal_channel_t*)channel);
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_net_channel_query_address(
    iree_hal_channel_t* base_channel, iree_host_size_t capacity, char* storage,
    iree_string_view_t* out_address) {
  *out_address = iree_string_view_empty();
  if (!iree_hal_net_channel_isa(base_channel)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "channel was not created by the net provider");
  }
  iree_hal_net_channel_t* channel = (iree_hal_net_channel_t*)base_channel;
  if (!channel->listener.handle) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "channel has no listening address");
  }
  return iree_net_listener_query_bound_address(channel->listener.handle,
                                               capacity, storage, out_address);
}
