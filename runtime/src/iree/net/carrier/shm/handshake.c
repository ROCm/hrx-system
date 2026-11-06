// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/shm/handshake.h"

#include <string.h>

static void iree_net_shm_handshake_drained(void* user_data) {
  iree_net_shm_handshake_t* handshake = user_data;
  iree_net_shm_connection_channel_deinitialize(&handshake->channel);
  iree_net_connection_release(handshake->connection);
  handshake->connection = NULL;
  iree_net_shm_storage_release(handshake->storage);
  handshake->storage = NULL;
  if (handshake->role == IREE_NET_SHM_HANDSHAKE_ROLE_CLIENT) {
    for (uint32_t i = 0; i < IREE_NET_SHM_STORAGE_HANDLE_COUNT; ++i) {
      iree_async_primitive_close(&handshake->handles[i]);
    }
  }
  handshake->phase = IREE_NET_SHM_HANDSHAKE_PHASE_COMPLETE;
  iree_status_t status = handshake->status;
  handshake->status = iree_ok_status();
  handshake->callback.fn(handshake->callback.user_data, status, NULL);
}

static void iree_net_shm_handshake_fail(iree_net_shm_handshake_t* handshake,
                                        iree_status_t status) {
  handshake->status = iree_status_join(handshake->status, status);
  if (handshake->phase == IREE_NET_SHM_HANDSHAKE_PHASE_DRAINING) {
    return;
  }
  handshake->phase = IREE_NET_SHM_HANDSHAKE_PHASE_DRAINING;
  IREE_CHECK_OK(iree_async_local_stream_deactivate(
      handshake->channel.stream,
      (iree_async_local_stream_deactivated_callback_t){
          .fn = iree_net_shm_handshake_drained,
          .user_data = handshake,
      }));
}

// Checks offer policy without using the tentative imported native resources.
static iree_status_t iree_net_shm_handshake_validate_offer(
    iree_net_shm_handshake_t* handshake) {
  iree_net_shm_region_layout_t* layout = &handshake->offered_layout;
  IREE_RETURN_IF_ERROR(iree_net_shm_bootstrap_decode_offer(
      iree_make_const_byte_span(handshake->record,
                                IREE_NET_SHM_BOOTSTRAP_OFFER_SIZE),
      layout));
  const iree_net_shm_region_options_t* limits = &handshake->limits->options;
  if (layout->options.endpoint_count > limits->endpoint_count ||
      layout->options.slot_count > limits->slot_count ||
      layout->options.slot_capacity > limits->slot_capacity) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "SHM offer (%u endpoints, %u slots, %u bytes/slot) exceeds local "
        "limits "
        "(%u endpoints, %u slots, %u bytes/slot)",
        layout->options.endpoint_count, layout->options.slot_count,
        layout->options.slot_capacity, limits->endpoint_count,
        limits->slot_count, limits->slot_capacity);
  }
  return iree_ok_status();
}

// READY confirms that the exporter held the offered resources through ACCEPT.
// Only now may mapping and notification setup use the imported handles.
static iree_status_t iree_net_shm_handshake_import(
    iree_net_shm_handshake_t* handshake) {
  IREE_RETURN_IF_ERROR(iree_net_shm_storage_import(
      &handshake->offered_layout, handshake->handles, handshake->host_allocator,
      &handshake->storage));
  return iree_net_shm_connection_create(
      handshake->proactor, handshake->storage, handshake->carrier_options,
      handshake->host_allocator, &handshake->connection);
}

static void iree_net_shm_handshake_publish(
    iree_net_shm_handshake_t* handshake) {
  iree_net_connection_t* connection = handshake->connection;
  handshake->connection = NULL;
  iree_net_shm_connection_publish(connection, &handshake->channel);
  iree_net_shm_storage_release(handshake->storage);
  handshake->storage = NULL;
  handshake->phase = IREE_NET_SHM_HANDSHAKE_PHASE_COMPLETE;
  handshake->callback.fn(handshake->callback.user_data, iree_ok_status(),
                         connection);
}

static void iree_net_shm_handshake_advance(void* user_data,
                                           iree_status_t status) {
  iree_net_shm_handshake_t* handshake = user_data;
  if (handshake->phase == IREE_NET_SHM_HANDSHAKE_PHASE_DRAINING) {
    handshake->status = iree_status_join(handshake->status, status);
    return;
  }
  if (!iree_status_is_ok(status)) {
    iree_net_shm_handshake_fail(handshake, status);
    return;
  }
  iree_async_local_stream_callback_t callback = {
      .fn = iree_net_shm_handshake_advance,
      .user_data = handshake,
  };
  iree_async_local_stream_t* stream = handshake->channel.stream;
  switch (handshake->phase) {
    case IREE_NET_SHM_HANDSHAKE_PHASE_SERVER_BEGIN: {
      status = iree_net_shm_storage_create(
          handshake->limits, handshake->host_allocator, &handshake->storage);
      if (iree_status_is_ok(status)) {
        status = iree_net_shm_connection_create(
            handshake->proactor, handshake->storage, handshake->carrier_options,
            handshake->host_allocator, &handshake->connection);
      }
      if (iree_status_is_ok(status)) {
        iree_net_shm_storage_export(handshake->storage, handshake->handles);
        iree_net_shm_bootstrap_encode_offer(handshake->limits,
                                            handshake->record);
        handshake->phase = IREE_NET_SHM_HANDSHAKE_PHASE_SERVER_RECEIVE_ACCEPT;
        status = iree_async_local_stream_send(
            stream,
            iree_make_const_byte_span(handshake->record,
                                      IREE_NET_SHM_BOOTSTRAP_OFFER_SIZE),
            IREE_NET_SHM_STORAGE_HANDLE_COUNT, handshake->handles, callback);
      }
      break;
    }
    case IREE_NET_SHM_HANDSHAKE_PHASE_SERVER_RECEIVE_ACCEPT:
      handshake->phase = IREE_NET_SHM_HANDSHAKE_PHASE_SERVER_SEND_READY;
      status = iree_async_local_stream_receive(
          stream,
          iree_make_byte_span(handshake->record,
                              IREE_NET_SHM_BOOTSTRAP_HEADER_SIZE),
          0, callback, NULL);
      break;
    case IREE_NET_SHM_HANDSHAKE_PHASE_SERVER_SEND_READY:
      status = iree_net_shm_bootstrap_decode_ack(
          iree_make_const_byte_span(handshake->record,
                                    IREE_NET_SHM_BOOTSTRAP_HEADER_SIZE),
          IREE_NET_SHM_BOOTSTRAP_TYPE_ACCEPT);
      if (iree_status_is_ok(status)) {
        iree_net_shm_bootstrap_encode_ack(IREE_NET_SHM_BOOTSTRAP_TYPE_READY,
                                          handshake->record);
        handshake->phase = IREE_NET_SHM_HANDSHAKE_PHASE_PUBLISH;
        status = iree_async_local_stream_send(
            stream,
            iree_make_const_byte_span(handshake->record,
                                      IREE_NET_SHM_BOOTSTRAP_HEADER_SIZE),
            0, NULL, callback);
      }
      break;
    case IREE_NET_SHM_HANDSHAKE_PHASE_CLIENT_BEGIN:
      handshake->phase = IREE_NET_SHM_HANDSHAKE_PHASE_CLIENT_SEND_ACCEPT;
      status = iree_async_local_stream_receive(
          stream,
          iree_make_byte_span(handshake->record,
                              IREE_NET_SHM_BOOTSTRAP_OFFER_SIZE),
          IREE_NET_SHM_STORAGE_HANDLE_COUNT, callback, handshake->handles);
      break;
    case IREE_NET_SHM_HANDSHAKE_PHASE_CLIENT_SEND_ACCEPT:
      status = iree_net_shm_handshake_validate_offer(handshake);
      if (iree_status_is_ok(status)) {
        iree_net_shm_bootstrap_encode_ack(IREE_NET_SHM_BOOTSTRAP_TYPE_ACCEPT,
                                          handshake->record);
        handshake->phase = IREE_NET_SHM_HANDSHAKE_PHASE_CLIENT_RECEIVE_READY;
        status = iree_async_local_stream_send(
            stream,
            iree_make_const_byte_span(handshake->record,
                                      IREE_NET_SHM_BOOTSTRAP_HEADER_SIZE),
            0, NULL, callback);
      }
      break;
    case IREE_NET_SHM_HANDSHAKE_PHASE_CLIENT_RECEIVE_READY:
      handshake->phase = IREE_NET_SHM_HANDSHAKE_PHASE_CLIENT_VALIDATE_READY;
      status = iree_async_local_stream_receive(
          stream,
          iree_make_byte_span(handshake->record,
                              IREE_NET_SHM_BOOTSTRAP_HEADER_SIZE),
          0, callback, NULL);
      break;
    case IREE_NET_SHM_HANDSHAKE_PHASE_CLIENT_VALIDATE_READY:
      status = iree_net_shm_bootstrap_decode_ack(
          iree_make_const_byte_span(handshake->record,
                                    IREE_NET_SHM_BOOTSTRAP_HEADER_SIZE),
          IREE_NET_SHM_BOOTSTRAP_TYPE_READY);
      if (iree_status_is_ok(status)) {
        status = iree_net_shm_handshake_import(handshake);
      }
      if (iree_status_is_ok(status)) {
        iree_net_shm_handshake_publish(handshake);
        return;
      }
      break;
    case IREE_NET_SHM_HANDSHAKE_PHASE_PUBLISH:
      iree_net_shm_handshake_publish(handshake);
      return;
    default:
      IREE_ASSERT_UNREACHABLE(
          "SHM handshake completion without an active transfer");
  }
  if (!iree_status_is_ok(status)) {
    iree_net_shm_handshake_fail(handshake, status);
  }
}

void iree_net_shm_handshake_begin(
    iree_net_shm_handshake_t* handshake, iree_net_shm_handshake_role_t role,
    iree_async_proactor_t* proactor, const iree_net_shm_region_layout_t* limits,
    const iree_net_shm_carrier_options_t* carrier_options,
    iree_net_shm_connection_channel_t* channel,
    iree_net_transport_connect_callback_t callback,
    iree_allocator_t host_allocator) {
  memset(handshake, 0, sizeof(*handshake));
  handshake->role = role;
  handshake->phase = role == IREE_NET_SHM_HANDSHAKE_ROLE_SERVER
                         ? IREE_NET_SHM_HANDSHAKE_PHASE_SERVER_BEGIN
                         : IREE_NET_SHM_HANDSHAKE_PHASE_CLIENT_BEGIN;
  handshake->proactor = proactor;
  handshake->limits = limits;
  handshake->carrier_options = carrier_options;
  handshake->channel = *channel;
  memset(channel, 0, sizeof(*channel));
  handshake->callback = callback;
  handshake->host_allocator = host_allocator;
  iree_net_shm_handshake_advance(handshake, iree_ok_status());
}

void iree_net_shm_handshake_cancel(iree_net_shm_handshake_t* handshake) {
  if (handshake->phase == IREE_NET_SHM_HANDSHAKE_PHASE_DRAINING) {
    return;
  }
  iree_net_shm_handshake_fail(handshake,
                              iree_status_from_code(IREE_STATUS_CANCELLED));
}
