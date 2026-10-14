// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/net/channel_peer.h"

#include "iree/base/alignment.h"
#include "iree/hal/device.h"
#include "iree/hal/net/channel.h"
#include "iree/hal/net/channel_transfer.h"

void iree_hal_net_channel_peer_fail(iree_hal_net_channel_peer_t* peer,
                                    iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return;
  }
  iree_slim_mutex_lock(&peer->mutex);
  if (iree_status_is_ok(peer->status) &&
      !(peer->closing && iree_status_is_cancelled(status))) {
    peer->status = status;
  } else {
    // Preserve the first failure. Expected cancellations during drain are
    // receipts; graceful close must still preserve a later genuine failure.
    iree_status_free(status);
  }
  peer->closing = true;
  iree_status_t failure = iree_status_clone(peer->status);
  iree_slim_mutex_unlock(&peer->mutex);
  iree_hal_net_channel_fail(peer->owner, failure);
}

void iree_hal_net_channel_peer_try_finish(iree_hal_net_channel_peer_t* peer) {
  iree_slim_mutex_lock(&peer->mutex);
  bool finished = peer->drain_started && !peer->servicing &&
                  !peer->goaway_pending && peer->session_drained &&
                  !peer->pending_count && !peer->drained;
  if (finished) {
    peer->drained = true;
  }
  iree_slim_mutex_unlock(&peer->mutex);
  if (!finished) {
    return;
  }
  iree_slim_mutex_lock(&peer->mutex);
  iree_status_t status = iree_status_clone(peer->status);
  iree_hal_channel_deactivated_callback_t callback = peer->deactivated_callback;
  iree_slim_mutex_unlock(&peer->mutex);
  callback.fn(callback.user_data, status);
}

static void iree_hal_net_channel_peer_on_goaway_sent(void* user_data,
                                                     iree_status_t status,
                                                     iree_host_size_t bytes) {
  iree_hal_net_channel_peer_t* peer = user_data;
  peer->goaway_pending = false;
  iree_slim_mutex_lock(&peer->mutex);
  peer->status = iree_status_join(peer->status, status);
  iree_slim_mutex_unlock(&peer->mutex);
  iree_net_session_deactivate(peer->session);
}

void iree_hal_net_channel_peer_service(iree_hal_net_channel_peer_t* peer) {
  if (peer->servicing) {
    return;
  }
  peer->servicing = true;
  iree_slim_mutex_lock(&peer->mutex);
  bool closing = peer->closing;
  iree_slim_mutex_unlock(&peer->mutex);
  if (!closing && peer->ready) {
    iree_hal_net_channel_transfers_service(peer);
  }
  iree_slim_mutex_lock(&peer->mutex);
  closing = peer->closing;
  iree_slim_mutex_unlock(&peer->mutex);
  if (closing && !peer->drain_started) {
    peer->drain_started = true;
    iree_slim_mutex_lock(&peer->mutex);
    const bool graceful = peer->ready && !peer->peer_closed &&
                          !peer->pending_count &&
                          iree_status_is_ok(peer->status);
    iree_slim_mutex_unlock(&peer->mutex);
    iree_hal_net_channel_transfers_abort(peer);
    if (graceful) {
      iree_status_t status = iree_net_session_send_goaway(
          peer->session, 0,
          (iree_net_send_completion_callback_t){
              .fn = iree_hal_net_channel_peer_on_goaway_sent,
              .user_data = peer});
      if (iree_status_is_ok(status)) {
        peer->goaway_pending = true;
      } else {
        iree_slim_mutex_lock(&peer->mutex);
        peer->status = iree_status_join(peer->status, status);
        iree_slim_mutex_unlock(&peer->mutex);
        iree_net_session_deactivate(peer->session);
      }
    } else if (peer->session) {
      iree_net_session_deactivate(peer->session);
    }
  }
  if (closing) {
    iree_hal_net_channel_service(peer->owner);
  }
  peer->servicing = false;
  iree_hal_net_channel_peer_try_finish(peer);
}

static void iree_hal_net_channel_peer_on_error(void* user_data,
                                               iree_status_t status) {
  iree_hal_net_channel_peer_t* peer = user_data;
  iree_hal_net_channel_peer_fail(peer, status);
  iree_hal_net_channel_peer_service(peer);
}

static void iree_hal_net_channel_peer_on_capacity(void* user_data) {
  iree_hal_net_channel_peer_service(user_data);
}

static void iree_hal_net_channel_peer_on_message_ready(
    void* user_data, iree_status_t status,
    iree_net_message_endpoint_t endpoint) {
  iree_hal_net_channel_peer_t* peer = user_data;
  if (iree_status_is_ok(status)) {
    status = iree_net_queue_channel_allocate(
        endpoint,
        (iree_net_queue_channel_callbacks_t){
            .on_capacity = iree_hal_net_channel_peer_on_capacity,
            .on_command = iree_hal_net_channel_on_command,
            .on_advance = iree_hal_net_channel_on_advance,
            .on_error = iree_hal_net_channel_peer_on_error,
            .user_data = peer},
        peer->host_allocator, &peer->protocol);
  }
  if (iree_status_is_ok(status)) {
    iree_net_queue_channel_attach(peer->protocol);
    status = iree_net_message_endpoint_activate(endpoint);
  }
  peer->ready = iree_status_is_ok(status);
  iree_hal_net_channel_peer_on_error(peer, status);
}

void iree_hal_net_channel_peer_open(iree_hal_net_channel_peer_t* peer) {
  iree_status_t status = iree_net_session_open_endpoint(
      peer->session,
      (iree_net_endpoint_ready_callback_t){
          .fn = iree_hal_net_channel_peer_on_message_ready, .user_data = peer});
  iree_hal_net_channel_peer_on_error(peer, status);
}

void iree_hal_net_channel_peer_on_goaway(iree_hal_net_channel_peer_t* peer,
                                         uint32_t reason) {
  peer->peer_closed = true;
  iree_slim_mutex_lock(&peer->mutex);
  bool unfinished = reason != 0;
  for (iree_host_size_t i = 0; i < peer->transfer_count; ++i) {
    const iree_hal_net_channel_transfer_t* transfer = &peer->transfers[i];
    // A peer can finish consuming a payload before its sender receives the
    // native source-return callback. That callback still owns local storage.
    if (transfer->phase != IREE_HAL_NET_CHANNEL_TRANSFER_PHASE_IDLE &&
        !(transfer->phase == IREE_HAL_NET_CHANNEL_TRANSFER_PHASE_ISSUED &&
          transfer->base.params.mode == IREE_HAL_CHANNEL_TRANSFER_MODE_SEND)) {
      unfinished = true;
    }
  }
  if (!unfinished) {
    peer->closing = true;
  }
  iree_slim_mutex_unlock(&peer->mutex);
  if (unfinished) {
    iree_hal_net_channel_peer_fail(
        peer, iree_make_status(IREE_STATUS_ABORTED,
                               "peer closed with unfinished exchanges"));
  }
  iree_hal_net_channel_peer_service(peer);
}

void iree_hal_net_channel_peer_free(iree_hal_net_channel_peer_t* peer) {
  // Background callbacks borrow this object until its captured drain receipt.
  IREE_ASSERT(peer->drained);
  iree_net_queue_channel_free(peer->protocol);
  iree_status_free(peer->status);
  iree_slim_mutex_deinitialize(&peer->mutex);
  iree_allocator_free(peer->host_allocator, peer);
}

iree_status_t iree_hal_net_channel_peer_allocate(
    iree_hal_channel_t* owner, iree_hal_device_t* device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_async_proactor_t* proactor,
    const iree_hal_net_channel_peer_options_t* options,
    iree_hal_channel_deactivated_callback_t callback,
    iree_allocator_t host_allocator, iree_hal_net_channel_peer_t** out_peer) {
  *out_peer = NULL;
  iree_host_size_t transfer_count = 0;
  if (!iree_host_size_checked_mul(options->tag_capacity, 2, &transfer_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "channel slots exceed host address space");
  }
  iree_host_size_t allocation_size = 0;
  iree_host_size_t transfers_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_net_channel_peer_t), &allocation_size,
      IREE_STRUCT_FIELD(transfer_count, iree_hal_net_channel_transfer_t,
                        &transfers_offset)));
  iree_hal_net_channel_peer_t* peer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, allocation_size, (void**)&peer));
  peer->owner = owner;
  peer->host_allocator = host_allocator;
  peer->device = device;
  peer->proactor = proactor;
  peer->queue_family_affinity = queue_family_affinity;
  peer->options = *options;
  peer->deactivated_callback = callback;
  peer->transfer_count = transfer_count;
  peer->transfers =
      (iree_hal_net_channel_transfer_t*)((uint8_t*)peer + transfers_offset);
  peer->session_drained = true;
  iree_slim_mutex_initialize(&peer->mutex);
  for (iree_host_size_t i = 0; i < transfer_count; ++i) {
    peer->transfers[i].peer = peer;
  }
  *out_peer = peer;
  return iree_ok_status();
}
