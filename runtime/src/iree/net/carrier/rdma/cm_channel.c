// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/cm_channel.h"

#include <errno.h>
#include <string.h>

#include "iree/net/carrier/rdma/pollable_fd.h"

struct iree_net_rdma_cm_channel_t {
  // Borrowed dynamically loaded librdmacm symbol table.
  const iree_net_librdmacm_t* librdmacm;

  // Proactor monitoring event_channel->fd. Retained by the channel.
  iree_async_proactor_t* proactor;

  // Native rdma_cm event channel owned by this wrapper.
  struct rdma_event_channel* event_channel;

  // Proactor event source registered for event_channel->fd.
  iree_async_event_source_t* event_source;

  // Event/error callback invoked from proactor poll.
  iree_net_rdma_cm_channel_callback_t callback;

  // Host allocator used for this channel allocation.
  iree_allocator_t host_allocator;
};

static bool iree_net_rdma_cm_event_has_connection_private_data(
    enum rdma_cm_event_type event_type) {
  switch (event_type) {
    case RDMA_CM_EVENT_CONNECT_REQUEST:
    case RDMA_CM_EVENT_CONNECT_RESPONSE:
    case RDMA_CM_EVENT_REJECTED:
    case RDMA_CM_EVENT_ESTABLISHED:
      return true;
    default:
      return false;
  }
}

static iree_net_rdma_cm_event_t iree_net_rdma_cm_event_view(
    struct rdma_cm_event* native_event) {
  iree_net_rdma_cm_event_t event;
  memset(&event, 0, sizeof(event));
  event.native_event = native_event;
  event.id = native_event->id;
  event.listen_id = native_event->listen_id;
  event.type = native_event->event;
  event.status = native_event->status;
  if (iree_net_rdma_cm_event_has_connection_private_data(event.type)) {
    event.private_data =
        iree_make_const_byte_span(native_event->param.conn.private_data,
                                  native_event->param.conn.private_data_len);
  }
  return event;
}

IREE_API_EXPORT iree_status_t
iree_net_rdma_cm_channel_drain(iree_net_rdma_cm_channel_t* channel) {
  if (!channel) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "channel must not be NULL");
  }

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    struct rdma_cm_event* native_event = NULL;
    int result = channel->librdmacm->rdma_get_cm_event(channel->event_channel,
                                                       &native_event);
    if (result != 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) break;
      status = iree_net_rdma_pollable_fd_status_from_errno(
          __FILE__, __LINE__, errno, "rdma_get_cm_event");
      break;
    }

    iree_net_rdma_cm_event_t event = iree_net_rdma_cm_event_view(native_event);
    channel->callback.fn(channel->callback.user_data, iree_ok_status(), &event);

    result = channel->librdmacm->rdma_ack_cm_event(native_event);
    if (result != 0) {
      status = iree_net_rdma_pollable_fd_status_from_errno(
          __FILE__, __LINE__, errno, "rdma_ack_cm_event");
    }
  }

  return status;
}

static void iree_net_rdma_cm_channel_on_event_source(
    void* user_data, iree_async_event_source_t* source,
    iree_async_poll_events_t events) {
  (void)source;
  iree_net_rdma_cm_channel_t* channel = (iree_net_rdma_cm_channel_t*)user_data;

  iree_status_t status = iree_ok_status();
  if (events & IREE_ASYNC_POLL_EVENT_IN) {
    status = iree_net_rdma_cm_channel_drain(channel);
  }
  if (iree_status_is_ok(status) && iree_async_poll_has_error(events)) {
    status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "rdma_cm event channel closed or failed");
  }

  if (!iree_status_is_ok(status)) {
    channel->callback.fn(channel->callback.user_data, status, NULL);
  }
}

IREE_API_EXPORT iree_status_t iree_net_rdma_cm_channel_create(
    const iree_net_librdmacm_t* librdmacm, iree_async_proactor_t* proactor,
    iree_net_rdma_cm_channel_callback_t callback,
    iree_allocator_t host_allocator, iree_net_rdma_cm_channel_t** out_channel) {
  if (!out_channel) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_channel must not be NULL");
  }
  *out_channel = NULL;
  if (!librdmacm) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "librdmacm must not be NULL");
  }
  if (!proactor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "proactor must not be NULL");
  }
  if (!callback.fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "callback.fn must not be NULL");
  }

  iree_net_rdma_cm_channel_t* channel = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*channel), (void**)&channel);

  if (iree_status_is_ok(status)) {
    memset(channel, 0, sizeof(*channel));
    channel->librdmacm = librdmacm;
    channel->proactor = proactor;
    iree_async_proactor_retain(proactor);
    channel->callback = callback;
    channel->host_allocator = host_allocator;

    channel->event_channel = librdmacm->rdma_create_event_channel();
    if (!channel->event_channel) {
      status = iree_net_rdma_pollable_fd_status_from_errno(
          __FILE__, __LINE__, errno, "rdma_create_event_channel");
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_pollable_fd_initialize(channel->event_channel->fd);
  }

  if (iree_status_is_ok(status)) {
    iree_async_event_source_callback_t event_callback = {
        iree_net_rdma_cm_channel_on_event_source,
        channel,
    };
    status = iree_async_proactor_register_event_source(
        proactor, iree_async_primitive_from_fd(channel->event_channel->fd),
        event_callback, &channel->event_source);
  }

  if (iree_status_is_ok(status)) {
    *out_channel = channel;
  } else if (channel) {
    iree_net_rdma_cm_channel_release(channel);
  }
  return status;
}

IREE_API_EXPORT void iree_net_rdma_cm_channel_release(
    iree_net_rdma_cm_channel_t* channel) {
  if (!channel) return;

  if (channel->event_source) {
    iree_async_proactor_unregister_event_source(channel->proactor,
                                                channel->event_source);
  }
  if (channel->event_channel) {
    channel->librdmacm->rdma_destroy_event_channel(channel->event_channel);
  }
  iree_async_proactor_release(channel->proactor);

  iree_allocator_t host_allocator = channel->host_allocator;
  iree_allocator_free(host_allocator, channel);
}

IREE_API_EXPORT struct rdma_event_channel*
iree_net_rdma_cm_channel_native_event_channel(
    const iree_net_rdma_cm_channel_t* channel) {
  return channel ? channel->event_channel : NULL;
}
