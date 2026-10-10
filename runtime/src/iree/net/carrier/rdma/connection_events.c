// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/connection_events.h"

#include <fcntl.h>

typedef enum iree_net_rdma_connection_events_flag_bits_e {
  IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_FAILED = 1u << 0,
  IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_STOPPING = 1u << 1,
  IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_STOPPED = 1u << 2,
  IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_PROGRESS = 1u << 3,
} iree_net_rdma_connection_events_flag_bits_t;
typedef uint32_t iree_net_rdma_connection_events_flags_t;

struct iree_net_rdma_connection_events_t {
  // Allocator for this service.
  iree_allocator_t host_allocator;
  // Retained native library/device owner.
  iree_net_rdma_context_t* context;
  // Retained poll owner dispatching all callbacks.
  iree_async_proactor_t* proactor;
  // Owned channel borrowed by explicit connection/listener ID owners.
  struct rdma_event_channel* channel;
  // Native fd monitor, joined before channel destruction.
  iree_async_event_source_t* monitor;
  // Ready-only continuation while a bounded visit leaves possible records.
  iree_async_progress_entry_t progress;
  // Poll-owner-only terminal and monitor lifetime state.
  iree_net_rdma_connection_events_flags_t flags;
  // Maximum native attempts per event/progress service visit.
  uint32_t service_batch_size;
  // Borrowed owner callbacks, valid through deactivation completion.
  iree_net_rdma_connection_events_callbacks_t callbacks;
  // Published before monitor retirement starts.
  iree_async_event_source_unregistered_callback_t deactivated_callback;
};

static void iree_net_rdma_connection_events_fail(
    iree_net_rdma_connection_events_t* events, iree_status_t status) {
  events->flags |= IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_FAILED;
  events->callbacks.on_error(events->callbacks.user_data, status);
}

// Returns true only when the visit budget was exhausted before EAGAIN. Native
// readiness may be edge-triggered, so unread records require ready progress.
static bool iree_net_rdma_connection_events_dispatch(
    iree_net_rdma_connection_events_t* events, uint32_t* out_completed_count) {
  *out_completed_count = 0;
  const iree_net_rdma_library_t* library =
      iree_net_rdma_context_library(events->context);
  iree_status_t status = iree_ok_status();
  uint32_t attempt_count = 0;
  for (;
       attempt_count < events->service_batch_size && iree_status_is_ok(status);
       ++attempt_count) {
    struct rdma_cm_event* native_event = NULL;
    if (library->rdma_get_cm_event(events->channel, &native_event)) {
      int error = errno;
      if (error == EAGAIN) {
        break;
      }
      if (error == EINTR) {
        continue;
      }
      status = iree_make_status(iree_status_code_from_errno(error),
                                "rdma_get_cm_event: %s", strerror(error));
      continue;
    }
    // RC private_data_len is an 8-bit native field. Capture it before ack
    // frees the record, so callbacks may safely migrate or destroy its ID.
    struct rdma_cm_event snapshot = *native_event;
    uint8_t private_data[UINT8_MAX];
    if (snapshot.param.conn.private_data_len) {
      memcpy(private_data, snapshot.param.conn.private_data,
             snapshot.param.conn.private_data_len);
      snapshot.param.conn.private_data = private_data;
    }
    int error = library->rdma_ack_cm_event(native_event);
    // A non-NULL obtained record has exactly one infallible acknowledgment.
    IREE_ASSERT(error == 0, "CM event acknowledgment violated ownership");
    (void)error;
    events->callbacks.on_event(events->callbacks.user_data, &snapshot);
    ++*out_completed_count;
  }
  bool has_more =
      attempt_count == events->service_batch_size && iree_status_is_ok(status);
  if (!iree_status_is_ok(status)) {
    iree_net_rdma_connection_events_fail(events, status);
  }
  return has_more;
}

static void iree_net_rdma_connection_events_progress_removed(void* user_data) {
  iree_net_rdma_connection_events_t* events = user_data;
  events->flags &= ~IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_PROGRESS;
}

static iree_status_t iree_net_rdma_connection_events_progress(
    void* user_data, iree_host_size_t* out_completed_count) {
  iree_net_rdma_connection_events_t* events = user_data;
  uint32_t count = 0;
  bool has_more = false;
  if (!iree_any_bit_set(events->flags,
                        IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_FAILED)) {
    has_more = iree_net_rdma_connection_events_dispatch(events, &count);
  }
  *out_completed_count = count;
  events->progress.remove_requested = !has_more;
  return iree_ok_status();
}

static void iree_net_rdma_connection_events_ready(
    void* user_data, iree_async_event_source_t* source,
    iree_async_poll_events_t ready) {
  (void)source;
  iree_net_rdma_connection_events_t* events = user_data;
  if (iree_any_bit_set(events->flags,
                       IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_FAILED)) {
    return;
  }
  if (iree_async_poll_has_error(ready)) {
    iree_net_rdma_connection_events_fail(
        events, iree_make_status(IREE_STATUS_UNAVAILABLE,
                                 "RDMA connection event channel failed"));
    return;
  }
  uint32_t count = 0;
  if (iree_net_rdma_connection_events_dispatch(events, &count) &&
      !iree_any_bit_set(events->flags,
                        IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_PROGRESS)) {
    events->flags |= IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_PROGRESS;
    iree_async_proactor_register_progress(events->proactor, &events->progress);
  }
}

static void iree_net_rdma_connection_events_free(
    iree_net_rdma_connection_events_t* events) {
  if (events->channel) {
    iree_net_rdma_context_library(events->context)
        ->rdma_destroy_event_channel(events->channel);
  }
  iree_net_rdma_context_release(events->context);
  iree_async_proactor_release(events->proactor);
  iree_allocator_free(events->host_allocator, events);
}

iree_status_t iree_net_rdma_connection_events_create(
    iree_net_rdma_context_t* context, iree_async_proactor_t* proactor,
    uint32_t service_batch_size,
    iree_net_rdma_connection_events_callbacks_t callbacks,
    iree_allocator_t host_allocator,
    iree_net_rdma_connection_events_t** out_events) {
  *out_events = NULL;
  if (!context || !proactor || !service_batch_size || !callbacks.on_event ||
      !callbacks.on_error) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid RDMA connection event configuration");
  }
  iree_net_rdma_connection_events_t* events = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*events), (void**)&events));
  events->host_allocator = host_allocator;
  events->context = context;
  iree_net_rdma_context_retain(context);
  events->proactor = proactor;
  iree_async_proactor_retain(proactor);
  events->service_batch_size = service_batch_size;
  events->callbacks = callbacks;
  events->progress.fn = iree_net_rdma_connection_events_progress;
  events->progress.on_remove = iree_net_rdma_connection_events_progress_removed;
  events->progress.user_data = events;
  events->channel =
      iree_net_rdma_context_library(context)->rdma_create_event_channel();
  iree_status_t status = iree_ok_status();
  if (!events->channel) {
    int error = errno;
    status = iree_make_status(iree_status_code_from_errno(error),
                              "rdma_create_event_channel: %s", strerror(error));
  }
  if (iree_status_is_ok(status)) {
    int flags = fcntl(events->channel->fd, F_GETFL);
    if (flags < 0 || fcntl(events->channel->fd, F_SETFL, flags | O_NONBLOCK)) {
      int error = errno;
      status = iree_make_status(iree_status_code_from_errno(error),
                                "RDMA connection channel flags: %s",
                                strerror(error));
    }
  }
  if (iree_status_is_ok(status)) {
    *out_events = events;
  } else {
    iree_net_rdma_connection_events_free(events);
  }
  return status;
}

iree_status_t iree_net_rdma_connection_events_activate(
    iree_net_rdma_connection_events_t* events) {
  if (events->monitor ||
      iree_any_bit_set(events->flags,
                       IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_STOPPING |
                           IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_STOPPED)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "RDMA connection events already active or retired");
  }
  return iree_async_proactor_register_event_source(
      events->proactor, iree_async_primitive_from_fd(events->channel->fd),
      (iree_async_event_source_callback_t){
          .fn = iree_net_rdma_connection_events_ready,
          .user_data = events,
      },
      &events->monitor);
}

struct rdma_event_channel* iree_net_rdma_connection_events_handle(
    iree_net_rdma_connection_events_t* events) {
  return events->channel;
}

static void iree_net_rdma_connection_events_unregistered(void* user_data,
                                                         iree_status_t status) {
  iree_net_rdma_connection_events_t* events = user_data;
  if (!iree_status_is_ok(status)) {
    iree_net_rdma_connection_events_fail(events, status);
    return;
  }
  events->monitor = NULL;
  events->flags &= ~IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_STOPPING;
  events->flags |= IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_STOPPED;
  iree_async_event_source_unregistered_callback_t callback =
      events->deactivated_callback;
  if (callback.fn) {
    callback.fn(callback.user_data, iree_ok_status());
  }
}

void iree_net_rdma_connection_events_deactivate(
    iree_net_rdma_connection_events_t* events,
    iree_async_event_source_unregistered_callback_t callback) {
  IREE_ASSERT(!iree_any_bit_set(
      events->flags, IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_STOPPING |
                         IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_STOPPED));
  events->flags |= IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_STOPPING;
  events->deactivated_callback = callback;
  if (iree_any_bit_set(events->flags,
                       IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_PROGRESS)) {
    iree_async_proactor_unregister_progress(events->proactor,
                                            &events->progress);
    events->flags &= ~IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_PROGRESS;
  }
  if (events->monitor) {
    iree_async_proactor_unregister_event_source(
        events->proactor, events->monitor,
        (iree_async_event_source_unregistered_callback_t){
            .fn = iree_net_rdma_connection_events_unregistered,
            .user_data = events,
        });
  } else {
    iree_net_rdma_connection_events_unregistered(events, iree_ok_status());
  }
}

void iree_net_rdma_connection_events_destroy(
    iree_net_rdma_connection_events_t* events) {
  if (!events) {
    return;
  }
  IREE_ASSERT(
      !events->monitor &&
          !iree_any_bit_set(events->flags,
                            IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_STOPPING |
                                IREE_NET_RDMA_CONNECTION_EVENTS_FLAG_PROGRESS),
      "active CM monitoring must join before destruction");
  iree_net_rdma_connection_events_free(events);
}
