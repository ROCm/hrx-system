// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/listener.h"

#include <netinet/in.h>

#include "iree/async/operations/scheduling.h"
#include "iree/base/alignment.h"
#include "iree/net/carrier/rdma/connection_events.h"

typedef struct iree_net_rdma_listener_t iree_net_rdma_listener_t;

typedef enum iree_net_rdma_listener_state_e {
  IREE_NET_RDMA_LISTENER_LISTENING = 0,
  IREE_NET_RDMA_LISTENER_FAILED,
  IREE_NET_RDMA_LISTENER_STOPPING,
  IREE_NET_RDMA_LISTENER_STOPPED,
} iree_net_rdma_listener_state_t;

typedef enum iree_net_rdma_listener_slot_state_e {
  IREE_NET_RDMA_LISTENER_SLOT_IDLE = 0,
  IREE_NET_RDMA_LISTENER_SLOT_HANDSHAKING,
  IREE_NET_RDMA_LISTENER_SLOT_CANCELLING,
  IREE_NET_RDMA_LISTENER_SLOT_DRAINING,
} iree_net_rdma_listener_slot_state_t;

enum iree_net_rdma_listener_flag_bits_e {
  IREE_NET_RDMA_LISTENER_PROGRESS_QUEUED = 1u << 0,
  IREE_NET_RDMA_LISTENER_EVENTS_DRAINING = 1u << 1,
  IREE_NET_RDMA_LISTENER_EVENTS_DRAINED = 1u << 2,
};
typedef uint32_t iree_net_rdma_listener_flags_t;

typedef struct iree_net_rdma_listener_slot_t {
  // Containing listener, kept alive by this unfinished accept.
  iree_net_rdma_listener_t* listener;
  // Poll-owner-only phase; IDLE owns neither pointer below.
  iree_net_rdma_listener_slot_state_t state;
  // Borrowed unpublished attempt until its exactly-once setup result.
  iree_net_rdma_connection_t* candidate;
  // Owned published result privately drained when stop wins acceptance.
  iree_net_connection_t* result;
} iree_net_rdma_listener_slot_t;

struct iree_net_rdma_listener_t {
  // Public listener, freed only after the stopped callback.
  iree_net_listener_t base;
  // Allocator for listener and incoming connection construction.
  iree_allocator_t host_allocator;
  // Retained registration/device owner.
  iree_net_rdma_context_t* context;
  // Retained executor for every native and user callback.
  iree_async_proactor_t* proactor;
  // Immutable per-connection options already validated by the factory.
  iree_net_rdma_connection_options_t connection_options;
  // Serializes stop admission and owner-handoff accounting.
  iree_slim_mutex_t mutex;
  // Admission/lifetime state protected by mutex.
  iree_net_rdma_listener_state_t state;
  // Coalesced handoff and native monitoring lifetime obligations.
  iree_net_rdma_listener_flags_t flags;
  // Queued/executing progress callback bodies, protected by mutex.
  uint32_t pending_progress_count;
  // Preallocated infallible stop/error handoff.
  iree_async_nop_operation_t progress;
  // Native bound ID, destroyed on the poll owner before monitor retirement.
  struct rdma_cm_id* id;
  // CM service; never owns published connection IDs.
  iree_net_rdma_connection_events_t* events;
  // Immutable bound address survives native listener retirement.
  iree_async_address_t address;
  // Public accept observer, joined by native monitoring and occupied slots.
  iree_net_listener_accept_callback_t accepted_callback;
  // Public final-stop observer, installed by the sole accepted stop call.
  iree_net_listener_stopped_callback_t stopped_callback;
  // Fixed bound on unfinished incoming handshakes.
  uint32_t slot_count;
  // Poll-owner-only admission slots, never a published-connection registry.
  iree_net_rdma_listener_slot_t slots[];
};

static iree_status_t iree_net_rdma_listener_native_error(
    const char* operation) {
  int error = errno;
  return iree_make_status(iree_status_code_from_errno(error), "%s: %s",
                          operation, strerror(error));
}

static bool iree_net_rdma_listener_is_listening(
    iree_net_rdma_listener_t* listener) {
  iree_slim_mutex_lock(&listener->mutex);
  bool listening = listener->state == IREE_NET_RDMA_LISTENER_LISTENING;
  iree_slim_mutex_unlock(&listener->mutex);
  return listening;
}

static void iree_net_rdma_listener_schedule_locked(
    iree_net_rdma_listener_t* listener) {
  if (iree_any_bit_set(listener->flags,
                       IREE_NET_RDMA_LISTENER_PROGRESS_QUEUED)) {
    return;
  }
  listener->flags |= IREE_NET_RDMA_LISTENER_PROGRESS_QUEUED;
  ++listener->pending_progress_count;
  iree_async_operation_t* operation = &listener->progress.base;
  iree_async_operation_initialize(operation, IREE_ASYNC_OPERATION_TYPE_NOP, 0,
                                  operation->completion_fn, listener);
  IREE_CHECK_OK(iree_async_proactor_submit_one(listener->proactor, operation));
}

// Final access: the callback may free the listener, even from another callback.
static void iree_net_rdma_listener_try_finish(
    iree_net_rdma_listener_t* listener) {
  for (uint32_t i = 0; i < listener->slot_count; ++i) {
    if (listener->slots[i].state != IREE_NET_RDMA_LISTENER_SLOT_IDLE) {
      return;
    }
  }
  iree_slim_mutex_lock(&listener->mutex);
  iree_net_listener_stopped_callback_t callback = {0};
  if (listener->state == IREE_NET_RDMA_LISTENER_STOPPING &&
      !listener->pending_progress_count &&
      iree_any_bit_set(listener->flags,
                       IREE_NET_RDMA_LISTENER_EVENTS_DRAINED)) {
    listener->state = IREE_NET_RDMA_LISTENER_STOPPED;
    callback = listener->stopped_callback;
  }
  iree_slim_mutex_unlock(&listener->mutex);
  if (callback.fn) {
    callback.fn(callback.user_data);
  }
}

static void iree_net_rdma_listener_result_drained(void* user_data) {
  iree_net_rdma_listener_slot_t* slot = user_data;
  iree_net_connection_release(slot->result);
  slot->result = NULL;
  slot->state = IREE_NET_RDMA_LISTENER_SLOT_IDLE;
  iree_net_rdma_listener_try_finish(slot->listener);
}

static void iree_net_rdma_listener_setup_done(
    void* user_data, iree_status_t status, iree_net_connection_t* connection) {
  iree_net_rdma_listener_slot_t* slot = user_data;
  iree_net_rdma_listener_t* listener = slot->listener;
  slot->candidate = NULL;
  if (!iree_net_rdma_listener_is_listening(listener) && connection) {
    slot->result = connection;
    slot->state = IREE_NET_RDMA_LISTENER_SLOT_DRAINING;
    iree_net_connection_deactivate(
        connection, (iree_net_connection_deactivate_callback_t){
                        iree_net_rdma_listener_result_drained, slot});
    return;
  }
  bool cancelled_by_stop =
      slot->state == IREE_NET_RDMA_LISTENER_SLOT_CANCELLING &&
      iree_status_is_cancelled(status);
  slot->state = IREE_NET_RDMA_LISTENER_SLOT_IDLE;
  if (cancelled_by_stop) {
    iree_status_free(status);
  } else {
    listener->accepted_callback.fn(listener->accepted_callback.user_data,
                                   status, connection);
  }
  iree_net_rdma_listener_try_finish(listener);
}

static void iree_net_rdma_listener_failed(void* user_data,
                                          iree_status_t status) {
  iree_net_rdma_listener_t* listener = user_data;
  iree_slim_mutex_lock(&listener->mutex);
  if (listener->state == IREE_NET_RDMA_LISTENER_LISTENING) {
    listener->state = IREE_NET_RDMA_LISTENER_FAILED;
    iree_net_rdma_listener_schedule_locked(listener);
  }
  iree_slim_mutex_unlock(&listener->mutex);
  listener->accepted_callback.fn(listener->accepted_callback.user_data, status,
                                 NULL);
}

static void iree_net_rdma_listener_destroy_id(
    iree_net_rdma_listener_t* listener, struct rdma_cm_id* id) {
  if (iree_net_rdma_context_library(listener->context)->rdma_destroy_id(id)) {
    iree_status_abort(
        iree_net_rdma_listener_native_error("destroying RDMA listener ID"));
  }
}

static void iree_net_rdma_listener_event(void* user_data,
                                         const struct rdma_cm_event* event) {
  iree_net_rdma_listener_t* listener = user_data;
  if (event->event != RDMA_CM_EVENT_CONNECT_REQUEST) {
    iree_net_rdma_listener_failed(
        listener, iree_make_status(IREE_STATUS_UNAVAILABLE,
                                   "RDMA listener event %d, status %d",
                                   event->event, event->status));
    return;
  }
  bool listening = iree_net_rdma_listener_is_listening(listener);
  iree_net_rdma_listener_slot_t* slot = NULL;
  if (listening) {
    for (uint32_t i = 0; i < listener->slot_count; ++i) {
      if (listener->slots[i].state == IREE_NET_RDMA_LISTENER_SLOT_IDLE) {
        slot = &listener->slots[i];
        break;
      }
    }
  }
  iree_status_t status = iree_ok_status();
  if (slot) {
    status = iree_net_rdma_connection_create(
        listener->context, listener->proactor, &listener->connection_options,
        listener->host_allocator, &slot->candidate);
  }
  if (slot && iree_status_is_ok(status)) {
    slot->state = IREE_NET_RDMA_LISTENER_SLOT_HANDSHAKING;
    iree_net_rdma_connection_accept(
        slot->candidate, event->id,
        iree_make_const_byte_span(event->param.conn.private_data,
                                  event->param.conn.private_data_len),
        (iree_net_transport_connect_callback_t){
            iree_net_rdma_listener_setup_done, slot});
  } else {
    // Obtained requests are acknowledged before this callback. Undelivered
    // requests remain kernel-owned and retire with the listener ID.
    if (iree_net_rdma_context_library(listener->context)
            ->rdma_reject(event->id, NULL, 0)) {
      status = iree_status_join(
          status, iree_net_rdma_listener_native_error("rdma_reject"));
    }
    iree_net_rdma_listener_destroy_id(listener, event->id);
    if (iree_status_is_ok(status) && listening && !slot) {
      status = iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    if (!iree_status_is_ok(status)) {
      listener->accepted_callback.fn(listener->accepted_callback.user_data,
                                     status, NULL);
    }
  }
}

static void iree_net_rdma_listener_events_drained(void* user_data,
                                                  iree_status_t status) {
  iree_net_rdma_listener_t* listener = user_data;
  if (!iree_status_is_ok(status)) {
    iree_net_rdma_listener_failed(listener, status);
    return;
  }
  iree_slim_mutex_lock(&listener->mutex);
  listener->flags |= IREE_NET_RDMA_LISTENER_EVENTS_DRAINED;
  iree_slim_mutex_unlock(&listener->mutex);
  iree_net_rdma_listener_try_finish(listener);
}

static void iree_net_rdma_listener_progress(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_net_rdma_listener_t* listener = user_data;
  if (!iree_status_is_ok(status)) {
    iree_net_rdma_listener_failed(listener, status);
  }
  iree_slim_mutex_lock(&listener->mutex);
  listener->flags &= ~IREE_NET_RDMA_LISTENER_PROGRESS_QUEUED;
  bool retire = !iree_any_bit_set(listener->flags,
                                  IREE_NET_RDMA_LISTENER_EVENTS_DRAINING);
  listener->flags |= IREE_NET_RDMA_LISTENER_EVENTS_DRAINING;
  iree_slim_mutex_unlock(&listener->mutex);
  if (retire) {
    iree_net_rdma_listener_destroy_id(listener, listener->id);
    listener->id = NULL;
    iree_net_rdma_connection_events_deactivate(
        listener->events, (iree_async_event_source_unregistered_callback_t){
                              iree_net_rdma_listener_events_drained, listener});
  }
  for (uint32_t i = 0; i < listener->slot_count; ++i) {
    iree_net_rdma_listener_slot_t* slot = &listener->slots[i];
    if (slot->state == IREE_NET_RDMA_LISTENER_SLOT_HANDSHAKING) {
      slot->state = IREE_NET_RDMA_LISTENER_SLOT_CANCELLING;
      iree_net_rdma_connection_cancel(slot->candidate);
    }
  }
  iree_slim_mutex_lock(&listener->mutex);
  --listener->pending_progress_count;
  iree_slim_mutex_unlock(&listener->mutex);
  iree_net_rdma_listener_try_finish(listener);
}

static void iree_net_rdma_listener_free(iree_net_listener_t* base) {
  iree_net_rdma_listener_t* listener = (iree_net_rdma_listener_t*)base;
  IREE_ASSERT(listener->state == IREE_NET_RDMA_LISTENER_STOPPED,
              "RDMA listener freed before callback retirement");
  iree_net_rdma_connection_events_destroy(listener->events);
  iree_slim_mutex_deinitialize(&listener->mutex);
  iree_async_proactor_release(listener->proactor);
  iree_net_rdma_context_release(listener->context);
  iree_allocator_free(listener->host_allocator, listener);
}

static iree_status_t iree_net_rdma_listener_stop(
    iree_net_listener_t* base, iree_net_listener_stopped_callback_t callback) {
  iree_net_rdma_listener_t* listener = (iree_net_rdma_listener_t*)base;
  iree_slim_mutex_lock(&listener->mutex);
  iree_status_t status = iree_ok_status();
  if (listener->state >= IREE_NET_RDMA_LISTENER_STOPPING) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "RDMA listener stop already requested");
  } else {
    listener->state = IREE_NET_RDMA_LISTENER_STOPPING;
    listener->stopped_callback = callback;
    iree_net_rdma_listener_schedule_locked(listener);
  }
  iree_slim_mutex_unlock(&listener->mutex);
  return status;
}

static iree_status_t iree_net_rdma_listener_query_bound_address(
    iree_net_listener_t* base, iree_host_size_t buffer_capacity, char* buffer,
    iree_string_view_t* out_address) {
  return iree_async_address_format(&((iree_net_rdma_listener_t*)base)->address,
                                   buffer_capacity, buffer, out_address);
}

static const iree_net_listener_vtable_t iree_net_rdma_listener_vtable = {
    .free = iree_net_rdma_listener_free,
    .stop = iree_net_rdma_listener_stop,
    .query_bound_address = iree_net_rdma_listener_query_bound_address,
};

iree_status_t iree_net_rdma_listener_create(
    iree_net_rdma_context_t* context, iree_async_proactor_t* proactor,
    const iree_async_address_t* address,
    const iree_net_rdma_connection_options_t* connection_options,
    uint32_t max_pending_connections, uint32_t listen_backlog,
    iree_net_listener_accept_callback_t callback,
    iree_allocator_t host_allocator, iree_net_listener_t** out_listener) {
  *out_listener = NULL;
  iree_host_size_t allocation_size = 0;
  IREE_RETURN_IF_ERROR(
      IREE_STRUCT_LAYOUT(sizeof(iree_net_rdma_listener_t), &allocation_size,
                         IREE_STRUCT_FIELD_FAM(max_pending_connections,
                                               iree_net_rdma_listener_slot_t)));
  iree_net_rdma_listener_t* listener = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, allocation_size,
                                             (void**)&listener));
  listener->base.vtable = &iree_net_rdma_listener_vtable;
  listener->host_allocator = host_allocator;
  listener->context = context;
  iree_net_rdma_context_retain(context);
  listener->proactor = proactor;
  iree_async_proactor_retain(proactor);
  listener->connection_options = *connection_options;
  iree_slim_mutex_initialize(&listener->mutex);
  listener->progress.base.completion_fn = iree_net_rdma_listener_progress;
  listener->accepted_callback = callback;
  listener->slot_count = max_pending_connections;
  for (uint32_t i = 0; i < max_pending_connections; ++i) {
    listener->slots[i].listener = listener;
  }
  const iree_net_rdma_library_t* library =
      iree_net_rdma_context_library(context);
  iree_status_t status = iree_net_rdma_connection_events_create(
      context, proactor, connection_options->control.service_batch_size,
      (iree_net_rdma_connection_events_callbacks_t){
          iree_net_rdma_listener_event, iree_net_rdma_listener_failed,
          listener},
      host_allocator, &listener->events);
  if (iree_status_is_ok(status) &&
      library->rdma_create_id(
          iree_net_rdma_connection_events_handle(listener->events),
          &listener->id, listener, RDMA_PS_TCP)) {
    status = iree_net_rdma_listener_native_error("rdma_create_id");
  }
  if (iree_status_is_ok(status) &&
      library->rdma_bind_addr(listener->id,
                              (struct sockaddr*)address->storage)) {
    status = iree_net_rdma_listener_native_error("rdma_bind_addr");
  }
  if (iree_status_is_ok(status) &&
      library->rdma_listen(listener->id, (int)listen_backlog)) {
    status = iree_net_rdma_listener_native_error("rdma_listen");
  }
  if (iree_status_is_ok(status)) {
    struct sockaddr* bound = rdma_get_local_addr(listener->id);
    listener->address.length = bound->sa_family == AF_INET
                                   ? sizeof(struct sockaddr_in)
                                   : sizeof(struct sockaddr_in6);
    memcpy(listener->address.storage, bound, listener->address.length);
    status = iree_net_rdma_connection_events_activate(listener->events);
  }
  if (iree_status_is_ok(status)) {
    *out_listener = &listener->base;
  } else {
    if (listener->id) {
      iree_net_rdma_listener_destroy_id(listener, listener->id);
    }
    listener->state = IREE_NET_RDMA_LISTENER_STOPPED;
    iree_net_rdma_listener_free(&listener->base);
  }
  return status;
}
