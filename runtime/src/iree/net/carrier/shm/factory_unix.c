// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cross-process SHM factory operations for POSIX platforms.
//
// Implements listener and connect over Unix domain sockets. Each accepted
// connection runs its handle exchange on a cancellable bootstrap worker, then
// creates an independent SHM carrier pair using the socket as a retained
// descriptor-rights sideband.

#include "iree/net/carrier/shm/factory_state.h"

#if !defined(IREE_PLATFORM_WINDOWS)

#include <string.h>

#include "iree/async/address.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/socket.h"
#include "iree/net/carrier/shm/factory_bootstrap.h"
#include "iree/net/carrier/shm/handshake.h"

//===----------------------------------------------------------------------===//
// Cross-process listener (Unix domain socket)
//===----------------------------------------------------------------------===//

// State machine for the Unix socket accept loop.
typedef enum iree_net_shm_unix_listener_state_e {
  IREE_NET_SHM_UNIX_LISTENER_STATE_LISTENING = 0,
  IREE_NET_SHM_UNIX_LISTENER_STATE_STOPPING,
  IREE_NET_SHM_UNIX_LISTENER_STATE_STOPPED,
} iree_net_shm_unix_listener_state_t;

// Bounds server-side bootstrap threads per listener. Connections beyond this
// limit remain in the kernel listen backlog until a worker completes.
#define IREE_NET_SHM_UNIX_MAX_PENDING_BOOTSTRAPS 16

typedef struct iree_net_shm_unix_listener_t iree_net_shm_unix_listener_t;

// One accepted channel being bootstrapped off the proactor thread.
typedef struct iree_net_shm_unix_pending_bootstrap_t {
  // Listener receiving the terminal bootstrap callback.
  iree_net_shm_unix_listener_t* listener;
  // Prepared worker operation, valid until its callback begins.
  iree_net_shm_bootstrap_t* bootstrap;
  // Intrusive linkage in the listener's cancellation set.
  struct iree_net_shm_unix_pending_bootstrap_t* next;
} iree_net_shm_unix_pending_bootstrap_t;

// Listener backed by a Unix domain stream socket. Each accepted connection
// runs a synchronous handshake to exchange SHM handles and notification
// primitives, then creates an independent SHM carrier pair. The accept loop
// supports both multishot (io_uring) and single-shot with re-arm.
struct iree_net_shm_unix_listener_t {
  iree_net_listener_t base;
  // Factory retained for listener lifetime.
  iree_net_shm_factory_t* factory;
  // Proactor delivering accept and bootstrap completions.
  iree_async_proactor_t* proactor;
  // Socket accepting cross-process channels.
  iree_async_socket_t* listen_socket;
  // Consumer accept callback.
  struct {
    // Function receiving accepted connections and terminal accept errors.
    iree_net_listener_accept_callback_t fn;
    // Opaque value passed to |fn|.
    void* user_data;
  } accept;
  // Single-shot accept operation; at most one is pending.
  iree_async_socket_accept_operation_t accept_operation;
  // Protects lifecycle and pending bootstrap fields from stop callers.
  iree_slim_mutex_t mutex;
  // Current listener lifecycle state.
  iree_net_shm_unix_listener_state_t state;
  // True while |accept_operation| is owned by the proactor.
  bool accept_pending;
  // True after the stopped callback has been claimed for delivery.
  bool stopped_delivered;
  // True while |stop_operation| is owned by the proactor.
  bool stop_operation_pending;
  // Accepted channels currently running bootstrap workers.
  iree_net_shm_unix_pending_bootstrap_t* pending_bootstraps;
  // Number of entries in |pending_bootstraps|.
  iree_host_size_t pending_bootstrap_count;
  // Callback delivered once accept and bootstrap work has drained.
  iree_net_listener_stopped_callback_t stopped_callback;
  // NOP used when stop begins with no asynchronous work to drain.
  iree_async_nop_operation_t stop_operation;
  // Allocator used for listener and pending bootstrap state.
  iree_allocator_t host_allocator;
  // Full address string (e.g., "unix:/tmp/iree.sock") for
  // query_bound_address. Null-terminated.
  iree_host_size_t address_length;
  char address[];
};

static const iree_net_listener_vtable_t iree_net_shm_unix_listener_vtable;

static void iree_net_shm_unix_listener_accept_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags);

static iree_status_t iree_net_shm_unix_listener_rearm(
    iree_net_shm_unix_listener_t* listener);

// Claims and delivers the stopped callback once every accepted channel and the
// kernel accept operation have reached terminal completion. Must run on the
// proactor thread and must be the caller's final access to |listener| because
// the stopped callback may free it.
static void iree_net_shm_unix_listener_maybe_finish_stop(
    iree_net_shm_unix_listener_t* listener) {
  iree_net_listener_stopped_callback_t callback = {0};
  iree_slim_mutex_lock(&listener->mutex);
  if (listener->state == IREE_NET_SHM_UNIX_LISTENER_STATE_STOPPING &&
      !listener->accept_pending && !listener->pending_bootstraps &&
      !listener->stop_operation_pending && !listener->stopped_delivered) {
    listener->stopped_delivered = true;
    listener->state = IREE_NET_SHM_UNIX_LISTENER_STATE_STOPPED;
    callback = listener->stopped_callback;
  }
  iree_slim_mutex_unlock(&listener->mutex);
  if (callback.fn) callback.fn(callback.user_data);
}

static void iree_net_shm_unix_pending_bootstrap_complete(
    void* user_data, iree_status_t status,
    iree_net_shm_bootstrap_completion_flags_t flags,
    iree_net_connection_t* connection) {
  iree_net_shm_unix_pending_bootstrap_t* pending =
      (iree_net_shm_unix_pending_bootstrap_t*)user_data;
  iree_net_shm_unix_listener_t* listener = pending->listener;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&listener->mutex);
  iree_net_shm_unix_pending_bootstrap_t** previous =
      &listener->pending_bootstraps;
  while (*previous && *previous != pending) {
    previous = &(*previous)->next;
  }
  IREE_ASSERT(*previous == pending,
              "completed SHM bootstrap missing from listener");
  *previous = pending->next;
  --listener->pending_bootstrap_count;
  iree_slim_mutex_unlock(&listener->mutex);

  iree_allocator_free(listener->host_allocator, pending);

  if (iree_any_bit_set(flags,
                       IREE_NET_SHM_BOOTSTRAP_COMPLETION_FLAG_CANCELLED)) {
    IREE_ASSERT(!connection);
    if (!iree_status_is_ok(status)) {
      listener->accept.fn(listener->accept.user_data, status, NULL);
    }
  } else {
    listener->accept.fn(listener->accept.user_data, status, connection);
  }

  iree_status_t rearm_status = iree_net_shm_unix_listener_rearm(listener);
  if (!iree_status_is_ok(rearm_status)) {
    listener->accept.fn(listener->accept.user_data, rearm_status, NULL);
  }
  iree_net_shm_unix_listener_maybe_finish_stop(listener);
  IREE_TRACE_ZONE_END(z0);
}

// Prepares and launches the worker for one accepted socket. The listener list
// owns the pending wrapper until terminal completion.
static iree_status_t iree_net_shm_unix_listener_start_bootstrap(
    iree_net_shm_unix_listener_t* listener, iree_async_socket_t* accepted) {
  iree_async_primitive_t channel = iree_async_primitive_none();
  iree_status_t status =
      iree_async_primitive_dup(accepted->primitive, &channel);
  iree_async_socket_release(accepted);

  iree_net_shm_unix_pending_bootstrap_t* pending = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(listener->host_allocator, sizeof(*pending),
                                   (void**)&pending);
  }
  if (iree_status_is_ok(status)) {
    memset(pending, 0, sizeof(*pending));
    pending->listener = listener;
    status = iree_net_shm_bootstrap_prepare(
        listener->factory, IREE_NET_SHM_BOOTSTRAP_ROLE_SERVER, &channel,
        listener->proactor,
        (iree_net_shm_bootstrap_callback_t){
            .fn = iree_net_shm_unix_pending_bootstrap_complete,
            .user_data = pending,
        },
        listener->host_allocator, &pending->bootstrap);
  }

  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&listener->mutex);
    pending->next = listener->pending_bootstraps;
    listener->pending_bootstraps = pending;
    ++listener->pending_bootstrap_count;
    bool cancel_immediately =
        listener->state == IREE_NET_SHM_UNIX_LISTENER_STATE_STOPPING;
    iree_slim_mutex_unlock(&listener->mutex);
    if (cancel_immediately) {
      iree_net_shm_bootstrap_cancel(pending->bootstrap);
    }
    iree_net_shm_bootstrap_launch(pending->bootstrap);
  } else {
    iree_allocator_free(listener->host_allocator, pending);
    iree_async_primitive_close(&channel);
  }
  return status;
}

// Re-arms the single-shot accept operation when bootstrap admission permits.
static iree_status_t iree_net_shm_unix_listener_rearm(
    iree_net_shm_unix_listener_t* listener) {
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&listener->mutex);
  bool can_accept =
      listener->state == IREE_NET_SHM_UNIX_LISTENER_STATE_LISTENING &&
      !listener->accept_pending &&
      listener->pending_bootstrap_count <
          IREE_NET_SHM_UNIX_MAX_PENDING_BOOTSTRAPS;
  if (can_accept) {
    memset(&listener->accept_operation, 0, sizeof(listener->accept_operation));
    iree_async_operation_initialize(
        &listener->accept_operation.base,
        IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT,
        IREE_ASYNC_OPERATION_FLAG_CANCELLATION_IS_SUCCESS,
        iree_net_shm_unix_listener_accept_complete, listener);
    listener->accept_operation.listen_socket = listener->listen_socket;
    status = iree_async_proactor_submit_one(listener->proactor,
                                            &listener->accept_operation.base);
    listener->accept_pending = iree_status_is_ok(status);
  }
  iree_slim_mutex_unlock(&listener->mutex);
  return status;
}

// Accept completion callback for the Unix domain socket listener.
// Accepted channels move to bootstrap workers before the listener re-arms.
static void iree_net_shm_unix_listener_accept_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_shm_unix_listener_t* listener =
      (iree_net_shm_unix_listener_t*)user_data;
  iree_async_socket_accept_operation_t* accept_op =
      (iree_async_socket_accept_operation_t*)operation;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT(!iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE),
              "SHM listener uses single-shot accepts");

  iree_slim_mutex_lock(&listener->mutex);
  IREE_ASSERT(listener->accept_pending);
  listener->accept_pending = false;
  iree_slim_mutex_unlock(&listener->mutex);

  bool was_cancelled =
      iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_CANCELLED);
  if (iree_status_is_ok(status) && !was_cancelled) {
    status = iree_net_shm_unix_listener_start_bootstrap(
        listener, accept_op->accepted_socket);
    accept_op->accepted_socket = NULL;
  } else {
    iree_async_socket_release(accept_op->accepted_socket);
    accept_op->accepted_socket = NULL;
  }

  if (!iree_status_is_ok(status)) {
    listener->accept.fn(listener->accept.user_data, status, NULL);
  }
  iree_status_t rearm_status = iree_net_shm_unix_listener_rearm(listener);
  if (!iree_status_is_ok(rearm_status)) {
    listener->accept.fn(listener->accept.user_data, rearm_status, NULL);
  }
  iree_net_shm_unix_listener_maybe_finish_stop(listener);
  IREE_TRACE_ZONE_END(z0);
}

static void iree_net_shm_unix_listener_stop_deferred_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_shm_unix_listener_t* listener =
      (iree_net_shm_unix_listener_t*)user_data;
  iree_slim_mutex_lock(&listener->mutex);
  IREE_ASSERT(listener->stop_operation_pending);
  listener->stop_operation_pending = false;
  iree_slim_mutex_unlock(&listener->mutex);
  if (!iree_status_is_ok(status)) {
    listener->accept.fn(listener->accept.user_data, status, NULL);
  }
  iree_net_shm_unix_listener_maybe_finish_stop(listener);
}

static void iree_net_shm_unix_listener_free(
    iree_net_listener_t* base_listener) {
  iree_net_shm_unix_listener_t* listener =
      (iree_net_shm_unix_listener_t*)base_listener;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT(listener->stopped_delivered,
              "SHM listener freed before stopped callback");
  IREE_ASSERT(!listener->accept_pending);
  IREE_ASSERT(!listener->pending_bootstraps);
  IREE_ASSERT(!listener->stop_operation_pending);
  iree_async_socket_release(listener->listen_socket);
  iree_async_proactor_release(listener->proactor);
  iree_net_transport_factory_release(&listener->factory->base);
  iree_slim_mutex_deinitialize(&listener->mutex);
  iree_allocator_t host_allocator = listener->host_allocator;
  iree_allocator_free(host_allocator, listener);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_net_shm_unix_listener_stop(
    iree_net_listener_t* base_listener,
    iree_net_listener_stopped_callback_t callback) {
  iree_net_shm_unix_listener_t* listener =
      (iree_net_shm_unix_listener_t*)base_listener;
  IREE_TRACE_ZONE_BEGIN(z0);

  bool cancel_accept = false;
  bool rearm_after_failure = false;
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&listener->mutex);
  if (listener->state != IREE_NET_SHM_UNIX_LISTENER_STATE_LISTENING) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "SHM listener is already stopping");
  } else {
    listener->state = IREE_NET_SHM_UNIX_LISTENER_STATE_STOPPING;
    listener->stopped_callback = callback;
    cancel_accept = listener->accept_pending;
    for (iree_net_shm_unix_pending_bootstrap_t* pending =
             listener->pending_bootstraps;
         pending; pending = pending->next) {
      iree_net_shm_bootstrap_cancel(pending->bootstrap);
    }
    if (!listener->accept_pending && !listener->pending_bootstraps) {
      iree_async_operation_initialize(
          &listener->stop_operation.base, IREE_ASYNC_OPERATION_TYPE_NOP,
          IREE_ASYNC_OPERATION_FLAG_CANCELLATION_IS_SUCCESS,
          iree_net_shm_unix_listener_stop_deferred_complete, listener);
      listener->stop_operation_pending = true;
      status = iree_async_proactor_submit_one(listener->proactor,
                                              &listener->stop_operation.base);
      if (!iree_status_is_ok(status)) {
        listener->stop_operation_pending = false;
        listener->state = IREE_NET_SHM_UNIX_LISTENER_STATE_LISTENING;
        listener->stopped_callback = (iree_net_listener_stopped_callback_t){0};
        rearm_after_failure = true;
      }
    }
  }
  iree_slim_mutex_unlock(&listener->mutex);

  if (cancel_accept) {
    iree_status_t cancel_status = iree_async_proactor_cancel(
        listener->proactor, &listener->accept_operation.base);
    if (iree_status_is_not_found(cancel_status)) {
      // The accept completion won the race and will retire the operation.
      iree_status_free(cancel_status);
    } else {
      status = iree_status_join(status, cancel_status);
    }
  }
  if (rearm_after_failure) {
    status =
        iree_status_join(status, iree_net_shm_unix_listener_rearm(listener));
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_net_shm_unix_listener_query_bound_address(
    iree_net_listener_t* base_listener, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_address) {
  iree_net_shm_unix_listener_t* listener =
      (iree_net_shm_unix_listener_t*)base_listener;
  if (buffer_capacity < listener->address_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffer too small for bound address");
  }
  memcpy(buffer, listener->address, listener->address_length);
  *out_address = iree_make_string_view(buffer, listener->address_length);
  return iree_ok_status();
}

static const iree_net_listener_vtable_t iree_net_shm_unix_listener_vtable = {
    .free = iree_net_shm_unix_listener_free,
    .stop = iree_net_shm_unix_listener_stop,
    .query_bound_address = iree_net_shm_unix_listener_query_bound_address,
};

// Creates a cross-process listener bound to a Unix domain socket path.
// Parses the address, creates the socket, binds, listens, and submits the
// initial accept operation.
iree_status_t iree_net_shm_factory_create_listener_unix(
    iree_net_shm_factory_t* factory, iree_string_view_t bind_address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_listener_accept_callback_t accept_callback, void* user_data,
    iree_allocator_t host_allocator, iree_net_listener_t** out_listener) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)recv_pool;
  *out_listener = NULL;

  // Parse the Unix domain socket address.
  iree_async_address_t address;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_address_from_string(bind_address, &address));

  // Create Unix domain stream socket.
  iree_async_socket_t* listen_socket = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_async_socket_create(proactor, IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM,
                               IREE_ASYNC_SOCKET_OPTION_NONE, &listen_socket));

  // Bind and listen.
  iree_status_t status = iree_async_socket_bind(listen_socket, &address);
  if (iree_status_is_ok(status)) {
    status = iree_async_socket_listen(listen_socket, /*backlog=*/128);
  }

  // Allocate listener with space for the address string (null-terminated).
  iree_host_size_t total_size = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        iree_sizeof_struct(iree_net_shm_unix_listener_t), &total_size,
        IREE_STRUCT_FIELD_FAM(bind_address.size + 1, char));
  }
  iree_net_shm_unix_listener_t* listener = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&listener);
  }
  if (iree_status_is_ok(status)) {
    memset(listener, 0, total_size);
    listener->base.vtable = &iree_net_shm_unix_listener_vtable;
    listener->factory = factory;
    iree_net_transport_factory_retain(&factory->base);
    listener->proactor = proactor;
    iree_async_proactor_retain(proactor);
    listener->listen_socket = listen_socket;
    listener->accept.fn = accept_callback;
    listener->accept.user_data = user_data;
    iree_slim_mutex_initialize(&listener->mutex);
    listener->host_allocator = host_allocator;
    listener->address_length = bind_address.size;
    memcpy(listener->address, bind_address.data, bind_address.size);
    listener->address[bind_address.size] = '\0';
  }

  // Submit one accept at a time so bootstrap admission remains bounded.
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_unix_listener_rearm(listener);
  }

  if (iree_status_is_ok(status)) {
    *out_listener = &listener->base;
  } else {
    if (listener) {
      iree_slim_mutex_deinitialize(&listener->mutex);
      iree_async_proactor_release(listener->proactor);
      iree_net_transport_factory_release(&listener->factory->base);
      iree_allocator_free(host_allocator, listener);
    }
    iree_async_socket_release(listen_socket);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Cross-process connect (Unix domain socket)
//===----------------------------------------------------------------------===//

// Heap-allocated state spanning socket connect and worker bootstrap.
typedef struct iree_net_shm_unix_connect_state_t {
  // Proactor operation establishing the Unix domain socket.
  iree_async_socket_connect_operation_t connect_operation;
  // Consumer connect callback.
  struct {
    // Function receiving the terminal connection result.
    iree_net_transport_connect_callback_t fn;
    // Opaque value passed to |fn|.
    void* user_data;
  } callback;
  // Factory retained until the terminal callback returns.
  iree_net_shm_factory_t* factory;
  // Proactor used for connect and bootstrap completion.
  iree_async_proactor_t* proactor;
  // Socket owned until its primitive moves to |bootstrap|.
  iree_async_socket_t* socket;
  // Bootstrap operation after socket connection succeeds.
  iree_net_shm_bootstrap_t* bootstrap;
  // Allocator used for this state and the bootstrap.
  iree_allocator_t host_allocator;
} iree_net_shm_unix_connect_state_t;

static void iree_net_shm_unix_connect_state_destroy(
    iree_net_shm_unix_connect_state_t* state) {
  iree_async_socket_release(state->socket);
  iree_net_transport_factory_release(&state->factory->base);
  iree_allocator_free(state->host_allocator, state);
}

static void iree_net_shm_unix_connect_bootstrap_complete(
    void* user_data, iree_status_t status,
    iree_net_shm_bootstrap_completion_flags_t flags,
    iree_net_connection_t* connection) {
  iree_net_shm_unix_connect_state_t* state =
      (iree_net_shm_unix_connect_state_t*)user_data;
  state->bootstrap = NULL;
  if (iree_any_bit_set(flags,
                       IREE_NET_SHM_BOOTSTRAP_COMPLETION_FLAG_CANCELLED) &&
      iree_status_is_ok(status)) {
    status = iree_make_status(IREE_STATUS_CANCELLED,
                              "SHM connection bootstrap cancelled");
  }
  state->callback.fn(state->callback.user_data, status, connection);
  iree_net_shm_unix_connect_state_destroy(state);
}

// Moves the connected socket primitive to a client bootstrap worker.
static iree_status_t iree_net_shm_unix_connect_start_bootstrap(
    iree_net_shm_unix_connect_state_t* state) {
  iree_async_primitive_t channel = iree_async_primitive_none();
  iree_status_t status =
      iree_async_primitive_dup(state->socket->primitive, &channel);
  iree_async_socket_release(state->socket);
  state->socket = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_bootstrap_prepare(
        state->factory, IREE_NET_SHM_BOOTSTRAP_ROLE_CLIENT, &channel,
        state->proactor,
        (iree_net_shm_bootstrap_callback_t){
            .fn = iree_net_shm_unix_connect_bootstrap_complete,
            .user_data = state,
        },
        state->host_allocator, &state->bootstrap);
  }
  if (iree_status_is_ok(status)) {
    iree_net_shm_bootstrap_launch(state->bootstrap);
  } else {
    iree_async_primitive_close(&channel);
  }
  return status;
}

// Connect completion callback for cross-process SHM connect. On successful
// connect, moves the channel to a bootstrap worker.
static void iree_net_shm_unix_connect_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_shm_unix_connect_state_t* state =
      (iree_net_shm_unix_connect_state_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (iree_status_is_ok(status)) {
    status = iree_net_shm_unix_connect_start_bootstrap(state);
  } else {
    iree_async_socket_release(state->socket);
    state->socket = NULL;
  }
  if (!iree_status_is_ok(status)) {
    state->callback.fn(state->callback.user_data, status, NULL);
    iree_net_shm_unix_connect_state_destroy(state);
  }
  IREE_TRACE_ZONE_END(z0);
}

// Initiates a cross-process connect to a Unix domain socket path. Parses the
// address, creates a socket, and submits an async connect operation. On
// completion, the handshake runs and the carrier is created.
iree_status_t iree_net_shm_factory_connect_unix(
    iree_net_shm_factory_t* factory, iree_string_view_t address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_transport_connect_callback_t callback, void* user_data) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)recv_pool;

  // Parse the Unix domain socket address.
  iree_async_address_t remote_address;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_address_from_string(address, &remote_address));

  // Create Unix domain stream socket.
  iree_async_socket_t* socket = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_socket_create(proactor, IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM,
                                   IREE_ASYNC_SOCKET_OPTION_NONE, &socket));

  // Allocate connect state.
  iree_net_shm_unix_connect_state_t* state = NULL;
  iree_status_t status = iree_allocator_malloc(factory->host_allocator,
                                               sizeof(*state), (void**)&state);
  if (!iree_status_is_ok(status)) {
    iree_async_socket_release(socket);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  memset(state, 0, sizeof(*state));
  state->callback.fn = callback;
  state->callback.user_data = user_data;
  state->factory = factory;
  iree_net_transport_factory_retain(&factory->base);
  state->proactor = proactor;
  state->socket = socket;
  state->host_allocator = factory->host_allocator;

  // Submit async connect.
  iree_async_operation_initialize(&state->connect_operation.base,
                                  IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  iree_net_shm_unix_connect_complete, state);
  state->connect_operation.socket = socket;
  state->connect_operation.address = remote_address;
  status =
      iree_async_proactor_submit_one(proactor, &state->connect_operation.base);
  if (!iree_status_is_ok(status)) {
    iree_async_socket_release(socket);
    state->socket = NULL;
    iree_net_shm_unix_connect_state_destroy(state);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

#endif  // !IREE_PLATFORM_WINDOWS
