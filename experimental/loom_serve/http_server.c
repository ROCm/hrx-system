// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/http_server.h"

#include <stdio.h>
#include <string.h>

#include "iree/async/api.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/util/proactor_thread.h"
#include "iree/base/threading/mutex.h"
#include "iree/net/carrier/tcp/carrier.h"

enum {
  HTTP_CONNECTION_CAPACITY = 16,
  HTTP_SEND_CAPACITY = 8,
};

typedef enum http_connection_flag_bits_e {
  HTTP_CONNECTION_OCCUPIED = 1u << 0,
  HTTP_CONNECTION_REQUEST_READY = 1u << 1,
  HTTP_CONNECTION_CLAIMED = 1u << 2,
  HTTP_CONNECTION_FINISHED = 1u << 3,
  HTTP_CONNECTION_DEACTIVATING = 1u << 4,
  HTTP_CONNECTION_DRAINED = 1u << 5,
} http_connection_flag_bits_t;
typedef uint32_t http_connection_flags_t;

enum {
  HTTP_ACCEPT_PENDING = 1u << 0,
  HTTP_ACCEPT_CANCEL_PENDING = 1u << 1,
  HTTP_SERVER_STOPPING = 1u << 0,
  HTTP_SERVER_SHUTDOWN_RECEIVED = 1u << 1,
};

struct loom_serve_http_connection_t {
  // Borrowed server outliving all callbacks and application claims.
  loom_serve_http_server_t* server;
  // Ownership and completion facts protected by the server mutex.
  http_connection_flags_t flags;
  // Carrier owns its socket and dedicated registered receive pool.
  iree_net_carrier_t* carrier;
  // Parser owns the complete request bytes through application release.
  loom_serve_http_request_parser_t* parser;
  // Borrowed complete request, published once under the mutex.
  const loom_serve_http_request_t* request;
  // Accepted sends not yet returned by completion callbacks.
  uint32_t pending_sends;
  // Owned terminal peer failures, diagnosed during reclamation.
  iree_status_t failure;
};

struct loom_serve_http_server_t {
  // Allocator owning this server and its connection resources.
  iree_allocator_t allocator;
  // Protects peer publication, ownership, send counts and shutdown facts.
  iree_slim_mutex_t mutex;
  // Advisory wakeup for the single application owner.
  iree_notification_t notification;
  // Network proactor, independent of the model's execution services.
  iree_async_proactor_t* proactor;
  // Standard polling owner; remains alive until all network resources retire.
  iree_async_proactor_thread_t* thread;
  // Owned loopback listener.
  iree_async_socket_t* listener;
  // Bound address retained once after bind, including the selected port.
  iree_async_address_t address;
  // Listener operation ownership, joined before this identity is reused.
  struct {
    // One reusable accept target, never reused across shutdown cancellation.
    iree_async_socket_accept_operation_t operation;
    // Owned cancellation receipt joined independently of target completion.
    iree_async_cancel_request_t cancellation;
    // Target and cancellation ownership bits protected by the server mutex.
    uint32_t pending;
  } accept;
  // Shutdown intent and owner-thread acknowledgement, protected by the mutex.
  uint32_t flags;
  // Owned listener failure propagated by destroy.
  iree_status_t failure;
  // Fixed connection identities; a slot is reused only after caller and I/O
  // retire.
  loom_serve_http_connection_t connections[HTTP_CONNECTION_CAPACITY];
};

static void http_notify(loom_serve_http_server_t* server) {
  iree_notification_post(&server->notification, IREE_ALL_WAITERS);
}

static void http_connection_deactivated(void* user_data) {
  loom_serve_http_connection_t* connection = user_data;
  loom_serve_http_server_t* server = connection->server;
  iree_slim_mutex_lock(&server->mutex);
  connection->flags |= HTTP_CONNECTION_DRAINED;
  iree_slim_mutex_unlock(&server->mutex);
  http_notify(server);
}

// All deactivation intent converges here. The occupied slot retains the carrier
// across a synchronous deactivation callback; reclamation is application-owned.
static void http_connection_close(loom_serve_http_connection_t* connection) {
  loom_serve_http_server_t* server = connection->server;
  iree_slim_mutex_lock(&server->mutex);
  const bool begin =
      !iree_any_bit_set(connection->flags,
                        HTTP_CONNECTION_DEACTIVATING | HTTP_CONNECTION_DRAINED);
  connection->flags |= HTTP_CONNECTION_DEACTIVATING;
  iree_slim_mutex_unlock(&server->mutex);
  if (begin) {
    iree_net_carrier_deactivate(connection->carrier,
                                http_connection_deactivated, connection);
  }
}

static void http_connection_error(void* user_data, iree_status_t status) {
  loom_serve_http_connection_t* connection = user_data;
  loom_serve_http_server_t* server = connection->server;
  iree_slim_mutex_lock(&server->mutex);
  connection->failure = iree_status_join(connection->failure, status);
  iree_slim_mutex_unlock(&server->mutex);
  http_connection_close(connection);
  http_notify(server);
}

static iree_status_t http_connection_receive(void* user_data,
                                             iree_async_span_t data,
                                             iree_async_buffer_lease_t* lease) {
  (void)lease;
  loom_serve_http_connection_t* connection = user_data;
  const loom_serve_http_request_t* request = NULL;
  iree_status_t status =
      data.length
          ? loom_serve_http_request_parser_feed(
                connection->parser, iree_async_span_const_data(data), &request)
          : loom_serve_http_request_parser_finalize(connection->parser,
                                                    &request);
  if (iree_status_is_ok(status) && request) {
    iree_slim_mutex_lock(&connection->server->mutex);
    connection->request = request;
    connection->flags |= HTTP_CONNECTION_REQUEST_READY;
    iree_slim_mutex_unlock(&connection->server->mutex);
    http_notify(connection->server);
  }
  // EOF is a half-close: a complete request may still receive its response.
  return status;
}

static void http_connection_send_completed(void* user_data,
                                           iree_status_t status,
                                           iree_host_size_t bytes_transferred) {
  (void)bytes_transferred;
  loom_serve_http_connection_t* connection = user_data;
  loom_serve_http_server_t* server = connection->server;
  iree_slim_mutex_lock(&server->mutex);
  --connection->pending_sends;
  connection->failure = iree_status_join(connection->failure, status);
  const bool close =
      !iree_status_is_ok(connection->failure) ||
      (iree_any_bit_set(connection->flags, HTTP_CONNECTION_FINISHED) &&
       !connection->pending_sends);
  iree_slim_mutex_unlock(&server->mutex);
  if (close) {
    http_connection_close(connection);
  }
  http_notify(server);
}

static iree_status_t http_connection_create(
    loom_serve_http_connection_t* connection, iree_async_socket_t* socket) {
  loom_serve_http_server_t* server = connection->server;
  const loom_serve_http_request_limits_t limits = {16384, 1024 * 1024, 64};
  IREE_RETURN_IF_ERROR(loom_serve_http_request_parser_create(
      limits, &connection->parser, server->allocator));
  iree_async_slab_t* slab = NULL;
  const iree_async_slab_options_t slab_options = {
      .buffer_size = 16384,
      .buffer_count = 4,
  };
  iree_status_t status =
      iree_async_slab_create(slab_options, server->allocator, &slab);
  iree_async_region_t* region = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_register_slab(
        server->proactor, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region);
  }
  iree_async_buffer_pool_t* pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_async_buffer_pool_create(region, server->allocator, &pool);
  }
  if (iree_status_is_ok(status)) {
    iree_net_tcp_carrier_options_t options =
        iree_net_tcp_carrier_options_default();
    options.max_send_operations = HTTP_SEND_CAPACITY;
    status =
        iree_net_tcp_carrier_create(server->proactor, socket, pool, &options,
                                    server->allocator, &connection->carrier);
  }
  iree_async_buffer_pool_release(pool);
  iree_async_region_release(region);
  iree_async_slab_release(slab);
  if (iree_status_is_ok(status)) {
    status = iree_net_carrier_set_handlers(
        connection->carrier,
        (iree_net_carrier_handlers_t){http_connection_receive,
                                      http_connection_error, connection});
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_carrier_activate(connection->carrier);
  }
  return status;
}

static void http_accept_cancelled(void* user_data) {
  loom_serve_http_server_t* server = user_data;
  iree_slim_mutex_lock(&server->mutex);
  server->accept.pending &= ~HTTP_ACCEPT_CANCEL_PENDING;
  iree_slim_mutex_unlock(&server->mutex);
  http_notify(server);
}

static void http_accept_completed(void* user_data,
                                  iree_async_operation_t* operation,
                                  iree_status_t status,
                                  iree_async_completion_flags_t flags) {
  (void)operation;
  loom_serve_http_server_t* server = user_data;
  iree_slim_mutex_lock(&server->mutex);
  server->accept.pending &= ~HTTP_ACCEPT_PENDING;
  const bool cancel_pending =
      iree_any_bit_set(server->accept.pending, HTTP_ACCEPT_CANCEL_PENDING);
  loom_serve_http_connection_t* connection = NULL;
  if (iree_status_is_ok(status) &&
      !iree_any_bit_set(server->flags, HTTP_SERVER_STOPPING) &&
      !iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_CANCELLED)) {
    for (iree_host_size_t i = 0; i < HTTP_CONNECTION_CAPACITY; ++i) {
      if (!server->connections[i].flags) {
        connection = &server->connections[i];
        connection->flags = HTTP_CONNECTION_OCCUPIED;
        break;
      }
    }
    if (!connection) {
      fprintf(stderr, "HTTP connection capacity exhausted; closing peer.\n");
    }
  }
  if (!iree_status_is_ok(status)) {
    server->failure = iree_status_join(server->failure, status);
    server->flags |= HTTP_SERVER_STOPPING;
  }
  iree_slim_mutex_unlock(&server->mutex);
  if (connection) {
    status = http_connection_create(connection,
                                    server->accept.operation.accepted_socket);
    if (!iree_status_is_ok(status)) {
      // Activation is transactional: failed creation has no accepted I/O.
      iree_slim_mutex_lock(&server->mutex);
      connection->failure = status;
      connection->flags |= HTTP_CONNECTION_DRAINED;
      iree_slim_mutex_unlock(&server->mutex);
    }
  }
  iree_async_socket_release(server->accept.operation.accepted_socket);
  server->accept.operation.accepted_socket = NULL;
  if (cancel_pending) {
    iree_async_proactor_cancel_request_target_retired(
        server->proactor, &server->accept.cancellation);
  }
  iree_slim_mutex_lock(&server->mutex);
  const bool rearm = !iree_any_bit_set(server->flags, HTTP_SERVER_STOPPING);
  if (rearm) {
    server->accept.pending |= HTTP_ACCEPT_PENDING;
  }
  iree_slim_mutex_unlock(&server->mutex);
  if (rearm) {
    iree_async_operation_initialize(
        &server->accept.operation.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT,
        IREE_ASYNC_OPERATION_FLAG_CANCELLATION_IS_SUCCESS,
        http_accept_completed, server);
    status = iree_async_proactor_submit_one(server->proactor,
                                            &server->accept.operation.base);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_lock(&server->mutex);
      server->accept.pending &= ~HTTP_ACCEPT_PENDING;
      server->failure = iree_status_join(server->failure, status);
      server->flags |= HTTP_SERVER_STOPPING;
      iree_slim_mutex_unlock(&server->mutex);
    }
  }
  http_notify(server);
}

static void http_shutdown_received(iree_async_proactor_t* proactor,
                                   uint64_t message, void* user_data) {
  (void)message;
  loom_serve_http_server_t* server = user_data;
  iree_slim_mutex_lock(&server->mutex);
  const bool cancel =
      iree_any_bit_set(server->accept.pending, HTTP_ACCEPT_PENDING);
  if (cancel) {
    server->accept.pending |= HTTP_ACCEPT_CANCEL_PENDING;
  }
  iree_slim_mutex_unlock(&server->mutex);
  if (cancel) {
    iree_status_t status = iree_async_proactor_request_cancel(
        proactor, &server->accept.operation.base, &server->accept.cancellation);
    if (!iree_status_is_ok(status)) {
      iree_status_abort(status);
    }
  }
  for (iree_host_size_t i = 0; i < HTTP_CONNECTION_CAPACITY; ++i) {
    loom_serve_http_connection_t* connection = &server->connections[i];
    if (connection->carrier) {
      http_connection_close(connection);
    }
  }
  iree_slim_mutex_lock(&server->mutex);
  server->flags |= HTTP_SERVER_SHUTDOWN_RECEIVED;
  iree_slim_mutex_unlock(&server->mutex);
  http_notify(server);
}

static void http_shutdown_signal(void* user_data, iree_async_signal_t signal) {
  (void)signal;
  loom_serve_http_server_t* server = user_data;
  iree_slim_mutex_lock(&server->mutex);
  server->flags |= HTTP_SERVER_STOPPING;
  iree_slim_mutex_unlock(&server->mutex);
  http_notify(server);
}

static void http_proactor_failed(void* user_data, iree_status_t status) {
  (void)user_data;
  // The polling owner cannot return outstanding I/O storage after fatal
  // failure.
  iree_status_abort(status);
}

static bool http_server_drained(void* user_data) {
  loom_serve_http_server_t* server = user_data;
  iree_slim_mutex_lock(&server->mutex);
  bool drained =
      iree_any_bit_set(server->flags, HTTP_SERVER_SHUTDOWN_RECEIVED) &&
      !server->accept.pending;
  for (iree_host_size_t i = 0; i < HTTP_CONNECTION_CAPACITY; ++i) {
    const http_connection_flags_t flags = server->connections[i].flags;
    drained &= !flags || iree_any_bit_set(flags, HTTP_CONNECTION_DRAINED);
  }
  iree_slim_mutex_unlock(&server->mutex);
  return drained;
}

// Called only after the slot has returned both application and I/O ownership.
static void http_connection_reclaim(loom_serve_http_connection_t* connection) {
  loom_serve_http_server_t* server = connection->server;
  iree_net_carrier_release(connection->carrier);
  loom_serve_http_request_parser_destroy(connection->parser);
  if (!iree_status_is_ok(connection->failure)) {
    fprintf(stderr, "HTTP peer retired: ");
    iree_status_fprint(stderr, connection->failure);
    iree_status_free(connection->failure);
  }
  iree_slim_mutex_lock(&server->mutex);
  memset(connection, 0, sizeof(*connection));
  connection->server = server;
  iree_slim_mutex_unlock(&server->mutex);
}

iree_status_t loom_serve_http_server_destroy(loom_serve_http_server_t* server) {
  if (!server) {
    return iree_ok_status();
  }
  iree_slim_mutex_lock(&server->mutex);
  server->flags |= HTTP_SERVER_STOPPING;
  iree_slim_mutex_unlock(&server->mutex);
  if (server->thread) {
    iree_status_t status =
        iree_async_proactor_send_message(server->proactor, 0);
    if (!iree_status_is_ok(status)) {
      iree_status_abort(status);
    }
    iree_notification_await(&server->notification, http_server_drained, server,
                            iree_infinite_timeout());
  } else if (iree_any_bit_set(server->accept.pending, HTTP_ACCEPT_PENDING)) {
    // Thread creation failed before polling ownership was established. The
    // creating thread can still join its accepted listener operation.
    http_shutdown_received(server->proactor, 0, server);
    while (!http_server_drained(server)) {
      iree_status_t status = iree_async_proactor_poll(
          server->proactor, iree_infinite_timeout(), NULL);
      if (!iree_status_is_ok(status)) {
        iree_status_abort(status);
      }
    }
  }
  for (iree_host_size_t i = 0; i < HTTP_CONNECTION_CAPACITY; ++i) {
    if (server->connections[i].flags) {
      http_connection_reclaim(&server->connections[i]);
    }
  }
  iree_async_socket_release(server->listener);
  iree_status_t status = server->failure;
  if (server->thread) {
    iree_async_proactor_thread_request_stop(server->thread);
    status =
        iree_status_join(status, iree_async_proactor_thread_join(
                                     server->thread, IREE_DURATION_INFINITE));
    status = iree_status_join(
        status, iree_async_proactor_thread_consume_status(server->thread));
    iree_async_proactor_thread_release(server->thread);
  } else if (server->proactor) {
    iree_async_proactor_end_polling(server->proactor);
  }
  iree_async_proactor_release(server->proactor);
  iree_notification_deinitialize(&server->notification);
  iree_slim_mutex_deinitialize(&server->mutex);
  iree_allocator_free(server->allocator, server);
  return status;
}

static iree_status_t http_server_initialize(loom_serve_http_server_t* server,
                                            uint16_t port) {
  IREE_RETURN_IF_ERROR(iree_async_signal_ignore_broken_pipe());
  iree_async_proactor_options_t options = iree_async_proactor_options_default();
  options.threading_mode = IREE_ASYNC_PROACTOR_THREADING_CROSS_THREAD;
  options.debug_name = IREE_SV("loom-http");
  IREE_RETURN_IF_ERROR(iree_async_proactor_create_platform(
      options, server->allocator, &server->proactor));
  iree_async_proactor_set_message_callback(
      server->proactor,
      (iree_async_proactor_message_callback_t){http_shutdown_received, server});
  const iree_async_signal_t signals[] = {IREE_ASYNC_SIGNAL_INTERRUPT,
                                         IREE_ASYNC_SIGNAL_TERMINATE};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(signals) && iree_status_is_ok(status); ++i) {
    iree_async_signal_subscription_t* subscription = NULL;
    status = iree_async_proactor_subscribe_signal(
        server->proactor, signals[i],
        (iree_async_signal_callback_t){http_shutdown_signal, server},
        &subscription);
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(iree_async_socket_create(
      server->proactor, IREE_ASYNC_SOCKET_TYPE_TCP,
      IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR | IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
      &server->listener));
  iree_async_address_t address;
  IREE_RETURN_IF_ERROR(
      iree_async_address_from_ipv4(IREE_SV("127.0.0.1"), port, &address));
  IREE_RETURN_IF_ERROR(iree_async_socket_bind(server->listener, &address));
  IREE_RETURN_IF_ERROR(
      iree_async_socket_listen(server->listener, HTTP_CONNECTION_CAPACITY));
  IREE_RETURN_IF_ERROR(iree_async_socket_query_local_address(server->listener,
                                                             &server->address));
  iree_async_cancel_request_initialize(
      (iree_async_cancel_callback_t){http_accept_cancelled, server},
      &server->accept.cancellation);
  iree_async_operation_initialize(
      &server->accept.operation.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT,
      IREE_ASYNC_OPERATION_FLAG_CANCELLATION_IS_SUCCESS, http_accept_completed,
      server);
  server->accept.operation.listen_socket = server->listener;
  IREE_RETURN_IF_ERROR(iree_async_proactor_submit_one(
      server->proactor, &server->accept.operation.base));
  server->accept.pending |= HTTP_ACCEPT_PENDING;
  iree_async_proactor_thread_options_t thread_options =
      iree_async_proactor_thread_options_default();
  thread_options.debug_name = IREE_SV("loom-http");
  thread_options.error_callback = (iree_async_proactor_thread_error_callback_t){
      http_proactor_failed, server};
  return iree_async_proactor_thread_create(server->proactor, thread_options,
                                           server->allocator, &server->thread);
}

iree_status_t loom_serve_http_server_create(
    uint16_t port, iree_allocator_t host_allocator,
    loom_serve_http_server_t** out_server) {
  *out_server = NULL;
  loom_serve_http_server_t* server = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*server), (void**)&server));
  server->allocator = host_allocator;
  iree_slim_mutex_initialize(&server->mutex);
  iree_notification_initialize(&server->notification);
  for (iree_host_size_t i = 0; i < HTTP_CONNECTION_CAPACITY; ++i) {
    server->connections[i].server = server;
  }
  iree_status_t status = http_server_initialize(server, port);
  if (iree_status_is_ok(status)) {
    *out_server = server;
  } else {
    status = iree_status_join(status, loom_serve_http_server_destroy(server));
  }
  return status;
}

const iree_async_address_t* loom_serve_http_server_address(
    const loom_serve_http_server_t* server) {
  return &server->address;
}

iree_notification_t* loom_serve_http_server_notification(
    loom_serve_http_server_t* server) {
  return &server->notification;
}

bool loom_serve_http_server_is_stopping(loom_serve_http_server_t* server) {
  iree_slim_mutex_lock(&server->mutex);
  const bool stopping = iree_any_bit_set(server->flags, HTTP_SERVER_STOPPING);
  iree_slim_mutex_unlock(&server->mutex);
  return stopping;
}

loom_serve_http_connection_t* loom_serve_http_server_take_request(
    loom_serve_http_server_t* server,
    const loom_serve_http_request_t** out_request) {
  *out_request = NULL;
  loom_serve_http_connection_t* selected = NULL;
  for (iree_host_size_t i = 0; i < HTTP_CONNECTION_CAPACITY; ++i) {
    loom_serve_http_connection_t* connection = &server->connections[i];
    iree_slim_mutex_lock(&server->mutex);
    const http_connection_flags_t flags = connection->flags;
    const bool reclaim = iree_any_bit_set(flags, HTTP_CONNECTION_DRAINED) &&
                         !iree_any_bit_set(flags, HTTP_CONNECTION_CLAIMED);
    if (!selected && iree_any_bit_set(flags, HTTP_CONNECTION_REQUEST_READY) &&
        !iree_any_bit_set(flags, HTTP_CONNECTION_CLAIMED |
                                     HTTP_CONNECTION_FINISHED |
                                     HTTP_CONNECTION_DEACTIVATING |
                                     HTTP_CONNECTION_DRAINED) &&
        iree_status_is_ok(connection->failure)) {
      connection->flags |= HTTP_CONNECTION_CLAIMED;
      selected = connection;
      *out_request = connection->request;
    }
    iree_slim_mutex_unlock(&server->mutex);
    if (reclaim) {
      http_connection_reclaim(connection);
    }
  }
  return selected;
}

bool loom_serve_http_connection_failed(
    loom_serve_http_connection_t* connection) {
  iree_slim_mutex_lock(&connection->server->mutex);
  const bool failed = !iree_status_is_ok(connection->failure);
  iree_slim_mutex_unlock(&connection->server->mutex);
  return failed;
}

bool loom_serve_http_connection_can_send(
    loom_serve_http_connection_t* connection) {
  return iree_net_carrier_query_send_budget(connection->carrier).slots != 0;
}

iree_status_t loom_serve_http_connection_send(
    loom_serve_http_connection_t* connection, iree_string_view_t bytes) {
  iree_slim_mutex_lock(&connection->server->mutex);
  ++connection->pending_sends;
  iree_slim_mutex_unlock(&connection->server->mutex);
  const iree_net_send_params_t params = {
      .generated_prefix = iree_net_send_prefix_from_bytes(
          iree_make_const_byte_span(bytes.data, bytes.size)),
      .completion_callback = {http_connection_send_completed, connection},
  };
  iree_status_t status = iree_net_carrier_send(connection->carrier, &params);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&connection->server->mutex);
    --connection->pending_sends;
    iree_slim_mutex_unlock(&connection->server->mutex);
  }
  return status;
}

void loom_serve_http_connection_finish(
    loom_serve_http_connection_t* connection) {
  iree_slim_mutex_lock(&connection->server->mutex);
  connection->flags &= ~HTTP_CONNECTION_CLAIMED;
  connection->flags |= HTTP_CONNECTION_FINISHED;
  const bool close = !connection->pending_sends;
  iree_slim_mutex_unlock(&connection->server->mutex);
  if (close) {
    http_connection_close(connection);
  }
}

void loom_serve_http_connection_abort(
    loom_serve_http_connection_t* connection) {
  iree_slim_mutex_lock(&connection->server->mutex);
  connection->flags &= ~HTTP_CONNECTION_CLAIMED;
  connection->flags |= HTTP_CONNECTION_FINISHED;
  iree_slim_mutex_unlock(&connection->server->mutex);
  http_connection_close(connection);
}
