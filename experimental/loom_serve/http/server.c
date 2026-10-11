// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/http/server.h"

#include <limits.h>
#include <string.h>

#include "iree/base/threading/mutex.h"

typedef enum http_connection_flag_bits_e {
  HTTP_CONNECTION_OCCUPIED = 1u << 0,
  HTTP_CONNECTION_REQUEST_READY = 1u << 1,
  HTTP_CONNECTION_CLAIMED = 1u << 2,
  HTTP_CONNECTION_FINISHED = 1u << 3,
  HTTP_CONNECTION_DEACTIVATING = 1u << 4,
  HTTP_CONNECTION_DRAINED = 1u << 5,
  HTTP_CONNECTION_INITIALIZING = 1u << 6,
} http_connection_flag_bits_t;
typedef uint32_t http_connection_flags_t;

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
  // Immutable connection and request budgets established before network I/O.
  loom_serve_http_server_options_t options;
  // Protects peer publication, ownership, send counts and shutdown facts.
  iree_slim_mutex_t mutex;
  // Advisory wakeup for the single application owner.
  iree_notification_t notification;
  // Application task wake target, with publication joined by the server mutex.
  loom_serve_http_observer_t observer;
  // Retained shared progress owner, outliving all accepted carrier callbacks.
  iree_async_proactor_pool_t* proactor_pool;
  // Admission closure, protected by the mutex; active claims remain owned.
  bool stopping;
  // Bounded history, with network/application writers serialized by mutex.
  loom_serve_telemetry_stream_t* diagnostics;
  // Fixed connection identities; a slot is reused only after caller and I/O
  // retire.
  loom_serve_http_connection_t* connections;
};

// Publication and the final callback access share the same mutex. A consumer
// observing retirement may immediately reclaim storage after acquiring it.
static void http_notify_locked(loom_serve_http_server_t* server) {
  if (server->observer.wake) {
    server->observer.wake(server->observer.user_data);
  }
  iree_notification_post(&server->notification, IREE_ALL_WAITERS);
}

static void http_connection_deactivated(void* user_data) {
  loom_serve_http_connection_t* connection = user_data;
  loom_serve_http_server_t* server = connection->server;
  iree_slim_mutex_lock(&server->mutex);
  connection->flags |= HTTP_CONNECTION_DRAINED;
  http_notify_locked(server);
  iree_slim_mutex_unlock(&server->mutex);
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
  http_notify_locked(server);
  iree_slim_mutex_unlock(&server->mutex);
  http_connection_close(connection);
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
    http_notify_locked(connection->server);
    iree_slim_mutex_unlock(&connection->server->mutex);
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
  http_notify_locked(server);
  iree_slim_mutex_unlock(&server->mutex);
  if (close) {
    http_connection_close(connection);
  }
}

void loom_serve_http_server_accept(loom_serve_http_server_t* server,
                                   iree_net_carrier_t* carrier) {
  loom_serve_http_connection_t* connection = NULL;
  iree_slim_mutex_lock(&server->mutex);
  if (!server->stopping) {
    for (iree_host_size_t i = 0; i < server->options.connection_capacity; ++i) {
      if (!server->connections[i].flags) {
        connection = &server->connections[i];
        connection->flags =
            HTTP_CONNECTION_OCCUPIED | HTTP_CONNECTION_INITIALIZING;
        connection->carrier = carrier;
        break;
      }
    }
    if (!connection) {
      const loom_serve_telemetry_record_t record = {
          .time_ns = iree_time_now(),
          .event = &LOOM_SERVE_TELEMETRY_ERROR,
          .data.error =
              {
                  .code = IREE_STATUS_RESOURCE_EXHAUSTED,
                  .length = sizeof("HTTP connection capacity exhausted") - 1,
                  .message = "HTTP connection capacity exhausted",
              },
      };
      loom_serve_telemetry_stream_append(server->diagnostics, &record);
    }
  }
  iree_slim_mutex_unlock(&server->mutex);
  if (!connection) {
    iree_net_carrier_release(carrier);
    return;
  }

  iree_status_t status = loom_serve_http_request_parser_create(
      server->options.request_limits, &connection->parser, server->allocator);
  if (iree_status_is_ok(status)) {
    status = iree_net_carrier_set_handlers(
        carrier,
        (iree_net_carrier_handlers_t){http_connection_receive,
                                      http_connection_error, connection});
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_carrier_activate(carrier);
  }
  iree_slim_mutex_lock(&server->mutex);
  if (!iree_status_is_ok(status)) {
    // Failed activation accepts no I/O. The owner reclaims the occupied slot.
    connection->failure = iree_status_join(connection->failure, status);
    connection->flags |= HTTP_CONNECTION_DRAINED;
  }
  connection->flags &= ~HTTP_CONNECTION_INITIALIZING;
  http_notify_locked(server);
  iree_slim_mutex_unlock(&server->mutex);
}

static bool http_server_drained(void* user_data) {
  loom_serve_http_server_t* server = user_data;
  iree_slim_mutex_lock(&server->mutex);
  bool drained = true;
  for (iree_host_size_t i = 0; i < server->options.connection_capacity; ++i) {
    const http_connection_flags_t flags = server->connections[i].flags;
    drained &= !flags || iree_any_bit_set(flags, HTTP_CONNECTION_DRAINED);
  }
  iree_slim_mutex_unlock(&server->mutex);
  return drained;
}

void loom_serve_http_server_record_error(loom_serve_http_server_t* server,
                                         iree_status_t status) {
  loom_serve_telemetry_record_t record = {.time_ns = iree_time_now(),
                                          .event = &LOOM_SERVE_TELEMETRY_ERROR};
  loom_serve_telemetry_diagnostic_capture(status, &record.data.error);
  iree_slim_mutex_lock(&server->mutex);
  loom_serve_telemetry_stream_append(server->diagnostics, &record);
  http_notify_locked(server);
  iree_slim_mutex_unlock(&server->mutex);
}

// Called only after the slot has returned both application and I/O ownership.
static void http_connection_reclaim(loom_serve_http_connection_t* connection) {
  loom_serve_http_server_t* server = connection->server;
  iree_net_carrier_release(connection->carrier);
  loom_serve_http_request_parser_destroy(connection->parser);
  if (!iree_status_is_ok(connection->failure)) {
    loom_serve_http_server_record_error(server, connection->failure);
    iree_status_free(connection->failure);
  }
  iree_slim_mutex_lock(&server->mutex);
  memset(connection, 0, sizeof(*connection));
  connection->server = server;
  iree_slim_mutex_unlock(&server->mutex);
}

void loom_serve_http_server_destroy(loom_serve_http_server_t* server) {
  if (!server) {
    return;
  }
  loom_serve_http_server_request_stop(server);
  for (iree_host_size_t i = 0; i < server->options.connection_capacity; ++i) {
    loom_serve_http_connection_t* connection = &server->connections[i];
    if (connection->carrier) {
      http_connection_close(connection);
    }
  }
  iree_notification_await(&server->notification, http_server_drained, server,
                          iree_infinite_timeout());
  for (iree_host_size_t i = 0; i < server->options.connection_capacity; ++i) {
    if (server->connections[i].flags) {
      http_connection_reclaim(&server->connections[i]);
    }
  }
  iree_async_proactor_pool_release(server->proactor_pool);
  loom_serve_telemetry_stream_destroy(server->diagnostics);
  iree_notification_deinitialize(&server->notification);
  iree_slim_mutex_deinitialize(&server->mutex);
  iree_allocator_free(server->allocator, server->connections);
  iree_allocator_free(server->allocator, server);
}

loom_serve_http_server_options_t loom_serve_http_server_options_default(void) {
  return (loom_serve_http_server_options_t){
      .connection_capacity = 64,
      .request_limits = {16384, 8 * 1024 * 1024, 64},
  };
}

iree_status_t loom_serve_http_server_create(
    iree_async_proactor_pool_t* proactor_pool,
    const loom_serve_http_server_options_t* options,
    loom_serve_http_server_t** out_server, iree_allocator_t host_allocator) {
  *out_server = NULL;
  if (!options->connection_capacity || options->connection_capacity > INT_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "connection_capacity must be in [1, INT_MAX]");
  }
  loom_serve_http_server_t* server = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*server), (void**)&server));
  server->allocator = host_allocator;
  server->proactor_pool = proactor_pool;
  iree_async_proactor_pool_retain(proactor_pool);
  iree_slim_mutex_initialize(&server->mutex);
  iree_notification_initialize(&server->notification);
  iree_status_t status = loom_serve_telemetry_stream_create(
      128, &server->diagnostics, host_allocator);
  if (iree_status_is_ok(status)) {
    const loom_serve_telemetry_record_t begin = {
        .time_ns = iree_time_now(),
        .event = &LOOM_SERVE_TELEMETRY_STREAM_BEGIN};
    loom_serve_telemetry_stream_append(server->diagnostics, &begin);
    status = iree_allocator_malloc_array(
        host_allocator, options->connection_capacity,
        sizeof(*server->connections), (void**)&server->connections);
  }
  if (iree_status_is_ok(status)) {
    server->options = *options;
    for (iree_host_size_t i = 0; i < options->connection_capacity; ++i) {
      server->connections[i].server = server;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_server = server;
  } else {
    loom_serve_http_server_destroy(server);
  }
  return status;
}

iree_async_proactor_pool_t* loom_serve_http_server_proactor_pool(
    loom_serve_http_server_t* server) {
  return server->proactor_pool;
}

iree_notification_t* loom_serve_http_server_notification(
    loom_serve_http_server_t* server) {
  return &server->notification;
}

iree_host_size_t loom_serve_http_server_connection_capacity(
    const loom_serve_http_server_t* server) {
  return server->options.connection_capacity;
}

const loom_serve_telemetry_stream_t* loom_serve_http_server_diagnostics(
    const loom_serve_http_server_t* server) {
  return server->diagnostics;
}

void loom_serve_http_server_set_observer(loom_serve_http_server_t* server,
                                         loom_serve_http_observer_t observer) {
  iree_slim_mutex_lock(&server->mutex);
  server->observer = observer;
  if (observer.wake) {
    observer.wake(observer.user_data);
  }
  iree_slim_mutex_unlock(&server->mutex);
}

void loom_serve_http_server_request_stop(loom_serve_http_server_t* server) {
  iree_slim_mutex_lock(&server->mutex);
  server->stopping = true;
  http_notify_locked(server);
  iree_slim_mutex_unlock(&server->mutex);
}

bool loom_serve_http_server_is_stopping(loom_serve_http_server_t* server) {
  iree_slim_mutex_lock(&server->mutex);
  const bool stopping = server->stopping;
  iree_slim_mutex_unlock(&server->mutex);
  return stopping;
}

loom_serve_http_connection_t* loom_serve_http_server_take_request(
    loom_serve_http_server_t* server,
    const loom_serve_http_request_t** out_request) {
  *out_request = NULL;
  loom_serve_http_connection_t* selected = NULL;
  for (iree_host_size_t i = 0; i < server->options.connection_capacity; ++i) {
    loom_serve_http_connection_t* connection = &server->connections[i];
    iree_slim_mutex_lock(&server->mutex);
    const http_connection_flags_t flags = connection->flags;
    const bool reclaim =
        iree_any_bit_set(flags, HTTP_CONNECTION_DRAINED) &&
        !iree_any_bit_set(
            flags, HTTP_CONNECTION_CLAIMED | HTTP_CONNECTION_INITIALIZING);
    if (!selected && iree_any_bit_set(flags, HTTP_CONNECTION_REQUEST_READY) &&
        !iree_any_bit_set(
            flags, HTTP_CONNECTION_CLAIMED | HTTP_CONNECTION_FINISHED |
                       HTTP_CONNECTION_DEACTIVATING | HTTP_CONNECTION_DRAINED |
                       HTTP_CONNECTION_INITIALIZING) &&
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
