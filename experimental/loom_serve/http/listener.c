// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/http/listener.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "iree/async/api.h"
#include "iree/base/threading/mutex.h"

enum {
  HTTP_LISTENER_ACCEPT = 1u << 0,
  HTTP_LISTENER_STOP = 1u << 1,
  HTTP_LISTENER_CANCEL = 1u << 2,
};

struct loom_serve_http_listener_t {
  // Allocator owning the listener and connection construction resources.
  iree_allocator_t allocator;
  // Borrowed HTTP owner, joined after this listener retires.
  loom_serve_http_server_t* server;
  // Retained shared poll runner and proactor, independent of this binding.
  iree_async_proactor_pool_entry_t* entry;
  // Borrowed proactor owned by entry.
  iree_async_proactor_t* proactor;
  // Owned listening socket, unrelated to accepted socket lifetimes.
  iree_async_socket_t* socket;
  // Bound address including any kernel-selected ephemeral port.
  iree_async_address_t address;
  // Exclusively owned filesystem path after successful Unix bind, else empty.
  char path[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  // Per-connection receive storage configuration, with default NUMA placement.
  iree_async_slab_options_t receive;
  // Per-connection bounded send policy copied during construction.
  iree_net_stream_carrier_options_t send;
  // Joins publication and final callback accesses with retirement queries.
  iree_slim_mutex_t mutex;
  // Advisory wake for listener retirement, not HTTP/model readiness.
  iree_notification_t notification;
  // Admission has closed, either explicitly or on a terminal accept failure.
  bool stopping;
  // Accept, stop handoff, and cancellation receipt ownership bits.
  uint32_t pending;
  // Owned terminal listener failure, returned by destroy.
  iree_status_t failure;
  // Reused accept operation; cancellation joins before this storage retires.
  iree_async_socket_accept_operation_t accept;
  // Poll-owner handoff for cancellation without blocking the caller.
  iree_async_nop_operation_t stop;
  // Independent receipt joining any native cancellation key.
  iree_async_cancel_request_t cancellation;
};

static iree_status_t http_listener_create_carrier(
    loom_serve_http_listener_t* listener, iree_async_socket_t* socket,
    iree_net_carrier_t** out_carrier) {
  *out_carrier = NULL;
  iree_async_slab_t* slab = NULL;
  IREE_RETURN_IF_ERROR(
      iree_async_slab_create(listener->receive, listener->allocator, &slab));
  iree_async_region_t* region = NULL;
  iree_status_t status = iree_async_proactor_register_slab(
      listener->proactor, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region);
  iree_async_buffer_pool_t* pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_async_buffer_pool_create(region, listener->allocator, &pool);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_stream_carrier_create(listener->proactor, socket, pool,
                                            &listener->send,
                                            listener->allocator, out_carrier);
  }
  iree_async_buffer_pool_release(pool);
  iree_async_region_release(region);
  iree_async_slab_release(slab);
  return status;
}

static void http_listener_cancel_retired(void* user_data) {
  loom_serve_http_listener_t* listener = user_data;
  iree_slim_mutex_lock(&listener->mutex);
  listener->pending &= ~HTTP_LISTENER_CANCEL;
  iree_notification_post(&listener->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&listener->mutex);
}

static void http_listener_accept_completed(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  loom_serve_http_listener_t* listener = user_data;
  iree_slim_mutex_lock(&listener->mutex);
  const bool admit =
      !listener->stopping && iree_status_is_ok(status) &&
      !iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_CANCELLED);
  iree_slim_mutex_unlock(&listener->mutex);
  if (admit) {
    iree_net_carrier_t* carrier = NULL;
    iree_status_t peer_status = http_listener_create_carrier(
        listener, listener->accept.accepted_socket, &carrier);
    if (iree_status_is_ok(peer_status)) {
      loom_serve_http_server_accept(listener->server, carrier);
    } else {
      loom_serve_http_server_record_error(listener->server, peer_status);
      iree_status_free(peer_status);
    }
  }
  iree_async_socket_release(listener->accept.accepted_socket);
  listener->accept.accepted_socket = NULL;

  iree_slim_mutex_lock(&listener->mutex);
  if (iree_status_is_ok(status) && !listener->stopping) {
    // Only this poll owner can complete or rearm the accept target.
    iree_async_operation_initialize(
        &listener->accept.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT,
        IREE_ASYNC_OPERATION_FLAG_CANCELLATION_IS_SUCCESS,
        http_listener_accept_completed, listener);
    status = iree_async_proactor_submit_one(listener->proactor,
                                            &listener->accept.base);
  } else {
    listener->pending &= ~HTTP_LISTENER_ACCEPT;
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_http_server_record_error(listener->server, status);
    listener->failure = iree_status_join(listener->failure, status);
    listener->stopping = true;
    listener->pending &= ~HTTP_LISTENER_ACCEPT;
  }
  const bool cancel_pending =
      iree_any_bit_set(listener->pending, HTTP_LISTENER_CANCEL);
  iree_notification_post(&listener->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&listener->mutex);
  if (cancel_pending) {
    // The receipt can release the last borrowed access; nothing follows it.
    iree_async_proactor_cancel_request_target_retired(listener->proactor,
                                                      &listener->cancellation);
  }
}

static void http_listener_stop_received(void* user_data,
                                        iree_async_operation_t* operation,
                                        iree_status_t status,
                                        iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  IREE_CHECK_OK(status);
  loom_serve_http_listener_t* listener = user_data;
  iree_slim_mutex_lock(&listener->mutex);
  const bool cancel = iree_any_bit_set(listener->pending, HTTP_LISTENER_ACCEPT);
  if (cancel) {
    listener->pending |= HTTP_LISTENER_CANCEL;
  }
  iree_slim_mutex_unlock(&listener->mutex);
  if (cancel) {
    // This poll owner holds a private live accept and an idle receipt.
    IREE_CHECK_OK(iree_async_proactor_request_cancel(
        listener->proactor, &listener->accept.base, &listener->cancellation));
  }
  iree_slim_mutex_lock(&listener->mutex);
  listener->pending &= ~HTTP_LISTENER_STOP;
  iree_notification_post(&listener->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&listener->mutex);
}

void loom_serve_http_listener_request_stop(
    loom_serve_http_listener_t* listener) {
  iree_slim_mutex_lock(&listener->mutex);
  const bool submit = !listener->stopping && listener->pending;
  listener->stopping = true;
  if (submit) {
    listener->pending |= HTTP_LISTENER_STOP;
  }
  iree_slim_mutex_unlock(&listener->mutex);
  if (submit) {
    iree_async_operation_initialize(
        &listener->stop.base, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE, http_listener_stop_received, listener);
    IREE_CHECK_OK(iree_async_proactor_submit_one(listener->proactor,
                                                 &listener->stop.base));
  }
}

bool loom_serve_http_listener_is_stopped(loom_serve_http_listener_t* listener) {
  iree_slim_mutex_lock(&listener->mutex);
  const bool stopped = listener->stopping && !listener->pending;
  iree_slim_mutex_unlock(&listener->mutex);
  return stopped;
}

iree_notification_t* loom_serve_http_listener_notification(
    loom_serve_http_listener_t* listener) {
  return &listener->notification;
}

static bool http_listener_stopped(void* user_data) {
  return loom_serve_http_listener_is_stopped(user_data);
}

iree_status_t loom_serve_http_listener_destroy(
    loom_serve_http_listener_t* listener) {
  if (!listener) {
    return iree_ok_status();
  }
  loom_serve_http_listener_request_stop(listener);
  iree_notification_await(&listener->notification, http_listener_stopped,
                          listener, iree_infinite_timeout());
  iree_async_socket_release(listener->socket);
  iree_status_t status = listener->failure;
  if (listener->path[0] && remove(listener->path) != 0) {
    status = iree_status_join(
        status,
        iree_make_status(iree_status_code_from_errno(errno),
                         "removing Unix listener path '%s'", listener->path));
  }
  iree_async_proactor_pool_entry_release(listener->entry);
  iree_notification_deinitialize(&listener->notification);
  iree_slim_mutex_deinitialize(&listener->mutex);
  iree_allocator_free(listener->allocator, listener);
  return status;
}

loom_serve_http_listener_options_t loom_serve_http_listener_options_default(
    void) {
  loom_serve_http_listener_options_t options = {
      .address = IREE_SV("127.0.0.1:8080"),
      .backlog = 64,
      .receive_buffer_size = 16384,
      .receive_buffer_count = 4,
      .send = iree_net_stream_carrier_options_default(),
  };
  options.send.max_send_operations = 8;
  return options;
}

iree_status_t loom_serve_http_listener_create(
    loom_serve_http_server_t* server,
    const loom_serve_http_listener_options_t* options,
    loom_serve_http_listener_t** out_listener, iree_allocator_t allocator) {
  *out_listener = NULL;
  if (options->backlog < 1 || !options->receive_buffer_size ||
      !iree_host_size_is_power_of_two(options->receive_buffer_count) ||
      !options->send.max_send_operations ||
      options->send.max_send_operations == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid listener backlog or carrier budgets");
  }
  iree_async_address_t address;
  IREE_RETURN_IF_ERROR(
      iree_async_address_from_string(options->address, &address));
  const bool is_unix =
      iree_string_view_starts_with(options->address, IREE_SV("unix:"));
  const iree_async_socket_type_t socket_type =
      is_unix                           ? IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM
      : options->address.data[0] == '[' ? IREE_ASYNC_SOCKET_TYPE_TCP6
                                        : IREE_ASYNC_SOCKET_TYPE_TCP;
  loom_serve_http_listener_t* listener = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*listener), (void**)&listener));
  listener->allocator = allocator;
  listener->server = server;
  listener->receive.buffer_size = options->receive_buffer_size;
  listener->receive.buffer_count = options->receive_buffer_count;
  listener->send = options->send;
  iree_slim_mutex_initialize(&listener->mutex);
  iree_notification_initialize(&listener->notification);
  iree_async_cancel_request_initialize(
      (iree_async_cancel_callback_t){http_listener_cancel_retired, listener},
      &listener->cancellation);
  iree_status_t status = iree_async_proactor_pool_acquire(
      loom_serve_http_server_proactor_pool(server), options->proactor_index,
      &listener->entry);
  if (iree_status_is_ok(status)) {
    listener->proactor =
        iree_async_proactor_pool_entry_proactor(listener->entry);
    status = iree_async_socket_create(
        listener->proactor, socket_type,
        is_unix ? IREE_ASYNC_SOCKET_OPTION_NONE
                : IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR |
                      IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
        &listener->socket);
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_socket_bind(listener->socket, &address);
    if (iree_status_is_ok(status) && is_unix &&
        options->address.data[5] != '@') {
      const iree_string_view_t path =
          iree_string_view_substr(options->address, 5, IREE_HOST_SIZE_MAX);
      memcpy(listener->path, path.data, path.size);
      listener->path[path.size] = 0;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_socket_listen(listener->socket, options->backlog);
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_socket_query_local_address(listener->socket,
                                                   &listener->address);
  }
  if (iree_status_is_ok(status)) {
    iree_async_operation_initialize(
        &listener->accept.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT,
        IREE_ASYNC_OPERATION_FLAG_CANCELLATION_IS_SUCCESS,
        http_listener_accept_completed, listener);
    listener->accept.listen_socket = listener->socket;
    listener->pending |= HTTP_LISTENER_ACCEPT;
    status = iree_async_proactor_submit_one(listener->proactor,
                                            &listener->accept.base);
    if (!iree_status_is_ok(status)) {
      listener->pending &= ~HTTP_LISTENER_ACCEPT;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_listener = listener;
  } else {
    status =
        iree_status_join(status, loom_serve_http_listener_destroy(listener));
  }
  return status;
}

const iree_async_address_t* loom_serve_http_listener_address(
    const loom_serve_http_listener_t* listener) {
  return &listener->address;
}
