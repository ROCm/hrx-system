// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/server.h"

#include "iree/base/internal/path.h"
#include "iree/base/internal/unicode.h"
#include "loom/serve/server/http/response.h"
#include "loom/serve/server/realtime/session.h"
#include "loom/serve/transport/websocket/handshake.h"

struct loom_serve_realtime_server_t {
  // Allocator owning session slots and the conversation owner.
  iree_allocator_t host_allocator;
  // Shared application executor, retained by the outer router call.
  iree_task_executor_t* executor;
  // Existing HTTP/carrier lifetime owner, including upgraded connections.
  loom_serve_http_server_t* transport;
  // Serialized registry shared with ordinary modality/control endpoints.
  loom_serve_registry_t* registry;
  // Borrowed deployment compositions and explicit browser-origin policy.
  loom_serve_realtime_options_t options;
  // Shared router wake target, live until all sessions retire.
  loom_serve_wake_callback_t observer;
  // Bounded session slots, including disconnected CPU/checkpoint borrowers.
  loom_serve_realtime_session_t** sessions;
  // Maximum long-lived claims, below transport capacity to reserve control.
  iree_host_size_t capacity;
  // Occupied session slots.
  iree_host_size_t count;
  // Rotating first session in a bounded serving turn.
  iree_host_size_t cursor;
  // Last assigned server session identity; identities survive slot reuse.
  uint64_t sequence;
};

iree_status_t loom_serve_realtime_server_allocate(
    iree_task_executor_t* executor, loom_serve_http_server_t* transport,
    loom_serve_registry_t* registry, loom_serve_realtime_options_t options,
    loom_serve_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_serve_realtime_server_t** out_server) {
  *out_server = NULL;
  for (iree_host_size_t i = 0; i < options.profile_count; ++i) {
    const loom_serve_realtime_profile_t* profile = &options.profiles[i];
    if (!profile->name.size || profile->name.size > 128 ||
        !iree_unicode_utf8_validate(profile->name) || !profile->text.size ||
        !profile->recognition.size || !profile->speech.size ||
        !profile->voice.size) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "realtime profile requires a name, three deployments, and a voice");
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (iree_string_view_equal(profile->name, options.profiles[j].name)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate realtime profile name");
      }
    }
  }
  loom_serve_realtime_server_t* server = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*server), (void**)&server));
  server->host_allocator = host_allocator;
  server->executor = executor;
  server->transport = transport;
  server->registry = registry;
  server->options = options;
  server->observer = observer;
  server->capacity = loom_serve_http_server_connection_capacity(transport) - 1;
  iree_status_t status = iree_ok_status();
  if (server->capacity) {
    status = iree_allocator_malloc_array(host_allocator, server->capacity,
                                         sizeof(*server->sessions),
                                         (void**)&server->sessions);
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, server);
    return status;
  }
  *out_server = server;
  return iree_ok_status();
}

void loom_serve_realtime_server_free(loom_serve_realtime_server_t* server) {
  if (!server) {
    return;
  }
  iree_allocator_free(server->host_allocator, server->sessions);
  iree_allocator_free(server->host_allocator, server);
}

bool loom_serve_realtime_server_handles(iree_string_view_t target) {
  return iree_string_view_equal(target, IREE_SV("/v1/realtime")) ||
         iree_string_view_starts_with(target, IREE_SV("/v1/realtime?"));
}

static int loom_serve_realtime_query_hex(char value) {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'A' && value <= 'F') {
    return value - 'A' + 10;
  }
  if (value >= 'a' && value <= 'f') {
    return value - 'a' + 10;
  }
  return -1;
}

static iree_status_t loom_serve_realtime_query_decode(
    iree_string_view_t encoded, char* storage, iree_host_size_t capacity,
    iree_string_view_t* decoded) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < encoded.size; ++i) {
    if (count == capacity) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "realtime query value is too long");
    }
    char value = encoded.data[i];
    if (value == '%') {
      if (encoded.size - i < 3 ||
          loom_serve_realtime_query_hex(encoded.data[i + 1]) < 0 ||
          loom_serve_realtime_query_hex(encoded.data[i + 2]) < 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid realtime query escape");
      }
      value = (char)((loom_serve_realtime_query_hex(encoded.data[i + 1]) << 4) |
                     loom_serve_realtime_query_hex(encoded.data[i + 2]));
      i += 2;
    } else if (value == '+') {
      value = ' ';
    }
    storage[count++] = value;
  }
  *decoded = iree_make_string_view(storage, count);
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_profile_select(
    loom_serve_realtime_server_t* server, iree_string_view_t target,
    const loom_serve_realtime_profile_t** out_profile) {
  *out_profile = NULL;
  iree_string_view_t path, query;
  iree_string_view_split(target, '?', &path, &query);
  iree_string_pair_t parameter;
  iree_host_size_t count = 0;
  if (!iree_uri_split_params(query, 1, &count, &parameter) || count != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "realtime requires exactly one model query parameter");
  }
  char key_storage[16], value_storage[128];
  iree_string_view_t key, value;
  IREE_RETURN_IF_ERROR(loom_serve_realtime_query_decode(
      parameter.key, key_storage, sizeof(key_storage), &key));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_query_decode(
      parameter.value, value_storage, sizeof(value_storage), &value));
  if (!iree_string_view_equal(key, IREE_SV("model"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported realtime query parameter");
  }
  for (iree_host_size_t i = 0; i < server->options.profile_count; ++i) {
    if (iree_string_view_equal(value, server->options.profiles[i].name)) {
      *out_profile = &server->options.profiles[i];
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "realtime assistant profile is not configured");
}

iree_status_t loom_serve_realtime_server_accept(
    loom_serve_realtime_server_t* server,
    loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request, bool allow_wait,
    iree_string_builder_t* scratch) {
  if (!allow_wait) {
    return loom_serve_http_response_reject(
        server->transport, connection, 429, "Too Many Requests",
        IREE_SV("long-lived connection capacity is full"), scratch);
  }
  const loom_serve_realtime_profile_t* profile = NULL;
  loom_serve_realtime_binding_t text, recognition, speech;
  loom_serve_realtime_binding_t activity = {0};
  iree_status_t status =
      loom_serve_realtime_profile_select(server, request->target, &profile);
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_binding_resolve(
        server->registry, profile->text, 1u << LOOM_SERVE_MODEL_KIND_TEXT,
        &text);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_binding_resolve(
        server->registry, profile->recognition,
        1u << LOOM_SERVE_MODEL_KIND_RECOGNITION, &recognition);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_binding_resolve(
        server->registry, profile->speech, 1u << LOOM_SERVE_MODEL_KIND_SPEECH,
        &speech);
  }
  if (iree_status_is_ok(status) && profile->activity.size) {
    status = loom_serve_realtime_binding_resolve(
        server->registry, profile->activity,
        (1u << LOOM_SERVE_MODEL_KIND_ACTIVITY) |
            (1u << LOOM_SERVE_MODEL_KIND_RECOGNITION),
        &activity);
  }
  if (iree_status_is_ok(status)) {
    iree_string_builder_reset(scratch);
    status =
        loom_serve_websocket_handshake(request, server->options.allowed_origins,
                                       iree_string_view_empty(), scratch);
  }
  if (!iree_status_is_ok(status)) {
    return loom_serve_http_response_error(server->transport, connection, status,
                                          scratch);
  }
  loom_serve_websocket_t* socket = NULL;
  status = loom_serve_websocket_upgrade(connection,
                                        loom_serve_websocket_limits_default(),
                                        server->host_allocator, &socket);
  if (!iree_status_is_ok(status)) {
    loom_serve_http_connection_abort(connection);
    return status;
  }
  status = loom_serve_http_connection_send(connection,
                                           iree_string_builder_view(scratch));
  if (!iree_status_is_ok(status)) {
    loom_serve_http_server_record_error(server->transport, status);
    iree_status_ignore(status);
    loom_serve_http_connection_abort(connection);
    return iree_ok_status();
  }
  loom_serve_realtime_session_t* session = NULL;
  status = loom_serve_realtime_session_allocate(
      ++server->sequence, profile, text, recognition, speech, activity,
      server->executor, server->registry, server->transport, connection, socket,
      server->observer, server->host_allocator, &session);
  if (!iree_status_is_ok(status)) {
    loom_serve_http_connection_abort(connection);
    return status;
  }
  iree_host_size_t index = 0;
  while (server->sessions[index]) {
    ++index;
  }
  server->sessions[index] = session;
  ++server->count;
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_server_advance(
    loom_serve_realtime_server_t* server, bool stopping, bool* out_progress) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < server->capacity && iree_status_is_ok(status); ++i) {
    const iree_host_size_t index = (server->cursor + i) % server->capacity;
    if (!server->sessions[index]) {
      continue;
    }
    bool complete = false;
    status = loom_serve_realtime_session_advance(
        server->sessions[index], stopping, out_progress, &complete);
    if (complete) {
      loom_serve_realtime_session_free(server->sessions[index]);
      server->sessions[index] = NULL;
      --server->count;
    }
  }
  if (++server->cursor >= server->capacity) {
    server->cursor = 0;
  }
  return status;
}

iree_host_size_t loom_serve_realtime_server_claims(
    const loom_serve_realtime_server_t* server) {
  return server->count;
}
