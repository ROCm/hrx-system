// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/dashboard/assets.h"

#include <string.h>

#include "loom/serve/dashboard/etags.h"
#include "loom/serve/dashboard/files.h"
#include "loom/serve/server/http/response.h"

static const iree_file_toc_t* loom_serve_dashboard_lookup(
    iree_string_view_t path) {
  if (!iree_string_view_consume_prefix(&path, IREE_SV("/dashboard/"))) {
    return NULL;
  }
  if (!path.size) {
    path = IREE_SV("index.html");
  }
  const iree_file_toc_t* files = loom_serve_dashboard_files_create();
  for (iree_host_size_t i = 0; i < loom_serve_dashboard_files_size(); ++i) {
    if (iree_string_view_equal(path, iree_make_cstring_view(files[i].name))) {
      return &files[i];
    }
  }
  return NULL;
}

// Only a positive RFC 9110 qvalue permits the compressed representation.
// Invalid weights cannot accidentally turn an explicit refusal into acceptance.
static bool loom_serve_dashboard_positive_quality(iree_string_view_t value) {
  if (!value.size || value.size > 5 ||
      (value.data[0] != '0' && value.data[0] != '1') ||
      (value.size > 1 && value.data[1] != '.')) {
    return false;
  }
  bool positive = value.data[0] == '1';
  for (iree_host_size_t i = 2; i < value.size; ++i) {
    const char digit = value.data[i];
    if (digit < '0' || digit > '9' || (value.data[0] == '1' && digit != '0')) {
      return false;
    }
    positive |= digit != '0';
  }
  return positive;
}

static bool loom_serve_dashboard_accepts_gzip(
    const loom_serve_http_request_t* request) {
  iree_string_view_t remaining;
  if (!loom_serve_http_request_lookup_header(
          request, IREE_SV("Accept-Encoding"), &remaining)) {
    return true;
  }
  // An explicit coding overrides '*'; a refusal wins among repeated entries.
  int gzip = -1;
  int wildcard = -1;
  while (remaining.size) {
    iree_string_view_t entry, coding, weight;
    iree_string_view_split(remaining, ',', &entry, &remaining);
    const bool has_weight =
        iree_string_view_split(entry, ';', &coding, &weight) >= 0;
    coding = iree_string_view_trim(coding);
    int* accepted = NULL;
    if (iree_string_view_equal_case(coding, IREE_SV("gzip")) ||
        iree_string_view_equal_case(coding, IREE_SV("x-gzip"))) {
      accepted = &gzip;
    } else if (iree_string_view_equal(coding, IREE_SV("*"))) {
      accepted = &wildcard;
    } else {
      continue;
    }
    bool positive = true;
    if (has_weight) {
      iree_string_view_t name, value;
      iree_string_view_split(weight, '=', &name, &value);
      positive =
          iree_string_view_equal_case(iree_string_view_trim(name),
                                      IREE_SV("q")) &&
          loom_serve_dashboard_positive_quality(iree_string_view_trim(value));
    }
    *accepted = positive && *accepted != 0;
  }
  return (gzip >= 0 ? gzip : wildcard) > 0;
}

// If-None-Match compares opaque tags weakly, including when a quoted tag
// contains a comma. A syntactically incomplete list cannot validate a body.
static bool loom_serve_dashboard_matches(
    const loom_serve_http_request_t* request, iree_string_view_t etag) {
  iree_string_view_t field;
  if (!loom_serve_http_request_lookup_header(request, IREE_SV("If-None-Match"),
                                             &field)) {
    return false;
  }
  field = iree_string_view_trim(field);
  if (iree_string_view_equal(field, IREE_SV("*"))) {
    return true;
  }
  bool matched = false;
  while (field.size) {
    if (iree_string_view_consume_prefix_char(&field, ',')) {
      field = iree_string_view_trim(field);
      continue;
    }
    iree_string_view_consume_prefix(&field, IREE_SV("W/"));
    if (!iree_string_view_consume_prefix_char(&field, '"')) {
      return false;
    }
    iree_string_view_t item;
    if (iree_string_view_split(field, '"', &item, &field) < 0) {
      return false;
    }
    matched |= iree_string_view_equal(item, etag);
    field = iree_string_view_trim(field);
    if (field.size && !iree_string_view_starts_with_char(field, ',')) {
      return false;
    }
  }
  return matched;
}

// Only the offline tool documents can execute model-authored content. The
// conversation itself communicates with the serving origin.
static const char* loom_serve_dashboard_policy(iree_string_view_t path) {
  if (iree_string_view_equal(path, IREE_SV("/dashboard/code.html"))) {
    return "sandbox allow-scripts; default-src 'none'; worker-src blob:; "
           "script-src 'unsafe-inline' 'unsafe-eval'; base-uri 'none'; "
           "form-action 'none'; frame-ancestors 'self'; webrtc 'block'";
  }
  if (iree_string_view_equal(path, IREE_SV("/dashboard/chat.html"))) {
    return "default-src 'none'; script-src 'self'; style-src 'self'; "
           "connect-src 'self'; img-src blob:; frame-src 'self'; worker-src "
           "'self'; "
           "form-action 'none'; base-uri 'none'; frame-ancestors 'none'";
  }
  if (iree_string_view_equal(path, IREE_SV("/dashboard/preview.html"))) {
    // The shell embeds only an opaque srcdoc. That document inherits this
    // policy, including the closed frame and connection destinations.
    return "default-src 'none'; script-src 'unsafe-inline'; "
           "style-src 'unsafe-inline'; img-src data: blob:; media-src data: "
           "blob:; "
           "font-src data:; form-action 'none'; base-uri 'none'; "
           "frame-ancestors 'self'; webrtc 'block'";
  }
  return "default-src 'none'; script-src 'self'; style-src 'self'; "
         "connect-src 'self'; base-uri 'none'; frame-ancestors 'none'";
}

typedef struct loom_serve_dashboard_response_t {
  // Stack header borrowed by the synchronous generated-send writer.
  iree_string_view_t header;
  // Immutable encoded representation. A validated GET has an empty body.
  iree_const_byte_span_t body;
} loom_serve_dashboard_response_t;

static iree_status_t loom_serve_dashboard_write(void* self,
                                                iree_byte_span_t storage) {
  const loom_serve_dashboard_response_t* response = self;
  memcpy(storage.data, response->header.data, response->header.size);
  memcpy(storage.data + response->header.size, response->body.data,
         response->body.data_length);
  return iree_ok_status();
}

bool loom_serve_dashboard_try_send(loom_serve_http_server_t* server,
                                   loom_serve_http_connection_t* connection,
                                   const loom_serve_http_request_t* request) {
  const iree_string_view_t path = request->path;
  if (iree_string_view_equal(path, IREE_SV("/")) ||
      iree_string_view_equal(path, IREE_SV("/dashboard"))) {
    loom_serve_http_response_send(
        server, connection,
        IREE_SV("HTTP/1.1 302 Found\r\nLocation: ./dashboard/\r\n"
                "Content-Length: 0\r\nConnection: close\r\n\r\n"));
    return true;
  }
  const iree_file_toc_t* file = loom_serve_dashboard_lookup(path);
  if (!file) {
    return false;
  }
  if (!loom_serve_dashboard_accepts_gzip(request)) {
    loom_serve_http_response_send(
        server, connection,
        IREE_SV("HTTP/1.1 406 Not Acceptable\r\nContent-Length: 0\r\n"
                "Vary: Accept-Encoding\r\nCache-Control: no-cache\r\n"
                "Connection: close\r\n\r\n"));
    return true;
  }
  const char* content_type = "text/html; charset=utf-8";
  if (iree_string_view_ends_with(path, IREE_SV(".css"))) {
    content_type = "text/css; charset=utf-8";
  } else if (iree_string_view_ends_with(path, IREE_SV(".js"))) {
    content_type = "text/javascript; charset=utf-8";
  }
  char header[512];
  iree_string_builder_t output;
  iree_string_builder_initialize_with_storage(header, sizeof(header), &output);
  const char* etag =
      loom_serve_dashboard_etags[file - loom_serve_dashboard_files_create()];
  const bool cached =
      loom_serve_dashboard_matches(request, iree_make_cstring_view(etag));
  iree_status_t status =
      cached
          ? iree_string_builder_append_cstring(&output,
                                               "HTTP/1.1 304 Not Modified\r\n")
          : iree_string_builder_append_format(
                &output,
                "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
                "Content-Encoding: gzip\r\n",
                content_type, file->size);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        &output,
        "ETag: \"%s\"\r\nVary: Accept-Encoding\r\nCache-Control: no-cache\r\n"
        "X-Content-Type-Options: nosniff\r\nContent-Security-Policy: %s\r\n"
        "Connection: close\r\n\r\n",
        etag, loom_serve_dashboard_policy(path));
  }
  if (iree_status_is_ok(status) &&
      loom_serve_http_connection_can_send(connection)) {
    const loom_serve_dashboard_response_t response = {
        iree_string_builder_view(&output),
        iree_make_const_byte_span(file->data, cached ? 0 : file->size)};
    status = loom_serve_http_connection_send_generated(
        connection, (iree_net_send_prefix_t){
                        response.header.size + response.body.data_length,
                        loom_serve_dashboard_write, (void*)&response});
  } else if (iree_status_is_ok(status)) {
    loom_serve_http_connection_abort(connection);
    iree_string_builder_deinitialize(&output);
    return true;
  }
  iree_string_builder_deinitialize(&output);
  if (iree_status_is_ok(status)) {
    loom_serve_http_connection_finish(connection);
  } else {
    loom_serve_http_server_record_error(server, status);
    iree_status_free(status);
    loom_serve_http_connection_abort(connection);
  }
  return true;
}
