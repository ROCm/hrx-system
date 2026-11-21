// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Strict bounded HTTP/1.1 request framing.

#ifndef LOOM_SERVE_TRANSPORT_HTTP_REQUEST_H_
#define LOOM_SERVE_TRANSPORT_HTTP_REQUEST_H_

#include "iree/base/api.h"
#include "loom/serve/transport/http/multipart.h"

#ifdef __cplusplus
extern "C" {
#endif

// Hard storage and structure limits for one HTTP request.
typedef struct loom_serve_http_request_limits_t {
  // Maximum request-line/header bytes and cumulative chunk framing bytes.
  iree_host_size_t header_byte_capacity;
  // Maximum body bytes after transfer decoding.
  iree_host_size_t body_byte_capacity;
  // Maximum unique request headers.
  iree_host_size_t header_capacity;
  // Maximum cumulative MIME header bytes, independent of payload bytes.
  iree_host_size_t multipart_header_byte_capacity;
  // Maximum retained multipart records, including repeated field names.
  iree_host_size_t multipart_part_capacity;
} loom_serve_http_request_limits_t;

// One fully framed HTTP/1.1 request.
typedef struct loom_serve_http_request_t {
  // Case-sensitive HTTP method token.
  iree_string_view_t method;
  // Complete origin-form request target, including any query string.
  iree_string_view_t target;
  // Encoded path beginning with '/', up to the first literal '?' in target.
  iree_string_view_t path;
  // Case-preserving unique headers in wire order.
  iree_string_pair_list_t headers;
  // Body after Content-Length or chunked framing, empty without either.
  // Chunk extensions and trailers are discarded, never merged into headers.
  iree_string_view_t body;
  // Parsed ordered form fields borrowing this request; NULL for other media.
  const loom_serve_http_multipart_t* multipart;
} loom_serve_http_request_t;

// Stateful parser for exactly one HTTP/1.1 request.
typedef struct loom_serve_http_request_parser_t
    loom_serve_http_request_parser_t;

// Creates an empty parser with explicit request capacities.
iree_status_t loom_serve_http_request_parser_allocate(
    loom_serve_http_request_limits_t limits, iree_allocator_t host_allocator,
    loom_serve_http_request_parser_t** out_parser);

// Destroys a parser and invalidates its completed request views. NULL is valid.
void loom_serve_http_request_parser_free(
    loom_serve_http_request_parser_t* parser);

// Appends arbitrary request-stream bytes.
//
// |out_request| is NULL until the complete framed request is present. Once
// returned, it and all nested views are borrowed until parser destruction.
// When supplied, |out_send_continue| is true only for the feed that accepts
// Expect: 100-continue headers while the peer still owes the entire body.
// Trailing bytes are rejected because this parser owns one request per
// connection.
iree_status_t loom_serve_http_request_parser_feed(
    loom_serve_http_request_parser_t* parser, iree_const_byte_span_t chunk,
    const loom_serve_http_request_t** out_request, bool* out_send_continue);

// Completes an EOF boundary or diagnoses a truncated request.
//
// A previously completed request is returned unchanged. Incomplete headers or
// body bytes fail with DATA_LOSS and terminally fail the parser.
iree_status_t loom_serve_http_request_parser_finalize(
    loom_serve_http_request_parser_t* parser,
    const loom_serve_http_request_t** out_request);

// Looks up one completed header name using ASCII case-insensitive comparison.
// Returns false when absent and leaves |out_value| empty.
bool loom_serve_http_request_lookup_header(
    const loom_serve_http_request_t* request, iree_string_view_t name,
    iree_string_view_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_TRANSPORT_HTTP_REQUEST_H_
