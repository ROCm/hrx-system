// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Strict bounded HTTP/1.1 request framing.

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_HTTP_REQUEST_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_HTTP_REQUEST_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Hard storage and structure limits for one HTTP request.
typedef struct loom_serve_http_request_limits_t {
  // Maximum request-line and header bytes including the terminal CRLF.
  iree_host_size_t header_byte_capacity;
  // Maximum body bytes declared by Content-Length.
  iree_host_size_t body_byte_capacity;
  // Maximum number of unique request headers.
  iree_host_size_t header_capacity;
} loom_serve_http_request_limits_t;

// One fully framed HTTP/1.1 request.
typedef struct loom_serve_http_request_t {
  // Case-sensitive HTTP method token.
  iree_string_view_t method;
  // Origin-form request target beginning with '/'.
  iree_string_view_t target;
  // Case-preserving unique headers in wire order.
  iree_string_pair_list_t headers;
  // Exact body bytes declared by Content-Length, or empty when absent.
  iree_string_view_t body;
} loom_serve_http_request_t;

// Stateful parser for exactly one HTTP/1.1 request.
typedef struct loom_serve_http_request_parser_t
    loom_serve_http_request_parser_t;

// Creates an empty parser with explicit request capacities.
iree_status_t loom_serve_http_request_parser_create(
    loom_serve_http_request_limits_t limits,
    loom_serve_http_request_parser_t** out_parser,
    iree_allocator_t host_allocator);

// Destroys a parser and invalidates its completed request views. NULL is valid.
void loom_serve_http_request_parser_destroy(
    loom_serve_http_request_parser_t* parser);

// Appends arbitrary request-stream bytes.
//
// |out_request| is NULL until the exact declared request is present. Once
// returned, it and all nested views are borrowed until parser destruction.
// Trailing bytes are rejected because this parser owns one request per
// connection.
iree_status_t loom_serve_http_request_parser_feed(
    loom_serve_http_request_parser_t* parser, iree_const_byte_span_t chunk,
    const loom_serve_http_request_t** out_request);

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

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_HTTP_REQUEST_H_
