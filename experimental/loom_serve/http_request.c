// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/http_request.h"

#include <string.h>

typedef enum loom_serve_http_request_parser_phase_e {
  LOOM_SERVE_HTTP_REQUEST_PARSER_PHASE_HEADERS = 0,
  LOOM_SERVE_HTTP_REQUEST_PARSER_PHASE_BODY,
  LOOM_SERVE_HTTP_REQUEST_PARSER_PHASE_COMPLETE,
  LOOM_SERVE_HTTP_REQUEST_PARSER_PHASE_FAILED,
} loom_serve_http_request_parser_phase_t;

typedef struct loom_serve_http_request_range_t {
  // Byte offset into the retained request storage.
  iree_host_size_t offset;
  // Byte length of the referenced request content.
  iree_host_size_t length;
} loom_serve_http_request_range_t;

typedef struct loom_serve_http_header_record_t {
  // Case-preserving header-name range.
  loom_serve_http_request_range_t name;
  // Header-value range with surrounding optional whitespace removed.
  loom_serve_http_request_range_t value;
} loom_serve_http_header_record_t;

typedef struct loom_serve_http_header_record_list_t {
  // Preallocated header records in wire order.
  loom_serve_http_header_record_t* values;
  // Number of populated records in |values|.
  iree_host_size_t count;
  // Maximum number of records available in |values|.
  iree_host_size_t capacity;
} loom_serve_http_header_record_list_t;

struct loom_serve_http_request_parser_t {
  // Allocator owning the parser and all retained storage.
  iree_allocator_t host_allocator;
  // Hard request storage and structure limits.
  loom_serve_http_request_limits_t limits;
  // Active framing phase or terminal lifecycle state.
  loom_serve_http_request_parser_phase_t phase;
  // Exact request bytes received from the connection.
  iree_string_builder_t storage;
  // Request method range established with the request line.
  loom_serve_http_request_range_t method;
  // Origin-form target range established with the request line.
  loom_serve_http_request_range_t target;
  // Parsed header ranges retained across body-storage growth.
  loom_serve_http_header_record_list_t header_records;
  // Total header bytes including the terminal empty line.
  iree_host_size_t header_length;
  // Total request bytes required by Content-Length.
  iree_host_size_t request_length;
  // Final contiguous public header pairs.
  iree_string_pair_t* header_pairs;
  // Final public request borrowing parser-owned storage.
  loom_serve_http_request_t request;
};

static iree_string_view_t loom_serve_http_request_storage_view(
    const loom_serve_http_request_parser_t* parser) {
  return iree_string_builder_view(&parser->storage);
}

static iree_string_view_t loom_serve_http_request_range_view(
    const loom_serve_http_request_parser_t* parser,
    loom_serve_http_request_range_t range) {
  const iree_string_view_t storage =
      loom_serve_http_request_storage_view(parser);
  return iree_make_string_view(
      storage.data ? storage.data + range.offset : NULL, range.length);
}

static bool loom_serve_http_is_token_character(uint8_t value) {
  if ((value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
      (value >= '0' && value <= '9')) {
    return true;
  }
  switch (value) {
    case '!':
    case '#':
    case '$':
    case '%':
    case '&':
    case '\'':
    case '*':
    case '+':
    case '-':
    case '.':
    case '^':
    case '_':
    case '`':
    case '|':
    case '~':
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_serve_http_validate_token(iree_string_view_t value,
                                                    const char* field_name) {
  if (iree_string_view_is_empty(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "HTTP %s is empty",
                            field_name);
  }
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    if (!loom_serve_http_is_token_character((uint8_t)value.data[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HTTP %s contains an invalid byte", field_name);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_http_validate_target(
    iree_string_view_t target) {
  if (iree_string_view_is_empty(target) || target.data[0] != '/') {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HTTP request target must use origin form");
  }
  for (iree_host_size_t i = 0; i < target.size; ++i) {
    const uint8_t value = (uint8_t)target.data[i];
    if (value < 0x21 || value > 0x7E || value == '#') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HTTP request target contains an invalid byte");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_http_validate_header_value(
    iree_string_view_t value) {
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    const uint8_t byte = (uint8_t)value.data[i];
    if (byte != '\t' && (byte < 0x20 || byte > 0x7E)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HTTP header value contains an invalid byte");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_http_parse_content_length(
    iree_string_view_t value, iree_host_size_t* out_length) {
  if (iree_string_view_is_empty(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HTTP Content-Length is empty");
  }
  iree_host_size_t length = 0;
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    const uint8_t byte = (uint8_t)value.data[i];
    if (byte < '0' || byte > '9') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HTTP Content-Length is not decimal");
    }
    iree_host_size_t scaled_length = 0;
    if (!iree_host_size_checked_mul(length, 10, &scaled_length) ||
        !iree_host_size_checked_add(scaled_length, byte - '0', &length)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HTTP Content-Length overflows host storage");
    }
  }
  *out_length = length;
  return iree_ok_status();
}

static bool loom_serve_http_header_name_equal(
    const loom_serve_http_request_parser_t* parser,
    const loom_serve_http_header_record_t* record, iree_string_view_t name) {
  return iree_string_view_equal_case(
      loom_serve_http_request_range_view(parser, record->name), name);
}

static iree_status_t loom_serve_http_parse_request_line(
    loom_serve_http_request_parser_t* parser, iree_string_view_t line) {
  const iree_host_size_t method_end = iree_string_view_find_char(line, ' ', 0);
  if (method_end == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HTTP request line has no target");
  }
  const iree_host_size_t target_end =
      iree_string_view_find_char(line, ' ', method_end + 1);
  if (target_end == IREE_STRING_VIEW_NPOS ||
      iree_string_view_find_char(line, ' ', target_end + 1) !=
          IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HTTP request line has invalid spacing");
  }
  const iree_string_view_t method =
      iree_make_string_view(line.data, method_end);
  const iree_string_view_t target = iree_make_string_view(
      line.data + method_end + 1, target_end - method_end - 1);
  const iree_string_view_t version = iree_make_string_view(
      line.data + target_end + 1, line.size - target_end - 1);
  IREE_RETURN_IF_ERROR(loom_serve_http_validate_token(method, "method"));
  IREE_RETURN_IF_ERROR(loom_serve_http_validate_target(target));
  if (!iree_string_view_equal(version, IREE_SV("HTTP/1.1"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HTTP version must be HTTP/1.1");
  }
  const iree_string_view_t storage =
      loom_serve_http_request_storage_view(parser);
  parser->method = (loom_serve_http_request_range_t){
      .offset = (iree_host_size_t)(method.data - storage.data),
      .length = method.size,
  };
  parser->target = (loom_serve_http_request_range_t){
      .offset = (iree_host_size_t)(target.data - storage.data),
      .length = target.size,
  };
  return iree_ok_status();
}

static iree_status_t loom_serve_http_append_header_record(
    loom_serve_http_request_parser_t* parser, iree_string_view_t name,
    iree_string_view_t value) {
  loom_serve_http_header_record_list_t* records = &parser->header_records;
  if (records->count == records->capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "HTTP request header capacity exhausted");
  }
  for (iree_host_size_t i = 0; i < records->count; ++i) {
    if (loom_serve_http_header_name_equal(parser, &records->values[i], name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HTTP request header is duplicated");
    }
  }
  const iree_string_view_t storage =
      loom_serve_http_request_storage_view(parser);
  records->values[records->count++] = (loom_serve_http_header_record_t){
      .name =
          {
              .offset = (iree_host_size_t)(name.data - storage.data),
              .length = name.size,
          },
      .value =
          {
              .offset = (iree_host_size_t)(value.data - storage.data),
              .length = value.size,
          },
  };
  return iree_ok_status();
}

static iree_status_t loom_serve_http_parse_headers(
    loom_serve_http_request_parser_t* parser) {
  const iree_string_view_t storage =
      loom_serve_http_request_storage_view(parser);
  const iree_host_size_t request_line_end =
      iree_string_view_find(storage, IREE_SV("\r\n"), 0);
  if (request_line_end == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HTTP request line is incomplete");
  }
  IREE_RETURN_IF_ERROR(loom_serve_http_parse_request_line(
      parser, iree_make_string_view(storage.data, request_line_end)));

  bool has_host = false;
  iree_host_size_t content_length = 0;
  iree_host_size_t cursor = request_line_end + 2;
  while (cursor < parser->header_length) {
    const iree_host_size_t line_end =
        iree_string_view_find(storage, IREE_SV("\r\n"), cursor);
    if (line_end == IREE_STRING_VIEW_NPOS ||
        line_end + 2 > parser->header_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HTTP header line is incomplete");
    }
    if (line_end == cursor) {
      cursor += 2;
      break;
    }
    const iree_string_view_t line =
        iree_make_string_view(storage.data + cursor, line_end - cursor);
    const iree_host_size_t colon = iree_string_view_find_char(line, ':', 0);
    if (colon == IREE_STRING_VIEW_NPOS || colon == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HTTP header line has no valid name");
    }
    const iree_string_view_t name = iree_make_string_view(line.data, colon);
    IREE_RETURN_IF_ERROR(loom_serve_http_validate_token(name, "header name"));
    iree_host_size_t value_start = colon + 1;
    while (value_start < line.size &&
           (line.data[value_start] == ' ' || line.data[value_start] == '\t')) {
      ++value_start;
    }
    iree_host_size_t value_end = line.size;
    while (value_end > value_start && (line.data[value_end - 1] == ' ' ||
                                       line.data[value_end - 1] == '\t')) {
      --value_end;
    }
    const iree_string_view_t value =
        iree_make_string_view(line.data + value_start, value_end - value_start);
    IREE_RETURN_IF_ERROR(loom_serve_http_validate_header_value(value));
    IREE_RETURN_IF_ERROR(
        loom_serve_http_append_header_record(parser, name, value));

    if (iree_string_view_equal_case(name, IREE_SV("Host"))) {
      if (iree_string_view_is_empty(value)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "HTTP Host is empty");
      }
      has_host = true;
    } else if (iree_string_view_equal_case(name, IREE_SV("Content-Length"))) {
      IREE_RETURN_IF_ERROR(
          loom_serve_http_parse_content_length(value, &content_length));
    } else if (iree_string_view_equal_case(name,
                                           IREE_SV("Transfer-Encoding"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HTTP Transfer-Encoding is unsupported");
    } else if (iree_string_view_equal_case(name, IREE_SV("Expect"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HTTP Expect is unsupported");
    } else if (iree_string_view_equal_case(name, IREE_SV("Content-Encoding")) &&
               !iree_string_view_equal_case(value, IREE_SV("identity"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HTTP Content-Encoding is unsupported");
    }
    cursor = line_end + 2;
  }
  if (cursor != parser->header_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HTTP headers have invalid line endings");
  }
  if (!has_host) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HTTP Host is required");
  }
  if (content_length > parser->limits.body_byte_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "HTTP request body capacity exhausted");
  }
  if (!iree_host_size_checked_add(parser->header_length, content_length,
                                  &parser->request_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HTTP request length overflows host storage");
  }
  parser->phase = LOOM_SERVE_HTTP_REQUEST_PARSER_PHASE_BODY;
  return iree_ok_status();
}

static iree_status_t loom_serve_http_complete_request(
    loom_serve_http_request_parser_t* parser) {
  const iree_host_size_t header_count = parser->header_records.count;
  iree_status_t status = iree_allocator_malloc_array(
      parser->host_allocator, header_count, sizeof(*parser->header_pairs),
      (void**)&parser->header_pairs);
  if (!iree_status_is_ok(status)) {
    return status;
  }
  for (iree_host_size_t i = 0; i < header_count; ++i) {
    parser->header_pairs[i] = (iree_string_pair_t){
        .key = loom_serve_http_request_range_view(
            parser, parser->header_records.values[i].name),
        .value = loom_serve_http_request_range_view(
            parser, parser->header_records.values[i].value),
    };
  }
  const iree_string_view_t storage =
      loom_serve_http_request_storage_view(parser);
  parser->request = (loom_serve_http_request_t){
      .method = loom_serve_http_request_range_view(parser, parser->method),
      .target = loom_serve_http_request_range_view(parser, parser->target),
      .headers =
          {
              .count = header_count,
              .pairs = parser->header_pairs,
          },
      .body =
          iree_make_string_view(storage.data + parser->header_length,
                                parser->request_length - parser->header_length),
  };
  parser->phase = LOOM_SERVE_HTTP_REQUEST_PARSER_PHASE_COMPLETE;
  return iree_ok_status();
}

static iree_status_t loom_serve_http_append_body_bytes(
    loom_serve_http_request_parser_t* parser, iree_const_byte_span_t chunk) {
  const iree_host_size_t storage_length =
      iree_string_builder_size(&parser->storage);
  if (storage_length > parser->request_length ||
      chunk.data_length > parser->request_length - storage_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HTTP request contains trailing or pipelined bytes");
  }
  if (chunk.data_length != 0) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        &parser->storage,
        iree_make_string_view((const char*)chunk.data, chunk.data_length)));
  }
  if (iree_string_builder_size(&parser->storage) == parser->request_length) {
    return loom_serve_http_complete_request(parser);
  }
  return iree_ok_status();
}

void loom_serve_http_request_parser_destroy(
    loom_serve_http_request_parser_t* parser) {
  if (!parser) {
    return;
  }
  iree_allocator_t host_allocator = parser->host_allocator;
  iree_allocator_free(host_allocator, parser->header_pairs);
  iree_allocator_free(host_allocator, parser->header_records.values);
  iree_string_builder_deinitialize(&parser->storage);
  iree_allocator_free(host_allocator, parser);
}

iree_status_t loom_serve_http_request_parser_create(
    loom_serve_http_request_limits_t limits,
    loom_serve_http_request_parser_t** out_parser,
    iree_allocator_t host_allocator) {
  if (!out_parser) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_parser is required");
  }
  *out_parser = NULL;
  if (limits.header_byte_capacity == 0 || limits.header_capacity == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HTTP header byte and record capacities must be nonzero");
  }

  loom_serve_http_request_parser_t* parser = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*parser), (void**)&parser);
  if (iree_status_is_ok(status)) {
    memset(parser, 0, sizeof(*parser));
    parser->host_allocator = host_allocator;
    parser->limits = limits;
    parser->phase = LOOM_SERVE_HTTP_REQUEST_PARSER_PHASE_HEADERS;
    iree_string_builder_initialize(host_allocator, &parser->storage);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc_array(host_allocator, limits.header_capacity,
                                    sizeof(*parser->header_records.values),
                                    (void**)&parser->header_records.values);
  }
  if (iree_status_is_ok(status)) {
    parser->header_records.capacity = limits.header_capacity;
    *out_parser = parser;
  } else {
    loom_serve_http_request_parser_destroy(parser);
  }
  return status;
}

iree_status_t loom_serve_http_request_parser_feed(
    loom_serve_http_request_parser_t* parser, iree_const_byte_span_t chunk,
    const loom_serve_http_request_t** out_request) {
  if (!parser || !out_request || (chunk.data_length != 0 && !chunk.data)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "active HTTP parser, valid bytes, and out_request are required");
  }
  *out_request = NULL;
  if (parser->phase == LOOM_SERVE_HTTP_REQUEST_PARSER_PHASE_COMPLETE ||
      parser->phase == LOOM_SERVE_HTTP_REQUEST_PARSER_PHASE_FAILED) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HTTP request parser is not active");
  }

  iree_host_size_t consumed_length = 0;
  iree_status_t status = iree_ok_status();
  if (parser->phase == LOOM_SERVE_HTTP_REQUEST_PARSER_PHASE_HEADERS) {
    const iree_host_size_t old_length =
        iree_string_builder_size(&parser->storage);
    const iree_host_size_t available_length =
        parser->limits.header_byte_capacity - old_length;
    const iree_host_size_t append_length =
        iree_min(chunk.data_length, available_length);
    if (append_length != 0) {
      status = iree_string_builder_append_string(
          &parser->storage,
          iree_make_string_view((const char*)chunk.data, append_length));
    }
    consumed_length = append_length;
    if (iree_status_is_ok(status)) {
      const iree_string_view_t storage =
          loom_serve_http_request_storage_view(parser);
      const iree_host_size_t scan_offset = old_length > 3 ? old_length - 3 : 0;
      const iree_host_size_t delimiter =
          iree_string_view_find(storage, IREE_SV("\r\n\r\n"), scan_offset);
      if (delimiter != IREE_STRING_VIEW_NPOS) {
        parser->header_length = delimiter + 4;
        status = loom_serve_http_parse_headers(parser);
      } else if (storage.size == parser->limits.header_byte_capacity) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "HTTP request header capacity exhausted");
      }
    }
  }
  if (iree_status_is_ok(status) &&
      parser->phase == LOOM_SERVE_HTTP_REQUEST_PARSER_PHASE_BODY) {
    status = loom_serve_http_append_body_bytes(
        parser, iree_make_const_byte_span(
                    chunk.data ? chunk.data + consumed_length : NULL,
                    chunk.data_length - consumed_length));
  }
  if (iree_status_is_ok(status) &&
      parser->phase == LOOM_SERVE_HTTP_REQUEST_PARSER_PHASE_COMPLETE) {
    *out_request = &parser->request;
  } else if (!iree_status_is_ok(status)) {
    parser->phase = LOOM_SERVE_HTTP_REQUEST_PARSER_PHASE_FAILED;
  }
  return status;
}

iree_status_t loom_serve_http_request_parser_finalize(
    loom_serve_http_request_parser_t* parser,
    const loom_serve_http_request_t** out_request) {
  if (!parser || !out_request) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HTTP parser and out_request are required");
  }
  *out_request = NULL;
  if (parser->phase == LOOM_SERVE_HTTP_REQUEST_PARSER_PHASE_COMPLETE) {
    *out_request = &parser->request;
    return iree_ok_status();
  }
  if (parser->phase == LOOM_SERVE_HTTP_REQUEST_PARSER_PHASE_FAILED) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HTTP request parser has failed");
  }
  const bool missing_headers =
      parser->phase == LOOM_SERVE_HTTP_REQUEST_PARSER_PHASE_HEADERS;
  parser->phase = LOOM_SERVE_HTTP_REQUEST_PARSER_PHASE_FAILED;
  return iree_make_status(IREE_STATUS_DATA_LOSS,
                          missing_headers
                              ? "HTTP connection ended inside the headers"
                              : "HTTP connection ended inside the body");
}

bool loom_serve_http_request_lookup_header(
    const loom_serve_http_request_t* request, iree_string_view_t name,
    iree_string_view_t* out_value) {
  if (out_value) {
    *out_value = iree_string_view_empty();
  }
  if (!request || !out_value || (name.size != 0 && !name.data)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < request->headers.count; ++i) {
    if (iree_string_view_equal_case(request->headers.pairs[i].key, name)) {
      *out_value = request->headers.pairs[i].value;
      return true;
    }
  }
  return false;
}
