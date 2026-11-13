// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <string>

#include "iree/base/api.h"
#include "loom/runtime/program/package.h"
#include "loom/serve/server/configuration.h"
#include "loom/serve/server/http/package.h"
#include "loom/serve/transport/http/request.h"

namespace {

constexpr size_t kMaximumInputLength = 64 * 1024;
constexpr iree_host_size_t kGeneratedBodyOverhead = 4096;

void AppendMultipartField(std::string* body, const char* name,
                          iree_string_view_t value, iree_string_view_t filename,
                          const char* content_type) {
  *body += "--loom-fuzz-boundary\r\nContent-Disposition: form-data; name=\"";
  *body += name;
  *body += "\"";
  if (filename.data) {
    *body += "; filename=\"";
    body->append(filename.data, filename.size);
    *body += "\"";
  }
  *body += "\r\n";
  if (content_type) {
    *body += "Content-Type: ";
    *body += content_type;
    *body += "\r\n";
  }
  *body += "\r\n";
  if (value.size) {
    body->append(value.data, value.size);
  }
  *body += "\r\n";
}

std::string JsonRequest(iree_string_view_t body) {
  std::string request =
      "POST /v1/models/register HTTP/1.1\r\nHost: local\r\n"
      "Content-Type: application/json\r\nContent-Length: ";
  request += std::to_string(body.size);
  request += "\r\n\r\n";
  request.append(body.data, body.size);
  return request;
}

std::string PackageRequest(iree_string_view_t input) {
  iree_string_view_t document = IREE_SV("deployment/request.json");
  iree_string_view_t first_path = IREE_SV("deployment/prepare.loom");
  iree_string_view_t first_contents = iree_string_view_empty();
  iree_string_view_t second_path = IREE_SV("deployment/source.loom");
  iree_string_view_t second_contents = iree_string_view_empty();
  iree_string_view_t json = input;
  iree_string_view_t* fields[] = {&document, &first_path, &first_contents,
                                  &second_path, &second_contents};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(fields); ++i) {
    iree_string_view_t field;
    iree_string_view_t remaining;
    if (iree_string_view_split(json, '\n', &field, &remaining) < 0) {
      break;
    }
    *fields[i] = field;
    json = remaining;
  }

  std::string body;
  body.reserve(input.size + 1024);
  AppendMultipartField(&body, "request", json, document, "application/json");
  AppendMultipartField(&body, "file", first_contents, first_path,
                       "application/octet-stream");
  AppendMultipartField(&body, "file", second_contents, second_path,
                       "application/octet-stream");
  body += "--loom-fuzz-boundary--\r\n";

  std::string request =
      "POST /v1/models/register HTTP/1.1\r\nHost: local\r\n"
      "Content-Type: multipart/form-data; "
      "boundary=loom-fuzz-boundary\r\nContent-Length: ";
  request += std::to_string(body.size());
  request += "\r\n\r\n";
  request += body;
  return request;
}

void Exercise(const std::string& wire_request) {
  const loom_serve_http_request_limits_t limits = {
      .header_byte_capacity = 8192,
      .body_byte_capacity = kMaximumInputLength + kGeneratedBodyOverhead,
      .header_capacity = 64,
      .multipart_header_byte_capacity = 8192,
      .multipart_part_capacity = 64,
  };
  loom_serve_http_request_parser_t* parser = nullptr;
  iree_status_t status = loom_serve_http_request_parser_allocate(
      limits, iree_allocator_system(), &parser);
  const loom_serve_http_request_t* request = nullptr;
  if (iree_status_is_ok(status)) {
    status = loom_serve_http_request_parser_feed(
        parser,
        iree_make_const_byte_span(wire_request.data(), wire_request.size()),
        &request);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_http_request_parser_finalize(parser, &request);
  }

  loom_serve_model_configuration_t configuration = {};
  iree_string_view_t json;
  loom_runtime_package_t* package = nullptr;
  if (iree_status_is_ok(status)) {
    status = loom_serve_http_package_parse(request, iree_allocator_system(),
                                           &json, &package);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_model_configuration_initialize(
        json, iree_string_view_empty(), package, iree_allocator_system(),
        &configuration);
  }
  loom_runtime_package_release(package);
  loom_serve_model_configuration_deinitialize(&configuration);
  iree_status_free(status);
  loom_serve_http_request_parser_free(parser);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (!size || size > kMaximumInputLength) {
    return 0;
  }
  const iree_string_view_t input =
      iree_make_string_view(reinterpret_cast<const char*>(data + 1), size - 1);
  switch (data[0]) {
    case 'j':
      Exercise(JsonRequest(input));
      break;
    case 'p':
      Exercise(PackageRequest(input));
      break;
    default:
      break;
  }
  return 0;
}
