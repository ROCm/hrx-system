// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "loom/serve/transport/http/multipart.h"
#include "loom/serve/transport/http/request.h"

namespace {

constexpr size_t kMaximumInputLength = 64 * 1024;

struct PartSnapshot {
  // Decoded form field name.
  std::string name;
  // Distinguishes an absent filename from an explicitly empty filename.
  bool has_filename;
  // Decoded opaque filename when present.
  std::string filename;
  // Unfolded media type.
  std::string content_type;
  // Exact binary part payload.
  std::string contents;

  bool operator==(const PartSnapshot& rhs) const {
    return name == rhs.name && has_filename == rhs.has_filename &&
           filename == rhs.filename && content_type == rhs.content_type &&
           contents == rhs.contents;
  }
};

struct RequestSnapshot {
  // Request method token.
  std::string method;
  // Origin-form request target.
  std::string target;
  // Case-preserving header fields in wire order.
  std::vector<std::pair<std::string, std::string>> headers;
  // Decoded request body.
  std::string body;
  // Parsed multipart fields when selected by Content-Type.
  std::vector<PartSnapshot> parts;

  bool operator==(const RequestSnapshot& rhs) const {
    return method == rhs.method && target == rhs.target &&
           headers == rhs.headers && body == rhs.body && parts == rhs.parts;
  }
};

std::string Copy(iree_string_view_t value) {
  return std::string(value.data ? value.data : "", value.size);
}

void Snapshot(const loom_serve_http_request_t* request,
              RequestSnapshot* out_snapshot) {
  out_snapshot->method = Copy(request->method);
  out_snapshot->target = Copy(request->target);
  out_snapshot->headers.reserve(request->headers.count);
  for (iree_host_size_t i = 0; i < request->headers.count; ++i) {
    out_snapshot->headers.emplace_back(Copy(request->headers.pairs[i].key),
                                       Copy(request->headers.pairs[i].value));
  }
  out_snapshot->body = Copy(request->body);
  const loom_serve_http_part_list_t parts =
      loom_serve_http_multipart_parts(request->multipart);
  out_snapshot->parts.reserve(parts.count);
  for (iree_host_size_t i = 0; i < parts.count; ++i) {
    const loom_serve_http_part_t* part = &parts.values[i];
    out_snapshot->parts.push_back({
        .name = Copy(part->name),
        .has_filename = part->filename.data != nullptr,
        .filename = Copy(part->filename),
        .content_type = Copy(part->content_type),
        .contents = Copy(part->contents),
    });
  }
}

uint64_t Hash(iree_const_byte_span_t bytes) {
  uint64_t hash = 14695981039346656037ull;
  for (iree_host_size_t i = 0; i < bytes.data_length; ++i) {
    hash = (hash ^ bytes.data[i]) * 1099511628211ull;
  }
  return hash;
}

bool Parse(iree_const_byte_span_t bytes, bool fragmented,
           RequestSnapshot* out_snapshot) {
  const loom_serve_http_request_limits_t limits = {
      .header_byte_capacity = 8192,
      .body_byte_capacity = kMaximumInputLength,
      .header_capacity = 64,
      .multipart_header_byte_capacity = 8192,
      .multipart_part_capacity = 64,
  };
  loom_serve_http_request_parser_t* parser = nullptr;
  iree_status_t status = loom_serve_http_request_parser_allocate(
      limits, iree_allocator_system(), &parser);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    return false;
  }

  const loom_serve_http_request_t* request = nullptr;
  status = loom_serve_http_request_parser_feed(
      parser, iree_const_byte_span_empty(), &request);
  if (iree_status_is_ok(status)) {
    if (fragmented && bytes.data_length) {
      const uint64_t hash = Hash(bytes);
      iree_host_size_t split_a = hash % (bytes.data_length + 1);
      iree_host_size_t split_b = (hash >> 32) % (bytes.data_length + 1);
      if (split_a > split_b) {
        std::swap(split_a, split_b);
      }
      const iree_const_byte_span_t chunks[] = {
          iree_make_const_byte_span(bytes.data, split_a),
          iree_make_const_byte_span(bytes.data + split_a, split_b - split_a),
          iree_make_const_byte_span(bytes.data + split_b,
                                    bytes.data_length - split_b),
      };
      for (iree_const_byte_span_t chunk : chunks) {
        if (!iree_status_is_ok(status)) {
          break;
        }
        if (!chunk.data_length) {
          continue;
        }
        status = loom_serve_http_request_parser_feed(parser, chunk, &request);
      }
    } else {
      status = loom_serve_http_request_parser_feed(parser, bytes, &request);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_http_request_parser_finalize(parser, &request);
  }

  const bool success = iree_status_is_ok(status);
  if (success) {
    if (!request) {
      iree_status_free(status);
      std::abort();
    }
    Snapshot(request, out_snapshot);
  }
  iree_status_free(status);
  loom_serve_http_request_parser_free(parser);
  return success;
}

void Exercise(iree_const_byte_span_t bytes) {
  RequestSnapshot contiguous;
  RequestSnapshot fragmented;
  const bool contiguous_success = Parse(bytes, false, &contiguous);
  const bool fragmented_success = Parse(bytes, true, &fragmented);
  if (contiguous_success != fragmented_success ||
      (contiguous_success && !(contiguous == fragmented))) {
    std::abort();
  }
}

std::string ContentLengthRequest(iree_const_byte_span_t body) {
  std::string request =
      "POST /fuzz HTTP/1.1\r\nHost: local\r\nContent-Type: "
      "application/octet-stream\r\nContent-Length: ";
  request += std::to_string(body.data_length);
  request += "\r\n\r\n";
  if (body.data_length) {
    request.append(reinterpret_cast<const char*>(body.data), body.data_length);
  }
  return request;
}

std::string ChunkedRequest(iree_const_byte_span_t body) {
  char size[32];
  const auto result =
      std::to_chars(size, size + sizeof(size), body.data_length, 16);
  if (result.ec != std::errc()) {
    std::abort();
  }
  std::string request =
      "POST /fuzz HTTP/1.1\r\nHost: local\r\nTransfer-Encoding: "
      "chunked\r\n\r\n";
  request.append(size, static_cast<size_t>(result.ptr - size));
  request += "\r\n";
  if (body.data_length) {
    request.append(reinterpret_cast<const char*>(body.data), body.data_length);
  }
  request += "\r\n0\r\n\r\n";
  return request;
}

std::string MultipartRequest(iree_const_byte_span_t contents) {
  std::string body =
      "--loom-fuzz-boundary\r\nContent-Disposition: form-data; name=\"file\"; "
      "filename=\"input.bin\"\r\nContent-Type: "
      "application/octet-stream\r\n\r\n";
  if (contents.data_length) {
    body.append(reinterpret_cast<const char*>(contents.data),
                contents.data_length);
  }
  body += "\r\n--loom-fuzz-boundary--\r\n";
  std::string request =
      "POST /fuzz HTTP/1.1\r\nHost: local\r\nContent-Type: "
      "multipart/form-data; boundary=loom-fuzz-boundary\r\nContent-Length: ";
  request += std::to_string(body.size());
  request += "\r\n\r\n";
  request += body;
  return request;
}

void ExerciseGenerated(const std::string& request) {
  Exercise(iree_make_const_byte_span(request.data(), request.size()));
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size > kMaximumInputLength) {
    return 0;
  }
  const iree_const_byte_span_t input = iree_make_const_byte_span(data, size);
  Exercise(input);
  ExerciseGenerated(ContentLengthRequest(input));
  ExerciseGenerated(ChunkedRequest(input));
  ExerciseGenerated(MultipartRequest(input));
  return 0;
}
