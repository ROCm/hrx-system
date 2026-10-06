// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/http/request.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

constexpr char kPostBody[] =
    R"({"model":"bundle-model","messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"echo","description":"echo","parameters":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}}],"stream":true,"stream_options":{"include_usage":true},"chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}})";
static_assert(sizeof(kPostBody) - 1 == 367);

struct HttpRequestParserDeleter {
  void operator()(loom_serve_http_request_parser_t* parser) const {
    loom_serve_http_request_parser_destroy(parser);
  }
};
using HttpRequestParserPtr =
    std::unique_ptr<loom_serve_http_request_parser_t, HttpRequestParserDeleter>;

struct RequestSnapshot {
  // Copied request method.
  std::string method;
  // Copied origin-form target.
  std::string target;
  // Copied header names and values in wire order.
  std::vector<std::pair<std::string, std::string>> headers;
  // Copied request body.
  std::string body;
};

std::string CopyString(iree_string_view_t value) {
  return std::string(value.data ? value.data : "", value.size);
}

loom_serve_http_request_limits_t DefaultLimits() {
  return {
      .header_byte_capacity = 4096,
      .body_byte_capacity = 4096,
      .header_capacity = 32,
  };
}

std::string CapturedPostRequest() {
  return std::string(
             "POST /v1/chat/completions HTTP/1.1\r\n"
             "host: 127.0.0.1:45991\r\n"
             "connection: keep-alive\r\n"
             "Accept: application/json\r\n"
             "User-Agent: OpenAI/JS 6.26.0\r\n"
             "X-Stainless-Retry-Count: 0\r\n"
             "X-Stainless-Lang: js\r\n"
             "X-Stainless-Package-Version: 6.26.0\r\n"
             "X-Stainless-OS: Linux\r\n"
             "X-Stainless-Arch: x64\r\n"
             "X-Stainless-Runtime: node\r\n"
             "X-Stainless-Runtime-Version: v24.11.1\r\n"
             "authorization: Bearer unused\r\n"
             "x-session-affinity: session-test\r\n"
             "content-type: application/json\r\n"
             "accept-language: *\r\n"
             "sec-fetch-mode: cors\r\n"
             "accept-encoding: gzip, deflate\r\n"
             "content-length: 367\r\n"
             "\r\n") +
         kPostBody;
}

std::string CapturedGetRequest() {
  return "GET /v1/models HTTP/1.1\r\n"
         "host: 127.0.0.1:45523\r\n"
         "connection: keep-alive\r\n"
         "Accept: application/json\r\n"
         "User-Agent: OpenAI/JS 6.26.0\r\n"
         "authorization: Bearer unused\r\n"
         "accept-encoding: gzip, deflate\r\n"
         "\r\n";
}

void CopyRequest(const loom_serve_http_request_t* request,
                 RequestSnapshot* out_snapshot) {
  out_snapshot->method = CopyString(request->method);
  out_snapshot->target = CopyString(request->target);
  for (iree_host_size_t i = 0; i < request->headers.count; ++i) {
    out_snapshot->headers.emplace_back(
        CopyString(request->headers.pairs[i].key),
        CopyString(request->headers.pairs[i].value));
  }
  out_snapshot->body = CopyString(request->body);
}

iree_status_t ParseRequest(const std::vector<iree_const_byte_span_t>& chunks,
                           loom_serve_http_request_limits_t limits,
                           RequestSnapshot* out_snapshot) {
  loom_serve_http_request_parser_t* parser = nullptr;
  IREE_RETURN_IF_ERROR(loom_serve_http_request_parser_create(
      limits, &parser, iree_allocator_system()));
  HttpRequestParserPtr parser_owner(parser);
  const loom_serve_http_request_t* request = nullptr;
  for (iree_const_byte_span_t chunk : chunks) {
    IREE_RETURN_IF_ERROR(
        loom_serve_http_request_parser_feed(parser, chunk, &request));
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_http_request_parser_finalize(parser, &request));
  CopyRequest(request, out_snapshot);
  return iree_ok_status();
}

iree_const_byte_span_t StringSpan(const std::string& value) {
  return iree_make_const_byte_span(value.data(), value.size());
}

void ExpectSnapshotsEqual(const RequestSnapshot& actual,
                          const RequestSnapshot& expected) {
  EXPECT_EQ(actual.method, expected.method);
  EXPECT_EQ(actual.target, expected.target);
  EXPECT_EQ(actual.headers, expected.headers);
  EXPECT_EQ(actual.body, expected.body);
}

TEST(HttpRequestTest, ParsesCapturedOpenAiPostAcrossEverySplit) {
  const std::string wire_request = CapturedPostRequest();
  RequestSnapshot expected;
  IREE_ASSERT_OK(
      ParseRequest({StringSpan(wire_request)}, DefaultLimits(), &expected));
  EXPECT_EQ(expected.method, "POST");
  EXPECT_EQ(expected.target, "/v1/chat/completions");
  EXPECT_EQ(expected.body, kPostBody);
  ASSERT_EQ(expected.headers.size(), 18u);
  EXPECT_EQ(
      expected.headers.front(),
      std::make_pair(std::string("host"), std::string("127.0.0.1:45991")));
  EXPECT_EQ(expected.headers.back(),
            std::make_pair(std::string("content-length"), std::string("367")));

  std::vector<iree_const_byte_span_t> byte_chunks;
  for (size_t i = 0; i < wire_request.size(); ++i) {
    byte_chunks.push_back(
        iree_make_const_byte_span(wire_request.data() + i, 1));
  }
  RequestSnapshot byte_fragmented;
  IREE_ASSERT_OK(ParseRequest(byte_chunks, DefaultLimits(), &byte_fragmented));
  ExpectSnapshotsEqual(byte_fragmented, expected);

  for (size_t split = 1; split < wire_request.size(); ++split) {
    SCOPED_TRACE(split);
    RequestSnapshot fragmented;
    IREE_ASSERT_OK(
        ParseRequest({iree_make_const_byte_span(wire_request.data(), split),
                      iree_make_const_byte_span(wire_request.data() + split,
                                                wire_request.size() - split)},
                     DefaultLimits(), &fragmented));
    ExpectSnapshotsEqual(fragmented, expected);
  }
}

TEST(HttpRequestTest, ParsesCapturedOpenAiGetAndLooksUpHeaders) {
  const std::string wire_request = CapturedGetRequest();
  loom_serve_http_request_parser_t* parser = nullptr;
  IREE_ASSERT_OK(loom_serve_http_request_parser_create(
      DefaultLimits(), &parser, iree_allocator_system()));
  HttpRequestParserPtr parser_owner(parser);
  const loom_serve_http_request_t* request = nullptr;
  IREE_ASSERT_OK(loom_serve_http_request_parser_feed(
      parser, StringSpan(wire_request), &request));
  ASSERT_NE(request, nullptr);
  EXPECT_TRUE(iree_string_view_equal(request->method, IREE_SV("GET")));
  EXPECT_TRUE(iree_string_view_equal(request->target, IREE_SV("/v1/models")));
  EXPECT_TRUE(iree_string_view_is_empty(request->body));

  iree_string_view_t value;
  EXPECT_TRUE(
      loom_serve_http_request_lookup_header(request, IREE_SV("HOST"), &value));
  EXPECT_TRUE(iree_string_view_equal(value, IREE_SV("127.0.0.1:45523")));
  EXPECT_FALSE(loom_serve_http_request_lookup_header(
      request, IREE_SV("missing"), &value));
  EXPECT_TRUE(iree_string_view_is_empty(value));
}

TEST(HttpRequestTest, TrimsHeaderWhitespace) {
  const std::string wire_request =
      "GET /resource?value=1 HTTP/1.1\r\n"
      "Host:\texample.test \t\r\n"
      "X-Value:\t value with spaces \t\r\n"
      "Content-Encoding: identity\r\n"
      "\r\n";
  RequestSnapshot snapshot;
  IREE_ASSERT_OK(
      ParseRequest({StringSpan(wire_request)}, DefaultLimits(), &snapshot));
  EXPECT_EQ(snapshot.target, "/resource?value=1");
  EXPECT_EQ(snapshot.headers[0].second, "example.test");
  EXPECT_EQ(snapshot.headers[1].second, "value with spaces");
}

TEST(HttpRequestTest, RejectsAmbiguousOrUnsupportedFraming) {
  const std::vector<std::string> invalid_requests = {
      "GET / HTTP/1.0\r\nHost: local\r\n\r\n",
      "GET http://local/ HTTP/1.1\r\nHost: local\r\n\r\n",
      "GET /#fragment HTTP/1.1\r\nHost: local\r\n\r\n",
      "GET / HTTP/1.1\r\nUser-Agent: test\r\n\r\n",
      "GET / HTTP/1.1\r\nHost: local\r\nHost: duplicate\r\n\r\n",
      "POST / HTTP/1.1\r\nHost: local\r\nContent-Length: 0\r\n"
      "Content-Length: 0\r\n\r\n",
      "POST / HTTP/1.1\r\nHost: local\r\nTransfer-Encoding: chunked\r\n"
      "\r\n0\r\n\r\n",
      "POST / HTTP/1.1\r\nHost: local\r\nContent-Length: 0\r\n"
      "Transfer-Encoding: chunked\r\n\r\n",
      "POST / HTTP/1.1\r\nHost: local\r\nExpect: 100-continue\r\n"
      "Content-Length: 0\r\n\r\n",
      "POST / HTTP/1.1\r\nHost: local\r\nContent-Encoding: gzip\r\n"
      "Content-Length: 0\r\n\r\n",
      "POST / HTTP/1.1\r\nHost: local\r\nContent-Length: +1\r\n\r\nx",
      "POST / HTTP/1.1\r\nHost: local\r\nContent-Length: 1 0\r\n\r\n",
      "GET / HTTP/1.1\r\nHost : local\r\n\r\n",
      "GET / HTTP/1.1\r\nHost: local\r\n folded: value\r\n\r\n",
      "GET / HTTP/1.1\nHost: local\r\n\r\n",
      "GET / HTTP/1.1\r\nHost: local\r\n\r\ntrailing",
  };
  for (size_t i = 0; i < invalid_requests.size(); ++i) {
    SCOPED_TRACE(i);
    RequestSnapshot ignored;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          ParseRequest({StringSpan(invalid_requests[i])},
                                       DefaultLimits(), &ignored));
  }

  const std::string overflow_request =
      "POST / HTTP/1.1\r\nHost: local\r\nContent-Length: " +
      std::string(128, '9') + "\r\n\r\n";
  RequestSnapshot ignored;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      ParseRequest({StringSpan(overflow_request)}, DefaultLimits(), &ignored));
}

TEST(HttpRequestTest, ReportsTruncatedHeadersAndBodyAtEof) {
  RequestSnapshot ignored;
  const std::string partial_headers = "GET / HTTP/1.1\r\nHost: local\r\n";
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      ParseRequest({StringSpan(partial_headers)}, DefaultLimits(), &ignored));

  const std::string partial_body =
      "POST / HTTP/1.1\r\nHost: local\r\nContent-Length: 4\r\n\r\nabc";
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      ParseRequest({StringSpan(partial_body)}, DefaultLimits(), &ignored));
}

TEST(HttpRequestTest, EnforcesHeaderBodyAndRecordCapacities) {
  RequestSnapshot ignored;
  loom_serve_http_request_limits_t limits = DefaultLimits();
  limits.header_byte_capacity = 16;
  const std::string request = CapturedGetRequest();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        ParseRequest({StringSpan(request)}, limits, &ignored));

  limits = DefaultLimits();
  limits.header_capacity = 1;
  const std::string two_headers =
      "GET / HTTP/1.1\r\nHost: local\r\nX-Value: one\r\n\r\n";
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      ParseRequest({StringSpan(two_headers)}, limits, &ignored));

  limits = DefaultLimits();
  limits.body_byte_capacity = 3;
  const std::string large_body =
      "POST / HTTP/1.1\r\nHost: local\r\nContent-Length: 4\r\n\r\nfour";
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      ParseRequest({StringSpan(large_body)}, limits, &ignored));
}

TEST(HttpRequestTest, TerminalStatesRejectAdditionalInput) {
  const std::string valid_request = CapturedGetRequest();
  loom_serve_http_request_parser_t* parser = nullptr;
  IREE_ASSERT_OK(loom_serve_http_request_parser_create(
      DefaultLimits(), &parser, iree_allocator_system()));
  HttpRequestParserPtr parser_owner(parser);
  const loom_serve_http_request_t* request = nullptr;
  IREE_ASSERT_OK(loom_serve_http_request_parser_feed(
      parser, StringSpan(valid_request), &request));
  ASSERT_NE(request, nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_http_request_parser_feed(
                            parser, iree_const_byte_span_empty(), &request));

  loom_serve_http_request_parser_t* failed_parser = nullptr;
  IREE_ASSERT_OK(loom_serve_http_request_parser_create(
      DefaultLimits(), &failed_parser, iree_allocator_system()));
  HttpRequestParserPtr failed_parser_owner(failed_parser);
  const std::string invalid_request = "GET / HTTP/1.0\r\nHost: local\r\n\r\n";
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_http_request_parser_feed(
          failed_parser, StringSpan(invalid_request), &request));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_serve_http_request_parser_finalize(failed_parser, &request));
}

}  // namespace
