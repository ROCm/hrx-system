// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/transport/http/request.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/transport/http/multipart.h"

namespace {

constexpr char kPostBody[] =
    R"({"model":"bundle-model","messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"echo","description":"echo","parameters":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}}],"stream":true,"stream_options":{"include_usage":true},"chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}})";
static_assert(sizeof(kPostBody) - 1 == 367);

struct loom_serve_test_http_request_parser_deleter_t {
  void operator()(loom_serve_http_request_parser_t* parser) const {
    loom_serve_http_request_parser_free(parser);
  }
};
using HttpRequestParserPtr =
    std::unique_ptr<loom_serve_http_request_parser_t,
                    loom_serve_test_http_request_parser_deleter_t>;

struct loom_serve_test_http_request_snapshot_t {
  // Copied request method.
  std::string method;
  // Copied origin-form target.
  std::string target;
  // Encoded path used for dispatch.
  std::string path;
  // Copied header names and values in wire order.
  std::vector<std::pair<std::string, std::string>> headers;
  // Copied request body.
  std::string body;
  // Copied ordered multipart names and binary payloads, when present.
  std::vector<std::pair<std::string, std::string>> form_fields;
};

std::string CopyString(iree_string_view_t value) {
  return std::string(value.data ? value.data : "", value.size);
}

loom_serve_http_request_limits_t DefaultLimits() {
  return {
      .header_byte_capacity = 4096,
      .body_byte_capacity = 4096,
      .header_capacity = 32,
      .multipart_header_byte_capacity = 4096,
      .multipart_part_capacity = 32,
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

std::string ChunkedRequest(const std::string& encoded_body) {
  return "POST /v1/audio/transcriptions HTTP/1.1\r\n"
         "Host: local\r\n"
         "Transfer-Encoding: chunked\r\n\r\n" +
         encoded_body;
}

void CopyRequest(const loom_serve_http_request_t* request,
                 loom_serve_test_http_request_snapshot_t* out_snapshot) {
  out_snapshot->method = CopyString(request->method);
  out_snapshot->target = CopyString(request->target);
  out_snapshot->path = CopyString(request->path);
  for (iree_host_size_t i = 0; i < request->headers.count; ++i) {
    out_snapshot->headers.emplace_back(
        CopyString(request->headers.pairs[i].key),
        CopyString(request->headers.pairs[i].value));
  }
  out_snapshot->body = CopyString(request->body);
  const auto parts = loom_serve_http_multipart_parts(request->multipart);
  for (size_t i = 0; i < parts.count; ++i) {
    out_snapshot->form_fields.emplace_back(
        CopyString(parts.values[i].name), CopyString(parts.values[i].contents));
  }
}

iree_status_t ParseRequest(
    const std::vector<iree_const_byte_span_t>& chunks,
    loom_serve_http_request_limits_t limits,
    loom_serve_test_http_request_snapshot_t* out_snapshot) {
  loom_serve_http_request_parser_t* parser = nullptr;
  IREE_RETURN_IF_ERROR(loom_serve_http_request_parser_allocate(
      limits, iree_allocator_system(), &parser));
  HttpRequestParserPtr parser_owner(parser);
  const loom_serve_http_request_t* request = nullptr;
  for (iree_const_byte_span_t chunk : chunks) {
    IREE_RETURN_IF_ERROR(
        loom_serve_http_request_parser_feed(parser, chunk, &request, nullptr));
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_http_request_parser_finalize(parser, &request));
  CopyRequest(request, out_snapshot);
  return iree_ok_status();
}

iree_const_byte_span_t StringSpan(const std::string& value) {
  return iree_make_const_byte_span(value.data(), value.size());
}

void ExpectSnapshotsEqual(
    const loom_serve_test_http_request_snapshot_t& actual,
    const loom_serve_test_http_request_snapshot_t& expected) {
  EXPECT_EQ(actual.method, expected.method);
  EXPECT_EQ(actual.target, expected.target);
  EXPECT_EQ(actual.path, expected.path);
  EXPECT_EQ(actual.headers, expected.headers);
  EXPECT_EQ(actual.body, expected.body);
  EXPECT_EQ(actual.form_fields, expected.form_fields);
}

void ExpectEverySplit(const std::string& wire_request,
                      const loom_serve_test_http_request_snapshot_t& expected) {
  std::vector<iree_const_byte_span_t> byte_chunks;
  for (size_t i = 0; i < wire_request.size(); ++i) {
    byte_chunks.push_back(
        iree_make_const_byte_span(wire_request.data() + i, 1));
  }
  loom_serve_test_http_request_snapshot_t byte_fragmented;
  IREE_ASSERT_OK(ParseRequest(byte_chunks, DefaultLimits(), &byte_fragmented));
  ExpectSnapshotsEqual(byte_fragmented, expected);
  for (size_t split = 1; split < wire_request.size(); ++split) {
    SCOPED_TRACE(split);
    loom_serve_test_http_request_snapshot_t fragmented;
    IREE_ASSERT_OK(
        ParseRequest({iree_make_const_byte_span(wire_request.data(), split),
                      iree_make_const_byte_span(wire_request.data() + split,
                                                wire_request.size() - split)},
                     DefaultLimits(), &fragmented));
    ExpectSnapshotsEqual(fragmented, expected);
  }
}

TEST(HttpRequestTest, ParsesCapturedOpenAiPostAcrossEverySplit) {
  const std::string wire_request = CapturedPostRequest();
  loom_serve_test_http_request_snapshot_t expected;
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

  ExpectEverySplit(wire_request, expected);
}

TEST(HttpRequestTest, ParsesChunkedMultipartAcrossEverySplit) {
  // aiohttp FormData sends boundaries, field headers, values and file reads as
  // separate chunks. Binary file bytes are opaque to HTTP request framing.
  const std::string file_bytes("a\0\r\nb", 5);
  const std::string wire_request =
      "POST /v1/audio/transcriptions HTTP/1.1\r\n"
      "Host: local\r\n"
      "Content-Type: multipart/form-data; boundary=sample\r\n"
      "Transfer-Encoding: ChUnKeD\r\n\r\n"
      "a\r\n--sample\r\n\r\n"
      "30\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n\r\n"
      "7\r\nfixture\r\n"
      "2\r\n\r\n\r\n"
      "A\r\n--sample\r\n\r\n"
      "49\r\nContent-Disposition: form-data; name=\"file\"; "
      "filename=\"recording.wav\"\r\n\r\n\r\n"
      "5\r\n" +
      file_bytes + "\r\n2\r\n\r\n\r\nc\r\n--sample--\r\n\r\n0\r\n\r\n";
  loom_serve_test_http_request_snapshot_t expected{
      .method = "POST",
      .target = "/v1/audio/transcriptions",
      .path = "/v1/audio/transcriptions",
      .headers = {{"Host", "local"},
                  {"Content-Type", "multipart/form-data; boundary=sample"},
                  {"Transfer-Encoding", "ChUnKeD"}},
      .body =
          "--sample\r\n"
          "Content-Disposition: form-data; name=\"model\"\r\n\r\n"
          "fixture\r\n--sample\r\n"
          "Content-Disposition: form-data; name=\"file\"; "
          "filename=\"recording.wav\"\r\n\r\n" +
          file_bytes + "\r\n--sample--\r\n",
      .form_fields = {{"model", "fixture"}, {"file", file_bytes}},
  };
  loom_serve_test_http_request_snapshot_t actual;
  IREE_ASSERT_OK(
      ParseRequest({StringSpan(wire_request)}, DefaultLimits(), &actual));
  ExpectSnapshotsEqual(actual, expected);
  ExpectEverySplit(wire_request, expected);
}

TEST(HttpRequestTest, IgnoresChunkExtensionsAndDiscardsTrailers) {
  const std::string wire_request = ChunkedRequest(
      "000a ; flag\t; value = token; quoted=\"semi;quote\\\"slash\\\\\"\r\n"
      "0123456789\r\n"
      "1;empty=\"\"\r\nx\r\n"
      "000;last=yes\r\n"
      "X-Checksum: ignored\r\nHost: different\r\nContent-Length: 999\r\n\r\n");
  const loom_serve_test_http_request_snapshot_t expected{
      .method = "POST",
      .target = "/v1/audio/transcriptions",
      .path = "/v1/audio/transcriptions",
      .headers = {{"Host", "local"}, {"Transfer-Encoding", "chunked"}},
      .body = "0123456789x",
  };
  loom_serve_test_http_request_snapshot_t actual;
  IREE_ASSERT_OK(
      ParseRequest({StringSpan(wire_request)}, DefaultLimits(), &actual));
  ExpectSnapshotsEqual(actual, expected);
  ExpectEverySplit(wire_request, expected);
}

TEST(HttpRequestTest, EmptyChunkedBodyHonorsZeroBodyCapacity) {
  auto limits = DefaultLimits();
  limits.body_byte_capacity = 0;
  loom_serve_test_http_request_snapshot_t actual;
  const std::string wire_request = ChunkedRequest("0\r\n\r\n");
  IREE_ASSERT_OK(ParseRequest({StringSpan(wire_request)}, limits, &actual));
  EXPECT_TRUE(actual.body.empty());
  ExpectEverySplit(wire_request, actual);
}

TEST(HttpRequestTest, PreservesHeaderViewsAcrossDecodedBodyGrowth) {
  const std::string body(32768, 'x');
  const std::string prefix = ChunkedRequest("8000\r\n");
  const std::string suffix = "\r\n0\r\n\r\n";
  auto limits = DefaultLimits();
  limits.body_byte_capacity = body.size();
  loom_serve_test_http_request_snapshot_t actual;
  IREE_ASSERT_OK(
      ParseRequest({StringSpan(prefix), StringSpan(body), StringSpan(suffix)},
                   limits, &actual));
  EXPECT_EQ(actual.body, body);
  EXPECT_EQ(actual.method, "POST");
  EXPECT_EQ(actual.headers[0],
            std::make_pair(std::string("Host"), std::string("local")));
  loom_serve_test_http_request_snapshot_t coalesced;
  const std::string wire_request = prefix + body + suffix;
  IREE_ASSERT_OK(ParseRequest({StringSpan(wire_request)}, limits, &coalesced));
  ExpectSnapshotsEqual(coalesced, actual);
}

TEST(HttpRequestTest, ParsesCapturedOpenAiGetAndLooksUpHeaders) {
  const std::string wire_request = CapturedGetRequest();
  loom_serve_http_request_parser_t* parser = nullptr;
  IREE_ASSERT_OK(loom_serve_http_request_parser_allocate(
      DefaultLimits(), iree_allocator_system(), &parser));
  HttpRequestParserPtr parser_owner(parser);
  const loom_serve_http_request_t* request = nullptr;
  IREE_ASSERT_OK(loom_serve_http_request_parser_feed(
      parser, StringSpan(wire_request), &request, nullptr));
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

TEST(HttpRequestTest, SplitsEncodedPathAndTrimsHeaderWhitespace) {
  const std::string wire_request =
      "GET /name%3Fpart?next=/other?value=1 HTTP/1.1\r\n"
      "Host:\texample.test \t\r\n"
      "X-Value:\t value with spaces \t\r\n"
      "Content-Encoding: identity\r\n"
      "\r\n";
  loom_serve_test_http_request_snapshot_t snapshot;
  IREE_ASSERT_OK(
      ParseRequest({StringSpan(wire_request)}, DefaultLimits(), &snapshot));
  EXPECT_EQ(snapshot.target, "/name%3Fpart?next=/other?value=1");
  EXPECT_EQ(snapshot.path, "/name%3Fpart");
  EXPECT_EQ(snapshot.headers[0].second, "example.test");
  EXPECT_EQ(snapshot.headers[1].second, "value with spaces");
  ExpectEverySplit(wire_request, snapshot);
}

TEST(HttpRequestTest, RequestsContinueOnlyWhenHeadersArriveAheadOfTheBody) {
  const std::pair<std::string, std::string> encodings[] = {
      {"Content-Length: 5", "hello"},
      {"Transfer-Encoding: chunked", "5\r\nhello\r\n0\r\n\r\n"},
  };
  for (const auto& [framing, body] : encodings) {
    SCOPED_TRACE(framing);
    const std::string headers =
        "POST / HTTP/1.1\r\nHost: local\r\nExpect: 100-CoNtInUe\r\n" + framing +
        "\r\n\r\n";
    const std::string wire = headers + body;
    for (size_t split = 0; split < wire.size(); ++split) {
      SCOPED_TRACE(split);
      loom_serve_http_request_parser_t* parser = nullptr;
      IREE_ASSERT_OK(loom_serve_http_request_parser_allocate(
          DefaultLimits(), iree_allocator_system(), &parser));
      HttpRequestParserPtr parser_owner(parser);
      const loom_serve_http_request_t* request = nullptr;
      bool send_continue = false;
      IREE_ASSERT_OK(loom_serve_http_request_parser_feed(
          parser, iree_make_const_byte_span(wire.data(), split), &request,
          &send_continue));
      EXPECT_EQ(send_continue, split == headers.size());
      EXPECT_EQ(request, nullptr);
      IREE_ASSERT_OK(loom_serve_http_request_parser_feed(
          parser,
          iree_make_const_byte_span(wire.data() + split, wire.size() - split),
          &request, &send_continue));
      EXPECT_FALSE(send_continue);
      ASSERT_NE(request, nullptr);
      EXPECT_EQ(CopyString(request->body), "hello");
    }
  }
}

TEST(HttpRequestTest, SkipsContinueForEmptyAndOversizedBodies) {
  for (bool oversized : {false, true}) {
    SCOPED_TRACE(oversized);
    const auto limits = DefaultLimits();
    const std::string headers =
        "POST / HTTP/1.1\r\nHost: local\r\nExpect: 100-continue\r\n"
        "Content-Length: " +
        std::to_string(oversized ? limits.body_byte_capacity + 1 : 0) +
        "\r\n\r\n";
    loom_serve_http_request_parser_t* parser = nullptr;
    IREE_ASSERT_OK(loom_serve_http_request_parser_allocate(
        limits, iree_allocator_system(), &parser));
    HttpRequestParserPtr parser_owner(parser);
    const loom_serve_http_request_t* request = nullptr;
    bool send_continue = true;
    iree_status_t status = loom_serve_http_request_parser_feed(
        parser, StringSpan(headers), &request, &send_continue);
    EXPECT_FALSE(send_continue);
    if (oversized) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
      EXPECT_EQ(request, nullptr);
    } else {
      IREE_EXPECT_OK(status);
      ASSERT_NE(request, nullptr);
      EXPECT_EQ(request->body.size, 0u);
    }
  }
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
      "POST / HTTP/1.1\r\nHost: local\r\nTransfer-Encoding: gzip\r\n\r\n",
      "POST / HTTP/1.1\r\nHost: local\r\nTransfer-Encoding: gzip, "
      "chunked\r\n\r\n",
      "POST / HTTP/1.1\r\nHost: local\r\nTransfer-Encoding: chunked, "
      "chunked\r\n\r\n",
      "POST / HTTP/1.1\r\nHost: local\r\nTransfer-Encoding: chunked; "
      "a=b\r\n\r\n",
      "POST / HTTP/1.1\r\nHost: local\r\nTransfer-Encoding: chunked\r\n"
      "Transfer-Encoding: chunked\r\n\r\n",
      "POST / HTTP/1.1\r\nHost: local\r\nContent-Length: 0\r\n"
      "Transfer-Encoding: chunked\r\n\r\n",
      "POST / HTTP/1.1\r\nHost: local\r\nTransfer-Encoding: chunked\r\n"
      "Content-Length: 0\r\n\r\n",
      "POST / HTTP/1.1\r\nHost: local\r\nExpect: unknown\r\n"
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
    loom_serve_test_http_request_snapshot_t ignored;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          ParseRequest({StringSpan(invalid_requests[i])},
                                       DefaultLimits(), &ignored));
  }

  const std::string overflow_request =
      "POST / HTTP/1.1\r\nHost: local\r\nContent-Length: " +
      std::string(128, '9') + "\r\n\r\n";
  loom_serve_test_http_request_snapshot_t ignored;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      ParseRequest({StringSpan(overflow_request)}, DefaultLimits(), &ignored));
}

TEST(HttpRequestTest, ReportsTruncatedHeadersAndBodyAtEof) {
  loom_serve_test_http_request_snapshot_t ignored;
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

TEST(HttpRequestTest, RejectsMalformedChunkFraming) {
  const std::vector<std::string> invalid_bodies = {
      "\r\n",
      "+1\r\nx\r\n0\r\n\r\n",
      "-1\r\n",
      "0x1\r\n",
      "g\r\n",
      "1 \r\n",
      "1;\r\n",
      "1;=value\r\n",
      "1;a=\r\n",
      "1;a=\"unterminated\r\n",
      "1;a=\"x\"junk\r\n",
      "1;a=one two\r\n",
      "1;a=\"bad\rcontrol\"\r\n",
      "1;a=\"bad\\\rcontrol\"\r\n",
      "1\nx\r\n0\r\n\r\n",
      "1\r\nx\n0\r\n\r\n",
      "1\r\nx\rX0\r\n\r\n",
      "0\r\n\n",
      "0\r\nBad-Trailer\r\n\r\n",
      "0\r\n Folded: no\r\n\r\n",
      "0\r\nField: bad\rvalue\r\n\r\n",
      "0\r\n\r\ntrailing",
      "0\r\n\r\nGET / HTTP/1.1\r\nHost: local\r\n\r\n",
      std::string(128, 'f') + "\r\n",
  };
  for (size_t i = 0; i < invalid_bodies.size(); ++i) {
    SCOPED_TRACE(i);
    const std::string wire_request = ChunkedRequest(invalid_bodies[i]);
    loom_serve_test_http_request_snapshot_t ignored;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        ParseRequest({StringSpan(wire_request)}, DefaultLimits(), &ignored));
  }
}

TEST(HttpRequestTest, ChunkedRequestRequiresTerminalTrailerDelimiter) {
  const std::string wire_request = ChunkedRequest(
      "3\r\nabc\r\n2;next=yes\r\nde\r\n0\r\nX-End: true\r\n\r\n");
  for (size_t length = 0; length < wire_request.size(); ++length) {
    SCOPED_TRACE(length);
    loom_serve_http_request_parser_t* parser = nullptr;
    IREE_ASSERT_OK(loom_serve_http_request_parser_allocate(
        DefaultLimits(), iree_allocator_system(), &parser));
    HttpRequestParserPtr parser_owner(parser);
    const loom_serve_http_request_t* request = nullptr;
    IREE_ASSERT_OK(loom_serve_http_request_parser_feed(
        parser, iree_make_const_byte_span(wire_request.data(), length),
        &request, nullptr));
    EXPECT_EQ(request, nullptr);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_DATA_LOSS,
        loom_serve_http_request_parser_finalize(parser, &request));
    EXPECT_EQ(request, nullptr);
  }
}

TEST(HttpRequestTest, BoundsDecodedContentAndCumulativeFramingIndependently) {
  auto limits = DefaultLimits();
  limits.body_byte_capacity = 4;
  const std::string exact_body =
      ChunkedRequest("2\r\nab\r\n2\r\ncd\r\n0\r\n\r\n");
  loom_serve_test_http_request_snapshot_t actual;
  IREE_ASSERT_OK(ParseRequest({StringSpan(exact_body)}, limits, &actual));
  EXPECT_EQ(actual.body, "abcd");
  for (const char* encoded_body : {"5\r\n", "2\r\nab\r\n3\r\n"}) {
    const std::string wire_request = ChunkedRequest(encoded_body);
    loom_serve_test_http_request_snapshot_t ignored;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        ParseRequest({StringSpan(wire_request)}, limits, &ignored));
  }

  limits.header_byte_capacity = 128;
  // "0;" + name + CRLF + empty trailer line has exactly 128 framing bytes.
  const std::string exact_framing =
      ChunkedRequest("0;" + std::string(122, 'a') + "\r\n\r\n");
  loom_serve_test_http_request_snapshot_t empty;
  IREE_ASSERT_OK(ParseRequest({StringSpan(exact_framing)}, limits, &empty));
  EXPECT_TRUE(empty.body.empty());
  const std::vector<std::string> excessive_framing = {
      "0;" + std::string(126, 'a'),
      "0;" + std::string(123, 'a') + "\r\n\r\n",
      "0\r\nX: " + std::string(123, 'a') + "\r\n\r\n",
      "1;a=" + std::string(60, 'a') + "\r\nx\r\n1;b=" + std::string(60, 'b') +
          "\r\ny\r\n0\r\n\r\n",
  };
  for (const auto& encoded_body : excessive_framing) {
    const std::string wire_request = ChunkedRequest(encoded_body);
    loom_serve_test_http_request_snapshot_t ignored;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        ParseRequest({StringSpan(wire_request)}, limits, &ignored));
    std::vector<iree_const_byte_span_t> fragments;
    for (size_t i = 0; i < wire_request.size(); ++i) {
      fragments.push_back(
          iree_make_const_byte_span(wire_request.data() + i, 1));
    }
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          ParseRequest(fragments, limits, &ignored));
  }
}

TEST(HttpRequestTest, EnforcesHeaderBodyAndRecordCapacities) {
  loom_serve_test_http_request_snapshot_t ignored;
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
  IREE_ASSERT_OK(loom_serve_http_request_parser_allocate(
      DefaultLimits(), iree_allocator_system(), &parser));
  HttpRequestParserPtr parser_owner(parser);
  const loom_serve_http_request_t* request = nullptr;
  IREE_ASSERT_OK(loom_serve_http_request_parser_feed(
      parser, StringSpan(valid_request), &request, nullptr));
  ASSERT_NE(request, nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_serve_http_request_parser_feed(parser, iree_const_byte_span_empty(),
                                          &request, nullptr));

  loom_serve_http_request_parser_t* failed_parser = nullptr;
  IREE_ASSERT_OK(loom_serve_http_request_parser_allocate(
      DefaultLimits(), iree_allocator_system(), &failed_parser));
  HttpRequestParserPtr failed_parser_owner(failed_parser);
  const std::string invalid_request = "GET / HTTP/1.0\r\nHost: local\r\n\r\n";
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_http_request_parser_feed(
          failed_parser, StringSpan(invalid_request), &request, nullptr));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_serve_http_request_parser_finalize(failed_parser, &request));
}

}  // namespace
