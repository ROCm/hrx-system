// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/transport/websocket/handshake.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class WebSocketHandshakeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_string_builder_initialize(iree_allocator_system(), &response_);
  }
  void TearDown() override { iree_string_builder_deinitialize(&response_); }

  iree_status_t Handshake(
      const std::string& headers,
      iree_string_view_list_t origins = iree_string_view_list_empty(),
      iree_string_view_t subprotocol = iree_string_view_empty()) {
    loom_serve_http_request_parser_t* parser = nullptr;
    const loom_serve_http_request_limits_t limits = {4096, 512, 32, 4096, 32};
    IREE_RETURN_IF_ERROR(loom_serve_http_request_parser_allocate(
        limits, iree_allocator_system(), &parser));
    const loom_serve_http_request_t* request = nullptr;
    iree_status_t status = loom_serve_http_request_parser_feed(
        parser, iree_make_const_byte_span(headers.data(), headers.size()),
        &request);
    if (iree_status_is_ok(status)) {
      status = loom_serve_websocket_handshake(request, origins, subprotocol,
                                              &response_);
    }
    loom_serve_http_request_parser_free(parser);
    return status;
  }

  std::string Response() const {
    const auto value = iree_string_builder_view(&response_);
    return std::string(value.data ? value.data : "", value.size);
  }

  static std::string Request(const std::string& extra = "") {
    return "GET /v1/realtime?model=assistant HTTP/1.1\r\nHost: local\r\n"
           "Upgrade: WebSocket\r\nConnection: keep-alive, Upgrade\r\n"
           "Sec-WebSocket-Version: 13\r\n"
           "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n" +
           extra + "\r\n";
  }

  // Owned handshake output reused only after explicit clear between cases.
  iree_string_builder_t response_;
};

TEST_F(WebSocketHandshakeTest, RfcChallengeAndOfferedSubprotocol) {
  IREE_ASSERT_OK(
      Handshake(Request("Sec-WebSocket-Protocol: other, realtime\r\n"
                        "Sec-WebSocket-Extensions: permessage-deflate\r\n"),
                iree_string_view_list_empty(), IREE_SV("realtime")));
  EXPECT_EQ(Response(),
            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n"
            "Sec-WebSocket-Protocol: realtime\r\n\r\n");
}

TEST_F(WebSocketHandshakeTest,
       BrowserOriginsAreExplicitAndSubprotocolIsOptional) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        Handshake(Request("Origin: https://chat.example\r\n")));
  EXPECT_EQ(Response(), "");
  const iree_string_view_t origins[] = {IREE_SV("https://chat.example")};
  IREE_ASSERT_OK(Handshake(Request("Origin: https://chat.example\r\n"),
                           {1, origins}, IREE_SV("realtime")));
  EXPECT_EQ(Response(),
            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n");
  iree_string_builder_reset(&response_);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      Handshake(Request("Origin: https://chat.example.attacker\r\n"),
                {1, origins}));
}

TEST_F(WebSocketHandshakeTest, RejectsInvalidUpgradeBeforeWritingResponse) {
  const std::pair<std::string, std::string> changes[] = {
      {"GET /", "POST /"},
      {"Host: local", "Host: "},
      {"Upgrade: WebSocket", "Upgrade: websocket-extra"},
      {"keep-alive, Upgrade", "keep-alive"},
      {"Version: 13", "Version: 8"},
      {"dGhlIHNhbXBsZSBub25jZQ==", "dGhlIHNhbXBsZSBub25jZQ"},
      {"dGhlIHNhbXBsZSBub25jZQ==", "dGhlIHNhbXBsZSBub25j!Q=="},
  };
  for (const auto& [from, to] : changes) {
    SCOPED_TRACE(from);
    std::string request = Request();
    request.replace(request.find(from), from.size(), to);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Handshake(request));
    EXPECT_EQ(Response(), "");
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Handshake(Request("Content-Length: 1\r\n") + "x"));
}

}  // namespace
