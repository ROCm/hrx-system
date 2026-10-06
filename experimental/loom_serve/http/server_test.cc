// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/http/server.h"

#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>

#include "iree/async/proactor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class HttpServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_signal_block_default());
    auto options = loom_serve_http_server_options_default();
    options.port = 0;
    options.connection_capacity = 24;
    options.request_limits.body_byte_capacity = 2 * 1024 * 1024;
    IREE_ASSERT_OK(loom_serve_http_server_create(&options, &server_,
                                                 iree_allocator_system()));
  }

  void TearDown() override {
    IREE_EXPECT_OK(loom_serve_http_server_destroy(server_));
  }

  int Connect() {
    const auto* address = loom_serve_http_server_address(server_);
    int client = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    EXPECT_GE(client, 0);
    EXPECT_EQ(
        connect(client, reinterpret_cast<const sockaddr*>(address->storage),
                address->length),
        0);
    return client;
  }

  loom_serve_http_connection_t* Take(
      const loom_serve_http_request_t** out_request) {
    auto* notification = loom_serve_http_server_notification(server_);
    while (true) {
      auto token = iree_notification_prepare_wait(notification);
      auto* connection =
          loom_serve_http_server_take_request(server_, out_request);
      if (connection || loom_serve_http_server_is_stopping(server_)) {
        iree_notification_cancel_wait(notification);
        return connection;
      }
      iree_notification_commit_wait(notification, token, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }

  static ::testing::AssertionResult Write(int client,
                                          const std::string& bytes) {
    size_t offset = 0;
    while (offset < bytes.size()) {
      ssize_t count = send(client, bytes.data() + offset, bytes.size() - offset,
                           MSG_NOSIGNAL);
      if (count <= 0) {
        return ::testing::AssertionFailure()
               << "send failed: " << strerror(errno);
      }
      offset += count;
    }
    return ::testing::AssertionSuccess();
  }

  static std::string Read(int client) {
    std::string result;
    char storage[256];
    ssize_t count = 0;
    while ((count = recv(client, storage, sizeof(storage), 0)) > 0) {
      result.append(storage, count);
    }
    EXPECT_EQ(count, 0);
    return result;
  }

  // Server under test, owning its real proactor thread and TCP carriers.
  loom_serve_http_server_t* server_ = nullptr;
};

TEST_F(HttpServerTest, FragmentedRequestAndOrderedStreamingResponse) {
  int client = Connect();
  ASSERT_TRUE(Write(
      client, "POST /v1/chat/completions HTTP/1.1\r\nHost: local\r\nContent-"));
  ASSERT_TRUE(Write(client, "Length: 5\r\n\r\nhe"));
  ASSERT_TRUE(Write(client, "llo"));
  ASSERT_EQ(shutdown(client, SHUT_WR), 0);
  const loom_serve_http_request_t* request = nullptr;
  auto* connection = Take(&request);
  ASSERT_NE(connection, nullptr);
  EXPECT_EQ(std::string(request->body.data, request->body.size), "hello");
  EXPECT_TRUE(loom_serve_http_connection_can_send(connection));
  IREE_ASSERT_OK(loom_serve_http_connection_send(
      connection, IREE_SV("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n")));
  IREE_ASSERT_OK(loom_serve_http_connection_send(connection, IREE_SV("first")));
  IREE_ASSERT_OK(
      loom_serve_http_connection_send(connection, IREE_SV("second")));
  loom_serve_http_connection_finish(connection);
  EXPECT_EQ(Read(client),
            "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nfirstsecond");
  ASSERT_EQ(close(client), 0);
}

TEST_F(HttpServerTest, ConnectionSlotsRecycleAfterDrain) {
  for (int iteration = 0; iteration < 32; ++iteration) {
    SCOPED_TRACE(iteration);
    int client = Connect();
    ASSERT_TRUE(Write(client, "GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n"));
    const loom_serve_http_request_t* request = nullptr;
    auto* connection = Take(&request);
    ASSERT_NE(connection, nullptr);
    IREE_ASSERT_OK(loom_serve_http_connection_send(
        connection, IREE_SV("HTTP/1.1 200 OK\r\nContent-Length: "
                            "2\r\nConnection: close\r\n\r\nOK")));
    loom_serve_http_connection_finish(connection);
    EXPECT_EQ(
        Read(client),
        "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK");
    ASSERT_EQ(close(client), 0);
  }
}

TEST_F(HttpServerTest, ConfiguredBodiesAndClaimsOutliveOtherPeers) {
  int clients[20];
  loom_serve_http_connection_t* connections[20];
  const loom_serve_http_request_t* requests[20];
  // More than the former fixed connection limit can wait for application
  // admission. A long request is allowed by this server's byte budget.
  const std::string body(1024 * 1024 + 1, 'x');
  for (int i = 0; i < 20; ++i) {
    clients[i] = Connect();
    const std::string payload = i == 0 ? body : std::to_string(i);
    ASSERT_TRUE(Write(clients[i],
                      "POST /v1/chat/completions HTTP/1.1\r\nHost: "
                      "local\r\nContent-Length: " +
                          std::to_string(payload.size()) + "\r\n\r\n" +
                          payload));
    connections[i] = Take(&requests[i]);
    ASSERT_NE(connections[i], nullptr);
  }
  for (int i = 19; i >= 0; --i) {
    EXPECT_EQ(std::string(requests[i]->body.data, requests[i]->body.size),
              i == 0 ? body : std::to_string(i));
    loom_serve_http_connection_abort(connections[i]);
    EXPECT_EQ(Read(clients[i]), "");
    ASSERT_EQ(close(clients[i]), 0);
  }
}

TEST_F(HttpServerTest, PeerResetReturnsOwnershipAndListenerSurvives) {
  int client = Connect();
  ASSERT_TRUE(Write(client, "GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n"));
  const loom_serve_http_request_t* request = nullptr;
  auto* connection = Take(&request);
  ASSERT_NE(connection, nullptr);
  const linger reset = {1, 0};
  ASSERT_EQ(setsockopt(client, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
            0);
  ASSERT_EQ(close(client), 0);
  auto* notification = loom_serve_http_server_notification(server_);
  iree_notification_await(
      notification,
      [](void* value) {
        return loom_serve_http_connection_failed(
            static_cast<loom_serve_http_connection_t*>(value));
      },
      connection, iree_infinite_timeout());
  loom_serve_http_connection_abort(connection);
  client = Connect();
  ASSERT_TRUE(Write(client, "GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n"));
  connection = Take(&request);
  ASSERT_NE(connection, nullptr);
  IREE_ASSERT_OK(loom_serve_http_connection_send(
      connection, IREE_SV("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nOK")));
  loom_serve_http_connection_finish(connection);
  EXPECT_EQ(Read(client), "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nOK");
  ASSERT_EQ(close(client), 0);
}

TEST_F(HttpServerTest, ShutdownDrainsOpenConnectionAndAccept) {
  int client = Connect();
  ASSERT_TRUE(Write(client, "GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n"));
  const loom_serve_http_request_t* request = nullptr;
  ASSERT_NE(Take(&request), nullptr);
  // Abandon application views before destroying the server with I/O pending.
  request = nullptr;
  IREE_ASSERT_OK(loom_serve_http_server_destroy(server_));
  server_ = nullptr;
  EXPECT_EQ(Read(client), "");
  ASSERT_EQ(close(client), 0);
}

TEST_F(HttpServerTest, ShutdownJoinsMessageAfterAcceptCompletesNaturally) {
  ASSERT_EQ(kill(getpid(), SIGTERM), 0);
  iree_notification_await(
      loom_serve_http_server_notification(server_),
      [](void* value) {
        return loom_serve_http_server_is_stopping(
            static_cast<loom_serve_http_server_t*>(value));
      },
      server_, iree_infinite_timeout());
  // Complete the pending accept after the stop signal. It closes this socket
  // without rearming, so shutdown must still join its owner-thread message.
  int client = Connect();
  EXPECT_EQ(Read(client), "");
  ASSERT_EQ(close(client), 0);
  IREE_ASSERT_OK(loom_serve_http_server_destroy(server_));
  server_ = nullptr;
}

}  // namespace
