// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/http_server.h"

#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <string>

#include "iree/async/proactor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class HttpServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_signal_block_default());
    IREE_ASSERT_OK(
        loom_serve_http_server_create(0, iree_allocator_system(), &server_));
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

  static void Write(int client, const std::string& bytes) {
    size_t offset = 0;
    while (offset < bytes.size()) {
      ssize_t count = send(client, bytes.data() + offset, bytes.size() - offset,
                           MSG_NOSIGNAL);
      ASSERT_GT(count, 0);
      offset += count;
    }
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
  Write(client,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: local\r\nContent-");
  Write(client, "Length: 5\r\n\r\nhe");
  Write(client, "llo");
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
    Write(client, "GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n");
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

TEST_F(HttpServerTest, PeerResetReturnsOwnershipAndListenerSurvives) {
  int client = Connect();
  Write(client, "GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n");
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
  Write(client, "GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n");
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
  Write(client, "GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n");
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
