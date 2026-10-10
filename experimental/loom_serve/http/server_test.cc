// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/http/server.h"

#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

#include "experimental/loom_serve/http/listener.h"
#include "experimental/loom_serve/telemetry/stream.h"
#include "iree/async/proactor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

class HttpServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &pool_));
    auto options = loom_serve_http_server_options_default();
    options.connection_capacity = 24;
    options.request_limits.body_byte_capacity = 2 * 1024 * 1024;
    IREE_ASSERT_OK(loom_serve_http_server_create(pool_, &options, &server_,
                                                 iree_allocator_system()));
    listener_ = Listen(IREE_SV("127.0.0.1:0"));
  }

  void TearDown() override {
    IREE_EXPECT_OK(loom_serve_http_listener_destroy(listener_));
    for (auto* listener : additional_listeners_) {
      IREE_EXPECT_OK(loom_serve_http_listener_destroy(listener));
    }
    loom_serve_http_server_destroy(server_);
    iree_async_proactor_pool_release(pool_);
  }

  loom_serve_http_listener_t* Listen(iree_string_view_t address) {
    auto options = loom_serve_http_listener_options_default();
    options.address = address;
    loom_serve_http_listener_t* listener = nullptr;
    IREE_EXPECT_OK(loom_serve_http_listener_create(server_, &options, &listener,
                                                   iree_allocator_system()));
    return listener;
  }

  int Connect(loom_serve_http_listener_t* listener = nullptr) {
    const auto* address =
        loom_serve_http_listener_address(listener ? listener : listener_);
    const auto* native = reinterpret_cast<const sockaddr*>(address->storage);
    int client = socket(native->sa_family, SOCK_STREAM | SOCK_CLOEXEC, 0);
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

  // Shared progress infrastructure used by every binding and connection.
  iree_async_proactor_pool_t* pool_ = nullptr;
  // Initial independently owned TCP binding.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Other bindings retired before their borrowed server.
  std::vector<loom_serve_http_listener_t*> additional_listeners_;
  // Server under test, retaining shared progress without private workers.
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

TEST_F(HttpServerTest, CapacityRejectionPublishesDiagnostic) {
  std::vector<int> clients;
  std::vector<loom_serve_http_connection_t*> connections;
  for (int i = 0; i < 24; ++i) {
    const int client = Connect();
    ASSERT_TRUE(Write(client, "GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n"));
    const loom_serve_http_request_t* request = nullptr;
    auto* connection = Take(&request);
    ASSERT_NE(connection, nullptr);
    clients.push_back(client);
    connections.push_back(connection);
  }
  const int rejected = Connect();
  EXPECT_EQ(Read(rejected), "");
  ASSERT_EQ(close(rejected), 0);

  const auto* stream = loom_serve_http_server_diagnostics(server_);
  uint64_t cursor = 0;
  loom_serve_telemetry_record_t record;
  ASSERT_EQ(loom_serve_telemetry_stream_read(stream, &cursor, &record),
            LOOM_SERVE_TELEMETRY_READ_RECORD);
  EXPECT_EQ(record.event, &LOOM_SERVE_TELEMETRY_STREAM_BEGIN);
  ASSERT_EQ(loom_serve_telemetry_stream_read(stream, &cursor, &record),
            LOOM_SERVE_TELEMETRY_READ_RECORD);
  EXPECT_EQ(record.event, &LOOM_SERVE_TELEMETRY_ERROR);
  EXPECT_EQ(record.data.error.code, IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(std::string(record.data.error.message, record.data.error.length),
            "HTTP connection capacity exhausted");
  EXPECT_EQ(record.data.error.truncated, 0u);
  for (size_t i = 0; i < clients.size(); ++i) {
    loom_serve_http_connection_abort(connections[i]);
    EXPECT_EQ(Read(clients[i]), "");
    ASSERT_EQ(close(clients[i]), 0);
  }
}

TEST_F(HttpServerTest, DiagnosticsOwnMessageAfterStatusRetires) {
  iree_status_t status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                          "malformed model selector");
  loom_serve_http_server_record_error(server_, status);
  iree_status_free(status);
  const auto* stream = loom_serve_http_server_diagnostics(server_);
  uint64_t cursor = 1;
  loom_serve_telemetry_record_t record;
  ASSERT_EQ(loom_serve_telemetry_stream_read(stream, &cursor, &record),
            LOOM_SERVE_TELEMETRY_READ_RECORD);
  EXPECT_EQ(record.event, &LOOM_SERVE_TELEMETRY_ERROR);
  EXPECT_EQ(record.data.error.code, IREE_STATUS_INVALID_ARGUMENT);
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS)
  EXPECT_EQ(std::string(record.data.error.message, record.data.error.length),
            "malformed model selector");
#else
  EXPECT_EQ(record.data.error.length, 0u);
#endif
}

TEST_F(HttpServerTest, ConfiguredBodiesAndClaimsOutliveOtherPeers) {
  int clients[20];
  loom_serve_http_connection_t* connections[20];
  const loom_serve_http_request_t* requests[20];
  // Multiple claims wait for application admission independently. A long
  // request is allowed by this server's byte budget.
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
  IREE_ASSERT_OK(loom_serve_http_listener_destroy(listener_));
  listener_ = nullptr;
  loom_serve_http_server_destroy(server_);
  server_ = nullptr;
  EXPECT_EQ(Read(client), "");
  ASSERT_EQ(close(client), 0);
}

TEST_F(HttpServerTest, ServerAdmissionClosesIndependentlyOfListener) {
  loom_serve_http_server_request_stop(server_);
  EXPECT_TRUE(loom_serve_http_server_is_stopping(server_));
  int client = Connect();
  EXPECT_EQ(Read(client), "");
  ASSERT_EQ(close(client), 0);
  IREE_ASSERT_OK(loom_serve_http_listener_destroy(listener_));
  listener_ = nullptr;
  loom_serve_http_server_destroy(server_);
  server_ = nullptr;
}

TEST_F(HttpServerTest, StreamOutlivesListenerWhileOtherBindingsAdmit) {
  iree::testing::TempFilePath path("http");
  const std::string unix_address = "unix:" + path.path();
  auto* unix_listener = Listen(iree_make_cstring_view(unix_address.c_str()));
  ASSERT_NE(unix_listener, nullptr);
  additional_listeners_.push_back(unix_listener);
  auto* ipv6_listener = Listen(IREE_SV("[::1]:0"));
  ASSERT_NE(ipv6_listener, nullptr);
  additional_listeners_.push_back(ipv6_listener);

  const int stream_client = Connect();
  ASSERT_TRUE(
      Write(stream_client, "GET /stream HTTP/1.1\r\nHost: local\r\n\r\n"));
  const loom_serve_http_request_t* request = nullptr;
  auto* stream = Take(&request);
  ASSERT_NE(stream, nullptr);
  const std::string prefix =
      "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
      "Connection: close\r\n\r\ndata: first\n\n";
  IREE_ASSERT_OK(loom_serve_http_connection_send(
      stream, iree_make_cstring_view(prefix.c_str())));
  loom_serve_http_listener_request_stop(listener_);
  iree_notification_await(
      loom_serve_http_listener_notification(listener_),
      [](void* value) {
        return loom_serve_http_listener_is_stopped(
            static_cast<loom_serve_http_listener_t*>(value));
      },
      listener_, iree_infinite_timeout());
  IREE_ASSERT_OK(loom_serve_http_listener_destroy(listener_));
  listener_ = nullptr;
  EXPECT_FALSE(loom_serve_http_server_is_stopping(server_));

  for (auto* listener : additional_listeners_) {
    const int client = Connect(listener);
    ASSERT_TRUE(Write(client, "GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n"));
    auto* connection = Take(&request);
    ASSERT_NE(connection, nullptr);
    IREE_ASSERT_OK(loom_serve_http_connection_send(
        connection, IREE_SV("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nOK")));
    loom_serve_http_connection_finish(connection);
    EXPECT_EQ(Read(client), "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nOK");
    EXPECT_EQ(close(client), 0);
  }
  // Drop the caller's pool reference too: the HTTP owner retains progress.
  iree_async_proactor_pool_release(pool_);
  pool_ = nullptr;
  IREE_ASSERT_OK(
      loom_serve_http_connection_send(stream, IREE_SV("data: last\n\n")));
  loom_serve_http_connection_finish(stream);
  EXPECT_EQ(Read(stream_client), prefix + "data: last\n\n");
  EXPECT_EQ(close(stream_client), 0);
}

TEST_F(HttpServerTest, FailedUnixBindDoesNotRemoveAnotherListener) {
  iree::testing::TempFilePath path("http");
  const std::string address = "unix:" + path.path();
  auto options = loom_serve_http_listener_options_default();
  options.address = iree_make_cstring_view(address.c_str());
  auto* first = Listen(options.address);
  ASSERT_NE(first, nullptr);
  additional_listeners_.push_back(first);
  loom_serve_http_listener_t* duplicate = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_serve_http_listener_create(server_, &options, &duplicate,
                                      iree_allocator_system()));
  EXPECT_EQ(duplicate, nullptr);
  EXPECT_TRUE(path.Exists());
  IREE_ASSERT_OK(loom_serve_http_listener_destroy(first));
  additional_listeners_.clear();
  EXPECT_FALSE(path.Exists());
  auto* rebound = Listen(options.address);
  ASSERT_NE(rebound, nullptr);
  additional_listeners_.push_back(rebound);
}

}  // namespace
