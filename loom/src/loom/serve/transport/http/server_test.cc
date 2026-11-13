// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/transport/http/server.h"

#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "iree/async/proactor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/runtime/telemetry/stream.h"
#include "loom/serve/transport/http/listener.h"
#include "loom/service/telemetry/record.h"

namespace {

class HttpServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        2, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &pool_));
    auto options = loom_serve_http_server_options_default();
    options.connection_capacity = 24;
    options.request_limits.body_byte_capacity = 2 * 1024 * 1024;
    IREE_ASSERT_OK(loom_serve_http_server_allocate(
        pool_, &options, iree_allocator_system(), &server_));
    IREE_ASSERT_OK(Listen(IREE_SV("127.0.0.1:0"), &listener_));
    // Unix socket names must fit sockaddr_un independently of the runner's
    // potentially long artifact directory. Listeners retire before this path.
    char directory[] = "/tmp/loom-http-XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    socket_directory_ = directory;
  }

  void TearDown() override {
    IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
    for (auto* listener : additional_listeners_) {
      IREE_EXPECT_OK(loom_serve_http_listener_free(listener));
    }
    while (!held_connections_.empty()) {
      ReleaseHeld();
    }
    loom_serve_http_server_free(server_);
    iree_async_proactor_pool_release(pool_);
    if (!socket_directory_.empty()) {
      EXPECT_EQ(rmdir(socket_directory_.c_str()), 0);
    }
  }

  iree_status_t Listen(iree_string_view_t address,
                       iree_host_size_t proactor_index,
                       loom_serve_http_listener_t** out_listener) {
    auto options = loom_serve_http_listener_options_default();
    options.address = address;
    options.proactor_index = proactor_index;
    return loom_serve_http_listener_allocate(
        server_, &options, iree_allocator_system(), out_listener);
  }

  iree_status_t Listen(iree_string_view_t address,
                       loom_serve_http_listener_t** out_listener) {
    return Listen(address, /*proactor_index=*/0, out_listener);
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

  void HoldCapacity() {
    for (size_t i = 0; i < loom_serve_http_server_connection_capacity(server_);
         ++i) {
      const int client = Connect();
      ASSERT_TRUE(Write(client, "GET /held HTTP/1.1\r\nHost: local\r\n\r\n"));
      const loom_serve_http_request_t* request = nullptr;
      auto* connection = Take(&request);
      ASSERT_NE(connection, nullptr);
      held_connections_.emplace_back(client, connection);
    }
  }

  void ReleaseHeld() {
    auto [client, connection] = held_connections_.back();
    held_connections_.pop_back();
    loom_serve_http_connection_abort(connection);
    EXPECT_EQ(Read(client), "");
    EXPECT_EQ(close(client), 0);
  }

  static void WaitCapacity(loom_serve_http_listener_t* listener) {
    iree_notification_await(
        loom_serve_http_listener_notification(listener),
        [](void* value) {
          return loom_serve_http_listener_is_waiting_for_capacity(
              static_cast<loom_serve_http_listener_t*>(value));
        },
        listener, iree_infinite_timeout());
  }

  static void ReadUnadmittedClose(int client) {
    char byte = 0;
    const ssize_t count = recv(client, &byte, 1, 0);
    // Closing an unadmitted socket may reset its unread request bytes.
    EXPECT_TRUE(count == 0 || (count == -1 && errno == ECONNRESET));
    EXPECT_EQ(close(client), 0);
  }

  // Shared progress infrastructure used by every binding and connection.
  iree_async_proactor_pool_t* pool_ = nullptr;
  // Initial independently owned TCP binding.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Other bindings retired before their borrowed server.
  std::vector<loom_serve_http_listener_t*> additional_listeners_;
  // Server under test, retaining shared progress without private workers.
  loom_serve_http_server_t* server_ = nullptr;
  // Client socket and real application claim held through capacity pressure.
  std::vector<std::pair<int, loom_serve_http_connection_t*>> held_connections_;
  // Private short socket directory removed after all listeners retire.
  std::string socket_directory_;
};

TEST_F(HttpServerTest, BrowserOriginTrustFollowsAcceptedTransport) {
  const std::string path = "unix:" + socket_directory_ + "/relay.sock";
  loom_serve_http_listener_t* relay = nullptr;
  IREE_ASSERT_OK(
      Listen(iree_make_string_view(path.data(), path.size()), &relay));
  additional_listeners_.push_back(relay);
  // Both requests carry the same headers. Only the Unix binding grants relay
  // authority; a subsequent TCP peer must not inherit it from a recycled slot.
  for (auto* listener : {listener_, relay, listener_}) {
    SCOPED_TRACE(listener == relay ? "Unix relay" : "TCP");
    const int client = Connect(listener);
    ASSERT_TRUE(Write(client,
                      "GET /v1/observe HTTP/1.1\r\nHost: serve.example\r\n"
                      "Origin: https://serve.example\r\n"
                      "X-Forwarded-Proto: https\r\n"
                      "Sec-Fetch-Site: same-origin\r\n\r\n"));
    const loom_serve_http_request_t* request = nullptr;
    auto* connection = Take(&request);
    ASSERT_NE(connection, nullptr);
    const auto policy = loom_serve_http_connection_origin_policy(connection);
    EXPECT_EQ(policy.source, listener == relay ? LOOM_SERVE_HTTP_ORIGIN_RELAY
                                               : LOOM_SERVE_HTTP_ORIGIN_DIRECT);
    IREE_EXPECT_STATUS_IS(
        listener == relay ? IREE_STATUS_OK : IREE_STATUS_PERMISSION_DENIED,
        loom_serve_http_origin_check(request, policy));
    loom_serve_http_connection_finish(connection);
    EXPECT_EQ(Read(client), "");
    EXPECT_EQ(close(client), 0);
  }
}

TEST_F(HttpServerTest, InvalidBrowserOriginConfigurationFailsBeforeAdmission) {
  auto options = loom_serve_http_server_options_default();
  const iree_string_view_t origin = IREE_SV("https://serve.example/path");
  options.allowed_origins = {1, &origin};
  loom_serve_http_server_t* server = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_http_server_allocate(
                            pool_, &options, iree_allocator_system(), &server));
  EXPECT_EQ(server, nullptr);
}

TEST_F(HttpServerTest, BrowserAuthorityMayUseATunnelPort) {
  const int client = Connect();
  ASSERT_TRUE(Write(client,
                    "GET /v1/observe HTTP/1.1\r\nHost: localhost:13000\r\n"
                    "Origin: http://localhost:13000\r\n\r\n"));
  const loom_serve_http_request_t* request = nullptr;
  auto* connection = Take(&request);
  ASSERT_NE(connection, nullptr);
  IREE_EXPECT_OK(loom_serve_http_origin_check(
      request, loom_serve_http_connection_origin_policy(connection)));
  loom_serve_http_connection_finish(connection);
  EXPECT_EQ(Read(client), "");
  EXPECT_EQ(close(client), 0);
}

TEST_F(HttpServerTest, FragmentedRequestAndOrderedStreamingResponse) {
  const auto limits = loom_serve_http_server_request_limits(server_);
  EXPECT_EQ(limits.header_byte_capacity, 16384u);
  EXPECT_EQ(limits.body_byte_capacity, 2 * 1024 * 1024u);
  EXPECT_EQ(limits.header_capacity, 64u);
  EXPECT_EQ(limits.multipart_header_byte_capacity, 64 * 1024u);
  EXPECT_EQ(limits.multipart_part_capacity, 1024u);
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

TEST_F(HttpServerTest, ChunkedUploadSurvivesPeerHalfClose) {
  int client = Connect();
  ASSERT_TRUE(Write(client,
                    "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: local\r\n"
                    "Transfer-Encoding: chunked\r\n\r\n3\r"));
  ASSERT_TRUE(Write(client, "\nabc\r\n2;part=last\r\n"));
  ASSERT_TRUE(Write(client, std::string("\0d", 2)));
  ASSERT_TRUE(Write(client, "\r\n0\r\nX-End: yes\r\n\r"));
  ASSERT_TRUE(Write(client, "\n"));
  ASSERT_EQ(shutdown(client, SHUT_WR), 0);
  const loom_serve_http_request_t* request = nullptr;
  auto* connection = Take(&request);
  ASSERT_NE(connection, nullptr);
  EXPECT_EQ(std::string(request->body.data, request->body.size),
            std::string("abc\0d", 5));
  EXPECT_EQ(request->headers.count, 2u);
  IREE_ASSERT_OK(loom_serve_http_connection_send(
      connection,
      IREE_SV("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nuploaded")));
  loom_serve_http_connection_finish(connection);
  EXPECT_EQ(Read(client),
            "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nuploaded");
  ASSERT_EQ(close(client), 0);
}

TEST_F(HttpServerTest, CapacityWaitReturnsTheOriginalRequestAfterRetirement) {
  ASSERT_NO_FATAL_FAILURE(HoldCapacity());
  const int client = Connect();
  ASSERT_TRUE(Write(client, "GET /waiting HTTP/1.1\r\nHost: local\r\n\r\n"));
  WaitCapacity(listener_);
  ReleaseHeld();
  const loom_serve_http_request_t* request = nullptr;
  auto* connection = Take(&request);
  ASSERT_NE(connection, nullptr);
  EXPECT_TRUE(iree_string_view_equal(request->target, IREE_SV("/waiting")));
  IREE_ASSERT_OK(loom_serve_http_connection_send(
      connection,
      IREE_SV("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nresumed")));
  loom_serve_http_connection_finish(connection);
  EXPECT_EQ(Read(client),
            "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nresumed");
  EXPECT_EQ(close(client), 0);
}

TEST_F(HttpServerTest, StoppingACapacityWaitPreservesAcceptedClaimsAndPeers) {
  ASSERT_NO_FATAL_FAILURE(HoldCapacity());
  const int waiting = Connect();
  ASSERT_TRUE(Write(waiting, "GET /waiting HTTP/1.1\r\nHost: local\r\n\r\n"));
  WaitCapacity(listener_);
  IREE_ASSERT_OK(loom_serve_http_listener_free(listener_));
  listener_ = nullptr;
  ReadUnadmittedClose(waiting);
  // Already accepted response ownership is independent of this listener.
  auto [client, connection] = held_connections_.back();
  held_connections_.pop_back();
  IREE_ASSERT_OK(loom_serve_http_connection_send(
      connection,
      IREE_SV("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nalive")));
  loom_serve_http_connection_finish(connection);
  EXPECT_EQ(Read(client), "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nalive");
  EXPECT_EQ(close(client), 0);

  loom_serve_http_listener_t* other = nullptr;
  IREE_ASSERT_OK(Listen(IREE_SV("127.0.0.1:0"), &other));
  additional_listeners_.push_back(other);
  const int peer = Connect(other);
  ASSERT_TRUE(Write(peer, "GET /peer HTTP/1.1\r\nHost: local\r\n\r\n"));
  const loom_serve_http_request_t* request = nullptr;
  connection = Take(&request);
  ASSERT_NE(connection, nullptr);
  IREE_ASSERT_OK(loom_serve_http_connection_send(
      connection, IREE_SV("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\npeer")));
  loom_serve_http_connection_finish(connection);
  EXPECT_EQ(Read(peer), "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\npeer");
  EXPECT_EQ(close(peer), 0);
}

TEST_F(HttpServerTest, ServerStopWakesCapacityWaitWithoutRevokingClaims) {
  ASSERT_NO_FATAL_FAILURE(HoldCapacity());
  const int waiting = Connect();
  ASSERT_TRUE(Write(waiting, "GET /waiting HTTP/1.1\r\nHost: local\r\n\r\n"));
  WaitCapacity(listener_);
  loom_serve_http_server_request_stop(server_);
  ReadUnadmittedClose(waiting);
  auto [client, connection] = held_connections_.back();
  held_connections_.pop_back();
  IREE_ASSERT_OK(loom_serve_http_connection_send(
      connection,
      IREE_SV("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nalive")));
  loom_serve_http_connection_finish(connection);
  EXPECT_EQ(Read(client), "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nalive");
  EXPECT_EQ(close(client), 0);
}

TEST_F(HttpServerTest, TcpAndUnixPollOwnersShareReclaimedCapacity) {
  const std::string address = "unix:" + socket_directory_ + "/socket";
  loom_serve_http_listener_t* other = nullptr;
  IREE_ASSERT_OK(Listen(iree_make_cstring_view(address.c_str()), 1, &other));
  additional_listeners_.push_back(other);
  ASSERT_NO_FATAL_FAILURE(HoldCapacity());
  const int clients[] = {Connect(), Connect(other)};
  ASSERT_TRUE(Write(clients[0], "GET /tcp HTTP/1.1\r\nHost: local\r\n\r\n"));
  ASSERT_TRUE(Write(clients[1], "GET /unix HTTP/1.1\r\nHost: local\r\n\r\n"));
  WaitCapacity(listener_);
  WaitCapacity(other);
  ReleaseHeld();
  bool completed[2] = {};
  for (int i = 0; i < 2; ++i) {
    const loom_serve_http_request_t* request = nullptr;
    auto* connection = Take(&request);
    ASSERT_NE(connection, nullptr);
    const int index =
        iree_string_view_equal(request->target, IREE_SV("/tcp")) ? 0 : 1;
    EXPECT_FALSE(completed[index]);
    EXPECT_EQ(loom_serve_http_connection_origin_policy(connection).source,
              index == 0 ? LOOM_SERVE_HTTP_ORIGIN_DIRECT
                         : LOOM_SERVE_HTTP_ORIGIN_RELAY);
    completed[index] = true;
    IREE_ASSERT_OK(loom_serve_http_connection_send(
        connection,
        IREE_SV("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\npeer")));
    loom_serve_http_connection_finish(connection);
    EXPECT_EQ(Read(clients[index]),
              "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\npeer");
    EXPECT_EQ(close(clients[index]), 0);
  }
}

TEST_F(HttpServerTest, DiagnosticsOwnMessageAfterStatusRetires) {
  iree_status_t status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                          "malformed model selector");
  loom_serve_http_server_record_error(server_, status);
  iree_status_free(status);
  const auto* stream = loom_serve_http_server_diagnostics(server_);
  uint64_t cursor = 1;
  loom_runtime_telemetry_record_t record;
  ASSERT_EQ(loom_runtime_telemetry_stream_read(stream, &cursor, &record),
            LOOM_RUNTIME_TELEMETRY_READ_RECORD);
  EXPECT_EQ(record.event, &LOOM_SERVICE_TELEMETRY_ERROR);
  loom_service_telemetry_diagnostic_t diagnostic;
  std::memcpy(&diagnostic, record.data.words, sizeof(diagnostic));
  EXPECT_EQ(diagnostic.code, IREE_STATUS_INVALID_ARGUMENT);
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS)
  EXPECT_EQ(std::string(diagnostic.message, diagnostic.length),
            "malformed model selector");
#else
  EXPECT_EQ(diagnostic.length, 0u);
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

TEST_F(HttpServerTest, CancellationPreservesClaimedInputThroughPeerRetirement) {
  int client = Connect();
  ASSERT_TRUE(Write(client,
                    "POST /held HTTP/1.1\r\nHost: local\r\n"
                    "Content-Length: 5\r\n\r\ninput"));
  const loom_serve_http_request_t* request = nullptr;
  auto* connection = Take(&request);
  ASSERT_NE(connection, nullptr);
  loom_serve_http_connection_cancel(connection);
  loom_serve_http_connection_cancel(connection);
  EXPECT_TRUE(loom_serve_http_connection_failed(connection));
  EXPECT_EQ(Read(client), "");
  ASSERT_EQ(close(client), 0);

  // Repeated admission collects retired peers and recycles every other slot.
  // This cancelled slot still owns its parser until the application releases
  // it.
  for (size_t i = 0; i < loom_serve_http_server_connection_capacity(server_);
       ++i) {
    client = Connect();
    ASSERT_TRUE(Write(client, "GET /peer HTTP/1.1\r\nHost: local\r\n\r\n"));
    const loom_serve_http_request_t* peer_request = nullptr;
    auto* peer = Take(&peer_request);
    ASSERT_NE(peer, nullptr);
    EXPECT_NE(peer, connection);
    EXPECT_TRUE(iree_string_view_equal(request->body, IREE_SV("input")));
    loom_serve_http_connection_abort(peer);
    EXPECT_EQ(Read(client), "");
    ASSERT_EQ(close(client), 0);
  }
  EXPECT_TRUE(iree_string_view_equal(request->body, IREE_SV("input")));
  loom_serve_http_connection_abort(connection);
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

TEST_F(HttpServerTest, PendingReceiveOutlivesListenerAndOtherPoolReuse) {
  const int pending = Connect();
  const std::string prefix(8192, 'p');
  const std::string suffix(20000, 's');
  ASSERT_TRUE(Write(
      pending, "POST /pending HTTP/1.1\r\nHost: local\r\nContent-Length: " +
                   std::to_string(prefix.size() + suffix.size()) + "\r\n\r\n" +
                   prefix));
  // Accepts are serialized. Receiving this later peer proves the partial
  // request's carrier is active before its listener and cache owner retire.
  const int fence = Connect();
  ASSERT_TRUE(Write(fence, "GET /fence HTTP/1.1\r\nHost: local\r\n\r\n"));
  const loom_serve_http_request_t* request = nullptr;
  auto* connection = Take(&request);
  ASSERT_NE(connection, nullptr);
  EXPECT_TRUE(iree_string_view_equal(request->target, IREE_SV("/fence")));
  loom_serve_http_connection_abort(connection);
  EXPECT_EQ(Read(fence), "");
  EXPECT_EQ(close(fence), 0);
  IREE_ASSERT_OK(loom_serve_http_listener_free(listener_));
  listener_ = nullptr;

  loom_serve_http_listener_t* other = nullptr;
  IREE_ASSERT_OK(Listen(IREE_SV("127.0.0.1:0"), 1, &other));
  additional_listeners_.push_back(other);
  for (int i = 0; i < 32; ++i) {
    const int peer = Connect(other);
    ASSERT_TRUE(Write(peer, "GET /reused HTTP/1.1\r\nHost: local\r\n\r\n"));
    connection = Take(&request);
    ASSERT_NE(connection, nullptr);
    EXPECT_TRUE(iree_string_view_equal(request->target, IREE_SV("/reused")));
    IREE_ASSERT_OK(loom_serve_http_connection_send(
        connection, IREE_SV("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")));
    loom_serve_http_connection_finish(connection);
    EXPECT_EQ(Read(peer), "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK");
    EXPECT_EQ(close(peer), 0);
  }

  ASSERT_TRUE(Write(pending, suffix));
  connection = Take(&request);
  ASSERT_NE(connection, nullptr);
  EXPECT_TRUE(iree_string_view_equal(request->target, IREE_SV("/pending")));
  EXPECT_EQ(std::string(request->body.data, request->body.size),
            prefix + suffix);
  IREE_ASSERT_OK(loom_serve_http_connection_send(
      connection, IREE_SV("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")));
  loom_serve_http_connection_finish(connection);
  EXPECT_EQ(Read(pending), "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK");
  EXPECT_EQ(close(pending), 0);
}

TEST_F(HttpServerTest, ShutdownDrainsOpenConnectionAndAccept) {
  int client = Connect();
  ASSERT_TRUE(Write(client, "GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n"));
  const loom_serve_http_request_t* request = nullptr;
  ASSERT_NE(Take(&request), nullptr);
  // Abandon application views before destroying the server with I/O pending.
  request = nullptr;
  IREE_ASSERT_OK(loom_serve_http_listener_free(listener_));
  listener_ = nullptr;
  loom_serve_http_server_free(server_);
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
  IREE_ASSERT_OK(loom_serve_http_listener_free(listener_));
  listener_ = nullptr;
  loom_serve_http_server_free(server_);
  server_ = nullptr;
}

TEST_F(HttpServerTest, StreamOutlivesListenerWhileOtherBindingsAdmit) {
  const std::string unix_address = "unix:" + socket_directory_ + "/socket";
  loom_serve_http_listener_t* unix_listener = nullptr;
  IREE_ASSERT_OK(
      Listen(iree_make_cstring_view(unix_address.c_str()), &unix_listener));
  additional_listeners_.push_back(unix_listener);
  loom_serve_http_listener_t* ipv6_listener = nullptr;
  IREE_ASSERT_OK(Listen(IREE_SV("[::1]:0"), &ipv6_listener));
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
  IREE_ASSERT_OK(loom_serve_http_listener_free(listener_));
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
  const std::string path = socket_directory_ + "/socket";
  const std::string address = "unix:" + path;
  auto options = loom_serve_http_listener_options_default();
  options.address = iree_make_cstring_view(address.c_str());
  loom_serve_http_listener_t* first = nullptr;
  IREE_ASSERT_OK(Listen(options.address, &first));
  additional_listeners_.push_back(first);
  loom_serve_http_listener_t* duplicate = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_serve_http_listener_allocate(server_, &options,
                                        iree_allocator_system(), &duplicate));
  EXPECT_EQ(duplicate, nullptr);
  EXPECT_EQ(access(path.c_str(), F_OK), 0);
  IREE_ASSERT_OK(loom_serve_http_listener_free(first));
  additional_listeners_.clear();
  EXPECT_EQ(access(path.c_str(), F_OK), -1);
  EXPECT_EQ(errno, ENOENT);
  loom_serve_http_listener_t* rebound = nullptr;
  IREE_ASSERT_OK(Listen(options.address, &rebound));
  additional_listeners_.push_back(rebound);
}

TEST_F(HttpServerTest, OverlongUnixPathDoesNotPublishOrStopListeners) {
  const std::string address = "unix:/" + std::string(256, 's');
  loom_serve_http_listener_t* rejected = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Listen(iree_make_cstring_view(address.c_str()), &rejected));
  EXPECT_EQ(rejected, nullptr);
  EXPECT_FALSE(loom_serve_http_server_is_stopping(server_));
  const int client = Connect();
  ASSERT_TRUE(Write(client, "GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n"));
  const loom_serve_http_request_t* request = nullptr;
  auto* connection = Take(&request);
  ASSERT_NE(connection, nullptr);
  IREE_ASSERT_OK(loom_serve_http_connection_send(
      connection, IREE_SV("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nOK")));
  loom_serve_http_connection_finish(connection);
  EXPECT_EQ(Read(client), "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nOK");
  EXPECT_EQ(close(client), 0);
}

}  // namespace
