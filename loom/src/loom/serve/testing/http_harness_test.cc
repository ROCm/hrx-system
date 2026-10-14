// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/testing/http_harness.h"

#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <string>

#include "iree/async/util/proactor_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom_serve::testing {
namespace {

class HttpHarnessTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &pool_));
    const auto options = loom_serve_http_server_options_default();
    IREE_ASSERT_OK(loom_serve_http_server_allocate(
        pool_, &options, iree_allocator_system(), &server_));
    auto listener_options = loom_serve_http_listener_options_default();
    listener_options.address = IREE_SV("127.0.0.1:0");
    IREE_ASSERT_OK(loom_serve_http_listener_allocate(
        server_, &listener_options, {this, ListenerAllocatorControl},
        &listener_));
    IREE_ASSERT_OK(client_.Initialize(server_, listener_));
  }

  void TearDown() override {
    if (server_) {
      loom_serve_http_server_request_stop(server_);
    }
    IREE_EXPECT_OK(client_.Join());
    IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
    loom_serve_http_server_free(server_);
    iree_async_proactor_pool_release(pool_);
  }

  static std::string Message(iree_status_t status) {
    char* text = nullptr;
    iree_host_size_t length = 0;
    const auto host_allocator = iree_allocator_system();
    EXPECT_TRUE(iree_status_to_string(status, &host_allocator, &text, &length));
    const std::string result(text ? text : "", length);
    iree_allocator_free(iree_allocator_system(), text);
    iree_status_free(status);
    return result;
  }

  static iree_status_t ListenerAllocatorControl(
      void* self, iree_allocator_command_t command, const void* parameters,
      void** inout_pointer) {
    const auto* fixture = static_cast<HttpHarnessTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        fixture->fail_listener_allocations_.load()) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "HTTP receive allocation witness");
    }
    const auto host_allocator = iree_allocator_system();
    return host_allocator.ctl(host_allocator.self, command, parameters,
                              inout_pointer);
  }

  void CheckServiceFailure(bool await_first_byte) {
    int peer = -1;
    IREE_ASSERT_OK(
        client_.Send("GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n", &peer));
    loom_serve_http_connection_t* connection = nullptr;
    const loom_serve_http_request_t* request = nullptr;
    IREE_ASSERT_OK(client_.TakeRequest(peer, &connection, &request));
    client_.Start([] {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "service initialization witness");
    });
    std::string response;
    const auto message =
        Message(await_first_byte
                    ? client_.AwaitResponse(peer)
                    : client_.Read(peer, /*allow_empty=*/false, &response));
    if (await_first_byte) {
      EXPECT_EQ(close(peer), 0);
    }
    EXPECT_NE(message.find("service initialization witness"), std::string::npos)
        << message;
    EXPECT_NE(message.find("service owner exited"), std::string::npos)
        << message;
    loom_serve_http_connection_abort(connection);
    EXPECT_NE(Message(client_.Join()).find("service initialization witness"),
              std::string::npos);
  }

  // Rejects new listener allocations after startup; teardown may still free.
  std::atomic<bool> fail_listener_allocations_{false};
  // Production progress owner shared by the listener and all accepted peers.
  iree_async_proactor_pool_t* pool_ = nullptr;
  // Production HTTP connection owner under direct test scheduling.
  loom_serve_http_server_t* server_ = nullptr;
  // Ephemeral loopback listener, retired before the server.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Client and optional outer service owner under test.
  HttpHarness client_;
};

TEST_F(HttpHarnessTest, DirectAdmissionAndResponseUseActualReadiness) {
  int peer = -1;
  IREE_ASSERT_OK(
      client_.Send("GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n", &peer));
  loom_serve_http_connection_t* connection = nullptr;
  const loom_serve_http_request_t* request = nullptr;
  IREE_ASSERT_OK(client_.TakeRequest(peer, &connection, &request));
  ASSERT_NE(connection, nullptr);
  IREE_ASSERT_OK(loom_serve_http_connection_send(
      connection,
      IREE_SV("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nready")));
  loom_serve_http_connection_finish(connection);
  IREE_ASSERT_OK(client_.AwaitResponse(peer));
  std::string response;
  IREE_ASSERT_OK(client_.Read(peer, /*allow_empty=*/false, &response));
  EXPECT_EQ(response, "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nready");
}

TEST_F(HttpHarnessTest, ReceiveAllocationFailureTerminatesAdmission) {
  fail_listener_allocations_.store(true);
  int peer = -1;
  iree_status_t status =
      client_.Send("GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n", &peer);
  loom_serve_http_connection_t* connection = nullptr;
  const loom_serve_http_request_t* request = nullptr;
  if (iree_status_is_ok(status)) {
    status = client_.TakeRequest(peer, &connection, &request);
  }
  const auto message = Message(status);
  EXPECT_EQ(connection, nullptr);
  EXPECT_NE(message.find("HTTP receive allocation witness"), std::string::npos)
      << message;
  EXPECT_TRUE(client_.failed());
  if (peer >= 0) {
    EXPECT_EQ(close(peer), 0);
  }
}

TEST_F(HttpHarnessTest, ReceiveAllocationFailureTerminatesResponseRead) {
  fail_listener_allocations_.store(true);
  int peer = -1;
  iree_status_t status =
      client_.Send("GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n", &peer);
  std::string response;
  if (iree_status_is_ok(status)) {
    status = client_.Read(peer, /*allow_empty=*/false, &response);
  }
  const auto message = Message(status);
  EXPECT_NE(message.find("HTTP receive allocation witness"), std::string::npos)
      << message;
  EXPECT_TRUE(response.empty());
  EXPECT_TRUE(client_.failed());
}

TEST_F(HttpHarnessTest, UnexpectedEmptyResponseIsTerminal) {
  int peer = -1;
  IREE_ASSERT_OK(
      client_.Send("GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n", &peer));
  loom_serve_http_connection_t* connection = nullptr;
  const loom_serve_http_request_t* request = nullptr;
  IREE_ASSERT_OK(client_.TakeRequest(peer, &connection, &request));
  loom_serve_http_connection_abort(connection);
  std::string response;
  const auto message =
      Message(client_.Read(peer, /*allow_empty=*/false, &response));
  EXPECT_NE(message.find("closed before sending a response"), std::string::npos)
      << message;
  EXPECT_TRUE(client_.failed());
}

TEST_F(HttpHarnessTest, ExpectedCancellationCanReturnEmpty) {
  int peer = -1;
  IREE_ASSERT_OK(
      client_.Send("GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n", &peer));
  loom_serve_http_connection_t* connection = nullptr;
  const loom_serve_http_request_t* request = nullptr;
  IREE_ASSERT_OK(client_.TakeRequest(peer, &connection, &request));
  loom_serve_http_connection_abort(connection);
  std::string response;
  IREE_ASSERT_OK(client_.Read(peer, /*allow_empty=*/true, &response));
  EXPECT_TRUE(response.empty());
}

TEST_F(HttpHarnessTest, ServiceFailureWakesReadWithoutClosingTheSocket) {
  CheckServiceFailure(/*await_first_byte=*/false);
}

TEST_F(HttpHarnessTest,
       ServiceFailureWakesFirstByteWaitWithoutClosingTheSocket) {
  CheckServiceFailure(/*await_first_byte=*/true);
}

TEST_F(HttpHarnessTest, ReceiveAllocationFailureTerminatesModelWait) {
  fail_listener_allocations_.store(true);
  int descriptor = -1;
  iree_status_t status =
      client_.Send("GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n", &descriptor);
  HttpClient peer(descriptor);
  if (iree_status_is_ok(status)) {
    status = client_.Await(peer.get(), [] { return false; });
  }
  const auto message = Message(status);
  EXPECT_NE(message.find("HTTP receive allocation witness"), std::string::npos)
      << message;
  EXPECT_TRUE(client_.failed());
}

TEST_F(HttpHarnessTest, ServiceFailureTerminatesModelWait) {
  client_.Start([] {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "model owner witness");
  });
  const auto message = Message(client_.Await(-1, [] { return false; }));
  EXPECT_NE(message.find("model owner witness"), std::string::npos) << message;
  EXPECT_NE(message.find("service owner exited"), std::string::npos) << message;
  EXPECT_TRUE(client_.failed());
  EXPECT_NE(Message(client_.Join()).find("model owner witness"),
            std::string::npos);
}

TEST_F(HttpHarnessTest, PublishedModelStateWinsOverSubsequentOwnerExit) {
  std::atomic<bool> ready{false};
  client_.Start([&] {
    ready.store(true, std::memory_order_release);
    client_.Notify();
    return iree_ok_status();
  });
  IREE_ASSERT_OK(
      client_.Await(-1, [&] { return ready.load(std::memory_order_acquire); }));
  IREE_ASSERT_OK(client_.Join());
  EXPECT_FALSE(client_.failed());
}

TEST_F(HttpHarnessTest, ModelWaitDoesNotConsumeUnexpectedResponse) {
  int descriptor = -1;
  IREE_ASSERT_OK(client_.Send("GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n",
                              &descriptor));
  HttpClient peer(descriptor);
  loom_serve_http_connection_t* connection = nullptr;
  const loom_serve_http_request_t* request = nullptr;
  IREE_ASSERT_OK(client_.TakeRequest(peer.get(), &connection, &request));
  IREE_ASSERT_OK(loom_serve_http_connection_send(
      connection, IREE_SV("HTTP/1.1 503 Unavailable\r\n\r\nrejected")));
  loom_serve_http_connection_finish(connection);
  EXPECT_NE(Message(client_.Await(peer.get(), [] { return false; }))
                .find("responded before expected model state"),
            std::string::npos);
  std::string response;
  IREE_ASSERT_OK(client_.Read(peer.release(), false, &response));
  EXPECT_EQ(response, "HTTP/1.1 503 Unavailable\r\n\r\nrejected");
}

}  // namespace
}  // namespace loom_serve::testing
