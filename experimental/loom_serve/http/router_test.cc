// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/http/router.h"

#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <deque>
#include <string>
#include <thread>

#include "iree/async/proactor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// The dependency is a bounded model service. Routing, HTTP claims, copied
// output, notification readiness and shutdown use the production carrier.
struct ModelService {
  // Deployment identifier, borrowed for the router lifetime.
  const char* name;
  // Accepted connection/request pairs owned only by the router thread.
  std::deque<std::pair<loom_serve_http_connection_t*,
                       const loom_serve_http_request_t*>>
      pending;
  // This model waits for another service's request/maintenance to retire.
  bool wait_for_shared_activity = false;

  static bool IsActive(void* self) {
    const auto* service = static_cast<ModelService*>(self);
    return !service->wait_for_shared_activity && !service->pending.empty();
  }

  static iree_status_t Accept(void* self,
                              loom_serve_http_connection_t* connection,
                              const loom_serve_http_request_t* request,
                              loom_serve_http_service_flags_t flags) {
    (void)flags;
    static_cast<ModelService*>(self)->pending.emplace_back(connection, request);
    return iree_ok_status();
  }
  static iree_status_t Advance(void* self,
                               loom_serve_http_service_flags_t flags,
                               bool* progress) {
    auto* service = static_cast<ModelService*>(self);
    if (service->wait_for_shared_activity &&
        iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_SHARED_ACTIVITY)) {
      *progress = false;
      return iree_ok_status();
    }
    *progress = !service->pending.empty();
    if (!*progress) {
      return iree_ok_status();
    }
    auto [connection, request] = service->pending.front();
    service->pending.pop_front();
    // The borrowed body is still usable after later claims were routed.
    const std::string response =
        "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n" +
        std::string(service->name) + ":" +
        std::string(request->target.data, request->target.size) + ":" +
        std::string(request->body.data ? request->body.data : "",
                    request->body.size);
    iree_status_t status = loom_serve_http_connection_send(
        connection, iree_make_string_view(response.data(), response.size()));
    if (iree_status_is_ok(status)) {
      loom_serve_http_connection_finish(connection);
    } else {
      loom_serve_http_connection_abort(connection);
    }
    return status;
  }
};

class HttpRouterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_signal_block_default());
    auto options = loom_serve_http_server_options_default();
    options.port = 0;
    IREE_ASSERT_OK(loom_serve_http_server_create(&options, &server_,
                                                 iree_allocator_system()));
    owner_ = std::thread([this] {
      const loom_serve_http_service_t services[] = {
          {IREE_SV("a"), &a_, ModelService::IsActive, ModelService::Accept,
           ModelService::Advance},
          {IREE_SV("b"), &b_, ModelService::IsActive, ModelService::Accept,
           ModelService::Advance}};
      status_ =
          loom_serve_http_router_run(server_, IREE_ARRAYSIZE(services),
                                     services, 4, iree_allocator_system());
    });
  }
  void TearDown() override {
    EXPECT_EQ(kill(getpid(), SIGTERM), 0);
    owner_.join();
    IREE_EXPECT_OK(status_);
    for (auto* service : {&a_, &b_}) {
      for (auto& [connection, request] : service->pending) {
        (void)request;
        loom_serve_http_connection_abort(connection);
      }
    }
    IREE_EXPECT_OK(loom_serve_http_server_destroy(server_));
  }
  int Send(const std::string& method, const std::string& target,
           const std::string& body = "", const std::string& headers = "") {
    const int client = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    EXPECT_GE(client, 0);
    const auto* address = loom_serve_http_server_address(server_);
    EXPECT_EQ(
        connect(client, reinterpret_cast<const sockaddr*>(address->storage),
                address->length),
        0);
    const std::string request =
        method + " " + target +
        " HTTP/1.1\r\nHost: local\r\nContent-Type: application/json\r\n" +
        headers + "Content-Length: " + std::to_string(body.size()) +
        "\r\n\r\n" + body;
    size_t offset = 0;
    while (offset < request.size()) {
      const ssize_t count = send(client, request.data() + offset,
                                 request.size() - offset, MSG_NOSIGNAL);
      EXPECT_GT(count, 0);
      if (count <= 0) {
        break;
      }
      offset += count;
    }
    return client;
  }
  std::string Read(int client) {
    std::string result;
    char buffer[4096];
    ssize_t count = 0;
    while ((count = recv(client, buffer, sizeof(buffer), 0)) > 0) {
      result.append(buffer, count);
    }
    EXPECT_EQ(count, 0);
    EXPECT_EQ(close(client), 0);
    return result;
  }
  // Real loopback transport and its separate proactor owner.
  loom_serve_http_server_t* server_ = nullptr;
  // First model dependency used only by the joined application owner.
  ModelService a_{"a"};
  // Second model dependency used only by the joined application owner.
  ModelService b_{"b", {}, true};
  // Application owner running the production router.
  std::thread owner_;
  // Terminal result published by joining owner_.
  iree_status_t status_ = iree_ok_status();
};

TEST_F(HttpRouterTest, DiscoveryAndIndependentBorrowedRequests) {
  EXPECT_NE(Read(Send("GET", "/healthz")).find(R"({"status":"ready"})"),
            std::string::npos);
  const auto models = Read(Send("GET", "/v1/models"));
  EXPECT_NE(models.find(R"({"object":"model","id":"a"})"), std::string::npos);
  EXPECT_NE(models.find(R"({"object":"model","id":"b"})"), std::string::npos);
  const std::string body_a = R"({"model":"\u0061","messages":["first"]})";
  const std::string body_b = R"({"model":"b","messages":["second"]})";
  const int a = Send("POST", "/v1/chat/completions", body_a);
  const int b = Send("POST", "/v1/chat/completions", body_b);
  EXPECT_NE(Read(b).find("b:/v1/chat/completions:" + body_b),
            std::string::npos);
  EXPECT_NE(Read(a).find("a:/v1/chat/completions:" + body_a),
            std::string::npos);
  EXPECT_NE(
      Read(Send("POST", "/v1/checkpoints/same", "", "X-Loom-Model: b\r\n"))
          .find("b:/v1/checkpoints/same:"),
      std::string::npos);
  EXPECT_NE(
      Read(Send("DELETE", "/v1/checkpoints/same", "", "X-Loom-Model: a\r\n"))
          .find("a:/v1/checkpoints/same:"),
      std::string::npos);
}

TEST_F(HttpRouterTest, RejectsAmbiguousSelectorsWithoutInvokingModels) {
  for (const auto* body : {"{}", R"({"model":null})", R"({"model":4})",
                           R"({"model":""})", R"({"model":"a","model":"b"})",
                           R"({"model":"a"} trailing)", R"({"model":"\q"})"}) {
    SCOPED_TRACE(body);
    EXPECT_EQ(
        Read(Send("POST", "/v1/chat/completions", body)).find("HTTP/1.1 400"),
        0u);
  }
  EXPECT_EQ(Read(Send("POST", "/v1/chat/completions", R"({"model":"missing"})"))
                .find("HTTP/1.1 404"),
            0u);
  EXPECT_EQ(Read(Send("POST", "/v1/checkpoints/same")).find("HTTP/1.1 400"),
            0u);
  EXPECT_EQ(Read(Send("DELETE", "/v1/checkpoints/same", "",
                      "X-Loom-Model: missing\r\n"))
                .find("HTTP/1.1 404"),
            0u);
  // Rejections leave the listener and both services available.
  EXPECT_NE(Read(Send("POST", "/v1/chat/completions", R"({"model":"b"})"))
                .find("b:/v1/chat/completions:"),
            std::string::npos);
}

}  // namespace
