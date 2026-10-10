// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/finite/service.h"

#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <string>
#include <thread>

#include "experimental/loom_serve/decision/protocol.h"
#include "experimental/loom_serve/embedding/protocol.h"
#include "experimental/loom_serve/http/listener.h"
#include "experimental/loom_serve/serving/router.h"
#include "iree/async/proactor.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

IREE_FLAG(string, model_source, "", "Deterministic decision bootstrap source.");
IREE_FLAG(string, embedding_source, "",
          "Independent embedding bootstrap source.");

namespace {

class SourceDecisionServiceTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  void CreateModel(const char* source, loom_serve_finite_model_t** output) {
    const std::string file = source;
    const std::string directory = file.substr(0, file.find_last_of('/'));
    const loom_serve_finite_model_options_t options = {
        .source_directory = iree_make_cstring_view(directory.c_str()),
        .asset_directory = iree_make_cstring_view(directory.c_str()),
        .options = IREE_SV("{\"batch_units\":64}")};
    IREE_ASSERT_OK(
        loom_serve_finite_model_create(device_, &options, output, allocator_));
  }
  void SetUp() override {
    iree_notification_initialize(&notification_);
    IREE_ASSERT_OK(loom_serve_host_create(GetParam(), &host_, allocator_));
    const loom_serve_device_options_t options = {.uri = IREE_SV("amdgpu")};
    IREE_ASSERT_OK(loom_serve_device_group_create(host_, 1, &options, &group_,
                                                  allocator_));
    device_ = loom_serve_device_group_at(group_, 0);
    ASSERT_NO_FATAL_FAILURE(CreateModel(FLAG_model_source, &models_[0]));
    IREE_ASSERT_OK(loom_serve_finite_service_create(
        host_, models_[0], loom_serve_decision_protocol(), IREE_SV("decisions"),
        16, 0, nullptr, &services_[0], allocator_));
    interfaces_[0] = loom_serve_finite_service_interface(services_[0]);
    EXPECT_EQ(interfaces_[0].capacity_domain,
              loom_serve_device_residency_cache(device_));
    interfaces_[0].set_observer(
        interfaces_[0].self,
        {[](void* self) {
           auto* test = static_cast<SourceDecisionServiceTest*>(self);
           test->wakes_.fetch_add(1);
           iree_notification_post(&test->notification_, IREE_ALL_WAITERS);
         },
         this});
    auto http_options = loom_serve_http_server_options_default();
    IREE_ASSERT_OK(
        loom_serve_http_server_create(loom_serve_host_proactor_pool(host_),
                                      &http_options, &server_, allocator_));
    auto listener_options = loom_serve_http_listener_options_default();
    listener_options.address = IREE_SV("127.0.0.1:0");
    IREE_ASSERT_OK(loom_serve_http_listener_create(server_, &listener_options,
                                                   &listener_, allocator_));
  }
  void TearDown() override {
    if (router_.joinable()) {
      loom_serve_http_server_request_stop(server_);
      router_.join();
      IREE_EXPECT_OK(status_);
    }
    for (size_t i = 0; i < 2; ++i) {
      if (services_[i]) {
        interfaces_[i].set_observer(interfaces_[i].self, {});
        loom_serve_finite_service_destroy(services_[i]);
      }
    }
    IREE_EXPECT_OK(loom_serve_http_listener_destroy(listener_));
    loom_serve_http_server_destroy(server_);
    for (auto* model : models_) {
      IREE_EXPECT_OK(loom_serve_finite_model_destroy(model));
    }
    IREE_EXPECT_OK(loom_serve_device_group_destroy(group_));
    loom_serve_host_release(host_);
    iree_notification_deinitialize(&notification_);
  }
  int Send(const std::string& body, const char* target = "/v1/systemone",
           const char* method = "POST") {
    const int client = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    EXPECT_GE(client, 0);
    const auto* address = loom_serve_http_listener_address(listener_);
    EXPECT_EQ(
        connect(client, reinterpret_cast<const sockaddr*>(address->storage),
                address->length),
        0);
    const std::string request = std::string(method) + " " + target +
                                " HTTP/1.1\r\nHost: local\r\nContent-Type: "
                                "application/json\r\nContent-Length: " +
                                std::to_string(body.size()) + "\r\n\r\n" + body;
    size_t offset = 0;
    while (offset < request.size()) {
      const ssize_t sent = send(client, request.data() + offset,
                                request.size() - offset, MSG_NOSIGNAL);
      EXPECT_GT(sent, 0);
      if (sent <= 0) {
        break;
      }
      offset += sent;
    }
    return client;
  }
  std::string Read(int client) {
    std::string response;
    char bytes[4096];
    ssize_t length;
    while ((length = recv(client, bytes, sizeof(bytes), 0)) > 0) {
      response.append(bytes, length);
    }
    EXPECT_EQ(length, 0);
    EXPECT_EQ(close(client), 0);
    return response;
  }
  loom_serve_http_connection_t* Accept() {
    auto* notification = loom_serve_http_server_notification(server_);
    while (true) {
      const auto ticket = iree_notification_prepare_wait(notification);
      const loom_serve_http_request_t* request = nullptr;
      auto* connection = loom_serve_http_server_take_request(server_, &request);
      if (connection) {
        iree_notification_cancel_wait(notification);
        IREE_EXPECT_OK(
            interfaces_[0].accept(interfaces_[0].self, connection, request, 0));
        return connection;
      }
      iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  bool Step() {
    const uint64_t previous = wakes_.load();
    bool progress = false;
    IREE_EXPECT_OK(interfaces_[0].advance(interfaces_[0].self, 0, &progress));
    if (interfaces_[0].is_active(interfaces_[0].self)) {
      struct Wait {
        // Shared release generation.
        std::atomic<uint64_t>* wakes;
        // Last observed generation before this owner turn.
        uint64_t previous;
      } wait{&wakes_, previous};
      iree_notification_await(
          &notification_,
          [](void* data) {
            const auto* wait = static_cast<Wait*>(data);
            return wait->wakes->load() > wait->previous;
          },
          &wait, iree_infinite_timeout());
    }
    return progress;
  }
  void Drain() {
    while (Step()) {
    }
  }
  void StartRouter() {
    ASSERT_NO_FATAL_FAILURE(CreateModel(FLAG_embedding_source, &models_[1]));
    IREE_ASSERT_OK(loom_serve_finite_service_create(
        host_, models_[1], loom_serve_embedding_protocol(), IREE_SV("vectors"),
        16, 0, nullptr, &services_[1], allocator_));
    interfaces_[1] = loom_serve_finite_service_interface(services_[1]);
    interfaces_[0].set_observer(interfaces_[0].self, {});
    router_ = std::thread([this] {
      loom_serve_registry_t* registry = nullptr;
      IREE_ASSERT_OK(loom_serve_registry_create(2, interfaces_, 0, {},
                                                &registry, allocator_));
      status_ = loom_serve_router_run(
          loom_serve_host_executor(host_), server_, registry,
          {.request_quota = 16, .holding_capacity = 8}, allocator_);
      IREE_ASSERT_OK(loom_serve_registry_destroy(registry));
    });
  }
  // Allocator shared by all test-owned production components.
  const iree_allocator_t allocator_ = iree_allocator_system();
  // Real shared CPU/JIT executor and asynchronous I/O owner.
  loom_serve_host_t* host_ = nullptr;
  // Application group enclosing all model and device resources.
  loom_serve_device_group_t* group_ = nullptr;
  // Borrowed device record within the application group.
  loom_serve_device_t* device_ = nullptr;
  // Independently compiled source models, not mock callbacks.
  loom_serve_finite_model_t* models_[2] = {};
  // Bounded modality services borrowing the corresponding models.
  loom_serve_finite_service_t* services_[2] = {};
  // Exact production scheduling and routing interfaces.
  loom_serve_http_service_t interfaces_[2] = {};
  // Shared HTTP connection and request owner.
  loom_serve_http_server_t* server_ = nullptr;
  // Independent binding borrowing the HTTP owner and shared host pool.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Outer direct-driver readiness, never a worker dependency wait.
  iree_notification_t notification_;
  // Completion/release generation for direct scheduling turns.
  std::atomic<uint64_t> wakes_{0};
  // Optional outer caller of the task-owned shared router.
  std::thread router_;
  // Terminal router result consumed after join.
  iree_status_t status_ = iree_ok_status();
};

const char* const kNoul =
    R"({"model":"decisions","state":"a","questions":{"truth":{"type":"noul","instructions":"???","criteria":{"false":"","true":"x"}}}})";

TEST_P(SourceDecisionServiceTest, PacksQuestionsAndRetainsResultsAcrossReset) {
  const int a = Send(R"({"model":"decisions","state":"a","questions":{
    "route":{"type":"choice","instructions":"???","criteria":{"z":null,"x":null}},
    "level":{"type":"score","instructions":"???","criteria":["","x"]}
  }})");
  Accept();
  const int b = Send(kNoul);
  Accept();
  const int c = Send(kNoul);
  auto* cancelled = Accept();
  const int bad = Send(
      R"({"model":"decisions","state":"a","questions":{"bad":{"type":"score","instructions":"?","criteria":["one"]}}})");
  Accept();
  EXPECT_TRUE(Step());
  EXPECT_TRUE(Step());
  EXPECT_TRUE(Step());
  const linger reset = {1, 0};
  ASSERT_EQ(setsockopt(c, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)), 0);
  ASSERT_EQ(close(c), 0);
  iree_notification_await(
      loom_serve_http_server_notification(server_),
      [](void* value) {
        return loom_serve_http_connection_failed(
            static_cast<loom_serve_http_connection_t*>(value));
      },
      cancelled, iree_infinite_timeout());
  Drain();
  const std::string first = Read(a);
  EXPECT_EQ(first.find("HTTP/1.1 200"), 0u);
  EXPECT_NE(
      first.find(
          R"("route":{"type":"choice","choice":"z","confidence":0.5,"probabilities":{"z":0.75,"x":0.25}})"),
      std::string::npos);
  EXPECT_NE(
      first.find(
          R"("level":{"type":"score","score":0.75,"legend":{"0":"","1":"x"},"confidence":0.5,"probabilities":{"0":0.25,"1":0.75}})"),
      std::string::npos);
  EXPECT_NE(Read(b).find(R"("truth":{"type":"noul","noul":0.75})"),
            std::string::npos);
  EXPECT_EQ(Read(bad).find("HTTP/1.1 422"), 0u);
  const int next = Send(kNoul);
  Accept();
  Drain();
  // A single-question cohort produces a different exact oracle distribution.
  // The earlier .75 requires all four questions in one device cohort.
  EXPECT_NE(Read(next).find(R"("noul":0.666666687)"), std::string::npos);
}

TEST_P(SourceDecisionServiceTest,
       PreservesWireOrderAndRejectsOversizeSourceInput) {
  for (const char* options : {R"("z":null,"v":null)", R"("v":null,"z":null)"}) {
    const int client = Send(
        std::string(
            R"({"model":"decisions","state":"a","questions":{"tie":{"type":"choice","instructions":"?","criteria":{)") +
        options + "}}}}");
    Accept();
    Drain();
    const std::string response = Read(client);
    EXPECT_NE(response.find(std::string("\"choice\":\"") + options[1] +
                            "\",\"confidence\":0"),
              std::string::npos);
  }
  std::string options;
  for (size_t i = 0; i < 64; ++i) {
    if (i) {
      options += ",";
    }
    options += "\"o" + std::to_string(i) + "\":null";
  }
  const int huge = Send(
      std::string(
          R"({"model":"decisions","state":"a","questions":{"q":{"type":"choice","instructions":"?","criteria":{)") +
      options + "}}}}");
  Accept();
  Drain();
  EXPECT_EQ(Read(huge).find("HTTP/1.1 422"), 0u);
  const int next = Send(kNoul);
  Accept();
  Drain();
  EXPECT_EQ(Read(next).find("HTTP/1.1 200"), 0u);
}

TEST_P(SourceDecisionServiceTest, SharesRouterWithEmbeddingsAndRawUtilities) {
  ASSERT_NO_FATAL_FAILURE(StartRouter());
  const int decision = Send(kNoul);
  const int embedding =
      Send(R"({"model":"vectors","input":["a","bbb"],"dimensions":2})",
           "/v1/embeddings");
  const int tokenize =
      Send(R"({"model":"vectors","content":"aba"})", "/tokenize");
  EXPECT_NE(Read(Send("", "/healthz", "GET")).find("\"status\":\"ready\""),
            std::string::npos);
  const std::string catalog = Read(Send("", "/v1/models", "GET"));
  EXPECT_NE(
      catalog.find(
          R"({"object":"model","id":"decisions","loom":{"kind":"decision","endpoint":"/v1/systemone","tokenize":false}})"),
      std::string::npos);
  EXPECT_NE(
      catalog.find(
          R"({"object":"model","id":"vectors","loom":{"kind":"embedding","endpoint":"/v1/embeddings","tokenize":true}})"),
      std::string::npos);
  EXPECT_NE(Read(tokenize).find(R"({"tokens":[0,1,0]})"), std::string::npos);
  EXPECT_NE(Read(decision).find(R"("noul":0.666666687)"), std::string::npos);
  const std::string vectors = Read(embedding);
  EXPECT_NE(vectors.find(R"("embedding":[1,1])"), std::string::npos);
  EXPECT_NE(vectors.find(R"("embedding":[6,3])"), std::string::npos);
  const std::string snapshot = Read(Send("", "/v1/status", "GET"));
  EXPECT_NE(
      snapshot.find(
          R"({"object":"model","id":"decisions","loom":{"kind":"decision","endpoint":"/v1/systemone","tokenize":false,"status":{"phase":"idle","queued_requests":0,"active_requests":0,"oldest_queued_age_seconds":0.000000000,"oldest_active_age_seconds":0.000000000,"admission":"open","held_requests":0,"tokenizer_requests":0}}})"),
      std::string::npos);
  EXPECT_NE(
      snapshot.find(
          R"({"object":"model","id":"vectors","loom":{"kind":"embedding","endpoint":"/v1/embeddings","tokenize":true,"status":{"phase":"idle","queued_requests":0,"active_requests":0,"oldest_queued_age_seconds":0.000000000,"oldest_active_age_seconds":0.000000000,"admission":"open","held_requests":0,"tokenizer_requests":0}}})"),
      std::string::npos);
  EXPECT_EQ(Read(Send(R"({"model":"decisions","content":"a"})", "/tokenize"))
                .find("HTTP/1.1 404"),
            0u);
  EXPECT_EQ(Read(Send(kNoul, "/v1/embeddings")).find("HTTP/1.1 404"), 0u);
  EXPECT_EQ(
      Read(
          Send(
              R"({"model":"decisions","state":"a","questions":{"q":{"type":"noul","instructions":"?"}},"images":[]})"))
          .find("HTTP/1.1 422"),
      0u);
  EXPECT_EQ(Read(Send(kNoul)).find("HTTP/1.1 200"), 0u);
}

INSTANTIATE_TEST_SUITE_P(Workers, SourceDecisionServiceTest,
                         ::testing::Values(1u, 4u));

}  // namespace
