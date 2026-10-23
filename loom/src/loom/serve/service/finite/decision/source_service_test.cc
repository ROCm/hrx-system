// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "iree/async/proactor.h"
#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/server/resources.h"
#include "loom/serve/server/router.h"
#include "loom/serve/service/finite/decision/protocol.h"
#include "loom/serve/service/finite/embedding/protocol.h"
#include "loom/serve/service/finite/service.h"
#include "loom/serve/testing/http_harness.h"
#include "loom/serve/testing/model.h"
#include "loom/serve/testing/package.h"
#include "loom/serve/testing/service_application.h"
#include "loom/serve/transport/http/listener.h"

namespace {

class SourceDecisionServiceTest : public ::testing::Test {
 protected:
  void CreateModel(const char* source, loom_serve_finite_model_t** output) {
    const std::string file = source;
    const std::string directory = file.substr(0, file.find_last_of('/'));
    loom_serve::testing::ModelDocument document;
    IREE_ASSERT_OK(document.Load(directory + "/model.json"));
    const loom_serve_finite_model_options_t options = {
        .sources = document.value().sources,
        .asset_directory = IREE_SV("."),
        .options = IREE_SV("{\"batch_units\":64}"),
    };
    IREE_ASSERT_OK(loom_serve_finite_model_allocate(device_, &options,
                                                    host_allocator_, output));
  }
  void SetUp() override {
    const char* model_source = std::getenv("LOOM_SERVE_TEST_MODEL_SOURCE");
    const char* embedding_source =
        std::getenv("LOOM_SERVE_TEST_EMBEDDING_SOURCE");
    ASSERT_NE(model_source, nullptr);
    ASSERT_NE(embedding_source, nullptr);
    model_source_ = model_source;
    embedding_source_ = embedding_source;
    iree_notification_initialize(&notification_);
    const auto application_options =
        loom_serve_test_application_options_default();
    loom_serve_test_application_t* application = nullptr;
    IREE_ASSERT_OK(loom_serve_test_application_acquire(
        &application_options, host_allocator_, &application));
    host_ = loom_serve_test_application_host(application);
    resources_.device_group =
        loom_serve_test_application_device_group(application);
    device_ = loom_serve_test_application_device_at(application, 0);
    ASSERT_NO_FATAL_FAILURE(CreateModel(model_source_.c_str(), &models_[0]));
    IREE_ASSERT_OK(loom_serve_finite_service_allocate(
        host_, models_[0], loom_serve_decision_protocol(), IREE_SV("decisions"),
        16, 16, 0, nullptr, host_allocator_, &services_[0]));
    interfaces_[0] = loom_serve_finite_service_interface(services_[0]);
    EXPECT_EQ(interfaces_[0].capacity_domain,
              loom_serve_device_residency_cache(device_));
    const loom_serve_wake_callback_t observer = {
        [](void* self) {
          auto* test = static_cast<SourceDecisionServiceTest*>(self);
          test->wakes_.fetch_add(1);
          iree_notification_post(&test->notification_, IREE_ALL_WAITERS);
        },
        this};
    interfaces_[0].set_observer(interfaces_[0].self, observer);
    loom_serve_resources_set_observer(&resources_, observer);
    auto http_options = loom_serve_http_server_options_default();
    IREE_ASSERT_OK(loom_serve_http_server_allocate(
        loom_serve_host_proactor_pool(host_), &http_options, host_allocator_,
        &server_));
    auto listener_options = loom_serve_http_listener_options_default();
    listener_options.address = IREE_SV("127.0.0.1:0");
    IREE_ASSERT_OK(loom_serve_http_listener_allocate(
        server_, &listener_options, host_allocator_, &listener_));
    IREE_ASSERT_OK(http_.Initialize(server_, listener_));
  }
  void TearDown() override {
    if (http_.joinable()) {
      loom_serve_http_server_request_stop(server_);
      IREE_EXPECT_OK(http_.Join());
    }
    for (size_t i = 0; i < 2; ++i) {
      if (services_[i]) {
        interfaces_[i].set_observer(interfaces_[i].self, {});
        loom_serve_finite_service_free(services_[i]);
      }
    }
    IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
    loom_serve_http_server_free(server_);
    for (auto* model : models_) {
      IREE_EXPECT_OK(loom_serve_finite_model_free(model));
    }
    loom_serve_resources_set_observer(&resources_, {});
    iree_notification_deinitialize(&notification_);
  }
  int Send(const std::string& body, const char* target = "/v1/systemone",
           const char* method = "POST") {
    if (http_.failed() || HasFatalFailure()) {
      return -1;
    }
    const std::string request = std::string(method) + " " + target +
                                " HTTP/1.1\r\nHost: local\r\nContent-Type: "
                                "application/json\r\nContent-Length: " +
                                std::to_string(body.size()) + "\r\n\r\n" + body;
    int client = -1;
    [&] { IREE_ASSERT_OK(http_.Send(request, &client)); }();
    return client;
  }
  std::string Read(int client) {
    std::string response;
    if (http_.failed() || HasFatalFailure()) {
      if (client >= 0) {
        EXPECT_EQ(close(client), 0);
      }
      return response;
    }
    [&] {
      IREE_ASSERT_OK(http_.Read(client, /*allow_empty=*/false, &response));
    }();
    return response;
  }
  void Accept(int client,
              loom_serve_http_connection_t** out_connection = nullptr) {
    ASSERT_FALSE(HasFatalFailure());
    const loom_serve_http_request_t* request = nullptr;
    loom_serve_http_connection_t* connection = nullptr;
    IREE_ASSERT_OK(http_.TakeRequest(client, &connection, &request));
    IREE_ASSERT_OK(
        interfaces_[0].accept(interfaces_[0].self, connection, request, 0));
    if (out_connection) {
      *out_connection = connection;
    }
  }
  bool Step() {
    const uint64_t previous = wakes_.load();
    IREE_EXPECT_OK(loom_serve_resources_advance(&resources_));
    bool progress = false;
    IREE_EXPECT_OK(interfaces_[0].advance(interfaces_[0].self, 0, &progress));
    if (interfaces_[0].is_active(interfaces_[0].self) ||
        loom_serve_resources_is_pending(&resources_)) {
      struct loom_serve_test_decision_wait_t {
        // Shared release generation.
        std::atomic<uint64_t>* wakes;
        // Last observed generation before this owner turn.
        uint64_t previous;
      } wait{&wakes_, previous};
      iree_notification_await(
          &notification_,
          [](void* data) {
            const auto* wait =
                static_cast<loom_serve_test_decision_wait_t*>(data);
            return wait->wakes->load() > wait->previous;
          },
          &wait, iree_infinite_timeout());
      return true;
    }
    // A completion can publish a wake during advance after admission deferred.
    // Consume it with another owner turn before declaring the drain complete.
    return progress || wakes_.load() > previous;
  }
  void Drain() {
    while (Step()) {
    }
  }
  void StartRouter() {
    ASSERT_NO_FATAL_FAILURE(
        CreateModel(embedding_source_.c_str(), &models_[1]));
    IREE_ASSERT_OK(loom_serve_finite_service_allocate(
        host_, models_[1], loom_serve_embedding_protocol(), IREE_SV("vectors"),
        16, 16, 0, nullptr, host_allocator_, &services_[1]));
    interfaces_[1] = loom_serve_finite_service_interface(services_[1]);
    interfaces_[0].set_observer(interfaces_[0].self, {});
    loom_serve_resources_set_observer(&resources_, {});
    http_.Start([this]() -> iree_status_t {
      loom_serve_registry_t* registry = nullptr;
      IREE_RETURN_IF_ERROR(loom_serve_registry_allocate(
          2, interfaces_, 0, {}, host_allocator_, &registry));
      iree_status_t status = loom_serve_router_run(
          loom_serve_host_executor(host_), server_, registry,
          {
              .request_quota = 16,
              .holding_capacity = 8,
              .resources = resources_,
          },
          host_allocator_);
      return iree_status_join(status, loom_serve_registry_free(registry));
    });
  }
  // Allocator shared by all test-owned production components.
  const iree_allocator_t host_allocator_ = iree_allocator_system();
  // Source package for the decision model under test.
  std::string model_source_;
  // Independent source package used for cross-modality routing.
  std::string embedding_source_;
  // Real shared CPU/JIT executor and asynchronous I/O owner.
  loom_serve_host_t* host_ = nullptr;
  // Borrowed application resources progressed by the direct owner or router.
  loom_serve_resources_t resources_ = {};
  // Borrowed device record within the application group.
  loom_serve_device_t* device_ = nullptr;
  // Independently compiled source models, not mock callbacks.
  loom_serve_finite_model_t* models_[2] = {};
  // Bounded modality services borrowing the corresponding models.
  loom_serve_finite_service_t* services_[2] = {};
  // Exact production scheduling and routing interfaces.
  loom_serve_service_t interfaces_[2] = {};
  // Shared HTTP connection and request owner.
  loom_serve_http_server_t* server_ = nullptr;
  // Independent binding borrowing the HTTP owner and shared host pool.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Outer direct-driver readiness, never a worker dependency wait.
  iree_notification_t notification_;
  // Completion/release generation for direct scheduling turns.
  std::atomic<uint64_t> wakes_{0};
  // HTTP readiness and optional outer owner, joined before model destruction.
  loom_serve::testing::HttpHarness http_;
};

const char* const kNoul =
    R"({"model":"decisions","state":"a","questions":{"truth":{"type":"noul","instructions":"???","criteria":{"false":"","true":"x"}}}})";

void CheckSingleNoulResponse(const std::string& response) {
  SCOPED_TRACE(response);
  ASSERT_EQ(response.find("HTTP/1.1 200"), 0u);
  const size_t separator = response.find("\r\n\r\n");
  ASSERT_NE(separator, std::string::npos);
  iree_string_view_t value = iree_make_string_view(
      response.data() + separator + 4, response.size() - separator - 4);
  for (const char* key : {"answers", "truth", "noul"}) {
    IREE_ASSERT_OK(iree_json_lookup_object_value(
        value, iree_make_cstring_view(key), &value));
  }
  double probability = 0;
  IREE_ASSERT_OK(iree_json_parse_double(value, &probability));
  // Vulkan F32 division permits 2.5 ULP error for normal divisors. At 2/3,
  // one ULP is 2^-24. Exact dyadic cohort probabilities remain wire-checked.
  EXPECT_NEAR(probability, 2.0 / 3.0, 2.5 * 0x1p-24);
}

TEST_F(SourceDecisionServiceTest, PacksQuestionsAndRetainsResultsAcrossReset) {
  const int a = Send(R"({"model":"decisions","state":"a","questions":{
    "route":{"type":"choice","instructions":"???","criteria":{"z":null,"x":null}},
    "level":{"type":"score","instructions":"???","criteria":["","x"]}
  }})");
  ASSERT_NO_FATAL_FAILURE(Accept(a));
  const int b = Send(kNoul);
  ASSERT_NO_FATAL_FAILURE(Accept(b));
  const int c = Send(kNoul);
  loom_serve_http_connection_t* cancelled = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(c, &cancelled));
  const int bad = Send(
      R"({"model":"decisions","state":"a","questions":{"bad":{"type":"score","instructions":"?","criteria":["one"]}}})");
  ASSERT_NO_FATAL_FAILURE(Accept(bad));
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
  ASSERT_NO_FATAL_FAILURE(Accept(next));
  Drain();
  // A single-question cohort produces a different exact oracle distribution.
  // The earlier .75 requires all four questions in one device cohort.
  CheckSingleNoulResponse(Read(next));
}

TEST_F(SourceDecisionServiceTest,
       PreservesWireOrderAndRejectsOversizeSourceInput) {
  for (const char* options : {R"("z":null,"v":null)", R"("v":null,"z":null)"}) {
    const int client = Send(
        std::string(
            R"({"model":"decisions","state":"a","questions":{"tie":{"type":"choice","instructions":"?","criteria":{)") +
        options + "}}}}");
    ASSERT_NO_FATAL_FAILURE(Accept(client));
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
  ASSERT_NO_FATAL_FAILURE(Accept(huge));
  Drain();
  EXPECT_EQ(Read(huge).find("HTTP/1.1 422"), 0u);
  const int next = Send(kNoul);
  ASSERT_NO_FATAL_FAILURE(Accept(next));
  Drain();
  EXPECT_EQ(Read(next).find("HTTP/1.1 200"), 0u);
}

TEST_F(SourceDecisionServiceTest, SelectsUploadedDecisionExecution) {
  ASSERT_NO_FATAL_FAILURE(StartRouter());
  CheckSingleNoulResponse(Read(Send(kNoul)));
  // With two routes, inventory requires an explicit model selector.
  const std::string inventory_request =
      "GET /v1/models/variants HTTP/1.1\r\nHost: local\r\n"
      "X-Loom-Model: decisions\r\n\r\n";
  int client = -1;
  IREE_ASSERT_OK(http_.Send(inventory_request, &client));
  const auto inventory = Read(client);
  auto number = [](const std::string& response, const char* field,
                   const char* object = nullptr) {
    uint64_t result = 0;
    [&] {
      const auto start = response.find("\r\n\r\n");
      ASSERT_NE(start, std::string::npos) << response;
      auto value = iree_make_string_view(response.data() + start + 4,
                                         response.size() - start - 4);
      if (object) {
        IREE_ASSERT_OK(iree_json_lookup_object_value(
            value, iree_make_cstring_view(object), &value));
      }
      IREE_ASSERT_OK(iree_json_lookup_object_value(
          value, iree_make_cstring_view(field), &value));
      IREE_ASSERT_OK(iree_json_parse_uint64(value, &result));
    }();
    return result;
  };
  const uint64_t original = number(inventory, "selected");
  const auto directory = std::filesystem::path(model_source_).parent_path();
  std::ifstream input(directory / "decide.loom");
  ASSERT_TRUE(input.good());
  std::string source((std::istreambuf_iterator<char>(input)), {});
  const std::string constant = "scalar.constant 1 : i32";
  const auto position = source.find(constant);
  ASSERT_NE(position, std::string::npos);
  source.replace(position, constant.size(), "scalar.constant 2 : i32");
  loom_serve::testing::SourcePackage package;
  package.Add("model/decide.loom", source);
  IREE_ASSERT_OK(package.AddFile("model/config.loom",
                                 (directory / "config.loom").string()));
  IREE_ASSERT_OK(http_.Send(
      package.Request(
          "/v1/models/variants",
          "{\"model\":\"decisions\",\"base\":" + std::to_string(original) +
              ",\"sources\":[\"config.loom\",\"decide.loom\"],\"wait\":true}"),
      &client));
  const auto staged = Read(client);
  ASSERT_EQ(staged.find("HTTP/1.1 200"), 0u) << staged;
  const uint64_t candidate = number(staged, "id", "variant");
  const auto selected = Read(
      Send("{\"model\":\"decisions\",\"variant\":" + std::to_string(candidate) +
               ",\"expected\":" + std::to_string(original) + "}",
           "/v1/models/variants/select"));
  ASSERT_EQ(selected.find("HTTP/1.1 200"), 0u) << selected;
  const auto changed = Read(Send(kNoul));
  EXPECT_NE(changed.find(R"("noul":0.6)"), std::string::npos) << changed;
  const auto restored = Read(
      Send("{\"model\":\"decisions\",\"variant\":" + std::to_string(original) +
               ",\"expected\":" + std::to_string(candidate) + "}",
           "/v1/models/variants/select"));
  ASSERT_EQ(restored.find("HTTP/1.1 200"), 0u) << restored;
  CheckSingleNoulResponse(Read(Send(kNoul)));
  EXPECT_NE(Read(Send(R"({"model":"vectors","input":"a"})", "/v1/embeddings"))
                .find(R"("embedding":[1,1,1,1])"),
            std::string::npos);
}

TEST_F(SourceDecisionServiceTest, SharesRouterWithEmbeddingsAndRawUtilities) {
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
          R"("input_modalities":["text"],"output_modalities":["decision"],"loom":{"kind":"decision","endpoint":"/v1/systemone","tokenize":false,"variants":)"),
      std::string::npos)
      << catalog;
  EXPECT_NE(
      catalog.find(
          R"("input_modalities":["text"],"output_modalities":["embedding"],"loom":{"kind":"embedding","endpoint":"/v1/embeddings","tokenize":true,"variants":)"),
      std::string::npos)
      << catalog;
  EXPECT_NE(Read(tokenize).find(R"({"tokens":[0,1,0]})"), std::string::npos);
  CheckSingleNoulResponse(Read(decision));
  const std::string vectors = Read(embedding);
  EXPECT_NE(vectors.find(R"("embedding":[1,1])"), std::string::npos);
  EXPECT_NE(vectors.find(R"("embedding":[6,3])"), std::string::npos);
  const std::string snapshot = Read(Send("", "/v1/status", "GET"));
  // Variant identities are process-issued rather than baked model constants.
  // Check each complete idle status within its own model object.
  for (const char* name : {"decisions", "vectors"}) {
    const auto begin = snapshot.find(std::string("\"id\":\"") + name + "\"");
    ASSERT_NE(begin, std::string::npos) << snapshot;
    const auto end = snapshot.find("{\"object\":\"model\"", begin);
    const auto model =
        snapshot.substr(begin, end == std::string::npos ? end : end - begin);
    EXPECT_NE(model.find(R"("variants":{"selected":)"), std::string::npos)
        << model;
    EXPECT_NE(
        model.find(
            R"(,"count":1},"status":{"phase":"idle","queued_requests":0,"active_requests":0,"oldest_queued_age_seconds":0.000000000,"oldest_active_age_seconds":0.000000000,"admission":"open","held_requests":0,"tokenizer_requests":0}}})"),
        std::string::npos)
        << model;
  }
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

}  // namespace
