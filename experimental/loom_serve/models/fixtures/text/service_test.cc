// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/service.h"

#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "experimental/loom_serve/http/listener.h"
#include "experimental/loom_serve/models/fixtures/stalled_stderr.h"
#include "experimental/loom_serve/serving/model_registration.h"
#include "experimental/loom_serve/serving/router.h"
#include "iree/async/proactor.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

IREE_FLAG(string, model_source, "", "Deterministic text bootstrap source.");

namespace {

// Independent dense arithmetic, with no knowledge of device packets, pooled
// addresses, slot ownership or scheduling chunks. EOS is a consumed delimiter
// in replayed history but an unconsumed prediction at generation completion.
class Oracle {
 public:
  explicit Oracle(uint32_t bias = 2) : bias_(bias) {}

  void Consume(int32_t token) {
    hidden_ = (5 * hidden_ + token + bias_) % 97;
    history_ = (7 * history_ + hidden_ + position_) % 101;
    sum_ += history_;
    ++position_;
  }

  void Consume(const std::string& text) {
    for (char character : text) {
      Consume(Id(character));
    }
  }

  int32_t Predict() const {
    return position_ % 11 == 0 ? 3 : (hidden_ + sum_) % 3;
  }

  static int32_t Id(char character) {
    return character == 'a'   ? 0
           : character == 'b' ? 1
           : character == 'x' ? 2
                              : 3;
  }

  static std::string Complete(const std::string& prompt, size_t limit,
                              uint32_t bias = 2) {
    Oracle oracle(bias);
    oracle.Consume(prompt);
    std::string response;
    for (size_t i = 0; i < limit; ++i) {
      const int32_t token = oracle.Predict();
      if (token == 3) {
        break;
      }
      response.push_back("abx"[token]);
      oracle.Consume(token);
    }
    return response;
  }

 private:
  // Actual immutable checkpoint value.
  const uint32_t bias_;
  // Recurrent value, independent of row identity.
  uint32_t hidden_ = 0;
  // Last cached history value.
  uint32_t history_ = 0;
  // Sum of all cached history values used by selection.
  uint32_t sum_ = 0;
  // Number of consumed inputs, excluding a pending prediction.
  uint32_t position_ = 0;
};

class SourceTextServiceTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(loom_serve_host_create(GetParam(), &host_, allocator_));
    const loom_serve_device_options_t device_options = {.uri =
                                                            IREE_SV("amdgpu")};
    IREE_ASSERT_OK(loom_serve_device_group_create(host_, 1, &device_options,
                                                  &group_, allocator_));
    device_ = loom_serve_device_group_at(group_, 0);
    const std::string source = FLAG_model_source;
    directory_ = source.substr(0, source.find_last_of('/'));
    tokenizer_ = directory_ + "/tokenizer.json";
    weights_ = iree::testing::MakeTempFilePath("text_fixture", ".safetensors");
    ASSERT_NO_FATAL_FAILURE(WriteWeights(2));
    auto options = ModelOptions();
    IREE_ASSERT_OK(
        loom_serve_text_model_create(device_, &options, &model_, allocator_));
    auto server_options = loom_serve_http_server_options_default();
    IREE_ASSERT_OK(
        loom_serve_http_server_create(loom_serve_host_proactor_pool(host_),
                                      &server_options, &server_, allocator_));
    auto listener_options = loom_serve_http_listener_options_default();
    listener_options.address = IREE_SV("127.0.0.1:0");
    IREE_ASSERT_OK(loom_serve_http_listener_create(server_, &listener_options,
                                                   &listener_, allocator_));
  }

  void WriteWeights(uint32_t bias) {
    std::string header =
        R"({"bias":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}})";
    header.append((8 - header.size() % 8) % 8, ' ');
    std::ofstream file(weights_, std::ios::binary);
    uint8_t bytes[8];
    iree_unaligned_store_le_u64(bytes, header.size());
    file.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    file.write(header.data(), header.size());
    iree_unaligned_store_le_u32(bytes, bias);
    file.write(reinterpret_cast<const char*>(bytes), 4);
    file.close();
    ASSERT_TRUE(file.good());
  }

  loom_serve_text_options_t ModelOptions() const {
    return {
        .source_directory = iree_make_cstring_view(directory_.c_str()),
        .prefill_capacity = 32,
        .context_capacity = 96,
        .pool_capacity = 256,
        .epoch_count = IREE_ARRAYSIZE(shapes_),
        .epoch_shapes = shapes_,
        .kernel_sanitizer = {.type = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
                             .structure_size =
                                 sizeof(loomc_sanitizer_options_t)},
        .weights_path = iree_make_cstring_view(weights_.c_str()),
        .tokenizer_path = iree_make_cstring_view(tokenizer_.c_str()),
        .row_count = 4,
        .checkpoint_capacity = 2,
    };
  }

  void TearDown() override {
    if (service_.joinable()) {
      loom_serve_http_server_request_stop(server_);
      service_.join();
      IREE_EXPECT_OK(status_);
    }
    IREE_EXPECT_OK(loom_serve_http_listener_destroy(listener_));
    loom_serve_http_server_destroy(server_);
    IREE_EXPECT_OK(loom_serve_text_model_destroy(model_));
    IREE_EXPECT_OK(loom_serve_device_group_destroy(group_));
    loom_serve_host_release(host_);
  }

  loom_serve_text_service_options_t ServiceOptions() const {
    return {
        .name = IREE_SV("fixture-text"),
        .row_count = 4,
        .chunk_size = 3,
        .default_max_tokens = 16,
        .pending_capacity = 16,
        .continuation_epochs = 1,
    };
  }

  void Start(iree_host_size_t chunk_size = 3) {
    service_ = std::thread([this, chunk_size] {
      auto options = ServiceOptions();
      options.chunk_size = chunk_size;
      loom_serve_text_service_t* service = nullptr;
      status_ = loom_serve_text_service_create(model_, &options, nullptr,
                                               &service, allocator_);
      if (iree_status_is_ok(status_)) {
        const auto interface = loom_serve_text_service_interface(service);
        loom_serve_registry_t* registry = nullptr;
        IREE_ASSERT_OK(loom_serve_registry_create(1, &interface, 0, {},
                                                  &registry, allocator_));
        status_ = loom_serve_router_run(
            loom_serve_host_executor(host_), server_, registry,
            {.request_quota = options.row_count,
             .holding_capacity =
                 loom_serve_http_server_connection_capacity(server_) / 2},
            allocator_);
        IREE_ASSERT_OK(loom_serve_registry_destroy(registry));
      }
      loom_serve_text_service_destroy(service, iree_status_code(status_));
    });
  }

  void StartManaged() {
    IREE_ASSERT_OK(loom_serve_text_model_destroy(model_));
    model_ = nullptr;
    service_ = std::thread([this] {
      const loom_serve_model_registration_options_t options = {
          .device = device_, .heartbeat_interval = 0};
      loom_serve_registry_t* registry = nullptr;
      IREE_ASSERT_OK(loom_serve_registry_create(
          0, nullptr, 3, loom_serve_model_registration_factory(&options),
          &registry, allocator_));
      status_ = loom_serve_router_run(
          loom_serve_host_executor(host_), server_, registry,
          {.request_quota = 16, .holding_capacity = 8}, allocator_);
      status_ =
          iree_status_join(status_, loom_serve_registry_destroy(registry));
    });
  }

  std::string Declaration(const std::string& name, const std::string& source,
                          const char* admission = "open") {
    return "{\"kind\":\"text\",\"name\":\"" + name + "\",\"source\":\"" +
           source + "\",\"weights\":\"" + weights_ + "\",\"tokenizer\":\"" +
           tokenizer_ +
           "\",\"rows\":4,\"context_capacity\":96,"
           "\"pool_capacity\":256,\"prefill_capacity\":32,"
           "\"checkpoint_capacity\":2,\"mtp_depth\":0,\"admission\":\"" +
           admission + "\"}";
  }

  std::string WaitModel(const std::string& name, const std::string& marker) {
    for (;;) {
      const auto listing = Read(Send("", "", "/v1/status", "GET"));
      const size_t begin = listing.find("\"id\":\"" + name + "\"");
      const size_t end = listing.find("{\"object\":\"model\"", begin);
      const auto state =
          begin == std::string::npos
              ? std::string()
              : listing.substr(begin,
                               end == std::string::npos ? end : end - begin);
      if (marker.empty() ? state.empty()
                         : state.find(marker) != std::string::npos) {
        return state;
      }
    }
  }

  int Send(const std::string& body, const std::string& headers = "",
           const std::string& target = "/v1/chat/completions",
           const std::string& method = "POST") {
    const int client = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    EXPECT_GE(client, 0);
    const auto* address = loom_serve_http_listener_address(listener_);
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
    std::string response;
    char bytes[4096];
    ssize_t count;
    while ((count = recv(client, bytes, sizeof(bytes), 0)) > 0) {
      response.append(bytes, count);
    }
    EXPECT_EQ(count, 0);
    EXPECT_EQ(close(client), 0);
    return response;
  }

  static std::string Message(const std::string& role,
                             const std::string& content) {
    return "{\"role\":\"" + role + "\",\"content\":\"" + content + "\"}";
  }

  static std::string Request(const std::string& messages, size_t limit = 16) {
    return "{\"model\":\"fixture-text\",\"stream\":true,\"stream_options\":{"
           "\"include_usage\":true},\"max_tokens\":" +
           std::to_string(limit) + ",\"messages\":[" + messages + "]}";
  }

  static std::string Content(const std::string& response) {
    EXPECT_EQ(response.find("HTTP/1.1 200"), 0u) << response;
    EXPECT_NE(response.find("data: [DONE]"), std::string::npos) << response;
    std::string result;
    const std::string marker = "\"content\":\"";
    size_t cursor = 0;
    while ((cursor = response.find(marker, cursor)) != std::string::npos) {
      cursor += marker.size();
      const size_t end = response.find('"', cursor);
      EXPECT_NE(end, std::string::npos);
      if (end == std::string::npos) {
        break;
      }
      result.append(response, cursor, end - cursor);
      cursor = end + 1;
    }
    EXPECT_EQ(result.find_first_not_of("abx"), std::string::npos);
    return result;
  }

  // Shared allocator for all production owners.
  const iree_allocator_t allocator_ = iree_allocator_system();
  // Host geometry buckets share one command in the source bootstrap.
  const loom_serve_packing_shape_t shapes_[3] = {{4, 1}, {16, 4}, {32, 4}};
  // External source package path.
  std::string directory_;
  // Real BPE tokenizer asset.
  std::string tokenizer_;
  // Tiny generated checkpoint owned by the test runner's temp directory.
  std::string weights_;
  // Shared task/JIT owner outliving device and model.
  loom_serve_host_t* host_ = nullptr;
  // Application group enclosing all model and device resources.
  loom_serve_device_group_t* group_ = nullptr;
  // Borrowed device record within the application group.
  loom_serve_device_t* device_ = nullptr;
  // Actual source-defined text model, with no native mock implementation.
  loom_serve_text_model_t* model_ = nullptr;
  // Real loopback HTTP carrier.
  loom_serve_http_server_t* server_ = nullptr;
  // Independent binding borrowing the HTTP owner and shared host pool.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Outer application thread, not a model worker dependency wait.
  std::thread service_;
  // Consumed after joining the application thread.
  iree_status_t status_ = iree_ok_status();
};

TEST_P(SourceTextServiceTest,
       DynamicTextAdmissionCheckpointRetirementAndReuse) {
  StartManaged();
  EXPECT_NE(Read(Send("", "", "/v1/models", "GET")).find(R"("data":[])"),
            std::string::npos);
  loom_serve::testing::StalledStderr stalled;
  ASSERT_NO_FATAL_FAILURE(stalled.Begin());
  EXPECT_EQ(Read(Send(Declaration("fixture-text", directory_, "drained"), "",
                      "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  const auto declared = WaitModel("fixture-text", R"("admission":"drained")");
  EXPECT_NE(declared.find(R"("registration":"registered")"), std::string::npos);
  const auto initial_log =
      Read(Send("", "X-Loom-Model: fixture-text\r\n", "/logz", "GET"));
  EXPECT_NE(initial_log.find(R"("event":"stream_begin","sequence":0)"),
            std::string::npos);
  const std::string prompt = "abxa";
  const int held =
      Send(Request(Message("user", prompt), 2), "X-Loom-Session: retained\r\n");
  const int tokenizer =
      Send(R"({"model":"fixture-text","content":"abx"})", "", "/tokenize");
  WaitModel("fixture-text", R"("held_requests":2)");
  EXPECT_EQ(Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/undrain"))
                .find("HTTP/1.1 200"),
            0u);
  EXPECT_EQ(Content(Read(held)), Oracle::Complete(prompt, 2));
  EXPECT_NE(Read(tokenizer).find(R"({"tokens":[0,1,2]})"), std::string::npos);
  const auto first = WaitModel("fixture-text", R"("registration":"ready")");
  const auto pinned = Read(
      Send("", "X-Loom-Model: fixture-text\r\nX-Loom-Session: retained\r\n",
           "/v1/checkpoints/base"));
  EXPECT_EQ(pinned.find("HTTP/1.1 201"), 0u) << pinned;
  EXPECT_EQ(
      Read(Send(Declaration("peer", directory_), "", "/v1/models/register"))
          .find("HTTP/1.1 202"),
      0u);
  EXPECT_EQ(Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/drain"))
                .find("HTTP/1.1 200"),
            0u);
  const int removed = Send(Request(Message("user", "b")));
  WaitModel("fixture-text", R"("held_requests":1)");
  std::string peer = Request(Message("user", "ab"), 2);
  peer.replace(peer.find("fixture-text"), 12, "peer");
  EXPECT_EQ(Content(Read(Send(peer))), Oracle::Complete("ab", 2));
  EXPECT_EQ(Read(Send(R"({"model":"fixture-text","mode":"drain"})", "",
                      "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  EXPECT_EQ(Read(removed).find("HTTP/1.1 410"), 0u);
  WaitModel("fixture-text", {});
  EXPECT_EQ(Read(Send(Declaration("fixture-text", directory_), "",
                      "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  EXPECT_EQ(Content(Read(Send(Request(Message("user", prompt), 2)))),
            Oracle::Complete(prompt, 2));
  const auto replacement =
      WaitModel("fixture-text", R"("registration":"ready")");
  const auto incarnation = [](const std::string& state) {
    const auto start = state.find("\"incarnation\":");
    EXPECT_NE(start, std::string::npos);
    return state.substr(start, state.find(',', start) - start);
  };
  EXPECT_NE(incarnation(first), incarnation(replacement));
  EXPECT_EQ(Read(Send(R"({"model":"fixture-text","mode":"abort"})", "",
                      "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModel("fixture-text", {});
  EXPECT_EQ(Content(Read(Send(peer))), Oracle::Complete("ab", 2));
}

TEST_P(SourceTextServiceTest, FailedTextSourceLeavesPeerAndDiagnosticsUsable) {
  const auto package = iree::testing::MakeTempFilePath("failed_text_source");
  ASSERT_TRUE(std::filesystem::create_directory(package));
  for (const char* file :
       {"prepare.loom", "control.loom", "sources.txt", "weights.loom"}) {
    ASSERT_TRUE(std::filesystem::copy_file(directory_ + "/" + file,
                                           package + "/" + file));
  }
  {
    std::ofstream source(package + "/advance.loom");
    source << "this is not a source program\n";
    ASSERT_TRUE(source.good());
  }
  StartManaged();
  loom_serve::testing::StalledStderr stalled;
  ASSERT_NO_FATAL_FAILURE(stalled.Begin());
  EXPECT_EQ(Read(Send(Declaration("fixture-text", directory_), "",
                      "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  EXPECT_EQ(Read(Send(Declaration("broken", package, "drained"), "",
                      "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  const int held =
      Send(R"({"model":"broken","content":"ab"})", "", "/tokenize");
  WaitModel("broken", R"("held_requests":1)");
  EXPECT_EQ(Read(Send(R"({"model":"broken"})", "", "/v1/models/undrain"))
                .find("HTTP/1.1 200"),
            0u);
  EXPECT_EQ(Read(held).find("HTTP/1.1 503"), 0u);
  const auto failed = WaitModel("broken", R"("registration":"failed")");
  EXPECT_NE(failed.find("advance.loom"), std::string::npos);
  EXPECT_NE(failed.find(R"("diagnostics":[{)"), std::string::npos);
  EXPECT_EQ(Content(Read(Send(Request(Message("user", "ab"), 2)))),
            Oracle::Complete("ab", 2));
  EXPECT_EQ(Read(Send(R"({"model":"broken"})", "", "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModel("broken", {});
}

TEST_P(SourceTextServiceTest, StoppingAccountsForQueuedControlAndCpuBorrowers) {
  auto options = ServiceOptions();
  loom_serve_text_service_t* service = nullptr;
  IREE_ASSERT_OK(loom_serve_text_service_create(model_, &options, nullptr,
                                                &service, allocator_));
  auto destroy = [](loom_serve_text_service_t* value) {
    loom_serve_text_service_destroy(value, IREE_STATUS_OK);
  };
  std::unique_ptr<loom_serve_text_service_t, decltype(destroy)> owner(service,
                                                                      destroy);
  const auto interface = loom_serve_text_service_interface(service);
  auto* notification = loom_serve_http_server_notification(server_);
  interface.set_observer(interface.self,
                         {[](void* payload) {
                            iree_notification_post(
                                static_cast<iree_notification_t*>(payload),
                                IREE_ALL_WAITERS);
                          },
                          notification});
  const auto accept = [&] {
    for (;;) {
      const auto ticket = iree_notification_prepare_wait(notification);
      const loom_serve_http_request_t* request = nullptr;
      auto* connection = loom_serve_http_server_take_request(server_, &request);
      if (connection) {
        iree_notification_cancel_wait(notification);
        IREE_ASSERT_OK(
            interface.accept(interface.self, connection, request, 0));
        return;
      }
      iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  };
  const int generation =
      Send(Request(Message("user", "ab")), "X-Loom-Session: first\r\n");
  ASSERT_NO_FATAL_FAILURE(accept());
  // CPU completion may already be posted, but only the owner can consume it.
  // Checkpoint and generation claims queue behind that real accepted borrower.
  const int control =
      Send("", "X-Loom-Session: first\r\n", "/v1/checkpoints/base");
  ASSERT_NO_FATAL_FAILURE(accept());
  const int queued =
      Send(Request(Message("user", "ab")), "X-Loom-Session: second\r\n");
  ASSERT_NO_FATAL_FAILURE(accept());
  EXPECT_EQ(interface.snapshot(interface.self).queued_requests, 3u);
  bool complete = false;
  while (!complete) {
    const auto ticket = iree_notification_prepare_wait(notification);
    bool progress = false;
    IREE_ASSERT_OK(interface.advance(
        interface.self, LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING, &progress));
    const auto state = interface.snapshot(interface.self);
    complete = !interface.is_pending(interface.self) &&
               !state.queued_requests && !state.active_requests;
    if (complete || progress) {
      iree_notification_cancel_wait(notification);
    } else {
      iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  EXPECT_EQ(interface.telemetry.counters->requests, 3u);
  EXPECT_EQ(interface.telemetry.counters->cancelled, 3u);
  EXPECT_EQ(interface.telemetry.counters->completed, 0u);
  EXPECT_TRUE(Read(generation).empty());
  EXPECT_TRUE(Read(control).empty());
  EXPECT_TRUE(Read(queued).empty());
  EXPECT_FALSE(interface.is_active(interface.self));
}

TEST_P(SourceTextServiceTest, TelemetryWrapDoesNotBlockTextOrChangeCounters) {
  Start();
  auto poll = [&](const std::string& cursor) {
    return Read(Send("",
                     "X-Loom-Model: fixture-text\r\n" +
                         (cursor.empty() ? std::string()
                                         : "X-Loom-Cursor: " + cursor + "\r\n"),
                     "/logz", "GET"));
  };
  auto cursor_of = [](const std::string& response) {
    const std::string marker = R"({"event":"cursor","value":")";
    const size_t start = response.find(marker);
    EXPECT_NE(start, std::string::npos) << response;
    if (start == std::string::npos) {
      return std::string();
    }
    const size_t first = start + marker.size();
    return response.substr(first, response.find('"', first) - first);
  };
  const std::string initial = poll("");
  ASSERT_EQ(initial.find("HTTP/1.1 200"), 0u);
  EXPECT_NE(initial.find(R"("event":"stream_begin","sequence":0)"),
            std::string::npos);
  const std::string slow = cursor_of(initial);
  std::string fast = slow;
  std::string events;
  auto drain = [&] {
    for (;;) {
      const std::string response = poll(fast);
      EXPECT_EQ(response.find(R"("event":"gap")"), std::string::npos);
      events += response;
      const std::string next = cursor_of(response);
      if (next == fast) {
        return response;
      }
      fast = next;
    }
  };
  loom_serve::testing::StalledStderr stalled;
  ASSERT_NO_FATAL_FAILURE(stalled.Begin());
  uint64_t input_tokens = 0;
  for (size_t wave = 0; wave < 10; ++wave) {
    std::vector<int> clients;
    std::vector<std::string> prompts;
    for (size_t row = 0; row < 4; ++row) {
      prompts.push_back("abxa" + std::string((wave * 4 + row) % 3, 'b'));
      clients.push_back(
          Send(Request(Message("user", prompts.back()), 2),
               "X-Loom-Session: row" + std::to_string(row) + "\r\n"));
      input_tokens += prompts.back().size() + 1;
    }
    for (size_t row = 0; row < clients.size(); ++row) {
      EXPECT_EQ(Content(Read(clients[row])), Oracle::Complete(prompts[row], 2));
    }
    drain();
  }
  // Retained continuation counts only appended work, not the shared prefix.
  const std::string first = Content(Read(
      Send(Request(Message("user", "abxa"), 2), "X-Loom-Session: root\r\n")));
  EXPECT_EQ(first, Oracle::Complete("abxa", 2));
  const std::string continued = Read(
      Send(Request(Message("user", "abxa") + "," + Message("assistant", first) +
                       "," + Message("user", "b"),
                   2),
           "X-Loom-Session: root\r\n"));
  EXPECT_EQ(Content(continued), Oracle::Complete("abxa" + first + "#b", 2));
  EXPECT_NE(continued.find(R"("cached_tokens":5)"), std::string::npos);
  input_tokens += 5 + 4;
  EXPECT_EQ(Read(Send("", "X-Loom-Session: root\r\n", "/v1/checkpoints/pin"))
                .find("HTTP/1.1 201"),
            0u);
  EXPECT_EQ(
      Read(Send("", "", "/v1/checkpoints/pin", "DELETE")).find("HTTP/1.1 200"),
      0u);
  EXPECT_EQ(
      Read(Send(Request(Message("user", "a")), "X-Loom-Session: bad space\r\n"))
          .find("HTTP/1.1 400"),
      0u);
  const std::string latest = drain();
  const std::string totals =
      R"("requests":45,"completed":44,"rejected":1,"cancelled":0,"failed":0,"input_tokens":)" +
      std::to_string(input_tokens) + R"(,"output_tokens":84,)";
  EXPECT_NE(latest.find(totals), std::string::npos) << latest;
  for (const char* event :
       {"prepared", "admit", "epoch", "span", "request_stats", "session_begin",
        "session_end", "checkpoint_pin", "checkpoint_release"}) {
    EXPECT_NE(events.find(std::string("\"event\":\"") + event + "\""),
              std::string::npos)
        << event;
  }
  // No prompt text enters observation records. Slow and new readers resume
  // independently after overwrite while current totals remain exact.
  EXPECT_EQ(events.find("abxa"), std::string::npos);
  const std::string resumed = poll(slow);
  EXPECT_NE(resumed.find(R"({"event":"gap","from":1,)"), std::string::npos);
  EXPECT_NE(resumed.find(totals), std::string::npos);
  EXPECT_NE(poll("").find(R"("partial":true)"), std::string::npos);
  EXPECT_EQ(cursor_of(poll(fast)), fast);
  loom_serve_http_server_request_stop(server_);
  service_.join();
  IREE_EXPECT_OK(status_);
}

TEST_P(SourceTextServiceTest, ChunkingRowsAndForksMatchDenseOracle) {
  const std::string prompt = "abxabbaaxbabx";
  std::vector<int32_t> tokens;
  for (char character : prompt) {
    tokens.push_back(Oracle::Id(character));
  }
  Oracle reference;
  reference.Consume(prompt);
  auto* full = loom_serve_text_model_row(model_, 0);
  auto* chunks = loom_serve_text_model_row(model_, 1);
  IREE_ASSERT_OK(
      loom_serve_text_row_prefill(full, tokens.size(), tokens.data()));
  for (size_t offset = 0; offset < tokens.size(); offset += 3) {
    IREE_ASSERT_OK(loom_serve_text_row_prefill(
        chunks, std::min<size_t>(3, tokens.size() - offset),
        tokens.data() + offset));
  }
  EXPECT_EQ(loom_serve_text_row_token(full), reference.Predict());
  EXPECT_EQ(loom_serve_text_row_token(chunks), reference.Predict());
  loom_serve_text_checkpoint_t* checkpoint = nullptr;
  IREE_ASSERT_OK(loom_serve_text_row_try_pin(full, &checkpoint));
  ASSERT_NE(checkpoint, nullptr);
  auto* fork = loom_serve_text_model_row(model_, 2);
  bool restored = false;
  IREE_ASSERT_OK(loom_serve_text_row_try_restore(fork, checkpoint, &restored));
  ASSERT_TRUE(restored);
  const auto before = loom_serve_text_model_pool_usage(model_);
  const int32_t a = 0;
  const int32_t b = 1;
  const loom_serve_text_span_t spans[] = {
      {0, 1, &a, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
      {2, 1, &b, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
  };
  IREE_ASSERT_OK(
      loom_serve_text_model_epoch(model_, 1, IREE_ARRAYSIZE(spans), spans));
  Oracle branch = reference;
  reference.Consume(a);
  branch.Consume(b);
  EXPECT_EQ(loom_serve_text_row_token(full), reference.Predict());
  EXPECT_EQ(loom_serve_text_row_token(fork), branch.Predict());
  // Both writers detach the pinned partial page; the immutable endpoint stays.
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available + 8,
            before.available);
  IREE_ASSERT_OK(
      loom_serve_text_row_try_restore(chunks, checkpoint, &restored));
  ASSERT_TRUE(restored);
  IREE_ASSERT_OK(loom_serve_text_row_prefill(chunks, 1, &b));
  EXPECT_EQ(loom_serve_text_row_token(chunks), branch.Predict());
  loom_serve_text_checkpoint_release(checkpoint);
  for (size_t i = 0; i < 4; ++i) {
    IREE_ASSERT_OK(
        loom_serve_text_row_reset(loom_serve_text_model_row(model_, i)));
  }
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available, 256u);
}

TEST_P(SourceTextServiceTest, CancelledPreparationRetainsPartialOwnership) {
  const auto before = loom_serve_device_memory_statistics(device_);
  auto options = ModelOptions();
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  loom_serve_text_load_t* load = nullptr;
  IREE_ASSERT_OK(loom_serve_text_load_create(
      device_, &options,
      [](void* payload) {
        iree_notification_post(static_cast<iree_notification_t*>(payload),
                               IREE_ALL_WAITERS);
      },
      &notification, &load, allocator_));
  bool complete = false;
  IREE_ASSERT_OK(loom_serve_text_load_advance(load, 0, &complete));
  ASSERT_FALSE(complete);
  iree_status_t status = iree_ok_status();
  while (!complete) {
    const auto ticket = iree_notification_prepare_wait(&notification);
    status = loom_serve_text_load_advance(
        load, LOOM_SERVE_TEXT_LOAD_FLAG_CANCEL, &complete);
    if (complete) {
      iree_notification_cancel_wait(&notification);
    } else {
      IREE_ASSERT_OK(status);
      iree_notification_commit_wait(&notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, status);
  auto* partial = loom_serve_text_load_take_model(load);
  ASSERT_NE(partial, nullptr);
  loom_serve_text_load_destroy(load);
  loom_serve_text_model_retire(partial, nullptr, nullptr);
  ASSERT_TRUE(loom_serve_text_model_is_retired(partial));
  IREE_ASSERT_OK(loom_serve_text_model_destroy_retired(partial));
  iree_notification_deinitialize(&notification);
  const auto after = loom_serve_device_memory_statistics(device_);
  EXPECT_EQ(after.retained.reserved_bytes, before.retained.reserved_bytes);
  EXPECT_EQ(after.retained.committed_bytes, before.retained.committed_bytes);
}

TEST_P(SourceTextServiceTest, FailedLinkRetiresAcceptedInitialization) {
  const auto package = iree::testing::MakeTempFilePath("failed_text_link");
  ASSERT_TRUE(std::filesystem::create_directory(package));
  for (const char* file :
       {"prepare.loom", "advance.loom", "sources.txt", "weights.loom"}) {
    ASSERT_TRUE(std::filesystem::copy_file(directory_ + "/" + file,
                                           package + "/" + file));
  }
  {
    std::ifstream input(directory_ + "/control.loom");
    std::string source((std::istreambuf_iterator<char>(input)), {});
    const auto offset = source.find("import(\"runner\")");
    ASSERT_NE(offset, std::string::npos);
    source.replace(offset, std::string("import(\"runner\")").size(),
                   "import(\"missing\")");
    std::ofstream output(package + "/control.loom");
    output << source;
    ASSERT_TRUE(output.good());
  }
  const auto before = loom_serve_device_memory_statistics(device_);
  auto options = ModelOptions();
  options.source_directory = iree_make_cstring_view(package.c_str());
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  const auto wake = [](void* payload) {
    iree_notification_post(static_cast<iree_notification_t*>(payload),
                           IREE_ALL_WAITERS);
  };
  loom_serve_text_load_t* load = nullptr;
  IREE_ASSERT_OK(loom_serve_text_load_create(device_, &options, wake,
                                             &notification, &load, allocator_));
  bool complete = false;
  iree_status_t status = iree_ok_status();
  while (!complete) {
    const auto ticket = iree_notification_prepare_wait(&notification);
    status = loom_serve_text_load_advance(load, 0, &complete);
    if (complete) {
      iree_notification_cancel_wait(&notification);
    } else {
      IREE_ASSERT_OK(status);
      iree_notification_commit_wait(&notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  auto* partial = loom_serve_text_load_take_model(load);
  loom_serve_text_load_destroy(load);
  loom_serve_text_model_retire(partial, wake, &notification);
  EXPECT_TRUE(iree_notification_await(
      &notification,
      [](void* payload) {
        return loom_serve_text_model_is_retired(
            static_cast<loom_serve_text_model_t*>(payload));
      },
      partial, iree_infinite_timeout()));
  IREE_ASSERT_OK(loom_serve_text_model_destroy_retired(partial));
  iree_notification_deinitialize(&notification);
  const auto after = loom_serve_device_memory_statistics(device_);
  EXPECT_EQ(after.retained.reserved_bytes, before.retained.reserved_bytes);
  EXPECT_EQ(after.retained.committed_bytes, before.retained.committed_bytes);
  auto* row = loom_serve_text_model_row(model_, 0);
  const int32_t token = Oracle::Id('a');
  IREE_ASSERT_OK(loom_serve_text_row_prefill(row, 1, &token));
  Oracle reference;
  reference.Consume("a");
  EXPECT_EQ(loom_serve_text_row_token(row), reference.Predict());
}

TEST_P(SourceTextServiceTest, RetirementOwnsAcceptedCheckpointCapture) {
  IREE_ASSERT_OK(loom_serve_text_model_destroy(model_));
  model_ = nullptr;
  IREE_ASSERT_OK(loom_serve_device_group_destroy(group_));
  group_ = nullptr;
  const loom_serve_device_options_t device_options = {
      .uri = IREE_SV("amdgpu"), .backing = LOOM_SERVE_DEVICE_BACKING_ELASTIC};
  IREE_ASSERT_OK(loom_serve_device_group_create(host_, 1, &device_options,
                                                &group_, allocator_));
  device_ = loom_serve_device_group_at(group_, 0);
  auto options = ModelOptions();
  IREE_ASSERT_OK(
      loom_serve_text_model_create(device_, &options, &model_, allocator_));
  const std::string prompt = "abxabbaaxbabx";
  std::vector<int32_t> tokens;
  for (const char character : prompt) {
    tokens.push_back(Oracle::Id(character));
  }
  auto* row = loom_serve_text_model_row(model_, 0);
  IREE_ASSERT_OK(
      loom_serve_text_row_prefill(row, tokens.size(), tokens.data()));
  Oracle reference;
  reference.Consume(prompt);
  EXPECT_EQ(loom_serve_text_row_token(row), reference.Predict());
  loom_serve_text_checkpoint_t* checkpoint = nullptr;
  IREE_ASSERT_OK(loom_serve_text_row_try_pin(row, &checkpoint));
  ASSERT_NE(checkpoint, nullptr);
  loom_serve_execution_frontier_t frontier;
  IREE_ASSERT_OK(
      loom_serve_text_checkpoint_submit_suspend(checkpoint, &frontier));

  // Retire after acceptance, without completing the capture or resetting its
  // idle row. Model teardown owns the snapshot and its queued buffer views.
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  loom_serve_text_model_retire(
      model_,
      [](void* payload) {
        iree_notification_post(static_cast<iree_notification_t*>(payload),
                               IREE_ALL_WAITERS);
      },
      &notification);
  EXPECT_TRUE(iree_notification_await(
      &notification,
      [](void* payload) {
        return loom_serve_text_model_is_retired(
            static_cast<loom_serve_text_model_t*>(payload));
      },
      model_, iree_infinite_timeout()));
  IREE_ASSERT_OK(loom_serve_text_model_destroy_retired(model_));
  model_ = nullptr;
  iree_notification_deinitialize(&notification);
  const auto memory = loom_serve_device_memory_statistics(device_);
  EXPECT_EQ(memory.retained.reserved_bytes, 0u);
  EXPECT_EQ(memory.retained.committed_bytes, 0u);

  // Same host, queues and parameter pool remain available to the replacement.
  IREE_ASSERT_OK(
      loom_serve_text_model_create(device_, &options, &model_, allocator_));
  row = loom_serve_text_model_row(model_, 0);
  IREE_ASSERT_OK(
      loom_serve_text_row_prefill(row, tokens.size(), tokens.data()));
  EXPECT_EQ(loom_serve_text_row_token(row), reference.Predict());
}

TEST_P(SourceTextServiceTest,
       HttpRetainsRewindsAndRejectsWithoutLosingHistory) {
  Start();
  const std::string base = "abxa";
  const std::string first_response =
      Read(Send(Request(Message("user", base), 2), "X-Loom-Session: root\r\n"));
  const std::string first = Content(first_response);
  ASSERT_EQ(first, Oracle::Complete(base, 2));
  EXPECT_NE(first_response.find("\"finish_reason\":\"length\""),
            std::string::npos);
  const std::string pinned =
      Read(Send("", "X-Loom-Session: root\r\n", "/v1/checkpoints/base"));
  ASSERT_EQ(pinned.find("HTTP/1.1 201"), 0u) << pinned;
  const std::string history =
      Message("user", base) + "," + Message("assistant", first);
  std::vector<int> clients;
  const std::vector<std::string> suffixes = {"a", "bx", "xaba"};
  for (size_t i = 0; i < suffixes.size(); ++i) {
    clients.push_back(
        Send(Request(history + "," + Message("user", suffixes[i])),
             "X-Loom-Session: fork" + std::to_string(i) +
                 "\r\nX-Loom-Checkpoint: base\r\n"));
  }
  for (size_t i = 0; i < clients.size(); ++i) {
    const std::string response = Read(clients[i]);
    EXPECT_EQ(Content(response),
              Oracle::Complete(base + first + "#" + suffixes[i], 16));
    EXPECT_NE(response.find("\"cached_tokens\":5"), std::string::npos)
        << response;
  }
  const std::string bad =
      Read(Send(Request(history + "," + Message("user", "?")),
                "X-Loom-Session: root\r\n"));
  EXPECT_EQ(bad.find("HTTP/1.1 400"), 0u) << bad;
  const std::string resumed_request =
      Request(history + "," + Message("user", "bax"));
  const std::string resumed =
      Read(Send(resumed_request, "X-Loom-Session: root\r\n"));
  EXPECT_EQ(Content(resumed), Oracle::Complete(base + first + "#bax", 16));
  EXPECT_NE(resumed.find("\"cached_tokens\":5"), std::string::npos) << resumed;
  const std::string replay = Read(Send(resumed_request));
  EXPECT_EQ(Content(replay), Content(resumed));
  const std::string rewind =
      Read(Send(Request(history + "," + Message("user", "xx")),
                "X-Loom-Session: root\r\nX-Loom-Checkpoint: base\r\n"));
  EXPECT_EQ(Content(rewind), Oracle::Complete(base + first + "#xx", 16));
  const std::string deleted =
      Read(Send("", "", "/v1/checkpoints/base", "DELETE"));
  EXPECT_EQ(deleted.find("HTTP/1.1 200"), 0u) << deleted;

  const std::string terminal_response = Read(
      Send(Request(Message("user", "abx")), "X-Loom-Session: terminal\r\n"));
  const std::string terminal = Content(terminal_response);
  EXPECT_EQ(terminal, Oracle::Complete("abx", 16));
  EXPECT_NE(terminal_response.find("\"finish_reason\":\"stop\""),
            std::string::npos);
  const std::string next =
      Request(Message("user", "abx") + "," + Message("assistant", terminal) +
              "," + Message("user", "ba"));
  const std::string continued =
      Read(Send(next, "X-Loom-Session: terminal\r\n"));
  EXPECT_EQ(Content(continued), Oracle::Complete("abx" + terminal + "#ba", 16));
  EXPECT_NE(continued.find("\"cached_tokens\":11"), std::string::npos)
      << continued;
  EXPECT_EQ(Content(Read(Send(next))), Content(continued));
}

TEST_P(SourceTextServiceTest,
       SuspendedCheckpointSurvivesRestoreReleaseAndNameReuse) {
  IREE_ASSERT_OK(loom_serve_text_model_destroy(model_));
  model_ = nullptr;
  IREE_ASSERT_OK(loom_serve_device_group_destroy(group_));
  group_ = nullptr;
  const loom_serve_device_options_t device_options = {
      .uri = IREE_SV("amdgpu"), .backing = LOOM_SERVE_DEVICE_BACKING_ELASTIC};
  IREE_ASSERT_OK(loom_serve_device_group_create(host_, 1, &device_options,
                                                &group_, allocator_));
  device_ = loom_serve_device_group_at(group_, 0);
  auto options = ModelOptions();
  IREE_ASSERT_OK(
      loom_serve_text_model_create(device_, &options, &model_, allocator_));
  Start();
  const std::string base = "abxa";
  const std::string first = Content(Read(
      Send(Request(Message("user", base), 2), "X-Loom-Session: root\r\n")));
  ASSERT_EQ(first, Oracle::Complete(base, 2));
  const std::string history =
      Message("user", base) + "," + Message("assistant", first);
  const std::string pinned =
      Read(Send("", "X-Loom-Session: root\r\n", "/v1/checkpoints/base"));
  ASSERT_EQ(pinned.find("HTTP/1.1 201"), 0u) << pinned;
  EXPECT_NE(pinned.find("\"host_snapshot_bytes\":0}"), std::string::npos);
  std::string snapshot;
  for (int i = 0; i < 3; ++i) {
    SCOPED_TRACE(i);
    const std::string suspended =
        Read(Send("", "", "/v1/checkpoints/base/suspend"));
    ASSERT_EQ(suspended.find("HTTP/1.1 200"), 0u) << suspended;
    EXPECT_NE(suspended.find("\"host_snapshot_bytes\":"), std::string::npos);
    EXPECT_EQ(suspended.find("\"host_snapshot_bytes\":0}"), std::string::npos);
    if (i == 0) {
      snapshot = suspended;
    }
    EXPECT_EQ(suspended, snapshot);
    EXPECT_EQ(Read(Send("", "", "/v1/checkpoints/base/suspend")), snapshot);
    const std::string response =
        Read(Send(Request(history + "," + Message("user", "bax")),
                  "X-Loom-Session: fork\r\nX-Loom-Checkpoint: base\r\n"));
    EXPECT_EQ(Content(response), Oracle::Complete(base + first + "#bax", 16));
    EXPECT_NE(response.find("\"cached_tokens\":5"), std::string::npos);
  }
  // Release a cold entry, then reuse its public name and native slot. The new
  // pin has no stale host snapshot even though the preceding one was suspended.
  EXPECT_EQ(Read(Send("", "", "/v1/checkpoints/base/suspend")), snapshot);
  const std::string released =
      Read(Send("", "", "/v1/checkpoints/base", "DELETE"));
  ASSERT_EQ(released.find("HTTP/1.1 200"), 0u) << released;
  EXPECT_NE(released.find("\"pinned\":false,\"host_snapshot_bytes\":0}"),
            std::string::npos);
  EXPECT_EQ(Read(Send("", "X-Loom-Session: root\r\n", "/v1/checkpoints/base")),
            pinned);
}

TEST_P(SourceTextServiceTest, WideSourceEpochsRetainExactHttpHistory) {
  for (size_t capacity : {513u, 1024u, 4097u}) {
    SCOPED_TRACE(capacity);
    IREE_ASSERT_OK(loom_serve_text_model_destroy(model_));
    model_ = nullptr;
    const loom_serve_packing_shape_t shapes[] = {{32, 4}, {capacity, 4}};
    auto options = ModelOptions();
    options.prefill_capacity = capacity;
    options.context_capacity = capacity + 64;
    options.pool_capacity = 4 * ((options.context_capacity + 3) / 4 * 4);
    options.epoch_count = IREE_ARRAYSIZE(shapes);
    options.epoch_shapes = shapes;
    IREE_ASSERT_OK(
        loom_serve_text_model_create(device_, &options, &model_, allocator_));
    EXPECT_EQ(loom_serve_text_model_prefill_capacity(model_), capacity);
    std::vector<int32_t> tokens(capacity - 9);
    Oracle reference;
    for (size_t i = 0; i < tokens.size(); ++i) {
      tokens[i] = i % 3;
      reference.Consume(tokens[i]);
    }
    auto* full = loom_serve_text_model_row(model_, 0);
    auto* chunks = loom_serve_text_model_row(model_, 1);
    IREE_ASSERT_OK(
        loom_serve_text_row_prefill(full, tokens.size(), tokens.data()));
    for (size_t offset = 0; offset < tokens.size(); offset += 31) {
      IREE_ASSERT_OK(loom_serve_text_row_prefill(
          chunks, std::min<size_t>(31, tokens.size() - offset),
          tokens.data() + offset));
    }
    EXPECT_EQ(loom_serve_text_row_token(full), reference.Predict());
    EXPECT_EQ(loom_serve_text_row_token(chunks), reference.Predict());

    // The decode row and a fresh long prefill fill the same wide epoch.
    const int32_t pending = loom_serve_text_row_token(full);
    tokens.resize(capacity - 1, 1);
    const loom_serve_text_span_t spans[] = {
        {0, 1, &pending, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
        {2, tokens.size(), tokens.data(), LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
    };
    IREE_ASSERT_OK(
        loom_serve_text_model_epoch(model_, 1, IREE_ARRAYSIZE(spans), spans));
    reference.Consume(pending);
    EXPECT_EQ(loom_serve_text_row_token(full), reference.Predict());
    Oracle fresh;
    for (int32_t token : tokens) {
      fresh.Consume(token);
    }
    EXPECT_EQ(loom_serve_text_row_token(loom_serve_text_model_row(model_, 2)),
              fresh.Predict());
    for (size_t row = 0; row < 4; ++row) {
      IREE_ASSERT_OK(
          loom_serve_text_row_reset(loom_serve_text_model_row(model_, row)));
    }
    EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available,
              options.pool_capacity);
  }

  // Same server binary and source package, now with a 4097-token host shape.
  Start(4097);
  std::string prompt;
  for (size_t i = 0; i < 4088; ++i) {
    prompt.push_back("abx"[i % 3]);
  }
  const int retained =
      Send(Request(Message("user", prompt), 2), "X-Loom-Session: wide\r\n");
  const int other = Send(Request(Message("user", std::string(769, 'x'))));
  const std::string first = Content(Read(retained));
  ASSERT_EQ(first, Oracle::Complete(prompt, 2));
  EXPECT_EQ(Content(Read(other)), Oracle::Complete(std::string(769, 'x'), 16));
  const std::string request =
      Request(Message("user", prompt) + "," + Message("assistant", first) +
              "," + Message("user", "bax"));
  const std::string resumed = Read(Send(request, "X-Loom-Session: wide\r\n"));
  EXPECT_EQ(Content(resumed), Oracle::Complete(prompt + first + "#bax", 16));
  EXPECT_NE(resumed.find("\"cached_tokens\":4089"), std::string::npos)
      << resumed;
  EXPECT_EQ(Content(Read(Send(request))), Content(resumed));
}

TEST_P(SourceTextServiceTest,
       SourceRejectsUnsupportedConfigurationAndUsesWeights) {
  auto options = ModelOptions();
  options.enable_mtp = true;
  loom_serve_text_model_t* unsupported = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_text_model_create(device_, &options,
                                                     &unsupported, allocator_));
  EXPECT_EQ(unsupported, nullptr);
  options.enable_mtp = false;
  options.pool_capacity = 0;
  options.checkpoint_capacity = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_text_model_create(device_, &options,
                                                     &unsupported, allocator_));
  EXPECT_EQ(unsupported, nullptr);
  IREE_ASSERT_OK(loom_serve_text_model_destroy(model_));
  model_ = nullptr;
  ASSERT_NO_FATAL_FAILURE(WriteWeights(7));
  options = ModelOptions();
  IREE_ASSERT_OK(
      loom_serve_text_model_create(device_, &options, &model_, allocator_));
  Start();
  const std::string prompt = "axb";
  const std::string actual =
      Content(Read(Send(Request(Message("user", prompt)))));
  EXPECT_EQ(actual, Oracle::Complete(prompt, 16, 7));
  EXPECT_NE(actual, Oracle::Complete(prompt, 16, 2));
}

TEST_P(SourceTextServiceTest,
       CancelledQueuedRequestReturnsClaimAndSessionName) {
  struct Readiness {
    // CPU preparation has released its borrow of the request and model VM.
    std::atomic<bool> released{false};
    // Shared wake channel for transport and this explicit outer test owner.
    iree_notification_t* notification;
  } readiness{.notification = loom_serve_http_server_notification(server_)};
  auto options = ServiceOptions();
  loom_serve_text_service_t* value = nullptr;
  IREE_ASSERT_OK(loom_serve_text_service_create(model_, &options, nullptr,
                                                &value, allocator_));
  auto destroy = [](loom_serve_text_service_t* service) {
    loom_serve_text_service_destroy(service, IREE_STATUS_OK);
  };
  std::unique_ptr<loom_serve_text_service_t, decltype(destroy)> service(
      value, destroy);
  auto interface = loom_serve_text_service_interface(value);
  EXPECT_EQ(interface.capacity_domain,
            loom_serve_device_residency_cache(device_));
  interface.set_observer(interface.self,
                         {[](void* self) {
                            auto* readiness = static_cast<Readiness*>(self);
                            readiness->released.store(true);
                            iree_notification_post(readiness->notification,
                                                   IREE_ALL_WAITERS);
                          },
                          &readiness});
  const int client =
      Send(Request(Message("user", "abx")), "X-Loom-Session: reusable\r\n");
  auto* notification = loom_serve_http_server_notification(server_);
  loom_serve_http_connection_t* connection = nullptr;
  // accept may launch CPU preparation immediately. The next owner turn is
  // withheld until peer failure and CPU release, before device admission.
  while (!connection) {
    const auto ticket = iree_notification_prepare_wait(notification);
    const loom_serve_http_request_t* request = nullptr;
    connection = loom_serve_http_server_take_request(server_, &request);
    if (connection) {
      iree_notification_cancel_wait(notification);
      IREE_ASSERT_OK(interface.accept(interface.self, connection, request, 0));
    } else {
      iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  const auto queued = interface.snapshot(interface.self);
  EXPECT_EQ(queued.queued_requests, 1u);
  EXPECT_GE(queued.oldest_queued_age, 0);
  EXPECT_EQ(queued.oldest_active_age, 0);
  const linger reset = {1, 0};
  ASSERT_EQ(setsockopt(client, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
            0);
  ASSERT_EQ(close(client), 0);
  iree_notification_await(
      notification,
      [](void* self) {
        return loom_serve_http_connection_failed(
            static_cast<loom_serve_http_connection_t*>(self));
      },
      connection, iree_infinite_timeout());
  iree_notification_await(
      notification,
      [](void* self) { return static_cast<Readiness*>(self)->released.load(); },
      &readiness, iree_infinite_timeout());
  bool progress = false;
  IREE_ASSERT_OK(interface.advance(interface.self, 0, &progress));
  EXPECT_TRUE(progress);
  const auto cancelled = interface.snapshot(interface.self);
  EXPECT_EQ(cancelled.queued_requests, 0u);
  EXPECT_EQ(cancelled.oldest_queued_age, 0);
  EXPECT_EQ(cancelled.oldest_active_age, 0);
  EXPECT_FALSE(interface.is_pending(interface.self));
  EXPECT_FALSE(interface.is_active(interface.self));
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available, 256u);
  interface.set_observer(interface.self, {});
  const auto counters = *interface.telemetry.counters;
  EXPECT_EQ(counters.requests, 1u);
  EXPECT_EQ(counters.cancelled, 1u);
  EXPECT_EQ(counters.completed + counters.rejected + counters.failed, 0u);
  EXPECT_EQ(counters.input_tokens + counters.output_tokens + counters.batches,
            0u);
  service.reset();
  Start();
  EXPECT_EQ(Content(Read(Send(Request(Message("user", "abx")),
                              "X-Loom-Session: reusable\r\n"))),
            Oracle::Complete("abx", 16));
}

INSTANTIATE_TEST_SUITE_P(Workers, SourceTextServiceTest,
                         ::testing::Values(1, 4));

}  // namespace
