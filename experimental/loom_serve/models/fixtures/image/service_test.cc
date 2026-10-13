// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/image/service.h"

#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "experimental/loom_serve/image/model.h"
#include "experimental/loom_serve/image/output.h"
#include "experimental/loom_serve/models/fixtures/stalled_stderr.h"
#include "experimental/loom_serve/serving/model_registration.h"
#include "experimental/loom_serve/serving/router.h"
#include "experimental/loom_serve/testing/model.h"
#include "iree/async/proactor.h"
#include "iree/base/internal/base64.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/serve/transport/http/listener.h"

IREE_FLAG(string, model_source, "", "Deterministic image bootstrap source.");
IREE_FLAG(string, test_device, "amdgpu",
          "HAL device executing the source-defined image fixture.");

namespace {

class SourceImageServiceTest
    : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t>> {
 protected:
  static constexpr uint32_t kWidth = 7;
  static constexpr uint32_t kHeight = 5;
  static constexpr size_t kPixels = kWidth * kHeight;

  void SetUp() override {
    IREE_ASSERT_OK(
        loom_serve_host_create(std::get<0>(GetParam()), &host_, allocator_));
    const loom_serve_device_options_t device_options = {
        .uri = iree_make_cstring_view(FLAG_test_device)};
    IREE_ASSERT_OK(loom_serve_device_group_create(host_, 1, &device_options,
                                                  &group_, allocator_));
    device_ = loom_serve_device_group_at(group_, 0);
    const std::string source = FLAG_model_source;
    directory_ = source.substr(0, source.find_last_of('/'));
    auto options = loom_serve_http_server_options_default();
    IREE_ASSERT_OK(loom_serve_http_server_create(
        loom_serve_host_proactor_pool(host_), &options, &server_, allocator_));
    auto listener_options = loom_serve_http_listener_options_default();
    listener_options.address = IREE_SV("127.0.0.1:0");
    IREE_ASSERT_OK(loom_serve_http_listener_create(server_, &listener_options,
                                                   &listener_, allocator_));
  }

  void TearDown() override {
    if (service_.joinable()) {
      loom_serve_http_server_request_stop(server_);
      service_.join();
      IREE_EXPECT_OK(status_);
    }
    IREE_EXPECT_OK(loom_serve_http_listener_destroy(listener_));
    loom_serve_http_server_destroy(server_);
    IREE_EXPECT_OK(loom_serve_image_model_destroy(model_));
    IREE_EXPECT_OK(loom_serve_device_group_destroy(group_));
    loom_serve_host_release(host_);
  }

  void Start() {
    service_ = std::thread([this] {
      const loom_serve_image_generator_t generator =
          loom_serve_image_model_generator(model_);
      EXPECT_EQ(generator.capacity_domain,
                loom_serve_device_residency_cache(device_));
      const loom_serve_image_service_options_t options = {
          .name = loom_serve_image_model_name(model_),
          .width = kWidth,
          .height = kHeight,
          .pending_capacity = 8,
          .heartbeat_interval = 0,
      };
      loom_serve_image_service_t* service = nullptr;
      status_ = loom_serve_image_service_create(host_, generator, &options,
                                                nullptr, &service, allocator_);
      if (iree_status_is_ok(status_)) {
        const auto interface = loom_serve_image_service_interface(service);
        loom_serve_registry_t* registry = nullptr;
        IREE_ASSERT_OK(loom_serve_registry_create(1, &interface, 0, {},
                                                  &registry, allocator_));
        status_ = loom_serve_router_run(
            loom_serve_host_executor(host_), server_, registry,
            {.request_quota = options.pending_capacity,
             .holding_capacity =
                 loom_serve_http_server_connection_capacity(server_) / 2},
            allocator_);
        IREE_ASSERT_OK(loom_serve_registry_destroy(registry));
      }
      loom_serve_image_service_destroy(service);
    });
  }

  void CreateStandaloneModel() {
    loom_serve::testing::ModelDocument document;
    IREE_ASSERT_OK(document.Load(directory_ + "/model.json"));
    const loom_serve_image_model_options_t options = {
        .sources = document.value().sources,
        .checkpoint_directory = IREE_SV("."),
        .height = kHeight,
        .width = kWidth,
        .text_tokens = 32,
        .command_quantum = std::get<1>(GetParam())};
    IREE_ASSERT_OK(
        loom_serve_image_model_create(device_, &options, &model_, allocator_));
  }

  void StartManaged() {
    service_ = std::thread([this] {
      const loom_serve_model_registration_options_t options = {
          .device_group = group_, .heartbeat_interval = 0};
      loom_serve_registry_t* registry = nullptr;
      IREE_ASSERT_OK(loom_serve_registry_create(
          0, nullptr, 2, loom_serve_model_registration_factory(&options),
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
    return "{\"name\":\"" + name + "\",\"base_directory\":\"" +
           std::filesystem::absolute(source).string() + "\",\"admission\":\"" +
           admission +
           "\",\"definition\":{\"kind\":\"image\",\"prepare\":\"prepare.loom\","
           "\"control\":\"control.loom\",\"sources\":[\"config.loom\","
           "\"generate.loom\"],"
           "\"checkpoint\":\"" +
           std::filesystem::absolute(directory_).string() +
           "\",\"width\":7,\"height\":5,\"text_tokens\":32,\"command_"
           "quantum\":" +
           std::to_string(std::get<1>(GetParam())) + "}}";
  }

  std::string ModelStatus(const std::string& name) {
    const auto listing = Read(Send("GET", "/v1/status"));
    const size_t begin = listing.find("\"id\":\"" + name + "\"");
    if (begin == std::string::npos) {
      return {};
    }
    const size_t end = listing.find("{\"object\":\"model\"", begin);
    return listing.substr(begin, end == std::string::npos ? end : end - begin);
  }

  std::string WaitModel(const std::string& name, const std::string& marker) {
    for (;;) {
      const auto state = ModelStatus(name);
      if (marker.empty() ? state.empty()
                         : state.find(marker) != std::string::npos) {
        return state;
      }
    }
  }

  int Send(const std::string& method, const std::string& target,
           const std::string& body = "", const std::string& headers = "") {
    const int client = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    EXPECT_GE(client, 0);
    const auto* address = loom_serve_http_listener_address(listener_);
    EXPECT_EQ(
        connect(client, reinterpret_cast<const sockaddr*>(address->storage),
                address->length),
        0);
    const std::string request =
        method + " " + target +
        " HTTP/1.1\r\nHost: local\r\n"
        "Content-Type: application/json\r\nContent-Length: " +
        std::to_string(body.size()) + "\r\n" + headers + "\r\n" + body;
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
    char buffer[4096];
    ssize_t count = 0;
    while ((count = recv(client, buffer, sizeof(buffer), 0)) > 0) {
      response.append(buffer, count);
    }
    EXPECT_EQ(count, 0);
    EXPECT_EQ(close(client), 0);
    return response;
  }

  static std::array<float, kPixels * 3> Pixels(size_t tokens, uint64_t seed) {
    std::array<float, kPixels * 3> pixels;
    for (size_t i = 0; i < kPixels; ++i) {
      pixels[i] = seed & 1 ? 1.0f : -1.0f;
      pixels[kPixels + i] = tokens & 1 ? 1.0f : -1.0f;
      pixels[2 * kPixels + i] = i & 1 ? 1.0f : -1.0f;
    }
    return pixels;
  }

  std::string EncodedPixels(size_t tokens, uint64_t seed) {
    const auto pixels = Pixels(tokens, seed);
    iree_byte_span_t png;
    IREE_EXPECT_OK(loom_serve_image_encode_rgb_f32_png(
        kWidth, kHeight,
        iree_make_const_byte_span(pixels.data(), sizeof(pixels)), &png,
        allocator_));
    std::string encoded(iree_base64_encoded_size(png.data_length), '\0');
    size_t length = 0;
    IREE_EXPECT_OK(iree_base64_encode(
        iree_make_const_byte_span(png.data, png.data_length),
        iree_make_mutable_string_view(encoded.data(), encoded.size()),
        &length));
    encoded.resize(length);
    iree_allocator_free(allocator_, png.data);
    return encoded;
  }

  void CheckAbandonedImage(loom_serve_http_service_flags_t flags) {
    ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
    const loom_serve_image_service_options_t options = {
        .name = loom_serve_image_model_name(model_),
        .width = kWidth,
        .height = kHeight,
        .pending_capacity = 8,
        .heartbeat_interval = 0};
    std::unique_ptr<loom_serve_image_service_t,
                    decltype(&loom_serve_image_service_destroy)>
        service(nullptr, loom_serve_image_service_destroy);
    loom_serve_http_service_t interface = {};
    auto* notification = loom_serve_http_server_notification(server_);
    auto create_service = [&] {
      loom_serve_image_service_t* value = nullptr;
      IREE_ASSERT_OK(loom_serve_image_service_create(
          host_, loom_serve_image_model_generator(model_), &options, nullptr,
          &value, allocator_));
      service.reset(value);
      interface = loom_serve_image_service_interface(value);
      interface.set_observer(interface.self,
                             {[](void* self) {
                                iree_notification_post(
                                    static_cast<iree_notification_t*>(self),
                                    IREE_ALL_WAITERS);
                              },
                              notification});
    };
    ASSERT_NO_FATAL_FAILURE(create_service());
    auto accept = [&]() -> loom_serve_http_connection_t* {
      for (;;) {
        const auto ticket = iree_notification_prepare_wait(notification);
        const loom_serve_http_request_t* request = nullptr;
        auto* connection =
            loom_serve_http_server_take_request(server_, &request);
        if (connection) {
          iree_notification_cancel_wait(notification);
          IREE_EXPECT_OK(
              interface.accept(interface.self, connection, request, 0));
          return connection;
        }
        iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    };
    auto advance_until = [&](auto ready,
                             loom_serve_http_service_flags_t flags = 0) {
      while (!ready()) {
        const auto ticket = iree_notification_prepare_wait(notification);
        bool progress = false;
        IREE_ASSERT_OK(interface.advance(interface.self, flags, &progress));
        if (ready() || progress) {
          iree_notification_cancel_wait(notification);
        } else {
          iree_notification_commit_wait(notification, ticket,
                                        IREE_DURATION_ZERO,
                                        IREE_TIME_INFINITE_FUTURE);
        }
      }
    };

    const int client = Send("POST", "/v1/images/generations",
                            R"({"model":"fixture-image","prompt":"ab"})");
    auto* connection = accept();
    ASSERT_NO_FATAL_FAILURE(advance_until([&] {
      return std::string(interface.snapshot(interface.self).phase) ==
             "generating";
    }));
    if (!iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING)) {
      const linger reset = {1, 0};
      ASSERT_EQ(
          setsockopt(client, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)), 0);
      ASSERT_EQ(close(client), 0);
      iree_notification_await(
          notification,
          [](void* self) {
            return loom_serve_http_connection_failed(
                static_cast<loom_serve_http_connection_t*>(self));
          },
          connection, iree_infinite_timeout());
    }
    // Retire the accepted first part and its workspace before releasing the
    // pin. Split execution must skip the second kernel and output feedback;
    // neither path needs a PNG/base64 borrower for an absent consumer.
    ASSERT_NO_FATAL_FAILURE(advance_until(
        [&] { return interface.telemetry.counters->cancelled == 1; }, flags));
    EXPECT_EQ(interface.telemetry.counters->batches, 1u);
    EXPECT_STREQ(interface.snapshot(interface.self).phase, "idle");
    EXPECT_FALSE(interface.is_pending(interface.self));
    EXPECT_EQ(interface.telemetry.counters->cancelled, 1u);
    ASSERT_NO_FATAL_FAILURE(
        advance_until([&] { return !interface.is_pending(interface.self); }));
    if (iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING)) {
      EXPECT_TRUE(Read(client).empty());
      interface.set_observer(interface.self, {});
      service.reset();
      ASSERT_NO_FATAL_FAILURE(create_service());
    }

    // Reuse the actual generator after retirement, including its prepared
    // input and output storage, and compare the complete encoded response.
    const int next =
        Send("POST", "/v1/images/generations",
             R"({"model":"fixture-image","prompt":"aba","seed":1})");
    accept();
    ASSERT_NO_FATAL_FAILURE(advance_until(
        [&] { return interface.telemetry.counters->completed == 1; }));
    EXPECT_NE(Read(next).find("\"b64_json\":\"" + EncodedPixels(3, 1) + "\""),
              std::string::npos);
    EXPECT_EQ(interface.telemetry.counters->failed, 0u);
    interface.set_observer(interface.self, {});
  }

  // Allocator for model, transport and expected encoded output.
  const iree_allocator_t allocator_ = iree_allocator_system();
  // Shared CPU/JIT resources outlive every model and service task.
  loom_serve_host_t* host_ = nullptr;
  // Application group enclosing all model and device resources.
  loom_serve_device_group_t* group_ = nullptr;
  // Borrowed device record within the application group.
  loom_serve_device_t* device_ = nullptr;
  // Source and tokenizer package borrowed by cold and HTTP registration.
  std::string directory_;
  // Real source-JIT model with two kernels and reflected private workspace.
  loom_serve_image_model_t* model_ = nullptr;
  // Production loopback carrier and request claim owner.
  loom_serve_http_server_t* server_ = nullptr;
  // Independent binding borrowing the HTTP owner and shared host pool.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Outer application join; model work uses the shared task executor.
  std::thread service_;
  // Terminal service result read only after joining the outer thread.
  iree_status_t status_ = iree_ok_status();
};

TEST_P(SourceImageServiceTest, SourcePixelsAndHttpPngPreserveRequestIdentity) {
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  loom_serve_image_prepared_t prepared;
  IREE_ASSERT_OK(loom_serve_image_model_prepare_request(model_, IREE_SV("x"), 0,
                                                        1.0f, &prepared));
  EXPECT_EQ(prepared.stage, 0u);
  EXPECT_EQ(prepared.tag, 32);
  EXPECT_EQ(prepared.input_bytes, 16u);
  loom_serve_image_model_discard_prepared_request(model_);
  // Verify source arithmetic independently of the image encoder.
  for (uint64_t seed : {0u, 1u}) {
    iree_const_byte_span_t rgb;
    IREE_ASSERT_OK(loom_serve_image_model_generate(model_, IREE_SV("aba"), seed,
                                                   1.0f, &prepared, &rgb));
    EXPECT_EQ(prepared.stage, 0u);
    EXPECT_EQ(prepared.tag, 32);
    EXPECT_EQ(prepared.input_bytes, 16u);
    const auto expected = Pixels(3, seed);
    ASSERT_EQ(rgb.data_length, sizeof(expected));
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(iree_unaligned_load_le_f32(rgb.data + 4 * i), expected[i]);
    }
  }
  Start();
  EXPECT_NE(
      Read(Send("GET", "/v1/models"))
          .find(
              R"("input_modalities":["text"],"output_modalities":["image"],"loom":{"kind":"image","endpoint":"/v1/images/generations","tokenize":true}})"),
      std::string::npos);
  EXPECT_NE(
      Read(Send("GET", "/v1/status"))
          .find(
              R"("status":{"phase":"idle","queued_requests":0,"active_requests":0,"oldest_queued_age_seconds":0.000000000,"oldest_active_age_seconds":0.000000000,"admission":"open","held_requests":0,"tokenizer_requests":0})"),
      std::string::npos);
  EXPECT_NE(Read(Send("GET", "/readyz")).find(R"({"status":"ready"})"),
            std::string::npos);
  EXPECT_NE(Read(Send("POST", "/tokenize",
                      R"({"model":"fixture-image","content":"aba"})"))
                .find(R"({"tokens":[0,1,0]})"),
            std::string::npos);
  EXPECT_NE(Read(Send("POST", "/detokenize",
                      R"({"model":"fixture-image","tokens":[1,0,1]})"))
                .find(R"({"content":"bab"})"),
            std::string::npos);
  // A rejected request must leave the same listener/model usable.
  EXPECT_EQ(Read(Send("POST", "/v1/images/generations",
                      R"({"model":"fixture-image","prompt":"ab","n":2})"))
                .find("HTTP/1.1 400"),
            0u);
  // This rejection comes from the source VM after ordinary HTTP parsing.
  const std::string too_long =
      Read(Send("POST", "/v1/images/generations",
                "{\"model\":\"fixture-image\",\"prompt\":\"" +
                    std::string(33, 'a') + "\"}"));
  ASSERT_EQ(too_long.find("HTTP/1.1 400"), 0u);
  EXPECT_NE(too_long.find("prompt exceeds fixture text capacity"),
            std::string::npos);
  const char* prompts[] = {"ab", "aba", "b", "abba"};
  std::vector<int> clients;
  for (size_t i = 0; i < IREE_ARRAYSIZE(prompts); ++i) {
    clients.push_back(
        Send("POST", "/v1/images/generations",
             std::string("{\"model\":\"fixture-image\",\"prompt\":\"") +
                 prompts[i] + "\",\"seed\":" + std::to_string(i) + "}"));
  }
  for (size_t i = 0; i < clients.size(); ++i) {
    SCOPED_TRACE(i);
    const std::string encoded =
        EncodedPixels(std::string(prompts[i]).size(), i);
    const std::string response = Read(clients[i]);
    EXPECT_EQ(response.find("HTTP/1.1 200"), 0u);
    EXPECT_NE(response.find("\"b64_json\":\"" + encoded + "\""),
              std::string::npos);
  }
  EXPECT_NE(Read(Send("GET", "/healthz")).find("\"status\":\"ready\""),
            std::string::npos);
}

TEST_P(SourceImageServiceTest, IndependentLogReadersCannotBlockColdImageWork) {
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  loom_serve::testing::StalledStderr stalled;
  ASSERT_NO_FATAL_FAILURE(stalled.Begin());
  Start();
  auto poll = [&](const std::string& cursor) {
    return Read(
        Send("GET", "/logz", "",
             "X-Loom-Model: fixture-image\r\n" +
                 (cursor.empty() ? std::string()
                                 : "X-Loom-Cursor: " + cursor + "\r\n")));
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
  ASSERT_NE(initial.find(R"("event":"stream_begin","sequence":0)"),
            std::string::npos);
  const std::string slow = cursor_of(initial);
  std::string fast = slow;
  std::string latest;
  std::string events;
  auto drain = [&] {
    for (;;) {
      latest = poll(fast);
      EXPECT_EQ(latest.find(R"("event":"gap")"), std::string::npos);
      events += latest;
      const std::string next = cursor_of(latest);
      if (next == fast) {
        break;
      }
      fast = next;
    }
  };
  for (size_t wave = 0; wave < 10; ++wave) {
    std::vector<int> clients;
    for (size_t i = 0; i < 4; ++i) {
      const std::string prompt = i & 1 ? "aba" : "ab";
      clients.push_back(Send("POST", "/v1/images/generations",
                             "{\"model\":\"fixture-image\",\"prompt\":\"" +
                                 prompt + "\",\"seed\":" + std::to_string(i) +
                                 "}"));
    }
    for (size_t i = 0; i < clients.size(); ++i) {
      const std::string response = Read(clients[i]);
      EXPECT_EQ(response.find("HTTP/1.1 200"), 0u);
      EXPECT_NE(response.find("\"b64_json\":\"" +
                              EncodedPixels(i & 1 ? 3 : 2, i) + "\""),
                std::string::npos);
    }
    drain();
  }
  EXPECT_EQ(Read(Send("POST", "/v1/images/generations",
                      R"({"model":"fixture-image","prompt":"aba","n":2})"))
                .find("HTTP/1.1 400"),
            0u);
  EXPECT_EQ(Read(Send("POST", "/v1/images/generations",
                      "{\"model\":\"fixture-image\",\"prompt\":\"" +
                          std::string(33, 'a') + "\"}"))
                .find("HTTP/1.1 400"),
            0u);
  drain();
  const std::string totals =
      std::string(
          R"("requests":42,"completed":40,"rejected":2,"cancelled":0,"failed":0,"input_tokens":0,"output_tokens":0,"batches":)") +
      (std::get<1>(GetParam()) ? "80" : "40");
  EXPECT_NE(latest.find(totals), std::string::npos);
  EXPECT_NE(events.find(R"("generated_images":40,"generated_pixels":1400)"),
            std::string::npos);
  EXPECT_NE(events.find(R"("event":"image_request_stats")"), std::string::npos);
  EXPECT_NE(events.find(R"("event":"image_prepared")"), std::string::npos);
  EXPECT_NE(events.find(R"("stage":0,"tag":32,"input_bytes":16)"),
            std::string::npos);
  EXPECT_NE(events.find(R"("event":"image_generated")"), std::string::npos);
  EXPECT_NE(events.find(R"("event":"weight_activity")"), std::string::npos);
  EXPECT_NE(
      events.find(
          R"("activations":0,"deactivations":0,"read_bytes":0,"prepared_bytes":0,"groups":0)"),
      std::string::npos);
  for (const std::string event : {"image_prepared", "image_generated"}) {
    const std::string key = "\"event\":\"" + event + "\"";
    size_t count = 0;
    for (size_t offset = events.find(key); offset != std::string::npos;
         offset = events.find(key, offset + key.size())) {
      ++count;
    }
    EXPECT_EQ(count, 40u) << event;
  }
  EXPECT_EQ(events.find("aba"), std::string::npos);
  const std::string resumed = poll(slow);
  EXPECT_NE(resumed.find(R"({"event":"gap","from":1,)"), std::string::npos);
  EXPECT_NE(resumed.find(totals), std::string::npos);
  EXPECT_NE(poll("").find(R"("partial":true)"), std::string::npos);
  EXPECT_EQ(cursor_of(poll(fast)), fast);
  loom_serve_http_server_request_stop(server_);
  service_.join();
  IREE_EXPECT_OK(status_);
}

TEST_P(SourceImageServiceTest, DynamicImageAdmissionRetirementAndReuse) {
  StartManaged();
  EXPECT_NE(Read(Send("GET", "/v1/models")).find(R"("data":[])"),
            std::string::npos);
  loom_serve::testing::StalledStderr stalled;
  ASSERT_NO_FATAL_FAILURE(stalled.Begin());
  EXPECT_EQ(Read(Send("POST", "/v1/models/register",
                      Declaration("pixels", directory_, "drained")))
                .find("HTTP/1.1 202"),
            0u);
  const auto initial = ModelStatus("pixels");
  EXPECT_NE(initial.find(R"("registration":"registered")"), std::string::npos);
  EXPECT_NE(initial.find(R"("admission":"drained")"), std::string::npos);
  const auto initial_log =
      Read(Send("GET", "/logz", "", "X-Loom-Model: pixels\r\n"));
  EXPECT_NE(initial_log.find(R"("event":"stream_begin","sequence":0)"),
            std::string::npos);
  const int held = Send("POST", "/v1/images/generations",
                        R"({"model":"pixels","prompt":"aba","seed":1})");
  const int tokens =
      Send("POST", "/tokenize", R"({"model":"pixels","content":"aba"})");
  WaitModel("pixels", R"("held_requests":2)");
  EXPECT_EQ(Read(Send("POST", "/v1/models/undrain", R"({"model":"pixels"})"))
                .find("HTTP/1.1 200"),
            0u);
  const auto response = Read(held);
  const auto tokenizer_response = Read(tokens);
  ASSERT_EQ(response.find("HTTP/1.1 200"), 0u) << response;
  EXPECT_NE(response.find("\"b64_json\":\"" + EncodedPixels(3, 1) + "\""),
            std::string::npos);
  EXPECT_NE(tokenizer_response.find(R"({"tokens":[0,1,0]})"),
            std::string::npos);
  const auto ready = ModelStatus("pixels");
  EXPECT_NE(ready.find(R"("registration":"ready")"), std::string::npos);
  EXPECT_NE(ready.find(R"("kernels":2)"), std::string::npos);
  const auto log = Read(Send("GET", "/logz", "", "X-Loom-Model: pixels\r\n"));
  EXPECT_NE(log.find(R"("event":"stream_begin","sequence":0)"),
            std::string::npos);
  EXPECT_NE(log.find(R"("event":"image_generated")"), std::string::npos);

  // The accepted registration belongs to the registry, not its admin peer.
  const int detached =
      Send("POST", "/v1/models/register", Declaration("peer", directory_));
  char first_byte;
  ASSERT_EQ(recv(detached, &first_byte, 1, MSG_PEEK), 1);
  EXPECT_EQ(close(detached), 0);
  const int peer = Send("POST", "/v1/images/generations",
                        R"({"model":"peer","prompt":"ab","seed":0})");
  EXPECT_EQ(Read(Send("POST", "/v1/models/drain", R"({"model":"pixels"})"))
                .find("HTTP/1.1 200"),
            0u);
  const int removed = Send("POST", "/v1/images/generations",
                           R"({"model":"pixels","prompt":"a"})");
  WaitModel("pixels", R"("held_requests":1)");
  EXPECT_EQ(Read(Send("POST", "/v1/models/unregister",
                      R"({"model":"pixels","mode":"drain"})"))
                .find("HTTP/1.1 202"),
            0u);
  EXPECT_EQ(Read(removed).find("HTTP/1.1 410"), 0u);
  WaitModel("pixels", {});
  EXPECT_NE(Read(peer).find("\"b64_json\":\"" + EncodedPixels(2, 0) + "\""),
            std::string::npos);

  // Reusing the route neither reuses its counters nor invalidates its peer.
  EXPECT_EQ(Read(Send("POST", "/v1/models/register",
                      Declaration("pixels", directory_)))
                .find("HTTP/1.1 202"),
            0u);
  EXPECT_NE(Read(Send("POST", "/v1/images/generations",
                      R"({"model":"pixels","prompt":"ab","seed":1})"))
                .find("\"b64_json\":\"" + EncodedPixels(2, 1) + "\""),
            std::string::npos);
  const auto metrics = Read(Send("GET", "/metrics"));
  EXPECT_NE(metrics.find("loom_serve_server_completed_total 3\n"),
            std::string::npos);
  EXPECT_NE(metrics.find("loom_serve_model_completed_total{model=\"pixels\","
                         "kind=\"image\",incarnation=\"3\"} 1\n"),
            std::string::npos);
  EXPECT_EQ(Read(Send("POST", "/v1/models/unregister",
                      R"({"model":"pixels","mode":"abort"})"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModel("pixels", {});
}

TEST_P(SourceImageServiceTest, FailedImageSourceLeavesPeerUsable) {
  const auto package = iree::testing::MakeTempFilePath("failed_image_source");
  ASSERT_TRUE(std::filesystem::create_directory(package));
  for (const char* file :
       {"config.loom", "control.loom", "prepare.loom", "model.json"}) {
    ASSERT_TRUE(std::filesystem::copy_file(directory_ + "/" + file,
                                           package + "/" + file));
  }
  {
    std::ofstream source(package + "/generate.loom");
    source << "this is not a source program\n";
    ASSERT_TRUE(source.good());
  }
  StartManaged();
  loom_serve::testing::StalledStderr stalled;
  ASSERT_NO_FATAL_FAILURE(stalled.Begin());
  EXPECT_EQ(
      Read(Send("POST", "/v1/models/register", Declaration("peer", directory_)))
          .find("HTTP/1.1 202"),
      0u);
  EXPECT_EQ(Read(Send("POST", "/v1/models/register",
                      Declaration("broken", package, "drained")))
                .find("HTTP/1.1 202"),
            0u);
  const int held = Send("POST", "/v1/images/generations",
                        R"({"model":"broken","prompt":"a"})");
  WaitModel("broken", R"("held_requests":1)");
  EXPECT_EQ(Read(Send("POST", "/v1/models/undrain", R"({"model":"broken"})"))
                .find("HTTP/1.1 200"),
            0u);
  EXPECT_EQ(Read(held).find("HTTP/1.1 503"), 0u);
  const auto failed = WaitModel("broken", R"("registration":"failed")");
  EXPECT_NE(failed.find("generate.loom"), std::string::npos);
  EXPECT_NE(failed.find(R"("diagnostics":[{)"), std::string::npos);
  EXPECT_NE(Read(Send("POST", "/v1/images/generations",
                      R"({"model":"peer","prompt":"aba","seed":1})"))
                .find("\"b64_json\":\"" + EncodedPixels(3, 1) + "\""),
            std::string::npos);
  EXPECT_EQ(Read(Send("POST", "/v1/models/unregister", R"({"model":"broken"})"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModel("broken", {});
}

TEST_P(SourceImageServiceTest, CancelledDeviceImageSkipsEncoding) {
  CheckAbandonedImage(0);
}

TEST_P(SourceImageServiceTest, StoppingDeviceImageSkipsEncoding) {
  CheckAbandonedImage(LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING);
}

TEST_P(SourceImageServiceTest, FailedPartReadinessPreservesActualRetirement) {
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  loom_serve_image_prepared_t prepared;
  IREE_ASSERT_OK(loom_serve_image_model_prepare_request(model_, IREE_SV("aba"),
                                                        1, 1.0f, &prepared));
  const auto generator = loom_serve_image_model_generator(model_);
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(generator.try_admit(generator.self, &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  loom_serve_execution_frontier_t frontier;
  bool complete = false;
  IREE_ASSERT_OK(generator.submit(generator.self, 0, &frontier, &complete));
  EXPECT_EQ(complete, std::get<1>(GetParam()) == 0);
  // Failure readiness does not authorize releasing the workspace or upload
  // payload while the native queue still owns its accepted first part.
  iree_hal_semaphore_fail(
      frontier.semaphores[0],
      iree_make_status(IREE_STATUS_ABORTED, "failed image part"));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        loom_serve_image_model_destroy(model_));
  model_ = nullptr;
}

INSTANTIATE_TEST_SUITE_P(WorkersAndQuanta, SourceImageServiceTest,
                         ::testing::Combine(::testing::Values(1u, 4u),
                                            ::testing::Values(0u, 1u)));

}  // namespace
