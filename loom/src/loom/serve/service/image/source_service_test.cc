// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "iree/async/proactor.h"
#include "iree/base/internal/base64.h"
#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/serve/runtime/program/package.h"
#include "loom/serve/runtime/program/variant_pool.h"
#include "loom/serve/server/model_registration.h"
#include "loom/serve/server/router.h"
#include "loom/serve/service/image/model.h"
#include "loom/serve/service/image/output.h"
#include "loom/serve/service/image/service.h"
#include "loom/serve/testing/http_harness.h"
#include "loom/serve/testing/model.h"
#include "loom/serve/testing/package.h"
#include "loom/serve/testing/service_application.h"
#include "loom/serve/testing/stalled_stderr.h"
#include "loom/serve/transport/http/listener.h"

namespace {

class SourceImageServiceTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  static constexpr uint32_t kWidth = 7;
  static constexpr uint32_t kHeight = 5;
  static constexpr size_t kPixels = kWidth * kHeight;

  void SetUp() override {
    const char* source_path = std::getenv("LOOM_SERVE_TEST_MODEL_SOURCE");
    ASSERT_NE(source_path, nullptr);
    const auto application_options =
        loom_serve_test_application_options_default();
    loom_serve_test_application_t* application = nullptr;
    IREE_ASSERT_OK(loom_serve_test_application_acquire(
        &application_options, host_allocator_, &application));
    host_ = loom_serve_test_application_host(application);
    group_ = loom_serve_test_application_device_group(application);
    device_ = loom_serve_test_application_device_at(application, 0);
    const std::string source = source_path;
    directory_ = source.substr(0, source.find_last_of('/'));
    auto options = loom_serve_http_server_options_default();
    IREE_ASSERT_OK(
        loom_serve_http_server_allocate(loom_serve_host_proactor_pool(host_),
                                        &options, host_allocator_, &server_));
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
    IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
    loom_serve_http_server_free(server_);
    IREE_EXPECT_OK(loom_serve_image_model_free(model_));
  }

  void Start() {
    http_.Start([this] {
      const loom_serve_image_generator_t generator =
          loom_serve_image_model_generator(model_);
      EXPECT_EQ(generator.capacity_domain,
                loom_serve_device_residency_cache(device_));
      const loom_serve_image_service_options_t options = {
          .name = loom_serve_image_model_name(model_),
          .width = kWidth,
          .height = kHeight,
          .pending_capacity = 8,
          .variant_capacity = 16,
          .heartbeat_interval = 0,
      };
      loom_serve_image_service_t* service = nullptr;
      iree_status_t status = loom_serve_image_service_allocate(
          host_, generator, &options, nullptr, host_allocator_, &service);
      if (iree_status_is_ok(status)) {
        const auto interface = loom_serve_image_service_interface(service);
        loom_serve_registry_t* registry = nullptr;
        status = loom_serve_registry_allocate(1, &interface, 0, {},
                                              host_allocator_, &registry);
        if (iree_status_is_ok(status)) {
          status = loom_serve_router_run(
              loom_serve_host_executor(host_), server_, registry,
              {
                  .request_quota = options.pending_capacity,
                  .holding_capacity =
                      loom_serve_http_server_connection_capacity(server_) / 2,
                  .resources = {.device_group = group_},
              },
              host_allocator_);
        }
        status = iree_status_join(status, loom_serve_registry_free(registry));
      }
      loom_serve_image_service_free(service);
      return status;
    });
  }

  void CreateStandaloneModel(
      loom_serve_variant_cache_options_t cache = {2, 16 * 1024 * 1024},
      iree_allocator_t allocator = iree_allocator_system()) {
    loom_serve::testing::ModelDocument document;
    IREE_ASSERT_OK(document.Load(directory_ + "/model.json"));
    const loom_serve_image_model_options_t options = {
        .sources = document.value().sources,
        .checkpoint_directory = IREE_SV("."),
        .height = kHeight,
        .width = kWidth,
        .text_tokens = 32,
        .command_quantum = GetParam(),
        .specialization_cache = cache,
    };
    IREE_ASSERT_OK(
        loom_serve_image_model_allocate(device_, &options, allocator, &model_));
  }

  void StartManaged() {
    http_.Start([this]() -> iree_status_t {
      const loom_serve_model_registration_options_t options = {
          .device_group = group_,
          .heartbeat_interval = 0,
      };
      loom_serve_registry_t* registry = nullptr;
      IREE_RETURN_IF_ERROR(loom_serve_registry_allocate(
          0, nullptr, 2, loom_serve_model_registration_factory(&options),
          host_allocator_, &registry));
      iree_status_t status = loom_serve_router_run(
          loom_serve_host_executor(host_), server_, registry,
          {
              .request_quota = 16,
              .holding_capacity = 8,
              .resources = {.device_group = group_},
          },
          host_allocator_);
      return iree_status_join(status, loom_serve_registry_free(registry));
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
           std::to_string(GetParam()) + "}}";
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
    while (!http_.failed() && !HasFatalFailure()) {
      const auto state = ModelStatus(name);
      if (marker.empty() ? state.empty()
                         : state.find(marker) != std::string::npos) {
        return state;
      }
    }
    return {};
  }

  int Send(const std::string& method, const std::string& target,
           const std::string& body = "", const std::string& headers = "") {
    if (http_.failed() || HasFatalFailure()) {
      return -1;
    }
    const std::string request =
        method + " " + target +
        " HTTP/1.1\r\nHost: local\r\n"
        "Content-Type: application/json\r\nContent-Length: " +
        std::to_string(body.size()) + "\r\n" + headers + "\r\n" + body;
    int client = -1;
    [&] { IREE_ASSERT_OK(http_.Send(request, &client)); }();
    return client;
  }

  std::string Read(int client, bool allow_empty = false) {
    std::string response;
    if (http_.failed() || HasFatalFailure()) {
      if (client >= 0) {
        EXPECT_EQ(close(client), 0);
      }
      return response;
    }
    [&] { IREE_ASSERT_OK(http_.Read(client, allow_empty, &response)); }();
    return response;
  }

  static std::vector<float> Pixels(size_t tokens, uint64_t seed,
                                   size_t count = kPixels) {
    std::vector<float> pixels(count * 3);
    for (size_t i = 0; i < count; ++i) {
      pixels[i] = seed & 1 ? 1.0f : -1.0f;
      pixels[count + i] = tokens & 1 ? 1.0f : -1.0f;
      pixels[2 * count + i] = i & 1 ? 1.0f : -1.0f;
    }
    return pixels;
  }

  std::string EncodedPixels(size_t tokens, uint64_t seed,
                            uint32_t width = kWidth,
                            uint32_t height = kHeight) {
    const auto pixels = Pixels(tokens, seed, (size_t)width * height);
    iree_byte_span_t png;
    IREE_EXPECT_OK(loom_serve_image_encode_rgb_f32_png(
        width, height,
        iree_make_const_byte_span(pixels.data(), pixels.size() * sizeof(float)),
        host_allocator_, &png));
    std::string encoded(iree_base64_encoded_size(png.data_length), '\0');
    size_t length = 0;
    IREE_EXPECT_OK(iree_base64_encode(
        iree_make_const_byte_span(png.data, png.data_length),
        iree_make_mutable_string_view(encoded.data(), encoded.size()),
        &length));
    encoded.resize(length);
    iree_allocator_free(host_allocator_, png.data);
    return encoded;
  }

  static uint64_t Number(const std::string& response, const char* field,
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
  }

  std::string SelectVariant(uint64_t variant, uint64_t expected,
                            const char* name = "fixture-image") {
    return Read(Send("POST", "/v1/models/variants/select",
                     "{\"model\":\"" + std::string(name) +
                         "\",\"variant\":" + std::to_string(variant) +
                         ",\"expected\":" + std::to_string(expected) + "}"));
  }

  void CheckAbandonedImage(loom_serve_http_service_flags_t flags) {
    ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
    const loom_serve_image_service_options_t options = {
        .name = loom_serve_image_model_name(model_),
        .width = kWidth,
        .height = kHeight,
        .pending_capacity = 8,
        .heartbeat_interval = 0,
    };
    std::unique_ptr<loom_serve_image_service_t,
                    decltype(&loom_serve_image_service_free)>
        service(nullptr, loom_serve_image_service_free);
    loom_serve_http_service_t interface = {};
    auto* notification = loom_serve_http_server_notification(server_);
    auto create_service = [&] {
      loom_serve_image_service_t* value = nullptr;
      IREE_ASSERT_OK(loom_serve_image_service_allocate(
          host_, loom_serve_image_model_generator(model_), &options, nullptr,
          host_allocator_, &value));
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
    auto accept = [&](int client, loom_serve_http_connection_t** connection) {
      const loom_serve_http_request_t* request = nullptr;
      IREE_ASSERT_OK(http_.TakeRequest(client, connection, &request));
      IREE_ASSERT_OK(interface.accept(interface.self, *connection, request, 0));
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
    loom_serve_http_connection_t* connection = nullptr;
    ASSERT_NO_FATAL_FAILURE(accept(client, &connection));
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
      EXPECT_TRUE(Read(client, /*allow_empty=*/true).empty());
      interface.set_observer(interface.self, {});
      service.reset();
      ASSERT_NO_FATAL_FAILURE(create_service());
    }

    // Reuse the actual generator after retirement, including its prepared
    // input and output storage, and compare the complete encoded response.
    const int next =
        Send("POST", "/v1/images/generations",
             R"({"model":"fixture-image","prompt":"aba","seed":1})");
    ASSERT_NO_FATAL_FAILURE(accept(next, &connection));
    ASSERT_NO_FATAL_FAILURE(advance_until(
        [&] { return interface.telemetry.counters->completed == 1; }));
    EXPECT_NE(Read(next).find("\"b64_json\":\"" + EncodedPixels(3, 1) + "\""),
              std::string::npos);
    EXPECT_EQ(interface.telemetry.counters->failed, 0u);
    interface.set_observer(interface.self, {});
  }

  // Allocator for model, transport and expected encoded output.
  const iree_allocator_t host_allocator_ = iree_allocator_system();
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
  // HTTP readiness and optional outer owner, joined before model destruction.
  loom_serve::testing::HttpHarness http_;
};

TEST_P(SourceImageServiceTest,
       UploadedExecutionChangesPartsAndPixelsWithoutRelinking) {
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  IREE_ASSERT_OK(loom_serve_image_model_activate(model_));
  const auto weights = loom_serve_image_model_weight_activity(model_);
  Start();
  const auto inventory = Read(Send("GET", "/v1/models/variants", {},
                                   "X-Loom-Model: fixture-image\r\n"));
  ASSERT_EQ(inventory.find("HTTP/1.1 200"), 0u) << inventory;
  const uint64_t original = Number(inventory, "selected");
  const std::string request =
      R"({"model":"fixture-image","prompt":"aba","seed":1})";
  auto generate = [&](uint64_t seed) {
    const auto response = Read(Send("POST", "/v1/images/generations", request));
    EXPECT_EQ(response.find("HTTP/1.1 200"), 0u) << response;
    EXPECT_NE(response.find("\"b64_json\":\"" + EncodedPixels(3, seed) + "\""),
              std::string::npos)
        << response;
  };
  generate(1);
  std::ifstream input(directory_ + "/generate.loom");
  ASSERT_TRUE(input.good());
  std::string source((std::istreambuf_iterator<char>(input)), {});
  const std::string red = "scf.select %odd_seed, %high, %low";
  const auto red_position = source.find(red);
  ASSERT_NE(red_position, std::string::npos);
  source.replace(red_position, red.size(), "scf.select %odd_seed, %low, %high");
  const std::string launch = "  kernel.launch @pixels(%colors, %rgb)";
  const auto launch_position = source.find(launch);
  ASSERT_NE(launch_position, std::string::npos);
  // A second producer is a third barrier wave under command_quantum=1. The
  // replacement table must grow, while the retained source VM sees new counts.
  source.insert(
      launch_position,
      "  kernel.launch @colors(%request, %colors) : (buffer, buffer)\n");
  const std::string allocation = "index.constant 8 : offset";
  const auto allocation_position = source.find(allocation);
  ASSERT_NE(allocation_position, std::string::npos);
  source.replace(allocation_position, allocation.size(),
                 "index.constant 4160 : offset");
  loom_serve::testing::SourcePackage package;
  package.Add("model/generate.loom", source);
  IREE_ASSERT_OK(
      package.AddFile("model/config.loom", directory_ + "/config.loom"));
  int client = -1;
  IREE_ASSERT_OK(http_.Send(
      package.Request(
          "/v1/models/variants",
          "{\"model\":\"fixture-image\",\"base\":" + std::to_string(original) +
              ",\"sources\":[\"config.loom\",\"generate.loom\"],\"wait\":"
              "true}"),
      &client));
  const auto staged = Read(client);
  ASSERT_EQ(staged.find("HTTP/1.1 200"), 0u) << staged;
  const uint64_t candidate = Number(staged, "id", "variant");
  EXPECT_EQ(Number(staged, "selected"), original);
  generate(1);
  const auto selected = SelectVariant(candidate, original);
  ASSERT_EQ(selected.find("HTTP/1.1 200"), 0u) << selected;
  generate(0);
  EXPECT_EQ(SelectVariant(original, original).find("HTTP/1.1 409"), 0u);
  ASSERT_EQ(SelectVariant(original, candidate).find("HTTP/1.1 200"), 0u);
  generate(1);
  const auto metrics = Read(Send("GET", "/metrics"));
  const std::string batches = "\nloom_serve_server_batches_total ";
  const auto position = metrics.find(batches);
  ASSERT_NE(position, std::string::npos) << metrics;
  EXPECT_EQ(std::stoull(metrics.substr(position + batches.size())),
            GetParam() ? 9u : 4u);
  loom_serve::testing::SourcePackage broken;
  broken.Add("model/generate.loom", "not a Loom program");
  IREE_ASSERT_OK(http_.Send(
      broken.Request(
          "/v1/models/variants",
          "{\"model\":\"fixture-image\",\"base\":" + std::to_string(original) +
              ",\"sources\":[\"generate.loom\"],\"wait\":true}"),
      &client));
  const auto failed = Read(client);
  EXPECT_EQ(failed.find("HTTP/1.1 422"), 0u) << failed;
  EXPECT_EQ(Number(failed, "selected"), original);
  generate(1);
  // The original preparation owns the budget even when a candidate's command
  // program grows. Reject before selection without disturbing the live model.
  const std::string grown_allocation = "index.constant 4160 : offset";
  source.replace(source.find(grown_allocation), grown_allocation.size(),
                 "index.constant 16384 : offset");
  loom_serve::testing::SourcePackage over_budget;
  over_budget.Add("model/generate.loom", source);
  IREE_ASSERT_OK(
      over_budget.AddFile("model/config.loom", directory_ + "/config.loom"));
  IREE_ASSERT_OK(http_.Send(
      over_budget.Request(
          "/v1/models/variants",
          "{\"model\":\"fixture-image\",\"base\":" + std::to_string(original) +
              ",\"sources\":[\"config.loom\",\"generate.loom\"]}"),
      &client));
  const auto rejected = Read(client);
  EXPECT_EQ(rejected.find("HTTP/1.1 422"), 0u) << rejected;
  EXPECT_NE(rejected.find("workspace"), std::string::npos) << rejected;
  EXPECT_EQ(Number(rejected, "selected"), original);
  generate(1);
  loom_serve_http_server_request_stop(server_);
  IREE_ASSERT_OK(http_.Join());
  const auto retained = loom_serve_image_model_weight_activity(model_);
  EXPECT_EQ(retained.read_bytes, weights.read_bytes);
  EXPECT_EQ(retained.prepared_bytes, weights.prepared_bytes);
  EXPECT_EQ(retained.activations, weights.activations);
}

TEST_P(SourceImageServiceTest, PreparedShapePinsCodeAcrossSelectionAndRemoval) {
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  const auto generator = loom_serve_image_model_generator(model_);
  const auto update = generator.execution_update;
  const std::string paths[] = {directory_ + "/config.loom",
                               directory_ + "/generate.loom"};
  std::string contents[2];
  for (size_t i = 0; i < 2; ++i) {
    std::ifstream input(paths[i]);
    ASSERT_TRUE(input.good());
    contents[i].assign(std::istreambuf_iterator<char>(input), {});
  }
  const std::string red = "scf.select %odd_seed, %high, %low";
  const auto red_position = contents[1].find(red);
  ASSERT_NE(red_position, std::string::npos);
  contents[1].replace(red_position, red.size(),
                      "scf.select %odd_seed, %low, %high");
  const std::string launch = "  kernel.launch @pixels(%colors, %rgb)";
  const auto launch_position = contents[1].find(launch);
  ASSERT_NE(launch_position, std::string::npos);
  contents[1].insert(
      launch_position,
      "  kernel.launch @colors(%request, %colors) : (buffer, buffer)\n");
  const loom_serve_package_file_t files[] = {
      {IREE_SV("model/config.loom"),
       iree_make_const_byte_span(contents[0].data(), contents[0].size())},
      {IREE_SV("model/generate.loom"),
       iree_make_const_byte_span(contents[1].data(), contents[1].size())}};
  loom_serve_package_t* package = nullptr;
  IREE_ASSERT_OK(loom_serve_package_create(IREE_SV("model/model.json"),
                                           IREE_ARRAYSIZE(files), files,
                                           host_allocator_, &package));
  std::unique_ptr<loom_serve_package_t, decltype(&loom_serve_package_release)>
      retained_package(package, loom_serve_package_release);
  const iree_string_view_t providers[] = {IREE_SV("config.loom"),
                                          IREE_SV("generate.loom")};
  const loom_serve_sources_t sources = {
      .count = IREE_ARRAYSIZE(providers),
      .values = providers,
      .package = package,
  };
  struct Compilation {
    // Joins the accepted compiler's publication with this outer test thread.
    std::mutex mutex;
    // Notifies completion without blocking a compiler or serving worker.
    std::condition_variable condition;
    // Compiler result, transferred after the callback's final access.
    void* candidate = nullptr;
    // Owned compilation outcome, consumed after publication.
    iree_status_t status = iree_ok_status();
    // Terminal callback has transferred ownership.
    bool complete = false;
    // Request compilation publications observed by the serialized test owner.
    uint64_t wakes = 0;
  } compilation;
  IREE_ASSERT_OK(update.compile(
      update.self, &sources,
      [](void* self, iree_status_t status, void* value,
         const loom_serve_compilation_report_t*) {
        auto* compilation = static_cast<Compilation*>(self);
        std::lock_guard<std::mutex> lock(compilation->mutex);
        compilation->candidate = value;
        compilation->status = status;
        compilation->complete = true;
        compilation->condition.notify_one();
      },
      &compilation));
  {
    std::unique_lock<std::mutex> lock(compilation.mutex);
    compilation.condition.wait(lock, [&] { return compilation.complete; });
  }
  std::unique_ptr<void, loom_serve_variant_dispose_fn_t> candidate(
      compilation.candidate, update.dispose);
  IREE_ASSERT_OK(std::exchange(compilation.status, iree_ok_status()));
  retained_package.reset();
  generator.set_observer(
      generator.self, {[](void* self) {
                         auto* publication = static_cast<Compilation*>(self);
                         std::lock_guard<std::mutex> lock(publication->mutex);
                         ++publication->wakes;
                         publication->condition.notify_all();
                       },
                       &compilation});

  auto complete = [&](uint64_t expected_seed, uint32_t expected_parts) {
    bool resolved = false;
    while (!resolved) {
      uint64_t before;
      {
        std::lock_guard<std::mutex> lock(compilation.mutex);
        before = compilation.wakes;
      }
      IREE_ASSERT_OK(
          generator.advance_prepared_request(generator.self, false, &resolved));
      if (!resolved) {
        std::unique_lock<std::mutex> lock(compilation.mutex);
        compilation.condition.wait(lock,
                                   [&] { return compilation.wakes != before; });
      }
    }
    IREE_ASSERT_OK(loom_serve_residency_acquire_wait(
        loom_serve_image_model_residency(model_)));
    bool finished = false;
    uint32_t parts = 0;
    while (!finished) {
      loom_serve_execution_frontier_t frontier;
      IREE_ASSERT_OK(generator.submit(generator.self, 0, &frontier, &finished));
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          {2, frontier.semaphores, frontier.values}, iree_infinite_timeout(),
          IREE_ASYNC_WAIT_FLAG_NONE));
      ++parts;
    }
    EXPECT_EQ(parts, expected_parts);
    const auto rgb = generator.complete(generator.self);
    const auto expected = Pixels(3, expected_seed, 15 * 9);
    ASSERT_EQ(rgb.data_length, expected.size() * sizeof(float));
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(iree_unaligned_load_le_f32(rgb.data + i * sizeof(float)),
                expected[i]);
    }
  };
  loom_serve_image_prepared_t prepared;
  IREE_ASSERT_OK(loom_serve_image_model_prepare_request(
      model_, IREE_SV("aba"), 1, 1.0f, 15, 9, &prepared));
  IREE_ASSERT_OK(update.select(update.self, candidate.get()));
  ASSERT_NO_FATAL_FAILURE(complete(1, GetParam() ? 2 : 1));

  IREE_ASSERT_OK(loom_serve_image_model_prepare_request(
      model_, IREE_SV("aba"), 1, 1.0f, 15, 9, &prepared));
  IREE_ASSERT_OK(update.select(update.self, nullptr));
  candidate.reset();
  ASSERT_NO_FATAL_FAILURE(complete(0, GetParam() ? 3 : 1));

  IREE_ASSERT_OK(loom_serve_image_model_prepare_request(
      model_, IREE_SV("aba"), 1, 1.0f, 15, 9, &prepared));
  ASSERT_NO_FATAL_FAILURE(complete(1, GetParam() ? 2 : 1));
  generator.set_observer(generator.self, {});
}

TEST_P(SourceImageServiceTest, SelectionWaitsForWholeImageAndEncoderRelease) {
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  const loom_serve_image_service_options_t options = {
      .name = loom_serve_image_model_name(model_),
      .width = kWidth,
      .height = kHeight,
      .pending_capacity = 8,
      .variant_capacity = 4,
  };
  loom_serve_image_service_t* value = nullptr;
  IREE_ASSERT_OK(loom_serve_image_service_allocate(
      host_, loom_serve_image_model_generator(model_), &options, nullptr,
      host_allocator_, &value));
  std::unique_ptr<loom_serve_image_service_t,
                  decltype(&loom_serve_image_service_free)>
      service(value, loom_serve_image_service_free);
  const auto interface = loom_serve_image_service_interface(value);
  auto* notification = loom_serve_http_server_notification(server_);
  interface.set_observer(interface.self,
                         {[](void* self) {
                            iree_notification_post(
                                static_cast<iree_notification_t*>(self),
                                IREE_ALL_WAITERS);
                          },
                          notification});
  auto advance_until = [&](auto ready) {
    while (!ready()) {
      const auto ticket = iree_notification_prepare_wait(notification);
      bool progress = false;
      IREE_ASSERT_OK(interface.advance(interface.self, 0, &progress));
      if (ready() || progress) {
        iree_notification_cancel_wait(notification);
      } else {
        iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
  };
  loom_serve::testing::ModelDocument document;
  IREE_ASSERT_OK(document.Load(directory_ + "/model.json"));
  auto sources = document.value().sources;
  sources.prepare = {};
  sources.control = {};
  const auto& configuration = interface.reconfiguration;
  const auto original = configuration.selected(configuration.self);
  loom_serve_variant_id_t candidate = 0;
  IREE_ASSERT_OK(configuration.stage_native(configuration.self, original,
                                            &sources, &candidate));
  loom_serve_variant_snapshot_t snapshot;
  ASSERT_NO_FATAL_FAILURE(advance_until([&] {
    return loom_serve_variant_pool_snapshot(configuration.pool, candidate,
                                            &snapshot) &&
           snapshot.phase != LOOM_SERVE_VARIANT_COMPILING;
  }));
  ASSERT_EQ(snapshot.phase, LOOM_SERVE_VARIANT_READY)
      << std::string(snapshot.message.data ? snapshot.message.data : "",
                     snapshot.message.size);
  const int client =
      Send("POST", "/v1/images/generations",
           R"({"model":"fixture-image","prompt":"aba","seed":1})");
  loom_serve_http_connection_t* connection = nullptr;
  const loom_serve_http_request_t* request = nullptr;
  IREE_ASSERT_OK(http_.TakeRequest(client, &connection, &request));
  IREE_ASSERT_OK(interface.accept(interface.self, connection, request, 0));
  ASSERT_NO_FATAL_FAILURE(advance_until([&] {
    return std::string(interface.snapshot(interface.self).phase) ==
           "generating";
  }));
  struct loom_serve_test_image_selection_t {
    // Borrowed live service, accessed only on this serialized owner.
    const loom_serve_http_service_t* interface;
    // Terminal selection publication after the entire invocation retires.
    bool complete;
  } selection{&interface, false};
  IREE_ASSERT_OK(configuration.select(
      configuration.self, candidate, original,
      [](void* user_data, iree_status_t status,
         loom_serve_variant_id_t selected) {
        auto* selection =
            static_cast<loom_serve_test_image_selection_t*>(user_data);
        IREE_EXPECT_OK(status);
        const auto* interface = selection->interface;
        EXPECT_STREQ(interface->snapshot(interface->self).phase, "idle");
        EXPECT_EQ(interface->telemetry.counters->completed, 1u);
        selection->complete = true;
      },
      &selection));
  EXPECT_FALSE(selection.complete);
  ASSERT_NO_FATAL_FAILURE(advance_until([&] { return selection.complete; }));
  EXPECT_EQ(configuration.selected(configuration.self), candidate);
  EXPECT_NE(Read(client).find("\"b64_json\":\"" + EncodedPixels(3, 1) + "\""),
            std::string::npos);
  EXPECT_EQ(interface.telemetry.counters->batches, GetParam() ? 2u : 1u);
  interface.set_observer(interface.self, {});
}

TEST_P(SourceImageServiceTest, SourcePixelsAndHttpPngPreserveRequestIdentity) {
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  loom_serve_image_prepared_t prepared;
  IREE_ASSERT_OK(loom_serve_image_model_prepare_request(
      model_, IREE_SV("x"), 0, 1.0f, kWidth, kHeight, &prepared));
  EXPECT_EQ(prepared.stage, 0u);
  EXPECT_EQ(prepared.tag, 32);
  EXPECT_EQ(prepared.input_bytes, 16u);
  loom_serve_image_model_discard_prepared_request(model_);
  // Verify source arithmetic independently of the image encoder.
  for (uint64_t seed : {0u, 1u}) {
    iree_const_byte_span_t rgb;
    IREE_ASSERT_OK(loom_serve_image_model_generate(
        model_, IREE_SV("aba"), seed, 1.0f, kWidth, kHeight, &prepared, &rgb));
    EXPECT_EQ(prepared.stage, 0u);
    EXPECT_EQ(prepared.tag, 32);
    EXPECT_EQ(prepared.input_bytes, 16u);
    const auto expected = Pixels(3, seed);
    ASSERT_EQ(rgb.data_length, expected.size() * sizeof(float));
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(iree_unaligned_load_le_f32(rgb.data + 4 * i), expected[i]);
    }
  }
  Start();
  EXPECT_NE(
      Read(Send("GET", "/v1/models"))
          .find(
              R"("input_modalities":["text"],"output_modalities":["image"],"loom":{"kind":"image","endpoint":"/v1/images/generations","tokenize":true,"variants":)"),
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

TEST_P(SourceImageServiceTest, RequestGeometryOwnsPixelsAndPreservesDefault) {
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  Start();
  const std::array<uint32_t, 2> dimensions[] = {
      {15, 9}, {9, 15}, {16, 11}, {1, 1}, {15, 9}, {kWidth, kHeight}};
  std::vector<int> clients;
  for (size_t i = 0; i < IREE_ARRAYSIZE(dimensions); ++i) {
    const auto [width, height] = dimensions[i];
    const std::string size = i + 1 == IREE_ARRAYSIZE(dimensions)
                                 ? ""
                                 : ",\"size\":\"" + std::to_string(width) +
                                       "x" + std::to_string(height) + "\"";
    clients.push_back(
        Send("POST", "/v1/images/generations",
             "{\"model\":\"fixture-image\",\"prompt\":\"aba\",\"seed\":" +
                 std::to_string(i) + size + "}"));
  }
  uint64_t generated_pixels = 0;
  for (size_t i = 0; i < clients.size(); ++i) {
    const auto [width, height] = dimensions[i];
    const auto response = Read(clients[i]);
    ASSERT_EQ(response.find("HTTP/1.1 200 OK\r\n"), 0u) << response;
    EXPECT_NE(response.find("\"b64_json\":\"" +
                            EncodedPixels(3, i, width, height) + "\""),
              std::string::npos);
    generated_pixels += (uint64_t)width * height;
  }
  const auto rejected =
      Read(Send("POST", "/v1/images/generations",
                R"({"model":"fixture-image","prompt":"aba","size":"65x9"})"));
  EXPECT_EQ(rejected.find("HTTP/1.1 400 Bad Request\r\n"), 0u) << rejected;
  EXPECT_NE(rejected.find("1-64 pixel extents"), std::string::npos);
  const auto restored =
      Read(Send("POST", "/v1/images/generations",
                R"({"model":"fixture-image","prompt":"aba","seed":1})"));
  EXPECT_EQ(restored.find("HTTP/1.1 200 OK\r\n"), 0u) << restored;
  EXPECT_NE(restored.find("\"b64_json\":\"" + EncodedPixels(3, 1) + "\""),
            std::string::npos);
  generated_pixels += kPixels;
  // The model log has a bounded page size. Follow its cursor to the final
  // cumulative snapshot instead of treating the first page as the whole log.
  std::string cursor;
  std::string events;
  std::string latest;
  for (;;) {
    latest =
        Read(Send("GET", "/logz", {},
                  "X-Loom-Model: fixture-image\r\n" +
                      (cursor.empty() ? std::string()
                                      : "X-Loom-Cursor: " + cursor + "\r\n")));
    ASSERT_EQ(latest.find("HTTP/1.1 200 OK\r\n"), 0u) << latest;
    ASSERT_EQ(latest.find(R"("event":"gap")"), std::string::npos) << latest;
    events += latest;
    const std::string marker = R"({"event":"cursor","value":")";
    const size_t position = latest.find(marker);
    ASSERT_NE(position, std::string::npos) << latest;
    const size_t first = position + marker.size();
    const std::string next =
        latest.substr(first, latest.find('"', first) - first);
    if (cursor == next) {
      break;
    }
    cursor = next;
  }
  EXPECT_NE(latest.find(R"("requests":8,"completed":7,"rejected":1)"),
            std::string::npos)
      << latest;
  EXPECT_NE(events.find("\"partial\":false"), std::string::npos) << events;
  const std::string pixel_count = "\"generated_pixels\":";
  const size_t position = events.rfind(pixel_count);
  ASSERT_NE(position, std::string::npos) << events;
  const uint64_t recorded_pixels =
      std::stoull(events.substr(position + pixel_count.size()));
  EXPECT_EQ(recorded_pixels, generated_pixels);
}

TEST_P(SourceImageServiceTest, ClosedTransportTerminatesModelPolling) {
  StartManaged();
  loom_serve_http_server_request_stop(server_);
  IREE_ASSERT_OK(http_.Join());
  ::testing::TestPartResultArray failures;
  {
    ::testing::ScopedFakeTestPartResultReporter reporter(
        ::testing::ScopedFakeTestPartResultReporter::
            INTERCEPT_ONLY_CURRENT_THREAD,
        &failures);
    EXPECT_TRUE(WaitModel("absent", "ready").empty());
  }
  ASSERT_EQ(failures.size(), 1);
  EXPECT_TRUE(failures.GetTestPartResult(0).fatally_failed());
  EXPECT_TRUE(http_.failed());
}

TEST_P(SourceImageServiceTest, ShapeCompilationFailureLeavesModelReusable) {
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  Start();
  const auto inventory = Read(Send("GET", "/v1/models/variants", {},
                                   "X-Loom-Model: fixture-image\r\n"));
  const uint64_t original = Number(inventory, "selected");
  std::ifstream input(directory_ + "/generate.loom");
  ASSERT_TRUE(input.good());
  std::string source((std::istreambuf_iterator<char>(input)), {});
  const std::string range = "range(%value, 1, 4096)";
  const auto offset = source.find(range);
  ASSERT_NE(offset, std::string::npos);
  source.replace(offset, range.size(), "range(%value, 1, 100)");
  loom_serve::testing::SourcePackage package;
  package.Add("model/generate.loom", source);
  IREE_ASSERT_OK(
      package.AddFile("model/config.loom", directory_ + "/config.loom"));
  int client = -1;
  IREE_ASSERT_OK(http_.Send(
      package.Request(
          "/v1/models/variants",
          "{\"model\":\"fixture-image\",\"base\":" + std::to_string(original) +
              ",\"sources\":[\"config.loom\",\"generate.loom\"],\"wait\":"
              "true}"),
      &client));
  const auto staged = Read(client);
  ASSERT_EQ(staged.find("HTTP/1.1 200"), 0u) << staged;
  ASSERT_EQ(SelectVariant(Number(staged, "id", "variant"), original)
                .find("HTTP/1.1 200"),
            0u);
  const auto rejected =
      Read(Send("POST", "/v1/images/generations",
                R"({"model":"fixture-image","prompt":"aba","size":"15x9"})"));
  EXPECT_EQ(rejected.find("HTTP/1.1 422"), 0u) << rejected;
  const auto valid = Read(Send(
      "POST", "/v1/images/generations",
      R"({"model":"fixture-image","prompt":"aba","seed":1,"size":"9x9"})"));
  EXPECT_EQ(valid.find("HTTP/1.1 200"), 0u) << valid;
  EXPECT_NE(valid.find("\"b64_json\":\"" + EncodedPixels(3, 1, 9, 9) + "\""),
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
      ASSERT_FALSE(HasFatalFailure());
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
      (GetParam() ? "80" : "40");
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
  IREE_EXPECT_OK(http_.Join());
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
  iree_status_t detached_status = http_.AwaitResponse(detached);
  EXPECT_EQ(close(detached), 0);
  IREE_ASSERT_OK(detached_status);
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

TEST_P(SourceImageServiceTest, UploadedImageRegistrationProducesExactPixels) {
  StartManaged();
  {
    loom_serve::testing::SourcePackage package;
    for (const auto* file : {"prepare.loom", "control.loom", "config.loom",
                             "generate.loom", "tokenizer.json"}) {
      IREE_ASSERT_OK(package.AddFile(std::string("model/") + file,
                                     directory_ + "/" + file));
    }
    const std::string declaration =
        R"({"name":"pixels","admission":"drained","definition":{)"
        R"("kind":"image","prepare":"prepare.loom","control":"control.loom",)"
        R"("sources":["config.loom","generate.loom"],"checkpoint":".",)"
        R"("width":7,"height":5,"text_tokens":32,"command_quantum":)" +
        std::to_string(GetParam()) + "}}";
    int client = -1;
    IREE_ASSERT_OK(http_.Send(
        package.Request("/v1/models/register", declaration), &client));
    const auto registered = Read(client);
    ASSERT_EQ(registered.find("HTTP/1.1 202"), 0u) << registered;
    EXPECT_NE(registered.find(R"("registration":"registered")"),
              std::string::npos);
  }
  ASSERT_EQ(Read(Send("POST", "/v1/models/undrain", R"({"model":"pixels"})"))
                .find("HTTP/1.1 200"),
            0u);
  const auto response =
      Read(Send("POST", "/v1/images/generations",
                R"({"model":"pixels","prompt":"aba","seed":1})"));
  EXPECT_NE(response.find("\"b64_json\":\"" + EncodedPixels(3, 1) + "\""),
            std::string::npos)
      << response;
  // Registration retirement also joins accepted, detached compilation, not
  // just the last image's device timeline. Uploaded providers outlive this
  // request and are released before the model's parameter roots.
  const uint64_t base = Number(
      Read(Send("GET", "/v1/models/variants", {}, "X-Loom-Model: pixels\r\n")),
      "selected");
  {
    loom_serve::testing::SourcePackage package;
    for (const char* file : {"config.loom", "generate.loom"}) {
      IREE_ASSERT_OK(package.AddFile(std::string("model/") + file,
                                     directory_ + "/" + file));
    }
    int client = -1;
    IREE_ASSERT_OK(http_.Send(
        package.Request(
            "/v1/models/variants",
            "{\"model\":\"pixels\",\"base\":" + std::to_string(base) +
                ",\"sources\":[\"config.loom\",\"generate.loom\"],\"wait\":"
                "false}"),
        &client));
    const auto accepted = Read(client);
    ASSERT_EQ(accepted.find("HTTP/1.1 202"), 0u) << accepted;
  }
  ASSERT_EQ(Read(Send("POST", "/v1/models/unregister", R"({"model":"pixels"})"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModel("pixels", {});
}

TEST_P(SourceImageServiceTest, TokenizerFailureRetiresCompiledModel) {
  const auto assets =
      iree::testing::MakeTempFilePath("missing_image_tokenizer");
  ASSERT_TRUE(std::filesystem::create_directory(assets));
  loom_serve::testing::ModelDocument document;
  IREE_ASSERT_OK(document.Load(directory_ + "/model.json"));
  const loom_serve_image_model_options_t options = {
      .sources = document.value().sources,
      .checkpoint_directory = iree_make_cstring_view(assets.c_str()),
      .height = kHeight,
      .width = kWidth,
      .text_tokens = 32,
      .command_quantum = GetParam(),
  };
  const auto before = loom_serve_device_memory_statistics(device_);
  loom_serve_image_model_t* failed = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        loom_serve_image_model_allocate(
                            device_, &options, host_allocator_, &failed));
  EXPECT_EQ(failed, nullptr);
  const auto after = loom_serve_device_memory_statistics(device_);
  EXPECT_EQ(after.retained.reserved_bytes, before.retained.reserved_bytes);
  EXPECT_EQ(after.retained.committed_bytes, before.retained.committed_bytes);

  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  Start();
  const auto response =
      Read(Send("POST", "/v1/images/generations",
                R"({"model":"fixture-image","prompt":"aba","seed":1})"));
  EXPECT_NE(response.find("\"b64_json\":\"" + EncodedPixels(3, 1) + "\""),
            std::string::npos)
      << response;
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
  IREE_ASSERT_OK(loom_serve_image_model_prepare_request(
      model_, IREE_SV("aba"), 1, 1.0f, kWidth, kHeight, &prepared));
  const auto generator = loom_serve_image_model_generator(model_);
  bool resolved = false;
  IREE_ASSERT_OK(
      generator.advance_prepared_request(generator.self, false, &resolved));
  ASSERT_TRUE(resolved);
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(generator.try_admit(generator.self, &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  loom_serve_execution_frontier_t frontier;
  bool complete = false;
  IREE_ASSERT_OK(generator.submit(generator.self, 0, &frontier, &complete));
  EXPECT_EQ(complete, GetParam() == 0);
  // Failure readiness does not authorize releasing the workspace or upload
  // payload while the native queue still owns its accepted first part.
  iree_hal_semaphore_fail(
      frontier.semaphores[0],
      iree_make_status(IREE_STATUS_ABORTED, "failed image part"));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        loom_serve_image_model_free(model_));
  model_ = nullptr;
}

INSTANTIATE_TEST_SUITE_P(CommandQuanta, SourceImageServiceTest,
                         ::testing::Values(0u, 1u));

class ImageSpecializationTest : public SourceImageServiceTest {
 protected:
  void TearDown() override {
    ReleaseCompiler();
    if (prepared_) {
      IREE_EXPECT_OK(Resolve(true));
      loom_serve_image_model_discard_prepared_request(model_);
    }
    SourceImageServiceTest::TearDown();
  }

  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<ImageSpecializationTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        std::this_thread::get_id() != test->owner_) {
      ++test->compiler_allocations_;
      std::unique_lock<std::mutex> lock(test->mutex_);
      if (test->holding_) {
        test->entered_ = true;
        test->condition_.notify_all();
        test->condition_.wait(lock, [&] { return !test->holding_; });
      }
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  static void Wake(void* self) {
    auto* test = static_cast<ImageSpecializationTest*>(self);
    std::lock_guard<std::mutex> lock(test->mutex_);
    ++test->wakes_;
    test->condition_.notify_all();
  }

  iree_status_t Resolve(bool cancel) {
    bool complete = false;
    while (!complete) {
      const auto before = wakes_.load();
      IREE_RETURN_IF_ERROR(
          loom_serve_image_model_advance_request(model_, cancel, &complete));
      if (!complete) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] { return wakes_.load() != before; });
      }
    }
    return iree_ok_status();
  }

  void Run(uint32_t width, uint32_t height, bool reuse) {
    const auto before = compiler_allocations_.load();
    loom_serve_image_prepared_t prepared;
    iree_const_byte_span_t rgb;
    IREE_ASSERT_OK(loom_serve_image_model_generate(
        model_, IREE_SV("aba"), 1, 1.0f, width, height, &prepared, &rgb));
    EXPECT_EQ(compiler_allocations_.load() == before, reuse);
    const auto expected = Pixels(3, 1, (size_t)width * height);
    ASSERT_EQ(rgb.data_length, expected.size() * sizeof(float));
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(iree_unaligned_load_le_f32(rgb.data + i * sizeof(float)),
                expected[i]);
    }
  }

  void ReleaseCompiler() {
    std::lock_guard<std::mutex> lock(mutex_);
    holding_ = false;
    condition_.notify_all();
  }

  // This outer thread performs preparation and drives the real native model.
  const std::thread::id owner_ = std::this_thread::get_id();
  // Model allocations on actual compiler workers distinguish misses from hits.
  std::atomic<size_t> compiler_allocations_ = 0;
  // Published compiler completions observed by the outer test owner.
  std::atomic<uint64_t> wakes_ = 0;
  // Protects the explicit dependency used to test accepted-compile retirement.
  std::mutex mutex_;
  // Signals worker entry, dependency release, and terminal publication.
  std::condition_variable condition_;
  // An accepted worker is held only for the cancellation experiment.
  bool holding_ = false;
  // A real compiler allocation has reached the held boundary.
  bool entered_ = false;
  // Teardown must collect and release an outstanding prepared request.
  bool prepared_ = false;
};

TEST_P(ImageSpecializationTest, ExactReuseAndLruEvictionPreserveOutput) {
  ASSERT_NO_FATAL_FAILURE(
      CreateStandaloneModel({2, 16 * 1024 * 1024}, {this, Allocate}));
  ASSERT_NO_FATAL_FAILURE(Run(15, 9, false));
  // This source specializes pixel count, so rotation reuses the same code.
  ASSERT_NO_FATAL_FAILURE(Run(9, 15, true));
  ASSERT_NO_FATAL_FAILURE(Run(16, 11, false));
  ASSERT_NO_FATAL_FAILURE(Run(15, 9, true));
  ASSERT_NO_FATAL_FAILURE(Run(1, 1, false));
  ASSERT_NO_FATAL_FAILURE(Run(16, 11, false));
  ASSERT_NO_FATAL_FAILURE(Run(kWidth, kHeight, true));
}

TEST_P(ImageSpecializationTest, DisabledReuseStillAllowsLargerRequests) {
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel({0, 0}, {this, Allocate}));
  ASSERT_NO_FATAL_FAILURE(Run(15, 9, false));
  ASSERT_NO_FATAL_FAILURE(Run(15, 9, false));
  ASSERT_NO_FATAL_FAILURE(Run(kWidth, kHeight, true));
}

TEST_P(ImageSpecializationTest, CancellationJoinsAcceptedCompilerBeforeReuse) {
  ASSERT_NO_FATAL_FAILURE(
      CreateStandaloneModel({2, 16 * 1024 * 1024}, {this, Allocate}));
  loom_serve_image_model_set_observer(model_, {Wake, this});
  loom_serve_image_prepared_t prepared;
  IREE_ASSERT_OK(loom_serve_image_model_prepare_request(
      model_, IREE_SV("aba"), 1, 1.0f, 15, 9, &prepared));
  prepared_ = true;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    holding_ = true;
  }
  bool complete = true;
  IREE_ASSERT_OK(
      loom_serve_image_model_advance_request(model_, false, &complete));
  EXPECT_FALSE(complete);
  {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return entered_; });
  }
  IREE_ASSERT_OK(
      loom_serve_image_model_advance_request(model_, true, &complete));
  EXPECT_FALSE(complete);
  ReleaseCompiler();
  IREE_ASSERT_OK(Resolve(true));
  loom_serve_image_model_discard_prepared_request(model_);
  prepared_ = false;
  loom_serve_image_model_set_observer(model_, {});
  ASSERT_NO_FATAL_FAILURE(Run(15, 9, true));
  ASSERT_NO_FATAL_FAILURE(Run(kWidth, kHeight, true));
}

INSTANTIATE_TEST_SUITE_P(CommandQuanta, ImageSpecializationTest,
                         ::testing::Values(0u, 1u));

}  // namespace
