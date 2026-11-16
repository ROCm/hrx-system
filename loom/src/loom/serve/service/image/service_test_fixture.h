// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_IMAGE_SERVICE_TEST_FIXTURE_H_
#define LOOM_SERVE_SERVICE_IMAGE_SERVICE_TEST_FIXTURE_H_

#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
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
#include "iree/vm/buffer.h"
#include "loom/runtime/program/package.h"
#include "loom/runtime/program/variant_pool.h"
#include "loom/runtime/service_control.h"
#include "loom/runtime/telemetry/field.h"
#include "loom/runtime/telemetry/stream.h"
#include "loom/runtime/testing/application.h"
#include "loom/runtime/testing/service_control.h"
#include "loom/runtime/testing/stalled_stderr.h"
#include "loom/serve/server/model_registration.h"
#include "loom/serve/server/router.h"
#include "loom/serve/service/image/model.h"
#include "loom/serve/service/image/output.h"
#include "loom/serve/service/image/service.h"
#include "loom/serve/telemetry/record.h"
#include "loom/serve/testing/http_harness.h"
#include "loom/serve/testing/model.h"
#include "loom/serve/testing/model_wait.h"
#include "loom/serve/testing/package.h"
#include "loom/serve/transport/http/listener.h"

namespace loom_serve::testing::image {

struct ImageRequest {
  ~ImageRequest() { loom_serve_image_request_deinitialize(&value); }

  iree_status_t Initialize(iree_string_view_t body, uint32_t width,
                           uint32_t height) {
    return loom_serve_image_request_initialize(body, IREE_SV("fixture-image"),
                                               width, height,
                                               iree_allocator_system(), &value);
  }

  // Decoded input remains live until source cohort capture has finished.
  loom_serve_image_request_t value = {};
};

class SourceImageServiceTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  static constexpr uint32_t kWidth = 7;
  static constexpr uint32_t kHeight = 5;
  static constexpr size_t kPixels = kWidth * kHeight;

  void SetUp() override {
    const char* source_path = std::getenv("LOOM_SERVE_TEST_MODEL_SOURCE");
    ASSERT_NE(source_path, nullptr);
    auto application_options = loom_runtime_test_application_options_default();
    if (application_options.backing == LOOM_RUNTIME_POOL_BACKING_ELASTIC) {
      // Small fixture images fit, but a 64x64 RGBA8 image does not.
      application_options.slab_size = 4096;
      application_options.memory_limit = 16 * 1024;
    }
    loom_runtime_test_application_t* application = nullptr;
    IREE_ASSERT_OK(loom_runtime_test_application_acquire(
        &application_options, host_allocator_, &application));
    host_ = loom_runtime_test_application_host(application);
    group_ = loom_runtime_test_application_device_group(application);
    device_ = loom_runtime_test_application_device_at(application, 0);
    storage_owner_ = loom_runtime_device_group_pool_at(
        loom_runtime_test_application_device_group(application), 0);
    const std::string source = source_path;
    directory_ = source.substr(0, source.find_last_of('/'));
    auto options = loom_serve_http_server_options_default();
    IREE_ASSERT_OK(
        loom_serve_http_server_allocate(loom_runtime_host_proactor_pool(host_),
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
                loom_runtime_pool_residency_cache(storage_owner_));
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
          host_, generator, &options, nullptr, scheduling_.binding(),
          host_allocator_, &service);
      if (iree_status_is_ok(status)) {
        const auto interface = loom_serve_image_service_interface(service);
        loom_serve_registry_t* registry = nullptr;
        status = loom_serve_registry_allocate(1, &interface, 0, {},
                                              host_allocator_, &registry);
        if (iree_status_is_ok(status)) {
          status = loom_serve_router_run(
              loom_runtime_host_executor(host_), server_, registry,
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
      loom_runtime_variant_cache_options_t cache = {2, 16 * 1024 * 1024},
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
    IREE_ASSERT_OK(loom_serve_image_model_allocate(
        device_, storage_owner_, &options, allocator, &model_));
  }

  void Wait(loom_runtime_execution_frontier_t frontier) {
    for (size_t i = 0; i < IREE_ARRAYSIZE(frontier.semaphores); ++i) {
      IREE_ASSERT_OK(loom_runtime_workspace_wait(
          loom_runtime_pool_workspace(storage_owner_), frontier.semaphores[i],
          frontier.values[i]));
    }
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
          loom_runtime_host_executor(host_), server_, registry,
          {
              .request_quota = 16,
              .holding_capacity = 8,
              .resources = {.device_group = group_},
          },
          host_allocator_);
      return iree_status_join(status, loom_serve_registry_free(registry));
    });
  }

  iree_status_t PrepareSingle(iree_string_view_t body, uint32_t width,
                              uint32_t height,
                              loom_service_image_prepared_t* out_prepared) {
    ImageRequest input;
    IREE_RETURN_IF_ERROR(input.Initialize(body, width, height));
    loom_service_prepared_request_t request = {};
    IREE_RETURN_IF_ERROR(
        loom_serve_image_model_prepare_request(model_, &input.value, &request));
    const iree_host_size_t output_length = (iree_host_size_t)width * height * 4;
    const iree_status_t status = loom_serve_image_model_prepare_batch(
        model_, 1, &request, &output_length, out_prepared);
    iree_vm_buffer_release(request.bytes);
    return status;
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

  std::string WaitModelState(const std::string& name,
                             const std::string& marker) {
    return loom_serve::testing::WaitModelState(
        http_, name, marker, [this, &name] { return ModelStatus(name); });
  }

  void WaitModelAbsent(const std::string& name) {
    loom_serve::testing::WaitModelAbsent(
        http_, [this, &name] { return ModelStatus(name); });
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

  static std::vector<uint8_t> Pixels(size_t tokens, uint64_t seed,
                                     size_t count = kPixels,
                                     float red_strength = 1.0f,
                                     float green_strength = 1.0f) {
    std::vector<uint8_t> pixels(count * 4);
    const auto quantize = [](float value) {
      return uint8_t(std::nearbyint((value / 2.0f + 0.5f) * 255.0f));
    };
    for (size_t i = 0; i < count; ++i) {
      pixels[4 * i] = quantize((seed & 1 ? 1.0f : -1.0f) * red_strength);
      pixels[4 * i + 1] =
          quantize((tokens & 1 ? 1.0f : -1.0f) * green_strength);
      pixels[4 * i + 2] = i & 1 ? 255 : 0;
      pixels[4 * i + 3] = 255;
    }
    return pixels;
  }

  std::string EncodedPixels(size_t tokens, uint64_t seed,
                            uint32_t width = kWidth, uint32_t height = kHeight,
                            float red_strength = 1.0f,
                            float green_strength = 1.0f) {
    const auto pixels = Pixels(tokens, seed, (size_t)width * height,
                               red_strength, green_strength);
    iree_byte_span_t png;
    IREE_EXPECT_OK(loom_serve_image_encode_rgba8_png(
        width, height, iree_make_const_byte_span(pixels.data(), pixels.size()),
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

  void CheckAbandonedImage(loom_runtime_service_flags_t flags) {
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
    loom_serve_service_t interface = {};
    auto* notification = loom_serve_http_server_notification(server_);
    auto create_service = [&] {
      scheduling_.Reset();
      loom_serve_image_service_t* value = nullptr;
      IREE_ASSERT_OK(loom_serve_image_service_allocate(
          host_, loom_serve_image_model_generator(model_), &options, nullptr,
          scheduling_.binding(), host_allocator_, &value));
      service.reset(value);
      interface = loom_serve_image_service_interface(value);
      interface.runtime.vtable->set_observer(
          interface.runtime.self,
          {[](void* self) {
             iree_notification_post(static_cast<iree_notification_t*>(self),
                                    IREE_ALL_WAITERS);
           },
           notification});
    };
    ASSERT_NO_FATAL_FAILURE(create_service());
    auto accept = [&](int client, loom_serve_http_connection_t** connection) {
      const loom_serve_http_request_t* request = nullptr;
      IREE_ASSERT_OK(http_.TakeRequest(client, connection, &request));
      IREE_ASSERT_OK(
          interface.accept(interface.runtime.self, *connection, request, 0));
    };
    auto advance_until = [&](auto ready,
                             loom_runtime_service_flags_t flags = 0) {
      while (!ready()) {
        const auto ticket = iree_notification_prepare_wait(notification);
        bool progress = false;
        IREE_ASSERT_OK(interface.runtime.vtable->advance(interface.runtime.self,
                                                         flags, &progress));
        if (ready() || progress) {
          iree_notification_cancel_wait(notification);
        } else {
          iree_notification_commit_wait(notification, ticket,
                                        IREE_DURATION_ZERO,
                                        IREE_TIME_INFINITE_FUTURE);
        }
      }
    };

    std::array<int, 2> clients;
    std::array<loom_serve_http_connection_t*, 2> connections = {};
    for (size_t i = 0; i < clients.size(); ++i) {
      clients[i] = Send("POST", "/v1/images/generations",
                        R"({"model":"fixture-image","prompt":"ab"})");
      ASSERT_NO_FATAL_FAILURE(accept(clients[i], &connections[i]));
    }
    ASSERT_NO_FATAL_FAILURE(advance_until([&] {
      return std::string(interface.snapshot(interface.runtime.self).phase) ==
             "generating";
    }));
    EXPECT_EQ(interface.snapshot(interface.runtime.self).active_requests, 2u);
    if (!iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_STOPPING)) {
      for (size_t i = 0; i < clients.size(); ++i) {
        const linger reset = {1, 0};
        ASSERT_EQ(setsockopt(clients[i], SOL_SOCKET, SO_LINGER, &reset,
                             sizeof(reset)),
                  0);
        ASSERT_EQ(close(clients[i]), 0);
        iree_notification_await(
            notification,
            [](void* self) {
              return loom_serve_http_connection_failed(
                  static_cast<loom_serve_http_connection_t*>(self));
            },
            connections[i], iree_infinite_timeout());
      }
    }
    // Retire the accepted first part and its workspace before releasing the
    // pin. Split execution must skip the second kernel and output feedback;
    // neither path needs a PNG/base64 borrower for an absent consumer.
    ASSERT_NO_FATAL_FAILURE(advance_until(
        [&] { return interface.telemetry.counters->cancelled == 2; }, flags));
    EXPECT_EQ(interface.telemetry.counters->batches, 1u);
    EXPECT_STREQ(interface.snapshot(interface.runtime.self).phase, "idle");
    EXPECT_FALSE(interface.runtime.vtable->is_pending(interface.runtime.self));
    EXPECT_EQ(interface.telemetry.counters->cancelled, 2u);
    ASSERT_NO_FATAL_FAILURE(advance_until([&] {
      return !interface.runtime.vtable->is_pending(interface.runtime.self);
    }));
    if (iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_STOPPING)) {
      for (int client : clients) {
        EXPECT_TRUE(Read(client, /*allow_empty=*/true).empty());
      }
      interface.runtime.vtable->set_observer(interface.runtime.self, {});
      service.reset();
      ASSERT_NO_FATAL_FAILURE(create_service());
    }

    // Reuse the actual generator after retirement, including its prepared
    // input and output storage, and compare the complete encoded response.
    const int next =
        Send("POST", "/v1/images/generations",
             R"({"model":"fixture-image","prompt":"aba","seed":1})");
    ASSERT_NO_FATAL_FAILURE(accept(next, &connections[0]));
    ASSERT_NO_FATAL_FAILURE(advance_until(
        [&] { return interface.telemetry.counters->completed == 1; }));
    EXPECT_NE(Read(next).find("\"b64_json\":\"" + EncodedPixels(3, 1) + "\""),
              std::string::npos);
    EXPECT_EQ(interface.telemetry.counters->failed, 0u);
    interface.runtime.vtable->set_observer(interface.runtime.self, {});
  }

  void CheckHttpCohort(const char* reset_phase) {
    ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
    const loom_serve_image_service_options_t options = {
        .name = loom_serve_image_model_name(model_),
        .width = kWidth,
        .height = kHeight,
        .pending_capacity = 3,
    };
    loom_serve_image_service_t* value = nullptr;
    IREE_ASSERT_OK(loom_serve_image_service_allocate(
        host_, loom_serve_image_model_generator(model_), &options, nullptr,
        scheduling_.binding(), host_allocator_, &value));
    const auto interface = loom_serve_image_service_interface(value);
    auto* notification = loom_serve_http_server_notification(server_);
    interface.runtime.vtable->set_observer(
        interface.runtime.self, {[](void* self) {
                                   iree_notification_post(
                                       static_cast<iree_notification_t*>(self),
                                       IREE_ALL_WAITERS);
                                 },
                                 notification});
    auto advance_until = [&](auto ready,
                             loom_runtime_service_flags_t flags = 0) {
      while (!ready()) {
        const auto ticket = iree_notification_prepare_wait(notification);
        bool progress = false;
        IREE_ASSERT_OK(
            loom_serve_image_service_progress(value, flags, &progress));
        if (!ready()) {
          loom_runtime_service_offer_t offer = {};
          ASSERT_EQ(loom_runtime_service_read_offer(interface.runtime.control,
                                                    0, &offer),
                    LOOM_RUNTIME_SERVICE_READ_COMPLETE);
          loom_runtime_service_operation_key_t accepted;
          IREE_ASSERT_OK(loom_serve_image_service_try_schedule(
              value, offer.key, 0, flags, &accepted));
          progress |= accepted.serial != 0;
        }
        if (ready() || progress) {
          iree_notification_cancel_wait(notification);
        } else {
          iree_notification_commit_wait(notification, ticket,
                                        IREE_DURATION_ZERO,
                                        IREE_TIME_INFINITE_FUTURE);
        }
      }
    };
    auto destroy = [&](loom_serve_image_service_t* service) {
      bool progress = false;
      IREE_EXPECT_OK(loom_serve_image_service_progress(
          service, LOOM_RUNTIME_SERVICE_FLAG_STOPPING, &progress));
      advance_until(
          [&] {
            return !interface.runtime.vtable->is_pending(
                interface.runtime.self);
          },
          LOOM_RUNTIME_SERVICE_FLAG_STOPPING);
      interface.runtime.vtable->set_observer(interface.runtime.self, {});
      loom_serve_image_service_free(service);
    };
    std::unique_ptr<loom_serve_image_service_t, decltype(destroy)> service(
        value, destroy);
    auto accept = [&](int client, loom_serve_http_connection_t** connection) {
      const loom_serve_http_request_t* request = nullptr;
      IREE_ASSERT_OK(http_.TakeRequest(client, connection, &request));
      IREE_ASSERT_OK(
          interface.accept(interface.runtime.self, *connection, request, 0));
    };
    const std::array<std::string, 4> inputs = {
        R"({"model":"fixture-image","prompt":"ab","seed":0,"loras":[{"id":"red","scale":-0.75},{"id":"green","scale":-0.5}]})",
        R"({"model":"fixture-image","prompt":"ab","seed":2,"size":"6x6"})",
        R"({"model":"fixture-image","prompt":"aba","seed":1,"loras":[{"id":"green","scale":-0.25},{"id":"red","scale":-1.5}],"size":"5x7"})",
        R"({"model":"fixture-image","prompt":"ab","loras":[{"id":"red"},{"id":"\u0072ed","scale":0}]})",
    };
    std::array<loom_serve::testing::HttpClient, 4> clients;
    std::array<loom_serve_http_connection_t*, 4> connections = {};
    for (size_t i = 0; i < inputs.size(); ++i) {
      *clients[i].put() = Send("POST", "/v1/images/generations", inputs[i]);
      ASSERT_NO_FATAL_FAILURE(accept(clients[i].get(), &connections[i]));
    }
    // Cohorts share the same total host credit as queued work. Overflow is
    // rejected before any source task; it cannot grow the pending snapshot.
    loom_serve::testing::HttpClient overflow(
        Send("POST", "/v1/images/generations", inputs[0]));
    loom_serve_http_connection_t* overflow_connection = nullptr;
    ASSERT_NO_FATAL_FAILURE(accept(overflow.get(), &overflow_connection));
    EXPECT_EQ(Read(overflow.release()).find("HTTP/1.1 503"), 0u);
    const auto* control = interface.runtime.control;
    loom_runtime_service_observation_t observation = {};
    ASSERT_EQ(loom_runtime_service_read_observation(control, &observation),
              LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    EXPECT_EQ(observation.owned_request_count, 4u);
    EXPECT_EQ(observation.waiting_request_count, 4u);
    EXPECT_EQ(observation.pending_operation_count, 0u);
    bool progress = false;
    IREE_ASSERT_OK(loom_serve_image_service_progress(value, 0, &progress));
    EXPECT_FALSE(progress);
    EXPECT_STREQ(interface.snapshot(interface.runtime.self).phase, "idle");
    loom_runtime_service_offer_t preparation = {};
    ASSERT_EQ(loom_runtime_service_read_offer(control, 0, &preparation),
              LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    EXPECT_EQ(preparation.work, LOOM_RUNTIME_SERVICE_WORK_PREPARATION);
    loom_runtime_service_operation_key_t accepted;
    IREE_ASSERT_OK(loom_serve_image_service_try_schedule(value, preparation.key,
                                                         0, 0, &accepted));
    ASSERT_NE(accepted.serial, 0u);
    ASSERT_NO_FATAL_FAILURE(advance_until([&] {
      return std::string(interface.snapshot(interface.runtime.self).phase) ==
             "prepared";
    }));
    ASSERT_EQ(loom_runtime_service_read_observation(control, &observation),
              LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    EXPECT_EQ(observation.owned_request_count, 3u);
    EXPECT_EQ(observation.waiting_request_count, 3u);
    EXPECT_EQ(observation.pending_operation_count, 0u);
    EXPECT_EQ(interface.telemetry.counters->batches, 0u);
    loom_runtime_service_offer_t execution = {};
    ASSERT_EQ(loom_runtime_service_read_offer(control, 0, &execution),
              LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    ASSERT_EQ(execution.work, LOOM_RUNTIME_SERVICE_WORK_EXECUTION);
    loom_runtime_service_choice_t choice = {};
    ASSERT_EQ(
        loom_runtime_service_read_choice(control, execution.key, 0, &choice),
        LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    EXPECT_EQ(choice.request_count, 2u);
    EXPECT_EQ(choice.devices, 1u);
    EXPECT_EQ(choice.estimates, 0u);
    IREE_ASSERT_OK(loom_serve_image_service_progress(value, 0, &progress));
    EXPECT_FALSE(progress);
    loom_runtime_service_use_t use = {};
    EXPECT_EQ(loom_runtime_service_read_choice_use(control, execution.key, 0,
                                                   choice.publication, 0, &use),
              LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    IREE_ASSERT_OK(loom_serve_image_service_try_schedule(value, preparation.key,
                                                         0, 0, &accepted));
    EXPECT_EQ(accepted.serial, 0u);
    ASSERT_NO_FATAL_FAILURE(advance_until([&] {
      return std::string(interface.snapshot(interface.runtime.self).phase) ==
             "generating";
    }));
    const auto snapshot = interface.snapshot(interface.runtime.self);
    EXPECT_EQ(snapshot.active_requests, 2u);
    EXPECT_EQ(snapshot.queued_requests, 1u);
    EXPECT_EQ(interface.telemetry.counters->rejected, 2u);
    EXPECT_EQ(Read(clients[3].release()).find("HTTP/1.1 400"), 0u);
    // The later compatible item overtakes the incompatible geometry without
    // changing either item's private palette or terminal image dimensions.
    loom_service_request_snapshot_t request;
    ASSERT_TRUE(
        interface.inspection.request_at(interface.runtime.self, 0, &request));
    EXPECT_EQ(request.id, 2u);
    ASSERT_TRUE(
        interface.inspection.request_at(interface.runtime.self, 1, &request));
    EXPECT_EQ(request.id, 1u);
    ASSERT_TRUE(
        interface.inspection.request_at(interface.runtime.self, 2, &request));
    EXPECT_EQ(request.id, 3u);
    if (reset_phase) {
      ASSERT_NO_FATAL_FAILURE(advance_until([&] {
        return std::string(interface.snapshot(interface.runtime.self).phase) ==
               reset_phase;
      }));
      const linger reset = {1, 0};
      ASSERT_EQ(setsockopt(clients[0].get(), SOL_SOCKET, SO_LINGER, &reset,
                           sizeof(reset)),
                0);
      ASSERT_EQ(close(clients[0].release()), 0);
      iree_notification_await(
          notification,
          [](void* self) {
            return loom_serve_http_connection_failed(
                static_cast<loom_serve_http_connection_t*>(self));
          },
          connections[0], iree_infinite_timeout());
    }
    ASSERT_NO_FATAL_FAILURE(advance_until([&] {
      return interface.telemetry.counters->completed +
                 interface.telemetry.counters->cancelled ==
             3;
    }));
    if (!reset_phase) {
      EXPECT_NE(Read(clients[0].release())
                    .find("\"b64_json\":\"" +
                          EncodedPixels(2, 0, 7, 5, 0.25f, 0.5f) + "\""),
                std::string::npos);
    }
    EXPECT_NE(Read(clients[2].release())
                  .find("\"b64_json\":\"" +
                        EncodedPixels(3, 1, 5, 7, -0.5f, 0.75f) + "\""),
              std::string::npos);
    EXPECT_NE(Read(clients[1].release())
                  .find("\"b64_json\":\"" + EncodedPixels(2, 2, 6, 6) + "\""),
              std::string::npos);
    EXPECT_EQ(interface.telemetry.counters->completed, reset_phase ? 2u : 3u);
    EXPECT_EQ(interface.telemetry.counters->cancelled, reset_phase ? 1u : 0u);
    EXPECT_EQ(interface.telemetry.counters->failed, 0u);
    // Two cohorts, not three serial images, each using the same source command.
    EXPECT_EQ(interface.telemetry.counters->batches, GetParam() ? 4u : 2u);
    EXPECT_STREQ(interface.snapshot(interface.runtime.self).phase, "idle");
    ASSERT_EQ(loom_runtime_service_read_observation(control, &observation),
              LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    EXPECT_EQ(observation.pending_operation_count, 0u);
    EXPECT_EQ(observation.owned_request_count, 0u);
    EXPECT_EQ(observation.offer_count, 0u);
    loom_serve::testing::HttpClient stopped(
        Send("POST", "/v1/images/generations", inputs[0]));
    loom_serve_http_connection_t* stopped_connection = nullptr;
    const loom_serve_http_request_t* stopped_request = nullptr;
    IREE_ASSERT_OK(http_.TakeRequest(stopped.get(), &stopped_connection,
                                     &stopped_request));
    IREE_ASSERT_OK(interface.accept(interface.runtime.self, stopped_connection,
                                    stopped_request,
                                    LOOM_RUNTIME_SERVICE_FLAG_STOPPING));
    EXPECT_EQ(Read(stopped.release()).find("HTTP/1.1 503"), 0u);
    EXPECT_FALSE(
        interface.runtime.vtable->has_requests(interface.runtime.self));
    std::vector<loom_runtime_telemetry_record_t> prepared_records;
    uint64_t cursor = 0;
    const auto head =
        loom_runtime_telemetry_stream_head(interface.telemetry.stream);
    while (cursor < head) {
      loom_runtime_telemetry_record_t record;
      ASSERT_EQ(loom_runtime_telemetry_stream_read(interface.telemetry.stream,
                                                   &cursor, &record),
                LOOM_RUNTIME_TELEMETRY_READ_RECORD);
      if (std::strcmp(record.event->name, "image_prepared") == 0) {
        prepared_records.push_back(record);
      }
    }
    service.reset();
    IREE_ASSERT_OK(loom_serve_image_model_free(model_));
    model_ = nullptr;
    ASSERT_EQ(prepared_records.size(), 3u);
    for (const auto& record : prepared_records) {
      loom_service_image_prepared_t expected;
      std::memcpy(&expected, record.data.words, sizeof(expected));
      IREE_ASSERT_OK(record.event->read_fields(
          &record,
          {&expected, [](void* self, iree_host_size_t count,
                         const loom_runtime_telemetry_field_t* fields) {
             const auto& expected =
                 *static_cast<loom_service_image_prepared_t*>(self);
             EXPECT_EQ(count, 3u);
             if (count == 3) {
               EXPECT_EQ(fields[0].value.uint64_value, expected.stage);
               EXPECT_EQ(fields[1].kind, LOOM_RUNTIME_TELEMETRY_FIELD_INT64);
               EXPECT_EQ(fields[1].value.int64_value, expected.tag);
               EXPECT_EQ(fields[2].value.uint64_value, expected.input_bytes);
             }
             return iree_ok_status();
           }}));
      iree_string_builder_t output;
      iree_string_builder_initialize(host_allocator_, &output);
      IREE_ASSERT_OK(loom_serve_telemetry_record_format(0, &record, &output));
      iree_string_view_t tag;
      IREE_ASSERT_OK(iree_json_lookup_object_value(
          iree_string_builder_view(&output), IREE_SV("tag"), &tag));
      int64_t value = 0;
      IREE_ASSERT_OK(iree_json_parse_int64(tag, &value));
      EXPECT_EQ(value, expected.tag);
      iree_string_builder_deinitialize(&output);
    }
  }

  // Allocator for model, transport and expected encoded output.
  const iree_allocator_t host_allocator_ = iree_allocator_system();
  // Shared CPU/JIT resources outlive every model and service task.
  loom_runtime_host_t* host_ = nullptr;
  // Application group enclosing all model and device resources.
  loom_runtime_device_group_t* group_ = nullptr;
  // Borrowed device record within the application group.
  loom_runtime_device_t* device_ = nullptr;
  // Explicit storage owner selected independently from execution placement.
  loom_runtime_pool_t* storage_owner_ = nullptr;
  // Source and tokenizer package borrowed by cold and HTTP registration.
  std::string directory_;
  // Real source-JIT model with two kernels and reflected private workspace.
  loom_serve_image_model_t* model_ = nullptr;
  // Retained cohort and management observations, enclosing model teardown.
  loom_runtime::testing::ServiceControl scheduling_{17};
  // Production loopback carrier and request claim owner.
  loom_serve_http_server_t* server_ = nullptr;
  // Independent binding borrowing the HTTP owner and shared host pool.
  loom_serve_http_listener_t* listener_ = nullptr;
  // HTTP readiness and optional outer owner, joined before model destruction.
  loom_serve::testing::HttpHarness http_;
};

}  // namespace loom_serve::testing::image

#endif  // LOOM_SERVE_SERVICE_IMAGE_SERVICE_TEST_FIXTURE_H_
