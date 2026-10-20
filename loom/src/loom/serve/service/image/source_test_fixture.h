// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_IMAGE_SOURCE_TEST_FIXTURE_H_
#define LOOM_SERVE_SERVICE_IMAGE_SOURCE_TEST_FIXTURE_H_

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "iree/base/internal/base64.h"
#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/server/model_registration.h"
#include "loom/serve/server/resources.h"
#include "loom/serve/server/router.h"
#include "loom/serve/service/image/model.h"
#include "loom/serve/service/image/output.h"
#include "loom/serve/service/image/service.h"
#include "loom/serve/testing/http_harness.h"
#include "loom/serve/testing/model.h"
#include "loom/serve/testing/service_application.h"
#include "loom/serve/transport/http/listener.h"

namespace loom_serve::testing::image {

struct loom_serve_test_image_request_t {
  ~loom_serve_test_image_request_t() {
    loom_serve_image_request_deinitialize(&value);
  }

  iree_status_t Initialize(iree_string_view_t body, uint32_t width,
                           uint32_t height) {
    return loom_serve_image_request_initialize(body, IREE_SV("fixture-image"),
                                               width, height,
                                               iree_allocator_system(), &value);
  }

  // Decoded input remains live until source cohort capture has finished.
  loom_serve_image_request_t value = {};
};

class SourceImageTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  static constexpr uint32_t kWidth = 7;
  static constexpr uint32_t kHeight = 5;
  static constexpr size_t kPixels = kWidth * kHeight;

  void SetUp() override {
    const char* source_path = std::getenv("LOOM_SERVE_TEST_MODEL_SOURCE");
    ASSERT_NE(source_path, nullptr);
    auto application_options = loom_serve_test_application_options_default();
    // Activation and the first numerical part must share this single slot.
    application_options.max_active = 1;
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

  iree_status_t PrepareSingle(iree_string_view_t body, uint32_t width,
                              uint32_t height,
                              loom_serve_image_prepared_t* out_prepared) {
    loom_serve_test_image_request_t input;
    IREE_RETURN_IF_ERROR(input.Initialize(body, width, height));
    loom_serve_prepared_request_t request = {};
    IREE_RETURN_IF_ERROR(
        loom_serve_image_model_prepare_request(model_, &input.value, &request));
    const iree_host_size_t output_length =
        (iree_host_size_t)width * height * 12;
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
                                   size_t count = kPixels,
                                   float red_strength = 1.0f,
                                   float green_strength = 1.0f) {
    std::vector<float> pixels(count * 3);
    for (size_t i = 0; i < count; ++i) {
      pixels[i] = (seed & 1 ? 1.0f : -1.0f) * red_strength;
      pixels[count + i] = (tokens & 1 ? 1.0f : -1.0f) * green_strength;
      pixels[2 * count + i] = i & 1 ? 1.0f : -1.0f;
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

}  // namespace loom_serve::testing::image

#endif  // LOOM_SERVE_SERVICE_IMAGE_SOURCE_TEST_FIXTURE_H_
