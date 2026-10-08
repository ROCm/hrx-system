// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/image/service.h"

#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <string>
#include <thread>
#include <vector>

#include "experimental/loom_serve/image/model.h"
#include "experimental/loom_serve/image/output.h"
#include "iree/async/proactor.h"
#include "iree/base/internal/base64.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

IREE_FLAG(string, model_source, "", "Deterministic image bootstrap source.");

namespace {

class SourceImageServiceTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  static constexpr uint32_t kWidth = 7;
  static constexpr uint32_t kHeight = 5;
  static constexpr size_t kPixels = kWidth * kHeight;

  void SetUp() override {
    IREE_ASSERT_OK(iree_async_signal_block_default());
    IREE_ASSERT_OK(loom_serve_host_create(GetParam(), &host_, allocator_));
    const loom_serve_device_options_t device_options = {.uri =
                                                            IREE_SV("amdgpu")};
    IREE_ASSERT_OK(
        loom_serve_device_create(host_, &device_options, &device_, allocator_));
    const std::string source = FLAG_model_source;
    const std::string directory = source.substr(0, source.find_last_of('/'));
    const loom_serve_image_model_options_t model_options = {
        .source_directory = iree_make_cstring_view(directory.c_str()),
        .checkpoint_directory = iree_make_cstring_view(directory.c_str()),
        .height = kHeight,
        .width = kWidth,
        .text_tokens = 32,
    };
    IREE_ASSERT_OK(loom_serve_image_model_create(device_, &model_options,
                                                 &model_, allocator_));
    auto options = loom_serve_http_server_options_default();
    options.port = 0;
    IREE_ASSERT_OK(
        loom_serve_http_server_create(&options, &server_, allocator_));
  }

  void TearDown() override {
    if (service_.joinable()) {
      EXPECT_EQ(kill(getpid(), SIGTERM), 0);
      service_.join();
      IREE_EXPECT_OK(status_);
    }
    IREE_EXPECT_OK(loom_serve_http_server_destroy(server_));
    IREE_EXPECT_OK(loom_serve_image_model_destroy(model_));
    IREE_EXPECT_OK(loom_serve_device_destroy(device_));
    loom_serve_host_release(host_);
  }

  void Start() {
    service_ = std::thread([this] {
      const loom_serve_image_generator_t generator =
          loom_serve_image_model_generator(model_);
      const loom_serve_image_service_options_t options = {
          .name = loom_serve_image_model_name(model_),
          .width = kWidth,
          .height = kHeight,
          .pending_capacity = 8,
          .heartbeat_interval = 0,
      };
      status_ = loom_serve_image_service_run(host_, generator, server_,
                                             &options, allocator_);
    });
  }

  int Send(const std::string& method, const std::string& target,
           const std::string& body = "") {
    const int client = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    EXPECT_GE(client, 0);
    const auto* address = loom_serve_http_server_address(server_);
    EXPECT_EQ(
        connect(client, reinterpret_cast<const sockaddr*>(address->storage),
                address->length),
        0);
    const std::string request =
        method + " " + target +
        " HTTP/1.1\r\nHost: local\r\n"
        "Content-Type: application/json\r\nContent-Length: " +
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

  // Allocator for model, transport and expected encoded output.
  const iree_allocator_t allocator_ = iree_allocator_system();
  // Shared CPU/JIT resources outlive every model and service task.
  loom_serve_host_t* host_ = nullptr;
  // Actual qualified device owner, not a mock model implementation.
  loom_serve_device_t* device_ = nullptr;
  // Real source-JIT model with two kernels and reflected private workspace.
  loom_serve_image_model_t* model_ = nullptr;
  // Production loopback carrier and request claim owner.
  loom_serve_http_server_t* server_ = nullptr;
  // Outer application join; model work uses the shared task executor.
  std::thread service_;
  // Terminal service result read only after joining the outer thread.
  iree_status_t status_ = iree_ok_status();
};

TEST_P(SourceImageServiceTest, SourcePixelsAndHttpPngPreserveRequestIdentity) {
  IREE_ASSERT_OK(
      loom_serve_image_model_prepare_request(model_, IREE_SV("x"), 0, 1.0f));
  loom_serve_image_model_discard_prepared_request(model_);
  // Verify source arithmetic independently of the image encoder.
  for (uint64_t seed : {0u, 1u}) {
    iree_const_byte_span_t rgb;
    IREE_ASSERT_OK(loom_serve_image_model_generate(model_, IREE_SV("aba"), seed,
                                                   1.0f, &rgb));
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
              R"({"object":"model","id":"fixture-image","loom":{"kind":"image","endpoint":"/v1/images/generations","tokenize":true}})"),
      std::string::npos);
  EXPECT_NE(
      Read(Send("GET", "/v1/status"))
          .find(
              R"("status":{"phase":"idle","queued_requests":0,"active_requests":0,"oldest_queued_age_seconds":0.000000000,"oldest_active_age_seconds":0.000000000})"),
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
    const auto pixels = Pixels(std::string(prompts[i]).size(), i);
    iree_byte_span_t png;
    IREE_ASSERT_OK(loom_serve_image_encode_rgb_f32_png(
        kWidth, kHeight,
        iree_make_const_byte_span(pixels.data(), sizeof(pixels)), &png,
        allocator_));
    std::string encoded(iree_base64_encoded_size(png.data_length), '\0');
    size_t length = 0;
    IREE_ASSERT_OK(iree_base64_encode(
        iree_make_const_byte_span(png.data, png.data_length),
        iree_make_mutable_string_view(encoded.data(), encoded.size()),
        &length));
    encoded.resize(length);
    iree_allocator_free(allocator_, png.data);
    const std::string response = Read(clients[i]);
    EXPECT_EQ(response.find("HTTP/1.1 200"), 0u);
    EXPECT_NE(response.find("\"b64_json\":\"" + encoded + "\""),
              std::string::npos);
  }
  EXPECT_NE(Read(Send("GET", "/healthz")).find("\"status\":\"ready\""),
            std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(Workers, SourceImageServiceTest,
                         ::testing::Values(1u, 4u));

}  // namespace
