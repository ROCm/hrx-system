// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/image/service.h"

#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <string>
#include <thread>
#include <tuple>

#include "experimental/loom_serve/runtime/device.h"
#include "iree/async/proactor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Only model math is replaced. Actual HAL queue dependencies, task ownership,
// proactor multi-waits, encoding, TCP and shutdown use the production stack.
struct ImageGate {
  // Borrowed model execution domain held by an explicit queue dependency.
  loom_serve_execution_t* execution = nullptr;
  // Borrowed valid CHW F32 result, outliving the joined service.
  float pixel[3] = {};
  // Successful completions consumed by the scheduler, read after service join.
  uint32_t completed = 0;

  static iree_status_t Acquire(void*, bool* out_admitted) {
    *out_admitted = true;
    return iree_ok_status();
  }
  static iree_status_t Prepare(void*, const loom_serve_image_request_t*) {
    return iree_ok_status();
  }
  static iree_status_t Submit(void* self,
                              loom_serve_execution_frontier_t* out_frontier) {
    *out_frontier =
        loom_serve_execution_frontier(static_cast<ImageGate*>(self)->execution);
    return iree_ok_status();
  }
  static iree_const_byte_span_t Complete(void* self) {
    auto* gate = static_cast<ImageGate*>(self);
    ++gate->completed;
    return iree_make_const_byte_span(gate->pixel, sizeof(gate->pixel));
  }
};

enum class CompletionMode { kResponse, kShutdown, kDeviceFailure };

class ImageServiceTest
    : public ::testing::TestWithParam<std::tuple<CompletionMode, uint32_t>> {};

TEST_P(ImageServiceTest, HttpAndHeartbeatStayLiveDuringDeviceWait) {
  const auto [completion_mode, worker_count] = GetParam();
  IREE_ASSERT_OK(iree_async_signal_block_default());
  loom_serve_host_t* host = nullptr;
  IREE_ASSERT_OK(
      loom_serve_host_create(worker_count, &host, iree_allocator_system()));
  loom_serve_device_t* device = nullptr;
  const loom_serve_device_options_t device_options = {.uri = IREE_SV("amdgpu")};
  IREE_ASSERT_OK(loom_serve_device_create(host, &device_options, &device,
                                          iree_allocator_system()));
  ImageGate gate;
  IREE_ASSERT_OK(loom_serve_device_create_execution(device, &gate.execution));
  iree_hal_semaphore_t* predecessor = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &predecessor));
  uint64_t held = 1;
  IREE_ASSERT_OK(
      loom_serve_execution_barrier(gate.execution, {1, &predecessor, &held}));
  auto transport_options = loom_serve_http_server_options_default();
  transport_options.port = 0;
  loom_serve_http_server_t* server = nullptr;
  IREE_ASSERT_OK(loom_serve_http_server_create(&transport_options, &server,
                                               iree_allocator_system()));

  int logs[2];
  ASSERT_EQ(pipe(logs), 0);
  const int saved_stderr = dup(STDERR_FILENO);
  ASSERT_GE(saved_stderr, 0);
  ASSERT_EQ(dup2(logs[1], STDERR_FILENO), STDERR_FILENO);
  ASSERT_EQ(close(logs[1]), 0);
  FILE* reader = fdopen(logs[0], "r");
  ASSERT_NE(reader, nullptr);

  const loom_serve_image_generator_t generator = {
      &gate, ImageGate::Acquire, ImageGate::Prepare, ImageGate::Submit,
      ImageGate::Complete};
  const loom_serve_image_service_options_t options = {
      .model = IREE_SV("image-witness"),
      .width = 1,
      .height = 1,
      .adapter_enabled = false,
      .pending_capacity = 1,
      .heartbeat_interval = 1000000,
  };
  iree_status_t service_status = iree_ok_status();
  std::thread service([&] {
    service_status = loom_serve_image_service_run(
        host, generator, server, &options, iree_allocator_system());
  });
  auto wait_event = [&](const char* event, const char* field) {
    char line[2048];
    while (fgets(line, sizeof(line), reader)) {
      const std::string record(line);
      if (record.find(event) != std::string::npos &&
          record.find(field) != std::string::npos) {
        return record;
      }
    }
    return std::string();
  };
  EXPECT_FALSE(wait_event("\"event\":\"image_ready\"", "").empty());
  const auto* address = loom_serve_http_server_address(server);
  const int client = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  EXPECT_GE(client, 0);
  EXPECT_EQ(connect(client, reinterpret_cast<const sockaddr*>(address->storage),
                    address->length),
            0);
  const std::string body = "{\"prompt\":\"heartbeat witness\"}";
  const std::string request =
      "POST /v1/images/generations HTTP/1.1\r\nHost: local\r\n"
      "Content-Type: application/json\r\nContent-Length: " +
      std::to_string(body.size()) + "\r\n\r\n" + body;
  EXPECT_EQ(send(client, request.data(), request.size(), MSG_NOSIGNAL),
            static_cast<ssize_t>(request.size()));
  const std::string active =
      wait_event("\"event\":\"image_heartbeat\"", "\"phase\":\"generating\"");
  EXPECT_NE(active.find("\"active_request\":1"), std::string::npos);
  EXPECT_NE(active.find("\"completed\":0"), std::string::npos);

  const int health = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  EXPECT_GE(health, 0);
  EXPECT_EQ(connect(health, reinterpret_cast<const sockaddr*>(address->storage),
                    address->length),
            0);
  const std::string health_request =
      "GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n";
  EXPECT_EQ(
      send(health, health_request.data(), health_request.size(), MSG_NOSIGNAL),
      static_cast<ssize_t>(health_request.size()));
  auto read_response = [](int socket) {
    std::string response;
    char buffer[4096];
    ssize_t count = 0;
    while ((count = recv(socket, buffer, sizeof(buffer), 0)) > 0) {
      response.append(buffer, count);
    }
    EXPECT_EQ(count, 0);
    return response;
  };
  EXPECT_NE(read_response(health).find("\"active\":1"), std::string::npos);
  EXPECT_EQ(close(health), 0);

  if (completion_mode == CompletionMode::kShutdown) {
    EXPECT_EQ(kill(getpid(), SIGTERM), 0);
    const std::string draining =
        wait_event("\"event\":\"image_heartbeat\"", "\"stopping\":true");
    EXPECT_NE(draining.find("\"active_request\":1"), std::string::npos);
    EXPECT_NE(draining.find("\"completed\":0"), std::string::npos);
  }
  if (completion_mode == CompletionMode::kDeviceFailure) {
    iree_hal_semaphore_fail(
        predecessor, iree_make_status(IREE_STATUS_ABORTED, "device witness"));
  } else {
    IREE_EXPECT_OK(iree_hal_semaphore_signal(predecessor, 1, nullptr));
  }
  if (completion_mode == CompletionMode::kResponse) {
    const std::string response = read_response(client);
    EXPECT_NE(response.find("200 OK"), std::string::npos);
    EXPECT_NE(response.find("\"b64_json\":\"iVBOR"), std::string::npos);
    EXPECT_EQ(kill(getpid(), SIGTERM), 0);
  }
  const bool failed = completion_mode == CompletionMode::kDeviceFailure;
  EXPECT_FALSE(wait_event("\"event\":\"image_stopped\"",
                          failed ? "\"completed\":0" : "\"completed\":1")
                   .empty());
  service.join();
  if (failed) {
    EXPECT_EQ(iree_status_code(service_status), IREE_STATUS_ABORTED);
    iree_status_free(service_status);
  } else {
    IREE_EXPECT_OK(service_status);
  }
  EXPECT_EQ(gate.completed, failed ? 0u : 1u);
  EXPECT_EQ(close(client), 0);
  IREE_EXPECT_OK(loom_serve_http_server_destroy(server));
  EXPECT_EQ(dup2(saved_stderr, STDERR_FILENO), STDERR_FILENO);
  EXPECT_EQ(close(saved_stderr), 0);
  EXPECT_EQ(fclose(reader), 0);
  iree_status_t drain_status = loom_serve_execution_drain(gate.execution);
  if (failed) {
    EXPECT_EQ(iree_status_code(drain_status), IREE_STATUS_ABORTED);
    iree_status_free(drain_status);
  } else {
    IREE_EXPECT_OK(drain_status);
  }
  loom_serve_execution_release(gate.execution);
  iree_hal_semaphore_release(predecessor);
  IREE_EXPECT_OK(loom_serve_device_destroy(device));
  loom_serve_host_release(host);
}

INSTANTIATE_TEST_SUITE_P(
    Completion, ImageServiceTest,
    ::testing::Combine(::testing::Values(CompletionMode::kResponse,
                                         CompletionMode::kShutdown,
                                         CompletionMode::kDeviceFailure),
                       ::testing::Values(1u, 4u)));

}  // namespace
