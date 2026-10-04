// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/image_service.h"

#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>

#include "iree/async/proactor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Only the expensive model dependency is gated. Admission, worker ownership,
// heartbeats, signal delivery and shutdown use the actual service and carrier.
struct ImageGate {
  // Protects completion permission shared with the model worker.
  std::mutex mutex;
  // Releases the accepted invocation after live shutdown has been observed.
  std::condition_variable condition;
  // Model completion permission, protected by mutex.
  bool released = false;
  // Borrowed valid CHW F32 result, outliving the joined service.
  float pixel[3] = {};

  static iree_status_t Generate(void* self, const loom_serve_image_request_t*,
                                iree_const_byte_span_t* out_rgb) {
    auto* gate = static_cast<ImageGate*>(self);
    std::unique_lock<std::mutex> lock(gate->mutex);
    gate->condition.wait(lock, [&] { return gate->released; });
    *out_rgb = iree_make_const_byte_span(gate->pixel, sizeof(gate->pixel));
    return iree_ok_status();
  }
};

TEST(ImageServiceTest, HeartbeatRemainsLiveThroughJoinedShutdown) {
  IREE_ASSERT_OK(iree_async_signal_block_default());
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

  ImageGate gate;
  const loom_serve_image_generator_t generator = {&gate, ImageGate::Generate};
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
    service_status = loom_serve_image_service_run(generator, server, &options,
                                                  iree_allocator_system());
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

  EXPECT_EQ(kill(getpid(), SIGTERM), 0);
  const std::string draining =
      wait_event("\"event\":\"image_heartbeat\"", "\"stopping\":true");
  EXPECT_NE(draining.find("\"active_request\":1"), std::string::npos);
  EXPECT_NE(draining.find("\"completed\":0"), std::string::npos);
  {
    std::lock_guard<std::mutex> lock(gate.mutex);
    gate.released = true;
  }
  gate.condition.notify_one();
  EXPECT_FALSE(
      wait_event("\"event\":\"image_stopped\"", "\"completed\":1").empty());
  service.join();
  IREE_EXPECT_OK(service_status);
  EXPECT_EQ(close(client), 0);
  IREE_EXPECT_OK(loom_serve_http_server_destroy(server));
  EXPECT_EQ(dup2(saved_stderr, STDERR_FILENO), STDERR_FILENO);
  EXPECT_EQ(close(saved_stderr), 0);
  EXPECT_EQ(fclose(reader), 0);
}

}  // namespace
