// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/image/service.h"

#include <errno.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <tuple>

#include "experimental/loom_serve/runtime/device.h"
#include "iree/async/proactor.h"
#include "iree/base/threading/notification.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Only model math is replaced. Actual HAL queue dependencies, task ownership,
// proactor multi-waits, encoding, TCP and shutdown use the production stack.
struct ImageGate {
  enum { kSubmitted = 1, kDenied = 2, kDiscarded = 4 };

  ImageGate() { iree_notification_initialize(&notification); }
  ~ImageGate() { iree_notification_deinitialize(&notification); }

  void Publish(uint32_t event) {
    events.fetch_or(event, std::memory_order_release);
    iree_notification_post(&notification, IREE_ALL_WAITERS);
  }

  void Await(uint32_t event) {
    for (;;) {
      const auto token = iree_notification_prepare_wait(&notification);
      if (events.load(std::memory_order_acquire) & event) {
        iree_notification_cancel_wait(&notification);
        return;
      }
      iree_notification_commit_wait(&notification, token, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }

  // Dependency outcomes observed by the outer test thread without sleeps.
  std::atomic<uint32_t> events{0};
  // Outer test readiness; no scheduler worker waits here.
  iree_notification_t notification;
  // Optional shared admission slot, mutated only by serialized owner turns.
  bool* occupied = nullptr;
  // Borrowed model execution domain held by an explicit queue dependency.
  loom_serve_execution_t* execution = nullptr;
  // Borrowed valid CHW F32 result, outliving the joined service.
  float pixel[3] = {};
  // Successful completions consumed by the scheduler, read after service join.
  uint32_t completed = 0;
  // Request-selected dependency outcome, published by CPU preparation.
  bool admission_available = true;
  // Unsubmitted payloads discarded on denied admission, read after join.
  uint32_t discarded = 0;

  static iree_status_t Admit(void* self,
                             loom_serve_residency_admission_t* out_admission) {
    auto* gate = static_cast<ImageGate*>(self);
    const bool admitted =
        gate->admission_available && (!gate->occupied || !*gate->occupied);
    *out_admission = admitted ? LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED
                              : LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY;
    if (admitted) {
      if (gate->occupied) {
        *gate->occupied = true;
      }
    } else {
      gate->Publish(kDenied);
    }
    return iree_ok_status();
  }
  static iree_status_t Prepare(void* self,
                               const loom_serve_image_request_t* request) {
    static_cast<ImageGate*>(self)->admission_available = request->seed == 0;
    return iree_ok_status();
  }
  static void DiscardPreparedRequest(void* self) {
    auto* gate = static_cast<ImageGate*>(self);
    ++gate->discarded;
    gate->Publish(kDiscarded);
  }
  static iree_status_t Submit(void* self,
                              loom_serve_execution_frontier_t* out_frontier) {
    auto* gate = static_cast<ImageGate*>(self);
    *out_frontier = loom_serve_execution_frontier(gate->execution);
    gate->Publish(kSubmitted);
    return iree_ok_status();
  }
  static iree_const_byte_span_t Complete(void* self) {
    auto* gate = static_cast<ImageGate*>(self);
    ++gate->completed;
    if (gate->occupied) {
      *gate->occupied = false;
    }
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
      &gate,
      ImageGate::Prepare,
      ImageGate::DiscardPreparedRequest,
      ImageGate::Admit,
      ImageGate::Submit,
      ImageGate::Complete};
  const loom_serve_image_service_options_t options = {
      .name = IREE_SV("image-witness"),
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
  // Admission denial follows CPU preparation but precedes the held device
  // frontier. The same service must remain usable for the next request.
  const int denied = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  EXPECT_GE(denied, 0);
  EXPECT_EQ(connect(denied, reinterpret_cast<const sockaddr*>(address->storage),
                    address->length),
            0);
  const std::string denied_body =
      R"({"model":"image-witness","prompt":"denied","seed":1})";
  const std::string denied_request =
      "POST /v1/images/generations HTTP/1.1\r\nHost: local\r\n"
      "Content-Type: application/json\r\nContent-Length: " +
      std::to_string(denied_body.size()) + "\r\n\r\n" + denied_body;
  EXPECT_EQ(
      send(denied, denied_request.data(), denied_request.size(), MSG_NOSIGNAL),
      static_cast<ssize_t>(denied_request.size()));
  EXPECT_EQ(read_response(denied).find("HTTP/1.1 503"), 0u);
  EXPECT_EQ(close(denied), 0);
  const int client = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  EXPECT_GE(client, 0);
  EXPECT_EQ(connect(client, reinterpret_cast<const sockaddr*>(address->storage),
                    address->length),
            0);
  const std::string body =
      "{\"model\":\"image-witness\",\"prompt\":\"heartbeat witness\"}";
  const std::string request =
      "POST /v1/images/generations HTTP/1.1\r\nHost: local\r\n"
      "Content-Type: application/json\r\nContent-Length: " +
      std::to_string(body.size()) + "\r\n\r\n" + body;
  EXPECT_EQ(send(client, request.data(), request.size(), MSG_NOSIGNAL),
            static_cast<ssize_t>(request.size()));
  const std::string active =
      wait_event("\"event\":\"image_heartbeat\"", "\"phase\":\"generating\"");
  EXPECT_NE(active.find("\"active_request\":2"), std::string::npos);
  EXPECT_NE(active.find("\"completed\":0"), std::string::npos);

  for (const std::string target :
       {"/healthz", "/readyz", "/v1/status", "/metrics"}) {
    const int probe = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    EXPECT_GE(probe, 0);
    EXPECT_EQ(
        connect(probe, reinterpret_cast<const sockaddr*>(address->storage),
                address->length),
        0);
    const std::string probe_request =
        "GET " + target + " HTTP/1.1\r\nHost: local\r\n\r\n";
    EXPECT_EQ(
        send(probe, probe_request.data(), probe_request.size(), MSG_NOSIGNAL),
        static_cast<ssize_t>(probe_request.size()));
    const std::string response = read_response(probe);
    if (target == "/v1/status") {
      EXPECT_NE(
          response.find(
              R"({"object":"model","id":"image-witness","loom":{"kind":"image","endpoint":"/v1/images/generations","tokenize":false,"status":{"phase":"generating","queued_requests":0,"active_requests":1,"oldest_queued_age_seconds":0.000000000,"oldest_active_age_seconds":)"),
          std::string::npos);
      const std::string key = "\"oldest_active_age_seconds\":";
      const auto offset = response.find(key);
      if (offset != std::string::npos) {
        EXPECT_GT(std::strtod(response.c_str() + offset + key.size(), nullptr),
                  0.0);
      }
    } else if (target == "/metrics") {
      EXPECT_NE(response.find("loom_serve_model_active_requests{"
                              "model=\"image-witness\",kind=\"image\"} 1\n"),
                std::string::npos);
      EXPECT_NE(response.find("loom_serve_model_phase{model=\"image-witness\","
                              "kind=\"image\",phase=\"generating\"} 1\n"),
                std::string::npos);
      const std::string key =
          "loom_serve_model_oldest_active_age_seconds{model=\"image-witness\","
          "kind=\"image\"} ";
      const auto offset = response.find(key);
      EXPECT_NE(offset, std::string::npos);
      if (offset != std::string::npos) {
        EXPECT_GT(std::strtod(response.c_str() + offset + key.size(), nullptr),
                  0.0);
      }
    } else {
      EXPECT_NE(response.find("\"status\":\"ready\""), std::string::npos);
    }
    EXPECT_EQ(close(probe), 0);
  }

  if (completion_mode == CompletionMode::kShutdown) {
    EXPECT_EQ(kill(getpid(), SIGTERM), 0);
    const std::string draining =
        wait_event("\"event\":\"image_heartbeat\"", "\"stopping\":true");
    EXPECT_NE(draining.find("\"active_request\":2"), std::string::npos);
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
  EXPECT_EQ(gate.discarded, 1u);
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

enum class AdmissionMode { kResponse, kCancel, kShutdown };

class ImageAdmissionTest
    : public ::testing::TestWithParam<std::tuple<AdmissionMode, uint32_t>> {};

TEST_P(ImageAdmissionTest, PreparedPayloadWaitsWithoutPinningSharedCapacity) {
  const auto [mode, workers] = GetParam();
  IREE_ASSERT_OK(iree_async_signal_block_default());
  const auto allocator = iree_allocator_system();
  loom_serve_host_t* host = nullptr;
  IREE_ASSERT_OK(loom_serve_host_create(workers, &host, allocator));
  loom_serve_device_t* device = nullptr;
  const loom_serve_device_options_t device_options = {.uri = IREE_SV("amdgpu")};
  IREE_ASSERT_OK(
      loom_serve_device_create(host, &device_options, &device, allocator));
  loom_serve_execution_t* execution = nullptr;
  IREE_ASSERT_OK(loom_serve_device_create_execution(device, &execution));
  iree_hal_semaphore_t* predecessor = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &predecessor));
  uint64_t held = 1;
  IREE_ASSERT_OK(
      loom_serve_execution_barrier(execution, {1, &predecessor, &held}));
  bool occupied = false;
  ImageGate gates[2];
  loom_serve_image_service_t* images[2] = {};
  loom_serve_http_service_t services[2];
  for (size_t i = 0; i < 2; ++i) {
    gates[i].execution = execution;
    gates[i].occupied = &occupied;
    const loom_serve_image_generator_t generator = {
        &gates[i],        ImageGate::Prepare, ImageGate::DiscardPreparedRequest,
        ImageGate::Admit, ImageGate::Submit,  ImageGate::Complete};
    const loom_serve_image_service_options_t options = {
        .name = i == 0 ? IREE_SV("first") : IREE_SV("second"),
        .width = 1,
        .height = 1,
        .pending_capacity = 1,
        .heartbeat_interval = 0};
    IREE_ASSERT_OK(loom_serve_image_service_create(host, generator, &options,
                                                   &images[i], allocator));
    services[i] = loom_serve_image_service_interface(images[i]);
  }
  auto transport_options = loom_serve_http_server_options_default();
  transport_options.port = 0;
  loom_serve_http_server_t* server = nullptr;
  IREE_ASSERT_OK(
      loom_serve_http_server_create(&transport_options, &server, allocator));
  iree_status_t status = iree_ok_status();
  std::thread router([&] {
    status = loom_serve_http_router_run(loom_serve_host_executor(host), server,
                                        2, services, 8, allocator);
    for (auto* image : images) {
      loom_serve_image_service_destroy(image);
    }
  });
  auto send_request = [&](const std::string& model) {
    const int peer = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    EXPECT_GE(peer, 0);
    const auto* address = loom_serve_http_server_address(server);
    EXPECT_EQ(connect(peer, reinterpret_cast<const sockaddr*>(address->storage),
                      address->length),
              0);
    const std::string body =
        "{\"model\":\"" + model + "\",\"prompt\":\"shared admission\"}";
    const std::string request =
        model.empty()
            ? "GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n"
            : "POST /v1/images/generations HTTP/1.1\r\nHost: local\r\n"
              "Content-Type: application/json\r\nContent-Length: " +
                  std::to_string(body.size()) + "\r\n\r\n" + body;
    EXPECT_EQ(send(peer, request.data(), request.size(), MSG_NOSIGNAL),
              static_cast<ssize_t>(request.size()));
    return peer;
  };
  auto receive = [](int peer) {
    std::string response;
    char buffer[1024];
    ssize_t count;
    while ((count = recv(peer, buffer, sizeof(buffer), 0)) > 0) {
      response.append(buffer, count);
    }
    EXPECT_TRUE(count == 0 || errno == ECONNRESET);
    EXPECT_EQ(close(peer), 0);
    return response;
  };
  const int first = send_request("first");
  gates[0].Await(ImageGate::kSubmitted);
  int second = send_request("second");
  gates[1].Await(ImageGate::kDenied);
  EXPECT_NE(receive(send_request("")).find("\"status\":\"ready\""),
            std::string::npos);
  EXPECT_EQ(gates[1].events.load() & ImageGate::kSubmitted, 0u);
  if (mode == AdmissionMode::kCancel) {
    const linger reset = {1, 0};
    EXPECT_EQ(setsockopt(second, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
              0);
    EXPECT_EQ(close(second), 0);
    gates[1].Await(ImageGate::kDiscarded);
  } else if (mode == AdmissionMode::kShutdown) {
    EXPECT_EQ(kill(getpid(), SIGTERM), 0);
    EXPECT_TRUE(receive(first).empty());
  }
  IREE_EXPECT_OK(iree_hal_semaphore_signal(predecessor, 1, nullptr));
  if (mode != AdmissionMode::kShutdown) {
    EXPECT_NE(receive(first).find("\"b64_json\":\"iVBOR"), std::string::npos);
    if (mode == AdmissionMode::kCancel) {
      second = send_request("second");
    }
    EXPECT_NE(receive(second).find("\"b64_json\":\"iVBOR"), std::string::npos);
    EXPECT_EQ(kill(getpid(), SIGTERM), 0);
  } else {
    EXPECT_TRUE(receive(second).empty());
  }
  router.join();
  IREE_EXPECT_OK(status);
  EXPECT_FALSE(occupied);
  EXPECT_EQ(gates[0].completed, 1u);
  EXPECT_EQ(gates[1].completed, mode == AdmissionMode::kShutdown ? 0u : 1u);
  EXPECT_EQ(gates[1].discarded, mode == AdmissionMode::kResponse ? 0u : 1u);
  IREE_EXPECT_OK(loom_serve_http_server_destroy(server));
  IREE_EXPECT_OK(loom_serve_execution_drain(execution));
  loom_serve_execution_release(execution);
  iree_hal_semaphore_release(predecessor);
  IREE_EXPECT_OK(loom_serve_device_destroy(device));
  loom_serve_host_release(host);
}

INSTANTIATE_TEST_SUITE_P(
    Capacity, ImageAdmissionTest,
    ::testing::Combine(::testing::Values(AdmissionMode::kResponse,
                                         AdmissionMode::kCancel,
                                         AdmissionMode::kShutdown),
                       ::testing::Values(1u, 4u)));

}  // namespace
