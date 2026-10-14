// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <string>
#include <thread>
#include <tuple>
#include <utility>

#include "iree/async/proactor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/server/router.h"
#include "loom/serve/service/image/service.h"
#include "loom/serve/testing/http_harness.h"
#include "loom/serve/transport/http/listener.h"

namespace {

using loom_serve::testing::HttpClient;
using loom_serve::testing::HttpHarness;

// Only model math is replaced. Actual HAL queue dependencies, task ownership,
// proactor multi-waits, encoding, TCP and shutdown use the production stack.
struct ImageGate {
  enum { kSubmitted = 1, kDenied = 2, kDiscarded = 4 };

  void Publish(uint32_t event) {
    events.fetch_or(event, std::memory_order_release);
    http->Notify();
  }

  iree_status_t Await(int client, uint32_t event) {
    return http->Await(client, [&] {
      return (events.load(std::memory_order_acquire) & event) != 0;
    });
  }

  // Dependency outcomes observed by the outer test thread without sleeps.
  std::atomic<uint32_t> events{0};
  // Borrowed outer test owner observing dependency and transport readiness.
  HttpHarness* http = nullptr;
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
                               const loom_serve_image_request_t* request,
                               loom_serve_image_prepared_t* out_prepared) {
    static_cast<ImageGate*>(self)->admission_available = request->seed == 0;
    *out_prepared = {.stage = 0, .tag = 17, .input_bytes = 16};
    return iree_ok_status();
  }
  static void DiscardPreparedRequest(void* self) {
    auto* gate = static_cast<ImageGate*>(self);
    ++gate->discarded;
    gate->Publish(kDiscarded);
  }
  static iree_status_t Submit(void* self, loom_serve_image_submit_flags_t flags,
                              loom_serve_execution_frontier_t* out_frontier,
                              bool* out_complete) {
    auto* gate = static_cast<ImageGate*>(self);
    *out_frontier = loom_serve_execution_frontier(gate->execution);
    *out_complete = true;
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

// Owns every dependency of the router. Fatal assertions leave the outer
// fixture in charge of stop, queue release, join and destruction, in that
// order.
class ImageServiceFixture : public ::testing::Test {
 protected:
  void Initialize(uint32_t worker_count, bool independent) {
    IREE_ASSERT_OK(
        loom_serve_host_create(worker_count, host_allocator_, &host_));
    const loom_serve_device_options_t options[] = {{.uri = IREE_SV("amdgpu")},
                                                   {.uri = IREE_SV("amdgpu")}};
    IREE_ASSERT_OK(loom_serve_device_group_allocate(
        host_, independent ? 2 : 1, options, host_allocator_, &group_));
    for (iree_host_size_t i = 0; i < loom_serve_device_group_count(group_);
         ++i) {
      IREE_ASSERT_OK(loom_serve_device_create_execution(
          loom_serve_device_group_at(group_, i), &executions_[i]));
    }
    auto* device = loom_serve_device_group_at(group_, 0);
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &predecessor_));
    uint64_t held = 1;
    IREE_ASSERT_OK(loom_serve_execution_barrier(executions_[0],
                                                {1, &predecessor_, &held}));
    const auto transport_options = loom_serve_http_server_options_default();
    IREE_ASSERT_OK(loom_serve_http_server_allocate(
        loom_serve_host_proactor_pool(host_), &transport_options,
        host_allocator_, &server_));
    auto listener_options = loom_serve_http_listener_options_default();
    listener_options.address = IREE_SV("127.0.0.1:0");
    IREE_ASSERT_OK(loom_serve_http_listener_allocate(
        server_, &listener_options, {this, ListenerAllocatorControl},
        &listener_));
    IREE_ASSERT_OK(http_.Initialize(server_, listener_));
  }

  void TearDown() override {
    if (server_) {
      loom_serve_http_server_request_stop(server_);
    }
    ReleaseDependency();
    if (http_.joinable()) {
      ExpectCompletion(http_.Join());
    }
    RetireImages();
    IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
    loom_serve_http_server_free(server_);
    for (auto* execution : executions_) {
      if (execution) {
        ExpectCompletion(loom_serve_execution_drain(execution));
        loom_serve_execution_release(execution);
      }
    }
    iree_hal_semaphore_release(predecessor_);
    IREE_EXPECT_OK(loom_serve_device_group_free(group_));
    loom_serve_host_release(host_);
  }

  void ReleaseDependency(bool fail = false) {
    if (!predecessor_ || dependency_released_) {
      return;
    }
    dependency_released_ = true;
    device_failed_ = fail;
    if (fail) {
      iree_hal_semaphore_fail(
          predecessor_,
          iree_make_status(IREE_STATUS_ABORTED, "device witness"));
    } else {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(predecessor_, 1, nullptr));
    }
  }

  void ExpectCompletion(iree_status_t status) {
    if (device_failed_) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, status);
    } else {
      IREE_EXPECT_OK(status);
    }
  }

  void AddImage(size_t index, iree_string_view_t name,
                iree_duration_t heartbeat_interval) {
    gates_[index].http = &http_;
    const loom_serve_image_generator_t generator = {
        .self = &gates_[index],
        .prepare = ImageGate::Prepare,
        .discard_prepared_request = ImageGate::DiscardPreparedRequest,
        .try_admit = ImageGate::Admit,
        .submit = ImageGate::Submit,
        .complete = ImageGate::Complete,
        .capacity_domain = loom_serve_device_residency_cache(
            loom_serve_device_group_at(group_, executions_[1] ? index : 0))};
    const loom_serve_image_service_options_t options = {
        .name = name,
        .width = 1,
        .height = 1,
        .pending_capacity = 1,
        .heartbeat_interval = heartbeat_interval};
    IREE_ASSERT_OK(loom_serve_image_service_allocate(
        host_, generator, &options, nullptr, host_allocator_, &images_[index]));
    services_[index] = loom_serve_image_service_interface(images_[index]);
  }

  void Start(size_t count, iree_host_size_t request_quota) {
    http_.Start([this, count, request_quota]() -> iree_status_t {
      loom_serve_registry_t* registry = nullptr;
      IREE_RETURN_IF_ERROR(loom_serve_registry_allocate(
          count, services_, 0, {}, host_allocator_, &registry));
      iree_status_t status = loom_serve_router_run(
          loom_serve_host_executor(host_), server_, registry,
          {.request_quota = request_quota, .holding_capacity = 8},
          host_allocator_);
      return iree_status_join(status, loom_serve_registry_free(registry));
    });
  }

  void RetireImages() {
    for (auto*& image : images_) {
      loom_serve_image_service_free(std::exchange(image, nullptr));
    }
  }

  iree_status_t SendImage(const std::string& model, uint64_t seed,
                          HttpClient* out_client) {
    const std::string body =
        "{\"model\":\"" + model +
        "\",\"prompt\":\"heartbeat witness\",\"seed\":" + std::to_string(seed) +
        "}";
    return http_.Send(
        "POST /v1/images/generations HTTP/1.1\r\nHost: local\r\n"
        "Content-Type: application/json\r\nContent-Length: " +
            std::to_string(body.size()) + "\r\n\r\n" + body,
        out_client->put());
  }

  iree_status_t Get(const std::string& target, std::string* out_response) {
    HttpClient client;
    IREE_RETURN_IF_ERROR(http_.Send(
        "GET " + target +
            " HTTP/1.1\r\nHost: local\r\nX-Loom-Model: image-witness\r\n\r\n",
        client.put()));
    return http_.Read(client.release(), false, out_response);
  }

  iree_status_t WaitEvent(int client, const char* event, const char* field,
                          std::string* out_record) {
    for (;;) {
      loom_serve_telemetry_record_t value;
      if (loom_serve_telemetry_stream_read(services_[0].telemetry.stream,
                                           &cursor_, &value) ==
          LOOM_SERVE_TELEMETRY_READ_RECORD) {
        iree_string_builder_t output;
        iree_string_builder_initialize(host_allocator_, &output);
        iree_status_t status =
            loom_serve_telemetry_record_format(cursor_ - 1, &value, &output);
        const auto view = iree_string_builder_view(&output);
        const std::string record(view.data, view.size);
        iree_string_builder_deinitialize(&output);
        IREE_RETURN_IF_ERROR(status);
        if (record.find(event) != std::string::npos &&
            record.find(field) != std::string::npos) {
          *out_record = record;
          return iree_ok_status();
        }
      }
      IREE_RETURN_IF_ERROR(http_.CheckPending(client));
      // Telemetry readers have independent cursors, not producer wake
      // callbacks. A missing record never prevents observing socket or owner
      // failure.
      std::this_thread::yield();
    }
  }

  size_t GeneratedEvents(size_t index) {
    // Periodic records are disabled for the admission cases, so their whole
    // request history fits. Read through the actual live cursor protocol.
    const auto* stream = services_[index].telemetry.stream;
    const uint64_t head = loom_serve_telemetry_stream_head(stream);
    uint64_t cursor = 0;
    size_t count = 0;
    while (cursor < head) {
      loom_serve_telemetry_record_t record;
      const auto result =
          loom_serve_telemetry_stream_read(stream, &cursor, &record);
      EXPECT_EQ(result, LOOM_SERVE_TELEMETRY_READ_RECORD);
      if (result != LOOM_SERVE_TELEMETRY_READ_RECORD) {
        break;
      }
      count += std::string(record.event->name) == "image_generated";
    }
    return count;
  }

  static iree_status_t ListenerAllocatorControl(
      void* self, iree_allocator_command_t command, const void* parameters,
      void** inout_pointer) {
    auto* fixture = static_cast<ImageServiceFixture*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        fixture->fail_listener_allocations_.load()) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "image receive allocation witness");
    }
    return fixture->host_allocator_.ctl(fixture->host_allocator_.self, command,
                                        parameters, inout_pointer);
  }

  // Allocator shared by every fixture-owned production object.
  const iree_allocator_t host_allocator_ = iree_allocator_system();
  // CPU executor and asynchronous I/O infrastructure.
  loom_serve_host_t* host_ = nullptr;
  // One device group, optionally with a second independent placement.
  loom_serve_device_group_t* group_ = nullptr;
  // Independently drained execution timelines; the second may be absent.
  loom_serve_execution_t* executions_[2] = {};
  // Explicitly held device predecessor, released before teardown joins work.
  iree_hal_semaphore_t* predecessor_ = nullptr;
  // The predecessor was signaled or failed exactly once.
  bool dependency_released_ = false;
  // A deliberate device failure is expected from router and execution drain.
  bool device_failed_ = false;
  // Production HTTP owner, alive through service and listener retirement.
  loom_serve_http_server_t* server_ = nullptr;
  // Production loopback listener with a fault-injectable receive allocator.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Dependency, socket and owner observation, joined before freeing services.
  HttpHarness http_;
  // Rejects new receives after listener startup; freeing remains available.
  std::atomic<bool> fail_listener_allocations_{false};
  // Admission capacity mutated only by the serialized router owner.
  bool occupied_ = false;
  // Model dependencies, alive until the services and executor retire.
  ImageGate gates_[2];
  // Production services kept alive until all telemetry readers finish.
  loom_serve_image_service_t* images_[2] = {};
  // Borrowed registry interfaces, valid until RetireImages.
  loom_serve_http_service_t services_[2] = {};
  // Independent reader position for heartbeat and shutdown evidence.
  uint64_t cursor_ = 0;
};

enum class CompletionMode { kResponse, kShutdown, kDeviceFailure };

class ImageServiceTest : public ImageServiceFixture,
                         public ::testing::WithParamInterface<
                             std::tuple<CompletionMode, uint32_t>> {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(Initialize(std::get<1>(GetParam()), false));
    gates_[0].execution = executions_[0];
    ASSERT_NO_FATAL_FAILURE(AddImage(0, IREE_SV("image-witness"), 1000000));
    Start(1, 1);
  }

  void ExerciseCompletion(CompletionMode mode) {
    std::string response;
    IREE_ASSERT_OK(WaitEvent(-1, "\"event\":\"stream_begin\"", "", &response));
    // Denial follows preparation but precedes the held device frontier.
    HttpClient denied;
    IREE_ASSERT_OK(SendImage("image-witness", 1, &denied));
    IREE_ASSERT_OK(http_.Read(denied.release(), false, &response));
    ASSERT_EQ(response.find("HTTP/1.1 503"), 0u);

    HttpClient client;
    IREE_ASSERT_OK(SendImage("image-witness", 0, &client));
    IREE_ASSERT_OK(WaitEvent(client.get(), "\"event\":\"snapshot\"",
                             "\"phase\":\"generating\"", &response));
    EXPECT_NE(response.find("\"request\":2"), std::string::npos);
    EXPECT_NE(response.find("\"completed\":0"), std::string::npos);

    for (const std::string target :
         {"/healthz", "/readyz", "/v1/status", "/metrics", "/v1/requests",
          "/v1/sessions"}) {
      IREE_ASSERT_OK(Get(target, &response));
      if (target == "/v1/status") {
        EXPECT_NE(
            response.find(
                R"("input_modalities":["text"],"output_modalities":["image"],"loom":{"kind":"image","endpoint":"/v1/images/generations","tokenize":false,"status":{"phase":"generating","queued_requests":0,"active_requests":1,"oldest_queued_age_seconds":0.000000000,"oldest_active_age_seconds":)"),
            std::string::npos);
        const std::string key = "\"oldest_active_age_seconds\":";
        const auto offset = response.find(key);
        ASSERT_NE(offset, std::string::npos);
        EXPECT_GT(std::strtod(response.c_str() + offset + key.size(), nullptr),
                  0.0);
      } else if (target == "/v1/requests") {
        EXPECT_EQ(response.find("HTTP/1.1 200"), 0u);
        EXPECT_NE(
            response.find(
                R"("data":[{"id":2,"phase":"generating","session":null,)"),
            std::string::npos)
            << response;
        EXPECT_NE(response.find(R"("next_offset":null)"), std::string::npos);
        EXPECT_EQ(response.find("heartbeat witness"), std::string::npos);
      } else if (target == "/v1/sessions") {
        EXPECT_EQ(response.find("HTTP/1.1 501"), 0u);
      } else if (target == "/metrics") {
        EXPECT_NE(response.find("loom_serve_model_active_requests{"
                                "model=\"image-witness\",kind=\"image\"} 1\n"),
                  std::string::npos);
        EXPECT_NE(
            response.find("loom_serve_model_phase{model=\"image-witness\","
                          "kind=\"image\",phase=\"generating\"} 1\n"),
            std::string::npos);
        const std::string key =
            "loom_serve_model_oldest_active_age_seconds{model=\"image-"
            "witness\","
            "kind=\"image\"} ";
        const auto offset = response.find(key);
        ASSERT_NE(offset, std::string::npos);
        EXPECT_GT(std::strtod(response.c_str() + offset + key.size(), nullptr),
                  0.0);
      } else {
        EXPECT_NE(response.find("\"status\":\"ready\""), std::string::npos);
      }
    }

    if (mode == CompletionMode::kShutdown) {
      loom_serve_http_server_request_stop(server_);
      IREE_ASSERT_OK(WaitEvent(-1, "\"event\":\"image_shutdown\"",
                               "\"request\":2", &response));
    }
    ReleaseDependency(mode == CompletionMode::kDeviceFailure);
    if (mode == CompletionMode::kResponse) {
      IREE_ASSERT_OK(http_.Read(client.release(), false, &response));
      EXPECT_NE(response.find("200 OK"), std::string::npos);
      EXPECT_NE(response.find("\"b64_json\":\"iVBOR"), std::string::npos);
      loom_serve_http_server_request_stop(server_);
    }
    ExpectCompletion(http_.Join());
    const auto counters = *services_[0].telemetry.counters;
    EXPECT_EQ(counters.requests, 2u);
    EXPECT_EQ(counters.rejected, 1u);
    EXPECT_EQ(counters.completed, mode == CompletionMode::kResponse ? 1u : 0u);
    EXPECT_EQ(counters.cancelled, mode == CompletionMode::kShutdown ? 1u : 0u);
    EXPECT_EQ(counters.failed, device_failed_ ? 1u : 0u);
    EXPECT_EQ(counters.batches, device_failed_ ? 0u : 1u);
    EXPECT_EQ(counters.input_tokens + counters.output_tokens, 0u);
    EXPECT_EQ(gates_[0].completed, device_failed_ ? 0u : 1u);
    EXPECT_EQ(gates_[0].discarded, 1u);
  }
};

TEST_P(ImageServiceTest, HttpAndHeartbeatStayLiveDuringDeviceWait) {
  ExerciseCompletion(std::get<0>(GetParam()));
}

TEST_P(ImageServiceTest, TransportFailureReleasesHeldDeviceDependency) {
  fail_listener_allocations_.store(true);
  ::testing::TestPartResultArray failures;
  {
    ::testing::ScopedFakeTestPartResultReporter reporter(
        ::testing::ScopedFakeTestPartResultReporter::
            INTERCEPT_ONLY_CURRENT_THREAD,
        &failures);
    ExerciseCompletion(CompletionMode::kResponse);
  }
  ASSERT_EQ(failures.size(), 1);
  EXPECT_TRUE(failures.GetTestPartResult(0).fatally_failed());
  EXPECT_NE(std::string(failures.GetTestPartResult(0).message())
                .find("image receive allocation witness"),
            std::string::npos);
  EXPECT_TRUE(http_.failed());
  EXPECT_EQ(gates_[0].events.load() & ImageGate::kSubmitted, 0u);
}

TEST_P(ImageServiceTest, TransportFailureTerminatesHeartbeatWait) {
  fail_listener_allocations_.store(true);
  ::testing::TestPartResultArray failures;
  {
    ::testing::ScopedFakeTestPartResultReporter reporter(
        ::testing::ScopedFakeTestPartResultReporter::
            INTERCEPT_ONLY_CURRENT_THREAD,
        &failures);
    [&] {
      HttpClient client;
      IREE_ASSERT_OK(SendImage("image-witness", 0, &client));
      std::string record;
      IREE_ASSERT_OK(WaitEvent(client.get(), "\"event\":\"snapshot\"",
                               "\"phase\":\"generating\"", &record));
    }();
  }
  ASSERT_EQ(failures.size(), 1);
  EXPECT_TRUE(failures.GetTestPartResult(0).fatally_failed());
  EXPECT_NE(std::string(failures.GetTestPartResult(0).message())
                .find("image receive allocation witness"),
            std::string::npos);
  EXPECT_TRUE(http_.failed());
  EXPECT_EQ(gates_[0].events.load() & ImageGate::kSubmitted, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    Completion, ImageServiceTest,
    ::testing::Combine(::testing::Values(CompletionMode::kResponse,
                                         CompletionMode::kShutdown,
                                         CompletionMode::kDeviceFailure),
                       ::testing::Values(1u, 4u)));

enum class AdmissionMode { kResponse, kCancel, kShutdown, kIndependent };

class ImageAdmissionTest : public ImageServiceFixture,
                           public ::testing::WithParamInterface<
                               std::tuple<AdmissionMode, uint32_t>> {
 protected:
  void SetUp() override {
    const bool independent =
        std::get<0>(GetParam()) == AdmissionMode::kIndependent;
    ASSERT_NO_FATAL_FAILURE(Initialize(std::get<1>(GetParam()), independent));
    if (independent) {
      auto* first = loom_serve_device_group_at(group_, 0);
      auto* second = loom_serve_device_group_at(group_, 1);
      EXPECT_EQ(loom_serve_device_group_count(group_), 2u);
      EXPECT_EQ(loom_serve_device_group(first),
                loom_serve_device_group(second));
      EXPECT_NE(loom_serve_device_residency_cache(first),
                loom_serve_device_residency_cache(second));
    }
    for (size_t i = 0; i < 2; ++i) {
      const bool separate = i == 1 && independent;
      gates_[i].execution = separate ? executions_[1] : executions_[0];
      gates_[i].occupied = separate ? nullptr : &occupied_;
      ASSERT_NO_FATAL_FAILURE(
          AddImage(i, i == 0 ? IREE_SV("first") : IREE_SV("second"), 0));
    }
    Start(2, 8);
  }

  void ExerciseAdmission(AdmissionMode mode) {
    HttpClient first;
    IREE_ASSERT_OK(SendImage("first", 0, &first));
    IREE_ASSERT_OK(gates_[0].Await(first.get(), ImageGate::kSubmitted));
    HttpClient second;
    IREE_ASSERT_OK(SendImage(
        "second", mode == AdmissionMode::kIndependent ? 1 : 0, &second));
    IREE_ASSERT_OK(gates_[1].Await(second.get(), ImageGate::kDenied));
    std::string response;
    IREE_ASSERT_OK(Get("/healthz", &response));
    EXPECT_NE(response.find("\"status\":\"ready\""), std::string::npos);
    EXPECT_EQ(gates_[1].events.load() & ImageGate::kSubmitted, 0u);
    if (mode == AdmissionMode::kIndependent) {
      // Denial cannot be repaired by the unrelated held domain. Both this
      // response and the next PNG must precede that domain's completion.
      IREE_ASSERT_OK(http_.Read(second.release(), false, &response));
      EXPECT_EQ(response.find("HTTP/1.1 503"), 0u);
      IREE_ASSERT_OK(SendImage("second", 0, &second));
      IREE_ASSERT_OK(http_.Read(second.release(), false, &response));
      EXPECT_NE(response.find("\"b64_json\":\"iVBOR"), std::string::npos);
    } else if (mode == AdmissionMode::kCancel) {
      const linger reset = {1, 0};
      ASSERT_EQ(setsockopt(second.get(), SOL_SOCKET, SO_LINGER, &reset,
                           sizeof(reset)),
                0);
      ASSERT_EQ(close(second.release()), 0);
      IREE_ASSERT_OK(gates_[1].Await(-1, ImageGate::kDiscarded));
    } else if (mode == AdmissionMode::kShutdown) {
      loom_serve_http_server_request_stop(server_);
      IREE_ASSERT_OK(http_.Read(first.release(), true, &response));
      EXPECT_TRUE(response.empty());
    }
    EXPECT_EQ(GeneratedEvents(0), 0u);
    ReleaseDependency();
    if (mode != AdmissionMode::kShutdown) {
      IREE_ASSERT_OK(http_.Read(first.release(), false, &response));
      EXPECT_NE(response.find("\"b64_json\":\"iVBOR"), std::string::npos);
      if (mode == AdmissionMode::kCancel) {
        IREE_ASSERT_OK(SendImage("second", 0, &second));
      }
      if (mode != AdmissionMode::kIndependent) {
        IREE_ASSERT_OK(http_.Read(second.release(), false, &response));
        EXPECT_NE(response.find("\"b64_json\":\"iVBOR"), std::string::npos);
      }
      loom_serve_http_server_request_stop(server_);
    }
    IREE_ASSERT_OK(http_.Join());
    EXPECT_FALSE(occupied_);
    EXPECT_EQ(gates_[0].completed, 1u);
    EXPECT_EQ(gates_[1].completed, mode == AdmissionMode::kShutdown ? 0u : 1u);
    EXPECT_EQ(gates_[1].discarded, mode == AdmissionMode::kResponse ? 0u : 1u);
    EXPECT_EQ(GeneratedEvents(0), 1u);
    EXPECT_EQ(GeneratedEvents(1), mode == AdmissionMode::kShutdown ? 0u : 1u);
    // Pending claims belong to the service until its outer owner retires it.
    // Join and retirement precede waiting for the queued peer's empty close.
    RetireImages();
    if (mode == AdmissionMode::kShutdown) {
      IREE_ASSERT_OK(http_.Read(second.release(), true, &response));
      EXPECT_TRUE(response.empty());
    }
  }
};

TEST_P(ImageAdmissionTest, PreparedPayloadWaitsWithoutPinningSharedCapacity) {
  ExerciseAdmission(std::get<0>(GetParam()));
}

TEST_P(ImageAdmissionTest, TransportFailureTerminatesDependencyWait) {
  fail_listener_allocations_.store(true);
  ::testing::TestPartResultArray failures;
  {
    ::testing::ScopedFakeTestPartResultReporter reporter(
        ::testing::ScopedFakeTestPartResultReporter::
            INTERCEPT_ONLY_CURRENT_THREAD,
        &failures);
    ExerciseAdmission(std::get<0>(GetParam()));
  }
  ASSERT_EQ(failures.size(), 1);
  EXPECT_TRUE(failures.GetTestPartResult(0).fatally_failed());
  EXPECT_NE(std::string(failures.GetTestPartResult(0).message())
                .find("image receive allocation witness"),
            std::string::npos);
  EXPECT_TRUE(http_.failed());
  EXPECT_EQ(gates_[0].events.load() & ImageGate::kSubmitted, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    Capacity, ImageAdmissionTest,
    ::testing::Combine(::testing::Values(AdmissionMode::kResponse,
                                         AdmissionMode::kCancel,
                                         AdmissionMode::kShutdown,
                                         AdmissionMode::kIndependent),
                       ::testing::Values(1u, 4u)));

}  // namespace
