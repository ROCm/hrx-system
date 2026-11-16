// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/socket.h>
#include <unistd.h>

#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "iree/base/threading/notification.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/runtime/device/device.h"
#include "loom/runtime/testing/application.h"
#include "loom/runtime/testing/service_control.h"
#include "loom/runtime/testing/worker_gate.h"
#include "loom/serve/service/recognition/model.h"
#include "loom/serve/service/recognition/service.h"
#include "loom/serve/testing/http_harness.h"

IREE_FLAG(string, model_source, "", "Recognition fixture bootstrap source.");
IREE_FLAG(bool, test_elastic, false, "Use elastic backing for recognition.");
IREE_FLAG(int32_t, test_workers, 1, "Shared application worker count.");

namespace {

// Zero rate selects encoded WAV; other inputs declare their raw PCM encoding.
std::string Request(uint32_t sample_rate, size_t frame_count = 0,
                    int16_t sample = 8192, bool floating = false,
                    bool nonfinite = false) {
  const bool encoded = sample_rate == 0;
  if (encoded) {
    sample_rate = 16000;
  }
  if (!frame_count) {
    frame_count =
        sample_rate == 16000 ? 19 : (18ull * sample_rate + 15999) / 16000;
  }
  const size_t sample_bytes = floating ? sizeof(float) : sizeof(int16_t);
  std::string pcm(frame_count * sample_bytes, '\0');
  for (size_t i = 0; i < frame_count; ++i) {
    if (floating) {
      iree_unaligned_store_le_f32(pcm.data() + i * sample_bytes,
                                  nonfinite
                                      ? std::numeric_limits<float>::quiet_NaN()
                                      : static_cast<float>(sample) / 32768.0f);
    } else {
      iree_unaligned_store_le_u16(pcm.data() + i * sample_bytes,
                                  static_cast<uint16_t>(sample));
    }
  }
  std::string content_type =
      "audio/pcm;rate=" + std::to_string(sample_rate) + ";channels=1";
  if (floating) {
    content_type += ";format=f32le";
  }
  if (encoded) {
    std::string header(44, '\0');
    memcpy(header.data(), "RIFF", 4);
    iree_unaligned_store_le_u32(header.data() + 4, 36 + pcm.size());
    memcpy(header.data() + 8, "WAVEfmt ", 8);
    iree_unaligned_store_le_u32(header.data() + 16, 16);
    iree_unaligned_store_le_u16(header.data() + 20, floating ? 3 : 1);
    iree_unaligned_store_le_u16(header.data() + 22, 1);
    iree_unaligned_store_le_u32(header.data() + 24, 16000);
    iree_unaligned_store_le_u32(header.data() + 28, 16000 * sample_bytes);
    iree_unaligned_store_le_u16(header.data() + 32, sample_bytes);
    iree_unaligned_store_le_u16(header.data() + 34, sample_bytes * 8);
    memcpy(header.data() + 36, "data", 4);
    iree_unaligned_store_le_u32(header.data() + 40, pcm.size());
    pcm = header + pcm;
    content_type = "audio/wav";
  }
  const std::string body =
      "--voice\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n"
      "recognition\r\n--voice\r\nContent-Disposition: form-data; "
      "name=\"file\"; filename=\"voice.bin\"\r\n"
      "Content-Type: " +
      content_type + "\r\n\r\n" + pcm + "\r\n--voice--\r\n";
  return "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: local\r\n"
         "Content-Type: multipart/form-data; boundary=voice\r\n"
         "Content-Length: " +
         std::to_string(body.size()) + "\r\n\r\n" + body;
}

class SourceRecognitionHttpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto options = loom_runtime_test_application_options_default();
    options.backing = FLAG_test_elastic ? LOOM_RUNTIME_POOL_BACKING_ELASTIC
                                        : LOOM_RUNTIME_POOL_BACKING_FIXED;
    options.slab_size = FLAG_test_elastic ? 2 * 1024 * 1024 : 0;
    options.worker_count = FLAG_test_workers;
    loom_runtime_test_application_t* application = nullptr;
    IREE_ASSERT_OK(loom_runtime_test_application_acquire(
        &options, iree_allocator_system(), &application));
    host_ = loom_runtime_test_application_host(application);
    group_ = loom_runtime_test_application_device_group(application);
    const auto directory =
        std::filesystem::path(FLAG_model_source).parent_path().string();
    const iree_string_view_t sources[] = {IREE_SVL("generate.loom")};
    const loom_serve_recognition_model_options_t model_options = {
        .sources =
            {
                .base_directory = iree_make_cstring_view(directory.c_str()),
                .prepare = IREE_SV("prepare.loom"),
                .control = IREE_SV("control.loom"),
                .count = 1,
                .values = sources,
            },
    };
    IREE_ASSERT_OK(loom_serve_recognition_model_allocate(
        loom_runtime_test_application_device_at(application, 0),
        loom_runtime_device_group_pool_at(
            loom_runtime_test_application_device_group(application), 0),
        &model_options, iree_allocator_system(), &model_));
    const auto server_options = loom_serve_http_server_options_default();
    IREE_ASSERT_OK(loom_serve_http_server_allocate(
        loom_runtime_host_proactor_pool(host_), &server_options,
        iree_allocator_system(), &server_));
    auto listener_options = loom_serve_http_listener_options_default();
    listener_options.address = IREE_SV("127.0.0.1:0");
    IREE_ASSERT_OK(loom_serve_http_listener_allocate(
        server_, &listener_options, iree_allocator_system(), &listener_));
    IREE_ASSERT_OK(http_.Initialize(server_, listener_));
    const loom_serve_recognition_service_options_t service_options = {
        .name = IREE_SV("recognition"),
        .kind = LOOM_SERVICE_KIND_RECOGNITION,
        .pending_capacity = 2,
        .request_limit = 1024,
        .document_limit = 4096,
        .decode_frames = 5,
        .conversion_limit = 8 * 1024 * 1024,
    };
    IREE_ASSERT_OK(loom_serve_recognition_service_allocate(
        host_, loom_serve_recognition_model_generator(model_), &service_options,
        nullptr, control_.binding(), iree_allocator_system(), &service_));
    interface_ = loom_serve_recognition_service_interface(service_);
    const loom_runtime_wake_callback_t observer = {
        [](void* self) {
          iree_notification_post(static_cast<iree_notification_t*>(self),
                                 IREE_ALL_WAITERS);
        },
        loom_serve_http_server_notification(server_)};
    interface_.runtime.vtable->set_observer(interface_.runtime.self, observer);
    loom_runtime_device_group_set_observer(group_, observer);
  }
  void TearDown() override {
    if (service_) {
      Advance(true);
      AdvanceUntil(
          [&] {
            return !interface_.runtime.vtable->is_pending(
                       interface_.runtime.self) &&
                   !loom_runtime_device_group_is_pending(group_);
          },
          true);
      interface_.runtime.vtable->set_observer(interface_.runtime.self, {});
      loom_serve_recognition_service_free(service_);
    }
    loom_runtime_device_group_set_observer(group_, {});
    IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
    loom_serve_http_server_free(server_);
    IREE_EXPECT_OK(loom_serve_recognition_model_free(model_));
  }
  loom_runtime_service_observation_t Observation() {
    loom_runtime_service_observation_t observation = {};
    EXPECT_EQ(loom_runtime_service_read_observation(control_.binding().control,
                                                    &observation),
              LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    return observation;
  }

  loom_runtime_service_operation_t Operation(uint64_t ordinal) {
    loom_runtime_service_operation_t operation = {};
    EXPECT_EQ(loom_runtime_service_read_operation(control_.binding().control,
                                                  ordinal, &operation),
              LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    return operation;
  }

  iree_status_t AdvanceService(bool stopping, bool* progress,
                               uint64_t allowed_offers = UINT64_MAX) {
    const loom_runtime_service_flags_t flags =
        stopping ? LOOM_RUNTIME_SERVICE_FLAG_STOPPING : 0;
    auto status = loom_runtime_device_group_advance(group_);
    status = iree_status_join(status, loom_serve_recognition_service_progress(
                                          service_, flags, progress));
    for (uint64_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
      loom_runtime_service_offer_t offer = {};
      EXPECT_EQ(loom_runtime_service_read_offer(control_.binding().control, i,
                                                &offer),
                LOOM_RUNTIME_SERVICE_READ_COMPLETE);
      if (offer.choice_count && (allowed_offers & (UINT64_C(1) << i))) {
        loom_runtime_service_schedule_outcome_t outcome = {};
        status = interface_.runtime.vtable->try_schedule(
            interface_.runtime.self, offer.key, 0, flags, &outcome);
        *progress |= outcome.accepted;
      }
    }
    return status;
  }
  void Advance(bool stopping, uint64_t allowed_offers = UINT64_MAX) {
    bool progress = false;
    IREE_ASSERT_OK(AdvanceService(stopping, &progress, allowed_offers));
  }
  template <typename Predicate>
  void AdvanceUntil(Predicate ready, bool stopping = false,
                    uint64_t allowed_offers = UINT64_MAX) {
    auto* notification = loom_serve_http_server_notification(server_);
    while (!ready()) {
      const auto ticket = iree_notification_prepare_wait(notification);
      bool progress = false;
      auto status = AdvanceService(stopping, &progress, allowed_offers);
      if (ready() || progress || !iree_status_is_ok(status)) {
        iree_notification_cancel_wait(notification);
      } else {
        iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
      IREE_ASSERT_OK(status);
    }
  }
  void Accept(int client, loom_serve_http_connection_t** connection,
              const loom_serve_http_request_t** request) {
    IREE_ASSERT_OK(http_.TakeRequest(client, connection, request));
    IREE_ASSERT_OK(
        interface_.accept(interface_.runtime.self, *connection, *request, 0));
  }

  // Shared application host; worker gates establish the pre-execution cut.
  loom_runtime_host_t* host_ = nullptr;
  // Borrowed application group owning shared pool progress and retirement.
  loom_runtime_device_group_t* group_ = nullptr;
  // Real source-owned request state and numerical execution.
  loom_serve_recognition_model_t* model_ = nullptr;
  // Independent HTTP connection owner.
  loom_serve_http_server_t* server_ = nullptr;
  // Real loopback socket listener.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Retained source/decode facts share one composition-owned control.
  loom_runtime::testing::ServiceControl control_{3, 2};
  // Adapter whose accepted conversion task borrows the claimed upload.
  loom_serve_recognition_service_t* service_ = nullptr;
  // Manually driven production owner interface.
  loom_serve_service_t interface_ = {};
  // Readiness-aware socket client; no router thread is running.
  loom_serve::testing::HttpHarness http_;
};

TEST_F(SourceRecognitionHttpTest, DecodeGrantPreservesQueuedSource) {
  loom_serve::testing::HttpClient client;
  IREE_ASSERT_OK(http_.Send(Request(16000, 4), client.put()));
  loom_serve_http_connection_t* connection = nullptr;
  const loom_serve_http_request_t* request = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(client.get(), &connection, &request));
  ASSERT_NO_FATAL_FAILURE(Advance(false, 0));
  EXPECT_EQ(Observation().owned_request_count, 1u);
  EXPECT_EQ(Observation().waiting_request_count, 1u);
  EXPECT_EQ(Observation().offer_count, 2u);
  ASSERT_NO_FATAL_FAILURE(Advance(false, UINT64_C(1) << 1));
  const auto decode = Operation(2);
  EXPECT_NE(decode.key.serial, 0u);
  EXPECT_EQ(Operation(0).key.serial, 0u);
  EXPECT_EQ(Observation().pending_operation_count, 1u);
  EXPECT_EQ(Observation().accepted_devices, 0u);
  EXPECT_EQ(Observation().waiting_request_count, 1u);
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return Operation(2).key.serial == 0; }, false, 0));
  EXPECT_EQ(Observation().pending_operation_count, 0u);
  EXPECT_EQ(Observation().offer_count, 1u);
  EXPECT_EQ(Observation().owned_request_count, 1u);
  EXPECT_EQ(interface_.telemetry.counters->completed, 0u);
  // The decoded block and EOF survive deferral. Only the independent source
  // offer is granted from here; its existing real model produces the response.
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return interface_.telemetry.counters->completed == 1; }, false, 1));
  EXPECT_EQ(Observation().owned_request_count, 0u);
  EXPECT_EQ(Observation().pending_operation_count, 0u);
  std::string response;
  IREE_ASSERT_OK(http_.Read(client.release(), false, &response));
  EXPECT_EQ(response.find("HTTP/1.1 200 OK\r\n"), 0u);
  EXPECT_NE(response.find(R"({"text":"B"})"), std::string::npos);
}

class SourceRecognitionHttpRetirementTest
    : public SourceRecognitionHttpTest,
      public ::testing::WithParamInterface<std::tuple<bool, bool, uint32_t>> {};

TEST_F(SourceRecognitionHttpTest, DeploymentRoleRequiresItsDeclaredOutput) {
  interface_.runtime.vtable->set_observer(interface_.runtime.self, {});
  loom_serve_recognition_service_free(service_);
  service_ = nullptr;
  interface_ = {};
  const struct {
    // Deployment contract checked before the service is published.
    loom_service_kind_t kind;
    // Capability mask supplied by the generator dependency.
    loom_serve_recognition_output_flags_t outputs;
    // Whether that source satisfies the deployment contract.
    bool valid;
    // Actual output modalities published on successful preparation.
    size_t modality_count;
  } cases[] = {
      {LOOM_SERVICE_KIND_RECOGNITION, LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT,
       true, 1},
      {LOOM_SERVICE_KIND_ACTIVITY, LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY, true,
       1},
      {LOOM_SERVICE_KIND_RECOGNITION, LOOM_SERVE_RECOGNITION_OUTPUT_ALL, true,
       2},
      {LOOM_SERVICE_KIND_ACTIVITY, LOOM_SERVE_RECOGNITION_OUTPUT_ALL, true, 2},
      {LOOM_SERVICE_KIND_RECOGNITION, LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY,
       false, 0},
      {LOOM_SERVICE_KIND_ACTIVITY, LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT,
       false, 0},
      {LOOM_SERVICE_KIND_TEXT, LOOM_SERVE_RECOGNITION_OUTPUT_ALL, false, 0},
  };
  for (const auto& row : cases) {
    SCOPED_TRACE(row.kind);
    SCOPED_TRACE(row.outputs);
    auto generator = loom_serve_recognition_model_generator(model_);
    generator.outputs = row.outputs;
    const loom_serve_recognition_service_options_t options = {
        .name = IREE_SV("audio"),
        .kind = row.kind,
        .pending_capacity = 1,
        .request_limit = 1024,
        .document_limit = 4096,
        .decode_frames = 5,
        .conversion_limit = 8 * 1024 * 1024,
    };
    control_.Reset();
    auto status = loom_serve_recognition_service_allocate(
        host_, generator, &options, nullptr, control_.binding(),
        iree_allocator_system(), &service_);
    if (row.valid) {
      IREE_ASSERT_OK(status);
      interface_ = loom_serve_recognition_service_interface(service_);
      EXPECT_EQ(interface_.kind, row.kind);
      EXPECT_EQ(interface_.capabilities.output_modalities.count,
                row.modality_count);
      EXPECT_EQ(interface_.accept != nullptr,
                row.kind == LOOM_SERVICE_KIND_RECOGNITION);
      EXPECT_EQ(interface_.endpoint != nullptr,
                row.kind == LOOM_SERVICE_KIND_RECOGNITION);
      EXPECT_NE(interface_.try_open_recognition, nullptr);
      loom_serve_recognition_service_free(service_);
      service_ = nullptr;
      interface_ = {};
    } else {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
      EXPECT_EQ(service_, nullptr);
    }
  }
}

TEST_P(SourceRecognitionHttpRetirementTest,
       CancellationRetainsAcceptedDecodeBorrow) {
  const auto [stopping, floating, sample_rate] = GetParam();
  loom_serve::testing::HttpClient client;
  IREE_ASSERT_OK(
      http_.Send(Request(sample_rate, 0, 8192, floating), client.put()));
  loom_serve_http_connection_t* connection = nullptr;
  const loom_serve_http_request_t* request = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(client.get(), &connection, &request));
  // A queued upload retains its claim but cannot release device capacity.
  EXPECT_FALSE(interface_.runtime.vtable->is_active(interface_.runtime.self));
  EXPECT_TRUE(interface_.runtime.vtable->is_pending(interface_.runtime.self));
  EXPECT_EQ(interface_.snapshot(interface_.runtime.self).queued_requests, 1u);
  const std::string body(request->body.data, request->body.size);
  std::vector<std::unique_ptr<loom_runtime::testing::WorkerGate>> gates;
  for (int32_t i = 0; i < FLAG_test_workers; ++i) {
    gates.push_back(std::make_unique<loom_runtime::testing::WorkerGate>(
        loom_runtime_host_executor(host_)));
  }
  ASSERT_NO_FATAL_FAILURE(Advance(false, 0));
  EXPECT_EQ(Observation().owned_request_count, 1u);
  EXPECT_EQ(Observation().waiting_request_count, 1u);
  EXPECT_EQ(Observation().offer_count, 2u);
  EXPECT_EQ(Observation().pending_operation_count, 0u);
  EXPECT_FALSE(interface_.runtime.vtable->is_active(interface_.runtime.self));
  ASSERT_NO_FATAL_FAILURE(Advance(false));
  EXPECT_EQ(Observation().pending_operation_count, 2u);
  EXPECT_EQ(Operation(0).work, LOOM_RUNTIME_SERVICE_WORK_PREPARATION |
                                   LOOM_RUNTIME_SERVICE_WORK_ACQUISITION);
  EXPECT_EQ(Operation(2).work, LOOM_RUNTIME_SERVICE_WORK_PREPARATION);
  EXPECT_EQ(Operation(2).devices, 0u);
  EXPECT_EQ(Operation(2).use_count, 0u);
  const auto decode = Operation(2);
  EXPECT_TRUE(interface_.runtime.vtable->is_active(interface_.runtime.self));
  ASSERT_TRUE(interface_.runtime.vtable->is_pending(interface_.runtime.self));
  if (!stopping) {
    const linger reset = {1, 0};
    ASSERT_EQ(
        setsockopt(client.get(), SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
        0);
    ASSERT_EQ(close(client.release()), 0);
    iree_notification_await(
        loom_serve_http_server_notification(server_),
        [](void* self) {
          return loom_serve_http_connection_failed(
              static_cast<loom_serve_http_connection_t*>(self));
        },
        connection, iree_infinite_timeout());
  }
  ASSERT_NO_FATAL_FAILURE(Advance(stopping, 0));
  EXPECT_EQ(Operation(2).key.serial, decode.key.serial);
  EXPECT_EQ(Observation().waiting_request_count, 0u);
  EXPECT_EQ(Observation().owned_request_count, 1u);
  EXPECT_EQ(Observation().pending_operation_count, 2u);
  EXPECT_EQ(interface_.telemetry.counters->cancelled, 0u);
  EXPECT_EQ(interface_.snapshot(interface_.runtime.self).active_requests, 1u);
  EXPECT_TRUE(interface_.runtime.vtable->is_pending(interface_.runtime.self));
  EXPECT_EQ(std::string(request->body.data, request->body.size), body);
  gates.clear();
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] {
        return interface_.telemetry.counters->cancelled == 1 &&
               !interface_.runtime.vtable->is_pending(interface_.runtime.self);
      },
      stopping, 0));
  EXPECT_EQ(Operation(2).key.serial, 0u);
  EXPECT_EQ(Observation().owned_request_count, 0u);
  EXPECT_EQ(Observation().pending_operation_count, 0u);
  EXPECT_EQ(interface_.snapshot(interface_.runtime.self).active_requests, 0u);
  EXPECT_FALSE(interface_.runtime.vtable->is_active(interface_.runtime.self));
  if (!stopping) {
    IREE_ASSERT_OK(
        http_.Send(Request(sample_rate, 0, 8192, floating), client.put()));
    ASSERT_NO_FATAL_FAILURE(Accept(client.get(), &connection, &request));
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
        [&] { return interface_.telemetry.counters->completed == 1; }));
    std::string response;
    IREE_ASSERT_OK(http_.Read(client.release(), false, &response));
    EXPECT_NE(response.find("HTTP/1.1 200 OK\r\n"), std::string::npos);
    EXPECT_NE(response.find(R"({"text":"E"})"), std::string::npos);
  }
}

std::vector<uint32_t> InputRates() {
  std::vector<uint32_t> rates = {8000, 16000, 22050, 24000, 48000};
  if (LOOM_TEST_AUDIO_WAV) {
    rates.push_back(0);
  }
  return rates;
}

INSTANTIATE_TEST_SUITE_P(DisconnectAndShutdown,
                         SourceRecognitionHttpRetirementTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::ValuesIn(InputRates())));

TEST_F(SourceRecognitionHttpTest, FloatingPcmUsesTheSameStreamingConversion) {
  uint64_t completed = 0;
  for (uint32_t rate : {8000, 16000, 22050, 24000, 44100, 48000}) {
    for (bool invalid : {true, false}) {
      loom_serve::testing::HttpClient client;
      IREE_ASSERT_OK(
          http_.Send(Request(rate, 0, 8192, true, invalid), client.put()));
      loom_serve_http_connection_t* connection = nullptr;
      const loom_serve_http_request_t* request = nullptr;
      ASSERT_NO_FATAL_FAILURE(Accept(client.get(), &connection, &request));
      ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
        return !interface_.runtime.vtable->is_pending(interface_.runtime.self);
      }));
      std::string response;
      IREE_ASSERT_OK(http_.Read(client.release(), false, &response));
      if (invalid) {
        EXPECT_NE(response.find("HTTP/1.1 400 Bad Request\r\n"),
                  std::string::npos);
      } else {
        EXPECT_NE(response.find("HTTP/1.1 200 OK\r\n"), std::string::npos);
        EXPECT_NE(response.find(R"({"text":"E"})"), std::string::npos);
        ++completed;
      }
      EXPECT_EQ(interface_.telemetry.counters->completed, completed);
    }
  }
}

TEST_F(SourceRecognitionHttpTest, RawPcmLimitsRejectOnlyTheAffectedRequest) {
  // The first upload exceeds normalized duration before admission. The second
  // fits that duration but exceeds filter storage on its accepted CPU task.
  for (uint32_t rate : {1u, UINT32_MAX}) {
    loom_serve::testing::HttpClient client;
    IREE_ASSERT_OK(http_.Send(Request(rate, 1), client.put()));
    loom_serve_http_connection_t* connection = nullptr;
    const loom_serve_http_request_t* request = nullptr;
    ASSERT_NO_FATAL_FAILURE(Accept(client.get(), &connection, &request));
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
      return !interface_.runtime.vtable->is_pending(interface_.runtime.self);
    }));
    std::string response;
    IREE_ASSERT_OK(http_.Read(client.release(), false, &response));
    EXPECT_NE(response.find("HTTP/1.1 400 Bad Request\r\n"), std::string::npos);
  }
  loom_serve::testing::HttpClient client;
  IREE_ASSERT_OK(http_.Send(Request(24000), client.put()));
  loom_serve_http_connection_t* connection = nullptr;
  const loom_serve_http_request_t* request = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(client.get(), &connection, &request));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return interface_.telemetry.counters->completed == 1; }));
  std::string response;
  IREE_ASSERT_OK(http_.Read(client.release(), false, &response));
  EXPECT_NE(response.find("HTTP/1.1 200 OK\r\n"), std::string::npos);
  EXPECT_NE(response.find(R"({"text":"E"})"), std::string::npos);
}

class SourceRecognitionHttpFailureTest
    : public SourceRecognitionHttpTest,
      public ::testing::WithParamInterface<std::tuple<size_t, bool>> {};

TEST_P(SourceRecognitionHttpFailureTest, SourceCompletionPreservesServing) {
  loom_serve::testing::HttpClient rejected;
  loom_serve::testing::HttpClient peer;
  const auto [frame_count, concurrent_peer] = GetParam();
  IREE_ASSERT_OK(
      http_.Send(Request(16000, frame_count, -8192), rejected.put()));
  loom_serve_http_connection_t* connection = nullptr;
  const loom_serve_http_request_t* request = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(rejected.get(), &connection, &request));
  if (concurrent_peer) {
    IREE_ASSERT_OK(http_.Send(Request(16000), peer.put()));
    ASSERT_NO_FATAL_FAILURE(Accept(peer.get(), &connection, &request));
  }
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    return !interface_.runtime.vtable->is_pending(interface_.runtime.self);
  }));
  std::string response;
  IREE_ASSERT_OK(http_.Read(rejected.release(), false, &response));
  EXPECT_EQ(response.find("HTTP/1.1 400 Bad Request\r\n"), 0u);
  EXPECT_NE(response.find("recognition fixture accumulator is outside its "
                          "unsigned index range"),
            std::string::npos);
  if (concurrent_peer) {
    IREE_ASSERT_OK(http_.Read(peer.release(), false, &response));
    EXPECT_EQ(response.find("HTTP/1.1 200 OK\r\n"), 0u);
    EXPECT_NE(response.find(R"({"text":"E"})"), std::string::npos);
  }
  loom_serve::testing::HttpClient replacement;
  IREE_ASSERT_OK(http_.Send(Request(16000), replacement.put()));
  ASSERT_NO_FATAL_FAILURE(Accept(replacement.get(), &connection, &request));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    return !interface_.runtime.vtable->is_pending(interface_.runtime.self);
  }));
  IREE_ASSERT_OK(http_.Read(replacement.release(), false, &response));
  EXPECT_EQ(response.find("HTTP/1.1 200 OK\r\n"), 0u);
  EXPECT_NE(response.find(R"({"text":"E"})"), std::string::npos);
}

// The five-frame upload is fully staged before the source rejects it. The
// longer upload still has input queued when that same source failure arrives.
// A lone row releases the model's backing; a concurrent peer keeps it resident.
INSTANTIATE_TEST_SUITE_P(QueuedInput, SourceRecognitionHttpFailureTest,
                         ::testing::Combine(::testing::Values(5u, 19u),
                                            ::testing::Bool()));

}  // namespace
