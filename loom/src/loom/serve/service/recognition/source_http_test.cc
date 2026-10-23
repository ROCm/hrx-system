// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/socket.h>
#include <unistd.h>

#include <filesystem>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "iree/base/threading/notification.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/server/resources.h"
#include "loom/serve/service/recognition/model.h"
#include "loom/serve/service/recognition/service.h"
#include "loom/serve/testing/http_harness.h"
#include "loom/serve/testing/service_application.h"
#include "loom/serve/testing/worker_gate.h"

IREE_FLAG(string, model_source, "", "Recognition fixture bootstrap source.");
IREE_FLAG(bool, test_elastic, false, "Use elastic backing for recognition.");
IREE_FLAG(int32_t, test_workers, 1, "Shared application worker count.");

namespace {

// Zero rate selects encoded WAV; positive rates select declared PCM16LE.
std::string Request(uint32_t sample_rate, size_t frame_count = 0,
                    int16_t sample = 8192) {
  const bool encoded = sample_rate == 0;
  if (encoded) {
    sample_rate = 16000;
  }
  if (!frame_count) {
    frame_count =
        sample_rate == 16000 ? 19 : (18ull * sample_rate + 15999) / 16000;
  }
  std::string pcm(frame_count * sizeof(int16_t), '\0');
  for (size_t i = 0; i < frame_count; ++i) {
    iree_unaligned_store_le_u16(pcm.data() + i * sizeof(int16_t),
                                static_cast<uint16_t>(sample));
  }
  std::string content_type =
      "audio/pcm;rate=" + std::to_string(sample_rate) + ";channels=1";
  if (encoded) {
    std::string header(44, '\0');
    memcpy(header.data(), "RIFF", 4);
    iree_unaligned_store_le_u32(header.data() + 4, 36 + pcm.size());
    memcpy(header.data() + 8, "WAVEfmt ", 8);
    iree_unaligned_store_le_u32(header.data() + 16, 16);
    iree_unaligned_store_le_u16(header.data() + 20, 1);
    iree_unaligned_store_le_u16(header.data() + 22, 1);
    iree_unaligned_store_le_u32(header.data() + 24, 16000);
    iree_unaligned_store_le_u32(header.data() + 28, 32000);
    iree_unaligned_store_le_u16(header.data() + 32, 2);
    iree_unaligned_store_le_u16(header.data() + 34, 16);
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
    auto options = loom_serve_test_application_options_default();
    options.backing = FLAG_test_elastic ? LOOM_SERVE_DEVICE_BACKING_ELASTIC
                                        : LOOM_SERVE_DEVICE_BACKING_FIXED;
    options.slab_size = FLAG_test_elastic ? 2 * 1024 * 1024 : 0;
    options.worker_count = FLAG_test_workers;
    loom_serve_test_application_t* application = nullptr;
    IREE_ASSERT_OK(loom_serve_test_application_acquire(
        &options, iree_allocator_system(), &application));
    host_ = loom_serve_test_application_host(application);
    resources_.device_group =
        loom_serve_test_application_device_group(application);
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
        loom_serve_test_application_device_at(application, 0), &model_options,
        iree_allocator_system(), &model_));
    const auto server_options = loom_serve_http_server_options_default();
    IREE_ASSERT_OK(loom_serve_http_server_allocate(
        loom_serve_host_proactor_pool(host_), &server_options,
        iree_allocator_system(), &server_));
    auto listener_options = loom_serve_http_listener_options_default();
    listener_options.address = IREE_SV("127.0.0.1:0");
    IREE_ASSERT_OK(loom_serve_http_listener_allocate(
        server_, &listener_options, iree_allocator_system(), &listener_));
    IREE_ASSERT_OK(http_.Initialize(server_, listener_));
    const loom_serve_recognition_service_options_t service_options = {
        .name = IREE_SV("recognition"),
        .kind = LOOM_SERVE_MODEL_KIND_RECOGNITION,
        .pending_capacity = 2,
        .request_limit = 1024,
        .document_limit = 4096,
        .decode_frames = 5,
        .conversion_limit = 8 * 1024 * 1024,
    };
    IREE_ASSERT_OK(loom_serve_recognition_service_allocate(
        host_, loom_serve_recognition_model_generator(model_), &service_options,
        nullptr, iree_allocator_system(), &service_));
    interface_ = loom_serve_recognition_service_interface(service_);
    const loom_serve_wake_callback_t observer = {
        [](void* self) {
          iree_notification_post(static_cast<iree_notification_t*>(self),
                                 IREE_ALL_WAITERS);
        },
        loom_serve_http_server_notification(server_)};
    interface_.set_observer(interface_.self, observer);
    loom_serve_resources_set_observer(&resources_, observer);
  }
  void TearDown() override {
    if (service_) {
      Advance(true);
      AdvanceUntil(
          [&] {
            return !interface_.is_pending(interface_.self) &&
                   !loom_serve_resources_is_pending(&resources_);
          },
          true);
      interface_.set_observer(interface_.self, {});
      loom_serve_recognition_service_free(service_);
    }
    loom_serve_resources_set_observer(&resources_, {});
    IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
    loom_serve_http_server_free(server_);
    IREE_EXPECT_OK(loom_serve_recognition_model_free(model_));
  }
  iree_status_t AdvanceService(bool stopping, bool* progress) {
    auto status = loom_serve_resources_advance(&resources_);
    return iree_status_join(
        status, interface_.advance(
                    interface_.self,
                    stopping ? LOOM_SERVE_SERVICE_FLAG_STOPPING : 0, progress));
  }
  void Advance(bool stopping) {
    bool progress = false;
    IREE_ASSERT_OK(AdvanceService(stopping, &progress));
  }
  template <typename Predicate>
  void AdvanceUntil(Predicate ready, bool stopping = false) {
    auto* notification = loom_serve_http_server_notification(server_);
    while (!ready()) {
      const auto ticket = iree_notification_prepare_wait(notification);
      bool progress = false;
      auto status = AdvanceService(stopping, &progress);
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
        interface_.accept(interface_.self, *connection, *request, 0));
  }

  // Shared application host; worker gates establish the pre-execution cut.
  loom_serve_host_t* host_ = nullptr;
  // Borrowed application resources progressed by this test's router role.
  loom_serve_resources_t resources_ = {};
  // Real source-owned request state and numerical execution.
  loom_serve_recognition_model_t* model_ = nullptr;
  // Independent HTTP connection owner.
  loom_serve_http_server_t* server_ = nullptr;
  // Real loopback socket listener.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Adapter whose accepted conversion task borrows the claimed upload.
  loom_serve_recognition_service_t* service_ = nullptr;
  // Manually driven production owner interface.
  loom_serve_service_t interface_ = {};
  // Readiness-aware socket client; no router thread is running.
  loom_serve::testing::HttpHarness http_;
};

class SourceRecognitionHttpRetirementTest
    : public SourceRecognitionHttpTest,
      public ::testing::WithParamInterface<std::tuple<bool, uint32_t>> {};

TEST_F(SourceRecognitionHttpTest, DeploymentRoleRequiresItsDeclaredOutput) {
  interface_.set_observer(interface_.self, {});
  loom_serve_recognition_service_free(service_);
  service_ = nullptr;
  interface_ = {};
  const struct {
    // Deployment contract checked before the service is published.
    loom_serve_model_kind_t kind;
    // Capability mask supplied by the generator dependency.
    loom_serve_recognition_output_flags_t outputs;
    // Whether that source satisfies the deployment contract.
    bool valid;
    // Actual output modalities published on successful preparation.
    size_t modality_count;
  } cases[] = {
      {LOOM_SERVE_MODEL_KIND_RECOGNITION,
       LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT, true, 1},
      {LOOM_SERVE_MODEL_KIND_ACTIVITY, LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY,
       true, 1},
      {LOOM_SERVE_MODEL_KIND_RECOGNITION, LOOM_SERVE_RECOGNITION_OUTPUT_ALL,
       true, 2},
      {LOOM_SERVE_MODEL_KIND_ACTIVITY, LOOM_SERVE_RECOGNITION_OUTPUT_ALL, true,
       2},
      {LOOM_SERVE_MODEL_KIND_RECOGNITION,
       LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY, false, 0},
      {LOOM_SERVE_MODEL_KIND_ACTIVITY, LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT,
       false, 0},
      {LOOM_SERVE_MODEL_KIND_TEXT, LOOM_SERVE_RECOGNITION_OUTPUT_ALL, false, 0},
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
    auto status = loom_serve_recognition_service_allocate(
        host_, generator, &options, nullptr, iree_allocator_system(),
        &service_);
    if (row.valid) {
      IREE_ASSERT_OK(status);
      interface_ = loom_serve_recognition_service_interface(service_);
      EXPECT_EQ(interface_.kind, row.kind);
      EXPECT_EQ(interface_.capabilities.output_modalities.count,
                row.modality_count);
      EXPECT_EQ(interface_.accept != nullptr,
                row.kind == LOOM_SERVE_MODEL_KIND_RECOGNITION);
      EXPECT_EQ(interface_.endpoint != nullptr,
                row.kind == LOOM_SERVE_MODEL_KIND_RECOGNITION);
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
  const auto [stopping, sample_rate] = GetParam();
  loom_serve::testing::HttpClient client;
  IREE_ASSERT_OK(http_.Send(Request(sample_rate), client.put()));
  loom_serve_http_connection_t* connection = nullptr;
  const loom_serve_http_request_t* request = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(client.get(), &connection, &request));
  // A queued upload retains its claim but cannot release device capacity.
  EXPECT_FALSE(interface_.is_active(interface_.self));
  EXPECT_TRUE(interface_.is_pending(interface_.self));
  EXPECT_EQ(interface_.snapshot(interface_.self).queued_requests, 1u);
  const std::string body(request->body.data, request->body.size);
  std::vector<std::unique_ptr<loom_serve::testing::WorkerGate>> gates;
  for (int32_t i = 0; i < FLAG_test_workers; ++i) {
    gates.push_back(std::make_unique<loom_serve::testing::WorkerGate>(
        loom_serve_host_executor(host_)));
  }
  ASSERT_NO_FATAL_FAILURE(Advance(false));
  EXPECT_TRUE(interface_.is_active(interface_.self));
  ASSERT_TRUE(interface_.is_pending(interface_.self));
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
  ASSERT_NO_FATAL_FAILURE(Advance(stopping));
  EXPECT_EQ(interface_.telemetry.counters->cancelled, 0u);
  EXPECT_EQ(interface_.snapshot(interface_.self).active_requests, 1u);
  EXPECT_TRUE(interface_.is_pending(interface_.self));
  EXPECT_EQ(std::string(request->body.data, request->body.size), body);
  gates.clear();
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] {
        return interface_.telemetry.counters->cancelled == 1 &&
               !interface_.is_pending(interface_.self);
      },
      stopping));
  EXPECT_EQ(interface_.snapshot(interface_.self).active_requests, 0u);
  EXPECT_FALSE(interface_.is_active(interface_.self));
  if (!stopping) {
    IREE_ASSERT_OK(http_.Send(Request(sample_rate), client.put()));
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
                                            ::testing::ValuesIn(InputRates())));

TEST_F(SourceRecognitionHttpTest, RawPcmLimitsRejectOnlyTheAffectedRequest) {
  // The first upload exceeds normalized duration before admission. The second
  // fits that duration but exceeds filter storage on its accepted CPU task.
  for (uint32_t rate : {1u, UINT32_MAX}) {
    loom_serve::testing::HttpClient client;
    IREE_ASSERT_OK(http_.Send(Request(rate, 1), client.put()));
    loom_serve_http_connection_t* connection = nullptr;
    const loom_serve_http_request_t* request = nullptr;
    ASSERT_NO_FATAL_FAILURE(Accept(client.get(), &connection, &request));
    ASSERT_NO_FATAL_FAILURE(
        AdvanceUntil([&] { return !interface_.is_pending(interface_.self); }));
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
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return !interface_.is_pending(interface_.self); }));
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
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return !interface_.is_pending(interface_.self); }));
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
