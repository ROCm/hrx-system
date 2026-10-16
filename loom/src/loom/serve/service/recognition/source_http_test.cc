// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/socket.h>
#include <unistd.h>

#include <filesystem>
#include <string>
#include <tuple>

#include "iree/base/threading/notification.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/service/recognition/http.h"
#include "loom/serve/service/recognition/model.h"
#include "loom/serve/testing/http_harness.h"
#include "loom/serve/testing/service_application.h"
#include "loom/serve/testing/worker_gate.h"

IREE_FLAG(string, model_source, "", "Recognition fixture bootstrap source.");

namespace {

std::string Request(bool encoded) {
  std::string pcm(19 * sizeof(int16_t), '\0');
  for (size_t i = 0; i < 19; ++i) {
    iree_unaligned_store_le_u16(pcm.data() + i * sizeof(int16_t), 8192);
  }
  std::string content_type = "audio/pcm;rate=16000;channels=1";
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

class SourceRecognitionHttpTest
    : public ::testing::TestWithParam<std::tuple<bool, bool>> {
 protected:
  void SetUp() override {
    const auto options = loom_serve_test_application_options_default();
    loom_serve_test_application_t* application = nullptr;
    IREE_ASSERT_OK(loom_serve_test_application_acquire(
        &options, iree_allocator_system(), &application));
    host_ = loom_serve_test_application_host(application);
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
    const loom_serve_recognition_http_options_t service_options = {
        .name = IREE_SV("recognition"),
        .pending_capacity = 2,
        .request_limit = 1024,
        .document_limit = 4096,
        .decode_frames = 5,
    };
    IREE_ASSERT_OK(loom_serve_recognition_http_allocate(
        host_, loom_serve_recognition_model_generator(model_), &service_options,
        nullptr, iree_allocator_system(), &service_));
    interface_ = loom_serve_recognition_http_interface(service_);
    interface_.set_observer(interface_.self,
                            {[](void* self) {
                               iree_notification_post(
                                   static_cast<iree_notification_t*>(self),
                                   IREE_ALL_WAITERS);
                             },
                             loom_serve_http_server_notification(server_)});
  }
  void TearDown() override {
    if (service_) {
      Advance(true);
      AdvanceUntil([&] { return !interface_.is_pending(interface_.self); },
                   true);
      interface_.set_observer(interface_.self, {});
      loom_serve_recognition_http_free(service_);
    }
    IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
    loom_serve_http_server_free(server_);
    IREE_EXPECT_OK(loom_serve_recognition_model_free(model_));
  }
  void Advance(bool stopping) {
    bool progress = false;
    IREE_ASSERT_OK(interface_.advance(
        interface_.self, stopping ? LOOM_SERVE_SERVICE_FLAG_STOPPING : 0,
        &progress));
  }
  template <typename Predicate>
  void AdvanceUntil(Predicate ready, bool stopping = false) {
    auto* notification = loom_serve_http_server_notification(server_);
    while (!ready()) {
      const auto ticket = iree_notification_prepare_wait(notification);
      bool progress = false;
      auto status = interface_.advance(
          interface_.self, stopping ? LOOM_SERVE_SERVICE_FLAG_STOPPING : 0,
          &progress);
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

  // Shared application host; its sole worker supplies the deterministic gate.
  loom_serve_host_t* host_ = nullptr;
  // Real source-owned request state and numerical execution.
  loom_serve_recognition_model_t* model_ = nullptr;
  // Independent HTTP connection owner.
  loom_serve_http_server_t* server_ = nullptr;
  // Real loopback socket listener.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Adapter whose accepted conversion task borrows the claimed upload.
  loom_serve_recognition_http_t* service_ = nullptr;
  // Manually driven production owner interface.
  loom_serve_service_t interface_ = {};
  // Readiness-aware socket client; no router thread is running.
  loom_serve::testing::HttpHarness http_;
};

TEST_P(SourceRecognitionHttpTest, CancellationRetainsAcceptedDecodeBorrow) {
  const auto [stopping, encoded] = GetParam();
  loom_serve::testing::HttpClient client;
  IREE_ASSERT_OK(http_.Send(Request(encoded), client.put()));
  loom_serve_http_connection_t* connection = nullptr;
  const loom_serve_http_request_t* request = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(client.get(), &connection, &request));
  // A queued upload retains its claim but cannot release device capacity.
  EXPECT_FALSE(interface_.is_active(interface_.self));
  EXPECT_TRUE(interface_.is_pending(interface_.self));
  EXPECT_EQ(interface_.snapshot(interface_.self).queued_requests, 1u);
  const std::string body(request->body.data, request->body.size);
  loom_serve::testing::WorkerGate gate(loom_serve_host_executor(host_));
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
  gate.Open();
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] {
        return interface_.telemetry.counters->cancelled == 1 &&
               !interface_.is_pending(interface_.self);
      },
      stopping));
  EXPECT_EQ(interface_.snapshot(interface_.self).active_requests, 0u);
  EXPECT_FALSE(interface_.is_active(interface_.self));
  if (!stopping) {
    IREE_ASSERT_OK(http_.Send(Request(encoded), client.put()));
    ASSERT_NO_FATAL_FAILURE(Accept(client.get(), &connection, &request));
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
        [&] { return interface_.telemetry.counters->completed == 1; }));
    std::string response;
    IREE_ASSERT_OK(http_.Read(client.release(), false, &response));
    EXPECT_NE(response.find("HTTP/1.1 200 OK\r\n"), std::string::npos);
    EXPECT_NE(response.find(R"({"text":"E"})"), std::string::npos);
  }
}

INSTANTIATE_TEST_SUITE_P(
    DisconnectAndShutdown, SourceRecognitionHttpTest,
    ::testing::Combine(::testing::Bool(), LOOM_TEST_AUDIO_WAV
                                              ? ::testing::Bool()
                                              : ::testing::ValuesIn({false})));

}  // namespace
