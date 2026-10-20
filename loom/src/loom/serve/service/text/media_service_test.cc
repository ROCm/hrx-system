// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdlib>
#include <initializer_list>

#include "loom/serve/media/image_codec.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/execution/scheduler.h"
#include "loom/serve/service/finite/component.h"
#include "loom/serve/service/text/service_test_fixture.h"
#include "loom/serve/service/text/stream.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::OwnerTurn;
using loom_serve::testing::text::SourceTextServiceTest;
using NativeStream = std::unique_ptr<loom_serve_text_stream_t,
                                     decltype(&loom_serve_text_stream_close)>;

class MediaServiceTest : public SourceTextServiceTest {
 protected:
  virtual iree_string_view_t ProjectorOptions() const { return IREE_SV("{}"); }

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(SourceTextServiceTest::SetUp());
    const char* source = std::getenv("LOOM_SERVE_TEST_PROJECTOR_SOURCE");
    ASSERT_NE(source, nullptr);
    IREE_ASSERT_OK(projector_document_.Load(source));
    const loom_serve_finite_model_options_t options = {
        .sources = projector_document_.value().sources,
        .asset_directory = IREE_SV("."),
        .options = ProjectorOptions(),
        .specialization_cache = {2, 1024 * 1024},
    };
    const loom_serve_wake_callback_t observer = {
        [](void* data) {
          iree_notification_post(static_cast<iree_notification_t*>(data),
                                 IREE_ALL_WAITERS);
        },
        loom_serve_http_server_notification(server_),
    };
    IREE_ASSERT_OK(loom_serve_finite_component_allocate(
        device_, &options, observer, host_allocator_, &image_));
    auto service_options = ServiceOptions();
    service_options.image = image_;
    IREE_ASSERT_OK(loom_serve_text_service_allocate(
        model_, &service_options, nullptr, host_allocator_, &service_));
    interface_ = loom_serve_text_service_interface(service_);
    Observe(interface_);
  }

  void TearDown() override {
    loom_serve_execution_permit_deinitialize(&peer_);
    if (service_) {
      AdvanceUntil(
          interface_, [&] { return Idle(); }, LOOM_SERVE_SERVICE_FLAG_STOPPING);
      interface_.set_observer(interface_.self, {});
      loom_serve_text_service_free(service_, IREE_STATUS_OK);
    }
    if (image_) {
      auto* notification = loom_serve_http_server_notification(server_);
      bool complete = false;
      while (!complete) {
        const auto ticket = iree_notification_prepare_wait(notification);
        auto status = loom_serve_finite_component_retire(image_, &complete);
        if (complete || !iree_status_is_ok(status)) {
          iree_notification_cancel_wait(notification);
          IREE_ASSERT_OK(status);
        } else {
          iree_notification_commit_wait(notification, ticket,
                                        IREE_DURATION_ZERO,
                                        IREE_TIME_INFINITE_FUTURE);
        }
      }
      loom_serve_finite_component_free(image_);
    }
    SourceTextServiceTest::TearDown();
  }

  bool Idle() const {
    return !interface_.is_pending(interface_.self) &&
           !interface_.is_active(interface_.self) &&
           !interface_.snapshot(interface_.self).queued_requests;
  }

  void PrepareProjector() {
    loom_serve_finite_model_t* model = nullptr;
    IREE_ASSERT_OK(loom_serve_finite_component_prepare(image_, &model));
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
      return loom_serve_finite_component_phase(image_) !=
             LOOM_SERVE_FINITE_COMPONENT_PREPARING;
    }));
    IREE_ASSERT_OK(loom_serve_finite_component_prepare(image_, &model));
    ASSERT_NE(model, nullptr);
  }

  void RequestVoice() {
    loom_serve_execution_permit_initialize(loom_serve_device_scheduler(device_),
                                           LOOM_SERVE_EXECUTION_RESOURCE_HOST,
                                           &peer_);
    loom_serve_execution_permit_request(&peer_,
                                        LOOM_SERVE_EXECUTION_QOS_REALTIME);
  }

  iree_status_t Open(const std::string& body, const char* session,
                     loom_serve_execution_qos_t qos, NativeStream* stream) {
    loom_serve_text_request_t request = {
        .session = iree_make_cstring_view(session),
        .qos = qos,
    };
    iree_string_builder_initialize(host_allocator_, &request.body);
    iree_status_t status = iree_string_builder_append_string(
        &request.body, iree_make_string_view(body.data(), body.size()));
    loom_serve_text_stream_t* value = nullptr;
    if (iree_status_is_ok(status)) {
      status = loom_serve_text_service_try_open(service_, &request, &value);
    }
    iree_string_builder_deinitialize(&request.body);
    stream->reset(value);
    return status;
  }

  void ReadFinal(const NativeStream& stream,
                 std::initializer_list<int32_t> input) {
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
      if (loom_serve_text_stream_peek(stream.get())) {
        loom_serve_text_stream_consume(stream.get());
      }
      return loom_serve_text_stream_is_complete(stream.get());
    }));
    IREE_ASSERT_OK(loom_serve_text_stream_status(stream.get()));
    const auto* result = loom_serve_text_stream_result(stream.get());
    ASSERT_NE(result, nullptr);
    Oracle oracle;
    for (int32_t value : input) {
      oracle.Consume(value);
    }
    EXPECT_EQ(std::string(result->content.data, result->content.size),
              Complete(oracle, 4));
  }

  void HoldBeforeDecode(const std::string& body, NativeStream* stream) {
    ASSERT_NO_FATAL_FAILURE(PrepareProjector());
    OwnerTurn::Run(loom_serve_host_executor(host_), [&] {
      IREE_ASSERT_OK(Open(body, "background",
                          LOOM_SERVE_EXECUTION_QOS_BACKGROUND, stream));
      // Accept text preparation before requesting competing host admission.
      bool progress = false;
      IREE_ASSERT_OK(Advance(interface_, 0, &progress));
      RequestVoice();
      EXPECT_FALSE(loom_serve_execution_permit_try_acquire(&peer_));
    });
    ASSERT_NO_FATAL_FAILURE(AtImagePhase("image_waiting_cpu", [&] {
      ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&peer_));
    }));
  }

  static int32_t Feature(float value) {
    int32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  }

  static std::string Complete(Oracle& oracle, size_t count) {
    std::string result;
    for (size_t i = 0; i < count; ++i) {
      const int32_t token = oracle.Predict();
      if (token == 3) {
        break;
      }
      result.push_back("abx"[token]);
      oracle.Consume(token);
    }
    return result;
  }

  // Capture real transport requests before the owner turn so tests never block
  // a shared worker on I/O. Every claim enters the same bounded queue window.
  void AcceptTogether(const std::vector<int>& clients) {
    std::vector<loom_serve_http_connection_t*> connections(clients.size());
    std::vector<const loom_serve_http_request_t*> requests(clients.size());
    for (size_t i = 0; i < clients.size(); ++i) {
      IREE_ASSERT_OK(
          http_.TakeRequest(clients[i], &connections[i], &requests[i]));
    }
    OwnerTurn::Run(loom_serve_host_executor(host_), [&] {
      for (size_t i = 0; i < clients.size(); ++i) {
        IREE_ASSERT_OK(
            interface_.accept(interface_.self, connections[i], requests[i], 0));
      }
    });
  }

  // Observe a phase within the serialized owner turn, before its newly queued
  // CPU job can start on the sole worker. Only the outer caller waits.
  void AtImagePhase(const char* phase, std::function<void()> action) {
    auto* notification = loom_serve_http_server_notification(server_);
    bool reached = false;
    while (!reached && !HasFatalFailure()) {
      const auto ticket = iree_notification_prepare_wait(notification);
      bool progress = false;
      OwnerTurn::Run(loom_serve_host_executor(host_), [&] {
        IREE_ASSERT_OK(Advance(interface_, 0, &progress));
        if (std::string(interface_.snapshot(interface_.self).phase) == phase) {
          reached = true;
          action();
        }
      });
      if (reached || progress || HasFatalFailure()) {
        iree_notification_cancel_wait(notification);
      } else {
        iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
  }

  // Source documents outlive lazy preparation and final component retirement.
  loom_serve::testing::ModelDocument projector_document_;
  // Independent optional image computation on the actual text device.
  loom_serve_finite_component_t* image_ = nullptr;
  // Serialized production HTTP owner, including its native projection window.
  loom_serve_text_service_t* service_ = nullptr;
  // Borrowed callbacks driven on the ordinary shared executor.
  loom_serve_service_t interface_ = {};
  // Independent realtime host owner, detached before service teardown.
  loom_serve_execution_permit_t peer_ = {};
};

class SplitMediaServiceTest : public MediaServiceTest {
 protected:
  iree_string_view_t ProjectorOptions() const override {
    return IREE_SV("{\"batch_units\":2}");
  }
};

class FailedMediaServiceTest : public MediaServiceTest {
 protected:
  iree_string_view_t ProjectorOptions() const override {
    return IREE_SV("{\"batch_units\":0}");
  }
};

TEST_F(MediaServiceTest,
       VoiceReplacesUnstartedDecodeWithoutPromotingBackground) {
  if (!loom_serve_media_image_codec(LOOM_SERVE_MEDIA_IMAGE_FORMAT_PAM)) {
    return;
  }
  const auto body = Request(
      ImageMessage("a", "image/x-portable-arbitrarymap", kPam, "low", "b"), 4);
  NativeStream background(nullptr, loom_serve_text_stream_close);
  ASSERT_NO_FATAL_FAILURE(HoldBeforeDecode(body, &background));
  NativeStream voice(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(Request(ImageMessage("x", "image/x-portable-arbitrarymap",
                                           kAlphaPam, "low", "a"),
                              4),
                      "voice", LOOM_SERVE_EXECUTION_QOS_REALTIME, &voice));
  ASSERT_NO_FATAL_FAILURE(ReadFinal(voice, {2, Feature(11), 0}));
  EXPECT_EQ(interface_.snapshot(interface_.self).queued_requests, 1u);
  EXPECT_FALSE(loom_serve_text_stream_is_complete(background.get()));
  loom_serve_execution_permit_complete(&peer_);
  ASSERT_NO_FATAL_FAILURE(ReadFinal(background, {0, Feature(511), 1}));
}

TEST_F(MediaServiceTest, CancelledDecodeReleasesSelectionWhileVoiceOwnsHost) {
  if (!loom_serve_media_image_codec(LOOM_SERVE_MEDIA_IMAGE_FORMAT_PAM)) {
    return;
  }
  const auto body = Request(
      ImageMessage("a", "image/x-portable-arbitrarymap", kPam, "low", "b"), 4);
  NativeStream background(nullptr, loom_serve_text_stream_close);
  ASSERT_NO_FATAL_FAILURE(HoldBeforeDecode(body, &background));
  background.reset();
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
  EXPECT_EQ(interface_.telemetry.counters->cancelled, 1u);
  NativeStream replacement(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(body, "background", LOOM_SERVE_EXECUTION_QOS_REALTIME,
                      &replacement));
  ASSERT_NO_FATAL_FAILURE(ReadFinal(replacement, {0, Feature(511), 1}));
  loom_serve_execution_permit_complete(&peer_);
}

class InterruptedMediaServiceTest
    : public MediaServiceTest,
      public ::testing::WithParamInterface<const char*> {};

TEST_P(InterruptedMediaServiceTest, VoiceWaitsForBorrowerThenReplacesWindow) {
  if (!loom_serve_media_image_codec(LOOM_SERVE_MEDIA_IMAGE_FORMAT_PAM)) {
    return;
  }
  NativeStream background(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(Open(
      Request(
          ImageMessage("a", "image/x-portable-arbitrarymap", kPam, "low", "b"),
          4),
      "background", LOOM_SERVE_EXECUTION_QOS_BACKGROUND, &background));
  NativeStream voice(nullptr, loom_serve_text_stream_close);
  ASSERT_NO_FATAL_FAILURE(AtImagePhase(GetParam(), [&] {
    RequestVoice();
    EXPECT_FALSE(loom_serve_execution_permit_try_acquire(&peer_));
    IREE_ASSERT_OK(
        Open(Request(ImageMessage("x", "image/x-portable-arbitrarymap",
                                  kAlphaPam, "low", "a"),
                     4),
             "voice", LOOM_SERVE_EXECUTION_QOS_REALTIME, &voice));
  }));
  ASSERT_NO_FATAL_FAILURE(ReadFinal(voice, {2, Feature(11), 0}));
  EXPECT_EQ(interface_.snapshot(interface_.self).queued_requests, 1u);
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&peer_));
  loom_serve_execution_permit_complete(&peer_);
  ASSERT_NO_FATAL_FAILURE(ReadFinal(background, {0, Feature(511), 1}));
}

INSTANTIATE_TEST_SUITE_P(CodecBatchAndCompiler, InterruptedMediaServiceTest,
                         ::testing::Values("image_prepare", "image_batch",
                                           "image_specialize"));

TEST_F(FailedMediaServiceTest, ComponentFailureRejectsImagesButPreservesText) {
  const std::string image =
      ImageMessage("a", "image/x-portable-arbitrarymap", kPam, "low", "b");
  const int first = Send(Request(image, 4));
  const int second = Send(Request(image, 4));
  const int text = Send(Request(Message("user", "abxa"), 4));
  ASSERT_NO_FATAL_FAILURE(AcceptTogether({first, second, text}));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
  const bool enabled = loom_serve_media_image_codec(
                           LOOM_SERVE_MEDIA_IMAGE_FORMAT_PAM) != nullptr;
  for (int client : {first, second}) {
    const std::string response = Read(client);
    EXPECT_EQ(response.find(enabled ? "HTTP/1.1 503" : "HTTP/1.1 501"), 0u)
        << response;
    if (enabled) {
      EXPECT_NE(response.find("image component is unavailable"),
                std::string::npos);
    }
  }
  EXPECT_EQ(Content(Read(text)), Oracle::Complete("abxa", 4));
  EXPECT_EQ(loom_serve_finite_component_phase(image_),
            enabled ? LOOM_SERVE_FINITE_COMPONENT_FAILED
                    : LOOM_SERVE_FINITE_COMPONENT_DECLARED);
  EXPECT_EQ(interface_.telemetry.counters->rejected, 2u);
  EXPECT_EQ(interface_.telemetry.counters->completed, 1u);
  EXPECT_EQ(interface_.telemetry.counters->failed, 0u);

  // The component retains its failed attempt, but ordinary text admission and
  // output remain available after the image requests relinquish their claims.
  const int followup = Send(Request(Message("user", "xbab"), 4));
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(followup, interface_, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
  EXPECT_EQ(Content(Read(followup)), Oracle::Complete("xbab", 4));
  EXPECT_EQ(interface_.telemetry.counters->completed, 2u);
}

TEST_F(SplitMediaServiceTest, OneRequestKeepsFeaturesAcrossBoundedCohorts) {
  const std::string first =
      ImageMessage("a", "image/x-portable-arbitrarymap", kPam, "low", "b");
  const std::string second =
      ImageMessage("x", "image/x-portable-arbitrarymap", kAlphaPam, "low", "a");
  const std::string third =
      ImageMessage("b", "image/x-portable-arbitrarymap", kPam, "low", "x");
  const int client = Send(Request(first + "," + second + "," + third, 4));
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(client, interface_, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
  if (!loom_serve_media_image_codec(LOOM_SERVE_MEDIA_IMAGE_FORMAT_PAM)) {
    EXPECT_EQ(Read(client).find("HTTP/1.1 501"), 0u);
    return;
  }
  Oracle oracle;
  for (int32_t value :
       {0, Feature(512), 1, 2, Feature(12), 0, 1, Feature(511), 2}) {
    oracle.Consume(value);
  }
  EXPECT_EQ(Content(Read(client)), Complete(oracle, 4));
  EXPECT_EQ(interface_.telemetry.counters->completed, 1u);
}

TEST_F(MediaServiceTest, LateCodecErrorRejectsOneRequestWithoutLosingPeers) {
  // A complete header with a truncated raster passes inspection, then fails
  // in the accepted CPU decode job alongside independently valid peers.
  const char* damaged =
      "UDcKV0lEVEggMgpIRUlHSFQgMgpERVBUSCAzCk1BWFZBTCAyNTUKVFVQTFRZUEUg"
      "UkdCCkVOREhEUgr/AAAA/wAAAP///w==";
  const int first = Send(Request(
      ImageMessage("a", "image/x-portable-arbitrarymap", kPam, "low", "b"), 4));
  const int invalid = Send(Request(
      ImageMessage("x", "image/x-portable-arbitrarymap", damaged, "low", ""),
      4));
  const int third = Send(Request(
      ImageMessage("b", "image/x-portable-arbitrarymap", kAlphaPam, "low", "a"),
      4));
  ASSERT_NO_FATAL_FAILURE(AcceptTogether({first, invalid, third}));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
  if (!loom_serve_media_image_codec(LOOM_SERVE_MEDIA_IMAGE_FORMAT_PAM)) {
    for (int client : {first, invalid, third}) {
      EXPECT_EQ(Read(client).find("HTTP/1.1 501"), 0u);
    }
    return;
  }
  Oracle first_oracle;
  for (int32_t value : {0, Feature(512), 1}) {
    first_oracle.Consume(value);
  }
  Oracle third_oracle;
  for (int32_t value : {1, Feature(12), 0}) {
    third_oracle.Consume(value);
  }
  EXPECT_EQ(Content(Read(first)), Complete(first_oracle, 4));
  EXPECT_EQ(Read(invalid).find("HTTP/1.1 400"), 0u);
  EXPECT_EQ(Content(Read(third)), Complete(third_oracle, 4));
  EXPECT_EQ(interface_.telemetry.counters->rejected, 1u);
  EXPECT_EQ(interface_.telemetry.counters->failed, 0u);
}

class CancelledMediaServiceTest
    : public MediaServiceTest,
      public ::testing::WithParamInterface<const char*> {};

TEST_P(CancelledMediaServiceTest,
       StopJoinsAcceptedWorkBeforeReleasingHttpBodies) {
  if (!loom_serve_media_image_codec(LOOM_SERVE_MEDIA_IMAGE_FORMAT_PAM)) {
    return;
  }
  const int first = Send(Request(
      ImageMessage("a", "image/x-portable-arbitrarymap", kPam, "low", "b"), 4));
  const int second = Send(Request(
      ImageMessage("x", "image/x-portable-arbitrarymap", kAlphaPam, "low", "a"),
      4));
  ASSERT_NO_FATAL_FAILURE(AcceptTogether({first, second}));
  ASSERT_NO_FATAL_FAILURE(AtImagePhase(GetParam(), [&] {
    bool progress = false;
    IREE_ASSERT_OK(
        Advance(interface_, LOOM_SERVE_SERVICE_FLAG_STOPPING, &progress));
    EXPECT_TRUE(interface_.is_pending(interface_.self));
    EXPECT_EQ(interface_.snapshot(interface_.self).queued_requests, 2u);
    EXPECT_EQ(interface_.telemetry.counters->cancelled, 0u);
  }));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      interface_, [&] { return Idle(); }, LOOM_SERVE_SERVICE_FLAG_STOPPING));
  EXPECT_TRUE(Read(first, true).empty());
  EXPECT_TRUE(Read(second, true).empty());
  EXPECT_EQ(interface_.telemetry.counters->cancelled, 2u);
}

INSTANTIATE_TEST_SUITE_P(CodecAndCompiler, CancelledMediaServiceTest,
                         ::testing::Values("image_prepare",
                                           "image_specialize"));

TEST_F(MediaServiceTest, CancelledMiddlePeerPreservesPreparedCohortOrdinals) {
  if (!loom_serve_media_image_codec(LOOM_SERVE_MEDIA_IMAGE_FORMAT_PAM)) {
    return;
  }
  const std::array<std::string, 3> bodies = {
      Request(
          ImageMessage("a", "image/x-portable-arbitrarymap", kPam, "low", "b"),
          4),
      Request(
          ImageMessage("x", "image/x-portable-arbitrarymap", kPam, "low", "a"),
          4),
      Request(ImageMessage("b", "image/x-portable-arbitrarymap", kAlphaPam,
                           "low", "a"),
              4),
  };
  std::array<int, 3> clients = {};
  std::array<loom_serve_http_connection_t*, 3> connections = {};
  std::array<const loom_serve_http_request_t*, 3> requests = {};
  // Claim each peer before sending the next: transport arrival order is not
  // client identity. All three claims still enter one owner turn below.
  for (size_t i = 0; i < clients.size(); ++i) {
    clients[i] = Send(bodies[i]);
    IREE_ASSERT_OK(
        http_.TakeRequest(clients[i], &connections[i], &requests[i]));
    EXPECT_EQ(std::string(requests[i]->body.data, requests[i]->body.size),
              bodies[i]);
  }
  OwnerTurn::Run(loom_serve_host_executor(host_), [&] {
    for (size_t i = 0; i < clients.size(); ++i) {
      IREE_ASSERT_OK(
          interface_.accept(interface_.self, connections[i], requests[i], 0));
    }
  });
  ASSERT_NO_FATAL_FAILURE(AtImagePhase("image_batch", [] {}));
  const linger reset = {1, 0};
  ASSERT_EQ(
      setsockopt(clients[1], SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)), 0);
  ASSERT_EQ(close(clients[1]), 0);
  iree_notification_await(
      loom_serve_http_server_notification(server_),
      [](void* data) {
        return loom_serve_http_connection_failed(
            static_cast<loom_serve_http_connection_t*>(data));
      },
      connections[1], iree_infinite_timeout());
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
  Oracle first;
  for (int32_t value : {0, Feature(513), 1}) {
    first.Consume(value);
  }
  Oracle third;
  for (int32_t value : {1, Feature(13), 0}) {
    third.Consume(value);
  }
  EXPECT_EQ(Content(Read(clients[0])), Complete(first, 4));
  EXPECT_EQ(Content(Read(clients[2])), Complete(third, 4));
  EXPECT_EQ(interface_.telemetry.counters->cancelled, 1u);
  EXPECT_EQ(interface_.telemetry.counters->completed, 2u);
}

TEST_F(MediaServiceTest, CrossRequestImagesConditionTextAndRetainedFollowup) {
  std::string first =
      ImageMessage("a", "image/x-portable-arbitrarymap", kPam, "low", "b");
  first.insert(first.size() - 2,
               std::string(",{\"type\":\"image_url\",\"image_url\":{\"url\":"
                           "\"data:image/x-portable-arbitrarymap;base64,") +
                   kAlphaPam + "\"}}");
  const std::string second =
      ImageMessage("x", "image/x-portable-arbitrarymap", kPam, "low", "a");
  const std::array<int, 3> clients = {
      Send(Request(first, 4), "X-Loom-Session: images\r\n"),
      Send(Request(second, 4)),
      Send(Request(Message("user", "abxa"), 4)),
  };
  std::array<loom_serve_http_connection_t*, 3> connections = {};
  std::array<const loom_serve_http_request_t*, 3> requests = {};
  for (size_t i = 0; i < clients.size(); ++i) {
    IREE_ASSERT_OK(
        http_.TakeRequest(clients[i], &connections[i], &requests[i]));
  }
  // All three real HTTP claims arrive in one owner turn. The single shared
  // worker cannot run the first queued CPU preparation until this returns.
  OwnerTurn::Run(loom_serve_host_executor(host_), [&] {
    for (size_t i = 0; i < clients.size(); ++i) {
      IREE_ASSERT_OK(
          interface_.accept(interface_.self, connections[i], requests[i], 0));
    }
  });
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
  if (!loom_serve_media_image_codec(LOOM_SERVE_MEDIA_IMAGE_FORMAT_PAM)) {
    EXPECT_EQ(Read(clients[0]).find("HTTP/1.1 501"), 0u);
    EXPECT_EQ(Read(clients[1]).find("HTTP/1.1 501"), 0u);
    EXPECT_EQ(Content(Read(clients[2])), Oracle::Complete("abxa", 4));
    EXPECT_EQ(loom_serve_finite_component_phase(image_),
              LOOM_SERVE_FINITE_COMPONENT_DECLARED);
    return;
  }
  // The projector incorporates the whole cohort width into its first feature.
  // Serial count=1 projection produces different output, even with correct RGB.
  Oracle first_oracle;
  for (int32_t value : {0, Feature(513), 1, Feature(13)}) {
    first_oracle.Consume(value);
  }
  Oracle second_oracle;
  for (int32_t value : {2, Feature(513), 0}) {
    second_oracle.Consume(value);
  }
  const std::string first_response = Content(Read(clients[0]));
  EXPECT_EQ(first_response, Complete(first_oracle, 4));
  EXPECT_EQ(Content(Read(clients[1])), Complete(second_oracle, 4));
  EXPECT_EQ(Content(Read(clients[2])), Oracle::Complete("abxa", 4));
  const auto report = *loom_serve_finite_component_report(image_);
  const int followup =
      Send(Request(first + "," + Message("assistant", first_response) + "," +
                       Message("user", "x"),
                   4),
           "X-Loom-Session: images\r\n");
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(followup, interface_, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
  first_oracle.Consume("#x");
  const auto response = Read(followup);
  EXPECT_EQ(Content(response), Complete(first_oracle, 4));
  EXPECT_EQ(response.find("\"cached_tokens\":0"), std::string::npos)
      << response;
  EXPECT_EQ(loom_serve_finite_component_report(image_)->commands,
            report.commands);
  EXPECT_EQ(interface_.telemetry.counters->completed, 4u);
  EXPECT_EQ(interface_.telemetry.counters->rejected, 0u);
}

}  // namespace
