// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/socket.h>

#include "loom/serve/runtime/execution/scheduler.h"
#include "loom/serve/service/image/source_test_fixture.h"

namespace {

using loom_serve::testing::HttpClient;

class ImageHttpQosTest : public loom_serve::testing::image::SourceImageTest {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(SourceImageTest::SetUp());
    ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
    const loom_serve_image_service_options_t options = {
        .name = loom_serve_image_model_name(model_),
        .width = kWidth,
        .height = kHeight,
        .pending_capacity = 3,
    };
    IREE_ASSERT_OK(loom_serve_image_service_allocate(
        host_, loom_serve_image_model_generator(model_), &options, nullptr,
        host_allocator_, &service_));
    interface_ = loom_serve_image_service_interface(service_);
    interface_.set_observer(interface_.self,
                            {[](void* self) {
                               iree_notification_post(
                                   static_cast<iree_notification_t*>(self),
                                   IREE_ALL_WAITERS);
                             },
                             loom_serve_http_server_notification(server_)});
  }

  void TearDown() override {
    loom_serve_execution_permit_deinitialize(&peer_);
    if (service_) {
      bool progress = false;
      IREE_EXPECT_OK(interface_.advance(
          interface_.self, LOOM_SERVE_SERVICE_FLAG_STOPPING, &progress));
      AdvanceUntil([&] { return !interface_.is_pending(interface_.self); },
                   LOOM_SERVE_SERVICE_FLAG_STOPPING);
      interface_.set_observer(interface_.self, {});
      loom_serve_image_service_free(service_);
    }
    SourceImageTest::TearDown();
  }

  template <typename Ready>
  void AdvanceUntil(Ready ready, loom_serve_service_flags_t flags = 0) {
    auto* notification = loom_serve_http_server_notification(server_);
    while (!ready()) {
      const auto ticket = iree_notification_prepare_wait(notification);
      bool progress = false;
      IREE_ASSERT_OK(interface_.advance(interface_.self, flags, &progress));
      if (!iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING)) {
        const loom_serve_resources_t resources = {.device_group = group_};
        progress |= loom_serve_resources_schedule(&resources);
      }
      if (ready() || progress) {
        iree_notification_cancel_wait(notification);
      } else {
        iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
  }

  std::string Phase() { return interface_.snapshot(interface_.self).phase; }

  void Accept(int client, loom_serve_http_connection_t** out_connection) {
    const loom_serve_http_request_t* request = nullptr;
    IREE_ASSERT_OK(http_.TakeRequest(client, out_connection, &request));
    IREE_ASSERT_OK(
        interface_.accept(interface_.self, *out_connection, request, 0));
  }

  int SendImage(const std::string& prompt, uint64_t seed,
                const std::string& headers = "") {
    return Send("POST", "/v1/images/generations",
                "{\"model\":\"fixture-image\",\"prompt\":\"" + prompt +
                    "\",\"seed\":" + std::to_string(seed) + "}",
                headers);
  }

  void ExpectImage(int client, size_t tokens, uint64_t seed) {
    const auto response = Read(client);
    EXPECT_EQ(response.find("HTTP/1.1 200"), 0u) << response;
    EXPECT_NE(
        response.find("\"b64_json\":\"" + EncodedPixels(tokens, seed) + "\""),
        std::string::npos);
  }

  void HoldHost() {
    loom_serve_execution_permit_initialize(loom_serve_device_scheduler(device_),
                                           LOOM_SERVE_EXECUTION_RESOURCE_HOST,
                                           &peer_);
    loom_serve_execution_permit_request(&peer_,
                                        LOOM_SERVE_EXECUTION_QOS_REALTIME);
    ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&peer_));
  }

  void CheckLatePriority(bool captured) {
    if (captured) {
      // Fill the device's one native slot while leaving background source
      // preparation eligible. No native image work can be accepted yet.
      const auto generator = loom_serve_image_model_generator(model_);
      loom_serve_execution_permit_initialize(
          generator.permit->scheduler,
          generator.permit->resources & ~LOOM_SERVE_EXECUTION_RESOURCE_HOST,
          &peer_);
      loom_serve_execution_permit_request(&peer_,
                                          LOOM_SERVE_EXECUTION_QOS_BACKGROUND);
      ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&peer_));
    } else {
      ASSERT_NO_FATAL_FAILURE(HoldHost());
    }
    HttpClient background(SendImage("ab", 0));
    loom_serve_http_connection_t* connection = nullptr;
    ASSERT_NO_FATAL_FAILURE(Accept(background.get(), &connection));
    HttpClient abandoned;
    loom_serve_http_connection_t* abandoned_connection = nullptr;
    if (captured) {
      *abandoned.put() = SendImage("aba", 4);
      ASSERT_NO_FATAL_FAILURE(Accept(abandoned.get(), &abandoned_connection));
    }
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
        [&] { return Phase() == (captured ? "waiting" : "waiting_cpu"); }));
    EXPECT_EQ(interface_.telemetry.counters->batches, 0u);
    if (captured) {
      // The abandoned member is still part of the captured cohort. Replacing
      // it must release that receipt without requeueing its closed connection.
      const linger reset = {1, 0};
      ASSERT_EQ(setsockopt(abandoned.get(), SOL_SOCKET, SO_LINGER, &reset,
                           sizeof(reset)),
                0);
      ASSERT_EQ(close(abandoned.release()), 0);
      iree_notification_await(
          loom_serve_http_server_notification(server_),
          [](void* self) {
            return loom_serve_http_connection_failed(
                static_cast<loom_serve_http_connection_t*>(self));
          },
          abandoned_connection, iree_infinite_timeout());
    }
    HttpClient realtime(SendImage("aba", 1, "X-Loom-QoS: realtime\r\n"));
    ASSERT_NO_FATAL_FAILURE(Accept(realtime.get(), &connection));
    if (captured) {
      // Publish the replacement before changing the external peer. Its CPU
      // task cannot start while the background peer owns the exclusion scope.
      bool progress = false;
      IREE_ASSERT_OK(interface_.advance(interface_.self, 0, &progress));
      EXPECT_TRUE(progress);
      EXPECT_EQ(interface_.telemetry.counters->cancelled, 1u);
      loom_serve_execution_permit_deinitialize(&peer_);
      ASSERT_NO_FATAL_FAILURE(HoldHost());
    }
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
        [&] { return interface_.telemetry.counters->completed == 1; }));
    ASSERT_NO_FATAL_FAILURE(ExpectImage(realtime.release(), 3, 1));
    IREE_ASSERT_OK(http_.CheckPending(background.get()));
    EXPECT_EQ(interface_.snapshot(interface_.self).queued_requests +
                  interface_.snapshot(interface_.self).active_requests,
              1u);
    EXPECT_EQ(interface_.telemetry.counters->batches, GetParam() ? 2u : 1u);
    loom_serve_execution_permit_deinitialize(&peer_);
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
        [&] { return interface_.telemetry.counters->completed == 2; }));
    ASSERT_NO_FATAL_FAILURE(ExpectImage(background.release(), 2, 0));
    EXPECT_EQ(interface_.telemetry.counters->failed, 0u);
  }

  // Manual serialized owner, driven at actual production phase boundaries.
  loom_serve_image_service_t* service_ = nullptr;
  // Public service interface borrowing service_.
  loom_serve_service_t interface_ = {};
  // Independent peer occupying the configured device or interference scope.
  loom_serve_execution_permit_t peer_ = {};
};

TEST_P(ImageHttpQosTest, RealtimeSupersedesUnacceptedPreparation) {
  CheckLatePriority(false);
}

TEST_P(ImageHttpQosTest,
       RealtimeReplacesCapturedCohortAndRetiresCancelledPeer) {
  CheckLatePriority(true);
}

TEST_P(ImageHttpQosTest, PendingRealtimeCannotPreemptItsOwnSourceDependency) {
  HttpClient background(SendImage("ab", 0));
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(background.get(), &connection));
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return Phase() == "generating"; }));
  HttpClient realtime(SendImage("aba", 1, "X-Loom-QoS: realtime\r\n"));
  ASSERT_NO_FATAL_FAILURE(Accept(realtime.get(), &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return interface_.telemetry.counters->completed == 2; }));
  ASSERT_NO_FATAL_FAILURE(ExpectImage(background.release(), 2, 0));
  ASSERT_NO_FATAL_FAILURE(ExpectImage(realtime.release(), 3, 1));
  EXPECT_EQ(interface_.telemetry.counters->batches, GetParam() ? 4u : 2u);
}

TEST_P(ImageHttpQosTest,
       FinalEncodingInheritsDependentRealtimeWithoutMoreMath) {
  HttpClient background(SendImage("ab", 0));
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(background.get(), &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    return Phase() == "generating" &&
           interface_.telemetry.counters->batches == (GetParam() ? 1u : 0u);
  }));
  loom_serve_execution_permit_initialize(loom_serve_device_scheduler(device_),
                                         LOOM_SERVE_EXECUTION_RESOURCE_HOST,
                                         &peer_);
  loom_serve_execution_permit_request(&peer_,
                                      LOOM_SERVE_EXECUTION_QOS_REALTIME);
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return Phase() == "waiting_cpu"; }));
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&peer_));
  const auto background_parts = interface_.telemetry.counters->batches;
  HttpClient realtime(SendImage("aba", 1, "X-Loom-QoS: realtime\r\n"));
  ASSERT_NO_FATAL_FAILURE(Accept(realtime.get(), &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return interface_.telemetry.counters->completed == 1; }));
  EXPECT_EQ(interface_.telemetry.counters->batches, background_parts);
  ASSERT_NO_FATAL_FAILURE(ExpectImage(background.release(), 2, 0));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return interface_.telemetry.counters->completed == 2; }));
  ASSERT_NO_FATAL_FAILURE(ExpectImage(realtime.release(), 3, 1));
}

TEST_P(ImageHttpQosTest, InvalidQosRejectedBeforePreparation) {
  HttpClient client(SendImage("ab", 0, "X-Loom-QoS: fastest\r\n"));
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(client.get(), &connection));
  EXPECT_EQ(Read(client.release()).find("HTTP/1.1 400"), 0u);
  EXPECT_EQ(interface_.telemetry.counters->rejected, 1u);
  EXPECT_EQ(interface_.telemetry.counters->batches, 0u);
  EXPECT_EQ(Phase(), "idle");
}

INSTANTIATE_TEST_SUITE_P(CommandQuanta, ImageHttpQosTest,
                         ::testing::Values(0u, 1u));

}  // namespace
