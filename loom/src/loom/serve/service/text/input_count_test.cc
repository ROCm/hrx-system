// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/input_count.h"

#include "loom/serve/runtime/execution/scheduler.h"
#include "loom/serve/service/text/service_test_fixture.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::OwnerTurn;
using loom_serve::testing::text::SourceTextServiceTest;
using loom_serve::testing::text::StalledClient;

constexpr char kInputTokens[] = "/v1/chat/completions/input_tokens";

class InputCountTest : public SourceTextServiceTest {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(SourceTextServiceTest::SetUp());
    auto options = ServiceOptions();
    options.row_count = 1;
    options.pending_capacity = 2;
    IREE_ASSERT_OK(loom_serve_text_service_allocate(
        model_, &options, nullptr, host_allocator_, &service_));
    interface_ = loom_serve_text_service_interface(service_);
    Observe(interface_);
  }

  void TearDown() override {
    if (service_) {
      AdvanceUntil(
          interface_, [&] { return Idle(); }, LOOM_SERVE_SERVICE_FLAG_STOPPING);
      interface_.set_observer(interface_.self, {});
      loom_serve_text_service_free(service_, IREE_STATUS_OK);
    }
    SourceTextServiceTest::TearDown();
  }

  bool Idle() const {
    return !interface_.is_pending(interface_.self) &&
           !interface_.is_active(interface_.self) &&
           !interface_.snapshot(interface_.self).queued_requests;
  }

  std::string Count(const std::string& body, const std::string& headers = "") {
    const int client = Send(body, headers, kInputTokens);
    loom_serve_http_connection_t* connection = nullptr;
    Accept(client, interface_, &connection);
    AdvanceUntil(interface_, [&] { return Idle(); });
    return Read(client);
  }

  void ExpectCold() {
    const auto weights = loom_serve_text_model_weight_activity(model_);
    EXPECT_EQ(weights.activations, 0u);
    EXPECT_EQ(weights.groups, 0u);
    EXPECT_EQ(weights.read_bytes, 0u);
    const auto pool = loom_serve_text_model_pool_usage(model_);
    EXPECT_EQ(pool.pending, 0u);
    EXPECT_EQ(pool.available, pool.capacity);
    EXPECT_EQ(interface_.telemetry.counters->input_tokens, 0u);
    EXPECT_EQ(interface_.telemetry.counters->output_tokens, 0u);
  }

  // Service owns claims until its CPU loans retire; model belongs to the
  // fixture.
  loom_serve_text_service_t* service_ = nullptr;
  // Borrowed production interface driven by serialized test-owner turns.
  loom_serve_service_t interface_ = {};
};

class TextHttpQosTest : public InputCountTest {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(InputCountTest::SetUp());
    loom_serve_execution_permit_initialize(
        loom_serve_text_model_permit(model_)->scheduler,
        LOOM_SERVE_EXECUTION_RESOURCE_HOST, &peer_);
  }

  void TearDown() override {
    loom_serve_execution_permit_deinitialize(&peer_);
    InputCountTest::TearDown();
  }

  // Peer models actual realtime work in the shared host interference scope.
  loom_serve_execution_permit_t peer_ = {};
};

TEST_F(TextHttpQosTest, RealtimeCountsPassQueuedBackgroundWithoutWarmingModel) {
  loom_serve_execution_permit_request(&peer_,
                                      LOOM_SERVE_EXECUTION_QOS_REALTIME);
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&peer_));
  loom_serve::testing::HttpClient background(
      Send(Request(Message("user", "abxa")), "", kInputTokens));
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(background.get(), interface_, &connection));
  loom_serve::testing::HttpClient realtime(Send(Request(Message("user", "bax")),
                                                "X-Loom-QoS: realtime\r\n",
                                                kInputTokens));
  ASSERT_NO_FATAL_FAILURE(Accept(realtime.get(), interface_, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
    return interface_.telemetry.counters->completed == 1;
  }));
  EXPECT_NE(Read(realtime.release()).find("\r\n\r\n{\"input_tokens\":3}"),
            std::string::npos);
  IREE_EXPECT_OK(http_.CheckPending(background.get()));
  EXPECT_EQ(interface_.snapshot(interface_.self).queued_requests, 1u);
  ExpectCold();
  loom_serve_execution_permit_complete(&peer_);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
  EXPECT_NE(Read(background.release()).find("\r\n\r\n{\"input_tokens\":4}"),
            std::string::npos);
  ExpectCold();
}

TEST_F(TextHttpQosTest, ChatPriorityPrecedesInputPreparation) {
  loom_serve_execution_permit_request(&peer_,
                                      LOOM_SERVE_EXECUTION_QOS_REALTIME);
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&peer_));
  loom_serve::testing::HttpClient background(
      Send(Request(Message("user", "abxa"), 4)));
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(background.get(), interface_, &connection));
  loom_serve::testing::HttpClient realtime(
      Send(Request(Message("user", "bax"), 4), "X-Loom-QoS: realtime\r\n"));
  ASSERT_NO_FATAL_FAILURE(Accept(realtime.get(), interface_, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
    return interface_.telemetry.counters->completed == 1;
  }));
  EXPECT_EQ(Content(Read(realtime.release())), Oracle::Complete("bax", 4));
  IREE_EXPECT_OK(http_.CheckPending(background.get()));
  EXPECT_EQ(interface_.snapshot(interface_.self).queued_requests, 1u);
  loom_serve_execution_permit_complete(&peer_);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
  EXPECT_EQ(Content(Read(background.release())), Oracle::Complete("abxa", 4));
}

TEST_F(TextHttpQosTest, CountReleasesDependentOutputWithoutNewBackgroundEpoch) {
  loom_serve::testing::HttpClient background(
      Send(Request(Message("user", "abxa"), 16)));
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(background.get(), interface_, &connection));
  // A partial prefill has no output borrower. Reach a decode epoch after the
  // first actual prediction so that holding host admission tests completion.
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
    return interface_.telemetry.counters->output_tokens &&
           std::string(interface_.snapshot(interface_.self).phase) == "execute";
  }));
  loom_serve_execution_permit_request(&peer_,
                                      LOOM_SERVE_EXECUTION_QOS_REALTIME);
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(&peer_));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
    return std::string(interface_.snapshot(interface_.self).phase) ==
           "waiting_cpu";
  }));
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&peer_));
  const uint64_t batches = interface_.telemetry.counters->batches;
  loom_serve::testing::HttpClient realtime(Send(Request(Message("user", "bax")),
                                                "X-Loom-QoS: realtime\r\n",
                                                kInputTokens));
  ASSERT_NO_FATAL_FAILURE(Accept(realtime.get(), interface_, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
    return interface_.telemetry.counters->completed == 1;
  }));
  EXPECT_NE(Read(realtime.release()).find("\r\n\r\n{\"input_tokens\":3}"),
            std::string::npos);
  EXPECT_EQ(interface_.telemetry.counters->batches, batches);
  EXPECT_EQ(peer_.phase, LOOM_SERVE_EXECUTION_PERMIT_ACCEPTED);
  loom_serve_execution_permit_complete(&peer_);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
  EXPECT_EQ(Content(Read(background.release())), Oracle::Complete("abxa", 16));
}

TEST_F(InputCountTest, UnknownQosRejectsBeforePreparation) {
  for (const char* endpoint : {kInputTokens, "/v1/chat/completions"}) {
    loom_serve::testing::HttpClient client(Send(
        Request(Message("user", "abxa")), "X-Loom-QoS: urgent\r\n", endpoint));
    loom_serve_http_connection_t* connection = nullptr;
    ASSERT_NO_FATAL_FAILURE(Accept(client.get(), interface_, &connection));
    EXPECT_EQ(Read(client.release()).find("HTTP/1.1 400"), 0u);
    EXPECT_TRUE(Idle());
  }
  ExpectCold();
}

TEST_F(InputCountTest, FreshChatCountsWithoutActivatingOrReserving) {
  ExpectCold();
  EXPECT_NE(Count(Request(Message("user", "abxa")))
                .find("\r\n\r\n{\"input_tokens\":4}"),
            std::string::npos);
  EXPECT_NE(
      Count(Request(Message("system", "ab") + "," + Message("user", "xa") +
                    "," + Message("assistant", "bx") + "," +
                    Message("user", "ba")))
          .find("\r\n\r\n{\"input_tokens\":9}"),
      std::string::npos);
  // Counting requires no output reservation: the whole context can be input.
  EXPECT_NE(Count(Request(Message("user", std::string(96, 'a')), 16))
                .find("\r\n\r\n{\"input_tokens\":96}"),
            std::string::npos);
  ExpectCold();
  EXPECT_EQ(interface_.telemetry.counters->completed, 3u);
}

TEST_F(InputCountTest, SourceMediaRowsNeedNoProjectionOrComponent) {
  const auto request = Request(
      ImageMessage("a", "image/x-portable-arbitrarymap", kPam, "low", "b"));
  const auto response = Count(request);
  EXPECT_EQ(response.find("HTTP/1.1 200"), 0u) << response;
  // The fixture's ordered_content emits one logical position per image.
  EXPECT_NE(response.find("\r\n\r\n{\"input_tokens\":3}"), std::string::npos)
      << response;
  ExpectCold();
  const auto multiple = Count(Request(
      ImageMessage("a", "image/x-portable-arbitrarymap", kPam, "low", "b") +
      "," +
      ImageMessage("x", "image/x-portable-arbitrarymap", kPam, "high", "")));
  EXPECT_NE(multiple.find("{\"input_tokens\":5}"), std::string::npos)
      << multiple;
  ExpectCold();
}

TEST_F(InputCountTest, ToolsAndOptionsUseTheSameSourcePolicy) {
  const auto tools = Count(
      R"({"model":"fixture-text","messages":[{"role":"user","content":"abxa"}],"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{}}}}]})");
  // chat_begin emits one catalog marker per tool before the four input IDs.
  EXPECT_NE(tools.find("{\"input_tokens\":5}"), std::string::npos) << tools;
  const auto options = Count(
      R"({"model":"fixture-text","messages":[{"role":"user","content":"abxa"}],"enable_thinking":false,"temperature":0,"top_p":1,"top_k":1,"seed":42})");
  EXPECT_NE(options.find("{\"input_tokens\":4}"), std::string::npos) << options;
  ExpectCold();
}

TEST_F(InputCountTest, CountingProgressesWithEveryExecutionRowOccupied) {
  StalledClient generation;
  ASSERT_NO_FATAL_FAILURE(
      generation.Send(server_, Request(Message("user", "abxa"), 4)));
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(AcceptLoopback(interface_, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
    const auto snapshot = interface_.snapshot(interface_.self);
    return snapshot.active_requests == 1 &&
           std::string(snapshot.phase) == "idle" &&
           !interface_.is_pending(interface_.self);
  }));
  const auto before = loom_serve_text_model_pool_usage(model_);
  const uint64_t outputs = interface_.telemetry.counters->output_tokens;
  const int client = Send(Request(Message("user", "bax")), "", kInputTokens);
  ASSERT_NO_FATAL_FAILURE(Accept(client, interface_, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
    return interface_.telemetry.counters->completed == 1;
  }));
  EXPECT_NE(Read(client).find("\r\n\r\n{\"input_tokens\":3}"),
            std::string::npos);
  EXPECT_EQ(interface_.snapshot(interface_.self).active_requests, 1u);
  const auto after = loom_serve_text_model_pool_usage(model_);
  EXPECT_EQ(after.available, before.available);
  EXPECT_EQ(after.pending, before.pending);
  EXPECT_EQ(interface_.telemetry.counters->output_tokens, outputs);
  ASSERT_NO_FATAL_FAILURE(generation.Close());
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
}

TEST_F(InputCountTest, StopJoinsBorrowedHeadAndCancelsQueuedQueries) {
  const auto body = Request(Message("user", "abxa"));
  loom_serve_http_connection_t* connections[2] = {};
  const loom_serve_http_request_t* requests[2] = {};
  const int first = Send(body, "", kInputTokens);
  IREE_ASSERT_OK(http_.TakeRequest(first, &connections[0], &requests[0]));
  const int second = Send(body, "X-Loom-QoS: realtime\r\n", kInputTokens);
  IREE_ASSERT_OK(http_.TakeRequest(second, &connections[1], &requests[1]));
  OwnerTurn::Run(loom_serve_host_executor(host_), [&] {
    for (size_t i = 0; i < 2; ++i) {
      IREE_ASSERT_OK(
          interface_.accept(interface_.self, connections[i], requests[i], 0));
    }
    // Later priority cannot move the head already borrowed by the CPU task.
    bool found_head = false;
    loom_serve_http_request_snapshot_t snapshot;
    for (size_t i = 0;
         interface_.inspection.request_at(interface_.self, i, &snapshot); ++i) {
      if (snapshot.id == 1) {
        found_head = true;
        EXPECT_STREQ(snapshot.phase, "count_input");
      }
    }
    EXPECT_TRUE(found_head);
    // The sole worker is still running this owner turn, so the first query
    // cannot have released its CPU loan. Stopping may cancel only the tail.
    bool progress = false;
    IREE_ASSERT_OK(
        Advance(interface_, LOOM_SERVE_SERVICE_FLAG_STOPPING, &progress));
    EXPECT_TRUE(progress);
    EXPECT_TRUE(interface_.is_pending(interface_.self));
    EXPECT_EQ(interface_.telemetry.counters->cancelled, 1u);
    EXPECT_EQ(interface_.snapshot(interface_.self).active_requests, 1u);
  });
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      interface_, [&] { return Idle(); }, LOOM_SERVE_SERVICE_FLAG_STOPPING));
  EXPECT_EQ(interface_.telemetry.counters->cancelled, 2u);
  EXPECT_EQ(interface_.telemetry.counters->completed, 0u);
  EXPECT_TRUE(Read(first, true).empty());
  EXPECT_TRUE(Read(second, true).empty());
  ExpectCold();
}

TEST_F(InputCountTest, FullQueueRejectsWithoutDisturbingAcceptedQueries) {
  const auto body = Request(Message("user", "abxa"));
  int clients[3];
  loom_serve_http_connection_t* connections[3] = {};
  const loom_serve_http_request_t* requests[3] = {};
  for (size_t i = 0; i < 3; ++i) {
    clients[i] = Send(body, "", kInputTokens);
    IREE_ASSERT_OK(
        http_.TakeRequest(clients[i], &connections[i], &requests[i]));
  }
  OwnerTurn::Run(loom_serve_host_executor(host_), [&] {
    for (size_t i = 0; i < 3; ++i) {
      IREE_ASSERT_OK(
          interface_.accept(interface_.self, connections[i], requests[i], 0));
    }
    EXPECT_EQ(interface_.snapshot(interface_.self).queued_requests, 1u);
    EXPECT_EQ(interface_.snapshot(interface_.self).active_requests, 1u);
    EXPECT_EQ(interface_.telemetry.counters->rejected, 1u);
  });
  EXPECT_EQ(Read(clients[2]).find("HTTP/1.1 503"), 0u);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_NE(Read(clients[i]).find("{\"input_tokens\":4}"), std::string::npos);
  }
  EXPECT_NE(Count(body).find("{\"input_tokens\":4}"), std::string::npos);
  ExpectCold();
}

TEST_F(InputCountTest, DisconnectRetiresTheHeadBeforeReusingItsClaim) {
  const auto body = Request(Message("user", "abxa"));
  loom_serve_http_connection_t* connections[2] = {};
  const loom_serve_http_request_t* requests[2] = {};
  // Claims follow server readiness, not client send order. Claim each peer
  // before sending the next so closing first cancels the borrowed head.
  const int first = Send(body, "", kInputTokens);
  IREE_ASSERT_OK(http_.TakeRequest(first, &connections[0], &requests[0]));
  const int second = Send(Request(Message("user", "bax")), "", kInputTokens);
  IREE_ASSERT_OK(http_.TakeRequest(second, &connections[1], &requests[1]));
  OwnerTurn::Run(loom_serve_host_executor(host_), [&] {
    for (size_t i = 0; i < 2; ++i) {
      IREE_ASSERT_OK(
          interface_.accept(interface_.self, connections[i], requests[i], 0));
    }
  });
  const linger reset = {1, 0};
  ASSERT_EQ(setsockopt(first, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)), 0);
  ASSERT_EQ(close(first), 0);
  ASSERT_TRUE(iree_notification_await(
      loom_serve_http_server_notification(server_),
      [](void* self) {
        return loom_serve_http_connection_failed(
            static_cast<loom_serve_http_connection_t*>(self));
      },
      connections[0], iree_infinite_timeout()));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
  EXPECT_NE(Read(second).find("{\"input_tokens\":3}"), std::string::npos);
  EXPECT_EQ(interface_.telemetry.counters->cancelled, 1u);
  EXPECT_EQ(interface_.telemetry.counters->completed, 1u);
  EXPECT_NE(Count(body).find("{\"input_tokens\":4}"), std::string::npos);
  ExpectCold();
}

TEST_F(InputCountTest, InvalidInputDoesNotPoisonTheModel) {
  for (
      const std::string& body :
      {Request(Message("user", "?")),
       Request(Message("user", std::string(97, 'a'))),
       std::string(
           R"({"model":"fixture-text","enable_thinking":true,"messages":[{"role":"user","content":"a"}]})"),
       std::string(R"({"model":"fixture-text","messages":false})")}) {
    const auto response = Count(body);
    EXPECT_EQ(response.find("HTTP/1.1 400"), 0u) << response;
    EXPECT_EQ(response.find("\"input_tokens\""), std::string::npos);
  }
  for (const char* header :
       {"X-Loom-Session: private\r\n", "X-Loom-Checkpoint: pin\r\n"}) {
    EXPECT_EQ(Count(Request(Message("user", "a")), header).find("HTTP/1.1 400"),
              0u);
  }
  EXPECT_NE(
      Count(Request(Message("user", "abxa"))).find("{\"input_tokens\":4}"),
      std::string::npos);
  ExpectCold();
}

// Completion consumes a task's returned status after release. Exercise that
// boundary with real transport claims and externally reported task failures.
class InputCountOutcomeTest : public SourceTextServiceTest {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(SourceTextServiceTest::SetUp());
    IREE_ASSERT_OK(loom_serve_text_telemetry_initialize(
        nullptr, host_allocator_, &telemetry_));
    IREE_ASSERT_OK(
        loom_serve_text_input_count_initialize(2, host_allocator_, &queries_));
    iree_string_builder_initialize(host_allocator_, &scratch_);
  }

  void TearDown() override {
    loom_serve_text_input_count_cancel(&queries_, false, true, &telemetry_);
    loom_serve_text_input_count_deinitialize(&queries_);
    iree_string_builder_deinitialize(&scratch_);
    loom_serve_telemetry_stream_free(telemetry_.stream);
    SourceTextServiceTest::TearDown();
  }

  void Enqueue(int* out_client) {
    *out_client = Send(Request(Message("user", "a")), "", kInputTokens);
    loom_serve_http_connection_t* connection = nullptr;
    const loom_serve_http_request_t* request = nullptr;
    IREE_ASSERT_OK(http_.TakeRequest(*out_client, &connection, &request));
    IREE_ASSERT_OK(loom_serve_text_input_count_enqueue(
        &queries_, connection, request,
        loom_serve_text_telemetry_begin(&telemetry_),
        LOOM_SERVE_EXECUTION_QOS_BACKGROUND, &telemetry_));
  }

  // Real HTTP claims retained until their task outcomes are consumed.
  loom_serve_text_input_count_t queries_ = {};
  // Owner-only request accounting, independent of the unused model fixture.
  loom_serve_text_telemetry_t telemetry_ = {};
  // Response assembly backing retained through the test owner lifetime.
  iree_string_builder_t scratch_ = {};
};

TEST_F(InputCountOutcomeTest, DisconnectCannotHideFatalSourceFailure) {
  int client = -1;
  ASSERT_NO_FATAL_FAILURE(Enqueue(&client));
  const linger reset = {1, 0};
  ASSERT_EQ(setsockopt(client, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
            0);
  ASSERT_EQ(close(client), 0);
  ASSERT_TRUE(iree_notification_await(
      loom_serve_http_server_notification(server_),
      [](void* self) {
        return loom_serve_http_connection_failed(
            static_cast<loom_serve_http_connection_t*>(self));
      },
      queries_.requests[0].connection, iree_infinite_timeout()));
  iree_status_t status = loom_serve_text_input_count_complete(
      &queries_, false, iree_make_status(IREE_STATUS_INTERNAL, "source failed"),
      &telemetry_, &scratch_);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INTERNAL);
  iree_status_free(status);
  EXPECT_EQ(queries_.count, 0u);
  EXPECT_EQ(telemetry_.counters.failed, 1u);
  EXPECT_EQ(telemetry_.counters.cancelled, 0u);
}

TEST_F(InputCountOutcomeTest, StopCannotHideFatalSourceFailure) {
  int client = -1;
  ASSERT_NO_FATAL_FAILURE(Enqueue(&client));
  iree_status_t status = loom_serve_text_input_count_complete(
      &queries_, true, iree_make_status(IREE_STATUS_INTERNAL, "source failed"),
      &telemetry_, &scratch_);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INTERNAL);
  iree_status_free(status);
  EXPECT_TRUE(Read(client, true).empty());
  EXPECT_EQ(telemetry_.counters.failed, 1u);
  EXPECT_EQ(telemetry_.counters.cancelled, 0u);
}

TEST_F(InputCountOutcomeTest, ModelFailureAccountsBothBorrowedAndQueuedClaims) {
  int first = -1, second = -1;
  ASSERT_NO_FATAL_FAILURE(Enqueue(&first));
  ASSERT_NO_FATAL_FAILURE(Enqueue(&second));
  iree_status_t failure = iree_make_status(IREE_STATUS_UNAVAILABLE);
  loom_serve_text_telemetry_fail(&telemetry_, failure);
  iree_status_free(failure);
  EXPECT_TRUE(
      loom_serve_text_input_count_cancel(&queries_, true, true, &telemetry_));
  EXPECT_EQ(queries_.count, 1u);
  EXPECT_EQ(telemetry_.counters.failed, 1u);
  IREE_ASSERT_OK(loom_serve_text_input_count_complete(
      &queries_, true, iree_ok_status(), &telemetry_, &scratch_));
  EXPECT_TRUE(Read(first, true).empty());
  EXPECT_TRUE(Read(second, true).empty());
  EXPECT_EQ(telemetry_.counters.failed, 2u);
  EXPECT_EQ(telemetry_.counters.cancelled, 0u);
  EXPECT_EQ(telemetry_.counters.input_tokens, 0u);
}

TEST_F(InputCountOutcomeTest, ValidationFailuresRemainRequestLocal) {
  for (const auto code :
       {IREE_STATUS_INVALID_ARGUMENT, IREE_STATUS_OUT_OF_RANGE,
        IREE_STATUS_UNIMPLEMENTED, IREE_STATUS_RESOURCE_EXHAUSTED}) {
    int client = -1;
    ASSERT_NO_FATAL_FAILURE(Enqueue(&client));
    IREE_ASSERT_OK(loom_serve_text_input_count_complete(
        &queries_, false, iree_make_status(code), &telemetry_, &scratch_));
    const auto response = Read(client);
    EXPECT_EQ(response.find(code == IREE_STATUS_UNIMPLEMENTED ? "HTTP/1.1 501"
                            : code == IREE_STATUS_RESOURCE_EXHAUSTED
                                ? "HTTP/1.1 503"
                                : "HTTP/1.1 400"),
              0u)
        << response;
  }
  EXPECT_EQ(telemetry_.counters.rejected, 4u);
  EXPECT_EQ(telemetry_.counters.failed, 0u);
  EXPECT_EQ(telemetry_.counters.cancelled, 0u);
}

}  // namespace
