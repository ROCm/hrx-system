// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/serving/router_test_fixture.h"

namespace {

using loom_serve::testing::HttpRouterTest;
using loom_serve::testing::ModelService;
using loom_serve::testing::ObservationClient;

TEST_P(HttpRouterTest,
       LiveReadersResumeAndYieldToPeersWhenTheirRingOverwrites) {
  ObservationClient slow;
  slow.Open(server_);
  const auto initial = slow.Frame();
  const auto cursor = ObservationClient::Cursor(initial);
  EXPECT_NE(initial.find("Content-Type: text/event-stream\r\n"),
            std::string::npos);
  slow.WaitCredit(true);
  // An independent diagnostic publisher, not an inference completion, must
  // wake an otherwise idle owner. This error has one explicit diagnostic sink.
  iree_status_t error =
      iree_make_status(IREE_STATUS_UNAVAILABLE, "stream probe");
  loom_serve_http_server_record_error(server_, error);
  slow.WaitCredit(false);
  const auto* stream = loom_serve_http_server_diagnostics(server_);
  for (size_t i = 0; i < loom_serve_telemetry_stream_capacity(stream) + 8;
       ++i) {
    loom_serve_http_server_record_error(server_, error);
  }
  iree_status_free(error);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  ObservationClient resumed;
  resumed.Open(server_, "Last-Event-ID: " + cursor + "\r\n");
  EXPECT_NE(resumed.Frame().find("\"event\":\"gap\""), std::string::npos);
  EXPECT_NE(slow.Frame().find("stream probe"), std::string::npos);
  EXPECT_NE(slow.Frame().find("\"event\":\"gap\""), std::string::npos);
  EXPECT_EQ(Read(Send("GET", "/logz/stream", "", "Last-Event-ID: bad\r\n"))
                .find("HTTP/1.1 400"),
            0u);
  EXPECT_EQ(Read(Send("GET", "/logz/stream", "",
                      "Last-Event-ID: " + cursor +
                          "\r\nX-Loom-Cursor: different\r\n"))
                .find("HTTP/1.1 400"),
            0u);
  EXPECT_EQ(Read(Send("GET", "/logz/stream", "", "X-Loom-Model: a\r\n"))
                .find("HTTP/1.1 501"),
            0u);
  // Shutdown cancels both claimed streams without waiting for their credit.
  loom_serve_http_server_request_stop(server_);
  owner_.join();
  slow.Join();
  resumed.Join();
}

TEST_P(HttpRouterTest, IdleObserverDoesNotProduceCompletionDrivenEmptyFrames) {
  ObservationClient observer;
  observer.Open(server_);
  const auto first = observer.Frame();
  observer.WaitCredit(true);
  EXPECT_EQ(Read(Send("GET", "/healthz")).find("HTTP/1.1 200"), 0u);
  ObservationClient resumed;
  resumed.Open(server_,
               "X-Loom-Cursor: " + ObservationClient::Cursor(first) + "\r\n");
  EXPECT_EQ(ObservationClient::Cursor(resumed.Frame()),
            ObservationClient::Cursor(first));
  resumed.WaitCredit(true);
  observer.WaitCredit(true);
  // No additional publication is needed to release either reader.
  loom_serve_http_server_request_stop(server_);
  owner_.join();
  observer.Join();
  resumed.Join();
}

TEST_P(HttpRouterTest, MetricsCopyHeldAndQueuedOwnershipWithEscapedLabels) {
  const int held = Send("POST", "/v1/test", R"({"model":"held"})");
  held_.WaitSubmitted();
  const int queued = Send("POST", "/v1/test", R"({"model":"b"})");
  b_.WaitAccepted(1);
  const auto metrics = Read(Send("GET", "/metrics"));
  EXPECT_EQ(
      metrics,
      "HTTP/1.1 200 OK\r\n"
      "Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n"
      "Connection: close\r\n\r\n"
      R"(# HELP loom_serve_model_queued_requests Requests outside the active invocation or cohort.
# TYPE loom_serve_model_queued_requests gauge
loom_serve_model_queued_requests{model="a",kind="fixture"} 0
loom_serve_model_queued_requests{model="b",kind="fixture"} 1
loom_serve_model_queued_requests{model="held",kind="fixture"} 0
loom_serve_model_queued_requests{model="other",kind="fixture"} 0
loom_serve_model_queued_requests{model="quoted\"\\name",kind="fixture"} 0
# HELP loom_serve_model_active_requests Requests owned by the active invocation or cohort, not hardware utilization.
# TYPE loom_serve_model_active_requests gauge
loom_serve_model_active_requests{model="a",kind="fixture"} 0
loom_serve_model_active_requests{model="b",kind="fixture"} 0
loom_serve_model_active_requests{model="held",kind="fixture"} 1
loom_serve_model_active_requests{model="other",kind="fixture"} 0
loom_serve_model_active_requests{model="quoted\"\\name",kind="fixture"} 0
# HELP loom_serve_model_phase Current scheduler ownership phase.
# TYPE loom_serve_model_phase gauge
loom_serve_model_phase{model="a",kind="fixture",phase="idle"} 1
loom_serve_model_phase{model="b",kind="fixture",phase="idle"} 1
loom_serve_model_phase{model="held",kind="fixture",phase="held"} 1
loom_serve_model_phase{model="other",kind="fixture",phase="idle"} 1
loom_serve_model_phase{model="quoted\"\\name",kind="fixture",phase="idle"} 1
# HELP loom_serve_model_oldest_queued_age_seconds Time since the oldest queued request was accepted, or zero if empty.
# TYPE loom_serve_model_oldest_queued_age_seconds gauge
loom_serve_model_oldest_queued_age_seconds{model="a",kind="fixture"} 0
loom_serve_model_oldest_queued_age_seconds{model="b",kind="fixture"} 12.5
loom_serve_model_oldest_queued_age_seconds{model="held",kind="fixture"} 0
loom_serve_model_oldest_queued_age_seconds{model="other",kind="fixture"} 0
loom_serve_model_oldest_queued_age_seconds{model="quoted\"\\name",kind="fixture"} 0
# HELP loom_serve_model_oldest_active_age_seconds Time since the oldest active request was accepted, including prior queue residence, or zero if empty.
# TYPE loom_serve_model_oldest_active_age_seconds gauge
loom_serve_model_oldest_active_age_seconds{model="a",kind="fixture"} 0
loom_serve_model_oldest_active_age_seconds{model="b",kind="fixture"} 0
loom_serve_model_oldest_active_age_seconds{model="held",kind="fixture"} 2.25
loom_serve_model_oldest_active_age_seconds{model="other",kind="fixture"} 0
loom_serve_model_oldest_active_age_seconds{model="quoted\"\\name",kind="fixture"} 0
# HELP loom_serve_model_held_requests Requests held outside model admission.
# TYPE loom_serve_model_held_requests gauge
# HELP loom_serve_model_admission_open Whether new requests may enter the model.
# TYPE loom_serve_model_admission_open gauge
loom_serve_model_held_requests{model="a",kind="fixture"} 0
loom_serve_model_admission_open{model="a",kind="fixture"} 1
loom_serve_model_held_requests{model="b",kind="fixture"} 0
loom_serve_model_admission_open{model="b",kind="fixture"} 1
loom_serve_model_held_requests{model="held",kind="fixture"} 0
loom_serve_model_admission_open{model="held",kind="fixture"} 1
loom_serve_model_held_requests{model="other",kind="fixture"} 0
loom_serve_model_admission_open{model="other",kind="fixture"} 1
loom_serve_model_held_requests{model="quoted\"\\name",kind="fixture"} 0
loom_serve_model_admission_open{model="quoted\"\\name",kind="fixture"} 1
# HELP loom_serve_server_counters_available Every model publishes work counters.
# TYPE loom_serve_server_counters_available gauge
loom_serve_server_counters_available 0
# HELP loom_serve_model_requests_total Requests reaching a model protocol, including rejections.
# TYPE loom_serve_model_requests_total counter
# HELP loom_serve_model_completed_total Successful responses copied into the HTTP carrier.
# TYPE loom_serve_model_completed_total counter
# HELP loom_serve_model_rejected_total Protocol, source validation or capacity rejections.
# TYPE loom_serve_model_rejected_total counter
# HELP loom_serve_model_cancelled_total Requests abandoned by peers or orderly shutdown.
# TYPE loom_serve_model_cancelled_total counter
# HELP loom_serve_model_failed_total Requests terminated by transport or execution failures.
# TYPE loom_serve_model_failed_total counter
# HELP loom_serve_model_input_tokens_total Tokens in completed model work, including abandoned work.
# TYPE loom_serve_model_input_tokens_total counter
# HELP loom_serve_model_output_tokens_total Selected model output tokens, including EOS.
# TYPE loom_serve_model_output_tokens_total counter
# HELP loom_serve_model_batches_total Completed model invocations or cohorts, not HTTP requests.
# TYPE loom_serve_model_batches_total counter
)");
  const auto log = Read(Send("GET", "/logz"));
  EXPECT_NE(log.find("\"counters_available\":false"), std::string::npos);
  EXPECT_EQ(log.find("\"event\":\"counters\""), std::string::npos);
  // Scraping changes no model ownership and does not wait on the held work.
  EXPECT_NE(
      Read(Send("GET", "/v1/status"))
          .find(R"("phase":"held","queued_requests":0,"active_requests":1)"),
      std::string::npos);
  held_.Release();
  EXPECT_NE(Read(held).find("held:/v1/test:"), std::string::npos);
  EXPECT_NE(Read(queued).find("b:/v1/test:"), std::string::npos);
  const auto idle = Read(Send("GET", "/metrics"));
  EXPECT_NE(idle.find("loom_serve_model_active_requests{model=\"held\","
                      "kind=\"fixture\"} 0\n"),
            std::string::npos);
  EXPECT_NE(idle.find("loom_serve_model_queued_requests{model=\"b\","
                      "kind=\"fixture\"} 0\n"),
            std::string::npos);
  EXPECT_NE(idle.find("loom_serve_model_phase{model=\"held\","
                      "kind=\"fixture\",phase=\"idle\"} 1\n"),
            std::string::npos);
  EXPECT_EQ(idle.find("phase=\"held\""), std::string::npos);
  EXPECT_NE(
      idle.find("loom_serve_model_oldest_active_age_seconds{model=\"held\","
                "kind=\"fixture\"} 0\n"),
      std::string::npos);
  EXPECT_NE(idle.find("loom_serve_model_oldest_queued_age_seconds{model=\"b\","
                      "kind=\"fixture\"} 0\n"),
            std::string::npos);
}

TEST_P(HttpRouterTest, UnrelatedCapacityNeverWaitsForHeldCompletion) {
  const int held = Send("POST", "/v1/test", R"({"model":"other"})");
  other_.WaitSubmitted();
  // Both dependencies honor SHARED_ACTIVITY. The held service has no shared
  // domain: neither a named owner nor another NULL identity can wait on it.
  EXPECT_NE(
      Read(Send("POST", "/v1/test", R"({"model":"b"})")).find("b:/v1/test:"),
      std::string::npos);
  EXPECT_NE(Read(Send("POST", "/v1/test", R"({"model":"quoted\"\\name"})"))
                .find(std::string(escaped_.name) + ":/v1/test:"),
            std::string::npos);
  const auto state = Read(Send("GET", "/v1/status"));
  char storage[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  iree_string_view_t address;
  IREE_EXPECT_OK(
      iree_async_address_format(loom_serve_http_listener_address(listener_),
                                sizeof(storage), storage, &address));
  const std::string expected =
      "\"server\":{\"worker_count\":" + std::to_string(GetParam()) +
      R"(,"model_capacity":5,"connection_capacity":64,"request_limits":{"header_byte_capacity":16384,"body_byte_capacity":8388608,"header_capacity":64},"held_request_capacity":2,"held_requests":0,"observer_capacity":63,"observers":0,"resources":{"listeners":[{"address":")" +
      std::string(address.data, address.size) +
      R"(","waiting_for_capacity":false}],"devices":[]}})";
  EXPECT_NE(state.find(expected), std::string::npos) << state;
  EXPECT_NE(
      state.find(R"("phase":"held","queued_requests":0,"active_requests":1)"),
      std::string::npos);
  other_.Release();
  EXPECT_NE(Read(held).find("other:/v1/test:"), std::string::npos);
}

TEST_P(HttpRouterTest, RawUtilitiesProgressWhileTheSelectedModelIsHeld) {
  const int held = Send("POST", "/v1/test", R"({"model":"held"})");
  held_.WaitSubmitted();
  std::vector<int> clients;
  for (int i = 0; i < 12; ++i) {
    const char* name = i % 2 ? "b" : "held";
    clients.push_back(
        Send("POST", "/tokenize",
             std::string("{\"model\":\"") + name + "\",\"content\":\"aab\"}"));
  }
  EXPECT_NE(Read(Send("GET", "/healthz")).find("\"status\":\"ready\""),
            std::string::npos);
  EXPECT_NE(Read(Send("GET", "/readyz")).find(R"({"status":"ready"})"),
            std::string::npos);
  const auto snapshot = Read(Send("GET", "/v1/status"));
  EXPECT_NE(
      snapshot.find(
          R"({"object":"model","id":"held","loom":{"kind":"fixture","endpoint":"/v1/test","tokenize":true,"status":{"phase":"held","queued_requests":0,"active_requests":1,"oldest_queued_age_seconds":0.000000000,"oldest_active_age_seconds":2.250000000,"admission":"open","held_requests":0,"tokenizer_requests":)"),
      std::string::npos);
  for (int i = 0; i < 12; ++i) {
    const auto response = Read(clients[i]);
    EXPECT_EQ(response.find("HTTP/1.1 200"), 0u);
    EXPECT_NE(response.find(i % 2 ? R"({"tokens":[1,1,0]})"
                                  : R"({"tokens":[0,0,1]})"),
              std::string::npos);
  }
  const auto decoded =
      Read(Send("POST", "/detokenize", R"({"model":"held","tokens":[1,0,0]})"));
  // No utility reached the model dependency or required its device frontier.
  held_.Release();
  EXPECT_NE(decoded.find(R"({"content":"baa"})"), std::string::npos);
  if (decoded.find("HTTP/1.1 200") == 0) {
    EXPECT_NE(Read(held).find("held:/v1/test:"), std::string::npos);
  } else {
    // A broken utility may stop the owner; release its held claim in teardown
    // instead of circularly waiting for a response after that failure.
    EXPECT_EQ(close(held), 0);
  }
}

TEST_P(HttpRouterTest, UtilityRejectionsAndResetPeersPreserveTheOwner) {
  EXPECT_EQ(
      Read(Send("POST", "/tokenize", R"({"model":"other","content":"a"})"))
          .find("HTTP/1.1 404"),
      0u);
  // This deployment has an encoder but no decoder. Its CPU failure belongs
  // to this request and must leave every model and later utility usable.
  EXPECT_EQ(Read(Send("POST", "/detokenize", R"({"model":"b","tokens":[0]})"))
                .find("HTTP/1.1 500"),
            0u);
  EXPECT_EQ(Read(Send("POST", "/tokenize",
                      R"({"model":"a","content":"a","with_pieces":true})"))
                .find("HTTP/1.1 400"),
            0u);
  EXPECT_EQ(Read(Send("POST", "/detokenize", R"({"model":"a","tokens":[2]})"))
                .find("HTTP/1.1 400"),
            0u);
  EXPECT_EQ(Read(Send("POST", "/tokenize",
                      "{\"model\":\"a\",\"content\":\"" +
                          std::string(256 * 1024 + 1, 'a') + "\"}"))
                .find("HTTP/1.1 413"),
            0u);
  for (int i = 0; i < 8; ++i) {
    const int client = Send("POST", "/tokenize",
                            "{\"model\":\"a\",\"content\":\"" +
                                std::string(64 * 1024, 'a') + "\"}");
    const linger reset = {1, 0};
    EXPECT_EQ(setsockopt(client, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
              0);
    EXPECT_EQ(close(client), 0);
  }
  EXPECT_NE(Read(Send("POST", "/tokenize", R"({"model":"a","content":"ab"})"))
                .find(R"({"tokens":[0,1]})"),
            std::string::npos);
  EXPECT_NE(
      Read(Send("POST", "/v1/test", R"({"model":"b"})")).find("b:/v1/test:"),
      std::string::npos);
}

TEST_P(HttpRouterTest, DiscoveryAndIndependentBorrowedRequests) {
  EXPECT_NE(Read(Send("GET", "/healthz")).find(R"({"status":"ready"})"),
            std::string::npos);
  const auto models = Read(Send("GET", "/v1/models"));
  EXPECT_NE(
      models.find(
          R"({"object":"model","id":"a","loom":{"kind":"fixture","endpoint":"/v1/test","tokenize":true}})"),
      std::string::npos);
  EXPECT_NE(
      models.find(
          R"({"object":"model","id":"other","loom":{"kind":"fixture","endpoint":"/v1/test","tokenize":false}})"),
      std::string::npos);
  EXPECT_EQ(models.find("\"status\":"), std::string::npos);
  const std::string body_a = R"({"model":"\u0061","messages":["first"]})";
  const std::string body_b = R"({"model":"b","messages":["second"]})";
  const int a = Send("POST", "/v1/chat/completions", body_a);
  const int b = Send("POST", "/v1/chat/completions", body_b);
  EXPECT_NE(Read(b).find("b:/v1/chat/completions:" + body_b),
            std::string::npos);
  EXPECT_NE(Read(a).find("a:/v1/chat/completions:" + body_a),
            std::string::npos);
  EXPECT_NE(
      Read(Send("POST", "/v1/checkpoints/same", "", "X-Loom-Model: b\r\n"))
          .find("b:/v1/checkpoints/same:"),
      std::string::npos);
  EXPECT_NE(
      Read(Send("DELETE", "/v1/checkpoints/same", "", "X-Loom-Model: a\r\n"))
          .find("a:/v1/checkpoints/same:"),
      std::string::npos);
}

TEST_P(HttpRouterTest, RejectsAmbiguousSelectorsWithoutInvokingModels) {
  for (const auto* body : {"{}", R"({"model":null})", R"({"model":4})",
                           R"({"model":""})", R"({"model":"a","model":"b"})",
                           R"({"model":"a"} trailing)", R"({"model":"\q"})"}) {
    SCOPED_TRACE(body);
    EXPECT_EQ(
        Read(Send("POST", "/v1/chat/completions", body)).find("HTTP/1.1 400"),
        0u);
  }
  EXPECT_EQ(Read(Send("POST", "/v1/chat/completions", R"({"model":"missing"})"))
                .find("HTTP/1.1 404"),
            0u);
  EXPECT_EQ(Read(Send("POST", "/v1/checkpoints/same")).find("HTTP/1.1 400"),
            0u);
  EXPECT_EQ(Read(Send("DELETE", "/v1/checkpoints/same", "",
                      "X-Loom-Model: missing\r\n"))
                .find("HTTP/1.1 404"),
            0u);
  // Rejections leave the listener and both services available.
  EXPECT_NE(Read(Send("POST", "/v1/chat/completions", R"({"model":"b"})"))
                .find("b:/v1/chat/completions:"),
            std::string::npos);
}

TEST_P(HttpRouterTest, HeldModelLeavesIndependentRequestsRunnable) {
  const std::string body = R"({"model":"held","messages":["retained"]})";
  const int held = Send("POST", "/v1/chat/completions", body);
  held_.WaitSubmitted();
  EXPECT_NE(Read(Send("POST", "/v1/chat/completions", R"({"model":"a"})"))
                .find("a:/v1/chat/completions:"),
            std::string::npos);
  EXPECT_NE(Read(Send("GET", "/healthz")).find(R"({"status":"ready"})"),
            std::string::npos);
  held_.Release();
  EXPECT_NE(Read(held).find("held:/v1/chat/completions:" + body),
            std::string::npos);
}

TEST_P(HttpRouterTest, DrainFinishesAdmittedQueueAndHoldsLaterModelBorrowers) {
  const int held = Send("POST", "/v1/test", R"({"model":"held"})");
  held_.WaitSubmitted();
  const int first = Send("POST", "/v1/test", R"({"model":"b","turn":1})");
  const int second = Send("POST", "/v1/test", R"({"model":"b","turn":2})");
  b_.WaitAccepted(2);
  const auto draining = Control("drain", "b");
  EXPECT_NE(draining.find(R"("queued_requests":2,"active_requests":0)"),
            std::string::npos);
  EXPECT_NE(draining.find(R"("admission":"draining")"), std::string::npos);

  const int later = Send("POST", "/v1/test", R"({"model":"b","turn":3})");
  const int tokens =
      Send("POST", "/tokenize", R"({"model":"b","content":"aab"})");
  EXPECT_NE(WaitHeld("b", 2).find(R"("tokenizer_requests":0)"),
            std::string::npos);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"b"})")).find("HTTP/1.1 429"),
      0u);
  EXPECT_NE(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("a:/v1/test:"),
      std::string::npos);
  const auto metrics = Read(Send("GET", "/metrics"));
  EXPECT_NE(metrics.find("loom_serve_model_queued_requests{model=\"b\","
                         "kind=\"fixture\"} 2\n"),
            std::string::npos);
  EXPECT_NE(metrics.find("loom_serve_model_held_requests{model=\"b\","
                         "kind=\"fixture\"} 2\n"),
            std::string::npos);
  EXPECT_NE(metrics.find("loom_serve_model_admission_open{model=\"b\","
                         "kind=\"fixture\"} 0\n"),
            std::string::npos);

  held_.Release();
  EXPECT_NE(Read(held).find("held:/v1/test:"), std::string::npos);
  EXPECT_NE(Read(first).find(R"(b:/v1/test:{"model":"b","turn":1})"),
            std::string::npos);
  EXPECT_NE(Read(second).find(R"(b:/v1/test:{"model":"b","turn":2})"),
            std::string::npos);
  const auto drained = Control("drain", "b");
  EXPECT_NE(drained.find(R"("queued_requests":0,"active_requests":0)"),
            std::string::npos);
  EXPECT_NE(drained.find(R"("admission":"drained","held_requests":2)"),
            std::string::npos);
  {
    std::lock_guard<std::mutex> lock(b_.mutex);
    EXPECT_EQ(b_.accepted, 2u);
  }
  EXPECT_NE(Control("undrain", "b").find(R"("admission":"open")"),
            std::string::npos);
  EXPECT_NE(Read(later).find(R"(b:/v1/test:{"model":"b","turn":3})"),
            std::string::npos);
  EXPECT_NE(Read(tokens).find(R"({"tokens":[1,1,0]})"), std::string::npos);
}

TEST_P(HttpRouterTest, AdmissionControlRejectsInvalidSelectorsWithoutClosing) {
  for (const auto* operation : {"drain", "undrain"}) {
    const std::string target = std::string("/v1/models/") + operation;
    for (const auto* body :
         {"{}", R"({"model":null})", R"({"model":""})",
          R"({"model":"a","model":"b"})", R"({"model":"a","extra":true})",
          R"({"model":"a"} trailing)"}) {
      SCOPED_TRACE(body);
      EXPECT_EQ(Read(Send("POST", target, body)).find("HTTP/1.1 400"), 0u);
    }
    EXPECT_EQ(
        Read(Send("GET", target, R"({"model":"a"})")).find("HTTP/1.1 404"), 0u);
    EXPECT_EQ(Control(operation, "missing").find("HTTP/1.1 404"), 0u);
  }
  EXPECT_NE(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("a:/v1/test:"),
      std::string::npos);
}

TEST_P(HttpRouterTest, TrimCannotSelectAnExternallyOwnedDevice) {
  for (const auto* body : {"{}", R"({"device":0,"target_bytes":0})",
                           R"({"device":0,"target_bytes":-1})"}) {
    SCOPED_TRACE(body);
    EXPECT_EQ(Read(Send("POST", "/v1/devices/trim", body)).find("HTTP/1.1 400"),
              0u);
  }
  EXPECT_EQ(Read(Send("GET", "/v1/devices/trim", "")).find("HTTP/1.1 404"), 0u);
  EXPECT_NE(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("a:/v1/test:"),
      std::string::npos);
}

TEST_P(HttpRouterTest, DisconnectedHeldClaimReturnsItsAdmissionCredit) {
  EXPECT_NE(Control("drain", "a").find(R"("admission":"drained")"),
            std::string::npos);
  const int abandoned = Send("POST", "/v1/test", R"({"model":"a"})");
  WaitHeld("a", 1);
  const linger reset = {1, 0};
  EXPECT_EQ(setsockopt(abandoned, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
            0);
  EXPECT_EQ(close(abandoned), 0);
  WaitHeld("a", 0);
  const int first = Send("POST", "/v1/test", R"({"model":"a"})");
  const int second =
      Send("POST", "/tokenize", R"({"model":"a","content":"ab"})");
  WaitHeld("a", 2);
  EXPECT_EQ(Control("undrain", "a").find("HTTP/1.1 200"), 0u);
  EXPECT_NE(Read(first).find("a:/v1/test:"), std::string::npos);
  EXPECT_NE(Read(second).find(R"({"tokens":[0,1]})"), std::string::npos);
}

TEST_P(HttpRouterTest, ShutdownReleasesHeldClaimsWithoutAdmittingThem) {
  EXPECT_EQ(Control("drain", "a").find("HTTP/1.1 200"), 0u);
  const int client = Send("POST", "/v1/test", R"({"model":"a"})");
  WaitHeld("a", 1);
  loom_serve_http_server_request_stop(server_);
  owner_.join();
  EXPECT_EQ(close(client), 0);
  EXPECT_EQ(a_.accepted, 0u);
}

TEST_P(HttpRouterTest, RejectsAQuotaThatCouldExcludeEveryControlConnection) {
  for (const auto options : {
           loom_serve_router_options_t{.request_quota = 0,
                                       .holding_capacity = 0},
           loom_serve_router_options_t{
               .request_quota = 4, .holding_capacity = connection_capacity_},
           loom_serve_router_options_t{
               .request_quota = 4,
               .holding_capacity = connection_capacity_ + 1},
       }) {
    // Validation happens before the owner installs any transport observer.
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_router_run(executor_, server_, nullptr, options,
                              iree_allocator_system()));
  }
  EXPECT_EQ(Read(Send("GET", "/healthz")).find("HTTP/1.1 200"), 0u);
}

TEST_P(HttpRouterTest, ShutdownDrainsAcceptedReadinessBeforeDetaching) {
  const int held = Send("POST", "/v1/chat/completions", R"({"model":"held"})");
  held_.WaitSubmitted();
  loom_serve_http_server_request_stop(server_);
  held_.WaitStopping();
  EXPECT_FALSE(returned_.load());
  held_.Release();
  owner_.join();
  EXPECT_EQ(close(held), 0);
}

TEST_P(HttpRouterTest, FailedReadinessStillDrainsOtherAcceptedCallbacks) {
  const int held = Send("POST", "/v1/chat/completions", R"({"model":"held"})");
  const int other =
      Send("POST", "/v1/chat/completions", R"({"model":"other"})");
  held_.WaitSubmitted();
  other_.WaitSubmitted();
  expected_code_ = IREE_STATUS_UNAVAILABLE;
  held_.Release(iree_make_status(IREE_STATUS_UNAVAILABLE, "device failed"));
  other_.WaitStopping();
  EXPECT_FALSE(returned_.load());
  other_.Release();
  owner_.join();
  EXPECT_EQ(close(held), 0);
  EXPECT_EQ(close(other), 0);
}

// Real finite CPU task on the router's executor. Release, not completion,
// returns its stack storage to the test owner.
struct CpuCompletion {
  // Borrowed dependency receiving this task's readiness publication.
  ModelService* service;
  // Intrusive executor ownership, outliving its final release callback.
  iree_task_process_t process = {};
  // Protects final release and joins its last access.
  std::mutex mutex;
  // Wakes only the test thread, never a task worker.
  std::condition_variable condition;
  // Executor has returned all intrusive process storage.
  bool released = false;

  static iree_status_t Run(iree_task_process_t* process,
                           const iree_task_worker_context_t*,
                           iree_task_process_drain_result_t* result) {
    auto* self = static_cast<CpuCompletion*>(process->user_data);
    self->service->Release();
    result->completed = true;
    result->did_work = true;
    return iree_ok_status();
  }
  static void Released(iree_task_process_t* process) {
    auto* self = static_cast<CpuCompletion*>(process->user_data);
    std::lock_guard<std::mutex> lock(self->mutex);
    self->released = true;
    self->condition.notify_all();
  }
  void Submit(iree_task_executor_t* executor) {
    iree_task_process_initialize(Run, 0, 1, &process);
    process.user_data = this;
    process.release_fn = Released;
    iree_task_executor_schedule_process(executor, &process);
  }
  void AwaitRelease() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return released; });
  }
};

TEST_P(HttpRouterTest, ContinuouslyReadyOwnerYieldsToSharedCpuWork) {
  {
    std::lock_guard<std::mutex> lock(held_.mutex);
    held_.advance_ready_work = true;
  }
  const std::string body = R"({"model":"held"})";
  const int held = Send("POST", "/v1/chat/completions", body);
  held_.WaitSubmitted();
  EXPECT_NE(Read(Send("GET", "/healthz")).find(R"({"status":"ready"})"),
            std::string::npos);
  CpuCompletion cpu{&held_};
  cpu.Submit(executor_);
  EXPECT_NE(Read(held).find("held:/v1/chat/completions:" + body),
            std::string::npos);
  cpu.AwaitRelease();
  std::lock_guard<std::mutex> lock(held_.mutex);
  EXPECT_GT(held_.ready_units, 0u);
}

class HttpRouterCountersTest : public HttpRouterTest {
 protected:
  void SetUp() override {
    size_t index = 0;
    for (auto* service : {&a_, &b_, &held_, &other_, &escaped_}) {
      const uint64_t value = index + 1;
      counters_[index] = {.requests = value,
                          .completed = value,
                          .input_tokens = (uint64_t{1} << 53) + value,
                          .output_tokens = 7 * value,
                          .batches = value};
      service->counters = &counters_[index++];
    }
    HttpRouterTest::SetUp();
  }
  // Distinct service owners, unchanged by cold observation/utility requests.
  loom_serve_telemetry_counters_t counters_[5] = {};
};

class HttpRouterTightCapacityTest : public HttpRouterTest {
 protected:
  void SetUp() override {
    connection_capacity_ = 3;
    HttpRouterTest::SetUp();
  }
};

TEST_P(HttpRouterTightCapacityTest,
       FullHoldingBudgetLeavesAnUndrainConnection) {
  EXPECT_EQ(Control("drain", "a").find("HTTP/1.1 200"), 0u);
  const int first = Send("POST", "/v1/test", R"({"model":"a"})");
  WaitHeld("a", 1);
  const int second =
      Send("POST", "/tokenize", R"({"model":"a","content":"ab"})");
  WaitHeld("a", 2);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 429"),
      0u);
  EXPECT_EQ(Control("undrain", "a").find("HTTP/1.1 200"), 0u);
  EXPECT_NE(Read(first).find("a:/v1/test:"), std::string::npos);
  EXPECT_NE(Read(second).find(R"({"tokens":[0,1]})"), std::string::npos);
}

TEST_P(HttpRouterTightCapacityTest, ObservationAndHoldingShareControlCredit) {
  EXPECT_EQ(Control("drain", "a").find("HTTP/1.1 200"), 0u);
  const int held = Send("POST", "/v1/test", R"({"model":"a"})");
  WaitHeld("a", 1);
  {
    ObservationClient observer;
    observer.Open(server_);
    observer.WaitCredit(false);
    const auto status = Read(Send("GET", "/v1/status"));
    EXPECT_NE(
        status.find(R"("held_requests":1,"observer_capacity":2,"observers":1)"),
        std::string::npos)
        << status;
    EXPECT_EQ(Read(Send("GET", "/logz/stream")).find("HTTP/1.1 429"), 0u);
    EXPECT_EQ(
        Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 429"),
        0u);
    EXPECT_EQ(Control("undrain", "a").find("HTTP/1.1 200"), 0u);
    EXPECT_NE(Read(held).find("a:/v1/test:"), std::string::npos);
    // Client departure returns observation ownership independently of shutdown.
  }
  EXPECT_EQ(Read(Send("GET", "/healthz")).find("HTTP/1.1 200"), 0u);
}

class HttpRouterNoHoldingTest : public HttpRouterTest {
 protected:
  void SetUp() override {
    holding_capacity_ = 0;
    HttpRouterTest::SetUp();
  }
};

TEST_P(HttpRouterNoHoldingTest, ClosedAdmissionRejectsWithoutRetainingClaims) {
  EXPECT_EQ(Control("drain", "a").find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 429"),
      0u);
  EXPECT_EQ(Read(Send("POST", "/tokenize", R"({"model":"a","content":"a"})"))
                .find("HTTP/1.1 429"),
            0u);
  EXPECT_NE(Control("undrain", "a").find(R"("held_requests":0)"),
            std::string::npos);
  EXPECT_NE(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("a:/v1/test:"),
      std::string::npos);
}

TEST_P(HttpRouterCountersTest, ServerTotalsPreserveIntegerPrecisionAndScope) {
  for (int i = 0; i < 2; ++i) {
    const auto metrics = Read(Send("GET", "/metrics"));
    EXPECT_NE(metrics.find("loom_serve_model_input_tokens_total{model=\"a\","
                           "kind=\"fixture\"} 9007199254740993\n"),
              std::string::npos);
    EXPECT_NE(metrics.find("loom_serve_server_requests_total 15\n"),
              std::string::npos);
    EXPECT_NE(metrics.find(
                  "loom_serve_server_input_tokens_total 45035996273704975\n"),
              std::string::npos);
    const auto log = Read(Send("GET", "/logz"));
    EXPECT_NE(log.find("\"counters_available\":true"), std::string::npos);
    EXPECT_NE(
        log.find("\"values\":{\"requests\":15,\"completed\":15,\"rejected\":0,"
                 "\"cancelled\":0,\"failed\":0,\"input_tokens\":"
                 "45035996273704975,\"output_tokens\":105,\"batches\":15}"),
        std::string::npos);
    EXPECT_NE(Read(Send("POST", "/tokenize", R"({"model":"a","content":"ab"})"))
                  .find("\"tokens\":[0,1]"),
              std::string::npos);
    EXPECT_NE(Read(Send("POST", "/v1/test", R"({"model":"missing"})"))
                  .find("404 Not Found"),
              std::string::npos);
  }
}

INSTANTIATE_TEST_SUITE_P(Workers, HttpRouterTest, ::testing::Values(1, 4));
INSTANTIATE_TEST_SUITE_P(Workers, HttpRouterCountersTest,
                         ::testing::Values(1, 4));
INSTANTIATE_TEST_SUITE_P(Workers, HttpRouterTightCapacityTest,
                         ::testing::Values(1, 4));
INSTANTIATE_TEST_SUITE_P(Workers, HttpRouterNoHoldingTest,
                         ::testing::Values(1, 4));

}  // namespace
