// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/service_test_fixture.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::RegisteredSourceTextServiceTest;
using loom_serve::testing::text::SourceTextServiceTest;
using loom_serve::testing::text::StalledClient;

class TextServiceCapacityTest : public SourceTextServiceTest {
 protected:
  loom_serve_http_server_options_t ServerOptions() const override {
    auto options = SourceTextServiceTest::ServerOptions();
    options.connection_capacity = 128;
    return options;
  }

  loom_serve_text_options_t ModelOptions() const override {
    auto options = SourceTextServiceTest::ModelOptions();
    options.prefill_capacity = 128;
    options.pool_capacity = 4096;
    options.epoch_count = IREE_ARRAYSIZE(shapes_);
    options.epoch_shapes = shapes_;
    return options;
  }

  loom_serve_text_service_options_t ServiceOptions() const override {
    auto options = SourceTextServiceTest::ServiceOptions();
    options.pending_capacity = 128;
    return options;
  }

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(SourceTextServiceTest::SetUp());
    const auto options = ServiceOptions();
    IREE_ASSERT_OK(loom_serve_text_service_allocate(
        model_, &options, nullptr, host_allocator_, &service_));
    interface_ = loom_serve_text_service_interface(service_);
    Observe(interface_);
  }

  void TearDown() override {
    if (service_) {
      // Even an assertion failure joins accepted CPU/device work before free.
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

  void Submit(size_t count, std::vector<int>* clients) {
    for (size_t i = 0; i < count; ++i) {
      const auto prompt = std::string(1, "abx"[i % 3]);
      const auto name =
          "X-Loom-Session: resident-" + std::to_string(i) + "\r\n";
      clients->push_back(Send(Request(Message("user", prompt), 8), name));
      loom_serve_http_connection_t* connection = nullptr;
      ASSERT_NO_FATAL_FAILURE(Accept(clients->back(), interface_, &connection));
    }
  }

  // One cohort is independent of the four initial residents.
  const loom_serve_packing_shape_t shapes_[1] = {{128, 128}};
  // Service owns only request state; the parent fixture retains its model.
  loom_serve_text_service_t* service_ = nullptr;
  // Borrowed public scheduler callbacks used by the real HTTP router.
  loom_serve_service_t interface_ = {};
};

TEST_F(TextServiceCapacityTest, HttpDemandGrowsWithoutDuplicatingTheModel) {
  ASSERT_EQ(loom_serve_text_model_row_count(model_), 4u);
  std::vector<int> clients;
  ASSERT_NO_FATAL_FAILURE(Submit(100, &clients));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
    return interface_.snapshot(interface_.self).active_requests ==
               clients.size() ||
           interface_.telemetry.counters->completed;
  }));
  EXPECT_EQ(interface_.snapshot(interface_.self).active_requests, 100u);
  EXPECT_EQ(interface_.telemetry.counters->completed, 0u);
  EXPECT_GE(loom_serve_text_model_row_count(model_), 100u);
  const auto first_outputs = [&] {
    uint64_t count = 0;
    for (uint64_t bucket : interface_.telemetry.latency->first_output.buckets) {
      count += bucket;
    }
    return count;
  };
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
    return first_outputs() == clients.size() ||
           interface_.telemetry.counters->completed;
  }));
  EXPECT_EQ(first_outputs(), 100u);
  EXPECT_EQ(interface_.telemetry.counters->completed, 0u);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
  for (size_t i = 0; i < clients.size(); ++i) {
    EXPECT_EQ(Content(Read(clients[i])),
              Oracle::Complete(std::string(1, "abx"[i % 3]), 8))
        << i;
  }
  EXPECT_EQ(interface_.telemetry.counters->completed, 100u);
  EXPECT_EQ(interface_.telemetry.counters->failed, 0u);
  EXPECT_EQ(loom_serve_text_model_weight_activity(model_).activations, 1u);

  // The last named history survives directory publication and resumes through
  // the same route and resident, without reconstructing its consumed prefix.
  const std::string response = Oracle::Complete("a", 8);
  const std::string request =
      Request(Message("user", "a") + "," + Message("assistant", response) +
                  "," + Message("user", "b"),
              8);
  const int resumed = Send(request, "X-Loom-Session: resident-99\r\n");
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(resumed, interface_, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
  EXPECT_EQ(Content(Read(resumed)), Oracle::Complete("a" + response + "#b", 8));
}

TEST_F(TextServiceCapacityTest, StoppingJoinsAcceptedCapacityGrowth) {
  std::vector<int> clients;
  ASSERT_NO_FATAL_FAILURE(Submit(5, &clients));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
    return std::string(interface_.snapshot(interface_.self).phase) == "grow";
  }));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      interface_, [&] { return Idle(); }, LOOM_SERVE_SERVICE_FLAG_STOPPING));
  // Stopping joins the host task and discards its unbound session records;
  // there is no reason to acquire native storage for cancelled demand.
  EXPECT_EQ(loom_serve_text_model_row_count(model_), 4u);
  EXPECT_EQ(interface_.telemetry.counters->cancelled, 5u);
  EXPECT_EQ(interface_.telemetry.counters->failed, 0u);
  for (int client : clients) {
    Read(client, true);
  }
}

TEST_F(TextServiceCapacityTest, RejectsMaximumBelowPreparedResidentCapacity) {
  auto options = ServiceOptions();
  options.max_sessions = options.row_count - 1;
  loom_serve_text_service_t* limited = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_service_allocate(model_, &options, nullptr,
                                       host_allocator_, &limited));
  EXPECT_EQ(limited, nullptr);
}

class LimitedTextServiceCapacityTest
    : public TextServiceCapacityTest,
      public ::testing::WithParamInterface<iree_host_size_t> {
 protected:
  loom_serve_text_options_t ModelOptions() const override {
    auto options = TextServiceCapacityTest::ModelOptions();
    options.row_count = std::min(options.row_count, GetParam());
    return options;
  }

  loom_serve_text_service_options_t ServiceOptions() const override {
    auto options = TextServiceCapacityTest::ServiceOptions();
    options.row_count = std::min(options.row_count, GetParam());
    options.max_sessions = GetParam();
    return options;
  }
};

TEST_P(LimitedTextServiceCapacityTest,
       BurstReusesOwnersWithoutExceedingMaximum) {
  std::vector<int> clients;
  ASSERT_NO_FATAL_FAILURE(Submit(13, &clients));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
    return interface_.snapshot(interface_.self).active_requests == GetParam() ||
           interface_.telemetry.counters->completed;
  }));
  EXPECT_EQ(interface_.snapshot(interface_.self).active_requests, GetParam());
  EXPECT_EQ(interface_.snapshot(interface_.self).queued_requests,
            clients.size() - GetParam());
  EXPECT_EQ(interface_.telemetry.counters->completed, 0u);
  EXPECT_EQ(loom_serve_text_model_row_count(model_), GetParam());
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
    EXPECT_LE(interface_.snapshot(interface_.self).active_requests, GetParam());
    EXPECT_EQ(loom_serve_text_model_row_count(model_), GetParam());
    return Idle();
  }));
  for (size_t i = 0; i < clients.size(); ++i) {
    EXPECT_EQ(Content(Read(clients[i])),
              Oracle::Complete(std::string(1, "abx"[i % 3]), 8))
        << i;
  }
  EXPECT_EQ(interface_.telemetry.counters->completed, clients.size());
  EXPECT_EQ(interface_.telemetry.counters->failed, 0u);
  EXPECT_EQ(loom_serve_text_model_weight_activity(model_).activations, 1u);
}

INSTANTIATE_TEST_SUITE_P(DeploymentLimits, LimitedTextServiceCapacityTest,
                         ::testing::Values(1, 4, 6));

TEST_F(RegisteredSourceTextServiceTest,
       SessionMaximumRetainsSlowOutputUntilCancellation) {
  StartManaged();
  EXPECT_EQ(Read(Send(Declaration("fixture-text", directory_, "open",
                                  R"(,"max_sessions":1,"pending_requests":1)"),
                      "", "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  StalledClient stalled;
  ASSERT_NO_FATAL_FAILURE(
      stalled.Send(server_, Request(Message("user", "abxa"), 4)));
  WaitModelState("fixture-text",
                 R"("phase":"idle","queued_requests":0,"active_requests":1)");
  const int pending = Send(Request(Message("user", "b"), 2));
  WaitModelState("fixture-text",
                 R"("phase":"idle","queued_requests":1,"active_requests":1)");
  // A full waiting queue rejects excess work, rather than growing another
  // resident to bypass the slow consumer's ownership.
  EXPECT_EQ(Read(Send(Request(Message("user", "x"), 2))).find("HTTP/1.1 503"),
            0u);
  // The ceiling belongs to this deployment, not the model's source identity
  // or shared device. A peer with the same programs can still serve work.
  EXPECT_EQ(
      Read(Send(Declaration("peer", directory_, "open", R"(,"max_sessions":1)"),
                "", "/v1/models/register"))
          .find("HTTP/1.1 202"),
      0u);
  const auto peer = Read(Send(
      R"({"model":"peer","stream":true,"max_tokens":2,"messages":[{"role":"user","content":"x"}]})"));
  EXPECT_EQ(Content(peer), Oracle::Complete("x", 2));
  ASSERT_NO_FATAL_FAILURE(stalled.Close());
  EXPECT_EQ(Content(Read(pending)), Oracle::Complete("b", 2));
  WaitModelState("fixture-text",
                 R"("phase":"idle","queued_requests":0,"active_requests":0)");
}

}  // namespace
