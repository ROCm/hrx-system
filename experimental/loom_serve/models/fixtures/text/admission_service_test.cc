// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/models/fixtures/text/service_fixture.h"
#include "experimental/loom_serve/text/telemetry.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;
using loom_serve::testing::text::StalledClient;

class AdmissionServiceTest : public SourceTextServiceTest {
 protected:
  enum class History { kFresh, kRetained, kCached, kPinned };

  loom_serve_text_options_t ModelOptions() const override {
    auto options = SourceTextServiceTest::ModelOptions();
    options.row_count = 2;
    options.epoch_shapes = admission_shapes_;
    options.pool_capacity = 64;
    return options;
  }

  loom_serve_text_service_options_t ServiceOptions() const override {
    auto options = SourceTextServiceTest::ServiceOptions();
    options.row_count = 2;
    return options;
  }

  void CheckBackfill(size_t budget, OverlapAction action,
                     History history = History::kFresh) {
    auto options = ServiceOptions();
    options.admission_overtakes = budget;
    options.session_cache_capacity = history == History::kCached ? 1 : 0;
    loom_serve_text_service_t* value = nullptr;
    IREE_ASSERT_OK(loom_serve_text_service_create(model_, &options, nullptr,
                                                  &value, allocator_));
    auto destroy = [](loom_serve_text_service_t* service) {
      loom_serve_text_service_destroy(service, IREE_STATUS_OK);
    };
    std::unique_ptr<loom_serve_text_service_t, decltype(destroy)> service(
        value, destroy);
    const auto interface = loom_serve_text_service_interface(value);
    Observe(interface);
    auto quiescent = [&] {
      return !interface.is_pending(interface.self) &&
             std::string(interface.snapshot(interface.self).phase) == "idle";
    };
    auto idle = [&] {
      return !interface.is_pending(interface.self) &&
             !interface.is_active(interface.self) &&
             !interface.snapshot(interface.self).queued_requests;
    };
    loom_serve_http_connection_t* connection = nullptr;
    std::string head_prompt(32, 'a');
    std::string head_messages = Message("user", head_prompt);
    std::string head_headers;
    size_t head_limit = 33;
    if (history != History::kFresh) {
      // A whole four-position page can be shared without partial-tail COW;
      // the full-pool head remains feasible after the active peer retires.
      const int warmup =
          Send(Request(Message("user", "abxa"), 1), "X-Loom-Session: kept\r\n");
      ASSERT_NO_FATAL_FAILURE(Accept(interface, &connection));
      ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, idle));
      const auto response = Content(Read(warmup));
      ASSERT_EQ(response, Oracle::Complete("abxa", 1));
      head_prompt = "abxa" + response + "#a";
      head_messages = Message("user", "abxa") + "," +
                      Message("assistant", response) + "," +
                      Message("user", "a");
      head_headers = "X-Loom-Session: kept\r\n";
      head_limit = 65 - head_prompt.size();
      if (history == History::kPinned) {
        const int pin = Send("", head_headers, "/v1/checkpoints/base");
        ASSERT_NO_FATAL_FAILURE(Accept(interface, &connection));
        ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, idle));
        const auto response = Read(pin);
        ASSERT_EQ(response.find("HTTP/1.1 201"), 0u) << response;
        head_headers += "X-Loom-Checkpoint: base\r\n";
      }
    }
    const auto initial = *interface.telemetry.counters;
    const uint64_t head_id = initial.requests + 2;

    // One row retains 28 positions while its only network credit is held.
    // The other row must serve fitting tails, not be occupied by a blocked
    // request's host preparation. These are real transport and native loans.
    StalledClient stalled;
    ASSERT_NO_FATAL_FAILURE(
        stalled.Send(server_, Request(Message("user", "abxa"), 25)));
    loom_serve_http_connection_t* active = nullptr;
    ASSERT_NO_FATAL_FAILURE(Accept(interface, &active));
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, quiescent));
    ASSERT_EQ(interface.telemetry.counters->output_tokens,
              initial.output_tokens + 1);
    ASSERT_FALSE(loom_serve_http_connection_can_send(active));
    ASSERT_EQ(loom_serve_text_model_pool_usage(model_).available,
              history == History::kFresh ? 36u : 32u);

    const int head = Send(Request(head_messages, head_limit), head_headers);
    loom_serve_http_connection_t* head_connection = nullptr;
    ASSERT_NO_FATAL_FAILURE(Accept(interface, &head_connection));
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, quiescent));
    ASSERT_EQ(interface.snapshot(interface.self).queued_requests, 1u);
    for (size_t i = 0; i < budget; ++i) {
      const int tail = Send(Request(Message("user", "a"), 4));
      ASSERT_NO_FATAL_FAILURE(Accept(interface, &connection));
      ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, [&] {
        return interface.telemetry.counters->completed ==
                   initial.completed + i + 1 &&
               quiescent();
      }));
      EXPECT_EQ(Content(Read(tail)), Oracle::Complete("a", 4));
      EXPECT_EQ(interface.snapshot(interface.self).queued_requests, 1u);
      EXPECT_EQ(interface.snapshot(interface.self).active_requests, 1u);
      EXPECT_FALSE(loom_serve_http_connection_can_send(active));
    }

    // A new arrival cannot replenish the oldest request's overtake budget.
    const int extra = Send(Request(Message("user", "b"), 4));
    ASSERT_NO_FATAL_FAILURE(Accept(interface, &connection));
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, quiescent));
    bool progress = false;
    IREE_ASSERT_OK(interface.advance(interface.self, 0, &progress));
    EXPECT_FALSE(progress);
    EXPECT_TRUE(quiescent());
    EXPECT_EQ(interface.snapshot(interface.self).queued_requests, 2u);
    EXPECT_EQ(interface.telemetry.counters->completed,
              initial.completed + budget);

    // Existing telemetry identifies the exact policy choices and proves the
    // unchanged oldest input was prepared once despite all intervening work.
    uint64_t cursor = 0;
    uint64_t prepared_head = 0, bypasses = 0;
    const auto end =
        loom_serve_telemetry_stream_head(interface.telemetry.stream);
    while (cursor < end) {
      loom_serve_telemetry_record_t record;
      ASSERT_EQ(loom_serve_telemetry_stream_read(interface.telemetry.stream,
                                                 &cursor, &record),
                LOOM_SERVE_TELEMETRY_READ_RECORD);
      if (std::string(record.event->name) == "prepared" &&
          record.request == head_id) {
        ++prepared_head;
      }
      if (std::string(record.event->name) == "admit" &&
          record.request > head_id) {
        loom_serve_text_admission_record_t admission;
        memcpy(&admission, record.data.words, sizeof(admission));
        ++bypasses;
        EXPECT_EQ(admission.queue_rank, 1u);
        EXPECT_EQ(admission.overtakes, bypasses);
        EXPECT_EQ(admission.overtake_budget, budget);
      }
    }
    EXPECT_EQ(prepared_head, history == History::kRetained ? 2u : 1u);
    EXPECT_EQ(bypasses, budget);

    if (action == OverlapAction::kStop) {
      ASSERT_NO_FATAL_FAILURE(
          AdvanceUntil(interface, idle, LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING));
      EXPECT_EQ(interface.telemetry.counters->cancelled, 3u);
      EXPECT_TRUE(Read(head).empty());
      EXPECT_TRUE(Read(extra).empty());
      ASSERT_NO_FATAL_FAILURE(stalled.Close());
    } else {
      if (action == OverlapAction::kDisconnect) {
        const linger reset = {1, 0};
        ASSERT_EQ(
            setsockopt(head, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)), 0);
        ASSERT_EQ(close(head), 0);
        iree_notification_await(
            loom_serve_http_server_notification(server_),
            [](void* self) {
              return loom_serve_http_connection_failed(
                  static_cast<loom_serve_http_connection_t*>(self));
            },
            head_connection, iree_infinite_timeout());
        ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, [&] {
          return interface.telemetry.counters->completed ==
                     initial.completed + budget + 1 &&
                 quiescent();
        }));
        EXPECT_FALSE(loom_serve_http_connection_can_send(active));
      }
      std::string response;
      std::thread reader([&] {
        response = stalled.Read();
        stalled.Close();
      });
      EXPECT_NO_FATAL_FAILURE(AdvanceUntil(interface, idle));
      reader.join();
      EXPECT_EQ(Content(response), Oracle::Complete("abxa", 25));
      if (action == OverlapAction::kComplete) {
        const auto response = Read(head);
        EXPECT_EQ(Content(response), Oracle::Complete(head_prompt, head_limit));
        EXPECT_NE(response.find(history == History::kPinned ||
                                        history == History::kCached
                                    ? "\"cached_tokens\":4}"
                                    : "\"cached_tokens\":0}"),
                  std::string::npos);
      }
      EXPECT_EQ(Content(Read(extra)), Oracle::Complete("b", 4));
      EXPECT_EQ(interface.telemetry.counters->completed,
                initial.completed + budget +
                    (action == OverlapAction::kComplete ? 3 : 2));
      EXPECT_EQ(interface.telemetry.counters->cancelled,
                action == OverlapAction::kDisconnect ? 1u : 0u);
    }
    EXPECT_EQ(interface.telemetry.counters->failed, 0u);
    EXPECT_EQ(interface.telemetry.counters->rejected, 0u);
    EXPECT_EQ(loom_serve_text_model_pool_usage(model_).pending, 0u);
    interface.set_observer(interface.self, {});
  }

  void CheckBorrowedProbeCancellation(OverlapAction action) {
    auto options = ServiceOptions();
    options.admission_overtakes = 2;
    loom_serve_text_service_t* value = nullptr;
    IREE_ASSERT_OK(loom_serve_text_service_create(model_, &options, nullptr,
                                                  &value, allocator_));
    auto destroy = [](loom_serve_text_service_t* service) {
      loom_serve_text_service_destroy(service, IREE_STATUS_OK);
    };
    std::unique_ptr<loom_serve_text_service_t, decltype(destroy)> service(
        value, destroy);
    const auto interface = loom_serve_text_service_interface(value);
    Observe(interface);
    auto quiescent = [&] {
      return !interface.is_pending(interface.self) &&
             std::string(interface.snapshot(interface.self).phase) == "idle";
    };
    StalledClient stalled;
    ASSERT_NO_FATAL_FAILURE(
        stalled.Send(server_, Request(Message("user", "abxa"), 25)));
    loom_serve_http_connection_t* connection = nullptr;
    ASSERT_NO_FATAL_FAILURE(Accept(interface, &connection));
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, quiescent));
    const int head = Send(Request(Message("user", std::string(32, 'a')), 33));
    loom_serve_http_connection_t* head_connection = nullptr;
    ASSERT_NO_FATAL_FAILURE(Accept(interface, &head_connection));
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, quiescent));
    const int tail = Send(Request(Message("user", "a"), 4));
    ASSERT_NO_FATAL_FAILURE(Accept(interface, &connection));
    // Acceptance owns the tail's CPU preparation until the next owner turn,
    // even if its task has already published completion. Removing the head
    // cannot move the borrowed entry before that release is consumed.
    ASSERT_TRUE(interface.is_pending(interface.self));
    EXPECT_EQ(interface.snapshot(interface.self).queued_requests, 2u);
    if (action == OverlapAction::kDisconnect) {
      const linger reset = {1, 0};
      ASSERT_EQ(setsockopt(head, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
                0);
      ASSERT_EQ(close(head), 0);
      iree_notification_await(
          loom_serve_http_server_notification(server_),
          [](void* self) {
            return loom_serve_http_connection_failed(
                static_cast<loom_serve_http_connection_t*>(self));
          },
          head_connection, iree_infinite_timeout());
      ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, [&] {
        return interface.telemetry.counters->completed == 1 && quiescent();
      }));
      EXPECT_EQ(Content(Read(tail)), Oracle::Complete("a", 4));
    }
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
        interface,
        [&] {
          return !interface.is_pending(interface.self) &&
                 !interface.is_active(interface.self) &&
                 !interface.snapshot(interface.self).queued_requests;
        },
        LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING));
    if (action == OverlapAction::kStop) {
      EXPECT_TRUE(Read(head).empty());
      EXPECT_TRUE(Read(tail).empty());
    }
    EXPECT_EQ(interface.telemetry.counters->completed,
              action == OverlapAction::kDisconnect ? 1u : 0u);
    EXPECT_EQ(interface.telemetry.counters->cancelled,
              action == OverlapAction::kDisconnect ? 2u : 3u);
    EXPECT_EQ(interface.telemetry.counters->failed, 0u);
    EXPECT_EQ(loom_serve_text_model_pool_usage(model_).pending, 0u);
    ASSERT_NO_FATAL_FAILURE(stalled.Close());
    interface.set_observer(interface.self, {});
  }

 private:
  // Two execution rows share one source command across token buckets.
  const loom_serve_packing_shape_t admission_shapes_[3] = {
      {4, 1}, {16, 2}, {32, 2}};
};

TEST_P(AdmissionServiceTest, FifoHoldsFittingTailUntilOldestCanProgress) {
  CheckBackfill(0, OverlapAction::kComplete);
}

TEST_P(AdmissionServiceTest, OneOvertakeThenOldestMakesProgress) {
  CheckBackfill(1, OverlapAction::kComplete);
}

TEST_P(AdmissionServiceTest, CohortOvertakesThenOldestMakesProgress) {
  CheckBackfill(2, OverlapAction::kComplete);
}

TEST_P(AdmissionServiceTest, CancellingOldestReleasesExhaustedBudget) {
  CheckBackfill(2, OverlapAction::kDisconnect);
}

TEST_P(AdmissionServiceTest, StopRetiresParkedInputsAndActiveReservations) {
  CheckBackfill(2, OverlapAction::kStop);
}

TEST_P(AdmissionServiceTest, RowReuseRepreparesEvictedHistoryExactlyOnce) {
  CheckBackfill(2, OverlapAction::kComplete, History::kRetained);
}

TEST_P(AdmissionServiceTest,
       PinnedHistorySurvivesOvertakesWithoutRepreparation) {
  CheckBackfill(2, OverlapAction::kComplete, History::kPinned);
}

TEST_P(AdmissionServiceTest,
       DetachedHistorySurvivesOvertakesWithoutRepreparation) {
  CheckBackfill(2, OverlapAction::kComplete, History::kCached);
}

TEST_P(AdmissionServiceTest, CancellingOldestPreservesBorrowedTailPreparation) {
  CheckBorrowedProbeCancellation(OverlapAction::kDisconnect);
}

TEST_P(AdmissionServiceTest, StopRetiresBorrowedTailPreparation) {
  CheckBorrowedProbeCancellation(OverlapAction::kStop);
}

INSTANTIATE_TEST_SUITE_P(Workers, AdmissionServiceTest,
                         ::testing::Values(1, 4));

}  // namespace
