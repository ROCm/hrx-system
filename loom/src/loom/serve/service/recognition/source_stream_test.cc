// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <atomic>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "iree/base/threading/notification.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/service/recognition/model.h"
#include "loom/serve/service/recognition/stream.h"
#include "loom/serve/telemetry/record.h"
#include "loom/serve/testing/native_allocator.h"
#include "loom/serve/testing/service_application.h"
#include "loom/serve/testing/worker_gate.h"

IREE_FLAG(string, model_source, "", "Recognition fixture bootstrap source.");
IREE_FLAG(int32_t, test_workers, 1,
          "Workers in the recognition test application.");
IREE_FLAG(bool, test_elastic, false, "Use elastic backing for recognition.");

namespace {

// Observes the real source generator; all preparation, numerical execution,
// feedback, and state release still go through the production model.
class SourceRecognitionStreamTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_notification_initialize(&notification_);
    auto options = loom_serve_test_application_options_default();
    options.worker_count = FLAG_test_workers;
    options.backing = FLAG_test_elastic ? LOOM_SERVE_DEVICE_BACKING_ELASTIC
                                        : LOOM_SERVE_DEVICE_BACKING_FIXED;
    options.slab_size = FLAG_test_elastic ? 2 * 1024 * 1024 : 0;
    loom_serve_test_application_t* application = nullptr;
    IREE_ASSERT_OK(loom_serve_test_application_acquire(&options, allocator_,
                                                       &application));
    host_ = loom_serve_test_application_host(application);
    device_ = loom_serve_test_application_device_at(application, 0);
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
        device_, &model_options, allocator_, &model_));
    actual_ = loom_serve_recognition_model_generator(model_);
    auto observed = actual_;
    observed.self = this;
    observed.prepare = [](void* self, iree_host_size_t row,
                          const loom_serve_recognition_request_t* request,
                          void** out) {
      auto* test = static_cast<SourceRecognitionStreamTest*>(self);
      IREE_RETURN_IF_ERROR(
          test->actual_.prepare(test->actual_.self, row, request, out));
      test->invocations_[row] = *out;
      return iree_ok_status();
    };
    observed.discard = [](void* self, void* invocation) {
      auto* test = static_cast<SourceRecognitionStreamTest*>(self);
      test->invocations_[test->Row(invocation)] = nullptr;
      test->actual_.discard(test->actual_.self, invocation);
    };
    observed.try_admit = [](void* self, void* invocation,
                            loom_serve_residency_admission_t* admission,
                            loom_serve_execution_frontier_t* frontier) {
      auto* test = static_cast<SourceRecognitionStreamTest*>(self);
      return test->actual_.try_admit(test->actual_.self, invocation, admission,
                                     frontier);
    };
    observed.complete_admit = [](void* self, void* invocation) {
      auto* test = static_cast<SourceRecognitionStreamTest*>(self);
      return test->actual_.complete_admit(test->actual_.self, invocation);
    };
    observed.prepare_work =
        [](void* self, void* invocation,
           const loom_serve_recognition_input_block_t* input,
           loom_serve_recognition_work_t* work) {
          auto* test = static_cast<SourceRecognitionStreamTest*>(self);
          ++test->offers_[test->Row(invocation)];
          return test->actual_.prepare_work(test->actual_.self, invocation,
                                            input, work);
        };
    observed.prepare_batch = [](void* self, iree_host_size_t count,
                                void* const* invocations) {
      auto* test = static_cast<SourceRecognitionStreamTest*>(self);
      test->members_.clear();
      for (size_t i = 0; i < count; ++i) {
        test->members_.push_back(test->Row(invocations[i]));
      }
      return test->actual_.prepare_batch(test->actual_.self, count,
                                         invocations);
    };
    observed.submit = [](void* self,
                         loom_serve_execution_frontier_t* frontier) {
      auto* test = static_cast<SourceRecognitionStreamTest*>(self);
      IREE_RETURN_IF_ERROR(test->actual_.submit(test->actual_.self, frontier));
      ++test->submitted_;
      test->largest_batch_ =
          std::max(test->largest_batch_, test->members_.size());
      if (test->failed_readiness_ && !test->fail_release_readiness_) {
        frontier->semaphores[0] = test->failed_readiness_;
        frontier->values[0] = 1;
      }
      return iree_ok_status();
    };
    observed.collect = [](void* self, iree_host_size_t ordinal,
                          loom_serve_recognition_events_t* events,
                          loom_serve_recognition_result_t* out) {
      auto* test = static_cast<SourceRecognitionStreamTest*>(self);
      ++test->collected_[test->members_[ordinal]];
      return test->actual_.collect(test->actual_.self, ordinal, events, out);
    };
    observed.release_batch = [](void* self) {
      auto* test = static_cast<SourceRecognitionStreamTest*>(self);
      test->actual_.release_batch(test->actual_.self);
    };
    observed.release = [](void* self, void* invocation, bool* out_complete,
                          loom_serve_execution_frontier_t* frontier) {
      auto* test = static_cast<SourceRecognitionStreamTest*>(self);
      const auto row = test->Row(invocation);
      IREE_RETURN_IF_ERROR(test->actual_.release(test->actual_.self, invocation,
                                                 out_complete, frontier));
      if (*out_complete) {
        test->invocations_[row] = nullptr;
        ++test->released_[row];
      } else {
        ++test->pending_releases_;
        if (test->fail_release_readiness_) {
          frontier->semaphores[0] = test->failed_readiness_;
          frontier->values[0] = 1;
        }
      }
      return iree_ok_status();
    };
    observed.weight_activity = nullptr;
    const loom_serve_recognition_stream_service_options_t service_options = {
        .request_limit = 1024,
        .document_limit = 4096,
    };
    IREE_ASSERT_OK(loom_serve_recognition_stream_service_allocate(
        host_, observed, &service_options, allocator_, &service_));
    loom_serve_recognition_stream_service_set_observer(
        service_, {[](void* self) {
                     iree_notification_post(
                         static_cast<iree_notification_t*>(self),
                         IREE_ALL_WAITERS);
                   },
                   &notification_});
  }

  void TearDown() override {
    RetireModel();
    for (auto* stream : streams_) {
      loom_serve_recognition_stream_close(stream);
    }
    iree_hal_semaphore_release(failed_readiness_);
    iree_notification_deinitialize(&notification_);
  }

  void RetireModel() {
    if (service_) {
      Settle(true);
      loom_serve_recognition_stream_service_set_observer(service_, {});
      loom_serve_recognition_stream_service_free(service_);
      service_ = nullptr;
    }
    IREE_EXPECT_OK(loom_serve_recognition_model_free(model_));
    model_ = nullptr;
  }

  size_t Row(void* invocation) const {
    for (size_t i = 0; i < invocations_.size(); ++i) {
      if (invocations_[i] == invocation) {
        return i;
      }
    }
    ADD_FAILURE() << "Source callback lost resident request identity";
    return 0;
  }

  template <typename Predicate>
  void AdvanceUntil(Predicate ready, bool stopping = false) {
    while (!ready()) {
      const auto ticket = iree_notification_prepare_wait(&notification_);
      bool progress = false;
      auto status = loom_serve_recognition_stream_service_advance(
          service_, stopping, false, &progress);
      if (ready() || progress || !iree_status_is_ok(status)) {
        iree_notification_cancel_wait(&notification_);
      } else {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
      IREE_ASSERT_OK(status);
    }
  }

  void Settle(bool stopping = false) {
    while (true) {
      const auto ticket = iree_notification_prepare_wait(&notification_);
      bool progress = false;
      auto status = loom_serve_recognition_stream_service_advance(
          service_, stopping, false, &progress);
      const bool idle =
          !progress &&
          !loom_serve_recognition_stream_service_is_pending(service_);
      if (idle || progress || !iree_status_is_ok(status)) {
        iree_notification_cancel_wait(&notification_);
      } else {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
      IREE_ASSERT_OK(status);
      if (idle) {
        break;
      }
    }
  }

  iree_status_t AdvanceToFailure() {
    iree_status_t status = iree_ok_status();
    while (iree_status_is_ok(status)) {
      const auto ticket = iree_notification_prepare_wait(&notification_);
      bool progress = false;
      status = loom_serve_recognition_stream_service_advance(service_, false,
                                                             false, &progress);
      if (progress || !iree_status_is_ok(status)) {
        iree_notification_cancel_wait(&notification_);
      } else {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
    return status;
  }

  void Open(loom_serve_recognition_stream_t** out,
            loom_serve_recognition_output_flags_t outputs =
                LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT) {
    const loom_serve_recognition_request_t request = {.outputs = outputs};
    const loom_serve_recognition_completion_t completion = {
        this, [](void* self, loom_serve_telemetry_outcome_t outcome,
                 const iree_status_t status) {
          auto* test = static_cast<SourceRecognitionStreamTest*>(self);
          ++test->end_count_;
          test->last_end_outcome_ = outcome;
          test->last_end_code_ = iree_status_code(status);
        }};
    IREE_ASSERT_OK(loom_serve_recognition_stream_service_try_open(
        service_, &request, completion, out));
    ASSERT_NE(*out, nullptr);
    streams_.push_back(*out);
  }

  void Close(loom_serve_recognition_stream_t* stream) {
    const auto position = std::find(streams_.begin(), streams_.end(), stream);
    ASSERT_NE(position, streams_.end());
    streams_.erase(position);
    loom_serve_recognition_stream_close(stream);
  }

  void Write(loom_serve_recognition_stream_t* stream,
             const std::vector<float>& samples, size_t expected) {
    std::vector<uint8_t> bytes(samples.size() * sizeof(float));
    for (size_t i = 0; i < samples.size(); ++i) {
      iree_unaligned_store_le_f32(bytes.data() + i * sizeof(float), samples[i]);
    }
    iree_host_size_t accepted = 0;
    IREE_ASSERT_OK(loom_serve_recognition_stream_write(
        stream, iree_make_const_byte_span(bytes.data(), bytes.size()),
        &accepted));
    EXPECT_EQ(accepted, expected);
  }

  static std::string Text(
      const loom_serve_recognition_transcript_t* transcript) {
    return std::string(transcript->text.data, transcript->text.size);
  }

  static std::string Text(const loom_serve_recognition_event_t* event) {
    EXPECT_EQ(event->kind, LOOM_SERVE_RECOGNITION_EVENT_TRANSCRIPT);
    return Text(&event->transcript);
  }

  bool Consume(loom_serve_recognition_stream_t* stream) {
    if (loom_serve_recognition_stream_peek(stream)) {
      loom_serve_recognition_stream_consume(stream);
    }
    return loom_serve_recognition_stream_is_complete(stream);
  }

  void CheckFinal(loom_serve_recognition_stream_t* stream,
                  const std::string& expected) {
    IREE_ASSERT_OK(loom_serve_recognition_stream_status(stream));
    const auto* document = loom_serve_recognition_stream_document(stream);
    ASSERT_NE(document, nullptr);
    ASSERT_EQ(document->count, 1u);
    EXPECT_EQ(Text(loom_serve_recognition_document_at(document, 0)), expected);
  }

  // Host allocations, shared application, and selected physical device.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Borrowed process host with the production executor/proactor.
  loom_serve_host_t* host_ = nullptr;
  // Borrowed physical device, also used for independent failure readiness.
  loom_serve_device_t* device_ = nullptr;
  // Real source owner, destroyed only after service callbacks retire.
  loom_serve_recognition_model_t* model_ = nullptr;
  // Unmodified source callbacks behind the observation adapter.
  loom_serve_recognition_generator_t actual_ = {};
  // Actual typed service under test.
  loom_serve_recognition_stream_service_t* service_ = nullptr;
  // Caller-owned results, closed independently of service/model teardown.
  std::vector<loom_serve_recognition_stream_t*> streams_;
  // Owner completion notifications, independent of retained caller handles.
  size_t end_count_ = 0;
  // Most recent producer-established outcome supplied to completion.
  loom_serve_telemetry_outcome_t last_end_outcome_ =
      LOOM_SERVE_TELEMETRY_COMPLETED;
  // Borrowed completion status observed without retaining the status payload.
  iree_status_code_t last_end_code_ = IREE_STATUS_OK;
  // Wake-driven test owner; no sleeps or deadline-based readiness guesses.
  iree_notification_t notification_;
  // Source identities, accessed only on serialized source calls.
  std::array<void*, 4> invocations_ = {};
  // Per-row CPU work offers, observable independently of callback timing.
  std::array<std::atomic<size_t>, 4> offers_ = {};
  // Original ordinal collections, including caller-cancelled requests.
  std::array<std::atomic<size_t>, 4> collected_ = {};
  // Source release calls on the owner, excluding terminal model teardown.
  std::array<size_t, 4> released_ = {};
  // Incomplete releases retaining source identities through physical trim.
  size_t pending_releases_ = 0;
  // CPU-selected rows, handed to the owner at batch preparation publication.
  std::vector<size_t> members_;
  // Accepted numerical cohorts, observed on the owner at submission.
  size_t submitted_ = 0;
  // Largest actual numerical cohort accepted by the source model.
  size_t largest_batch_ = 0;
  // Optional independent failed wait; it never poisons the actual model queue.
  iree_hal_semaphore_t* failed_readiness_ = nullptr;
  // Selects backing-release readiness instead of numerical batch readiness.
  bool fail_release_readiness_ = false;
};

TEST_F(SourceRecognitionStreamTest,
       StaggeredCohortsStalledReaderAcceptedCancellationAndReuse) {
  std::array<loom_serve_recognition_stream_t*, 4> streams = {};
  ASSERT_NO_FATAL_FAILURE(Open(&streams[0]));
  ASSERT_NO_FATAL_FAILURE(Write(streams[0], {1, 2, 3, 4, 5, 6, 7}, 7));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return loom_serve_recognition_stream_peek(streams[0]); }));
  EXPECT_EQ(Text(loom_serve_recognition_stream_peek(streams[0])), "G");
  ASSERT_NO_FATAL_FAILURE(Write(streams[0], {8, 9, 10}, 3));
  loom_serve_recognition_stream_finish(streams[0]);
  ASSERT_NO_FATAL_FAILURE(Open(&streams[1]));
  ASSERT_NO_FATAL_FAILURE(Write(streams[1], {2, 2}, 2));
  ASSERT_NO_FATAL_FAILURE(Settle());
  ASSERT_NO_FATAL_FAILURE(Open(&streams[2]));
  ASSERT_NO_FATAL_FAILURE(Write(streams[2], {9}, 1));
  ASSERT_NO_FATAL_FAILURE(Settle());
  ASSERT_NO_FATAL_FAILURE(Open(&streams[3]));
  ASSERT_NO_FATAL_FAILURE(Settle());
  EXPECT_EQ(submitted_, 1u);

  ASSERT_NO_FATAL_FAILURE(Write(streams[1], {2, 2}, 2));
  ASSERT_NO_FATAL_FAILURE(Write(streams[2], {1, 1, 1, 1, 1}, 5));
  ASSERT_NO_FATAL_FAILURE(Write(streams[3], {4, 5}, 2));
  for (size_t i = 1; i < streams.size(); ++i) {
    loom_serve_recognition_stream_finish(streams[i]);
  }
  loom_serve_recognition_stream_consume(streams[0]);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return submitted_ == 2; }));
  EXPECT_EQ(largest_batch_, 4u);
  ASSERT_NO_FATAL_FAILURE(Close(streams[2]));
  streams[2] = nullptr;
  const loom_serve_recognition_request_t request = {
      .outputs = LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT,
  };
  loom_serve_recognition_stream_t* replacement = nullptr;
  IREE_ASSERT_OK(loom_serve_recognition_stream_service_try_open(
      service_, &request, {}, &replacement));
  EXPECT_EQ(replacement, nullptr);
  EXPECT_EQ(released_[2], 0u);

  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    const bool first = Consume(streams[1]);
    const bool third = Consume(streams[3]);
    return first && third && released_[2] == 1;
  }));
  EXPECT_EQ(collected_[2].load(), 1u);
  ASSERT_NO_FATAL_FAILURE(CheckFinal(streams[1], "I"));
  ASSERT_NO_FATAL_FAILURE(CheckFinal(streams[3], "J"));
  const auto* stalled = loom_serve_recognition_stream_peek(streams[0]);
  ASSERT_NE(stalled, nullptr);
  EXPECT_EQ(stalled->transcript.end, 6u);
  EXPECT_EQ(Text(stalled), "V");
  const auto stalled_offers = offers_[0].load();

  ASSERT_NO_FATAL_FAILURE(Open(&replacement));
  ASSERT_NO_FATAL_FAILURE(Write(replacement, {5}, 1));
  loom_serve_recognition_stream_finish(replacement);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return Consume(replacement); }));
  ASSERT_NO_FATAL_FAILURE(CheckFinal(replacement, "F"));
  EXPECT_EQ(offers_[0].load(), stalled_offers);
  EXPECT_EQ(Text(stalled), "V");
  EXPECT_EQ(stalled->transcript.end, 6u);
  loom_serve_recognition_stream_consume(streams[0]);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return Consume(streams[0]); }));
  ASSERT_NO_FATAL_FAILURE(CheckFinal(streams[0], "D"));
}

TEST_F(SourceRecognitionStreamTest,
       SourceCompletionFailureRetiresOnlyItsOriginalMember) {
  std::array<loom_serve_recognition_stream_t*, 3> streams = {};
  for (auto& stream : streams) {
    ASSERT_NO_FATAL_FAILURE(Open(&stream));
  }
  ASSERT_NO_FATAL_FAILURE(Settle());
  ASSERT_NO_FATAL_FAILURE(Write(streams[0], {-1}, 1));
  ASSERT_NO_FATAL_FAILURE(Write(streams[1], {2, 3}, 2));
  ASSERT_NO_FATAL_FAILURE(Write(streams[2], {4}, 1));
  for (auto* stream : streams) {
    loom_serve_recognition_stream_finish(stream);
  }
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return submitted_ == 1; }));
  EXPECT_EQ(largest_batch_, 3u);
  ASSERT_NO_FATAL_FAILURE(Close(streams[1]));
  EXPECT_EQ(released_[1], 0u);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    return Consume(streams[2]) && released_[0] == 1 && released_[1] == 1;
  }));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_recognition_stream_status(streams[0]));
  EXPECT_EQ(loom_serve_recognition_stream_peek(streams[0]), nullptr);
  EXPECT_EQ(loom_serve_recognition_stream_document(streams[0]), nullptr);
  ASSERT_NO_FATAL_FAILURE(CheckFinal(streams[2], "E"));
  for (size_t i = 0; i < streams.size(); ++i) {
    EXPECT_EQ(collected_[i].load(), 1u);
    EXPECT_EQ(released_[i], 1u);
  }
  loom_serve_recognition_stream_t* replacement = nullptr;
  ASSERT_NO_FATAL_FAILURE(Open(&replacement));
  ASSERT_NO_FATAL_FAILURE(Write(replacement, {5}, 1));
  loom_serve_recognition_stream_finish(replacement);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return Consume(replacement); }));
  ASSERT_NO_FATAL_FAILURE(CheckFinal(replacement, "F"));
  EXPECT_EQ(released_[0], 2u);
}

TEST_F(SourceRecognitionStreamTest, InputStarvationWaitsForNewFramesOrEof) {
  loom_serve_recognition_stream_t* stream = nullptr;
  ASSERT_NO_FATAL_FAILURE(Open(&stream));
  ASSERT_NO_FATAL_FAILURE(Write(stream, {1}, 1));
  ASSERT_NO_FATAL_FAILURE(Settle());
  EXPECT_EQ(submitted_, 0u);
  const auto initial_offers = offers_[0].load();
  for (size_t i = 0; i < 4; ++i) {
    bool progress = false;
    IREE_ASSERT_OK(loom_serve_recognition_stream_service_advance(
        service_, false, false, &progress));
    EXPECT_FALSE(progress);
    EXPECT_FALSE(loom_serve_recognition_stream_service_is_pending(service_));
  }
  EXPECT_EQ(offers_[0].load(), initial_offers);
  ASSERT_NO_FATAL_FAILURE(Write(stream, {1}, 1));
  ASSERT_NO_FATAL_FAILURE(Settle());
  EXPECT_EQ(offers_[0].load(), initial_offers + 1);
  EXPECT_EQ(submitted_, 0u);
  loom_serve_recognition_stream_finish(stream);
  loom_serve_recognition_stream_finish(stream);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return Consume(stream); }));
  ASSERT_NO_FATAL_FAILURE(CheckFinal(stream, "C"));
  EXPECT_EQ(submitted_, 1u);
  EXPECT_EQ(loom_serve_recognition_stream_credit(stream), 0u);
}

TEST_F(SourceRecognitionStreamTest, InputValidationAndIndependentCredit) {
  loom_serve_recognition_stream_t* stream = nullptr;
  ASSERT_NO_FATAL_FAILURE(Open(&stream));
  ASSERT_NO_FATAL_FAILURE(Write(stream, {1, 1, 1, 1, 1, 1, 1, 1}, 7));
  ASSERT_NO_FATAL_FAILURE(Write(stream, {1}, 0));
  EXPECT_EQ(loom_serve_recognition_stream_credit(stream), 0u);
  std::array<uint8_t, 4> invalid = {};
  iree_host_size_t accepted = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_serve_recognition_stream_write(
          stream, iree_make_const_byte_span(invalid.data(), 3), &accepted));
  iree_unaligned_store_le_f32(invalid.data(),
                              std::numeric_limits<float>::quiet_NaN());
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_recognition_stream_write(
          stream, iree_make_const_byte_span(invalid.data(), invalid.size()),
          &accepted));
  EXPECT_EQ(accepted, 0u);
  loom_serve_recognition_stream_finish(stream);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_recognition_stream_write(
                            stream, iree_const_byte_span_empty(), &accepted));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return Consume(stream); }));
  ASSERT_NO_FATAL_FAILURE(CheckFinal(stream, "H"));
}

TEST_F(SourceRecognitionStreamTest, EmptyEofCompletesOnceAndCloseReusesSlot) {
  loom_serve_recognition_stream_t* stream = nullptr;
  ASSERT_NO_FATAL_FAILURE(Open(&stream));
  loom_serve_recognition_stream_finish(stream);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return Consume(stream); }));
  ASSERT_NO_FATAL_FAILURE(CheckFinal(stream, "A"));
  EXPECT_EQ(submitted_, 1u);
  ASSERT_NO_FATAL_FAILURE(Close(stream));
  ASSERT_NO_FATAL_FAILURE(Settle());
  ASSERT_NO_FATAL_FAILURE(Open(&stream));
  ASSERT_NO_FATAL_FAILURE(Write(stream, {3}, 1));
  loom_serve_recognition_stream_finish(stream);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return Consume(stream); }));
  ASSERT_NO_FATAL_FAILURE(CheckFinal(stream, "D"));
  EXPECT_EQ(submitted_, 2u);
}

TEST_F(SourceRecognitionStreamTest, ShutdownCollectsAcceptedOriginalMembers) {
  loom_serve_recognition_stream_t* stream = nullptr;
  ASSERT_NO_FATAL_FAILURE(Open(&stream));
  ASSERT_NO_FATAL_FAILURE(Write(stream, {1, 2, 3}, 3));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return submitted_ == 1; }));
  ASSERT_NO_FATAL_FAILURE(Settle(true));
  EXPECT_EQ(collected_[0].load(), 1u);
  EXPECT_EQ(released_[0], 1u);
  EXPECT_EQ(submitted_, 1u);
  EXPECT_EQ(end_count_, 1u);
  EXPECT_EQ(last_end_outcome_, LOOM_SERVE_TELEMETRY_CANCELLED);
  EXPECT_EQ(last_end_code_, IREE_STATUS_CANCELLED);
  ASSERT_NO_FATAL_FAILURE(RetireModel());
  EXPECT_TRUE(loom_serve_recognition_stream_is_complete(stream));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        loom_serve_recognition_stream_status(stream));
  EXPECT_EQ(loom_serve_recognition_stream_peek(stream), nullptr);
  EXPECT_EQ(loom_serve_recognition_stream_document(stream), nullptr);
  EXPECT_EQ(loom_serve_recognition_stream_credit(stream), 0u);
  iree_host_size_t accepted = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_recognition_stream_write(
                            stream, iree_const_byte_span_empty(), &accepted));
}

TEST_F(SourceRecognitionStreamTest,
       FinalRevisionSurvivesRowReuseAndModelDestruction) {
  loom_serve_recognition_stream_t* first = nullptr;
  ASSERT_NO_FATAL_FAILURE(Open(&first));
  ASSERT_NO_FATAL_FAILURE(Write(first, {1, 2}, 2));
  loom_serve_recognition_stream_finish(first);
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return loom_serve_recognition_stream_peek(first); }));
  const auto* retained = loom_serve_recognition_stream_peek(first);
  ASSERT_NE(retained, nullptr);
  EXPECT_EQ(Text(retained), "D");
  EXPECT_TRUE(retained->transcript.flags &
              LOOM_SERVE_RECOGNITION_TRANSCRIPT_FINAL);
  ASSERT_NO_FATAL_FAILURE(Settle());
  EXPECT_EQ(released_[0], 1u);
  EXPECT_EQ(end_count_, 1u);
  EXPECT_EQ(last_end_outcome_, LOOM_SERVE_TELEMETRY_COMPLETED);
  EXPECT_EQ(last_end_code_, IREE_STATUS_OK);
  EXPECT_FALSE(loom_serve_recognition_stream_is_complete(first));

  loom_serve_recognition_stream_t* replacement = nullptr;
  ASSERT_NO_FATAL_FAILURE(Open(&replacement));
  ASSERT_NO_FATAL_FAILURE(Write(replacement, {4, 5}, 2));
  loom_serve_recognition_stream_finish(replacement);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return Consume(replacement); }));
  ASSERT_NO_FATAL_FAILURE(CheckFinal(replacement, "J"));
  EXPECT_EQ(released_[0], 2u);
  ASSERT_NO_FATAL_FAILURE(RetireModel());

  EXPECT_EQ(loom_serve_recognition_stream_peek(first), retained);
  EXPECT_EQ(Text(retained), "D");
  EXPECT_EQ(retained->transcript.end, 2u);
  EXPECT_EQ(loom_serve_recognition_stream_format(first).sample_rate, 16000u);
  EXPECT_EQ(loom_serve_recognition_stream_format(first).channels, 1u);
  EXPECT_EQ(loom_serve_recognition_stream_credit(first), 0u);
  loom_serve_recognition_stream_consume(first);
  EXPECT_TRUE(loom_serve_recognition_stream_is_complete(first));
  ASSERT_NO_FATAL_FAILURE(CheckFinal(first, "D"));
  ASSERT_NO_FATAL_FAILURE(CheckFinal(replacement, "J"));
}

TEST_F(SourceRecognitionStreamTest, ClosedCallerRetainsAcceptedInputBorrow) {
  loom_serve_recognition_stream_t* stream = nullptr;
  ASSERT_NO_FATAL_FAILURE(Open(&stream));
  ASSERT_NO_FATAL_FAILURE(Settle());
  const auto offers = offers_[0].load();
  std::vector<std::unique_ptr<loom_serve::testing::WorkerGate>> workers;
  for (int i = 0; i < FLAG_test_workers; ++i) {
    workers.push_back(std::make_unique<loom_serve::testing::WorkerGate>(
        loom_serve_host_executor(host_)));
  }
  ASSERT_NO_FATAL_FAILURE(Write(stream, {1, 2, 3}, 3));
  loom_serve_recognition_stream_finish(stream);
  bool progress = false;
  IREE_ASSERT_OK(loom_serve_recognition_stream_service_advance(
      service_, false, false, &progress));
  EXPECT_TRUE(progress);
  EXPECT_TRUE(loom_serve_recognition_stream_service_is_pending(service_));
  ASSERT_NO_FATAL_FAILURE(Close(stream));
  IREE_ASSERT_OK(loom_serve_recognition_stream_service_advance(
      service_, true, false, &progress));
  EXPECT_FALSE(progress);
  EXPECT_TRUE(loom_serve_recognition_stream_service_is_pending(service_));
  EXPECT_EQ(offers_[0].load(), offers);
  EXPECT_EQ(released_[0], 0u);
  EXPECT_EQ(end_count_, 0u);
  workers.clear();
  ASSERT_NO_FATAL_FAILURE(Settle(true));
  EXPECT_EQ(offers_[0].load(), offers + 1);
  EXPECT_EQ(submitted_, 0u);
  EXPECT_EQ(released_[0], 1u);
  EXPECT_EQ(end_count_, 1u);
  EXPECT_EQ(last_end_outcome_, LOOM_SERVE_TELEMETRY_CANCELLED);
  EXPECT_EQ(last_end_code_, IREE_STATUS_CANCELLED);
}

TEST_F(SourceRecognitionStreamTest,
       ActivityTracksIndependentSpeechAndSilenceTransitions) {
  loom_serve_recognition_stream_t* first = nullptr;
  loom_serve_recognition_stream_t* second = nullptr;
  ASSERT_NO_FATAL_FAILURE(Open(&first, LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY));
  ASSERT_NO_FATAL_FAILURE(
      Open(&second, LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY));
  EXPECT_EQ(loom_serve_recognition_stream_input_capacity(first), 7u);
  const float samples[] = {0.25f, 0.0f, 0.5f, 0.0f};
  for (size_t i = 0; i < IREE_ARRAYSIZE(samples); ++i) {
    const float first_sample = samples[i];
    const float second_sample = samples[IREE_ARRAYSIZE(samples) - 1 - i];
    ASSERT_NO_FATAL_FAILURE(
        Write(first, {first_sample, first_sample, first_sample}, 3));
    EXPECT_EQ(loom_serve_recognition_stream_credit(first), 4u);
    EXPECT_EQ(loom_serve_recognition_stream_input_capacity(first), 7u);
    ASSERT_NO_FATAL_FAILURE(
        Write(second, {second_sample, second_sample, second_sample}, 3));
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
      return loom_serve_recognition_stream_peek(first) &&
             loom_serve_recognition_stream_peek(second);
    }));
    const loom_serve_recognition_stream_t* streams[] = {first, second};
    const float expected[] = {first_sample ? 0.75f : 0.0f,
                              second_sample ? 0.75f : 0.0f};
    for (size_t row = 0; row < IREE_ARRAYSIZE(streams); ++row) {
      const auto* event = loom_serve_recognition_stream_peek(streams[row]);
      ASSERT_EQ(event->kind, LOOM_SERVE_RECOGNITION_EVENT_ACTIVITY);
      EXPECT_EQ(event->activity.begin, i * 3);
      EXPECT_EQ(event->activity.end, (i + 1) * 3);
      EXPECT_EQ(event->activity.speech_probability, expected[row]);
    }
    loom_serve_recognition_stream_consume(first);
    loom_serve_recognition_stream_consume(second);
  }
  loom_serve_recognition_stream_finish(first);
  loom_serve_recognition_stream_finish(second);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    return loom_serve_recognition_stream_is_complete(first) &&
           loom_serve_recognition_stream_is_complete(second);
  }));
  EXPECT_EQ(loom_serve_recognition_stream_peek(first), nullptr);
  EXPECT_EQ(loom_serve_recognition_stream_peek(second), nullptr);
  IREE_ASSERT_OK(loom_serve_recognition_stream_status(first));
  IREE_ASSERT_OK(loom_serve_recognition_stream_status(second));
}

TEST_F(SourceRecognitionStreamTest,
       HeldPacketRetainsEachRevisionWhilePeerRuns) {
  loom_serve_recognition_stream_t* stream = nullptr;
  ASSERT_NO_FATAL_FAILURE(Open(&stream, LOOM_SERVE_RECOGNITION_OUTPUT_ALL));
  ASSERT_NO_FATAL_FAILURE(Write(stream, {1, 1, 1, 1, 1, 1}, 6));
  loom_serve_recognition_stream_finish(stream);
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return loom_serve_recognition_stream_peek(stream); }));
  const auto* activity = loom_serve_recognition_stream_peek(stream);
  ASSERT_EQ(activity->kind, LOOM_SERVE_RECOGNITION_EVENT_ACTIVITY);
  EXPECT_EQ(activity->activity.begin, 0u);
  EXPECT_EQ(activity->activity.end, 3u);
  EXPECT_EQ(activity->activity.speech_probability, 0.75f);
  loom_serve_recognition_stream_consume(stream);
  const auto* provisional = loom_serve_recognition_stream_peek(stream);
  ASSERT_NE(provisional, nullptr);
  EXPECT_EQ(Text(provisional), "d");
  EXPECT_EQ(provisional->transcript.revision, 1u);
  const auto held_offers = offers_[0].load();

  loom_serve_recognition_stream_t* peer = nullptr;
  ASSERT_NO_FATAL_FAILURE(Open(&peer, LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY));
  ASSERT_NO_FATAL_FAILURE(Write(peer, {5, 6}, 2));
  loom_serve_recognition_stream_finish(peer);
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return loom_serve_recognition_stream_peek(peer); }));
  const auto* peer_event = loom_serve_recognition_stream_peek(peer);
  ASSERT_EQ(peer_event->kind, LOOM_SERVE_RECOGNITION_EVENT_ACTIVITY);
  EXPECT_EQ(peer_event->activity.end, 2u);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return Consume(peer); }));
  IREE_ASSERT_OK(loom_serve_recognition_stream_status(peer));
  EXPECT_EQ(loom_serve_recognition_stream_document(peer), nullptr);
  EXPECT_EQ(offers_[0].load(), held_offers);
  EXPECT_EQ(loom_serve_recognition_stream_peek(stream), provisional);
  EXPECT_EQ(Text(provisional), "d");
  loom_serve_recognition_stream_consume(stream);
  const auto* replacement = loom_serve_recognition_stream_peek(stream);
  ASSERT_NE(replacement, nullptr);
  EXPECT_EQ(Text(replacement), "D");
  EXPECT_EQ(replacement->transcript.revision, 2u);
  ASSERT_NO_FATAL_FAILURE(Settle());
  EXPECT_EQ(offers_[0].load(), held_offers);
  EXPECT_EQ(submitted_, 2u);
  loom_serve_recognition_stream_consume(stream);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return Consume(stream); }));
  EXPECT_EQ(submitted_, 3u);
  ASSERT_NO_FATAL_FAILURE(CheckFinal(stream, "G"));
}

TEST_F(SourceRecognitionStreamTest,
       FinalPacketSurvivesSlotReuseAndModelDestruction) {
  loom_serve_recognition_stream_t* stream = nullptr;
  ASSERT_NO_FATAL_FAILURE(Open(&stream, LOOM_SERVE_RECOGNITION_OUTPUT_ALL));
  ASSERT_NO_FATAL_FAILURE(Write(stream, {1, 2}, 2));
  loom_serve_recognition_stream_finish(stream);
  ASSERT_NO_FATAL_FAILURE(Settle());
  const auto* activity = loom_serve_recognition_stream_peek(stream);
  ASSERT_NE(activity, nullptr);
  ASSERT_EQ(activity->kind, LOOM_SERVE_RECOGNITION_EVENT_ACTIVITY);
  EXPECT_EQ(released_[0], 1u);
  EXPECT_FALSE(loom_serve_recognition_stream_is_complete(stream));

  loom_serve_recognition_stream_t* peer = nullptr;
  ASSERT_NO_FATAL_FAILURE(Open(&peer, LOOM_SERVE_RECOGNITION_OUTPUT_ALL));
  ASSERT_NO_FATAL_FAILURE(Write(peer, {4, 5}, 2));
  loom_serve_recognition_stream_finish(peer);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return Consume(peer); }));
  ASSERT_NO_FATAL_FAILURE(CheckFinal(peer, "J"));
  EXPECT_EQ(released_[0], 2u);
  ASSERT_NO_FATAL_FAILURE(RetireModel());

  EXPECT_EQ(loom_serve_recognition_stream_peek(stream), activity);
  EXPECT_EQ(loom_serve_recognition_stream_input_capacity(stream), 7u);
  EXPECT_EQ(activity->activity.end, 2u);
  loom_serve_recognition_stream_consume(stream);
  const auto* provisional = loom_serve_recognition_stream_peek(stream);
  ASSERT_NE(provisional, nullptr);
  EXPECT_EQ(Text(provisional), "d");
  EXPECT_EQ(provisional->transcript.flags, 0u);
  EXPECT_FALSE(loom_serve_recognition_stream_is_complete(stream));
  loom_serve_recognition_stream_consume(stream);
  const auto* final = loom_serve_recognition_stream_peek(stream);
  ASSERT_NE(final, nullptr);
  EXPECT_EQ(Text(final), "D");
  EXPECT_EQ(final->transcript.revision, 2u);
  EXPECT_TRUE(final->transcript.flags &
              LOOM_SERVE_RECOGNITION_TRANSCRIPT_FINAL);
  loom_serve_recognition_stream_consume(stream);
  EXPECT_TRUE(loom_serve_recognition_stream_is_complete(stream));
  ASSERT_NO_FATAL_FAILURE(CheckFinal(stream, "D"));
}

TEST_F(SourceRecognitionStreamTest,
       CancelledCompoundBatchCollectsBeforeRelease) {
  loom_serve_recognition_stream_t* stream = nullptr;
  ASSERT_NO_FATAL_FAILURE(Open(&stream, LOOM_SERVE_RECOGNITION_OUTPUT_ALL));
  ASSERT_NO_FATAL_FAILURE(Write(stream, {1, 2, 3}, 3));
  loom_serve_recognition_stream_finish(stream);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return submitted_ == 1; }));
  ASSERT_NO_FATAL_FAILURE(Close(stream));
  EXPECT_EQ(collected_[0].load(), 0u);
  EXPECT_EQ(released_[0], 0u);
  ASSERT_NO_FATAL_FAILURE(Settle());
  EXPECT_EQ(collected_[0].load(), 1u);
  EXPECT_EQ(released_[0], 1u);
  EXPECT_EQ(end_count_, 1u);
  EXPECT_EQ(last_end_outcome_, LOOM_SERVE_TELEMETRY_CANCELLED);
}

TEST_F(SourceRecognitionStreamTest, ExplicitSelectionAndEmptyActivityEof) {
  for (uint32_t outputs : {0u, 4u}) {
    const loom_serve_recognition_request_t request = {.outputs = outputs};
    loom_serve_recognition_stream_t* rejected = nullptr;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_recognition_stream_service_try_open(
                              service_, &request, {}, &rejected));
    EXPECT_EQ(rejected, nullptr);
  }
  loom_serve_recognition_stream_t* stream = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      Open(&stream, LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY));
  loom_serve_recognition_stream_finish(stream);
  ASSERT_NO_FATAL_FAILURE(Settle());
  EXPECT_EQ(loom_serve_recognition_stream_peek(stream), nullptr);
  EXPECT_TRUE(loom_serve_recognition_stream_is_complete(stream));
  IREE_ASSERT_OK(loom_serve_recognition_stream_status(stream));
  EXPECT_EQ(loom_serve_recognition_stream_document(stream), nullptr);
  EXPECT_EQ(collected_[0].load(), 1u);
  EXPECT_EQ(released_[0], 1u);
}

TEST_F(SourceRecognitionStreamTest, CloseReturnsRowBeforeNativeCapacity) {
  if (!FLAG_test_elastic) {
    GTEST_SKIP() << "Requires asynchronous elastic backing";
  }
  loom_serve_recognition_stream_t* stream = nullptr;
  ASSERT_NO_FATAL_FAILURE(Open(&stream));
  ASSERT_NO_FATAL_FAILURE(Write(stream, {1}, 1));
  ASSERT_NO_FATAL_FAILURE(Settle());

  loom_serve::testing::ScopedNativeHold gate(device_,
                                             IREE_ALLOCATOR_COMMAND_FREE);
  ASSERT_NO_FATAL_FAILURE(Close(stream));
  bool progress = false;
  IREE_ASSERT_OK(loom_serve_recognition_stream_service_advance(
      service_, false, false, &progress));
  EXPECT_TRUE(progress);
  gate.WaitEntered();
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return released_[0] == 1; }));
  EXPECT_GT(
      loom_serve_memory_pool_statistics(loom_serve_device_memory_pool(device_))
          .releasing_bytes,
      0u);

  loom_serve_recognition_stream_t* peer = nullptr;
  ASSERT_NO_FATAL_FAILURE(Open(&peer));
  const auto previous_offers = offers_[0].load();
  ASSERT_NO_FATAL_FAILURE(Write(peer, {2}, 1));
  loom_serve_recognition_stream_finish(peer);
  IREE_ASSERT_OK(loom_serve_recognition_stream_service_advance(
      service_, false, false, &progress));
  EXPECT_EQ(offers_[0].load(), previous_offers);
  EXPECT_EQ(released_[0], 1u);

  gate.Release();
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return Consume(peer); }));
  ASSERT_NO_FATAL_FAILURE(CheckFinal(peer, "C"));
  EXPECT_EQ(released_[0], 2u);
  loom_serve_recognition_stream_t* replacement = nullptr;
  ASSERT_NO_FATAL_FAILURE(Open(&replacement));
  ASSERT_NO_FATAL_FAILURE(Write(replacement, {4}, 1));
  loom_serve_recognition_stream_finish(replacement);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return Consume(replacement); }));
  ASSERT_NO_FATAL_FAILURE(CheckFinal(replacement, "E"));
  EXPECT_EQ(released_[0], 3u);
}

TEST_F(SourceRecognitionStreamTest,
       FailedReadinessLeavesAcceptedModelOwnership) {
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device_), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &failed_readiness_));
  loom_serve_recognition_stream_t* stream = nullptr;
  ASSERT_NO_FATAL_FAILURE(Open(&stream));
  ASSERT_NO_FATAL_FAILURE(Write(stream, {1, 2, 3}, 3));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return submitted_ == 1; }));
  bool progress = false;
  IREE_ASSERT_OK(loom_serve_recognition_stream_service_advance(
      service_, true, false, &progress));
  EXPECT_FALSE(loom_serve_recognition_stream_is_complete(stream));
  iree_hal_semaphore_fail(
      failed_readiness_,
      iree_make_status(IREE_STATUS_ABORTED, "lost readiness"));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, AdvanceToFailure());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        loom_serve_recognition_stream_status(stream));
  EXPECT_FALSE(loom_serve_recognition_stream_service_is_pending(service_));
  EXPECT_EQ(loom_serve_recognition_stream_peek(stream), nullptr);
  EXPECT_EQ(collected_[0].load(), 0u);
  EXPECT_EQ(released_[0], 0u);
  EXPECT_EQ(end_count_, 0u);
  ASSERT_NO_FATAL_FAILURE(Settle());
  EXPECT_TRUE(loom_serve_recognition_stream_is_complete(stream));
  EXPECT_EQ(released_[0], 0u);
  EXPECT_EQ(end_count_, 1u);
  EXPECT_EQ(last_end_outcome_, LOOM_SERVE_TELEMETRY_FAILED);
  EXPECT_EQ(last_end_code_, IREE_STATUS_ABORTED);
  ASSERT_NO_FATAL_FAILURE(RetireModel());
  EXPECT_EQ(end_count_, 1u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        loom_serve_recognition_stream_status(stream));
  // The failed wait does not authorize source release or feedback collection.
}

TEST_F(SourceRecognitionStreamTest,
       FailedFinalReleaseSurvivesModelDestruction) {
  if (!FLAG_test_elastic) {
    GTEST_SKIP() << "Requires asynchronous elastic backing";
  }
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device_), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &failed_readiness_));
  fail_release_readiness_ = true;
  loom_serve_recognition_stream_t* stream = nullptr;
  ASSERT_NO_FATAL_FAILURE(Open(&stream));
  loom_serve_recognition_stream_finish(stream);
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return loom_serve_recognition_stream_peek(stream); }));
  EXPECT_EQ(Text(loom_serve_recognition_stream_peek(stream)), "A");
  EXPECT_EQ(pending_releases_, 1u);
  EXPECT_EQ(released_[0], 0u);
  EXPECT_EQ(end_count_, 0u);
  iree_hal_semaphore_fail(
      failed_readiness_,
      iree_make_status(IREE_STATUS_ABORTED, "lost release readiness"));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, AdvanceToFailure());
  ASSERT_NO_FATAL_FAILURE(RetireModel());
  EXPECT_EQ(end_count_, 1u);
  EXPECT_EQ(last_end_outcome_, LOOM_SERVE_TELEMETRY_FAILED);
  EXPECT_EQ(last_end_code_, IREE_STATUS_ABORTED);
  EXPECT_TRUE(loom_serve_recognition_stream_is_complete(stream));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        loom_serve_recognition_stream_status(stream));
  EXPECT_EQ(loom_serve_recognition_stream_peek(stream), nullptr);
  EXPECT_EQ(loom_serve_recognition_stream_document(stream), nullptr);
}

}  // namespace
