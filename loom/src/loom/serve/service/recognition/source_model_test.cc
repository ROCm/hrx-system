// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/storage/memory.h"
#include "loom/serve/service/recognition/model.h"
#include "loom/serve/testing/service_application.h"

IREE_FLAG(string, model_source, "", "Recognition fixture bootstrap source.");
IREE_FLAG(bool, test_elastic, false,
          "Use elastic backing for recognition rows.");

namespace {

constexpr uint64_t kRecognitionSlabSize = 4096;

class SourceRecognitionModelTest : public ::testing::Test {
 protected:
  struct loom_serve_test_recognition_row_t {
    // Input owned by the serving caller, separate from model request state.
    loom_serve_recognition_input_t input = {};
    // Admitted source state, independently resident until release.
    void* invocation = nullptr;
    // Cold caller-owned storage, reused after each consumed event packet.
    loom_serve_recognition_events_t* events = nullptr;
  };

  void SetUp() override {
    auto options = loom_serve_test_application_options_default();
    options.backing = FLAG_test_elastic ? LOOM_SERVE_DEVICE_BACKING_ELASTIC
                                        : LOOM_SERVE_DEVICE_BACKING_FIXED;
    options.slab_size = FLAG_test_elastic ? kRecognitionSlabSize : 0;
    options.memory_limit = FLAG_test_elastic ? 64 * 1024 : 0;
    loom_serve_test_application_t* application = nullptr;
    IREE_ASSERT_OK(loom_serve_test_application_acquire(&options, allocator_,
                                                       &application));
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
    generator_ = loom_serve_recognition_model_generator(model_);
    EXPECT_EQ(generator_.format.sample_rate, 16000u);
    EXPECT_EQ(generator_.format.channels, 1u);
    EXPECT_EQ(generator_.input_capacity, 7u);
    EXPECT_EQ(generator_.frame_limit, 2048u);
    EXPECT_EQ(generator_.event_bytes, 146u);
    EXPECT_EQ(generator_.event_capacity, 3u);
    EXPECT_EQ(generator_.outputs, LOOM_SERVE_RECOGNITION_OUTPUT_ALL);
    EXPECT_EQ(generator_.request_capacity, rows_.size());
    EXPECT_EQ(generator_.batch_units, rows_.size());
    for (auto& row : rows_) {
      IREE_ASSERT_OK(loom_serve_recognition_input_initialize(
          generator_.input_capacity, generator_.format.channels, allocator_,
          &row.input));
      IREE_ASSERT_OK(loom_serve_recognition_events_allocate(
          generator_.event_capacity, generator_.event_bytes, allocator_,
          &row.events));
    }
  }

  void TearDown() override {
    if (FLAG_test_elastic) {
      EXPECT_GT(deferred_release_count_, 0u);
    }
    IREE_EXPECT_OK(loom_serve_recognition_model_free(model_));
    for (auto& row : rows_) {
      loom_serve_recognition_input_deinitialize(&row.input);
      loom_serve_recognition_events_free(row.events);
    }
  }

  iree_status_t Wait(const loom_serve_execution_frontier_t& frontier) {
    iree_status_t status = iree_ok_status();
    for (size_t i = 0;
         i < IREE_ARRAYSIZE(frontier.semaphores) && iree_status_is_ok(status);
         ++i) {
      status =
          loom_serve_workspace_wait(loom_serve_device_workspace(device_),
                                    frontier.semaphores[i], frontier.values[i]);
    }
    return status;
  }

  void Admit(size_t index, iree_string_view_t prompt = iree_string_view_empty(),
             loom_serve_recognition_output_flags_t outputs =
                 LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT) {
    auto& row = rows_[index];
    const loom_serve_recognition_request_t request = {
        .prompt = prompt,
        .outputs = outputs,
    };
    IREE_ASSERT_OK(
        generator_.prepare(generator_.self, index, &request, &row.invocation));
    loom_serve_residency_admission_t admission;
    loom_serve_execution_frontier_t frontier;
    IREE_ASSERT_OK(generator_.try_admit(generator_.self, row.invocation,
                                        &admission, &frontier));
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
    IREE_ASSERT_OK(Wait(frontier));
    generator_.complete_admit(generator_.self, row.invocation);
  }

  void Release(size_t index) {
    auto& row = rows_[index];
    bool complete = false;
    while (!complete) {
      loom_serve_execution_frontier_t frontier;
      IREE_ASSERT_OK(generator_.release(generator_.self, row.invocation,
                                        &complete, &frontier));
      if (!complete) {
        ++deferred_release_count_;
        IREE_ASSERT_OK(Wait(frontier));
      }
    }
    row.invocation = nullptr;
    loom_serve_recognition_input_deinitialize(&row.input);
    IREE_ASSERT_OK(loom_serve_recognition_input_initialize(
        generator_.input_capacity, generator_.format.channels, allocator_,
        &row.input));
  }

  void Append(size_t index, const std::vector<float>& samples) {
    std::vector<uint8_t> bytes(samples.size() * sizeof(float));
    for (size_t i = 0; i < samples.size(); ++i) {
      iree_unaligned_store_le_f32(bytes.data() + i * sizeof(float), samples[i]);
    }
    ASSERT_EQ(loom_serve_recognition_input_append(
                  &rows_[index].input,
                  iree_make_const_byte_span(bytes.data(), bytes.size())),
              samples.size());
  }

  void Offer(size_t index, uint64_t units) {
    auto& row = rows_[index];
    const auto snapshot = loom_serve_recognition_input_peek(&row.input);
    loom_serve_recognition_work_t work;
    IREE_ASSERT_OK(generator_.prepare_work(generator_.self, row.invocation,
                                           &snapshot, &work));
    EXPECT_EQ(work.units, units);
    EXPECT_EQ(work.tag, 0u);
  }

  void Submit(const std::vector<size_t>& members) {
    members_ = members;
    std::vector<void*> invocations;
    for (auto index : members) {
      ASSERT_NO_FATAL_FAILURE(Offer(index, 1));
      invocations.push_back(rows_[index].invocation);
    }
    IREE_ASSERT_OK(generator_.prepare_batch(generator_.self, invocations.size(),
                                            invocations.data()));
    IREE_ASSERT_OK(generator_.submit(generator_.self, &frontier_));
  }

  // Check the entire cohort after every complete call. The source intentionally
  // returns the same mutable event buffer for each member.
  void Collect(const std::vector<char>& text,
               const std::vector<uint64_t>& positions,
               const std::vector<bool>& complete) {
    IREE_ASSERT_OK(Wait(frontier_));
    std::vector<loom_serve_recognition_result_t> results(members_.size());
    for (size_t ordinal = 0; ordinal < members_.size(); ++ordinal) {
      IREE_ASSERT_OK(generator_.collect(generator_.self, ordinal,
                                        rows_[members_[ordinal]].events,
                                        &results[ordinal]));
    }
    for (size_t ordinal = 0; ordinal < members_.size(); ++ordinal) {
      const auto& result = results[ordinal];
      const auto* events = rows_[members_[ordinal]].events;
      ASSERT_EQ(loom_serve_recognition_events_count(events), 1u);
      const auto* event = loom_serve_recognition_events_at(events, 0);
      ASSERT_EQ(event->kind, LOOM_SERVE_RECOGNITION_EVENT_TRANSCRIPT);
      const auto& transcript = event->transcript;
      ASSERT_EQ(transcript.text.size, 1u);
      EXPECT_EQ(transcript.text.data[0], text[ordinal]);
      EXPECT_EQ(transcript.end, positions[ordinal]);
      EXPECT_EQ(result.complete, complete[ordinal]);
      EXPECT_EQ(iree_any_bit_set(transcript.flags,
                                 LOOM_SERVE_RECOGNITION_TRANSCRIPT_FINAL),
                complete[ordinal]);
      auto& input = rows_[members_[ordinal]].input;
      EXPECT_EQ(input.consumed + result.consumed_frames, positions[ordinal]);
      loom_serve_recognition_input_consume(&input, result.consumed_frames);
    }
    generator_.release_batch(generator_.self);
  }

  iree_allocator_t allocator_ = iree_allocator_system();
  // Borrowed process device and finite state pool, enclosing model lifetime.
  loom_serve_device_t* device_ = nullptr;
  loom_serve_recognition_model_t* model_ = nullptr;
  loom_serve_recognition_generator_t generator_ = {};
  // Native backing releases observed before their completion frontier retires.
  size_t deferred_release_count_ = 0;
  std::array<loom_serve_test_recognition_row_t, 4> rows_;
  std::vector<size_t> members_;
  loom_serve_execution_frontier_t frontier_ = {};
};

TEST_F(SourceRecognitionModelTest, ChangingMembershipRetainsInputAndRevisions) {
  ASSERT_NO_FATAL_FAILURE(Admit(0));
  ASSERT_NO_FATAL_FAILURE(Append(0, {1}));
  ASSERT_NO_FATAL_FAILURE(Offer(0, 0));
  ASSERT_NO_FATAL_FAILURE(Append(0, {1, 1}));
  const auto accepted = loom_serve_recognition_input_peek(&rows_[0].input);
  ASSERT_NO_FATAL_FAILURE(Submit({0}));
  // Later input occupies disjoint free bytes while accepted work retains its
  // original snapshot. No credit is returned by submission itself.
  ASSERT_NO_FATAL_FAILURE(Append(0, {1, 1, 1, 1}));
  EXPECT_EQ(loom_serve_recognition_input_credit(&rows_[0].input), 0u);
  EXPECT_EQ(accepted.frame_count, 3u);
  for (size_t i = 0; i < accepted.frame_count; ++i) {
    EXPECT_EQ(iree_unaligned_load_le_f32(accepted.spans[0].data + i * 4), 1.0f);
  }
  ASSERT_NO_FATAL_FAILURE(Collect({'D'}, {3}, {false}));
  for (size_t i = 1; i < rows_.size(); ++i) {
    ASSERT_NO_FATAL_FAILURE(Admit(i));
  }
  ASSERT_NO_FATAL_FAILURE(Append(1, {2, 2, 2, 2, 2, 2}));
  ASSERT_NO_FATAL_FAILURE(Append(2, {3, 3, 3, 3}));
  ASSERT_NO_FATAL_FAILURE(Append(3, {4}));
  rows_[2].input.complete = true;
  rows_[3].input.complete = true;
  ASSERT_NO_FATAL_FAILURE(Submit({3, 1, 0, 2}));
  ASSERT_NO_FATAL_FAILURE(
      Collect({'E', 'G', 'G', 'J'}, {1, 3, 6, 3}, {true, false, false, false}));
  ASSERT_NO_FATAL_FAILURE(Release(3));
  ASSERT_NO_FATAL_FAILURE(Admit(3));
  ASSERT_NO_FATAL_FAILURE(Append(3, {5, 5}));
  rows_[3].input.complete = true;
  rows_[0].input.complete = true;
  ASSERT_NO_FATAL_FAILURE(Submit({2, 0, 3}));
  // Row 1 is ineligible while its consumer is stalled. Other rows can finish;
  // accepted row 2 is cancelled, but still collects at its original ordinal.
  ASSERT_NO_FATAL_FAILURE(Append(1, {2, 2, 2, 2}));
  EXPECT_EQ(loom_serve_recognition_input_credit(&rows_[1].input), 0u);
  EXPECT_EQ(rows_[2].input.consumed, 3u);
  ASSERT_NO_FATAL_FAILURE(
      Collect({'M', 'H', 'K'}, {4, 7, 2}, {true, true, true}));
  ASSERT_NO_FATAL_FAILURE(Release(2));
  ASSERT_NO_FATAL_FAILURE(Release(0));
  ASSERT_NO_FATAL_FAILURE(Release(3));
  const auto wrapped = loom_serve_recognition_input_peek(&rows_[1].input);
  ASSERT_GT(wrapped.spans[0].data_length, 0u);
  ASSERT_GT(wrapped.spans[1].data_length, 0u);
  rows_[1].input.complete = true;
  ASSERT_NO_FATAL_FAILURE(Submit({1}));
  ASSERT_NO_FATAL_FAILURE(Collect({'M'}, {6}, {false}));
  ASSERT_NO_FATAL_FAILURE(Submit({1}));
  ASSERT_NO_FATAL_FAILURE(Collect({'S'}, {9}, {false}));
  ASSERT_NO_FATAL_FAILURE(Submit({1}));
  ASSERT_NO_FATAL_FAILURE(Collect({'U'}, {10}, {true}));
  ASSERT_NO_FATAL_FAILURE(Release(1));
}

TEST_F(SourceRecognitionModelTest, ActivityLeavesUnacknowledgedInputAvailable) {
  ASSERT_NO_FATAL_FAILURE(Admit(0, iree_string_view_empty(),
                                LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY));
  ASSERT_NO_FATAL_FAILURE(Append(0, {1, 2, 3, 4, 5, 6, 7}));
  auto& row = rows_[0];
  for (uint64_t end : {3u, 6u, 9u, 10u}) {
    ASSERT_NO_FATAL_FAILURE(Submit({0}));
    IREE_ASSERT_OK(Wait(frontier_));
    loom_serve_recognition_result_t result;
    IREE_ASSERT_OK(generator_.collect(generator_.self, 0, row.events, &result));
    ASSERT_EQ(loom_serve_recognition_events_count(row.events), 1u);
    const auto* event = loom_serve_recognition_events_at(row.events, 0);
    ASSERT_EQ(event->kind, LOOM_SERVE_RECOGNITION_EVENT_ACTIVITY);
    EXPECT_EQ(event->activity.begin, row.input.consumed);
    EXPECT_EQ(event->activity.end, end);
    EXPECT_EQ(result.consumed_frames, end - row.input.consumed);
    EXPECT_EQ(result.complete, end == 10);
    loom_serve_recognition_input_consume(&row.input, result.consumed_frames);
    generator_.release_batch(generator_.self);

    if (end == 3) {
      const auto remaining = loom_serve_recognition_input_peek(&row.input);
      ASSERT_EQ(remaining.frame_count, 4u);
      ASSERT_EQ(remaining.spans[0].data_length, 4 * sizeof(float));
      for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(iree_unaligned_load_le_f32(remaining.spans[0].data + i * 4),
                  static_cast<float>(i + 4));
      }
      ASSERT_NO_FATAL_FAILURE(Append(0, {8, 9, 10}));
      row.input.complete = true;
    }
  }
  ASSERT_NO_FATAL_FAILURE(Release(0));
}

TEST_F(SourceRecognitionModelTest,
       EmptyFinalInputDrainsWithoutInventingFrames) {
  ASSERT_NO_FATAL_FAILURE(Admit(0));
  ASSERT_NO_FATAL_FAILURE(Offer(0, 0));
  rows_[0].input.complete = true;
  ASSERT_NO_FATAL_FAILURE(Submit({0}));
  ASSERT_NO_FATAL_FAILURE(Collect({'A'}, {0}, {true}));
  ASSERT_NO_FATAL_FAILURE(Release(0));
}

TEST_F(SourceRecognitionModelTest, RepeatedOffersDoNotAdvanceSourceState) {
  ASSERT_NO_FATAL_FAILURE(Admit(0, IREE_SV("speech")));
  ASSERT_NO_FATAL_FAILURE(Append(0, {2, 4, 6}));
  ASSERT_NO_FATAL_FAILURE(Offer(0, 1));
  ASSERT_NO_FATAL_FAILURE(Offer(0, 1));
  ASSERT_NO_FATAL_FAILURE(Submit({0}));
  ASSERT_NO_FATAL_FAILURE(Collect({'M'}, {3}, {false}));
  rows_[0].input.complete = true;
  ASSERT_NO_FATAL_FAILURE(Submit({0}));
  ASSERT_NO_FATAL_FAILURE(Collect({'M'}, {3}, {true}));
  ASSERT_NO_FATAL_FAILURE(Release(0));
}

TEST_F(SourceRecognitionModelTest, RowCapacityDenialPreservesPreparedRequest) {
  auto* pool = loom_serve_device_memory_pool(device_);
  auto* residency = loom_serve_recognition_model_residency(model_);
  loom_serve_memory_statistics_t peer_statistics = {};
  const auto release_peer = [](loom_serve_virtual_buffer_t* buffer) {
    IREE_EXPECT_OK(loom_serve_virtual_buffer_free(buffer));
  };
  std::unique_ptr<loom_serve_virtual_buffer_t, decltype(release_peer)> peer(
      nullptr, release_peer);
  if (pool) {
    // Tiny row roots each occupy one minimum native extent, not a full slab.
    // Leave two extents free; admission needs all three roots together.
    const auto geometry = loom_serve_device_state_geometry(device_);
    const uint64_t peer_bytes =
        loom_serve_memory_pool_limit(pool) - 2 * geometry.minimum_size;
    loom_serve_virtual_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
        pool, loom_serve_device_state_geometry(device_), peer_bytes, 16,
        &peer_statistics, &buffer));
    peer.reset(buffer);
    IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(buffer, 0, peer_bytes));
  }
  auto& row = rows_[0];
  const loom_serve_recognition_request_t request = {
      .outputs = LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT,
  };
  IREE_ASSERT_OK(
      generator_.prepare(generator_.self, 0, &request, &row.invocation));
  loom_serve_residency_admission_t admission;
  loom_serve_execution_frontier_t frontier;
  if (pool) {
    const uint64_t live = loom_serve_memory_pool_live_bytes(pool);
    IREE_ASSERT_OK(generator_.try_admit(generator_.self, row.invocation,
                                        &admission, &frontier));
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
    EXPECT_EQ(loom_serve_memory_pool_live_bytes(pool), live);
    EXPECT_EQ(loom_serve_residency_activity(residency).activations, 0u);
    peer.reset();
    IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  }
  IREE_ASSERT_OK(generator_.try_admit(generator_.self, row.invocation,
                                      &admission, &frontier));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  IREE_ASSERT_OK(Wait(frontier));
  generator_.complete_admit(generator_.self, row.invocation);
  ASSERT_NO_FATAL_FAILURE(Append(0, {1, 1, 1}));
  row.input.complete = true;
  ASSERT_NO_FATAL_FAILURE(Submit({0}));
  ASSERT_NO_FATAL_FAILURE(Collect({'D'}, {3}, {true}));
  ASSERT_NO_FATAL_FAILURE(Release(0));
}

}  // namespace
