// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/service_test_fixture.h"

namespace {

using loom_serve::testing::text::SourceTextServiceTest;

class SourceEncodingExecutionTest : public SourceTextServiceTest {
 protected:
  loom_serve_text_options_t ModelOptions() const override {
    auto options = SourceTextServiceTest::ModelOptions();
    options.mode = LOOM_SERVE_TEXT_MODE_ENCODE;
    options.tokenizer_path = iree_string_view_empty();
    return options;
  }

  void TearDown() override {
    // Model retirement must not depend on separately retained outputs.
    SourceTextServiceTest::TearDown();
    for (auto* snapshot : snapshots_) {
      loom_serve_text_capture_snapshot_release(snapshot);
    }
    for (auto* capture : captures_) {
      loom_serve_text_capture_release(capture);
    }
  }

  static void Wait(loom_serve_execution_frontier_t ready) {
    for (size_t i = 0; i < IREE_ARRAYSIZE(ready.semaphores); ++i) {
      IREE_ASSERT_OK(iree_hal_semaphore_wait(
          ready.semaphores[i], ready.values[i], iree_infinite_timeout(),
          IREE_ASYNC_WAIT_FLAG_NONE));
    }
  }

  static uint32_t Hidden(const std::vector<int32_t>& tokens) {
    uint32_t value = 0;
    for (int32_t token : tokens) {
      value = (5 * value + token + 2) % 97;
    }
    return value;
  }

  static void Check(loom_serve_text_capture_snapshot_t* snapshot, size_t span,
                    uint32_t hidden) {
    const auto bytes = loom_serve_text_capture_snapshot_span(snapshot, span);
    ASSERT_EQ(bytes.data_length, 4u);
    // Source order, not native tensor interpretation: high half precedes low.
    EXPECT_EQ(iree_unaligned_load_le_u16(bytes.data), hidden >> 16);
    EXPECT_EQ(iree_unaligned_load_le_u16(bytes.data + 2), hidden & 0xFFFF);
  }

  // Owned outputs deliberately outlive their producer and subsequent epochs.
  loom_serve_text_capture_t* captures_[2] = {};
  // Owned downloaded payloads survive independent device capture release.
  loom_serve_text_capture_snapshot_t* snapshots_[2] = {};
};

TEST_F(SourceEncodingExecutionTest, KnownInputOutputOutlivesEpochAndModel) {
  EXPECT_EQ(loom_serve_text_model_tokenizer(model_), nullptr);
  const std::vector<int32_t> prefix = {0, 1, 2, 0, 1, 1, 0, 0, 2, 1, 0};
  const loom_serve_text_span_t first = {0, prefix.size(), prefix.data(), 0};
  loom_serve_execution_frontier_t ready;
  IREE_ASSERT_OK(loom_serve_text_model_submit_epoch(
      model_, 1, 1, &first, nullptr, nullptr, &captures_[0], &ready));
  ASSERT_NO_FATAL_FAILURE(Wait(ready));
  loom_serve_text_result_t progress[2] = {};
  IREE_ASSERT_OK(loom_serve_text_model_complete_epoch(model_, progress));
  EXPECT_EQ(progress[0].consumed_count, 11u);
  EXPECT_EQ(progress[0].known_count, 11u);
  EXPECT_EQ(progress[0].output_count, 0u);
  auto* row = loom_serve_text_model_row(model_, 0);
  EXPECT_EQ(loom_serve_text_row_position(row), 11u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_text_row_submit_decode(row, &ready));

  const int32_t appended = 1;
  const std::vector<int32_t> peer = {2, 2, 0};
  const loom_serve_text_span_t next[] = {{1, peer.size(), peer.data(), 0},
                                         {0, 1, &appended, 0}};
  IREE_ASSERT_OK(loom_serve_text_model_submit_epoch(
      model_, 1, 2, next, nullptr, nullptr, &captures_[1], &ready));
  ASSERT_NO_FATAL_FAILURE(Wait(ready));
  IREE_ASSERT_OK(loom_serve_text_model_complete_epoch(model_, progress));
  EXPECT_EQ(progress[0].consumed_count, 3u);
  EXPECT_EQ(progress[1].consumed_count, 1u);
  EXPECT_EQ(progress[0].output_count, 0u);
  EXPECT_EQ(progress[1].output_count, 0u);
  EXPECT_EQ(loom_serve_text_row_position(row), 12u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_text_row_submit_decode(row, &ready));

  // This frees source code, parameters, KV, and reusable residual storage.
  // Only independently allocated captures remain; downloading does not need
  // a model pointer or a live row, and the first epoch was not overwritten.
  IREE_ASSERT_OK(loom_serve_text_model_free(model_));
  model_ = nullptr;
  for (size_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(
        loom_serve_text_capture_download(captures_[i], &snapshots_[i], &ready));
    loom_serve_text_capture_release(captures_[i]);
    captures_[i] = nullptr;
    ASSERT_NO_FATAL_FAILURE(Wait(ready));
  }
  auto full = prefix;
  full.push_back(appended);
  ASSERT_NO_FATAL_FAILURE(Check(snapshots_[0], 0, Hidden(prefix)));
  ASSERT_NO_FATAL_FAILURE(Check(snapshots_[1], 0, Hidden(peer)));
  ASSERT_NO_FATAL_FAILURE(Check(snapshots_[1], 1, Hidden(full)));
}

TEST_F(SourceEncodingExecutionTest, SelectionRejectsBeforeMutation) {
  const int32_t token = 0;
  loom_serve_execution_frontier_t ready;
  const auto initial = loom_serve_text_model_pool_usage(model_);
  for (uint32_t flags : {uint32_t{LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
                         uint32_t{LOOM_SERVE_TEXT_SPAN_FLAG_SELECT |
                                  LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE}}) {
    const loom_serve_text_span_t span = {0, 1, &token, flags};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_text_model_submit_epoch(model_, 0, 1, &span, nullptr,
                                           nullptr, &captures_[0], &ready));
    EXPECT_EQ(captures_[0], nullptr);
    EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available,
              initial.available);
    EXPECT_EQ(
        loom_serve_text_row_position(loom_serve_text_model_row(model_, 0)), 0u);
  }
}

}  // namespace
