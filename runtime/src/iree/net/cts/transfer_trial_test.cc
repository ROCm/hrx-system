// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/cts/transfer_trial.h"

#include <algorithm>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::net::cts {
namespace {

class TransferTrialTest
    : public ::testing::TestWithParam<iree::async::cts::BackendInfo> {
 protected:
  void Run(const TransferTrialOptions& options) {
    TransferTrialResult result;
    iree_status_t status = RunTransferTrial(
        GetTransportBackend(), GetParam().factory, options, {}, &result);
    if (!result.available && iree_status_is_unavailable(status)) {
      const std::string reason = iree::Status::ToString(status);
      iree_status_free(status);
      GTEST_SKIP() << reason;
    }
    IREE_ASSERT_OK(status);
    EXPECT_EQ(result.records,
              2 * options.connection_count * options.measured_records);
    EXPECT_EQ(result.payload_bytes, result.records * options.record_size);
    EXPECT_EQ(result.source_completions, result.command_messages);
    EXPECT_GE(result.command_messages,
              2 * options.connection_count *
                  ((options.measured_records + options.batch_size - 1) /
                   options.batch_size));
    EXPECT_LE(result.command_messages, result.records);
    EXPECT_GT(result.progress_messages, 0u);
    EXPECT_LE(result.progress_messages,
              result.command_messages + result.saved_progress_messages);
    if (options.progress_order == TransferProgressOrder::kNewestThenSaved) {
      EXPECT_GT(result.saved_progress_messages, 0u);
    } else {
      EXPECT_EQ(result.saved_progress_messages, 0u);
    }
    EXPECT_GT(result.window_high_water, 0u);
    EXPECT_LE(result.window_high_water, options.window_size);
    if (options.consumer_mode == TransferConsumerMode::kRetainedWindow) {
      EXPECT_EQ(result.retained.records,
                options.connection_count * options.measured_records);
      EXPECT_EQ(result.retained.windows,
                options.connection_count *
                    ((options.measured_records + options.window_size - 1) /
                     options.window_size));
      EXPECT_GT(result.retained.messages_high_water, 0u);
      EXPECT_LE(result.retained.messages_high_water, options.window_size);
      EXPECT_EQ(
          result.retained.bytes_high_water,
          std::min<uint64_t>(options.window_size, options.measured_records) *
              options.record_size);
      EXPECT_GT(result.independent_progress_messages, 0u);
    } else {
      EXPECT_EQ(result.retained.records, 0u);
      EXPECT_EQ(result.retained.messages_high_water, 0u);
    }
  }
};

TEST_P(TransferTrialTest, SingleRecordWindowMakesIndependentProgress) {
  TransferTrialOptions options;
  options.window_size = 1;
  options.warmup_records = 3;
  options.measured_records = 19;
  options.progress_policy = TransferProgressPolicy::kImmediate;
  Run(options);
}

TEST_P(TransferTrialTest, PartialBatchesFlushAnIsolatedTail) {
  TransferTrialOptions options;
  options.batch_size = 7;
  options.window_size = 13;
  options.fragment_count = 4;
  options.warmup_records = 17;
  options.measured_records = 113;
  Run(options);
}

TEST_P(TransferTrialTest, BorrowedFragmentsCrossReceiveBufferBoundaries) {
  TransferTrialOptions options;
  options.record_size = 65537;
  options.batch_size = 2;
  options.window_size = 7;
  options.fragment_count = 4;
  options.warmup_records = 3;
  options.measured_records = 11;
  Run(options);
}

TEST_P(TransferTrialTest, ManyConnectionsSharePollOwnersUnderPressure) {
  TransferTrialOptions options;
  options.connection_count = 16;
  options.batch_size = 3;
  options.window_size = 257;
  options.fragment_count = 4;
  options.warmup_records = 7;
  options.measured_records = 263;
  Run(options);
}

TEST_P(TransferTrialTest, RetainedUnitWindowsPreserveIndependentProgress) {
  TransferTrialOptions options;
  options.consumer_mode = TransferConsumerMode::kRetainedWindow;
  options.window_size = 1;
  options.warmup_records = 3;
  options.measured_records = 19;
  options.progress_policy = TransferProgressPolicy::kImmediate;
  Run(options);
}

TEST_P(TransferTrialTest, RetainedPartialWindowsReleasePhaseTails) {
  TransferTrialOptions options;
  options.consumer_mode = TransferConsumerMode::kRetainedWindow;
  options.batch_size = 7;
  options.window_size = 13;
  options.fragment_count = 4;
  options.warmup_records = 17;
  options.measured_records = 113;
  Run(options);
}

TEST_P(TransferTrialTest, RetainedViewsOutliveReceiveStorageRecycling) {
  TransferTrialOptions options;
  options.consumer_mode = TransferConsumerMode::kRetainedWindow;
  options.record_size = 65537;
  options.window_size = 65;
  options.fragment_count = 4;
  options.warmup_records = 3;
  options.measured_records = 131;
  Run(options);
}

TEST_P(TransferTrialTest, RetainedWindowsSharePollOwnersUnderPressure) {
  TransferTrialOptions options;
  options.consumer_mode = TransferConsumerMode::kRetainedWindow;
  options.connection_count = 16;
  options.batch_size = 3;
  options.window_size = 257;
  options.fragment_count = 4;
  options.warmup_records = 7;
  options.measured_records = 263;
  Run(options);
}

TEST_P(TransferTrialTest, DominatingProgressPrecedesSavedPrefixes) {
  TransferTrialOptions options;
  options.consumer_mode = TransferConsumerMode::kRetainedWindow;
  options.progress_order = TransferProgressOrder::kNewestThenSaved;
  options.batch_size = 7;
  options.window_size = 13;
  options.fragment_count = 4;
  options.warmup_records = 17;
  options.measured_records = 113;
  Run(options);
}

TEST_P(TransferTrialTest, SavedPrefixesJoinAcrossIndependentConnections) {
  TransferTrialOptions options;
  options.consumer_mode = TransferConsumerMode::kRetainedWindow;
  options.progress_order = TransferProgressOrder::kNewestThenSaved;
  options.connection_count = 16;
  options.batch_size = 3;
  options.window_size = 257;
  options.fragment_count = 4;
  options.warmup_records = 7;
  options.measured_records = 263;
  options.progress_policy = TransferProgressPolicy::kImmediate;
  Run(options);
}

CTS_REGISTER_TEST_SUITE(TransferTrialTest);

}  // namespace
}  // namespace iree::net::cts
