// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/cts/direct_transfer_trial.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::net::cts {
namespace {

class DirectTransferTrialTest
    : public ::testing::TestWithParam<iree::async::cts::BackendInfo> {
 protected:
  void Run(const DirectTransferTrialOptions& options) {
    DirectTransferTrialResult result;
    iree_status_t status = RunDirectTransferTrial(
        GetTransportBackend(), GetParam().factory, options, {}, &result);
    if (!result.available && iree_status_is_unavailable(status)) {
      std::string reason = iree::Status::ToString(status);
      iree_status_free(status);
      GTEST_SKIP() << reason;
    }
    IREE_ASSERT_OK(status);
    EXPECT_EQ(result.records,
              2 * options.connection_count * options.measured_records);
    EXPECT_EQ(result.payload_bytes, result.records * options.record_size);
    EXPECT_EQ(result.source_completions, result.records);
    size_t record_extent = options.record_size;
    if (options.layout == DirectTransferLayout::kPermutedPages) {
      record_extent =
          (options.record_size / options.fragment_count +
           (options.record_size % options.fragment_count != 0) + 1) *
          options.fragment_count;
    }
    EXPECT_EQ(
        result.payload_storage_bytes,
        4 * options.connection_count * options.window_size * record_extent);
    EXPECT_GT(result.progress_messages, 0u);
    EXPECT_LE(result.progress_messages, result.records);
    EXPECT_GT(result.window_high_water, 0u);
    EXPECT_LE(result.window_high_water, options.window_size);
    if (options.consumer_mode == TransferConsumerMode::kRetainedWindow) {
      EXPECT_EQ(result.retained_records,
                options.connection_count * options.measured_records);
      EXPECT_GT(result.independent_progress_messages, 0u);
    } else {
      EXPECT_EQ(result.retained_records, 0u);
    }
  }
};

TEST_P(DirectTransferTrialTest, ReusedUnitSlotsJoinSourceAndConsumer) {
  DirectTransferTrialOptions options;
  options.window_size = 1;
  options.measured_records = 39;
  options.progress_policy = TransferProgressPolicy::kImmediate;
  Run(options);
}

TEST_P(DirectTransferTrialTest, RetainedTargetsDoNotConsumeNotificationCredit) {
  DirectTransferTrialOptions options;
  options.consumer_mode = TransferConsumerMode::kRetainedWindow;
  options.window_size = 257;
  options.fragment_count = 4;
  options.measured_records = 1031;
  Run(options);
}

TEST_P(DirectTransferTrialTest, PartialWindowsReturnAnIsolatedTail) {
  DirectTransferTrialOptions options;
  options.consumer_mode = TransferConsumerMode::kRetainedWindow;
  options.window_size = 13;
  options.record_size = 65537;
  options.fragment_count = 4;
  options.warmup_records = 17;
  options.measured_records = 31;
  Run(options);
}

TEST_P(DirectTransferTrialTest, DescriptorBoundariesPreservePlacement) {
  for (size_t count : {1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129}) {
    SCOPED_TRACE(count);
    DirectTransferTrialOptions options;
    options.record_size = 4099;
    options.fragment_count = count;
    options.window_size = 13;
    options.warmup_records = 3;
    options.measured_records = 29;
    options.consumer_mode = TransferConsumerMode::kRetainedWindow;
    ASSERT_NO_FATAL_FAILURE(Run(options));
  }
}

TEST_P(DirectTransferTrialTest,
       ConnectionsShareRegistrationNotCompletionState) {
  DirectTransferTrialOptions options;
  options.connection_count = 16;
  options.consumer_mode = TransferConsumerMode::kRetainedWindow;
  options.window_size = 65;
  options.fragment_count = 4;
  options.measured_records = 263;
  Run(options);
}

TEST_P(DirectTransferTrialTest, RemappedPagesReuseSlotsWithDifferentLayouts) {
  DirectTransferTrialOptions options;
  options.layout = DirectTransferLayout::kPermutedPages;
  options.window_size = 1;
  options.record_size = 7 * 4096;
  options.fragment_count = 7;
  options.warmup_records = 3;
  options.measured_records = 19;
  options.progress_policy = TransferProgressPolicy::kImmediate;
  Run(options);
}

TEST_P(DirectTransferTrialTest, RemappedPageBatchesCrossPostingBoundaries) {
  for (size_t count : {31, 32, 33, 63, 64, 65, 127, 128, 129}) {
    SCOPED_TRACE(count);
    DirectTransferTrialOptions options;
    options.layout = DirectTransferLayout::kPermutedPages;
    options.record_size = count * 4096;
    options.fragment_count = count;
    options.window_size = 3;
    options.warmup_records = 2;
    options.measured_records = 7;
    options.consumer_mode = TransferConsumerMode::kRetainedWindow;
    ASSERT_NO_FATAL_FAILURE(Run(options));
  }
}

TEST_P(DirectTransferTrialTest, RemappedPartialPagesRetainIndependentProgress) {
  DirectTransferTrialOptions options;
  options.layout = DirectTransferLayout::kPermutedPages;
  options.connection_count = 4;
  options.record_size = 65537;
  options.fragment_count = 17;
  options.window_size = 5;
  options.warmup_records = 3;
  options.measured_records = 17;
  options.consumer_mode = TransferConsumerMode::kRetainedWindow;
  Run(options);
}

CTS_REGISTER_TEST_SUITE(DirectTransferTrialTest);

}  // namespace
}  // namespace iree::net::cts
