// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/cts/collective_trial_test_util.h"

#include <algorithm>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::net::cts {

void CheckCollectiveTrial(
    const iree::async::cts::ProactorFactory& create_proactor,
    CollectiveTrialOptions options) {
  for (auto delivery :
       {CollectiveDelivery::kMessage, CollectiveDelivery::kRegistered}) {
    if (delivery == CollectiveDelivery::kRegistered &&
        !GetTransportBackend().create_registered_factory) {
      continue;
    }
    SCOPED_TRACE(delivery == CollectiveDelivery::kMessage ? "message"
                                                          : "registered");
    options.delivery = delivery;
    CollectiveTrialResult result;
    iree_status_t status = RunCollectiveTrial(
        GetTransportBackend(), create_proactor, options, {}, &result);
    if (!result.available && iree_status_is_unavailable(status)) {
      std::string reason = iree::Status::ToString(status);
      iree_status_free(status);
      GTEST_SKIP() << reason;
    }
    IREE_ASSERT_OK(status);
    EXPECT_GT(result.control_sends, 0u);
    EXPECT_EQ(result.control_completions, result.control_sends);
    EXPECT_GT(result.source_window_high_water, 0u);
    EXPECT_LE(result.source_window_high_water, options.window_size);
    if (options.schedule == CollectiveSchedule::kPipeline) {
      size_t chunks = 1 + (options.tensor_size - 1) / options.block_size;
      EXPECT_EQ(result.microbatches, options.measured_rounds);
      EXPECT_EQ(result.collectives, 0u);
      EXPECT_EQ(result.sends,
                options.measured_rounds * (options.rank_count - 1) * chunks);
      EXPECT_EQ(result.source_completions, result.sends);
      EXPECT_EQ(result.payload_bytes, options.measured_rounds *
                                          (options.rank_count - 1) *
                                          options.tensor_size);
      EXPECT_GT(result.window_high_water, 0u);
      EXPECT_LE(result.window_high_water, chunks * options.pipeline_depth);
      EXPECT_GT(result.pipeline_high_water, 0u);
      EXPECT_LE(result.pipeline_high_water, options.pipeline_depth);
      EXPECT_GT(result.first_completion_seconds, 0.0);
      EXPECT_LE(result.first_completion_seconds, result.elapsed_seconds);
      continue;
    }
    size_t chunks =
        1 + (options.tensor_size / options.rank_count - 1) / options.block_size;
    EXPECT_EQ(result.collectives, options.measured_rounds);
    EXPECT_EQ(result.sends, options.measured_rounds * options.rank_count * 2 *
                                (options.rank_count - 1) * chunks);
    EXPECT_EQ(result.source_completions, result.sends);
    EXPECT_EQ(result.payload_bytes, options.measured_rounds * 2 *
                                        (options.rank_count - 1) *
                                        options.tensor_size);
    EXPECT_GT(result.window_high_water, 0u);
    EXPECT_LE(result.window_high_water, options.window_size);
  }
}

void CheckExpertTrial(const iree::async::cts::ProactorFactory& create_proactor,
                      ExpertTrialOptions options) {
  for (auto delivery :
       {CollectiveDelivery::kMessage, CollectiveDelivery::kRegistered}) {
    if (delivery == CollectiveDelivery::kRegistered &&
        !GetTransportBackend().create_registered_factory) {
      continue;
    }
    SCOPED_TRACE(delivery == CollectiveDelivery::kMessage ? "message"
                                                          : "registered");
    options.delivery = delivery;
    ExpertTrialResult result;
    iree_status_t status = RunExpertTrial(
        GetTransportBackend(), create_proactor, options, {}, &result);
    if (!result.available && iree_status_is_unavailable(status)) {
      std::string reason = iree::Status::ToString(status);
      iree_status_free(status);
      GTEST_SKIP() << reason;
    }
    IREE_ASSERT_OK(status);
    EXPECT_GT(result.control_sends, 0u);
    EXPECT_EQ(result.control_completions, result.control_sends);
    EXPECT_EQ(result.rounds, options.measured_rounds);
    EXPECT_EQ(result.sends, result.source_completions);
    EXPECT_GT(result.sends, 0u);
    EXPECT_LE(result.source_window_high_water, options.window_size);
    EXPECT_EQ(result.dispatch_bytes,
              result.remote_tokens * options.dispatch_bytes);
    EXPECT_EQ(result.scale_bytes, result.remote_tokens * options.scale_bytes);
    EXPECT_EQ(result.combine_bytes,
              result.remote_tokens * options.combine_bytes);
    EXPECT_EQ(result.header_bytes, uint64_t(options.measured_rounds) *
                                       options.rank_count *
                                       (options.rank_count - 1) * 2 * 16);
    EXPECT_EQ(result.payload_bytes, result.dispatch_bytes + result.scale_bytes +
                                        result.combine_bytes +
                                        result.metadata_bytes +
                                        result.header_bytes);
    EXPECT_EQ(
        result.retained_handles,
        uint64_t(options.rank_count) *
            (options.measured_rounds -
             (options.measured_rounds + options.depth - 1) / options.depth));
    if (options.traffic == ExpertTraffic::kUniform) {
      EXPECT_EQ(result.tokens, uint64_t(options.measured_rounds) *
                                   options.rank_count * options.token_capacity);
    }
    if (options.traffic == ExpertTraffic::kHotspot) {
      EXPECT_EQ(result.maximum_receiver_tokens,
                uint64_t(options.rank_count - 1) * options.token_capacity);
      EXPECT_EQ(result.tokens, result.remote_tokens);
      EXPECT_EQ(result.local_routes, 0u);
    }
  }
}

void CheckGatherTrial(const iree::async::cts::ProactorFactory& create_proactor,
                      GatherTrialOptions options) {
  for (auto delivery :
       {CollectiveDelivery::kMessage, CollectiveDelivery::kRegistered}) {
    if (delivery == CollectiveDelivery::kRegistered &&
        !GetTransportBackend().create_registered_factory) {
      continue;
    }
    SCOPED_TRACE(delivery == CollectiveDelivery::kMessage ? "message"
                                                          : "registered");
    options.delivery = delivery;
    GatherTrialResult result;
    iree_status_t status = RunGatherTrial(
        GetTransportBackend(), create_proactor, options, {}, &result);
    if (!result.available && iree_status_is_unavailable(status)) {
      std::string reason = iree::Status::ToString(status);
      iree_status_free(status);
      GTEST_SKIP() << reason;
    }
    IREE_ASSERT_OK(status);
    EXPECT_GT(result.control_sends, 0u);
    EXPECT_EQ(result.control_completions, result.control_sends);
    size_t chunks = 1 + (options.shard_size - 1) / options.block_size;
    uint64_t edges = uint64_t(options.rank_count) * (options.rank_count - 1);
    EXPECT_EQ(result.gathers, options.measured_rounds);
    EXPECT_EQ(result.sends, options.measured_rounds * edges * chunks);
    EXPECT_EQ(result.source_completions, result.sends);
    EXPECT_EQ(result.payload_bytes,
              options.measured_rounds * edges * options.shard_size);
    EXPECT_LE(result.source_window_high_water, options.window_size);
    EXPECT_LE(result.window_high_water, options.depth * chunks);
    EXPECT_EQ(result.handles_high_water,
              std::min(options.depth, options.measured_rounds));
    EXPECT_EQ(
        result.out_of_order_reads,
        uint64_t(options.rank_count) *
            (options.measured_rounds -
             (options.measured_rounds + options.depth - 1) / options.depth));
    EXPECT_EQ(result.payload_storage_bytes,
              options.rank_count * options.depth *
                  (options.shard_size +
                   (options.rank_count - 1) * chunks * options.block_size));
  }
}

void CheckPipelineRoundTrip(
    const iree::async::cts::ProactorFactory& create_proactor,
    PipelineRoundTripOptions options) {
  for (auto delivery :
       {CollectiveDelivery::kMessage, CollectiveDelivery::kRegistered}) {
    if (delivery == CollectiveDelivery::kRegistered &&
        !GetTransportBackend().create_registered_factory) {
      continue;
    }
    SCOPED_TRACE(delivery == CollectiveDelivery::kMessage ? "message"
                                                          : "registered");
    options.delivery = delivery;
    PipelineRoundTripResult result;
    iree_status_t status = RunPipelineRoundTrip(
        GetTransportBackend(), create_proactor, options, {}, &result);
    if (!result.available && iree_status_is_unavailable(status)) {
      std::string reason = iree::Status::ToString(status);
      iree_status_free(status);
      GTEST_SKIP() << reason;
    }
    IREE_ASSERT_OK(status);
    EXPECT_GT(result.control_sends, 0u);
    EXPECT_EQ(result.control_completions, result.control_sends);
    size_t forward_chunks =
        1 + (options.activation_size - 1) / options.block_size;
    size_t backward_chunks =
        1 + (options.gradient_size - 1) / options.block_size;
    uint64_t edges = options.rank_count - 1;
    EXPECT_EQ(result.microbatches, options.measured_rounds);
    EXPECT_EQ(result.sends, options.measured_rounds * edges *
                                (forward_chunks + backward_chunks));
    EXPECT_EQ(result.source_completions, result.sends);
    EXPECT_EQ(result.payload_bytes,
              options.measured_rounds * edges *
                  (options.activation_size + options.gradient_size));
    EXPECT_LE(result.source_window_high_water, options.window_size);
    EXPECT_LE(result.window_high_water,
              options.depth * std::max(forward_chunks, backward_chunks));
    EXPECT_GT(result.pipeline_high_water, 0u);
    EXPECT_LE(result.pipeline_high_water, options.depth);
    EXPECT_GT(result.first_completion_seconds, 0.0);
    EXPECT_LE(result.first_completion_seconds, result.elapsed_seconds);
  }
}

}  // namespace iree::net::cts
