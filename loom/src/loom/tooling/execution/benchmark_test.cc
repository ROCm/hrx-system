// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/benchmark.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

typedef struct FakeBatchContext {
  iree_host_size_t call_count;
} FakeBatchContext;

static iree_status_t FakeBatch(void* user_data) {
  FakeBatchContext* context = static_cast<FakeBatchContext*>(user_data);
  ++context->call_count;
  return iree_ok_status();
}

TEST(BenchmarkTest, OptionsInitializeForBatchTiming) {
  loom_run_benchmark_options_t options = {};
  loom_run_benchmark_options_initialize(&options);

  EXPECT_EQ(options.batch_size, 1u);
  EXPECT_EQ(options.warmup_batch_count, 1u);
  EXPECT_GT(options.min_batch_count, 0u);
  EXPECT_GE(options.max_batch_count, options.min_batch_count);
  EXPECT_GT(options.stable_p90_to_p50_delta_ppm, 0u);
}

TEST(BenchmarkTest, StopReasonNamesAreStable) {
  EXPECT_TRUE(iree_string_view_equal(
      loom_run_benchmark_stop_reason_name(
          LOOM_RUN_BENCHMARK_STOP_REASON_MINIMUM_REACHED),
      IREE_SV("minimum_reached")));
  EXPECT_TRUE(iree_string_view_equal(loom_run_benchmark_stop_reason_name(
                                         LOOM_RUN_BENCHMARK_STOP_REASON_STABLE),
                                     IREE_SV("stable")));
  EXPECT_TRUE(iree_string_view_equal(
      loom_run_benchmark_stop_reason_name(
          LOOM_RUN_BENCHMARK_STOP_REASON_MAX_BATCH_COUNT),
      IREE_SV("max_batch_count")));
}

TEST(BenchmarkTest, ComputesNearestRankTimingStats) {
  iree_duration_t durations[] = {90, 10, 30, 20, 50};
  loom_run_benchmark_timing_stats_t stats = {};
  IREE_ASSERT_OK(loom_run_benchmark_compute_timing_stats(
      durations, IREE_ARRAYSIZE(durations), &stats));

  EXPECT_EQ(stats.count, 5u);
  EXPECT_EQ(stats.total_ns, 200);
  EXPECT_EQ(stats.minimum_ns, 10);
  EXPECT_EQ(stats.maximum_ns, 90);
  EXPECT_DOUBLE_EQ(stats.mean_ns, 40.0);
  EXPECT_EQ(stats.p50_ns, 30);
  EXPECT_EQ(stats.p90_ns, 90);
  EXPECT_EQ(stats.p90_to_p50_delta_ppm, 2000000u);
}

TEST(BenchmarkTest, RunBatchesHonorsWarmupAndMinimums) {
  FakeBatchContext context = {};
  loom_run_benchmark_options_t options = {};
  loom_run_benchmark_options_initialize(&options);
  options.batch_size = 4;
  options.warmup_batch_count = 2;
  options.min_batch_count = 3;
  options.min_duration_ns = 0;
  options.max_batch_count = 10;
  options.stable_p90_to_p50_delta_ppm = 0;

  loom_run_benchmark_result_t result = {};
  IREE_ASSERT_OK(loom_run_benchmark_run_batches(
      (loom_run_benchmark_batch_callback_t){
          .fn = FakeBatch,
          .user_data = &context,
      },
      &options, iree_allocator_system(), &result));

  EXPECT_EQ(context.call_count, 5u);
  EXPECT_EQ(result.batch_size, 4u);
  EXPECT_EQ(result.warmup_batch_count, 2u);
  EXPECT_EQ(result.measured_batch_count, 3u);
  EXPECT_EQ(result.measured_operation_count, 12u);
  EXPECT_EQ(result.stop_reason, LOOM_RUN_BENCHMARK_STOP_REASON_MINIMUM_REACHED);
  EXPECT_EQ(result.batch_timing.count, 3u);
  EXPECT_EQ(result.operation_timing.count, 3u);
}

TEST(BenchmarkTest, RunBatchesRejectsInvalidOptions) {
  FakeBatchContext context = {};
  loom_run_benchmark_options_t options = {};
  loom_run_benchmark_options_initialize(&options);
  options.batch_size = 0;

  loom_run_benchmark_result_t result = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_run_benchmark_run_batches(
                            (loom_run_benchmark_batch_callback_t){
                                .fn = FakeBatch,
                                .user_data = &context,
                            },
                            &options, iree_allocator_system(), &result));
}

}  // namespace
}  // namespace loom
