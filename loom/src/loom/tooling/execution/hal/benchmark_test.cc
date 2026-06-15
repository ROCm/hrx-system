// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/benchmark.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

TEST(HalBenchmarkTest, OptionsInitializeForFastDispatchBatchTiming) {
  loom_run_hal_benchmark_options_t options = {};
  loom_run_hal_benchmark_options_initialize(&options);

  EXPECT_EQ(options.flags, LOOM_RUN_HAL_BENCHMARK_FLAG_NONE);
  EXPECT_EQ(options.timing.batch_size, 1u);
  EXPECT_EQ(options.dispatch_batch.dispatch_count, options.timing.batch_size);
  EXPECT_TRUE(iree_all_bits_set(options.dispatch_batch.command_buffer_mode,
                                IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED));
  EXPECT_TRUE(iree_all_bits_set(options.dispatch_batch.command_buffer_mode,
                                IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED));
  EXPECT_TRUE(
      iree_all_bits_set(options.dispatch_batch.execute_flags,
                        IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME));
  EXPECT_EQ(options.profile_flags, IREE_HAL_DEVICE_PROFILING_FLAG_NONE);
  EXPECT_EQ(options.profile_data_families,
            IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA);
}

TEST(HalBenchmarkTest, DispatchPlanRejectsMismatchedBatchSizesBeforeDeviceUse) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_prepared_candidate_t candidate = {};
  loom_run_hal_invocation_plan_t plan = {};
  loom_run_hal_invocation_plan_initialize(&plan);
  loom_run_hal_benchmark_options_t options = {};
  loom_run_hal_benchmark_options_initialize(&options);
  options.timing.batch_size = 4;
  options.dispatch_batch.dispatch_count = 3;

  loom_run_hal_benchmark_result_t result = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_run_hal_benchmark_dispatch_plan(
                            &runtime, &candidate, &plan, &options,
                            iree_allocator_system(), &result));

  loom_run_hal_invocation_plan_deinitialize(&plan);
}

TEST(HalBenchmarkTest, DispatchPlanRejectsUnknownFlagsBeforeDeviceUse) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_prepared_candidate_t candidate = {};
  loom_run_hal_invocation_plan_t plan = {};
  loom_run_hal_invocation_plan_initialize(&plan);
  loom_run_hal_benchmark_options_t options = {};
  loom_run_hal_benchmark_options_initialize(&options);
  options.flags = 1u << 31;

  loom_run_hal_benchmark_result_t result = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_run_hal_benchmark_dispatch_plan(
                            &runtime, &candidate, &plan, &options,
                            iree_allocator_system(), &result));

  loom_run_hal_invocation_plan_deinitialize(&plan);
}

}  // namespace
}  // namespace loom
