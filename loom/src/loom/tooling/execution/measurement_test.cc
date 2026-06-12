// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/measurement.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

iree_status_t IncrementStep(void* user_data) {
  auto* value = reinterpret_cast<int*>(user_data);
  ++*value;
  return iree_ok_status();
}

iree_status_t FailStep(void* user_data) {
  (void)user_data;
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "intentional failure");
}

TEST(MeasurementTest, DisabledMeasurementRunsCallbackWithoutSamples) {
  loom_run_measurement_options_t options = {};
  loom_run_measurement_options_initialize(&options);
  loom_run_measurement_sample_t samples[1] = {};
  loom_run_measurement_result_t result = {};
  loom_run_measurement_result_initialize(samples, IREE_ARRAYSIZE(samples),
                                         &result);

  int value = 0;
  const loom_run_measurement_step_callback_t callback = {
      /*.fn=*/IncrementStep,
      /*.user_data=*/&value,
  };
  IREE_ASSERT_OK(loom_run_measurement_run_step(
      &options, LOOM_RUN_MEASUREMENT_BOUNDARY_INVOKE, callback, &result));

  EXPECT_EQ(value, 1);
  EXPECT_EQ(result.sample_count, 0u);
}

TEST(MeasurementTest, TimingRecordsSelectedBoundary) {
  loom_run_measurement_options_t options = {};
  loom_run_measurement_options_initialize(&options);
  options.kind_flags = LOOM_RUN_MEASUREMENT_KIND_TIMING;
  options.boundary_flags = LOOM_RUN_MEASUREMENT_BOUNDARY_INVOKE;
  loom_run_measurement_sample_t samples[1] = {};
  loom_run_measurement_result_t result = {};
  loom_run_measurement_result_initialize(samples, IREE_ARRAYSIZE(samples),
                                         &result);

  int value = 0;
  const loom_run_measurement_step_callback_t callback = {
      /*.fn=*/IncrementStep,
      /*.user_data=*/&value,
  };
  IREE_ASSERT_OK(loom_run_measurement_run_step(
      &options, LOOM_RUN_MEASUREMENT_BOUNDARY_INVOKE, callback, &result));

  EXPECT_EQ(value, 1);
  ASSERT_EQ(result.sample_count, 1u);
  EXPECT_EQ(samples[0].boundary, LOOM_RUN_MEASUREMENT_BOUNDARY_INVOKE);
  EXPECT_EQ(samples[0].kind_flags, LOOM_RUN_MEASUREMENT_KIND_TIMING);
  EXPECT_GE(samples[0].duration_ns, 0);
  EXPECT_EQ(samples[0].status_code, IREE_STATUS_OK);
}

TEST(MeasurementTest, TimingSkipsUnselectedBoundary) {
  loom_run_measurement_options_t options = {};
  loom_run_measurement_options_initialize(&options);
  options.kind_flags = LOOM_RUN_MEASUREMENT_KIND_TIMING;
  options.boundary_flags = LOOM_RUN_MEASUREMENT_BOUNDARY_COLLECT_RESULTS;
  loom_run_measurement_sample_t samples[1] = {};
  loom_run_measurement_result_t result = {};
  loom_run_measurement_result_initialize(samples, IREE_ARRAYSIZE(samples),
                                         &result);

  int value = 0;
  const loom_run_measurement_step_callback_t callback = {
      /*.fn=*/IncrementStep,
      /*.user_data=*/&value,
  };
  IREE_ASSERT_OK(loom_run_measurement_run_step(
      &options, LOOM_RUN_MEASUREMENT_BOUNDARY_INVOKE, callback, &result));

  EXPECT_EQ(value, 1);
  EXPECT_EQ(result.sample_count, 0u);
}

TEST(MeasurementTest, TimingRequiresSampleCapacityBeforeCallback) {
  loom_run_measurement_options_t options = {};
  loom_run_measurement_options_initialize(&options);
  options.kind_flags = LOOM_RUN_MEASUREMENT_KIND_TIMING;
  options.boundary_flags = LOOM_RUN_MEASUREMENT_BOUNDARY_INVOKE;
  loom_run_measurement_result_t result = {};
  loom_run_measurement_result_initialize(nullptr, 0, &result);

  int value = 0;
  const loom_run_measurement_step_callback_t callback = {
      /*.fn=*/IncrementStep,
      /*.user_data=*/&value,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_measurement_run_step(
          &options, LOOM_RUN_MEASUREMENT_BOUNDARY_INVOKE, callback, &result));

  EXPECT_EQ(value, 0);
  EXPECT_EQ(result.sample_count, 0u);
}

TEST(MeasurementTest, CallbackFailureStillRecordsStatusCode) {
  loom_run_measurement_options_t options = {};
  loom_run_measurement_options_initialize(&options);
  options.kind_flags = LOOM_RUN_MEASUREMENT_KIND_TIMING;
  options.boundary_flags = LOOM_RUN_MEASUREMENT_BOUNDARY_INVOKE;
  loom_run_measurement_sample_t samples[1] = {};
  loom_run_measurement_result_t result = {};
  loom_run_measurement_result_initialize(samples, IREE_ARRAYSIZE(samples),
                                         &result);

  const loom_run_measurement_step_callback_t callback = {
      /*.fn=*/FailStep,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_measurement_run_step(
          &options, LOOM_RUN_MEASUREMENT_BOUNDARY_INVOKE, callback, &result));

  ASSERT_EQ(result.sample_count, 1u);
  EXPECT_EQ(samples[0].status_code, IREE_STATUS_INVALID_ARGUMENT);
}

TEST(MeasurementTest, UnsupportedMeasurementKindsFailLoud) {
  loom_run_measurement_options_t options = {};
  loom_run_measurement_options_initialize(&options);
  options.kind_flags = LOOM_RUN_MEASUREMENT_KIND_DEEP_PROFILE;
  options.boundary_flags = LOOM_RUN_MEASUREMENT_BOUNDARY_INVOKE;
  loom_run_measurement_sample_t samples[1] = {};
  loom_run_measurement_result_t result = {};
  loom_run_measurement_result_initialize(samples, IREE_ARRAYSIZE(samples),
                                         &result);

  int value = 0;
  const loom_run_measurement_step_callback_t callback = {
      /*.fn=*/IncrementStep,
      /*.user_data=*/&value,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      loom_run_measurement_run_step(
          &options, LOOM_RUN_MEASUREMENT_BOUNDARY_INVOKE, callback, &result));

  EXPECT_EQ(value, 0);
  EXPECT_EQ(result.sample_count, 0u);
}

TEST(MeasurementTest, BoundaryNamesAreReadable) {
  EXPECT_TRUE(iree_string_view_equal(
      loom_run_measurement_boundary_name(
          LOOM_RUN_MEASUREMENT_BOUNDARY_PREPARE_CANDIDATE),
      IREE_SV("prepare-candidate")));
  EXPECT_TRUE(iree_string_view_equal(
      loom_run_measurement_boundary_name(LOOM_RUN_MEASUREMENT_BOUNDARY_INVOKE),
      IREE_SV("invoke")));
}

}  // namespace
}  // namespace loom
