// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/runtime_requirements.h"

#include "iree/testing/gtest.h"
#include "loom/sanitizer/options.h"

namespace {

TEST(AmdgpuRuntimeRequirementsTest, NullOptionsRequireNothing) {
  EXPECT_EQ(loom_amdgpu_runtime_requirements_from_target_pipeline_options(
                /*options=*/nullptr),
            LOOM_AMDGPU_RUNTIME_REQUIREMENT_NONE);
}

TEST(AmdgpuRuntimeRequirementsTest, DisabledSanitizersRequireNothing) {
  const loom_target_pipeline_options_t options = {0};
  EXPECT_EQ(
      loom_amdgpu_runtime_requirements_from_target_pipeline_options(&options),
      LOOM_AMDGPU_RUNTIME_REQUIREMENT_NONE);
}

TEST(AmdgpuRuntimeRequirementsTest, ValueSanitizerRequiresFeedbackOnly) {
  loom_target_pipeline_options_t options = {};
  options.sanitizer.checks = LOOM_SANITIZER_CHECK_VALUE;
  EXPECT_EQ(
      loom_amdgpu_runtime_requirements_from_target_pipeline_options(&options),
      LOOM_AMDGPU_RUNTIME_REQUIREMENT_FEEDBACK);
}

TEST(AmdgpuRuntimeRequirementsTest, AccessSanitizerRequiresAsanShadow) {
  loom_target_pipeline_options_t options = {};
  options.sanitizer.checks = LOOM_SANITIZER_CHECK_ACCESS;
  EXPECT_EQ(
      loom_amdgpu_runtime_requirements_from_target_pipeline_options(&options),
      LOOM_AMDGPU_RUNTIME_REQUIREMENT_FEEDBACK |
          LOOM_AMDGPU_RUNTIME_REQUIREMENT_ASAN_SHADOW);
}

TEST(AmdgpuRuntimeRequirementsTest, RaceSanitizerRequiresTsanShadow) {
  loom_target_pipeline_options_t options = {};
  options.sanitizer.checks = LOOM_SANITIZER_CHECK_RACE;
  EXPECT_EQ(
      loom_amdgpu_runtime_requirements_from_target_pipeline_options(&options),
      LOOM_AMDGPU_RUNTIME_REQUIREMENT_FEEDBACK |
          LOOM_AMDGPU_RUNTIME_REQUIREMENT_TSAN_SHADOW);
}

TEST(AmdgpuRuntimeRequirementsTest, TrapReportingDoesNotRequireFeedback) {
  loom_target_pipeline_options_t options = {};
  options.sanitizer.checks = LOOM_SANITIZER_CHECK_ACCESS;
  options.sanitizer.reporting_mode = LOOM_SANITIZER_REPORTING_MODE_TRAP;
  EXPECT_EQ(
      loom_amdgpu_runtime_requirements_from_target_pipeline_options(&options),
      LOOM_AMDGPU_RUNTIME_REQUIREMENT_ASAN_SHADOW);

  options.sanitizer.checks = LOOM_SANITIZER_CHECK_VALUE;
  EXPECT_EQ(
      loom_amdgpu_runtime_requirements_from_target_pipeline_options(&options),
      LOOM_AMDGPU_RUNTIME_REQUIREMENT_NONE);

  options.sanitizer.checks = LOOM_SANITIZER_CHECK_RACE;
  EXPECT_EQ(
      loom_amdgpu_runtime_requirements_from_target_pipeline_options(&options),
      LOOM_AMDGPU_RUNTIME_REQUIREMENT_TSAN_SHADOW);
}

}  // namespace
