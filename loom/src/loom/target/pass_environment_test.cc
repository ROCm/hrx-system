// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/pass_environment.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

TEST(TargetPassEnvironmentTest, EnvironmentCarriesInvocationState) {
  const loom_target_environment_t* target_environment =
      reinterpret_cast<const loom_target_environment_t*>(uintptr_t{1});
  const loom_target_specialization_context_t specialization_context = {};
  const loom_function_version_list_t function_versions = {};
  const loom_target_pass_capability_t target_capability =
      loom_target_pass_capability_make(
          target_environment, &specialization_context, &function_versions);
  const loom_pass_environment_capability_t* capabilities[] = {
      &target_capability.base,
  };
  const loom_pass_environment_t environment =
      loom_pass_environment_make(capabilities, IREE_ARRAYSIZE(capabilities));

  IREE_ASSERT_OK(loom_pass_environment_verify(&environment));
  const loom_target_pass_capability_t* found_capability =
      loom_target_pass_capability_from_environment(&environment);
  ASSERT_EQ(found_capability, &target_capability);
  EXPECT_EQ(loom_target_pass_capability_target_environment(found_capability),
            target_environment);
  EXPECT_EQ(
      loom_target_pass_capability_specialization_context(found_capability),
      &specialization_context);
  EXPECT_EQ(loom_target_pass_capability_function_versions(found_capability),
            &function_versions);
  EXPECT_EQ(loom_target_pass_capability_specialization_profile(
                found_capability, /*module=*/nullptr, /*function=*/{}),
            nullptr);
}

TEST(TargetPassEnvironmentTest, MissingCapabilityHasEmptyAccessors) {
  EXPECT_EQ(loom_target_pass_capability_from_environment(nullptr), nullptr);
  EXPECT_EQ(loom_target_pass_capability_target_environment(nullptr), nullptr);
  EXPECT_EQ(loom_target_pass_capability_specialization_context(nullptr),
            nullptr);
  EXPECT_EQ(loom_target_pass_capability_function_versions(nullptr), nullptr);
  EXPECT_EQ(loom_target_pass_capability_specialization_profile(
                /*capability=*/nullptr, /*module=*/nullptr, /*function=*/{}),
            nullptr);
}

}  // namespace
}  // namespace loom
