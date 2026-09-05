// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/profile.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/spirv/records/target_records.h"

namespace loom {
namespace {

TEST(SpirvTargetProfileTest, PreservesStructuredProfileFacts) {
  const loom_target_bundle_t target_bundle = {};
  loom_spirv_target_profile_t profile = {};

  loom_spirv_target_profile_initialize(&target_bundle, &profile);

  EXPECT_EQ(loom_spirv_target_profile_cast(&profile.base), &profile);
  EXPECT_EQ(loom_target_profile_bundle(&profile.base), &target_bundle);
}

TEST(SpirvTargetProfileTest, SelectsImmutableVulkanProfile) {
  const loom_spirv_target_profile_t* profile = nullptr;
  IREE_ASSERT_OK(
      loom_spirv_target_profile_select(IREE_SV("vulkan1.3+bda"), &profile));
  ASSERT_NE(profile, nullptr);
  EXPECT_EQ(profile->base.target_bundle,
            &loom_spirv_low_target_bundle_vulkan1_3);

  const loom_spirv_target_profile_t* repeated_profile = nullptr;
  IREE_ASSERT_OK(loom_spirv_target_profile_select(IREE_SV("vulkan1.3+bda"),
                                                  &repeated_profile));
  EXPECT_EQ(repeated_profile, profile);
}

TEST(SpirvTargetProfileTest, RejectsUnknownNamedProfile) {
  const loom_spirv_target_profile_t* profile = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_spirv_target_profile_select(IREE_SV("vulkan1.2"), &profile));
  EXPECT_EQ(profile, nullptr);
}

TEST(SpirvTargetProfileTest, CheckedCastRejectsAnotherFamily) {
  static const loom_target_profile_type_t kOtherProfileType = {
      /*.name=*/IREE_SVL("other"),
  };
  const loom_target_profile_t other_profile = {
      /*.type=*/&kOtherProfileType,
      /*.target_bundle=*/nullptr,
  };

  EXPECT_EQ(loom_spirv_target_profile_cast(&other_profile), nullptr);
}

}  // namespace
}  // namespace loom
