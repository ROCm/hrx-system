// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/profile.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(SpirvTargetProfileTest, PreservesStructuredProfileFacts) {
  const loom_target_bundle_t target_bundle = {};
  const loom_spirv_cooperative_property_set_t cooperative_properties = {};
  loom_spirv_target_profile_t profile = {};

  loom_spirv_target_profile_initialize(&target_bundle, &cooperative_properties,
                                       &profile);

  EXPECT_EQ(loom_spirv_target_profile_cast(&profile.base), &profile);
  EXPECT_EQ(loom_target_profile_bundle(&profile.base), &target_bundle);
  EXPECT_EQ(profile.cooperative_properties, &cooperative_properties);
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
