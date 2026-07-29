// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/profile.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/records/target_records.h"

namespace loom {
namespace {

TEST(AmdgpuTargetProfileTest, PreservesStructuredTargetFacts) {
  const loom_amdgpu_processor_info_t* processor = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_processor(IREE_SV("gfx1151"), &processor));
  ASSERT_NE(processor, nullptr);
  const loom_amdgpu_amdhsa_profile_facts_t amdhsa = {
      /*.processor=*/processor,
      /*.gfx1250_revision=*/LOOM_AMDGPU_GFX1250_REVISION_UNSPECIFIED,
      /*.sramecc=*/LOOM_AMDGPU_TARGET_FEATURE_ON,
      /*.xnack=*/LOOM_AMDGPU_TARGET_FEATURE_OFF,
  };

  loom_amdgpu_target_profile_t profile = {};
  IREE_ASSERT_OK(loom_amdgpu_target_profile_initialize(&amdhsa, &profile));

  EXPECT_EQ(loom_amdgpu_target_profile_cast(&profile.base), &profile);
  EXPECT_EQ(profile.amdhsa.processor, processor);
  EXPECT_EQ(profile.amdhsa.gfx1250_revision,
            LOOM_AMDGPU_GFX1250_REVISION_UNSPECIFIED);
  EXPECT_EQ(profile.amdhsa.sramecc, LOOM_AMDGPU_TARGET_FEATURE_ON);
  EXPECT_EQ(profile.amdhsa.xnack, LOOM_AMDGPU_TARGET_FEATURE_OFF);
  EXPECT_EQ(loom_target_profile_bundle(&profile.base),
            loom_amdgpu_target_bundle_for_descriptor_set(
                processor->descriptor_set.ordinal));
}

TEST(AmdgpuTargetProfileTest, ValidatesGfx1250RevisionIdentity) {
  const loom_amdgpu_processor_info_t* gfx1250 = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_processor(IREE_SV("gfx1250"), &gfx1250));
  const loom_amdgpu_processor_info_t* gfx1151 = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_processor(IREE_SV("gfx1151"), &gfx1151));

  loom_amdgpu_target_profile_t profile = {};
  const loom_amdgpu_amdhsa_profile_facts_t missing_revision = {
      /*.processor=*/gfx1250,
      /*.gfx1250_revision=*/LOOM_AMDGPU_GFX1250_REVISION_UNSPECIFIED,
      /*.sramecc=*/LOOM_AMDGPU_TARGET_FEATURE_DEFAULT,
      /*.xnack=*/LOOM_AMDGPU_TARGET_FEATURE_DEFAULT,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_target_profile_initialize(&missing_revision, &profile));

  const loom_amdgpu_amdhsa_profile_facts_t misplaced_revision = {
      /*.processor=*/gfx1151,
      /*.gfx1250_revision=*/LOOM_AMDGPU_GFX1250_REVISION_A0,
      /*.sramecc=*/LOOM_AMDGPU_TARGET_FEATURE_DEFAULT,
      /*.xnack=*/LOOM_AMDGPU_TARGET_FEATURE_DEFAULT,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_target_profile_initialize(&misplaced_revision, &profile));
}

}  // namespace
}  // namespace loom
