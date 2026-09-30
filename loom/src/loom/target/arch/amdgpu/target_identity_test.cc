// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/target_identity.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/target_info.h"

namespace loom {
namespace {

static const loom_amdgpu_target_info_t* LookupTarget(const char* name) {
  const loom_amdgpu_target_info_t* target = nullptr;
  IREE_CHECK_OK(loom_amdgpu_target_info_lookup_target(
      iree_make_cstring_view(name), &target));
  return target;
}

TEST(AmdgpuTargetIdentityTest, ExhaustsTargetFeatureSatisfactionRelation) {
  const iree_host_size_t target_count = loom_amdgpu_target_info_target_count();
  ASSERT_NE(target_count, 0u);
  for (iree_host_size_t i = 0; i < target_count; ++i) {
    const loom_amdgpu_target_info_t* target =
        loom_amdgpu_target_info_target_at(i);
    ASSERT_NE(target, nullptr);
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_target_processor(target);
    ASSERT_NE(processor, nullptr);

    loom_amdgpu_target_identity_t base_identity = {};
    loom_amdgpu_target_identity_initialize(target, &base_identity);
    EXPECT_TRUE(
        loom_amdgpu_target_identity_equal(&base_identity, &base_identity));
    EXPECT_TRUE(loom_amdgpu_target_identity_satisfies_requirement(
        &base_identity, &base_identity));

    loom_amdgpu_target_id_feature_support_flags_t remaining_features =
        LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_KNOWN_FLAGS;
    while (remaining_features != 0) {
      const loom_amdgpu_target_id_feature_support_bit_t feature =
          static_cast<loom_amdgpu_target_id_feature_support_bit_t>(
              remaining_features & (0u - remaining_features));
      remaining_features &= ~feature;
      loom_amdgpu_target_identity_t effective = base_identity;
      loom_amdgpu_target_identity_t requirement = base_identity;
      loom_amdgpu_target_feature_state_t* effective_state =
          loom_amdgpu_amdhsa_feature_state_select(&effective.amdhsa_features,
                                                  feature);
      loom_amdgpu_target_feature_state_t* requirement_state =
          loom_amdgpu_amdhsa_feature_state_select(&requirement.amdhsa_features,
                                                  feature);
      ASSERT_NE(effective_state, nullptr);
      ASSERT_NE(requirement_state, nullptr);

      loom_amdgpu_target_identity_t distinct_identity = base_identity;
      loom_amdgpu_target_feature_state_t* distinct_state =
          loom_amdgpu_amdhsa_feature_state_select(
              &distinct_identity.amdhsa_features, feature);
      ASSERT_NE(distinct_state, nullptr);
      *distinct_state = *distinct_state == LOOM_AMDGPU_TARGET_FEATURE_ON
                            ? LOOM_AMDGPU_TARGET_FEATURE_OFF
                            : LOOM_AMDGPU_TARGET_FEATURE_ON;
      EXPECT_FALSE(loom_amdgpu_target_identity_equal(&base_identity,
                                                     &distinct_identity));

      if (!loom_amdgpu_processor_supports_target_id_features(processor,
                                                             feature)) {
        EXPECT_EQ(*effective_state, LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED);
        EXPECT_EQ(*requirement_state, LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED);
        continue;
      }

      *requirement_state = LOOM_AMDGPU_TARGET_FEATURE_ANY;
      *effective_state = LOOM_AMDGPU_TARGET_FEATURE_OFF;
      EXPECT_TRUE(loom_amdgpu_target_identity_satisfies_requirement(
          &effective, &requirement));

      *requirement_state = LOOM_AMDGPU_TARGET_FEATURE_ON;
      *effective_state = LOOM_AMDGPU_TARGET_FEATURE_ON;
      EXPECT_TRUE(loom_amdgpu_target_identity_satisfies_requirement(
          &effective, &requirement));
      *effective_state = LOOM_AMDGPU_TARGET_FEATURE_OFF;
      EXPECT_FALSE(loom_amdgpu_target_identity_satisfies_requirement(
          &effective, &requirement));

      *requirement_state = LOOM_AMDGPU_TARGET_FEATURE_OFF;
      EXPECT_TRUE(loom_amdgpu_target_identity_satisfies_requirement(
          &effective, &requirement));
      *effective_state = LOOM_AMDGPU_TARGET_FEATURE_ON;
      EXPECT_FALSE(loom_amdgpu_target_identity_satisfies_requirement(
          &effective, &requirement));
    }

    for (iree_host_size_t other_ordinal = 0; other_ordinal < target_count;
         ++other_ordinal) {
      const loom_amdgpu_target_info_t* other_target =
          loom_amdgpu_target_info_target_at(other_ordinal);
      ASSERT_NE(other_target, nullptr);
      loom_amdgpu_target_identity_t other_identity = {};
      loom_amdgpu_target_identity_initialize(other_target, &other_identity);
      EXPECT_EQ(
          loom_amdgpu_target_identity_equal(&base_identity, &other_identity),
          target == other_target);
    }
  }
}

TEST(AmdgpuTargetIdentityTest, StrictTargetRemainsExact) {
  loom_amdgpu_target_identity_t a0 = {};
  loom_amdgpu_target_identity_initialize(LookupTarget("gfx1250-strict"), &a0);
  loom_amdgpu_target_identity_t b0 = {};
  loom_amdgpu_target_identity_initialize(LookupTarget("gfx1250"), &b0);
  loom_amdgpu_target_identity_t generic = {};
  loom_amdgpu_target_identity_initialize(LookupTarget("gfx12-5-generic"),
                                         &generic);

  EXPECT_FALSE(loom_amdgpu_target_identity_equal(&a0, &b0));
  EXPECT_FALSE(loom_amdgpu_target_identity_satisfies_requirement(&a0, &b0));
  EXPECT_FALSE(loom_amdgpu_target_identity_satisfies_requirement(&b0, &a0));
  EXPECT_FALSE(
      loom_amdgpu_target_identity_satisfies_requirement(&a0, &generic));
  EXPECT_TRUE(loom_amdgpu_target_identity_satisfies_requirement(&b0, &generic));
}

}  // namespace
}  // namespace loom
