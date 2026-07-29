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
      loom_amdgpu_target_info_lookup_processor(IREE_SV("gfx942"), &processor));
  ASSERT_NE(processor, nullptr);
  const loom_amdgpu_target_identity_t identity = {
      /*.processor=*/processor,
      /*.amdhsa_features=*/
      {
          /*.sramecc=*/LOOM_AMDGPU_TARGET_FEATURE_ON,
          /*.xnack=*/LOOM_AMDGPU_TARGET_FEATURE_OFF,
      },
  };

  loom_amdgpu_target_profile_t profile = {};
  IREE_ASSERT_OK(loom_amdgpu_target_profile_initialize(&identity, &profile));

  EXPECT_EQ(loom_amdgpu_target_profile_cast(&profile.base), &profile);
  EXPECT_EQ(profile.identity.processor, processor);
  EXPECT_EQ(profile.identity.amdhsa_features.sramecc,
            LOOM_AMDGPU_TARGET_FEATURE_ON);
  EXPECT_EQ(profile.identity.amdhsa_features.xnack,
            LOOM_AMDGPU_TARGET_FEATURE_OFF);
  EXPECT_EQ(loom_target_profile_bundle(&profile.base),
            loom_amdgpu_target_bundle_for_descriptor_set(
                processor->properties.descriptor_set.ordinal));
  EXPECT_EQ(profile.properties.processor, &processor->properties);
  EXPECT_EQ(profile.properties.common,
            loom_target_profile_bundle(&profile.base));
  EXPECT_EQ(profile.properties.amdhsa_features.sramecc,
            LOOM_AMDGPU_TARGET_FEATURE_ON);
  EXPECT_EQ(profile.properties.amdhsa_features.xnack,
            LOOM_AMDGPU_TARGET_FEATURE_OFF);
}

TEST(AmdgpuTargetProfileTest, RejectsUnsupportedTargetFeatures) {
  const loom_amdgpu_processor_info_t* gfx1151 = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_processor(IREE_SV("gfx1151"), &gfx1151));
  loom_amdgpu_target_profile_t profile = {};
  const loom_amdgpu_target_identity_t default_identity = {
      /*.processor=*/gfx1151,
  };
  IREE_ASSERT_OK(
      loom_amdgpu_target_profile_initialize(&default_identity, &profile));
  EXPECT_EQ(profile.identity.amdhsa_features.sramecc,
            LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED);
  EXPECT_EQ(profile.identity.amdhsa_features.xnack,
            LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED);

  loom_amdgpu_target_identity_t invalid_identity = default_identity;
  invalid_identity.amdhsa_features.xnack = LOOM_AMDGPU_TARGET_FEATURE_ON;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_target_profile_initialize(&invalid_identity, &profile));
}

TEST(AmdgpuTargetProfileTest, ResolvesEveryGeneratedRevisionRow) {
  const loom_target_bundle_t common = {};
  const iree_host_size_t processor_count =
      loom_amdgpu_target_info_processor_count();
  for (iree_host_size_t processor_ordinal = 0;
       processor_ordinal < processor_count; ++processor_ordinal) {
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_processor_at(processor_ordinal);
    ASSERT_NE(processor, nullptr);
    loom_amdgpu_target_identity_t identity = {};
    loom_amdgpu_target_identity_initialize(processor, &identity);

    loom_amdgpu_target_properties_t properties = {};
    loom_amdgpu_target_properties_resolve(&identity, &common, &properties);
    EXPECT_EQ(properties.processor, &processor->properties);
    EXPECT_EQ(properties.common, &common);
    EXPECT_EQ(properties.instruction_constraints,
              processor->properties.instructions.base_constraints |
                  (identity.asic_revision != nullptr
                       ? identity.asic_revision->instruction_constraints
                       : 0));
    EXPECT_EQ(properties.lds_bank_service_model_set_ordinal,
              identity.asic_revision != nullptr
                  ? identity.asic_revision->lds_bank_service_model_set_ordinal
                  : processor->properties.features
                        .lds_bank_service_model_set_ordinal);
    if (processor->asic_revisions.count == 0) {
      EXPECT_EQ(identity.asic_revision, nullptr);
      EXPECT_EQ(properties.kernel_metadata_extensions.entries, nullptr);
      EXPECT_EQ(properties.kernel_metadata_extensions.count, 0u);
      continue;
    }

    ASSERT_LT(processor->asic_revisions.default_ordinal,
              processor->asic_revisions.count);
    EXPECT_EQ(identity.asic_revision,
              &processor->asic_revisions
                   .entries[processor->asic_revisions.default_ordinal]);
    for (uint16_t revision_ordinal = 0;
         revision_ordinal < processor->asic_revisions.count;
         ++revision_ordinal) {
      const loom_amdgpu_processor_asic_revision_info_t* revision =
          &processor->asic_revisions.entries[revision_ordinal];
      identity.asic_revision = revision;
      loom_amdgpu_target_properties_resolve(&identity, &common, &properties);
      EXPECT_EQ(properties.instruction_constraints,
                processor->properties.instructions.base_constraints |
                    revision->instruction_constraints);
      EXPECT_EQ(properties.lds_bank_service_model_set_ordinal,
                revision->lds_bank_service_model_set_ordinal);
      EXPECT_EQ(properties.kernel_metadata_extensions.entries,
                revision->kernel_metadata_extensions.entries);
      EXPECT_EQ(properties.kernel_metadata_extensions.count,
                revision->kernel_metadata_extensions.count);

      loom_amdgpu_target_profile_t profile = {};
      IREE_ASSERT_OK(
          loom_amdgpu_target_profile_initialize(&identity, &profile));
      EXPECT_EQ(profile.identity.asic_revision, revision);
      EXPECT_EQ(profile.properties.instruction_constraints,
                properties.instruction_constraints);
      EXPECT_EQ(profile.properties.lds_bank_service_model_set_ordinal,
                properties.lds_bank_service_model_set_ordinal);
      EXPECT_EQ(profile.properties.kernel_metadata_extensions.entries,
                properties.kernel_metadata_extensions.entries);
      EXPECT_EQ(profile.properties.kernel_metadata_extensions.count,
                properties.kernel_metadata_extensions.count);
    }
  }
}

TEST(AmdgpuTargetProfileTest, RejectsRevisionFromAnotherProcessor) {
  const loom_amdgpu_processor_info_t* revisioned_processor =
      loom_amdgpu_target_info_find_processor(IREE_SV("gfx1250"));
  const loom_amdgpu_processor_info_t* other_processor =
      loom_amdgpu_target_info_find_processor(IREE_SV("gfx1151"));
  ASSERT_NE(revisioned_processor, nullptr);
  ASSERT_GT(revisioned_processor->asic_revisions.count, 0u);
  ASSERT_NE(other_processor, nullptr);

  const loom_amdgpu_target_identity_t misplaced_revision = {
      /*.processor=*/other_processor,
      /*.amdhsa_features=*/{},
      /*.asic_revision=*/&revisioned_processor->asic_revisions.entries[0],
  };
  loom_amdgpu_target_profile_t profile = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_target_profile_initialize(&misplaced_revision, &profile));
}

TEST(AmdgpuTargetProfileTest, ExhaustsTargetFeatureSatisfactionRelation) {
  const iree_host_size_t processor_count =
      loom_amdgpu_target_info_processor_count();
  ASSERT_NE(processor_count, 0u);
  for (iree_host_size_t i = 0; i < processor_count; ++i) {
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_processor_at(i);
    ASSERT_NE(processor, nullptr);

    loom_amdgpu_target_identity_t base_identity = {};
    loom_amdgpu_target_identity_initialize(processor, &base_identity);
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

    for (uint16_t effective_revision_ordinal = 0;
         effective_revision_ordinal < processor->asic_revisions.count;
         ++effective_revision_ordinal) {
      loom_amdgpu_target_identity_t effective = base_identity;
      effective.asic_revision =
          &processor->asic_revisions.entries[effective_revision_ordinal];
      for (uint16_t required_revision_ordinal = 0;
           required_revision_ordinal < processor->asic_revisions.count;
           ++required_revision_ordinal) {
        loom_amdgpu_target_identity_t requirement = base_identity;
        requirement.asic_revision =
            &processor->asic_revisions.entries[required_revision_ordinal];
        const bool revisions_match =
            effective_revision_ordinal == required_revision_ordinal;
        EXPECT_EQ(loom_amdgpu_target_identity_equal(&effective, &requirement),
                  revisions_match);
        EXPECT_EQ(loom_amdgpu_target_identity_satisfies_requirement(
                      &effective, &requirement),
                  revisions_match);
      }
    }
  }
}

}  // namespace
}  // namespace loom
