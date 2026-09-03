// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/profile.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/records/target_records.h"

namespace loom {
namespace {

static const loom_amdgpu_target_info_t* LookupTarget(const char* name) {
  const loom_amdgpu_target_info_t* target = nullptr;
  IREE_CHECK_OK(loom_amdgpu_target_info_lookup_target(
      iree_make_cstring_view(name), &target));
  return target;
}

TEST(AmdgpuTargetProfileTest, PreservesStructuredTargetFacts) {
  const loom_amdgpu_target_info_t* target = LookupTarget("gfx942");
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_target_processor(target);
  ASSERT_NE(processor, nullptr);
  const loom_amdgpu_target_identity_t identity = {
      /*.target=*/target,
      /*.amdhsa_features=*/
      {
          /*.sramecc=*/LOOM_AMDGPU_TARGET_FEATURE_ON,
          /*.xnack=*/LOOM_AMDGPU_TARGET_FEATURE_OFF,
      },
  };

  loom_amdgpu_target_profile_t profile = {};
  IREE_ASSERT_OK(loom_amdgpu_target_profile_initialize(&identity, &profile));

  EXPECT_EQ(loom_amdgpu_target_profile_cast(&profile.base), &profile);
  EXPECT_EQ(profile.identity.target, target);
  EXPECT_EQ(profile.identity.amdhsa_features.sramecc,
            LOOM_AMDGPU_TARGET_FEATURE_ON);
  EXPECT_EQ(profile.identity.amdhsa_features.xnack,
            LOOM_AMDGPU_TARGET_FEATURE_OFF);
  EXPECT_EQ(loom_target_profile_bundle(&profile.base),
            loom_amdgpu_target_profile_bundle_for_descriptor_set(
                target->descriptor_set_ordinal));
  EXPECT_EQ(profile.properties.target, target);
  EXPECT_EQ(profile.properties.processor, &processor->properties);
  EXPECT_EQ(profile.properties.common,
            loom_target_profile_bundle(&profile.base));
  EXPECT_EQ(profile.properties.amdhsa_features.sramecc,
            LOOM_AMDGPU_TARGET_FEATURE_ON);
  EXPECT_EQ(profile.properties.amdhsa_features.xnack,
            LOOM_AMDGPU_TARGET_FEATURE_OFF);
}

TEST(AmdgpuTargetProfileTest, ProjectsCompilerOwnedTypedFacts) {
  const loom_amdgpu_target_info_t* target = LookupTarget("gfx1151");
  loom_amdgpu_target_identity_t identity = {};
  loom_amdgpu_target_identity_initialize(target, &identity);
  loom_amdgpu_target_profile_t profile = {};
  IREE_ASSERT_OK(loom_amdgpu_target_profile_initialize(&identity, &profile));

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_target_facts_t* base_facts = nullptr;
  IREE_ASSERT_OK(
      loom_target_profile_project_facts(&profile.base, &arena, &base_facts));
  const loom_amdgpu_target_facts_t* facts =
      loom_amdgpu_target_facts_cast(base_facts);
  ASSERT_NE(facts, nullptr);
  EXPECT_EQ(facts->base.selector, target->target_kind);
  EXPECT_EQ(facts->identity.target, target);
  EXPECT_EQ(facts->identity.amdhsa_features.xnack,
            profile.identity.amdhsa_features.xnack);
  EXPECT_TRUE(iree_string_view_equal(
      loom_target_facts_identity_name(base_facts), IREE_SV("gfx1151")));
  EXPECT_EQ(facts->properties.target, target);
  EXPECT_EQ(facts->properties.common, &facts->base.storage.bundle);
  EXPECT_NE(facts->properties.common, profile.base.target_bundle);
  EXPECT_EQ(facts->base.storage.snapshot.codegen_format,
            LOOM_TARGET_CODEGEN_FORMAT_UNKNOWN);
  EXPECT_EQ(facts->base.storage.snapshot.artifact_format,
            LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN);
  EXPECT_EQ(facts->base.storage.export_plan.abi_kind, LOOM_TARGET_ABI_UNKNOWN);
  EXPECT_FALSE(facts->subgroup_size_explicit);
  EXPECT_FALSE(facts->contract_set_key_explicit);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(AmdgpuTargetProfileTest, SelectsTargetLocalDescriptorContract) {
  const loom_amdgpu_target_info_t* gfx1250 = LookupTarget("gfx1250");
  const loom_amdgpu_target_info_t* gfx1250_a0 = LookupTarget("gfx1250-a0");
  ASSERT_NE(gfx1250->descriptor_set_ordinal,
            gfx1250_a0->descriptor_set_ordinal);
  EXPECT_FALSE(iree_string_view_equal(gfx1250->descriptor_set_key,
                                      gfx1250_a0->descriptor_set_key));

  loom_amdgpu_target_identity_t b0_identity = {};
  loom_amdgpu_target_identity_initialize(gfx1250, &b0_identity);
  loom_amdgpu_target_profile_t b0_profile = {};
  IREE_ASSERT_OK(
      loom_amdgpu_target_profile_initialize(&b0_identity, &b0_profile));

  loom_amdgpu_target_identity_t a0_identity = {};
  loom_amdgpu_target_identity_initialize(gfx1250_a0, &a0_identity);
  loom_amdgpu_target_profile_t a0_profile = {};
  IREE_ASSERT_OK(
      loom_amdgpu_target_profile_initialize(&a0_identity, &a0_profile));

  EXPECT_NE(loom_target_profile_bundle(&b0_profile.base),
            loom_target_profile_bundle(&a0_profile.base));
  EXPECT_EQ(b0_profile.properties.processor, a0_profile.properties.processor);
  EXPECT_EQ(b0_profile.properties.target, gfx1250);
  EXPECT_EQ(a0_profile.properties.target, gfx1250_a0);
}

TEST(AmdgpuTargetProfileTest, RejectsUnsupportedTargetFeatures) {
  const loom_amdgpu_target_info_t* gfx1151 = LookupTarget("gfx1151");
  loom_amdgpu_target_profile_t profile = {};
  const loom_amdgpu_target_identity_t default_identity = {
      /*.target=*/gfx1151,
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

TEST(AmdgpuTargetProfileTest, ResolvesEveryGeneratedTargetRow) {
  const loom_target_bundle_t common = {};
  const iree_host_size_t target_count = loom_amdgpu_target_info_target_count();
  ASSERT_NE(target_count, 0u);
  for (iree_host_size_t target_ordinal = 0; target_ordinal < target_count;
       ++target_ordinal) {
    const loom_amdgpu_target_info_t* target =
        loom_amdgpu_target_info_target_at(target_ordinal);
    ASSERT_NE(target, nullptr);
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_target_processor(target);
    ASSERT_NE(processor, nullptr);

    loom_amdgpu_target_identity_t identity = {};
    loom_amdgpu_target_identity_initialize(target, &identity);
    EXPECT_EQ(identity.target, target);

    loom_amdgpu_target_properties_t properties = {};
    loom_amdgpu_target_properties_resolve(&identity, &common, &properties);
    EXPECT_EQ(properties.target, target);
    EXPECT_EQ(properties.processor, &processor->properties);
    EXPECT_EQ(properties.common, &common);
    EXPECT_EQ(properties.instruction_constraints,
              target->instruction_constraints);
    EXPECT_EQ(properties.lds_bank_service_model_set_ordinal,
              target->lds_bank_service_model_set_ordinal);
    EXPECT_EQ(properties.kernel_metadata_extensions.entries,
              target->kernel_metadata_extensions.entries);
    EXPECT_EQ(properties.kernel_metadata_extensions.count,
              target->kernel_metadata_extensions.count);

    loom_amdgpu_target_profile_t profile = {};
    IREE_ASSERT_OK(loom_amdgpu_target_profile_initialize(&identity, &profile));
    EXPECT_EQ(profile.identity.target, target);
    EXPECT_EQ(profile.properties.target, target);
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

TEST(AmdgpuTargetProfileTest, ExhaustsTargetFeatureSatisfactionRelation) {
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

TEST(AmdgpuTargetProfileTest, OverlayTargetsRemainExact) {
  loom_amdgpu_target_identity_t a0 = {};
  loom_amdgpu_target_identity_initialize(LookupTarget("gfx1250-a0"), &a0);
  loom_amdgpu_target_identity_t b0 = {};
  loom_amdgpu_target_identity_initialize(LookupTarget("gfx1250"), &b0);
  loom_amdgpu_target_identity_t generic = {};
  loom_amdgpu_target_identity_initialize(LookupTarget("gfx12-5-generic"),
                                         &generic);

  EXPECT_FALSE(loom_amdgpu_target_identity_equal(&a0, &b0));
  EXPECT_FALSE(loom_amdgpu_target_identity_satisfies_requirement(&a0, &b0));
  EXPECT_FALSE(loom_amdgpu_target_identity_satisfies_requirement(&b0, &a0));
  EXPECT_TRUE(loom_amdgpu_target_identity_satisfies_requirement(&a0, &generic));
  EXPECT_TRUE(loom_amdgpu_target_identity_satisfies_requirement(&b0, &generic));
}

}  // namespace
}  // namespace loom
