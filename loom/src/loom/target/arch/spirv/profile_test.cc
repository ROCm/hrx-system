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
  const loom_spirv_cooperative_property_set_t cooperative_properties = {};
  loom_spirv_target_profile_t profile = {};

  loom_spirv_target_profile_initialize(&target_bundle, 0,
                                       &cooperative_properties, &profile);

  EXPECT_EQ(loom_spirv_target_profile_cast(&profile.base), &profile);
  EXPECT_EQ(loom_target_profile_bundle(&profile.base), &target_bundle);
  EXPECT_EQ(profile.cooperative_properties, &cooperative_properties);
}

TEST(SpirvTargetProfileTest, ProjectsOwnedCooperativePropertyFacts) {
  char matrix_name[] = "profile.matrix";
  loom_spirv_cooperative_matrix_property_t matrix_property = {
      /*.name=*/iree_make_cstring_view(matrix_name),
      /*.required_feature_bits=*/LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR,
      /*.m_size=*/16,
      /*.n_size=*/16,
      /*.k_size=*/16,
  };
  const loom_spirv_cooperative_property_span_t matrix_span = {
      /*.shape_key=*/loom_spirv_cooperative_matrix_shape_key(16, 16, 16),
      /*.start=*/0,
      /*.count=*/1,
  };
  char vector_name[] = "profile.vector";
  loom_spirv_cooperative_vector_property_t vector_property = {
      /*.name=*/iree_make_cstring_view(vector_name),
      /*.required_feature_bits=*/LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV,
      /*.m_size=*/32,
      /*.k_size=*/32,
  };
  const loom_spirv_cooperative_property_span_t vector_span = {
      /*.shape_key=*/(UINT64_C(32) << 16) | 32,
      /*.start=*/0,
      /*.count=*/1,
  };
  const loom_spirv_cooperative_property_set_t cooperative_properties = {
      /*.feature_bits=*/LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR |
          LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV,
      /*.matrix_properties=*/&matrix_property,
      /*.matrix_property_count=*/1,
      /*.matrix_shape_spans=*/&matrix_span,
      /*.matrix_shape_span_count=*/1,
      /*.vector_properties=*/&vector_property,
      /*.vector_property_count=*/1,
      /*.vector_shape_spans=*/&vector_span,
      /*.vector_shape_span_count=*/1,
  };
  loom_spirv_target_profile_t profile = {};
  loom_spirv_target_profile_initialize(&loom_spirv_low_target_bundle_vulkan1_3,
                                       0, &cooperative_properties, &profile);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_target_facts_t* base_facts = nullptr;
  IREE_ASSERT_OK(
      loom_target_profile_project_facts(&profile.base, &arena, &base_facts));
  const loom_spirv_target_facts_t* facts =
      loom_spirv_target_facts_cast(base_facts);
  ASSERT_NE(facts, nullptr);
  EXPECT_EQ(facts->base.selector, LOOM_SPIRV_TARGET_KIND_VULKAN1_3);
  EXPECT_NE(facts->base.storage.bundle.snapshot,
            profile.base.target_bundle->snapshot);
  ASSERT_EQ(facts->cooperative_properties.matrix_property_count, 1);
  ASSERT_EQ(facts->cooperative_properties.vector_property_count, 1);
  EXPECT_NE(facts->cooperative_properties.matrix_properties, &matrix_property);
  EXPECT_NE(facts->cooperative_properties.matrix_shape_spans, &matrix_span);
  EXPECT_NE(facts->cooperative_properties.vector_properties, &vector_property);
  EXPECT_NE(facts->cooperative_properties.vector_shape_spans, &vector_span);
  EXPECT_NE(facts->cooperative_properties.matrix_properties[0].name.data,
            matrix_name);
  EXPECT_NE(facts->cooperative_properties.vector_properties[0].name.data,
            vector_name);

  matrix_name[0] = 'x';
  matrix_property.m_size = 8;
  vector_name[0] = 'x';
  vector_property.m_size = 8;
  EXPECT_TRUE(iree_string_view_equal(
      facts->cooperative_properties.matrix_properties[0].name,
      IREE_SV("profile.matrix")));
  EXPECT_EQ(facts->cooperative_properties.matrix_properties[0].m_size, 16);
  EXPECT_TRUE(iree_string_view_equal(
      facts->cooperative_properties.vector_properties[0].name,
      IREE_SV("profile.vector")));
  EXPECT_EQ(facts->cooperative_properties.vector_properties[0].m_size, 32);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(SpirvTargetProfileTest, SelectsImmutableVulkanProfile) {
  const loom_spirv_target_profile_t* profile = nullptr;
  IREE_ASSERT_OK(
      loom_spirv_target_profile_select(IREE_SV("vulkan1.3+bda"), &profile));
  ASSERT_NE(profile, nullptr);
  EXPECT_EQ(profile->base.target_bundle,
            &loom_spirv_low_target_bundle_vulkan1_3);
  EXPECT_EQ(profile->cooperative_properties, nullptr);

  const loom_spirv_target_profile_t* repeated_profile = nullptr;
  IREE_ASSERT_OK(loom_spirv_target_profile_select(IREE_SV("vulkan1.3+bda"),
                                                  &repeated_profile));
  EXPECT_EQ(repeated_profile, profile);
}

TEST(SpirvTargetProfileTest, PreservesExplicitZeroAndPresetValues) {
  loom_target_fact_field_set_t explicit_fields = 0;
  loom_target_fact_field_set_insert(&explicit_fields,
                                    LOOM_TARGET_FACT_FIELD_LINKAGE);
  loom_target_fact_field_set_insert(&explicit_fields,
                                    LOOM_TARGET_FACT_FIELD_INDEX_BITWIDTH);
  const loom_spirv_cooperative_property_set_t no_cooperative_properties = {};
  loom_spirv_target_profile_t profile = {};
  loom_spirv_target_profile_initialize(&loom_spirv_low_target_bundle_vulkan1_3,
                                       explicit_fields,
                                       &no_cooperative_properties, &profile);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_target_facts_t* facts = nullptr;
  IREE_ASSERT_OK(
      loom_target_profile_project_facts(&profile.base, &arena, &facts));
  EXPECT_EQ(facts->explicit_fields, explicit_fields);
  EXPECT_EQ(facts->storage.export_plan.linkage, LOOM_TARGET_LINKAGE_DEFAULT);
  EXPECT_EQ(facts->storage.snapshot.index_bitwidth, 32u);
  EXPECT_FALSE(loom_target_facts_field_is_explicit(
      facts, LOOM_TARGET_FACT_FIELD_DEFAULT_POINTER_BITWIDTH));
  const auto* spirv_facts = loom_spirv_target_facts_cast(facts);
  ASSERT_NE(spirv_facts, nullptr);
  EXPECT_EQ(spirv_facts->cooperative_properties.matrix_property_count, 0u);
  EXPECT_EQ(spirv_facts->cooperative_properties.vector_property_count, 0u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(SpirvTargetProfileTest, SelectsExplicitTypeAndAbiRequirements) {
  const loom_spirv_target_profile_t* profile = nullptr;
  IREE_ASSERT_OK(loom_spirv_target_profile_select(
      IREE_SV("vulkan1.3+bda+extended-types"), &profile));
  EXPECT_TRUE(loom_target_fact_field_set_contains(
      profile->base.explicit_fields,
      LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS));
  EXPECT_TRUE(iree_all_bits_set(
      profile->base.target_bundle->config->contract_feature_bits,
      LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA | LOOM_SPIRV_FEATURE_FLOAT16 |
          LOOM_SPIRV_FEATURE_FLOAT64 | LOOM_SPIRV_FEATURE_INT8 |
          LOOM_SPIRV_FEATURE_INT16 |
          LOOM_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS |
          LOOM_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS |
          LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR));

  IREE_ASSERT_OK(
      loom_spirv_target_profile_select(IREE_SV("vulkan1.3+bda+hal"), &profile));
  EXPECT_TRUE(loom_target_fact_field_set_contains(profile->base.explicit_fields,
                                                  LOOM_TARGET_FACT_FIELD_ABI));
  EXPECT_EQ(profile->base.target_bundle->export_plan->abi_kind,
            LOOM_TARGET_ABI_HAL_KERNEL);
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
