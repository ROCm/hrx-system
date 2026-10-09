// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/provider.h"

#include "iree/base/cpu_data.h"
#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/x86/facts.h"
#include "loom/target/arch/x86/feature_bits.h"
#include "loom/target/arch/x86/ops/ops.h"

namespace loom {
namespace {

static constexpr uint64_t kAvx2CpuFeatures = IREE_CPU_DATA0_X86_64_AVX |
                                             IREE_CPU_DATA0_X86_64_FMA |
                                             IREE_CPU_DATA0_X86_64_AVX2;
static constexpr uint64_t kAvx512CpuFeatures =
    kAvx2CpuFeatures | IREE_CPU_DATA0_X86_64_AVX512F |
    IREE_CPU_DATA0_X86_64_AVX512VL | IREE_CPU_DATA0_X86_64_AVX512DQ |
    IREE_CPU_DATA0_X86_64_AVX512BW;

class X86ProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  const loom_target_profile_t* Profile(iree_string_view_t selector) {
    const loom_target_profile_t* profile = nullptr;
    IREE_EXPECT_OK(loom_x86_target_provider.select_profile(selector, &profile));
    return profile;
  }

  loom_target_profile_selection_t Select(
      const iree_cpu_data_t* cpu_data,
      const loom_target_facts_t* requirement = nullptr,
      const loom_target_profile_t* profile = nullptr) {
    loom_target_profile_selection_t selection = {};
    IREE_EXPECT_OK(loom_x86_target_provider.select_cpu_profile(
        cpu_data, requirement, profile, &selection, iree_allocator_system()));
    return selection;
  }

  loom_x86_target_facts_t* Project(const loom_target_profile_t* profile) {
    loom_target_facts_t* facts = nullptr;
    IREE_EXPECT_OK(loom_target_profile_project_facts(profile, &arena_, &facts));
    if (facts == nullptr || facts->fact_type != &loom_x86_target_fact_type) {
      return nullptr;
    }
    return reinterpret_cast<loom_x86_target_facts_t*>(facts);
  }

  void ExpectStaticSelection(const iree_cpu_data_t* cpu_data,
                             const loom_target_profile_t* profile,
                             const loom_target_profile_t* expected_profile) {
    loom_target_profile_selection_t selection =
        Select(cpu_data, nullptr, profile);
    EXPECT_EQ(selection.profile, expected_profile);
    EXPECT_EQ(selection.destroy, nullptr);
    loom_target_profile_selection_release(&selection, iree_allocator_system());
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(X86ProviderTest, PreservesLowCalls) {
  ASSERT_NE(loom_x86_target_provider.select_call_policy, nullptr);
  const loom_resolved_target_t resolved_target = {};
  EXPECT_EQ(loom_x86_target_provider.select_call_policy(
                &resolved_target, nullptr, LOOM_CALL_LIKE_KIND_LOW_INTERNAL,
                loom_call_like_t{}, loom_func_like_t{}),
            LOOM_TARGET_CALL_POLICY_DIRECT);
}

TEST_F(X86ProviderTest, SelectsStrongestExecutableCpuProfile) {
  struct SelectionCase {
    uint64_t cpu_features;
    loom_x86_target_kind_t selector;
    iree_string_view_t contract_set_key;
  } cases[] = {
      {0, LOOM_X86_TARGET_KIND_SCALAR, IREE_SV("x86.scalar.core")},
      {IREE_CPU_DATA0_X86_64_AVX, LOOM_X86_TARGET_KIND_SIMD128,
       IREE_SV("x86.simd128.core")},
      {kAvx2CpuFeatures, LOOM_X86_TARGET_KIND_AVX2,
       IREE_SV("x86.avx2_features.core")},
      {kAvx512CpuFeatures, LOOM_X86_TARGET_KIND_AVX512,
       IREE_SV("x86.avx512_features.core")},
  };
  for (const auto& selection_case : cases) {
    iree_cpu_data_t cpu_data = {
        .architecture = IREE_CPU_ARCHITECTURE_X86_64,
        .fields = {selection_case.cpu_features},
    };
    loom_target_profile_selection_t selection = Select(&cpu_data);
    ASSERT_NE(selection.profile, nullptr);
    EXPECT_NE(selection.destroy, nullptr);
    const loom_x86_target_facts_t* facts = Project(selection.profile);
    ASSERT_NE(facts, nullptr);
    EXPECT_EQ(facts->base.selector, selection_case.selector);
    EXPECT_TRUE(
        iree_string_view_equal(facts->base.storage.config.contract_set_key,
                               selection_case.contract_set_key));
    const bool composite =
        selection_case.selector == LOOM_X86_TARGET_KIND_AVX2 ||
        selection_case.selector == LOOM_X86_TARGET_KIND_AVX512;
    EXPECT_EQ(loom_target_facts_field_is_explicit(
                  &facts->base, LOOM_TARGET_FACT_FIELD_CONTRACT_SET_KEY),
              composite);
    EXPECT_EQ(loom_target_facts_field_is_explicit(
                  &facts->base, LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS),
              composite);
    loom_target_profile_selection_release(&selection, iree_allocator_system());
  }
}

TEST_F(X86ProviderTest, RequiresCompleteCpuFeatureClosures) {
  struct ProfileCase {
    const char* selector;
    uint64_t required_features;
  } cases[] = {
      {"simd128", IREE_CPU_DATA0_X86_64_AVX},
      {"avx2", kAvx2CpuFeatures},
      {"avx512", kAvx512CpuFeatures},
  };
  iree_cpu_data_t cpu_data = {
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
  };
  for (const auto& profile_case : cases) {
    const loom_target_profile_t* profile =
        Profile(iree_make_cstring_view(profile_case.selector));
    cpu_data.fields[0] = profile_case.required_features;
    SCOPED_TRACE(profile_case.selector);
    ExpectStaticSelection(&cpu_data, profile, profile);
    for (uint64_t feature = 1; feature != 0; feature <<= 1) {
      if (!iree_any_bit_set(profile_case.required_features, feature)) {
        continue;
      }
      cpu_data.fields[0] = profile_case.required_features & ~feature;
      SCOPED_TRACE(feature);
      ExpectStaticSelection(&cpu_data, profile, nullptr);
    }
  }
}

TEST_F(X86ProviderTest, PreservesAuthoredRequirementIdentity) {
  const loom_target_profile_t* avx2_profile = Profile(IREE_SV("avx2"));
  const loom_x86_target_facts_t* requirement = Project(avx2_profile);
  ASSERT_NE(requirement, nullptr);

  iree_cpu_data_t cpu_data = {
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
      .fields = {kAvx512CpuFeatures},
  };
  loom_target_profile_selection_t selection =
      Select(&cpu_data, &requirement->base);
  ASSERT_NE(selection.profile, nullptr);
  const loom_x86_target_facts_t* selected_facts = Project(selection.profile);
  ASSERT_NE(selected_facts, nullptr);
  EXPECT_EQ(selected_facts->base.selector, LOOM_X86_TARGET_KIND_AVX2);
  EXPECT_TRUE(iree_string_view_equal(
      selected_facts->base.storage.config.contract_set_key,
      IREE_SV("x86.avx2_features.core")));
  EXPECT_TRUE(loom_target_facts_field_is_explicit(
      &selected_facts->base, LOOM_TARGET_FACT_FIELD_CONTRACT_SET_KEY));
  EXPECT_TRUE(loom_target_facts_field_is_explicit(
      &selected_facts->base, LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS));
  EXPECT_TRUE(loom_target_facts_satisfy_specialization_requirement(
      &selected_facts->base, &requirement->base));
  loom_target_profile_selection_release(&selection, iree_allocator_system());
}

TEST_F(X86ProviderTest, PreservesExplicitCoreContract) {
  const loom_target_profile_t* avx2_profile = Profile(IREE_SV("avx2"));
  loom_x86_target_facts_t* requirement = Project(avx2_profile);
  ASSERT_NE(requirement, nullptr);
  loom_target_fact_field_set_insert(&requirement->base.explicit_fields,
                                    LOOM_TARGET_FACT_FIELD_CONTRACT_SET_KEY);

  iree_cpu_data_t cpu_data = {
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
      .fields = {kAvx2CpuFeatures | IREE_CPU_DATA0_X86_64_AVXVNNIINT8},
  };
  loom_target_profile_selection_t selection =
      Select(&cpu_data, &requirement->base);
  ASSERT_NE(selection.profile, nullptr);
  const loom_target_bundle_t* bundle =
      loom_target_profile_bundle(selection.profile);
  EXPECT_TRUE(iree_string_view_equal(bundle->config->contract_set_key,
                                     IREE_SV("x86.avx2.core")));
  EXPECT_EQ(bundle->config->contract_feature_bits, 0u);
  loom_target_profile_selection_release(&selection, iree_allocator_system());
}

TEST_F(X86ProviderTest, ProjectsCompleteCpuFactsAndExecutableFeatures) {
  iree_cpu_data_t cpu_data = {
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
      .fields = {kAvx2CpuFeatures | IREE_CPU_DATA0_X86_64_AVXVNNI |
                     IREE_CPU_DATA0_X86_64_AVXVNNIINT8 |
                     IREE_CPU_DATA0_X86_64_AVXVNNIINT16 |
                     IREE_CPU_DATA0_X86_64_AVXNECONVERT,
                 2, 3, 4, 5, 6, 7, 8},
  };
  loom_target_profile_selection_t selection = Select(&cpu_data);
  ASSERT_NE(selection.profile, nullptr);
  const loom_x86_target_facts_t* facts = Project(selection.profile);
  ASSERT_NE(facts, nullptr);
  EXPECT_EQ(facts->cpu_data.architecture, cpu_data.architecture);
  for (iree_host_size_t i = 0; i < IREE_CPU_DATA_FIELD_COUNT; ++i) {
    EXPECT_EQ(facts->cpu_data.fields[i], cpu_data.fields[i]);
  }
  EXPECT_EQ(facts->base.storage.config.contract_feature_bits,
            LOOM_X86_FEATURE_AVX_VNNI | LOOM_X86_FEATURE_AVX_VNNI_INT8 |
                LOOM_X86_FEATURE_AVX_VNNI_INT16 |
                LOOM_X86_FEATURE_AVX_NE_CONVERT);
  loom_target_profile_selection_release(&selection, iree_allocator_system());
}

TEST_F(X86ProviderTest, ProjectsOptionalInstructionFeaturesIndependently) {
  struct FeatureCase {
    uint64_t cpu_feature;
    loom_x86_feature_bits_t contract_feature;
  } cases[] = {
      {IREE_CPU_DATA0_X86_64_AVXVNNI, LOOM_X86_FEATURE_AVX_VNNI},
      {IREE_CPU_DATA0_X86_64_AVXVNNIINT8, LOOM_X86_FEATURE_AVX_VNNI_INT8},
      {IREE_CPU_DATA0_X86_64_AVXVNNIINT16, LOOM_X86_FEATURE_AVX_VNNI_INT16},
      {IREE_CPU_DATA0_X86_64_AVXNECONVERT, LOOM_X86_FEATURE_AVX_NE_CONVERT},
      {IREE_CPU_DATA0_X86_64_AVX512VNNI, LOOM_X86_FEATURE_AVX512_VNNI},
      {IREE_CPU_DATA0_X86_64_AVX512BF16, LOOM_X86_FEATURE_AVX512_BF16},
      {IREE_CPU_DATA0_X86_64_AVX512FP16, LOOM_X86_FEATURE_AVX512_FP16},
  };
  for (const auto& feature_case : cases) {
    iree_cpu_data_t cpu_data = {
        .architecture = IREE_CPU_ARCHITECTURE_X86_64,
        .fields = {kAvx512CpuFeatures | feature_case.cpu_feature},
    };
    loom_target_profile_selection_t selection = Select(&cpu_data);
    ASSERT_NE(selection.profile, nullptr);
    const loom_target_bundle_t* bundle =
        loom_target_profile_bundle(selection.profile);
    const loom_x86_feature_bits_t baseline = LOOM_X86_FEATURE_AVX512_VL;
    EXPECT_EQ(bundle->config->contract_feature_bits,
              baseline | feature_case.contract_feature);
    loom_target_profile_selection_release(&selection, iree_allocator_system());
  }
}

TEST_F(X86ProviderTest, DynamicProfileRequiresItsExactCpuFacts) {
  iree_cpu_data_t cpu_data = {
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
      .fields = {kAvx2CpuFeatures | IREE_CPU_DATA0_X86_64_AVXVNNIINT8, 17},
  };
  loom_target_profile_selection_t owned = Select(&cpu_data);
  ASSERT_NE(owned.profile, nullptr);
  const loom_x86_target_facts_t* effective_facts = Project(owned.profile);
  ASSERT_NE(effective_facts, nullptr);
  loom_x86_target_facts_t different_facts = *effective_facts;
  ++different_facts.cpu_data.fields[1];
  EXPECT_FALSE(loom_target_facts_satisfy_specialization_requirement(
      &effective_facts->base, &different_facts.base));

  loom_target_profile_selection_t borrowed =
      Select(&cpu_data, nullptr, owned.profile);
  EXPECT_EQ(borrowed.profile, owned.profile);
  EXPECT_EQ(borrowed.destroy, nullptr);

  cpu_data.fields[1] = 18;
  loom_target_profile_selection_t mismatched =
      Select(&cpu_data, nullptr, owned.profile);
  EXPECT_EQ(mismatched.profile, nullptr);
  loom_target_profile_selection_release(&owned, iree_allocator_system());
}

TEST_F(X86ProviderTest, RejectsNonExecutableAndForeignProfiles) {
  iree_cpu_data_t cpu_data = {
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
      .fields = {UINT64_MAX},
  };
  for (iree_string_view_t selector :
       {IREE_SV("packed_dot"), IREE_SV("avx512_packed_dot")}) {
    ExpectStaticSelection(&cpu_data, Profile(selector), nullptr);
  }

  cpu_data.architecture = IREE_CPU_ARCHITECTURE_ARM_64;
  ExpectStaticSelection(&cpu_data, Profile(IREE_SV("scalar")), nullptr);
}

}  // namespace
}  // namespace loom
