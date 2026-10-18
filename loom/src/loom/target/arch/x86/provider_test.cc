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
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/target/arch/x86/facts.h"
#include "loom/target/arch/x86/feature_bits.h"
#include "loom/target/arch/x86/ops/ops.h"
#include "loom/target/arch/x86/ops/registry.h"
#include "loom/target/arch/x86/records/target_records.h"
#include "loom/target/facts_builder.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

static constexpr uint64_t kAvx2CpuFeatures = IREE_CPU_DATA0_X86_64_AVX |
                                             IREE_CPU_DATA0_X86_64_FMA |
                                             IREE_CPU_DATA0_X86_64_AVX2;
static constexpr uint64_t kAvx512CpuFeatures =
    kAvx2CpuFeatures | IREE_CPU_DATA0_X86_64_AVX512F |
    IREE_CPU_DATA0_X86_64_AVX512DQ | IREE_CPU_DATA0_X86_64_AVX512BW;

class X86ProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_x86_ops_register_dialect(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    loom_context_deinitialize(&context_);
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

  loom_x86_target_facts_t* FeatureRequirement(
      iree_string_view_t selector, loom_x86_feature_bits_t positive_features,
      loom_x86_feature_bits_t negative_features) {
    loom_x86_target_facts_t* facts = Project(Profile(selector));
    if (facts == nullptr) {
      return nullptr;
    }
    const loom_x86_target_policy_t* policy =
        loom_x86_target_policy_lookup(facts->base.selector);
    facts->base.storage.config.contract_set_key =
        policy->feature_contract_set_key;
    facts->base.storage.config.contract_feature_bits =
        (facts->base.storage.config.contract_feature_bits | positive_features) &
        ~negative_features;
    loom_target_fact_field_set_insert(&facts->base.explicit_fields,
                                      LOOM_TARGET_FACT_FIELD_CONTRACT_SET_KEY);
    facts->positive_feature_assertions = positive_features;
    facts->negative_feature_assertions = negative_features;
    return facts;
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

  ModulePtr Materialize(const loom_target_facts_t* facts,
                        iree_string_view_t target_name,
                        loom_op_t** out_target_op) {
    loom_module_t* raw_module = nullptr;
    IREE_EXPECT_OK(loom_module_allocate(&context_, IREE_SV("materialized"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &raw_module));
    ModulePtr module(raw_module);
    if (module == nullptr) {
      return module;
    }

    loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
    IREE_EXPECT_OK(
        loom_module_intern_string(module.get(), target_name, &target_name_id));
    loom_symbol_id_t target_symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_EXPECT_OK(loom_module_add_symbol(module.get(), target_name_id,
                                          &target_symbol_id));
    const loom_symbol_ref_t target_ref = {
        .module_id = 0,
        .symbol_id = target_symbol_id,
    };
    loom_builder_t builder;
    loom_builder_initialize(module.get(), &module->arena,
                            loom_module_block(module.get()), &builder);
    const loom_resolved_target_t resolved_target = {
        .provider = &loom_x86_target_provider,
        .facts = facts,
    };
    IREE_EXPECT_OK(loom_x86_target_provider.materialize_definition(
        &builder, &resolved_target, target_ref, LOOM_LOCATION_UNKNOWN));
    *out_target_op = module->symbols.entries[target_symbol_id].defining_op;
    return module;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
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
      {IREE_CPU_DATA0_X86_64_AVX, LOOM_X86_TARGET_KIND_SCALAR,
       IREE_SV("x86.scalar.core")},
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
    EXPECT_TRUE(loom_target_facts_field_is_explicit(
        &facts->base, LOOM_TARGET_FACT_FIELD_CONTRACT_SET_KEY));
    EXPECT_FALSE(loom_target_facts_field_is_explicit(
        &facts->base, LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS));
    loom_target_profile_selection_release(&selection, iree_allocator_system());
  }
}

TEST_F(X86ProviderTest, RequiresCompleteCpuFeatureClosures) {
  struct ProfileCase {
    const char* selector;
    uint64_t required_features;
  } cases[] = {
      {"avx2", kAvx2CpuFeatures},
      {"avx512", kAvx512CpuFeatures | IREE_CPU_DATA0_X86_64_AVX512VL},
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
  EXPECT_FALSE(loom_target_facts_field_is_explicit(
      &selected_facts->base, LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS));
  EXPECT_TRUE(loom_target_facts_satisfy_specialization_requirement(
      &selected_facts->base, &requirement->base));
  loom_target_profile_selection_release(&selection, iree_allocator_system());
}

TEST_F(X86ProviderTest, ProjectsCompleteCpuFactsAndExecutableFeatures) {
  iree_cpu_data_t cpu_data = {
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
      .fields = {kAvx2CpuFeatures | IREE_CPU_DATA0_X86_64_F16C |
                     IREE_CPU_DATA0_X86_64_AVXVNNI |
                     IREE_CPU_DATA0_X86_64_AVXVNNIINT8 |
                     IREE_CPU_DATA0_X86_64_AVXVNNIINT16 |
                     IREE_CPU_DATA0_X86_64_AVXNECONVERT |
                     IREE_CPU_DATA0_X86_64_BMI | IREE_CPU_DATA0_X86_64_BMI2 |
                     IREE_CPU_DATA0_X86_64_POPCNT | IREE_CPU_DATA0_X86_64_LZCNT,
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
            LOOM_X86_FEATURE_F16C | LOOM_X86_FEATURE_AVX_VNNI |
                LOOM_X86_FEATURE_AVX_VNNI_INT8 |
                LOOM_X86_FEATURE_AVX_VNNI_INT16 |
                LOOM_X86_FEATURE_AVX_NE_CONVERT | LOOM_X86_FEATURE_BMI |
                LOOM_X86_FEATURE_BMI2 | LOOM_X86_FEATURE_POPCNT |
                LOOM_X86_FEATURE_LZCNT);
  EXPECT_EQ(facts->positive_feature_assertions,
            facts->base.storage.config.contract_feature_bits);
  EXPECT_EQ(facts->negative_feature_assertions, 0u);
  loom_target_profile_selection_release(&selection, iree_allocator_system());
}

TEST_F(X86ProviderTest, ProjectsScalarFeaturesWithoutVectorProfile) {
  iree_cpu_data_t cpu_data = {
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
      .fields = {IREE_CPU_DATA0_X86_64_BMI | IREE_CPU_DATA0_X86_64_BMI2 |
                 IREE_CPU_DATA0_X86_64_POPCNT | IREE_CPU_DATA0_X86_64_LZCNT},
  };
  loom_target_profile_selection_t selection = Select(&cpu_data);
  ASSERT_NE(selection.profile, nullptr);
  const loom_x86_target_facts_t* facts = Project(selection.profile);
  ASSERT_NE(facts, nullptr);
  EXPECT_EQ(facts->base.selector, LOOM_X86_TARGET_KIND_SCALAR);
  EXPECT_EQ(facts->base.storage.config.contract_feature_bits,
            LOOM_X86_FEATURE_BMI | LOOM_X86_FEATURE_BMI2 |
                LOOM_X86_FEATURE_POPCNT | LOOM_X86_FEATURE_LZCNT);
  loom_target_profile_selection_release(&selection, iree_allocator_system());
}

TEST_F(X86ProviderTest, ProjectsOptionalInstructionFeaturesIndependently) {
  struct FeatureCase {
    // Runtime CPU feature independently advertised by the device.
    uint64_t cpu_feature;
    // Matching optional instruction-contract feature.
    loom_x86_feature_bits_t contract_feature;
  } cases[] = {
      {IREE_CPU_DATA0_X86_64_F16C, LOOM_X86_FEATURE_F16C},
      {IREE_CPU_DATA0_X86_64_AVXVNNI, LOOM_X86_FEATURE_AVX_VNNI},
      {IREE_CPU_DATA0_X86_64_AVXVNNIINT8, LOOM_X86_FEATURE_AVX_VNNI_INT8},
      {IREE_CPU_DATA0_X86_64_AVXVNNIINT16, LOOM_X86_FEATURE_AVX_VNNI_INT16},
      {IREE_CPU_DATA0_X86_64_AVXNECONVERT, LOOM_X86_FEATURE_AVX_NE_CONVERT},
      {IREE_CPU_DATA0_X86_64_AVX512VNNI, LOOM_X86_FEATURE_AVX512_VNNI},
      {IREE_CPU_DATA0_X86_64_AVX512VL, LOOM_X86_FEATURE_AVX512_VL},
      {IREE_CPU_DATA0_X86_64_AVX512BF16, LOOM_X86_FEATURE_AVX512_BF16},
      {IREE_CPU_DATA0_X86_64_AVX512FP16, LOOM_X86_FEATURE_AVX512_FP16},
      {IREE_CPU_DATA0_X86_64_BMI, LOOM_X86_FEATURE_BMI},
      {IREE_CPU_DATA0_X86_64_BMI2, LOOM_X86_FEATURE_BMI2},
      {IREE_CPU_DATA0_X86_64_POPCNT, LOOM_X86_FEATURE_POPCNT},
      {IREE_CPU_DATA0_X86_64_LZCNT, LOOM_X86_FEATURE_LZCNT},
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
    EXPECT_EQ(bundle->config->contract_feature_bits,
              feature_case.contract_feature);
    const loom_x86_target_facts_t* facts = Project(selection.profile);
    ASSERT_NE(facts, nullptr);
    EXPECT_EQ(facts->positive_feature_assertions,
              feature_case.contract_feature);
    EXPECT_EQ(facts->negative_feature_assertions,
              LOOM_X86_FEATURE_KNOWN_BITS & ~feature_case.contract_feature);
    loom_target_profile_selection_release(&selection, iree_allocator_system());
  }
}

TEST_F(X86ProviderTest, FeatureAssertionsPreserveExactProvenance) {
  const loom_x86_target_facts_t* empty_avx512 =
      FeatureRequirement(IREE_SV("avx512"), 0, 0);
  const loom_x86_target_facts_t* explicit_vl =
      FeatureRequirement(IREE_SV("avx512"), LOOM_X86_FEATURE_AVX512_VL, 0);
  ASSERT_NE(empty_avx512, nullptr);
  ASSERT_NE(explicit_vl, nullptr);
  EXPECT_EQ(empty_avx512->base.storage.config.contract_feature_bits,
            explicit_vl->base.storage.config.contract_feature_bits);
  EXPECT_FALSE(loom_target_facts_are_equivalent(&empty_avx512->base,
                                                &explicit_vl->base));

  const loom_x86_target_facts_t* empty_avx2 =
      FeatureRequirement(IREE_SV("avx2"), 0, 0);
  const loom_x86_target_facts_t* disabled_bmi2 =
      FeatureRequirement(IREE_SV("avx2"), 0, LOOM_X86_FEATURE_BMI2);
  ASSERT_NE(empty_avx2, nullptr);
  ASSERT_NE(disabled_bmi2, nullptr);
  EXPECT_EQ(empty_avx2->base.storage.config.contract_feature_bits,
            disabled_bmi2->base.storage.config.contract_feature_bits);
  EXPECT_FALSE(loom_target_facts_are_equivalent(&empty_avx2->base,
                                                &disabled_bmi2->base));
}

TEST_F(X86ProviderTest, FeatureRequirementApplicationChangesPolarity) {
  iree_cpu_data_t cpu_data = {
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
      .fields = {kAvx512CpuFeatures | IREE_CPU_DATA0_X86_64_AVX512VL |
                 IREE_CPU_DATA0_X86_64_AVX512BF16 | IREE_CPU_DATA0_X86_64_BMI2},
  };
  loom_target_profile_selection_t selection = Select(&cpu_data);
  ASSERT_NE(selection.profile, nullptr);
  loom_x86_target_facts_t* effective = Project(selection.profile);
  loom_x86_target_facts_t* requirement =
      FeatureRequirement(IREE_SV("avx512"), LOOM_X86_FEATURE_AVX512_BF16,
                         LOOM_X86_FEATURE_AVX512_VL);
  ASSERT_NE(effective, nullptr);
  ASSERT_NE(requirement, nullptr);
  ASSERT_TRUE(loom_target_facts_satisfy_specialization_requirement(
      &effective->base, &requirement->base));

  loom_target_facts_builder_apply_requirement(&requirement->base,
                                              &effective->base);
  EXPECT_FALSE(iree_any_bit_set(effective->positive_feature_assertions,
                                LOOM_X86_FEATURE_AVX512_VL));
  EXPECT_TRUE(iree_all_bits_set(effective->positive_feature_assertions,
                                LOOM_X86_FEATURE_AVX512_BF16));
  EXPECT_TRUE(iree_all_bits_set(effective->negative_feature_assertions,
                                LOOM_X86_FEATURE_AVX512_VL));
  EXPECT_FALSE(
      iree_any_bit_set(effective->base.storage.config.contract_feature_bits,
                       LOOM_X86_FEATURE_AVX512_VL));
  EXPECT_TRUE(
      iree_all_bits_set(effective->base.storage.config.contract_feature_bits,
                        LOOM_X86_FEATURE_AVX512_BF16));
  loom_target_profile_selection_release(&selection, iree_allocator_system());
}

TEST_F(X86ProviderTest, MaterializesSemanticFeatureAssertions) {
  const loom_x86_target_facts_t* signed_facts =
      FeatureRequirement(IREE_SV("avx512"), LOOM_X86_FEATURE_AVX512_BF16,
                         LOOM_X86_FEATURE_AVX512_VL);
  ASSERT_NE(signed_facts, nullptr);
  loom_op_t* signed_target = nullptr;
  ModulePtr signed_module =
      Materialize(&signed_facts->base, IREE_SV("signed"), &signed_target);
  ASSERT_NE(signed_module, nullptr);
  ASSERT_NE(signed_target, nullptr);
  EXPECT_TRUE(loom_x86_target_has_features(signed_target));
  EXPECT_FALSE(loom_x86_target_has_contract_set_key(signed_target));
  EXPECT_FALSE(loom_x86_target_has_contract_feature_bits(signed_target));
  const loom_signed_enum_set_t signed_features =
      loom_x86_target_features(signed_target);
  EXPECT_TRUE(loom_signed_enum_set_contains_positive(
      signed_features, LOOM_X86_TARGET_FEATURES_AVX512BF16));
  EXPECT_TRUE(loom_signed_enum_set_contains_negative(
      signed_features, LOOM_X86_TARGET_FEATURES_AVX512VL));

  const loom_x86_target_facts_t* empty_facts =
      FeatureRequirement(IREE_SV("avx2"), 0, 0);
  ASSERT_NE(empty_facts, nullptr);
  loom_op_t* empty_target = nullptr;
  ModulePtr empty_module =
      Materialize(&empty_facts->base, IREE_SV("empty"), &empty_target);
  ASSERT_NE(empty_module, nullptr);
  ASSERT_NE(empty_target, nullptr);
  EXPECT_TRUE(loom_x86_target_has_features(empty_target));
  EXPECT_EQ(loom_x86_target_features(empty_target).word_count, 0u);
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
