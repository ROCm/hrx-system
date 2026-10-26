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
static constexpr uint64_t kAmdVendorLow = UINT64_C(0x69746e6568747541);
static constexpr uint32_t kAmdVendorHigh = UINT32_C(0x444d4163);

static iree_cpu_data_t AmdCpuData(uint16_t family, uint8_t model,
                                  uint64_t features = kAvx2CpuFeatures) {
  const uint32_t base_family = family < 0x0F ? family : 0x0F;
  const uint32_t extended_family = family < 0x0F ? 0 : family - 0x0F;
  const uint32_t signature = (base_family << 8) | (extended_family << 20) |
                             ((uint32_t)(model & 0x0F) << 4) |
                             ((uint32_t)(model >> 4) << 16);
  return iree_cpu_data_t{
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
      .fields =
          {
              features,
              kAmdVendorLow,
              (uint64_t)kAmdVendorHigh | ((uint64_t)signature << 32),
          },
  };
}

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
    facts->base.storage.config.contract_set = policy->feature_contract_set;
    facts->base.storage.config.contract_feature_bits =
        (facts->base.storage.config.contract_feature_bits | positive_features) &
        ~negative_features;
    loom_target_fact_field_set_insert(&facts->base.explicit_fields,
                                      LOOM_TARGET_FACT_FIELD_CONTRACT_SET_KEY);
    facts->positive_feature_assertions = positive_features;
    facts->negative_feature_assertions = negative_features;
    return facts;
  }

  loom_x86_target_facts_t* TuneRequirement(iree_string_view_t selector,
                                           loom_x86_target_tune_t tune) {
    loom_x86_target_facts_t* facts = Project(Profile(selector));
    if (facts != nullptr) {
      facts->tune = tune;
      facts->tune_is_explicit = true;
      facts->base.fact_type->rebind(&facts->base);
    }
    return facts;
  }

  void ExpectExplicitSelectionAccepted(const iree_cpu_data_t* cpu_data,
                                       const loom_target_profile_t* profile) {
    loom_target_profile_selection_t selection =
        Select(cpu_data, nullptr, profile);
    ASSERT_NE(selection.profile, nullptr);
    EXPECT_NE(selection.destroy, nullptr);
    const loom_x86_target_facts_t* facts = Project(selection.profile);
    ASSERT_NE(facts, nullptr);
    const loom_x86_target_facts_t* requested = Project(profile);
    ASSERT_NE(requested, nullptr);
    EXPECT_EQ(facts->base.selector, requested->base.selector);
    EXPECT_EQ(facts->executable_target_kinds,
              LOOM_X86_TARGET_KIND_BIT(facts->base.selector));
    EXPECT_EQ(facts->cpu_data.architecture, cpu_data->architecture);
    for (iree_host_size_t i = 0; i < IREE_CPU_DATA_FIELD_COUNT; ++i) {
      EXPECT_EQ(facts->cpu_data.fields[i], cpu_data->fields[i]);
    }
    loom_target_profile_selection_release(&selection, iree_allocator_system());
  }

  void ExpectExplicitSelectionRejected(const iree_cpu_data_t* cpu_data,
                                       const loom_target_profile_t* profile) {
    loom_target_profile_selection_t selection =
        Select(cpu_data, nullptr, profile);
    EXPECT_EQ(selection.profile, nullptr);
    EXPECT_EQ(selection.destroy, nullptr);
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
    loom_x86_target_kind_set_t executable_target_kinds;
    uint64_t contract_set_stable_id;
    iree_string_view_t contract_set_spelling;
  } cases[] = {
      {0, LOOM_X86_TARGET_KIND_SCALAR,
       LOOM_X86_TARGET_KIND_BIT(LOOM_X86_TARGET_KIND_SCALAR),
       UINT64_C(0x374da30c241d0cea), IREE_SV("x86.scalar.core")},
      {IREE_CPU_DATA0_X86_64_AVX, LOOM_X86_TARGET_KIND_SCALAR,
       LOOM_X86_TARGET_KIND_BIT(LOOM_X86_TARGET_KIND_SCALAR),
       UINT64_C(0x374da30c241d0cea), IREE_SV("x86.scalar.core")},
      {kAvx2CpuFeatures, LOOM_X86_TARGET_KIND_AVX2,
       LOOM_X86_TARGET_KIND_BIT(LOOM_X86_TARGET_KIND_SCALAR) |
           LOOM_X86_TARGET_KIND_BIT(LOOM_X86_TARGET_KIND_AVX2),
       UINT64_C(0x4a4769fa07a2ddeb), IREE_SV("x86.avx2_features.core")},
      {kAvx512CpuFeatures, LOOM_X86_TARGET_KIND_AVX512,
       LOOM_X86_TARGET_KIND_BIT(LOOM_X86_TARGET_KIND_AVX512) |
           LOOM_X86_TARGET_KIND_BIT(LOOM_X86_TARGET_KIND_SCALAR) |
           LOOM_X86_TARGET_KIND_BIT(LOOM_X86_TARGET_KIND_AVX2),
       UINT64_C(0x7ddea86e8df4a0d5), IREE_SV("x86.avx512_features.core")},
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
    EXPECT_EQ(facts->base.storage.config.contract_set.stable_id,
              selection_case.contract_set_stable_id);
    EXPECT_TRUE(
        iree_string_view_equal(facts->base.storage.config.contract_set.spelling,
                               selection_case.contract_set_spelling));
    EXPECT_TRUE(loom_target_facts_field_is_explicit(
        &facts->base, LOOM_TARGET_FACT_FIELD_CONTRACT_SET_KEY));
    EXPECT_FALSE(loom_target_facts_field_is_explicit(
        &facts->base, LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS));
    EXPECT_EQ(facts->executable_target_kinds,
              selection_case.executable_target_kinds);
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
    ExpectExplicitSelectionAccepted(&cpu_data, profile);
    for (uint64_t feature = 1; feature != 0; feature <<= 1) {
      if (!iree_any_bit_set(profile_case.required_features, feature)) {
        continue;
      }
      cpu_data.fields[0] = profile_case.required_features & ~feature;
      SCOPED_TRACE(feature);
      ExpectExplicitSelectionRejected(&cpu_data, profile);
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
      selected_facts->base.storage.config.contract_set.spelling,
      IREE_SV("x86.avx2_features.core")));
  EXPECT_TRUE(loom_target_facts_field_is_explicit(
      &selected_facts->base, LOOM_TARGET_FACT_FIELD_CONTRACT_SET_KEY));
  EXPECT_FALSE(loom_target_facts_field_is_explicit(
      &selected_facts->base, LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS));
  EXPECT_EQ(selected_facts->executable_target_kinds,
            LOOM_X86_TARGET_KIND_BIT(LOOM_X86_TARGET_KIND_AVX2));
  EXPECT_TRUE(loom_target_facts_satisfy_specialization_requirement(
      &selected_facts->base, &requirement->base));
  loom_target_profile_selection_release(&selection, iree_allocator_system());
}

TEST_F(X86ProviderTest, RefinesAutomaticCpuProfileToAuthoredTarget) {
  const iree_cpu_data_t cpu_data =
      AmdCpuData(0x1A, 0x70,
                 kAvx512CpuFeatures | IREE_CPU_DATA0_X86_64_F16C |
                     IREE_CPU_DATA0_X86_64_BMI2);
  loom_target_profile_selection_t selection = Select(&cpu_data);
  ASSERT_NE(selection.profile, nullptr);
  const loom_x86_target_facts_t* available = Project(selection.profile);
  loom_x86_target_facts_t* requirement = FeatureRequirement(
      IREE_SV("avx2"), LOOM_X86_FEATURE_F16C, LOOM_X86_FEATURE_BMI2);
  ASSERT_NE(available, nullptr);
  ASSERT_NE(requirement, nullptr);
  requirement->tune = LOOM_X86_TARGET_TUNE_ZNVER5;
  requirement->tune_is_explicit = true;
  requirement->base.fact_type->rebind(&requirement->base);
  ASSERT_EQ(available->base.selector, LOOM_X86_TARGET_KIND_AVX512);
  ASSERT_EQ(available->executable_target_kinds,
            LOOM_X86_TARGET_KIND_BIT(LOOM_X86_TARGET_KIND_AVX512) |
                LOOM_X86_TARGET_KIND_BIT(LOOM_X86_TARGET_KIND_SCALAR) |
                LOOM_X86_TARGET_KIND_BIT(LOOM_X86_TARGET_KIND_AVX2));
  ASSERT_TRUE(loom_target_facts_satisfy_specialization_requirement(
      &available->base, &requirement->base));

  loom_target_facts_t* refined_base = nullptr;
  IREE_ASSERT_OK(loom_target_facts_builder_clone(&available->base, &arena_,
                                                 &refined_base));
  loom_target_facts_builder_apply_requirement(&requirement->base, refined_base);
  const auto* refined =
      reinterpret_cast<const loom_x86_target_facts_t*>(refined_base);
  EXPECT_EQ(refined->base.selector, LOOM_X86_TARGET_KIND_AVX2);
  EXPECT_EQ(refined->executable_target_kinds,
            LOOM_X86_TARGET_KIND_BIT(LOOM_X86_TARGET_KIND_AVX2));
  EXPECT_TRUE(iree_string_view_equal(refined->base.storage.bundle.name,
                                     IREE_SV("x86-avx2")));
  EXPECT_TRUE(
      iree_string_view_equal(refined->base.storage.config.contract_set.spelling,
                             IREE_SV("x86.avx2_features.core")));
  EXPECT_TRUE(
      iree_all_bits_set(refined->base.storage.config.contract_feature_bits,
                        LOOM_X86_FEATURE_F16C));
  EXPECT_FALSE(
      iree_any_bit_set(refined->base.storage.config.contract_feature_bits,
                       LOOM_X86_FEATURE_BMI2));
  EXPECT_TRUE(iree_all_bits_set(refined->positive_feature_assertions,
                                LOOM_X86_FEATURE_F16C));
  EXPECT_TRUE(iree_all_bits_set(refined->negative_feature_assertions,
                                LOOM_X86_FEATURE_BMI2));
  EXPECT_EQ(refined->tune, LOOM_X86_TARGET_TUNE_ZNVER5);
  EXPECT_TRUE(refined->tune_is_explicit);
  EXPECT_EQ(refined->cpu_data.architecture, cpu_data.architecture);
  for (iree_host_size_t i = 0; i < IREE_CPU_DATA_FIELD_COUNT; ++i) {
    EXPECT_EQ(refined->cpu_data.fields[i], cpu_data.fields[i]);
  }

  const loom_target_facts_t* execution = nullptr;
  IREE_ASSERT_OK(loom_target_facts_builder_select_execution(
      refined_base, &arena_, &execution));
  EXPECT_EQ(execution, refined_base);
  loom_target_profile_selection_release(&selection, iree_allocator_system());
}

TEST_F(X86ProviderTest, CommitsPreferredAutomaticTargetBeforeExecution) {
  const iree_cpu_data_t cpu_data = {
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
      .fields = {kAvx512CpuFeatures},
  };
  loom_target_profile_selection_t selection = Select(&cpu_data);
  ASSERT_NE(selection.profile, nullptr);
  const loom_x86_target_facts_t* available = Project(selection.profile);
  ASSERT_NE(available, nullptr);
  ASSERT_EQ(available->executable_target_kinds,
            LOOM_X86_TARGET_KIND_BIT(LOOM_X86_TARGET_KIND_AVX512) |
                LOOM_X86_TARGET_KIND_BIT(LOOM_X86_TARGET_KIND_SCALAR) |
                LOOM_X86_TARGET_KIND_BIT(LOOM_X86_TARGET_KIND_AVX2));

  const loom_target_facts_t* execution_base = nullptr;
  IREE_ASSERT_OK(loom_target_facts_builder_select_execution(
      &available->base, &arena_, &execution_base));
  ASSERT_NE(execution_base, &available->base);
  const auto* execution =
      reinterpret_cast<const loom_x86_target_facts_t*>(execution_base);
  EXPECT_EQ(execution->base.selector, LOOM_X86_TARGET_KIND_AVX512);
  EXPECT_EQ(execution->executable_target_kinds,
            LOOM_X86_TARGET_KIND_BIT(LOOM_X86_TARGET_KIND_AVX512));
  EXPECT_FALSE(
      loom_target_facts_are_equivalent(&available->base, execution_base));

  const loom_x86_target_facts_t* avx2 = Project(Profile(IREE_SV("avx2")));
  ASSERT_NE(avx2, nullptr);
  EXPECT_TRUE(loom_target_facts_satisfy_specialization_requirement(
      &available->base, &avx2->base));
  EXPECT_FALSE(loom_target_facts_satisfy_specialization_requirement(
      execution_base, &avx2->base));
  const loom_x86_target_facts_t* packed_dot =
      Project(Profile(IREE_SV("packed_dot")));
  ASSERT_NE(packed_dot, nullptr);
  EXPECT_FALSE(loom_target_facts_satisfy_specialization_requirement(
      &available->base, &packed_dot->base));

  const loom_x86_target_facts_t* exact_avx512 =
      Project(Profile(IREE_SV("avx512")));
  ASSERT_NE(exact_avx512, nullptr);
  EXPECT_EQ(exact_avx512->executable_target_kinds,
            LOOM_X86_TARGET_KIND_BIT(LOOM_X86_TARGET_KIND_AVX512));
  EXPECT_FALSE(loom_target_facts_satisfy_specialization_requirement(
      &exact_avx512->base, &avx2->base));
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

TEST_F(X86ProviderTest, RecognizesSupportedProcessorDeploymentRanges) {
  struct DeploymentCase {
    uint16_t family;
    uint8_t model;
    loom_x86_deployment_profile_ordinal_t expected_profile;
  } cases[] = {
      {0x1A, 0x00, LOOM_X86_DEPLOYMENT_PROFILE_ZNVER5},
      {0x1A, 0x4F, LOOM_X86_DEPLOYMENT_PROFILE_ZNVER5},
      {0x1A, 0x50, LOOM_X86_DEPLOYMENT_PROFILE_ZNVER6},
      {0x1A, 0x5F, LOOM_X86_DEPLOYMENT_PROFILE_ZNVER6},
      {0x1A, 0x60, LOOM_X86_DEPLOYMENT_PROFILE_ZNVER5},
      {0x1A, 0x77, LOOM_X86_DEPLOYMENT_PROFILE_ZNVER5},
      {0x1A, 0x78, LOOM_X86_DEPLOYMENT_PROFILE_NONE},
      {0x1A, 0x80, LOOM_X86_DEPLOYMENT_PROFILE_ZNVER6},
      {0x1A, 0xCF, LOOM_X86_DEPLOYMENT_PROFILE_ZNVER6},
      {0x1A, 0xD0, LOOM_X86_DEPLOYMENT_PROFILE_ZNVER5},
      {0x1A, 0xD7, LOOM_X86_DEPLOYMENT_PROFILE_ZNVER5},
      {0x1A, 0xD8, LOOM_X86_DEPLOYMENT_PROFILE_ZNVER6},
      {0x1A, 0xE7, LOOM_X86_DEPLOYMENT_PROFILE_ZNVER6},
      {0x1A, 0xE8, LOOM_X86_DEPLOYMENT_PROFILE_NONE},
      {0x19, 0x70, LOOM_X86_DEPLOYMENT_PROFILE_NONE},
  };
  for (const auto& deployment_case : cases) {
    const iree_cpu_data_t cpu_data =
        AmdCpuData(deployment_case.family, deployment_case.model);
    SCOPED_TRACE(deployment_case.model);
    const loom_x86_deployment_profile_t* profile =
        loom_x86_deployment_profile_select_cpu(&cpu_data);
    EXPECT_EQ(profile != nullptr ? profile->ordinal
                                 : LOOM_X86_DEPLOYMENT_PROFILE_NONE,
              deployment_case.expected_profile);
  }

  iree_cpu_data_t non_amd = AmdCpuData(0x1A, 0x70);
  non_amd.fields[1] = UINT64_C(0x756e65476c65746e);
  non_amd.fields[2] =
      (non_amd.fields[2] & UINT64_C(0xFFFFFFFF00000000)) | UINT32_C(0x49656e69);
  EXPECT_EQ(loom_x86_deployment_profile_select_cpu(&non_amd), nullptr);
}

TEST_F(X86ProviderTest, ProjectsNamedProcessorProfilesOffline) {
  struct ProfileCase {
    iree_string_view_t name;
    uint8_t model;
  } cases[] = {
      {IREE_SV("znver5"), 0x70},
      {IREE_SV("znver6"), 0x50},
  };
  for (const auto& profile_case : cases) {
    const loom_x86_deployment_profile_t* deployment =
        loom_x86_deployment_profile_lookup(profile_case.name);
    ASSERT_NE(deployment, nullptr);
    const loom_x86_target_facts_t* offline =
        Project(Profile(profile_case.name));
    ASSERT_NE(offline, nullptr);

    const iree_cpu_data_t cpu_data =
        AmdCpuData(0x1A, profile_case.model, deployment->cpu_field0_bits);
    loom_target_profile_selection_t runtime_selection = Select(&cpu_data);
    ASSERT_NE(runtime_selection.profile, nullptr);
    const loom_x86_target_facts_t* runtime = Project(runtime_selection.profile);
    ASSERT_NE(runtime, nullptr);

    EXPECT_EQ(offline->base.selector, runtime->base.selector);
    EXPECT_EQ(offline->executable_target_kinds,
              runtime->executable_target_kinds);
    EXPECT_EQ(offline->cpu_data.fields[0], runtime->cpu_data.fields[0]);
    EXPECT_EQ(offline->tune, runtime->tune);
    EXPECT_EQ(offline->base.storage.config.schedule_model_stable_id,
              runtime->base.storage.config.schedule_model_stable_id);
    EXPECT_EQ(offline->base.storage.config.contract_set.stable_id,
              runtime->base.storage.config.contract_set.stable_id);
    EXPECT_EQ(offline->base.storage.config.contract_feature_bits,
              runtime->base.storage.config.contract_feature_bits);
    loom_target_profile_selection_release(&runtime_selection,
                                          iree_allocator_system());
  }
}

TEST_F(X86ProviderTest, ValidatesNamedProcessorGuaranteesOnLiveCpu) {
  const loom_x86_deployment_profile_t* znver5 =
      loom_x86_deployment_profile_lookup(IREE_SV("znver5"));
  ASSERT_NE(znver5, nullptr);
  iree_cpu_data_t cpu_data = {
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
      .fields = {znver5->cpu_field0_bits},
  };
  const loom_target_profile_t* profile = Profile(IREE_SV("znver5"));
  loom_target_profile_selection_t accepted =
      Select(&cpu_data, nullptr, profile);
  EXPECT_EQ(accepted.profile, profile);
  EXPECT_EQ(accepted.destroy, nullptr);

  cpu_data.fields[0] &= ~IREE_CPU_DATA0_X86_64_AVX512BF16;
  ExpectExplicitSelectionRejected(&cpu_data, profile);
  ExpectExplicitSelectionRejected(&cpu_data, Profile(IREE_SV("znver6")));
}

TEST_F(X86ProviderTest, ProjectsDetectedTuneIntoTargetConfig) {
  struct TuneCase {
    uint8_t model;
    loom_x86_target_tune_t expected_tune;
  } cases[] = {
      {0x70, LOOM_X86_TARGET_TUNE_ZNVER5},
      {0x50, LOOM_X86_TARGET_TUNE_ZNVER6},
      {0x78, LOOM_X86_TARGET_TUNE_NONE},
  };
  for (const auto& tune_case : cases) {
    const iree_cpu_data_t cpu_data = AmdCpuData(0x1A, tune_case.model);
    loom_target_profile_selection_t selection = Select(&cpu_data);
    ASSERT_NE(selection.profile, nullptr);
    const loom_x86_target_facts_t* facts = Project(selection.profile);
    ASSERT_NE(facts, nullptr);
    EXPECT_EQ(facts->tune, tune_case.expected_tune);
    EXPECT_FALSE(facts->tune_is_explicit);
    const loom_x86_target_tune_info_t* tune_info =
        loom_x86_target_tune_info_lookup(tune_case.expected_tune);
    EXPECT_EQ(facts->base.storage.config.schedule_model_stable_id,
              tune_info->schedule_model_stable_id);
    loom_target_profile_selection_release(&selection, iree_allocator_system());
  }
}

TEST_F(X86ProviderTest, AuthoredTuneConstrainsAndMaterializes) {
  const iree_cpu_data_t cpu_data = AmdCpuData(0x1A, 0x70);
  loom_target_profile_selection_t selection = Select(&cpu_data);
  ASSERT_NE(selection.profile, nullptr);
  loom_x86_target_facts_t* detected = Project(selection.profile);
  loom_x86_target_facts_t* required_zen5 =
      TuneRequirement(IREE_SV("avx2"), LOOM_X86_TARGET_TUNE_ZNVER5);
  loom_x86_target_facts_t* required_zen6 =
      TuneRequirement(IREE_SV("avx2"), LOOM_X86_TARGET_TUNE_ZNVER6);
  ASSERT_NE(detected, nullptr);
  ASSERT_NE(required_zen5, nullptr);
  ASSERT_NE(required_zen6, nullptr);
  EXPECT_EQ(required_zen5->base.storage.config.schedule_model_stable_id,
            loom_x86_target_tune_info_lookup(LOOM_X86_TARGET_TUNE_ZNVER5)
                ->schedule_model_stable_id);
  EXPECT_EQ(required_zen6->base.storage.config.schedule_model_stable_id,
            loom_x86_target_tune_info_lookup(LOOM_X86_TARGET_TUNE_ZNVER6)
                ->schedule_model_stable_id);
  EXPECT_TRUE(loom_target_facts_satisfy_specialization_requirement(
      &detected->base, &required_zen5->base));
  EXPECT_FALSE(loom_target_facts_satisfy_specialization_requirement(
      &detected->base, &required_zen6->base));

  loom_op_t* detected_target = nullptr;
  ModulePtr detected_module =
      Materialize(&detected->base, IREE_SV("detected"), &detected_target);
  ASSERT_NE(detected_module, nullptr);
  ASSERT_NE(detected_target, nullptr);
  EXPECT_FALSE(loom_x86_target_has_tune(detected_target));

  loom_target_facts_builder_apply_requirement(&required_zen5->base,
                                              &detected->base);
  EXPECT_TRUE(detected->tune_is_explicit);
  EXPECT_EQ(detected->base.storage.config.schedule_model_stable_id,
            loom_x86_target_tune_info_lookup(LOOM_X86_TARGET_TUNE_ZNVER5)
                ->schedule_model_stable_id);
  loom_op_t* specialized_target = nullptr;
  ModulePtr specialized_module =
      Materialize(&detected->base, IREE_SV("specialized"), &specialized_target);
  ASSERT_NE(specialized_module, nullptr);
  ASSERT_NE(specialized_target, nullptr);
  EXPECT_TRUE(loom_x86_target_has_tune(specialized_target));
  EXPECT_EQ(loom_x86_target_tune(specialized_target),
            LOOM_X86_TARGET_TUNE_ZNVER5);
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

TEST_F(X86ProviderTest, DynamicProfileReselectionOwnsExactCpuFacts) {
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

  loom_target_profile_selection_t reselected =
      Select(&cpu_data, nullptr, owned.profile);
  ASSERT_NE(reselected.profile, nullptr);
  EXPECT_NE(reselected.profile, owned.profile);
  EXPECT_NE(reselected.destroy, nullptr);
  const loom_x86_target_facts_t* reselected_facts = Project(reselected.profile);
  ASSERT_NE(reselected_facts, nullptr);
  EXPECT_TRUE(loom_target_facts_are_equivalent(&effective_facts->base,
                                               &reselected_facts->base));

  cpu_data.fields[1] = 18;
  loom_target_profile_selection_t mismatched =
      Select(&cpu_data, nullptr, owned.profile);
  EXPECT_EQ(mismatched.profile, nullptr);
  loom_target_profile_selection_release(&reselected, iree_allocator_system());
  loom_target_profile_selection_release(&owned, iree_allocator_system());
}

TEST_F(X86ProviderTest, RejectsNonExecutableAndForeignProfiles) {
  iree_cpu_data_t cpu_data = {
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
      .fields = {UINT64_MAX},
  };
  for (iree_string_view_t selector :
       {IREE_SV("packed_dot"), IREE_SV("avx512_packed_dot")}) {
    ExpectExplicitSelectionRejected(&cpu_data, Profile(selector));
  }

  cpu_data.architecture = IREE_CPU_ARCHITECTURE_ARM_64;
  ExpectExplicitSelectionRejected(&cpu_data, Profile(IREE_SV("scalar")));
}

}  // namespace
}  // namespace loom
