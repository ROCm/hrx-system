// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/provider.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/target/facts.h"
#include "loom/pass/builder.h"
#include "loom/target/arch/spirv/features.h"
#include "loom/target/arch/spirv/ops/ops.h"
#include "loom/target/arch/spirv/profile.h"
#include "loom/target/arch/spirv/records/target_records.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class SpirvProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_target_environment_initialize(
        &loom_spirv_target_provider_set, &target_environment_));
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_target_environment_register_context(
        &target_environment_, &context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_symbol_fact_table_initialize(&requirement_fact_table_,
                                      &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_target_environment_deinitialize(&target_environment_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr Parse(iree_string_view_t source) {
    loom_text_parse_options_t parse_options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(source, IREE_SV("spirv_provider_test.loom"),
                                  &context_, &block_pool_, &parse_options,
                                  &module));
    IREE_ASSERT(module != nullptr);
    return ModulePtr(module);
  }

  loom_symbol_ref_t FindSymbolRef(const loom_module_t* module,
                                  iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    const uint16_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return (loom_symbol_ref_t){
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
  }

  const loom_target_symbol_facts_t* Target(loom_symbol_fact_table_t* fact_table,
                                           const loom_module_t* module,
                                           loom_symbol_ref_t target_ref) {
    const loom_symbol_facts_base_t* base_facts = nullptr;
    IREE_CHECK_OK(loom_symbol_fact_table_lookup_ref(fact_table, module,
                                                    target_ref, &base_facts));
    const loom_target_symbol_facts_t* target_facts =
        loom_target_symbol_facts_cast(base_facts);
    IREE_ASSERT(target_facts != nullptr);
    return target_facts;
  }

  const loom_target_symbol_facts_t* Requirement(const loom_module_t* module,
                                                iree_string_view_t name) {
    return Target(&requirement_fact_table_, module,
                  FindSymbolRef(module, name));
  }

  bool SatisfiesSpecialization(const loom_target_facts_t* effective,
                               const loom_target_symbol_facts_t* requirement) {
    return loom_target_facts_satisfy_specialization_requirement(
        effective, requirement->projection);
  }

  bool SatisfiesIdentity(const loom_target_facts_t* effective,
                         const loom_target_symbol_facts_t* requirement) {
    return loom_target_facts_satisfy_identity_requirement(
        effective, requirement->projection);
  }

  // Block pool shared by parsed modules and analysis storage.
  iree_arena_block_pool_t block_pool_;

  // Finalized context containing the production SPIR-V dialect package.
  loom_context_t context_;

  // Production SPIR-V target provider environment under test.
  loom_target_environment_t target_environment_;

  // Arena retaining indexed target facts for both test modules.
  iree_arena_allocator_t analysis_arena_;

  // Dense target fact index for authored requirements.
  loom_symbol_fact_table_t requirement_fact_table_;
};

TEST_F(SpirvProviderTest, RequiresLowCallsInline) {
  EXPECT_EQ(loom_spirv_target_provider.target_fact_type,
            &loom_spirv_target_fact_type);
  ASSERT_NE(loom_spirv_target_provider.select_low_call_policy, nullptr);
  const loom_resolved_target_t resolved_target = {};
  EXPECT_EQ(loom_spirv_target_provider.select_low_call_policy(&resolved_target),
            LOOM_TARGET_LOW_CALL_POLICY_REQUIRE_INLINE);
}

TEST_F(SpirvProviderTest, ProjectedProfileSatisfiesStructuredRequirements) {
  ModulePtr requirements =
      Parse(IREE_SV("spirv.target<vulkan1_3> @baseline_a\n"
                    "spirv.target<vulkan1_3> @baseline_b\n"
                    "spirv.target<vulkan1_3> @fitting "
                    "{max_workgroup_size_x = 128, subgroup_size = 32}\n"
                    "spirv.target<vulkan1_3> @too_large "
                    "{max_workgroup_size_x = 512}\n"
                    "spirv.target<vulkan1_3> @subgroup64 {subgroup_size = 64}\n"
                    "spirv.target<vulkan1_3> @index64 {index_bitwidth = 64}\n"
                    "spirv.target<vulkan1_3> @float16 "
                    "{contract_feature_bits = 8}\n"
                    "spirv.target<vulkan1_3> @float64 "
                    "{contract_feature_bits = 16}\n"));
  loom_target_bundle_storage_t live_storage = {
      /*.snapshot=*/*loom_spirv_target_profile_bundle_vulkan1_3.snapshot,
      /*.export_plan=*/*loom_spirv_target_profile_bundle_vulkan1_3.export_plan,
      /*.config=*/*loom_spirv_target_profile_bundle_vulkan1_3.config,
      /*.bundle=*/loom_spirv_target_profile_bundle_vulkan1_3,
  };
  loom_target_bundle_storage_rebind(&live_storage);
  live_storage.bundle.name = IREE_SV("live-vulkan-device");
  live_storage.snapshot.max_workgroup_size = {
      /*.x=*/256,
      /*.y=*/128,
      /*.z=*/64,
  };
  live_storage.snapshot.max_flat_workgroup_size = 256;
  live_storage.snapshot.subgroup_size = 32;
  live_storage.snapshot.max_workgroup_count = {
      /*.x=*/65535,
      /*.y=*/65535,
      /*.z=*/65535,
  };
  live_storage.config.contract_feature_bits |= LOOM_SPIRV_FEATURE_FLOAT16;

  loom_spirv_target_profile_t profile = {};
  loom_spirv_target_profile_initialize(
      &live_storage.bundle, /*cooperative_properties=*/nullptr, &profile);
  loom_target_facts_t* effective = nullptr;
  IREE_ASSERT_OK(loom_target_profile_project_facts(
      &profile.base, &analysis_arena_, &effective));
  ASSERT_NE(effective, nullptr);
  EXPECT_EQ(effective->storage.export_plan.abi_kind, LOOM_TARGET_ABI_UNKNOWN);

  const loom_target_symbol_facts_t* baseline_a =
      Requirement(requirements.get(), IREE_SV("baseline_a"));
  const loom_target_symbol_facts_t* baseline_b =
      Requirement(requirements.get(), IREE_SV("baseline_b"));
  EXPECT_TRUE(SatisfiesIdentity(effective, baseline_a));
  EXPECT_TRUE(SatisfiesIdentity(effective, baseline_b));
  EXPECT_TRUE(SatisfiesSpecialization(effective, baseline_a));
  EXPECT_TRUE(SatisfiesSpecialization(effective, baseline_b));
  EXPECT_TRUE(SatisfiesSpecialization(
      effective, Requirement(requirements.get(), IREE_SV("fitting"))));
  EXPECT_FALSE(SatisfiesSpecialization(
      effective, Requirement(requirements.get(), IREE_SV("too_large"))));
  EXPECT_FALSE(SatisfiesSpecialization(
      effective, Requirement(requirements.get(), IREE_SV("subgroup64"))));
  EXPECT_FALSE(SatisfiesSpecialization(
      effective, Requirement(requirements.get(), IREE_SV("index64"))));
  EXPECT_TRUE(SatisfiesSpecialization(
      effective, Requirement(requirements.get(), IREE_SV("float16"))));
  EXPECT_FALSE(SatisfiesSpecialization(
      effective, Requirement(requirements.get(), IREE_SV("float64"))));

  EXPECT_EQ(baseline_a->projection->storage.export_plan.abi_kind,
            LOOM_TARGET_ABI_SHADER_ENTRY_POINT);
  EXPECT_FALSE(loom_target_facts_field_is_explicit(baseline_a->projection,
                                                   LOOM_TARGET_FACT_FIELD_ABI));
}

TEST_F(SpirvProviderTest, SelectedProfileRoundTripsThroughTargetDefinition) {
  loom_target_profile_selection_t selection = {};
  IREE_ASSERT_OK(loom_target_environment_select_profile(
      &target_environment_, IREE_SV("spirv:vulkan1.3+bda"),
      iree_allocator_system(), &selection));
  EXPECT_EQ(selection.provider, &loom_spirv_target_provider);
  EXPECT_TRUE(
      iree_string_view_equal(selection.selector, IREE_SV("vulkan1.3+bda")));

  loom_target_facts_t* selected_facts = nullptr;
  IREE_ASSERT_OK(loom_target_profile_project_facts(
      selection.profile, &analysis_arena_, &selected_facts));
  ASSERT_NE(selected_facts, nullptr);

  loom_module_t* raw_module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("materialized"),
                                      &block_pool_, nullptr,
                                      iree_allocator_system(), &raw_module));
  ModulePtr module(raw_module);
  loom_string_id_t symbol_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module.get(), IREE_SV("selected"),
                                           &symbol_name_id));
  loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(module.get(), symbol_name_id, &symbol_id));
  const loom_symbol_ref_t symbol = {
      /*.module_id=*/0,
      /*.symbol_id=*/symbol_id,
  };
  loom_builder_t builder;
  loom_builder_initialize(module.get(), &module->arena,
                          loom_module_block(module.get()), &builder);
  const loom_resolved_target_t resolved_target = {
      /*.provider=*/selection.provider,
      /*.facts=*/selected_facts,
  };
  IREE_ASSERT_OK(selection.provider->materialize_definition(
      &builder, &resolved_target, symbol, LOOM_LOCATION_UNKNOWN));
  ASSERT_TRUE(
      loom_spirv_target_isa(module->symbols.entries[symbol_id].defining_op));

  loom_symbol_fact_table_reset(&requirement_fact_table_);
  const loom_target_symbol_facts_t* materialized =
      Target(&requirement_fact_table_, module.get(), symbol);
  EXPECT_TRUE(loom_target_facts_are_equivalent(selected_facts,
                                               materialized->projection));

  loom_target_profile_selection_deinitialize(&selection);
}

TEST_F(SpirvProviderTest, RejectsUnknownFamilySelector) {
  loom_target_profile_selection_t selection = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_target_environment_select_profile(
                            &target_environment_, IREE_SV("spirv:opencl3.0"),
                            iree_allocator_system(), &selection));
}

}  // namespace
}  // namespace loom
