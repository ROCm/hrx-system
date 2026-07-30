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
    loom_symbol_fact_table_initialize(&effective_fact_table_, &analysis_arena_);
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

  loom_target_record_view_t Target(loom_symbol_fact_table_t* fact_table,
                                   const loom_module_t* module,
                                   loom_symbol_ref_t target_ref) {
    const loom_symbol_facts_base_t* base_facts = nullptr;
    IREE_CHECK_OK(loom_symbol_fact_table_lookup_ref(fact_table, module,
                                                    target_ref, &base_facts));
    const loom_target_symbol_facts_t* target_facts =
        loom_target_symbol_facts_cast(base_facts);
    IREE_ASSERT(target_facts != nullptr);
    return loom_target_record_view_make(module, target_facts);
  }

  loom_target_record_view_t Requirement(const loom_module_t* module,
                                        iree_string_view_t name) {
    return Target(&requirement_fact_table_, module,
                  FindSymbolRef(module, name));
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

  // Dense target fact index for the materialized effective target.
  loom_symbol_fact_table_t effective_fact_table_;
};

TEST_F(SpirvProviderTest, MaterializedProfileSatisfiesStructuredRequirements) {
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
  ModulePtr effective_module = Parse(IREE_SV(""));

  loom_target_bundle_storage_t live_storage = {
      /*.snapshot=*/*loom_spirv_low_target_bundle_vulkan1_3.snapshot,
      /*.export_plan=*/*loom_spirv_low_target_bundle_vulkan1_3.export_plan,
      /*.config=*/*loom_spirv_low_target_bundle_vulkan1_3.config,
      /*.bundle=*/loom_spirv_low_target_bundle_vulkan1_3,
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
  live_storage.export_plan.abi_kind = LOOM_TARGET_ABI_HAL_KERNEL;
  live_storage.config.contract_feature_bits |= LOOM_SPIRV_FEATURE_FLOAT16;

  loom_spirv_target_profile_t profile = {};
  loom_spirv_target_profile_initialize(
      &live_storage.bundle, /*cooperative_properties=*/nullptr, &profile);
  loom_symbol_ref_t effective_ref = loom_symbol_ref_null();
  IREE_ASSERT_OK(loom_target_environment_materialize_effective_target(
      &target_environment_, effective_module.get(), &profile.base,
      /*authored_target_op=*/nullptr, &effective_ref));
  const loom_target_record_view_t effective =
      Target(&effective_fact_table_, effective_module.get(), effective_ref);
  ASSERT_TRUE(loom_target_record_view_is_valid(effective));
  EXPECT_EQ(loom_spirv_target_abi(effective.facts->target.op),
            LOOM_TARGET_ABI_HAL_KERNEL);

  const loom_target_record_view_t baseline_a =
      Requirement(requirements.get(), IREE_SV("baseline_a"));
  const loom_target_record_view_t baseline_b =
      Requirement(requirements.get(), IREE_SV("baseline_b"));
  const loom_op_t* baseline_op = baseline_a.facts->target.op;
  EXPECT_TRUE(loom_target_satisfies_requirement(&target_environment_, effective,
                                                baseline_a));
  EXPECT_TRUE(loom_target_satisfies_requirement(&target_environment_, effective,
                                                baseline_b));
  EXPECT_TRUE(loom_target_satisfies_requirement(
      &target_environment_, effective,
      Requirement(requirements.get(), IREE_SV("fitting"))));
  EXPECT_FALSE(loom_target_satisfies_requirement(
      &target_environment_, effective,
      Requirement(requirements.get(), IREE_SV("too_large"))));
  EXPECT_FALSE(loom_target_satisfies_requirement(
      &target_environment_, effective,
      Requirement(requirements.get(), IREE_SV("subgroup64"))));
  EXPECT_FALSE(loom_target_satisfies_requirement(
      &target_environment_, effective,
      Requirement(requirements.get(), IREE_SV("index64"))));
  EXPECT_TRUE(loom_target_satisfies_requirement(
      &target_environment_, effective,
      Requirement(requirements.get(), IREE_SV("float16"))));
  EXPECT_FALSE(loom_target_satisfies_requirement(
      &target_environment_, effective,
      Requirement(requirements.get(), IREE_SV("float64"))));

  EXPECT_EQ(baseline_a.facts->target.op, baseline_op);
  EXPECT_EQ(loom_spirv_target_abi(baseline_op), LOOM_TARGET_ABI_UNKNOWN);
}

TEST_F(SpirvProviderTest, MaterializesByDurableProjectionIdentity) {
  ModulePtr module = Parse(IREE_SV(""));

  loom_target_bundle_storage_t first_storage = {
      /*.snapshot=*/*loom_spirv_low_target_bundle_vulkan1_3.snapshot,
      /*.export_plan=*/*loom_spirv_low_target_bundle_vulkan1_3.export_plan,
      /*.config=*/*loom_spirv_low_target_bundle_vulkan1_3.config,
      /*.bundle=*/loom_spirv_low_target_bundle_vulkan1_3,
  };
  loom_target_bundle_storage_rebind(&first_storage);
  first_storage.bundle.name = iree_string_view_empty();
  first_storage.snapshot.subgroup_size = 32;

  loom_spirv_cooperative_property_set_t first_inventory = {};
  loom_spirv_target_profile_t first_profile = {};
  loom_spirv_target_profile_initialize(&first_storage.bundle, &first_inventory,
                                       &first_profile);
  loom_symbol_ref_t first_ref = loom_symbol_ref_null();
  IREE_ASSERT_OK(loom_target_environment_materialize_effective_target(
      &target_environment_, module.get(), &first_profile.base,
      /*authored_target_op=*/nullptr, &first_ref));
  EXPECT_TRUE(iree_string_view_equal(
      module->strings
          .entries[module->symbols.entries[first_ref.symbol_id].name_id],
      IREE_SV("spirv_vulkan1_3")));

  loom_target_bundle_storage_t equal_storage = first_storage;
  loom_target_bundle_storage_rebind(&equal_storage);
  equal_storage.bundle.name = IREE_SV("different-diagnostic-name");
  loom_spirv_cooperative_property_set_t different_inventory = {
      /*.feature_bits=*/LOOM_SPIRV_FEATURE_FLOAT16,
  };
  loom_spirv_target_profile_t equal_profile = {};
  loom_spirv_target_profile_initialize(&equal_storage.bundle,
                                       &different_inventory, &equal_profile);
  loom_symbol_ref_t equal_ref = loom_symbol_ref_null();
  IREE_ASSERT_OK(loom_target_environment_materialize_effective_target(
      &target_environment_, module.get(), &equal_profile.base,
      /*authored_target_op=*/nullptr, &equal_ref));
  EXPECT_EQ(equal_ref.module_id, first_ref.module_id);
  EXPECT_EQ(equal_ref.symbol_id, first_ref.symbol_id);

  loom_target_bundle_storage_t distinct_storage = first_storage;
  loom_target_bundle_storage_rebind(&distinct_storage);
  distinct_storage.bundle.name = equal_storage.bundle.name;
  distinct_storage.snapshot.subgroup_size = 64;
  loom_spirv_target_profile_t distinct_profile = {};
  loom_spirv_target_profile_initialize(&distinct_storage.bundle,
                                       &first_inventory, &distinct_profile);
  loom_symbol_ref_t distinct_ref = loom_symbol_ref_null();
  IREE_ASSERT_OK(loom_target_environment_materialize_effective_target(
      &target_environment_, module.get(), &distinct_profile.base,
      /*authored_target_op=*/nullptr, &distinct_ref));
  EXPECT_NE(distinct_ref.symbol_id, first_ref.symbol_id);

  loom_target_bundle_storage_t distinct_copy = distinct_storage;
  loom_target_bundle_storage_rebind(&distinct_copy);
  loom_spirv_target_profile_t distinct_copy_profile = {};
  loom_spirv_target_profile_initialize(&distinct_copy.bundle,
                                       /*cooperative_properties=*/nullptr,
                                       &distinct_copy_profile);
  loom_symbol_ref_t distinct_copy_ref = loom_symbol_ref_null();
  IREE_ASSERT_OK(loom_target_environment_materialize_effective_target(
      &target_environment_, module.get(), &distinct_copy_profile.base,
      /*authored_target_op=*/nullptr, &distinct_copy_ref));
  EXPECT_EQ(distinct_copy_ref.symbol_id, distinct_ref.symbol_id);

  const loom_target_record_view_t first_target =
      Target(&effective_fact_table_, module.get(), first_ref);
  const loom_target_record_view_t distinct_target =
      Target(&effective_fact_table_, module.get(), distinct_ref);
  EXPECT_EQ(first_target.facts->storage.snapshot.subgroup_size, 32u);
  EXPECT_EQ(distinct_target.facts->storage.snapshot.subgroup_size, 64u);
}

}  // namespace
}  // namespace loom
