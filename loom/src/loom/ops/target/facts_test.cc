// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/target/facts.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/target/ops.h"
#include "loom/target/materialization.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class TargetFactsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_symbol_fact_table_initialize(&fact_table_, &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("target_facts_test.loom"), &context_,
                                  &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t *
                                                              out_count);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  loom_symbol_id_t FindSymbol(const loom_module_t* module,
                              iree_string_view_t name) {
    loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return symbol_id;
  }

  const loom_target_symbol_facts_t* LookupTarget(const loom_module_t* module,
                                                 iree_string_view_t name) {
    const loom_symbol_facts_base_t* base_facts = nullptr;
    IREE_CHECK_OK(loom_symbol_fact_table_lookup(
        &fact_table_, module, FindSymbol(module, name), &base_facts));
    const loom_target_symbol_facts_t* target_facts =
        loom_target_symbol_facts_cast(base_facts);
    IREE_ASSERT(target_facts != nullptr);
    return target_facts;
  }

  // Block pool shared by parser, module allocation, and analysis storage.
  iree_arena_block_pool_t block_pool_;

  // Context with the target dialect registered.
  loom_context_t context_;

  // Arena for symbol fact table storage and fact payloads.
  iree_arena_allocator_t analysis_arena_;

  // Dense symbol fact table under test.
  loom_symbol_fact_table_t fact_table_;
};

TEST_F(TargetFactsTest, ProjectsLaunchBoundsFromGenericTargetRecord) {
  ModulePtr module = ParseModule(R"(
target.generic<reference> @gpu {
  max_workgroup_size_x = 256,
  max_workgroup_size_y = 8,
  max_workgroup_size_z = 4,
  max_flat_workgroup_size = 1024,
  max_workgroup_storage_bytes = 65536,
  subgroup_size = 32,
  max_grid_size_x = 4096,
  max_grid_size_y = 2048,
  max_grid_size_z = 1024,
  max_flat_grid_size = 8589934592
}
)");

  const loom_target_symbol_facts_t* facts =
      LookupTarget(module.get(), IREE_SV("gpu"));
  ASSERT_NE(facts->projection, nullptr);
  EXPECT_TRUE(iree_string_view_equal(facts->projection->fact_type->name,
                                     IREE_SV("target.generic")));
  EXPECT_TRUE(loom_target_facts_attr_is_authored(
      facts->projection, loom_target_generic_subgroup_size_ATTR_INDEX));
  EXPECT_FALSE(loom_target_facts_attr_is_authored(
      facts->projection, loom_target_generic_codegen_format_ATTR_INDEX));
  const loom_target_bundle_storage_t& storage = facts->projection->storage;
  EXPECT_EQ(storage.snapshot.max_workgroup_size.x, 256u);
  EXPECT_EQ(storage.snapshot.max_workgroup_size.y, 8u);
  EXPECT_EQ(storage.snapshot.max_workgroup_size.z, 4u);
  EXPECT_EQ(storage.snapshot.max_flat_workgroup_size, 1024u);
  EXPECT_EQ(storage.snapshot.max_workgroup_storage_bytes, 65536u);
  EXPECT_EQ(storage.snapshot.subgroup_size, 32u);
  EXPECT_EQ(storage.snapshot.max_grid_size.x, 4096u);
  EXPECT_EQ(storage.snapshot.max_grid_size.y, 2048u);
  EXPECT_EQ(storage.snapshot.max_grid_size.z, 1024u);
  EXPECT_EQ(storage.snapshot.max_flat_grid_size, 8589934592ull);
}

TEST_F(TargetFactsTest, StructuralSatisfactionUsesFactsNotSymbolIdentity) {
  ModulePtr module = ParseModule(R"(
target.generic<reference> @effective {
  max_workgroup_size_x = 256
}
target.generic<reference> @equivalent {
  max_workgroup_size_x = 256
}
target.generic<reference> @smaller_requirement {
  max_workgroup_size_x = 128
}
)");

  const loom_target_facts_t* effective =
      LookupTarget(module.get(), IREE_SV("effective"))->projection;
  const loom_target_facts_t* equivalent =
      LookupTarget(module.get(), IREE_SV("equivalent"))->projection;
  const loom_target_facts_t* smaller_requirement =
      LookupTarget(module.get(), IREE_SV("smaller_requirement"))->projection;

  EXPECT_NE(effective, equivalent);
  EXPECT_TRUE(loom_target_facts_satisfy_requirement(effective, equivalent));
  EXPECT_TRUE(
      loom_target_facts_satisfy_requirement(effective, smaller_requirement));
  EXPECT_FALSE(
      loom_target_facts_satisfy_requirement(smaller_requirement, effective));
}

TEST_F(TargetFactsTest, MaterializedProjectionRoundTripsDurableFields) {
  ModulePtr module = ParseModule(R"(
target.generic<reference> @base
)");
  const loom_target_symbol_facts_t* base_facts =
      LookupTarget(module.get(), IREE_SV("base"));
  loom_target_bundle_storage_t selected_storage =
      base_facts->projection->storage;
  loom_target_bundle_storage_rebind(&selected_storage);
  selected_storage.bundle.name = IREE_SV("incidental-profile-name");
  selected_storage.snapshot.codegen_format = LOOM_TARGET_CODEGEN_FORMAT_SPIRV;
  selected_storage.snapshot.artifact_format =
      LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY;
  selected_storage.snapshot.default_pointer_bitwidth = 32;
  selected_storage.snapshot.index_bitwidth = 32;
  selected_storage.snapshot.offset_bitwidth = 32;
  selected_storage.snapshot.max_workgroup_size = {
      /*.x=*/256,
      /*.y=*/8,
      /*.z=*/4,
  };
  selected_storage.snapshot.max_flat_workgroup_size = 512;
  selected_storage.snapshot.max_workgroup_storage_bytes = 131072;
  selected_storage.snapshot.subgroup_size = 32;
  selected_storage.snapshot.max_grid_size = {
      /*.x=*/4096,
      /*.y=*/2048,
      /*.z=*/1024,
  };
  selected_storage.snapshot.max_flat_grid_size = UINT64_C(8589934592);
  selected_storage.snapshot.max_workgroup_count = {
      /*.x=*/65535,
      /*.y=*/32768,
      /*.z=*/16384,
  };
  selected_storage.snapshot.memory_spaces = {
      /*.generic=*/1,
      /*.global=*/2,
      /*.workgroup=*/3,
      /*.constant=*/4,
      /*.private_memory=*/5,
      /*.host=*/6,
      /*.descriptor=*/7,
  };
  selected_storage.export_plan.export_symbol = IREE_SV("materialized_kernel");
  selected_storage.export_plan.abi_kind = LOOM_TARGET_ABI_HAL_KERNEL;
  selected_storage.export_plan.linkage = LOOM_TARGET_LINKAGE_DSO_LOCAL;
  selected_storage.config.contract_set_key = IREE_SV("test.materialized");
  selected_storage.config.contract_feature_bits = UINT64_C(0x1234);

  loom_string_id_t symbol_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(
      module.get(), IREE_SV("materialized"), &symbol_name_id));
  loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(module.get(), symbol_name_id, &symbol_id));
  const loom_symbol_ref_t symbol = {
      /*.module_id=*/0,
      /*.symbol_id=*/symbol_id,
  };

  loom_builder_t builder = {};
  loom_builder_initialize(module.get(), &module->arena,
                          loom_module_block(module.get()), &builder);
  loom_op_t* target_op = nullptr;
  IREE_ASSERT_OK(loom_target_record_projection_build(
      &builder, LOOM_OP_TARGET_GENERIC, LOOM_TARGET_GENERIC_KIND_REFERENCE,
      symbol, &selected_storage.bundle, /*authored_target_op=*/nullptr,
      /*extension_attrs=*/nullptr,
      /*extension_attr_count=*/0, LOOM_LOCATION_UNKNOWN, &target_op));
  ASSERT_NE(target_op, nullptr);
  EXPECT_TRUE(loom_target_record_projection_matches_bundle(
      module.get(), target_op, &selected_storage.bundle,
      /*authored_target_op=*/nullptr));
  const loom_target_like_descriptor_t* descriptor = loom_target_like_descriptor(
      loom_target_like_cast(module.get(), target_op));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->projection_count, 30u);

  const loom_target_symbol_facts_t* facts =
      LookupTarget(module.get(), IREE_SV("materialized"));
  const loom_target_bundle_storage_t& storage = facts->projection->storage;
  EXPECT_EQ(storage.snapshot.codegen_format, LOOM_TARGET_CODEGEN_FORMAT_SPIRV);
  EXPECT_EQ(storage.snapshot.artifact_format,
            LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY);
  EXPECT_EQ(storage.snapshot.default_pointer_bitwidth, 32u);
  EXPECT_EQ(storage.snapshot.index_bitwidth, 32u);
  EXPECT_EQ(storage.snapshot.offset_bitwidth, 32u);
  EXPECT_EQ(storage.snapshot.max_workgroup_size.x, 256u);
  EXPECT_EQ(storage.snapshot.max_workgroup_size.y, 8u);
  EXPECT_EQ(storage.snapshot.max_workgroup_size.z, 4u);
  EXPECT_EQ(storage.snapshot.max_flat_workgroup_size, 512u);
  EXPECT_EQ(storage.snapshot.max_workgroup_storage_bytes, 131072u);
  EXPECT_EQ(storage.snapshot.subgroup_size, 32u);
  EXPECT_EQ(storage.snapshot.max_grid_size.x, 4096u);
  EXPECT_EQ(storage.snapshot.max_grid_size.y, 2048u);
  EXPECT_EQ(storage.snapshot.max_grid_size.z, 1024u);
  EXPECT_EQ(storage.snapshot.max_flat_grid_size, UINT64_C(8589934592));
  EXPECT_EQ(storage.snapshot.max_workgroup_count.x, 65535u);
  EXPECT_EQ(storage.snapshot.max_workgroup_count.y, 32768u);
  EXPECT_EQ(storage.snapshot.max_workgroup_count.z, 16384u);
  EXPECT_EQ(storage.snapshot.memory_spaces.generic, 1u);
  EXPECT_EQ(storage.snapshot.memory_spaces.global, 2u);
  EXPECT_EQ(storage.snapshot.memory_spaces.workgroup, 3u);
  EXPECT_EQ(storage.snapshot.memory_spaces.constant, 4u);
  EXPECT_EQ(storage.snapshot.memory_spaces.private_memory, 5u);
  EXPECT_EQ(storage.snapshot.memory_spaces.host, 6u);
  EXPECT_EQ(storage.snapshot.memory_spaces.descriptor, 7u);
  EXPECT_TRUE(iree_string_view_equal(storage.export_plan.export_symbol,
                                     IREE_SV("materialized_kernel")));
  EXPECT_EQ(storage.export_plan.abi_kind, LOOM_TARGET_ABI_HAL_KERNEL);
  EXPECT_EQ(storage.export_plan.linkage, LOOM_TARGET_LINKAGE_DSO_LOCAL);
  EXPECT_TRUE(iree_string_view_equal(storage.config.contract_set_key,
                                     IREE_SV("test.materialized")));
  EXPECT_EQ(storage.config.contract_feature_bits, UINT64_C(0x1234));
}

TEST_F(TargetFactsTest,
       MaterializedProjectionRefinesProfileWithAuthoredFunctionFacts) {
  ModulePtr module = ParseModule(R"(
target.generic<reference> @profile
target.generic<reference> @authored {
  max_workgroup_size_x = 64,
  abi = hal_kernel,
  export_symbol = "authored_kernel",
  linkage = dso_local
}
)");
  const loom_target_symbol_facts_t* profile_facts =
      LookupTarget(module.get(), IREE_SV("profile"));
  loom_target_bundle_storage_t profile_storage =
      profile_facts->projection->storage;
  loom_target_bundle_storage_rebind(&profile_storage);
  profile_storage.snapshot.max_workgroup_size.x = 256;
  profile_storage.export_plan.abi_kind = LOOM_TARGET_ABI_OBJECT_FUNCTION;
  profile_storage.export_plan.export_symbol = IREE_SV("profile_function");
  profile_storage.export_plan.linkage = LOOM_TARGET_LINKAGE_DEFAULT;

  const loom_symbol_id_t authored_symbol_id =
      FindSymbol(module.get(), IREE_SV("authored"));
  const loom_op_t* authored_target_op =
      module->symbols.entries[authored_symbol_id].defining_op;

  loom_string_id_t refined_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module.get(), IREE_SV("refined"),
                                           &refined_name_id));
  loom_symbol_id_t refined_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module.get(), refined_name_id,
                                        &refined_symbol_id));
  const loom_symbol_ref_t refined_ref = {
      /*.module_id=*/0,
      /*.symbol_id=*/refined_symbol_id,
  };

  loom_builder_t builder = {};
  loom_builder_initialize(module.get(), &module->arena,
                          loom_module_block(module.get()), &builder);
  loom_op_t* refined_target_op = nullptr;
  IREE_ASSERT_OK(loom_target_record_projection_build(
      &builder, LOOM_OP_TARGET_GENERIC, LOOM_TARGET_GENERIC_KIND_REFERENCE,
      refined_ref, &profile_storage.bundle, authored_target_op,
      /*extension_attrs=*/nullptr,
      /*extension_attr_count=*/0, LOOM_LOCATION_UNKNOWN, &refined_target_op));
  ASSERT_NE(refined_target_op, nullptr);
  EXPECT_TRUE(loom_target_record_projection_matches_bundle(
      module.get(), refined_target_op, &profile_storage.bundle,
      authored_target_op));
  EXPECT_FALSE(loom_target_record_projection_matches_bundle(
      module.get(), refined_target_op, &profile_storage.bundle,
      /*authored_target_op=*/nullptr));

  const loom_target_symbol_facts_t* refined_facts =
      LookupTarget(module.get(), IREE_SV("refined"));
  const loom_target_bundle_storage_t& refined_storage =
      refined_facts->projection->storage;
  EXPECT_EQ(refined_storage.snapshot.max_workgroup_size.x, 256u);
  EXPECT_EQ(refined_storage.export_plan.abi_kind, LOOM_TARGET_ABI_HAL_KERNEL);
  EXPECT_TRUE(iree_string_view_equal(refined_storage.export_plan.export_symbol,
                                     IREE_SV("authored_kernel")));
  EXPECT_EQ(refined_storage.export_plan.linkage, LOOM_TARGET_LINKAGE_DSO_LOCAL);
}

TEST_F(TargetFactsTest, InvalidSelectorProducesNoFacts) {
  ModulePtr module = ParseModule(R"(
target.generic<reference> @gpu
)");
  const loom_symbol_id_t symbol_id = FindSymbol(module.get(), IREE_SV("gpu"));
  loom_op_t* target_op = module->symbols.entries[symbol_id].defining_op;
  loom_op_attrs(target_op)[loom_target_generic_kind_ATTR_INDEX] =
      loom_attr_enum(UINT8_MAX);

  const loom_symbol_facts_base_t* facts = nullptr;
  IREE_ASSERT_OK(loom_symbol_fact_table_lookup(&fact_table_, module.get(),
                                               symbol_id, &facts));
  EXPECT_EQ(facts, nullptr);
}

}  // namespace
}  // namespace loom
