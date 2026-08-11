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

TEST(TargetFactRelationTest, DistinctFactsRequireFamilyIdentityRelation) {
  static const loom_target_fact_type_t kFactType = {
      /*.name=*/IREE_SVL("no-identity"),
      /*.storage_size=*/sizeof(loom_target_facts_t),
  };
  const loom_target_facts_t lhs = {
      /*.fact_type=*/&kFactType,
      /*.selector=*/7,
  };
  const loom_target_facts_t rhs = {
      /*.fact_type=*/&kFactType,
      /*.selector=*/7,
  };

  EXPECT_TRUE(loom_target_facts_satisfy_identity_requirement(&lhs, &lhs));
  EXPECT_FALSE(loom_target_facts_satisfy_identity_requirement(&lhs, &rhs));
}

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
                                     IREE_SV("target")));
  EXPECT_TRUE(loom_target_facts_field_is_explicit(
      facts->projection, LOOM_TARGET_FACT_FIELD_SUBGROUP_SIZE));
  EXPECT_FALSE(loom_target_facts_field_is_explicit(
      facts->projection, LOOM_TARGET_FACT_FIELD_CODEGEN_FORMAT));
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

TEST_F(TargetFactsTest,
       GeneratedTargetSeparatesSelectorIdentityFromStructuralSpecialization) {
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
  EXPECT_TRUE(loom_target_facts_satisfy_specialization_requirement(effective,
                                                                   equivalent));
  EXPECT_TRUE(loom_target_facts_satisfy_specialization_requirement(
      effective, smaller_requirement));
  EXPECT_FALSE(loom_target_facts_satisfy_specialization_requirement(
      smaller_requirement, effective));

  EXPECT_TRUE(
      loom_target_facts_satisfy_identity_requirement(effective, effective));
  EXPECT_TRUE(
      loom_target_facts_satisfy_identity_requirement(effective, equivalent));
  EXPECT_TRUE(loom_target_facts_satisfy_identity_requirement(
      smaller_requirement, effective));
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
