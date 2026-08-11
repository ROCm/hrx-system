// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/linker.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/module_ptr.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class GlobalFactsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_GLOBAL, loom_global_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn = decltype(&loom_global_dialect_vtables);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("global_facts_test.loom"), &context_,
                                  &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  ModulePtr LinkModules(const loom_module_t* first_module,
                        const loom_module_t* second_module) {
    const loom_module_t* source_modules[] = {first_module, second_module};
    loom_link_options_t options = {
        /*.module_name=*/IREE_SV("linked"),
    };
    loom_module_t* linked_module = nullptr;
    IREE_CHECK_OK(loom_link_materialized_modules(
        source_modules, IREE_ARRAYSIZE(source_modules), &options, &block_pool_,
        iree_allocator_system(), &linked_module));
    return ModulePtr(linked_module);
  }

  struct LoadFacts {
    // Buffer reference facts produced for the rodata load result.
    loom_value_fact_buffer_reference_t reference;
    // SSA result to which |reference| belongs.
    loom_value_id_t result_id;
  };

  LoadFacts ComputeLoadFacts(const loom_module_t* module,
                             iree_string_view_t function_name) {
    loom_string_id_t name_id = loom_module_lookup_string(module, function_name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
    loom_func_like_t function =
        loom_func_like_cast(module, symbol->defining_op);
    IREE_ASSERT(function.op != nullptr);

    const loom_block_t* entry_block =
        loom_region_entry_block(loom_func_like_body(function));
    const loom_op_t* load_op = loom_block_const_op(entry_block, 0);
    IREE_ASSERT(load_op != nullptr && loom_global_load_isa(load_op));
    loom_value_id_t result_id = loom_global_load_result(load_op).values[0];

    loom_value_fact_table_t fact_table = {};
    IREE_CHECK_OK(loom_value_fact_table_initialize(
        &fact_table, &analysis_arena_, module->values.count));
    IREE_CHECK_OK(loom_value_fact_table_compute(&fact_table, module, function));
    loom_value_facts_t facts =
        loom_value_fact_table_lookup(&fact_table, result_id);
    LoadFacts load_facts = {};
    IREE_ASSERT(loom_value_facts_query_buffer_reference(
        &fact_table.context, facts, &load_facts.reference));
    load_facts.result_id = result_id;
    return load_facts;
  }

  // Block pool shared by parser, module allocation, and analysis storage.
  iree_arena_block_pool_t block_pool_;
  // Arena retaining value-fact extension payloads for each test.
  iree_arena_allocator_t analysis_arena_;
  // Context containing the function and global dialects under test.
  loom_context_t context_;
};

TEST_F(GlobalFactsTest, RodataDefinitionProvidesExactStorageFacts) {
  ModulePtr module = ParseModule(R"(
global.rodata.def @message = align(16) bytes("6c6f6f6d")

test.func @load_message() -> (buffer) {
  %message = global.load @message : buffer
  test.yield %message : buffer
}
)");

  LoadFacts facts = ComputeLoadFacts(module.get(), IREE_SV("load_message"));
  EXPECT_TRUE(loom_value_facts_is_exact(facts.reference.maximum_byte_extent));
  EXPECT_EQ(facts.reference.maximum_byte_extent.range_lo, 4);
  EXPECT_EQ(facts.reference.maximum_byte_extent.range_hi, 4);
  EXPECT_EQ(facts.reference.minimum_alignment, 16u);
  EXPECT_EQ(facts.reference.memory_space,
            LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT);
  EXPECT_EQ(facts.reference.root_value_id, facts.result_id);
  EXPECT_EQ(facts.reference.alias_scope_id,
            LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE);
  EXPECT_EQ(facts.reference.nullability,
            LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NON_NULL);
}

TEST_F(GlobalFactsTest, RodataDeclarationProvidesConservativeStorageFacts) {
  ModulePtr module = ParseModule(R"(
global.rodata.decl @message

test.func @load_message() -> (buffer) {
  %message = global.load @message : buffer
  test.yield %message : buffer
}
)");

  LoadFacts facts = ComputeLoadFacts(module.get(), IREE_SV("load_message"));
  EXPECT_FALSE(loom_value_facts_is_exact(facts.reference.maximum_byte_extent));
  EXPECT_EQ(facts.reference.maximum_byte_extent.range_lo, 0);
  EXPECT_EQ(facts.reference.maximum_byte_extent.range_hi, INT64_MAX);
  EXPECT_EQ(facts.reference.minimum_alignment, 1u);
  EXPECT_EQ(facts.reference.memory_space,
            LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT);
  EXPECT_EQ(facts.reference.root_value_id, facts.result_id);
  EXPECT_EQ(facts.reference.alias_scope_id,
            LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE);
  EXPECT_EQ(facts.reference.nullability,
            LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NON_NULL);
}

TEST_F(GlobalFactsTest, RodataDefinitionRefinesDeclarationAfterLinking) {
  ModulePtr declaration = ParseModule(R"(
global.rodata.decl @message

test.func @load_message() -> (buffer) {
  %message = global.load @message : buffer
  test.yield %message : buffer
}
)");
  ModulePtr definition = ParseModule(R"(
global.rodata.def @message = align(32) bytes("6c6f6f6d")
)");
  ModulePtr linked = LinkModules(declaration.get(), definition.get());

  LoadFacts facts = ComputeLoadFacts(linked.get(), IREE_SV("load_message"));
  EXPECT_TRUE(loom_value_facts_is_exact(facts.reference.maximum_byte_extent));
  EXPECT_EQ(facts.reference.maximum_byte_extent.range_lo, 4);
  EXPECT_EQ(facts.reference.maximum_byte_extent.range_hi, 4);
  EXPECT_EQ(facts.reference.minimum_alignment, 32u);
  EXPECT_EQ(facts.reference.memory_space,
            LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT);
  EXPECT_EQ(facts.reference.root_value_id, facts.result_id);
  EXPECT_EQ(facts.reference.alias_scope_id,
            LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE);
  EXPECT_EQ(facts.reference.nullability,
            LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NON_NULL);
}

}  // namespace
}  // namespace loom
