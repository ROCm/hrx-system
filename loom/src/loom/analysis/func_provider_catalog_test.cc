// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/func_provider_catalog.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class FuncProviderCatalogTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_func_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_FUNC, vtables, (uint16_t)vtable_count));
    vtables = loom_test_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_symbol_fact_table_initialize(&fact_table_, &analysis_arena_);
    loom_func_provider_catalog_initialize(&catalog_, &analysis_arena_);
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
                                  IREE_SV("func_provider_catalog_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_symbol_id_t FindSymbol(const loom_module_t* module,
                              iree_string_view_t name) {
    loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return symbol_id;
  }

  loom_func_provider_slice_t RebuildAndLookup(const loom_module_t* module,
                                              iree_string_view_t key) {
    loom_symbol_fact_table_reset(&fact_table_);
    IREE_CHECK_OK(loom_func_provider_catalog_build_local(&catalog_, module,
                                                         &fact_table_));
    return loom_func_provider_catalog_lookup_key(&catalog_, key);
  }

  void AddUkernelProvider(loom_module_t* module, iree_string_view_t contract,
                          iree_string_view_t name, int64_t priority) {
    loom_string_id_t contract_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(module, contract, &contract_id));
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(module, name, &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_func_ukernel_build(
        &builder, LOOM_FUNC_UKERNEL_BUILD_FLAG_HAS_PRIORITY, contract_id, 0, 0,
        0, 0, 0, loom_symbol_ref_null(), priority,
        (loom_symbol_ref_t){/*.module_id=*/0, /*.symbol_id=*/symbol_id}, &i32,
        1, &i32, 1, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &op));
    ASSERT_NE(op, nullptr);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  iree_arena_allocator_t analysis_arena_;
  loom_symbol_fact_table_t fact_table_;
  loom_func_provider_catalog_t catalog_;
};

TEST_F(FuncProviderCatalogTest, GroupsProvidersByContractAndPriority) {
  ModulePtr module = ParseModule(R"(
func.template<qwen.q4.matmul> priority(1) @slow(%arg0: i32) -> (i32) {
  func.return %arg0 : i32
}
func.ukernel<qwen.q4.matmul> priority(5) @asm_impl(%arg0: i32) -> (i32)
func.template<other.contract> public priority(3) @other(%arg0: i32) -> (i32) {
  func.return %arg0 : i32
}
func.template<qwen.q4.matmul> public priority(10) @fast(%arg0: i32) -> (i32) {
  func.return %arg0 : i32
}
)");

  loom_func_provider_slice_t qwen =
      RebuildAndLookup(module.get(), IREE_SV("qwen.q4.matmul"));
  ASSERT_EQ(qwen.count, 3u);
  EXPECT_TRUE(iree_string_view_equal(qwen.providers[0].name, IREE_SV("fast")));
  EXPECT_EQ(qwen.providers[0].kind, LOOM_FUNC_PROVIDER_KIND_TEMPLATE);
  EXPECT_TRUE(qwen.providers[0].is_public);
  EXPECT_TRUE(qwen.providers[0].has_body);
  EXPECT_EQ(qwen.providers[0].calling_convention, 0);
  EXPECT_EQ(qwen.providers[0].purity, 0);
  EXPECT_EQ(qwen.providers[0].temperature, 0);
  EXPECT_EQ(qwen.providers[0].inline_policy, 0);
  EXPECT_EQ(qwen.providers[0].priority, 10);
  ASSERT_EQ(qwen.providers[0].argument_count, 1u);
  ASSERT_EQ(qwen.providers[0].result_count, 1u);
  EXPECT_TRUE(loom_type_equal(qwen.providers[0].argument_types[0],
                              loom_type_scalar(LOOM_SCALAR_TYPE_I32)));
  EXPECT_TRUE(loom_type_equal(qwen.providers[0].result_types[0],
                              loom_type_scalar(LOOM_SCALAR_TYPE_I32)));
  EXPECT_TRUE(
      iree_string_view_equal(qwen.providers[1].name, IREE_SV("asm_impl")));
  EXPECT_EQ(qwen.providers[1].kind, LOOM_FUNC_PROVIDER_KIND_UKERNEL);
  EXPECT_FALSE(qwen.providers[1].is_public);
  EXPECT_FALSE(qwen.providers[1].has_body);
  EXPECT_EQ(qwen.providers[1].priority, 5);
  EXPECT_TRUE(iree_string_view_equal(qwen.providers[2].name, IREE_SV("slow")));
  EXPECT_EQ(qwen.providers[2].priority, 1);

  loom_func_provider_slice_t other = loom_func_provider_catalog_lookup_key(
      &catalog_, IREE_SV("other.contract"));
  ASSERT_EQ(other.count, 1u);
  EXPECT_TRUE(
      iree_string_view_equal(other.providers[0].name, IREE_SV("other")));
  EXPECT_TRUE(other.providers[0].is_public);

  loom_func_provider_slice_t missing =
      loom_func_provider_catalog_lookup_key(&catalog_, IREE_SV("missing"));
  EXPECT_EQ(missing.count, 0u);
}

TEST_F(FuncProviderCatalogTest, RebuildDropsErasedProvider) {
  ModulePtr module = ParseModule(R"(
func.template<demo.contract> priority(2) @kept(%arg0: i32) -> (i32) {
  func.return %arg0 : i32
}
func.template<demo.contract> priority(3) @removed(%arg0: i32) -> (i32) {
  func.return %arg0 : i32
}
)");

  loom_func_provider_slice_t initial =
      RebuildAndLookup(module.get(), IREE_SV("demo.contract"));
  ASSERT_EQ(initial.count, 2u);
  EXPECT_TRUE(
      iree_string_view_equal(initial.providers[0].name, IREE_SV("removed")));

  loom_symbol_id_t removed_id = FindSymbol(module.get(), IREE_SV("removed"));
  loom_op_t* removed_op = module->symbols.entries[removed_id].defining_op;
  ASSERT_NE(removed_op, nullptr);
  IREE_ASSERT_OK(loom_op_erase(module.get(), removed_op));

  loom_func_provider_slice_t rebuilt =
      RebuildAndLookup(module.get(), IREE_SV("demo.contract"));
  ASSERT_EQ(rebuilt.count, 1u);
  EXPECT_TRUE(
      iree_string_view_equal(rebuilt.providers[0].name, IREE_SV("kept")));
}

TEST_F(FuncProviderCatalogTest, RebuildSeesAddedProvider) {
  ModulePtr module = ParseModule(R"(
func.template<demo.contract> priority(2) @base(%arg0: i32) -> (i32) {
  func.return %arg0 : i32
}
)");

  loom_func_provider_slice_t initial =
      RebuildAndLookup(module.get(), IREE_SV("demo.contract"));
  ASSERT_EQ(initial.count, 1u);
  EXPECT_TRUE(
      iree_string_view_equal(initial.providers[0].name, IREE_SV("base")));

  AddUkernelProvider(module.get(), IREE_SV("demo.contract"), IREE_SV("added"),
                     9);

  loom_func_provider_slice_t rebuilt =
      RebuildAndLookup(module.get(), IREE_SV("demo.contract"));
  ASSERT_EQ(rebuilt.count, 2u);
  EXPECT_TRUE(
      iree_string_view_equal(rebuilt.providers[0].name, IREE_SV("added")));
  EXPECT_EQ(rebuilt.providers[0].kind, LOOM_FUNC_PROVIDER_KIND_UKERNEL);
  EXPECT_TRUE(
      iree_string_view_equal(rebuilt.providers[1].name, IREE_SV("base")));
}

TEST_F(FuncProviderCatalogTest, CapturesProviderTargetApplicability) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @gfx11

func.template<demo.contract> target(@gfx11) priority(3) @gfx11_provider(%arg0: i32) -> (i32) {
  func.return %arg0 : i32
}

func.template<demo.contract> priority(1) @fallback(%arg0: i32) -> (i32) {
  func.return %arg0 : i32
}
)");

  loom_func_provider_slice_t providers =
      RebuildAndLookup(module.get(), IREE_SV("demo.contract"));
  ASSERT_EQ(providers.count, 2u);
  EXPECT_TRUE(iree_string_view_equal(providers.providers[0].name,
                                     IREE_SV("gfx11_provider")));

  loom_symbol_id_t target_id = FindSymbol(module.get(), IREE_SV("gfx11"));
  EXPECT_TRUE(loom_symbol_ref_is_valid(providers.providers[0].target_symbol));
  EXPECT_EQ(providers.providers[0].target_symbol.module_id, 0);
  EXPECT_EQ(providers.providers[0].target_symbol.symbol_id, target_id);
  EXPECT_FALSE(loom_symbol_ref_is_valid(providers.providers[1].target_symbol));
}

}  // namespace
}  // namespace loom
