// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/func_symbol_facts.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/types.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class FuncSymbolFactsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_KERNEL, loom_kernel_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_symbol_fact_table_initialize(&fact_table_, &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t *
                                                              out_count);

  void RegisterDialect(loom_dialect_id_t dialect_id, DialectVtablesFn fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("func_symbol_facts_test.loom"),
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

  const loom_func_symbol_facts_t* LookupFunc(const loom_module_t* module,
                                             iree_string_view_t name) {
    const loom_symbol_facts_base_t* base_facts = nullptr;
    IREE_CHECK_OK(loom_symbol_fact_table_lookup(
        &fact_table_, module, FindSymbol(module, name), &base_facts));
    const loom_func_symbol_facts_t* facts =
        loom_func_symbol_facts_cast(base_facts);
    IREE_ASSERT(facts != nullptr);
    return facts;
  }

  // Block pool shared by parser, module allocation, and analysis storage.
  iree_arena_block_pool_t block_pool_;

  // Context with the func-like dialects and synthetic test target dialect.
  loom_context_t context_;

  // Arena for symbol fact table storage and fact payloads.
  iree_arena_allocator_t analysis_arena_;

  // Dense symbol fact table under test.
  loom_symbol_fact_table_t fact_table_;
};

TEST_F(FuncSymbolFactsTest, SourceFuncFactsRemainTargetIndependent) {
  ModulePtr module = ParseModule(R"(
func.def public device @semantic() {
  func.return
}
)");

  const loom_func_symbol_facts_t* facts =
      LookupFunc(module.get(), IREE_SV("semantic"));
  EXPECT_EQ(facts->base.domain, &loom_func_symbol_fact_domain);
  EXPECT_EQ(facts->base.symbol_kind, LOOM_SYMBOL_FUNC_DEF);
  EXPECT_TRUE(facts->has_body);
  EXPECT_EQ(facts->visibility, 1);
  EXPECT_EQ(facts->calling_convention, 2);
  EXPECT_EQ(facts->purity, 0);
  EXPECT_EQ(facts->temperature, 0);
  EXPECT_EQ(facts->inline_policy, 0);
  EXPECT_EQ(facts->implements_id, LOOM_STRING_ID_INVALID);
  EXPECT_TRUE(iree_string_view_is_empty(facts->implements));
  EXPECT_EQ(facts->priority, 0);
  EXPECT_EQ(facts->predicate_count, 0);
  EXPECT_EQ(facts->argument_count, 0);
  EXPECT_EQ(facts->result_count, 0);
  EXPECT_FALSE(loom_symbol_ref_is_valid(facts->target_symbol));
  EXPECT_FALSE(facts->has_abi);
  EXPECT_FALSE(facts->exports);
}

TEST_F(FuncSymbolFactsTest, TemplateFactsCarryProviderContract) {
  ModulePtr module = ParseModule(R"(
func.template<qwen.q4.matmul> public device pure hot inline priority(7) @impl(%m: index) -> (index) where [mul(%m, 16)] {
  func.return %m : index
}
)");

  const loom_func_symbol_facts_t* facts =
      LookupFunc(module.get(), IREE_SV("impl"));
  EXPECT_EQ(facts->base.symbol_kind, LOOM_SYMBOL_FUNC_TEMPLATE);
  EXPECT_TRUE(facts->has_body);
  EXPECT_EQ(facts->visibility, 1);
  EXPECT_EQ(facts->calling_convention, 2);
  EXPECT_EQ(facts->purity, 1);
  EXPECT_EQ(facts->temperature, 1);
  EXPECT_EQ(facts->inline_policy, 1);
  EXPECT_TRUE(
      iree_string_view_equal(facts->implements, IREE_SV("qwen.q4.matmul")));
  EXPECT_EQ(facts->priority, 7);
  EXPECT_EQ(facts->argument_count, 1);
  EXPECT_EQ(facts->result_count, 1);
  EXPECT_EQ(facts->predicate_count, 1);
}

TEST_F(FuncSymbolFactsTest, DeclarationFactsCarryImportContract) {
  ModulePtr module = ParseModule(R"(
func.decl import("env", "do.work") @do_work(%arg0: i32) -> (i32)
)");

  const loom_func_symbol_facts_t* facts =
      LookupFunc(module.get(), IREE_SV("do_work"));
  EXPECT_FALSE(facts->has_body);
  EXPECT_TRUE(facts->imports);
  EXPECT_TRUE(iree_string_view_equal(facts->import_module, IREE_SV("env")));
  EXPECT_TRUE(iree_string_view_equal(facts->import_symbol, IREE_SV("do.work")));
  EXPECT_EQ(facts->argument_count, 1);
  EXPECT_EQ(facts->result_count, 1);
}

TEST_F(FuncSymbolFactsTest, ImportSymbolDefaultsToDeclarationName) {
  ModulePtr module = ParseModule(R"(
func.decl import("env") @do_work(%arg0: i32) -> (i32)
)");

  const loom_func_symbol_facts_t* facts =
      LookupFunc(module.get(), IREE_SV("do_work"));
  EXPECT_TRUE(facts->imports);
  EXPECT_TRUE(iree_string_view_equal(facts->import_module, IREE_SV("env")));
  EXPECT_TRUE(iree_string_view_equal(facts->import_symbol, IREE_SV("do_work")));
}

TEST_F(FuncSymbolFactsTest, KernelDefFactsCarryExportAndAbiContract) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @target

kernel.def target(@target) export("dispatch") linkage(dso_local) @kernel() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch() {
  kernel.return
}
)");

  const loom_func_symbol_facts_t* facts =
      LookupFunc(module.get(), IREE_SV("kernel"));
  EXPECT_TRUE(iree_string_view_equal(facts->name, IREE_SV("kernel")));
  EXPECT_TRUE(loom_symbol_ref_is_valid(facts->target_symbol));
  EXPECT_TRUE(facts->has_abi);
  EXPECT_EQ(facts->abi_kind, LOOM_TARGET_ABI_HAL_KERNEL);
  EXPECT_EQ(facts->abi_attrs.count, 0u);
  EXPECT_TRUE(facts->exports);
  EXPECT_TRUE(
      iree_string_view_equal(facts->export_symbol, IREE_SV("dispatch")));
  EXPECT_TRUE(facts->has_export_linkage);
  EXPECT_EQ(facts->export_linkage, LOOM_TARGET_LINKAGE_DSO_LOCAL);
}

TEST_F(FuncSymbolFactsTest, KernelDefFactsDefaultToExportedSymbolName) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @target

kernel.def target(@target) @kernel() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch() {
  kernel.return
}
)");

  const loom_func_symbol_facts_t* facts =
      LookupFunc(module.get(), IREE_SV("kernel"));
  EXPECT_TRUE(facts->has_abi);
  EXPECT_EQ(facts->abi_kind, LOOM_TARGET_ABI_HAL_KERNEL);
  EXPECT_TRUE(facts->exports);
  EXPECT_TRUE(iree_string_view_is_empty(facts->export_symbol));
  EXPECT_FALSE(facts->has_export_linkage);
}

TEST_F(FuncSymbolFactsTest, LowKernelDefFactsDefaultToExportedSymbolName) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @target

low.kernel.def target(@target) workgroup_size(1, 1, 1) @kernel() {
  low.return
}
)");

  const loom_func_symbol_facts_t* facts =
      LookupFunc(module.get(), IREE_SV("kernel"));
  EXPECT_TRUE(facts->has_abi);
  EXPECT_EQ(facts->abi_kind, LOOM_TARGET_ABI_HAL_KERNEL);
  EXPECT_TRUE(facts->exports);
  EXPECT_TRUE(iree_string_view_is_empty(facts->export_symbol));
  EXPECT_FALSE(facts->has_export_linkage);
}

TEST_F(FuncSymbolFactsTest, TargetlessAbiFactsRemainTargetIndependent) {
  ModulePtr module = ParseModule(R"(
func.def abi(object_function) @semantic() {
  func.return
}
)");

  const loom_func_symbol_facts_t* facts =
      LookupFunc(module.get(), IREE_SV("semantic"));
  EXPECT_TRUE(facts->has_abi);
  EXPECT_EQ(facts->abi_kind, LOOM_TARGET_ABI_OBJECT_FUNCTION);
  EXPECT_FALSE(loom_symbol_ref_is_valid(facts->target_symbol));
  EXPECT_FALSE(facts->exports);
}

}  // namespace
}  // namespace loom
