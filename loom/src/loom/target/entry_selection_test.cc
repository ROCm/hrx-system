// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/entry_selection.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/test/ops.h"
#include "loom/target/facts_builder.h"
#include "loom/target/function_contract.h"
#include "loom/target/test/target_records.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

static bool AcceptEntry(void* user_data, const loom_target_entry_t* entry) {
  (void)user_data;
  (void)entry;
  return true;
}

class TargetEntrySelectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(loom_dialect_id_t dialect_id, DialectVtablesFn fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, dialect_id, vtables, static_cast<uint16_t>(vtable_count)));
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("target_entry_selection_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_symbol_id_t FindSymbol(const loom_module_t* module,
                              iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return symbol_id;
  }

  loom_func_like_t FindFunction(loom_module_t* module,
                                iree_string_view_t name) {
    loom_op_t* op =
        module->symbols.entries[FindSymbol(module, name)].defining_op;
    IREE_ASSERT(op != nullptr);
    loom_func_like_t function = loom_func_like_cast(module, op);
    IREE_ASSERT(loom_func_like_isa(function));
    return function;
  }

  const loom_func_symbol_facts_t* LookupFunctionFacts(
      const loom_module_t* module, iree_string_view_t name,
      loom_symbol_fact_table_t* symbol_facts) {
    const loom_symbol_facts_base_t* base_facts = nullptr;
    IREE_CHECK_OK(loom_symbol_fact_table_lookup(
        symbol_facts, module, FindSymbol(module, name), &base_facts));
    const loom_func_symbol_facts_t* function_facts =
        loom_func_symbol_facts_cast(base_facts);
    IREE_ASSERT(function_facts != nullptr);
    return function_facts;
  }

  const loom_target_facts_t* RefineFunctionFacts(
      const loom_module_t* module,
      const loom_func_symbol_facts_t* function_facts,
      loom_test_target_kind_t target_kind) {
    loom_target_facts_t base_facts = {};
    loom_target_facts_builder_initialize(
        &loom_test_target_fact_type,
        loom_target_bundle_table_lookup(&loom_test_target_bundles, target_kind),
        &base_facts);
    bool valid = false;
    const loom_target_facts_t* function_target_facts = nullptr;
    IREE_CHECK_OK(loom_target_function_contract_refine_facts(
        module, function_facts, IREE_SV("invocation-exact"), &base_facts,
        iree_diagnostic_emitter_t{}, &analysis_arena_, &valid,
        &function_target_facts));
    IREE_ASSERT(valid);
    IREE_ASSERT(function_target_facts != nullptr);
    return function_target_facts;
  }

  loom_target_entry_t SelectNamedEntry(
      loom_module_t* module, iree_string_view_t function_name,
      const loom_function_version_list_t* function_versions) {
    loom_target_entry_options_t options = {};
    options.entry_symbol = function_name;
    options.function_versions = function_versions;
    loom_target_entry_diagnostic_emitter_t diagnostic_emitter = {};
    loom_target_entry_diagnostic_emitter_initialize(
        module, &options, LOOM_EMITTER_VERIFIER, &diagnostic_emitter);
    const loom_target_entry_predicate_t predicate = {
        /*.fn=*/AcceptEntry,
    };
    bool selected = false;
    loom_target_entry_t entry = {};
    IREE_CHECK_OK(loom_target_entry_select_entry(
        module, &options, predicate, &diagnostic_emitter, IREE_SV("test"),
        &analysis_arena_, &selected, &entry));
    IREE_ASSERT(selected);
    return entry;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  iree_arena_allocator_t analysis_arena_;
};

TEST_F(TargetEntrySelectionTest, RefinedVersionOverridesAuthoredTarget) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @generic

func.def public target(@generic) @entry() {
  func.return
}
)");

  loom_symbol_fact_table_t symbol_facts = {};
  loom_symbol_fact_table_initialize(&symbol_facts, &analysis_arena_);
  const loom_func_symbol_facts_t* function_facts =
      LookupFunctionFacts(module.get(), IREE_SV("entry"), &symbol_facts);
  const loom_target_facts_t* function_target_facts = RefineFunctionFacts(
      module.get(), function_facts, LOOM_TEST_TARGET_KIND_QUIRKY);

  loom_target_function_version_t function_version = {};
  function_version.base.type = &loom_target_function_version_type;
  function_version.base.function = FindFunction(module.get(), IREE_SV("entry"));
  function_version.function_target_facts = function_target_facts;
  loom_function_version_t* version_values[] = {
      &function_version.base,
  };
  const loom_function_version_list_t function_versions = {
      /*.values=*/version_values,
      /*.count=*/IREE_ARRAYSIZE(version_values),
  };

  const loom_target_entry_t entry =
      SelectNamedEntry(module.get(), IREE_SV("entry"), &function_versions);
  EXPECT_EQ(entry.function_version, &function_version);
  EXPECT_EQ(entry.target_facts, function_target_facts);
  const loom_target_bundle_t* bundle = loom_target_entry_bundle(&entry);
  EXPECT_EQ(bundle, loom_target_facts_bundle(function_target_facts));
  EXPECT_EQ(bundle->snapshot->subgroup_size, 7u);
  EXPECT_TRUE(iree_string_view_equal(bundle->name, IREE_SV("test-quirky")));
}

TEST_F(TargetEntrySelectionTest, RefinedVersionSelectsTargetlessFunction) {
  ModulePtr module = ParseModule(R"(
func.def public @targetless() {
  func.return
}
)");

  loom_symbol_fact_table_t symbol_facts = {};
  loom_symbol_fact_table_initialize(&symbol_facts, &analysis_arena_);
  const loom_func_symbol_facts_t* function_facts =
      LookupFunctionFacts(module.get(), IREE_SV("targetless"), &symbol_facts);
  const loom_target_facts_t* function_target_facts = RefineFunctionFacts(
      module.get(), function_facts, LOOM_TEST_TARGET_KIND_LOW_CORE);

  loom_target_function_version_t function_version = {};
  function_version.base.type = &loom_target_function_version_type;
  function_version.base.function =
      FindFunction(module.get(), IREE_SV("targetless"));
  function_version.function_target_facts = function_target_facts;
  loom_function_version_t* version_values[] = {
      &function_version.base,
  };
  const loom_function_version_list_t function_versions = {
      /*.values=*/version_values,
      /*.count=*/IREE_ARRAYSIZE(version_values),
  };

  const loom_target_entry_t entry =
      SelectNamedEntry(module.get(), IREE_SV("targetless"), &function_versions);
  EXPECT_EQ(entry.function_version, &function_version);
  EXPECT_EQ(entry.target_facts, function_target_facts);
  EXPECT_TRUE(iree_string_view_equal(
      loom_target_entry_bundle(&entry)->export_plan->name,
      IREE_SV("targetless")));
}

}  // namespace
}  // namespace loom
