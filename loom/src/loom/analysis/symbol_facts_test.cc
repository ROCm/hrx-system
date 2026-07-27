// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbol_facts.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/facts.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class SymbolFactsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
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

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("symbol_facts_test.loom"), &context_,
                                  &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

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

  const loom_test_record_symbol_facts_t* LookupRecord(
      const loom_module_t* module, iree_string_view_t name) {
    const loom_symbol_facts_base_t* base_facts = nullptr;
    IREE_CHECK_OK(loom_symbol_fact_table_lookup(
        &fact_table_, module, FindSymbol(module, name), &base_facts));
    const loom_test_record_symbol_facts_t* record_facts =
        loom_test_record_symbol_facts_cast(base_facts);
    IREE_ASSERT(record_facts != nullptr);
    return record_facts;
  }

  // Block pool shared by parser, module allocation, and analysis storage.
  iree_arena_block_pool_t block_pool_;

  // Context with the test dialect registered.
  loom_context_t context_;

  // Arena for symbol fact table storage and fact payloads.
  iree_arena_allocator_t analysis_arena_;

  // Dense symbol fact table under test.
  loom_symbol_fact_table_t fact_table_;
};

TEST_F(SymbolFactsTest, DescriptorDomainComputesAndCachesRecordFacts) {
  ModulePtr module = ParseModule(R"(
test.record @target {arch = "gfx1100", lanes = 64}
)");

  const loom_test_record_symbol_facts_t* facts =
      LookupRecord(module.get(), IREE_SV("target"));
  EXPECT_EQ(facts->base.domain, &loom_test_record_symbol_fact_domain);
  EXPECT_EQ(facts->base.symbol_kind, LOOM_SYMBOL_RECORD);
  EXPECT_EQ(facts->symbol.symbol_id,
            FindSymbol(module.get(), IREE_SV("target")));
  ASSERT_NE(facts->arch_id, LOOM_STRING_ID_INVALID);
  EXPECT_TRUE(iree_string_view_equal(module->strings.entries[facts->arch_id],
                                     IREE_SV("gfx1100")));
  EXPECT_EQ(facts->lanes, 64);
  EXPECT_FALSE(loom_symbol_ref_is_valid(facts->dependency_symbol));
  EXPECT_EQ(facts->dependency_facts, nullptr);

  const loom_symbol_facts_base_t* cached_facts = nullptr;
  IREE_EXPECT_OK(loom_symbol_fact_table_lookup(
      &fact_table_, module.get(), FindSymbol(module.get(), IREE_SV("target")),
      &cached_facts));
  EXPECT_EQ(cached_facts, &facts->base);
}

TEST_F(SymbolFactsTest, SymbolsWithoutDomainsReturnNullFacts) {
  ModulePtr module = ParseModule(R"(
test.func @ordinary() {
  test.yield
}
)");

  const loom_symbol_facts_base_t* facts = nullptr;
  IREE_EXPECT_OK(loom_symbol_fact_table_lookup(
      &fact_table_, module.get(), FindSymbol(module.get(), IREE_SV("ordinary")),
      &facts));
  EXPECT_EQ(facts, nullptr);
}

TEST_F(SymbolFactsTest, DomainsCanQueryOtherSymbols) {
  ModulePtr module = ParseModule(R"(
test.record @base {lanes = 32}
test.record @derived {depends = @base, lanes = 64}
)");

  const loom_test_record_symbol_facts_t* derived =
      LookupRecord(module.get(), IREE_SV("derived"));
  ASSERT_TRUE(loom_symbol_ref_is_valid(derived->dependency_symbol));
  const loom_test_record_symbol_facts_t* dependency =
      loom_test_record_symbol_facts_cast(derived->dependency_facts);
  ASSERT_NE(dependency, nullptr);
  EXPECT_EQ(dependency->symbol.symbol_id,
            FindSymbol(module.get(), IREE_SV("base")));
  EXPECT_EQ(dependency->lanes, 32);
}

TEST_F(SymbolFactsTest, RecursiveDomainsReportCycles) {
  ModulePtr module = ParseModule(R"(
test.record @a {depends = @b}
test.record @b {depends = @a}
)");

  const loom_symbol_facts_base_t* facts = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_symbol_fact_table_lookup(
                            &fact_table_, module.get(),
                            FindSymbol(module.get(), IREE_SV("a")), &facts));
  EXPECT_EQ(facts, nullptr);
}

TEST_F(SymbolFactsTest, ResetInvalidatesCachedFacts) {
  ModulePtr module = ParseModule(R"(
test.record @target {lanes = 64}
)");
  const loom_test_record_symbol_facts_t* first =
      LookupRecord(module.get(), IREE_SV("target"));
  EXPECT_EQ(first->lanes, 64);

  loom_symbol_id_t symbol_id = FindSymbol(module.get(), IREE_SV("target"));
  loom_op_t* record_op = module->symbols.entries[symbol_id].defining_op;
  loom_string_id_t lanes_name_id =
      loom_module_lookup_string(module.get(), IREE_SV("lanes"));
  ASSERT_NE(lanes_name_id, LOOM_STRING_ID_INVALID);
  loom_named_attr_t entry = {
      /*.name_id=*/lanes_name_id,
      /*.reserved=*/0,
      /*.value=*/loom_attr_i64(128),
  };
  loom_attribute_t dict = {};
  IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
      module.get(), loom_make_named_attr_slice(&entry, 1), &dict));
  loom_op_attrs(record_op)[loom_test_record_dict_ATTR_INDEX] = dict;

  loom_symbol_fact_table_reset(&fact_table_);
  const loom_test_record_symbol_facts_t* second =
      LookupRecord(module.get(), IREE_SV("target"));
  EXPECT_NE(second, first);
  EXPECT_EQ(second->lanes, 128);
}

TEST_F(SymbolFactsTest, TablesAreAttachedToOneModuleUntilReset) {
  ModulePtr first_module = ParseModule(R"(
test.record @first {lanes = 1}
)");
  ModulePtr second_module = ParseModule(R"(
test.record @second {lanes = 2}
)");

  const loom_symbol_facts_base_t* facts = nullptr;
  IREE_ASSERT_OK(loom_symbol_fact_table_lookup(
      &fact_table_, first_module.get(),
      FindSymbol(first_module.get(), IREE_SV("first")), &facts));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_symbol_fact_table_lookup(
          &fact_table_, second_module.get(),
          FindSymbol(second_module.get(), IREE_SV("second")), &facts));

  loom_symbol_fact_table_reset(&fact_table_);
  IREE_EXPECT_OK(loom_symbol_fact_table_lookup(
      &fact_table_, second_module.get(),
      FindSymbol(second_module.get(), IREE_SV("second")), &facts));
  ASSERT_NE(facts, nullptr);
  const loom_test_record_symbol_facts_t* second_facts =
      loom_test_record_symbol_facts_cast(facts);
  ASSERT_NE(second_facts, nullptr);
  EXPECT_EQ(second_facts->lanes, 2);
}

TEST_F(SymbolFactsTest, DomainsCanUseInjectedResources) {
  const loom_test_record_symbol_fact_resource_t resource = {
      /*.lane_bias=*/8,
  };
  const loom_symbol_fact_resource_t resources[] = {{
      /*.key=*/&loom_test_record_symbol_fact_resource_key,
      /*.value=*/&resource,
  }};
  const loom_symbol_fact_table_options_t options = {
      /*.resources=*/loom_make_symbol_fact_resource_list(
          resources, IREE_ARRAYSIZE(resources)),
  };
  loom_symbol_fact_table_initialize_with_options(&fact_table_, &options,
                                                 &analysis_arena_);

  ModulePtr module = ParseModule(R"(
test.record @target {lanes = 64, use_resource = true}
)");
  const loom_test_record_symbol_facts_t* facts =
      LookupRecord(module.get(), IREE_SV("target"));
  EXPECT_EQ(facts->lane_bias, 8);
  EXPECT_EQ(facts->lanes, 72);
}

TEST_F(SymbolFactsTest, MissingInjectedResourceFailsLoudly) {
  ModulePtr module = ParseModule(R"(
test.record @target {lanes = 64, use_resource = true}
)");

  const loom_symbol_facts_base_t* facts = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_symbol_fact_table_lookup(&fact_table_, module.get(),
                                    FindSymbol(module.get(), IREE_SV("target")),
                                    &facts));
  EXPECT_EQ(facts, nullptr);
}

}  // namespace
}  // namespace loom
