// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/analysis/template_provider_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/template/ops.h"
#include "loom/testing/module_ptr.h"
#include "loom/transforms/symbol/template_selection.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class TemplateSelectionQueryTest : public ::testing::Test {
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
    vtables = loom_template_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEMPLATE, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("template_selection_query_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  std::string PrintModule(const loom_module_t* module) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_EXPECT_OK(loom_text_print_module_to_builder(module, &builder,
                                                     LOOM_TEXT_PRINT_DEFAULT));
    std::string text(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
    iree_string_builder_deinitialize(&builder);
    return text;
  }

  loom_symbol_id_t FindSymbol(const loom_module_t* module,
                              iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return symbol_id;
  }

  loom_template_provider_summary_t ExternalizeProvider(
      loom_module_t* module, iree_string_view_t family_name,
      iree_string_view_t provider_name, iree_host_size_t origin_ordinal) {
    loom_symbol_fact_table_t fact_table = {};
    loom_symbol_fact_table_initialize(&fact_table, &arena_);
    loom_template_provider_catalog_t local_catalog = {};
    loom_template_provider_catalog_initialize(&local_catalog, &arena_);
    IREE_CHECK_OK(loom_template_provider_catalog_build_local(
        &local_catalog, module, &fact_table));
    const loom_symbol_ref_t family = {
        /*.module_id=*/0,
        /*.symbol_id=*/FindSymbol(module, family_name),
    };
    const loom_template_provider_slice_t providers =
        loom_template_provider_catalog_lookup(&local_catalog, family);
    const loom_template_provider_summary_t* selected = nullptr;
    for (iree_host_size_t i = 0; i < providers.count; ++i) {
      if (iree_string_view_equal(providers.providers[i].name, provider_name)) {
        selected = &providers.providers[i];
        break;
      }
    }
    IREE_ASSERT(selected != nullptr);
    loom_template_provider_summary_t external = *selected;
    external.symbol = loom_symbol_ref_null();
    external.function = {};
    external.func_facts = nullptr;
    external.origin_ordinal = origin_ordinal;

    const loom_symbol_id_t provider_symbol_id =
        FindSymbol(module, provider_name);
    IREE_CHECK_OK(loom_op_erase(
        module, module->symbols.entries[provider_symbol_id].defining_op));
    return external;
  }

  loom_template_selection_query_result_t Query(
      loom_module_t* module,
      const loom_template_provider_summary_t* external_providers,
      iree_host_size_t external_provider_count, iree_host_size_t origin_count,
      loom_template_selection_mode_t mode =
          LOOM_TEMPLATE_SELECTION_MODE_EARLY) {
    loom_symbol_fact_table_t fact_table = {};
    loom_symbol_fact_table_initialize(&fact_table, &arena_);
    loom_template_provider_catalog_t catalog = {};
    loom_template_provider_catalog_initialize(&catalog, &arena_);
    IREE_CHECK_OK(loom_template_provider_catalog_build(
        &catalog, module, &fact_table, external_providers,
        external_provider_count));
    loom_template_selection_query_result_t result = {};
    const loom_template_selection_query_options_t query_options = {
        /*.mode=*/mode,
        /*.catalog=*/&catalog,
        /*.function_versions=*/nullptr,
        /*.origin_count=*/origin_count,
    };
    IREE_CHECK_OK(loom_template_selection_query(
        module, &query_options, &block_pool_, &arena_, &result));
    return result;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  iree_arena_allocator_t arena_;
};

TEST_F(TemplateSelectionQueryTest, SelectsExternalProviderWithoutMutation) {
  ModulePtr module = ParseModule(R"(
template.decl @demo.family(%x: i32) -> (i32)

func.def public @entry(%x: i32) -> (i32) {
  %a = template.apply<@demo.family>(%x) : (i32) -> (i32)
  %b = template.apply<@demo.family>(%a) : (i32) -> (i32)
  func.return %b : i32
}

template.def<@demo.family> @external(%x: i32) -> (i32) {
  template.return %x : i32
}
)");
  const loom_template_provider_summary_t external = ExternalizeProvider(
      module.get(), IREE_SV("demo.family"), IREE_SV("external"), 7);
  const std::string module_before = PrintModule(module.get());

  const loom_template_selection_query_result_t result =
      Query(module.get(), &external, 1, /*origin_count=*/8);

  ASSERT_EQ(result.selected_origins.count, 1u);
  EXPECT_EQ(result.selected_origins.values[0], 7u);
  EXPECT_EQ(result.unresolved_site_count, 0u);
  EXPECT_EQ(PrintModule(module.get()), module_before);
}

TEST_F(TemplateSelectionQueryTest,
       SelectsBorrowedProviderAcrossModuleValueDomains) {
  ModulePtr source = ParseModule(R"(
template.decl @source.family(%m: index, %arg: tensor<[%m]xf32>) -> (tensor<[%m]xf32>)

template.def<@source.family> @source.provider(%m: index, %arg: tensor<[%m]xf32>) -> (tensor<[%m]xf32>) where [mul(%m, 16)] {
  template.return %arg : tensor<[%m]xf32>
}
)");
  ModulePtr application = ParseModule(R"(
template.decl @application.family(%m: index, %arg: tensor<[%m]xf32>) -> (tensor<[%m]xf32>)

func.def @padding(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def public @entry(%m: index, %arg: tensor<[%m]xf32>) -> (tensor<[%m]xf32>) where [range(%m, 32, 32)] {
  %result = template.apply<@application.family>(%m, %arg) : (index, tensor<[%m]xf32>) -> (tensor<[%m]xf32>)
  func.return %result : tensor<[%m]xf32>
}
)");
  loom_template_provider_summary_t external = ExternalizeProvider(
      source.get(), IREE_SV("source.family"), IREE_SV("source.provider"), 11);
  ASSERT_TRUE(external.signature_is_module_independent);
  external.family = {
      /*.module_id=*/0,
      /*.symbol_id=*/
      FindSymbol(application.get(), IREE_SV("application.family")),
  };
  const loom_symbol_t* family_symbol =
      &application->symbols.entries[external.family.symbol_id];
  external.family_name = application->strings.entries[family_symbol->name_id];
  const std::string module_before = PrintModule(application.get());

  const loom_template_selection_query_result_t result =
      Query(application.get(), &external, 1, /*origin_count=*/12);

  ASSERT_EQ(result.selected_origins.count, 1u);
  EXPECT_EQ(result.selected_origins.values[0], 11u);
  EXPECT_EQ(result.unresolved_site_count, 0u);
  EXPECT_EQ(PrintModule(application.get()), module_before);
}

TEST_F(TemplateSelectionQueryTest,
       FamilyPredicateGatesExternalProviderSelection) {
  ModulePtr rejected_module = ParseModule(R"(
template.decl @demo.family(%m: index) -> (index) where [mul(%m, 16)]

func.def public @entry(%m: index) -> (index) where [range(%m, 15, 15)] {
  %result = template.apply<@demo.family>(%m) : (index) -> (index)
  func.return %result : index
}

template.def<@demo.family> @external(%m: index) -> (index) {
  template.return %m : index
}
)");
  const loom_template_provider_summary_t rejected_provider =
      ExternalizeProvider(rejected_module.get(), IREE_SV("demo.family"),
                          IREE_SV("external"), 3);

  const loom_template_selection_query_result_t rejected_result =
      Query(rejected_module.get(), &rejected_provider, 1,
            /*origin_count=*/4);

  EXPECT_EQ(rejected_result.selected_origins.count, 0u);
  EXPECT_EQ(rejected_result.unresolved_site_count, 1u);

  ModulePtr matched_module = ParseModule(R"(
template.decl @demo.family(%m: index) -> (index) where [mul(%m, 16)]

func.def public @entry(%m: index) -> (index) where [range(%m, 32, 32)] {
  %result = template.apply<@demo.family>(%m) : (index) -> (index)
  func.return %result : index
}

template.def<@demo.family> @external(%m: index) -> (index) {
  template.return %m : index
}
)");
  const loom_template_provider_summary_t matched_provider = ExternalizeProvider(
      matched_module.get(), IREE_SV("demo.family"), IREE_SV("external"), 3);

  const loom_template_selection_query_result_t matched_result =
      Query(matched_module.get(), &matched_provider, 1,
            /*origin_count=*/4);

  ASSERT_EQ(matched_result.selected_origins.count, 1u);
  EXPECT_EQ(matched_result.selected_origins.values[0], 3u);
  EXPECT_EQ(matched_result.unresolved_site_count, 0u);
}

TEST_F(TemplateSelectionQueryTest, ExactCallChecksFamilyAndProviderContracts) {
  ModulePtr family_rejected_module = ParseModule(R"(
template.decl @demo.family(%m: index) -> (index) where [mul(%m, 16)]

template.def<@demo.family> @implementation(%m: index) -> (index) {
  template.return %m : index
}

func.def public @entry(%m: index) -> (index) where [range(%m, 15, 15)] {
  %result = template.call @implementation(%m) : (index) -> (index)
  func.return %result : index
}
)");
  const loom_template_selection_query_result_t family_rejected_result =
      Query(family_rejected_module.get(), nullptr, 0, /*origin_count=*/0);
  EXPECT_EQ(family_rejected_result.unresolved_site_count, 1u);

  ModulePtr provider_rejected_module = ParseModule(R"(
template.decl @demo.family(%m: index) -> (index)

template.def<@demo.family> @implementation(%m: index) -> (index) where [mul(%m, 16)] {
  template.return %m : index
}

func.def public @entry(%m: index) -> (index) where [range(%m, 15, 15)] {
  %result = template.call @implementation(%m) : (index) -> (index)
  func.return %result : index
}
)");
  const loom_template_selection_query_result_t provider_rejected_result =
      Query(provider_rejected_module.get(), nullptr, 0, /*origin_count=*/0);
  EXPECT_EQ(provider_rejected_result.unresolved_site_count, 1u);

  ModulePtr matched_module = ParseModule(R"(
template.decl @demo.family(%m: index) -> (index) where [mul(%m, 16)]

template.def<@demo.family> @implementation(%m: index) -> (index) where [range(%m, 32, 32)] {
  template.return %m : index
}

func.def public @entry(%m: index) -> (index) where [range(%m, 32, 32)] {
  %result = template.call @implementation(%m) : (index) -> (index)
  func.return %result : index
}
)");
  const loom_template_selection_query_result_t matched_result =
      Query(matched_module.get(), nullptr, 0, /*origin_count=*/0);
  EXPECT_EQ(matched_result.unresolved_site_count, 0u);
}

TEST_F(TemplateSelectionQueryTest,
       LocalProviderBodyExposesNestedExternalDemand) {
  ModulePtr module = ParseModule(R"(
template.decl @demo.outer(%x: i32) -> (i32)
template.decl @demo.inner(%x: i32) -> (i32)

func.def public @entry(%x: i32) -> (i32) {
  %result = template.apply<@demo.outer>(%x) : (i32) -> (i32)
  func.return %result : i32
}

template.def<@demo.outer> @outer(%x: i32) -> (i32) {
  %result = template.apply<@demo.inner>(%x) : (i32) -> (i32)
  template.return %result : i32
}

template.def<@demo.inner> @inner(%x: i32) -> (i32) {
  template.return %x : i32
}
)");
  const loom_template_provider_summary_t inner = ExternalizeProvider(
      module.get(), IREE_SV("demo.inner"), IREE_SV("inner"), 11);

  const loom_template_selection_query_result_t result =
      Query(module.get(), &inner, 1, /*origin_count=*/12);

  ASSERT_EQ(result.selected_origins.count, 1u);
  EXPECT_EQ(result.selected_origins.values[0], 11u);
  EXPECT_EQ(result.unresolved_site_count, 0u);
}

TEST_F(TemplateSelectionQueryTest, ReportsUnresolvedApplications) {
  ModulePtr module = ParseModule(R"(
template.decl @demo.missing(%x: i32) -> (i32)

func.def public @entry(%x: i32) -> (i32) {
  %result = template.apply<@demo.missing>(%x) : (i32) -> (i32)
  func.return %result : i32
}
)");

  const loom_template_selection_query_result_t result =
      Query(module.get(), nullptr, 0, /*origin_count=*/0);
  EXPECT_EQ(result.selected_origins.count, 0u);
  EXPECT_EQ(result.unresolved_site_count, 1u);

  loom_symbol_fact_table_t fact_table = {};
  loom_symbol_fact_table_initialize(&fact_table, &arena_);
  loom_template_provider_catalog_t catalog = {};
  loom_template_provider_catalog_initialize(&catalog, &arena_);
  IREE_ASSERT_OK(loom_template_provider_catalog_build_local(
      &catalog, module.get(), &fact_table));
  loom_template_selection_query_result_t final_result = {};
  const loom_template_selection_query_options_t query_options = {
      /*.mode=*/LOOM_TEMPLATE_SELECTION_MODE_FINAL,
      /*.catalog=*/&catalog,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_template_selection_query(module.get(), &query_options, &block_pool_,
                                    &arena_, &final_result));
}

}  // namespace
}  // namespace loom
