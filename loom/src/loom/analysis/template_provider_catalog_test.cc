// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/template_provider_catalog.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/template/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class TemplateProviderCatalogTest : public ::testing::Test {
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
    vtables = loom_test_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)vtable_count));
    vtables = loom_target_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TARGET, vtables, (uint16_t)vtable_count));
    iree_host_size_t parameterized_attr_count = 0;
    const loom_parameterized_attr_descriptor_t* parameterized_attrs =
        loom_target_dialect_parameterized_attrs(&parameterized_attr_count);
    IREE_ASSERT_OK(loom_context_register_parameterized_attrs(
        &context_, LOOM_DIALECT_TARGET, parameterized_attrs,
        parameterized_attr_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_symbol_fact_table_initialize(&fact_table_, &analysis_arena_);
    loom_template_provider_catalog_initialize(&catalog_, &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(
        loom_text_parse(iree_make_cstring_view(source),
                        IREE_SV("template_provider_catalog_test.loom"),
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

  loom_template_provider_slice_t RebuildAndLookup(const loom_module_t* module,
                                                  iree_string_view_t name) {
    loom_symbol_fact_table_reset(&fact_table_);
    IREE_CHECK_OK(loom_template_provider_catalog_build_local(&catalog_, module,
                                                             &fact_table_));
    return loom_template_provider_catalog_lookup(
        &catalog_, {.module_id = 0, .symbol_id = FindSymbol(module, name)});
  }

  const loom_template_provider_summary_t* FindProvider(
      loom_template_provider_slice_t providers, iree_string_view_t name) {
    for (iree_host_size_t i = 0; i < providers.count; ++i) {
      if (iree_string_view_equal(providers.providers[i].name, name)) {
        return &providers.providers[i];
      }
    }
    return nullptr;
  }

  void AddUkernelProvider(loom_module_t* module, iree_string_view_t family,
                          iree_string_view_t name, int64_t priority) {
    const loom_symbol_ref_t family_ref = {
        .module_id = 0,
        .symbol_id = FindSymbol(module, family),
    };
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(module, name, &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_template_ukernel_build(
        &builder, LOOM_TEMPLATE_UKERNEL_BUILD_FLAG_HAS_PRIORITY, family_ref,
        /*visibility=*/0, /*retain=*/0, /*cc=*/0, /*effect_contract=*/0,
        /*temperature=*/0, loom_symbol_ref_null(),
        loom_parameterized_attr_array_empty(), priority,
        loom_symbol_ref_t{.module_id = 0, .symbol_id = symbol_id}, &i32, 1,
        &i32, 1, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &op));
    ASSERT_NE(op, nullptr);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  iree_arena_allocator_t analysis_arena_;
  loom_symbol_fact_table_t fact_table_;
  loom_template_provider_catalog_t catalog_;
};

TEST_F(TemplateProviderCatalogTest, GroupsProvidersByFamily) {
  ModulePtr module = ParseModule(R"(
template.decl @qwen.q4.matmul(%arg0: i32) -> (i32)
template.decl @other.contract(%arg0: i32) -> (i32)

template.def<@qwen.q4.matmul> priority(1) @slow(%arg0: i32) -> (i32) {
  template.return %arg0 : i32
}
template.ukernel<@qwen.q4.matmul> priority(5) @asm_impl(%arg0: i32) -> (i32)
template.def<@other.contract> public priority(3) @other(%arg0: i32) -> (i32) {
  template.return %arg0 : i32
}
template.def<@qwen.q4.matmul> public priority(10) @fast(%arg0: i32) -> (i32) {
  template.return %arg0 : i32
}
)");

  loom_template_provider_slice_t qwen =
      RebuildAndLookup(module.get(), IREE_SV("qwen.q4.matmul"));
  ASSERT_EQ(qwen.count, 3u);
  const loom_template_provider_summary_t* fast =
      FindProvider(qwen, IREE_SV("fast"));
  ASSERT_NE(fast, nullptr);
  EXPECT_EQ(fast->kind, LOOM_TEMPLATE_PROVIDER_KIND_DEF);
  EXPECT_TRUE(fast->has_body);
  EXPECT_EQ(fast->priority, 10);
  ASSERT_EQ(fast->argument_count, 1u);
  ASSERT_EQ(fast->result_count, 1u);
  EXPECT_TRUE(loom_type_equal(
      loom_module_value_type(fast->module, fast->argument_ids[0]),
      loom_type_scalar(LOOM_SCALAR_TYPE_I32)));
  EXPECT_TRUE(
      loom_type_equal(loom_module_value_type(fast->module, fast->result_ids[0]),
                      loom_type_scalar(LOOM_SCALAR_TYPE_I32)));

  const loom_template_provider_summary_t* asm_impl =
      FindProvider(qwen, IREE_SV("asm_impl"));
  ASSERT_NE(asm_impl, nullptr);
  EXPECT_EQ(asm_impl->kind, LOOM_TEMPLATE_PROVIDER_KIND_UKERNEL);
  EXPECT_FALSE(asm_impl->has_body);
  EXPECT_EQ(asm_impl->priority, 5);

  const loom_template_provider_summary_t* slow =
      FindProvider(qwen, IREE_SV("slow"));
  ASSERT_NE(slow, nullptr);
  EXPECT_EQ(slow->priority, 1);

  loom_template_provider_slice_t other = loom_template_provider_catalog_lookup(
      &catalog_,
      {
          .module_id = 0,
          .symbol_id = FindSymbol(module.get(), IREE_SV("other.contract")),
      });
  ASSERT_EQ(other.count, 1u);
  EXPECT_TRUE(
      iree_string_view_equal(other.providers[0].name, IREE_SV("other")));

  loom_template_provider_slice_t missing =
      loom_template_provider_catalog_lookup(&catalog_, loom_symbol_ref_null());
  EXPECT_EQ(missing.count, 0u);
}

TEST_F(TemplateProviderCatalogTest,
       PreservesExternalGroupsAndMixedSourceOrder) {
  ModulePtr module = ParseModule(R"(
template.decl @a(%value: i32) -> (i32)
template.decl @b(%value: i32) -> (i32)
)");
  const loom_symbol_ref_t families[] = {
      {.module_id = 0, .symbol_id = FindSymbol(module.get(), IREE_SV("a"))},
      {.module_id = 0, .symbol_id = FindSymbol(module.get(), IREE_SV("b"))},
  };
  const char* orders[] = {"aabb", "bbaa", "abab", "baba", "abba", "baab"};
  for (bool has_local : {false, true}) {
    SCOPED_TRACE(has_local);
    if (has_local) {
      AddUkernelProvider(module.get(), IREE_SV("a"), IREE_SV("local.a"), 3);
      AddUkernelProvider(module.get(), IREE_SV("b"), IREE_SV("local.b"), 7);
    }
    for (size_t order_index = 0; order_index < IREE_ARRAYSIZE(orders);
         ++order_index) {
      const char* order = orders[order_index];
      SCOPED_TRACE(order);
      loom_template_provider_summary_t external[4] = {};
      for (size_t i = 0; i < IREE_ARRAYSIZE(external); ++i) {
        const loom_template_provider_contract_t contract = {
            .kind = LOOM_TEMPLATE_PROVIDER_KIND_UKERNEL,
            .argument_count = 1,
            .result_count = 1,
            .name = IREE_SV("external"),
            .priority = (int64_t)i,
        };
        IREE_ASSERT_OK(loom_template_provider_contract_bind_family(
            &contract, module.get(), families[order[i] - 'a'],
            loom_symbol_ref_null(), i, &external[i]));
      }
      IREE_ASSERT_OK(loom_template_provider_catalog_build(
          &catalog_, module.get(), &fact_table_, external,
          IREE_ARRAYSIZE(external)));
      EXPECT_EQ(catalog_.provider_count, has_local ? 6u : 4u);
      if (!has_local && order_index < 2) {
        // Both ascending and descending family order already form a catalog.
        EXPECT_EQ(catalog_.providers, external);
      } else {
        EXPECT_NE(catalog_.providers, external);
      }
      for (size_t family_index = 0; family_index < IREE_ARRAYSIZE(families);
           ++family_index) {
        const auto providers = loom_template_provider_catalog_lookup(
            &catalog_, families[family_index]);
        ASSERT_EQ(providers.count, has_local ? 3u : 2u);
        size_t position = 0;
        if (has_local) {
          EXPECT_EQ(providers.providers[position++].origin_ordinal,
                    IREE_HOST_SIZE_MAX);
        }
        for (size_t i = 0; i < IREE_ARRAYSIZE(external); ++i) {
          if (order[i] == 'a' + family_index) {
            const auto& provider = providers.providers[position++];
            EXPECT_EQ(provider.origin_ordinal, i);
            EXPECT_EQ(provider.priority, (int64_t)i);
            EXPECT_EQ(provider.family.symbol_id,
                      families[family_index].symbol_id);
          }
        }
      }
    }
  }
}

TEST_F(TemplateProviderCatalogTest, EmptyExternalCatalogRetainsNoProviders) {
  ModulePtr module = ParseModule(R"(
template.decl @empty(%value: i32) -> (i32)
)");
  IREE_ASSERT_OK(loom_template_provider_catalog_build(
      &catalog_, module.get(), &fact_table_, nullptr, 0));
  EXPECT_EQ(catalog_.provider_count, 0u);
  EXPECT_EQ(catalog_.providers, nullptr);
  const auto providers = loom_template_provider_catalog_lookup(
      &catalog_, {
                     .module_id = 0,
                     .symbol_id = FindSymbol(module.get(), IREE_SV("empty")),
                 });
  EXPECT_EQ(providers.count, 0u);
}

TEST_F(TemplateProviderCatalogTest, RebuildDropsErasedProvider) {
  ModulePtr module = ParseModule(R"(
template.decl @demo.contract(%arg0: i32) -> (i32)

template.def<@demo.contract> priority(2) @kept(%arg0: i32) -> (i32) {
  template.return %arg0 : i32
}
template.def<@demo.contract> priority(3) @removed(%arg0: i32) -> (i32) {
  template.return %arg0 : i32
}
)");

  loom_template_provider_slice_t initial =
      RebuildAndLookup(module.get(), IREE_SV("demo.contract"));
  ASSERT_EQ(initial.count, 2u);
  EXPECT_NE(FindProvider(initial, IREE_SV("removed")), nullptr);

  loom_symbol_id_t removed_id = FindSymbol(module.get(), IREE_SV("removed"));
  loom_op_t* removed_op = module->symbols.entries[removed_id].defining_op;
  ASSERT_NE(removed_op, nullptr);
  IREE_ASSERT_OK(loom_op_erase(module.get(), removed_op));

  loom_template_provider_slice_t rebuilt =
      RebuildAndLookup(module.get(), IREE_SV("demo.contract"));
  ASSERT_EQ(rebuilt.count, 1u);
  EXPECT_TRUE(
      iree_string_view_equal(rebuilt.providers[0].name, IREE_SV("kept")));
}

TEST_F(TemplateProviderCatalogTest, RebuildSeesAddedProvider) {
  ModulePtr module = ParseModule(R"(
template.decl @demo.contract(%arg0: i32) -> (i32)

template.def<@demo.contract> priority(2) @base(%arg0: i32) -> (i32) {
  template.return %arg0 : i32
}
)");

  loom_template_provider_slice_t initial =
      RebuildAndLookup(module.get(), IREE_SV("demo.contract"));
  ASSERT_EQ(initial.count, 1u);
  EXPECT_TRUE(
      iree_string_view_equal(initial.providers[0].name, IREE_SV("base")));

  AddUkernelProvider(module.get(), IREE_SV("demo.contract"), IREE_SV("added"),
                     9);

  loom_template_provider_slice_t rebuilt =
      RebuildAndLookup(module.get(), IREE_SV("demo.contract"));
  ASSERT_EQ(rebuilt.count, 2u);
  const loom_template_provider_summary_t* added =
      FindProvider(rebuilt, IREE_SV("added"));
  ASSERT_NE(added, nullptr);
  EXPECT_EQ(added->kind, LOOM_TEMPLATE_PROVIDER_KIND_UKERNEL);
  EXPECT_NE(FindProvider(rebuilt, IREE_SV("base")), nullptr);
}

TEST_F(TemplateProviderCatalogTest, CapturesProviderIdentityAndConditions) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @gfx11
template.decl @demo.contract(%arg0: i32) -> (i32)

// Both constraints are deliberate here: the catalog must preserve target
// identity and normalized facts independently when a provider requires both.
template.def<@demo.contract> target(@gfx11) requires [#target.subgroup.size<64>] priority(3) @gfx11_provider(%arg0: i32) -> (i32) {
  template.return %arg0 : i32
}

template.def<@demo.contract> priority(1) @fallback(%arg0: i32) -> (i32) {
  template.return %arg0 : i32
}
)");

  loom_template_provider_slice_t providers =
      RebuildAndLookup(module.get(), IREE_SV("demo.contract"));
  ASSERT_EQ(providers.count, 2u);
  EXPECT_TRUE(iree_string_view_equal(providers.providers[0].name,
                                     IREE_SV("gfx11_provider")));

  loom_symbol_id_t target_id = FindSymbol(module.get(), IREE_SV("gfx11"));
  EXPECT_TRUE(loom_symbol_ref_is_valid(providers.providers[0].target_symbol));
  EXPECT_EQ(providers.providers[0].target_symbol.module_id, 0);
  EXPECT_EQ(providers.providers[0].target_symbol.symbol_id, target_id);
  ASSERT_EQ(providers.providers[0].target_condition_count, 1);
  ASSERT_NE(providers.providers[0].target_conditions, nullptr);
  EXPECT_EQ(providers.providers[0].target_conditions[0].descriptor,
            &loom_target_subgroup_size_condition);
  EXPECT_EQ(loom_target_subgroup_size_attr_size(
                providers.providers[0].target_conditions[0].value),
            64);
  EXPECT_FALSE(loom_symbol_ref_is_valid(providers.providers[1].target_symbol));
  EXPECT_EQ(providers.providers[1].target_condition_count, 0);
}

TEST_F(TemplateProviderCatalogTest,
       BindsProviderWithoutCopyingSourcePredicates) {
  ModulePtr source = ParseModule(R"(
test.target<low_core> @source.gfx11
template.decl @source.family(%m: index, %arg: tensor<[%m]xf32>) -> (tensor<[%m]xf32>)

template.def<@source.family> target(@source.gfx11) requires [#target.subgroup.size<32>] priority(7) @source.provider(%m: index, %arg: tensor<[%m]xf32>) -> (tensor<[%m]xf32>) where [multiple_of(%m, 16)] {
  template.return %arg : tensor<[%m]xf32>
}
)");
  ModulePtr target = ParseModule(R"(
test.target<low_core> @target.gfx11
template.decl @target.padding(%a: i32, %b: i32, %c: i32) -> (i32)
template.decl @target.family(%n: index, %arg: tensor<[%n]xf32>) -> (tensor<[%n]xf32>)
)");

  loom_template_provider_slice_t source_providers =
      RebuildAndLookup(source.get(), IREE_SV("source.family"));
  ASSERT_EQ(source_providers.count, 1u);
  const loom_template_provider_summary_t& source_provider =
      source_providers.providers[0];

  const loom_symbol_ref_t target_target = {
      .module_id = 0,
      .symbol_id = FindSymbol(target.get(), IREE_SV("target.gfx11")),
  };
  const loom_symbol_ref_t target_family = {
      .module_id = 0,
      .symbol_id = FindSymbol(target.get(), IREE_SV("target.family")),
  };
  const iree_host_size_t target_value_count_before = target->values.count;
  loom_template_provider_summary_t bound = {};
  IREE_ASSERT_OK(loom_template_provider_summary_bind_family(
      &source_provider, target.get(), target_family, target_target,
      /*origin_ordinal=*/123, &bound));

  EXPECT_EQ(target->values.count, target_value_count_before);
  EXPECT_EQ(bound.module, target.get());
  EXPECT_FALSE(loom_symbol_ref_is_valid(bound.symbol));
  EXPECT_EQ(bound.origin_ordinal, 123u);
  EXPECT_EQ(bound.family.symbol_id, target_family.symbol_id);
  EXPECT_EQ(bound.target_symbol.symbol_id, target_target.symbol_id);
  EXPECT_EQ(bound.target_facts, source_provider.target_facts);

  ASSERT_EQ(bound.argument_count, 2u);
  ASSERT_EQ(bound.result_count, 1u);
  EXPECT_FALSE(bound.predicates_use_ordinals);
  EXPECT_EQ(bound.argument_ids, source_provider.argument_ids);
  EXPECT_EQ(bound.result_ids, source_provider.result_ids);
  EXPECT_EQ(bound.predicates, source_provider.predicates);

  ASSERT_EQ(bound.predicate_count, 1u);
  ASSERT_NE(bound.predicates, nullptr);
  EXPECT_EQ(bound.predicates[0].kind, LOOM_PREDICATE_MULTIPLE_OF);
  EXPECT_EQ(bound.predicates[0].arg_tags[0], LOOM_PRED_ARG_VALUE);
  EXPECT_EQ((loom_value_id_t)bound.predicates[0].args[0],
            bound.argument_ids[0]);
  EXPECT_EQ(bound.predicates[0].arg_tags[1], LOOM_PRED_ARG_CONST);
  EXPECT_EQ(bound.predicates[0].args[1], 16);

  ASSERT_EQ(bound.target_condition_count, 1u);
  ASSERT_NE(bound.target_conditions, nullptr);
  EXPECT_EQ(bound.target_conditions[0].descriptor,
            &loom_target_subgroup_size_condition);
  EXPECT_EQ(
      loom_target_subgroup_size_attr_size(bound.target_conditions[0].value),
      32);
}

}  // namespace
}  // namespace loom
