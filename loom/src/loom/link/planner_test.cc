// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/planner.h"

#include <memory>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/stream.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/testdata/planner_testdata.h"
#include "loom/ops/check/ops.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/template/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"

namespace loom {
namespace {

struct IndexDeleter {
  void operator()(loom_link_module_index_t* index) const {
    loom_link_module_index_free(index);
  }
};
using IndexPtr = std::unique_ptr<loom_link_module_index_t, IndexDeleter>;

struct PlanDeleter {
  void operator()(loom_link_plan_t* plan) const { loom_link_plan_free(plan); }
};
using PlanPtr = std::unique_ptr<loom_link_plan_t, PlanDeleter>;

std::string StringViewToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

class LinkPlannerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CHECK, loom_check_dialect_vtables,
                    loom_check_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_COMMAND, loom_command_dialect_vtables,
                    loom_command_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_CONFIG, loom_config_dialect_vtables,
                    loom_config_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables,
                    loom_func_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_KERNEL, loom_kernel_dialect_vtables,
                    loom_kernel_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables,
                    loom_target_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_TEMPLATE, loom_template_dialect_vtables,
                    loom_template_dialect_op_semantics);
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    for (loom_module_t* module : modules_) {
      loom_module_free(module);
    }
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);
  using DialectSemanticsFn = const loom_op_semantics_t* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id, DialectVtablesFn dialect_vtables_fn,
                       DialectSemanticsFn dialect_semantics_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
    iree_host_size_t semantics_count = 0;
    const loom_op_semantics_t* semantics =
        dialect_semantics_fn(&semantics_count);
    IREE_ASSERT_OK(loom_context_register_dialect_semantics(
        &context_, dialect_id, semantics, (uint16_t)semantics_count));
  }

  loom_module_t* Parse(iree_string_view_t source,
                       iree_string_view_t filename = IREE_SV("test.loom")) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t parse_options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    IREE_EXPECT_OK(loom_text_parse(source, filename, &context_, &block_pool_,
                                   &parse_options, &module));
    EXPECT_NE(module, nullptr);
    if (module) {
      modules_.push_back(module);
    }
    return module;
  }

  iree_string_view_t Fixture(iree_string_view_t name) {
    const iree_file_toc_t* files = loom_link_planner_testdata_create();
    for (iree_host_size_t i = 0; i < loom_link_planner_testdata_size(); ++i) {
      const iree_string_view_t file_name =
          iree_make_cstring_view(files[i].name);
      if (iree_string_view_equal(file_name, name)) {
        return iree_make_string_view(files[i].data, files[i].size);
      }
    }
    ADD_FAILURE() << "planner fixture '" << std::string(name.data, name.size)
                  << "' was not embedded";
    return iree_string_view_empty();
  }

  IndexPtr CreateIndex() {
    loom_link_module_index_t* index = nullptr;
    IREE_CHECK_OK(loom_link_module_index_allocate(
        &context_, &block_pool_, iree_allocator_system(), &index));
    return IndexPtr(index);
  }

  IndexPtr CreateOverlay(const loom_link_module_index_t* base_index) {
    loom_link_module_index_t* index = nullptr;
    IREE_CHECK_OK(loom_link_module_index_allocate_overlay(
        base_index, &block_pool_, iree_allocator_system(), &index));
    return IndexPtr(index);
  }

  std::vector<uint8_t> WriteModule(const loom_module_t* module) {
    iree_io_stream_t* stream = nullptr;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    IREE_CHECK_OK(loom_bytecode_write_module(module, stream,
                                             /*options=*/nullptr,
                                             &block_pool_));

    iree_io_stream_pos_t length = iree_io_stream_length(stream);
    std::vector<uint8_t> bytes(length);
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }

  iree_host_size_t AddMaterialized(loom_link_module_index_t* index,
                                   const loom_module_t* module,
                                   iree_string_view_t name,
                                   loom_link_provider_role_t role) {
    loom_link_module_index_add_options_t options = {
        /*.provider_name=*/name,
        /*.role=*/role,
    };
    iree_host_size_t provider_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    IREE_CHECK_OK(loom_link_module_index_add_materialized(
        index, module, &options, &provider_ordinal));
    return provider_ordinal;
  }

  iree_host_size_t AddBytecode(loom_link_module_index_t* index,
                               const std::vector<uint8_t>& bytes,
                               iree_string_view_t name,
                               loom_link_provider_role_t role) {
    loom_link_module_index_add_options_t options = {
        /*.provider_name=*/name,
        /*.role=*/role,
    };
    iree_host_size_t provider_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    IREE_CHECK_OK(loom_link_module_index_add_bytecode(
        index, iree_make_const_byte_span(bytes.data(), bytes.size()), name,
        /*index_options=*/nullptr, &options, &provider_ordinal));
    return provider_ordinal;
  }

  iree_host_size_t AddText(loom_link_module_index_t* index,
                           iree_string_view_t source, iree_string_view_t name,
                           loom_link_provider_role_t role) {
    loom_link_module_index_add_options_t options = {
        /*.provider_name=*/name,
        /*.role=*/role,
    };
    loom_text_parse_options_t parse_options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    iree_host_size_t provider_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    IREE_CHECK_OK(loom_link_module_index_add_text(
        index, source, name, &parse_options, &options, &provider_ordinal));
    return provider_ordinal;
  }

  PlanPtr BuildPlan(const loom_link_module_index_t* index,
                    const loom_link_plan_options_t* options) {
    loom_link_plan_t* plan = nullptr;
    IREE_CHECK_OK(
        loom_link_plan_build(index, options, iree_allocator_system(), &plan));
    return PlanPtr(plan);
  }

  iree_status_t BuildPlanStatus(const loom_link_module_index_t* index,
                                const loom_link_plan_options_t* options,
                                PlanPtr* out_plan) {
    loom_link_plan_t* plan = nullptr;
    iree_status_t status =
        loom_link_plan_build(index, options, iree_allocator_system(), &plan);
    if (iree_status_is_ok(status)) {
      *out_plan = PlanPtr(plan);
    }
    return status;
  }

  const loom_link_plan_symbol_t* FindPlannedSymbol(
      const loom_link_plan_t* plan,
      const loom_link_module_index_symbol_t* symbol) {
    if (!symbol) {
      return nullptr;
    }
    for (iree_host_size_t i = 0; i < loom_link_plan_symbol_count(plan); ++i) {
      const loom_link_plan_symbol_t* planned =
          loom_link_plan_symbol_at(plan, i);
      if (planned && planned->symbol_ordinal == symbol->ordinal) {
        return planned;
      }
    }
    return nullptr;
  }

  bool ContainsSymbol(const loom_link_plan_t* plan,
                      const loom_link_module_index_symbol_t* symbol) {
    return symbol && loom_link_plan_contains_symbol(plan, symbol->ordinal);
  }

  const loom_link_module_index_template_family_t* DemandedTemplateFamily(
      const loom_link_plan_t* plan, iree_host_size_t ordinal) {
    const loom_link_template_family_ordinal_t family_ordinal =
        loom_link_plan_demanded_template_family_at(plan, ordinal);
    if (family_ordinal == LOOM_LINK_TEMPLATE_FAMILY_ORDINAL_INVALID) {
      return nullptr;
    }
    return loom_link_module_index_template_family_at(loom_link_plan_index(plan),
                                                     family_ordinal);
  }

  std::vector<std::string> PlannedNames(const loom_link_plan_t* plan) {
    const loom_link_module_index_t* index = loom_link_plan_index(plan);
    std::vector<std::string> names;
    for (iree_host_size_t i = 0; i < loom_link_plan_symbol_count(plan); ++i) {
      const loom_link_plan_symbol_t* planned =
          loom_link_plan_symbol_at(plan, i);
      const loom_link_module_index_symbol_t* symbol =
          loom_link_module_index_symbol_at(index, planned->symbol_ordinal);
      names.push_back(StringViewToString(symbol->name));
    }
    return names;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  std::vector<loom_module_t*> modules_;
};

TEST_F(LinkPlannerTest, MergeSelectsInputSymbolsInStableIndexOrder) {
  loom_module_t* first = Parse(Fixture(
      IREE_SV("merge_selects_all_symbols_in_stable_index_order_first.loom")));
  loom_module_t* second = Parse(Fixture(
      IREE_SV("merge_selects_all_symbols_in_stable_index_order_second.loom")));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), first, IREE_SV("first"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), second, IREE_SV("second"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), second, IREE_SV("library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);

  PlanPtr plan = BuildPlan(index.get(), /*options=*/nullptr);

  EXPECT_EQ(loom_link_plan_symbol_count(plan.get()), 5u);
  EXPECT_EQ(PlannedNames(plan.get()),
            (std::vector<std::string>{"entry_a", "helper", "external",
                                      "entry_b", "helper"}));
  for (iree_host_size_t i = 0; i < loom_link_plan_symbol_count(plan.get());
       ++i) {
    const loom_link_plan_symbol_t* symbol =
        loom_link_plan_symbol_at(plan.get(), i);
    ASSERT_NE(symbol, nullptr);
    EXPECT_EQ(symbol->reason, LOOM_LINK_PLAN_LIVE_MERGE);
    EXPECT_EQ(symbol->cause_ordinal, LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL);
  }
}

TEST_F(LinkPlannerTest, MergeSelectsMoreThanOneSymbolBitsetWord) {
  std::string source;
  for (int symbol_index = 0; symbol_index < 65; ++symbol_index) {
    source += "func.def @symbol_" + std::to_string(symbol_index) +
              "() {\n  func.return\n}\n\n";
  }
  loom_module_t* module =
      Parse(iree_make_string_view(source.data(), source.size()));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), module, IREE_SV("input"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  PlanPtr plan = BuildPlan(index.get(), /*options=*/nullptr);

  ASSERT_EQ(loom_link_plan_symbol_count(plan.get()), 65u);
  const std::vector<std::string> planned_names = PlannedNames(plan.get());
  EXPECT_EQ(planned_names.front(), "symbol_0");
  EXPECT_EQ(planned_names.back(), "symbol_64");
}

TEST_F(LinkPlannerTest, LinkRootClosureSelectsPrivateDependencyOnly) {
  loom_module_t* module = Parse(Fixture(IREE_SV(
      "link_root_closure_selects_private_dependency_only_module.loom")));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), module, IREE_SV("input"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("entry"));
  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_module_at(index.get(), 0);
  ASSERT_NE(indexed_module, nullptr);
  const loom_link_module_index_symbol_t* helper =
      loom_link_module_index_lookup_private(index.get(), indexed_module,
                                            IREE_SV("helper"));
  const loom_link_module_index_symbol_t* unused =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("unused"));
  const loom_link_module_index_symbol_t* unused_private =
      loom_link_module_index_lookup_private(index.get(), indexed_module,
                                            IREE_SV("unused_private"));

  ASSERT_TRUE(ContainsSymbol(plan.get(), entry));
  ASSERT_TRUE(ContainsSymbol(plan.get(), helper));
  EXPECT_FALSE(ContainsSymbol(plan.get(), unused));
  EXPECT_FALSE(ContainsSymbol(plan.get(), unused_private));

  const loom_link_plan_symbol_t* planned_entry =
      FindPlannedSymbol(plan.get(), entry);
  const loom_link_plan_symbol_t* planned_helper =
      FindPlannedSymbol(plan.get(), helper);
  ASSERT_NE(planned_entry, nullptr);
  ASSERT_NE(planned_helper, nullptr);
  EXPECT_EQ(planned_entry->reason, LOOM_LINK_PLAN_LIVE_ROOT);
  EXPECT_EQ(planned_helper->reason, LOOM_LINK_PLAN_LIVE_DEPENDENCY);
  EXPECT_EQ(planned_helper->cause_ordinal, planned_entry->ordinal);
}

TEST_F(LinkPlannerTest, LinkExactRootOrdinalPreservesPrivateIdentity) {
  loom_module_t* first = Parse(IREE_SV(R"(
func.def @entry(%x: i32) -> (i32) {
  %result = func.call @helper(%x) : (i32) -> (i32)
  func.return %result : i32
}
func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)"),
                               IREE_SV("first.loom"));
  loom_module_t* second = Parse(IREE_SV(R"(
func.def @entry(%x: i32) -> (i32) {
  %result = func.call @helper(%x) : (i32) -> (i32)
  func.return %result : i32
}
func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)"),
                                IREE_SV("second.loom"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), first, IREE_SV("first"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), second, IREE_SV("second"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  const loom_link_module_index_module_t* first_module =
      loom_link_module_index_module_at(index.get(), 0);
  const loom_link_module_index_module_t* second_module =
      loom_link_module_index_module_at(index.get(), 1);
  ASSERT_NE(first_module, nullptr);
  ASSERT_NE(second_module, nullptr);
  const loom_link_module_index_symbol_t* first_entry =
      loom_link_module_index_lookup_private(index.get(), first_module,
                                            IREE_SV("entry"));
  const loom_link_module_index_symbol_t* first_helper =
      loom_link_module_index_lookup_private(index.get(), first_module,
                                            IREE_SV("helper"));
  const loom_link_module_index_symbol_t* second_entry =
      loom_link_module_index_lookup_private(index.get(), second_module,
                                            IREE_SV("entry"));
  const loom_link_module_index_symbol_t* second_helper =
      loom_link_module_index_lookup_private(index.get(), second_module,
                                            IREE_SV("helper"));
  ASSERT_NE(first_entry, nullptr);
  ASSERT_NE(first_helper, nullptr);
  ASSERT_NE(second_entry, nullptr);
  ASSERT_NE(second_helper, nullptr);

  const iree_host_size_t root_ordinals[] = {second_entry->ordinal};
  loom_link_plan_options_t options = {};
  options.mode = LOOM_LINK_PLAN_LINK;
  options.root_symbol_ordinals = {
      /*.count=*/IREE_ARRAYSIZE(root_ordinals),
      /*.values=*/root_ordinals,
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  EXPECT_FALSE(ContainsSymbol(plan.get(), first_entry));
  EXPECT_FALSE(ContainsSymbol(plan.get(), first_helper));
  EXPECT_TRUE(ContainsSymbol(plan.get(), second_entry));
  EXPECT_TRUE(ContainsSymbol(plan.get(), second_helper));
}

TEST_F(LinkPlannerTest, LinkExactRootOrdinalRejectsOutOfRangeIdentity) {
  loom_module_t* module = Parse(Fixture(IREE_SV(
      "link_root_closure_selects_private_dependency_only_module.loom")));
  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), module, IREE_SV("input"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);

  const iree_host_size_t root_ordinals[] = {
      loom_link_module_index_symbol_count(index.get()),
  };
  loom_link_plan_options_t options = {};
  options.mode = LOOM_LINK_PLAN_LINK;
  options.root_symbol_ordinals = {
      /*.count=*/IREE_ARRAYSIZE(root_ordinals),
      /*.values=*/root_ordinals,
  };
  PlanPtr plan;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        BuildPlanStatus(index.get(), &options, &plan));
}

TEST_F(LinkPlannerTest,
       OrdinaryDefinitionFacetCombinesEveryRootAcrossProviderForms) {
  const iree_string_view_t source =
      Fixture(IREE_SV("ordinary_definition_facet_combines_every_root_across_"
                      "provider_forms_source.loom"));
  loom_module_t* module = Parse(source);
  const std::vector<uint8_t> bytecode = WriteModule(module);

  auto verify_index = [&](const loom_link_module_index_t* index) {
    const loom_link_module_index_module_t* indexed_module =
        loom_link_module_index_module_at(index, 0);
    ASSERT_NE(indexed_module, nullptr);
    const loom_link_module_index_symbol_t* root =
        loom_link_module_index_lookup_private(index, indexed_module,
                                              IREE_SV("split_root"));
    const loom_link_module_index_symbol_t* config_dependency =
        loom_link_module_index_lookup_private(index, indexed_module,
                                              IREE_SV("config_dependency"));
    const loom_link_module_index_symbol_t* implementation_dependency =
        loom_link_module_index_lookup_private(
            index, indexed_module, IREE_SV("implementation_dependency"));
    ASSERT_NE(root, nullptr);
    ASSERT_NE(config_dependency, nullptr);
    ASSERT_NE(implementation_dependency, nullptr);

    EXPECT_EQ(root->facets.schema.root_region_count, 2u);
    EXPECT_EQ(root->facets.schema.facet_count, 1u);
    EXPECT_EQ(loom_link_module_index_symbol_facet_kind_at(root, 0),
              LOOM_LINK_SYMBOL_FACET_DEFINITION);
    EXPECT_EQ(loom_link_module_index_symbol_source_root_facet_kind(root, 0),
              LOOM_LINK_SYMBOL_FACET_DEFINITION);
    EXPECT_EQ(loom_link_module_index_symbol_source_root_facet_kind(root, 1),
              LOOM_LINK_SYMBOL_FACET_DEFINITION);
    EXPECT_EQ(loom_link_module_index_symbol_source_root_facet_kind(root, 2),
              LOOM_LINK_SYMBOL_FACET_DEFINITION);

    const loom_link_plan_root_facet_t definition_root = {
        /*.symbol_ordinal=*/root->ordinal,
        /*.kind=*/LOOM_LINK_SYMBOL_FACET_DEFINITION,
    };
    loom_link_plan_options_t options = {};
    options.mode = LOOM_LINK_PLAN_LINK;
    options.root_facets = {1, &definition_root};
    PlanPtr plan = BuildPlan(index, &options);
    EXPECT_TRUE(ContainsSymbol(plan.get(), root));
    EXPECT_TRUE(ContainsSymbol(plan.get(), config_dependency));
    EXPECT_TRUE(ContainsSymbol(plan.get(), implementation_dependency));
    EXPECT_EQ(loom_link_plan_facet_count(plan.get()), 3u);
    EXPECT_TRUE(loom_link_plan_contains_facet(
        plan.get(), root->ordinal, LOOM_LINK_SYMBOL_FACET_DEFINITION));
  };

  IndexPtr materialized_index = CreateIndex();
  AddMaterialized(materialized_index.get(), module, IREE_SV("materialized"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  verify_index(materialized_index.get());

  IndexPtr text_index = CreateIndex();
  AddText(text_index.get(), source, IREE_SV("module.loom"),
          LOOM_LINK_PROVIDER_ROLE_INPUT);
  verify_index(text_index.get());

  IndexPtr bytecode_index = CreateIndex();
  AddBytecode(bytecode_index.get(), bytecode, IREE_SV("module.loombc"),
              LOOM_LINK_PROVIDER_ROLE_INPUT);
  verify_index(bytecode_index.get());
}

TEST_F(LinkPlannerTest,
       CommandFacetsSeparateContractAndImplementationAcrossProviderForms) {
  const iree_string_view_t source =
      Fixture(IREE_SV("command_facets_separate_contract_and_implementation_"
                      "across_provider_forms_source.loom"));
  loom_module_t* module = Parse(source);
  const std::vector<uint8_t> bytecode = WriteModule(module);

  auto verify_index = [&](const loom_link_module_index_t* index) {
    const loom_link_module_index_symbol_t* root =
        loom_link_module_index_lookup_name(index, IREE_SV("root"));
    const loom_link_module_index_symbol_t* leaf =
        loom_link_module_index_lookup_name(index, IREE_SV("leaf"));
    ASSERT_NE(root, nullptr);
    ASSERT_NE(leaf, nullptr);
    EXPECT_EQ(root->facets.schema.facet_count, 2u);
    EXPECT_EQ(loom_link_module_index_symbol_facet_kind_at(root, 0),
              LOOM_LINK_SYMBOL_FACET_COMMAND_CONTRACT);
    EXPECT_EQ(loom_link_module_index_symbol_facet_kind_at(root, 1),
              LOOM_LINK_SYMBOL_FACET_COMMAND_IMPLEMENTATION);

    const loom_link_plan_root_facet_t contract_root = {
        /*.symbol_ordinal=*/root->ordinal,
        /*.kind=*/LOOM_LINK_SYMBOL_FACET_COMMAND_CONTRACT,
    };
    loom_link_plan_options_t contract_options = {};
    contract_options.mode = LOOM_LINK_PLAN_LINK;
    contract_options.root_facets = {1, &contract_root};
    PlanPtr contract_plan = BuildPlan(index, &contract_options);
    EXPECT_TRUE(ContainsSymbol(contract_plan.get(), root));
    EXPECT_FALSE(ContainsSymbol(contract_plan.get(), leaf));
    EXPECT_TRUE(
        loom_link_plan_contains_facet(contract_plan.get(), root->ordinal,
                                      LOOM_LINK_SYMBOL_FACET_COMMAND_CONTRACT));
    EXPECT_FALSE(loom_link_plan_contains_facet(
        contract_plan.get(), root->ordinal,
        LOOM_LINK_SYMBOL_FACET_COMMAND_IMPLEMENTATION));

    const loom_link_plan_root_facet_t invalid_root = {
        /*.symbol_ordinal=*/root->ordinal,
        /*.kind=*/LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION,
    };
    loom_link_plan_options_t invalid_options = {};
    invalid_options.mode = LOOM_LINK_PLAN_LINK;
    invalid_options.root_facets = {1, &invalid_root};
    PlanPtr invalid_plan;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        BuildPlanStatus(index, &invalid_options, &invalid_plan));

    const loom_link_plan_root_facet_t implementation_root = {
        /*.symbol_ordinal=*/root->ordinal,
        /*.kind=*/LOOM_LINK_SYMBOL_FACET_COMMAND_IMPLEMENTATION,
    };
    loom_link_plan_options_t implementation_options = {};
    implementation_options.mode = LOOM_LINK_PLAN_LINK;
    implementation_options.root_facets = {1, &implementation_root};
    PlanPtr implementation_plan = BuildPlan(index, &implementation_options);
    EXPECT_TRUE(ContainsSymbol(implementation_plan.get(), root));
    EXPECT_TRUE(ContainsSymbol(implementation_plan.get(), leaf));
    EXPECT_TRUE(
        loom_link_plan_contains_facet(implementation_plan.get(), root->ordinal,
                                      LOOM_LINK_SYMBOL_FACET_COMMAND_CONTRACT));
    EXPECT_TRUE(loom_link_plan_contains_facet(
        implementation_plan.get(), root->ordinal,
        LOOM_LINK_SYMBOL_FACET_COMMAND_IMPLEMENTATION));
  };

  IndexPtr materialized_index = CreateIndex();
  AddMaterialized(materialized_index.get(), module, IREE_SV("materialized"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  verify_index(materialized_index.get());

  IndexPtr text_index = CreateIndex();
  AddText(text_index.get(), source, IREE_SV("module.loom"),
          LOOM_LINK_PROVIDER_ROLE_INPUT);
  verify_index(text_index.get());

  IndexPtr bytecode_index = CreateIndex();
  AddBytecode(bytecode_index.get(), bytecode, IREE_SV("module.loombc"),
              LOOM_LINK_PROVIDER_ROLE_INPUT);
  verify_index(bytecode_index.get());
}

TEST_F(LinkPlannerTest, InterleavedKernelFacetUpgradesPreservePerSymbolChains) {
  const iree_string_view_t source =
      Fixture(IREE_SV("interleaved_kernel_facet_upgrades_preserve_per_symbol_"
                      "chains_source.loom"));
  loom_module_t* module = Parse(source);
  const std::vector<uint8_t> bytecode = WriteModule(module);

  auto verify_index = [&](const loom_link_module_index_t* index) {
    const loom_link_module_index_module_t* indexed_module =
        loom_link_module_index_module_at(index, 0);
    ASSERT_NE(indexed_module, nullptr);
    const loom_link_module_index_symbol_t* target =
        loom_link_module_index_lookup_private(index, indexed_module,
                                              IREE_SV("target"));
    const loom_link_module_index_symbol_t* configuration_dependency =
        loom_link_module_index_lookup_global(
            index, IREE_SV("configuration_dependency"));
    const loom_link_module_index_symbol_t* implementation_dependency =
        loom_link_module_index_lookup_private(
            index, indexed_module, IREE_SV("implementation_dependency"));
    const loom_link_module_index_symbol_t* interleaved_dependency =
        loom_link_module_index_lookup_private(
            index, indexed_module, IREE_SV("interleaved_dependency"));
    ASSERT_NE(target, nullptr);
    ASSERT_NE(configuration_dependency, nullptr);
    ASSERT_NE(implementation_dependency, nullptr);
    ASSERT_NE(interleaved_dependency, nullptr);

    const loom_link_plan_root_facet_t roots[] = {
        {
            /*.symbol_ordinal=*/target->ordinal,
            /*.kind=*/LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION,
        },
        {
            /*.symbol_ordinal=*/interleaved_dependency->ordinal,
            /*.kind=*/LOOM_LINK_SYMBOL_FACET_DEFINITION,
        },
        {
            /*.symbol_ordinal=*/target->ordinal,
            /*.kind=*/LOOM_LINK_SYMBOL_FACET_KERNEL_IMPLEMENTATION,
        },
    };
    loom_link_plan_options_t options = {};
    options.mode = LOOM_LINK_PLAN_LINK;
    options.root_facets.count = IREE_ARRAYSIZE(roots);
    options.root_facets.values = roots;
    PlanPtr plan = BuildPlan(index, &options);

    EXPECT_TRUE(ContainsSymbol(plan.get(), target));
    EXPECT_TRUE(loom_link_plan_contains_facet(
        plan.get(), target->ordinal, LOOM_LINK_SYMBOL_FACET_KERNEL_CONTRACT));
    EXPECT_TRUE(loom_link_plan_contains_facet(
        plan.get(), target->ordinal,
        LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION));
    EXPECT_TRUE(loom_link_plan_contains_facet(
        plan.get(), target->ordinal,
        LOOM_LINK_SYMBOL_FACET_KERNEL_IMPLEMENTATION));
    EXPECT_TRUE(ContainsSymbol(plan.get(), configuration_dependency));
    EXPECT_TRUE(ContainsSymbol(plan.get(), interleaved_dependency));
    EXPECT_TRUE(ContainsSymbol(plan.get(), implementation_dependency));
  };

  IndexPtr materialized_index = CreateIndex();
  AddMaterialized(materialized_index.get(), module, IREE_SV("materialized"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  verify_index(materialized_index.get());

  IndexPtr text_index = CreateIndex();
  AddText(text_index.get(), source, IREE_SV("module.loom"),
          LOOM_LINK_PROVIDER_ROLE_INPUT);
  verify_index(text_index.get());

  IndexPtr bytecode_index = CreateIndex();
  AddBytecode(bytecode_index.get(), bytecode, IREE_SV("module.loombc"),
              LOOM_LINK_PROVIDER_ROLE_INPUT);
  verify_index(bytecode_index.get());
}

TEST_F(LinkPlannerTest, KernelReferencesSelectOnlyTheirRequiredFacets) {
  const iree_string_view_t harness_source =
      Fixture(IREE_SV("kernel_references_select_only_their_required_facets_"
                      "harness_source.loom"));
  const iree_string_view_t wrong_library_source =
      Fixture(IREE_SV("kernel_references_select_only_their_required_facets_"
                      "wrong_library_source.loom"));
  const iree_string_view_t library_source =
      Fixture(IREE_SV("kernel_references_select_only_their_required_facets_"
                      "library_source.loom"));
  loom_module_t* harness = Parse(harness_source);
  ASSERT_NE(harness, nullptr);
  loom_module_t* wrong_library = Parse(wrong_library_source);
  ASSERT_NE(wrong_library, nullptr);
  loom_module_t* library = Parse(library_source);
  ASSERT_NE(library, nullptr);
  const std::vector<uint8_t> harness_bytecode = WriteModule(harness);
  const std::vector<uint8_t> wrong_library_bytecode =
      WriteModule(wrong_library);
  const std::vector<uint8_t> library_bytecode = WriteModule(library);

  auto verify_index = [&](const loom_link_module_index_t* index) {
    iree_string_view_t roots[] = {IREE_SV("@entry")};
    loom_link_plan_options_t options = {
        /*.mode=*/LOOM_LINK_PLAN_LINK,
        /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
        /*.include_input_exports=*/false,
        /*.unresolved_policy=*/LOOM_LINK_PLAN_UNRESOLVED_ERROR,
        /*.test_symbol_policy=*/LOOM_LINK_PLAN_TEST_SYMBOL_KEEP,
        /*.strip_symbol=*/nullptr,
        /*.strip_symbol_user_data=*/nullptr,
        /*.template_provider_roots=*/{},
        /*.root_facets=*/{},
        /*.dependency_policy=*/LOOM_LINK_PLAN_DEPENDENCY_REQUESTED_FACETS,
    };
    PlanPtr plan = BuildPlan(index, &options);

    const loom_link_module_index_symbol_t* configured_declaration =
        loom_link_module_index_lookup_global(index, IREE_SV("configured"));
    ASSERT_NE(configured_declaration, nullptr);
    const loom_link_module_index_symbol_t* wrong_provider =
        loom_link_module_index_next_same_name(index, configured_declaration);
    ASSERT_NE(wrong_provider, nullptr);
    const loom_link_module_index_symbol_t* configured_provider =
        loom_link_module_index_next_same_name(index, wrong_provider);
    ASSERT_NE(configured_provider, nullptr);
    const loom_link_module_index_symbol_t* dispatched_declaration =
        loom_link_module_index_lookup_global(index, IREE_SV("dispatched"));
    ASSERT_NE(dispatched_declaration, nullptr);
    const loom_link_module_index_symbol_t* dispatched_provider =
        loom_link_module_index_next_same_name(index, dispatched_declaration);
    ASSERT_NE(dispatched_provider, nullptr);
    const loom_link_module_index_module_t* library_module =
        loom_link_module_index_module_at(index, 2);
    const loom_link_module_index_module_t* harness_module =
        loom_link_module_index_module_at(index, 0);
    ASSERT_NE(harness_module, nullptr);
    ASSERT_NE(library_module, nullptr);
    const loom_link_module_index_symbol_t* local =
        loom_link_module_index_lookup_private(index, harness_module,
                                              IREE_SV("local"));
    const loom_link_module_index_symbol_t* local_configuration_dependency =
        loom_link_module_index_lookup_global(
            index, IREE_SV("local_configuration_dependency"));
    const loom_link_module_index_symbol_t* local_implementation_dependency =
        loom_link_module_index_lookup_private(
            index, harness_module, IREE_SV("local_implementation_dependency"));
    const loom_link_module_index_symbol_t* configuration_dependency =
        loom_link_module_index_lookup_global(
            index, IREE_SV("configuration_dependency"));
    const loom_link_module_index_symbol_t* implementation_dependency =
        loom_link_module_index_lookup_private(
            index, library_module, IREE_SV("implementation_dependency"));
    const loom_link_module_index_symbol_t* dispatched_configuration_dependency =
        loom_link_module_index_lookup_global(
            index, IREE_SV("dispatched_configuration_dependency"));
    ASSERT_NE(configuration_dependency, nullptr);
    ASSERT_NE(dispatched_configuration_dependency, nullptr);
    ASSERT_NE(implementation_dependency, nullptr);
    ASSERT_NE(local, nullptr);
    ASSERT_NE(local_configuration_dependency, nullptr);
    ASSERT_NE(local_implementation_dependency, nullptr);

    EXPECT_TRUE(ContainsSymbol(plan.get(), local));
    EXPECT_TRUE(loom_link_plan_contains_facet(
        plan.get(), local->ordinal, LOOM_LINK_SYMBOL_FACET_KERNEL_CONTRACT));
    EXPECT_TRUE(loom_link_plan_contains_facet(
        plan.get(), local->ordinal,
        LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION));
    EXPECT_FALSE(loom_link_plan_contains_facet(
        plan.get(), local->ordinal,
        LOOM_LINK_SYMBOL_FACET_KERNEL_IMPLEMENTATION));
    EXPECT_TRUE(ContainsSymbol(plan.get(), local_configuration_dependency));
    EXPECT_FALSE(ContainsSymbol(plan.get(), local_implementation_dependency));

    EXPECT_TRUE(ContainsSymbol(plan.get(), configured_declaration));
    EXPECT_FALSE(ContainsSymbol(plan.get(), wrong_provider));
    EXPECT_TRUE(ContainsSymbol(plan.get(), configured_provider));
    EXPECT_TRUE(
        loom_link_plan_contains_facet(plan.get(), configured_provider->ordinal,
                                      LOOM_LINK_SYMBOL_FACET_KERNEL_CONTRACT));
    EXPECT_TRUE(loom_link_plan_contains_facet(
        plan.get(), configured_provider->ordinal,
        LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION));
    EXPECT_FALSE(loom_link_plan_contains_facet(
        plan.get(), configured_provider->ordinal,
        LOOM_LINK_SYMBOL_FACET_KERNEL_IMPLEMENTATION));
    EXPECT_TRUE(ContainsSymbol(plan.get(), configuration_dependency));

    EXPECT_TRUE(ContainsSymbol(plan.get(), dispatched_declaration));
    EXPECT_TRUE(ContainsSymbol(plan.get(), dispatched_provider));
    EXPECT_TRUE(
        loom_link_plan_contains_facet(plan.get(), dispatched_provider->ordinal,
                                      LOOM_LINK_SYMBOL_FACET_KERNEL_CONTRACT));
    EXPECT_FALSE(loom_link_plan_contains_facet(
        plan.get(), dispatched_provider->ordinal,
        LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION));
    EXPECT_FALSE(loom_link_plan_contains_facet(
        plan.get(), dispatched_provider->ordinal,
        LOOM_LINK_SYMBOL_FACET_KERNEL_IMPLEMENTATION));
    EXPECT_FALSE(
        ContainsSymbol(plan.get(), dispatched_configuration_dependency));
    EXPECT_FALSE(ContainsSymbol(plan.get(), implementation_dependency));
  };

  IndexPtr materialized_index = CreateIndex();
  AddMaterialized(materialized_index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(materialized_index.get(), wrong_library,
                  IREE_SV("wrong-library"), LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  AddMaterialized(materialized_index.get(), library, IREE_SV("library"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  verify_index(materialized_index.get());

  IndexPtr text_index = CreateIndex();
  AddText(text_index.get(), harness_source, IREE_SV("harness.loom"),
          LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddText(text_index.get(), wrong_library_source, IREE_SV("wrong-library.loom"),
          LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  AddText(text_index.get(), library_source, IREE_SV("library.loom"),
          LOOM_LINK_PROVIDER_ROLE_INPUT);
  verify_index(text_index.get());

  IndexPtr bytecode_index = CreateIndex();
  AddBytecode(bytecode_index.get(), harness_bytecode, IREE_SV("harness.loombc"),
              LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddBytecode(bytecode_index.get(), wrong_library_bytecode,
              IREE_SV("wrong-library.loombc"), LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  AddBytecode(bytecode_index.get(), library_bytecode, IREE_SV("library.loombc"),
              LOOM_LINK_PROVIDER_ROLE_INPUT);
  verify_index(bytecode_index.get());
}

TEST_F(LinkPlannerTest, LinkRootIgnoresAvailabilityReferences) {
  loom_module_t* module = Parse(Fixture(
      IREE_SV("link_root_ignores_availability_references_module.loom")));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), module, IREE_SV("input"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("entry"));
  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_module_at(index.get(), 0);
  ASSERT_NE(indexed_module, nullptr);
  const loom_link_module_index_symbol_t* available =
      loom_link_module_index_lookup_private(index.get(), indexed_module,
                                            IREE_SV("available"));
  EXPECT_TRUE(ContainsSymbol(plan.get(), entry));
  EXPECT_FALSE(ContainsSymbol(plan.get(), available));
}

TEST_F(LinkPlannerTest, LinkBytecodePlanningUsesSerializedDependencies) {
  loom_module_t* used = Parse(Fixture(IREE_SV(
      "link_bytecode_planning_uses_serialized_dependencies_used.loom")));
  loom_module_t* unused = Parse(Fixture(IREE_SV(
      "link_bytecode_planning_uses_serialized_dependencies_unused.loom")));
  std::vector<uint8_t> used_bytes = WriteModule(used);
  std::vector<uint8_t> unused_bytes = WriteModule(unused);

  IndexPtr index = CreateIndex();
  loom_link_module_index_add_options_t used_options = {
      /*.provider_name=*/IREE_SV("used-lib"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      index.get(),
      iree_make_const_byte_span(used_bytes.data(), used_bytes.size()),
      IREE_SV("used.loombc"), /*index_options=*/nullptr, &used_options,
      /*out_provider_ordinal=*/nullptr));
  loom_link_module_index_add_options_t unused_options = {
      /*.provider_name=*/IREE_SV("unused-lib"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      index.get(),
      iree_make_const_byte_span(unused_bytes.data(), unused_bytes.size()),
      IREE_SV("unused.loombc"), /*index_options=*/nullptr, &unused_options,
      /*out_provider_ordinal=*/nullptr));

  const loom_link_module_index_module_t* used_module =
      loom_link_module_index_module_at(index.get(), 0);
  const loom_link_module_index_module_t* unused_module =
      loom_link_module_index_module_at(index.get(), 1);
  ASSERT_NE(used_module, nullptr);
  ASSERT_NE(unused_module, nullptr);
  EXPECT_EQ(used_module->materialized_module, nullptr);
  EXPECT_EQ(unused_module->materialized_module, nullptr);

  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("entry"));
  const loom_link_module_index_symbol_t* helper =
      loom_link_module_index_lookup_private(index.get(), used_module,
                                            IREE_SV("helper"));
  const loom_link_module_index_symbol_t* unused_symbol =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("unused"));
  EXPECT_TRUE(ContainsSymbol(plan.get(), entry));
  EXPECT_TRUE(ContainsSymbol(plan.get(), helper));
  EXPECT_FALSE(ContainsSymbol(plan.get(), unused_symbol));
  EXPECT_EQ(used_module->materialized_module, nullptr);
  EXPECT_EQ(unused_module->materialized_module, nullptr);
}

TEST_F(LinkPlannerTest, LinkApplyReportsBytecodeFamilyDemand) {
  loom_module_t* harness = Parse(Fixture(
      IREE_SV("link_apply_reports_bytecode_family_demand_harness.loom")));
  loom_module_t* used = Parse(
      Fixture(IREE_SV("link_apply_reports_bytecode_family_demand_used.loom")));
  loom_module_t* unused = Parse(Fixture(
      IREE_SV("link_apply_reports_bytecode_family_demand_unused.loom")));
  std::vector<uint8_t> used_bytes = WriteModule(used);
  std::vector<uint8_t> unused_bytes = WriteModule(unused);

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  loom_link_module_index_add_options_t used_options = {
      /*.provider_name=*/IREE_SV("used-lib"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      index.get(),
      iree_make_const_byte_span(used_bytes.data(), used_bytes.size()),
      IREE_SV("used.loombc"), /*index_options=*/nullptr, &used_options,
      /*out_provider_ordinal=*/nullptr));
  loom_link_module_index_add_options_t unused_options = {
      /*.provider_name=*/IREE_SV("unused-lib"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      index.get(),
      iree_make_const_byte_span(unused_bytes.data(), unused_bytes.size()),
      IREE_SV("unused.loombc"), /*index_options=*/nullptr, &unused_options,
      /*out_provider_ordinal=*/nullptr));

  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/
      {
          /*.count=*/IREE_ARRAYSIZE(roots),
          /*.values=*/roots,
      },
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("entry"));
  const loom_link_module_index_module_t* used_module =
      loom_link_module_index_module_at(index.get(), 1);
  const loom_link_module_index_module_t* unused_module =
      loom_link_module_index_module_at(index.get(), 2);
  ASSERT_NE(used_module, nullptr);
  ASSERT_NE(unused_module, nullptr);
  const loom_link_module_index_symbol_t* bytecode_provider =
      loom_link_module_index_lookup_private(index.get(), used_module,
                                            IREE_SV("bytecode_provider"));
  const loom_link_module_index_symbol_t* unused_provider =
      loom_link_module_index_lookup_private(index.get(), unused_module,
                                            IREE_SV("unused_provider"));

  EXPECT_TRUE(ContainsSymbol(plan.get(), entry));
  EXPECT_FALSE(ContainsSymbol(plan.get(), bytecode_provider));
  EXPECT_FALSE(ContainsSymbol(plan.get(), unused_provider));
  ASSERT_EQ(loom_link_plan_demanded_template_family_count(plan.get()), 1u);
  const loom_link_module_index_template_family_t* demanded_family =
      DemandedTemplateFamily(plan.get(), 0);
  ASSERT_NE(demanded_family, nullptr);
  EXPECT_EQ(StringViewToString(demanded_family->name), "demo.bytecode");
  EXPECT_EQ(used_module->materialized_module, nullptr);
  EXPECT_EQ(unused_module->materialized_module, nullptr);
}

TEST_F(LinkPlannerTest, LinkRootMayNameUniquePrivateSymbol) {
  loom_module_t* module = Parse(
      Fixture(IREE_SV("link_root_may_name_unique_private_symbol_module.loom")));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), module, IREE_SV("input"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_module_at(index.get(), 0);
  ASSERT_NE(indexed_module, nullptr);
  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_private(index.get(), indexed_module,
                                            IREE_SV("entry"));
  const loom_link_module_index_symbol_t* helper =
      loom_link_module_index_lookup_private(index.get(), indexed_module,
                                            IREE_SV("helper"));
  EXPECT_TRUE(ContainsSymbol(plan.get(), entry));
  EXPECT_TRUE(ContainsSymbol(plan.get(), helper));
}

TEST_F(LinkPlannerTest, LinkImportFreeDeclarationPullsConcreteDefinition) {
  const iree_string_view_t harness_source =
      Fixture(IREE_SV("link_import_free_declaration_pulls_concrete_"
                      "definition_harness_source.loom"));
  const iree_string_view_t library_source =
      Fixture(IREE_SV("link_import_free_declaration_pulls_concrete_"
                      "definition_library_source.loom"));
  loom_module_t* harness = Parse(harness_source);
  loom_module_t* library = Parse(library_source);
  const std::vector<uint8_t> harness_bytecode = WriteModule(harness);
  const std::vector<uint8_t> library_bytecode = WriteModule(library);

  auto verify_index = [&](const loom_link_module_index_t* index) {
    iree_string_view_t roots[] = {IREE_SV("@entry")};
    loom_link_plan_options_t options = {
        /*.mode=*/LOOM_LINK_PLAN_LINK,
        /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
    };
    PlanPtr plan = BuildPlan(index, &options);

    const loom_link_module_index_symbol_t* entry =
        loom_link_module_index_lookup_global(index, IREE_SV("entry"));
    const loom_link_module_index_symbol_t* callee_decl =
        loom_link_module_index_lookup_global(index, IREE_SV("callee"));
    ASSERT_NE(callee_decl, nullptr);
    const loom_link_module_index_symbol_t* callee_def =
        loom_link_module_index_next_global_duplicate(index, callee_decl);
    ASSERT_NE(callee_def, nullptr);
    const loom_link_module_index_symbol_t* unused =
        loom_link_module_index_lookup_global(index, IREE_SV("unused"));

    EXPECT_TRUE(ContainsSymbol(plan.get(), entry));
    EXPECT_TRUE(ContainsSymbol(plan.get(), callee_decl));
    EXPECT_TRUE(ContainsSymbol(plan.get(), callee_def));
    EXPECT_FALSE(ContainsSymbol(plan.get(), unused));

    const loom_link_module_index_provider_t* def_provider =
        loom_link_module_index_symbol_provider(index, callee_def);
    ASSERT_NE(def_provider, nullptr);
    EXPECT_EQ(StringViewToString(def_provider->name), "library");

    const loom_link_plan_symbol_t* planned_decl =
        FindPlannedSymbol(plan.get(), callee_decl);
    const loom_link_plan_symbol_t* planned_def =
        FindPlannedSymbol(plan.get(), callee_def);
    ASSERT_NE(planned_decl, nullptr);
    ASSERT_NE(planned_def, nullptr);
    EXPECT_EQ(planned_decl->reason, LOOM_LINK_PLAN_LIVE_DEPENDENCY);
    EXPECT_EQ(planned_def->reason, LOOM_LINK_PLAN_LIVE_DEPENDENCY);
    EXPECT_EQ(planned_def->cause_ordinal, planned_decl->ordinal);
  };

  IndexPtr materialized_index = CreateIndex();
  AddMaterialized(materialized_index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(materialized_index.get(), library, IREE_SV("library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  verify_index(materialized_index.get());

  IndexPtr reversed_index = CreateIndex();
  AddMaterialized(reversed_index.get(), library, IREE_SV("library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  AddMaterialized(reversed_index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  verify_index(reversed_index.get());

  IndexPtr text_index = CreateIndex();
  AddText(text_index.get(), harness_source, IREE_SV("harness"),
          LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddText(text_index.get(), library_source, IREE_SV("library"),
          LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  verify_index(text_index.get());

  IndexPtr bytecode_index = CreateIndex();
  AddBytecode(bytecode_index.get(), harness_bytecode, IREE_SV("harness"),
              LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddBytecode(bytecode_index.get(), library_bytecode, IREE_SV("library"),
              LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  verify_index(bytecode_index.get());
}

TEST_F(LinkPlannerTest, OverlayPlansInputAgainstImmutableLibrary) {
  loom_module_t* harness =
      Parse(Fixture(IREE_SV("link_import_free_declaration_pulls_concrete_"
                            "definition_harness_source.loom")));
  loom_module_t* library =
      Parse(Fixture(IREE_SV("link_import_free_declaration_pulls_concrete_"
                            "definition_library_source.loom")));
  const std::vector<uint8_t> harness_bytecode = WriteModule(harness);
  const std::vector<uint8_t> library_bytecode = WriteModule(library);

  IndexPtr base_index = CreateIndex();
  AddBytecode(base_index.get(), library_bytecode, IREE_SV("library"),
              LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  IndexPtr overlay = CreateOverlay(base_index.get());
  AddBytecode(overlay.get(), harness_bytecode, IREE_SV("harness"),
              LOOM_LINK_PROVIDER_ROLE_INPUT);

  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_global(overlay.get(), IREE_SV("entry"));
  const loom_link_module_index_symbol_t* declaration =
      loom_link_module_index_lookup_global(overlay.get(), IREE_SV("callee"));
  ASSERT_NE(entry, nullptr);
  ASSERT_NE(declaration, nullptr);
  const loom_link_module_index_symbol_t* definition =
      loom_link_module_index_next_global_duplicate(overlay.get(), declaration);
  ASSERT_NE(definition, nullptr);

  iree_string_view_t roots[] = {IREE_SV("@entry")};
  const iree_host_size_t exact_roots[] = {declaration->ordinal};
  loom_link_plan_options_t options = {};
  options.mode = LOOM_LINK_PLAN_LINK;
  options.root_symbols = {
      /*.count=*/IREE_ARRAYSIZE(roots),
      /*.values=*/roots,
  };
  options.root_symbol_ordinals = {
      /*.count=*/IREE_ARRAYSIZE(exact_roots),
      /*.values=*/exact_roots,
  };
  PlanPtr plan = BuildPlan(overlay.get(), &options);

  EXPECT_TRUE(ContainsSymbol(plan.get(), entry));
  EXPECT_TRUE(ContainsSymbol(plan.get(), declaration));
  EXPECT_TRUE(ContainsSymbol(plan.get(), definition));
  const loom_link_plan_symbol_t* declaration_selection =
      FindPlannedSymbol(plan.get(), declaration);
  ASSERT_NE(declaration_selection, nullptr);
  EXPECT_EQ(declaration_selection->reason, LOOM_LINK_PLAN_LIVE_ROOT);

  const loom_link_module_index_provider_t* definition_provider =
      loom_link_module_index_symbol_provider(overlay.get(), definition);
  ASSERT_NE(definition_provider, nullptr);
  EXPECT_EQ(definition_provider,
            loom_link_module_index_provider_at(base_index.get(), 0));
  EXPECT_EQ(StringViewToString(definition_provider->name), "library");
}

TEST_F(LinkPlannerTest, LinkTargetRequirementUsesConcreteEnvironment) {
  loom_module_t* harness = Parse(Fixture(IREE_SV(
      "link_target_requirement_uses_concrete_environment_harness.loom")));
  loom_module_t* library = Parse(Fixture(IREE_SV(
      "link_target_requirement_uses_concrete_environment_library.loom")));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), library, IREE_SV("library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("entry"));
  const loom_link_module_index_symbol_t* target_decl =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("gpu"));
  ASSERT_NE(target_decl, nullptr);
  const loom_link_module_index_module_t* library_module =
      loom_link_module_index_module_at(index.get(), 1);
  ASSERT_NE(library_module, nullptr);
  const loom_link_module_index_symbol_t* target_def =
      loom_link_module_index_lookup_private(index.get(), library_module,
                                            IREE_SV("gpu"));
  ASSERT_NE(target_def, nullptr);

  EXPECT_TRUE(ContainsSymbol(plan.get(), entry));
  EXPECT_TRUE(ContainsSymbol(plan.get(), target_decl));
  EXPECT_TRUE(ContainsSymbol(plan.get(), target_def));
}

TEST_F(LinkPlannerTest, LinkTargetRequirementMayRemainUnbound) {
  loom_module_t* harness = Parse(Fixture(
      IREE_SV("link_target_requirement_may_remain_unbound_harness.loom")));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_symbol_t* target_decl =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("gpu"));
  EXPECT_TRUE(ContainsSymbol(plan.get(), target_decl));
  EXPECT_EQ(loom_link_plan_symbol_count(plan.get()), 2u);
}

TEST_F(LinkPlannerTest, LinkDeclarationRejectsPrivateLibraryDefinition) {
  loom_module_t* harness =
      Parse(Fixture(IREE_SV("link_declaration_rejects_private_library_"
                            "definition_harness.loom")));
  loom_module_t* library =
      Parse(Fixture(IREE_SV("link_declaration_rejects_private_library_"
                            "definition_library.loom")));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), library, IREE_SV("library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  PlanPtr plan;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        BuildPlanStatus(index.get(), &options, &plan));
}

TEST_F(LinkPlannerTest, LinkDeclarationRejectsWrongSymbolInterface) {
  loom_module_t* harness = Parse(Fixture(
      IREE_SV("link_declaration_rejects_wrong_symbol_interface_harness.loom")));
  loom_module_t* library = Parse(Fixture(
      IREE_SV("link_declaration_rejects_wrong_symbol_interface_library.loom")));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), library, IREE_SV("library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  PlanPtr plan;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        BuildPlanStatus(index.get(), &options, &plan));
}

TEST_F(LinkPlannerTest, LinkDeclarationMayUsePrivateOwnerDefinition) {
  loom_module_t* harness = Parse(Fixture(IREE_SV(
      "link_declaration_may_use_private_owner_definition_harness.loom")));
  loom_module_t* sibling_source =
      Parse(Fixture(IREE_SV("link_declaration_may_use_private_owner_"
                            "definition_sibling_source.loom")));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), sibling_source, IREE_SV("sibling-source"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_module_t* sibling_module =
      loom_link_module_index_module_at(index.get(), 1);
  ASSERT_NE(sibling_module, nullptr);
  const loom_link_module_index_symbol_t* callee_decl =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("callee"));
  const loom_link_module_index_symbol_t* callee_def =
      loom_link_module_index_lookup_private(index.get(), sibling_module,
                                            IREE_SV("callee"));
  EXPECT_TRUE(ContainsSymbol(plan.get(), callee_decl));
  EXPECT_TRUE(ContainsSymbol(plan.get(), callee_def));
}

TEST_F(LinkPlannerTest, UnresolvedDeclarationIgnoresPrivateLibraryDefinition) {
  loom_module_t* harness =
      Parse(Fixture(IREE_SV("unresolved_declaration_ignores_private_library_"
                            "definition_harness.loom")));
  loom_module_t* library =
      Parse(Fixture(IREE_SV("unresolved_declaration_ignores_private_library_"
                            "definition_library.loom")));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), library, IREE_SV("library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{},
      /*.include_input_exports=*/true,
      /*.unresolved_policy=*/LOOM_LINK_PLAN_UNRESOLVED_ALLOW,
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_module_t* library_module =
      loom_link_module_index_module_at(index.get(), 1);
  ASSERT_NE(library_module, nullptr);
  const loom_link_module_index_symbol_t* callee_decl =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("callee"));
  const loom_link_module_index_symbol_t* callee_def =
      loom_link_module_index_lookup_private(index.get(), library_module,
                                            IREE_SV("callee"));
  const loom_link_module_index_symbol_t* helper =
      loom_link_module_index_lookup_private(index.get(), library_module,
                                            IREE_SV("helper"));
  EXPECT_TRUE(ContainsSymbol(plan.get(), callee_decl));
  EXPECT_FALSE(ContainsSymbol(plan.get(), callee_def));
  EXPECT_FALSE(ContainsSymbol(plan.get(), helper));
}

TEST_F(LinkPlannerTest, RuntimeImportDoesNotResolveFromLoomLibrary) {
  loom_module_t* harness = Parse(Fixture(IREE_SV(
      "runtime_import_does_not_resolve_from_loom_library_harness.loom")));
  loom_module_t* library = Parse(Fixture(IREE_SV(
      "runtime_import_does_not_resolve_from_loom_library_library.loom")));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), library, IREE_SV("library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_symbol_t* import_decl =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("callee"));
  ASSERT_NE(import_decl, nullptr);
  const loom_link_module_index_symbol_t* library_def =
      loom_link_module_index_next_global_duplicate(index.get(), import_decl);
  ASSERT_NE(library_def, nullptr);
  EXPECT_TRUE(ContainsSymbol(plan.get(), import_decl));
  EXPECT_FALSE(ContainsSymbol(plan.get(), library_def));
}

TEST_F(LinkPlannerTest, LinkApplyRequiresExplicitProviderSelection) {
  loom_module_t* harness = Parse(Fixture(
      IREE_SV("link_apply_requires_explicit_provider_selection_harness.loom")));
  loom_module_t* library = Parse(Fixture(
      IREE_SV("link_apply_requires_explicit_provider_selection_library.loom")));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), library, IREE_SV("library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/
      {
          /*.count=*/IREE_ARRAYSIZE(roots),
          /*.values=*/roots,
      },
  };
  PlanPtr demand_plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("entry"));
  const loom_link_module_index_module_t* library_module =
      loom_link_module_index_module_at(index.get(), 1);
  ASSERT_NE(library_module, nullptr);
  const loom_link_module_index_symbol_t* gfx11 =
      loom_link_module_index_lookup_private(index.get(), library_module,
                                            IREE_SV("gfx11"));
  const loom_link_module_index_symbol_t* gfx12 =
      loom_link_module_index_lookup_private(index.get(), library_module,
                                            IREE_SV("gfx12"));
  const loom_link_module_index_symbol_t* gfx11_provider =
      loom_link_module_index_lookup_private(index.get(), library_module,
                                            IREE_SV("gfx11_provider"));
  const loom_link_module_index_symbol_t* gfx12_provider =
      loom_link_module_index_lookup_private(index.get(), library_module,
                                            IREE_SV("gfx12_provider"));
  const loom_link_module_index_symbol_t* fallback_provider =
      loom_link_module_index_lookup_private(index.get(), library_module,
                                            IREE_SV("fallback_provider"));
  const loom_link_module_index_symbol_t* unused_provider =
      loom_link_module_index_lookup_private(index.get(), library_module,
                                            IREE_SV("unused_provider"));

  EXPECT_TRUE(ContainsSymbol(demand_plan.get(), entry));
  EXPECT_FALSE(ContainsSymbol(demand_plan.get(), gfx11_provider));
  EXPECT_FALSE(ContainsSymbol(demand_plan.get(), gfx12_provider));
  EXPECT_FALSE(ContainsSymbol(demand_plan.get(), fallback_provider));
  EXPECT_FALSE(ContainsSymbol(demand_plan.get(), unused_provider));
  EXPECT_FALSE(ContainsSymbol(demand_plan.get(), gfx11));
  EXPECT_FALSE(ContainsSymbol(demand_plan.get(), gfx12));
  ASSERT_EQ(loom_link_plan_demanded_template_family_count(demand_plan.get()),
            1u);
  const loom_link_module_index_template_family_t* demanded_family =
      DemandedTemplateFamily(demand_plan.get(), 0);
  ASSERT_NE(demanded_family, nullptr);
  EXPECT_EQ(StringViewToString(demanded_family->name), "demo.targeted");

  const iree_host_size_t provider_root_ordinals[] = {
      gfx11_provider->ordinal,
  };
  options.template_provider_roots = {
      /*.count=*/IREE_ARRAYSIZE(provider_root_ordinals),
      /*.values=*/provider_root_ordinals,
  };
  PlanPtr rooted_plan = BuildPlan(index.get(), &options);
  EXPECT_TRUE(ContainsSymbol(rooted_plan.get(), entry));
  EXPECT_TRUE(ContainsSymbol(rooted_plan.get(), gfx11_provider));
  EXPECT_FALSE(ContainsSymbol(rooted_plan.get(), gfx12_provider));
  EXPECT_FALSE(ContainsSymbol(rooted_plan.get(), fallback_provider));
  EXPECT_FALSE(ContainsSymbol(rooted_plan.get(), unused_provider));
  EXPECT_TRUE(ContainsSymbol(rooted_plan.get(), gfx11));
  EXPECT_FALSE(ContainsSymbol(rooted_plan.get(), gfx12));
  ASSERT_EQ(loom_link_plan_demanded_template_family_count(rooted_plan.get()),
            1u);

  const loom_link_plan_symbol_t* planned_gfx11_provider =
      FindPlannedSymbol(rooted_plan.get(), gfx11_provider);
  ASSERT_NE(planned_gfx11_provider, nullptr);
  EXPECT_EQ(planned_gfx11_provider->reason, LOOM_LINK_PLAN_LIVE_PROVIDER);
  EXPECT_EQ(planned_gfx11_provider->cause_ordinal,
            LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL);
}

TEST_F(LinkPlannerTest, ProviderRootsExposeTransitiveDiamondDemands) {
  loom_module_t* harness = Parse(Fixture(IREE_SV(
      "selected_providers_expose_transitive_diamond_demands_harness.loom")));
  loom_module_t* library = Parse(Fixture(IREE_SV(
      "selected_providers_expose_transitive_diamond_demands_library.loom")));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), library, IREE_SV("library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  const loom_link_module_index_module_t* library_module =
      loom_link_module_index_module_at(index.get(), 1);
  ASSERT_NE(library_module, nullptr);
  const loom_link_module_index_symbol_t* left_provider =
      loom_link_module_index_lookup_private(index.get(), library_module,
                                            IREE_SV("left_provider"));
  const loom_link_module_index_symbol_t* right_provider =
      loom_link_module_index_lookup_private(index.get(), library_module,
                                            IREE_SV("right_provider"));
  const loom_link_module_index_symbol_t* shared_provider =
      loom_link_module_index_lookup_private(index.get(), library_module,
                                            IREE_SV("shared_provider"));
  const loom_link_module_index_symbol_t* helper =
      loom_link_module_index_lookup_private(index.get(), library_module,
                                            IREE_SV("helper"));
  ASSERT_NE(left_provider, nullptr);
  ASSERT_NE(right_provider, nullptr);
  ASSERT_NE(shared_provider, nullptr);
  ASSERT_NE(helper, nullptr);

  iree_string_view_t roots[] = {IREE_SV("@entry")};
  const iree_host_size_t first_provider_ordinals[] = {
      left_provider->ordinal,
      right_provider->ordinal,
  };
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  options.template_provider_roots = {
      /*.count=*/IREE_ARRAYSIZE(first_provider_ordinals),
      /*.values=*/first_provider_ordinals,
  };
  PlanPtr first_plan = BuildPlan(index.get(), &options);

  EXPECT_TRUE(ContainsSymbol(first_plan.get(), left_provider));
  EXPECT_TRUE(ContainsSymbol(first_plan.get(), right_provider));
  EXPECT_FALSE(ContainsSymbol(first_plan.get(), shared_provider));
  EXPECT_FALSE(ContainsSymbol(first_plan.get(), helper));
  ASSERT_EQ(loom_link_plan_demanded_template_family_count(first_plan.get()),
            3u);
  EXPECT_EQ(loom_link_plan_template_demand_occurrence_count(first_plan.get()),
            4u);
  const std::string first_family =
      StringViewToString(DemandedTemplateFamily(first_plan.get(), 0)->name);
  const std::string second_family =
      StringViewToString(DemandedTemplateFamily(first_plan.get(), 1)->name);
  EXPECT_TRUE((first_family == "demo.left" && second_family == "demo.right") ||
              (first_family == "demo.right" && second_family == "demo.left"));
  EXPECT_EQ(
      StringViewToString(DemandedTemplateFamily(first_plan.get(), 2)->name),
      "demo.shared");

  const iree_host_size_t complete_provider_ordinals[] = {
      left_provider->ordinal,
      right_provider->ordinal,
      shared_provider->ordinal,
      shared_provider->ordinal,
  };
  options.template_provider_roots = {
      /*.count=*/IREE_ARRAYSIZE(complete_provider_ordinals),
      /*.values=*/complete_provider_ordinals,
  };
  PlanPtr complete_plan = BuildPlan(index.get(), &options);

  EXPECT_TRUE(ContainsSymbol(complete_plan.get(), shared_provider));
  EXPECT_TRUE(ContainsSymbol(complete_plan.get(), helper));
  // Source-local family declarations remain in the exact closure needed to
  // decode each provider module and merge to canonical output identities.
  EXPECT_EQ(loom_link_plan_symbol_count(complete_plan.get()), 10u);
  EXPECT_EQ(loom_link_plan_demanded_template_family_count(complete_plan.get()),
            3u);
  EXPECT_EQ(
      loom_link_plan_template_demand_occurrence_count(complete_plan.get()), 4u);
}

TEST_F(LinkPlannerTest, ProviderRootOrdinalsAreValidated) {
  loom_module_t* module = Parse(
      Fixture(IREE_SV("selected_provider_ordinals_are_validated_module.loom")));
  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), module, IREE_SV("input"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("entry"));
  ASSERT_NE(entry, nullptr);

  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  options.template_provider_roots = {
      /*.count=*/1,
      /*.values=*/nullptr,
  };
  PlanPtr plan;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        BuildPlanStatus(index.get(), &options, &plan));

  const iree_host_size_t out_of_range_ordinal =
      loom_link_module_index_symbol_count(index.get());
  options.template_provider_roots.values = &out_of_range_ordinal;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        BuildPlanStatus(index.get(), &options, &plan));

  options.template_provider_roots.values = &entry->ordinal;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        BuildPlanStatus(index.get(), &options, &plan));
}

TEST_F(LinkPlannerTest, LinkRootIgnoresUnreachableDuplicateDefinition) {
  loom_module_t* harness = Parse(Fixture(IREE_SV(
      "link_root_ignores_unreachable_duplicate_definition_harness.loom")));
  loom_module_t* first = Parse(Fixture(IREE_SV(
      "link_root_ignores_unreachable_duplicate_definition_first.loom")));
  loom_module_t* second = Parse(Fixture(IREE_SV(
      "link_root_ignores_unreachable_duplicate_definition_second.loom")));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), first, IREE_SV("first"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  AddMaterialized(index.get(), second, IREE_SV("second"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("entry"));
  const loom_link_module_index_symbol_t* unused =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("unused"));
  ASSERT_TRUE(ContainsSymbol(plan.get(), entry));
  EXPECT_FALSE(ContainsSymbol(plan.get(), unused));
  EXPECT_EQ(loom_link_plan_symbol_count(plan.get()), 1u);
}

TEST_F(LinkPlannerTest, InputExportRejectsAmbiguousLibraryDefinitions) {
  loom_module_t* harness = Parse(Fixture(IREE_SV(
      "input_export_rejects_ambiguous_library_definitions_harness.loom")));
  loom_module_t* first = Parse(Fixture(IREE_SV(
      "input_export_rejects_ambiguous_library_definitions_first.loom")));
  loom_module_t* second = Parse(Fixture(IREE_SV(
      "input_export_rejects_ambiguous_library_definitions_second.loom")));

  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{},
      /*.include_input_exports=*/true,
  };
  for (bool reverse_libraries : {false, true}) {
    IndexPtr index = CreateIndex();
    AddMaterialized(index.get(), harness, IREE_SV("harness"),
                    LOOM_LINK_PROVIDER_ROLE_INPUT);
    AddMaterialized(index.get(), reverse_libraries ? second : first,
                    reverse_libraries ? IREE_SV("second") : IREE_SV("first"),
                    LOOM_LINK_PROVIDER_ROLE_LIBRARY);
    AddMaterialized(index.get(), reverse_libraries ? first : second,
                    reverse_libraries ? IREE_SV("first") : IREE_SV("second"),
                    LOOM_LINK_PROVIDER_ROLE_LIBRARY);
    PlanPtr plan;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                          BuildPlanStatus(index.get(), &options, &plan));
  }
}

TEST_F(LinkPlannerTest, ExplicitLibraryRootRejectsAmbiguousDefinitions) {
  loom_module_t* first = Parse(Fixture(IREE_SV(
      "explicit_library_root_rejects_ambiguous_definitions_first.loom")));
  loom_module_t* second = Parse(Fixture(IREE_SV(
      "explicit_library_root_rejects_ambiguous_definitions_second.loom")));
  iree_string_view_t roots[] = {IREE_SV("@same")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };

  for (bool reverse_libraries : {false, true}) {
    IndexPtr index = CreateIndex();
    AddMaterialized(index.get(), reverse_libraries ? second : first,
                    reverse_libraries ? IREE_SV("second") : IREE_SV("first"),
                    LOOM_LINK_PROVIDER_ROLE_LIBRARY);
    AddMaterialized(index.get(), reverse_libraries ? first : second,
                    reverse_libraries ? IREE_SV("first") : IREE_SV("second"),
                    LOOM_LINK_PROVIDER_ROLE_LIBRARY);
    PlanPtr plan;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                          BuildPlanStatus(index.get(), &options, &plan));
  }
}

TEST_F(LinkPlannerTest, OwnerDefinitionDoesNotExtractLibraryAlternatives) {
  loom_module_t* harness = Parse(Fixture(IREE_SV(
      "owner_definition_does_not_extract_library_alternatives_harness.loom")));
  loom_module_t* sibling_source =
      Parse(Fixture(IREE_SV("owner_definition_does_not_extract_library_"
                            "alternatives_sibling_source.loom")));
  loom_module_t* first_library =
      Parse(Fixture(IREE_SV("owner_definition_does_not_extract_library_"
                            "alternatives_first_library.loom")));
  loom_module_t* second_library =
      Parse(Fixture(IREE_SV("owner_definition_does_not_extract_library_"
                            "alternatives_second_library.loom")));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), first_library, IREE_SV("first-library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  AddMaterialized(index.get(), second_library, IREE_SV("second-library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  AddMaterialized(index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), sibling_source, IREE_SV("sibling-source"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_module_t* sibling_module =
      loom_link_module_index_module_at(index.get(), 3);
  ASSERT_NE(sibling_module, nullptr);
  const loom_link_module_index_symbol_t* owner_definition =
      loom_link_module_index_lookup_private(index.get(), sibling_module,
                                            IREE_SV("same"));
  EXPECT_TRUE(ContainsSymbol(plan.get(), owner_definition));
  EXPECT_EQ(loom_link_plan_symbol_count(plan.get()), 3u);
}

TEST_F(LinkPlannerTest, LinkReportsMissingRoot) {
  loom_module_t* module =
      Parse(Fixture(IREE_SV("link_reports_missing_root_module.loom")));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), module, IREE_SV("input"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  iree_string_view_t roots[] = {IREE_SV("@missing")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };

  PlanPtr plan;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        BuildPlanStatus(index.get(), &options, &plan));
}

static bool StripNamedSymbol(void* user_data,
                             const loom_link_module_index_t* index,
                             const loom_link_module_index_symbol_t* symbol) {
  (void)index;
  iree_string_view_t* stripped_name = (iree_string_view_t*)user_data;
  return iree_string_view_equal(symbol->name, *stripped_name);
}

TEST_F(LinkPlannerTest, StripPolicyControlsRequiredDependencies) {
  loom_module_t* module = Parse(Fixture(
      IREE_SV("strip_policy_controls_required_dependencies_module.loom")));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), module, IREE_SV("input"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  iree_string_view_t stripped_name = IREE_SV("helper");
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
      /*.include_input_exports=*/{},
      /*.unresolved_policy=*/{},
      /*.test_symbol_policy=*/{},
      /*.strip_symbol=*/StripNamedSymbol,
      /*.strip_symbol_user_data=*/&stripped_name,
  };

  PlanPtr plan;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        BuildPlanStatus(index.get(), &options, &plan));

  options.unresolved_policy = LOOM_LINK_PLAN_UNRESOLVED_ALLOW;
  plan = BuildPlan(index.get(), &options);
  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("entry"));
  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_module_at(index.get(), 0);
  ASSERT_NE(indexed_module, nullptr);
  const loom_link_module_index_symbol_t* helper =
      loom_link_module_index_lookup_private(index.get(), indexed_module,
                                            IREE_SV("helper"));
  EXPECT_TRUE(ContainsSymbol(plan.get(), entry));
  EXPECT_FALSE(ContainsSymbol(plan.get(), helper));
}

TEST_F(LinkPlannerTest, TestSymbolStripPolicyFiltersImplicitExports) {
  loom_module_t* module = Parse(Fixture(IREE_SV(
      "test_symbol_strip_policy_removes_bytecode_symbols_module.loom")));
  ASSERT_NE(module, nullptr);
  std::vector<uint8_t> bytes = WriteModule(module);

  IndexPtr index = CreateIndex();
  loom_link_module_index_add_options_t provider_options = {
      /*.provider_name=*/IREE_SV("kernel-lib"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
  };
  iree_host_size_t provider_ordinal = 0;
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      index.get(), iree_make_const_byte_span(bytes.data(), bytes.size()),
      IREE_SV("kernel-lib.loombc"), /*index_options=*/nullptr,
      &provider_options, &provider_ordinal));
  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_module_at(index.get(), 0);
  ASSERT_NE(indexed_module, nullptr);
  EXPECT_EQ(indexed_module->materialized_module, nullptr);

  loom_link_plan_options_t strip_options = {
      /*.mode=*/LOOM_LINK_PLAN_MERGE,
      /*.root_symbols=*/{},
      /*.include_input_exports=*/{},
      /*.unresolved_policy=*/{},
      /*.test_symbol_policy=*/LOOM_LINK_PLAN_TEST_SYMBOL_STRIP,
  };
  PlanPtr plan = BuildPlan(index.get(), &strip_options);

  const loom_link_module_index_symbol_t* kernel =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("kernel"));
  const loom_link_module_index_symbol_t* check_case =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("kernel_case"));
  const loom_link_module_index_symbol_t* benchmark =
      loom_link_module_index_lookup_private(index.get(), indexed_module,
                                            IREE_SV("kernel_bench"));
  EXPECT_TRUE(ContainsSymbol(plan.get(), kernel));
  EXPECT_FALSE(ContainsSymbol(plan.get(), check_case));
  EXPECT_FALSE(ContainsSymbol(plan.get(), benchmark));

  strip_options.mode = LOOM_LINK_PLAN_LINK;
  strip_options.root_providers.count = 1;
  strip_options.root_providers.values = &provider_ordinal;
  plan = BuildPlan(index.get(), &strip_options);
  EXPECT_TRUE(ContainsSymbol(plan.get(), kernel));
  EXPECT_FALSE(ContainsSymbol(plan.get(), check_case));
  EXPECT_FALSE(ContainsSymbol(plan.get(), benchmark));
}

TEST_F(LinkPlannerTest, KeepTestSymbolPolicyPreservesDependencies) {
  loom_module_t* module = Parse(Fixture(
      IREE_SV("keep_test_symbol_policy_preserves_dependencies_module.loom")));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), module, IREE_SV("input"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  iree_string_view_t roots[] = {IREE_SV("@kernel_case")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_symbol_t* kernel =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("kernel"));
  const loom_link_module_index_symbol_t* check_case =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("kernel_case"));
  EXPECT_TRUE(ContainsSymbol(plan.get(), kernel));
  EXPECT_TRUE(ContainsSymbol(plan.get(), check_case));
}

TEST_F(LinkPlannerTest, InputTestRootsExcludeLibraryTests) {
  loom_module_t* root_module = Parse(IREE_SV(R"(
func.decl @library_helper(%x: i32) -> (i32)

check.case @root_case {
  %input = check.literal value(1) : i32
  %actual = func.call @library_helper(%input) : (i32) -> (i32)
  check.expect.equal actual(%actual) expected(%input) : i32
  check.return
}

check.benchmark<@root_case> @root_benchmark
)"),
                                     IREE_SV("root.loom"));
  loom_module_t* library_module = Parse(IREE_SV(R"(
func.def public @library_helper(%x: i32) -> (i32) {
  func.return %x : i32
}

check.case public @library_case {
  check.return
}

check.benchmark<@library_case> @library_benchmark
)"),
                                        IREE_SV("library.loom"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), root_module, IREE_SV("root"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), library_module, IREE_SV("library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
  };
  options.include_input_tests = true;
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_module_t* root_index_module =
      loom_link_module_index_module_at(index.get(), 0);
  const loom_link_module_index_module_t* library_index_module =
      loom_link_module_index_module_at(index.get(), 1);
  ASSERT_NE(root_index_module, nullptr);
  ASSERT_NE(library_index_module, nullptr);
  const loom_link_module_index_symbol_t* root_case =
      loom_link_module_index_lookup_private(index.get(), root_index_module,
                                            IREE_SV("root_case"));
  const loom_link_module_index_symbol_t* root_benchmark =
      loom_link_module_index_lookup_private(index.get(), root_index_module,
                                            IREE_SV("root_benchmark"));
  const loom_link_module_index_symbol_t* library_helper =
      loom_link_module_index_lookup_global(index.get(),
                                           IREE_SV("library_helper"));
  const loom_link_module_index_symbol_t* library_case =
      loom_link_module_index_lookup_global(index.get(),
                                           IREE_SV("library_case"));
  const loom_link_module_index_symbol_t* library_benchmark =
      loom_link_module_index_lookup_private(index.get(), library_index_module,
                                            IREE_SV("library_benchmark"));

  EXPECT_TRUE(ContainsSymbol(plan.get(), root_case));
  EXPECT_TRUE(ContainsSymbol(plan.get(), root_benchmark));
  EXPECT_TRUE(ContainsSymbol(plan.get(), library_helper));
  EXPECT_FALSE(ContainsSymbol(plan.get(), library_case));
  EXPECT_FALSE(ContainsSymbol(plan.get(), library_benchmark));
}

TEST_F(LinkPlannerTest, TestSymbolStripPolicyRejectsStrippedRoots) {
  loom_module_t* module = Parse(Fixture(
      IREE_SV("test_symbol_strip_policy_rejects_stripped_roots_module.loom")));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), module, IREE_SV("input"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  iree_string_view_t roots[] = {IREE_SV("@kernel_case")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
      /*.include_input_exports=*/{},
      /*.unresolved_policy=*/{},
      /*.test_symbol_policy=*/LOOM_LINK_PLAN_TEST_SYMBOL_STRIP,
  };

  PlanPtr plan;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        BuildPlanStatus(index.get(), &options, &plan));
}

TEST_F(LinkPlannerTest, ExportedRootPolicySelectsExportsAndDependencies) {
  loom_module_t* module = Parse(Fixture(IREE_SV(
      "exported_root_policy_selects_exports_and_dependencies_module.loom")));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), module, IREE_SV("input"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{},
      /*.include_input_exports=*/true,
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("entry"));
  const loom_link_module_index_symbol_t* second =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("second"));
  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_module_at(index.get(), 0);
  ASSERT_NE(indexed_module, nullptr);
  const loom_link_module_index_symbol_t* helper =
      loom_link_module_index_lookup_private(index.get(), indexed_module,
                                            IREE_SV("helper"));
  EXPECT_TRUE(ContainsSymbol(plan.get(), entry));
  EXPECT_TRUE(ContainsSymbol(plan.get(), second));
  EXPECT_TRUE(ContainsSymbol(plan.get(), helper));
}

}  // namespace
}  // namespace loom
