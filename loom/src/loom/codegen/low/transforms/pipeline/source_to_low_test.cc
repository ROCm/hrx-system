// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/pipeline/source_to_low.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower/lower_rules.h"
#include "loom/codegen/low/lower/source_selection.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/error/error_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/pass/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/pass/registry.h"
#include "loom/pass/tooling.h"
#include "loom/pass/value_facts.h"
#include "loom/target/test/contracts/core_lower_rules.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"
#include "loom/target/test/target_records.h"
#include "loom/testing/module_ptr.h"
#include "loom/transforms/symbol/inline_callables.h"
#include "loom/transforms/symbol/symbol_dce.h"
#include "loom/transforms/symbol/template_selection.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

struct DiagnosticEmissionCollector {
  int count = 0;
  const loom_error_def_t* last_error = nullptr;
};

static iree_status_t CollectDiagnosticEmission(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  auto* collector = static_cast<DiagnosticEmissionCollector*>(user_data);
  ++collector->count;
  collector->last_error = emission->error;
  return iree_ok_status();
}

class LowLowerPassTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_PASS, loom_pass_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&registry_);
    loom_test_low_lower_policy_registry_initialize(&policy_registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
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

  ModulePtr Parse(iree_string_view_t source) {
    loom_text_parse_options_t parse_options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(source, IREE_SV("source_to_low_test.loom"),
                                  &context_, &block_pool_, &parse_options,
                                  &module));
    IREE_ASSERT(module != nullptr);
    return ModulePtr(module);
  }

  loom_symbol_ref_t FindSymbolRef(const loom_module_t* module,
                                  iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    const uint16_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return (loom_symbol_ref_t){/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  }

  iree_status_t RunSourceToLow(
      loom_low_lower_policy_registry_t* policy_registry, loom_module_t* module,
      DiagnosticEmissionCollector* collector = nullptr,
      loom_target_selection_t target_selection = loom_target_selection_empty(),
      loom_symbol_ref_t target_ref = loom_symbol_ref_null()) {
    iree_arena_allocator_t instance_arena;
    iree_arena_initialize(&block_pool_, &instance_arena);
    loom_pass_value_fact_owner_t value_facts = {};
    loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts);
    const loom_pass_info_t* pass_info = loom_low_source_to_low_pass_info();
    std::vector<uint8_t> statistic_storage(
        pass_info->statistic_layout->storage_size, 0);
    loom_low_pass_environment_storage_t low_pass_environment_storage;
    loom_pass_environment_t environment =
        loom_low_pass_environment_storage_initialize(
            &registry_.registry, policy_registry, nullptr, nullptr, nullptr,
            nullptr, target_selection, target_ref,
            &low_pass_environment_storage);
    loom_pass_t pass = {};
    pass.info = pass_info;
    pass.module_run = loom_low_source_to_low_run;
    pass.instance_arena = &instance_arena;
    pass.arena = &instance_arena;
    pass.statistic_storage = statistic_storage.data();
    pass.environment = &environment;
    pass.value_facts = &value_facts;
    if (collector != nullptr) {
      pass.diagnostic_emitter = {
          /*.fn=*/CollectDiagnosticEmission,
          /*.user_data=*/collector,
      };
    }

    iree_status_t status =
        loom_low_source_to_low_create(&pass, iree_string_view_empty());
    if (iree_status_is_ok(status)) {
      status = loom_low_source_to_low_run(&pass, module);
    }
    loom_pass_value_fact_owner_deinitialize(&value_facts);
    iree_arena_deinitialize(&instance_arena);
    return status;
  }

  iree_status_t RunFlatPipeline(
      loom_module_t* module, iree_string_view_t pipeline,
      loom_target_selection_t target_selection = loom_target_selection_empty(),
      loom_symbol_ref_t target_ref = loom_symbol_ref_null(),
      DiagnosticEmissionCollector* collector = nullptr) {
    static const loom_pass_descriptor_t kPassDescriptors[] = {
        {
            /*.key=*/IREE_SVL("inline-callables"),
            /*.info=*/loom_inline_callables_pass_info,
            /*.module_run=*/{loom_inline_callables_run},
            /*.create=*/nullptr,
            /*.destroy=*/nullptr,
            /*.flags=*/0,
            /*.unavailable_reason=*/{},
            /*.option_schema=*/nullptr,
            /*.option_schema_count=*/0,
            /*.requirement_defs=*/nullptr,
            /*.requirement_count=*/0,
        },
        {
            /*.key=*/IREE_SVL("select-templates"),
            /*.info=*/loom_template_selection_pass_info,
            /*.module_run=*/{loom_template_selection_run},
            /*.create=*/loom_template_selection_create,
            /*.destroy=*/nullptr,
            /*.flags=*/0,
            /*.unavailable_reason=*/{},
            /*.option_schema=*/nullptr,
            /*.option_schema_count=*/0,
            /*.requirement_defs=*/nullptr,
            /*.requirement_count=*/0,
        },
        {
            /*.key=*/IREE_SVL("source-to-low"),
            /*.info=*/loom_low_source_to_low_pass_info,
            /*.module_run=*/{loom_low_source_to_low_run},
            /*.create=*/loom_low_source_to_low_create,
            /*.destroy=*/nullptr,
            /*.flags=*/0,
            /*.unavailable_reason=*/{},
            /*.option_schema=*/nullptr,
            /*.option_schema_count=*/0,
            /*.requirement_defs=*/nullptr,
            /*.requirement_count=*/0,
        },
        {
            /*.key=*/IREE_SVL("symbol-dce"),
            /*.info=*/loom_symbol_dce_pass_info,
            /*.module_run=*/{loom_symbol_dce_run},
            /*.create=*/nullptr,
            /*.destroy=*/nullptr,
            /*.flags=*/0,
            /*.unavailable_reason=*/{},
            /*.option_schema=*/nullptr,
            /*.option_schema_count=*/0,
            /*.requirement_defs=*/nullptr,
            /*.requirement_count=*/0,
        },
    };
    static const loom_pass_registry_t kPassRegistry = {
        /*.descriptors=*/kPassDescriptors,
        /*.descriptor_count=*/IREE_ARRAYSIZE(kPassDescriptors),
    };

    loom_low_pass_environment_storage_t low_pass_environment_storage;
    loom_pass_environment_t environment =
        loom_low_pass_environment_storage_initialize(
            &registry_.registry, &policy_registry_, nullptr, nullptr, nullptr,
            nullptr, target_selection, target_ref,
            &low_pass_environment_storage);
    loom_pass_tool_run_options_t run_options = {
        /*.registry=*/&kPassRegistry,
        /*.environment=*/environment,
        /*.predicate_provider=*/{},
        /*.block_pool=*/&block_pool_,
    };
    if (collector != nullptr) {
      run_options.diagnostic_emitter = {
          /*.fn=*/CollectDiagnosticEmission,
          /*.user_data=*/collector,
      };
    }
    loom_pass_run_result_t run_result = {};
    IREE_RETURN_IF_ERROR(loom_pass_tool_run_flat_pipeline(
        module, pipeline, &run_options, &run_result));
    if (run_result.error_count > 0) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "pass pipeline emitted errors");
    }
    return iree_ok_status();
  }

  bool HasSymbol(const loom_module_t* module, iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    if (name_id == LOOM_STRING_ID_INVALID) {
      return false;
    }
    return loom_module_find_symbol(module, name_id) != LOOM_SYMBOL_ID_INVALID;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t registry_ = {};
  loom_low_lower_policy_registry_t policy_registry_ = {};
};

static loom_target_bundle_storage_t CopyTargetBundle(
    const loom_target_bundle_t* bundle) {
  loom_target_bundle_storage_t storage = {
      /*.snapshot=*/*bundle->snapshot,
      /*.export_plan=*/*bundle->export_plan,
      /*.config=*/*bundle->config,
      /*.bundle=*/*bundle,
  };
  loom_target_bundle_storage_rebind(&storage);
  return storage;
}

TEST_F(LowLowerPassTest,
       SourceSelectionAppliesCompatibleRuntimeTargetSelection) {
  ModulePtr module = Parse(IREE_SV(
      "test.target<low_core> @test_target\n"
      "func.def target(@test_target) @add(%lhs: i32, %rhs: i32) -> (i32) {\n"
      "  %sum = scalar.addi %lhs, %rhs : i32\n"
      "  func.return %sum : i32\n"
      "}\n"));
  ASSERT_GT(loom_test_target_bundles.count, 2u);
  const int target_payload = 42;
  loom_target_bundle_storage_t selected_storage =
      CopyTargetBundle(loom_test_target_bundles.values[2]);

  loom_low_lower_policy_registry_t policy_registry = {};
  loom_test_low_lower_policy_registry_initialize(&policy_registry);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_source_selection_options_t options = {
      /*.policy_registry=*/&policy_registry,
      /*.diagnostic_emitter=*/{},
      /*.lowering_kind=*/{},
      /*.target_selection=*/
      {
          /*.bundle=*/&selected_storage.bundle,
          /*.data=*/&target_payload,
      },
  };
  loom_low_source_selection_list_t selections = {};
  IREE_ASSERT_OK(loom_low_select_source_symbols(module.get(), &options, &arena,
                                                &selections));

  ASSERT_EQ(selections.count, 1u);
  const loom_symbol_ref_t target_ref =
      FindSymbolRef(module.get(), IREE_SV("test_target"));
  EXPECT_EQ(selections.values[0].target_ref.module_id, target_ref.module_id);
  EXPECT_EQ(selections.values[0].target_ref.symbol_id, target_ref.symbol_id);
  EXPECT_TRUE(
      iree_string_view_equal(selections.values[0].target_bundle->snapshot->name,
                             IREE_SV("test-quirky")));
  EXPECT_EQ(selections.values[0].target_data, &target_payload);
  iree_arena_deinitialize(&arena);
}

TEST_F(LowLowerPassTest, SourceSelectionUsesInvocationTargetRef) {
  ModulePtr module =
      Parse(IREE_SV("test.target<low_core> @test_target\n"
                    "func.def @add(%lhs: i32, %rhs: i32) -> (i32) {\n"
                    "  %sum = scalar.addi %lhs, %rhs : i32\n"
                    "  func.return %sum : i32\n"
                    "}\n"));
  const loom_symbol_ref_t target_ref =
      FindSymbolRef(module.get(), IREE_SV("test_target"));

  loom_low_lower_policy_registry_t policy_registry = {};
  loom_test_low_lower_policy_registry_initialize(&policy_registry);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_source_selection_options_t options = {
      /*.policy_registry=*/&policy_registry,
      /*.diagnostic_emitter=*/{},
      /*.lowering_kind=*/{},
      /*.target_selection=*/{},
      /*.target_ref=*/target_ref,
  };
  loom_low_source_selection_list_t selections = {};
  IREE_ASSERT_OK(loom_low_select_source_symbols(module.get(), &options, &arena,
                                                &selections));

  ASSERT_EQ(selections.count, 1u);
  EXPECT_EQ(selections.values[0].target_ref.module_id, target_ref.module_id);
  EXPECT_EQ(selections.values[0].target_ref.symbol_id, target_ref.symbol_id);
  EXPECT_TRUE(iree_string_view_equal(selections.values[0].target_bundle->name,
                                     IREE_SV("test_target")));
  iree_arena_deinitialize(&arena);
}

TEST_F(LowLowerPassTest, SourceToLowUsesInvocationTargetRef) {
  ModulePtr module =
      Parse(IREE_SV("test.target<low_core> @test_target\n"
                    "func.def @add(%lhs: i32, %rhs: i32) -> (i32) {\n"
                    "  %sum = scalar.addi %lhs, %rhs : i32\n"
                    "  func.return %sum : i32\n"
                    "}\n"));
  const loom_symbol_ref_t target_ref =
      FindSymbolRef(module.get(), IREE_SV("test_target"));

  loom_low_lower_policy_registry_t policy_registry = {};
  loom_test_low_lower_policy_registry_initialize(&policy_registry);
  DiagnosticEmissionCollector collector;
  IREE_ASSERT_OK(RunSourceToLow(&policy_registry, module.get(), &collector,
                                loom_target_selection_empty(), target_ref));
  EXPECT_EQ(collector.count, 0);

  const loom_symbol_ref_t add_ref = FindSymbolRef(module.get(), IREE_SV("add"));
  const loom_symbol_t* add_symbol = &module->symbols.entries[add_ref.symbol_id];
  ASSERT_NE(add_symbol->defining_op, nullptr);
  ASSERT_TRUE(loom_low_func_def_isa(add_symbol->defining_op));
  const loom_symbol_ref_t lowered_target =
      loom_low_func_def_target(add_symbol->defining_op);
  EXPECT_EQ(lowered_target.module_id, target_ref.module_id);
  EXPECT_EQ(lowered_target.symbol_id, target_ref.symbol_id);
}

TEST_F(LowLowerPassTest,
       InvocationTargetPrunesOffTargetProvidersBeforeLowering) {
  ModulePtr module = Parse(IREE_SV(
      "func.template<demo.targeted> target(@gfx12) priority(20) "
      "@gfx12_bad(%value: i32) -> (i32) {\n"
      "  test.use %value : i32\n"
      "  func.return %value : i32\n"
      "}\n"
      "\n"
      "func.template<demo.targeted> target(@gfx11) priority(10) "
      "@gfx11_good(%value: i32) -> (i32) {\n"
      "  %doubled = scalar.addi %value, %value : i32\n"
      "  func.return %doubled : i32\n"
      "}\n"
      "\n"
      "func.template<demo.targeted> priority(1) @fallback(%value: i32) -> "
      "(i32) {\n"
      "  func.return %value : i32\n"
      "}\n"
      "\n"
      "test.target<low_core> @gfx11\n"
      "test.target<low_core> @gfx12\n"
      "\n"
      "func.def public @entry(%arg: i32) -> (i32) {\n"
      "  %result = func.apply<demo.targeted>(%arg) : (i32) -> (i32)\n"
      "  func.return %result : i32\n"
      "}\n"));
  const loom_symbol_ref_t gfx11_ref =
      FindSymbolRef(module.get(), IREE_SV("gfx11"));

  IREE_ASSERT_OK(RunFlatPipeline(
      module.get(), IREE_SV("select-templates,inline-callables,symbol-dce"),
      loom_target_selection_empty(), gfx11_ref));
  EXPECT_FALSE(HasSymbol(module.get(), IREE_SV("gfx12_bad")));
  EXPECT_FALSE(HasSymbol(module.get(), IREE_SV("fallback")));

  IREE_ASSERT_OK(RunFlatPipeline(module.get(), IREE_SV("source-to-low"),
                                 loom_target_selection_empty(), gfx11_ref));
  const loom_symbol_ref_t entry_ref =
      FindSymbolRef(module.get(), IREE_SV("entry"));
  ASSERT_TRUE(loom_symbol_ref_is_valid(entry_ref));
  const loom_symbol_t* entry_symbol =
      &module->symbols.entries[entry_ref.symbol_id];
  ASSERT_NE(entry_symbol->defining_op, nullptr);
  ASSERT_TRUE(loom_low_func_def_isa(entry_symbol->defining_op));
  const loom_symbol_ref_t lowered_target =
      loom_low_func_def_target(entry_symbol->defining_op);
  EXPECT_EQ(lowered_target.module_id, gfx11_ref.module_id);
  EXPECT_EQ(lowered_target.symbol_id, gfx11_ref.symbol_id);
}

TEST_F(LowLowerPassTest, SourceSelectionAppliesRuntimeTargetDataOnly) {
  ModulePtr module = Parse(IREE_SV(
      "test.target<low_core> @test_target\n"
      "func.def target(@test_target) @add(%lhs: i32, %rhs: i32) -> (i32) {\n"
      "  %sum = scalar.addi %lhs, %rhs : i32\n"
      "  func.return %sum : i32\n"
      "}\n"));
  const int target_payload = 42;

  loom_low_lower_policy_registry_t policy_registry = {};
  loom_test_low_lower_policy_registry_initialize(&policy_registry);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_source_selection_options_t options = {
      /*.policy_registry=*/&policy_registry,
      /*.diagnostic_emitter=*/{},
      /*.lowering_kind=*/{},
      /*.target_selection=*/
      {
          /*.bundle=*/NULL,
          /*.data=*/&target_payload,
      },
  };
  loom_low_source_selection_list_t selections = {};
  IREE_ASSERT_OK(loom_low_select_source_symbols(module.get(), &options, &arena,
                                                &selections));

  ASSERT_EQ(selections.count, 1u);
  EXPECT_TRUE(iree_string_view_equal(selections.values[0].target_bundle->name,
                                     IREE_SV("test_target")));
  EXPECT_EQ(selections.values[0].target_data, &target_payload);
  iree_arena_deinitialize(&arena);
}

TEST_F(LowLowerPassTest,
       SourceSelectionRejectsIncompatibleRuntimeTargetSelection) {
  ModulePtr module = Parse(IREE_SV(
      "test.target<low_core> @test_target\n"
      "func.def target(@test_target) @add(%lhs: i32, %rhs: i32) -> (i32) {\n"
      "  %sum = scalar.addi %lhs, %rhs : i32\n"
      "  func.return %sum : i32\n"
      "}\n"));
  ASSERT_GT(loom_test_target_bundles.count, 2u);
  const int target_payload = 42;
  loom_target_bundle_storage_t selected_storage =
      CopyTargetBundle(loom_test_target_bundles.values[2]);
  selected_storage.snapshot.codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LLVMIR;

  loom_low_lower_policy_registry_t policy_registry = {};
  loom_test_low_lower_policy_registry_initialize(&policy_registry);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  DiagnosticEmissionCollector collector;
  loom_low_source_selection_options_t options = {
      /*.policy_registry=*/&policy_registry,
      /*.diagnostic_emitter=*/
      {
          /*.fn=*/CollectDiagnosticEmission,
          /*.user_data=*/&collector,
      },
      /*.lowering_kind=*/{},
      /*.target_selection=*/
      {
          /*.bundle=*/&selected_storage.bundle,
          /*.data=*/&target_payload,
      },
      /*.target_ref=*/loom_symbol_ref_null(),
  };
  loom_low_source_selection_list_t selections = {};
  IREE_ASSERT_OK(loom_low_select_source_symbols(module.get(), &options, &arena,
                                                &selections));

  EXPECT_EQ(selections.count, 0u);
  EXPECT_EQ(collector.count, 1);
  EXPECT_EQ(collector.last_error, LOOM_ERR_TARGET_052);
  iree_arena_deinitialize(&arena);
}

TEST_F(LowLowerPassTest,
       ContractFragmentDrivesRuleSelectionWithoutLegacySpans) {
  ModulePtr module = Parse(IREE_SV(
      "test.target<low_core> @test_target\n"
      "func.def target(@test_target) @add(%lhs: i32, %rhs: i32) -> (i32) {\n"
      "  %sum = scalar.addi %lhs, %rhs : i32\n"
      "  func.return %sum : i32\n"
      "}\n"));

  loom_low_lower_rule_set_t no_span_rule_set =
      loom_test_low_core_lower_rule_set;
  no_span_rule_set.spans = nullptr;
  no_span_rule_set.span_count = 0;
  const loom_low_lower_rule_set_t* rule_sets[] = {
      &no_span_rule_set,
  };
  loom_low_lower_policy_t policy = *loom_test_low_lower_policy();
  policy.rule_sets = {
      /*.count=*/IREE_ARRAYSIZE(rule_sets),
      /*.values=*/rule_sets,
  };
  const loom_low_lower_policy_registry_entry_t entries[] = {
      {
          /*.contract_set_key=*/IREE_SVL("test.low.core"),
          /*.policy=*/&policy,
      },
  };
  loom_low_lower_policy_registry_t policy_registry = {};
  loom_low_lower_policy_registry_initialize_from_entries(
      &policy_registry, entries, IREE_ARRAYSIZE(entries));

  DiagnosticEmissionCollector collector;
  iree_status_t status =
      RunSourceToLow(&policy_registry, module.get(), &collector);
  EXPECT_EQ(collector.count, 0);
  IREE_ASSERT_OK(status);
}

}  // namespace
}  // namespace loom
