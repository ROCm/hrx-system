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
#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/lower/lower_rules.h"
#include "loom/codegen/low/lower/source_selection.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/codegen/low/transforms/allocation.h"
#include "loom/codegen/low/transforms/dce.h"
#include "loom/error/error_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/pass/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/target/facts.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/template/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/pass/registry.h"
#include "loom/pass/tooling.h"
#include "loom/pass/value_facts.h"
#include "loom/target/facts_builder.h"
#include "loom/target/function_contract.h"
#include "loom/target/function_version.h"
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

static loom_pass_descriptor_t MakeFunctionPassDescriptor(
    iree_string_view_t key, loom_pass_info_fn_t info,
    loom_function_pass_fn_t function_run,
    loom_pass_create_fn_t create = nullptr) {
  loom_pass_descriptor_t descriptor = {};
  descriptor.key = key;
  descriptor.info = info;
  descriptor.function_run = function_run;
  descriptor.create = create;
  return descriptor;
}

class LowLowerPassTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_PASS, loom_pass_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_CFG, loom_cfg_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEMPLATE, loom_template_dialect_vtables);
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
      const loom_function_version_list_t* function_versions = nullptr) {
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
            nullptr, /*target_environment=*/nullptr, function_versions,
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
      const loom_function_version_list_t* function_versions = nullptr,
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
        MakeFunctionPassDescriptor(IREE_SV("low-dce"), loom_low_dce_pass_info,
                                   loom_low_dce_run),
        MakeFunctionPassDescriptor(IREE_SV("low-materialize-allocation"),
                                   loom_low_materialize_allocation_pass_info,
                                   loom_low_materialize_allocation_run,
                                   loom_low_materialize_allocation_create),
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
            nullptr, /*target_environment=*/nullptr, function_versions,
            &low_pass_environment_storage);
    loom_pass_tool_run_options_t run_options = {
        /*.registry=*/&kPassRegistry,
        /*.environment=*/environment,
        /*.function_versions=*/function_versions,
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

TEST_F(LowLowerPassTest, SourceSelectionUsesPerFunctionTargetFacts) {
  ModulePtr module = Parse(IREE_SV(
      "test.target<low_core> @test_target\n"
      "func.def target(@test_target) @add(%lhs: i32, %rhs: i32) -> (i32) {\n"
      "  %sum = scalar.addi %lhs, %rhs : i32\n"
      "  func.return %sum : i32\n"
      "}\n"));
  ASSERT_GT(loom_test_target_bundles.count, 2u);

  loom_low_lower_policy_registry_t policy_registry = {};
  loom_test_low_lower_policy_registry_initialize(&policy_registry);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  const loom_symbol_ref_t target_ref =
      FindSymbolRef(module.get(), IREE_SV("test_target"));
  loom_symbol_fact_table_t symbol_facts = {};
  loom_symbol_fact_table_initialize(&symbol_facts, &arena);
  const loom_symbol_facts_base_t* base_target_facts = nullptr;
  IREE_ASSERT_OK(loom_symbol_fact_table_lookup_ref(
      &symbol_facts, module.get(), target_ref, &base_target_facts));
  const loom_target_symbol_facts_t* target_requirement_symbol_facts =
      loom_target_symbol_facts_cast(base_target_facts);
  ASSERT_NE(target_requirement_symbol_facts, nullptr);
  loom_target_facts_t* function_target_facts = nullptr;
  IREE_ASSERT_OK(loom_target_facts_builder_clone(
      target_requirement_symbol_facts->projection, &arena,
      &function_target_facts));
  loom_target_facts_builder_replace_bundle(loom_test_target_bundles.values[2],
                                           function_target_facts);
  function_target_facts->selector = LOOM_TEST_TARGET_KIND_QUIRKY;

  const loom_symbol_ref_t function_ref =
      FindSymbolRef(module.get(), IREE_SV("add"));
  loom_target_function_version_t function_version = {};
  function_version.base.type = &loom_target_function_version_type;
  function_version.base.function = loom_func_like_cast(
      module.get(),
      module->symbols.entries[function_ref.symbol_id].defining_op);
  function_version.authored_target_name = target_requirement_symbol_facts->name;
  function_version.target_requirement_facts =
      target_requirement_symbol_facts->projection;
  function_version.function_target_facts = function_target_facts;
  loom_function_version_t* function_version_values[] = {
      &function_version.base,
  };
  const loom_function_version_list_t function_versions = {
      /*.values=*/function_version_values,
      /*.count=*/IREE_ARRAYSIZE(function_version_values),
  };
  loom_low_source_selection_options_t options = {
      /*.policy_registry=*/&policy_registry,
      /*.diagnostic_emitter=*/{},
      /*.lowering_kind=*/{},
      /*.function_versions=*/&function_versions,
      /*.collect_target_candidates=*/false,
  };
  loom_low_source_selection_list_t selections = {};
  IREE_ASSERT_OK(loom_low_select_source_symbols(module.get(), &options, &arena,
                                                &selections));

  ASSERT_EQ(selections.count, 1u);
  EXPECT_EQ(selections.values[0].version_handle, &function_version.base);
  EXPECT_EQ(selections.values[0].target_ref.module_id, target_ref.module_id);
  EXPECT_EQ(selections.values[0].target_ref.symbol_id, target_ref.symbol_id);
  EXPECT_EQ(selections.values[0].target_facts, function_target_facts);
  EXPECT_EQ(selections.values[0].target_facts->selector,
            LOOM_TEST_TARGET_KIND_QUIRKY);
  EXPECT_TRUE(iree_string_view_equal(
      selections.values[0].target_facts->storage.config.contract_set_key,
      IREE_SV("test.low.core")));
  const loom_target_bundle_t* selected_bundle =
      loom_low_source_selection_target_bundle(&selections.values[0]);
  ASSERT_NE(selected_bundle, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(selected_bundle->name, IREE_SV("test-quirky")));
  EXPECT_TRUE(iree_string_view_equal(selected_bundle->snapshot->name,
                                     IREE_SV("test-quirky")));
  EXPECT_EQ(selected_bundle->snapshot->index_bitwidth, 32u);
  EXPECT_EQ(selected_bundle->snapshot->subgroup_size, 7u);
  EXPECT_TRUE(iree_string_view_equal(selected_bundle->config->name,
                                     IREE_SV("test.low.core")));
  EXPECT_TRUE(iree_string_view_equal(selected_bundle->config->contract_set_key,
                                     IREE_SV("test.low.core")));
  EXPECT_EQ(selections.values[0].target_source,
            LOOM_TARGET_BINDING_SOURCE_SPECIALIZATION);
  iree_arena_deinitialize(&arena);
}

TEST_F(LowLowerPassTest, ModuleInternalVersionLowersWithoutArtifactAbi) {
  ModulePtr module =
      Parse(IREE_SV("test.target<low_core> @test_target\n"
                    "func.def @helper() {\n"
                    "  func.return\n"
                    "}\n"));

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_symbol_fact_table_t symbol_facts = {};
  loom_symbol_fact_table_initialize(&symbol_facts, &arena);
  const loom_symbol_ref_t target_ref =
      FindSymbolRef(module.get(), IREE_SV("test_target"));
  const loom_symbol_facts_base_t* base_target_facts = nullptr;
  IREE_ASSERT_OK(loom_symbol_fact_table_lookup_ref(
      &symbol_facts, module.get(), target_ref, &base_target_facts));
  const loom_target_symbol_facts_t* target_facts =
      loom_target_symbol_facts_cast(base_target_facts);
  ASSERT_NE(target_facts, nullptr);

  const loom_symbol_ref_t function_ref =
      FindSymbolRef(module.get(), IREE_SV("helper"));
  const loom_symbol_facts_base_t* base_function_facts = nullptr;
  IREE_ASSERT_OK(loom_symbol_fact_table_lookup_ref(
      &symbol_facts, module.get(), function_ref, &base_function_facts));
  const loom_func_symbol_facts_t* function_facts =
      loom_func_symbol_facts_cast(base_function_facts);
  ASSERT_NE(function_facts, nullptr);

  bool contract_valid = false;
  const loom_target_facts_t* function_target_facts = nullptr;
  IREE_ASSERT_OK(loom_target_function_contract_refine_internal_facts(
      module.get(), function_facts, target_facts->name,
      target_facts->projection, iree_diagnostic_emitter_t{}, &arena,
      &contract_valid, &function_target_facts));
  ASSERT_TRUE(contract_valid);
  ASSERT_NE(function_target_facts, nullptr);
  ASSERT_EQ(function_target_facts->storage.export_plan.abi_kind,
            LOOM_TARGET_ABI_UNKNOWN);

  loom_target_function_version_t function_version = {};
  function_version.base.type = &loom_target_function_version_type;
  function_version.base.function = loom_func_like_cast(
      module.get(),
      module->symbols.entries[function_ref.symbol_id].defining_op);
  function_version.function_target_facts = function_target_facts;
  loom_function_version_t* function_version_values[] = {
      &function_version.base,
  };
  const loom_function_version_list_t function_versions = {
      /*.values=*/function_version_values,
      /*.count=*/IREE_ARRAYSIZE(function_version_values),
  };

  IREE_ASSERT_OK(RunSourceToLow(&policy_registry_, module.get(), nullptr,
                                &function_versions));
  ASSERT_TRUE(loom_low_func_def_isa(function_version.base.function.op));
  EXPECT_EQ(loom_func_like_abi(function_version.base.function),
            LOOM_TARGET_ABI_UNKNOWN);
  EXPECT_FALSE(loom_symbol_ref_is_valid(
      loom_low_func_def_target(function_version.base.function.op)));
  iree_arena_deinitialize(&arena);
}

TEST_F(LowLowerPassTest,
       InvocationBoundTargetlessFunctionRunsLowPassesWithoutWitness) {
  ModulePtr module = Parse(IREE_SV(
      "test.target<low_core> @available_target\n"
      "test.target<quirky> @other_target\n"
      "template.decl @demo.targeted(%value: i32) -> (i32)\n"
      "template.def<@demo.targeted> target(@available_target) priority(20) "
      "@selected(%value: i32) -> (i32) {\n"
      "  %sum = scalar.addi %value, %value : i32\n"
      "  template.return %sum : i32\n"
      "}\n"
      "template.def<@demo.targeted> target(@other_target) priority(30) "
      "@other(%value: i32) -> (i32) {\n"
      "  template.return %value : i32\n"
      "}\n"
      "func.def public @entry(%arg: i32) -> (i32) {\n"
      "  %result = template.apply<@demo.targeted>(%arg) : (i32) -> (i32)\n"
      "  func.return %result : i32\n"
      "}\n"));
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  const loom_symbol_ref_t available_target_ref =
      FindSymbolRef(module.get(), IREE_SV("available_target"));
  loom_symbol_fact_table_t symbol_facts = {};
  loom_symbol_fact_table_initialize(&symbol_facts, &arena);
  const loom_symbol_facts_base_t* base_target_facts = nullptr;
  IREE_ASSERT_OK(loom_symbol_fact_table_lookup_ref(
      &symbol_facts, module.get(), available_target_ref, &base_target_facts));
  const loom_target_symbol_facts_t* available_target_facts =
      loom_target_symbol_facts_cast(base_target_facts);
  ASSERT_NE(available_target_facts, nullptr);

  const loom_symbol_ref_t function_ref =
      FindSymbolRef(module.get(), IREE_SV("entry"));
  loom_target_function_version_t function_version = {};
  function_version.base.type = &loom_target_function_version_type;
  function_version.base.function = loom_func_like_cast(
      module.get(),
      module->symbols.entries[function_ref.symbol_id].defining_op);
  function_version.function_target_facts = available_target_facts->projection;
  loom_function_version_t* function_version_values[] = {
      &function_version.base,
  };
  const loom_function_version_list_t function_versions = {
      /*.values=*/function_version_values,
      /*.count=*/IREE_ARRAYSIZE(function_version_values),
  };

  IREE_ASSERT_OK(RunFlatPipeline(
      module.get(), IREE_SV("select-templates,inline-callables,symbol-dce"),
      &function_versions));
  bool selected_provider_inlined = false;
  loom_block_t* source_entry = loom_region_entry_block(
      loom_func_like_body(function_version.base.function));
  ASSERT_NE(source_entry, nullptr);
  for (const loom_op_t* op = source_entry->first_op; op != nullptr;
       op = op->next_op) {
    selected_provider_inlined |= loom_scalar_addi_isa(op);
  }
  EXPECT_TRUE(selected_provider_inlined);

  IREE_ASSERT_OK(RunSourceToLow(&policy_registry_, module.get(), nullptr,
                                &function_versions));
  ASSERT_TRUE(loom_low_func_def_isa(function_version.base.function.op));
  EXPECT_FALSE(loom_symbol_ref_is_valid(
      loom_low_func_def_target(function_version.base.function.op)));
  const loom_string_id_t descriptor_set_id =
      loom_low_func_def_descriptor_set(function_version.base.function.op);
  ASSERT_LT(descriptor_set_id, module->strings.count);
  EXPECT_TRUE(iree_string_view_equal(module->strings.entries[descriptor_set_id],
                                     IREE_SV("test.low.core")));

  loom_block_t* low_entry = loom_region_entry_block(
      loom_func_like_body(function_version.base.function));
  ASSERT_NE(low_entry, nullptr);
  iree_host_size_t low_packet_count = 0;
  for (const loom_op_t* op = low_entry->first_op; op != nullptr;
       op = op->next_op) {
    low_packet_count += loom_low_op_isa(op) ? 1 : 0;
  }
  EXPECT_EQ(low_packet_count, 1u);

  IREE_ASSERT_OK(
      RunFlatPipeline(module.get(), IREE_SV("low-dce"), &function_versions));
  low_packet_count = 0;
  for (const loom_op_t* op = low_entry->first_op; op != nullptr;
       op = op->next_op) {
    low_packet_count += loom_low_op_isa(op) ? 1 : 0;
  }
  EXPECT_EQ(low_packet_count, 1u);
  IREE_ASSERT_OK(RunFlatPipeline(
      module.get(), IREE_SV("low-materialize-allocation"), &function_versions));
  iree_arena_deinitialize(&arena);
}

TEST_F(LowLowerPassTest,
       DurableFunctionTargetPrunesOffTargetProvidersBeforeLowering) {
  ModulePtr module = Parse(IREE_SV(
      "template.decl @demo.targeted(%value: i32) -> (i32)\n"
      "template.def<@demo.targeted> target(@quirky) priority(20) "
      "@quirky_bad(%value: i32) -> (i32) {\n"
      "  test.use %value : i32\n"
      "  template.return %value : i32\n"
      "}\n"
      "\n"
      "template.def<@demo.targeted> target(@low_core) priority(10) "
      "@low_core_good(%value: i32) -> (i32) {\n"
      "  %doubled = scalar.addi %value, %value : i32\n"
      "  template.return %doubled : i32\n"
      "}\n"
      "\n"
      "template.def<@demo.targeted> priority(1) @fallback(%value: i32) -> "
      "(i32) {\n"
      "  template.return %value : i32\n"
      "}\n"
      "\n"
      "test.target<low_core> @low_core\n"
      "test.target<quirky> @quirky\n"
      "\n"
      "func.def public target(@low_core) @entry(%arg: i32) -> (i32) {\n"
      "  %result = template.apply<@demo.targeted>(%arg) : (i32) -> (i32)\n"
      "  func.return %result : i32\n"
      "}\n"));
  IREE_ASSERT_OK(RunFlatPipeline(
      module.get(), IREE_SV("select-templates,inline-callables,symbol-dce")));
  EXPECT_FALSE(HasSymbol(module.get(), IREE_SV("quirky_bad")));
  EXPECT_FALSE(HasSymbol(module.get(), IREE_SV("fallback")));

  IREE_ASSERT_OK(RunFlatPipeline(module.get(), IREE_SV("source-to-low")));
  const loom_symbol_ref_t low_core_ref =
      FindSymbolRef(module.get(), IREE_SV("low_core"));
  const loom_symbol_ref_t entry_ref =
      FindSymbolRef(module.get(), IREE_SV("entry"));
  ASSERT_TRUE(loom_symbol_ref_is_valid(entry_ref));
  const loom_symbol_t* entry_symbol =
      &module->symbols.entries[entry_ref.symbol_id];
  ASSERT_NE(entry_symbol->defining_op, nullptr);
  ASSERT_TRUE(loom_low_func_def_isa(entry_symbol->defining_op));
  const loom_symbol_ref_t lowered_target =
      loom_low_func_def_target(entry_symbol->defining_op);
  EXPECT_EQ(lowered_target.module_id, low_core_ref.module_id);
  EXPECT_EQ(lowered_target.symbol_id, low_core_ref.symbol_id);
  const loom_string_id_t descriptor_set_id =
      loom_low_func_def_descriptor_set(entry_symbol->defining_op);
  ASSERT_LT(descriptor_set_id, module->strings.count);
  EXPECT_TRUE(iree_string_view_equal(module->strings.entries[descriptor_set_id],
                                     IREE_SV("test.low.core")));
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

TEST_F(LowLowerPassTest, DenseSwitchLowersToDescriptorBackedTargetTable) {
  ModulePtr module = Parse(
      IREE_SV("test.target<low_core> @test_target\n"
              "func.def target(@test_target) @dispatch(%selector: index) {\n"
              "  cfg.switch %selector cases [0, 2] -> [^case0, ^case2] default "
              "^fallback\n"
              "^case0:\n"
              "  func.return\n"
              "^case2:\n"
              "  func.return\n"
              "^fallback:\n"
              "  func.return\n"
              "}\n"));

  IREE_ASSERT_OK(RunSourceToLow(&policy_registry_, module.get()));
  const loom_symbol_ref_t function_ref =
      FindSymbolRef(module.get(), IREE_SV("dispatch"));
  const loom_func_like_t function = loom_func_like_cast(
      module.get(),
      module->symbols.entries[function_ref.symbol_id].defining_op);
  const loom_region_t* body = loom_func_like_body(function);
  ASSERT_NE(body, nullptr);

  const loom_op_t* switch_op = nullptr;
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_op_t* op = nullptr;
    loom_block_for_each_op(body->blocks[block_index], op) {
      if (loom_low_switch_isa(op)) {
        ASSERT_EQ(switch_op, nullptr);
        switch_op = op;
      }
    }
  }
  ASSERT_NE(switch_op, nullptr);
  const loom_successor_slice_t targets =
      loom_low_switch_target_dests(switch_op);
  ASSERT_EQ(targets.count, 3u);
  EXPECT_NE(targets.blocks[0], loom_low_switch_default_dest(switch_op));
  EXPECT_EQ(targets.blocks[1], loom_low_switch_default_dest(switch_op));
  EXPECT_NE(targets.blocks[2], loom_low_switch_default_dest(switch_op));
}

TEST_F(LowLowerPassTest, UnsupportedSwitchExpandsBeforeLowLowering) {
  ModulePtr module = Parse(
      IREE_SV("test.target<low_core> @test_target\n"
              "func.def target(@test_target) @dispatch(%selector: index) {\n"
              "  cfg.switch %selector cases [-1, 2] -> [^negative, ^positive] "
              "default ^fallback\n"
              "^negative:\n"
              "  func.return\n"
              "^positive:\n"
              "  func.return\n"
              "^fallback:\n"
              "  func.return\n"
              "}\n"));

  IREE_ASSERT_OK(RunSourceToLow(&policy_registry_, module.get()));
  const loom_symbol_ref_t function_ref =
      FindSymbolRef(module.get(), IREE_SV("dispatch"));
  const loom_func_like_t function = loom_func_like_cast(
      module.get(),
      module->symbols.entries[function_ref.symbol_id].defining_op);
  const loom_region_t* body = loom_func_like_body(function);
  ASSERT_NE(body, nullptr);

  iree_host_size_t switch_count = 0;
  iree_host_size_t conditional_branch_count = 0;
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_op_t* op = nullptr;
    loom_block_for_each_op(body->blocks[block_index], op) {
      switch_count += loom_low_switch_isa(op) ? 1 : 0;
      conditional_branch_count += loom_low_cond_br_isa(op) ? 1 : 0;
    }
  }
  EXPECT_EQ(switch_count, 0u);
  EXPECT_EQ(conditional_branch_count, 2u);
}

}  // namespace
}  // namespace loom
