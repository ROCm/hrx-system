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
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/target/test/contracts/core_lower_rules.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"
#include "loom/target/test/target_records.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

struct DiagnosticEmissionCollector {
  int count = 0;
};

static iree_status_t CollectDiagnosticEmission(
    void* user_data, const loom_diagnostic_emission_t*) {
  auto* collector = static_cast<DiagnosticEmissionCollector*>(user_data);
  ++collector->count;
  return iree_ok_status();
}

class LowLowerPassTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&registry_);
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

  iree_status_t RunSourceToLow(
      loom_low_lower_policy_registry_t* policy_registry, loom_module_t* module,
      DiagnosticEmissionCollector* collector = nullptr) {
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
            nullptr, loom_target_selection_empty(),
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

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t registry_ = {};
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
  EXPECT_TRUE(
      iree_string_view_equal(selections.values[0].target_bundle->snapshot->name,
                             IREE_SV("test-quirky")));
  EXPECT_EQ(selections.values[0].target_data, &target_payload);
  iree_arena_deinitialize(&arena);
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
       SourceSelectionIgnoresIncompatibleRuntimeTargetSelection) {
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
  selected_storage.export_plan.abi_kind = LOOM_TARGET_ABI_HAL_KERNEL;

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
  EXPECT_EQ(selections.values[0].target_bundle->export_plan->abi_kind,
            LOOM_TARGET_ABI_OBJECT_FUNCTION);
  EXPECT_EQ(selections.values[0].target_bundle->snapshot->subgroup_size, 0u);
  EXPECT_EQ(selections.values[0].target_data, nullptr);
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
