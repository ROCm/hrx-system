// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/verify.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/error_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/facts.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/function_contract.h"
#include "loom/target/function_version.h"
#include "loom/target/test/low_registry.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ::loom::testing::CapturedDiagnosticEmission;
using ::loom::testing::DiagnosticEmissionCapture;
using ModulePtr = ::loom::testing::ModulePtr;

class LowVerifyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&registry_);
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
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  ModulePtr ParseModule(const char* source) {
    loom_text_parse_options_t options = {};
    options.max_errors = 20;
    loom_low_descriptor_text_asm_environment_initialize(
        &registry_.registry, &options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("low_verify_test.loom"), &context_,
                                  &block_pool_, &options, &module));
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

  void VerifyModule(
      loom_module_t* module, DiagnosticEmissionCapture* capture,
      loom_low_verify_result_t* out_result,
      const loom_function_version_list_t* function_versions = nullptr) {
    loom_low_verify_options_t options = {};
    options.descriptor_registry = &registry_.registry;
    options.function_versions = function_versions;
    options.emitter = capture->emitter();
    options.provider_list = loom_low_verify_provider_list_empty();
    options.max_errors = 20;
    loom_low_verify_scratch_t scratch =
        loom_low_verify_scratch_for_module(module);
    IREE_EXPECT_OK(
        loom_low_verify_module(module, &options, &scratch, out_result));
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  iree_arena_allocator_t analysis_arena_;
  loom_target_low_descriptor_registry_t registry_ = {};
};

TEST_F(LowVerifyTest, AcceptsWorkgroupStorageWithoutTargetLimit) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @target
low.func.def target<test.low.core>(@target) @uses_workgroup_storage() {
  %storage = low.storage.reserve {byte_alignment = 16, byte_length = 80} : low.storage<workgroup>
  low.return
}
)");
  DiagnosticEmissionCapture capture;
  loom_low_verify_result_t result = {};
  VerifyModule(module.get(), &capture, &result);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.emissions.empty());
}

TEST_F(LowVerifyTest, RejectsWorkgroupStorageAboveDurableTargetLimit) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @target {max_workgroup_storage_bytes = 64}
low.func.def target<test.low.core>(@target) @uses_workgroup_storage() {
  %storage = low.storage.reserve {byte_alignment = 16, byte_length = 80} : low.storage<workgroup>
  low.return
}
)");

  DiagnosticEmissionCapture capture;
  loom_low_verify_result_t result = {};
  VerifyModule(module.get(), &capture, &result);
  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(capture.emissions.size(), 1u);

  const CapturedDiagnosticEmission& emission = capture.emissions[0];
  EXPECT_EQ(emission.error, LOOM_ERR_TARGET_051);
  ASSERT_EQ(emission.params.size(), 4u);
  ASSERT_EQ(emission.string_params.size(), 2u);
  EXPECT_EQ(emission.string_params[0], "uses_workgroup_storage");
  EXPECT_EQ(emission.string_params[1], "target");
  ASSERT_EQ(emission.u64_params.size(), 2u);
  EXPECT_EQ(emission.u64_params[0], 80u);
  EXPECT_EQ(emission.u64_params[1], 64u);
}

TEST_F(LowVerifyTest, UsesFunctionTargetFactsForTargetlessFunction) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @exact {max_workgroup_storage_bytes = 64}
low.func.def target<test.low.core> @uses_workgroup_storage() {
  %storage = low.storage.reserve {byte_alignment = 16, byte_length = 80} : low.storage<workgroup>
  low.return
}
)");

  loom_symbol_fact_table_t symbol_facts = {};
  loom_symbol_fact_table_initialize(&symbol_facts, &analysis_arena_);

  const loom_symbol_id_t function_symbol_id =
      FindSymbol(module.get(), IREE_SV("uses_workgroup_storage"));
  const loom_symbol_facts_base_t* base_function_facts = nullptr;
  IREE_ASSERT_OK(loom_symbol_fact_table_lookup(
      &symbol_facts, module.get(), function_symbol_id, &base_function_facts));
  const loom_func_symbol_facts_t* function_facts =
      loom_func_symbol_facts_cast(base_function_facts);
  ASSERT_NE(function_facts, nullptr);

  const loom_symbol_facts_base_t* base_target_facts = nullptr;
  IREE_ASSERT_OK(loom_symbol_fact_table_lookup(
      &symbol_facts, module.get(), FindSymbol(module.get(), IREE_SV("exact")),
      &base_target_facts));
  const loom_target_symbol_facts_t* target_facts =
      loom_target_symbol_facts_cast(base_target_facts);
  ASSERT_NE(target_facts, nullptr);

  bool contract_valid = false;
  const loom_target_facts_t* function_target_facts = nullptr;
  IREE_ASSERT_OK(loom_target_function_contract_refine_facts(
      module.get(), function_facts, target_facts->name,
      target_facts->projection, iree_diagnostic_emitter_t{}, &analysis_arena_,
      &contract_valid, &function_target_facts));
  ASSERT_TRUE(contract_valid);
  ASSERT_NE(function_target_facts, nullptr);

  loom_target_function_version_t function_version = {};
  function_version.base.type = &loom_target_function_version_type;
  function_version.base.function =
      loom_func_like_cast(module.get(), function_facts->func_op);
  ASSERT_TRUE(loom_func_like_isa(function_version.base.function));
  function_version.function_target_facts = function_target_facts;
  loom_function_version_t* version_values[] = {
      &function_version.base,
  };
  const loom_function_version_list_t function_versions = {
      /*.values=*/version_values,
      /*.count=*/IREE_ARRAYSIZE(version_values),
  };

  DiagnosticEmissionCapture capture;
  loom_low_verify_result_t result = {};
  VerifyModule(module.get(), &capture, &result, &function_versions);
  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(capture.emissions.size(), 1u);

  const CapturedDiagnosticEmission& emission = capture.emissions[0];
  EXPECT_EQ(emission.error, LOOM_ERR_TARGET_051);
  ASSERT_EQ(emission.string_params.size(), 2u);
  EXPECT_EQ(emission.string_params[0], "uses_workgroup_storage");
  EXPECT_EQ(emission.string_params[1], "exact");
  ASSERT_EQ(emission.u64_params.size(), 2u);
  EXPECT_EQ(emission.u64_params[0], 80u);
  EXPECT_EQ(emission.u64_params[1], 64u);
}

}  // namespace
}  // namespace loom
