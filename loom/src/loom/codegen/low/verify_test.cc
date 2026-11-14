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

struct CountingVerifyProvider : loom_low_verify_provider_t {
  // Lifecycle calls observed for this provider during one verification run.
  struct Counts {
    // Module initialization and finalization calls, independent of scope.
    unsigned module_calls = 0;
    // Function initialization and finalization calls, restricted by scope.
    unsigned function_calls = 0;
    // Body operations presented to this provider.
    unsigned op_calls = 0;
  };
  // Test-owned observation state; registration metadata remains immutable.
  mutable Counts counts;

  explicit CountingVerifyProvider(loom_low_descriptor_scope_t scope)
      : loom_low_verify_provider_t{
            .name = IREE_SVL("counting"),
            .scope = scope,
            .begin_module = BeginModule,
            .begin_function = BeginFunction,
            .verify_op = VerifyOp,
            .end_function = EndFunction,
            .end_module = EndModule,
        } {}

  static iree_status_t BeginModule(const loom_low_verify_provider_t* provider,
                                   loom_low_verify_module_context_t*,
                                   void** out_state) {
    auto* counts =
        &static_cast<const CountingVerifyProvider*>(provider)->counts;
    ++counts->module_calls;
    *out_state = counts;
    return iree_ok_status();
  }

  static iree_status_t BeginFunction(const loom_low_verify_provider_t*,
                                     loom_low_verify_context_t* context,
                                     void** out_state) {
    auto* counts = static_cast<Counts*>(
        loom_low_verify_context_provider_module_state(context));
    ++counts->function_calls;
    *out_state = counts;
    return iree_ok_status();
  }

  static iree_status_t VerifyOp(const loom_low_verify_provider_t*,
                                loom_low_verify_context_t*, void* state,
                                const loom_low_descriptor_packet_t*) {
    ++static_cast<Counts*>(state)->op_calls;
    return iree_ok_status();
  }

  static iree_status_t EndFunction(const loom_low_verify_provider_t*,
                                   loom_low_verify_context_t*, void* state) {
    ++static_cast<Counts*>(state)->function_calls;
    return iree_ok_status();
  }

  static iree_status_t EndModule(const loom_low_verify_provider_t*,
                                 loom_low_verify_module_context_t*,
                                 void* state) {
    ++static_cast<Counts*>(state)->module_calls;
    return iree_ok_status();
  }
};

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
    loom_text_parse_options_t options = {.max_errors = 20};
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

  void VerifyModule(loom_module_t* module, DiagnosticEmissionCapture* capture,
                    const loom_function_version_list_t* function_versions,
                    loom_low_verify_provider_list_t providers,
                    loom_low_verify_result_t* out_result) {
    loom_low_verify_options_t options = {
        .descriptor_registry = &registry_.registry,
        .function_versions = function_versions,
        .emitter = capture->emitter(),
        .provider_list = providers,
        .max_errors = 20,
    };
    loom_low_verify_scratch_t scratch =
        loom_low_verify_scratch_for_module(module);
    IREE_EXPECT_OK(
        loom_low_verify_module(module, &options, &scratch, out_result));
  }

  void VerifyModule(loom_module_t* module, DiagnosticEmissionCapture* capture,
                    loom_low_verify_result_t* out_result) {
    VerifyModule(module, capture, /*function_versions=*/nullptr, {},
                 out_result);
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
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_registry_lookup(&registry_.registry,
                                          IREE_SV("test.low.core"));
  ASSERT_NE(descriptor_set, nullptr);
  const CountingVerifyProvider providers[] = {
      CountingVerifyProvider(LOOM_LOW_DESCRIPTOR_SCOPE_SHARED),
      CountingVerifyProvider(
          {.target_stable_id = descriptor_set->target_stable_id}),
      CountingVerifyProvider(
          {.descriptor_set_stable_id = descriptor_set->stable_id}),
      CountingVerifyProvider(
          {.target_stable_id = descriptor_set->target_stable_id ^ 1}),
  };
  const loom_low_verify_provider_t* provider_values[] = {
      &providers[0],
      &providers[1],
      &providers[2],
      &providers[3],
  };
  VerifyModule(module.get(), &capture, nullptr,
               loom_low_verify_provider_list_make(
                   provider_values, IREE_ARRAYSIZE(provider_values)),
               &result);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.emissions.empty());
  for (size_t i = 0; i < IREE_ARRAYSIZE(providers); ++i) {
    EXPECT_EQ(providers[i].counts.module_calls, 2u) << i;
    EXPECT_EQ(providers[i].counts.function_calls, i < 3 ? 2u : 0u) << i;
    EXPECT_EQ(providers[i].counts.op_calls, i < 3 ? 2u : 0u) << i;
  }
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
      .values = version_values,
      .count = IREE_ARRAYSIZE(version_values),
  };

  DiagnosticEmissionCapture capture;
  loom_low_verify_result_t result = {};
  VerifyModule(module.get(), &capture, &function_versions, {}, &result);
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
