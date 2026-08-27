// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/target_binding.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/error_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/facts.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/function_contract.h"
#include "loom/target/test/alt_descriptors.h"
#include "loom/target/test/descriptors.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

static const loom_low_descriptor_set_provider_t kDescriptorSetProviders[] = {
    loom_test_low_core_descriptor_set,
    loom_test_low_alt_descriptor_set,
};

struct DiagnosticCapture {
  const loom_error_def_t* error = nullptr;
};

static iree_status_t CaptureDiagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  auto* capture = static_cast<DiagnosticCapture*>(user_data);
  capture->error = emission->error;
  return iree_ok_status();
}

class LowTargetBindingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_symbol_fact_table_initialize(&symbol_facts_, &analysis_arena_);
    registry_.descriptor_set_providers = kDescriptorSetProviders;
    registry_.descriptor_set_provider_count =
        IREE_ARRAYSIZE(kDescriptorSetProviders);
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
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    options.max_errors = 20;
    loom_low_descriptor_text_asm_environment_initialize(
        &registry_, &options.low_asm_environment);
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("target_binding_test.loom"),
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

  const loom_op_t* LookupFunctionOp(const loom_module_t* module,
                                    iree_string_view_t name) {
    const loom_symbol_id_t symbol_id = FindSymbol(module, name);
    IREE_ASSERT(symbol_id < module->symbols.count);
    const loom_op_t* op = module->symbols.entries[symbol_id].defining_op;
    IREE_ASSERT(op != nullptr);
    return op;
  }

  const loom_func_symbol_facts_t* LookupFunctionFacts(
      const loom_module_t* module, iree_string_view_t name) {
    const loom_symbol_facts_base_t* base_facts = nullptr;
    IREE_CHECK_OK(loom_symbol_fact_table_lookup(
        &symbol_facts_, module, FindSymbol(module, name), &base_facts));
    const loom_func_symbol_facts_t* facts =
        loom_func_symbol_facts_cast(base_facts);
    IREE_ASSERT(facts != nullptr);
    return facts;
  }

  const loom_target_symbol_facts_t* LookupTargetFacts(
      const loom_module_t* module, iree_string_view_t name) {
    const loom_symbol_facts_base_t* base_facts = nullptr;
    IREE_CHECK_OK(loom_symbol_fact_table_lookup(
        &symbol_facts_, module, FindSymbol(module, name), &base_facts));
    const loom_target_symbol_facts_t* facts =
        loom_target_symbol_facts_cast(base_facts);
    IREE_ASSERT(facts != nullptr);
    return facts;
  }

  const loom_target_facts_t* RefineFunctionFacts(
      const loom_module_t* module, const loom_func_symbol_facts_t* function,
      const loom_target_symbol_facts_t* target) {
    bool valid = false;
    const loom_target_facts_t* function_target_facts = nullptr;
    IREE_CHECK_OK(loom_target_function_contract_refine_facts(
        module, function, target->name, target->projection,
        iree_diagnostic_emitter_t{}, &analysis_arena_, &valid,
        &function_target_facts));
    IREE_ASSERT(valid);
    IREE_ASSERT(function_target_facts != nullptr);
    return function_target_facts;
  }

  // Block pool shared by parser, module allocation, and analysis storage.
  iree_arena_block_pool_t block_pool_;

  // Context containing the target-aware dialects used by the fixtures.
  loom_context_t context_;

  // Arena retaining symbol facts and function target refinements.
  iree_arena_allocator_t analysis_arena_;

  // Dense symbol facts for each parsed test module.
  loom_symbol_fact_table_t symbol_facts_;

  // Synthetic Low descriptor registry selected by representation contracts.
  loom_low_descriptor_registry_t registry_ = {};
};

TEST_F(LowTargetBindingTest, ResolvedTargetRetainsImmutableFacts) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @target {
  contract_feature_bits = 7,
  default_pointer_bitwidth = 64,
  index_bitwidth = 64,
  offset_bitwidth = 64
}
low.func.def target<test.low.core>(@target) @kernel() {
  low.return
}
)");

  loom_low_resolved_target_t target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module.get(), &symbol_facts_,
      LookupFunctionOp(module.get(), IREE_SV("kernel")),
      /*function_target_facts=*/nullptr, &registry_,
      iree_diagnostic_emitter_t{}, &target));

  ASSERT_NE(target.target_facts, nullptr);
  EXPECT_TRUE(iree_string_view_equal(target.target_name, IREE_SV("target")));
  EXPECT_EQ(target.feature_bits, 7u);
  ASSERT_NE(target.descriptor_set, nullptr);
  EXPECT_TRUE(iree_string_view_equal(target.descriptor_set_key,
                                     IREE_SV("test.low.core")));

  const loom_target_bundle_t* bundle = loom_low_resolved_target_bundle(&target);
  ASSERT_NE(bundle, nullptr);
  EXPECT_EQ(bundle, &target.target_facts->storage.bundle);
  EXPECT_EQ(bundle->snapshot, &target.target_facts->storage.snapshot);
  EXPECT_EQ(bundle->config, &target.target_facts->storage.config);
  EXPECT_EQ(bundle->snapshot->default_pointer_bitwidth, 64u);

  const loom_low_resolved_target_t copied_target = target;
  EXPECT_EQ(copied_target.target_facts, target.target_facts);
  EXPECT_EQ(loom_low_resolved_target_bundle(&copied_target), bundle);
}

TEST_F(LowTargetBindingTest, FunctionTargetFactsOverrideAuthoredTarget) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @generic {
  contract_feature_bits = 1,
  default_pointer_bitwidth = 32,
  index_bitwidth = 32,
  offset_bitwidth = 32
}
test.target<low_core> @exact {
  contract_feature_bits = 7,
  default_pointer_bitwidth = 64,
  index_bitwidth = 64,
  offset_bitwidth = 64
}
low.func.def target<test.low.core>(@generic) @kernel() {
  low.return
}
)");

  const loom_func_symbol_facts_t* function_facts =
      LookupFunctionFacts(module.get(), IREE_SV("kernel"));
  const loom_target_facts_t* exact_facts =
      RefineFunctionFacts(module.get(), function_facts,
                          LookupTargetFacts(module.get(), IREE_SV("exact")));
  const loom_op_t* function_op =
      LookupFunctionOp(module.get(), IREE_SV("kernel"));

  loom_low_resolved_target_t exact_target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module.get(), &symbol_facts_, function_op, exact_facts, &registry_,
      iree_diagnostic_emitter_t{}, &exact_target));
  EXPECT_EQ(exact_target.target_facts, exact_facts);
  EXPECT_TRUE(
      iree_string_view_equal(exact_target.target_name, IREE_SV("exact")));
  EXPECT_EQ(exact_target.feature_bits, 7u);
  EXPECT_EQ(loom_low_resolved_target_bundle(&exact_target),
            &exact_facts->storage.bundle);
  EXPECT_EQ(loom_low_resolved_target_bundle(&exact_target)
                ->snapshot->default_pointer_bitwidth,
            64u);
  ASSERT_NE(exact_target.descriptor_set, nullptr);

  loom_low_resolved_target_t authored_target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module.get(), &symbol_facts_, function_op,
      /*function_target_facts=*/nullptr, &registry_,
      iree_diagnostic_emitter_t{}, &authored_target));
  EXPECT_NE(authored_target.target_facts, exact_facts);
  EXPECT_TRUE(
      iree_string_view_equal(authored_target.target_name, IREE_SV("generic")));
  EXPECT_EQ(authored_target.feature_bits, 1u);
  EXPECT_EQ(loom_low_resolved_target_bundle(&authored_target)
                ->snapshot->default_pointer_bitwidth,
            32u);
}

TEST_F(LowTargetBindingTest, FunctionTargetFactsBindTargetlessFunction) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @exact {
  contract_feature_bits = 7,
  default_pointer_bitwidth = 64,
  index_bitwidth = 64,
  offset_bitwidth = 64
}
low.func.def target<test.low.core> @kernel() {
  low.return
}
)");

  const loom_func_symbol_facts_t* function_facts =
      LookupFunctionFacts(module.get(), IREE_SV("kernel"));
  const loom_target_facts_t* exact_facts =
      RefineFunctionFacts(module.get(), function_facts,
                          LookupTargetFacts(module.get(), IREE_SV("exact")));

  loom_low_resolved_target_t target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module.get(), &symbol_facts_,
      LookupFunctionOp(module.get(), IREE_SV("kernel")), exact_facts,
      &registry_, iree_diagnostic_emitter_t{}, &target));
  EXPECT_EQ(target.target_facts, exact_facts);
  EXPECT_TRUE(iree_string_view_equal(target.target_name, IREE_SV("exact")));
  EXPECT_EQ(target.feature_bits, 7u);
  ASSERT_NE(target.descriptor_set, nullptr);
}

TEST_F(LowTargetBindingTest, TargetlessFunctionUsesPortableRepresentation) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @portable() {
  low.return
}
)");

  loom_low_resolved_target_t target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module.get(), &symbol_facts_,
      LookupFunctionOp(module.get(), IREE_SV("portable")),
      /*function_target_facts=*/nullptr, &registry_,
      iree_diagnostic_emitter_t{}, &target));

  EXPECT_EQ(target.target_facts, nullptr);
  EXPECT_TRUE(iree_string_view_is_empty(target.target_name));
  EXPECT_EQ(target.feature_bits, 0u);
  ASSERT_NE(target.descriptor_set, nullptr);
  EXPECT_TRUE(iree_string_view_equal(target.descriptor_set_key,
                                     IREE_SV("test.low.core")));
}

TEST_F(LowTargetBindingTest, MissingRepresentationStopsBeforeTargetBinding) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @target
low.func.def target<test.low.core>(@target) @kernel() {
  low.return
}
)");

  DiagnosticCapture capture;
  loom_low_descriptor_registry_t empty_registry = {};
  loom_low_resolved_target_t target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module.get(), &symbol_facts_,
      LookupFunctionOp(module.get(), IREE_SV("kernel")),
      /*function_target_facts=*/nullptr, &empty_registry,
      {
          /*.fn=*/CaptureDiagnostic,
          /*.user_data=*/&capture,
      },
      &target));

  EXPECT_EQ(capture.error, LOOM_ERR_TARGET_044);
  EXPECT_EQ(target.target_facts, nullptr);
  EXPECT_EQ(target.descriptor_set, nullptr);
}

TEST_F(LowTargetBindingTest,
       PortableRepresentationSupportsDeclaredTargetContract) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @exact {
  contract_set_key = "test.low.alt"
}
low.func.def target<test.low.core>(@exact) @kernel() {
  low.return
}
)");

  loom_low_resolved_target_t target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module.get(), &symbol_facts_,
      LookupFunctionOp(module.get(), IREE_SV("kernel")),
      /*function_target_facts=*/nullptr, &registry_,
      iree_diagnostic_emitter_t{}, &target));

  ASSERT_NE(target.descriptor_set, nullptr);
  EXPECT_TRUE(iree_string_view_equal(target.descriptor_set_key,
                                     IREE_SV("test.low.core")));
  EXPECT_TRUE(iree_string_view_equal(
      loom_low_resolved_target_bundle(&target)->config->contract_set_key,
      IREE_SV("test.low.alt")));
}

TEST_F(LowTargetBindingTest, RepresentationCompatibilityIsDirectional) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @target
low.func.def target<test.low.alt>(@target) @kernel() {
  low.return
}
)");

  DiagnosticCapture capture;
  loom_low_resolved_target_t target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module.get(), &symbol_facts_,
      LookupFunctionOp(module.get(), IREE_SV("kernel")),
      /*function_target_facts=*/nullptr, &registry_,
      {
          /*.fn=*/CaptureDiagnostic,
          /*.user_data=*/&capture,
      },
      &target));

  EXPECT_EQ(capture.error, LOOM_ERR_TARGET_065);
  EXPECT_EQ(target.descriptor_set, nullptr);
}

}  // namespace
}  // namespace loom
