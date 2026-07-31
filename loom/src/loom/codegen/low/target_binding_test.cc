// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/target_binding.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/test/low_registry.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

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
    loom_test_low_descriptor_registry_initialize(&registry_);
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

  // Block pool shared by parser, module allocation, and analysis storage.
  iree_arena_block_pool_t block_pool_;

  // Context containing the target-aware dialects used by the fixtures.
  loom_context_t context_;

  // Arena retaining symbol facts and effective target refinements.
  iree_arena_allocator_t analysis_arena_;

  // Dense symbol facts for each parsed test module.
  loom_symbol_fact_table_t symbol_facts_;

  // Synthetic Low descriptor registry selected by representation contracts.
  loom_target_low_descriptor_registry_t registry_ = {};
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
      LookupFunctionOp(module.get(), IREE_SV("kernel")), &registry_.registry,
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

}  // namespace
}  // namespace loom
