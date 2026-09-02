// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/callable_effects.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"

namespace loom {
namespace {

class CallableEffectsTest : public ::testing::Test {
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
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder_);
    iree_arena_initialize(&block_pool_, &rewriter_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&rewriter_arena_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void AddSymbol(iree_string_view_t name, loom_symbol_ref_t* out_symbol) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(module_, name, &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    *out_symbol = loom_symbol_ref_t{/*.module_id=*/0,
                                    /*.symbol_id=*/symbol_id};
  }

  void BuildDeclaration(iree_string_view_t name, uint8_t purity,
                        loom_func_like_t* out_function) {
    loom_symbol_ref_t symbol = loom_symbol_ref_null();
    AddSymbol(name, &symbol);
    const loom_func_decl_build_flags_t build_flags =
        purity != 0 ? LOOM_FUNC_DECL_BUILD_FLAG_HAS_PURITY : 0;
    loom_op_t* function_op = nullptr;
    IREE_ASSERT_OK(loom_func_decl_build(
        &module_builder_, build_flags, /*visibility=*/0, /*retain=*/0,
        /*import_module=*/LOOM_STRING_ID_INVALID,
        /*import_symbol=*/LOOM_STRING_ID_INVALID, /*cc=*/0, purity,
        /*temperature=*/0, /*inline_policy=*/0, loom_symbol_ref_null(),
        /*abi=*/0, loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_empty(), symbol,
        /*arg_types=*/nullptr, /*arg_types_count=*/0,
        /*result_types=*/nullptr, /*result_count=*/0,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN,
        &function_op));
    *out_function = loom_func_like_cast(module_, function_op);
    ASSERT_TRUE(loom_func_like_isa(*out_function));
  }

  void BuildDefinition(iree_string_view_t name,
                       loom_func_like_t* out_function) {
    loom_symbol_ref_t symbol = loom_symbol_ref_null();
    AddSymbol(name, &symbol);
    loom_op_t* function_op = nullptr;
    IREE_ASSERT_OK(loom_func_def_build(
        &module_builder_, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0,
        /*cc=*/0, /*purity=*/0, /*temperature=*/0, /*inline_policy=*/0,
        loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol,
        /*arg_types=*/nullptr, /*arg_types_count=*/0,
        /*result_types=*/nullptr, /*result_count=*/0,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN,
        &function_op));
    *out_function = loom_func_like_cast(module_, function_op);
    ASSERT_TRUE(loom_func_like_isa(*out_function));
  }

  loom_builder_t BodyBuilder(loom_func_like_t function) {
    loom_builder_t builder;
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_func_like_body(function)), &builder);
    builder.ip.parent_op = function.op;
    return builder;
  }

  void AppendCall(loom_func_like_t caller, loom_func_like_t callee,
                  loom_op_t** out_call_op) {
    loom_builder_t builder = BodyBuilder(caller);
    IREE_ASSERT_OK(loom_func_call_build(
        &builder, /*build_flags=*/0, /*purity=*/0, /*temperature=*/0,
        /*inline_policy=*/0, loom_func_like_callee(callee),
        /*operands=*/nullptr, /*operands_count=*/0,
        /*result_types=*/nullptr, /*result_count=*/0,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        LOOM_LOCATION_UNKNOWN, out_call_op));
  }

  void AppendReturn(loom_func_like_t function) {
    loom_builder_t builder = BodyBuilder(function);
    loom_op_t* return_op = nullptr;
    IREE_ASSERT_OK(loom_func_return_build(&builder, /*operands=*/nullptr,
                                          /*operands_count=*/0,
                                          LOOM_LOCATION_UNKNOWN, &return_op));
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t module_builder_ = {};
  iree_arena_allocator_t rewriter_arena_;
};

TEST_F(CallableEffectsTest, DeclarationsRequireExplicitPureContract) {
  loom_func_like_t unspecified = {};
  BuildDeclaration(IREE_SV("unspecified"), /*purity=*/0, &unspecified);
  EXPECT_FALSE(loom_callable_effects_is_pure(unspecified));

  loom_func_like_t pure = {};
  BuildDeclaration(IREE_SV("pure"), LOOM_FUNC_PURITY_PURE, &pure);
  EXPECT_TRUE(loom_callable_effects_is_pure(pure));
}

TEST_F(CallableEffectsTest, DefinitionsUseBodyEffects) {
  loom_func_like_t pure = {};
  BuildDefinition(IREE_SV("pure"), &pure);
  AppendReturn(pure);
  EXPECT_TRUE(loom_callable_effects_is_pure(pure));

  loom_func_like_t unknown_effects = {};
  BuildDeclaration(IREE_SV("unknown_effects"), /*purity=*/0, &unknown_effects);
  loom_func_like_t impure = {};
  BuildDefinition(IREE_SV("impure"), &impure);
  loom_op_t* call_op = nullptr;
  AppendCall(impure, unknown_effects, &call_op);
  AppendReturn(impure);
  EXPECT_FALSE(loom_callable_effects_is_pure(impure));
}

TEST_F(CallableEffectsTest, PropagationRefreshesCallerEffects) {
  loom_func_like_t pure = {};
  BuildDefinition(IREE_SV("pure"), &pure);
  AppendReturn(pure);

  loom_func_like_t caller = {};
  BuildDefinition(IREE_SV("caller"), &caller);
  loom_op_t* call_op = nullptr;
  AppendCall(caller, pure, &call_op);
  AppendReturn(caller);
  EXPECT_TRUE(loom_region_has_read_effects(loom_func_like_body(caller)));
  EXPECT_EQ(
      loom_callable_effects_traits(call_op, loom_func_call_purity_ATTR_INDEX),
      LOOM_TRAIT_UNKNOWN_EFFECTS);

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  IREE_EXPECT_OK(loom_callable_effects_propagate_purity(
      call_op, loom_func_like_callee(pure), loom_func_call_purity_ATTR_INDEX,
      &rewriter));
  loom_rewriter_deinitialize(&rewriter);

  EXPECT_EQ(loom_func_call_purity(call_op), LOOM_FUNC_PURITY_PURE);
  EXPECT_EQ(
      loom_callable_effects_traits(call_op, loom_func_call_purity_ATTR_INDEX),
      LOOM_TRAIT_PURE);
  EXPECT_FALSE(loom_region_has_read_effects(loom_func_like_body(caller)));
  EXPECT_FALSE(loom_region_has_write_effects(loom_func_like_body(caller)));
  EXPECT_FALSE(loom_region_has_convergent_effects(loom_func_like_body(caller)));
}

}  // namespace
}  // namespace loom
