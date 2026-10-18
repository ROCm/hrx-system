// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/callable_effect_inference.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"

namespace loom {
namespace {

class CallableEffectInferenceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_BUFFER, loom_buffer_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using VtableProvider = const loom_op_vtable_t* const* (*)(iree_host_size_t *
                                                            out_count);

  void RegisterDialect(loom_dialect_id_t dialect_id,
                       VtableProvider vtable_provider) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = vtable_provider(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  loom_symbol_ref_t AddSymbol(iree_string_view_t name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module_, name, &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return {.module_id = 0, .symbol_id = symbol_id};
  }

  loom_func_like_t BuildDefinition(
      iree_string_view_t name, loom_callable_effect_contract_t effect_contract =
                                   LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED) {
    const loom_func_def_build_flags_t build_flags =
        effect_contract == LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED
            ? 0
            : LOOM_FUNC_DEF_BUILD_FLAG_HAS_EFFECT_CONTRACT;
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_func_def_build(
        &module_builder_, build_flags, /*visibility=*/0, /*linkage=*/0,
        /*retain=*/0,
        /*cc=*/0, effect_contract, /*temperature=*/0, /*inline_policy=*/0,
        loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), AddSymbol(name),
        /*arg_types=*/nullptr, /*arg_types_count=*/0,
        /*result_types=*/nullptr, /*result_count=*/0,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN,
        &op));
    return loom_func_like_cast(module_, op);
  }

  loom_func_like_t BuildDeclaration(
      iree_string_view_t name, loom_callable_effect_contract_t effect_contract =
                                   LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED) {
    const loom_func_decl_build_flags_t build_flags =
        effect_contract == LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED
            ? 0
            : LOOM_FUNC_DECL_BUILD_FLAG_HAS_EFFECT_CONTRACT;
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_func_decl_build(
        &module_builder_, build_flags, /*visibility=*/0, /*retain=*/0,
        /*import_module=*/LOOM_STRING_ID_INVALID,
        /*import_symbol=*/LOOM_STRING_ID_INVALID, /*cc=*/0, effect_contract,
        /*temperature=*/0, /*inline_policy=*/0, loom_symbol_ref_null(),
        /*abi=*/0, loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_empty(), AddSymbol(name),
        /*arg_types=*/nullptr, /*arg_types_count=*/0,
        /*result_types=*/nullptr, /*result_count=*/0,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN,
        &op));
    return loom_func_like_cast(module_, op);
  }

  loom_builder_t BodyBuilder(loom_func_like_t function) {
    loom_builder_t builder;
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_func_like_body(function)), &builder);
    builder.ip.parent_op = function.op;
    return builder;
  }

  loom_op_t* AppendCall(loom_func_like_t caller, loom_func_like_t callee,
                        loom_callable_effect_contract_t effect_contract =
                            LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED) {
    const loom_func_call_build_flags_t build_flags =
        effect_contract == LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED
            ? 0
            : LOOM_FUNC_CALL_BUILD_FLAG_HAS_EFFECT_CONTRACT;
    loom_builder_t builder = BodyBuilder(caller);
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_func_call_build(
        &builder, build_flags, effect_contract, /*temperature=*/0,
        /*inline_policy=*/0, loom_func_like_callee(callee),
        /*operands=*/nullptr, /*operands_count=*/0,
        /*result_types=*/nullptr, /*result_count=*/0,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  void AppendAllocation(loom_func_like_t function) {
    loom_builder_t builder = BodyBuilder(function);
    loom_op_t* length_op = nullptr;
    IREE_ASSERT_OK(loom_index_constant_build(
        &builder, loom_attr_i64(16), loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
        LOOM_LOCATION_UNKNOWN, &length_op));
    loom_op_t* allocation_op = nullptr;
    IREE_ASSERT_OK(loom_buffer_alloca_build(
        &builder, /*build_flags=*/0, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE,
        LOOM_VALUE_ID_INVALID, /*base_alignment=*/16,
        loom_index_constant_result(length_op), loom_type_buffer(),
        LOOM_LOCATION_UNKNOWN, &allocation_op));
  }

  void AppendReturn(loom_func_like_t function) {
    loom_builder_t builder = BodyBuilder(function);
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_func_return_build(&builder, /*operands=*/nullptr,
                                          /*operands_count=*/0,
                                          LOOM_LOCATION_UNKNOWN, &op));
  }

  void FinalizeAndInfer() {
    IREE_ASSERT_OK(loom_module_compute_uses(module_));
    IREE_ASSERT_OK(loom_callable_effect_inference_run(module_));
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t module_builder_ = {};
};

TEST_F(CallableEffectInferenceTest, InfersTransitivePureAndReadonlyCalls) {
  const loom_func_like_t pure_leaf = BuildDefinition(IREE_SV("pure_leaf"));
  const loom_func_like_t pure_middle = BuildDefinition(IREE_SV("pure_middle"));
  const loom_func_like_t readonly_decl = BuildDeclaration(
      IREE_SV("readonly_decl"), LOOM_CALLABLE_EFFECT_CONTRACT_READONLY);
  const loom_func_like_t readonly_middle =
      BuildDefinition(IREE_SV("readonly_middle"));
  const loom_func_like_t root = BuildDefinition(IREE_SV("root"));

  AppendReturn(pure_leaf);
  loom_op_t* pure_leaf_call = AppendCall(pure_middle, pure_leaf);
  AppendReturn(pure_middle);
  loom_op_t* readonly_decl_call = AppendCall(readonly_middle, readonly_decl);
  AppendReturn(readonly_middle);
  loom_op_t* pure_middle_call = AppendCall(root, pure_middle);
  loom_op_t* readonly_middle_call = AppendCall(root, readonly_middle);
  AppendReturn(root);

  FinalizeAndInfer();

  EXPECT_EQ(loom_func_call_effect_contract(pure_leaf_call),
            LOOM_CALLABLE_EFFECT_CONTRACT_PURE);
  EXPECT_EQ(loom_func_call_effect_contract(pure_middle_call),
            LOOM_CALLABLE_EFFECT_CONTRACT_PURE);
  EXPECT_EQ(loom_func_call_effect_contract(readonly_decl_call),
            LOOM_CALLABLE_EFFECT_CONTRACT_READONLY);
  EXPECT_EQ(loom_func_call_effect_contract(readonly_middle_call),
            LOOM_CALLABLE_EFFECT_CONTRACT_READONLY);
  EXPECT_EQ(loom_func_like_effect_contract(pure_middle),
            LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED);
  EXPECT_EQ(loom_func_like_effect_contract(readonly_middle),
            LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED);
  EXPECT_TRUE(loom_region_has_read_effects(loom_func_like_body(root)));
  EXPECT_FALSE(loom_region_has_write_effects(loom_func_like_body(root)));
}

TEST_F(CallableEffectInferenceTest, PreservesAuthoredUpperBounds) {
  const loom_func_like_t pure_leaf = BuildDefinition(IREE_SV("pure_leaf"));
  const loom_func_like_t readonly_definition = BuildDefinition(
      IREE_SV("readonly_definition"), LOOM_CALLABLE_EFFECT_CONTRACT_READONLY);
  const loom_func_like_t caller = BuildDefinition(IREE_SV("caller"));
  const loom_func_like_t unknown_decl =
      BuildDeclaration(IREE_SV("unknown_decl"));
  const loom_func_like_t unknown_caller =
      BuildDefinition(IREE_SV("unknown_caller"));

  AppendReturn(pure_leaf);
  AppendReturn(readonly_definition);
  loom_op_t* authored_readonly_call =
      AppendCall(caller, pure_leaf, LOOM_CALLABLE_EFFECT_CONTRACT_READONLY);
  loom_op_t* inferred_readonly_call = AppendCall(caller, readonly_definition);
  AppendReturn(caller);
  loom_op_t* unknown_call = AppendCall(unknown_caller, unknown_decl);
  AppendReturn(unknown_caller);

  FinalizeAndInfer();

  EXPECT_EQ(loom_func_call_effect_contract(authored_readonly_call),
            LOOM_CALLABLE_EFFECT_CONTRACT_READONLY);
  EXPECT_EQ(loom_func_call_effect_contract(inferred_readonly_call),
            LOOM_CALLABLE_EFFECT_CONTRACT_READONLY);
  EXPECT_EQ(loom_func_call_effect_contract(unknown_call),
            LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED);
}

TEST_F(CallableEffectInferenceTest, PropagatesReadonlyThroughRecursiveScc) {
  const loom_func_like_t a = BuildDefinition(IREE_SV("a"));
  const loom_func_like_t b = BuildDefinition(IREE_SV("b"));
  const loom_func_like_t readonly_decl = BuildDeclaration(
      IREE_SV("readonly_decl"), LOOM_CALLABLE_EFFECT_CONTRACT_READONLY);

  loom_op_t* a_to_b = AppendCall(a, b);
  AppendReturn(a);
  loom_op_t* b_to_a = AppendCall(b, a);
  loom_op_t* b_to_readonly = AppendCall(b, readonly_decl);
  AppendReturn(b);

  FinalizeAndInfer();

  EXPECT_EQ(loom_func_call_effect_contract(a_to_b),
            LOOM_CALLABLE_EFFECT_CONTRACT_READONLY);
  EXPECT_EQ(loom_func_call_effect_contract(b_to_a),
            LOOM_CALLABLE_EFFECT_CONTRACT_READONLY);
  EXPECT_EQ(loom_func_call_effect_contract(b_to_readonly),
            LOOM_CALLABLE_EFFECT_CONTRACT_READONLY);
}

TEST_F(CallableEffectInferenceTest, FreshIdentityPreventsPureInference) {
  const loom_func_like_t allocator = BuildDefinition(IREE_SV("allocator"));
  const loom_func_like_t caller = BuildDefinition(IREE_SV("caller"));
  AppendAllocation(allocator);
  AppendReturn(allocator);
  loom_op_t* call = AppendCall(caller, allocator);
  AppendReturn(caller);

  FinalizeAndInfer();

  EXPECT_EQ(loom_func_call_effect_contract(call),
            LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED);
}

}  // namespace
}  // namespace loom
