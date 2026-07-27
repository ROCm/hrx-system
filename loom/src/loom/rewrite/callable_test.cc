// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/callable.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

class CallableInlineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
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

  using VTableFn = const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(loom_dialect_id_t dialect_id, VTableFn vtables_fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = vtables_fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  loom_symbol_ref_t MakeSymbol(iree_string_view_t name) {
    loom_string_id_t name_id = InternString(name);
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return (loom_symbol_ref_t){/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  }

  loom_string_id_t InternString(iree_string_view_t string) {
    loom_string_id_t string_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_builder_intern_string(&module_builder_, string, &string_id));
    return string_id;
  }

  void SetValueName(loom_value_id_t value_id, iree_string_view_t name) {
    loom_string_id_t name_id = InternString(name);
    loom_module_value(module_, value_id)->name_id = name_id;
  }

  loom_builder_t BodyBuilder(loom_op_t* func_op) {
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    loom_builder_t builder = {};
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(loom_func_like_body(func)),
                            &builder);
    builder.ip.parent_op = func_op;
    return builder;
  }

  loom_op_t* BuildNegateFunction(loom_symbol_ref_t callee,
                                 loom_type_t value_type) {
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_func_def_build(
        &module_builder_, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
        loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_empty(), callee, &value_type, 1, &value_type, 1,
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    IREE_ASSERT_EQ(arg_count, 1);

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* neg_op = nullptr;
    IREE_CHECK_OK(loom_test_neg_build(&body_builder, args[0], value_type,
                                      LOOM_LOCATION_UNKNOWN, &neg_op));
    loom_value_id_t negated = loom_test_neg_result(neg_op);
    loom_op_t* return_op = nullptr;
    IREE_CHECK_OK(loom_func_return_build(&body_builder, &negated, 1,
                                         LOOM_LOCATION_UNKNOWN, &return_op));
    return func_op;
  }

  loom_op_t* BuildNegateTemplate(loom_symbol_ref_t callee,
                                 loom_type_t value_type) {
    loom_string_id_t implements = InternString(IREE_SV("test.neg"));
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_func_template_build(
        &module_builder_, 0, implements, 0, 0, 0, 0, 0, 0,
        loom_symbol_ref_null(), 0, callee, &value_type, 1, &value_type, 1,
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    IREE_ASSERT_EQ(arg_count, 1);

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* neg_op = nullptr;
    IREE_CHECK_OK(loom_test_neg_build(&body_builder, args[0], value_type,
                                      LOOM_LOCATION_UNKNOWN, &neg_op));
    loom_value_id_t negated = loom_test_neg_result(neg_op);
    loom_op_t* return_op = nullptr;
    IREE_CHECK_OK(loom_func_return_build(&body_builder, &negated, 1,
                                         LOOM_LOCATION_UNKNOWN, &return_op));
    return func_op;
  }

  loom_op_t* BuildSwapAndNegateFunction(loom_symbol_ref_t callee,
                                        loom_type_t value_type) {
    loom_type_t arg_types[2] = {value_type, value_type};
    loom_type_t result_types[2] = {value_type, value_type};
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_func_def_build(
        &module_builder_, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
        loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_empty(), callee, arg_types,
        IREE_ARRAYSIZE(arg_types), result_types, IREE_ARRAYSIZE(result_types),
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    IREE_ASSERT_EQ(arg_count, 2);

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* neg_op = nullptr;
    IREE_CHECK_OK(loom_test_neg_build(&body_builder, args[0], value_type,
                                      LOOM_LOCATION_UNKNOWN, &neg_op));
    loom_value_id_t returned[2] = {args[1], loom_test_neg_result(neg_op)};
    loom_op_t* return_op = nullptr;
    IREE_CHECK_OK(loom_func_return_build(&body_builder, returned,
                                         IREE_ARRAYSIZE(returned),
                                         LOOM_LOCATION_UNKNOWN, &return_op));
    return func_op;
  }

  loom_op_t* BuildCaller(loom_symbol_ref_t caller, loom_symbol_ref_t callee,
                         loom_type_t value_type, loom_op_t** out_call_op) {
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_func_def_build(
        &module_builder_, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
        loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_empty(), caller, &value_type, 1, &value_type, 1,
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    IREE_ASSERT_EQ(arg_count, 1);

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* call_op = nullptr;
    IREE_CHECK_OK(loom_func_call_build(&body_builder, 0, 0, 0, 0, callee, args,
                                       1, &value_type, 1, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &call_op));
    loom_value_id_t call_result = loom_func_call_results(call_op).values[0];
    SetValueName(call_result, IREE_SV("call_result"));
    loom_op_t* return_op = nullptr;
    IREE_CHECK_OK(loom_func_return_build(&body_builder, &call_result, 1,
                                         LOOM_LOCATION_UNKNOWN, &return_op));
    *out_call_op = call_op;
    return func_op;
  }

  loom_op_t* BuildTwoResultCaller(loom_symbol_ref_t caller,
                                  loom_symbol_ref_t callee,
                                  loom_type_t value_type,
                                  loom_op_t** out_call_op) {
    loom_type_t arg_types[2] = {value_type, value_type};
    loom_type_t result_types[2] = {value_type, value_type};
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_func_def_build(
        &module_builder_, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
        loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_empty(), caller, arg_types,
        IREE_ARRAYSIZE(arg_types), result_types, IREE_ARRAYSIZE(result_types),
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    IREE_ASSERT_EQ(arg_count, 2);

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* call_op = nullptr;
    IREE_CHECK_OK(loom_func_call_build(&body_builder, 0, 0, 0, 0, callee, args,
                                       IREE_ARRAYSIZE(arg_types), result_types,
                                       IREE_ARRAYSIZE(result_types), nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &call_op));
    loom_value_slice_t call_results = loom_func_call_results(call_op);
    SetValueName(call_results.values[0], IREE_SV("swapped"));
    SetValueName(call_results.values[1], IREE_SV("negated"));
    loom_op_t* return_op = nullptr;
    IREE_CHECK_OK(loom_func_return_build(&body_builder, call_results.values,
                                         call_results.count,
                                         LOOM_LOCATION_UNKNOWN, &return_op));
    *out_call_op = call_op;
    return func_op;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t module_builder_ = {};
  iree_arena_allocator_t rewriter_arena_;
};

TEST_F(CallableInlineTest, InlinesDirectCallAndReplacesReturnOperand) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t callee_ref = MakeSymbol(IREE_SV("negate"));
  loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));
  loom_op_t* callee_op = BuildNegateFunction(callee_ref, i32);
  loom_op_t* call_op = nullptr;
  loom_op_t* caller_op = BuildCaller(caller_ref, callee_ref, i32, &call_op);

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  IREE_ASSERT_OK(loom_callable_inline_direct_call(&rewriter, call_op));
  loom_rewriter_deinitialize(&rewriter);

  EXPECT_TRUE(iree_any_bit_set(call_op->flags, LOOM_OP_FLAG_DEAD));
  loom_region_t* caller_body =
      loom_func_like_body(loom_func_like_cast(module_, caller_op));
  loom_block_t* caller_block = loom_region_entry_block(caller_body);
  ASSERT_EQ(caller_block->op_count, 2u);
  ASSERT_TRUE(loom_test_neg_isa(loom_block_op(caller_block, 0)));
  ASSERT_TRUE(loom_func_return_isa(loom_block_op(caller_block, 1)));
  EXPECT_EQ(loom_block_op(caller_block, 0)->parent_op, caller_op);

  loom_value_id_t negated =
      loom_test_neg_result(loom_block_op(caller_block, 0));
  EXPECT_TRUE(iree_string_view_equal(
      module_->strings.entries[loom_module_value(module_, negated)->name_id],
      IREE_SV("call_result")));
  loom_value_slice_t returned =
      loom_func_return_operands(loom_block_op(caller_block, 1));
  ASSERT_EQ(returned.count, 1u);
  EXPECT_EQ(returned.values[0], negated);

  loom_region_t* callee_body =
      loom_func_like_body(loom_func_like_cast(module_, callee_op));
  EXPECT_EQ(loom_region_entry_block(callee_body)->op_count, 2u);
}

TEST_F(CallableInlineTest, ConsumingInlineMovesBodyAndErasesCallee) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t callee_ref = MakeSymbol(IREE_SV("negate"));
  loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));
  loom_op_t* callee_op = BuildNegateFunction(callee_ref, i32);
  loom_func_like_t callee = loom_func_like_cast(module_, callee_op);
  loom_block_t* callee_block =
      loom_region_entry_block(loom_func_like_body(callee));
  loom_op_t* moved_neg_op = loom_block_op(callee_block, 0);
  loom_op_t* call_op = nullptr;
  loom_op_t* caller_op = BuildCaller(caller_ref, callee_ref, i32, &call_op);

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  IREE_ASSERT_OK(
      loom_callable_inline_consuming_direct_call(&rewriter, call_op));
  loom_rewriter_deinitialize(&rewriter);

  EXPECT_TRUE(iree_any_bit_set(call_op->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_TRUE(iree_any_bit_set(callee_op->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_EQ(module_->symbols.entries[callee_ref.symbol_id].defining_op,
            nullptr);
  EXPECT_EQ(module_->symbols.entries[callee_ref.symbol_id].kind,
            LOOM_SYMBOL_NONE);

  loom_func_like_t caller = loom_func_like_cast(module_, caller_op);
  loom_block_t* caller_block =
      loom_region_entry_block(loom_func_like_body(caller));
  ASSERT_EQ(caller_block->op_count, 2u);
  EXPECT_EQ(loom_block_op(caller_block, 0), moved_neg_op);
  ASSERT_TRUE(loom_func_return_isa(loom_block_op(caller_block, 1)));
  loom_value_id_t negated = loom_test_neg_result(moved_neg_op);
  loom_value_slice_t returned =
      loom_func_return_operands(loom_block_op(caller_block, 1));
  ASSERT_EQ(returned.count, 1u);
  EXPECT_EQ(returned.values[0], negated);
  EXPECT_TRUE(iree_string_view_equal(
      module_->strings.entries[loom_module_value(module_, negated)->name_id],
      IREE_SV("call_result")));
}

TEST_F(CallableInlineTest, ConsumingInlineRejectsPublicCallee) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t callee_ref = MakeSymbol(IREE_SV("negate"));
  loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));
  loom_op_t* callee_op = BuildNegateFunction(callee_ref, i32);
  module_->symbols.entries[callee_ref.symbol_id].flags |=
      LOOM_SYMBOL_FLAG_PUBLIC;
  loom_op_t* call_op = nullptr;
  BuildCaller(caller_ref, callee_ref, i32, &call_op);

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_callable_inline_consuming_direct_call(&rewriter, call_op));
  loom_rewriter_deinitialize(&rewriter);

  EXPECT_FALSE(iree_any_bit_set(call_op->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_FALSE(iree_any_bit_set(callee_op->flags, LOOM_OP_FLAG_DEAD));
}

TEST_F(CallableInlineTest, ConsumingInlineRejectsAdditionalCalleeReference) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t callee_ref = MakeSymbol(IREE_SV("negate"));
  loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));
  loom_symbol_ref_t other_caller_ref = MakeSymbol(IREE_SV("other_caller"));
  loom_op_t* callee_op = BuildNegateFunction(callee_ref, i32);
  loom_op_t* call_op = nullptr;
  BuildCaller(caller_ref, callee_ref, i32, &call_op);
  loom_op_t* other_call_op = nullptr;
  BuildCaller(other_caller_ref, callee_ref, i32, &other_call_op);

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_callable_inline_consuming_direct_call(&rewriter, call_op));
  loom_rewriter_deinitialize(&rewriter);

  EXPECT_FALSE(iree_any_bit_set(call_op->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_FALSE(iree_any_bit_set(other_call_op->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_FALSE(iree_any_bit_set(callee_op->flags, LOOM_OP_FLAG_DEAD));
}

TEST_F(CallableInlineTest, InlinesExactTemplateCall) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t callee_ref = MakeSymbol(IREE_SV("template_negate"));
  loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));
  BuildNegateTemplate(callee_ref, i32);
  loom_op_t* call_op = nullptr;
  loom_op_t* caller_op = BuildCaller(caller_ref, callee_ref, i32, &call_op);

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  IREE_ASSERT_OK(loom_callable_inline_direct_call(&rewriter, call_op));
  loom_rewriter_deinitialize(&rewriter);

  EXPECT_TRUE(iree_any_bit_set(call_op->flags, LOOM_OP_FLAG_DEAD));
  loom_region_t* caller_body =
      loom_func_like_body(loom_func_like_cast(module_, caller_op));
  loom_block_t* caller_block = loom_region_entry_block(caller_body);
  ASSERT_EQ(caller_block->op_count, 2u);
  ASSERT_TRUE(loom_test_neg_isa(loom_block_op(caller_block, 0)));
  ASSERT_TRUE(loom_func_return_isa(loom_block_op(caller_block, 1)));
}

TEST_F(CallableInlineTest, InlinesMultiResultCall) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t callee_ref = MakeSymbol(IREE_SV("swap_and_negate"));
  loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));
  BuildSwapAndNegateFunction(callee_ref, i32);
  loom_op_t* call_op = nullptr;
  loom_op_t* caller_op =
      BuildTwoResultCaller(caller_ref, callee_ref, i32, &call_op);
  loom_func_like_t caller = loom_func_like_cast(module_, caller_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(caller, &arg_count);
  ASSERT_EQ(arg_count, 2u);

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  IREE_ASSERT_OK(loom_callable_inline_direct_call(&rewriter, call_op));
  loom_rewriter_deinitialize(&rewriter);

  EXPECT_TRUE(iree_any_bit_set(call_op->flags, LOOM_OP_FLAG_DEAD));
  loom_region_t* caller_body = loom_func_like_body(caller);
  loom_block_t* caller_block = loom_region_entry_block(caller_body);
  ASSERT_EQ(caller_block->op_count, 2u);
  ASSERT_TRUE(loom_test_neg_isa(loom_block_op(caller_block, 0)));
  ASSERT_TRUE(loom_func_return_isa(loom_block_op(caller_block, 1)));
  loom_value_id_t negated =
      loom_test_neg_result(loom_block_op(caller_block, 0));
  EXPECT_TRUE(iree_string_view_equal(
      module_->strings.entries[loom_module_value(module_, negated)->name_id],
      IREE_SV("negated")));
  loom_value_slice_t returned =
      loom_func_return_operands(loom_block_op(caller_block, 1));
  ASSERT_EQ(returned.count, 2u);
  EXPECT_EQ(returned.values[0], args[1]);
  EXPECT_EQ(returned.values[1], negated);
}

TEST_F(CallableInlineTest, RejectsRecursiveSelfInline) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t self_ref = MakeSymbol(IREE_SV("self"));
  loom_op_t* func_op = nullptr;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
      loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), self_ref, &i32, 1, &i32, 1, nullptr, 0,
      nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 1u);

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* call_op = nullptr;
  IREE_ASSERT_OK(loom_func_call_build(&body_builder, 0, 0, 0, 0, self_ref, args,
                                      1, &i32, 1, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &call_op));
  loom_value_id_t call_result = loom_func_call_results(call_op).values[0];
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &call_result, 1,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_callable_inline_direct_call(&rewriter, call_op));
  loom_rewriter_deinitialize(&rewriter);
}

TEST_F(CallableInlineTest, OutlinesRangeIntoFunctionAndCall) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));
  loom_symbol_ref_t outlined_ref = MakeSymbol(IREE_SV("outlined"));
  loom_op_t* caller_op = nullptr;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
      loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), caller_ref, &i32, 1, &i32, 1, nullptr, 0,
      nullptr, 0, LOOM_LOCATION_UNKNOWN, &caller_op));
  loom_func_like_t caller = loom_func_like_cast(module_, caller_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(caller, &arg_count);
  ASSERT_EQ(arg_count, 1u);

  loom_builder_t body_builder = BodyBuilder(caller_op);
  loom_op_t* first_neg_op = nullptr;
  IREE_ASSERT_OK(loom_test_neg_build(&body_builder, args[0], i32,
                                     LOOM_LOCATION_UNKNOWN, &first_neg_op));
  loom_value_id_t first_neg = loom_test_neg_result(first_neg_op);
  loom_op_t* second_neg_op = nullptr;
  IREE_ASSERT_OK(loom_test_neg_build(&body_builder, first_neg, i32,
                                     LOOM_LOCATION_UNKNOWN, &second_neg_op));
  loom_value_id_t second_neg = loom_test_neg_result(second_neg_op);
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &second_neg, 1,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  loom_callable_outline_result_t outline = {};
  IREE_ASSERT_OK(loom_callable_outline_range(&rewriter, first_neg_op, return_op,
                                             outlined_ref, &outline));
  loom_rewriter_deinitialize(&rewriter);

  ASSERT_TRUE(loom_func_like_isa(outline.outlined));
  ASSERT_TRUE(loom_func_call_isa(outline.call_op));
  EXPECT_EQ(module_->symbols.entries[outlined_ref.symbol_id].defining_op,
            outline.outlined.op);

  loom_block_t* caller_block =
      loom_region_entry_block(loom_func_like_body(caller));
  ASSERT_EQ(caller_block->op_count, 2u);
  ASSERT_EQ(loom_block_op(caller_block, 0), outline.call_op);
  ASSERT_TRUE(loom_func_return_isa(loom_block_op(caller_block, 1)));
  loom_value_slice_t call_results = loom_func_call_results(outline.call_op);
  ASSERT_EQ(call_results.count, 1u);
  loom_value_slice_t returned =
      loom_func_return_operands(loom_block_op(caller_block, 1));
  ASSERT_EQ(returned.count, 1u);
  EXPECT_EQ(returned.values[0], call_results.values[0]);
  loom_value_slice_t call_operands = loom_func_call_operands(outline.call_op);
  ASSERT_EQ(call_operands.count, 1u);
  EXPECT_EQ(call_operands.values[0], args[0]);

  loom_block_t* outlined_block =
      loom_region_entry_block(loom_func_like_body(outline.outlined));
  ASSERT_EQ(outlined_block->op_count, 3u);
  ASSERT_TRUE(loom_test_neg_isa(loom_block_op(outlined_block, 0)));
  ASSERT_TRUE(loom_test_neg_isa(loom_block_op(outlined_block, 1)));
  ASSERT_TRUE(loom_func_return_isa(loom_block_op(outlined_block, 2)));
  EXPECT_TRUE(iree_any_bit_set(first_neg_op->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_TRUE(iree_any_bit_set(second_neg_op->flags, LOOM_OP_FLAG_DEAD));
}

TEST_F(CallableInlineTest, OutlineReturnsMultipleLiveOuts) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));
  loom_symbol_ref_t outlined_ref = MakeSymbol(IREE_SV("outlined"));
  loom_type_t result_types[2] = {i32, i32};
  loom_op_t* caller_op = nullptr;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
      loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), caller_ref, &i32, 1, result_types,
      IREE_ARRAYSIZE(result_types), nullptr, 0, nullptr, 0,
      LOOM_LOCATION_UNKNOWN, &caller_op));
  loom_func_like_t caller = loom_func_like_cast(module_, caller_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(caller, &arg_count);
  ASSERT_EQ(arg_count, 1u);

  loom_builder_t body_builder = BodyBuilder(caller_op);
  loom_op_t* first_neg_op = nullptr;
  IREE_ASSERT_OK(loom_test_neg_build(&body_builder, args[0], i32,
                                     LOOM_LOCATION_UNKNOWN, &first_neg_op));
  loom_value_id_t first_neg = loom_test_neg_result(first_neg_op);
  loom_op_t* second_neg_op = nullptr;
  IREE_ASSERT_OK(loom_test_neg_build(&body_builder, first_neg, i32,
                                     LOOM_LOCATION_UNKNOWN, &second_neg_op));
  loom_value_id_t returned_values[2] = {first_neg,
                                        loom_test_neg_result(second_neg_op)};
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, returned_values,
                                        IREE_ARRAYSIZE(returned_values),
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  loom_callable_outline_result_t outline = {};
  IREE_ASSERT_OK(loom_callable_outline_range(&rewriter, first_neg_op, return_op,
                                             outlined_ref, &outline));
  loom_rewriter_deinitialize(&rewriter);

  loom_value_slice_t call_results = loom_func_call_results(outline.call_op);
  ASSERT_EQ(call_results.count, 2u);
  loom_block_t* caller_block =
      loom_region_entry_block(loom_func_like_body(caller));
  loom_value_slice_t returned =
      loom_func_return_operands(loom_block_op(caller_block, 1));
  ASSERT_EQ(returned.count, 2u);
  EXPECT_EQ(returned.values[0], call_results.values[0]);
  EXPECT_EQ(returned.values[1], call_results.values[1]);

  loom_block_t* outlined_block =
      loom_region_entry_block(loom_func_like_body(outline.outlined));
  loom_value_slice_t outlined_returned =
      loom_func_return_operands(loom_block_op(outlined_block, 2));
  ASSERT_EQ(outlined_returned.count, 2u);
}

TEST_F(CallableInlineTest, OutlineRemapsDynamicResultTypeRefs) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t input_type = loom_type_shaped_1d(
      LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t placeholder_result_types[2] = {input_type, index_type};
  loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));
  loom_symbol_ref_t outlined_ref = MakeSymbol(IREE_SV("outlined"));
  loom_op_t* caller_op = nullptr;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
      loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), caller_ref, &input_type, 1,
      placeholder_result_types, IREE_ARRAYSIZE(placeholder_result_types),
      nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &caller_op));
  loom_func_like_t caller = loom_func_like_cast(module_, caller_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(caller, &arg_count);
  ASSERT_EQ(arg_count, 1u);

  loom_builder_t body_builder = BodyBuilder(caller_op);
  loom_value_id_t reserved_results[2] = {};
  IREE_ASSERT_OK(loom_builder_reserve_results(
      &body_builder, IREE_ARRAYSIZE(reserved_results), reserved_results));
  loom_type_t deflate_result_types[2] = {
      loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(reserved_results[1]), 0),
      index_type,
  };
  loom_op_t* deflate_op = nullptr;
  IREE_ASSERT_OK(
      loom_test_deflate_build(&body_builder, args[0], deflate_result_types,
                              IREE_ARRAYSIZE(deflate_result_types), nullptr, 0,
                              LOOM_LOCATION_UNKNOWN, &deflate_op));
  loom_value_slice_t deflate_results = loom_test_deflate_results(deflate_op);
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, deflate_results.values,
                                        deflate_results.count,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  loom_callable_outline_result_t outline = {};
  IREE_ASSERT_OK(loom_callable_outline_range(&rewriter, deflate_op, return_op,
                                             outlined_ref, &outline));
  loom_rewriter_deinitialize(&rewriter);

  loom_block_t* caller_block =
      loom_region_entry_block(loom_func_like_body(caller));
  loom_value_slice_t returned =
      loom_func_return_operands(loom_block_op(caller_block, 1));
  ASSERT_EQ(returned.count, 2u);
  loom_type_t returned_tensor_type =
      loom_module_value_type(module_, returned.values[0]);
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(returned_tensor_type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(returned_tensor_type, 0),
            returned.values[1]);
  EXPECT_TRUE(iree_any_bit_set(deflate_op->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_NE(loom_type_dim_value_id_at(returned_tensor_type, 0),
            deflate_results.values[1]);
}

TEST_F(CallableInlineTest, OutlineReturnsTypeOnlyDynamicDeps) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t input_type = loom_type_shaped_1d(
      LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));
  loom_symbol_ref_t outlined_ref = MakeSymbol(IREE_SV("outlined"));
  loom_op_t* caller_op = nullptr;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
      loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), caller_ref, &input_type, 1, &input_type, 1,
      nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &caller_op));
  loom_func_like_t caller = loom_func_like_cast(module_, caller_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(caller, &arg_count);
  ASSERT_EQ(arg_count, 1u);

  loom_builder_t body_builder = BodyBuilder(caller_op);
  loom_value_id_t reserved_results[2] = {};
  IREE_ASSERT_OK(loom_builder_reserve_results(
      &body_builder, IREE_ARRAYSIZE(reserved_results), reserved_results));
  loom_type_t deflate_result_types[2] = {
      loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(reserved_results[1]), 0),
      index_type,
  };
  loom_op_t* deflate_op = nullptr;
  IREE_ASSERT_OK(
      loom_test_deflate_build(&body_builder, args[0], deflate_result_types,
                              IREE_ARRAYSIZE(deflate_result_types), nullptr, 0,
                              LOOM_LOCATION_UNKNOWN, &deflate_op));
  loom_value_slice_t deflate_results = loom_test_deflate_results(deflate_op);
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, deflate_results.values,
                                        1, LOOM_LOCATION_UNKNOWN, &return_op));

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  loom_callable_outline_result_t outline = {};
  IREE_ASSERT_OK(loom_callable_outline_range(&rewriter, deflate_op, return_op,
                                             outlined_ref, &outline));
  loom_rewriter_deinitialize(&rewriter);

  loom_value_slice_t call_results = loom_func_call_results(outline.call_op);
  ASSERT_EQ(call_results.count, 2u);
  loom_type_t call_tensor_type =
      loom_module_value_type(module_, call_results.values[1]);
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(call_tensor_type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(call_tensor_type, 0),
            call_results.values[0]);
  EXPECT_TRUE(loom_module_value_has_type_uses(module_, call_results.values[0]));

  loom_block_t* caller_block =
      loom_region_entry_block(loom_func_like_body(caller));
  loom_value_slice_t returned =
      loom_func_return_operands(loom_block_op(caller_block, 1));
  ASSERT_EQ(returned.count, 1u);
  EXPECT_EQ(returned.values[0], call_results.values[1]);
  EXPECT_TRUE(iree_any_bit_set(deflate_op->flags, LOOM_OP_FLAG_DEAD));
}

TEST_F(CallableInlineTest, OutlineRejectsAlreadyDefinedSymbol) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));
  loom_symbol_ref_t outlined_ref = MakeSymbol(IREE_SV("outlined"));
  BuildNegateFunction(outlined_ref, i32);
  loom_op_t* caller_op = nullptr;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
      loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), caller_ref, &i32, 1, &i32, 1, nullptr, 0,
      nullptr, 0, LOOM_LOCATION_UNKNOWN, &caller_op));
  loom_func_like_t caller = loom_func_like_cast(module_, caller_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(caller, &arg_count);
  ASSERT_EQ(arg_count, 1u);
  loom_builder_t body_builder = BodyBuilder(caller_op);
  loom_op_t* neg_op = nullptr;
  IREE_ASSERT_OK(loom_test_neg_build(&body_builder, args[0], i32,
                                     LOOM_LOCATION_UNKNOWN, &neg_op));
  loom_value_id_t negated = loom_test_neg_result(neg_op);
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &negated, 1,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  loom_callable_outline_result_t outline = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ALREADY_EXISTS,
      loom_callable_outline_range(&rewriter, neg_op, return_op, outlined_ref,
                                  &outline));
  loom_rewriter_deinitialize(&rewriter);

  EXPECT_FALSE(loom_func_like_isa(outline.outlined));
  EXPECT_FALSE(iree_any_bit_set(neg_op->flags, LOOM_OP_FLAG_DEAD));
}

class CallableImportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("source"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &source_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("target"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &target_));
    loom_builder_initialize(source_, &source_->arena,
                            loom_module_block(source_), &source_builder_);
    loom_builder_initialize(target_, &target_->arena,
                            loom_module_block(target_), &target_builder_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&scratch_arena_);
    loom_module_free(target_);
    loom_module_free(source_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using VTableFn = const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(loom_dialect_id_t dialect_id, VTableFn vtables_fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = vtables_fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  loom_symbol_ref_t MakeSymbol(loom_module_t* module, loom_builder_t* builder,
                               iree_string_view_t name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(builder, name, &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    return (loom_symbol_ref_t){/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  }

  loom_builder_t BodyBuilder(loom_module_t* module, loom_op_t* func_op) {
    loom_func_like_t func = loom_func_like_cast(module, func_op);
    loom_builder_t builder = {};
    loom_builder_initialize(module, &module->arena,
                            loom_region_entry_block(loom_func_like_body(func)),
                            &builder);
    builder.ip.parent_op = func_op;
    return builder;
  }

  loom_op_t* BuildNegateFunction(loom_module_t* module,
                                 loom_builder_t* module_builder,
                                 loom_symbol_ref_t callee,
                                 loom_type_t value_type) {
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_func_def_build(
        module_builder, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
        loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_empty(), callee, &value_type, 1, &value_type, 1,
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    IREE_ASSERT_EQ(arg_count, 1);

    loom_builder_t body_builder = BodyBuilder(module, func_op);
    loom_op_t* neg_op = nullptr;
    IREE_CHECK_OK(loom_test_neg_build(&body_builder, args[0], value_type,
                                      LOOM_LOCATION_UNKNOWN, &neg_op));
    loom_value_id_t negated = loom_test_neg_result(neg_op);
    loom_op_t* return_op = nullptr;
    IREE_CHECK_OK(loom_func_return_build(&body_builder, &negated, 1,
                                         LOOM_LOCATION_UNKNOWN, &return_op));
    return func_op;
  }

  loom_op_t* BuildCallWrapperFunction(loom_module_t* module,
                                      loom_builder_t* module_builder,
                                      loom_symbol_ref_t caller,
                                      loom_symbol_ref_t callee,
                                      loom_type_t value_type) {
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_func_def_build(
        module_builder, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
        loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_empty(), caller, &value_type, 1, &value_type, 1,
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    IREE_ASSERT_EQ(arg_count, 1);

    loom_builder_t body_builder = BodyBuilder(module, func_op);
    loom_op_t* call_op = nullptr;
    IREE_CHECK_OK(loom_func_call_build(&body_builder, 0, 0, 0, 0, callee, args,
                                       1, &value_type, 1, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &call_op));
    loom_value_slice_t call_results = loom_func_call_results(call_op);
    IREE_ASSERT_EQ(call_results.count, 1u);
    loom_value_id_t call_result = call_results.values[0];
    loom_op_t* return_op = nullptr;
    IREE_CHECK_OK(loom_func_return_build(&body_builder, &call_result, 1,
                                         LOOM_LOCATION_UNKNOWN, &return_op));
    return func_op;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* source_ = nullptr;
  loom_module_t* target_ = nullptr;
  loom_builder_t source_builder_ = {};
  loom_builder_t target_builder_ = {};
  iree_arena_allocator_t scratch_arena_;
};

static iree_status_t RemapSymbolByName(void* user_data,
                                       const loom_module_t* source_module,
                                       loom_module_t* target_module,
                                       loom_symbol_ref_t source_ref,
                                       loom_symbol_ref_t* out_target_ref) {
  (void)user_data;
  if (!loom_symbol_ref_is_valid(source_ref) || source_ref.module_id != 0 ||
      source_ref.symbol_id >= source_module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source symbol ref is invalid");
  }
  const loom_symbol_t* source_symbol =
      &source_module->symbols.entries[source_ref.symbol_id];
  iree_string_view_t source_name =
      source_module->strings.entries[source_symbol->name_id];
  loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(target_module, source_name, &target_name_id));
  uint16_t target_symbol_id =
      loom_module_find_symbol(target_module, target_name_id);
  if (target_symbol_id == LOOM_SYMBOL_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(target_module, target_name_id,
                                                &target_symbol_id));
  }
  *out_target_ref = {/*.module_id=*/0, /*.symbol_id=*/target_symbol_id};
  return iree_ok_status();
}

TEST_F(CallableImportTest, ImportsDefinitionWithDistinctTargetStorage) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t source_ref =
      MakeSymbol(source_, &source_builder_, IREE_SV("negate"));
  loom_op_t* source_op =
      BuildNegateFunction(source_, &source_builder_, source_ref, i32);
  loom_func_like_t source_func = loom_func_like_cast(source_, source_op);

  loom_func_like_t imported = {};
  IREE_ASSERT_OK(loom_callable_import_definition(&target_builder_, source_,
                                                 source_func, nullptr,
                                                 &imported, &scratch_arena_));

  ASSERT_TRUE(loom_func_like_isa(imported));
  EXPECT_NE(imported.op, source_op);
  ASSERT_EQ(target_->symbols.count, 1u);
  const loom_symbol_t* target_symbol = &target_->symbols.entries[0];
  EXPECT_TRUE(iree_string_view_equal(
      target_->strings.entries[target_symbol->name_id], IREE_SV("negate")));
  EXPECT_EQ(loom_symbol_bytecode_kind(target_symbol), LOOM_SYMBOL_FUNC_DEF);
  EXPECT_EQ(target_symbol->defining_op, imported.op);
  loom_symbol_ref_t imported_ref = loom_func_like_callee(imported);
  EXPECT_EQ(imported_ref.module_id, 0u);
  EXPECT_EQ(imported_ref.symbol_id, 0u);

  loom_region_t* source_body = loom_func_like_body(source_func);
  loom_region_t* imported_body = loom_func_like_body(imported);
  ASSERT_NE(imported_body, nullptr);
  EXPECT_NE(imported_body, source_body);
  loom_block_t* source_block = loom_region_entry_block(source_body);
  loom_block_t* imported_block = loom_region_entry_block(imported_body);
  ASSERT_NE(imported_block, nullptr);
  EXPECT_NE(imported_block, source_block);
  ASSERT_EQ(imported_block->op_count, 2u);
  loom_op_t* imported_neg_op = loom_block_op(imported_block, 0);
  ASSERT_TRUE(loom_test_neg_isa(imported_neg_op));
  EXPECT_NE(imported_neg_op, loom_block_op(source_block, 0));
  EXPECT_EQ(imported_neg_op->parent_op, imported.op);
  EXPECT_EQ(loom_test_neg_input(imported_neg_op),
            loom_region_entry_arg_id(imported_body, 0));
  ASSERT_TRUE(loom_func_return_isa(loom_block_op(imported_block, 1)));
  loom_value_slice_t returned =
      loom_func_return_operands(loom_block_op(imported_block, 1));
  ASSERT_EQ(returned.count, 1u);
  EXPECT_EQ(returned.values[0], loom_test_neg_result(imported_neg_op));
}

TEST_F(CallableImportTest, ImportRejectsTargetNameCollision) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t source_ref =
      MakeSymbol(source_, &source_builder_, IREE_SV("negate"));
  loom_op_t* source_op =
      BuildNegateFunction(source_, &source_builder_, source_ref, i32);
  MakeSymbol(target_, &target_builder_, IREE_SV("negate"));

  loom_func_like_t imported = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ALREADY_EXISTS,
      loom_callable_import_definition(&target_builder_, source_,
                                      loom_func_like_cast(source_, source_op),
                                      nullptr, &imported, &scratch_arena_));

  EXPECT_FALSE(loom_func_like_isa(imported));
  EXPECT_EQ(target_->symbols.count, 1u);
  EXPECT_EQ(loom_module_block(target_)->op_count, 0u);
}

TEST_F(CallableImportTest, ImportRejectsExternalSymbolWithoutPolicy) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t helper_ref =
      MakeSymbol(source_, &source_builder_, IREE_SV("helper"));
  BuildNegateFunction(source_, &source_builder_, helper_ref, i32);
  loom_symbol_ref_t wrapper_ref =
      MakeSymbol(source_, &source_builder_, IREE_SV("wrapper"));
  loom_op_t* wrapper_op = BuildCallWrapperFunction(
      source_, &source_builder_, wrapper_ref, helper_ref, i32);

  loom_func_like_t imported = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_callable_import_definition(&target_builder_, source_,
                                      loom_func_like_cast(source_, wrapper_op),
                                      nullptr, &imported, &scratch_arena_));

  EXPECT_FALSE(loom_func_like_isa(imported));
  EXPECT_EQ(target_->symbols.count, 0u);
  EXPECT_EQ(loom_module_block(target_)->op_count, 0u);
}

TEST_F(CallableImportTest, ImportRemapsExternalSymbolWithPolicy) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t helper_ref =
      MakeSymbol(source_, &source_builder_, IREE_SV("helper"));
  BuildNegateFunction(source_, &source_builder_, helper_ref, i32);
  loom_symbol_ref_t wrapper_ref =
      MakeSymbol(source_, &source_builder_, IREE_SV("wrapper"));
  loom_op_t* wrapper_op = BuildCallWrapperFunction(
      source_, &source_builder_, wrapper_ref, helper_ref, i32);
  loom_callable_import_options_t options = {
      /*.external_symbol_remap=*/
      loom_ir_remap_symbol_callback_make(RemapSymbolByName, NULL),
  };

  loom_func_like_t imported = {};
  IREE_ASSERT_OK(loom_callable_import_definition(
      &target_builder_, source_, loom_func_like_cast(source_, wrapper_op),
      &options, &imported, &scratch_arena_));

  loom_string_id_t target_helper_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(target_, IREE_SV("helper"),
                                           &target_helper_name_id));
  uint16_t target_helper_symbol_id =
      loom_module_find_symbol(target_, target_helper_name_id);
  ASSERT_NE(target_helper_symbol_id, LOOM_SYMBOL_ID_INVALID);
  ASSERT_EQ(target_->symbols.count, 2u);
  EXPECT_EQ(target_->symbols.entries[target_helper_symbol_id].defining_op,
            nullptr);

  loom_region_t* imported_body = loom_func_like_body(imported);
  loom_block_t* imported_block = loom_region_entry_block(imported_body);
  ASSERT_EQ(imported_block->op_count, 2u);
  loom_op_t* imported_call_op = loom_block_op(imported_block, 0);
  ASSERT_TRUE(loom_func_call_isa(imported_call_op));
  loom_symbol_ref_t imported_call_ref = loom_func_call_callee(imported_call_op);
  EXPECT_EQ(imported_call_ref.module_id, 0u);
  EXPECT_EQ(imported_call_ref.symbol_id, target_helper_symbol_id);
}

}  // namespace
}  // namespace loom
