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
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/template/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

class CallableInlineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CFG, loom_cfg_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEMPLATE, loom_template_dialect_vtables);
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

  loom_availability_analysis_t InitializeAvailability() {
    loom_availability_analysis_t availability = {};
    IREE_CHECK_OK(loom_availability_analysis_initialize(
        module_, &rewriter_arena_, &availability));
    return availability;
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
    const loom_symbol_ref_t family = MakeSymbol(IREE_SV("test.neg"));
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_template_def_build(
        &module_builder_, /*build_flags=*/0, family, /*visibility=*/0,
        /*retain=*/0, /*cc=*/0, /*purity=*/0, /*temperature=*/0,
        loom_symbol_ref_null(), loom_parameterized_attr_array_empty(),
        /*priority=*/0, callee, &value_type, 1, &value_type, 1, nullptr, 0,
        nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
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
    IREE_CHECK_OK(loom_template_return_build(
        &body_builder, &negated, 1, LOOM_LOCATION_UNKNOWN, &return_op));
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

  loom_op_t* BuildTemplateCaller(loom_symbol_ref_t caller,
                                 loom_symbol_ref_t callee,
                                 loom_type_t value_type,
                                 loom_op_t** out_call_op) {
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
    IREE_CHECK_OK(loom_template_call_build(
        &body_builder, /*build_flags=*/0, /*purity=*/0, /*temperature=*/0,
        callee, args, 1, &value_type, 1, nullptr, 0, LOOM_LOCATION_UNKNOWN,
        &call_op));
    loom_value_id_t call_result = loom_template_call_results(call_op).values[0];
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

TEST_F(CallableInlineTest, EmptyBodyIsNotLinear) {
  const loom_symbol_ref_t callee_ref = MakeSymbol(IREE_SV("empty"));
  loom_op_t* callee_op = nullptr;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
      loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), callee_ref, nullptr, 0, nullptr, 0,
      nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &callee_op));

  EXPECT_FALSE(loom_callable_body_is_linear(
      module_, loom_func_like_cast(module_, callee_op)));
}

TEST_F(CallableInlineTest, InlinesOneBlockSelfLoopThroughCfgSplice) {
  const loom_symbol_ref_t callee_ref = MakeSymbol(IREE_SV("spin"));
  const loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));
  loom_op_t* callee_op = nullptr;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
      loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), callee_ref, nullptr, 0, nullptr, 0,
      nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &callee_op));
  loom_func_like_t callee = loom_func_like_cast(module_, callee_op);
  loom_block_t* callee_block =
      loom_region_entry_block(loom_func_like_body(callee));
  loom_builder_t callee_builder = BodyBuilder(callee_op);
  loom_op_t* loop_branch = nullptr;
  IREE_ASSERT_OK(loom_cfg_br_build(&callee_builder, callee_block, nullptr, 0,
                                   LOOM_LOCATION_UNKNOWN, &loop_branch));
  EXPECT_FALSE(loom_callable_body_is_linear(module_, callee));

  loom_op_t* caller_op = nullptr;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
      loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), caller_ref, nullptr, 0, nullptr, 0,
      nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &caller_op));
  loom_builder_t caller_builder = BodyBuilder(caller_op);
  loom_op_t* call_op = nullptr;
  IREE_ASSERT_OK(loom_func_call_build(&caller_builder, 0, 0, 0, 0, callee_ref,
                                      nullptr, 0, nullptr, 0, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &call_op));
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_func_return_build(&caller_builder, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  IREE_ASSERT_OK(loom_callable_inline_direct_call(&rewriter, call_op));
  loom_rewriter_deinitialize(&rewriter);

  loom_region_t* caller_body =
      loom_func_like_body(loom_func_like_cast(module_, caller_op));
  ASSERT_EQ(caller_body->block_count, 3u);
  loom_block_t* caller_entry = loom_region_block(caller_body, 0);
  loom_block_t* cloned_loop = loom_region_block(caller_body, 1);
  loom_block_t* continuation = loom_region_block(caller_body, 2);
  ASSERT_TRUE(loom_cfg_br_isa(caller_entry->last_op));
  EXPECT_EQ(loom_cfg_br_dest(caller_entry->last_op), cloned_loop);
  ASSERT_TRUE(loom_cfg_br_isa(cloned_loop->last_op));
  EXPECT_EQ(loom_cfg_br_dest(cloned_loop->last_op), cloned_loop);
  ASSERT_TRUE(loom_func_return_isa(continuation->last_op));

  const loom_verify_options_t verify_options = {};
  loom_verify_result_t verify_result = {};
  IREE_ASSERT_OK(loom_verify_module(module_, &verify_options, &verify_result));
}

TEST_F(CallableInlineTest,
       RejectsCfgSpliceIntoSingleBlockRegionBeforeMutation) {
  const loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  const loom_type_t tile = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  const loom_symbol_ref_t callee_ref = MakeSymbol(IREE_SV("forward"));
  const loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));

  loom_op_t* callee_op = nullptr;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
      loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), callee_ref, &f32, 1, &f32, 1, nullptr, 0,
      nullptr, 0, LOOM_LOCATION_UNKNOWN, &callee_op));
  loom_func_like_t callee = loom_func_like_cast(module_, callee_op);
  uint16_t callee_arg_count = 0;
  const loom_value_id_t* callee_args =
      loom_func_like_arg_ids(callee, &callee_arg_count);
  ASSERT_EQ(callee_arg_count, 1u);
  loom_region_t* callee_body = loom_func_like_body(callee);
  loom_block_t* exit_block = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, callee_body, &exit_block));
  loom_builder_t callee_builder = BodyBuilder(callee_op);
  loom_value_id_t forwarded = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(&callee_builder, exit_block, f32,
                                               &forwarded));
  loom_op_t* branch_op = nullptr;
  IREE_ASSERT_OK(loom_cfg_br_build(&callee_builder, exit_block, callee_args, 1,
                                   LOOM_LOCATION_UNKNOWN, &branch_op));
  loom_builder_set_block(&callee_builder, exit_block);
  loom_op_t* callee_return_op = nullptr;
  IREE_ASSERT_OK(loom_func_return_build(&callee_builder, &forwarded, 1,
                                        LOOM_LOCATION_UNKNOWN,
                                        &callee_return_op));

  loom_op_t* caller_op = nullptr;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
      loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), caller_ref, &tile, 1, &tile, 1, nullptr, 0,
      nullptr, 0, LOOM_LOCATION_UNKNOWN, &caller_op));
  loom_func_like_t caller = loom_func_like_cast(module_, caller_op);
  uint16_t caller_arg_count = 0;
  const loom_value_id_t* caller_args =
      loom_func_like_arg_ids(caller, &caller_arg_count);
  ASSERT_EQ(caller_arg_count, 1u);
  loom_builder_t caller_builder = BodyBuilder(caller_op);
  loom_op_t* map_op = nullptr;
  IREE_ASSERT_OK(loom_test_map_build(&caller_builder, caller_args, 1, tile,
                                     nullptr, 0, LOOM_LOCATION_UNKNOWN,
                                     &map_op));
  loom_region_t* map_body = loom_test_map_body(map_op);
  loom_builder_ip_t saved_ip =
      loom_builder_enter_region(&caller_builder, map_op, map_body);
  const loom_value_id_t element = loom_region_entry_arg_id(map_body, 0);
  loom_op_t* call_op = nullptr;
  IREE_ASSERT_OK(loom_func_call_build(&caller_builder, 0, 0, 0, 0, callee_ref,
                                      &element, 1, &f32, 1, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &call_op));
  const loom_value_id_t call_result = loom_func_call_results(call_op).values[0];
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&caller_builder, &call_result, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&caller_builder, saved_ip);
  const loom_value_id_t map_result = loom_test_map_result(map_op);
  loom_op_t* caller_return_op = nullptr;
  IREE_ASSERT_OK(loom_func_return_build(&caller_builder, &map_result, 1,
                                        LOOM_LOCATION_UNKNOWN,
                                        &caller_return_op));

  const loom_verify_options_t verify_options = {};
  loom_verify_result_t verify_result = {};
  IREE_ASSERT_OK(loom_verify_module(module_, &verify_options, &verify_result));
  ASSERT_EQ(verify_result.error_count, 0u);
  ASSERT_FALSE(loom_callable_call_site_allows_cfg_splice(module_, call_op));
  const uint32_t value_count = module_->values.count;

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_callable_inline_direct_call(&rewriter, call_op));
  loom_rewriter_deinitialize(&rewriter);

  EXPECT_FALSE(iree_any_bit_set(call_op->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_EQ(call_op->parent_block, loom_region_entry_block(map_body));
  EXPECT_EQ(loom_region_entry_block(map_body)->op_count, 2u);
  EXPECT_EQ(map_body->block_count, 1u);
  EXPECT_EQ(loom_func_like_body(caller)->block_count, 1u);
  EXPECT_EQ(callee_body->block_count, 2u);
  EXPECT_EQ(module_->values.count, value_count);
  verify_result = {};
  IREE_ASSERT_OK(loom_verify_module(module_, &verify_options, &verify_result));
  EXPECT_EQ(verify_result.error_count, 0u);
}

TEST_F(CallableInlineTest, InlinesFuncLikeWithDeclaredTerminator) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t callee_ref = MakeSymbol(IREE_SV("test_negate"));
  loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));

  loom_op_t* callee_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(&module_builder_, 0, 0, 0, callee_ref,
                                      &i32, 1, &i32, 1, nullptr, 0, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &callee_op));
  loom_func_like_t callee = loom_func_like_cast(module_, callee_op);
  uint16_t callee_arg_count = 0;
  const loom_value_id_t* callee_args =
      loom_func_like_arg_ids(callee, &callee_arg_count);
  ASSERT_EQ(callee_arg_count, 1u);
  loom_builder_t callee_builder = BodyBuilder(callee_op);
  loom_op_t* neg_op = nullptr;
  IREE_ASSERT_OK(loom_test_neg_build(&callee_builder, callee_args[0], i32,
                                     LOOM_LOCATION_UNKNOWN, &neg_op));
  loom_value_id_t negated = loom_test_neg_result(neg_op);
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&callee_builder, &negated, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));

  loom_op_t* caller_op = nullptr;
  IREE_ASSERT_OK(loom_func_def_build(
      &module_builder_, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
      loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), caller_ref, &i32, 1, &i32, 1, nullptr, 0,
      nullptr, 0, LOOM_LOCATION_UNKNOWN, &caller_op));
  loom_func_like_t caller = loom_func_like_cast(module_, caller_op);
  uint16_t caller_arg_count = 0;
  const loom_value_id_t* caller_args =
      loom_func_like_arg_ids(caller, &caller_arg_count);
  ASSERT_EQ(caller_arg_count, 1u);
  loom_builder_t caller_builder = BodyBuilder(caller_op);
  loom_op_t* invoke_op = nullptr;
  IREE_ASSERT_OK(loom_test_invoke_build(&caller_builder, callee_ref,
                                        caller_args, 1, &i32, 1, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &invoke_op));
  loom_value_id_t invoke_result = loom_test_invoke_results(invoke_op).values[0];
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_func_return_build(&caller_builder, &invoke_result, 1,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  IREE_ASSERT_OK(loom_callable_inline_direct_call(&rewriter, invoke_op));
  loom_rewriter_deinitialize(&rewriter);

  EXPECT_TRUE(iree_any_bit_set(invoke_op->flags, LOOM_OP_FLAG_DEAD));
  loom_block_t* caller_block =
      loom_region_entry_block(loom_func_like_body(caller));
  ASSERT_EQ(caller_block->op_count, 2u);
  ASSERT_TRUE(loom_test_neg_isa(loom_block_op(caller_block, 0)));
  ASSERT_TRUE(loom_func_return_isa(loom_block_op(caller_block, 1)));
  loom_value_slice_t returned =
      loom_func_return_operands(loom_block_op(caller_block, 1));
  ASSERT_EQ(returned.count, 1u);
  EXPECT_EQ(returned.values[0],
            loom_test_neg_result(loom_block_op(caller_block, 0)));
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
  loom_availability_analysis_t availability = InitializeAvailability();
  IREE_ASSERT_OK(loom_callable_inline_consuming_call(&rewriter, &availability,
                                                     call_op, callee));
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

TEST_F(CallableInlineTest, InlinesExactTemplateCall) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t callee_ref = MakeSymbol(IREE_SV("template_negate"));
  loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));
  BuildNegateTemplate(callee_ref, i32);
  loom_op_t* call_op = nullptr;
  loom_op_t* caller_op =
      BuildTemplateCaller(caller_ref, callee_ref, i32, &call_op);

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

TEST_F(CallableInlineTest, RejectsMoveIntoDescendantBlockBeforeMutation) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t function_ref = MakeSymbol(IREE_SV("negate"));
  loom_op_t* function_op = BuildNegateFunction(function_ref, i32);
  loom_block_t* module_block = loom_module_block(module_);
  loom_block_t* body_block = loom_region_entry_block(
      loom_func_like_body(loom_func_like_cast(module_, function_op)));
  ASSERT_NE(body_block, nullptr);

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_rewriter_move_to_block_end(
                            &rewriter, function_op, body_block, function_op));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_rewriter_move_to_block_end(&rewriter, function_op, body_block,
                                      /*target_parent_op=*/nullptr));
  loom_rewriter_deinitialize(&rewriter);

  EXPECT_EQ(function_op->parent_block, module_block);
  EXPECT_EQ(function_op->parent_op, nullptr);
  EXPECT_EQ(module_block->op_count, 1u);
  EXPECT_EQ(body_block->op_count, 2u);
}

TEST_F(CallableInlineTest, CloneDefinitionRemapsOnlySelfReferences) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t helper_ref = MakeSymbol(IREE_SV("helper"));
  BuildNegateFunction(helper_ref, i32);
  loom_symbol_ref_t source_ref = MakeSymbol(IREE_SV("source"));
  loom_op_t* source_self_call = nullptr;
  loom_op_t* source_op =
      BuildCaller(source_ref, source_ref, i32, &source_self_call);

  loom_func_like_t source = loom_func_like_cast(module_, source_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(source, &arg_count);
  ASSERT_EQ(arg_count, 1u);
  loom_block_t* source_block =
      loom_region_entry_block(loom_func_like_body(source));
  loom_op_t* source_return = loom_block_op(source_block, 1);
  loom_builder_t body_builder = BodyBuilder(source_op);
  loom_builder_set_before(&body_builder, source_return);
  loom_op_t* source_helper_call = nullptr;
  IREE_ASSERT_OK(loom_func_call_build(
      &body_builder, 0, 0, 0, 0, helper_ref, args, 1, &i32, 1, nullptr, 0,
      LOOM_LOCATION_UNKNOWN, &source_helper_call));

  loom_symbol_ref_t target_ref = MakeSymbol(IREE_SV("target"));
  loom_func_like_t cloned = {};
  IREE_ASSERT_OK(loom_callable_clone_definition(
      &module_builder_, source, target_ref, &cloned, &rewriter_arena_));

  ASSERT_EQ(source_block->op_count, 3u);
  EXPECT_EQ(loom_func_call_callee(source_self_call).symbol_id,
            source_ref.symbol_id);
  EXPECT_EQ(loom_func_call_callee(source_helper_call).symbol_id,
            helper_ref.symbol_id);

  ASSERT_TRUE(loom_func_like_isa(cloned));
  EXPECT_NE(cloned.op, source.op);
  EXPECT_EQ(loom_func_like_callee(cloned).symbol_id, target_ref.symbol_id);
  EXPECT_EQ(module_->symbols.entries[target_ref.symbol_id].defining_op,
            cloned.op);
  loom_block_t* cloned_block =
      loom_region_entry_block(loom_func_like_body(cloned));
  ASSERT_EQ(cloned_block->op_count, 3u);
  loom_op_t* cloned_self_call = loom_block_op(cloned_block, 0);
  loom_op_t* cloned_helper_call = loom_block_op(cloned_block, 1);
  ASSERT_TRUE(loom_func_call_isa(cloned_self_call));
  ASSERT_TRUE(loom_func_call_isa(cloned_helper_call));
  EXPECT_EQ(loom_func_call_callee(cloned_self_call).symbol_id,
            target_ref.symbol_id);
  EXPECT_EQ(loom_func_call_callee(cloned_helper_call).symbol_id,
            helper_ref.symbol_id);
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

}  // namespace
}  // namespace loom
