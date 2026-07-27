// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Unit tests for the CallLike, FuncLike, LoopLike, and RegionBranch interface
// cast functions and inline accessors. These tests verify the
// .rodata interface vtables on cache line 3 of the op vtable are
// wired correctly: cast() returns a non-null fat reference for ops
// that implement the interface and {NULL, NULL} for ops that don't,
// and the inline accessors return the right block args / regions /
// operands derived from the interface vtable's stored indices.

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

TEST(OpIsaTest, GeneratedIsaReturnsFalseForNull) {
  EXPECT_FALSE(loom_test_addi_isa(nullptr));
}

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class InterfaceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);

    // Register test, scalar, and scf dialects. test provides
    // test.func to host the body and test.constant for value
    // factories; scf provides the loop/branch ops under test;
    // scalar isn't strictly required but kept available for any
    // future expansion.
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_test_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                 vtables, (uint16_t)count));
    vtables = loom_scalar_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_SCALAR,
                                                 vtables, (uint16_t)count));
    vtables = loom_scf_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_SCF,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));

    // Build a test.func as a body to host the ops we test.
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&module_builder,
                                              IREE_SV("host_fn"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    func_ref_ = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    IREE_ASSERT_OK(loom_test_func_build(&module_builder, 0, 0, 0, func_ref_,
                                        NULL, 0, NULL, 0, NULL, 0, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &func_op_));
    func_like_ = loom_func_like_cast(module_, func_op_);
    body_ = loom_func_like_body(func_like_);
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(body_), &builder_);
    builder_.ip.parent_op = func_op_;
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Builds a test.constant of type i32 with the given value.
  loom_op_t* build_i32(int64_t value) {
    loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_test_constant_build(&builder_, loom_attr_i64(value),
                                            i32, LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  // Builds a test.constant of type index with the given value.
  loom_op_t* build_index(int64_t value) {
    loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_test_constant_build(&builder_, loom_attr_i64(value),
                                            index, LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  // Builds a test.constant of type i1 with the given boolean value.
  loom_op_t* build_i1(bool value) {
    loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_test_constant_build(&builder_, loom_attr_bool(value),
                                            i1, LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  void build_region_branch_yield(loom_op_t* owner_op, uint8_t region_index,
                                 const loom_value_id_t* values,
                                 iree_host_size_t value_count) {
    loom_region_branch_t branch = loom_region_branch_cast(module_, owner_op);
    loom_region_t* region =
        loom_region_branch_region(module_, branch, region_index);
    loom_builder_ip_t saved_ip =
        loom_builder_enter_region(&builder_, owner_op, region);
    loom_op_t* yield_op = nullptr;
    IREE_EXPECT_OK(loom_region_branch_build_region_terminator(
        &builder_, module_, branch, region_index, values, value_count,
        LOOM_LOCATION_UNKNOWN, &yield_op));
    loom_builder_restore(&builder_, saved_ip);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_op_t* func_op_ = nullptr;
  loom_symbol_ref_t func_ref_ = {};
  loom_func_like_t func_like_;
  loom_region_t* body_ = nullptr;
  loom_builder_t builder_;
};

//===----------------------------------------------------------------------===//
// CallLike interface
//===----------------------------------------------------------------------===//

TEST_F(InterfaceTest, CallLikeCastReturnsValidForInvoke) {
  loom_op_t* input = build_i32(42);
  loom_value_id_t input_id = loom_op_results(input)[0];
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* invoke_op = nullptr;
  IREE_ASSERT_OK(loom_test_invoke_build(&builder_, func_ref_, &input_id, 1,
                                        &i32, 1, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &invoke_op));

  loom_call_like_t call = loom_call_like_cast(module_, invoke_op);
  EXPECT_TRUE(loom_call_like_isa(call));
  EXPECT_EQ(call.op, invoke_op);
  EXPECT_NE(call.vtable, nullptr);
  EXPECT_EQ(loom_call_like_kind(call), LOOM_CALL_LIKE_KIND_SEMANTIC);
  EXPECT_EQ(loom_call_like_purity(call), 0);
  EXPECT_EQ(loom_call_like_operand_offset(call), 0);
  EXPECT_EQ(loom_call_like_result_offset(call), 0);
  EXPECT_EQ(loom_call_like_callee(call).symbol_id, func_ref_.symbol_id);

  loom_value_slice_t operands = loom_call_like_operands(call);
  ASSERT_EQ(operands.count, 1);
  EXPECT_EQ(operands.values[0], input_id);
  loom_value_slice_t results = loom_call_like_results(call);
  ASSERT_EQ(results.count, 1);
  EXPECT_EQ(results.values[0], loom_op_results(invoke_op)[0]);
}

TEST_F(InterfaceTest, CallLikeCastReturnsNullForNonCall) {
  loom_op_t* constant_op = build_i32(42);
  loom_call_like_t call = loom_call_like_cast(module_, constant_op);
  EXPECT_FALSE(loom_call_like_isa(call));
  EXPECT_EQ(call.op, nullptr);
  EXPECT_EQ(call.vtable, nullptr);
}

TEST_F(InterfaceTest, CallLikeCastReturnsNullForNullOp) {
  loom_call_like_t call = loom_call_like_cast(module_, nullptr);
  EXPECT_FALSE(loom_call_like_isa(call));
  EXPECT_EQ(call.op, nullptr);
  EXPECT_EQ(call.vtable, nullptr);
}

//===----------------------------------------------------------------------===//
// FuncLike interface
//===----------------------------------------------------------------------===//

TEST_F(InterfaceTest, FuncLikeCastReturnsValidForFunc) {
  loom_func_like_t func_like = loom_func_like_cast(module_, func_op_);
  EXPECT_TRUE(loom_func_like_isa(func_like));
  EXPECT_EQ(func_like.op, func_op_);
  EXPECT_NE(func_like.vtable, nullptr);
  EXPECT_EQ(loom_func_like_body(func_like), body_);
}

TEST_F(InterfaceTest, FuncLikeCastReturnsNullForNonFunc) {
  loom_op_t* constant_op = build_i32(42);
  loom_func_like_t func_like = loom_func_like_cast(module_, constant_op);
  EXPECT_FALSE(loom_func_like_isa(func_like));
  EXPECT_EQ(func_like.op, nullptr);
  EXPECT_EQ(func_like.vtable, nullptr);
}

TEST_F(InterfaceTest, FuncLikeCastReturnsNullForNullOp) {
  loom_func_like_t func_like = loom_func_like_cast(module_, nullptr);
  EXPECT_FALSE(loom_func_like_isa(func_like));
  EXPECT_EQ(func_like.op, nullptr);
  EXPECT_EQ(func_like.vtable, nullptr);
}

//===----------------------------------------------------------------------===//
// LoopLike interface
//===----------------------------------------------------------------------===//

TEST_F(InterfaceTest, LoopLikeCastReturnsValidForScfFor) {
  loom_op_t* lower = build_index(0);
  loom_op_t* upper = build_index(8);
  loom_op_t* step = build_index(1);
  loom_value_id_t lower_id = loom_op_results(lower)[0];
  loom_value_id_t upper_id = loom_op_results(upper)[0];
  loom_value_id_t step_id = loom_op_results(step)[0];

  loom_op_t* for_op = nullptr;
  IREE_ASSERT_OK(loom_scf_for_build(
      &builder_, /*build_flags=*/0, lower_id, upper_id, step_id, nullptr, 0,
      nullptr, 0, nullptr, 0, LOOM_VALUE_ID_INVALID, /*unroll_policy=*/0,
      /*unroll_schedule=*/0, LOOM_LOCATION_UNKNOWN, &for_op));

  loom_loop_like_t loop = loom_loop_like_cast(module_, for_op);
  EXPECT_TRUE(loom_loop_like_isa(loop));
  EXPECT_EQ(loop.op, for_op);
  EXPECT_NE(loop.vtable, nullptr);
}

TEST_F(InterfaceTest, LoopLikeCastReturnsNullForNonLoop) {
  loom_op_t* constant_op = build_i32(42);
  loom_loop_like_t loop = loom_loop_like_cast(module_, constant_op);
  EXPECT_FALSE(loom_loop_like_isa(loop));
  EXPECT_EQ(loop.op, nullptr);
  EXPECT_EQ(loop.vtable, nullptr);
}

TEST_F(InterfaceTest, LoopLikeCastReturnsNullForNullOp) {
  loom_loop_like_t loop = loom_loop_like_cast(module_, nullptr);
  EXPECT_FALSE(loom_loop_like_isa(loop));
}

TEST_F(InterfaceTest, LoopLikeCastReturnsNullForScfIf) {
  // scf.if is region-branch-like, not loop-like.
  loom_op_t* condition = build_i1(true);
  loom_value_id_t condition_id = loom_op_results(condition)[0];
  loom_op_t* if_op = nullptr;
  IREE_ASSERT_OK(loom_scf_if_build(
      &builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION, condition_id, nullptr,
      0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &if_op));

  loom_loop_like_t loop = loom_loop_like_cast(module_, if_op);
  EXPECT_FALSE(loom_loop_like_isa(loop));
}

TEST_F(InterfaceTest, LoopLikeAccessorsForScfFor) {
  loom_op_t* lower = build_index(0);
  loom_op_t* upper = build_index(16);
  loom_op_t* step = build_index(1);
  loom_value_id_t lower_id = loom_op_results(lower)[0];
  loom_value_id_t upper_id = loom_op_results(upper)[0];
  loom_value_id_t step_id = loom_op_results(step)[0];

  loom_op_t* for_op = nullptr;
  IREE_ASSERT_OK(loom_scf_for_build(
      &builder_, /*build_flags=*/0, lower_id, upper_id, step_id, nullptr, 0,
      nullptr, 0, nullptr, 0, LOOM_VALUE_ID_INVALID, /*unroll_policy=*/0,
      /*unroll_schedule=*/0, LOOM_LOCATION_UNKNOWN, &for_op));

  loom_loop_like_t loop = loom_loop_like_cast(module_, for_op);
  ASSERT_TRUE(loom_loop_like_isa(loop));

  // body region matches the auto-created body region.
  loom_region_t* body = loom_loop_like_body(loop);
  ASSERT_NE(body, nullptr);
  EXPECT_EQ(body, loom_scf_for_body(for_op));

  // scf.for has no separate condition region.
  EXPECT_EQ(loom_loop_like_condition_region(loop), nullptr);

  // The IV is the first block arg of the body's entry block.
  loom_value_id_t iv = loom_loop_like_iv(loop);
  EXPECT_NE(iv, LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(iv, loom_block_arg_id(loom_region_const_entry_block(body), 0));

  // Counted-loop range operands are exposed generically.
  EXPECT_EQ(loom_loop_like_lower_bound(loop), lower_id);
  EXPECT_EQ(loom_loop_like_upper_bound(loop), upper_id);
  EXPECT_EQ(loom_loop_like_step(loop), step_id);
  EXPECT_TRUE(loom_loop_like_has_counted_range(loop));
}

TEST_F(InterfaceTest, LoopLikeIterArgsEmpty) {
  loom_op_t* lower = build_index(0);
  loom_op_t* upper = build_index(8);
  loom_op_t* step = build_index(1);
  loom_op_t* for_op = nullptr;
  IREE_ASSERT_OK(loom_scf_for_build(
      &builder_, /*build_flags=*/0, loom_op_results(lower)[0],
      loom_op_results(upper)[0], loom_op_results(step)[0], nullptr, 0, nullptr,
      0, nullptr, 0, LOOM_VALUE_ID_INVALID, /*unroll_policy=*/0,
      /*unroll_schedule=*/0, LOOM_LOCATION_UNKNOWN, &for_op));

  loom_loop_like_t loop = loom_loop_like_cast(module_, for_op);
  loom_value_slice_t iter_args = loom_loop_like_iter_args(loop);
  EXPECT_EQ(iter_args.count, 0);
}

TEST_F(InterfaceTest, LoopLikeIterArgsNonEmpty) {
  loom_op_t* lower = build_index(0);
  loom_op_t* upper = build_index(8);
  loom_op_t* step = build_index(1);
  loom_op_t* factor = build_index(2);
  loom_op_t* init0 = build_i32(10);
  loom_op_t* init1 = build_i32(20);
  loom_value_id_t init_ids[2] = {loom_op_results(init0)[0],
                                 loom_op_results(init1)[0]};
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t result_types[2] = {i32, i32};

  loom_op_t* for_op = nullptr;
  IREE_ASSERT_OK(
      loom_scf_for_build(&builder_, LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_FACTOR,
                         loom_op_results(lower)[0], loom_op_results(upper)[0],
                         loom_op_results(step)[0], init_ids, 2, result_types, 2,
                         nullptr, 0, loom_op_results(factor)[0],
                         /*unroll_policy=*/0, /*unroll_schedule=*/0,
                         LOOM_LOCATION_UNKNOWN, &for_op));

  loom_loop_like_t loop = loom_loop_like_cast(module_, for_op);
  loom_value_slice_t iter_args = loom_loop_like_iter_args(loop);
  EXPECT_EQ(iter_args.count, 2);
  EXPECT_EQ(iter_args.values[0], init_ids[0]);
  EXPECT_EQ(iter_args.values[1], init_ids[1]);
}

//===----------------------------------------------------------------------===//
// RegionBranch interface
//===----------------------------------------------------------------------===//

TEST_F(InterfaceTest, RegionBranchCastReturnsValidForScfIf) {
  loom_op_t* condition = build_i1(true);
  loom_value_id_t condition_id = loom_op_results(condition)[0];
  loom_op_t* if_op = nullptr;
  IREE_ASSERT_OK(loom_scf_if_build(
      &builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION, condition_id, nullptr,
      0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &if_op));

  loom_region_branch_t branch = loom_region_branch_cast(module_, if_op);
  EXPECT_TRUE(loom_region_branch_isa(branch));
  EXPECT_EQ(branch.op, if_op);
  EXPECT_NE(branch.vtable, nullptr);
}

TEST_F(InterfaceTest, RegionBranchCastReturnsNullForNonBranch) {
  loom_op_t* constant_op = build_i32(42);
  loom_region_branch_t branch = loom_region_branch_cast(module_, constant_op);
  EXPECT_FALSE(loom_region_branch_isa(branch));
  EXPECT_EQ(branch.op, nullptr);
  EXPECT_EQ(branch.vtable, nullptr);
}

TEST_F(InterfaceTest, RegionBranchCastReturnsNullForNullOp) {
  loom_region_branch_t branch = loom_region_branch_cast(module_, nullptr);
  EXPECT_FALSE(loom_region_branch_isa(branch));
}

TEST_F(InterfaceTest, RegionBranchCastReturnsNullForScfFor) {
  // scf.for is loop-like, not region-branch-like.
  loom_op_t* lower = build_index(0);
  loom_op_t* upper = build_index(8);
  loom_op_t* step = build_index(1);
  loom_op_t* for_op = nullptr;
  IREE_ASSERT_OK(loom_scf_for_build(
      &builder_, /*build_flags=*/0, loom_op_results(lower)[0],
      loom_op_results(upper)[0], loom_op_results(step)[0], nullptr, 0, nullptr,
      0, nullptr, 0, LOOM_VALUE_ID_INVALID, /*unroll_policy=*/0,
      /*unroll_schedule=*/0, LOOM_LOCATION_UNKNOWN, &for_op));

  loom_region_branch_t branch = loom_region_branch_cast(module_, for_op);
  EXPECT_FALSE(loom_region_branch_isa(branch));
}

TEST_F(InterfaceTest, RegionBranchSelectorForScfIf) {
  loom_op_t* condition = build_i1(false);
  loom_value_id_t condition_id = loom_op_results(condition)[0];
  loom_op_t* if_op = nullptr;
  IREE_ASSERT_OK(loom_scf_if_build(
      &builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION, condition_id, nullptr,
      0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &if_op));

  loom_region_branch_t branch = loom_region_branch_cast(module_, if_op);
  ASSERT_TRUE(loom_region_branch_isa(branch));
  EXPECT_EQ(loom_region_branch_selector(branch), condition_id);
}

TEST_F(InterfaceTest, RegionBranchYieldOnlyOperandsForScfIf) {
  loom_op_t* condition = build_i1(false);
  loom_op_t* then_value = build_i32(10);
  loom_op_t* else_value = build_i32(20);
  loom_value_id_t then_id = loom_op_results(then_value)[0];
  loom_value_id_t else_id = loom_op_results(else_value)[0];
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* if_op = nullptr;
  IREE_ASSERT_OK(loom_scf_if_build(&builder_,
                                   LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                                   loom_op_results(condition)[0], &i32, 1,
                                   nullptr, 0, LOOM_LOCATION_UNKNOWN, &if_op));
  build_region_branch_yield(if_op, 0, &then_id, 1);
  build_region_branch_yield(if_op, 1, &else_id, 1);

  loom_region_branch_t branch = loom_region_branch_cast(module_, if_op);
  ASSERT_TRUE(loom_region_branch_isa(branch));
  EXPECT_EQ(loom_region_branch_region(module_, branch, 0),
            loom_scf_if_then_region(if_op));
  EXPECT_EQ(loom_region_branch_region(module_, branch, 1),
            loom_scf_if_else_region(if_op));
  EXPECT_EQ(loom_region_branch_region(module_, branch, 2), nullptr);

  loom_op_t* terminator =
      loom_region_branch_region_terminator(module_, branch, 0);
  ASSERT_NE(terminator, nullptr);
  EXPECT_TRUE(loom_scf_yield_isa(terminator));

  loom_value_slice_t yielded_values = {0};
  EXPECT_TRUE(loom_region_branch_region_yield_only_operands(
      module_, branch, 0, 1, &yielded_values));
  ASSERT_EQ(yielded_values.count, 1);
  EXPECT_EQ(yielded_values.values[0], then_id);
  EXPECT_FALSE(loom_region_branch_region_yield_only_operands(
      module_, branch, 0, 2, &yielded_values));
}

TEST_F(InterfaceTest, RegionBranchYieldOnlyRejectsBranchBody) {
  loom_op_t* condition = build_i1(false);
  loom_op_t* fallback_value = build_i32(20);
  loom_value_id_t fallback_id = loom_op_results(fallback_value)[0];
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* if_op = nullptr;
  IREE_ASSERT_OK(loom_scf_if_build(&builder_,
                                   LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                                   loom_op_results(condition)[0], &i32, 1,
                                   nullptr, 0, LOOM_LOCATION_UNKNOWN, &if_op));

  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &builder_, if_op, loom_scf_if_then_region(if_op));
  loom_op_t* local_value = build_i32(10);
  loom_value_id_t local_id = loom_op_results(local_value)[0];
  loom_op_t* yield_op = nullptr;
  loom_region_branch_t branch = loom_region_branch_cast(module_, if_op);
  IREE_EXPECT_OK(loom_region_branch_build_region_terminator(
      &builder_, module_, branch, 0, &local_id, 1, LOOM_LOCATION_UNKNOWN,
      &yield_op));
  loom_builder_restore(&builder_, saved_ip);
  build_region_branch_yield(if_op, 1, &fallback_id, 1);

  ASSERT_TRUE(loom_region_branch_isa(branch));
  EXPECT_NE(loom_region_branch_region_terminator(module_, branch, 0), nullptr);

  loom_value_slice_t yielded_values = {0};
  EXPECT_FALSE(loom_region_branch_region_yield_only_operands(
      module_, branch, 0, 1, &yielded_values));
}

TEST_F(InterfaceTest, RegionBranchRegionsForScfSwitch) {
  loom_op_t* selector = build_index(1);
  loom_op_t* case0_value = build_i32(10);
  loom_op_t* case1_value = build_i32(20);
  loom_op_t* default_value = build_i32(30);
  loom_value_id_t case0_id = loom_op_results(case0_value)[0];
  loom_value_id_t case1_id = loom_op_results(case1_value)[0];
  loom_value_id_t default_id = loom_op_results(default_value)[0];
  int64_t case_keys[2] = {0, 1};
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* switch_op = nullptr;
  IREE_ASSERT_OK(loom_scf_switch_build(
      &builder_, loom_op_results(selector)[0], &i32, 1, nullptr, 0, case_keys,
      IREE_ARRAYSIZE(case_keys), LOOM_LOCATION_UNKNOWN, &switch_op));
  build_region_branch_yield(switch_op, 0, &default_id, 1);
  loom_region_slice_t case_regions = loom_scf_switch_case_regions(switch_op);
  ASSERT_EQ(case_regions.count, 2);
  build_region_branch_yield(switch_op, 1, &case0_id, 1);
  build_region_branch_yield(switch_op, 2, &case1_id, 1);

  loom_region_branch_t branch = loom_region_branch_cast(module_, switch_op);
  ASSERT_TRUE(loom_region_branch_isa(branch));
  EXPECT_EQ(loom_region_branch_selector(branch), loom_op_results(selector)[0]);
  EXPECT_EQ(loom_region_branch_region(module_, branch, 0),
            loom_scf_switch_default_region(switch_op));
  EXPECT_EQ(loom_region_branch_region(module_, branch, 1),
            case_regions.regions[0]);
  EXPECT_EQ(loom_region_branch_region(module_, branch, 2),
            case_regions.regions[1]);
  EXPECT_EQ(loom_region_branch_region(module_, branch, 3), nullptr);

  loom_value_slice_t yielded_values = {0};
  EXPECT_TRUE(loom_region_branch_region_yield_only_operands(
      module_, branch, 2, 1, &yielded_values));
  ASSERT_EQ(yielded_values.count, 1);
  EXPECT_EQ(yielded_values.values[0], case1_id);
}

}  // namespace
}  // namespace loom
