// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Unit tests for generic operation interface casts and accessors. Synthetic
// test ops cover each supported field layout without coupling the generic
// interface contract to production dialects.

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
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

    // The synthetic test dialect owns every operation shape used below.
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_test_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
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

  loom_string_id_t replacement_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("replacement"),
                                           &replacement_name_id));
  uint16_t replacement_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, replacement_name_id,
                                        &replacement_symbol_id));
  const loom_symbol_ref_t replacement_ref = {
      /*.module_id=*/0,
      /*.symbol_id=*/replacement_symbol_id,
  };
  loom_call_like_set_callee(module_, call, replacement_ref);
  EXPECT_EQ(loom_call_like_callee(call).symbol_id, replacement_symbol_id);

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

TEST_F(InterfaceTest, CallLikeSpansTrailingOperandPartitions) {
  loom_value_id_t prefix[] = {
      loom_op_results(build_index(16))[0],
      loom_op_results(build_index(32))[0],
  };
  loom_value_id_t specializations[] = {
      loom_op_results(build_index(64))[0],
  };
  loom_value_id_t bindings[] = {
      loom_op_results(build_i32(42))[0],
  };
  loom_op_t* call_op = nullptr;
  IREE_ASSERT_OK(loom_test_partitioned_call_build(
      &builder_, func_ref_, prefix, IREE_ARRAYSIZE(prefix), specializations,
      IREE_ARRAYSIZE(specializations), bindings, IREE_ARRAYSIZE(bindings),
      LOOM_LOCATION_UNKNOWN, &call_op));

  loom_call_like_t call = loom_call_like_cast(module_, call_op);
  EXPECT_EQ(loom_call_like_kind(call), LOOM_CALL_LIKE_KIND_COMMAND_PROGRAM);
  EXPECT_EQ(loom_call_like_operand_offset(call), 2);
  loom_value_slice_t operands = loom_call_like_operands(call);
  ASSERT_EQ(operands.count, 2);
  EXPECT_EQ(operands.values[0], specializations[0]);
  EXPECT_EQ(operands.values[1], bindings[0]);
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
  EXPECT_EQ(loom_func_like_repr_contract(func_like), LOOM_STRING_ID_INVALID);
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

TEST_F(InterfaceTest, LoopLikeCastReturnsValidForLoop) {
  loom_op_t* lower = build_index(0);
  loom_op_t* upper = build_index(8);
  loom_op_t* step = build_index(1);
  loom_value_id_t lower_id = loom_op_results(lower)[0];
  loom_value_id_t upper_id = loom_op_results(upper)[0];
  loom_value_id_t step_id = loom_op_results(step)[0];

  loom_op_t* loop_op = nullptr;
  IREE_ASSERT_OK(loom_test_loop_build(&builder_, lower_id, upper_id, step_id,
                                      nullptr, 0, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &loop_op));

  loom_loop_like_t loop = loom_loop_like_cast(module_, loop_op);
  EXPECT_TRUE(loom_loop_like_isa(loop));
  EXPECT_EQ(loop.op, loop_op);
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

TEST_F(InterfaceTest, LoopLikeCastReturnsNullForBranch) {
  // test.branch is region-branch-like, not loop-like.
  loom_op_t* condition = build_i1(true);
  loom_value_id_t condition_id = loom_op_results(condition)[0];
  loom_op_t* branch_op = nullptr;
  IREE_ASSERT_OK(loom_test_branch_build(&builder_, condition_id, nullptr, 0,
                                        nullptr, 0, LOOM_LOCATION_UNKNOWN,
                                        &branch_op));

  loom_loop_like_t loop = loom_loop_like_cast(module_, branch_op);
  EXPECT_FALSE(loom_loop_like_isa(loop));
}

TEST_F(InterfaceTest, LoopLikeAccessorsForLoop) {
  loom_op_t* lower = build_index(0);
  loom_op_t* upper = build_index(16);
  loom_op_t* step = build_index(1);
  loom_value_id_t lower_id = loom_op_results(lower)[0];
  loom_value_id_t upper_id = loom_op_results(upper)[0];
  loom_value_id_t step_id = loom_op_results(step)[0];

  loom_op_t* loop_op = nullptr;
  IREE_ASSERT_OK(loom_test_loop_build(&builder_, lower_id, upper_id, step_id,
                                      nullptr, 0, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &loop_op));

  loom_loop_like_t loop = loom_loop_like_cast(module_, loop_op);
  ASSERT_TRUE(loom_loop_like_isa(loop));

  // The generic body accessor returns the operation's body region.
  loom_region_t* body = loom_loop_like_body(loop);
  ASSERT_NE(body, nullptr);
  EXPECT_EQ(body, loom_test_loop_body(loop_op));

  // A counted loop has no separate condition region.
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
  loom_op_t* loop_op = nullptr;
  IREE_ASSERT_OK(loom_test_loop_build(
      &builder_, loom_op_results(lower)[0], loom_op_results(upper)[0],
      loom_op_results(step)[0], nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN,
      &loop_op));

  loom_loop_like_t loop = loom_loop_like_cast(module_, loop_op);
  loom_value_slice_t iter_args = loom_loop_like_iter_args(loop);
  EXPECT_EQ(iter_args.count, 0);
}

TEST_F(InterfaceTest, LoopLikeIterArgsNonEmpty) {
  loom_op_t* lower = build_index(0);
  loom_op_t* upper = build_index(8);
  loom_op_t* step = build_index(1);
  loom_op_t* init0 = build_i32(10);
  loom_op_t* init1 = build_i32(20);
  loom_value_id_t init_ids[2] = {loom_op_results(init0)[0],
                                 loom_op_results(init1)[0]};

  loom_op_t* loop_op = nullptr;
  IREE_ASSERT_OK(loom_test_loop_build(
      &builder_, loom_op_results(lower)[0], loom_op_results(upper)[0],
      loom_op_results(step)[0], init_ids, IREE_ARRAYSIZE(init_ids), nullptr, 0,
      LOOM_LOCATION_UNKNOWN, &loop_op));

  loom_loop_like_t loop = loom_loop_like_cast(module_, loop_op);
  loom_value_slice_t iter_args = loom_loop_like_iter_args(loop);
  EXPECT_EQ(iter_args.count, 2);
  EXPECT_EQ(iter_args.values[0], init_ids[0]);
  EXPECT_EQ(iter_args.values[1], init_ids[1]);
  ASSERT_EQ(loop_op->result_count, 2);
  const loom_value_id_t* results = loom_op_const_results(loop_op);
  for (uint16_t i = 0; i < loop_op->result_count; ++i) {
    EXPECT_TRUE(loom_type_equal(loom_module_value_type(module_, results[i]),
                                loom_module_value_type(module_, init_ids[i])));
  }
}

//===----------------------------------------------------------------------===//
// RegionBranch interface
//===----------------------------------------------------------------------===//

TEST_F(InterfaceTest, RegionBranchCastReturnsValidForBranch) {
  loom_op_t* condition = build_i1(true);
  loom_value_id_t condition_id = loom_op_results(condition)[0];
  loom_op_t* branch_op = nullptr;
  IREE_ASSERT_OK(loom_test_branch_build(&builder_, condition_id, nullptr, 0,
                                        nullptr, 0, LOOM_LOCATION_UNKNOWN,
                                        &branch_op));

  loom_region_branch_t branch = loom_region_branch_cast(module_, branch_op);
  EXPECT_TRUE(loom_region_branch_isa(branch));
  EXPECT_EQ(branch.op, branch_op);
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

TEST_F(InterfaceTest, RegionBranchCastReturnsNullForLoop) {
  // test.loop is loop-like, not region-branch-like.
  loom_op_t* lower = build_index(0);
  loom_op_t* upper = build_index(8);
  loom_op_t* step = build_index(1);
  loom_op_t* loop_op = nullptr;
  IREE_ASSERT_OK(loom_test_loop_build(
      &builder_, loom_op_results(lower)[0], loom_op_results(upper)[0],
      loom_op_results(step)[0], nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN,
      &loop_op));

  loom_region_branch_t branch = loom_region_branch_cast(module_, loop_op);
  EXPECT_FALSE(loom_region_branch_isa(branch));
}

TEST_F(InterfaceTest, RegionBranchSelectorForBranch) {
  loom_op_t* condition = build_i1(false);
  loom_value_id_t condition_id = loom_op_results(condition)[0];
  loom_op_t* branch_op = nullptr;
  IREE_ASSERT_OK(loom_test_branch_build(&builder_, condition_id, nullptr, 0,
                                        nullptr, 0, LOOM_LOCATION_UNKNOWN,
                                        &branch_op));

  loom_region_branch_t branch = loom_region_branch_cast(module_, branch_op);
  ASSERT_TRUE(loom_region_branch_isa(branch));
  EXPECT_EQ(loom_region_branch_selector(branch), condition_id);
}

TEST_F(InterfaceTest, RegionBranchYieldOnlyOperandsForBranch) {
  loom_op_t* condition = build_i1(false);
  loom_op_t* then_value = build_i32(10);
  loom_op_t* else_value = build_i32(20);
  loom_value_id_t then_id = loom_op_results(then_value)[0];
  loom_value_id_t else_id = loom_op_results(else_value)[0];
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* branch_op = nullptr;
  IREE_ASSERT_OK(
      loom_test_branch_build(&builder_, loom_op_results(condition)[0], &i32, 1,
                             nullptr, 0, LOOM_LOCATION_UNKNOWN, &branch_op));
  build_region_branch_yield(branch_op, 0, &then_id, 1);
  build_region_branch_yield(branch_op, 1, &else_id, 1);

  loom_region_branch_t branch = loom_region_branch_cast(module_, branch_op);
  ASSERT_TRUE(loom_region_branch_isa(branch));
  EXPECT_EQ(loom_region_branch_region(module_, branch, 0),
            loom_test_branch_then_region(branch_op));
  EXPECT_EQ(loom_region_branch_region(module_, branch, 1),
            loom_test_branch_else_region(branch_op));
  EXPECT_EQ(loom_region_branch_region(module_, branch, 2), nullptr);

  loom_op_t* terminator =
      loom_region_branch_region_terminator(module_, branch, 0);
  ASSERT_NE(terminator, nullptr);
  EXPECT_TRUE(loom_test_yield_isa(terminator));

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

  loom_op_t* branch_op = nullptr;
  IREE_ASSERT_OK(
      loom_test_branch_build(&builder_, loom_op_results(condition)[0], &i32, 1,
                             nullptr, 0, LOOM_LOCATION_UNKNOWN, &branch_op));

  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &builder_, branch_op, loom_test_branch_then_region(branch_op));
  loom_op_t* local_value = build_i32(10);
  loom_value_id_t local_id = loom_op_results(local_value)[0];
  loom_op_t* yield_op = nullptr;
  loom_region_branch_t branch = loom_region_branch_cast(module_, branch_op);
  IREE_EXPECT_OK(loom_region_branch_build_region_terminator(
      &builder_, module_, branch, 0, &local_id, 1, LOOM_LOCATION_UNKNOWN,
      &yield_op));
  loom_builder_restore(&builder_, saved_ip);
  build_region_branch_yield(branch_op, 1, &fallback_id, 1);

  ASSERT_TRUE(loom_region_branch_isa(branch));
  EXPECT_NE(loom_region_branch_region_terminator(module_, branch, 0), nullptr);

  loom_value_slice_t yielded_values = {0};
  EXPECT_FALSE(loom_region_branch_region_yield_only_operands(
      module_, branch, 0, 1, &yielded_values));
}

TEST_F(InterfaceTest, RegionBranchRegionsForRegionTable) {
  loom_op_t* selector = build_index(1);
  int64_t case_keys[2] = {0, 1};

  loom_op_t* table_op = nullptr;
  IREE_ASSERT_OK(loom_test_region_table_build(
      &builder_, loom_op_results(selector)[0], case_keys,
      IREE_ARRAYSIZE(case_keys), LOOM_LOCATION_UNKNOWN, &table_op));
  build_region_branch_yield(table_op, 0, nullptr, 0);
  loom_region_slice_t case_regions =
      loom_test_region_table_case_regions(table_op);
  ASSERT_EQ(case_regions.count, 2);
  build_region_branch_yield(table_op, 1, nullptr, 0);
  build_region_branch_yield(table_op, 2, nullptr, 0);

  loom_region_branch_t branch = loom_region_branch_cast(module_, table_op);
  ASSERT_TRUE(loom_region_branch_isa(branch));
  EXPECT_EQ(loom_region_branch_selector(branch), loom_op_results(selector)[0]);
  EXPECT_EQ(loom_region_branch_region(module_, branch, 0),
            loom_test_region_table_default_region(table_op));
  EXPECT_EQ(loom_region_branch_region(module_, branch, 1),
            case_regions.regions[0]);
  EXPECT_EQ(loom_region_branch_region(module_, branch, 2),
            case_regions.regions[1]);
  EXPECT_EQ(loom_region_branch_region(module_, branch, 3), nullptr);

  loom_value_slice_t yielded_values = {0};
  EXPECT_TRUE(loom_region_branch_region_yield_only_operands(
      module_, branch, 2, 0, &yielded_values));
  EXPECT_EQ(yielded_values.count, 0);
}

}  // namespace
}  // namespace loom
