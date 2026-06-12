// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/greedy.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"
#include "loom/target/types.h"

namespace loom {
namespace {

class GreedyRewriteTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_test_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));

    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&module_builder,
                                              IREE_SV("test_fn"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_test_func_build(&module_builder, 0, 0, 0, callee, NULL,
                                        0, NULL, 0, NULL, 0, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &func_op));
    function_ = loom_func_like_cast(module_, func_op);
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_func_like_body(function_)), &builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t function_;
  loom_builder_t builder_;
};

static iree_status_t pattern_one_to_two(const loom_pattern_t*, loom_op_t* op,
                                        loom_rewriter_t* rewriter) {
  if (!loom_test_constant_isa(op)) return iree_ok_status();
  int64_t value = loom_attr_as_i64(loom_op_attrs(op)[0]);
  if (value != 1) return iree_ok_status();
  loom_value_id_t old_result = loom_test_constant_result(op);
  loom_type_t type = loom_module_value_type(rewriter->module, old_result);
  loom_op_t* replacement = NULL;
  IREE_RETURN_IF_ERROR(loom_test_constant_build(
      &rewriter->builder, loom_attr_i64(2), type, op->location, &replacement));
  loom_value_id_t new_result = loom_test_constant_result(replacement);
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &new_result, 1);
}

static iree_status_t pattern_two_no_match(const loom_pattern_t*, loom_op_t*,
                                          loom_rewriter_t*) {
  return iree_ok_status();
}

static iree_status_t pattern_two_to_ten(const loom_pattern_t*, loom_op_t* op,
                                        loom_rewriter_t* rewriter) {
  if (!loom_test_constant_isa(op)) return iree_ok_status();
  int64_t value = loom_attr_as_i64(loom_op_attrs(op)[0]);
  if (value != 2) return iree_ok_status();
  loom_value_id_t old_result = loom_test_constant_result(op);
  loom_type_t type = loom_module_value_type(rewriter->module, old_result);
  loom_op_t* replacement = NULL;
  IREE_RETURN_IF_ERROR(loom_test_constant_build(
      &rewriter->builder, loom_attr_i64(10), type, op->location, &replacement));
  loom_value_id_t new_result = loom_test_constant_result(replacement);
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &new_result, 1);
}

static iree_status_t pattern_two_error(const loom_pattern_t*, loom_op_t* op,
                                       loom_rewriter_t*) {
  if (!loom_test_constant_isa(op)) return iree_ok_status();
  int64_t value = loom_attr_as_i64(loom_op_attrs(op)[0]);
  if (value != 2) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INTERNAL, "pattern error on value 2");
}

TEST_F(GreedyRewriteTest, ChainedPatternsReachFixedPoint) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  loom_value_id_t original = loom_test_constant_result(const_op);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &original, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));

  loom_pattern_t patterns[] = {
      {LOOM_OP_TEST_CONSTANT, pattern_one_to_two},
      {LOOM_OP_TEST_CONSTANT, pattern_two_no_match},
      {LOOM_OP_TEST_CONSTANT, pattern_two_to_ten},
  };

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  IREE_ASSERT_OK(
      loom_greedy_rewrite(&arena, module_, function_, patterns, 3, NULL));
  iree_arena_deinitialize(&arena);

  loom_value_id_t final_result = loom_op_operands(use)[0];
  loom_value_t* value = loom_module_value(module_, final_result);
  loom_op_t* final_const = loom_value_def_op(value);
  ASSERT_NE(final_const, nullptr);
  EXPECT_EQ(loom_attr_as_i64(loom_op_attrs(final_const)[0]), 10);
}

TEST_F(GreedyRewriteTest, PatternErrorPropagates) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  loom_value_id_t original = loom_test_constant_result(const_op);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &original, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));

  loom_pattern_t patterns[] = {
      {LOOM_OP_TEST_CONSTANT, pattern_one_to_two},
      {LOOM_OP_TEST_CONSTANT, pattern_two_error},
  };

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      loom_greedy_rewrite(&arena, module_, function_, patterns, 2, NULL));
  iree_arena_deinitialize(&arena);
}

TEST_F(GreedyRewriteTest, UnmatchedPatternsLeaveIrUntouched) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));

  loom_pattern_t patterns[] = {
      {LOOM_OP_TEST_CONSTANT, pattern_one_to_two},
      {LOOM_OP_TEST_CONSTANT, pattern_two_to_ten},
  };

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  IREE_ASSERT_OK(
      loom_greedy_rewrite(&arena, module_, function_, patterns, 2, NULL));
  iree_arena_deinitialize(&arena);

  EXPECT_EQ(loom_attr_as_i64(loom_op_attrs(const_op)[0]), 42);
}

TEST_F(GreedyRewriteTest, SeedFactsPreserveTargetBundleScope) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  ASSERT_NE(const_op, nullptr);

  loom_target_bundle_t target_bundle = {};
  iree_arena_allocator_t seed_arena;
  iree_arena_initialize(&block_pool_, &seed_arena);
  loom_value_fact_table_t seed_facts;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&seed_facts, &seed_arena,
                                                  module_->values.capacity));
  seed_facts.context.target_bundle = &target_bundle;

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_pass_value_fact_owner_t fact_owner;
  loom_pass_value_fact_owner_initialize(&block_pool_, &fact_owner);
  loom_greedy_rewrite_driver_t driver;
  loom_greedy_rewrite_driver_initialize(module_, &arena, &fact_owner, &driver);

  const loom_greedy_rewrite_options_t options = {
      /*.max_iterations=*/{},
      /*.seed_facts=*/&seed_facts,
  };
  IREE_ASSERT_OK(loom_greedy_rewrite_run_region(
      &driver, function_, loom_func_like_body(function_), function_.op,
      &options, /*callbacks=*/NULL, /*out_result=*/NULL));

  const loom_value_fact_table_t* latest_facts =
      loom_greedy_rewrite_driver_fact_table(&driver);
  ASSERT_NE(latest_facts, nullptr);
  EXPECT_EQ(latest_facts->context.target_bundle, &target_bundle);

  loom_greedy_rewrite_driver_deinitialize(&driver);
  loom_pass_value_fact_owner_deinitialize(&fact_owner);
  iree_arena_deinitialize(&arena);
  iree_arena_deinitialize(&seed_arena);
}

TEST_F(GreedyRewriteTest, NamePolicyCanDisableOptionalNames) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module_, index_type, &source));
  loom_value_id_t target = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module_, index_type, &target));

  loom_string_id_t source_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("head"), &source_name));
  IREE_ASSERT_OK(loom_module_set_value_name(module_, source, source_name));

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_rewriter_t rewriter;
  IREE_ASSERT_OK(loom_rewriter_initialize(&rewriter, module_, &arena));
  rewriter.name_policy = 0;

  IREE_ASSERT_OK(loom_rewriter_copy_value_name(&rewriter, source, target));
  EXPECT_EQ(loom_module_value(module_, target)->name_id,
            LOOM_STRING_ID_INVALID);
  IREE_ASSERT_OK(loom_rewriter_try_set_derived_value_name(
      &rewriter, source, target, IREE_SV("bounded")));
  EXPECT_EQ(loom_module_value(module_, target)->name_id,
            LOOM_STRING_ID_INVALID);

  loom_rewriter_deinitialize(&rewriter);
  iree_arena_deinitialize(&arena);
}

}  // namespace
}  // namespace loom
