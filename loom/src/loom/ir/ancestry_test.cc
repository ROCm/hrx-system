// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/ancestry.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

class AncestryTest : public ::testing::Test {
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
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_op_t* BuildFunction(loom_type_t argument_type) {
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&builder, IREE_SV("f"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*module_id=*/0, /*symbol_id=*/symbol_id};
    loom_op_t* function_op = NULL;
    IREE_CHECK_OK(loom_test_func_build(
        &builder, 0, 0, 0, callee, &argument_type, 1, NULL, 0, NULL, 0, NULL, 0,
        LOOM_LOCATION_UNKNOWN, &function_op));
    return function_op;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
};

TEST_F(AncestryTest, FindsNestedOperationsBlocksAndValues) {
  loom_op_t* function_op = BuildFunction(loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  loom_func_like_t function = loom_func_like_cast(module_, function_op);
  loom_block_t* function_block =
      loom_region_entry_block(loom_func_like_body(function));
  loom_value_id_t condition = loom_block_arg_id(function_block, 0);

  loom_builder_t builder;
  loom_builder_initialize(module_, &module_->arena, function_block, &builder);
  builder.ip.parent_op = function_op;
  loom_op_t* region_op = NULL;
  IREE_ASSERT_OK(loom_test_optional_region_build(
      &builder, 0, condition, LOOM_LOCATION_UNKNOWN, &region_op));
  loom_region_t* region = loom_test_optional_region_body(region_op);
  loom_block_t* region_block = loom_region_entry_block(region);
  loom_value_id_t region_argument = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_I32), &region_argument));
  IREE_ASSERT_OK(loom_block_add_arg(module_, region_block, region_argument));

  loom_builder_ip_t previous_ip =
      loom_builder_enter_region(&builder, region_op, region);
  loom_op_t* constant_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(
      &builder, loom_attr_i64(1), loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      LOOM_LOCATION_UNKNOWN, &constant_op));
  loom_builder_restore(&builder, previous_ip);
  loom_value_id_t constant = loom_test_constant_result(constant_op);

  EXPECT_TRUE(loom_op_is_ancestor_of(region_op, region_op));
  EXPECT_TRUE(loom_op_is_ancestor_of(region_op, constant_op));
  EXPECT_TRUE(loom_op_is_ancestor_of(function_op, constant_op));
  EXPECT_FALSE(loom_op_is_ancestor_of(constant_op, region_op));
  EXPECT_TRUE(loom_op_contains_block(region_op, region_block));
  EXPECT_TRUE(loom_op_contains_block(function_op, region_block));
  EXPECT_FALSE(loom_op_contains_block(region_op, function_block));
  EXPECT_TRUE(loom_op_subtree_defines_value(module_, region_op, constant));
  EXPECT_TRUE(
      loom_op_subtree_defines_value(module_, region_op, region_argument));
  EXPECT_FALSE(loom_op_subtree_defines_value(module_, region_op, condition));

  loom_block_unlink_op(module_, constant_op);
  ASSERT_EQ(region_block->first_op, nullptr);
  EXPECT_TRUE(loom_op_contains_block(region_op, region_block));
  EXPECT_TRUE(loom_op_contains_block(function_op, region_block));
  EXPECT_TRUE(
      loom_op_subtree_defines_value(module_, region_op, region_argument));
}

}  // namespace
}  // namespace loom
