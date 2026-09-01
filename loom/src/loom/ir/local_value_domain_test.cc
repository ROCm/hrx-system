// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/local_value_domain.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

class LocalValueDomainTest : public ::testing::Test {
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
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(LocalValueDomainTest, RegionTreeUsesExactStructuralCapacity) {
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_type_t arg_types[] = {i32_type, i32_type};
  loom_op_t* function_op = NULL;
  IREE_ASSERT_OK(loom_test_func_build(
      &builder_, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0,
      /*callee=*/{0, 1}, arg_types, IREE_ARRAYSIZE(arg_types),
      /*result_types=*/NULL, /*result_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, /*predicates=*/NULL, /*predicates_count=*/0,
      LOOM_LOCATION_UNKNOWN, &function_op));
  loom_region_t* function_body = loom_test_func_body(function_op);
  loom_block_t* function_block = loom_region_entry_block(function_body);
  const loom_builder_ip_t saved_function_ip =
      loom_builder_enter_region(&builder_, function_op, function_body);

  loom_op_t* constant_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &constant_op));
  loom_op_t* add_op = NULL;
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, loom_block_arg_id(function_block, 0),
                           loom_test_constant_result(constant_op), i32_type,
                           LOOM_LOCATION_UNKNOWN, &add_op));

  loom_op_t* isolated_op = NULL;
  IREE_ASSERT_OK(loom_test_isolated_region_build(
      &builder_, &i32_type, /*result_count=*/1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN, &isolated_op));
  loom_region_t* isolated_body = loom_test_isolated_region_body(isolated_op);
  const loom_builder_ip_t saved_isolated_ip =
      loom_builder_enter_region(&builder_, isolated_op, isolated_body);
  loom_op_t* nested_constant_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32_type,
                                          LOOM_LOCATION_UNKNOWN,
                                          &nested_constant_op));
  loom_builder_restore(&builder_, saved_isolated_ip);
  loom_builder_restore(&builder_, saved_function_ip);

  ASSERT_EQ(function_body->value_definition_count, 6u);
  ASSERT_EQ(isolated_body->value_definition_count, 1u);

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool_, &scratch_arena);
  loom_local_value_domain_t domain = {};
  IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region_tree(
      module_, function_body, &scratch_arena, &domain));
  EXPECT_EQ(domain.value_count, function_body->value_definition_count);
  EXPECT_EQ(domain.value_capacity, domain.value_count);
  loom_local_value_domain_release(&domain);
  iree_arena_deinitialize(&scratch_arena);
}

}  // namespace
}  // namespace loom
