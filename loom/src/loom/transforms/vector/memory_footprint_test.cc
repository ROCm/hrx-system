// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/memory_footprint.h"

#include <cstring>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"
#include "loom/pass/value_facts.h"

namespace loom {
namespace {

class VectorMemoryFootprintPassTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &pass_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts_);
  }

  void TearDown() override {
    loom_pass_value_fact_owner_deinitialize(&value_facts_);
    if (module_) {
      loom_module_free(module_);
    }
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&pass_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_func_like_t BuildFunction() {
    IREE_CHECK_OK(loom_module_allocate(&context_, IREE_SV("footprint"),
                                       &block_pool_, nullptr,
                                       iree_allocator_system(), &module_));
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&builder, IREE_SV("main"), &name));
    uint16_t symbol = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name, &symbol));
    loom_op_t* function_op = nullptr;
    IREE_CHECK_OK(loom_test_func_build(&builder, 0, 0, 0, {0, symbol}, nullptr,
                                       0, nullptr, 0, nullptr, 0, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &function_op));
    const auto function = loom_func_like_cast(module_, function_op);
    loom_builder_enter_region(&builder, function_op,
                              loom_func_like_body(function));
    loom_op_t* constant = nullptr;
    IREE_CHECK_OK(loom_test_constant_build(
        &builder, loom_attr_i64(42), loom_type_scalar(LOOM_SCALAR_TYPE_I32),
        LOOM_LOCATION_UNKNOWN, &constant));
    loom_op_t* terminator = nullptr;
    IREE_CHECK_OK(loom_test_yield_build(&builder, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &terminator));
    return function;
  }

  iree_arena_block_pool_t block_pool_ = {};
  iree_arena_allocator_t pass_arena_ = {};
  loom_context_t context_ = {};
  loom_module_t* module_ = nullptr;
  loom_pass_value_fact_owner_t value_facts_ = {};
};

TEST_F(VectorMemoryFootprintPassTest, PreservesReadOnlyFunctionFacts) {
  const auto function = BuildFunction();

  loom_pass_value_fact_lifecycle_counts_t counts = {};
  value_facts_.lifecycle_counts = &counts;

  loom_pass_t pass = {
      .info = loom_vector_memory_footprint_pass_info(),
      .instance_arena = &pass_arena_,
      .arena = &pass_arena_,
      .value_facts = &value_facts_,
  };
  ASSERT_NE(pass.info->statistic_layout, nullptr);
  void* statistic_storage = nullptr;
  IREE_ASSERT_OK(iree_arena_allocate(&pass_arena_,
                                     pass.info->statistic_layout->storage_size,
                                     &statistic_storage));
  pass.statistic_storage = statistic_storage;
  memset(pass.statistic_storage, 0, pass.info->statistic_layout->storage_size);

  IREE_ASSERT_OK(loom_vector_memory_footprint_run(&pass, module_, function));

  loom_value_fact_table_t* reused_facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_facts_acquire(
      &pass, module_, loom_pass_value_fact_scope_function(function),
      &reused_facts));
  ASSERT_NE(reused_facts, nullptr);
  EXPECT_EQ(counts.acquisition_count, 2u);
  EXPECT_EQ(counts.recomputation_count, 1u);
  EXPECT_EQ(counts.cache_hit_count, 1u);
  EXPECT_EQ(counts.invalidation_count, 0u);
  EXPECT_EQ(counts.scope_clear_count, 0u);
  EXPECT_EQ(value_facts_.active_scope.kind,
            LOOM_PASS_VALUE_FACT_SCOPE_FUNCTION);

  value_facts_.lifecycle_counts = nullptr;
}

}  // namespace
}  // namespace loom
