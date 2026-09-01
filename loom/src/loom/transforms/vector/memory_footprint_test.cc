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
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
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
    if (module_) loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&pass_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* Parse(iree_string_view_t source) {
    const loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/{loom_diagnostic_stderr_sink, nullptr},
        /*.max_errors=*/20,
    };
    IREE_EXPECT_OK(
        loom_text_parse(source, IREE_SV("memory_footprint_test.loom"),
                        &context_, &block_pool_, &options, &module_));
    EXPECT_NE(module_, nullptr);
    return module_;
  }

  iree_arena_block_pool_t block_pool_ = {};
  iree_arena_allocator_t pass_arena_ = {};
  loom_context_t context_ = {};
  loom_module_t* module_ = nullptr;
  loom_pass_value_fact_owner_t value_facts_ = {};
};

TEST_F(VectorMemoryFootprintPassTest, PreservesReadOnlyFunctionFacts) {
  loom_module_t* module =
      Parse(IREE_SV("test.func @main() {\n"
                    "  %value = test.constant 42 : i32\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);
  loom_op_t* function_op = loom_block_op(loom_module_block(module), 0);
  ASSERT_NE(function_op, nullptr);
  loom_func_like_t function = loom_func_like_cast(module, function_op);
  ASSERT_TRUE(loom_func_like_isa(function));

  loom_pass_value_fact_lifecycle_counts_t counts = {};
  value_facts_.lifecycle_counts = &counts;

  loom_pass_t pass = {};
  pass.info = loom_vector_memory_footprint_pass_info();
  pass.instance_arena = &pass_arena_;
  pass.arena = &pass_arena_;
  pass.value_facts = &value_facts_;
  ASSERT_NE(pass.info->statistic_layout, nullptr);
  void* statistic_storage = nullptr;
  IREE_ASSERT_OK(iree_arena_allocate(&pass_arena_,
                                     pass.info->statistic_layout->storage_size,
                                     &statistic_storage));
  pass.statistic_storage = statistic_storage;
  memset(pass.statistic_storage, 0, pass.info->statistic_layout->storage_size);

  IREE_ASSERT_OK(loom_vector_memory_footprint_run(&pass, module, function));

  loom_value_fact_table_t* reused_facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_facts_acquire(
      &pass, module, loom_pass_value_fact_scope_function(function),
      &reused_facts));
  ASSERT_NE(reused_facts, nullptr);
  EXPECT_EQ(counts.acquisition_count, 2u);
  EXPECT_EQ(counts.recomputation_count, 1u);
  EXPECT_EQ(counts.cache_hit_count, 1u);
  EXPECT_EQ(counts.invalidation_count, 0u);
  EXPECT_EQ(counts.scope_clear_count, 0u);

  value_facts_.lifecycle_counts = nullptr;
}

}  // namespace
}  // namespace loom
