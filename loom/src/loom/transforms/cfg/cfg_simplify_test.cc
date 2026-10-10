// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cfg/cfg_simplify.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/util/fact_cfg.h"

namespace loom {
namespace {

class CFGSimplifyFactsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    Register(LOOM_DIALECT_CFG, loom_cfg_dialect_vtables);
    Register(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    Register(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_pass_value_fact_owner_initialize(&pool_, &owner_);
    owner_.lifecycle_counts = &counts_;
  }

  void TearDown() override {
    loom_pass_value_fact_owner_deinitialize(&owner_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Register(uint8_t dialect,
                const loom_op_vtable_t* const* (*get)(iree_host_size_t*)) {
    iree_host_size_t count = 0;
    const auto* vtables = get(&count);
    IREE_ASSERT_OK(
        loom_context_register_dialect(&context_, dialect, vtables, count));
  }

  void Parse(iree_string_view_t source) {
    IREE_ASSERT_OK(loom_text_parse(source, IREE_SV("cfg_facts.loom"), &context_,
                                   &pool_, nullptr, &module_));
    function_ =
        loom_func_like_cast(module_, loom_module_block(module_)->first_op);
  }

  iree_status_t Simplify(bool* out_changed) {
    iree_arena_allocator_t arena;
    iree_arena_initialize(&pool_, &arena);
    loom_pass_t pass = {
        .info = loom_cfg_simplify_pass_info(),
        .instance_arena = &arena,
        .arena = &arena,
        .value_facts = &owner_,
    };
    const auto* layout = pass.info->statistic_layout;
    iree_status_t status = iree_arena_allocate(&arena, layout->storage_size,
                                               &pass.statistic_storage);
    if (iree_status_is_ok(status)) {
      memset(pass.statistic_storage, 0, layout->storage_size);
      status = loom_cfg_simplify_run(&pass, module_, function_);
    }
    *out_changed = pass.changed;
    iree_arena_deinitialize(&arena);
    return status;
  }

  // Storage shared by the module, fact owner, and individual pass invocations.
  iree_arena_block_pool_t pool_ = {};
  // Registry for the three dialects used by these source functions.
  loom_context_t context_ = {};
  // Parsed source owned by the fixture.
  loom_module_t* module_ = nullptr;
  // Function whose facts cross the simplify pass boundary.
  loom_func_like_t function_ = {};
  // Fact storage with a lifetime independent of the simplify pass arena.
  loom_pass_value_fact_owner_t owner_ = {};
  // Observed inference and reuse events across pass invocations.
  loom_pass_value_fact_lifecycle_counts_t counts_ = {};
};

TEST_F(CFGSimplifyFactsTest, UnchangedFactsOutlivePassStorage) {
  Parse(IREE_SV(R"(
test.func @choose(%condition: i1) -> (i32) {
  cfg.cond_br %condition, ^left, ^right
^left:
  %left = scalar.constant 7 : i32
  test.yield %left : i32
^right:
  %right = scalar.constant 11 : i32
  test.yield %right : i32
}
)"));
  ASSERT_NE(module_, nullptr);
  bool changed = false;
  IREE_ASSERT_OK(Simplify(&changed));
  ASSERT_FALSE(changed);
  EXPECT_EQ(owner_.active_scope.kind, LOOM_PASS_VALUE_FACT_SCOPE_FUNCTION);
  auto* body = loom_func_like_body(function_);
  const auto* retained =
      loom_value_fact_table_lookup_cfg_region(&owner_.table, body);
  ASSERT_NE(retained, nullptr);
  const auto recomputations = counts_.recomputation_count;
  const auto hits = counts_.cache_hit_count;

  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner_, module_, loom_pass_value_fact_scope_function(function_),
      &facts));
  EXPECT_EQ(counts_.recomputation_count, recomputations);
  EXPECT_EQ(counts_.cache_hit_count, hits + 1);
  EXPECT_EQ(loom_value_fact_table_lookup_cfg_region(facts, body), retained);
  ASSERT_EQ(retained->graph.block_count, 3);
  for (uint16_t i = 1; i < 3; ++i) {
    const auto* block = retained->graph.blocks[i].block;
    const auto value = loom_op_operands(block->last_op)[0];
    int64_t constant = 0;
    ASSERT_TRUE(loom_value_facts_as_exact_i64(
        loom_value_fact_table_lookup(facts, value), &constant));
    EXPECT_EQ(constant, i == 1 ? 7 : 11);
  }
}

TEST_F(CFGSimplifyFactsTest, ChangedControlInvalidatesFacts) {
  Parse(IREE_SV(R"(
test.func @choose() -> (i32) {
  %condition = scalar.constant true : i1
  cfg.cond_br %condition, ^left, ^right
^left:
  %left = scalar.constant 7 : i32
  test.yield %left : i32
^right:
  %right = scalar.constant 11 : i32
  test.yield %right : i32
}
)"));
  ASSERT_NE(module_, nullptr);
  bool changed = false;
  IREE_ASSERT_OK(Simplify(&changed));
  ASSERT_TRUE(changed);
  EXPECT_EQ(owner_.active_scope.kind, LOOM_PASS_VALUE_FACT_SCOPE_NONE);
  const auto recomputations = counts_.recomputation_count;
  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner_, module_, loom_pass_value_fact_scope_function(function_),
      &facts));
  EXPECT_EQ(counts_.recomputation_count, recomputations + 1);
  auto* body = loom_func_like_body(function_);
  ASSERT_EQ(body->block_count, 1);
  const auto value =
      loom_op_operands(loom_region_entry_block(body)->last_op)[0];
  int64_t constant = 0;
  ASSERT_TRUE(loom_value_facts_as_exact_i64(
      loom_value_fact_table_lookup(facts, value), &constant));
  EXPECT_EQ(constant, 7);
}

}  // namespace
}  // namespace loom
