// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/fact_propagation.h"
#include "loom/rewrite/rewriter.h"

namespace loom {
namespace {

class FactSummaryTest : public ::testing::Test {
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
    vtables = loom_scf_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SCF, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));

    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&module_builder,
                                              IREE_SV("test_fn"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t callee = {.module_id = 0, .symbol_id = symbol_id};
    loom_op_t* func_op = nullptr;
    IREE_ASSERT_OK(loom_test_func_build(
        &module_builder, 0, 0, 0, callee, nullptr, 0, nullptr, 0, nullptr, 0,
        nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    function_ = loom_func_like_cast(module_, func_op);
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_func_like_body(function_)), &builder_);
    loom_builder_enter_region(&builder_, function_.op,
                              loom_func_like_body(function_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Shared backing for module and analysis arenas.
  iree_arena_block_pool_t block_pool_;
  // Registered operation semantics used by the fixture.
  loom_context_t context_;
  // Source module owned by the fixture.
  loom_module_t* module_ = nullptr;
  // Function whose body is transformed.
  loom_func_like_t function_;
  // Builder positioned in the function body.
  loom_builder_t builder_;
};

class NestedSummaryTest : public FactSummaryTest,
                          public ::testing::WithParamInterface<bool> {};

// Lexical containment alone creates no fact dependency. When the inner result
// is yielded by the outer branch, semantic closure must queue both owners
// before solving either, allowing one outer solve to cover the inner summary.
TEST_P(NestedSummaryTest, ResultDependencies) {
  const bool uses_inner_result = GetParam();
  const auto i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* constants[3] = {};
  const int64_t values[] = {1, 2, 9};
  for (int i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(values[i]),
                                            i32, LOOM_LOCATION_UNKNOWN,
                                            &constants[i]));
  }
  const auto one = loom_test_constant_result(constants[0]);
  const auto two = loom_test_constant_result(constants[1]);
  const auto nine = loom_test_constant_result(constants[2]);
  loom_op_t* outer = nullptr;
  IREE_ASSERT_OK(loom_test_branch_build(&builder_, one, &i32, 1, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &outer));
  loom_op_t* inner = nullptr;
  loom_op_t* producer = nullptr;
  for (int branch_index = 0; branch_index < 2; ++branch_index) {
    auto* region = loom_op_regions(outer)[branch_index];
    auto nested = builder_;
    loom_builder_enter_region(&nested, outer, region);
    loom_builder_set_block(&nested, loom_region_entry_block(region));
    if (branch_index == 0) {
      IREE_ASSERT_OK(loom_test_branch_build(&nested, one, &i32, 1, nullptr, 0,
                                            LOOM_LOCATION_UNKNOWN, &inner));
      for (int inner_index = 0; inner_index < 2; ++inner_index) {
        auto* inner_region = loom_op_regions(inner)[inner_index];
        auto inner_builder = nested;
        loom_builder_enter_region(&inner_builder, inner, inner_region);
        loom_builder_set_block(&inner_builder,
                               loom_region_entry_block(inner_region));
        loom_op_t* add = nullptr;
        IREE_ASSERT_OK(loom_test_addi_build(&inner_builder, two, two, i32,
                                            LOOM_LOCATION_UNKNOWN, &add));
        if (inner_index == 0) {
          producer = add;
        }
        const auto value = loom_test_addi_result(add);
        loom_op_t* yield = nullptr;
        IREE_ASSERT_OK(loom_test_yield_build(&inner_builder, &value, 1,
                                             LOOM_LOCATION_UNKNOWN, &yield));
      }
      loom_op_t* use = nullptr;
      IREE_ASSERT_OK(loom_test_use_build(&nested, loom_op_results(inner), 1,
                                         LOOM_LOCATION_UNKNOWN, &use));
    }
    const auto outer_value = branch_index == 0 && uses_inner_result
                                 ? loom_op_results(inner)[0]
                                 : two;
    loom_op_t* yield = nullptr;
    IREE_ASSERT_OK(loom_test_yield_build(&nested, &outer_value, 1,
                                         LOOM_LOCATION_UNKNOWN, &yield));
  }
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &return_op));
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_pass_value_fact_owner_t owner;
  loom_pass_value_fact_owner_initialize(&block_pool_, &owner);
  const auto scope = loom_pass_value_fact_scope_function(function_);
  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(
      loom_pass_value_fact_owner_acquire(&owner, module_, scope, &facts));
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module_, &arena);
  IREE_ASSERT_OK(loom_rewriter_enable_worklist(&rewriter));
  loom_rewriter_attach_value_facts(&rewriter, facts);
  struct Observer {
    // Production listener preserving pending-summary coalescing.
    decltype(facts->summary_computed) previous;
    // Enclosing owner with an optional dependency on the inner result.
    const loom_op_t* outer;
    // Direct owner of the mutated result equation.
    const loom_op_t* inner;
    // Completed enclosing summaries during this publication.
    size_t outer_count = 0;
    // Completed direct summaries during this publication.
    size_t inner_count = 0;
  } observer{facts->summary_computed, outer, inner};
  facts->summary_computed = {
      .user_data = &observer,
      .fn =
          [](void* user_data, const loom_op_t* op) {
            auto* observer = static_cast<Observer*>(user_data);
            observer->outer_count += op == observer->outer;
            observer->inner_count += op == observer->inner;
            observer->previous.fn(observer->previous.user_data, op);
          },
  };
  for (auto replacement : {nine, two, nine, two}) {
    observer.outer_count = 0;
    observer.inner_count = 0;
    IREE_ASSERT_OK(
        loom_rewriter_set_operand(&rewriter, producer, 0, replacement));
    IREE_ASSERT_OK(loom_rewriter_propagate_value_facts(&rewriter));
    EXPECT_EQ(observer.outer_count, uses_inner_result ? 1u : 0u);
    EXPECT_EQ(observer.inner_count, 1u);
    EXPECT_EQ(
        loom_value_fact_table_lookup(facts, loom_op_results(outer)[0]).range_lo,
        2);
    EXPECT_EQ(
        loom_value_fact_table_lookup(facts, loom_op_results(inner)[0]).range_lo,
        4);
    bool saw_outer_rewrite = false;
    while (auto* op = loom_rewriter_pop(&rewriter)) {
      saw_outer_rewrite |= op == outer;
    }
    EXPECT_TRUE(saw_outer_rewrite);
    loom_pass_value_fact_owner_t fresh_owner;
    loom_pass_value_fact_owner_initialize(&block_pool_, &fresh_owner);
    loom_value_fact_table_t* fresh = nullptr;
    IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(&fresh_owner, module_,
                                                      scope, &fresh));
    for (auto* op : {producer, inner, outer}) {
      const auto value = loom_op_results(op)[0];
      EXPECT_TRUE(loom_value_fact_table_facts_equal_for_type(
          module_, i32, facts, loom_value_fact_table_lookup(facts, value),
          fresh, loom_value_fact_table_lookup(fresh, value)));
    }
    loom_pass_value_fact_owner_deinitialize(&fresh_owner);
  }
  loom_rewriter_deinitialize(&rewriter);
  loom_pass_value_fact_owner_deinitialize(&owner);
  iree_arena_deinitialize(&arena);
}

INSTANTIATE_TEST_SUITE_P(Dependencies, NestedSummaryTest, ::testing::Bool());

class NestedFeedbackTest : public FactSummaryTest,
                           public ::testing::WithParamInterface<bool> {};

// Semantic edits can invalidate feedback even when derived rows are already
// broad. Publishing through a nested summary must both weaken and recover the
// enclosing loop's facts, without relying on the optional rewrite worklist.
TEST_P(NestedFeedbackTest, RestoresExactState) {
  const bool is_conditional_loop = GetParam();
  const auto i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const auto index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  auto constant = [&](int64_t value, loom_type_t type) {
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_test_constant_build(&builder_, loom_attr_i64(value),
                                            type, LOOM_LOCATION_UNKNOWN, &op));
    return op;
  };
  const auto begin = loom_test_constant_result(constant(0, index));
  const auto limit = loom_test_constant_result(constant(4, index));
  const auto step = loom_test_constant_result(constant(1, index));
  const auto initial = loom_test_constant_result(constant(2, i32));
  const auto selector = loom_test_constant_result(constant(1, i32));
  auto* increment = constant(0, i32);
  loom_op_t* loop = nullptr;
  if (is_conditional_loop) {
    const auto enabled = loom_test_constant_result(
        constant(1, loom_type_scalar(LOOM_SCALAR_TYPE_I1)));
    loom_op_t* unknown = nullptr;
    IREE_ASSERT_OK(loom_test_attrs_build(&builder_, 0, enabled, {},
                                         loom_type_scalar(LOOM_SCALAR_TYPE_I1),
                                         LOOM_LOCATION_UNKNOWN, &unknown));
    IREE_ASSERT_OK(loom_scf_while_build(&builder_, &initial, 1, nullptr,
                                        nullptr, 1, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &loop));
    auto* before =
        loom_loop_like_condition_region(loom_loop_like_cast(module_, loop));
    auto* before_block = loom_region_entry_block(before);
    auto before_builder = builder_;
    loom_builder_enter_region(&before_builder, loop, before);
    loom_builder_set_block(&before_builder, before_block);
    const auto before_value = loom_block_arg_id(before_block, 0);
    loom_op_t* condition = nullptr;
    IREE_ASSERT_OK(loom_scf_condition_build(
        &before_builder, loom_test_attrs_result(unknown), &before_value, 1,
        LOOM_LOCATION_UNKNOWN, &condition));
  } else {
    IREE_ASSERT_OK(loom_scf_for_build(
        &builder_, 0, begin, limit, step, &initial, 1, nullptr, nullptr, 0,
        LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, 0, 0,
        LOOM_LOCATION_UNKNOWN, &loop));
  }
  auto* body = loom_loop_like_body(loom_loop_like_cast(module_, loop));
  auto* block = loom_region_entry_block(body);
  const auto carried = loom_block_arg_id(block, is_conditional_loop ? 0 : 1);
  auto body_builder = builder_;
  loom_builder_enter_region(&body_builder, loop, body);
  loom_builder_set_block(&body_builder, block);
  loom_op_t* branch = nullptr;
  IREE_ASSERT_OK(loom_test_branch_build(&body_builder, selector, &i32, 1,
                                        nullptr, 0, LOOM_LOCATION_UNKNOWN,
                                        &branch));
  loom_op_t* additions[2] = {};
  for (int i = 0; i < 2; ++i) {
    auto* region = loom_op_regions(branch)[i];
    auto inner_builder = body_builder;
    loom_builder_enter_region(&inner_builder, branch, region);
    loom_builder_set_block(&inner_builder, loom_region_entry_block(region));
    IREE_ASSERT_OK(loom_test_addi_build(
        &inner_builder, carried, loom_test_constant_result(increment), i32,
        LOOM_LOCATION_UNKNOWN, &additions[i]));
    const auto value = loom_test_addi_result(additions[i]);
    loom_op_t* yield = nullptr;
    IREE_ASSERT_OK(loom_test_yield_build(&inner_builder, &value, 1,
                                         LOOM_LOCATION_UNKNOWN, &yield));
  }
  loom_op_t* yield = nullptr;
  IREE_ASSERT_OK(loom_scf_yield_build(&body_builder, loom_op_results(branch), 1,
                                      LOOM_LOCATION_UNKNOWN, &yield));
  loom_op_t* end = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &end));
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_pass_value_fact_owner_t owner;
  loom_pass_value_fact_owner_initialize(&block_pool_, &owner);
  const auto scope = loom_pass_value_fact_scope_function(function_);
  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(
      loom_pass_value_fact_owner_acquire(&owner, module_, scope, &facts));
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module_, &arena);
  loom_rewriter_attach_value_facts(&rewriter, facts);
  for (int64_t delta : {1, 0, 1, 0}) {
    IREE_ASSERT_OK(loom_test_constant_rewrite_value(&rewriter, increment,
                                                    loom_attr_i64(delta)));
    IREE_ASSERT_OK(loom_rewriter_propagate_value_facts(&rewriter));
    loom_pass_value_fact_owner_t fresh_owner;
    loom_pass_value_fact_owner_initialize(&block_pool_, &fresh_owner);
    loom_value_fact_table_t* fresh = nullptr;
    IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(&fresh_owner, module_,
                                                      scope, &fresh));
    for (auto value : {carried, loom_test_addi_result(additions[0]),
                       loom_test_addi_result(additions[1]),
                       loom_op_results(branch)[0], loom_op_results(loop)[0]}) {
      EXPECT_TRUE(loom_value_fact_table_facts_equal_for_type(
          module_, i32, facts, loom_value_fact_table_lookup(facts, value),
          fresh, loom_value_fact_table_lookup(fresh, value)))
          << delta << ":" << value;
      if (delta == 0) {
        EXPECT_EQ(loom_value_fact_table_lookup(facts, value).range_lo, 2);
        EXPECT_EQ(loom_value_fact_table_lookup(facts, value).range_hi, 2);
      }
    }
    loom_pass_value_fact_owner_deinitialize(&fresh_owner);
  }
  loom_rewriter_deinitialize(&rewriter);
  loom_pass_value_fact_owner_deinitialize(&owner);
  iree_arena_deinitialize(&arena);
}

INSTANTIATE_TEST_SUITE_P(CountedAndConditional, NestedFeedbackTest,
                         ::testing::Bool());
}  // namespace
}  // namespace loom
