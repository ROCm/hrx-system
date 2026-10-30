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
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/fact_propagation.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_origin.h"
#include "loom/util/fact_vector_partition.h"

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
    vtables = loom_index_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_INDEX, vtables, (uint16_t)vtable_count));
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

  // The no-op refinement changes no operands, but forces the ordinary full
  // body evaluator in every feedback round through its public context API.
  void ComputeFullTraversalFacts(loom_pass_value_fact_owner_t* owner,
                                 loom_value_fact_table_t** out_facts) {
    loom_pass_value_fact_owner_initialize(&block_pool_, owner);
    IREE_ASSERT_OK(loom_pass_value_fact_owner_prepare(
        owner, module_, loom_pass_value_fact_scope_function(function_),
        out_facts));
    (*out_facts)->context.refine_operands.fn =
        [](void*, const loom_value_fact_table_t*, const loom_op_t*,
           loom_value_facts_t*) {};
    IREE_ASSERT_OK(
        loom_value_fact_table_compute(*out_facts, module_, function_));
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
// is yielded by the outer branch, semantic closure queues both result joins.
// A parent joined before its child must receive the child's final publication,
// while neither join reevaluates an unaffected sibling body.
TEST_P(NestedSummaryTest, ResultDependencies) {
  const bool uses_inner_result = GetParam();
  const auto i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* constants[4] = {};
  const int64_t values[] = {1, 2, 9, 1};
  for (int i = 0; i < 4; ++i) {
    IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(values[i]),
                                            i32, LOOM_LOCATION_UNKNOWN,
                                            &constants[i]));
  }
  const auto one = loom_test_constant_result(constants[0]);
  const auto two = loom_test_constant_result(constants[1]);
  const auto nine = loom_test_constant_result(constants[2]);
  loom_op_t* outer = nullptr;
  IREE_ASSERT_OK(loom_test_branch_build(
      &builder_, loom_test_constant_result(constants[3]), &i32, 1, nullptr, 0,
      LOOM_LOCATION_UNKNOWN, &outer));
  loom_op_t* inner = nullptr;
  loom_op_t* sibling = nullptr;
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
    } else {
      IREE_ASSERT_OK(loom_test_branch_build(&nested, one, &i32, 1, nullptr, 0,
                                            LOOM_LOCATION_UNKNOWN, &sibling));
      for (int sibling_index = 0; sibling_index < 2; ++sibling_index) {
        auto* sibling_region = loom_op_regions(sibling)[sibling_index];
        auto sibling_builder = nested;
        loom_builder_enter_region(&sibling_builder, sibling, sibling_region);
        loom_builder_set_block(&sibling_builder,
                               loom_region_entry_block(sibling_region));
        loom_op_t* yield = nullptr;
        IREE_ASSERT_OK(loom_test_yield_build(&sibling_builder, &two, 1,
                                             LOOM_LOCATION_UNKNOWN, &yield));
      }
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
    // Independent body whose facts stay current across result-only changes.
    const loom_op_t* sibling;
    // Completed enclosing summaries during this publication.
    size_t outer_count = 0;
    // Completed direct summaries during this publication.
    size_t inner_count = 0;
    // Completed independent summaries during this publication.
    size_t sibling_count = 0;
  } observer{facts->summary_computed, outer, inner, sibling};
  facts->summary_computed = {
      .user_data = &observer,
      .fn =
          [](void* user_data, const loom_op_t* op) {
            auto* observer = static_cast<Observer*>(user_data);
            observer->outer_count += op == observer->outer;
            observer->inner_count += op == observer->inner;
            observer->sibling_count += op == observer->sibling;
            observer->previous.fn(observer->previous.user_data, op);
          },
  };
  for (auto replacement : {nine, two, nine, two}) {
    observer.outer_count = 0;
    observer.inner_count = 0;
    observer.sibling_count = 0;
    IREE_ASSERT_OK(
        loom_rewriter_set_operand(&rewriter, producer, 0, replacement));
    IREE_ASSERT_OK(loom_rewriter_propagate_value_facts(&rewriter));
    EXPECT_EQ(observer.outer_count != 0, uses_inner_result);
    EXPECT_LE(observer.outer_count, 2u);
    EXPECT_EQ(observer.inner_count, 1u);
    EXPECT_EQ(observer.sibling_count, 0u);
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
    for (auto* op : {producer, inner, outer, sibling}) {
      const auto value = loom_op_results(op)[0];
      EXPECT_TRUE(loom_value_fact_table_facts_equal_for_type(
          module_, i32, facts, loom_value_fact_table_lookup(facts, value),
          fresh, loom_value_fact_table_lookup(fresh, value)));
    }
    loom_pass_value_fact_owner_deinitialize(&fresh_owner);
  }
  // A direct selector edit still refreshes nested context. Coalesce it with
  // the yielded-value edit in either order, then compare every live fact.
  for (bool selector_first : {false, true}) {
    auto change_selector = [&]() {
      return loom_test_constant_rewrite_value(
          &rewriter, constants[3], loom_attr_i64(selector_first ? 1 : 0));
    };
    if (selector_first) {
      IREE_ASSERT_OK(change_selector());
    }
    IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter, producer, 0,
                                             selector_first ? two : nine));
    if (!selector_first) {
      IREE_ASSERT_OK(change_selector());
    }
    observer.sibling_count = 0;
    IREE_ASSERT_OK(loom_rewriter_propagate_value_facts(&rewriter));
    EXPECT_EQ(observer.sibling_count, 1u);
    loom_pass_value_fact_owner_t fresh_owner;
    loom_pass_value_fact_owner_initialize(&block_pool_, &fresh_owner);
    loom_value_fact_table_t* fresh = nullptr;
    IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(&fresh_owner, module_,
                                                      scope, &fresh));
    for (iree_host_size_t i = 0; i < fresh->touched_count; ++i) {
      const auto value = fresh->touched_values[i];
      EXPECT_TRUE(loom_value_fact_table_facts_equal_for_type(
          module_, loom_module_value_type(module_, value), facts,
          loom_value_fact_table_lookup(facts, value), fresh,
          loom_value_fact_table_lookup(fresh, value)));
    }
    loom_pass_value_fact_owner_deinitialize(&fresh_owner);
  }
  loom_rewriter_deinitialize(&rewriter);
  loom_pass_value_fact_owner_deinitialize(&owner);
  iree_arena_deinitialize(&arena);
}

INSTANTIATE_TEST_SUITE_P(Dependencies, NestedSummaryTest, ::testing::Bool());

class NewBranchContextTest : public FactSummaryTest,
                             public ::testing::WithParamInterface<bool> {};

TEST_P(NewBranchContextTest, PublishesContextBeforeBodyConstruction) {
  const bool keyed = GetParam();
  const auto type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* constant = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), type,
                                          LOOM_LOCATION_UNKNOWN, &constant));
  const auto one = loom_test_constant_result(constant);
  loom_op_t* function_yield = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &function_yield));
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_pass_value_fact_owner_t owner;
  loom_pass_value_fact_owner_initialize(&block_pool_, &owner);
  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner, module_, loom_pass_value_fact_scope_function(function_), &facts));
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module_, &arena);
  loom_rewriter_attach_value_facts(&rewriter, facts);
  loom_builder_set_before(&rewriter.builder, function_yield);
  loom_op_t* branch = nullptr;
  if (keyed) {
    const int64_t key = 0;
    IREE_ASSERT_OK(loom_scf_switch_build(&rewriter.builder, one, &type, 1,
                                         nullptr, 0, &key, 1,
                                         LOOM_LOCATION_UNKNOWN, &branch));
  } else {
    IREE_ASSERT_OK(loom_test_branch_build(&rewriter.builder, one, &type, 1,
                                          nullptr, 0, LOOM_LOCATION_UNKNOWN,
                                          &branch));
  }
  // Entry context belongs to the region producer, before any child inference.
  EXPECT_EQ(
      loom_value_fact_table_lookup_region_branch_truth(
          facts, loom_op_regions(branch)[0]),
      keyed ? LOOM_REGION_BRANCH_TRUTH_UNKNOWN : LOOM_REGION_BRANCH_TRUTH_TRUE);
  EXPECT_EQ(loom_value_fact_table_lookup_region_branch_truth(
                facts, loom_op_regions(branch)[1]),
            keyed ? LOOM_REGION_BRANCH_TRUTH_UNKNOWN
                  : LOOM_REGION_BRANCH_TRUTH_FALSE);
  for (int region_index = 0; region_index < 2; ++region_index) {
    auto* region = loom_op_regions(branch)[region_index];
    auto nested = rewriter.builder;
    loom_builder_enter_region(&nested, branch, region);
    loom_builder_set_block(&nested, loom_region_entry_block(region));
    loom_op_t* yield = nullptr;
    if (keyed) {
      IREE_ASSERT_OK(loom_scf_yield_build(&nested, &one, 1,
                                          LOOM_LOCATION_UNKNOWN, &yield));
    } else {
      IREE_ASSERT_OK(loom_test_yield_build(&nested, &one, 1,
                                           LOOM_LOCATION_UNKNOWN, &yield));
    }
  }
  IREE_ASSERT_OK(loom_rewriter_propagate_value_facts(&rewriter));
  EXPECT_EQ(
      loom_value_fact_table_lookup_region_branch_truth(
          facts, loom_op_regions(branch)[0]),
      keyed ? LOOM_REGION_BRANCH_TRUTH_UNKNOWN : LOOM_REGION_BRANCH_TRUTH_TRUE);
  EXPECT_EQ(loom_value_fact_table_lookup_region_branch_truth(
                facts, loom_op_regions(branch)[1]),
            keyed ? LOOM_REGION_BRANCH_TRUTH_UNKNOWN
                  : LOOM_REGION_BRANCH_TRUTH_FALSE);
  EXPECT_TRUE(loom_value_facts_equal(
      loom_value_fact_table_lookup(facts, loom_op_results(branch)[0]),
      loom_value_fact_table_lookup(facts, one)));
  loom_rewriter_deinitialize(&rewriter);
  loom_pass_value_fact_owner_deinitialize(&owner);
  iree_arena_deinitialize(&arena);
}

INSTANTIATE_TEST_SUITE_P(BooleanAndKeyed, NewBranchContextTest,
                         ::testing::Bool());

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

// An independent nested branch may be retained across carried-state feedback.
// Rewriting its selector must still publish a fresh summary before the next
// loop solve; compare the retained owner with an independently built owner.
TEST_F(FactSummaryTest, CountedFeedbackWithIndependentNestedBranch) {
  const loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  auto index_constant = [&](int64_t value) {
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value), index, LOOM_LOCATION_UNKNOWN, &op));
    return loom_index_constant_result(op);
  };
  const loom_value_id_t begin = index_constant(0);
  const loom_value_id_t limit = index_constant(16);
  const loom_value_id_t step = index_constant(1);
  const loom_value_id_t initial[] = {index_constant(0), index_constant(5)};
  const loom_value_id_t increment = index_constant(1);
  loom_op_t* selector_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i1,
                                          LOOM_LOCATION_UNKNOWN,
                                          &selector_op));
  loom_op_t* loop = nullptr;
  IREE_ASSERT_OK(loom_scf_for_build(
      &builder_, 0, begin, limit, step, initial, 2, nullptr, nullptr, 0,
      LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, 0, 0,
      LOOM_LOCATION_UNKNOWN, &loop));
  loom_region_t* body = loom_loop_like_body(loom_loop_like_cast(module_, loop));
  loom_block_t* block = loom_region_entry_block(body);
  auto body_builder = builder_;
  loom_builder_enter_region(&body_builder, loop, body);
  loom_builder_set_block(&body_builder, block);
  loom_op_t* branch = nullptr;
  IREE_ASSERT_OK(loom_test_branch_build(
      &body_builder, loom_test_constant_result(selector_op), &index, 1,
      nullptr, 0, LOOM_LOCATION_UNKNOWN, &branch));
  loom_op_t* first_arm_yield = nullptr;
  for (int arm_index = 0; arm_index < 2; ++arm_index) {
    loom_region_t* region = loom_op_regions(branch)[arm_index];
    auto arm_builder = body_builder;
    loom_builder_enter_region(&arm_builder, branch, region);
    loom_builder_set_block(&arm_builder, loom_region_entry_block(region));
    loom_value_id_t arm_value = arm_index == 0 ? increment : initial[0];
    loom_op_t* arm_yield = nullptr;
    IREE_ASSERT_OK(loom_test_yield_build(&arm_builder, &arm_value, 1,
                                         LOOM_LOCATION_UNKNOWN, &arm_yield));
    if (arm_index == 0) first_arm_yield = arm_yield;
  }
  loom_value_id_t next[2];
  loom_op_t* additions[2] = {};
  for (uint16_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(loom_index_add_build(
        &body_builder, loom_block_arg_id(block, i + 1),
        loom_op_results(branch)[0], index, LOOM_LOCATION_UNKNOWN,
        &additions[i]));
    next[i] = loom_index_add_result(additions[i]);
  }
  loom_op_t* body_yield = nullptr;
  IREE_ASSERT_OK(loom_scf_yield_build(&body_builder, next, 2,
                                      LOOM_LOCATION_UNKNOWN, &body_yield));
  loom_op_t* function_yield = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &function_yield));

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
  for (int64_t selector : {1, 0, 1}) {
    IREE_ASSERT_OK(loom_test_constant_rewrite_value(
        &rewriter, selector_op, loom_attr_i64(selector)));
    IREE_ASSERT_OK(loom_rewriter_propagate_value_facts(&rewriter));
    const loom_value_facts_t branch_facts = loom_value_fact_table_lookup(
        facts, loom_op_results(branch)[0]);
    EXPECT_EQ(branch_facts.range_lo, 0);
    EXPECT_EQ(branch_facts.range_hi, 1);
    loom_pass_value_fact_owner_t fresh_owner;
    loom_value_fact_table_t* fresh = nullptr;
    ComputeFullTraversalFacts(&fresh_owner, &fresh);
    for (loom_value_id_t value : {loom_op_results(branch)[0], next[0],
                                  next[1], loom_op_results(loop)[0],
                                  loom_op_results(loop)[1]}) {
      EXPECT_TRUE(loom_value_fact_table_facts_equal_for_type(
          module_, index, facts, loom_value_fact_table_lookup(facts, value),
          fresh, loom_value_fact_table_lookup(fresh, value)))
          << selector << ":" << value;
      EXPECT_EQ(loom_value_fact_table_query_identity(facts, value),
                loom_value_fact_table_query_identity(fresh, value));
    }
    loom_pass_value_fact_owner_deinitialize(&fresh_owner);
  }

  auto expect_fresh = [&]() {
    loom_pass_value_fact_owner_t fresh_owner;
    loom_value_fact_table_t* fresh = nullptr;
    ComputeFullTraversalFacts(&fresh_owner, &fresh);
    for (loom_value_id_t value : {loom_op_results(branch)[0], next[0],
                                  next[1], loom_op_results(loop)[0],
                                  loom_op_results(loop)[1]}) {
      EXPECT_TRUE(loom_value_fact_table_facts_equal_for_type(
          module_, index, facts, loom_value_fact_table_lookup(facts, value),
          fresh, loom_value_fact_table_lookup(fresh, value)))
          << value;
      EXPECT_EQ(loom_value_fact_table_query_identity(facts, value),
                loom_value_fact_table_query_identity(fresh, value));
    }
    loom_pass_value_fact_owner_deinitialize(&fresh_owner);
  };

  // Moving a region owner across execution scopes and editing its nested
  // yield changes its facts before the outer loop's result-only update.
  IREE_ASSERT_OK(loom_rewriter_move_before(&rewriter, branch, loop));
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter, first_arm_yield, 0,
                                           initial[1]));
  IREE_ASSERT_OK(loom_rewriter_move_before(&rewriter, branch, additions[0]));
  IREE_ASSERT_OK(
      loom_rewriter_set_operand(&rewriter, body_yield, 0, initial[0]));
  IREE_ASSERT_OK(loom_rewriter_propagate_value_facts(&rewriter));
  EXPECT_EQ(loom_value_fact_table_lookup(facts, loom_op_results(branch)[0])
                .range_hi,
            5);
  expect_fresh();

  // Inserting and erasing ordinary facts does not invalidate the established
  // region scope. Repeated yield edits still converge to a fresh full solve.
  loom_builder_enter_region(&rewriter.builder, loop, body);
  for (loom_value_id_t first_yield : {next[0], initial[0], next[0]}) {
    loom_builder_set_before(&rewriter.builder, body_yield);
    loom_op_t* temporary = nullptr;
    IREE_ASSERT_OK(loom_index_add_build(
        &rewriter.builder, initial[0], increment, index,
        LOOM_LOCATION_UNKNOWN, &temporary));
    IREE_ASSERT_OK(loom_rewriter_erase(&rewriter, temporary));
    IREE_ASSERT_OK(
        loom_rewriter_set_operand(&rewriter, body_yield, 0, first_yield));
    IREE_ASSERT_OK(loom_rewriter_propagate_value_facts(&rewriter));
    expect_fresh();
  }
  loom_rewriter_deinitialize(&rewriter);
  loom_pass_value_fact_owner_deinitialize(&owner);
  iree_arena_deinitialize(&arena);
}

// A result-only outer yield can be queued after two sibling branches. One
// branch feeds the other's selector, so publishing them in either LIFO order
// must converge before the retained loop body is refreshed.
TEST_F(FactSummaryTest, CountedFeedbackWithPendingSiblingBranches) {
  const loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  auto index_constant = [&](int64_t value) {
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value), index, LOOM_LOCATION_UNKNOWN, &op));
    return loom_index_constant_result(op);
  };
  auto bool_constant = [&](int64_t value) {
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_test_constant_build(
        &builder_, loom_attr_i64(value), i1, LOOM_LOCATION_UNKNOWN, &op));
    return loom_test_constant_result(op);
  };
  const loom_value_id_t begin = index_constant(0);
  const loom_value_id_t limit = index_constant(8);
  const loom_value_id_t step = index_constant(1);
  const loom_value_id_t zero = index_constant(0);
  const loom_value_id_t one = index_constant(1);
  const loom_value_id_t two = index_constant(2);
  const loom_value_id_t truth = bool_constant(1);
  const loom_value_id_t falsehood = bool_constant(0);
  const loom_value_id_t same_falsehood = bool_constant(0);
  loom_op_t* loop = nullptr;
  IREE_ASSERT_OK(loom_scf_for_build(
      &builder_, 0, begin, limit, step, &zero, 1, nullptr, nullptr, 0,
      LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, 0, 0,
      LOOM_LOCATION_UNKNOWN, &loop));
  loom_region_t* body = loom_loop_like_body(loom_loop_like_cast(module_, loop));
  loom_block_t* block = loom_region_entry_block(body);
  auto body_builder = builder_;
  loom_builder_enter_region(&body_builder, loop, body);
  loom_builder_set_block(&body_builder, block);

  loom_op_t* first = nullptr;
  IREE_ASSERT_OK(loom_test_branch_build(&body_builder, truth, &i1, 1,
                                        nullptr, 0, LOOM_LOCATION_UNKNOWN,
                                        &first));
  loom_op_t* first_yield = nullptr;
  for (uint8_t arm = 0; arm < 2; ++arm) {
    loom_region_t* region = loom_op_regions(first)[arm];
    auto arm_builder = body_builder;
    loom_builder_enter_region(&arm_builder, first, region);
    loom_builder_set_block(&arm_builder, loom_region_entry_block(region));
    loom_value_id_t value = arm == 0 ? truth : falsehood;
    loom_op_t* yield = nullptr;
    IREE_ASSERT_OK(loom_test_yield_build(&arm_builder, &value, 1,
                                         LOOM_LOCATION_UNKNOWN, &yield));
    if (arm == 0) first_yield = yield;
  }
  loom_op_t* second = nullptr;
  IREE_ASSERT_OK(loom_test_branch_build(
      &body_builder, loom_op_results(first)[0], &index, 1, nullptr, 0,
      LOOM_LOCATION_UNKNOWN, &second));
  for (uint8_t arm = 0; arm < 2; ++arm) {
    loom_region_t* region = loom_op_regions(second)[arm];
    auto arm_builder = body_builder;
    loom_builder_enter_region(&arm_builder, second, region);
    loom_builder_set_block(&arm_builder, loom_region_entry_block(region));
    loom_value_id_t value = arm == 0 ? one : two;
    loom_op_t* yield = nullptr;
    IREE_ASSERT_OK(loom_test_yield_build(&arm_builder, &value, 1,
                                         LOOM_LOCATION_UNKNOWN, &yield));
  }
  loom_op_t* add = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(
      &body_builder, loom_block_arg_id(block, 1),
      loom_op_results(second)[0], index, LOOM_LOCATION_UNKNOWN, &add));
  const loom_value_id_t next = loom_index_add_result(add);
  loom_op_t* body_yield = nullptr;
  IREE_ASSERT_OK(loom_scf_yield_build(&body_builder, &next, 1,
                                      LOOM_LOCATION_UNKNOWN, &body_yield));
  loom_op_t* function_yield = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN,
                                       &function_yield));

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
  struct FullBodyObserver {
    decltype(facts->summary_computed) previous;
    const loom_op_t* first;
    const loom_op_t* second;
    uint32_t first_count = 0;
    uint32_t second_count = 0;
  } observer{facts->summary_computed, first, second};
  facts->summary_computed = {
      .user_data = &observer,
      .fn = [](void* user_data, const loom_op_t* op) {
        auto* state = static_cast<FullBodyObserver*>(user_data);
        state->first_count += op == state->first;
        state->second_count += op == state->second;
        state->previous.fn(state->previous.user_data, op);
      },
  };
  auto expect_fresh = [&]() {
    loom_pass_value_fact_owner_t fresh_owner;
    loom_value_fact_table_t* fresh = nullptr;
    ComputeFullTraversalFacts(&fresh_owner, &fresh);
    for (loom_value_id_t value : {loom_op_results(first)[0],
                                  loom_op_results(second)[0], next,
                                  loom_op_results(loop)[0]}) {
      const loom_type_t type = loom_module_value_type(module_, value);
      EXPECT_TRUE(loom_value_fact_table_facts_equal_for_type(
          module_, type, facts, loom_value_fact_table_lookup(facts, value),
          fresh, loom_value_fact_table_lookup(fresh, value)))
          << value;
      EXPECT_EQ(loom_value_fact_table_query_identity(facts, value),
                loom_value_fact_table_query_identity(fresh, value));
      loom_value_fact_origin_snapshot_t actual_origin;
      loom_value_fact_origin_snapshot_t expected_origin;
      loom_value_fact_table_snapshot_origins(facts, value, &actual_origin);
      loom_value_fact_table_snapshot_origins(fresh, value, &expected_origin);
      EXPECT_TRUE(loom_value_fact_origin_snapshot_equal(&actual_origin,
                                                        &expected_origin));
      EXPECT_TRUE(loom_value_fact_vector_partition_equal(
          loom_value_fact_table_query_vector_partition(facts, value),
          loom_value_fact_table_query_vector_partition(fresh, value)));
    }
    loom_pass_value_fact_owner_deinitialize(&fresh_owner);
  };

  // A generic attached owner has no per-loop witness until a queued full
  // summary completes. Subsequent edits exercise parent-first publication.
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter, body_yield, 0, zero));
  IREE_ASSERT_OK(loom_rewriter_propagate_value_facts(&rewriter));
  EXPECT_GT(observer.first_count, 0u);
  EXPECT_GT(observer.second_count, 0u);
  expect_fresh();
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter, first_yield, 0,
                                           falsehood));
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter, body_yield, 0, next));
  IREE_ASSERT_OK(loom_rewriter_propagate_value_facts(&rewriter));
  expect_fresh();

  // The first branch has a semantic edit with equal numeric falsehood while the
  // second branch's selector changes directly before the outer yield joins.
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter, first_yield, 0,
                                           same_falsehood));
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter, second, 0, truth));
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter, body_yield, 0, zero));
  IREE_ASSERT_OK(loom_rewriter_propagate_value_facts(&rewriter));
  expect_fresh();
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter, second, 0,
                                           loom_op_results(first)[0]));
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter, body_yield, 0, next));
  IREE_ASSERT_OK(loom_rewriter_propagate_value_facts(&rewriter));
  expect_fresh();

  loom_rewriter_deinitialize(&rewriter);
  loom_pass_value_fact_owner_deinitialize(&owner);
  iree_arena_deinitialize(&arena);
}

// scf.select may inspect an index.cmp through its condition identity. Its
// comparison inputs must remain in the feedback dependency closure even though
// they are not direct select operands.
TEST_F(FactSummaryTest, CountedFeedbackWithSelectComparison) {
  const loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  auto constant = [&](int64_t value) {
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value), index, LOOM_LOCATION_UNKNOWN, &op));
    return op;
  };
  const loom_value_id_t begin = loom_index_constant_result(constant(0));
  const loom_value_id_t limit = loom_index_constant_result(constant(16));
  const loom_value_id_t step = loom_index_constant_result(constant(1));
  const loom_value_id_t initial = loom_index_constant_result(constant(0));
  loom_op_t* threshold = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(8), index,
                                          LOOM_LOCATION_UNKNOWN, &threshold));
  loom_op_t* loop = nullptr;
  IREE_ASSERT_OK(loom_scf_for_build(
      &builder_, 0, begin, limit, step, &initial, 1, nullptr, nullptr, 0,
      LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, 0, 0,
      LOOM_LOCATION_UNKNOWN, &loop));
  loom_region_t* body = loom_loop_like_body(loom_loop_like_cast(module_, loop));
  loom_block_t* block = loom_region_entry_block(body);
  const loom_value_id_t carried = loom_block_arg_id(block, 1);
  auto body_builder = builder_;
  loom_builder_enter_region(&body_builder, loop, body);
  loom_builder_set_block(&body_builder, block);
  loom_op_t* comparison = nullptr;
  IREE_ASSERT_OK(loom_index_cmp_build(
      &body_builder, LOOM_INDEX_CMP_PREDICATE_SLT, carried,
      loom_test_constant_result(threshold), LOOM_LOCATION_UNKNOWN,
      &comparison));
  loom_op_t* select = nullptr;
  IREE_ASSERT_OK(loom_scf_select_build(
      &body_builder, loom_index_cmp_result(comparison), carried,
      loom_test_constant_result(threshold), index, LOOM_LOCATION_UNKNOWN,
      &select));
  loom_op_t* add = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(
      &body_builder, loom_scf_select_result(select), step, index,
      LOOM_LOCATION_UNKNOWN, &add));
  const loom_value_id_t next = loom_index_add_result(add);
  loom_op_t* body_yield = nullptr;
  IREE_ASSERT_OK(loom_scf_yield_build(&body_builder, &next, 1,
                                      LOOM_LOCATION_UNKNOWN, &body_yield));
  loom_op_t* function_yield = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &function_yield));

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
  for (int64_t bound : {8, 4, 8}) {
    IREE_ASSERT_OK(loom_test_constant_rewrite_value(
        &rewriter, threshold, loom_attr_i64(bound)));
    IREE_ASSERT_OK(loom_rewriter_propagate_value_facts(&rewriter));
    loom_pass_value_fact_owner_t fresh_owner;
    loom_value_fact_table_t* fresh = nullptr;
    ComputeFullTraversalFacts(&fresh_owner, &fresh);
    for (loom_value_id_t value : {loom_index_cmp_result(comparison),
                                  loom_scf_select_result(select), next,
                                  loom_op_results(loop)[0]}) {
      const loom_type_t type = loom_module_value_type(module_, value);
      EXPECT_TRUE(loom_value_fact_table_facts_equal_for_type(
          module_, type, facts, loom_value_fact_table_lookup(facts, value),
          fresh, loom_value_fact_table_lookup(fresh, value)))
          << bound << ":" << value;
    }
    loom_pass_value_fact_owner_deinitialize(&fresh_owner);
  }
  loom_rewriter_deinitialize(&rewriter);
  loom_pass_value_fact_owner_deinitialize(&owner);
  iree_arena_deinitialize(&arena);
}

// Rewriters may publish a newly built loop before its body terminator exists.
// Its provisional summary must retain the established incomplete-body path.
TEST_F(FactSummaryTest, UnfinishedCountedBodyHasProvisionalFacts) {
  const loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  auto constant = [&](int64_t value) {
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value), index, LOOM_LOCATION_UNKNOWN, &op));
    return loom_index_constant_result(op);
  };
  const loom_value_id_t begin = constant(0);
  const loom_value_id_t limit = constant(16);
  const loom_value_id_t step = constant(1);
  const loom_value_id_t initial = constant(3);
  loom_op_t* loop = nullptr;
  IREE_ASSERT_OK(loom_scf_for_build(
      &builder_, 0, begin, limit, step, &initial, 1, nullptr, nullptr, 0,
      LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, 0, 0,
      LOOM_LOCATION_UNKNOWN, &loop));
  loom_op_t* function_yield = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &function_yield));
  loom_pass_value_fact_owner_t owner;
  loom_pass_value_fact_owner_initialize(&block_pool_, &owner);
  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner, module_, loom_pass_value_fact_scope_function(function_),
      &facts));
  EXPECT_TRUE(loom_value_fact_table_has_entry(facts, loom_op_results(loop)[0]));
  loom_pass_value_fact_owner_deinitialize(&owner);
}

// More carried values than the sparse work budget must use the full solver.
// Keep one changing recurrence so the second feedback round is reached.
TEST_F(FactSummaryTest, CountedFeedbackBeyondSparseBudget) {
  constexpr uint16_t kCarriedCount = 129;
  const loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  auto constant = [&](int64_t value) {
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value), index, LOOM_LOCATION_UNKNOWN, &op));
    return loom_index_constant_result(op);
  };
  const loom_value_id_t begin = constant(0);
  const loom_value_id_t limit = constant(16);
  const loom_value_id_t step = constant(1);
  const loom_value_id_t zero = constant(0);
  loom_value_id_t initial[kCarriedCount];
  for (loom_value_id_t& value : initial) value = zero;
  loom_op_t* loop = nullptr;
  IREE_ASSERT_OK(loom_scf_for_build(
      &builder_, 0, begin, limit, step, initial, kCarriedCount, nullptr,
      nullptr, 0, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, 0, 0,
      LOOM_LOCATION_UNKNOWN, &loop));
  loom_region_t* body = loom_loop_like_body(loom_loop_like_cast(module_, loop));
  loom_block_t* block = loom_region_entry_block(body);
  auto body_builder = builder_;
  loom_builder_enter_region(&body_builder, loop, body);
  loom_builder_set_block(&body_builder, block);
  loom_value_id_t next[kCarriedCount];
  for (uint16_t i = 0; i < kCarriedCount; ++i) {
    next[i] = loom_block_arg_id(block, i + 1);
  }
  loom_op_t* increment = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&body_builder, next[0], step, index,
                                       LOOM_LOCATION_UNKNOWN, &increment));
  next[0] = loom_index_add_result(increment);
  loom_op_t* body_yield = nullptr;
  IREE_ASSERT_OK(loom_scf_yield_build(&body_builder, next, kCarriedCount,
                                      LOOM_LOCATION_UNKNOWN, &body_yield));
  loom_op_t* function_yield = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &function_yield));

  loom_pass_value_fact_owner_t owner;
  loom_pass_value_fact_owner_initialize(&block_pool_, &owner);
  loom_value_fact_table_t* facts = nullptr;
  const auto scope = loom_pass_value_fact_scope_function(function_);
  IREE_ASSERT_OK(
      loom_pass_value_fact_owner_acquire(&owner, module_, scope, &facts));
  loom_pass_value_fact_owner_t full_owner;
  loom_value_fact_table_t* full = nullptr;
  ComputeFullTraversalFacts(&full_owner, &full);
  for (uint16_t i = 0; i < kCarriedCount; ++i) {
    const loom_value_id_t result = loom_op_results(loop)[i];
    EXPECT_TRUE(loom_value_fact_table_facts_equal_for_type(
        module_, index, facts, loom_value_fact_table_lookup(facts, result),
        full, loom_value_fact_table_lookup(full, result))) << i;
  }
  loom_pass_value_fact_owner_deinitialize(&full_owner);
  loom_pass_value_fact_owner_deinitialize(&owner);
}
}  // namespace
}  // namespace loom
