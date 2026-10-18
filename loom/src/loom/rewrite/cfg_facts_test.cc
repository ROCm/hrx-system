// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/rewrite/rewriter.h"

namespace loom {
namespace {

// An acyclic join whose left predecessor is governed by a nonlocal selector.
// The two arguments and their arithmetic consumer must track independent
// payload, numeric, control, and topology edits through the public rewriter.
class CFGFactsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const auto* vtables = loom_test_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                 vtables, count));
    vtables = loom_cfg_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_CFG,
                                                 vtables, count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("cfg_facts"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&builder, IREE_SV("join"), &name));
    uint16_t symbol = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
    loom_op_t* function = nullptr;
    IREE_ASSERT_OK(loom_test_func_build(
        &builder, 0, 0, 0, {.module_id = 0, .symbol_id = symbol}, nullptr, 0,
        nullptr, 0, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &function));
    function_ = loom_func_like_cast(module_, function);
    region_ = loom_func_like_body(function_);
    region_->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
    loom_builder_enter_region(&builder, function, region_);
    loom_builder_set_block(&builder, loom_region_entry_block(region_));
    const loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
    loom_op_t* uniform = nullptr;
    IREE_ASSERT_OK(loom_test_constant_build(&builder, loom_attr_i64(1), i1,
                                            LOOM_LOCATION_UNKNOWN, &uniform));
    uniform_ = loom_test_constant_result(uniform);
    loom_op_t* unknown = nullptr;
    IREE_ASSERT_OK(loom_test_attrs_build(&builder, 0, uniform_, {}, i1,
                                         LOOM_LOCATION_UNKNOWN, &unknown));
    unknown_ = loom_test_attrs_result(unknown);
    for (int i = 0; i < 2; ++i) {
      IREE_ASSERT_OK(
          loom_test_constant_build(&builder, loom_attr_i64(10 + i * 10), type_,
                                   LOOM_LOCATION_UNKNOWN, &constants_[i]));
      inputs_[i] = loom_test_constant_result(constants_[i]);
    }
    loom_block_t* bridge = nullptr;
    loom_block_t* join = nullptr;
    for (loom_block_t** block : {&left_, &bridge, &right_, &join}) {
      IREE_ASSERT_OK(loom_region_append_block(module_, region_, block));
    }
    for (auto& value : arguments_) {
      IREE_ASSERT_OK(
          loom_builder_define_block_arg(&builder, join, type_, &value));
    }
    IREE_ASSERT_OK(loom_cfg_cond_br_build(&builder, uniform_, left_, right_,
                                          LOOM_LOCATION_UNKNOWN, &selector_));
    loom_op_t* branch = nullptr;
    loom_builder_set_block(&builder, left_);
    IREE_ASSERT_OK(loom_cfg_br_build(&builder, bridge, nullptr, 0,
                                     LOOM_LOCATION_UNKNOWN, &branch));
    loom_builder_set_block(&builder, bridge);
    IREE_ASSERT_OK(loom_cfg_br_build(&builder, join, inputs_, 2,
                                     LOOM_LOCATION_UNKNOWN, &incoming_[0]));
    const loom_value_id_t reversed[] = {inputs_[1], inputs_[0]};
    loom_builder_set_block(&builder, right_);
    IREE_ASSERT_OK(loom_cfg_br_build(&builder, join, reversed, 2,
                                     LOOM_LOCATION_UNKNOWN, &incoming_[1]));
    loom_builder_set_block(&builder, join);
    loom_op_t* sum = nullptr;
    IREE_ASSERT_OK(loom_test_addi_build(&builder, arguments_[0], arguments_[1],
                                        type_, LOOM_LOCATION_UNKNOWN, &sum));
    sum_ = loom_test_addi_result(sum);
    IREE_ASSERT_OK(loom_test_use_build(&builder, &sum_, 1,
                                       LOOM_LOCATION_UNKNOWN, &branch));
    IREE_ASSERT_OK(loom_test_yield_build(&builder, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &branch));
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts_, &arena_, 0));
    IREE_ASSERT_OK(loom_value_fact_table_compute(&facts_, module_, function_));
    loom_rewriter_initialize(&rewriter_, module_, &arena_);
    IREE_ASSERT_OK(loom_rewriter_enable_worklist(&rewriter_));
    loom_rewriter_attach_value_facts(&rewriter_, &facts_);
  }

  void TearDown() override {
    loom_rewriter_deinitialize(&rewriter_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void CheckFresh(bool uniform) {
    while (loom_op_t* op = loom_rewriter_pop(&rewriter_)) {
      bool folded = false;
      IREE_ASSERT_OK(loom_rewriter_try_fold(&rewriter_, op, &folded));
      EXPECT_FALSE(folded);
    }
    iree_arena_allocator_t arena;
    iree_arena_initialize(&pool_, &arena);
    loom_value_fact_table_t fresh;
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&fresh, &arena, 0));
    IREE_ASSERT_OK(loom_value_fact_table_compute(&fresh, module_, function_));
    for (loom_value_id_t value : {arguments_[0], arguments_[1], sum_}) {
      const auto actual = loom_value_fact_table_lookup(&facts_, value);
      EXPECT_TRUE(loom_value_fact_table_facts_equal_for_type(
          module_, type_, &facts_, actual, &fresh,
          loom_value_fact_table_lookup(&fresh, value)));
      EXPECT_EQ(loom_value_facts_is_cluster_uniform(actual), uniform);
    }
    iree_arena_deinitialize(&arena);
  }

  // Shared backing for IR, retained facts, and temporary fresh analyses.
  iree_arena_block_pool_t pool_ = {};
  // Fact and rewriter storage for this fixture.
  iree_arena_allocator_t arena_ = {};
  // Registered CFG and test operation contracts.
  loom_context_t context_ = {};
  // Owned source module.
  loom_module_t* module_ = nullptr;
  // Function containing the acyclic graph.
  loom_func_like_t function_ = {};
  // Borrowed function body, edited by topology checks.
  loom_region_t* region_ = nullptr;
  // Scalar payload type shared by the inputs, join, and arithmetic consumer.
  loom_type_t type_ = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  // Input definitions whose numeric facts can change without identity edits.
  loom_op_t* constants_[2] = {};
  // Input value identities available at both incoming branches.
  loom_value_id_t inputs_[2] = {};
  // Joined values whose maintenance is under test.
  loom_value_id_t arguments_[2] = {};
  // Downstream arithmetic result used to check propagation after the join.
  loom_value_id_t sum_ = LOOM_VALUE_ID_INVALID;
  // Selector with known uniform distribution.
  loom_value_id_t uniform_ = LOOM_VALUE_ID_INVALID;
  // Selector with unknown distribution.
  loom_value_id_t unknown_ = LOOM_VALUE_ID_INVALID;
  // Entry branch selecting the two nonlocal incoming paths.
  loom_op_t* selector_ = nullptr;
  // Original left entry successor, retained across topology changes.
  loom_block_t* left_ = nullptr;
  // Original right entry successor, retained across topology changes.
  loom_block_t* right_ = nullptr;
  // Branches providing the two incoming payloads.
  loom_op_t* incoming_[2] = {};
  // Canonical value facts maintained across the mutations.
  loom_value_fact_table_t facts_ = {};
  // Mutation and worklist owner attached to the canonical facts.
  loom_rewriter_t rewriter_ = {};
};

TEST_F(CFGFactsTest, IncomingFactsAndPayloadEditsMatchFreshAnalysis) {
  IREE_ASSERT_OK(loom_rewriter_seed_function(&rewriter_, function_));
  CheckFresh(true);
  EXPECT_FALSE(
      iree_any_bit_set(rewriter_.flags, LOOM_REWRITER_FLAG_FACTS_CHANGED));
  IREE_ASSERT_OK(
      loom_rewriter_set_attr(&rewriter_, constants_[0], 0, loom_attr_i64(7)));
  CheckFresh(true);
  IREE_ASSERT_OK(loom_rewriter_set_operands(&rewriter_, incoming_[1], inputs_));
  CheckFresh(true);
  EXPECT_TRUE(loom_value_facts_is_exact(
      loom_value_fact_table_lookup(&facts_, arguments_[0])));
  IREE_ASSERT_OK(
      loom_rewriter_set_operand(&rewriter_, incoming_[0], 0, inputs_[1]));
  CheckFresh(true);
  IREE_ASSERT_OK(loom_rewriter_replace_all_uses_except(
      &rewriter_, inputs_[0], inputs_[1], incoming_[0]));
  CheckFresh(true);
  IREE_ASSERT_OK(
      loom_rewriter_replace_all_uses_with(&rewriter_, inputs_[1], inputs_[0]));
  CheckFresh(true);
}

TEST_F(CFGFactsTest, NonlocalControlAndTopologyEditsMatchFreshAnalysis) {
  for (loom_value_id_t selector : {unknown_, uniform_, unknown_}) {
    IREE_ASSERT_OK(
        loom_rewriter_set_operand(&rewriter_, selector_, 0, selector));
    CheckFresh(selector == uniform_);
  }
  // Replacing both incoming alternatives with one value removes the join's
  // dependence on the selector distribution.
  IREE_ASSERT_OK(
      loom_rewriter_replace_all_uses_with(&rewriter_, inputs_[1], inputs_[0]));
  CheckFresh(true);
  IREE_ASSERT_OK(
      loom_rewriter_set_attr(&rewriter_, constants_[1], 0, loom_attr_i64(20)));
  const loom_value_id_t repeated[] = {inputs_[1], inputs_[1]};
  IREE_ASSERT_OK(
      loom_rewriter_set_operands(&rewriter_, incoming_[1], repeated));
  CheckFresh(false);
  for (loom_block_t* successor : {right_, left_, right_}) {
    loom_op_successors(selector_)[0] = successor;
    IREE_ASSERT_OK(loom_rewriter_refresh_cfg_facts(&rewriter_, region_));
    CheckFresh(successor == right_);
  }
}

}  // namespace
}  // namespace loom
