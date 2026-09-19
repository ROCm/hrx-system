// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/greedy.h"

#include <cstdint>
#include <tuple>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/control_uniformity.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/target/facts.h"
#include "loom/target/types.h"

namespace loom {
namespace {

static const loom_target_fact_type_t kTestTargetFactType = {
    /*.name=*/IREE_SVL("test"),
    /*.storage_size=*/sizeof(loom_target_facts_t),
};

static void InitializeTestTargetFacts(loom_target_facts_t* out_facts) {
  *out_facts = {
      /*.fact_type=*/&kTestTargetFactType,
      /*.selector=*/0,
      /*.explicit_fields=*/0,
      /*.storage=*/
      {
          /*.snapshot=*/{/*.name=*/IREE_SVL("test")},
          /*.export_plan=*/{/*.name=*/IREE_SVL("test")},
          /*.config=*/{/*.name=*/IREE_SVL("test")},
          /*.bundle=*/{/*.name=*/IREE_SVL("test")},
      },
  };
  loom_target_bundle_storage_rebind(&out_facts->storage);
}

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
    vtables = loom_cfg_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_CFG, vtables, (uint16_t)vtable_count));
    vtables = loom_buffer_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_BUFFER, vtables, (uint16_t)vtable_count));
    vtables = loom_scf_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SCF, vtables, (uint16_t)vtable_count));
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
    loom_builder_enter_region(&builder_, function_.op,
                              loom_func_like_body(function_));
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
  if (!loom_test_constant_isa(op)) {
    return iree_ok_status();
  }
  int64_t value = loom_attr_as_i64(loom_op_attrs(op)[0]);
  if (value != 1) {
    return iree_ok_status();
  }
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
  if (!loom_test_constant_isa(op)) {
    return iree_ok_status();
  }
  int64_t value = loom_attr_as_i64(loom_op_attrs(op)[0]);
  if (value != 2) {
    return iree_ok_status();
  }
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
  if (!loom_test_constant_isa(op)) {
    return iree_ok_status();
  }
  int64_t value = loom_attr_as_i64(loom_op_attrs(op)[0]);
  if (value != 2) {
    return iree_ok_status();
  }
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

TEST_F(GreedyRewriteTest, ExplicitTargetFactsSetAnalysisScope) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  ASSERT_NE(const_op, nullptr);

  loom_target_facts_t target_facts;
  InitializeTestTargetFacts(&target_facts);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_pass_value_fact_owner_t fact_owner;
  loom_pass_value_fact_owner_initialize(&block_pool_, &fact_owner);
  loom_value_fact_table_t* fact_table = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &fact_owner, module_,
      loom_pass_value_fact_scope_function_for_target(function_, &target_facts),
      &fact_table));
  loom_greedy_rewrite_driver_t driver;
  loom_greedy_rewrite_driver_initialize(module_, &arena, fact_table, &driver);

  const loom_greedy_rewrite_options_t options = {
      /*.max_iterations=*/{},
  };
  IREE_ASSERT_OK(loom_greedy_rewrite_run_region(
      &driver, function_, loom_func_like_body(function_), function_.op,
      &options, /*callbacks=*/NULL, /*out_result=*/NULL));

  const loom_value_fact_table_t* maintained_facts =
      loom_greedy_rewrite_driver_fact_table(&driver);
  ASSERT_NE(maintained_facts, nullptr);
  EXPECT_EQ(maintained_facts->context.target_facts, &target_facts);

  loom_greedy_rewrite_driver_deinitialize(&driver);
  loom_pass_value_fact_owner_deinitialize(&fact_owner);
  iree_arena_deinitialize(&arena);
}

TEST_F(GreedyRewriteTest, AttributeMutationRefreshesConstantFacts) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* constant_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &constant_op));
  loom_value_id_t result = loom_test_constant_result(constant_op);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_pass_value_fact_owner_t fact_owner;
  loom_pass_value_fact_owner_initialize(&block_pool_, &fact_owner);
  loom_value_fact_table_t* facts = NULL;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_prepare(
      &fact_owner, module_, loom_pass_value_fact_scope_function(function_),
      &facts));
  loom_rewriter_t rewriter;
  IREE_ASSERT_OK(loom_rewriter_initialize(&rewriter, module_, &arena));
  IREE_ASSERT_OK(loom_rewriter_enable_analysis(&rewriter, function_, facts));

  int64_t value = 0;
  ASSERT_TRUE(loom_value_facts_as_exact_i64(
      loom_rewriter_value_facts(&rewriter, result), &value));
  EXPECT_EQ(value, 1);
  IREE_ASSERT_OK(loom_rewriter_set_attr(&rewriter, constant_op,
                                        loom_test_constant_value_ATTR_INDEX,
                                        loom_attr_i64(7)));
  ASSERT_TRUE(loom_value_facts_as_exact_i64(
      loom_rewriter_value_facts(&rewriter, result), &value));
  EXPECT_EQ(value, 7);

  loom_rewriter_deinitialize(&rewriter);
  loom_pass_value_fact_owner_deinitialize(&fact_owner);
  iree_arena_deinitialize(&arena);
}

TEST_F(GreedyRewriteTest, OperandMutationRequeuesUsersWhenFactsRemainEqual) {
  const loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* first = nullptr;
  loom_op_t* second = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(7), i32,
                                          LOOM_LOCATION_UNKNOWN, &first));
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(7), i32,
                                          LOOM_LOCATION_UNKNOWN, &second));
  loom_op_t* producer = nullptr;
  IREE_ASSERT_OK(loom_test_neg_build(&builder_,
                                     loom_test_constant_result(first), i32,
                                     LOOM_LOCATION_UNKNOWN, &producer));
  loom_op_t* user = nullptr;
  IREE_ASSERT_OK(loom_test_neg_build(&builder_, loom_test_neg_result(producer),
                                     i32, LOOM_LOCATION_UNKNOWN, &user));

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_pass_value_fact_owner_t fact_owner;
  loom_pass_value_fact_owner_initialize(&block_pool_, &fact_owner);
  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &fact_owner, module_, loom_pass_value_fact_scope_function(function_),
      &facts));
  loom_rewriter_t rewriter;
  IREE_ASSERT_OK(loom_rewriter_initialize(&rewriter, module_, &arena));
  loom_rewriter_attach_value_facts(&rewriter, facts);

  const loom_value_facts_t before =
      loom_rewriter_value_facts(&rewriter, loom_test_neg_result(producer));
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter, producer, 0,
                                           loom_test_constant_result(second)));
  EXPECT_TRUE(loom_value_fact_table_facts_equal_for_type(
      module_, i32, facts, before, facts,
      loom_rewriter_value_facts(&rewriter, loom_test_neg_result(producer))));

  bool saw_producer = false;
  bool saw_user = false;
  while (loom_op_t* op = loom_rewriter_pop(&rewriter)) {
    saw_producer |= op == producer;
    saw_user |= op == user;
  }
  EXPECT_TRUE(saw_producer);
  EXPECT_TRUE(saw_user);

  loom_rewriter_deinitialize(&rewriter);
  loom_pass_value_fact_owner_deinitialize(&fact_owner);
  iree_arena_deinitialize(&arena);
}

TEST_F(GreedyRewriteTest, CyclicFactsNarrowAfterSemanticUpdates) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_region_t* body = loom_func_like_body(function_);
  body->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
  loom_op_t* seed = nullptr;
  loom_op_t* increment = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32,
                                          LOOM_LOCATION_UNKNOWN, &seed));
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32,
                                          LOOM_LOCATION_UNKNOWN, &increment));
  loom_block_t* header = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &header));
  loom_value_id_t carried = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_define_block_arg(&builder_, header, i32, &carried));
  loom_value_id_t seed_value = loom_test_constant_result(seed);
  loom_op_t* entry_branch = nullptr;
  IREE_ASSERT_OK(loom_cfg_br_build(&builder_, header, &seed_value, 1,
                                   LOOM_LOCATION_UNKNOWN, &entry_branch));
  loom_builder_set_block(&builder_, header);
  loom_op_t* sum = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, carried,
                                      loom_test_constant_result(increment), i32,
                                      LOOM_LOCATION_UNKNOWN, &sum));
  loom_value_id_t sum_value = loom_test_addi_result(sum);
  loom_op_t* backedge = nullptr;
  IREE_ASSERT_OK(loom_cfg_br_build(&builder_, header, &sum_value, 1,
                                   LOOM_LOCATION_UNKNOWN, &backedge));

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_pass_value_fact_owner_t owner;
  loom_pass_value_fact_owner_initialize(&block_pool_, &owner);
  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner, module_, loom_pass_value_fact_scope_function(function_), &facts));
  loom_rewriter_t rewriter;
  IREE_ASSERT_OK(loom_rewriter_initialize(&rewriter, module_, &arena));
  loom_rewriter_attach_value_facts(&rewriter, facts);
  iree_host_size_t touched_count = facts->touched_count;
  EXPECT_TRUE(
      loom_value_facts_is_exact(loom_rewriter_value_facts(&rewriter, carried)));
  EXPECT_EQ(loom_rewriter_value_facts(&rewriter, carried).range_lo, 2);

  // Changing the recurrence first widens the loop, then restores its exact
  // fixed point. Old arithmetic feedback must not survive the second edit.
  for (int64_t delta : {1, 0, 1, 0}) {
    IREE_ASSERT_OK(loom_rewriter_set_attr(&rewriter, increment,
                                          loom_test_constant_value_ATTR_INDEX,
                                          loom_attr_i64(delta)));
    while (loom_op_t* op = loom_rewriter_pop(&rewriter)) {
      bool folded = false;
      IREE_ASSERT_OK(loom_rewriter_try_fold(&rewriter, op, &folded));
      EXPECT_FALSE(folded);
    }
    loom_pass_value_fact_owner_t fresh_owner;
    loom_pass_value_fact_owner_initialize(&block_pool_, &fresh_owner);
    loom_value_fact_table_t* fresh = nullptr;
    IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
        &fresh_owner, module_, loom_pass_value_fact_scope_function(function_),
        &fresh));
    for (loom_value_id_t value : {carried, sum_value}) {
      EXPECT_TRUE(loom_value_fact_table_facts_equal_for_type(
          module_, i32, facts, loom_value_fact_table_lookup(facts, value),
          fresh, loom_value_fact_table_lookup(fresh, value)));
    }
    EXPECT_EQ(loom_value_facts_is_exact(
                  loom_rewriter_value_facts(&rewriter, carried)),
              delta == 0);
    EXPECT_EQ(facts->touched_count, touched_count);
    loom_pass_value_fact_owner_deinitialize(&fresh_owner);
  }
  // Structural snapshots belong to the rewriter and leave the caller-owned
  // table before their backing arenas are released or the table is reattached.
  std::vector<const loom_cfg_graph_t*> graphs;
  loom_value_fact_cfg_graph_callback_t callback = {};
  callback.user_data = &graphs;
  callback.fn = [](void* user_data, const loom_cfg_graph_t* graph) {
    static_cast<std::vector<const loom_cfg_graph_t*>*>(user_data)->push_back(
        graph);
    return iree_ok_status();
  };
  // Non-CFG region context must not be exposed as a structural snapshot.
  IREE_ASSERT_OK(loom_value_fact_table_set_region_temporal_scope(
      facts, module_->body, loom_value_facts_unknown()));
  IREE_ASSERT_OK(loom_rewriter_refresh_cfg_facts(&rewriter, body));
  IREE_ASSERT_OK(loom_rewriter_refresh_cfg_facts(&rewriter, body));
  EXPECT_NE(loom_value_fact_table_lookup_cfg_graph(facts, body), nullptr);
  IREE_ASSERT_OK(loom_value_fact_table_enumerate_cfg_graphs(facts, callback));
  EXPECT_THAT(graphs, ::testing::ElementsAre(
                          loom_value_fact_table_lookup_cfg_graph(facts, body)));
  loom_value_fact_cfg_graph_callback_t failing_callback = {};
  failing_callback.fn = [](void*, const loom_cfg_graph_t*) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_value_fact_table_enumerate_cfg_graphs(facts, failing_callback));
  IREE_ASSERT_OK(loom_rewriter_enable_analysis(&rewriter, function_, facts));
  IREE_ASSERT_OK(loom_rewriter_refresh_cfg_facts(&rewriter, body));
  loom_rewriter_deinitialize(&rewriter);
  EXPECT_EQ(loom_value_fact_table_lookup_cfg_graph(facts, body), nullptr);
  graphs.clear();
  IREE_ASSERT_OK(loom_value_fact_table_enumerate_cfg_graphs(facts, callback));
  EXPECT_TRUE(graphs.empty());
  loom_pass_value_fact_owner_deinitialize(&owner);
  iree_arena_deinitialize(&arena);
}

TEST_F(GreedyRewriteTest, SelectorEditsRefreshNonlocalAndCyclicControlFacts) {
  const loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  const loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_region_t* region = loom_func_like_body(function_);
  region->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
  loom_op_t* uniform = nullptr;
  loom_op_t* unknown = nullptr;
  loom_op_t* lhs = nullptr;
  loom_op_t* rhs = nullptr;
  loom_op_t* step = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i1,
                                          LOOM_LOCATION_UNKNOWN, &uniform));
  const loom_value_id_t uniform_id = loom_test_constant_result(uniform);
  IREE_ASSERT_OK(loom_test_attrs_build(&builder_, 0, uniform_id, {}, i1,
                                       LOOM_LOCATION_UNKNOWN, &unknown));
  const loom_value_id_t unknown_id = loom_test_attrs_result(unknown);
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(10), i32,
                                          LOOM_LOCATION_UNKNOWN, &lhs));
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(20), i32,
                                          LOOM_LOCATION_UNKNOWN, &rhs));
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &step));
  loom_block_t* left = nullptr;
  loom_block_t* left_more = nullptr;
  loom_block_t* right = nullptr;
  loom_block_t* header = nullptr;
  loom_block_t* body = nullptr;
  loom_block_t* exit = nullptr;
  for (loom_block_t** block :
       {&left, &left_more, &right, &header, &body, &exit}) {
    IREE_ASSERT_OK(loom_region_append_block(module_, region, block));
  }
  loom_value_id_t carried = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_define_block_arg(&builder_, header, i32, &carried));
  loom_op_t* entry_selector = nullptr;
  IREE_ASSERT_OK(loom_cfg_cond_br_build(&builder_, uniform_id, left, right,
                                        LOOM_LOCATION_UNKNOWN,
                                        &entry_selector));
  loom_op_t* branch = nullptr;
  loom_builder_set_block(&builder_, left);
  IREE_ASSERT_OK(loom_cfg_br_build(&builder_, left_more, nullptr, 0,
                                   LOOM_LOCATION_UNKNOWN, &branch));
  const loom_value_id_t lhs_id = loom_test_constant_result(lhs);
  loom_builder_set_block(&builder_, left_more);
  IREE_ASSERT_OK(loom_cfg_br_build(&builder_, header, &lhs_id, 1,
                                   LOOM_LOCATION_UNKNOWN, &branch));
  const loom_value_id_t rhs_id = loom_test_constant_result(rhs);
  loom_builder_set_block(&builder_, right);
  IREE_ASSERT_OK(loom_cfg_br_build(&builder_, header, &rhs_id, 1,
                                   LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_set_block(&builder_, header);
  loom_op_t* loop_selector = nullptr;
  IREE_ASSERT_OK(loom_cfg_cond_br_build(&builder_, uniform_id, body, exit,
                                        LOOM_LOCATION_UNKNOWN, &loop_selector));
  loom_builder_set_block(&builder_, body);
  loom_op_t* increment = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, carried,
                                      loom_test_constant_result(step), i32,
                                      LOOM_LOCATION_UNKNOWN, &increment));
  const loom_value_id_t next = loom_test_addi_result(increment);
  IREE_ASSERT_OK(loom_cfg_br_build(&builder_, header, &next, 1,
                                   LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_set_block(&builder_, exit);
  loom_op_t* use = nullptr;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &carried, 1, LOOM_LOCATION_UNKNOWN, &use));
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &branch));

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_pass_value_fact_owner_t owner;
  loom_pass_value_fact_owner_initialize(&block_pool_, &owner);
  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner, module_, loom_pass_value_fact_scope_function(function_), &facts));
  loom_rewriter_t rewriter;
  IREE_ASSERT_OK(loom_rewriter_initialize(&rewriter, module_, &arena));
  loom_rewriter_attach_value_facts(&rewriter, facts);

  auto check_fresh = [&](bool expected_uniform) {
    while (loom_op_t* op = loom_rewriter_pop(&rewriter)) {
      bool folded = false;
      IREE_ASSERT_OK(loom_rewriter_try_fold(&rewriter, op, &folded));
      EXPECT_FALSE(folded);
    }
    loom_pass_value_fact_owner_t fresh_owner;
    loom_pass_value_fact_owner_initialize(&block_pool_, &fresh_owner);
    loom_value_fact_table_t* fresh = nullptr;
    IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
        &fresh_owner, module_, loom_pass_value_fact_scope_function(function_),
        &fresh));
    for (loom_value_id_t value : {carried, next}) {
      EXPECT_TRUE(loom_value_fact_table_facts_equal_for_type(
          module_, i32, facts, loom_value_fact_table_lookup(facts, value),
          fresh, loom_value_fact_table_lookup(fresh, value)));
      EXPECT_EQ(loom_value_facts_is_cluster_uniform(
                    loom_value_fact_table_lookup(facts, value)),
                expected_uniform);
    }
    loom_pass_value_fact_owner_deinitialize(&fresh_owner);
  };
  for (loom_op_t* selector : {entry_selector, loop_selector}) {
    for (loom_value_id_t condition :
         {unknown_id, uniform_id, unknown_id, uniform_id}) {
      IREE_ASSERT_OK(
          loom_rewriter_set_operand(&rewriter, selector, 0, condition));
      check_fresh(condition == uniform_id);
      loom_control_uniformity_info_t info;
      loom_control_uniformity_info_initialize(module_, facts, &arena, &info);
      loom_control_uniformity_failure_t failure;
      const loom_op_t* governed =
          selector == entry_selector ? left_more->last_op : increment;
      EXPECT_EQ(
          loom_control_uniformity_prove_execution(
              &info, governed, LOOM_VALUE_FACT_UNIFORM_SCOPE_CLUSTER, &failure),
          condition == uniform_id);
      if (condition == unknown_id) {
        EXPECT_EQ(failure.control_value, unknown_id);
      }
    }
  }
  // A remote topology edit changes which selector governs the incoming values
  // without changing the loop's immediate predecessor edges or reachability.
  loom_block_t* gate = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, region, &gate));
  loom_builder_set_block(&builder_, gate);
  IREE_ASSERT_OK(loom_cfg_cond_br_build(&builder_, uniform_id, left, right,
                                        LOOM_LOCATION_UNKNOWN, &branch));
  IREE_ASSERT_OK(
      loom_rewriter_set_operand(&rewriter, entry_selector, 0, unknown_id));
  for (bool use_gate : {true, false, true, false}) {
    loom_op_successors(entry_selector)[0] = use_gate ? gate : left;
    loom_op_successors(entry_selector)[1] = use_gate ? gate : right;
    IREE_ASSERT_OK(loom_rewriter_refresh_cfg_facts(&rewriter, region));
    check_fresh(use_gate);
  }
  loom_rewriter_deinitialize(&rewriter);
  loom_pass_value_fact_owner_deinitialize(&owner);
  iree_arena_deinitialize(&arena);
}

class TemporalFactsRewriteTest : public GreedyRewriteTest,
                                 public ::testing::WithParamInterface<bool> {};

TEST_P(TemporalFactsRewriteTest, TracksSelectorsAndMovedObservers) {
  const bool has_carried_argument = GetParam();
  const loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  const loom_type_t offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  const loom_type_t buffer = loom_type_buffer();
  loom_region_t* region = loom_func_like_body(function_);
  region->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
  loom_op_t* uniform = nullptr;
  loom_op_t* unknown = nullptr;
  loom_op_t* length = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i1,
                                          LOOM_LOCATION_UNKNOWN, &uniform));
  const loom_value_id_t uniform_id = loom_test_constant_result(uniform);
  IREE_ASSERT_OK(loom_test_attrs_build(&builder_, 0, uniform_id, {}, i1,
                                       LOOM_LOCATION_UNKNOWN, &unknown));
  const loom_value_id_t unknown_id = loom_test_attrs_result(unknown);
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(16), offset,
                                          LOOM_LOCATION_UNKNOWN, &length));
  const loom_value_id_t length_id = loom_test_constant_result(length);
  loom_block_t* header = nullptr;
  loom_block_t* body = nullptr;
  loom_block_t* exit = nullptr;
  for (loom_block_t** block : {&header, &body, &exit}) {
    IREE_ASSERT_OK(loom_region_append_block(module_, region, block));
  }
  loom_value_id_t carried = LOOM_VALUE_ID_INVALID;
  if (has_carried_argument) {
    IREE_ASSERT_OK(
        loom_builder_define_block_arg(&builder_, header, offset, &carried));
  }
  const iree_host_size_t payload_count = has_carried_argument ? 1 : 0;
  loom_op_t* entry_branch = nullptr;
  IREE_ASSERT_OK(loom_cfg_br_build(&builder_, header, &length_id, payload_count,
                                   LOOM_LOCATION_UNKNOWN, &entry_branch));
  loom_builder_set_block(&builder_, header);
  loom_op_t* allocation = nullptr;
  IREE_ASSERT_OK(loom_buffer_alloca_build(
      &builder_, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, 16, length_id, buffer,
      LOOM_LOCATION_UNKNOWN, &allocation));
  const loom_value_id_t allocation_id = loom_buffer_alloca_result(allocation);
  loom_op_t* use = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &allocation_id, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));
  loom_op_t* nested_owner = nullptr;
  IREE_ASSERT_OK(loom_test_block_args_build(
      &builder_, nullptr, 0, LOOM_LOCATION_UNKNOWN, &nested_owner));
  loom_op_t* selector = nullptr;
  IREE_ASSERT_OK(loom_cfg_cond_br_build(&builder_, uniform_id, body, exit,
                                        LOOM_LOCATION_UNKNOWN, &selector));
  const loom_builder_ip_t saved = loom_builder_enter_region(
      &builder_, nested_owner, loom_test_block_args_body(nested_owner));
  loom_op_t* nested_allocation = nullptr;
  IREE_ASSERT_OK(loom_buffer_alloca_build(
      &builder_, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, 16, length_id, buffer,
      LOOM_LOCATION_UNKNOWN, &nested_allocation));
  const loom_value_id_t nested_id =
      loom_buffer_alloca_result(nested_allocation);
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &nested_id, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));
  loom_op_t* terminator = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));
  loom_builder_restore(&builder_, saved);
  loom_builder_set_block(&builder_, body);
  IREE_ASSERT_OK(loom_cfg_br_build(&builder_, header, &carried, payload_count,
                                   LOOM_LOCATION_UNKNOWN, &terminator));
  loom_builder_set_block(&builder_, exit);
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_pass_value_fact_owner_t owner;
  loom_pass_value_fact_owner_initialize(&block_pool_, &owner);
  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner, module_, loom_pass_value_fact_scope_function(function_), &facts));
  loom_rewriter_t rewriter;
  IREE_ASSERT_OK(loom_rewriter_initialize(&rewriter, module_, &arena));
  loom_rewriter_attach_value_facts(&rewriter, facts);

  auto check_fresh = [&](loom_value_id_t value, bool expected_uniform) {
    while (loom_op_t* op = loom_rewriter_pop(&rewriter)) {
      bool folded = false;
      IREE_ASSERT_OK(loom_rewriter_try_fold(&rewriter, op, &folded));
      EXPECT_FALSE(folded);
    }
    loom_pass_value_fact_owner_t fresh_owner;
    loom_pass_value_fact_owner_initialize(&block_pool_, &fresh_owner);
    loom_value_fact_table_t* fresh = nullptr;
    IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
        &fresh_owner, module_, loom_pass_value_fact_scope_function(function_),
        &fresh));
    for (loom_value_id_t compared : {allocation_id, nested_id, value}) {
      EXPECT_TRUE(loom_value_fact_table_facts_equal_for_type(
          module_, buffer, facts, loom_value_fact_table_lookup(facts, compared),
          fresh, loom_value_fact_table_lookup(fresh, compared)))
          << "value " << compared;
    }
    EXPECT_EQ(loom_value_facts_is_cluster_uniform(
                  loom_value_fact_table_lookup(facts, value)),
              expected_uniform);
    loom_pass_value_fact_owner_deinitialize(&fresh_owner);
  };
  check_fresh(allocation_id, true);
  for (loom_value_id_t condition :
       {unknown_id, uniform_id, unknown_id, uniform_id, unknown_id}) {
    IREE_ASSERT_OK(
        loom_rewriter_set_operand(&rewriter, selector, 0, condition));
    check_fresh(allocation_id, condition == uniform_id);
    check_fresh(nested_id, condition == uniform_id);
  }
  // Moving an ordinary region owner must refresh its inherited cycle context,
  // even though that owner does not have a structured result-fact summary.
  for (loom_op_t* insertion :
       {exit->last_op, selector, exit->last_op, selector}) {
    IREE_ASSERT_OK(
        loom_rewriter_move_before(&rewriter, nested_owner, insertion));
    check_fresh(nested_id, insertion == exit->last_op);
  }
  IREE_ASSERT_OK(
      loom_rewriter_move_before(&rewriter, allocation, entry_branch));
  check_fresh(allocation_id, true);
  IREE_ASSERT_OK(
      loom_rewriter_move_before(&rewriter, allocation, nested_owner));
  check_fresh(allocation_id, false);

  // Removing and restoring the backedge changes cyclic membership while the
  // surviving blocks and their observer identities remain the same.
  for (bool has_cycle : {false, true, false, true}) {
    if (has_carried_argument) {
      break;
    }
    loom_op_successors(body->last_op)[0] = has_cycle ? header : exit;
    IREE_ASSERT_OK(loom_rewriter_refresh_cfg_facts(&rewriter, region));
    check_fresh(allocation_id, !has_cycle);
    check_fresh(nested_id, !has_cycle);
  }
  if (!has_carried_argument) {
    // A backedge may include the entry block. Its ordinary definitions still
    // need to be revisited when that cycle is broken again.
    for (loom_block_t* destination :
         {loom_region_entry_block(region), exit,
          loom_region_entry_block(region), header}) {
      loom_op_successors(body->last_op)[0] = destination;
      IREE_ASSERT_OK(loom_rewriter_refresh_cfg_facts(&rewriter, region));
      check_fresh(allocation_id, destination == exit);
    }
  }

  // Builder callbacks seed an ordinary region before its first observer is
  // finalized, without requiring another whole-function analysis.
  loom_builder_set_before(&rewriter.builder, exit->last_op);
  loom_op_t* created_owner = nullptr;
  IREE_ASSERT_OK(loom_test_block_args_build(
      &rewriter.builder, nullptr, 0, LOOM_LOCATION_UNKNOWN, &created_owner));
  loom_builder_enter_region(&rewriter.builder, created_owner,
                            loom_test_block_args_body(created_owner));
  loom_op_t* created_allocation = nullptr;
  IREE_ASSERT_OK(loom_buffer_alloca_build(
      &rewriter.builder, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, 16, length_id,
      buffer, LOOM_LOCATION_UNKNOWN, &created_allocation));
  const loom_value_id_t created_id =
      loom_buffer_alloca_result(created_allocation);
  EXPECT_TRUE(loom_value_facts_is_cluster_uniform(
      loom_value_fact_table_lookup(facts, created_id)));
  IREE_ASSERT_OK(loom_test_use_build(&rewriter.builder, &created_id, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));
  IREE_ASSERT_OK(loom_test_yield_build(&rewriter.builder, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));
  check_fresh(created_id, true);
  IREE_ASSERT_OK(loom_rewriter_move_before(&rewriter, created_owner, selector));
  check_fresh(created_id, false);
  loom_rewriter_deinitialize(&rewriter);
  loom_pass_value_fact_owner_deinitialize(&owner);
  iree_arena_deinitialize(&arena);
}

INSTANTIATE_TEST_SUITE_P(CarriedAndObserverOnlyCycles, TemporalFactsRewriteTest,
                         ::testing::Bool());

TEST_F(GreedyRewriteTest, ForwardingComponentsTrackPayloadReplacements) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_region_t* body = loom_func_like_body(function_);
  body->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
  loom_value_id_t seeds[2];
  for (int i = 0; i < 2; ++i) {
    loom_op_t* constant = nullptr;
    IREE_ASSERT_OK(loom_test_constant_build(&builder_,
                                            loom_attr_i64(i == 0 ? 2 : 9), i32,
                                            LOOM_LOCATION_UNKNOWN, &constant));
    seeds[i] = loom_test_constant_result(constant);
  }
  loom_block_t* header = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &header));
  loom_value_id_t carried[2];
  for (auto& value : carried) {
    IREE_ASSERT_OK(
        loom_builder_define_block_arg(&builder_, header, i32, &value));
  }
  loom_op_t* entry_branch = nullptr;
  IREE_ASSERT_OK(loom_cfg_br_build(&builder_, header, seeds, 2,
                                   LOOM_LOCATION_UNKNOWN, &entry_branch));
  loom_builder_set_block(&builder_, header);
  loom_value_id_t swapped[2] = {carried[1], carried[0]};
  loom_op_t* backedge = nullptr;
  IREE_ASSERT_OK(loom_cfg_br_build(&builder_, header, swapped, 2,
                                   LOOM_LOCATION_UNKNOWN, &backedge));

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_pass_value_fact_owner_t owner;
  loom_pass_value_fact_owner_initialize(&block_pool_, &owner);
  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner, module_, loom_pass_value_fact_scope_function(function_), &facts));
  loom_rewriter_t rewriter;
  IREE_ASSERT_OK(loom_rewriter_initialize(&rewriter, module_, &arena));
  loom_rewriter_attach_value_facts(&rewriter, facts);

  auto check_facts = [&](bool expect_exact) {
    while (loom_op_t* op = loom_rewriter_pop(&rewriter)) {
      bool folded = false;
      IREE_ASSERT_OK(loom_rewriter_try_fold(&rewriter, op, &folded));
    }
    loom_pass_value_fact_owner_t fresh_owner;
    loom_pass_value_fact_owner_initialize(&block_pool_, &fresh_owner);
    loom_value_fact_table_t* fresh = nullptr;
    IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
        &fresh_owner, module_, loom_pass_value_fact_scope_function(function_),
        &fresh));
    for (loom_value_id_t value : carried) {
      EXPECT_TRUE(loom_value_fact_table_facts_equal_for_type(
          module_, i32, facts, loom_value_fact_table_lookup(facts, value),
          fresh, loom_value_fact_table_lookup(fresh, value)));
    }
    EXPECT_EQ(loom_value_facts_is_exact(
                  loom_rewriter_value_facts(&rewriter, carried[0])),
              expect_exact);
    loom_pass_value_fact_owner_deinitialize(&fresh_owner);
  };

  // Split and rejoin a mutual forwarding cycle without changing CFG edges.
  // Once the first argument forwards itself, it only receives the seed 2.
  for (int edit = 0; edit < 3; ++edit) {
    if (edit == 0) {
      IREE_ASSERT_OK(
          loom_rewriter_set_operand(&rewriter, backedge, 0, carried[0]));
    } else if (edit == 1) {
      IREE_ASSERT_OK(loom_rewriter_replace_all_uses_with(&rewriter, carried[1],
                                                         carried[0]));
    } else {
      IREE_ASSERT_OK(loom_rewriter_replace_all_uses_except(
          &rewriter, carried[1], carried[0], entry_branch));
    }
    check_facts(true);
    IREE_ASSERT_OK(
        loom_rewriter_set_operand(&rewriter, backedge, 0, carried[1]));
    check_facts(false);
  }
  loom_rewriter_deinitialize(&rewriter);
  loom_pass_value_fact_owner_deinitialize(&owner);
  iree_arena_deinitialize(&arena);
}

class ForwardingFactsRewriteTest
    : public GreedyRewriteTest,
      public ::testing::WithParamInterface<std::tuple<uint16_t, bool>> {};

TEST_P(ForwardingFactsRewriteTest, OpenQueuesRestartAcrossSemanticEdits) {
  const uint16_t count = std::get<0>(GetParam());
  const bool reverse_layout = std::get<1>(GetParam());
  const loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  const loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_region_t* region = loom_func_like_body(function_);
  region->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
  std::vector<loom_value_id_t> seeds(count);
  std::vector<loom_value_id_t> carried(count);
  std::vector<loom_value_id_t> forwarded(count);
  for (uint16_t i = 0; i < count; ++i) {
    loom_op_t* constant = nullptr;
    IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(i), i32,
                                            LOOM_LOCATION_UNKNOWN, &constant));
    seeds[i] = loom_test_constant_result(constant);
  }
  loom_op_t* injected = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(count + 8),
                                          i32, LOOM_LOCATION_UNKNOWN,
                                          &injected));
  const loom_value_id_t injected_id = loom_test_constant_result(injected);
  loom_op_t* uniform = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i1,
                                          LOOM_LOCATION_UNKNOWN, &uniform));
  const loom_value_id_t uniform_id = loom_test_constant_result(uniform);
  loom_op_t* unknown = nullptr;
  IREE_ASSERT_OK(loom_test_attrs_build(&builder_, 0, uniform_id, {}, i1,
                                       LOOM_LOCATION_UNKNOWN, &unknown));
  const loom_value_id_t unknown_id = loom_test_attrs_result(unknown);
  loom_op_t* unknown_input = nullptr;
  IREE_ASSERT_OK(loom_test_attrs_build(&builder_, 0, injected_id, {}, i32,
                                       LOOM_LOCATION_UNKNOWN, &unknown_input));
  const loom_value_id_t unknown_input_id =
      loom_test_attrs_result(unknown_input);
  loom_block_t* header = nullptr;
  loom_block_t* latch = nullptr;
  loom_block_t* exit = nullptr;
  const auto block_order = reverse_layout ? std::vector{&exit, &latch, &header}
                                          : std::vector{&header, &latch, &exit};
  for (loom_block_t** block : block_order) {
    IREE_ASSERT_OK(loom_region_append_block(module_, region, block));
  }
  for (loom_value_id_t& value : carried) {
    IREE_ASSERT_OK(
        loom_builder_define_block_arg(&builder_, header, i32, &value));
  }
  for (uint16_t i = 0; i < count; ++i) {
    forwarded[i] = i + 1 < count ? carried[i + 1] : injected_id;
  }
  loom_op_t* branch = nullptr;
  IREE_ASSERT_OK(loom_cfg_br_build(&builder_, header, seeds.data(), count,
                                   LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_set_block(&builder_, header);
  loom_op_t* selector = nullptr;
  IREE_ASSERT_OK(loom_cfg_cond_br_build(&builder_, uniform_id, latch, exit,
                                        LOOM_LOCATION_UNKNOWN, &selector));
  loom_builder_set_block(&builder_, latch);
  loom_op_t* backedge = nullptr;
  IREE_ASSERT_OK(loom_cfg_br_build(&builder_, header, forwarded.data(), count,
                                   LOOM_LOCATION_UNKNOWN, &backedge));
  loom_builder_set_block(&builder_, exit);
  IREE_ASSERT_OK(loom_test_use_build(&builder_, carried.data(), count,
                                     LOOM_LOCATION_UNKNOWN, &branch));
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &branch));

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_pass_value_fact_owner_t owner;
  loom_pass_value_fact_owner_initialize(&block_pool_, &owner);
  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner, module_, loom_pass_value_fact_scope_function(function_), &facts));
  loom_rewriter_t rewriter;
  IREE_ASSERT_OK(loom_rewriter_initialize(&rewriter, module_, &arena));
  loom_rewriter_attach_value_facts(&rewriter, facts);

  enum class QueueState { kOpenUniform, kOpenUnknown, kClosed, kUnknownInput };
  int64_t injected_value = count + 8;
  auto check = [&](QueueState state) {
    while (loom_op_t* op = loom_rewriter_pop(&rewriter)) {
      bool folded = false;
      IREE_ASSERT_OK(loom_rewriter_try_fold(&rewriter, op, &folded));
    }
    loom_pass_value_fact_owner_t fresh_owner;
    loom_pass_value_fact_owner_initialize(&block_pool_, &fresh_owner);
    loom_value_fact_table_t* fresh = nullptr;
    IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
        &fresh_owner, module_, loom_pass_value_fact_scope_function(function_),
        &fresh));
    for (uint16_t i = 0; i < count; ++i) {
      SCOPED_TRACE(i);
      const loom_value_facts_t value =
          loom_rewriter_value_facts(&rewriter, carried[i]);
      EXPECT_TRUE(loom_value_fact_table_facts_equal_for_type(
          module_, i32, facts, value, fresh,
          loom_value_fact_table_lookup(fresh, carried[i])));
      if (state != QueueState::kUnknownInput) {
        EXPECT_EQ(value.range_lo, state == QueueState::kClosed ? 0 : i);
        EXPECT_EQ(value.range_hi,
                  state == QueueState::kClosed ? count - 1 : injected_value);
      }
      EXPECT_EQ(
          loom_value_facts_is_cluster_uniform(value),
          state == QueueState::kOpenUniform || state == QueueState::kClosed);
    }
    loom_pass_value_fact_owner_deinitialize(&fresh_owner);
  };
  check(QueueState::kOpenUniform);

  for (int edit = 0; edit < 3; ++edit) {
    // Changing an external producer must reach the entire queue and preserve
    // each suffix's independent lower bound.
    injected_value += count;
    IREE_ASSERT_OK(loom_rewriter_set_attr(&rewriter, injected, 0,
                                          loom_attr_i64(injected_value)));
    check(QueueState::kOpenUniform);
    IREE_ASSERT_OK(
        loom_rewriter_set_operand(&rewriter, selector, 0, unknown_id));
    check(QueueState::kOpenUnknown);
    IREE_ASSERT_OK(
        loom_rewriter_set_operand(&rewriter, selector, 0, uniform_id));
    check(QueueState::kOpenUniform);

    // Joining every slot into a cycle changes the component count. Reopening
    // it must restore dependency order instead of retaining the broad cycle.
    IREE_ASSERT_OK(
        loom_rewriter_set_operand(&rewriter, backedge, count - 1, carried[0]));
    check(QueueState::kClosed);
    IREE_ASSERT_OK(
        loom_rewriter_set_operand(&rewriter, backedge, count - 1, injected_id));
    check(QueueState::kOpenUniform);
    IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter, backedge, count - 1,
                                             unknown_input_id));
    check(QueueState::kUnknownInput);
    IREE_ASSERT_OK(
        loom_rewriter_set_operand(&rewriter, backedge, count - 1, injected_id));
    check(QueueState::kOpenUniform);
  }
  loom_rewriter_deinitialize(&rewriter);
  loom_pass_value_fact_owner_deinitialize(&owner);
  iree_arena_deinitialize(&arena);
}

INSTANTIATE_TEST_SUITE_P(IndependentQueueDepth, ForwardingFactsRewriteTest,
                         ::testing::Combine(::testing::Values(1, 2, 17, 65),
                                            ::testing::Bool()));

TEST_F(GreedyRewriteTest, NonEquationEditsPreserveCyclicFacts) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_region_t* body = loom_func_like_body(function_);
  body->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
  loom_op_t* seed = nullptr;
  loom_op_t* input = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32,
                                          LOOM_LOCATION_UNKNOWN, &seed));
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(9), i32,
                                          LOOM_LOCATION_UNKNOWN, &input));
  loom_value_id_t seed_value = loom_test_constant_result(seed);
  loom_block_t* header = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &header));
  loom_value_id_t carried = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_define_block_arg(&builder_, header, i32, &carried));
  loom_op_t* entry_branch = nullptr;
  IREE_ASSERT_OK(loom_cfg_br_build(&builder_, header, &seed_value, 1,
                                   LOOM_LOCATION_UNKNOWN, &entry_branch));
  loom_builder_set_block(&builder_, header);
  loom_op_t* opaque = nullptr;
  IREE_ASSERT_OK(
      loom_test_convergent_build(&builder_, loom_test_constant_result(input),
                                 i32, LOOM_LOCATION_UNKNOWN, &opaque));
  loom_op_t* backedge = nullptr;
  IREE_ASSERT_OK(loom_cfg_br_build(&builder_, header, &carried, 1,
                                   LOOM_LOCATION_UNKNOWN, &backedge));

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_pass_value_fact_owner_t owner;
  loom_pass_value_fact_owner_initialize(&block_pool_, &owner);
  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner, module_, loom_pass_value_fact_scope_function(function_), &facts));
  loom_rewriter_t rewriter;
  IREE_ASSERT_OK(loom_rewriter_initialize(&rewriter, module_, &arena));
  loom_rewriter_attach_value_facts(&rewriter, facts);
  auto drain = [&]() {
    iree_host_size_t count = 0;
    while (loom_op_t* op = loom_rewriter_pop(&rewriter)) {
      bool folded = false;
      IREE_EXPECT_OK(loom_rewriter_try_fold(&rewriter, op, &folded));
      EXPECT_FALSE(folded);
      ++count;
    }
    return count;
  };

  // Creating and removing an unused definition only visits the new operation
  // and its operand providers. The loop's payload equation does not change.
  loom_builder_set_before(&rewriter.builder, backedge);
  loom_op_t* unused = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&rewriter.builder, carried, seed_value,
                                      i32, LOOM_LOCATION_UNKNOWN, &unused));
  EXPECT_EQ(drain(), 1u);
  IREE_ASSERT_OK(loom_rewriter_erase(&rewriter, unused));
  EXPECT_EQ(drain(), 1u);

  // An opaque operation must be revisited after input replacement, while its
  // unknown result facts do not acquire an input-dependent equation.
  IREE_ASSERT_OK(loom_rewriter_replace_all_uses_and_erase(&rewriter, input,
                                                          &seed_value, 1));
  EXPECT_EQ(drain(), 1u);
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter, opaque, 0, carried));
  EXPECT_EQ(drain(), 1u);

  // The branch has no inference callback either, but it feeds the cyclic join.
  // Replacing its payload must both weaken and recover the loop's facts.
  loom_value_id_t opaque_value = loom_test_convergent_result(opaque);
  for (loom_value_id_t payload : {opaque_value, carried}) {
    IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter, backedge, 0, payload));
    drain();
    loom_pass_value_fact_owner_t fresh_owner;
    loom_pass_value_fact_owner_initialize(&block_pool_, &fresh_owner);
    loom_value_fact_table_t* fresh = nullptr;
    IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
        &fresh_owner, module_, loom_pass_value_fact_scope_function(function_),
        &fresh));
    for (loom_value_id_t value : {carried, opaque_value}) {
      EXPECT_TRUE(loom_value_fact_table_facts_equal_for_type(
          module_, i32, facts, loom_value_fact_table_lookup(facts, value),
          fresh, loom_value_fact_table_lookup(fresh, value)));
    }
    EXPECT_EQ(loom_value_facts_is_exact(
                  loom_rewriter_value_facts(&rewriter, carried)),
              payload == carried);
    loom_pass_value_fact_owner_deinitialize(&fresh_owner);
  }

  loom_rewriter_deinitialize(&rewriter);
  loom_pass_value_fact_owner_deinitialize(&owner);
  iree_arena_deinitialize(&arena);
}

class StructuredForwardingFactsRewriteTest
    : public GreedyRewriteTest,
      public ::testing::WithParamInterface<std::tuple<uint16_t, bool>> {};

TEST_P(StructuredForwardingFactsRewriteTest,
       RebuildsAfterYieldAndConditionEdits) {
  const uint16_t count = std::get<0>(GetParam());
  const bool is_while = std::get<1>(GetParam());
  const loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  const loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  auto constant = [&](int64_t value, loom_type_t type) {
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_test_constant_build(&builder_, loom_attr_i64(value),
                                            type, LOOM_LOCATION_UNKNOWN, &op));
    return op;
  };
  std::vector<loom_value_id_t> seeds(count);
  for (uint16_t i = 0; i < count; ++i) {
    seeds[i] = loom_test_constant_result(constant(i, i32));
  }
  loom_op_t* injected = constant(count + 8, i32);
  const auto injected_id = loom_test_constant_result(injected);
  const auto begin = loom_test_constant_result(constant(0, index));
  const auto limit = loom_test_constant_result(constant(4, index));
  const auto step = loom_test_constant_result(constant(1, index));
  const auto enabled = loom_test_constant_result(constant(1, i1));
  loom_op_t* unknown_input = nullptr;
  IREE_ASSERT_OK(loom_test_attrs_build(&builder_, 0, injected_id, {}, i32,
                                       LOOM_LOCATION_UNKNOWN, &unknown_input));
  const auto unknown_input_id = loom_test_attrs_result(unknown_input);
  loom_op_t* selector = nullptr;
  IREE_ASSERT_OK(loom_test_attrs_build(&builder_, 0, enabled, {}, i1,
                                       LOOM_LOCATION_UNKNOWN, &selector));
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
  IREE_ASSERT_OK(loom_rewriter_initialize(&rewriter, module_, &arena));
  loom_rewriter_attach_value_facts(&rewriter, facts);
  loom_builder_set_before(&rewriter.builder, function_yield);

  // Real rewriter builders publish the shell before its region terminators.
  // The attached fact owner must recover as each region is completed.
  loom_op_t* loop = nullptr;
  if (is_while) {
    IREE_ASSERT_OK(loom_scf_while_build(&rewriter.builder, seeds.data(), count,
                                        nullptr, 0, LOOM_LOCATION_UNKNOWN,
                                        &loop));
  } else {
    IREE_ASSERT_OK(loom_scf_for_build(
        &rewriter.builder, 0, begin, limit, step, seeds.data(), count, nullptr,
        0, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, 0, 0,
        LOOM_LOCATION_UNKNOWN, &loop));
  }
  auto expect_summary_deferred = [&]() {
    for (uint16_t i = 0; i < count; ++i) {
      EXPECT_FALSE(
          loom_value_fact_table_has_entry(facts, loom_op_results(loop)[i]));
    }
  };
  expect_summary_deferred();
  const auto loop_like = loom_loop_like_cast(module_, loop);
  loom_region_t* body = loom_loop_like_body(loop_like);
  loom_region_t* before = loom_loop_like_condition_region(loop_like);
  loom_block_t* body_block = loom_region_entry_block(body);
  loom_block_t* before_block =
      before ? loom_region_entry_block(before) : body_block;
  std::vector<loom_value_id_t> carried(count);
  std::vector<loom_value_id_t> body_args(count);
  for (uint16_t i = 0; i < count; ++i) {
    carried[i] = loom_block_arg_id(before_block, i + (is_while ? 0 : 1));
    body_args[i] = loom_block_arg_id(body_block, i + (is_while ? 0 : 1));
  }
  loom_op_t* condition = nullptr;
  if (is_while) {
    loom_builder_t condition_builder = rewriter.builder;
    loom_builder_enter_region(&condition_builder, loop, before);
    loom_builder_set_block(&condition_builder, before_block);
    std::vector<loom_value_id_t> reversed(carried.rbegin(), carried.rend());
    IREE_ASSERT_OK(loom_scf_condition_build(
        &condition_builder, loom_test_attrs_result(selector), reversed.data(),
        count, LOOM_LOCATION_UNKNOWN, &condition));
    expect_summary_deferred();
  }
  loom_builder_t body_builder = rewriter.builder;
  loom_builder_enter_region(&body_builder, loop, body);
  loom_builder_set_block(&body_builder, body_block);
  std::vector<loom_value_id_t> yielded(count);
  for (uint16_t i = 0; i < count; ++i) {
    yielded[i] = i + 1 == count ? injected_id
                                : body_args[is_while ? count - 2 - i : i + 1];
  }
  loom_op_t* yield = nullptr;
  IREE_ASSERT_OK(loom_scf_yield_build(&body_builder, yielded.data(), count,
                                      LOOM_LOCATION_UNKNOWN, &yield));
  loom_op_t* use = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&rewriter.builder, loom_op_results(loop),
                                     count, LOOM_LOCATION_UNKNOWN, &use));
  enum class State { kOpen, kClosed, kConditionInput, kUnknownInput };
  int64_t injected_value = count + 8;
  auto check = [&](State state) {
    SCOPED_TRACE(static_cast<int>(state));
    while (loom_op_t* op = loom_rewriter_pop(&rewriter)) {
      bool folded = false;
      IREE_ASSERT_OK(loom_rewriter_try_fold(&rewriter, op, &folded));
    }
    loom_pass_value_fact_owner_t fresh_owner;
    loom_pass_value_fact_owner_initialize(&block_pool_, &fresh_owner);
    loom_value_fact_table_t* fresh = nullptr;
    IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
        &fresh_owner, module_, loom_pass_value_fact_scope_function(function_),
        &fresh));
    for (uint16_t i = 0; i < count; ++i) {
      SCOPED_TRACE(i);
      // Both region projections and the externally visible results must match
      // recomputation, including stronger facts after reopening a rotation.
      for (auto value : {carried[i], body_args[i], loom_op_results(loop)[i]}) {
        EXPECT_TRUE(loom_value_fact_table_facts_equal_for_type(
            module_, i32, facts, loom_value_fact_table_lookup(facts, value),
            fresh, loom_value_fact_table_lookup(fresh, value)));
      }
      const auto value = loom_value_fact_table_lookup(facts, carried[i]);
      if (state != State::kUnknownInput) {
        EXPECT_EQ(value.range_lo, state == State::kClosed ? 0 : i);
        EXPECT_EQ(value.range_hi,
                  state == State::kClosed ? count - 1 : injected_value);
        EXPECT_TRUE(loom_value_facts_is_cluster_uniform(value));
      } else {
        EXPECT_FALSE(loom_value_facts_is_cluster_uniform(value));
      }
      if (state == State::kConditionInput && i == count - 1) {
        const auto output =
            loom_value_fact_table_lookup(facts, loom_op_results(loop)[i]);
        EXPECT_EQ(output.range_lo, injected_value);
        EXPECT_EQ(output.range_hi, injected_value);
      }
    }
    loom_pass_value_fact_owner_deinitialize(&fresh_owner);
  };
  check(State::kOpen);
  for (int edit = 0; edit < 3; ++edit) {
    injected_value += count;
    IREE_ASSERT_OK(loom_rewriter_set_attr(&rewriter, injected, 0,
                                          loom_attr_i64(injected_value)));
    check(State::kOpen);
    IREE_ASSERT_OK(loom_rewriter_set_operand(
        &rewriter, yield, count - 1, body_args[is_while ? count - 1 : 0]));
    check(State::kClosed);
    if (is_while) {
      // Only the condition changes: transport through the before/after join
      // must break and restore the closed rotation without a yield edit.
      IREE_ASSERT_OK(
          loom_rewriter_set_operand(&rewriter, condition, count, injected_id));
      check(State::kConditionInput);
      IREE_ASSERT_OK(
          loom_rewriter_set_operand(&rewriter, condition, count, carried[0]));
      check(State::kClosed);
    }
    IREE_ASSERT_OK(
        loom_rewriter_set_operand(&rewriter, yield, count - 1, injected_id));
    check(State::kOpen);
    IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter, yield, count - 1,
                                             unknown_input_id));
    check(State::kUnknownInput);
    IREE_ASSERT_OK(
        loom_rewriter_set_operand(&rewriter, yield, count - 1, injected_id));
    check(State::kOpen);
  }
  loom_rewriter_deinitialize(&rewriter);
  loom_pass_value_fact_owner_deinitialize(&owner);
  iree_arena_deinitialize(&arena);
}

INSTANTIATE_TEST_SUITE_P(IndependentQueueDepth,
                         StructuredForwardingFactsRewriteTest,
                         ::testing::Combine(::testing::Values(1, 2, 17, 65),
                                            ::testing::Bool()));

TEST_F(GreedyRewriteTest,
       AttributeOnlyUsersAreScheduledOnFactChangeAndReplacement) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* input_op = nullptr;
  loom_op_t* bound_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0),
                                          index_type, LOOM_LOCATION_UNKNOWN,
                                          &input_op));
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(8),
                                          index_type, LOOM_LOCATION_UNKNOWN,
                                          &bound_op));
  loom_value_id_t input = loom_test_constant_result(input_op);
  loom_value_id_t bound = loom_test_constant_result(bound_op);
  loom_predicate_t predicate = {LOOM_PREDICATE_LT,
                                2,
                                {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
                                {},
                                {input, bound}};
  loom_op_t* predicate_user = nullptr;
  IREE_ASSERT_OK(loom_test_assume_build(&builder_, &input, 1, &predicate, 1,
                                        &index_type, 1, LOOM_LOCATION_UNKNOWN,
                                        &predicate_user));

  loom_type_t shape =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_INDEX,
                          loom_dim_pack_dynamic(bound), 0);
  loom_type_id_t shape_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_type_id(module_, shape, &shape_id));
  loom_string_id_t shape_key = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("shape"), &shape_key));
  const loom_named_attr_t attribute = {shape_key, {}, loom_attr_type(shape_id)};
  loom_op_t* type_user = nullptr;
  IREE_ASSERT_OK(
      loom_test_attrs_build(&builder_, LOOM_TEST_ATTRS_BUILD_FLAG_HAS_DICT,
                            input, loom_make_named_attr_slice(&attribute, 1),
                            index_type, LOOM_LOCATION_UNKNOWN, &type_user));
  ASSERT_EQ(loom_module_value(module_, bound)->use_count, 0u);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_pass_value_fact_owner_t owner;
  loom_pass_value_fact_owner_initialize(&block_pool_, &owner);
  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner, module_, loom_pass_value_fact_scope_function(function_), &facts));
  loom_rewriter_t rewriter;
  IREE_ASSERT_OK(loom_rewriter_initialize(&rewriter, module_, &arena));
  loom_rewriter_attach_value_facts(&rewriter, facts);
  auto expect_users = [&]() {
    bool saw_predicate = false;
    bool saw_type = false;
    while (loom_op_t* op = loom_rewriter_pop(&rewriter)) {
      saw_predicate |= op == predicate_user;
      saw_type |= op == type_user;
    }
    EXPECT_TRUE(saw_predicate);
    EXPECT_TRUE(saw_type);
  };
  IREE_ASSERT_OK(loom_rewriter_set_attr(&rewriter, bound_op,
                                        loom_test_constant_value_ATTR_INDEX,
                                        loom_attr_i64(16)));
  expect_users();
  IREE_ASSERT_OK(loom_rewriter_replace_all_uses_with(&rewriter, bound, input));
  expect_users();

  loom_rewriter_deinitialize(&rewriter);
  loom_pass_value_fact_owner_deinitialize(&owner);
  iree_arena_deinitialize(&arena);
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
