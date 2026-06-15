// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/value_facts.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ops/test/ops.h"
#include "loom/pass/test/harness.h"

namespace loom {
namespace {

class PassValueFactsTest : public PassTestHarness {
 protected:
  loom_value_id_t FirstConstantResult(loom_func_like_t function) {
    loom_region_t* body = loom_func_like_body(function);
    if (!body) {
      ADD_FAILURE() << "function body required";
      return LOOM_VALUE_ID_INVALID;
    }
    const loom_op_t* op = loom_block_const_op(loom_region_entry_block(body), 0);
    if (!op) {
      ADD_FAILURE() << "first op required";
      return LOOM_VALUE_ID_INVALID;
    }
    if (!loom_test_constant_isa(op)) {
      ADD_FAILURE() << "first op must be test.constant";
      return LOOM_VALUE_ID_INVALID;
    }
    return loom_test_constant_result(op);
  }

  loom_value_id_t FirstConstantResult(loom_region_t* region) {
    if (!region) {
      ADD_FAILURE() << "region required";
      return LOOM_VALUE_ID_INVALID;
    }
    const loom_op_t* op =
        loom_block_const_op(loom_region_entry_block(region), 0);
    if (!op) {
      ADD_FAILURE() << "first op required";
      return LOOM_VALUE_ID_INVALID;
    }
    if (!loom_test_constant_isa(op)) {
      ADD_FAILURE() << "first op must be test.constant";
      return LOOM_VALUE_ID_INVALID;
    }
    return loom_test_constant_result(op);
  }
};

TEST_F(PassValueFactsTest, FunctionScopeComputesAndReusesCurrentFunction) {
  loom_module_t* module =
      Parse(IREE_SV("test.func @first() {\n"
                    "  %value = test.constant 42 : i32\n"
                    "  test.yield\n"
                    "}\n"
                    "test.func @second() {\n"
                    "  %value = test.constant 7 : i32\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);
  loom_func_like_t first = Function(module, 0);
  loom_func_like_t second = Function(module, 1);
  loom_value_id_t first_value = FirstConstantResult(first);
  loom_value_id_t second_value = FirstConstantResult(second);
  ASSERT_NE(first_value, LOOM_VALUE_ID_INVALID);
  ASSERT_NE(second_value, LOOM_VALUE_ID_INVALID);

  loom_pass_value_fact_owner_t owner = {};
  loom_pass_value_fact_owner_initialize(block_pool(), &owner);

  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner, module, loom_pass_value_fact_scope_function(first), &facts));
  ASSERT_NE(facts, nullptr);
  EXPECT_EQ(loom_value_fact_table_lookup(facts, first_value).range_lo, 42);
  EXPECT_TRUE(loom_value_facts_is_unknown(
      loom_value_fact_table_lookup(facts, second_value)));
  iree_host_size_t touched_count = facts->touched_count;

  loom_value_fact_table_t* reused_facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner, module, loom_pass_value_fact_scope_function(first),
      &reused_facts));
  EXPECT_EQ(reused_facts, facts);
  EXPECT_EQ(reused_facts->touched_count, touched_count);

  loom_value_fact_table_t* second_facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner, module, loom_pass_value_fact_scope_function(second),
      &second_facts));
  EXPECT_EQ(second_facts, facts);
  EXPECT_TRUE(loom_value_facts_is_unknown(
      loom_value_fact_table_lookup(second_facts, first_value)));
  EXPECT_EQ(loom_value_fact_table_lookup(second_facts, second_value).range_lo,
            7);

  loom_pass_value_fact_owner_deinitialize(&owner);
}

TEST_F(PassValueFactsTest, ModuleScopeComputesAllFunctionsExplicitly) {
  loom_module_t* module =
      Parse(IREE_SV("test.func @first() {\n"
                    "  %value = test.constant 42 : i32\n"
                    "  test.yield\n"
                    "}\n"
                    "test.func @second() {\n"
                    "  %value = test.constant 7 : i32\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);
  loom_value_id_t first_value = FirstConstantResult(Function(module, 0));
  loom_value_id_t second_value = FirstConstantResult(Function(module, 1));
  ASSERT_NE(first_value, LOOM_VALUE_ID_INVALID);
  ASSERT_NE(second_value, LOOM_VALUE_ID_INVALID);

  loom_pass_value_fact_owner_t owner = {};
  loom_pass_value_fact_owner_initialize(block_pool(), &owner);

  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner, module, loom_pass_value_fact_scope_module(), &facts));
  ASSERT_NE(facts, nullptr);
  EXPECT_EQ(loom_value_fact_table_lookup(facts, first_value).range_lo, 42);
  EXPECT_EQ(loom_value_fact_table_lookup(facts, second_value).range_lo, 7);

  loom_pass_value_fact_owner_deinitialize(&owner);
}

TEST_F(PassValueFactsTest, RegionScopeComputesRequestedProjection) {
  loom_module_t* module =
      Parse(IREE_SV("test.split_func @projected(%arg: i32) {\n"
                    "  %config = test.constant 7 : i32\n"
                    "  test.yield\n"
                    "} launch {\n"
                    "  %body = test.constant 42 : i32\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);
  loom_func_like_t function = Function(module, 0);
  loom_region_t* body = loom_func_like_body(function);
  loom_region_t* config = loom_test_split_func_config(function.op);
  loom_value_id_t body_value = FirstConstantResult(body);
  loom_value_id_t config_value = FirstConstantResult(config);
  ASSERT_NE(body_value, LOOM_VALUE_ID_INVALID);
  ASSERT_NE(config_value, LOOM_VALUE_ID_INVALID);

  loom_pass_value_fact_owner_t owner = {};
  loom_pass_value_fact_owner_initialize(block_pool(), &owner);

  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner, module,
      loom_pass_value_fact_scope_region(function, config, function.op),
      &facts));
  ASSERT_NE(facts, nullptr);
  EXPECT_TRUE(loom_value_facts_is_unknown(
      loom_value_fact_table_lookup(facts, body_value)));
  EXPECT_EQ(loom_value_fact_table_lookup(facts, config_value).range_lo, 7);
  iree_host_size_t touched_count = facts->touched_count;

  loom_value_fact_table_t* reused_facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner, module,
      loom_pass_value_fact_scope_region(function, config, function.op),
      &reused_facts));
  EXPECT_EQ(reused_facts, facts);
  EXPECT_EQ(reused_facts->touched_count, touched_count);

  loom_value_fact_table_t* body_facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner, module, loom_pass_value_fact_scope_function(function),
      &body_facts));
  EXPECT_EQ(body_facts, facts);
  EXPECT_EQ(loom_value_fact_table_lookup(body_facts, body_value).range_lo, 42);
  EXPECT_EQ(loom_value_fact_table_lookup(body_facts, config_value).range_lo, 7);

  loom_pass_value_fact_owner_deinitialize(&owner);
}

TEST_F(PassValueFactsTest, InvalidateClearsActiveScopeWithoutLosingStorage) {
  loom_module_t* module =
      Parse(IREE_SV("test.func @main() {\n"
                    "  %value = test.constant 99 : i32\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);
  loom_func_like_t function = Function(module, 0);
  loom_value_id_t value = FirstConstantResult(function);
  ASSERT_NE(value, LOOM_VALUE_ID_INVALID);

  loom_pass_value_fact_owner_t owner = {};
  loom_pass_value_fact_owner_initialize(block_pool(), &owner);

  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner, module, loom_pass_value_fact_scope_function(function), &facts));
  ASSERT_NE(facts, nullptr);
  loom_value_facts_t* entries = owner.table.entries;
  EXPECT_EQ(loom_value_fact_table_lookup(facts, value).range_lo, 99);

  loom_pass_value_fact_owner_invalidate(&owner);

  EXPECT_EQ(owner.table.entries, entries);
  EXPECT_TRUE(loom_value_facts_is_unknown(
      loom_value_fact_table_lookup(&owner.table, value)));
  EXPECT_EQ(owner.table.touched_count, 0u);

  loom_pass_value_fact_owner_deinitialize(&owner);
}

}  // namespace
}  // namespace loom
