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
#include "loom/ops/func/ops.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class FuncFactsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_func_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(
        loom_context_register_dialect(&context_, LOOM_DIALECT_FUNC, vtables,
                                      static_cast<uint16_t>(vtable_count)));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("facts"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_op_t* BuildCall(loom_value_id_t operand) {
    const loom_type_t result_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    loom_op_t* call = nullptr;
    IREE_CHECK_OK(loom_func_call_build(&builder_, 0, 0, 0, 0, {0, 0}, &operand,
                                       1, &result_type, 1, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &call));
    return call;
  }

  static iree_status_t ResolveCallResults(
      void* user_data, loom_fact_context_t* context,
      const loom_module_t* module, const loom_op_t* op,
      const loom_value_facts_t* operand_facts,
      loom_value_facts_t* result_facts) {
    auto* invocation_count = static_cast<unsigned*>(user_data);
    ++*invocation_count;
    EXPECT_NE(context->table, nullptr);
    EXPECT_EQ(op->operand_count, 1u);
    EXPECT_EQ(op->result_count, 1u);
    EXPECT_EQ(operand_facts[0].range_lo, 7);
    EXPECT_EQ(operand_facts[0].range_hi, 7);
    result_facts[0] = loom_value_facts_exact_i64(11);
    return iree_ok_status();
  }

  iree_arena_block_pool_t pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(FuncFactsTest, AnalysisContextResolvesCallResults) {
  loom_value_id_t operand = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_value(
      &builder_, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &operand));
  loom_op_t* call = BuildCall(operand);

  loom_value_fact_table_t table = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &analysis_arena_,
                                                  module_->values.count));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, operand,
                                              loom_value_facts_exact_i64(7)));
  unsigned invocation_count = 0;
  table.context.resolve_call_results = {
      .user_data = &invocation_count,
      .fn = ResolveCallResults,
  };

  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table, module_, call));

  EXPECT_EQ(invocation_count, 1u);
  const loom_value_facts_t result =
      loom_value_fact_table_lookup(&table, loom_op_const_results(call)[0]);
  EXPECT_TRUE(loom_value_facts_is_exact(result));
  EXPECT_EQ(result.range_lo, 11);
  EXPECT_EQ(result.range_hi, 11);
}

TEST_F(FuncFactsTest, GenericContextRetainsTypeDomain) {
  loom_value_id_t operand = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_value(
      &builder_, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &operand));
  loom_op_t* call = BuildCall(operand);

  loom_value_fact_table_t table = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &analysis_arena_,
                                                  module_->values.count));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table, module_, call));

  const loom_value_facts_t result =
      loom_value_fact_table_lookup(&table, loom_op_const_results(call)[0]);
  EXPECT_FALSE(loom_value_facts_is_exact(result));
  EXPECT_EQ(result.range_lo, INT64_MIN);
  EXPECT_EQ(result.range_hi, INT64_MAX);
}

TEST_F(FuncFactsTest, GenericContextRetainsSeededSummary) {
  loom_value_id_t operand = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_value(
      &builder_, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &operand));
  loom_op_t* call = BuildCall(operand);
  const loom_value_id_t result_id = loom_op_const_results(call)[0];

  loom_value_fact_table_t source = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &analysis_arena_,
                                                  module_->values.count));
  IREE_ASSERT_OK(loom_value_fact_table_define(&source, result_id,
                                              loom_value_facts_exact_i64(13)));
  loom_value_fact_table_t table = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &analysis_arena_,
                                                  module_->values.count));
  IREE_ASSERT_OK(loom_value_fact_table_seed_values(
      &table, {&source, &result_id, 1}, module_));

  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table, module_, call));

  const loom_value_facts_t result =
      loom_value_fact_table_lookup(&table, result_id);
  EXPECT_TRUE(loom_value_facts_is_exact(result));
  EXPECT_EQ(result.range_lo, 13);
  EXPECT_EQ(result.range_hi, 13);
}

}  // namespace
}  // namespace loom
