// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/view/offset_expression.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbolic_expr_test_fixture.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"

namespace loom {
namespace {

TEST_F(SymbolicExprTest, ReusesCompleteCompoundRelation) {
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const loom_type_t offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  const loom_value_id_t left = DefineIndexValue();
  const loom_value_id_t right = DefineIndexValue();

  loom_op_t* sum_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, left, right, index_type,
                                      LOOM_LOCATION_UNKNOWN, &sum_op));
  ComputeFacts(sum_op);
  loom_op_t* one_op = BuildIndexConstant(1);
  ComputeFacts(one_op);
  loom_op_t* next_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, loom_index_add_result(sum_op),
                                      loom_index_constant_result(one_op),
                                      index_type, LOOM_LOCATION_UNKNOWN,
                                      &next_op));
  ComputeFacts(next_op);

  const loom_value_id_t next = loom_index_add_result(next_op);
  const loom_predicate_t next_range = {
      .kind = LOOM_PREDICATE_RANGE,
      .arg_count = 3,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                   LOOM_PRED_ARG_CONST},
      .reserved = {},
      .args = {next, 0, 181},
  };
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(&builder_, &next, 1, &next_range, 1,
                                         &index_type, 1, LOOM_LOCATION_UNKNOWN,
                                         &assume_op));
  ComputeFacts(assume_op);
  const loom_value_id_t bounded =
      loom_index_assume_results(assume_op).values[0];

  loom_op_t* stride_op = nullptr;
  IREE_ASSERT_OK(loom_index_constant_build(&builder_, loom_attr_i64(16384),
                                           offset_type, LOOM_LOCATION_UNKNOWN,
                                           &stride_op));
  ComputeFacts(stride_op);
  loom_op_t* scaled_op = nullptr;
  IREE_ASSERT_OK(loom_index_scale_build(
      &builder_, bounded, loom_index_constant_result(stride_op), offset_type,
      LOOM_LOCATION_UNKNOWN, &scaled_op));
  ComputeFacts(scaled_op);

  loom_op_t* base_op = nullptr;
  IREE_ASSERT_OK(loom_index_constant_build(&builder_, loom_attr_i64(65536),
                                           offset_type, LOOM_LOCATION_UNKNOWN,
                                           &base_op));
  ComputeFacts(base_op);
  loom_op_t* address_op = nullptr;
  IREE_ASSERT_OK(
      loom_index_add_build(&builder_, loom_index_constant_result(base_op),
                           loom_index_scale_result(scaled_op), offset_type,
                           LOOM_LOCATION_UNKNOWN, &address_op));
  ComputeFacts(address_op);
  const loom_value_id_t address = loom_index_add_result(address_op);

  loom_symbolic_expr_t bounded_expression = {};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, bounded,
                                               &bounded_expression));
  ASSERT_TRUE(loom_symbolic_expr_is_linear(&bounded_expression));
  ASSERT_EQ(bounded_expression.term_count, 2u);
  EXPECT_EQ(bounded_expression.constant, 1);
  EXPECT_EQ(bounded_expression.facts.range_lo, 0);
  EXPECT_EQ(bounded_expression.facts.range_hi, 181);
  ASSERT_TRUE(loom_symbolic_expr_has_relation_value(&bounded_expression));
  EXPECT_EQ(bounded_expression.relation_value_id, bounded);

  loom_value_id_t materialized = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_view_materialize_offset_expression(
      &builder_, &bounded_expression, LOOM_VALUE_ID_INVALID, address,
      &materialized));
  const loom_op_t* cast_op =
      loom_value_def_op(loom_module_value(module_, materialized));
  ASSERT_TRUE(loom_index_cast_isa(cast_op));
  EXPECT_EQ(loom_index_cast_input(cast_op), bounded);

  loom_symbolic_expr_t address_expression = {};
  IREE_ASSERT_OK(loom_symbolic_expr_from_value(&expression_context_, address,
                                               &address_expression));
  ASSERT_TRUE(loom_symbolic_expr_is_linear(&address_expression));
  ASSERT_EQ(address_expression.term_count, 2u);
  EXPECT_EQ(address_expression.constant, 81920);
  EXPECT_EQ(address_expression.facts.range_lo, 65536);
  EXPECT_EQ(address_expression.facts.range_hi, 3031040);
  ASSERT_TRUE(loom_symbolic_expr_has_relation_value(&address_expression));
  EXPECT_EQ(address_expression.relation_value_id, address);

  const iree_host_size_t value_count = module_->values.count;
  IREE_ASSERT_OK(loom_view_materialize_offset_expression(
      &builder_, &address_expression, LOOM_VALUE_ID_INVALID, address,
      &materialized));
  EXPECT_EQ(materialized, address);
  EXPECT_EQ(module_->values.count, value_count);
}

}  // namespace
}  // namespace loom
