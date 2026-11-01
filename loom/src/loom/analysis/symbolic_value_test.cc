// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_value.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbolic_expr_test_fixture.h"
#include "loom/ir/float_facts.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scf/ops.h"

namespace loom {
namespace {

class SymbolicValueTest : public SymbolicExprTest {
 protected:
  loom_value_id_t DefineValue(loom_scalar_type_t scalar_type,
                              loom_value_facts_t facts) {
    loom_value_id_t value = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_value(
        &builder_, loom_type_scalar(scalar_type), &value));
    DefineFacts(value, facts);
    return value;
  }

  loom_value_id_t Select(loom_value_id_t condition, loom_value_id_t true_value,
                         loom_value_id_t false_value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(
        loom_scf_select_build(&builder_, condition, true_value, false_value,
                              loom_module_value_type(module_, true_value),
                              LOOM_LOCATION_UNKNOWN, &op));
    ComputeFacts(op);
    return loom_scf_select_result(op);
  }

  bool HasConditionMemo(loom_value_id_t value) const {
    // This fixture uses module IDs directly, without a compact local domain.
    for (iree_host_size_t i = 0;
         i < expression_context_.touched_condition_fact_memo_ordinal_count;
         ++i) {
      if (expression_context_.touched_condition_fact_memo_ordinals[i] ==
          value) {
        return true;
      }
    }
    return false;
  }
};

TEST_F(SymbolicValueTest, ProvenScalarSelectInfersOnlyTheChosenArm) {
  const loom_value_id_t selector = DefineIndexValue();
  DefineFacts(selector, loom_value_facts_make(0, 1, 1));
  const loom_value_id_t zero =
      loom_index_constant_result(BuildIndexConstant(0));
  DefineFacts(zero, loom_value_facts_exact_i64(0));
  loom_op_t* compare = nullptr;
  IREE_ASSERT_OK(loom_index_cmp_build(&builder_, LOOM_INDEX_CMP_PREDICATE_EQ,
                                      selector, zero, LOOM_LOCATION_UNKNOWN,
                                      &compare));
  ComputeFacts(compare);
  const loom_value_id_t condition = loom_index_cmp_result(compare);

  for (loom_scalar_type_t type = LOOM_SCALAR_TYPE_INDEX;
       type < LOOM_SCALAR_TYPE_COUNT_; ++type) {
    SCOPED_TRACE(loom_scalar_type_name(type));
    const bool floating = loom_scalar_type_is_float(type);
    const loom_value_id_t arms[] = {
        DefineValue(type, floating ? loom_value_facts_exact_float(type, 1.0)
                                   : loom_value_facts_exact_i64(1)),
        DefineValue(type, floating ? loom_value_facts_exact_float(type, 0.0)
                                   : loom_value_facts_exact_i64(0)),
    };
    const loom_value_id_t selected = Select(condition, arms[0], arms[1]);
    for (int64_t selector_value : {0, 1}) {
      SCOPED_TRACE(selector_value);
      auto relation = LiteralRelation(LOOM_SYMBOLIC_INTEGER_RELATION_EQ,
                                      selector, selector_value);
      auto facts = SingleRelationFacts(&relation);
      {
        ScopedConditionFacts scope(&expression_context_, &facts);
        loom_value_facts_t result;
        IREE_ASSERT_OK(loom_symbolic_value_lookup_condition_refined_facts(
            &expression_context_, selected, &result));
        if (floating) {
          double exact = 0.0;
          ASSERT_TRUE(loom_value_facts_as_exact_float(type, result, &exact));
          EXPECT_EQ(exact, selector_value == 0 ? 1.0 : 0.0);
        } else {
          int64_t exact = 0;
          ASSERT_TRUE(loom_value_facts_as_exact_i64(result, &exact));
          EXPECT_EQ(exact, selector_value == 0 ? 1 : 0);
        }
        EXPECT_TRUE(HasConditionMemo(arms[selector_value]));
        EXPECT_FALSE(HasConditionMemo(arms[1 - selector_value]));
        EXPECT_FALSE(HasConditionMemo(condition));
      }
      EXPECT_EQ(expression_context_.touched_condition_fact_memo_ordinal_count,
                0);
    }
  }
}

TEST_F(SymbolicValueTest, UnprovenSelectInfersBothArms) {
  const loom_value_id_t condition = DefineI1Value();
  DefineFacts(condition, loom_value_facts_make(0, 1, 1));
  const loom_value_id_t left =
      DefineValue(LOOM_SCALAR_TYPE_INDEX, loom_value_facts_exact_i64(3));
  const loom_value_id_t right =
      DefineValue(LOOM_SCALAR_TYPE_INDEX, loom_value_facts_exact_i64(7));
  const loom_value_id_t selected = Select(condition, left, right);
  const loom_condition_fact_set_t conditions = {};
  ScopedConditionFacts scope(&expression_context_, &conditions);
  loom_value_facts_t result;
  IREE_ASSERT_OK(loom_symbolic_value_lookup_condition_refined_facts(
      &expression_context_, selected, &result));
  EXPECT_EQ(result.range_lo, 3);
  EXPECT_EQ(result.range_hi, 7);
  EXPECT_TRUE(HasConditionMemo(left));
  EXPECT_TRUE(HasConditionMemo(right));
  EXPECT_TRUE(HasConditionMemo(condition));
}

TEST_F(SymbolicValueTest, NestedSelectionInfersChosenProducerUnderItsScope) {
  const loom_value_id_t source = DefineIndexValue();
  DefineFacts(source, loom_value_facts_make(0, 99, 1));
  const loom_value_id_t limit =
      loom_index_constant_result(BuildIndexConstant(16));
  DefineFacts(limit, loom_value_facts_exact_i64(16));
  loom_op_t* compare = nullptr;
  IREE_ASSERT_OK(loom_index_cmp_build(&builder_, LOOM_INDEX_CMP_PREDICATE_ULT,
                                      source, limit, LOOM_LOCATION_UNKNOWN,
                                      &compare));
  ComputeFacts(compare);
  const loom_value_id_t condition = loom_index_cmp_result(compare);
  const loom_value_id_t offset =
      loom_index_constant_result(BuildIndexConstant(2));
  DefineFacts(offset, loom_value_facts_exact_i64(2));
  loom_op_t* add = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, source, offset,
                                      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                      LOOM_LOCATION_UNKNOWN, &add));
  ComputeFacts(add);
  const loom_value_id_t chosen = loom_index_add_result(add);
  const loom_value_id_t inactive =
      DefineValue(LOOM_SCALAR_TYPE_INDEX, loom_value_facts_exact_i64(100));
  const loom_value_id_t inner = Select(condition, chosen, inactive);
  const loom_value_id_t outer = Select(condition, inner, inactive);

  auto relation =
      LiteralRelation(LOOM_SYMBOLIC_INTEGER_RELATION_LT, source, 16);
  auto conditions = SingleRelationFacts(&relation);
  {
    ScopedConditionFacts scope(&expression_context_, &conditions);
    loom_value_facts_t result;
    IREE_ASSERT_OK(loom_symbolic_value_lookup_condition_refined_facts(
        &expression_context_, outer, &result));
    EXPECT_EQ(result.range_lo, 2);
    EXPECT_EQ(result.range_hi, 17);
    EXPECT_TRUE(HasConditionMemo(source));
    EXPECT_TRUE(HasConditionMemo(chosen));
    EXPECT_FALSE(HasConditionMemo(inactive));
    EXPECT_FALSE(HasConditionMemo(condition));
  }
  const loom_condition_fact_set_t unconstrained = {};
  ScopedConditionFacts scope(&expression_context_, &unconstrained);
  loom_value_facts_t result;
  IREE_ASSERT_OK(loom_symbolic_value_lookup_condition_refined_facts(
      &expression_context_, outer, &result));
  EXPECT_EQ(result.range_lo, 2);
  EXPECT_EQ(result.range_hi, 101);
  EXPECT_TRUE(HasConditionMemo(inactive));
}

}  // namespace
}  // namespace loom
