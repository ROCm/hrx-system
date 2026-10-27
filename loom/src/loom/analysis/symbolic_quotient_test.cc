// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/condition_facts.h"
#include "loom/analysis/symbolic_expr_proof.h"
#include "loom/analysis/symbolic_expr_test_fixture.h"
#include "loom/ir/attribute.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"

namespace loom {
namespace {

TEST_F(SymbolicExprTest, ProvesIndexRemainderIsBelowDynamicDivisor) {
  loom_value_id_t dividend = DefineIndexValue();
  loom_value_id_t divisor = DefineIndexValue();
  DefineFacts(dividend, loom_value_facts_make(0, 1024, 1));
  DefineFacts(divisor, loom_value_facts_make(1, 512, 1));
  loom_op_t* remainder_op = nullptr;
  IREE_ASSERT_OK(loom_index_rem_build(&builder_, dividend, divisor,
                                      LOOM_LOCATION_UNKNOWN, &remainder_op));
  loom_value_id_t remainder = loom_index_rem_result(remainder_op);

  IREE_ASSERT_OK(loom_op_erase(module_, remainder_op));
  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  loom_symbolic_expr_context_reset(&expression_context_);

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LT, remainder,
      divisor, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);

  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_GE, remainder,
      divisor, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_FALSE);

  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_GT, divisor,
      remainder, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
}

TEST_F(SymbolicExprTest, ProvesAssumedIndexRemainderIsBelowDynamicDivisor) {
  loom_value_id_t dividend = DefineIndexValue();
  loom_value_id_t divisor = DefineIndexValue();
  DefineFacts(dividend, loom_value_facts_make(0, 1024, 1));
  DefineFacts(divisor, loom_value_facts_make(1, 512, 1));
  loom_op_t* remainder_op = nullptr;
  IREE_ASSERT_OK(loom_index_rem_build(&builder_, dividend, divisor,
                                      LOOM_LOCATION_UNKNOWN, &remainder_op));
  loom_value_id_t remainder = loom_index_rem_result(remainder_op);

  loom_predicate_t predicate = {
      .kind = LOOM_PREDICATE_GE,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
      .reserved = {},
      .args = {remainder, 0},
  };
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(&builder_, &remainder, 1, &predicate,
                                         1, &index_type, 1,
                                         LOOM_LOCATION_UNKNOWN, &assume_op));
  loom_value_id_t assumed_remainder =
      loom_index_assume_results(assume_op).values[0];

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LT,
      assumed_remainder, divisor, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
}

TEST_F(SymbolicExprTest, RemainderBoundRequiresUnsignedInputFacts) {
  loom_value_id_t dividend = DefineIndexValue();
  loom_value_id_t divisor = DefineIndexValue();
  DefineFacts(dividend, loom_value_facts_make(-8, 1024, 1));
  DefineFacts(divisor, loom_value_facts_make(1, 512, 1));
  loom_op_t* remainder_op = nullptr;
  IREE_ASSERT_OK(loom_index_rem_build(&builder_, dividend, divisor,
                                      LOOM_LOCATION_UNKNOWN, &remainder_op));
  loom_value_id_t remainder = loom_index_rem_result(remainder_op);

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LT, remainder,
      divisor, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_UNKNOWN);
}

TEST_F(SymbolicExprTest, ProvesScalarUnsignedRemainderIsBelowDivisor) {
  loom_value_id_t dividend = DefineI64Value();
  loom_value_id_t divisor = DefineI64Value();
  DefineFacts(dividend, loom_value_facts_make(0, 1024, 1));
  DefineFacts(divisor, loom_value_facts_make(1, 512, 1));
  loom_op_t* remainder_op = nullptr;
  IREE_ASSERT_OK(loom_scalar_remui_build(&builder_, dividend, divisor,
                                         loom_type_scalar(LOOM_SCALAR_TYPE_I64),
                                         LOOM_LOCATION_UNKNOWN, &remainder_op));
  loom_value_id_t remainder = loom_scalar_remui_result(remainder_op);

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LE, remainder,
      divisor, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);

  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_EQ, remainder,
      divisor, &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_FALSE);
}

TEST_F(SymbolicExprTest, ProvesQuotientBoundFromActiveProductRelation) {
  const loom_value_id_t dividend = DefineIndexValue();
  const loom_value_id_t divisor = DefineIndexValue();
  const loom_value_id_t bound = DefineIndexValue();
  DefineFacts(dividend, loom_value_facts_make(0, 65535, 1));
  // These ranges deliberately allow multiplication to wrap. A true relation
  // to the materialized residue remains sufficient in the accepted direction.
  DefineFacts(divisor, loom_value_facts_make(1, INT64_MAX, 1));
  DefineFacts(bound, loom_value_facts_make(0, INT64_MAX, 1));

  loom_op_t* quotient_op = nullptr;
  IREE_ASSERT_OK(loom_index_div_build(&builder_, dividend, divisor,
                                      LOOM_LOCATION_UNKNOWN, &quotient_op));
  const loom_value_id_t quotient = loom_index_div_result(quotient_op);
  ComputeFacts(quotient_op);

  for (int orientation = 0; orientation < 2; ++orientation) {
    for (int factor_order = 0; factor_order < 2; ++factor_order) {
      SCOPED_TRACE(orientation);
      SCOPED_TRACE(factor_order);
      loom_op_t* product_op = nullptr;
      IREE_ASSERT_OK(loom_index_mul_build(&builder_,
                                          factor_order == 0 ? divisor : bound,
                                          factor_order == 0 ? bound : divisor,
                                          LOOM_LOCATION_UNKNOWN, &product_op));
      const loom_value_id_t product = loom_index_mul_result(product_op);
      ComputeFacts(product_op);
      loom_condition_integer_relation_t active_relation =
          ValueRelation(orientation == 0 ? LOOM_SYMBOLIC_INTEGER_RELATION_LT
                                         : LOOM_SYMBOLIC_INTEGER_RELATION_GT,
                        orientation == 0 ? dividend : product,
                        orientation == 0 ? product : dividend);
      loom_condition_fact_set_t condition_facts =
          SingleRelationFacts(&active_relation);
      ScopedConditionFacts condition_scope(&expression_context_,
                                           &condition_facts);

      loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
      IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
          &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LT, quotient,
          bound, &proof));
      EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
      IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
          &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_GE, quotient,
          bound, &proof));
      EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_FALSE);
      IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
          &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_GT, bound,
          quotient, &proof));
      EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
    }
  }
}

TEST_F(SymbolicExprTest, ProvesShiftedQuotientBoundFromActiveShiftedExtent) {
  const loom_value_id_t dividend = DefineIndexValue();
  const loom_value_id_t bound = DefineIndexValue();
  const loom_value_id_t shift_two =
      loom_index_constant_result(BuildIndexConstant(2));
  const loom_value_id_t shift_three =
      loom_index_constant_result(BuildIndexConstant(3));
  const loom_value_id_t shift_sixty_four =
      loom_index_constant_result(BuildIndexConstant(64));
  const loom_value_id_t dynamic_shift = DefineIndexValue();
  DefineFacts(dividend, loom_value_facts_make(0, 65535, 1));
  DefineFacts(bound, loom_value_facts_make(0, INT64_MAX, 1));
  DefineFacts(shift_two, loom_value_facts_exact_i64(2));
  DefineFacts(shift_three, loom_value_facts_exact_i64(3));
  DefineFacts(shift_sixty_four, loom_value_facts_exact_i64(64));
  DefineFacts(dynamic_shift, loom_value_facts_make(1, 3, 1));

  const struct {
    loom_value_id_t quotient_shift;
    loom_value_id_t extent_shift;
    loom_symbolic_proof_result_t expected;
  } cases[] = {
      {shift_two, shift_two, LOOM_SYMBOLIC_PROOF_TRUE},
      {shift_two, shift_three, LOOM_SYMBOLIC_PROOF_UNKNOWN},
      {shift_sixty_four, shift_sixty_four, LOOM_SYMBOLIC_PROOF_UNKNOWN},
      {dynamic_shift, dynamic_shift, LOOM_SYMBOLIC_PROOF_TRUE},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.quotient_shift);
    SCOPED_TRACE(test_case.extent_shift);
    loom_op_t* quotient_op = nullptr;
    IREE_ASSERT_OK(loom_index_shrui_build(&builder_, dividend,
                                          test_case.quotient_shift,
                                          LOOM_LOCATION_UNKNOWN, &quotient_op));
    const loom_value_id_t quotient = loom_index_shrui_result(quotient_op);
    ComputeFacts(quotient_op);
    loom_op_t* extent_op = nullptr;
    IREE_ASSERT_OK(loom_index_shli_build(&builder_, bound,
                                         test_case.extent_shift,
                                         LOOM_LOCATION_UNKNOWN, &extent_op));
    const loom_value_id_t extent = loom_index_shli_result(extent_op);
    ComputeFacts(extent_op);

    loom_condition_integer_relation_t active_relation =
        ValueRelation(LOOM_SYMBOLIC_INTEGER_RELATION_LT, dividend, extent);
    loom_condition_fact_set_t condition_facts =
        SingleRelationFacts(&active_relation);
    ScopedConditionFacts condition_scope(&expression_context_,
                                         &condition_facts);

    loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
        &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LT, quotient,
        bound, &proof));
    EXPECT_EQ(proof, test_case.expected);
  }
}

TEST_F(SymbolicExprTest, ProvesScalarQuotientBoundFromActiveProductRelation) {
  const loom_value_id_t dividend = DefineI64Value();
  const loom_value_id_t divisor = DefineI64Value();
  const loom_value_id_t bound = DefineI64Value();
  DefineFacts(dividend, loom_value_facts_make(0, 65535, 1));
  DefineFacts(divisor, loom_value_facts_make(1, 256, 1));
  DefineFacts(bound, loom_value_facts_make(0, 1024, 1));

  const loom_type_t i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  loom_op_t* product_op = nullptr;
  IREE_ASSERT_OK(loom_scalar_muli_build(&builder_, /*instance_flags=*/0,
                                        divisor, bound, i64_type,
                                        LOOM_LOCATION_UNKNOWN, &product_op));
  const loom_value_id_t product = loom_scalar_muli_result(product_op);
  ComputeFacts(product_op);
  loom_op_t* quotient_op = nullptr;
  IREE_ASSERT_OK(loom_scalar_divui_build(&builder_, dividend, divisor, i64_type,
                                         LOOM_LOCATION_UNKNOWN, &quotient_op));
  const loom_value_id_t quotient = loom_scalar_divui_result(quotient_op);
  ComputeFacts(quotient_op);

  loom_condition_integer_relation_t active_relation =
      ValueRelation(LOOM_SYMBOLIC_INTEGER_RELATION_LT, dividend, product);
  loom_condition_fact_set_t condition_facts =
      SingleRelationFacts(&active_relation);
  ScopedConditionFacts condition_scope(&expression_context_, &condition_facts);

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LT, quotient, bound,
      &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
}

TEST_F(SymbolicExprTest,
       QuotientPairStrictOrderCoversUnsignedTypesAndSpellings) {
  enum class QuotientForm { kDivide, kShift };
  const loom_scalar_type_t scalar_types[] = {
      LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_I1,  LOOM_SCALAR_TYPE_I8,
      LOOM_SCALAR_TYPE_I16,   LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I64,
  };
  const QuotientForm forms[] = {QuotientForm::kDivide, QuotientForm::kShift};
  auto define_value = [&](loom_scalar_type_t scalar_type) {
    loom_value_id_t value = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_value(
        &builder_, loom_type_scalar(scalar_type), &value));
    return value;
  };
  auto build_quotient = [&](QuotientForm form, loom_value_id_t dividend,
                            loom_value_id_t scale,
                            loom_scalar_type_t scalar_type) {
    loom_op_t* op = nullptr;
    if (scalar_type == LOOM_SCALAR_TYPE_INDEX) {
      if (form == QuotientForm::kDivide) {
        IREE_CHECK_OK(loom_index_div_build(&builder_, dividend, scale,
                                           LOOM_LOCATION_UNKNOWN, &op));
      } else {
        IREE_CHECK_OK(loom_index_shrui_build(&builder_, dividend, scale,
                                             LOOM_LOCATION_UNKNOWN, &op));
      }
    } else {
      const loom_type_t type = loom_type_scalar(scalar_type);
      if (form == QuotientForm::kDivide) {
        IREE_CHECK_OK(loom_scalar_divui_build(&builder_, dividend, scale, type,
                                              LOOM_LOCATION_UNKNOWN, &op));
      } else {
        IREE_CHECK_OK(loom_scalar_shrui_build(&builder_, dividend, scale, type,
                                              LOOM_LOCATION_UNKNOWN, &op));
      }
    }
    ComputeFacts(op);
    return loom_op_const_results(op)[0];
  };

  for (loom_scalar_type_t scalar_type : scalar_types) {
    const int64_t divisor = scalar_type == LOOM_SCALAR_TYPE_I1 ? 1 : 4;
    const int64_t shift = scalar_type == LOOM_SCALAR_TYPE_I1 ? 0 : 2;
    const int64_t maximum = scalar_type == LOOM_SCALAR_TYPE_I1 ? 1 : 124;
    for (QuotientForm left_form : forms) {
      for (QuotientForm right_form : forms) {
        SCOPED_TRACE(loom_scalar_type_name(scalar_type));
        SCOPED_TRACE(static_cast<int>(left_form));
        SCOPED_TRACE(static_cast<int>(right_form));
        const loom_value_id_t left_dividend = define_value(scalar_type);
        const loom_value_id_t right_dividend = define_value(scalar_type);
        const loom_value_id_t left_scale = define_value(scalar_type);
        const loom_value_id_t right_scale = define_value(scalar_type);
        DefineFacts(left_dividend,
                    loom_value_facts_make(0, maximum, /*known_divisor=*/1));
        DefineFacts(right_dividend, loom_value_facts_make(0, maximum, divisor));
        DefineFacts(left_scale,
                    loom_value_facts_exact_i64(
                        left_form == QuotientForm::kDivide ? divisor : shift));
        DefineFacts(right_scale,
                    loom_value_facts_exact_i64(
                        right_form == QuotientForm::kDivide ? divisor : shift));
        const loom_value_id_t left_quotient =
            build_quotient(left_form, left_dividend, left_scale, scalar_type);
        const loom_value_id_t right_quotient = build_quotient(
            right_form, right_dividend, right_scale, scalar_type);

        loom_condition_integer_relation_t active_relation = ValueRelation(
            LOOM_SYMBOLIC_INTEGER_RELATION_LT, left_dividend, right_dividend);
        loom_condition_fact_set_t condition_facts =
            SingleRelationFacts(&active_relation);
        ScopedConditionFacts condition_scope(&expression_context_,
                                             &condition_facts);
        loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
        IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
            &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LT,
            left_quotient, right_quotient, &proof));
        EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
        IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
            &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_GE,
            left_quotient, right_quotient, &proof));
        EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_FALSE);
      }
    }

    const int64_t bit_count = loom_scalar_type_bitwidth(scalar_type);
    const loom_value_id_t left_dividend = define_value(scalar_type);
    const loom_value_id_t right_dividend = define_value(scalar_type);
    const loom_value_id_t invalid_shift = define_value(scalar_type);
    DefineFacts(left_dividend,
                loom_value_facts_make(0, maximum, /*known_divisor=*/1));
    DefineFacts(right_dividend, loom_value_facts_make(0, maximum, divisor));
    DefineFacts(invalid_shift, loom_value_facts_exact_i64(bit_count));
    const loom_value_id_t left_quotient = build_quotient(
        QuotientForm::kShift, left_dividend, invalid_shift, scalar_type);
    const loom_value_id_t right_quotient = build_quotient(
        QuotientForm::kShift, right_dividend, invalid_shift, scalar_type);
    loom_condition_integer_relation_t active_relation = ValueRelation(
        LOOM_SYMBOLIC_INTEGER_RELATION_LT, left_dividend, right_dividend);
    loom_condition_fact_set_t condition_facts =
        SingleRelationFacts(&active_relation);
    ScopedConditionFacts condition_scope(&expression_context_,
                                         &condition_facts);
    loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
        &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LE, left_quotient,
        right_quotient, &proof));
    EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_UNKNOWN)
        << loom_scalar_type_name(scalar_type);
  }
}

TEST_F(SymbolicExprTest, QuotientPairOrderMatchesRoundingOracle) {
  const loom_symbolic_integer_relation_t relations[] = {
      LOOM_SYMBOLIC_INTEGER_RELATION_EQ, LOOM_SYMBOLIC_INTEGER_RELATION_NE,
      LOOM_SYMBOLIC_INTEGER_RELATION_LT, LOOM_SYMBOLIC_INTEGER_RELATION_LE,
      LOOM_SYMBOLIC_INTEGER_RELATION_GT, LOOM_SYMBOLIC_INTEGER_RELATION_GE,
  };
  const int64_t divisor = 4;
  const loom_value_id_t divisor_value =
      loom_index_constant_result(BuildIndexConstant(divisor));
  DefineFacts(divisor_value, loom_value_facts_exact_i64(divisor));
  for (loom_symbolic_integer_relation_t known_relation : relations) {
    for (unsigned divisibility = 0; divisibility < 4; ++divisibility) {
      SCOPED_TRACE(known_relation);
      SCOPED_TRACE(divisibility);
      const bool left_divisible = (divisibility & 1) != 0;
      const bool right_divisible = (divisibility & 2) != 0;
      const loom_value_id_t left_dividend = DefineIndexValue();
      const loom_value_id_t right_dividend = DefineIndexValue();
      DefineFacts(left_dividend,
                  loom_value_facts_make(0, 31, left_divisible ? divisor : 1));
      DefineFacts(right_dividend,
                  loom_value_facts_make(0, 31, right_divisible ? divisor : 1));
      loom_op_t* left_op = nullptr;
      IREE_ASSERT_OK(loom_index_div_build(&builder_, left_dividend,
                                          divisor_value, LOOM_LOCATION_UNKNOWN,
                                          &left_op));
      ComputeFacts(left_op);
      const loom_value_id_t left_quotient = loom_index_div_result(left_op);
      loom_op_t* right_op = nullptr;
      IREE_ASSERT_OK(loom_index_div_build(&builder_, right_dividend,
                                          divisor_value, LOOM_LOCATION_UNKNOWN,
                                          &right_op));
      ComputeFacts(right_op);
      const loom_value_id_t right_quotient = loom_index_div_result(right_op);

      loom_integer_order_outcome_bits_t expected_outcomes = 0;
      for (int64_t left = 0; left <= 31; ++left) {
        if (left_divisible && left % divisor != 0) {
          continue;
        }
        for (int64_t right = 0; right <= 31; ++right) {
          if ((right_divisible && right % divisor != 0) ||
              !EvaluateIntegerRelation(known_relation, left, right)) {
            continue;
          }
          const int64_t left_result = left / divisor;
          const int64_t right_result = right / divisor;
          const loom_integer_order_outcome_t outcome =
              left_result < right_result ? LOOM_INTEGER_ORDER_OUTCOME_LESS
              : left_result == right_result
                  ? LOOM_INTEGER_ORDER_OUTCOME_EQUAL
                  : LOOM_INTEGER_ORDER_OUTCOME_GREATER;
          expected_outcomes |= 1u << outcome;
        }
      }
      ASSERT_NE(expected_outcomes, 0u);

      loom_condition_integer_relation_t active_relation =
          ValueRelation(known_relation, left_dividend, right_dividend);
      loom_condition_fact_set_t condition_facts =
          SingleRelationFacts(&active_relation);
      ScopedConditionFacts condition_scope(&expression_context_,
                                           &condition_facts);
      for (loom_symbolic_integer_relation_t queried_relation : relations) {
        SCOPED_TRACE(queried_relation);
        bool expected = false;
        const bool resolved = loom_symbolic_integer_relation_resolve(
            expected_outcomes, queried_relation, &expected);
        loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
        IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
            &expression_context_, queried_relation, left_quotient,
            right_quotient, &proof));
        EXPECT_EQ(proof, resolved ? (expected ? LOOM_SYMBOLIC_PROOF_TRUE
                                              : LOOM_SYMBOLIC_PROOF_FALSE)
                                  : LOOM_SYMBOLIC_PROOF_UNKNOWN);
      }
    }
  }
}

TEST_F(SymbolicExprTest, QuotientPairSupportsSharedDynamicScales) {
  const loom_value_id_t left_dividend = DefineIndexValue();
  const loom_value_id_t right_dividend = DefineIndexValue();
  const loom_value_id_t divisor = DefineIndexValue();
  DefineFacts(left_dividend, loom_value_facts_make(0, 255, 1));
  DefineFacts(right_dividend, loom_value_facts_make(0, 255, 1));
  DefineFacts(divisor, loom_value_facts_make(1, 8, 1));
  loom_op_t* left_op = nullptr;
  IREE_ASSERT_OK(loom_index_div_build(&builder_, left_dividend, divisor,
                                      LOOM_LOCATION_UNKNOWN, &left_op));
  ComputeFacts(left_op);
  loom_op_t* right_op = nullptr;
  IREE_ASSERT_OK(loom_index_div_build(&builder_, right_dividend, divisor,
                                      LOOM_LOCATION_UNKNOWN, &right_op));
  ComputeFacts(right_op);
  loom_condition_integer_relation_t active_relation = ValueRelation(
      LOOM_SYMBOLIC_INTEGER_RELATION_LT, left_dividend, right_dividend);
  loom_condition_fact_set_t condition_facts =
      SingleRelationFacts(&active_relation);
  ScopedConditionFacts condition_scope(&expression_context_, &condition_facts);

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LE,
      loom_index_div_result(left_op), loom_index_div_result(right_op), &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LT,
      loom_index_div_result(left_op), loom_index_div_result(right_op), &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_UNKNOWN);
}

TEST_F(SymbolicExprTest, QuotientPairDynamicShiftUsesWorstCaseDivisibility) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  auto define_value = [&]() {
    loom_value_id_t value = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_value(&builder_, type, &value));
    return value;
  };
  const loom_value_id_t left_dividend = define_value();
  const loom_value_id_t right_dividend = define_value();
  const loom_value_id_t shift = define_value();
  DefineFacts(left_dividend, loom_value_facts_make(0, 252, 1));
  DefineFacts(right_dividend, loom_value_facts_make(0, 252, 4));
  DefineFacts(shift, loom_value_facts_make(1, 2, 1));
  loom_op_t* left_op = nullptr;
  IREE_ASSERT_OK(loom_scalar_shrui_build(&builder_, left_dividend, shift, type,
                                         LOOM_LOCATION_UNKNOWN, &left_op));
  ComputeFacts(left_op);
  loom_op_t* right_op = nullptr;
  IREE_ASSERT_OK(loom_scalar_shrui_build(&builder_, right_dividend, shift, type,
                                         LOOM_LOCATION_UNKNOWN, &right_op));
  ComputeFacts(right_op);
  loom_condition_integer_relation_t active_relation = ValueRelation(
      LOOM_SYMBOLIC_INTEGER_RELATION_LT, left_dividend, right_dividend);
  loom_condition_fact_set_t condition_facts =
      SingleRelationFacts(&active_relation);
  ScopedConditionFacts condition_scope(&expression_context_, &condition_facts);

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LT,
      loom_scalar_shrui_result(left_op), loom_scalar_shrui_result(right_op),
      &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
}

TEST_F(SymbolicExprTest, QuotientPairRejectsMismatchedScales) {
  const loom_value_id_t left_dividend = DefineIndexValue();
  const loom_value_id_t right_dividend = DefineIndexValue();
  const loom_value_id_t four =
      loom_index_constant_result(BuildIndexConstant(4));
  const loom_value_id_t eight =
      loom_index_constant_result(BuildIndexConstant(8));
  DefineFacts(left_dividend, loom_value_facts_make(0, 63, 1));
  DefineFacts(right_dividend, loom_value_facts_make(0, 63, 8));
  DefineFacts(four, loom_value_facts_exact_i64(4));
  DefineFacts(eight, loom_value_facts_exact_i64(8));
  loom_op_t* left_op = nullptr;
  IREE_ASSERT_OK(loom_index_div_build(&builder_, left_dividend, four,
                                      LOOM_LOCATION_UNKNOWN, &left_op));
  ComputeFacts(left_op);
  loom_op_t* right_op = nullptr;
  IREE_ASSERT_OK(loom_index_div_build(&builder_, right_dividend, eight,
                                      LOOM_LOCATION_UNKNOWN, &right_op));
  ComputeFacts(right_op);
  loom_condition_integer_relation_t active_relation = ValueRelation(
      LOOM_SYMBOLIC_INTEGER_RELATION_LT, left_dividend, right_dividend);
  loom_condition_fact_set_t condition_facts =
      SingleRelationFacts(&active_relation);
  ScopedConditionFacts condition_scope(&expression_context_, &condition_facts);

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LE,
      loom_index_div_result(left_op), loom_index_div_result(right_op), &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_UNKNOWN);
}

TEST_F(SymbolicExprTest,
       QuotientPairRequiresNonNegativeDividendsAndPositiveDivisor) {
  const loom_value_id_t left_dividend = DefineIndexValue();
  const loom_value_id_t right_dividend = DefineIndexValue();
  const loom_value_id_t divisor = DefineIndexValue();
  loom_op_t* left_op = nullptr;
  IREE_ASSERT_OK(loom_index_div_build(&builder_, left_dividend, divisor,
                                      LOOM_LOCATION_UNKNOWN, &left_op));
  loom_op_t* right_op = nullptr;
  IREE_ASSERT_OK(loom_index_div_build(&builder_, right_dividend, divisor,
                                      LOOM_LOCATION_UNKNOWN, &right_op));
  loom_condition_integer_relation_t active_relation = ValueRelation(
      LOOM_SYMBOLIC_INTEGER_RELATION_LT, left_dividend, right_dividend);
  loom_condition_fact_set_t condition_facts =
      SingleRelationFacts(&active_relation);
  ScopedConditionFacts condition_scope(&expression_context_, &condition_facts);

  const struct {
    loom_value_facts_t left;
    loom_value_facts_t right;
    loom_value_facts_t divisor;
    loom_symbolic_proof_result_t expected;
  } cases[] = {
      {loom_value_facts_make(-1, 63, 1), loom_value_facts_make(0, 64, 8),
       loom_value_facts_exact_i64(8), LOOM_SYMBOLIC_PROOF_UNKNOWN},
      {loom_value_facts_make(0, 63, 1), loom_value_facts_make(-8, 64, 8),
       loom_value_facts_exact_i64(8), LOOM_SYMBOLIC_PROOF_TRUE},
      {loom_value_facts_make(0, 63, 1), loom_value_facts_make(0, 64, 8),
       loom_value_facts_make(0, 8, 1), LOOM_SYMBOLIC_PROOF_UNKNOWN},
  };
  for (const auto& test_case : cases) {
    DefineFacts(left_dividend, test_case.left);
    DefineFacts(right_dividend, test_case.right);
    DefineFacts(divisor, test_case.divisor);
    ComputeFacts(left_op);
    ComputeFacts(right_op);
    loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
        &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LE,
        loom_index_div_result(left_op), loom_index_div_result(right_op),
        &proof));
    EXPECT_EQ(proof, test_case.expected);
  }
}

TEST_F(SymbolicExprTest, ProvesQuotientBoundFromDividendIdentityPredicate) {
  const loom_value_id_t source_dividend = DefineIndexValue();
  const loom_value_id_t divisor = DefineIndexValue();
  const loom_value_id_t bound = DefineIndexValue();
  DefineFacts(source_dividend, loom_value_facts_make(0, 65535, 1));
  DefineFacts(divisor, loom_value_facts_make(1, 256, 1));
  DefineFacts(bound, loom_value_facts_make(0, 1024, 1));

  loom_op_t* product_op = nullptr;
  IREE_ASSERT_OK(loom_index_mul_build(&builder_, divisor, bound,
                                      LOOM_LOCATION_UNKNOWN, &product_op));
  const loom_value_id_t product = loom_index_mul_result(product_op);
  ComputeFacts(product_op);

  loom_predicate_t product_bound_predicate =
      ValuePredicate(LOOM_PREDICATE_LT, source_dividend, product);
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(
      &builder_, &source_dividend, 1, &product_bound_predicate, 1, &index_type,
      1, LOOM_LOCATION_UNKNOWN, &assume_op));
  const loom_value_id_t dividend =
      loom_index_assume_results(assume_op).values[0];
  ComputeFacts(assume_op);

  loom_op_t* quotient_op = nullptr;
  IREE_ASSERT_OK(loom_index_div_build(&builder_, dividend, divisor,
                                      LOOM_LOCATION_UNKNOWN, &quotient_op));
  const loom_value_id_t quotient = loom_index_div_result(quotient_op);
  ComputeFacts(quotient_op);

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
      &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LT, quotient, bound,
      &proof));
  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_TRUE);
}

TEST_F(SymbolicExprTest, QuotientProductRelationRejectsInsufficientProofs) {
  const loom_value_id_t dividend = DefineIndexValue();
  const loom_value_id_t divisor = DefineIndexValue();
  const loom_value_id_t bound = DefineIndexValue();
  const loom_value_id_t other_bound = DefineIndexValue();
  DefineFacts(dividend, loom_value_facts_make(0, 65535, 1));
  DefineFacts(divisor, loom_value_facts_make(1, 256, 1));
  DefineFacts(bound, loom_value_facts_make(0, 1024, 1));
  DefineFacts(other_bound, loom_value_facts_make(0, 1024, 1));

  loom_op_t* product_op = nullptr;
  IREE_ASSERT_OK(loom_index_mul_build(&builder_, divisor, bound,
                                      LOOM_LOCATION_UNKNOWN, &product_op));
  const loom_value_id_t product = loom_index_mul_result(product_op);
  ComputeFacts(product_op);
  loom_op_t* other_product_op = nullptr;
  IREE_ASSERT_OK(loom_index_mul_build(&builder_, divisor, other_bound,
                                      LOOM_LOCATION_UNKNOWN,
                                      &other_product_op));
  const loom_value_id_t other_product = loom_index_mul_result(other_product_op);
  ComputeFacts(other_product_op);
  loom_op_t* quotient_op = nullptr;
  IREE_ASSERT_OK(loom_index_div_build(&builder_, dividend, divisor,
                                      LOOM_LOCATION_UNKNOWN, &quotient_op));
  const loom_value_id_t quotient = loom_index_div_result(quotient_op);
  ComputeFacts(quotient_op);

  const struct {
    loom_symbolic_integer_relation_t active_relation;
    loom_value_id_t active_product;
    loom_symbolic_integer_relation_t queried_relation;
  } cases[] = {
      // Equality with the product still permits quotient == bound.
      {LOOM_SYMBOLIC_INTEGER_RELATION_LE, product,
       LOOM_SYMBOLIC_INTEGER_RELATION_LT},
      // A wrapped product can make this edge weaker than the mathematical
      // product comparison, so it cannot establish the converse quotient.
      {LOOM_SYMBOLIC_INTEGER_RELATION_GE, product,
       LOOM_SYMBOLIC_INTEGER_RELATION_GE},
      // A strict relation to a different product says nothing about bound.
      {LOOM_SYMBOLIC_INTEGER_RELATION_LT, other_product,
       LOOM_SYMBOLIC_INTEGER_RELATION_LT},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.active_relation);
    loom_condition_integer_relation_t active_relation = ValueRelation(
        test_case.active_relation, dividend, test_case.active_product);
    loom_condition_fact_set_t condition_facts =
        SingleRelationFacts(&active_relation);
    ScopedConditionFacts condition_scope(&expression_context_,
                                         &condition_facts);
    loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
        &expression_context_, test_case.queried_relation, quotient, bound,
        &proof));
    EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_UNKNOWN);
  }
}

TEST_F(SymbolicExprTest,
       QuotientProductRelationRequiresUnsignedInputsAndPositiveDivisor) {
  const loom_value_id_t dividend = DefineIndexValue();
  const loom_value_id_t divisor = DefineIndexValue();
  const loom_value_id_t bound = DefineIndexValue();

  loom_op_t* product_op = nullptr;
  IREE_ASSERT_OK(loom_index_mul_build(&builder_, divisor, bound,
                                      LOOM_LOCATION_UNKNOWN, &product_op));
  const loom_value_id_t product = loom_index_mul_result(product_op);
  loom_op_t* quotient_op = nullptr;
  IREE_ASSERT_OK(loom_index_div_build(&builder_, dividend, divisor,
                                      LOOM_LOCATION_UNKNOWN, &quotient_op));
  const loom_value_id_t quotient = loom_index_div_result(quotient_op);

  loom_condition_integer_relation_t active_relation =
      ValueRelation(LOOM_SYMBOLIC_INTEGER_RELATION_LT, dividend, product);
  loom_condition_fact_set_t condition_facts =
      SingleRelationFacts(&active_relation);
  ScopedConditionFacts condition_scope(&expression_context_, &condition_facts);

  const struct {
    loom_value_facts_t dividend_facts;
    loom_value_facts_t divisor_facts;
    loom_value_facts_t bound_facts;
  } cases[] = {
      {loom_value_facts_make(-1, 65535, 1), loom_value_facts_make(1, 256, 1),
       loom_value_facts_make(0, 1024, 1)},
      {loom_value_facts_make(0, 65535, 1), loom_value_facts_make(0, 256, 1),
       loom_value_facts_make(0, 1024, 1)},
      {loom_value_facts_make(0, 65535, 1), loom_value_facts_make(1, 256, 1),
       loom_value_facts_make(-1, 1024, 1)},
  };
  for (const auto& test_case : cases) {
    DefineFacts(dividend, test_case.dividend_facts);
    DefineFacts(divisor, test_case.divisor_facts);
    DefineFacts(bound, test_case.bound_facts);
    ComputeFacts(product_op);
    ComputeFacts(quotient_op);

    loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
        &expression_context_, LOOM_SYMBOLIC_INTEGER_RELATION_LT, quotient,
        bound, &proof));
    EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_UNKNOWN);
  }
}

TEST_F(SymbolicExprTest, RetainedPacketBoundsMatchIntegerOracle) {
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const loom_symbolic_integer_relation_t relations[] = {
      LOOM_SYMBOLIC_INTEGER_RELATION_EQ, LOOM_SYMBOLIC_INTEGER_RELATION_NE,
      LOOM_SYMBOLIC_INTEGER_RELATION_LT, LOOM_SYMBOLIC_INTEGER_RELATION_LE,
      LOOM_SYMBOLIC_INTEGER_RELATION_GT, LOOM_SYMBOLIC_INTEGER_RELATION_GE,
  };
  for (int64_t width : {1, 3, 8, 31}) {
    SCOPED_TRACE(width);
    for (bool rounded : {false, true}) {
      SCOPED_TRACE(rounded);
      const int64_t padding = rounded ? width - 1 : 0;
      for (bool divisible : {false, true}) {
        SCOPED_TRACE(divisible);
        for (auto relation : relations) {
          SCOPED_TRACE(relation);
          for (bool reversed : {false, true}) {
            SCOPED_TRACE(reversed);
            const auto extent = DefineIndexValue();
            const auto group = DefineIndexValue();
            const auto lane = DefineIndexValue();
            DefineFacts(extent, loom_value_facts_make(1, 4 * width + 1,
                                                      divisible ? width : 1));
            DefineFacts(group, loom_value_facts_make(0, 7, 1));
            DefineFacts(lane, loom_value_facts_make(0, width - 1, 1));
            loom_op_t* width_op = BuildIndexConstant(width);
            ComputeFacts(width_op);
            loom_op_t* padding_op = BuildIndexConstant(padding);
            ComputeFacts(padding_op);
            loom_op_t* padded_op = nullptr;
            IREE_ASSERT_OK(loom_index_add_build(
                &builder_, extent, loom_index_constant_result(padding_op),
                index_type, LOOM_LOCATION_UNKNOWN, &padded_op));
            ComputeFacts(padded_op);
            loom_op_t* quotient_op = nullptr;
            IREE_ASSERT_OK(loom_index_div_build(
                &builder_, loom_index_add_result(padded_op),
                loom_index_constant_result(width_op), LOOM_LOCATION_UNKNOWN,
                &quotient_op));
            ComputeFacts(quotient_op);
            const auto quotient = loom_index_div_result(quotient_op);
            const auto predicate = ValuePredicate(
                loom_symbolic_integer_relation_to_predicate_kind(relation),
                reversed ? quotient : group, reversed ? group : quotient);
            loom_op_t* assume_op = nullptr;
            IREE_ASSERT_OK(loom_index_assume_build(
                &builder_, &group, 1, &predicate, 1, &index_type, 1,
                LOOM_LOCATION_UNKNOWN, &assume_op));
            ComputeFacts(assume_op);
            const auto bounded_group =
                loom_index_assume_results(assume_op).values[0];

            bool always_inside = true;
            bool always_outside = true;
            int selected_count = 0;
            for (int64_t size = 1; size <= 4 * width + 1; ++size) {
              if (divisible && size % width != 0) {
                continue;
              }
              const int64_t count = (size + padding) / width;
              for (int64_t block = 0; block <= 7; ++block) {
                if (!EvaluateIntegerRelation(relation, reversed ? count : block,
                                             reversed ? block : count)) {
                  continue;
                }
                for (int64_t element = 0; element < width; ++element) {
                  const bool inside = block * width + element < size;
                  always_inside &= inside;
                  always_outside &= !inside;
                  ++selected_count;
                }
              }
            }
            ASSERT_GT(selected_count, 0);
            const bool complete_packet =
                ((!reversed && relation == LOOM_SYMBOLIC_INTEGER_RELATION_LT) ||
                 (reversed && relation == LOOM_SYMBOLIC_INTEGER_RELATION_GT)) &&
                (!rounded || divisible);
            for (bool scoped : {false, true}) {
              SCOPED_TRACE(scoped);
              loom_op_t* origin_op = nullptr;
              IREE_ASSERT_OK(loom_index_mul_build(
                  &builder_, scoped ? bounded_group : group,
                  loom_index_constant_result(width_op), LOOM_LOCATION_UNKNOWN,
                  &origin_op));
              ComputeFacts(origin_op);
              loom_op_t* position_op = nullptr;
              IREE_ASSERT_OK(loom_index_add_build(
                  &builder_, loom_index_mul_result(origin_op), lane, index_type,
                  LOOM_LOCATION_UNKNOWN, &position_op));
              ComputeFacts(position_op);
              for (bool complement : {false, true}) {
                SCOPED_TRACE(complement);
                loom_symbolic_proof_result_t proof =
                    LOOM_SYMBOLIC_PROOF_UNKNOWN;
                IREE_ASSERT_OK(loom_symbolic_expr_prove_value_relation(
                    &expression_context_,
                    complement ? LOOM_SYMBOLIC_INTEGER_RELATION_GE
                               : LOOM_SYMBOLIC_INTEGER_RELATION_LT,
                    loom_index_add_result(position_op), extent, &proof));
                if (!scoped) {
                  EXPECT_EQ(proof, LOOM_SYMBOLIC_PROOF_UNKNOWN);
                } else {
                  if (proof != LOOM_SYMBOLIC_PROOF_UNKNOWN) {
                    EXPECT_TRUE((proof == LOOM_SYMBOLIC_PROOF_TRUE) !=
                                        complement
                                    ? always_inside
                                    : always_outside);
                  }
                  if (complete_packet) {
                    EXPECT_EQ(proof, complement ? LOOM_SYMBOLIC_PROOF_FALSE
                                                : LOOM_SYMBOLIC_PROOF_TRUE);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace
}  // namespace loom
