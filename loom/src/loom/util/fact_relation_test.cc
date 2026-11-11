// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_relation.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/util/fact_snapshot.h"
#include "loom/util/fact_table.h"
#include "loom/util/predicate_facts.h"

namespace loom {
namespace {

struct VisitedRelation {
  // Canonical value constrained by the scoped alias.
  loom_value_id_t counterpart;
  // Possible signed comparison outcomes for that alias.
  loom_integer_order_outcome_bits_t outcomes;
};

static bool CollectRelation(void* user_data, loom_value_id_t counterpart,
                            loom_integer_order_outcome_bits_t outcomes) {
  static_cast<std::vector<VisitedRelation>*>(user_data)->push_back(
      {counterpart, outcomes});
  return true;
}

class FactRelationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_index_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_INDEX, vtables, static_cast<uint16_t>(count)));
    vtables = loom_scalar_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SCALAR, vtables, static_cast<uint16_t>(count)));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("relations"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&table_, &arena_, 4));
    for (loom_value_id_t& input : inputs_) {
      IREE_ASSERT_OK(loom_builder_define_value(&builder_, type_, &input));
      IREE_ASSERT_OK(loom_value_fact_table_define(&table_, input,
                                                  loom_value_facts_unknown()));
    }
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  iree_status_t BuildIdentity(loom_value_id_t input, loom_op_t** out_op) {
    return loom_index_assume_build(&builder_, &input, 1, nullptr, 0, &type_, 1,
                                   LOOM_LOCATION_UNKNOWN, out_op);
  }

  iree_status_t BuildAssume(const std::vector<loom_value_id_t>& values,
                            const std::vector<loom_predicate_t>& predicates,
                            loom_op_t** out_op) {
    std::vector<loom_type_t> result_types(values.size(), type_);
    return loom_index_assume_build(&builder_, values.data(), values.size(),
                                   predicates.data(), predicates.size(),
                                   result_types.data(), result_types.size(),
                                   LOOM_LOCATION_UNKNOWN, out_op);
  }

  static loom_predicate_t RelationPredicate(loom_predicate_kind_t kind,
                                            loom_value_id_t lhs,
                                            loom_value_id_t rhs) {
    return {
        .kind = kind,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE,
                     LOOM_PRED_ARG_NONE},
        .reserved = {},
        .args = {lhs, rhs, 0},
    };
  }

  static loom_predicate_t LiteralPredicate(loom_predicate_kind_t kind,
                                           loom_value_id_t value,
                                           int64_t literal) {
    return {
        .kind = kind,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                     LOOM_PRED_ARG_NONE},
        .reserved = {},
        .args = {value, literal, 0},
    };
  }

  loom_value_id_t BuildConstant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(&builder_, loom_attr_i64(value),
                                            type_, LOOM_LOCATION_UNKNOWN, &op));
    IREE_CHECK_OK(loom_value_fact_table_compute_op(&table_, module_, op));
    return loom_index_constant_result(op);
  }

  loom_value_id_t DefineValue(loom_type_t type) {
    loom_value_id_t value = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_value(&builder_, type, &value));
    IREE_CHECK_OK(loom_value_fact_table_define(&table_, value,
                                               loom_value_facts_unknown()));
    return value;
  }

  loom_value_fact_auxiliary_snapshot_t Snapshot(loom_value_id_t value) {
    loom_value_fact_auxiliary_snapshot_t snapshot;
    loom_value_fact_table_snapshot_auxiliary(&table_, value, &snapshot);
    return snapshot;
  }

  template <typename Build>
  loom_value_id_t Convert(Build build, loom_value_id_t input,
                          loom_scalar_type_t result_type) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(
        build(&builder_, input, loom_module_value_type(module_, input),
              loom_type_scalar(result_type), LOOM_LOCATION_UNKNOWN, &op));
    IREE_CHECK_OK(loom_value_fact_table_compute_op(&table_, module_, op));
    return loom_op_results(op)[0];
  }

  loom_value_id_t ScalarLessThan(loom_value_id_t input, loom_value_id_t bound) {
    const auto predicate = RelationPredicate(LOOM_PREDICATE_LT, input, bound);
    const auto type = loom_module_value_type(module_, input);
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_scalar_assume_build(&builder_, &input, 1, &predicate, 1,
                                           &type, 1, LOOM_LOCATION_UNKNOWN,
                                           &op));
    IREE_CHECK_OK(loom_value_fact_table_compute_op(&table_, module_, op));
    return loom_op_results(op)[0];
  }

  iree_arena_block_pool_t pool_;
  iree_arena_allocator_t arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
  loom_value_fact_table_t table_;
  const loom_type_t type_ = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t inputs_[2];
};

TEST_F(FactRelationTest, RelationFreeIdentityAllocatesNoRelationState) {
  loom_op_t* identity = nullptr;
  IREE_ASSERT_OK(BuildIdentity(inputs_[0], &identity));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, identity));
  EXPECT_EQ(table_.integer_relations, nullptr);
}

TEST_F(FactRelationTest, DisjointUpperBoundIntervalsAllocateNoRelationState) {
  const auto value_facts = loom_value_facts_make(0, 4, 1);
  const auto bytes = arena_.used_allocation_size;
  for (const auto relation :
       {LOOM_SYMBOLIC_INTEGER_RELATION_LT, LOOM_SYMBOLIC_INTEGER_RELATION_LE}) {
    SCOPED_TRACE(relation);
    const auto bound_facts = loom_value_facts_exact_i64(
        relation == LOOM_SYMBOLIC_INTEGER_RELATION_LT ? 5 : 4);
    bool changed = false;
    IREE_ASSERT_OK(loom_value_fact_integer_relations_define_upper_bounds(
        &table_, inputs_[0], &inputs_[1], nullptr, &bound_facts, 1, relation,
        &value_facts, &changed));
    EXPECT_FALSE(changed);
    EXPECT_EQ(table_.integer_relations, nullptr);
    EXPECT_EQ(arena_.used_allocation_size, bytes);
  }
}

TEST_F(FactRelationTest, ExactUpperBoundRetainsDynamicEndpointRelations) {
  const auto value = DefineValue(type_);
  const auto lower = BuildConstant(0);
  const auto bound = BuildConstant(5);
  const auto value_facts = loom_value_facts_make(0, 4, 1);
  const auto bound_facts = loom_value_facts_exact_i64(5);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, value, value_facts));
  IREE_ASSERT_OK(loom_value_fact_integer_relations_define_counted_range(
      &table_, bound, lower, inputs_[1]));
  bool changed = false;
  IREE_ASSERT_OK(loom_value_fact_integer_relations_define_upper_bounds(
      &table_, value, &bound, nullptr, &bound_facts, 1,
      LOOM_SYMBOLIC_INTEGER_RELATION_LT, &value_facts, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, value,
                                                             inputs_[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
  const auto before = Snapshot(value);
  const auto bytes = arena_.used_allocation_size;
  changed = false;
  IREE_ASSERT_OK(loom_value_fact_integer_relations_define_upper_bounds(
      &table_, value, &bound, nullptr, &bound_facts, 1,
      LOOM_SYMBOLIC_INTEGER_RELATION_LT, &value_facts, &changed));
  EXPECT_FALSE(changed);
  EXPECT_EQ(arena_.used_allocation_size, bytes);
  const auto repeated = Snapshot(value);
  EXPECT_TRUE(loom_value_fact_auxiliary_snapshot_equal(&before, &repeated));

  IREE_ASSERT_OK(loom_value_fact_integer_relations_define_upper_bounds(
      &table_, value, nullptr, nullptr, nullptr, 0,
      LOOM_SYMBOLIC_INTEGER_RELATION_LT, &value_facts, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, value,
                                                             inputs_[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
}

TEST_F(FactRelationTest, ScopedEquivalentBoundsPreserveBothEndpoints) {
  const auto result = DefineValue(type_);
  const auto original = DefineValue(type_);
  const auto equivalent = DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I32));
  const auto facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_fact_integer_relations_define_upper_bounds(
      &table_, original, &inputs_[0], nullptr, &facts, 1,
      LOOM_SYMBOLIC_INTEGER_RELATION_LE, &facts, nullptr));
  IREE_ASSERT_OK(loom_value_fact_integer_relations_define_upper_bounds(
      &table_, equivalent, &inputs_[1], nullptr, &facts, 1,
      LOOM_SYMBOLIC_INTEGER_RELATION_LE, &facts, nullptr));
  bool changed = false;
  IREE_ASSERT_OK(loom_value_fact_integer_relations_define_upper_bounds(
      &table_, result, &original, &equivalent, &facts, 1,
      LOOM_SYMBOLIC_INTEGER_RELATION_LT, &facts, &changed));
  EXPECT_TRUE(changed);
  for (const auto bound : {original, equivalent, inputs_[0], inputs_[1]}) {
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, result,
                                                               bound),
              LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
  }
  // Equality holds only where the result was defined. Earlier values retain
  // their independent identities and relations everywhere else.
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, original,
                                                             equivalent),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
  const auto bytes = arena_.used_allocation_size;
  changed = false;
  IREE_ASSERT_OK(loom_value_fact_integer_relations_define_upper_bounds(
      &table_, result, &original, &equivalent, &facts, 1,
      LOOM_SYMBOLIC_INTEGER_RELATION_LT, &facts, &changed));
  EXPECT_FALSE(changed);
  EXPECT_EQ(arena_.used_allocation_size, bytes);

  IREE_ASSERT_OK(loom_value_fact_integer_relations_define_upper_bounds(
      &table_, result, &original, nullptr, &facts, 1,
      LOOM_SYMBOLIC_INTEGER_RELATION_LT, &facts, &changed));
  EXPECT_TRUE(changed);
  for (const auto bound : {original, inputs_[0]}) {
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, result,
                                                               bound),
              LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
  }
  for (const auto bound : {equivalent, inputs_[1]}) {
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, result,
                                                               bound),
              LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
  }
}

TEST_F(FactRelationTest, RemainderPublicationHonorsQueryScopeAndWithdrawal) {
  for (const auto scalar : {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_I32}) {
    SCOPED_TRACE(loom_scalar_type_name(scalar));
    const auto type = loom_type_scalar(scalar);
    const auto dividend = DefineValue(type);
    const auto divisor = DefineValue(type);
    const auto dividend_facts = loom_value_facts_make(0, 4095, 1);
    const auto unrestricted_divisor = loom_value_facts_make(0, 127, 1);
    const auto positive_divisor = loom_value_facts_make(1, 127, 1);
    IREE_ASSERT_OK(
        loom_value_fact_table_define(&table_, dividend, dividend_facts));
    IREE_ASSERT_OK(
        loom_value_fact_table_define(&table_, divisor, unrestricted_divisor));
    loom_op_t* op = nullptr;
    if (scalar == LOOM_SCALAR_TYPE_INDEX) {
      IREE_ASSERT_OK(loom_index_divrem_build(&builder_, dividend, divisor,
                                             LOOM_LOCATION_UNKNOWN, &op));
    } else {
      const loom_type_t types[] = {type, type};
      IREE_ASSERT_OK(loom_scalar_divremui_build(
          &builder_, dividend, divisor, types, LOOM_LOCATION_UNKNOWN, &op));
    }
    IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, op));
    const auto quotient = loom_op_const_results(op)[0];
    const auto remainder = loom_op_const_results(op)[1];
    const auto original = Snapshot(remainder);

    // Temporary condition evaluation may return narrower numeric facts, but
    // cannot change the owning solve's retained relation or quotient role.
    auto query = table_.context;
    ASSERT_EQ(query.publication, nullptr);
    const loom_value_facts_t operands[] = {dividend_facts, positive_divisor};
    loom_value_facts_t results[2];
    IREE_ASSERT_OK(loom_op_vtable(module_, op)
                       ->infer_facts(&query, module_, op, operands, results));
    EXPECT_EQ(results[1].range_hi, 126);
    const auto queried = Snapshot(remainder);
    EXPECT_TRUE(loom_value_fact_auxiliary_snapshot_equal(&original, &queried));
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                  &table_, remainder, divisor),
              LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);

    IREE_ASSERT_OK(
        loom_value_fact_table_define(&table_, divisor, positive_divisor));
    bool changed = false;
    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                               op, &changed));
    EXPECT_TRUE(changed);
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                  &table_, remainder, divisor),
              LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                  &table_, quotient, divisor),
              LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
    const auto bytes = arena_.used_allocation_size;
    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                               op, &changed));
    EXPECT_FALSE(changed);
    EXPECT_EQ(arena_.used_allocation_size, bytes);

    IREE_ASSERT_OK(
        loom_value_fact_table_define(&table_, divisor, unrestricted_divisor));
    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                               op, &changed));
    EXPECT_TRUE(changed);
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                  &table_, remainder, divisor),
              LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                               op, &changed));
    EXPECT_FALSE(changed);

    // A logical index additionally needs its dividend domain. Scalar unsigned
    // remainder instead interprets the entire represented input bit pattern.
    IREE_ASSERT_OK(
        loom_value_fact_table_define(&table_, divisor, positive_divisor));
    IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, op));
    IREE_ASSERT_OK(loom_value_fact_table_define(
        &table_, dividend, loom_value_facts_make(-4095, -1, 1)));
    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                               op, &changed));
    EXPECT_TRUE(changed);
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                  &table_, remainder, divisor),
              scalar == LOOM_SCALAR_TYPE_INDEX
                  ? LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL
                  : LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
  }
}

TEST_F(FactRelationTest, CountedBoundsComposeEveryEndpointOrder) {
  const auto other_endpoint = DefineValue(type_);
  const auto induction = DefineValue(type_);
  for (int relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
       relation <= LOOM_SYMBOLIC_INTEGER_RELATION_GE; ++relation) {
    const auto order = static_cast<loom_symbolic_integer_relation_t>(relation);
    const auto predicate = RelationPredicate(
        loom_symbolic_integer_relation_to_predicate_kind(order), inputs_[0],
        inputs_[1]);
    loom_op_t* assume = nullptr;
    IREE_ASSERT_OK(BuildAssume({inputs_[0]}, {predicate}, &assume));
    IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
    const auto endpoint = loom_op_results(assume)[0];
    for (bool upper : {false, true}) {
      SCOPED_TRACE(relation);
      SCOPED_TRACE(upper);
      IREE_ASSERT_OK(loom_value_fact_integer_relations_define_counted_range(
          &table_, induction, upper ? other_endpoint : endpoint,
          upper ? endpoint : other_endpoint));
      const auto range_order = upper
                                   ? LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS
                                   : LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL |
                                         LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER;
      EXPECT_EQ(
          loom_value_fact_table_query_signed_integer_order(&table_, induction,
                                                           inputs_[1]),
          loom_integer_order_outcomes_compose(
              range_order, loom_symbolic_integer_relation_outcomes(order)));
    }
  }
}

TEST_F(FactRelationTest, InclusiveUpperBoundsComposeEveryEndpointOrder) {
  const auto result = DefineValue(type_);
  const auto unknown = loom_value_facts_unknown();
  for (int relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
       relation <= LOOM_SYMBOLIC_INTEGER_RELATION_GE; ++relation) {
    SCOPED_TRACE(relation);
    const auto order = static_cast<loom_symbolic_integer_relation_t>(relation);
    loom_op_t* assume = nullptr;
    IREE_ASSERT_OK(
        BuildAssume({inputs_[0]},
                    {RelationPredicate(
                        loom_symbolic_integer_relation_to_predicate_kind(order),
                        inputs_[0], inputs_[1])},
                    &assume));
    IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
    const auto endpoint = loom_op_results(assume)[0];
    bool changed = false;
    IREE_ASSERT_OK(loom_value_fact_integer_relations_define_upper_bounds(
        &table_, result, &endpoint, nullptr, &unknown, 1,
        LOOM_SYMBOLIC_INTEGER_RELATION_LE, &unknown, &changed));
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, result,
                                                               inputs_[1]),
              loom_integer_order_outcomes_compose(
                  LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS |
                      LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL,
                  loom_symbolic_integer_relation_outcomes(order)));
  }
}

TEST_F(FactRelationTest, MaskPublicationHonorsQueryScopeAndWithdrawal) {
  for (const auto scalar :
       {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_I16,
        LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I64}) {
    SCOPED_TRACE(loom_scalar_type_name(scalar));
    const auto type = loom_type_scalar(scalar);
    const auto left = DefineValue(type);
    const auto right = DefineValue(type);
    const auto unrestricted = loom_value_facts_make(-127, 127, 1);
    const auto nonnegative = loom_value_facts_make(0, 127, 1);
    IREE_ASSERT_OK(loom_value_fact_table_define(&table_, left, unrestricted));
    IREE_ASSERT_OK(loom_value_fact_table_define(&table_, right, unrestricted));
    loom_op_t* op = nullptr;
    if (scalar == LOOM_SCALAR_TYPE_INDEX) {
      IREE_ASSERT_OK(loom_index_andi_build(&builder_, left, right,
                                           LOOM_LOCATION_UNKNOWN, &op));
    } else {
      IREE_ASSERT_OK(loom_scalar_andi_build(&builder_, left, right, type,
                                            LOOM_LOCATION_UNKNOWN, &op));
    }
    IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, op));
    const auto result = loom_op_results(op)[0];
    const auto original = Snapshot(result);

    // Narrowing a query's input does not publish its branch-local relation.
    auto query = table_.context;
    ASSERT_EQ(query.publication, nullptr);
    const loom_value_facts_t operands[] = {nonnegative, unrestricted};
    loom_value_facts_t result_facts;
    IREE_ASSERT_OK(
        loom_op_vtable(module_, op)
            ->infer_facts(&query, module_, op, operands, &result_facts));
    EXPECT_EQ(result_facts.range_lo, 0);
    const auto queried = Snapshot(result);
    EXPECT_TRUE(loom_value_fact_auxiliary_snapshot_equal(&original, &queried));

    IREE_ASSERT_OK(loom_value_fact_table_define(&table_, left, nonnegative));
    bool changed = false;
    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                               op, &changed));
    EXPECT_TRUE(changed);
    EXPECT_EQ(
        loom_value_fact_table_query_signed_integer_order(&table_, result, left),
        LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS |
            LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL);
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, result,
                                                               right),
              LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
    const auto bytes = arena_.used_allocation_size;
    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                               op, &changed));
    EXPECT_FALSE(changed);
    EXPECT_EQ(arena_.used_allocation_size, bytes);

    IREE_ASSERT_OK(loom_value_fact_table_define(&table_, right, nonnegative));
    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                               op, &changed));
    EXPECT_TRUE(changed);
    for (const auto operand : {left, right}) {
      EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                    &table_, result, operand),
                LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS |
                    LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL);
    }

    IREE_ASSERT_OK(loom_value_fact_table_define(&table_, left, unrestricted));
    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                               op, &changed));
    EXPECT_TRUE(changed);
    EXPECT_EQ(
        loom_value_fact_table_query_signed_integer_order(&table_, result, left),
        LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, result,
                                                               right),
              LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS |
                  LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL);
    IREE_ASSERT_OK(loom_value_fact_table_define(&table_, right, unrestricted));
    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                               op, &changed));
    EXPECT_TRUE(changed);
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, result,
                                                               right),
              LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
  }
}

TEST_F(FactRelationTest, InclusiveBoundsShareTransitiveStorage) {
  const auto facts = loom_value_facts_unknown();
  loom_value_id_t endpoint = inputs_[0];
  iree_host_size_t midpoint_bytes = 0;
  for (int i = 0; i < 1024; ++i) {
    const auto result = DefineValue(type_);
    IREE_ASSERT_OK(loom_value_fact_integer_relations_define_upper_bounds(
        &table_, result, &endpoint, nullptr, &facts, 1,
        LOOM_SYMBOLIC_INTEGER_RELATION_LE, &facts, nullptr));
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, result,
                                                               inputs_[0]),
              LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS |
                  LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL);
    endpoint = result;
    if (i == 511) {
      midpoint_bytes = arena_.used_allocation_size;
    }
  }
  EXPECT_LT(arena_.used_allocation_size, midpoint_bytes * 3);
}

TEST_F(FactRelationTest, ContradictoryEndpointsDoNotProveTransitiveOrder) {
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(BuildAssume(
      {inputs_[0]},
      {RelationPredicate(LOOM_PREDICATE_LT, inputs_[0], inputs_[1]),
       RelationPredicate(LOOM_PREDICATE_GE, inputs_[0], inputs_[1])},
      &assume));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
  const auto endpoint = loom_op_results(assume)[0];
  const auto other_endpoint = DefineValue(type_);
  const auto induction = DefineValue(type_);
  for (bool upper : {false, true}) {
    IREE_ASSERT_OK(loom_value_fact_integer_relations_define_counted_range(
        &table_, induction, upper ? other_endpoint : endpoint,
        upper ? endpoint : other_endpoint));
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                  &table_, induction, inputs_[1]),
              LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
  }
  const auto facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_fact_integer_relations_define_upper_bounds(
      &table_, induction, &endpoint, nullptr, &facts, 1,
      LOOM_SYMBOLIC_INTEGER_RELATION_LE, &facts, nullptr));
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, induction,
                                                             inputs_[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
  // Reinference replaces the entire relation publication, including the
  // contradictory entry. An ordinary upper bound must become usable again.
  IREE_ASSERT_OK(loom_value_fact_integer_relations_define_counted_range(
      &table_, induction, other_endpoint, inputs_[1]));
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, induction,
                                                             inputs_[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
}

TEST_F(FactRelationTest, BoundCompositionKeepsCounterpartOwnedOrder) {
  const auto zero = BuildConstant(0);
  const auto child = DefineValue(type_);
  IREE_ASSERT_OK(loom_value_fact_integer_relations_define_counted_range(
      &table_, child, zero, inputs_[0]));
  // The child publishes child < limit. A weaker predicate on the limit must
  // retain that established ordering before another counted range consumes it.
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(BuildAssume(
      {inputs_[0]}, {RelationPredicate(LOOM_PREDICATE_NE, inputs_[0], child)},
      &assume));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
  const auto limit = loom_op_results(assume)[0];
  const auto induction = DefineValue(type_);
  IREE_ASSERT_OK(loom_value_fact_integer_relations_define_counted_range(
      &table_, induction, limit, inputs_[1]));
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, induction,
                                                             child),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER);
}

TEST_F(FactRelationTest, NestedCountedBoundsShareTransitiveStorage) {
  const auto lower = BuildConstant(0);
  const auto upper = inputs_[0];
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &table_, upper, loom_value_facts_make(1, INT64_MAX, 1)));
  loom_value_id_t endpoint = upper;
  iree_host_size_t midpoint_bytes = 0;
  for (int i = 0; i < 1024; ++i) {
    const auto induction = DefineValue(type_);
    IREE_ASSERT_OK(loom_value_fact_table_define(
        &table_, induction, loom_value_facts_make(0, INT64_MAX - i - 1, 1)));
    IREE_ASSERT_OK(loom_value_fact_integer_relations_define_counted_range(
        &table_, induction, lower, endpoint));
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                  &table_, induction, upper),
              LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
    endpoint = induction;
    if (i == 511) {
      midpoint_bytes = arena_.used_allocation_size;
    }
  }
  std::vector<VisitedRelation> relations;
  loom_value_fact_table_for_each_signed_integer_relation_while(
      &table_, endpoint, CollectRelation, &relations);
  EXPECT_EQ(relations.size(), 1025u);
  // Doubling a chain may grow radix depth and slab capacity, but must not
  // retain the quadratic intermediate sets from rebuilding every ancestor.
  EXPECT_LT(arena_.used_allocation_size, midpoint_bytes * 3);
}

TEST_F(FactRelationTest, IntegerConversionFamilyKeepsOnlyNumericIdentities) {
  const loom_scalar_type_t types[] = {
      LOOM_SCALAR_TYPE_I1,  LOOM_SCALAR_TYPE_I8,  LOOM_SCALAR_TYPE_I16,
      LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I64,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(types); ++i) {
    int64_t source_lo = 0;
    int64_t source_hi = 0;
    ASSERT_TRUE(
        loom_value_facts_scalar_type_domain(types[i], &source_lo, &source_hi));
    const auto input = DefineValue(loom_type_scalar(types[i]));
    IREE_ASSERT_OK(loom_value_fact_table_define(
        &table_, input, loom_value_facts_make(source_lo, source_hi, 1)));
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(types); ++j) {
      if (i == j) {
        continue;
      }
      SCOPED_TRACE(loom_scalar_type_name(types[i]));
      SCOPED_TRACE(loom_scalar_type_name(types[j]));
      if (i < j) {
        const auto signed_result =
            Convert(loom_scalar_extsi_build, input, types[j]);
        const auto unsigned_result =
            Convert(loom_scalar_extui_build, input, types[j]);
        EXPECT_EQ(loom_value_fact_table_query_numeric_identity(&table_,
                                                               signed_result),
                  types[i] == LOOM_SCALAR_TYPE_I1 ? signed_result : input);
        EXPECT_EQ(loom_value_fact_table_query_numeric_identity(&table_,
                                                               unsigned_result),
                  types[i] == LOOM_SCALAR_TYPE_I1 ? input : unsigned_result);
        // Mathematical equality never authorizes replacing a typed SSA use.
        EXPECT_EQ(loom_value_fact_table_query_identity(&table_, signed_result),
                  signed_result);
        EXPECT_EQ(
            loom_value_fact_table_query_identity(&table_, unsigned_result),
            unsigned_result);
      } else {
        const auto result = Convert(loom_scalar_trunci_build, input, types[j]);
        EXPECT_EQ(loom_value_fact_table_query_numeric_identity(&table_, result),
                  result);
      }
    }
  }
  // Relation-free casts do not allocate the relation index.
  EXPECT_EQ(table_.integer_relations, nullptr);

  for (auto source : types) {
    for (auto destination : types) {
      const auto source_width = loom_scalar_type_bitwidth(source);
      const auto destination_width = loom_scalar_type_bitwidth(destination);
      if (source_width <= destination_width) {
        continue;
      }
      SCOPED_TRACE(loom_scalar_type_name(source));
      SCOPED_TRACE(loom_scalar_type_name(destination));
      const auto input = DefineValue(loom_type_scalar(source));
      const int64_t maximum = destination == LOOM_SCALAR_TYPE_I1 ? 1 : 63;
      IREE_ASSERT_OK(loom_value_fact_table_define(
          &table_, input, loom_value_facts_make(0, maximum, 1)));
      const auto narrow = Convert(loom_scalar_trunci_build, input, destination);
      const auto restored = Convert(loom_scalar_extui_build, narrow, source);
      EXPECT_EQ(loom_value_fact_table_query_numeric_identity(&table_, narrow),
                input);
      EXPECT_EQ(loom_value_fact_table_query_numeric_identity(&table_, restored),
                input);
      EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, input,
                                                                 restored),
                LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL);
      if (destination != LOOM_SCALAR_TYPE_I1) {
        const auto negative = DefineValue(loom_type_scalar(source));
        IREE_ASSERT_OK(loom_value_fact_table_define(
            &table_, negative, loom_value_facts_make(-63, -1, 1)));
        const auto signed_narrow =
            Convert(loom_scalar_trunci_build, negative, destination);
        const auto signed_restored =
            Convert(loom_scalar_extsi_build, signed_narrow, source);
        EXPECT_EQ(loom_value_fact_table_query_numeric_identity(&table_,
                                                               signed_restored),
                  negative);
      }
    }
  }
}

TEST_F(FactRelationTest, SignedPredicateExtensionPreservesOnlyZero) {
  for (int64_t value : {0, 1}) {
    const auto input = DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1));
    IREE_ASSERT_OK(loom_value_fact_table_define(
        &table_, input, loom_value_facts_exact_i64(value)));
    const auto result =
        Convert(loom_scalar_extsi_build, input, LOOM_SCALAR_TYPE_I32);
    EXPECT_EQ(loom_value_fact_table_lookup(&table_, result).range_lo, -value);
    EXPECT_EQ(loom_value_fact_table_query_numeric_identity(&table_, result),
              value == 0 ? input : result);
  }
}

TEST_F(FactRelationTest, ConvertedAliasKeepsLiteralExclusionsLocal) {
  const auto input = DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I32));
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &table_, input, loom_value_facts_make(-63, 63, 1)));
  const auto type = loom_module_value_type(module_, input);
  const auto predicate = LiteralPredicate(LOOM_PREDICATE_NE, input, 42);
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_scalar_assume_build(&builder_, &input, 1, &predicate, 1,
                                          &type, 1, LOOM_LOCATION_UNKNOWN,
                                          &op));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, op));
  const auto alias = loom_op_results(op)[0];
  const auto narrow =
      Convert(loom_scalar_trunci_build, alias, LOOM_SCALAR_TYPE_I8);
  const auto restored =
      Convert(loom_scalar_extsi_build, narrow, LOOM_SCALAR_TYPE_I32);
  for (auto value : {alias, narrow, restored}) {
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_literal_order(
                  &table_, value, 42),
              LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS |
                  LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER);
  }
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_literal_order(&table_,
                                                                     input, 42),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
}

TEST_F(FactRelationTest, NumericConversionPreservesAliasLocalRelations) {
  const auto i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const auto input = DefineValue(i32);
  const auto bound = DefineValue(i32);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, input,
                                              loom_value_facts_make(0, 63, 1)));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, bound,
                                              loom_value_facts_make(1, 64, 1)));
  const auto alias = ScalarLessThan(input, bound);
  const auto narrow =
      Convert(loom_scalar_trunci_build, alias, LOOM_SCALAR_TYPE_I8);
  const auto wide =
      Convert(loom_scalar_extui_build, narrow, LOOM_SCALAR_TYPE_I64);
  const auto coordinate =
      Convert(loom_index_cast_build, wide, LOOM_SCALAR_TYPE_INDEX);
  const auto bound_wide =
      Convert(loom_scalar_extsi_build, bound, LOOM_SCALAR_TYPE_I64);
  const auto extent =
      Convert(loom_index_cast_build, bound_wide, LOOM_SCALAR_TYPE_INDEX);
  for (auto value : {alias, narrow, wide, coordinate}) {
    for (auto endpoint : {bound, bound_wide, extent}) {
      EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, value,
                                                                 endpoint),
                LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
      EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                    &table_, endpoint, value),
                LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER);
    }
  }
  EXPECT_EQ(
      loom_value_fact_table_query_signed_integer_order(&table_, input, bound),
      LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
  const auto unrefined =
      Convert(loom_scalar_trunci_build, input, LOOM_SCALAR_TYPE_I8);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, unrefined,
                                                             extent),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);

  const loom_value_id_t selected[] = {coordinate, extent};
  loom_value_fact_table_t clone = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&clone, &arena_, 0));
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &clone, {&table_, selected, IREE_ARRAYSIZE(selected)}, module_));
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&clone, coordinate,
                                                             extent),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
  loom_value_fact_table_clear_scope(&table_);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                &table_, coordinate, extent),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&clone, coordinate,
                                                             extent),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
}

TEST_F(FactRelationTest, ConversionReinferenceWithdrawsOriginsAndRelations) {
  const auto i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const auto input = DefineValue(i32);
  const auto bound = DefineValue(i32);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, input,
                                              loom_value_facts_make(0, 63, 1)));
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &table_, bound, loom_value_facts_make(1, 2048, 1)));
  const auto alias = ScalarLessThan(input, bound);
  const auto narrow =
      Convert(loom_scalar_trunci_build, alias, LOOM_SCALAR_TYPE_I8);
  const auto widen =
      Convert(loom_scalar_extui_build, narrow, LOOM_SCALAR_TYPE_I32);
  EXPECT_EQ(
      loom_value_fact_table_query_signed_integer_order(&table_, widen, bound),
      LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
  const auto original = Snapshot(narrow);

  IREE_ASSERT_OK(loom_value_fact_table_define(
      &table_, input, loom_value_facts_make(0, 1023, 1)));
  for (auto value : {alias, narrow, widen}) {
    auto* op = loom_value_def_op(loom_module_value(module_, value));
    bool changed = false;
    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                               op, &changed));
    EXPECT_TRUE(changed);
    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                               op, &changed));
    EXPECT_FALSE(changed);
  }
  const auto current = Snapshot(narrow);
  EXPECT_FALSE(loom_value_fact_auxiliary_snapshot_equal(&original, &current));
  for (auto value : {narrow, widen}) {
    EXPECT_EQ(loom_value_fact_table_query_numeric_identity(&table_, value),
              value);
    EXPECT_EQ(
        loom_value_fact_table_query_signed_integer_order(&table_, value, bound),
        LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
  }
  EXPECT_EQ(
      loom_value_fact_table_query_signed_integer_order(&table_, alias, bound),
      LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
}

TEST_F(FactRelationTest, IntegerOrderUsesIdentityAndExactFacts) {
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                nullptr, inputs_[0], inputs_[0]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                &table_, inputs_[0], inputs_[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, inputs_[0],
                                              loom_value_facts_exact_i64(4)));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, inputs_[1],
                                              loom_value_facts_exact_i64(9)));
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                &table_, inputs_[0], inputs_[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                &table_, inputs_[1], inputs_[0]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER);
  EXPECT_EQ(table_.integer_relations, nullptr);
}

TEST_F(FactRelationTest, IntegerOrderUsesIntervalsAndNonzeroFacts) {
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &table_, inputs_[0], loom_value_facts_make(-16, -1, 1)));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, inputs_[1],
                                              loom_value_facts_make(0, 16, 1)));
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                &table_, inputs_[0], inputs_[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);

  loom_value_facts_t nonzero = loom_value_facts_unknown();
  nonzero.flags |= LOOM_VALUE_FACT_NON_ZERO;
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, inputs_[0], nonzero));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, inputs_[1],
                                              loom_value_facts_exact_i64(0)));
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                &table_, inputs_[0], inputs_[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS |
                LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER);
  EXPECT_EQ(table_.integer_relations, nullptr);
}

TEST_F(FactRelationTest, IntegerLiteralOrderUsesScalarFacts) {
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_literal_order(
                nullptr, inputs_[0], 42),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, inputs_[0],
                                              loom_value_facts_make(4, 9, 1)));
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_literal_order(
                &table_, inputs_[0], 3),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_literal_order(
                &table_, inputs_[0], 7),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_literal_order(
                &table_, inputs_[0], 10),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
  EXPECT_EQ(table_.integer_relations, nullptr);
}

TEST_F(FactRelationTest, InteriorLiteralDisequalityRetainsScopedExclusion) {
  const loom_value_id_t literal = BuildConstant(42);
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(BuildAssume(
      {inputs_[0]}, {LiteralPredicate(LOOM_PREDICATE_NE, inputs_[0], 42)},
      &assume));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
  const loom_value_id_t result = loom_index_assume_results(assume).values[0];

  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, result,
                                                             literal),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS |
                LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_literal_order(
                &table_, result, 42),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS |
                LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER);
  const loom_predicate_t repeated =
      LiteralPredicate(LOOM_PREDICATE_NE, result, 42);
  EXPECT_EQ(loom_value_fact_table_evaluate_predicate(&table_, &repeated),
            LOOM_DECISION_TRUTH_TRUE);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                &table_, inputs_[0], literal),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_literal_order(
                &table_, inputs_[0], 42),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
}

TEST_F(FactRelationTest, ValueAndLiteralAtomsUseIndependentNamespaces) {
  // inputs_[1] has value ID 1, deliberately matching the literal bits below.
  // Both relation atoms must remain independently queryable.
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(
      BuildAssume({inputs_[0]},
                  {RelationPredicate(LOOM_PREDICATE_GT, inputs_[0], inputs_[1]),
                   LiteralPredicate(LOOM_PREDICATE_NE, inputs_[0], 1)},
                  &assume));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
  const loom_value_id_t result = loom_index_assume_results(assume).values[0];
  const loom_value_id_t literal = BuildConstant(1);

  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, result,
                                                             inputs_[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, result,
                                                             literal),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS |
                LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER);
}

TEST_F(FactRelationTest, NonzeroLiteralUsesScalarFactsWithoutRelationState) {
  const loom_value_id_t zero = BuildConstant(0);
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(BuildAssume(
      {inputs_[0]}, {LiteralPredicate(LOOM_PREDICATE_NE, inputs_[0], 0)},
      &assume));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
  const loom_value_id_t result = loom_index_assume_results(assume).values[0];

  EXPECT_EQ(
      loom_value_fact_table_query_signed_integer_order(&table_, result, zero),
      LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS |
          LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER);
  EXPECT_EQ(table_.integer_relations, nullptr);
}

TEST_F(FactRelationTest, LiteralBoundsStayInScalarFacts) {
  const loom_value_id_t literal = BuildConstant(42);
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(BuildAssume(
      {inputs_[0]}, {LiteralPredicate(LOOM_PREDICATE_LT, inputs_[0], 42)},
      &assume));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
  const loom_value_id_t result = loom_index_assume_results(assume).values[0];

  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, result,
                                                             literal),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
  EXPECT_EQ(table_.integer_relations, nullptr);
}

TEST_F(FactRelationTest, LiteralExclusionsInternAndInheritAsPersistentSets) {
  std::vector<loom_value_id_t> literals;
  std::vector<loom_predicate_t> predicates;
  for (int64_t value = 1; value <= 10; ++value) {
    literals.push_back(BuildConstant(value));
    predicates.push_back(
        LiteralPredicate(LOOM_PREDICATE_NE, inputs_[0], value));
  }
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(BuildAssume({inputs_[0]}, predicates, &assume));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
  loom_value_id_t result = loom_index_assume_results(assume).values[0];

  for (int i = 0; i < 32; ++i) {
    loom_op_t* identity = nullptr;
    IREE_ASSERT_OK(BuildIdentity(result, &identity));
    IREE_ASSERT_OK(
        loom_value_fact_table_compute_op(&table_, module_, identity));
    result = loom_index_assume_results(identity).values[0];
  }
  for (loom_value_id_t literal : literals) {
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, result,
                                                               literal),
              LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS |
                  LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER);
  }
}

TEST_F(FactRelationTest, LiteralExclusionsRefineNarrowedIntervalEndpoints) {
  for (uint32_t excluded = 0; excluded < 32; ++excluded) {
    std::vector<loom_predicate_t> predicates;
    for (int64_t value = -2; value <= 2; ++value) {
      if (excluded & (1u << (value + 2))) {
        predicates.push_back(
            LiteralPredicate(LOOM_PREDICATE_NE, inputs_[0], value));
      }
    }
    loom_op_t* assume = nullptr;
    IREE_ASSERT_OK(BuildAssume({inputs_[0]}, predicates, &assume));
    IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
    const auto result = loom_index_assume_results(assume).values[0];
    for (int64_t lower = -3; lower <= 3; ++lower) {
      for (int64_t upper = lower; upper <= 3; ++upper) {
        std::vector<int64_t> allowed;
        for (int64_t value = lower; value <= upper; ++value) {
          if (value < -2 || value > 2 || !(excluded & (1u << (value + 2)))) {
            allowed.push_back(value);
          }
        }
        if (allowed.empty()) {
          continue;
        }
        auto facts = loom_value_facts_make(lower, upper, 1);
        loom_value_fact_table_refine_integer_exclusions(&table_, result,
                                                        &facts);
        EXPECT_EQ(facts.range_lo, allowed.front());
        EXPECT_EQ(facts.range_hi, allowed.back());
      }
    }
  }
}

TEST_F(FactRelationTest, LiteralExclusionRefinementPreservesIntegerLimits) {
  for (bool lower : {false, true}) {
    const int64_t edge = lower ? INT64_MIN : INT64_MAX;
    const int64_t neighbor = lower ? edge + 1 : edge - 1;
    loom_op_t* assume = nullptr;
    IREE_ASSERT_OK(
        BuildAssume({inputs_[0]},
                    {LiteralPredicate(LOOM_PREDICATE_NE, inputs_[0], edge),
                     LiteralPredicate(LOOM_PREDICATE_NE, inputs_[0], neighbor)},
                    &assume));
    IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
    const auto result = loom_index_assume_results(assume).values[0];
    auto facts = loom_value_facts_unknown();
    loom_value_fact_table_refine_integer_exclusions(&table_, result, &facts);
    EXPECT_EQ(facts.range_lo, lower ? INT64_MIN + 2 : INT64_MIN);
    EXPECT_EQ(facts.range_hi, lower ? INT64_MAX : INT64_MAX - 2);
  }
}

TEST_F(FactRelationTest, SignedRelationsRetainEveryOrientation) {
  struct RelationCase {
    // Authored signed predicate.
    loom_predicate_kind_t predicate_kind;
    // Expected retained comparison semantics.
    loom_symbolic_integer_relation_t relation;
  };
  const RelationCase cases[] = {
      {LOOM_PREDICATE_EQ, LOOM_SYMBOLIC_INTEGER_RELATION_EQ},
      {LOOM_PREDICATE_NE, LOOM_SYMBOLIC_INTEGER_RELATION_NE},
      {LOOM_PREDICATE_LT, LOOM_SYMBOLIC_INTEGER_RELATION_LT},
      {LOOM_PREDICATE_LE, LOOM_SYMBOLIC_INTEGER_RELATION_LE},
      {LOOM_PREDICATE_GT, LOOM_SYMBOLIC_INTEGER_RELATION_GT},
      {LOOM_PREDICATE_GE, LOOM_SYMBOLIC_INTEGER_RELATION_GE},
  };
  for (const RelationCase& relation_case : cases) {
    loom_op_t* assume = nullptr;
    IREE_ASSERT_OK(BuildAssume({inputs_[0], inputs_[1]},
                               {RelationPredicate(relation_case.predicate_kind,
                                                  inputs_[0], inputs_[1])},
                               &assume));
    IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
    const loom_value_slice_t results = loom_index_assume_results(assume);
    const loom_integer_order_outcome_bits_t expected =
        loom_symbolic_integer_relation_outcomes(relation_case.relation);
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                  &table_, results.values[0], results.values[1]),
              expected);
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                  &table_, results.values[1], results.values[0]),
              loom_integer_order_outcomes_swap(expected));
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                  &table_, results.values[0], inputs_[1]),
              expected);
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                  &table_, inputs_[0], results.values[1]),
              expected);
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                  &table_, inputs_[0], inputs_[1]),
              LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
    for (int side = 0; side < 2; ++side) {
      std::vector<VisitedRelation> visited;
      EXPECT_TRUE(loom_value_fact_table_for_each_signed_integer_relation_while(
          &table_, results.values[side], CollectRelation, &visited));
      ASSERT_EQ(visited.size(), 1u);
      EXPECT_EQ(visited[0].counterpart, inputs_[1 - side]);
      EXPECT_EQ(
          visited[0].outcomes,
          side == 0 ? expected : loom_integer_order_outcomes_swap(expected));
    }
    const loom_predicate_t repeated = RelationPredicate(
        relation_case.predicate_kind, results.values[0], results.values[1]);
    EXPECT_EQ(loom_value_fact_table_evaluate_predicate(&table_, &repeated),
              LOOM_DECISION_TRUTH_TRUE);
  }
}

TEST_F(FactRelationTest, VisitsEachScopedCounterpartOnce) {
  loom_value_id_t third = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_value(&builder_, type_, &third));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table_, third, loom_value_facts_unknown()));
  const std::vector<loom_predicate_t> predicates = {
      RelationPredicate(LOOM_PREDICATE_LT, inputs_[0], inputs_[1]),
      RelationPredicate(LOOM_PREDICATE_NE, inputs_[0], third),
      LiteralPredicate(LOOM_PREDICATE_NE, inputs_[0], 17),
  };
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(BuildAssume({inputs_[0]}, predicates, &assume));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
  const loom_value_id_t scoped = loom_index_assume_results(assume).values[0];

  std::vector<VisitedRelation> visited;
  EXPECT_TRUE(loom_value_fact_table_for_each_signed_integer_relation_while(
      &table_, scoped, CollectRelation, &visited));
  ASSERT_EQ(visited.size(), 2u);
  for (const VisitedRelation& relation : visited) {
    if (relation.counterpart == inputs_[1]) {
      EXPECT_EQ(relation.outcomes, LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
    } else {
      EXPECT_EQ(relation.counterpart, third);
      EXPECT_EQ(relation.outcomes, LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS |
                                       LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER);
    }
  }
}

TEST_F(FactRelationTest, UnsignedRelationDoesNotEnterSignedStore) {
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(BuildAssume(
      {inputs_[0], inputs_[1]},
      {RelationPredicate(LOOM_PREDICATE_ULT, inputs_[0], inputs_[1])},
      &assume));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
  const loom_value_slice_t results = loom_index_assume_results(assume);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                &table_, results.values[0], results.values[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
  const loom_predicate_t repeated = RelationPredicate(
      LOOM_PREDICATE_ULT, results.values[0], results.values[1]);
  EXPECT_EQ(loom_value_fact_table_evaluate_predicate(&table_, &repeated),
            LOOM_DECISION_TRUTH_UNKNOWN);
  EXPECT_EQ(table_.integer_relations, nullptr);
}

TEST_F(FactRelationTest, UnsignedOrderRetainsOnlyJustifiedSignedConsequences) {
  const loom_predicate_kind_t predicates[] = {
      LOOM_PREDICATE_ULT, LOOM_PREDICATE_ULE, LOOM_PREDICATE_UGT,
      LOOM_PREDICATE_UGE};
  const loom_symbolic_integer_relation_t relations[] = {
      LOOM_SYMBOLIC_INTEGER_RELATION_LT, LOOM_SYMBOLIC_INTEGER_RELATION_LE,
      LOOM_SYMBOLIC_INTEGER_RELATION_GT, LOOM_SYMBOLIC_INTEGER_RELATION_GE};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(predicates); ++i) {
    for (const auto facts :
         {loom_value_facts_make(0, 63, 1), loom_value_facts_make(-64, -1, 1),
          loom_value_facts_make(-64, 63, 1)}) {
      for (bool tuple : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << "predicate=" << predicates[i]
                     << " lower=" << facts.range_lo
                     << " upper=" << facts.range_hi << " tuple=" << tuple);
        const auto lhs = DefineValue(type_);
        const auto rhs = DefineValue(type_);
        IREE_ASSERT_OK(loom_value_fact_table_define(&table_, lhs, facts));
        IREE_ASSERT_OK(loom_value_fact_table_define(&table_, rhs, facts));
        loom_op_t* assume = nullptr;
        IREE_ASSERT_OK(
            BuildAssume(tuple ? std::vector<loom_value_id_t>{lhs, rhs}
                              : std::vector<loom_value_id_t>{lhs},
                        {RelationPredicate(predicates[i], lhs, rhs)}, &assume));
        IREE_ASSERT_OK(
            loom_value_fact_table_compute_op(&table_, module_, assume));
        const auto result = loom_index_assume_results(assume);
        const auto counterpart = tuple ? result.values[1] : rhs;
        const auto expected =
            facts.range_lo < 0 && facts.range_hi >= 0
                ? LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL
                : loom_symbolic_integer_relation_outcomes(relations[i]);
        EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                      &table_, result.values[0], counterpart),
                  expected);
        EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                      &table_, counterpart, result.values[0]),
                  loom_integer_order_outcomes_swap(expected));
        EXPECT_EQ(
            loom_value_fact_table_query_signed_integer_order(&table_, lhs, rhs),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
      }
    }
  }
}

TEST_F(FactRelationTest, UnsignedOrderUsesItsCompleteNumericConjunction) {
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(BuildAssume(
      {inputs_[0], inputs_[1]},
      {LiteralPredicate(LOOM_PREDICATE_GE, inputs_[0], 0),
       LiteralPredicate(LOOM_PREDICATE_GE, inputs_[1], 0),
       RelationPredicate(LOOM_PREDICATE_ULT, inputs_[0], inputs_[1])},
      &assume));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
  const auto results = loom_index_assume_results(assume);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                &table_, results.values[0], results.values[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                &table_, inputs_[0], inputs_[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
}

TEST_F(FactRelationTest, FloatEqualityDoesNotEnterSignedStore) {
  const loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  const loom_value_id_t lhs = DefineValue(f32_type);
  const loom_value_id_t rhs = DefineValue(f32_type);
  const loom_value_id_t values[] = {lhs, rhs};
  const loom_type_t result_types[] = {f32_type, f32_type};
  loom_predicate_t predicate = RelationPredicate(LOOM_PREDICATE_EQ, lhs, rhs);
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(loom_scalar_assume_build(
      &builder_, values, IREE_ARRAYSIZE(values), &predicate, 1, result_types,
      IREE_ARRAYSIZE(result_types), LOOM_LOCATION_UNKNOWN, &assume));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
  EXPECT_EQ(table_.integer_relations, nullptr);
}

TEST_F(FactRelationTest, DynamicRangeRetainsBothEndpoints) {
  const loom_value_id_t upper = DefineValue(type_);
  const loom_predicate_t range = {
      .kind = LOOM_PREDICATE_RANGE,
      .arg_count = 3,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE,
                   LOOM_PRED_ARG_VALUE},
      .reserved = {},
      .args = {inputs_[0], inputs_[1], upper},
  };
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(BuildAssume({inputs_[0]}, {range}, &assume));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
  const loom_value_id_t result = loom_index_assume_results(assume).values[0];
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, result,
                                                             inputs_[1]),
            loom_symbolic_integer_relation_outcomes(
                LOOM_SYMBOLIC_INTEGER_RELATION_GE));
  EXPECT_EQ(
      loom_value_fact_table_query_signed_integer_order(&table_, result, upper),
      loom_symbolic_integer_relation_outcomes(
          LOOM_SYMBOLIC_INTEGER_RELATION_LE));
  loom_predicate_t repeated = range;
  repeated.args[0] = result;
  EXPECT_EQ(loom_value_fact_table_evaluate_predicate(&table_, &repeated),
            LOOM_DECISION_TRUTH_TRUE);
}

TEST_F(FactRelationTest, TupleAliasesShareLocalRelationsButKeepInheritedScope) {
  const loom_value_id_t third = DefineValue(type_);
  loom_op_t* first = nullptr;
  IREE_ASSERT_OK(BuildAssume(
      {inputs_[0]},
      {RelationPredicate(LOOM_PREDICATE_LT, inputs_[0], inputs_[1])}, &first));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, first));
  const loom_value_id_t scoped = loom_index_assume_results(first).values[0];

  loom_op_t* tuple = nullptr;
  IREE_ASSERT_OK(BuildAssume(
      {scoped, inputs_[0], inputs_[1]},
      {RelationPredicate(LOOM_PREDICATE_GT, inputs_[0], third)}, &tuple));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, tuple));
  const loom_value_slice_t results = loom_index_assume_results(tuple);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                &table_, results.values[0], inputs_[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                &table_, results.values[1], inputs_[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                &table_, results.values[0], third),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                &table_, results.values[1], third),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER);
}

TEST_F(FactRelationTest, RelationsPropagateAcrossDeepIdentityChains) {
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(BuildAssume(
      {inputs_[0]},
      {RelationPredicate(LOOM_PREDICATE_LE, inputs_[0], inputs_[1])}, &assume));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
  loom_value_id_t current = loom_index_assume_results(assume).values[0];
  for (int i = 0; i < 128; ++i) {
    loom_op_t* identity = nullptr;
    IREE_ASSERT_OK(BuildIdentity(current, &identity));
    IREE_ASSERT_OK(
        loom_value_fact_table_compute_op(&table_, module_, identity));
    current = loom_index_assume_results(identity).values[0];
  }
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, current,
                                                             inputs_[1]),
            loom_symbolic_integer_relation_outcomes(
                LOOM_SYMBOLIC_INTEGER_RELATION_LE));
}

TEST_F(FactRelationTest, ContradictoryRelationsProveNoOutcome) {
  loom_op_t* first = nullptr;
  IREE_ASSERT_OK(BuildAssume(
      {inputs_[0]},
      {RelationPredicate(LOOM_PREDICATE_LT, inputs_[0], inputs_[1])}, &first));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, first));
  const loom_value_id_t first_result =
      loom_index_assume_results(first).values[0];
  loom_op_t* second = nullptr;
  IREE_ASSERT_OK(BuildAssume(
      {first_result},
      {RelationPredicate(LOOM_PREDICATE_GT, inputs_[0], inputs_[1])}, &second));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, second));
  EXPECT_EQ(
      loom_value_fact_table_query_signed_integer_order(
          &table_, loom_index_assume_results(second).values[0], inputs_[1]),
      LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
}

TEST_F(FactRelationTest, RelationMutationUndefinitionAndScopeReuse) {
  loom_predicate_t predicate =
      RelationPredicate(LOOM_PREDICATE_LT, inputs_[0], inputs_[1]);
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(BuildAssume({inputs_[0], inputs_[1]}, {predicate}, &assume));
  bool changed = false;
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             assume, &changed));
  EXPECT_TRUE(changed);
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             assume, &changed));
  EXPECT_FALSE(changed);
  const loom_value_slice_t results = loom_index_assume_results(assume);
  auto expect_enumeration = [&](loom_integer_order_outcome_bits_t expected) {
    std::vector<VisitedRelation> visited;
    EXPECT_TRUE(loom_value_fact_table_for_each_signed_integer_relation_while(
        &table_, results.values[0], CollectRelation, &visited));
    if (expected == LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL) {
      EXPECT_TRUE(visited.empty());
    } else {
      ASSERT_EQ(visited.size(), 1u);
      EXPECT_EQ(visited[0].counterpart, inputs_[1]);
      EXPECT_EQ(visited[0].outcomes, expected);
    }
  };
  expect_enumeration(LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);

  const auto first = Snapshot(results.values[0]);
  predicate.kind = LOOM_PREDICATE_GT;
  IREE_ASSERT_OK(loom_index_assume_set_predicates(
      module_, assume, loom_attr_predicate_list(&predicate, 1)));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             assume, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                &table_, results.values[0], results.values[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER);
  expect_enumeration(LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER);

  auto current = Snapshot(results.values[0]);
  EXPECT_FALSE(loom_value_fact_auxiliary_snapshot_equal(&first, &current));

  IREE_ASSERT_OK(loom_index_assume_set_predicates(
      module_, assume, loom_attr_predicate_list(nullptr, 0)));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             assume, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                &table_, results.values[0], results.values[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
  expect_enumeration(LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);

  current = Snapshot(results.values[0]);
  EXPECT_FALSE(loom_value_fact_auxiliary_snapshot_equal(&first, &current));

  predicate.kind = LOOM_PREDICATE_LT;
  IREE_ASSERT_OK(loom_index_assume_set_predicates(
      module_, assume, loom_attr_predicate_list(&predicate, 1)));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
  loom_value_fact_table_undefine(&table_, results.values[0]);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                &table_, results.values[0], inputs_[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
  expect_enumeration(LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             assume, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                &table_, results.values[0], inputs_[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
  expect_enumeration(LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);

  current = Snapshot(results.values[0]);
  EXPECT_TRUE(loom_value_fact_auxiliary_snapshot_equal(&first, &current));

  loom_value_fact_table_clear_scope(&table_);
  EXPECT_EQ(table_.integer_relations, nullptr);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                &table_, results.values[0], inputs_[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
  expect_enumeration(LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
}

TEST_F(FactRelationTest, LiteralRelationLifecycleAndTransfer) {
  const loom_value_id_t literal = BuildConstant(42);
  loom_predicate_t predicate =
      LiteralPredicate(LOOM_PREDICATE_NE, inputs_[0], 42);
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(BuildAssume({inputs_[0]}, {predicate}, &assume));
  bool changed = false;
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             assume, &changed));
  EXPECT_TRUE(changed);
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             assume, &changed));
  EXPECT_FALSE(changed);
  const loom_value_id_t result = loom_index_assume_results(assume).values[0];

  const auto first = Snapshot(result);
  IREE_ASSERT_OK(loom_index_assume_set_predicates(
      module_, assume, loom_attr_predicate_list(nullptr, 0)));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             assume, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, result,
                                                             literal),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);

  auto current = Snapshot(result);
  EXPECT_FALSE(loom_value_fact_auxiliary_snapshot_equal(&first, &current));
  IREE_ASSERT_OK(loom_index_assume_set_predicates(
      module_, assume, loom_attr_predicate_list(&predicate, 1)));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
  loom_value_fact_table_undefine(&table_, result);
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&table_, result,
                                                             literal),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL);
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));

  current = Snapshot(result);
  EXPECT_TRUE(loom_value_fact_auxiliary_snapshot_equal(&first, &current));

  const loom_value_id_t selected[] = {result, literal};
  loom_value_fact_table_t clone = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&clone, &arena_, 0));
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &clone, {&table_, selected, IREE_ARRAYSIZE(selected)}, module_));
  EXPECT_EQ(
      loom_value_fact_table_query_signed_integer_order(&clone, result, literal),
      LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS |
          LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER);

  loom_value_fact_table_t seed = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&seed, &arena_, 0));
  IREE_ASSERT_OK(loom_value_fact_table_seed_values(
      &seed, {&table_, selected, IREE_ARRAYSIZE(selected)}, module_));
  EXPECT_EQ(
      loom_value_fact_table_query_signed_integer_order(&seed, result, literal),
      LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS |
          LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER);

  loom_value_fact_table_clear_scope(&table_);
  EXPECT_EQ(table_.integer_relations, nullptr);
}

TEST_F(FactRelationTest, CloneAndSeedPreserveSelectedRelationScope) {
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(BuildAssume(
      {inputs_[0]},
      {RelationPredicate(LOOM_PREDICATE_LT, inputs_[0], inputs_[1])}, &assume));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
  const loom_value_id_t result = loom_index_assume_results(assume).values[0];

  loom_value_fact_table_t clone = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&clone, &arena_, 0));
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &clone, {&table_, &result, 1}, module_));
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&clone, result,
                                                             inputs_[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);

  loom_value_fact_table_t seed = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&seed, &arena_, 0));
  IREE_ASSERT_OK(
      loom_value_fact_table_seed_values(&seed, {&table_, &result, 1}, module_));
  EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(&seed, result,
                                                             inputs_[1]),
            LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
  for (auto* selected : {&clone, &seed}) {
    std::vector<VisitedRelation> visited;
    EXPECT_TRUE(loom_value_fact_table_for_each_signed_integer_relation_while(
        selected, result, CollectRelation, &visited));
    ASSERT_EQ(visited.size(), 1u);
    EXPECT_EQ(visited[0].counterpart, inputs_[1]);
    EXPECT_EQ(visited[0].outcomes, LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
  }
}

TEST_F(FactRelationTest, LargeTupleUsesGroupedRelationPlan) {
  constexpr uint16_t kValueCount = 24;
  std::vector<loom_value_id_t> values;
  values.reserve(kValueCount);
  for (uint16_t i = 0; i < kValueCount; ++i) {
    values.push_back(DefineValue(type_));
  }
  std::vector<loom_predicate_t> predicates;
  for (uint16_t i = 0; i < 5; ++i) {
    predicates.push_back(
        RelationPredicate(LOOM_PREDICATE_LE, values[i], values.back()));
  }
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(BuildAssume(values, predicates, &assume));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
  const loom_value_slice_t results = loom_index_assume_results(assume);
  for (uint16_t i = 0; i < predicates.size(); ++i) {
    EXPECT_EQ(loom_value_fact_table_query_signed_integer_order(
                  &table_, results.values[i], results.values[kValueCount - 1]),
              loom_symbolic_integer_relation_outcomes(
                  LOOM_SYMBOLIC_INTEGER_RELATION_LE));
  }
}

}  // namespace
}  // namespace loom
