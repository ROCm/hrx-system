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
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/util/fact_relation_inference.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

// Exercises the immutable replay contract used by nested loop and CFG owners.
class FactTableReplayTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_scalar_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SCALAR, vtables, static_cast<uint16_t>(count)));
    vtables = loom_index_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_INDEX, vtables, static_cast<uint16_t>(count)));
    vtables = loom_vector_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_VECTOR, vtables, static_cast<uint16_t>(count)));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("replay"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&table_, &arena_, 4));
    lhs_ = DefineValue(integer_type_);
    rhs_ = DefineValue(integer_type_);
    SetInput(lhs_, 3);
    SetInput(rhs_, 4);
    IREE_ASSERT_OK(loom_scalar_addi_build(
        &builder_, 0, lhs_, rhs_, integer_type_, LOOM_LOCATION_UNKNOWN, &sum_));
    IREE_ASSERT_OK(loom_scalar_addi_build(
        &builder_, 0, loom_scalar_addi_result(sum_), rhs_, integer_type_,
        LOOM_LOCATION_UNKNOWN, &consumer_));
    IREE_ASSERT_OK(loom_value_fact_table_enable_replay_generations(&table_));
    Expect(sum_, 7);
    Expect(consumer_, 11);
    table_.replaying = true;
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_value_id_t DefineValue(loom_type_t type) {
    loom_value_id_t value = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_value(&builder_, type, &value));
    IREE_CHECK_OK(loom_value_fact_table_define(&table_, value,
                                               loom_value_facts_unknown()));
    return value;
  }

  void SetInput(loom_value_id_t value, int64_t number) {
    IREE_ASSERT_OK(loom_value_fact_table_define(
        &table_, value, loom_value_facts_exact_i64(number)));
  }

  void Expect(loom_op_t* op, int64_t number) {
    IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, op));
    const auto facts =
        loom_value_fact_table_lookup(&table_, loom_scalar_addi_result(op));
    ASSERT_TRUE(loom_value_facts_is_exact(facts));
    EXPECT_EQ(facts.range_lo, number);
  }

  // Module and facts share the test's pool, matching the inference lifetime.
  iree_arena_block_pool_t pool_;
  // Storage owned by the fact table.
  iree_arena_allocator_t arena_;
  // Registered operation semantics.
  loom_context_t context_;
  // SSA definitions used by the equations.
  loom_module_t* module_ = nullptr;
  // Builder attached to the module's root block.
  loom_builder_t builder_;
  // Mutable publications and replay watermarks.
  loom_value_fact_table_t table_;
  // Scalar type used by the baseline arithmetic equations.
  const loom_type_t integer_type_ = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  // Input whose numerical publications change between immutable visits.
  loom_value_id_t lhs_;
  // Independent input shared by the producer and its consumer.
  loom_value_id_t rhs_;
  // Producer revisited while an inner solve converges.
  loom_op_t* sum_ = nullptr;
  // Consumer that observes the producer after convergence.
  loom_op_t* consumer_ = nullptr;
};

TEST_F(FactTableReplayTest, InnerConvergencePreservesUnconsumedChanges) {
  SetInput(lhs_, 5);
  Expect(sum_, 9);
  Expect(sum_, 9);
  Expect(consumer_, 13);
  SetInput(lhs_, 3);
  Expect(sum_, 7);
  Expect(sum_, 7);
  Expect(consumer_, 11);
}

TEST_F(FactTableReplayTest, ReplacedAndUndefinedResultsMustRecompute) {
  const auto result = loom_scalar_addi_result(sum_);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, result,
                                              loom_value_facts_unknown()));
  Expect(sum_, 7);
  loom_value_fact_table_undefine(&table_, result);
  Expect(sum_, 7);
}

TEST_F(FactTableReplayTest, RefinementCanReadAnUnlistedValue) {
  auto source = DefineValue(integer_type_);
  table_.context.refine_operands.user_data = &source;
  table_.context.refine_operands.fn =
      [](void* user_data, const loom_value_fact_table_t* table,
         const loom_op_t* op, loom_value_facts_t* facts) {
        facts[0] = loom_value_fact_table_lookup(
            table, *static_cast<loom_value_id_t*>(user_data));
        return loom_op_const_operands(op);
      };
  SetInput(source, 11);
  Expect(sum_, 15);
  SetInput(source, 13);
  Expect(sum_, 17);
  table_.context.refine_operands.fn = nullptr;
  table_.context.refine_operands.user_data = nullptr;
}

TEST_F(FactTableReplayTest, CounterRestartExpiresEveryEvaluation) {
  table_.replay.generation = UINT32_MAX - 1;
  SetInput(lhs_, 5);
  Expect(sum_, 9);
  Expect(consumer_, 13);
  SetInput(rhs_, 6);
  Expect(sum_, 11);
  Expect(consumer_, 17);
}

TEST_F(FactTableReplayTest, StorageGrowthAndScopeResetPreserveInvalidation) {
  IREE_ASSERT_OK(loom_value_fact_table_reserve(&table_, 4096));
  SetInput(lhs_, 5);
  Expect(sum_, 9);
  Expect(consumer_, 13);
  loom_value_fact_table_clear_scope(&table_);
  SetInput(lhs_, 11);
  SetInput(rhs_, 7);
  IREE_ASSERT_OK(loom_value_fact_table_enable_replay_generations(&table_));
  Expect(sum_, 18);
  Expect(consumer_, 25);
}

TEST_F(FactTableReplayTest, SemanticEditsStartOutsideReplay) {
  table_.replaying = false;
  IREE_ASSERT_OK(loom_op_set_operand(module_, sum_, 0, rhs_));
  Expect(sum_, 8);
  table_.replaying = true;
  Expect(consumer_, 12);
}

TEST_F(FactTableReplayTest, EveryResultOfAnEquationMustRemainPublished) {
  loom_op_t* division = nullptr;
  const loom_type_t types[] = {integer_type_, integer_type_};
  IREE_ASSERT_OK(loom_scalar_divremsi_build(&builder_, lhs_, rhs_, types,
                                            LOOM_LOCATION_UNKNOWN, &division));
  const auto quotient = loom_scalar_divremsi_quotient(division);
  const auto remainder = loom_scalar_divremsi_remainder(division);
  auto expect = [&](int64_t expected_quotient, int64_t expected_remainder) {
    IREE_ASSERT_OK(
        loom_value_fact_table_compute_op(&table_, module_, division));
    EXPECT_EQ(loom_value_fact_table_lookup(&table_, quotient).range_lo,
              expected_quotient);
    EXPECT_EQ(loom_value_fact_table_lookup(&table_, remainder).range_lo,
              expected_remainder);
  };
  SetInput(lhs_, 11);
  SetInput(rhs_, 3);
  expect(3, 2);
  loom_value_fact_table_undefine(&table_, remainder);
  expect(3, 2);
  SetInput(lhs_, 12);
  expect(4, 0);
}

TEST_F(FactTableReplayTest, ChangedExtensionInvalidatesEqualNumericEnvelope) {
  const auto type = loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                                        loom_dim_pack_static(2), 0);
  const auto input = DefineValue(type);
  loom_op_t* sum = nullptr;
  IREE_ASSERT_OK(loom_vector_addi_build(&builder_, 0, input, input, type,
                                        LOOM_LOCATION_UNKNOWN, &sum));
  for (int64_t first : {1, 9, 1}) {
    const loom_value_facts_t lanes[] = {
        loom_value_facts_exact_i64(first),
        loom_value_facts_exact_i64(10 - first),
    };
    loom_value_facts_t facts;
    IREE_ASSERT_OK(loom_value_facts_make_small_static_lanes(
        &table_.context, {lanes, IREE_ARRAYSIZE(lanes)}, &facts));
    IREE_ASSERT_OK(loom_value_fact_table_define(&table_, input, facts));
    for (unsigned visit = 0; visit < 2; ++visit) {
      IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, sum));
      loom_value_fact_small_static_lanes_t result;
      ASSERT_TRUE(loom_value_facts_query_small_static_lanes(
          &table_.context,
          loom_value_fact_table_lookup(&table_, loom_vector_addi_result(sum)),
          &result));
      ASSERT_EQ(result.count, 2u);
      EXPECT_EQ(result.lanes[0].range_lo, 2 * first);
      EXPECT_EQ(result.lanes[1].range_lo, 2 * (10 - first));
    }
  }
}

TEST_F(FactTableReplayTest, IdentityChangesDoNotRequireNumericChanges) {
  for (auto input : {lhs_, rhs_}) {
    IREE_ASSERT_OK(loom_value_fact_table_define(&table_, input,
                                                loom_value_facts_unknown()));
  }
  loom_op_t* comparison = nullptr;
  IREE_ASSERT_OK(
      loom_scalar_cmpi_build(&builder_, LOOM_SCALAR_CMPI_PREDICATE_EQ, lhs_,
                             rhs_, LOOM_LOCATION_UNKNOWN, &comparison));
  const auto result = loom_scalar_cmpi_result(comparison);
  IREE_ASSERT_OK(
      loom_value_fact_table_compute_op(&table_, module_, comparison));
  EXPECT_FALSE(
      loom_value_facts_is_exact(loom_value_fact_table_lookup(&table_, result)));
  IREE_ASSERT_OK(
      loom_value_fact_table_define_identity(&table_, rhs_, lhs_, nullptr));
  IREE_ASSERT_OK(
      loom_value_fact_table_compute_op(&table_, module_, comparison));
  EXPECT_TRUE(
      loom_value_facts_is_exact(loom_value_fact_table_lookup(&table_, result)));
  EXPECT_EQ(loom_value_fact_table_lookup(&table_, result).range_lo, 1);
  loom_value_fact_table_undefine_identity(&table_, rhs_);
  IREE_ASSERT_OK(
      loom_value_fact_table_compute_op(&table_, module_, comparison));
  EXPECT_FALSE(
      loom_value_facts_is_exact(loom_value_fact_table_lookup(&table_, result)));
}

// Conversions can withdraw an identity while retaining their full destination
// range. A multi-result identity must carry that change through either API.
class FactOriginReplayTest
    : public FactTableReplayTest,
      public ::testing::WithParamInterface<loom_scalar_type_t> {};

TEST_P(FactOriginReplayTest,
       ConversionIdentityAppearsAndWithdrawsThroughAliases) {
  const auto source_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  const auto result_type = loom_type_scalar(GetParam());
  const auto source = DefineValue(source_type);
  const auto zero = DefineValue(result_type);
  SetInput(zero, 0);
  const int bit_count = loom_scalar_type_bitwidth(GetParam());
  const int64_t upper = GetParam() == LOOM_SCALAR_TYPE_I1
                            ? 1
                            : (INT64_C(1) << (bit_count - 1)) - 1;
  const int64_t lower = GetParam() == LOOM_SCALAR_TYPE_I1 ? 0 : -upper - 1;
  loom_op_t* conversion = nullptr;
  IREE_ASSERT_OK(loom_scalar_trunci_build(&builder_, source, source_type,
                                          result_type, LOOM_LOCATION_UNKNOWN,
                                          &conversion));
  const auto converted = loom_scalar_trunci_result(conversion);
  const loom_value_id_t inputs[] = {zero, converted};
  const loom_type_t types[] = {result_type, result_type};
  loom_op_t* identity = nullptr;
  IREE_ASSERT_OK(loom_scalar_assume_build(&builder_, inputs, 2, nullptr, 0,
                                          types, 2, LOOM_LOCATION_UNKNOWN,
                                          &identity));
  const auto aliases = loom_scalar_assume_results(identity);
  loom_op_t* sum = nullptr;
  IREE_ASSERT_OK(loom_scalar_addi_build(&builder_, 0, aliases.values[1], zero,
                                        result_type, LOOM_LOCATION_UNKNOWN,
                                        &sum));
  const auto result = loom_scalar_addi_result(sum);
  for (bool report_change : {false, true}) {
    SCOPED_TRACE(report_change);
    for (bool preserves_value : {true, false, true, false}) {
      SCOPED_TRACE(preserves_value);
      IREE_ASSERT_OK(loom_value_fact_table_define(
          &table_, source,
          loom_value_facts_make(lower - !preserves_value,
                                upper + !preserves_value, 1)));
      for (int visit = 0; visit < 2; ++visit) {
        SCOPED_TRACE(visit);
        bool changed = false;
        IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(
            &table_, module_, conversion, report_change ? &changed : nullptr));
        IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(
            &table_, module_, identity, report_change ? &changed : nullptr));
        if (report_change) {
          EXPECT_EQ(changed, visit == 0);
        }
        IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, sum));
        EXPECT_EQ(loom_value_fact_table_query_numeric_identity(&table_, result),
                  preserves_value ? source : converted);
        const auto facts = loom_value_fact_table_lookup(&table_, result);
        EXPECT_EQ(facts.range_lo, lower);
        EXPECT_EQ(facts.range_hi, upper);
        EXPECT_EQ(
            loom_value_fact_table_lookup(&table_, aliases.values[0]).range_lo,
            0);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(IntegerWidths, FactOriginReplayTest,
                         ::testing::Values(LOOM_SCALAR_TYPE_I1,
                                           LOOM_SCALAR_TYPE_I8,
                                           LOOM_SCALAR_TYPE_I16,
                                           LOOM_SCALAR_TYPE_I32));

TEST_F(FactTableReplayTest,
       DirectOriginChangesInvalidateAnEqualNumericOperand) {
  const auto input = DefineValue(integer_type_);
  const auto zero = DefineValue(integer_type_);
  SetInput(zero, 0);
  loom_op_t* sum = nullptr;
  IREE_ASSERT_OK(loom_scalar_addi_build(
      &builder_, 0, input, zero, integer_type_, LOOM_LOCATION_UNKNOWN, &sum));
  for (auto source : {lhs_, rhs_, LOOM_VALUE_ID_INVALID, lhs_}) {
    SCOPED_TRACE(source);
    IREE_ASSERT_OK(loom_value_fact_table_define_exact_scalar_origin(
        &table_, input, {source}));
    for (int visit = 0; visit < 2; ++visit) {
      IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, sum));
      EXPECT_EQ(loom_value_fact_table_query_numeric_identity(
                    &table_, loom_scalar_addi_result(sum)),
                source == LOOM_VALUE_ID_INVALID ? input : source);
    }
  }
}

TEST_F(FactTableReplayTest, OriginPublicationCounterRestartExpiresEvaluations) {
  const auto input = DefineValue(integer_type_);
  const auto zero = DefineValue(integer_type_);
  SetInput(zero, 0);
  loom_op_t* sum = nullptr;
  IREE_ASSERT_OK(loom_scalar_addi_build(
      &builder_, 0, input, zero, integer_type_, LOOM_LOCATION_UNKNOWN, &sum));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, sum));
  table_.replay.generation = UINT32_MAX - 1;
  IREE_ASSERT_OK(
      loom_value_fact_table_define_exact_scalar_origin(&table_, input, {lhs_}));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, sum));
  EXPECT_EQ(loom_value_fact_table_query_numeric_identity(
                &table_, loom_scalar_addi_result(sum)),
            lhs_);
  IREE_ASSERT_OK(
      loom_value_fact_table_define_exact_scalar_origin(&table_, input, {rhs_}));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, sum));
  EXPECT_EQ(loom_value_fact_table_query_numeric_identity(
                &table_, loom_scalar_addi_result(sum)),
            rhs_);
}

TEST_F(FactTableReplayTest, RelationChangesInvalidateAnEqualNumericOperand) {
  const auto index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const auto position = DefineValue(index_type);
  const auto extent = DefineValue(index_type);
  const auto begin = DefineValue(index_type);
  SetInput(begin, 0);
  for (auto value : {position, extent}) {
    IREE_ASSERT_OK(loom_value_fact_table_define(
        &table_, value, loom_value_facts_make(0, 100, 1)));
  }
  loom_op_t* difference = nullptr;
  IREE_ASSERT_OK(loom_index_sub_build(&builder_, extent, position, index_type,
                                      LOOM_LOCATION_UNKNOWN, &difference));
  bool previously_bounded = false;
  for (bool bounded : {false, true, false, true}) {
    const auto previous_generation = table_.replay.generation;
    if (bounded) {
      IREE_ASSERT_OK(loom_value_fact_integer_relations_define_counted_range(
          &table_, position, begin, extent));
    } else {
      loom_value_fact_integer_relations_undefine(&table_, position);
    }
    if (bounded != previously_bounded) {
      EXPECT_GT(loom_value_fact_table_replay_generation(&table_, position),
                previous_generation);
    } else {
      EXPECT_EQ(table_.replay.generation, previous_generation);
    }
    previously_bounded = bounded;
    for (int visit = 0; visit < 2; ++visit) {
      IREE_ASSERT_OK(
          loom_value_fact_table_compute_op(&table_, module_, difference));
      const auto facts = loom_value_fact_table_lookup(
          &table_, loom_index_sub_result(difference));
      EXPECT_EQ(facts.range_lo, bounded ? 1 : -100);
      EXPECT_EQ(facts.range_hi, 100);
    }
  }
}

}  // namespace
}  // namespace loom
