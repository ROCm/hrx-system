// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/predicate_facts.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class PredicateFactsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

TEST(PredicateFactsExportTest, ExportsRangeAndDivisibility) {
  constexpr loom_value_id_t kValue = 7;
  loom_predicate_t predicates[LOOM_VALUE_FACT_INTEGER_REFINEMENT_CAPACITY] = {};
  const uint16_t count = loom_value_facts_export_integer_refinements(
      kValue, loom_value_facts_unknown(), loom_value_facts_make(-12, 24, 4),
      predicates, IREE_ARRAYSIZE(predicates));

  ASSERT_EQ(count, 2);
  EXPECT_EQ(predicates[0].kind, LOOM_PREDICATE_RANGE);
  EXPECT_EQ(predicates[0].args[0], kValue);
  EXPECT_EQ(predicates[0].args[1], -12);
  EXPECT_EQ(predicates[0].args[2], 24);
  EXPECT_EQ(predicates[1].kind, LOOM_PREDICATE_MULTIPLE_OF);
  EXPECT_EQ(predicates[1].args[0], kValue);
  EXPECT_EQ(predicates[1].args[1], 4);
}

TEST(PredicateFactsExportTest, ExportsPredicateOnlyIntegerFacts) {
  constexpr loom_value_id_t kValue = 11;
  loom_value_facts_t proven = loom_value_facts_make(0, 1024, 8);
  proven.flags |= LOOM_VALUE_FACT_POWER_OF_TWO | LOOM_VALUE_FACT_NON_ZERO;
  loom_predicate_t predicates[LOOM_VALUE_FACT_INTEGER_REFINEMENT_CAPACITY] = {};
  uint16_t count = loom_value_facts_export_integer_refinements(
      kValue, loom_value_facts_unknown(), proven, predicates,
      IREE_ARRAYSIZE(predicates));

  ASSERT_EQ(count, 3);
  EXPECT_EQ(predicates[0].kind, LOOM_PREDICATE_RANGE);
  EXPECT_EQ(predicates[1].kind, LOOM_PREDICATE_MULTIPLE_OF);
  EXPECT_EQ(predicates[2].kind, LOOM_PREDICATE_POWER_OF_TWO);
  EXPECT_EQ(predicates[2].args[0], kValue);

  proven.flags &= ~LOOM_VALUE_FACT_POWER_OF_TWO;
  count = loom_value_facts_export_integer_refinements(
      kValue, loom_value_facts_make(0, 1024, 8), proven, predicates,
      IREE_ARRAYSIZE(predicates));
  ASSERT_EQ(count, 1);
  EXPECT_EQ(predicates[0].kind, LOOM_PREDICATE_NE);
  EXPECT_EQ(predicates[0].args[0], kValue);
  EXPECT_EQ(predicates[0].args[1], 0);
}

TEST(PredicateFactsExportTest, OmitsFactsImpliedByExactOrRange) {
  constexpr loom_value_id_t kValue = 13;
  loom_predicate_t predicates[LOOM_VALUE_FACT_INTEGER_REFINEMENT_CAPACITY] = {};
  loom_value_facts_t exact = loom_value_facts_exact_i64(16);
  EXPECT_EQ(loom_value_facts_export_integer_refinements(
                kValue, loom_value_facts_unknown(), exact, predicates,
                IREE_ARRAYSIZE(predicates)),
            1);
  EXPECT_EQ(predicates[0].kind, LOOM_PREDICATE_EQ);

  loom_value_facts_t positive = loom_value_facts_make(1, 64, 1);
  positive.flags |= LOOM_VALUE_FACT_NON_ZERO;
  EXPECT_EQ(loom_value_facts_export_integer_refinements(
                kValue, loom_value_facts_unknown(), positive, predicates,
                IREE_ARRAYSIZE(predicates)),
            1);
  EXPECT_EQ(predicates[0].kind, LOOM_PREDICATE_RANGE);
}

TEST(PredicateFactsExportTest, ExactAndUnknownStayMinimal) {
  loom_predicate_t predicate = {};
  EXPECT_EQ(loom_value_facts_export_integer_refinements(
                3, loom_value_facts_unknown(), loom_value_facts_exact_i64(42),
                &predicate, 1),
            1);
  EXPECT_EQ(predicate.kind, LOOM_PREDICATE_EQ);
  EXPECT_EQ(predicate.args[0], 3);
  EXPECT_EQ(predicate.args[1], 42);
  EXPECT_EQ(loom_value_facts_export_integer_refinements(
                3, loom_value_facts_unknown(), loom_value_facts_unknown(),
                &predicate, 1),
            0);
  EXPECT_EQ(loom_value_facts_export_integer_refinements(
                3, loom_value_facts_make(0, 63, 1),
                loom_value_facts_make(0, 63, 1), &predicate, 1),
            0);
}

TEST_F(PredicateFactsTest, SmallListStaysAllocationFree) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 1));
  const loom_value_id_t value = 0;
  loom_value_facts_t facts = loom_value_facts_unknown();
  loom_predicate_t predicate = {.kind = LOOM_PREDICATE_RANGE, .arg_count = 3};
  predicate.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicate.arg_tags[1] = LOOM_PRED_ARG_CONST;
  predicate.arg_tags[2] = LOOM_PRED_ARG_CONST;
  predicate.args[0] = value;
  predicate.args[1] = 0;
  predicate.args[2] = 255;

  IREE_ASSERT_OK(loom_value_fact_table_apply_alias_predicates(
      &table, &value, 1, &predicate, 1, &facts));

  EXPECT_EQ(facts.range_lo, 0);
  EXPECT_EQ(facts.range_hi, 255);
  EXPECT_EQ(table.scratch.alias_ordinals.buckets, nullptr);
  EXPECT_EQ(table.scratch.alias_ordinals.capacity, 0u);
}

TEST_F(PredicateFactsTest, RangeUsesLiteralAndExternalEndpointFacts) {
  constexpr loom_value_id_t kTarget = 0;
  constexpr loom_value_id_t kUpper = 1;
  constexpr loom_value_id_t kExactLower = 2;
  constexpr loom_value_id_t kExactUpper = 3;
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 4));
  const loom_value_facts_t upper_facts = loom_value_facts_make(1, 709, 1);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, kUpper, upper_facts));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, kExactLower,
                                              loom_value_facts_exact_i64(-4)));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, kExactUpper,
                                              loom_value_facts_exact_i64(17)));

  loom_value_facts_t target_facts = loom_value_facts_unknown();
  loom_predicate_t predicate = {.kind = LOOM_PREDICATE_RANGE, .arg_count = 3};
  predicate.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicate.arg_tags[1] = LOOM_PRED_ARG_CONST;
  predicate.arg_tags[2] = LOOM_PRED_ARG_VALUE;
  predicate.args[0] = kTarget;
  predicate.args[1] = 0;
  predicate.args[2] = kUpper;

  IREE_ASSERT_OK(loom_value_fact_table_apply_alias_predicates(
      &table, &kTarget, 1, &predicate, 1, &target_facts));

  EXPECT_EQ(target_facts.range_lo, 0);
  EXPECT_EQ(target_facts.range_hi, 709);
  EXPECT_EQ(loom_value_fact_table_lookup(&table, kUpper).range_lo,
            upper_facts.range_lo);
  EXPECT_EQ(loom_value_fact_table_lookup(&table, kUpper).range_hi,
            upper_facts.range_hi);

  target_facts = loom_value_facts_unknown();
  predicate.arg_tags[1] = LOOM_PRED_ARG_VALUE;
  predicate.args[1] = kExactLower;
  predicate.args[2] = kExactUpper;
  IREE_ASSERT_OK(loom_value_fact_table_apply_alias_predicates(
      &table, &kTarget, 1, &predicate, 1, &target_facts));
  EXPECT_EQ(target_facts.range_lo, -4);
  EXPECT_EQ(target_facts.range_hi, 17);
}

TEST_F(PredicateFactsTest, RangeUsesBothExternalEndpointIntervals) {
  constexpr loom_value_id_t kTarget = 0;
  constexpr loom_value_id_t kLower = 1;
  constexpr loom_value_id_t kUpper = 2;
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 3));
  const loom_value_facts_t lower_facts = loom_value_facts_make(-128, -32, 1);
  const loom_value_facts_t upper_facts = loom_value_facts_make(32, 4096, 1);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, kLower, lower_facts));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, kUpper, upper_facts));

  loom_value_facts_t target_facts = loom_value_facts_unknown();
  loom_predicate_t predicate = {.kind = LOOM_PREDICATE_RANGE, .arg_count = 3};
  predicate.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicate.arg_tags[1] = LOOM_PRED_ARG_VALUE;
  predicate.arg_tags[2] = LOOM_PRED_ARG_VALUE;
  predicate.args[0] = kTarget;
  predicate.args[1] = kLower;
  predicate.args[2] = kUpper;

  IREE_ASSERT_OK(loom_value_fact_table_apply_alias_predicates(
      &table, &kTarget, 1, &predicate, 1, &target_facts));

  EXPECT_EQ(target_facts.range_lo, -128);
  EXPECT_EQ(target_facts.range_hi, 4096);
  EXPECT_EQ(loom_value_fact_table_lookup(&table, kLower).range_hi,
            lower_facts.range_hi);
  EXPECT_EQ(loom_value_fact_table_lookup(&table, kUpper).range_lo,
            upper_facts.range_lo);
}

TEST_F(PredicateFactsTest, RangeRefinesAliasEndpointsReciprocally) {
  constexpr loom_value_id_t kTarget = 0;
  constexpr loom_value_id_t kLower = 1;
  constexpr loom_value_id_t kUpper = 2;
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 3));
  const loom_value_id_t values[] = {kTarget, kLower, kUpper};
  loom_value_facts_t facts[] = {loom_value_facts_make(0, 100, 1),
                                loom_value_facts_unknown(),
                                loom_value_facts_unknown()};
  loom_predicate_t predicate = {.kind = LOOM_PREDICATE_RANGE, .arg_count = 3};
  predicate.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicate.arg_tags[1] = LOOM_PRED_ARG_VALUE;
  predicate.arg_tags[2] = LOOM_PRED_ARG_VALUE;
  predicate.args[0] = kTarget;
  predicate.args[1] = kLower;
  predicate.args[2] = kUpper;

  IREE_ASSERT_OK(loom_value_fact_table_apply_alias_predicates(
      &table, values, IREE_ARRAYSIZE(values), &predicate, 1, facts));

  EXPECT_EQ(facts[0].range_lo, 0);
  EXPECT_EQ(facts[0].range_hi, 100);
  EXPECT_EQ(facts[1].range_lo, INT64_MIN);
  EXPECT_EQ(facts[1].range_hi, 100);
  EXPECT_EQ(facts[2].range_lo, 0);
  EXPECT_EQ(facts[2].range_hi, INT64_MAX);
}

TEST_F(PredicateFactsTest, RangeRejectsFloatEndpointIntervals) {
  constexpr loom_value_id_t kTarget = 0;
  constexpr loom_value_id_t kUpper = 1;
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 2));
  loom_value_facts_t upper_facts = loom_value_facts_make(1, 709, 1);
  upper_facts.flags |= LOOM_VALUE_FACT_FLOAT;
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, kUpper, upper_facts));

  loom_value_facts_t target_facts = loom_value_facts_unknown();
  loom_predicate_t predicate = {.kind = LOOM_PREDICATE_RANGE, .arg_count = 3};
  predicate.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicate.arg_tags[1] = LOOM_PRED_ARG_CONST;
  predicate.arg_tags[2] = LOOM_PRED_ARG_VALUE;
  predicate.args[0] = kTarget;
  predicate.args[1] = 0;
  predicate.args[2] = kUpper;

  IREE_ASSERT_OK(loom_value_fact_table_apply_alias_predicates(
      &table, &kTarget, 1, &predicate, 1, &target_facts));

  EXPECT_EQ(target_facts.range_lo, 0);
  EXPECT_EQ(target_facts.range_hi, INT64_MAX);
}

TEST_F(PredicateFactsTest, LargeDisorderedListUsesReusableOrdinalIndex) {
  constexpr uint16_t kAliasCount = 257;
  constexpr loom_value_id_t kExternalBound = 600;
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(
      loom_value_fact_table_initialize(&table, &arena_, kExternalBound + 1));
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &table, kExternalBound, loom_value_facts_make(1, 65535, 1)));

  std::vector<loom_value_id_t> values(kAliasCount);
  std::vector<loom_value_facts_t> facts(kAliasCount,
                                        loom_value_facts_unknown());
  for (uint16_t i = 0; i < kAliasCount; ++i) {
    values[i] = ((i * 131) % kAliasCount) * 2;
  }

  constexpr uint16_t kTargetOrdinal = 251;
  loom_predicate_t predicates[5] = {};
  for (uint16_t i = 0; i < IREE_ARRAYSIZE(predicates); ++i) {
    loom_predicate_t* predicate = &predicates[i];
    predicate->kind = i == 0 ? LOOM_PREDICATE_RANGE : LOOM_PREDICATE_LE;
    predicate->arg_count = i == 0 ? 3 : 2;
    predicate->arg_tags[0] = LOOM_PRED_ARG_VALUE;
    predicate->args[0] = values[kTargetOrdinal - i];
    if (i == 0) {
      predicate->arg_tags[1] = LOOM_PRED_ARG_CONST;
      predicate->arg_tags[2] = LOOM_PRED_ARG_VALUE;
      predicate->args[1] = 0;
      predicate->args[2] = kExternalBound;
    } else {
      predicate->arg_tags[1] = LOOM_PRED_ARG_VALUE;
      predicate->args[1] = kExternalBound;
    }
  }

  IREE_ASSERT_OK(loom_value_fact_table_apply_alias_predicates(
      &table, values.data(), kAliasCount, predicates,
      IREE_ARRAYSIZE(predicates), facts.data()));

  for (uint16_t i = 0; i < IREE_ARRAYSIZE(predicates); ++i) {
    EXPECT_EQ(facts[kTargetOrdinal - i].range_lo, i == 0 ? 0 : INT64_MIN);
    EXPECT_EQ(facts[kTargetOrdinal - i].range_hi, 65535);
  }
  ASSERT_NE(table.scratch.alias_ordinals.buckets, nullptr);
  for (iree_host_size_t i = 0; i < table.scratch.alias_ordinals.capacity; ++i) {
    EXPECT_EQ(table.scratch.alias_ordinals.buckets[i], 0);
  }

  const iree_host_size_t used_size = arena_.used_allocation_size;
  std::fill(facts.begin(), facts.end(), loom_value_facts_unknown());
  IREE_ASSERT_OK(loom_value_fact_table_apply_alias_predicates(
      &table, values.data(), kAliasCount, predicates,
      IREE_ARRAYSIZE(predicates), facts.data()));
  EXPECT_EQ(arena_.used_allocation_size, used_size);
  for (uint16_t i = 0; i < IREE_ARRAYSIZE(predicates); ++i) {
    EXPECT_EQ(facts[kTargetOrdinal - i].range_hi, 65535);
  }
}

TEST_F(PredicateFactsTest, SparseAliasBatchesReusePopulationSizedStorage) {
  constexpr uint16_t kAliasCounts[] = {17, 257, 17, 129, 257, 17, 33, 17};
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  const iree_host_size_t initial_size = arena_.used_allocation_size;
  uint16_t maximum_alias_count = 0;
  for (uint16_t batch = 0; batch < IREE_ARRAYSIZE(kAliasCounts); ++batch) {
    const uint16_t alias_count = kAliasCounts[batch];
    std::vector<loom_value_id_t> values(alias_count);
    std::vector<loom_value_facts_t> facts(alias_count,
                                          loom_value_facts_unknown());
    std::vector<loom_predicate_t> predicates(alias_count);
    for (uint16_t i = 0; i < alias_count; ++i) {
      values[i] = (batch + 1) * 65536u + ((i * 7) % alias_count) * 193u;
      predicates[i].kind = LOOM_PREDICATE_RANGE;
      predicates[i].arg_count = 3;
      predicates[i].arg_tags[0] = LOOM_PRED_ARG_VALUE;
      predicates[i].arg_tags[1] = LOOM_PRED_ARG_CONST;
      predicates[i].arg_tags[2] = LOOM_PRED_ARG_CONST;
      predicates[i].args[0] = values[i];
      predicates[i].args[1] = batch + i;
      predicates[i].args[2] = batch + i + 1;
    }
    const iree_host_size_t previous_size = arena_.used_allocation_size;
    IREE_ASSERT_OK(loom_value_fact_table_apply_alias_predicates(
        &table, values.data(), alias_count, predicates.data(), alias_count,
        facts.data()));
    for (uint16_t i = 0; i < alias_count; ++i) {
      EXPECT_EQ(facts[i].range_lo, batch + i);
      EXPECT_EQ(facts[i].range_hi, batch + i + 1);
    }
    if (alias_count <= maximum_alias_count) {
      EXPECT_EQ(arena_.used_allocation_size, previous_size);
    }
    maximum_alias_count = iree_max(maximum_alias_count, alias_count);
    EXPECT_LE(arena_.used_allocation_size - initial_size,
              maximum_alias_count * 16u);
  }
}

TEST_F(PredicateFactsTest, DuplicateAliasKeepsFirstOrdinalSemantics) {
  constexpr uint16_t kAliasCount = 17;
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 32));
  loom_value_id_t values[kAliasCount];
  loom_value_facts_t facts[kAliasCount];
  for (uint16_t i = 0; i < kAliasCount; ++i) {
    values[i] = i;
    facts[i] = loom_value_facts_unknown();
  }
  values[kAliasCount - 1] = values[0];

  loom_predicate_t predicates[5] = {};
  for (loom_predicate_t& predicate : predicates) {
    predicate.kind = LOOM_PREDICATE_RANGE;
    predicate.arg_count = 3;
    predicate.arg_tags[0] = LOOM_PRED_ARG_VALUE;
    predicate.arg_tags[1] = LOOM_PRED_ARG_CONST;
    predicate.arg_tags[2] = LOOM_PRED_ARG_CONST;
    predicate.args[0] = values[0];
    predicate.args[1] = 4;
    predicate.args[2] = 8;
  }

  IREE_ASSERT_OK(loom_value_fact_table_apply_alias_predicates(
      &table, values, kAliasCount, predicates, IREE_ARRAYSIZE(predicates),
      facts));

  EXPECT_EQ(facts[0].range_lo, 4);
  EXPECT_EQ(facts[0].range_hi, 8);
  EXPECT_TRUE(loom_value_facts_is_unknown(facts[kAliasCount - 1]));
}

TEST_F(PredicateFactsTest, FloatRelationsDoNotUseIntegerIntervals) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 2));
  const loom_value_id_t values[] = {0, 1};
  loom_value_facts_t facts[] = {loom_value_facts_unknown(),
                                loom_value_facts_unknown()};
  facts[0].flags |= LOOM_VALUE_FACT_FLOAT;
  facts[1].flags |= LOOM_VALUE_FACT_FLOAT;
  loom_predicate_t predicate = {.kind = LOOM_PREDICATE_EQ, .arg_count = 2};
  predicate.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicate.arg_tags[1] = LOOM_PRED_ARG_VALUE;
  predicate.args[0] = values[0];
  predicate.args[1] = values[1];

  IREE_ASSERT_OK(loom_value_fact_table_apply_alias_predicates(
      &table, values, IREE_ARRAYSIZE(values), &predicate, 1, facts));

  EXPECT_TRUE(loom_value_facts_is_float(facts[0]));
  EXPECT_TRUE(loom_value_facts_is_float(facts[1]));
  EXPECT_FALSE(loom_value_facts_is_exact(facts[0]));
  EXPECT_FALSE(loom_value_facts_is_exact(facts[1]));
}

TEST_F(PredicateFactsTest, PredicateProofCombinesScalarAndLiteralFacts) {
  constexpr loom_value_id_t kMultiple = 0;
  constexpr loom_value_id_t kPowerOfTwo = 1;
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 2));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, kMultiple,
                                              loom_value_facts_make(0, 64, 8)));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, kPowerOfTwo,
                                              loom_value_facts_exact_i64(16)));

  const loom_predicate_t predicates[] = {
      {
          .kind = LOOM_PREDICATE_RANGE,
          .arg_count = 3,
          .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                       LOOM_PRED_ARG_CONST},
          .args = {kMultiple, 0, 64},
      },
      {
          .kind = LOOM_PREDICATE_MULTIPLE_OF,
          .arg_count = 2,
          .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                       LOOM_PRED_ARG_NONE},
          .args = {kMultiple, 8, 0},
      },
      {
          .kind = LOOM_PREDICATE_NE,
          .arg_count = 2,
          .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                       LOOM_PRED_ARG_NONE},
          .args = {kMultiple, 10, 0},
      },
      {
          .kind = LOOM_PREDICATE_POWER_OF_TWO,
          .arg_count = 1,
          .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_NONE,
                       LOOM_PRED_ARG_NONE},
          .args = {kPowerOfTwo, 0, 0},
      },
  };

  EXPECT_TRUE(loom_value_fact_table_proves_predicate_list(
      &table, predicates, IREE_ARRAYSIZE(predicates)));
  EXPECT_TRUE(loom_value_fact_table_proves_predicate_list(&table, nullptr, 0));

  loom_predicate_t stronger = predicates[0];
  stronger.args[2] = 32;
  EXPECT_EQ(loom_value_fact_table_evaluate_predicate(&table, &stronger),
            LOOM_DECISION_TRUTH_UNKNOWN);
  EXPECT_FALSE(
      loom_value_fact_table_proves_predicate_list(&table, &stronger, 1));
}

static loom_predicate_t MakeBinaryPredicate(loom_predicate_kind_t kind,
                                            loom_value_id_t lhs,
                                            loom_value_id_t rhs) {
  loom_predicate_t predicate = {.kind = kind, .arg_count = 2};
  predicate.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicate.arg_tags[1] = LOOM_PRED_ARG_VALUE;
  predicate.args[0] = lhs;
  predicate.args[1] = rhs;
  return predicate;
}

TEST(PredicateFactsImplicationTest, IdenticalPredicatesCoverVocabulary) {
  for (loom_predicate_kind_t kind = 0; kind < LOOM_PREDICATE_COUNT_; ++kind) {
    loom_predicate_t predicate = {
        .kind = kind,
        .arg_count = loom_predicate_kind_argument_count(kind),
    };
    predicate.arg_tags[0] = LOOM_PRED_ARG_VALUE;
    predicate.args[0] = 1;
    for (uint8_t i = 1; i < predicate.arg_count; ++i) {
      predicate.arg_tags[i] = LOOM_PRED_ARG_CONST;
      predicate.args[i] = i == 1 ? 2 : 8;
    }
    EXPECT_TRUE(loom_value_fact_table_predicate_implies(
        /*table=*/nullptr, &predicate, &predicate))
        << "predicate kind " << static_cast<int>(kind);
  }
}

TEST(PredicateFactsImplicationTest, PreservesPredicateDomainSemantics) {
  loom_predicate_t premise = {
      .kind = LOOM_PREDICATE_RANGE,
      .arg_count = 3,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                   LOOM_PRED_ARG_CONST},
      .args = {1, 4, 32},
  };
  loom_predicate_t conclusion = premise;
  conclusion.args[1] = 0;
  conclusion.args[2] = 64;
  EXPECT_TRUE(loom_value_fact_table_predicate_implies(
      /*table=*/nullptr, &premise, &conclusion));
  EXPECT_FALSE(loom_value_fact_table_predicate_implies(
      /*table=*/nullptr, &conclusion, &premise));

  premise = {
      .kind = LOOM_PREDICATE_MULTIPLE_OF,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                   LOOM_PRED_ARG_NONE},
      .args = {1, 16, 0},
  };
  conclusion = premise;
  conclusion.args[1] = 8;
  EXPECT_TRUE(loom_value_fact_table_predicate_implies(
      /*table=*/nullptr, &premise, &conclusion));
  EXPECT_FALSE(loom_value_fact_table_predicate_implies(
      /*table=*/nullptr, &conclusion, &premise));

  premise = MakeBinaryPredicate(LOOM_PREDICATE_LT, 1, 2);
  conclusion = MakeBinaryPredicate(LOOM_PREDICATE_LE, 1, 2);
  EXPECT_TRUE(loom_value_fact_table_predicate_implies(
      /*table=*/nullptr, &premise, &conclusion));
  conclusion = MakeBinaryPredicate(LOOM_PREDICATE_GT, 2, 1);
  EXPECT_TRUE(loom_value_fact_table_predicate_implies(
      /*table=*/nullptr, &premise, &conclusion));
  conclusion = MakeBinaryPredicate(LOOM_PREDICATE_ULT, 1, 2);
  EXPECT_FALSE(loom_value_fact_table_predicate_implies(
      /*table=*/nullptr, &premise, &conclusion));

  premise = MakeBinaryPredicate(LOOM_PREDICATE_ULT, 1, 2);
  conclusion = MakeBinaryPredicate(LOOM_PREDICATE_ULE, 1, 2);
  EXPECT_TRUE(loom_value_fact_table_predicate_implies(
      /*table=*/nullptr, &premise, &conclusion));
  conclusion = MakeBinaryPredicate(LOOM_PREDICATE_UGT, 2, 1);
  EXPECT_TRUE(loom_value_fact_table_predicate_implies(
      /*table=*/nullptr, &premise, &conclusion));
  conclusion = MakeBinaryPredicate(LOOM_PREDICATE_NE, 1, 2);
  EXPECT_TRUE(loom_value_fact_table_predicate_implies(
      /*table=*/nullptr, &premise, &conclusion));
  conclusion = MakeBinaryPredicate(LOOM_PREDICATE_LT, 1, 2);
  EXPECT_FALSE(loom_value_fact_table_predicate_implies(
      /*table=*/nullptr, &premise, &conclusion));

  premise = MakeBinaryPredicate(LOOM_PREDICATE_EQ, 1, 2);
  conclusion = MakeBinaryPredicate(LOOM_PREDICATE_ULE, 1, 2);
  EXPECT_TRUE(loom_value_fact_table_predicate_implies(
      /*table=*/nullptr, &premise, &conclusion));
  conclusion = MakeBinaryPredicate(LOOM_PREDICATE_UGE, 1, 2);
  EXPECT_TRUE(loom_value_fact_table_predicate_implies(
      /*table=*/nullptr, &premise, &conclusion));
  conclusion = MakeBinaryPredicate(LOOM_PREDICATE_ULT, 1, 2);
  EXPECT_FALSE(loom_value_fact_table_predicate_implies(
      /*table=*/nullptr, &premise, &conclusion));

  premise = {.kind = LOOM_PREDICATE_FINITE, .arg_count = 1};
  premise.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  premise.args[0] = 1;
  conclusion = {.kind = LOOM_PREDICATE_NOT_NAN, .arg_count = 1};
  conclusion.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  conclusion.args[0] = 1;
  EXPECT_TRUE(loom_value_fact_table_predicate_implies(
      /*table=*/nullptr, &premise, &conclusion));
  EXPECT_FALSE(loom_value_fact_table_predicate_implies(
      /*table=*/nullptr, &conclusion, &premise));
}

TEST(PredicateFactsImplicationTest, IntegerRelationMatrix) {
  struct Relation {
    // Predicate interpreted by the compiler query.
    loom_predicate_kind_t kind;
    // Independent concrete integer semantics.
    bool (*evaluate)(int64_t, int64_t);
  };
  const Relation relations[] = {
      {LOOM_PREDICATE_EQ, [](int64_t a, int64_t b) { return a == b; }},
      {LOOM_PREDICATE_NE, [](int64_t a, int64_t b) { return a != b; }},
      {LOOM_PREDICATE_LT, [](int64_t a, int64_t b) { return a < b; }},
      {LOOM_PREDICATE_LE, [](int64_t a, int64_t b) { return a <= b; }},
      {LOOM_PREDICATE_GT, [](int64_t a, int64_t b) { return a > b; }},
      {LOOM_PREDICATE_GE, [](int64_t a, int64_t b) { return a >= b; }},
      {LOOM_PREDICATE_ULT,
       [](int64_t a, int64_t b) { return uint64_t(a) < uint64_t(b); }},
      {LOOM_PREDICATE_ULE,
       [](int64_t a, int64_t b) { return uint64_t(a) <= uint64_t(b); }},
      {LOOM_PREDICATE_UGT,
       [](int64_t a, int64_t b) { return uint64_t(a) > uint64_t(b); }},
      {LOOM_PREDICATE_UGE,
       [](int64_t a, int64_t b) { return uint64_t(a) >= uint64_t(b); }},
  };
  for (const Relation& premise : relations) {
    for (const Relation& conclusion : relations) {
      for (bool swapped : {false, true}) {
        bool implies = true;
        // These representatives cover both signs and all relative orders.
        for (int64_t left = -2; left <= 2; ++left) {
          for (int64_t right = -2; right <= 2; ++right) {
            implies &= !premise.evaluate(left, right) ||
                       conclusion.evaluate(swapped ? right : left,
                                           swapped ? left : right);
          }
        }
        const auto premise_predicate = MakeBinaryPredicate(premise.kind, 1, 2);
        const auto conclusion_predicate = MakeBinaryPredicate(
            conclusion.kind, swapped ? 2 : 1, swapped ? 1 : 2);
        EXPECT_EQ(
            loom_value_fact_table_predicate_implies(
                /*table=*/nullptr, &premise_predicate, &conclusion_predicate),
            implies)
            << "premise " << int(premise.kind) << " conclusion "
            << int(conclusion.kind) << " swapped " << swapped;
      }
    }
  }
}

}  // namespace
}  // namespace loom
