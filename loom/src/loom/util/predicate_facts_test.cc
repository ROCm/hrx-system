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

TEST_F(PredicateFactsTest, SmallListStaysAllocationFree) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 1));
  const loom_value_id_t value = 0;
  loom_value_facts_t facts = loom_value_facts_unknown();
  loom_predicate_t predicate = {};
  predicate.kind = LOOM_PREDICATE_RANGE;
  predicate.arg_count = 3;
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
  EXPECT_EQ(table.scratch.alias_ordinals.values, nullptr);
  EXPECT_EQ(table.scratch.alias_ordinals.capacity, 0u);
}

TEST_F(PredicateFactsTest, LargeDisorderedListUsesReusableDirectOrdinalMap) {
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
    predicate->kind = LOOM_PREDICATE_LE;
    predicate->arg_count = 2;
    predicate->arg_tags[0] = LOOM_PRED_ARG_VALUE;
    predicate->arg_tags[1] = LOOM_PRED_ARG_VALUE;
    predicate->args[0] = values[kTargetOrdinal - i];
    predicate->args[1] = kExternalBound;
  }

  IREE_ASSERT_OK(loom_value_fact_table_apply_alias_predicates(
      &table, values.data(), kAliasCount, predicates,
      IREE_ARRAYSIZE(predicates), facts.data()));

  for (uint16_t i = 0; i < IREE_ARRAYSIZE(predicates); ++i) {
    EXPECT_EQ(facts[kTargetOrdinal - i].range_lo, INT64_MIN);
    EXPECT_EQ(facts[kTargetOrdinal - i].range_hi, 65535);
  }
  ASSERT_NE(table.scratch.alias_ordinals.values, nullptr);
  for (loom_value_id_t value_id : values) {
    EXPECT_EQ(table.scratch.alias_ordinals.values[value_id], 0);
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

}  // namespace
}  // namespace loom
