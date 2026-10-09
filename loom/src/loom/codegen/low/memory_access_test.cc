// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/memory_access.h"

#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbolic_congruence.h"

namespace loom {
namespace {

static loom_low_memory_access_summary_t MakeStridedSummary(
    uint32_t alias_root_id, uint64_t stride_bytes, uint64_t begin_bytes,
    uint64_t end_bytes) {
  return (loom_low_memory_access_summary_t){
      /*.memory_space=*/LOOM_LOW_MEMORY_SPACE_WORKGROUP,
      /*.source_flags=*/0,
      /*.alias_root_id=*/alias_root_id,
      /*.alias_group_id=*/LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      /*.precision_flags=*/LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE |
          LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT |
          LOOM_LOW_MEMORY_ACCESS_PRECISION_STRIDED_INTERVAL,
      /*.strided_interval=*/
      {
          .stride_bytes = stride_bytes,
          .begin_bytes = begin_bytes,
          .end_bytes = end_bytes,
      },
  };
}

static loom_low_memory_access_summary_t MakeIntervalSummary(
    loom_low_byte_interval_t* interval, uint32_t alias_root_id,
    int64_t begin_bytes, int64_t end_bytes) {
  *interval = (loom_low_byte_interval_t){
      /*.begin_facts=*/loom_value_facts_make(begin_bytes, begin_bytes, 1),
      /*.end_facts=*/loom_value_facts_make(end_bytes, end_bytes, 1),
      /*.begin_expr_id=*/LOOM_LOW_MEMORY_EXPR_ID_NONE,
      /*.end_expr_id=*/LOOM_LOW_MEMORY_EXPR_ID_NONE,
      /*.precision_flags=*/LOOM_LOW_BYTE_INTERVAL_PRECISION_BEGIN_RANGE |
          LOOM_LOW_BYTE_INTERVAL_PRECISION_END_RANGE,
  };
  return (loom_low_memory_access_summary_t){
      /*.memory_space=*/LOOM_LOW_MEMORY_SPACE_WORKGROUP,
      /*.source_flags=*/0,
      /*.alias_root_id=*/alias_root_id,
      /*.alias_group_id=*/LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      /*.precision_flags=*/LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE |
          LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT |
          LOOM_LOW_MEMORY_ACCESS_PRECISION_INTERVAL,
      /*.strided_interval=*/{},
      /*.byte_interval=*/interval,
  };
}

TEST(MemoryAccessTest, StridedSlotsWithinOneRootAreDisjoint) {
  const loom_low_memory_access_summary_t slot0 =
      MakeStridedSummary(/*alias_root_id=*/7, /*stride_bytes=*/64,
                         /*begin_bytes=*/0, /*end_bytes=*/16);
  const loom_low_memory_access_summary_t slot1 =
      MakeStridedSummary(/*alias_root_id=*/7, /*stride_bytes=*/64,
                         /*begin_bytes=*/16, /*end_bytes=*/32);
  EXPECT_FALSE(loom_low_memory_access_summaries_may_alias(
      &slot0, &slot1, LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT));
}

TEST(MemoryAccessTest, OverlappingStridedSlotsMayAlias) {
  const loom_low_memory_access_summary_t slot0 =
      MakeStridedSummary(/*alias_root_id=*/7, /*stride_bytes=*/64,
                         /*begin_bytes=*/0, /*end_bytes=*/16);
  const loom_low_memory_access_summary_t overlap =
      MakeStridedSummary(/*alias_root_id=*/7, /*stride_bytes=*/64,
                         /*begin_bytes=*/8, /*end_bytes=*/24);
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      &slot0, &overlap, LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT));
}

TEST(MemoryAccessTest, EqualResiduesMayAliasAcrossStrideInstances) {
  const loom_low_memory_access_summary_t left =
      MakeStridedSummary(/*alias_root_id=*/7, /*stride_bytes=*/64,
                         /*begin_bytes=*/0, /*end_bytes=*/16);
  const loom_low_memory_access_summary_t right = left;
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      &left, &right, LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT));
}

TEST(MemoryAccessTest, StridedProofRequiresComparableRootAndStride) {
  loom_low_memory_access_summary_t no_root =
      MakeStridedSummary(/*alias_root_id=*/7, /*stride_bytes=*/64,
                         /*begin_bytes=*/0, /*end_bytes=*/16);
  no_root.alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE;
  no_root.precision_flags &= ~LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT;
  const loom_low_memory_access_summary_t slot1 =
      MakeStridedSummary(/*alias_root_id=*/7, /*stride_bytes=*/64,
                         /*begin_bytes=*/16, /*end_bytes=*/32);
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      &no_root, &slot1, LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT));

  const loom_low_memory_access_summary_t other_stride =
      MakeStridedSummary(/*alias_root_id=*/7, /*stride_bytes=*/128,
                         /*begin_bytes=*/16, /*end_bytes=*/32);
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      &slot1, &other_stride, LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT));
}

TEST(MemoryAccessTest, IntervalEnvelopeRequiresComparableRoot) {
  loom_low_byte_interval_t left_interval;
  loom_low_byte_interval_t right_interval;
  loom_low_memory_access_summary_t left =
      MakeIntervalSummary(&left_interval, /*alias_root_id=*/11,
                          /*begin_bytes=*/0,
                          /*end_bytes=*/16);
  loom_low_memory_access_summary_t right =
      MakeIntervalSummary(&right_interval, /*alias_root_id=*/11,
                          /*begin_bytes=*/32,
                          /*end_bytes=*/48);
  EXPECT_FALSE(loom_low_memory_access_summaries_may_alias(
      &left, &right, LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT));

  left.alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE;
  left.precision_flags &= ~LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT;
  right.alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE;
  right.precision_flags &= ~LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT;
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      &left, &right, LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT));
}

TEST(MemoryAccessTest, SharedSpaceSummariesPreserveConservativeAliasing) {
  const auto* global =
      loom_low_memory_access_summary_for_space(LOOM_LOW_MEMORY_SPACE_GLOBAL);
  const auto* workgroup =
      loom_low_memory_access_summary_for_space(LOOM_LOW_MEMORY_SPACE_WORKGROUP);
  const auto* generic =
      loom_low_memory_access_summary_for_space(LOOM_LOW_MEMORY_SPACE_NONE);
  loom_low_byte_interval_t interval;
  const auto precise = MakeIntervalSummary(&interval, /*alias_root_id=*/11,
                                           /*begin_bytes=*/32,
                                           /*end_bytes=*/48);

  EXPECT_FALSE(loom_low_memory_access_summaries_may_alias(
      global, workgroup, LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT));
  EXPECT_FALSE(loom_low_memory_access_summaries_may_alias(
      global, &precise, LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT));
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      workgroup, &precise, LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT));
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      generic, &precise, LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT));
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      generic, global, LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT));
}

TEST(MemoryAccessTest, EqualSummariesPreserveIdentityAndFootprintFacts) {
  const auto original = MakeStridedSummary(7, 64, 0, 16);
  auto other = original;
  EXPECT_TRUE(loom_low_memory_access_summaries_equal(&original, &other));
  ++other.alias_root_id;
  EXPECT_FALSE(loom_low_memory_access_summaries_equal(&original, &other));
  other = original;
  other.strided_interval.begin_bytes = 8;
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      &original, &other, LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT));
  EXPECT_FALSE(loom_low_memory_access_summaries_equal(&original, &other));
  other = original;
  other.strided_interval.end_bytes = 32;
  EXPECT_FALSE(loom_low_memory_access_summaries_equal(&original, &other));
  other = original;
  other.strided_interval.stride_bytes = 128;
  EXPECT_FALSE(loom_low_memory_access_summaries_equal(&original, &other));
  other = original;
  other.memory_space = LOOM_LOW_MEMORY_SPACE_GLOBAL;
  EXPECT_FALSE(loom_low_memory_access_summaries_equal(&original, &other));
  other = original;
  other.precision_flags &= ~LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT;
  EXPECT_FALSE(loom_low_memory_access_summaries_equal(&original, &other));
}

TEST(MemoryAccessTest, EqualIntervalsCompareOwnedFactsAndExpressions) {
  loom_low_byte_interval_t left_interval;
  loom_low_byte_interval_t right_interval;
  const auto left = MakeIntervalSummary(&left_interval, 11, 0, 16);
  auto right = MakeIntervalSummary(&right_interval, 11, 0, 16);
  EXPECT_TRUE(loom_low_memory_access_summaries_equal(&left, &right));
  right_interval.end_facts.range_hi = 32;
  EXPECT_FALSE(loom_low_memory_access_summaries_equal(&left, &right));
  right_interval = left_interval;
  right_interval.begin_expr_id = 42;
  EXPECT_FALSE(loom_low_memory_access_summaries_equal(&left, &right));
  right.byte_interval = nullptr;
  EXPECT_FALSE(loom_low_memory_access_summaries_equal(&left, &right));
}

TEST(MemoryAccessTest, EqualSummariesIgnoreAbsentIdentityPayloads) {
  const auto* space =
      loom_low_memory_access_summary_for_space(LOOM_LOW_MEMORY_SPACE_GLOBAL);
  auto other = *space;
  other.alias_root_id = 42;
  other.alias_group_id = 7;
  other.strided_interval = {64, 16, 32};
  EXPECT_TRUE(loom_low_memory_access_summaries_equal(space, &other));

  other.precision_flags |= LOOM_LOW_MEMORY_ACCESS_PRECISION_GROUP;
  auto group = other;
  EXPECT_TRUE(loom_low_memory_access_summaries_equal(&group, &other));
  ++other.alias_group_id;
  EXPECT_FALSE(loom_low_memory_access_summaries_equal(&group, &other));
}

TEST(MemoryAccessTest, EqualAliasSummariesIgnoreSourceSemantics) {
  const auto* space =
      loom_low_memory_access_summary_for_space(LOOM_LOW_MEMORY_SPACE_GLOBAL);
  auto atomic_observation = *space;
  atomic_observation.source_flags =
      LOOM_LOW_MEMORY_ACCESS_SOURCE_FLAG_ATOMIC_OBSERVATION;
  EXPECT_TRUE(
      loom_low_memory_access_summaries_equal(space, &atomic_observation));
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      space, &atomic_observation, LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT));
}

TEST(MemoryAccessTest, DisjointStorageRequiresOneCapturedEvaluation) {
  const int scopes[2] = {};
  loom_low_memory_relative_interval_t intervals[2] = {};
  loom_low_memory_access_summary_t accesses[2] = {};
  for (size_t i = 0; i < 2; ++i) {
    intervals[i].scope = &scopes[0];
    intervals[i].storage_id = i;
    intervals[i].disjoint_storage_ordinal = i + 1;
    intervals[i].upper = 16;
    loom_symbolic_expr_constant(0, &intervals[i].origin);
    accesses[i].memory_space = LOOM_LOW_MEMORY_SPACE_WORKGROUP;
    accesses[i].relative_interval = &intervals[i];
  }
  EXPECT_FALSE(loom_low_memory_access_summaries_may_alias(
      &accesses[0], &accesses[1], LOOM_LOW_MEMORY_COMPARISON_SAME_EVALUATION));
  EXPECT_FALSE(loom_low_memory_access_summaries_may_alias(
      &accesses[0], &accesses[1],
      LOOM_LOW_MEMORY_COMPARISON_SAME_ACYCLIC_INVOCATION));
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      &accesses[0], &accesses[1], LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT));
  intervals[1].scope = &scopes[1];
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      &accesses[0], &accesses[1], LOOM_LOW_MEMORY_COMPARISON_SAME_EVALUATION));
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      &accesses[0], &accesses[1],
      LOOM_LOW_MEMORY_COMPARISON_SAME_ACYCLIC_INVOCATION));
  intervals[1].scope = &scopes[0];
  intervals[1].disjoint_storage_ordinal = 0;
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      &accesses[0], &accesses[1], LOOM_LOW_MEMORY_COMPARISON_SAME_EVALUATION));
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      &accesses[0], &accesses[1],
      LOOM_LOW_MEMORY_COMPARISON_SAME_ACYCLIC_INVOCATION));
}

TEST(MemoryAccessTest, PeriodicBanksRequireOneEvaluationAndCompleteEnvelopes) {
  const int scope = 0;
  const loom_symbolic_term_t phase = {32768, 7, 7};
  const loom_symbolic_term_t exact_values[] = {{1, 8, 8}, {1, 9, 9}};
  loom_symbolic_congruence_t periodic[2] = {};
  loom_low_memory_relative_interval_t intervals[2] = {};
  loom_low_memory_access_summary_t accesses[2] = {};
  for (size_t i = 0; i < 2; ++i) {
    periodic[i].modulus = 65536;
    periodic[i].expression.constant = i * 32768;
    periodic[i].expression.terms = &phase;
    periodic[i].expression.term_count = 1;
    periodic[i].expression.flags = LOOM_SYMBOLIC_EXPR_FLAG_LINEAR;
    auto& interval = intervals[i];
    interval.scope = &scope;
    interval.storage_id = 4;
    interval.origin.terms = &exact_values[i];
    interval.origin.term_count = 1;
    interval.origin.flags = LOOM_SYMBOLIC_EXPR_FLAG_LINEAR;
    interval.origin.facts = loom_value_facts_make(0, 32768, 32768);
    interval.origin.congruence = &periodic[i];
    interval.upper = 32240;
    accesses[i].memory_space = LOOM_LOW_MEMORY_SPACE_WORKGROUP;
    accesses[i].relative_interval = &interval;
  }
  EXPECT_FALSE(loom_low_memory_access_summaries_may_alias(
      &accesses[0], &accesses[1], LOOM_LOW_MEMORY_COMPARISON_SAME_EVALUATION));
  // A following iteration reads the bank published by this iteration.
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      &accesses[0], &accesses[1], LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT));
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      &accesses[0], &accesses[0], LOOM_LOW_MEMORY_COMPARISON_SAME_EVALUATION));
  intervals[1].upper = 32769;
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      &accesses[0], &accesses[1], LOOM_LOW_MEMORY_COMPARISON_SAME_EVALUATION));
  intervals[1].upper = 32240;
  intervals[1].storage_id = 5;
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      &accesses[0], &accesses[1], LOOM_LOW_MEMORY_COMPARISON_SAME_EVALUATION));
  intervals[1].storage_id = 4;
  intervals[1].scope = &intervals;
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      &accesses[0], &accesses[1], LOOM_LOW_MEMORY_COMPARISON_SAME_EVALUATION));
}

TEST(MemoryAccessTest, AffinePingPongBanksShareOneParityIdentity) {
  constexpr int64_t kStageBytes = 34048;
  constexpr int64_t kFootprintBytes = 32240;
  constexpr loom_value_id_t kParityValue = 17;
  const int scope = 0;
  const loom_symbolic_term_t terms[2] = {
      {.coefficient = kStageBytes,
       .value_id = kParityValue,
       .relation_value_id = kParityValue},
      {.coefficient = -kStageBytes,
       .value_id = kParityValue,
       .relation_value_id = kParityValue},
  };
  loom_low_memory_relative_interval_t intervals[2] = {};
  loom_low_memory_access_summary_t accesses[2] = {};
  for (size_t i = 0; i < 2; ++i) {
    intervals[i].scope = &scope;
    intervals[i].storage_id = 4;
    intervals[i].origin.constant = i == 0 ? 0 : kStageBytes;
    intervals[i].origin.terms = &terms[i];
    intervals[i].origin.term_count = 1;
    intervals[i].origin.flags = LOOM_SYMBOLIC_EXPR_FLAG_LINEAR;
    intervals[i].upper = kFootprintBytes;
    accesses[i].memory_space = LOOM_LOW_MEMORY_SPACE_WORKGROUP;
    accesses[i].relative_interval = &intervals[i];
  }

  // active = 34048 * parity
  // inactive = 34048 - 34048 * parity
  EXPECT_FALSE(loom_low_memory_access_summaries_may_alias(
      &accesses[0], &accesses[1], LOOM_LOW_MEMORY_COMPARISON_SAME_EVALUATION));
  EXPECT_FALSE(loom_low_memory_access_summaries_may_alias(
      &accesses[0], &accesses[1],
      LOOM_LOW_MEMORY_COMPARISON_SAME_ACYCLIC_INVOCATION));
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      &accesses[0], &accesses[1], LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT));
}

TEST(MemoryAccessTest, AffinePingPongBanksRequireTheSameParityIdentity) {
  constexpr int64_t kStageBytes = 34048;
  const int scope = 0;
  const loom_symbolic_term_t terms[2] = {
      {.coefficient = kStageBytes, .value_id = 17, .relation_value_id = 17},
      {.coefficient = -kStageBytes, .value_id = 18, .relation_value_id = 18},
  };
  loom_low_memory_relative_interval_t intervals[2] = {};
  loom_low_memory_access_summary_t accesses[2] = {};
  for (size_t i = 0; i < 2; ++i) {
    intervals[i].scope = &scope;
    intervals[i].storage_id = 4;
    intervals[i].origin.constant = i == 0 ? 0 : kStageBytes;
    intervals[i].origin.terms = &terms[i];
    intervals[i].origin.term_count = 1;
    intervals[i].origin.flags = LOOM_SYMBOLIC_EXPR_FLAG_LINEAR;
    intervals[i].upper = 32240;
    accesses[i].memory_space = LOOM_LOW_MEMORY_SPACE_WORKGROUP;
    accesses[i].relative_interval = &intervals[i];
  }

  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      &accesses[0], &accesses[1], LOOM_LOW_MEMORY_COMPARISON_SAME_EVALUATION));
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      &accesses[0], &accesses[1],
      LOOM_LOW_MEMORY_COMPARISON_SAME_ACYCLIC_INVOCATION));
}

class MemoryAccessMapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &source_arena_);
    iree_arena_initialize(&pool_, &target_arena_);
    iree_arena_initialize(&pool_, &scratch_arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_deinitialize(&target_arena_);
    iree_arena_deinitialize(&source_arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }
  // Shared block allocator; each arena has independent reset boundaries.
  iree_arena_block_pool_t pool_;
  // Original compilation's payload lifetime.
  iree_arena_allocator_t source_arena_;
  // Surviving cloned compilation's payload lifetime.
  iree_arena_allocator_t target_arena_;
  // Transient clone correspondence lifetime.
  iree_arena_allocator_t scratch_arena_;
};

TEST_F(MemoryAccessMapTest,
       EffectBindingsSurviveReplacementAndIndependentClones) {
  loom_low_memory_access_map_t* source = nullptr;
  loom_low_memory_access_map_t* target = nullptr;
  IREE_ASSERT_OK(loom_low_memory_access_map_create(&source_arena_, &source));
  IREE_ASSERT_OK(loom_low_memory_access_map_create(&target_arena_, &target));
  loom_op_t packets[4] = {};
  const loom_symbolic_term_t term = {4, 7, 7};
  loom_symbolic_congruence_t periodic = {};
  periodic.modulus = 64;
  periodic.expression.terms = &term;
  periodic.expression.term_count = 1;
  periodic.expression.flags = LOOM_SYMBOLIC_EXPR_FLAG_LINEAR;
  loom_low_memory_relative_interval_t interval = {};
  interval.scope = source;
  interval.storage_id = 2;
  interval.disjoint_storage_ordinal = 3;
  interval.origin = periodic.expression;
  interval.origin.facts = loom_value_facts_unknown();
  interval.origin.congruence = &periodic;
  interval.upper = 16;
  loom_low_memory_access_summary_t access = {};
  access.memory_space = LOOM_LOW_MEMORY_SPACE_WORKGROUP;
  access.source_flags = LOOM_LOW_MEMORY_ACCESS_SOURCE_FLAG_ATOMIC_OBSERVATION;
  access.relative_interval = &interval;
  IREE_ASSERT_OK(
      loom_low_memory_access_map_insert(source, &packets[0], 0, &access));
  interval.lower = 16;
  interval.upper = 32;
  IREE_ASSERT_OK(
      loom_low_memory_access_map_insert(source, &packets[0], 2, &access));
  interval.storage_id = 3;
  interval.disjoint_storage_ordinal = 4;
  interval.lower = 0;
  interval.upper = 16;
  IREE_ASSERT_OK(
      loom_low_memory_access_map_insert(source, &packets[0], 4, &access));
  EXPECT_EQ(loom_low_memory_access_map_lookup(source, &packets[0], 1), nullptr);
  IREE_ASSERT_OK(
      loom_low_memory_access_map_replace(source, &packets[0], &packets[1]));
  EXPECT_EQ(loom_low_memory_access_map_lookup(source, &packets[0], 2),
            loom_low_memory_access_map_lookup(source, &packets[1], 2));
  for (size_t i = 2; i < 4; ++i) {
    loom_low_memory_access_clone_t* clone = nullptr;
    IREE_ASSERT_OK(loom_low_memory_access_clone_create(
        source, target, &scratch_arena_, &clone));
    IREE_ASSERT_OK(
        loom_low_memory_access_clone_op(clone, &packets[1], &packets[i]));
    iree_arena_reset(&scratch_arena_);
  }
  iree_arena_reset(&source_arena_);
  const auto* left = loom_low_memory_access_map_lookup(target, &packets[2], 0);
  const auto* right = loom_low_memory_access_map_lookup(target, &packets[2], 2);
  const auto* other_call =
      loom_low_memory_access_map_lookup(target, &packets[3], 2);
  ASSERT_NE(left, nullptr);
  ASSERT_NE(right, nullptr);
  ASSERT_NE(other_call, nullptr);
  EXPECT_EQ(left->source_flags,
            LOOM_LOW_MEMORY_ACCESS_SOURCE_FLAG_ATOMIC_OBSERVATION);
  EXPECT_EQ(left->relative_interval->origin.terms[0].coefficient, 4);
  EXPECT_EQ(
      left->relative_interval->origin.congruence->expression.terms[0].value_id,
      7u);
  EXPECT_FALSE(loom_low_memory_access_summaries_may_alias(
      left, right, LOOM_LOW_MEMORY_COMPARISON_SAME_EVALUATION));
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      left, other_call, LOOM_LOW_MEMORY_COMPARISON_SAME_EVALUATION));
  const auto* separate_storage =
      loom_low_memory_access_map_lookup(target, &packets[2], 4);
  const auto* separate_call_storage =
      loom_low_memory_access_map_lookup(target, &packets[3], 4);
  ASSERT_NE(separate_storage, nullptr);
  ASSERT_NE(separate_call_storage, nullptr);
  EXPECT_EQ(separate_storage->relative_interval->disjoint_storage_ordinal, 4u);
  EXPECT_FALSE(loom_low_memory_access_summaries_may_alias(
      left, separate_storage, LOOM_LOW_MEMORY_COMPARISON_SAME_EVALUATION));
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      left, separate_call_storage, LOOM_LOW_MEMORY_COMPARISON_SAME_EVALUATION));
  loom_low_memory_access_map_t* moved = nullptr;
  IREE_ASSERT_OK(loom_low_memory_access_map_create(&target_arena_, &moved));
  IREE_ASSERT_OK(loom_low_memory_access_map_transfer(target, moved));
  EXPECT_EQ(loom_low_memory_access_map_lookup(moved, &packets[2], 0), left);
}

}  // namespace
}  // namespace loom
