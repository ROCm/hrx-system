// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/representation_plan.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class RepresentationPlanTest : public ::testing::Test {
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

  loom_low_representation_plan_t MakePlan(loom_value_ordinal_t value_count) {
    loom_low_representation_plan_t plan;
    loom_low_representation_plan_initialize(value_count, &arena_, &plan);
    return plan;
  }

  static loom_low_representation_candidate_t Candidate(
      loom_low_representation_id_t representation, uint32_t runtime,
      uint32_t code_size = 0) {
    return {
        .representation = representation,
        .cost = {.runtime = runtime, .code_size = code_size},
    };
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(RepresentationPlanTest, UnconstrainedValuesRemainUnselected) {
  auto plan = MakePlan(1024);
  IREE_ASSERT_OK(loom_low_representation_plan_union(&plan, 7, 999));
  EXPECT_EQ(plan.node_count, 2u);
  EXPECT_TRUE(loom_low_representation_plan_solve(&plan, nullptr));

  loom_low_representation_id_t representation = 0;
  EXPECT_FALSE(loom_low_representation_plan_lookup(&plan, 7, &representation));
  EXPECT_EQ(representation, LOOM_LOW_REPRESENTATION_ID_NONE);
  EXPECT_FALSE(
      loom_low_representation_plan_lookup(&plan, 500, &representation));
}

TEST_F(RepresentationPlanTest, IntersectsDomainsAcrossUnionedValues) {
  auto plan = MakePlan(4);
  int producer = 0;
  int consumer = 0;
  const loom_low_representation_candidate_t producer_candidates[] = {
      Candidate(10, 1), Candidate(20, 2)};
  const loom_low_representation_candidate_t consumer_candidates[] = {
      Candidate(20, 3), Candidate(30, 1)};
  IREE_ASSERT_OK(loom_low_representation_plan_constrain(
      &plan, 0, &producer, producer_candidates,
      IREE_ARRAYSIZE(producer_candidates)));
  IREE_ASSERT_OK(loom_low_representation_plan_constrain(
      &plan, 1, &consumer, consumer_candidates,
      IREE_ARRAYSIZE(consumer_candidates)));
  IREE_ASSERT_OK(loom_low_representation_plan_union(&plan, 0, 1));
  EXPECT_TRUE(loom_low_representation_plan_solve(&plan, nullptr));

  loom_low_representation_id_t representation = 0;
  EXPECT_TRUE(loom_low_representation_plan_lookup(&plan, 0, &representation));
  EXPECT_EQ(representation, 20u);
  EXPECT_TRUE(loom_low_representation_plan_lookup(&plan, 1, &representation));
  EXPECT_EQ(representation, 20u);
}

TEST_F(RepresentationPlanTest, SumsComponentRuntimeCosts) {
  auto plan = MakePlan(3);
  const loom_low_representation_candidate_t first_candidates[] = {
      Candidate(10, 1), Candidate(20, 3)};
  const loom_low_representation_candidate_t second_candidates[] = {
      Candidate(10, 5), Candidate(20, 1)};
  IREE_ASSERT_OK(loom_low_representation_plan_constrain(
      &plan, 0, nullptr, first_candidates, IREE_ARRAYSIZE(first_candidates)));
  IREE_ASSERT_OK(loom_low_representation_plan_constrain(
      &plan, 2, nullptr, second_candidates, IREE_ARRAYSIZE(second_candidates)));
  IREE_ASSERT_OK(loom_low_representation_plan_union(&plan, 0, 2));
  EXPECT_TRUE(loom_low_representation_plan_solve(&plan, nullptr));

  loom_low_representation_id_t representation = 0;
  EXPECT_TRUE(loom_low_representation_plan_lookup(&plan, 0, &representation));
  EXPECT_EQ(representation, 20u);
}

TEST_F(RepresentationPlanTest, BreaksRuntimeTiesByCodeSizeThenIdentity) {
  auto plan = MakePlan(2);
  const loom_low_representation_candidate_t candidates[] = {
      Candidate(30, 4, 1), Candidate(20, 4, 1), Candidate(10, 4, 2)};
  IREE_ASSERT_OK(loom_low_representation_plan_constrain(
      &plan, 1, nullptr, candidates, IREE_ARRAYSIZE(candidates)));
  EXPECT_TRUE(loom_low_representation_plan_solve(&plan, nullptr));

  loom_low_representation_id_t representation = 0;
  EXPECT_TRUE(loom_low_representation_plan_lookup(&plan, 1, &representation));
  EXPECT_EQ(representation, 20u);
}

TEST_F(RepresentationPlanTest, ReportsConstraintThatEmptiesDomain) {
  auto plan = MakePlan(3);
  int first_owner = 0;
  int narrowing_owner = 0;
  int emptying_owner = 0;
  const loom_low_representation_candidate_t first_candidates[] = {
      Candidate(20, 0), Candidate(10, 0)};
  const loom_low_representation_candidate_t narrowing_candidates[] = {
      Candidate(10, 0)};
  const loom_low_representation_candidate_t emptying_candidates[] = {
      Candidate(20, 0)};
  IREE_ASSERT_OK(loom_low_representation_plan_constrain(
      &plan, 0, &first_owner, first_candidates,
      IREE_ARRAYSIZE(first_candidates)));
  IREE_ASSERT_OK(loom_low_representation_plan_constrain(
      &plan, 1, &narrowing_owner, narrowing_candidates,
      IREE_ARRAYSIZE(narrowing_candidates)));
  IREE_ASSERT_OK(loom_low_representation_plan_constrain(
      &plan, 2, &emptying_owner, emptying_candidates,
      IREE_ARRAYSIZE(emptying_candidates)));
  IREE_ASSERT_OK(loom_low_representation_plan_union(&plan, 0, 1));
  IREE_ASSERT_OK(loom_low_representation_plan_union(&plan, 1, 2));

  loom_low_representation_conflict_t conflict = {};
  EXPECT_FALSE(loom_low_representation_plan_solve(&plan, &conflict));
  EXPECT_EQ(conflict.first_owner, &first_owner);
  EXPECT_EQ(conflict.incompatible_owner, &emptying_owner);
}

TEST_F(RepresentationPlanTest, KeepsIndependentComponentsIndependent) {
  auto plan = MakePlan(4);
  const loom_low_representation_candidate_t left_candidates[] = {
      Candidate(10, 0)};
  const loom_low_representation_candidate_t right_candidates[] = {
      Candidate(20, 0)};
  IREE_ASSERT_OK(loom_low_representation_plan_constrain(
      &plan, 0, nullptr, left_candidates, IREE_ARRAYSIZE(left_candidates)));
  IREE_ASSERT_OK(loom_low_representation_plan_constrain(
      &plan, 3, nullptr, right_candidates, IREE_ARRAYSIZE(right_candidates)));
  EXPECT_TRUE(loom_low_representation_plan_solve(&plan, nullptr));

  loom_low_representation_id_t representation = 0;
  EXPECT_TRUE(loom_low_representation_plan_lookup(&plan, 0, &representation));
  EXPECT_EQ(representation, 10u);
  EXPECT_TRUE(loom_low_representation_plan_lookup(&plan, 3, &representation));
  EXPECT_EQ(representation, 20u);
}

}  // namespace
