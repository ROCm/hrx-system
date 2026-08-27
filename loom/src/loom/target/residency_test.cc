// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/residency.h"

#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

static const iree_string_view_t kDirectResourceNames[] = {
    IREE_SVL("vector"),
    IREE_SVL("scalar"),
};

static const loom_target_residency_cliff_t kDirectResourceCliffs[] = {
    {
        /*.resource_id=*/0,
        /*.cliff_units=*/5,
        /*.tier_before=*/4,
        /*.tier_after=*/2,
    },
    {
        /*.resource_id=*/0,
        /*.cliff_units=*/9,
        /*.tier_before=*/2,
        /*.tier_after=*/1,
    },
    {
        /*.resource_id=*/1,
        /*.cliff_units=*/3,
        /*.tier_before=*/4,
        /*.tier_after=*/3,
    },
    {
        /*.resource_id=*/1,
        /*.cliff_units=*/7,
        /*.tier_before=*/3,
        /*.tier_after=*/2,
    },
    {
        /*.resource_id=*/1,
        /*.cliff_units=*/11,
        /*.tier_before=*/2,
        /*.tier_after=*/0,
    },
};

static const loom_target_residency_cliff_range_t kDirectResourceCliffRanges[] =
    {
        {/*.start=*/0, /*.count=*/2},
        {/*.start=*/2, /*.count=*/3},
};

static const loom_target_residency_derived_member_t kDerivedMembers[] = {
    {
        /*.resource_id=*/0,
        /*.direct_resource_id=*/0,
        /*.contribution_granularity=*/4,
    },
    {
        /*.resource_id=*/0,
        /*.direct_resource_id=*/1,
        /*.contribution_granularity=*/2,
    },
};

static const loom_target_residency_cliff_t kDerivedResourceCliffs[] = {
    {
        /*.resource_id=*/0,
        /*.cliff_units=*/9,
        /*.tier_before=*/4,
        /*.tier_after=*/2,
    },
    {
        /*.resource_id=*/0,
        /*.cliff_units=*/17,
        /*.tier_before=*/2,
        /*.tier_after=*/1,
    },
};

static const loom_target_residency_derived_resource_t kDerivedResources[] = {
    {
        /*.name=*/IREE_SVL("shared_register_file"),
        /*.pool_units=*/64,
        /*.allocation_granularity=*/1,
        /*.member_start=*/0,
        /*.member_count=*/2,
        /*.cliff_start=*/0,
        /*.cliff_count=*/2,
    },
};

static const uint16_t kMemberIndicesByDirectResource[] = {0, 1};

static const loom_target_residency_derived_member_range_t
    kMemberRangesByDirectResource[] = {
        {/*.start=*/0, /*.count=*/1},
        {/*.start=*/1, /*.count=*/1},
};

static const loom_target_residency_model_t kModel = {
    /*.best_tier=*/4,
    /*.direct_resources=*/
    {
        /*.names=*/kDirectResourceNames,
        /*.cliffs=*/kDirectResourceCliffs,
        /*.cliff_count=*/IREE_ARRAYSIZE(kDirectResourceCliffs),
        /*.cliff_ranges=*/kDirectResourceCliffRanges,
        /*.resource_count=*/IREE_ARRAYSIZE(kDirectResourceNames),
    },
    /*.derived_resources=*/
    {
        /*.resources=*/kDerivedResources,
        /*.resource_count=*/IREE_ARRAYSIZE(kDerivedResources),
        /*.members=*/kDerivedMembers,
        /*.member_count=*/IREE_ARRAYSIZE(kDerivedMembers),
        /*.cliffs=*/kDerivedResourceCliffs,
        /*.cliff_count=*/IREE_ARRAYSIZE(kDerivedResourceCliffs),
        /*.member_indices_by_direct_resource=*/
        kMemberIndicesByDirectResource,
        /*.member_ranges_by_direct_resource=*/
        kMemberRangesByDirectResource,
    },
};

class ResidencyTest : public ::testing::Test {
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

  loom_target_residency_query_t Query(uint64_t vector_units,
                                      uint64_t scalar_units) {
    const uint64_t direct_units[] = {vector_units, scalar_units};
    loom_target_residency_query_t query;
    IREE_EXPECT_OK(loom_target_residency_query(
        &kModel, direct_units, IREE_ARRAYSIZE(direct_units), &arena_, &query));
    return query;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(ResidencyTest, CliffEvaluationUsesExactBoundaries) {
  loom_target_residency_cliff_evaluation_t evaluation;
  loom_target_residency_evaluate_cliffs(kDirectResourceCliffs, 2, 4, 4,
                                        &evaluation);
  EXPECT_EQ(evaluation.tier, 4u);
  EXPECT_EQ(evaluation.worse_tier, 2u);
  EXPECT_EQ(evaluation.additional_units_to_worse_tier, 1u);

  loom_target_residency_evaluate_cliffs(kDirectResourceCliffs, 2, 4, 5,
                                        &evaluation);
  EXPECT_EQ(evaluation.tier, 2u);
  EXPECT_EQ(evaluation.better_tier, 4u);
  EXPECT_EQ(evaluation.reduction_units_to_better_tier, 1u);
  EXPECT_EQ(evaluation.worse_tier, 1u);
  EXPECT_EQ(evaluation.additional_units_to_worse_tier, 4u);

  loom_target_residency_evaluate_cliffs(kDirectResourceCliffs, 2, 4, 9,
                                        &evaluation);
  EXPECT_EQ(evaluation.tier, 1u);
  EXPECT_EQ(evaluation.better_tier, 2u);
  EXPECT_EQ(evaluation.reduction_units_to_better_tier, 1u);
  EXPECT_FALSE(iree_any_bit_set(
      evaluation.flags,
      LOOM_TARGET_RESIDENCY_CLIFF_EVALUATION_FLAG_HAS_WORSE_TIER));
}

TEST_F(ResidencyTest, DirectResourceOverrideEvaluatesWholeModelTier) {
  const uint32_t direct_units[] = {4, 2};
  EXPECT_EQ(loom_target_residency_evaluate_tier_with_direct_resource_override(
                &kModel, direct_units, /*direct_resource_id=*/0,
                /*override_units=*/4),
            4u);
  EXPECT_EQ(loom_target_residency_evaluate_tier_with_direct_resource_override(
                &kModel, direct_units, /*direct_resource_id=*/0,
                /*override_units=*/5),
            2u);

  // The scalar resource remains at tier 3, but its rounded contribution makes
  // the derived shared register file the tier-2 limiter.
  EXPECT_EQ(loom_target_residency_evaluate_tier_with_direct_resource_override(
                &kModel, direct_units, /*direct_resource_id=*/1,
                /*override_units=*/5),
            2u);
}

TEST_F(ResidencyTest, UnavailableModelIsExplicit) {
  loom_target_residency_query_t query;
  IREE_ASSERT_OK(
      loom_target_residency_query(nullptr, nullptr, 0, nullptr, &query));
  EXPECT_FALSE(query.model_available);
  EXPECT_EQ(query.resource_count, 0u);
  EXPECT_EQ(query.tier, 0u);
}

TEST_F(ResidencyTest, AvailableCliffFreeModelKeepsBestTier) {
  const loom_target_residency_model_t empty_model = {
      /*.best_tier=*/7,
  };
  loom_target_residency_query_t query;
  IREE_ASSERT_OK(
      loom_target_residency_query(&empty_model, nullptr, 0, nullptr, &query));
  EXPECT_TRUE(query.model_available);
  EXPECT_EQ(query.best_tier, 7u);
  EXPECT_EQ(query.tier, 7u);
  EXPECT_FALSE(query.has_next_better_tier);
  EXPECT_EQ(query.resource_count, 0u);
}

TEST_F(ResidencyTest, BestTierReportsPerResourceHeadroom) {
  const loom_target_residency_query_t query = Query(0, 0);
  ASSERT_TRUE(query.model_available);
  ASSERT_EQ(query.resource_count, 3u);
  EXPECT_EQ(query.tier, 4u);
  EXPECT_EQ(query.limiting_resource_count, 0u);
  EXPECT_FALSE(query.has_next_better_tier);

  EXPECT_EQ(query.resources[0].additional_units_to_next_worse_tier, 5u);
  EXPECT_EQ(query.resources[0].next_worse_tier, 2u);
  EXPECT_EQ(query.resources[1].additional_units_to_next_worse_tier, 3u);
  EXPECT_EQ(query.resources[1].next_worse_tier, 3u);
  EXPECT_EQ(query.resources[2].units, 0u);
  EXPECT_EQ(query.resources[2].additional_units_to_next_worse_tier, 9u);
  EXPECT_EQ(query.resources[2].next_worse_tier, 2u);
}

TEST_F(ResidencyTest, AllLimitersShareOneRecoveryTarget) {
  const loom_target_residency_query_t query = Query(5, 7);
  ASSERT_EQ(query.resource_count, 3u);
  EXPECT_EQ(query.tier, 2u);
  EXPECT_EQ(query.limiting_resource_count, 3u);
  EXPECT_TRUE(query.has_next_better_tier);
  EXPECT_EQ(query.next_better_tier, 3u);

  EXPECT_EQ(query.resources[0].units, 5u);
  EXPECT_EQ(query.resources[0].tier, 2u);
  EXPECT_EQ(query.resources[0].reduction_units_to_next_better_tier, 1u);
  EXPECT_EQ(query.resources[0].additional_units_to_next_worse_tier, 4u);
  EXPECT_EQ(query.resources[0].next_worse_tier, 1u);

  EXPECT_EQ(query.resources[1].units, 7u);
  EXPECT_EQ(query.resources[1].tier, 2u);
  EXPECT_EQ(query.resources[1].reduction_units_to_next_better_tier, 1u);
  EXPECT_EQ(query.resources[1].additional_units_to_next_worse_tier, 4u);
  EXPECT_EQ(query.resources[1].next_worse_tier, 0u);

  EXPECT_EQ(query.resources[2].units, 16u);
  EXPECT_EQ(query.resources[2].tier, 2u);
  EXPECT_EQ(query.resources[2].reduction_units_to_next_better_tier, 8u);
  EXPECT_EQ(query.resources[2].additional_units_to_next_worse_tier, 1u);
  EXPECT_EQ(query.resources[2].next_worse_tier, 1u);
}

TEST_F(ResidencyTest, SaturatingDerivedUnitsPreserveWorstTier) {
  const loom_target_residency_query_t query = Query(UINT64_MAX, UINT64_MAX);
  ASSERT_EQ(query.resource_count, 3u);
  EXPECT_EQ(query.resources[2].units, UINT64_MAX);
  EXPECT_EQ(query.tier, 0u);
  EXPECT_EQ(query.limiting_resource_count, 1u);
  EXPECT_EQ(query.next_better_tier, 1u);
  EXPECT_EQ(query.resources[0].reduction_units_to_next_better_tier, 0u);
  EXPECT_EQ(query.resources[1].reduction_units_to_next_better_tier,
            UINT64_MAX - 10u);
  EXPECT_EQ(query.resources[2].reduction_units_to_next_better_tier, 0u);
}

TEST_F(ResidencyTest, DirectResourceVectorMustMatchModel) {
  const uint64_t direct_units[] = {1};
  loom_target_residency_query_t query;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_target_residency_query(
                            &kModel, direct_units, IREE_ARRAYSIZE(direct_units),
                            &arena_, &query));
}

}  // namespace
}  // namespace loom
