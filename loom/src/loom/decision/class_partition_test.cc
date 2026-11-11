// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/decision/class_partition.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class DecisionClassPartitionTest : public ::testing::Test {
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

TEST_F(DecisionClassPartitionTest, CorrelatedOutcomesFormObservedQuotient) {
  constexpr iree_host_size_t kSiteCount = 1000;
  constexpr loom_decision_class_ordinal_t kClassLimit = 128;
  loom_decision_class_partition_t partition;
  IREE_ASSERT_OK(loom_decision_class_partition_initialize(
      kSiteCount, kClassLimit, /*maximum_outcome_count=*/2, &arena_,
      &partition));

  for (uint32_t decision = 0; decision < 16; ++decision) {
    loom_decision_class_partition_begin(&partition, /*outcome_count=*/2);
    for (iree_host_size_t site = 0; site < kSiteCount; ++site) {
      ASSERT_TRUE(loom_decision_class_partition_record(&partition, site,
                                                       site >= 128 ? 1 : 0));
    }
    EXPECT_EQ(partition.candidate_class_count, 2);
    loom_decision_class_partition_commit(&partition);
  }

  EXPECT_EQ(partition.class_count, 2);
  for (iree_host_size_t site = 0; site < kSiteCount; ++site) {
    EXPECT_EQ(partition.site_classes[site], site >= 128 ? 1 : 0);
  }
}

TEST_F(DecisionClassPartitionTest,
       IndependentOutcomesStopAtLimitWithoutMutation) {
  constexpr iree_host_size_t kSiteCount = 1000;
  constexpr loom_decision_class_ordinal_t kClassLimit = 128;
  loom_decision_class_partition_t partition;
  IREE_ASSERT_OK(loom_decision_class_partition_initialize(
      kSiteCount, kClassLimit, /*maximum_outcome_count=*/2, &arena_,
      &partition));

  for (uint32_t decision = 0; decision < 7; ++decision) {
    loom_decision_class_partition_begin(&partition, /*outcome_count=*/2);
    for (iree_host_size_t site = 0; site < kSiteCount; ++site) {
      ASSERT_TRUE(loom_decision_class_partition_record(&partition, site,
                                                       (site >> decision) & 1));
    }
    loom_decision_class_partition_commit(&partition);
  }
  ASSERT_EQ(partition.class_count, kClassLimit);

  loom_decision_class_ordinal_t classes_before[kSiteCount];
  memcpy(classes_before, partition.site_classes, sizeof(classes_before));
  loom_decision_class_partition_begin(&partition, /*outcome_count=*/2);
  bool within_limit = true;
  for (iree_host_size_t site = 0; site < kSiteCount; ++site) {
    if (!loom_decision_class_partition_record(&partition, site,
                                              (site >> 7) & 1)) {
      within_limit = false;
      break;
    }
  }
  EXPECT_FALSE(within_limit);
  EXPECT_EQ(partition.class_count, kClassLimit);
  EXPECT_EQ(
      memcmp(classes_before, partition.site_classes, sizeof(classes_before)),
      0);
}

TEST_F(DecisionClassPartitionTest, MatchesObservedPairsAcrossVocabularySizes) {
  for (iree_host_size_t site_count : {1, 2, 7, 128, 257, 1024}) {
    for (loom_decision_class_ordinal_t class_limit : {1, 3, 128}) {
      for (uint32_t maximum_outcomes : {1u, 2u, 64u, 65536u, UINT32_MAX}) {
        SCOPED_TRACE(site_count);
        SCOPED_TRACE(class_limit);
        SCOPED_TRACE(maximum_outcomes);
        iree_arena_reset(&arena_);
        loom_decision_class_partition_t partition;
        IREE_ASSERT_OK(loom_decision_class_partition_initialize(
            site_count, class_limit, maximum_outcomes, &arena_, &partition));
        const auto initialized_bytes = arena_.used_allocation_size;
        std::vector<loom_decision_class_ordinal_t> committed(site_count, 0);
        for (uint32_t round = 0; round < 16; ++round) {
          SCOPED_TRACE(round);
          // Alternating small and wide vocabularies exercises the same backing
          // through repeated refinement, abandonment, and exact retry.
          const uint32_t outcome_count =
              round % 3 ? maximum_outcomes : iree_min(maximum_outcomes, 2u);
          const uint32_t cohort_count =
              iree_min(outcome_count, round % 2 ? 7u : 128u);
          const uint32_t stride =
              cohort_count > 1 ? (outcome_count - 1) / (cohort_count - 1) : 0;
          loom_decision_class_partition_begin(&partition, outcome_count);
          using Pair = std::pair<loom_decision_class_ordinal_t, uint32_t>;
          std::map<Pair, loom_decision_class_ordinal_t> observed;
          std::vector<loom_decision_class_ordinal_t> candidate(site_count);
          bool fits = true;
          for (iree_host_size_t site = 0; site < site_count; ++site) {
            const uint32_t outcome =
                outcome_count - 1 - ((site + round) % cohort_count) * stride;
            const Pair pair(committed[site], outcome);
            auto found = observed.find(pair);
            if (found == observed.end()) {
              if (observed.size() == class_limit) {
                EXPECT_FALSE(loom_decision_class_partition_record(
                    &partition, site, outcome));
                fits = false;
                break;
              }
              found = observed.emplace(pair, observed.size()).first;
            }
            ASSERT_TRUE(loom_decision_class_partition_record(&partition, site,
                                                             outcome));
            candidate[site] = found->second;
            EXPECT_EQ(partition.candidate_site_classes[site], found->second);
            EXPECT_EQ(partition.candidate_parent_classes[found->second],
                      pair.first);
            EXPECT_EQ(partition.candidate_outcomes[found->second], pair.second);
          }
          EXPECT_EQ(partition.candidate_class_count, observed.size());
          EXPECT_EQ(arena_.used_allocation_size, initialized_bytes);
          EXPECT_EQ(memcmp(committed.data(), partition.site_classes,
                           site_count * sizeof(committed[0])),
                    0);
          if (fits) {
            loom_decision_class_partition_commit(&partition);
            committed = std::move(candidate);
          }
        }
      }
    }
  }
}

TEST_F(DecisionClassPartitionTest, WideOutcomeVocabularyDoesNotSizeStorage) {
  iree_host_size_t previous_bytes = 0;
  for (uint32_t outcomes : {4098u, UINT32_MAX}) {
    iree_arena_reset(&arena_);
    loom_decision_class_partition_t partition;
    IREE_ASSERT_OK(loom_decision_class_partition_initialize(
        /*site_count=*/2, /*class_limit=*/128, outcomes, &arena_, &partition));
    // Two launch sites need two class records, regardless of how many provider
    // ordinals the library can return. Everything fits in one small arena
    // block.
    EXPECT_EQ(arena_.allocation_head, nullptr);
    if (previous_bytes) {
      EXPECT_EQ(arena_.used_allocation_size, previous_bytes);
    }
    previous_bytes = arena_.used_allocation_size;
    loom_decision_class_partition_begin(&partition, outcomes);
    ASSERT_TRUE(loom_decision_class_partition_record(&partition, 0, 0));
    ASSERT_TRUE(
        loom_decision_class_partition_record(&partition, 1, outcomes - 1));
    EXPECT_EQ(partition.candidate_class_count, 2);
    EXPECT_NE(partition.candidate_site_classes[0],
              partition.candidate_site_classes[1]);
  }
}

TEST_F(DecisionClassPartitionTest, PreservesFullParentAndOutcomeIdentities) {
  constexpr loom_decision_class_ordinal_t kClassCount =
      LOOM_DECISION_CLASS_ORDINAL_INVALID - 1;
  loom_decision_class_partition_t partition;
  IREE_ASSERT_OK(loom_decision_class_partition_initialize(
      kClassCount, kClassCount, UINT32_MAX, &arena_, &partition));
  for (uint32_t round = 0; round < 3; ++round) {
    const uint32_t outcomes = round == 1 ? 2 : UINT32_MAX;
    loom_decision_class_partition_begin(&partition, outcomes);
    for (iree_host_size_t site = 0; site < kClassCount; ++site) {
      const uint32_t outcome = round == 0   ? UINT32_MAX - 1 - site
                               : round == 1 ? site % 2
                                            : UINT32_MAX - 1;
      ASSERT_TRUE(
          loom_decision_class_partition_record(&partition, site, outcome));
      EXPECT_EQ(partition.candidate_site_classes[site], site);
    }
    loom_decision_class_partition_commit(&partition);
    EXPECT_EQ(partition.class_count, kClassCount);
  }
}

TEST_F(DecisionClassPartitionTest, RejectsSiteStorageOverflow) {
  loom_decision_class_partition_t partition;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_decision_class_partition_initialize(
                            SIZE_MAX, 128, UINT32_MAX, &arena_, &partition));
}

}  // namespace
}  // namespace loom
