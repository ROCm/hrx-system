// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/decision/class_partition.h"

#include <cstdint>
#include <cstring>

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

}  // namespace
}  // namespace loom
