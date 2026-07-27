// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/queue.h"

#include <cstdint>

#include "iree/testing/gtest.h"

namespace iree::hal {
namespace {

TEST(QueueAffinityTest, IteratesEmptyMask) {
  int iteration_count = 0;
  IREE_HAL_FOR_QUEUE_AFFINITY(0) { ++iteration_count; }
  EXPECT_EQ(iteration_count, 0);
}

TEST(QueueAffinityTest, IteratesSparseMask) {
  int expected_ordinals[] = {1, 4, 7};
  int iteration_count = 0;
  IREE_HAL_FOR_QUEUE_AFFINITY(0x92ull) {
    ASSERT_LT(iteration_count, 3);
    EXPECT_EQ(queue_count, 3);
    EXPECT_EQ(queue_index, iteration_count);
    EXPECT_EQ(queue_ordinal, expected_ordinals[iteration_count]);
    ++iteration_count;
  }
  EXPECT_EQ(iteration_count, 3);
}

TEST(QueueAffinityTest, IteratesFinalBitWithoutTrailingZeroScan) {
  int iteration_count = 0;
  IREE_HAL_FOR_QUEUE_AFFINITY(UINT64_C(1) << 63) {
    EXPECT_EQ(queue_count, 1);
    EXPECT_EQ(queue_index, 0);
    EXPECT_EQ(queue_ordinal, 63);
    ++iteration_count;
  }
  EXPECT_EQ(iteration_count, 1);
}

}  // namespace
}  // namespace iree::hal
